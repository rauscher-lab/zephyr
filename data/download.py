import hashlib
import io
import os
import tempfile
import threading
import warnings
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
import shutil
import time

import requests
from tqdm import tqdm

import config

ZENODO_RECORD_ID = '23177444'
ZENODO_URL = "https://zenodo.org"

# Optional share token, only needed while the record is a draft.
token_filepath = Path(config.data_dir, 'zenodo_token.txt').resolve()
ZENODO_TOKEN = token_filepath.read_text().strip() if token_filepath.exists() else None
ZENODO_PARAMS = {"token": ZENODO_TOKEN} if ZENODO_TOKEN else {}
IS_DRAFT = ZENODO_TOKEN is not None  # a draft is only reachable with a share token

# Optional client identification, in the format Zenodo asks for: "AppName/1.0 (+url; contact@email)". Unset by default.
USER_AGENT = os.environ.get("ZENODO_USER_AGENT")

# Partly downloaded datasets are staged here, outside the destination tree, and only moved into
# place once complete. Override with the ZENODO_STAGING_DIR environment variable - worth doing if
# the system temporary directory is a RAM disk (tmpfs) or sits on a smaller/slower filesystem than
# the destination, since a staged dataset is as large as the dataset itself.
PROJECT_NAME = getattr(config, "project_name", "zenodo")
STAGING_ROOT = Path(os.environ.get("ZENODO_STAGING_DIR",
                                   Path(tempfile.gettempdir()) / f"{PROJECT_NAME}_download_staging"))

DEFAULT_CHUNK_MB = 32  # MB fetched per Range request when reading the zip index
DEFAULT_WORKERS = 4  # files fetched in parallel: hides per-request latency (1 = sequential)
STREAM_PIECE = 256 * 1024  # bytes read per socket read; also the progress-bar update granularity
MAX_ATTEMPTS = 5
RETRY_STATUS = (408, 425, 429, 500, 502, 503, 504)
TIMEOUT = (30, 120)  # (connect, read) timeout in seconds

SESSION = requests.Session()
if USER_AGENT:
    SESSION.headers.update({"User-Agent": USER_AGENT})

_url_cache: dict[tuple[str, str], str] = {}
_advice_given = False


def set_user_agent(user_agent: str):
    """
    Identifies this client to Zenodo, which reduces the chance of being throttled.
    Use the format "AppName/1.0 (+url; contact@email)", e.g.
        set_user_agent("MyProject/1.0 (+https://example.org/myproject; me@example.org)")
    """
    global USER_AGENT
    USER_AGENT = user_agent
    SESSION.headers.update({"User-Agent": user_agent})


def _advise_user_agent():
    """Printed once, when Zenodo throttles or refuses requests from an unidentified client."""
    global _advice_given
    if _advice_given:
        return
    _advice_given = True
    if USER_AGENT:
        print("  Zenodo is throttling this client despite a custom User-Agent. Wait a few minutes "
              "and retry; see https://blog.zenodo.org for current limits.")
    else:
        print("  Zenodo throttles unidentified clients. If this keeps happening, identify yours with\n"
              '    export ZENODO_USER_AGENT="MyProject/1.0 (+https://example.org/project; me@example.org)"\n'
              "  or call download.set_user_agent(...) before downloading.")


def _sleep_time(response: requests.Response | None, attempt: int) -> float:
    """Honours Retry-After when Zenodo sends it, otherwise exponential backoff with jitter."""
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), 300.0)
            except ValueError:
                pass
    return min(2 ** attempt, 60) + random.uniform(0, 2)


def _request(method: str, url: str, max_attempts: int = MAX_ATTEMPTS, **kwargs) -> requests.Response:
    """GET/HEAD with retries on rate limits, server errors and dropped connections."""
    kwargs.setdefault("timeout", TIMEOUT)
    kwargs.setdefault("params", ZENODO_PARAMS)
    response = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = SESSION.request(method, url, **kwargs)
        except (requests.ConnectionError, requests.Timeout) as error:
            if attempt == max_attempts:
                raise
            wait = _sleep_time(None, attempt)
            print(f"  {type(error).__name__}; retrying in {wait:.0f}s ({attempt}/{max_attempts})")
            time.sleep(wait)
            continue

        if response.status_code in (429, 403):
            _advise_user_agent()
        if response.status_code in RETRY_STATUS and attempt < max_attempts:
            wait = _sleep_time(response, attempt)
            reason = "rate-limited by Zenodo" if response.status_code == 429 else f"HTTP {response.status_code}"
            print(f"  {reason}; retrying in {wait:.0f}s ({attempt}/{max_attempts})")
            time.sleep(wait)
            continue
        return response
    return response


class BufferedRemoteZipStream(io.RawIOBase):
    """
    A single-threaded seekable stream caching large HTTP Range blocks in RAM
    to eliminate network latency from zipfile micro-reads.
    """

    def __init__(self, url: str, headers: dict = None, chunk_size: int = DEFAULT_CHUNK_MB):
        self.url = url
        self.headers = headers or {}
        self.chunk_size = chunk_size * 1024 * 1024
        self._pos = 0

        self._buffer = b""
        self._buffer_offset = 0

        response = _request("HEAD", self.url, headers=self.headers, allow_redirects=True)
        if response.status_code in (403, 405) or not response.headers.get("Content-Length"):
            response = _request("GET", self.url, headers=self.headers, stream=True)
            response.close()
        response.raise_for_status()

        content_length = response.headers.get("Content-Length")
        if not content_length:
            raise ValueError("Server did not return Content-Length required for Range requests.")
        self._length = int(content_length)
        if response.headers.get("Accept-Ranges", "").lower() == "none":
            raise ValueError("Server does not accept Range requests; partial download is impossible.")

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._length + offset

        self._pos = max(0, min(self._pos, self._length))
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size == 0 or self._pos >= self._length:
            return b""

        if self._buffer and self._buffer_offset <= self._pos < self._buffer_offset + len(self._buffer):
            offset_in_buf = self._pos - self._buffer_offset
            if size > 0 and offset_in_buf + size <= len(self._buffer):
                data = self._buffer[offset_in_buf:offset_in_buf + size]
                self._pos += len(data)
                return data

        fetch_size = max(size if size > 0 else self.chunk_size, self.chunk_size)
        end = min(self._pos + fetch_size - 1, self._length - 1)

        headers = {**self.headers, "Range": f"bytes={self._pos}-{end}"}
        response = _request("GET", self.url, headers=headers)
        response.raise_for_status()

        content = response.content
        if response.status_code != 206:
            # Range ignored: the whole file was returned, so slice out the requested window
            content = content[self._pos:end + 1]

        self._buffer = content
        self._buffer_offset = self._pos

        data = self._buffer[:size] if size > 0 else self._buffer
        self._pos += len(data)
        return data


def get_zenodo_file_url(record_id: str, target_filename: str, is_draft: bool = IS_DRAFT) -> str:
    """
    Fetches download URL for a file on a Zenodo repository
    Args:
        record_id: record ID of the Zenodo repository
        target_filename: target filename in the repository
        is_draft: set to True if repository is a draft (requires a share token)

    Returns:
        download url
    """
    cache_key = (record_id, target_filename)
    if cache_key in _url_cache:
        return _url_cache[cache_key]

    api_url = f"{ZENODO_URL}/api/records/{record_id}"
    if is_draft:
        api_url += "/draft"

    response = _request("GET", api_url)
    if response.status_code in (403, 404) and is_draft:
        response = _request("GET", f"{ZENODO_URL}/api/records/{record_id}")  # published version
    if response.status_code in (401, 403):
        raise PermissionError(
            f"Zenodo denied access to record {record_id!r} ({response.status_code}). If the record is "
            f"not public yet, save the token of a share link in {token_filepath}."
        )
    response.raise_for_status()

    data = response.json()
    files_data = data.get("files", {})
    files_list = files_data.get("entries", files_data) if isinstance(files_data, dict) else files_data
    if isinstance(files_list, dict):
        files_list = list(files_list.values())

    available = []
    for file_info in files_list:
        filename = file_info.get("key") or file_info.get("filename")
        available.append(filename)
        if filename == target_filename:
            links = file_info.get("links", {})
            download_url = links.get("content") or links.get("self") or links.get("download")
            if download_url:
                _url_cache[cache_key] = download_url
                return download_url

    raise FileNotFoundError(f"File {target_filename!r} not found in record/draft {record_id!r}. "
                            f"Available files: {sorted(f for f in available if f)}")


def get_zenodo_zip_content(record_id: str, zip_filename: str) -> list[Path]:
    """Lists the files and directories contained in a .zip file stored on Zenodo."""
    url = get_zenodo_file_url(record_id, zip_filename)
    stream = BufferedRemoteZipStream(url)
    with zipfile.ZipFile(stream) as zf:
        names = zf.namelist()

    # Include directories that are implied by file paths but have no entry of their own
    paths = set()
    for name in names:
        path = Path(name)
        paths.add(path)
        paths.update(path.parents)
    paths.discard(Path('.'))
    return sorted(paths)


def extract_file(zf: zipfile.ZipFile, member: zipfile.ZipInfo, staging_dir: Path,
                 chunk_size: int = 1024 * 1024, on_bytes=None):
    """
    Extracts a member through the shared zipfile stream (fallback path).
    Not thread-safe: the caller must hold a lock on zf.
    """
    staged_file_path = staging_dir / member.filename
    staged_file_path.parent.mkdir(parents=True, exist_ok=True)

    existing_size = staged_file_path.stat().st_size if staged_file_path.exists() else 0
    if member.file_size == 0:
        staged_file_path.touch()  # an empty member still has to produce an empty file
        return
    if existing_size >= member.file_size:
        return

    mode = "ab" if existing_size > 0 else "wb"
    with zf.open(member) as src_file, staged_file_path.open(mode) as dst_file:
        if existing_size > 0:
            src_file.seek(existing_size)
        while True:
            chunk = src_file.read(chunk_size)
            if not chunk:
                break
            dst_file.write(chunk)
            if on_bytes:
                on_bytes(len(chunk))


class UnsupportedMember(Exception):
    """Raised when a zip member cannot be fetched with a plain Range request."""


MIN_GROUP_MB = 8  # smallest span covered by one request when splitting work across workers
MAX_GAP_MB = 1  # unselected bytes worth streaming past rather than paying a new round trip


def is_streamable(member: zipfile.ZipInfo) -> bool:
    """True if a member can be fetched with a plain Range request and decompressed with zlib."""
    return (not member.flag_bits & 0x1
            and member.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED))


class _RangeReader:
    """Sequential reader over a streamed Range response, reporting bytes to a progress callback."""

    def __init__(self, response, start: int, on_bytes=None):
        self._pieces = response.iter_content(STREAM_PIECE)
        self._buffer = b""
        self._on_bytes = on_bytes
        self.pos = start

    def _pull(self) -> bool:
        for piece in self._pieces:
            if piece:
                self._buffer += piece
                if self._on_bytes:
                    self._on_bytes(len(piece))
                return True
        return False

    def read_upto(self, size: int) -> bytes:
        while not self._buffer:
            if not self._pull():
                return b""
        data, self._buffer = self._buffer[:size], self._buffer[size:]
        self.pos += len(data)
        return data

    def read_exact(self, size: int) -> bytes:
        parts, remaining = [], size
        while remaining > 0:
            piece = self.read_upto(remaining)
            if not piece:
                raise IOError(f"stream ended {remaining} bytes early")
            parts.append(piece)
            remaining -= len(piece)
        return b"".join(parts)

    def skip_to(self, offset: int):
        while self.pos < offset:
            if not self.read_upto(offset - self.pos):
                raise IOError("stream ended while skipping")


def group_members(members: list[zipfile.ZipInfo], max_workers: int = DEFAULT_WORKERS
                  ) -> list[list[zipfile.ZipInfo]]:
    """
    Splits members into runs that each become ONE streamed Range request.

    Members of a directory sit next to each other in the archive, so a single sequential request
    fetches many files at once - far cheaper than one request per file. The runs are sized so that
    there are roughly max_workers of them, which keeps that efficiency while still using several
    connections in parallel. A run is also cut whenever skipping unselected bytes would cost more
    than starting a new request.
    """
    ordered = sorted(members, key=lambda m: m.header_offset)
    if not ordered:
        return []

    span = sum(m.compress_size for m in ordered)
    target = max(MIN_GROUP_MB * 1024 ** 2, span // max(1, max_workers))
    groups, current, current_span, previous_end = [], [], 0, None

    for member in ordered:
        gap = member.header_offset - previous_end if previous_end is not None else 0
        if current and (current_span >= target or gap > MAX_GAP_MB * 1024 ** 2):
            groups.append(current)
            current, current_span = [], 0
        current.append(member)
        current_span += member.compress_size
        previous_end = member.header_offset + member.compress_size
    if current:
        groups.append(current)
    return groups


def download_members(url: str, members: list[zipfile.ZipInfo], staging_dir: Path, on_bytes=None,
                     extra_field_slack: int = 256):
    """
    Downloads a run of zip members with a single streamed Range request, decompressing each one on
    the fly. Data is written as it arrives, so progress is continuous rather than per file.
    Raises UnsupportedMember for encrypted or unusual members.
    """
    unsupported = [m.filename for m in members if not is_streamable(m)]
    if unsupported:
        raise UnsupportedMember(f"encrypted or unusual members: {unsupported}")

    start = members[0].header_offset
    last = members[-1]
    end = (last.header_offset + 30 + len(last.filename.encode("utf-8")) + extra_field_slack
           + last.compress_size - 1)

    response = _request("GET", url, headers={"Range": f"bytes={start}-{end}"}, stream=True)
    response.raise_for_status()
    if response.status_code != 206:
        response.close()
        raise UnsupportedMember("server ignored the Range request")

    with response:
        reader = _RangeReader(response, start, on_bytes)
        for member in members:
            reader.skip_to(member.header_offset)
            header = reader.read_exact(30)
            if header[:4] != b"PK\x03\x04":
                raise UnsupportedMember(f"{member.filename!r} has no local file header")
            reader.read_exact(int.from_bytes(header[26:28], "little")  # file name
                              + int.from_bytes(header[28:30], "little"))  # extra field
            _write_member(reader, member, staging_dir)


def _write_member(reader: _RangeReader, member: zipfile.ZipInfo, staging_dir: Path):
    """Decompresses one member out of the stream into staging_dir, verifying size and CRC."""
    destination = staging_dir / member.filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(destination.name + ".part")

    decompressor = zlib.decompressobj(-15) if member.compress_type == zipfile.ZIP_DEFLATED else None
    remaining, crc, written = member.compress_size, 0, 0
    try:
        with part_path.open("wb") as output:
            while remaining > 0:
                piece = reader.read_upto(min(STREAM_PIECE, remaining))
                if not piece:
                    raise IOError(f"{member.filename!r}: stream ended early")
                remaining -= len(piece)
                chunk = decompressor.decompress(piece) if decompressor else piece
                if chunk:
                    crc = zlib.crc32(chunk, crc)
                    written += len(chunk)
                    output.write(chunk)
            if decompressor:
                tail = decompressor.flush()
                if tail:
                    crc = zlib.crc32(tail, crc)
                    written += len(tail)
                    output.write(tail)

        if written != member.file_size:
            raise IOError(f"{member.filename!r}: got {written} bytes, expected {member.file_size}")
        if member.CRC and crc != member.CRC:
            raise IOError(f"{member.filename!r}: CRC mismatch, the download is corrupt")
        part_path.replace(destination)
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise


def download_zenodo_content(remote_filepath: str | Path, local_filepath: str | Path,
                            record_id: str = ZENODO_RECORD_ID, zip_filename: str = "datasets.zip",
                            is_draft: bool = IS_DRAFT, max_workers: int = DEFAULT_WORKERS):
    """
    Downloads files from a .zip file stored in a Zenodo repository
    Args:
        remote_filepath: filepath or directory in the remote .zip file.
        local_filepath: local filepath or directory where the content of the downloaded files will be saved
        record_id: record ID of the Zenodo repository
        zip_filename: filename of the .zip file
        is_draft: True if downloading from a draft repository (requires a share token)
        max_workers: number of files fetched in parallel (1 = sequential). Parallelism hides the
            round-trip latency of each request, which dominates when files are small.

    Returns:
        None
    """
    url = get_zenodo_file_url(record_id, zip_filename, is_draft=is_draft)
    remote_filepath = Path(remote_filepath)
    local_filepath = Path(local_filepath).expanduser().resolve()
    if remote_filepath.suffix:
        remote_dir = remote_filepath.parent
        local_dir = local_filepath.parent
    else:
        remote_dir = remote_filepath
        local_dir = local_filepath

    # Downloads land in a staging directory under STAGING_ROOT; neither it nor the final directory
    # is created before there is something to download, and the final one only once every file is
    # complete, so an interrupted download never leaves a half-populated dataset directory behind.
    # The destination path is hashed into the name so that two datasets sharing a name, or the same
    # dataset downloaded to two places, never share a staging directory.
    staging_dir = staging_directory(local_dir)

    stream = BufferedRemoteZipStream(url)
    with zipfile.ZipFile(stream) as zf:
        remote_files = [f for f in zf.infolist()
                        if not f.is_dir() and _matches(Path(f.filename), remote_filepath)]
        if not remote_files:
            print(f"No files found in {str(remote_filepath)!r}")
            return

        # Define local filepath of each remote file. A single requested file keeps the name given in
        # local_filepath; files taken from a directory keep their name inside that directory.
        def local_path_of(member_name: Path) -> Path:
            if remote_filepath.suffix and member_name == remote_filepath:
                return local_filepath
            return local_dir / member_name.relative_to(remote_dir)

        def is_complete(path: Path, member: zipfile.ZipInfo) -> bool:
            return path.exists() and path.stat().st_size == member.file_size

        pairs = [(member, local_path_of(Path(member.filename))) for member in remote_files]
        # A file already staged by an interrupted run is kept: it only needs to be moved into place.
        staged_ready = [member for member, _ in pairs
                        if is_complete(staging_dir / member.filename, member)]
        todo = [(member, local_file) for member, local_file in pairs
                if not is_complete(local_file, member)
                and not is_complete(staging_dir / member.filename, member)]

        total_bytes = sum(m.file_size for m in remote_files)
        print(f"Found {len(remote_files)} file(s) ({total_bytes / (1024 ** 2):.2f} MB total) in"
              f" {remote_dir} inside Zenodo's {zip_filename!r}.")
        if not todo and not staged_ready:
            print(f"All files already present in {local_dir}")
            shutil.rmtree(staging_dir, ignore_errors=True)
            return
        if staged_ready:
            print(f"Resuming: {len(staged_ready)} file(s) already downloaded to {staging_dir}")
        print(f"Saving to {local_dir}")
        staging_dir.mkdir(parents=True, exist_ok=True)

        # The bar measures bytes coming off the network (i.e. compressed bytes), so it advances
        # continuously while a file is being fetched instead of jumping once per completed file.
        transfer_total = sum(m.compress_size for m, _ in todo)
        pbar = tqdm(total=transfer_total, desc=f"Downloading {remote_filepath} from Zenodo", unit="B",
                    unit_scale=True, unit_divisor=1024, smoothing=0.1)
        progress_lock = threading.Lock()
        zf_lock = threading.Lock()
        transferred = 0

        def on_bytes(count: int):
            nonlocal transferred
            with progress_lock:
                allowed = max(0, min(count, transfer_total - transferred))
                transferred += allowed
            if allowed:
                pbar.update(allowed)

        def fetch(group: list[zipfile.ZipInfo]):
            try:
                download_members(url, group, staging_dir, on_bytes=on_bytes)
            except UnsupportedMember as error:
                warnings.warn(f"Falling back to block reads: {error}")
                for member in group:
                    with zf_lock:
                        extract_file(zf, member, staging_dir, on_bytes=on_bytes)

        wanted = [member for member, _ in todo]
        # Encrypted or exotically compressed members (rare) go through zipfile instead of being
        # streamed, and are kept out of the groups so they do not slow the others down.
        groups = group_members([m for m in wanted if is_streamable(m)], max_workers=max_workers)
        odd_members = [m for m in wanted if not is_streamable(m)]
        started = time.monotonic()
        with pbar:
            if max_workers > 1 and len(groups) > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    for future in [pool.submit(fetch, group) for group in groups]:
                        future.result()  # re-raise the first failure
            else:
                for group in groups:
                    fetch(group)
            for member in odd_members:
                warnings.warn(f"{member.filename!r} cannot be streamed; using block reads")
                with zf_lock:
                    extract_file(zf, member, staging_dir, on_bytes=on_bytes)
        elapsed = time.monotonic() - started

        # Every file is complete: create the final directory and move the staged files into it
        local_dir.mkdir(parents=True, exist_ok=True)
        for member, local_file in pairs:
            staged_filepath = staging_dir / member.filename
            if staged_filepath.exists() and staged_filepath.stat().st_size == member.file_size:
                local_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(staged_filepath, local_file)

        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        if STAGING_ROOT.exists() and not any(STAGING_ROOT.iterdir()):
            STAGING_ROOT.rmdir()

    if elapsed > 0:
        print(f"Download complete: {transferred / 1024 ** 2:.1f} MB transferred in {elapsed:.0f}s "
              f"({transferred / 1024 ** 2 / elapsed:.2f} MB/s).")
    else:
        print("Download complete.")


def staging_directory(local_dir: Path) -> Path:
    """Returns the staging directory holding the partial download destined for local_dir."""
    digest = hashlib.sha1(str(local_dir).encode()).hexdigest()[:8]
    return STAGING_ROOT / f"{local_dir.name}-{digest}"


def clear_staging(older_than_days: float = None) -> int:
    """
    Deletes staging directories left behind by interrupted downloads.
    Args:
        older_than_days: only delete directories untouched for this long (None deletes all)

    Returns:
        number of directories removed
    """
    if not STAGING_ROOT.exists():
        return 0
    removed = 0
    for directory in STAGING_ROOT.iterdir():
        if not directory.is_dir():
            continue
        if older_than_days is not None:
            age_days = (time.time() - directory.stat().st_mtime) / 86400
            if age_days < older_than_days:
                continue
        shutil.rmtree(directory, ignore_errors=True)
        removed += 1
    return removed


def _matches(member: Path, target: Path) -> bool:
    """True if a zip member is the target file itself or lies inside the target directory."""
    return member == target or target in member.parents


def download_dataset(name: str, type: str):
    supported_types = ['MD', 'processed', 'generated']
    assert type in supported_types, f"type must be one of the following options:{supported_types}."

    # Find dataset path in zenodo repository
    dataset_zip_filename = f'{type}_datasets.zip'
    zenodo_files = get_zenodo_zip_content(ZENODO_RECORD_ID, zip_filename=dataset_zip_filename)
    zenodo_dirs = [f for f in zenodo_files if f.name == name]
    if len(zenodo_dirs) < 1:
        warnings.warn(f"No Zenodo directory found for {type} dataset {name!r}.")
        return
    elif len(zenodo_dirs) > 1:
        raise ValueError(f"{len(zenodo_dirs)} Zenodo directories found for dataset {name!r}.\n{zenodo_dirs}")
    dataset_zenodo_dir = zenodo_dirs[0]

    # Download dataset to local directory
    dataset_parent_dirs = {'processed': config.proc_datasets_dir, 'MD': config.MD_sim_dir,
                           'generated': config.gen_datasets_dir}
    dataset_local_dir = dataset_parent_dirs[type] / name
    download_zenodo_content(remote_filepath=dataset_zenodo_dir, local_filepath=dataset_local_dir,
                            zip_filename=dataset_zip_filename)


def download_model_weights(weight_filepath: str | Path):
    weight_filepath = Path(weight_filepath)
    if not weight_filepath.is_absolute():
        weight_filepath = config.weights_dir / weight_filepath
    remote_filepath = Path('model_weights') / weight_filepath.name
    download_zenodo_content(remote_filepath=remote_filepath, local_filepath=weight_filepath,
                            zip_filename='model_weights.zip')


if __name__ == '__main__':
    # Test download of a processed dataset
    # download_dataset('nup98_12_1', 'processed')

    # Test download of an MD dataset
    download_dataset('RS', 'MD')

    # Test download of a generated dataset
    # download_dataset('Diff_CS1_HA1_nup98_12_1_C1H1', 'generated')
