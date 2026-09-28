import re
import shutil
import copy
import argparse
import functools
import pathlib
from collections import defaultdict
import time
import tqdm

import config
from chemistry.grains import coarse_grains_sets
from data.forcefields import charmm22star
from data.datasets import BiomoleculeDataset, XTCDataset, Nup98Dataset, Nup98ADataset, RSDataset, ChigDDataset, \
    AAQAADataset, RichDataset, PDBEnsDataset, dataset_tempdir, supported_datasets, proc_datasets, main_datasets
from data.data_classes import Biomolecule, c_alpha_sidechain_class_name, heavy_atoms_class_name, all_atoms_class_name
from data.download import download_dataset
from analysis.metrics import analyze_dataset, calculate_gmx_energy, defined_props
from utils import set_rng

element_classes_setup = [all_atoms_class_name, heavy_atoms_class_name, c_alpha_sidechain_class_name]

# Keep a cache of loaded molecules to avoid reloading them when performing setups of splits
molecules_cache: dict[pathlib.Path, dict[str, dict[str, Biomolecule]]] = defaultdict(dict)

# XTCDataset raw directories
nup98_12_raw_dir = pathlib.Path(config.MD_sim_dir, 'nup98_12mer')
nup98C_12_raw_dir = pathlib.Path(config.MD_sim_dir, 'nup98_12mer_capped_charmm')
nup98AC_12_raw_dir = pathlib.Path(config.MD_sim_dir, 'nup98_12mer_capped_amber')
nup98_24_raw_dir = pathlib.Path(config.MD_sim_dir, 'nup98_24mer')
CLN_raw_dir = pathlib.Path(config.MD_sim_dir, 'DESRES-Trajectory_CLN025-0-protein')
AAQ_raw_dir = pathlib.Path(config.MD_sim_dir, 'AAQAA3')
RS_raw_dir = pathlib.Path(config.MD_sim_dir, 'RS')
pdb_2rvd_raw_dir = pathlib.Path(config.MD_sim_dir.parent, 'pdb', '2RVD_raw', 'all_2RVD_patched_strucs')
Rich_raw_dir = pathlib.Path(config.MD_sim_dir.parent, 'pdb', 'Richardson_top2018')

# Track which dataset has been processed at least once. Useful to perform one-time operations
processed_datasets = set()


def fetch_cached_mols(dataset_dir: str | pathlib.Path, element_class: str, mols_ID: list[str]) -> list[Biomolecule]:
    """
    Fetches molecules of a given dataset from the cache
    Args:
        dataset_dir: directory of the dataset
        element_class: element_class of the molecules to fetch
        mols_ID: IDs of the molecules to fetch

    Returns:
        list of Biomolecules with given IDs and element_class
    """
    dataset_dir = pathlib.Path(dataset_dir)

    # Initialize the dataset if it hasn't been cached yet
    if dataset_dir not in molecules_cache:
        all_atoms_dataset = XTCDataset(directory=dataset_dir, element_class=all_atoms_class_name)
        all_atoms_dataset.load_all()
        molecules_cache[dataset_dir] = {}
        molecules_cache[dataset_dir][all_atoms_class_name] = {mol.ID: mol for mol in all_atoms_dataset}

    # Initialize the element class of the given dataset by coarsening the all-atoms class
    if element_class not in molecules_cache[dataset_dir]:
        msg = f'Saving {element_class!r} molecules of {dataset_dir!r} to cache.'
        molecules_cache[dataset_dir][element_class] = {}
        for mol in tqdm.tqdm(molecules_cache[dataset_dir][all_atoms_class_name].values(), msg):
            mol_copy = copy.deepcopy(mol)
            mol_copy.coarsen(element_class=element_class)
            molecules_cache[dataset_dir][element_class][mol.ID] = mol_copy

    molecules = [molecules_cache[dataset_dir][element_class][ID] for ID in mols_ID]
    return molecules


def replace_load_all(dataset: BiomoleculeDataset):
    """
    Replaces the load_all() method and the element_class setter of the given dataset to load from cache instead.
    Args:
        dataset: BiomoleculeDataset whose load_all() method will be wrapped
    Returns:
        None
    """
    assert not dataset.is_subset, "Cannot wrapped subsets of dataset."

    load_all_func = dataset.load_all.__func__

    @functools.wraps(load_all_func)
    def load_all_wrapper(self: BiomoleculeDataset, force=False):
        # Since the wrapper does not call the original function, this is strictly-speaking not a wrapper.
        if not self.is_mols_loaded:
            parent_name = self.main_name
            self.molecules = fetch_cached_mols(self.dir, self.element_class, self.molecules_ID)
            print(f'Loaded {self.element_class!r} molecules of {self.name!r} from {parent_name!r} cache.')

    dataset.load_all = load_all_wrapper.__get__(dataset, BiomoleculeDataset)

    # Replace the element_class property of the BiomoleculeDataset class so that molecules from the cache
    # are fetched before coarsening them. This replaces a class attribute of BiomoleculeDataset so all future instances
    # will have the wrapped property.
    element_class_setter_ori = BiomoleculeDataset.element_class.fset

    def element_class_setter_wrapper(self: BiomoleculeDataset, element_class: str, *args, **kwargs):
        parent_dataset_dir = self.dir
        if self.is_mols_loaded and parent_dataset_dir in molecules_cache and self.element_class != element_class:
            print(f'Changing element_class of {self.name!r} cached molecules to {element_class!r}.')
            self.molecules = fetch_cached_mols(parent_dataset_dir, element_class, self.molecules_ID)

        # Call the original method
        element_class_setter_ori(self, element_class, *args, **kwargs)

    getter = BiomoleculeDataset.element_class.fget
    deleter = BiomoleculeDataset.element_class.fdel
    new_element_class_prop = property(fget=getter, fset=element_class_setter_wrapper, fdel=deleter,
                                      doc=BiomoleculeDataset.element_class.__doc__)
    BiomoleculeDataset.element_class = new_element_class_prop


def element_class_iterator(func):
    @functools.wraps(func)
    def wrapper(*args, element_class: str | list[str] = None, **kwargs):
        # If many element classes are provided, loop through each of them to calculate statistics for all possible
        # classes
        if element_class is None:
            element_class = element_classes_setup
        if isinstance(element_class, str):
            element_class = [element_class]

        for elem_class in element_class:
            func(*args, element_class=elem_class, **kwargs)

    return wrapper


@element_class_iterator
def calculate_stats(datasets: list[BiomoleculeDataset], recalc_stats: list[str] = None,
                    element_class: str = all_atoms_class_name):
    # Set the element class of all analyzed datasets
    for d in datasets:
        d.element_class = element_class

    # Delete statistics if regeneration is requested
    for d in datasets:
        if args.delete_expired_stats:
            d.delete_expired_statistics()

    # Calculate statistics of the dataset and its splits
    stats_name = defined_props[element_class] - {'rmsd', 'elements_dist_CA_SI'}

    for dataset in datasets:
        print(f"Starting statistics calculation of {dataset.name!r} with element_class={element_class!r}")
        analyze_dataset(dataset, stats=stats_name, recalc_stats=recalc_stats)


def preparation_wrapper(func):
    @functools.wraps(func)
    def wrapper(dataset_name, *func_args, split_method='random_run', n_splits_set=1, split_sets_ID: list[str] = None,
                splits_ratio: list | None = None, raw_dir: str | pathlib.Path = '', **func_kwargs):
        if dataset_name in datasets_name:
            # Check if the raw directory exists. If not, attempt download from repository
            raw_dir = pathlib.Path(raw_dir)
            if not raw_dir.exists() and raw_dir.parent == config.MD_sim_dir:
                download_dataset(raw_dir.name, 'MD')

            # Remove directory of previous dataset if it will be regenerated
            if 'directory' in func_kwargs:
                dataset_dir = func_kwargs['directory']
            else:
                dataset_dir = pathlib.Path(config.proc_datasets_dir, dataset_name)
            if args.regen and dataset_dir.exists():
                shutil.rmtree(dataset_dir)

            # Initialize the dataset
            dataset: BiomoleculeDataset = func(dataset_name, *func_args, **func_kwargs)

            # Resave molecules and metadata
            if args.resave and dataset_name not in processed_datasets:
                dataset.resave_molecules()
                dataset.save_metadata()

            # Delete splits (if requested)
            for split_set_ID in args.delete_splits_sets:
                if split_set_ID in dataset.splits_sets:
                    dataset.delete_splits_set(split_set_ID)

            # Skip statistics calculation for datasets that are not XTCDatasets.
            if not isinstance(dataset, XTCDataset):
                return

            # Wrap the dataset load_all method to speed up loading of molecules. This also wraps the subsets' load_all.
            replace_load_all(dataset)

            # Gather the various splits of the dataset to calculate their statistics
            dataset_splits = []
            if split_sets_ID is None:
                split_sets_ID = [str(i + 1) for i in range(n_splits_set)]
            if splits_ratio is not None:
                for split_set_ID in split_sets_ID:
                    # Create splits
                    if split_set_ID not in dataset.splits_sets:
                        dataset.create_splits(splits_ratio, method=split_method, set_ID=split_set_ID)
                    dataset_splits += dataset.load_splits(set_ID=split_set_ID)

            # Recalculate GROMACS energies if any energies are requested for recalculation.
            if args.recalc_gmx_en and dataset_name not in processed_datasets:
                calculate_gmx_energy(dataset, regen=True)

            analyzed_datasets = dataset_splits + [dataset]
            calculate_stats(datasets=analyzed_datasets, recalc_stats=args.recalc_stats)

            # Add dataset to set of processed datasets
            processed_datasets.add(dataset_name)
            print(f'Setup of {dataset_name!r} is complete.')

    return wrapper


############################################ PDB Datasets ############################################
@preparation_wrapper
def prepare_Rich(name, raw_dir=Rich_raw_dir, **kwargs):
    # Reference: "High quality protein residues: Top2018 mainchain-filtered residues"
    #            https://zenodo.org/records/5777651
    # Richardson setup steps
    # #1: wget "https://zenodo.org/api/records/5777651/files-archive"
    # #2 untar all .tar.gz files in raw directory
    if not raw_dir.exists():
        raise FileNotFoundError("Raw directory of Top2018 dataset was not found.\n"
                                "1) Download dataset from 'https://zenodo.org/records/5777651'\n"
                                f"2) untar all .tar.gz files and move into {str(raw_dir)}.")
    rich_dataset = RichDataset(name=name, raw_dir=raw_dir, **kwargs)

    return rich_dataset


@preparation_wrapper
def prepare_2RVD(name='PDB_2RVD', raw_dir=pdb_2rvd_raw_dir, forcefield=charmm22star, terminals_gmx_code=None, **kwargs):
    set_rng(2028)
    if terminals_gmx_code is None:
        terminals_gmx_code = ['0', '0']
    return PDBEnsDataset(name=name, raw_dir=raw_dir, forcefield=forcefield, terminals_gmx_code=terminals_gmx_code,
                         **kwargs)


############################################ Nup98 Datasets ############################################
@preparation_wrapper
def prepare_Nup98(name, sampling_dt=1000, **kwargs):
    set_rng(2027)
    return Nup98Dataset(name=name, sampling_dt=sampling_dt, **kwargs)


@preparation_wrapper
def prepare_Nup98A(name, sampling_dt=1000, **kwargs):
    set_rng(2030)
    return Nup98ADataset(name=name, sampling_dt=sampling_dt, **kwargs)


############################################ RS Datasets ############################################


@preparation_wrapper
def prepare_RS(name, sampling_dt=1000, raw_dir=RS_raw_dir, top_pattern='.*RS.top', **kwargs):
    set_rng(2028)
    return RSDataset(name=name, sampling_dt=sampling_dt, raw_dir=raw_dir, top_pattern=top_pattern, **kwargs)


############################################ Chignolin Datasets ############################################
@preparation_wrapper
def prepare_ChigD(name, sampling_dt=1000, raw_dir=CLN_raw_dir, **kwargs):
    set_rng(2028)
    return ChigDDataset(name=name, sampling_dt=sampling_dt, raw_dir=raw_dir, **kwargs)


############################################ AAQAA_3 Datasets ############################################
@preparation_wrapper
def prepare_AAQAA(name, sampling_dt=1000, raw_dir=AAQ_raw_dir, **kwargs):
    set_rng(2029)
    return AAQAADataset(name=name, sampling_dt=sampling_dt, raw_dir=raw_dir, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', default='all', help='Filename of the dataset to setup')
    parser.add_argument('--regen', action='store_true', help='Regenerate dataset')
    parser.add_argument('--delete_expired_stats', action='store_true', help='Delete expired statistics files')
    parser.add_argument('--recalc_stats', default=[], nargs='*', help='Name of statistics to recalculate.')
    parser.add_argument('--delete_splits_sets', default=[], nargs='*', help='Name of split sets that will re-created.')
    parser.add_argument('--recalc_gmx_en', action='store_true', help='Recalculate GROMACS energies')
    parser.add_argument('--resave', action='store_true', help='Re-save dataset molecules and metadata')
    args = parser.parse_args()

    # Determine which dataset will be generated
    datasets_name = set()
    for name in args.datasets:
        if name == 'all':
            datasets_name |= set(proc_datasets)
        elif name == 'main':
            datasets_name |= set(main_datasets.values())
        elif name in supported_datasets:
            datasets_name |= {name}
        else:
            # Search through all supported datasets omitting deprecated ones
            datasets_name |= set(d for d in set(supported_datasets) if re.fullmatch(name, d))

    unsupported_datasets_name = datasets_name - set(supported_datasets)
    if unsupported_datasets_name:
        raise ValueError(f"dataset={unsupported_datasets_name} is not supported. "
                         f"Supported datasets:\n{supported_datasets}")
    if not datasets_name:
        print(f"No datasets to setup. Exiting.")
        exit(0)

    print(f'Starting setup for the following datasets:\n{datasets_name}')
    time.sleep(1)

    # Richardson 2018 PDB Datasets
    prepare_Rich('Rich_2018_debug', debug=True)
    prepare_Rich('Rich_2018')

    # Calculate the coarse grains geometries using the 'Rich_2018' dataset. 'Rich_2018' must be configured.
    rich_sidechain_grains = coarse_grains_sets[('Rich_2018', 'sidechain_heavy')]

    # Setup 2RVD dataset. Needed to define fold state of CLN025 dataset
    prepare_2RVD('PDB_2RVD')

    # Debugging datasets
    prepare_Nup98('nup98_12_debug', n_runs=6, splits_ratio=[4, 1, 1], sampling_dt=40000, n_splits_set=2,
                  xtc_pattern=r"prod(_)?(?P<ID>\d+)(_full)?_prot.xtc", raw_dir=nup98_12_raw_dir)
    prepare_Nup98('nup98_12_debug2', splits_ratio=[6, 1, 5], n_splits_set=1,
                  xtc_pattern=r"prod(_)?(?P<ID>\d+)_full_prot.xtc", raw_dir=nup98_12_raw_dir)
    test_dataset_dir2 = dataset_tempdir / 'nup98_12_debug3'
    prepare_Nup98(test_dataset_dir2.name, directory=test_dataset_dir2, n_runs=6, splits_ratio=[4, 1, 1],
                  sampling_dt=40000, n_splits_set=2, xtc_pattern=r"prod(_)?(?P<ID>\d+)_full_prot.xtc",
                  raw_dir=nup98_12_raw_dir)

    # Main datasets
    prepare_Nup98('nup98_12', n_splits_set=1, splits_ratio=[6, 1, 5], raw_dir=nup98_12_raw_dir,
                  xtc_pattern=r"prod(_)?(?P<ID>\d+)_full_prot.xtc")
    prepare_Nup98('nup98_24', n_splits_set=1, splits_ratio=[6, 1, 5], raw_dir=nup98_24_raw_dir)
    prepare_AAQAA('AAQAA3', n_splits_set=1, splits_ratio=[6, 1, 5])
    prepare_RS('RS', n_splits_set=1, splits_ratio=[6, 1, 5])
    prepare_ChigD('CLN025', n_splits_set=1, splits_ratio=[6, 1, 6])

    # Datasets that test the effect of the force field
    prepare_Nup98('nup98C_12', n_splits_set=1, splits_ratio=[6, 1, 5],
                  xtc_pattern=r"prod(?P<ID>\d+)_prot.xtc", sampling_dt=500, terminals_gmx_code=['3', '4'],
                  raw_dir=nup98C_12_raw_dir)
    prepare_Nup98A('nup98AC_12', n_splits_set=1, splits_ratio=[6, 1, 5],
                   xtc_pattern=r"prod(?P<ID>\d+)_prot.xtc", sampling_dt=500, terminals_gmx_code=['3', '4'],
                   raw_dir=nup98AC_12_raw_dir)

    # Datasets with very small train splits
    prepare_Nup98('nup98_12', split_sets_ID=['S1'], splits_ratio=[1, 1, 10])
    prepare_Nup98('nup98_24', split_sets_ID=['S1'], splits_ratio=[1, 1, 10])
    prepare_AAQAA('AAQAA3', split_sets_ID=['S1'], splits_ratio=[1, 1, 10])
    prepare_RS('RS', split_sets_ID=['S1'], splits_ratio=[1, 1, 10])
    prepare_ChigD('CLN025', split_sets_ID=['S1'], splits_ratio=[1, 1, 11])

    # Chignolin datasets split on the fold state
    prepare_ChigD('CLN025', split_sets_ID=['FinTr2_5'], split_method='FoldedHA_2.5InTrain',
                  splits_ratio=[6, 1, 6])
    prepare_ChigD('CLN025', split_sets_ID=['UinTr2_5'], split_method='UnfoldedHA_2.5InTrain',
                  splits_ratio=[6, 1, 6])
    prepare_ChigD('CLN025', split_sets_ID=['FinTr3'], split_method='FoldedHA_3InTrain',
                  splits_ratio=[6, 1, 6])
    prepare_ChigD('CLN025', split_sets_ID=['UinTr3'], split_method='UnfoldedHA_3InTrain',
                  splits_ratio=[6, 1, 6])
    prepare_ChigD('CLN025', split_sets_ID=['UinTr4'], split_method='UnfoldedHA_4InTrain',
                  splits_ratio=[6, 1, 6])
    prepare_ChigD('CLN025', split_sets_ID=['FinTr4'], split_method='FoldedHA_4InTrain',
                  splits_ratio=[6, 1, 6])
    prepare_ChigD('CLN025', split_sets_ID=['UinTr5'], split_method='UnfoldedHA_5InTrain',
                  splits_ratio=[6, 1, 6])
    prepare_ChigD('CLN025', split_sets_ID=['FinTr5'], split_method='FoldedHA_5InTrain',
                  splits_ratio=[6, 1, 6])
    prepare_ChigD('CLN025', split_sets_ID=['UinTr6'], split_method='UnfoldedHA_6InTrain',
                  splits_ratio=[6, 1, 6])
    prepare_ChigD('CLN025', split_sets_ID=['FinTr6'], split_method='FoldedHA_6InTrain',
                  splits_ratio=[6, 1, 6])
