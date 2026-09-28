import datetime
import os
import pathlib
import random
import re
import shutil
import tempfile
import copy
import hashlib
import typing
import warnings
import uuid
import functools
import multiprocessing

from pathlib import Path
from typing import Self, TYPE_CHECKING

import MDAnalysis as mda

import numpy as np
import torch
from torch.utils.data import Dataset
import tqdm

import config
from utils import gmx_regen_top, gmx_test_top, gmx_anonymize, def_atom_name_to_type, find_duplicates, pint_reg, \
    addH_to_pdb
from chemistry.energy import gmx_calc_subdir
from data.data_classes import Biomolecule, c_alpha_class_name, c_alpha_sidechain_class_name, heavy_atoms_class_name, \
    all_atoms_class_name, element_classes, element_classes_abb, pdb_length_units, mda_length_units, prot_sel_crit, \
    stats_expiry_date, uni_from_top
from data.download import download_dataset
from data.forcefields import Forcefield, charmm36m, charmm22star, amber14sb_OL15, forcefields, canonical_aa_codes_3
from analysis.stats import Statistics

if TYPE_CHECKING:
    from models import DiffPGNN

# Define name of various datasets
supported_datasets = {'nup98_12_debug', 'nup98_12_debug2', 'nup98_12_debug3',
                      'nup98_12', 'nup98_24', 'AAQAA3', 'RS', 'CLN025', 'nup98C_12', 'nup98AC_12',
                      'Rich_2018', 'Rich_2018_debug', 'PDB_2RVD'}

# New dataset names
proc_datasets = ['nup98_12', 'nup98_24', 'AAQAA3', 'RS', 'CLN025', 'nup98C_12', 'nup98AC_12', 'Rich_2018', 'PDB_2RVD']
main_datasets = {'nup98_12': 'nup98_12', 'nup98_24': 'nup98_24', 'CLN': 'CLN025', 'RS': 'RS', 'AAQ': 'AAQAA3'}
gen_datasets = {'nup98_12': 'Diff_CS1_HA1_nup98_12_1_C1H1',
                'nup98_24': 'Diff_CS1_HA1_nup98_24_1_C1H1',
                'CLN': 'Diff_CS1_HA1_CLN025_1_C1H1',
                'RS': 'Diff_CS1_HA1_RS_1_C1H1',
                'AAQ': 'Diff_CS1_HA1_AAQAA3_1_C1H1',
                'nup98C_12': 'Diff_CS1_HA1_nup98C_12_1_C1H1',
                'nup98AC_12': 'Diff_CS1_HA1_nup98AC_12_1_C1H1'}
gen_datasets_main = {k: v for k, v in gen_datasets.items() if k in main_datasets}
gen_datasets_min = {k: gen_datasets[k].replace('_1_', '_S1_') for k in main_datasets}
thermalized_gen_datasets = {k: v + '_T100' for k, v in gen_datasets.items() if k in main_datasets}
thermalized_gen_datasets['CLN'] = thermalized_gen_datasets['CLN'].replace('_T100', '_U100')
chig_exp_gen_datasets = {'Folded_HA_2.5A_in_train': 'Diff_CS1_HA1_CLN025_FinTr2_5_C1H1',
                         'Folded_HA_3A_in_train': 'Diff_CS1_HA1_CLN025_FinTr3_C1H1',
                         'Unfolded_HA_4A_in_train': 'Diff_CS1_HA1_CLN025_UinTr4_C1H1',
                         'Unfolded_HA_5A_in_train': 'Diff_CS1_HA1_CLN025_UinTr5_C1H1',
                         'Unfolded_HA_6A_in_train': 'Diff_CS1_HA1_CLN025_UinTr6_C1H1'}
psi_heavy_scan_gen_datasets = {f'psi={k}': f'Diff_HA1_nup98_12_1_Psi{k}' for k in [0, 1, 2, 5, 10]}
psi_coarse_scan_gen_datasets = {f'psi={k}': f'Diff_CS1_HA1_nup98_12_1_Psi{k}_2' for k in [0, 1, 2, 5, 10]}

all_gen_datasets = list(gen_datasets_min.values())
all_gen_datasets += list(gen_datasets.values())
all_gen_datasets += list(thermalized_gen_datasets.values())
all_gen_datasets += list(chig_exp_gen_datasets.values())
all_gen_datasets += list(psi_heavy_scan_gen_datasets.values())
all_gen_datasets += list(psi_coarse_scan_gen_datasets.values())

# Create a temporary dataset directory for temporary datasets
dataset_tempdir = Path(tempfile.gettempdir()) / f'{config.project_name}_datasets'
dataset_tempdir.mkdir(exist_ok=True)

# Default patterns for finding raw MD files in XTCDataset processing.
default_xtc_pattern = r".*?prod(_)?(?P<ID>\d+)_prot.xtc"
default_tpr_pattern = r".*?prod(_)?(?P<ID>\d+)_prot.tpr"
default_top_pattern = r".*.top"


class BiomoleculeDataset(Dataset):
    statistics_subdir = 'statistics'  # Subdirectory where statistics are saved
    calc_subdir = Path('calculations_cache')  # Subdirectory where calculations are cached.
    processed_npz_filename = 'processed.npz'  # Name of the archive file storing the processed molecules
    metadata_filename = 'metadata.npz'  # Name of the archive file storing dataset metadata
    metadata_attrs_name = ['top_filepath', 'forcefield', 'terminals_gmx_code', 'splits_sets',
                           'remarks']  # Attributes saved as metadata
    noninherited_attrs: list[str] = ['splits_sets', 'remarks']  # Metadata attributes that are not inherited.
    splits_name = ['train', 'valid', 'test']  # Name of default dataset splits

    def __init__(self, directory: str | Path = '', name='', raw_dir: str | Path = '', load_all=False,
                 debug: bool = None, reprocess=False, top_filepath: str | Path = None,
                 forcefield: Forcefield = None, terminals_gmx_code=None, element_class: str = None, **process_kwargs):
        # Directories and filepaths
        if directory:
            directory = Path(directory)
        elif name:
            directory = Path(config.proc_datasets_dir, name)
        else:
            raise ValueError(f"'name' or 'directory' must be given as input")

        self.name = name if name else directory.stem
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = Path(raw_dir)
        self.calc_dir = self.dir / self.calc_subdir
        self.processed_npz_filepath = self.dir / self.processed_npz_filename
        self.metadata_filepath = self.dir / self.metadata_filename
        self.gmx_energy_filepath = self.dir / gmx_calc_subdir / 'gmx_energy.npz'

        self.debug = config.debug if debug is None else debug  # Only load 100 files in test mode
        self._statistics = None  # Defined later, if needed
        self.top_filepath = pathlib.Path(top_filepath) if top_filepath is not None else None
        self.forcefield = forcefield
        self.terminals_gmx_code = terminals_gmx_code
        self._element_class = element_class
        self.splits_sets: dict[str, dict[str, torch.Tensor]] = dict()  # Sets of splits that are defined.
        self.split_ID: tuple[str, str] = None  # Not None when the dataset is a split of the original dataset
        self.molecules: list[Biomolecule] = []
        self.remarks = ''

        # Process raw files (if necessary) and create a .npz file of all data samples
        if reprocess or not self.processed_npz_filepath.exists():
            self.process(**process_kwargs)

        # Check if process() saved the processed file
        if not self.processed_npz_filepath.exists():
            raise FileExistsError(f"process() did not save the processed file {self.processed_npz_filepath!r}")

        # Initialize the molecules and filenames attributes.
        # Open the processed npz file to read all the processed data samples.
        self.processed_npzfile = np.load(self.processed_npz_filepath, allow_pickle=True)
        self.molecules_ID: list[str] = list(self.processed_npzfile.keys())
        self.molecules_ID_all = self.molecules_ID

        # Load metadata if any
        self.load_metadata()

        # Retain only 100 molecules for testing
        test_size = 100
        if self.debug and len(self) > test_size:
            self.subset(ind=np.arange(test_size), inplace=True)

        # Inherit basic info about the biomolecules using the first molecule in the dataset
        biomolecule_template = self.load_mol(0)
        self.element_types = biomolecule_template.supported_types('element')
        self.residue_types = biomolecule_template.supported_types('residue')
        biomolecule_finest_template = self.load_mol(0, post_process=False)
        self.element_class_finest = copy.copy(biomolecule_finest_template.element_class)

        # If element-class has not been given, set it to the class of the saved molecules
        if self._element_class is None:
            self._element_class = copy.copy(biomolecule_template.element_class)

        # Load all the processed data, if requested
        if load_all:
            self.load_all()

    def process(self, *args, **kwargs):
        """
        Processes the raw data files and creates .npz file containing the processed data.
        """
        raise NotImplementedError(f'process() must be implemented to process the raw files and save them into '
                                  f'{self.processed_npz_filepath!r}')

    def load_mol(self, i, post_process=True, file_handle=None) -> Biomolecule:
        if file_handle is None:
            if self.processed_npzfile.zip is None:
                self.processed_npzfile = np.load(self.processed_npz_filepath, allow_pickle=True)
            file_handle = self.processed_npzfile

        loaded_mol: Biomolecule = file_handle[self.molecules_ID[i]].item()

        # Check that the loaded molecule ID matches with the saved ID.
        if self.molecules_ID[i] != loaded_mol.ID:
            raise ValueError(f"ID of loaded molecule at index {i} does not match with metadata.")

        # Inherit the topology filename from the dataset
        loaded_mol.top_file = self.top_filepath

        # Set the element class of the loaded molecule to match the element class of the dataset
        if post_process and self.element_class is not None:
            loaded_mol.coarsen(element_class=self.element_class)

        return loaded_mol

    def load_all(self, force=False, **load_kwargs):
        """
        Loads all the molecules in the molecules attribute
        """
        if not self.is_mols_loaded or force:
            self.molecules = [None for _ in range(len(self.molecules_ID))]
            for i in tqdm.tqdm(range(len(self)), desc=f'Loading {self.name!r} dataset.'):
                self.molecules[i] = self.load_mol(i, **load_kwargs)

    def create_splits(self, ratio: list[float] = None, index: np.ndarray = None, method='random', save=True,
                      names: list[str] = None, set_ID=None, regen=False):
        """
        Create a set of split indices and stores it into self.splits_sets.
        Args:
            ratio: Percentage of the original dataset contained in the respective splits
            index: Split index of each molecule in the dataset when method='index'
            method: Split the dataset randomly ('random'), in order ('fixed') or using a given split index ('index')
            save: Save the split indices in self.splits_sets for later use
            names: Names of the splits used.
            set_ID: ID of the splits set
            regen: Regenerate splits set and overwrites any previous splits set with the same set_ID
        """
        if set_ID is None:
            set_ID = str(len(self.splits_sets) + 1)

        # Check if the splits set has already been created. Skip if not overwriting.
        if set_ID in self.splits_sets and not regen:
            return self.splits_sets[set_ID]

        if method == 'index':
            n_splits = np.max(index) + 1
        else:
            n_splits = len(ratio)

        assert n_splits > 1, f'A minimum of 2 splits is needed. Received {n_splits} splits.'
        if names is None:
            if n_splits == len(self.splits_name):
                names = self.splits_name
            else:
                names = [uuid.uuid4().hex[:8] for _ in range(n_splits)]

        # Find the new order of the molecules
        if method in ['random', 'fixed']:
            if method == 'random':
                data_ind = torch.randperm(len(self))
            elif method == 'fixed':
                data_ind = torch.arange(len(self))
            else:
                raise ValueError
            ratio_cumsum = np.cumsum(ratio)
            splits_end_ind = np.round(len(self) * ratio_cumsum / ratio_cumsum[-1]).astype(int)
            splits_mol_ind = np.split(data_ind, splits_end_ind[:-1])
        elif method == 'index':
            data_ind = torch.arange(len(self))
            splits_mol_ind = [data_ind[index == i] for i in range(n_splits)]
        else:
            raise ValueError(f"Method {method!r} not supported")
        splits_set = {s: ind for s, ind in zip(names, splits_mol_ind)}

        # Store the splits indices for future use
        if save:
            self.splits_sets[set_ID] = splits_set
            self.save_metadata()

        return splits_set

    def delete_splits_set(self, set_ID: str, verbose=True):
        if set_ID not in self.splits_sets:
            if verbose:
                print(f'Split set {set_ID!r} of {self.name!r} was not found. Nothing was deleted.')
            return

        # Delete statistics of the splits
        splits = self.load_splits(set_ID=set_ID)
        for split in splits:
            split.delete_statistics()

        # Remove the splits_set and update metadata
        self.splits_sets.pop(set_ID)
        self.save_metadata()

        if verbose:
            print(f'Split set {set_ID!r} of {self.name!r} was successfully deleted.')

    def load_splits(self, name: str | list[str] = None, set_ID: str = None, load_all=None) -> Self | list[Self]:
        """
        Loads a given split if it exists. Returns FileExistError otherwise
        Args:
            name: name of the splits to be loaded
            set_ID: ID of the splits set. Defaults to the first element found in self.splits_sets.
            load_all: loads all molecules in the splits
        Returns:
            splits of the dataset
        """
        if set_ID is None:
            set_ID = config.dataset_split_set_ID

        # If no name is given, load all splits in the split set
        if name is None:
            name = list(self.splits_sets[set_ID].keys())
        return_single = not isinstance(name, list)
        if return_single:
            name = [name]

        # If debug is on, create a temporary set of splits
        if self.debug:
            splits_ind_set = self.create_splits(3 * [1 / 3], method='fixed', set_ID='test', save=False)
        else:
            if set_ID not in self.splits_sets:
                raise ValueError(f"split set {set_ID!r} does not exist. Create it first with create_splits().")
            splits_ind_set = self.splits_sets[set_ID]

        # Load the splits
        dataset_splits = []
        for n in name:
            dataset_split = self.subset(ind=splits_ind_set[n], inplace=False, inmemory=True)
            dataset_split.name = f'{self.name}_{set_ID}_{n}'  # split name maintains name of original dataset
            dataset_split.splits_sets = {}
            dataset_split.split_ID = (set_ID, n)

            dataset_splits.append(dataset_split)

        if return_single:
            return dataset_splits[0]
        else:
            return dataset_splits

    @property
    def ID(self):
        """
        Creates the ID of the dataset from the hash of all its molecule filenames
        Returns:
            Dataset ID (str)
        """
        hashed_content = " ".join(sorted(self.molecules_ID)) + self.element_class
        ID = hashlib.md5(hashed_content.encode('utf-8')).hexdigest()
        return ID

    @property
    def element_class(self):
        return self._element_class

    @element_class.setter
    def element_class(self, element_class: str):
        if element_class not in element_classes:
            raise ValueError(f"element_class {element_class!r} is not supported. Choices are:{element_classes}")

        # Check if it is possible to set the element_class to the given element_class by ensuring it is as coarse as the
        # finest element class
        if element_classes.index(element_class) > element_classes.index(self.element_class_finest):
            raise ValueError(f"Cannot set element_class of {self.name!r} to {element_class!r}. "
                             f"Saved molecules have a coarser element_class {self.element_class_finest!r}.")

        if element_class != self._element_class:
            element_class_prev = self._element_class
            self._element_class = element_class

            # Reload molecules if the new element class is coarser than the current one. Otherwise, reload.
            if self.is_mols_loaded:
                if element_classes.index(element_class) < element_classes.index(element_class_prev):
                    # Coarsen the currently-loaded molecules.
                    for mol in tqdm.tqdm(self.molecules, f'Coarsening {self.name!r} to {element_class!r}'):
                        mol.coarsen(element_class=element_class)
                else:
                    # Reload all molecules.
                    self.load_all(force=True)

            # Inherit basic info about the biomolecules using the first molecule in the dataset
            biomolecule_template = self.load_mol(0)
            self.element_types = biomolecule_template.supported_types('element')
            self.residue_types = biomolecule_template.supported_types('residue')

            # Reset the value of _statistics since they need to be reloaded/re-calculated
            self._statistics = None

    @property
    def element_class_abb(self):
        """
        Returns an abbreviated form of the dataset's element class
        Returns:
            Short form of element_class
        """
        element_class_abb_map = {v: k for k, v in element_classes_abb.items()}
        return element_class_abb_map[self.element_class]

    @property
    def is_mols_loaded(self):
        """
        Returns True if all molecules are loaded

        Returns: bool

        """
        return len(self.molecules) > 0

    @property
    def is_split(self):
        """Returns True if the dataset is a split of another dataset"""
        # name_ends_with_split_suffix = any([self.dir.endswith(split_name) for split_name in self.splits_name])
        # parent_dir_is_main = os.path.dirname(self.dir).endswith(self.main_name)
        # return name_ends_with_split_suffix and parent_dir_is_main
        return self.split_ID is not None

    @property
    def is_subset(self):
        if hasattr(self, 'molecules_ID') and hasattr(self, 'molecules_ID_all'):
            return self.molecules_ID != self.molecules_ID_all
        else:
            # The dataset is still in construction so it is not a subset.
            return False

    @property
    def has_splits(self):
        # # In split(), name of split subdirectories start with self.ID. Use that to find splits in self.dir.
        # dir_path = Path(self.dir)
        # split_subdirs = [p for p in dir_path.iterdir() if p.stem.startswith(self.ID)]
        # return bool(split_subdirs)
        return len(self.splits_sets) > 0

    @property
    def main_name(self):
        if self.is_split:
            return self.name.removesuffix(f"_{self.split_ID[0]}_{self.split_ID[1]}")

        return self.name

    def subset(self, criterion: 'Criterion' = None, ind: torch.Tensor | np.ndarray | list = None, inplace=True,
               inmemory=True, **constructor_kwargs) -> None | Self:
        """
        Takes a subset of all the molecules following an input Criterion applied on the molecule or an index array
        Args:
            criterion: instance of the Criterion class
            ind: logical or integer indexing representing the subset
            inplace: performs the subsetting inplace. Otherwise, returns a copy of self. MOLECULES ARE NOT COPIED.
            inmemory: performs the subsetting in memory. Otherwise, creates a temporary directory to store the subset.
            constructor_kwargs:  extra kwargs passed to the class constructor
        """
        if isinstance(ind, list):
            ind = np.array(ind)
        if isinstance(ind, torch.Tensor):
            ind = ind.numpy()

        if criterion is not None:
            self.load_all()
            is_molecule_kept = np.array([criterion(mol) for mol in self.molecules])
            molecule_kept_ind = np.nonzero(is_molecule_kept)[0]
        elif ind.dtype == bool:
            molecule_kept_ind = np.nonzero(ind)[0]
        elif ind.dtype == int:
            molecule_kept_ind = ind
        else:
            raise ValueError(f"index of dtype {type(ind.dtype)} is not supported")

        if inmemory:
            # Temporarily detach molecules to avoid copying molecules if not inplace
            molecules = self.molecules
            self.molecules = []

            if inplace:
                dataset_subset = self
            else:
                dataset_subset = copy.deepcopy(self)

            dataset_subset.molecules_ID = [self.molecules_ID[i] for i in molecule_kept_ind]
            if molecules:
                dataset_subset.molecules = [molecules[i] for i in molecule_kept_ind]

            # Reset statistics since the subset will change them
            dataset_subset._statistics = None

            # Re-attach molecules
            self.molecules = molecules
        else:
            # Create a standalone dataset in a temporary folder and transfer metadata
            constructor_kwargs = copy.deepcopy(constructor_kwargs)
            constructor_kwargs['element_class'] = self.element_class
            if 'debug' not in constructor_kwargs:
                constructor_kwargs['debug'] = self.debug
            if 'name' not in constructor_kwargs and 'directory' not in constructor_kwargs:
                constructor_kwargs['name'] = f"{self.name}_subset"
            if 'top_filepath' not in constructor_kwargs:
                constructor_kwargs['top_filepath'] = self.top_filepath

            molecules = [self.load_mol(i) for i in molecule_kept_ind]
            dataset_subset = self.__class__.from_molecules(molecules, **constructor_kwargs)
            dataset_subset.inherit_metadata(self)

        return dataset_subset

    @property
    def statistics_filepath(self):
        filename = self.element_class
        if self.is_split:
            filename += f"_{self.split_ID[0]}_{self.split_ID[1]}"
        filename += f"_{self.ID[:8]}.npz"
        filepath = self.dir / self.statistics_subdir / filename
        return filepath

    def statistics(self, props_name: str | list[str] = None) -> dict[str:Statistics] | Statistics:
        """
            Returns various statistics of the given dataset
        """
        if self._statistics is None:
            self._statistics = self.load_statistics()

        if props_name is None:
            return self._statistics

        single_prop = isinstance(props_name, str)
        if single_prop:
            props_name = [props_name]

        # Calculate any missing props
        missing_props = set(props_name) - set(self._statistics.keys())
        if missing_props:
            from analysis.metrics import calculate_statistics
            self._statistics |= calculate_statistics(self, props_name=missing_props, save=not config.debug)

        if single_prop:
            return self._statistics[props_name[0]]
        else:
            return {k: self._statistics[k] for k in props_name}

    def load_statistics(self) -> dict[str:Statistics]:
        """
        Loads various statistics of the dataset if it exists. Otherwise, returns empty dictionary.
        Returns:
            stats_dict: dictionary of the statistics
        """
        statistics_filepath = self.statistics_filepath

        if statistics_filepath.exists():
            with np.load(statistics_filepath, allow_pickle=True) as npzfile:
                saved_stats_dict = {}
                for k in npzfile:
                    saved_stats_dict[k] = npzfile[k].item()

            # Rename old stats
            old_stats_new_name = {}
            old_stats_names = set(old_stats_new_name) & set(saved_stats_dict)
            if old_stats_names:
                for old_name in old_stats_names:
                    new_name = old_stats_new_name[old_name]
                    saved_stats_dict[new_name] = saved_stats_dict.pop(old_name)
                    warnings.warn(f"Renaming old stats {old_name!r} to {new_name!r} in {self.name!r}.")

                # Resave stats
                self.save_statistics(saved_stats_dict, update=False)

            return saved_stats_dict
        else:
            return {}

    def save_statistics(self, stats_dict: dict[str:Statistics], update=True):
        """
        Saves given statistics dictionary to the statistics .npz file
        Args:
            stats_dict: statistics dictionary to be saved in the statistics file
            update: updates the saved statistics (if any) with the given statistics dictionary
        Returns:
            None
        """
        # Load the statistics file if it exists.
        statistics_filepath = self.statistics_filepath
        statistics_filepath.parent.mkdir(parents=True, exist_ok=True)
        if statistics_filepath.exists() and update:
            saved_stats_dict = self.load_statistics()
        else:
            saved_stats_dict = {}
        saved_stats_dict.update(stats_dict)
        np.savez(statistics_filepath, **saved_stats_dict)

    def delete_statistics(self, props_name: str | list[str] = None):
        """
        Deletes statistics of given properties in the saved statistics file
        Args:
            props_name: list of properties to delete. Defaults to all properties (file is deleted as well)

        Returns:
            None
        """
        if props_name is None:
            self.statistics_filepath.unlink(missing_ok=True)
            self._statistics = None
        else:
            # Delete only the given properties in the statistics dictionary.
            props_name = props_name if isinstance(props_name, list) else [props_name]
            stats_dict = self.statistics()
            resave = False
            for name in props_name:
                if name in stats_dict:
                    stats_dict.pop(name)
                    print(f'Deleted {name!r} statistics of {self.name!r}.')
                    resave = True
            if resave:
                self.save_statistics(stats_dict, update=False)
                self._statistics = stats_dict

    def delete_expired_statistics(self, expiry_date: str | datetime.datetime = stats_expiry_date):
        """
        Deletes statistics file if it's creation time is older than the given expiry date.
        Args:
            expiry_date: string or datetime object giving the expiry date

        Returns:
            None
        """
        if isinstance(expiry_date, str):
            expiry_date = datetime.datetime.strptime(expiry_date, "%Y-%m-%d %H:%M:%S")

        stats_filepath = self.statistics_filepath
        if stats_filepath.exists():
            created_datetime = datetime.datetime.fromtimestamp(stats_filepath.stat().st_ctime)
            if created_datetime < expiry_date:
                self.delete_statistics()

    @classmethod
    def save_molecules(cls, biomolecules: list[Biomolecule], directory: str | Path = None,
                       filepath: str | Path = None, overwrite=False):
        """
        Saves a list of biomolecules into a .npz file
        Args:
            biomolecules: list of Biomolecule objects
            directory: directory where biomolecules data file will be saved (with filename cls.processed_npz_filename)
            filepath: filepath of the processed file (alternative option to giving directory)
            overwrite: overwrite any previous .npz file found in directory. Otherwise, raise FileExistsError
        """

        # Create the dataset directory if it does not exist
        if filepath is not None:
            filepath = pathlib.Path(filepath)
            assert filepath.is_absolute(), "Only absolute filepaths are supported."
            directory = filepath.parent
            filename = filepath.name
        else:
            filename = cls.processed_npz_filename

        if directory is None:
            directory = tempfile.mkdtemp(dir=dataset_tempdir)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        processed_npz_filepath = directory / filename

        # Check if the processed file exists and can be overwritten
        if processed_npz_filepath.exists() and not overwrite:
            raise FileExistsError(f"{filename!r} already exists in {str(directory)!r}.")

        # Save the molecules in the processed .npz file to bypass raw processing
        mols_ID = np.array([mol.ID for mol in biomolecules])
        is_mol_ID_well_defined = np.array([ID is not None and ID != '' for ID in mols_ID])
        if not is_mol_ID_well_defined.all():
            raise ValueError(f"The following molecule IDs are not allowed:{mols_ID[is_mol_ID_well_defined]}")

        # Check that the molecules ID are unique
        mols_ID_duplicates = find_duplicates(mols_ID)
        if mols_ID_duplicates:
            raise ValueError(f"The following molecule IDs are duplicated:\n{mols_ID_duplicates}")

        # Remove topology attr in saved mols since top_filepath contains private path info
        saved_mols = {mol.ID: mol for mol in biomolecules}
        mol_top_files = {mol.ID: mol.top_file for mol in biomolecules}
        for mol in biomolecules:
            mol.top_file = None

        # Save the molecules in a compressed .npz file.
        # Use temporary file to avoid errors when process is interrupted in the middle of save
        temp_filepath = pathlib.Path(tempfile.mktemp(dir=dataset_tempdir)).with_suffix('.npz')
        np.savez_compressed(temp_filepath, **saved_mols)
        shutil.move(temp_filepath, processed_npz_filepath)

        # Restore top file
        for mol in biomolecules:
            mol.top_file = mol_top_files[mol.ID]

        return directory

    def resave_molecules(self):
        """
        Resaves the processed file of the dataset using the new annotations of the Biomolecule class
        Returns:
                None
        """
        self.load_all()
        mols_new = []
        for mol in tqdm.tqdm(self, f'Re-saving molecules of {self.name!r}'):
            mol_attrs = {n: getattr(mol, n) for n in Biomolecule.__annotations__}
            mols_new.append(Biomolecule(**mol_attrs))

        self.save_molecules(mols_new, filepath=self.processed_npz_filepath, overwrite=True)
        self.molecules = []

    @classmethod
    def from_molecules(cls, biomolecules: list[Biomolecule], directory: str | Path = None, **constructor_kwargs):
        """
        Creates a dataset from a set of biomolecules
        Args:
            biomolecules: list of Biomolecule objects
            directory: directory where biomolecules data will be saved
        Returns:
            object of type cls
        """
        directory = cls.save_molecules(biomolecules=biomolecules, directory=directory)
        return cls(directory=directory, **constructor_kwargs)

    @classmethod
    def random(cls, n_molecules=100, biomolecule_kwargs: dict = None, **dataset_kwargs):
        """
        Creates a dataset of random molecules
        Args:
            n_molecules: Number of molecules in the dataset
            biomolecule_kwargs: kwargs passed to the random() constructor of Biomolecule
            dataset_kwargs: kwargs passed to the from_molecules() constructor of BiomoleculeDataset

        Returns: BiomoleculeDataset

        """
        # Define the default kwargs
        if biomolecule_kwargs is None:
            biomolecule_kwargs = {}
        if 'element_class' not in biomolecule_kwargs:
            biomolecule_kwargs['element_class'] = c_alpha_class_name

        # Create a biomolecule template and let other molecules inherit its fixed attributes
        biomol_template = Biomolecule.random(**biomolecule_kwargs)
        inherited_attrs = ['properties_name']
        for attr in inherited_attrs:
            biomolecule_kwargs[attr] = getattr(biomol_template, attr)

        biomolecules = [Biomolecule.random(**biomolecule_kwargs) for _ in range(n_molecules)]
        dataset = cls.from_molecules(biomolecules, **dataset_kwargs)
        return dataset

    @classmethod
    def is_processed(cls, dataset_dir: str | pathlib.Path):
        return (dataset_dir / cls.processed_npz_filename).exists()

    def refine(self, element_class: str, **kwargs):
        """
        Refines molecules in the dataset to a finer element_class
        Args:
            element_class: new element class of the dataset's molecules
            **kwargs: kwargs given to constructor

        Returns:
            BiomoleculeDataset with refined molecules
        """
        mol_template = Biomolecule.from_topology(top_filepath=self.top_filepath, element_class=element_class)
        mol_iter = tqdm.tqdm(self, desc=f'Refining molecules of {self.name!r}')
        new_mols = [mol.refine(mol_template=mol_template) for mol in mol_iter]
        dataset = self.__class__.from_molecules(new_mols, **kwargs)
        dataset.inherit_metadata(self, omit_default=False)
        return dataset

    def fix_topology(self):
        # Ensures that the saved topology filepath points to the local dataset directory. If not, copy the file.
        resave_metadata = False
        if self.top_filepath is not None:
            if self.top_filepath.parent != self.dir:
                new_top_filepath = self.dir / self.top_filepath.name
                if not new_top_filepath.exists():
                    shutil.copy(self.top_filepath, new_top_filepath)
                self.top_filepath = new_top_filepath
                resave_metadata = True

            # Try loading the topology file to ensure it is error-free.
            gmx_test_top(self.top_filepath)

        return resave_metadata

    def load_metadata(self):
        resave_metadata = False
        metadata_dict = {}
        if self.metadata_filepath.exists():
            with np.load(self.metadata_filepath, allow_pickle=True) as npzfile:
                for k, v in dict(npzfile).items():
                    metadata_dict = v.item()  # npzfile has only one key

            for attr_name, attr_val in metadata_dict.items():
                if not hasattr(self, attr_name):
                    warn_msg = (f"Found attribute {attr_name!r} in saved metadata but {self.name!r} does not define"
                                f"this metadata attribute. Attribute will not be loaded.")
                    warnings.warn(warn_msg)
                elif attr_name == 'top_filepath' and attr_val is not None and not pathlib.Path(attr_val).is_absolute():
                    self.top_filepath = self.dir / attr_val
                elif attr_name == 'forcefield' and isinstance(attr_val, str):
                    self.forcefield = forcefields[attr_val]
                elif attr_name in self.metadata_attrs_name:
                    setattr(self, attr_name, attr_val)
        resave_metadata |= self.fix_topology()

        if resave_metadata:
            self.save_metadata()

        return metadata_dict

    def save_metadata(self):
        assert not self.is_split, "Cannot save metadata when the dataset is a split"
        assert not self.is_subset, "Cannot save metadata when dataset is a subset"
        saved_metadata = {k: getattr(self, k) for k in self.metadata_attrs_name}

        # Save only the name of the forcefield. The Forcefield object is fetched from the name
        if self.forcefield:
            saved_metadata['forcefield'] = self.forcefield.name
        # Save only the filename of the topology file since the dataset directory may change
        if self.top_filepath:
            saved_metadata['top_filepath'] = self.top_filepath.relative_to(self.dir)

        # Save metadata as a dictionary to avoid calling item() on all dict values when loading
        np.savez(self.metadata_filepath, **dict(attrs=saved_metadata))

    def inherit_metadata(self, dataset: Self, omit_default=True, omitted_attrs: list[str] = None):
        transfered_attrs_name = set(self.metadata_attrs_name) & set(dataset.metadata_attrs_name)
        if omit_default:
            transfered_attrs_name -= set(dataset.noninherited_attrs)
        if omitted_attrs:
            transfered_attrs_name -= set(omitted_attrs)

        for attr_name in transfered_attrs_name:
            setattr(self, attr_name, getattr(dataset, attr_name))

        self.fix_topology()
        self.save_metadata()
        self.load_metadata()  # Necessary since some attributes are derived from metadata attributes

    def save_to_pdb(self, ind: list[int] | int = None, mols: list[Biomolecule] = None, pdb_dir: pathlib.Path = None,
                    regen=False, verbose=True, **save_pdb_kwargs):
        """
        Saves all or a subset of the dataset molecules in .pdb format to a subdirectory called 'pdb'
        Args:
            ind: indices of saved molecules. Defaults to range(len(self)).
            mols: list of Biomolecules to save to .gro (alternative to giving ind argument)
            pdb_dir: directory where .pdb files will be saved. Defaults to self.dir / 'pdb'
            regen: delete any previously saved .pdb files found in the pdb directory and regenerate them
            save_pdb_kwargs: kwargs given to the save_pdb method of Biomolecule
            verbose: Display tqdm progress bar
        Returns:
            None
        """
        # Check and sanitize inputs
        assert mols is None or ind is None, "Only 'ind' or 'mols' argument must be given."
        if isinstance(ind, int):
            ind = [ind]

        if mols is None:
            if ind is None:
                self.load_all()
                mols = self.molecules
            else:
                mols = [self[i] for i in ind]
        ind = range(len(mols))
        mols_ID = [mol.ID for mol in mols]

        # Set the default pdb directory
        if pdb_dir is None:
            pdb_dir = self.dir / 'pdb'
        pdb_dir.mkdir(exist_ok=True)

        if 'top_filepath' in save_pdb_kwargs:
            top_filepath = save_pdb_kwargs.pop('top_filepath')
        else:
            top_filepath = self.top_filepath

        # Remove .pdb files that already exist if not regenerating
        pdb_filepaths = {i: pdb_dir / f'{mols_ID[i]}.pdb' for i in ind}
        if regen:
            pdb_filepaths_remaining = pdb_filepaths
            for p in pdb_filepaths.values():
                p.unlink(missing_ok=True)
        else:
            pdb_filepaths_remaining = {i: p for i, p in pdb_filepaths.items() if not p.exists()}

        if pdb_filepaths_remaining:
            tasks_iter = pdb_filepaths_remaining.items()
            if verbose:
                tasks_iter = tqdm.tqdm(tasks_iter, desc=f'Saving molecules of {self.name!r} to .pdb files')

            for i, p in tasks_iter:
                mols[i].save_pdb(top_filepath=top_filepath, pdb_filepath=p, **save_pdb_kwargs)

        return pdb_filepaths

    def save_to_gro(self, ind: list[int] | int = None, mols: list[Biomolecule] = None, gro_dir: pathlib.Path = None,
                    regen=False, parallel=True, verbose=True, delete_temp=True, template_uni: mda.Universe = None,
                    pdb2gmx_kwargs: dict = None):
        """
        Saves all or a subset of the dataset molecules in .gro format
        Args:
            ind: indices of saved molecules. Defaults to range(len(self)).
            mols: list of Biomolecules to save to .gro (alternative to giving ind argument)
            gro_dir: directory where .gro files will be saved
            regen: delete any previously saved .gro files found in the gro directory and regenerate them
            parallel: performs the pdb2gmx() call in parallel
            verbose: Display tqdm progress bar
            delete_temp: delete temporary files created by pdb2gmx()
            template_uni: template MDAnalysis.Universe given to the save_to_pdb() method
            pdb2gmx_kwargs: dictionary of kwargs given to pdb2gmx() util function
        Returns:
            None
        """
        # Check and sanitize inputs
        assert mols is None or ind is None, "Only 'ind' or 'mols' argument must be given."
        if mols is None:
            mols = self.molecules
            mols_ID = self.molecules_ID
        else:
            ind = range(len(mols))
            mols_ID = [mol.ID for mol in mols]

        if ind is None:
            ind = range(len(self))
        elif isinstance(ind, int):
            ind = [ind]

        # Set the default gro directory
        if gro_dir is None:
            gro_dir = self.dir / 'gro'
        gro_dir.mkdir(exist_ok=True)

        # Delete previous .gro files found in gro dir
        gro_filepaths = {i: gro_dir / f'{mols_ID[i]}.gro' for i in ind}
        if regen:
            gro_remaining = gro_filepaths
            for p in gro_filepaths.values():
                p.unlink(missing_ok=True)
        else:
            gro_remaining = {i: p for i, p in gro_filepaths.items() if not p.exists()}

        # Set the default pdb2gmx_kwargs if None are given
        if pdb2gmx_kwargs is None:
            pdb2gmx_kwargs = dict(lb=0.0, forcefield_dir=self.forcefield.dir,
                                  ter=None, terminals_protonation=self.terminals_gmx_code)
        if 'working_dir' not in pdb2gmx_kwargs:
            pdb2gmx_kwargs['working_dir'] = gro_dir
        if 'ignh' not in pdb2gmx_kwargs:
            pdb2gmx_kwargs['ignh'] = self.element_class == heavy_atoms_class_name

        if gro_remaining:
            # First, save the molecules in .pdb format
            # Initialize a template MDAnalysis Universe
            if template_uni is None:
                template_uni = uni_from_top(top_filepath=self.top_filepath, element_class=self.element_class,
                                            trajectory=True)

            # # Fix the size of the universe to fit every molecule
            # box_dim_all = np.array([mol.box_params()[1] for mol in molecules]).max()
            # box_dim_all = pint_reg.Quantity(box_dim_all, mol_ex.length_units).to(pdb_length_units)
            # template_uni.dimensions = 3 * [box_dim_all.magnitude] + 3 * [90]
            pdb_dir = gro_dir / 'pdb_temp'
            pdb_filepaths = self.save_to_pdb(mols=mols, pdb_dir=pdb_dir, template_uni=template_uni)

            # Use GROMACS' pdb2gmx to create the .gro files of the molecules
            pdb_filepaths_gro_remaining = [pdb_filepaths[i] for i in gro_remaining]
            tqdm_desc = f'Saving molecules of {self.name!r} to .gro files (ignH={pdb2gmx_kwargs["ignh"]})' if verbose else ''
            curr_dir = os.getcwd()
            os.chdir(pdb2gmx_kwargs['working_dir'])
            addH_to_pdb(pdb_filepaths_gro_remaining, tqdm_desc=tqdm_desc, parallel=parallel, **pdb2gmx_kwargs)
            os.chdir(curr_dir)

            # Delete temporary pdb directory
            shutil.rmtree(pdb_dir)

        if delete_temp:
            for p in gro_filepaths.values():
                p.with_suffix('.top').unlink(missing_ok=True)
                p.with_stem(p.stem + '_posre').with_suffix('.itp').unlink(missing_ok=True)

        return gro_filepaths

    def save_to_trr(self, filepath: str | Path = None, template_uni: mda.Universe = None):
        """
        Saves the dataset's molecules into a GROMACS .trr file
        Args:
            filepath: filepath of the .trr file. Defaults to self.dir + 'molecules.trr'
            template_uni: template universe used to build the .trr trajectory with MDAnalysis

        Returns:
            filepath of the .trr file
        """

        if filepath is None:
            filepath = self.dir / 'molecules.trr'
        filepath = pathlib.Path(filepath)
        if filepath.exists():
            return filepath

        # Load all molecules
        self.load_all()

        # Initialize universe from topology
        if template_uni is None:
            template_uni = uni_from_top(top_filepath=self.top_filepath, element_class=all_atoms_class_name,
                                        trajectory=True)

            # Determine the size of the simulation box that will fit every molecule
            box_dim_all = np.array([mol.box_params()[1] for mol in self.molecules]).max()
            box_dim_all = pint_reg.Quantity(box_dim_all, self.molecules[0].length_units).to(pdb_length_units)
            box_dim_all = box_dim_all.magnitude
            template_uni.dimensions = np.array(3 * [box_dim_all] + 3 * [90.0])

        # Gather all molecules positions
        gro_temp_dir = self.dir / 'trr_gro_temp'

        # Try loading all-atom molecules if available
        dataset_element_class = self.element_class
        try:
            self.element_class = all_atoms_class_name
        except:
            pass

        if self.element_class == all_atoms_class_name:
            mol_to_mda_pos_conv = pint_reg.Quantity(1, self.molecules[0].length_units).to(mda_length_units).magnitude
            atoms_pos_all = np.stack([mol.elements_position * mol_to_mda_pos_conv for mol in self], axis=0)
        else:
            # If molecules are not all-atoms, use GROMACS' pdb2gmx to add hydrogen and concatenate the .gro files
            gro_filepaths = self.save_to_gro(gro_dir=gro_temp_dir, delete_temp=False)

            atoms_pos_all = np.nan * np.ones((len(self), len(template_uni.atoms), 3))
            for i in range(len(self)):
                uni = mda.Universe(gro_filepaths[i], in_memory=True)
                atoms_pos_all[i] = uni.atoms.positions

            if np.isnan(atoms_pos_all).any():
                raise ValueError(f"An error occurred in loading the molecules' positions from .gro files.")

        # Concatenate all positions into a .trr file
        template_uni.load_new(atoms_pos_all, format=mda.coordinates.memory.MemoryReader, order='fac',
                              dimensions=template_uni.dimensions)
        template_uni.atoms.write(filepath, frames='all')

        # Delete temporary .gro directory if it was created
        if not self.element_class == all_atoms_class_name and gro_temp_dir.exists():
            shutil.rmtree(gro_temp_dir)

        # Restore the element_class in case it changed in the try statement above
        self.element_class = dataset_element_class

    def hydrogenize(self, molecules: list[Biomolecule] = None, save=True, regen=False, parallel=True,
                    temp_dir: str | pathlib.Path = None):
        """
        Add hydrogen atoms to a set of molecules
        Args:
            molecules: molecules to hydrogenize. Defaults to self.molecules.
            save: Save the hydrogenized molecules in the processed file of the dataset
            regen: regenerate any previous temporary files found in the dataset directory
            parallel: performs the pdb2gmx() call in parallel
            temp_dir: name of the temporary subdirectory located in self.dir where all temporary files are stored.
        """
        assert not self.is_subset, ("Hydrogenization cannot be performed on a subset of a dataset. "
                                    "Call hydrogenize() on the full dataset.")
        assert element_classes.index(self.element_class) >= element_classes.index(heavy_atoms_class_name), \
            (f"Hydrogenization can only be performed on molecules of element class >= {heavy_atoms_class_name}."
             f"{self.name!r} has molecules of class={self.element_class!r}.")
        use_dataset_mols = molecules is None

        # Return None if the dataset's molecules were already hydrogenized and no input molecules are given
        if not regen and use_dataset_mols and 'Hydrogenized' in self.remarks:
            return None

        # Load all molecules in the dataset if None are given
        if use_dataset_mols:
            self.load_all()
            molecules = self.molecules

        # Create temporary dir where .pdb and .gro will be saved
        if temp_dir is None:
            temp_dir = self.dir / 'hydrogenization_temp'
        if temp_dir.exists() and regen:
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        pdb2gmx_kwargs = dict(lb=0.0, working_dir=temp_dir, forcefield_dir=self.forcefield.dir, ignh=True,
                              ter=None, terminals_protonation=self.terminals_gmx_code)
        gro_filepaths = self.save_to_gro(mols=molecules, gro_dir=temp_dir, parallel=parallel,
                                         pdb2gmx_kwargs=pdb2gmx_kwargs)

        # Load the .gro filenames into a mda.Universe and initialize the hydrogenized molecules
        molecules_hydrogenized = []
        atom_name_to_type = def_atom_name_to_type(self.top_filepath)
        tasks_iter = tqdm.tqdm(gro_filepaths.items(), desc=f'Assembling hydrogenized molecules of {self.name!r}')
        for i, gro_filepath in tasks_iter:
            uni = mda.Universe(gro_filepath, convert_units=False, in_memory=True)
            mol = Biomolecule.from_universe(md_uni=uni, element_class=all_atoms_class_name,
                                            ID=molecules[i].ID,
                                            atom_name_to_type=atom_name_to_type,
                                            forcefield=self.forcefield.name)
            molecules_hydrogenized.append(mol)

        # Save the hydrogenized molecules
        if save:
            # If molecules were given has inputs, their ordering may not agree with the ordering that is already defined
            # in the dataset by the 'molecules_ID' attribute. Ensure that the ordering of the hydrogenized molecules match
            # the ordering that is already defined. This is only necessary if the dataset has been built already.
            # If hydrogenize() is called in a dataset process(), 'molecules_ID' won't be defined, in which case, there
            # is no pre-determined ordering to follow.
            if hasattr(self, 'molecules_ID'):
                molecules_hydrogenized_dict = {mol.ID: mol for mol in molecules_hydrogenized}
                molecules_hydrogenized_saved = [molecules_hydrogenized_dict[ID] for ID in self.molecules_ID]
            else:
                molecules_hydrogenized_saved = molecules_hydrogenized
            self.save_molecules(biomolecules=molecules_hydrogenized_saved, filepath=self.processed_npz_filepath,
                                overwrite=True)
            self.element_class_finest = all_atoms_class_name
            self.remarks += 'Hydrogenized\n'
            self.save_metadata()

            # Close the npzfile if it was previously open. Below, hasattr() is needed in case hydrogenize is called in
            # a class constructor that may not have defined the 'processed_npzfile' attribute
            if hasattr(self, 'processed_npzfile') and self.processed_npzfile.zip is not None:
                self.processed_npzfile.close()

            # Replace the molecules attribute with the hydrogenized molecules.
            is_mols_loaded = self.is_mols_loaded
            self.molecules = []  # Necessary to avoid reloading the molecules when element_class is changed in next line
            self.element_class = all_atoms_class_name
            if is_mols_loaded:
                self.molecules = molecules_hydrogenized_saved

        # Delete temporary directory
        shutil.rmtree(temp_dir)

        return molecules_hydrogenized

    def compare(self, other: typing.Self):
        """
        Compares statistics and attributes with other dataset and sends warnings when differences are found
        Args:
            other: BiomoleculeDataset to compare with

        Returns:
            None
        """

        def simple_warning(message, category, filename, lineno, line=None):
            return f"{message}\n"

        warnings_original = warnings.formatwarning
        warnings.formatwarning = simple_warning

        print(f"Starting comparison of {self.name!r} with {other.name!r}.")
        if self.molecules_ID != other.molecules_ID:
            self_mols_mismatch = set(self.molecules_ID) - set(other.molecules_ID)
            other_mols_mismatch = set(other.molecules_ID) - set(self.molecules_ID)
            warnings.warn(f"Molecules ID don't match for {self.name!r} and {other.name!r}.\n"
                          f"Molecules ID in dataset {self.name!r} not found in {other.name!r}:\n"
                          f"{self_mols_mismatch}\n"
                          f"Molecules ID in dataset {other.name!r} not found in {self.name!r}:\n"
                          f"{other_mols_mismatch}\n")

        if self.element_class != other.element_class:
            warnings.warn(f'Element class do not match:\n'
                          f'{self.name!r} element_class={self.element_class}, {other.name!r} element_class={other.element_class}')

        self_stats = self.statistics()
        other_stats = other.statistics()
        stats_common = set(self_stats) & set(other_stats)
        stats_missing_self = set(self_stats) - stats_common
        stats_missing_other = set(other_stats) - stats_common
        if stats_missing_self:
            warnings.warn(f'{self.name!r} does not have the following statistics:{stats_missing_self}')

        if stats_missing_other:
            warnings.warn(f'{other.name!r} does not have the following statistics:{stats_missing_other}')

        for stat_name in stats_common:
            stat_self = self_stats[stat_name]
            stat_other = other_stats[stat_name]

            if stat_self != stat_other:
                warnings.warn(f"Statistics {stat_name!r} does not match.")
                warnings.warn(f"{stat_name!r} stats of {self.name!r}:\n{stat_self}")
                warnings.warn(f"{stat_name!r} stats of {other.name!r}:\n{stat_other}")
            else:
                print(f"{stat_name!r} matches.")

        warnings.formatwarning = warnings_original

    def __getitem__(self, i) -> Biomolecule | list[Biomolecule]:
        if self.is_mols_loaded:
            return self.molecules[i]
        else:
            if isinstance(i, slice):
                return [self.load_mol(j) for j in range(len(self))[i]]
            else:
                return self.load_mol(i)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __add__(self, other: Self) -> Self:
        directory = dataset_tempdir / f"{self.ID}_{other.ID}"

        # Delete any previous directory
        if os.path.exists(directory):
            shutil.rmtree(directory)

        if not self.is_mols_loaded:
            self.load_all()
        if not other.is_mols_loaded:
            other.load_all()
        molecules = self.molecules + other.molecules
        return self.__class__.from_molecules(biomolecules=molecules, directory=directory)

    def __iadd__(self, other: Self | list[Biomolecule]):
        """
        Adds a set of molecules to the current dataset
        Args:
            other:
        """
        if isinstance(other, BiomoleculeDataset):
            mols_ID = other.molecules_ID
            mols = [mol for mol in other]
        else:
            mols_ID = [mol.ID for mol in other]
            mols = other

        # Check if IDs in the other dataset are found in the current dataset
        duplicate_IDs = set(mols_ID) & set(self.molecules_ID)
        if duplicate_IDs:
            raise ValueError(f"Cannot add dataset. The following filenames are duplicated:\n{duplicate_IDs}")

        # Create a backup of the all_processed filename
        processed_backup_npz_filepath = self.processed_npz_filepath.with_stem(
            self.processed_npz_filepath.stem + '_backup')
        shutil.copy(self.processed_npz_filepath, processed_backup_npz_filepath)

        # Load all molecules and append the new ones
        with np.load(self.processed_npz_filepath, allow_pickle=True) as npzfile:
            molecules_dict = dict(npzfile.items())
        new_molecules_dict = {ID: mol for ID, mol in zip(mols_ID, mols)}
        molecules_dict.update(new_molecules_dict)
        self.save_molecules(list(molecules_dict.values()), filepath=self.processed_npz_filepath)

        # Reload the process npzfile
        self.processed_npzfile = np.load(self.processed_npz_filepath, allow_pickle=True)
        self.molecules_ID = list(self.processed_npzfile.keys())
        self.molecules_ID_all = copy.copy(self.molecules_ID)

        # Delete the backup
        os.remove(processed_backup_npz_filepath)

        # Reset molecule-dependent attributes
        self._statistics = None

        # Load new molecules if previous ones were loaded.
        if self.is_mols_loaded:
            new_molecules = [self.load_mol(i) for i in range(len(self) - len(mols), len(self))]
            self.molecules += new_molecules

        return self

    def __len__(self):
        return len(self.molecules_ID)

    def __repr__(self):
        omitted_attrs = ['statistics', 'molecules', '_molecules', 'molecules_ID', 'molecules_ID_all', 'load_all']
        content_str = [f'{k}={v}' for k, v in self.__dict__.items() if k not in omitted_attrs]
        content_str = '\n'.join(content_str)
        return f'{self.__class__.__name__}({content_str})'

    def __getstate__(self):
        # Close the processed npzfile since iobuffers cannot be copied.
        self.processed_npzfile.close()
        return self.__dict__


class XTCDataset(BiomoleculeDataset):
    """
    Dataset class aggregating protein structures stored in .xtc files produced by GROMACS
    """
    metadata_attrs_name = BiomoleculeDataset.metadata_attrs_name + ['trajectories_dt', 'H_method']
    noninherited_attrs = BiomoleculeDataset.noninherited_attrs + ['trajectories_dt']

    def __init__(self, forcefield=charmm36m, terminals_gmx_code: list[str] = None,
                 element_class: str = all_atoms_class_name, H_method='pdb2gmx', **kwargs):
        """
        Args:
            forcefield: forcefield used to generate the associated .xtc files
            terminals_gmx_code: GROMACS code to determine the terminal's protonation state
            element_class: element class of the saved molecules
            H_method: method used to add the hydrogen atoms
            **kwargs: process() kwargs
        """
        self.trajectories_dt = {}  # Time step (ps) of the raw trajectories
        self.H_method = H_method
        super().__init__(forcefield=forcefield, terminals_gmx_code=terminals_gmx_code, element_class=element_class,
                         **kwargs)
        self.n_runs = len(self.trajectories_dt)

    def process(self, xtc_pattern=default_xtc_pattern, tpr_pattern=default_tpr_pattern, top_pattern=default_top_pattern,
                sampling_dt=None, shuffle_runs=False, runs_ID: list[int] = None, align=False,
                sidechain_rep_name: dict = None, n_runs: int = None):
        """
        Saves structures from .xtc trajectories produced in GROMACS to create the dataset
        Args:
            xtc_pattern: pattern of the .xtc trajectory files in the raw dir
            tpr_pattern: pattern of the .tpr file associated to the .xtc file
            top_pattern: pattern of the .top file used to define the protein's topology.
            sampling_dt: time interval (ps) between two samples.
            shuffle_runs: shuffle the order of the runs to obtain randomization at the run level
            runs_ID: IDs of runs to process
            sidechain_rep_name: dictionary giving the atom name that represents the sidechain of each residue type. Only relevant for coarse_grain=c_alpha_sidechain
            n_runs: number of runs to process from the raw directory
        """

        # Temporarily change the name of the processed file to track the processing progress.
        def_processed_npz_filename = self.processed_npz_filename
        def_processed_npz_filepath = self.processed_npz_filepath
        temp_processed_npz_filename = str(pathlib.Path(self.processed_npz_filename).with_stem('processed_temp'))
        self.processed_npz_filename = temp_processed_npz_filename
        self.processed_npz_filepath = self.dir / self.processed_npz_filename

        # Find a .top file in the raw directory using the .top pattern, if None is given
        if self.top_filepath is None:
            top_files = list(self.raw_dir.rglob('*.top'))
            top_re = re.compile(top_pattern)
            top_files = [f for f in top_files if top_re.search(str(f.name))]
            if len(top_files) > 1:
                raise FileExistsError(f"More than 1 .top file found in {str(self.raw_dir)!r}.\n"
                                      f"top_files={top_files},pattern={top_pattern!r}")
            elif len(top_files) == 0:
                raise FileNotFoundError(f"No .top file found in {str(self.raw_dir)!r}.\npattern={top_pattern!r}")
            self.top_filepath = top_files[0]

        # Create another .top file without virtual sites or water. This will be used to add hydrogen with pdb2gmx
        top_novsite_filepath = self.dir / self.top_filepath.with_stem(self.top_filepath.stem + '_novsite').name
        if not top_novsite_filepath.exists():
            if self.terminals_gmx_code is None:
                raise ValueError("No terminal GROMACS code defined. Cannot create regenerate .top file without vsites.")
            gmx_regen_top(self.top_filepath, output_filepath=top_novsite_filepath, include_vsites=False,
                          terminal_states=self.terminals_gmx_code, water='none')
        self.top_filepath = top_novsite_filepath

        # Find all the .xtc files in the raw directory
        xtc_filepaths_dict = dict()
        xtc_reg = re.compile(xtc_pattern)
        for xtc_path in self.raw_dir.rglob('*.' + xtc_pattern[-3:]):
            xtc_match = xtc_reg.search(str(xtc_path.name))
            if xtc_match is not None:
                run_ID = int(xtc_match.group('ID'))
                if runs_ID is None or run_ID in runs_ID:
                    if run_ID not in xtc_filepaths_dict:
                        xtc_filepaths_dict[run_ID] = xtc_path
                    else:
                        raise ValueError(f"Duplicate run ID found. filepath={xtc_path}, "
                                         f"ID={run_ID}, other_filepath={xtc_filepaths_dict[run_ID]}")

        # Find all the .tpr files in the raw directory
        # Check if a unique .trp file should be used for all .xtc files
        is_tpr_unique = '?P<ID>' not in tpr_pattern
        tpr_filepaths_dict = dict()
        tpr_reg = re.compile(tpr_pattern)
        if is_tpr_unique:
            tpr_paths = [p for p in self.raw_dir.rglob(f'*{tpr_pattern[-4:]}') if tpr_reg.search(p.name)]
            assert len(tpr_paths) == 1, 'topology file is nonexistent or not unique.'
            tpr_path = tpr_paths[0]

            for run_ID in xtc_filepaths_dict:
                tpr_filepaths_dict[run_ID] = tpr_path
        else:
            for tpr_path in self.raw_dir.rglob(f'*{tpr_pattern[-4:]}'):
                tpr_match = tpr_reg.search(tpr_path.name)
                if tpr_match is not None:
                    run_ID = int(tpr_match.group('ID'))
                    if runs_ID is None or run_ID in runs_ID:
                        if run_ID not in tpr_filepaths_dict:
                            tpr_filepaths_dict[run_ID] = tpr_path
                        else:
                            raise ValueError(f"Duplicate run ID found. filepath={tpr_path}, "
                                             f"ID={run_ID}, other_filepath={tpr_filepaths_dict[run_ID]}")

        # Check that all xtc have a matching tpr
        unmatched_IDs = set(xtc_filepaths_dict.keys()) ^ set(tpr_filepaths_dict.keys())
        if unmatched_IDs:
            raise ValueError(f"The following run IDs have missing .xtc or .tpr filename:\n{unmatched_IDs}")

        # Check that all requested run IDs were found.
        files_run_ID = sorted(list(xtc_filepaths_dict.keys()))
        if runs_ID:
            notfound_runs_ID = set(runs_ID) - set(files_run_ID)
            if notfound_runs_ID:
                raise ValueError(f"The following run IDs were not found:{notfound_runs_ID}")
            files_run_ID = runs_ID

        # Sort, shuffle and subset the run files
        if shuffle_runs:
            random.shuffle(files_run_ID)
        if n_runs is not None:
            files_run_ID = files_run_ID[:n_runs]
        xtc_filepaths_dict = {k: xtc_filepaths_dict[k] for k in files_run_ID}
        tpr_filepaths_dict = {k: tpr_filepaths_dict[k] for k in files_run_ID}

        # Define a dictionary that maps atom names to atom types (.tpr files created with convert-tpr have errors)
        atom_name_to_type = def_atom_name_to_type(self.top_filepath)

        # Load the universe represented by the topology file
        with warnings.catch_warnings(category=UserWarning, action="ignore"):
            top_uni = mda.Universe(self.top_filepath, topology_format='ITP', include_dir=config.data_dir)
        top_uni = top_uni.select_atoms(prot_sel_crit)

        # Load each .xtc file and .tpr with MDAnalysis.
        biomolecules = []
        if not self.processed_npz_filepath.exists():
            for run_ID in tqdm.tqdm(xtc_filepaths_dict, desc=f'Processing raw files of {self.name} dataset'):
                xtc_filepath, tpr_filepath = xtc_filepaths_dict[run_ID], tpr_filepaths_dict[run_ID]
                universe = mda.Universe(str(tpr_filepath), str(xtc_filepath), convert_units=False)
                self.trajectories_dt[run_ID] = universe.trajectory.dt

                # Select all protein atoms (no solvents or virtual sites)
                prot_sel = universe.select_atoms(prot_sel_crit)

                # Check that the residue ids of the loaded .tpr matches with the residue ids of the .top file
                if np.any(top_uni.residues.resids != prot_sel.residues.resids):
                    old_to_new_map = {id1: id2 for id1, id2 in zip(prot_sel.residues.resids, top_uni.residues.resids)}
                    prot_sel.residues.resids = np.array([old_to_new_map[ID] for ID in prot_sel.residues.resids])

                # Downsample trajectories and save each frame as a separated biomolecule.
                if sampling_dt is None:
                    downsampling_factor = 1
                else:
                    downsampling_factor = np.ceil(sampling_dt / universe.trajectory.dt).astype(int)

                for ts in universe.trajectory[downsampling_factor::downsampling_factor]:  # Skip first frame
                    # Make sure the protein is whole to avoid sudden jumps caused by periodic boundary conditions
                    prot_sel.unwrap(compound='fragments')
                    mol_ID = f"run{run_ID}_frame{ts.frame}"
                    biomolecule = Biomolecule.from_universe(md_uni=prot_sel, element_class=self.element_class,
                                                            atom_name_to_type=atom_name_to_type,
                                                            ID=mol_ID,
                                                            forcefield=self.forcefield.name,
                                                            sidechain_rep_name=sidechain_rep_name)

                    # Save the molecule and append the frame number to its filename
                    biomolecules.append(biomolecule)

            # Save the processed biomolecules in the .npz archive file
            self.save_molecules(biomolecules=biomolecules, filepath=self.processed_npz_filepath)

            # Save metadata
            self.save_metadata()
        else:
            self.load_metadata()
            with np.load(self.processed_npz_filepath, allow_pickle=True) as npzfile:
                mol_IDs = list(npzfile.keys())
                for k in tqdm.tqdm(mol_IDs, f'Loading molecules to complete process.'):
                    biomolecules.append(npzfile[k].item())

        # Add hydrogen atoms
        if self.element_class == all_atoms_class_name:
            if self.H_method == 'pdb2gmx':
                self.hydrogenize(molecules=biomolecules)
            elif self.H_method == 'MD':
                pass
            else:
                raise NotImplementedError(f"H method {self.H_method!r} is not implemented.")

        # Rename the processed .npz file to the default value to indicate that processing is complete
        shutil.move(self.processed_npz_filepath, def_processed_npz_filepath)
        self.processed_npz_filename = def_processed_npz_filename
        self.processed_npz_filepath = def_processed_npz_filepath

    def create_splits(self, ratio: list[float] = None, *args, method='random_run', **kwargs):
        """
        Defines a new method for creating splits that splits the dataset across runs.
        Args:
            ratio: ratios of the splits. Can be whole number or fraction.
            method: method used to split the molecules
        """
        if method == 'random_run':
            # When splitting across runs, use the 'index' method of the parent method to split the molecules

            # Shuffle runs
            runs_ID = np.array(self.runs_ID)
            np.random.shuffle(runs_ID)
            runs_ID = runs_ID.tolist()

            # Define a map that provides the split index of a given run index
            ratio_cumsum = np.cumsum(ratio)
            splits_end_ind = np.round(self.n_runs * ratio_cumsum / ratio_cumsum[-1]).astype(int)
            runs_split_ind = np.zeros((self.n_runs,), dtype=int)
            runs_split_ind[splits_end_ind[:-1]] = 1
            runs_split_ind = np.cumsum(runs_split_ind, axis=-1)

            mol_run_ind = np.array([runs_ID.index(ID) for ID in self.mol_run_ID])
            mol_split_ind = runs_split_ind[mol_run_ind]

            return super().create_splits(*args, method='index', index=mol_split_ind, **kwargs)
        else:
            return super().create_splits(ratio, *args, method=method, **kwargs)

    @property
    def runs_ID(self):
        runs_ID = torch.tensor(self.mol_run_ID).unique(sorted=False).tolist()
        return runs_ID

    @property
    def mol_run_ID(self):
        mols_run_ID = []
        ID_patt = re.compile(r'(?<=run)\d+')
        for mol_ID in self.molecules_ID:
            run_ID_match = ID_patt.findall(mol_ID)
            if len(run_ID_match) > 0:
                mols_run_ID.append(int(run_ID_match[0]))
            else:
                raise ValueError(f"Cannot identify run ID in {mol_ID!r}")

        return mols_run_ID

    @property
    def mol_frame_ind(self):
        mols_frame_ind = []
        reg_pat = re.compile(r'(?<=frame)\d+')
        for mol_ID in self.molecules_ID:
            frame_ind_match = reg_pat.findall(mol_ID)
            if len(frame_ind_match) == 1:
                mols_frame_ind.append(int(frame_ind_match[-1]))
            elif len(frame_ind_match) > 0:
                raise ValueError(f"More than 1 frame keyword found in mol_ID. Cannot identify frame index.")
            else:
                raise ValueError(f"Cannot identify run ID in mol_ID={mol_ID}")

        return mols_frame_ind

    def mol_pairs_ind(self, pair_dframe=None):
        molecules_run_ID = torch.tensor(self.mol_run_ID)
        molecules_frame_ind = torch.tensor(self.mol_frame_ind)
        pairs_ind = []
        for run_ID in molecules_run_ID.unique():
            mol_ind = torch.nonzero(molecules_run_ID == run_ID).squeeze()
            frame_diff = torch.diff(molecules_frame_ind[mol_ind]).unique()
            assert torch.all(frame_diff > 0), 'The frame indices must be monotonically increasing'
            assert torch.unique(frame_diff).nelement() == 1, 'Only a unique temporal difference (dt) is supported'
            if pair_dframe is None:
                pair_dframe = frame_diff

            step = int(pair_dframe / frame_diff)
            pairs_ind_temp = torch.stack([mol_ind[:-step], mol_ind[step:]], dim=1)
            pairs_ind.append(pairs_ind_temp)
        pairs_ind = torch.cat(pairs_ind, dim=0)
        return pairs_ind


class GeneratedDataset(BiomoleculeDataset):
    metadata_attrs_name = BiomoleculeDataset.metadata_attrs_name + ['configs', 'ref_dataset', 'model_dataset',
                                                                    'model_split_ID', 'pruning_crit', 'sampling_time']
    pruned_set_ID = 'PR'  # Split set ID for loading energy-pruned set

    def __init__(self, name='', directory='', configs: dict = None, ref_dataset: str = None, model_dataset: str = None,
                 model_split_ID: tuple[str, str] = None, sampling_time: float = None, **process_kwargs):
        if name and not directory:
            directory = config.gen_datasets_dir / name
        self.configs = configs
        self.ref_dataset = ref_dataset
        self.model_dataset = model_dataset
        self.model_split_ID = model_split_ID
        self.sampling_time = sampling_time
        self.pruning_crit: dict[str, MetricSigmaCriterion] | MetricSigmaCriterion = {}
        super().__init__(name=name, directory=directory, **process_kwargs)

    @staticmethod
    def generate_dir(ID: str, config_tag=None, model_epoch=None):
        if config_tag is None:
            config_tag = f"{config.ID}_{config.dataset}_{config.dataset_split_set_ID}"
        epoch_tag = '_' + model_epoch if model_epoch else ''
        dataset_name = f'Diff_{config_tag}{epoch_tag}_{ID}'
        dataset_dir = pathlib.Path(config.gen_datasets_dir, dataset_name)
        return dataset_dir

    def prune(self, method='gmx_energy', ref_dataset_name=None):
        # Split the GeneratedDataset into two parts: 'pruned' and 'not_pruned' and keep only the 'not_pruned' split
        if self.pruned_set_ID not in self.splits_sets:
            if ref_dataset_name is None:
                ref_dataset_name = self.model_dataset
            ref_dataset = load_dataset(ref_dataset_name, element_class=all_atoms_class_name)

            # Use the 'train' split of the split set that was used during training to define the energy criteria
            ref_dataset = ref_dataset.load_splits(name=self.model_split_ID[1], set_ID=self.model_split_ID[0])
            if method == 'gmx_energy':
                criterion = EnergyCriterion(dataset=self, ref_dataset=ref_dataset)
            else:
                raise NotImplementedError(f"Pruning method {method!r} is not implemented.")

            if criterion.rejection_ratio == 1.0:
                warnings.warn(f"All molecules in {self.name!r} have been pruned.")

            mol_splits_ind = criterion.is_mol_retained.int().numpy()
            self.pruning_crit[self.pruned_set_ID] = criterion
            self.create_splits(names=['0', '1'], set_ID=self.pruned_set_ID, index=mol_splits_ind, method='index')

        pruned_dataset = self.load_splits('1', set_ID=self.pruned_set_ID)
        pruned_dataset.pruning_crit = pruned_dataset.pruning_crit[self.pruned_set_ID]
        return pruned_dataset


class DenoisingTrajectoriesDataset(GeneratedDataset):
    name_prefix = 'Traj_'
    mol_ID_format = "{prefix}_{mol_ind}_{time_ind}"
    metadata_attrs_name = GeneratedDataset.metadata_attrs_name + ['noise_sch_t']  # Attributes saved as metadata

    def __init__(self, *args, noise_sch_t: torch.Tensor = None, **kwargs):
        self.noise_sch_t = noise_sch_t
        super().__init__(*args, **kwargs)

    @property
    def mol_time_indices(self):
        """
        Returns the molecule and time index for each molecule in the dataset.
        Returns:
            mol_ind = molecule index with shape=(n_mol,)
            time_ind = time index with shape=(n_mol,)
        """

        time_ind = torch.zeros(len(self.molecules_ID), dtype=torch.long)
        mol_ind = torch.zeros(len(self.molecules_ID), dtype=torch.long)
        # reg_pat = re.compile(r'(?<=_)\d+')
        reg_pat = re.compile(r'_(?P<mol_ind>\d+)_(?P<time_ind>\d+)')
        for i, filename in enumerate(self.molecules_ID):
            reg_result = reg_pat.search(filename)
            mol_ind[i] = int(reg_result.group('mol_ind'))
            time_ind[i] = int(reg_result.group('time_ind'))
        return mol_ind, time_ind

    @property
    def mol_time_to_data_ind(self):
        """
        Returns a 2D array that maps the molecule and time index of the molecules to the dataset index
        Returns:
            mol_ind_1D shape=(n_mol,n_t) where mol_ind_1D[i,j] gives the 1D index of the ith molecule at the jth time.
        """

        mol_ind, time_ind = self.mol_time_indices
        n_mols, n_t = int(mol_ind.max() + 1), int(time_ind.max() + 1)
        mol_ind_1D = torch.zeros((n_mols, n_t), dtype=torch.long)
        mol_ind_1D[mol_ind, time_ind] = torch.arange(n_mols * n_t)
        return mol_ind_1D

    @classmethod
    def format_mol_ID(cls, prefix: str, mol_ind: int, time_ind: int):
        return cls.mol_ID_format.format(prefix=prefix, mol_ind=mol_ind, time_ind=time_ind)

    @classmethod
    def generate_dir(cls, *args, **kwargs):
        default_dir = super().generate_dir(*args, **kwargs)
        directory = default_dir.with_stem(cls.name_prefix + default_dir.stem)
        return directory

    def to_xtc(self, ind: list[int] = None):
        """
            Saves denoising trajectories in .xtc files
            Args:
                ind: indices of saved molecules. Defaults to range(n_mols).
        """
        xtc_dir = self.dir / 'xtc'
        xtc_dir.mkdir(parents=True, exist_ok=True)
        elem_class = heavy_atoms_class_name if self.element_class == c_alpha_sidechain_class_name else self.element_class
        uni = uni_from_top(top_filepath=self.top_filepath, element_class=elem_class, trajectory=True)

        mol_ind_1D = self.mol_time_to_data_ind
        n_mols, n_t = mol_ind_1D.shape
        mol_ex = self[0]

        # Select the atoms whose positions will be written to .xtc
        elements_resid = mol_ex.elements_resid
        selected_atoms_ID = [[resid.item(), name] for resid, name in zip(elements_resid, mol_ex.elements_name)]
        uni_atoms_ID = [[a.resid, a.name] for a in uni.atoms]
        atom_is_sel = torch.tensor([atom_ID in selected_atoms_ID for atom_ID in uni_atoms_ID])
        atoms = uni.atoms[atom_is_sel]

        reader_kwargs = dict(order='fac')
        if ind is None:
            ind = range(n_mols)
        for i in ind:
            positions_traj = torch.stack([self[j].elements_position for j in mol_ind_1D[i]])
            positions_traj = self[0].convert_length(positions_traj, pdb_length_units)
            if self.element_class == c_alpha_sidechain_class_name:
                # Add placeholder positions for atoms that do not belong to the element class
                positions_traj_element_class = positions_traj
                positions_traj = torch.zeros((n_t, uni.atoms.n_atoms, 3))
                positions_traj[:, atom_is_sel, :] = positions_traj_element_class
            uni.load_new(positions_traj.numpy(), format=mda.coordinates.memory.MemoryReader, **reader_kwargs)
            xtc_filepath = xtc_dir / f"denoising_traj_{i}.xtc"
            with mda.Writer(str(xtc_filepath), atoms.n_atoms) as W:
                for ts in uni.trajectory:
                    W.write(atoms)

        # Save a .pdb template file to load the .xtc
        pdb_filename = self.top_filepath.stem + f'_{self.element_class}.pdb'
        pdb_filepath = xtc_dir / pdb_filename
        with mda.coordinates.PDB.PDBWriter(filename=str(pdb_filepath), convert_units=False) as pdb:
            pdb.write(atoms)


class PDBEnsDataset(BiomoleculeDataset):
    """
    Ensemble of PDB structures
    """

    def __init__(self, forcefield=charmm36m, terminals_gmx_code=None, **kwargs):
        if terminals_gmx_code is None:
            terminals_gmx_code = ['0', '0']
        super().__init__(forcefield=forcefield, terminals_gmx_code=terminals_gmx_code, **kwargs)

    def process(self, save_top_file=True, **kwargs):
        # Directories
        temp_dir = Path(self.dir, 'pdb2gmx_temp')
        state_filepath = temp_dir / 'process_state.npz'
        if state_filepath.exists():
            with np.load(state_filepath, allow_pickle=True) as file:
                process_state = dict(file)
            process_state['omitted_pdbs'] = process_state['omitted_pdbs'].tolist()
        else:
            process_state = dict(omitted_pdbs=[])
        curr_dir = os.getcwd()
        prot_pdb_dir = temp_dir / 'pdb_prot'
        prot_pdb_dir.mkdir(parents=True, exist_ok=True)

        # Find all .pdb files in the raw directory
        n = 100 if self.debug else np.inf
        pdb_files = {p.stem: p for i, p in enumerate(self.raw_dir.rglob("*.pdb")) if i < n}

        # Fix the case of pdb IDs to have unique IDs.
        pdb_files = {k[:4].upper() + k[4:]: v for k, v in pdb_files.items()}

        if not pdb_files:
            raise ValueError(f"No .pdb files found in raw directory:{self.raw_dir}")

        # Extract protein structure from raw .pdb
        omitted_pdbs = process_state['omitted_pdbs']
        canonical_amino_acids = set(canonical_aa_codes_3)

        pdb_prot_files = {p.stem: p for i, p in enumerate(prot_pdb_dir.rglob("*.pdb"))}
        pdb_files = {k: v for k, v in pdb_files.items() if k not in pdb_prot_files and k not in omitted_pdbs}
        pdb_files: dict[str, Path]

        if pdb_files:
            for pdb_stem, pdb_file in tqdm.tqdm(pdb_files.items(), f'Building {self.name!r} - Filtering .pdb files'):
                uni = mda.Universe(pdb_file, in_memory=True)
                prot_sel = uni.select_atoms('protein')

                # Filter out .pdb files that contain non-standard residues
                non_standard_residues = set(prot_sel.resnames) - set(canonical_amino_acids)
                if non_standard_residues:
                    warnings.warn(f"{pdb_stem!r} has non-standard residues. Skipping this PDB structure.")
                    omitted_pdbs.append(pdb_stem)
                    continue

                # Remove duplicate residue IDs
                dup_resid = find_duplicates(prot_sel.residues.resids)
                if dup_resid:
                    removed_resinds = np.array([a for k, v in dup_resid.items() for a in v[1:]])
                    ind_map = {r: i for i, r in enumerate(prot_sel.residues)}
                    resindices = np.array([ind_map[a.residue] for a in prot_sel.atoms])
                    # resindices = prot_sel.atoms.resindices # This gives the global residue index. Can't use
                    atoms_kept_ind = ~np.isin(resindices, removed_resinds)
                    prot_sel = prot_sel.atoms[atoms_kept_ind]

                    # Re-check duplication of ID
                    dup_resid = find_duplicates(prot_sel.residues.resids)
                    if dup_resid:
                        raise ValueError(f"Residues with duplicate IDs are still present.")

                # Remove PDBs that have non-monotonically increasing residue ID.
                if np.any(np.diff(prot_sel.resids) < 0):
                    warnings.warn(f"{pdb_stem!r} has non-monotonic residue IDs. Skipping this PDB structure.")
                    omitted_pdbs.append(pdb_stem)
                    continue

                # Remove residue icodes
                res = prot_sel.residues
                if hasattr(res, 'icodes'):
                    res.icodes = np.array(prot_sel.n_residues * [''])

                # Save protein-only .pdb
                pdb_pruned_file = prot_pdb_dir / pdb_file.name

                # Ensure the pdb ID is capitalized in the prot .pdb file
                pdb_stem = pdb_pruned_file.stem
                pdb_ID = pdb_stem[:4].upper()
                pdb_pruned_file = pdb_pruned_file.with_stem(pdb_ID + pdb_stem[4:])

                with mda.Writer(str(pdb_pruned_file)) as pdb:
                    pdb.write(prot_sel)

            process_state['omitted_pdbs'] = omitted_pdbs
            np.savez(state_filepath, **process_state)
        pdb_prot_files = {p.stem: p for i, p in enumerate(Path(prot_pdb_dir).rglob("*.pdb"))}

        # Add hydrogen to the .pdb files
        gro_dir = temp_dir / 'gro'
        gro_dir.mkdir(exist_ok=True)
        top_existent = {p.stem: p for p in Path(temp_dir).rglob('*.top')}
        pdb_prot_filepaths_remaining = [v for k, v in pdb_prot_files.items() if k not in top_existent]

        if pdb_prot_filepaths_remaining:
            pdb2gmx_kwargs = dict(lb=0.0, working_dir=gro_dir, forcefield_dir=self.forcefield.dir, ignh=True,
                                  ter=None, terminals_protonation=self.terminals_gmx_code)
            addH_to_pdb(pdb_prot_filepaths_remaining, parallel=False, raise_if_error=False,
                        tqdm_desc=f'Building {self.name!r} - Adding hydrogen', **pdb2gmx_kwargs)

        # Import .gro files and initialize them into Biomolecule objects
        gro_existent = {p.stem: p for p in Path(gro_dir).rglob('*.gro')}

        # Define worker function that loads a .gro file and initializes the associated Biomolecule
        worker_args = list(gro_existent.values())
        worker_kwargs = dict(forcefield=self.forcefield.name, element_class=self.element_class, save_bonds=True,
                             save_types=False)
        worker_func = functools.partial(Biomolecule.from_gro, **worker_kwargs)
        parallel = True
        if parallel:
            pool = multiprocessing.Pool()
            tasks_iter = pool.imap_unordered(worker_func, worker_args)
        else:
            tasks_iter = map(worker_func, worker_args)

        tasks_iter = tqdm.tqdm(tasks_iter, total=len(gro_existent), desc=f'Building {self.name!r} - Reading .gro file')
        biomolecules = {}
        for biomol in tasks_iter:
            # Copy the molecule to avoid memory leaks. Otherwise, workers run out of memory since main thread still
            # references biomol
            biomol_copy = copy.deepcopy(biomol)
            biomolecules[biomol_copy.ID] = biomol_copy
            del biomol

        if parallel:
            pool.close()
            pool.join()
        biomolecules = [biomolecules[f] for f in sorted(biomolecules.keys())]
        self.save_molecules(biomolecules=biomolecules, filepath=self.processed_npz_filepath)

        # Copy one of the generated .top files to define the topology file of the ensemble.
        # All generated .top files should represent the same sequence.
        if save_top_file:
            top_existent = {p.stem: p for p in Path(gro_dir).rglob('*.top')}
            top_template_filepath = top_existent[sorted(top_existent.keys())[0]]
            self.top_filepath = self.dir / f"{self.name}.top"
            shutil.copy(top_template_filepath, self.top_filepath)
            gmx_anonymize(self.top_filepath, self.top_filepath)

        # Save metadata
        self.save_metadata()

        # Test loading of all molecules
        with np.load(self.processed_npz_filepath, allow_pickle=True) as npzfile:
            for mol_name, mol in tqdm.tqdm(npzfile.items(), desc=f"Testing molecule loading for {self.name!r}"):
                pass

        # Delete temp folder
        shutil.rmtree(temp_dir)
        os.chdir(curr_dir)


class RichDataset(PDBEnsDataset):
    """
    Richardson 2018 dataset of high-quality PDB structures
    Reference: "High quality protein residues: Top2018 mainchain-filtered residues"
                https://zenodo.org/records/5777651
    """

    def __init__(self, **kwargs):
        super().__init__(save_top_file=False, **kwargs)


class Nup98Dataset(XTCDataset):
    def __init__(self, name, forcefield=charmm36m, terminals_gmx_code=None, **kwargs):
        if terminals_gmx_code is None:
            terminals_gmx_code = ['1', '1']

        super().__init__(name=name, forcefield=forcefield, terminals_gmx_code=terminals_gmx_code, **kwargs)


class Nup98ADataset(Nup98Dataset):
    def __init__(self, name, forcefield=amber14sb_OL15, terminals_gmx_code=None, **kwargs):
        if terminals_gmx_code is None:
            terminals_gmx_code = ['1', '1']
        super().__init__(name=name, forcefield=forcefield, terminals_gmx_code=terminals_gmx_code,
                         **kwargs)


class RSDataset(XTCDataset):
    def __init__(self, name, forcefield=charmm36m, terminals_gmx_code=None, **kwargs):
        if terminals_gmx_code is None:
            terminals_gmx_code = ['0', '0']

        super().__init__(name=name, forcefield=forcefield, terminals_gmx_code=terminals_gmx_code, **kwargs)


class ChigDDataset(XTCDataset):
    def __init__(self, name, forcefield=charmm22star, terminals_gmx_code=None, **kwargs):
        if terminals_gmx_code is None:
            terminals_gmx_code = ['0', '0']

        super().__init__(name=name, forcefield=forcefield, terminals_gmx_code=terminals_gmx_code, **kwargs)

    def process(self, xtc_pattern='', tpr_pattern='', *args, **kwargs):
        if not xtc_pattern:
            xtc_pattern = r"CLN025-0-protein-(?P<ID>\d+)\.xtc"
        if not tpr_pattern:
            tpr_pattern = r"CLN025\.tpr"

        # Convert .dcd files to .xtc format. This is done to fix time issues with MDAnalysis and .dcd files.
        gmx_traj_dir = self.raw_dir / 'CLN025-0-protein-gmx'
        dcd_dir = self.raw_dir / 'CLN025-0-protein'
        gro_file = gmx_traj_dir / 'CLN025.gro'
        pdb_file = gmx_traj_dir / 'CLN025-0-protein.pdb'
        dt = 200  # time interval in picoseconds. See 'CLN025-0-protein_times.csv' in raw dcd_dir.

        # The ordering of atoms in the .dcd files differ from the .gro files. To allow for easy importing of .xtc files
        # using the .tpr file derived from the .gro file, atom coordinates from the .dcd file need to be reordered.
        # Determine the new atom ordering by matching atom names and residue ID.
        pdb_uni = mda.Universe(pdb_file)
        gro_uni = mda.Universe(gro_file)

        # Map pdb atom name to GROMACS atom name
        atom_renaming_map = {('HT1', 1): ('H1', 1), ('HT2', 1): ('H2', 1), ('HT3', 1): ('H3', 1)}
        renaming_func = lambda i: atom_renaming_map[i] if i in atom_renaming_map else i
        pdb_atom_id = [renaming_func((a.name, a.resid)) for a in pdb_uni.atoms]
        sorting_ind = np.array([pdb_atom_id.index((a.name, a.resid)) for a in gro_uni.atoms])
        assert len(np.unique(sorting_ind)) == len(sorting_ind), 'Some atoms are not mapped'

        # Gather all .dcd files. Skip the last trajectory since it has a different length than the others
        dcd_filepaths = [p for p in Path(dcd_dir).rglob('*.dcd') if not str(p).endswith('053.dcd')]
        xtc_filepaths = [gmx_traj_dir / p.with_suffix('.xtc').name for p in dcd_filepaths]
        fpaths = {xtc_fp: dcd_fp for xtc_fp, dcd_fp in zip(xtc_filepaths, dcd_filepaths) if not xtc_fp.exists()}
        if fpaths:
            for xtc_fp, dcd_fp in tqdm.tqdm(fpaths.items(), f'Converting .dcd files to .xtc for {self.name!r}'):
                uni = mda.Universe(pdb_file, str(dcd_fp))
                with mda.coordinates.XTC.XTCWriter(str(xtc_fp), n_atoms=uni.atoms.n_atoms) as writer:
                    for i, ts in enumerate(uni.trajectory):
                        ts.time = i * dt
                        writer.write(uni.atoms[sorting_ind])

        super().process(*args, xtc_pattern=xtc_pattern, tpr_pattern=tpr_pattern, **kwargs)

    def calculate_mols_fold_state(self):
        from analysis.metrics import calculate_chig_fold_state

        self_copy: typing.Self = load_dataset(self.name)  # Load new dataset to not interfere with self
        mols_fold_state, mols_rmsd_min_2rvd = calculate_chig_fold_state(self_copy)
        return mols_fold_state, mols_rmsd_min_2rvd

    def create_splits(self, ratio: list[float] = None, set_ID: str = None, **kwargs):
        if 'method' in kwargs and (kwargs['method'].endswith('InTest') or kwargs['method'].endswith('InTrain')):
            method = kwargs.pop('method')
            assert ratio is not None, f"Splits ratio must be given when method={method!r}."
            assert set_ID is not None, f"Splits_set_ID=None is invalid when method={method!r}."

            # Split the molecules according to the given splitting method
            method_patt = rf"(Folded|Unfolded)(HA|BB)_(\d+\.?\d*).*"
            patt_match = re.match(method_patt, method)
            if patt_match:
                fold_state, struct, cutoff = patt_match.groups()
                cutoff = float(cutoff)

                # Calculate the RMSD of each molecule compared to the 2RVD ensemble
                _, mols_rmsd_min_2rvd = self.calculate_mols_fold_state()

                # Round the RMSD values in angstroms to 3 decimals to avoid floating point error.
                mols_rmsd_min_2rvd = np.round(mols_rmsd_min_2rvd, decimals=3)
                if fold_state == 'Folded':
                    primary_ind = mols_rmsd_min_2rvd <= cutoff
                elif fold_state == 'Unfolded':
                    primary_ind = mols_rmsd_min_2rvd > cutoff
                else:
                    raise NotImplementedError(f"Fold state {fold_state!r} is not implemented.")
            else:
                raise ValueError(f"method {method!r} is not supported.")

            if method.endswith('InTest'):
                # Include all primary state in the test split and divide the rest between the train and valid splits
                test_ind = primary_ind
                train_valid_ind = np.random.permutation(np.nonzero(~test_ind))
                n_train_mol = np.round(ratio[0] / (ratio[0] + ratio[1]) * train_valid_ind.size).astype(int)
                train_ind, valid_ind = train_valid_ind[:n_train_mol], train_valid_ind[n_train_mol:]
            elif method.endswith('InTrain'):
                # Include all primary state in the train split and divide the rest between the test and valid splits
                train_ind = primary_ind
                valid_test_ind = np.random.permutation(np.nonzero(~train_ind))
                n_valid_mol = np.round(ratio[1] / (ratio[1] + ratio[2]) * valid_test_ind.size).astype(int)
                valid_ind, test_ind = valid_test_ind[:n_valid_mol], valid_test_ind[n_valid_mol:]
            else:
                raise ValueError(f"method {method!r} is not supported.")

            split_index = -np.ones(len(self), dtype=int)
            split_index[train_ind] = 0
            split_index[valid_ind] = 1
            split_index[test_ind] = 2
            n_unassigned_index = (split_index < 0).sum()
            if n_unassigned_index > 0:
                raise ValueError(f"{n_unassigned_index} molecules were not assigned a split index in {self.name!r}")

            return super().create_splits(method='index', index=split_index, set_ID=set_ID, **kwargs)
        else:
            return super().create_splits(ratio=ratio, set_ID=set_ID, **kwargs)

    def prepare_for_publication(self):
        """
        Nans all saved molecules' atom positions to avoid publishing proprietary data
        """
        mol_ex = self.load_mol(0)
        if mol_ex.elements_position.isnan().all():
            return

        self.load_all()
        for mol in tqdm.tqdm(self.molecules, f'Overwriting positions of {self.name!r}'):
            mol.elements_position = torch.nan * mol.elements_position
        self.save_molecules(self.molecules, directory=self.dir, overwrite=True)


class AAQAADataset(XTCDataset):
    def __init__(self, name, forcefield=amber14sb_OL15, terminals_gmx_code=None, **kwargs):
        if terminals_gmx_code is None:
            terminals_gmx_code = ['1', '1']

        super().__init__(name=name, forcefield=forcefield, terminals_gmx_code=terminals_gmx_code, **kwargs)


class ProcessedDataset(Dataset):
    def __init__(self, dataset: BiomoleculeDataset, model: 'DiffPGNN', inMemory: bool = None, device=None):
        self.dataset = dataset
        self.model_gather_tensors = model.gather_input_tensors
        self.pos_scale = model.pos_scale.item()
        self.device = None if device in [None, 'cpu'] else device
        self.inMemory = self.dataset.is_mols_loaded if inMemory is None else inMemory

        # Initialize features that are constant across all molecules
        self.const_feat = model.gather_constant_input_tensors(dataset[0])

        if self.inMemory:
            self.dataset.load_all()
            self.tensors_dict = self.gather_tensors(self.dataset.molecules)
            self.move_tensors_to_device(self.tensors_dict)
        else:
            self.file_handle = None  # Initialized later by each worker

    def gather_tensors(self, molecules: list[Biomolecule]):
        tensors_dict = self.model_gather_tensors(molecules, pos_scale=self.pos_scale, verbose=self.inMemory)

        # Add constant features
        for k, v in self.const_feat.items():
            if isinstance(v, torch.Tensor):
                tensors_dict[k] = v.expand(len(molecules), *(v.ndim - 1) * [-1])
            else:
                raise NotImplementedError

        return tensors_dict

    def __getitems__(self, items: list[int]):
        if self.inMemory:
            tensors_dict_out = dict()
            items = torch.tensor(items, device=self.device)

            for name, T in self.tensors_dict.items():
                if isinstance(T, torch.Tensor | np.ndarray):
                    # Slice the batch dimension if the tensor or ndarray has 3 or more dimensions.
                    if T.ndim > 2:
                        tensors_dict_out[name] = T[items, ...]
                    else:
                        tensors_dict_out[name] = T
                elif T is None:
                    tensors_dict_out[name] = T
                else:
                    # Copy other info that are constants
                    tensors_dict_out[name] = torch.as_tensor(T)
        else:
            if self.file_handle is None:
                self.file_handle = np.load(self.dataset.processed_npz_filepath, allow_pickle=True, mmap_mode='r')

            molecules = [self.dataset.load_mol(i, file_handle=self.file_handle) for i in items]
            tensors_dict_out = self.gather_tensors(molecules)
            for k, v in tensors_dict_out.items():
                if isinstance(v, torch.Tensor):
                    tensors_dict_out[k] = v.clone()

        # Add batch size
        tensors_dict_out['batch_size'] = len(items)
        return tensors_dict_out

    def __len__(self):
        return len(self.dataset)

    def move_tensors_to_device(self, tensors_dict: dict = None):
        if self.device is None:
            return

        if tensors_dict is None:
            if self.inMemory:
                tensors_dict = self.tensors_dict
            else:
                raise ValueError(f"No tensors given")

        for name, T in tensors_dict.items():
            if isinstance(T, torch.Tensor):
                tensors_dict[name] = T.to(self.device)


class Criterion:
    """
    Criterion to select a subset of a BiomoleculeDataset
    """

    def __call__(self, mol: Biomolecule) -> torch.BoolTensor | bool:
        raise NotImplementedError


class MetricSigmaCriterion(Criterion):
    def __init__(self, dataset: BiomoleculeDataset, ref_dataset: BiomoleculeDataset, metrics: list[str],
                 sig_thresh: float = 5.0):
        self.metrics = metrics
        self.sig_thresh = sig_thresh
        self.dataset_name = dataset.name
        self.ref_dataset_name = ref_dataset.name

        # Gather metrics of the dataset and the reference dataset
        from analysis.metrics import gather_mol_properties
        dataset_metrics = gather_mol_properties(dataset, properties_name=metrics)
        ref_dataset_metrics_stats = ref_dataset.statistics(metrics)

        metrics_mean = torch.tensor([ref_dataset_metrics_stats[n].mean for n in metrics])
        metrics_std = torch.tensor([ref_dataset_metrics_stats[n].std for n in metrics])
        metrics_values = torch.stack([dataset_metrics[n] for n in metrics], dim=1)
        metrics_z_score = torch.abs(metrics_values - metrics_mean) / metrics_std

        # Keep the molecule if all metrics value falls inside the given sigma threshold
        is_mol_retained = torch.all(metrics_z_score <= self.sig_thresh, dim=-1)
        self.is_mol_retained = is_mol_retained
        self.is_mol_retained_dict = {mol.ID: self.is_mol_retained[i].item() for i, mol in enumerate(dataset)}
        self.rejection_ratio = 1 - self.is_mol_retained.sum() / self.is_mol_retained.numel()

    def __call__(self, mol: Biomolecule) -> torch.BoolTensor | bool:
        return self.is_mol_retained_dict[mol.ID]

    def __repr__(self):
        return (f'{self.__class__.__name__}(metrics={self.metrics}, '
                f'sig_thresh={self.sig_thresh}, '
                f'rejection_ratio={self.rejection_ratio:.3f})')


class EnergyCriterion(MetricSigmaCriterion):
    def __init__(self, dataset: BiomoleculeDataset, ref_dataset: BiomoleculeDataset, **kwargs):
        metrics = list(dataset.forcefield.gmx_energy_names.keys())

        # Check that the ordering of the energy entries matches the ordering
        # of the molecule in the dataset.
        from analysis.metrics import calculate_gmx_energy
        gmx_energy_dict, _ = calculate_gmx_energy(dataset)
        if not np.all(gmx_energy_dict['ID'] == np.array(dataset.molecules_ID)):
            raise ValueError('Ordering of the gmx energies do not match the dataset ordering.')

        super().__init__(dataset=dataset, ref_dataset=ref_dataset, metrics=metrics, **kwargs)


def load_dataset(name: str = None, directory: str | pathlib.Path = None, fetch_samples=True, force_download=False,
                 **kwargs) -> BiomoleculeDataset | GeneratedDataset | DenoisingTrajectoriesDataset:
    """
    Loads a dataset by name or directory
    Args:
        name: Name of dataset. If no name is given, defaults to pathlib.Path(directory).name.
              If no directory is given, it defaults to config.dataset.
        directory: full directory of dataset. If no directory is given, a directory with name 'name' must be located
                   in config.dataset_dir or config.gen_datasets_dir
        fetch_samples: fetches given dataset from remotes if not found in local directories.
        force_download: force download from remote location
        **kwargs: other kwargs passed to dataset class constructor

    Returns:
        dataset object
    """
    if directory:
        directory = Path(directory)
        name = name if name else directory.stem
    elif not name:
        if config.dataset:
            name = config.dataset
        else:
            raise ValueError(f"Cannot load default dataset. Name is not set.")

    is_pruned_samples = False
    if name.endswith('temp'):
        dataset_cls = BiomoleculeDataset
    elif name.startswith('nup98A'):
        dataset_cls = Nup98ADataset
    elif name.startswith('nup98') or name.startswith('Nup98'):
        dataset_cls = Nup98Dataset
    elif name.startswith('RS'):
        dataset_cls = RSDataset
    elif name.startswith('ChigD') or name.startswith('CLN'):
        dataset_cls = ChigDDataset
    elif name.startswith('AAQAA'):
        dataset_cls = AAQAADataset
    elif name.startswith(DenoisingTrajectoriesDataset.name_prefix):
        dataset_cls = DenoisingTrajectoriesDataset
    elif name.startswith('Diff') or pathlib.Path(config.gen_datasets_dir, name).is_dir():
        dataset_cls = GeneratedDataset

        # Check if the dataset is pruned
        pruned_suffix = f"_{GeneratedDataset.pruned_set_ID}_1"
        if name.endswith(pruned_suffix):
            name = name.removesuffix(pruned_suffix)
            is_pruned_samples = True
    elif name.startswith('Rich_2018'):
        dataset_cls = RichDataset
    elif name.startswith('PDB'):
        dataset_cls = PDBEnsDataset
    else:
        raise ValueError(f"Dataset class for {name!r} is not configured.")

    # Define the dataset directory based on the inferred class
    if not directory:
        if issubclass(dataset_cls, GeneratedDataset):
            directory = pathlib.Path(config.gen_datasets_dir, name)
        else:
            directory = pathlib.Path(config.proc_datasets_dir, name)

    # Fetch dataset from repository if it does not exist locally
    if issubclass(dataset_cls, GeneratedDataset):
        dataset_type = 'generated'
    elif issubclass(dataset_cls, XTCDataset):
        dataset_type = 'processed'
    else:
        dataset_type = None
    if fetch_samples and dataset_type is not None and (not directory.exists() or force_download):
        download_dataset(name, dataset_type)

    # Check if the dataset exists after attempting to download it
    if not directory.exists():
        raise FileNotFoundError(f'dataset {name!r} does not exist in {str(directory.parent)!r}.')

    dataset = dataset_cls(name=name, directory=directory, **kwargs)
    if is_pruned_samples:
        dataset = dataset.prune()

    return dataset


def compare_mols_rmsd(dataset1: BiomoleculeDataset, dataset2: BiomoleculeDataset):
    print(f"Checking molecules rmsd between {dataset2.name!r} and {dataset1.name!r}")
    if dataset1.molecules_ID != dataset2.molecules_ID:
        dataset1_mols_mismatch = set(dataset1.molecules_ID) - set(dataset2.molecules_ID)
        dataset2_mols_mismatch = set(dataset2.molecules_ID) - set(dataset1.molecules_ID)
        warnings.warn(f"Molecules ID don't match for {dataset1.name!r} and {dataset2.name!r}.\n"
                      f"Molecules ID in dataset {dataset1.name!r} not found in {dataset2.name!r}:\n"
                      f"{dataset1_mols_mismatch}\n"
                      f"Molecules ID in dataset {dataset2.name!r} not found in {dataset1.name!r}:\n"
                      f"{dataset2_mols_mismatch}\n")
        return

    dataset1.load_all()
    dataset2.load_all()
    mol_pair_iter = zip(dataset1.molecules, dataset2.molecules)
    mol_pair_iter = tqdm.tqdm(mol_pair_iter, desc=f'Checking RMSD between {dataset1.name!r} and {dataset2.name!r}',
                              total=len(dataset1))
    n_mismatches = 0
    for mol_new, mol_old in mol_pair_iter:
        mol_new: Biomolecule
        mol_old: Biomolecule
        pair_rmsd = mol_new.rmsd(mol_old)
        if pair_rmsd.abs() > 1e-6:
            warnings.warn(f"RMSD is nonzero {pair_rmsd} for mol {mol_new.ID!r}.")
            n_mismatches += 1

    if n_mismatches == 0:
        print('All molecules position match.')
    else:
        print(f'{n_mismatches} mismatches found.')


if __name__ == "__main__":
    # Test fetching of dataset
    dataset = load_dataset('nup98_12', force_download=True)

    # # Test ProcessedDataset
    # config.import_configs('config_test.py')
    # import models
    #
    # dataset = load_dataset()
    # dataset_train = dataset.load_splits('train')
    # dataset_train.load_all()
    # model = models.initialize(dataset_train=dataset_train)
    # dataset_train_proc = ProcessedDataset(dataset_train, model=model)
    # tensors = dataset_train_proc.__getitems__([0, 2, 3, 1])
    # pass

    # Test RichDataset
    # rich_dataset = RichDataset(name='Rich_2018')
    # coarse_grains.calculate_packings(rich_dataset.name, 'sidechain_heavy')
    # cg = coarse_grains[(rich_dataset.name, 'sidechain_heavy')]
    # exit()
    # stats = rich_dataset.statistics()
    # pass

    # # Test dataset refinement
    # nup98_dataset = Nup98Dataset('nup98_12mer_CA_3').load_splits('train')
    # nup98_dataset_ref = nup98_dataset.refine(coarse_grain=heavy_atoms_class_name)
    # pass

    # Testing datasets
    # chigD_dataset_debug = ChigDDataset('ChigD_HA_debug', coarse_grain=heavy_atoms_class_name,
    #                                    sampling_dt=20000, n_runs=3, reprocess=True, runs_ID=[1, 2, 3])
    nup98_dataset_debug = Nup98Dataset('nup98_12mer_debug', directory=dataset_tempdir / 'nup98_12mer_debug',
                                       element_class=all_atoms_class_name,
                                       sampling_dt=20000, reprocess=True, runs_ID=[3, 5, 9, 6, 4, 7],
                                       align=True, raw_dir=Path(config.MD_sim_dir, 'nup98_12'))
    nup98_dataset_debug.create_splits(ratio=[1, 1, 1], method='random_run')
    pass

    # Test the generation of a random dataset
    dataset_random = BiomoleculeDataset.random(n_molecules=100)
    dataset_random.load_all()

    # Test splitting a BiomoleculeDataset
    # set_rng(1)
    # dataset_train, dataset_valid, dataset_test = dataset_random.split(ratio=[0.8, 0.1, 0.1], save=True, names=['train', 'valid', 'test'])
    # dataset_train, dataset_valid, dataset_test = dataset_random.split(ratio=[0.8, 0.1, 0.1], resave=False,
    #                                                                   names=['train', 'valid', 'test'], method='fixed')
    # dataset_train, dataset_valid, dataset_test = dataset_random.split(ratio=[0.8, 0.1, 0.1])
    dataset_train, dataset_valid, dataset_test = dataset_random.create_splits(ratio=[0.8, 0.1, 0.1], save=False,
                                                                              names=['train', 'valid', 'test'])
    pass
