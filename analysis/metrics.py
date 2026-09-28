import copy
import itertools
import os
import pathlib
import re
import shutil
import warnings
import math
import inspect
import functools
from typing import Iterable
from collections import defaultdict
import multiprocessing

import numpy as np
from numpy.typing import NDArray
import scipy.sparse
import sklearn.base
import torch
import matplotlib.pyplot as plt
import tqdm
import MDAnalysis as mda
from MDAnalysis.coordinates.PDB import PDBWriter
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from chemistry.grains import coarse_grains_sets, calculate_backbone_packings, supported_adj_methods
from chemistry.energy import BondEnergy, BondAngleEnergy, DihedralAngleEnergy, LJEnergy, CMAPEnergy, \
    single_point_en_params_filepath
from chemistry.chem_utils import DIHEDRAL_STATE_REGIONS
from data.forcefields import gmx_energy_names, gmx_derived_energy_names, charmm_gmx_energy_names, amber_gmx_energy_names
from data.data_classes import Biomolecule, c_alpha_class_name, c_alpha_sidechain_class_name, heavy_atoms_class_name, \
    all_atoms_class_name, uni_from_top, element_classes
from data.datasets import BiomoleculeDataset, GeneratedDataset, XTCDataset, ChigDDataset, load_dataset
from analysis.stats import Statistics, J_dist_mvn, PCA_proj_matrix, JS_div
from utils import remove_centroid, parse_gmx_mdp, gmx_anonymize, gmx_mdrun, pint_reg, GMX_LENGTH_units, delayed_delete, \
    rmsd, n_parallel_processes, remove_rot, DictCheckpointer, calculate_units_conversion_factor, cast_to_optimal_dtype

import config

# Biomolecule properties
Biomolecule_prop_names = ['radial_2nd_mom', 'radial_2nd_mom_CA', 'R_g', 'end_to_end_dist', 'elements_pdist',
                          'elements_position_local', 'CA_pdist', 'backbone_dihedrals', 'dihedral_omega',
                          'backbone_bond_lengths', 'backbone_bond_angles', 'chirality', 'n_elements',
                          'elements_position']
Biomolecule_pair_prop_names = ['rmsd', 'rmsd_min_train', 'rmsd_min_2rvd']
Biomolecule_gmx_prop_names = list(gmx_energy_names)
Biomolecule_adj_prop_names = [f"elements_position_{m}_local" for m in ['CA_SI']]
Biomolecule_energy_prop_names = ['BondEnergy_tot', 'BondAngleEnergy_tot', 'DihedralEnergy_tot', 'LJEnergy_tot',
                                 'CMAPEnergy_tot', 'BondEnergy']
Biomolecule_UMAP_prop_names = ['elements_position_UMAP', 'dihedral_angle_UMAP']
Biomolecule_Z_prop_names = ['bond_length_Z', 'bond_angle_Z']
Biomolecule_misc_prop_names = ['LJ_pdist_all', 'LJ_pdist_diff_all', 'h_bond_map', 'pdist_adj_CA_SI_0.2']
Biomolecule_misc_prop_names += Biomolecule_UMAP_prop_names + Biomolecule_Z_prop_names
Biomolecule_misc_prop_names += ['bond_length', 'bond_angle', 'dihedral_angle', 'bond_angle_enddist',
                                'backbone_sphere_radius', 'elements_dist_CA_SI']
Biomolecule_misc_prop_names += Biomolecule_adj_prop_names

# Define the defined and default molecule properties for each element class
all_defined_props = (Biomolecule_prop_names + Biomolecule_gmx_prop_names + Biomolecule_pair_prop_names
                     + Biomolecule_energy_prop_names + Biomolecule_misc_prop_names)
defined_props = {k: set(all_defined_props) for k in element_classes}

# c_alpha
defined_props[c_alpha_class_name] -= (set(Biomolecule_gmx_prop_names) | set(Biomolecule_energy_prop_names) |
                                      set(Biomolecule_misc_prop_names))
defined_props[c_alpha_class_name] -= {'backbone_bond_angles', 'backbone_bond_lengths', 'backbone_dihedrals',
                                      'dihedral_omega', 'dihedral_angle_UMAP',
                                      'backbone_sphere_radius', 'elements_dist_CA_SI', 'pdist_adj_CA_SI_0.2'}
defined_props[c_alpha_class_name] |= {'LJ_pdist_all', 'LJ_pdist_diff_all'}

# c_alpha_sidechain
defined_props[c_alpha_sidechain_class_name] = copy.deepcopy(defined_props[c_alpha_class_name])
defined_props[c_alpha_sidechain_class_name] -= {'LJ_pdist_all', 'LJ_pdist_diff_all', 'h_bond_map',
                                                'pdist_adj_CA_SI_0.2'}

# heavy-atoms
defined_props[heavy_atoms_class_name] -= {'h_bond_map', 'elements_pdist'}

# all-atoms
defined_props[all_atoms_class_name] -= set(Biomolecule_UMAP_prop_names) | {'elements_pdist', 'pdist_adj_CA_SI_0.2'}

# Default properties
default_props = copy.deepcopy(defined_props)
nondefault_props_name = (set(Biomolecule_pair_prop_names) | set(Biomolecule_adj_prop_names) |
                         {'elements_pdist', 'CA_pdist', 'residue_type',
                          'backbone_sphere_radius', 'elements_dist_CA_SI', 'rmsd'})
for class_name in default_props:
    default_props[class_name] -= nondefault_props_name

# Properties units
distance_properties = ['end_to_end_dist', 'radial_2nd_mom', 'radial_2nd_mom_CA', 'R_g', 'elements_pdist', 'CA_pdist',
                       'rmsd', 'rmsd_min_train', 'rmsd_min_2rvd', 'backbone_bond_lengths', 'bond_length',
                       'LJ_pdist_all', 'LJ_pdist_diff_all', 'backbone_sphere_radius', 'bond_angle_enddist',
                       'elements_dist_CA_SI', 'pdist_adj_CA_SI_0.2']
distance_properties += Biomolecule_adj_prop_names
position_properties = ['elements_position', 'elements_position_local']
position_properties += [f"elements_position_{m}_local" for m in supported_adj_methods]
angle_properties = ['bond_angle', 'dihedral_angle', 'backbone_dihedrals', 'dihedral_omega', 'backbone_bond_angles']

prop_units = defaultdict(lambda: '')
for length_prop_name in distance_properties + position_properties:
    prop_units[length_prop_name] = 'nm'
for angle_prop_name in angle_properties:
    prop_units[angle_prop_name] = '\u00B0'  # Degree symbol unicode
for energy_prop_name in list(gmx_energy_names.keys()) + Biomolecule_energy_prop_names:
    prop_units[energy_prop_name] = 'kJ/mol'

# Properties binwidth
# Defaults
prop_binwidth: dict[str, float | None] = defaultdict(lambda: None)
prop_binwidth['n_elements'] = 1
prop_binwidth['h_bond_map'] = 1
for length_prop_name in distance_properties:
    prop_binwidth[length_prop_name] = 0.1  # nm
for prop_name in position_properties:
    prop_binwidth[prop_name] = 0.01  # nm
for angle_prop_name in angle_properties:
    prop_binwidth[angle_prop_name] = 1  # deg
for energy_prop_name in gmx_energy_names:
    prop_binwidth[energy_prop_name] = 5  # kJ/mol
for energy_prop_name in Biomolecule_energy_prop_names:
    prop_binwidth[energy_prop_name] = 5  # kJ/mol
for prop_name in Biomolecule_adj_prop_names:
    prop_binwidth[prop_name] = 0.01  # nm
for UMAP_prop_name in Biomolecule_UMAP_prop_names:
    prop_binwidth[UMAP_prop_name] = 0.1  # Unitless
for Z_prop_name in Biomolecule_Z_prop_names:
    prop_binwidth[Z_prop_name] = 0.01  # Unitless

# Override binwidth for special cases
for rg_prop_name in ['radial_2nd_mom', 'radial_2nd_mom_CA', 'R_g']:
    prop_binwidth[rg_prop_name] = 0.05  # nm
prop_binwidth['elements_position'] = 0.05  # nm
prop_binwidth['backbone_bond_lengths'] = 0.001  # nm
prop_binwidth['bond_length'] = 0.001  # nm
prop_binwidth['bond_angle_enddist'] = 0.001  # nm
prop_binwidth['rmsd'] = 0.05  # nm
prop_binwidth['rmsd_min_train'] = 0.01  # nm
prop_binwidth['rmsd_min_2rvd'] = 0.005  # nm
prop_binwidth['chirality'] = 1  # unitless

# Check for properties that have undefined binwidth
missing_binwidth_keys = set(Biomolecule_prop_names) - set(prop_binwidth.keys())
if missing_binwidth_keys:
    raise ValueError(f"The following Biomolecule properties do not have a defined binwidth:\n{missing_binwidth_keys}")

# Properties bin range
prop_binlims: dict[str, tuple] = defaultdict(lambda: (None, None))
for prop_name in distance_properties + ['n_elements']:
    prop_binlims[prop_name] = (0, None)
prop_binlims['backbone_bond_lengths'] = (None, None)
prop_binlims['LJ_pdist_diff_all'] = (None, None)
prop_binlims['chirality'] = (-1.5, 1.5)
prop_binlims['h_bond_map'] = (-0.5, 1.5)

for angle_prop_name in angle_properties:
    prop_binlims[angle_prop_name] = (-180, 180)

# Properties bin type
prop_bintype = defaultdict(lambda: 'proj')
prop_bintype['backbone_dihedrals'] = 'full'
for prop_name in Biomolecule_UMAP_prop_names:
    prop_bintype[prop_name] = 'full'

# Define properties whose stats will be overwritten
overwritten_stats = []


def npz_cache(filename: str, key_params: list[str] = None):
    """
    Decorator to cache metric outputs of a given dataset in a .npz file. The function must have a 'dataset' arg.
    If a 'resume' kwarg is defined, cache is loaded only when resume=True. Otherwise, cached data is overwritten.

    Args:
    filename: Path of the .npz file where cache will be stored. If not absolute, appends dataset.dir.
    key_params: List of args and kwargs name used to define the cache key. Defaults to all args and kwargs
    """

    def decorator(func):
        # Get the signature of the function to map args to names
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Map all passed args and kwargs to their parameter names
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            all_params = bound_args.arguments
            dataset = all_params['dataset']  # function must define 'dataset' arg

            # Filter parameters to only include those specified in key_params. Include dataset argument.
            if key_params is None:
                cache_key_params = dict(all_params)
                cache_key_params.pop('resume', None)  # Remove 'resume' kwarg if defined
            else:
                cache_key_params = {k: all_params[k] for k in ['dataset'] + key_params}

            # Check if a 'resume' kwarg is defined
            resume = all_params.get('resume', True)

            # Convert parameters into a hashable, string-based key
            cache_key_elements = []
            for k, v in cache_key_params.items():
                if isinstance(v, BiomoleculeDataset):
                    cache_key_elements.append(v.ID)
                else:
                    cache_key_elements.append(repr(v))
            cache_key = "_".join(cache_key_elements)

            # Load the existing cache if the file exists
            cache_filepath = pathlib.Path(filename).with_suffix('.npz')
            if not cache_filepath.is_absolute():
                cache_filepath = dataset.dir / cache_filepath
            cache = {}
            try:
                with np.load(cache_filepath, allow_pickle=True) as npzfile:
                    for k in npzfile:
                        v = npzfile[k]
                        if v.ndim == 0:
                            cache[k] = v.item()
                        else:
                            cache[k] = v
            except Exception:
                pass

            # Load from cache
            if resume and cache_key in cache:
                return cache[cache_key]

            # Calculate outputs and save them if missing from cache
            result = func(*args, **kwargs)
            if isinstance(result, tuple | list):
                obj_arr = np.empty((), dtype=object)
                obj_arr[()] = result
                cache[cache_key] = obj_arr
            else:
                cache[cache_key] = result
            cache_filepath.parent.mkdir(exist_ok=True, parents=True)
            np.savez(cache_filepath, **cache)

            return result

        return wrapper

    return decorator


class UMAPs:
    """
    Basic class for saving and loading UMAP models using a numpy .npz file
    """

    def __init__(self, ref_dataset: BiomoleculeDataset, load_models=True):
        self.ref_dataset = ref_dataset
        self.load_models = load_models  # Loads models from the saved npz file, if found
        self.filepath = ref_dataset.dir / "UMAPs.npz"
        self.estimators_name = ['standardizer', 'umap', 'rescaler']
        self.umaps: dict[str, sklearn.base.BaseEstimator] = {}
        self.updated = True

    def save(self):
        new_umaps = copy.copy(self.umaps)
        self.load()  # Load all saved umaps. self.umaps is overwritten in load()
        self.umaps |= new_umaps  # Update saved umaps with new ones
        # joblib.dump(self.umaps, self.filepath, compress=3)
        np.savez_compressed(self.filepath, **self.umaps)
        self.updated = True

    def save_updates(self):
        if not self.updated:
            self.save()

    def load(self, model_key: str = None):
        if self.filepath.exists():
            # # Load with joblib
            # if not self.umaps:
            #     self.umaps = joblib.load(self.filepath)
            # Load with numpy
            with np.load(self.filepath, allow_pickle=True) as npzfile:
                # Determine which models will be loaded
                if model_key is None:
                    loaded_keys = list(npzfile.keys())
                elif model_key in npzfile:
                    loaded_keys = [model_key]
                else:
                    loaded_keys = []

                for k in loaded_keys:
                    # Reconstruct the pipeline since numpy saves each pipeline steps in an array
                    v = Pipeline(steps=[(n, e) for n, e in zip(self.estimators_name, npzfile[k])])

                    # Turn verbose off
                    v[1].verbose = False
                    if v[1].tqdm_kwds is not None:
                        v[1].tqdm_kwds |= {'disable': True}
                    v[1].force_approximation_algorithm = True
                    self.umaps |= {k: v}

    def fit(self, prop_name: str):
        fitted_base_prop = get_UMAP_base_prop(self.ref_dataset, prop_name=prop_name)
        verbose_msg = f'Fitting UMAP on {prop_name!r} of {self.ref_dataset.name!r}...'
        print(verbose_msg)
        import umap  # Do local import since it takes time to import the umap module
        umap_fitter = umap.UMAP(verbose=True, random_state=2024, force_approximation_algorithm=True)
        model = Pipeline(steps=[(self.estimators_name[0], StandardScaler()),
                                (self.estimators_name[1], umap_fitter),
                                (self.estimators_name[2], StandardScaler())])
        model.fit(fitted_base_prop)
        umap_fitter.verbose = False

        # Remove attributes not needed for inference to reduce memory footprint
        deleted_attrs = ['graph_', '_rp_forest', '_search_graph']
        for attr in deleted_attrs:
            if hasattr(umap_fitter, attr):
                delattr(umap_fitter, attr)
        emptied_arrays = ['_raw_data']
        for attr in emptied_arrays:
            if hasattr(umap_fitter, attr):
                array = getattr(umap_fitter, attr)
                array_empty = scipy.sparse.csr_array(array.shape, dtype=array.dtype)
                setattr(umap_fitter, attr, array_empty)

        self.umaps[self.format_npzkey(prop_name)] = model
        self.updated = False

        return model

    def format_npzkey(self, prop_name: str):
        return f"{prop_name}_{self.ref_dataset.ID}"

    def __getitem__(self, item: str):
        umap_key = self.format_npzkey(item)
        if umap_key not in self.umaps and self.load_models:
            self.load(umap_key)
        if umap_key not in self.umaps:
            self.fit(item)

        return self.umaps[umap_key]

    def __contains__(self, item):
        umap_key = self.format_npzkey(item)
        return umap_key in self.umaps


def get_UMAP_base_prop(dataset: BiomoleculeDataset, prop_name: str):
    """
    Gather the base molecule property that is passed to a UMAP fitter for calculating UMAPed properties.
    Args:
        dataset: dataset of biomolecules from which to gather the given properties
        prop_name: name of the base property to be transformed with UMAP

    Returns:
        torch.Tensor
    """
    if prop_name == 'elements_position':
        dataset.load_all()
        elements_positions = torch.stack([mol.elements_position for mol in dataset])
        elements_positions = remove_rot(elements_positions)
        prop_val = elements_positions.flatten(-2, -1)
    else:
        prop_val = gather_mol_properties(dataset, properties_name=prop_name)[prop_name]

        # When angles are embedded, embed (sin(theta), cos(theta)) instead
        if prop_name in angle_properties:
            prop_val_rad = torch.deg2rad(prop_val)
            prop_val = torch.cat([torch.sin(prop_val_rad), torch.cos(prop_val_rad)], -1)

    return prop_val


def calculate_binedges(x: torch.Tensor | np.ndarray, binwidth: float, Min=None, Max=None):
    """
    Calculates edges of bins of a given binwidth covering the input data
    Args:
        x: input data
        binwidth: width of each bin

    Returns: bin edges of shape=(n+1,) where n is the number of bins

    """
    numpy_input = isinstance(x, np.ndarray)
    if numpy_input:
        x = torch.Tensor(x)

    if Min is not None:
        x_min = torch.tensor(Min)
    else:
        x_min = torch.floor(x.min() / binwidth) * binwidth
    if Max is not None:
        x_max = torch.tensor(Max)
    else:
        x_max = torch.ceil(x.max() / binwidth) * binwidth
    n_bins = torch.round((x_max - x_min) / binwidth).clamp(min=1).to(torch.int)
    if n_bins > 10000:
        warnings.warn(f'Number of bin edges is higher than 10000 (n_bins={n_bins}).')
    bin_edges = torch.linspace(x_min, x_max, n_bins + 1)

    if numpy_input:
        bin_edges = bin_edges.numpy()
    return bin_edges


def analyze_dataset(dataset: BiomoleculeDataset, stats: Iterable[str] | str = None, recalc_stats: Iterable[str] = None):
    """
    Analyzes a given dataset by calculation various statistics and properties
    Args:
        dataset: dataset to analyze
        stats: name of properties whose statistics will be calculated
        recalc_stats: name of statistics to recalculate
    Returns:
        None
    """
    # Define default stats if None are given
    if stats is None:
        stats = set(default_props[dataset.element_class]) | {'rmsd_min_train'}
    elif isinstance(stats, str):
        stats = {stats}
    else:
        stats = set(stats)

    # Parse statistics that need to be recalculated
    recalc_stats = [] if recalc_stats is None else list(recalc_stats)
    if recalc_stats:
        if 'gmx_energies' in recalc_stats:
            recalc_stats.pop(recalc_stats.index('gmx_energies'))
            recalc_stats += list(dataset.forcefield.gmx_energy_names)
        if 'UMAP' in recalc_stats:
            recalc_stats.pop(recalc_stats.index('UMAP'))
            recalc_stats.extend(list(Biomolecule_UMAP_prop_names))

    # Remove GMX energies since they will be calculated later
    gmx_energy_stats = stats & set(dataset.forcefield.gmx_energy_names)
    stats -= gmx_energy_stats

    # Remove statistics that are only applicable to Chignolin datasets
    is_chig = isinstance(dataset, ChigDDataset)
    is_chig |= isinstance(dataset, GeneratedDataset) and dataset.model_dataset.startswith('CLN')
    if is_chig:
        stats |= {'rmsd_min_2rvd'}
    else:
        stats -= {'rmsd_min_2rvd'}

    # Calculate requested statistics one by one
    for stat_name in stats:
        recalculate = stat_name in recalc_stats
        calculate_statistics(dataset, props_name=stat_name, recalculate=recalculate)

    # Calculate energy statistics
    if gmx_energy_stats:
        recalculate_energy_stats = bool(set(recalc_stats) & gmx_energy_stats)
        try:
            calculate_statistics(dataset, props_name=gmx_energy_stats, recalculate=recalculate_energy_stats)
        except Exception as e:
            print(f"GMX energy calculations failed with error:{e}")
            print('Skipping GROMACS energy statistics calculations')

    # Calculate the JS distance of subsets of the dataset for specific datasets
    main_datasets = ['nup98_12', 'nup98_24', 'AAQAA3', 'RS', 'CLN025']
    calc_extra_prop = dataset.main_name in main_datasets and dataset.is_split and dataset.split_ID[0] == '1'
    if isinstance(dataset, GeneratedDataset):
        calc_extra_prop |= dataset.model_dataset in main_datasets and dataset.model_split_ID[0] == '1'
    calc_extra_prop &= dataset.element_class == heavy_atoms_class_name  # Only calculate extra props on heavy-atoms
    if calc_extra_prop:
        if dataset.name.endswith('train') or isinstance(dataset, GeneratedDataset):
            resume = 'subsets_JS_dist' not in recalc_stats
            calculate_JS_dist_subsets(dataset, resume=resume)
        resume = 'dihedral_sampling_rate' not in recalc_stats
        calculate_dihedral_sampling_rate(dataset, resume=resume)


def calculate_statistics(dataset: BiomoleculeDataset, props_name: list[str] | set[str] | str = None, save=True,
                         recalculate=False, recalculate_prop=False, inplace=True) -> dict[str, Statistics]:
    """
    Calculates various statistics of a given dataset
    Args:
        dataset: BiomoleculeDataset for which statistics are computed
        props_name: list of names of properties for which statistics should be computed
        save: save the computed statistics if they were not saved before
        recalculate: recalculate statistics of the given properties omitting any previously saved statistics
        recalculate_prop: recalculate cached properties
        inplace: modifies the statistics dictionary of the dataset in place
    Returns:
        stats_dict: dictionary of Statistics
    """
    if props_name is None:
        props_name = set(default_props[dataset.element_class])
        # Remove GMX energies that are undefined by the dataset's forcefield.
        props_name -= set(Biomolecule_gmx_prop_names) - set(dataset.forcefield.gmx_energy_names)
    elif isinstance(props_name, str):
        props_name = {props_name}
    elif isinstance(props_name, list):
        props_name = set(props_name)
    else:
        props_name = copy.copy(props_name)

    # Check for undefined input properties
    undefined_props_name = props_name - defined_props[dataset.element_class]
    if undefined_props_name:
        raise ValueError(f"The following properties are undefined for {dataset.element_class!r}:{undefined_props_name}")

    # Ensure all properties have binning parameters defiend
    missing_binwidth_prop = props_name - set(prop_binwidth.keys())
    if missing_binwidth_prop:
        raise ValueError(f"The following properties have no defined binwidth:{missing_binwidth_prop}")

    # all_props_name = element_props_name + props_name
    stats_name = ['min', 'max', 'mean', 'std', 'mu2', 'cov']

    # Find the properties in the dataset statistics dictionary that have valid stats.
    stats_dict = dataset.statistics()
    if not inplace:
        stats_dict = copy.deepcopy(stats_dict)
    valid_stats_name = []
    for stat_name, stat_val in list(stats_dict.items()):
        curr_binwidth = stat_val.bin_widths
        if curr_binwidth is not None and prop_binwidth[stat_name] is not None:
            def_binwidth = torch.as_tensor(prop_binwidth[stat_name], dtype=curr_binwidth.dtype)
            binwidth_is_match = torch.isclose(curr_binwidth, def_binwidth.to(curr_binwidth.dtype), atol=1e-6).all()
        else:
            binwidth_is_match = True

        if (isinstance(stat_val, Statistics)
                and set(stat_val.names) >= set(stats_name)
                and not (recalculate and stat_name in props_name)
                and binwidth_is_match
                and stat_name not in overwritten_stats):
            valid_stats_name.append(stat_name)

    # Remove properties that have already been calculated
    props_name -= set(valid_stats_name)

    # Return the dictionary if no additional properties need statistics
    if not props_name:
        return stats_dict

    if isinstance(dataset, GeneratedDataset) or dataset.is_subset or not dataset.has_splits:
        # Gather the properties from all molecules
        props_dict = gather_mol_properties(dataset, properties_name=props_name, resume=not recalculate_prop)

        # Determine the independent group that each molecule belongs to
        if isinstance(dataset, XTCDataset):
            data_group_ID = torch.tensor(dataset.mol_run_ID)
        else:
            data_group_ID = None

        # Calculate various statistics of the input properties
        for prop_name in props_dict:
            if data_group_ID is not None and props_dict[prop_name].shape[0] == data_group_ID.numel():
                group_ID = data_group_ID
            else:
                group_ID = None

            if prop_name == 'h_bond_map':
                note = ('Each dimension corresponds to an entry in a 2D H-bond matrix. To reshape to 2D matrix, run:"\n'
                        'map_d = int(np.sqrt(self.d))\n'
                        'map_2D = self.mu1.reshape(map_d,map_d)\n"')
            else:
                note = ''

            prop_stats = Statistics.from_data(props_dict[prop_name].cpu(),
                                              bin_width=prop_binwidth[prop_name],
                                              bin_lims=prop_binlims[prop_name],
                                              bin_type=prop_bintype[prop_name],
                                              data_group_ID=group_ID,
                                              note=note)
            stats_dict[prop_name] = prop_stats
    else:
        # If the dataset has splits, accumulate their stats dict and return the sum of all split stats
        split_set_ID = next(iter(dataset.splits_sets))
        dataset_splits = dataset.load_splits(set_ID=split_set_ID)
        split_stats = []
        for d in dataset_splits:
            split_stats.append(calculate_statistics(d, props_name=props_name, save=save, recalculate=recalculate,
                                                    recalculate_prop=recalculate_prop, inplace=inplace))
        for k in split_stats[0]:
            stats_dict[k] = sum([stat_dict[k] for stat_dict in split_stats[1:]], split_stats[0][k])

    # Save the calculated statistics
    if save:
        dataset.save_statistics(stats_dict)

    return stats_dict


def calculate_training_metrics(dataset: BiomoleculeDataset, calculate_pairwise_stat=False):
    """
    Calculates metrics tracked during training for a given dataset of molecules
    Args:
        dataset: dataset of molecules
        calculate_pairwise_stat: calculates pairwise metrics (long calculation)

    Returns:
        dictionary of the metrics' name and value
    """
    metrics = dict()
    metrics['radial_2nd_mom_avg'] = dataset.statistics('radial_2nd_mom').mean.item()
    metrics['radial_2nd_mom_std'] = dataset.statistics('radial_2nd_mom').std.item()
    metrics['end_to_end_dist_avg'] = dataset.statistics('end_to_end_dist').mean.item()
    metrics['end_to_end_dist_std'] = dataset.statistics('end_to_end_dist').std.item()

    energy_props_name = {'BondEnergy', 'LJEnergy_tot', 'BondAngleEnergy_tot', 'DihedralEnergy_tot',
                         'CMAPEnergy_tot'} & defined_props[dataset.element_class]
    other_props_name = {'bond_length', 'bond_angle'} & defined_props[dataset.element_class]
    extra_props_name = energy_props_name | other_props_name
    extra_stats = calculate_statistics(dataset, props_name=list(extra_props_name))

    for energy_name in energy_props_name:
        metrics[f'{energy_name}_avg'] = extra_stats[energy_name].mean.sum()
        metrics[f'{energy_name}_std'] = torch.sqrt(extra_stats[energy_name].cov.sum())

    for prop_name in other_props_name:
        metrics[f'{prop_name}_avg'] = extra_stats[prop_name].mean
        metrics[f'{prop_name}_cov'] = extra_stats[prop_name].cov

    if calculate_pairwise_stat:
        calculate_statistics(dataset, props_name=Biomolecule_pair_prop_names)

    return metrics


def calculate_pair_prop(args: tuple[Biomolecule, Biomolecule, str]):
    """
    Worker function to calculate pairwise properties between two given Biomolecules.
    Args:
        args: tuple of the first and second Biomolecule compared along with a string that defines the property method.

    Returns:
        torch.Tensor representing the calculated pairwise property. shape = (n_elements,prop_dim) or (prop_dim,)
    """
    if isinstance(args[0], Biomolecule):
        mol1, mol2, prop_name = args
    else:
        global dataset_global
        prop_name = args[2]
        mol1, mol2 = dataset_global[args[0]], dataset_global[args[1]]

    prop_val = getattr(mol1, prop_name)(mol2)
    if not isinstance(prop_val, torch.Tensor):
        prop_val = torch.tensor(prop_val).float()

    if prop_val.nelement() > 1 and prop_val.shape[0] == mol1.n_elements:
        prop_val = prop_val.reshape(mol1.n_elements, -1)
    elif prop_val.nelement() == 1:
        prop_val = prop_val.reshape(1, )
    return prop_val


def worker_init_func(data):
    global dataset_global
    dataset_global = copy.deepcopy(data)


def gather_mol_properties(dataset: BiomoleculeDataset, properties_name: str | list[str] | set[str] = None, resume=True) \
        -> dict[str, torch.Tensor]:
    """
    Gathers Biomolecule properties from a dataset of molecules
    Args:
        dataset: BiomoleculeDataset dataset
        properties_name: names of the properties to be calculated
        resume: resume calculations on properties that are cached

    Returns:
        dictionary of properties where keys correspond to property names and values are the calculated property
    """
    if properties_name is None:
        properties_name = set(default_props[dataset.element_class])
    if isinstance(properties_name, str):
        properties_name = [properties_name]
    if isinstance(properties_name, list):
        properties_name = set(properties_name)

    props_dict = defaultdict(list)
    single_properties_name = properties_name & set(Biomolecule_prop_names)
    if dataset.forcefield is not None:
        gmx_en_properties_name = properties_name & set(dataset.forcefield.gmx_energy_names)
    else:
        gmx_en_properties_name = {}
    pair_properties_name = properties_name & set(Biomolecule_pair_prop_names)
    pair_normsd_min_props_name = pair_properties_name - {'rmsd_min_train', 'rmsd_min_2rvd'}
    energy_properties_name = properties_name & set(Biomolecule_energy_prop_names)
    misc_properties_name = properties_name & set(Biomolecule_misc_prop_names)
    UMAP_properties_name = misc_properties_name & set(Biomolecule_UMAP_prop_names)
    Z_properties_name = misc_properties_name & set(Biomolecule_Z_prop_names)
    misc_properties_name -= UMAP_properties_name | Z_properties_name

    # Load all molecules in the dataset to speed up fetching of properties. GROMACS properties do not need fetching.
    if single_properties_name or pair_normsd_min_props_name:
        dataset.load_all()
    mol_ex = dataset[0]

    # Gather the properties from all molecules
    if single_properties_name:
        # print(f'Starting calculation of {single_properties_name} for {dataset.name}, ID={dataset.ID}')
        for prop_name in single_properties_name:
            mol_iterator = tqdm.tqdm(dataset, desc=f'Calculating {prop_name!r} for {dataset.name!r}')
            for i, mol in enumerate(mol_iterator):
                mol: Biomolecule
                base_prop_val = getattr(mol, prop_name)
                if prop_name == 'elements_position':
                    base_prop_val = remove_centroid(base_prop_val)

                if not isinstance(base_prop_val, torch.Tensor):
                    base_prop_val = torch.tensor(base_prop_val).float()

                if base_prop_val.nelement() > 1 and base_prop_val.shape[0] == mol.n_elements:
                    base_prop_val = base_prop_val.reshape(mol.n_elements, -1)
                elif base_prop_val.nelement() == 1:
                    base_prop_val = base_prop_val.reshape(1, )

                # On the first iteration, initialize the tensor that will accumulate all data
                if i == 0:
                    props_dict[prop_name] = base_prop_val.repeat((len(dataset), *(base_prop_val.ndim - 1) * [1]))
                    prop_val_dim0 = base_prop_val.shape[0]

                props_dict[prop_name][i * prop_val_dim0:(i + 1) * prop_val_dim0] = base_prop_val

    # Gather GROMACS energy properties
    if gmx_en_properties_name:
        gmx_energy_dict, _ = calculate_gmx_energy(dataset, regen=not resume)
        for prop_name in tqdm.tqdm(gmx_en_properties_name, f'Gathering gmx energies for {dataset.name!r}'):
            base_prop_val = gmx_energy_dict[prop_name]
            if not isinstance(base_prop_val, torch.Tensor):
                base_prop_val = torch.tensor(base_prop_val)
            if not torch.is_floating_point(base_prop_val):
                base_prop_val = base_prop_val.double()
            props_dict[prop_name] = base_prop_val

    # Initialize the reference dataset, which is may be used downstream for some properties
    ref_dataset_is_needed = (bool(UMAP_properties_name) or bool(Z_properties_name)
                             or 'rmsd_min_train' in pair_properties_name)
    if ref_dataset_is_needed:
        if isinstance(dataset, GeneratedDataset):
            ref_dataset_name = dataset.model_dataset
            ref_dataset_dir = None
        else:
            ref_dataset_name = dataset.main_name
            ref_dataset_dir = dataset.dir
        ref_dataset = load_dataset(ref_dataset_name, directory=ref_dataset_dir, element_class=dataset.element_class,
                                   debug=False)

        # Load the 'train' split of the reference dataset
        if isinstance(dataset, GeneratedDataset):
            ref_split_set_ID = dataset.model_split_ID[0]
        else:
            ref_split_set_ID = dataset.split_ID[0]
        ref_dataset_train = ref_dataset.load_splits('train', set_ID=ref_split_set_ID)

    # Calculate pairwise properties
    if pair_properties_name:
        for prop_name in pair_normsd_min_props_name:
            n_pairs = int(len(dataset) * (len(dataset) - 1) / 2)
            mol_pair_iter = itertools.combinations(dataset.molecules, 2)
            worker_args = ((*mol_pair, prop_name) for mol_pair in mol_pair_iter)
            n_workers_temp = 0
            if n_workers_temp > 0:
                pool = multiprocessing.Pool(n_parallel_processes, initializer=worker_init_func,
                                            initargs=(dataset.molecules,))
                tasks_iter = pool.imap_unordered(calculate_pair_prop, worker_args)
            else:
                tasks_iter = map(calculate_pair_prop, worker_args)

            tasks_iter = tqdm.tqdm(tasks_iter, total=n_pairs, desc=f'Calculating {prop_name} for {dataset.name}.')
            prop_val_tensor = torch.zeros((n_pairs,))
            for i, base_prop_val in enumerate(tasks_iter):
                prop_val_tensor[i] = base_prop_val

            props_dict[prop_name] = prop_val_tensor

            if n_workers_temp > 0:
                pool.close()
                pool.join()

        # Special cases
        if 'rmsd_min_train' in pair_properties_name:
            if dataset.ID == ref_dataset_train.ID:
                props_dict['rmsd_min_train'] = torch.zeros(len(dataset))
            else:
                checkpoint_filepath = 'rmsd_min_train' if isinstance(dataset, GeneratedDataset) else None
                props_dict['rmsd_min_train'] = calculate_rmsd_min(dataset, ref_dataset=ref_dataset_train, resume=resume,
                                                                  checkpoint_filepath=checkpoint_filepath)

        if 'rmsd_min_2rvd' in pair_properties_name:
            pdb_2rvd_dataset = load_dataset('PDB_2RVD', element_class=dataset.element_class, debug=False)

            # Check that topology of molecules in dataset matches the topology of 2RVD
            if not mol_ex.check_top_match(pdb_2rvd_dataset[0]):
                raise ValueError(f"Cannot calculate 'rmsd_min_2rvd'. "
                                 f"Topology of {dataset.name!r} does not match topology of {pdb_2rvd_dataset.name!r}")

            props_dict['rmsd_min_2rvd'] = calculate_rmsd_min(dataset, ref_dataset=pdb_2rvd_dataset, resume=resume,
                                                             checkpoint_filepath='rmsd_min_2rvd')

    # Calculate energy-type properties
    energy_and_misc_props_name = energy_properties_name | misc_properties_name
    if energy_and_misc_props_name:
        dataset.load_all()
        if energy_and_misc_props_name & {'BondEnergy', 'BondEnergy_tot', 'bond_length', 'backbone_sphere_radius'}:
            bond_energy_mod = BondEnergy(top_file=dataset.top_filepath, element_class=dataset.element_class)
        if energy_and_misc_props_name & {'bond_angle', 'bond_angle_enddist', 'BondAngleEnergy_tot'}:
            BA_mod = BondAngleEnergy(top_file=dataset.top_filepath, element_class=dataset.element_class)
        if energy_and_misc_props_name & {'LJEnergy_tot', 'LJ_pdist_all', 'LJ_pdist_diff_all'}:
            LJ_mod = LJEnergy(top_file=dataset.top_filepath, element_class=dataset.element_class)
        if energy_and_misc_props_name & {'CMAPEnergy_tot'}:
            CMAP_mod = CMAPEnergy(top_file=dataset.top_filepath, element_class=dataset.element_class)
        if energy_and_misc_props_name & {'dihedral_angle', 'DihedralEnergy_tot'}:
            dih_mod = DihedralAngleEnergy(top_file=dataset.top_filepath, element_class=dataset.element_class)

        for prop_name in energy_and_misc_props_name:
            if prop_name == 'BondEnergy':
                prop_func = lambda x: bond_energy_mod(x.elements_position).unsqueeze(0)
            elif prop_name == 'BondEnergy_tot':
                prop_func = lambda x: bond_energy_mod(x.elements_position).sum().unsqueeze(0)
            elif prop_name == 'bond_length':
                prop_func = lambda x: bond_energy_mod.calculate_bond_length(x.elements_position).unsqueeze(0)
            elif prop_name == 'bond_angle':
                prop_func = lambda x: BA_mod.calculate_bond_angle(x.elements_position).rad2deg().unsqueeze(0)
            elif prop_name == 'bond_angle_enddist':
                prop_func = lambda x: BA_mod.calculate_endpoints_dist(x.elements_position).unsqueeze(0)
            elif prop_name == 'BondAngleEnergy_tot':
                prop_func = lambda x: BA_mod(x.elements_position).sum().unsqueeze(0)
            elif prop_name == 'dihedral_angle':
                prop_func = lambda x: dih_mod.calculate_dihedral_angle(x.elements_position,
                                                                       unique=True).rad2deg().unsqueeze(0)
            elif prop_name == 'DihedralEnergy_tot':
                prop_func = lambda x: dih_mod(x.elements_position).sum().unsqueeze(0)
            elif prop_name == 'LJEnergy_tot':
                prop_func = lambda x: LJ_mod(x.elements_position).sum().unsqueeze(0)
            elif prop_name == 'CMAPEnergy_tot':
                prop_func = lambda x: CMAP_mod(x.elements_position).sum().unsqueeze(0)
            elif prop_name == 'LJ_pdist_all':
                prop_func = lambda x: LJ_mod.calculate_pair_dist(x.elements_position)
            elif prop_name == 'LJ_pdist_diff_all':
                prop_func = lambda x: LJ_mod.calculate_pair_dist(x.elements_position) - LJ_mod.sigma
            elif prop_name == 'backbone_sphere_radius':
                prop_func = lambda x: calculate_backbone_packings(x, bond_ind=bond_energy_mod.bonds_atom_ind)
            elif prop_name.startswith('elements_position') and prop_name in Biomolecule_adj_prop_names:
                match = re.search("elements_position_(.*)_local", prop_name)
                coarse_graining_method = match.group(1)
                if coarse_graining_method not in supported_adj_methods:
                    raise NotImplementedError(f"coarse graining method {coarse_graining_method!r} is not implemented.")
                dataset_coarse_grains = coarse_grains_sets.get_adj_grains(method=coarse_graining_method)

                def prop_func(mol: Biomolecule):
                    atom_positions = mol.elements_position
                    atom_coarse_positions = mol.elements_coarse_pos(method=coarse_graining_method,
                                                                    coarse_grains=dataset_coarse_grains)
                    return atom_positions - atom_coarse_positions
            elif prop_name.startswith('pdist_adj'):
                m = re.search('pdist_adj_(.*)_([.0-9]+)', prop_name)
                coarse_graining_method, adj_range = m[1], float(m[2])
                if coarse_graining_method not in supported_adj_methods:
                    raise NotImplementedError(f"coarse graining method {coarse_graining_method!r} is not implemented.")
                dataset_coarse_grains = coarse_grains_sets.get_adj_grains(method=coarse_graining_method)

                def prop_func(mol: Biomolecule):
                    return mol.calculate_adj_elements_pdist(buffer_dist=adj_range, method=coarse_graining_method,
                                                            coarse_grains=dataset_coarse_grains)
            elif prop_name == 'elements_dist_CA_SI':
                dataset_coarse_grains = coarse_grains_sets.get_adj_grains(method='CA_SI')

                def prop_func(mol: Biomolecule):
                    # The radius is defined as an n_cg vector where n_cg is the number of coarse grains found in the molecule
                    elems_coarse_pos = mol.elements_coarse_pos(method='CA_SI', coarse_grains=dataset_coarse_grains)
                    elems_coarse_dist = torch.linalg.norm(mol.elements_position - elems_coarse_pos, dim=-1)
                    elems_coarse_ind = mol.elements_coarse_ind(method='CA_SI')
                    elems_n_coarse_grains = elems_coarse_ind.max() + 1

                    # On columns that do not correspond to the element coarse grain, assign a value of 0.0.
                    # This means the mean value of the radii should not be trusted since it will be underestimated.
                    prop_val = torch.zeros(mol.n_elements, elems_n_coarse_grains)
                    prop_val[torch.arange(mol.n_elements), elems_coarse_ind] = elems_coarse_dist
                    return prop_val
            elif prop_name == 'h_bond_map':
                def prop_func(mol: Biomolecule):
                    h_bond_counts = calculate_hbond_counts(mol)
                    h_bond_map = h_bond_counts.float().clip(max=1.0).flatten().unsqueeze(0)
                    return h_bond_map
            else:
                raise ValueError(f"energy property '{prop_name}' is undefined.")

            mol_iterator = tqdm.tqdm(dataset, desc=f'Calculating {prop_name!r} for {dataset.name!r}')

            # Handle properties with large memory footprint differently
            if prop_name in ['LJ_pdist_all', 'LJ_pdist_diff_all']:
                prop_d = LJ_mod.sigma.numel()
                # This tensor can have a very large memory footprint.
                props_dict[prop_name] = torch.zeros(len(dataset) * prop_d, dtype=torch.double)
                for i, mol in enumerate(mol_iterator):
                    props_dict[prop_name][i * prop_d:(i + 1) * prop_d] = prop_func(mol)
            elif prop_name.startswith('pdist_adj') or prop_name in ['backbone_sphere_radius']:
                # The size of these properties varies for each molecule
                for i, mol in enumerate(mol_iterator):
                    props_dict[prop_name].append(prop_func(mol))
            else:
                prop_val_ex = prop_func(mol_ex)
                props_dict[prop_name] = prop_val_ex.repeat((len(dataset), *(prop_val_ex.ndim - 1) * [1]))
                prop_val_dim0 = prop_val_ex.shape[0]
                for i, mol in enumerate(mol_iterator):
                    props_dict[prop_name][i * prop_val_dim0:(i + 1) * prop_val_dim0] = prop_func(mol)

    # Special properties
    # UMAP properties
    if UMAP_properties_name:
        umaps = UMAPs(ref_dataset=ref_dataset, load_models=resume)

        # Apply UMAP transform
        for prop_name in UMAP_properties_name:
            base_prop_name = prop_name.removesuffix('_UMAP')
            umap_model = umaps[base_prop_name]

            # Apply the fitted transform on the associated property
            base_prop_val = get_UMAP_base_prop(dataset, prop_name=base_prop_name)
            transformed_prop_val = []
            N_chunks = len(dataset) // 1000 + 1

            base_prop_iter = np.array_split(base_prop_val, N_chunks, axis=0)
            base_prop_iter = tqdm.tqdm(base_prop_iter, desc=f'Calculating {prop_name!r}', total=N_chunks)
            for base_prop_val_temp in base_prop_iter:
                transformed_prop_val_temp = torch.tensor(umap_model.transform(base_prop_val_temp))
                transformed_prop_val.append(transformed_prop_val_temp)
            transformed_prop_val = torch.cat(transformed_prop_val, dim=0)
            props_dict[prop_name] = transformed_prop_val

        # Save UMAP models that were fitted above (if any).
        umaps.save_updates()

    # Z-score properties
    for prop_name in Z_properties_name:
        base_prop_name = prop_name.removesuffix('_Z')
        prop_val = gather_mol_properties(dataset, properties_name=base_prop_name)[base_prop_name]
        prop_stats: Statistics = ref_dataset_train.statistics(base_prop_name)
        prop_Z = (prop_val - prop_stats.mean) / prop_stats.std
        props_dict[prop_name] = prop_Z

    # Accumulate properties from all molecules
    for prop_name in props_dict:
        if isinstance(props_dict[prop_name], list):
            props_dict[prop_name] = torch.cat(props_dict[prop_name], dim=0)

    return props_dict


def J_pdist(mol_subsets1: list[Iterable[Biomolecule]], mol_subsets2: list[Iterable[Biomolecule]], prop: str,
            projection=None):
    """
    Calculates Jeffrey's distance of a given property between all pairs of given subsets of biomolecules
    Args:
        mol_subsets1: list of size M containing the first molecule subsets compared in each comparison
        mol_subsets2: list of size N containing the first molecule subsets compared in each comparison
        prop: property for which the Jeffrey's distance is calculated
        projection: projection matrix to reduce the dimension of the compared property. Size depends on given 'prop'

    Returns:
        np.ndarray of shape=(M,N)
    """
    M, N = len(mol_subsets1), len(mol_subsets2)

    # Calculate the biomolecule property on the subsets
    def get_prop_value(molecules: list[Biomolecule]):
        prop_values = len(molecules) * [torch.Tensor()]
        for i, mol in enumerate(molecules):
            if prop == 'elements_pdist':
                prop_value = mol.elements_pdist
            else:
                raise NotImplementedError(f"property '{prop}' is not implemented.")
            if prop_value.device != 'cpu':
                prop_value = prop_value.cpu()
            if projection is not None:
                prop_value = prop_value @ projection
            prop_values[i] = prop_value.squeeze()
        return torch.stack(prop_values, dim=0).numpy()

    dataset1_splits_prop = len(mol_subsets1) * [None]
    for i, split_mols in enumerate(mol_subsets1):
        dataset1_splits_prop[i] = get_prop_value(split_mols)
    dataset2_splits_prop = len(mol_subsets2) * [None]
    for i, split_mols in enumerate(mol_subsets2):
        dataset2_splits_prop[i] = get_prop_value(split_mols)

    # Calculate Jeffrey's distance between each pair of subsets
    pdist = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            pdist[i, j] = J_dist_mvn(dataset1_splits_prop[i], dataset2_splits_prop[j])
    return pdist


def calculate_split_pair_J_pdist(dataset_splits: list[BiomoleculeDataset], dataset_gen: BiomoleculeDataset, prop: str,
                                 n_splits=5, n_gen_splits=1, plot=False):
    """
    Calculates Jeffrey's distance of a given property between 3 pairs of datasets. The pairs of datasets compared are:
    'valid & test splits vs train split', 'train split vs generated', 'valid & test splits vs generated'.
    Args:
        dataset_splits: splits of a given dataset ordered as 'train', 'valid' and 'test.
                        Optionally, the 'test' split can be omitted to compare only the 'valid'
        dataset_gen: model dataset
        prop: property compared
        n_splits: number of subsets of valid & test molecules
        n_gen_splits: number of subsets of generated molecules
        plot: plot the distribution of the various Jeffrey's distances calculated

    Returns:

    """
    dataset_train, dataset_valid = dataset_splits[0], dataset_splits[1]

    if isinstance(dataset_train, XTCDataset) and isinstance(dataset_valid, XTCDataset):
        ref_molecules = dataset_valid.molecules
        if len(dataset_splits) > 2:
            assert isinstance(dataset_splits[2], XTCDataset), "Test dataset is not an XTCDataset."
            ref_molecules = ref_molecules + dataset_splits[2].molecules
        train_molecules_subsets = np.array_split(np.array(dataset_train.molecules), n_splits)
        ref_molecules_subsets = np.array_split(np.array(ref_molecules), n_splits)
    else:
        if not dataset_train.is_mols_loaded:
            raise ValueError(f'All molecules must be loaded to calculate J dist')
        train_molecules_subsets = [dataset_train.molecules]
        ref_molecules_subsets = [dataset_valid.molecules]
        if len(dataset_splits) > 2:
            ref_molecules_subsets = [dataset_splits[2].molecules]

    dataset_gen.load_all()
    generated_molecules = np.array_split(np.array(dataset_gen.molecules), n_gen_splits)

    # Find a projection matrix based on a subspace of the PC space calculated with the train and validation datasets
    train_stats = dataset_train.statistics(prop)
    valid_stats = dataset_valid.statistics(prop)
    train_valid_stats = train_stats + valid_stats
    train_valid_cov = train_valid_stats.cov
    prop_projection_matrix = PCA_proj_matrix(train_valid_cov, var_explained=0.99)

    # Perform three kinds of comparisons between the dataset splits and generated dataset
    ref_name = 'valid-test' if len(dataset_splits) > 2 else 'valid'
    comparison_names = [f'{ref_name}_vs_train', 'train_vs_gen', f"{ref_name}_vs_gen"]
    compared_molecules = [[] for _ in range(3)]
    compared_molecules[0] = [ref_molecules_subsets, train_molecules_subsets]
    compared_molecules[1] = [train_molecules_subsets, generated_molecules]
    compared_molecules[2] = [ref_molecules_subsets, generated_molecules]
    comparison_J_pdist = {}
    for i, name in enumerate(comparison_names):
        comparison_J_pdist[name] = J_pdist(compared_molecules[i][0], compared_molecules[i][1], prop=prop,
                                           projection=prop_projection_matrix)

    if plot:
        element_class = dataset_train[0].element_class
        prop_custom = prop.replace('elements', element_class)

        dataset_name = dataset_train.name.replace('_train', '')
        plot_dir = pathlib.Path(config.plots_dir, f'{dataset_name}_vs_{dataset_gen.name}')
        plot_filepath = plot_dir / f"{prop}_J_pdist_{dataset_gen.name}.pdf"

        fig, ax = plt.subplots()
        hist_alpha = 1 / 3
        J_dist_all = np.concatenate(list(v.flatten() for v in comparison_J_pdist.values()))
        binwidth = (J_dist_all.max() - J_dist_all.min()) / 25
        unit = 5 * 10 ** (np.floor(np.log10(binwidth)) - 1)
        binwidth = np.round(binwidth / unit) * unit
        bin_edges = calculate_binedges(J_dist_all, binwidth)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        for comparison_name, J_pdists in comparison_J_pdist.items():
            bin_counts, _ = np.histogram(J_pdists.flatten(), bin_edges, density=False)
            if comparison_name == f'{ref_name}_vs_train':
                ax.hist(bin_centers, bin_edges, weights=bin_counts, label=comparison_name, histtype='step')
            else:
                ax.hist(bin_centers, bin_edges, weights=bin_counts, label=comparison_name, alpha=0.5)

        # Add edge lines
        # ax.vlines(bin_edges, np.zeros_like(bin_edges), ax.get_ylim()[1] * np.ones_like(bin_edges), linestyles='--',
        #           color='k', alpha=hist_alpha)

        ax.legend()
        ax.set_xlabel(f"Jeffrey's distance of {prop_custom}")
        ax.set_ylabel('counts')
        ax.set_xlim(left=0.5)
        fig.savefig(plot_filepath, dpi=500, pad_inches=None, bbox_inches='tight')
    return comparison_J_pdist


def calculate_split_sample_J_dist(dataset_splits: list[BiomoleculeDataset], gen_dataset: GeneratedDataset,
                                  props: str | list[str]) -> dict[str, np.ndarray]:
    """
    Calculates Jeffrey's distance for a set of given properties between a samples dataset and splits of a dataset.
    Args:
        dataset_splits: list of datasets.
        gen_dataset: dataset of generated samples
        props: list of properties

    Returns:
        dictionary of the jeffrey's distance value for each property. The dictionary values are 1D arrays of size=len(dataset_splits)
    """
    if isinstance(props, str):
        props = [props]
    props_Jdist = dict()

    for prop in props:
        J_dist_pair = np.zeros(len(dataset_splits))
        samples_stat = gen_dataset.statistics(prop)
        for i, dataset_split in enumerate(dataset_splits):
            J_dist_pair[i] = samples_stat.prob_dist(dataset_split.statistics(prop), 'JD')
        props_Jdist[prop] = J_dist_pair
    return props_Jdist


def calculate_gmx_energy(dataset: XTCDataset | BiomoleculeDataset, top_filepath: str = None, forcefield_dir: str = None,
                         params_filepath=single_point_en_params_filepath, calc_dir: str | pathlib.Path = None,
                         regen=False):
    """
    Calculates various GROMACS energies of a given dataset
    Args:
        dataset:
        top_filepath: filepath of the topology .top file. Defaults to dataset.top_filepath
        forcefield_dir: forcefield directory. Defaults to dataset.forcefield.dir
        params_filepath: .mdp file definining the gmx mdrun parameters used to perform the single-point energy calculations
        calc_dir: directory where calculated energies and temporary files are saved
        regen: restarts calculation and regenerates all files.

    Returns:
        energies_dict: dictionary of energies where with keys corresponding to energy types and values corresponding to np.ndarrays
                      of the corresponding energy value for each molecule.
        saved_data_dict: dictionary of all data saved in the 'gmx_energy.npz' file.

    """
    if isinstance(dataset, Biomolecule | list):
        raise NotImplementedError(f"Input that consists of Biomolecule or List[Bimolecule] is deprecated. "
                                  f"Dataset is needed in pdb2gmx to define terminal protonation states.")
    supported_element_classes = [heavy_atoms_class_name, all_atoms_class_name]
    if dataset.element_class not in supported_element_classes:
        raise ValueError(f"Cannot calculate GROMACS' energies on dataset with element_class={dataset.element_class!r}. "
                         f"Supported classes are:{supported_element_classes}")

    # Determine the molecules ID
    mols_ID = np.array(dataset.molecules_ID)

    # Check if the dataset is a subset of a larger dataset. If yes, gather the energy values from the full dataset.
    if dataset.is_subset:
        assert not regen, "Call calculate_gmx_energy(...,regen=True) on full dataset instead of subset."
        full_dataset = load_dataset(name=dataset.main_name, directory=dataset.dir, debug=False)
        energies_dict, saved_data = calculate_gmx_energy(full_dataset, top_filepath=top_filepath,
                                                         forcefield_dir=forcefield_dir,
                                                         params_filepath=params_filepath, calc_dir=calc_dir,
                                                         regen=False)

        # Remove molecules that are not in the split dataset
        all_mols_ID = energies_dict['ID']
        is_mol_in_split = np.isin(all_mols_ID, mols_ID)
        energies_dict = {k: v[is_mol_in_split] for k, v in energies_dict.items()}
        saved_data['energies'] = energies_dict

        return energies_dict, saved_data

    if calc_dir is None:
        calc_dir = dataset.gmx_energy_filepath.parent
        data_filepath = dataset.gmx_energy_filepath
    else:
        calc_dir = pathlib.Path(calc_dir)
        data_filepath = calc_dir / 'gmx_energy.npz'

    if top_filepath is None:
        top_filepath = dataset.top_filepath
    if forcefield_dir is None:
        forcefield_dir = dataset.forcefield.dir
    else:
        forcefield_dir = pathlib.Path(forcefield_dir)

    # Directories
    current_dir = os.getcwd()
    if regen and calc_dir.exists():
        delayed_delete(calc_dir)
    calc_dir.mkdir(parents=True, exist_ok=True)
    forcefield_name = forcefield_dir.stem

    if not data_filepath.exists():
        # Load all molecules in the dataset
        dataset.load_all()
        molecules = dataset.molecules
        mol_ex = molecules[0]

        # Read the parameter file to determine the maximum cutoff distance
        pdb_length_units = mda.coordinates.PDB.PDBWriter.units['length']
        mdp_params = parse_gmx_mdp(params_filepath)
        cutoff_names = {'rcoulomb', 'rcoulomb-switch', 'rvdw', 'rlist'} & set(mdp_params.keys())
        cutoff_max = pint_reg.Quantity(max([mdp_params[k] for k in cutoff_names]), GMX_LENGTH_units)
        cutoff_max = cutoff_max.to(pdb_length_units)

        # Determine the size of the simulation box that will fit every molecule
        box_dim_all = np.array([mol.box_params()[1] for mol in molecules]).max()
        box_dim_all = pint_reg.Quantity(box_dim_all, mol_ex.length_units).to(pdb_length_units)
        box_enlargement = max([pint_reg.Quantity(4, 'nm').to(pdb_length_units), 4 * cutoff_max])
        box_dim_all += box_enlargement
        box_dim_all = box_dim_all.magnitude
        uni_dimensions = np.array(3 * [box_dim_all] + 3 * [90.0])

        # Initialize universe from topology
        template_uni = uni_from_top(top_filepath=top_filepath, element_class=all_atoms_class_name, trajectory=True,
                                    rename_C_OO=False)
        template_uni.dimensions = uni_dimensions

        # Concatenate all positions into a .trr file
        trr_filepath = calc_dir / 'molecules.trr'
        dataset.save_to_trr(filepath=trr_filepath, template_uni=template_uni)
        template_uni.load_new(trr_filepath)
        template_uni.dimensions = uni_dimensions

        # Define filenames for the examples files
        output_basename = 'gmx_energy'
        mol_ex_gro_filename = f'{output_basename}.gro'
        mol_ex_tpr_filename = f'{output_basename}.tpr'
        mol_ex_top_filename = f'{output_basename}.top'
        params_out_filename = f'{output_basename}.mdp'
        energy_filepath = calc_dir / f"{output_basename}.edr"
        mol_ex_gro_filepath = calc_dir / mol_ex_gro_filename
        mol_ex_top_filepath = calc_dir / mol_ex_top_filename
        shutil.copy(dataset.top_filepath, mol_ex_top_filepath)

        # Create an example .gro file, which will be used to generate a .tpr file
        template_uni.atoms.write(mol_ex_gro_filepath)

        # Recompute the energy of the .trr file with mdrun. Each frame of the .trr maps to a molecule
        gmx_mdrun(gro_filepath=mol_ex_gro_filepath, top_filename=mol_ex_top_filename, params_filepath=params_filepath,
                  output_basename=output_basename, progress_filepath=energy_filepath,
                  grompp_kwargs={'params_out_filename': params_out_filename, 'maxwarn': 2},
                  rerun=trr_filepath.name)

        # Read the edr file and store the calculated potential energies
        edr_reader = mda.auxiliary.EDR.EDRReader(str(energy_filepath))
        if 'charmm' in forcefield_name:
            forcefield_energy_names = charmm_gmx_energy_names
        elif 'amber' in forcefield_name:
            forcefield_energy_names = amber_gmx_energy_names
        else:
            raise ValueError(f"energy names for forcefield={forcefield_name} are undefined.")

        energies_dict = {k: edr_reader.data_dict[v] for k, v in forcefield_energy_names.items() if
                         k not in gmx_derived_energy_names}
        energies_dict['ID'] = mols_ID

        # Save all the energies plus extra metadata
        saved_data_dict = dict(topology_filename=top_filepath, forcefield_dir=forcefield_dir, mdp_params=mdp_params,
                               energies=energies_dict)
        np.savez(data_filepath, **saved_data_dict)

        # Delete temporary files and directories
        temp_filenames = [mol_ex_gro_filename, mol_ex_tpr_filename, mol_ex_top_filename, f'{output_basename}.log',
                          trr_filepath.name]
        for filename in temp_filenames:
            filepath = calc_dir / filename
            filepath.unlink()

        # Anonymize .mdp file
        mdp_filepath = calc_dir / params_out_filename
        gmx_anonymize(mdp_filepath, mdp_filepath)

        # Delete offsets hidden files created by MDAnalysis
        for fp in calc_dir.rglob('.*_offsets*'):
            fp.unlink()

        os.chdir(current_dir)
    else:
        with np.load(data_filepath, allow_pickle=True) as file:
            saved_data_dict = {k: v.item() for k, v in file.items()}
        energies_dict = saved_data_dict['energies']

    # Add derived properties
    for name in gmx_derived_energy_names:
        if name == 'V_dih_prop+improp':
            energies_dict[name] = energies_dict['V_dih_prop'] + energies_dict['V_dih_improp']
        else:
            raise NotImplementedError

    return energies_dict, saved_data_dict


def calculate_rmsd_min(dataset: BiomoleculeDataset, ref_dataset: BiomoleculeDataset,
                       use_device=True, resume=True, checkpoint_filepath: str | pathlib.Path = None):
    """
    Calculates the distribution of the rmsd of each molecule in the input dataset paired to the best match found in a reference dataset.
    Args:
        dataset: target dataset of molecules
        ref_dataset: reference set of molecules which will be used to search for closest match.
        use_device: calculate rmsd on config.device
        checkpoint_filepath: filepath of the .npz checkpoint if resuming is used
        resume: resume calculations using the given checkpoint .npz file. Only works if checkpoint_filepath is not None
    Returns:
        rmsd_min : 1D tensor storing minimum value of rmsd found when comparing each molecule to the reference dataset.
    """
    if dataset.element_class != ref_dataset.element_class:
        raise ValueError(f"The element class of the target({dataset.element_class!r}) "
                         f"and reference({ref_dataset.element_class!r}) dataset do not match.")

    mols_ID: np.ndarray[np.str_] = np.array(dataset.molecules_ID)
    rmsd_min = torch.full((len(dataset),), torch.nan)

    # Load checkpoint file if one is given
    if checkpoint_filepath is not None:
        # If only a filename is given, save the filename in dataset.calc_dir
        checkpoint_filepath = pathlib.Path(checkpoint_filepath)
        if not checkpoint_filepath.is_absolute():
            checkpoint_filepath = dataset.calc_dir / checkpoint_filepath
        checkpointer = DictCheckpointer(filepath=checkpoint_filepath)
        checkpoints = checkpointer.dictionary

        # Initialize checkpoint dict
        # Define checkpoint ID. Use main_name to avoid re-calculation for splits.
        # ref_dataset.ID accounts for element_class so the checkpoint ID is element_class sensitive.
        checkpoint_ID = f"{dataset.main_name}_vs_{ref_dataset.ID}"
        if checkpoint_ID not in checkpoints:
            checkpoints[checkpoint_ID] = dict(mols_ID=mols_ID, mols_rmsd_min=rmsd_min.numpy().astype(np.float32))
        check_dict = checkpoints[checkpoint_ID]

        # Make sure all given molecules are represented in the checkpointed file
        mols_ID_is_missing = ~np.isin(mols_ID, check_dict['mols_ID'])
        if mols_ID_is_missing.any():
            mols_ID_new = mols_ID[mols_ID_is_missing]
            check_dict['mols_ID'] = np.concatenate([check_dict['mols_ID'], mols_ID_new])
            check_dict['mols_rmsd_min'] = np.concatenate(
                [check_dict['mols_rmsd_min'], np.full(mols_ID_new.size, np.nan, dtype=np.float32)])

        # Check that molecules ID are unique. Otherwise, checkpoint data structure will cause errors
        if check_dict['mols_ID'].size != np.unique(check_dict['mols_ID']).size:
            raise ValueError(f"Molecules must have unique IDs to save checkpoint.")

        # Find the indices that map the given molecules to the checkpointed molecules
        mols_ID_map = {ID: i for i, ID in enumerate(check_dict['mols_ID'])}
        mols_check_ind = np.array([mols_ID_map[ID] for ID in mols_ID])

        # Load checkpointed values if resuming
        if resume:
            rmsd_min[:] = torch.from_numpy(check_dict['mols_rmsd_min'][mols_check_ind])

    mols_to_do_ind = rmsd_min.isnan().nonzero().flatten().tolist()
    if mols_to_do_ind:
        # Load all molecules and move them to the device to speed up calculation.
        dataset.load_all()
        molecules = dataset.molecules
        molecules_to_do = {i: molecules[i] for i in mols_to_do_ind}
        ref_dataset.load_all()
        molecules_ref = ref_dataset.molecules
        molecules_ref_pos = torch.stack([ref_mol.elements_position for ref_mol in molecules_ref])

        # Move to device
        device = torch.device(config.device) if use_device else None
        molecules_ref_pos = molecules_ref_pos.to(device)
        rmsd_min = rmsd_min.to(device)

        # Cast positions to double for better precision
        molecules_ref_pos = molecules_ref_pos.double()

        msg = f'Calculating rmsd_min of {dataset.name!r} compared to {ref_dataset.name!r}'
        for i, mol in tqdm.tqdm(molecules_to_do.items(), msg):
            mol_pos = mol.elements_position.to(device).double()
            rmsd_ref = rmsd(positions=mol_pos, positions_ref=molecules_ref_pos, reflect=False)
            rmsd_min[i] = rmsd_ref.min().to(torch.get_default_dtype())

            if rmsd_min[i].isnan():
                raise ValueError(f"rmsd_min is nan. "
                                 f"mol_ID={mol.ID!r}, dataset={dataset.name!r}, ref_dataset={ref_dataset.name!r}.")

            # Save checkpoint
            if checkpoint_filepath is not None and checkpointer.is_checkpoint_ready():
                check_dict['mols_rmsd_min'][mols_check_ind] = rmsd_min.cpu().numpy().astype(np.float32)
                checkpointer.save()

        # Move output back to cpu
        rmsd_min = rmsd_min.cpu()

        # Final checkpoint
        if checkpoint_filepath is not None:
            check_dict['mols_rmsd_min'][mols_check_ind] = rmsd_min.cpu().numpy().astype(np.float32)
            checkpointer.save(force=True)

    return rmsd_min


def calculate_chig_fold_state(dataset: ChigDDataset, rmsd_cutoff=3.0):
    """
    Calculates the fold state of a given ChigDDataset
    Args:
        dataset: dataset of the Chignolin structure
        rmsd_cutoff: RMSD cutoff (angstrom) used to define the folded (<=) and unfolded state(>)

    Returns:
        mols_fold_state: shape=(n_mols,). array of 0 (folded) and 1 (unfolded)
        mols_rmsd_min_2rvd: shape=(n_mols,). array of minimum RMSD of each structure found by comparing to 2RVD ensemble
    """
    assert isinstance(dataset, ChigDDataset | GeneratedDataset), \
        f"chig fold state is only defined for {ChigDDataset.__class__.__name__}"

    # Load reference dataset
    pdb_2rvd_dataset = load_dataset('PDB_2RVD', debug=False)

    # Consider only heavy atoms
    pdb_2rvd_dataset.element_class = heavy_atoms_class_name
    dataset.element_class = heavy_atoms_class_name

    # Calculate minimum RMSD of each molecule in dataset
    mols_rmsd_min_2rvd = gather_mol_properties(dataset, 'rmsd_min_2rvd')['rmsd_min_2rvd']

    # Convert rmsd to Angstroms
    conv_factor = calculate_units_conversion_factor(dataset[0].length_units, 'angstrom')
    mols_rmsd_min_2rvd = conv_factor * mols_rmsd_min_2rvd

    # Define fold state using pre-determined cutoff
    mols_fold_state = mols_rmsd_min_2rvd <= rmsd_cutoff  # 0=folded, 1=unfolded

    return mols_fold_state, mols_rmsd_min_2rvd


def calculate_hbond_counts(mol: Biomolecule, donor_hydrogen_dist_max=0.12, donor_acceptor_dist_max=0.3,
                           donor_hydrogen_acceptor_angle_min=150):
    """
    Calculates the number of hydrogen bonds (H-bonds) between each pair of residues in a given Biomolecule
    Args:
        mol: analyzed molecule
        donor_hydrogen_dist_max: maximum distance (nm) allowed between donor and hydrogen to be counted as H-bond
        donor_acceptor_dist_max: maximum distance (nm) allowed between donor and acceptor to be counted as H-bond
        donor_hydrogen_acceptor_angle_min: minimum donor-hydrogen-acceptor angle allowed to be counted as H-bond

    Returns:
        h_bond_counts with shape=(n_res,n_res) where n_res corresponds to the number of residues in the molecule
        h_bond_counts[i,j] denotes the number of instances a donor atom in residue i donated a hydrogen to an acceptor atom in residue j
    """
    # Gather indices of the donors, acceptors and hydrogens
    atoms_is_H = torch.from_numpy(np.char.startswith(mol.elements_name, 'H'))
    atoms_is_O = torch.from_numpy(np.char.startswith(mol.elements_name, 'O'))
    atoms_is_N = torch.from_numpy(np.char.startswith(mol.elements_name, 'N'))
    atoms_is_donor = atoms_is_O | atoms_is_N
    atoms_is_acceptor = atoms_is_donor

    # Gather residue indices of the donors and acceptors
    donors_res_ind = mol.elements_resind[atoms_is_donor]
    acceptors_res_ind = mol.elements_resind[atoms_is_donor]

    # Gather positions of the donors, acceptors and hydrogens
    donor_pos = mol.elements_position[atoms_is_donor].reshape(-1, 1, 1, 3)  # dim 0 = donor
    acceptor_pos = mol.elements_position[atoms_is_acceptor].reshape(1, -1, 1, 3)  # dim 1 = acceptor
    H_pos = mol.elements_position[atoms_is_H].reshape(1, 1, -1, 3)  # dim 2 = H

    donor_H_vec = H_pos - donor_pos
    H_acceptor_vec = acceptor_pos - H_pos
    donor_H_dist2 = torch.square(donor_H_vec).sum(-1)  # Squared distance between donor and hydrogen
    donor_acc_dist2 = torch.square(donor_pos - acceptor_pos).sum(-1)  # Squared distance between donor and acceptor
    donor_H_is_pair = donor_H_dist2 <= donor_hydrogen_dist_max ** 2
    donor_acceptor_is_pair = (donor_acc_dist2 > 0.0) & (donor_acc_dist2 <= donor_acceptor_dist_max ** 2)
    donor_H_pair_ind = donor_H_is_pair.squeeze(1).nonzero()

    # Keep track of the number of H-bonds between each pair of residues and the h-bond geometry
    n_res = mol.n_residues
    hbond_counts = torch.zeros((n_res, n_res), dtype=torch.int)
    # DHA_angles, DA_dist, DH_dist = [], [], []
    for donor_i, hydrogen_i in donor_H_pair_ind:
        vec_DH = donor_H_vec[donor_i, 0, hydrogen_i]
        vec_DH_norm = donor_H_dist2[donor_i, 0, hydrogen_i].sqrt()

        acceptors_cand_ind = donor_acceptor_is_pair[donor_i, :, 0].nonzero()
        for acc_i in acceptors_cand_ind:
            vec_HA = H_acceptor_vec[0, acc_i, hydrogen_i]
            vec_HA_norm = vec_HA.norm()

            # Check the D-H-A angle between the HD vector and the HA vector
            # cos_theta = torch.linalg.vecdot(vec_DH, vec_HA) / (vec_DH_norm * vec_HA_norm)
            # angle = 180 - cos_theta.clip(-1.0, 1.0).acos().rad2deg()
            cos_theta = torch.linalg.vecdot(-vec_DH, vec_HA) / (vec_DH_norm * vec_HA_norm)  # -1 since we need HD vec
            angle = cos_theta.clip(-1.0, 1.0).acos().rad2deg()
            if angle >= donor_hydrogen_acceptor_angle_min:
                hbond_counts[donors_res_ind[donor_i], acceptors_res_ind[acc_i]] += 1

                # # Save info of the H-bond geometry
                # DHA_angles.append(angle)
                # DA_dist.append(vec_HA_norm)
                # DH_dist.append(vec_DH_norm)

    return hbond_counts


def generate_mol_orderings(dataset: BiomoleculeDataset, n_orderings: int = 100, seed=None):
    """
    Generates random orderings of the molecules in a given dataset. The method for reordering differs for XTCDatasets.
    Args:
        dataset: dataset that the reordered molecules belong to
        n_orderings: number of re-orderings
        seed: fix random seed for reproducibility

    Returns:
        np.ndarray with shape=(n_orderings, len(dataset))
    """
    rng = np.random.default_rng(seed)
    if isinstance(dataset, XTCDataset):
        # Find unique permutations of the runs (if possible)
        runs_ID = np.array(dataset.runs_ID)
        n_runs = len(runs_ID)
        n_unique_perms = math.factorial(n_runs)
        if n_orderings < n_unique_perms:
            perms = set()
            while len(perms) < n_orderings:
                perms.add(tuple(rng.permutation(n_runs)))
            perms = list(perms)
        else:
            perms = [rng.permutation(n_runs) for _ in range(n_orderings)]
        runs_ID_perms = runs_ID[np.stack(perms)]

        # Ensure molecules are sorted based on run_ID (first) and frame index (2nd)
        mols_run_ID = np.array(dataset.mol_run_ID)
        mol_frame_ind = np.array(dataset.mol_frame_ind)

        sorting_ind = np.lexsort((mol_frame_ind, mols_run_ID))
        mols_run_ID = mols_run_ID[sorting_ind]

        # Order the molecules based on the sampled run IDs
        mols_run_subset = {run_ID: np.nonzero(mols_run_ID == run_ID)[0] for run_ID in runs_ID}
        mols_ordering_ind = np.ones((n_orderings, mols_run_ID.size), dtype=int)
        for i in range(n_orderings):
            mols_ordering_ind[i] = np.concatenate([mols_run_subset[j] for j in runs_ID_perms[i]])

        # Map back to the original ordering
        mols_ordering_ind = sorting_ind[mols_ordering_ind]
    elif isinstance(dataset, GeneratedDataset):
        # For GeneratedDatasets, we can shuffle all molecules together since they do not have a run ID.
        mols_ordering_ind = np.stack([rng.permutation(len(dataset)) for _ in range(n_orderings)])
    else:
        raise NotImplementedError

    return mols_ordering_ind


def calculate_JS_dist_subsets(dataset: BiomoleculeDataset, ref_dataset: BiomoleculeDataset = None,
                              properties_name: str | list[str] = None,
                              n_subsets=200, n_resamples=1000, subset_size_max=10000,
                              seed=67, save_checkpoint=True, resume=True) -> dict[str, np.ndarray]:
    """
    Calculates Jensen-Shannon distances between subsets of increasing size from a given dataset and a reference dataset.
    Several properties are compared depending on the given input.
    Args:
        dataset: target dataset
        ref_dataset: reference dataset
        properties_name: list of properties to compare
        n_subsets: number of subsets used to span the range [1,subset_size_max] of subset size in log space.
        n_resamples: number of times datasets are resampled for each subset size. Used to evaluate variation of the JS distance
        subset_size_max: maximum size of the subsets
        seed: random seed for fixing sampling of random subsets
        save_checkpoint: save calculated values to checkpoint file
        resume: resume from previous checkpoint

    Returns:
        JS_dist_dict: dictionary where JS distance arrays are saved for each given property.
                     JS_dist_dict[prop_name][i,j] denotes the 'prop_name' JS dist. of the ith subset for the jth random ordering
                     JS_dist_dict['n_samples'] is a 1D array of size n_subsets denoting the number of samples in each compared subset.
    """
    # Use the associated test set as the reference dataset to compare against
    if ref_dataset is None:
        if isinstance(dataset, XTCDataset):
            if dataset.is_split:
                ref_dataset = load_dataset(dataset.main_name, element_class=dataset.element_class)
                ref_dataset = ref_dataset.load_splits('test', set_ID=dataset.split_ID[0])
            else:
                raise ValueError(f"There is no default reference dataset for the given dataset. ")
        elif isinstance(dataset, GeneratedDataset):
            ref_dataset = load_dataset(dataset.model_dataset, element_class=dataset.element_class)
            ref_dataset = ref_dataset.load_splits('test', set_ID=dataset.model_split_ID[0])
        else:
            raise NotImplementedError

    if properties_name is None:
        properties_name = ['R_g', 'bond_length', 'bond_angle', 'backbone_dihedrals']
    elif isinstance(properties_name, str):
        properties_name = [properties_name]

    # Initialize the checkpointer
    checkpoint_filepath = dataset.calc_dir / f"JSdist_subsets_{dataset.ID[:8]}_{ref_dataset.ID[:8]}.npz"
    checkpointer = DictCheckpointer(filepath=checkpoint_filepath)
    checkpoint_key = f"{n_subsets}_{subset_size_max}_{seed}"
    checkpoints_dict = checkpointer.dictionary
    if checkpoint_key not in checkpoints_dict:
        checkpoints_dict[checkpoint_key] = {}
    JS_dist_dict: dict[str, NDArray[float]] = checkpoints_dict[checkpoint_key]

    # If not resuming, remove the requested property keys from the checkpoint file
    if not resume:
        for prop_name in properties_name:
            JS_dist_dict.pop(prop_name, None)
        checkpointer.save(force=True)

    # Initialize the n_samples array
    if subset_size_max > len(dataset):
        warnings.warn(f'Reducing maximum subset size ({subset_size_max}) to {len(dataset)}.')
        subset_size_max = len(dataset)
    subsets_size = np.logspace(0, np.log10(subset_size_max), n_subsets)
    subsets_size = np.cumsum(np.ceil(np.diff(subsets_size, prepend=0))).astype(int)  # Ensures step is at least 1

    # Reduce the size of the subsets since enforcing a minimum step in size of 1 makes the last subset size exceeds
    # the given max.
    subsets_size_excedent = subsets_size[-1] - subset_size_max
    size_red_max = np.cumsum(np.clip(np.diff(subsets_size, prepend=0) - 1, a_min=0, a_max=np.inf))
    subsets_size_red = np.minimum(subsets_size / subsets_size[-1] * subsets_size_excedent, size_red_max).astype(int)
    subsets_size = subsets_size - subsets_size_red
    JS_dist_dict['n_samples'] = subsets_size
    subsets_size = torch.from_numpy(subsets_size).to(config.device)

    # Define the start and end index of sections that will be used to build the subsets cumulatively
    sections_startend_ind = torch.nn.functional.pad(subsets_size, (1, 0))

    # Initialize JS distance with nan to track progress
    for prop_name in properties_name:
        if prop_name not in JS_dist_dict:
            JS_dist_dict[prop_name] = np.full((n_subsets, n_resamples), np.nan)

        # Check if cached values have enough resamples. Otherwise, pad array to the needed size
        N_new_resamples = n_resamples - JS_dist_dict[prop_name].shape[1]
        if N_new_resamples > 0:
            pad_width = np.array([[0, 0], [0, N_new_resamples]], dtype=int)
            JS_dist_dict[prop_name] = np.pad(JS_dist_dict[prop_name], pad_width, 'constant', constant_values=np.nan)

    # Generate unique orderings of the molecules in the given dataset to form the subsets
    orderings_mol_ind = generate_mol_orderings(dataset, n_orderings=n_resamples, seed=seed)
    orderings_mol_ind = torch.from_numpy(orderings_mol_ind).to(config.device)

    # Define the binwidth used to calculate the prob. density of each property
    binwidth = copy.deepcopy(prop_binwidth)
    binwidth['R_g'] = 0.01  # Reduce binwidth for R_g compared to default

    for prop_name in properties_name:
        # Skip to next property if all JS distances have been calculated
        JS_dist_arr = JS_dist_dict[prop_name]
        JS_dist_is_nan = np.isnan(JS_dist_arr[-1])
        if JS_dist_is_nan.any():
            resample_start_ind = JS_dist_is_nan.nonzero()[0][0]
        else:
            continue

        # Gather the properties of the given dataset and the reference
        prop_val = gather_mol_properties(dataset, properties_name=prop_name)[prop_name].to(config.device)
        prop_val_ref = gather_mol_properties(ref_dataset, properties_name=prop_name)[prop_name].to(config.device)

        # Add trailing dimension of 1 if 1D to be consistent with the multi-d case
        if prop_val.ndim == 1:
            prop_val = prop_val.unsqueeze(-1)
            prop_val_ref = prop_val_ref.unsqueeze(-1)

        # Calculate statistics of the reference dataset.
        stat_ref = Statistics.from_data(prop_val_ref, bin_width=binwidth[prop_name], bin_lims=prop_binlims[prop_name],
                                        bin_type=prop_bintype[prop_name])

        # Initialize the statistics object for the given property
        stat = Statistics.from_data(prop_val, bin_width=binwidth[prop_name], bin_lims=prop_binlims[prop_name],
                                    bin_type=prop_bintype[prop_name])

        # Extend the bin limits of the target and reference stats to ensure their number of bins and bin edges match.
        lims1 = stat.bin_lims
        lims2 = stat_ref.bin_lims
        bin_lims = torch.stack([torch.min(lims1[:, 0], lims2[:, 0]), torch.max(lims1[:, -1], lims2[:, -1])], -1)
        stat.extend_bins(bin_lims)
        stat_ref.extend_bins(bin_lims)

        # Initialize the resample index iterator
        resample_ind_iter = range(resample_start_ind, n_resamples)
        tqdm_desc = f'Calculating {prop_name!r} JS dist. comparing {dataset.name!r} vs {ref_dataset.name!r}'
        resample_ind_iter = tqdm.tqdm(resample_ind_iter, desc=tqdm_desc)

        # Find the bin index of each molecule for each property dimension
        prop_bin_ind = torch.zeros_like(prop_val, dtype=torch.int)
        for i in range(stat.d):
            bin_ind = torch.bucketize(prop_val[:, i].contiguous(), boundaries=stat.binedges[i], right=True) - 1
            bin_ind[bin_ind == stat.binedges[i].numel() - 1] -= 1  # Needed if some data is on the last edge
            prop_bin_ind[:, i] = bin_ind

        for i in resample_ind_iter:
            # Gather the random ordering of the molecules to form all the subsets
            mol_ind = orderings_mol_ind[i, :subsets_size[-1]]

            # The subsets of increasing size are formed by cumulatively adding sets (sections) of molecule
            sections_size = sections_startend_ind[1:] - sections_startend_ind[:-1]
            mol_section_ind = torch.repeat_interleave(torch.arange(n_subsets, device=config.device), sections_size)
            JS_dist_temp = torch.zeros(n_subsets, device=config.device)

            if stat.bin_type == 'proj':
                for k in range(stat.d):
                    n_bins = stat._bincounts[k].numel()
                    prop_bin_ind_temp = prop_bin_ind[mol_ind, k]

                    # Create 2D grid to save bin counts for every additional set of molecules
                    section_counts = torch.zeros(n_subsets, n_bins, dtype=stat.bincounts_dtype, device=config.device)
                    ones = torch.ones_like(prop_bin_ind_temp, dtype=stat.bincounts_dtype)
                    section_counts.index_put_((mol_section_ind, prop_bin_ind_temp), ones, accumulate=True)

                    # Calculate JS Distance
                    subsets_counts = section_counts.cumsum(dim=0)  # shape=(n_subsets,n_bins)
                    ref_stat_counts = stat_ref._bincounts[k].flatten()  # shape=(n_bins,)
                    JS_dist_temp += torch.sqrt(JS_div(subsets_counts, ref_stat_counts)) / stat.d
            elif stat.bin_type == 'full':
                ref_stat_counts = stat_ref._bincounts.flatten()
                stat_counts = torch.zeros_like(stat._bincounts)
                for j in range(n_subsets):
                    new_prop_bin_ind = prop_bin_ind[mol_ind[sections_startend_ind[j]:sections_startend_ind[j + 1]]]
                    counts_source = torch.ones_like(new_prop_bin_ind[:, :1], dtype=stat.bincounts_dtype)
                    stat_counts.index_put_(indices=new_prop_bin_ind.unbind(-1), values=counts_source,
                                           accumulate=True)
                    JS_dist_temp[j] = torch.sqrt(JS_div(stat_counts.flatten(), ref_stat_counts))
            else:
                raise NotImplementedError(f"Bin type {stat.bin_type!r} is not implemented")

            # Checkpoint
            JS_dist_arr[:, i] = JS_dist_temp.cpu().numpy()
            if save_checkpoint:
                checkpointer.save()

        # Force checkpoint when done with a given property
        if save_checkpoint:
            checkpointer.save(force=True)

    # Make sure JS dist arrays have the correct number of resamples, which does not necessarily match the cached arrays.
    for prop_name in properties_name:
        JS_dist_dict[prop_name] = JS_dist_dict.pop(prop_name)[:, :n_resamples]

    return JS_dist_dict


def calculate_dihedral_sampling_rate(dataset: BiomoleculeDataset, method='Simple-improved', res_type='L',
                                     n_resamples=1000, n_subsets=2000, subset_size_max=10000,
                                     seed=73, save_checkpoint=True, resume=True):
    """
    Calculates the number of unique protein dihedral angle state for random subsets of a given dataset.
    The protein dihedral angle state is defined as the ordered tuple of backbone dihedral angle state of each non-terminal residue
    Args:
        dataset: BiomoleculeDataset
        method: Method for assigning a unique state to a backbone dihedral angle pair.
                'Simple' : Divides the Ramachandran space into 4 quadrants.
                'Simple-improved': Divides the Ramachandran map into 4 bins without splitting up regions that belong to
                                   the same secondary structure element.
                'Box' : Divides the Ramachandran space into 7 regions. See https://doi.org/10.1093/protein/gzu037
        res_type : Stereoisomer of amino acid ('L' or 'D') to consider when defining the dihedral angle state
        n_resamples: number of bootstrapped resamples used to evaluate the number of unique dihedral states
        n_subsets: number of subsets of the given dataset considered in evaluating the number of unique dihedral states
        subset_size_max: maximum size of the random subsets
        seed: random seed to fix the random sampling of subsets
        save_checkpoint: save calculated values to checkpoint file
        resume: resume from previous checkpoint

    Returns:
        subsets_size: torch.Tensor (shape=(n_subsets,)) number of molecules contained in each random subset
        n_unique_dihedral_states: torch.Tensor (shape=(n_subsets,n_resamples)) number of unique protein dihedral states
    """
    supported_methods = ['Simple', 'Box', 'Simple-improved']
    if method not in supported_methods:
        raise NotImplementedError(f"method {method} is not supported. Supported methods={supported_methods}")

    # Calculate the size of the subsets and check if there are enough molecules to form the subsets
    if subset_size_max > len(dataset):
        warnings.warn(f'Reducing maximum subset size ({subset_size_max}) to dataset length ({len(dataset)}).')
        subset_size_max = len(dataset)
    # subsets_size = np.logspace(0, np.log10(subset_size_max), n_subsets)
    subsets_size = np.linspace(1, subset_size_max, n_subsets)
    subsets_size = np.cumsum(np.diff(subsets_size, prepend=0).round().clip(min=1)).astype(int)  # Ensures step >= 1
    subsets_size = subsets_size[subsets_size <= subset_size_max]  # Ensures subset size does not exceed maximum size
    subsets_size[-1] = subset_size_max  # Ensures last subset has the maximum size
    n_subsets = subsets_size.shape[0]  # Redefine n_subsets in case it was not possible to find enough subsets

    # Initialize the checkpointer
    checkpoint_filepath = dataset.calc_dir / f"dihedral_state_sampling_{dataset.ID[:8]}.npz"
    checkpointer = DictCheckpointer(filepath=checkpoint_filepath)
    checkpoint_key = f"{method}_{res_type}_{n_subsets}_{subset_size_max}_{seed}"

    # If resuming, initialize checkpoints from the saved checkpoints in the checkpoint file
    checkpoints = checkpointer.dictionary if resume else {}

    # Initialize n_unique_dihedral_states array
    if checkpoint_key not in checkpoints:
        checkpoints[checkpoint_key] = np.full((n_subsets, n_resamples), 0, dtype=np.uint64)

    # If cached array does not have enough resamples, pad array to the needed size
    N_new_resamples = n_resamples - checkpoints[checkpoint_key].shape[1]
    if N_new_resamples > 0:
        pad_width = np.array([[0, 0], [0, N_new_resamples]], dtype=int)
        checkpoints[checkpoint_key] = np.pad(checkpoints[checkpoint_key], pad_width, 'constant', constant_values=0)
    n_unique_dihedral_states = checkpoints[checkpoint_key]

    # Return output if all resamples have been calculated
    n_unique_dihedral_states_resample_is_zero = np.equal(n_unique_dihedral_states[0, :n_resamples], 0)
    if not n_unique_dihedral_states_resample_is_zero.any():
        return subsets_size, n_unique_dihedral_states[:, :n_resamples]

    # Gather backbone dihedral angles
    mol_backbone_dihedrals = gather_mol_properties(dataset, 'backbone_dihedrals')['backbone_dihedrals']
    mol_backbone_dihedrals = mol_backbone_dihedrals.unflatten(dim=0, sizes=(len(dataset), -1))  # shape=(N,M,2)
    if not mol_backbone_dihedrals[0].eq(dataset[0].backbone_dihedrals).all():
        raise ValueError(f"Unflattening assumption is incorrect.")
    mol_phi, mol_psi = mol_backbone_dihedrals.unsqueeze(-2).unbind(-1)

    # Convert region boundaries to torch.tensor
    # N = number of molecules
    # M = number of dihedral pairs (= number of residues - 2)
    region_boundaries = DIHEDRAL_STATE_REGIONS[method][res_type]
    region_state_ID = torch.tensor(region_boundaries['state_idx'])  # shape=(n,) for n regions. Non-unique
    region_phi_bounds = torch.tensor(region_boundaries['phi'])  # shape=(n,2). state_phi_bounds[i] = [min,max]
    region_psi_bounds = torch.tensor(region_boundaries['psi'])  # shape=(n,2). region_psi_bounds[i] = [min,max]

    # Find region index of each pair of backbone dihedral angle
    backbone_dihedrals_region_logind = (mol_phi >= region_phi_bounds[:, 0]) & (mol_phi <= region_phi_bounds[:, 1])
    backbone_dihedrals_region_logind &= (mol_psi >= region_psi_bounds[:, 0]) & (mol_psi <= region_psi_bounds[:, 1])

    # Check that all pair of dihedral angle is assigned to exactly 1 region in the Ramachandran plot
    mol_dihedral_pair_n_regions = backbone_dihedrals_region_logind.sum(dim=-1)
    mol_has_no_regions = mol_dihedral_pair_n_regions.lt(1).any(-1)
    mol_has_mult_regions = mol_dihedral_pair_n_regions.gt(1).any(-1)
    if mol_has_mult_regions.any():
        raise ValueError(f"{mol_has_mult_regions.sum()} molecules have dihedral angles assigned to more than 1 region.")
    if mol_has_no_regions.any():
        raise ValueError(f"{mol_has_no_regions.sum()} molecules have dihedral angles assigned to 0 region.")

    # Find state ID of each pair of backbone dihedral angle
    backbone_dihedrals_region_ind = backbone_dihedrals_region_logind.float().argmax(-1)
    mol_backbone_dihedrals_state_ID = region_state_ID[backbone_dihedrals_region_ind]  # shape=(N,M)

    # Generate random re-orderings of the molecules
    mol_ind_orderings = generate_mol_orderings(dataset, n_orderings=n_resamples, seed=seed)

    # Determine the number of resamples that remain to be calculated
    resample_start_ind = n_unique_dihedral_states_resample_is_zero.nonzero()[0][0]
    resample_ind_iter = range(resample_start_ind, n_resamples)
    tqdm_desc = f'Calculating number of unique dihedral angle states in {dataset.name!r}'
    resample_ind_iter = tqdm.tqdm(resample_ind_iter, desc=tqdm_desc)

    # Calculate the number of unique dihedral states for each re-ordering of the dataset's molecules
    for i in resample_ind_iter:
        mol_ind_reorder = mol_ind_orderings[i]
        unique_states = set()
        for j in range(len(subsets_size)):
            start_ind, end_ind = 0 if j == 0 else subsets_size[j - 1], subsets_size[j]
            new_mol_ind = mol_ind_reorder[start_ind:end_ind]
            new_states_ID = mol_backbone_dihedrals_state_ID[new_mol_ind]
            unique_states |= set(tuple(state) for state in new_states_ID.tolist())
            n_unique_dihedral_states[j, i] = len(unique_states)

    # Checkpoint
    if save_checkpoint:
        # Optimize dtype since numbers are positive integers
        checkpoints[checkpoint_key] = cast_to_optimal_dtype(checkpoints[checkpoint_key], max=subset_size_max)

        checkpointer.dictionary |= checkpoints
        checkpointer.save(force=True)

    return subsets_size, n_unique_dihedral_states[:, :n_resamples]


if __name__ == '__main__':
    # dataset = load_dataset('nup98_12_debug')
    # dataset = load_dataset('nup98_12')

    # # Test calculate_gmx_energy()
    # calc_dir = dataset.calc_dir / 'gmx_energy_test'
    # calculate_gmx_energy(dataset, calc_dir=calc_dir)

    # Test calculate_dihedral_sampling_rate()
    # dataset = load_dataset('Diff_CS1_HA1_nup98_12_1_C1H1', element_class=heavy_atoms_class_name)
    dataset = load_dataset('Diff_CS1_HA1_nup98_24_1_C1H1', element_class=heavy_atoms_class_name)
    subsets_size, n_dihedral_states = calculate_dihedral_sampling_rate(dataset, resume=False, save_checkpoint=False)
    dataset_train = load_dataset(dataset.model_dataset, element_class=heavy_atoms_class_name).load_splits('train', '1')
    subsets_size, n_dihedral_states_train = calculate_dihedral_sampling_rate(dataset_train, resume=False)

    # # Test UMAP
    # dataset = load_dataset()
    # dataset_valid = dataset.load_splits('valid')
    # UMAP_stats = calculate_statistics(dataset_valid, 'dihedral_angle_UMAP', recalculate=True, recalculate_prop=True,
    #                                   save=False)

    # # Test calculate_chig_fold_state
    # dataset = load_dataset('CLN_1')
    # CLN_fold_state = calculate_chig_fold_state(dataset)

    # # Test calculation of JS distance
    # dataset = load_dataset('nup98_12_debug', element_class=heavy_atoms_class_name)
    # dataset_train = dataset.load_splits('train')
    # JS_dist_dict = calculate_subsets_JS_dist(dataset_train, resume=False)

    # # dataset = load_dataset('nup98_12', element_class=heavy_atoms_class_name)
    # dataset = load_dataset('CLN025', element_class=heavy_atoms_class_name)
    # dataset_train = dataset.load_splits('train')
    # JS_dist_dict = calculate_subsets_JS_dist(dataset_train, resume=True, save_checkpoint=False)
    # # JS_dist_dict = calculate_subsets_JS_dist(dataset_train, properties_name='R_g', resume=False, save_checkpoint=False)
    # # JS_dist_dict = calculate_subsets_JS_dist(dataset_train, properties_name='backbone_dihedrals', resume=False,
    # #                                          save_checkpoint=False)
    # # JS_dist_dict = calculate_subsets_JS_dist(dataset_train, properties_name='bond_length', resume=False,
    # #                                          save_checkpoint=False)

    # Test calculate_rmsd_min
    dataset = load_dataset('CLN_1').load_splits('train')
    pdb_dataset = load_dataset('PDB_2RVD', element_class=dataset.element_class)
    rmsd_val = calculate_rmsd_min(dataset, pdb_dataset, checkpoint_filepath='rmsd_min_2rvd_debug')
    pass
