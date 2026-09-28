from __future__ import annotations
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
import torch
import tqdm

import config

if TYPE_CHECKING:
    from data.data_classes import Biomolecule
    from data.datasets import BiomoleculeDataset

all_atoms_class_name = 'all_atoms'
supported_adj_methods = ['CA', 'CA_SI', 'RES', 'RES_CENT', 'RES_CA']  # Supported adjacency methods


@dataclass
class CoarseGrains:
    type: str  # type of grain
    geometries: dict[str, list[str, torch.Tensor]]  # format: dict['residue_name',['center_atom_name','radius'[nm]])

    def __post_init__(self):
        self.geometries = {k: self.geometries[k] for k in sorted(self.geometries.keys())}

    @property
    def centers_name(self):
        return {k: v[0] for k, v in self.geometries.items()}

    @property
    def radii(self):
        return {k: v[1] for k, v in self.geometries.items()}


@dataclass
class CoarseGrainsSets:
    """Dictionary-like dataclass to store information on sets of coarse grains for various datasets and grain types"""
    filename: str
    filepath: pathlib.Path = None
    grains_sets: dict[(str, str), CoarseGrains] = None  # dict[(dataset_name,grain_type),CoarseGrains]

    def __post_init__(self):
        self.filepath = config.data_dir / self.filename
        self.grains_sets = dict()
        self.load()

    def __getitem__(self, item: tuple[str, str]):
        if item not in self.grains_sets:
            self.calculate_grains_geometries(*item)
        return self.grains_sets[item]

    def rep_atom_name(self, item: tuple | dict):
        if isinstance(item, tuple):
            rep_name_dict = {k: v[0] for k, v in self[item].items()}
        elif isinstance(item, dict):
            rep_name_dict = {k: v[0] for k, v in item.items()}
        else:
            raise ValueError
        return rep_name_dict

    def calculate_grains_geometries(self, dataset_name: str, grain_type: str):
        supported_grain_types = ['sidechain', 'sidechain_heavy', 'residue_CA', 'residue', 'residue_centroid']
        if grain_type not in supported_grain_types:
            raise NotImplementedError(f"grain_type {grain_type!r} is not supported.\n"
                                      f"Supported choices:{supported_grain_types}")

        from data.datasets import load_dataset

        # Load the dataset with all-atom structures
        dataset = load_dataset(name=dataset_name, debug=False, element_class=all_atoms_class_name)
        if dataset.has_splits:
            dataset = dataset.load_splits('train')  # Use the train split to calculate the grains geometries
        packings = calculate_optimal_packings(dataset, grain_type=grain_type, verbose=True)

        # Add grains for capped residues (AMBER)
        if 'ACE' not in packings:
            packings['ACE'] = ['CH3', torch.tensor(0.3)]
        if 'NME' not in packings:
            packings['NME'] = ['CH3', torch.tensor(0.3)]

        grains = CoarseGrains(type=grain_type, geometries=packings)
        self.grains_sets[(dataset_name, grain_type)] = grains
        self.save(verbose=True)

    def get_adj_grains(self, method: str | None, dataset_name: str | None = None) -> CoarseGrains | None:
        """
        Return the coarse grains used for the given adjacency method
        Args:
            method: name of the adjacency method
            dataset_name (optional): name of the dataset used to define the coarse grains

        Returns:
            CoarseGrains object defining the coarse grains geometrics of the associated adjacency method

        """
        # Return none if no method is given
        if method is None or method == 'CA':
            return None

        # Default dataset_name is
        if dataset_name is None:
            dataset_name = config.coarse_grains_dataset

        if method == 'CA_SI':
            return self[(dataset_name, 'sidechain_heavy')]
        elif method == 'RES':
            return self[(dataset_name, 'residue')]
        elif method == 'RES_CENT':
            return self[(dataset_name, 'residue_centroid')]
        elif method == 'RES_CA':
            return self[(dataset_name, 'residue_CA')]
        else:
            raise NotImplementedError

    def load(self):
        if self.filepath.exists():
            with np.load(self.filepath, allow_pickle=True) as file:
                loaded_dict = dict(file)
            grains = {}
            for k, v in loaded_dict.items():
                grain_key = tuple(k.split(','))
                grains[grain_key] = v.item()
            self.grains_sets = grains

    def save(self, verbose=False):
        if verbose:
            print(f'Saving packings in {str(self.filepath)!r} for {list(self.grains_sets.keys())}.')
        saved_dict = {','.join(k): v for k, v in self.grains_sets.items()}
        np.savez(self.filepath, **saved_dict)

    def print(self):
        sorted_grains_sets = {k: self.grains_sets[k] for k in sorted(self.grains_sets.keys())}
        with open(self.filepath.with_suffix('.txt'), 'w') as f:
            for grain_set_name, grains_set in sorted_grains_sets.items():
                f.write(f"{grain_set_name}\n")
                for grain_name, grain_geometry in grains_set.geometries.items():
                    f.write(f"\t{grain_name:<4}:{grain_geometry[0]:<3},{float(grain_geometry[1]):.2f}\n")


def calculate_heavy_packings(mol: Biomolecule):
    """
    Iterates through each heavy atom in the given molecule and finds the length of the longest bond for each heavy atom.
    Peptide bonds are ignored.
    Args:
        mol: Biomolecule for which the search will be performed
    Returns:
        torch.Tensor: radius for each residue
    """
    assert mol.bonds_ind is not None, "bond info is needed to calculate the heavy atoms packings"
    bonds_ind = mol.bonds_ind
    bonds_length = mol.calculate_bonds_length()

    # Remove peptide bonds since they can be erroneously large
    is_bond_peptide = (mol.elements_name[bonds_ind[:, 0]] == 'C') & (mol.elements_name[bonds_ind[:, 1]] == 'N')
    is_bond_peptide |= (mol.elements_name[bonds_ind[:, 1]] == 'C') & (mol.elements_name[bonds_ind[:, 0]] == 'N')
    bonds_ind = bonds_ind[~is_bond_peptide]
    bonds_length = bonds_length[~is_bond_peptide]

    elem_bond_length_max = torch.zeros(mol.n_elements)
    elem_bond_length_max.scatter_reduce_(dim=0, index=bonds_ind[:, 0], src=bonds_length, reduce='amax',
                                         include_self=False)
    elem_bond_length_max.scatter_reduce_(dim=0, index=bonds_ind[:, 1], src=bonds_length, reduce='amax',
                                         include_self=True)
    heavy_atom_ind = (~mol.elements_is_H).nonzero()[:, 0]
    packings_res_name = mol.residues_name[mol.elements_resind[heavy_atom_ind]].tolist()
    packings_atom_name = mol.elements_name[heavy_atom_ind]
    packings_radius = elem_bond_length_max[heavy_atom_ind]

    return packings_res_name, packings_atom_name, packings_radius


def calculate_sidechain_packings(mol: Biomolecule, center_type: str = None):
    """
    Iterates through each sidechain in the given molecule and finds the largest radius around a given sidechain atom
    that contains all other sidechain atoms
    Args:
        mol: Biomolecule for which the search will be performed
        center_type: Type of atoms considered for the center. Choices are [None,'HA'].

    Returns:
        list: residue name for each packing
        list: atom name for each packing center
        torch.Tensor: radius of each packing
    """
    # Iterates through each residue and calculates the smallest radius centered around each sidechain atoms that contains the entire sidechain.
    elem_is_backbone = mol.elements_is_backbone
    packings_res_name = list()
    packings_atom_name = list()
    packings_radius = list()

    for i, res_name in enumerate(mol.residues_name):
        # For Glycine, use the 'C' backbone atom as the representative to simplify logic and data structure
        if res_name == 'GLY':
            packings_res_name.extend([res_name])
            packings_atom_name.extend('C')
            packings_radius.append(torch.tensor(0.0).unsqueeze(0))
            continue
        is_elem_selected = torch.eq(mol.elements_resind, i) & ~elem_is_backbone
        side_elems_name = mol.elements_name[is_elem_selected]
        side_elems_pos = mol.elements_position[is_elem_selected, :]
        n_sidechain_atoms = side_elems_pos.shape[0]

        # Calculate the radius of a sphere centered at each sidechain atom that contains all other sidechain atoms
        if center_type is None:
            side_elems_pdist = torch.pdist(side_elems_pos)
            triu_ind = torch.triu_indices(n_sidechain_atoms, n_sidechain_atoms, offset=1)
            sphere_packing_radius = torch.zeros(n_sidechain_atoms)
            sphere_packing_radius.scatter_reduce_(dim=0, index=triu_ind[0], src=side_elems_pdist, reduce='amax')
            sphere_packing_radius.scatter_reduce_(dim=0, index=triu_ind[1], src=side_elems_pdist, reduce='amax')
            centers_name = side_elems_name
        elif center_type == 'HA':
            # Select only heavy atoms as possible centers
            is_elem_H = np.array([s.startswith('H') for s in side_elems_name])
            mask = torch.ones((n_sidechain_atoms, n_sidechain_atoms), dtype=torch.bool)
            mask[is_elem_H, :] = False
            pair_ind = mask.nonzero()

            pair_dist = torch.diff(side_elems_pos[pair_ind, :], dim=-2).norm(dim=-1).squeeze(-1)
            sphere_packing_radius = torch.nan * torch.ones(n_sidechain_atoms)
            sphere_packing_radius.scatter_reduce_(dim=0, index=pair_ind[:, 0], src=pair_dist, reduce='amax',
                                                  include_self=False)
            centers_name = side_elems_name[~is_elem_H]
            sphere_packing_radius = sphere_packing_radius[~is_elem_H]  # Removes all nans if indexing is correct
        else:
            raise NotImplementedError

        # Add sidechain packings to the list
        packings_res_name.extend(sphere_packing_radius.numel() * [res_name])
        packings_atom_name.extend(list(centers_name))
        packings_radius.append(sphere_packing_radius)

    packings_radius = torch.cat(packings_radius)
    return packings_res_name, packings_atom_name, packings_radius


def calculate_residue_packings(mol: Biomolecule, center_type: str = None):
    """
    Iterates through each sidechain in the given molecule and finds the radius of the largest sphere around
    each sidechain atom that contains all other sidechain atoms
    Args:
        mol: Biomolecule for which the search will be performed
        center_type: type of atoms considered for the center. Choices are [None,'HA'].

    Returns:
        list: residue name for each packing
        list: atom name for each packing center
        torch.Tensor: radius of each packing
    """

    # Iterates through each residue and calculates the smallest radius centered around a given atom that contains all residue atoms
    packings_res_name = list()
    packings_atom_name = list()
    packings_radius = list()
    for i, res_name in enumerate(mol.residues_fullname):
        elem_is_in_res = torch.eq(mol.elements_resind, i)
        res_elems_name = mol.elements_name[elem_is_in_res]
        res_elems_pos = mol.elements_position[elem_is_in_res, :]

        # Calculate the radius of a sphere centered at each atom that contains all other atoms
        if center_type is None:
            n_elems = res_elems_pos.shape[0]
            elems_pdist = torch.pdist(res_elems_pos)
            triu_ind = torch.triu_indices(n_elems, n_elems, offset=1)
            res_packing_radius = torch.zeros(n_elems)
            res_packing_radius.scatter_reduce_(dim=0, index=triu_ind[0], src=elems_pdist, reduce='amax')
            res_packing_radius.scatter_reduce_(dim=0, index=triu_ind[1], src=elems_pdist, reduce='amax')
        elif center_type == 'centroid':
            n_elems = 1
            res_elems_name = ['centroid']
            res_centroid = res_elems_pos.mean(dim=-2, keepdim=True)
            res_packing_radius = torch.linalg.norm(res_elems_pos - res_centroid, dim=-1).max().unsqueeze(0)
        else:
            n_elems = 1
            elem_is_center = res_elems_name == center_type
            res_elems_name = res_elems_name[elem_is_center]
            res_packing_radius = torch.linalg.norm(res_elems_pos - res_elems_pos[elem_is_center],
                                                   dim=-1).max().unsqueeze(0)

        # Add packings to the list
        packings_res_name.extend(n_elems * [res_name])
        packings_atom_name.extend(list(res_elems_name))
        packings_radius.append(res_packing_radius)

    packings_radius = torch.cat(packings_radius)
    return packings_res_name, packings_atom_name, packings_radius


def calculate_backbone_packings(mol: Biomolecule, bond_ind: torch.Tensor = None):
    """
    Iterates through each backbone in the given molecule and finds the radius of the largest sphere around the C-alpha
     atom that contains all other backbone atoms.
    Args:
        mol: Biomolecule for which the search will be performed
        bond_ind: nx2 array identifying bonds in the given Biomolecule. Required when mol.element_class='all_atoms'.

    Returns:
        torch.Tensor: radius for each residue
    """
    elem_is_heavy_backbone = mol.elements_is_backbone

    # If the molecule is all-atom, also include hydrogen atoms attached to backbone atoms
    if mol.element_class == all_atoms_class_name:
        is_elem_H = mol.elements_is_H
        backbone_bond_matrix = torch.zeros((mol.n_elements, mol.n_elements), dtype=torch.bool)
        backbone_bond_matrix[bond_ind.unbind(-1)] = True
        backbone_bond_matrix[bond_ind.flip(dims=[-1]).unbind(-1)] = True
        backbone_bond_matrix[:, ~elem_is_heavy_backbone] = False
        is_H_elem_attached_to_backbone = backbone_bond_matrix.any(dim=-1) & is_elem_H
        elem_is_backbone = elem_is_heavy_backbone | is_H_elem_attached_to_backbone
    else:
        elem_is_backbone = elem_is_heavy_backbone

    CA_atoms_ind = mol.elements_is_CA.nonzero()[:, 0]
    packings_radius = torch.zeros(CA_atoms_ind.shape)
    for i, CA_atom_ind in enumerate(CA_atoms_ind):
        CA_resind = mol.elements_resind[CA_atom_ind]
        CA_atom_pos = mol.elements_position[CA_atom_ind]
        res_backbone_elems = torch.eq(mol.elements_resind, CA_resind) & elem_is_backbone
        backbone_elems_pos = mol.elements_position[res_backbone_elems, :]
        backbone_elem_dist = torch.linalg.norm(backbone_elems_pos - CA_atom_pos, dim=-1)
        packings_radius[i] = backbone_elem_dist.max()

    return packings_radius


def calculate_optimal_packings(dataset: BiomoleculeDataset, grain_type='sidechain', radius_prec=0.01, verbose=False) \
        -> dict[str, [str, torch.Tensor]]:
    """
    Finds the optimal way to pack a given set of atoms in a given dataset of protein structures
    Args:
        dataset: dataset used to calculate the packings
        grain_type: type of coarse grain to pack.
                    Choices are: ['sidechain','sidechain_heavy','residue','residue_centroid','heavy_atoms']
        radius_prec: precision (nm) of the calculated radii. Radii are rounded up to the highest multiple of this value.
        verbose: display calculation progress

    Returns:
        dictionary of residue names where the values correspond to the atom name of the center and the radius
    """
    packings_radius_max = defaultdict(lambda: defaultdict(lambda: torch.tensor(0.0)))
    if verbose:
        mol_iter = tqdm.tqdm(dataset, desc=f'Calculating optimal {grain_type!r} packings for {dataset.name!r}')
    else:
        mol_iter = dataset

    # Determine the bond indices of the molecule
    if grain_type == 'backbone_CA':
        from chemistry.energy import BondEnergy
        bond_energy_mod = BondEnergy(top_file=dataset.top_filepath, element_class=dataset.element_class)

    # For each residue type and each atom name, find the largest sphere that encapsulates all other atoms belonging to
    # the given grain.
    for mol in mol_iter:
        mol: Biomolecule
        if grain_type == 'sidechain':
            pack_res_name, pack_atom_name, pack_radius = calculate_sidechain_packings(mol)
        elif grain_type == 'sidechain_heavy':
            pack_res_name, pack_atom_name, pack_radius = calculate_sidechain_packings(mol, center_type='HA')
        elif grain_type == 'residue':
            pack_res_name, pack_atom_name, pack_radius = calculate_residue_packings(mol)
        elif grain_type == 'residue_CA':
            pack_res_name, pack_atom_name, pack_radius = calculate_residue_packings(mol, center_type='CA')
        elif grain_type == 'residue_centroid':
            pack_res_name, pack_atom_name, pack_radius = calculate_residue_packings(mol, center_type='centroid')
        elif grain_type == 'heavy_atoms':
            pack_res_name, pack_atom_name, pack_radius = calculate_heavy_packings(mol)
        elif grain_type == 'backbone_CA':
            pack_res_name, pack_atom_name = mol.residues_name, mol.elements_name[mol.elements_is_CA]
            pack_radius = calculate_backbone_packings(mol, bond_ind=bond_energy_mod.bonds_atom_ind)
        else:
            raise NotImplementedError(f"Grain type {grain_type!r} is not implemented.")

        # Find the maximum radius needed to cover all atoms for each candidate center and each residue
        for res_name, atom_name, radius in zip(pack_res_name, pack_atom_name, pack_radius):
            packings_radius_max[res_name][atom_name] = torch.maximum(packings_radius_max[res_name][atom_name], radius)

    # Ceil radii
    for res_name, packings in packings_radius_max.items():
        for atom_name, atom_radius in packings.items():
            packings[atom_name] = torch.ceil(atom_radius / radius_prec) * radius_prec

    # Format output
    packings_info = dict()
    if grain_type == 'heavy_atoms':
        # Return a dictionary with keys: "RES_ATOM" that determine the maximal size of each heavy atom in each residue.
        for res_name, packings in packings_radius_max.items():
            for atom_name, atom_radius in packings.items():
                packings_info[f"{res_name}_{atom_name}"] = atom_radius
    else:
        # For each residue, find the packing center (i.e. atom name) that has the smallest radius
        for res_name, packings in packings_radius_max.items():
            packings_center_name = list(packings.keys())
            packings_radius = torch.tensor(list(packings.values()))
            smallest_rad_ind = torch.argmin(packings_radius).item()
            packings_info[res_name] = [packings_center_name[smallest_rad_ind], packings_radius[smallest_rad_ind]]

    return packings_info


coarse_grains_sets = CoarseGrainsSets(filename='coarse_grains.npz')

if __name__ == '__main__':
    coarse_grains_sets.print()

    # Test calculation of packings
    from data.datasets import load_dataset

    rich_dataset = load_dataset('Rich_2018')
    rich_dataset.subset(ind=torch.arange(100))
    # calculate_optimal_packings(rich_dataset, 'sidechain_heavy', verbose=True)
    packings = calculate_optimal_packings(rich_dataset, 'heavy_atoms', verbose=True)
    pass
