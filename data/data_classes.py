import copy
import dataclasses
import os
import tempfile
import typing
import warnings
import random
import pathlib
import uuid
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import torch
import MDAnalysis as mda
import MDAnalysis.coordinates.memory

from data.forcefields import Forcefield, pdb_terminal_OO_names, charmm_C_terminal_O_names, amber_C_terminal_O_names, \
    charmm36m, forcefields, canonical_aa_codes_3, force_field_atom_types, force_fields_heavy_atom_types
from chemistry.grains import coarse_grains_sets, CoarseGrains
from utils import align_PC, remove_centroid, rmsd, rmsd_align, calc_dihedral_angle, calc_bond_angle, \
    calculate_units_conversion_factor, pint_reg, def_atom_name_to_type, uni_from_seq, gmx_pdb2gmx, parse_gmx_header, \
    RigidBodyMotion
import config


@dataclass
class Biomolecule:
    """Dataclass to store data related to a biomolecule including its elements positions"""
    ID: str = None  # ID of the molecule
    model_ID: str = ''  # ID of a model molecule from which some of the properties were derived (when applicable)
    device: torch.device = torch.device('cpu')
    element_class: str = ''  # Class of the stored elements
    forcefield: str = ''  # Forcefield associated with the molecule. Used to define element types
    top_file: str | None = None  # Filepath of the topology that defines the molecule
    length_units: str = ''  # Units of length used to store properties with length dimension
    n_elements: int = None  # Number of elements in the biomolecule
    elements_name: NDArray[np.str_] = None  # Name of each element
    elements_type: list[str] = None  # Type of the elements
    elements_position: torch.Tensor = None  # 3D position of each element
    elements_mass: torch.Tensor = None  # Mass of each element
    elements_charge: torch.Tensor = None  # Charge of each element
    elements_resind: torch.Tensor = None  # Residue index of each element
    residues_name: NDArray[np.str_] = None  # Name of residues
    residues_id: torch.Tensor = None  # ID of residues
    bonds_ind: torch.Tensor = None  # shape=(n_bonds x 2). Atom indices of each pair of atoms that form a bond

    def __post_init__(self):
        self.n_elements = self.elements_position.size(0)

        # Ensure dtypes for certain attributes
        self.elements_position = self.elements_position.float()
        self.elements_name = np.asarray(self.elements_name, dtype=str)
        self.residues_name = np.asarray(self.residues_name, dtype=str)

        # Check that the element types are defined for the given element class
        if self.element_class not in supported_types:
            raise ValueError(f'Default encoded types are undefined for element_class={self.element_class}')

        # Convert charge to a float32 tensor
        if self.elements_charge is not None:
            self.elements_charge = self.elements_charge.float()

        # Check that the element types are all defined.
        if self.elements_type is not None:
            supported_elem_types = self.supported_types('element')
            unsupported_elem_types = set(self.elements_type) - set(supported_elem_types)
            if unsupported_elem_types:
                raise ValueError(f'{self.ID!r} has unsupported element types.\n'
                                 f'Unsupported types:{unsupported_elem_types}\nSupported types:{supported_elem_types}')

        # Assign a random ID if none is given
        if self.ID is None:
            self.ID = uuid.uuid4().hex[:16]

        if self.bonds_ind is not None:
            # bonds_ind_dtype = torch.uint16
            bonds_ind_dtype = torch.int16
            n_elem_max = torch.iinfo(bonds_ind_dtype).max
            assert self.n_elements <= n_elem_max, f"Cannot use {bonds_ind_dtype}. n_elem={self.n_elements}>{n_elem_max}"
            self.bonds_ind = torch.as_tensor(self.bonds_ind, dtype=bonds_ind_dtype)

    def rename_old_attrs(self):
        old_to_new_names = {}
        for old_name, new_name in old_to_new_names.items():
            if hasattr(self, old_name):
                warnings.warn(f"found old attribute name {old_name}. Renaming attribute to {new_name}.")
                setattr(self, new_name, getattr(self, old_name))
                delattr(self, old_name)

    @classmethod
    def random(cls, n_elements=None, element_class=None, **kwargs):
        """
        Creates a random Biomolecule with random properties
        Args:
            n_elements: number of elements in the molecule. Default=random integer in [0,10]

        Returns:
            Biomolecule
        """
        if n_elements is None:
            n_elements = random.randint(1, 10)
        if element_class is None:
            element_class = c_alpha_class_name
        if element_class in [heavy_atoms_class_name, all_atoms_class_name]:
            supported_elem_types = supported_types[element_class][charmm36m.name]
        else:
            supported_elem_types = supported_types[element_class]
        elements_pos = torch.randn(n_elements, 3)
        elements_type = random.choices(supported_elem_types, k=n_elements)
        elements_name = elements_type
        elements_mass = 5 * torch.randn(n_elements).abs()
        return cls(elements_type=elements_type, elements_name=elements_name,
                   elements_position=elements_pos, elements_mass=elements_mass, element_class=element_class, **kwargs)

    @classmethod
    def from_universe(cls, md_uni: mda.Universe, element_class: str, atom_name_to_type: dict = None,
                      topology: str | mda.Universe = None, sidechain_rep_name: dict[str, str] = None, save_bonds=False,
                      save_types=True, **constructor_kwargs):
        """
        Constructs a Biomolecule from a MDAnalysis.Universe object
        Args:
            md_uni: MDAnalysis.Universe object
            element_class: element_class of the output biomolecule
            atom_name_to_type: dictionary that maps atom names to their types. md_uni.atoms.types has errors
            topology: MDAnalysis.Universe with topology attributes or .top filepath. Used to define atom_name_to_type (if undefined).
            sidechain_rep_name: dictionary that identifies the atom name of the sidechain coarse grain (values) for each residue (keys).
                                This is needed only when element_class = c_alpha_sidechain_class_name
            save_bonds: save the index of the pair of atoms that form each bond into self.bonds_ind
            save_types: save atom types gathered from the input universe
            **constructor_kwargs:

        Returns:
            Biomolecule
        """
        # Select only the protein
        prot_sel = md_uni.select_atoms(prot_sel_crit)
        residues_name = prot_sel.residues.resnames.copy()
        residues_id = torch.tensor(prot_sel.residues.resids)
        bonds_ind = None

        # Initialize dictionary that maps atom names to atom types (.tpr files created with convert-tpr have errors)
        if topology and not atom_name_to_type:
            atom_name_to_type = def_atom_name_to_type(topology=topology)

        if element_class == c_alpha_class_name:
            atoms_sel = prot_sel.select_atoms('name CA')
            elements_name = atoms_sel.names
            elements_type = residues_name if save_types else None
            elements_resid = torch.tensor([r.resid for r in prot_sel.residues])
            elements_mass = torch.from_numpy(atoms_sel.masses)
            elements_charge = torch.from_numpy(atoms_sel.charges) if hasattr(atoms_sel, 'charges') else None
            elements_coordinates = torch.from_numpy(atoms_sel.positions)
        elif element_class == c_alpha_sidechain_class_name:
            # Determine which atoms are sidechain representatives using the given dictionary
            n_res = len(residues_name)
            is_atom_in_res = torch.eq(torch.tensor(prot_sel.resids) - 1,
                                      torch.arange(n_res).unsqueeze(-1))  # shape=(n_res,n_atoms)
            res_rep_name = np.array([sidechain_rep_name[res_name] for res_name in residues_name])
            is_atom_rep = torch.tensor(np.equal(prot_sel.names, res_rep_name[:, None]))  # shape=(n_res,n_atoms)
            is_atom_rep = torch.logical_and(is_atom_in_res, is_atom_rep)

            # Ensure that there is only one sidechain representative for each residue
            N_rep_per_res = is_atom_rep.sum(-1)
            assert N_rep_per_res.eq(1).all(), f'More than 1 representative was found for some residues {N_rep_per_res}'

            # Select the CA atoms and the sidechain representatives
            is_atom_CA = torch.tensor(np.equal(prot_sel.names, ['CA']))
            is_atom_sel = torch.logical_or(is_atom_CA, is_atom_rep.any(dim=0))
            atoms_sel = prot_sel.atoms[is_atom_sel]

            # Define other elements properties
            elements_name = atoms_sel.names
            if not save_types:
                elements_type = None
            elif atom_name_to_type:
                elements_type = [atom_name_to_type[(a.resindex, a.name)] for a in atoms_sel]
            else:
                elements_type = list(atoms_sel.types)
            elements_resid = torch.tensor([a.resid for a in atoms_sel])
            elements_mass = torch.from_numpy(atoms_sel.masses)
            elements_charge = torch.from_numpy(atoms_sel.charges) if hasattr(atoms_sel, 'charges') else None
            elements_coordinates = torch.from_numpy(atoms_sel.positions)

        elif element_class in [heavy_atoms_class_name, all_atoms_class_name]:
            if element_class == heavy_atoms_class_name:
                atoms_sel = prot_sel.select_atoms('not name H*')
            else:
                atoms_sel = prot_sel

            # Find atom indices of all bonds in the molecule if bonds are saved
            if save_bonds:
                bonds_ind = atoms_sel.bonds.indices
                if element_class == heavy_atoms_class_name:
                    bond_matrix = np.zeros((len(prot_sel), len(prot_sel)), dtype=bool)
                    bond_matrix[bonds_ind[:, 0], bonds_ind[:, 1]] = True
                    bond_matrix = bond_matrix[atoms_sel.atoms.indices[:, None], atoms_sel.atoms.indices[None, :]]
                    bonds_ind = np.stack(bond_matrix.nonzero(), axis=1)

            elements_name = atoms_sel.names

            # Define other elements properties
            if not save_types:
                elements_type = None
            elif atom_name_to_type:
                elements_type = [atom_name_to_type[(a.resindex, a.name)] for a in atoms_sel]
            else:
                elements_type = list(atoms_sel.types)
            elements_resid = torch.tensor([a.resid for a in atoms_sel])
            elements_mass = torch.from_numpy(atoms_sel.masses)
            elements_charge = torch.from_numpy(atoms_sel.charges) if hasattr(atoms_sel, 'charges') else None
            elements_coordinates = torch.from_numpy(atoms_sel.positions)
        else:
            raise NotImplementedError(f'element_class={element_class} is not implemented.')
        elements_mass = elements_mass.float()

        # Define the residue index of each element
        if not torch.all(torch.diff(residues_id) >= 0):
            raise ValueError("Cannot determine elements index when residue IDs are not monotonically increasing.")
        elements_resind = torch.bucketize(elements_resid.contiguous(), boundaries=residues_id, right=False)

        # Set the centroid of the structure to the origin
        elements_coordinates -= elements_coordinates.mean(dim=-2)

        biomolecule = cls(length_units=mda.coordinates.XTC.XTCReader.units['length'],
                          element_class=element_class,
                          elements_name=elements_name,
                          elements_type=elements_type,
                          elements_resind=elements_resind,
                          elements_position=elements_coordinates,
                          elements_mass=elements_mass,
                          elements_charge=elements_charge,
                          residues_name=residues_name,
                          residues_id=residues_id,
                          bonds_ind=bonds_ind,
                          **constructor_kwargs)
        return biomolecule

    @classmethod
    def from_gro(cls, gro_file: pathlib.Path | str, element_class=None, forcefield=None, save_bonds=False,
                 **kwargs):
        """
        Constructs a Biomolecule from a GROMACS .gro file
        Args:
            gro_file: filepath of the .gro file
            element_class: element class of the output molecule
            forcefield: forcefield associated with the molecule
            save_bonds: save the index of the pair of atoms that form each bond into self.bonds_ind. A .top file in the same directory is needed.
            **kwargs: extra kwargs passed to from_universe() constructor

        Returns:
            Biomolecule
        """
        if element_class is None:
            element_class = all_atoms_class_name
        gro_file = pathlib.Path(gro_file)

        if save_bonds:
            # Check if there is a .top file in the same directory as the .gro file
            top_file = gro_file.with_suffix('.top')
            if not top_file.exists():
                raise FileNotFoundError(f"When save_bonds=True, a .top file is needed.\n"
                                        f"The .top file {top_file} was not found.")

            uni = mda.Universe(top_file, gro_file, dt=1.0, convert_units=False, topology_format='ITP',
                               include_dir=config.data_dir)

            # If no forcefield was given, find the one used in the .top file
            if forcefield is None:
                top_header_attrs = parse_gmx_header(filepath=top_file)
                forcefield_name = top_header_attrs['forcefield']
                forcefield = next(f.name for f in forcefields.values() if f.gmx_name == forcefield_name)
        else:
            uni = mda.Universe(str(gro_file), dt=1.0, convert_units=False)
            if forcefield is None:
                forcefield = charmm36m.name

        # Transfer coordinates to Memory to avoid too many unclosed files.
        uni.transfer_to_memory()

        biomolecule = cls.from_universe(md_uni=uni, element_class=element_class, ID=gro_file.stem,
                                        forcefield=forcefield, save_bonds=save_bonds, **kwargs)
        return biomolecule

    @classmethod
    def from_topology(cls, top_filepath: str | pathlib.Path, element_class: str, forcefield: str = None,
                      **constructor_kwargs):
        """
        Constructs a Biomolecule of given element class from a GROMACS .top file

        Args:
            top_filepath: filepath of the GROMACS .top file
            element_class: element class for the constructed Biomolecule
            forcefield: forcefield used to define the types of the elements. Defaults to forcefield defined in .top file
            **constructor_kwargs: extra kwargs passed to the init() method

        Returns:
            a biomolecule whose elements match those found in the given .top file
        """

        with warnings.catch_warnings(category=UserWarning, action="ignore"):
            top_uni = mda.Universe(top_filepath, topology_format='ITP', include_dir=config.data_dir)

        # Define the elements coordinates
        if 'elements_position' in constructor_kwargs:
            elements_position = constructor_kwargs.pop('elements_position')
        else:
            elements_position = np.zeros((len(top_uni.atoms), 3))
        top_uni.load_new(elements_position, format=mda.coordinates.memory.MemoryReader)

        # Define forcefield as the used in the .top file
        if forcefield is None:
            top_header_attrs = parse_gmx_header(filepath=top_filepath)
            forcefield_name = top_header_attrs['forcefield']
            forcefield = next(f.name for f in forcefields.values() if f.gmx_name == forcefield_name)

        biomolecule = cls.from_universe(md_uni=top_uni, element_class=element_class, topology=top_uni,
                                        forcefield=forcefield, **constructor_kwargs)
        top_uni.trajectory.close()
        return biomolecule

    @classmethod
    def from_sequence(cls, sequence: list[str], forcefield: Forcefield, terminals_protonation: list[str] = None,
                      element_class: str = None, **constructor_kwargs):
        """
        Constructs a Biomolecule from a given amino-acid sequence and a given force field.
        Args:
            sequence: sequence of the amino acid of the output Biomolecule
            forcefield: force field used to define the topology of the Biomolecule's residues
            terminals_protonation: protonation state of the terminals
            element_class: element_class of the output Biomolecule
            **constructor_kwargs: extra kwargs given to from_topology() constructor

        Returns:
            Biomolecule
        """
        # Create a .pdb file of the sequence in a temporary directory
        if element_class is None:
            element_class = heavy_atoms_class_name
        prot_uni = uni_from_seq(sequence=sequence, forcefield=forcefield)
        working_dir = pathlib.Path(tempfile.mkdtemp())
        basename = os.path.basename(tempfile.mktemp(dir=working_dir))
        pdb_filepath = working_dir / f"{basename}.pdb"
        top_filepath = pdb_filepath.with_suffix('.top')

        with mda.coordinates.PDB.PDBWriter(filename=str(pdb_filepath), convert_units=False) as pdb:
            pdb.write(prot_uni)

        # Run pdb2gmx on the output pdb to create a .top file with default inner protonation states and
        # the given terminals protonation states
        gmx_pdb2gmx(pdb_filepath=pdb_filepath, terminals_protonation=terminals_protonation,
                    forcefield_dir=forcefield.dir, output_basename=basename, working_dir=working_dir)

        biomol = cls.from_topology(top_filepath=top_filepath, element_class=element_class, forcefield=forcefield.name,
                                   **constructor_kwargs)
        return biomol, top_filepath

    def refine(self, mol_template: typing.Self):
        """
        Constructs a Biomolecule from a template Biomolecule that inherits the positions of all elements found in self
        Args:
            mol_template: Biomolecule template whose element class is equal or finer than self.element_class

        Returns:
            A Biomolecule whose elements positions are identical to self for elements common to both and zeroed otherwise.
        """

        self_order = element_classes.index(self.element_class)
        template_order = element_classes.index(mol_template.element_class)
        if self_order > template_order:
            raise ValueError(f"The given template element class ({mol_template.element_class!r}) is coarser "
                             f"than self.element_class ({self.element_class!r}).")

        # Copy elements position of self onto the template elements position by matching their name and residue index
        is_res_match = torch.eq(self.elements_resind[:, None], mol_template.elements_resind)
        is_name_match = torch.tensor(np.equal(self.elements_name[:, None], mol_template.elements_name))
        unmatched_ind = torch.nonzero(~is_name_match.any(dim=-1))
        if unmatched_ind.numel() > 0:
            unmatched_name = self.elements_name[unmatched_ind]
            raise ValueError(f"The following elements have not been found in the template:{unmatched_name}")

        is_atom_match = torch.logical_and(is_res_match, is_name_match)
        multiple_matches_ind = is_atom_match.sum(dim=-1).gt(1).nonzero()
        if multiple_matches_ind.numel() > 0:
            multimatched_name = self.elements_name[multiple_matches_ind]
            raise ValueError(f"The following elements have multiple matches in the template:{multimatched_name}")

        match_pair_ind = is_atom_match.nonzero()
        elements_pos = torch.nan * torch.randn_like(mol_template.elements_position)
        elements_pos[match_pair_ind[:, 1]] = self.elements_position[match_pair_ind[:, 0]]
        biomol = dataclasses.replace(mol_template, ID=f"{self.ID}_refined", elements_position=elements_pos)

        return biomol

    def coarsen(self, element_class: str, inplace=True, sidechain_rep_name: dict[str, str] = None) -> typing.Self:
        """
        Coarsens biomolecule by removing elements that do not belong to the given element class
        Args:
            element_class: element class of the coarsened molecule
            inplace: perform the coarsening inplace
            sidechain_rep_name: representative of the sidechain when element_class=c_alpha_sidechain_class_name

        Returns:
            New coarsened molecule (if inplace=False) or coarsen version of self (if inplace=True)
        """

        mol_out = self if inplace else dataclasses.replace(self)
        if element_class == self.element_class:
            return mol_out

        # Check that the new element class is coarser than the current one
        if element_classes.index(element_class) > element_classes.index(self.element_class):
            raise ValueError(f"Cannot coarsen molecule of class {self.element_class!r} to class {element_class!r}.")

        if element_class == heavy_atoms_class_name:
            is_element_kept = ~self.elements_is_H
        elif element_class == c_alpha_sidechain_class_name:
            is_element_kept = torch.zeros(self.n_elements, dtype=torch.bool)
            if sidechain_rep_name is None:
                sidechain_rep_name = coarse_grains_sets[(config.coarse_grains_dataset, 'sidechain_heavy')].centers_name
            sidechain_rep_ind = self.residue_rep_ind(residue_rep_name=sidechain_rep_name)
            is_element_kept[sidechain_rep_ind] = True

            is_element_kept |= self.elements_is_CA
        elif element_class == c_alpha_class_name:
            is_element_kept = self.elements_is_CA
        else:
            raise NotImplementedError(f"element class {element_class!r} is not implemented.")
        element_kept_ind = is_element_kept.nonzero().flatten()

        # Prune elements' properties
        elements_attrs_name = ['elements_name', 'elements_type', 'elements_position', 'elements_mass',
                               'elements_charge', 'elements_resind']
        for attr_name in elements_attrs_name:
            attr_val = getattr(self, attr_name)
            if isinstance(attr_val, list):
                attr_val = [attr_val[i] for i in element_kept_ind]
            elif isinstance(attr_val, (torch.Tensor, np.ndarray)):
                attr_val = attr_val[element_kept_ind, ...]
            elif isinstance(attr_val, dict):
                attr_val = {k: v[is_element_kept, ...] for k, v in attr_val.items()}
            setattr(mol_out, attr_name, attr_val)

        mol_out.element_class = element_class
        if element_class == c_alpha_class_name:
            self.elements_type = self.residues_name[self.elements_resind].tolist()

        mol_out.__post_init__()

        return mol_out

    def to(self, device: torch.device):
        """
        Moves attributes that are torch.Tensor to the given device
        Args:
            device: torch device
        """
        # End early if the molecule is already on the device
        if self.device == device:
            return

        self.device = device
        for attr, attr_val in self.__dict__.items():
            if isinstance(attr_val, torch.Tensor):
                setattr(self, attr, attr_val.to(device))

    def plot(self, **kwargs):
        from visualization import plot_molecule
        return plot_molecule(self, **kwargs)

    def save_pdb(self, top_filepath: str | pathlib.Path = None, template_uni: mda.Universe = None, pdb_filepath='',
                 pdb_dir=''):
        """
        Saves biomolecule into a .pdb file
        Args:
            top_filepath: filepath of the topology .top file
            template_uni: template universe used to save to .pdb
            pdb_filepath: filepath of the output .pdb file. If undefined, saves to pdb_dir/{ID}.pdb
            pdb_dir: directory where output .pdb is saved

        Returns:
            pdb_filepath: filepath of the output .pdb
        """
        if self.element_class not in [heavy_atoms_class_name, all_atoms_class_name]:
            raise NotImplementedError(f"pdb file saving is not implemented for element_class {self.element_class!r}.")

        if not pdb_filepath:
            if not pdb_dir:
                raise ValueError(f"Cannot determine filepath of .pdb output.")
            pdb_filepath = pathlib.Path(pdb_dir) / f"{self.ID}.pdb"
        pdb_filepath = pathlib.Path(pdb_filepath)
        pdb_filepath.parent.mkdir(parents=True, exist_ok=True)

        # Initialize template universe if None is given
        if template_uni is None:
            template_uni = uni_from_top(top_filepath=top_filepath, element_class=self.element_class, trajectory=True)

            # Set dimension of template universe
            box_center, box_dim = pint_reg.Quantity(list(self.box_params()), self.length_units).to(pdb_length_units)
            template_uni.dimensions = np.concatenate([box_dim.magnitude, 90 * np.ones(3)])

        # Check that the atom ordering of the universe and self matches
        atoms_ID_uni = np.array([f"{a.resid}_{a.name}" for a in template_uni.atoms])
        atoms_ID_mol = np.array([f"{r}_{n}" for (r, n) in zip(self.elements_resid, self.elements_pdb_name())])
        atoms_ID_ismatch = atoms_ID_uni == atoms_ID_mol

        if np.any(~atoms_ID_ismatch):
            # Find the indices that orders the sorted atom_ID index in the same order as atoms in the universe
            uni_sorting_ind = np.argsort(np.argsort(atoms_ID_uni))
            elements_sorting_ind = np.argsort(atoms_ID_mol)
            ordering_ind = elements_sorting_ind[uni_sorting_ind]

            # Recheck ordering
            atoms_ID_ismatch = atoms_ID_uni == atoms_ID_mol[ordering_ind]
            if np.any(~atoms_ID_ismatch):
                not_matching_ind = np.where(~atoms_ID_ismatch)[0]
                not_matching_uni = np.array(atoms_ID_uni)[not_matching_ind]
                not_matching_mol = np.array(atoms_ID_mol[ordering_ind])[not_matching_ind]
                raise ValueError(f"Ordering of atoms did not work.\n"
                                 f"not_matching_uni={not_matching_uni},not_matching_mol={not_matching_mol}")
        else:
            ordering_ind = np.arange(self.n_elements)

        # Convert the units of the positions to the PDB units
        elements_pos = self.convert_length(self.elements_position, units=pdb_length_units)[ordering_ind]

        # Center elements within a large enough box that prevents periodic image interactions
        # box_center, box_dim = pint_reg.Quantity(list(self.box_params()), self.length_units).to(pdb_length_units)

        # Center position in the middle of the template universe.
        if template_uni.dimensions is not None:
            elements_pos = remove_centroid(elements_pos) + torch.tensor(template_uni.dimensions[:3] / 2).unsqueeze(0)

        # Write coordinates to PDB file
        elements_pos = elements_pos.unsqueeze(0).numpy()
        loader_kwargs = dict(order='fac', dimensions=template_uni.dimensions)
        if template_uni.dimensions is not None:
            loader_kwargs['dimensions'] = template_uni.dimensions
        template_uni.load_new(elements_pos, format=mda.coordinates.memory.MemoryReader, **loader_kwargs)
        with mda.coordinates.PDB.PDBWriter(filename=str(pdb_filepath), convert_units=False) as pdb:
            pdb.write(template_uni)

        return pdb_filepath

    def __eq__(self, other: typing.Self):
        if other.__dict__.keys() != self.__dict__.keys():
            return False
        for n in self.__dict__:
            self_val, other_val = self.__dict__[n], other.__dict__[n]
            if isinstance(self_val, torch.Tensor):
                if not torch.eq(self_val, other_val).all():
                    return False
            elif isinstance(self_val, np.ndarray):
                if not np.array_equal(self_val, other_val):
                    return False
            elif self_val != other_val:
                return False
        return True

    def convert_length(self, arrays: torch.Tensor | list[torch.Tensor], units: str):
        """
        Converts input arrays with dimensions of length from the units of the given molecule to a given units
        Args:
            arrays: tensor or list of tensors to convert
            units: length units of the output arrays

        Returns:
            arrays in the same format as the input
        """
        if not self.length_units:
            raise ValueError(f"Current molecule is unitless. Cannot convert units.")
        conversion_factor = calculate_units_conversion_factor(self.length_units, units)
        if isinstance(arrays, list):
            arrays = [conversion_factor * a for a in arrays]
        else:
            arrays = conversion_factor * arrays
        return arrays

    @property
    def radial_2nd_mom(self) -> torch.float:
        """
        Calculates the 2nd moment of the distribution of the elements' radius about the centroid.
        The metric is equivalent to the radius of gyration when all masses are equal.
        Returns:
            torch.float
        """
        return torch.sqrt(torch.sum(torch.var(self.elements_position, dim=0, correction=0)))

    @property
    def radial_2nd_mom_CA(self) -> torch.float:
        """
        Calculates the 2nd moment of the distribution of the CA elements' radius about the centroid.
        The metric is equivalent to the radius of gyration when all masses are equal.
        Returns:
            torch.float
        """
        return torch.sqrt(torch.sum(torch.var(self.CA_pos, dim=0, correction=0)))

    @property
    def R_g(self) -> torch.float:
        """
        Calculates the radius of gyration using the elements mass
        Returns:
            torch.float
        """
        pos = self.elements_position.clone()
        if self.elements_mass is None:
            mass = self.elements_position.new_ones((self.n_elements, 1))
        else:
            mass = self.elements_mass.reshape(-1, 1)
        w = mass / torch.sum(mass)
        COM = torch.sum(w * pos, dim=0)
        pos -= COM
        Rg = torch.sqrt(torch.sum(w * pos.square()))
        return Rg

    @property
    def end_to_end_dist(self) -> torch.float:
        """
        Calculates the distance between the first and last element
        Returns:
            torch.float: Euclidean distance between the first and last element
        """
        return torch.linalg.norm(torch.diff(self.elements_position[[0, -1], :], dim=0), dim=1)

    @property
    def elements_pdist(self) -> torch.Tensor:
        """
        Calculates the Euclidean distance between elements of the biomolecule
        Returns:
            pairwise distance, element indices of each pair
        """
        elem_pdist = torch.pdist(self.elements_position, p=2)
        return elem_pdist.reshape(1, -1)

    def element_pairs_label(self):
        """
        Returns a label for each pair of element in the molecule
        Returns:
            dictionary with (i,j) tuple keys where i,j indexes the elements and i < j
        """
        pairs_name = dict()
        elements_resid = self.elements_resid
        for i in range(self.n_elements):
            res_id_i = self.elements_resid[i]
            res_name_i = self.residues_name[self.elements_resind[i]]
            atom_name_i = self.elements_name[i]
            for j in range(i + 1, self.n_elements):
                res_id_j = elements_resid[j]
                res_name_j = self.residues_name[self.elements_resind[j]]
                atom_name_j = self.elements_name[j]

                if res_id_i == res_id_j:
                    pair_name = f"{res_name_i}{res_id_i}:{atom_name_i}-{atom_name_j}"
                else:
                    pair_name = f"{res_name_i}{res_id_i}:{atom_name_i}-{res_name_j}{res_id_j}:{atom_name_j}"
                pairs_name[(i, j)] = pair_name

        return pairs_name

    @property
    def elements_position_local(self):
        """
        Calculates the position of elements relative to the position of their associated C-alpha atom
        Returns:
            torch.Tensor with shape=(self.n_elements,3)
        """
        if self.element_class == c_alpha_class_name:
            return torch.zeros_like(self.elements_position)
        elif self.element_class in [c_alpha_sidechain_class_name, heavy_atoms_class_name, all_atoms_class_name]:
            elements_CA_pos = self.elements_CA_pos
            elements_position_local = self.elements_position - elements_CA_pos
            return elements_position_local
        else:
            raise ValueError(f"Local positions is not implemented for element class{self.element_class!r}.")

    @property
    def CA_pdist(self) -> torch.Tensor:
        """
        Calculates the Euclidean distance between the alpha carbons of the biomolecule
        Returns:
            pairwise distance, element indices of each pair
        """
        elem_pdist = torch.pdist(self.CA_pos)
        return elem_pdist.reshape(1, -1)

    @property
    def backbone_atoms_ind(self):
        """
        Returns the indices of the 3 atoms (N, C-alpha, C) that form the backbone of each residue.
        Returns:
            torch.Tensor with shape=(self.n_residues,3)
        """
        N_atom_ind = torch.tensor(self.elements_name == 'N').nonzero()[:, 0]
        CA_atom_ind = self.elements_is_CA.nonzero()[:, 0]
        C_atom_ind = torch.tensor(self.elements_name == 'C').nonzero()[:, 0]
        if self.has_caps:
            N_atom_ind = N_atom_ind[:-1]
            C_atom_ind = C_atom_ind[1:]
        backbone_atoms_ind = torch.stack([N_atom_ind, CA_atom_ind, C_atom_ind], dim=-1)
        return backbone_atoms_ind

    def backbone_dihedral_quint_ind(self):
        """
        Returns the quintet of indices of backbone atoms that form the two backbone dihedral angles (phi,psi).
        Returns:
            torch.Tensor with shape=(self.n_residues-2,5) where columns denote [C_i, N_i+1, C-alpha_i+1, C_i+1, N_i+2]
        """
        backbone_atoms_ind = self.backbone_atoms_ind
        quint_ind = torch.cat([backbone_atoms_ind[:-2, -1:], backbone_atoms_ind[1:-1], backbone_atoms_ind[2:, :1]], -1)
        return quint_ind

    @property
    def backbone_atoms_pos(self) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Returns the positions of backbone atoms (N, C-alpha, C)
        Returns:
            N_pos, CA_pos, C_pos
        """
        backbone_atoms_pos = self.elements_position[self.backbone_atoms_ind]
        return backbone_atoms_pos.unbind(dim=1)

    @property
    def backbone_dihedrals(self):
        """
        Calculates all Ramachandran backbone dihedral angles (phi,psi) in degree of the molecule
        Returns:
            torch.Tensor with shape=(self.n_residues-2,2) where columns denote the angles [phi,psi].
            None is returned if the molecule's element_class doesn't define backbone heavy-atoms
        """
        if self.element_class in [heavy_atoms_class_name, all_atoms_class_name]:
            N_atoms_pos, CA_atoms_pos, C_atoms_pos = self.backbone_atoms_pos

            quint_pos = [C_atoms_pos[:-2], N_atoms_pos[1:-1], CA_atoms_pos[1:-1], C_atoms_pos[1:-1], N_atoms_pos[2:]]
            quint_pos = torch.stack(quint_pos, dim=1)
            phi_angles = calc_dihedral_angle(quint_pos[:, :-1, :], deg=True)
            psi_angles = calc_dihedral_angle(quint_pos[:, 1:, :], deg=True)
            phi_psi_angle = torch.stack([phi_angles, psi_angles], dim=1)

            return phi_psi_angle
        else:
            raise ValueError(f"Backbone dihedrals are undefined for element_class={self.element_class!r}")

    @property
    def dihedral_omega(self):
        """
        Calculates the omega backbone dihedral angle in degree, which is formed by the quartet (C-alpha_i,C_i,N_i+1,C-alpha_i+1)
        Returns:
            torch.Tensor with shape=(self.n_residues-1,)
            None is returned if the molecule's element_class doesn't define backbone heavy-atoms
        """
        if self.element_class in [heavy_atoms_class_name, all_atoms_class_name]:
            N_atoms_pos, CA_atoms_pos, C_atoms_pos = self.backbone_atoms_pos
            quartet_pos = torch.stack([CA_atoms_pos[:-1], C_atoms_pos[:-1], N_atoms_pos[1:], CA_atoms_pos[1:]], dim=1)
            omega_angle = calc_dihedral_angle(quartet_pos, deg=True)
            return omega_angle
        else:
            raise ValueError(f"Backbone dihedrals are undefined for element_class={self.element_class}")

    @property
    def backbone_bond_lengths(self):
        """
        Calculates the length of 3 specific bonds on the backbone of each residue.
        The pair of atoms for each bond type are: (N_i,C-alpha_i), (C-alpha_i,C_i), (C_i,N_i+1)
        Returns:
            torch.Tensor with shape=(self.n_residues-1,3)
        """
        if self.element_class in [heavy_atoms_class_name, all_atoms_class_name]:
            N_atoms_pos, CA_atoms_pos, C_atoms_pos = self.backbone_atoms_pos
            N_CA_bond_length = torch.linalg.norm(N_atoms_pos[:-1] - CA_atoms_pos[:-1], dim=-1)
            CA_C_bond_length = torch.linalg.norm(CA_atoms_pos[:-1] - C_atoms_pos[:-1], dim=-1)
            C_N_bond_length = torch.linalg.norm(C_atoms_pos[:-1] - N_atoms_pos[1:], dim=-1)
            bonds_length = torch.stack([N_CA_bond_length, CA_C_bond_length, C_N_bond_length], dim=1)
            return bonds_length
        else:
            raise ValueError(f"Backbone bond lengths are undefined for element_class={self.element_class}")

    @property
    def backbone_bond_angles(self):
        """
        Calculates 3 specific bond angles on the backbone of each residue.
        The triplet of atoms for each bond angle are: (N_i, C-alpha_i, C_i), (C-alpha_i, C_i, N_i+1), (C_i, N_i+1, C-alpha_i+1)
        Returns:
            torch.Tensor with shape=(self.n_residues-1,3)
        """
        if self.element_class in [heavy_atoms_class_name, all_atoms_class_name]:
            N_atoms_pos, CA_atoms_pos, C_atoms_pos = self.backbone_atoms_pos
            quintet_pos = [N_atoms_pos[:-1], CA_atoms_pos[:-1], C_atoms_pos[:-1], N_atoms_pos[1:], CA_atoms_pos[1:]]
            quintet_pos = torch.stack(quintet_pos, dim=1)
            N_CA_C_angle = calc_bond_angle(quintet_pos[:, 0:3], deg=True)
            CA_C_N_angle = calc_bond_angle(quintet_pos[:, 1:4], deg=True)
            C_N_CA_angle = calc_bond_angle(quintet_pos[:, 2:5], deg=True)
            bond_angles = torch.stack([N_CA_C_angle, CA_C_N_angle, C_N_CA_angle], dim=1)
            return bond_angles
        else:
            raise ValueError(f"Backbone bond angles are undefined for element_class={self.element_class}")

    @property
    def chirality(self):
        """
        Calculates the chirality of the C-alpha backbone as defined in the reference.
        Reference: "Protein Helical Structures: Defining Handedness and Localization Features", https://www.mdpi.com/2073-8994/13/5/879
        Returns:
            torch.Float that equals 1 or -1
        """
        # Reference: "Protein Helical Structures: Defining Handedness and Localization Features", https://www.mdpi.com/2073-8994/13/5/879
        CA_pos = self.CA_pos
        CA_vec = torch.diff(CA_pos, dim=-2)
        CA_vec_cross = torch.linalg.cross(CA_vec[:-1], CA_vec[1:])
        CA_vec_prod_sum = torch.linalg.vecdot(CA_vec_cross[:-1], CA_vec[2:]).sum()
        return CA_vec_prod_sum.sign()

    @property
    def elements_resid(self):
        return self.residues_id[self.elements_resind]

    @property
    def elements_is_H(self):
        return torch.tensor(np.char.startswith(self.elements_name, 'H'))

    @property
    def elements_is_CA(self):
        return torch.tensor(self.elements_name == 'CA')

    @property
    def elements_is_backbone(self):
        if self.element_class == c_alpha_class_name:
            is_atom_backbone = torch.ones(self.n_elements, dtype=torch.bool)
        elif self.forcefield:
            backbone_atoms_name = forcefields[self.forcefield].backbone_atoms_name
            is_atom_backbone = torch.tensor(np.isin(self.elements_name, backbone_atoms_name))
        else:
            raise ValueError(f"Cannot determine backbone atoms")

        # If the molecule has caps, do not count them as backbone
        if self.has_caps:
            is_atom_backbone &= (self.elements_resind > 0) & (self.elements_resind < self.n_residues - 1)

        return is_atom_backbone

    @property
    def n_residues(self):
        return len(self.residues_name)

    @property
    def n_CA_atoms(self):
        return int(self.elements_is_CA.sum())

    @property
    def n_sidechain_elems(self):
        n_side_atoms = torch.zeros(self.n_residues, dtype=torch.int)
        side_atom_resind = self.elements_resind[~self.elements_is_backbone]
        n_side_atoms.index_add_(dim=0, index=side_atom_resind,
                                source=torch.ones_like(side_atom_resind, dtype=torch.int))
        return n_side_atoms

    @property
    def has_caps(self):
        return self.residues_name[0] == 'ACE' and self.residues_name[-1] == 'NME'

    @property
    def residues_fullname(self):
        # For Amber molecules, add the terminal prefixes to the residue names
        residues_name = self.residues_name
        if 'amber' in self.forcefield:
            residues_name = copy.copy(residues_name)
            residues_name[0] = 'N' + residues_name[0]
            residues_name[-1] = 'C' + residues_name[-1]
        return residues_name

    def reflect(self):
        """
        Reflects the molecule around its centroid
        """
        COM = self.elements_position.mean(dim=-2)
        self.elements_position = -(self.elements_position - COM) + COM

    def normalize_element_pos(self, test=False):
        """
        Translates and rotates the molecule such that the centroid is at the origin and the covariance matrix of its
         position is the identity matrix
        Args:
            test: test the alignment
        """
        remove_centroid(self.elements_position, inplace=True)
        align_PC(self.elements_position, inplace=True, test=test)

    def calculate_bonds_length(self):
        """
        Calculates the length of the bonds defined by self.bonds_ind. Throws AssertionError if self.bonds_ind is undefined.
        Returns:
            torch.tensor with shape=(self.bonds_ind.shape[0],)
        """
        assert self.bonds_ind is not None, f"Molecule has no bonds indices. Cannot define bond length."
        bonds_vec = self.elements_position[self.bonds_ind].diff(dim=-2).squeeze(-2)
        bonds_length = torch.linalg.norm(bonds_vec, dim=-1)
        return bonds_length

    def rmsd(self, mol: typing.Self, mask: str | torch.Tensor = None):
        """
        Calculates the root-mean-square deviation (RMSD) between the molecule and a target molecule.
        ||y-x*R||^2 is minimized such x,y are matrices of the elements positions (shape=(n,3)) centered at their centroid.

        Reference: Umeyama 1991, "Least-squares estimation of transformation parameters between two point patterns"

        Args:
            mol: other molecule with respect to which the RMSD is calculated
            mask: mask array to compare rmsd of a subset of atoms
        Returns:
            scalar corresponding to the RMSD
        """

        if mask is not None:
            if mask == 'CA':
                x, y = self.CA_pos, mol.CA_pos
            else:
                x, y = self.elements_position[mask], mol.elements_position[mask]
        else:
            x, y = self.elements_position, mol.elements_position

        if x.shape != y.shape:
            raise ValueError(f"Cannot calculate RMSD for molecules with different shapes.\n"
                             f"self.elements_position.shape={x.shape}, other.elements_position.shape={y.shape}")
        xy_rmsd = rmsd(x, y, reflect=False)
        return xy_rmsd

    def rmsd_align(self, mol_ref: typing.Self):
        """
        Aligns molecule to a reference molecule such that their position RMSD (||y-x*R||^2) is minimized.
        Args:
            mol_ref: reference molecule
        """
        x, y = self.elements_position, mol_ref.elements_position
        if x.shape != y.shape:
            raise ValueError(f"Cannot align two molecules with different shapes.\n"
                             f"self.elements_position.shape={x.shape}, mol_ref.elements_position.shape={y.shape}")
        x = remove_centroid(x)
        self.elements_position = rmsd_align(positions=x, positions_ref=y, reflect=False)

    def align_PC(self, PC_axes=None):
        """
        Rotates the structure to align the principal components along the given axes
        """
        if PC_axes is None:
            PC_axes = ['x', 'y', 'z']
        PC_axes_ind = torch.tensor([{'x': 0, 'y': 1, 'z': 2}[i] for i in PC_axes])

        align_PC(self.elements_position, PC_ind=PC_axes_ind, inplace=True)

    @property
    def CA_pos(self):
        """
        Gathers the position of the C-alpha atoms
        Returns:
            torch.Tensor with shape=(self.n_residues,3)
        """
        c_alpha_pos = self.elements_position[self.elements_is_CA]
        return c_alpha_pos

    @property
    def elements_CA_pos(self):
        """
        For each element, gathers the position of the C-alpha atom of the residue that each element belongs to.
        Returns:
            torch.Tensor with shape=(self.n_elements,3)
        """
        c_alpha_positions = self.CA_pos
        elements_CA_ind = self.elements_CA_ind
        return c_alpha_positions[elements_CA_ind, :]

    @property
    def elements_CA_ind(self):
        """
        Returns the index of the CA atom associated with each element.
        Returns:
            torch.Tensor with shape=(n_elements,3)
        """
        # If the molecule has caps, associate atoms in caps with the CA atom of the residue they are connected to
        if self.has_caps:
            elements_CA_ind = (self.elements_resind - 1).clamp(min=0, max=self.n_residues - 3)
        else:
            elements_CA_ind = self.elements_resind
        return elements_CA_ind

    def residue_rep_ind(self, residue_rep_name: dict[str, str]):
        """
        Returns the atom index of the sidechain representative for each residue in the molecule
        Args:
            residue_rep_name: dict[R:N] where keys (R) are residue names and values (N) indicate the atom name that
                              represents residue R.

        Returns:
            torch.Tensor of shape (self.n_residues,)
        """
        residues_name = self.residues_name
        resind = torch.arange(self.n_residues, device=self.device).unsqueeze(-1)
        is_atom_in_res = torch.eq(self.elements_resind, resind)  # shape=(n_res,n_atoms)
        res_rep_name = np.array([residue_rep_name[res_name] for res_name in residues_name])
        is_atom_name_match = torch.tensor(np.equal(self.elements_name, res_rep_name[:, None])).to(self.device)
        is_atom_rep = torch.logical_and(is_atom_in_res, is_atom_name_match)

        # Check that all residues have a representative
        is_residue_represented = is_atom_rep.any(dim=1)
        if not is_residue_represented.all():
            res_unrep = {i: res_name for i, res_name in enumerate(residues_name) if not is_residue_represented[i]}
            raise ValueError(f"The following residue sidechains are not represented by an atom.\n{res_unrep}")

        rep_atom_ind = is_atom_rep.nonzero()[:, 1]
        return rep_atom_ind

    def elements_coarse_pos(self, method='CA', coarse_grains: CoarseGrains = None):
        """
        Gathers the coarse grain position for each element in the molecule
        Args:
            method: coarse graining method
            coarse_grains: info about the coarse grains

        Returns:
            torch.Tensor with shape=(self.n_elements,3) where n denotes the number of elements
        """
        elements_coarse_ind = self.elements_coarse_ind(method=method)
        coarse_pos = self.coarse_pos(method=method, coarse_grains=coarse_grains)
        elements_coarse_pos = coarse_pos[elements_coarse_ind]
        return elements_coarse_pos

    def coarse_pos(self, method='CA', coarse_grains: CoarseGrains = None):
        """
        Gathers the position of the coarse grains in the molecule
        Args:
            method: coarse graining method
            coarse_grains: info about the coarse grains

        Returns:
            torch.Tensor with shape=(n_CG,3) where n_CG denotes the number of coarse grains
        """
        if method == 'RES_CENT':
            coarse_pos = torch.nan * torch.ones((self.n_residues, 3))
            coarse_pos.index_reduce_(source=self.elements_position, index=self.elements_resind, dim=0, reduce='mean',
                                     include_self=False)
        else:
            coarse_pos = self.elements_position[self.coarse_atom_ind(method=method, coarse_grains=coarse_grains)]
        return coarse_pos

    def coarse_radius(self, method='CA', coarse_grains: CoarseGrains = None):
        """
        Defines the radius of each coarse grain in the molecule following the given coarse grain geometries
        Args:
            method: method used to define the coarse grains
            coarse_grains: object that defines the geometries of the coarse grains (center and radius)

        Returns:
            torch.Tensor with shape=(n_coarse_grains,)
        """
        if method == 'CA':
            coarse_radius = torch.zeros(self.n_residues, device=self.device)
        elif method in ['RES_CA', 'RES', 'RES_CENT']:
            residue_grains_radius = coarse_grains.radii
            coarse_radius = torch.stack([residue_grains_radius[k] for k in self.residues_name]).to(device=self.device)
        elif method.startswith('CA_SI'):
            # Get the sidechain grain radius from the input calculated coarse grains
            sidechain_grains_radius = coarse_grains.radii
            side_radius = torch.stack([sidechain_grains_radius[k] for k in self.residues_name]).to(device=self.device)

            # Define the backbone grain radius
            n_CA_atoms = self.n_CA_atoms
            if method == 'CA_SI':
                CA_radius = config.backbone_grain_radius * torch.ones(n_CA_atoms, device=self.device)
            else:
                raise NotImplementedError
            coarse_radius = torch.cat([CA_radius, side_radius])
        else:
            raise NotImplementedError(f"{method!r} method is not implemented.")

        return coarse_radius

    def coarse_atom_ind(self, method='CA', coarse_grains: CoarseGrains = None):
        """
        Returns the index of the atoms that corresponds to the center of the coarse grains in the biomolecule
        Args:
            method: method used for coarse graining
            coarse_grains: info about the coarse grains

        Returns:
            torch.Tensor with shape=(n_CG,) where n_CG denotes the number of coarse grains
        """
        CA_atom_ind = self.elements_is_CA.nonzero().squeeze().to(self.device)
        if method in ['CA', 'RES_CA']:
            # Adjacency of two atoms is determined by the adjacency of the CA atom of the residue that they belong to
            coarse_atom_ind = CA_atom_ind
        elif method.startswith('CA_SI'):
            # In this method, there are two types of coarse grains:
            # 1) backbone coarse grains centered on CA
            # 2) sidechain coarse grains centered on a pre-determined center
            # The given coarse grains represent the sidechains
            sidechain_rep_atom_ind = self.residue_rep_ind(residue_rep_name=coarse_grains.centers_name)
            coarse_atom_ind = torch.cat([CA_atom_ind, sidechain_rep_atom_ind], dim=0)
        elif method == 'RES':
            coarse_atom_ind = self.residue_rep_ind(residue_rep_name=coarse_grains.centers_name)
        elif method == 'RES_CENT':
            coarse_atom_ind = torch.nan * torch.ones(self.n_residues)
        else:
            raise NotImplementedError(f"{method!r} method is not implemented.")
        return coarse_atom_ind

    def elements_coarse_ind(self, method='CA'):
        """
        Returns the coarse-grain index of each element in the biomolecule
        Args:
            method: method used for coarse graining

        Returns:
            torch.Tensor with shape=(n,) where n denotes the number of elements
        """
        if method in ['CA', 'RES_CA', 'RES', 'RES_CENT']:
            # Adjacency of two atoms is determined by the adjacency of the CA atom of the residue that they belong to
            coarse_ind = self.elements_resind.clone()
        elif method.startswith('CA_SI'):
            # In this method, there are two types of coarse grains: 1) CA and 2) sidechain representative atom.
            # If an atom belongs to the backbone, use CA as the coarse grain. Otherwise, use the sidechain rep.
            elem_is_backbone = self.elements_is_backbone
            coarse_ind = self.elements_CA_ind.clone()
            coarse_ind[~elem_is_backbone] = self.elements_resind[~elem_is_backbone] + self.n_CA_atoms
        else:
            raise NotImplementedError(f"{method!r} method is not implemented.")
        return coarse_ind

    def calculate_elements_adj(self, method='CA', coarse_grains: CoarseGrains = None,
                               buffer_dist: float | torch.Tensor = 0.0):
        """
        Calculates upper triangular adjacency matrix of the elements given an adjacency method and coarse grains
        Args:
            method: method used to calculated adjacency
            coarse_grains: info of coarse grains used with the associated adjacency method
            buffer_dist: distance added to distance threshold for determining adjacency of two coarse grains

        Returns:
            torch.Tensor of shape=(self.n_elements,self.n_elements)
        """
        if buffer_dist < torch.inf and method is not None:
            if self.length_units != 'nm':
                raise ValueError(f"Elements position must be in given in 'nm'. Received {self.length_units!r}.")

            # Preliminary calculations
            coarse_pos = self.coarse_pos(method=method, coarse_grains=coarse_grains)
            coarse_pdist = torch.pdist(coarse_pos)
            pair_ind = torch.triu_indices(coarse_pos.shape[0], coarse_pos.shape[0], offset=1, device=self.device)
            coarse_adj = torch.diag(torch.ones(coarse_pos.shape[0], dtype=torch.bool, device=self.device))

            # It should be added after the coarse radius is calculated since it's only used for calculating adjacency
            # Also remove the coarse radius correction
            coarse_radius = self.coarse_radius(method=method, coarse_grains=coarse_grains)

            # Calculate adjacency threshold d_th = r_CG_1 + r_CG_2 + b
            # where r_CG_1, r_CG_2 are the coarse grain radii of each element of the pair and b is the buffer distance.
            coarse_dist_thresh = coarse_radius[pair_ind].sum(dim=0) + buffer_dist

            # Calculate the coarse grain adjacency matrix
            are_coarse_adj = torch.le(coarse_pdist, coarse_dist_thresh)
            coarse_adj[pair_ind[0], pair_ind[1]] = are_coarse_adj
            coarse_adj[pair_ind[1], pair_ind[0]] = are_coarse_adj

            # Expand the coarse grain adjacency matrix to elements
            elems_coarse_ind = self.elements_coarse_ind(method=method)
            elements_adj = coarse_adj[elems_coarse_ind, elems_coarse_ind.unsqueeze(-1)]
        else:
            elements_adj = torch.ones((self.n_elements, self.n_elements), dtype=torch.bool, device=self.device)

        elements_adj = elements_adj.triu(diagonal=1)

        return elements_adj

    def calculate_adj_elements_pdist(self, **adj_method_kwargs):
        """
        Calculates the pairwise distance between adjacent elements
        Returns:
            adj_pdist: shape=(n,) where n is the number of unique adjacent pairs
        """
        elements_adj = self.calculate_elements_adj(**adj_method_kwargs)
        edges_ind = elements_adj.nonzero()
        edges_vec = torch.diff(self.elements_position[edges_ind, :], dim=-2).squeeze(-2)
        elements_pdist = edges_vec.norm(dim=-1)
        return elements_pdist

    def box_params(self) -> (torch.Tensor, torch.Tensor):
        """
        Calculates dimension of a cubic box that fits all elements of the molecule
        Returns:
            (box_center,box_dimension)
        """
        pos = self.elements_position
        box_center = torch.mean(pos, dim=0)
        box_dim = torch.max(pos, dim=0)[0] - torch.min(pos, dim=0)[0]
        return box_center, box_dim

    def supported_types(self, entity: str):
        if entity == 'element':
            if self.element_class == c_alpha_class_name:
                supported_types_temp = supported_types[self.element_class]
            else:
                if not self.forcefield:
                    raise ValueError(f"Cannot define element types without a defined forcefield.")
                supported_types_temp = supported_types[self.element_class][self.forcefield]
        elif entity == 'residue':
            supported_types_temp = supported_types[residue_class_name]
            if self.has_caps:
                supported_types_temp = supported_types_temp + ['ACE', 'NME']
        else:
            raise NotImplementedError(f"entity {entity!r} is not implemented.")

        return supported_types_temp

    def elements_pdb_name(self):
        """
        Returns elements name in the PDB convention. Specificaly, OO atoms in the C-terminal are renamed to O, OXT
        Returns:
            np.ndarray of shape=(self.n_elements,)
        """
        # Rename terminal O atoms
        is_atom_in_C_terminal = self.elements_resind == self.elements_resind[-1]
        C_term_atoms_name = self.elements_name[is_atom_in_C_terminal]
        pdb_elements_name = self.elements_name.copy()

        if np.all(np.isin(charmm_C_terminal_O_names, C_term_atoms_name)):
            OO_atom_ind = np.array([np.nonzero(C_term_atoms_name == n)[0].item() for n in charmm_C_terminal_O_names])
        elif np.all(np.isin(amber_C_terminal_O_names, C_term_atoms_name)):
            OO_atom_ind = np.array([np.nonzero(C_term_atoms_name == n)[0].item() for n in amber_C_terminal_O_names])
        else:
            OO_atom_ind = None

        if OO_atom_ind is not None:
            OO_atom_ind = OO_atom_ind + is_atom_in_C_terminal.nonzero()[0].item()
            pdb_elements_name[OO_atom_ind] = pdb_terminal_OO_names

        return pdb_elements_name

    def check_top_match(self, other: typing.Self):
        """
        Checks that topology of self matches with input molecule
        Args:
            other: other Biomolecule whose topology is compared to self

        Returns:
            is_match: true if topologies of self and other are matching
        """
        if self.n_residues != other.n_residues:
            return False
        if ~np.all(self.residues_name == other.residues_name):
            return False
        if self.n_elements != other.n_elements:
            return False
        if ~np.all(self.elements_name == other.elements_name):
            return False

        return True

    def __setstate__(self, state):
        for k, v in state.items():
            if k == 'top_filepath':
                pass
            else:
                setattr(self, k, v)

        # Ensure the bonds ind are torch.long to avoid indexing errors
        if self.bonds_ind is not None:
            self.bonds_ind = self.bonds_ind.long()

        self.rename_old_attrs()


def uni_from_top(top_filepath: str | pathlib.Path, element_class: str, trajectory=True, rename_C_OO=True):
    """
    Initializes a MDAnalysis.Universe from a GROMACS' .top file
    Args:
        top_filepath: filepath of the .top file
        element_class: element class of the molecule in the universe
        trajectory: initializes Universe as a trajectory
        rename_C_OO: Renames the 2 OO atoms in the C terminal to O and OXT

    Returns:
        MDAnalysis.Universe
    """
    with warnings.catch_warnings(category=UserWarning, action="ignore"):
        top_uni = mda.Universe(topology=str(top_filepath), topology_format='ITP', include_dir=config.data_dir)
    prot_sel = top_uni.select_atoms(prot_sel_crit)
    top_uni_has_caps = prot_sel.residues[0].resname == 'ACE' and prot_sel.residues[-1].resname == 'NME'
    if element_class == heavy_atoms_class_name:
        atoms_sel = prot_sel.select_atoms('not name H* and not name M*')
    elif element_class == all_atoms_class_name or element_class is None:
        atoms_sel = prot_sel.select_atoms('not name M*')
    else:
        raise NotImplementedError

    # Find the residue index of each atom considering only residues in the selection
    atom_resindex = np.nonzero(atoms_sel.resids[:, None] == atoms_sel.residues.resids[None, :])[1]

    uni = mda.Universe.empty(n_atoms=atoms_sel.n_atoms,
                             n_residues=atoms_sel.n_residues,
                             atom_resindex=atom_resindex,
                             trajectory=trajectory)

    # Rename C terminal OO atoms
    atom_names = atoms_sel.atoms.names
    if rename_C_OO and not top_uni_has_caps:
        is_atom_in_C_terminal = atoms_sel.atoms.resids == atoms_sel.residues[-1].resid
        C_terminal_atom_names = atom_names[is_atom_in_C_terminal]

        if np.all(np.isin(charmm_C_terminal_O_names, C_terminal_atom_names)):
            C_terminal_OO_names = charmm_C_terminal_O_names
        elif np.all(np.isin(amber_C_terminal_O_names, C_terminal_atom_names)):
            C_terminal_OO_names = amber_C_terminal_O_names
        else:
            raise ValueError(f"Cannot identify OO atom names in C terminal.")
        terminal_1stO_ind = (atom_names == C_terminal_OO_names[0]) & is_atom_in_C_terminal
        atom_names[terminal_1stO_ind] = 'O'
        terminal_2ndO_ind = (atom_names == C_terminal_OO_names[1]) & is_atom_in_C_terminal
        atom_names[terminal_2ndO_ind] = 'OXT'

    # Add topology attributes
    uni.add_TopologyAttr('name', atom_names)
    uni.add_TopologyAttr('resname', atoms_sel.residues.resnames)
    uni.add_TopologyAttr('resid', atoms_sel.residues.resids)
    attr_names = ['altLocs', 'icodes', 'tempfactors', 'formalcharges', 'record_types']
    for name in attr_names:
        uni.add_TopologyAttr(name)
    uni.add_TopologyAttr('elements', [a[0] for a in atoms_sel.atoms.names])
    uni.add_TopologyAttr('masses', atoms_sel.masses)
    uni.add_TopologyAttr('occupancies', np.ones(atoms_sel.n_atoms))
    uni.add_TopologyAttr('chainIDs', atoms_sel.n_atoms * ['A'])
    uni.add_TopologyAttr('segids', ['1'])

    return uni


# Define the various classes of biomolecules and elements
residue_class_name = 'residue'
c_alpha_class_name = 'c_alpha'  # Structure contains one CA atom for each residue
c_alpha_sidechain_class_name = 'c_alpha_sidechain'  # Structure contains one CA atom and one sidechain atom for each residue
heavy_atoms_class_name = 'heavy_atoms'  # Structure contains all non-hydrogen atoms
all_atoms_class_name = 'all_atoms'  # Structure contains all atoms
element_classes = [c_alpha_class_name, c_alpha_sidechain_class_name, heavy_atoms_class_name, all_atoms_class_name]
element_classes_abb = {'CA': c_alpha_class_name,
                       'CS': c_alpha_sidechain_class_name,
                       'HA': heavy_atoms_class_name,
                       'AA': all_atoms_class_name}

# Supported types
supported_types = {residue_class_name: canonical_aa_codes_3,
                   c_alpha_class_name: canonical_aa_codes_3,
                   c_alpha_sidechain_class_name: force_fields_heavy_atom_types,
                   heavy_atoms_class_name: force_fields_heavy_atom_types,
                   all_atoms_class_name: force_field_atom_types}

# Define names of backbone bond lengths and backbone bond angles
backbone_bond_length_name = ['N-CA', 'CA-C', 'C-N']
backbone_bond_angle_name = ['N-CA-C', 'CA-C-N', 'C-N-CA']

# MDAnalysis
prot_sel_crit = 'not (resname SOL or resname NA or resname CL or name M*)'  # Protein selection criterion for MDAnalysis
pdb_length_units = mda.coordinates.PDB.PDBWriter.units['length']
if mda.units.lengthUnit_factor['angstrom'] == 1.0:
    mda_length_units = 'angstrom'  # Preferred for pint since 'A' is confused with Ampere
else:
    mda_length_units = mda.units.MDANALYSIS_BASE_UNITS['length']

# Statistics files older than this date are deleted and recalculated
stats_expiry_date = '2025-12-10 14:00:00'

if __name__ == '__main__':
    # Test from_gro() constructor
    fp = config.data_dir / 'MD_box' / 'AAQAA3.gro'
    mol = Biomolecule.from_gro(fp, 'heavy_atoms', save_bonds=True)

    # Test Biomolecule coarsen()
    from data.datasets import load_dataset

    dataset = load_dataset()
    mol = dataset[0]
    mol_coarse = mol.coarsen(element_class=c_alpha_sidechain_class_name, inplace=False)

    # Test molecule creation from sequence
    seq = ['GLY', 'ALA', 'MET', 'GLY', 'PRO', 'SER', 'TYR', 'GLY', 'ARG', 'SER']
    mol, top_filepath = Biomolecule.from_sequence(seq, terminals_protonation=['0', '0'], forcefield=charmm36m,
                                                  element_class=heavy_atoms_class_name)
    pass

    # Test the rmsd method
    prot1 = Biomolecule.random(element_class=c_alpha_class_name, n_elements=100)
    prot2 = Biomolecule.random(element_class=c_alpha_class_name, n_elements=prot1.n_elements)
    prot12_rmsd_before = prot1.rmsd(prot2)
    rbm = RigidBodyMotion.random((1,))
    prot1.elements_position = rbm.transform(prot1.elements_position.unsqueeze(0)).squeeze(0)
    prot12_rmsd_after = prot1.rmsd(prot2)
    is_rmsd_equal = prot12_rmsd_after.isclose(prot12_rmsd_before)
