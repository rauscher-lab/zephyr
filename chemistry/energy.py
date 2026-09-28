from __future__ import annotations

import functools
import copy
import warnings
import hashlib
import pathlib
from typing import TYPE_CHECKING

from scipy.interpolate import RectBivariateSpline
import torch
from torch import nn
import numpy as np
import networkx
import openmm.unit
from openmm.app import Simulation, Modeller, GromacsTopFile, Atom
from openmm.openmm import LangevinIntegrator, Vec3
from openmm.unit import kelvin, seconds, meter, nano, pico, kilojoules_per_mole, BOLTZMANN_CONSTANT_kB, \
    AVOGADRO_CONSTANT_NA

import config
from data.data_classes import c_alpha_class_name, heavy_atoms_class_name, all_atoms_class_name
from data.forcefields import gmx_energy_names, charmm_gmx_energy_names, amber_gmx_energy_names
from utils import calc_dihedral_angle, calc_bond_angle, parse_gmx_mdp

if TYPE_CHECKING:
    pass
nanometer = nano * meter
picosecond = pico * seconds

# MDP parameters used single-point energy estimation
mdp_dir = pathlib.Path(config.data_dir, 'mdp')
# single_point_en_params_filename = mdp_dir / 'single_point_energy.mdp'
# single_point_en_params_filepath = mdp_dir / 'single_point_energy_no_PME.mdp'
single_point_en_params_filepath = mdp_dir / 'single_point_energy_H_cons.mdp'
single_point_en_params = parse_gmx_mdp(single_point_en_params_filepath)

# Subdirectory in dataset directories where gmx_energy related files are stored
gmx_calc_subdir = 'gmx_energy' + single_point_en_params_filepath.stem.removeprefix('single_point_energy')

# Remove reciprocal coulomb energy in the forcefield energy names if not computed in the single point energy calculation
if single_point_en_params['coulombtype'] != 'PME':
    gmx_energy_names.pop('V_coulomb_recip', None)
    charmm_gmx_energy_names.pop('V_coulomb_recip', None)
    amber_gmx_energy_names.pop('V_coulomb_recip', None)


class PotentialEnergy(nn.Module):
    def __init__(self, top_file: str | pathlib.Path | GromacsTopFile, element_class=heavy_atoms_class_name,
                 loss_weight=1.0, loss_offset=0.0, pos_scale=1.0):
        super().__init__()
        if isinstance(top_file, str | pathlib.Path):
            top_file = GromacsTopFile(top_file, includeDir=config.data_dir)
        assert len(top_file._molecules) == 1, (f'Found {len(top_file._molecules)} molecules in {top_file}.'
                                               f'Only one molecule is supported.')
        self.top_file = top_file

        # Sort the keys of the various types to facilitate symmetric query
        # sorted_types = ['_bondTypes', '_angleTypes', '_dihedralTypes', '_pairTypes', '_nonbondTypes', '_cmapTypes']
        self.molecules_name = [mol[0] for mol in self.top_file._molecules]
        assert len(self.molecules_name) == 1, 'VPot is only defined with 1 molecule'
        self.molecule: GromacsTopFile._MoleculeType = self.top_file._moleculeTypes[self.molecules_name[0]]
        self.topology = self.top_file.topology
        positions = torch.randn((self.top_file.topology.getNumAtoms(), 3))
        positions = openmm.unit.Quantity([Vec3(*p) for p in positions.squeeze(0)], unit=openmm.unit.nanometer)
        modeller = Modeller(self.top_file.topology, positions)

        # Remove atoms that do not belong to the specific element class
        self.element_class = element_class
        if self.element_class == heavy_atoms_class_name:
            omitted_atoms = [a for a in self.topology.atoms() if a.name.startswith('H')]
        elif self.element_class == c_alpha_class_name:
            omitted_atoms = [a for a in self.topology.atoms() if not a.name == 'CA']
        elif self.element_class == all_atoms_class_name:
            omitted_atoms = []
        else:
            raise ValueError(f"element_class = {self.element_class} is not defined.")
        modeller.delete(omitted_atoms)
        self.topology = modeller.getTopology()

        self.omitted_atoms_id = [a.id for a in omitted_atoms]
        self.atoms = [a for a in self.top_file.topology.atoms() if a.id not in self.omitted_atoms_id]
        self.n_atoms = len(self.atoms)

        # Add mass and charge of atoms
        atoms_mass_charge_dict = {line[0]: [float(line[6]), float(line[7])] for line in self.molecule.atoms}
        self.atoms_charge = torch.tensor([atoms_mass_charge_dict[a.id][0] for a in self.atoms])
        self.atoms_mass = torch.tensor([atoms_mass_charge_dict[a.id][1] for a in self.atoms])

        self.atom_id_to_ind = {a.id: i for i, a in enumerate(self.atoms)}
        self.atom_id_to_type = {line[0]: line[1] for line in self.molecule.atoms}
        self.atom_ind_to_all_ind = {self.atom_id_to_ind[a.id]: i for i, a in enumerate(self.top_file.topology.atoms())
                                    if a.id not in self.omitted_atoms_id}
        self.pos_scale = nn.parameter.Buffer(torch.as_tensor(pos_scale), persistent=False)

        self.loss_weight = nn.parameter.Buffer(torch.tensor(loss_weight))
        self.loss_offset = nn.parameter.Buffer(torch.tensor(loss_offset))

    def random_positions(self, n_batches=10, openmm_output=False):
        positions = torch.randn((n_batches, self.top_file.topology.getNumAtoms() - len(self.omitted_atoms_id), 3))
        if openmm_output:
            positions = positions.squeeze(0)
            positions = openmm.unit.Quantity([Vec3(*p.tolist()) for p in positions], unit=openmm.unit.nanometer)
        return positions

    def test(self):
        positions = self.random_positions()
        energies = self(positions)
        return energies

    def init_openmm(self, positions: torch.Tensor = None, system: openmm.System = None):
        if positions is None:
            positions = self.random_positions(n_batches=1)[0]

        if positions.ndim > 2:
            warnings.warn(f"Can only initialize system for 1 batch. Keeping first batch only.")
            positions = positions[0]

        if system is None:
            system = self.top_file.createSystem()

        for i, f in enumerate(system.getForces()):
            f.setForceGroup(i)
        integrator = LangevinIntegrator(300 * openmm.unit.kelvin, 1 / openmm.unit.picosecond,
                                        0.004 * openmm.unit.picoseconds)
        simulation = Simulation(self.top_file.topology, system, integrator)
        positions_openmm = openmm.unit.Quantity([Vec3(*p.tolist()) for p in positions], unit=openmm.unit.nanometer)
        simulation.context.setPositions(positions_openmm)
        return system, simulation, positions

    def run_openmm_tests(self, force_types: type | list[type], system=None, n_checks=5, tol=1e-6,
                         **forward_kwargs):
        if not self.element_class == all_atoms_class_name:
            raise ValueError(f"Cannot perform openmm test when element_class is not {all_atoms_class_name}.")

        if not isinstance(force_types, list):
            force_types = [force_types]

        for i in range(n_checks):
            system, simulation, positions = self.init_openmm(system=system)
            force_groups_ind = [i for i, f in enumerate(system.getForces()) if type(f) in force_types]
            force_names, openmm_energy = [], 0
            for ind in force_groups_ind:
                state = simulation.context.getState(getEnergy=True, groups={ind})
                force = system.getForces()[ind]
                force_names.append(force.getName())
                openmm_energy += state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)

            calculated_energy = self(positions, **forward_kwargs).sum()
            relative_diff = np.abs(openmm_energy - calculated_energy) / np.abs(openmm_energy)
            if relative_diff > tol:
                raise ValueError(
                    f"OpenMM test {i + 1} failed for {self.__class__.__name__} with a relative diff. of {relative_diff}.\n"
                    f"OpenMM Energy: {openmm_energy} ({'+'.join(force_names)})\n"
                    f"Calculated Energy: {calculated_energy}")
            else:
                print(
                    f"OMM test #{i + 1} succeeded for {self.__class__.__name__}. rel_diff={relative_diff:.3g}\n"
                    f"OpenMM Energy: {openmm_energy} ({'+'.join(force_names)})\n"
                    f"Calculated Energy: {calculated_energy}")

    def check_atom_ordering(self, atoms_type: list | np.ndarray, atoms_resid):
        if not isinstance(atoms_type, np.ndarray):
            atoms_type = np.array(atoms_type)
        if not isinstance(atoms_resid, np.ndarray):
            atoms_resid = np.array(atoms_resid)
        # top_atom_names = np.array([a.name for a in self.top_file.topology.atoms() if a.id not in self.omitted_atoms_id])
        # atoms = [a for a in self.top_file.topology.atoms() if a.id not in self.omitted_atoms_id]
        top_atom_types = np.array([self.atom_id_to_type[a.id] for a in self.atoms])
        is_atom_type_matching = atoms_type == top_atom_types
        non_matching_atoms = np.where(~is_atom_type_matching)[0]
        atoms_ind = np.array([self.atom_id_to_ind[a.id] for a in self.atoms])
        if not is_atom_type_matching.all():
            raise ValueError(f"Atom types do not match at ind={atoms_ind[non_matching_atoms]}.\n"
                             f"Expecting:{top_atom_types[non_matching_atoms]}\n"
                             f"Received:{atoms_type[non_matching_atoms]}")

        top_atom_resid = np.array([int(a.residue.id) for a in self.atoms])
        is_atom_resid_matching = atoms_resid == top_atom_resid
        non_matching_atoms = np.where(~is_atom_resid_matching)[0]
        atoms_ind = np.array([self.atom_id_to_ind[a.id] for a in self.atoms])
        if not is_atom_resid_matching.all():
            raise ValueError(f"Atom residue id do not match at ind={atoms_ind[non_matching_atoms]}.\n"
                             f"Expecting:{top_atom_resid[non_matching_atoms]}\n"
                             f"Received:{atoms_resid[non_matching_atoms]}")

    def total_energy(self, positions: torch.Tensor):
        calculated_energies = self(positions)
        return calculated_energies.sum(dim=-1)

    @property
    def ID(self):
        params = [p.flatten() for p in self.parameters()]
        params = [p.round(decimals=6) if p.is_floating_point() else p for p in params]
        params = [p.tolist() for p in params]
        ID = hashlib.md5(str(params).encode('utf-8')).hexdigest()
        return ID

    @property
    def full_ind_to_ind_map(self):
        return {a.index: i for i, a in enumerate(self.atoms)}


class BondEnergy(PotentialEnergy):
    """
    Evaluates the bond energy for a given GROMACS topology
    References:
        GROMACS forces doc: https://manual.gromacs.org/current/reference-manual/functions/bonded-interactions.html
        GROMACS topology format: https://manual.gromacs.org/2024.0/reference-manual/topologies/topology-file-formats.html#id30
        GROMACS defines the parameters of each bond type in the ffbonded.itp file of the forcefield directory
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Omit bonds that involves omitted atoms
        bonds = [b for b in self.molecule.bonds if not (set(b[:2]) & set(self.omitted_atoms_id))]
        bonds_atom_ind = torch.tensor([[self.atom_id_to_ind[a] for a in b[:2]] for b in bonds])

        self.n_bonds = len(bonds)
        self.bonds_atom_ind = nn.parameter.Buffer(bonds_atom_ind, persistent=False)
        self.bonds_info = [[self.atoms[b_ind[0]], self.atoms[b_ind[1]]] for b_ind in bonds_atom_ind]

        sorted_type_dict = dict()
        for bond_type_key, bond_info in self.top_file._bondTypes.items():
            sorted_key = tuple(sorted(bond_type_key[:2]) + [bond_type_key[-1]])
            sorted_type_dict[sorted_key] = bond_info

        bonds_atom_type = [tuple(sorted([self.atom_id_to_type[t] for t in b[:-1]]) + [b[-1]]) for b in bonds]
        bonds_params = torch.tensor(np.array([sorted_type_dict[bt][3:] for bt in bonds_atom_type]).astype(float))
        self.b0 = nn.parameter.Buffer(bonds_params[:, 0], persistent=False)
        self.kb = nn.parameter.Buffer(bonds_params[:, 1], persistent=False)
        self.weights = None

    @property
    def bond_matrix(self):
        bond_matrix = torch.zeros(self.n_atoms, self.n_atoms, dtype=torch.bool)
        bond_matrix[self.bonds_atom_ind.unbind(-1)] = True
        bond_matrix[self.bonds_atom_ind.flip(dims=[-1]).unbind(-1)] = True
        return bond_matrix

    def bond_ind_with_atom(self, atom_name: str):
        """
        Find the index of all bonds that involve at least one of the given atom name.
        Args:
            atom_name: name of the atom
        Returns:
            set of bond index
        """
        atom_ind = torch.tensor([i for i, a in enumerate(self.atoms) if a.name.startswith(atom_name)])
        bond_is_atom_involved = torch.isin(self.bonds_atom_ind, atom_ind).any(dim=-1)
        return bond_is_atom_involved.nonzero().squeeze()

    def bond_length_std(self, T=298):
        """
        Returns the expected standard deviation of the bond length for a given temperature
        Args:
            T: temperatue (K)

        Returns:
            torch.float
        """
        kb_T = BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA * T * kelvin  # K_b*T
        b_std = torch.sqrt(kb_T.value_in_unit(kilojoules_per_mole) / self.kb)
        return b_std

    def calculate_bond_vec(self, atom_positions: torch.Tensor):
        return torch.diff(atom_positions[..., self.bonds_atom_ind, :], dim=-2).squeeze(dim=-2)

    def calculate_bond_length(self, atom_positions: torch.Tensor):
        bond_vec = self.calculate_bond_vec(atom_positions=atom_positions)
        bond_length = torch.linalg.norm(bond_vec, dim=-1)
        return bond_length

    def calculate_hierarchical_bond_weight(self, method='CA_inv_bond_dist'):
        # Build the graph of the protein
        g = networkx.Graph()
        nodes_id = self.bonds_atom_ind.flatten().unique().tolist()
        bonds_atom_ind_tup = [(b[0].item(), b[1].item()) for b in self.bonds_atom_ind]
        g.add_nodes_from([(b, {'resid': self.atoms[b].residue.id}) for b in nodes_id])
        g.add_edges_from([(*b, {'weight': 0.0}) for b in bonds_atom_ind_tup])

        # Calculate the shortest distance to the CA atom for each atom
        if method == 'CA_inv_bond_dist':
            residues_CA_atom_ind = {a.residue.id: self.atom_id_to_ind[a.id] for a in self.atoms if a.name == 'CA'}
            for i, data_dict in g.nodes(data=True):
                CA_atom_ind = residues_CA_atom_ind[data_dict['resid']]
                CA_dist = networkx.shortest_path_length(g, i, CA_atom_ind)
                data_dict['CA_dist'] = CA_dist

            inter_residue_weight = np.nan
            for start_ind, end_ind, data_dict in g.edges(data=True):
                start_resid = self.atoms[start_ind].residue.id
                end_resid = self.atoms[end_ind].residue.id
                if start_resid == end_resid:
                    data_dict['weight'] = 1 / max(g.nodes[start_ind]['CA_dist'], g.nodes[end_ind]['CA_dist'])
                else:
                    # Inter-residue bond
                    data_dict['weight'] = inter_residue_weight
            bond_weights = np.array([g.edges[b]['weight'] for b in bonds_atom_ind_tup])
            weight_min = np.nanmin(bond_weights)
            bond_weights[np.isnan(bond_weights)] = 1 / (1 / weight_min + 1)
        else:
            raise ValueError(f"method={method} is not supported.")

        bond_weights = torch.tensor(bond_weights)
        # self.weights = bond_weights / bond_weights.sum()
        return bond_weights

    def generate_bonds_name(self):
        bonds_name = generate_top_feature_label(self.bonds_atom_ind, self.atoms)
        return bonds_name

    def forward(self, atom_positions: torch.Tensor):
        atom_positions = self.pos_scale * atom_positions
        bonds_length = self.calculate_bond_length(atom_positions)
        if self.weights is None:
            bonds_energy = self.kb * torch.square(bonds_length - self.b0) / 2
        else:
            bonds_energy = self.weights * torch.square(bonds_length - self.b0)
        return bonds_energy

    def check(self, **kwargs):
        # OpenMM adds the harmonic bond energy terms of Urey-Bradley as a HarmonicBondForce.
        # Delete angles temporarily to deactivate the Urey-Bradley contributions.
        angles = self.molecule.angles
        self.molecule.angles = []

        self.run_openmm_tests(force_types=openmm.openmm.HarmonicBondForce, **kwargs)
        self.molecule.angles = angles


class BondAngleEnergy(PotentialEnergy):
    """
    Evaluates the bond angle energy for a given GROMACS topology
    References:
        GROMACS forces doc: https://manual.gromacs.org/current/reference-manual/functions/bonded-interactions.html
        GROMACS topology format: https://manual.gromacs.org/2024.0/reference-manual/topologies/topology-file-formats.html#id30
        GROMACS defines the parameters of each bond angle type in the ffbonded.itp file of the forcefield directory
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Omit bonds that involves omitted atoms
        angles = [b for b in self.molecule.angles if not (set(b[:3]) & set(self.omitted_atoms_id))]
        angles_atom_ind = torch.tensor([[self.atom_id_to_ind[a] for a in b[:3]] for b in angles])
        self.n_bond_angles = len(angles)
        self.angles_atom_ind = nn.parameter.Buffer(angles_atom_ind, persistent=False)

        # Sort the angle type in lexicographic order. The angle type is symmetric in swapping i,k
        def sort_type_key(type_key):
            if type_key[0] <= type_key[2]:
                sorted_key = (type_key[0], type_key[1], type_key[2])
            else:
                sorted_key = (type_key[2], type_key[1], type_key[0])
            return sorted_key

        sorted_type_dict = dict()
        for type_key, angle_info in self.top_file._angleTypes.items():
            sorted_type_dict[sort_type_key(type_key)] = angle_info

        angles_atom_type = [sort_type_key([self.atom_id_to_type[t] for t in b]) for b in angles]
        angles_info = np.array([sorted_type_dict[bt] for bt in angles_atom_type])
        angles_func_type = torch.tensor(angles_info[:, 3].astype(int))
        angles_params = torch.tensor(angles_info[:, 4:].astype(float)).to(torch.float)

        # Check if all types are supported
        supported_func_types = {1, 5}
        unsupported_func_types = set(angles_func_type.tolist()) - supported_func_types
        if unsupported_func_types:
            raise NotImplementedError(f"Bond angle energy is not implemented for types={unsupported_func_types!r}. "
                                      f"Supported types are {supported_func_types!r}.")

        # Convert angles to rad
        angles_params[:, 0] = torch.deg2rad(angles_params[:, 0])

        # Harmonic type
        harm_type_mask = torch.eq(angles_func_type, 1)
        self.n_harm_types = nn.parameter.Buffer(harm_type_mask.sum(), persistent=False)
        if self.n_harm_types > 0:
            self.harm_type_mask = nn.parameter.Buffer(harm_type_mask, persistent=False)
            self.harm_theta0 = nn.parameter.Buffer((angles_params[harm_type_mask, 0]), persistent=False)
            self.harm_k_theta = nn.parameter.Buffer(angles_params[harm_type_mask, 1], persistent=False)

        # Urey-Bradley type
        UB_type_mask = torch.eq(angles_func_type, 5)
        self.n_UB_types = nn.parameter.Buffer(UB_type_mask.sum(), persistent=False)
        if self.n_UB_types > 0:
            self.UB_type_mask = nn.parameter.Buffer(UB_type_mask, persistent=False)
            self.UB_theta0 = nn.parameter.Buffer((angles_params[UB_type_mask, 0]), persistent=False)
            self.UB_k_theta = nn.parameter.Buffer(angles_params[UB_type_mask, 1], persistent=False)
            self.UB_b0 = nn.parameter.Buffer(angles_params[UB_type_mask, 2], persistent=False)
            self.UB_k_ub = nn.parameter.Buffer(angles_params[UB_type_mask, 3], persistent=False)

    def calculate_endpoints_dist(self, atom_positions: torch.Tensor):
        bond_vec = torch.diff(atom_positions[..., self.angles_atom_ind[:, [0, 2]], :], dim=-2).squeeze(dim=-2)
        bond_length = torch.linalg.norm(bond_vec, dim=-1)
        return bond_length

    def calculate_bond_angle(self, atom_positions: torch.Tensor):
        angles = calc_bond_angle(atom_positions[..., self.angles_atom_ind, :], deg=False)
        return angles

    def generate_angles_name(self):
        bond_angle_names = generate_top_feature_label(self.angles_atom_ind, self.atoms)
        return bond_angle_names

    def forward(self, atom_positions: torch.Tensor):
        atom_positions = self.pos_scale * atom_positions
        bond_angles = self.calculate_bond_angle(atom_positions)
        bond_angle_energy = torch.zeros_like(bond_angles)
        if self.n_harm_types > 0:
            harm_energy = self.harm_k_theta * torch.square(bond_angles[..., self.harm_type_mask] - self.harm_theta0) / 2
            bond_angle_energy[..., self.harm_type_mask] = harm_energy

        if self.n_UB_types > 0:
            UB_energy = self.UB_k_theta * torch.square(bond_angles[..., self.UB_type_mask] - self.UB_theta0) / 2
            endpoints_bond_length = self.calculate_endpoints_dist(atom_positions)[..., self.UB_type_mask]
            UB_energy += self.UB_k_ub * torch.square(endpoints_bond_length - self.UB_b0) / 2
            bond_angle_energy[..., self.UB_type_mask] = UB_energy
        return bond_angle_energy

    def check(self):
        # Set all bond energies to zero to only count the Urey-Bradley bond term with openmm.HarmonicBondForce
        bond_types_dict = self.top_file._bondTypes
        mod_bond_types_dict = copy.deepcopy(bond_types_dict)
        for bond_type, bond_val in mod_bond_types_dict.items():
            bond_val[4] = '0'  # Set kb to 0
            mod_bond_types_dict[bond_type] = bond_val
        self.top_file._bondTypes = mod_bond_types_dict
        self.run_openmm_tests(force_types=[openmm.openmm.HarmonicBondForce, openmm.openmm.HarmonicAngleForce],
                              n_checks=5)
        self.top_file._bondTypes = bond_types_dict


class DihedralAngleEnergy(PotentialEnergy):
    """
    Evaluates the dihedral angle energy for a given GROMACS topology
    References:
        GROMACS forces doc: https://manual.gromacs.org/current/reference-manual/functions/bonded-interactions.html
        GROMACS topology format: https://manual.gromacs.org/2024.0/reference-manual/topologies/topology-file-formats.html#id30
        GROMACS defines the parameters of each dihedral type in the ffbonded.itp file of the forcefield directory
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Omit dihedrals that involves omitted atoms
        dihedrals = [b for b in self.molecule.dihedrals if not (set(b[:4]) & set(self.omitted_atoms_id))]
        dih_atom_ind = torch.tensor([[self.atom_id_to_ind[a] for a in b[:4]] for b in dihedrals])

        def format_type_key(type_key):
            # Sort the dihedral type for easier lookup.
            # The dihedral type is symmetric in reading forward or backward.
            if type_key > type_key[-2::-1]:
                type_key = tuple(type_key[-2::-1]) + (type_key[-1],)
            else:
                type_key = tuple(type_key)
            return type_key

        sorted_type_dict = {format_type_key(k): v for k, v in self.top_file._dihedralTypes.items()}

        # Define a function that finds a matching dihedral type in the ffbonded.itp table.
        # The function handles wildcard matching
        wilcard_masks = [[0, 0, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]]

        def find_matching_dihedral_types(type_key) -> list:
            matched_types = []
            for wildcard_mask in wilcard_masks:
                new_key = tuple(['X' if wildcard_mask[i] else a for i, a in enumerate(type_key[:4])] + [type_key[-1]])
                new_key = format_type_key(new_key)
                if new_key in sorted_type_dict:
                    matched_types.extend(sorted_type_dict[new_key])
                    if 'X' not in new_key:
                        break
            return matched_types

        # Useful for debugging key searches
        def is_key_match(key1, key2):
            is_match = all([i == j or i == 'X' for i, j in zip(key1[:-1], key2[:-1])])
            is_match |= all([i == j or i == 'X' for i, j in zip(key1[-2::-1], key2[:-1])])
            is_match &= key1[-1] == key2[-1]
            return is_match

        # Find all dihedral types that match the molecule's dihedrals.
        dihedrals_atom_type = [[self.atom_id_to_type[t] for t in b[:4]] + [b[4]] for b in dihedrals]
        dihedrals_info = []
        dihedrals_atom_ind = []
        for i, dih_type in enumerate(dihedrals_atom_type):
            # matching_keys = [k for k in sorted_type_dict if is_key_match(k, dih_type)]
            if len(dihedrals[i]) > 5:
                # Use the custom parameters defined in the .top file
                matched_dih_types = [dihedrals[i]]
            else:
                matched_dih_types = find_matching_dihedral_types(dih_type)
                if not matched_dih_types:
                    # Try a lookup of reversed key if not match were found
                    matched_dih_types = find_matching_dihedral_types(dih_type[-2::-1] + dih_type[-1:])

                if not matched_dih_types:
                    raise ValueError(f"No matching type found for dihedral #{i + 1} of type {dih_type}")

            for dih_type_params in matched_dih_types:
                # Get info about the dihedral type
                f_type = dih_type_params[4]
                if f_type in ['2', '4', '9']:
                    k = float(dih_type_params[6])
                else:
                    raise NotImplementedError

                # Skip dihedrals that are zeroed-out
                if np.isclose(k, 0.0):
                    continue

                if len(dih_type_params) == 7:
                    if dih_type_params[4] == '2':
                        # This type defines only two parameters. Add a placeholder for easier concatenation
                        dih_type_params = dih_type_params + ['0']
                    else:
                        raise NotImplementedError(f"dihedral type {dih_type_params} is not supported.")
                elif len(dih_type_params) != 8:
                    raise NotImplementedError(f"dihedral type {dih_type_params} is not supported.")
                dihedrals_info.append(np.array(dih_type_params))
                dihedrals_atom_ind.append(dih_atom_ind[i])

        dihedrals_info = np.array(dihedrals_info)
        dihedrals_atom_ind = torch.stack(dihedrals_atom_ind, dim=0)
        dihedrals_unique_atom_ind = torch.unique(dihedrals_atom_ind, dim=0)
        self.dihedrals_atom_ind = nn.parameter.Buffer(dihedrals_atom_ind, persistent=False)
        self.dihedrals_unique_atom_ind = nn.parameter.Buffer(dihedrals_unique_atom_ind, persistent=False)
        self.n_angles = dihedrals_atom_ind.shape[0]
        self.n_angles_unique = dihedrals_unique_atom_ind.shape[0]

        # dihedrals_info = np.array([sorted_type_dict[bt] for bt in dihedrals_atom_type])
        dihedrals_func_type = torch.tensor(dihedrals_info[:, 4].astype(int))
        supported_func_types = {2, 4, 9}
        unsupported_func_types = set(dihedrals_func_type.tolist()) - supported_func_types
        if not unsupported_func_types:
            # Parse through each dihedral type and define its parameters
            dih_params = torch.tensor(dihedrals_info[:, 5:].astype(float)).to(torch.float)

            # Convert angles to rad
            dih_params[:, 0] = torch.deg2rad(dih_params[:, 0])

            # Periodic dihedrals (proper or improper)
            Periodic_mask = torch.isin(dihedrals_func_type, torch.tensor([4, 9]))
            self.n_Periodic = nn.parameter.Buffer(Periodic_mask.sum(), persistent=False)
            if self.n_Periodic > 0:
                self.Periodic_mask = nn.parameter.Buffer(Periodic_mask, persistent=False)
                self.Periodic_phi_s = nn.parameter.Buffer(dih_params[Periodic_mask, 0], persistent=False)
                self.Periodic_k = nn.parameter.Buffer(dih_params[Periodic_mask, 1], persistent=False)
                self.Periodic_n = nn.parameter.Buffer(dih_params[Periodic_mask, 2], persistent=False)

            # Improper harmonic dihedrals
            ImpropHarm_mask = torch.eq(dihedrals_func_type, 2)
            self.n_ImpropHarm = nn.parameter.Buffer(ImpropHarm_mask.sum(), persistent=False)
            if self.n_ImpropHarm > 0:
                self.ImpropHarm_mask = nn.parameter.Buffer(ImpropHarm_mask, persistent=False)
                self.ImpropHarm_xi0 = nn.parameter.Buffer(dih_params[ImpropHarm_mask, 0], persistent=False)
                self.ImpropHarm_k = nn.parameter.Buffer(dih_params[ImpropHarm_mask, 1], persistent=False)
        else:
            raise NotImplementedError(f"Dihedral energy is not implemented for func types={unsupported_func_types!r}. "
                                      f"Supported types are {supported_func_types!r}.")

    def calculate_dihedral_angle(self, atom_positions: torch.Tensor, unique=False):
        """
        Calculates dihedral angle for each quartet of atoms
        Args:
            atom_positions: shape=(...,4,3) sets of 4 atom positions that participate in the dihedral angle
        Returns:
            torch.Tensor shape=(...,n_dihedrals) dihedral angles for each given set
        """
        if unique:
            atoms_ind = self.dihedrals_unique_atom_ind
        else:
            atoms_ind = self.dihedrals_atom_ind
        position_quartet = atom_positions[..., atoms_ind, :]
        dihedral_angles = calc_dihedral_angle(position_quartet, deg=False)
        return dihedral_angles

    def generate_angles_name(self, unique=False):
        if unique:
            atoms_ind = self.dihedrals_unique_atom_ind
        else:
            atoms_ind = self.dihedrals_atom_ind
        return generate_top_feature_label(atoms_ind, self.atoms)

    def forward(self, atom_positions: torch.Tensor):
        atom_positions = self.pos_scale * atom_positions
        dihedral_angles = self.calculate_dihedral_angle(atom_positions)
        dihedral_energy = torch.zeros_like(dihedral_angles)

        # Periodic proper dihedrals
        if self.n_Periodic > 0:
            phi = dihedral_angles[..., self.Periodic_mask]
            Periodic_U = self.Periodic_k * (1 + torch.cos(self.Periodic_n * phi - self.Periodic_phi_s))
            dihedral_energy[..., self.Periodic_mask] = Periodic_U

        # Periodic improper dihedrals
        if self.n_ImpropHarm > 0:
            xi = dihedral_angles[..., self.ImpropHarm_mask]
            ImpropHarm_U = 0.5 * self.ImpropHarm_k * torch.square(xi - self.ImpropHarm_xi0)
            dihedral_energy[..., self.ImpropHarm_mask] = ImpropHarm_U
        return dihedral_energy

    def check(self):
        system, simulation, positions = self.init_openmm()
        system_forces = system.getForces()

        # Check the Proper Periodic Force
        atom_ind = self.dihedrals_atom_ind[self.Periodic_mask, :].tolist()
        k, phi_s, n = self.Periodic_k.data, self.Periodic_phi_s.data, self.Periodic_n.data.long()
        omm_periodic_force = next(f for f in system_forces if isinstance(f, openmm.openmm.PeriodicTorsionForce))
        for i in range(omm_periodic_force.getNumTorsions()):
            omm_force_params = omm_periodic_force.getTorsionParameters(i)
            omm_force_params[-2:] = [v._value for v in omm_force_params[-2:]]
            if atom_ind[i] != omm_force_params[:4]:
                raise ValueError(f"atom index does not match for torsion force {i} with params={omm_force_params}")
            if not torch.isclose(k[i], torch.tensor(omm_force_params[-1])):
                raise ValueError(f"k value does not match for force {i} with params={omm_force_params}")
            if not torch.isclose(phi_s[i], torch.tensor(omm_force_params[-2])):
                raise ValueError(f"phi_s value does not match for force {i} with params={omm_force_params}")
            if not torch.isclose(n[i], torch.tensor(omm_force_params[-3])):
                raise ValueError(f"multiplicity value does not match for force {i} with params={omm_force_params}")

        # Check the Improper Periodic Force
        if self.n_ImpropHarm > 0:
            atom_ind = self.dihedrals_atom_ind[self.ImpropHarm_mask, :].tolist()
            k, xi0 = self.ImpropHarm_k.data, self.ImpropHarm_xi0.data
            omm_custom_torsion_force = next(f for f in system_forces if isinstance(f, openmm.openmm.CustomTorsionForce))
            for i in range(omm_custom_torsion_force.getNumTorsions()):
                omm_force_params = omm_custom_torsion_force.getTorsionParameters(i)
                if atom_ind[i] != omm_force_params[:4]:
                    raise ValueError(f"atom index does not match for torsion force {i} with params={omm_force_params}")
                if not torch.isclose(xi0[i], torch.tensor(omm_force_params[-1][0])):
                    raise ValueError(f"phi_s value does not match for force {i} with params={omm_force_params}")
                if not torch.isclose(k[i], torch.tensor(omm_force_params[-1][1])):
                    raise ValueError(f"k value does not match for force {i} with params={omm_force_params}")

        force_types = [openmm.openmm.PeriodicTorsionForce, openmm.openmm.CustomTorsionForce]
        self.run_openmm_tests(force_types=force_types)


class LJEnergy(PotentialEnergy):
    """
    Evaluates the Lennard-Jones energy for a given GROMACS topology
    References:
        GROMACS forces doc: https://manual.gromacs.org/current/reference-manual/functions/bonded-interactions.html
        GROMACS topology format: https://manual.gromacs.org/2024.0/reference-manual/topologies/topology-file-formats.html#id30
    """

    def __init__(self, *args, local=False, **kwargs):
        super().__init__(*args, **kwargs)
        exceptions_sigeps, particles_sigeps, is_pair_normal, is_pair_exception = self.gather_system_LJ_params()
        self.particles_sigeps = nn.parameter.Buffer(torch.tensor(particles_sigeps))
        self.local = local
        if not local:
            normal_pairs_ind = torch.nonzero(is_pair_normal)
            exceptions_ind = torch.nonzero(is_pair_exception)
            normal_pairs_sig, normal_pairs_eps = self.gather_normal_pairs_params(normal_pairs_ind)
            normal_pairs_sigeps = torch.stack([normal_pairs_sig, normal_pairs_eps], dim=1)

            pairs_sigeps = torch.cat([exceptions_sigeps.values(), normal_pairs_sigeps], dim=0)
            pairs_ind = torch.cat([exceptions_ind, normal_pairs_ind], dim=0)

            self.n_pairs = pairs_ind.shape[0]
            self.pairs_ind = nn.parameter.Buffer(pairs_ind, persistent=False)
            self.sigma = nn.parameter.Buffer(pairs_sigeps[:, 0], persistent=False)
            self.epsilon = nn.parameter.Buffer(pairs_sigeps[:, 1], persistent=False)
        else:
            self.exceptions_sigeps = nn.parameter.Buffer(exceptions_sigeps)
            self.is_pair_exception = nn.parameter.Buffer(is_pair_exception, persistent=False)
            self.is_pair_normal = nn.parameter.Buffer(is_pair_normal, persistent=False)

    def gather_system_LJ_params(self):
        # Gather parameters about exception pairs
        system = self.top_file.createSystem()
        nb_force = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
        is_pair_exception = torch.zeros(self.n_atoms, self.n_atoms, dtype=torch.bool)
        is_exception_omitted = torch.zeros(self.n_atoms, self.n_atoms, dtype=torch.bool)
        exceptions_sig_eps = []
        exceptions_ind = []
        atom_all_ind_to_ind = {v: k for k, v in self.atom_ind_to_all_ind.items()}

        for i in range(nb_force.getNumExceptions()):
            params = nb_force.getExceptionParameters(i)

            # Check if this exception involves omitted atoms. If yes, skip to next
            pair_all_atom_ind = params[:2]
            is_pair_omitted = any([ind not in atom_all_ind_to_ind for ind in pair_all_atom_ind])
            if is_pair_omitted:
                continue

            pair_ind = [atom_all_ind_to_ind[ind] for ind in pair_all_atom_ind]
            is_pair_exception[*pair_ind] = True
            is_exception_omitted[*pair_ind] = not params[-1]._value > 0
            if not is_exception_omitted[*pair_ind]:
                sig, eps = params[-2].value_in_unit(nanometer), params[-1].value_in_unit(kilojoules_per_mole)
                exceptions_sig_eps.append([sig, eps])
                exceptions_ind.append(pair_ind)
        exceptions_sig_eps, exceptions_ind = torch.tensor(exceptions_sig_eps), torch.tensor(exceptions_ind)
        # exceptions_sigeps = torch.sparse_coo_tensor(indices=exceptions_ind.T, values=exceptions_sig_eps,
        #                                             size=(self.n_atoms, self.n_atoms, 2)).coalesce()
        exceptions_ind = torch.unsqueeze(self.n_atoms * exceptions_ind[:, 0] + exceptions_ind[:, 1], dim=-1)
        exceptions_sigeps = torch.sparse_coo_tensor(indices=exceptions_ind.T, values=exceptions_sig_eps,
                                                    size=(self.n_atoms ** 2, 2)).coalesce()

        # Find all normal LJ pairs and remove the omitted exceptions in the pair mask
        is_pair_normal = (~is_pair_exception).triu(diagonal=1)
        is_pair_exception = torch.logical_and(is_pair_exception, ~is_exception_omitted)

        # Gather particles LJ parameters to calculate normal LJ pairs using combination rules
        particles_sigeps = np.zeros((self.n_atoms, 2))
        for i in range(self.n_atoms):
            _, sig, eps = nb_force.getParticleParameters(self.atom_ind_to_all_ind[i])
            particles_sigeps[i, :] = sig.value_in_unit(nanometer), eps.value_in_unit(kilojoules_per_mole)

        return exceptions_sigeps, particles_sigeps, is_pair_normal, is_pair_exception

    def calculate_pairs_params(self, system=None):
        if system is None:
            system = self.top_file.createSystem()
        nb_force = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
        is_pair_exception = torch.zeros(self.n_atoms, self.n_atoms, dtype=torch.bool)
        pairs_sig_eps = []
        pairs_ind = []
        atom_all_ind_to_ind = {v: k for k, v in self.atom_ind_to_all_ind.items()}

        for i in range(nb_force.getNumExceptions()):
            params = nb_force.getExceptionParameters(i)

            # Check if this exception involves omitted atoms. If yes, skip to next
            pair_all_atom_ind = params[:2]
            is_pair_omitted = any([ind not in atom_all_ind_to_ind for ind in pair_all_atom_ind])
            if is_pair_omitted:
                continue

            pair_ind = [atom_all_ind_to_ind[ind] for ind in pair_all_atom_ind]
            is_pair_exception[*pair_ind] = True
            if params[-1]._value > 0:
                sig, eps = params[-2].value_in_unit(nanometer), params[-1].value_in_unit(kilojoules_per_mole)
                pairs_sig_eps.append([sig, eps])
                pairs_ind.append(pair_ind)
        pairs_sig_eps, pairs_ind = np.array(pairs_sig_eps), np.array(pairs_ind)

        # Add normal LJ pairs
        particles_sig = np.zeros(self.n_atoms)
        particles_eps = np.zeros(self.n_atoms)
        for i in range(self.n_atoms):
            _, sig, eps = nb_force.getParticleParameters(self.atom_ind_to_all_ind[i])
            particles_sig[i], particles_eps[i] = sig.value_in_unit(nanometer), eps.value_in_unit(kilojoules_per_mole)

        normal_pairs_ind = torch.nonzero((~is_pair_exception).triu(diagonal=1))
        normal_pairs_sig, normal_pairs_eps = self.gather_normal_pairs_params(normal_pairs_ind)
        normal_pairs_sig_eps = np.stack([normal_pairs_sig, normal_pairs_eps], axis=1)

        pairs_sig_eps = np.concatenate([pairs_sig_eps, normal_pairs_sig_eps], axis=0)
        pairs_ind = np.concatenate([pairs_ind, normal_pairs_ind], axis=0)

        # pairs_sigma_eps = []
        # for i in range(self.n_atoms):
        #     params1 = nb_force.getParticleParameters(self.atom_ind_to_all_ind[i])
        #     sig1, eps1 = params1[1].value_in_unit(nanometer), params1[2].value_in_unit(kilojoules_per_mole)
        #     for j in range(i + 1, self.n_atoms):
        #         if not is_pair_exception[i, j]:
        #             pair_ind = [i, j]
        #             params2 = nb_force.getParticleParameters(self.atom_ind_to_all_ind[j])
        #             sig2, eps2 = params2[1].value_in_unit(nanometer), params2[2].value_in_unit(kilojoules_per_mole)
        #             sig_eps = [(sig1 + sig2) / 2, np.sqrt(eps1 * eps2)]
        #
        #             # pair_type_key = tuple(sorted([atom_types[self.atom_ind_to_all_ind[k]] for k in pair_ind]))
        #             # if pair_type_key not in pairtypes_sigma_eps:
        #             #     pairtypes_sigma_eps[pair_type_key] = sig_eps
        #             # elif pairtypes_sigma_eps[pair_type_key][0] != sig_eps[0] or pairtypes_sigma_eps[pair_type_key][1] != sig_eps[1]:
        #             #     raise ValueError
        #             # pairs_indices[pair_type_key].append(pair_ind)
        #
        #             pairs_sigma_eps.append(sig_eps)
        #             # pairs_ind.append(pair_ind)
        #
        #             # pairs_indices.append([i, j])
        #             # pairtypes_sigma_eps.append([(sig1 + sig2) / 2, np.sqrt(eps1 * eps2)])

        pairs_ind = torch.tensor(pairs_ind)
        pairs_sig, pairs_eps = torch.tensor(pairs_sig_eps).unbind(-1)
        return pairs_ind, pairs_sig, pairs_eps, is_pair_exception

    def gather_normal_pairs_params(self, pairs_ind: torch.Tensor):
        normal_pairs_sig = self.particles_sigeps[pairs_ind, 0].mean(dim=1)  # arithmetic mean
        normal_pairs_eps = torch.exp(torch.log(self.particles_sigeps[pairs_ind, 1]).mean(dim=1))  # geometric mean
        return normal_pairs_sig, normal_pairs_eps

    def calculate_pair_dist(self, atom_positions: torch.Tensor, pairs_ind: torch.Tensor = None):
        """
        Calculates distances between each Lennard-Jones pairs
        Args:
            atom_positions: shape=(...,3) sets of atom positions
            pairs_ind: shape=(n_pairs,2) indices of the n pairs of atoms. If none are given, use all possible pairs
        Returns:
            torch.Tensor shape=(...,n_pairs) distances between each pair
        """
        if pairs_ind is None:
            atom_pair_positions_pair = atom_positions[..., self.pairs_ind, :]
        else:
            if pairs_ind.shape[-1] == 2:
                atom_pair_positions_pair = atom_positions[..., pairs_ind, :]
            elif pairs_ind.shape[-1] == 3:
                atom_pair_positions_pair = atom_positions[pairs_ind[:, :1], pairs_ind[:, 1:], :]
            else:
                raise ValueError(f"Calculate distance undefined for pairs_ind.shape={pairs_ind.shape}")
        pair_vec = torch.diff(atom_pair_positions_pair, dim=-2).squeeze(-2)
        pair_dist = torch.linalg.norm(pair_vec, dim=-1)
        return pair_dist

    def forward(self, atom_positions: torch.Tensor, adj_mat: torch.Tensor = None, epsilon=1e-8):
        atom_positions = self.pos_scale * atom_positions
        if adj_mat is None:
            pairs_dist = self.calculate_pair_dist(atom_positions)
            sigma, eps = self.sigma, self.epsilon
        else:
            # LJ pairs (exceptions)
            pair_adj_except = torch.logical_and(adj_mat, self.is_pair_exception)
            pairs_except_ind = pair_adj_except.nonzero()
            pair_except_dist = self.calculate_pair_dist(atom_positions, pairs_ind=pairs_except_ind)

            pairs_except_ind1D = self.n_atoms * pairs_except_ind[:, 1] + pairs_except_ind[:, 2]
            pairs_except_sigeps = torch.index_select(self.exceptions_sigeps, dim=0, index=pairs_except_ind1D).to_dense()

            # LJ pairs (normal)
            pairs_adj_normal = torch.logical_and(adj_mat, self.is_pair_normal)
            pairs_normal_ind = pairs_adj_normal.nonzero()
            pairs_normal_dist = self.calculate_pair_dist(atom_positions, pairs_ind=pairs_normal_ind)
            pairs_normal_sig, pairs_normal_eps = self.gather_normal_pairs_params(pairs_normal_ind[:, 1:])

            # Combine pairs and their parameters
            pairs_dist = torch.cat([pair_except_dist, pairs_normal_dist])
            pairs_batch_ind = torch.cat([pairs_except_ind[:, 0], pairs_normal_ind[:, 0]])
            sigma = torch.cat([pairs_except_sigeps[:, 0], pairs_normal_sig])
            eps = torch.cat([pairs_except_sigeps[:, 1], pairs_normal_eps])

        pdist_scaled_inv = sigma / (pairs_dist + epsilon)
        pair_dist_inv_6 = torch.pow(pdist_scaled_inv, 6)
        pair_dist_inv_12 = pair_dist_inv_6.square()
        LJ_energy = 4 * eps * (pair_dist_inv_12 - pair_dist_inv_6)

        if adj_mat is not None:
            LJ_energy_batch = torch.zeros_like(atom_positions[:, 0, 0], dtype=LJ_energy.dtype)
            LJ_energy_batch.index_put_(indices=(pairs_batch_ind,), values=LJ_energy, accumulate=True)
            LJ_energy = LJ_energy_batch.unsqueeze(dim=-1)
        return LJ_energy

    def test(self, adj_matrix=False):
        positions = self.random_positions()
        if self.local:
            adj_matrix = torch.randn((positions.shape[0], self.n_atoms, self.n_atoms)) > 0
        else:
            adj_matrix = None
        energies = self.forward(positions, adj_mat=adj_matrix)
        return energies

    def check(self):
        system = self.top_file.createSystem()
        # The LJ force is a NonbondedForce in OpenMM.
        force_types = [openmm.openmm.NonbondedForce]

        # However, it also includes Coulomb interactions. Set all charges to zero-out the Coulomb force.
        nb_force = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
        for i in range(nb_force.getNumParticles()):
            part_params = nb_force.getParticleParameters(i)
            part_params[0] *= 0
            nb_force.setParticleParameters(i, *part_params)
        for i in range(nb_force.getNumExceptions()):
            except_params = nb_force.getExceptionParameters(i)
            except_params[2] *= 0
            nb_force.setExceptionParameters(i, *except_params)

        self.run_openmm_tests(force_types=force_types, system=system, epsilon=0.0)


class CMAPEnergy(PotentialEnergy):
    def __init__(self, *args, openmm_correction=False, **kwargs):
        super().__init__(*args, **kwargs)

        # Create a bicubic interpolator for each cmap
        system = self.top_file.createSystem()
        cmap_forces = [f for f in system.getForces() if isinstance(f, openmm.CMAPTorsionForce)]
        self.cmap_interp = []
        if cmap_forces:
            cmap_force = cmap_forces[0]
            cmaps = [cmap_force.getMapParameters(i) for i in range(cmap_force.getNumMaps())]

            # Get raw CMAPs. Maps are transposed since OpenMM use column-major ordering
            self.cmaps_raw = [np.array(map._value).reshape(s, s).T for s, map in cmaps]

            # In OpenMM, maps are defined from 0 to 360. Shift the maps' range to [-180,180].
            self.cmaps_raw = [np.roll(m, shift=m.shape[0] // 2, axis=(0, 1)) for m in self.cmaps_raw]

            # Add first row and column at the end to ensure its periodic
            self.cmaps_raw = [np.pad(m, pad_width=((0, 1), (0, 1)), mode='wrap') for m in self.cmaps_raw]

            for m in self.cmaps_raw:
                phi_val = np.linspace(-180, 180, m.shape[0])
                psi_val = np.linspace(-180, 180, m.shape[1])
                interpolator = RectBivariateSpline(x=phi_val, y=psi_val, z=m, kx=3, ky=3)
                self.cmap_interp.append(interpolator)

            # Find the atom indices that participate in the calculation of the dihedral angle that have a CMAP term
            cmap_ind = []
            dih1_atom_ind = []
            dih2_atom_ind = []
            full_ind_to_ind_map = self.full_ind_to_ind_map
            for i in range(cmap_force.getNumTorsions()):
                torsion_params = cmap_force.getTorsionParameters(i)
                all_atoms_ind = set(torsion_params[1:])
                if all(ind in full_ind_to_ind_map for ind in all_atoms_ind):
                    cmap_ind.append(torsion_params[0])
                    dih1_atom_ind.append([full_ind_to_ind_map[i] for i in torsion_params[1:5]])
                    dih2_atom_ind.append([full_ind_to_ind_map[i] for i in torsion_params[5:]])

            self.cmap_ind = nn.parameter.Buffer(torch.tensor(cmap_ind), persistent=False)
            self.dih1_atoms_ind = nn.parameter.Buffer(torch.tensor(dih1_atom_ind), persistent=False)
            self.dih2_atoms_ind = nn.parameter.Buffer(torch.tensor(dih2_atom_ind), persistent=False)
        else:
            self.cmap_ind = nn.parameter.Buffer(torch.empty((0, 1)), persistent=False)
            self.dih1_atoms_ind = nn.parameter.Buffer(torch.empty((0, 1)), persistent=False)
            self.dih2_atoms_ind = nn.parameter.Buffer(torch.empty((0, 1)), persistent=False)
        self.n_cmap_terms = len(self.cmap_ind)
        self.openmm_correction = openmm_correction

    def forward(self, atom_positions: torch.Tensor):
        if self.n_cmap_terms > 0:
            atom_positions = self.pos_scale * atom_positions
            dih1_angle, dih2_angle = self.calculate_dihedral_angles(atom_positions)
            CMAP_energy = torch.zeros_like(dih1_angle)
            for i, cmap_ind in enumerate(self.cmap_ind):
                energy_temp = self.cmap_interp[cmap_ind](dih1_angle[..., i], dih2_angle[..., i], grid=False)
                # energy_temp = self.cmap_interp[cmap_ind](dih1_angle[..., i], dih2_angle[..., i])
                CMAP_energy[..., i] = torch.tensor(energy_temp).to(dtype=CMAP_energy.dtype, device=CMAP_energy.device)
        else:
            CMAP_energy = torch.zeros_like(atom_positions)[..., 0, :1]
        return CMAP_energy

    def calculate_dihedral_angles(self, atom_positions: torch.Tensor):
        """
        Calculates dihedral angle for each quartet of atoms
        Args:
            atom_positions: shape=(...,4,3) sets of 4 atom positions that participate in the dihedral angle
        Returns:
            torch.Tensor shape=(...,n_dihedrals) dihedral angles for each given set
        """
        position_quartet1 = atom_positions[..., self.dih1_atoms_ind, :]
        dih1_angles = calc_dihedral_angle(position_quartet1, deg=True)
        position_quartet2 = atom_positions[..., self.dih2_atoms_ind, :]
        dih2_angles = calc_dihedral_angle(position_quartet2, deg=True)

        if self.openmm_correction:
            dih1_angles = (torch.clamp((dih1_angles + 360) % 360, max=345) - 180) % 360 - 180
            dih2_angles = (torch.clamp((dih2_angles + 360) % 360, max=345) - 180) % 360 - 180

        return dih1_angles, dih2_angles

    def check(self, **kwargs):
        system = self.top_file.createSystem()
        force_types = [openmm.openmm.CMAPTorsionForce]
        self.run_openmm_tests(force_types=force_types, system=system, tol=1e-1, **kwargs)

    def plot_cmaps(self, smooth=False, filename=''):
        # Plot or save CMAPs in a .pdf
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        if filename:
            plot_filepath = pathlib.Path(config.plots_dir, 'cmaps', filename).with_suffix('.pdf')
            plot_filepath.parent.mkdir(parents=True, exist_ok=True)
            pdf = PdfPages(plot_filepath)  # Save the plots in a multipage pdf

        for i in range(self.n_cmap_terms):
            cmap_raw = self.cmaps_raw[self.cmap_ind[i]]
            cmap_size = cmap_raw.shape[0]
            dx_raw = 360 / (cmap_size - 1)
            extent = ((180 + dx_raw / 2) * np.array([-1, 1, -1, 1])).tolist()
            if smooth:
                dx = dx_raw / 21
                dy = dx
                Nx = int((extent[1] - extent[0]) / dx)
                Ny = int((extent[3] - extent[2]) / dx)
                x_span = np.linspace(extent[0] + dx / 2, extent[1] - dx / 2, Nx)
                y_span = np.linspace(extent[2] + dy / 2, extent[3] - dy / 2, Ny)
                x, y = np.meshgrid(x_span, y_span)
                x2 = ((x + 180) % 360) - 180
                y2 = ((y + 180) % 360) - 180
                plotted_cmap = self.cmap_interp[self.cmap_ind[i]](x2, y2, grid=False)
            else:
                plotted_cmap = cmap_raw.T

            # Shift the CMAP to bring back to the [-180,180] range
            dih1_atom_ind = self.dih1_atoms_ind[i].tolist()
            dih2_atom_ind = self.dih2_atoms_ind[i].tolist()

            # Form the labels that identify the atoms participating in each dihedral
            dih1_label = generate_top_feature_label(dih1_atom_ind, self.atoms)
            dih2_label = generate_top_feature_label(dih2_atom_ind, self.atoms)

            # # Plot interpolation in 3D to check its accuracy
            # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            # ax.plot_surface(x, y, plotted_cmap, cmap='viridis', antialiased=False,alpha=0.5)
            # x_raw, y_raw = np.meshgrid(np.linspace(-180, 180, cmap_raw.shape[0]), np.linspace(-180, 180, cmap_raw.shape[1]))
            # ax.scatter(xs=x_raw.flatten(), ys=y_raw.flatten(), zs=cmap_raw.T.flatten(), c='black')

            fig = plt.figure()
            plt.imshow(plotted_cmap, extent=extent, origin='lower', interpolation=None)
            plt.xlabel(rf'$\phi$ ({dih1_label})')
            plt.ylabel(rf'$\psi$ ({dih2_label})')
            plt.xticks(90 * (np.arange(5) - 2))
            plt.yticks(90 * (np.arange(5) - 2))
            plt.colorbar(label='V [kJ/mol]')
            plt.clim(cmap_raw.min(), cmap_raw.max())
            if filename:
                pdf.savefig(figure=fig, dpi=500, pad_inches=None, bbox_inches='tight')
                plt.close(fig)
        if filename:
            pdf.close()
        else:
            plt.show(block=False)


class TotalEnergy(PotentialEnergy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.energy_modules = nn.ModuleList()
        self.energy_modules.append(BondEnergy(*args, **kwargs))
        self.energy_modules.append(BondAngleEnergy(*args, **kwargs))
        self.energy_modules.append(DihedralAngleEnergy(*args, **kwargs))
        self.energy_modules.append(LJEnergy(*args, **kwargs))
        self.n_terms = len(self.energy_modules)

    def forward(self, atom_positions: torch.Tensor):
        U = torch.stack([en_mod(atom_positions).sum(dim=-1) for en_mod in self.energy_modules], dim=-1)
        return U


def generate_top_feature_label(atoms_ind: list[list[int]] | torch.Tensor, topology_atoms: list[Atom]):
    """
    Generates a topological feature label that include names of residues and atoms that define the feature
    Args:
        atoms_ind: list of atom indices that define a set of n features. Can be list or Tensor of shape (n,d)
                   where n is the number of features and d is the number of atoms that define each feature.
        topology_atoms: atoms in the topology to be indexed by atoms_ind

    Returns:
        list of strings of len() = n
    """
    single_label = isinstance(atoms_ind[0], int)
    if single_label:
        atoms_ind = [atoms_ind]

    feats_name = []
    for ind in atoms_ind:
        atoms = [topology_atoms[i] for i in ind]
        atoms_res_id = [a.residue.id for a in atoms]
        if len(set(atoms_res_id)) == 1:
            # If all atoms belong to the same residue, add it at the beginning of the label
            atom_full_names = [f"{a.name}" for a in atoms]
            feat_name = f"{atoms[0].residue.name}{atoms[0].residue.id}:" + "-".join(atom_full_names)
        else:
            # If atoms belong different residues, add residue label for each atom.
            atom_full_names = [f"{a.residue.name}{a.residue.id}:{a.name}" for a in atoms]
            feat_name = "-".join(atom_full_names)

        feats_name.append(feat_name)

    if single_label:
        feats_name = feats_name[0]

    return feats_name


@functools.cache
def fetch_energy_module(top_file: str | pathlib.Path, element_class: str, mod_type: type[PotentialEnergy]):
    if element_class in [heavy_atoms_class_name, all_atoms_class_name]:
        return mod_type(top_file=top_file, element_class=element_class)
    else:
        return None


if __name__ == '__main__':
    from data.datasets import load_dataset

    dataset = load_dataset('nup98_12_1')

    # # Test error of bond energy induced by Gromacs 3 decimals precision
    # from analysis.stats import Statistics
    # import tqdm
    #
    # bond_en = BondEnergy(top_file=dataset.top_filepath, element_class=heavy_atoms_class_name)
    # bond_E_stats = None
    # for mol in tqdm.tqdm(dataset, desc='Calculating noisy bond energy'):
    #     noisy_pos = 1e-4 * torch.rand(1000, *mol.elements_position.shape) + mol.elements_position
    #     noisy_bond_energy = bond_en(noisy_pos).sum(-1)
    #     if bond_E_stats is None:
    #         bond_E_stats = Statistics.from_data(noisy_bond_energy)
    #     else:
    #         bond_E_stats += Statistics.from_data(noisy_bond_energy)
    # pass

    # # Test Energy modules
    # torch.manual_seed(0)

    # bond_en = BondEnergy(top_file=dataset.top_filepath, element_class=all_atoms_class_name)
    # #bond_en.check()
    # bond_names = bond_en.generate_bonds_name()
    # pass

    # # cmap_en = CMAPEnergy(top_file=dataset.top_filepath, element_class=all_atoms_class_name,
    # #                      openmm_correction=False)
    # cmap_en = CMAPEnergy(top_file=dataset.top_filepath, element_class=heavy_atoms_class_name)
    # # cmap_en.test()
    # # cmap_en.check(n_checks=10)
    # cmap_en.plot_cmaps()
    # # cmap_en.plot_cmaps(filename=f'{dataset.name}_cmaps.pdf', smooth=False)
    # # cmap_en.plot_cmaps(filename=f'{dataset.name}_cmaps_smooth.pdf', smooth=True)

    # angle_en = BondAngleEnergy(top_file=dataset.top_filepath, element_class=all_atoms_class_name)
    # # angle_en.test()
    # angle_en.check()
    # bond_angle_names = angle_en.generate_angles_name()

    # Dih_en = DihedralAngleEnergy(top_file=dataset.top_filepath, element_class=all_atoms_class_name)
    # Dih_en.check()
    # dih_angle_names = Dih_en.generate_angles_name()

    # # Dih_en = DihedralAngleEnergy(top_file=dataset.top_filepath, element_class=heavy_atoms_class_name)
    # # Dih_en.test()
    #
    # LG_en = LJEnergy(top_file=dataset.top_filepath, element_class=all_atoms_class_name)
    # LG_en.check()
    #
    # # LG_en = LJEnergy(top_file=dataset.top_filepath, element_class=heavy_atoms_class_name, local=True)
    # # LG_en.test(adj_matrix=True)
    #
    # tot_en = TotalEnergy(top_file=dataset.top_filepath, element_class=all_atoms_class_name)
    # tot_en.test()
    # pass

    # # Find the b0 parameter for all hydrogen bonds
    # bond_en = BondEnergy(top_file=dataset.top_filepath, element_class=all_atoms_class_name)
    # H_bond_ind = bond_en.bond_ind_with_atom(atom_name='H')
    # H_bond_b0 = bond_en.b0[H_bond_ind]
    # pass
    pass
