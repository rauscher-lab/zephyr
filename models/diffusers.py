import copy
import pathlib
import re
import warnings
import math
from collections.abc import Iterable

import networkx
import numpy as np
import torch
import tqdm
from openmm.app import GromacsTopFile
from torch import nn as nn
from torch.utils.data import DataLoader
import scipy

import config
from chemistry.grains import CoarseGrains
from chemistry.energy import BondEnergy, BondAngleEnergy, TotalEnergy
from data.data_classes import Biomolecule, c_alpha_class_name, c_alpha_sidechain_class_name, heavy_atoms_class_name, \
    all_atoms_class_name
from data.datasets import ProcessedDataset
from models.conditioners import Conditioner, CombinedConditioner, DiffusionConditioner, PositionConditioner, \
    CoarseCenterDistConditioner
from models.losses import AuxiliaryLoss
from models.noise_schedules import NoiseSchedule
from utils import rmsd_align, rmsd_align_transform, remove_centroid, repeat_iterator, mvn_cond_params, \
    find_first_element_of_class
from analysis.stats import StatisticsArray


class CorrelatedNoiser(nn.Module):
    def __init__(self, topology: str | GromacsTopFile | dict, element_class=heavy_atoms_class_name,
                 chain_scaling: str | float = None, centroid_std: float = None, d=3, pos_scale: float = 1.0,
                 sidechain_coarse_grains: CoarseGrains = None):
        super().__init__()
        self.d = d

        # Determine the number of atoms and residues
        if isinstance(topology, str | pathlib.Path | GromacsTopFile):
            bond_energy_mod = BondEnergy(top_file=topology, element_class=all_atoms_class_name)
            atoms = bond_energy_mod.atoms

            # Check that the atoms are ordered in ascending residue index
            atoms_res_ind = np.array([a.residue.index for a in atoms])
            atoms_res_ind_is_sorted = np.all(np.diff(atoms_res_ind) >= 0)
            if not atoms_res_ind_is_sorted:
                raise ValueError(f"The atoms in topology must be ordered in ascending residue index.")

            R_full = self.calculate_R_bonds_full(bond_energy_mod=bond_energy_mod)
            # bond_energy_mod_restricted = BondEnergy(top_file=topology, element_class=element_class)
            # R2 = self.calculate_R_bonds_full(bond_energy_mod=bond_energy_mod_restricted)
            mol_ex = Biomolecule.from_topology(top_filepath=topology, element_class=all_atoms_class_name)

            if element_class == c_alpha_class_name:
                atoms_ind = torch.tensor([a.name == 'CA' for a in atoms]).nonzero().squeeze(-1)
            elif element_class == heavy_atoms_class_name:
                atoms_ind = torch.tensor([not a.name.startswith('H') for a in atoms]).nonzero().squeeze(-1)
            elif element_class == all_atoms_class_name:
                atoms_ind = torch.arange(len(atoms))
            elif element_class == c_alpha_sidechain_class_name:
                atoms_ind = mol_ex.coarse_atom_ind(method='CA_SI', coarse_grains=sidechain_coarse_grains)
                atoms_ind = torch.sort(atoms_ind)[0]  # Sorting may not be necessary.
            else:
                raise ValueError(f"Correlated diffuser is undefined for element class {element_class!r}")

            # Restrict R to atoms belonging to the given element class
            C = R_full @ R_full.T
            R = torch.linalg.cholesky(C[atoms_ind[:, None], atoms_ind[None, :]])
            R = torch.round(R, decimals=14)

            self.n_residues = bond_energy_mod.topology.getNumResidues()
            self.n_atoms = len(atoms_ind)
            self.atoms_resid = torch.tensor([int(a.residue.id) for a in atoms])[atoms_ind]
            self.CA_atoms_ind = torch.tensor([a.name == 'CA' for a in atoms])[atoms_ind].nonzero().squeeze(-1)
        else:
            # Creates an ideal polymer chain where each residue has 1 atom (CA atom).
            bond_energy_mod = None
            self.n_residues = topology['n_residues']
            self.n_atoms = self.n_residues
            self.CA_atoms_ind = torch.arange(self.n_residues)
            R = torch.eye(self.n_residues).to(torch.double)

        # Determine the parameters for the R_g scaling
        # r,nu defines the targeted scaling or R_g ~ r N_r^nu where N_r denotes the number of residues
        # The reference quote values for the scaling law R_g ~ rho_0 N_p^nu where N_p is the number of peptide bonds (N_p = N_r-1).
        # We assume the same parameters for R_g ~ r N_r^nu.
        #
        # a defines the average inter-residue distance of distance between CA-atoms
        # a,r are given in nanometers, which matches the units of the bond lengths in the energy module
        #
        # References:
        # disordered scaling parameters:
        #   eq.3 in Hofmann 2012, "Polymer scaling laws ..." https://www.pnas.org/doi/pdf/10.1073/pnas.1207719109
        if chain_scaling == 'globular':
            chain_scaling_params = {'a': 0.1559 * np.sqrt(3), 'r': 0.2, 'nu': 0.4}
        elif chain_scaling == 'folded':
            a, lp, nu = 0.38, 0.53, 0.34
            r = np.sqrt(2 * lp * a / ((2 * nu + 1) * (2 * nu + 2)))  # eq.3 in reference
            chain_scaling_params = {'a': a, 'r': r, 'nu': nu}
        elif chain_scaling == 'disordered':
            # r is a rough average of experimental values cited in reference.
            chain_scaling_params = {'a': 0.38, 'r': 0.2, 'nu': 0.6}
        elif chain_scaling == 'disordered2':
            chain_scaling_params = {'a': 0.5, 'r': 0.2, 'nu': 0.6}
        elif chain_scaling is None:
            chain_scaling_params = {}
        elif isinstance(chain_scaling, float | torch.Tensor):
            # Fix the value of R_g by solving for r and nu
            R_g_target = float(chain_scaling)
            nu = 1.0
            r = R_g_target / self.n_residues ** nu
            chain_scaling_params = {'a': R_g_target / np.sqrt(self.d), 'r': r, 'nu': nu}
            # chain_scaling_params = {'a': np.sqrt(3) * r + np.sqrt(self.n_residues/3), 'r': r, 'nu': nu}
        else:
            raise ValueError(f"Chain scaling={chain_scaling} is not supported.")
        self.chain_scaling = chain_scaling
        self.chain_scaling_params = chain_scaling_params

        if self.chain_scaling:
            R_chain = self.calculate_R_chain(**chain_scaling_params, bond_energy_mod=bond_energy_mod)
            R = R @ R_chain.double()

        # # Restrict the R matrix to c-alpha only if element class = c_alpha is used
        # if element_class == c_alpha_class_name:
        #     R = R[self.CA_atoms_ind[:, None], self.CA_atoms_ind[None, :]]  # Keep only the c-alpha atoms
        #     self.CA_atoms_ind = torch.arange(self.CA_atoms_ind.numel())
        # elif element_class == heavy_atoms_class_name:
        #     pass
        # else:
        #     raise ValueError(f"Correlated diffuser is undefined for element class {element_class!r}")

        # Define R_center to rescale the centroid variance to the given variance.
        if centroid_std is not None:
            R_centroid_std = self.R_centroid_var(R=R).sqrt()
            # R_center enforces: centroid_std = (1-xi) * R_centroid_std -> xi = 1 - centroid_std/R_centroid_std
            R_center = self.calculate_R_center(xi=1.0 - centroid_std / R_centroid_std, R=R)
            R = R_center @ R

        # Parameters
        C = R @ R.T
        C_inv_R = torch.linalg.cholesky(torch.linalg.inv(C), upper=True)  # x^T C^-1 x = ||C_inv_R x||^2
        self.pos_scale = nn.parameter.Buffer(torch.tensor(1.0), persistent=True)  # Track the position scale in 'nm'
        self.R = nn.parameter.Buffer(R, persistent=True)
        self.C = nn.parameter.Buffer(C, persistent=True)
        self.C_inv_R = nn.parameter.Buffer(C_inv_R, persistent=True)

        self.change_scale(pos_scale)

    def calculate_R_chain(self, a: float, r: float, nu: float, bond_energy_mod: BondEnergy = None):
        b_eff = self.b_eff(a, r, nu, method='approx')
        # b_eff = self.b_eff(a, r, nu, method='exact')
        v = 1 / np.sqrt(1 - b_eff ** 2)

        # First, build the matrix for c-alpha positions for 1 coordinate
        R = torch.zeros((self.n_residues, self.n_residues))
        b_powers = 1.0
        for i in range(self.n_residues):
            R += torch.diag(b_powers * torch.ones(self.n_residues - i), diagonal=-i)
            b_powers *= b_eff
        R[:, 0] *= v
        R *= a

        # Find indices of the c-alphas in the topology
        if bond_energy_mod is not None:
            N_atoms = len(bond_energy_mod.atoms)
            R_full = torch.eye(N_atoms)
            R_full[self.CA_atoms_ind[:, None], self.CA_atoms_ind[None, :]] = R
        else:
            # Assume each residue has 1 atom
            R_full = R

        if R_full.isnan().any():
            raise ValueError(f"Found nans in R_chain")

        return R_full

    def calculate_R_residue_coarse(self, sigma_res, bond_energy_mod: BondEnergy):
        residues_atoms = [[] for _ in range(self.n_residues)]
        for a in bond_energy_mod.atoms:
            residues_atoms[a.residue.index].append(a)
        residues_R = []
        for res_atom in residues_atoms:
            N_atoms = len(res_atom)
            CA_ind = [a.name for a in res_atom].index('CA')
            res_block = torch.diag(sigma_res / np.sqrt(self.d) * torch.ones(N_atoms))
            res_block[:, CA_ind] = 1.0
            residues_R.append(res_block)

        R = torch.block_diag(*residues_R)
        return R

    def calculate_R_residue_bonds(self, bond_energy_mod: BondEnergy):
        residues_atoms = [[] for _ in range(self.n_residues)]
        global_to_local_ind = {}  # Maps glocal atom ind to residue-local index
        for i, a in enumerate(bond_energy_mod.atoms):
            residues_atoms[a.residue.index].append(a)
            global_to_local_ind[i] = len(residues_atoms[a.residue.index]) - 1

        residues_bonds = [[] for _ in range(self.n_residues)]
        bonds_atom_ind = bond_energy_mod.bonds_atom_ind.data.tolist()
        bonds_length = bond_energy_mod.b0.data.tolist()
        for bond, b0 in zip(bonds_atom_ind, bonds_length):
            atom1_res_ind = bond_energy_mod.atoms[bond[0]].residue.index
            atom2_res_ind = bond_energy_mod.atoms[bond[1]].residue.index

            # Skip inter-residue bonds
            if atom1_res_ind != atom2_res_ind:
                continue
            bond = [global_to_local_ind[ind] for ind in
                    bond]  # Map the bond atom global indices to residue-local indices
            residues_bonds[atom1_res_ind].append(bond + [b0])

        # Build a graph of the protein to facilitate finding the bond path for each atom in a given residue
        residues_R = []
        for res_atom, res_bonds in zip(residues_atoms, residues_bonds):
            # For each atom in the residue, find the shortest path to the CA atom
            g = networkx.Graph()
            g.add_edges_from([(b[0], b[1], {'b0': b[2]}) for b in res_bonds])
            paths: dict[int, dict[int, list]] = dict(networkx.all_pairs_shortest_path(g))

            atoms_global_ind = [bond_energy_mod.atom_id_to_ind[a.id] for a in res_atom]
            CA_atom_local_ind = [a.name for a in res_atom].index('CA')
            CA_global_ind = atoms_global_ind[CA_atom_local_ind]
            CA_atom = [a for a in res_atom if a.name == 'CA'][0]
            CA_atom_ind = bond_energy_mod.atom_id_to_ind[CA_atom.id]

            # For each atom in the residue find the shortest path to the CA atom.
            R_res = torch.zeros(2 * [len(g)])
            for i in range(R_res.shape[0]):
                if i == CA_atom_local_ind:
                    continue
                path = paths[CA_atom_local_ind][i]
                n_edges = len(path) - 1
                path_bonds_length = torch.tensor([g.edges[(path[i], path[i + 1])]['b0'] for i in range(n_edges)])
                R_res[i, path[1:]] = path_bonds_length / np.sqrt(self.d)

            R_res[:, CA_atom_local_ind] = 1.0
            residues_R.append(R_res)

        R = torch.block_diag(*residues_R)
        return R

    def calculate_R_bonds_full(self, bond_energy_mod: BondEnergy):
        n_atoms = bond_energy_mod.n_atoms
        bonds_atom_ind = bond_energy_mod.bonds_atom_ind.data.tolist()
        bonds_length = bond_energy_mod.b0.data.tolist()
        bonds_length_2D = torch.sparse_coo_tensor(bond_energy_mod.bonds_atom_ind.T, bond_energy_mod.b0,
                                                  size=(n_atoms, n_atoms))
        # R = define_correlated_R(pair_dist=bonds_length_2D)

        # For each atom, find all bonds that ends at the given atom
        atoms_bond_pair_ind = [[] for _ in range(n_atoms)]
        atoms_bond_length = [[] for _ in range(n_atoms)]
        for i, (bond_ind, b0) in enumerate(zip(bonds_atom_ind, bonds_length)):
            atoms_bond_pair_ind[bond_ind[1]] += [bond_ind[0]]
            atoms_bond_length[bond_ind[1]] += [b0]

        R = torch.eye(n_atoms).double().to_sparse_coo()
        template_ind = torch.arange(n_atoms).unsqueeze(-1).repeat((1, 2))
        template_val = torch.ones((n_atoms,)).double()
        for i, (bond_start_ind, bond_lengths) in enumerate(zip(atoms_bond_pair_ind, atoms_bond_length)):
            if not bond_start_ind:
                continue
            bond_start_ind = torch.tensor(bond_start_ind)
            dist_std = torch.tensor(bond_lengths).max()
            n_bonds = bond_start_ind.shape[0]

            # To ensure that the end atom of the bond is on average b0 away from the first atom, we must have
            # R[bind2, bind1] = 1, R[bind2,bind2] = b0/np.sqrt(3) where bond_ind = [bind1,bind2]

            # Position atom i such that it is bond_lengths_avg away from the average position of all atoms that share a bond with atom i.
            R_2D_ind = torch.ones((n_bonds, 2), dtype=torch.long)
            R_2D_ind[:, 0] = i
            R_2D_ind[:, 1] = bond_start_ind
            sparse_ind = torch.cat([template_ind, R_2D_ind])
            sparse_val = torch.cat([template_val, 1 / n_bonds * torch.ones((n_bonds,))])
            sparse_val[i] = dist_std / np.sqrt(self.d)
            R_temp = torch.sparse_coo_tensor(sparse_ind.T, sparse_val, R.size())
            R = R_temp @ R

        # for i, (bond_ind, b0) in enumerate(zip(bonds_atom_ind, bonds_length)):
        #     # To ensure that the end atom of the bond is b0 away from the first atom, we must have
        #     # R[bind2, bind1] = 1, R[bind2,bind2] = b0/np.sqrt(3) where bond_ind = [bind1,bind2]
        #     template_val[bond_ind[1]] = b0 / np.sqrt(3)
        #     template_ind[-1] = torch.tensor(bond_ind[::-1])
        #     R_temp = torch.sparse_coo_tensor(template_ind.T, template_val, R.size())
        #     R = R_temp @ R
        #
        #     # Undo changes to the template
        #     template_val[bond_ind[1]] = 1.0
        R = R.to_dense()
        return R

    def calculate_R_bonds_length_and_angle(self, bond_length_en_mod: BondEnergy, bond_angle_en_mod: BondAngleEnergy):
        n_atoms = bond_length_en_mod.n_atoms
        n_bonds = bond_length_en_mod.n_bonds
        n_bond_angles = bond_angle_en_mod.n_bond_angles
        n_constraints_max = n_bonds + n_bond_angles + 1
        A = torch.zeros(n_constraints_max, n_atoms, n_atoms)
        b = torch.zeros(n_constraints_max)
        n = 0

        # Add first bond length constraint
        A[n, 0, 0] = 1
        b[n] = 1.0
        n += 1

        # Add bond length constraints
        for l, (i, j) in enumerate(bond_length_en_mod.bonds_atom_ind.sort(dim=-1)[0]):
            A[n, i, i] = 1
            A[n, i, j] = -2
            A[n, j, j] = 1
            b[n] = bond_length_en_mod.b0[l] ** 2
            n += 1
            # if n > 5:
            #     break

        # Add bond angle constraints
        bond_lengths_dict = {tuple(k.tolist()): v for k, v in
                             zip(bond_length_en_mod.bonds_atom_ind, bond_length_en_mod.b0)}
        for l, (i, j, k) in enumerate(bond_angle_en_mod.angles_atom_ind.sort(dim=-1)[0].tolist()):
            A[n, i, k] = 1
            A[n, i, j] = -1
            A[n, j, k] = -1
            A[n, j, j] = 1

            if bond_angle_en_mod.UB_type_mask[l]:
                theta = bond_angle_en_mod.UB_b0[l]
            elif bond_angle_en_mod.harm_type_mask[l]:
                theta = bond_angle_en_mod.harm_theta0[l]

                # # Add the constraint only if both bonds are constrained
                # if (i, j) in bond_lengths_dict and (j, k) in bond_lengths_dict:
                #     b[n] = bond_lengths_dict[(i, j)] * bond_lengths_dict[(j, k)] * torch.cos(theta)
                #     n += 1

            if (i, j) in bond_lengths_dict and (j, k) in bond_lengths_dict:
                b[n] = bond_lengths_dict[(i, j)] * bond_lengths_dict[(j, k)] * torch.cos(theta)
                n += 1
            else:
                print('constraint omitted')

        # Solve for the covariance matrix that satisfies the constraints
        tril_ind = torch.tril_indices(n_atoms, n_atoms, offset=0)
        triu_ind = torch.triu_indices(n_atoms, n_atoms, offset=0)

        A_1D = A.transpose(-1, -2)[:, *tril_ind]  # Take the lower triangular part only since C is symmetric
        c_lb = torch.where(triu_ind[0] == triu_ind[1], 1e-3 * torch.ones(triu_ind.shape[-1]),
                           -torch.inf * torch.ones(triu_ind.shape[-1]))
        c_up = torch.inf * torch.ones(triu_ind.shape[-1])
        c_bounds = (c_lb.numpy(), c_up.numpy())

        # A_1D_opt, b_opt = A_1D.numpy()[:n], b.numpy()[:n]
        A_1D_opt, b_opt = A_1D[:n], b[:n]

        # def obj_func(x):
        #     R = np.zeros((n_atoms, n_atoms))
        #     R[*tril_ind] = x
        #     C = R @ R.transpose()
        #     obj = np.sum(((A_1D_opt @ C[*tril_ind].flatten() - b_opt) / np.abs(b_opt)) ** 2)
        #     return obj

        # Optimize with LBFGS
        R = torch.zeros((n_atoms, n_atoms))
        x = torch.eye(n_atoms)[*tril_ind]
        x.requires_grad = True
        optim = torch.optim.LBFGS([x], max_iter=10, history_size=10)

        def obj_func():
            optim.zero_grad()
            R = torch.zeros((n_atoms, n_atoms))
            R[*tril_ind] = x
            C = R @ R.t()
            obj = torch.sum(((A_1D_opt @ C[*tril_ind].flatten() - b_opt) / torch.abs(b_opt)) ** 2)
            obj.backward()
            return obj

        obj_val_prev = torch.tensor(torch.inf)
        verbose, max_iter, tol = True, 1000, 1e-8
        for i in range(max_iter):
            obj_val = optim.step(closure=obj_func)
            obj_val_rel_diff = torch.abs(1 - obj_val_prev / obj_val)
            if verbose:
                print(f'{i}:SE={obj_val}, rel_diff={obj_val_rel_diff}')
            if obj_val_rel_diff < tol:
                break
                pass
            obj_val_prev = obj_val

        # R0 = torch.eye(n_atoms)[*tril_ind]
        # opt_res = scipy.optimize.least_squares(obj_func, x0=R0[*tril_ind].numpy())

        # opt_res = scipy.optimize.lsq_linear(A_1D[:n], b[:n], bounds=c_bounds)
        R = torch.zeros(n_atoms, n_atoms)
        R[*tril_ind] = x
        # R[*tril_ind] = torch.tensor(opt_res.x, dtype=torch.float)
        C = R @ R.t()

        # lstsq_results = torch.linalg.lstsq(A_1D[:n], b[:n])
        # c = lstsq_results.solution
        # c = torch.tensor(opt_res.x).to(torch.float)
        # C_upper = torch.zeros(n_atoms, n_atoms)
        # C_upper[*triu_ind] = c
        # C = (C_upper + C_upper.t()) / 2

        # C = torch.zeros(n_atoms, n_atoms)
        # C[triu_ind[0, :], triu_ind[1, :]] = c
        # C[triu_ind[1, :], triu_ind[0, :]] = c
        # C_diag = C.diag()
        # C_diag_zero = C_diag.isclose(torch.tensor(0.0))
        # C[C_diag_zero, C_diag_zero] = 1.0

        R = torch.cholesky(C)
        pass

    def calculate_R_center(self, xi: float, R: torch.Tensor):
        R = torch.eye(R.shape[0]) - xi * torch.ones_like(R) / R.shape[0]
        return R

    def change_scale(self, pos_scale: float | torch.Tensor):
        """
        Changes the scale of the correlated diffuser noise to the given scale.
        Args:
            pos_scale: float corresponding to value of the new position scale in units of 'nm'. eg. 0.1 -> angstroms

        Returns:
            None
        """
        # conv_factor = pos_scale / self.pos_scale.data
        conv_factor = self.pos_scale.data / pos_scale
        self.pos_scale.data /= conv_factor
        self.R.data *= conv_factor
        self.C.data *= conv_factor ** 2
        self.C_inv_R.data *= conv_factor ** (-1)

    @property
    def dim_var(self):
        """
        Calculates the variance of positions along each spatial dimension averaged over all positions
        Returns:
            scalar
        """
        # avg_var(x_i) = 1/n sum_i var(x_i) = 1/n sum_ijk <R_ik R_ij eps_j eps_k> = 1/n sum_ij R_ij R_ij = tr(R R.T)/n
        return torch.trace(self.R @ self.R.T) / self.R.shape[0]

    @property
    def dim_prec(self):
        """
        Calculates the precision of positions averaged over all positions and dimension
        Returns:
            scalar
        """
        # avg_prec(x_i) = 1/n sum_i prec(x_i) = 1/n sum_ijk <R_inv_ik R_inv_ij eps_j eps_k>
        #                                     = 1/n sum_ij R_inv_ij R_inv_ij = tr(R_inv R_inv.T)/n
        return torch.trace(self.C_inv_R.T @ self.C_inv_R) / self.R.shape[0]

    def R_centroid_var(self, R=None):
        """
        Calculates the variance of the 1D centroid of R*z where z is a standard normal variable (z ~ N(0,1)).
        Args:
            R: input R. If None, uses self.R
        Returns:
            scalar
        """
        if R is None:
            R = self.R
        # The centroid of R*z has a variance of 1/N^2 * sum_ij C_ij where C = R R^T:
        # Proof (Einstein summation implied):
        # com = 1/N*sum_i R_ij z_j  -> var(com) = 1/N^2*sum_ik <R_ij z_j R_kl z_l>
        #                                       = 1/N^2sum_ik R_ij R_kj
        #                                       = 1/N^2sum_ik (R @ R^T)_ik
        return torch.mean(R @ R.T)

    @property
    def Rg(self):
        """
        Calculates the square root of the expected value of Rg^2, i.e. sqrt(<Rg^2>), of the generated samples.
        Returns:
            scalar corresponding to sqrt(<Rg^2>)
        """
        # This formula is derived by evaluating <Rg^2> = 1/(2*N^2) sum_ij <d_ij^2>
        #                                              = 1/(2*N^2) sum_ij <(xi - xj)^2>  (x = R @ z , z ~ N(0,1))
        #                                              = 1/(2*N^2) sum_ij <x_i^2> - 2 <x_i x_j> + <x_j^2>
        #                                              = 1/(N^2) (N sum_i <x_i^2> - sum_ij <x_i x_j>)
        #                                              = tr(C)/N - 1/N^2 sum_ij C_ij where C = R @ R^T
        return torch.sqrt(self.d * (self.C.trace() / self.C.shape[0] - self.C.mean()))

    @property
    def Rg_CA(self):
        """
        Calculates the square root of the expected value of Rg^2, i.e. sqrt(<Rg^2>), accounting for CA atoms only
        Returns:
            scalar corresponding to sqrt(<Rg^2>)
        """
        C_CA = self.C[self.CA_atoms_ind[:, None], self.CA_atoms_ind[None, :]]
        return torch.sqrt(self.d * (C_CA.trace() / C_CA.shape[0] - C_CA.mean()))

    def b_eff(self, a, r, nu, method='exact'):
        N = np.double(self.n_residues)
        l = np.sqrt(self.d) * a  # The formula for b_eff in the Chroma paper is wrong. Replace a by l=sqrt(d)*a.
        b_prec = 1e-4
        b_min, b_max = b_prec, 1 - b_prec
        if method == 'approx':
            A = 3 / N
            C, D = N ** (2 * (nu - 1)) * (N ** 2 + 9), (l / r) ** 2
            if C >= D:
                B = N ** (-nu) * np.sqrt(C - D)
                b = A + B
            else:
                b = b_max
        elif method == 'exact':
            Rg2_exp_num = lambda b: 2 * b ** (N + 1) - N * (N + 1) * b ** 2 + 2 * (N ** 2 - 1) * b - N * (N - 1)
            Rg2_exp_denom = lambda b: (b - 1) ** 3 * (b + 1) * N ** 2
            obj_func = lambda b: Rg2_exp_num(b) / Rg2_exp_denom(b) - (r / a * N ** nu) ** 2
            # x0 = 3 / N + N ** (-nu) * np.sqrt(N ** (2 * (nu - 1)) * (N ** 2 + 9) - (l / r) ** 2)
            opt_results = scipy.optimize.least_squares(obj_func, x0=0.5, bounds=(0, 1))
            b = opt_results.x[0]
            # x_opt = scipy.optimize.fsolve(obj_func, x0=0.5)
            # b = np.clip(x_opt[0], 1e-4, 1.0 - 1e-4)
            # if np.abs(opt_results.fun) > 1e-5:
            #     warnings.warn(f'b s')
            # if opt_results.success:
            #     b = opt_results.x[0]
            # else:
            #     warnings.warn(f"Exact method to solve for b in {self} did not succeed. Resorting to 'approx' method.")
            #     b = self.b_eff(a=a, r=r, nu=nu, method='approx')

        if np.isnan(b):
            b = b_max

        if not (b_min <= b <= b_max):
            warnings.warn(f"b is out of bounds for scaling={self.chain_scaling} and N={self.n_residues}.\n"
                          f"Clipping b={b} to [{b_min},{b_max}]. Exact chain scaling will not be achieved.")
            b = np.clip(b, b_min, b_max)

        # if np.isnan(b):
        #     raise ValueError(f"b_eff is nan. Consider decreasing the a/r ratio.")
        # if b > 1.0:
        #     raise ValueError(f"b_eff ({b}) is greater than 1. Consider increasing the a/r ratio.")
        return b

    @property
    def R_3D(self):
        # Defines a 3D R matrix that extends the 1D R to all spatial dimensions.
        # The final matrix can be applied on a vector of flattened coordinates [X_0,Y_0,Z_0, X_1,Y_1,Z_1, ..., Z_N-1]
        N, d = self.n_atoms, self.d
        R2 = torch.eye(d, device=self.R.device, dtype=self.R.dtype)
        R_prod = torch.kron(R2, self.R)  # coordinates: [X_0,X_1,X_2,..., Y_0,Y_1,Y_2,..., Z_0,Z_1,Z_2,...]
        ind = torch.flatten(torch.arange(N).reshape(-1, 1) + (N * torch.arange(d)).reshape((1, -1)))
        R_full = R_prod[ind[:, None], ind[None, :]]
        return R_full

    def generate_samples(self, n_samples=1, size=None, device=None):
        if size is None:
            size = (n_samples, self.R.shape[0], 3)
        eps = torch.randn(size, device=device, dtype=self.R.dtype)
        samples = (self.R @ eps).squeeze(0)
        return samples

    def generate_CA_pos_cond_samples(self, CA_pos: torch.Tensor, **kwargs):
        CA_mask = torch.zeros((self.n_atoms, 3), dtype=torch.bool, device=CA_pos.device)
        CA_mask[self.CA_atoms_ind, :] = True
        return self.generate_cond_samples(y=CA_pos, cond_mask=CA_mask, **kwargs)

    def C_mul(self, x: torch.Tensor):
        x_new = self.C @ x
        return x_new

    def C_inv_R_mul(self, x: torch.Tensor):
        x_new = self.C_inv_R @ x
        return x_new

    def plot_sample(self, sample=None):
        if sample is None:
            sample = self.generate_samples()
        coords_sample = sample.T
        CA_ind = self.CA_atoms_ind

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # points_color = [int(a.residue.id) for a in self.bond_energy_module.atoms]
        points_color = self.atoms_resid
        ax.scatter(*coords_sample, c=points_color, cmap='jet')
        ax.plot(*coords_sample[:, CA_ind], color='k')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    @classmethod
    def test_Rg_scaling(cls, N_range=np.arange(50) + 1, scaling_types: list = None, **constructor_kwargs):
        """
        Plots the R_g scaling of generated samples with the expected scaling.
        Args:
            **constructor_kwargs: kwargs given to the constructor. 'chain_scaling_type' is ignored
        Returns:
            None
        """
        if scaling_types is None:
            scaling_types = ['globular', 'folded', 'disordered']
        n_samples = int(1e5)
        Rg2_expected = dict()
        Rg2_calculated_avg = {k: np.zeros_like(N_range, dtype=float) for k in scaling_types}
        Rg2_calculated_SEM = {k: np.zeros_like(N_range, dtype=float) for k in scaling_types}
        for scaling_type in scaling_types:
            for i, N in enumerate(N_range):
                corr_diffuser = cls(topology={'n_residues': N}, chain_scaling=scaling_type, **constructor_kwargs)

                # Change the number of residues in R
                samples_pos = corr_diffuser.generate_samples(n_samples=n_samples)

                # Calculate R_g
                Rg_squared = torch.var(samples_pos, correction=0, dim=1).sum(dim=-1)
                Rg2_calculated_avg[scaling_type][i] = Rg_squared.mean()
                Rg2_calculated_SEM[scaling_type][i] = (Rg_squared.std() / np.sqrt(Rg_squared.shape[0] - 1))

            # Calculate the expected R_g scaling
            r, nu = [corr_diffuser.chain_scaling_params[k] for k in ['r', 'nu']]
            Rg2_expected[scaling_type] = (r * N_range ** nu) ** 2

        # Plot the calculated and expected scaling for each type
        import matplotlib.pyplot as plt
        fig = plt.figure()
        for scaling_type in scaling_types:
            Rg_exp = Rg2_expected[scaling_type]
            Rg_avg = Rg2_calculated_avg[scaling_type]
            Rg_err = Rg2_calculated_SEM[scaling_type]
            exp_line_h = plt.plot(N_range, Rg_exp, label=f"{scaling_type} - exp.")[0]
            plt.fill_between(x=N_range, y1=Rg_avg - Rg_err, y2=Rg_avg + Rg_err, color=exp_line_h.get_color(), alpha=0.5)
            plt.plot(N_range, Rg_avg, '--', label=f"{scaling_type} - est.", color=exp_line_h.get_color())

        # plt.xscale('log'), plt.yscale('log')
        plt.legend(), plt.xlabel('N'), plt.ylabel('R_g^2 (nm)')

    def generate_cond_samples(self, y: torch.Tensor, cond_mask: torch.Tensor, sigma=1.0, n_samples=None):
        """
        Generates conditional samples x such that x[cond_mask] = y
        Args:
            y: values of the conditioned variables shape = (M,3) or (B,M,3) B=Batch size, M=# of conditioned variables
            cond_mask: mask identifying the conditioned variables. shape = (n_atoms,3)

        Returns:
            x conditioned samples generated with the associate covariance matrix
        """
        y_is_batched = y.ndim > 2
        if y_is_batched:
            n_samples = y.shape[0]
        else:
            y = y.unsqueeze(0)
            n_samples = 1 if n_samples is None else n_samples
        y = y.to(self.R.dtype).flatten(-2, -1)

        # Define the mean and covariance of the unconditioned variables
        cons_kwargs = dict(device=self.R.device, dtype=self.R.dtype)
        mu_3D = torch.zeros(1, self.R_3D.shape[0], **cons_kwargs)
        R_3D = self.R_3D
        C_3D = (R_3D @ R_3D.T).unsqueeze(0)
        cond_mask_3D = cond_mask.flatten()
        mu_x, covar_x = mvn_cond_params(mu=mu_3D, C=C_3D, cond_mask=cond_mask_3D, cond_values=y)
        R_x = torch.linalg.cholesky(covar_x)

        # # Find the sorting order that puts all unconditioned variables first in the covariance matrix
        # sorting_ind = torch.argsort(cond_mask.flatten().long(), stable=True)
        # n_cond_var = cond_mask.sum()
        # C_3D = C_3D[sorting_ind[:, None], sorting_ind[None, :]]
        # sigma_xx = C_3D[:-n_cond_var, :-n_cond_var]
        # sigma_xy = C_3D[:-n_cond_var, -n_cond_var:]
        # sigma_yy = C_3D[-n_cond_var:, -n_cond_var:]
        # sigma_yy_inv = sigma_yy.inverse()
        # # mu_x = sigma_xy @ sigma_y_inv @ y  # The unconditional mean is zero
        # mu_x = y @ sigma_yy_inv.T @ sigma_xy.T  # The unconditional mean is zero
        # covar_x = sigma_xx - sigma_xy @ sigma_yy_inv @ sigma_xy.t()
        # R_x = torch.linalg.cholesky(covar_x)

        # Generate samples of the unconditioned variables and concatenate the conditioned variables values
        samples = torch.zeros(n_samples, R_3D.shape[-1], **cons_kwargs)
        samples_x = mu_x + sigma * (R_x @ torch.randn((n_samples, R_x.shape[-1], 1), **cons_kwargs)).squeeze(-1)
        samples_y = y.expand(n_samples, -1)
        samples[:, ~cond_mask_3D] = samples_x
        samples[:, cond_mask_3D] = samples_y

        # Reshape to sets of 3D vectors
        samples = samples.reshape(n_samples, -1, 3)
        return samples


class DiffusionModel(nn.Module):

    def __init__(self, denoiser, noise_sch: NoiseSchedule, t_steps: torch.Tensor = None, t_func=None,
                 Heun_sampler=config.Heun_sampler,
                 SDE_sampler=config.SDE_sampler, SDE_sampler_params=config.SDE_sampler_params,
                 corr_noiser: CorrelatedNoiser = None, aux_losses: list[AuxiliaryLoss] = None,
                 conditioners: list[Conditioner] = None,
                 aux_losses_weight=config.aux_losses_weight, aux_loss_reweight=config.aux_loss_reweight,
                 t_sampling_method=config.t_sampling_method, remove_noise_COM=config.remove_noise_COM,
                 importance_sampling_logSNR=config.importance_sampling_logSNR,
                 sampling_dtype=config.sampling_dtype):
        # Defaults
        if t_func is None:
            t_func = lambda tau: noise_sch.Karras_t(t=tau)
        if t_steps is None:
            t_steps = t_func(noise_sch.t_arr)

        # Checks
        assert t_steps.diff().le(0).all(), 'The time steps must be monotonically decreasing'
        if SDE_sampler and Heun_sampler:
            warnings.warn("Heun sampler is ignored when the SDE sampler is used")
        if t_steps.isnan().any():
            raise ValueError(f"nans found in sampling time_steps\ntime_steps={t_steps}")
        super().__init__()

        # Training
        self.denoiser = denoiser
        self.t_steps = nn.parameter.Buffer(t_steps, persistent=False)
        self.t_func = t_func
        self.remove_noise_COM = nn.parameter.Buffer(torch.as_tensor(remove_noise_COM), persistent=False)
        self.importance_sampling_logSNR = nn.parameter.Buffer(torch.as_tensor(importance_sampling_logSNR),
                                                              persistent=False)
        self.noise_sch = noise_sch

        # Auxiliary losss
        self.aux_losses: list[AuxiliaryLoss] = nn.ModuleList(aux_losses if aux_losses else [])
        self.aux_losses_weight = aux_losses_weight
        self.aux_loss_reweight = aux_loss_reweight

        # Sampling
        self.Heun_sampler = Heun_sampler
        self.SDE_sampler = SDE_sampler
        self.SDE_sampler_params = SDE_sampler_params
        self.t_sampling_method = t_sampling_method
        self.correlated_noiser = corr_noiser
        self.U_tot_mod: TotalEnergy = None
        self.U_tot_stats: StatisticsArray = None
        self.conditioners: Iterable[Conditioner] = nn.ModuleList(conditioners if conditioners else [])
        self.sampling_dtype = torch.float32 if sampling_dtype is None else sampling_dtype

        # Define function placeholders
        self.sig_pos = nn.parameter.Buffer(torch.tensor(1.0), persistent=True)
        self.sigma = lambda t: t
        self.sigma_der = lambda t: t
        self.s = lambda t: t
        self.s_der = lambda t: t
        self.c_skip = lambda t: t
        self.c_out = lambda t: t
        self.c_in = lambda t: t
        self.c_noise = lambda t: t
        self.loss_weight = lambda t: t

        # Define lognormal distribution when using it for sampling t during training
        if self.t_sampling_method.startswith('lognormal'):
            params_match = re.search('lognormal_(.*)_(.*)', self.t_sampling_method)
            if params_match is None:
                t_log_mean, t_log_std = 0, 1
            else:
                t_log_mean, t_log_std = float(params_match.group(1)), float(params_match.group(2))
            self.t_dist = torch.distributions.LogNormal(t_log_mean, t_log_std)
            t_cdf_low, t_cdf_high = self.t_dist.cdf(self.t_steps[-1]), self.t_dist.cdf(self.t_steps[0])
            self.t_cdf_dist = torch.distributions.Uniform(t_cdf_low, t_cdf_high)

    def post_init(self):
        # Assign sigma function to conditioners
        for conditioner in self.conditioners:
            conditioner.sigma_func = self.sigma

    @property
    def device(self):
        return self.t_steps.device

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for attr_name, attr_val in self.__dict__.items():
            if isinstance(attr_val, torch.Tensor):
                setattr(self, attr_name, attr_val.to(self.device))
            elif isinstance(attr_val, list) and attr_val and isinstance(attr_val[0], torch.Tensor):
                setattr(self, attr_name, [T.to(self.device) for T in attr_val])

    def add_noise(self, x: torch.Tensor, t: torch.Tensor):
        epsilon = self.sigma(t).to(x.dtype) * self.sample_noise(output_ex=x)
        x_noisy = x + epsilon
        return x_noisy

    def loss(self, x: torch.Tensor, normalize=True, **denoiser_kwargs):
        batch_size = x.shape[0]
        t_size = torch.Size((batch_size, 1, 1))
        t, t_ind, tau = self.sample_t(t_size)
        x_noisy = self.add_noise(x=x, t=t)
        x_denoised, extra_denoiser_output = self.D(x_noisy, t, **denoiser_kwargs)

        # Position loss
        x_diff = x - x_denoised
        if self.correlated_noiser is not None:
            # Double precision is needed to get accurate matmul in next step
            x_diff = self.correlated_noiser.C_inv_R_mul(x_diff.double())
        loss_weights = self.loss_weight(t).to(x.dtype)
        loss_pos = torch.sum(loss_weights * x_diff.square(), dim=[-2, -1])  # Sum over atoms and coordinates

        # Reweight each batch sample by 1/p(logSNR(t)) where t is the sampled diffusion time
        if self.importance_sampling_logSNR:
            p_logSNR = self.noise_sch.logSNR_pdf(t=tau)
            loss_pos /= p_logSNR

        if normalize:
            n_elements = x.values().numel() if x.is_sparse else x.numel()
            loss_pos = loss_pos / n_elements

        # Auxiliary losses
        loss_dict = dict()
        for i, aux_loss in enumerate(self.aux_losses):
            aux_loss: AuxiliaryLoss
            if not aux_loss.is_active:
                continue
            loss_aux = aux_loss.U(x=x, x_denoised=x_denoised, t=t, t_ind=t_ind, reweight=self.aux_loss_reweight,
                                  sum_batch=False, adj_mat=denoiser_kwargs['nodes_adj'])

            # Reweight each batch sample by 1/p(logSNR(t))
            if self.importance_sampling_logSNR:
                loss_aux /= p_logSNR.flatten()

            # Reweight each sample by the given weight
            if self.aux_losses_weight:
                if isinstance(self.aux_losses_weight[i], float | int):
                    cond_weight = self.aux_losses_weight[i]
                elif self.aux_losses_weight[i] == 'pos_weight':
                    cond_weight = loss_weights.reshape(-1)
                else:
                    raise ValueError(f"Conditional weight {self.aux_losses_weight[i]} is not supported.")
                loss_aux *= cond_weight

            if normalize:
                loss_aux /= batch_size

            loss_dict[aux_loss.name_short] = loss_aux.sum()

        return loss_pos, loss_dict, x_denoised, loss_weights, t, extra_denoiser_output

    def D(self, x, t, **denoiser_kwargs):
        c_skip, c_out, c_in = self.c_skip(t).to(x.dtype), self.c_out(t).to(x.dtype), self.c_in(t).to(x.dtype)
        c_noise = self.c_noise(t).to(x.dtype)
        x_in = c_in * x
        y = self.denoiser(x_in, c_noise, **denoiser_kwargs)
        if isinstance(y, torch.Tensor):
            extra_denoiser_output = []
        else:
            extra_denoiser_output = list(y)
            y = extra_denoiser_output.pop(0)

        x_out = c_skip * x + c_out * y

        # Add centroid skip connection to ensure translation equivariance of the denoiser
        x_centroid = x.mean(dim=-2, keepdim=True)
        x_out += (1 - c_skip) * x_centroid

        return x_out, extra_denoiser_output

    def sample_noise(self, size: torch.Size = None, output_ex: torch.Tensor = None):
        if output_ex is not None:
            dtype = output_ex.dtype
        else:
            dtype = torch.get_default_dtype()

        output_is_sparse = output_ex is not None and output_ex.is_sparse
        if output_is_sparse:
            output_ex = output_ex.coalesce()
            size = torch.Size((output_ex._nnz(), *output_ex.shape[output_ex.sparse_dim():]))

        if size is None:
            size = output_ex.size()

        if self.correlated_noiser:
            epsilon = self.correlated_noiser.generate_samples(size=size, device=self.device)
        else:
            epsilon = torch.randn(size, device=self.device, dtype=dtype)

        if self.remove_noise_COM:
            remove_centroid(epsilon, inplace=True)

        epsilon = epsilon.to(dtype)
        if output_is_sparse:
            ind = output_ex.indices()
            epsilon = torch.sparse_coo_tensor(ind, epsilon, size=output_ex.shape).coalesce()

        return epsilon

    def sample(self, positions_init: torch.Tensor, T=None, verbose=False, conditioners: list[Conditioner] = None,
               save_trajectory=False, save_cond_U=False, **denoiser_kwargs):
        if positions_init.isnan().any():
            raise ValueError(f"Nans found in initial positions.")

        # Gather all conditioner modules
        conditioners = list() if conditioners is None else copy.copy(conditioners)
        conditioners += self.conditioners

        # Preprocessing for position conditioners
        pos_restrainers = [c for c in conditioners if isinstance(c, PositionConditioner)]
        assert len(pos_restrainers) < 2, f"Up to 1 position restrainer can be used. {len(pos_restrainers)} were given."
        pos_rest_cond = find_first_element_of_class(conditioners, PositionConditioner)
        coarse_center_dist_cond = find_first_element_of_class(conditioners, CoarseCenterDistConditioner)
        coarse_center_dist_cond: CoarseCenterDistConditioner

        # Determine the times that will be sampled
        N = len(self.t_steps)
        if T is None:
            T = N - 1
        t_ind = torch.linspace(0, N - 1, T + 1, dtype=torch.int).to(self.device)
        t_arr = self.t_steps.double()[t_ind]
        s_t, s_t_der = self.s(t_arr), self.s_der(t_arr)
        sigma_t, sigma_t_der = self.sigma(t_arr), self.sigma_der(t_arr)

        # Checks on the time array
        dt = torch.diff(t_arr)
        dt_sign = torch.sign(dt[0])
        is_t_arr_monotonic = dt.sign().unique().nelement() == 1
        if not is_t_arr_monotonic:
            raise ValueError(f"time step array is not monotonic.\n t_arr={t_arr}")
        is_sigma_decreasing = sigma_t.diff().lt(0).all()
        if not is_sigma_decreasing:
            raise ValueError(f"Sigma value must be monotonically decreasing.")

        # Define sampler timeseries
        # eq. 34 in Karras et al. "Elucidating the Design Space of Diffusion-Based Generative Models"
        # g(t) = s(t) sqrt(2 sigma(t) sigma'(t))
        g_t = s_t * torch.sqrt(2 * sigma_t * sigma_t_der)
        g_t_squared = g_t.square()
        if self.SDE_sampler:
            psi, lambda0 = self.SDE_sampler_params['psi'], self.SDE_sampler_params['lambda0']
            sigma2_t = sigma_t.square()
            sigma_data = self.sig_pos.detach().squeeze()
            sigma2_data = sigma_data.square()
            if self.correlated_noiser is not None:
                sigma2_prior = self.correlated_noiser.dim_var.to(self.device)
            else:
                sigma2_prior = torch.tensor(1.0).to(self.device)

            # Define lambda(t) following Chroma (https://www.nature.com/articles/s41586-023-06728-8)
            # This works for both variance-exploding and variance-preserving scheme since alpha(t) = s(t) and
            # sigma(t) = s(t) sigma_K(t) where sigma_K(t) is the sigma function defined in Karras' EDM model.
            lambda_t = (sigma2_data + sigma2_t * sigma2_prior) / (sigma2_data / lambda0 + sigma2_t * sigma2_prior)

            # Define the psi schedule, which controls the equilibration rate (amount of extra noise)
            psi = torch.as_tensor(self.SDE_sampler_params['psi']).to(self.device) * torch.ones_like(sigma_t)

            # Multiply the Wiener's process sigma by sign(dt)/sqrt(dt) since dxdt is multiplied by dt
            Wiener_sigma = dt_sign * g_t[:-1] * ((1 + psi[:-1]) / dt.abs()).sqrt()
            a = lambda_t + lambda0 * psi / 2
        else:
            a = torch.tensor(1 / 2).to(self.device)

        # Define conditioners' force weight. Multiply by dt_sign since the force is already signed in the conditioner F() method
        cond_F_weight = a * dt_sign * g_t_squared

        # In Karras, the gradient is calculated with respect to the scaled variable (x). However, since the conditioners
        # receive x/s,they will calculate the gradient with respect to x/s. Since grad_(x/s) = s*grad(x),
        # the conditioner's forces must be rescaled by 1/s to effectively calculate the gradient with respect to x.
        cond_F_weight /= s_t

        # Define the time index iterator based on the number of time steps requested
        t_ind_iter = range(T)
        if verbose:
            tqdm_desc = f'Generating {positions_init.size(0)} samples with {self.__class__.__name__} using {T} time steps.'
            t_ind_iter = tqdm.tqdm(t_ind_iter, desc=tqdm_desc)

        # Sample initial conformations
        x = positions_init + s_t[0] * sigma_t[0] * self.sample_noise(output_ex=positions_init)

        # Convert all sampling timeseries to the required dtype
        t_arr = t_arr.to(self.sampling_dtype)
        s_t, s_t_der = s_t.to(self.sampling_dtype), s_t_der.to(self.sampling_dtype)
        x = x.to(self.sampling_dtype)
        cond_F_weight = cond_F_weight.to(self.sampling_dtype)
        if sigma_t.dtype != self.sampling_dtype:
            sigma_func = lambda t: self.sigma(t).to(self.sampling_dtype)
        else:
            sigma_func = self.sigma

        if self.SDE_sampler:
            Wiener_sigma = Wiener_sigma.to(self.sampling_dtype)

        # Combine all conditioners with the diffusion model to calculate the sum of all conditioners energy in one pass
        def denoiser(*args, **kwargs):
            return self.D(*args, **kwargs, **denoiser_kwargs)

        diff_cond = DiffusionConditioner(denoiser=denoiser, corr_noiser=self.correlated_noiser, sigma_func=sigma_func)
        comb_cond = CombinedConditioner(sigma_func=sigma_func, conditioners=conditioners + [diff_cond])
        cond_kwargs = dict(adj_mat=denoiser_kwargs['nodes_adj'])

        # Move conditioners to device
        comb_cond.to(self.device)

        # Define dictionary to save the conditioners' energies during denoising (if requested)
        if config.dynamic_denoising and save_cond_U:
            raise NotImplementedError  # Saved energies will vary in shape depending which sample is denoising

        if save_cond_U:
            conditioners_U = torch.zeros((T, x.shape[0], len(comb_cond.conditioners)), device=self.device)
            conditioners_name = [c.name_short for c in comb_cond.conditioners]
            conditioners_info = {'energy': conditioners_U, 'type': conditioners_name}

        def dxdt_func(x, i):
            # Calculate the conditioners' force on the unscaled variables
            cond_F, cond_U = comb_cond.F(x=x / s_t[i], t=t_arr[i], t_ind=t_ind[i], output_U=True, **cond_kwargs)

            # Scale the conditioner's force by C = R @ R^T when using correlated noise
            if self.correlated_noiser:
                cond_F = self.correlated_noiser.C_mul(cond_F)
            dxdt = cond_F_weight[i] * cond_F

            # Add scale-dependent term
            if not s_t_der[i].eq(0.0).all():
                dxdt += s_t_der[i] / s_t[i] * x.detach()

            # Add noise if using SDE sampler
            if self.SDE_sampler:
                dxdt += Wiener_sigma[i] * self.sample_noise(output_ex=dxdt)

            return dxdt, cond_U

        # Compile the derivative function when sampling many samples
        if x.shape[0] > 200 and config.compile_model:
            dxdt_func = torch.compile(dxdt_func, dynamic=True)

        if config.dynamic_denoising:
            denoiser_kwargs_ori = copy.deepcopy(denoiser_kwargs)
            U_dist = torch.distributions.normal.Normal(self.U_tot_stats.mean.unsqueeze(1),
                                                       self.U_tot_stats.std.unsqueeze(1))
            samples_is_denoising = torch.ones((x.shape[0],), device=self.device, dtype=torch.bool)
            samples_t_ind = torch.zeros((x.shape[0], 1, 1), dtype=torch.long)
            t_ind_window = [int(T * 0.1), 0]
            no_early_stop = True

        # Prepare array that will save denoising trajectories
        if save_trajectory:
            x_traj = x.new_full((T + 1, *x.shape), torch.nan)  # shape = (T+1,B,N,3)
            x_traj[0] = x.clone()

        # Integration loop
        x.requires_grad = True
        for i in t_ind_iter:
            if config.dynamic_denoising:
                # Evaluate the logits of each denoising time based on the energy values of the samples
                x_temp = x[samples_is_denoising]
                x_temp_U = self.U_tot_mod(comb_cond.transform_x(x_temp.detach()))
                t_logits = U_dist.log_prob(x_temp_U).sum(dim=-1).T  # shape=(B,T+1)

                # Restrict the possible times that can be sampled
                # Option 1: Allow jumps to t value that must be within a given window around i
                is_t_prohibited = t_ind.lt(i - t_ind_window[0]) | t_ind.gt(i + t_ind_window[1])

                # # Option 2: Allow jumps in t that are maximum 1% of the total number of time steps in the range.
                # is_t_prohibited = (samples_t_ind.reshape(-1, 1) - t_ind).abs().gt(T * 0.01)

                t_logits[:, is_t_prohibited] = -torch.inf
                # Prevent sampling from finishing early.
                if no_early_stop:
                    t_logits[:, -1] = -torch.inf

                # Sample the denoising times for each sample using the logits evaluated from U_tot
                t_mult = torch.distributions.multinomial.Multinomial(logits=t_logits)
                samples_t_ind = t_mult.sample().nonzero()[:, 1].reshape(-1, 1, 1)
                samples_sub_is_denoising = ~samples_t_ind.eq(t_ind[-1]).flatten()
                if config.debug and x.shape[0] < 10:
                    print(f'Sampled t index at i={i}:{samples_t_ind.flatten()}')

                # Update the denoising boolean for each sample that was denoising
                samples_is_denoising[samples_is_denoising.clone()] = samples_sub_is_denoising

                # End early if there are no more samples that need to be denoised
                if not samples_is_denoising.any():
                    break

                # Remove samples that have reached the last denoising time
                x_temp = x_temp[samples_sub_is_denoising]
                samples_t_ind = samples_t_ind[samples_sub_is_denoising]

                if not samples_sub_is_denoising.all():
                    for k, v in denoiser_kwargs.items():
                        denoiser_kwargs[k] = denoiser_kwargs_ori[k][samples_is_denoising, ...]

                    if pos_rest_cond is not None:
                        pos_rest_cond.apply_batch_mask(samples_is_denoising)
                    if coarse_center_dist_cond is not None:
                        coarse_center_dist_cond.apply_batch_mask(samples_is_denoising)
            else:
                x_temp = x
                samples_t_ind = i

            # Rotate the structures to minimize the RMSD of the coarse grain centroid with the target positions
            if coarse_center_dist_cond is not None and config.sampling_align_coarse_grain:
                coarse_pos_target = coarse_center_dist_cond.coarse_pos.double()
                grain_centroid_pos = torch.full_like(coarse_pos_target, torch.nan)
                grain_centroid_pos.index_reduce_(source=x_temp.detach(), index=coarse_center_dist_cond.coarse_grain_ind,
                                                 dim=-2, reduce='mean', include_self=False)
                if grain_centroid_pos.isnan().any():
                    raise ValueError(f"Nans found in calculated coarse grain centroid position at iter={i}.")
                rot, trans = rmsd_align_transform(positions=grain_centroid_pos, positions_ref=coarse_pos_target)
                with torch.no_grad():
                    x_temp = x_temp @ rot + trans
                    x_temp = x_temp.to(self.sampling_dtype)

            # Denoising step
            dxdt, cond_U = dxdt_func(x_temp, samples_t_ind)
            dx = dxdt * dt[samples_t_ind]
            with torch.no_grad():
                x_temp += dx

            # Perform second step if using Heun Sampling
            if not self.SDE_sampler and self.Heun_sampler:
                samples_mask = samples_t_ind < T - 1  # Remove samples that have reached step T-1
                dxdt_next, _ = dxdt_func(x_temp[samples_mask], samples_t_ind[samples_mask] + 1)
                dx_next = dxdt_next * dt[samples_t_ind[samples_mask]]
                with torch.no_grad():
                    x_temp[samples_mask] += (dx_next - dx[samples_mask]) / 2

            # Print conditioner warnings if any anomalies occured during sampling
            comb_cond.print_warnings()

            # Save the conditioner energies
            if save_cond_U:
                conditioners_U[t_ind[i]] = cond_U

            if config.dynamic_denoising:
                with torch.no_grad():
                    x[samples_is_denoising] = x_temp
            else:
                x = x_temp

            # Check for GPU memory leaks
            if config.debug and self.device == 'cuda' and i % 10 == 0:
                allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
                print(f"GPU MEM (i={i}) Active Tensors: {allocated_gb:.3f} GB | Reserved Cache: {reserved_gb:.3f} GB")

            # Check for nans
            x_is_nan = x.isnan()
            if x_is_nan.any():
                warnings.warn(f"{x_is_nan.sum()} nans found in sampled positions of shape {x.shape} at i={i}.")

            # Save state of denoising trajectory
            if save_trajectory:
                if config.dynamic_denoising:
                    x_traj[i + 1, samples_is_denoising] = x[samples_is_denoising].detach().clone()
                else:
                    x_traj[i + 1] = x.detach().clone()

        # Detach compute graph
        x = x.detach()

        # Apply linear constraints (if any) on the final unconstrained samples
        x = comb_cond.transform_x(x)

        # Perform conditioner transforms on saved trajectory frames and align them to the final structure
        if save_trajectory:
            x_traj = comb_cond.transform_x(x_traj)
            x_traj = rmsd_align(positions=x_traj, positions_ref=x_traj[-1:], reflect=False, remove_mean=False)
            return x, x_traj

        if save_cond_U:
            return x, conditioners_info

        return x

    def sample_t(self, size: int | tuple[int] | torch.Size):
        if isinstance(size, int):
            size = (size,)
        if not isinstance(size, torch.Size):
            size = torch.Size(size)

        if self.t_sampling_method.startswith('lognormal'):
            t = self.t_dist.icdf(self.t_cdf_dist.sample(size)).to(self.device)
            t_ind = torch.argmin((t.reshape(-1, 1) - self.t_steps).abs(), dim=-1).reshape(size)
            tau = None
        else:
            if self.t_sampling_method == 'random':
                tau = torch.rand(size, device=self.device)
            elif self.t_sampling_method == 'uniform':
                tau = torch.arange(0, 1, 1 / size.numel(), device=self.device) + torch.rand(1, device=self.device)
                tau = torch.remainder(tau, 1.0).reshape(size)
            else:
                raise ValueError(f"Method={self.t_sampling_method} is not supported.")
            t = self.t_func(tau)
            t_nan_ind = torch.nonzero(t.isnan())
            if t_nan_ind.nelement() > 0:
                raise ValueError(f"Nans were sampled for the timesteps.\ntau={tau[t_nan_ind]}")
            t_ind = torch.round(tau * (self.t_steps.shape[0] - 1)).long()

        return t, t_ind, tau

    def estimate_aux_loss_stats(self, dataset: ProcessedDataset, regen=False):
        # Check if any auxiliary losses are missing stats
        aux_losses = [c for c in self.aux_losses if c.need_stats or regen]
        if not aux_losses:
            return

        # Determine how many more iterations are needed based on the saved stats
        conditioner_n_min = math.inf
        for aux_loss in aux_losses:
            aux_loss.init_var_stats()
            conditioner_n_min = min(conditioner_n_min, aux_loss.n_stats)
        n_total = config.aux_loss_stats_n_min
        n_remaining = n_total - conditioner_n_min
        if n_remaining <= 0:
            warnings.warn(
                f"More statistics were accumulated than necessary: {conditioner_n_min}>{config.aux_loss_stats_n_min}")
            return

        batch_size = min(len(dataset), config.batch_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=config.N_workers, pin_memory=config.pin_memory,
                                collate_fn=lambda x: x, drop_last=True)
        tensors_iter = repeat_iterator(dataloader)
        batches_t_ind = torch.split(torch.arange(len(self.t_steps), device=self.device), batch_size)
        tasks_iter = tqdm.tqdm(range(n_remaining), 'Estimating aux_loss stats')

        # Swap the denoiser to a zero denoiser
        def zero_denoiser(x, t_embedding, **kwargs):
            return torch.zeros_like(x)

        denoiser = self.denoiser
        self.denoiser = zero_denoiser

        for _ in tasks_iter:
            for t_ind in batches_t_ind:
                # Fetch samples
                tensors_dict = next(tensors_iter)
                for k, T in tensors_dict.items():
                    if isinstance(T, torch.Tensor) and T.ndim > 2 and T.shape[0] != t_ind.shape[0]:
                        tensors_dict[k] = T[:t_ind.shape[0]]
                x = tensors_dict['nodes_pos']
                nodes_adj = tensors_dict['nodes_adj']

                # Add noise
                t_temp = self.t_steps[t_ind].reshape(-1, *(x.ndim - 1) * [1])
                x_noisy = self.add_noise(x=x, t=t_temp)
                x_denoised_zero, _ = self.D(x=x_noisy, t=t_temp, **tensors_dict)

                for aux_loss in aux_losses:
                    # Accumulate var stats to estimate covariance matrix
                    if aux_loss.estimate_var_cov:
                        aux_loss.accumulate_var_stats(x=x, x_noisy=x_noisy, t_ind=t_ind, adj_mat=nodes_adj)

                    # Accumulate var stats to estimate U weights
                    if aux_loss.estimate_U_weights:
                        aux_loss.accumulate_U_stats(x=x, x_denoised=x_denoised_zero, t_ind=t_ind)

        # Restore the denoiser
        self.denoiser = denoiser

        # Check that parameters were assigned
        for aux_loss in aux_losses:
            if aux_loss.estimate_var_cov and not aux_loss.R.any():
                raise ValueError(f"R was not defined for {self.__class__.__name__}")
            if aux_loss.U_weights is not None and not aux_loss.U_weights.any():
                raise ValueError(f"U_weights was not defined for {self.__class__.__name__}")

    def estimate_U_tot_stats(self, dataset: ProcessedDataset, regen=False):
        """
        Estimate statistics of the total energy used to sample denoising times in dynamic denoising
        Args:
            dataset: dataset used to evaluate the energy statistics
            regen: regenerate statistics and overwrite previously saved in module

        Returns:
            None
        """
        # Parameters
        n_samples_tot = int(1e2 if config.debug else 1e4)
        n_t_steps = len(self.t_steps)

        # Initialize module and stats
        # pos_scale is needed since ProcessedDataset unscales the positions
        total_en_mod = TotalEnergy(top_file=dataset.dataset.top_filepath,
                                   element_class=dataset.dataset.element_class,
                                   pos_scale=dataset.pos_scale)
        total_en_mod.to(self.device)
        self.U_tot_mod = total_en_mod
        if self.U_tot_stats is None or regen:
            self.U_tot_stats = StatisticsArray(shape=(n_t_steps,), d=total_en_mod.n_terms, dtype=torch.double)
            self.U_tot_stats.to(self.device)
        U_tot_stats = self.U_tot_stats

        # Determine how many more samples are needed to evaluate the stats.
        n_remaining = n_samples_tot - int(U_tot_stats.n.min())
        if n_remaining <= 0:
            return

        batch_size = int(np.minimum(len(dataset), config.batch_size))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=config.N_workers, pin_memory=config.pin_memory,
                                collate_fn=lambda x: x, drop_last=True)
        tensors_iter = repeat_iterator(dataloader)
        batches_t_ind = torch.split(torch.arange(n_t_steps), batch_size)
        samples_ind_iter = tqdm.tqdm(range(n_remaining), f'Estimating U_tot stats of {dataset.dataset.name!r}.')
        t_steps_ind = torch.arange(n_t_steps, device=self.device)

        # Fetch positions, noise them and calculate their energies for each denoising time
        U_tot = torch.zeros((n_t_steps, total_en_mod.n_terms), device=self.device, dtype=torch.double)
        for i in samples_ind_iter:
            U_tot.zero_()
            for t_ind in batches_t_ind:
                tensors_dict = next(tensors_iter)
                x = tensors_dict['nodes_pos'].to(self.device)
                if x.shape[0] != t_ind.shape[0]:
                    x = x[:t_ind.shape[0]]
                t_steps_temp = self.t_steps[t_ind].reshape(-1, *(x.ndim - 1) * [1])
                x_noisy = self.add_noise(x=x, t=t_steps_temp)
                U_tot[t_ind] = total_en_mod(atom_positions=x_noisy)

            U_tot_stats += StatisticsArray(shape=U_tot_stats.shape, ind=t_steps_ind, values=U_tot)


class EDM(DiffusionModel):
    def __init__(self, noise_sch: NoiseSchedule, denoiser, sigma_data: dict[str, torch.Tensor],
                 sig_pos_in: torch.Tensor = None, sig_pos_out: torch.Tensor = None, **kwargs):
        super().__init__(denoiser, noise_sch=noise_sch, **kwargs)
        self.sigma = lambda t: t
        self.sigma_der = lambda t: torch.ones_like(t)
        self.s = lambda t: torch.ones_like(t)
        self.s_der = lambda t: torch.zeros_like(t)

        if self.correlated_noiser:
            self.sig_prior = self.correlated_noiser.dim_var.double().sqrt().item()
            self.prec_prior = self.correlated_noiser.dim_prec.double().sqrt().item()
        else:
            self.sig_prior = 1.0
            self.prec_prior = 1.0

        # Set sigma_pos_in to double precision to have precise c functions
        if sig_pos_in is None:
            sig_pos_in = sigma_data['pos']
        sig_pos_in = torch.as_tensor(sig_pos_in).double().reshape(1, 1, 1)

        self.sig_pos_in = nn.parameter.Buffer(sig_pos_in)
        self.c_in = lambda t: torch.pow(self.sig_pos_in.square() + (self.sig_prior * t).square(), -0.5)

        if sig_pos_out is None:
            sig_pos_out = sigma_data[f'pos']
        sig_pos_out = torch.as_tensor(sig_pos_out).double().reshape(1, 1, 1)

        self.sig_pos.data = sig_pos_out
        self.c_skip = lambda t: (self.sig_pos * self.prec_prior).square() / (
                (self.sig_pos * self.prec_prior).square() + t.square())
        self.c_out = lambda t: t * self.sig_pos * self.prec_prior / torch.sqrt(
            (self.sig_pos * self.prec_prior).square() + t.square())
        self.loss_weight = lambda t: 1 / self.c_out(t).square()
        self.c_noise = lambda t: torch.log(t) / 4

        self.post_init()


class EDM2(EDM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.c_in = lambda t: torch.ones_like(t)
        self.c_noise = lambda t: t


class VP(DiffusionModel):
    """
        Variance-preserving model using the diffusion model formalism of the EDM model.
        Reference: Karras et al. 2022, 'Elucidating the Design Space of Diffusion-Based Generative Models'
    """

    def __init__(self, noise_sch: NoiseSchedule, denoiser, sigma_data: dict[str, torch.Tensor], **kwargs):
        # Define the function that produces the t_steps array out of the tau samples.
        t_func = lambda t: 1 - t  # In Karras' convention, the smallest time is noisefull
        super().__init__(denoiser, noise_sch=noise_sch, t_func=t_func, **kwargs)

        # Define s(t) and sigma(t) from alpha(t) and sigma(t) in noise_sch using eq. 11
        # The noising kernel defined in noise_sch is: x(t) = alpha(t)*x_0 + sigma(t)*eps while Karras' kernel is:
        # x(t) = s(t)*x_0 + s(t)*sigma(t)*eps. Match the kernel to find the correspondence between the functions
        self.s = lambda t: noise_sch.alpha(t)
        self.sigma = lambda t: torch.exp(-0.5 * noise_sch.SNR(t, log=True))  # noise_sch.sigma(t)/noise_sch.alpha(t)
        self.s_der = lambda t: torch.autograd.functional.vjp(self.s, t, v=torch.ones_like(t))[1]
        self.sigma_der = lambda t: torch.autograd.functional.vjp(self.sigma, t, v=torch.ones_like(t))[1]

        sigma_pos = sigma_data[f'pos']
        sigma_pos = sigma_pos.double().reshape(1, 1, sigma_pos.nelement())
        self.sig_pos.data = sigma_pos

        self.c_skip = lambda t: torch.ones_like(t)
        self.c_out = lambda t: -self.sigma(t)
        if self.correlated_noiser:
            self.sigma_prior = self.correlated_noiser.dim_var.double().sqrt()
            self.c_in = lambda t: torch.pow(self.sig_pos.square() + (self.sigma_prior * self.sigma(t)).square(), -0.5)
        else:
            self.c_in = lambda t: torch.pow(self.sig_pos.square() + self.sigma(t).square(), -0.5)
        self.c_noise = lambda t: t
        self.loss_weight = lambda t: torch.pow(self.sigma(t), -2)

        self.post_init()


class VP2(DiffusionModel):
    """
        Variant of variance-preserving model using the preconditioning functions of the EDM model
        Reference: Karras et al. 2022, 'Elucidating the Design Space of Diffusion-Based Generative Models'
    """

    def __init__(self, noise_sch: NoiseSchedule, denoiser, **kwargs):
        super().__init__(denoiser, noise_sch=noise_sch, **kwargs)
        self.c_skip = lambda t: self.sigma_data_2 / (self.sigma_data_2 + self.sigma(t).square())
        self.c_out = lambda t: self.sigma(t) * self.sigma_data / torch.sqrt(self.sigma_data_2 + self.sigma(t).square())
        if self.correlated_noiser:
            self.sigma_prior = self.correlated_noiser.dim_var.double().sqrt()
            self.c_in = lambda t: torch.pow(self.sigma_data_2 + (self.sigma_prior * self.sigma(t)).square(), -0.5)
        else:
            self.c_in = lambda t: torch.pow(self.sigma_data_2 + self.sigma(t).square(), -0.5)
        self.c_noise = lambda t: t
        self.loss_weight = lambda t: 1 / self.sigma_data_2 + 1 / self.sigma(t).square()

        self.post_init()


class VE(DiffusionModel):
    def __init__(self, noise_sch: NoiseSchedule, denoiser, sigma_data: dict[str, torch.Tensor],
                 sig_pos_in: torch.Tensor = None, **kwargs):
        super().__init__(denoiser, noise_sch=noise_sch, **kwargs)

        self.sigma = lambda t: t
        self.sigma_der = lambda t: torch.ones_like(t)
        self.s = lambda t: torch.ones_like(t)
        self.s_der = lambda t: torch.zeros_like(t)

        if self.correlated_noiser:
            self.sig_prior = self.correlated_noiser.dim_var.double().sqrt()
            self.prec_prior = self.correlated_noiser.dim_prec.double().sqrt()
        else:
            self.sig_prior = torch.tensor(1.0)
            self.prec_prior = torch.tensor(1.0)

        # Set to double to have good precision in the c functions below
        if sig_pos_in is None:
            sig_pos_in = sigma_data['pos']
        sig_pos_in = torch.as_tensor(sig_pos_in).double().reshape(1, 1, 1)
        self.sig_pos_in = nn.parameter.Buffer(sig_pos_in)

        sigma_pos = sigma_data[f'pos'].double().reshape(1, 1, 1)
        self.sig_pos.data = sigma_pos

        self.c_in = lambda t: torch.pow(self.sig_pos_in.square() + (self.sig_prior * t).square(), -0.5)
        self.c_skip = lambda t: torch.ones_like(t)
        self.c_out = lambda t: self.sigma(t)
        self.loss_weight = lambda t: 1 / self.c_out(t).square()
        self.c_noise = lambda t: torch.log(t) / 4

        self.post_init()


class VE2(VE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_in = lambda t: 1 / self.sig_pos_in
        self.c_out = lambda t: torch.ones_like(t)
        self.loss_weight = lambda t: 1 / self.sigma(t).square()

        self.post_init()


def plot_c_funcs(diff_models: list[DiffusionModel] | DiffusionModel):
    if not isinstance(diff_models, list):
        diff_models = [diff_models]

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')

    plt.figure()
    for model in diff_models:
        plt.plot(model.t_steps, model.c_in(model.t_steps).flatten().detach(), label=model.__class__.__name__)
    plt.legend()
    plt.xscale('log')
    plt.xlabel('t')
    plt.ylabel('c_in')

    plt.figure()
    for model in diff_models:
        plt.plot(model.t_steps, model.c_out(model.t_steps).flatten().detach(), label=model.__class__.__name__)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('t')
    plt.ylabel('c_out')

    plt.figure()
    for model in diff_models:
        plt.plot(model.t_steps, model.c_skip(model.t_steps).flatten().detach(), label=model.__class__.__name__)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('t')
    plt.ylabel('c_skip')

    plt.figure()
    for model in diff_models:
        plt.plot(model.t_steps, model.c_noise(model.t_steps).flatten().detach(), label=model.__class__.__name__)
    plt.legend()
    plt.xscale('log')
    plt.xlabel('t')
    plt.ylabel('c_noise')

    plt.figure()
    for model in diff_models:
        plt.plot(model.t_steps, model.loss_weight(model.t_steps).flatten().detach(), label=model.__class__.__name__)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('t')
    plt.ylabel('loss_weight')

    plt.show(block=False)


if __name__ == '__main__':
    # Testing correlated diffuser
    import matplotlib.pyplot as plt
    from data.datasets import load_dataset
    from chemistry.grains import coarse_grains_sets

    dataset = load_dataset('nup98_12_1')
    coarse_grains = coarse_grains_sets.get_adj_grains(method='CA_SI')

    # # Test various types of chain scaling of CorrelatedNoiser
    # scaling_types = ['disordered']
    # scaling_types = ['disordered2']
    # # scaling_types = ['globular']
    # CorrelatedNoiser.test_Rg_scaling(N_range=np.arange(0, 200, 10) + 1, scaling_types=scaling_types)
    # # CorrelatedNoiser.test_Rg_scaling(N_range=np.arange(0, 10, 1) + 1, scaling_types=scaling_types)

    # corr_noiser = CorrelatedNoiser(topology=dataset.top_filepath, element_class=c_alpha_class_name,
    #                                chain_scaling=None, centroid_std=1.0)
    corr_noiser = CorrelatedNoiser(topology=dataset.top_filepath, element_class=c_alpha_sidechain_class_name,
                                   chain_scaling=None, centroid_std=1e-2, sidechain_coarse_grains=coarse_grains,
                                   pos_scale=0.1)
    # corr_noiser = CorrelatedNoiser(topology=dataset.top_filepath, element_class=heavy_atoms_class_name,
    #                                chain_scaling=None, centroid_std=1.0)
    # corr_noiser = CorrelatedNoiser(topology=dataset.top_filepath, element_class=all_atoms_class_name,
    #                                chain_scaling=None, centroid_std=1.0)
    # corr_noiser.plot_sample()

    # Plot a sample of the correlated noise
    # corr_noiser = CorrelatedNoiser(topology=dataset.top_filepath, chain_scaling='disordered2', centroid_std=1.0)
    # corr_noiser = CorrelatedNoiser(topology=dataset.top_filepath, chain_scaling=None, centroid_std=1.0)
    # corr_noiser = CorrelatedNoiser(topology=dataset.top_filepath, element_class=c_alpha_class_name,
    #                                centroid_std=1.0)
    # corr_noiser_no_scaling = CorrelatedNoiser(topology=dataset.top_filepath, chain_scaling=None,
    #                                           centroid_std=1e-4)
    # corr_noiser_fixed_Rg = CorrelatedNoiser(topology=dataset.top_filepath, chain_scaling=dataset_CA_Rg,
    #                                         centroid_std=1.0)

    # corr_noiser = CorrelatedDiffuser(top_file=dataset.top_filepath, chain_scaling_type='globular')
    #
    # corr_noiser_no_scaling.plot_sample()

    # # Generated conditional samples
    # N = 10
    # # cond_mask = torch.rand(corr_noiser.R.shape[0]).unsqueeze(-1).expand(-1, 3) < 0.2
    # CA_mask = torch.zeros((corr_noiser.n_atoms, 3), dtype=torch.bool)
    # CA_mask[corr_noiser.CA_atoms_ind, :] = 1
    # # y_cond_values = corr_noiser.generate_samples(N)[:, CA_mask].reshape(N, -1, 3)
    # y_cond_values = torch.zeros(corr_noiser.n_residues, 3)
    # y_cond_values[:, 0] = torch.arange(corr_noiser.n_residues)
    # y_cond_samples = corr_noiser.generate_cond_samples(y=y_cond_values, cond_mask=CA_mask)
    # # corr_noiser.plot_sample(y_cond_samples[0])
    # pass

    # # Generate samples and check stats
    # corr_noiser.change_scale(pos_scale=0.1)
    # # corr_noiser.change_scale(pos_scale=1.0)
    # y = corr_noiser.generate_samples(100000)
    # Rg_avg = torch.var(y, dim=1, correction=0).sum(-1).mean().sqrt()
    # Rg_exp = corr_noiser.Rg
    # y_dim_var = torch.var(y, dim=[0, 1])
    # y_center_std = torch.mean(y, dim=-2).std(dim=0)
    #
    # y = corr_noiser_no_scaling.generate_samples(100000)
    # Rg_noscaling_avg = torch.var(y, dim=1, correction=0).sum(-1).mean().sqrt()
    # Rg_noscaling_exp = corr_noiser_no_scaling.Rg
    # y_center_std = torch.mean(y, dim=-2).std(dim=0)
    # pass
