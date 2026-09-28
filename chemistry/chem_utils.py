import torch

import config
from data.data_classes import Biomolecule, c_alpha_class_name, c_alpha_sidechain_class_name, heavy_atoms_class_name, \
    all_atoms_class_name
from chemistry.energy import BondEnergy

# Definition of Ramachandran phi/psi state boundaries for each named dihedral state, for both L- and D-stereoisomers
# See https://doi.org/10.1093/protein/gzu037 for the 'Box' method.
DIHEDRAL_STATE_REGIONS = {
    'Simple': {
        'L': {
            'state_names': ['Q_o', 'Q_αL', 'Q_β', 'Q_αR'],
            'state_idx': [0, 1, 2, 3],
            'phi': [[0, 180], [0, 180], [-180, 0], [-180, 0]],
            'psi': [[-180, 0], [0, 180], [0, 180], [-180, 0]],
        },
        'D': {
            'state_names': ['Q_Dβ', 'Q_DαR', 'Q_Do', 'Q_DαL'],
            'state_idx': [2, 3, 0, 1],
            'phi': [[0, 180], [0, 180], [-180, 0], [-180, 0]],
            'psi': [[-180, 0], [0, 180], [0, 180], [-180, 0]],
        },
    },
    'Simple-improved': {
        'L': {
            'state_names': ['Upper-Beta', 'R-Alpha', 'Lower-Beta', 'Upper-4', 'L-Alpha', 'Lower-4'],
            'state_idx': [0, 1, 0, 2, 3, 2],
            'phi': [[-180, 0], [-180, 0], [-180, 0], [0, 180], [0, 180], [0, 180]],
            'psi': [[50, 180], [-130, 50], [-180, -130], [100, 180], [-80, 100], [-180, -80]],
        },
        'D': {
            'state_names': ['Upper-Beta', 'R-Alpha', 'Lower-Beta', 'Upper-4', 'L-Alpha', 'Lower-4'],
            'state_idx': [0, 1, 0, 2, 3, 2],
            'phi': [[-180, 0], [-180, 0], [-180, 0], [0, 180], [0, 180], [0, 180]],
            'psi': [[50, 180], [-130, 50], [-180, -130], [100, 180], [-80, 100], [-180, -80]],
        },
    },
    'Box': {
        'L': {
            'state_names': ['o', 'αR', 'near-αR', 'αL', 'β', 'β', 'PIIL', 'PIR'],
            'state_idx': [0, 1, 2, 3, 4, 4, 5, 6],
            'phi': [[-100, -30], [-175, -100], [5, 75], [-180, -50], [-180, -50], [-110, -50], [-180, -115]],
            'psi': [[-80, -5], [-55, -5], [25, 120], [-180, -170], [80, 180], [120, 180], [50, 100]],
        },
        'D': {
            'state_names': ['Do', 'DαR', 'near-DαR', 'DαL', 'Dβ', 'Dβ', 'DPIIL', 'DPIR'],
            'state_idx': [0, 1, 2, 3, 4, 4, 5, 6],
            'phi': [[30, 100], [100, 175], [-75, -5], [50, 180], [50, 180], [50, 110], [115, 180]],
            'psi': [[5, 80], [5, 55], [-120, -25], [-180, -80], [170, 180], [-180, -120], [-100, -50]],
        },
    },
}


def define_node_triplets(mol: Biomolecule, bond_energy_module: BondEnergy = None):
    """
    Defines the set of triplet node indices for a given molecule with a given element_class
    Args:
        mol: Biomolecule for which the set of node triplets are defined
        bond_energy_module: bond energy module used to define the bond matrix for the heavy-atoms and all-atoms cases.

    Returns:
        torch.Tensor of shape (n,3) where n is the number of triplets
    """
    if mol.element_class == c_alpha_class_name:
        # Define a triplet as the set of alpha carbon atoms that are adjacent along the sequence.
        n_residues = mol.n_elements
        triplets_ind = torch.arange(1, n_residues - 1).reshape(-1, 1) + torch.tensor([-1, 0, 1])
        return triplets_ind
    elif mol.element_class in [heavy_atoms_class_name, all_atoms_class_name]:
        # Define the set of triplets from the bond adjacency matrix
        # In this method, the set of triplet indices depends on the order of atoms in the adjacency matrix.
        bond_matrix = torch.zeros(*2 * [bond_energy_module.n_atoms], dtype=torch.bool)
        bond_matrix[bond_energy_module.bonds_atom_ind.unbind(-1)] = 1
        bond_matrix[bond_energy_module.bonds_atom_ind.flip(-1).unbind(-1)] = 1
    elif mol.element_class == c_alpha_sidechain_class_name:
        is_atom_CA = mol.elements_is_CA
        CA_ind = is_atom_CA.nonzero()[:, 0]
        SI_ind = (~is_atom_CA).nonzero()[:, 0]
        if config.coarse_triplet_ind_method in ['1', '2']:
            # Use the same procedure as the heavy-atom model to define the triplets by defining a bond matrix.
            # The coarse grain bond matrix can be defined in multiple ways (see methods below).
            resind_diff = torch.abs(mol.elements_resind.unsqueeze(0) - mol.elements_resind.unsqueeze(1))

            if config.coarse_triplet_ind_method == '1':
                # Coarse grains are bonded to all other coarse grains that are in the previous or next residue.

                # Add bonds between sidechain and backbone coarse grains for each residue
                bond_matrix = resind_diff == 0

                # Add bond between backbone coarse grains that are adjacent along the sequence.
                bond_matrix |= resind_diff == 1 & is_atom_CA & is_atom_CA[:, None]
            elif config.coarse_triplet_ind_method == '2':
                # Coarse grains are bonded to all other coarse grains that are in the previous or next residue.
                # Bonds between first and last residue are also included.
                # Bonds between sidechain coarse grain are removed.

                # Connect all coarse grains whose residues are adjacent along the sequence
                bond_matrix = resind_diff < 2

                # Add bond between coarse grains in first and last residue
                bond_matrix |= resind_diff == (mol.elements_resind.max() - mol.elements_resind.min())

                # Remove bonds between sidechain coarse grains
                bond_matrix[SI_ind, SI_ind[:, None]] = False

            # Remove self bonds (diagonal elements of bond matrix)
            bond_matrix &= ~torch.diag(torch.ones_like(bond_matrix).diag())
        else:
            raise ValueError(f"{config.coarse_triplet_ind_method!r} method not implemented")
    else:
        raise ValueError(f"triplet indices are undefined for element_class {mol.element_class!r}")

    # For each atom, define a triplet for all pairs of atoms that are bonded to the given atom.
    triplets_ind = []
    for i in range(mol.n_elements):
        bonds_atom_ind = torch.nonzero(bond_matrix[i])
        if bonds_atom_ind.numel() < 2:
            continue  # Not enough atoms to form a triplet
        all_pairs_ind = torch.combinations(bonds_atom_ind.flatten(), r=2)
        center_atom_ind = torch.tensor(i).expand(all_pairs_ind.shape[0], 1)
        triplet_ind_tmp = torch.cat([center_atom_ind, all_pairs_ind], dim=1)
        triplets_ind.append(triplet_ind_tmp)
    triplets_ind = torch.cat(triplets_ind)

    return triplets_ind
