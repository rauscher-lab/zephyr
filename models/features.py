from collections import defaultdict
import numpy as np
import torch
from torch.nn.functional import one_hot
import tqdm

from chemistry.grains import coarse_grains_sets
from chemistry.energy import fetch_energy_module, BondEnergy, LJEnergy
from data.data_classes import Biomolecule
from data.forcefields import Forcefield, forcefields
from utils import remove_centroid

import config

topological_node_features = {'type': None, 'residue_type': None, 'res_elem_type': None, 'element_ind': None,
                             'element_ind_within_residue': None, 'residue_ind': None, 'charge': None,
                             'sequence_type': None, 'is_backbone': None,
                             'element_coarse_ind': None, 'coarse_atom_ind': None, 'coarse_radius': None}
variable_node_features = ['element_pos', 'coarse_pos']
supported_node_features = variable_node_features + list(topological_node_features)
topological_node_features_cache = defaultdict(lambda: topological_node_features)

topological_edge_features = {'topology_bond_length': None, 'is_bonded': None, 'eps_LJ': None, 'sig_LJ': None,
                             'qq': None, 'res_sim': None, 'seq_pos_diff_ter': None}
variable_edge_features = ['element_adj']
supported_edge_features = variable_edge_features + list(topological_edge_features)
topological_edge_features_cache = defaultdict(lambda: topological_edge_features)
coarse_grains = coarse_grains_sets.get_adj_grains(method=config.node_adj_method)


def calculate_node_features(mol: Biomolecule, element_props_name: list[str], element_types: list[str] = None,
                            pos_scale=1.0, residue_types: list[str] = None, forcefield: Forcefield = None):
    """
    Calculates various node features derived from Biomolecule properties
    Args:
        mol: Biomolecule from which features originate
        element_props_name: list of property names. Names must coincide with the dataset's property names
        element_types: list of element types that will be encoded
        pos_scale: position scale used to rescale position-like features
        residue_types: list of residue types that will be encoded
        forcefield: Forcefield associated with elements (used to define topology-dependent properties)
    Returns:
        dictionary of the requested node features
    """
    if element_types is None:
        element_types = mol.supported_types('element')
    if residue_types is None:
        residue_types = mol.supported_types('residue')

    if forcefield is None:
        forcefield = forcefields[mol.forcefield]

    feat_dict = dict()
    def_float_type = torch.get_default_dtype()
    n_residues = mol.n_residues
    for prop_name in element_props_name:
        if prop_name == 'element_pos':
            prop_val = mol.elements_position.to(def_float_type) / pos_scale
        elif prop_name == 'type':
            # Encode element type as a one-hot vector
            elements_type_ind = torch.tensor([element_types.index(t) for t in mol.elements_type])
            elements_type_onehot = one_hot(elements_type_ind, len(element_types)).to(def_float_type)
            prop_val = elements_type_onehot
        elif prop_name == 'residue_type':
            # Encode residue type as a one-hot vector
            res_names, elements_res_ind = mol.residues_name, mol.elements_resind
            residues_type_ind = torch.tensor([residue_types.index(res_names[i]) for i in elements_res_ind])
            residues_type_onehot = one_hot(residues_type_ind, len(residue_types)).to(def_float_type)
            prop_val = residues_type_onehot
        elif prop_name == 'res_elem_type':
            elements_restype = mol.residues_name[mol.elements_resind]
            elements_res_elem_type = np.char.add(elements_restype, mol.elements_type)
            res_elem_types = np.char.add(np.char.array(residue_types)[:, None],
                                         np.char.array(element_types)[None, :]).flatten()
            prop_val = torch.tensor(elements_res_elem_type[:, None] == res_elem_types[None, :]).to(def_float_type)
        elif prop_name == 'is_backbone':
            prop_val = mol.elements_is_backbone.reshape(-1, 1).to(def_float_type)
        elif prop_name == 'element_ind':
            prop_val = torch.arange(mol.n_elements, dtype=def_float_type).reshape(-1, 1) / mol.n_elements
        elif prop_name == 'element_ind_within_residue':
            residues_name = mol.residues_fullname  # Takes cake of Amber residue names
            elements_resname = residues_name[mol.elements_resind]
            elements_ind_in_res = torch.zeros((mol.n_elements, 1))
            residues_info = forcefield.residue_info

            for i, (resname, atom_name) in enumerate(zip(elements_resname, mol.elements_name)):
                # Handle extra elements in terminal residues
                if mol.elements_resind[i] == 0 and atom_name in ['H1', 'H2']:
                    elem_ind = -int(atom_name[1])
                elif mol.elements_resind[i] == n_residues - 1 and atom_name in ['OT1', 'OT2', 'HT2']:
                    O_ind = residues_info[resname]['atoms']['name'].index('O')
                    elem_ind = {'OT1': O_ind, 'OT2': O_ind + 1, 'HT2': O_ind + 2}[atom_name]
                else:
                    elem_ind = residues_info[resname]['atoms']['name'].index(atom_name)
                elements_ind_in_res[i] = elem_ind
            prop_val = elements_ind_in_res / 20.0
        elif prop_name == 'residue_ind':
            prop_val = (2 * mol.elements_resind / (n_residues - 1)).reshape(-1, 1).to(def_float_type)
        elif prop_name == 'charge':
            prop_val = mol.elements_charge.reshape(-1, 1).to(def_float_type)
        elif prop_name == 'sequence_type':
            prop_val = one_hot(torch.ceil(mol.elements_resind / (n_residues - 2)).long(), 3)
            prop_val = prop_val.to(def_float_type)
        elif prop_name == 'element_coarse_ind':
            prop_val = mol.elements_coarse_ind(method=config.node_adj_method).unsqueeze(-1)
        elif prop_name == 'coarse_pos':
            prop_val = mol.coarse_pos(method=config.node_adj_method, coarse_grains=coarse_grains)
            prop_val = prop_val.to(def_float_type) / pos_scale
        elif prop_name == 'coarse_radius':
            prop_val = mol.coarse_radius(method=config.node_adj_method, coarse_grains=coarse_grains)
            prop_val = prop_val.unsqueeze(-1).to(def_float_type) / pos_scale
        elif prop_name == 'coarse_atom_ind':
            prop_val = mol.coarse_atom_ind(method=config.node_adj_method, coarse_grains=coarse_grains)
            prop_val = prop_val.unsqueeze(-1)
        else:
            raise NotImplementedError(f"property {prop_name!r} is not implemented.")
        feat_dict[prop_name] = prop_val

    return feat_dict


def calculate_edge_features(mol: Biomolecule, edge_feats_name: list[str], pos_scale: float = 1.0):
    bond_energy_module = fetch_energy_module(mol.top_file, mol.element_class, BondEnergy)
    LJ_energy_module = fetch_energy_module(mol.top_file, mol.element_class, LJEnergy)

    # Edge features attributes
    edge_features = dict()
    def_float_type = torch.get_default_dtype()
    for edge_feat_name in edge_feats_name:
        if edge_feat_name == 'topology_bond_length':
            # Float feature that denotes the bond length between two atoms. -1 is given to atoms that are not bonded.
            b0 = (bond_energy_module.b0 / pos_scale).to(def_float_type)
            top_bond_length = -torch.ones(mol.n_elements, mol.n_elements)  # -1=non-bonded atoms
            top_bond_length[bond_energy_module.bonds_atom_ind.unbind(-1)] = b0
            top_bond_length[bond_energy_module.bonds_atom_ind.flip(dims=[-1]).unbind(-1)] = b0
            edge_features[edge_feat_name] = top_bond_length.unsqueeze(-1)
        elif edge_feat_name == 'is_bonded':
            # Boolean feature that indicates whether two atoms share a bond
            edge_features[edge_feat_name] = (2 * bond_energy_module.bond_matrix - 1).to(def_float_type).unsqueeze(-1)
        elif edge_feat_name == 'eps_LJ':
            # eps_LJ (Lennard-Jones parameter) parameter between two atoms
            eps_LJ = - torch.ones(mol.n_elements, mol.n_elements)
            eps_LJ_1D = LJ_energy_module.epsilon.to(def_float_type)
            eps_LJ[LJ_energy_module.pairs_ind.unbind(-1)] = eps_LJ_1D
            eps_LJ[LJ_energy_module.pairs_ind.flip(dims=[-1]).unbind(-1)] = eps_LJ_1D
            edge_features[edge_feat_name] = eps_LJ.unsqueeze(-1)
        elif edge_feat_name == 'sig_LJ':
            # sig_LJ (Lennard-Jones parameter) parameter between two atoms
            sig_LJ = - torch.ones(mol.n_elements, mol.n_elements)
            sig_LJ_1D = (LJ_energy_module.sigma / pos_scale).to(def_float_type)
            sig_LJ[LJ_energy_module.pairs_ind.unbind(-1)] = sig_LJ_1D
            sig_LJ[LJ_energy_module.pairs_ind.flip(dims=[-1]).unbind(-1)] = sig_LJ_1D
            edge_features[edge_feat_name] = sig_LJ.unsqueeze(-1)
        elif edge_feat_name == 'qq':
            # Charge product between two nodes
            q = bond_energy_module.atoms_charge
            qq = torch.zeros(mol.n_elements, mol.n_elements)
            qq_1D = (q.reshape(-1, 1) * q).to(def_float_type)[LJ_energy_module.pairs_ind.unbind(-1)]
            qq[LJ_energy_module.pairs_ind.unbind(-1)] = qq_1D
            qq[LJ_energy_module.pairs_ind.flip(dims=[-1]).unbind(-1)] = qq_1D
            edge_features[edge_feat_name] = qq.unsqueeze(-1)
        elif edge_feat_name == 'res_sim':
            # Boolean feature that indicates whether two atoms are part of the residue
            res_similarity = mol.elements_resind[None, :] == mol.elements_resind[:, None]
            edge_features[edge_feat_name] = res_similarity.unsqueeze(-1).to(torch.get_default_dtype())
        elif edge_feat_name == 'seq_pos_diff_ter':
            res_ind_diff = torch.abs(mol.elements_resind[None, :] - mol.elements_resind[:, None])
            seq_pos_diff_ter = torch.stack([res_ind_diff == 0, res_ind_diff == 1, res_ind_diff > 1], dim=-1)
            edge_features[edge_feat_name] = seq_pos_diff_ter.to(torch.get_default_dtype())
        elif edge_feat_name == 'element_adj':
            # Calculate the nodes' adjacency matrix
            elem_adj = mol.calculate_elements_adj(method=config.node_adj_method, coarse_grains=coarse_grains,
                                                  buffer_dist=config.node_adj_range)
            edge_features[edge_feat_name] = elem_adj.unsqueeze(-1)
        else:
            raise NotImplementedError(f"Edge feature {edge_feat_name!r} is not implemented.")

    return edge_features


def assemble_node_features(mols: list[Biomolecule], feats_name: list[str], verbose=False, **kwargs):
    unsupported_feats = set(feats_name) - set(supported_node_features)
    if unsupported_feats:
        raise ValueError(f"The following node features are not supported:\n{unsupported_feats}")
    feats_dict = {k: None for k in feats_name}

    # Gather the constant node features that depend on the topology.
    const_node_features = [f for f in feats_name if f in topological_node_features]

    # Calculate the constant features that are not cached
    mol_ex = mols[0]
    cache_key = (mol_ex.top_file, mol_ex.element_class)
    top_feat_cache = topological_node_features_cache[cache_key]
    top_feat_missing = [f for f in const_node_features if top_feat_cache[f] is None]
    if top_feat_missing:
        mol_feats_const = calculate_node_features(mol_ex, top_feat_missing, **kwargs)
        for k, v in mol_feats_const.items():
            top_feat_cache[k] = v.unsqueeze(0)
    feats_dict |= {k: top_feat_cache[k].expand(len(mols), -1, -1) for k in const_node_features}

    # Assemble the variable node features
    var_node_features = [f for f in feats_name if f not in topological_node_features]
    if var_node_features:
        var_feats = defaultdict(lambda: list())
        for mol in tqdm.tqdm(mols, 'Assembling node features', disable=not verbose):
            mol_feats = calculate_node_features(mol, var_node_features, **kwargs)
            for k, v in mol_feats.items():
                var_feats[k].append(v)
        feats_dict |= {k: torch.stack(v) for k, v in var_feats.items()}

    return feats_dict


def assemble_edge_features(mols: list[Biomolecule], feats_name: list[str], verbose=False, **kwargs):
    unsupported_feats = set(feats_name) - set(supported_edge_features)
    if unsupported_feats:
        raise ValueError(f"The following edge features are not supported:\n{unsupported_feats}")
    feats_dict = {k: None for k in feats_name}

    # Gather the constant edge features that depend on the topology.
    const_edge_features = [f for f in feats_name if f in topological_edge_features]

    # Calculate features that are not cached
    mol_ex = mols[0]
    cache_key = (mol_ex.top_file, mol_ex.element_class)
    top_feat_cache = topological_edge_features_cache[cache_key]
    top_feat_missing = [f for f in const_edge_features if top_feat_cache[f] is None]
    if top_feat_missing:
        mol_feats_const = calculate_edge_features(mol_ex, top_feat_missing, **kwargs)
        for k, v in mol_feats_const.items():
            top_feat_cache[k] = v.unsqueeze(0)
    feats_dict |= {k: top_feat_cache[k].expand(len(mols), -1, -1, -1) for k in const_edge_features}

    # Assemble the variable edge features
    var_edge_features = [f for f in feats_name if f not in topological_edge_features]
    if var_edge_features:
        var_feats = defaultdict(lambda: list())
        for mol in tqdm.tqdm(mols, 'Assembling edge features', disable=not verbose):
            mol_feats = calculate_edge_features(mol, var_edge_features, **kwargs)
            for k, v in mol_feats.items():
                var_feats[k].append(v)
        feats_dict |= {k: torch.stack(v) for k, v in var_feats.items()}

    return feats_dict


def assemble_features(mols: list[Biomolecule], node_features: list[str] = None, edge_features: list[str] = None,
                      pos_scale: float = 1.0, verbose=False):
    node_feats = {}
    edge_feats = {}
    if node_features:
        node_feats |= assemble_node_features(mols, feats_name=node_features, pos_scale=pos_scale, verbose=verbose)
    if edge_features:
        edge_feats |= assemble_edge_features(mols, feats_name=edge_features, pos_scale=pos_scale, verbose=verbose)

    return node_feats, edge_feats


def gather_input_tensors(biomolecules: list[Biomolecule], pos_scale=1.0, verbose=False):
    # Gather variable tensors
    nodes_feat = assemble_node_features(biomolecules, ['element_pos'], pos_scale=pos_scale, verbose=verbose)
    edges_feat = assemble_edge_features(biomolecules, ['element_adj'], pos_scale=pos_scale, verbose=verbose)
    nodes_pos = nodes_feat['element_pos']
    nodes_adj = edges_feat['element_adj'].squeeze(-1)

    # Remove centroid from structures. Check for nans in case biomolecules are refined from coarser molecules
    COM_mask = ~nodes_pos.isnan()
    nodes_pos, nodes_pos_centroid = remove_centroid(nodes_pos, mask=COM_mask, return_centroid=True)  # Remove centroid

    tensors_dict = dict(nodes_pos=nodes_pos, nodes_adj=nodes_adj, nodes_adj_range=config.node_adj_range / pos_scale)

    # Gather tensors needed when node adjacency matrices are not full
    if config.node_adj_range < torch.inf:
        extra_node_feats_name = ['element_coarse_ind', 'coarse_pos', 'coarse_radius', 'coarse_atom_ind']
        tensors_dict |= assemble_node_features(biomolecules, extra_node_feats_name, pos_scale=pos_scale,
                                               verbose=verbose)
        tensors_dict['coarse_pos'] = tensors_dict['coarse_pos'] - nodes_pos_centroid  # Remove same centroid as above
        tensors_dict['coarse_rad'] = tensors_dict.pop('coarse_radius')  # rename
        tensors_dict['nodes_coarse_ind'] = tensors_dict.pop('element_coarse_ind')  # rename

    return tensors_dict


if __name__ == '__main__':
    from data.datasets import load_dataset

    config.import_configs('config_test')

    dataset = load_dataset(load_all=True)
    tensors = gather_input_tensors(dataset.molecules)
