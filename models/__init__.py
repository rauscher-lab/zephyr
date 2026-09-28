import re

import torch

import config
from chemistry.grains import coarse_grains_sets, supported_adj_methods
from data.datasets import BiomoleculeDataset, load_dataset
from data.download import download_model_weights
from models.conditioners import variable_conditioners, BondLengthBoundsConditioner, \
    BondAngleEnddistBoundsConditioner, GlobalLJDistMinConditioner, LocalLJDistMinConditioner
from models.losses import BondLengthLoss, BondAngleLoss, DihedralAngleLoss, BondEnergyLoss, BondAngleEnergyLoss, \
    DihedralAngleEnergyLoss, LJEnergyLoss, LJDistLoss, RgLoss, DiffLoss
from models.base import DiffGraph, DiffPGNN
from models.diffusers import CorrelatedNoiser
from models.features import assemble_node_features
from models.generate import generate_samples

try:
    from sync import download_file
except ImportError:
    download_file = None


def format_checkpoint_filepath(config_ID: str = None, dataset: str = None, split_set_ID: str = None,
                               epoch: str | int = None):
    """
    Formats the filepath of model checkpoints
    Args:
        config_ID: ID of the config. Defaults=config.ID
        dataset: name of the dataset. Defaults=config.dataset
        split_set_ID: split set ID used to train the given model. Defaults=config.dataset_split_set_ID
        epoch: add epoch tag (optional)

    Returns:
        filename
    """
    if config_ID is None:
        config_ID = config.ID
    if dataset is None:
        dataset = config.dataset
    if split_set_ID is None:
        split_set_ID = config.dataset_split_set_ID

    epoch_tag = f"_{epoch}" if epoch else ''
    split_set_ID_tag = f"_{split_set_ID}"
    filename = f"{config_ID}_{dataset}{split_set_ID_tag}{epoch_tag}.pth"
    filepath = config.weights_dir / filename

    return filepath


def initialize_conditioners(dataset_train: BiomoleculeDataset, ref_dataset: BiomoleculeDataset = None,
                            corr_noiser: CorrelatedNoiser = None, pos_scale: float = 1.0, conditioners_name=None):
    """
    Initializes the set of conditioners used during sampling
    Args:
        dataset_train: training dataset used during training. Statistics of this dataset are used to defined restraints of properties by conditioners.
        ref_dataset: reference dataset. Used to define more general conditioners that do not need statistics.
        corr_noiser: CorrelatedNoiser used during sampling (if applicable).
        pos_scale: float to rescale the positions
        conditioners_name: list of conditioners used during sampling

    Returns:
        list of conditioners
    """
    if conditioners_name is None:
        conditioners_name = config.conditioners
    if ref_dataset is None:
        ref_dataset = dataset_train

    # Remove variable conditioners. They are defined later in sample() method of the model.
    conditioners_name = [n for n in conditioners_name if n not in variable_conditioners]

    conditioners = []
    for conditioner in conditioners_name:
        sig_deactivate = config.sampling_cond_sig_deact[conditioner]
        if conditioner == 'bond_length_bounds':
            conditioner = BondLengthBoundsConditioner(top_filepath=dataset_train.top_filepath,
                                                      dataset_stats=dataset_train.statistics('bond_length'),
                                                      corr_noiser=corr_noiser,
                                                      pos_scale=pos_scale,
                                                      element_class=dataset_train.element_class,
                                                      sigma_deactivate=sig_deactivate)
        elif conditioner == 'bond_length_top_bounds':
            conditioner = BondLengthBoundsConditioner(top_filepath=ref_dataset.top_filepath,
                                                      dataset_stats=None,
                                                      corr_noiser=corr_noiser,
                                                      pos_scale=pos_scale,
                                                      element_class=ref_dataset.element_class,
                                                      sigma_deactivate=sig_deactivate)
        elif conditioner == 'bond_angle_enddist_bounds':
            conditioner = BondAngleEnddistBoundsConditioner(top_filepath=dataset_train.top_filepath,
                                                            dataset_stats=dataset_train.statistics(
                                                                'bond_angle_enddist'),
                                                            corr_noiser=corr_noiser,
                                                            pos_scale=pos_scale,
                                                            element_class=dataset_train.element_class,
                                                            sigma_deactivate=sig_deactivate)
        elif conditioner.startswith('LJ_global_pdist_min'):
            cond_patt = r'LJ_global_pdist_min_([\d\.]+)'
            match_res = re.match(cond_patt, conditioner)
            if not match_res:
                raise ValueError(f"Formatting of conditioner must match pattern:{cond_patt!r}")
            pdist_min = float(match_res.group(1)) / pos_scale  # Assumes float value string is given in nm
            conditioner = GlobalLJDistMinConditioner(top_filepath=ref_dataset.top_filepath,
                                                     pdist_min=pdist_min,
                                                     corr_noiser=corr_noiser,
                                                     pos_scale=pos_scale,
                                                     element_class=ref_dataset.element_class,
                                                     sigma_deactivate=sig_deactivate)
        elif conditioner.startswith('LJ_local_pdist_min'):
            cond_patt = r'LJ_local_pdist_min_([\d\.]+)'
            match_res = re.match(cond_patt, conditioner)
            if not match_res:
                raise ValueError(f"Formatting of conditioner must match pattern:{cond_patt!r}")
            pdist_min = float(match_res.group(1)) / pos_scale  # Assumes float value string is given in nm
            conditioner = LocalLJDistMinConditioner(top_filepath=ref_dataset.top_filepath,
                                                    pdist_min=pdist_min,
                                                    corr_noiser=corr_noiser,
                                                    pos_scale=pos_scale,
                                                    element_class=ref_dataset.element_class,
                                                    sigma_deactivate=sig_deactivate)
        else:
            raise ValueError(f"conditioner={conditioner} is not supported.")
        conditioners.append(conditioner)

    return conditioners


def initialize_aux_losses(dataset_train: BiomoleculeDataset, pos_scale: float = 1.0):
    losses = []
    for loss_name in config.aux_losses:
        if loss_name in ['bond_length', 'bond_length_sq']:
            power = 1.0 if loss_name == 'bond_length' else 2.0
            loss = BondLengthLoss(top_filepath=dataset_train.top_filepath, power=power, bond_weights=None,
                                  pos_scale=pos_scale)
        elif loss_name == 'bond_angle':
            loss = BondAngleLoss(top_filepath=dataset_train.top_filepath,
                                 pos_scale=pos_scale)
        elif loss_name == 'dihedral_angle':
            loss = DihedralAngleLoss(top_filepath=dataset_train.top_filepath)
        elif loss_name == 'LJ_dist':
            loss = LJDistLoss(top_filepath=dataset_train.top_filepath,
                              pos_scale=pos_scale)
        elif loss_name.startswith('bond_energy'):
            loss = BondEnergyLoss(top_filepath=dataset_train.top_filepath, tot=loss_name.endswith('tot'),
                                  pos_scale=pos_scale)
        elif loss_name.startswith('bond_angle_energy'):
            loss = BondAngleEnergyLoss(top_filepath=dataset_train.top_filepath, tot=loss_name.endswith('tot'),
                                       pos_scale=pos_scale)
        elif loss_name.startswith('dihedral_angle_energy'):
            loss = DihedralAngleEnergyLoss(top_filepath=dataset_train.top_filepath, tot=loss_name.endswith('tot'),
                                           pos_scale=pos_scale)
        elif loss_name.startswith('LJ_energy'):
            loss = LJEnergyLoss(top_filepath=dataset_train.top_filepath, tot=loss_name.endswith('tot'),
                                pos_scale=pos_scale)
        elif loss_name == 'Rg':
            loss = RgLoss()
        elif loss_name == 'Diff':
            n_elements = dataset_train[0].n_elements
            loss = DiffLoss(d=3 * n_elements)
        else:
            raise ValueError(f"conditioner={loss_name} is not supported.")
        losses.append(loss)

        return losses


def initialize(dataset_train: BiomoleculeDataset | str = None, dataset_sampling: BiomoleculeDataset = None,
               config_name: str = None):
    if config_name:
        config.import_configs(config_name)

    # Load the training dataset
    if dataset_train is None:
        dataset_train = config.dataset
    if isinstance(dataset_train, str):
        dataset_train = load_dataset(name=dataset_train, element_class=config.dataset_element_class)
        dataset_train = dataset_train.load_splits('train', set_ID=config.dataset_split_set_ID)
    if not dataset_train.is_split or not dataset_train.name.endswith('train'):
        raise ValueError(f"Cannot initialize model using a dataset that is not a training dataset.")

    # Determine feature scales using the train dataset
    features_scale = config.feats_scale
    data_scales = dict()
    data_scales['pos'] = dataset_train.statistics('elements_position').std
    data_scales['pos_local'] = dataset_train.statistics('elements_position_local').std
    if config.node_adj_method is None:
        pass
    elif config.node_adj_method in supported_adj_methods:
        stat_name = f'elements_position_{config.node_adj_method}_local'
        data_scales[f'pos_{config.node_adj_method}_local'] = dataset_train.statistics(stat_name).std
    else:
        raise NotImplementedError

    # Determine the dimension of the conditional features
    cond_node_features = {}
    for feat_name in config.cond_node_features:
        if feat_name in ['element_ind', 'element_ind_within_residue', 'residue_ind', 'charge', 'is_backbone']:
            cond_node_features[feat_name] = 1
        elif feat_name == 'sequence_type':
            cond_node_features[feat_name] = 3
        elif feat_name == 'type':
            cond_node_features[feat_name] = len(dataset_train.element_types)
        elif feat_name == 'residue_type':
            cond_node_features[feat_name] = len(dataset_train.residue_types)
        elif feat_name == 'res_elem_type':
            cond_node_features[feat_name] = len(dataset_train.element_types) * len(dataset_train.residue_types)
        else:
            raise ValueError(f"conditional feature {feat_name!r} is not supported.")
    cond_edge_features = {}
    for feat_name in config.cond_edge_features:
        if feat_name in ['topology_bond_length', 'is_bonded', 'eps_LJ', 'sig_LJ', 'qq', 'res_sim']:
            cond_edge_features[feat_name] = 1
        elif feat_name == 'seq_pos_diff_ter':
            cond_edge_features[feat_name] = 3
        else:
            raise ValueError(f"conditional feature {feat_name!r} is not supported.")

    # Initialize CorrelatedNoiser
    corr_diff_pos_scale = features_scale['pos'] if config.correlated_diffusion_scale else 1.0
    coarse_grains = coarse_grains_sets.get_adj_grains(method=config.node_adj_method)

    if config.correlated_diffusion == 'chain+bonds':
        corr_noiser = CorrelatedNoiser(topology=dataset_train.top_filepath,
                                       element_class=dataset_train.element_class,
                                       chain_scaling='disordered2', centroid_std=1.0,
                                       pos_scale=corr_diff_pos_scale, sidechain_coarse_grains=coarse_grains)
    elif config.correlated_diffusion == 'bonds':
        corr_noiser = CorrelatedNoiser(topology=dataset_train.top_filepath,
                                       element_class=dataset_train.element_class,
                                       chain_scaling=None, centroid_std=1.0,
                                       pos_scale=corr_diff_pos_scale, sidechain_coarse_grains=coarse_grains)
    else:
        corr_noiser = None

    # Auxiliary losses
    losses = initialize_aux_losses(dataset_train=dataset_train, pos_scale=features_scale['pos'])

    # Conditioners used during sampling
    conditioners = initialize_conditioners(dataset_train=dataset_train, ref_dataset=dataset_sampling,
                                           corr_noiser=corr_noiser, pos_scale=features_scale['pos'])

    net = DiffGraph(cond_node_features=cond_node_features,
                    cond_edge_features=cond_edge_features,
                    features_scale=features_scale,
                    corr_noiser=corr_noiser,
                    data_scales=data_scales,
                    aux_losses=losses,
                    conditioners=conditioners)

    return net


def load_checkpoint(config_ID: str = None, dataset: str = None, split_set_ID: str = None,
                    epoch: str | int = None):
    """
    Loads training checkpoint
    Args:
        config_ID: ID of the config. Defaults=config.ID
        dataset: name of the dataset. Defaults=config.dataset
        split_set_ID: split set ID used to train the given model. Defaults=config.dataset_split_set_ID
        epoch: add epoch tag (optional)
    Returns:
        checkpoint dictionary
    """

    checkpoint_filepath = format_checkpoint_filepath(config_ID=config_ID, dataset=dataset, split_set_ID=split_set_ID,
                                                     epoch=epoch)
    if not checkpoint_filepath.exists():
        download_model_weights(checkpoint_filepath)
    if not checkpoint_filepath.exists():
        raise FileNotFoundError(f"Model weights located at {str(checkpoint_filepath)!r} was not found")
    checkpoint_dict = torch.load(checkpoint_filepath, torch.device('cpu'), weights_only=False)
    return checkpoint_dict


def load(config_filename: str = None, dataset: str = None, epoch: str | int = None, **init_kwargs) -> DiffGraph:
    """
    Loads model for generating structures
    Args:
        config_filename: filename of the config used to train the model
        dataset: dataset on which the model was trained. Defaults to config.dataset
        epoch: epoch number of the specific model state to load.  Defaults to last epoch
        **init_kwargs: kwargs given to initialize()

    Returns:
        model
    """
    if config_filename:
        config.import_configs(config_filename)
    model = initialize(dataset_train=dataset, **init_kwargs)

    # Load model weights from the net_ema_state if it was used.
    checkpoint_dict = load_checkpoint(dataset=dataset, epoch=epoch)
    net_state_key = 'net_ema_state' if config.model_ema_beta > 0 else 'net_state'
    model.load_state_dict(checkpoint_dict[net_state_key], strict=False)

    return model


def load_train_stats(config_ID: str, dataset: str, split_set_ID: str = '1'):
    """
    Loads training statistics of a particular model
    Args:
        config_ID: ID of the config with which the model was trained
        dataset: dataset onto which the model was trained
        split_set_ID: set ID of the dataset splits onto which the model was trained

    Returns:
        dictionary of stats
    """
    checkpoint_dict = load_checkpoint(config_ID=config_ID, dataset=dataset, split_set_ID=split_set_ID)
    train_stats = checkpoint_dict['epoch_stats']
    return train_stats
