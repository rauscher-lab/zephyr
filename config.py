import copy
import datetime
import os
import pathlib
import sys
import warnings
import inspect
import uuid
import importlib.util
import argparse
import tokenize
import io
import json
from pathlib import Path

import torch
from types import ModuleType, FunctionType
import MDAnalysis  # This is needed for the warning filter below to take precedence over filters added by MDAnalysis

# Ignore DeprecationWarning from ITPParser module from MDAnalysis
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*MDAnalysis\\.topology\\.ITPParser.*")

root_dir = Path(__file__).parent  # Root directory

local_vars_begin = list(locals().keys()) + ['local_vars_begin']

##################### Default configurations #####################
# Directories
configs_dir = Path(root_dir, 'configs')  # Directory where configs are saved
proc_datasets_dir = Path(root_dir, 'datasets', 'processed')  # Directory where processed datasets are saved
MD_sim_dir = Path(root_dir, 'datasets', 'MD')  # Directory where raw MD trajectories are saved
gen_datasets_dir = Path(root_dir, 'datasets', 'generated')  # Directory where generated datasets are saved
weights_dir = Path(root_dir, 'weights')  # Directory where model weights are saved
plots_dir = Path(root_dir, 'plots')  # Directory where plots are saved
wandb_logs_dir = ''  # Path where WandB (if used) saves logs

# Training
ID = 'default'  # ID of the configs
project_name = 'myproject'  # Name of the project used by WandB logging
wandb = False  # Log various epoch stats with wandb
wandb_log_params = False  # Log parameters and their gradients in wandb.
optimizer = 'Adam'  # Name of the optimizer. Can be ['Adam','AdamW','AdamWScheduleFree','AdEMAMix','RAdam']
learning_rate = 0.0001  # Learning rate
weight_decay = 0.0  # Weight decay parameter. Used when optimizer='AdamW'.
lr_piecewise_params = None  # Dictionary defining piecewise LR parameters. Must have 'epochs', 'lrs' and 'funcs' keys. Overrides learning_rate
batch_size = 64  # Number of dataset samples used to evaluate gradients of each optimization step
batch_size_sch = None  # Dictionary {epoch:batch_size} defining the epoch when the batch size changes
minibatch_size = None  # Number of dataset samples used per backward pass. Must be <=batch_size and an integer multiple. Defaults to value of batch_size.
n_minibatches = 1  # Number of minibatches used to accumulate gradients before optimization step
n_batches_per_epoch = 32  # Number of training batches in 1 epoch of training.
drop_last_batch = True  # Drop last batch if it doesn't have enough samples
dataloader_sampler = None  # Sample training samples with replacements. Can be ['with_replacement',None]
N_epochs = 4000  # Number of training epochs
N_epochs_session = None  # Number of training epochs to run in the given session. Run all training epochs by default
adam_betas = (0.9, 0.9)  # Beta parameters of the Adam optimizer
amsgrad = False  # AMSGRAD option for adam optimizer
optim_extra_kwargs = {}  # Extra keyword arguments passed to optimizer
resume = True  # Resume training
init_ID = None  # ID of a previous config used to initialize model weights and stats before training starts
N_workers = 0  # Number of workers to load the data onto the device
seed = 1  # Random seed used by torch, numpy and random
checkpoint_period = 120  # Period (sec) at which a checkpoint of the network parameters is saved
checkpoint_extra_period = 500  # Period (epochs) at which extra checkpoints are performed. Extra checkpoints are not overwritten.
device = 'default'  # Device used for training. Can be ['cpu', 'mps', 'cuda', 'default']. See below for default device.
pin_memory = False  # Pin data to device memory in data loader
feats_scale = {'pos': 0.1}  # Scale of model features
grad_rel_norm_max = None  # Maximal relative norm of parameter gradients. Used to do gradient clipping.
grad_norm_max = None  # Maximal absolute norm of parameter gradients. Used to do gradient clipping.
weighted_t_sampling = False  # weight the sampling of the diffusion in the loss
importance_sampling_logSNR = False  # Weighs each batch sample by 1/p(logSNR) where p(logSNR) is the pdf of the sampled logSNR
model_ema_beta = 0  # beta parameter to calculate a model EMA during training. The averaged model is used for eval. Default is no EMA.
t_sampling_method = 'random'  # Method used to sample batches of timesteps during training
aux_losses = []  # Name of auxiliary losses used during training
aux_loss_estimate_cov = False  # Estimate covariance of auxiliary losses before training starts
aux_loss_reweight = False  # Reweight auxiliary loss such that <L> = 1 with a zeroed denoiser
aux_loss_weight_method = 'mu2_inv'  # Method used to auto-normalize the auxiliary loss
aux_loss_stats_n_min = int(1e4)  # Number of samples collected to evaluate statistics used to define losses properties
aux_losses_weight = None  # Weight applied to each auxiliary loss. Must be None or list[float] with same length as aux_losses
aux_losses_act_epoch = []  # Activation epoch of each auxiliary loss. Defaults to active at beginning of training
amp = False  # Use Automatic Mixed Precision during training
amp_dtype = torch.float16  # Precision used with amp
compile_model = False  # Compile model with torch.compile()

# Sampling
n_samples = 100  # Number of samples generated at each sampling period during training
sampling_period = 50  # Period (epochs) at which samples are generated and analyzed during training
Heun_sampler = False  # Use Heun's 2nd order sampling method when SDE sampling is off
conditioners = list()  # List of conditioners used during sampling
sampling_cond_bounds_sig = 5.0  # Sigma used to define the bounds of conditioners (when applicable)
sampling_cond_sig_deact = {}  # Dictionary of sigma value at which each sampling conditioner is deactivated. Default=0.0.
sampling_ref_dataset = 'valid'  # Reference dataset used for generating conditioned samples
correlated_diffusion = None  # Use a correlated diffusion process to encore scaling laws in the prior distribution
correlated_diffusion_scale = False  # Scale the correlated noise in the correlated diffuser
SDE_sampler = False  # Sample with the SDE. Otherwise, sample with the probabilistic flow ODE
SDE_sampler_params = {'psi': 0.0, 'lambda0': 1.0}  # Parameters of the SDE sampler
estimate_cond_covariance = True  # Estimate conditioner's covariance matrix from noised data
coarse_pos_precision = 0.1  # Precision (nm) of coarse grain position targets in PositionConditioner
dynamic_denoising = False  # Denoise samples dynamically using the potential energy of the samples to estimate the denoising time
dynamic_denoising_params = {'sig_thresh': 3.0}  # Parameters for dynamic denoising
sampling_align_coarse_grain = False  # Align coarse grain backbone at each sampling iteration when sampling heavy-atoms
sampling_dtype = torch.float32  # Dtype used for sampling

# Dataset
dataset = 'nup98_12'  # Name of the dataset
dataset_element_class = 'heavy_atoms'  # Element class of the dataset
dataset_split_set_ID = '1'  # Default split set ID used when loading dataset splits
dataset_load_all = False  # Load all molecules onto device prior to training
coarse_grains_dataset = 'Rich_2018'  # Dataset used to define the coarse grains center and radius. Uses all atoms

# Architecture
diff_model = 'EDM'  # Model used to perform the diffusion steps (loss, sampling, etc.)
model_bias = False  # Add bias in model layers
dropout = 0.0  # Dropout rate of Dropout layers. No layers will be added when dropout=0.0
model_output_pos_diff = False  # Output the node position difference as opposed to the absolute value
edge_features = []  # Edge features used in the message and position network (phi_msg and phi_x)
global_features = []  # Global features used in the message and position network (phi_msg and phi_x).
cond_node_features = []  # Node features used for conditioning the model
cond_edge_features = []  # Edge features used for conditioning the model
t_as_input_feature = True  # Concatenate time to the input node features before processing them through the layers.
n_layers = 4  # Number of message-passing layers in the graph neural network.
pos_msg_agg = 'sum_unnorm'  # Method to aggregate messages for the node position update. Can be ['sum','mean','sum_unnorm']
d_inv_edge_feat = False  # Add d_ij^-1 = 1/(d_ij + epsilon) as an edge feature in message-passing layers
epsilon_x = 1.0  # Stabilization parameter used in the position update aggregation (x_i-x_j)/(d_ij + epsilon_x)
epsilon_d_inv = 5e-1  # Epsilon parameter used in defining inverse distance features d^-1 = 1/(d + epsilon)
phi_x_weight = False  # Multiply the output of phi_x in layers with a learnable weight
phi_weight_act = 'local_softmax'  # Function used at the end of the phi_msg and phi_x weights layers.
feat_msg_agg = 'sum'  # Method to aggregate messages for the node features update
node_latent_dim = 64  # Dimension of the latent node feature space
edge_latent_dim = 64  # Dimension of the latent edge feature space
n_diff_steps = 1000  # Number of diffusion steps (also denoted as "T" in reference)
n_diff_steps_monitoring = None  # Number of diffusion steps used to produce monitoring samples during training
cross_prod_layer = None  # Type of cross-product layer that updates node positions with cross-products of triplets of elements
cross_prod_layer_hidden_d = None  # Hidden dimension of the cross-product layer phi module. Default = node_latent_dim/4
cross_prod_layer_phi_weight_act = 'local_softmax'  # Activation function used in the phi_weight module of cross-product layers
cross_prod_use_edge_feat = False  # Use edge features in cross product layer
coarse_triplet_ind_method = None  # Method used to define node triplets used in cross product modules
n_cross_prod_layers = 0  # Number of cross-product layers used
act_fn = torch.nn.SiLU()  # Activation function used in network's blocks.
noise_sch_type = 'logSNR-linear'  # Type of the noise schedule. Can be ['polynomial','cosine','sigmoid','half-sigmoid','exp','logSNR-linear']
noise_sch_params = None  # Parameter for the noise schedule
node_pos_res = True  # Add residual connection for the node position update in message-passing layers
edge_feat_res = True  # Add residual connection for the edge feature update in message-passing layers
node_feat_res = True  # Add residual connection to the node features update in message-passing layers
global_feat_res = True  # Add residual connection for the global feature update in message-passing layers
model_progressive = False  # Progressively activate layers of the model during training.
omega_step = 1e-4  # Step size of the progression when model_progressive=True
omega_max = 4.0  # Maximum value of omega that triggers a layer growth when model is progressive
norm_layer = None  # Normalization layers used after each activation. Can be ['layernorm',None]. Default=None.
norm_layer_bias = True  # Add bias parameters in normalization layers.
enc_dec_bias = False  # Add bias parameters in the encoder and decoder
node_adj_range = torch.inf  # Increase the coarse grain adjacency distance threshold (nm) which determines whether coarse grains are adjacent in the graph topology.
node_adj_method = None  # Method used to determine the nodes adjacency matrix
backbone_grain_radius = 0.27  # Radius (nm) of backbone coarse grain centered at CA atom. Used to determine atom adjacencies when node_adj_method='CA_SI'
node_feat_agg_in_edge_updt = None  # Aggregation method of node features in the edge feature update.
remove_noise_COM = True  # Remove the centroid (or center-of-mass) of the sampled noise applied to each structure
sig_pos_in = None  # Input position sigma (nm) used in EDM diffusion model
sig_pos_out = None  # Output position sigma (nm) used in EDM diffusion model
output_gain_init = None  # Initial value of output gain in denoiser
use_rot_layer = False  # Replaces all instances of square nn.Linear layers with RotationLayers

# Misc
debug = False  # Turn on debug mode to do extra checks
float_precision = torch.float32  # Default float precision of torch

################################### END OF CONFIGURATION VARIABLES ###################################

# Determine the local variables. Used to get the name of all configurations.
local_vars = list(locals().keys())

# Fixed configurations
filename = pathlib.Path(__file__).name
data_dir = Path(root_dir, 'data')  # Directory where auxiliary data is saved

# Gather the name of all configurations and their default values
default_configs = {var: eval(var) for var in local_vars if var not in local_vars_begin}

# Add data directory to the Gromacs lib PATH variable to ensure custom force fields are found by GROMACS
if 'GMXLIB' in os.environ:
    if not os.environ['GMXLIB'].startswith(str(data_dir)):
        os.environ['GMXLIB'] = f"{str(data_dir)}:{os.environ['GMXLIB']}"
else:
    os.environ['GMXLIB'] = f"{str(data_dir)}"


def is_config_logged(x: str):
    """
    # Determines if a config should be logged with wandb
    Args:
        x: name of config

    Returns:
        bool
    """
    not_logged_configs = ['checkpoint_period', 'checkpoint_extra_period', 'N_workers', 'resume',
                          'device', 'wandb', 'project_name', 'sampling_period', 'n_samples']
    return x in default_configs and not x.endswith('dir') and x not in not_logged_configs


def format_config(config_val):
    if isinstance(config_val, torch.nn.Module):
        return str(config_val)
    elif isinstance(config_val, FunctionType):
        declaration_str = inspect.getsource(config_val)
        if 'lambda' in declaration_str:
            # Log expressions of lambda functions by inspecting its declaration
            config_str = declaration_str[declaration_str.index('lambda'):]

            # Remove trailing comments and white space
            if '#' in config_str:
                config_str = config_str[:config_str.index('#')]
            config_str = config_str.rstrip()
            return config_str
        else:
            return config_val
    elif isinstance(config_val, Path):
        return str(config_val)
    else:
        return config_val


def logged_configs(configs_dict: dict = None):
    """
    # Outputs configs that are logged with wandb
    Returns:
        dictionary of logged configs with their value
    """
    if configs_dict is None:
        configs_dict = {name: eval(name) for name in default_configs}
    logged_configs_dict = {name: format_config(val) for name, val in configs_dict.items() if is_config_logged(name)}
    return logged_configs_dict


def print_configs():
    """
    Prints the value of the configurations
    """
    print('Configurations:')
    for name in default_configs:
        config_val = format_config(eval(name))
        print(f'{name}:', config_val)


def validate_configs():
    # Value validation
    pos_msg_agg_choices = ['sum', 'sum_unnorm', 'mean', 'kipf', 'sumN', 'sumN_unnorm', 'weightN', 'weightN_unnorm']
    assert isinstance(edge_features, list), 'edge_features must be a list'
    assert pos_msg_agg in pos_msg_agg_choices, f"'{pos_msg_agg}' not supported for pos_msg_agg. Expecting {pos_msg_agg_choices}."

    # Checks
    assert device in ['cpu', 'cuda', 'mps'], f'device:{device} is not supported.'
    assert ID != '', 'The training ID is undefined.'

    if model_progressive:
        assert omega_step is not None, f'omega_step must be set when model_progressive=True'
        assert omega_max is not None, f'omega_max must be set when model_progressive=True'

    if batch_size % minibatch_size != 0:
        raise ValueError(f"minibatch_size must be an integer multiple of batch_size. "
                         f"minibatch_size={minibatch_size}, batch_size={batch_size}.")

    if node_adj_method is None and node_adj_range != torch.inf:
        raise ValueError(f"The adjacency range is finite ({node_adj_range}), but no adjacency method is defined.")


def post_init_process():
    """
    Process configurations after initialization
    """
    global ID, configs_dir, gen_datasets_dir, proc_datasets_dir, device, weights_dir, plots_dir, wandb_logs_dir, \
        float_precision, N_epochs_session, debug, wandb, aux_losses_act_epoch, dataset_load_all, \
        minibatch_size, batch_size_sch, dataset_element_class, sampling_cond_sig_deact

    # Disable wandb in debugging or test mode
    if wandb & debug:
        warnings.warn('Disabling Wandb since debugging is True.')
        wandb = False

    # Try loading wandb if wandb=True. If not found, turn off wandb
    try:
        importlib.import_module("wandb")
    except ImportError:
        if wandb:
            warnings.warn(f"wandb package was not imported. Turning off wandb")
            wandb = False

    if debug and N_epochs_session is None:
        N_epochs_session = 10

    # Ensure directories are absolute and Path objects
    configs_dir = Path(configs_dir).resolve()
    proc_datasets_dir = Path(proc_datasets_dir).resolve()
    gen_datasets_dir = Path(gen_datasets_dir).resolve()
    weights_dir = Path(weights_dir).resolve()
    plots_dir = Path(plots_dir).resolve()

    # Create the directories, if they don't exist
    configs_dir.mkdir(parents=True, exist_ok=True)
    proc_datasets_dir.mkdir(parents=True, exist_ok=True)
    gen_datasets_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    if wandb_logs_dir:
        wandb_logs_dir = Path(wandb_logs_dir).resolve()
        wandb_logs_dir.mkdir(parents=True, exist_ok=True)

    # Define the default device
    if device == 'default':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            # device_default = 'mps'
            device = 'cpu'  # Disable mps for now to avoid `commit command buffer with uncommitted encoder' bug
        else:
            device = 'cpu'

    # Set the default float precision
    if device == 'mps' and float_precision != torch.float32:
        warnings.warn(f"Setting precision to {torch.float32} since device='mps'")
        float_precision = torch.float32
    torch.set_default_dtype(float_precision)

    # Define default activation epoch of training conditioners
    if aux_losses_act_epoch is None or len(aux_losses_act_epoch) == 0:
        aux_losses_act_epoch = len(aux_losses) * [0]

    # If a batch_size schedule is used, make sure the given dict is ordered in ascending epoch.
    if batch_size_sch is not None:
        if 0 not in batch_size_sch:
            warnings.warn(f"Initial batch size is undefined in schedule. Assuming batch_size={batch_size} at epoch=0")
            batch_size_sch[0] = batch_size
        batch_size_sch = dict(sorted(batch_size_sch.items()))

    # Set default value of minibatch_size
    if minibatch_size is None:
        minibatch_size = batch_size

    # Set the default values of sampling_cond_sig_deact
    sampling_cond_sig_deact = {k: 0.0 for k in conditioners} | sampling_cond_sig_deact

    # Turn off dataset_load_all if workers are used
    if N_workers > 0 and dataset_load_all:
        warnings.warn(f"Turning off 'dataset_load_all' since N_workers={N_workers} > 0.")
        dataset_load_all = False

    validate_configs()


def get_configs_dict():
    config_module = sys.modules[__name__]
    return {config_name: getattr(config_module, config_name) for config_name in default_configs}


def import_configs(configs_filename='', overwritten_configs=None):
    """
    Overwrites configurations from input file and/or dictionary
    Args:
        configs_filename: filename of the .py configs file
        overwritten_configs: dictionary of configs

    Returns:
        dictionary of new configs
    """
    # Load the configs file (if any)
    if configs_filename:
        # Ensure the config file is a .py file
        config_filepath = configs_dir / configs_filename
        if config_filepath.suffix == '':
            config_filepath = config_filepath.with_suffix('.py')
        elif config_filepath.suffix != '.py':
            raise ValueError('Filename must be a .py file')

        # Ensure the file exists
        if not config_filepath.exists():
            raise FileNotFoundError(f'The configuration file {config_filepath.name} does not exist in {configs_dir}')

        # Import python config module dynamically with its filename
        spec = importlib.util.spec_from_file_location('user.config', config_filepath)
        imported_configs_module = importlib.util.module_from_spec(spec)
        sys.modules["user.config"] = imported_configs_module
        spec.loader.exec_module(imported_configs_module)

        # Verify that all imported configs are defined and keep only config variables in the imported module.
        undefined_configs_name = []
        imported_configs_name = []
        for config_name, config_val in imported_configs_module.__dict__.items():
            skip_config = isinstance(config_val, ModuleType | FunctionType) or config_name.startswith('__')
            skip_config |= config_name in local_vars_begin
            if skip_config:
                continue
            elif config_name not in default_configs:
                undefined_configs_name.append(config_name)
            else:
                imported_configs_name.append(config_name)

        if undefined_configs_name:
            raise ValueError(f"The following imported configs are undefined:\n{undefined_configs_name!r}\n"
                             f"Use 'del var' to delete all non-config variables at the end of config module.")

        imported_configs_dict = copy.deepcopy(default_configs)
        for config_name in imported_configs_name:
            imported_configs_dict[config_name] = getattr(imported_configs_module, config_name)
    else:
        imported_configs_dict = dict()

    config_module = sys.modules[__name__]

    # Overwrite filename if a configs file was imported
    if configs_filename:
        config_module.filename = configs_filename

    # Overwrite configs given in dictionary
    if overwritten_configs is None:
        overwritten_configs = dict()

    if overwritten_configs:
        # Verify that all overwritten configs are defined.
        for config_name in overwritten_configs:
            if config_name not in default_configs:
                raise ValueError(f"The overwritten config '{config_name}' is not defined.")

        # Overwrite the imported configs with the configs given in input
        imported_configs_dict.update(overwritten_configs)

    if imported_configs_dict:
        # Set the new configs value
        for config_name, config_val in imported_configs_dict.items():
            setattr(config_module, config_name, config_val)

        # Run post-processing
        post_init_process()

        # Reload all models modules that are currently loaded since some classes default kwargs depend on config module
        models_modules = {k: v for k, v in sys.modules.items() if k.startswith('models')}
        for k, v in models_modules.items():
            importlib.reload(v)

    # Format the final configs in a dict and return
    configs_dict = get_configs_dict()
    return configs_dict


def compare_configs(filename1, filename2=''):
    """
    Compares two sets of configs and print their similarities and differences
    Args:
        filename1: filename of first config
        filename2: filename of second config
    """
    configs1 = import_configs(filename1)
    if filename2:
        configs2 = import_configs(filename2)
    else:
        configs2 = default_configs

    def isequal(x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            return x.isclose(y).all()
        elif isinstance(x, dict) and isinstance(y, dict):
            x_keys, y_keys = x.keys(), y.keys()
            return x_keys == y_keys and all([isequal(x[k], y[k]) for k in x_keys])
        elif isinstance(x, torch.nn.Module) and isinstance(y, torch.nn.Module):
            return isinstance(x, y.__class__)
        else:
            return x == y

    similar_configs = {k: v for k, v in configs1.items() if isequal(v, configs2[k])}
    print(f'Similar configurations between {filename1} and {filename2}:\n')
    for name, val in similar_configs.items():
        print(f'{name}: {val}')

    omitted = []
    different_configs = {k: (v, configs2[k]) for k, v in configs1.items() if
                         not isequal(v, configs2[k]) and k not in omitted}
    print(f'\nDifferent configurations (name: {filename1} | {filename2}):\n')
    for name, val in different_configs.items():
        print(f'{name}: {val[0]} | {val[1]}')


def find_configs(**kwargs):
    """
    Finds all configs that match the input keyword arguments
    Args:
        **kwargs: configs and configuration values
    Returns:

    """
    list(configs_dir.glob('config.*'))
    configs_filenames = sorted(list(p.name for p in configs_dir.glob('config*')))
    matching_configs = []
    for config_filename in configs_filenames:
        config_temp = import_configs(config_filename)
        is_config_match = all([config_temp[k] == v for k, v in kwargs.items()])
        if is_config_match:
            matching_configs.append(config_temp)

    # Print all the IDs that match the configs
    print('Configs with the following configurations:')
    for name, val in kwargs.items():
        print(f'{name}={val}')
    print('\nID:', ','.join([c['ID'] for c in matching_configs]))
    return matching_configs


def extract_config_description():
    """Uses inspect to get  inline comments of each defined config."""
    source_code = inspect.getsource(sys.modules[__name__])
    tokens = tokenize.generate_tokens(io.StringIO(source_code).readline)

    config_desc = {}
    current_var = None

    for tok in tokens:
        # Capture the first variable name on a line
        if tok.type == tokenize.NAME and current_var is None:
            current_var = tok.string
        # Capture the trailing comment on the same line
        elif tok.type == tokenize.COMMENT:
            if current_var:
                config_desc[current_var] = tok.string.lstrip("# ").strip()
        # Reset variable tracking when moving to a new line
        elif tok.type in (tokenize.NEWLINE, tokenize.NL):
            current_var = None

    return config_desc


def get_config_parser():
    parser = argparse.ArgumentParser(description="Global Configs Parser")

    # Extract the comments via inspect and tokenize
    config_desc = extract_config_description()

    def str2bool(s: str):
        """
        Convert various strings with boolean meaning to bool
        Args:
            s: string

        Returns:
            bool
        """
        return s.lower() in ("y", "yes", "true", "t", "1")

    def prec_parser(prec: str):
        choices = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32,
                   'float64': torch.float64}
        return choices[prec]

    # Add configs to argparse
    for name, value in default_configs.items():
        help_text = config_desc.get(name, f"No description")

        if isinstance(value, bool):
            parser.add_argument(f"--{name}", type=str2bool, default=value, help=help_text)
        elif isinstance(value, dict):
            parser.add_argument(f"--{name}", type=json.loads, default=value, help=help_text)
        elif isinstance(value, list):
            parser.add_argument(f"--{name}", nargs='*', default=value, help=help_text)
        elif name in ['sampling_dtype', 'amp_dtype', 'float_precision']:
            parser.add_argument(f"--{name}", type=prec_parser, default=value, help=help_text)
        elif name in ['N_epochs_session']:
            parser.add_argument(f"--{name}", type=int, default=value, help=help_text)
        else:
            parser.add_argument(f"--{name}", type=type(value), default=value, help=help_text)

    parser.add_argument('--configs', type=str, default='', help='Filename of configurations stored in configs_dir')

    return parser


def is_config_outdated(date: datetime.datetime):
    from utils import get_file_creation_date
    global filename
    config_filepath = configs_dir / filename
    if not config_filepath.exists():
        return False
    return get_file_creation_date(config_filepath) > date


def download_wandb_run_code(run_ID: str):
    import wandb
    api = wandb.Api()
    run = api.run(f"{project_name}/{run_ID}")

    saved_directory = Path(Path.home(), 'Downloads', 'wandb', 'artifacts', run_ID)
    for art in run.logged_artifacts():
        root_dir = saved_directory / art.version
        art.download(root=str(root_dir))
    print(f'Run code saved in {saved_directory}')


def update_wandb():
    import wandb
    api = wandb.Api()
    filters = {"$and": [{"createdAt": {"$gte": "2026-07-29T00:00:00Z"}},
                        {"createdAt": {"$lte": "2026-08-22T23:59:59Z"}}]}
    runs = list(api.runs(f"{project_name}", filters=filters))
    for run in runs:
        update = False
        if run.config['ID'] == 'HA1':
            run.config['ID'] = 'HA1B'
            update = True
        if run.config['ID'] == 'CS1':
            run.config['ID'] = 'CS1B'
            update = True

        if update:
            run.update()
            print(f"run {run.id!r} updated.")


def generate_run_ID(length: int = 8):
    """
    Generates a random run ID. If wandb is found, run_ID is generated such that it does not match any previous run IDs for the given WandB project.
    Args:
        length: length of run_ID string
    Returns:
        string representing the ID
    """
    try:
        import wandb as wandb_mod
        from wandb_mod.sdk.lib.runid import generate_id
    except ImportError:
        wandb_mod = None

    global wandb  # Import config value
    if not wandb or wandb_mod is None:
        run_ID = uuid.uuid4().hex[:length]
        return run_ID

    # Check that the run ID does not collide with previous runs
    api = wandb_mod.Api()
    N_trials_max = 100
    for _ in range(N_trials_max):
        run_ID = generate_id(length)
        try:
            api.run(f"{project_name}/{run_ID}")
        except Exception:
            return run_ID
    raise ValueError(f"Could not find a run ID that does not collide with previous IDs. N_trials={N_trials_max}.")


def initialize():
    # Modify defaults with values defined in 'config_defaults.py' (if found)
    global default_configs
    defauls_config_filepath = configs_dir / 'config_defaults.py'
    if defauls_config_filepath.exists():
        default_configs |= import_configs(defauls_config_filepath.name)
    post_init_process()


# Run initialize() the first time the module is loaded
initialize()

if __name__ == '__main__':
    # download_wandb_run_code('52wx')

    # Update wandb configs
    # update_wandb()

    # Find configs with certain config values
    # find_configs(weight_decay=0.0)

    # compare_configs('config_HA_193', 'config_HA_196')
    compare_configs('config_CS_206', 'config_CS1')
    # compare_configs('config_HA_193', 'config_HA_193_debug')
    # configs_logged = logged_configs()
    # print_configs()
