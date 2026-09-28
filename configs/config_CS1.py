import os
import re
import torch

# Training
ID = re.findall(r'(?<=_).+(?=\.py)', os.path.basename(__file__))[0]  # ID of the training run
wandb = True  # Log various epoch stats with wandb
optimizer = 'AdamW'  # Name of the optimizer. Can be ['Adam','AdamW','AdamWScheduleFree','AdEMAMix','RAdam']
learning_rate = 1e-4  # Learning rate
n_minibatches = 1  # Number of minibatches used to accumulate gradients before optimization step
lr_piecewise_params = {'epochs': [0, 1000, 2500, 3000, 3500], 'lrs': [1e-4, 1e-3, 1e-3, 1e-4, 1e-5],
                       'funcs': ['poly1', 'poly1', 'exp',
                                 'exp']}  # Dictionary defining piecewise LR parameters. Must have 'epochs', 'lrs' and 'funcs' keys. Overrides learning_rate
weight_decay = 0  # Weight decay parameter. Used when optimizer='AdamW'.
batch_size = 64  # Number of dataset samples used to evaluate gradients of each optimization step
n_batches_per_epoch = 32  # Number of training batches in 1 epoch of training.
drop_last_batch = True  # Drop last batch if it doesn't have enough samples
N_epochs = 4000  # Number of training epochs test
N_epochs_session = None  # Number of training epochs to run in the given session. Run all training epochs by default.
adam_betas = (0.9, 0.9)  # Beta parameters of the Adam optimizer
amsgrad = False  # AMSGRAD option for adam optimizer
resume = True  # Resume training
N_workers = 0  # Number of workers to load the data onto the device
seed = 1  # Random seed used by torch, numpy and random
device = 'default'  # Device used for training. Can be ['cpu', 'mps', 'cuda', 'default']. See below for default device.
pin_memory = False  # Pin data to device memory in data loader
feats_scale = {'pos': 0.1}  # Scale of model features
weighted_t_sampling = False  # weight the sampling of the diffusion time for evaluating the loss with MCMC
importance_sampling_logSNR = False  # Weighs each batch sample by 1/p(logSNR) where p(logSNR) is the pdf of the sampled logSNR
model_ema_beta = 0.999  # beta parameter to calculate a model EMA during training. The averaged model is used for eval. Default is no EMA.
aux_losses = []  # Name of auxiliary losses used during training
amp = False  # Use Automatic Mixed Precision during training
amp_dtype = torch.float16  # Precision used with amp
compile_model = True  # Compile model with torch.compile()

# Sampling
n_samples = 100  # Number of samples generated at each sampling period
sampling_period = 50  # Period (epochs) at which samples are generated and analyzed during training
conditioners = []  # List of conditioners used during sampling
sampling_cond_bounds_sig = 5.0  # Sigma used to define the bounds of conditioners (when applicable)
sampling_ref_dataset = 'valid'  # Reference dataset used for generating conditioned samples
correlated_diffusion = None  # Use a correlated diffusion process to encore scaling laws in the prior distribution
correlated_diffusion_scale = True  # Scale the correlated noise in the correlated diffuser
SDE_sampler = True  # Sample with the SDE. Otherwise, sample with the probabilistic flow ODE
SDE_sampler_params = {'psi': 10.0, 'lambda0': 1.0}  # Parameters of the SDE sampler

# Dataset
dataset = None  # Name of the dataset
dataset_element_class = 'c_alpha_sidechain'  # Element class of the loaded dataset
dataset_split_set_ID = '1'  # Default split set ID used when loading splits
dataset_load_all = True  # Load all molecules prior to training
coarse_grains_dataset = 'Rich_2018'  # Dataset used to define the coarse grains center and radius. Uses all atoms

# Architecture
diff_model = 'EDM'  # Model used to perform the diffusion steps (loss, sampling, etc.)
model_bias = False  # Add bias in model layers
model_output_pos_diff = True  # Output the node position difference as opposed to the absolute value
edge_features = ['d_0', 'd_0^-1']  # Edge features used in the message and position network (phi_msg and phi_x)
cond_edge_features = ['seq_pos_diff_ter']  # Edge features used for conditioning the model
global_features = None  # Global features used in the message and position network (phi_msg and phi_x).
cond_node_features = ['type', 'residue_type', 'sequence_type']  # Types of variables used for conditioning the model
t_as_input_feature = True  # Concatenate time to the input node features before processing them through the layers.
n_layers = 10  # Number of layers (or iterations) in the EGNN network.
pos_msg_agg = 'sum_unnorm'  # Method to aggregate messages for the node position update. Can be ['sum','mean','sum_unnorm']
feat_msg_agg = 'sum'  # Method to aggregate messages for the node features update
epsilon_x = 1e-4  # Stabilization parameter used in the position update aggregation (x_i-x_j)/(d_ij + epsilon_x)
epsilon_d_inv = 5e-1  # Epsilon parameter used in defining inverse distance features d^-1 = 1/(d + epsilon)
phi_x_weight = True  # Multiply the output of phi_x in layers with a learnable weight
phi_weight_act = 'local_softmax'  # Function used at the end of the phi_msg and phi_x weights layers.
node_latent_dim = 64  # Dimension of the latent node feature space
edge_latent_dim = 64  # Dimension of the latent edge feature space
n_diff_steps = 1000  # Number of diffusion steps (also denoted as "T" in reference)
n_diff_steps_monitoring = 100  # Number of diffusion steps used to produce monitoring samples during training
cross_prod_layer = 'ECPL'  # Type of cross-product layer that updates node positions with cross-products of triplets of elements
cross_prod_layer_hidden_d = node_latent_dim  # Hidden dimension of the cross-product layer phi module
cross_prod_layer_phi_weight_act = 'local_softmax'  # Activation function used in the phi_weight module of cross-product layers
coarse_triplet_ind_method = '2'  # Method used to define node triplets used in cross product modules
cross_prod_use_edge_feat = False  # Use edge features in cross product layer
n_cross_prod_layers = 10  # Number of cross-product layers used
noise_sch_type = 'logSNR-linear'  # Type of the noise schedule. Can be ['polynomial','cosine','sigmoid','half-sigmoid','exp']
noise_sch_params = dict(SNR_min=(1 / 50) ** 2, SNR_max=(1 / 0.1) ** 2)  # Parameter for the noise schedule
node_pos_res = True  # Add residual connection for the node position update
model_progressive = False  # Progressively activate layers of the model during training.
omega_step = 2e-4  # Step size of the progression when model_progressive is True
omega_max = 2.0  # Maximum value of omega that triggers a layer growth
norm_layer = 'layernorm'  # Normalization layers used after each activation. Can be ['layernorm',None]. Default=None.
norm_layer_bias = False  # Add bias in normalization layers.
enc_dec_bias = True  # Add bias in the encoder and decoder
node_adj_range = torch.inf  # Increase the coarse grain adjacency distance threshold (nm) which determines if two coarse grains are adjacent in the graph topology.
node_adj_method = None  # Method used to determine the nodes adjacency matrix
node_feat_agg_in_edge_updt = 'sum'  # Aggregation method of node features in the edge feature update.
remove_noise_COM = True  # Remove the centroid (or center-of-mass) of the sampled noise applied to each structure
sig_pos_in = None  # Input position sigma (nm) used in EDM diffusion model
