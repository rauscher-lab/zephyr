import copy
import dataclasses
import itertools
import pathlib
import time
import warnings
from typing import Self

import numpy as np
import torch
import tqdm
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from utils import symmetrize, remove_centroid, set_rng, Archiver, Timer, RigidBodyMotion
from chemistry.energy import BondEnergy
from chemistry.chem_utils import define_node_triplets
from data.data_classes import Biomolecule, heavy_atoms_class_name, all_atoms_class_name
from data.datasets import BiomoleculeDataset, ProcessedDataset, DenoisingTrajectoriesDataset, load_dataset
from models.losses import AuxiliaryLoss
from models.noise_schedules import init_noise_schedule
from models.diffusers import CorrelatedNoiser, DiffusionModel, VP, EDM, EDM2, VP2, VE, VE2
from models.layers import MessagePassingLayer, EGCL, EquiCrossProdLayer, Gain, RotationLayer
from models.features import gather_input_tensors, assemble_features
from models.conditioners import CoarsePosConstraintConditioner, CoarsePositionConditioner, \
    CoarseCenterDistConditioner, variable_conditioners, Conditioner
from analysis.stats import StatisticsArray


class ProgGNN(nn.Module):
    """
    Progressive Graph Neural Network (PGNN) that progressively adds layers onto the network during training.
    """

    def __init__(self, layers: list[MessagePassingLayer], progressive=config.model_progressive,
                 node_pos_in_d=3, node_feat_in_d=1, edge_feat_in_d=0, global_feat_in_d=0,
                 node_encoder: nn.Module = None, node_decoder: nn.Module = None,
                 edge_encoder: nn.Module = None, edge_decoder: nn.Module = None,
                 global_encoder: nn.Module = None, global_decoder: nn.Module = None):
        super().__init__()
        self.node_pos_in_d = node_pos_in_d
        self.node_feat_in_d = node_feat_in_d
        self.edge_feat_in_d = edge_feat_in_d
        self.global_feat_in_d = global_feat_in_d
        self.layers = layers
        self.progressive = progressive

        self.node_encoder = (lambda x: x) if node_encoder is None else node_encoder
        self.node_decoder = (lambda x: x) if node_decoder is None else node_decoder
        self.edge_encoder = (lambda x: x) if edge_encoder is None else edge_encoder
        self.edge_decoder = (lambda x: x) if edge_decoder is None else edge_decoder
        self.global_encoder = (lambda x: x) if global_encoder is None else global_encoder
        self.global_decoder = (lambda x: x) if global_decoder is None else global_decoder

        if self.progressive:
            self.n_layers = nn.parameter.Buffer(torch.tensor(1), persistent=True)
            self.n_layers_max = nn.parameter.Buffer(torch.tensor(len(self.layers)), persistent=True)
            self.omega = nn.parameter.Buffer(torch.tensor(1.0), persistent=True)
            self.omega_max = nn.parameter.Buffer(torch.tensor(config.omega_max), persistent=True)
            self.is_built = nn.parameter.Buffer(torch.tensor(False), persistent=True)
        else:
            self.n_layers = nn.parameter.Buffer(torch.tensor(len(self.layers)), persistent=True)
            self.is_built = nn.parameter.Buffer(torch.tensor(True), persistent=True)

    def forward(self, node_pos: torch.Tensor = None, node_feat: torch.Tensor = None,
                edge_feat: torch.Tensor = None, global_feat: torch.Tensor = None,
                edges_ind: torch.Tensor = None, nodes_adj: torch.Tensor = None,
                extra_node_feat: torch.Tensor = None, extra_edge_feat: torch.Tensor = None,
                extra_global_feat: torch.Tensor = None, **kwargs):
        """
        Predicts the noise on the node positions and features
        Args:
            node_pos: node 3D position. shape=(N,3) or (B,N,3), B=batch size, N=number of nodes
            edge_feat: edge features. shape=(E,d_e), E=number of edges, d_e=edge feature dimension
            global_feat: global features. shape=(B,d_g), d_g=dimension of global features
            node_feat: node features. shape=(N,D) or (B,N,D),  D=node feature space dimension
            edges_ind: global edges index. shape=(N_edges,3) where index 0=batch dimension and index 1,2=node indices
            nodes_adj: nodes adjacency matrix. shape=(N,N) or (B,N,N)
        Returns:
            epsilon: noise prediction on the nodes' position and features. shape=(N,3+D) or (B,N,3+D)
        """

        # Encoders
        node_feat = self.node_encoder(node_feat)
        edge_feat = self.edge_encoder(edge_feat)
        global_feat = self.global_encoder(global_feat)

        # Layers
        constant_layer_kwargs = dict(extra_node_feat=extra_node_feat, extra_edge_feat=extra_edge_feat,
                                     extra_global_feat=extra_global_feat, nodes_adj=nodes_adj, edges_ind=edges_ind)
        for i in range(self.n_layers):
            # When progressing, detach the input of the last layer to make sure the state of previous layers is not
            # disturbed by the untrained state of the last layer
            if self.progressive and self.omega < 1 and i == self.n_layers - 1:
                node_pos = node_pos.detach() if node_pos is not None else node_pos
                node_feat = node_feat.detach() if node_feat is not None else node_feat
                edge_feat = edge_feat.detach() if edge_feat is not None else edge_feat
                global_feat = global_feat.detach() if global_feat is not None else global_feat

            node_pos, node_feat, edge_feat, global_feat = self.layers[i](node_pos=node_pos, node_feat=node_feat,
                                                                         edge_feat=edge_feat, global_feat=global_feat,
                                                                         **constant_layer_kwargs, **kwargs)

            node_pos, node_feat, edge_feat, global_feat = self.apply_extra_hidden_layers(i, node_pos=node_pos,
                                                                                         node_feat=node_feat,
                                                                                         edge_feat=edge_feat,
                                                                                         global_feat=global_feat,
                                                                                         **constant_layer_kwargs,
                                                                                         **kwargs)
            # Save penultimate model output if the model is progressing
            if self.progressive and self.omega < 1 and i == self.n_layers - 2:
                node_pos_prev, node_feat_prev = node_pos, node_feat
                edge_feat_prev, global_feat_prev = edge_feat, global_feat

        # Interpolate the last two layers when the network is in progression
        if self.progressive and self.omega < 1:
            if node_pos is not None:
                node_pos = torch.lerp(node_pos_prev, end=node_pos, weight=self.omega)
            if node_feat is not None:
                node_feat = torch.lerp(node_feat_prev, end=node_feat, weight=self.omega)
            if edge_feat is not None:
                edge_feat = torch.lerp(edge_feat_prev, end=edge_feat, weight=self.omega)
            if global_feat is not None:
                global_feat = torch.lerp(global_feat_prev, end=global_feat, weight=self.omega)

        # Decoders
        node_feat = self.node_decoder(node_feat)
        edge_feat = self.edge_decoder(edge_feat)
        global_feat = self.global_decoder(global_feat)

        return node_pos, node_feat, edge_feat, global_feat

    def apply_extra_hidden_layers(self, i, node_pos: torch.Tensor, node_feat: torch.Tensor, edge_feat: torch.Tensor,
                                  global_feat: torch.Tensor, **kwargs):
        return node_pos, node_feat, edge_feat, global_feat

    def start_transition(self):
        # Ensure that there is no ongoing transition.
        assert self.omega >= 1, 'The previous transition has not ended.'

        # Start a resolution transition.
        self.omega *= 0.0
        self.n_layers += 1

        # Check that the image size has not exceeded the maximum.
        assert self.n_layers <= self.n_layers_max, (
            f'The number of layers must be <= {self.n_layers_max}. n_layers = {self.n_layers}')

    def advance_transition(self, omega_step=0.0001, verbose=True):
        # Increase alpha by the given step to advance the resolution transition
        if not self.is_built:
            self.omega += omega_step
            self.is_built.data = self.n_layers == self.n_layers_max and self.omega >= 1
            if self.omega >= self.omega_max - omega_step / 2:
                if verbose:
                    n_layers_new = self.n_layers + 1
                    print(f'Starting model transition:\nomega={self.omega.item()}, n_layers_new={n_layers_new}')

                # Start a transition
                self.start_transition()

    def parameters_train(self):
        """
        Gathers all trainable parameters
        Returns: iterator of trainable parameters
        """
        return (p for p in self.parameters() if p.requires_grad)

    def EMA_update(self, model_new: Self, beta: float):
        if self.progressive:
            self.n_layers = copy.copy(model_new.n_layers)
            self.omega.data = model_new.omega.data.clone()
        if self.is_built:
            weight = 1.0 - beta
        else:
            weight = 1.0  # Do not use ema when network is still progressing
        self_params = [p.data for p in self.parameters_train()]
        new_params = [p.data for p in model_new.parameters_train()]
        torch._foreach_lerp_(self_params, new_params, weight=weight)

    def gen_rand_inputs(self, B=5, N=20):
        nodes_adj = torch.triu(symmetrize(torch.rand((B, N, N), device=self.device)) < 0.5, diagonal=1)
        edges_ind = torch.nonzero(nodes_adj)
        node_pos = torch.randn((B, N, self.node_pos_in_d), device=self.device) if self.node_pos_in_d > 0 else None
        node_feat = torch.randn((B, N, self.node_feat_in_d), device=self.device)
        edge_feat = torch.randn((edges_ind.shape[0], self.edge_feat_in_d),
                                device=self.device) if self.edge_feat_in_d > 0 else None
        global_feat = torch.randn((B, self.global_feat_in_d), device=self.device) if self.global_feat_in_d > 0 else None
        inputs = dict(node_pos=node_pos, node_feat=node_feat, edge_feat=edge_feat, global_feat=global_feat,
                      edges_ind=edges_ind, nodes_adj=nodes_adj)
        return inputs

    def is_equivariant(self, tol=1e-5, operator=None):
        """
        Tests for SE(3) equivariance of a given operator
        Args:
            tol: numerical tolerance used to evaluate the differences between the model output and its rotated output.
            operator: operator tested for equivariance. Defaults to self.

        Returns:
            Bool indicating whether the module is equivariant or not
        """
        # Create some random input position and features
        if operator is None:
            operator = self
        if operator == self and self.node_pos_in_d == 0:
            return torch.nan

        # Create some random input position and features
        B, N = 10, 80
        rand_inputs = self.gen_rand_inputs(B=B, N=N)
        node_pos_out, _, _, _ = operator(**rand_inputs)

        # Generate random rigid-body motions (trans+rot)
        rand_RBM = RigidBodyMotion.random(size=(B, 1), device=self.device)

        # Rotate the input positions and revaluate the model's output
        rand_inputs['node_pos'] = rand_RBM.transform(rand_inputs['node_pos'])
        node_pos_out_trans, _, _, _ = operator(**rand_inputs)

        # Compare the output of the non-rotated input with the non-rotated output of the rotated input
        node_pos_trans_after = rand_RBM.transform(node_pos_out)
        rotated_pos_diff = node_pos_trans_after - node_pos_out_trans
        max_scale = torch.cat([node_pos_out_trans, node_pos_trans_after], dim=0).max()
        diff_max = rotated_pos_diff.abs().max() / max_scale
        is_equiv = diff_max < tol
        if not is_equiv:
            warnings.warn(f'{self.__class__.__name__} is not SE(3) equivariant.\nPos diff max>tol:{diff_max}>{tol}')

        return is_equiv

    @property
    def n_params(self):
        return sum([p.numel() for p in self.parameters_train() if p.requires_grad])

    @property
    def device(self):
        return next(self.parameters()).device

    def gather_input_tensors(self, biomolecules: list[Biomolecule], **kwargs) -> dict[str, torch.Tensor]:
        """
        Gathers tensors needed for the model's input from a list of biomolecules.
        Args:
            biomolecules: list of biomolecules from which tensors are calculated

        Returns: dictionary of tensors

        """
        raise NotImplementedError

    def gather_constant_input_tensors(self, biomolecule: Biomolecule, **kwargs) -> dict[str, torch.Tensor]:
        """
        Gathers constant tensors needed for the model's input from a given molecule
        Args:
            biomolecule: example biomolecule

        Returns: dictionary of constant tensors

        """
        raise NotImplementedError

    def plot_params_dist(self, tensorboard_dir: str | pathlib.Path = None, ID=None):
        if ID is None:
            ID = config.ID
        if tensorboard_dir is None:
            tensorboard_dir = pathlib.Path(config.plots_dir, 'model parameters distributions')
        summary_dir = pathlib.Path(tensorboard_dir, ID)

        writer = SummaryWriter(str(summary_dir))
        params_iter = (p for p in self.named_parameters() if p[1].requires_grad)
        for p in params_iter:
            if 'layers' in p[0]:
                if p[0].endswith('bias'):
                    tag = f'layers_bias_parameters/'
                else:
                    tag = f'layers_weights_parameters/'
            else:
                tag = 'other_parameters/'
            tag += p[0]
            writer.add_histogram(tag=tag, values=p[1])
        writer.close()

    def plot_compute_graph(self, tensorboard_dir: str | pathlib.Path = None):
        if tensorboard_dir is None:
            tensorboard_dir = pathlib.Path(config.plots_dir, 'model parameters distributions', config.ID)
        writer = SummaryWriter(str(tensorboard_dir))
        kwargs = self.gen_rand_inputs(B=1)
        node_pos = kwargs.pop('node_pos')

        class Wrapper(nn.Module):
            def __init__(self, model: nn.Module):
                super().__init__()
                self.model = model

            def forward(self2, inputs):
                output = tuple([o for o in self2.model(inputs, **kwargs) if isinstance(o, torch.Tensor)])
                return output

        writer.add_graph(Wrapper(model=self), node_pos, use_strict_trace=False)
        writer.close()

    def replace_Linear_with_Rotation(self, module=None):
        module = self if module is None else module
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and child.in_features == child.out_features:
                rotation_layer = RotationLayer(d=child.in_features, bias=child.bias is not None,
                                               device=child.weight.device, dtype=child.weight.dtype)
                setattr(module, name, rotation_layer)
            else:
                self.replace_Linear_with_Rotation(child)  # If child is a module container, recurse


class DiffPGNN(ProgGNN):
    def __init__(self, diff_model: DiffusionModel, pos_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.diff_model = diff_model
        self.diff_model.denoiser = self.denoise  # Overwrite the denoiser attribute of diff_model to use self method
        self.pos_scale = nn.parameter.Buffer(torch.as_tensor(pos_scale), persistent=True)

    def format_denoiser_input(self, tensors_batch: dict[str, torch.Tensor]) -> (torch.Tensor, dict[str, torch.Tensor]):
        """
        Parses a batch of tensors to format inputs to the denoiser
        Args:
            tensors_batch: dictionary of tensors

        Returns:
                noised_properties, denoiser_kwargs
        """
        raise NotImplementedError

    def retrieve_nodes_pos_from_denoiser_output(self, denoiser_output) -> torch.Tensor:
        raise NotImplementedError

    def denoise(self, noisy_props: torch.Tensor, t_embedding: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Denoises noisy input properties
        Args:
            noisy_props: noisy properties
            t_embedding: embedding of diffusion time
        Returns:
            denoised_props with same shape as noisy_props
        """
        raise NotImplementedError

    def loss(self, tensors_batch: dict[str, torch.Tensor], normalize=True, sum_batch=True) \
            -> (torch.Tensor, dict[str, torch.Tensor]):
        """
        Computes diffusion and auxiliary losses for a given a batch of tensors.
        Args:
            tensors_batch: batch of tensors
            normalize: normalize the loss by the total number of elements in the batch.
            sum_batch: sum loss over batch. If False, returned loss is a 1D tensor giving the loss of each batch element
        Returns:
            loss_tot, loss_dict (ensures values() are detached())
        """
        nodes_pos, denoiser_kwargs = self.format_denoiser_input(tensors_batch=tensors_batch)
        loss_pos, loss_dict, denoised_props, loss_weights, t, _ = self.diff_model.loss(nodes_pos,
                                                                                       normalize=normalize,
                                                                                       **denoiser_kwargs)

        # Check for nans
        loss_pos_is_nan = loss_pos.isnan()
        n_loss_pos_nans = loss_pos_is_nan.sum()
        if n_loss_pos_nans > 0:
            warnings.warn(f"Found {n_loss_pos_nans} nans in position loss of {self.__class__.__name__}!")
        loss_dict['pos'] = loss_pos

        # If losses were not normalized inside the diffusion model, normalize at least by the batch size.
        if not normalize:
            batch_size = loss_pos.shape[0]
            loss_dict = {k: v / batch_size for k, v in loss_dict.items()}

        if sum_batch:
            loss_dict = {k: v.sum() for k, v in loss_dict.items()}

        loss_tot = sum(loss_dict.values())

        # Detach the loss values in the return dict
        loss_dict = {k: v.detach() for k, v in loss_dict.items()}

        return loss_tot, loss_dict

    def sample(self, size: int | tuple, ref_dataset: BiomoleculeDataset, T=None, seed=None, verbose=False,
               batch_size=100, conditioners: list[str] = config.conditioners,
               temp_dir='', timer: Timer = None, save_trajectory=False, save_cond_U=False):
        """
        Samples a set of molecules with the diffusion GNN
        Args:
            size: shape of the output tensor
            ref_dataset: Reference dataset used to determine the type of molecules to sample
            T: number of diffusion steps used for sampling
            seed: sets the random seed to the given value if not None
            verbose: outputs sampling progress in terminal
            batch_size: size of a batch of samples
            conditioners: list of extra conditioners used during sampling
            temp_dir: temporary directory where completed batches are saved.
            timer: Timer object used to time the sampling procedure
            save_trajectory: saves the entire denoising trajectory of the produced samples
            save_cond_U: saves the conditioners' energy of each sample at each time denoising step
        Returns:
            torch.Tensor of shape 'size'
        """
        # Start timer if any
        if timer:
            timer.start()

        # Determine the number of samples
        if isinstance(size, int):
            size = torch.Size((size,))
        else:
            size = torch.Size(size)
        n_samples = size.numel()
        large_sample = n_samples > 200

        # Initialize the archiver
        arch = Archiver(dir=temp_dir) if temp_dir else None

        # If the archive is not empty, use the batch size that was used previously to preserve the molecule index
        if len(arch) > 1:
            batch_size = len(arch.load(arch.chunk_filenames[0]))
        n_batches, last_batch_n_samples = divmod(n_samples, batch_size)
        batch_n_samples = n_batches * [batch_size]
        if last_batch_n_samples > 0:
            n_batches += 1
            batch_n_samples.append(last_batch_n_samples)

        # Set the random seed.
        if seed is not None:
            set_rng(seed)

        # Get an example of the kind of molecule to generate
        mol_ex = ref_dataset[0]

        # Set the model in eval mode and restore the training mode after sampling
        training_mode = self.training
        self.eval()

        # Define a dataloader to sample fixed properties from the reference dataset
        # If dataset will be loaded inMemory and if n_samples<len(dataset), load only the necessary mols to save time
        dataset_in_mem = config.N_workers == 0 or not large_sample
        if dataset_in_mem and n_samples < len(ref_dataset):
            ref_dataset = ref_dataset.subset(ind=torch.arange(n_samples), inplace=False, inmemory=True)

        ref_dataset_proc = ProcessedDataset(ref_dataset, model=self, device=self.device, inMemory=dataset_in_mem)
        infinite_dataset_iter = itertools.cycle(range(len(ref_dataset_proc)))
        dataset_sampler = (next(infinite_dataset_iter) for _ in range(n_samples))
        ref_ID_func = lambda mol_ind: mol_ind % len(ref_dataset_proc)
        dataloader = DataLoader(ref_dataset_proc, sampler=dataset_sampler, batch_size=batch_size, shuffle=False,
                                num_workers=0 if dataset_in_mem else config.N_workers, collate_fn=lambda x: x)

        # Restart timer if it has resumed from a previous session since preprocessing time should be counted only once
        if timer and timer.resume and timer.has_checkpoint:
            timer.start()

        saved_data = dict()
        for batch_ind, tensors_dict in enumerate(dataloader):
            # Skip previous batches if they have been saved in the archive directory
            if arch is not None and batch_ind < len(arch):
                print(f'Batch {batch_ind + 1}/{n_batches} is already generated. Skipping to next.')
                continue

            # Move tensors to device if using parallel workers
            if not dataset_in_mem:
                ref_dataset_proc.move_tensors_to_device(tensors_dict)
            elements_pos, denoiser_kwargs = self.format_denoiser_input(tensors_dict)

            # Define dataset-dependent sampling conditioners
            conditioners_list = list()
            extra_conditioners = set(conditioners) & set(variable_conditioners)
            if extra_conditioners:
                for conditioner in extra_conditioners:
                    sig_deactivate = config.sampling_cond_sig_deact[conditioner]
                    if conditioner == 'coarse_pos_cons':
                        coarse_pos = tensors_dict['coarse_pos']
                        coarse_atom_ind = tensors_dict['coarse_atom_ind']
                        conditioner = CoarsePosConstraintConditioner(pos_target=coarse_pos, ind=coarse_atom_ind,
                                                                     mask_shape=(elements_pos.shape[1],),
                                                                     sigma_func=self.diff_model.sigma,
                                                                     corr_noiser=self.diff_model.correlated_noiser,
                                                                     sigma_deactivate=sig_deactivate)
                    elif conditioner in ['coarse_pos', 'elements_coarse_pos']:
                        expand_force_to_elements = conditioner == 'elements_coarse_pos'
                        nodes_coarse_ind = tensors_dict['nodes_coarse_ind']
                        coarse_atom_ind = tensors_dict['coarse_atom_ind']
                        coarse_pos = tensors_dict['coarse_pos']

                        # Remove coarse positions that are not associated with any elements. This happens if the sequence
                        # has GLY residues, which has no coarse grain for the sidechain.
                        represented_coarse_ind = nodes_coarse_ind[0, :, 0].unique()
                        new_coarse_ind_map = -1 * torch.ones_like(coarse_atom_ind)[0, :, 0]
                        new_coarse_ind_map[represented_coarse_ind] = torch.arange(represented_coarse_ind.numel()).to(
                            coarse_atom_ind.device)

                        nodes_coarse_ind = new_coarse_ind_map[nodes_coarse_ind]
                        coarse_pos = coarse_pos[:, represented_coarse_ind, :]
                        coarse_atom_ind = coarse_atom_ind[:, represented_coarse_ind, :]
                        conditioner = CoarsePositionConditioner(coarse_pos_target=coarse_pos,
                                                                coarse_atom_ind=coarse_atom_ind,
                                                                nodes_coarse_ind=nodes_coarse_ind,
                                                                sigma_func=self.diff_model.sigma,
                                                                corr_noiser=self.diff_model.correlated_noiser,
                                                                expand_force_to_elements=expand_force_to_elements,
                                                                sigma_deactivate=sig_deactivate)
                    elif conditioner == 'coarse_center_dist':
                        coarse_ind = tensors_dict['nodes_coarse_ind']
                        coarse_pos = tensors_dict['coarse_pos']
                        coarse_rad = tensors_dict['coarse_rad']

                        conditioner = CoarseCenterDistConditioner(coarse_grain_pos=coarse_pos,
                                                                  coarse_grain_radius=coarse_rad,
                                                                  coarse_grain_ind=coarse_ind,
                                                                  sigma_func=self.diff_model.sigma,
                                                                  corr_noiser=self.diff_model.correlated_noiser,
                                                                  precision=config.coarse_pos_precision / self.pos_scale,
                                                                  pos_scale=self.pos_scale,
                                                                  sigma_deactivate=sig_deactivate)
                    else:
                        raise ValueError(f"conditioner={conditioner} is not supported.")
                    conditioners_list.append(conditioner)

            # Determine the initial element positions
            if mol_ex.element_class == heavy_atoms_class_name:
                coarse_pos = tensors_dict['coarse_pos']
                nodes_coarse_ind = tensors_dict['nodes_coarse_ind'].expand(-1, -1, 3)
                elements_pos_init = torch.gather(coarse_pos, dim=1, index=nodes_coarse_ind)
            else:
                elements_pos_init = torch.zeros_like(elements_pos)

            # Generate samples using the diffusion model
            model_output = self.diff_model.sample(positions_init=elements_pos_init, T=T, verbose=verbose,
                                                  conditioners=conditioners_list,
                                                  save_trajectory=save_trajectory, save_cond_U=save_cond_U,
                                                  **denoiser_kwargs)
            if save_trajectory:
                node_pos_traj = model_output[-1]  # shape = (T+1,B,N,3)
                n_frames, B = node_pos_traj.shape[:2]
                nodes_pos = node_pos_traj.flatten(0, 1)  # Flatten time and batch dimension.
            elif save_cond_U:
                conditioners_info = model_output[-1]
                conditioners_U = conditioners_info['energy'].cpu()
                if batch_ind == 0:
                    saved_data['cond_type'] = conditioners_info['type']
                nodes_pos = self.retrieve_nodes_pos_from_denoiser_output(model_output)
            else:
                nodes_pos = self.retrieve_nodes_pos_from_denoiser_output(model_output)

            # Reapply the position scale since the network was trained with scaled properties
            nodes_pos = nodes_pos * self.pos_scale

            # Remove centroid for convenience
            nodes_pos = remove_centroid(nodes_pos, inplace=False)

            # Move positions back to cpu
            nodes_pos = nodes_pos.cpu()

            # Parse the output into the molecule type given in the example
            for i, elements_pos in enumerate(nodes_pos):
                if save_trajectory:
                    dataset_ind = i + batch_ind * n_frames * batch_size
                    mol_ind = (i % B) + batch_ind * batch_size
                    time_ind = i // B
                    mol_ID = DenoisingTrajectoriesDataset.format_mol_ID(prefix=self.__class__.__name__,
                                                                        mol_ind=mol_ind, time_ind=time_ind)
                else:
                    dataset_ind = i + batch_ind * batch_size
                    mol_ind = dataset_ind
                    mol_ID = f"{self.__class__.__name__}_{mol_ind + 1}"
                model_mol = ref_dataset[ref_ID_func(mol_ind)]
                saved_data[f"mol_{dataset_ind}"] = dataclasses.replace(mol_ex, ID=mol_ID, model_ID=model_mol.ID,
                                                                       elements_position=elements_pos.clone(),
                                                                       top_file=None)
                if save_cond_U:
                    saved_data[f"cond_U_{dataset_ind}"] = conditioners_U[:, i, :]

            # Perform checkpoint
            if arch is not None:
                arch.save(**saved_data)
                saved_data.clear()

            if timer:
                timer.save_checkpoint()

        # Restore the training mode
        self.train(training_mode)

        # Gather all molecules
        if arch is not None:
            saved_data = arch.load()
        mol_keys = sorted([k for k in saved_data if k.startswith('mol')], key=lambda x: int(x.removeprefix('mol_')))
        molecules = np.array([saved_data[k] for k in mol_keys], dtype=Biomolecule)
        if not save_trajectory:
            molecules = molecules[:n_samples].reshape(size)

        # Gather conditioners energy timeseries (if requested)
        if save_cond_U:
            cond_U_keys = [k for k in saved_data if k.startswith('cond_U')]
            cond_U_keys = sorted(cond_U_keys, key=lambda x: int(x.removeprefix('cond_U_')))
            conditioners_U = np.array([saved_data[k] for k in cond_U_keys])
            conditioners_type = saved_data['cond_type']
            conditioner_info = dict(energy=conditioners_U, type=conditioners_type)

        # Destroy archive and stop timer
        if arch is not None:
            arch.destroy()

        if timer:
            timer.stop()
            timer.save_checkpoint()

        if save_cond_U:
            return molecules, conditioner_info

        return molecules

    def calculate_loss_stats(self, ref_dataset: BiomoleculeDataset = None, n_samples=1000, seed=None, verbose=True,
                             batch_size=config.batch_size, regen=False, plot=False):
        """
        Calculates statistics of the model loss for each denoising step
        Args:
            ref_dataset: Reference dataset used to determine the type of molecules to sample
            n_samples: number of samples used to estimate the statistics at each denoising step
            seed: sets the random seed to the given value if not None
            verbose: outputs sampling progress in terminal
            batch_size: size of each batch of samples
            regen: regenerate loss stats and overwrite previously saved stats
            plot: plot the loss statistics in the subfolder .../plots/loss_stats/
        Returns:
            torch.Tensor of shape 'size'
        """
        # Determine the number of samples
        size = torch.Size((n_samples,))
        n_samples = size.numel()

        # Set the random seed.
        if seed is not None:
            set_rng(seed)

        # Load the config dataset 'train' split if None were given
        if ref_dataset is None:
            ref_dataset = load_dataset(element_class=config.dataset_element_class)
            ref_dataset = ref_dataset.load_splits('train')

        T = len(self.diff_model.t_steps)

        # Define directory where calculations will be saved
        stats_filepath = pathlib.Path(config.root_dir, 'plots', 'loss_stats', f"{config.ID}_{ref_dataset.name}.npz")
        stats_filepath.parent.mkdir(exist_ok=True)

        if stats_filepath.exists() and not regen:
            with np.load(stats_filepath, allow_pickle=True) as npzfile:
                stats_dict = {k: v.item() for k, v in npzfile.items()}
            loss_stats = stats_dict['loss_stats']
            t_start = torch.nonzero(loss_stats.n.long() == n_samples)[-1].item() + 1
        else:
            loss_stats = StatisticsArray(shape=(T,), d=1, cov_method='diag', dtype=torch.double)
            t_start = 0

        if t_start < T:
            loss_stats.to(self.device)

            # Load all molecules
            ref_dataset.load_all()

            # Define a dataloader to sample fixed properties from the reference dataset
            ref_dataset_proc = ProcessedDataset(ref_dataset, model=self, device=self.device)
            sampler = torch.utils.data.RandomSampler(ref_dataset_proc, replacement=True, num_samples=n_samples)
            batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=False)
            dataloader_ref = DataLoader(ref_dataset_proc, batch_sampler=batch_sampler, num_workers=config.N_workers,
                                        collate_fn=lambda x: x)

            # Set the model in eval mode and restore the training mode after sampling
            training_mode = self.training
            self.eval()

            t_ind_iter = range(t_start, T)
            if verbose:
                t_ind_iter = tqdm.tqdm(t_ind_iter, f'Estimating loss stats of {config.ID!r} model '
                                                   f'with {ref_dataset.name!r} dataset.')

            diff_model_t_sampler = self.diff_model.sample_t
            checkpoint_period = 60
            last_checkpoint_time = time.time()
            for t_ind in t_ind_iter:
                # Fix the t sampler of the diffusion model to sample t deterministically
                def t_sampler(size: int | tuple[int] | torch.Size):
                    if isinstance(size, int):
                        size = (size,)
                    if not isinstance(size, torch.Size):
                        size = torch.Size(size)
                    t_indices = t_ind * torch.ones(size, device=self.device, dtype=torch.long)
                    t = self.diff_model.t_steps[t_indices]
                    return t, t_indices, None

                self.diff_model.sample_t = t_sampler
                tensors_iter = dataloader_ref
                if verbose:
                    tensors_iter = tqdm.tqdm(tensors_iter, f'Estimating loss stats at t={t_ind}')

                for tensors_dict in tensors_iter:
                    with torch.no_grad():
                        losses, _ = self.loss(tensors_batch=tensors_dict, sum_batch=False)

                    # Renormalize each element loss
                    losses = losses * losses.numel()
                    t_ind_arr = t_ind * torch.ones(losses.shape, device=self.device, dtype=torch.long)
                    stats_temp = StatisticsArray(shape=loss_stats.shape, cov_method=loss_stats.cov_method,
                                                 ind=t_ind_arr, values=losses)
                    loss_stats += stats_temp

                # Save updated stats
                curr_time = time.time()
                if curr_time - last_checkpoint_time > checkpoint_period or t_ind == T - 1:
                    np.savez(stats_filepath, loss_stats=loss_stats)
                    last_checkpoint_time = curr_time

            # Restore attributes and training mode
            self.diff_model.sample_t = diff_model_t_sampler
            self.train(training_mode)

            loss_stats.to('cpu')
            np.savez(stats_filepath, loss_stats=loss_stats)

        if plot:
            # # Fit the negative loss vs time with a lognormal distribution
            # from scipy.optimize import curve_fit
            # data_x = self.diff_model.t_steps.cpu().log()
            #
            # # Flip the loss' sign to morph it into a distribution
            # data_y = -(loss_stats.mean.flatten() - loss_stats.mean.max().clamp_min(1))
            # pdf_total = torch.sum(data_y * data_x.diff().mean().abs())
            # data_y /= pdf_total
            # data_y_err = loss_stats.std.flatten() / pdf_total

            import matplotlib.pyplot as plt
            plot_filepath = stats_filepath.with_suffix('.pdf')
            fig = plt.figure()
            t_steps, loss_mu, loss_std = self.diff_model.t_steps.cpu(), loss_stats.mean.flatten(), loss_stats.std.flatten()
            mean_line_h = plt.plot(t_steps, loss_mu)[0]
            plt.fill_between(t_steps, loss_mu - loss_std, loss_mu + loss_std, facecolor=mean_line_h.get_color(),
                             alpha=0.5)
            plt.xscale('log')
            plt.xlabel('sigma(t)')
            plt.ylabel('loss')
            plt.title(f'loss statistics estimated with {ref_dataset.name}')
            fig.savefig(plot_filepath, format='pdf', pad_inches=None, bbox_inches='tight')

        return loss_stats


class DiffGraph(DiffPGNN):
    def __init__(self, n_layers=config.n_layers, edge_features=config.edge_features,
                 features_scale: dict[str, float] = None, data_scales: dict[str, torch.Tensor] = None,
                 cond_node_features: dict[str, int] = None, cond_edge_features: dict[str, int] = None,
                 node_latent_dim=config.node_latent_dim, edge_latent_dim=config.edge_latent_dim,
                 output_pos_diff=config.model_output_pos_diff, t_as_input_feature=config.t_as_input_feature,
                 bias=config.model_bias, norm_layer_bias=config.norm_layer_bias,
                 diff_model=config.diff_model, corr_noiser: CorrelatedNoiser = None,
                 aux_losses: list[AuxiliaryLoss] = None, conditioners: list[Conditioner] = None):
        if features_scale is None:
            features_scale = {'pos': 1.0}
        self.features_scale = features_scale
        self.output_pos_diff = output_pos_diff
        self.t_as_input_feature = t_as_input_feature

        # Node features
        if cond_node_features is None:
            cond_node_features = {}
        self.cond_node_features = cond_node_features
        node_feat_in_d = sum(cond_node_features.values())
        if self.t_as_input_feature:
            node_feat_in_d += 1
        self.node_latent_dim = node_latent_dim

        # Edge features
        supported_edge_features_d = {'d_0': 1, 'd_0^-1': 1}
        self.edge_features = dict()
        for edge_feat in edge_features:
            if edge_feat in supported_edge_features_d:
                self.edge_features[edge_feat] = supported_edge_features_d[edge_feat]
            else:
                raise ValueError(f"edge feature {edge_feat!r} is not supported.")
        if cond_edge_features is None:
            cond_edge_features = {}
        self.cond_edge_features = cond_edge_features
        self.edge_latent_dim = edge_latent_dim
        edge_feat_in_d = sum(self.edge_features.values()) + sum(self.cond_edge_features.values())

        # Use input edge features in all layers if latent edge dimension is 0
        edge_feat_extra_d = 0 if edge_latent_dim > 0 else edge_feat_in_d

        # Diffusion model
        noise_sch = init_noise_schedule()
        sigma_pos_data = {k: v.square().mean().sqrt() / features_scale['pos'] for k, v in data_scales.items()}
        if isinstance(config.sig_pos_in, str):
            sig_pos_in = sigma_pos_data[f'pos_{config.sig_pos_in}']
        elif isinstance(config.sig_pos_in, float):
            sig_pos_in = config.sig_pos_in / features_scale['pos']
        else:
            sig_pos_in = None
        if isinstance(config.sig_pos_out, str):
            sig_pos_out = sigma_pos_data[f'pos_{config.sig_pos_out}']
        elif isinstance(config.sig_pos_out, float):
            sig_pos_out = config.sig_pos_out / features_scale['pos']
        else:
            sig_pos_out = None

        if diff_model == 'VP':
            diff_model = VP(noise_sch, denoiser=self.denoise, sigma_data=sigma_pos_data,
                            corr_noiser=corr_noiser, aux_losses=aux_losses,
                            conditioners=conditioners)
        elif diff_model == 'EDM':
            diff_model = EDM(noise_sch, denoiser=self.denoise, sigma_data=sigma_pos_data, sig_pos_in=sig_pos_in,
                             sig_pos_out=sig_pos_out, corr_noiser=corr_noiser, aux_losses=aux_losses,
                             conditioners=conditioners)
        elif diff_model == 'EDM2':
            diff_model = EDM2(noise_sch, denoiser=self.denoise, sigma_data=sigma_pos_data, sig_pos_in=sig_pos_in,
                              sig_pos_out=sig_pos_out, corr_noiser=corr_noiser, aux_losses=aux_losses,
                              conditioners=conditioners)
        elif diff_model == 'VP2':
            diff_model = VP2(noise_sch, denoiser=self.denoise, sigma_data=sigma_pos_data, sig_pos_in=sig_pos_in,
                             corr_noiser=corr_noiser, aux_losses=aux_losses,
                             conditioners=conditioners)
        elif diff_model == 'VE':
            diff_model = VE(noise_sch, denoiser=self.denoise, sigma_data=sigma_pos_data, sig_pos_in=sig_pos_in,
                            corr_noiser=corr_noiser, aux_losses=aux_losses,
                            conditioners=conditioners)
        elif diff_model == 'VE2':
            diff_model = VE2(noise_sch, denoiser=self.denoise, sigma_data=sigma_pos_data, sig_pos_in=sig_pos_in,
                             corr_noiser=corr_noiser, aux_losses=aux_losses,
                             conditioners=conditioners)
        else:
            raise ValueError(f'diff_model={diff_model} is not supported.')

        # Encoders and decoders
        node_encoder = nn.Linear(node_feat_in_d, node_latent_dim, bias=config.enc_dec_bias)
        if edge_feat_in_d > 0:
            edge_encoder = nn.Linear(edge_feat_in_d, edge_latent_dim, bias=config.enc_dec_bias)
        else:
            edge_encoder = lambda x: x

        # Hidden layers
        layers = nn.ModuleList()
        for i in range(n_layers):
            GCL_bias = i == 0 if bias == 'layer1' else bias
            GCL_norm_layer_bias = i == 0 if norm_layer_bias == 'layer1' else norm_layer_bias
            layer = EGCL(node_feat_d=node_latent_dim, edge_feat_d=edge_latent_dim, edge_feat_extra_d=edge_feat_extra_d,
                         bias=GCL_bias, norm_layer_bias=GCL_norm_layer_bias)
            layers.append(layer)

        super().__init__(node_encoder=node_encoder, edge_encoder=edge_encoder, layers=layers,
                         node_feat_in_d=node_feat_in_d, edge_feat_in_d=edge_feat_in_d, diff_model=diff_model,
                         pos_scale=features_scale['pos'])

        # Extra layers
        # Cross product layers
        if config.n_cross_prod_layers > 0 and config.cross_prod_layer is not None:
            hidden_d = config.cross_prod_layer_hidden_d
            if hidden_d is None:
                hidden_d = node_latent_dim // 4
            self.cross_prod_layers = nn.ModuleList()
            for i in range(config.n_cross_prod_layers):
                layer_bias = i == 0 if bias == 'layer1' else bias
                layer_norm_layer_bias = i == 0 if norm_layer_bias == 'layer1' else norm_layer_bias

                if config.cross_prod_layer == 'ECPL':
                    layer = EquiCrossProdLayer(node_feat_d=node_latent_dim, hidden_d=hidden_d, bias=layer_bias,
                                               norm_layer_bias=layer_norm_layer_bias,
                                               edge_feat_d=edge_latent_dim,
                                               use_edge_feat=config.cross_prod_use_edge_feat)
                else:
                    raise NotImplementedError
                self.cross_prod_layers.append(layer)
        else:
            self.cross_prod_layers = None

        # Replace all square nn.Linear() layers with RotationLayer
        if config.use_rot_layer:
            self.replace_Linear_with_Rotation()

        if config.output_gain_init is not None:
            self.output_gain = Gain(config.output_gain_init)
        else:
            self.output_gain = lambda x: x

        # Buffers
        self.epsilon_d0_inv = nn.parameter.Buffer(torch.tensor(config.epsilon_d_inv), persistent=True)

    def format_denoiser_input(self, tensors_batch: dict[str, torch.Tensor]) -> (torch.Tensor, dict[str, torch.Tensor]):
        """
        Parses a batch of tensors to format inputs to the denoiser
        Args:
            tensors_batch: dictionary of tensors

        Returns:
                noised_properties, denoiser_kwargs
        """
        nodes_pos = tensors_batch['nodes_pos']
        denoiser_kwargs_name = ['nodes_adj', 'nodes_adj_range', 'cond_node_features', 'cond_edge_features',
                                'cross_prod_triplet_ind']
        denoiser_kwargs = {k: tensors_batch[k] for k in denoiser_kwargs_name if k in tensors_batch}
        return nodes_pos, denoiser_kwargs

    def retrieve_nodes_pos_from_denoiser_output(self, denoiser_output) -> torch.Tensor:
        if isinstance(denoiser_output, tuple):
            nodes_pos = denoiser_output[0]
        elif isinstance(denoiser_output, torch.Tensor):
            nodes_pos = denoiser_output
        else:
            raise ValueError(f"Cannot parse output of {self}.")
        return nodes_pos

    def apply_extra_hidden_layers(self, i, node_pos: torch.Tensor, node_feat: torch.Tensor, edge_feat: torch.Tensor,
                                  global_feat: torch.Tensor, cross_prod_triplet_ind: torch.Tensor = None,
                                  cross_prod_triplet_edge_ind: torch.Tensor = None, **kwargs):
        node_pos_out = node_pos
        if self.cross_prod_layers:
            ind = i * len(self.cross_prod_layers) // self.n_layers
            layer = self.cross_prod_layers[ind]
            node_pos_out = layer(node_pos=node_pos_out, node_feat=node_feat, triplet_ind=cross_prod_triplet_ind,
                                 edge_feat=edge_feat, triplet_edges_ind=cross_prod_triplet_edge_ind)

        return node_pos_out, node_feat, edge_feat, global_feat

    def gather_input_tensors(self, biomolecules: list[Biomolecule], **kwargs) -> dict[str, torch.Tensor]:
        return gather_input_tensors(biomolecules=biomolecules, **kwargs)

    def gather_constant_input_tensors(self, biomolecule: Biomolecule, **kwargs):
        const_feat = {}
        cond_node_feat, cond_edge_feat = assemble_features(mols=[biomolecule],
                                                           node_features=list(self.cond_node_features),
                                                           edge_features=list(self.cond_edge_features),
                                                           pos_scale=self.pos_scale.item())
        if cond_node_feat:
            const_feat['cond_node_features'] = torch.cat(list(cond_node_feat.values()), dim=-1)
        if cond_edge_feat:
            const_feat['cond_edge_features'] = torch.cat(list(cond_edge_feat.values()), dim=-1)

        # Define the triplet indices for the cross-product layers
        if self.cross_prod_layers is not None:
            if biomolecule.element_class in [heavy_atoms_class_name, all_atoms_class_name]:
                bond_energy_module = BondEnergy(top_file=biomolecule.top_file, element_class=biomolecule.element_class)
            else:
                bond_energy_module = None
            const_feat['cross_prod_triplet_ind'] = define_node_triplets(biomolecule, bond_energy_module).unsqueeze(0)

        return const_feat

    def denoise(self, node_pos: torch.Tensor, t_embedding: torch.Tensor, nodes_adj: torch.Tensor,
                cond_node_features: torch.Tensor = None, cond_edge_features: torch.Tensor = None,
                cross_prod_triplet_ind: torch.Tensor = None, **kwargs):
        B, N = node_pos.shape[:2]

        # Assemble the node features
        node_features = []
        if self.t_as_input_feature:
            node_features.append(t_embedding.reshape(-1, 1, 1).expand(B, N, 1))

        # Add conditional node features
        if cond_node_features is not None:
            node_features.append(cond_node_features)
        node_features = torch.cat(node_features, dim=-1)

        # Gather the node indices of each edge in the batch
        edges_ind = nodes_adj.nonzero()  # shape (E,3)

        # Assemble the edge features
        edge_features = []

        # Calculate the distance between each node (if needed)
        if set(self.edge_features) & {'d_0', 'd_0^-1'}:
            edges_node_pos = node_pos[edges_ind[:, 0].unsqueeze(-1), edges_ind[:, 1:], :]  # shape=(E,2,3)
            edges_dist = torch.diff(edges_node_pos, dim=-2).norm(dim=-1)

        for f in self.edge_features:
            if f == 'd_0':
                edge_features.append(edges_dist)
            elif f == 'd_0^-1':
                edge_features.append(1 / (edges_dist + self.epsilon_d0_inv))
            else:
                raise NotImplementedError(f"Edge features {f!r} is not implemented.")

        if cond_edge_features is not None:
            edge_features.append(cond_edge_features[edges_ind.unbind(-1)])
        edge_features = torch.cat(edge_features, dim=-1) if edge_features else None

        # Keep reference of initial positions
        node_pos_start = node_pos

        # Call the forward method
        node_pos, _, _, _ = self(node_pos=node_pos,
                                 node_feat=node_features,
                                 edge_feat=edge_features,
                                 edges_ind=edges_ind,
                                 nodes_adj=nodes_adj,
                                 cross_prod_triplet_ind=cross_prod_triplet_ind)

        # Subtract initial positions
        if self.output_pos_diff:
            node_pos = node_pos - node_pos_start

        node_pos = self.output_gain(node_pos)

        return node_pos

    def forward(self, *args, cross_prod_triplet_ind=None, nodes_adj=None, edges_ind=None, **kwargs):
        # Find the global edge index of the two edges that define each given cross_product triplet
        cross_prod_triplet_edge_ind = None
        if cross_prod_triplet_ind is not None and any(layer.use_edge_feat for layer in self.cross_prod_layers):
            # Define a map of shape=nodes_adj.shape that returns the global edge ind for each non-zero entry in nodes_adj.
            edges_local_to_global_ind = -1 * torch.ones_like(nodes_adj)
            edges_local_to_global_ind[edges_ind.unbind(-1)] = torch.arange(edges_ind.shape[0], device=edges_ind.device)
            cross_prod_batch_ind = torch.arange(cross_prod_triplet_ind.shape[0], device=edges_ind.device).reshape(-1, 1)

            # Find the global edge index for each 2 edges that define each given triplet.
            # cross_prod_triplet_ind provides the node index of the nodes involved in the cross product
            edge1_ind = edges_local_to_global_ind[
                cross_prod_batch_ind, *cross_prod_triplet_ind[..., [0, 1]].sort(-1)[0].unbind(-1)]
            edge2_ind = edges_local_to_global_ind[
                cross_prod_batch_ind, *cross_prod_triplet_ind[..., [0, 2]].sort(-1)[0].unbind(-1)]
            cross_prod_triplet_edge_ind = torch.stack([edge1_ind, edge2_ind], dim=-1)

        return super().forward(*args, cross_prod_triplet_ind=cross_prod_triplet_ind,
                               nodes_adj=nodes_adj, edges_ind=edges_ind,
                               cross_prod_triplet_edge_ind=cross_prod_triplet_edge_ind,
                               **kwargs)

    def gen_rand_inputs(self, B=5, N=20):
        inputs = super().gen_rand_inputs(B=B, N=N)
        if self.cross_prod_layers is not None:
            if any(layer.use_edge_feat for layer in self.cross_prod_layers):
                # When cross-product layers use edge features, elements of the triplets must be connected with edges.
                # Sometimes, since the adjacency matrix is generated randomly, a node can have less than 2 neighbors
                # Since at least one triplet index is needed for each node, such cases will cause errors in unflattening
                # Generate inputs until this condition is satisfied
                for j in range(10):
                    # Construct a full adjacency matrix
                    nodes_adj_full = inputs['nodes_adj']
                    nodes_adj_full = (nodes_adj_full | nodes_adj_full.transpose(-1, -2)) & ~nodes_adj_full.triu().tril()

                    # Find all nodes that are connected to at least two nodes
                    node_is_sufficiently_conn = nodes_adj_full.sum(-1) >= 2
                    connected_nodes_ind = node_is_sufficiently_conn.nonzero()
                    if node_is_sufficiently_conn.all():
                        break
                    else:
                        inputs = super().gen_rand_inputs(B=B, N=N)

                conn_nodes_neigh_ind = nodes_adj_full[connected_nodes_ind.unbind(-1)].nonzero()  # shape=(M,2)

                # Select two neighbor nodes for each connected node
                conn_nodes_triplet_ind = [conn_nodes_neigh_ind[conn_nodes_neigh_ind[:, 0] == i, 1][:2]
                                          for i in range(connected_nodes_ind.shape[0])]
                conn_nodes_triplet_ind = torch.stack(conn_nodes_triplet_ind)

                # Add connected node index to the triplet index
                conn_nodes_triplet_ind = torch.cat([connected_nodes_ind[:, -1:], conn_nodes_triplet_ind], -1)

                inputs['cross_prod_triplet_ind'] = conn_nodes_triplet_ind.unflatten(dim=0,
                                                                                    sizes=nodes_adj_full.shape[:2])
            else:
                inputs['cross_prod_triplet_ind'] = torch.randint(0, N, (B, 5, 3), device=self.device)

        return inputs

    @property
    def element_feature_sizeof(self):
        """
        Calculates an approximate value of the memory footprint of passing 1 element of 1 molecule through the model.
        Returns: size (bytes) of node tensors
        """
        n_layers = self.n_layers.item() if isinstance(self.n_layers, torch.Tensor) else self.n_layers
        element_size = (self.node_latent_dim + 3) * n_layers * torch.tensor(1.0).__sizeof__()
        return element_size
