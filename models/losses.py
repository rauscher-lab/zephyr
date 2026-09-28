import copy
import typing
import warnings
from typing import Any

import torch
from torch import nn as nn

import config
from chemistry.energy import BondEnergy, BondAngleEnergy, DihedralAngleEnergy, LJEnergy
from data.data_classes import heavy_atoms_class_name
from analysis.stats import StatisticsArray
from utils import calculate_func_mu_sigma, calc_dihedral_angle


class AuxiliaryLoss(nn.Module):
    """
    Module defining loss as the squared error of a multivariate (potentially correlated) variable.
    """

    def __init__(self, d: int, T=config.n_diff_steps, pos_scale=1.0, sigma_func: typing.Callable = None,
                 estimate_covariance=config.aux_loss_estimate_cov,
                 weight_method=config.aux_loss_weight_method,
                 stats_n_min=config.aux_loss_stats_n_min, estimate_U_weights=config.aux_loss_reweight, **kwargs):
        super().__init__()
        self.d = d
        self.sigma_func = sigma_func
        self.name_short = self.__class__.__name__.replace('Loss', 'L')
        self.is_active = True
        self.pos_scale = nn.parameter.Buffer(torch.as_tensor(pos_scale), persistent=True)
        self.estimate_var_cov = estimate_covariance
        self.estimate_U_weights = estimate_U_weights
        self.weight_method = weight_method
        self.stats_n_min = int(stats_n_min)
        self.var_stats: StatisticsArray = None
        self.U_stats: StatisticsArray = None
        n_t = T + 1
        if self.estimate_var_cov:
            if self.weight_method == 'cov_inv':
                R_shape = (n_t, self.d, self.d)
            else:
                R_shape = (n_t, self.d)
            self.R = nn.parameter.Buffer(torch.zeros(R_shape, dtype=torch.double), persistent=True)
        else:
            self.R = None

        if self.estimate_U_weights:
            self.U_weights = nn.parameter.Buffer(torch.zeros((n_t,), dtype=torch.double), persistent=True)
        else:
            self.U_weights = None

        if self.weight_method == 'cov_inv' and self.stats_n_min < self.d:
            raise ValueError(f"stats_n_min is smaller that the conditioner's random variable dimension."
                             f"The accumulated covariance matrix will not be invertible.")

    def rand_var(self, x: torch.Tensor):
        raise NotImplementedError

    def calculate_rand_var(self, x: torch.Tensor):
        y = self.rand_var(self.pos_scale * x)  # Scale the positions with the input scale
        return y

    def rand_var_diff(self, x: torch.Tensor, x_ref: torch.Tensor):
        return self.calculate_rand_var(x=x) - self.calculate_rand_var(x=x_ref)

    def U(self, x: torch.Tensor, x_denoised: torch.Tensor, t: torch.Tensor, t_ind: torch.Tensor, sum_batch=True, *args,
          reweight=False, **kwargs):
        if t_ind.ndim > 1:
            t_ind = t_ind.flatten()
        rand_var_diff = self.rand_var_diff(x=x_denoised, x_ref=x)

        if self.R is not None:
            if self.R.ndim > 2:
                rand_var_diff = (self.R[t_ind] @ rand_var_diff.to(self.R.dtype).unsqueeze(-1)).squeeze(-1)
            else:
                rand_var_diff = self.R[t_ind] * rand_var_diff
        squared_error = torch.square(rand_var_diff) / 2

        if reweight:
            squared_error *= self.U_weights[t_ind].reshape(-1, 1)

        if sum_batch:
            squared_error = squared_error.flatten()
        else:
            squared_error = squared_error.reshape(squared_error.shape[0], -1)
        U = torch.mean(squared_error, dim=-1)
        return U

    def init_var_stats(self):
        # Initialize var stats if not already initialized
        cov_method = 'diag' if self.weight_method in ['mu2_inv', 'var_inv'] else 'full'
        if self.estimate_var_cov:
            if self.var_stats is None:
                self.var_stats = StatisticsArray(shape=(self.R.shape[0],), d=self.d, cov_method=cov_method,
                                                 dtype=torch.double)
            self.var_stats.to(self.R.device)

            self.R.data.zero_()

        # Initialize D0 var stats if not already initialized
        if self.estimate_U_weights:
            if self.U_stats is None:
                self.U_stats = StatisticsArray(shape=(self.U_weights.shape[0],), d=self.d, cov_method=cov_method,
                                               dtype=torch.double)
            self.U_stats.to(self.U_weights.device)

            self.U_weights.data.zero_()

    @property
    def need_stats(self):
        """
        Returns a bool to indicate if the conditioner still needs to estimate statistices needed to define its paramters.
        Returns:
            bool = True if statistics are needed
        """
        need_var_stats = self.estimate_var_cov and not self.R.any()
        need_D0_var_stats = self.estimate_U_weights and not self.U_weights.any()
        return need_var_stats or need_D0_var_stats

    @property
    def n_stats(self):
        if self.estimate_var_cov:
            if self.var_stats is not None:
                var_stats_n = int(self.var_stats.n.min())
            elif not self.R.any():
                var_stats_n = 0
            else:
                var_stats_n = config.aux_loss_stats_n_min
        else:
            var_stats_n = None

        if self.estimate_U_weights:
            if self.U_stats is not None:
                D0_var_stats_n = int(self.U_stats.n.min())
            elif not self.U_weights.any():
                D0_var_stats_n = 0
            else:
                D0_var_stats_n = config.aux_loss_stats_n_min
        else:
            D0_var_stats_n = None

        if var_stats_n is not None and D0_var_stats_n is not None:
            n_stats = min(var_stats_n, D0_var_stats_n)
        elif var_stats_n is not None or D0_var_stats_n is not None:
            n_stats = var_stats_n if D0_var_stats_n is None else D0_var_stats_n
        else:
            n_stats = None
        return n_stats

    def accumulate_var_stats(self, x: torch.Tensor, x_noisy: torch.Tensor, t_ind: torch.Tensor, **kwargs):
        rand_var_diff = self.rand_var_diff(x=x_noisy, x_ref=x)
        nan_inds = torch.nonzero(rand_var_diff.isnan())
        if nan_inds.nelement() > 0:
            raise ValueError(f"Found nans in {self.__class__.__name__}. indices:\n{nan_inds}")
        self.var_stats += StatisticsArray(shape=self.var_stats.shape, cov_method=self.var_stats.cov_method,
                                          ind=t_ind.flatten(), values=rand_var_diff)

        # Calculate R when enough stats will be collected
        self.define_R_mu()

    def accumulate_U_stats(self, x: torch.Tensor, x_denoised: torch.Tensor, t_ind: torch.Tensor):
        rand_var_diff = self.rand_var_diff(x=x_denoised, x_ref=x)
        nan_inds = torch.nonzero(rand_var_diff.isnan())
        if nan_inds.nelement() > 0:
            raise ValueError(f"Found nans in {self.__class__.__name__}. indices:\n{nan_inds}")
        self.U_stats += StatisticsArray(shape=self.U_stats.shape, cov_method=self.U_stats.cov_method,
                                        ind=t_ind.flatten(), values=rand_var_diff)

        # Calculate U_weights when enough stats will be collected
        self.define_U_weights()

    def rand_var_mu_sigma_2ndorder(self, x_mu: torch.Tensor, x_cov: torch.Tensor):
        """
        Second order estimates of the random variable average and covariance given the average and covariance of its inputs
        Args:
            x_mu: average of the input of the rand_var method
            x_cov: covariance of the input of the rand_var method
        Returns:
            rand_var_mu, rand_var_cov
        """
        B, N, d = x_mu.shape
        var_func = lambda x: self.calculate_rand_var(x=x.reshape(B, N, d))
        rand_var_mu, rand_var_cov = calculate_func_mu_sigma(var_func, mu=x_mu.reshape(B, -1), cov=x_cov.square(),
                                                            create_graph=True)
        return rand_var_mu, rand_var_cov

    def define_R_mu(self):
        if self.estimate_var_cov and torch.all(self.var_stats.n >= self.stats_n_min):
            print(f"Defining {self.__class__.__name__} covariance")
            if self.weight_method == 'var_tot' or self.weight_method == 'var_tot_inv':
                R = 1 / self.var_stats.cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).sqrt()
            elif self.weight_method == 'cov_inv':
                # Use the Cholesky decomposition of the covariance matrix inverse to weight the conditioner energy
                try:
                    R = torch.linalg.cholesky(torch.linalg.inv(self.var_stats.cov), upper=True)
                except RuntimeError as e:
                    # Take the inverse of the cholesky composition as opposed to the cholesky of the inverse.
                    # This is not a true Cholesky decomposition, but it is sufficient for calculating squared errors.
                    warnings.warn(f"Using approximate Cholesky decomposition for {self.__class__.__name__}.")
                    R = torch.linalg.cholesky(self.var_stats.cov, upper=False).inverse().tril()
            elif self.weight_method == 'var_inv':
                R = 1 / self.var_stats.cov.diagonal(dim1=-2, dim2=-1).sqrt()
            elif self.weight_method == 'avg_inv':
                R = 1 / self.var_stats.mean.abs()
            elif self.weight_method == 'avg_tot_inv':
                R = 1 / self.var_stats.mean.abs().sum(dim=-1, keepdim=True)
            elif self.weight_method == 'mu2_tot_inv':
                R = 1 / self.var_stats.mu2.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).sqrt()
            elif self.weight_method == 'mu2_inv':
                R = 1 / self.var_stats.mu2_diag.sqrt()
            else:
                raise ValueError(f"weight method={self.weight_method} is undefined.")
            self.R.data = R

            # Discard stats since they are no longer needed
            self.var_stats = None

    def define_U_weights(self):
        """
        Defines weights of U such that <U(t)> = 1 at all noise levels t under a denoiser with a zeroed network.
        """
        if self.estimate_U_weights and torch.all(self.U_stats.n >= self.stats_n_min):
            print(f"Defining {self.__class__.__name__} U_weights.")

            # To define the loss weights we require that < y.T C^-1 y > / 2 = 1 where y is the conditioner's random variable
            # with zeroed-denoiser. Stats of y were accumulated in D0_var_stats and R should already be defined.
            # < y.T C^-1 y > = < y.T R^T R y > = tr(S R^T R) = tr(S C^-1) where S_ij = <y_i y_j>.
            # Therefore, loss_w = 2/tr(S C^-1) for each noise level.

            # Since y was collected under a denoiser with a skip connection, D0_var_stats will contain correlations of the data itself.
            # Since we are only interested in the overall scale of the variable in each dimension, we neglect these correlations.
            S = self.U_stats.mu2_diag
            if self.U_stats.cov_method == 'full':
                S = torch.diag_embed(S)

            # Define R
            if self.estimate_var_cov:
                R = self.R.data
                if R.ndim < 3 and self.U_stats.cov_method == 'full':
                    R = torch.diag_embed(R)
            else:
                if self.U_stats.cov_method == 'full':
                    R = torch.eye(self.d, dtype=S.dtype, device=S.device)
                else:
                    R = torch.ones(self.d, dtype=S.dtype, device=S.device)
                R = R.unsqueeze(0)

            # Average over all dimensions of the conditioners' random variable to find the average scale at each noise level.
            # C = torch.linalg.inv(R.transpose(-1, -2) @ R)  # C^-1 = R^T R
            # C_inv = R.transpose(-1, -2) @ R

            # import matplotlib.pyplot as plt
            # plt.figure(); plt.matshow(C_inv[0,:10,:10].abs())
            # C_diag, S_diag = C.diagonal(dim1=-1, dim2=-2), S.diagonal(dim1=-1, dim2=-2)
            # C_var_avg, S_var_avg = C_diag.mean(-1), S_diag.mean(-1)
            # U_weights = C_var_avg / S_var_avg
            # plt.figure();plt.plot(U_weights)

            # Define the weights
            if self.U_stats.cov_method == 'diag':
                U_weights = 2 / (R * S * R).sum(dim=-1)
            else:
                U_weights = 2 / torch.einsum('...ii -> ...', R @ S @ R.transpose(-1, -2))

            # Since loss is averaged over feat. dim., we require < y.T C^-1 y > = d where d is the rand var dimension
            # because the loss is divided by d
            U_weights *= self.d

            # # Tests for the diffusion loss where the weights should correspond to Karras' EDM weights
            # # Average over all atoms by averaging each 3x3 blocks on the diagonal of the empirical C
            # block_avg = lambda x: x.unflatten(dim=-2, sizes=(-1, 3)).unflatten(dim=-1, sizes=(-1, 3)).diagonal(dim1=-2,dim2=-4).mean(1)
            # C_3D = block_avg(C)
            # R_3D = torch.cholesky(torch.linalg.inv(C_3D), upper=True)
            # S_3D = block_avg(S)
            #
            # # The following quantity corresponds to lambda(t) in Karras' diffusion loss because lambda(t) is derived by requiring that
            # # <z^T z>_y,n = 1 where z = D_0(y+n) - y where D_0 is the output of the denoiser with 0 network output.
            # W = 1 / S_3D.diagonal(dim1=-2, dim2=-1).mean(-1)
            # # The following quantity is another way to derive the weights by averaging over all 3 dimensions.
            # W2 = 1 / (torch.einsum('ijj -> i', R_3D @ S_3D @ R_3D.transpose(-1, -2)) / 3)
            # # W and W2 correspond to sigma(t)^2*lambda(t) in Karras' paper since the squared error loss is not divided by sigma^2.

            # self.U_weights = 1 / torch.einsum('ijj -> i', R @ S @ R.transpose(-1, -2))
            self.U_weights.data = U_weights

            # Discard stats since they are no longer needed
            self.U_stats = None

    def get_extra_state(self) -> Any:
        return {'var_stats': self.var_stats, 'D0_var_stats': self.U_stats}

    def set_extra_state(self, state: Any):
        self.var_stats = state['var_stats']
        self.U_stats = state['D0_var_stats']


class CombinedLoss(AuxiliaryLoss):
    def __init__(self, aux_losses: list[AuxiliaryLoss], **kwargs):
        self.aux_losses = aux_losses
        cond_d = [c.d for c in aux_losses]
        d = sum(cond_d)
        self.cond_var_ind = torch.split(torch.arange(d), cond_d)
        cond_is_diff = torch.tensor([isinstance(c, DiffLoss) for c in aux_losses])
        self.diff_cond_ind = cond_is_diff.nonzero().item() if cond_is_diff.any() else None
        super().__init__(d=d, **kwargs)

    def rand_var(self, x: torch.Tensor):
        rand_vars = []
        for c in self.aux_losses:
            c_var = c.rand_var(x=x)
            rand_vars.append(c_var)
        return torch.cat(rand_vars, dim=-1)

    def calculate_diff_C_mask(self):
        diff_cond_var_ind = self.cond_var_ind[self.diff_cond_ind]
        diff_cond_d = self.aux_losses[self.diff_cond_ind]
        C_mask = torch.ones((self.d, self.d), dtype=torch.bool)
        diff_var_2D_ind = (diff_cond_var_ind[:, None], diff_cond_var_ind[None, :])
        C_mask[diff_var_2D_ind] = torch.eye(diff_cond_d.d, dtype=torch.bool)
        return C_mask

    def define_R_mu(self):
        if self.estimate_var_cov and torch.all(self.var_stats.n >= self.stats_n_min):
            # Zero out cross-correlations in dimensions that correspond to the diffusion conditioner (if any)
            if self.diff_cond_ind is not None:
                C_diff_mask = self.calculate_diff_C_mask().to(torch.double).unsqueeze(dim=0)
                self.var_stats.S2.data = torch.abs(C_diff_mask * self.var_stats.S2.data)
            super().define_R_mu()

            # Distribute the R blocks to each conditioner (assumes zero cross-correlation across conditioners)
            for i, c in enumerate(self.aux_losses):
                var_ind = self.cond_var_ind[i]
                c.R.data = self.R.data[:, var_ind[:, None], var_ind[None, :]]

    def define_U_weights(self):
        if self.estimate_U_weights and torch.all(self.U_stats.n >= self.stats_n_min):
            if self.diff_cond_ind is not None:
                C_diff_mask = self.calculate_diff_C_mask().unsqueeze(dim=0)
                self.U_stats.S2.data = C_diff_mask * self.U_stats.S2.data
            super().define_U_weights()

            # Copy the U_weights to each conditioner
            for c in self.aux_losses:
                c.U_weights = self.U_weights


class DiffLoss(AuxiliaryLoss):
    def __init__(self, d: int = 3, sigma_func: typing.Callable = None):
        super().__init__(d=d, sigma_func=sigma_func)

    def rand_var(self, x: torch.Tensor):
        return x.reshape(x.shape[0], -1)


class BondLengthLoss(AuxiliaryLoss):
    def __init__(self, top_filepath: str, bond_weights: torch.Tensor = None, power=1.0, **kwargs):
        bond_energy_module = BondEnergy(top_file=top_filepath, element_class=heavy_atoms_class_name)
        super().__init__(d=bond_energy_module.n_bonds, **kwargs)
        self.bond_energy_module = bond_energy_module
        assert bond_weights is None, "bond weights is deprecated"
        self.power = nn.parameter.Buffer(torch.tensor(power), persistent=True)

    def rand_var(self, x: torch.Tensor):
        y = self.bond_energy_module.calculate_bond_length(x)
        if self.power != 1.0:
            y = y.pow(y)
        return y


class BondAngleLoss(AuxiliaryLoss):
    def __init__(self, top_filepath: str, power=1.0, **kwargs):
        UB_energy_module = BondAngleEnergy(top_file=top_filepath, element_class=heavy_atoms_class_name)
        super().__init__(d=UB_energy_module.n_bond_angles, **kwargs)
        self.BA_energy_module = UB_energy_module
        self.power = nn.parameter.Buffer(torch.tensor(power), persistent=True)

    def rand_var(self, x: torch.Tensor):
        y = self.BA_energy_module.calculate_bond_angle(x)
        if self.power != 1.0:
            y = y.pow(self.power)
        return y


class DihedralAngleLoss(AuxiliaryLoss):
    def __init__(self, top_filepath: str, **kwargs):
        dih_energy_module = DihedralAngleEnergy(top_file=top_filepath, element_class=heavy_atoms_class_name)
        super().__init__(d=2 * dih_energy_module.n_angles_unique, **kwargs)
        self.dih_energy_module = dih_energy_module

    def rand_var(self, x: torch.Tensor):
        position_quartet = x[..., self.dih_energy_module.dihedrals_unique_atom_ind, :]
        dih_angles = calc_dihedral_angle(position_quartet)
        return dih_angles


class BondEnergyLoss(AuxiliaryLoss):
    def __init__(self, top_filepath: str, weights: torch.Tensor = None, tot=False, **kwargs):
        bond_energy_module = BondEnergy(top_file=top_filepath, element_class=heavy_atoms_class_name)

        # Normalize the bond weights to stabilize the numerical precision
        norm_factor = 2 * bond_energy_module.kb.sum()
        bond_energy_module.kb /= norm_factor
        if 'dataset_stats' in kwargs:
            kwargs['dataset_stats'] = copy.deepcopy(kwargs['dataset_stats'])
            kwargs['dataset_stats'].scale(s=1 / norm_factor)

        super().__init__(d=1 if tot else bond_energy_module.n_bonds, **kwargs)
        self.name_short = self.__class__.__name__.replace('Conditioner', ('Sum' if tot else '') + 'C')
        self.bond_energy_module = bond_energy_module
        self.tot = tot

        if weights is not None:
            warnings.warn(f"Weights are not used with {self.__class__.__name__}")

    def rand_var(self, x: torch.Tensor):
        energy = self.bond_energy_module(x)
        if self.tot:
            energy = energy.sum(dim=-1, keepdim=True)
        return energy


class BondAngleEnergyLoss(AuxiliaryLoss):
    def __init__(self, top_filepath: str, tot=False, **kwargs):
        BondAngleEnergy_module = BondAngleEnergy(top_file=top_filepath, element_class=heavy_atoms_class_name)
        super().__init__(d=1 if tot else BondAngleEnergy_module.n_bond_angles, **kwargs)
        self.name_short = self.__class__.__name__.replace('Conditioner', ('Tot' if tot else '') + 'C')
        self.BondAngleEnergy_module = BondAngleEnergy_module
        self.tot = tot

    def rand_var(self, x: torch.Tensor):
        energy = self.BondAngleEnergy_module(x)
        if self.tot:
            energy = energy.sum(dim=-1, keepdim=True)
        return energy


class DihedralAngleEnergyLoss(AuxiliaryLoss):
    def __init__(self, top_filepath: str, tot=False, **kwargs):
        DihedralAngleEnergy_module = DihedralAngleEnergy(top_file=top_filepath, element_class=heavy_atoms_class_name)
        super().__init__(d=1 if tot else DihedralAngleEnergy_module.n_angles, **kwargs)
        self.name_short = self.__class__.__name__.replace('Conditioner', ('Tot' if tot else '') + 'C')
        self.DihedralAngleEnergy_module = DihedralAngleEnergy_module
        self.tot = tot

    def rand_var(self, x: torch.Tensor):
        energy = self.DihedralAngleEnergy_module(x)
        if self.tot:
            energy = energy.sum(dim=-1, keepdim=True)
        return energy


class LJEnergyLoss(AuxiliaryLoss):
    def __init__(self, top_filepath: str, tot=False, **kwargs):
        LJEnergy_module = LJEnergy(top_file=top_filepath, element_class=heavy_atoms_class_name)
        super().__init__(d=1 if tot else LJEnergy_module.n_pairs, **kwargs)
        self.name_short = self.__class__.__name__.replace('Conditioner', ('Tot' if tot else '') + 'C')
        self.LJEnergy_module = LJEnergy_module
        self.tot = tot

    def rand_var(self, x: torch.Tensor):
        energy = self.LJEnergy_module(x, epsilon=1e-2)
        if self.tot:
            energy = energy.sum(dim=-1, keepdim=True)
        return energy


class LJDistLoss(AuxiliaryLoss):
    def __init__(self, top_filepath: str, **kwargs):
        LJEnergy_module = LJEnergy(top_file=top_filepath, element_class=heavy_atoms_class_name)
        super().__init__(d=LJEnergy_module.n_pairs, **kwargs)
        # self.name_short = self.__class__.__name__.replace('Conditioner', ('Tot' if tot else '') + 'C')
        self.LJEnergy_module = LJEnergy_module

    def rand_var(self, x: torch.Tensor):
        pair_dist = self.LJEnergy_module.calculate_pair_dist(x)
        return pair_dist


class RgLoss(AuxiliaryLoss):
    def __init__(self, d=1, **kwargs):
        super().__init__(d=d, **kwargs)

    def rand_var(self, x: torch.Tensor):
        # x_com = torch.mean(x, dim=-2, keepdim=True)
        # x = x - x_com
        # x_rg = torch.sum(x ** 2, dim=-1).mean(dim=-1, keepdim=True).sqrt()
        x_rg = torch.sqrt(torch.var(x, correction=0, dim=-2).sum(dim=-1, keepdim=True) + 1e-8)
        # x_rg = torch.std(x, correction=0, dim=[-1,-2]).unsqueeze(-1)*math.sqrt(x.shape[-1])
        return x_rg
