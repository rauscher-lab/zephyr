import math
import typing
import warnings
import functools
from typing import TYPE_CHECKING

import torch
from torch import nn

import config
from analysis.stats import Statistics
from chemistry.energy import BondEnergy, BondAngleEnergy, LJEnergy
from data.data_classes import heavy_atoms_class_name
from utils import NonCentralChicdf, mvn_cond_params

variable_conditioners = ['coarse_pos_cons', 'coarse_pos', 'elements_coarse_pos', 'coarse_center_dist']

if TYPE_CHECKING:
    from models.diffusers import CorrelatedNoiser


def U_deactivate_decorator(U_ori):
    """
    Defines a wrapper of conditioners U() method that sets energy to zero when sigma is below the deactivation sigma
    Args:
        U_ori: Original U method

    Returns:
        U_wrapper
    """

    @functools.wraps(U_ori)
    def U_wrapper(self: 'Conditioner', x: torch.Tensor, t: torch.Tensor, *args, sum_batch=True, **kwargs):
        t_is_active = self.sigma_func(t).gt(self.sigma_deactivate)
        if t_is_active.any():
            U = U_ori(self, x, t, *args, sum_batch=False, **kwargs)
            if not t_is_active.all():
                U = t_is_active.reshape((-1, *(U.ndim - 1) * [1])) * U
            if sum_batch:
                U = U.sum(dim=0)
        else:
            U_shape = (1,) if sum_batch else (x.shape[0], 1)
            U = x.new_zeros(U_shape, requires_grad=x.requires_grad)

        return U

    return U_wrapper


class Conditioner(nn.Module):
    """
    Conditioner framework introduced in the Chroma model

    Reference: 2023, "Illuminating protein space with a programmable generative model", Section O.2
    """

    def __init__(self, sigma_func: typing.Callable = None, sigma_deactivate=0.0,
                 corr_noiser: 'CorrelatedNoiser' = None):
        super().__init__()
        self.sigma_func = sigma_func
        self.corr_noiser = corr_noiser
        self.name_short = self.__class__.__name__.replace('Conditioner', 'C')
        self.is_active = True
        self.sigma_deactivate = sigma_deactivate
        self.nonfin_force_counters = nn.parameter.Buffer(torch.tensor(0, dtype=torch.int32))

    def transform_x(self, x: torch.Tensor, t: torch.Tensor = None):
        return x

    @U_deactivate_decorator
    def U(self, x: torch.Tensor, t: torch.Tensor, *args, sum_batch=True, **kwargs):
        raise NotImplementedError

    @torch.enable_grad()
    def F(self, x: torch.Tensor, t: torch.Tensor, output_U=False, **kwargs):
        if not x.requires_grad:
            raise ValueError(f"x.requires_grad must be True to calculate forces on x")

        # Transform x
        x_trans = self.transform_x(x=x, t=t)

        # Calculate the force on the unconstrained variable
        U = self.U(x=x_trans, t=t, sum_batch=False, **kwargs)
        U_tot = U.sum()
        x_grad = torch.autograd.grad(outputs=U_tot, inputs=x, retain_graph=False, create_graph=False,
                                     allow_unused=True)[0]
        x_force = -x_grad if x_grad is not None else torch.zeros_like(x)

        if x_force.is_sparse:
            force_is_non_finite = ~torch.isfinite(x_force.coalesce().values())
        else:
            force_is_non_finite = ~torch.isfinite(x_force)
        x_force = torch.where(force_is_non_finite, torch.zeros_like(x_force), x_force)

        # Track occurrence of non-finite forces
        self.nonfin_force_counters.add_(force_is_non_finite.any().to(torch.int32))
        if not torch.compiler.is_compiling():
            self.print_warnings()

        if output_U:
            return x_force, U.detach()
        else:
            return x_force

    def print_warnings(self):
        if self.nonfin_force_counters > 0:
            warnings.warn(f"{self.__class__.__name__} forces were non-finite. Non-finite forces were set to zero.")
            self.nonfin_force_counters.zero_()


class CombinedConditioner(Conditioner):
    def __init__(self, conditioners: list[Conditioner], **kwargs):
        super().__init__(**kwargs)
        self.conditioners = nn.ModuleList(conditioners)

    def transform_x(self, x: torch.Tensor, t: torch.Tensor = None):
        for c in self.conditioners:
            x = c.transform_x(x=x, t=t)
        return x

    def U(self, x: torch.Tensor, t: torch.Tensor, *args, sum_batch=True, **kwargs):
        cond_U = []
        for c in self.conditioners:
            c_U = c.U(x=x, t=t, sum_batch=sum_batch, **kwargs)
            c_U_nonfinite = ~c_U.isfinite()
            if c_U_nonfinite.any():
                warnings.warn(f"{c_U_nonfinite.sum()} {c.__class__.__name__!r} energies were non-finite. Setting U=0.")
                c_U[c_U_nonfinite] = 0.0 * c_U[c_U_nonfinite]
            if sum_batch:
                cond_U.append(c_U.reshape((1,)))
            else:
                cond_U.append(c_U.reshape(-1, 1))
        U = torch.cat(cond_U, dim=-1)
        return U

    def F(self, *args, **kwargs):
        # Compute each force separately to allow setting non-finite conditioner forces to zero
        output = []
        for c in self.conditioners:
            output.append(c.F(*args, **kwargs))
        if isinstance(output[0], tuple):
            F_tot = sum([output[j][0] for j in range(len(output))])
            U = torch.cat([output[j][1].reshape(-1, 1) for j in range(len(output))], dim=-1)
            return F_tot, U
        else:
            return sum(output)

    def print_warnings(self):
        for c in self.conditioners:
            c.print_warnings()


class DiffusionConditioner(Conditioner):
    def __init__(self, denoiser: typing.Callable, sigma_func: typing.Callable, **kwargs):
        super().__init__(sigma_func=sigma_func, **kwargs)
        self.denoiser = denoiser

    def U(self, x: torch.Tensor, t: torch.Tensor, sum_batch=True, **kwargs):
        x, t = x.to(torch.get_default_dtype()), t.to(torch.get_default_dtype())
        with torch.no_grad(), torch.autocast(device_type=x.device.type, dtype=config.amp_dtype, enabled=config.amp):
            x_denoised, _ = self.denoiser(x=x.detach(), t=t)

        if x_denoised.isnan().any():
            warnings.warn(f"Denoiser returned {x_denoised.isnan().sum()} nans.")

        x_diff_scaled = (x - x_denoised) / self.sigma_func(t)
        if self.corr_noiser:
            x_diff_scaled = self.corr_noiser.C_inv_R @ x_diff_scaled.double()
            x_diff_scaled = x_diff_scaled.to(x.dtype)

        if sum_batch:
            U = x_diff_scaled.square().sum() / 2
        else:
            U = x_diff_scaled.square().sum(dim=[1, 2]) / 2

        return U


class PositionConditioner(Conditioner):
    """
    Restrains a set of atom positions to a set of given target atom positions
    """

    def __init__(self, target_pos: torch.Tensor, target_ind: torch.Tensor, atoms_rep_ind: torch.Tensor,
                 expand_force_to_elements=False, **kwargs):
        """
        Restrains a set of atom positions to a set of given target atom positions
        Args:
            target_pos: target positions. shape=(B,N,3)
            target_ind: indices of the atoms whose positions are restrained. shape=(B,n_target,1) or shape=(B,n_target)
            expand_force_to_elements: copies the force of targets to all elements using forces on their representative
            atoms_rep_ind: index from 0 to n_target-1 that indicates the representative of each atom. shape=(B,N,1)
            **kwargs:
        """
        assert target_pos.shape[:2] == target_ind.shape[:2], "The target pos and index have different shapes."
        if target_ind.ndim == 2:
            target_ind = target_ind.unsqueeze(-1)
        elif target_ind.ndim == 3 and target_ind.shape[-1] != 1:
            raise ValueError(f"Unexpected shape for target_ind: {target_ind.shape}")
        super().__init__(**kwargs)
        self.target_pos = nn.parameter.Buffer(target_pos.double(), persistent=False)
        self.target_ind = nn.parameter.Buffer(target_ind.expand(-1, -1, 3), persistent=False)  # shape=(B,n_targets,3)
        self.atoms_rep_ind = nn.parameter.Buffer(atoms_rep_ind.long(), persistent=False)  # shape=(B,N,1)
        self.expand_force_to_elements = expand_force_to_elements
        self._target_pos = None
        self._target_ind = None
        self._atoms_rep_ind = None

        # Define the prior covariance for each target positions
        if self.corr_noiser is not None:
            if self.expand_force_to_elements:
                C_inv_R = self.corr_noiser.C_inv_R
            else:
                # Define prior only for target atoms
                mask_1D_ind = self.target_ind[0, :, 0]
                C = self.corr_noiser.C[mask_1D_ind[:, None], mask_1D_ind[None, :]]
                C_inv_R = torch.linalg.cholesky(torch.linalg.inv(C), upper=True)  # upper since we want C^-1 = R.T @ R
            self.C_prior_inv_R = nn.parameter.Buffer(C_inv_R, persistent=False)
        else:
            self.C_prior_inv_R = nn.parameter.Buffer(torch.tensor(1.0), persistent=False)

    def gather_target_vec(self, atoms_vec):
        target_vec = torch.gather(atoms_vec, dim=-2, index=self.target_ind)
        return target_vec

    def apply_batch_mask(self, mask: torch.Tensor):
        if self._target_pos is None:
            # Save original tensors
            self._target_pos = self.target_pos.data
            self._target_ind = self.target_ind.data
            self._atoms_rep_ind = self.atoms_rep_ind.data
        self.target_pos.data = self._target_pos[mask, ...]
        self.target_ind.data = self._target_ind[mask, ...]
        self.atoms_rep_ind.data = self._atoms_rep_ind[mask, ...]

    def translate_cluster_to_target(self, x):
        cluster_center = torch.nan * torch.ones_like(self.target_pos)
        cluster_center.index_reduce_(source=x, index=self.atoms_rep_ind[0, :, 0], dim=-2, reduce='mean',
                                     include_self=False)
        target_dx = self.target_pos - cluster_center
        dx = self.expand_target_vec_to_elements(target_dx)
        y = x + dx
        return y

    def expand_target_vec_to_elements(self, target_vec: torch.Tensor):
        # Expand vectors of the target atoms to all atoms associated with that target
        elements_target_vec = torch.gather(target_vec, dim=-2, index=self.atoms_rep_ind.expand(-1, -1, 3))
        return elements_target_vec

    @U_deactivate_decorator
    def U(self, x: torch.Tensor, t: torch.Tensor, sum_batch=True, *args, **kwargs):
        sigma = self.sigma_func(t)

        if self.expand_force_to_elements:
            x_target = self.expand_target_vec_to_elements(self.target_pos)
            target_pos_diff = x - x_target
        else:
            target_pos_diff = self.gather_target_vec(x) - self.target_pos

        if self.C_prior_inv_R.nelement() > 1:
            target_pos_diff = self.C_prior_inv_R @ target_pos_diff
        else:
            target_pos_diff = self.C_prior_inv_R * target_pos_diff
        squared_diff = (target_pos_diff / sigma).square()

        if sum_batch:
            U = squared_diff.sum() / 2
        else:
            U = squared_diff.reshape(x.shape[0], -1).sum(-1) / 2

        return U


class DistBoundsConditioner(Conditioner):
    """
    Conditioner that restrains pairwise distances be within given bounds (0 and inf bounds allowed).
    """

    def __init__(self, dist_bounds: torch.Tensor = None, dist_pair_ind: torch.Tensor = None,
                 dist_DOF: torch.Tensor | int = None, sigma_prior: torch.Tensor = None, pos_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.pos_scale = pos_scale

        # Define parameters to calculate the posterior distribution of the denoised coordinates
        if dist_bounds is not None:
            if dist_bounds.shape[0] != 2:
                raise ValueError(f"Expecting distance bounds with shape=(2,N). Input shape={dist_bounds.shape}")
            if not torch.all(torch.diff(dist_bounds, dim=0) > 0):
                raise ValueError('Upper bound must be > than the lower bound')

            # Add batch dimension to dist_bounds if the input tensor does not have one
            if dist_bounds.ndim < 3:
                dist_bounds = dist_bounds.unsqueeze(-2)

            self.dist_bounds = nn.parameter.Buffer(dist_bounds, persistent=False)

        if dist_pair_ind is not None:
            self.dist_pair_ind = nn.parameter.Buffer(dist_pair_ind, persistent=False)
        else:
            self.dist_pair_ind = None

        if dist_DOF is None:
            if self.corr_noiser is None:
                dist_DOF = torch.tensor(1)
            else:
                dist_DOF = torch.tensor(2)
        self.dist_DOF_sqrt = nn.parameter.Buffer(torch.as_tensor(dist_DOF).sqrt(), persistent=False)

        if self.corr_noiser is None:
            if sigma_prior is None:
                sigma_prior = torch.tensor(1 / math.sqrt(3)) / pos_scale  # Define prior such that x_0 has R_g=1 nm.
            self.sigma_prior = nn.parameter.Buffer(sigma_prior, persistent=False)

    def calculate_dist_bounds(self, *args, **kwargs):
        return self.dist_bounds

    def calculate_dist_pair_ind(self, *args, **kwargs):
        return self.dist_pair_ind

    def calculate_dist_prior_std(self, pair_ind: torch.Tensor, **kwargs):
        # Defines the prior's sigma for each given conditioned pairwise distance
        if self.corr_noiser is not None:
            C_conv_factor = (self.corr_noiser.pos_scale / self.pos_scale).square()
            C = self.corr_noiser.C * C_conv_factor
            pdist_var_prior = (C[pair_ind[:, 0], pair_ind[:, 0]] + C[pair_ind[:, 1], pair_ind[:, 1]]
                               - 2 * C[pair_ind[:, 0], pair_ind[:, 1]])

            # Check for zero values of std
            n_zeros_std = pdist_var_prior.le(0).sum()
            if n_zeros_std > 0:
                raise ValueError(f"{n_zeros_std} pairwise distances have a <=0 std.")
            pdist_sigma_prior = pdist_var_prior.sqrt()
        else:
            pdist_sigma_prior = self.dist_DOF_sqrt

        return pdist_sigma_prior

    def calculate_distances(self, atom_positions: torch.Tensor, dist_pair_ind: torch.Tensor, *args, **kwargs):
        """
        Calculates distances of given atom pairs using a batch of atom positions
        Args:
            atom_positions: batch of sets of atom positions. shape=(B,N,3)
            dist_pair_ind: indices of atom pairs for each conditioner distance. shape=(N_pairs,2)

        Returns:
            distances: shape=(B,M) where M is the total number of distances bounded.
        """
        pair_vec = torch.diff(atom_positions[..., dist_pair_ind, :], dim=-2).squeeze(-2)
        dist = torch.linalg.norm(pair_vec, dim=-1)
        return dist

    @U_deactivate_decorator
    def U(self, x: torch.Tensor, t: torch.Tensor, *args, t_ind: torch.Tensor = None, sum_batch=True, **kwargs):
        dist_pair_ind = self.calculate_dist_pair_ind(**kwargs)
        dist = self.calculate_distances(x, dist_pair_ind=dist_pair_ind, **kwargs)
        dist_bounds = self.calculate_dist_bounds(**kwargs)
        dist_prior_std = self.calculate_dist_prior_std(dist_pair_ind)
        if t.ndim != dist.ndim:
            t = t.reshape([t.numel()] + (dist.ndim - 1) * [1])
        sigma = self.sigma_func(t)

        # Define the parameters of the distance (d_ij) cdf, i.e. p(d_ij < C) = noncentral_chi_cdf(B,Lambda,3)
        if self.corr_noiser is None:
            # Assume the prior distribution has a constant uniform covariance:
            # p(x_0) = N(x_0, 0, sigma_0^2 * I)
            # using,
            # p(x_t|x_0) = N(x_t; x_0, sigma_t^2 I)
            # we have: p(x_0|x_t) ~= p(x_t|x_0) * p(x_0) = N(x_0; mu_t, Sigma_t)
            #
            # A_t = sigma_0^2/(sigma_t^2+sigma_0^2)
            # mu_t = A_t * x_t
            # Sigma_t = sigma_t^2 * A_t * I
            A = 1 / (1 + (sigma / self.sigma_prior).square())
        else:
            # Assume the prior distribution (p(x_0)) has the same covariance as the noiser kernel, i.e. Sigma = R R^T:
            # p(x_0) = N(x_0,0,R R^T).
            # Using, p(x_t|x_0) = N(x_t; x_0, sigma_t^2 R R^T), we have:
            # p(x_0|x_t) = N(x_0; mu_t, Sigma_t)
            #
            # A_t = 1/(1 + sigma_t^2)
            # mu_t = A_t * x_t
            # Sigma_t = sigma_t^2 * A_t * Sigma
            A = 1 / (1 + sigma.square())
        dist_mu = A * dist
        dist_sigma = dist_prior_std * A.sqrt() * sigma
        NonCentralChi_lims = dist_bounds / dist_sigma
        NonCentralChi_lambda = dist_mu / dist_sigma

        logcdf = NonCentralChicdf(Lambda=NonCentralChi_lambda, k=3, log_output=True,
                                  x_lower=NonCentralChi_lims[0], x_upper=NonCentralChi_lims[1])

        if sum_batch:
            U = -logcdf.sum()
        else:
            U = -logcdf.reshape(logcdf.shape[0], -1).sum(-1)

        if not torch.isfinite(U).all():
            n_nonfinite = (~torch.isfinite(logcdf)).sum()
            warnings.warn(f"{n_nonfinite} nans or infs found in {self.__class__.__name__} energy.\n")
        return U

    def plot_gradient_profile(self, t: torch.Tensor, ind=0, dist_max=None):
        assert self.dist_bounds is not None, f"dist_bounds must be define to plot gradient profile of {self.__class__.__name__}"
        assert self.dist_pair_ind is not None, f"dist_pair_ind must be define to plot gradient profile of {self.__class__.__name__}"
        if dist_max is None:
            dist_max = 2 * self.dist_bounds[1, :, ind].max()
        dist = torch.linspace(self.dist_bounds[0, :, ind].min() / 2, dist_max, 1000, requires_grad=True)
        t = t.reshape(-1, 1)
        sigma = self.sigma_func(t)
        sig_post = sigma * self.calculate_dist_prior_std(self.dist_pair_ind[ind])
        NonCentralChi2_lambda = dist / sig_post
        NonCentralChi2_lims = self.dist_bounds / sig_post
        logcdf = NonCentralChicdf(Lambda=NonCentralChi2_lambda, k=3, log_output=True,
                                  x_lower=NonCentralChi2_lims[0], x_upper=NonCentralChi2_lims[1])

        import matplotlib.pyplot as plt
        for i in range(t.nelement()):
            # Calculate gradients
            dist.grad = None
            Energy = -logcdf[i].sum()
            Energy.backward(retain_graph=True)

            plt.figure()
            plt.title(rf"t={t[i].item()}, $\sigma$={sigma[i].item()}")
            plt.plot(dist.detach().numpy(), -dist.grad.numpy())
            plt.xlabel('d')
            plt.ylabel(r'$\nabla$cdf(d)')
            ax = plt.gca()
            if self.dist_bounds.shape[1] == 1:
                plt.vlines(self.dist_bounds[0, 0, ind], *ax.get_ylim(), linestyles='--', color='r', label='lower bound')
                plt.vlines(self.dist_bounds[1, 0, ind], *ax.get_ylim(), linestyles='--', color='g', label='upper bound')
            plt.legend()


class CoarsePosConstraintConditioner(Conditioner):
    def __init__(self, pos_target: torch.Tensor, mask: torch.Tensor = None, ind: torch.Tensor = None,
                 mask_shape: tuple = None, **kwargs):
        super().__init__(**kwargs)
        is_batched = pos_target.ndim > 2
        if not is_batched:
            pos_target = pos_target.unsqueeze(dim=0)
        cons_kwargs = dict(device=pos_target.device, dtype=torch.double)
        if mask is not None:
            if mask.ndim > 2:
                if not torch.all(mask == mask[0]):
                    raise NotImplementedError(f'Dissimilar batched masks are not implemented')
                mask = mask[0]
            mask = mask.flatten()
        elif ind is not None:
            if ind.ndim > 2:
                if not torch.all(ind == ind[0]):
                    raise NotImplementedError(f'Dissimilar batched masks are not implemented')
                ind = ind[0]
            ind = ind.flatten()
            mask = torch.zeros(mask_shape, device=pos_target.device, dtype=torch.bool)
            mask[ind] = True

            # Reorder the position target using the given index so that they match the order of the flattened mask
            sorting_ind = torch.argsort(ind)
            pos_target = pos_target[:, sorting_ind]
        else:
            raise ValueError(f"mask or ind and mask_shape arguments must be given.")
        mask_3D_flat = mask.reshape(-1, 1).expand((-1, 3)).flatten(-2, -1)
        pos_target_flat = pos_target.to(cons_kwargs['dtype']).flatten(-2, -1)
        self.pos_target = nn.parameter.Buffer(pos_target, persistent=False)
        self.mask = nn.parameter.Buffer(mask, persistent=False)

        # Define the mean and covariance of the non C-alpha atoms (unconditioned variables)
        if self.corr_noiser is not None:
            R_3D = self.corr_noiser.R_3D
        else:
            R_3D = torch.eye(mask_3D_flat.shape[-1], **cons_kwargs)
        C_3D = R_3D @ R_3D.T
        mu_3D = torch.zeros(C_3D.shape[-1], **cons_kwargs)
        if is_batched:
            C_3D = C_3D.unsqueeze(0)
            mu_3D = mu_3D.unsqueeze(0)
        cond_mu, cond_covar = mvn_cond_params(mu=mu_3D, C=C_3D, cond_mask=mask_3D_flat, cond_values=pos_target_flat)
        cond_R = torch.linalg.cholesky(cond_covar)

        # Construct the full mean and Cholesky factorization R
        R_full = torch.zeros_like(C_3D)
        R_full[..., torch.outer(~mask_3D_flat, ~mask_3D_flat)] = cond_R.flatten(-2, -1)
        mu_full = torch.zeros(cond_mu.shape[0], C_3D.shape[-1], **cons_kwargs)
        mu_full[..., ~mask_3D_flat] = cond_mu
        mu_full[..., mask_3D_flat] = pos_target_flat

        # Define parameters for the linear transform
        A, b = R_full @ R_3D.inverse(), mu_full
        self.A = nn.parameter.Buffer(A, persistent=False)
        self.b = nn.parameter.Buffer(b, persistent=False)
        self.A_is_diag = torch.all(torch.diag_embed(A.diagonal(dim1=-2, dim2=-1)) == A)

        # Define the prior sigma for each CA positions
        if self.corr_noiser is not None:
            constrained_ind = mask.nonzero().squeeze(-1)
            C_cons = self.corr_noiser.C[constrained_ind[:, None], constrained_ind[None, :]]
            C_inv_R = torch.linalg.cholesky(torch.linalg.inv(C_cons), upper=True)  # upper since we want C^-1 = R.T@R
            self.C_prior_inv_R = nn.parameter.Buffer(C_inv_R, persistent=False)
        else:
            self.C_prior_inv_R = nn.parameter.Buffer(torch.tensor(1.0), persistent=False)

    def transform_x(self, x: torch.Tensor, t: torch.Tensor = None):
        x_flat = x.flatten(-2, -1)
        if self.A_is_diag:
            y_flat = self.A.diagonal(dim1=-2, dim2=-1) * x_flat + self.b
        else:
            y_flat = torch.squeeze(self.A @ x_flat.unsqueeze(-1), -1) + self.b
        y = y_flat.reshape(x.shape)
        return y

    @U_deactivate_decorator
    def U(self, x: torch.Tensor, t: torch.Tensor, sum_batch=True, *args, **kwargs):
        # Return energy of value 0. This is needed to ensure U.backward() does not error out
        if sum_batch:
            U = x.sum() * 0.0
        else:
            U = x.sum(dim=[1, 2]) * 0.0
        return U

        x_CA_pos = x[..., self.mask, :]
        pos_diff_scaled = (x_CA_pos - self.pos_target) / self.sigma_func(t)
        if self.C_prior_inv_R.nelement() > 1:
            pos_diff_scaled = self.C_prior_inv_R @ pos_diff_scaled
        else:
            pos_diff_scaled = self.C_prior_inv_R * pos_diff_scaled

        U = pos_diff_scaled.square().sum() / 2
        return U


class CoarsePositionConditioner(PositionConditioner):
    def __init__(self, coarse_pos_target: torch.Tensor, coarse_atom_ind: torch.Tensor, nodes_coarse_ind: torch.Tensor,
                 corr_noiser=None, **kwargs):
        if corr_noiser is not None:
            raise NotImplementedError("Remove coarse positions in correlated noiser.")

        super().__init__(target_pos=coarse_pos_target, target_ind=coarse_atom_ind, atoms_rep_ind=nodes_coarse_ind,
                         corr_noiser=corr_noiser, **kwargs)


class CoarseCenterDistConditioner(DistBoundsConditioner):
    def __init__(self, coarse_grain_pos: torch.Tensor, coarse_grain_radius: torch.Tensor,
                 coarse_grain_ind: torch.Tensor, precision=0.0, **kwargs):
        coarse_grain_ind = coarse_grain_ind.squeeze(-1)
        if torch.all(coarse_grain_ind == coarse_grain_ind[0]):
            coarse_grain_ind = coarse_grain_ind[0]
        else:
            raise NotImplementedError(f"Batched coarse grain indices is not implemented.")

        upper_bounds = coarse_grain_radius[0, coarse_grain_ind].squeeze().unsqueeze(0)
        lower_bounds = torch.zeros_like(upper_bounds)

        # Increase the upper bounds to reflect the requested precision of the coarse pos.
        # A precision of delta in a 3D position implies a precision of sqrt(3)*delta in the distance.
        upper_bounds += math.sqrt(3) * precision

        dist_bounds = torch.stack([lower_bounds, upper_bounds])
        super().__init__(dist_bounds=dist_bounds, dist_DOF=1, **kwargs)
        self.coarse_pos = nn.parameter.Buffer(coarse_grain_pos, persistent=False)
        self.coarse_grain_ind = nn.parameter.Buffer(coarse_grain_ind, persistent=False)
        self._coarse_pos = None
        if self.corr_noiser is not None:
            raise NotImplementedError(f"Change sigma_prior according to the correlated noiser.")

    def calculate_distances(self, atom_positions: torch.Tensor, **kwargs):
        pos_centered = atom_positions - self.coarse_pos[..., self.coarse_grain_ind, :]
        dist_to_center = torch.linalg.norm(pos_centered, dim=-1)
        return dist_to_center

    def apply_batch_mask(self, mask: torch.Tensor):
        if self._coarse_pos is None:
            self._coarse_pos = self.coarse_pos.data
        self.coarse_pos.data = self._coarse_pos[mask, ...]


class BondLengthBoundsConditioner(DistBoundsConditioner):
    def __init__(self, top_filepath: str, dataset_stats: Statistics = None,
                 bounds_sigma=config.sampling_cond_bounds_sig, pos_scale: float | torch.Tensor = 1.0,
                 element_class=heavy_atoms_class_name, **kwargs):
        bond_energy_module = BondEnergy(top_file=top_filepath, element_class=element_class)
        bonds_atom_ind = bond_energy_module.bonds_atom_ind
        if dataset_stats is not None:
            lower_bounds = dataset_stats.mean - bounds_sigma * dataset_stats.std
            upper_bounds = dataset_stats.mean + bounds_sigma * dataset_stats.std
            lower_bounds = torch.minimum(lower_bounds, dataset_stats.min - dataset_stats.std)
            upper_bounds = torch.maximum(upper_bounds, dataset_stats.max + dataset_stats.std)
            dist_bounds = torch.stack([lower_bounds, upper_bounds], dim=0)
        else:
            # Define the bounds from the bond energy parameters
            bond_length_std = bond_energy_module.bond_length_std().unsqueeze(0)
            bond_length_mean = bond_energy_module.b0.data.unsqueeze(0)
            dist_bounds = bond_length_mean + torch.tensor([-1, 1]).reshape(-1, 1) * bounds_sigma * bond_length_std

        # Bounds are given in nm. Convert dist bounds to the given scale.
        dist_bounds = dist_bounds / pos_scale

        super().__init__(dist_bounds=dist_bounds, dist_pair_ind=bonds_atom_ind, pos_scale=pos_scale, **kwargs)


class BondAngleEnddistBoundsConditioner(DistBoundsConditioner):
    def __init__(self, top_filepath: str, dataset_stats: Statistics = None,
                 bounds_sigma=config.sampling_cond_bounds_sig, pos_scale: float | torch.Tensor = 1.0,
                 element_class=heavy_atoms_class_name, **kwargs):
        bond_angle_energy_module = BondAngleEnergy(top_file=top_filepath, element_class=element_class)
        pdist_atom_ind = bond_angle_energy_module.angles_atom_ind[:, [0, -1]]
        if dataset_stats is not None:
            lower_bounds = dataset_stats.mean - bounds_sigma * dataset_stats.std
            upper_bounds = dataset_stats.mean + bounds_sigma * dataset_stats.std
            lower_bounds = torch.minimum(lower_bounds, dataset_stats.min - dataset_stats.std)
            upper_bounds = torch.maximum(upper_bounds, dataset_stats.max + dataset_stats.std)
            dist_bounds = torch.stack([lower_bounds, upper_bounds], dim=0)
        else:
            raise NotImplementedError

        # Bounds are given in nm. Convert dist bounds to the given scale.
        dist_bounds = dist_bounds / pos_scale

        super().__init__(dist_bounds=dist_bounds, dist_pair_ind=pdist_atom_ind, pos_scale=pos_scale, **kwargs)


class DynamicDistBoundsConditioner(DistBoundsConditioner):
    def __init__(self, *args, dist_bounds: torch.Tensor, **kwargs):
        """
        Defines a dynamic distance bounds conditioner that conditions distances defined in an adjacency matrix.
        Args:
            dist_bounds: distances bounds. Matrix of shape=(2,N,N) where N=number of atoms or floats of shape (2,)
        """
        super().__init__(*args, **kwargs)
        if not torch.all(torch.diff(dist_bounds, dim=0) > 0):
            raise ValueError('Upper bound must be > than the lower bound')
        if dist_bounds.ndim < 2:
            dist_bounds = dist_bounds.reshape(-1, 1)
        self.dist_bounds = nn.parameter.Buffer(dist_bounds, persistent=False)

    def calculate_dist_pair_ind(self, adj_pair_ind: torch.Tensor, **kwargs):
        return adj_pair_ind

    def calculate_dist_bounds(self, *args, adj_pair_ind: torch.Tensor, **kwargs):
        """
        Define the bounds of each pair by looking the bounds in the bounds matrix using the pair ind
        Args:
            adj_pair_ind: atom indices of each pair
            **kwargs:

        Returns:

        """
        if self.dist_bounds.ndim == 3:
            return self.dist_bounds[:, adj_pair_ind[:, 0], adj_pair_ind[:, 1]]
        else:
            return self.dist_bounds

    def calculate_distances(self, atom_positions: torch.Tensor, dist_pair_ind: torch.Tensor,
                            adj_pair_batch_ind: torch.Tensor = None, **kwargs):
        """
        Calculates distances of given atom pairs using a batch of atom positions
        Args:
            atom_positions: batch of sets of atom positions. shape=(B,N,3)
            dist_pair_ind: indices of atom pairs for each conditioner distance. shape=(N_pairs,2)
            adj_pair_batch_ind: batch index of each given pairs. shape=(N_pairs,1)
        Returns:
            distances: shape=(B,M) where M is the total number of distances bounded.
        """
        pair_vec = torch.diff(atom_positions[adj_pair_batch_ind, dist_pair_ind, :], dim=-2).squeeze(-2)
        dist = torch.linalg.norm(pair_vec, dim=-1)
        return dist

    def U(self, x: torch.Tensor, t: torch.Tensor, adj_mat: torch.Tensor = None, sum_batch=True, **kwargs):
        pair_ind_3D = adj_mat.triu(diagonal=1).nonzero()
        adj_pair_batch_ind, adj_pair_ind = pair_ind_3D[..., :1], pair_ind_3D[..., 1:]
        if t.ndim > 0 and t.shape[0] == x.shape[0]:
            t = t[adj_pair_batch_ind]

        U = super().U(x=x, t=t, adj_pair_batch_ind=adj_pair_batch_ind, adj_pair_ind=adj_pair_ind, sum_batch=False,
                      **kwargs)
        if sum_batch:
            U = U.sum()
        else:
            U = x.new_zeros((x.shape[0],)).index_add_(dim=0, index=adj_pair_batch_ind[:, 0], source=U)

        return U


class GlobalLJDistMinConditioner(DistBoundsConditioner):
    def __init__(self, pdist_min: torch.Tensor | float, top_filepath: str, element_class=heavy_atoms_class_name,
                 **kwargs):
        LJ_module = LJEnergy(top_file=top_filepath, element_class=element_class)
        dist_pair_ind = LJ_module.pairs_ind.data
        dist_bounds = torch.tensor([pdist_min, torch.inf]).reshape(-1, 1)
        super().__init__(dist_bounds=dist_bounds, dist_pair_ind=dist_pair_ind, **kwargs)


class LocalLJDistMinConditioner(DynamicDistBoundsConditioner):
    def __init__(self, pdist_min: torch.Tensor | float, top_filepath: str, element_class=heavy_atoms_class_name,
                 **kwargs):
        LJ_module = LJEnergy(top_file=top_filepath, element_class=element_class)
        dist_pair_ind = LJ_module.pairs_ind.data
        LJ_dist_mask = torch.zeros((LJ_module.n_atoms, LJ_module.n_atoms), dtype=torch.bool)
        LJ_dist_mask[dist_pair_ind[:, 0], dist_pair_ind[:, 1]] = True
        dist_bounds = torch.tensor([pdist_min, torch.inf])
        super().__init__(dist_bounds=dist_bounds, dist_pair_ind=dist_pair_ind, **kwargs)
        self.LJ_dist_mask = nn.parameter.Buffer(LJ_dist_mask, persistent=False)

    def U(self, *args, adj_mat: torch.Tensor = None, **kwargs):
        adj_mat = self.LJ_dist_mask * adj_mat  # Mask adjacency matrices to make sure that only LJ pairs are restrained
        return super().U(*args, adj_mat=adj_mat, **kwargs)


class CoarseGrainAdjConditioner(DistBoundsConditioner):
    def __init__(self, coarse_grain_pos: torch.Tensor, coarse_grain_radius: torch.Tensor,
                 coarse_grain_ind: torch.Tensor, **kwargs):
        coarse_grain_ind = coarse_grain_ind.squeeze(-1)
        if coarse_grain_ind.ndim > 1:
            raise NotImplementedError(f"Batched coarse grain indices is not implemented.")

        B, N = coarse_grain_pos.shape[:2]
        pair_ind = torch.triu_indices(N, N, offset=1).T
        coarse_grain_pdist = coarse_grain_pos[:, pair_ind, :].diff(dim=-2).squeeze(dim=-2).norm(dim=-1)
        adj_dist = coarse_grain_radius[:, pair_ind].squeeze(-1).sum(dim=-1)
        are_coarse_grain_adj = coarse_grain_pdist < adj_dist

        # Impose upper bounds on coarse grains that are adjacent and lower bounds on those that aren't
        dist_bounds = torch.ones((1, *coarse_grain_pdist.shape), device=coarse_grain_pos.device)
        dist_bounds = dist_bounds * torch.tensor([-torch.inf, torch.inf], device=coarse_grain_pos.device).reshape(-1, 1,
                                                                                                                  1)
        dist_bounds[0, ~are_coarse_grain_adj] = adj_dist[~are_coarse_grain_adj]
        dist_bounds[1, are_coarse_grain_adj] = adj_dist[are_coarse_grain_adj]

        super().__init__(dist_bounds=dist_bounds, dist_pair_ind=coarse_grain_ind[pair_ind], **kwargs)


if __name__ == '__main__':
    # # Test DistBoundsConditioner with correlated noiser
    # from models.diffusers import CorrelatedNoiser
    # from data.datasets import load_dataset
    #
    # dataset = load_dataset()
    # corr_noiser = CorrelatedNoiser(topology=dataset.top_filepath, element_class=heavy_atoms_class_name,
    #                                chain_scaling=None, centroid_std=1.0)
    # cond = DistBoundsConditioner(dist_bounds=torch.tensor([0.0, 2.0]).reshape(-1, 1),
    #                              dist_pair_ind=torch.tensor([1, 2]).reshape(1, 2),
    #                              sigma_func=lambda x: x, corr_noiser=corr_noiser)
    # x = torch.randn((1, 20, 3))
    # cond_U = cond.U(x=x, t=torch.tensor(1))
    # pass

    # # Test LocalDistBoundsConditioner
    # B, N = 4, 30
    # x = torch.randn((B, N, 3))
    # adj_mat = torch.rand((B, N, N)) < 0.5
    # # dist_bounds = torch.tensor([-1, 1]).reshape(2, 1)
    # dist_bounds = torch.rand((2, N, N))
    # dist_bounds[1] = dist_bounds[0] + torch.rand((N, N))
    # cond = DynamicDistBoundsConditioner(dist_bounds=dist_bounds, sigma_func=lambda x: x)
    # U = cond.U(x, t=torch.tensor(1), adj_mat=adj_mat)
    # pass

    # # Test LocalLJDistMinConditioner
    # from data.datasets import load_dataset
    #
    # dataset = load_dataset(element_class=heavy_atoms_class_name)
    # mol_ex = dataset[0]
    # B, N = 4, mol_ex.n_elements
    # x = torch.randn((B, N, 3))
    # adj_mat = torch.rand((B, N, N)) < 0.5
    # cond = LocalLJDistMinConditioner(pdist_min=2, top_filepath=dataset.top_filepath, sigma_func=lambda x: x)
    # U = cond.U(x, t=torch.tensor(1), adj_mat=adj_mat)
    pass