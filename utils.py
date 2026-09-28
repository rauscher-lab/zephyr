import copy
import functools
import math
import os
import pathlib
import re
import tempfile
import time
import itertools
import typing
import warnings
import bisect
import subprocess
import shutil
import inspect
import random
from collections import defaultdict
from collections.abc import Iterable
import multiprocessing
import datetime

import numpy as np
import scipy.special
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, Sampler
import MDAnalysis as mda
import MDAnalysis.transformations
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.collections import PolyCollection
import pint
import tqdm

pint_reg = pint.get_application_registry()
if 'Angstrom' not in pint_reg:
    pint_reg.define('@alias angstrom = Angstrom')

# On SLURM-based system use the 'SLURM_CPUS_PER_TASK' environment variable to define the number of parallel processes.
if 'SLURM_CPUS_PER_TASK' in os.environ:
    n_parallel_processes = int(os.environ['SLURM_CPUS_PER_TASK'])
else:
    n_parallel_processes = None

data_dir = pathlib.Path(__file__).parent / 'data'
LOG_2 = 0.6931471805599453
force_compile = False


################################################## ARCHITECTURE ##################################################
class Positive(nn.Module):
    def forward(self, x):
        return x.exp()


def index_softmax(src: torch.Tensor, dim: int, index: torch.Tensor, size: tuple):
    # Remove max for each group to improve numerical stability
    src_max = torch.zeros(size, dtype=src.dtype, device=src.device)
    src_max.index_reduce_(dim=dim, index=index, source=src, reduce='amax', include_self=False)
    src_exp = torch.exp(src - src_max[index])

    # Perform softmax over each group
    src_exp_sum = torch.zeros(size, dtype=src_exp.dtype, device=src.device)
    src_exp_sum.index_add_(dim=dim, source=src_exp, index=index)
    output = src_exp / src_exp_sum[index]
    return output


################################################## MATH ##################################################
class RigidBodyMotion:
    def __init__(self, O: torch.Tensor = None, t: torch.Tensor = None, dtype=None):
        if isinstance(O, torch.Tensor) and O.ndim == 2:
            O = O.unsqueeze(dim=0)
        if isinstance(t, torch.Tensor):
            if t.ndim == 1:
                t = t.unsqueeze(dim=0)
            if t.shape[-1] != 1:
                t = t.unsqueeze(dim=-1)

        if O is not None:
            batch_shape = O.shape[:-2]
            dtype = O.dtype if dtype is None else dtype
        elif t is not None:
            batch_shape = t.shape[:-2]
            dtype = t.dtype if dtype is None else dtype
        else:
            batch_shape = torch.Size((1,))
            dtype = torch.get_default_dtype() if dtype is None else dtype

        if t is None:
            t = torch.zeros(size=(*batch_shape, 3, 1)).to(dtype)
        if O is None:
            O = torch.eye(3).expand(size=(*batch_shape, 3, 3)).to(dtype)

        self.O = O
        self.t = t
        self.shape = batch_shape
        self.n = self.shape.numel()

    def __matmul__(self, other: typing.Self):
        O = torch.matmul(self.O, other.O)
        t = torch.matmul(self.O, other.t) + self.t
        return self.__class__(O=O, t=t)

    def __add__(self, other: typing.Self):
        O = self.O + other.O
        t = self.t + other.t
        return self.__class__(O=O, t=t)

    def __sub__(self, other):
        O = self.O - other.O
        t = self.t - other.t
        return self.__class__(O=O, t=t)

    def __repr__(self):
        shape = self.O.shape[:-2]
        return f"{self.__class__.__name__}(shape={tuple(shape)})"

    def __eq__(self, other):
        norm = (self - other).norm()
        is_norm_zero = torch.isclose(torch.zeros_like(norm), other=norm, atol=1e-5, rtol=0).all().item()
        return is_norm_zero

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item,)
        return self.__class__(O=self.O[*item, ..., :, :], t=self.t[*item, ..., :, :])

    def reshape(self, shape: tuple):
        O = self.O.reshape(*shape, 3, 3)
        t = self.t.reshape(*shape, 3, 1)
        return self.__class__(O=O, t=t)

    def unsqueeze(self, **kwargs):
        O = self.O.unsqueeze(**kwargs)
        t = self.t.unsqueeze(**kwargs)
        return self.__class__(O=O, t=t)

    @property
    def inv(self):
        O = self.O.transpose(dim0=-2, dim1=-1)  # equivalent to O^-1 since O is orthogonal
        t = - torch.matmul(O, self.t)
        return self.__class__(O=O, t=t)

    def mean(self, dim=-1, **kwargs):
        if dim < 0:
            dim = len(self.shape) + dim
        O = SOn_proj(self.O.mean(dim=dim, **kwargs))
        t = self.t.mean(dim=dim, **kwargs)
        return self.__class__(O=O, t=t)

    def norm(self, type='euclidean'):
        O_norm2 = self.O.square().sum(dim=[-1, -2])
        t_norm2 = self.t.square().sum(dim=[-1, -2])
        norm2 = O_norm2 + t_norm2
        if type == 'euclidean':
            return norm2.sqrt()
        elif type == 'sqeuclidean':
            return norm2
        else:
            raise ValueError(f"type={type} is not supported")

    def align(self, other: typing.Self):
        """
        Globally aligns frames with another of frames by translating and rotating their translation vectors
        Args:
            other: target set of frames
        Returns:
            Frame: aligned set of frames
        """
        self_mean = self.mean(keepdim=True)
        other_mean = other.mean(keepdim=True)
        return other_mean @ self_mean.inv @ self
        # self_mean_inv = self_mean.inv
        # self_inv_mean = self.inv.mean(keepdim=True)
        # # self_undo = self.__class__(t=self_mean_inv.t) @ self.__class__(O=self_mean_inv.O)
        # s = self @ self_inv_mean
        # s = self.__class__(t=s.mean().inv.t) @ s
        # self_aligned = other_mean @ s
        # return self_aligned
        #
        # global_O_trans = SOn_proj((other.O @ self.O.transpose(-1, -2)).mean(dim=-3, keepdim=True))
        # self_aligned = self.__class__(O=global_O_trans) @ self
        # return self_aligned
        #
        # # Align the translation vectors
        # # self_t, target_t = self.t.squeeze(-1), other.t.squeeze(-1)
        # # dtype = self.t.dtype
        # # frame_vecs = self.__class__(t=torch.eye(3).to(dtype).reshape(3, *len(self.shape) * [1], 3), dtype=dtype)
        # # self_frame_vecs, target_frame_vecs = self @ frame_vecs, other @ frame_vecs  # shape (3,B,N)
        # self_t, target_t = self_aligned.t, other.t  # shape=(B,N,3,1)
        # self_centroid, target_centroid = self_t.mean(dim=-3, keepdim=True), target_t.mean(dim=-3, keepdim=True)
        # self_t_row_vecs = self_t - self_centroid
        # target_t_col_vecs = target_t - target_centroid
        # covar = torch.mean(target_t_col_vecs * self_t_row_vecs.transpose(-1, -2), dim=-3, keepdim=True)
        #
        # t_rot_mat = SOn_proj(covar)  # shape (B,N,3,3)
        # trans_t = target_centroid - torch.matmul(t_rot_mat, self_centroid)
        # trans = self.__class__(O=t_rot_mat, t=trans_t)
        # self_aligned = trans @ self_aligned
        #
        # # t_rot_mat = SOn_proj(covar).transpose(-1,-2)  # shape (B,N,3,3)
        # # trans_t = target_centroid - torch.matmul(t_rot_mat, self_centroid)
        # # trans = self.__class__(O=t_rot_mat, t=trans_t)
        # # self_aligned = self @ trans
        # return self_aligned

        # Fix the global orientation of the frames
        # average_O_func = lambda O: SOn_proj(O.mean(dim=-3, keepdim=True))
        # self_O_vectors = self_aligned.O.transpose(-1, -2).reshape(*self_aligned.O.shape[:-3], -1, 3)
        # target_O_vectors = other.O.transpose(-1, -2).reshape(*other.O.shape[:-3], -1, 3)
        # # self_aligned.O = torch.matmul(global_O_trans, self_aligned.O.transpose(-1, -2)).transpose(-1, -2)
        # self_O_vectors2 = rmsd_align(positions=self_O_vectors, positions_ref=target_O_vectors, remove_mean=False)
        # # global_O_trans = SOn_proj(torch.mean(target_O_vectors.unsqueeze(-1) * self_O_vectors.unsqueeze(-2), dim=-3)).unsqueeze(-3)
        # # self_O_vectors3 = (global_O_trans.squeeze(-3) @ self_O_vectors.transpose(-1, -2)).transpose(-1, -2)
        # self_aligned_O = self_O_vectors2.reshape(self_aligned.O.shape).transpose(-1, -2)
        # self_aligned.O = self_aligned_O

        # # self_global_O = average_O_func(self_aligned.O)
        # # target_global_O = average_O_func(other.O)

        # # # global_diff_O = SOn_proj(torch.matmul(target_global_O, self_global_O.transpose(-1, -2)))
        # # global_O_trans2 = SOn_proj(torch.mean(other.O @ self_aligned.O.transpose(-1, -2), -3, keepdim=True))
        # # self_aligned.O = torch.matmul(global_O_trans, self_aligned.O)
        # # # final_O_avg = average_O_func(self_aligned.O)
        #
        # return self_aligned

    def cat(self, other: typing.Self, **kwargs):
        O = torch.cat([self.O, other.O], **kwargs)
        t = torch.cat([self.t, other.t], **kwargs)
        return self.__class__(O=O, t=t)

    @classmethod
    def from_qt(cls, qt: torch.Tensor, normalize=False):
        """
        Constructs a set of frames using a unit quaternion for O and 3 scalars for t
        Reference: https://en.wikipedia.org/wiki/Rotation_matrix
        Args:
            qt: set of 7D vectors that define the quaternion and translation vector of each transformation
            normalize: normalize the quaternions before defining the orthogonal matrix

        Returns:
            Frame
        """
        # Form the rotation matrix from the quaternion representation
        q, t = qt[..., :4], qt[..., 4:]
        if normalize:
            q = q / (torch.linalg.norm(q, dim=-1, keepdim=True))
        w, x, y, z = q.unbind(dim=-1)
        w2, x2, y2, z2 = q.square().unbind(dim=-1)
        wx, wy, wz, xy, xz, yz = w * x, w * y, w * z, x * y, x * z, y * z
        O_rows = list()
        O_rows.append(torch.stack([1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy)], dim=-1))
        O_rows.append(torch.stack([2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx)], dim=-1))
        O_rows.append(torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)], dim=-1))
        O = torch.stack(O_rows, dim=-1).transpose(-1, -2)
        return cls(O=O, t=t)

    @classmethod
    def from_at(cls, at: torch.Tensor):
        """
        Constructs a set of frames using 3 angles for O and 3 scalars for t
        Args:
            qt: set of 6D vectors that define the angles of the unit quaternion and translation vector of each transformation
        Returns:
            Frame
        """
        # Form the rotation matrix from the quaternion representation
        a, t = at[..., :3], at[..., 3:]
        theta, phi, psi = [T.unsqueeze(-1) for T in a.unbind(-1)]
        c_theta, s_theta = torch.cos(theta), torch.sin(theta)
        c_phi, s_phi = torch.cos(phi), torch.sin(phi)
        c_psi, s_psi = torch.cos(psi), torch.sin(psi)
        qt = torch.cat([c_theta * c_phi, c_theta * s_phi, s_theta * c_psi, s_theta * s_psi, t], dim=-1)
        return cls.from_qt(qt=qt, normalize=False)

    @classmethod
    def random(cls, size=(100,), qt=False, rot=True, trans=True, **kwargs):
        if qt:
            qt = torch.randn((*size, 7), **kwargs)
            if not rot:
                qt[..., 1:4] = 0.0
            if not trans:
                qt[..., 4:] = 0.0
            return cls.from_qt(qt=qt, normalize=True)
        else:
            if rot:
                S = torch.randn((*size, 3, 3), **kwargs)
                S = S - S.transpose(-1, -2)
                O = torch.matrix_exp(S)
            else:
                O = torch.eye(3, **kwargs).expand(size=(*size, 3, 3))
            if trans:
                t = torch.randn((*size, 3), **kwargs)
            else:
                t = torch.zeros((*size, 3), **kwargs)
        return cls(O=O, t=t)

    def transform(self, positions: torch.Tensor, frames_ind: tuple[torch.Tensor, ...] = None):
        if frames_ind is None:
            frames_1D = self
        else:
            frames_1D = self[frames_ind]
        new_pos = (torch.matmul(frames_1D.O, positions.unsqueeze(-1)) + frames_1D.t).squeeze(-1)
        return new_pos


def factorint(n):
    """
    Native replacement of sympy.factorint().
    Args:
        n: integer

    Returns:
        dictionary of factors
    """
    if n <= 0:
        raise ValueError("factorint is only defined for positive integers")
    if n == 1:
        return {}

    factors = {}

    # 1. Pull out all factors of 2 quickly
    if n % 2 == 0:
        count = 0
        while n % 2 == 0:
            count += 1
            n //= 2
        factors[2] = count

    # 2. Pull out all factors of 3 quickly
    if n % 3 == 0:
        count = 0
        while n % 3 == 0:
            count += 1
            n //= 3
        factors[3] = count

    # 3. Trial division for remaining odd primes using 6k +/- 1 optimization
    d = 5
    while d * d <= n:
        # Check 6k - 1
        if n % d == 0:
            count = 0
            while n % d == 0:
                count += 1
                n //= d
            factors[d] = count

        # Check 6k + 1
        d_plus_2 = d + 2
        if n % d_plus_2 == 0:
            count = 0
            while n % d_plus_2 == 0:
                count += 1
                n //= d_plus_2
            factors[d_plus_2] = count

        d += 6

    # 4. If n is still greater than 1, then n itself is a prime factor
    if n > 1:
        factors[n] = 1

    return factors


def round_to_sig(x: np.ndarray, n: int):
    """
    Rounds an array of float to a given number of significant digits.
    Reference:
        Adapted from Scott Gigante's answer on:
        https://stackoverflow.com/questions/18915378/rounding-to-significant-figures-in-numpy

    Args:
        x: array of numbers to round
        n: number of significant digits

    Returns:
        y = rounded array
    """
    x_is_finite = np.isfinite(x)
    x_mag = np.where(x_is_finite & (x != 0), np.abs(x), 10 ** (n - 1))
    x_mag_sig = 10 ** (n - np.ceil(np.log10(x_mag)))
    y = np.where(x_is_finite, np.round(x * x_mag_sig) / x_mag_sig, x)
    return y


def symmetrize(x: torch.Tensor, dim0=-1, dim1=-2):
    return (x + x.transpose(dim0=dim0, dim1=dim1)) / 2


def remove_centroid(x: torch.Tensor, mask: torch.Tensor = None, inplace=False, return_centroid=False):
    """
    Removes centroid in sets of 3D positions
    Args:
        x: sets of  3D positions. shape= (B,N,3) or (N,3)
        mask: mask to indicate which positions are considered in calculating the center
        inplace: bool indicating whether to perform the operation inplace or not
        return_centroid: return the calculated center-of-mass
    Returns:
        x or None: When inplace=True, returns the new tensor x with center removed
    """
    if mask is None:
        x_centroid = torch.mean(x, dim=-2, keepdim=True)
    else:
        # Make sure the mask has the same dimensions has x to properly broadcast dimensions
        # If 1D mask is given, assume it's masking the second-to-last dimension
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)  # Add batch dimension

        # If DD mask is given, assume it's masking the batch and second-to-last dimensions
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)  # Add position dimension

        mask = mask.expand(*x.shape)
        x_masked = torch.where(mask, x, torch.zeros_like(x))
        x_centroid = x_masked.sum(-2, keepdim=True) / mask.sum(-2, keepdim=True)

    if inplace:
        x -= x_centroid
        if return_centroid:
            return x_centroid
    else:
        x = x - x_centroid
        if return_centroid:
            return x, x_centroid
        else:
            return x


def remove_rot(positions: torch.Tensor, remove_centroid=True, method='svd', fix_axes_sign=True):
    """
    Removes rotation degrees of freedom in a cloud of 3D points by rotating the cloud to its principal axes using SVD

    Args:
        positions: batch of point clouds of shape (B,N,3) or (N,3)
        remove_centroid: Remove centroid in output positions
        method: Method used to remove the rotation in the set of point clouds ('svd' or 'PCA' see below)
        fix_axes_sign: fixes the sign of the rotated output to ensure consistency of the projected axes
    Returns:
        rotated_positions: sets of points rotated such that the x,y,z axes has the largest-to-lowest variance
    """
    # Remove centroids
    pos_centroid = positions.mean(dim=-2, keepdim=True)
    positions_out = positions - pos_centroid

    if method == 'svd':
        # Calculate SVD decomposition for each point cloud. Equivalent to PCA.
        U, S, Vh = torch.linalg.svd(positions_out)

        # Find the rotation matrix that rotates each point to its principal axes disallowing for inversion
        V_det = Vh.det()
        Vh_mask = torch.ones_like(Vh)
        Vh_mask[..., -1, :] = V_det.unsqueeze(-1)
        R = (Vh_mask * Vh).transpose(-2, -1)
    elif method == 'PCA':
        # Remove rotations with PCA
        _, PCA_vec = torch.linalg.eigh(positions_out.transpose(-2, -1) @ positions_out)
        # eigh outputs eigval in ascending order. Need descending order to be consistent with svd
        PCA_vec = PCA_vec.flip(dims=[-1])
        # Remove inversions
        PCA_vec[..., :, -1] *= PCA_vec.det().unsqueeze(-1)
        R = PCA_vec
    else:
        raise NotImplementedError(f"method {method!r} is not implemented")

    positions_out = positions_out @ R

    # Fix axes signs
    if fix_axes_sign:
        # Method 1: Fix the product of the sign of all dimensions except the last one
        # dim_sign_flips = positions_out.sign().prod(dim=-2, keepdim=True)

        # Method 2: Ensure that the majority of the positions are on the positive side of each axis (except last one)
        dim_sign_flips = positions_out.sign().sum(dim=-2, keepdim=True).round().int().sign()
        # Edge case: If there is an equal amount of + and -, sign() will return 0.0. Overwrite to 1.0 (do not flip sign).
        dim_sign_flips[dim_sign_flips.eq(0)] = 1

        # For the last dimension, set the sign to be equal to the number of sign flips in the other (d-1) dimensions to
        # make sure that inversions are not introduced.
        dim_sign_flips[..., -1] = dim_sign_flips[..., :-1].prod(dim=-1)

        positions_out *= dim_sign_flips.to(positions_out.dtype)

    if not remove_centroid:
        positions_out += pos_centroid
    return positions_out


def SOn_proj(X: torch.Tensor, with_reflection=False):
    X_det_sign = X.det().sign()

    # Set the sign to 1 for cases where the determinant is zero
    X_det_sign_iszero = torch.isclose(X_det_sign, torch.zeros_like(X_det_sign))
    X_det_sign[X_det_sign_iszero] = 1.0

    U, D, Vh = torch.linalg.svd(X)
    if not with_reflection:
        Vh_mask = torch.ones_like(Vh)
        Vh_mask[..., -1, :] = X_det_sign.unsqueeze(-1)
        # Vh_mask = torch.stack([torch.ones_like(X_det_sign), torch.ones_like(X_det_sign), X_det_sign], dim=-1).unsqueeze(
        #     dim=-2)
        # Vh_mask2 = torch.nn.functional.pad(X_det_sign[..., None, None], pad=(Vh.shape[-1] - 1, 0), value=1.0)
        Vh = Vh_mask * Vh
    # if X_det_sign < 0:
    #     Vh[:, -1] = -Vh[:, -1]
    # S_diag = torch.ones((d,))
    # S_diag[-1] = xy_covar.det().sign()
    # S = torch.diag(S_diag)
    R = U @ Vh
    return R


def rmsd_align_transform(positions: torch.Tensor, positions_ref: torch.Tensor, reflect=True):
    """
    Finds the rigid motion (rotation + translation + reflection[optional]) that minimizes the rmsd of two sets of positions.

    ||y-(x*R+t)||^2 is minimized where x=input positions and y=reference positions
    R is a member of the SO(n) (reflect=False) or O(n) (reflect=True)
    Reference: Umeyama 1991, "Least-squares estimation of transformation parameters between two point patterns"
               https://en.wikipedia.org/wiki/Kabsch_algorithm

    Args:
        positions: set of positions with ndim=>=2. If ndim>2, the alignment for each set of points with the last 2 dimensions representing the point and 3D coordinate index.
        positions_ref: reference positions
        reflect: allow reflections of the positions if necessary

    Returns:
        R: torch.Tensor. shape=(B,3,3)
        t: torch.Tensor. shape=(B,3)
    """

    # Remove centroids
    pos_centroid = positions.mean(dim=-2, keepdim=True)
    pos_ref_centroid = positions_ref.mean(dim=-2, keepdim=True)
    positions = positions - pos_centroid
    positions_ref = positions_ref - pos_ref_centroid

    # Calculate covariance.
    # In Umeyama's formula, C = y.T @ x minimizes ||y-Rx||^2. We need C.T to minimize ||y-xR||^2
    covar = (positions_ref.transpose(-2, -1) @ positions) / positions_ref.shape[-2]
    # covar = torch.sum(positions.unsqueeze(-1) * positions_ref.unsqueeze(-2), dim=-3) / positions_ref.shape[-2]
    R = SOn_proj(covar.transpose(-2, -1), with_reflection=reflect)
    t = -pos_centroid @ R + pos_ref_centroid

    return R, t


def rmsd_align(positions: torch.Tensor, positions_ref: torch.Tensor, remove_mean=True, reflect=False):
    """
    Aligns set of positions to a reference set of positions such that their position RMSD (||y-x*R||^2) is minimized.
    R is a member of the SO(n) (reflect=False) or O(n) (reflect=True)
    Reference: Umeyama 1991, "Least-squares estimation of transformation parameters between two point patterns"
               https://en.wikipedia.org/wiki/Kabsch_algorithm
    Args:
        positions: set of positions with ndim=>=2. If ndim>2, the alignment for each set of points with the last 2 dimensions representing the point and 3D coordinate index.
        positions_ref: reference positions
        remove_mean: remove the mean positions of the input positions and reference before alignment.
        reflect: allow reflections of the positions when aligning to reference positions
    """
    if positions_ref.ndim < 3:
        positions_ref.unsqueeze(dim=0)

    # Remove centroids of positions and reference positions
    if remove_mean:
        positions = positions - torch.mean(positions, dim=-2, keepdim=True)
        positions_ref_mean = torch.mean(positions_ref, dim=-2, keepdim=True)
        positions_ref = positions_ref - positions_ref_mean

    # Calculate covariance.
    # In Umeyama's formula, C = y.T @ x since ||y-Rx|| is minimized. We need C = x.T @ y (transposed) to minimize ||y-xR||
    # covar = torch.sum(positions.unsqueeze(-1) * positions_ref.unsqueeze(-2), dim=-3) / positions_ref.shape[-2]
    covar = positions.transpose(-2, -1) @ positions_ref / positions_ref.shape[-2]
    R = SOn_proj(covar, with_reflection=reflect)
    positions_new = positions @ R

    # Add the reference mean if it was removed
    if remove_mean:
        positions_new += positions_ref_mean
    return positions_new


def rmsd(positions: torch.Tensor, positions_ref: torch.Tensor, remove_mean=True, reflect=True):
    """
    Computes the minimal RMSD between a set of positions and a reference set of positions.
    Reference: Umeyama 1991, "Least-squares estimation of transformation parameters between two point patterns"
               https://en.wikipedia.org/wiki/Kabsch_algorithm
    Args:
        positions: set of positions with ndim=>=2. If ndim>2, the alignment for each set of points with the last 2 dimensions representing the point and 3D coordinate index.
        positions_ref: reference positions
        remove_mean: remove the mean positions of the input positions and reference before alignment.
        reflect: allow reflections of the positions if necessary
    """
    positions_aligned = rmsd_align(positions=positions, positions_ref=positions_ref, remove_mean=remove_mean,
                                   reflect=reflect)
    rmsd = (positions_aligned - positions_ref).square().sum(-1).mean(-1).sqrt()
    return rmsd


def align_PC(positions: torch.Tensor, inplace=False, test=False, PC_ind=None):
    """
    Aligns principal components of a set of 3D positions along the given axes
    Args:
        positions: set of 3D positions shape=(N,3)
        inplace: perform the rotation inplace
        test: test the alignment before returning rotated positions
        PC_ind: index of the principal components in descending order.
                PC_ind[i] provides the 3D index of ith component with the highest eigenvalue

    Returns:
        positions_rotated: rotated 3D positions shape=(N,3)
    """

    if PC_ind is None:
        PC_ind = torch.arange(3)
    N = positions.shape[-2]
    positions_in = positions
    positions = positions[..., :3]

    # Remove centroid
    centroid = torch.mean(positions, dim=-2, keepdim=True)
    positions = positions - centroid

    # Find the covariance matrix
    covariance = torch.matmul(torch.transpose(positions, -1, -2), positions) / N
    eigval, Q = torch.linalg.eigh(covariance)

    # Remove reflections
    Q[:, 0] *= Q.det()

    # Order coordinates in descending order of variance
    sorting_ind = torch.argsort(eigval, descending=True)
    Q[:, PC_ind] = Q[:, sorting_ind]

    # Rotate using the centroid as the center
    positions_out = torch.matmul(positions, Q)
    if test:
        covar_mat_rotated = torch.matmul(torch.transpose(positions_out, -1, -2), positions_out) / N
        covar_diff = covar_mat_rotated[PC_ind.unsqueeze(-1), PC_ind] - torch.diag(eigval[sorting_ind])
        assert torch.all(covar_diff.abs() < 1e-6), 'The PC alignment is incorrect.'

    # Reapply centroid
    positions_out = positions_out + centroid

    if inplace:
        positions_in[..., :] = positions_out
    else:
        return positions_out


def calc_dihedral_angle(positions: torch.Tensor, deg=False):
    """
    Calculates the dihedral angle for a batch of 4 3D positions
    Reference: https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
    Args:
        positions: shape=(N,4,3) or (4,3)
        deg: outputs angles in degrees instead of radians
    Returns:
        angles: torch.Tensor with shape=(N,)
    """
    # Formulas
    # cos(phi) = (u_1 x u_2).(u_2 x u_3) / |(u_1 x u_2).(u_2 x u_3)|
    # sin(phi) = u_2.((u_1 x u_2) x (u_2 x u_3)) / (|u_2| |(u_1 x u_2) x (u_2 x u_3)|)
    is_batched = positions.ndim >= 3
    if not is_batched:
        positions = positions.unsqueeze(0)

    u = torch.diff(positions, dim=-2)
    u01_cross = torch.linalg.cross(u[..., 0, :], u[..., 1, :], dim=-1)
    u12_cross = torch.linalg.cross(u[..., 1, :], u[..., 2, :], dim=-1)
    x = torch.sum(u01_cross * u12_cross, dim=-1)
    u1_norm = torch.linalg.norm(u[..., 1, :], dim=-1)

    # Method 1
    # y = torch.sum(u[..., 1, :] * torch.linalg.cross(u01_cross, u12_cross), dim=-1)
    # # Multiply x by |u_2| so that x,y have the same norm i.e. |u_2| |(u_1 x u_2)| |(u_2 x u_3)|
    # x *= u1_norm

    # Method 2 (uses the vector quadruple product identity to avoid one cross-product)
    y = torch.linalg.vecdot(u[..., 0, :], u12_cross) * u1_norm

    angles = torch.atan2(y, x)
    if deg:
        angles = angles.rad2deg()
    if not is_batched:
        angles = angles.squeeze(0)
    return angles


def rotate_dih_quartet(positions: torch.Tensor, phi: torch.Tensor, plot=False):
    """
    Rotates a dihedral quartet of 3D positions along the torsion bond to increase the dihedral angle by phi
    Args:
        positions: quartet of 3D positions, shape=(...,4,3)
        phi: dihedral angle chaneg shape=(...,)
        plot: plot the positions and the rotated positions.

    Returns:
        positions_rotated, shape=(...,4,3)
    """
    u = torch.diff(positions, dim=-2)  # bond vectors of the quartet
    t = torch.nn.functional.normalize(u[..., 1, :], dim=-1).unsqueeze(dim=-2)  # Unit vector of the torsion bond
    u02 = u[..., [0, 2], :]

    # Use Rodrigues' formula to rotate u0 and u2
    phi = phi.reshape(*positions.shape[:-2], 1, 1)
    c_phi, s_phi = torch.cos(phi / 2), torch.sin(phi / 2)  # Rotate each end position by half the angle
    t_u02_dot, t_u02_cross = torch.linalg.vecdot(t, u02).unsqueeze(-1), torch.linalg.cross(t, u02)
    a = torch.arange(-1, 2, 2, device=u.device, dtype=u.dtype).expand(*u.shape[:-2], -1).unsqueeze(-1)
    u02_rot = c_phi * u02 + a * s_phi * t_u02_cross + (1 - c_phi) * t * t_u02_dot
    p0 = positions[..., 1, :] - u02_rot[..., 0, :]
    p3 = positions[..., 2, :] + u02_rot[..., 1, :]
    positions_rot = torch.cat([p0.unsqueeze(-2), positions[..., [1, 2], :], p3.unsqueeze(-2)], dim=-2)

    if plot:
        import matplotlib.pyplot as plt
        u_rot = torch.diff(positions_rot, dim=-2)
        phi_flat = phi.flatten()
        pos_flat = positions.flatten(0, -3)
        pos_rot_flat = positions_rot.flatten(0, -3)
        dih_angle_start = calc_dihedral_angle(pos_flat, deg=True)
        dih_angle_end = calc_dihedral_angle(pos_rot_flat, deg=True)
        for i, (p, p_rot, ph) in enumerate(zip(pos_flat, pos_rot_flat, phi_flat)):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            plt.plot(*p.t(), color='r')
            plt.plot(*p_rot.t(), color='b')
            title = (f"\N{GREEK SMALL LETTER PHI}_start={dih_angle_start[i]:.0f}\N{DEGREE SIGN}, "
                     f"\N{GREEK SMALL LETTER PHI}_end={dih_angle_end[i]:.0f}\N{DEGREE SIGN}, "
                     f"\N{GREEK CAPITAL LETTER DELTA}\N{GREEK SMALL LETTER PHI}={ph.rad2deg():.0f} \N{DEGREE SIGN}")
            plt.title(title)
            fig.show()

    return positions_rot


def calc_bond_angle(positions: torch.Tensor, deg=False, epsilon=1e-8):
    """
    Calculates the bond angle for a batch of 3 3D positions
    Args:
        positions: shape=(...,3,3)
        deg: outputs angles in degrees. Otherwise, outputs in radians
        epsilon: small number to stabilize calculation of bond vecs with small norms

    Returns:
        angles: torch.Tensor with shape=(...,)
    """
    is_batched = positions.ndim >= 3
    if not is_batched:
        positions = positions.unsqueeze(0)

    # Calculate the bond vectors r_ij = x_j - x_i and r_jk = x_k - x_j
    bond_vecs = torch.diff(positions, dim=-2)  # bond_vecs[...,0,:] = r_ij and bond_vecs[...,1,:] = r_jk
    bond_norm = torch.linalg.norm(bond_vecs, dim=-1, keepdim=True)
    bond_vecs = bond_vecs / (bond_norm + epsilon)

    # Calculate the angle between r_ji and r_jk using their dot product
    bond_vecs_dot = torch.linalg.vecdot(-bond_vecs[..., 0, :], bond_vecs[..., 1, :], dim=-1)
    angles = torch.acos(bond_vecs_dot.clamp(min=-1.0 + epsilon, max=1.0 - epsilon))  # clamp min, max to avoid nans.

    if deg:
        angles = angles.rad2deg()
    if not is_batched:
        angles = angles.squeeze(0)
    return angles


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically accurate evaluation of log(1 - exp(-x)) assuming x > 0
    References: https://github.com/pytorch/pytorch/issues/39242
                https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    x_is_small = x < LOG_2
    x_safe = x.clamp(min=torch.finfo(x.dtype).eps)

    # x is small
    z_small = torch.log(-torch.expm1(-x_safe))

    # x is large
    # Since exp(-x) ~=0 log(1-exp(-x)) ~= log(1-1) -> -inf. Ensure this doesn't happen by clamping min of x for this case
    x_large = torch.where(x_is_small, torch.full_like(x, LOG_2), x_safe)  # x_large >= log_2
    z_large = torch.log1p(-torch.exp(-x_large))
    return torch.where(x_is_small, z_small, z_large)


def logerfc(x: torch.Tensor):
    """
    Evaluates log(erfc(x))
    Args:
        x: torch.Tensor

    Returns:
        torch.Tensor with same shape as x
    """
    # For x < 0, use the fact that erfc(x) = 2 - erfc(-x):
    # log(erfc(x)) = log(2 - erfc(-x)) = log(1 - erfc(-x)/2) + log(2)
    #                                  = log[1 - exp(log(erfc(-x)) - log(2))] + log(2)
    #                                  = log[1 - exp(y)] + log(2) where y = log(erfc(-x)) - log(2)
    # Also, since log(erfc(x)) = log(erfcx(x)) - x^2 -> y = log(erfcx(-x)) - x^2 - log(2)
    # For x >= 0, use log(erfc(x)) = log(erfcx(x)) - x^2
    x_neg_mask = x < 0
    x2 = x.square()
    if torch.compiler.is_compiling() or force_compile:
        x_neg_safe = torch.where(x_neg_mask, x, torch.zeros_like(x))  # Ensures x_neg_safe <= 0 in all values
        y_neg = torch.log(torch.special.erfcx(-x_neg_safe)) - x2 - LOG_2
        z_neg = log1mexp(torch.where(x_neg_mask, -y_neg, torch.ones_like(x))) + LOG_2
        x_pos_safe = torch.where(x_neg_mask, torch.zeros_like(x), x)  # Ensures x_pos_safe >= 0 in all values
        z_pos = torch.log(torch.special.erfcx(x_pos_safe)) - x2
        z = torch.where(x_neg_mask, z_neg, z_pos)
    else:
        x_neg, x_pos = x[x_neg_mask], x[~x_neg_mask]
        y_neg = torch.log(torch.special.erfcx(-x_neg)) - x2[x_neg_mask] - LOG_2
        z_neg = log1mexp(-y_neg) + LOG_2
        z_pos = torch.log(torch.special.erfcx(x_pos)) - x2[~x_neg_mask]

        z = torch.zeros_like(x)
        z[x_neg_mask] = z_neg
        z[~x_neg_mask] = z_pos

    return z


def logerfcsum(x: torch.Tensor, y: torch.Tensor, a=1.0) -> torch.Tensor:
    """
    Evaluates log(erfc(x) + a*erfc(y)) where a = 1 or -1
    Args:
        x: torch.Tensor
        y: torch.Tensor
        a: constant that is either 1 or -1
    Returns:
        z: torch.Tensor
    """
    log_erfc_x = logerfc(x)
    log_erfc_y = logerfc(y)
    if a > 0:
        z = torch.logaddexp(log_erfc_x, log_erfc_y)
    else:
        z = logdiffexp(log_erfc_x, log_erfc_y)
    return z


def logdiffexp(x: torch.Tensor, y: torch.Tensor):
    """
    Evaluates log(e^x - e^y) assuming x > y
    Args:
        x: torch.Tensor
        y: torch.Tensor

    Returns: torch.Tensor
    """
    # assert torch.all(y < x), "y must be < x"
    return x + log1mexp(x - y)


def logsinh(x: torch.Tensor):
    y = x + log1mexp(2 * x) - LOG_2  # Assumes x >= 0
    return y


def MarcumQ(a: torch.Tensor, b: torch.Tensor, nu: float, complement=False, log_output=False, compile=False):
    """
    Computes the MarcumQ function Q_nu(a,b)
    Reference: https://en.wikipedia.org/wiki/Marcum_Q-function
    Args:
        a: first argument of MarcumQ function
        b: second argument of MarcumQ function
        nu: 1/2 or 3/2
        complement: calculates the complement 1 - Q_nu(a,b) instead of Q_nu(a,b)
        log_output: calculates the log of the output, i.e. log(Q_nu(a,b)) or log(1-Q_nu(a,b)) if complement=True
        compile: uses the compile-safe version that uses torch.where() calls
    Returns:
        torch.Tensor
    """
    # Definitions:
    # A = erfc((a-b)/sqrt(2))/2
    # B = erfc((a+b)/sqrt(2))/2
    # The constant term of Q_nu(a,b,nu) (common to all values of n) corresponds to A + B
    # Edge cases
    # Q_nu(a,0) = 1
    # Q_nu(a,inf) = 0

    # Overwrite b on edges cases to prevent nans in the False branch when using torch.where(b_is_zero,...)
    b_is_zero, b_is_inf = b.eq(0.0), b.isinf()
    b = torch.where(b_is_zero, a / 2, b)
    b = torch.where(b_is_inf, 2 * a, b)

    c = (2 / torch.pi) ** 0.5
    C1 = 1 / 2 ** 0.5
    ab_sum = C1 * (b + a)
    ab_diff = C1 * (b - a)
    compile |= torch.compiler.is_compiling() or force_compile
    if log_output:
        # Evaluate log(A+B) as log(exp(A_log) + exp(B_log))
        y = logerfcsum(ab_diff, ab_sum) - LOG_2
        if complement:
            # When 1 - MarcumQ is required, use the fact that 1 - erfc((a-b)/sqrt(2))/2 = erfc(-(a-b)/sqrt(2))/2
            # for cases where ab_diff < 0.
            # When ab_diff < 0, erfc((a-b)/sqrt(2))/2 is close to one, so the rewriting avoids unstable cancellation.
            # When ab_diff > 0, both A and B will be small
            # Therefore, user log(1-A-B) = log(1 - exp(log(A+B))) = log(1 - exp(y)) where exp(y) will be small.
            ab_diff_neg_mask = ab_diff < 0
            y_diff_neg = logerfcsum(-ab_diff, ab_sum, a=-1.0) - LOG_2
            if compile:
                y_diff_pos = log1mexp(torch.where(ab_diff_neg_mask, torch.ones_like(y), -y))  # log(1-exp(y))
                y = torch.where(ab_diff_neg_mask, y_diff_neg, y_diff_pos)
            else:
                y_diff_neg = y_diff_neg[ab_diff_neg_mask]
                y_diff_pos = log1mexp(-y[~ab_diff_neg_mask])  # log(1-exp(y))
                y[ab_diff_neg_mask] = y_diff_neg
                y[~ab_diff_neg_mask] = y_diff_pos

        if nu == 1 / 2:
            pass
        elif nu == 3 / 2:
            # Add extra term for nu=3/2
            # Definitions
            # C = sqrt(2/pi) * sinh(ab)/a exp(-(a^2+b^2)/2)
            # D = log(sinh(ab)/a)
            # log(C) = log(sqrt(2/pi)) + D - (a^2+b^2)/2

            # There are 2 cases to consider to properly handle D
            # 1) 0 < b < inf, a >> 0
            #    D = log(sinh(ab)) - log(a) is stable
            # 2) 0 < b < inf, a~=0
            #    sinh(ab)/a approaches 0/0. Approximate to its limit, sinh(ab)/a ~= b, i.e. D = log(b)

            # When a is close to zero, C is unstable due to sinh(ab)/a approaching 0/0, which leads to cancellation of
            # two infinities in log space. In this case, simplify sinh(ab)/a to its limit as a->0, i.e. b.
            is_a_close_to_zero = a.lt(10 * torch.finfo(a.dtype).eps)
            a_safe = torch.where(is_a_close_to_zero, torch.ones_like(a), a)
            D = torch.where(is_a_close_to_zero, torch.log(b), logsinh(a_safe * b) - torch.log(a_safe))
            C_log = math.log(c) + D - (a.square() + b.square()) / 2

            if complement:
                y = logdiffexp(y, C_log)  # = log(1-A-B-C) = log[exp(log(1-A-B)) - exp(log(C))]
            else:
                y = torch.logaddexp(y, C_log)
        else:
            raise NotImplementedError(f"log output for nu={nu} is not implemented")

        # Add edge cases when b is zero or infinity
        if complement:
            y = torch.where(b_is_zero, torch.full_like(y, -torch.inf), y)
            y = torch.where(b_is_inf, torch.zeros_like(y), y)
        else:
            y = torch.where(b_is_zero, torch.zeros_like(y), y)
            y = torch.where(b_is_inf, torch.full_like(y, -torch.inf), y)

        return y

    else:
        if complement:
            erf_func = torch.erf
            Sign = -1
        else:
            erf_func = torch.erfc
            Sign = 1

        y = 0.5 * (erf_func(ab_diff) + erf_func(ab_sum))

        if nu == 3 / 2:
            exp_sum = torch.exp(-ab_sum.square())
            exp_diff = torch.exp(-ab_diff.square())
            sinh_d = (exp_diff - exp_sum) / 2  # sinh(ab) * exp(-(a^2+b^2)/2)
            y += Sign * c * sinh_d / a
            if not torch.isfinite(y).all():
                raise ValueError(f"MarcumQ function overflowed")

        # Add edge cases when b is zero or infinity
        if complement:
            y = torch.where(b_is_zero, torch.zeros_like(y), y)
            y = torch.where(b_is_inf, torch.ones_like(y), y)
        else:
            y = torch.where(b_is_zero, torch.ones_like(y), y)
            y = torch.where(b_is_inf, torch.zeros_like(y), y)

        return y


def norm_cdf(mu: torch.Tensor, sigma: torch.Tensor = 1.0, x_lower: torch.Tensor = None,
             x_upper: torch.Tensor = None, log=False):
    """
    Calculate the cdf of a normal distribution: p(x_lower <= x <= x_upper)
    Args:
        mu: mean of the normal distribution
        sigma: standard deviation of the normal distribution
        x_lower: lower bound of the cdf
        x_upper: upper bound of the cdf
        log: returns the log of the cdf
    Returns:
        p(x_lower < x < x_upper) or log(p(x_lower < x < x_upper)) where x ~ N(mu,sigma)
    """

    # let y_l = (x_lower - mu)/(sqrt(2)*sigma)   and    y_h = (x_high - mu)/(sqrt(2)*sigma)
    # z = (y - mu)/(sqrt(2)*sigma)
    # p(x <= y) = 1/2*(1 + erf(z)) = 1/2*(2 - erfc(z)) = 1 - 0.5*erfc(z) = 1 - 0.5*(2 - erfc(-z)) = 0.5*erfc(-z)
    # p(x >= y) = 1 - p(x <= y) = 1/2*(1 - erf(z)) = 1/2*(erfc(z)) = 0.5*erfc(z)
    is_upper_trivial = x_upper is None or x_upper.isinf().all()
    is_lower_trivial = x_lower is None or x_lower.isinf().all()
    log_cdf = torch.zeros_like(mu)
    if is_upper_trivial and is_lower_trivial:
        pass
    elif is_upper_trivial:
        # Using eq. derived above, log(p(x >= x_lower)) = log(1-p(x <= x_lower)) = log(0.5*erfc(y_l))
        y_l = (x_lower - mu) / (math.sqrt(2) * sigma)
        log_cdf = logerfc(y_l) - LOG_2
    elif is_lower_trivial:
        # Using eq. derived above, log(p(x <= x_upper)) = log(0.5*erfc(-y_h)) = logerfc(-y_h) - log(2)
        y_h = (x_upper - mu) / (math.sqrt(2) * sigma)
        log_cdf = logerfc(-y_h) - LOG_2
    else:
        # To get numerically stable results, 2 cases need to be handled depending on how far mu is to the lower and upper bounds
        # 1) mu is close to both bounds (y_l ~= y_h)
        # In this case, we need to consider both bound simultaneously
        # p(x_lower <= x <= x_upper) = p(x <= x_upper) - p(x <= x_lower) = 0.5*(erfc(y_l) - erfc(y_h))

        # 2) y_h >> y_l
        # In this case, the upper bound is much farther than the lower bound. We can handle each bound separately.
        # For mu_x ~= x_lower, p(x <= x_upper) ~= 1 -> p(x_lower <= x <= x_upper) ~= 1 - p(x <= x_lower) = p(x_lower >= x)
        # For mu_x >> x_lower but x << x_upper, p(x <= x_upper) ~= 1 and p(x_lower <= x) ~= 1 so p(x_lower <= x <= x_upper) ~= 1
        # For mu_x ~= x_upper, p(x <= x_lower) ~= 0 -> p(x_lower <= x <= x_upper) ~= p(x <= x_upper)
        # Therefore, when bounds are far from each other, p(x_lower <= x <= x_upper) ~= p(x_lower >= x) p(x <= x_upper)
        z_score_thresh = math.sqrt(2) * 2
        y_l = (x_lower - mu) / (math.sqrt(2) * sigma)
        y_h = (x_upper - mu) / (math.sqrt(2) * sigma)
        mu_isclose_to_upper = y_h.abs() < z_score_thresh
        mu_isclose_to_lower = y_l.abs() < z_score_thresh
        mu_isclose_to_both = torch.logical_and(mu_isclose_to_lower, mu_isclose_to_upper)

        # bounds are close
        logerfc_low = logerfc(y_l[mu_isclose_to_both])
        logerfc_high = logerfc(y_h[mu_isclose_to_both])
        log_cdf[mu_isclose_to_both] = logdiffexp(logerfc_low, logerfc_high) - LOG_2

        # bounds are far
        logerfc_low = logerfc(y_l[~mu_isclose_to_both]) - LOG_2
        logerfc_high = logerfc(-y_h[~mu_isclose_to_both]) - LOG_2
        log_cdf[~mu_isclose_to_both] = logerfc_low + logerfc_high

    if log_cdf.isnan().any():
        warnings.warn(f"nans found in norm_log_cdf")

    output = log_cdf if log else log_cdf.exp()
    return output


def mvn_cond_params(mu: torch.Tensor, C: torch.Tensor, cond_mask: torch.Tensor, cond_values: torch.Tensor):
    """
    Defines the average and covariance matrix of a conditional multivariate normal distribution
    Args:
        mu: average of the unconditional multivariate normal distribution
        C: covariance matrix of the unconditional multivariate normal distribution
        cond_mask: 1D mask identifying the conditioned variables
        cond_values: values of the conditioned variables. shape=(n,d)

    Returns:
        cond_mu: average of the conditional multivariate normal distribution
        cond_covar: covariance matrix of the conditional multivariate normal distribution
    """
    # x = unconditioned variables, y = conditioned variables
    uncond_mask = ~cond_mask
    sigma_xx = C[..., uncond_mask, :][..., :, uncond_mask]
    sigma_xy = C[..., uncond_mask, :][..., :, cond_mask]
    sigma_yy = C[..., cond_mask, :][..., :, cond_mask]

    sigma_yy_inv = sigma_yy.inverse()
    mu_x, mu_y = mu[..., uncond_mask], mu[..., cond_mask]
    cond_mu = mu_x + torch.squeeze(sigma_xy @ sigma_yy_inv @ (cond_values - mu_y).unsqueeze(-1), -1)
    cond_covar = sigma_xx - sigma_xy @ sigma_yy_inv @ sigma_xy.transpose(-1, -2)
    return cond_mu, cond_covar


def gaussian_prod(mu: torch.Tensor, cov: torch.Tensor):
    """
    Calculates the mean and covariance matrix of a multivariate normal dist. formed by the product of n multivariate normal dist.
    References:
        https://compbio.fmph.uniba.sk/vyuka/ml/old/2008/handouts/matrix-cookbook.pdf
        http://www.lucamartino.altervista.org/2003-003.pdf   eq. 9
    Args:
        mu: shape=(n,d)
        cov: shape=(n,d,d)

    Returns:
        mu_n,cov_n
    """
    Lambda = torch.linalg.inv(cov)  # precision matrices
    Lambda_n = Lambda.sum(dim=0)
    cov_n = Lambda_n.inverse()
    mu_n = cov_n @ torch.sum(Lambda @ mu.unsqueeze(-1), dim=0)
    return mu_n, cov_n


def NonCentralChicdf(Lambda: torch.Tensor, k: int, x_lower: torch.Tensor = None, x_upper: torch.Tensor = None,
                     log_output=False, compile=False):
    """
    Calculates the integral of the density function of the non-central chi distribution between given bounds
    Args:
        Lambda: non-centrality parameter of the chi distribution, i.e. Lambda = sqrt(sum_i x_i^2/sigma_i^2)
        k: # of degrees of freedom of the chi distribution
        x_lower: lower bounds, a value of 0 or -inf implies no bounds
        x_upper: upper bounds, a value of inf implies no bounds
        log_output: If True, returns log(cdf(x_low,x_upp)) otherwise returns cdf(x_low,x_upp)
        compile: uses the compile-safe version that uses torch.where() calls

    Returns:
        cdf(x_low,x_upp) or log(cdf(x_low,x_upp))
    """
    if k > 3:
        raise NotImplementedError(f"Cdf is not implemented for k > 3.")
    z_score_thresh = 5  # z-score threshold for determining when bounds are close to Lambda
    nu = k / 2
    compile |= torch.compiler.is_compiling() or force_compile

    # Broadcast dimensions
    if x_lower is not None and x_upper is not None:
        Lambda, x_lower, x_upper = torch.broadcast_tensors(Lambda, x_lower, x_upper)
    elif x_lower is not None:
        Lambda, x_lower = torch.broadcast_tensors(Lambda, x_lower)
    elif x_upper is not None:
        Lambda, x_upper = torch.broadcast_tensors(Lambda, x_upper)

    if compile:
        if x_lower is None:
            x_lower = torch.zeros_like(Lambda)
            is_x_lower_nontrivial = torch.zeros_like(Lambda, dtype=torch.bool)
        else:
            is_x_lower_nontrivial = x_lower.gt(0.0)

        if x_upper is None:
            x_upper = torch.full_like(Lambda, torch.inf)
            is_x_upper_nontrivial = torch.zeros_like(Lambda, dtype=torch.bool)
        else:
            is_x_upper_nontrivial = ~x_upper.isinf()

        # Calculate log(p(x >= x_lower)). This is safe for all values of x_lower >=0
        log_cdf_lower_comp = MarcumQ(Lambda, x_lower, nu=nu, log_output=True, complement=False, compile=compile)

        # Calculate log(p(x <= x_upper)). This is safe for all values of x_upper, even x_upper=inf
        log_cdf_upper = MarcumQ(Lambda, x_upper, nu=nu, log_output=True, complement=True, compile=compile)

        # Handle case when Lambda is close to both limits
        y_l = (x_lower - Lambda) / math.sqrt(2)
        y_h = (x_upper - Lambda) / math.sqrt(2)
        Lambda_is_close_to_lower = y_l.abs() < z_score_thresh
        Lambda_is_close_to_upper = y_h.abs() < z_score_thresh
        is_x_low_upp_nontrivial = is_x_lower_nontrivial & is_x_upper_nontrivial
        Lambda_is_close_to_both = is_x_low_upp_nontrivial & Lambda_is_close_to_lower & Lambda_is_close_to_upper

        # When one of the bounds is far from Lambda, log(p(x_lower <= x <= x_upper)) = log(p(x >= x_lower))) + log(p(x <= x_upper))
        log_cdf_apart = log_cdf_lower_comp + log_cdf_upper

        # When Lambda is close to both bounds, log(p(x_lower <= x <= x_upper)) = log(p(x <= x_upper) - p(x <= x_lower))
        # First, calculate p(x <= x_lower)
        x_lower_safe = torch.where(Lambda_is_close_to_both, x_lower,
                                   torch.minimum(torch.ones_like(Lambda), x_upper / 2))
        log_cdf_lower = MarcumQ(Lambda, x_lower_safe, nu=nu, log_output=True, complement=True, compile=compile)
        log_cdf_close = logdiffexp(log_cdf_upper, log_cdf_lower)

        log_cdf = torch.where(Lambda_is_close_to_both, log_cdf_close, log_cdf_apart)
        return log_cdf if log_output else log_cdf.exp()

    # Calculate log(p(x >= x_lower))
    if x_lower is not None:
        log_cdf_lower_comp = torch.zeros_like(Lambda)

        # Handle inf or 0 lower limits
        is_x_lower_nontrivial = x_lower.gt(0.0)
        is_any_lower_non_trivial = is_x_lower_nontrivial.any()
        if is_any_lower_non_trivial:
            log_cdf_lower_nontriv = MarcumQ(Lambda[is_x_lower_nontrivial], x_lower[is_x_lower_nontrivial], nu=nu,
                                            log_output=True, complement=False)
            log_cdf_lower_comp[is_x_lower_nontrivial] = log_cdf_lower_nontriv
    else:
        is_any_lower_non_trivial = False

    # Calculate log(p(x <= x_upper))
    if x_upper is not None:
        log_cdf_upper = torch.zeros_like(Lambda)

        # Handle inf upper limits
        is_x_upper_nontrivial = ~x_upper.isinf()
        is_any_upper_non_trivial = is_x_upper_nontrivial.any()
        if is_any_upper_non_trivial:
            log_cdf_upper_nontriv = MarcumQ(Lambda[is_x_upper_nontrivial], x_upper[is_x_upper_nontrivial], nu=nu,
                                            log_output=True, complement=True)
            log_cdf_upper[is_x_upper_nontrivial] = log_cdf_upper_nontriv
    else:
        is_any_upper_non_trivial = False

    if is_any_lower_non_trivial and is_any_upper_non_trivial:
        # When both bounds are used, 2 cases need to be handled to get numerically stable results.
        # The 2 cases depend on how far the lower and upper bounds are from one another
        # 1) mu is close to both bounds (y_l ~= yh)
        # In this case, we need to consider both bounds simultaneously
        # p(x_lower <= x <= x_upper) = p(x <= x_upper) - p(x <= x_lower)

        # 2) y_h >> y_l
        # In this case, the upper bound is much farther than the lower bound. We can handle each bound separately.
        # When Lambda << x_lower, p(x <= x_upper) ~= 1 -> p(x_lower <= x <= x_upper) ~= 1 - p(x <= x_lower) = p(x_lower >= x)
        # When Lambda ~= x_lower, p(x <= x_upper) ~= 1 -> p(x_lower <= x <= x_upper) ~= 1 - p(x <= x_lower) = p(x_lower >= x)
        # When Lambda >> x_lower but Lambda << x_upper, p(x_lower <= x <= x_upper) ~= 1 and p(x <= x_upper) ~= 1 and p(x >= x_lower) ~= 1
        # When Lambda ~= x_upper, p(x <= x_lower) ~= 0 or p(x_lower >= x) ~= 1 -> p(x_lower <= x <= x_upper) ~= p(x <= x_upper)
        # Therefore, in all cases, p(x_lower <= x <= x_upper) ~= p(x_lower >= x) p(x <= x_upper)
        y_l = (x_lower - Lambda) / math.sqrt(2)
        y_h = (x_upper - Lambda) / math.sqrt(2)
        log_cdf = torch.zeros_like(y_l)
        Lambda_is_close_to_lower = y_l.abs() < z_score_thresh
        Lambda_is_close_to_upper = y_h.abs() < z_score_thresh
        is_low_and_upp_nontrivial = is_x_lower_nontrivial & is_x_upper_nontrivial
        Lambda_is_close_to_both = is_low_and_upp_nontrivial & Lambda_is_close_to_lower & Lambda_is_close_to_upper
        Lambda_is_not_close_to_both = ~Lambda_is_close_to_both

        # bounds are close
        if Lambda_is_close_to_both.any():
            # Calculate p(x <= x_lower) and subtract it from  p(x <= x_upper)
            log_cdf_lower_close = MarcumQ(Lambda[Lambda_is_close_to_both], x_lower[Lambda_is_close_to_both], nu=nu,
                                          log_output=True, complement=True)
            log_cdf_upper_close = log_cdf_upper[Lambda_is_close_to_both]
            log_cdf[Lambda_is_close_to_both] = logdiffexp(log_cdf_upper_close, log_cdf_lower_close)

        # bounds are far
        if Lambda_is_not_close_to_both.any():
            log_cdf_lower_comp_notclose = log_cdf_lower_comp[Lambda_is_not_close_to_both]
            log_cdf_upper_notclose = log_cdf_upper[Lambda_is_not_close_to_both]
            log_cdf[Lambda_is_not_close_to_both] = log_cdf_lower_comp_notclose + log_cdf_upper_notclose

    elif is_any_lower_non_trivial:
        log_cdf = log_cdf_lower_comp  # =log(1 - p(x<x_lower)) = log(p(x>=x_lower))
    else:
        log_cdf = log_cdf_upper

    if not log_cdf.isfinite().all():
        n_non_finite = torch.count_nonzero(~log_cdf.isfinite())
        warnings.warn(f"NonCentralChicdf returned {n_non_finite.item()} non-finite values.")

    return log_cdf if log_output else log_cdf.exp()


def NonCentralChipdf(x: torch.Tensor, Lambda: torch.Tensor, k: int = 3, log_output=False):
    if k != 3:
        raise NotImplementedError

    log_pdf = -0.5 * (x - Lambda).square() + x.log() + log1mexp(2 * Lambda * x)
    log_pdf -= torch.log(math.sqrt(2 * torch.pi) * Lambda)
    if not log_output:
        return log_pdf.exp()
    else:
        return log_pdf


def NonCentralChi2cdf(Lambda: torch.Tensor, k: int, x_lower: torch.Tensor = None, x_upper: torch.Tensor = None,
                      log_output=False, sqrt_input=False, **kwargs):
    """
    Calculates the integral of the density function of the non-central chi^2 distribution between given bounds
    Args:
        Lambda: non-centrality parameter of the chi distribution, i.e. Lambda = sum_i x_i^2/sigma_i^2
        k: # of degrees of freedom of the chi distribution
        x_lower: lower bounds, a value of 0 or -inf implies no bounds
        x_upper: upper bounds, a value of inf implies no bounds
        log_output: If True, returns log(cdf(x_low,x_upp)) otherwise returns cdf(x_low,x_upp)
        sqrt_input: assumes that inputs are already square-rooted so that they can be given to NonCentralChicdf directly

    Returns:
        cdf(x_low,x_upp) or log(cdf(x_low,x_upp))
    """
    if sqrt_input:
        Lambda_sqrt = Lambda
        x_lower_sqrt = x_lower
        x_upper_sqrt = x_upper
    else:
        Lambda_sqrt = torch.sqrt(Lambda)
        x_lower_sqrt = torch.sqrt(x_lower) if x_lower is not None else None
        x_upper_sqrt = torch.sqrt(x_upper) if x_upper is not None else None

    return NonCentralChicdf(Lambda=Lambda_sqrt, k=k, x_lower=x_lower_sqrt, x_upper=x_upper_sqrt, log_output=log_output,
                            **kwargs)


def calculate_func_mu_sigma(f: typing.Callable, mu: torch.Tensor, cov: torch.Tensor = None, **kwargs):
    """
    Calculates a 2nd order approximation of the mean and variance of a function f(x) of a multidimensional random variable x.
    f(x) is approximated as f(x) ~ f(mu) + sum_i df/dx_i (mu) (x_i-muI) + sum_ij d^2f/(dx_i dx_j) (x_i-mu_i)(x_j-mu_j)
    and the mean and covariance matrix are computed
    <f_i(mu)> and <f_i(mu) f_j(mu)> where i,j indexes the output dimension of f

    Reference: https://en.wikipedia.org/wiki/Taylor_expansions_for_the_moments_of_functions_of_random_variables
    Args:
        f: function that returns a 1D vector of dimension d_out
        mu: average (shape=(...,d_in,))
        cov: covariance matrix or None (shape=(...,d_in,d_in)). When None, covariance is assumed to be identity matrix

    Returns:
        mu_f: torch.Tensor (shape=(..., d_out,))
        cov_f: torch.Tensor (shape=(..., d_out,d_out))
    """
    from torch.autograd.functional import jacobian
    # Squash all batch dimensions into one
    # batch_shape, d_in = mu.shape[:-1], mu.shape[-1]

    # Add a batch dimension if input is 1D
    input_is_flat = mu.ndim == 1
    if input_is_flat:
        mu = mu.unsqueeze(0)
        if cov is not None:
            cov = cov.unsqueeze(0)
    batch_dims = list(range(len(mu.shape[:-1])))
    n_batch_dims = len(batch_dims)

    # If sigma has the same number of dimension as the batch dimensions, assume it is a constant for each batch
    sigma_is_constant = cov is not None and cov.shape[n_batch_dims:].numel() == 1
    if sigma_is_constant:
        cov = cov.reshape(cov.shape[:n_batch_dims])

    # Compute function, jacobian and hessian at the given mean
    batch_dims = list(range(len(mu.shape[:-1])))

    def sum_batch(g, x):
        y = g(x).sum(dim=batch_dims)  # sums over batch dimensions
        return y

    def jac(g, x):
        df = jacobian(lambda y: sum_batch(g, y), x, **kwargs)
        permutation = list(range(df.ndim))
        n = df.ndim - x.ndim
        permutation = permutation[n:] + permutation[:n]  # rotates the dimensions n times
        # permutation.insert(-1, permutation.pop(0))  # permutation = (1,2,..., 0, d-1)
        df = df.permute(permutation)  # shape=(...,d_out,d_in)
        return df

    df = jac(f, mu)  # shape=(...,d_in,d_out)
    d2f = jac(lambda x: jac(f, x), mu)  # shape=(...,d_in,d_in,d_out)

    # Compute the product d_k d_l f_i(mu) Sigma_kl which will be used twice (shape (...,d_out))
    if cov is None or sigma_is_constant:
        d2f_sigma_prod = d2f.diagonal(dim1=-2, dim2=-3).sum(-1)
        if sigma_is_constant:
            d2f_sigma_prod *= cov[..., None]
    else:
        d2f_sigma_prod = torch.sum(d2f * cov.unsqueeze(-1), dim=[-2, -3])

    # Estimate the mean and covariance (formulas use Einstein convention)
    # f_mu_j = f_j(mu) + 1/2*d_k d_l f_i(mu) Sigma_kl
    #        = f_j(mu) + 1/2 trace(S @ Sigma)
    # mu_f = f(mu) + 0.5*torch.einsum('...kli,...kl->...i', d2f, sigma) # slow
    mu_f = f(mu) + 0.5 * d2f_sigma_prod  # Use Hadamard product to compute trace

    # Estimate the covariance matrix
    # f_Sigma_ij = d_k f_i d_l f_j Sigma_kl - 1/4 (d2_kl f_i Sigma_kl d2_uv f_j Sigma_uv)
    # einsum is too memory intensive
    # cov_f = torch.einsum('...ki,...lj,...kl->...ij', df, df, sigma) # too much memory
    # cov_f += -0.5 * torch.einsum('...kli,...kl,...uvj,...uv->...ij', d2f, sigma, d2f, sigma)
    if cov is None or sigma_is_constant:
        cov_f = df.transpose(-1, -2) @ df
        if sigma_is_constant:
            cov_f *= cov[..., None, None]
    else:
        cov_f = df.transpose(-1, -2) @ cov @ df
    cov_f += -0.25 * d2f_sigma_prod[..., None] * d2f_sigma_prod[..., None, :]

    # Remove the batch dimension if input was flat
    if input_is_flat:
        mu_f = mu_f.squeeze(0)
        cov_f = cov_f.squeeze(0)

    return mu_f, cov_f


def GFT(f: torch.Tensor, A: torch.Tensor):
    """
    Calculates the Graph Fourier Transform (GFT) of a function f(i) defined on nodes of a graph
    References:
        https://en.wikipedia.org/wiki/Graph_Fourier_transform
    Args:
        f: function values on each node. shape = (n,d) n = number of nodes,  d = function output dimension
        A: adjacency matrix

    Returns:
        fourier transform of f. shape (n,d)
    """
    # Calculate the graph Laplacian
    D = torch.diag(A.sum(dim=0))  # degree matrix
    L = D - A.to(torch.float)

    # Calculate the eigenvectors and eigenvalues of the Laplacian matrix
    L_eigval, L_eigvec = torch.linalg.eigh(L)
    F = L_eigvec.T @ f  # F(k,j) = sum_i=1...n f(i,j) v_k(i)
    return F


################################################## TRAINING ##################################################


def simple_collate(x):
    return x[0]


class AdEMAMix(Optimizer):
    """
    Implementation of AdEMAMix optimizer with extra features
    Reference: "The AdEMAMix Optimizer: Better, Faster, Older", http://arxiv.org/abs/2409.03137
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999, 0.9999), alpha=5.0, T_start: int | None = None,
                 T_warmup: int | None = 1000, eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, alpha=alpha, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.device = self.param_groups[0]['params'][0].device

        for group in self.param_groups:
            # Cast parameters into tensors moved to parameters' device.
            # beta1, beta2, beta3 = group["betas"]
            # group["betas"] = (beta1, beta2, torch.tensor(beta3, device=self.device))
            # group["betas"] = tuple([torch.tensor(b, device=self.device, dtype=torch.double) for b in group["betas"]])
            # group["alpha"] = torch.tensor(group["alpha"], device=self.device)

            # Initialize parameter state
            for p in group['params']:
                self.init_param_state(p)

        self.T_start = None if T_start is None else torch.tensor(T_start, device=self.device)
        self.T_warmup = None if T_warmup is None else torch.tensor(T_warmup, device=self.device)

    def init_param_state(self, p):
        state = self.state[p]
        state["step"] = 0
        state["g1_fast"] = torch.zeros_like(p)
        state["g1_slow"] = torch.zeros_like(p)
        state["g2"] = torch.zeros_like(p)

    def alpha_scheduler(self, t, alpha):
        return float(torch.clamp(t / self.T_warmup, max=1.0) * alpha)

    def beta3_scheduler(self, t, beta_start, beta3):
        beta_start, beta3 = torch.tensor(beta_start), torch.tensor(beta3)
        beta_start_log, beta3_log = beta_start.log(), beta3.log()
        tau = t / self.T_warmup
        beta3_warm = torch.exp((beta3_log * beta_start_log) / beta3_log.lerp(end=beta_start_log, weight=tau))
        return float(torch.min(beta3_warm, beta3_log))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2, beta3 = group["betas"]
            alpha = group["alpha"]

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]
                g1_fast, g1_slow, g2 = state["g1_fast"], state["g1_slow"], state["g2"]
                state["step"] += 1

                # Update the EMAs
                g1_fast.lerp_(grad, 1 - beta1)
                g2.lerp_(grad.square(), 1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Compute parameter delta
                p_delta_num = g1_fast.div(bias_correction1)
                p_delta_denom = g2.div(bias_correction2).sqrt().add_(eps)

                # Update and add slow gradient EMA
                if self.T_start is None or (self.T_start is not None and state['step'] >= self.T_start):

                    # Call the schedulers for alpha and beta3 if warmup is used
                    if self.T_warmup is not None and state['step'] < self.T_warmup:
                        alpha_temp = self.alpha_scheduler(state["step"], alpha=alpha)
                        beta3_temp = self.beta3_scheduler(state["step"], beta_start=beta1, beta3=beta3)
                    else:
                        alpha_temp, beta3_temp = alpha, beta3

                    g1_slow.lerp_(grad, 1 - beta3_temp)
                    p_delta_num.add_(g1_slow, alpha=alpha_temp)

                # Apply parameter update including weight decay
                p_delta = p_delta_num.div(p_delta_denom)
                if weight_decay > 0.0:
                    p_delta.add_(p, alpha=weight_decay)
                p.add_(p_delta, alpha=-lr)
        return None


class ProgressiveSampler(Sampler):
    def __init__(self, dataset: Dataset, rate: float = 0.001):
        super().__init__()
        self.N = len(dataset)
        self.rate = rate
        self.epoch = 0

    def __len__(self):
        return min(100 + int(self.rate * self.epoch * self.N), self.N)

    def __iter__(self):
        N_max = len(self)
        indices = torch.randperm(N_max).tolist()
        yield from indices


class Checkpointer:
    """
    Performs checkpoints of a model and associated modules and loads from previous checkpoints.
    """

    def __init__(self, net, optimizer, filepath: str | pathlib.Path, lr_scheduler=None, net_ema=None,
                 extra_modules: dict = None, N_epochs=100, run_ID=None, checkpoint_period: int = 120,
                 extra_checkpoint_period: int = 500, verbose=True):
        self.net = net
        self.net_ema = None if id(net) == id(net_ema) else net_ema
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.epoch = 0
        self.N_epochs = N_epochs
        self.epoch_stats = {}

        self.run_ID = run_ID
        self.verbose = verbose
        self.filepath = pathlib.Path(filepath)
        self.extra_checkpoint_period = extra_checkpoint_period
        self.checkpoint_period = checkpoint_period
        self.last_checkpoint_time = time.time()

        if extra_modules is not None:
            for name, mod in extra_modules.items():
                if not hasattr(mod, 'state_dict'):
                    raise ValueError(f"Extra module {name!r} does not define a state_dict() method.")
                if not hasattr(mod, 'load_state_dict'):
                    raise ValueError(f"Extra module {name!r} does not define a load_state_dict() method.")
            self.extra_modules = extra_modules
        else:
            self.extra_modules = {}

    def save_state(self, epoch=None):
        if epoch is not None:
            self.epoch = epoch
        checkpoint_dict = {'run_ID': self.run_ID,
                           'epoch': self.epoch,
                           'net_state': self.net.state_dict(),
                           'net_ema_state': self.net_ema.state_dict() if self.net_ema else None,
                           'optim_state': self.optimizer.state_dict(),
                           'scheduler_state': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                           'torch_rng_state': torch.get_rng_state(),
                           'torch_cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                           'np_rng_state': np.random.get_state(),
                           'epoch_stats': self.epoch_stats}

        # Add extra modules
        for name, mod in self.extra_modules.items():
            checkpoint_dict['extra_' + name] = mod.state_dict()  # Make sure names don't collide with default names

        torch.save(checkpoint_dict, self.filepath)
        if self.verbose:
            print(f'Training state at epoch {self.epoch} saved in {self.filepath}.')

        # Perform extra checkpoint
        if epoch is not None and self.epoch % self.extra_checkpoint_period == 0:
            suffix = '_{:d}'.format(int(epoch))
            filepath_extra = self.filepath.with_stem(self.filepath.stem + suffix)
            torch.save(checkpoint_dict, filepath_extra)
            if self.verbose:
                print(f'Extra training state at epoch {self.epoch} saved in {self.filepath}.')

    def load_state(self, resume_filepath: str | pathlib.Path = None):
        filepath = self.filepath if resume_filepath is None else resume_filepath
        checkpoint_dict = torch.load(filepath, torch.device('cpu'), weights_only=False)

        self.epoch = checkpoint_dict['epoch']
        if 'run_ID' in checkpoint_dict and resume_filepath is None:
            self.run_ID = checkpoint_dict['run_ID']

        # Epoch stats
        for stat, val in checkpoint_dict['epoch_stats'].items():
            if stat not in self.epoch_stats:
                self.epoch_stats[stat] = np.zeros((self.N_epochs,), dtype=float)

            if stat == 'epoch':
                self.epoch_stats[stat] = val
            else:
                self.epoch_stats[stat][:self.epoch] = val[:self.epoch]

        # Update model parameters and its ema (if any)
        self.net.load_state_dict(checkpoint_dict['net_state'], strict=False)
        if self.net_ema is not None and checkpoint_dict['net_ema_state'] is not None:
            self.net_ema.load_state_dict(checkpoint_dict['net_ema_state'], strict=False)

        # Optimizer
        if 'optim_state' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optim_state'])
            except Exception as e:
                warnings.warn(f'Optimizer state was not loaded!\n{e}')

        # Scheduler
        if self.lr_scheduler and 'scheduler_state' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['scheduler_state'])
            except Exception as e:
                warnings.warn(f'Scheduler state was not loaded!\n{e}')

        # Extra modules
        for name, mod in self.extra_modules.items():
            mod.load_state_dict(checkpoint_dict['extra_' + name])

        # Rng states
        if 'np_rng_state' in checkpoint_dict:
            np.random.set_state(checkpoint_dict['np_rng_state'])

        if 'np_rng_state' in checkpoint_dict:
            torch.set_rng_state(checkpoint_dict['torch_rng_state'])

        if 'cuda_rng_state' in checkpoint_dict and checkpoint_dict['cuda_rng_state'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint_dict['cuda_rng_state'])

        if self.verbose:
            print(f'Loaded training state at epoch {self.epoch} from {filepath!r}')

    def check(self, epoch: int, force=False):
        current_time = time.time()
        elapsed_time = current_time - self.last_checkpoint_time
        if force or elapsed_time > self.checkpoint_period:
            self.save_state(epoch)
            self.last_checkpoint_time = current_time

    def update_epoch_stats(self, epoch_stats: dict):
        epoch_ind = epoch_stats['epoch'] - 1  # The epoch value is a 1-based index
        for stat_name, stat_val in epoch_stats.items():
            if stat_name == 'epoch':
                self.epoch_stats[stat_name] = stat_val
                continue

            if stat_name not in self.epoch_stats:
                self.epoch_stats[stat_name] = np.zeros((self.N_epochs,), dtype=float)
            self.epoch_stats[stat_name][epoch_ind] = stat_val


class DictCheckpointer:
    """
    Saves a given dictionary to a .npz file at a given interval. During initialization, the given dictionary
    is updated with the content of the .npz file.
    """

    def __init__(self, filepath: pathlib.Path | str, dictionary: dict = None, interval=60):
        self.interval = interval
        self.last_checkpoint = time.time()
        self.filepath = pathlib.Path(filepath).with_suffix('.npz')
        self.dictionary = dict() if dictionary is None else dictionary
        self.numpy_arr_keys_key = 'numpy_arr_keys_9876543121'
        self.load()

    def is_checkpoint_ready(self):
        return time.time() > self.last_checkpoint + self.interval

    def save(self, force=False):
        if force or self.is_checkpoint_ready():
            numpy_arr_keys = [k for k, v in self.dictionary.items() if isinstance(v, np.ndarray)]
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.filepath, **self.dictionary, **{self.numpy_arr_keys_key: numpy_arr_keys})
            self.last_checkpoint = time.time()

    def load(self):
        if self.filepath.exists():
            with np.load(self.filepath, allow_pickle=True) as npzfile:
                npzfile_content = dict(npzfile)
                numpy_arr_keys = npzfile_content.pop(self.numpy_arr_keys_key)
                for k, v in npzfile_content.items():
                    if k not in numpy_arr_keys:
                        v = v.item()
                    self.dictionary[k] = v


class PiecewiseLR(torch.optim.lr_scheduler.LambdaLR):
    """
    Scheduler of a piecewise learning rate
    """

    def __init__(self, optimizer, epochs: list, lrs: list, funcs: list | str = 'poly1', *args, **kwargs):
        """
        Learning rate scheduler where each piece is a polynomial of the form a (x-x0)^n + b
        Args:
            optimizer: optimizer
            epochs: epochs when a transition occurs. epochs[0] must equal 0 to define the initial lr
            lrs: learning rates at the anchor points. lrs[i] denotes the learning rate at the beginning of segment i
            funcs: functions used in each piece
        """
        # Checks
        if isinstance(funcs, str):
            funcs = [funcs for _ in range(len(epochs) - 1)]
        elif len(funcs) != len(epochs) - 1:
            raise ValueError(f'The number of epochs must match the number of functions + 1.'
                             f'Expecting {len(epochs) - 1} functions. Received {len(funcs)} functions.')
        num_funcs = len(funcs)

        if len(epochs) != len(lrs):
            raise ValueError(f'The number of anchor epochs {len(epochs)} '
                             f'and the number of anchor learning rates {len(lrs)} do not match.')
        assert epochs[0] == 0.0, 'The first epoch must be 0.'
        assert (np.diff(np.array(epochs))[1:] > 0).all(), 'The epochs array must be monotonically increasing'

        self.transit_epochs = epochs[1:]

        lr_init = list(set([p['lr'] for p in optimizer.param_groups]))
        if len(lr_init) > 1:
            raise NotImplementedError('Parameter groups are not supported .')
        self.lr_init = float(lr_init[0])

        # Define the function in each interval
        funcs_lambda = []

        def init_func(func, i):
            x0 = epochs[i]
            if 'poly' in func:
                n = float(func.removeprefix('poly'))
                a = copy.copy((lrs[i + 1] - lrs[i]) / ((epochs[i + 1] - epochs[i]) ** n))
                b = lrs[i]
                func_lambda = lambda x: a * (x - x0) ** n + b
            elif 'exp' in func:
                b = lrs[i]
                a = np.log(lrs[i + 1] / b) / (epochs[i + 1] - epochs[i])
                func_lambda = lambda x: b * np.exp(a * (x - x0))
            elif 'cos' in func:
                # lr = b*(1+cos(a(x-x0))
                b = lrs[i] / 2
                a = np.arccos(lrs[i + 1] / b - 1) / (epochs[i + 1] - epochs[i])
                func_lambda = lambda x: b * (1 + np.cos(a * (x - x0)))
            else:
                raise NotImplementedError(f"func={func} is not implemented.")
            return func_lambda

        for i, func in enumerate(funcs):
            funcs_lambda.append(init_func(func, i))
        funcs_lambda.append(lambda x: lrs[-1])

        # # Interpolate the piecewise functions
        # funcs_pow = [float(p.removeprefix('poly')) for p in funcs] + [1.0]
        # a, b = [0.0 for _ in range(num_funcs + 1)], [0.0 for _ in range(num_funcs + 1)]
        # for i in range(num_funcs):
        #     a[i] = (lrs[i + 1] - lrs[i]) / ((epochs[i + 1] - epochs[i]) ** funcs_pow[i])
        #     b[i] = lrs[i]
        #     # a[i] = (lrs[i + 1] - lrs[i]) / (epochs[i + 1] ** funcs_pow[i] - epochs[i] ** funcs_pow[i])
        #     # b[i] = lrs[i] - a[i] * (epochs[i]) ** funcs_pow[i]
        # a[-1], b[-1] = 0, lrs[-1]
        # poly_func = lambda x, x0, a, b, n: a * (x - x0) ** n + b
        # # poly_func = lambda x, x0, a, b, n: a * (x) ** n + b
        # funcs_params = list(zip(a, b, funcs_pow))

        def lr_func(epoch):
            ind = bisect.bisect_right(epochs, epoch) - 1
            # lr = a[ind] * ((epoch - epochs[ind]) ** funcs_pow[ind]) + b[ind]
            # lr = poly_func(epoch, epochs[ind], *funcs_params[ind])
            lr = funcs_lambda[ind](epoch)
            return lr / self.lr_init  # Divide by base_lr to undo the scaling in LambdaLR

        self.lr_func = lr_func
        super(PiecewiseLR, self).__init__(optimizer, lr_lambda=lr_func, *args, **kwargs)

    def plot_lr(self, epoch_max: int = None, log=False, include_transitions=True):
        if epoch_max is None:
            epoch_max = 1.1 * self.transit_epochs[-1]
        epochs = np.arange(0, epoch_max + 1)
        lrs = np.vectorize(self.lr_func)(epochs) * self.lr_init
        plt.plot(epochs, lrs)

        # Add vertical lines for each separator
        if include_transitions:
            for epoch_transit in self.transit_epochs:
                plt.axvline(epoch_transit, linestyle='--', color='r')
        ax = plt.gca()
        ax.set_xlabel('Epoch'), ax.set_ylabel('Learning rate')
        if log:
            plt.yscale('log')
        plt.show(block=False)

    def state_dict(self):
        state_dict = super(PiecewiseLR, self).state_dict()
        del state_dict['lr_func']
        return state_dict


class LossesDict(defaultdict):
    def __init__(self, *args, device='cpu', **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

    def to(self, device: torch.device):
        for k, v in self.items():
            self[k] = v.to(device)

    def reset(self):
        for v in self.values():
            v.zero_()


def repeat_iterator(Iterator):
    "Reference: https://discuss.pytorch.org/t/implementing-an-infinite-loop-dataset-dataloader-combo/35567"
    for iter_temp in itertools.repeat(Iterator):
        for elem in iter_temp:
            yield elem


################################################## GROMACS ##################################################
gmx_cmd_choices = [c for c in ['gmx_mpi', 'gmx'] if shutil.which(c)]
if gmx_cmd_choices:
    gmx_cmd = gmx_cmd_choices[0]
else:
    gmx_cmd = ''
    warnings.warn(f"No Gromacs command found!")
gmx_cmd_single = 'gmx' if 'gmx' in gmx_cmd_choices else gmx_cmd

GMX_LENGTH_units = 'nm'
GMX_TIME_units = 'ps'

# Disable GROMACS backups if not already set as environment variable
if 'GMX_MAXBACKUP' not in os.environ:
    os.environ['GMX_MAXBACKUP'] = '-1'


def format_gmx_cmd(cmd: str, prefix: str = None, gmx_cmd=gmx_cmd, **kwargs):
    if gmx_cmd not in gmx_cmd_choices:
        raise ValueError(f"Cannot format GROMACS command. GROMACS shell command {gmx_cmd!r} is undefined.")

    if prefix:
        cmd_str = prefix
        if not cmd_str.endswith(' '):
            cmd_str += ' '
    else:
        cmd_str = ''

    cmd_str += f'{gmx_cmd} {cmd}'
    for arg_name, arg_val in kwargs.items():
        cmd_str += f' -{arg_name}'
        if arg_val is not None:
            cmd_str += f' {arg_val}'
    return cmd_str


def gmx_anonymize(input_filepath: str | pathlib.Path, output_filepath: str | pathlib.Path) -> None:
    """
    Anonymizes a GROMACS file by stripping identifying names, system descriptions, and miscellaneous comments,
    in its header while keeping the command line details intact.
    Args:
        input_filepath: Input filepath
        output_filepath: Filepath of the anonymized file
    Returns:
        None
    """
    with open(input_filepath, 'r') as infile:
        lines = infile.readlines()
    header_attrs = parse_gmx_header(input_filepath)

    cmd_path_args = ['-f', '-o', '-p', '-i', '-c']  # command args referencing filepaths
    anonymized_lines = []
    i = 0
    finished_parsing_header = False
    while i < len(lines):
        line = lines[i]
        line_is_comment = line.startswith(';')
        if not finished_parsing_header and not line_is_comment:
            finished_parsing_header = True

        # Add the rest of the lines
        if finished_parsing_header:
            anonymized_lines.extend(lines[i:])
            break

        # Add line that specifies GROMACS version
        if line.startswith(";	Created by:"):
            anonymized_lines.extend(lines[i:i + 2])
            i += 1

        # Parse the command string and change all paths to local filename.
        # Referenced filenames should be in the same directory as the anonymized file
        if line.startswith(";	Command line:"):
            anonymized_lines.append(line)
            cmd_str: str = header_attrs['cmd']
            cmd_str_elems = cmd_str.split(" ")
            for j, arg_val in enumerate(cmd_str_elems):
                for replaced_arg in cmd_path_args:
                    if arg_val == replaced_arg:
                        cmd_str_elems[j + 1] = str(pathlib.Path(cmd_str_elems[j + 1]).name)

            cmd_str_new = " ".join(cmd_str_elems)
            next_line = lines[i + 1]
            rep_ind = next_line.index(cmd_str_elems[0])
            next_line = next_line[:rep_ind] + cmd_str_new
            anonymized_lines.append(next_line)
            i += 1

        i += 1

    # If file content didn't change, do not overwrite
    write_new_lines = input_filepath != output_filepath or lines != anonymized_lines
    if write_new_lines:
        with open(output_filepath, 'w') as outfile:
            outfile.writelines(anonymized_lines)

        # For .top files, also convert include paths to local paths
        if output_filepath.suffix == '.top':
            gmx_top_convert_include_paths(output_filepath)


def gmx_regen_top(top_filepath: str | pathlib.Path, output_filepath: str | pathlib.Path,
                  terminal_states: list[str], working_dir: str = None, include_vsites=False, water: str = ''):
    """
    Regenerates a .top file with options to not include virtual sites and water
    Args:
        top_filepath: filepath of the .top file to be regenerated
        output_filepath: filepath of the output .top file
        terminal_states: GROMACS code to determine the protonation states of the terminals
        working_dir: working directory where gromacs command will be launched
        include_vsites: include virtual sites in the output .top file
    Returns: None

    """
    top_filepath = pathlib.Path(top_filepath)
    output_filepath = pathlib.Path(output_filepath)
    assert top_filepath.is_absolute(), "The input filepath needs to be absolute."

    # Create a working directory
    working_dir_is_temp = working_dir is None
    if working_dir_is_temp:
        working_dir = tempfile.mkdtemp()
    working_dir = pathlib.Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    # If the output filepath is not absolute, output to the same directory as the input file
    if not output_filepath.is_absolute():
        output_filepath = top_filepath.parent / output_filepath
    output_filename = output_filepath.name
    temp_output_filepath = working_dir / output_filename
    current_dir = os.getcwd()
    os.chdir(working_dir)

    # Read the lines in the topology file to find the working directory and the command line
    top_header_attrs = parse_gmx_header(filepath=top_filepath)
    pdb2gmx_cmd = top_header_attrs['cmd']
    forcefield_name = top_header_attrs['forcefield']

    cmd_str_items = pdb2gmx_cmd.split()[2:]
    cmd_kwargs = {}
    while cmd_str_items:
        arg_name = cmd_str_items.pop(0)
        if arg_name.startswith('-'):
            if not cmd_str_items[0].startswith('-'):
                cmd_kwargs[arg_name[1:]] = cmd_str_items.pop(0)
            else:
                cmd_kwargs[arg_name[1:]] = None

    # Find the associated .pdb filepath. The search priority is the following:
    # 1) The directory where the .top file is located
    # 2) The 'pdb' directory in data_dir
    pdb_filepath = pathlib.Path(cmd_kwargs['f'])
    pdb_filepath_local_dir = pathlib.Path(top_filepath).parent / pdb_filepath.name
    pdb_filepath_data_dir = pathlib.Path(data_dir, 'pdb') / pdb_filepath.name
    if pdb_filepath_local_dir.exists():
        pdb_filepath = pdb_filepath_local_dir
    elif pdb_filepath_data_dir.exists():
        pdb_filepath = pdb_filepath_data_dir
    else:
        raise ValueError(f"Pdb file {pdb_filepath} does not exists.")
    cmd_kwargs['f'] = str(pdb_filepath)

    # Add forcefield name if not already in command line
    if 'ff' not in cmd_kwargs:
        assert forcefield_name, "Could not not find a forcefield name in the .top file"
        cmd_kwargs['ff'] = forcefield_name

    # Water and virtual sites
    if not include_vsites and 'vsite' in cmd_kwargs:
        del cmd_kwargs['vsite']
    if not water:
        water = 'none'
    cmd_kwargs['water'] = water

    # Change the name of the output gro file and topology file
    suffix = '_' + os.path.basename(tempfile.mktemp())
    gro_filepath = pathlib.Path(cmd_kwargs['o'])
    gro_filepath = gro_filepath.with_stem(gro_filepath.stem + suffix)
    cmd_kwargs['o'] = gro_filepath.name  # Keep only the filename of the output .gro file
    cmd_kwargs['p'] = output_filename

    protonation_inputs = "\n".join(terminal_states) + "\n"
    cmd_str = format_gmx_cmd('pdb2gmx', prefix=f"printf '{protonation_inputs}' | ", **cmd_kwargs)
    run_cmd(cmd_str, print_output=True, output_filepath=output_filename)

    # Anonymize the temporary .top file
    gmx_anonymize(temp_output_filepath, temp_output_filepath)

    # Move the temporary output file to the given output filepath and cleanup temporary files
    temp_output_filepath.rename(output_filepath)
    if working_dir_is_temp:
        shutil.rmtree(working_dir)
    else:
        gro_filepath = working_dir / cmd_kwargs['o']
        posre_filename = working_dir / cmd_kwargs['i']
        gro_filepath.unlink()
        posre_filename.unlink()

    os.chdir(current_dir)


def gmx_test_top(top_filepath: str | pathlib.Path):
    """
    Tests opening of .top file and convert forcefield paths if needed
    Args:
        top_filepath: filepath of the .top file

    Returns:
        None
    """
    has_tried_fix = False
    from openmm.app.gromacstopfile import GromacsTopFile
    while True:
        try:
            # Try loading the topology with openmm to see if it has any include problems
            GromacsTopFile(top_filepath, includeDir=data_dir)

            # Try loading the topology with mda.Universe
            with warnings.catch_warnings(category=UserWarning, action="ignore"):
                mda.Universe(topology=top_filepath, topology_format='ITP', include_dir=data_dir)
            break
        except Exception as e:
            if has_tried_fix:
                # Throw error if fix didn't fix the problem
                raise e
            else:
                # Try fixing by converting paths
                gmx_top_convert_include_paths(top_filepath=top_filepath)
                has_tried_fix = True


def gmx_top_convert_include_paths(top_filepath: str | pathlib.Path):
    """
    Converts include paths found in a GROMACS topology file to local paths
    Args:
        top_filepath: filepath of the .top file

    Returns:
        None
    """

    def replace_path(match_obj: re.Match):
        filepath = pathlib.Path(match_obj.group(0))
        if filepath.parents[0].suffix == '.ff':
            # This is a forcefield path. Remove root path
            new_filepath = filepath.relative_to(filepath.parents[1])
        else:
            new_filepath = filepath
        return str(new_filepath)

    with open(top_filepath, 'r') as top_file:
        top_file_lines = top_file.readlines()

    resave = False
    re_path_pattern = re.compile(r'(?<=").*(?=")')  # Include paths are enclosed with ""
    for i, line in enumerate(top_file_lines):
        if line.startswith('#include'):
            top_file_lines[i] = re_path_pattern.sub(replace_path, line)
            resave |= top_file_lines[i] != line

    if resave:
        with open(top_filepath, 'w') as top_file:
            top_file.writelines(top_file_lines)


def gmx_top_modify_posres(top_filepath: str | pathlib.Path, pos_res_filename: str):
    """
    Modifies the position restraint file in a GROMACS topology file
    Args:
        top_filepath: filepath of the .top file
        pos_res_filename: filename of the new position restraint file

    Returns:
        None
    """
    with open(top_filepath, 'r') as top_file:
        top_file_lines = top_file.readlines()
    pos_res_filename = pathlib.Path(pos_res_filename).with_suffix('.itp')

    for i, line in enumerate(top_file_lines):
        if line.startswith('#ifdef POSRES'):
            top_file_lines[i + 1] = f'#include "{pos_res_filename.name}"\n'

    with open(top_filepath, 'w') as top_file:
        top_file.writelines(top_file_lines)


def gmx_top_modify_molecules(top_filepath: str | pathlib.Path, molecules: dict[str, int]):
    """
    Modifies the [molecules] section of a GROMACS topology file
    Args:
        top_filepath: filepath of the .top file
        molecules: dictionary specifying each line in the [molecules] section

    Returns:
        None
    """
    with open(top_filepath, 'r') as top_file:
        top_file_lines = top_file.readlines()

    for i, line in enumerate(top_file_lines):
        if line.startswith('[ molecules ]'):
            for j, (mol_name, mol_n) in enumerate(molecules.items()):
                top_file_lines[i + j + 1] = f'{mol_name}    {mol_n:d}\n'
            top_file_lines = top_file_lines[:i + len(molecules) + 1]
            break

    with open(top_filepath, 'w') as top_file:
        top_file.writelines(top_file_lines)


def gmx_info(gmx_cmd=gmx_cmd):
    """
        Pulls useful info about GROMACS binary using output of --version
        Return:
            dictionary of info parsing the output of --version
    """
    info = dict()
    if not gmx_cmd:
        raise FileNotFoundError(f"GROMACS executable {gmx_cmd!r} not found.")

    try:
        # Parse the output of --version to get configs info
        result = subprocess.run([gmx_cmd, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                check=True)
        config_regex = re.compile(r'\A(?P<config_name>.+):\s+(?P<config_val>.*)')
        for line in result.stdout.splitlines():
            config_match = config_regex.match(line)
            if config_match:
                info[config_match.group('config_name')] = config_match.group('config_val')

    except subprocess.CalledProcessError as e:
        print(f"Failed to execute GROMACS binary: {e.stderr}")

    return info


def parse_gmx_header(filepath: str | pathlib.Path):
    """
    Reads a Gromacs .top file or .mdp file and parses the header to find useful information
    Args:
        filepath: filepath of the .top or .mdp file

    Returns:
        dictionary of the attributes (see below) found in the header
    """
    # Read the lines in the file to find the working directory and the command line
    filepath = pathlib.Path(filepath)
    attrs_dict = {}
    prev_line = ''
    with open(filepath, 'r+') as f:
        for i, l in enumerate(f):
            if l.startswith(';\tWorking dir'):
                matches = re.findall(r'\/.*[\w:]+', l)
                attrs_dict['working_dir'] = matches[0]
            if l.startswith(';\t  gmx') and 'Command line' in prev_line:
                matches = re.search(r'gmx', l)
                attrs_dict['cmd'] = l[matches.start():]
            if prev_line.startswith(';') and 'forcefield' in prev_line:
                matches = re.search(r'(?<=\").*(?=\")', l)
                forcefield_itp_path = pathlib.Path(matches[0])
                attrs_dict['forcefield'] = forcefield_itp_path.parent.name.rstrip('.ff')
            if l.startswith('[ moleculetype ]'):
                break
            prev_line = l

    # Checks
    if 'forcefield' not in attrs_dict and filepath.suffix == '.top':
        warnings.warn(f"force field not found in {filepath}")

    # Parse the command line to get the arguments
    if 'cmd' in attrs_dict:
        attrs_dict['cmd_args'] = dict()
        cmd_args = attrs_dict['cmd'].removesuffix('\n').split(' ')
        for i, arg_name in enumerate(cmd_args):
            if arg_name.startswith('-'):
                if i == len(cmd_args) - 1 or cmd_args[i + 1].startswith('-'):
                    attrs_dict['cmd_args'][arg_name] = None
                else:
                    attrs_dict['cmd_args'][arg_name] = cmd_args[i + 1]

    return attrs_dict


def parse_gmx_tpr_mol_types(filename):
    tpr_dump = subprocess.run([gmx_cmd, 'dump', '-s', filename], capture_output=True)
    if tpr_dump.returncode:
        raise ValueError(tpr_dump.stderr)
    tpr_dump.stdout = tpr_dump.stdout.decode()
    std_out_lines = tpr_dump.stdout.split('\n')

    mol_type_section_space_header = 0
    in_moltype = False
    mol_types = []
    for i, line in enumerate(std_out_lines):
        line_strip = line.lstrip(' ')
        spacer_length = len(line) - len(line_strip)

        if in_moltype:
            if spacer_length > mol_type_section_space_header:
                line_strip_2 = line[mol_type_section_space_header:]
                # Rename sections
                if line_strip.startswith('atom ('):
                    line_strip_2 = (spacer_length - mol_type_section_space_header) * ' '
                    # Look ahead to figure out which data section we are in.
                    if 'atomnumber' in std_out_lines[i + 1]:
                        line_strip_2 += 'attributes'
                    else:
                        line_strip_2 += 'names'
                if line_strip.startswith('type ('):
                    line_strip_2 = (spacer_length - mol_type_section_space_header) * ' ' + 'types'
                if line_strip.startswith('residue ('):
                    line_strip_2 = (spacer_length - mol_type_section_space_header) * ' ' + 'residues'

                mol_types[-1].append(line_strip_2)
            else:
                in_moltype = False

        if not in_moltype and line_strip.startswith('moltype'):
            in_moltype = True
            mol_type_section_space_header = spacer_length
            mol_types.append([])

    for i, mol_type_lines in enumerate(mol_types):
        mol_types[i] = parse_tpr_section(lines=mol_type_lines)

    mol_types.pop(0)  # There is a 'mol type' that appears at the beginning before the molecule type sections

    for mol_type in mol_types:
        n_atoms = len(mol_type['atoms']['attributes'])
        data = [{} for _ in range(n_atoms)]

        # Parse attributes
        Atom = 'Atom'
        VSite = 'VSite'
        for attr in mol_type['atoms']['attributes']:
            exec_str = attr.replace(' ', '').replace('{', 'dict(').replace('}', ')').replace('atom[', 'data[')
            exec(exec_str)
        mol_type['atoms']['attributes'] = np.array(copy.deepcopy(data))

        # Parse names
        for attr in mol_type['atoms']['names']:
            exec_str = attr.replace(' ', '').replace('{', 'dict(').replace('}', ')').replace('atom[', 'data[')
            exec(exec_str)
        mol_type['atoms']['names'] = np.array([a.pop('name') for a in data])

        # Parse types
        for attr in mol_type['atoms']['types']:
            exec_str = attr.replace(' ', '').replace('{', 'dict(').replace('}', ')').replace('type[', 'data[')
            exec(exec_str)
        mol_type['atoms']['types'] = np.array([t.pop('name') for t in data])
        mol_type['atoms']['typesB'] = np.array([t.pop('nameB') for t in data])

        # Parse residues
        for attr in mol_type['atoms']['residues']:
            exec_str = attr.replace(' ', '').replace('{', 'dict(').replace('}', ')').replace('residue[', 'data[')
            exec(exec_str)
        mol_type['atoms']['residues'] = np.array(copy.deepcopy(data))
    return mol_types


def parse_tpr_section(lines: list[str]):
    if not lines:
        return lines
    start_pos = len(lines[0]) - len(lines[0].lstrip(' '))
    sections_dict = dict()
    # section_pointers = [sections]
    while lines:
        line = lines.pop(0)
        if line.endswith(':'):
            line = line[:-1]
        line_stripped = line.lstrip(' ')
        line_front_length = len(line) - len(line_stripped)
        if line_front_length == start_pos:
            # Start new section with line as the section name
            curr_section = line_stripped
            sections_dict[curr_section] = []
        elif line_front_length > start_pos:
            sections_dict[curr_section].append(line)

        # if line_front_length == front_length:
        #     section_pointers[-1].append(line)
        # elif line_front_length > front_length:
        #     # Start new section
        #     new_section = [line]
        #     section_pointers[-1].append(new_section)
        #     section_pointers.append(new_section)
        # else:
        #     # Go to the previous section
        #     section_pointers[-1].pop(-1)
        #     section_pointers[-1].append(line)
        # front_length = line_front_length

    # Recurse on the subsections
    is_list = True
    for sub_name, sub_lines in sections_dict.items():
        if sub_lines:
            sections_dict[sub_name] = parse_tpr_section(sub_lines)
            is_list = False

    if is_list:
        sections = list(sections_dict.keys())
    else:
        sections = sections_dict
    return sections


def parse_gmx_mdp(mdp_filepath: str | pathlib.Path) -> dict:
    """
    Parses a GROMACS .mdp parameter file to gather all defined parameters in dictionary
    Args:
        mdp_filepath: filepath of the .mdp file

    Returns:
        dict['param_name':param_val]
    """
    with open(mdp_filepath) as file:
        lines = file.readlines()
    params_dict = dict()
    param_line_patt = re.compile(r'\A([a-zA-Z0-9_\-]+)\s*=\s*(.*)')
    param_num_patt = re.compile(r'[+\-]?\d+[.]?\d*([eE][+\-]\d*)?')
    for i, line in enumerate(lines):
        matches = param_line_patt.match(line)
        if matches is None:
            continue
        param_name, param_val = matches.groups()

        param_num_match = param_num_patt.fullmatch(param_val)
        if param_num_match:
            if '.' in param_val or 'e' in param_val:
                param_val = float(param_val)
            else:
                param_val = int(param_val)

        if param_name not in params_dict:
            params_dict[param_name] = param_val
        else:
            raise ValueError(f"{param_name} is multiply-defined at line {i + 1}")
    return params_dict


def modify_gmx_mdp(mdp_filepath: str | pathlib.Path, **params):
    """
    Modifies a GROMACS .mdp parameter file by adding or overwriting parameters with given parameters
    Args:
        mdp_filepath: filepath of the .mdp file
        **params: kwargs identifying modified parameter names and values
    Returns:
        None
    """
    if not params:
        return
    params = dict(params)

    with open(mdp_filepath) as file:
        lines = file.readlines()
    new_lines = copy.copy(lines)
    param_line_patt = re.compile(r'\A(?P<param_name>[a-zA-Z0-9_\-]+)\s*=\s*(?P<param_val>.*)')
    param_num_patt = re.compile(r'[+\-]?\d+[.]?\d*([eE][+\-]\d*)?')
    for i, line in enumerate(lines):
        param_line_match = param_line_patt.match(line)
        if param_line_match is None:
            continue

        param_name, param_val = param_line_match.groups()
        if param_name not in params:
            continue
        param_new_val = params.pop(param_name)

        # Check if the parameter value is numerical and cast to relevant type
        param_num_match = param_num_patt.fullmatch(param_val)
        if param_num_match:
            if '.' in param_val or 'e' in param_val:
                param_new_val = float(param_new_val)
            else:
                param_new_val = int(param_new_val)
        param_val_start_ind = param_line_match.start('param_val')
        new_lines[i] = line[:param_val_start_ind] + str(param_new_val)

    # Add parameters that remain
    for param_name, param_val in params.items():
        new_lines.append(f"{param_name:<20}=  {param_val}")

    # Ensure all lines, except the last, end with \n
    for i in range(len(new_lines) - 1):
        new_lines[i] = new_lines[i].removesuffix('\n') + '\n'

    if new_lines != lines:
        with open(mdp_filepath, 'w') as outfile:
            outfile.writelines(new_lines)


def def_atom_name_to_type(topology: str | pathlib.Path | mda.Universe, solvent=False):
    """
    Defines a dictionary that maps atom names to types using a GROMACS .top file
    Args:
        topology: filename of the topology file (.top) or MDAnalysis universe
        solvent: define map for solvent molecules
    Returns:

    """
    if isinstance(topology, str | pathlib.Path):
        with warnings.catch_warnings(category=UserWarning, action="ignore"):
            top_uni = mda.Universe(topology, topology_format='ITP', include_dir=data_dir)

        if not solvent:
            top_uni = top_uni.select_atoms('not (resname SOL or resname NA or resname CL)')
    elif isinstance(topology, mda.Universe):
        top_uni = topology
    else:
        raise ValueError(f"Unsupported topology object of class {topology.__class__!r}")

    atoms = top_uni.atoms
    atom_name_to_type = {(r, n): t for r, n, t in zip(atoms.resindices, atoms.names, atoms.types)}
    return atom_name_to_type


def gmx_pdb2gmx(pdb_filepath: str | pathlib.Path, forcefield_dir: str | pathlib.Path, output_basename: str = None,
                working_dir: str | pathlib.Path = None, terminals_protonation: str | list[str] = None, verbose=False,
                ignh=True, water='tip3p', raise_if_error=True, **pdb2gmx_kwargs):
    if isinstance(forcefield_dir, str):
        forcefield_dir = pathlib.Path(forcefield_dir)
    forcefield_stem = forcefield_dir.stem

    curr_dir = os.getcwd()
    if working_dir is not None and str(working_dir) == curr_dir:
        # No need to change directory
        working_dir = None
    if working_dir is not None:
        os.chdir(working_dir)

    if isinstance(pdb_filepath, str):
        pdb_filepath = pathlib.Path(pdb_filepath)
    if output_basename is None:
        output_basename = pdb_filepath.stem

    starting_structure_filename = f"{output_basename}.gro"
    topology_filename = f"{output_basename}.top"
    position_restraints_filename = f"{output_basename}_posre.itp"

    kwargs = dict(f=str(pdb_filepath), o=starting_structure_filename, p=topology_filename,
                  i=position_restraints_filename, water=water, v=None)
    if ignh:
        kwargs['ignh'] = None

    kwargs = {**kwargs, **pdb2gmx_kwargs}

    if terminals_protonation is not None:
        if isinstance(terminals_protonation, list):
            if len(terminals_protonation) != 2:
                raise ValueError(f"2 strings are expected to define terminals. Received:{terminals_protonation}")
            terminals_protonation = "\n".join(terminals_protonation)

        if not terminals_protonation.endswith('\n'):
            terminals_protonation = terminals_protonation + '\n'
        kwargs['ter'] = None  # Add terminal option to activate the interactive mode
        prefix = f"printf '{terminals_protonation}' | "
    elif 'ter' in kwargs:
        raise ValueError(f"No terminals code were given. Terminals code are need with the -ter option.")
    else:
        prefix = None
    kwargs['ff'] = f'\'{forcefield_stem}\''

    # pdb2gmx_cmd = format_gmx_cmd('pdb2gmx', prefix=prefix, **kwargs)
    pdb2gmx_cmd = format_gmx_cmd('pdb2gmx', prefix=prefix, gmx_cmd=gmx_cmd_single, **kwargs)
    run_cmd(pdb2gmx_cmd, output_filepath=starting_structure_filename, print_output=verbose,
            raise_if_error=raise_if_error)

    if working_dir:
        os.chdir(curr_dir)


def gmx_solvate(gro_filename: str, top_filename: str, output_gro_filename: str, verbose=False, **kwargs):
    solvate_cmd = format_gmx_cmd('solvate',
                                 cp=gro_filename, cs='spc216.gro', o=output_gro_filename, p=top_filename,
                                 **kwargs)
    run_cmd(solvate_cmd, output_filepath=output_gro_filename, print_output=verbose)


def gmx_genion(tpr_filename: str, output_gro_filename: str, output_top_filename: str, verbose=False, **kwargs):
    # Run genion
    genion_cmd = format_gmx_cmd('genion', prefix="printf 'SOL\n' | ",
                                s=tpr_filename, p=output_top_filename, o=output_gro_filename, **kwargs)
    run_cmd(genion_cmd, output_filepath=output_gro_filename, print_output=verbose)


def gmx_grompp(params_filename: str | pathlib.Path, gro_filename: str | pathlib.Path, top_filename: str | pathlib.Path,
               output_tpr_filename: str | pathlib.Path, params_out_filename='', restraint_filename='', verbose=True,
               **extra_kwargs):
    kwargs = dict(f=str(params_filename), c=str(gro_filename), p=str(top_filename), o=str(output_tpr_filename))
    if params_out_filename:
        kwargs['po'] = params_out_filename
    if restraint_filename:
        kwargs['r'] = restraint_filename

    grompp_cmd = format_gmx_cmd('grompp', gmx_cmd=gmx_cmd_single, **kwargs, **extra_kwargs)
    run_cmd(grompp_cmd, output_tpr_filename, print_output=verbose, raise_if_error=True)


def gmx_editconf(gro_filename: str, gro_output_filename: str, d: float, verbose=False, **extra_kwargs):
    kwargs = dict(f=gro_filename, o=gro_output_filename, c=None, d=d, bt='dodecahedron')
    box_cmd = format_gmx_cmd('editconf', gmx_cmd=gmx_cmd_single, **kwargs, **extra_kwargs)
    run_cmd(box_cmd, output_filepath=gro_output_filename, print_output=verbose)


def gmx_mdrun(gro_filepath: str | pathlib.Path, top_filename: str | pathlib.Path,
              params_filepath: str | pathlib.Path, output_basename=None, verbose=False,
              restraint_filename: str = None, progress_filepath=None, grompp_kwargs: dict = None, **mdrun_kwargs):
    # Defaults and paths
    gro_filepath = pathlib.Path(gro_filepath)
    top_filename = pathlib.Path(top_filename)
    params_filepath = pathlib.Path(params_filepath)
    working_dir = gro_filepath.parent  # The working directory corresponds to the directory of the .gro file
    gro_filename = gro_filepath.name
    if output_basename is None:
        output_basename = f"{gro_filepath.stem}_mdrun"
    output_tpr_filename = f"{output_basename}.tpr"
    if progress_filepath is None:
        progress_filepath = working_dir / f"{output_basename}.gro"
    if verbose:
        mdrun_kwargs['v'] = None
    if grompp_kwargs is None:
        grompp_kwargs = {}

    # Move to the working directory
    os.chdir(working_dir)

    # Run grompp
    gmx_grompp(params_filename=params_filepath, gro_filename=gro_filename, top_filename=top_filename.name,
               output_tpr_filename=output_tpr_filename, restraint_filename=restraint_filename, verbose=verbose,
               **grompp_kwargs)

    # Run mdrun
    mdrun_cmd = format_gmx_cmd(cmd='mdrun', gmx_cmd=gmx_cmd, deffnm=output_basename, **mdrun_kwargs)
    run_cmd(mdrun_cmd, output_filepath=progress_filepath, print_output=verbose, raise_if_error=True)


def uni_from_seq(sequence: list[str], forcefield: 'Forcefield'):
    forcefield_residues = forcefield.residue_info
    atoms_name = []
    atoms_resname = []
    atoms_resind = []
    for i, resname in enumerate(sequence):
        res = forcefield_residues[resname]
        n_atoms = len(res['atoms']['name'])
        atoms_name.extend(res['atoms']['name'])
        atoms_resname.extend(n_atoms * [resname])
        atoms_resind.extend(n_atoms * [i])

    # Add topology attributes
    n_residues = len(sequence)
    n_atoms = len(atoms_name)
    uni = mda.Universe.empty(n_atoms, n_residues=n_residues, atom_resindex=atoms_resind, trajectory=True)
    uni.add_TopologyAttr('name', atoms_name)
    # uni.add_TopologyAttr('elements', [n[0] for n in atoms_name])
    uni.add_TopologyAttr('resname', sequence)
    uni.add_TopologyAttr('resid', range(1, n_residues + 1))
    uni.add_TopologyAttr('occupancies', np.ones(n_atoms))
    uni.add_TopologyAttr('chainIDs', n_atoms * ['A'])
    default_attr_names = ['altLocs', 'icodes', 'tempfactors', 'formalcharges', 'record_types', 'segids']
    for name in default_attr_names:
        uni.add_TopologyAttr(name)

    # Add default atom positions and box dimensions
    dim = np.array([1, 1, 1, 90, 90, 90])
    transform = mda.transformations.boxdimensions.set_dimensions(dim)
    uni.trajectory.add_transformations(transform)
    uni.atoms.positions = np.zeros((n_atoms, 3))

    return uni


def addH_to_pdb(pdb_filepaths: list[str], parallel=True, tqdm_desc='', **pdb2gmx_kwargs):
    """
    Uses GROMACS' pdb2gmx to add hydrogen atoms to a given set of .pdb
    Args:
        pdb_filepaths: filepaths of .pdb files
        parallel: runs pdb2gmx in parallel. Parallel is deactivated if there are less than 200 .pdb files.
        tqdm_desc: description string given to tqdm.tqdm progress bar
        **pdb2gmx_kwargs: kwargs passed to pdb2gmx

    Returns:
        None
    """
    worker_func = functools.partial(gmx_pdb2gmx, **pdb2gmx_kwargs)
    parallel &= len(pdb_filepaths) > 200  # Deactivate parallel if a small number of files is given
    if parallel:
        pool = multiprocessing.Pool(n_parallel_processes)
        tasks_iter = pool.imap_unordered(worker_func, pdb_filepaths, chunksize=32)
    else:
        tasks_iter = map(worker_func, pdb_filepaths)

    tasks_iter = tqdm.tqdm(tasks_iter, total=len(pdb_filepaths), desc=tqdm_desc)
    for _ in tasks_iter:
        pass

    if parallel:
        pool.close()


################################################## MISCELLANEOUS ##################################################

def get_file_creation_date(filepath: str | pathlib.Path):
    filepath = pathlib.Path(filepath)
    stat = filepath.stat()

    if hasattr(stat, 'st_birthtime'):
        timestamp = stat.st_birthtime
    elif hasattr(stat, 'st_ctime'):
        timestamp = stat.st_ctime
    else:
        raise ValueError(f'Cannot find creation date of {filepath}')

    return datetime.datetime.fromtimestamp(timestamp)


def calculate_units_conversion_factor(units_from, units_to):
    if units_from == units_to:
        conversion_factor = 1.0
    else:
        if not isinstance(units_from, pint.Unit):
            units_from = pint_reg.parse_units(units_from)
        if not isinstance(units_to, pint.Unit):
            units_to = pint_reg.parse_units(units_to)
        conversion_factor = (1.0 * units_from).to(units_to).magnitude
    return conversion_factor


def run_cmd(cmd, output_filepath: str | pathlib.Path = None, resume=True, print_output=False, raise_if_error=False):
    if not resume or (output_filepath is not None and not os.path.exists(str(output_filepath))):
        cmd_out = subprocess.run(cmd, shell=True, capture_output=True)
        if cmd_out.returncode != 0:
            if raise_if_error:
                raise RuntimeError(cmd_out.stderr.decode('utf8'))
            else:
                print("Error occured\n:", cmd_out.stderr.decode('utf8'))

        if print_output:
            print(cmd_out.stdout.decode('utf8'))
            print(cmd_out.stderr.decode('utf8'))


class Archiver:
    def __init__(self, dir: str | pathlib.Path, chunk_basename='chunk', buffer_saving_period=120):
        self.dir = pathlib.Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tensors_keys_key = 'tensors_keys_9876543121'
        self.buffer_saving_period = buffer_saving_period
        self.last_buffer_save_time = time.time()
        self.chunk_basename = chunk_basename
        self.chunk_filenames = [path.name for path in self.dir.glob(f"{self.chunk_basename}*.npz")]
        self.keys = {}

        # Initialize the keys by reading all chunks in the directory
        for filename in self.chunk_filenames:
            filepath = self.dir / filename
            with np.load(filepath, allow_pickle=True) as file:
                chunk_keys = {k: True for k in file}
                chunk_keys.pop(self.tensors_keys_key)
            self.keys |= chunk_keys

        self.buffer = {}

    def save(self, **kwargs):
        new_keys = {k: True for k in kwargs}
        existent_keys = set(new_keys.keys()) & set(self.keys.keys())
        assert not existent_keys, f"The following keys already exists in the archive:\n{existent_keys}"
        new_archive_filename = f"{self.chunk_basename}_{len(self.chunk_filenames) + 1}.npz"
        new_archive_filepath = self.dir / new_archive_filename

        # Save information about saved data that are tensors
        tensor_keys = [k for k, v in kwargs.items() if isinstance(v, torch.Tensor)]

        if self.tensors_keys_key in kwargs:
            raise ValueError(f"{self.tensors_keys_key} is the name of a saved variable.")
        kwargs = kwargs | {self.tensors_keys_key: tensor_keys}

        np.savez(new_archive_filepath, **kwargs)
        self.chunk_filenames.append(new_archive_filename)
        self.keys |= new_keys

    def save_to_buffer(self, **kwargs):
        self.buffer |= kwargs

        # Empty buffer when enough time has elapsed.
        curr_time = time.time()
        if curr_time > self.last_buffer_save_time + self.buffer_saving_period:
            self.empty_buffer()

    def empty_buffer(self):
        if self.buffer:
            self.save(**self.buffer)
            self.buffer = {}
        self.last_buffer_save_time = time.time()

    def load(self, filename=''):
        if filename:
            filenames = [filename] if filename in self.chunk_filenames else []
        else:
            filenames = self.chunk_filenames  # Load all files and concatenate them

        content = dict()
        for filename in filenames:
            filepath = self.dir / filename
            with np.load(filepath, allow_pickle=True) as file:
                content_temp = dict(file)
            tensor_keys = content_temp.pop(self.tensors_keys_key)
            for k, v in content_temp.items():
                if v.size == 1 and not isinstance(v.item(), np.ndarray):
                    v = v.item()
                content_temp[k] = v

            # Convert np.ndarray that were tensors into tensors
            for k in tensor_keys:
                content_temp[k] = torch.tensor(content_temp[k])

            # Check for duplicate keys
            duplicate_content = set(content.keys()) & set(content_temp.keys())
            if duplicate_content:
                warnings.warn(f'The following keys are duplicated in {self.dir}:\n{duplicate_content}')
            content |= content_temp
        return content

    def destroy(self):
        if self.dir.exists():
            shutil.rmtree(self.dir)

    def reset(self):
        self.destroy()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.chunk_filenames = []
        self.keys = {}
        self.buffer = {}

    def __len__(self):
        return len(self.chunk_filenames)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.empty_buffer()
        return False


class Lambda:
    def __init__(self, func):
        self.func = func

        # Check if function is a lambda function
        self.func_str = inspect.getsource(func)
        assert 'lambda' in self.func_str, f'{func} is not a lambda function'
        self.func_str = self.func_str[self.func_str.index('lambda'):]

        # Remove trailing comments and white space
        if '#' in self.func_str:
            self.func_str = self.func_str[:self.func_str.index('#')]
        self.func_str = self.func_str.rstrip()

    def __call__(self, *args):
        return self.func(*args)

    def __eq__(self, other):
        if not isinstance(other, Lambda):
            return False
        return self.func_str == other.func_str

    def __repr__(self):
        return f'{self.func_str}'


def str2bool(s: str):
    """
    Convert various strings with boolean meaning to bool
    Args:
        s: string

    Returns:
        bool
    """
    return s.lower() in ("y", "yes", "true", "t", "1")


def set_rng(seed: int):
    """
    Sets the seed of torch, numpy and random
    Args:
        seed: integer value of seed

    Returns:
        None
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def countdown(T):
    """
    Basic timer to countdown for a given amount of seconds
    Args:
        T: starting time (seconds) to count down from
    """
    for t in range(T, 0, -1):
        print(f'{t}')
        time.sleep(1)


def delayed_delete(filepath: str | pathlib.Path, delay=10):
    """
    Deletes a file or directory with a countdown
    Args:
        filepath: path of the file or directory
        delay: delay time (seconds) before deletion

    Returns:
        None
    """
    filepath = pathlib.Path(filepath)
    if filepath.exists():
        print(f'Deleting {filepath.name} located in: {str(filepath.parent)!r}\nin {delay} seconds...')
        countdown(delay)
        shutil.rmtree(filepath)


def find_duplicates(seq: Iterable):
    """
    Finds duplicates and their indices in a python list
    Args:
        seq: list of objects

    Returns:
        dictionary where keys are the duplicated objects and values are indices where they are found
    """
    indices = defaultdict(list)
    for i, item in enumerate(seq):
        indices[item].append(i)
    indices = {key: ind for key, ind in indices.items() if len(ind) > 1}
    return indices


def find_first_element_of_class(my_list: list, target_class: type):
    """
    Returns the first element of a list that is an instance of target_class.
    If no such element is found, returns None.
    """
    for item in my_list:
        if isinstance(item, target_class):
            return item
    return None


def cast_to_optimal_dtype(array: np.ndarray, max=None, min=None):
    """
    Cast the given array to a dtype that has the smallest memory size given its content
    Args:
        array: array to be recasted
        max: maximum value allowed. Defaults to maximum found in array
        min: minimmum value allowed. Defaults to minimum found in array
    Returns:
        array in lowest dtype
    """
    # Do not recast for string arrays
    if np.issubdtype(array.dtype, str):
        return array
    if max is None:
        max = array.max()
    if min is None:
        min = array.min()

    if np.iscomplexobj(array):
        dtypes_candidates = [np.complex64, np.complex128, np.clongdouble]
        type_info = np.finfo
    elif np.issubdtype(array.dtype, np.floating):
        dtypes_candidates = [np.float16, np.float32, np.float64, np.longdouble]
        type_info = np.finfo
    elif min < 0:
        dtypes_candidates = [np.int8, np.int16, np.int32, np.int64]
        type_info = np.iinfo
    else:
        dtypes_candidates = [np.uint8, np.uint16, np.uint32, np.uint64]
        type_info = np.iinfo

    opt_dtype = next(t for t in dtypes_candidates if type_info(t).min <= min and type_info(t).max >= max)
    array_recasted = array.astype(opt_dtype)
    return array_recasted


################################################## MONITORING ##################################################

class Timer:
    """
    Timer class that can resume from previous sessions using checkpoints.
    It can also be used as a context manager to monitor time spent in the context.
    """

    def __init__(self, checkpoint_filepath: str | pathlib.Path = None, resume=True):
        self.elapsed_time_prev = 0.0  # Elapsed time in previous sessions
        self.start_time = 0.0
        self.stop_time = 0.0
        self.is_running = False
        self.resume = resume

        # Save timing statistics in a numpy .npz file
        self.checkpoint_filepath = checkpoint_filepath
        if checkpoint_filepath is not None:
            self.checkpoint_filepath = pathlib.Path(self.checkpoint_filepath).with_suffix('.npz')
            self.checkpoint_filepath.parent.mkdir(parents=True, exist_ok=True)
            self.load_checkpoint()

    @property
    def elapsed_time(self):
        if self.is_running:
            stop_time = time.perf_counter()
            elapsed_time_tot = stop_time - self.start_time + self.elapsed_time_prev
        else:
            elapsed_time_tot = self.elapsed_time_prev
        return elapsed_time_tot

    @property
    def has_checkpoint(self):
        return self.checkpoint_filepath is not None and self.checkpoint_filepath.exists()

    def print_elapsed_time(self):
        print(f'Elapsed time:{self.elapsed_time:.1f} s')

    def load_checkpoint(self):
        if self.resume and self.has_checkpoint:
            with np.load(self.checkpoint_filepath) as npzfile:
                self.elapsed_time_prev = npzfile['elapsed_time'].item()

    def save_checkpoint(self):
        if self.checkpoint_filepath is not None:
            np.savez(self.checkpoint_filepath, elapsed_time=self.elapsed_time)

    def delete_checkpoint(self):
        if self.checkpoint_filepath is not None:
            self.checkpoint_filepath.unlink(missing_ok=True)

    def start(self):
        self.start_time = time.perf_counter()
        self.is_running = True

    def stop(self):
        if self.is_running:
            self.elapsed_time_prev += time.perf_counter() - self.start_time
            self.is_running = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # On exit, stop timer and record the total elapsed time to the checkpoint file
        self.stop()
        self.save_checkpoint()


################################################## PLOTTING ##################################################
class SingleLabelLogFormatter(matplotlib.ticker.LogFormatterSciNotation):
    def format_ticks(self, values):
        if len(values) == 0:
            return []

        # Get current axis view limits and geometric center
        vmin, vmax = self.axis.get_view_interval()
        log_mid = (np.log10(vmin) + np.log10(vmax)) / 2

        # Get active major and minor ticks
        major_locs = self.axis.get_majorticklocs()
        minor_locs = self.axis.get_minorticklocs()

        # Filter both sets to only look at what is currently visible on screen
        visible_majors = major_locs[(major_locs >= vmin) & (major_locs <= vmax)]
        visible_minors = minor_locs[(minor_locs >= vmin) & (minor_locs <= vmax)]

        # If there is at least one visible major tick, prioritize them exclusively.
        if len(visible_majors) > 0:
            candidates = visible_majors
        elif len(visible_minors) > 0:
            candidates = visible_minors
        else:
            return [""] * len(values)

        # Find the single absolute winner within the chosen candidate pool
        closest_tick = candidates[np.argmin(np.abs(np.log10(candidates) - log_mid))]

        labels = []
        for v in values:
            # Only label the tick if it matches the closest tick
            if np.isclose(v, closest_tick) and v > 0:
                labels.append(super(SingleLabelLogFormatter, self).__call__(v))
            else:
                labels.append("")

        return labels


class AdaptivePrecisionEdgeLocator(matplotlib.ticker.Locator):
    def __init__(self, num_ticks=2, tick_max_extent=0.25):
        super().__init__()
        self.num_ticks = max(2, num_ticks)
        self.tick_max_extent = tick_max_extent

    def _to_sig_figs(self, val, sig_figs, direction='round'):
        """Rounds a value to a specific number of significant digits."""
        if val == 0:
            return 0.0

        order = int(np.floor(np.log10(abs(val))))
        factor = 10 ** (order - (sig_figs - 1))

        if direction == 'up':
            return float(np.ceil(val / factor) * factor)
        elif direction == 'down':
            return float(np.floor(val / factor) * factor)
        else:
            return float(np.round(val / factor) * factor)

    def _try_generate_ticks(self, vmin, vmax, max_sf, renderer):
        """Attempts to fit ticks minimizing significant figures independently per tick up to max_sf."""
        formatter = self.axis.get_major_formatter()
        ticklabel_fontsize = self.axis.majorTicks[0].label1.get_size()
        txt_min = self.axis.figure.text(0, 0, formatter(vmin), fontsize=ticklabel_fontsize)
        txt_max = self.axis.figure.text(0, 0, formatter(vmax), fontsize=ticklabel_fontsize)

        bbox_min = txt_min.get_window_extent(renderer)
        bbox_max = txt_max.get_window_extent(renderer)

        txt_min.remove()
        txt_max.remove()

        axis_bbox = self.axis.axes.get_window_extent(renderer)
        axis_data_span = vmax - vmin

        data_offset_min = (bbox_min.width / 2) / axis_bbox.width * axis_data_span
        data_offset_max = (bbox_max.width / 2) / axis_bbox.width * axis_data_span

        min_allowed_data = vmin + data_offset_min
        max_allowed_data = vmax - data_offset_max

        raw_positions = np.linspace(min_allowed_data, max_allowed_data, self.num_ticks)
        final_ticks = []

        success = True
        for i in range(self.num_ticks):
            raw_val = raw_positions[i]
            found_tick = None

            # Find the minimum significant figures independently for this tick
            for sf in range(0, max_sf + 1):
                if i == 0:
                    val = self._to_sig_figs(raw_val, sf, direction='up')
                    if val >= min_allowed_data and (val - vmin) / axis_data_span <= self.tick_max_extent:
                        found_tick = val
                        break
                elif i == self.num_ticks - 1:
                    val = self._to_sig_figs(raw_val, sf, direction='down')
                    if val <= max_allowed_data and (vmax - val) / axis_data_span <= self.tick_max_extent:
                        found_tick = val
                        break
                else:
                    val = self._to_sig_figs(raw_val, sf, direction='round')
                    found_tick = val
                    break

            if found_tick is None:
                success = False
                break
            final_ticks.append(found_tick)

        if not success:
            return None

        # Verify uniqueness and strict increasing order
        if len(np.unique(final_ticks)) < self.num_ticks or not np.all(np.diff(final_ticks) > 0):
            return None

        return final_ticks

    def __call__(self):
        """Iterates through precision ceiling levels until a valid fit is found."""
        vmin, vmax = self.axis.get_view_interval()
        if vmin == vmax:
            return [vmin, vmax]

        renderer = None

        # Loop over precision ceiling levels
        for max_sf in range(1, 4):
            ticks = self._try_generate_ticks(vmin, vmax, max_sf, renderer)
            if ticks is not None:
                return ticks

        # Fallback
        return np.linspace(vmin, vmax, self.num_ticks)


def get_axes_bbox_inch(ax: plt.Axes) -> matplotlib.transforms.Bbox:
    """
    Returns the Bbox object of a matplotlib Axes with measurements in inches.
    Args:
        ax: matplotlib Axes object

    Returns:
        matplotlib Bbox object
    """
    # Get the bounding box of the axes in display units
    bbox = ax.get_window_extent()

    # Invert the figure's DPI scale transform and apply it to the bbox
    bbox_inches = bbox.transformed(ax.figure.dpi_scale_trans.inverted())

    return bbox_inches


def draw_ax_bbox(ax: plt.Axes, type='bbox', color='red', **kwargs):
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if type == 'bbox':
        bbox = ax.get_window_extent(renderer)
    elif type == 'tbbox':
        bbox = ax.get_tightbbox(renderer)
    else:
        raise NotImplementedError
    bbox_fig = bbox.transformed(fig.transFigure.inverted())
    rect = matplotlib.patches.Rectangle((bbox_fig.x0, bbox_fig.y0), bbox_fig.width, bbox_fig.height,
                                        transform=fig.transFigure, fill=False, color=color, **kwargs)

    fig.add_artist(rect)
    plt.show()


def copy_props(src_obj, target_obj, props_name):
    for prop in props_name:
        setattr(target_obj, prop, getattr(src_obj, prop))


def copy_line(src: matplotlib.lines.Line2D, target: matplotlib.lines.Line2D):
    target._label = src._label
    target._linestyle = src._linestyle
    target._linewidth = src._linewidth
    target._color = src._color
    target._gapcolor = src._gapcolor
    target._markersize = src._markersize
    target._markerfacecolor = src._markerfacecolor
    target._markerfacecoloralt = src._markerfacecoloralt
    target._markeredgecolor = src._markeredgecolor
    target._markeredgewidth = src._markeredgewidth
    target._unscaled_dash_pattern = target._unscaled_dash_pattern
    target._dash_pattern = target._dash_pattern
    target._dashcapstyle = target._dashcapstyle
    target._dashjoinstyle = target._dashjoinstyle
    target._solidcapstyle = target._solidcapstyle
    target._solidjoinstyle = target._solidjoinstyle
    target._linestyle = target._linestyle
    target._marker = matplotlib.markers.MarkerStyle(marker=target._marker)
    target._drawstyle = target._drawstyle


def copy_legend(src, target):
    props_name = ['handlelength', '_loc', '_label', '_alignment', '_fontsize', '_ncols']
    copy_props(src, target, props_name=props_name)


def copy_ax(src: matplotlib.pyplot.axes, target: matplotlib.pyplot.axes):
    """Move an Axes object from a figure to a new pyplot managed Figure in
          the specified subplot."""

    # ax = src
    # fig = target.figure
    # subplot_spec='1111'
    #
    # # get a reference to the old figure context so we can release it
    # old_fig = ax.figure
    #
    # # remove the Axes from it's original Figure context
    # ax.remove()
    #
    # # set the pointer from the Axes to the new figure
    # ax.figure = fig
    #
    # # add the Axes to the registry of axes for the figure
    # fig.axes.append(ax)
    # # twice, I don't know why...
    # fig.add_axes(ax)
    #
    # # then to actually show the Axes in the new figure we have to make
    # # a subplot with the positions etc for the Axes to go, so make a
    # # subplot which will have a dummy Axes
    # dummy_ax = fig.add_subplot(subplot_spec)
    #
    # # then copy the relevant data from the dummy to the ax
    # ax.set_position(dummy_ax.get_position())
    #
    # # then remove the dummy
    # dummy_ax.remove()
    #
    # # close the figure the original axis was bound to
    # plt.close(old_fig)

    # Transfer axes properties
    properties = ['xlabel', 'xticks', 'xticklabels', 'xlim', 'ylabel', 'yticks', 'yticklabels', 'ylim', 'aspect',
                  'visible', 'frame_on']
    for prop in properties:
        getter_attr = f"get_{prop}"
        setter_attr = f"set_{prop}"
        getattr(target, setter_attr)(getattr(src, getter_attr)())
    target.axis(src.axison)

    # target.xaxis.update_from(src.xaxis)
    for h in src._children:
        if isinstance(h, matplotlib.lines.Line2D):
            h_target = target.plot(h.get_xdata(), h.get_ydata())[0]
            copy_line(src=h, target=h_target)
        elif isinstance(h, matplotlib.collections.LineCollection):
            line_segments = matplotlib.collections.LineCollection(h.get_segments(), linewidths=h.get_linewidth(),
                                                                  colors=h.get_colors(), linestyle=h.get_linestyle())
            target.add_collection(line_segments)
        elif isinstance(h, matplotlib.collections.PolyCollection):
            paths = h.get_paths()
            verts = [p.vertices for p in paths]
            poly_colls = matplotlib.collections.PolyCollection(verts, linewidths=h.get_linewidths(),
                                                               facecolors=h.get_facecolors(),
                                                               linestyle=h.get_linestyle(), alpha=h.get_alpha(),
                                                               transform=target.transData)

            # 4. Add the new collection to the destination axes
            target.add_collection(poly_colls)
        elif isinstance(h, matplotlib.image.AxesImage):
            img_kwargs = {
                "X": h.get_array(),
                "extent": h.get_extent(),
                "cmap": copy.deepcopy(h.get_cmap()),
                "norm": copy.deepcopy(h.norm),
                "origin": h.origin,
                "interpolation": h.get_interpolation(),
                "alpha": h.get_alpha(),
                "visible": h.get_visible(),
                "zorder": h.get_zorder(),
                "filternorm": h.get_filternorm(),
                "filterrad": h.get_filterrad(),
                "resample": h.get_resample(),
                "url": h.get_url(),
            }
            target.imshow(**img_kwargs)
        elif isinstance(h, matplotlib.text.Text):
            extra_kwargs = {}
            if not isinstance(h.get_transform(), matplotlib.transforms.CompositeGenericTransform):
                extra_kwargs['transform'] = target.transAxes
            target.text(*h.get_position(), h.get_text(), color=h.get_color(),
                        horizontalalignment=h.get_horizontalalignment(), verticalalignment=h.get_verticalalignment(),
                        size=h.get_size(), weight=h.get_weight(), **extra_kwargs)
        else:
            raise ValueError(f"copy of {h} children is not supported")

    if src.legend_:
        target_leg = target.legend()
        target_leg.update_from(src.legend())
        copy_legend(src=src.legend_, target=target_leg)


def plot_mu_sigma(mu: torch.Tensor | np.ndarray, sigma: torch.Tensor | np.ndarray, x: torch.Tensor | np.ndarray = None,
                  **kwargs):
    import matplotlib.pyplot as plt
    if isinstance(mu, np.ndarray):
        mu = torch.from_numpy(mu)
    if isinstance(sigma, np.ndarray):
        sigma = torch.from_numpy(sigma)
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if mu.ndim == 1:
        mu = mu.unsqueeze(-1)
    if sigma.ndim == 1:
        sigma = sigma.unsqueeze(-1)
    if x is None:
        x = torch.arange(mu.shape[0]).unsqueeze(-1).expand((-1, mu.shape[1]))
    if x.ndim == 1:
        x = x.unsqueeze(-1)

    y_low = mu - sigma
    y_up = mu + sigma
    x = x.expand(y_low.shape)
    mean_lines = []
    for i in range(mu.shape[1]):
        mean_line_h = plt.plot(x[:, i], mu[:, i], **kwargs)[0]
        plt.fill_between(x[:, i], y_low[:, i], y_up[:, i], facecolor=mean_line_h.get_color(), alpha=0.5)
        mean_lines.append(mean_line_h)
    return mean_lines if len(mean_lines) > 1 else mean_lines[0]


def trim_xlim_by_height(ax: plt.Axes, y_min_show_frac=0.01):
    """
    Redefines the x-axis limits of an axes object so that the displayed region only includes points where the
    plotted function height is at or above y_min_show_frac relative to the y-axis extent.
    Args:
        ax: axis whose x limits will be trimmed
        y_min_show_frac: minimum y-value (as a fraction of the y-axis extent) for the plotted function to be shown.
    """

    ymin, ymax = ax.get_ylim()
    threshold = ymin + y_min_show_frac * (ymax - ymin)

    min_x = np.inf
    max_x = -np.inf

    for line in ax.get_lines():
        x = np.array(line.get_xdata())
        y = np.array(line.get_ydata())
        if len(x) == 0:
            continue

        mask = y >= threshold
        if np.any(mask):
            x_above = x[mask]
            min_x = min(min_x, np.min(x_above))
            max_x = max(max_x, np.max(x_above))

    if min_x < max_x:
        ax.set_xlim(min_x, max_x)


if __name__ == '__main__':
    torch.manual_seed(3)

    # # Test gmx_anonymize on .mdp file
    # fp = pathlib.Path.home() / 'Downloads' / 'mdout.mdp'
    # header_attrs = parse_gmx_header(fp)
    # gmx_anonymize(fp, fp.with_stem(fp.stem + '_ano'))
    # pass

    # # Test gmx_anonymize on .top file
    # from data.datasets import load_dataset
    #
    # dataset = load_dataset()
    # temp_dir = tempfile.gettempdir()
    # output_filepath = pathlib.Path(temp_dir) / dataset.top_filepath.with_stem(
    #     dataset.top_filepath.stem + '_anonymized').name
    # gmx_anonymize_top(dataset.top_filepath, output_filepath)
    # gmx_anonymize_top(output_filepath, output_filepath)
    # gmx_regen_top(output_filepath, output_filepath, terminal_states=['1', '1'])
    # gmx_test_top(output_filepath)
    # pass

    # # Test Timer and Timer resuming
    # Timer_filepath = tempfile.mktemp()
    #
    # timer = Timer(checkpoint_filepath=Timer_filepath)
    # for i in range(5):
    #     with timer:
    #         time.sleep(5)
    #     timer.print_elapsed_time()
    #     time.sleep(2)

    # # Test round_to_sig
    # x = np.random.randn(100) * 1000
    # y = round_to_sig(x, 2)
    # pass

    # # Check if sum of variance changes with rotation. It shouldn't
    # x0 = torch.randn((1000, 35, 3)).double()
    # T = RigidBodyMotion.random(size=(1000, 1), dtype=torch.double)
    # x0_var = x0.var(dim=-2).mean(-1)
    # x0_rot_var = T.transform(x0).var(dim=-2).mean(-1)

    # # Test remove_rot
    # # Check if fixing axes sign is necessary in defining a rotation-invariant of a point cloud.
    # # (Fixing axes sign is necessary)
    # x0 = torch.randn((1, 35, 3)).double()
    # x0 -= x0.mean(dim=-2, keepdim=True)
    # T = RigidBodyMotion.random(size=(1000, 1), dtype=torch.double)
    # x0_rot = T.transform(x0)
    # x0_svd = remove_rot(x0_rot, method='svd', fix_axes_sign=True)
    # x0_svd_eq_x0 = x0_svd.isclose(x0_svd[0].unsqueeze(0))
    #
    # x0_svd2 = remove_rot(x0_svd, method='svd', fix_axes_sign=True)
    # x0_svd2_eq_x0_svd_all = x0_svd2.isclose(x0_svd).all()
    # x0_PCA = remove_rot(x0_rot, method='PCA', fix_axes_sign=True)
    # x0_PCA2 = remove_rot(x0_PCA, method='PCA', fix_axes_sign=True)
    # x0_PCA2_eq_x0_PCA_all = x0_PCA2.isclose(x0_PCA).all()
    # x0_PCA_eq_x0_svd_all = x0_PCA.isclose(x0_svd).all()
    #
    # x0_eq_all = x0_svd_eq_x0.all()
    # pass

    # mu = torch.randn(1000, 3)
    # sig = 10 * torch.rand_like(mu)
    # plot_mu_sigma(mu=mu, sigma=sig)

    # # Test mvn conditional stats
    # B, d = 1, 10
    # mu = torch.randn(B, d).squeeze()
    # R = torch.randn(B, d, d).squeeze()
    # C = R @ R.transpose(-1, -2)
    # cond_mask = torch.randn(d) > 0.2
    # cond_values = torch.randn(B, d).squeeze()[..., cond_mask]
    # mvn_cond_params(mu=mu, C=C, cond_mask=cond_mask, cond_values=cond_values)

    # # Test Archiver
    # arch = Archiver(dir=tempfile.mkdtemp())
    # for i in range(10):
    #     data_dict = {f"{i}_a": torch.randn(10, 10), f"{i}_b": 'str', f"{i}_c": np.random.randn(10)}
    #     arch.save(**data_dict)
    # arch_content = arch.load()
    # arch.destroy()

    # # Test index_softmax
    # N, n, d = 100, 5, 10
    # x = torch.randn(N, d)
    # ind = torch.randint(0, n, (N,))
    # #ind = torch.cat([torch.ones((N // 2,), dtype=torch.int), 2 * torch.ones((N // 2,), dtype=torch.int)])
    # ind = torch.cat([torch.ones((1,), dtype=torch.int), 2 * torch.ones((N-1,), dtype=torch.int)])
    # x2 = index_softmax(dim=0, src=x, index=ind)
    # pass

    # # Test rotation of dihedral quartet
    # positions = torch.randn(4, 4, 3)
    # phi = torch.distributions.Uniform(-torch.pi / 2, torch.pi / 2).sample((positions.shape[0],))
    # rotate_dih_quartet(positions=positions, phi=phi, plot=True)

    # # Test edge cases of MarcumQ
    # a = 10 * torch.rand(10000)
    # b = torch.rand_like(a)
    # b_is_inf = torch.rand_like(b) < 0.01
    # b[b_is_inf] = torch.inf
    # b_is_zero = torch.rand_like(b) < 0.01
    # b[b_is_zero] = 0.0
    # y1 = MarcumQ(a, b, 3 / 2, complement=True, log_output=False)
    # y2 = MarcumQ(a, b, 3 / 2, complement=True, log_output=True)
    # y3 = MarcumQ(a, b, 3 / 2, complement=False, log_output=True)
    # pass

    # # Test NonCentralChicdf on edge cases
    # fp = pathlib.Path.home() / 'Downloads' / 'NonCentralChicdf_debug_kwargs.pt'
    # kwargs = torch.load(fp, torch.device('cpu'), weights_only=False)
    # with torch.autograd.detect_anomaly():
    #     logcdf = NonCentralChicdf(**kwargs)
    #     logcdf.sum().backward()
    # pass

    # # Test autograd on NonCentralChicdf with edge cases
    # Lambda = 10 * torch.rand(10000)
    # x_lower = 1e-5 * torch.rand_like(Lambda)
    # x_upper = x_lower + 10 * torch.rand_like(Lambda)
    # x_low_is_zero = torch.rand_like(Lambda) < 0.01
    # x_lower[x_low_is_zero] = 0.0
    # x_upp_is_inf = torch.rand_like(Lambda) < 0.01
    # x_upper[x_upp_is_inf] = torch.inf
    # Lambda.requires_grad = True
    # with torch.autograd.detect_anomaly():
    #     log_cdf = NonCentralChicdf(Lambda, x_lower=x_lower, x_upper=x_upper, k=3, log_output=True, compile=True)
    #     log_cdf.sum().backward()
    # pass

    # Test logerfc
    # x1 = torch.linspace(-100,100,10000)
    # y = logerfc(x1)
    # plt.figure()
    # plt.plot(x1,y)
    # plt.plot(x1,torch.log(torch.erfc(x1)))
    # logerfcsum(x, y)

    # # Test the Non-central chi^2 cdf against scipy ncx2
    # from scipy.stats import ncx2
    #
    # k = 3
    # compile = True
    # # x = torch.linspace(1, 1000, 1000).reshape(-1, 1).double()
    # # x = torch.logspace(-6, 6, 1000).reshape(-1, 1).double()
    # # x_lower = torch.tensor(10.0)
    # # Case 1
    # x_lower = torch.linspace(1, 10, 10)
    # x_upper = x_lower + 10 * torch.tensor(1.0)
    # # Case 2
    # # x_lower = torch.zeros(20)
    # # x_upper = x_lower + torch.logspace(1, 4, x_lower.shape[0])
    # Lambda_arr = torch.logspace(-3, np.log10(x_upper[-1] * 2), 1000).reshape(-1, 1).double()
    # dist = 'chi'
    # # dist = 'chi-squared'
    #
    # # Compare with scipy function
    # if dist == 'chi-squared':
    #     logcdf_cust = NonCentralChi2cdf(Lambda=Lambda_arr, k=k, x_lower=x_lower, x_upper=x_upper,
    #                                     log_output=True, compile=compile).numpy()
    #     log_cdf_lower_scipy = ncx2.logcdf(x_lower, df=k, nc=Lambda_arr)
    #     log_cdf_upper_scipy = ncx2.logcdf(x_upper, df=k, nc=Lambda_arr)
    #     logcdf_scipy = scipy.special.logsumexp(np.stack([log_cdf_upper_scipy, log_cdf_lower_scipy]),
    #                                            b=np.array([1, -1])[:, None, None], axis=0)
    # else:
    #     logcdf_cust = NonCentralChicdf(Lambda=Lambda_arr, k=k, x_lower=x_lower, x_upper=x_upper,
    #                                    log_output=True, compile=compile).numpy()
    #     log_cdf_lower_scipy = ncx2.logcdf(x_lower ** 2, df=k, nc=Lambda_arr.square())
    #     log_cdf_upper_scipy = ncx2.logcdf(x_upper ** 2, df=k, nc=Lambda_arr.square())
    #     logcdf_scipy = scipy.special.logsumexp(np.stack([log_cdf_upper_scipy, log_cdf_lower_scipy]),
    #                                            b=np.array([1, -1])[:, None, None], axis=0)
    #
    # for i in range(x_upper.numel()):
    #     plt.figure()
    #     lines_custom_h = plt.plot(Lambda_arr.squeeze(), np.exp(logcdf_cust[:, i]),
    #                               label=f'Custom non-central {dist} cdf', color='k')
    #     lines_scipy_h = plt.plot(Lambda_arr.squeeze(), np.exp(logcdf_scipy[:, i]),
    #                              label=f'Scipy non-central {dist} cdf', color='r')
    #     plt.xlabel('Lambda')
    #     plt.yscale('log')
    #     plt.legend()

    # # Test GFT
    # n, d = 100, 3
    # f = torch.randn(n, d)
    # A = torch.tril(torch.rand(n, n) > 0.8, diagonal=-1)
    # A = A + A.T
    # GFT(f=f, A=A)

    # # Test gaussian_prod
    # n, d = 10, 3
    # mu = torch.randn((n, d))
    # S = torch.randn((n, d, d)).tril()
    # cov = S @ S.transpose(-1, -2)
    # gaussian_prod(mu=mu, cov=cov)

    # Test calculate_func_mu_sigma
    # B, d = 5, 10
    # def f(x: torch.Tensor):
    #     return torch.stack([torch.exp(x[..., 2]), torch.sinh(x[..., 8]), torch.cos(x[..., 0])], dim=-1)
    #
    #
    # S = torch.randn((2, B, d, d)).tril()
    # mu = torch.randn(2, B, d)
    # Sigma = S @ S.transpose(-1, -2)
    # calculate_func_mu_sigma(f=f, mu=mu, cov=Sigma)

    # # Test batched rmsd_align
    # x = torch.randn((10, 20, 10, 3))
    # y = rmsd_align(x, x[0,0])
    # #y = rmsd_align(x, x[:1, :1])

    # # Test rmsd alignment
    # x = torch.randn(64, 100, 3)
    # x_ref = torch.randn(64, 100, 3)
    # R, t = rmsd_align_transform(positions=x, positions_ref=x_ref)
    # x_aligned = rmsd_align(x, x_ref)
    # x_aligned2 = x @ R + t
    # rmsd = (x_aligned - x_ref).square().sum(-1).mean(-1).sqrt()
    # pass

    # # Test lr scheduler
    # x = torch.randn((100,))
    # optimizer = torch.optim.Adam([x])
    # params = {'epochs': [0, 1000, 2000, 10000], 'lrs': [1e-4, 1e-3, 1e-3, 1e-6], 'funcs': ['poly1', 'poly1', 'exp']}
    # lr_scheduler = PiecewiseLR(optimizer, epochs=params['epochs'], lrs=params['lrs'], funcs=params['funcs'],
    #                            verbose=True)
    # lr_scheduler.plot_lr(epoch_max=10000, log=True)
    # pass

    # # Test dihedral angle and bond angle calculation agains MDAnalysis
    # from MDAnalysis.lib.distances import calc_dihedrals, calc_angles
    #
    # torch.manual_seed(10)
    # positions = 10 * torch.randn((1000, 4, 3))
    # phi_pos = [positions[:, i].numpy() for i in range(4)]
    # phi = calc_dihedrals(*phi_pos) / np.pi * 180
    # phi2 = calc_dihedral_angle(positions, deg=True)
    # phi_abs_diff = torch.abs(torch.tensor(phi) - phi2)
    # phi_abs_diff_max = phi_abs_diff.max()
    #
    # # Test bond angle calculation
    # angle_pos = [positions[:, i].numpy() for i in range(3)]
    # bond_angles = calc_angles(*angle_pos) / np.pi * 180
    # bond_angles2 = calc_bond_angle(positions[:, :3]).numpy()
    # bond_angles_diff = np.abs(bond_angles - bond_angles2) / np.abs(bond_angles)
