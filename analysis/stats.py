import copy
import os
import tempfile
import warnings
from typing import Self
import itertools

import numpy as np
import scipy
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
from scipy.stats._multivariate import multivariate_normal_frozen as mvn_type
from scipy.spatial.distance import jensenshannon

from utils import plot_mu_sigma, cast_to_optimal_dtype


def safe_cholesky(M):
    try:
        M_chol = np.linalg.cholesky(M)
    except np.linalg.LinAlgError as e:
        if 'Matrix is not positive definite' in str(e):
            M_chol_eig, M_chol_O = np.linalg.eigh(M)
            print(f"M_eig={M_chol_eig}")
            M_chol = None
        else:
            raise
    return M_chol


def PCA_proj_matrix(C, var_explained: float = 0.99, n: int = None):
    """
    Defines a projection matrix based on a subset of PCA vectors
    :param C: covariance matrix. shape=(d,d)
    :param var_explained: variance explained (%) by the PCA components that defines the projection. The minimum number of components is chose.
    :param n: Number of the most important (highest variance) PCA components used to define the projection
    :return: projection matrix of size (d_new,d) where d_new <= d
    """
    # Find the set of principal components whose explained variance is above the threshold
    cov_lambda, cov_O = np.linalg.eigh(C)
    if np.any(cov_lambda < 0):
        neg_lambda = cov_lambda[cov_lambda < 0]
        warnings.warn(f"Found negative eigenvalues in input covariance matrix:\n"
                      f"eig_min={neg_lambda.min()},n_neg_lambda={neg_lambda.shape[0]},neg_lambda_mean={neg_lambda.mean()}")

    # Find the set of principal components that defines the low-dimensional PCA space
    if n is not None:
        n_PCA = n
    else:
        var_total = cov_lambda.sum()
        var_explained_cum = np.cumsum(cov_lambda[::-1]) / var_total
        n_PCA = np.searchsorted(var_explained_cum, var_explained, 'left') + 1
    proj_matrix = cov_O[:, -n_PCA:]  # shape=(new_dim_size, old_dim_size)
    return proj_matrix


def KL_mvn(P_mvn: mvn_type, Q_mvn: mvn_type):
    """
    Calculates the KL divergence KL(P||Q) of P from Q where P,Q are multivariate normal distributions
    Args:
        P_mvn: reference multivariate normal distribution
        Q_mvn: target multivariate normal distribution

    Returns:
        float: value of the KL divergence
    """

    d = P_mvn.dim
    mu0, mu1 = P_mvn.mean, Q_mvn.mean

    # Calculate the Cholesky decomposition of each covariance matrix to facilitate KL divergence calculation
    cov0_chol = safe_cholesky(P_mvn.cov)
    cov1_chol = safe_cholesky(Q_mvn.cov)
    if cov0_chol is None or cov1_chol is None:
        return np.nan

    M = scipy.linalg.solve_triangular(cov1_chol, cov0_chol, lower=True)  # L1*M = L0
    y = scipy.linalg.solve_triangular(cov1_chol, mu1 - mu0, lower=True)  # L1*y = mu1 - mu0

    M_norm_2 = np.dot(M.flatten().T, M.flatten())
    y_norm_2 = np.dot(y.T, y)
    cov0_det_log = 2 * np.sum(np.log(cov0_chol.diagonal()))
    cov1_det_log = 2 * np.sum(np.log(cov1_chol.diagonal()))
    KL = 0.5 * (M_norm_2 - d + y_norm_2 + cov1_det_log - cov0_det_log)
    return KL


def JS_div(P: torch.Tensor, Q: torch.Tensor):
    """
    Computes the Jensen-Shannon divergence between two probability distributions.
    Args:
        P: First probability distribution array
        Q: Second probability distribution array

    Returns:
        float
    """
    P = P / P.sum(dim=-1, keepdim=True)
    Q = Q / Q.sum(dim=-1, keepdim=True)
    M = (P + Q) / 2
    M_log = M.log()
    KL_PM = torch.where(P > 0, P * (P.log() - M_log), torch.zeros_like(M)).sum(dim=-1)  # D_KL(P || M)
    KL_QM = torch.where(Q > 0, Q * (Q.log() - M_log), torch.zeros_like(M)).sum(dim=-1)  # D_KL(Q || M)
    # KL_PM = torch.nn.functional.kl_div(M_log, target=P, log_target=False)  # D_KL(P || M)
    # KL_QM = torch.nn.functional.kl_div(M_log, target=Q, log_target=False)  # D_KL(Q || M)
    JSD = (KL_PM + KL_QM) / 2
    return JSD


def J_dist_mvn(P: mvn_type | np.ndarray, Q: mvn_type | np.ndarray):
    """
    Calculates the Jeffrey's distance between two samples of multidimensional variables by approximating them as multivariate normal distributions
    Jeffrey's distance is defined as the square root of Jeffrey's divergence.
    Args:
        P: Samples of one distribution or mvn distribution
        Q: Samples of another distribution or mvn distribution

    Returns:
        float: np.sqrt(JD/2) where JD is JD(P,Q) = KL(P||Q) + KL(Q||P)
    """

    if isinstance(P, np.ndarray):
        if np.isnan(P).any():
            return np.nan
        P_mean = np.mean(P, axis=0)
        P_cov = np.cov(P.T)
        P_mvn = multivariate_normal(mean=P_mean, cov=P_cov, allow_singular=True)
    else:
        P_mvn = P

    if isinstance(Q, np.ndarray):
        if np.isnan(Q).any():
            return np.nan
        Q_mean = np.mean(Q, axis=0)
        Q_cov = np.cov(Q.T)
        Q_mvn = multivariate_normal(mean=Q_mean, cov=Q_cov, allow_singular=True)
    else:
        Q_mvn = Q

    # Calculate Jeffrey's divergence
    KL_PQ = KL_mvn(P_mvn, Q_mvn)
    KL_QP = KL_mvn(Q_mvn, P_mvn)
    JD = KL_PQ + KL_QP

    # Normalize Jeffrey's divergence similar to the way the Jensen-Shannon distance is computed in scipy
    J_dist = np.sqrt(JD / 2)
    return J_dist


def J_dist_bin(P: torch.Tensor | np.ndarray, Q: torch.Tensor | np.ndarray, epsilon=1e-8):
    """
    Calculates the Jeffrey's distance between two distributions represented as histograms
    Jeffrey's distance is defined as the square root of Jeffrey's divergence.
    Args:
        P: Array of probability density of first distribution
        Q: Array of probability density of second distribution
        epsilon:minimum density used to avoid log(0) cases
    Returns:
        float: np.sqrt(JD/2) where JD is JD(P,Q) = KL(P||Q) + KL(Q||P)
    """

    if isinstance(P, np.ndarray):
        P = torch.from_numpy(P)
    if isinstance(Q, np.ndarray):
        Q = torch.from_numpy(Q)
    jeffreys_div = scipy.special.rel_entr(P, Q + epsilon) + scipy.special.rel_entr(Q, P + epsilon)
    jeffreys_dist = np.sqrt(jeffreys_div.clamp(min=0.0).sum() / 2)
    return jeffreys_dist


def extend_bins(bin_edges: torch.Tensor | list[torch.Tensor], bin_counts: torch.Tensor | np.ndarray,
                new_lims: torch.Tensor | np.ndarray):
    """
    Extends bins (edges and counts) to a new given range.
    Args:
        bin_edges: 1D array (or list of n 1D arrays for n-dimensional bins) representing the bin edges
        bin_counts: 1D array (or n-D array) representing the counts in each bin.
        new_lims: 1x2 or nx2 array determining the new lower([:,0]) and upper([:,1]) limits of the extended range.

    Returns:
        bin_edges_extended: extended bin edges in the same format as bin_edges
        bin_counts_extended extended bin counts in the same format as bin_counts
    """
    input_is_ndarray = isinstance(bin_counts, np.ndarray)
    if input_is_ndarray:
        if isinstance(bin_edges, list):
            bin_edges = [torch.as_tensor(e) for e in bin_edges]
        else:
            bin_edges = torch.as_tensor(bin_edges)
    bin_counts = torch.as_tensor(bin_counts)
    new_lims = torch.as_tensor(new_lims)
    dev = bin_counts.device

    # Check if the bin counts is 1D or multi-D.
    multi_d = isinstance(bin_edges, list)
    if multi_d:
        d = len(bin_edges)
        assert bin_counts.ndim >= d, f"Bincounts must have at least {d} dim. Received array with {bin_counts.ndim} dim."
        # Check if new_lims is expandable
        new_lims = torch.ones((d, 2), device=dev) * new_lims.to(dev)

        # Determine the bin width in each dimension
        bin_widths = torch.cat([e.diff().round(decimals=5).unique() for e in bin_edges])

        # Determine the final bin edges for each dimension
        n_bins_before = [int(torch.round((e[0] - new_lims[i, 0]) / bin_widths[i])) for i, e in enumerate(bin_edges)]
        n_bins_after = [int(torch.round(-(e[-1] - new_lims[i, 1]) / bin_widths[i])) for i, e in enumerate(bin_edges)]
        for i, edges in enumerate(bin_edges):
            assert n_bins_before[i] >= 0, f"Lower bin limit in dim={i} ({new_lims[i, 0]}) must be <={edges[0].item()}"
            assert n_bins_after[i] >= 0, f"Upper bin limit ({new_lims[i, 1]}) must be >={edges[1].item()}"
        n_bins_tot = [len(e) - 1 + n_b + n_a for e, n_b, n_a in zip(bin_edges, n_bins_before, n_bins_after)]
        bin_edges_new = [torch.linspace(new_lims[i, 0], new_lims[i, 1], n_bins_tot[i] + 1) for i in range(d)]
        bin_edges_new = [e_n.to(dtype=e.dtype, device=e.device) for e, e_n in zip(bin_edges, bin_edges_new)]

        # Determine the pad size needed to expand the current bin_counts array to the new bin range in each dim.
        pad_size = torch.stack([torch.tensor(n_bins_before), torch.tensor(n_bins_after)], dim=-1)
        pad_size = torch.cat([pad_size, torch.zeros((bin_counts.ndim - d, 2), dtype=pad_size.dtype)])
        pad_size = pad_size.int().flip(dims=[0]).flatten().tolist()
        bin_counts_new = torch.nn.functional.pad(bin_counts, pad_size, value=0)
    else:
        binwidth = torch.diff(bin_edges[:2])
        new_lims = new_lims.reshape((2,))
        n_bins_before = int(torch.round((bin_edges[0] - new_lims[0]) / binwidth))
        n_bins_after = int(torch.round(-(bin_edges[-1] - new_lims[1]) / binwidth))
        if n_bins_before > 0 or n_bins_after > 0:
            assert n_bins_before >= 0, f"Lower bin limit ({new_lims[0]}) must be <={bin_edges[0].item()}"
            assert n_bins_after >= 0, f"Upper bin limit ({new_lims[1]}) must be >={bin_edges[-1].item()}"
            new_binedges_before = torch.arange(n_bins_before, device=dev) * binwidth + new_lims[0]
            new_binedges_after = torch.arange(1, n_bins_after + 1, device=dev) * binwidth + bin_edges[-1]
            bin_edges_new = torch.cat([new_binedges_before, bin_edges, new_binedges_after])
            pad_size = torch.tensor([n_bins_before, n_bins_after]).reshape(1, 2).to(dev)
            pad_size = torch.cat([pad_size, torch.zeros((bin_counts.ndim - 1, 2), dtype=pad_size.dtype, device=dev)])
            pad_size = pad_size.int().flip(dims=[0]).flatten().tolist()
            bin_counts_new = torch.nn.functional.pad(bin_counts, pad_size, value=0)
        else:
            bin_edges_new, bin_counts_new = bin_edges, bin_counts

    if input_is_ndarray:
        if isinstance(bin_edges_new, list):
            bin_edges_new = [e.numpy() for e in bin_edges_new]
        else:
            bin_edges_new = bin_edges_new.numpy()
        bin_counts_new = bin_counts_new.numpy()

    return bin_edges_new, bin_counts_new


def combine_bins(bin_counts: list[np.ndarray | torch.Tensor], bin_edges: list[np.ndarray | torch.Tensor]):
    """
    Combines a set of 1D or multi-D bins (edges and counts) into a single histogram
    Args:
        bin_counts: list of bin counts
        bin_edges:  list of bin edges

    Returns:
        bin_counts: 1D or nD array representing the combined counts
        bin_edges: 1D array or list of 1D arrays representing the new bin edges of the combined bins
    """
    input_is_numpy = isinstance(bin_counts[0], np.ndarray)

    if input_is_numpy:
        bin_counts = [torch.tensor(x) for x in bin_counts]
        bin_edges = [torch.tensor(x) for x in bin_edges]
    dtype = bin_counts[0].dtype
    dev = bin_counts[0].device

    # Check if the bincounts are 1D or multidimensional
    n_entries, n_d = len(bin_counts), bin_counts[0].ndim
    multi_d = isinstance(bin_edges[0], list)

    if multi_d:
        assert all([isinstance(e, list) for e in bin_edges]), 'binedges must be list of list'

        # Check that the bin widths are all the same in each dimension
        bin_widths = torch.zeros(n_d, device=dev)
        for i in range(n_d):
            bin_widths_temp = torch.stack([e_all[i].diff().round(decimals=5).unique() for e_all in bin_edges])
            if not torch.all(bin_widths_temp == bin_widths_temp[0]):
                raise ValueError(f"Can't add bin edges with different bin widths: bin widths={bin_widths_temp}")
            bin_widths[i] = bin_widths_temp[0]

        # Determine the final bin edges for each dimension
        bin_edges_start = torch.stack([torch.stack([x[i][0] for x in bin_edges]).min() for i in range(n_d)])
        bin_edges_end = torch.stack([torch.stack([x[i][-1] for x in bin_edges]).max() for i in range(n_d)])
        bin_edges_lims = torch.stack([bin_edges_start, bin_edges_end], dim=-1)
        n_bins = [int(torch.round((e - s) / w)) for s, e, w in zip(bin_edges_start, bin_edges_end, bin_widths)]

        # Extend each given set of bins and accumulate onto the combined bin counts
        bin_counts_combined = torch.zeros(n_bins, dtype=dtype, device=dev)
        for counts, edges in zip(bin_counts, bin_edges):
            bin_edges_combined, counts_ext = extend_bins(bin_edges=edges, bin_counts=counts, new_lims=bin_edges_lims)
            bin_counts_combined += counts_ext
    else:
        # Check that the binwidth is uniform
        bin_width = torch.cat([torch.unique(torch.diff(x).round(decimals=5)) for x in bin_edges]).unique()
        if bin_width.nelement() > 1:
            raise ValueError(f"Can't add bin edges with different bin widths: bin widths={bin_width}")

        # bin_edges_start = torch.tensor([x[0] for x in bin_edges]).min()
        # bin_edges_end = torch.tensor([x[-1] for x in bin_edges]).max()
        new_range = torch.stack(torch.cat(bin_edges).aminmax())
        n_bins = int(torch.round((new_range[-1] - new_range[0]) / bin_width))
        bin_edges_combined = torch.linspace(new_range[0], new_range[-1], n_bins + 1, dtype=bin_width.dtype, device=dev)

        # Extend bin counts and accumulate them
        bin_counts_extra_dim = bin_counts[0].shape[1:]
        bin_counts_combined = torch.zeros((n_bins, *bin_counts_extra_dim), dtype=dtype, device=dev)
        for edges, counts in zip(bin_edges, bin_counts):
            _, counts_ext = extend_bins(bin_edges=edges, bin_counts=counts, new_lims=new_range)
            bin_counts_combined += counts_ext

    if input_is_numpy:
        bin_edges_combined = [e.numpy() for e in bin_edges_combined]
        bin_counts_combined = bin_counts_combined.numpy()

    return bin_counts_combined, bin_edges_combined


def downsample_bins(bin_edges: list[np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor,
                    bin_counts: np.ndarray | torch.Tensor,
                    factor: int, dim: np.ndarray | torch.Tensor | list = None):
    """
    Downsamples bins (edges and counts) by a given integer factor
    Args:
        bin_edges: bin edges of the input bins. When many dimensions are downsampled, bin edges must be a list.
        bin_counts: bin counts of the input bins.
        factor: downsampling factor
        dim: dimension of input bin_counts that will be downsampled

    Returns:
        bin_edges_downsampled: 1D array (or list of 1D arrays for multi-D case) of the downsampled edges' position
        bin_counts_downsampled: 1D array (or list of 1D arrays for multi-D case) of the downsampled edges' position
    """
    ndim = bin_counts.ndim
    if dim is None:
        dim = torch.arange(ndim)
    dim = torch.as_tensor(dim).reshape(-1, )
    assert len(bin_edges) >= dim.max() + 1, f"The length of the input bin edges list must be at least {dim.max() + 1}."

    bin_edges_is_list = isinstance(bin_edges, list)
    if bin_edges_is_list:
        # Copy bin edges list to avoid performing inplace changes
        bin_edges = copy.deepcopy(bin_edges)
    else:
        assert dim.numel() == 1, "A list of bin edges is expected when multiple dimensions are downsampled."
        bin_edges = [bin_edges if i == dim else torch.nan for i in range(ndim)]
    is_input_tensor = isinstance(bin_counts, torch.Tensor)
    bin_edges = [torch.as_tensor(e) for e in bin_edges]
    bin_counts = torch.as_tensor(bin_counts)

    # Downsample the counts by convolving a kernel of 1's onto the bin_counts array using the cpu.
    kernel_shape = [factor if i in dim else 1 for i in range(ndim)]
    kernel = torch.ones(kernel_shape)
    bin_counts_summed = torch.from_numpy(scipy.signal.convolve(bin_counts.cpu(), kernel, mode='valid'))
    bin_counts_summed = bin_counts_summed.to(bin_counts.device, bin_counts.dtype)
    slices = [slice(0, None, factor if i in dim else 1) for i in range(ndim)]
    bin_counts_downsampled = bin_counts_summed[*slices]

    # Downsample the bin edges
    for i in dim:
        bin_edges[i] = bin_edges[i][::factor]
        # Add missing edge if the number of edges is smaller than the number of bins + 1
        if len(bin_edges[i]) < bin_counts_downsampled.shape[i] + 1:
            binwidth = torch.diff(bin_edges[i][:2])
            last_bin_edge = bin_edges[i][-1] + factor * binwidth
            bin_edges[i] = torch.pad(bin_edges[i], (0, 1), mode='constant', constant_values=(0, last_bin_edge))

    # for i, counts in enumerate(bin_counts):
    #     downsampled_indices = [
    #         np.arange(factor - 1, (s // factor * factor) + 1, factor).reshape(*k * [1], -1, *(ndim - k - 1) * [1])
    #         for k, s in enumerate(counts.shape)]
    #     counts_convolved = scipy.signal.convolve(counts, kernel, mode='full')
    #     counts_convolved = np.round(counts_convolved).astype(int)
    #     if is_input_tensor:
    #         counts_convolved = torch.tensor(counts_convolved)
    #
    #     bin_counts[i] = counts_convolved[tuple(downsampled_indices)]
    #     for j in range(len(bin_edges[i])):
    #         binwidth = bin_edges[i][j][1] - bin_edges[i][j][0]
    #         bin_edges[i][j] = bin_edges[i][j][:downsampled_indices[j].flatten()[-1].item():factor]
    #         last_bin_edge = bin_edges[i][j][-1] + factor * binwidth
    #         bin_edges[i][j] = np.pad(bin_edges[i][j], (0, 1), mode='constant', constant_values=(0, last_bin_edge))
    #         if is_input_tensor:
    #             bin_edges[i][j] = torch.tensor(bin_edges[i][j])
    #
    #         if len(bin_edges[i][j]) != bin_counts[i].shape[j] + 1:
    #             raise ValueError(f"Size of downsampled bin counts ({bin_counts[i].shape[j]}) "
    #                              f"and edges ({bin_edges[i][j].shape}) do not match.")

    if not is_input_tensor:
        bin_edges = [e.numpy() for e in bin_edges]
        bin_counts_downsampled = bin_counts_downsampled.numpy()

    if not bin_edges_is_list:
        bin_edges = bin_edges[dim]

    return bin_edges, bin_counts_downsampled


def fast_mu2(X: np.ndarray | torch.Tensor, Y: np.ndarray | torch.Tensor, recursion_limit=50):
    """
    Estimates matrix of second moments M_ij = <X_i Y_j> using samples of two multidimensional random variables X,Y.
    Args:
        X: array of shape=(N,d_X)
        Y: array of shape=(N,d_Y)
        recursion_limit: Maximum number of recursions allowed.

    Returns:
        array of shape=(d_X,d_Y)
    """
    torch_input = isinstance(X, torch.Tensor)
    n_X, d_X = X.shape
    n_Y, d_Y = Y.shape
    assert n_X == n_Y, 'X and Y must have the same number of samples'
    n = n_X
    if recursion_limit < 0:
        raise ValueError(f"Recursion limit has been reached.")

    if n * d_X * d_Y * X.dtype.itemsize < 1e9:
        if torch_input:
            mu2 = torch.mean(X[..., None] * Y[:, None, :], dim=0)
        else:
            mu2 = np.mean(X[..., None] * Y[:, None, :], axis=0)
    elif d_X > 1 and d_Y > 1:
        # Split the data into two parts and calculate the second moment separately for each block
        if torch_input:
            X1, X2 = torch.tensor_split(X, 2, dim=1)
            Y1, Y2 = torch.tensor_split(Y, 2, dim=1)
        else:
            X1, X2 = np.array_split(X, 2, axis=1)
            Y1, Y2 = np.array_split(Y, 2, axis=1)

        new_recursion_limit = recursion_limit - 1
        mu2_11 = fast_mu2(X1, Y1, recursion_limit=new_recursion_limit)
        mu2_12 = fast_mu2(X1, Y2, recursion_limit=new_recursion_limit)
        if id(X) == id(Y):
            mu2_21 = mu2_12.T
        else:
            mu2_21 = fast_mu2(X2, Y1, recursion_limit=new_recursion_limit)
        mu2_22 = fast_mu2(X2, Y2, recursion_limit=new_recursion_limit)

        if torch_input:
            row1 = torch.cat([mu2_11, mu2_12], dim=1)
            row2 = torch.cat([mu2_21, mu2_22], dim=1)
            mu2 = torch.cat([row1, row2], dim=0)
        else:
            mu2 = np.block([[mu2_11, mu2_12], [mu2_21, mu2_22]])
    elif n > 1e4:
        # Split the data into two parts
        if torch_input:
            X1, X2 = torch.tensor_split(X, 2, dim=0)
            Y1, Y2 = torch.tensor_split(Y, 2, dim=0)
        else:
            X1, X2 = np.array_split(X, 2, axis=0)
            Y1, Y2 = np.array_split(Y, 2, axis=0)
        mu2_1 = fast_mu2(X1, Y1, recursion_limit=recursion_limit - 1)
        mu2_2 = fast_mu2(X2, Y2, recursion_limit=recursion_limit - 1)
        mu2 = X1.shape[0] / X.shape[0] * mu2_1 + X2.shape[0] / X.shape[0] * mu2_2

    return mu2


class Statistics:
    """
    Basic statistics class that accumulates the first and second moments of a set of samples
    """
    init_kwargs = ['n', 'mu1', 'mu2', 'min', 'max', 'bin_type', 'binedges', 'bincounts', 'groups_ID', 'note']
    bincounts_dtype = torch.long

    def __init__(self, n: int = 0, mu1: np.ndarray | torch.Tensor = None, mu2: np.ndarray | torch.Tensor = None,
                 min: np.ndarray | torch.Tensor = None, max: np.ndarray | torch.Tensor = None,
                 bincounts: np.ndarray | torch.Tensor | list[np.ndarray] | list[torch.Tensor] = None,
                 binedges: list[np.ndarray] | list[torch.Tensor] = None, bin_type: str = None,
                 groups_ID: torch.IntTensor = None, note=''):
        if bin_type is None:
            bin_type = 'proj' if isinstance(bincounts, list) else 'full'

        self.n = n  # number of datapoints accounted for in the statistics object
        self.d = len(mu1) if mu1 is not None else 0
        self.note = note
        self.min, self.max = min, max
        self.mu1 = mu1  # First raw moments:  <x_i>, i=1...d
        self.mu2 = mu2  # Second central moments: <(x_i-x_i_avg)(x_j-x_j_avg)>, i,j=1...d
        self.bin_type = bin_type

        if binedges is not None:
            assert isinstance(binedges, list), "binedges argument must be a list."
            assert len(binedges) == self.d, \
                f"binedges must have len()={self.d}. Received list has len()={len(binedges)}."
            self.binedges = [torch.as_tensor(e) for e in binedges]

        # Perform checks on the given bincounts
        # If there are no groups, add a trailing dimension of size 1 at the end of all bincounts
        self.groups_ID = torch.as_tensor(groups_ID) if groups_ID is not None else None
        self.n_groups = groups_ID.numel() if groups_ID is not None else 0
        self._bincounts = [] if bincounts is not None else None
        if bincounts is not None:
            if not self.has_groups:
                if bin_type == 'proj':
                    for i, b in enumerate(bincounts):
                        b = torch.as_tensor(b).to(self.bincounts_dtype)
                        assert b.ndim == 1, f"bincounts in dim={i} must be 1D. A {b.ndim} array/tensor was given."
                        self._bincounts.append(b.unsqueeze(-1))
                else:
                    bincounts = torch.as_tensor(bincounts).to(self.bincounts_dtype)
                    assert bincounts.ndim == self.d, \
                        f"Bincounts has {bincounts.ndim} dimension. Expecting {self.d} dimensions."
                    self._bincounts = bincounts.unsqueeze(-1)
            else:
                if bin_type == 'proj':
                    for i, b in enumerate(bincounts):
                        b = torch.as_tensor(b).to(self.bincounts_dtype)
                        assert b.ndim == 2, f"bincounts in dim={i} must be 2D. A {b.ndim}D array was given."
                        assert b.shape[-1] == self.n_groups, f'Trailing dim. of bincounts[{i}] must be {self.n_groups}.'
                        self._bincounts.append(b)
                else:
                    bincounts = torch.as_tensor(bincounts).to(self.bincounts_dtype)
                    assert bincounts.ndim == self.d + 1, \
                        f"Bincounts has {bincounts.ndim} dimension. Expecting {self.d + 1} dimensions."
                    assert bincounts.shape[-1] == self.n_groups, f'Trailing dim. of bincounts must be {self.n_groups}.'
                    self._bincounts = bincounts

    @property
    def sum(self):
        return self.n * self.mu1

    @property
    def mean(self):
        return self.mu1

    @property
    def S2(self):
        return self.n * self.mu2

    @property
    def cov(self):
        if self.n > 1:
            w = self.n / (self.n - 1)
            cov = self.mu2 * w
        elif self.d > 0:
            cov = torch.zeros((self.d, self.d), dtype=self.mu1.dtype, device=self.mu1.device)
        else:
            cov = None
        return cov

    @property
    def std(self):
        cov = self.cov
        if cov is None:
            return None
        if cov.ndim > 1:
            cov = cov.diagonal()
        return torch.sqrt(cov)

    @property
    def sem(self):
        return self.std / np.sqrt(self.n)

    @property
    def names(self):
        defined_stats = list(vars(self).keys()) + ['mean', 'sum', 'cov', 'std', 'sem', 'mu2']
        defined_stats = [s for s in defined_stats if hasattr(self, s) and getattr(self, s) is not None]
        return defined_stats

    @property
    def bin_widths(self):
        if not self.has_bins:
            return None
        bin_widths = torch.stack([(e[-1] - e[0]) / (e.numel() - 1) for e in self.binedges])
        return bin_widths

    @property
    def bin_lims(self):
        if self.has_bins:
            return torch.stack([e[[0, -1]] for e in self.binedges], dim=0)
        else:
            return None

    @property
    def bincounts(self) -> torch.Tensor | list[torch.Tensor]:
        """
        Retrieves the bin counts and averages the counts over groups (if any).
        If groups are defined, counts is a float. Otherwise, the returned counts are integers
        Returns:
            bincounts
        """
        if self.bin_type == 'proj':
            if self.has_groups:
                return [b.double().mean(dim=-1) for b in self._bincounts]
            else:
                return [b.squeeze(-1) for b in self._bincounts]

        elif self.bin_type == 'full':
            if self.has_groups:
                return self._bincounts.double().mean(dim=-1)
            else:
                return self._bincounts.squeeze(-1)
        else:
            raise NotImplementedError

    @property
    def bincounts_err(self) -> torch.Tensor | list[torch.Tensor]:
        bincounts = self._bincounts
        if self.bin_type == 'full':
            bincounts = [bincounts]

        err_type = self.bincounts_err_type
        bincounts_err = []
        for bincounts_dim in bincounts:
            if err_type == 'Poisson':
                # Return Poisson error of the counts when there are no groups defined
                bincounts_err_temp = bincounts_dim.sqrt().squeeze(-1)
            elif err_type == 'Standard':
                if self.n_groups > 1:
                    # Calculate the error over the group dimension (last dim)
                    bincounts_std = bincounts_dim.double().std(dim=-1)
                    bincounts_err_temp = bincounts_std / torch.sqrt(torch.as_tensor(self.n_groups))
                else:
                    bincounts_err_temp = torch.zeros_like(bincounts_dim[..., 0])
            else:
                raise NotImplementedError
            bincounts_err.append(bincounts_err_temp)

        if self.bin_type == 'full':
            bincounts_err = bincounts_err[0]

        return bincounts_err

    @property
    def bincounts_err_type(self):
        # return 'Poisson' if self.n_groups == 0 else 'Standard'
        return 'Standard'

    @property
    def pdf(self) -> tuple[list, list] | tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """
        Calculates the probability density and its error using the bin info
        Returns:
            pdf, pdf_err
        """
        if not self.has_bins:
            return None, None

        bin_widths = self.bin_widths
        bincounts = self.bincounts
        bincounts_err = self.bincounts_err
        if self.bin_type == 'full':
            prob_tot = torch.sum(bincounts) * bin_widths.prod()
            pdf = bincounts / prob_tot
            pdf_err = bincounts_err / prob_tot
        else:
            pdf, pdf_err = [], []
            for i in range(self.d):
                prob_tot = torch.sum(bincounts[i]) * bin_widths[i]
                pdf.append(bincounts[i] / prob_tot)
                pdf_err.append(bincounts_err[i] / prob_tot)

        return pdf, pdf_err

    @property
    def has_groups(self):
        return self.groups_ID is not None

    @property
    def has_bins(self):
        return self._bincounts is not None

    @property
    def device(self):
        if self._bincounts is not None:
            if self.bin_type == 'proj':
                return self._bincounts[0].device
            else:
                return self._bincounts.device
        return None

    @classmethod
    def from_data(cls, data: np.ndarray | torch.Tensor | float, bin_width=None, bin_lims=(None, None), bin_type='proj',
                  remove_nan=True, data_group_ID: np.ndarray | torch.Tensor = None, **kwargs):
        data = torch.as_tensor(data).contiguous()

        if data.ndim < 2:
            data = data.reshape(-1, 1)

        # Remove entries that have nans in any of their dimension
        data_isnan = torch.isnan(data).any(dim=-1)
        n_nans = data_isnan.sum().item()
        all_nans = n_nans == data_isnan.shape[0]
        if data_isnan.any():
            warnings.warn(f"Found {n_nans} nans in data of shape={data.shape}.")
            if remove_nan:
                if not all_nans:
                    warnings.warn(f"Removing nans...")
                    data = data[~data_isnan]
                else:
                    warnings.warn(f"All data points were nans. Data will be set to zero.")
                    # data[data_isnan, :] = 0.0

        n, d = data.shape
        n_half = n // 2
        if bin_type not in ['proj', 'full']:
            raise NotImplementedError("Only projected (proj) or full (full) bin_type is supported.")
        if bin_type == 'full':
            assert d < 5, f'Full bin type is only possible for d<5. Received d={d}'
        Min = torch.min(data, dim=0)[0]
        Max = torch.max(data, dim=0)[0]
        mu1 = torch.mean(data, dim=0)

        data_centered = data - mu1
        mu2 = fast_mu2(data_centered, data_centered)

        # Check the number of independent group IDs and change ID into a 0-based index
        # data_group_ID = None
        groups_ID = None
        if data_group_ID is not None:
            # Find the group index of each data point
            data_group_ID = torch.as_tensor(data_group_ID)
            groups_ID = data_group_ID.unique()
            n_groups = groups_ID.numel()
            data_group_ID = data_group_ID.unsqueeze(-1) == groups_ID.unsqueeze(0)
            data_group_ind = data_group_ID.nonzero().long()[:, 1]

        # Bin data along each dimension separately if bin info were given
        binedges, bincounts, bincounts_sq = None, None, None
        if bin_width is not None:
            # Default bin limits
            if all_nans:
                bin_min = -bin_width * torch.ones_like(mu1)
                bin_max = bin_width * torch.ones_like(mu1)
            else:
                bin_min = torch.floor(Min / bin_width) * bin_width
                bin_max = torch.ceil(Max / bin_width) * bin_width

            # Redefine bin limits if one of the bin limits is defined
            if bin_lims[0] is None and bin_lims[1] is not None:
                bin_max = bin_lims[1] * torch.ones_like(mu1)
                bin_min = bin_max - torch.ceil((bin_max - Min) / bin_width) * bin_width  # Ensures int number of bins
            elif bin_lims[0] is not None and bin_lims[1] is None:
                bin_min = bin_lims[0] * torch.ones_like(mu1)
                bin_max = torch.ceil((Max - bin_min) / bin_width) * bin_width + bin_min  # Ensures int number of bins
            elif bin_lims[0] is not None and bin_lims[1] is not None:
                bin_min = bin_lims[0] * torch.ones_like(mu1)
                bin_max = bin_lims[1] * torch.ones_like(mu1)

            # Ensure there is at least 1 bin by making sure maximum is at least 1 binwidth away from min.
            bin_max = torch.maximum(bin_min + bin_width, bin_max)

            # Reduce the bin ranges if the number of bins is above threshold
            n_bins_thresh = int(1e4)
            bin_lim = torch.stack([bin_min, bin_max], dim=1)
            n_bins = torch.round(bin_lim.diff(dim=1) / bin_width).reshape(d, )
            n_bins = n_bins.clamp(min=1, max=torch.iinfo(torch.long).max - 1024).long()
            n_bins_argmax = torch.argmax(n_bins)
            n_bins_max = n_bins[n_bins_argmax]
            if n_bins_max > n_bins_thresh:
                warnings.warn(f'The number of bins is very high(n={n_bins_max}) for data of shape={data.shape}.\n'
                              f'binwidth={bin_width}, bin_lim={bin_lim[n_bins_argmax]}.')

                # Reduce the bin range to reduce the number of bins
                # Sort the data and remove 1 data point on each side of the range until the number of bins is below threshold
                truncated_data = torch.sort(data, dim=0)[0]
                while torch.max(n_bins) > n_bins_thresh:
                    # Decide whether to remove the smallest or largest point for reducing the range
                    if truncated_data[:2].diff(dim=0).mean() < truncated_data[-2:].diff(dim=0).mean():
                        # Remove largest
                        truncated_data = truncated_data[:-1]
                        bin_lim[:, 1] = torch.ceil(truncated_data[-1, :] / bin_width) * bin_width
                    else:
                        # Remove smallest
                        truncated_data = truncated_data[1:]
                        bin_lim[:, 0] = torch.floor(truncated_data[0, :] / bin_width) * bin_width
                    n_bins = torch.round(bin_lim.diff(dim=1) / bin_width).reshape(d, )
                    n_bins = n_bins.clamp(min=1, max=torch.iinfo(torch.long).max - 1024).long()
                trunc_ratio = 1 - truncated_data.shape[0] / data.shape[0]
                warnings.warn(f'{trunc_ratio:.2%} of data points were removed to reduce the bin range.')

            binedges = [torch.linspace(bin_lim[i, 0], bin_lim[i, 1], n_bins[i] + 1).to(data.device) for i in range(d)]
            if bin_type == 'proj':
                # Bin data along each dimension separately. Each entry in self._bincounts represents a marginal dist.
                bincounts = []
                for i in range(d):
                    data_bin_ind = torch.bucketize(data[:, i].contiguous(), boundaries=binedges[i], right=True)
                    data_bin_ind -= 1
                    data_bin_ind[data_bin_ind == n_bins[i]] -= 1  # Needed if some data is on the last edge

                    if data_group_ID is None:
                        # counts, edges = np.histogram(data[:, i], bins=n_bins[i], range=tuple(bin_lim[i].tolist()))
                        # counts = counts.astype(int)
                        counts = torch.zeros(n_bins[i], dtype=torch.long, device=data.device)
                        counts.index_put_(indices=(data_bin_ind,),
                                          values=torch.ones_like(data_bin_ind, dtype=counts.dtype),
                                          accumulate=True)
                    else:
                        counts = torch.zeros((n_bins[i], n_groups), dtype=torch.long, device=data.device)
                        counts.index_put_(indices=(data_bin_ind, data_group_ind),
                                          values=torch.ones_like(data_bin_ind, dtype=counts.dtype),
                                          accumulate=True)
                    bincounts.append(counts)
            elif bin_type == 'full':
                # Determine the bin multi-index of each data point
                data_bin_ind = torch.zeros((data.shape[0], data.shape[1] + 1), dtype=torch.long, device=data.device)
                for i in range(d):
                    data_bin_ind[:, i] = torch.bucketize(data[:, i].contiguous(), boundaries=binedges[i], right=True)
                    data_bin_ind[:, i] -= 1
                    data_bin_ind[data_bin_ind[:, i] == n_bins[i], i] -= 1  # Needed if some data is on the last edge

                if data_group_ID is not None:
                    data_bin_ind[:, -1] = data_group_ind
                    bincounts_shape = (*n_bins, n_groups)
                else:
                    bincounts_shape = n_bins
                    data_bin_ind = data_bin_ind[:, :-1]  # Remove group index since it is not needed

                bincounts = torch.zeros(*bincounts_shape, dtype=torch.long, device=data.device)
                bincounts.index_put_(indices=data_bin_ind.unbind(-1),
                                     values=torch.ones_like(data_bin_ind[:, 0], dtype=bincounts.dtype),
                                     accumulate=True)
            else:
                raise NotImplementedError(f"Bin type {bin_type!r} is not implemented")

        return cls(n=n, mu1=mu1, mu2=mu2, min=Min, max=Max, binedges=binedges, bincounts=bincounts,
                   bin_type=bin_type, groups_ID=groups_ID, **kwargs)

    @classmethod
    def empty(cls, d: int, bin_width: torch.Tensor | float = None, bin_lims: torch.Tensor | tuple[float, float] = None):
        mu1 = torch.zeros(d, )
        mu2 = torch.zeros(d, d)
        min, max = torch.inf * torch.ones(d, ), -torch.inf * torch.ones(d, )
        if bin_width is not None and bin_lims is not None:
            n_bins = int(torch.diff(bin_lims) / bin_width + 1)
            binedges = torch.linspace(bin_lims[0], bin_lims[1], n_bins)
            bincounts = torch.zeros_like(binedges)[:-1]
        else:
            binedges, bincounts = None, None
        return cls(n=0, mu1=mu1, mu2=mu2, min=min, max=max, binedges=binedges, bincounts=bincounts)

    def clamp_n_bins(self, n_bins_max: int):
        """
        Downsamples bins in each dimension such that the number of bins is less than the given maximum
        Args:
            n_bins_max: maximum number of bins for each dimension

        Returns:
            None
        """
        n_bins = torch.tensor([e.numel() - 1 for e in self.binedges])
        for i in range(self.d):
            downsampling_rate = int(torch.ceil(n_bins[i] / n_bins_max))
            if downsampling_rate > 1:
                self.downsample_bins(factor=downsampling_rate, dim=i)

    def extend_bins(self, new_bin_lims: np.ndarray | torch.Tensor | list):
        new_bin_lims = torch.as_tensor(new_bin_lims).to(self.device)
        if new_bin_lims.numel() == 2:
            new_bin_lims = torch.ones((self.d, 1), device=self.device) * new_bin_lims.reshape(1, 2)
        assert tuple(new_bin_lims.shape) == (self.d, 2), "The new bin limits must have a shape of (self.d,2)"

        if self.bin_type == 'proj':
            for i, (binedges, bincounts) in enumerate(zip(self.binedges, self._bincounts)):
                self.binedges[i], self._bincounts[i] = extend_bins(binedges, bincounts, new_lims=new_bin_lims[i])
        else:
            self.binedges, self._bincounts = extend_bins(self.binedges, self._bincounts, new_lims=new_bin_lims)

    def downsample_bins(self, factor: int = 2, dim: np.ndarray | torch.Tensor | list | int = None, inplace=True):
        """
        Downsamples bin-related attributes by a given integer factor
        Args:
            factor: integer determining the level of down sampling
            dim: dimension which will be downsampled. Defaults to all dimensions
            inplace: performs the downsampling inplace or returns a new Statistics object with downsampled bins
        Returns:
            None or Statistics (if inplace=False)
        """
        if dim is None:
            dim = torch.arange(self.d, device=self.device)
        downsampled_dims = torch.as_tensor(dim).reshape(-1, )

        # Find the new bin limits in each dimension
        new_bin_lims = self.bin_lims
        bin_widths = self.bin_widths
        new_bin_widths = factor * bin_widths
        for j in downsampled_dims:
            # Round the bin edges to integer-multiples of the bin width to ensure floor and ceil do not overshoot due
            # to floating-point accuracy.
            bin_lims_rounded = torch.round(self.binedges[j][[0, -1]] / bin_widths[j])
            new_bin_lims[j, 0] = torch.floor(bin_lims_rounded[0] / factor) * new_bin_widths[j]
            new_bin_lims[j, 1] = torch.ceil(bin_lims_rounded[1] / factor) * new_bin_widths[j]

        # Copy self for output if not done inplace
        if inplace:
            output_stats = self
        else:
            output_stats = copy.deepcopy(self)

        if self.bin_type == 'proj':
            # In this case, bincounts represent many one-dimensional distributions
            # Each distribution needs to be downsampled separately
            for j in downsampled_dims:
                curr_bin_lims = self.binedges[j][[0, -1]]
                if not (curr_bin_lims == new_bin_lims[j]).all():
                    edges_ext, counts_ext = extend_bins(self.binedges[j], self._bincounts[j], new_lims=new_bin_lims[j])
                else:
                    edges_ext, counts_ext = self.binedges[j], self._bincounts[j]
                edges_down, counts_down = downsample_bins(bin_edges=edges_ext, bin_counts=counts_ext, factor=factor,
                                                          dim=0)
                output_stats.binedges[j] = edges_down
                output_stats._bincounts[j] = counts_down
        else:
            # In this case, bincounts represent a d-dimensional density
            edges_ext, counts_ext = extend_bins(self.binedges, self._bincounts, new_lims=new_bin_lims)
            output_stats.binedges, output_stats._bincounts = downsample_bins(bin_edges=edges_ext,
                                                                             bin_counts=counts_ext,
                                                                             factor=factor,
                                                                             dim=downsampled_dims)
        return output_stats

    def cast_bins_to_lists(self):
        raise NotImplementedError
        if not isinstance(self.binedges, list):
            self.binedges = [self.binedges]
        if not isinstance(self.bincounts, list):
            self.bincounts = [self.bincounts]
        if self.bincounts_sq is not None and not isinstance(self.bincounts_sq, list):
            self.bincounts_sq = [self.bincounts_sq]

    def to(self, inplace=True, device=None) -> Self:
        """
        Modifies the device of torch.Tensor attributes
        Returns: Statistics
        """
        if inplace:
            stats = self
        else:
            stats = copy.deepcopy(self)

        for attr_name, attr_val in stats.__dict__.items():
            if isinstance(attr_val, torch.Tensor):
                setattr(stats, attr_name, attr_val.to(device=device))
            elif isinstance(attr_val, list) and isinstance(attr_val[0], torch.Tensor):
                setattr(stats, attr_name, [v.to(device=device) for v in attr_val])

        return stats

    def collapse_dim(self):
        """
        Collapses all dimensions into a single dimension assuming that each dimension is independent of one another.
        """
        if self.bin_type == 'full':
            warnings.warn("Cannot collapse dimensions when bin_type='full'. Dimensions won't be collapsed.")
            return
        if self.d == 1:
            return
        self.n = self.n * self.d
        self.d = 1
        self.mu1 = self.mu1.mean().reshape(1, )
        s2 = torch.mean(self.mu2.diag() + self.mu1.square())  # <x^2> = 1/d sum_i <(x_i - x_i_avg)^2> + x_i_avg^2
        self.mu2 = (s2 - self.mu1.square()).reshape(1, )
        self.min, self.max = self.min.min().reshape(1, ), self.max.max().reshape(1, )
        bincounts, binedges = combine_bins(bin_counts=self._bincounts, bin_edges=self.binedges)
        self._bincounts, self.binedges = [bincounts], [binedges]

    def transform(self, P: torch.Tensor):
        """
        Transforms statistics to obtain the statistics of x' = x P where x' and x are row vectors representing the
        new and old data, respectively.
        Args:
            P: Matrix representing the transformation. shape=(d,d_new) where d_new is the new dimension of the data
        """
        self.mu1 = self.mu1 @ P
        self.mu2 = P.T @ self.mu2 @ P
        self.d = len(self.mu1)
        self.min, self.max = None, None

    def scale(self, s: torch.Tensor | float):
        """
        Transforms statistics to obtain the statistics of y = s*x
        Args:
            s: Scale vector of shape=(d,) or (1,)
        """
        if isinstance(s, float) or s.ndim == 0 or s.shape[-1] == 1:
            s = s * torch.ones((self.d,))
        self.mu1 = s * self.mu1
        self.mu2 = torch.outer(s, s) * self.mu2
        self.min, self.max = s * self.min, s * self.max
        if self.binedges:
            if self.d > 1:
                for i, bin_edges in enumerate(self.binedges):
                    self.binedges[i] = s[i] * bin_edges
            else:
                self.binedges = s * self.binedges

    def prob_dist(self, other: Self, dist_type='JSD', average=True):
        """
        Calculates the given probability distance between self and other by averaging the distance over all d distributions.
        Args:
            other: Statistics to compare with
            dist_type: distance type used to compare the distributions
                    'JSD': Jensen-Shannon distance as computed in scipy
                    'JD': Jeffrey's distance as computed by J_dist_bin()
            average: Returns the distance averaged over all dimensions
        Returns: scalar or torch.Tensor with shape=(d,) (if average=False)
        """
        assert self.d == other.d, f"dimensions do not match self.d={self.d}, other.d={other.d}"
        assert self.bin_type == other.bin_type, (f"Bin type must match to calculated prob_dist. "
                                                 f"Received: {[self.bin_type, other.bin_type]}")
        prob_dist_dim = torch.zeros(self.d, device=self.device)
        self_bincounts, other_bincounts = self.bincounts, other.bincounts
        self_binedges, other_binedges = self.binedges, other.binedges
        self_binwidths, other_binwidths = self.bin_widths, other.bin_widths

        # Calculate the new bin edges limits
        self_lims, other_lims = self.bin_lims, other.bin_lims
        bin_edges_min = torch.min(self_lims[:, 0], other_lims[:, 0])
        bin_edges_max = torch.max(self_lims[:, -1], other_lims[:, -1])
        new_bin_lim = torch.stack([bin_edges_min, bin_edges_max], -1)  # shape=(d,2)
        if self.bin_type == 'full':
            self_bincounts = [self_bincounts]
            other_bincounts = [other_bincounts]
            self_binedges = [self_binedges]
            other_binedges = [other_binedges]
            new_bin_lim = [new_bin_lim]

        for i in range(len(other_bincounts)):
            p_counts = torch.as_tensor(self_bincounts[i])
            q_counts = torch.as_tensor(other_bincounts[i])
            p_bin_edges = self_binedges[i]
            q_bin_edges = other_binedges[i]

            # Extend bins so that both self and other cover the same range
            p_bin_edges, p_counts = extend_bins(p_bin_edges, p_counts, new_bin_lim[i])
            q_bin_edges, q_counts = extend_bins(q_bin_edges, q_counts, new_bin_lim[i])

            if dist_type == 'JSD':
                prob_dist_dim[i] = torch.sqrt(JS_div(p_counts.flatten(), q_counts.flatten()))
            elif dist_type == 'JSD_scipy':
                prob_dist_dim[i] = scipy.spatial.distance.jensenshannon(p_counts.cpu().flatten(),
                                                                        q_counts.cpu().flatten())
            elif dist_type == 'JD':
                # Calculate the probability density since J_dist_bin requires normalized densities
                if self.bin_type == 'full':
                    p_prob_tot = torch.sum(p_counts) * self_binwidths.prod()
                    q_prob_tot = torch.sum(q_counts) * other_binwidths.prod()
                else:
                    p_prob_tot = torch.sum(p_counts) * self_binwidths[i]
                    q_prob_tot = torch.sum(q_counts) * other_binwidths[i]
                p_prob = p_counts / p_prob_tot
                q_prob = q_counts / q_prob_tot

                prob_dist_dim[i] = J_dist_bin(p_prob, q_prob)
            else:
                raise NotImplementedError(f"Distance {dist_type!r} is not implemented.")

        if self.bin_type == 'full':
            prob_dist_dim = prob_dist_dim[0]

        # Average over all dimensions
        if average:
            return prob_dist_dim.mean()
        else:
            return prob_dist_dim

    def plot_dist(self, dim=0, **kwargs):
        import matplotlib.pyplot as plt
        plt.figure()
        bin_edges = self.binedges[dim]
        bin_counts_err = self.bincounts_err
        if self.bin_type == 'proj':
            bin_counts = self.bincounts[dim]
            if bin_counts_err is not None:
                bin_counts_err = bin_counts_err[dim]
        else:
            bin_counts = self.bincounts
            summed_dims = tuple(i for i in range(self.d) if i != dim)
            if summed_dims:
                bin_counts = bin_counts.sum(dim=summed_dims)
                if bin_counts_err is not None:
                    bin_counts_err = bin_counts_err.sum(dim=summed_dims)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.hist(bin_centers, bins=bin_edges, weights=bin_counts, **kwargs)
        plt.ylabel('counts')
        ax = plt.gca()
        if bin_counts_err is not None:
            plot_mu_sigma(mu=bin_counts, sigma=bin_counts_err, x=bin_centers)

        # Restrict the x-axis range
        bin_width = torch.diff(bin_edges[:2])
        first_nonzero_bin_ind, last_nonzero_bin_ind = torch.nonzero(bin_counts)[[0, -1]]
        x_lim_min = bin_edges[first_nonzero_bin_ind] - 2 * bin_width
        x_lim_max = bin_edges[last_nonzero_bin_ind] + 2 * bin_width
        ax.set_xlim(x_lim_min, x_lim_max)

    def __add__(self, other):
        if isinstance(other, Statistics):
            if self.d and other.d and self.d != other.d:
                raise ValueError(f'Cannot add two statistics of different dimension: d1={self.d}, d2={other.d}')
            if self.d == 0 and other.d == 0:
                return self.__class__()  # Return empty statistics

            is_stats_groups_def_agree = self.has_groups == other.has_groups
            assert is_stats_groups_def_agree, "Cannot combine two statistics where only one defines ind. groups."

            if self._bincounts is not None and other._bincounts is not None:
                bin_types = [self.bin_type, other.bin_type]
                bin_widths = [self.bin_widths, other.bin_widths]
                if len(set(bin_types)) > 1:
                    raise ValueError(f"Cannot combine Statistics with different bin type:{bin_types}")
                if not torch.isclose(bin_widths[0], bin_widths[1], atol=1e-6).all():
                    raise ValueError(f"Cannot combine Statistics with different bin width:{bin_widths}")

                if self.has_groups:
                    # In this case, bincounts are concatenated along the group dimension (last dim)

                    # Check if the ID of the groups that are combined are distinct
                    self_grps_ID = self.groups_ID.tolist()
                    other_grps_ID = other.groups_ID.tolist()
                    common_group_IDs = set(self_grps_ID) & set(other_grps_ID)
                    if common_group_IDs:
                        common_grp_ind_self = torch.tensor(
                            [self_grps_ID.index(ind) for ind in other_grps_ID if ind in self_grps_ID])
                        other_grp_is_in_self = torch.tensor([ind in self_grps_ID for ind in other_grps_ID])

                    # Extend the bin edges and counts to span the range of both self and other
                    bin_edges_min = [torch.min(e1[0], e2[0]) for e1, e2 in zip(self.binedges, other.binedges)]
                    bin_edges_max = [torch.max(e1[-1], e2[-1]) for e1, e2 in zip(self.binedges, other.binedges)]
                    new_bin_lims = torch.stack([torch.stack(bin_edges_min), torch.stack(bin_edges_max)], dim=-1)
                    if self.bin_type == 'full':
                        binedges, counts1 = extend_bins(self.binedges, self._bincounts, new_bin_lims)
                        _, counts2 = extend_bins(other.binedges, other._bincounts, new_bin_lims)
                        if common_group_IDs:
                            bincounts = counts1.clone()
                            bincounts[..., common_grp_ind_self] += counts2[..., other_grp_is_in_self]
                            bincounts = torch.cat([bincounts, counts2[..., ~other_grp_is_in_self]], dim=-1)
                        else:
                            bincounts = torch.cat([counts1, counts2], dim=-1)
                    elif self.bin_type == 'proj':
                        # Combine bincounts in each dimension
                        binedges = [torch.empty((0,)) for _ in range(self.d)]
                        bincounts = [torch.empty((0,)) for _ in range(self.d)]
                        for i in range(self.d):
                            binedges[i], counts1 = extend_bins(self.binedges[i], self._bincounts[i], new_bin_lims[i])
                            _, counts2 = extend_bins(other.binedges[i], other._bincounts[i], new_bin_lims[i])
                            if common_group_IDs:
                                bincounts[i] = counts1.clone()
                                bincounts[i][..., common_grp_ind_self] += counts2[..., other_grp_is_in_self]
                                bincounts[i] = torch.cat([bincounts[i], counts2[..., ~other_grp_is_in_self]], dim=-1)
                            else:
                                bincounts[i] = torch.cat([counts1, counts2], dim=-1)
                    else:
                        raise NotImplementedError

                else:
                    if self.bin_type == 'full':
                        bincounts, binedges = combine_bins(
                            bin_counts=[self._bincounts.squeeze(-1), other._bincounts.squeeze(-1)],
                            bin_edges=[self.binedges, other.binedges])
                    elif self.bin_type == 'proj':
                        binedges = [torch.empty((0,)) for _ in range(self.d)]
                        bincounts = [torch.empty((0,)) for _ in range(self.d)]
                        for i in range(self.d):
                            bincounts[i], binedges[i] = combine_bins(
                                bin_counts=[self._bincounts[i].squeeze(-1), other._bincounts[i].squeeze(-1)],
                                bin_edges=[self.binedges[i], other.binedges[i]])
                    else:
                        raise NotImplementedError
            elif not self._bincounts == other._bincounts:
                raise ValueError(f"Cannot add bin counts from a statistics that has no bin counts.")
            else:
                bincounts, binedges = None, None

            # Calculate the new summary statistics
            n = self.n + other.n
            mu1, mu2 = torch.zeros_like(self.mu1), torch.zeros_like(self.mu2)
            Min, Max = torch.ones_like(self.mu1), torch.ones_like(self.mu1)
            Min *= torch.inf
            Max *= -torch.inf

            if self.d > 0:
                w_self = self.n / n
                mu1 += w_self * self.mu1
                mu2 += w_self * self.mu2
                Min = torch.minimum(Min, self.min)
                Max = torch.maximum(Max, self.max)

            if other.d > 0:
                w_other = other.n / n
                mu1 += w_other * other.mu1
                mu2 += w_other * other.mu2
                Min = torch.minimum(Min, other.min)
                Max = torch.maximum(Max, other.max)

            # Reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance, "Parallel Algorithm"
            if self.d > 0 and other.d > 0:
                delta = self.mu1 - other.mu1
                mu2 += torch.outer(delta, delta) * w_self * w_other
            if self.has_groups:
                if common_group_IDs:
                    new_groups_ID = other.groups_ID[~other_grp_is_in_self]
                else:
                    new_groups_ID = other.groups_ID
                groups_ID = torch.cat([self.groups_ID, new_groups_ID])
            else:
                groups_ID = None

            return self.__class__(n=n, mu1=mu1, mu2=mu2, min=Min, max=Max, bincounts=bincounts, binedges=binedges,
                                  groups_ID=groups_ID)

        elif isinstance(other, np.ndarray | torch.Tensor | float):
            other = torch.as_tensor(other).to(device=self.mu1.device)
            return self + self.from_data(data=other)
        else:
            raise ValueError(f"Adding method not supported for {other.__class__}")

    def __eq__(self, other):
        if isinstance(other, Statistics):
            if self.n != other.n or self.d != other.d:
                return False

            # Check equality of first moments
            tolerance = 1e2 * torch.finfo(self.binedges[0].dtype).eps
            is_mu1_equal = torch.isclose(self.mu1, other.mu1, atol=tolerance)
            if not is_mu1_equal.all():
                return False

            # Check equality of second moments
            is_mu2_equal = torch.isclose(self.mu2, other.mu2, atol=tolerance)
            if not is_mu2_equal.all():
                return False

            return True
        else:
            raise ValueError(f"__eq__ method not supported for {other.__class__}")

    def __repr__(self):
        return f"{self.__class__.__name__}(n={self.n},\nmu1={self.mu1},\ncov={self.cov})"

    def __setstate__(self, state: dict):
        # Cast bincounts to given dtype
        if state['bincounts'] is not None:
            if isinstance(state['bincounts'], list):
                bincounts = [torch.from_numpy(b).to(self.bincounts_dtype) for b in state.pop('bincounts')]
                n_bins = [b.shape[0] for b in bincounts]
            else:
                bincounts = torch.from_numpy(state.pop('bincounts')).to(self.bincounts_dtype)
                n_bins = bincounts.shape
                if state['groups_ID'] is not None:
                    n_bins = n_bins[:-1]

            # Reform the bin edges
            bin_lims = state.pop('bin_lims')
            binedges = [torch.linspace(lim[0], lim[1], n + 1) for lim, n in zip(bin_lims, n_bins)]
        else:
            bincounts, binedges = None, None

        init_kwargs = {n: state.pop(n) for n in set(self.init_kwargs) & set(state.keys())}
        self.__init__(bincounts=bincounts, binedges=binedges, **init_kwargs)
        self.__dict__.update(state)

        if 'device' in state:
            self.to(device=state['device'], inplace=True)

    def __getstate__(self):
        state = {k: copy.deepcopy(getattr(self, k)) for k in self.__dict__}
        state['bincounts'] = state.pop('_bincounts')

        # Save device of tensors
        state['device'] = self.device

        # Switch to cpu()
        for attr_name, attr_val in state.items():
            if isinstance(attr_val, torch.Tensor):
                state[attr_name] = attr_val.cpu()

        # Save only the binedges limits and binwidth. Binedges array will be reformed in __set_state__
        if self.has_bins:
            state.pop('binedges')
            state['bin_lims'] = self.bin_lims.cpu().numpy()

            if self.bin_type == 'full':
                state['bincounts'] = [state['bincounts']]

            # Cast bincounts to numpy arrays with given saved dtype
            for i, b in enumerate(state['bincounts']):
                if not self.has_groups:
                    b = b.squeeze(-1)
                state['bincounts'][i] = b.cpu().numpy()
                state['bincounts'][i] = cast_to_optimal_dtype(state['bincounts'][i])

            if self.bin_type == 'full':
                state['bincounts'] = state['bincounts'][0]

        return state

    @classmethod
    def test(cls):
        dims = list(range(1, 4))
        bin_types = ['proj', 'full']
        use_groups = [True, False]
        filepath = tempfile.mktemp(suffix='_Statistics_test.npz')
        arr_types = ['numpy'] + (['cuda'] if torch.cuda.is_available() else [])
        for d, bin_type, use_grp, arr_type in itertools.product(dims, bin_types, use_groups, arr_types):
            data1_grp_ID = None
            data2_grp_ID = None
            if arr_type == 'cuda':
                data1 = torch.randn(10000, d).cuda()
                data2 = torch.randn(9000, d).cuda()
                if use_grp:
                    data1_grp_ID = torch.randint(0, 4, (data1.shape[0],)).cuda()
                    data2_grp_ID = torch.randint(2, 8, (data2.shape[0],)).cuda()
            else:
                data1 = np.random.randn(10000, d)
                data2 = np.random.randn(9000, d)
                if use_grp:
                    data1_grp_ID = np.random.randint(0, 4, (data1.shape[0],))
                    data2_grp_ID = np.random.randint(2, 8, (data2.shape[0],))

            stat1 = cls.from_data(data1, bin_width=0.1, bin_lims=(-10.1, 10.1), bin_type=bin_type,
                                  data_group_ID=data1_grp_ID)
            stat2 = cls.from_data(data2, bin_width=0.1, bin_lims=(-7, 8), bin_type=bin_type,
                                  data_group_ID=data2_grp_ID)
            # Check addition
            stat3 = stat1 + stat2
            if stat3.bin_type == 'full':
                assert stat3._bincounts.sum() == stat1._bincounts.sum() + stat2._bincounts.sum()
            else:
                for i in range(stat3.d):
                    assert stat3._bincounts[i].sum() == stat1._bincounts[i].sum() + stat2._bincounts[i].sum()

            # Test bin limits extension
            stat3.extend_bins(new_bin_lims=[-11.1, 12.1])

            # Test downsampling of bins
            stat3.downsample_bins(factor=2, dim=d - 1)
            stat3.downsample_bins(factor=2, dim=None)

            # Test reduction of number of bins on large floats
            stat4 = cls.from_data(1e6 * data1, bin_width=0.1, bin_lims=(None, None), bin_type=bin_type,
                                  data_group_ID=data1_grp_ID)

            # Test calculation of JSD
            jdist12 = stat1.prob_dist(stat2, dist_type='JSD')
            jdist12_scipy = stat1.prob_dist(stat2, dist_type='JSD_scipy')
            assert jdist12.isclose(jdist12_scipy), f"JS dist does not agree with scipy's. {jdist12}!={jdist12_scipy}"

            # Test dimension collapse
            stat1.collapse_dim()

            # Test saving and loading of Statistics object
            saved_stats_dict = {'stat1': stat1, 'stat2': stat2}
            np.savez(filepath, **saved_stats_dict)
            with np.load(filepath, allow_pickle=True) as npzfile:
                loaded_stats = {k: v.item() for k, v in npzfile.items()}

            if loaded_stats != saved_stats_dict:
                raise ValueError(f"Saved and loaded stats do not agree.")

            # Test plotting of distribution
            # stat3.plot_dist()

        # Clean up
        os.remove(filepath)


class StatisticsArray(nn.Module):
    """
    Class defining an 1D array of statistics that share the same set of bins.
    """

    def __init__(self, shape: tuple | torch.Size = torch.Size([]), d: int = None, dtype=None,
                 cov_method='full', values: torch.Tensor = None, ind: torch.Tensor = None):
        super().__init__()
        assert not ((values is None) ^ (ind is None)), "values and ind must be both None or not None"
        self.shape = torch.Size(shape)
        if values is not None:
            device = values.device
            if dtype is None:
                dtype = values.dtype
            else:
                values = values.to(dtype)

            # If the array is 1D, add a trailing dimension of size 1 to ind if it doesn't have one.
            if len(self.shape) == 1 and ind.shape[-1] != 1:
                ind = ind.unsqueeze(-1)

            # Check that the trailing dim of ind has the same size as the number of dimensions of the array.
            if ind.shape[-1] != len(shape):
                raise ValueError(f"ind.shape[-1] and len(shape) do not match.\n"
                                 f"ind.shape={ind.shape}, shape={shape}")

            # Determine d
            if d is None:
                d = int(values.nelement() / ind.shape[:-1].numel())

            # Add a trailing dimension to values when d=1
            if d == 1 and values.shape[-1] != d:
                values = values.unsqueeze(-1)
            elif values.shape[-1] != d:
                raise ValueError(f"The dimension of values ({values.shape[-1]}) does not match d ({d})")

            # Check that the leading dimensions of ind and values match
            if ind.shape[:-1] != values.shape[:-1]:
                raise ValueError(f"ind.shape[:-1] and values.shape[:-1] do not match.\n"
                                 f"ind.shape={ind.shape}, values.shape={values.shape}")
        else:
            device = None

        self.d = d
        self.n = nn.parameter.Buffer(torch.zeros(*shape, device=device, dtype=dtype), persistent=True)
        self.mean = nn.parameter.Buffer(torch.zeros((*shape, self.d), device=device, dtype=dtype), persistent=True)
        self.min = nn.parameter.Buffer(torch.zeros((*shape, self.d), device=device, dtype=dtype), persistent=True)
        self.max = nn.parameter.Buffer(torch.zeros((*shape, self.d), device=device, dtype=dtype), persistent=True)
        self.cov_method = cov_method
        if self.cov_method == 'diag':
            self.S2 = nn.parameter.Buffer(torch.zeros((*shape, self.d), device=device, dtype=dtype), persistent=True)
        else:
            self.S2 = nn.Parameter(torch.zeros((*shape, self.d, self.d), device=device, dtype=dtype),
                                   requires_grad=False)
        self.dtype = self.mean.dtype
        self.cov_inv_chol_ = None

        if values is not None:
            ind_tuple = ind.unbind(dim=-1)

            # Calculate min and max
            if len(shape) != 1:
                raise NotImplementedError
            self.min.scatter_reduce_(dim=0, index=ind, src=values, reduce='amin', include_self=False)
            self.max.scatter_reduce_(dim=0, index=ind, src=values, reduce='amax', include_self=False)

            # Calculate n and average
            self.n.index_put_(ind_tuple, torch.ones((ind.shape[0],), device=device, dtype=dtype), accumulate=True)
            self.mean.index_put_(ind_tuple, values, accumulate=True)
            self.mean /= self.n.unsqueeze(-1) + 1e-8

            # Calculate sum of squares S2_ij = sum_k (x_i,k - <x_i,k>)(x_j,k - <x_j,k>)
            delta = values - self.mean[ind_tuple]
            if self.cov_method == 'diag':
                delta_outer = torch.einsum('...i,...i->...i', delta, delta)
            else:
                delta_outer = torch.einsum('...i,...j->...ij', delta, delta)
            self.S2.index_put_(ind_tuple, delta_outer, accumulate=True)

    @property
    def mu2(self):
        if self.cov_method == 'diag':
            return self.S2 / self.n[..., None] + self.mean.square()
        else:
            return self.S2 / self.n[..., None, None] + self.mean[..., None, :] * self.mean[..., None]

    @property
    def mu2_diag(self):
        if self.cov_method == 'diag':
            return self.mu2
        else:
            return self.mu2.diagonal(dim1=-2, dim2=-1)

    @property
    def sem(self):
        return self.std / self.n[..., None]

    @property
    def cov(self):
        if self.cov_method == 'diag':
            return self.S2 / (self.n[..., None] - 1)
        else:
            return self.S2 / (self.n[..., None, None] - 1)

    @property
    def cov_inv_chol(self):
        if self.cov_inv_chol_ is None:
            if self.cov_method == 'full':
                self.cov_inv_chol_, info = torch.linalg.cholesky_ex(torch.linalg.inv(self.cov))
                n_errors = (info != 0).sum()
                if n_errors > 0:
                    warnings.warn(f"{n_errors} cholesky decompositions returned with errors for {self}")
            elif self.cov_method == 'diag':
                self.cov_inv_chol_ = 1 / self.cov.sqrt()
            else:
                raise NotImplementedError
        return self.cov_inv_chol_

    @property
    def std(self):
        if self.cov_method == 'diag':
            return self.cov.sqrt()
        else:
            return self.cov.diagonal(dim1=-1, dim2=-2).sqrt()

    def __add__(self, other: Self) -> Self:
        output_stats = StatisticsArray(shape=self.shape, d=self.d, dtype=self.dtype)
        output_stats += self
        output_stats += other
        return output_stats

    def __iadd__(self, other: Self) -> Self:
        # Calculate new min and max
        self.min.data = torch.minimum(self.min, other.min)
        self.max.data = torch.maximum(self.max, other.max)

        # Calculate new mean and sum of squares
        n_new = self.n + other.n + 1e-8  # 1e-8 to avoid nans in entries that have zero counts
        self_w, other_w = self.n / n_new, other.n / n_new
        S2_w = self_w * other_w * n_new
        delta = self.mean - other.mean
        if self.cov_method == 'diag':
            delta_outer = torch.einsum('...i,...i->...i', delta, delta)
            delta_S2 = delta_outer * S2_w[..., None]
        else:
            delta_outer = torch.einsum('...i,...j->...ij', delta, delta)
            delta_S2 = delta_outer * S2_w[..., None, None]
        self.n += other.n
        self.mean *= self_w[..., None]
        self.mean += other_w[..., None] * other.mean
        self.S2 += other.S2 + delta_S2
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape},d={self.d})"

    @classmethod
    def test(cls):
        N = 1000  # Number of samples
        N_arrays = 100
        d = 80  # Dimension of random variable
        stats_array = cls((100,), d=d)

        # Check accumulation
        for _ in range(N_arrays):
            inds = torch.stack([torch.randint(0, stats_array.shape[i], (N,)) for i in range(len(stats_array.shape))],
                               dim=1)
            values = torch.randn((N, stats_array.d)) + 10
            stats_array += cls(shape=stats_array.shape, ind=inds, values=values)

        # Check the value of the accumulated covariance matrix
        stats_array = cls((10,), d=d, dtype=torch.double)
        values_array = []
        inds_array = []
        for _ in range(100):
            inds = torch.stack([torch.randint(0, stats_array.shape[i], (N,)) for i in range(len(stats_array.shape))],
                               dim=1)
            values = torch.randn((N, stats_array.d)).double() + 10
            stats_array += cls(shape=stats_array.shape, ind=inds, values=values)
            values_array.append(values)
            inds_array.append(inds)
        values_array = torch.cat(values_array)
        inds_array = torch.cat(inds_array).squeeze()

        C1 = stats_array.cov
        C2 = torch.stack([torch.cov(values_array[inds_array == i].T) for i in range(stats_array.shape[0])])
        is_C_matching = C1.isclose(C2).all()
        assert is_C_matching, "Some covariance matrices do not match."


if __name__ == '__main__':
    # Run various tests of the Statistics object
    Statistics.test()

    # Run various tests of the StatisticsArray object
    StatisticsArray.test()
