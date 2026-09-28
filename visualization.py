import copy
import pathlib
import csv
import shutil
import subprocess
import warnings

import torch
import numpy as np
import matplotlib
import tqdm
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import art3d
from PIL import Image

import config
from utils import round_to_sig, plot_mu_sigma, SingleLabelLogFormatter, rmsd_align, remove_centroid
from data.data_classes import Biomolecule, c_alpha_sidechain_class_name
from data.datasets import DenoisingTrajectoriesDataset, load_dataset, BiomoleculeDataset
from analysis.metrics import calculate_JS_dist_subsets, calculate_dihedral_sampling_rate
from chemistry.energy import BondEnergy

# Visualization configs
bond_color = '#818a91'  # Grayish color
bond_linewidth = 1.5
datasets_fig_colors = {'model': '#DC267F', 'train': '#FFB000', 'valid': '#9616db', 'test': '#648FFF'}
fig_width = 7.08


def shorten_labels(labels: list[str], max_size=20):
    first_half_size = max_size // 2 - 1
    second_half_size = max_size - 3 - first_half_size
    labels = copy.copy(labels)
    for i, label in enumerate(labels):
        # Apply the max size on all lines of the label
        label_lines = label.split('\n')

        # Shorten label to the first n chars and last n chars
        for j, l in enumerate(label_lines):
            if len(l) > max_size:
                l = l[:first_half_size] + '...' + l[-second_half_size:]
                label_lines[j] = l

        labels[i] = '\n'.join(label_lines)
    return labels


def plot_2D_map(bincounts: list[np.ndarray] | np.ndarray | torch.Tensor,
                binedges_x: np.ndarray | torch.Tensor | list,
                binedges_y: np.ndarray | torch.Tensor | list,
                binweights: np.ndarray | torch.Tensor | list = None,
                map_type='PMF', filename='', labels: list[str] = None, xlabel='', ylabel='', aspect='auto',
                return_dens=False):
    """
    Plots a single (or a list of) 2D maps.
    When given lists, each entry is plotted on a different matplotlib.axes using subplots().
    Args:
        bincounts: 2D array giving the counts of each bin
        binedges_x: 2D array giving the x-coordinate of the binedges. The given array (or list of arrays) must have at
                    least 1 dimension that is greater than 1 and this non-trivial dimension indicates which dimension of
                    the 2D counts array is mapped to the x-axis
        binedges_y: 2D array giving the y-coordinate of the binedges. The given array (or list of arrays) must have at
                    least 1 dimension that is greater than 1 and this non-trivial dimension indicates which dimension of
                    the 2D counts array is mapped to the y-axis
        binweights: 2D array giving the weights of each bin. Can be used to plot non-integer values.
        map_type: type of map to plot. Choices are: ['counts', 'frequency', 'probability', 'density', 'PMF']
        filename: filename where plot is saved.
        labels: labels of each map
        xlabel: x-label of each 2D map
        ylabel: y-label of each 2D map
        aspect: aspect ratio used to each axis.
        return_dens: if True, return densities instead of figure

    Returns:
        None or matplotlib.figure of the maps
    """

    if isinstance(bincounts, (np.ndarray, torch.Tensor)):
        bincounts = [bincounts]
    if isinstance(binweights, (np.ndarray, torch.Tensor)):
        binweights = [binweights]
    n_panels = len(bincounts)
    if not isinstance(binedges_x, list):
        binedges_x = n_panels * [binedges_x]
    if not isinstance(binedges_y, list):
        binedges_y = n_panels * [binedges_y]
    cmap_name = 'magma'

    # Calculate the 2D map intensity for each set
    intensities = []
    bins_range = np.zeros((n_panels, 2, 2))  # ind = (panel_ind, x/y ind, min/max ind)
    for i in range(n_panels):
        n_bins = bincounts[i].shape
        assert binedges_x[i].ndim == 2, f'binedges_x[{i}] must be given as a 2D array'
        assert binedges_y[i].ndim == 2, f'binedges_y[{i}] must be given as a 2D array'

        # Calculate the (x,y) coordinate of each bin center
        binedges_x_temp = binedges_x[i] * np.ones((n_bins[0] + 1, n_bins[1] + 1))
        bincenters_x = (binedges_x_temp[:-1, :-1] + binedges_x_temp[1:, 1:]) / 2
        binedges_y_temp = binedges_y[i] * np.ones((n_bins[0] + 1, n_bins[1] + 1))
        bincenters_y = (binedges_y_temp[:-1, :-1] + binedges_y_temp[1:, 1:]) / 2

        # bins_range_temp = np.array([[binedges_x[i][0], binedges_x[i][-1]], [binedges_y[i][0], binedges_y[i][-1]]])
        # bins_range_all[:, 0] = np.minimum(bins_range_all[:, 0], bins_range_temp[:, 0])
        # bins_range_all[:, 1] = np.maximum(bins_range_all[:, 1], bins_range_temp[:, 1])
        # bincenters_x = np.tile(((binedges_x[i][:-1] + binedges_x[i][1:]) / 2).reshape(-1, 1), (1, n_bins[1]))
        # bincenters_y = np.tile(((binedges_y[i][:-1] + binedges_y[i][1:]) / 2).reshape(1, -1), (n_bins[0], 1))
        bins_range[i, 0, 0] = binedges_x[i].min()
        bins_range[i, 0, 1] = binedges_x[i].max()
        bins_range[i, 1, 0] = binedges_y[i].min()
        bins_range[i, 1, 1] = binedges_y[i].max()

        # Calculate the map intensity depending on the type
        binweights_temp = bincounts[i].flatten()
        if binweights is not None:
            binweights_temp = binweights[i].flatten() * binweights_temp

        if map_type in ['counts', 'frequency', 'probability']:
            counts, _, _ = np.histogram2d(bincenters_x.flatten(), bincenters_y.flatten(), bins=n_bins,
                                          range=bins_range[i], weights=binweights_temp, density=False)
            if map_type == 'probability':
                intensity = counts / counts.sum()
            else:
                intensity = counts
        elif map_type in ['density', 'PMF']:
            density, _, _ = np.histogram2d(bincenters_x.flatten(), bincenters_y.flatten(), bins=n_bins,
                                           range=bins_range[i], weights=binweights_temp, density=True)
            if map_type == 'density':
                intensity = density
            else:
                with np.errstate(divide='ignore'):
                    intensity = -np.log(density)
        else:
            raise NotImplementedError(f"Map type {map_type!r} is not implemented.")
        intensities.append(intensity)

    # Calculate a bin range that is wide enough for all panels
    bins_range_all = np.zeros((2, 2))
    bins_range_all[0, 0] = bins_range[:, 0, 0].min()
    bins_range_all[0, 1] = bins_range[:, 0, 1].max()
    bins_range_all[1, 0] = bins_range[:, 1, 0].min()
    bins_range_all[1, 1] = bins_range[:, 1, 1].max()

    # Combine all intensities to calculate a suitable range
    intensities_all = np.concatenate([d.flatten() for d in intensities])
    intensities_all = intensities_all[~np.isinf(intensities_all)]
    intensity_min, intensity_max = intensities_all.min(), intensities_all.max()

    if map_type == 'PMF':
        # For PMF maps, rescale the energy to [0,1] or [0,max]
        if 0:
            intensity_range = intensity_max - intensity_min
        else:
            intensity_range = 1

        intensities = [(d - intensity_min) / intensity_range for d in intensities]
        intensity_min, intensity_max = 0.0, (intensity_max - intensity_min) / intensity_range
        bins_range_all = round_to_sig(bins_range_all, 2)
    else:
        # Invert the colormap
        cmap_name += '_r'

        # Set the minimum to the smallest non-zero value
        intensity_min = intensities_all[np.greater(intensities_all, 0.0)].min()

    # Determine the size of the figure given the number of rows and columns of the grid and the margins.
    n_cols = np.ceil(np.sqrt(n_panels)).astype(int)
    n_rows = np.ceil(n_panels / n_cols).astype(int)
    margin_bottom = 0.06
    margin_left, margin_right = 0.08, 0.1
    fig_ar = n_rows / n_cols * (1 - margin_left - margin_right) / (1 - margin_bottom)  # ar=h/w
    fig_size = 8 * np.array([1, fig_ar])

    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes_1D = axes.ravel()
    row_inds, col_inds = np.unravel_index(range(n_panels), (n_rows, n_cols))

    for i in range(n_panels):
        ax: plt.Axes = axes_1D[i]
        im = ax.imshow(intensities[i].T, origin='lower', extent=bins_range[i].flatten(), aspect=aspect, cmap=cmap_name,
                       vmin=intensity_min, vmax=intensity_max)
        im.cmap.set_over(color='w')
        im.cmap.set_under(color='w')

        # Set axes limits
        ax.set_xlim(*bins_range_all[0])
        ax.set_ylim(*bins_range_all[1])

        # ticks
        ax.tick_params(length=4, width=2, axis="both", direction="in")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, steps=np.arange(1, 10)))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, steps=np.arange(1, 10)))

        if not (row_inds[i] == n_rows - 1 and col_inds[i] == 0):
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        if labels is not None:
            ax.set_title(labels[i])

    # Add axis labels on the bottom left panel
    for ax, row_ind, col_ind in zip(axes_1D, row_inds, col_inds):
        if row_ind == n_rows - 1 and col_ind == 0:
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)

    # Tighten the figure before doing custom positioning
    fig.tight_layout()

    # Format colorbar
    fig.subplots_adjust(bottom=margin_bottom, right=1 - margin_right - 0.01)
    bottom_pos = min([a.get_position().bounds[1] for a in axes.ravel()])
    top_pos = max([a.get_position().bounds[1] + a.get_position().bounds[3] for a in axes.ravel()])
    cbar_width = 0.025  # normalized coords
    cbar_ax = fig.add_axes([1 - margin_right, bottom_pos, cbar_width, top_pos - bottom_pos])
    cbar = fig.colorbar(im, cax=cbar_ax, label=map_type)
    cbar.locator = matplotlib.ticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10])
    cbar.update_ticks()

    # Delete empty axes
    for ax in axes_1D[n_panels:]:
        ax.remove()

    # Save the plot
    if filename:
        fig.savefig(filename + '.pdf', dpi=500, pad_inches=None, bbox_inches='tight')
        plt.close(fig)
    elif return_dens:
        return intensities
    else:
        return fig


def plot_rama(phi_psi_counts: list[np.ndarray] | np.ndarray | torch.Tensor, filename='',
              labels: list[str] = None, return_dens=False):
    if isinstance(phi_psi_counts, (np.ndarray, torch.Tensor)):
        phi_psi_counts = [phi_psi_counts]
    n_bins = phi_psi_counts[0].shape
    bins_range = np.array(2 * [[-180.0, 180.0]])
    binedges_phi = np.linspace(bins_range[0, 0], bins_range[0, 1], n_bins[0] + 1).reshape(-1, 1)
    binedges_psi = np.linspace(bins_range[1, 0], bins_range[1, 1], n_bins[1] + 1).reshape(1, -1)

    return plot_2D_map(phi_psi_counts, binedges_x=binedges_phi, binedges_y=binedges_psi, filename=filename,
                       labels=labels, xlabel=r'$\phi$ ($^\circ$)', ylabel=r'$\psi$ ($^\circ$)', aspect='equal',
                       map_type='PMF', return_dens=return_dens)


def plot_UMAP(bincounts: list[np.ndarray] | np.ndarray | torch.Tensor,
              binedges_x: np.ndarray | torch.Tensor | list, binedges_y: np.ndarray | torch.Tensor | list,
              filename='', labels: list[str] = None, prop_name=''):
    if not prop_name:
        prop_name = 'UMAP'
    xlabel = f"{prop_name} 1"
    ylabel = f"{prop_name} 2"

    # x_axis spans the first UMAP dim
    if isinstance(binedges_x, list):
        binedges_x = [e.reshape(-1, 1) for e in binedges_x]
    else:
        binedges_x = binedges_x.reshape(-1, 1)

    # y-axis spans the second UMAP dim
    if isinstance(binedges_y, list):
        binedges_y = [e.reshape(1, -1) for e in binedges_y]
    else:
        binedges_y = binedges_y.reshape(1, -1)

    return plot_2D_map(bincounts, binedges_x=binedges_x, binedges_y=binedges_y, filename=filename, labels=labels,
                       xlabel=xlabel, ylabel=ylabel, map_type='PMF')


def plot_hbond_map(h_bond_counts_1D: list[np.ndarray] | np.ndarray | torch.Tensor, labels: list[str] = None, **kwargs):
    input_is_list = isinstance(h_bond_counts_1D, list)
    if not input_is_list:
        h_bond_counts_1D = [h_bond_counts_1D]

    # Reshape 1D frequencies into 2D
    h_bond_maps = []
    h_bond_counts = []
    binedges_x = []
    binedges_y = []
    for freq_1D in h_bond_counts_1D:
        d = int(np.sqrt(freq_1D.shape[0]))
        freq_2D = freq_1D.reshape(d, d)

        # According to calculate_hbond_counts(), the first dim corresponds to the donor and 2nd dim the acceptor.
        # Define the binedges arrays to reflect this
        binedges_x_temp = np.arange(d + 1).reshape(1, -1) - 0.5  # x-axis spans the acceptor ind (dim=1)
        binedges_y_temp = np.arange(d + 1).reshape(-1, 1) - 0.5  # y-axis spans the donor ind (dim=0)

        h_bond_maps.append(freq_2D)
        h_bond_counts.append(np.ones_like(freq_2D))
        binedges_x.append(binedges_x_temp)
        binedges_y.append(binedges_y_temp)

    if not input_is_list:
        h_bond_maps = h_bond_maps[0]
        binedges_x = binedges_x[0]
        binedges_y = binedges_y[0]

    xlabel = f"Acceptor residue"
    ylabel = f"Donor residue"

    return plot_2D_map(h_bond_counts, binedges_x=binedges_x, binedges_y=binedges_y, binweights=h_bond_maps,
                       labels=labels, xlabel=xlabel, ylabel=ylabel, map_type='frequency', **kwargs)


def plot_JS_dist_subsets(datasets: BiomoleculeDataset | list[BiomoleculeDataset], properties_name=None,
                         filepath: str | pathlib.Path = None):
    if not isinstance(datasets, list):
        datasets = [datasets]
    n_datasets = len(datasets)

    if filepath is None:
        if n_datasets == 1:
            filepath = config.plots_dir / f"JS_dist_subsets_{datasets[0].name}.pdf"
        else:
            filepath = config.plots_dir / f"JS_dist_subsets.pdf"
    filepath = pathlib.Path(filepath).with_suffix('.pdf')
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Assemble the JS dist data
    JS_dist_dicts = []
    for dataset in datasets:
        JS_dist_dict = calculate_JS_dist_subsets(dataset, properties_name=properties_name)
        JS_dist_dicts.append(JS_dist_dict)
    n_props = len(JS_dist_dicts[0]) - 1
    fig_height = fig_width * n_props / n_datasets
    fig, axes = plt.subplots(n_props, n_datasets, figsize=(fig_width, fig_height), sharey='row', sharex='all')
    axes = axes.reshape((n_props, n_datasets))
    for j, (JS_dist_dict, dataset) in enumerate(zip(JS_dist_dicts, datasets)):
        n_samples = JS_dist_dict.pop('n_samples')
        bottom_ax, top_ax = axes[-1, j], axes[0, j]
        top_ax.set_title(dataset.name)
        bottom_ax.set_xlabel('N_samples')

        for i, (prop_name, JS_dist_values) in enumerate(JS_dist_dict.items()):
            ax = axes[i, j]
            fig.sca(ax)
            JS_dist_avg = JS_dist_values.mean(-1)
            JS_dist_std = JS_dist_values.std(-1)
            plot_mu_sigma(JS_dist_avg, JS_dist_std, x=n_samples)

            # Format axis ticks and tick labels
            ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            x_locator = MaxNLocator(nbins=3, steps=[1, 2, 5, 10])
            ax.xaxis.set_major_locator(x_locator)
            ax.set_xlim(-10, n_samples[-1])

            ax.set_yscale('log', base=10)
            ax.yaxis.set_minor_formatter(SingleLabelLogFormatter())
            ax.yaxis.set_major_formatter(SingleLabelLogFormatter())

            if j == 0:
                plt.ylabel(f'{prop_name}\nJS dist.')

    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    fig.savefig(filepath, dpi=500, pad_inches=None)
    plt.close(fig)


def plot_dihedral_angle_sampling_rate(datasets: BiomoleculeDataset | list[BiomoleculeDataset],
                                      filepath: str | pathlib.Path = None):
    if not isinstance(datasets, list):
        datasets = [datasets]
    n_datasets = len(datasets)

    if filepath is None:
        if n_datasets == 1:
            filepath = config.plots_dir / f"dih_angle_samp_rate_{datasets[0].name}.pdf"
        else:
            filepath = config.plots_dir / f"dih_angle_samp_rate.pdf"
    filepath = pathlib.Path(filepath).with_suffix('.pdf')
    filepath.parent.mkdir(parents=True, exist_ok=True)

    n_props = 1
    fig_height = fig_width * n_props / n_datasets
    fig, axes = plt.subplots(n_props, n_datasets, figsize=(fig_width, fig_height))
    axes = axes.reshape((n_props, n_datasets))
    for j, dataset in enumerate(datasets):
        ax = axes[0, j]
        ax.set_title(dataset.name)
        fig.sca(ax)

        # Plot curve for each split
        for i, dataset_split in enumerate(dataset.load_splits()):
            dataset_split: BiomoleculeDataset
            n_samples, n_states = calculate_dihedral_sampling_rate(dataset_split)

            n_states_avg = n_states.mean(-1)
            n_states_std = n_states.std(-1)
            plot_color = datasets_fig_colors[dataset_split.split_ID[1]]
            plot_mu_sigma(n_states_avg, n_states_std, x=n_samples, color=plot_color, label=dataset_split.split_ID[1])

        # Format axis ticks and tick labels
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        ax.set_ylim(0, None)
        ax.set_xlim(0, None)

        if j == 0:
            ax.set_xlabel('n_samples')
            ax.set_ylabel('Unique dihedral states')
            ax.legend()

    plt.tight_layout(pad=0.1)
    fig.savefig(filepath, dpi=500, pad_inches=None)
    plt.close(fig)


def import_cpk_colors():
    cpk_colors_filepath = config.root_dir / 'chemistry' / 'cpk_colors.csv'
    with open(cpk_colors_filepath) as csv_file:
        spamreader = csv.reader(csv_file)
        n_header_lines = 3
        for _ in range(n_header_lines):
            next(spamreader)
        cpk_colors = {}
        for line in spamreader:
            atom_n, atom_syms, RBG_str, Hex_str = line
            atom_syms = atom_syms.split(', ')
            for atom_sym in atom_syms:
                cpk_colors[atom_sym] = {}
                cpk_colors[atom_sym]['RGB'] = eval(RBG_str)
                cpk_colors[atom_sym]['HEX'] = Hex_str
    return cpk_colors


def generate_CS_pseudobonds_ind(mol: Biomolecule):
    # Add bonds between c_alpha
    is_atom_CA = mol.elements_is_CA
    CA_ind = torch.nonzero(is_atom_CA)[:, 0]
    bonds_back_ind = torch.stack([CA_ind[:-1], CA_ind[1:]], dim=-1)

    # Add bonds between c_alpha and sidechain rep
    sidechain_rep_ind = torch.nonzero(~is_atom_CA)[:, 0]
    CA_res_ind = mol.elements_resind[CA_ind]
    sidechain_rep_res_ind = mol.elements_resind[sidechain_rep_ind]
    CA_side_pair_ind = torch.nonzero(CA_res_ind[:, None] == sidechain_rep_res_ind[None, :])
    bonds_side_ind = torch.stack([CA_ind[CA_side_pair_ind[:, 0]], sidechain_rep_ind[CA_side_pair_ind[:, 1]]],
                                 -1)
    bonds_ind = torch.cat([bonds_back_ind, bonds_side_ind])
    return bonds_ind


def determine_atom_colors(mol: Biomolecule, coloring_mode: str = 'type'):
    """
    Determines the colors of displayed atoms, which depends on the type of the molecule and the coloring mode

    Args:
        mol: molecule whose elements are colored
        coloring_mode: 'type' (CPK / CA-sidechain colouring) or 'residue' (color ramp along the chain)

    Returns:
        list or array of colours, one per element
    """
    if coloring_mode == 'type':
        if mol.element_class == c_alpha_sidechain_class_name:
            return ['#000000' if is_CA else '#FF0000' for is_CA in mol.elements_is_CA]
        return ['#' + cpk_colors[name[0]]['HEX'] for name in mol.elements_name]
    elif coloring_mode == 'residue':
        cmap = matplotlib.colormaps['jet']
        return cmap(mol.elements_resind / mol.elements_resind.max())
    raise NotImplementedError(f"coloring_mode={coloring_mode!r} is not implemented")


def plot_molecule(mol: Biomolecule, ax: plt.Axes = None, adjacency='bonds', coloring_mode='type'):
    init_fig = ax is None
    if init_fig:
        px_per_inch = 1 / plt.rcParams['figure.dpi']
        fig_size = tuple([px_per_inch * x for x in [1200, 800]])
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(projection='3d')
    else:
        fig = ax.get_figure()

    # Add basic info in title
    if init_fig:
        ax.set_title(f'ID={mol.ID}' + f', model_ID={mol.model_ID}' if mol.model_ID else '')

    # Plot atom positions
    atoms_center = mol.elements_position
    atoms_color = determine_atom_colors(mol, coloring_mode)
    ax.scatter(*atoms_center.unbind(-1), c=atoms_color)

    # Plot bonds
    if adjacency == 'bonds':
        if mol.element_class == c_alpha_sidechain_class_name:
            bonds_ind = generate_CS_pseudobonds_ind(mol)
        else:
            bond_mod = BondEnergy(mol.top_file, element_class=mol.element_class)
            bonds_ind = bond_mod.bonds_atom_ind

        for bond_ind in bonds_ind:
            line_coords = mol.elements_position[bond_ind]
            ax.plot(*line_coords.unbind(-1), color=bond_color, linewidth=bond_linewidth)
    else:
        raise NotImplementedError(f"{adjacency!r} is not implemented")

    if init_fig:
        fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9)
        ax.set_aspect('equal')
        ax.set_box_aspect([10, 10, 10])

        ax.set_xlabel(f'x ({mol.length_units})')
        ax.set_ylabel(f'y ({mol.length_units})')
        ax.set_zlabel(f'z ({mol.length_units})')

    return fig


def calculate_content_bbox(frames: list[np.ndarray], bg_threshold: int = 250, pad: int = 2):
    """
    Calculates the bounding box of all non-background (non-white) pixels across a set of RGB frames.
    Args:
        frames: list of (H,W,3) uint8 RGB frames
        bg_threshold: a pixel counts as content if any channel is below this value
        pad: extra pixels kept around the content on every side

    Returns:
        ((x0, y0, w, h), clipped) where w and h are forced even as required by yuv420p H.264, and
        `clipped` flags that content reached the canvas edge, i.e. the figure itself cut something off
    """
    H, W = frames[0].shape[:2]
    x0, y0, x1, y1 = W, H, -1, -1
    for frame in frames:
        ink = (frame[..., :3] < bg_threshold).any(-1)
        rows = np.flatnonzero(ink.any(1))
        cols = np.flatnonzero(ink.any(0))
        if rows.size == 0:
            continue
        y0, y1 = min(y0, int(rows[0])), max(y1, int(rows[-1]))
        x0, x1 = min(x0, int(cols[0])), max(x1, int(cols[-1]))

    if x1 < 0:  # every sampled frame was blank
        return (0, 0, W - W % 2, H - H % 2), False

    clipped = x0 == 0 or y0 == 0 or x1 == W - 1 or y1 == H - 1
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(W - 1, x1 + pad), min(H - 1, y1 + pad)
    return (x0, y0, (x1 - x0 + 1) // 2 * 2, (y1 - y0 + 1) // 2 * 2), clipped


def write_mp4(frames, filepath, fps, crf=18):
    """
    Encodes RGB frames to H.264 by piping raw video straight into ffmpeg.

    Args:
        frames: iterable of (H,W,3) uint8 RGB frames, all the same shape
        filepath: destination .mp4
        fps: frames per second
        crf: x264 quality (lower is better; 18 is visually lossless)
    """
    frames = iter(frames)
    first = next(frames)
    h, w = first.shape[:2]
    ffmpeg_exe = matplotlib.rcParams.get('animation.ffmpeg_path') or 'ffmpeg'
    if not (pathlib.Path(ffmpeg_exe).exists() or shutil.which(ffmpeg_exe)):
        raise RuntimeError(f"ffmpeg not found at {ffmpeg_exe!r}; set rcParams['animation.ffmpeg_path'].")

    cmd = [ffmpeg_exe, '-y', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
           '-an', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', str(crf), '-preset', 'medium',
           str(filepath)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        proc.stdin.write(np.ascontiguousarray(first).tobytes())
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr.read().decode(errors='replace')}")


def write_gif(frames, filepath, fps, width=400, n_colors=32, diff_threshold=12, key_colors=None):
    """
    Writes an optimised looping GIF.

    A single global palette is built from the whole clip and applied without dithering, so flat regions stay flat.
    Every pixel that is within `diff_threshold` of what is already on the canvas is made transparent,
    which turns the static background into longruns that LZW compresses well.
    A threshold of 12/255 is visually indistinguishable from exact
    differencing but roughly a third smaller.

    Args:
        frames: list of (H,W,3) uint8 RGB frames
        filepath: destination .gif
        fps: frames per second (GIF delays are integer centiseconds, so use a divisor of 100)
        width: output width in pixels; height follows the aspect ratio
        n_colors: palette size
        diff_threshold: max per-channel deviation tolerated before a pixel is redrawn
        key_colors: colours that must appear in the palette exactly, whatever their pixel count.
                    A median-cut palette is allocated by pixel population, and these frames are
                    ~90% near-white pane, so without this every slot goes to white gradations and
                    the few hundred pixels of coloured atoms collapse onto a grey.
    """
    h, w = frames[0].shape[:2]
    size = (width, max(1, round(h * width / w)))
    small = [np.asarray(Image.fromarray(f).resize(size, Image.LANCZOS)) for f in frames]

    # One global palette, derived from an even sample of the clip. Leave at least one index free so
    # there is always a slot available for transparency.
    n_colors = int(np.clip(n_colors, 2, 255))

    # Reserve slots for the colours the figure actually draws with, before quantising anything.
    # Median cut distributes slots by how many pixels have each colour, and the atoms occupy a tiny
    # fraction of a mostly-white frame, so they lose every slot to the background otherwise.
    key_rgb = []
    if key_colors is not None:
        key_rgb = list(dict.fromkeys(
            tuple(int(round(255 * v)) for v in matplotlib.colors.to_rgb(c)) for c in key_colors
        ))[:max(0, n_colors // 2)]

    sample = np.concatenate(small[::max(1, len(small) // 24)], axis=0)
    pal_img = Image.fromarray(sample).quantize(colors=max(2, n_colors - len(key_rgb)),
                                               method=Image.MEDIANCUT)

    # Recover the colours the quantiser actually assigned. Pillow is inconsistent here across
    # versions (python-pillow/Pillow#6046): getpalette() may return only the used colours or the
    # whole 768-value table padded with (0,0,0), and getcolors() may report indices well outside the
    # requested colour count. So: keep only in-range indices, look each one up, and de-duplicate.
    # Assuming n_colors entries and slicing is what produced 'index N is out of bounds' here.
    palette_flat = list(pal_img.getpalette() or [])
    n_entries = len(palette_flat) // 3
    used_ind = sorted({ind for _, ind in (pal_img.getcolors(maxcolors=1 << 24) or [])
                       if 0 <= ind < n_entries})
    colours = list(dict.fromkeys(tuple(palette_flat[3 * k:3 * k + 3]) for k in used_ind))
    colours = list(dict.fromkeys(key_rgb + colours))[:255]
    palette = np.array(colours or [(0, 0, 0), (255, 255, 255)], dtype=np.int16)
    n_pal = len(palette)

    # Map the frames against a palette image whose 256 slots are the real colours tiled, rather than
    # against pal_img directly. On the Pillow versions that pad the palette with (0,0,0), quantising
    # against pal_img sends every pure black pixel -- the title text, and the black CA atoms of the
    # coarse-grained representation -- to a padding slot at index >= n_pal, which is not a real
    # colour. Tiling guarantees every slot holds a genuine colour, and % n_pal folds it back.
    pal_ref = Image.new('P', (1, 1))
    pal_ref.putpalette([int(v) for i in range(256) for v in palette[i % n_pal]])
    indexed = [np.asarray(Image.fromarray(f).quantize(palette=pal_ref, dither=Image.Dither.NONE)) % n_pal
               for f in small]

    transparent = n_pal
    canvas = palette[indexed[0]]
    out = [indexed[0]]
    for idx in indexed[1:]:
        quantized = palette[idx]
        redraw = np.abs(quantized - canvas).max(-1) > diff_threshold
        out.append(np.where(redraw, idx, transparent).astype(np.uint8))
        canvas = np.where(redraw[..., None], quantized, canvas)

    full_palette = sum(palette.tolist() + [[0, 0, 0]] * (256 - n_pal), [])
    images = []
    for frame in out:
        img = Image.fromarray(frame.astype(np.uint8), mode='P')
        img.putpalette(full_palette)
        images.append(img)
    images[0].save(filepath, save_all=True, append_images=images[1:], loop=0,
                   duration=round(1000 / fps), transparency=transparent, disposal=1, optimize=False)


def make_denoising_movie(dataset: DenoisingTrajectoriesDataset | str, movie_length=10, fps=25,
                         mol_ind: list[int] | int = None, atom_color='type', align=True,
                         limits_margin=0.05, figsize_px=(1200, 800), dpi=100, crop=True,
                         subplot_margins=None,
                         save_mp4=True, save_gif=True, crf=18,
                         gif_width=400, gif_colors=32, gif_diff_threshold=12):
    """
    Makes movies of denoising trajectories.

    The trajectory is evenly subsampled to `movie_length * fps` frames, each frame is RMSD-aligned to
    the final denoised structure so the eye follows the folding rather than the global tumbling, and
    the axis limits are fixed to that final structure. Frames are rendered once and then encoded to
    .mp4 and/or .gif from the same buffer, cropped to the tightest box that contains drawn content in
    every frame.

    Args:
        dataset: dataset of denoising trajectories, or its name
        movie_length: length of the movies in seconds
        fps: playback frame rate. For GIF output prefer a divisor of 100 (10, 20, 25, 50) so the
             frame delay is an exact whole number of centiseconds
        mol_ind: index or indices of the molecules to render. Defaults to all of them
        atom_color: 'type' or 'residue', as in plot_molecule()
        align: RMSD-align every frame to the final structure (utils.rmsd_align, no reflections).
               Pass 'centroid' to only remove the per-frame centroid, or False to render the
               trajectory exactly as sampled
        limits_margin: fractional padding added around the final structure when fixing the axis limits
        figsize_px: size of the render canvas in pixels, before cropping
        dpi: figure dpi
        subplot_margins: dict passed to Figure.subplots_adjust. The default leaves room for the tick
                         labels below the 3D box, which the previous implementation cut off by
                         placing the axes flush against the bottom of the canvas. Whatever margin is
                         not used gets cropped away anyway, so being generous only costs render time
        crop: crop away the figure margin that is blank in every frame
        save_mp4: write a .mp4
        save_gif: also write an optimised looping .gif next to the .mp4
        crf: x264 quality for the .mp4; lower is better, 18 is visually lossless
        gif_width: width of the .gif in pixels
        gif_colors: .gif palette size
        gif_diff_threshold: per-channel tolerance (0-255) used by the .gif frame differencing

    Returns:
        list of pathlib.Path of the files that were written
    """
    if isinstance(dataset, str):
        dataset = load_dataset(dataset, element_class=config.dataset_element_class)
    denoising_movie_dir = pathlib.Path(config.plots_dir, 'denoising_movies', dataset.name)
    denoising_movie_dir.mkdir(parents=True, exist_ok=True)

    if not (save_mp4 or save_gif):
        raise ValueError('Nothing to write: both save_mp4 and save_gif are False.')

    # Get info about the dataset
    mol_ex = dataset[0]

    # Get the position scale that was used during training and sampling
    pos_scale = dataset.configs['feats_scale']['pos']

    # Load the noise schedule
    # Multiply by position scale since the noise schedule is applied to the scaled positions.
    noise_sch_t = dataset.noise_sch_t * pos_scale

    # Determine the set of bonded nodes (true bonds (heavy-atom structure) or pseudo bonds (coarse-grained structure))
    if mol_ex.element_class == c_alpha_sidechain_class_name:
        bonds_ind = generate_CS_pseudobonds_ind(mol_ex)
    else:
        bond_mod = BondEnergy(mol_ex.top_file, element_class=mol_ex.element_class)
        bonds_ind = bond_mod.bonds_atom_ind

    mol_ind_1D = dataset.mol_time_to_data_ind
    n_mols, n_frames = mol_ind_1D.shape

    # Evenly subsample the trajectory to the requested playback length, always keeping the first and
    # last frames. The original implementation played back every frame at interval=10 ms, i.e. 100 fps.
    n_movie_frames = max(2, min(n_frames, round(movie_length * fps)))
    frame_ind = np.unique(np.linspace(0, n_frames - 1, n_movie_frames).round().astype(int))

    if mol_ind is None:
        mol_ind = range(n_mols)
    elif isinstance(mol_ind, int):
        mol_ind = [mol_ind]

    written = []
    for i in mol_ind:
        desc = f'Making {dataset.name} denoising movie {i + 1}/{n_mols}'

        # Load the whole trajectory once, rather than indexing the dataset inside the draw callback
        mol_i = dataset[mol_ind_1D[i, frame_ind[-1]]]
        positions = torch.stack([dataset[k].elements_position for k in mol_ind_1D[i, frame_ind]])
        positions = positions.detach().cpu()  # rendering is CPU-only; the dataset may sit on a GPU

        # Align every frame to the final (denoised) structure. Both utils helpers are batched over
        # leading dimensions, so the whole (n_movie_frames, n_elements, 3) trajectory goes in one call.
        # rmsd_align adds the reference centroid back afterwards, and the reference is centred here,
        # so the aligned trajectory comes back centred on the origin.
        if align == 'centroid':
            positions = remove_centroid(positions)
        elif align:
            positions = rmsd_align(positions=positions, positions_ref=remove_centroid(positions[-1]),
                                   reflect=False)

        # Fix the axis limits to the final structure, as a cube so the aspect ratio is honest
        final_pos = positions[-1]
        centre = (final_pos.amax(0) + final_pos.amin(0)) / 2
        half_extent = float((final_pos.amax(0) - final_pos.amin(0)).max()) / 2 * (1 + limits_margin)
        limits = [(float(c) - half_extent, float(c) + half_extent) for c in centre]

        # Build the figure without pyplot, so nothing accumulates in the global figure registry
        fig = Figure(figsize=(figsize_px[0] / dpi, figsize_px[1] / dpi), dpi=dpi)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(projection='3d')
        fig.subplots_adjust(**(subplot_margins or dict(left=0.02, right=0.98, bottom=0.06, top=0.95)))

        atoms_colors = determine_atom_colors(mol_i, atom_color)
        atoms_h = ax.scatter(*positions[-1].unbind(-1), c=atoms_colors)

        # One Line3DCollection for every bond, instead of one Line3D artist per bond. With a few
        # hundred bonds this is the difference between redrawing hundreds of artists per frame and one.
        bonds_h = art3d.Line3DCollection(positions[-1][bonds_ind].numpy(), colors=bond_color,
                                         linewidths=bond_linewidth)
        ax.add_collection3d(bonds_h)

        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_zlim(*limits[2])
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel(f'x ({mol_i.length_units})')
        ax.set_ylabel(f'y ({mol_i.length_units})')
        ax.set_zlabel(f'z ({mol_i.length_units})')
        ax.set_title('')

        def draw_frame(t):
            """Renders movie frame t and returns it as an (H,W,3) uint8 array."""
            pos = positions[t]
            atoms_h._offsets3d = tuple(pos.numpy().T)
            bonds_h.set_segments(pos[bonds_ind].numpy())
            # Reuse the existing Text artist rather than building a new title every frame
            ax.title.set_text(rf"Mol ID={i + 1}, Frame={frame_ind[t] + 1}, "
                              rf"$\sigma$={float(noise_sch_t[frame_ind[t]]): >#5.3g} nm")
            fig.canvas.draw()
            return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

        # Pass 1: find the crop box from an even sample of frames, so the box holds for the whole clip
        if crop:
            sample_t = np.unique(np.linspace(0, len(frame_ind) - 1, 16).round().astype(int))
            (x0, y0, cw, ch), clipped = calculate_content_bbox([draw_frame(t) for t in sample_t])
            if clipped:
                warnings.warn(f"Molecule {i + 1}: drawn content reaches the edge of the {figsize_px[0]}x"
                              f"{figsize_px[1]} canvas, so part of it is cut off. Increase figsize_px or "
                              f"widen subplot_margins.", stacklevel=2)
        else:
            x0, y0 = 0, 0
            cw, ch = figsize_px[0] // 2 * 2, figsize_px[1] // 2 * 2

        # Pass 2: render every frame, cropping as we go
        frame_iter = tqdm.tqdm(range(len(frame_ind)), desc)
        frames = [draw_frame(t)[y0:y0 + ch, x0:x0 + cw] for t in frame_iter]
        frame_iter.close()

        if save_mp4:
            mp4_filepath = denoising_movie_dir / f"{dataset.name}_{i + 1}.mp4"
            write_mp4(frames, mp4_filepath, fps=fps, crf=crf)
            written.append(mp4_filepath)
        if save_gif:
            gif_filepath = denoising_movie_dir / f"{dataset.name}_{i + 1}.gif"
            # Hand the writer the exact colours this figure draws with, so the atoms survive
            # quantisation: the atom colours themselves, the bond grey, and the black of the
            # title and tick labels against the white background.
            write_gif(frames, gif_filepath, fps=fps, width=gif_width, n_colors=gif_colors,
                      diff_threshold=gif_diff_threshold,
                      key_colors=list(atoms_colors) + [bond_color, '#000000', '#FFFFFF'])
            written.append(gif_filepath)

    return written


cpk_colors = import_cpk_colors()

if __name__ == '__main__':
    dataset = load_dataset()

    # Test plot_molecule
    # plot_molecule(dataset[0], atom_color='residue')

    # Test make_denoising_movie
    dataset_traj = load_dataset('Traj_Diff_HA1_nup98_12_1_H1', force_download=False)
    dataset_traj.to_xtc()

    # adjust_kwargs = dict(left=-.12, right=1.1, bottom=-0.2, top=1.2, wspace=-0.65)
    # plot_denoising_mosaic(dataset_traj, figsize=[7, 2])
    # make_denoising_movie(dataset_traj)

    # # Test plotting of models parameter distributions
    # import models
    #
    # config.import_configs('config_HA_193')
    # config.dataset = 'nup98_12'
    # model = models.load()
    # tensorboard_dir = os.path.join(config.root_dir, 'tensorboard', f"{config.ID}")
    # model.plot_params_dist(tensorboard_dir=tensorboard_dir)
