import copy
import functools
import pathlib
import os
import json
from collections import defaultdict
from typing import Iterable

import matplotlib
import matplotlib.ticker
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, FuncFormatter
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.transforms as mtransforms

import config
from analysis.comparisons import compare_prop_dist
from analysis.metrics import Biomolecule_UMAP_prop_names, defined_props
from analysis.metrics import calculate_JS_dist_subsets
from analysis.metrics import calculate_dihedral_sampling_rate
from data.data_classes import heavy_atoms_class_name, all_atoms_class_name
from data.datasets import GeneratedDataset, load_dataset, BiomoleculeDataset, main_datasets
from data.forcefields import amber14sb_OL15
from utils import copy_ax, AdaptivePrecisionEdgeLocator, draw_ax_bbox
from visualization import datasets_fig_colors

maplike_metrics = ['h_bond_map', 'backbone_dihedrals'] + Biomolecule_UMAP_prop_names  # Metrics that are 2D dist.
equal_aspect_metrics = ['h_bond_map', 'backbone_dihedrals']
fine_metrics = ['bond_length_Z', 'V_bonds', 'bond_angle_Z', 'V_angles', 'V_dih_prop+improp', 'V_dih_cmap',
                'backbone_dihedrals']
coarse_metrics = ['h_bond_map', 'R_g', 'V_LJ_SR', 'V_coulomb_SR', 'V_tot']
sampling_metrics = ['performance', 'JSD_bond_lengths', 'JSD_bond_angles', 'JSD_rama', 'JSD_Rg', 'dihedral_states']
downsampling_factor = {'V_coulomb_SR': 2, 'V_tot': 2}  # Downsample bins of given metrics from default binwidth

SI_figs_dir = pathlib.Path(config.plots_dir, 'figures_SI')
SI_figs_dir.mkdir(parents=True, exist_ok=True)

main_figs_dir = pathlib.Path(config.plots_dir, 'figures_main')
main_figs_dir.mkdir(parents=True, exist_ok=True)

# Dataset specifications
datasets_fig_label = {'nup98_12': 'Nup98-12 (c36m)',
                      'nup98C_12': 'Nup98-12 capped\n(c36m)',
                      'nup98AC_12': 'Nup98-12 capped\n(ff14sb)',
                      'nup98_24': 'Nup98-24 (c36m)',
                      'AAQ': r'$(\text{AAQAA})_3$ (ff14sb)',
                      'CLN': 'CLN025 (c22*)',
                      'RS': 'RS (c36m)'}
datasets_fig_label['AAQAA3'] = datasets_fig_label['AAQ']
datasets_fig_label['CLN025'] = datasets_fig_label['CLN']
main_datasets_shortname = {v: k for k, v in main_datasets.items()}
main_configs_ID = {'CS': 'CS1', 'HA': 'HA1'}

# Figure dimensions
main_fig_widths = {'1col': 3.54, '2col': 7.08}
SI_fig_width = 6.5  # inches
main_fig_width = 6.5  # inches
fig_height_max = 7.28
fig_dpi = 1000

# Formatter for epoch ticks
epoch_tick_formatter = matplotlib.ticker.FuncFormatter(lambda x, pos: f"{x / 1000:.0f}{'k' if x > 0 else ''}")


def thousands_formatter(x, pos):
    return f'{int(x / 1000)}k' if x >= 1000 else str(int(x))


# Dictionary of names of metrics as they will appear on the plot
metrics_label = {'R_g': '$R_g$\n[nm]',
                 'bond_length_Z': 'Bond length\nZ-scores',
                 'bond_angle_Z': 'Bond angle\nZ-scores',
                 'V_bonds': 'Bond length $V$\n[kJ/mol]',
                 'V_angles': 'Bond angle $V$\n[kJ/mol]',
                 'V_dih_prop+improp': 'Dihedral $V$\n[kJ/mol]',
                 'V_LJ_SR': 'L-J $V$\n[kJ/mol]',
                 'V_coulomb_SR': 'Coulomb $V$\n[kJ/mol]',
                 'V_tot': 'Total $V$\n[kJ/mol]',
                 'V_coulomb_recip': 'Coulomb $V$\n[kJ/mol]',
                 'V_dih_cmap': 'CMAP $V$\n[kJ/mol]',
                 'rmsd_min_train': 'RMSD-train min.\n[nm]',
                 'rmsd_min_2RVD': 'RMSD-2RVD min.\n[nm]',
                 'elements_position_UMAP': 'Pos. UMAP',
                 'dihedral_angle_UMAP': 'Dih. UMAP',
                 'backbone_dihedrals': 'Ramachandran PMF',
                 'h_bond_map': 'H-bond map',
                 'Jeffreys_Rg': '',
                 'Jeffreys_rama': ''}


def load_paper_mpl_configs():
    """
    Defines default Matplotlib parameters for publication figures
    Returns:
        None
    """
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial']
    matplotlib.rcParams['axes.titlesize'] = 7
    matplotlib.rcParams['axes.labelsize'] = 6
    matplotlib.rcParams['xtick.labelsize'] = 5
    matplotlib.rcParams['ytick.labelsize'] = 5
    matplotlib.rcParams['xtick.major.size'] = 3
    matplotlib.rcParams['ytick.major.size'] = 3
    matplotlib.rcParams['xtick.minor.size'] = 1.5
    matplotlib.rcParams['ytick.minor.size'] = 1.5
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    matplotlib.rcParams['legend.handlelength'] = 0.5
    matplotlib.rcParams['legend.handletextpad'] = 0.4
    matplotlib.rcParams['legend.borderpad'] = 0.0
    matplotlib.rcParams['legend.borderaxespad'] = 0.2
    matplotlib.rcParams['legend.labelspacing'] = 0.2
    matplotlib.rcParams['legend.fontsize'] = 6
    matplotlib.rcParams['legend.fancybox'] = False
    matplotlib.rcParams['legend.frameon'] = False
    matplotlib.rcParams['lines.linewidth'] = 0.75
    matplotlib.rcParams['pdf.fonttype'] = 42


def compute_grid_geometry(fig_width, max_fig_height, n_rows: int, n_cols: int, row_h=0.5, equal_aspect_row_ind=[],
                          img_shape=[1.0, 1.0]):
    """
    Calculates the exact figure height and GridSpec height ratios to perfectly
    fill equal-aspect panels, allocating the remainder to other rows.
    """
    # 1. Physical width of a single column
    col_width = fig_width / n_cols

    # 2. Required physical height for an equal-aspect subplot to avoid padding
    # img_shape expected as (rows/height, cols/width)
    img_aspect_ratio = img_shape[0] / img_shape[1]
    h_equal = col_width * img_aspect_ratio
    h_nonequal = row_h

    # Calculate total height of equal-aspect rows and colorbar rows
    n_equal_rows = len(equal_aspect_row_ind)
    n_nonequal_rows = n_rows - len(equal_aspect_row_ind)
    total_equal_height = h_equal * n_equal_rows + h_nonequal * n_nonequal_rows

    # Check if remaining height is positive
    remaining_height = max_fig_height - total_equal_height
    if remaining_height <= 0:
        raise ValueError(
            f"max. fig. height is too small to accommodate {n_equal_rows} equal-aspect rows "
            f"and {n_nonequal_rows} rows of height {h_nonequal}.")

    # Determine the absolute height of each row
    absolute_heights = []
    for r in range(n_rows):
        if r in equal_aspect_row_ind:
            absolute_heights.append(h_equal)
        else:
            absolute_heights.append(h_nonequal)

    # Calculate total figure height and row height ratios
    fig_height = sum(absolute_heights)
    height_ratios = [h / fig_height for h in absolute_heights]

    return fig_height, height_ratios


def tightbbox_wrapper(ax: plt.Axes, extra_artists=None):
    if extra_artists is None:
        extra_artists = []
    if not isinstance(extra_artists, list):
        extra_artists = [extra_artists]
    func = ax.get_tightbbox

    @functools.wraps(func)
    def wrapper(*args, bbox_extra_artists=None, **kwargs):
        if bbox_extra_artists is None:
            bbox_extra_artists = ax.get_default_bbox_extra_artists()
        bbox_extra_artists = bbox_extra_artists + extra_artists
        return func(*args, bbox_extra_artists=bbox_extra_artists, **kwargs)

    return wrapper


def optimize_gridspec(fig: plt.Figure, axes: Iterable[plt.Axes], row_unequal_h: float, inter_pad: str | float = None,
                      pad_buffer=0.02, fig_height_max: float = None):
    """
    Resizes figure height and calculates new GridSpec parameters to tightly fit axes within the figure
    Args:
        fig: figure
        axes: grid of axes
        inter_pad: amount of padding between subplots. If string, interpreted as percentage of default
        row_unequal_h: target height (inches) of rows that are stretchable (not 'equal' aspect ratio)
        pad_buffer: buffer (inches) added to calculated pads.
        fig_height_max: max figure height (inches) allowable. Raise errors if calculated height is greater than size.

    Returns:
        None
    """
    # Disable active layout engine
    fig.set_layout_engine(None)

    axs_2d = np.atleast_2d(axes)
    n_rows, n_cols = axs_2d.shape

    # Align labels so positioning across rows/cols is consistent
    fig.align_ylabels(axs_2d)
    fig.align_xlabels(axs_2d)

    # Render artist extents
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi

    # Measure outer edge margins (includes outer y-axis labels, x-axis labels, titles)
    margin_left = 0.001
    margin_right = 0.001
    margin_bottom = 0.001
    margin_top = 0.001

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axs_2d[r, c]
            ax_bbox = ax.get_window_extent(renderer)
            tb = ax.get_tightbbox(renderer)

            if tb is None:
                continue

            left_ext = max(0.0, (ax_bbox.x0 - tb.x0) / dpi)
            right_ext = max(0.0, (tb.x1 - ax_bbox.x1) / dpi)
            bot_ext = max(0.0, (ax_bbox.y0 - tb.y0) / dpi)
            top_ext = max(0.0, (tb.y1 - ax_bbox.y1) / dpi)

            if c == 0:
                margin_left = max(margin_left, left_ext + pad_buffer)
            if c == n_cols - 1:
                margin_right = max(margin_right, right_ext + pad_buffer)
            if r == n_rows - 1:
                margin_bottom = max(margin_bottom, bot_ext + pad_buffer)
            if r == 0:
                margin_top = max(margin_top, top_ext + pad_buffer)

    # Measure ONLY in-between decorations (inspect specific internal artists)
    max_h_between = pad_buffer
    max_v_between = pad_buffer

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axs_2d[r, c]
            ax_bbox = ax.get_window_extent(renderer)

            # Horizontal in-between decorations
            if c < n_cols - 1:
                # Right-side decorations (e.g., tick labels / right annotations)
                for tick in ax.yaxis.get_major_ticks():
                    if tick.tick2line.get_visible():
                        t_bbox = tick.tick2line.get_window_extent(renderer)
                        max_h_between = max(max_h_between, (t_bbox.x1 - ax_bbox.x1) / dpi)
                    if tick.label2.get_visible() and tick.label2.get_text():
                        l_bbox = tick.label2.get_window_extent(renderer)
                        max_h_between = max(max_h_between, (l_bbox.x1 - ax_bbox.x1) / dpi)

            if c > 0:
                # Left-side decorations (excluding Column 0's big y-labels)
                for tick in ax.yaxis.get_major_ticks():
                    if tick.tick1line.get_visible():
                        t_bbox = tick.tick1line.get_window_extent(renderer)
                        max_h_between = max(max_h_between, (ax_bbox.x0 - t_bbox.x0) / dpi)
                    if tick.label1.get_visible() and tick.label1.get_text():
                        l_bbox = tick.label1.get_window_extent(renderer)
                        max_h_between = max(max_h_between, (ax_bbox.x0 - l_bbox.x0) / dpi)

            # Vertical in-between decorations
            if r < n_rows - 1:
                # Bottom decorations (x-tick labels & x-labels)
                if ax.xaxis.get_visible():
                    lbl = ax.xaxis.get_label()
                    if lbl.get_visible() and lbl.get_text():
                        lbl_bbox = lbl.get_window_extent(renderer)
                        max_v_between = max(max_v_between, (ax_bbox.y0 - lbl_bbox.y0) / dpi)

                    for tick in ax.xaxis.get_major_ticks():
                        if tick.label1.get_visible() and tick.label1.get_text():
                            t_bbox = tick.label1.get_window_extent(renderer)
                            max_v_between = max(max_v_between, (ax_bbox.y0 - t_bbox.y0) / dpi)

            if r > 0:
                # Top decorations of row r (titles)
                title = ax.title
                if title.get_visible() and title.get_text():
                    t_bbox = title.get_window_extent(renderer)
                    max_v_between = max(max_v_between, (t_bbox.y1 - ax_bbox.y1) / dpi)

    # Single uniform inter_pad strictly derived from in-between elements
    inter_pad_def = max(0.005, max(max_h_between, max_v_between))
    if inter_pad is None:
        inter_pad = inter_pad_def
    elif isinstance(inter_pad, str):
        assert inter_pad.endswith('%'), 'inter_pad string must be a %'
        inter_pad = float(inter_pad[:-1]) / 100 * inter_pad_def
    elif isinstance(inter_pad, float):
        pass
    else:
        raise ValueError

    # Identify equal-aspect rows
    is_equal_aspect_row = [False] * n_rows
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axs_2d[r, c]
            asp = ax.get_aspect()
            if asp in ('equal', 1.0, 1) or (isinstance(asp, (float, int)) and asp != 'auto'):
                is_equal_aspect_row[r] = True
            elif hasattr(ax, 'get_box_aspect') and ax.get_box_aspect() is not None:
                is_equal_aspect_row[r] = True

    # Compute column frame width (W_ax) and row heights
    fig_w_in, _ = fig.get_size_inches()
    avail_w_in = fig_w_in - margin_left - margin_right - ((n_cols - 1) * inter_pad)
    col_width_in = max(0.01, avail_w_in / n_cols)

    row_ax_heights = []
    for r in range(n_rows):
        if is_equal_aspect_row[r]:
            row_ax_heights.append(col_width_in)
        else:
            row_ax_heights.append(row_unequal_h)

    # Compute total figure height
    total_grid_h = sum(row_ax_heights) + ((n_rows - 1) * inter_pad)
    fig_h_in = total_grid_h + margin_bottom + margin_top
    if fig_height_max is not None and fig_h_in > fig_height_max:
        n_equal_rows = sum(is_equal_aspect_row)
        n_nonequal_rows = n_rows - n_equal_rows
        raise ValueError(
            f"fig. height ({fig_h_in}) exceeds maximum ({fig_height_max}). "
            f"Cannot accommodate {n_equal_rows} equal-aspect rows and {n_nonequal_rows} rows of height {row_unequal_h}.")
    fig.set_size_inches(fig_w_in, fig_h_in, forward=True)

    # Convert to relative GridSpec parameters
    left_frac = margin_left / fig_w_in
    right_frac = 1.0 - (margin_right / fig_w_in)
    bottom_frac = margin_bottom / fig_h_in
    top_frac = 1.0 - (margin_top / fig_h_in)

    wspace = inter_pad / col_width_in
    avg_h_ax = sum(row_ax_heights) / n_rows
    hspace = inter_pad / max(0.01, avg_h_ax)

    # Update GridSpec
    gs = axs_2d[0, 0].get_gridspec()
    gs.set_height_ratios(row_ax_heights)
    gs.update(left=left_frac, right=right_frac, bottom=bottom_frac, top=top_frac, wspace=wspace, hspace=hspace)


def fit_figure_layout(fig, axes, row_unequal_h, pad_clearance=0.01):
    """
    Fits figure dimensions and GridSpec parameters analytically with tight margins
    and correct inter-row gap calculations (eliminating redundant tick height additions).
    """
    # 1. Disable active layout engine
    fig.set_layout_engine(None)

    axs_2d = np.atleast_2d(axes)
    n_rows, n_cols = axs_2d.shape

    # 2. Align labels for clean alignment
    fig.align_ylabels(axs_2d)
    fig.align_xlabels(axs_2d)

    # --- Analytic Metric Helpers (Outputs Physical Inches) ---
    def get_text_extents(text_obj):
        if not text_obj or not text_obj.get_visible() or not text_obj.get_text():
            return 0.0, 0.0
        fs = text_obj.get_fontsize()
        lines = text_obj.get_text().split('\n')
        num_lines = len(lines)
        max_chars = max((len(l) for l in lines), default=0)

        w = max_chars * (fs / 72.0) * 0.45
        h = num_lines * (fs / 72.0) * 1.0

        rot = text_obj.get_rotation()
        if rot in (90, 270, 'vertical', '90', '270'):
            return h, w
        return w, h

    def get_axis_left_footprint(ax):
        ylabel = ax.yaxis.get_label()
        yw, _ = get_text_extents(ylabel)

        ytick_labels = [l.get_text() for l in ax.get_yticklabels() if l.get_visible() and l.get_text()]
        ytick_w = 0.0
        if ytick_labels and ax.yaxis.get_visible() and ax.axison:
            fs = ax.get_yticklabels()[0].get_fontsize()
            max_c = max(len(l) for l in ytick_labels)
            tick_len = ax.yaxis.get_major_ticks()[
                           0].tick1line.get_markersize() / 72.0 if ax.yaxis.get_major_ticks() else 0.03
            ytick_w = (max_c * (fs / 72.0) * 0.45) + tick_len

        annotation_w = 0.0
        for txt in ax.texts:
            tw, _ = get_text_extents(txt)
            if tw > 0:
                annotation_w = max(annotation_w, tw)

        return max(yw, annotation_w) + ytick_w

    def get_axis_right_footprint(ax):
        ylabel = ax.yaxis.get_label()
        yw, _ = get_text_extents(ylabel)

        ytick_w = 0.0
        for tick in ax.yaxis.get_major_ticks():
            if tick.label2.get_visible() and tick.label2.get_text():
                tw, _ = get_text_extents(tick.label2)
                ytick_w = max(ytick_w, tw)

        annotation_w = 0.0
        for txt in ax.texts:
            tw, _ = get_text_extents(txt)
            if tw > 0:
                annotation_w = max(annotation_w, tw)

        return max(yw, annotation_w) + ytick_w

    def get_axis_bottom_footprint(ax):
        xlabel = ax.xaxis.get_label()
        _, xh = get_text_extents(xlabel)

        xtick_labels = [l.get_text() for l in ax.get_xticklabels() if l.get_visible() and l.get_text()]
        xtick_h = 0.0
        if xtick_labels and ax.xaxis.get_visible() and ax.axison:
            fs = ax.get_xticklabels()[0].get_fontsize()
            tick_len = ax.xaxis.get_major_ticks()[
                           0].tick1line.get_markersize() / 72.0 if ax.xaxis.get_major_ticks() else 0.03
            xtick_h = (fs / 72.0) * 1.0 + tick_len

        annotation_h = 0.0
        for txt in ax.texts:
            _, th = get_text_extents(txt)
            if th > 0:
                annotation_h = max(annotation_h, th)

        return max(xh, annotation_h) + xtick_h

    def get_axis_top_footprint(ax):
        title = ax.title
        _, th = get_text_extents(title)

        annotation_h = 0.0
        for txt in ax.texts:
            _, ah = get_text_extents(txt)
            if ah > 0:
                annotation_h = max(annotation_h, ah)

        return max(th, annotation_h)

    def ytick_w_internal(ax):
        if not ax.axison or not ax.yaxis.get_visible():
            return 0.0
        labels = [l.get_text() for l in ax.get_yticklabels() if l.get_visible() and l.get_text()]
        if not labels:
            return 0.0
        max_chars = max((len(l) for l in labels), default=0)
        if max_chars == 0:
            return 0.0
        fs = ax.get_yticklabels()[0].get_fontsize()
        tick_len = ax.yaxis.get_major_ticks()[
                       0].tick1line.get_markersize() / 72.0 if ax.yaxis.get_major_ticks() else 0.03
        return (max_chars * (fs / 72.0) * 0.45) + tick_len

    # 3. Detect equal-aspect rows
    is_equal_aspect_row = [False] * n_rows
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axs_2d[r, c]
            asp = ax.get_aspect()
            if asp in ('equal', 1.0, 1) or (isinstance(asp, (float, int)) and asp != 'auto'):
                is_equal_aspect_row[r] = True
            elif hasattr(ax, 'get_box_aspect') and ax.get_box_aspect() is not None:
                is_equal_aspect_row[r] = True

    # 4. Compute strict border margins dynamically
    margin_left = max(get_axis_left_footprint(axs_2d[r, 0]) for r in range(n_rows)) + pad_clearance
    margin_right = max(get_axis_right_footprint(axs_2d[r, n_cols - 1]) for r in range(n_rows)) + pad_clearance
    margin_bottom = max(get_axis_bottom_footprint(axs_2d[n_rows - 1, c]) for c in range(n_cols)) + pad_clearance
    margin_top = max(get_axis_top_footprint(axs_2d[0, c]) for c in range(n_cols)) + pad_clearance

    margin_left = max(0.01, margin_left)
    margin_right = max(0.01, margin_right)
    margin_bottom = max(0.01, margin_bottom)
    margin_top = max(0.01, margin_top)

    # 5. Estimate in-between gaps compactly
    max_h_between = 0.0
    for c in range(n_cols - 1):
        gap = max(ytick_w_internal(axs_2d[r, c + 1]) for r in range(n_rows))
        max_h_between = max(max_h_between, gap)

    max_v_between = 0.0
    for r in range(n_rows - 1):
        # Only account for top decorations (titles/annotations) of the lower row
        top_decorations = max(get_axis_top_footprint(axs_2d[r + 1, c]) for c in range(n_cols))
        max_v_between = max(max_v_between, top_decorations)

    inter_pad = max(0.002, max(max_h_between, max_v_between) + pad_clearance)

    # 6. Compute column frame width (W_ax) and row heights
    fig_w_in, _ = fig.get_size_inches()
    avail_w_in = fig_w_in - margin_left - margin_right - ((n_cols - 1) * inter_pad)
    col_width_in = max(0.01, avail_w_in / n_cols)

    row_ax_heights = []
    for r in range(n_rows):
        if is_equal_aspect_row[r]:
            row_ax_heights.append(col_width_in)
        else:
            row_ax_heights.append(row_unequal_h)

    # 7. Compute total figure height
    total_grid_h = sum(row_ax_heights) + ((n_rows - 1) * inter_pad)
    fig_h_in = total_grid_h + margin_bottom + margin_top

    fig.set_size_inches(fig_w_in, fig_h_in, forward=True)

    # 8. Convert to relative GridSpec parameters
    left_frac = margin_left / fig_w_in
    right_frac = 1.0 - (margin_right / fig_w_in)
    bottom_frac = margin_bottom / fig_h_in
    top_frac = 1.0 - (margin_top / fig_h_in)

    wspace = inter_pad / col_width_in
    avg_h_ax = sum(row_ax_heights) / n_rows
    hspace = inter_pad / max(0.01, avg_h_ax)

    # 9. Update GridSpec
    gs = axs_2d[0, 0].get_gridspec()
    gs.set_height_ratios(row_ax_heights)
    gs.update(
        left=left_frac,
        right=right_frac,
        bottom=bottom_frac,
        top=top_frac,
        wspace=wspace,
        hspace=hspace
    )


def plot_metrics_grid(gen_datasets: list[GeneratedDataset | str], metrics: list[str], col_labels: list[str] = None,
                      filepath: str | pathlib.Path = None, fig_width=main_fig_widths['2col'], row_height=0.4,
                      inter_pad=None, row_label_size=6.0, col_label_size=6.0):
    n_cols = len(gen_datasets)

    # Load the GeneratedDataset and their associated training datasets
    model_splits_dataset = []
    # splits_name = BiomoleculeDataset.splits_name  # Name of the splits shown
    splits_name = ['train', 'test']  # Name of the splits shown
    dataset_labels = splits_name + ['model']
    dataset_colors = [datasets_fig_colors[l] for l in dataset_labels]

    gen_datasets = copy.copy(gen_datasets)
    for j, gen_dataset in enumerate(gen_datasets):
        if isinstance(gen_dataset, str):
            gen_dataset = load_dataset(gen_dataset)

        # Always get the pruned version of the GeneratedDataset
        gen_dataset = gen_dataset.prune()
        gen_dataset.element_class = heavy_atoms_class_name
        gen_datasets[j] = gen_dataset

        # Load the associated model split datasets
        model_dataset: BiomoleculeDataset = load_dataset(gen_dataset.model_dataset)
        model_dataset.element_class = heavy_atoms_class_name
        model_split_datasets_temp = model_dataset.load_splits(splits_name, set_ID=gen_dataset.model_split_ID[0])
        model_splits_dataset.append(model_split_datasets_temp)

    # Define the row labels
    row_labels = []
    row_metric_ind = []
    row_equal_ar_ind = []
    for i, metric in enumerate(metrics):
        if metric in equal_aspect_metrics:
            row_equal_ar_ind += [len(row_labels) + j for j in range(len(dataset_labels))]

        if metric in maplike_metrics:
            row_labels += [metrics_label[metric] + f'\n{label}' for label in dataset_labels]
            row_metric_ind += len(dataset_labels) * [i]
        else:
            row_labels.append(metrics_label[metric])
            row_metric_ind += [i]
    metrics_row_ind = {m: row_metric_ind.index(i) for i, m in enumerate(metrics)}  # Starting row index of each metric
    metrics_1D = [m for m in metrics if m not in maplike_metrics]
    legend_row_ind = metrics_row_ind[metrics_1D[0]] if metrics_1D else -1
    n_rows = len(row_labels)

    # Compute approximate figure size for the grid
    fig_width_max = fig_width
    fig_w = fig_width_max
    fig_h, height_ratios = compute_grid_geometry(fig_width_max, fig_height_max, n_rows=n_rows, n_cols=n_cols,
                                                 row_h=row_height, equal_aspect_row_ind=row_equal_ar_ind)
    figsize = (fig_w, fig_h)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, layout='constrained')
    axes = axes.reshape((n_rows, n_cols))  # Necessary since array is 1D when n_rows=1
    heavy_atom_metrics = [m for m in metrics if m in defined_props[heavy_atoms_class_name]]
    combined_metrics = ['bond_length_Z', 'bond_angle_Z']

    for j in range(n_cols):
        # Compare the requested metrics for the given system
        compared_datasets = model_splits_dataset[j] + [gen_datasets[j]]

        # First, compare metrics of datasets from the same system

        # Check if the forcefield of the compared dataset defines V_dih_cmap
        if compared_datasets[0].forcefield == amber14sb_OL15 and 'V_dih_cmap' in heavy_atom_metrics:
            # Remove 'V_dih_cmap' and add placeholder figure.
            metrics_temp = copy.copy(heavy_atom_metrics)
            V_dih_cmap_ind = metrics_temp.index('V_dih_cmap')
            metrics_temp.pop(V_dih_cmap_ind)

            _, figs_temp = compare_prop_dist(datasets=compared_datasets, properties_name=metrics_temp, plot=True,
                                             datasets_color=dataset_colors, combined=combined_metrics,
                                             downsampling_factor=downsampling_factor)

            div_cmap_fig_placeholder, ax = plt.subplots()
            ax.axis('off')
            ax.annotate('N/A', (0.5, 0.5), xycoords='axes fraction', ha='center', va='center', fontsize=8)
            figs_temp.insert(V_dih_cmap_ind, div_cmap_fig_placeholder)
        else:
            _, figs_temp = compare_prop_dist(datasets=compared_datasets, properties_name=heavy_atom_metrics, plot=True,
                                             datasets_color=dataset_colors, combined=combined_metrics,
                                             downsampling_factor=downsampling_factor)

        # If h_bond_map is requested, the metric has to be calculated from the all-atoms version of the dataset
        if 'h_bond_map' in metrics:
            for d in compared_datasets:
                d.element_class = all_atoms_class_name
            _, fig_hbond = compare_prop_dist(datasets=compared_datasets, properties_name=['h_bond_map'], plot=True,
                                             datasets_color=dataset_colors)
            h_bond_map_metric_index = metrics.index('h_bond_map')
            figs_temp = figs_temp[:h_bond_map_metric_index] + fig_hbond + figs_temp[h_bond_map_metric_index:]

        for i, metric in enumerate(metrics):
            axis_row_ind = metrics_row_ind[metric]  # Get the starting row index of the given metric

            # Handle map-like metrics differently since they need to be expanded.
            if metric in maplike_metrics:
                # Unpack multiple axes in figures
                for k, panel_ax in enumerate(figs_temp[i].axes[:len(compared_datasets)]):
                    # Copy axis content to main figure panel
                    ax: plt.Axes = axes[axis_row_ind + k, j]
                    copy_ax(panel_ax, ax)

                    # Remove title and tick labels
                    ax.set_title('')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])

                    # Remove ticks
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if metric in ['backbone_dihedrals']:
                        ax.set_xticks([0.0])
                        ax.set_yticks([0.0])

                    # Remove axis frame
                    if metric in ['h_bond_map']:
                        ax.set_frame_on(False)

                    # Aspect ratio
                    if metric in ['backbone_dihedrals', 'h_bond_map']:
                        ax.set_aspect('equal')
                    else:
                        ax.set_aspect('auto')

                    # Add axes labels on the bottom panel of the first column
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                    if k == len(compared_datasets) - 1 and j == 0:
                        if metric == 'backbone_dihedrals':
                            ax.set_xlabel('\u03C6', labelpad=0.25)
                            ax.set_ylabel('\u03A8', labelpad=0.25)
                        elif metric == 'h_bond_map':
                            ax.set_xlabel('Acceptor residue')
                            ax.set_ylabel('Donor residue')

                plt.close(figs_temp[i])
                continue

            # Copy axis content
            ax: plt.Axes = axes[axis_row_ind, j]
            ax_temp = figs_temp[i].gca()
            copy_ax(ax_temp, ax)
            plt.close(figs_temp[i])

            # Axis labels and limits
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_yticks([])
            ax.set_ylim(0.0, None)

            # The default xtick formatter uses a dictionary of floats for looking up which ticks are labelled. Sometimes the xticks
            # are not found due to floating-point errors. Fix that by changing the major tick formatter.
            ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            if metric.endswith('_Z'):
                ax.xaxis.set_ticks([-3.0, 3.0])
                ax.set_xlim([-3.5, 3.5])
            elif metric in maplike_metrics:
                pass
            else:
                ax.xaxis.set_major_locator(AdaptivePrecisionEdgeLocator(num_ticks=2, tick_max_extent=0.3))

            # Remove axis frame except bottom line
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # Set the model line on top of all other lines
            for line in ax.lines:
                if gen_datasets[j].name in line._label:
                    line.set_zorder(len(compared_datasets))

            # Add legend on last column of first 1D distribution metric
            if j == n_cols - 1 and axis_row_ind == legend_row_ind:
                legended_handles, _ = ax.get_legend_handles_labels()
                for k, h in enumerate(legended_handles):
                    h.set_label(dataset_labels[k])
                ax.legend(loc='upper right', bbox_to_anchor=(1.02, 1.1), borderpad=0.0, borderaxespad=0.0,
                          labelspacing=0.2)
            else:
                leg = ax.get_legend()
                if leg:
                    leg.remove()

    # Add colorbar for maplike metrics
    cbar_pos_y = 0.1  # colorbar vertical position (inches) relative to axis
    cbar_width = '95%'  # colorbar bar width as a fraction of the axis width
    cbar_height = 0.025  # colorbar bar height (inches)
    cbar_label = defaultdict(lambda: '', {'h_bond_map': 'Frequency'})
    cbar_tick_params = dict(bottom=True, top=False, direction='in', pad=1.0, length=4.0)
    for metric in set(metrics) & set(maplike_metrics):
        axis_row_ind = metrics_row_ind[metric]

        cbars = []
        if metric == 'h_bond_map':
            # Add colorbar above each axis on every column
            for j in range(n_cols):
                metric_column_axes = axes[axis_row_ind:axis_row_ind + len(dataset_labels), j]

                # Make sure images use the same colormap
                h_bond_maps = [ax.get_children()[0] for ax in metric_column_axes]
                map_cmax = max([im.get_clim()[1] for im in h_bond_maps])
                for h_bond_map in h_bond_maps:
                    h_bond_map.set_clim(0.0, map_cmax)

                target_ax = metric_column_axes[0]
                bbox_trans = mtransforms.offset_copy(target_ax.transAxes, fig=fig, x=0.0, y=cbar_pos_y, units="inches")
                cax = inset_axes(target_ax, width=cbar_width, height=cbar_height, loc="upper center",
                                 bbox_to_anchor=(0.0, 0.0, 1.0, 1.0), bbox_transform=bbox_trans, borderpad=0)

                cbar = fig.colorbar(h_bond_maps[0], ax=target_ax, cax=cax, orientation='horizontal')
                cbars.append(cbar)

                # Wrap get_tightbbox to use the extra cax in the tight box calculations. Needed to calculate paddings.
                target_ax.get_tightbbox = tightbbox_wrapper(target_ax, cax)

        elif metric == 'backbone_dihedrals':
            # Add colorbar below axis on the last column
            metric_column_axes = axes[axis_row_ind + len(dataset_labels) - 1, :]

            # Make sure images use the same colormap
            rama_maps = [ax.get_children()[0] for ax in metric_column_axes]
            map_cmax = max([im.get_clim()[1] for im in rama_maps])
            for rama_map in rama_maps:
                rama_map.set_clim(None, map_cmax)

            target_ax = metric_column_axes[-1]
            bbox_trans = mtransforms.offset_copy(target_ax.transAxes, fig=fig, x=0.0, y=-cbar_pos_y, units="inches")
            cax = inset_axes(target_ax, width=cbar_width, height=cbar_height, loc="lower center",
                             bbox_to_anchor=(0.0, 0.0, 1.0, 1.0), bbox_transform=bbox_trans, borderpad=0)

            cbar = fig.colorbar(rama_maps[0], cax=cax, orientation='horizontal')
            cbars.append(cbar)

            # Wrap get_tightbbox to use the extra cax in the tight box calculations. Needed to calculate paddings.
            target_ax.get_tightbbox = tightbbox_wrapper(target_ax, cax)

        # Format colorbars
        for j, cbar in enumerate(cbars):
            vmin, vmax = cbar.mappable.get_clim()
            # push = 0.05 * (vmax - vmin)
            # cbar.set_ticks([vmin, vmax - push])
            cbar.set_ticks([vmin, vmax])
            cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            cbar.ax.tick_params(**cbar_tick_params)
            cbar.outline.set_visible(False)

        # Add colorbar label
        if cbar_label[metric]:
            cbar = cbars[0]
            cbar.ax.xaxis.set_label_position('top')
            cbar.set_label(cbar_label[metric], loc='center', labelpad=2.0)

    # Add row labels
    for i, ax in enumerate(axes[:, 0]):
        ax.annotate(row_labels[i], xy=(0, 0.5), xytext=(-20, 0), fontsize=row_label_size, xycoords='axes fraction',
                    textcoords='offset points', va='center', ha='center', rotation='vertical')

    # Add column labels
    if col_labels is None:
        col_labels = []
    col_labels = copy.copy(col_labels)
    for d in gen_datasets:
        col_labels.append(next(v for k, v in datasets_fig_label.items() if d.model_dataset.startswith(k)))
    col_label_y_pos = max([(ax.get_tightbbox().y1 - ax.bbox.y1) / fig.dpi * 72 for ax in axes[0, :]]) + 5
    for j, ax in enumerate(axes[0, :]):
        ax.annotate(col_labels[j], xy=(0.5, 1.0), xytext=(0, col_label_y_pos), fontsize=col_label_size,
                    xycoords='axes fraction', textcoords='offset points', va='bottom', ha='center',
                    rotation='horizontal')

    # Optimize grid layout
    optimize_gridspec(fig, axes, row_height, inter_pad=inter_pad, fig_height_max=fig_height_max)

    # Save plot
    if filepath:
        filepath = pathlib.Path(filepath)
        fig.savefig(filepath, dpi=fig_dpi)
        plt.close(fig)

    return fig, axes


def plot_sampling_metrics(gen_dataset_names, metrics, filepath: str | pathlib.Path = None):
    plot_args = {'left': 0.1, 'right': 0.99, 'top': 0.95, 'bottom': 0.05, 'wspace': 0.25, 'hspace': 0.4,
                 'bbox_to_anchor': (0.985, 0.82), 'title_pad': 8, 'save_file_suffix': 'figure5',
                 'subplots_size_factor': 0.75
                 }
    cols, rows = len(gen_dataset_names), len(metrics)
    max_w, max_h = 7.08, 7.28
    square_aspect, rect_aspect = 1.0, 0.5

    square_rows = 0
    rect_rows = rows - square_rows

    wspace, hspace = plot_args['wspace'], plot_args['hspace']
    subplot_size = (max_w / cols) * plot_args['subplots_size_factor']
    content_height = subplot_size * (square_rows * square_aspect + rect_rows * rect_aspect)
    spacing_height = (rows - 1) * hspace * subplot_size

    fig_h = content_height + spacing_height
    if fig_h > max_h:
        raise ValueError('Figure too tall')
    fig_w = max_w

    # --- Step 2: create figure with computed size ---
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)

    height_ratios = (rect_rows * [rect_aspect])

    gs = gridspec.GridSpec(rows, cols, wspace=wspace, hspace=hspace, height_ratios=height_ratios)

    for gen_dataset_name in gen_dataset_names:
        gen_dataset = load_dataset(gen_dataset_name, element_class=heavy_atoms_class_name)
        gen_dataset = gen_dataset.prune()

        dataset_model = load_dataset(gen_dataset.model_dataset, element_class=heavy_atoms_class_name)
        dataset_train, dataset_test = dataset_model.load_splits(['train', 'test'],
                                                                set_ID=gen_dataset.model_split_ID[0])
        dataset_shortname = main_datasets_shortname[dataset_model.name]

        for metric in metrics:
            if metric.startswith('JSD'):
                if metric == 'JSD_Rg':
                    metric_key = 'R_g'
                elif metric == 'JSD_bond_lengths':
                    metric_key = 'bond_length'
                elif metric == 'JSD_bond_angles':
                    metric_key = 'bond_angle'
                elif metric == 'JSD_rama':
                    metric_key = 'backbone_dihedrals'
                else:
                    raise NotImplementedError

                model_data = calculate_JS_dist_subsets(gen_dataset, dataset_test, metric_key)
                train_data = calculate_JS_dist_subsets(dataset_train, dataset_test, metric_key)
                N_samples_model, model_JSDs = model_data['n_samples'], model_data[metric_key]
                model_avg_JSDs = np.mean(model_JSDs, axis=1)
                model_error_JSDs = np.std(model_JSDs, axis=1)
                N_samples_train, train_JSDs = train_data['n_samples'], train_data[metric_key]
                train_avg_JSDs = np.mean(train_JSDs, axis=1)
                train_error_JSDs = np.std(train_JSDs, axis=1)

            elif metric == 'dihedral_states':
                x2_model, model_curves = calculate_dihedral_sampling_rate(gen_dataset)
                avg_model, std_model = np.mean(model_curves, axis=1), np.std(model_curves, axis=1)

                x2_train, train_curves = calculate_dihedral_sampling_rate(dataset_train)
                avg_train, std_train = np.mean(train_curves, axis=1), np.std(train_curves, axis=1)

                x2_test, test_curves = calculate_dihedral_sampling_rate(dataset_test)
                avg_test, std_test = np.mean(test_curves, axis=1), np.std(test_curves, axis=1)

            elif metric == 'performance':
                # Benchmarking performance ns/day
                md_benchmarking_file = config.root_dir / pathlib.Path('data', 'benchmarking', 'md_benchmarking.json')

                with open(md_benchmarking_file, "r") as json_file:
                    data = json.load(json_file)
                benchmarking_performance = data['md_benchmarking']

            ################################
            # PLOTTING#
            ################################

            ax = plt.subplot(gs[metrics.index(metric), gen_dataset_names.index(gen_dataset_name)])
            # Follow IBM Design Library colorblind accessible colorscheme
            model_label_kwargs, test_label_kwargs, train_label_kwargs = {'label': 'model', 'color': '#DC267F'}, {
                'label': 'test', 'color': '#648FFF'}, {'label': 'train', 'color': '#FFB000'}
            md_label_kwargs = {'label': 'MD', 'color': '#117733'}
            ytext_kwargs = {'xycoords': 'axes fraction', 'ha': 'center', 'va': 'center', 'rotation': 'vertical',
                            'fontsize': 7}
            err_alpha = 0.45

            ax.set_box_aspect(rect_aspect)

            if metric == 'test':
                continue

            elif metric == 'JSD_Rg':
                ax.plot(N_samples_model, model_avg_JSDs, **model_label_kwargs)
                ax.fill_between(N_samples_model, model_avg_JSDs + model_error_JSDs, model_avg_JSDs - model_error_JSDs,
                                alpha=err_alpha, color=model_label_kwargs['color'])

                ax.plot(N_samples_train, train_avg_JSDs, **train_label_kwargs)
                ax.fill_between(N_samples_train, train_avg_JSDs + train_error_JSDs, train_avg_JSDs - train_error_JSDs,
                                alpha=err_alpha, color=train_label_kwargs['color'])
                ax.set_yscale('log', base=10)

            elif metric == 'JSD_rama':
                ax.plot(N_samples_model, model_avg_JSDs, **model_label_kwargs)
                ax.fill_between(N_samples_model, model_avg_JSDs + model_error_JSDs, model_avg_JSDs - model_error_JSDs,
                                alpha=err_alpha, color=model_label_kwargs['color'])
                # ax.plot(N_samples_test, test_avg_JSDs, **test_label_kwargs)
                # ax.fill_between(N_samples_test, test_avg_JSDs + test_error_JSDs, test_avg_JSDs - test_error_JSDs, alpha=err_alpha, color=test_label_kwargs['color'])
                ax.plot(N_samples_train, train_avg_JSDs, **train_label_kwargs)
                ax.fill_between(N_samples_train, train_avg_JSDs + train_error_JSDs, train_avg_JSDs - train_error_JSDs,
                                alpha=err_alpha, color=train_label_kwargs['color'])
                ax.set_yscale('log', base=10)

            elif metric == 'JSD_bond_lengths':
                ax.plot(N_samples_model, model_avg_JSDs, **model_label_kwargs)
                ax.fill_between(N_samples_model, model_avg_JSDs + model_error_JSDs, model_avg_JSDs - model_error_JSDs,
                                alpha=err_alpha, color=model_label_kwargs['color'])
                # ax.plot(N_samples_test, test_avg_JSDs, **test_label_kwargs)
                # ax.fill_between(N_samples_test, test_avg_JSDs + test_error_JSDs, test_avg_JSDs - test_error_JSDs, alpha=err_alpha, color=test_label_kwargs['color'])
                ax.plot(N_samples_train, train_avg_JSDs, **train_label_kwargs)
                ax.fill_between(N_samples_train, train_avg_JSDs + train_error_JSDs, train_avg_JSDs - train_error_JSDs,
                                alpha=err_alpha, color=train_label_kwargs['color'])
                ax.set_yscale('log', base=10)
                if gen_dataset_names.index(gen_dataset_name) == len(gen_dataset_names) - 1 and metrics.index(
                        'JSD_bond_lengths') == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.972, 0.9))

            elif metric == 'JSD_bond_angles':
                ax.plot(N_samples_model, model_avg_JSDs, **model_label_kwargs)
                ax.fill_between(N_samples_model, model_avg_JSDs + model_error_JSDs, model_avg_JSDs - model_error_JSDs,
                                alpha=err_alpha, color=model_label_kwargs['color'])
                # ax.plot(N_samples_test, test_avg_JSDs, **test_label_kwargs)
                # ax.fill_between(N_samples_test, test_avg_JSDs + test_error_JSDs, test_avg_JSDs - test_error_JSDs, alpha=err_alpha, color=test_label_kwargs['color'])
                ax.plot(N_samples_train, train_avg_JSDs, **train_label_kwargs)
                ax.fill_between(N_samples_train, train_avg_JSDs + train_error_JSDs, train_avg_JSDs - train_error_JSDs,
                                alpha=err_alpha, color=train_label_kwargs['color'])
                ax.set_yscale('log', base=10)
                if gen_dataset_names.index(gen_dataset_name) == len(gen_dataset_names) - 1 and metrics.index(
                        'JSD_bond_angles') == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.972, 0.9))

            elif metric == 'dihedral_states':
                ax.plot(x2_model, avg_model, **model_label_kwargs)
                ax.fill_between(x2_model, avg_model + std_model, avg_model - std_model, alpha=err_alpha,
                                color=model_label_kwargs['color'])
                ax.plot(x2_train, avg_train, **train_label_kwargs)
                ax.fill_between(x2_train, avg_train + std_train, avg_train - std_train, alpha=err_alpha,
                                color=train_label_kwargs['color'])
                ax.plot(x2_test, avg_test, **test_label_kwargs)
                ax.fill_between(x2_test, avg_test + std_test, avg_test - std_test, alpha=err_alpha,
                                color=test_label_kwargs['color'])
                ax.set_xlim(0, None)
                ax.set_ylim(0, None)

            elif metric == 'performance':
                # Load model benchmarking stats
                model_benchmarking_file = config.root_dir / pathlib.Path('data', 'benchmarking',
                                                                         'training_sampling_times.json')
                with open(model_benchmarking_file, "r") as json_file:
                    data = json.load(json_file)
                samp_tot_dict = data['sampling_CS+HA_per_sample']
                train_CS_dict = data['training_CS']
                train_HA_dict = data['training_HA']
                md_samples_to_use = len(dataset_train)

                samp_h_per_sample = samp_tot_dict[dataset_shortname]
                train_CS_h = train_CS_dict[dataset_shortname]
                train_HA_h = train_HA_dict[dataset_shortname]
                samp_hours_sample = samp_h_per_sample
                train_tot = (train_CS_h + train_HA_h)

                downsampling_rates = [1]  # structures separated by 1 ns [ns/sample]
                N_samples = np.arange(0, md_samples_to_use * 3)

                for downsampling_rate in downsampling_rates:
                    ns_day = benchmarking_performance[dataset_shortname]
                    ns_hour = ns_day / 24
                    samples_hour = ns_hour / downsampling_rate  # [ns/hour] * 1 / [ns/sample] = [sample/hour]
                    hours_sample = 1 / samples_hour
                    md_hours = N_samples * hours_sample

                    ax.plot(N_samples, md_hours, color=md_label_kwargs['color'])

                    if downsampling_rate == 1:
                        model_hours = np.zeros_like(N_samples)
                        # Before training
                        mask1 = N_samples <= md_samples_to_use
                        model_hours[mask1] = hours_sample * N_samples[mask1]
                        # model_hours[mask1] = md_samples_to_use * hours_sample

                        # After switch
                        mask2 = N_samples > md_samples_to_use
                        model_hours[mask2] = (
                                hours_sample * md_samples_to_use  # time to produce first N products
                                + train_tot  # training time
                                + samp_hours_sample * (N_samples[mask2] - md_samples_to_use)
                        )

                        # Plot hours vs. N_samples
                        ax.plot(N_samples, model_hours, **model_label_kwargs)

                        x2 = [md_samples_to_use, md_samples_to_use]
                        t2 = [md_samples_to_use * hours_sample, md_samples_to_use * hours_sample + train_tot]
                        ax.plot(x2, t2, color='#b37b00')

                sample_value_to_annotate = md_samples_to_use
                model_y_to_annotate, md_y_to_annotate = model_hours[sample_value_to_annotate], md_hours[
                    sample_value_to_annotate]
                max_N = max(N_samples) * 0.7
                max_Y = max(md_hours) * 0.5

                ax.annotate(f'{int(hours_sample * 10000)} h/10k',
                            ((sample_value_to_annotate / max_N) * 1.4, (md_y_to_annotate / max_Y) * 1.2), fontsize=5,
                            color=md_label_kwargs['color'], xycoords='axes fraction')

                ax.annotate(f'{np.round((samp_hours_sample * 10000), 1)} h/10k',
                            (sample_value_to_annotate / max_N, (model_y_to_annotate / max_Y) * 0.65), fontsize=5,
                            color=model_label_kwargs['color'], xycoords='axes fraction')

                ax.annotate(f'$T_{{train}}$={int(train_tot)}h', (0.05, 0.6), fontsize=5, color='#b37b00',
                            xycoords='axes fraction')

                ax.set_ylim(0, max_Y)
                ax.set_xlim(0, max_N)

            ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2))

            if metric == 'performance':
                ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2))
                old_ticks = ax.get_xticks()
                new_ticks = [x for x in old_ticks if x != 0]
                ax.set_xticks(new_ticks)
            elif metric[0:3] == 'JSD':
                ax.set_xticks([5000, 10000])
                ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                if metric == 'JSD_rama':
                    ax.set_yticks([0.6])
                else:
                    ax.set_yticks([0.1])
            else:
                ax.set_xticks([5000, 10000])
                ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
                if metric == 'dihedral_states':
                    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
            old_ticks = ax.get_yticks()
            new_ticks = [x for x in old_ticks if x != 0]
            ax.set_yticks(new_ticks)

            # Set specific labels on the first dataset shown (leftmost column)
            metric_names = {
                'JSD_Rg': '$R_g$',
                'JSD_bond_lengths': 'Bond lengths',
                'JSD_bond_angles': 'Bond angles',
                'JSD_rama': 'Ramachandran map',
                'performance': 'Compute time',
                'dihedral_states': 'Unique dih. states'
            }
            if gen_dataset_names.index(gen_dataset_name) == 0:
                xy = (-0.4, 0.5)
                if metric[0:3] == 'JSD':
                    text = metric_names[metric] + '\n' + 'JS dist.'
                    ax.annotate(text, xy=xy, **ytext_kwargs)
                elif metric == 'performance':
                    text = 'Compute time' + '\n' + '[hours]'
                    ax.annotate(text, xy=xy, **ytext_kwargs)
                elif metric == 'dihedral_states':
                    text = 'Unique dihedral' + '\n' + 'states'
                    ax.annotate(text, xy=xy, **ytext_kwargs)
                else:
                    text = metric_names[metric]
                    ax.annotate(text, xy=xy, **ytext_kwargs)

            # Add panel labels when necessary
            if gen_dataset_names.index(gen_dataset_name) == 0:
                panel_label_xy = (-0.65, 1.2)
                panel_label_size = 10
                if metric == 'performance':
                    text = 'a'
                    ax.annotate(text, xy=panel_label_xy, fontsize=panel_label_size, xycoords='axes fraction',
                                fontweight='bold')
                elif metric == 'JSD_bond_lengths':
                    text = 'b'
                    ax.annotate(text, xy=panel_label_xy, fontsize=panel_label_size, xycoords='axes fraction',
                                fontweight='bold')
                elif metric == 'dihedral_states':
                    text = 'c'
                    ax.annotate(text, xy=panel_label_xy, fontsize=panel_label_size, xycoords='axes fraction',
                                fontweight='bold')

            # Set plot titles on the first metric (top row)
            if metrics.index(metric) == 0:
                ax.set_title(datasets_fig_label[dataset_model.name], pad=plot_args['title_pad'])

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    colors = [model_label_kwargs['color'], test_label_kwargs['color'], train_label_kwargs['color']]
    labels = [model_label_kwargs['label'], test_label_kwargs['label'], train_label_kwargs['label']]
    handles = [Line2D([0], [0], color=c, label=lab) for c, lab in zip(colors, labels)]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=plot_args['bbox_to_anchor'], borderpad=0.0,
               borderaxespad=0.0, labelspacing=0.2)

    plt.subplots_adjust(
        left=plot_args['left'],  # margins around the whole figure
        right=plot_args['right'],
        top=plot_args['top'],
        bottom=plot_args['bottom'],
        wspace=plot_args['wspace'],
        hspace=plot_args['hspace']
    )
    if filepath:
        filepath = pathlib.Path(filepath)
        plt.savefig(filepath, format='pdf', pad_inches=0)
        plt.close()

    return fig
