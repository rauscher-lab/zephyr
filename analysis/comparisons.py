import copy
import pathlib
import typing
import argparse
from collections import defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import config
from analysis.stats import Statistics
from chemistry.energy import BondEnergy, BondAngleEnergy, DihedralAngleEnergy
from data.data_classes import backbone_bond_length_name, backbone_bond_angle_name, element_classes_abb, \
    c_alpha_sidechain_class_name
from data.datasets import BiomoleculeDataset, load_dataset, GeneratedDataset
from analysis.metrics import prop_binwidth, prop_units, Biomolecule_prop_names, Biomolecule_adj_prop_names, \
    Biomolecule_energy_prop_names, Biomolecule_Z_prop_names, Biomolecule_misc_prop_names, defined_props
from visualization import plot_rama, plot_UMAP, plot_hbond_map, shorten_labels


def compare_prop_dist(datasets: list[BiomoleculeDataset], properties_name: list = None,
                      plot=True, plot_filename: str | pathlib.Path = '',
                      cdf: bool | dict[str, bool] = False, combined: bool | list[str] = False,
                      return_fig_crit: typing.Callable = lambda fig: True, datasets_color: list[str] = None,
                      n_bins_plot=200, downsampling_factor: int | dict[str, int] = 1):
    """
    Compare statistics of the given properties between the given datasets.
    Args:
        datasets: list of dataset that will be compared
        properties_name: list of properties to compare
        plot: produce plots in addition to returning property histograms in a dictionary
        plot_filename:filename where plots are saved
        cdf: plot the cumulative distribution instead of the pdf.
            When a dictionary is given, the dictionary overrides the cdf plotting bool for each defined key.
        combined: list of multidimensional properties to combine into a single distribution. Applies to all if bool.
        return_fig_crit: criterion that returns a bool to determine if figure handle should be returned
        datasets_color: colors used for plotting each dataset. Defaults to the matplotlib default.
        n_bins_plot: maximum number of bins that will be plotted. If above threshold, bins are downsampled.
        downsampling_factor: bin downsampling factor applied to all (bool) or given (dict) properties.
                             Applies after imposing n_bins_plot threshold

    Returns:
        dictionary whose keys are property names and values are list of Statistics objects for each dataset
        list of figures for each property
    """
    # Check if all datasets need to have the element class
    datasets_element_class = [d.element_class for d in datasets]
    gmx_en_prop_names_com = set.intersection(*[set(d.forcefield.gmx_energy_names) for d in datasets])
    gmx_en_prop_names = [k for k in datasets[0].forcefield.gmx_energy_names if k in gmx_en_prop_names_com]  # keep order
    cross_class_properties = set(gmx_en_prop_names_com) | {'radial_2nd_mom', 'radial_2nd_mom_CA', 'R_g'}
    is_cross_class_allowed = properties_name is not None and set(properties_name) <= cross_class_properties
    if len(set(datasets_element_class)) > 1 and not is_cross_class_allowed:
        raise ValueError(f"datasets have different element classes.\nelement_classes={datasets_element_class}")
    datasets_element_class = datasets_element_class[0]
    datasets_top_filepath = datasets[0].top_filepath

    # Define default properties
    if properties_name is None:
        properties_name = Biomolecule_prop_names + gmx_en_prop_names + Biomolecule_energy_prop_names
        properties_name += ['rmsd_min_train']
        properties_name += Biomolecule_misc_prop_names

        # Remove undefined or omitted properties
        omitted_prop_name = {'n_elements', 'elements_position', 'elements_position_local',
                             'backbone_sphere_radius', 'elements_dist_CA_SI', 'CA_pdist', 'BondEnergy'}
        omitted_prop_name |= set(Biomolecule_adj_prop_names) | set(Biomolecule_Z_prop_names)
        if datasets_element_class != c_alpha_sidechain_class_name:
            omitted_prop_name |= {'elements_pdist'}
        omitted_prop_name |= set(properties_name) - defined_props[datasets_element_class]
        properties_name = [n for n in properties_name if n not in omitted_prop_name]

    if isinstance(properties_name, set):
        properties_name = list(properties_name)

    if not isinstance(properties_name, list):
        properties_name = [properties_name]

    if isinstance(cdf, dict):
        cdf_dict = defaultdict(lambda: False)
        cdf_dict.update(cdf)
    else:
        cdf_dict = defaultdict(lambda: cdf)

    if isinstance(combined, bool):
        combined = properties_name if combined else []
    elif isinstance(combined, list):
        combined = list(set(combined) & set(properties_name))
        pass
    else:
        raise ValueError(f'combine_props {combined} is not supported.')

    if isinstance(downsampling_factor, int):
        if downsampling_factor < 1:
            raise ValueError(f"Downsampling factor ({downsampling_factor}) is not a positive integer")
        factor = downsampling_factor
        downsampling_factor = defaultdict(lambda: factor)
    elif isinstance(downsampling_factor, dict):
        for prop_name, factor in downsampling_factor.items():
            if not isinstance(factor, int) or factor < 1:
                raise ValueError(f"Downsampling factor for property {prop_name!r} is not a positive integer ({factor})")
        downsampling_factor = defaultdict(lambda: 1, downsampling_factor)

    # Make sure elements_pdist is last
    if 'elements_pdist' in properties_name:
        ind = properties_name.index('elements_pdist')
        properties_name.append(properties_name.pop(ind))

    # Initialize energy modules that may be needed if related properties are plotted
    if 'bond_length' in properties_name:
        BondEnergy_module = BondEnergy(top_file=datasets_top_filepath, element_class=datasets_element_class)
        bonds_name = BondEnergy_module.generate_bonds_name()
    if 'bond_angle' in properties_name:
        BondAngleEnergy_module = BondAngleEnergy(top_file=datasets_top_filepath, element_class=datasets_element_class)
        bond_angles_name = BondAngleEnergy_module.generate_angles_name()
    if 'dihedral_angle' in properties_name:
        DihedralEnergy_module = DihedralAngleEnergy(top_file=datasets_top_filepath,
                                                    element_class=datasets_element_class)
        dihedral_angles_name = DihedralEnergy_module.generate_angles_name(unique=True)

    missing_binwidth = set(properties_name) - set(prop_binwidth.keys())
    if missing_binwidth:
        raise ValueError(f"The following properties do not have a defined binwidth:{missing_binwidth}")

    # Gather stats of all datasets
    datasets_stats_dict = []
    for dataset in datasets:
        stat_dict: dict[str, Statistics] = dataset.statistics(props_name=properties_name)
        stat_dict = copy.deepcopy(stat_dict)
        datasets_stats_dict.append(stat_dict)

        # Combine all dimensions of given properties
        for prop_name in combined:
            stat_dict[prop_name].collapse_dim()

    # Define a dictionary whose values are list of Statistics objects of the given property for each dataset.
    datasets_stats = {name: [stat_dict[name] for stat_dict in datasets_stats_dict] for name in properties_name}
    datasets_stats: dict[str, list[Statistics]]

    # Calculate histograms for each compared dataset and each property
    n_bins_max = int(1e5)  # Skip plotting if the number of bins is above this maximum
    for prop_name in properties_name:
        datasets_prop_stats = datasets_stats[prop_name]

        # Find a bin range in each dimension that works for all datasets' statistics
        bin_ranges_all = torch.stack([stat.bin_lims for stat in datasets_prop_stats], dim=-1)
        bin_ranges_global = torch.stack([bin_ranges_all[:, 0].min(-1)[0], bin_ranges_all[:, 1].max(-1)[0]], dim=-1)
        bin_width_global = torch.stack([stat.bin_widths for stat in datasets_prop_stats], dim=-1).min(-1)[0]

        # If there are too many bins, skip to the next property
        n_bins = torch.diff(bin_ranges_global, dim=1) / bin_width_global
        if torch.any(n_bins > n_bins_max):
            print(f'skipping prop_name={prop_name}, bin ranges:{bin_ranges_global}, n_bins={n_bins}')
            continue

        # Extend the bins to the common range and clamp the number of bins
        for stat in datasets_prop_stats:
            stat.extend_bins(new_bin_lims=bin_ranges_global)
            stat.clamp_n_bins(n_bins_max=n_bins_plot)

            # Downsample bins further when requested
            if downsampling_factor[prop_name] > 1:
                stat.downsample_bins(factor=downsampling_factor[prop_name])

    # Plot statistics
    figs = []
    if plot:
        if datasets_color is None:
            datasets_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(datasets)]

        # Define dataset labels
        dataset_labels_long = []
        for dataset in datasets:
            label = copy.copy(dataset.name)
            if isinstance(dataset, GeneratedDataset) and dataset.is_split and dataset.split_ID[0] == 'PR':
                label = label + (f"\n(\u03C3={dataset.pruning_crit.sig_thresh:.0f},"
                                 f"{dataset.pruning_crit.rejection_ratio:.2%} rej.)")
            dataset_labels_long.append(label)
        dataset_labels_short = shorten_labels(dataset_labels_long)

        if plot_filename:
            plot_filename = pathlib.Path(plot_filename).with_suffix('.pdf')
            if not plot_filename.is_absolute():
                plot_filename = config.plots_dir / plot_filename
            plot_filename.parent.mkdir(parents=True, exist_ok=True)
            pdf = PdfPages(plot_filename)  # Save the plots in a multipage pdf

        for prop_name in properties_name:
            print(f'Plotting {prop_name}')
            datasets_prop_stats = datasets_stats[prop_name]
            prop_d = datasets_prop_stats[0].d

            # Use long labels for the first plot and short ones for subsequent plots
            if prop_name == properties_name[0]:
                dataset_labels = dataset_labels_long
            else:
                dataset_labels = dataset_labels_short

            # Special cases
            is_prop_special = prop_name in ['backbone_dihedrals', 'h_bond_map'] or prop_name.endswith('UMAP')
            if is_prop_special:
                bincounts_all = [stat.bincounts for stat in datasets_prop_stats]
                binedges_all = [stat.binedges for stat in datasets_prop_stats]

                if prop_name == 'backbone_dihedrals':
                    fig = plot_rama(bincounts_all, labels=dataset_labels)
                elif prop_name == 'h_bond_map':
                    h_bond_maps = [stat.mean for stat in datasets_prop_stats]
                    fig = plot_hbond_map(h_bond_maps, labels=dataset_labels)
                elif prop_name.endswith('UMAP'):
                    binedges_x = [e[0] for e in binedges_all]
                    binedges_y = [e[1] for e in binedges_all]
                    fig = plot_UMAP(bincounts_all, binedges_x=binedges_x, binedges_y=binedges_y,
                                    prop_name=prop_name, labels=dataset_labels)
                else:
                    raise ValueError(f"Special property {prop_name} is not configured.")

                if plot_filename:
                    pdf.savefig(figure=fig, dpi=500, pad_inches=None, bbox_inches='tight')
                    plt.close(fig)
                elif return_fig_crit(fig):
                    figs.append(fig)
                continue

            if prop_name == 'elements_pdist':
                n_elements = np.ceil(np.sqrt(2 * prop_d)).astype(int)
                pair_ind_2D = np.triu_indices(n_elements, 1)
                mol_ex = datasets[0][0]
                pairs_label = mol_ex.element_pairs_label()
                pairs_name = [pairs_label[(i, j)] for i, j in zip(*pair_ind_2D)]

            # Property unit
            prop_unit = prop_units[prop_name]
            if prop_unit:
                prop_unit = f" [{prop_unit}]"

            # Loop over each dimension of the property and plot its distribution
            fig, ax = plt.subplots(figsize=(8, 5))
            lines_h = []
            lines_err_h = []
            bin_densities_info: list[tuple[list, list]] = [stat.pdf for stat in datasets_prop_stats]
            bin_densities, bin_densities_err = tuple([T[i] for T in bin_densities_info] for i in range(2))
            for j in range(prop_d):
                x_lim_min, x_lim_max = np.inf, -np.inf  # Find axes limits suitable for all compared datasets
                for i, dataset in enumerate(datasets):
                    bin_edges = datasets_prop_stats[i].binedges[j]
                    bin_width = bin_edges[1] - bin_edges[0]
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    bin_density = bin_densities[i][j]
                    bin_density_err = bin_densities_err[i][j]

                    # Special cases
                    # For rmsd_min_train, set density to nan when plotting the train split to avoid plotting a large density at 0.0
                    if prop_name == 'rmsd_min_train' and dataset.is_split and dataset.split_ID[1] == 'train':
                        bin_density = torch.nan * bin_density

                    # Plot the mean += the error. Recycle the previous plot to accelerate plotting.
                    y_low = bin_density - bin_density_err
                    y_up = bin_density + bin_density_err
                    if j == 0:
                        line_h = ax.plot(bin_centers, bin_density, label=dataset_labels[i], color=datasets_color[i])[0]
                        line_err_h = ax.fill_between(bin_centers, y_low, y_up, facecolor=line_h.get_color(), alpha=0.3)
                        lines_h.append(line_h)
                        lines_err_h.append(line_err_h)
                    else:
                        lines_err_h[i].remove()
                        lines_err_h[i] = ax.fill_between(bin_centers, y_low, y_up, facecolor=lines_h[i].get_color(),
                                                         alpha=0.3)
                        lines_h[i].set_data([bin_centers, bin_density])

                    # Find the first nonzero bin and the last nonzero bin to define the x-axis limits
                    nonzero_bindensity = bin_density.gt(1e-3 * bin_density.max())
                    if nonzero_bindensity.any():
                        first_nonzero_bin_ind, last_nonzero_bin_ind = nonzero_bindensity.nonzero()[[0, -1]]
                        x_lim_min = np.minimum(x_lim_min, bin_edges[first_nonzero_bin_ind] - 2 * bin_width)
                        x_lim_max = np.maximum(x_lim_max, bin_edges[last_nonzero_bin_ind] + 2 * bin_width)

                # Special case: draw the bond length in the topology as a vertical dashed line
                if prop_name == 'bond_length':
                    B0 = BondEnergy_module.b0[j].item()
                    if j == 0:
                        line_h = ax.axvline(B0, linestyle='--', label='topology', color='k')
                        lines_h.append(line_h)
                    else:
                        bond_length_h = lines_h[-1]
                        bond_length_h.set_xdata(2 * [B0])

                # Axes label
                if prop_name == 'elements_pdist':
                    ind_2D = (pair_ind_2D[0][j], pair_ind_2D[1][j])
                    xlabel = f"pdist_{ind_2D[0] + 1}_{ind_2D[1] + 1} ({pairs_name[j]}){prop_unit}"
                elif prop_name == 'backbone_bond_lengths':
                    xlabel = f"{backbone_bond_length_name[j]} Bond Length{prop_unit}"
                elif prop_name == 'backbone_bond_angles':
                    xlabel = f"{backbone_bond_angle_name[j]} Bond Angle{prop_unit}"
                elif prop_name == 'dihedral_omega':
                    xlabel = rf'Backbone dihedral $\omega$ {prop_unit}'
                elif prop_name == 'bond_length':
                    xlabel = f"{prop_name}_{j + 1} ({bonds_name[j]}){prop_unit}"
                elif prop_name in ['bond_angle', 'bond_angle_enddist']:
                    xlabel = f"{prop_name}_{j + 1} ({bond_angles_name[j]}){prop_unit}"
                elif prop_name == 'dihedral_angle':
                    xlabel = f"{prop_name}_{j + 1} ({dihedral_angles_name[j]}){prop_unit}"
                elif prop_d > 1:
                    xlabel = f"{prop_name}_{j + 1}{prop_unit}"
                else:
                    xlabel = f"{prop_name}{' (combined) ' if prop_name in combined else ''}{prop_unit}"
                ax.set_xlabel(xlabel)

                if cdf_dict[prop_name]:
                    ax.set_ylabel('CDF')
                else:
                    err_types = list(set([stat.bincounts_err_type for stat in datasets_prop_stats]))
                    err_tag = '/'.join(err_types) + ' Err.'
                    ax.set_ylabel(f"PDF (\u00B1 {err_tag})")

                # Legend and limits
                ax.legend(handlelength=0.5)
                ax.relim()
                ax.autoscale_view()
                ax.set_xlim(x_lim_min, x_lim_max)
                plt.draw()

                if plot_filename:
                    pdf.savefig(figure=fig, dpi=500, pad_inches=None, bbox_inches='tight')
                elif return_fig_crit(fig):
                    figs.append(copy.deepcopy(fig))

            plt.close(fig)

        if plot_filename:
            pdf.close()
            print(f'Combined plots saved in {plot_filename!r}')

    return datasets_stats, figs


def compare_datasets(datasets: list[str | pathlib.Path | BiomoleculeDataset], splits=True, element_class: str = None,
                     plot_filename: str | pathlib.Path = None, **compare_kwargs):
    """
    Compares the distribution of given properties between multiple datasets
    Args:
        datasets: list of datasets to compare
        element_class: element class of the compared datasets
        splits: split input datasets into splits if splits are defined. GeneratedDataset do not have splits.
        **compare_kwargs: kwargs given to the compare_prop_dist() function
        plot_filename: filename where plots are saved
    Returns:
        None
    """
    for i, d in enumerate(datasets):
        if isinstance(d, str):
            datasets[i] = load_dataset(name=d, element_class=element_class)
        elif isinstance(d, pathlib.Path):
            datasets[i] = load_dataset(directory=d, element_class=element_class)
        elif isinstance(d, BiomoleculeDataset):
            pass
        else:
            raise ValueError

    # Set the element class of the compared datasets if given
    if element_class is not None:
        for d in datasets:
            d.element_class = element_class

    # Load the dataset splits if requested
    if splits:
        compared_datasets = []
        for d in datasets:
            d: BiomoleculeDataset
            if isinstance(d, GeneratedDataset) or not d.has_splits:
                compared_datasets.append(d)
            else:
                compared_datasets.extend(list(d.load_splits()))
    else:
        compared_datasets = datasets

    # Define default value of the plot filename
    datasets_name = [d.name for d in datasets]
    plot_basename = '_vs_'.join(datasets_name) + f"_{datasets[0].element_class_abb}"
    plots_dir = pathlib.Path(config.plots_dir, plot_basename)
    if plot_filename is None:
        plot_filename = plots_dir / plot_basename
    plot_filename = pathlib.Path(plot_filename)
    if not plot_filename.is_absolute():
        plot_filename = plots_dir / plot_filename

    # Compare the distributions of various properties
    return compare_prop_dist(compared_datasets, plot_filename=plot_filename, **compare_kwargs)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', default=None, nargs='*', help='Names of datasets to compare')
    p.add_argument('--element_class', default=None, help='Type of elements to compare. Choices=["AA","HA","CS","CA"]')
    args = p.parse_args()
    if args.element_class in element_classes_abb:
        args.element_class = element_classes_abb[args.element_class]

    if args.datasets is not None:
        compare_datasets(args.datasets, element_class=args.element_class)
