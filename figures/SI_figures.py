import itertools
import json
from collections import defaultdict
import pathlib

import numpy as np
import torch
import matplotlib

import matplotlib.text
from matplotlib import pyplot as plt, gridspec
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

import config

from utils import PiecewiseLR, plot_mu_sigma, copy_ax
from data.data_classes import heavy_atoms_class_name
from data.datasets import GeneratedDataset, load_dataset, main_datasets, gen_datasets, gen_datasets_main, \
    gen_datasets_min, psi_heavy_scan_gen_datasets, psi_coarse_scan_gen_datasets, thermalized_gen_datasets, \
    chig_exp_gen_datasets
from analysis.metrics import Biomolecule_UMAP_prop_names, calculate_rmsd_min
from analysis.comparisons import compare_prop_dist
from models import load_train_stats
from figures import main_configs_ID, SI_figs_dir, SI_fig_width, fig_height_max, fig_dpi, main_fig_widths, \
    epoch_tick_formatter, load_paper_mpl_configs, datasets_fig_label, plot_metrics_grid, fine_metrics, coarse_metrics


def plot_lr_schedules():
    plot_filename = pathlib.Path(SI_figs_dir, 'lr_schedules').with_suffix('.pdf')
    Adam_eps = defaultdict(lambda: 1e-8, {torch.half: 1e-4})[torch.get_default_dtype()]

    # Define coarse-grain model LR schedule
    config.import_configs(f"config_{main_configs_ID['CS']}")
    params = config.lr_piecewise_params
    optimizer = torch.optim.AdamW([torch.tensor(1.0, requires_grad=True)], lr=config.learning_rate,
                                  betas=config.adam_betas,
                                  amsgrad=config.amsgrad, weight_decay=config.weight_decay, eps=Adam_eps)
    lr_scheduler_coarse = PiecewiseLR(optimizer, epochs=params['epochs'], lrs=params['lrs'], funcs=params['funcs'])
    N_epochs_coarse = config.N_epochs

    # Define heavy-atom model LR schedule
    config.import_configs(f"config_{main_configs_ID['HA']}")
    params = config.lr_piecewise_params
    optimizer = torch.optim.AdamW([torch.tensor(1.0, requires_grad=True)], lr=config.learning_rate,
                                  betas=config.adam_betas,
                                  amsgrad=config.amsgrad, weight_decay=config.weight_decay, eps=Adam_eps)
    lr_scheduler_fine = PiecewiseLR(optimizer, epochs=params['epochs'], lrs=params['lrs'], funcs=params['funcs'])
    N_epochs_fine = config.N_epochs

    # Option 1: Plot schedules on different axies
    # fig_width = 1.5 * config.paper_figures_width['1col']
    # fig_size = [fig_width, fig_width / 2]
    # fig, axes = plt.subplots(1, 2, sharey=True, figsize=fig_size)
    # fig.sca(axes[0])
    # lr_scheduler_coarse.plot_lr(log=True, epoch_max=N_epochs_coarse, include_transitions=False)
    # fig.sca(axes[1])
    # lr_scheduler_fine.plot_lr(log=True, epoch_max=N_epochs_fine, include_transitions=False)
    # axes[1].set_ylabel('')
    # axes[0].xaxis.set_major_locator(MaxNLocator(nbins=5))
    # axes[1].xaxis.set_major_locator(MaxNLocator(nbins=5))
    #
    # axes[0].legend(['coarse grain'])
    # axes[1].legend(['heavy atom'])

    # # Option 2: Plot schedules on same axis
    fig_size = (main_fig_widths['1col'], 0.75 * main_fig_widths['1col'])
    fig = plt.figure(figsize=fig_size)
    lr_scheduler_coarse.plot_lr(log=True, epoch_max=N_epochs_coarse, include_transitions=False)
    lr_scheduler_fine.plot_lr(log=True, epoch_max=N_epochs_fine, include_transitions=False)
    ax = fig.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.legend(['coarse-structure model', 'fine-structure model'])

    plt.tight_layout(pad=0.1)
    plt.show(block=False)
    fig.savefig(plot_filename, dpi=fig_dpi)
    plt.close(fig)
    pass


def plot_training_metrics(model_type='CS'):
    if model_type == 'CS':
        metrics_name = ['loss_train', 'loss_valid',
                        'radial_2nd_mom_avg_train_rel_err', 'radial_2nd_mom_avg_valid_rel_err',
                        'radial_2nd_mom_std_train_rel_err', 'radial_2nd_mom_std_valid_rel_err']
    else:
        metrics_name = ['loss_train', 'loss_valid',
                        'BondEnergy_avg_train_rel_err', 'BondEnergy_avg_valid_rel_err',
                        'BondAngleEnergy_tot_avg_train_rel_err', 'BondAngleEnergy_tot_avg_valid_rel_err',
                        'DihedralEnergy_tot_avg_train_rel_err', 'DihedralEnergy_tot_avg_valid_rel_err',
                        'LJEnergy_tot_avg_train_rel_err', 'LJEnergy_tot_avg_valid_rel_err']
    metrics_label = {'loss': 'Loss',
                     'radial_2nd_mom_avg_rel_err': r'$\Delta<\tilde{R}_g>$',
                     'radial_2nd_mom_std_rel_err': r'$\Delta\text{std}(\tilde{R}_g)$',
                     'BondEnergy_avg_rel_err': r'$\Delta<V_\text{B}>$',
                     'BondAngleEnergy_tot_avg_rel_err': r'$\Delta<V_\text{BA}>$',
                     'DihedralEnergy_tot_avg_rel_err': r'$\Delta_<V_\text{Dih.}>$',
                     'LJEnergy_tot_avg_rel_err': r'$\Delta<V_\text{LJ}>$',
                     }
    metrics = defaultdict(dict)
    for sys in main_datasets:
        # Gather the training metrics of the coarse-grain model
        config_ID = main_configs_ID[model_type]
        train_stats = load_train_stats(config_ID=config_ID, dataset=main_datasets[sys], split_set_ID='1')
        for metric_name in metrics_name:
            metrics[sys][metric_name] = train_stats[metric_name]

    # Compute fig size
    n_cols = len(main_datasets)
    n_rows = len(metrics_name) // 2
    fig_h_min = 2.0
    row_h_max = 0.75
    fig_h = fig_height_max
    fig_h = max(min(fig_h, n_rows * row_h_max), fig_h_min)
    fig_w = SI_fig_width
    figsize = (fig_w, fig_h)

    # Plot all metrics in a grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey='row', sharex='all', layout='constrained')
    fig.get_layout_engine().set(w_pad=0.025, h_pad=0.025, wspace=0, hspace=0)
    n_epochs = len(metrics[sys][metric_name])
    for i in range(n_rows):
        metric_name = metrics_name[2 * i].replace('_train', '')

        for j, sys in enumerate(main_datasets):
            ax: plt.Axes = axes[i, j]

            # Plot non-zero values of the train metric
            metric_train = metrics[sys][metrics_name[2 * i]]
            nonzero_ind = metric_train.nonzero()[0]
            epoch_value = nonzero_ind + 1
            ax.plot(epoch_value, metric_train[nonzero_ind], label='Train')

            # Plot non-zero values of the valid metric
            metric_valid = metrics[sys][metrics_name[2 * i + 1]]
            nonzero_ind = metric_valid.nonzero()[0]
            epoch_value = nonzero_ind + 1
            ax.plot(epoch_value, metric_valid[nonzero_ind], label='Valid')

        # Set y-axis limits
        if metric_name == 'loss':
            ax.set_ylim(0.3, 1.0)
        elif metric_name == 'radial_2nd_mom_avg_rel_err':
            ax.set_ylim(-25, None)
        elif metric_name == 'radial_2nd_mom_std_rel_err':
            ax.set_ylim(-40, None)
        elif metric_name in ['BondEnergy_avg_rel_err']:
            ax.set_ylim(-20, 20)
        elif metric_name in ['BondAngleEnergy_tot_avg_rel_err']:
            ax.set_ylim(-10, 100)
        elif metric_name in ['DihedralEnergy_tot_avg_rel_err']:
            ax.set_ylim(-10, 50)
        elif metric_name in ['LJEnergy_tot_avg_rel_err']:
            ax.set_ylim(-50, 150)

        # Format x-axis
        ax.set_xlim(0, None)
        ax.xaxis.set_ticks([0, n_epochs])
        ax.xaxis.set_ticks([n_epochs / 2], minor=True)
        ax.xaxis.set_major_formatter(epoch_tick_formatter)

        # Format y-axis
        axes[i, 0].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axes[i, 0].yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 1.5, 2, 2.5, 4, 5, 10]))

    # Add legend and epoch label
    axes[0, -1].legend()
    axes[-1, 0].set_xlabel('Epoch', labelpad=-4.0)

    # Add col labels
    for i, sys in enumerate(main_datasets):
        axes[0, i].set_title(datasets_fig_label[sys], va='top')

    # Add row labels
    for i, metric_name_train in enumerate(metrics_name[::2]):
        metric_name = metric_name_train.replace('_train', '')
        axes[i, 0].annotate(metrics_label[metric_name], xy=(0, 0.5), xytext=(-20, 0), fontsize=7,
                            xycoords='axes fraction', textcoords='offset points', va='center', ha='center',
                            rotation='vertical')

    # Save the plot
    plot_filepath = pathlib.Path(SI_figs_dir, f"training_metrics_{model_type.replace('CS', 'CG')}.pdf")
    fig.savefig(plot_filepath, dpi=fig_dpi)
    plt.close(fig)


def plot_conditioner_energies():
    # Plot statistics of the conditioners' energies during sampling
    plot_conditioners_name_map = {'CoarseCenterDistC': 'Coarse-grained bead size',
                                  'BondLengthBoundsC': 'Bond length',
                                  'BondAngleEnddistBoundsC': 'Bond angle',
                                  'LJPDistMinC': 'LJ pairs distance',
                                  'LocalLJDistMinC': 'LJ pairs distance',
                                  'DiffusionC': 'Diffusion'}

    # Load the conditioners' energy timeseries
    dataset: GeneratedDataset = load_dataset('Diff_HA1_nup98_12_1_CondU')
    conditioners_U_stats_filepath = dataset.calc_dir / 'conditioners_U.npz'
    with np.load(conditioners_U_stats_filepath, allow_pickle=True) as npzfile:
        conditioners_U_stats = {k: npzfile[k] for k in npzfile}
    energies = conditioners_U_stats['energy']
    N, T, C = energies.shape

    # config.import_configs('config_' + dataset.configs['ID'])
    # noise_sch = init_noise_schedule()

    # Remove upper outliers
    energies_sorted = np.sort(energies, axis=0)
    n_omitted = int(0.001 * N)
    energies_truncated = energies_sorted[:-n_omitted]
    energies_avg = np.mean(energies_truncated, axis=0)
    energies_std = np.std(energies_truncated, axis=0)
    energies_SE = energies_std / np.sqrt(energies_truncated.shape[0])

    # Plot mean plus/min std
    fig = plt.figure(figsize=(0.75 * SI_fig_width, 0.5 * SI_fig_width))
    plot_mu_sigma(energies_avg, energies_std)
    # plot_mu_sigma(energies_avg, energies_std, x=noise_sch.Karras_t(noise_sch.t_arr)[:-1])
    plot_labels = [plot_conditioners_name_map[t] for t in conditioners_U_stats['type']]
    plt.legend(fig.gca().lines, plot_labels)
    plt.xlabel('Denoising step')
    # ax = fig.gca()
    # ax.xaxis.set_inverted(True)
    # plt.xlabel('Noise level [$\AA$]')
    # plt.xscale('log')
    plt.ylabel(f"Energy mean \u00B1 std (N={N:d}) [AU]")
    plt.xlim([0, T + 1])

    plot_filepath = pathlib.Path(SI_figs_dir, f"conditioners_energy_nup98_12.pdf")
    fig.savefig(plot_filepath, dpi=fig_dpi, bbox_inches='tight')
    plt.close(fig)


def plot_noise_schedules():
    """
    Plots noise schedules of the heavy-atom and coarse-grain models used in the paper.
    """
    from models.noise_schedules import init_noise_schedule

    config.import_configs(f"config_{main_configs_ID['CS']}")
    coarse_noise_sch = init_noise_schedule()
    coarse_noise_sch.name = 'coarse-structure model'

    config.import_configs(f"config_{main_configs_ID['HA']}")
    fine_noise_sch = init_noise_schedule()
    fine_noise_sch.name = 'fine-structure model'

    schedules = [coarse_noise_sch, fine_noise_sch]
    fig = plt.figure(figsize=(main_fig_widths['1col'], 0.75 * main_fig_widths['1col']))
    for sch in schedules:
        sch_h = sch.plot_attr_timeseries('Karras_t')
        sch_h.set_label(sch.name)

    plt.yscale('log')
    plt.ylabel(r'$\sigma(\tau)$ $[\AA]$')
    plt.xlabel(r'$\tau$')
    plt.ylim([1e-3, 1e2])
    plt.legend()
    plt.tight_layout(pad=0.1)
    plt.show(block=False)

    plot_filepath = SI_figs_dir / pathlib.Path(f'noise_schedules.pdf')
    fig.savefig(plot_filepath, dpi=fig_dpi)
    plt.close(fig)


def plot_noCP_model_failures():
    """
    Plots examples of dihedral angle distributions for datasets generated with models that do not have a cross-product module.
    Returns:

    """
    dih_angle_ind = [22, 36, 65, 96]
    gen_dataset: GeneratedDataset = load_dataset(gen_datasets_main['nup98_12'])
    gen_dataset_noCP_name = gen_datasets_main['nup98_12']
    gen_dataset_noCP_name = gen_dataset_noCP_name.replace(f"_{main_configs_ID['CS']}_",
                                                          f"_{main_configs_ID['CS']}_NoCP_")
    gen_dataset_noCP_name = gen_dataset_noCP_name.replace(f"_{main_configs_ID['HA']}_",
                                                          f"_{main_configs_ID['HA']}_NoCP_")
    gen_dataset_noCP: GeneratedDataset = load_dataset(gen_dataset_noCP_name)
    ref_dataset = load_dataset(gen_dataset.model_dataset)
    ref_dataset_splits = ref_dataset.load_splits()
    compared_datasets = ref_dataset_splits + [gen_dataset, gen_dataset_noCP]
    for d in compared_datasets:
        d.element_class = heavy_atoms_class_name
    dataset_labels = ref_dataset_splits[0].splits_name + ['model', 'model (no cross-product)']

    def return_fig_crit(fig):
        Xaxis_label = fig.gca().xaxis.get_label_text()
        return any(f'dihedral_angle_{i}' in Xaxis_label for i in dih_angle_ind)

    _, figs = compare_prop_dist(datasets=compared_datasets, properties_name=['dihedral_angle'], plot=True,
                                return_fig_crit=return_fig_crit)
    plt.show(block=False)
    figsize = (0.9 * SI_fig_width, 0.5 * SI_fig_width)
    fig_all, axes = plt.subplots(2, 2, figsize=figsize)
    for i, fig in enumerate(figs):
        # Copy axis content
        ax = fig.gca()
        axes_ind = np.unravel_index(i, shape=axes.shape)
        copy_ax(ax, axes[axes_ind])
        plt.close(fig)

        # Recalculate ylims
        ax = axes[axes_ind]
        ax.autoscale_view()
        ylim_top = max([c._y.max() for c in ax._children[:len(compared_datasets)]]) * 1.1
        ax.set_ylim(bottom=0.0, top=ylim_top)
        ax.set_yticks([])
        ax.set_xticks(np.linspace(-180, 180, 5))
        ax.set_xticklabels(np.linspace(-180, 180, 5).astype(int))
        ax.set_xlim([-180, 180])

        leg = ax.get_legend()
        if i == 0:
            ax.legend(dataset_labels, borderaxespad=0.5)
        else:
            leg.remove()

        # Remove y labels on all columns except the first one
        if axes_ind[1] > 0:
            ax.set_ylabel('')

    plt.tight_layout(pad=0.5)
    plt.show(block=False)
    plot_filepath = pathlib.Path(SI_figs_dir, f"NoCPModel_failures.pdf")
    fig_all.savefig(plot_filepath, dpi=fig_dpi)
    plt.close(fig_all)


def plot_overfitting_assessment():
    plot_filepath = pathlib.Path(SI_figs_dir, f"overfitting_assessment.pdf")
    fig, _ = plot_metrics_grid(list(gen_datasets_main.values()), metrics=['rmsd_min_train'], fig_width=SI_fig_width,
                               row_height=1.5)
    axes = fig.axes

    # Remove the row label
    row_label = [c for c in axes[0].get_children() if isinstance(c, matplotlib.text.Annotation)][0]
    row_label.remove()
    axes[0].set_xlabel(r'$\text{RMSD}_\text{train, min}$ [nm]')

    # Move the column labels down
    col_labels = [c for ax in axes for c in ax.get_children() if isinstance(c, matplotlib.text.Annotation)]
    for col_label in col_labels:
        col_label.set_y(5)

    # Remove the 'train' label in the legend.
    for h in axes[-1].get_legend_handles_labels()[0]:
        if h.get_label() == 'train':
            h.set_label('')
    axes[-1].legend()

    # Tighten layout
    gs = axes[0].get_gridspec()
    gs.tight_layout(fig, rect=[0, 0, 1, 1], pad=0.2, w_pad=0.25)

    fig.savefig(plot_filepath, dpi=fig_dpi)
    plt.close(fig)


def plot_main_UMAPs():
    plot_filepath = pathlib.Path(SI_figs_dir, f"UMAPs_main.pdf")
    plotted_datasets = list(gen_datasets_main.values())
    fig, _ = plot_metrics_grid(plotted_datasets, metrics=Biomolecule_UMAP_prop_names, fig_width=SI_fig_width,
                               row_height=0.75)
    axes = np.array(fig.axes).reshape(-1, len(plotted_datasets))

    # Set the vmax of all images to the global vmax
    vmax_global = max([ax.get_children()[0].get_clim()[1] for ax in axes.flatten()])
    for ax in axes.flatten():
        ax.get_children()[0].set_clim(0, vmax_global)
    im = axes[0, 0].get_children()[0]

    # Add colorbar for all plots
    margin_bottom = 0.05
    gs = axes[0, 0].get_gridspec()
    gs.update(bottom=margin_bottom, wspace=0.0, hspace=0.0)
    axes_leftmost_pos = min([a.get_position().bounds[0] for a in axes.flatten()])
    axes_rightmost_pos = max([a.get_position().bounds[0] + a.get_position().bounds[2] for a in axes.flatten()])
    cbar_width = 0.95 * (axes_rightmost_pos - axes_leftmost_pos)
    cbar_x_pos = (axes_rightmost_pos + axes_leftmost_pos) / 2 - cbar_width / 2
    cbar_height = 0.01
    cbar_y_pos = 0.025
    cbar_ax = fig.add_axes([cbar_x_pos, cbar_y_pos, cbar_width, cbar_height])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

    cbar.outline.set_visible(False)
    cbar.set_ticks([0.0, vmax_global])
    cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    fig.savefig(plot_filepath, dpi=fig_dpi)
    plt.close(fig)


def assemble_benchmark_stats():
    times = {'training_CS': {}, 'training_HA': {}, 'sampling_CS+HA_per_sample': {},
             'sampling_CS+HA_h_per_10ksample': {}}
    for sys in main_datasets:
        # Gather the training time of the coarse-grain model
        CS_train_stats = load_train_stats(config_ID=main_configs_ID['CS'], dataset=main_datasets[sys],
                                          split_set_ID='1')
        CS_train_time = np.sum(CS_train_stats['epoch_time']) / 3600  # hours

        # Gather the training time of the heavy-atom model
        HA_train_stats = load_train_stats(config_ID=main_configs_ID['HA'], dataset=main_datasets[sys],
                                          split_set_ID='1')
        HA_train_time = np.sum(HA_train_stats['epoch_time']) / 3600  # hours

        # Gather the sampling time of the generated dataset
        gen_dataset: GeneratedDataset = load_dataset(gen_datasets_main[sys], fetch_samples=True)
        sampling_time_tot = gen_dataset.sampling_time / 3600  # hours

        # Assemble all times
        times['training_CS'][sys] = CS_train_time
        times['training_HA'][sys] = HA_train_time
        times['sampling_CS+HA_per_sample'][sys] = sampling_time_tot / len(gen_dataset)
        times['sampling_CS+HA_h_per_10ksample'][sys] = np.round(sampling_time_tot / len(gen_dataset) * 10000, 3)

    times['README'] = {'training_CS': 'Training time (h) of the coarse-grain model for each system',
                       'training_HA': 'Training time (h) of the heavy-atom model for each system',
                       'sampling_CS+HA_per_sample': 'Sampling time (h) of coarse grains and heavy atoms per sample for each system. '
                                                    'The time is calculated by averaging the total time needed to sample 10000 samples.',
                       'sampling_CS+HA_h_per_10ksample': 'Sampling time (h) per 10k samples.'}

    # Save the times in a .json file
    times_filepath = pathlib.Path(config.data_dir, 'benchmarking', 'training_sampling_times.json')
    times_filepath.parent.mkdir(exist_ok=True, parents=True)
    with open(times_filepath, "w") as json_file:
        json.dump(times, json_file, indent=4)


def plot_psi_scan_assessments():
    psi_scan_metrics = ['V_bonds', 'V_angles', 'V_dih_prop+improp', 'V_dih_cmap', 'R_g', 'V_LJ_SR', 'V_coulomb_SR',
                        'V_tot']

    # Plot grid of metrics to analyze the effect of psi on fine-structure sampling
    plotted_datasets = list(psi_heavy_scan_gen_datasets.values())
    col_labels = [n.replace('psi', '\u03A8') for n in psi_heavy_scan_gen_datasets]
    plot_filepath = SI_figs_dir / 'psi_fine_scan_combined_metrics.pdf'
    plot_metrics_grid(plotted_datasets, metrics=psi_scan_metrics, filepath=plot_filepath, col_labels=col_labels,
                      fig_width=SI_fig_width)
    # plot_filepath = SI_figs_dir / 'psi_fine_scan_coarse_metrics.pdf'
    # plot_metrics_grid(plotted_datasets, metrics=coarse_metrics, filepath=plot_filepath, col_labels=col_labels,
    #                   fig_width=SI_fig_width)
    plot_filepath = SI_figs_dir / 'psi_fine_scan_fine_metrics.pdf'
    plot_metrics_grid(plotted_datasets, metrics=fine_metrics, filepath=plot_filepath, col_labels=col_labels,
                      fig_width=SI_fig_width)

    # Plot grid of metrics to analyze the effect of psi on coarse-structure sampling
    plotted_datasets = list(psi_coarse_scan_gen_datasets.values())
    col_labels = [n.replace('psi', '\u03A8') for n in psi_coarse_scan_gen_datasets]
    plot_filepath = SI_figs_dir / 'psi_coarse_scan_combined_metrics.pdf'
    plot_metrics_grid(plotted_datasets, metrics=psi_scan_metrics, filepath=plot_filepath, col_labels=col_labels,
                      fig_width=SI_fig_width)
    # plot_filepath = SI_figs_dir / 'psi_coarse_scan_coarse_metrics.pdf'
    # plot_metrics_grid(plotted_datasets, metrics=coarse_metrics, filepath=plot_filepath, col_labels=col_labels,
    #                   fig_width=SI_fig_width)


def plot_minimal_datasets_assessment():
    plotted_datasets = [gen_datasets_min[k] for k in main_datasets]
    plot_filepath = SI_figs_dir / 'minimal_datasets_coarse.pdf'
    plot_metrics_grid(plotted_datasets, fig_width=SI_fig_width, metrics=coarse_metrics, filepath=plot_filepath)

    plot_filepath = SI_figs_dir / 'minimal_datasets_fine.pdf'
    plot_metrics_grid(plotted_datasets, fig_width=SI_fig_width, metrics=fine_metrics, filepath=plot_filepath)


def plot_minimal_dataset_training_losses():
    metrics_name = ['loss_CS_train', 'loss_CS_valid', 'loss_HA_train', 'loss_HA_valid']
    metrics_label = {'loss_CS': 'Coarse-model loss', 'loss_HA': 'Fine-model loss'}
    metrics = defaultdict(dict)
    for sys in main_datasets:
        CS_train_stats = load_train_stats(config_ID=main_configs_ID['CS'], dataset=main_datasets[sys],
                                          split_set_ID='S1')
        HA_train_stats = load_train_stats(config_ID=main_configs_ID['HA'], dataset=main_datasets[sys],
                                          split_set_ID='S1')
        for metric_name in metrics_name:
            if '_CS' in metric_name:
                metrics[sys][metric_name] = CS_train_stats[metric_name.replace('_CS', '')]
            elif '_HA' in metric_name:
                metrics[sys][metric_name] = HA_train_stats[metric_name.replace('_HA', '')]

    # Compute fig size
    n_cols = len(main_datasets)
    n_rows = len(metrics_name) // 2
    fig_w = SI_fig_width
    fig_h_min = 2.0
    row_h_max = 0.75
    fig_h = fig_height_max
    fig_h = max(min(fig_h, n_rows * row_h_max), fig_h_min)
    figsize = (fig_w, fig_h)

    # Plot losses
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey='row', sharex='all', layout="constrained")
    fig.get_layout_engine().set(w_pad=0.025, h_pad=0.025, wspace=0.0, hspace=0.0)
    n_epochs = len(metrics[sys][metric_name])
    for i in range(n_rows):
        metric_name = metrics_name[2 * i].replace('_train', '')

        for j, sys in enumerate(main_datasets):
            ax: plt.Axes = axes[i, j]

            # Plot non-zero values of the train metric
            metric_train = metrics[sys][metrics_name[2 * i]]
            nonzero_ind = metric_train.nonzero()[0]
            epoch_value = nonzero_ind + 1
            ax.plot(epoch_value, metric_train[nonzero_ind], label='Train')

            # Plot non-zero values of the valid metric
            metric_valid = metrics[sys][metrics_name[2 * i + 1]]
            nonzero_ind = metric_valid.nonzero()[0]
            epoch_value = nonzero_ind + 1
            ax.plot(epoch_value, metric_valid[nonzero_ind], label='Valid')

        # Set y-axis limits
        ax.set_ylim(0.3, 1.0)

        # Format x-axis
        ax.set_xlim(0, None)
        ax.xaxis.set_ticks([0, n_epochs])
        ax.xaxis.set_ticks([n_epochs / 2], minor=True)
        ax.xaxis.set_major_formatter(epoch_tick_formatter)

        # Format y-axis
        axes[i, 0].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axes[i, 0].yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 1.5, 2, 2.5, 4, 5, 10]))

    # Add legend and epoch label
    axes[0, -1].legend()
    axes[-1, 0].set_xlabel('Epoch', labelpad=-4.0)

    # Add col labels
    for i, sys in enumerate(main_datasets):
        axes[0, i].set_title(datasets_fig_label[sys], va='top')

    # Add row labels
    for i, metric_name_train in enumerate(metrics_name[::2]):
        metric_name = metric_name_train.replace('_train', '')
        axes[i, 0].annotate(metrics_label[metric_name], xy=(0, 0.5), xytext=(-20, 0), fontsize=7,
                            xycoords='axes fraction', textcoords='offset points', va='center', ha='center',
                            rotation='vertical')
    # fig.canvas.draw()
    # fig.tight_layout(pad=0.2)

    # Save the plot
    plot_filepath = pathlib.Path(SI_figs_dir, f"minimal_datasets_losses.pdf")
    fig.savefig(plot_filepath, dpi=fig_dpi)
    plt.close(fig)


def plot_thermalized_datasets_assessment():
    plotted_datasets = [thermalized_gen_datasets[k] for k in main_datasets]
    plot_filepath = SI_figs_dir / 'thermalized_datasets_coarse.pdf'
    plot_metrics_grid(plotted_datasets, fig_width=SI_fig_width, metrics=coarse_metrics, filepath=plot_filepath)

    plot_filepath = SI_figs_dir / 'thermalized_datasets_fine.pdf'
    plot_metrics_grid(plotted_datasets, fig_width=SI_fig_width, metrics=fine_metrics, filepath=plot_filepath)


def calculate_nup_capped_rama_JSdist(binwidth: int = 1):
    assert isinstance(binwidth, int), "The binwidth in degrees must be an integer value."
    gen_dataset_charmm: GeneratedDataset = load_dataset(gen_datasets['nup98C_12']).prune()
    gen_dataset_amber: GeneratedDataset = load_dataset(gen_datasets['nup98AC_12']).prune()
    dataset_charmm = load_dataset(gen_dataset_charmm.model_dataset)
    dataset_amber = load_dataset(gen_dataset_amber.model_dataset)
    dataset_charmm_train, dataset_charmm_test = dataset_charmm.load_splits(['train', 'test'])
    dataset_amber_train, dataset_amber_test = dataset_amber.load_splits(['train', 'test'])
    compared_datasets = [dataset_charmm_train, dataset_charmm_test, dataset_amber_train, dataset_amber_test,
                         gen_dataset_charmm, gen_dataset_amber]
    for d in compared_datasets:
        d.element_class = heavy_atoms_class_name
        if isinstance(d, GeneratedDataset):
            d.comparison_name = d.model_dataset + '_model'
        else:
            d.comparison_name = d.main_name + f"_{d.split_ID[1]}"

    JS_dists = {}
    for d1, d2 in itertools.combinations(compared_datasets, 2):
        s1, s2 = d1.statistics('backbone_dihedrals'), d2.statistics('backbone_dihedrals')
        # Downsample bins
        if not s1.bin_widths[0] == binwidth:
            s1.downsample_bins(binwidth)
        if not s2.bin_widths[0] == binwidth:
            s2.downsample_bins(binwidth)
        JS_dist = s1.prob_dist(s2, dist_type='JSD')
        JS_dists[(d1.comparison_name, d2.comparison_name)] = JS_dist.item()
    return JS_dists


def plot_capped_systems_metrics():
    plotted_datasets = [gen_datasets['nup98C_12'], gen_datasets['nup98AC_12']]
    panel_width = (SI_fig_width - 0.75) / 2
    unequal_ar_panel_height = 0.3
    interpad = '150%'
    row_label_size = 6

    plot_filepath = SI_figs_dir / 'nup98_capped_coarse_metrics.pdf'
    plot_metrics_grid(plotted_datasets, metrics=coarse_metrics, filepath=plot_filepath, fig_width=panel_width,
                      row_height=unequal_ar_panel_height, inter_pad=interpad, row_label_size=row_label_size)

    plot_filepath = SI_figs_dir / 'nup98_capped_fine_metrics.pdf'
    # fig, axes = plot_metrics_grid(plotted_datasets, metrics=fine_metrics, filepath=plot_filepath, fig_width=panel_width,
    #                               row_height=0.4, inter_pad='150%')
    fig, axes = plot_metrics_grid(plotted_datasets, metrics=fine_metrics, filepath=None, fig_width=panel_width,
                                  row_height=unequal_ar_panel_height, inter_pad=interpad, row_label_size=row_label_size)

    # Find axes that correspond to Ramachandran maps
    rama_axes = []
    rama_axes_dataset_type = []
    dataset_types = ['train', 'test', 'model']
    for i in range(axes.shape[0]):
        ax = axes[i, 0]
        texts = [c.get_text() for c in ax.get_children() if isinstance(c, matplotlib.text.Text)]
        rama_texts = [t for t in texts if 'Ramachandran' in t]
        if rama_texts:
            dataset_type = [t for t in dataset_types if any(t in text for text in rama_texts)][0]
            rama_axes.append(axes[i, :])
            rama_axes_dataset_type.append(dataset_type)
    rama_axes = np.array(rama_axes)

    # Calculate JS distances
    JSD_dists = calculate_nup_capped_rama_JSdist()

    # Add dataset name synonyms
    model_datasets = [load_dataset(n) for n in plotted_datasets]

    # Add vertical arrows indicating JS dists across dataset types for the same system
    for j in range(rama_axes.shape[1]):
        for i in range(rama_axes.shape[0] - 1):
            ax1, ax2 = rama_axes[i, j], rama_axes[i + 1, j]
            ax1_name = f"{model_datasets[j].model_dataset}_{rama_axes_dataset_type[i]}"
            ax2_name = f"{model_datasets[j].model_dataset}_{rama_axes_dataset_type[i + 1]}"
            JS_dist_temp = JSD_dists[(ax1_name, ax2_name)]
            con = matplotlib.patches.ConnectionPatch(
                xyA=(0.5, 0.0), xyB=(0.5, 1.0), coordsA="axes fraction", coordsB="axes fraction", axesA=ax1, axesB=ax2,
                arrowstyle="<|-|>", color="black")
            ax1.add_artist(con)

            ax1.annotate(f"{JS_dist_temp:.2f}", xy=(0.5, 0.5), xycoords=con, xytext=(10, 0), textcoords="offset points",
                         ha="center", va="center", fontsize=5)

    # Add horizontal arrows indicating JS dists across system for same dataset type
    for j in range(rama_axes.shape[1] - 1):
        for i in range(rama_axes.shape[0]):
            ax1, ax2 = rama_axes[i, j], rama_axes[i, j + 1]
            ax1_name = f"{model_datasets[j].model_dataset}_{rama_axes_dataset_type[i]}"
            ax2_name = f"{model_datasets[j + 1].model_dataset}_{rama_axes_dataset_type[i]}"
            JS_dist_temp = JSD_dists[(ax1_name, ax2_name)]
            con = matplotlib.patches.ConnectionPatch(
                xyA=(1.0, 0.5), xyB=(0.0, 0.5), coordsA="axes fraction", coordsB="axes fraction", axesA=ax1, axesB=ax2,
                arrowstyle="<|-|>", color="black")
            ax1.add_artist(con)

            ax1.annotate(f"{JS_dist_temp:.2f}", xy=(0.5, 0.5), xycoords=con, xytext=(0, 5), textcoords="offset points",
                         ha="center", va="center", fontsize=5)
    fig.savefig(plot_filepath, dpi=fig_dpi)
    plt.close(fig)


def plot_CLN025_experiments():
    filepath = SI_figs_dir / 'CLN025_experiments.pdf'

    cutoffs = ['2.5', '3', '4', '5', '6']
    fold_cutoff = 3

    model_label_kwargs, train_label_kwargs, test_label_kwargs = {'label': 'model', 'color': '#DC267F'}, {
        'label': 'train', 'color': '#FFB000'}, {'label': 'test', 'color': '#648FFF'}
    md_label_kwargs = {'label': 'MD', 'color': '#117733'}

    fig = plt.figure(figsize=(5.48, 5.68), dpi=300)

    gs = gridspec.GridSpec(len(cutoffs), 2)

    def get_bincenters(binedges):
        """Bin centers from (possibly nested) bin edges."""
        if not np.ndim(binedges) == 1:
            binedges = binedges[0]
        return 0.5 * (binedges[:-1] + binedges[1:])

    # Plot DEShaw RMSDs
    ax = plt.subplot(gs[0, 0])

    main_CLN_dataset_name = main_datasets['CLN']
    DEShaw_dataset = load_dataset(main_CLN_dataset_name)
    DEShaw_dataset.element_class = 'heavy_atoms'
    DEShaw_data = DEShaw_dataset.statistics('rmsd_min_2rvd')
    DEShaw_bins, DEShaw_pdf_tup = DEShaw_data.binedges, DEShaw_data.pdf
    DEShaw_pdf = DEShaw_pdf_tup[0][0]
    DEShaw_bincenters = get_bincenters(DEShaw_bins) * 10

    def calc_pop_folded(bins, pdf, index=61):
        bincenters = get_bincenters(bins)
        binwidths = float(bincenters[1] - bincenters[0])
        cutoff_pdf = np.array(pdf)[0:index]
        pop_folded = np.sum(cutoff_pdf * binwidths)
        return np.round(pop_folded, 3)

    MD_pop_folded = calc_pop_folded(DEShaw_bins, DEShaw_pdf)
    print(f'MD pop. folded:{MD_pop_folded}')

    ax.plot(DEShaw_bincenters, DEShaw_pdf, **md_label_kwargs)
    ax.set_xlabel(r'$\mathrm{RMSD}_{\mathrm{min}}$ [Å]')
    ax.set_yticks([])
    ax.set_ylim(0, None)
    ax.set_xlim(0, 10)
    ax.legend(loc='upper right', borderpad=0.0, borderaxespad=0.5, labelspacing=0.2)

    panel_label_xy = (-0.05, 1.15)
    panel_label_size = 10
    text = 'a'
    ax.annotate(text, xy=panel_label_xy, fontsize=panel_label_size, xycoords='axes fraction', fontweight='bold')

    # Plot min RMSD for main CLN025 dataset
    ax = plt.subplot(gs[1, 0])

    dataset_model_name = gen_datasets['CLN']
    dataset_model = load_dataset(name=dataset_model_name, element_class='heavy_atoms').prune()

    main_CLN_dataset = load_dataset(main_CLN_dataset_name, element_class='heavy_atoms')
    dataset_train = main_CLN_dataset.load_splits(['train'], set_ID='1')[0]
    dataset_test = main_CLN_dataset.load_splits(['test'], set_ID='1')[0]

    metric = 'rmsd_min_2rvd'
    model_bins, model_pdf_tup = dataset_model.statistics(metric).binedges, dataset_model.statistics(metric).pdf
    train_bins, train_pdf_tup = dataset_train.statistics(metric).binedges, dataset_train.statistics(metric).pdf
    test_bins, test_pdf_tup = dataset_test.statistics(metric).binedges, dataset_test.statistics(metric).pdf

    model_pdf, train_pdf, test_pdf = model_pdf_tup[0][0], train_pdf_tup[0][0], test_pdf_tup[0][0]

    # model_pop_folded = calc_pop_folded(model_bins, model_pdf)
    # train_pop_folded = calc_pop_folded(train_bins, train_pdf)
    # test_pop_folded = calc_pop_folded(test_bins, test_pdf)
    # print(f'model pop. folded:{model_pop_folded}')
    # print(f'train pop. folded:{train_pop_folded}')
    # print(f'test pop. folded:{test_pop_folded}')

    model_bincenters, train_bincenters, test_bincenters = get_bincenters(model_bins) * 10, get_bincenters(
        train_bins) * 10, get_bincenters(test_bins) * 10

    ax.plot(model_bincenters, model_pdf, **model_label_kwargs)
    ax.plot(train_bincenters, train_pdf, **train_label_kwargs)
    ax.plot(test_bincenters, test_pdf, **test_label_kwargs)

    ax.set_xlabel(r'$\mathrm{RMSD}_{\mathrm{min}}$ [Å]')
    ax.set_ylim(-0.0001, None)
    ax.set_xlim(0, 10)
    ax.set_yticks([])
    text = 'b'
    ax.annotate(text, xy=panel_label_xy, fontsize=panel_label_size, xycoords='axes fraction', fontweight='bold')
    ax.legend(loc='upper right', borderpad=0.0, borderaxespad=0.5, labelspacing=0.2)

    # Plot min rmsds of filtered datasets
    for cutoff in cutoffs:
        if cutoff in ['4', '5', '6']:
            fold_type = 'Unfolded'  # Specifies if the training data was unfolded or folded
            fold_letter = 'U'
        elif cutoff in ['2.5', '3']:
            fold_type = 'Folded'
            fold_letter = 'F'

        cutoff_float = float(cutoff)
        sub_dataset_name = chig_exp_gen_datasets[f'{fold_type}_HA_{cutoff}A_in_train']

        sub_model_dataset = load_dataset(name=sub_dataset_name, element_class='heavy_atoms').prune()

        main_CLN_dataset = load_dataset(main_CLN_dataset_name, element_class='heavy_atoms')
        if cutoff == '2.5':
            filtered_dataset = main_CLN_dataset.load_splits(['train'], set_ID=f'{fold_letter}inTr2_5')[0]
        else:
            filtered_dataset = main_CLN_dataset.load_splits(['train'], set_ID=f'{fold_letter}inTr{cutoff}')[0]

        ax = plt.subplot(gs[cutoffs.index(cutoff), 1])

        model_bins, model_pdf_tup = sub_model_dataset.statistics(metric).binedges, sub_model_dataset.statistics(
            metric).pdf
        train_bins, train_pdf_tup = filtered_dataset.statistics(metric).binedges, filtered_dataset.statistics(
            metric).pdf

        model_pdf, train_pdf = model_pdf_tup[0][0], train_pdf_tup[0][0]

        model_bincenters, train_bincenters = get_bincenters(model_bins) * 10, get_bincenters(train_bins) * 10

        train_bin_width = train_bincenters[1] - train_bincenters[0]
        train_bincenters = np.concatenate(
            [[train_bincenters[0] - train_bin_width], train_bincenters, [train_bincenters[-1] + train_bin_width]])
        train_pdf = np.concatenate([[0], train_pdf, [0]])

        ax.plot(model_bincenters, model_pdf, **model_label_kwargs)
        ax.plot(train_bincenters, train_pdf, **train_label_kwargs)

        ref_dataset = load_dataset('PDB_2RVD', element_class='heavy_atoms')
        model_rmsds = calculate_rmsd_min(sub_model_dataset, ref_dataset, checkpoint_filepath='rmsd_min_2rvd')
        model_rmsds = model_rmsds * 10

        if fold_type == 'Unfolded':
            ax.set_title('Training set ' + r' $\mathrm{RMSD}_{\mathrm{min}}$' f' > {np.round(cutoff_float, 1)} Å')
            model_unlike_count = sum(i < fold_cutoff for i in model_rmsds)
            text = rf'$N_{{<{fold_cutoff}\mathrm{{\AA}}}}$' + f'={int(model_unlike_count)}'
            data_cutoff = fold_cutoff
            xy = (0.08, 0.5)
        elif fold_type == 'Folded':
            ax.set_title('Training set' + r' $\mathrm{RMSD}_{\mathrm{min}}$' + f' < {np.round(cutoff_float, 1)} Å')
            model_unlike_count = sum(i > fold_cutoff for i in model_rmsds)
            text = rf'$N_{{>{fold_cutoff}\mathrm{{\AA}}}}$' + f'={int(model_unlike_count)}'
            data_cutoff = fold_cutoff
            xy = (0.5, 0.5)

        ax.annotate(text, xy=xy, color=model_label_kwargs['color'], fontsize=6, xycoords='axes fraction')
        ax.axvline(data_cutoff, linestyle='dashed', alpha=0.25, color='black')
        ax.set_ylim(-0.0001, None)
        ax.set_xlim(0, 10)

        if cutoff == '2.5':
            ax.legend(loc='upper right', borderpad=0.0, borderaxespad=0.5, labelspacing=0.2)
            text = 'c'
            ax.annotate(text, xy=panel_label_xy, fontsize=panel_label_size, xycoords='axes fraction', fontweight='bold')
        if cutoff != '6':
            # ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(axis='x', labelbottom=False)
        if cutoff == '6':
            ax.set_xlabel(r'$\mathrm{RMSD}_{\mathrm{min}}$ [Å]')
            # ax.set_ylabel('PDF')
            ax.set_yticks([])

    ax = plt.subplot(gs[2, 0])
    text = 'd'
    ax.annotate(text, xy=panel_label_xy, fontsize=panel_label_size, xycoords='axes fraction', fontweight='bold')
    ax.axis('off')

    ax = plt.subplot(gs[3, 0])
    text = 'e'
    panel_label_xy = (-0.05, 0.1)
    ax.annotate(text, xy=panel_label_xy, fontsize=panel_label_size, xycoords='axes fraction', fontweight='bold')
    ax.axis('off')

    plt.tight_layout(h_pad=0.001, w_pad=0.05)
    fig.savefig(filepath, format='pdf', pad_inches=0)
    plt.close(fig)


load_paper_mpl_configs()

# # Create .pdb of example molecules for paper
# fig1_gen_dataset = load_dataset('DiffGraph2_NU_CS3_182_NU_HA_172_E3', force_download=True)
# fig1_gen_dataset.save_to_pdb()
# samples_rmsd = torch.tensor([mol.R_g for mol in fig1_gen_dataset])
# max_Rg_ind = torch.argmax(samples_rmsd)
# nup98_AA = load_dataset('nup98_12mer_AA_411')
# nup98_AA.save_to_pdb(ind=[0, 1, 2, 3, 4])

# Assemble timing stats for benchmarking
# assemble_benchmark_stats()

# Plot the noise schedules used for the coarse-structure and fine-structure model
# plot_noise_schedules()

# Plot the tracked metrics of each trained model (both heavy-atom and coarse grain)
# plot_training_metrics('CS')
# plot_training_metrics('HA')

# Plot training losses for models trained on minimal datasets
# plot_minimal_dataset_training_losses()

# Save a plot of the learning rate schedule for the heavy-atom and coarse-grain model
# plot_lr_schedules()

# Plot timeseries of the conditioners energy for the nup98_12mer system
# plot_conditioner_energies()

# Plot examples of failures for models that do not have a cross-product module
# plot_noCP_model_failures()

# Plot the minimum rmsd to the train split to assessment model's overfitting
# plot_overfitting_assessment()

# Plot UMAPs for the main datasets
# plot_main_UMAPs()

# Calculate the JS distance between various pairs of Ramachandran maps of the nup98-12 capped systems
# plot_capped_systems_metrics()

# Plot grid of metrics to analyze the effect of psi
# plot_psi_scan_assessments()

# Plot coarse and fine metrics of minimal datasets
# plot_minimal_datasets_assessment()

# Plot coarse and fine metrics of minimal datasets
# plot_thermalized_datasets_assessment()

# Plot CLN025 experiments
# plot_CLN025_experiments()

# Print the fraction of pruned samples for each of the main GeneratedDatasets
# for k, v in gen_datasets_main.items():
#     dataset: GeneratedDataset = load_dataset(v)
#     print(k, f"{dataset.pruning_crit['PR'].rejection_ratio:.2%}")
# pass
