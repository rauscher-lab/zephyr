from data.datasets import gen_datasets_main
from figures import load_paper_mpl_configs, main_figs_dir, plot_metrics_grid, coarse_metrics, fine_metrics, \
    sampling_metrics, main_fig_width, plot_sampling_metrics

load_paper_mpl_configs()

plotted_dataset = list(gen_datasets_main.values())

plot_filepath = main_figs_dir / 'figure3.pdf'
plot_metrics_grid(plotted_dataset, metrics=fine_metrics, filepath=plot_filepath, fig_width=main_fig_width)

plot_filepath = main_figs_dir / 'figure4.pdf'
plot_metrics_grid(plotted_dataset, metrics=coarse_metrics, filepath=plot_filepath, fig_width=main_fig_width)

plot_filepath = main_figs_dir / 'figure5.pdf'
plot_sampling_metrics(plotted_dataset, metrics=sampling_metrics, filepath=plot_filepath)
