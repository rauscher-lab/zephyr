from data.datasets import load_dataset
from data.data_classes import heavy_atoms_class_name
from analysis.comparisons import compare_prop_dist, compare_datasets

if __name__ == '__main__':
    # Load heavy atoms version of the nup98_12mer dataset
    nup98_dataset = load_dataset(name='nup98_12', element_class=heavy_atoms_class_name)

    # Load a dataset of generated samples. Generated datasets are located in config.gen_datasets_dir
    gen_dataset = load_dataset(name='Diff_CS1_HA1_nup98_12_1_C1H1', element_class=heavy_atoms_class_name)

    # Load splits of the dataset
    nup98_dataset_train, nup98_dataset_valid = nup98_dataset.load_splits(['train', 'valid'])

    # Load dictionary of statistics of the datasets
    # Each value in the dictionary is a Statistics object (see analysis.stats.Statistics)
    nup98_dataset_stats = nup98_dataset.statistics()
    nup98_dataset_train_stats = nup98_dataset_train.statistics()

    # Compare distributions of all properties for a dataset and a GeneratedDataset
    compare_datasets([nup98_dataset, gen_dataset])

    # Compare specific properties
    compare_prop_dist([nup98_dataset_train, gen_dataset], properties_name=['R_g', 'backbone_bond_lengths'])
