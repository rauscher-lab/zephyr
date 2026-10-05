# Zephyr

A generative diffusion model for sampling disordered protein structures trained on MD datasets.

## Introduction

This codebase provides workflows to train a generative graph neural network (GNN) for the task of sampling conformations of disordered proteins.
The generative model consists of 2 separate networks: 1) A GNN to sample coarse-grained protein structures and 2)
another GNN to sample heavy atom positions with a given coarse-grained structure.
The coarse-grained structure consists of a backbone bead (centered on the alpha-carbon atom) and a sidechain bead for each residue in the sequence.
The models are trained on structures from molecular dynamics (MD) trajectories.
Each model is trained on MD trajectories of a single protein sequence.

## Installation
### Option 1: Install packages with pip
1. Open a terminal and navigate to the root directory of this repository.
2. Create a virtual environment using `python=3.11`.
3. Install required packages with pip:
```
pip install -r requirements.txt
```

### Option 2: Install environment with conda
Create a conda environment using the `.yml` file:
```
conda env create -f requirements.yml
```
If training with cuda device is required, reinstall pytorch with gpu support:
```
conda install -c nvidia -c pytorch pytorch pytorch-cuda torchvision --force-reinstall
```

## Configs

`config.py` holds variables that define global configurations.
The default values can be overwritten by modifying the config variables in `config.py` or by defining
a `config_defaults.py` located in the `configs` subdirectory.
Configuration values for training and sampling with specific models must be stored in the `./configs` directory.
Since configurations are `.py` files, one can import modules and use complex logic to define the configuration variables.
However, at the end of the config file, one must delete all temporary variables to make sure that the module namespace
contains only variables that are defined as configuration.

Four configuration files are included in the repository:

* `config_CS1.py`: configuration of the coarse-structure model
* `config_CS1_NoCP.py`: configuration of the coarse-structure model that doesn't have cross-product modules.
* `config_HA1.py`: configuration of the fine-structure model
* `config_HA1_NoCP.py`: configuration of the fine-structure model that doesn't have cross-product modules.

These configuration files were used to train models and sample datasets analyzed in the associated publication.

To train without wandb logging, set `wandb = False` in your config file.

## Datasets

Three kinds of datasets are associated with the codebase: MD, processed and generated datasets.
The datasets are stored on [Zenodo](https://zenodo.org/records/20492056).
Functions in `data.download` download a dataset as it is needed.
Alternatively, one can download the dataset `.zip` files and extract them in a given directory.
The config variables `MD_datasets_dir`, `proc_datasets_dir` and `gen_datasets_dir` specify the directories where the MD,
processed and generated datasets are stored, respectively.
Once datasets are downloaded, use the function `load_dataset(name)` in `data/datasets.py` to load a dataset named `name`.
The name of a dataset matches the name of its directory.
For example, `load_dataset('nup98_12')` will load data from the processed dataset directory `nup98_12` located in the
directory defined by `config.proc_datasets_dir`.

## Weights

Model weights are stored on [Zenodo](https://zenodo.org/records/20492056) in the `model_weights.zip` file.
When specific model weights are needed for retraining or sampling, they are fetched from the Zenodo repository.
Alternatively, one can download all model weights and store them in the directory defined by `config.weights_dir`.

## Training

Model training is achieved by calling `train.py` with a given set of configurations.
For example,

```
python train.py --configs my_config.py
```

will train a model using configurations defined in `my_config.py`.
Configuration values can also be overwritten when calling `train.py`:

```
python train.py --configs my_config.py --batch_size 32 --N_workers 2
```

will use a batch size of 32 and 2 workers, regardless of the values defined in `my_config.py`.

One can also change the dataset used to train the model and its particular set of `train/valid/test` splits as follows:
```
python train.py --configs config_CS1.py --dataset nup98_12 --dataset_split_set_ID 1
```

See `run_all.sh` for a summary of all training runs associated with the publication.

## Sampling

Sampling with a given model is carried out with the `models.generate` module. For example,

```
python -m models.generate --ID_coarse C1 --ID_fine H1 --dataset nup98_12 --configs_coarse config_CS1.py --configs_fine config_HA1.py
```

will generate structures with the same topology as the `nup98_12` dataset using `config_CS1.py` for the coarse-structure
model configurations and `config_HA1.py` for the fine-structure model configurations.
The ID of the coarse-structure sampling is `C1` and the ID of the fine-structure sampling is `H1`.
The dataset of generated structures will be stored in the directory defined by `config.gen_datasets_dir`.

See `run_all.sh` for a summary of all sampling runs associated with the publication.

### Denoising Trajectories

Examples of denoising trajectories for sampling the coarse structure and fine structure of the `nup98_12` sequence are shown.

<img width="400" height="450" alt="Image" src="https://github.com/user-attachments/assets/d8cc397a-ffa6-4b90-9a5e-239e9b1bad8f" />

## Analysis

See `./analysis/analysis_tutorial.py` for an example of various ways to compare properties between datasets.

## Figures

Modules in the `figures` directory contain plotting functions to recreate the figures of the associated publication.

## License

This project is licensed under the MIT License — see the license file for details.

## References

Olivier Trottier, Jack H. Gwozdecky and Sarah Rauscher (2026) "Generating Structural Ensembles of Disordered Proteins with Diffusion Models" bioRxiv.
