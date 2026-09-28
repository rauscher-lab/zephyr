import pathlib
import shutil
import sys
import warnings
import argparse
import psutil
import uuid

import torch

import numpy as np
import config
from utils import Timer
from analysis.comparisons import compare_datasets
from analysis.metrics import analyze_dataset
from data.datasets import BiomoleculeDataset, GeneratedDataset, load_dataset, ProcessedDataset, \
    DenoisingTrajectoriesDataset, dataset_tempdir
from data.data_classes import element_classes, heavy_atoms_class_name
from visualization import make_denoising_movie
import models  # Can't do "from models import ..." since kwargs defaults depending on global configs won't update.


def generate_samples(config_name='', overwritten_configs: dict = None, model=None, model_epoch='',
                     T=None, ref_dataset: BiomoleculeDataset | str = None, ref_dataset_split: str = None,
                     n_samples=None, ID='', regen=False, batch_size: int = None, samples_dir=None,
                     save_trajectory=False, save_cond_U=False) -> GeneratedDataset | DenoisingTrajectoriesDataset:
    """
    Generates a dataset of structures using a trained model
    Args:
        config_name: name of the configuration file used to define the model
        overwritten_configs: configuration values that are overwritten after loading the configuration file
        model: DiffGraph model used to generate the samples. If none, initializes model with models.load()
        model_epoch: specific epoch value that determines the model state that will be used. Defaults to last epoch.
        T: Number of diffusion step to use for generating samples. Defaults to config.n_diff_steps
        ref_dataset: reference dataset used to define the structure's topology and initial coarse grain positions (when sampling heavy-atoms)
        ref_dataset_split: specific split of the reference dataset to use. Defaults to using no splits
        n_samples: number of samples to generate
        ID: ID of the generated samples (included in GeneratedDataset directory name)
        regen: delete previous samples and regenerate them
        batch_size: number of samples generated in one batch
        samples_dir: directory where samples are saved. Defaults to directory generated with generate_dir()
        save_trajectory: save denoising trajectories of the generated samples
        save_cond_U: save energies of the conditioners throughout the denoising trajectories

    Returns:
        dataset of generated structures
    """

    # Import configs if any are given
    if config_name or overwritten_configs:
        config.import_configs(config_name, overwritten_configs=overwritten_configs)
        if config_name:
            print(f'Starting sampling of {ID!r} with the following configs:')
            config.print_configs()

    # Determine the samples directory
    if not ID:
        ID = uuid.uuid4().hex[:8]

    if samples_dir is None:
        if save_trajectory:
            samples_dir = DenoisingTrajectoriesDataset.generate_dir(ID=ID, model_epoch=model_epoch)
        else:
            samples_dir = GeneratedDataset.generate_dir(ID=ID, model_epoch=model_epoch)
    else:
        samples_dir = pathlib.Path(samples_dir)

    # Remove previous samples directory if regenerating. Otherwise, load the GeneratedDataset and return it.
    if samples_dir.exists() and regen:
        shutil.rmtree(samples_dir)

    # Return samples if it has already been generated.
    if BiomoleculeDataset.is_processed(samples_dir):
        warnings.warn(f"Samples already exists in {str(samples_dir)!r}.\nSkipping generation.")
        if save_trajectory:
            gen_dataset = DenoisingTrajectoriesDataset(directory=samples_dir)
        else:
            gen_dataset = GeneratedDataset(directory=samples_dir)
        return gen_dataset

    # Load the dataset if None or a name is given
    if ref_dataset is None or isinstance(ref_dataset, str):
        ref_dataset = load_dataset(name=ref_dataset)

    # Check the element class of the reference dataset and refine molecules to match the generated element_class
    if element_classes.index(config.dataset_element_class) > element_classes.index(ref_dataset.element_class):
        ref_dataset_refined_dir = pathlib.Path(dataset_tempdir, ref_dataset.name + '_refined')
        if regen and ref_dataset_refined_dir.exists():
            shutil.rmtree(ref_dataset_refined_dir)
        if not ref_dataset_refined_dir.exists():
            ref_dataset = ref_dataset.refine(directory=ref_dataset_refined_dir,
                                             element_class=config.dataset_element_class)
        else:
            ref_dataset = ref_dataset.__class__(directory=ref_dataset_refined_dir)
    else:
        ref_dataset_refined_dir = None

    # Set the element class of the reference dataset
    ref_dataset.element_class = config.dataset_element_class

    # If the reference dataset is a DenoisingTrajectoriesDataset, use only molecules corresponding to last frames.
    if isinstance(ref_dataset, DenoisingTrajectoriesDataset):
        mol_ind, time_ind = ref_dataset.mol_time_indices
        mol_retained_ind = torch.nonzero(time_ind.eq(time_ind.max()))[:, 0]
        mol_ind, time_ind = mol_ind[mol_retained_ind], time_ind[mol_retained_ind]
        mol_retained_ind = mol_retained_ind[torch.argsort(mol_ind)]
        ref_dataset = ref_dataset.subset(ind=mol_retained_ind, inplace=False)

    # Select the requested split of the reference dataset
    if ref_dataset_split is not None:
        ref_dataset = ref_dataset.load_splits(ref_dataset_split)

    # Load the model and its trained weights if no model is given
    model_dataset = config.dataset
    if model is None:
        model = models.load(dataset=model_dataset, epoch=model_epoch)
        model.to(torch.device(config.device))

        # Estimate statistics for dynamic denoising
        if config.dynamic_denoising:
            if isinstance(ref_dataset, GeneratedDataset):
                stats_dataset = load_dataset(ref_dataset.model_dataset, element_class=config.dataset_element_class)
                stats_dataset = stats_dataset.load_splits('train')
            else:
                stats_dataset = ref_dataset
            stats_dataset.load_all()
            stats_dataset = ProcessedDataset(stats_dataset, model=model, device=model.device)
            model.diff_model.estimate_U_tot_stats(dataset=stats_dataset)

            # Delete stats dataset to free-up memory
            del stats_dataset

    # Determine the number of samples
    if n_samples is None:
        n_samples = config.n_samples
    elif n_samples in ['train', 'valid', 'test']:
        n_samples = len(load_dataset(config.dataset).load_splits(n_samples))
    elif n_samples == 'dataset':
        n_samples = len(ref_dataset)
    elif isinstance(n_samples, int):
        pass
    else:
        raise ValueError(f"n_samples value ({n_samples}) is not supported.")

    # Calculate an appropriate batch size based on the memory footprint of one molecule passed through the network
    mem_frac = 0.9
    if model.device.type == 'cuda':
        memory_avail = mem_frac * torch.cuda.mem_get_info(model.device)[0]  # Define avail. memory from free GPU memory
    else:
        memory_avail = mem_frac * psutil.virtual_memory().available  # Define avail. memory from free RAM
    if batch_size is not None:
        pass
    elif n_samples < 200:
        batch_size = n_samples
    else:
        model_element_size = torch.tensor(model.element_feature_sizeof)
        model_mol_size = model_element_size * (ref_dataset[0].n_elements ** 2) / model.n_layers.item()
        batch_size = int(torch.ceil(memory_avail / model_mol_size))
    batch_size = min(batch_size, 2000)

    # Generate samples
    mols_temp_dir = samples_dir / 'molecules_temp'
    timer = Timer(checkpoint_filepath=samples_dir / 'sampling_time_stats', resume=True)
    sampling_output = model.sample(n_samples, T=T, batch_size=batch_size, ref_dataset=ref_dataset, verbose=True,
                                   temp_dir=mols_temp_dir, timer=timer, save_trajectory=save_trajectory,
                                   save_cond_U=save_cond_U)
    if save_cond_U:
        molecules_sample = sampling_output[0]
    else:
        molecules_sample = sampling_output

    # Save the generated molecules along with useful metadata
    model_split_ID = (config.dataset_split_set_ID, 'train')
    configs_dict = {k: v for k, v in config.get_configs_dict().items() if config.is_config_logged(k)}
    ref_dataset_name = ref_dataset.name
    if ref_dataset_refined_dir is not None:
        ref_dataset_name = ref_dataset_name.removesuffix('_refined')

    if save_trajectory:
        gen_dataset = DenoisingTrajectoriesDataset.from_molecules(molecules_sample, directory=samples_dir,
                                                                  configs=configs_dict,
                                                                  ref_dataset=ref_dataset_name,
                                                                  model_dataset=model_dataset,
                                                                  model_split_ID=model_split_ID,
                                                                  sampling_time=timer.elapsed_time,
                                                                  noise_sch_t=model.diff_model.t_steps.cpu())
    else:
        gen_dataset = GeneratedDataset.from_molecules(molecules_sample, directory=samples_dir,
                                                      configs=configs_dict,
                                                      ref_dataset=ref_dataset_name,
                                                      model_dataset=model_dataset,
                                                      model_split_ID=model_split_ID,
                                                      sampling_time=timer.elapsed_time)
    timer.delete_checkpoint()
    gen_dataset.inherit_metadata(ref_dataset, omitted_attrs=['configs', 'ref_dataset', 'model_dataset',
                                                             'model_split_ID', 'pruning_crit', 'sampling_time'])

    # If the reference dataset was a GeneratedDataset itself, add its sampling time to gen_dataset.
    if isinstance(ref_dataset, GeneratedDataset) and ref_dataset.sampling_time is not None:
        gen_dataset.sampling_time += ref_dataset.sampling_time
        gen_dataset.save_metadata()

    # Save the conditioners' energy timeseries in a .npz file in the dataset directory
    if save_cond_U:
        conditioners_info = sampling_output[1]
        gen_dataset.calc_dir.mkdir(exist_ok=True)
        conditioner_U_filepath = gen_dataset.calc_dir / 'conditioners_U.npz'
        np.savez(conditioner_U_filepath, **conditioners_info)

    # Remove temporary refined dataset directory if it was created.
    if ref_dataset_refined_dir is not None and ref_dataset_refined_dir.exists():
        shutil.rmtree(ref_dataset_refined_dir)

    return gen_dataset


def generate_end_to_end_samples(coarse_model_configs: str, fine_model_configs: str, ID_coarse: str, ID_fine: str,
                                heavy_model_epoch='', regen=False, overwritten_configs: dict = None,
                                save_trajectory=False, **kwargs):
    # Remove overwritten configs for coarse sampling
    omitted_overwritten_coarse_configs = {'conditioners', 'dynamic_denoising'}
    omitted_coarse_configs_keys = set(overwritten_configs.keys()) - omitted_overwritten_coarse_configs
    overwritten_coarse_configs = {k: overwritten_configs[k] for k in omitted_coarse_configs_keys}

    # Generate coarse samples using the coarse model
    coarse_gen_dataset = generate_samples(config_name=coarse_model_configs, ID=ID_coarse,
                                          regen=regen, overwritten_configs=overwritten_coarse_configs,
                                          save_trajectory=save_trajectory, **kwargs)
    coarse_config_ID = config.ID

    # Load the fine model config and reload models modules to redefine default arguments with the new configs
    config.import_configs(configs_filename=fine_model_configs, overwritten_configs=overwritten_configs)
    fine_config_ID = config.ID

    # Generate a directory name for the fine samples
    config_tag = f"{coarse_config_ID}_{fine_config_ID}_{config.dataset}_{config.dataset_split_set_ID}"
    ID_end_to_end = f"{ID_coarse}{ID_fine}"
    if save_trajectory:
        fine_samples_dir = DenoisingTrajectoriesDataset.generate_dir(ID=ID_end_to_end, config_tag=config_tag)
    else:
        fine_samples_dir = GeneratedDataset.generate_dir(ID=ID_end_to_end, config_tag=config_tag)

    # Reset dynamo cache in case the heavy-atom model uses configs that break with the cached compiled CS model
    if config.compile_model:
        torch._dynamo.reset()

    # Generate fine samples
    fine_gen_dataset = generate_samples(config_name=fine_model_configs, ID=ID_end_to_end, model_epoch=heavy_model_epoch,
                                        regen=regen, overwritten_configs=overwritten_configs,
                                        ref_dataset=coarse_gen_dataset, save_trajectory=save_trajectory,
                                        samples_dir=fine_samples_dir, **kwargs)

    # Add hydrogen atoms
    if not save_trajectory:
        fine_gen_dataset.hydrogenize()

    return coarse_gen_dataset, fine_gen_dataset


if __name__ == '__main__':
    config_parser = config.get_config_parser()  # Start from config parser so that config values can be overwritten.
    config_parser.add_help = False
    p = argparse.ArgumentParser(parents=[config_parser], allow_abbrev=False, conflict_handler='resolve')
    p.add_argument('--configs', type=str, default='', help='Filename of configurations stored in ./configs')
    p.add_argument('--configs_coarse', type=str, default=None,
                   help='Filename of configurations for generating coarse structures')
    p.add_argument('--configs_fine', type=str, default=None,
                   help='Filename of configurations for generating fine structures')
    p.add_argument('--ID', type=str, default='2', help='ID of the generated samples')
    p.add_argument('--ID_coarse', type=str, default=None, help='ID of the coarse samples (end-to-end sampling)')
    p.add_argument('--ID_fine', type=str, default=None, help='ID of the fine samples (end-to-end sampling)')
    p.add_argument('--model_epoch', type=str, default='', help='Epoch of the model checkpoint used for sampling')
    p.add_argument('--regen', action='store_true', default=False, help='Regenerate samples')
    p.add_argument('--psi', type=float, default=None, help='psi parameter of the SDE sampler')
    p.add_argument('--lambda0', type=float, default=None, help='lambda0 parameter of the SDE sampler')
    p.add_argument('--ref_dataset', type=str, default=None,
                   help='Dataset used for defining initial and fixed properties')
    p.add_argument('--ref_dataset_split', type=str, default=None, help='split of the reference dataset used')
    p.add_argument('--n_samples', default='train', help='Number of samples to generate')
    p.add_argument('--batch_size', type=int, default=None, help='Number of samples generated in each batch')
    p.add_argument('--T', type=int, default=None, help='Number of diffusion steps. Synonym of n_diff_steps')
    p.add_argument('--save_trajectory', default=False, action='store_true', help='Save denoising trajectories')
    p.add_argument('--save_cond_U', default=False, action='store_true', help='Save conditioners energy')
    p.add_argument('--pruning_method', type=str, default=None, help='Method for pruning samples')
    p.add_argument('--recalc_stats', default=[], nargs='*', help='Name of statistics that will recalculated.')
    p.add_argument('--skip_comparison', action='store_true', help='Skips metric comparison')
    args = p.parse_args()

    # Assign default args
    if args.n_samples.isdigit():
        args.n_samples = int(args.n_samples)
    if args.dataset is None and not config.dataset:
        raise ValueError(f"Cannot identify dataset. Use --dataset argument to specify the dataset.")

    # Determine the configs that are overwritten
    overwritten_configs = {}
    omitted_args = ['ID', 'n_samples', 'batch_size']  # Input args that do not overwrite configs
    input_args_name = [arg[2:] for arg in sys.argv if arg.startswith('-') and arg[2:] not in omitted_args]
    input_args = {n: getattr(args, n) for n in input_args_name}
    for arg_name, arg_val in input_args.items():
        if arg_name in ['psi', 'lambda0']:
            if 'SDE_sampler_params' not in overwritten_configs:
                overwritten_configs['SDE_sampler_params'] = config.SDE_sampler_params
            overwritten_configs['SDE_sampler_params'][arg_name] = arg_val
        elif arg_name == 'T':
            overwritten_configs['n_diff_steps'] = args.T
        elif arg_name in config.default_configs:
            overwritten_configs[arg_name] = arg_val

    # Generate samples
    if args.configs:
        gen_dataset = generate_samples(config_name=args.configs, model_epoch=args.model_epoch,
                                       n_samples=args.n_samples, batch_size=args.batch_size,
                                       ID=args.ID, regen=args.regen,
                                       ref_dataset=args.ref_dataset, ref_dataset_split=args.ref_dataset_split,
                                       overwritten_configs=overwritten_configs,
                                       save_trajectory=args.save_trajectory,
                                       save_cond_U=args.save_cond_U)
    elif args.configs_coarse and args.configs_fine:
        coarse_gen_dataset, gen_dataset = generate_end_to_end_samples(coarse_model_configs=args.configs_coarse,
                                                                      fine_model_configs=args.configs_fine,
                                                                      heavy_model_epoch=args.model_epoch,
                                                                      regen=args.regen,
                                                                      n_samples=args.n_samples,
                                                                      batch_size=args.batch_size,
                                                                      ID_coarse=args.ID_coarse,
                                                                      ID_fine=args.ID_fine,
                                                                      overwritten_configs=overwritten_configs,
                                                                      save_trajectory=args.save_trajectory,
                                                                      save_cond_U=args.save_cond_U)
    else:
        raise ValueError(f"No configs were given.")

    # If trajectory is saved, make movies and exit early since statistics are not needed.
    if args.save_trajectory:
        make_denoising_movie(gen_dataset)
        if args.configs_coarse and args.configs_fine:
            make_denoising_movie(coarse_gen_dataset)
        sys.exit()

    analyzed_datasets = [gen_dataset]

    # Prune generated dataset, if requested
    gen_dataset_pruned = None
    if args.pruning_method is not None:
        try:
            gen_dataset_pruned = gen_dataset.prune(method=args.pruning_method)
            analyzed_datasets.append(gen_dataset_pruned)
        except Exception as e:
            print(f'Pruning method had errors:{e}')

    # If end-to-end dataset is generated, only the h_bond_map statistics of the all-atoms version is needed.
    # After that, switch to heavy-atoms to calculate heavy-atom statistics.
    if args.configs_coarse and args.configs_fine:
        for d in analyzed_datasets:
            analyze_dataset(d, stats='h_bond_map', recalc_stats=args.recalc_stats)
        # Switch to heavy-atoms after all-atom analysis is done.
        for d in analyzed_datasets:
            d.element_class = heavy_atoms_class_name

    # Analyze heavy-atom statistics
    for d in analyzed_datasets:
        analyze_dataset(d, recalc_stats=args.recalc_stats)

    # Compare the generated samples against the model dataset
    if not args.skip_comparison:
        dataset_compared = gen_dataset.model_dataset
        if gen_dataset_pruned is not None:
            compare_datasets(datasets=[dataset_compared, gen_dataset_pruned], element_class=gen_dataset.element_class)
        else:
            compare_datasets(datasets=[dataset_compared, gen_dataset], element_class=gen_dataset.element_class)
