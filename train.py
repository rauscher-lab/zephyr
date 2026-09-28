import copy
import os
import pathlib
import time
import sys
import warnings

import numpy as np
from collections import defaultdict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from schedulefree import AdamWScheduleFree

try:
    import wandb
except ImportError:
    pass

from utils import Checkpointer, PiecewiseLR, LossesDict, repeat_iterator, AdEMAMix, set_rng, factorint, \
    ProgressiveSampler
from data.datasets import load_dataset, BiomoleculeDataset, ProcessedDataset
from data.data_classes import heavy_atoms_class_name, all_atoms_class_name
from analysis.metrics import calculate_training_metrics, calculate_split_sample_J_dist, defined_props
import config
import models  # Can't do "from models import ..." since kwargs defaults depending on global configs won't update.

# Force 'fork' start method for multiprocessing to avoid freeze_method() error
torch.multiprocessing.set_start_method('fork', force=True)


def calculate_minibatch_size(dataset_train: torch.utils.data.Dataset, batch_size: int):
    """
    Reduces minibatch size to ensure that it satisfies the following conditions:
        1) minibatch_size <= number of training samples
        2) minibatch_size <= batch_size
        3) minibatch_size is an integer multiple of batch_size
    Args:
        dataset_train: training dataset
        batch_size: size of batch
    Returns:
        int
    """
    minibatch_size = config.minibatch_size
    while minibatch_size > len(dataset_train) or minibatch_size > batch_size:
        minibatch_size_factors = factorint(minibatch_size)  # minibatch_size is usually small.
        smallest_fact = next(iter(minibatch_size_factors))
        minibatch_size //= smallest_fact
    return minibatch_size


def update_batch_size(epoch: int):
    """
    Updates batch size when using a batch size schedule
    Args:
        epoch: epoch number

    Returns:
        None
    """
    global batch_size, minibatch_size, n_minibatches_per_batch, n_train_mols_per_epoch, dataloader_train, dataloader_train_rep
    batch_size_prev = batch_size
    if config.batch_size_sch is not None:
        batch_size = [v for k, v in config.batch_size_sch.items() if epoch >= k][-1]

    # Redefine batch size dependent variables if it has changed
    if batch_size != batch_size_prev:
        # Redefine the train dataloader if the minibatch_size is smaller than batch_size. This can happen if schedule is used.
        minibatch_size_prev = minibatch_size
        minibatch_size = calculate_minibatch_size(dataset_proc_train, batch_size)
        if minibatch_size != minibatch_size_prev:
            dataloader_train = DataLoader(dataset_proc_train, batch_size=minibatch_size, shuffle=shuffle,
                                          num_workers=config.N_workers, pin_memory=config.pin_memory,
                                          collate_fn=id_collate, drop_last=config.drop_last_batch, sampler=sampler)
            dataloader_train_rep = repeat_iterator(dataloader_train)

        n_minibatches_per_batch = batch_size // minibatch_size
        n_train_mols_per_epoch = torch.tensor(n_train_batches * batch_size, device=device)

        msg = f"Batch size changed! epoch:{epoch}, batch_size:{batch_size}, n_train_batches:{n_train_batches}"
        if minibatch_size != batch_size:
            msg += f', minibatch_size:{minibatch_size}, n_minibatches_per_batch:{n_minibatches_per_batch}'
        print(msg)


def clip_gradients():
    """
    Clips gradients if gradient clipping is activated
    Returns:
        None
    """
    if config.grad_rel_norm_max is not None or config.grad_norm_max is not None:
        params = [p for p in net.parameters_train() if p.grad is not None]
        for p in params:
            p_norm = torch.abs(p)
            p_grad_mag_max = torch.inf if config.grad_norm_max is None else config.grad_norm_max
            if config.grad_rel_norm_max is not None:
                p_grad_mag_max = torch.clamp(config.grad_rel_norm_max * p_norm, max=p_grad_mag_max)
                is_p_norm_small = p_norm < 1e-10
                p_grad_mag_max[is_p_norm_small] = 1.0

            torch.clamp_(p.grad, min=-p_grad_mag_max, max=p_grad_mag_max)


def train_epoch(epoch: int):
    # Initialize epoch stats
    epoch_start_time = time.time()
    loss_train_prev = 0.0
    epoch_stats = dict(epoch=epoch)
    epoch_losses.reset()

    # Update the batch size (only applicable if using batch_size schedule)
    update_batch_size(epoch)

    # Update progressive sampler
    if isinstance(sampler, ProgressiveSampler):
        sampler.epoch = epoch

    # Activate auxiliary losses (if applicable)
    for aux_loss, act_epoch in zip(net.diff_model.aux_losses, config.aux_losses_act_epoch):
        aux_loss.is_active = epoch >= act_epoch

    # Activate train mode
    if isinstance(optimizer, AdamWScheduleFree):
        optimizer.train()
    net.train()

    for i in range(n_train_batches):
        net.zero_grad()
        for j in range(n_minibatches_per_batch):
            tensors_minibatch = next(dataloader_train_rep)
            minibatch_size_curr = tensors_minibatch.pop('batch_size')
            if config.N_workers > 0:
                dataset_proc_train.move_tensors_to_device(tensors_minibatch)

            with torch.autocast(device_type=device.type, dtype=config.amp_dtype, enabled=config.amp):
                loss_train_mini, losses_dict_train_mini = train_loss_func(tensors_batch=tensors_minibatch)

            # Renormalize loss to calculate average over entire batch
            loss_train_mini_renorm = loss_train_mini * minibatch_size_curr / batch_size

            # Accumulate gradients of each minibatch
            amp_scaler.scale(loss_train_mini_renorm).backward()

            # Accumulate train losses for entire epoch
            renorm_epoch_factor = minibatch_size_curr / n_train_mols_per_epoch
            epoch_losses['loss_train'] += renorm_epoch_factor * loss_train_mini.detach()
            for loss_name, loss_val in losses_dict_train_mini.items():
                epoch_losses[f'loss_{loss_name}_train'] += renorm_epoch_factor * loss_val.detach()

        # Clip gradients
        clip_gradients()

        # Optim step
        amp_scaler.step(optimizer)
        amp_scaler_scale_prev = amp_scaler.get_scale()
        amp_scaler.update()
        amp_scaler_scale = amp_scaler.get_scale()
        if amp_scaler_scale != amp_scaler_scale_prev:
            print(f"AMP scaler scale changed from {amp_scaler_scale_prev} to {amp_scaler_scale}.")

        loss_train = epoch_losses['loss_train']
        if loss_train.isnan():
            warnings.warn(f"Nan found in train loss at epoch {epoch}, batch {i}")

        # Update evaluation model
        if config.model_ema_beta > 0:
            net_eval.EMA_update(net, config.model_ema_beta)

        # Advance the network progression if in progressive mode.
        if net.progressive:
            net.advance_transition(omega_step)

        # Iteration monitoring
        if config.debug:
            print(f'batch {i + 1}/{n_train_batches}. loss_train:{loss_train}')

        # Check for instabilities.
        if config.debug and epoch > 10 and loss_train > 10 * loss_train_prev and loss_train_prev > 0.0:
            print(f'Encountered numerical overflow at iteration {i + 1}!!!')
            print('Previous train loss:', loss_train_prev)
            print('Current train loss:', loss_train)
        loss_train_prev = loss_train

    # Calculate loss on validation dataset
    if isinstance(optimizer, AdamWScheduleFree):
        optimizer.eval()

    net_eval.eval()
    for i, tensors_batch in enumerate(dataloader_valid):
        batch_size_curr = tensors_batch.pop('batch_size')
        if config.N_workers > 0:
            dataset_proc_train.move_tensors_to_device(tensors_batch)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=config.amp_dtype, enabled=config.amp):
            loss_valid, losses_dict_valid = eval_loss_func(tensors_batch=tensors_batch)

        # Accumulate epoch losses for monitoring
        renorm_factor = batch_size_curr / n_valid_mols_per_epoch
        epoch_losses['loss_valid'] += renorm_factor * loss_valid
        for loss_name, loss_val in losses_dict_valid.items():
            epoch_losses[f'loss_{loss_name}_valid'] += renorm_factor * loss_val

    # Transfer monitored losses accumulated on device to cpu
    for loss_name, loss_val in epoch_losses.items():
        epoch_stats[loss_name] = loss_val.cpu().item()

    # Update learning rate
    if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        lr_scheduler.step(epoch_stats['loss_valid'])
    elif lr_scheduler is not None:
        lr_scheduler.step()

    # Monitoring
    if epoch % monitoring_period == 0:
        Monitored_values['Epoch'] = f"{epoch}"
        Monitored_values['time(s)'] = f"{time.time() - epoch_start_time:.1f}"
        Monitored_values['loss_train'] = f"{epoch_stats['loss_train']: >#7.4g}"
        Monitored_values['loss_valid'] = f"{epoch_stats['loss_valid']: >#7.4g}"
        msg = ', '.join([s + ':' + v for s, v in Monitored_values.items()])
        print(msg)

    # If network is progressive, record the current number of layers
    if net.progressive:
        epoch_stats['n_layers'] = net.n_layers

    # Generate samples and calculated monitored metrics
    if epoch % config.sampling_period == 0 or epoch == config.N_epochs:
        gen_dataset = models.generate_samples(model=net_eval, ref_dataset=sampling_ref_dataset,
                                              ID=samples_ID, regen=True, T=config.n_diff_steps_monitoring)

        # Compare the samples metric with the train and validation datasets
        samples_metrics = calculate_training_metrics(gen_dataset)
        metrics_diff = {}
        for metric in samples_metrics:
            metric_pred = samples_metrics[metric]
            metric_train, metric_valid = train_metrics[metric], valid_metrics[metric]
            metric_train_rel_err = (metric_pred - metric_train) / (np.abs(metric_train) + 1e-8)
            metric_valid_rel_err = (metric_pred - metric_valid) / (np.abs(metric_valid) + 1e-8)
            if isinstance(metric_pred, torch.Tensor):
                metric_train_rel_err = metric_train_rel_err.mean().item()
                metric_valid_rel_err = metric_valid_rel_err.mean().item()
            metrics_diff[f"{metric}_train_rel_err"] = 100 * metric_train_rel_err
            metrics_diff[f"{metric}_valid_rel_err"] = 100 * metric_valid_rel_err
        epoch_stats.update(metrics_diff)
        for name, val in metrics_diff.items():
            print(f"{name}: {val}")

        # Calculate various Jeffrey's distances between the generated dataset and the dataset splits
        Jdist_props_name = list({'dihedral_angle'} & defined_props[gen_dataset.element_class])
        if gen_dataset.element_class not in [heavy_atoms_class_name, all_atoms_class_name]:
            Jdist_props_name.append('elements_pdist')
        if Jdist_props_name:
            Jdist_props = calculate_split_sample_J_dist(dataset_splits=dataset_splits, gen_dataset=gen_dataset,
                                                        props=Jdist_props_name)
            for prop_name, prop_val_arr in Jdist_props.items():
                for i, split_name in enumerate(dataset_splits_name[:2]):
                    epoch_stats[f"{prop_name}_jdist_{split_name}"] = prop_val_arr[i].item()

    # Wandb logs
    if config.wandb:
        wandb_stats = {k: v for k, v in epoch_stats.items() if k not in wandb_omitted_stats}
        wandb.log(wandb_stats, step=epoch)

    # Update epoch stats
    epoch_end_time = time.time()
    epoch_stats['epoch_time'] = epoch_end_time - epoch_start_time
    checkpointer.update_epoch_stats(epoch_stats)

    # Checkpoint
    force_checkpoint = epoch % config.sampling_period == 0 or epoch == epoch_final - 1
    checkpointer.check(epoch, force=force_checkpoint)


if __name__ == '__main__':
    # Parse arguments
    parser = config.get_config_parser()
    args = parser.parse_args()

    # Import the configurations given in the configs file and overwrite configs given in parsed args
    input_configs = [arg[2:] for arg in sys.argv if arg.startswith('-') and arg[2:] in config.default_configs]
    overwritten_configs = {arg: getattr(args, arg) for arg in input_configs}

    # Overwrite the SDE sample parameters to psi=0.0 and lambda0=1.0 for training
    overwritten_configs['SDE_sampler_params'] = {'psi': 0.0, 'lambda0': 1.0}

    config.import_configs(args.configs, overwritten_configs=overwritten_configs)

    # Print configurations
    config.print_configs()

    # Fix random seed for reproducibility
    set_rng(config.seed)

    # Device
    if config.device == 'cuda':
        device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True
    elif config.device == 'mps':
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    ######################################## Dataset ########################################
    dataset = load_dataset(element_class=config.dataset_element_class)

    # Split the dataset into train, valid and test
    dataset_splits_name = ['train', 'valid']
    dataset_splits: list[BiomoleculeDataset] = dataset.load_splits(name=dataset_splits_name,
                                                                   load_all=config.N_workers == 0)
    dataset_train, dataset_valid = tuple(dataset_splits)  # tuple to facilitate type inference

    ######################################## Network ########################################
    net = models.initialize(dataset_train)
    net.to(device)
    omega_step = torch.tensor(config.omega_step).to(device)  # step size for progressive models
    if config.debug:
        print(f"Is network equivariant? {net.is_equivariant()}")
    n_elements_train = int(dataset_train.statistics('n_elements').sum)
    print(f"The model has {net.n_params:,} parameters. "
          f"The training dataset has {n_elements_train * 3:,} DOF ({n_elements_train:,} elements with 3D positions.)")

    # Evaluation model
    if config.model_ema_beta > 0:
        net_eval = copy.deepcopy(net)
        net_eval.diff_model.aux_losses = net.diff_model.aux_losses
        net_eval.diff_model.conditioners = net.diff_model.conditioners
    else:
        net_eval = net

    # Loss functions
    train_loss_func = net.loss
    eval_loss_func = net_eval.loss

    # Compile models
    if config.compile_model:
        train_loss_func = torch.compile(train_loss_func, dynamic=True)
        if config.model_ema_beta > 0:
            eval_loss_func = torch.compile(eval_loss_func, dynamic=True)
        torch._dynamo.reset()

    params_train = net.parameters_train()
    Adam_eps = defaultdict(lambda: 1e-8, {torch.half: 1e-4})[torch.get_default_dtype()]
    if config.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(params_train, lr=config.learning_rate, **config.optim_extra_kwargs)
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(params_train, lr=config.learning_rate, betas=config.adam_betas,
                               amsgrad=config.amsgrad, eps=Adam_eps, **config.optim_extra_kwargs)
    elif config.optimizer == 'AdamW':
        optimizer = optim.AdamW(params_train, lr=config.learning_rate, betas=config.adam_betas,
                                amsgrad=config.amsgrad, weight_decay=config.weight_decay, eps=Adam_eps,
                                **config.optim_extra_kwargs)
    elif config.optimizer == 'AdamWScheduleFree':
        optimizer = AdamWScheduleFree(params_train, lr=config.learning_rate, betas=config.adam_betas,
                                      weight_decay=config.weight_decay, eps=Adam_eps,
                                      **config.optim_extra_kwargs)
    elif config.optimizer == 'AdEMAMix':
        optimizer = AdEMAMix(params_train, lr=config.learning_rate, betas=config.adam_betas,
                             weight_decay=config.weight_decay, eps=Adam_eps, **config.optim_extra_kwargs)
    elif config.optimizer == 'RAdam':
        optimizer = optim.RAdam(params_train, lr=config.learning_rate, betas=config.adam_betas,
                                weight_decay=config.weight_decay, decoupled_weight_decay=True, eps=Adam_eps,
                                **config.optim_extra_kwargs)
    else:
        raise NotImplementedError(f'{config.optimizer} is not implemented.')

    if config.lr_piecewise_params is not None:
        params = config.lr_piecewise_params
        lr_scheduler = PiecewiseLR(optimizer, epochs=params['epochs'], lrs=params['lrs'], funcs=params['funcs'])
    else:
        lr_scheduler = None

    # AMP
    amp_scaler = torch.amp.GradScaler(device=device.type, enabled=config.amp)

    # ######################################## Checkpointer ########################################
    run_ID = config.generate_run_ID(4)
    checkpoint_filepath = models.format_checkpoint_filepath()
    extra_modules = dict()
    if config.amp:
        extra_modules['amp_scaler'] = amp_scaler
    checkpointer = Checkpointer(net, optimizer, filepath=checkpoint_filepath,
                                run_ID=run_ID, net_ema=net_eval, lr_scheduler=lr_scheduler, extra_modules=extra_modules,
                                N_epochs=config.N_epochs,
                                checkpoint_period=config.checkpoint_period,
                                extra_checkpoint_period=config.checkpoint_extra_period)

    # Send warning that previous weights will be overwritten when not resumin
    if not config.resume and checkpoint_filepath.exists():
        warnings.warn(f'Checkpoint {checkpoint_filepath.name!r} already exists. '
                      f'Training checkpoints will be overwritten. Resuming in 5 seconds...')
        time.sleep(5)

    # Load previous state if training is resumed
    if config.resume and checkpoint_filepath.exists():
        checkpointer.load_state()
    elif config.init_ID is not None:
        init_state_filepath = pathlib.Path(config.weights_dir, config.init_ID)
        checkpointer.load_state(init_state_filepath)

    # Determine initial and final epoch
    epoch_init = checkpointer.epoch + 1  # The first epoch starts at 1
    if config.N_epochs_session:
        epoch_final = epoch_init + config.N_epochs_session
    else:
        epoch_final = 1 + config.N_epochs

    ######################################## Dataloaders ########################################
    dataset_proc_train = ProcessedDataset(dataset_train, model=net, inMemory=config.dataset_load_all, device=device)
    dataset_proc_valid = ProcessedDataset(dataset_valid, model=net, inMemory=config.dataset_load_all, device=device)

    if config.dataloader_sampler is None:
        sampler, shuffle = None, True
    elif config.dataloader_sampler == 'with_replacement':
        sampler, shuffle = torch.utils.data.sampler.RandomSampler(data_source=dataset_proc_train,
                                                                  replacement=True), None
    elif config.dataloader_sampler == 'progressive':
        sampler, shuffle = ProgressiveSampler(dataset=dataset), False
    else:
        raise NotImplementedError(f"{config.dataloader_sampler!r} sampler method is not implemented")

    # Initialize dataloaders and batch size
    if config.batch_size_sch is not None:
        batch_size = [v for k, v in config.batch_size_sch.items() if epoch_init >= k][-1]
    else:
        batch_size = config.batch_size

    # Define the size of a mini-batch. This should remain fixed even if a batch size schedule is used.
    minibatch_size = calculate_minibatch_size(dataset_proc_train, batch_size)
    id_collate = lambda x: x
    dataloader_train = DataLoader(dataset_proc_train, batch_size=minibatch_size, shuffle=shuffle,
                                  num_workers=config.N_workers, pin_memory=config.pin_memory, collate_fn=id_collate,
                                  drop_last=config.drop_last_batch, sampler=sampler)
    dataloader_train_rep = repeat_iterator(dataloader_train)
    dataloader_valid = DataLoader(dataset_proc_valid, batch_size=min(len(dataset_proc_valid), config.minibatch_size),
                                  shuffle=False, num_workers=config.N_workers, pin_memory=False, collate_fn=id_collate)
    n_minibatches = len(dataloader_train)
    n_minibatches_per_batch = batch_size // minibatch_size
    n_train_batches = config.n_batches_per_epoch
    n_train_mols_per_epoch = torch.tensor(n_train_batches * batch_size, device=device)
    n_valid_mols_per_epoch = torch.tensor(len(dataset_valid), device=device)
    print(f"batch_size:{batch_size}, n_train_batches:{n_train_batches}")

    # Monitoring
    Monitored_values = dict()
    monitoring_period = 1  # Number of epochs between print out of monitored values.
    epoch_losses = LossesDict(lambda: torch.tensor(0.0, device=device))
    train_metrics = calculate_training_metrics(dataset_train)
    valid_metrics = calculate_training_metrics(dataset_valid)

    # Samples
    samples_ID = 'debug1' if config.debug else 'train1'
    if config.sampling_ref_dataset == 'train':
        sampling_ref_dataset = dataset_train
    elif config.sampling_ref_dataset == 'valid':
        sampling_ref_dataset = dataset_valid
    else:
        raise ValueError(f"sampling_ref_dataset={config.sampling_ref_dataset!r} is not supported.")

    if config.debug:
        warnings.warn(f'Overwriting the number of samples to len(sampling_dataset)={len(sampling_ref_dataset)}')
        config.n_samples = len(sampling_ref_dataset)

    # Shuffle the reference dataset for the generation of monitoring samples
    with torch.random.fork_rng():
        torch.manual_seed(2024)  # Fix seed to allow comparisons across different configs
        ref_ind = torch.randperm(len(sampling_ref_dataset))[:config.n_samples]
    sampling_ref_dataset = sampling_ref_dataset.subset(ind=ref_ind, inmemory=False, load_all=True)

    # Auxiliary losses
    if config.aux_loss_estimate_cov or config.aux_loss_reweight:
        recalculate_stats = config.init_ID is not None and not os.path.exists(checkpoint_filepath)
        net.diff_model.estimate_aux_loss_stats(dataset_proc_train, regen=recalculate_stats)

        # Save the stats at the first epoch in case there are needed again when training is resumed
        if checkpointer.epoch == 0 or recalculate_stats:
            checkpointer.save_state(epoch=0)

    # Wandb
    if config.wandb:
        wandb_init_kwargs = {'project': config.project_name, 'id': checkpointer.run_ID,
                             'config': config.logged_configs(), 'save_code': True, 'resume': 'allow',
                             'settings': wandb.Settings(code_dir=str(config.root_dir))}
        if config.wandb_logs_dir:
            wandb_init_kwargs['dir'] = str(config.wandb_logs_dir)

        wandb.init(**wandb_init_kwargs)
        if config.wandb_log_params:
            wandb.watch(net, log='all', log_freq=n_train_batches * config.sampling_period)

        # Stats are not logged with Wandb
        wandb_omitted_stats = ['epoch', 'epoch_time']

    # Start training
    for epoch in range(epoch_init, epoch_final):
        train_epoch(epoch)

    if config.wandb:
        wandb.finish()
