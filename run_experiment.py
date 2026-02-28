import wandb
import inspect
import os
import shlex
import sys
import torch
import shutil
import random
import numpy as np

from datetime import datetime
from omegaconf import OmegaConf
from dynamics import dynamics
from experiments import experiments
from utils import modules, dataio, losses
from utils.config import load_config, save_config, load_experiment_config

cfg = load_config()
mode = cfg.mode
use_wandb = cfg.use_wandb

# start wandb
if use_wandb:
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        group=cfg.wandb_group,
        name=cfg.wandb_name,
    )
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

experiment_dir = os.path.join(cfg.experiments_dir, cfg.experiment_name)
if (mode == 'all') or (mode == 'train'):
    # create experiment dir
    if os.path.exists(experiment_dir):
        overwrite = input("The experiment directory %s already exists. Overwrite? (y/n)" % experiment_dir)
        if not (overwrite == 'y'):
            print('Exiting.')
            quit()
        shutil.rmtree(experiment_dir)
    os.makedirs(experiment_dir)
elif mode == 'test':
    # confirm that experiment dir already exists
    if not os.path.exists(experiment_dir):
        raise RuntimeError('Cannot run test mode: experiment directory not found!')

current_time = datetime.now()
# log command invocation
invoked_command = "python " + " ".join(shlex.quote(arg) for arg in sys.argv)
with open(os.path.join(experiment_dir, 'run_command.txt'), 'w') as f:
    f.write(invoked_command + '\n')
with open(os.path.join(experiment_dir, 'run_command.sh'), 'w') as f:
    f.write('#!/usr/bin/env bash\n')
    f.write(invoked_command + '\n')
try:
    os.chmod(os.path.join(experiment_dir, 'run_command.sh'), 0o750)
except OSError:
    pass

# load config for experiment setup (save happens after dataset creation)
orig_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

# set the experiment seed
torch.manual_seed(orig_cfg.seed)
random.seed(orig_cfg.seed)
np.random.seed(orig_cfg.seed)

dynamics_class = getattr(dynamics, orig_cfg.dynamics_class)
dynamics_kwargs = {argname: orig_cfg[argname] for argname in inspect.signature(dynamics_class).parameters.keys() if argname != 'self' and argname in orig_cfg}
dynamics = dynamics_class(**dynamics_kwargs)
dynamics.deepreach_model = orig_cfg.deepreach_model
if orig_cfg.training_objective == "temporal_consistency" and orig_cfg.dynamics_class not in ('CartPole', 'Quadrotor2D', 'Quadrotor3D', 'Pendulum'):
    raise RuntimeError(
        "training_objective=temporal_consistency is currently supported only for CartPole, Quadrotor2D, or Quadrotor3D with trajectory-backed CartPoleDataset."
    )
if orig_cfg.dynamics_class in ('CartPole', 'Quadrotor2D', 'Quadrotor3D', 'Pendulum'):
    if orig_cfg.data_root is None:
        raise RuntimeError(f'{orig_cfg.dynamics_class} requires data_root for trajectory dataset')
    _dataset_kwargs = dict(
        dynamics=dynamics, numpoints=orig_cfg.numpoints,
        pretrain=orig_cfg.pretrain, pretrain_iters=orig_cfg.pretrain_iters,
        tMin=orig_cfg.tMin, tMax=orig_cfg.tMax,
        counter_start=orig_cfg.counter_start, counter_end=orig_cfg.counter_end,
        num_src_samples=orig_cfg.num_src_samples, num_target_samples=orig_cfg.num_target_samples,
        data_root=orig_cfg.data_root,
        num_supervised=orig_cfg.num_supervised,
        supervised_value_safe=orig_cfg.supervised_value_safe,
        supervised_value_unsafe=orig_cfg.supervised_value_unsafe,
        supervised_labels_file=orig_cfg.supervised_labels_file,
        supervised_balanced_sampling=orig_cfg.supervised_balanced_sampling,
        trajectory_uniform_sampling=orig_cfg.trajectory_uniform_sampling,
        max_trajectory_files=orig_cfg.max_trajectory_files,
        use_shuffled_indices_only=orig_cfg.use_shuffled_indices_only,
        shuffled_indices_file=orig_cfg.shuffled_indices_file,
        load_trajectories_in_ram=orig_cfg.load_trajectories_in_ram,
        training_objective=orig_cfg.training_objective,
        val_fraction=orig_cfg.val_fraction,
    )
    if orig_cfg.dynamics_class == 'Quadrotor2D':
        dataset = dataio.Quadrotor2DDataset(**_dataset_kwargs)
    elif orig_cfg.dynamics_class == 'Quadrotor3D':
        dataset = dataio.Quadrotor3DDataset(**_dataset_kwargs)
    elif orig_cfg.dynamics_class == 'Pendulum':
        dataset = dataio.PendulumDataset(**_dataset_kwargs)
    else:
        dataset = dataio.CartPoleDataset(**_dataset_kwargs)
else:
    dataset = dataio.ReachabilityDataset(
        dynamics=dynamics, numpoints=orig_cfg.numpoints,
        pretrain=orig_cfg.pretrain, pretrain_iters=orig_cfg.pretrain_iters,
        tMin=orig_cfg.tMin, tMax=orig_cfg.tMax,
        counter_start=orig_cfg.counter_start, counter_end=orig_cfg.counter_end,
        num_src_samples=orig_cfg.num_src_samples, num_target_samples=orig_cfg.num_target_samples)
    if orig_cfg.training_objective == "temporal_consistency":
        raise RuntimeError(
            "training_objective=temporal_consistency requires trajectory-backed CartPoleDataset; ReachabilityDataset is not supported."
        )

# Update config with dataset-adjusted values (e.g. tMax from trajectory data)
cfg.tMax = dataset.tMax
orig_cfg.tMax = dataset.tMax

# Save config after dataset creation so adjusted values (e.g. tMax) are recorded
if (mode == 'all') or (mode == 'train'):
    save_config(cfg, experiment_dir)
with open(os.path.join(experiment_dir, 'config_%s.txt' % current_time.strftime('%m_%d_%Y_%H_%M')), 'w') as f:
    for key, val in OmegaConf.to_container(cfg, resolve=True).items():
        f.write(key + ' = ' + str(val) + '\n')

model = modules.SingleBVPNet(in_features=dynamics.input_dim, out_features=1, type=orig_cfg.model, mode=orig_cfg.model_mode,
                             final_layer_factor=1., hidden_features=orig_cfg.num_nl, num_hidden_layers=orig_cfg.num_hl,
                             omega_0=orig_cfg.omega_0)
model.to(cfg.device)

experiment_class = getattr(experiments, orig_cfg.experiment_class)
experiment = experiment_class(model=model, dataset=dataset, experiment_dir=experiment_dir, use_wandb=use_wandb)
experiment.init_special(**{argname: orig_cfg[argname] for argname in inspect.signature(experiment_class.init_special).parameters.keys() if argname != 'self' and argname in orig_cfg})

if (mode == 'all') or (mode == 'train'):
    training_objective = orig_cfg.training_objective
    if training_objective == "temporal_consistency":
        loss_fn = losses.init_temporal_consistency_loss()
    elif training_objective == "hj_pde":
        if dynamics.loss_type == 'brt_hjivi':
            loss_fn = losses.init_brt_hjivi_loss(dynamics, orig_cfg.minWith, orig_cfg.dirichlet_loss_divisor)
        elif dynamics.loss_type == 'brat_hjivi':
            loss_fn = losses.init_brat_hjivi_loss(dynamics, orig_cfg.minWith, orig_cfg.dirichlet_loss_divisor)
        else:
            raise NotImplementedError
    else:
        raise RuntimeError(f"Unknown training objective: {training_objective}")
    experiment.train(
        device=cfg.device, batch_size=1, epochs=orig_cfg.num_epochs, lr=orig_cfg.lr,
        steps_til_summary=orig_cfg.steps_til_summary, epochs_til_checkpoint=orig_cfg.epochs_til_ckpt,
        loss_fn=loss_fn, clip_grad=orig_cfg.clip_grad, use_lbfgs=orig_cfg.use_lbfgs, adjust_relative_grads=orig_cfg.adj_rel_grads,
        val_x_resolution=orig_cfg.val_x_resolution, val_y_resolution=orig_cfg.val_y_resolution, val_z_resolution=orig_cfg.val_z_resolution, val_time_resolution=orig_cfg.val_time_resolution,
        use_CSL=orig_cfg.use_CSL, CSL_lr=orig_cfg.CSL_lr, CSL_dt=orig_cfg.CSL_dt, epochs_til_CSL=orig_cfg.epochs_til_CSL, num_CSL_samples=orig_cfg.num_CSL_samples, CSL_loss_frac_cutoff=orig_cfg.CSL_loss_frac_cutoff, max_CSL_epochs=orig_cfg.max_CSL_epochs, CSL_loss_weight=orig_cfg.CSL_loss_weight, CSL_batch_size=orig_cfg.CSL_batch_size,
        supervised_weight=orig_cfg.supervised_weight,
        supervised_safe_weight=orig_cfg.supervised_safe_weight,
        supervised_unsafe_weight=orig_cfg.supervised_unsafe_weight,
        early_stopping_patience=orig_cfg.early_stopping_patience,
        eval_batch_size=orig_cfg.eval_batch_size,
        num_src_samples_decay_epochs=orig_cfg.num_src_samples_decay_epochs,
    )

if (mode == 'all') or (mode == 'test'):
    experiment.test(
        device=cfg.device, current_time=current_time,
        last_checkpoint=orig_cfg.num_epochs, checkpoint_dt=orig_cfg.epochs_til_ckpt,
        checkpoint_toload=cfg.checkpoint_toload, dt=cfg.dt,
        num_scenarios=cfg.num_scenarios, num_violations=cfg.num_violations,
        set_type='BRT' if orig_cfg.minWith in ['zero', 'target'] else 'BRS', control_type=cfg.control_type, data_step=cfg.data_step)
