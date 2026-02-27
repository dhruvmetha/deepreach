import copy
import wandb
import torch
import os
import shutil
import time
import math
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.io as spio

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from collections import OrderedDict
from datetime import datetime
from sklearn import svm 
from utils import diff_operators
from utils.error_evaluators import scenario_optimization, ValueThresholdValidator, MultiValidator, MLPConditionedValidator, target_fraction, MLP, MLPValidator, SliceSampleGenerator

class Experiment(ABC):
    def __init__(self, model, dataset, experiment_dir, use_wandb):
        self.model = model
        self.dataset = dataset
        self.experiment_dir = experiment_dir
        self.use_wandb = use_wandb

    @abstractmethod
    def init_special(self):
        raise NotImplementedError

    def _load_checkpoint(self, epoch):
        if epoch == -1:
            model_path = os.path.join(self.experiment_dir, 'training', 'checkpoints', 'model_final.pth')
            self.model.load_state_dict(torch.load(model_path))
        else:
            model_path = os.path.join(self.experiment_dir, 'training', 'checkpoints', 'model_epoch_%04d.pth' % epoch)
            self.model.load_state_dict(torch.load(model_path)['model'])

    def validate(self, device, epoch, save_path, x_resolution, y_resolution, z_resolution, time_resolution):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)

        plot_config = self.dataset.dynamics.plot_config()

        state_test_range = self.dataset.dynamics.state_test_range()
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]

        # Check if z_axis is distinct from x/y axes (i.e., state_dim > 2)
        z_axis_idx = plot_config.get('z_axis_idx')
        has_z_axis = (z_axis_idx is not None
                      and z_axis_idx != plot_config['x_axis_idx']
                      and z_axis_idx != plot_config['y_axis_idx'])

        times = torch.linspace(0, self.dataset.tMax, time_resolution)
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        if has_z_axis:
            z_min, z_max = state_test_range[z_axis_idx]
            zs = torch.linspace(z_min, z_max, z_resolution)
        else:
            zs = torch.tensor([0.0])  # single dummy slice
        xys = torch.cartesian_prod(xs, ys)

        fig = plt.figure(figsize=(5*len(times), 5*len(zs)))
        for i in range(len(times)):
            for j in range(len(zs)):
                coords = torch.zeros(x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
                coords[:, 0] = times[i]
                coords[:, 1:] = torch.tensor(plot_config['state_slices'])
                coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
                if has_z_axis:
                    coords[:, 1 + z_axis_idx] = zs[j]

                with torch.no_grad():
                    model_results = self.model({'coords': self.dataset.dynamics.coord_to_input(coords.to(device))})
                    values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())

                ax = fig.add_subplot(len(times), len(zs), (j+1) + i*len(zs))
                if has_z_axis:
                    ax.set_title('t = %0.2f, %s = %0.2f' % (times[i], plot_config['state_labels'][z_axis_idx], zs[j]))
                else:
                    ax.set_title('t = %0.2f' % times[i])
                s = ax.imshow(1*(values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
                fig.colorbar(s)
        fig.savefig(save_path)

        # Value function plot (continuous values)
        value_save_path = save_path.replace('.png', '_values.png')
        fig_val = plt.figure(figsize=(5*len(times), 5*len(zs)))
        for i in range(len(times)):
            for j in range(len(zs)):
                coords = torch.zeros(x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
                coords[:, 0] = times[i]
                coords[:, 1:] = torch.tensor(plot_config['state_slices'])
                coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
                if has_z_axis:
                    coords[:, 1 + z_axis_idx] = zs[j]

                with torch.no_grad():
                    model_results = self.model({'coords': self.dataset.dynamics.coord_to_input(coords.to(device))})
                    values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())

                val_np = values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T
                val_np = np.clip(val_np, -1.0, 1.0)
                ax = fig_val.add_subplot(len(times), len(zs), (j+1) + i*len(zs))
                if has_z_axis:
                    ax.set_title('t = %0.2f, %s = %0.2f' % (times[i], plot_config['state_labels'][z_axis_idx], zs[j]))
                else:
                    ax.set_title('t = %0.2f' % times[i])
                s = ax.imshow(val_np, cmap='RdBu', origin='lower', extent=(-1., 1., -1., 1.), vmin=-1.0, vmax=1.0)
                fig_val.colorbar(s)
        fig_val.savefig(value_save_path)

        if self.use_wandb:
            wandb.log({
                'step': epoch,
                'val_plot': wandb.Image(fig),
                'val_values_plot': wandb.Image(fig_val),
            })
        plt.close(fig)
        plt.close(fig_val)

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)
    
    def train(
            self, device, batch_size, epochs, lr, 
            steps_til_summary, epochs_til_checkpoint, 
            loss_fn, clip_grad, use_lbfgs, adjust_relative_grads, 
            val_x_resolution, val_y_resolution, val_z_resolution, val_time_resolution,
            use_CSL, CSL_lr, CSL_dt, epochs_til_CSL, num_CSL_samples, CSL_loss_frac_cutoff, max_CSL_epochs, CSL_loss_weight, CSL_batch_size,
            supervised_weight=0.0,
            supervised_safe_weight=1.0,
            supervised_unsafe_weight=1.0,
            early_stopping_patience=0,
            eval_batch_size=10000,
            num_src_samples_decay_epochs=0,
        ):
        was_eval = not self.model.training
        self.model.train()
        self.model.requires_grad_(True)

        train_dataloader = DataLoader(self.dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=0)

        optim = torch.optim.Adam(lr=lr, params=self.model.parameters())

        # copy settings from Raissi et al. (2019) and here 
        # https://github.com/maziarraissi/PINNs
        if use_lbfgs:
            optim = torch.optim.LBFGS(lr=lr, params=self.model.parameters(), max_iter=50000, max_eval=50000,
                                    history_size=50, line_search_fn='strong_wolfe')

        training_dir = os.path.join(self.experiment_dir, 'training')
        
        summaries_dir = os.path.join(training_dir, 'summaries')
        if not os.path.exists(summaries_dir):
            os.makedirs(summaries_dir)

        checkpoints_dir = os.path.join(training_dir, 'checkpoints')
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

        total_steps = 0
        new_weight = 1
        training_objective = getattr(self.dataset, 'training_objective', 'hj_pde')
        if training_objective not in ['hj_pde', 'temporal_consistency']:
            raise RuntimeError(
                f"Unsupported training objective: {training_objective}. Expected 'hj_pde' or 'temporal_consistency'."
            )
        temporal_mode = training_objective == 'temporal_consistency'
        if temporal_mode and use_CSL:
            raise RuntimeError("use_CSL is not supported with training_objective=temporal_consistency.")

        # --- Src samples decay setup ---
        num_src_samples_initial = self.dataset.num_src_samples
        if num_src_samples_decay_epochs > 0 and temporal_mode:
            print(f"[SrcDecay] num_src_samples will decay from {num_src_samples_initial} to 0 "
                  f"over {num_src_samples_decay_epochs} epochs")

        # --- Early stopping setup ---
        val_transitions = None
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        # Top-K best checkpoints: list of (val_loss, filename) sorted best→worst
        top_k_best = 3
        best_checkpoints = []  # [(val_loss, filename), ...]
        early_stop_enabled = (
            temporal_mode and early_stopping_patience > 0
            and hasattr(self.dataset, 'compute_all_val_transitions')
        )
        if early_stop_enabled:
            val_transitions = self.dataset.compute_all_val_transitions()
            if val_transitions is None:
                early_stop_enabled = False
                print("[EarlyStopping] No val transitions available, early stopping disabled.")
            else:
                val_transitions = {k: v.to(device) for k, v in val_transitions.items()}
                print(f"[EarlyStopping] Enabled with patience={early_stopping_patience}, "
                      f"{val_transitions['tc_model_coords_curr'].shape[0]} val transitions")

        def _compute_val_loss():
            """Compute mean TC loss over all val transitions in mini-batches."""
            self.model.eval()
            total_loss = 0.0
            n_total = val_transitions['tc_model_coords_curr'].shape[0]
            batch_size_val = eval_batch_size
            for start in range(0, n_total, batch_size_val):
                end = min(start + batch_size_val, n_total)
                coords_batch = val_transitions['tc_model_coords_curr'][start:end].detach().requires_grad_(True)
                obs_flow_batch = val_transitions['tc_obs_flow'][start:end]
                boundary_batch = val_transitions['tc_boundary_values'][start:end]

                results = self.model({'coords': coords_batch})
                model_in = results['model_in']
                output = results['model_out'].squeeze(dim=-1)

                values = self.dataset.dynamics.io_to_value(
                    model_in.detach(), output,
                )
                dvs = self.dataset.dynamics.io_to_dv(
                    model_in, output, create_graph=False,
                )
                dvdt = dvs[..., 0]
                dvds = dvs[..., 1:]

                batch_losses = loss_fn(values, dvdt, dvds, obs_flow_batch, boundary_batch)
                # loss_fn returns .sum() over batch; accumulate raw sum
                batch_loss = sum(l.item() for l in batch_losses.values())
                total_loss += batch_loss

            self.model.train()
            return total_loss / n_total

        with tqdm(total=len(train_dataloader) * epochs) as pbar:
            train_losses = []
            last_CSL_epoch = -1
            for epoch in range(0, epochs):
                # Decay num_src_samples linearly
                if num_src_samples_decay_epochs > 0 and not self.dataset.pretrain:
                    frac = min(epoch / num_src_samples_decay_epochs, 1.0)
                    self.dataset.num_src_samples = max(5, int(num_src_samples_initial * (1.0 - frac)))

                if self.dataset.pretrain: # skip CSL
                    last_CSL_epoch = epoch
                time_interval_length = (self.dataset.counter/self.dataset.counter_end)*(self.dataset.tMax-self.dataset.tMin)
                CSL_tMax = self.dataset.tMin + int(time_interval_length/CSL_dt)*CSL_dt
                
                # self-supervised learning
                for step, (model_input, gt) in enumerate(train_dataloader):
                    start_time = time.time()
                
                    model_input = {key: value.to(device) for key, value in model_input.items()}
                    gt = {key: value.to(device) for key, value in gt.items()}

                    if temporal_mode:
                        if self.dataset.pretrain:
                            # Pretrain: push raw network output → 0 (matches HJ exact-model pretrain)
                            pretrain_results = self.model({'coords': model_input['tc_model_coords_curr']})
                            pretrain_output = pretrain_results['model_out'].squeeze(dim=-1)
                            losses = {'pretrain': torch.abs(pretrain_output).sum()}
                        else:
                            curr_results = self.model({'coords': model_input['tc_model_coords_curr']})
                            values_curr = self.dataset.dynamics.io_to_value(
                                curr_results['model_in'].detach(),
                                curr_results['model_out'].squeeze(dim=-1),
                            )
                            dvs_curr = self.dataset.dynamics.io_to_dv(
                                curr_results['model_in'],
                                curr_results['model_out'].squeeze(dim=-1),
                            )
                            dvdt_curr = dvs_curr[..., 0]
                            dvds_curr = dvs_curr[..., 1:]

                            losses = loss_fn(
                                values_curr,
                                dvdt_curr,
                                dvds_curr,
                                model_input['tc_obs_flow'],
                                gt['tc_boundary_values'],
                            )

                            # Dirichlet loss: two-sided boundary anchor at t=0
                            if self.dataset.num_src_samples > 0:
                                dirichlet_mask = (curr_results['model_in'][..., 0] == self.dataset.tMin)
                                if dirichlet_mask.any():
                                    if self.dataset.dynamics.deepreach_model == 'exact':
                                        # For exact model, V=l(x) at t=0 by construction,
                                        # so |V-l| is trivially 0. Instead push raw output→0
                                        # to prevent dvdt = 50*output from drifting unconstrained.
                                        dirichlet_loss = torch.abs(
                                            curr_results['model_out'].squeeze(dim=-1)[dirichlet_mask]
                                        ).sum()
                                    else:
                                        # For diff/vanilla, directly anchor V = l(x) at t=0
                                        dirichlet_loss = torch.abs(
                                            values_curr[dirichlet_mask] - gt['tc_boundary_values'][dirichlet_mask]
                                        ).sum()
                                    losses['dirichlet'] = dirichlet_loss
                    else:
                        model_results = self.model({'coords': model_input['model_coords']})

                        states = self.dataset.dynamics.input_to_coord(model_results['model_in'].detach())[..., 1:]
                        values = self.dataset.dynamics.io_to_value(
                            model_results['model_in'].detach(),
                            model_results['model_out'].squeeze(dim=-1),
                        )
                        dvs = self.dataset.dynamics.io_to_dv(model_results['model_in'], model_results['model_out'].squeeze(dim=-1))
                        boundary_values = gt['boundary_values']
                        if self.dataset.dynamics.loss_type == 'brat_hjivi':
                            reach_values = gt['reach_values']
                            avoid_values = gt['avoid_values']
                        dirichlet_masks = gt['dirichlet_masks']

                        if self.dataset.dynamics.loss_type == 'brt_hjivi':
                            losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, dirichlet_masks, model_results['model_out'])
                        elif self.dataset.dynamics.loss_type == 'brat_hjivi':
                            losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, reach_values, avoid_values, dirichlet_masks, model_results['model_out'])
                        else:
                            raise NotImplementedError

                    if temporal_mode:
                        if 'pretrain' in losses:
                            train_loss = losses['pretrain'].mean()
                        else:
                            train_loss = 0.
                            for loss in losses.values():
                                train_loss += loss.mean()
                    else:
                        train_loss = 0.
                        for loss in losses.values():
                            train_loss += loss.mean()

                    supervised_loss = None
                    if supervised_weight > 0.0 and 'supervised_coords' in model_input and 'supervised_values' in gt:
                        sup_results = self.model({'coords': model_input['supervised_coords']})
                        sup_values = self.dataset.dynamics.io_to_value(
                            sup_results['model_in'], sup_results['model_out'].squeeze(dim=-1)
                        )
                        sup_targets = gt['supervised_values']
                        sup_errors = torch.pow(sup_values - sup_targets, 2)
                        if 'supervised_labels' in gt:
                            sup_labels = gt['supervised_labels']
                            sample_weights = torch.where(
                                sup_labels > 0.5,
                                torch.full_like(sup_labels, supervised_safe_weight),
                                torch.full_like(sup_labels, supervised_unsafe_weight),
                            )
                            supervised_loss = supervised_weight * torch.mean(sample_weights * sup_errors)
                        else:
                            supervised_loss = supervised_weight * torch.mean(sup_errors)

                    if use_lbfgs:
                        def closure():
                            optim.zero_grad()
                            closure_train_loss = train_loss
                            if supervised_loss is not None:
                                closure_train_loss += supervised_loss
                            closure_train_loss.backward()
                            return closure_train_loss
                        optim.step(closure)

                    # Adjust the relative magnitude of the losses if required
                    if (not temporal_mode) and self.dataset.dynamics.deepreach_model in ['vanilla', 'diff'] and adjust_relative_grads:
                        if losses['diff_constraint_hom'] > 0.01:
                            params = OrderedDict(self.model.named_parameters())
                            # Gradients with respect to the PDE loss
                            optim.zero_grad()
                            losses['diff_constraint_hom'].backward(retain_graph=True)
                            grads_PDE = []
                            for key, param in params.items():
                                grads_PDE.append(param.grad.view(-1))
                            grads_PDE = torch.cat(grads_PDE)

                            # Gradients with respect to the boundary loss
                            optim.zero_grad()
                            losses['dirichlet'].backward(retain_graph=True)
                            grads_dirichlet = []
                            for key, param in params.items():
                                grads_dirichlet.append(param.grad.view(-1))
                            grads_dirichlet = torch.cat(grads_dirichlet)

                            # # Plot the gradients
                            # import seaborn as sns
                            # import matplotlib.pyplot as plt
                            # fig = plt.figure(figsize=(5, 5))
                            # ax = fig.add_subplot(1, 1, 1)
                            # ax.set_yscale('symlog')
                            # sns.distplot(grads_PDE.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                            # sns.distplot(grads_dirichlet.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                            # fig.savefig('gradient_visualization.png')

                            # fig = plt.figure(figsize=(5, 5))
                            # ax = fig.add_subplot(1, 1, 1)
                            # ax.set_yscale('symlog')
                            # grads_dirichlet_normalized = grads_dirichlet * torch.mean(torch.abs(grads_PDE))/torch.mean(torch.abs(grads_dirichlet))
                            # sns.distplot(grads_PDE.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                            # sns.distplot(grads_dirichlet_normalized.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                            # ax.set_xlim([-1000.0, 1000.0])
                            # fig.savefig('gradient_visualization_normalized.png')

                            # Set the new weight according to the paper
                            # num = torch.max(torch.abs(grads_PDE))
                            num = torch.mean(torch.abs(grads_PDE))
                            den = torch.mean(torch.abs(grads_dirichlet))
                            new_weight = 0.9*new_weight + 0.1*num/den
                            losses['dirichlet'] = new_weight*losses['dirichlet']
                        writer.add_scalar('weight_scaling', new_weight, total_steps)
                        train_loss = 0.
                        for loss in losses.values():
                            train_loss += loss.mean()

                    # import ipdb; ipdb.set_trace()
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()

                        if loss_name == 'dirichlet' and (not temporal_mode):
                            writer.add_scalar(loss_name, single_loss/new_weight, total_steps)
                        else:
                            writer.add_scalar(loss_name, single_loss, total_steps)
                    if supervised_loss is not None:
                        writer.add_scalar("supervised_loss", supervised_loss, total_steps)
                        train_loss += supervised_loss

                    train_loss_mean = train_loss.item() / self.dataset.numpoints
                    train_losses.append(train_loss_mean)
                    writer.add_scalar("total_train_loss", train_loss_mean, total_steps)

                    if not total_steps % steps_til_summary:
                        pass  # checkpoint saving handled by top-K best logic
                        # summary_fn(model, model_input, gt, model_output, writer, total_steps)

                    if not use_lbfgs:
                        optim.zero_grad()
                        train_loss.backward()

                        if clip_grad:
                            if isinstance(clip_grad, bool):
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.)
                            else:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad)

                        optim.step()

                    pbar.update(1)

                    if not total_steps % steps_til_summary:
                        tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss_mean, time.time() - start_time))
                        if self.use_wandb:
                            wandb_payload = {
                                'step': epoch,
                                'train_loss': train_loss_mean,
                            }
                            if 'diff_constraint_hom' in losses:
                                wandb_payload['pde_loss'] = losses['diff_constraint_hom'].item() / self.dataset.numpoints
                            if 'dirichlet' in losses:
                                wandb_payload['dirichlet_loss'] = losses['dirichlet'].item() / self.dataset.numpoints
                            wandb.log(wandb_payload)

                    total_steps += 1

                # --- Early stopping val check ---
                early_stopped = False
                if (early_stop_enabled
                        and not self.dataset.pretrain
                        and total_steps > 0
                        and not total_steps % steps_til_summary):
                    val_loss = _compute_val_loss()
                    writer.add_scalar("val_loss", val_loss, total_steps)
                    if self.use_wandb:
                        wandb.log({'step': epoch, 'val_loss': val_loss})
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_model_state = copy.deepcopy(self.model.state_dict())
                        tqdm.write(f"[EarlyStopping] New best val_loss={val_loss:.6f} at epoch {epoch}")

                    # Top-K best checkpoint management
                    save_this = (len(best_checkpoints) < top_k_best
                                 or val_loss < best_checkpoints[-1][0])
                    if save_this:
                        ckpt_name = f'model_best_epoch{epoch}_val{val_loss:.6f}.pth'
                        torch.save(self.model.state_dict(),
                                   os.path.join(checkpoints_dir, ckpt_name))
                        best_checkpoints.append((val_loss, ckpt_name))
                        best_checkpoints.sort(key=lambda x: x[0])
                        # Remove worst if over budget
                        while len(best_checkpoints) > top_k_best:
                            _, removed_name = best_checkpoints.pop()
                            removed_path = os.path.join(checkpoints_dir, removed_name)
                            if os.path.exists(removed_path):
                                os.remove(removed_path)
                                tqdm.write(f"[TopK] Removed {removed_name}")

                    if val_loss >= best_val_loss:
                        patience_counter += 1
                        tqdm.write(f"[EarlyStopping] val_loss={val_loss:.6f} (best={best_val_loss:.6f}), "
                                   f"patience {patience_counter}/{early_stopping_patience}")
                    if patience_counter >= early_stopping_patience:
                        tqdm.write(f"[EarlyStopping] Stopping at epoch {epoch} "
                                   f"(no improvement for {early_stopping_patience} checks)")
                        early_stopped = True

                if early_stopped:
                    break

                # cost-supervised learning (CSL) phase
                if use_CSL and not self.dataset.pretrain and (epoch-last_CSL_epoch) >= epochs_til_CSL:
                    last_CSL_epoch = epoch
                    
                    # generate CSL datasets
                    self.model.eval()

                    CSL_dataset = scenario_optimization(
                        device=device, model=self.model, policy=self.model, dynamics=self.dataset.dynamics,
                        tMin=self.dataset.tMin, tMax=CSL_tMax, dt=CSL_dt,
                        set_type="BRT", control_type="value", # TODO: implement option for BRS too
                        scenario_batch_size=min(num_CSL_samples, 100000), sample_batch_size=min(10*num_CSL_samples, 1000000),
                        sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=0.0),
                        max_scenarios=num_CSL_samples, tStart_generator=lambda n : torch.zeros(n).uniform_(self.dataset.tMin, CSL_tMax)
                    )
                    CSL_coords = torch.cat((CSL_dataset['times'].unsqueeze(-1), CSL_dataset['states']), dim=-1)
                    CSL_costs = CSL_dataset['costs']

                    num_CSL_val_samples = int(0.1*num_CSL_samples)
                    CSL_val_dataset = scenario_optimization(
                        model=self.model, policy=self.model, dynamics=self.dataset.dynamics,
                        tMin=self.dataset.tMin, tMax=CSL_tMax, dt=CSL_dt,
                        set_type="BRT", control_type="value", # TODO: implement option for BRS too
                        scenario_batch_size=min(num_CSL_val_samples, 100000), sample_batch_size=min(10*num_CSL_val_samples, 1000000),
                        sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=0.0),
                        max_scenarios=num_CSL_val_samples, tStart_generator=lambda n : torch.zeros(n).uniform_(self.dataset.tMin, CSL_tMax)
                    )
                    CSL_val_coords = torch.cat((CSL_val_dataset['times'].unsqueeze(-1), CSL_val_dataset['states']), dim=-1)
                    CSL_val_costs = CSL_val_dataset['costs']

                    CSL_val_tMax_dataset = scenario_optimization(
                        model=self.model, policy=self.model, dynamics=self.dataset.dynamics,
                        tMin=self.dataset.tMin, tMax=self.dataset.tMax, dt=CSL_dt,
                        set_type="BRT", control_type="value", # TODO: implement option for BRS too
                        scenario_batch_size=min(num_CSL_val_samples, 100000), sample_batch_size=min(10*num_CSL_val_samples, 1000000),
                        sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=0.0),
                        max_scenarios=num_CSL_val_samples # no tStart_generator, since I want all tMax times
                    )
                    CSL_val_tMax_coords = torch.cat((CSL_val_tMax_dataset['times'].unsqueeze(-1), CSL_val_tMax_dataset['states']), dim=-1)
                    CSL_val_tMax_costs = CSL_val_tMax_dataset['costs']
                    
                    self.model.train()

                    # CSL optimizer
                    CSL_optim = torch.optim.Adam(lr=CSL_lr, params=self.model.parameters())

                    # initial CSL val loss
                    CSL_val_results = self.model({'coords': self.dataset.dynamics.coord_to_input(CSL_val_coords.to(device))})
                    CSL_val_preds = self.dataset.dynamics.io_to_value(CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
                    CSL_val_errors = CSL_val_preds - CSL_val_costs.to(device)
                    CSL_val_loss = torch.mean(torch.pow(CSL_val_errors, 2))
                    CSL_initial_val_loss = CSL_val_loss
                    if self.use_wandb:
                        wandb.log({
                            "step": epoch,
                            "CSL_val_loss": CSL_val_loss.item()
                        })

                    # initial self-supervised learning (SSL) val loss
                    # right now, just took code from dataio.py and the SSL training loop above; TODO: refactor all this for cleaner modular code
                    CSL_val_states = CSL_val_coords[..., 1:].to(device)
                    CSL_val_dvs = self.dataset.dynamics.io_to_dv(CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
                    CSL_val_boundary_values = self.dataset.dynamics.boundary_fn(CSL_val_states)
                    if self.dataset.dynamics.loss_type == 'brat_hjivi':
                        CSL_val_reach_values = self.dataset.dynamics.reach_fn(CSL_val_states)
                        CSL_val_avoid_values = self.dataset.dynamics.avoid_fn(CSL_val_states)
                    CSL_val_dirichlet_masks = CSL_val_coords[:, 0].to(device) == self.dataset.tMin # assumes time unit in dataset (model) is same as real time units
                    if self.dataset.dynamics.loss_type == 'brt_hjivi':
                        SSL_val_losses = loss_fn(CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_dirichlet_masks)
                    elif self.dataset.dynamics.loss_type == 'brat_hjivi':
                        SSL_val_losses = loss_fn(CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_reach_values, CSL_val_avoid_values, CSL_val_dirichlet_masks)
                    else:
                        NotImplementedError
                    SSL_val_loss = SSL_val_losses['diff_constraint_hom'].mean() # I assume there is no dirichlet (boundary) loss here, because I do not ever explicitly generate source samples at tMin (i.e. torch.all(CSL_val_dirichlet_masks == False))
                    if self.use_wandb:
                        wandb.log({
                            "step": epoch,
                            "SSL_val_loss": SSL_val_loss.item()
                        })

                    # CSL training loop
                    for CSL_epoch in tqdm(range(max_CSL_epochs)):
                        CSL_idxs = torch.randperm(num_CSL_samples)
                        for CSL_batch in range(math.ceil(num_CSL_samples/CSL_batch_size)):
                            CSL_batch_idxs = CSL_idxs[CSL_batch*CSL_batch_size:(CSL_batch+1)*CSL_batch_size]
                            CSL_batch_coords = CSL_coords[CSL_batch_idxs]

                            CSL_batch_results = self.model({'coords': self.dataset.dynamics.coord_to_input(CSL_batch_coords.to(device))})
                            CSL_batch_preds = self.dataset.dynamics.io_to_value(CSL_batch_results['model_in'], CSL_batch_results['model_out'].squeeze(dim=-1))
                            CSL_batch_costs = CSL_costs[CSL_batch_idxs].to(device)
                            CSL_batch_errors = CSL_batch_preds - CSL_batch_costs
                            CSL_batch_loss = CSL_loss_weight*torch.mean(torch.pow(CSL_batch_errors, 2))

                            CSL_batch_states = CSL_batch_coords[..., 1:].to(device)
                            CSL_batch_dvs = self.dataset.dynamics.io_to_dv(CSL_batch_results['model_in'], CSL_batch_results['model_out'].squeeze(dim=-1))
                            CSL_batch_boundary_values = self.dataset.dynamics.boundary_fn(CSL_batch_states)
                            if self.dataset.dynamics.loss_type == 'brat_hjivi':
                                CSL_batch_reach_values = self.dataset.dynamics.reach_fn(CSL_batch_states)
                                CSL_batch_avoid_values = self.dataset.dynamics.avoid_fn(CSL_batch_states)
                            CSL_batch_dirichlet_masks = CSL_batch_coords[:, 0].to(device) == self.dataset.tMin # assumes time unit in dataset (model) is same as real time units
                            if self.dataset.dynamics.loss_type == 'brt_hjivi':
                                SSL_batch_losses = loss_fn(CSL_batch_states, CSL_batch_preds, CSL_batch_dvs[..., 0], CSL_batch_dvs[..., 1:], CSL_batch_boundary_values, CSL_batch_dirichlet_masks)
                            elif self.dataset.dynamics.loss_type == 'brat_hjivi':
                                SSL_batch_losses = loss_fn(CSL_batch_states, CSL_batch_preds, CSL_batch_dvs[..., 0], CSL_batch_dvs[..., 1:], CSL_batch_boundary_values, CSL_batch_reach_values, CSL_batch_avoid_values, CSL_batch_dirichlet_masks)
                            else:
                                NotImplementedError
                            SSL_batch_loss = SSL_batch_losses['diff_constraint_hom'].mean() # I assume there is no dirichlet (boundary) loss here, because I do not ever explicitly generate source samples at tMin (i.e. torch.all(CSL_batch_dirichlet_masks == False))
                            
                            CSL_optim.zero_grad()
                            SSL_batch_loss.backward(retain_graph=True)
                            if (not use_lbfgs) and clip_grad: # no adjust_relative_grads, because I assume even with adjustment, the diff_constraint_hom remains unaffected and the only other loss (dirichlet) is zero
                                if isinstance(clip_grad, bool):
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.)
                                else:
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad)
                            CSL_batch_loss.backward()
                            CSL_optim.step()
                        
                        # evaluate on CSL_val_dataset
                        CSL_val_results = self.model({'coords': self.dataset.dynamics.coord_to_input(CSL_val_coords.to(device))})
                        CSL_val_preds = self.dataset.dynamics.io_to_value(CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
                        CSL_val_errors = CSL_val_preds - CSL_val_costs.to(device)
                        CSL_val_loss = torch.mean(torch.pow(CSL_val_errors, 2))
                    
                        CSL_val_states = CSL_val_coords[..., 1:].to(device)
                        CSL_val_dvs = self.dataset.dynamics.io_to_dv(CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
                        CSL_val_boundary_values = self.dataset.dynamics.boundary_fn(CSL_val_states)
                        if self.dataset.dynamics.loss_type == 'brat_hjivi':
                            CSL_val_reach_values = self.dataset.dynamics.reach_fn(CSL_val_states)
                            CSL_val_avoid_values = self.dataset.dynamics.avoid_fn(CSL_val_states)
                        CSL_val_dirichlet_masks = CSL_val_coords[:, 0].to(device) == self.dataset.tMin # assumes time unit in dataset (model) is same as real time units
                        if self.dataset.dynamics.loss_type == 'brt_hjivi':
                            SSL_val_losses = loss_fn(CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_dirichlet_masks)
                        elif self.dataset.dynamics.loss_type == 'brat_hjivi':
                            SSL_val_losses = loss_fn(CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_reach_values, CSL_val_avoid_values, CSL_val_dirichlet_masks)
                        else:
                            raise NotImplementedError
                        SSL_val_loss = SSL_val_losses['diff_constraint_hom'].mean() # I assume there is no dirichlet (boundary) loss here, because I do not ever explicitly generate source samples at tMin (i.e. torch.all(CSL_val_dirichlet_masks == False))
                    
                        CSL_val_tMax_results = self.model({'coords': self.dataset.dynamics.coord_to_input(CSL_val_tMax_coords.to(device))})
                        CSL_val_tMax_preds = self.dataset.dynamics.io_to_value(CSL_val_tMax_results['model_in'], CSL_val_tMax_results['model_out'].squeeze(dim=-1))
                        CSL_val_tMax_errors = CSL_val_tMax_preds - CSL_val_tMax_costs.to(device)
                        CSL_val_tMax_loss = torch.mean(torch.pow(CSL_val_tMax_errors, 2))
                        
                        # log CSL losses, recovered_safe_set_fracs
                        if self.dataset.dynamics.set_mode == 'reach':
                            CSL_train_batch_theoretically_recoverable_safe_set_frac = torch.sum(CSL_batch_costs.to(device) < 0) / len(CSL_batch_preds)
                            CSL_train_batch_recovered_safe_set_frac = torch.sum(CSL_batch_preds < torch.min(CSL_batch_preds[CSL_batch_costs.to(device) > 0])) / len(CSL_batch_preds)
                            CSL_val_theoretically_recoverable_safe_set_frac = torch.sum(CSL_val_costs.to(device) < 0) / len(CSL_val_preds)
                            CSL_val_recovered_safe_set_frac = torch.sum(CSL_val_preds < torch.min(CSL_val_preds[CSL_val_costs.to(device) > 0])) / len(CSL_val_preds)
                            CSL_val_tMax_theoretically_recoverable_safe_set_frac = torch.sum(CSL_val_tMax_costs.to(device) < 0) / len(CSL_val_tMax_preds)
                            CSL_val_tMax_recovered_safe_set_frac = torch.sum(CSL_val_tMax_preds < torch.min(CSL_val_tMax_preds[CSL_val_tMax_costs.to(device) > 0])) / len(CSL_val_tMax_preds)
                        elif self.dataset.dynamics.set_mode == 'avoid':
                            CSL_train_batch_theoretically_recoverable_safe_set_frac = torch.sum(CSL_batch_costs.to(device) > 0) / len(CSL_batch_preds)
                            CSL_train_batch_recovered_safe_set_frac = torch.sum(CSL_batch_preds > torch.max(CSL_batch_preds[CSL_batch_costs.to(device) < 0])) / len(CSL_batch_preds)
                            CSL_val_theoretically_recoverable_safe_set_frac = torch.sum(CSL_val_costs.to(device) > 0) / len(CSL_val_preds)
                            CSL_val_recovered_safe_set_frac = torch.sum(CSL_val_preds > torch.max(CSL_val_preds[CSL_val_costs.to(device) < 0])) / len(CSL_val_preds)
                            CSL_val_tMax_theoretically_recoverable_safe_set_frac = torch.sum(CSL_val_tMax_costs.to(device) > 0) / len(CSL_val_tMax_preds)
                            CSL_val_tMax_recovered_safe_set_frac = torch.sum(CSL_val_tMax_preds > torch.max(CSL_val_tMax_preds[CSL_val_tMax_costs.to(device) < 0])) / len(CSL_val_tMax_preds)
                        else:
                            raise NotImplementedError
                        if self.use_wandb:
                            wandb.log({
                                "step": epoch+(CSL_epoch+1)*int(0.5*epochs_til_CSL/max_CSL_epochs),
                                "CSL_train_batch_loss": CSL_batch_loss.item(),
                                "SSL_train_batch_loss": SSL_batch_loss.item(),
                                "CSL_val_loss": CSL_val_loss.item(),
                                "SSL_val_loss": SSL_val_loss.item(),
                                "CSL_val_tMax_loss": CSL_val_tMax_loss.item(),
                                "CSL_train_batch_theoretically_recoverable_safe_set_frac": CSL_train_batch_theoretically_recoverable_safe_set_frac.item(),
                                "CSL_val_theoretically_recoverable_safe_set_frac": CSL_val_theoretically_recoverable_safe_set_frac.item(),
                                "CSL_val_tMax_theoretically_recoverable_safe_set_frac": CSL_val_tMax_theoretically_recoverable_safe_set_frac.item(),
                                "CSL_train_batch_recovered_safe_set_frac": CSL_train_batch_recovered_safe_set_frac.item(),
                                "CSL_val_recovered_safe_set_frac": CSL_val_recovered_safe_set_frac.item(),
                                "CSL_val_tMax_recovered_safe_set_frac": CSL_val_tMax_recovered_safe_set_frac.item(),
                            })

                        if CSL_val_loss < CSL_loss_frac_cutoff*CSL_initial_val_loss:
                            break

                if not (epoch+1) % epochs_til_checkpoint:
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % (epoch+1)),
                        np.array(train_losses))
                    self.validate(
                        device=device, epoch=epoch+1, save_path=os.path.join(checkpoints_dir, 'BRS_validation_plot_epoch_%04d.png' % (epoch+1)),
                        x_resolution = val_x_resolution, y_resolution = val_y_resolution, z_resolution=val_z_resolution, time_resolution=val_time_resolution)

        # --- Post-training: restore best weights and save model_final ---
        if best_checkpoints:
            best_loss, best_name = best_checkpoints[0]
            best_path = os.path.join(checkpoints_dir, best_name)
            self.model.load_state_dict(torch.load(best_path, map_location=device))
            print(f"[PostTraining] Loaded best checkpoint {best_name} (val_loss={best_loss:.6f})")
        elif early_stop_enabled and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"[EarlyStopping] Restored best model weights (val_loss={best_val_loss:.6f})")

        torch.save(self.model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))

        # --- Post-training eval on train/val/test sets ---
        if temporal_mode:
            self.model.eval()
            self.model.requires_grad_(False)
            import csv

            def _eval_and_save_csv(states, labels, csv_path):
                """Batched forward pass at tMax → CSV with state cols, label, value."""
                times = torch.full((states.shape[0], 1), self.dataset.tMax)
                coords = torch.cat((times, states), dim=-1)
                model_coords = self.dataset._coord_to_model_input(coords)
                values = []
                batch_sz = eval_batch_size
                for start in range(0, model_coords.shape[0], batch_sz):
                    end = min(start + batch_sz, model_coords.shape[0])
                    batch = model_coords[start:end].to(device)
                    with torch.no_grad():
                        results = self.model({'coords': batch})
                        v = self.dataset.dynamics.io_to_value(
                            results['model_in'], results['model_out'].squeeze(dim=-1)
                        )
                    values.append(v.cpu())
                values = torch.cat(values, dim=0)
                sd = self.dataset.dynamics.state_dim
                with open(csv_path, 'w', newline='') as f:
                    writer_csv = csv.writer(f)
                    header = [f's{i}' for i in range(sd)] + ['label', 'value']
                    writer_csv.writerow(header)
                    for j in range(states.shape[0]):
                        row = [f'{states[j, k].item():.6f}' for k in range(sd)]
                        row.append(str(int(labels[j].item())))
                        row.append(f'{values[j].item():.6f}')
                        writer_csv.writerow(row)
                print(f"[PostTrainEval] Saved {states.shape[0]} rows to {csv_path}")

            # Eval on train/val trajectory sets
            if (hasattr(self.dataset, 'get_all_states_and_labels')
                    and self.dataset.traj_labels is not None):
                for split in ['train', 'val']:
                    states, labels = self.dataset.get_all_states_and_labels(split=split)
                    if states is None:
                        continue
                    _eval_and_save_csv(states, labels,
                                       os.path.join(checkpoints_dir, f'eval_{split}.csv'))

            # Eval on test set (initial conditions with labels, e.g. test_set.txt)
            data_root = getattr(self.dataset, 'data_root', None)
            if data_root is not None:
                sd = self.dataset.dynamics.state_dim
                test_candidates = ['test_set.txt', 'cal_set.txt']
                for candidate in test_candidates:
                    test_path = os.path.join(data_root, candidate)
                    if os.path.exists(test_path):
                        test_data = torch.tensor(
                            self.dataset._load_txt(test_path), dtype=torch.float32)
                        if test_data.ndim == 1:
                            test_data = test_data.unsqueeze(0)
                        test_states = test_data[:, :sd]
                        test_labels = test_data[:, -1]
                        _eval_and_save_csv(test_states, test_labels,
                                           os.path.join(checkpoints_dir, 'eval_test.csv'))
                        print(f"[PostTrainEval] Test set source: {test_path}")
                        break

            self.model.train()
            self.model.requires_grad_(True)

        if was_eval:
            self.model.eval()
            self.model.requires_grad_(False)

    def test(self, device, current_time, last_checkpoint, checkpoint_dt, dt, num_scenarios, num_violations, set_type, control_type, data_step, checkpoint_toload=None):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)

        testing_dir = os.path.join(self.experiment_dir, 'testing_%s' % current_time.strftime('%m_%d_%Y_%H_%M'))
        if os.path.exists(testing_dir):
            overwrite = input("The testing directory %s already exists. Overwrite? (y/n)"%testing_dir)
            if not (overwrite == 'y'):
                print('Exiting.')
                quit()
            shutil.rmtree(testing_dir)
        os.makedirs(testing_dir)

        if checkpoint_toload is None:
            print('running cross-checkpoint testing')

            for i in tqdm(range(sidelen), desc='Checkpoint'):
                self._load_checkpoint(epoch=checkpoints[i])
                raise NotImplementedError

        else:
            print('running specific-checkpoint testing')
            self._load_checkpoint(checkpoint_toload)

            model = self.model
            dataset = self.dataset
            dynamics = dataset.dynamics
            raise NotImplementedError

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)

class DeepReach(Experiment):
    def init_special(self):
        pass
