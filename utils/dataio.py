import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import math

# uses model input and real boundary fn
class CustomDataset(Dataset):
    def __init__(self, dynamics, numpoints, pretrain, pretrain_iters, tMin, tMax, counter_start, counter_end, num_src_samples, num_target_samples, file_path):
        self.dynamics = dynamics
        self.numpoints = numpoints
        self.pretrain = pretrain
        self.pretrain_counter = 0
        self.pretrain_iters = pretrain_iters
        self.tMin = tMin 
        self.tMax = tMax 
        self.counter = counter_start 
        self.counter_end = counter_end 
        self.num_src_samples = num_src_samples
        self.num_target_samples = num_target_samples
        self.data = pd.read_csv(file_path)

        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            data = self.data.iloc[idx]
            time = data['time']
            state = data['state']
            return time, state







class ReachabilityDataset(Dataset):
    def __init__(self, dynamics, numpoints, pretrain, pretrain_iters, tMin, tMax, counter_start, counter_end, num_src_samples, num_target_samples):
        self.dynamics = dynamics
        self.numpoints = numpoints
        self.pretrain = pretrain
        self.pretrain_counter = 0
        self.pretrain_iters = pretrain_iters
        self.tMin = tMin 
        self.tMax = tMax 
        self.counter = counter_start 
        self.counter_end = counter_end 
        self.num_src_samples = num_src_samples
        self.num_target_samples = num_target_samples

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # uniformly sample domain and include coordinates where source is non-zero 
        model_states = torch.zeros(self.numpoints, self.dynamics.state_dim).uniform_(-1, 1)
        if self.num_target_samples > 0:
            target_state_samples = self.dynamics.sample_target_state(self.num_target_samples)
            model_states[-self.num_target_samples:] = self.dynamics.coord_to_input(torch.cat((torch.zeros(self.num_target_samples, 1), target_state_samples), dim=-1))[:, 1:self.dynamics.state_dim+1]

        if self.pretrain:
            # only sample in time around the initial condition
            times = torch.full((self.numpoints, 1), self.tMin)
        else:
            # slowly grow time values from start time
            if self.counter_end > 0:
                time_window = (self.tMax - self.tMin) * (self.counter / self.counter_end)
            else:
                time_window = self.tMax - self.tMin  # Fallback if counter_end is invalid
            times = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, time_window)
            # make sure we always have training samples at the initial time
            times[-self.num_src_samples:, 0] = self.tMin
        model_coords = torch.cat((times, model_states), dim=1)        
        if self.dynamics.input_dim > self.dynamics.state_dim + 1: # temporary workaround for having to deal with dynamics classes for parametrized models with extra inputs
            model_coords = torch.cat((model_coords, torch.zeros(self.numpoints, self.dynamics.input_dim - self.dynamics.state_dim - 1)), dim=1)      

        boundary_values = self.dynamics.boundary_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])
        if self.dynamics.loss_type == 'brat_hjivi':
            reach_values = self.dynamics.reach_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])
            avoid_values = self.dynamics.avoid_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])
        
        if self.pretrain:
            dirichlet_masks = torch.ones(model_coords.shape[0]) > 0
        else:
            # only enforce initial conditions around self.tMin
            dirichlet_masks = (model_coords[:, 0] == self.tMin)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.counter_end:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        if self.dynamics.loss_type == 'brt_hjivi':
            return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'dirichlet_masks': dirichlet_masks}
        elif self.dynamics.loss_type == 'brat_hjivi':
            return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'reach_values': reach_values, 'avoid_values': avoid_values, 'dirichlet_masks': dirichlet_masks}
        else:
            raise NotImplementedError


class CartPoleDataset(Dataset):
    """
    Loads CartPole trajectories and optional supervised ROA labels.

    Supports two training objectives:
    - hj_pde (default): original DeepReach PDE batch format.
    - temporal_consistency: observed-flow PDE residual batches from trajectory data.
    """
    def __init__(
        self, dynamics, numpoints, pretrain, pretrain_iters, tMin, tMax,
        counter_start, counter_end, num_src_samples, num_target_samples,
        data_root, dt=0.01, num_supervised=0, supervised_value_safe=-1.0, supervised_value_unsafe=1.0,
        supervised_labels_file=None, supervised_balanced_sampling=False, trajectory_uniform_sampling=False,
        max_trajectory_files=0, use_shuffled_indices_only=False, shuffled_indices_file=None,
        load_trajectories_in_ram=False,
        training_objective='hj_pde', tc_target_mode='one_step', tc_n_step=4, tc_sample_terminal=False,
        angle_wrap_dims=None,
    ):
        self.dynamics = dynamics
        self.numpoints = numpoints
        self.pretrain = pretrain
        self.pretrain_counter = 0
        self.pretrain_iters = pretrain_iters
        self.tMin = tMin
        self.tMax = tMax
        self.counter = counter_start
        self.counter_end = counter_end
        self.num_src_samples = num_src_samples
        self.num_target_samples = num_target_samples
        self.data_root = data_root
        self.dt = dt
        self.pad_short_trajectories = True
        self.required_traj_points = self._required_traj_points_for_tmax()
        self.num_supervised = num_supervised
        self.supervised_value_safe = supervised_value_safe
        self.supervised_value_unsafe = supervised_value_unsafe
        self.supervised_labels_file = supervised_labels_file
        self.supervised_balanced_sampling = supervised_balanced_sampling
        self.trajectory_uniform_sampling = trajectory_uniform_sampling
        self.max_trajectory_files = max_trajectory_files
        self.use_shuffled_indices_only = use_shuffled_indices_only
        self.shuffled_indices_file = shuffled_indices_file
        self.load_trajectories_in_ram = load_trajectories_in_ram

        self.training_objective = training_objective
        self.tc_target_mode = tc_target_mode
        self.tc_n_step = tc_n_step
        self.tc_sample_terminal = tc_sample_terminal
        self.angle_wrap_dims = angle_wrap_dims if angle_wrap_dims is not None else [1]
        if self.training_objective not in ['hj_pde', 'temporal_consistency']:
            raise RuntimeError(
                f"Unsupported training_objective={self.training_objective}. Expected 'hj_pde' or 'temporal_consistency'."
            )
        if self.tc_target_mode not in ['one_step', 'n_step']:
            raise RuntimeError(
                f"Unsupported tc_target_mode={self.tc_target_mode}. Expected 'one_step' or 'n_step'."
            )
        if self.tc_n_step < 1:
            raise RuntimeError("tc_n_step must be >= 1.")
        if self.training_objective == 'temporal_consistency' and self.dt <= 0:
            raise RuntimeError("Temporal consistency mode requires dt > 0 to compute observed flow.")

        traj_dir = os.path.join(self.data_root, "trajectories")
        self.traj_files = sorted(
            [os.path.join(traj_dir, f) for f in os.listdir(traj_dir)
             if f.startswith("sequence_") and f.endswith(".txt")]
        )
        if self.use_shuffled_indices_only:
            shuffled_path = self._resolve_shuffled_indices_file()
            self.traj_files = self._filter_traj_files_by_shuffled_indices(self.traj_files, shuffled_path)
        if not self.traj_files:
            raise RuntimeError(f"No trajectory files found in {traj_dir}")
        if self.max_trajectory_files > 0 and self.max_trajectory_files < len(self.traj_files):
            # When using shuffled indices, take first N (deterministic). Otherwise random sample.
            if self.use_shuffled_indices_only:
                self.traj_files = self.traj_files[: self.max_trajectory_files]
            else:
                perm = torch.randperm(len(self.traj_files))
                keep = perm[:self.max_trajectory_files].tolist()
                self.traj_files = [self.traj_files[i] for i in keep]

        self.traj_state_cache = None
        self.traj_lengths = None
        if self.load_trajectories_in_ram:
            self.traj_state_cache = self._load_trajectories_in_ram()
            self.traj_lengths = torch.tensor(
                [traj_states.shape[0] for traj_states in self.traj_state_cache],
                dtype=torch.long
            )

        self.traj_sampling_weights = None
        if self.trajectory_uniform_sampling:
            if self.traj_lengths is None:
                self.traj_lengths = self._load_or_build_traj_lengths()
            total_points = float(self.traj_lengths.sum().item())
            if total_points <= 0.0:
                raise RuntimeError("Trajectory uniform sampling requested, but total trajectory points is zero.")
            self.traj_sampling_weights = self.traj_lengths.float() / total_points

        self.supervised_states = None
        self.supervised_labels = None
        self.safe_supervised_idx = None
        self.unsafe_supervised_idx = None
        # Supervised labels
        # Supported formats:
        # - roa_labels.txt: x,theta,x_dot,theta_dot,label (5 cols)
        # - cal_set.txt / test_set.txt: x,theta,x_dot,theta_dot,x_next,theta_next,x_dot_next,theta_dot_next,label (9 cols)
        if self.supervised_labels_file:
            sup_path = self.supervised_labels_file
            if not os.path.isabs(sup_path):
                sup_path = os.path.join(self.data_root, sup_path)
            if not os.path.exists(sup_path):
                raise RuntimeError(f"Supervised label file not found: {sup_path}")
        else:
            roa_path = os.path.join(self.data_root, "roa_labels.txt")
            cal_path = os.path.join(self.data_root, "cal_set.txt")
            sup_path = roa_path if os.path.exists(roa_path) else (cal_path if os.path.exists(cal_path) else None)

        if sup_path is not None:
            sd = self.dynamics.state_dim
            sup = torch.tensor(self._load_txt(sup_path), dtype=torch.float32)
            if sup.ndim == 1:
                sup = sup[None, :]
            if sup.shape[1] >= 2 * sd + 1:
                # cal_set format: state, next_state, label
                self.supervised_states = sup[:, :sd]
                self.supervised_labels = sup[:, -1]
            elif sup.shape[1] >= sd + 1:
                # roa_labels format: state, label
                self.supervised_states = sup[:, :sd]
                self.supervised_labels = sup[:, sd]
            else:
                raise RuntimeError(
                    f"Supervised label file {sup_path} has {sup.shape[1]} columns; "
                    f"expected at least {sd + 1} (state_dim={sd} + label)."
                )

            safe_mask = self.supervised_labels > 0.5
            self.safe_supervised_idx = torch.where(safe_mask)[0]
            self.unsafe_supervised_idx = torch.where(~safe_mask)[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.training_objective == 'temporal_consistency':
            model_input, gt = self._sample_temporal_batch()
        else:
            model_input, gt = self._sample_hj_batch()

        self._advance_counters()
        self._append_supervised_samples(model_input, gt)
        return model_input, gt

    def _sample_hj_batch(self):
        if self.traj_sampling_weights is not None:
            file_indices = torch.multinomial(self.traj_sampling_weights, self.numpoints, replacement=True)
        else:
            file_indices = torch.randint(0, len(self.traj_files), (self.numpoints,))

        coords = torch.zeros(self.numpoints, self.dynamics.state_dim + 1)
        counts = {}
        for file_idx in file_indices.tolist():
            counts[file_idx] = counts.get(file_idx, 0) + 1

        pos = 0
        for f_idx, k in counts.items():
            states = self._load_states_by_file_index(f_idx)
            if states.shape[0] == 0:
                continue
            row_idx = torch.randint(0, states.shape[0], (k,))
            times = row_idx.float().unsqueeze(-1) * self.dt + self.tMin
            coords[pos:pos+k] = torch.cat((times, states[row_idx]), dim=-1)
            pos += k

        if self.pretrain:
            coords[:, 0] = self.tMin
        else:
            if self.counter_end > 0:
                time_window = (self.tMax - self.tMin) * (self.counter / self.counter_end)
            else:
                time_window = self.tMax - self.tMin
            coords[:, 0] = torch.clamp(coords[:, 0], self.tMin, self.tMin + time_window)
            coords[-self.num_src_samples:, 0] = self.tMin

        model_coords = self._coord_to_model_input(coords)
        boundary_values = self.dynamics.boundary_fn(coords[..., 1:])
        if self.pretrain:
            dirichlet_masks = torch.ones(model_coords.shape[0]) > 0
        else:
            dirichlet_masks = (model_coords[:, 0] == self.tMin)

        return (
            {'model_coords': model_coords},
            {'boundary_values': boundary_values, 'dirichlet_masks': dirichlet_masks},
        )

    def _sample_temporal_batch(self):
        curr_coords_list = []
        obs_flow_list = []
        horizon_list = []
        boundary_list = []

        remaining = self.numpoints
        no_progress_attempts = 0
        max_no_progress_attempts = 5
        while remaining > 0 and no_progress_attempts < max_no_progress_attempts:
            sampled = self._sample_transition_batch_once(remaining)
            if sampled is None:
                no_progress_attempts += 1
                continue
            sampled_count = sampled['tc_model_coords_curr'].shape[0]
            if sampled_count == 0:
                no_progress_attempts += 1
                continue
            curr_coords_list.append(sampled['tc_model_coords_curr'])
            obs_flow_list.append(sampled['tc_obs_flow'])
            horizon_list.append(sampled['tc_horizon'])
            boundary_list.append(sampled['tc_boundary_values'])
            remaining -= sampled_count

        if remaining > 0:
            raise RuntimeError(
                "No valid transitions found; check trajectory lengths / dataset."
            )

        model_input = {
            'tc_model_coords_curr': torch.cat(curr_coords_list, dim=0),
            'tc_obs_flow': torch.cat(obs_flow_list, dim=0),
        }
        gt = {
            'tc_horizon': torch.cat(horizon_list, dim=0),
            'tc_boundary_values': torch.cat(boundary_list, dim=0),
        }
        return model_input, gt

    def _sample_transition_batch_once(self, batch_size):
        file_indices = self._sample_file_indices(batch_size)
        counts = {}
        for file_idx in file_indices.tolist():
            counts[file_idx] = counts.get(file_idx, 0) + 1

        curr_model_coords = []
        obs_flow_values = []
        horizons = []
        boundary_vals = []

        for f_idx, k in counts.items():
            states = self._load_states_by_file_index(f_idx)
            traj_len = states.shape[0]
            if traj_len < 2:
                continue

            start_idx = torch.randint(0, traj_len - 1, (k,), dtype=torch.long)
            next_idx = start_idx + 1
            horizon = torch.ones_like(start_idx)
            curr_states = states[start_idx]
            next_states = states[next_idx]
            boundary_vals.append(self.dynamics.boundary_fn(curr_states))
            curr_times = start_idx.float().unsqueeze(-1) * self.dt + self.tMin

            curr_coords = torch.cat((curr_times, curr_states), dim=-1)
            curr_model_coords.append(self._coord_to_model_input(curr_coords))

            state_delta = next_states - curr_states
            for wrap_dim in self.angle_wrap_dims:
                state_delta[..., wrap_dim] = self._wrapped_angle_diff(next_states[..., wrap_dim], curr_states[..., wrap_dim])
            obs_flow = state_delta / self.dt
            obs_flow_values.append(obs_flow)
            horizons.append(horizon)

        if not curr_model_coords:
            return None

        return {
            'tc_model_coords_curr': torch.cat(curr_model_coords, dim=0),
            'tc_obs_flow': torch.cat(obs_flow_values, dim=0),
            'tc_horizon': torch.cat(horizons, dim=0),
            'tc_boundary_values': torch.cat(boundary_vals, dim=0),
        }

    def _wrapped_angle_diff(self, theta_next, theta_curr):
        return (theta_next - theta_curr + math.pi) % (2 * math.pi) - math.pi

    def _sample_file_indices(self, num_samples):
        if self.traj_sampling_weights is not None:
            return torch.multinomial(self.traj_sampling_weights, num_samples, replacement=True)
        return torch.randint(0, len(self.traj_files), (num_samples,))

    def _load_states_by_file_index(self, file_index):
        if self.traj_state_cache is not None:
            return self.traj_state_cache[file_index]
        states = torch.tensor(self._load_txt(self.traj_files[file_index]), dtype=torch.float32)
        return self._pad_states_to_tmax(states)

    def _required_traj_points_for_tmax(self):
        if self.dt <= 0:
            return 1
        horizon = max(self.tMax - self.tMin, 0.0)
        return int(horizon / self.dt + 1e-9) + 1

    def _pad_states_to_tmax(self, states):
        if (not self.pad_short_trajectories) or states.shape[0] == 0:
            return states
        if states.shape[0] >= self.required_traj_points:
            return states
        pad_count = self.required_traj_points - states.shape[0]
        pad_block = states[-1:].repeat(pad_count, 1)
        return torch.cat((states, pad_block), dim=0)

    def _coord_to_model_input(self, coords):
        model_coords = self.dynamics.coord_to_input(coords)
        if self.dynamics.input_dim > self.dynamics.state_dim + 1:
            extra_dims = self.dynamics.input_dim - self.dynamics.state_dim - 1
            model_coords = torch.cat(
                (model_coords, torch.zeros(model_coords.shape[0], extra_dims, dtype=model_coords.dtype)),
                dim=1,
            )
        return model_coords

    def _append_supervised_samples(self, model_input, gt):
        if self.num_supervised <= 0 or self.supervised_states is None:
            return

        if (
            self.supervised_balanced_sampling
            and self.safe_supervised_idx is not None
            and self.unsafe_supervised_idx is not None
            and len(self.safe_supervised_idx) > 0
            and len(self.unsafe_supervised_idx) > 0
        ):
            num_safe = self.num_supervised // 2
            num_unsafe = self.num_supervised - num_safe
            safe_pick = self.safe_supervised_idx[torch.randint(0, len(self.safe_supervised_idx), (num_safe,))]
            unsafe_pick = self.unsafe_supervised_idx[torch.randint(0, len(self.unsafe_supervised_idx), (num_unsafe,))]
            sup_idx = torch.cat((safe_pick, unsafe_pick), dim=0)
            sup_idx = sup_idx[torch.randperm(sup_idx.shape[0])]
        else:
            sup_idx = torch.randint(0, self.supervised_states.shape[0], (self.num_supervised,))

        sup_states = self.supervised_states[sup_idx]
        sup_labels = self.supervised_labels[sup_idx]
        sup_times = self._sample_supervised_times(self.num_supervised)
        sup_coords = torch.cat((sup_times, sup_states), dim=1)
        sup_model_coords = self._coord_to_model_input(sup_coords)
        sup_values = torch.where(
            sup_labels > 0.5,
            torch.full_like(sup_labels, self.supervised_value_safe),
            torch.full_like(sup_labels, self.supervised_value_unsafe),
        )
        model_input['supervised_coords'] = sup_model_coords
        gt['supervised_values'] = sup_values
        gt['supervised_labels'] = sup_labels

    def _sample_supervised_times(self, num_samples):
        if self.pretrain:
            return torch.full((num_samples, 1), self.tMin)
        if self.counter_end > 0:
            time_window = (self.tMax - self.tMin) * (self.counter / self.counter_end)
        else:
            time_window = self.tMax - self.tMin
        return self.tMin + torch.zeros(num_samples, 1).uniform_(0, time_window)

    def _advance_counters(self):
        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.counter_end:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

    def _load_txt(self, path):
        rows = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append([float(x) for x in line.split(",")])
        return rows

    def _load_or_build_traj_lengths(self):
        if self.traj_state_cache is not None:
            return torch.tensor([traj_states.shape[0] for traj_states in self.traj_state_cache], dtype=torch.long)

        cache_path = os.path.join(self.data_root, "trajectory_lengths_cache.txt")
        if os.path.exists(cache_path):
            lengths = self._read_lengths_cache(cache_path)
            if lengths is not None:
                return lengths

        lengths = []
        for path in self.traj_files:
            line_count = 0
            with open(path, "r") as f:
                for line in f:
                    if line.strip():
                        line_count += 1
            lengths.append(max(line_count, self.required_traj_points))
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        self._write_lengths_cache(cache_path, lengths_tensor)
        return lengths_tensor

    def _read_lengths_cache(self, cache_path):
        lengths_by_name = {}
        try:
            with open(cache_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) != 2:
                        continue
                    lengths_by_name[parts[0]] = int(parts[1])
        except (OSError, ValueError):
            return None

        lengths = []
        for path in self.traj_files:
            name = os.path.basename(path)
            if name not in lengths_by_name:
                return None
            lengths.append(lengths_by_name[name])
        lengths = [max(length, self.required_traj_points) for length in lengths]
        return torch.tensor(lengths, dtype=torch.long)

    def _write_lengths_cache(self, cache_path, lengths_tensor):
        try:
            with open(cache_path, "w") as f:
                for path, length in zip(self.traj_files, lengths_tensor.tolist()):
                    f.write(f"{os.path.basename(path)},{int(length)}\n")
        except OSError:
            # Cache write failure should not break training.
            pass

    def _resolve_shuffled_indices_file(self):
        candidate_paths = []
        if self.shuffled_indices_file:
            candidate_paths.append(
                self.shuffled_indices_file if os.path.isabs(self.shuffled_indices_file)
                else os.path.join(self.data_root, self.shuffled_indices_file)
            )
        else:
            candidate_paths.extend([
                os.path.join(self.data_root, "train_test_splits", "shuffled_indices.txt"),
                os.path.join(self.data_root, "shuffled_indices.txt"),
            ])

        for path in candidate_paths:
            if os.path.exists(path):
                return path
        raise RuntimeError(
            "Could not find shuffled indices file. Checked: " + ", ".join(candidate_paths)
        )

    def _filter_traj_files_by_shuffled_indices(self, traj_files, shuffled_indices_path):
        name_to_path = {os.path.basename(path): path for path in traj_files}
        selected_paths = []
        seen = set()
        with open(shuffled_indices_path, "r") as f:
            for line in f:
                trajectory_name = line.strip()
                if (not trajectory_name) or (trajectory_name in seen):
                    continue
                path = name_to_path.get(trajectory_name)
                if path is not None:
                    selected_paths.append(path)
                    seen.add(trajectory_name)

        if not selected_paths:
            raise RuntimeError(
                f"No trajectories from {shuffled_indices_path} matched files in trajectories/."
            )
        return selected_paths

    def _load_trajectories_in_ram(self):
        traj_state_cache = []
        for path in self.traj_files:
            states = torch.tensor(self._load_txt(path), dtype=torch.float32)
            traj_state_cache.append(self._pad_states_to_tmax(states))
        return traj_state_cache


class Quadrotor2DDataset(CartPoleDataset):
    """CartPoleDataset with angle wrapping on dimension 2 (theta for 6D quadrotor)."""
    def __init__(self, **kwargs):
        kwargs.setdefault('angle_wrap_dims', [2])
        super().__init__(**kwargs)


class Quadrotor3DDataset(CartPoleDataset):
    """CartPoleDataset with no angle wrapping (quaternion representation)."""
    def __init__(self, **kwargs):
        kwargs.setdefault('angle_wrap_dims', [])
        super().__init__(**kwargs)
