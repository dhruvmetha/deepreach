# DeepReach: A Deep Learning Approach to High-Dimensional Reachability
### [Project Page](http://people.eecs.berkeley.edu/~somil/index.html) | [Paper](https://arxiv.org/pdf/2011.02082.pdf)<br>

Repository Maintainers<br>
[Albert Lin](https://www.linkedin.com/in/albertkuilin/),
[Zeyuan Feng](https://thezeyuanfeng.github.io/),
[Javier Borquez](https://javierborquez.github.io/),
[Somil Bansal](http://people.eecs.berkeley.edu/~somil/index.html)<br>
University of Southern California

Original Authors<br>
[Somil Bansal](http://people.eecs.berkeley.edu/~somil/index.html),
Claire Tomlin<br>
University of California, Berkeley

(Still to come...) The Safe and Intelligent Autonomy (SIA) Lab at the University of Southern California
is still working on an easy-to-use DeepReach Python package which will follow much of the same organizational principles as
the [hj_reachability package in JAX](https://github.com/StanfordASL/hj_reachability) from the Autonomous Systems Lab at Stanford.
The future version will include the newest tips and tricks of DeepReach developed by SIA.

(In the meantime...) This branch provides a moderately refactored version of DeepReach to facilitate easier outside research on DeepReach.

## Full Technical Notebook
For the complete implementation walkthrough (data path, model, losses, training flow, full dynamics math derivations, configs, and diagnostics), read:

- `MEGA_NOTEBOOK.md`

## High-Level Structure
The code is organized as follows:
* `dynamics/dynamics.py` defines the dynamics of the system.
* `experiments/experiments.py` contains generic training routines.
* `utils/modules.py` contains neural network layers and modules.
* `utils/dataio.py` loads training and testing data.
* `utils/diff_operators.py` contains implementations of differential operators.
* `utils/losses.py` contains loss functions for the different reachability cases.
* `run_experiment.py` starts a standard DeepReach experiment run.

## External Tutorial
Follow along these [tutorial slides](https://docs.google.com/presentation/d/19zxhvZAHgVYDCRpCej2svCw21iRvcxQ0/edit?usp=drive_link&ouid=113852163991034806329&rtpof=true&sd=true) to get started, or continue reading below.

## Environment Setup
Create and activate a virtual python environment (env) to manage dependencies:
```
python -m venv env
env\Scripts\activate
```
Install DeepReach dependencies:
```
pip install -r requirements.txt
```
Install the appropriate PyTorch package for your system. For example, for a Windows system with CUDA 12.1:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Running a DeepReach Experiment
`run_experiment.py` implements a standard DeepReach experiment. For example, to learn the value function for the avoid Dubins3D system with parameters `goalR=0.25`, `velocity=0.6`, `omega_max=1.1`, run:
```
python run_experiment.py --mode train --experiment_class DeepReach --dynamics_class Dubins3D --experiment_name dubins3d_tutorial_run --minWith target --goalR 0.25 --velocity 0.6 --omega_max 1.1 --angle_alpha_factor 1.2 --set_mode avoid
```
Note that the script provides many common training arguments, like `num_epochs` and the option to `pretrain`. For detailed training internals and option behavior, see `MEGA_NOTEBOOK.md`. `use_CSL` is an experimental training option (similar in spirit to actor-critic methods) being developed by SIA for improved value function learning.

## CartPole Hybrid Training (HJ PDE + Supervised Labels)
This repo now includes an extended CartPole path:

* `dynamics_class=CartPole` uses a CartPole dynamics model and Hamiltonian in `dynamics/dynamics.py`.
* Training data is loaded from files via `CartPoleDataset` in `utils/dataio.py`.
* If a supervised label file is present and `num_supervised > 0`, an additional supervised value MSE term is added during training (`roa_labels.txt` by default, or override with `--supervised_labels_file`).
* To reduce class-imbalance collapse, you can enable `--supervised_balanced_sampling` and/or set class weights with `--supervised_safe_weight` and `--supervised_unsafe_weight`.
* To make PDE state sampling approximately uniform over stored trajectory points, enable `--trajectory_uniform_sampling`.
* To run on a smaller trajectory subset, set `--max_trajectory_files` (e.g., `100`).

When `--dynamics_class CartPole` is selected, the dataset path is chosen automatically in `run_experiment.py`.

Expected dataset root (`--data_root`):

* `trajectories/sequence_*.txt`
* optional `roa_labels.txt` (or `cal_set.txt` if you use the 9-column format and set `--supervised_labels_file cal_set.txt`)
* `dataset_description.json` (used to auto-load `gravity`, `cart_mass`, `pole_mass`, `pole_length` if not provided)

Small run example:
```
python run_experiment.py --mode train \
  --experiment_class DeepReach \
  --dynamics_class CartPole \
  --experiment_name cartpole_small \
  --minWith target \
  --u_max 2000 \
  --x_bound 6 --xdot_bound 5 --thetadot_bound 5 \
  --set_mode avoid \
  --data_root /common/users/ss5772/DeepReach/deepreach/cartpole_pybullet \
  --supervised_labels_file cal_set.txt \
  --num_supervised 256 \
  --supervised_weight 1.0 \
  --supervised_balanced_sampling \
  --supervised_safe_weight 2.0 \
  --supervised_unsafe_weight 1.0 \
  --trajectory_uniform_sampling \
  --max_trajectory_files 100 \
  --tMin 0.0 --tMax 2.0 \
  --numpoints 2000 \
  --num_epochs 200 \
  --pretrain --pretrain_iters 200 \
  --lr 1e-4 \
  --num_hl 2 --num_nl 128 \
  --model sine
```

Temporal consistency mode (CartPole-only, observed-flow PDE residual from trajectories):
```bash
python run_experiment.py --mode train \
  --experiment_class DeepReach \
  --dynamics_class CartPole \
  --experiment_name cartpole_temporal \
  --minWith target \
  --u_max 2000 \
  --x_bound 6 --xdot_bound 5 --thetadot_bound 5 \
  --set_mode avoid \
  --data_root /common/users/ss5772/DeepReach/deepreach/cartpole_pybullet \
  --training_objective temporal_consistency \
  --tc_loss_weight 1.0 \
  --tc_anchor_weight 0.1 \
  --tc_t0_mode weighted \
  --num_supervised 128 \
  --supervised_labels_file cal_set.txt \
  --supervised_weight 1.0 \
  --trajectory_uniform_sampling \
  --numpoints 2000 \
  --num_epochs 200 \
  --lr 1e-4 \
  --num_hl 2 --num_nl 128 \
  --model sine
```
Notes:
- `training_objective=hj_pde` remains the default and preserves previous behavior.
- `training_objective=temporal_consistency` currently supports `CartPoleDataset` only.
- Temporal loss is `|dV/dt + min(0, ∇V·ẋ_obs)|^2` with `ẋ_obs` from one-step trajectory differences.
- `--tc_t0_mode {weighted,fixed,off}` controls whether/how `t=tMin` boundary loss is applied.
- Temporal mode logs `tc_backup`, `tc_anchor`, and `tc_total`.

For implementation-level details of this integration, see `MEGA_NOTEBOOK.md`.

Evaluation tip for imbalanced labels:

1. calibrate threshold on `cal_set.txt`,
2. then evaluate once on `test_set.txt` with that threshold.
3. optionally set `--separatrix_margin M` to create a no-decision band `[threshold-M, threshold+M]` and track coverage.

```bash
python evaluation/eval_roa.py \
  --experiment_dir runs/cartpole_small \
  --checkpoint model_final.pth \
  --cal_set cartpole_pybullet/cal_set.txt \
  --test_set cartpole_pybullet/test_set.txt \
  --t_eval 2.0 \
  --auto_threshold \
  --optimize_metric f1 \
  --separatrix_margin 0.0 \
  --threshold_steps 1001
```

`evaluation/eval_roa.py` reports specificity in addition to precision/recall/F1/accuracy/balanced accuracy.
It also supports discrete timestamps via `--timestamp_index` (e.g. `613`), converted using `t_eval = tMin + timestamp_index * dt` (set `--timestamp_dt` or infer from dataset metadata).

## Monitoring a DeepReach Experiment
Results for the Dubins3D system specified in the above section can be found in this [online WandB project](https://wandb.ai/aklin/DeepReachTutorial).
We highly recommend users use the `--use_wandb` flag to log training progress to the free cloud-based Weights & Biases AI Developer Platform, where it can be easily viewed and shared.

Throughout training, the training loss curves, value function plots, and model checkpoints are saved locally to `runs/experiment_name/training/summaries` and `runs/experiment_name/training/checkpoints` (and to WandB, if specified).

## Defining a Custom System
Systems are defined in `dynamics/dynamics.py` and inherit from the abstract `Dynamics` class. At a minimum, users must define:
* `__init(self, ...)__`, which must call `super().__init__(loss_type, set_mode, state_dim, ...)`
* `state_test_range(self)`, which specifies the state space that will be visualized in training plots
* `dsdt(self, state, control, disturbance)`, which implements the forward dynamics
* `boundary_fn(self, state)`,  which implements the boundary function that implicitly represents the target set
* `hamiltonian(self, state, dvds)`, which implements the system's hamiltonian
* `plot_config(self)`, which specifies the state slices and axes visualized in training plots

## Citation
If you find our work useful in your research, please cite:
```
@software{deepreach2024,
  author = {Lin, Albert and Feng, Zeyuan and Borquez, Javier and Bansal, Somil},
  title = {{DeepReach Repository}},
  url = {https://github.com/smlbansal/deepreach},
  year = {2024}
}
```

```
@inproceedings{bansal2020deepreach,
    author = {Bansal, Somil
              and Tomlin, Claire},
    title = {{DeepReach}: A Deep Learning Approach to High-Dimensional Reachability},
    booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
    year={2021}
}
```

## Contact
If you have any questions, please feel free to email the authors.
