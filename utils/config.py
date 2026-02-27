"""Centralized configuration loading/saving via OmegaConf.

Replaces the old ConfigArgParse approach with flat YAML configs + CLI overrides.

Usage:
    python run_experiment.py -c configs/cartpole_50k_70k.yaml mode=train lr=1e-4
"""

import inspect
import os
import pickle
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf, DictConfig

# ---------------------------------------------------------------------------
# Default values for every parameter (serves as the schema).
# These mirror the old add_argument() defaults from run_experiment.py.
# ---------------------------------------------------------------------------
DEFAULTS = {
    # core
    "mode": "train",
    "experiments_dir": "./runs",
    "experiment_name": None,  # required

    # wandb
    "use_wandb": False,
    "wandb_project": None,
    "wandb_entity": None,
    "wandb_group": None,
    "wandb_name": None,

    # experiment / dynamics class selection
    "experiment_class": "DeepReach",
    "dynamics_class": None,  # required
    "seed": 0,

    # device
    "device": "cuda:0",

    # data source
    "numpoints": 65000,
    "pretrain": False,
    "pretrain_iters": 2000,
    "tMin": 0.0,
    "tMax": 1.0,
    "counter_start": 0,
    "counter_end": -1,
    "num_src_samples": 1000,
    "num_target_samples": 0,
    "data_root": None,
    "num_supervised": 0,
    "supervised_value_safe": -1.0,
    "supervised_value_unsafe": 1.0,
    "supervised_weight": 0.0,
    "supervised_labels_file": None,
    "supervised_balanced_sampling": False,
    "supervised_safe_weight": 1.0,
    "supervised_unsafe_weight": 1.0,
    "trajectory_uniform_sampling": False,
    "max_trajectory_files": 0,
    "use_shuffled_indices_only": False,
    "shuffled_indices_file": None,
    "load_trajectories_in_ram": False,
    "training_objective": "hj_pde",

    # model
    "model": "sine",
    "model_mode": "mlp",
    "num_hl": 3,
    "num_nl": 512,
    "deepreach_model": "exact",

    # training
    "epochs_til_ckpt": 1000,
    "steps_til_summary": 100,
    "lr": 2e-5,
    "num_epochs": 100000,
    "clip_grad": 0.0,
    "use_lbfgs": False,
    "adj_rel_grads": True,
    "dirichlet_loss_divisor": 1.0,

    # early stopping
    "val_fraction": 0.1,
    "early_stopping_patience": 50,

    # post-training eval
    "eval_batch_size": 10000,

    # src samples decay
    "num_src_samples_decay_epochs": 0,  # linearly decay num_src_samples to 0 over this many epochs (0 = no decay)

    # CSL
    "use_CSL": False,
    "CSL_lr": 2e-5,
    "CSL_dt": 0.0025,
    "epochs_til_CSL": 10000,
    "num_CSL_samples": 1000000,
    "CSL_loss_frac_cutoff": 0.1,
    "max_CSL_epochs": 100,
    "CSL_loss_weight": 1.0,
    "CSL_batch_size": 1000,

    # validation
    "val_x_resolution": 200,
    "val_y_resolution": 200,
    "val_z_resolution": 5,
    "val_time_resolution": 3,

    # loss
    "minWith": "target",

    # test mode
    "dt": 0.0025,
    "checkpoint_toload": None,
    "num_scenarios": 100000,
    "num_violations": 1000,
    "control_type": "value",
    "data_step": "run_basic_recovery",
}

# Keys that are runtime-only and should not be persisted with the experiment.
RUNTIME_KEYS = {"device", "use_wandb", "wandb_project", "wandb_entity", "wandb_group", "wandb_name"}


def _inject_dynamic_defaults(cfg: DictConfig) -> DictConfig:
    """Fill in missing keys whose defaults come from dynamics/experiment class signatures.

    Gracefully skips if the modules can't be imported (e.g. missing wandb in a
    lightweight environment).
    """
    # Dynamics class constructor params
    dynamics_class_name = cfg.get("dynamics_class")
    if dynamics_class_name:
        try:
            from dynamics import dynamics as dynamics_mod
            dyn_cls = getattr(dynamics_mod, dynamics_class_name, None)
        except ImportError:
            dyn_cls = None
        if dyn_cls is not None:
            for name, param in inspect.signature(dyn_cls).parameters.items():
                if name == "self":
                    continue
                if name not in cfg or cfg[name] is None:
                    if param.default is not inspect.Parameter.empty:
                        cfg[name] = param.default

    # Experiment class init_special params
    experiment_class_name = cfg.get("experiment_class", "DeepReach")
    if experiment_class_name:
        try:
            from experiments import experiments as experiments_mod
            exp_cls = getattr(experiments_mod, experiment_class_name, None)
        except ImportError:
            exp_cls = None
        if exp_cls is not None and hasattr(exp_cls, "init_special"):
            for name, param in inspect.signature(exp_cls.init_special).parameters.items():
                if name == "self":
                    continue
                if name not in cfg or cfg[name] is None:
                    if param.default is not inspect.Parameter.empty:
                        cfg[name] = param.default

    return cfg


def load_config(argv=None) -> DictConfig:
    """Parse config from YAML file + CLI overrides.

    Supports:
        python run_experiment.py -c configs/foo.yaml key=value key2=value2
    """
    if argv is None:
        argv = sys.argv[1:]

    # Separate -c/--config flag from OmegaConf key=value overrides.
    # Accepts both new-style (key=value) and old-style (--key value / --flag).
    yaml_path = None
    overrides = []
    i = 0
    while i < len(argv):
        if argv[i] in ("-c", "--config", "--config_filepath"):
            if i + 1 < len(argv):
                yaml_path = argv[i + 1]
                i += 2
                continue
            else:
                raise ValueError(f"{argv[i]} requires a path argument")
        elif argv[i] == "--help":
            _print_help()
            sys.exit(0)
        elif argv[i].startswith("--"):
            # Old-style dashed arg: --key value or --flag (bare boolean)
            key = argv[i].lstrip("-")
            if i + 1 < len(argv) and "=" not in argv[i + 1] and not argv[i + 1].startswith("-"):
                overrides.append(f"{key}={argv[i + 1]}")
                i += 2
                continue
            else:
                # Bare flag like --pretrain → pretrain=true
                overrides.append(f"{key}=true")
        else:
            # New-style key=value
            overrides.append(argv[i])
        i += 1

    # Build config layers: defaults < yaml < cli overrides
    defaults_cfg = OmegaConf.create(DEFAULTS)

    if yaml_path:
        yaml_cfg = OmegaConf.load(yaml_path)
    else:
        yaml_cfg = OmegaConf.create({})

    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
    else:
        cli_cfg = OmegaConf.create({})

    cfg = OmegaConf.merge(defaults_cfg, yaml_cfg, cli_cfg)

    # Inject dynamic defaults from class signatures
    cfg = _inject_dynamic_defaults(cfg)

    # Fix counter_end sentinel
    if cfg.get("counter_end") == -1:
        cfg.counter_end = cfg.num_epochs

    return cfg


def save_config(cfg: DictConfig, experiment_dir: str) -> None:
    """Save config as YAML + pickle (backward compat) in experiment_dir."""
    # Strip runtime keys for the persisted copy
    persist = {k: v for k, v in OmegaConf.to_container(cfg, resolve=True).items()
               if k not in RUNTIME_KEYS}
    persist_cfg = OmegaConf.create(persist)
    OmegaConf.save(persist_cfg, os.path.join(experiment_dir, "config.yaml"))

    # Also write pickle for backward compat with eval scripts / old tooling
    ns = SimpleNamespace(**OmegaConf.to_container(cfg, resolve=True))
    with open(os.path.join(experiment_dir, "orig_opt.pickle"), "wb") as f:
        pickle.dump(ns, f)


def load_experiment_config(experiment_dir: str) -> DictConfig:
    """Load saved experiment config, trying config.yaml first, pickle fallback."""
    yaml_path = os.path.join(experiment_dir, "config.yaml")
    pickle_path = os.path.join(experiment_dir, "orig_opt.pickle")

    if os.path.exists(yaml_path):
        saved = OmegaConf.load(yaml_path)
    elif os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            obj = pickle.load(f)
        saved = OmegaConf.create(vars(obj) if hasattr(obj, "__dict__") else obj)
    else:
        raise FileNotFoundError(
            f"No config.yaml or orig_opt.pickle found in {experiment_dir}"
        )

    # Merge with defaults so all keys exist
    defaults_cfg = OmegaConf.create(DEFAULTS)
    cfg = OmegaConf.merge(defaults_cfg, saved)
    cfg = _inject_dynamic_defaults(cfg)

    # Fix counter_end sentinel (old pickles may have -1)
    if cfg.get("counter_end") == -1:
        cfg.counter_end = cfg.num_epochs

    return cfg


def _print_help():
    """Print available config keys and their defaults."""
    print("DeepReach experiment runner")
    print("===========================")
    print()
    print("Usage: python run_experiment.py -c CONFIG.yaml [key=value ...]")
    print()
    print("All configuration keys and their defaults:")
    print()
    for key, default in sorted(DEFAULTS.items()):
        print(f"  {key:40s} = {default}")
    print()
    print("Dynamics/experiment class parameters are discovered automatically.")
    print("Override any key with key=value on the command line.")
