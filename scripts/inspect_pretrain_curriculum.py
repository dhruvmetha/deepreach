#!/usr/bin/env python3
"""
Inspect how DeepReach generates training batches and computes V and loss terms for:
  1) pretraining (all t == tMin, Dirichlet-only)
  2) curriculum training (time window grows from [tMin, tMax])
  3) temporal consistency (observed-flow PDE residual mode)

This script mirrors the critical parts of the training loop in:
  - experiments/experiments.py
  - utils/dataio.py
  - utils/losses.py
  - dynamics/dynamics.py

Example (CartPole):
  python scripts/inspect_pretrain_curriculum.py \
    --data_root /common/users/ss5772/DeepReach/deepreach/cartpole_pybullet \
    --numpoints 2000 \
    --num_src_samples 256 \
    --tMin 0.0 --tMax 10.0 \
    --minWith target \
    --deepreach_model exact \
    --num_supervised 128 \
    --supervised_labels_file cal_set.txt \
    --trajectory_uniform_sampling \
    --max_trajectory_files 100 \
    --device cpu
"""

import argparse
import os
import sys
import warnings

import torch

# Make deepreach/ importable no matter where this is launched from.
DEEPREACH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DEEPREACH_DIR not in sys.path:
    sys.path.insert(0, DEEPREACH_DIR)

from dynamics import dynamics as dynamics_mod
from utils import dataio, losses, modules


def _summarize(name: str, x: torch.Tensor) -> str:
    x = x.detach()
    if x.numel() == 0:
        return f"{name}: <empty>"
    return (
        f"{name}: shape={tuple(x.shape)} "
        f"min={float(x.min()):.4g} max={float(x.max()):.4g} mean={float(x.mean()):.4g}"
    )


def run_one_batch(
    *,
    stage: str,
    dataset,
    dynamics,
    model,
    loss_fn,
    device: str,
    training_objective: str,
    minWith: str,
    supervised_weight: float,
    supervised_safe_weight: float,
    supervised_unsafe_weight: float,
):
    model_input, gt = dataset[0]
    model_input = {k: v.to(device) for k, v in model_input.items()}
    gt = {k: v.to(device) for k, v in gt.items()}

    if training_objective == "temporal_consistency":
        curr_results = model({"coords": model_input["tc_model_coords_curr"]})
        values_curr = dynamics.io_to_value(
            curr_results["model_in"].detach(),
            curr_results["model_out"].squeeze(dim=-1),
        )
        dvs_curr = dynamics.io_to_dv(
            curr_results["model_in"],
            curr_results["model_out"].squeeze(dim=-1),
        )
        dvdt_curr = dvs_curr[..., 0]
        dvds_curr = dvs_curr[..., 1:]
        obs_flow = model_input["tc_obs_flow"]
        boundary_values = gt["tc_boundary_values"]
        flow_dot = torch.sum(dvds_curr * obs_flow, dim=-1)
        diff_constraint_hom = dvdt_curr - flow_dot
        flow_residual = torch.max(diff_constraint_hom, values_curr - boundary_values)

        loss_dict = loss_fn(
            values_curr,
            dvdt_curr,
            dvds_curr,
            obs_flow,
            boundary_values,
        )
    else:
        # --- PDE batch (same as experiments/experiments.py) ---
        model_results = model({"coords": model_input["model_coords"]})
        states = dynamics.input_to_coord(model_results["model_in"].detach())[..., 1:]
        values = dynamics.io_to_value(
            model_results["model_in"].detach(),
            model_results["model_out"].squeeze(dim=-1),
        )
        dvs = dynamics.io_to_dv(model_results["model_in"], model_results["model_out"].squeeze(dim=-1))
        dvdt = dvs[..., 0]
        dvds = dvs[..., 1:]
        boundary_values = gt["boundary_values"]
        dirichlet_masks = gt["dirichlet_masks"]

        # Hamiltonian / residual (for inspection; loss_fn re-computes pieces internally)
        ham = dynamics.hamiltonian(states, dvds)
        residual = dvdt - ham
        if minWith == "zero":
            ham = torch.clamp(ham, max=0.0)
            residual = dvdt - ham
        if minWith == "target":
            residual = torch.max(residual, values - boundary_values)

        if dynamics.loss_type == "brt_hjivi":
            loss_dict = loss_fn(
                states,
                values,
                dvdt,
                dvds,
                boundary_values,
                dirichlet_masks,
                model_results["model_out"],
            )
        else:
            raise RuntimeError(f"Expected brt_hjivi for this inspector, got {dynamics.loss_type}")

    # --- Optional supervised batch (same as experiments/experiments.py) ---
    supervised_loss = None
    sup_info = ""
    if supervised_weight > 0.0 and ("supervised_coords" in model_input) and ("supervised_values" in gt):
        sup_results = model({"coords": model_input["supervised_coords"]})
        sup_values = dynamics.io_to_value(sup_results["model_in"], sup_results["model_out"].squeeze(dim=-1))
        sup_targets = gt["supervised_values"]
        sup_errors = torch.pow(sup_values - sup_targets, 2)
        if "supervised_labels" in gt:
            sup_labels = gt["supervised_labels"]
            sample_weights = torch.where(
                sup_labels > 0.5,
                torch.full_like(sup_labels, supervised_safe_weight),
                torch.full_like(sup_labels, supervised_unsafe_weight),
            )
            supervised_loss = supervised_weight * torch.mean(sample_weights * sup_errors)
            sup_safe = int(torch.sum(sup_labels > 0.5).item())
            sup_info = f" supervised_labels safe={sup_safe} unsafe={int(sup_labels.numel()) - sup_safe}"
        else:
            supervised_loss = supervised_weight * torch.mean(sup_errors)

    # --- Print report ---
    print()
    print(f"=== {stage} ===")
    print(f"dataset.pretrain={dataset.pretrain} counter={dataset.counter}/{dataset.counter_end} tMin={dataset.tMin} tMax={dataset.tMax}")
    if training_objective == "temporal_consistency":
        t_curr = model_input["tc_model_coords_curr"][..., 0].detach().cpu()
        print(
            "time(curr): min=%0.4g max=%0.4g mean=%0.4g"
            % (
                float(t_curr.min()), float(t_curr.max()), float(t_curr.mean()),
            )
        )
        print(_summarize("horizon", gt["tc_horizon"]))
        print(_summarize("obs_flow", obs_flow))
        print(_summarize("v_curr", values_curr))
        print(_summarize("dvdt", dvdt_curr))
        print(_summarize("flow_dot", flow_dot))
        print(_summarize("flow_residual", flow_residual))
    else:
        t = model_input["model_coords"][..., 0].detach().cpu()
        num_dir = int(torch.sum(dirichlet_masks).item())
        print(f"time: min={float(t.min()):.4g} max={float(t.max()):.4g} mean={float(t.mean()):.4g} dirichlet_true={num_dir}/{t.numel()}")
        print(_summarize("states(real)", states))
        print(_summarize("boundary_values", boundary_values))
        print(_summarize("values(V)", values))
        print(_summarize("dvdt", dvdt))
        print(_summarize("dvds", dvds))
        print(_summarize("hamiltonian", ham))
        print(_summarize("residual(after minWith)", residual))
    for k, v in loss_dict.items():
        print(f"loss[{k}] = {float(v.detach().cpu()):.6g}")
    if supervised_loss is not None:
        print(f"supervised_loss = {float(supervised_loss.detach().cpu()):.6g}{sup_info}")


def main():
    p = argparse.ArgumentParser(description="Inspect DeepReach pretrain/curriculum batch generation and loss terms (CartPole-focused).")
    p.add_argument("--data_root", required=True, help="CartPole dataset root (contains trajectories/, dataset_description.json, labels).")
    p.add_argument("--device", default="cpu", help="cpu or cuda:0")
    p.add_argument("--seed", type=int, default=0)

    # Sampling / curriculum
    p.add_argument("--numpoints", type=int, default=2000)
    p.add_argument("--num_src_samples", type=int, default=256)
    p.add_argument("--tMin", type=float, default=0.0)
    p.add_argument("--tMax", type=float, default=10.0)
    p.add_argument("--counter_end", type=int, default=100000)

    # Loss/model knobs
    p.add_argument("--training_objective", choices=["hj_pde", "temporal_consistency"], default="hj_pde")
    p.add_argument("--minWith", choices=["none", "zero", "target"], default="target")
    p.add_argument("--dirichlet_loss_divisor", type=float, default=1.0)
    p.add_argument("--deepreach_model", choices=["exact", "diff", "vanilla"], default="exact")
    p.add_argument("--model", choices=["sine", "tanh", "sigmoid", "relu"], default="sine")
    p.add_argument("--model_mode", choices=["mlp", "rbf", "pinn"], default="mlp")
    p.add_argument("--num_hl", type=int, default=2)
    p.add_argument("--num_nl", type=int, default=128)
    p.add_argument("--tc_target_mode", choices=["one_step", "n_step"], default="one_step")
    p.add_argument("--tc_n_step", type=int, default=1)
    p.add_argument("--tc_detach_next", dest="tc_detach_next", action="store_true", help="Legacy compatibility flag; ignored in observed-flow temporal mode.")
    p.add_argument("--no_tc_detach_next", dest="tc_detach_next", action="store_false", help="Legacy compatibility flag; ignored in observed-flow temporal mode.")
    p.set_defaults(tc_detach_next=True)

    # CartPole dynamics (physics params auto-load from dataset_description.json)
    p.add_argument("--u_max", type=float, default=2000.0)
    p.add_argument("--x_bound", type=float, default=6.0)
    p.add_argument("--xdot_bound", type=float, default=5.0)
    p.add_argument("--thetadot_bound", type=float, default=5.0)
    p.add_argument("--set_mode", choices=["reach", "avoid"], default="avoid")

    # Optional supervised batch
    p.add_argument("--num_supervised", type=int, default=0)
    p.add_argument("--supervised_labels_file", default=None, help="e.g., roa_labels.txt or cal_set.txt (relative to data_root ok).")
    p.add_argument("--supervised_value_safe", type=float, default=-1.0)
    p.add_argument("--supervised_value_unsafe", type=float, default=1.0)
    p.add_argument("--supervised_weight", type=float, default=0.0)
    p.add_argument("--supervised_safe_weight", type=float, default=1.0)
    p.add_argument("--supervised_unsafe_weight", type=float, default=1.0)
    p.add_argument("--supervised_balanced_sampling", action="store_true", default=False)

    # Trajectory sampling knobs
    p.add_argument("--trajectory_uniform_sampling", action="store_true", default=False)
    p.add_argument("--max_trajectory_files", type=int, default=0)
    p.add_argument("--dt", type=float, default=0.01)

    args = p.parse_args()

    if args.num_src_samples > args.numpoints:
        raise ValueError("--num_src_samples must be <= --numpoints (otherwise everything is Dirichlet points).")

    torch.manual_seed(args.seed)

    if args.training_objective == "temporal_consistency":
        if args.tc_target_mode != "one_step":
            warnings.warn("--tc_target_mode is ignored in observed-flow temporal consistency mode.")
        if args.tc_n_step != 1:
            warnings.warn("--tc_n_step is ignored in observed-flow temporal consistency mode.")
        if not args.tc_detach_next:
            warnings.warn("--tc_detach_next is ignored in observed-flow temporal consistency mode.")

    # Build dynamics (physics params read from dataset_description.json using data_root)
    dynamics = dynamics_mod.CartPole(
        u_max=args.u_max,
        x_bound=args.x_bound,
        xdot_bound=args.xdot_bound,
        thetadot_bound=args.thetadot_bound,
        set_mode=args.set_mode,
        data_root=args.data_root,
    )
    dynamics.deepreach_model = args.deepreach_model

    # Build dataset
    # NOTE: For Quadrotor2D use dataio.Quadrotor2DDataset (angle_wrap_dims=[2]),
    #       for Quadrotor3D use dataio.Quadrotor3DDataset (angle_wrap_dims=[]).
    dataset = dataio.CartPoleDataset(
        dynamics=dynamics,
        numpoints=args.numpoints,
        pretrain=True,
        pretrain_iters=10**9,  # keep "pretrain mode" until we flip it manually
        tMin=args.tMin,
        tMax=args.tMax,
        counter_start=0,
        counter_end=args.counter_end,
        num_src_samples=args.num_src_samples,
        num_target_samples=0,
        data_root=args.data_root,
        dt=args.dt,
        num_supervised=args.num_supervised,
        supervised_value_safe=args.supervised_value_safe,
        supervised_value_unsafe=args.supervised_value_unsafe,
        supervised_labels_file=args.supervised_labels_file,
        supervised_balanced_sampling=args.supervised_balanced_sampling,
        trajectory_uniform_sampling=args.trajectory_uniform_sampling,
        max_trajectory_files=args.max_trajectory_files,
        training_objective=args.training_objective,
        tc_target_mode=args.tc_target_mode,
        tc_n_step=args.tc_n_step,
        tc_sample_terminal=False,
    )

    # Build model + loss
    model = modules.SingleBVPNet(
        in_features=dynamics.input_dim,
        out_features=1,
        type=args.model,
        mode=args.model_mode,
        hidden_features=args.num_nl,
        num_hidden_layers=args.num_hl,
    ).to(args.device)

    if args.training_objective == "temporal_consistency":
        loss_fn = losses.init_temporal_consistency_loss()
    else:
        loss_fn = losses.init_brt_hjivi_loss(dynamics, args.minWith, args.dirichlet_loss_divisor)

    # 1) Pretraining batch
    dataset.pretrain = True
    dataset.pretrain_counter = 0
    run_one_batch(
        stage="PRETRAIN (all t == tMin; Dirichlet-focused)",
        dataset=dataset,
        dynamics=dynamics,
        model=model,
        loss_fn=loss_fn,
        device=args.device,
        training_objective=args.training_objective,
        minWith=args.minWith,
        supervised_weight=args.supervised_weight,
        supervised_safe_weight=args.supervised_safe_weight,
        supervised_unsafe_weight=args.supervised_unsafe_weight,
    )

    # 2) Early curriculum (small time window)
    dataset.pretrain = False
    dataset.counter = max(1, int(0.05 * dataset.counter_end))
    run_one_batch(
        stage="CURRICULUM (early; small time window)",
        dataset=dataset,
        dynamics=dynamics,
        model=model,
        loss_fn=loss_fn,
        device=args.device,
        training_objective=args.training_objective,
        minWith=args.minWith,
        supervised_weight=args.supervised_weight,
        supervised_safe_weight=args.supervised_safe_weight,
        supervised_unsafe_weight=args.supervised_unsafe_weight,
    )

    # 3) Late curriculum (full time window)
    dataset.pretrain = False
    dataset.counter = dataset.counter_end
    run_one_batch(
        stage="CURRICULUM (late; full time window up to tMax)",
        dataset=dataset,
        dynamics=dynamics,
        model=model,
        loss_fn=loss_fn,
        device=args.device,
        training_objective=args.training_objective,
        minWith=args.minWith,
        supervised_weight=args.supervised_weight,
        supervised_safe_weight=args.supervised_safe_weight,
        supervised_unsafe_weight=args.supervised_unsafe_weight,
    )


if __name__ == "__main__":
    main()
