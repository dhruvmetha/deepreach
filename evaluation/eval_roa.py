#!/usr/bin/env python
import argparse
import os
import sys
import json
import numpy as np
import torch

# Make repo modules importable
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from dynamics import dynamics as dynamics_mod
from utils import modules
from utils.config import load_experiment_config


def load_test_set(path, state_dim):
    # Rows: state (state_dim cols), optionally next_state (state_dim cols), label (1 col)
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < state_dim + 1:
        raise ValueError(f"Expected >={state_dim + 1} columns (state_dim={state_dim} + label), got {data.shape[1]}")
    states = data[:, 0:state_dim]
    labels = data[:, -1]
    return states, labels


def compute_metrics(y_true, y_pred):
    # y_true: binary labels (1 = safe, 0 = unsafe)
    # y_pred: 1 = safe, 0 = unsafe, -1 = separatrix (no-decision region)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    separatrix_mask = y_pred == -1
    eval_mask = ~separatrix_mask
    y_true_eval = y_true[eval_mask]
    y_pred_eval = y_pred[eval_mask]

    tp = int(np.sum((y_true_eval == 1) & (y_pred_eval == 1)))
    tn = int(np.sum((y_true_eval == 0) & (y_pred_eval == 0)))
    fp = int(np.sum((y_true_eval == 0) & (y_pred_eval == 1)))
    fn = int(np.sum((y_true_eval == 1) & (y_pred_eval == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    total_samples = int(y_true.shape[0])
    evaluated_samples = int(np.sum(eval_mask))
    separatrix_samples = int(np.sum(separatrix_mask))
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": acc,
        "evaluated_samples": evaluated_samples,
        "separatrix_samples": separatrix_samples,
        "coverage": (evaluated_samples / total_samples) if total_samples > 0 else 0.0,
        "separatrix_rate": (separatrix_samples / total_samples) if total_samples > 0 else 0.0,
    }


def compute_balanced_accuracy(metrics):
    return 0.5 * (metrics["recall"] + metrics["specificity"])


def predict_labels(values_np, decision_threshold, separatrix_margin):
    if separatrix_margin <= 0.0:
        return (values_np <= decision_threshold).astype(int)

    lower_threshold = decision_threshold - separatrix_margin
    upper_threshold = decision_threshold + separatrix_margin
    y_pred = np.full(values_np.shape, -1, dtype=int)
    y_pred[values_np <= lower_threshold] = 1
    y_pred[values_np >= upper_threshold] = 0
    return y_pred


def compute_values(model, dynamics, states_np, t_eval, device):
    t_col = np.full((states_np.shape[0], 1), t_eval, dtype=np.float32)
    coords = np.concatenate([t_col, states_np.astype(np.float32)], axis=1)
    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)

    with torch.no_grad():
        model_out = model({"coords": dynamics.coord_to_input(coords_t)})
        values = dynamics.io_to_value(model_out["model_in"], model_out["model_out"].squeeze(dim=-1))
    return values.detach().cpu().numpy()


def resolve_timestamp_dt(timestamp_dt_arg, opt):
    if timestamp_dt_arg is not None:
        if timestamp_dt_arg <= 0.0:
            raise ValueError("--timestamp_dt must be > 0")
        return float(timestamp_dt_arg), "cli"

    opt_dt = getattr(opt, "dt", None)
    if opt_dt is not None:
        try:
            opt_dt = float(opt_dt)
        except (TypeError, ValueError):
            opt_dt = None
        if opt_dt is not None and opt_dt > 0.0:
            return opt_dt, "orig_opt.dt"

    data_root = getattr(opt, "data_root", None)
    if data_root:
        desc_path = os.path.join(data_root, "dataset_description.json")
        if os.path.exists(desc_path):
            with open(desc_path, "r") as f:
                desc = json.load(f)
            sim_params = desc.get("generation_parameters", {}).get("simulation_parameters", {})
            for key in ("save_freq", "ctrl_timestep", "physics_timestep"):
                value = sim_params.get(key)
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if value > 0.0:
                    return value, f"dataset_description:{key}"

    raise ValueError(
        "Could not infer timestamp dt. Provide --timestamp_dt explicitly "
        "(e.g. 0.01 for CartPole save_freq)."
    )


def resolve_eval_time(args, opt):
    if args.timestamp_index is None:
        if args.t_eval is not None:
            return float(args.t_eval), None
        # Default to tMax from training config
        t_max = opt.get("tMax")
        if t_max is None:
            raise ValueError("--t_eval not provided and tMax not found in training config")
        print(f"[eval_roa] Using t_eval={float(t_max)} from training config tMax")
        return float(t_max), None

    if args.timestamp_index < 0:
        raise ValueError("--timestamp_index must be >= 0")

    timestamp_dt, dt_source = resolve_timestamp_dt(args.timestamp_dt, opt)
    t_min = float(getattr(opt, "tMin", 0.0))
    t_eval = t_min + (float(args.timestamp_index) * timestamp_dt)
    timestamp_info = {
        "timestamp_index": int(args.timestamp_index),
        "timestamp_dt": timestamp_dt,
        "timestamp_dt_source": dt_source,
        "timestamp_t_min": t_min,
    }
    return t_eval, timestamp_info


def optimize_threshold(values_np, y_true, metric_name, separatrix_margin=0.0, threshold_min=None, threshold_max=None, threshold_steps=1001):
    if threshold_steps < 2:
        raise ValueError("--threshold_steps must be >= 2 for auto threshold search")

    if threshold_min is None:
        threshold_min = float(values_np.min())
    if threshold_max is None:
        threshold_max = float(values_np.max())
    if threshold_max <= threshold_min:
        threshold_max = threshold_min + 1e-6

    thresholds = np.linspace(threshold_min, threshold_max, threshold_steps, dtype=np.float64)
    best_threshold = float(thresholds[0])
    best_metrics = None
    best_score = -float("inf")

    for threshold in thresholds:
        y_pred = predict_labels(values_np, threshold, separatrix_margin)
        metrics = compute_metrics(y_true, y_pred)
        if metric_name == "f1":
            score = metrics["f1"]
        elif metric_name == "accuracy":
            score = metrics["accuracy"]
        elif metric_name == "balanced_accuracy":
            score = compute_balanced_accuracy(metrics)
        elif metric_name == "specificity":
            score = metrics["specificity"]
        else:
            raise ValueError(f"Unknown metric {metric_name}")

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_score, best_metrics


def find_label_file(data_root, candidates):
    """Search data_root for the first existing label file from candidates."""
    if not data_root:
        return None
    for name in candidates:
        path = os.path.join(data_root, name)
        if os.path.exists(path):
            return path
    return None


def main():
    p = argparse.ArgumentParser(description="Evaluate ROA on test set and write summary to experiment folder.")
    p.add_argument("--experiment_dir", required=True, help="Path to experiment folder (runs/<experiment_name>)")
    p.add_argument("--checkpoint", default="model_final.pth", help="Checkpoint filename under training/checkpoints")
    p.add_argument("--test_set", default=None, help="Path to test_set.txt (if omitted, searches data_root from training config)")
    p.add_argument("--t_eval", type=float, default=None, help="Evaluation time in seconds (default: tMax from training config)")
    p.add_argument("--timestamp_index", type=int, default=None, help="Optional discrete timestamp index to evaluate (e.g. 613). If set, t_eval is computed as tMin + timestamp_index * dt.")
    p.add_argument("--timestamp_dt", type=float, default=None, help="Optional dt used with --timestamp_index. If omitted, inferred from orig_opt.dt or dataset_description.json.")
    p.add_argument("--label_safe", type=float, default=1.0, help="Label value indicating safe class (default 1)")
    p.add_argument("--decision_threshold", type=float, default=0.0, help="Predict safe if V <= threshold")
    p.add_argument("--separatrix_margin", type=float, default=0.0, help="Half-width of no-decision band around threshold. Safe if V <= threshold-margin, unsafe if V >= threshold+margin, else separatrix.")
    p.add_argument("--auto_threshold", action="store_true", help="Tune threshold on --cal_set before evaluating --test_set.")
    p.add_argument("--cal_set", default=None, help="Path to calibration set (same format as test_set). Required if --auto_threshold.")
    p.add_argument("--optimize_metric", default="f1", choices=["f1", "accuracy", "balanced_accuracy", "specificity"], help="Metric to optimize during threshold search.")
    p.add_argument("--threshold_min", type=float, default=None, help="Optional threshold search lower bound.")
    p.add_argument("--threshold_max", type=float, default=None, help="Optional threshold search upper bound.")
    p.add_argument("--threshold_steps", type=int, default=1001, help="Number of threshold candidates for auto-threshold search.")
    p.add_argument("--device", default="cpu", help="Device for evaluation: cpu or cuda:0")
    p.add_argument("--output_name", default="roa_eval.txt", help="Output filename in experiment_dir")
    args = p.parse_args()
    if args.separatrix_margin < 0.0:
        raise ValueError("--separatrix_margin must be >= 0")

    # Load original options (config.yaml with pickle fallback)
    opt = load_experiment_config(args.experiment_dir)
    data_root = opt.get("data_root")

    # Resolve test_set from data_root if not provided
    if args.test_set is None:
        args.test_set = find_label_file(data_root, ["test_set.txt", "cal_set.txt", "roa_labels.txt"])
        if args.test_set is None:
            raise ValueError(
                "--test_set not provided and no test_set.txt/cal_set.txt/roa_labels.txt found in data_root"
            )
        print(f"[eval_roa] Using test_set: {args.test_set}")

    # Resolve cal_set from data_root if auto_threshold but no cal_set provided
    if args.auto_threshold and args.cal_set is None:
        args.cal_set = find_label_file(data_root, ["cal_set.txt", "test_set.txt", "roa_labels.txt"])
        if args.cal_set is not None:
            print(f"[eval_roa] Using cal_set: {args.cal_set}")

    eval_time, timestamp_info = resolve_eval_time(args, opt)
    if "tMax" in opt and eval_time > float(opt.tMax):
        print(
            f"Warning: computed t_eval={eval_time} exceeds training tMax={float(opt.tMax)}; "
            "this uses time extrapolation.",
            file=sys.stderr,
        )

    # Build dynamics
    import inspect
    dynamics_class = getattr(dynamics_mod, opt.dynamics_class)
    dynamics = dynamics_class(**{argname: opt[argname] for argname in inspect.signature(dynamics_class).parameters.keys() if argname != 'self' and argname in opt})
    dynamics.deepreach_model = opt.deepreach_model

    # Build model
    model = modules.SingleBVPNet(in_features=dynamics.input_dim, out_features=1, type=opt.model, mode=opt.model_mode,
                                 final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl,
                                 omega_0=getattr(opt, 'omega_0', 30.0))
    model.to(args.device)
    model.eval()

    # Load checkpoint
    ckpt_path = os.path.join(args.experiment_dir, "training", "checkpoints", args.checkpoint)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=args.device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)

    selected_threshold = args.decision_threshold
    calibration_metrics = None
    calibration_score = None

    if args.auto_threshold:
        if args.cal_set is None:
            raise ValueError("--cal_set is required when --auto_threshold is used")
        cal_states_np, cal_labels_np = load_test_set(args.cal_set, dynamics.state_dim)
        cal_true = np.where(cal_labels_np == args.label_safe, 1, 0).astype(int)
        cal_values_np = compute_values(model, dynamics, cal_states_np, eval_time, args.device)
        selected_threshold, calibration_score, calibration_metrics = optimize_threshold(
            values_np=cal_values_np,
            y_true=cal_true,
            metric_name=args.optimize_metric,
            separatrix_margin=args.separatrix_margin,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_steps=args.threshold_steps,
        )

    # Load test set and evaluate with selected threshold
    states_np, labels_np = load_test_set(args.test_set, dynamics.state_dim)
    y_true = np.where(labels_np == args.label_safe, 1, 0).astype(int)
    values_np = compute_values(model, dynamics, states_np, eval_time, args.device)
    y_pred = predict_labels(values_np, selected_threshold, args.separatrix_margin)

    metrics = compute_metrics(y_true, y_pred)
    balanced_acc = compute_balanced_accuracy(metrics)

    # Write report
    out_path = os.path.join(args.experiment_dir, args.output_name)
    with open(out_path, "w") as f:
        f.write("ROA evaluation summary\n")
        f.write("=======================\n")
        f.write(f"experiment_dir = {args.experiment_dir}\n")
        f.write(f"checkpoint = {args.checkpoint}\n")
        f.write(f"test_set = {args.test_set}\n")
        f.write(f"t_eval = {eval_time}\n")
        if timestamp_info is not None:
            f.write(f"timestamp_index = {timestamp_info['timestamp_index']}\n")
            f.write(f"timestamp_dt = {timestamp_info['timestamp_dt']}\n")
            f.write(f"timestamp_dt_source = {timestamp_info['timestamp_dt_source']}\n")
            f.write(f"timestamp_t_min = {timestamp_info['timestamp_t_min']}\n")
        f.write(f"decision_threshold = {selected_threshold}\n")
        f.write(f"separatrix_margin = {args.separatrix_margin}\n")
        f.write(f"separatrix_lower_threshold = {selected_threshold - args.separatrix_margin}\n")
        f.write(f"separatrix_upper_threshold = {selected_threshold + args.separatrix_margin}\n")
        f.write(f"auto_threshold = {args.auto_threshold}\n")
        if args.auto_threshold:
            f.write(f"cal_set = {args.cal_set}\n")
            f.write(f"optimize_metric = {args.optimize_metric}\n")
            f.write(f"threshold_search = [{args.threshold_min}, {args.threshold_max}] steps={args.threshold_steps}\n")
            f.write(f"calibration_score = {calibration_score:.6f}\n")
            if calibration_metrics is not None:
                f.write(f"calibration_f1 = {calibration_metrics['f1']:.6f}\n")
                f.write(f"calibration_accuracy = {calibration_metrics['accuracy']:.6f}\n")
                f.write(f"calibration_specificity = {calibration_metrics['specificity']:.6f}\n")
                f.write(f"calibration_balanced_accuracy = {compute_balanced_accuracy(calibration_metrics):.6f}\n")
                f.write(f"calibration_coverage = {calibration_metrics['coverage']:.6f}\n")
                f.write(f"calibration_separatrix_rate = {calibration_metrics['separatrix_rate']:.6f}\n")
        f.write(f"label_safe = {args.label_safe}\n")
        f.write(f"num_samples = {len(y_true)}\n")
        f.write(f"evaluated_samples = {metrics['evaluated_samples']}\n")
        f.write(f"separatrix_samples = {metrics['separatrix_samples']}\n")
        f.write(f"coverage = {metrics['coverage']:.6f}\n")
        f.write(f"separatrix_rate = {metrics['separatrix_rate']:.6f}\n")
        f.write("\n")
        f.write(f"TP = {metrics['tp']}\n")
        f.write(f"TN = {metrics['tn']}\n")
        f.write(f"FP = {metrics['fp']}\n")
        f.write(f"FN = {metrics['fn']}\n")
        f.write(f"precision = {metrics['precision']:.6f}\n")
        f.write(f"recall = {metrics['recall']:.6f}\n")
        f.write(f"specificity = {metrics['specificity']:.6f}\n")
        f.write(f"f1 = {metrics['f1']:.6f}\n")
        f.write(f"accuracy = {metrics['accuracy']:.6f}\n")
        f.write(f"balanced_accuracy = {balanced_acc:.6f}\n")

    print(f"Wrote evaluation report to {out_path}")


if __name__ == "__main__":
    main()
