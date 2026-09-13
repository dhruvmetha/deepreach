#!/usr/bin/env python
"""Qualitative value-function heatmaps for trained DeepReach experiments.

Produces 2-D slice heatmaps of V(x, t_eval) — both continuous (diverging
colormap + zero-level contour) and binary ROA (safe / unsafe).  Supports
multiple experiments side-by-side and caching for fast re-plotting.

The model is evaluated on the *exact same* grid points from the ground-truth
viz CSVs (e.g. CartPole), so results are directly comparable with the
olympics-classifier and GT overlays.  Every slice must define gt_csv,
gt_ax0_col, and gt_ax1_col.

Outputs per (experiment, slice):
  - Bare single images: no axes, no title, just the heatmap (for paper figures)
  - Styled single images: with axes, tick labels, colorbar
  - Combined multi-panel figure (experiments x slices) with constrained_layout

Usage:
    # From a YAML config (recommended):
    python scripts/plot_qualitative_value.py -c configs/plot_qualitative.yaml

    # CLI only:
    python scripts/plot_qualitative_value.py \\
        --experiment_dirs runs/my_exp --device cpu --all_slices

    # Mix: YAML defaults + CLI overrides:
    python scripts/plot_qualitative_value.py -c configs/plot_qualitative.yaml \\
        --recompute --device cuda:0
"""

import argparse
import inspect
import os
import sys
import yaml

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# Path setup — repo root must be importable
# ---------------------------------------------------------------------------
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from dynamics import dynamics as dynamics_mod
from utils import modules
from utils.config import load_experiment_config
from evaluation.eval_roa import compute_values

# ── DeepReach calibrated thresholds (CartPole 1000 best) ─────────────────────
# V <= c_low  → success (+1)
# V >= c_high → failure (-1)
# else        → separatrix (0)
DEFAULT_C_LOW = -0.264
DEFAULT_C_HIGH = -0.194

# ---------------------------------------------------------------------------
# Per-system default slice definitions
# ---------------------------------------------------------------------------
# Each slice:
#   sweep:  (dim_i, dim_j) — indices into state vector to sweep
#   fixed:  {dim_idx: value} — remaining dims pinned at these values
#   name:   string identifier
#   xlabel, ylabel:         LaTeX axis labels
#   xticks, xticklabels:    custom tick positions and labels
#   yticks, yticklabels:    custom tick positions and labels
#   gt_csv: optional path to ground-truth CSV

# ── Ground truth CSV directory (same CSVs used by olympics-classifier) ───────
_GT_DIR = os.path.join(
    "/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole_pybullet"
)

SYSTEM_SLICES = {
    "CartPole": [
        dict(
            sweep=(1, 3), fixed={0: 0.0, 2: 0.0}, name="theta_thetadot",
            xlabel=r"$\theta$ (rad)",
            ylabel=r"$\dot{\theta}$ (rad/s)",
            xticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
            xticklabels=[r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"],
            yticks=[-2*np.pi, -np.pi, 0, np.pi, 2*np.pi],
            yticklabels=[r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"],
            gt_csv=os.path.join(_GT_DIR, "viz_theta_vs_thetadot.csv"),
            gt_ax0_col="theta",
            gt_ax1_col="theta_dot",
        ),
        dict(
            sweep=(0, 2), fixed={1: 0.0, 3: 0.0}, name="x_xdot",
            xlabel=r"$x$ (m)",
            ylabel=r"$\dot{x}$ (m/s)",
            xticks=[-6, -3, 0, 3, 6],
            xticklabels=["-6", "-3", "0", "3", "6"],
            yticks=[-6, -3, 0, 3, 6],
            yticklabels=["-6", "-3", "0", "3", "6"],
            gt_csv=os.path.join(_GT_DIR, "viz_x_vs_xdot.csv"),
            gt_ax0_col="x",
            gt_ax1_col="x_dot",
        ),
    ],
}


def auto_slices(dynamics):
    """Return default slice definitions for *dynamics*.

    Raises ValueError if the system has no slice definitions with gt_csv.
    """
    cls_name = type(dynamics).__name__
    if cls_name not in SYSTEM_SLICES:
        raise ValueError(
            f"No slice definitions for {cls_name}. "
            f"Supported systems: {list(SYSTEM_SLICES.keys())}"
        )
    return SYSTEM_SLICES[cls_name]


# ---------------------------------------------------------------------------
# Grid loading — from GT CSV
# ---------------------------------------------------------------------------

def load_slice_data(slice_def, state_dim):
    """Load grid + GT labels from the viz CSV defined in the slice.

    This is the same CSV used by plot_qualitative_cartpole.py — a dense grid
    with columns for all state dimensions plus a ``label`` column.

    Returns
    -------
    grid_np : ndarray, shape (N, state_dim) — full-dimensional states
    gt_labels : ndarray, shape (N,) — 1=safe, 0=unsafe
    axis0 : sorted unique values for sweep dim 0
    axis1 : sorted unique values for sweep dim 1
    n0, n1 : grid dimensions
    """
    import pandas as pd
    gt_csv = slice_def["gt_csv"]
    ax0_col = slice_def["gt_ax0_col"]
    ax1_col = slice_def["gt_ax1_col"]

    df = pd.read_csv(gt_csv)

    axis0 = np.sort(df[ax0_col].unique())
    axis1 = np.sort(df[ax1_col].unique())
    n0, n1 = len(axis0), len(axis1)

    # Sort by (ax0, ax1) so it matches meshgrid indexing="ij"
    df = df.sort_values([ax0_col, ax1_col]).reset_index(drop=True)

    # Extract all state columns (everything except 'label')
    state_cols = [c for c in df.columns if c != "label"]
    grid_np = df[state_cols].values.astype(np.float32)
    gt_labels = df["label"].values.astype(np.int8)

    print(f"  Loaded {os.path.basename(gt_csv)}: {n0}x{n1} = {len(df)} points")
    return grid_np, gt_labels, axis0, axis1, n0, n1


# ---------------------------------------------------------------------------
# Experiment loading (mirrors eval_roa.py pattern)
# ---------------------------------------------------------------------------

def load_experiment(experiment_dir, checkpoint, device):
    """Load config -> dynamics -> model -> checkpoint.  Returns (dynamics, model, opt)."""
    opt = load_experiment_config(experiment_dir)

    # Build dynamics
    dyn_cls = getattr(dynamics_mod, opt.dynamics_class)
    sig = inspect.signature(dyn_cls)
    dyn_kwargs = {k: opt[k] for k in sig.parameters if k != "self" and k in opt}
    dynamics = dyn_cls(**dyn_kwargs)
    dynamics.deepreach_model = opt.deepreach_model

    # Build model
    model = modules.SingleBVPNet(
        in_features=dynamics.input_dim,
        out_features=1,
        type=opt.model,
        mode=opt.model_mode,
        final_layer_factor=1.0,
        hidden_features=opt.num_nl,
        num_hidden_layers=opt.num_hl,
        omega_0=getattr(opt, "omega_0", 30.0),
    )
    model.to(device)
    model.eval()

    # Load checkpoint
    ckpt_path = os.path.join(experiment_dir, "training", "checkpoints", checkpoint)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)

    return dynamics, model, opt


def compute_grid_values(model, dynamics, states, t_eval, device, batch_size=50000):
    """Evaluate V on a grid, batched to avoid OOM."""
    N = states.shape[0]
    all_vals = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        vals = compute_values(model, dynamics, states[start:end], t_eval, device)
        all_vals.append(vals)
    return np.concatenate(all_vals)


# ---------------------------------------------------------------------------
# Colormaps (matching reference style)
# ---------------------------------------------------------------------------

def _roa_discrete_cmap():
    """Red/yellow/green discrete colormap for ROA: failure=-1, separatrix=0, success=+1."""
    colors = ["#d73027", "#fee08b", "#1a9850"]
    cmap = mcolors.ListedColormap(colors)
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def _gt_discrete_cmap():
    """Red/green discrete colormap for GT labels: failure=-1, success=+1."""
    colors = ["#d73027", "#1a9850"]
    cmap = mcolors.ListedColormap(colors)
    bounds = [-1.5, 0.0, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def _value_cmap():
    """5-color diverging colormap for continuous value function."""
    return mcolors.LinearSegmentedColormap.from_list(
        "value_div", ["#d73027", "#fc8d59", "#ffffbf", "#91bfdb", "#4575b4"]
    )


def classify_values(values, c_low, c_high):
    """Classify V into 3 classes using calibrated thresholds.

    V <= c_low  → +1 (success / safe)
    V >= c_high → -1 (failure / unsafe)
    else        →  0 (separatrix / uncertain)
    """
    pred = np.zeros_like(values, dtype=np.float32)
    pred[values <= c_low] = 1.0
    pred[values >= c_high] = -1.0
    return pred


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

def _style_ax(ax, sl):
    """Apply custom ticks and labels from slice definition."""
    if "xticks" in sl:
        ax.set_xticks(sl["xticks"])
    if "xticklabels" in sl:
        ax.set_xticklabels(sl["xticklabels"], fontsize=8)
    if "yticks" in sl:
        ax.set_yticks(sl["yticks"])
    if "yticklabels" in sl:
        ax.set_yticklabels(sl["yticklabels"], fontsize=8)
    ax.tick_params(labelsize=8)
    if "xlabel" in sl:
        ax.set_xlabel(sl["xlabel"], fontsize=9)
    if "ylabel" in sl:
        ax.set_ylabel(sl["ylabel"], fontsize=9)


def _safe_name(name):
    """Sanitize a name for use in filenames."""
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")


# ---------------------------------------------------------------------------
# Bare single images — no axis, no title, just the heatmap (for papers)
# ---------------------------------------------------------------------------

def _save_bare(heatmap_2d, extent, cmap, output_dir, name, norm=None,
               vmin=None, vmax=None, interp="nearest", contour_data=None):
    """Save a single bare image — no axes, no title, nothing."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(
        heatmap_2d, origin="lower", aspect="auto",
        cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
        extent=extent, interpolation=interp,
    )
    if contour_data is not None:
        xi, xj, vals_2d = contour_data
        try:
            ax.contour(xi, xj, vals_2d.T, levels=[0.0], colors="k", linewidths=1.0)
        except ValueError:
            pass
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    os.makedirs(output_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        out = os.path.join(output_dir, f"{name}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0)
    print(f"  Saved bare: {name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Styled single images — with axes, ticks, colorbar
# ---------------------------------------------------------------------------

def _save_styled(heatmap_2d, extent, cmap, sl, output_dir, name, formats,
                 norm=None, vmin=None, vmax=None, interp="nearest",
                 contour_data=None, title=None, cbar_label=None):
    """Save a styled image with axes, tick labels, optional colorbar."""
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(
        heatmap_2d, origin="lower", aspect="auto",
        cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
        extent=extent, interpolation=interp,
    )
    if contour_data is not None:
        xi, xj, vals_2d = contour_data
        try:
            ax.contour(xi, xj, vals_2d.T, levels=[0.0], colors="k", linewidths=1.2)
        except ValueError:
            pass
    _style_ax(ax, sl)
    if title:
        ax.set_title(title, fontsize=10)
    if cbar_label is not None or norm is None:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=9)
        cbar.ax.tick_params(labelsize=7)

    os.makedirs(output_dir, exist_ok=True)
    for fmt in formats:
        out = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"    Saved styled: {name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def cache_path(output_dir, exp_name, slice_name, t_eval):
    return os.path.join(output_dir, f"cache_{exp_name}_{slice_name}_t{t_eval:.4f}.npz")


def save_cache(path, values_2d, axis0, axis1, pred_labels, c_low, c_high, gt_labels=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_dict = dict(
        values=values_2d, axis0=axis0, axis1=axis1,
        n0=np.array(len(axis0)), n1=np.array(len(axis1)),
        pred_labels=pred_labels,
        c_low=np.array(c_low), c_high=np.array(c_high),
    )
    if gt_labels is not None:
        save_dict["gt_labels"] = gt_labels
    np.savez_compressed(path, **save_dict)


def load_cache(path):
    data = np.load(path)
    gt_labels = data["gt_labels"] if "gt_labels" in data else None
    pred_labels = data["pred_labels"] if "pred_labels" in data else None
    return (data["values"], data["axis0"], data["axis1"],
            int(data["n0"]), int(data["n1"]), gt_labels, pred_labels)


# ---------------------------------------------------------------------------
# Combined multi-panel figures
# ---------------------------------------------------------------------------

def plot_combined_value(results, slices, labels, output_dir, vmin, vmax):
    """Combined continuous V heatmap: rows=experiments, cols=slices."""
    n_exp = len(results)
    n_slc = len(slices)
    cmap = _value_cmap()

    fig, axes = plt.subplots(
        n_exp, n_slc,
        figsize=(3.8 * n_slc, 3.0 * n_exp),
        squeeze=False,
        constrained_layout=True,
    )

    im = None
    for r, (label, exp_data) in enumerate(zip(labels, results)):
        for c, sl in enumerate(slices):
            ax = axes[r, c]
            vals_2d, axis0, axis1, _pred, _gt, _dyn = exp_data[sl["name"]]
            ext = [axis0[0], axis0[-1], axis1[0], axis1[-1]]

            vlim = max(abs(vmin or vals_2d.min()), abs(vmax or vals_2d.max()))
            v_lo = vmin if vmin is not None else -vlim
            v_hi = vmax if vmax is not None else vlim

            im = ax.imshow(
                vals_2d.T, origin="lower", extent=ext, aspect="auto",
                cmap=cmap, vmin=v_lo, vmax=v_hi, interpolation="bilinear",
            )
            try:
                ax.contour(axis0, axis1, vals_2d.T, levels=[0.0], colors="k", linewidths=1.0)
            except ValueError:
                pass
            _style_ax(ax, sl)
            if r == 0:
                ax.set_title(sl["name"], fontsize=10)
            if c == 0:
                ax.annotate(
                    label, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords=ax.yaxis.label, textcoords="offset points",
                    fontsize=9, ha="right", va="center",
                )

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label(r"$V(x, t)$", fontsize=9)
        cbar.ax.tick_params(labelsize=7)

    os.makedirs(output_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        out = os.path.join(output_dir, f"combined_value.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved {out}")
    plt.close(fig)


def plot_combined_roa(results, slices, labels, output_dir, gt_data=None):
    """Combined binary ROA heatmap: rows = (GT + experiments), cols = slices."""
    has_gt = gt_data is not None and any(gt_data.get(sl["name"]) for sl in slices)
    n_gt = 1 if has_gt else 0
    n_exp = len(results)
    n_rows = n_gt + n_exp
    n_slc = len(slices)
    row_labels = (["Ground Truth"] if has_gt else []) + list(labels)

    cmap, norm = _roa_discrete_cmap()
    gt_cmap, gt_norm = _gt_discrete_cmap()

    fig, axes = plt.subplots(
        n_rows, n_slc,
        figsize=(3.5 * n_slc, 3.0 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for col, sl in enumerate(slices):
        # Ground truth row
        if has_gt and gt_data.get(sl["name"]):
            ax = axes[0, col]
            gt_hm, gt_axis0, gt_axis1, _, _ = gt_data[sl["name"]]
            gt_ext = [gt_axis0[0], gt_axis0[-1], gt_axis1[0], gt_axis1[-1]]
            ax.imshow(
                gt_hm.T, origin="lower", aspect="auto",
                cmap=gt_cmap, norm=gt_norm, extent=gt_ext, interpolation="nearest",
            )
            _style_ax(ax, sl)
        elif has_gt:
            axes[0, col].set_visible(False)

        # Model rows
        for r_off, (label, exp_data) in enumerate(zip(labels, results)):
            ax = axes[n_gt + r_off, col]
            vals_2d, axis0, axis1, pred, _gt, _dyn = exp_data[sl["name"]]
            ext_sl = [axis0[0], axis0[-1], axis1[0], axis1[-1]]
            ax.imshow(
                pred.T, origin="lower", aspect="auto",
                cmap=cmap, norm=norm, extent=ext_sl, interpolation="nearest",
            )
            _style_ax(ax, sl)

        # Column titles
        if n_slc > 1:
            axes[0, col].set_title(sl["name"], fontsize=10)

    # Row labels
    for row, lbl in enumerate(row_labels):
        ax = axes[row, 0]
        ax.annotate(
            lbl, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
            xycoords=ax.yaxis.label, textcoords="offset points",
            fontsize=9, ha="right", va="center",
        )

    os.makedirs(output_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        out = os.path.join(output_dir, f"combined_roa.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    # Pre-parse to check for -c / --config YAML file
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("-c", "--config", default=None, help="YAML config file")
    pre_args, _remaining = pre.parse_known_args()

    # Load YAML defaults if provided
    yaml_defaults = {}
    if pre_args.config:
        with open(pre_args.config, "r") as f:
            yaml_defaults = yaml.safe_load(f) or {}

    p = argparse.ArgumentParser(
        parents=[pre],
        description="Qualitative value-function heatmap plots for DeepReach experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment_dirs", nargs="+", default=None,
                    help="Paths to experiment folders (runs/<name>)")
    p.add_argument("--labels", nargs="+", default=None,
                    help="Display labels per experiment (default: folder names)")
    p.add_argument("--checkpoint", default="model_final.pth",
                    help="Checkpoint filename under training/checkpoints")
    p.add_argument("--t_eval", type=float, default=None,
                    help="Evaluation time (default: tMax from first experiment config)")
    p.add_argument("--slice", nargs="+", default=None,
                    help="Slice name(s) to plot (e.g. theta_thetadot)")
    p.add_argument("--all_slices", action="store_true",
                    help="Plot all default slices for the system")
    p.add_argument("--device", default="cpu",
                    help="Torch device (cpu or cuda:N)")
    p.add_argument("--recompute", action="store_true",
                    help="Ignore cached grids and recompute")
    p.add_argument("--output_dir", default="results/figures",
                    help="Output directory for figures and caches")
    p.add_argument("--format", nargs="+", default=["png", "pdf"],
                    help="Output format(s) for styled images (default: png pdf)")
    p.add_argument("--vmin", type=float, default=None,
                    help="Colorbar lower bound for value plots")
    p.add_argument("--vmax", type=float, default=None,
                    help="Colorbar upper bound for value plots")
    p.add_argument("--no_combined", action="store_true",
                    help="Skip combined multi-panel figures")
    p.add_argument("--no_bare", action="store_true",
                    help="Skip bare (no-axis) single images")
    p.add_argument("--no_styled", action="store_true",
                    help="Skip styled (with-axis) single images")
    p.add_argument("--c_low", type=float, default=DEFAULT_C_LOW,
                    help=f"Calibrated safe threshold: V <= c_low is success (default: {DEFAULT_C_LOW})")
    p.add_argument("--c_high", type=float, default=DEFAULT_C_HIGH,
                    help=f"Calibrated unsafe threshold: V >= c_high is failure (default: {DEFAULT_C_HIGH})")
    p.add_argument("--no_gt", action="store_true",
                    help="Skip ground truth overlay even if gt_csv exists in slice def")
    p.add_argument("--batch_size", type=int, default=50000,
                    help="Batch size for model evaluation (default 50000)")

    # Apply YAML defaults (CLI takes priority)
    p.set_defaults(**yaml_defaults)
    args = p.parse_args()

    if not args.experiment_dirs:
        p.error("--experiment_dirs is required (via CLI or YAML config)")

    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load experiments
    # ------------------------------------------------------------------
    experiments = []
    for edir in args.experiment_dirs:
        print(f"Loading {edir} ...")
        dyn, model, opt = load_experiment(edir, args.checkpoint, args.device)
        exp_name = os.path.basename(os.path.normpath(edir))
        experiments.append((exp_name, dyn, model, opt))

    labels = args.labels or [e[0] for e in experiments]
    if len(labels) != len(experiments):
        raise ValueError(f"Got {len(labels)} labels but {len(experiments)} experiments")

    # Use tMax from first experiment if t_eval not given
    if args.t_eval is None:
        args.t_eval = float(experiments[0][3].get("tMax", 1.0))
        print(f"Using t_eval={args.t_eval} from first experiment's tMax")

    # ------------------------------------------------------------------
    # Determine slices
    # ------------------------------------------------------------------
    ref_dyn = experiments[0][1]
    all_slices = auto_slices(ref_dyn)

    if args.slice and not args.all_slices:
        requested = set(args.slice)
        slices = [s for s in all_slices if s["name"] in requested]
        missing = requested - {s["name"] for s in slices}
        if missing:
            avail = [s["name"] for s in all_slices]
            raise ValueError(f"Unknown slice(s): {missing}. Available: {avail}")
    else:
        slices = all_slices

    print(f"Slices to plot: {[s['name'] for s in slices]}")

    # ------------------------------------------------------------------
    # Evaluate and plot
    # ------------------------------------------------------------------
    print(f"Calibrated thresholds: c_low={args.c_low}, c_high={args.c_high}")
    all_results = []  # one dict per experiment: {slice_name: (vals_2d, axis0, axis1, pred_labels, gt_labels, dyn)}
    gt_data = {}      # {slice_name: (gt_heatmap, axis0, axis1, n0, n1)}
    value_cmap = _value_cmap()
    roa_cmap, roa_norm = _roa_discrete_cmap()

    for exp_name, dyn, model, opt in experiments:
        exp_data = {}
        safe_exp = _safe_name(exp_name)

        for sl in slices:
            cp = cache_path(args.output_dir, exp_name, sl["name"], args.t_eval)

            if os.path.exists(cp) and not args.recompute:
                print(f"  [{exp_name}] Loading cache for {sl['name']}")
                vals_2d, axis0, axis1, n0, n1, gt_labels, pred_labels = load_cache(cp)
                if pred_labels is None:
                    # Old cache without pred_labels — recompute classification
                    pred_labels = classify_values(vals_2d, args.c_low, args.c_high)
            else:
                grid_np, gt_labels, axis0, axis1, n0, n1 = load_slice_data(sl, dyn.state_dim)

                print(f"  [{exp_name}] Computing V on {n0}x{n1} grid for {sl['name']} ...")
                vals = compute_grid_values(model, dyn, grid_np, args.t_eval, args.device, args.batch_size)
                vals_2d = vals.reshape(n0, n1)
                pred_labels = classify_values(vals_2d, args.c_low, args.c_high)
                save_cache(cp, vals_2d, axis0, axis1, pred_labels, args.c_low, args.c_high, gt_labels)
                print(f"    Cached -> {cp}")

            n_safe = np.sum(pred_labels == 1.0)
            n_unsafe = np.sum(pred_labels == -1.0)
            n_sep = np.sum(pred_labels == 0.0)
            total = pred_labels.size
            print(f"    {n_safe} safe, {n_unsafe} unsafe, {n_sep} separatrix "
                  f"({100*n_sep/total:.1f}% uncertain)")

            exp_data[sl["name"]] = (vals_2d, axis0, axis1, pred_labels, gt_labels, dyn)
            ext = [axis0[0], axis0[-1], axis1[0], axis1[-1]]
            contour_data = (axis0, axis1, vals_2d)

            # Collect GT data (only once, from first experiment)
            if sl["name"] not in gt_data and gt_labels is not None and not args.no_gt:
                gt_hm = np.where(gt_labels.reshape(n0, n1) == 1, 1.0, -1.0)
                gt_data[sl["name"]] = (gt_hm, axis0, axis1, n0, n1)

            # ---- Bare images (no axis, no title — for paper figures) ----
            if not args.no_bare:
                # Continuous value
                vlim = max(abs(args.vmin or vals_2d.min()), abs(args.vmax or vals_2d.max()))
                v_lo = args.vmin if args.vmin is not None else -vlim
                v_hi = args.vmax if args.vmax is not None else vlim
                _save_bare(
                    vals_2d.T, ext, value_cmap, args.output_dir,
                    f"{safe_exp}_{sl['name']}_value_bare",
                    vmin=v_lo, vmax=v_hi, interp="bilinear",
                    contour_data=contour_data,
                )
                # Calibrated ROA (3-class)
                _save_bare(
                    pred_labels.T, ext, roa_cmap, args.output_dir,
                    f"{safe_exp}_{sl['name']}_roa_bare",
                    norm=roa_norm,
                )

            # ---- Styled images (with axes, ticks, colorbar) ----
            if not args.no_styled:
                vlim = max(abs(args.vmin or vals_2d.min()), abs(args.vmax or vals_2d.max()))
                v_lo = args.vmin if args.vmin is not None else -vlim
                v_hi = args.vmax if args.vmax is not None else vlim
                _save_styled(
                    vals_2d.T, ext, value_cmap, sl, args.output_dir,
                    f"{safe_exp}_{sl['name']}_value", args.format,
                    vmin=v_lo, vmax=v_hi, interp="bilinear",
                    contour_data=contour_data,
                    title=f"{exp_name}", cbar_label=r"$V(x, t)$",
                )
                _save_styled(
                    pred_labels.T, ext, roa_cmap, sl, args.output_dir,
                    f"{safe_exp}_{sl['name']}_roa", args.format,
                    norm=roa_norm,
                    title=f"{exp_name}",
                )

        all_results.append(exp_data)

        # Free GPU memory between experiments
        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Ground truth bare images
    # ------------------------------------------------------------------
    if not args.no_bare and gt_data:
        gt_cmap, gt_norm = _gt_discrete_cmap()
        for sl in slices:
            if sl["name"] in gt_data:
                gt_hm, gt_a0, gt_a1, _, _ = gt_data[sl["name"]]
                gt_ext = [gt_a0[0], gt_a0[-1], gt_a1[0], gt_a1[-1]]
                _save_bare(gt_hm.T, gt_ext, gt_cmap, args.output_dir,
                           f"ground_truth_{sl['name']}_bare", norm=gt_norm)

    # ------------------------------------------------------------------
    # Combined multi-panel figures
    # ------------------------------------------------------------------
    if not args.no_combined:
        plot_combined_value(all_results, slices, labels, args.output_dir,
                            args.vmin, args.vmax)
        plot_combined_roa(all_results, slices, labels, args.output_dir,
                          gt_data=gt_data if gt_data else None)

    print("Done.")


if __name__ == "__main__":
    main()
