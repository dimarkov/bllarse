#!/usr/bin/env python3
"""Fetch W&B sweep results and compute baselines, then plot metrics.

This script:
1. Fetches run data from W&B for the specified sweep group
2. Computes baseline metrics by loading in21k_cifar pretrained models
3. Saves all data to a CSV file
4. Creates publication-ready figures showing acc, ece, nll vs batch_size/num_iters

Usage:
    python scripts/plot_sweep_results.py --group sweep7a1_fnf_dataaug_adamw_batchsize_numiters
    python scripts/plot_sweep_results.py --csv-only  # Only fetch/save CSV, skip plotting
    python scripts/plot_sweep_results.py --plot-only --input results.csv  # Only plot from existing CSV
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Lazy imports for JAX/model loading (only needed for baseline computation)
_jax_loaded = False


def _ensure_jax():
    """Lazy load JAX and related dependencies."""
    global _jax_loaded
    if _jax_loaded:
        return
    
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    
    import jax
    jax.config.update("jax_default_matmul_precision", "highest")
    _jax_loaded = True


# --------------------------------------------------------------------------
# W&B Data Fetching
# --------------------------------------------------------------------------

def fetch_wandb_runs(project: str, group: str) -> pd.DataFrame:
    """Fetch all runs from a W&B group and return as DataFrame."""
    import wandb
    import sys
    
    api = wandb.Api(timeout=60)
    print(f"  Querying runs with group={group}...")
    sys.stdout.flush()
    
    runs = api.runs(
        path=project,
        filters={"group": group},
    )
    
    # Convert to list to get count (runs is a lazy iterator)
    print(f"  Fetching run list...")
    sys.stdout.flush()
    runs_list = list(runs)
    total_runs = len(runs_list)
    print(f"  Found {total_runs} runs, fetching history...")
    sys.stdout.flush()
    
    records = []
    for i, run in enumerate(runs_list):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing run {i + 1}/{total_runs}: {run.name}")
            sys.stdout.flush()
        
        config = run.config
        try:
            history = run.history(pandas=True)
        except Exception as e:
            print(f"    Warning: could not fetch history for {run.name}: {e}")
            continue
        
        if history.empty:
            continue
        
        # Extract run metadata
        base_record = {
            "run_id": run.id,
            "run_name": run.name,
            "dataset": config.get("dataset", "unknown"),
            "batch_size": config.get("batch_size", 0),
            "num_update_iters": config.get("num_vb_iters", config.get("num_update_iters", 16)),
            "nodataaug": config.get("nodataaug", False),
            "optimizer": config.get("optimizer", "adamw"),
            "learning_rate": config.get("learning_rate", 0),
            "weight_decay": config.get("weight_decay", 0),
            "seed": config.get("seed", 0),
            "sequential_update": config.get("sequential_update", False),
            "reset_loss_per_epoch": config.get("reset_loss_per_epoch", False),
        }
        
        # Extract metrics for each epoch
        for _, row in history.iterrows():
            epoch = int(row.get("epoch", row.name + 1))
            record = base_record.copy()
            record.update({
                "epoch": epoch,
                "acc": row.get("acc", np.nan),
                "ece": row.get("ece", np.nan),
                "nll": row.get("nll", np.nan),
                "loss": row.get("loss", np.nan),
                "type": "experiment",
            })
            records.append(record)
    
    print(f"  Done fetching {len(records)} records from {total_runs} runs")
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Baseline Computation
# --------------------------------------------------------------------------

def compute_baselines(datasets: list[str]) -> pd.DataFrame:
    """Load in21k_cifar pretrained models and compute baseline metrics."""
    _ensure_jax()
    
    import jax.numpy as jnp
    import equinox as eqx
    from jax import random as jr
    from datasets import load_dataset
    from functools import partial
    
    from mlpox.load_models import load_model
    from bllarse.losses import CrossEntropy
    from bllarse.utils import (
        resize_images,
        augmentdata,
        evaluate_model,
        MEAN_DICT,
        STD_DICT,
    )
    
    records = []
    
    for dataset in datasets:
        print(f"Computing baseline for {dataset}...")
        
        # Load test data
        ds = load_dataset(dataset).with_format("jax")
        label_key = 'fine_label' if dataset == 'cifar100' else 'label'
        test_ds = {
            'image': ds['test']['img'][:].astype(jnp.float32),
            'label': ds['test'][label_key][:]
        }
        
        # Resize images
        test_ds['image'] = resize_images(test_ds['image'], 64)
        
        # Setup augmentation (no random aug for eval)
        mean = MEAN_DICT[dataset]
        std = STD_DICT[dataset]
        augdata = partial(augmentdata, mean=mean, std=std)
        _augdata = lambda img, key=None, **kwargs: augdata(img, key=None, **kwargs)
        
        # Load pretrained model
        # Model naming: B_{num_blocks}-Wi_{embed_dim}_res_64_in21k_{dataset}
        for num_blocks, embed_dim in [(12, 1024)]:
            name = f"B_{num_blocks}-Wi_{embed_dim}_res_64_in21k_{dataset}"
            try:
                nnet = eqx.nn.inference_mode(load_model(name), True)
            except Exception as e:
                print(f"  Could not load {name}: {e}")
                continue
            
            num_classes = 10 if dataset == 'cifar10' else 100
            loss_fn = CrossEntropy(0.0, num_classes)
            
            acc, nll, ece = evaluate_model(
                _augdata, loss_fn, nnet,
                test_ds['image'], test_ds['label']
            )
            
            print(f"  {name}: acc={float(acc):.4f}, ece={float(ece):.4f}, nll={float(nll):.4f}")
            
            records.append({
                "dataset": dataset,
                "num_blocks": num_blocks,
                "embed_dim": embed_dim,
                "acc": float(acc),
                "ece": float(ece),
                "nll": float(nll),
                "type": "baseline",
            })
    
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

METRICS = ["acc", "ece", "nll"]
METRIC_LABELS = {"acc": "Accuracy", "ece": "ECE", "nll": "NLL"}
EPOCHS_TO_PLOT = [1, 5, 10]


def make_figure(
    df: pd.DataFrame,
    output_dir: str,
    sequential_update: bool | None = None,
    reset_loss_per_epoch: bool | None = None,
):
    """Create 3x6 figure: rows are datasets (cifar10/cifar100), columns are metrics × data_aug.
    
    Layout:
    - Rows 1-3: cifar10 (acc, ece, nll)
    - Rows 4-6: cifar100 (acc, ece, nll)
    - Columns 1-3: No data augmentation (epochs 1, 5, 10)
    - Columns 4-6: With data augmentation (epochs 1, 5, 10)
    
    Each subplot: x-axis = batch_size, lines = different num_update_iters
    
    Args:
        sequential_update: If not None, filter to runs with this value
        reset_loss_per_epoch: If not None, filter to runs with this value
    """
    df_exp = df[df["type"] == "experiment"].copy()
    df_bl = df[df["type"] == "baseline"]
    
    # Apply filters
    filter_desc = []
    if sequential_update is not None:
        df_exp = df_exp[df_exp["sequential_update"] == sequential_update]
        filter_desc.append(f"seq_update={sequential_update}")
    if reset_loss_per_epoch is not None:
        df_exp = df_exp[df_exp["reset_loss_per_epoch"] == reset_loss_per_epoch]
        filter_desc.append(f"reset_loss={reset_loss_per_epoch}")
    
    filter_suffix = "_" + "_".join(filter_desc) if filter_desc else ""
    filter_title = " (" + ", ".join(filter_desc) + ")" if filter_desc else ""
    
    datasets = ["cifar10", "cifar100"]
    dataaug_settings = [True, False]  # nodataaug: False means WITH aug
    
    # Figure: 6 rows (3 metrics × 2 datasets), 6 columns (3 epochs × 2 aug settings)
    n_rows = len(datasets) * len(METRICS)  # 6
    n_cols = len(EPOCHS_TO_PLOT) * len(dataaug_settings)  # 6
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(20, 15),
        squeeze=False,
        sharex=True,
        sharey='row'
    )
    
    # Get unique values for coloring
    update_iters_list = sorted(df_exp["num_update_iters"].dropna().unique())
    batch_sizes = sorted(df_exp["batch_size"].dropna().unique())
    
    # Color map for num_update_iters
    cmap = plt.cm.viridis
    colors = {
        it: cmap(i / max(len(update_iters_list) - 1, 1))
        for i, it in enumerate(update_iters_list)
    }
    
    for ds_idx, dataset in enumerate(datasets):
        df_ds = df_exp[df_exp["dataset"] == dataset]
        df_bl_ds = df_bl[df_bl["dataset"] == dataset]
        
        # Get baseline value (assuming embed_dim=1024, num_blocks=12 for main experiments)
        bl_row = df_bl_ds[(df_bl_ds["num_blocks"] == 12) & (df_bl_ds["embed_dim"] == 1024)]
        if bl_row.empty:
            bl_row = df_bl_ds.iloc[:1] if not df_bl_ds.empty else None
        
        for m_idx, metric in enumerate(METRICS):
            row_idx = ds_idx * len(METRICS) + m_idx
            
            for aug_idx, nodataaug in enumerate([True, False]):  # No aug first, then with aug
                df_aug = df_ds[df_ds["nodataaug"] == nodataaug]
                aug_label = "No Aug" if nodataaug else "With Aug"
                
                for ep_idx, epoch in enumerate(EPOCHS_TO_PLOT):
                    col_idx = aug_idx * len(EPOCHS_TO_PLOT) + ep_idx
                    ax = axes[row_idx, col_idx]
                    
                    df_epoch = df_aug[df_aug["epoch"] == epoch]
                    
                    # Plot baseline as horizontal dashed line
                    if bl_row is not None and not bl_row.empty:
                        bl_val = bl_row[metric].values[0]
                        ax.axhline(bl_val, color="grey", ls="--", lw=1.5, label="baseline", zorder=1)
                    
                    # Plot one line per num_update_iters
                    for ui in update_iters_list:
                        df_ui = df_epoch[df_epoch["num_update_iters"] == ui]
                        if df_ui.empty:
                            continue
                        
                        # Aggregate over seeds
                        stats = df_ui.groupby("batch_size")[metric].agg(["mean", "std"])
                        stats = stats.reindex(batch_sizes).dropna()
                        
                        if stats.empty:
                            continue
                        
                        x = np.array(stats.index)
                        y = stats["mean"].values
                        yerr = stats["std"].values
                        
                        ax.plot(x, y, marker="o", ms=4, color=colors[ui], 
                               label=f"iters={int(ui)}", zorder=2)
                        ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=colors[ui])
                    
                    # Formatting
                    ax.set_xscale("log", base=2)
                    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
                    if batch_sizes:
                        ax.xaxis.set_major_locator(ticker.FixedLocator(batch_sizes))
                    ax.tick_params(axis="x", rotation=45)
                    
                    # Titles on top row
                    if row_idx == 0:
                        ax.set_title(f"{aug_label}\nEpoch {epoch}", fontsize=10, fontweight="bold")
                    
                    # X-label on bottom row
                    if row_idx == n_rows - 1:
                        ax.set_xlabel("Batch Size")
                    
                    # Y-label on left column
                    if col_idx == 0:
                        ax.set_ylabel(f"{dataset}\n{METRIC_LABELS[metric]}", fontsize=9)
    
    # Single legend
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    
    fig.legend(
        handles, labels,
        loc="center left",
        ncol=1,
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(1.01, 0.5),
    )
    
    fig.suptitle(
        f"Sweep Results: acc/ece/nll vs Batch Size{filter_title}",
        fontsize=14,
        y=1.005,
    )
    fig.tight_layout()
    
    fname = f"{output_dir}/sweep_results{filter_suffix}.pdf"
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    print(f"Saved {fname}")
    
    fname_png = f"{output_dir}/sweep_results{filter_suffix}.png"
    fig.savefig(fname_png, bbox_inches="tight", dpi=150)
    print(f"Saved {fname_png}")
    
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--project",
        type=str,
        default="verses_ai/bllarse_experiments",
        help="W&B project path",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="sweep7a1_fnf_dataaug_adamw_batchsize_numiters",
        help="W&B run group name",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to existing CSV (skip W&B fetch if provided)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scripts/sweep_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scripts/figures",
        help="Directory for output figures",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Only fetch data and save CSV, skip plotting",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only plot from existing CSV (requires --input)",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip computing baseline metrics",
    )
    parser.add_argument(
        "--sequential-update",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Filter plots by sequential_update value (true/false)",
    )
    parser.add_argument(
        "--reset-loss-per-epoch",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Filter plots by reset_loss_per_epoch value (true/false)",
    )
    args = parser.parse_args()
    
    # Convert string args to bool
    seq_update = None if args.sequential_update is None else (args.sequential_update == "true")
    reset_loss = None if args.reset_loss_per_epoch is None else (args.reset_loss_per_epoch == "true")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.plot_only:
        if not args.input:
            parser.error("--plot-only requires --input")
        df = pd.read_csv(args.input)
    else:
        # Fetch W&B data
        if args.input and os.path.exists(args.input):
            print(f"Loading existing data from {args.input}")
            df = pd.read_csv(args.input)
        else:
            print(f"Fetching runs from W&B: {args.project} / {args.group}")
            df = fetch_wandb_runs(args.project, args.group)
            print(f"  Found {len(df)} records from {df['run_id'].nunique()} runs")
        
        # Compute baselines
        if not args.skip_baselines:
            datasets = df[df["type"] == "experiment"]["dataset"].unique().tolist()
            if not datasets:
                datasets = ["cifar10", "cifar100"]
            df_bl = compute_baselines(datasets)
            df = pd.concat([df, df_bl], ignore_index=True)
        
        # Save CSV
        df.to_csv(args.output, index=False)
        print(f"Saved {len(df)} records to {args.output}")
    
    if args.csv_only:
        return
    
    # Plot
    make_figure(df, args.output_dir, sequential_update=seq_update, reset_loss_per_epoch=reset_loss)


if __name__ == "__main__":
    main()
