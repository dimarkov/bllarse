#!/usr/bin/env python3
"""Plot sweep results from local results directory.

This script:
1. Loads all run data from scripts/results directory (config.json + parquet files)
2. Creates pandas DataFrames with experiment data and baselines
3. Creates publication-ready figures showing acc, ece, nll vs batch_size/num_iters

Usage:
    python scripts/plot_sweep_results.py
    python scripts/plot_sweep_results.py --results-dir scripts/results
    python scripts/plot_sweep_results.py --output-dir scripts/figures
"""

import os
import json
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# --------------------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------------------

def load_run_data(run_dir: Path) -> tuple[dict, pd.DataFrame | None]:
    """Load config and metrics from a single run directory.
    
    Returns:
        Tuple of (config dict, metrics DataFrame or None if no data)
    """
    config_path = run_dir / "config.json"
    
    if not config_path.exists():
        return {}, None
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Find parquet files (can be multiple shards)
    parquet_files = list(run_dir.glob("*.parquet"))
    if not parquet_files:
        return config, None
    
    # Load and concatenate all parquet files
    dfs = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not read {pf}: {e}")
    
    if not dfs:
        return config, None
    
    metrics_df = pd.concat(dfs, ignore_index=True)
    return config, metrics_df


def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Load all run results from the results directory.
    
    Args:
        results_dir: Path to directory containing run subdirectories
        
    Returns:
        DataFrame with all experiment records
    """
    print(f"Loading results from {results_dir}...")
    
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    total_runs = len(run_dirs)
    print(f"  Found {total_runs} run directories")
    
    records = []
    loaded_runs = 0
    
    for i, run_dir in enumerate(run_dirs):
        
        config, metrics_df = load_run_data(run_dir)
        
        if metrics_df is None or metrics_df.empty:
            continue
        
        loaded_runs += 1
        run_id = run_dir.name
        
        # Base record from config
        base_record = {
            "run_id": run_id,
            "run_name": run_id,
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
            "loss_fn": config.get("loss_fn", "CrossEntropy"),
            "embed_dim": config.get("embed_dim", 1024),
            "num_blocks": config.get("num_blocks", 12),
        }
        
        # Extract metrics for each epoch
        for _, row in metrics_df.iterrows():
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
    
    print(f"  Loaded {len(records)} records from {loaded_runs} runs")
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Baseline and Linear Probing Data Loading
# --------------------------------------------------------------------------

def load_reference_data(
    csv_path: str = "scripts/results_ibprobit_large_batch.csv",
    n_samples: int = 16,
    seed: int = 42,
) -> pd.DataFrame:
    """Load baseline and linear probing (ibprobit) data from CSV.
    
    Args:
        csv_path: Path to the CSV file with results
        n_samples: Number of random runs to sample for linear probing averaging
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with baseline and linear_probing records per dataset
    """
    if not os.path.exists(csv_path):
        print(f"Warning: CSV not found: {csv_path}")
        return pd.DataFrame()
    
    print(f"Loading reference data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter for matching model architecture (num_blocks=12, embed_dim=1024)
    df_filtered = df[
        (df["num_blocks"] == 12) & 
        (df["embed_dim"] == 1024)
    ].copy()
    
    if df_filtered.empty:
        print("  No matching data found for num_blocks=12, embed_dim=1024")
        return pd.DataFrame()
    
    np.random.seed(seed)
    
    records = []
    
    for dataset in df_filtered["dataset"].unique():
        df_ds = df_filtered[df_filtered["dataset"] == dataset]
        
        # Load baseline (pretrained model, type="baseline")
        df_bl = df_ds[df_ds["type"] == "baseline"]
        if not df_bl.empty:
            # Take first baseline value (should be unique per dataset)
            bl_row = df_bl.iloc[0]
            records.append({
                "dataset": dataset,
                "acc": bl_row["acc"],
                "ece": bl_row["ece"],
                "nll": bl_row["nll"],
                "type": "baseline",
            })
            print(f"  {dataset} baseline: acc={bl_row['acc']:.4f}, ece={bl_row['ece']:.4f}, nll={bl_row['nll']:.4f}")
        
        # Load linear probing (ibprobit, type="ibprobit")
        # Take best values at largest batch size across all num_iters
        # Filter for pretrained_source="in21k" and nodataaug (data_aug=False)
        df_lp = df_ds[
            (df_ds["type"] == "ibprobit") &
            (df_ds["pretrained_source"] == "in21k") &
            (df_ds["data_aug"] == False)
        ]
        if not df_lp.empty:
            # Filter for largest batch size
            max_batch = df_lp["batch_size"].max()
            df_lp_max_batch = df_lp[df_lp["batch_size"] == max_batch]
            
            # For each num_iters: subsample 16 runs, compute mean of each metric
            # Then find the num_iters with best mean accuracy
            best_mean_acc = -1
            best_metrics = None
            best_num_iters = None
            
            for num_iters in df_lp_max_batch["update_iters"].unique():
                df_iters = df_lp_max_batch[df_lp_max_batch["update_iters"] == num_iters]
                
                # Subsample to 16 runs (or all if fewer)
                n_to_sample = min(16, len(df_iters))
                sampled = df_iters.sample(n=n_to_sample, random_state=seed)
                
                mean_acc = sampled["acc"].mean()
                mean_ece = sampled["ece"].mean()
                mean_nll = sampled["nll"].mean()
                
                if mean_acc > best_mean_acc:
                    best_mean_acc = mean_acc
                    best_metrics = (mean_acc, mean_ece, mean_nll)
                    best_num_iters = num_iters
            
            if best_metrics is not None:
                records.append({
                    "dataset": dataset,
                    "acc": best_metrics[0],
                    "ece": best_metrics[1],
                    "nll": best_metrics[2],
                    "type": "linear_probing",
                })
                print(f"  {dataset} linear probing (in21k, no aug, batch={max_batch}, iters={best_num_iters}): acc={best_metrics[0]:.4f}, ece={best_metrics[1]:.4f}, nll={best_metrics[2]:.4f}")
    
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
    """Create separate 3x6 figures for with/without data augmentation.
    
    Layout per figure:
    - Rows: acc, ece, nll (3 metrics)
    - Columns 1-3: cifar10 epochs 1, 5, 10
    - Columns 4-6: cifar100 epochs 1, 5, 10
    
    Y-axis is aligned within each dataset's columns (first 3 cols share y, last 3 cols share y).
    """
    df_exp = df[df["type"] == "experiment"].copy()
    df_bl = df[df["type"] == "baseline"]
    df_lp = df[df["type"] == "linear_probing"]
    
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
    
    # Get unique values for coloring
    update_iters_list = sorted(df_exp["num_update_iters"].dropna().unique())
    batch_sizes = sorted(df_exp["batch_size"].dropna().unique())
    
    # Color map for num_update_iters - use tab10 for divergent/distinct colors
    cmap = plt.cm.tab10
    colors = {
        it: cmap(i % 10)
        for i, it in enumerate(update_iters_list)
    }
    
    # Create separate figures for each augmentation setting
    for nodataaug in [True, False]:
        aug_label = "No Data Augmentation" if nodataaug else "With Data Augmentation"
        aug_suffix = "_no_aug" if nodataaug else "_with_aug"
        
        df_aug = df_exp[df_exp["nodataaug"] == nodataaug]
        
        # Figure: 3 rows (metrics) × 6 columns (3 epochs × 2 datasets)
        n_rows = len(METRICS)  # 3
        n_cols = len(EPOCHS_TO_PLOT) * len(datasets)  # 6
        
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(18, 9),
            squeeze=False,
            sharex=True,
        )
        
        for m_idx, metric in enumerate(METRICS):
            # Collect y-limits per dataset for alignment
            ylims_per_dataset = {ds: [] for ds in datasets}
            
            for ds_idx, dataset in enumerate(datasets):
                df_ds = df_aug[df_aug["dataset"] == dataset]
                df_bl_ds = df_bl[df_bl["dataset"] == dataset]
                df_lp_ds = df_lp[df_lp["dataset"] == dataset]
                
                # Get baseline value
                bl_row = df_bl_ds.iloc[0] if not df_bl_ds.empty else None
                
                # Get linear probing value
                lp_row = df_lp_ds.iloc[0] if not df_lp_ds.empty else None
                
                for ep_idx, epoch in enumerate(EPOCHS_TO_PLOT):
                    col_idx = ds_idx * len(EPOCHS_TO_PLOT) + ep_idx
                    ax = axes[m_idx, col_idx]
                    
                    df_epoch = df_ds[df_ds["epoch"] == epoch]
                    
                    # Plot baseline as horizontal grey dashed line
                    if bl_row is not None:
                        bl_val = bl_row[metric]
                        ax.axhline(bl_val, color="grey", ls="--", lw=1.5, label="baseline", zorder=1)
                    
                    # Plot linear probing as horizontal orange dotted line
                    if lp_row is not None:
                        lp_val = lp_row[metric]
                        ax.axhline(lp_val, color="orange", ls=":", lw=2, label="linear probing", zorder=1)
                    
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
                    if m_idx == 0:
                        ax.set_title(f"{dataset}\nEpoch {epoch}", fontsize=10, fontweight="bold")
                    
                    # X-label on bottom row
                    if m_idx == n_rows - 1:
                        ax.set_xlabel("Batch Size")
                    
                    # Y-label on left column of each dataset group
                    if ep_idx == 0:
                        ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
                    
                    # Collect y-limits for this dataset
                    ylims_per_dataset[dataset].append(ax.get_ylim())
            
            # Align y-axis within each dataset's columns
            for ds_idx, dataset in enumerate(datasets):
                ylims = ylims_per_dataset[dataset]
                if ylims:
                    ymin = min(yl[0] for yl in ylims)
                    ymax = max(yl[1] for yl in ylims)
                    for ep_idx in range(len(EPOCHS_TO_PLOT)):
                        col_idx = ds_idx * len(EPOCHS_TO_PLOT) + ep_idx
                        axes[m_idx, col_idx].set_ylim(ymin, ymax)
        
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
            f"Sweep Results: {aug_label}{filter_title}",
            fontsize=14,
            y=1.005,
        )
        fig.tight_layout()
        
        fname = f"{output_dir}/sweep_results{aug_suffix}{filter_suffix}.pdf"
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        print(f"Saved {fname}")
        
        plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-dir",
        type=str,
        default="scripts/results",
        help="Directory containing run subdirectories with config.json and parquet files",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional: Save combined data to CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scripts/figures",
        help="Directory for output figures",
    )
    parser.add_argument(
        "--reference-csv",
        type=str,
        default="scripts/results_ibprobit_large_batch.csv",
        help="Path to CSV with baseline and linear probing results",
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
    
    # Step 1: Load all data from results directory
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    df = load_all_results(results_dir)
    
    # Step 2: Load baseline and linear probing reference data from CSV
    df_ref = load_reference_data(args.reference_csv, n_samples=16)
    if not df_ref.empty:
        df = pd.concat([df, df_ref], ignore_index=True)
    
    # Step 3: Optionally save to CSV
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"Saved {len(df)} records to {args.output_csv}")
    
    # Step 4: Create plots
    make_figure(df, args.output_dir, sequential_update=seq_update, reset_loss_per_epoch=reset_loss)


if __name__ == "__main__":
    main()
