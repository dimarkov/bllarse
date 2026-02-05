#!/usr/bin/env python3
"""Plot comparison of binary settings (data aug, sequential update, reset loss).

This script filters for largest batch size and num_iters=32, then compares
different combinations of binary settings across all epochs, metrics, and datasets.

Usage:
    python scripts/plot_binary_settings.py
    python scripts/plot_binary_settings.py --results-dir scripts/results
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
# Data Loading (reuse from plot_sweep_results.py)
# --------------------------------------------------------------------------

def load_run_data(run_dir: Path) -> tuple[dict, pd.DataFrame | None]:
    """Load config and metrics from a single run directory."""
    config_path = run_dir / "config.json"
    
    if not config_path.exists():
        return {}, None
    
    with open(config_path) as f:
        config = json.load(f)
    
    parquet_files = list(run_dir.glob("*.parquet"))
    if not parquet_files:
        return config, None
    
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
    """Load all run results from the results directory."""
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
        
        base_record = {
            "run_id": run_id,
            "dataset": config.get("dataset", "unknown"),
            "batch_size": config.get("batch_size", 0),
            "num_update_iters": config.get("num_vb_iters", config.get("num_update_iters", 16)),
            "nodataaug": config.get("nodataaug", False),
            "seed": config.get("seed", 0),
            "sequential_update": config.get("sequential_update", False),
            "reset_loss_per_epoch": config.get("reset_loss_per_epoch", False),
        }
        
        for _, row in metrics_df.iterrows():
            epoch = int(row.get("epoch", row.name + 1))
            record = base_record.copy()
            record.update({
                "epoch": epoch,
                "acc": row.get("acc", np.nan),
                "ece": row.get("ece", np.nan),
                "nll": row.get("nll", np.nan),
            })
            records.append(record)
    
    print(f"  Loaded {len(records)} records from {loaded_runs} runs")
    return pd.DataFrame(records)


def load_reference_data(
    csv_path: str = "scripts/results_ibprobit_large_batch.csv",
    n_samples: int = 16,
    seed: int = 42,
) -> pd.DataFrame:
    """Load baseline and linear probing data from CSV."""
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    
    print(f"Loading reference data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    df_filtered = df[
        (df["num_blocks"] == 12) & 
        (df["embed_dim"] == 1024)
    ].copy()
    
    if df_filtered.empty:
        return pd.DataFrame()
    
    np.random.seed(seed)
    records = []
    
    for dataset in df_filtered["dataset"].unique():
        df_ds = df_filtered[df_filtered["dataset"] == dataset]
        
        # Baseline
        df_bl = df_ds[df_ds["type"] == "baseline"]
        if not df_bl.empty:
            bl_row = df_bl.iloc[0]
            records.append({
                "dataset": dataset,
                "acc": bl_row["acc"],
                "ece": bl_row["ece"],
                "nll": bl_row["nll"],
                "type": "baseline",
            })
        
        # Linear probing
        df_lp = df_ds[df_ds["type"] == "ibprobit"]
        if not df_lp.empty:
            n_to_sample = min(n_samples, len(df_lp))
            sampled = df_lp.sample(n=n_to_sample, random_state=seed)
            records.append({
                "dataset": dataset,
                "acc": sampled["acc"].mean(),
                "ece": sampled["ece"].mean(),
                "nll": sampled["nll"].mean(),
                "type": "linear_probing",
            })
    
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

METRICS = ["acc", "ece", "nll"]
METRIC_LABELS = {"acc": "Accuracy", "ece": "ECE", "nll": "NLL"}


def make_binary_settings_figure(
    df: pd.DataFrame,
    output_dir: str,
    num_iters: int = 32,
    batch_size: int = 1024,
):
    """Create figure comparing binary settings across epochs.
    
    Layout: 3 rows (metrics) × 2 columns (datasets)
    Each subplot: x-axis = epoch, lines = different binary setting combinations
    """
    # Filter for largest batch size and specified num_iters
    df_filtered = df[
        (df["batch_size"] == batch_size) & 
        (df["num_update_iters"] == num_iters)
    ].copy()
    
    print(f"Filtered to batch_size={batch_size}, num_update_iters={num_iters}")
    print(f"  {len(df_filtered)} records remaining")
    
    df_bl = df[df["type"] == "baseline"] if "type" in df.columns else pd.DataFrame()
    df_lp = df[df["type"] == "linear_probing"] if "type" in df.columns else pd.DataFrame()
    
    datasets = ["cifar10", "cifar100"]
    
    # Define binary setting combinations
    # (nodataaug, sequential_update, reset_loss_per_epoch)
    settings = [
        (False, False, False),
        (False, False, True),
        (False, True, False),
        (False, True, True),
        (True, False, False),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    ]
    
    def setting_label(nodataaug, seq_update, reset_loss):
        parts = []
        parts.append("NoAug" if nodataaug else "Aug")
        parts.append("Seq" if seq_update else "NoSeq")
        parts.append("Reset" if reset_loss else "NoReset")
        return "/".join(parts)
    
    # Colors for seq/reset combinations (4 colors), linestyles for aug/noaug
    # (sequential_update, reset_loss_per_epoch) -> color
    seq_reset_combos = [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ]
    cmap = plt.cm.tab10
    combo_colors = {combo: cmap(i) for i, combo in enumerate(seq_reset_combos)}
    
    # Linestyles: solid for Aug (nodataaug=False), dashed for NoAug (nodataaug=True)
    linestyles = {False: "-", True: "--"}  # nodataaug -> linestyle
    
    # Figure: 3 rows (metrics) × 2 columns (datasets)
    fig, axes = plt.subplots(
        len(METRICS), len(datasets),
        figsize=(12, 10),
        squeeze=False,
        sharex=True,
    )
    
    for m_idx, metric in enumerate(METRICS):
        for ds_idx, dataset in enumerate(datasets):
            ax = axes[m_idx, ds_idx]
            
            df_ds = df_filtered[df_filtered["dataset"] == dataset]
            
            # Plot baseline
            df_bl_ds = df_bl[df_bl["dataset"] == dataset] if not df_bl.empty else pd.DataFrame()
            if not df_bl_ds.empty:
                bl_val = df_bl_ds.iloc[0][metric]
                ax.axhline(bl_val, color="grey", ls=":", lw=1.5, label="baseline", zorder=1)
            
            # Plot linear probing
            df_lp_ds = df_lp[df_lp["dataset"] == dataset] if not df_lp.empty else pd.DataFrame()
            if not df_lp_ds.empty:
                lp_val = df_lp_ds.iloc[0][metric]
                ax.axhline(lp_val, color="orange", ls=":", lw=2, label="linear probing", zorder=1)
            
            # Plot each setting combination
            for setting in settings:
                nodataaug, seq_update, reset_loss = setting
                df_setting = df_ds[
                    (df_ds["nodataaug"] == nodataaug) &
                    (df_ds["sequential_update"] == seq_update) &
                    (df_ds["reset_loss_per_epoch"] == reset_loss)
                ]
                
                if df_setting.empty:
                    continue
                
                # Aggregate over seeds per epoch
                stats = df_setting.groupby("epoch")[metric].agg(["mean", "std"])
                
                if stats.empty:
                    continue
                
                x = np.array(stats.index)
                y = stats["mean"].values
                yerr = stats["std"].values
                
                color = combo_colors[(seq_update, reset_loss)]
                ls = linestyles[nodataaug]
                label = setting_label(*setting)
                ax.plot(x, y, marker="o", ms=4, color=color, ls=ls, lw=1.5, label=label, zorder=2)
                ax.fill_between(x, y - yerr, y + yerr, alpha=0.1, color=color)
            
            # Formatting
            ax.set_xlabel("Epoch" if m_idx == len(METRICS) - 1 else "")
            ax.set_ylabel(METRIC_LABELS[metric] if ds_idx == 0 else "")
            
            if m_idx == 0:
                ax.set_title(dataset, fontsize=12, fontweight="bold")
    
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
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(1.01, 0.5),
    )
    
    fig.suptitle(
        f"Binary Settings Comparison (batch_size={batch_size}, num_iters={num_iters})",
        fontsize=14,
        y=1.005,
    )
    fig.tight_layout()
    
    fname = f"{output_dir}/binary_settings_comparison.pdf"
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
        help="Directory containing run subdirectories",
    )
    parser.add_argument(
        "--reference-csv",
        type=str,
        default="scripts/results_ibprobit_large_batch.csv",
        help="Path to CSV with baseline and linear probing results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scripts/figures",
        help="Directory for output figures",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=32,
        help="Filter for specific num_update_iters value",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Filter for specific batch_size value",
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    df = load_all_results(results_dir)
    
    # Load reference data
    df_ref = load_reference_data(args.reference_csv, n_samples=16)
    if not df_ref.empty:
        df = pd.concat([df, df_ref], ignore_index=True)
    
    # Create plot
    make_binary_settings_figure(df, args.output_dir, num_iters=args.num_iters, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
