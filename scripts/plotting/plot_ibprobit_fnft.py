#!/usr/bin/env python3
"""Plot comparison of binary settings (data aug, sequential update, reset loss).

This script filters for largest batch size and num_iters=32, then compares
different combinations of binary settings across all epochs, metrics, and datasets.

Usage:
    python scripts/plot_binary_settings.py --input scripts/sweep_results.csv
"""

import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------------------

def load_reference_data(
    csv_path: str = "scripts/results_ibprobit_last_layer.csv",
    n_samples: int = 25,
    seed: int = 137,
    pretrained_source: str = "in21k",
    batch_size: int = 16384,
    update_iters: int = 64,
) -> pd.DataFrame:
    """Load baseline and linear probing data from CSV."""
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    print(f"Loading reference data from {csv_path}...")
    df = pd.read_csv(csv_path)

    df_filtered = df[
        (df["num_blocks"] == 12) &
        (df["embed_dim"] == 1024) &
        (df["pretrained_source"] == pretrained_source) &
        (df["data_aug"] == False) &
        (df["batch_size"] == batch_size) &
        (df["update_iters"] == update_iters)
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

    Layout: 2 rows (cifar10/cifar100) × 3 columns (Accuracy, ECE, NLL)
    Each subplot: x-axis = epoch, lines = different binary setting combinations
    """
    df_filtered = df[
        (df["batch_size"] == batch_size) &
        (df["num_update_iters"] == num_iters)
    ].copy()

    print(f"Filtered to batch_size={batch_size}, num_update_iters={num_iters}")
    print(f"  {len(df_filtered)} records remaining")

    df_bl = df[df["type"] == "baseline"] if "type" in df.columns else pd.DataFrame()
    df_lp = df[df["type"] == "linear_probing"] if "type" in df.columns else pd.DataFrame()

    datasets = ["cifar10", "cifar100"]
    dataset_labels = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}

    # Only non-sequential, no-reset variants; vary data augmentation
    settings = [(False, False, False), (True, False, False)]
    setting_colors = {False: "steelblue", True: "tomato"}
    setting_labels = {False: "DA", True: "No-DA"}

    # Layout: 2 rows (datasets) × 3 columns (metrics)
    fig, axes = plt.subplots(
        len(datasets), len(METRICS),
        figsize=(12, 5),
        squeeze=False,
        sharex=True,
    )

    legend_handles, legend_labels = [], []

    for ds_idx, dataset in enumerate(datasets):
        df_ds = df_filtered[df_filtered["dataset"] == dataset]

        for m_idx, metric in enumerate(METRICS):
            ax = axes[ds_idx, m_idx]

            df_bl_ds = df_bl[df_bl["dataset"] == dataset] if not df_bl.empty else pd.DataFrame()
            if not df_bl_ds.empty:
                bl_val = df_bl_ds.iloc[0][metric]
                ax.axhline(bl_val, color="grey", ls=":", lw=1.5, label="baseline", zorder=1)

            df_lp_ds = df_lp[df_lp["dataset"] == dataset] if not df_lp.empty else pd.DataFrame()
            if not df_lp_ds.empty:
                lp_val = df_lp_ds.iloc[0][metric]
                ax.axhline(lp_val, color="orange", ls=":", lw=2, label="linear-probing", zorder=1)

            for nodataaug, seq_update, reset_loss in settings:
                df_setting = df_ds[
                    (df_ds["nodataaug"] == nodataaug) &
                    (df_ds["sequential_update"] == seq_update) &
                    (df_ds["reset_loss_per_epoch"] == reset_loss)
                ]
                if df_setting.empty:
                    continue

                stats = df_setting.groupby("epoch")[metric].agg(["mean", "std"])
                if stats.empty:
                    continue

                x = np.array(stats.index)
                y = stats["mean"].values
                yerr = stats["std"].values
                color = setting_colors[nodataaug]
                lbl = setting_labels[nodataaug]
                ax.plot(x, y, color=color, lw=1.5, label=lbl, zorder=2)
                ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=color)

            # Collect legend entries once (from first subplot)
            if ds_idx == 0 and m_idx == 0:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

            if ds_idx == 0:
                ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight="bold")
            if ds_idx == len(datasets) - 1:
                ax.set_xlabel("Epoch")
            if m_idx == 0:
                ax.set_ylabel(dataset_labels[dataset], fontsize=11)

    # Place legend in the first subplot (cifar10 / Accuracy), bottom-right
    axes[0, 0].legend(
        legend_handles, legend_labels,
        loc="lower right",
        fontsize=9,
        frameon=True,
    )

    fig.suptitle(
        "IBProbit full-network fine-tuning — DA vs No-DA",
        fontsize=13,
    )
    fig.tight_layout()

    fname = f"{output_dir}/full_network_finetuning.pdf"
    fig.savefig(fname, bbox_inches="tight", dpi=300)
    print(f"Saved {fname}")

    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, required=True, help="Path to CSV produced by plot_sweep_results.py")
    parser.add_argument(
        "--reference-csv",
        type=str,
        default="scripts/results/last_layer/results_ibprobit_last_layer.csv",
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
        default=16384,
        help="Filter for specific batch_size value",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    _bool = lambda x: str(x).lower() == "true"
    _bool_cols = {"nodataaug": _bool, "sequential_update": _bool, "reset_loss_per_epoch": _bool}
    df = pd.read_csv(args.input, converters=_bool_cols)
    df_ref = load_reference_data(args.reference_csv, n_samples=16)
    if not df_ref.empty:
        df = pd.concat([df, df_ref], ignore_index=True)

    make_binary_settings_figure(df, args.output_dir, num_iters=args.num_iters, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
