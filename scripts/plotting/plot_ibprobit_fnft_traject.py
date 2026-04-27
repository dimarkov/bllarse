#!/usr/bin/env python3
"""Plot IBProbit full-network fine-tuning trajectories.

Fetches child runs from an MLflow parent (or loads a previously cached CSV)
and produces publication-ready figures showing acc / ECE / NLL vs batch size
at selected epochs, with baseline and linear-probing references.

Usage:
    python scripts/plotting/plot_ibprobit_fnft_traject.py --parent-run-name <name>
    python scripts/plotting/plot_ibprobit_fnft_traject.py --input <cached.csv>
"""

import os
import sys
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# --------------------------------------------------------------------------
# MLflow Data Fetching
# --------------------------------------------------------------------------

def fetch_mlflow_runs(experiment_name: str, parent_run_name: str) -> pd.DataFrame:
    """Fetch child runs of the named parent from MLflow and return per-epoch records."""
    import mlflow
    from mlflow.tracking import MlflowClient
    from dotenv import load_dotenv

    load_dotenv()
    load_dotenv(".env.secrets", override=False)
    experiment = mlflow.set_experiment(experiment_name)
    experiment_id = experiment.experiment_id
    client = MlflowClient()

    parents = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"run_name = '{parent_run_name}'",
    )
    if not parents:
        print(f"ERROR: parent run '{parent_run_name}' not found in experiment '{experiment_name}' (id={experiment_id})", file=sys.stderr)
        sys.exit(1)

    parent_id = parents[0].info.run_id
    print(f"Parent run: {parent_id} ({parent_run_name})")

    all_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_id}' and attributes.status = 'FINISHED'",
        max_results=5000,
    )
    total_runs = len(all_runs)
    print(f"Found {total_runs} FINISHED child runs, fetching metric histories...")
    sys.stdout.flush()

    records = []
    for i, run in enumerate(all_runs):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing run {i + 1}/{total_runs}: {run.info.run_name}")
            sys.stdout.flush()

        p = run.data.params
        run_id = run.info.run_id

        base_record = {
            "run_id": run_id,
            "run_name": run.info.run_name,
            "dataset": p.get("dataset", "unknown"),
            "batch_size": int(p.get("batch_size", 0)),
            "num_update_iters": int(p.get("num_vb_iters", p.get("num_update_iters", 16))),
            "nodataaug": p.get("nodataaug", "False").lower() == "true",
            "optimizer": p.get("optimizer", "adamw"),
            "learning_rate": float(p.get("learning_rate", 0) or 0),
            "weight_decay": float(p.get("weight_decay", 0) or 0),
            "seed": int(p.get("seed", 0)),
            "sequential_update": p.get("sequential_update", "False").lower() == "true",
            "reset_loss_per_epoch": p.get("reset_loss_per_epoch", "False").lower() == "true",
        }

        metric_map = {
            "acc": ["acc", "test_acc"],
            "ece": ["ece", "test_ece"],
            "nll": ["nll", "test_nll"],
            "loss": ["loss", "train_loss"],
        }

        epoch_data: dict[int, dict] = {}
        for col, candidates in metric_map.items():
            history = []
            for metric_name in candidates:
                try:
                    history = client.get_metric_history(run_id, metric_name)
                    if history:
                        break
                except Exception:
                    continue
            for point in history:
                epoch = point.step
                if epoch not in epoch_data:
                    epoch_data[epoch] = {}
                epoch_data[epoch][col] = point.value

        for epoch, vals in epoch_data.items():
            record = base_record.copy()
            record.update({
                "epoch": epoch,
                "acc": vals.get("acc", np.nan),
                "ece": vals.get("ece", np.nan),
                "nll": vals.get("nll", np.nan),
                "loss": vals.get("loss", np.nan),
                "type": "experiment",
            })
            records.append(record)

    print(f"Done fetching {len(records)} records from {total_runs} runs")
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Baseline and Linear Probing Data Loading
# --------------------------------------------------------------------------

def load_reference_data(
    csv_path: str = "scripts/results_ibprobit_last_layer.csv",
    n_samples: int = 25,
    seed: int = 137,
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
            
            # For each num_iters: subsample n_samples runs, compute mean of each metric
            # Then pick the num_iters with the lowest mean NLL and report all three
            # metrics for that num_iters.
            best_mean_nll = np.inf
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

                if mean_nll < best_mean_nll:
                    best_mean_nll = mean_nll
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
EPOCHS_TO_PLOT = [1, 5]


def make_figure(
    df: pd.DataFrame,
    output_dir: str,
    sequential_update: bool | None = None,
    reset_loss_per_epoch: bool | None = None,
    label: str = "sweep_results",
):
    """Create separate 3x6 figures for with/without data augmentation.
    
    Layout per figure:
    - Rows: acc, ece, nll (3 metrics)
    - Columns 1-3: cifar10 epochs 1, 5,
    - Columns 4-6: cifar100 epochs 1, 5
    
    Y-axis is aligned within each dataset's columns (first 3 cols share y, last 3 cols share y).
    """
    df_exp = df[df["type"] == "experiment"].copy()
    df_bl = df[df["type"] == "baseline"]
    df_lp = df[df["type"] == "linear_probing"]
    
    # Apply filters
    filter_desc = []
    if sequential_update is not None:
        df_exp = df_exp[df_exp["sequential_update"] == sequential_update]
        filter_desc.append(f"sequential-update={sequential_update}")
    if reset_loss_per_epoch is not None:
        df_exp = df_exp[df_exp["reset_loss_per_epoch"] == reset_loss_per_epoch]
        filter_desc.append(f"reset-loss={reset_loss_per_epoch}")
    
    filter_suffix = "_" + "_".join(filter_desc) if filter_desc else ""
    filter_title = " (" + ", ".join(filter_desc) + ")" if filter_desc else ""
    
    datasets = ["cifar10", "cifar100"]
    dataset_labels = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}
    
    # Get unique values for coloring
    update_iters_list = sorted(df_exp["num_update_iters"].dropna().unique())
    batch_sizes = sorted(df_exp["batch_size"].dropna().unique())
    
    # Color map for num_update_iters - use tab10 for divergent/distinct colors
    cmap = plt.cm.RdYlBu
    colors = {
        it: cmap(i / max(len(update_iters_list) - 1, 1))
        for i, it in enumerate(update_iters_list)
    }
    
    # Create separate figures for each augmentation setting
    for nodataaug in [True, False]:
        aug_label = "without DA" if nodataaug else "with DA"
        aug_suffix = "_no_aug" if nodataaug else "_with_aug"
        
        df_aug = df_exp[df_exp["nodataaug"] == nodataaug]
        
        # Figure: 3 rows (metrics) × 6 columns (3 epochs × 2 datasets)
        n_rows = len(METRICS)  # 3
        n_cols = len(EPOCHS_TO_PLOT) * len(datasets)  # 6
        
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(12, 8),
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
                        ax.set_title(f"{dataset_labels[dataset]}\nEpoch {epoch}", fontsize=10, fontweight="bold")
                    
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
            f"Full network fine-tuning with IBProbit last layer {aug_label}",
            fontsize=14,
            y=1.005,
        )
        fig.tight_layout()
        
        fname = f"{output_dir}/{label}{aug_suffix}{filter_suffix}.pdf"
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        print(f"Saved {fname}")
        
        plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-name", type=str, default="bllarse", help="MLflow experiment name")
    parser.add_argument("--parent-run-name", type=str, default=None, help="MLflow parent run name to fetch child runs from")
    parser.add_argument("--input", type=str, default=None, help="Path to existing CSV (skip MLflow fetch if provided)")
    parser.add_argument("--output", type=str, default="scripts/results/full_network/ibprobit_deepmlp_results.csv", help="Output CSV path")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scripts/figures",
        help="Directory for output figures",
    )
    parser.add_argument(
        "--reference-csv",
        type=str,
        default="scripts/results/last_layer/results_ibprobit_last_layer.csv",
        help="Path to CSV with baseline and linear probing results",
    )
    parser.add_argument("--label", type=str, default="sweep_results", help="Prefix for output figure filenames")
    parser.add_argument("--csv-only", action="store_true", help="Only fetch/save CSV, skip plotting")
    parser.add_argument("--plot-only", action="store_true", help="Only plot from existing CSV (requires --input)")
    parser.add_argument(
        "--sequential-update",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Filter plots by sequential_update value (true/false)",
    )
    parser.add_argument(
        "--reset-loss-per-epoch",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Filter plots by reset_loss_per_epoch value (true/false)",
    )
    args = parser.parse_args()

    seq_update = None if args.sequential_update is None else (args.sequential_update == "true")
    reset_loss = None if args.reset_loss_per_epoch is None else (args.reset_loss_per_epoch == "true")

    os.makedirs(args.output_dir, exist_ok=True)

    _bool = lambda x: str(x).lower() == "true"
    _bool_cols = {"nodataaug": _bool, "sequential_update": _bool, "reset_loss_per_epoch": _bool}

    if args.plot_only:
        if not args.input:
            parser.error("--plot-only requires --input")
        df = pd.read_csv(args.input, converters=_bool_cols)
    elif args.input and os.path.exists(args.input):
        print(f"Loading existing data from {args.input}")
        df = pd.read_csv(args.input, converters=_bool_cols)
    elif args.parent_run_name:
        print(f"Fetching runs from MLflow: {args.experiment_name} / {args.parent_run_name}")
        df = fetch_mlflow_runs(args.experiment_name, args.parent_run_name)
        df.to_csv(args.output, index=False)
        print(f"Saved {len(df)} records to {args.output}")
    else:
        parser.error("Provide one of: --parent-run-name or --input")

    # Always merge reference rows (drop any stale ones first so --input works too)
    if "type" in df.columns:
        df = df[~df["type"].isin(["baseline", "linear_probing"])]
    df_ref = load_reference_data(args.reference_csv, n_samples=16)
    if not df_ref.empty:
        df = pd.concat([df, df_ref], ignore_index=True)

    if not args.csv_only:
        make_figure(df, args.output_dir, sequential_update=seq_update, reset_loss_per_epoch=reset_loss, label=args.label)


if __name__ == "__main__":
    main()
