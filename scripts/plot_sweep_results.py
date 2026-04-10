#!/usr/bin/env python3
"""Fetch MLflow sweep results and compute baselines, then plot metrics.

This script:
1. Fetches run data from MLflow for the specified parent run
2. Computes baseline metrics by loading in21k_cifar pretrained models
3. Saves all data to a CSV file
4. Creates publication-ready figures showing acc, ece, nll vs batch_size/num_iters

Usage:
    python scripts/plot_sweep_results.py --parent-run-name sweep7a1_fnf_dataaug_adamw_batchsize_numiters
    python scripts/plot_sweep_results.py --csv-only  # Only fetch/save CSV, skip plotting
    python scripts/plot_sweep_results.py --plot-only --input results.csv  # Only plot from existing CSV
"""

import os
import sys
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
# MLflow Data Fetching
# --------------------------------------------------------------------------

def fetch_mlflow_runs(experiment_name: str, parent_run_name: str) -> pd.DataFrame:
    """Fetch child runs of the named parent from MLflow and return per-epoch records."""
    import mlflow
    from mlflow.tracking import MlflowClient
    from dotenv import load_dotenv

    load_dotenv()
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    # Find parent run
    parents = mlflow.search_runs(
        filter_string=f"run_name = '{parent_run_name}'",
        output_format="list",
    )
    if not parents:
        print(f"ERROR: parent run '{parent_run_name}' not found in experiment '{experiment_name}'", file=sys.stderr)
        sys.exit(1)

    parent_id = parents[0].info.run_id
    print(f"Parent run: {parent_id} ({parent_run_name})")

    # Fetch all FINISHED child runs
    all_runs = mlflow.search_runs(
        filter_string=f"tags.mlflow.parentRunId = '{parent_id}' and attributes.status = 'FINISHED'",
        output_format="list",
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

        # Fetch per-epoch metric history; support both finetuning.py and vit_classification.py names
        metric_map = {
            "acc": ["acc", "test_acc"],
            "ece": ["ece", "test_ece"],
            "nll": ["nll", "test_nll"],
            "loss": ["loss", "train_loss"],
        }

        # Build epoch -> values dict
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
        "--experiment-name",
        type=str,
        default="bllarse",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--parent-run-name",
        type=str,
        default="sweep7a1_fnf_dataaug_adamw_batchsize_numiters",
        help="MLflow parent run name",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to existing CSV (skip MLflow fetch if provided)",
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
        # Fetch MLflow data
        if args.input and os.path.exists(args.input):
            print(f"Loading existing data from {args.input}")
            df = pd.read_csv(args.input)
        else:
            print(f"Fetching runs from MLflow: {args.experiment_name} / {args.parent_run_name}")
            df = fetch_mlflow_runs(args.experiment_name, args.parent_run_name)
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
