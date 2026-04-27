#!/usr/bin/env python3
"""Compare VBLL (diagonal) and IBProbit full-network fine-tuning.

Pulls VBLL-diagonal child runs from MLflow (parent
``sweep9a1_vbll_discriminative_diagonal_fnf_llf_adamw_bs_lr``) and loads
IBProbit full-network results from the local sweep CSV, then produces a
single figure with rows=datasets (CIFAR-10, CIFAR-100) and cols=metrics
(acc, ECE, NLL). For each (method, dataset, batch_size), the epoch shown
is chosen as ``argmin`` of mean-over-seeds NLL (or ``argmax`` accuracy
via ``--selection-criterion acc``) — a post-hoc early-stopping proxy.
Both methods are shown as two lines with shaded +/- 1 sigma bands over
seeds; the selected epoch is annotated next to each marker.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

VBLL_PARENT_RUN = "sweep9a1_vbll_discriminative_diagonal_fnf_llf_adamw_bs_lr"

METRICS = ["acc", "ece", "nll"]
METRIC_LABELS = {"acc": "Accuracy", "ece": "ECE", "nll": "NLL"}
DATASETS = ["cifar10", "cifar100"]
DATASET_LABELS = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}

METHODS = ["VBLL-diag", "IBProbit"]
METHOD_COLORS = {"VBLL-diag": "steelblue", "IBProbit": "tomato"}


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------

def fetch_vbll_diagonal(experiment_name: str) -> pd.DataFrame:
    """Fetch VBLL-diagonal child runs from MLflow as per-epoch records."""
    import mlflow
    from mlflow.tracking import MlflowClient
    from dotenv import load_dotenv

    load_dotenv()
    load_dotenv(".env.secrets", override=False)

    experiment = mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    parents = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"run_name = '{VBLL_PARENT_RUN}'",
    )
    if not parents:
        print(f"ERROR: parent run '{VBLL_PARENT_RUN}' not found", file=sys.stderr)
        return pd.DataFrame()

    parent_id = parents[0].info.run_id
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.mlflow.parentRunId = '{parent_id}' "
            f"and attributes.status = 'FINISHED'"
        ),
        max_results=5000,
    )
    print(f"{VBLL_PARENT_RUN}: {len(runs)} FINISHED child runs")
    sys.stdout.flush()

    records = []
    for i, run in enumerate(runs):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(runs)}")
            sys.stdout.flush()

        p = run.data.params
        run_id = run.info.run_id

        base = {
            "run_id": run_id,
            "dataset": p.get("dataset", "unknown"),
            "batch_size": int(p.get("batch_size", 0)),
            "learning_rate": float(p.get("learning_rate", 0) or 0),
            "nodataaug": p.get("nodataaug", "False").lower() == "true",
            "parameterization": p.get("parameterization", "unknown"),
            "tune_mode": p.get("tune_mode", "unknown"),
            "pretrained": p.get("pretrained", "unknown"),
            "seed": int(p.get("seed", 0)),
        }

        epoch_data: dict[int, dict] = {}
        for col, mlflow_name in [
            ("acc", "test_accuracy"),
            ("ece", "test_ece"),
            ("nll", "test_nll"),
        ]:
            try:
                history = client.get_metric_history(run_id, mlflow_name)
            except Exception:
                continue
            for pt in history:
                epoch_data.setdefault(pt.step, {})[col] = pt.value

        for epoch, vals in epoch_data.items():
            rec = base.copy()
            rec.update({
                "epoch": int(epoch),
                "acc": vals.get("acc", np.nan),
                "ece": vals.get("ece", np.nan),
                "nll": vals.get("nll", np.nan),
            })
            records.append(rec)

    df = pd.DataFrame(records)
    print(f"Fetched {len(df)} VBLL records")
    return df


def load_vbll_diagonal(
    experiment_name: str,
    cache_csv: str | None,
    use_cache: bool,
) -> pd.DataFrame:
    """Load VBLL-diagonal runs, optionally from a cached CSV."""
    if use_cache and cache_csv and os.path.exists(cache_csv):
        print(f"Loading VBLL from cache: {cache_csv}")
        df = pd.read_csv(cache_csv)
        df["nodataaug"] = df["nodataaug"].map(lambda x: str(x).lower() == "true")
    else:
        df = fetch_vbll_diagonal(experiment_name)
        if cache_csv and not df.empty:
            os.makedirs(os.path.dirname(cache_csv) or ".", exist_ok=True)
            df.to_csv(cache_csv, index=False)
            print(f"Cached VBLL CSV -> {cache_csv}")

    # Filter to the conditions we compare against IBProbit
    df = df[
        (df["parameterization"] == "diagonal") &
        (df["pretrained"] == "in21k") &
        (df["tune_mode"] == "full_network") &
        (df["learning_rate"] == 1e-4)
    ].copy()
    print(f"  After VBLL filters: {len(df)} records")
    return df[["dataset", "batch_size", "nodataaug", "seed", "epoch", "acc", "ece", "nll"]]


def load_ibprobit_full_network(csv_path: str, update_iters: int) -> pd.DataFrame:
    """Load IBProbit full-network sweep CSV and filter to the comparison config."""
    print(f"Loading IBProbit from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)

    # Coerce bool-ish columns (CSV has mixed types because of a stray header row)
    for col in ["nodataaug", "sequential_update", "reset_loss_per_epoch"]:
        df[col] = df[col].astype(str).str.lower() == "true"

    df = df[
        (df["type"] == "experiment") &
        (df["optimizer"] == "adamw") &
        (df["learning_rate"] == 1e-4) &
        (df["sequential_update"] == False) &
        (df["reset_loss_per_epoch"] == False) &
        (df["num_update_iters"] == update_iters)
    ].copy()

    # Cast numeric ids to int for clean groupby
    for col in ["batch_size", "seed", "epoch"]:
        df[col] = df[col].astype(int)

    print(f"  After IBProbit filters (update_iters={update_iters}): {len(df)} records")
    return df[["dataset", "batch_size", "nodataaug", "seed", "epoch", "acc", "ece", "nll"]]


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def _save_fig(fig, path: str):
    fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"Saved {path}")
    plt.close(fig)


def _method_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(
            by_label.values(), by_label.keys(),
            loc="lower left", fontsize=10, frameon=True,
        )


def _select_best_epoch(df_m: pd.DataFrame, criterion: str) -> dict[int, int]:
    """Pick best epoch per batch_size for one method, via mean-over-seeds criterion.

    Returns {batch_size: best_epoch}. Criterion is 'nll' (argmin) or 'acc' (argmax).
    """
    agg = df_m.groupby(["batch_size", "epoch"])[criterion].mean().reset_index()
    if criterion == "acc":
        idx = agg.groupby("batch_size")[criterion].idxmax()
    else:
        idx = agg.groupby("batch_size")[criterion].idxmin()
    picks = agg.loc[idx, ["batch_size", "epoch"]].values
    return {int(bs): int(ep) for bs, ep in picks}


def make_best_epoch_figure(
    df: pd.DataFrame,
    output_dir: str,
    nodataaug: bool,
    ibprobit_update_iters: int,
    criterion: str,
):
    """Rows=datasets, cols=metrics; x=batch_size (log2); one line per method.

    Per-method, per-(dataset, batch_size), select the epoch that optimizes
    mean-over-seeds ``criterion`` (nll: argmin, acc: argmax) and report all three
    metrics at that selected epoch. Acts as post-hoc early stopping.
    """
    n_rows = len(DATASETS)
    n_cols = len(METRICS)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )

    for row_idx, dataset in enumerate(DATASETS):
        df_ds = df[df["dataset"] == dataset]
        if df_ds.empty:
            print(f"  (no data for {dataset}) skipping row")
            continue

        batch_sizes = sorted(df_ds["batch_size"].unique())

        selections: dict[str, dict[int, int]] = {}
        for method in METHODS:
            df_m = df_ds[df_ds["method"] == method]
            selections[method] = (
                _select_best_epoch(df_m, criterion) if not df_m.empty else {}
            )

        print(f"  [{dataset}] selected epochs by {criterion}:")
        for method in METHODS:
            if selections[method]:
                pairs = ", ".join(
                    f"bs{bs}:ep{ep}" for bs, ep in sorted(selections[method].items())
                )
                print(f"    {method}: {pairs}")

        for m_idx, metric in enumerate(METRICS):
            ax = axes[row_idx, m_idx]

            for method in METHODS:
                sel = selections[method]
                if not sel:
                    continue
                xs, ys, yerrs, eps = [], [], [], []
                for bs in batch_sizes:
                    ep = sel.get(bs)
                    if ep is None:
                        continue
                    df_cell = df_ds[
                        (df_ds["method"] == method)
                        & (df_ds["batch_size"] == bs)
                        & (df_ds["epoch"] == ep)
                    ]
                    if df_cell.empty:
                        continue
                    xs.append(bs)
                    ys.append(df_cell[metric].mean())
                    yerrs.append(df_cell[metric].std())
                    eps.append(ep)
                if not xs:
                    continue
                xs, ys, yerrs = np.array(xs), np.array(ys), np.array(yerrs)
                color = METHOD_COLORS[method]
                ax.plot(xs, ys, marker="o", ms=5, color=color, lw=1.5,
                        label=method, zorder=2)
                ax.fill_between(xs, ys - yerrs, ys + yerrs, alpha=0.2, color=color)
                for x, y, ep in zip(xs, ys, eps):
                    ax.annotate(
                        str(ep), (x, y), textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color=color,
                    )

            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.xaxis.set_major_locator(ticker.FixedLocator(batch_sizes))
            ax.tick_params(axis="x", rotation=45)
            if row_idx == 0:
                ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight="bold")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Batch Size")
            if m_idx == n_cols - 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(
                    DATASET_LABELS[dataset],
                    fontsize=12, fontweight="bold", rotation=270, labelpad=18,
                )

    _method_legend(axes[-1, 0])

    aug_tag = "noaug" if nodataaug else "aug"
    aug_str = "No DA" if nodataaug else "DA"
    crit_label = {"nll": "argmin mean NLL", "acc": "argmax mean accuracy"}[criterion]
    fig.suptitle(
        f"VBLL-diag vs IBProbit",
        fontsize=13,
    )
    fig.tight_layout()
    _save_fig(
        fig,
        f"{output_dir}/vbll_vs_ibprobit_best_epoch_{criterion}_{aug_tag}.pdf",
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--experiment-name", type=str, default="bllarse")
    parser.add_argument(
        "--ibprobit-csv", type=str,
        default="scripts/results/full_network/ibprobit_deepmlp_results.csv",
    )
    parser.add_argument("--ibprobit-update-iters", type=int, default=64)
    parser.add_argument(
        "--vbll-cache", type=str, default="scripts/results/full_network/vbll_deepmlp_results.csv",
        help="CSV to cache VBLL MLflow fetch; reused on subsequent runs",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Re-fetch VBLL from MLflow even if cache exists",
    )
    parser.add_argument("--output-dir", type=str, default="scripts/figures")
    parser.add_argument(
        "--data-aug", action="store_true",
        help="Use nodataaug=False rows (default is nodataaug=True)",
    )
    parser.add_argument(
        "--selection-criterion", choices=["nll", "acc"], default="nll",
        help="Post-hoc epoch selection: 'nll' (argmin mean NLL) or 'acc' (argmax mean accuracy)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    nodataaug = not args.data_aug

    df_vbll = load_vbll_diagonal(
        args.experiment_name, args.vbll_cache, use_cache=not args.refresh,
    )
    df_ib = load_ibprobit_full_network(args.ibprobit_csv, args.ibprobit_update_iters)

    df = pd.concat(
        [df_vbll.assign(method="VBLL-diag"), df_ib.assign(method="IBProbit")],
        ignore_index=True,
    )
    df = df[df["nodataaug"] == nodataaug]
    print(f"After nodataaug={nodataaug} filter: {len(df)} rows "
          f"(VBLL={len(df[df.method=='VBLL-diag'])}, "
          f"IBProbit={len(df[df.method=='IBProbit'])})")

    make_best_epoch_figure(
        df, args.output_dir, nodataaug,
        args.ibprobit_update_iters, args.selection_criterion,
    )


if __name__ == "__main__":
    main()
