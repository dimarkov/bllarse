"""Query an MLflow sweep for the best hyperparameters of a (model, dataset).

Selection metric is the same model-selection NLL used by each sweep:
  - CrossEntropy: mean ``best_test_nll`` (val-NLL-selected, early stopped)
  - IBProbit:     mean ``test_nll``      (single-pass CAVI)
averaged across seeds, lowest wins.

Usage (shell-consumable KEY=VALUE lines):
    eval "$(uv run get_best_hparams.py \
        --parent-run-name sweep5a-alt-classification-dinov3-IBProbit \
        --model dinov3_huge --dataset imagenet1k --method IBProbit)"
    # -> BATCH_SIZE=16384
    #    NUM_UPDATE_ITERS=256
"""

import argparse
import logging
import sys

from dotenv import load_dotenv


SELECTION = {
    "CrossEntropy": (("batch_size", "lr"), "best_test_nll"),
    "IBProbit": (("batch_size", "num_update_iters"), "test_nll"),
}
INT_KEYS = {"batch_size", "num_update_iters"}


def best_hparams(parent_run_name, model, dataset, method, experiment_name="bllarse"):
    """Return (hparams dict, selection_nll) for the best combo, or raise."""
    import mlflow
    import pandas as pd

    if method not in SELECTION:
        raise ValueError(f"Unknown method: {method}. Choose from {list(SELECTION)}")
    keys, sel_metric = SELECTION[method]

    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment not found: {experiment_name}")

    parents = client.search_runs(
        [exp.experiment_id],
        filter_string=f"run_name = '{parent_run_name}'",
        max_results=5,
    )
    if not parents:
        raise ValueError(f"Parent run not found: {parent_run_name}")
    parent_id = parents[0].info.run_id

    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_id}' AND status = 'FINISHED'",
        max_results=50000,
    )
    df = pd.DataFrame([{**r.data.params, **r.data.metrics} for r in runs])
    if df.empty:
        raise ValueError(f"No finished child runs under {parent_run_name}")

    df = df[(df["model"] == model) & (df["dataset"] == dataset)].copy()
    if df.empty or sel_metric not in df.columns:
        raise ValueError(f"No usable runs for {model}/{dataset} under {parent_run_name}")

    for k in keys:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    df[sel_metric] = pd.to_numeric(df[sel_metric], errors="coerce")
    df = df.dropna(subset=list(keys) + [sel_metric])

    grouped = df.groupby(list(keys))[sel_metric].mean().reset_index().sort_values(sel_metric)
    if grouped.empty:
        raise ValueError(f"No usable runs for {model}/{dataset} under {parent_run_name}")
    best = grouped.iloc[0]

    hparams = {k: (int(best[k]) if k in INT_KEYS else float(best[k])) for k in keys}
    return hparams, float(best[sel_metric])


def main():
    parser = argparse.ArgumentParser(description="Print best sweep hyperparameters as KEY=VALUE")
    parser.add_argument("--parent-run-name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True, choices=list(SELECTION))
    parser.add_argument("--experiment-name", default="bllarse")
    args = parser.parse_args()

    load_dotenv()
    load_dotenv(".env.secrets", override=False)
    logging.getLogger("mlflow").setLevel(logging.WARNING)

    hparams, sel = best_hparams(
        args.parent_run_name, args.model, args.dataset, args.method, args.experiment_name
    )
    # Shell-consumable: BATCH_SIZE=..., LR=..., NUM_UPDATE_ITERS=...
    for k, v in hparams.items():
        print(f"{k.upper()}={v}")
    print(f"# selection_nll={sel:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
