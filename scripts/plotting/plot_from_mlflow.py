#!/usr/bin/env python3
"""Build the figures from the MLflow runs alone -- no dump tree, no JSON.

    python scripts/plotting/plot_from_mlflow.py --figure fig1

Requires `scripts/analysis/sync_mlflow.py --apply` to have run, which is what
puts the scored numbers on the runs in the first place.

The figures themselves are NOT redefined here. This assembles rows out of the
tracking server and hands them to the same `summarize` and the same plotting
functions the JSON path uses, so the two routes cannot drift into disagreeing
about what Fig 1 is -- which would be worse than either being wrong, because
nothing would look broken.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bllarse import mlflow_sync
from bllarse.mlflow_utils import load_mlflow_env_defaults, resolve_experiment_id

import fig1_curves  # noqa: E402
import fig2_detection  # noqa: E402
import plot_fig1  # noqa: E402
import plot_fig2_draft  # noqa: E402


def _fig1(rows, out_dir):
    summary = fig1_curves.summarize(rows)
    for ext in ("pdf", "png"):
        plot_fig1.make_figure(summary, os.path.join(out_dir, f"fig1.{ext}"))
        plot_fig1.make_overfit_figure(
            summary, os.path.join(out_dir, f"fig1_endpoint_penalty.{ext}")
        )
    return summary


def _fig2(rows, out_dir):
    summary = fig2_detection.summarize(rows)
    id_summary = fig2_detection.summarize_id(rows)
    for ext in ("pdf", "png"):
        plot_fig2_draft.make_figure(summary, os.path.join(out_dir, f"fig2_draft.{ext}"))
        if id_summary:
            plot_fig2_draft.make_id_figure(
                id_summary, os.path.join(out_dir, f"fig2_id_control.{ext}")
            )
    return summary


FIGURES = {
    "fig1": (fig1_curves, _fig1),
    "fig2": (fig2_detection, _fig2),
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figure", choices=sorted(FIGURES), required=True)
    p.add_argument("--experiment", default="bllarse-probing")
    p.add_argument("--out-dir", default="scripts/figures")
    args = p.parse_args()

    module, build = FIGURES[args.figure]

    load_mlflow_env_defaults()
    import mlflow
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise SystemExit("set MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(uri)
    experiment_id = resolve_experiment_id(mlflow, args.experiment)

    rows = list(mlflow_sync.fetch_rows(
        mlflow.MlflowClient(), experiment_id, module.GROUPS, module.UID_RE
    ))
    if not rows:
        # Never an empty figure: a blank panel grid looks like a result.
        raise SystemExit(
            f"no scored runs found for {args.figure} in {args.experiment!r}. "
            f"Run scripts/analysis/sync_mlflow.py --figure {args.figure} "
            f"--apply first -- unsynced runs carry no scored metrics."
        )

    os.makedirs(args.out_dir, exist_ok=True)
    summary = build(rows, args.out_dir)
    print(f"\nbuilt {args.figure} from {len(rows)} MLflow runs "
          f"({len(summary)} cells); no dump tree touched")


if __name__ == "__main__":
    main()
