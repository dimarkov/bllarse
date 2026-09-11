#!/usr/bin/env python3
"""Push scored results onto the MLflow runs that produced them.

Makes the tracking server self-describing: after this, every child run carries
the metrics the analysis scored it at, the crashed runs carry the verdict that
excluded them, and `plot_from_mlflow.py` can build the figures without the dump
tree being mounted anywhere.

    # once, where the dumps are
    python scripts/analysis/fig1_curves.py --dump-root $BLLARSE_EVAL_DUMP_ROOT --publish

    # then from anywhere -- dry run first, it prints what it would touch
    python scripts/analysis/sync_mlflow.py --figure fig1 --json mlflow:fig1_curves
    python scripts/analysis/sync_mlflow.py --figure fig1 --json mlflow:fig1_curves --apply

Dry by default, because this writes to hundreds of runs on a shared server.

It does NOT reconstruct the per-step metric history, which whole cells of the
campaign are missing. That data was computed transiently during training and
never written to disk -- the dumps hold one evaluation per run. See
bllarse.mlflow_sync.
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bllarse import mlflow_sync
from bllarse.mlflow_utils import (
    load_mlflow_env_defaults, resolve_analysis_path, resolve_experiment_id,
)

import fig1_curves  # noqa: E402
import fig2_detection  # noqa: E402

FIGURES = {"fig1": fig1_curves, "fig2": fig2_detection}


def provenance() -> str:
    """What produced these numbers, recorded on every run touched.

    The campaign was run from a commit that no longer matches the tree (the
    deployed probe.py still had `n_dumps`), which is exactly the confusion this
    stamp exists to prevent for the scored values.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        sha = "unknown"
    return f"sync_mlflow.py@{sha}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figure", choices=sorted(FIGURES), required=True)
    p.add_argument("--json", required=True,
                   help="Analysis JSON: a local path, or mlflow:NAME for a "
                        "published artifact.")
    p.add_argument("--experiment", default="bllarse-probing")
    p.add_argument("--apply", action="store_true",
                   help="Actually write. Without it this only reports.")
    args = p.parse_args()

    module = FIGURES[args.figure]
    with open(resolve_analysis_path(args.json)) as fh:
        rows = json.load(fh)["rows"]

    load_mlflow_env_defaults()
    import mlflow
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise SystemExit("set MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(uri)
    experiment_id = resolve_experiment_id(mlflow, args.experiment)

    log = mlflow_sync.sync_rows(
        mlflow.MlflowClient(), experiment_id, rows,
        group_ids=module.GROUPS,
        provenance=provenance(),
        # Only Fig 1 carries an exclusion list; Fig 2's sweep had no casualties.
        excluded=getattr(module, "INCOMPLETE_RUNS", None),
        dry_run=not args.apply,
    )

    # Counted per category rather than as "everything minus the exceptions":
    # a new outcome added to sync_rows would otherwise be silently absorbed
    # into the scored total, which is the number a reader trusts.
    def _count(prefix):
        return [line for line in log if line.startswith(prefix)]

    scored, skipped = _count("sync"), _count("skip")
    excluded, missing = _count("exclude"), _count("MISSING")
    accounted = len(scored) + len(skipped) + len(excluded) + len(missing)
    assert accounted == len(log), f"unclassified sync outcomes: {len(log) - accounted}"

    print(f"{len(scored)} runs scored, {len(excluded)} marked excluded, "
          f"{len(skipped)} rows skipped as excluded, {len(missing)} unmatched")
    for line in missing + excluded + skipped:
        print(f"  {line}")
    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
    else:
        print(f"\nwritten, stamped {provenance()}")


if __name__ == "__main__":
    main()
