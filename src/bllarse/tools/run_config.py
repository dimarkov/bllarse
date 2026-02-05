#!/usr/bin/env python3
import argparse
import os
from bllarse.tools.module import get_module_from_source_path

def _format_sweep_name(sweep_source: str) -> str:
    base = os.path.basename(sweep_source)
    if base.endswith(".py"):
        base = base[:-3]
    return base

def run_config(sweep_source: str, config_idx: int):
    sweep = get_module_from_source_path(sweep_source)
    all_configs = sweep.create_configs()
    if config_idx < 0 or config_idx >= len(all_configs):
        raise IndexError(f"{config_idx=} out of range (0..{len(all_configs)-1})")
    cfg = all_configs[config_idx]

    # If MLflow is enabled, create (or reuse) a parent run for the sweep and
    # pass the parent run id via env var so finetuning.py creates a nested run.
    enable_mlflow = bool(cfg.get("enable_mlflow", False)) or bool(cfg.get("enable_wandb", False))
    tracking_uri = cfg.get("mlflow_tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI")
    experiment = cfg.get("mlflow_experiment") or os.environ.get("MLFLOW_EXPERIMENT_NAME") or "bllarse"
    group_id = cfg.get("group_id")
    sweep_name = group_id or _format_sweep_name(sweep_source)

    if enable_mlflow:
        import mlflow
        from mlflow.tracking import MlflowClient

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

        parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID")
        if not parent_run_id:
            client = MlflowClient()
            exp = mlflow.get_experiment_by_name(experiment)
            exp_id = exp.experiment_id if exp else None
            if exp_id is not None:
                filter_string = (
                    f"tags.group_id = '{group_id or sweep_name}' "
                    f"and tags.is_parent = 'true' "
                    f"and tags.sweep_source = '{sweep_source}'"
                )
                runs = client.search_runs(
                    experiment_ids=[exp_id],
                    filter_string=filter_string,
                    order_by=["attributes.start_time ASC"],
                    max_results=1,
                )
                if runs:
                    parent_run_id = runs[0].info.run_id

        if not parent_run_id:
            parent_tags = {
                "sweep_source": sweep_source,
                "sweep_name": sweep_name,
                "group_id": group_id or sweep_name,
                "is_parent": "true",
            }
            # Create a parent run once per process; child runs are created in finetuning.py.
            with mlflow.start_run(run_name=sweep_name, tags=parent_tags) as parent:
                os.environ["MLFLOW_PARENT_RUN_ID"] = parent.info.run_id
                sweep.run(cfg)
        else:
            os.environ["MLFLOW_PARENT_RUN_ID"] = parent_run_id
            sweep.run(cfg)
    else:
        sweep.run(cfg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_source", type=str)
    ap.add_argument("config_idx", type=int)
    args = ap.parse_args()
    run_config(args.sweep_source, args.config_idx)

if __name__ == "__main__":
    main()
