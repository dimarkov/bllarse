#!/usr/bin/env python3
import argparse
import os
from bllarse.tools.module import get_module_from_source_path

def run_config(sweep_source: str, config_idx: int):
    sweep = get_module_from_source_path(sweep_source)
    all_configs = sweep.create_configs()
    if config_idx < 0 or config_idx >= len(all_configs):
        raise IndexError(f"{config_idx=} out of range (0..{len(all_configs)-1})")
    cfg = all_configs[config_idx]

    # If MLflow is enabled, rely on a pre-created parent run (from run_sweep)
    # and pass the parent run id via env var so finetuning.py creates a nested run.
    enable_mlflow = bool(cfg.get("enable_mlflow", False)) or bool(cfg.get("enable_wandb", False))
    if enable_mlflow:
        from bllarse.mlflow_utils import load_mlflow_env_defaults

        load_mlflow_env_defaults()
        if not os.environ.get("MLFLOW_PARENT_RUN_ID"):
            print(
                "[bllarse] WARNING: MLFLOW_PARENT_RUN_ID is not set. "
                "Runs will not be nested under a sweep parent. "
                "Use run_sweep.py to create the parent run first."
            )

    sweep.run(cfg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_source", type=str)
    ap.add_argument("config_idx", type=int)
    args = ap.parse_args()
    run_config(args.sweep_source, args.config_idx)

if __name__ == "__main__":
    main()
