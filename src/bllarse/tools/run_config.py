#!/usr/bin/env python3
import argparse
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bllarse.tools.module import get_module_from_source_path


def _config_uses_mlflow(config: Any) -> bool:
    return isinstance(config, Mapping) and bool(config.get("enable_mlflow", False))


def run_config(sweep_source: str | Path, config_idx: int) -> None:
    """Run one indexed configuration from a sweep source file."""
    sweep = get_module_from_source_path(sweep_source)
    configs = sweep.create_configs()

    if config_idx < 0 or config_idx >= len(configs):
        raise IndexError(f"config_idx={config_idx} out of range (0..{len(configs) - 1})")

    config = configs[config_idx]
    if _config_uses_mlflow(config):
        from bllarse.mlflow_utils import load_mlflow_env_defaults

        load_mlflow_env_defaults()
        if not os.environ.get("MLFLOW_PARENT_RUN_ID"):
            print(
                "[bllarse] WARNING: MLFLOW_PARENT_RUN_ID is not set; "
                "this run will not be nested under a sweep parent."
            )

    sweep.run(config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_source", type=Path)
    parser.add_argument("config_idx", type=int)
    args = parser.parse_args()
    run_config(args.sweep_source, args.config_idx)


if __name__ == "__main__":
    main()
