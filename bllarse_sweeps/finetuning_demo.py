from typing import Any

from bllarse.tools.adapters import run_training_from_config
from bllarse.tools.config import get_config_grid


def _create_config(seed: int, num_update_iters: int) -> dict[str, Any]:
    return {
        "dataset": "cifar10",
        "tune_mode": "last_layer",
        "loss_fn": "IBProbit",
        "epochs": 1,
        "batch_size": 64,
        "num_update_iters": num_update_iters,
        "nodataaug": True,
        "pretrained": "in21k_cifar",
        "seed": seed,
        "enable_mlflow": True,
        "group_id": "finetuning_demo",
        "uid": f"seed{seed}_iters{num_update_iters}",
    }


def create_configs() -> list[dict[str, Any]]:
    return get_config_grid(
        _create_config,
        {
            "seed": [0, 1],
            "num_update_iters": [8, 16],
        },
    )


def run(config: dict[str, Any]) -> None:
    run_training_from_config(config)
