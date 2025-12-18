from typing import Any, Dict, List

from bllarse.tools.adapters import run_training_from_config

OPTIMIZERS = ["adamw", "lion", "ivon"]

# Global defaults
BASE = dict(
    dataset="cifar10",
    tune_mode="full_network",
    label_smooth=0.0,
    enable_wandb=True,
    device="gpu",
    save_every=1,
    epochs=3,
    embed_dim=512,
    num_blocks=6,
    batch_size=64,
    nodataaug=True,
    pretrained="in21k",
    num_update_iters=16,
    seed=0,
)

def _mk_cfg(
    optimizer: str,
) -> Dict[str, Any]:
    """
    Create a config dictionary for a single experiment.
    """
    cfg: Dict[str, Any] = dict(
        **BASE,
        optimizer=optimizer,
        group_id="sweepXYZ_optimizer_fnf",
    )

    return cfg

def create_configs() -> List[Dict[str, Any]]:
    """
    Create a list of config dictionaries for the sweep.
    Each config corresponds to a different combination of hyperparameters.
    """
    configs: List[Dict[str, Any]] = []

    for optimizer in OPTIMIZERS:
            configs.append(
                _mk_cfg(
                    optimizer=optimizer,
                )
            )

    return configs

def run(config: Dict[str, Any]) -> None:
    run_training_from_config(config)