from itertools import product
from typing import Any, Dict, List

from bllarse.tools.adapters import run_training_from_config

# Sweep axes
DATASETS = ["cifar10", "cifar100"]
SEEDS = [0, 1, 2, 3]
OPTIMIZERS = ["adamw", "lion"]
BATCH_SIZES = [4096, 8192, 16384, 20480]
USE_DATA_AUG = [True, False]  # toggles nodataaug flag

# Optimizer-specific LR / WD grids
LR_GRID = {
    "adamw": [1e-4, 1e-3, 1e-2],
    "lion":  [1e-5, 2e-5, 1e-4],
}
WD_GRID = {
    "adamw": [1e-3, 2e-3, 1e-2],
    "lion":  [1e-4, 1e-3, 1e-2],
}

# Global defaults for unified finetuning.py (full network, IBProbit head)
BASE = dict(
    tune_mode="full_network",
    loss_fn="IBProbit",
    sequential_update=True,
    reset_loss_per_epoch=True,
    label_smooth=0.0,
    enable_wandb=True,
    device="gpu",
    save_every=1,
    embed_dim=1024,
    num_blocks=12,
    pretrained="in21k",
    num_update_iters=16,
)


def _epochs_for(dataset: str) -> int:
    return 20 if dataset == "cifar10" else 40


def _mk_cfg(
    dataset: str,
    optimizer: str,
    seed: int,
    batch_size: int,
    use_data_aug: bool,
    learning_rate: float,
    weight_decay: float,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = dict(
        **BASE,
        dataset=dataset,
        optimizer=optimizer,
        epochs=_epochs_for(dataset),
        group_id="sweep7_fnf_dataaug_optimizer_batchsize_lrwd",
        seed=seed,
        batch_size=batch_size,
        nodataaug=not use_data_aug,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    return cfg


def create_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []

    for dataset, seed, use_data_aug, optimizer, batch_size in product(
        DATASETS, SEEDS, USE_DATA_AUG, OPTIMIZERS, BATCH_SIZES
    ):
        for lr, wd in product(LR_GRID[optimizer], WD_GRID[optimizer]):
            configs.append(
                _mk_cfg(
                    dataset=dataset,
                    optimizer=optimizer,
                    seed=seed,
                    batch_size=batch_size,
                    use_data_aug=use_data_aug,
                    learning_rate=lr,
                    weight_decay=wd,
                )
            )

    return configs


def run(config: Dict[str, Any]) -> None:
    run_training_from_config(config)
