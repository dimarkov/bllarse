from itertools import product
from typing import Any, Dict, List

from bllarse.tools.adapters import run_training_from_config

# Sweep axes
DATASETS = ["cifar10", "cifar100"]
SEEDS = [0, 1, 2, 3, 4]
BATCH_SIZES = [512, 1024, 2048, 4096, 8192, 16384]
USE_DATA_AUG = [True, False]  # toggles nodataaug flag
NUM_ITERS = [2, 4, 8, 16, 32, 64]
SEQUENTIAL_UPDATE = [True, False]
RESET_LOSS_PER_EPOCH = [True, False]

# Global defaults for unified finetuning.py (full network, IBProbit head)
BASE = dict(
    tune_mode="full_network",
    loss_fn="IBProbit",
    label_smooth=0.0,
    enable_wandb=True,
    device="gpu",
    save_every=1,
    embed_dim=1024,
    num_blocks=12,
    pretrained="in21k",
    optimizer="adamw",
    epochs=20,
    learning_rate=1e-4,
    weight_decay=1e-4,
)


def _mk_cfg(
    dataset: str,
    seed: int,
    batch_size: int,
    use_data_aug: bool,
    num_update_iters: int,
    sequential_update: bool,
    reset_loss_per_epoch: bool,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = dict(
        **BASE,
        dataset=dataset,
        group_id="sweep7a1_fnf_dataaug_adamw_batchsize_numiters",
        seed=seed,
        batch_size=batch_size,
        nodataaug=not use_data_aug,
        num_update_iters=num_update_iters,
        sequential_update=sequential_update,
        reset_loss_per_epoch=reset_loss_per_epoch,
    )
    return cfg


def create_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []

    for (
        dataset,
        seed,
        use_data_aug,
        batch_size,
        num_update_iters,
        sequential_update,
        reset_loss_per_epoch,
    ) in product(
        DATASETS,
        SEEDS,
        USE_DATA_AUG,
        BATCH_SIZES,
        NUM_ITERS,
        SEQUENTIAL_UPDATE,
        RESET_LOSS_PER_EPOCH,
    ):
        configs.append(
            _mk_cfg(
                dataset=dataset,
                seed=seed,
                batch_size=batch_size,
                use_data_aug=use_data_aug,
                num_update_iters=num_update_iters,
                sequential_update=sequential_update,
                reset_loss_per_epoch=reset_loss_per_epoch,
            )
        )

    return configs


def run(config: Dict[str, Any]) -> None:
    run_training_from_config(config)
