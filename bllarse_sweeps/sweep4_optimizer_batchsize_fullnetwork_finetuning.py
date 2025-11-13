from itertools import product
from typing import Any, Dict, List

from bllarse.tools.adapters import run_training_from_config

# -------------------- Sweep axes --------------------
DATASETS = ["cifar10", "cifar100"]
BATCH_SIZES = [4096, 8192, 16384, 20480]
SEEDS = [0, 1, 2, 3, 4]
ADAMW_BEST_LR = 3e-3 # determined from analysis of sweep2a2 -- best at larger batch sizes
LION_BEST_LR = 3e-4  # determined from analysis of sweep2a2 -- best at larger batch sizes
OPTIMIZERS = ["adamw", "lion", "ivon"]

# Global defaults for full_network_finetuning.py
BASE = dict(
    label_smooth=0.0,
    enable_wandb=True,
    device="gpu",
    save_every=1,
    embed_dim=1024,
    num_blocks=12,
    mc_samples=1,
    nodataaug=True,
    pretrained="ini21k",
    hess_init=0.1,
    ivon_weight_decay=1e-6,
    weight_decay=0.02, # analysis of sweep2a2 indicated no difference between 0.02 and other WD conditions
)

def _epochs_for(dataset: str) -> int:
    return 20 if dataset == "cifar10" else 40

def _learning_rate_for(optimizer: str) -> float:
    if optimizer == "adamw":
        return ADAMW_BEST_LR
    elif optimizer == "lion":
        return LION_BEST_LR
    else:
        return 1e-3  # default when 'ivon' optimizer is used, but has no effect since 'ivon' has its own learning_rate configuration dict

def _mk_cfg(
    dataset: str,
    batch_size: int,
    optimizer: str,
    seed: int,
) -> Dict[str, Any]:
    return dict(
        **BASE,
        dataset=dataset,
        batch_size=batch_size,
        optimizer=optimizer,
        epochs=_epochs_for(dataset),
        group_id="sweep4a1_optimizer_batchsize_fullnetwork_training",
        learning_rate=_learning_rate_for(optimizer),
        seed=seed,
    )

def create_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for dataset, batch_size, seed, optimizer in product(
        DATASETS, BATCH_SIZES, SEEDS, OPTIMIZERS
    ):
        configs.append(
            _mk_cfg(
                dataset=dataset,
                batch_size=batch_size,
                optimizer=optimizer,
                seed=seed,
            )
        )
    return configs


def run(config: Dict[str, Any]) -> None:
    run_training_from_config(config, finetuning_type="full_network")