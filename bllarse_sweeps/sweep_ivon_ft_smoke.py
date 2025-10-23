from itertools import product
from typing import Any, Dict, List

from bllarse.tools.adapters import run_training_from_config

# -------------------- Smoke sweep axes --------------------
DATASETS = ["cifar10", "cifar100"]
PRETRAINED = ["in21k", "in21k_cifar"]
BATCH_SIZES = [512]
SEEDS = [0]
WEIGHT_DECAYS = [1e-6]
HESS_INITS = [0.1]

# Lighter defaults to keep the run quick
BASE: Dict[str, Any] = dict(
    optimizer="ivon",
    loss_fn="CrossEntropy",
    label_smooth=0.0,
    enable_wandb=False,
    device="gpu",
    save_every=1,
    embed_dim=512,
    num_blocks=6,
    mc_samples=1,
    learning_rate=5e-4,
    nodataaug=True,
)


def _epochs_for(dataset: str) -> int:
    return 2 if dataset == "cifar10" else 3


def _mk_cfg(
    dataset: str,
    pretrained: str,
    batch_size: int,
    seed: int,
    weight_decay: float,
    hess_init: float,
) -> Dict[str, Any]:
    return dict(
        **BASE,
        dataset=dataset,
        pretrained=pretrained,
        batch_size=batch_size,
        seed=seed,
        epochs=_epochs_for(dataset),
        weight_decay=weight_decay,
        hess_init=hess_init,
        group_id="smoke_ivon_ft",
    )


def create_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for (dataset, pretrained, batch_size, seed) in product(
        DATASETS, PRETRAINED, BATCH_SIZES, SEEDS
    ):
        for (weight_decay, hess_init) in product(WEIGHT_DECAYS, HESS_INITS):
            configs.append(
                _mk_cfg(dataset, pretrained, batch_size, seed, weight_decay, hess_init)
            )
    return configs


def run(config: Dict[str, Any]) -> None:
    run_training_from_config(config)
