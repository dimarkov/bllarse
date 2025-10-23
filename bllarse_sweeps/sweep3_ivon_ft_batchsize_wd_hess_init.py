from itertools import product
from typing import Any, Dict, List

from bllarse.tools.adapters import run_training_from_config

# -------------------- Sweep axes --------------------
DATASETS = ["cifar10", "cifar100"]
DATAAUG = [True, False]
PRETRAINED = ["in21k", "in21k_cifar"]
BATCH_SIZES = [512, 1024, 2048, 4096, 8192, 16384]
SEEDS = [0, 1, 2, 3, 4]
WEIGHT_DECAYS = [1e-7, 1e-6, 1e-5, 1e-4]
HESS_INITS = [0.1, 0.5, 1.0]

# Global defaults for last_layer_finetuning.py
BASE = dict(
    optimizer="ivon",
    loss_fn="CrossEntropy",
    label_smooth=0.0,
    enable_wandb=True,
    device="gpu",
    save_every=1,
    embed_dim=1024,
    num_blocks=12,
    mc_samples=1,
    learning_rate=1e-3,
)


def _epochs_for(dataset: str) -> int:
    return 20 if dataset == "cifar10" else 40


def _mk_cfg(
    dataset: str,
    dataaug: bool,
    pretrained: str,
    batch_size: int,
    seed: int,
    weight_decay: float,
    hess_init: float,
) -> Dict[str, Any]:
    return dict(
        **BASE,
        dataset=dataset,
        nodataaug=not dataaug,
        pretrained=pretrained,
        batch_size=batch_size,
        seed=seed,
        epochs=_epochs_for(dataset),
        ivon_weight_decay=weight_decay,
        ivon_hess_init=hess_init,
        group_id="sweep3a2_ivon_batchsize_wd_hessinit",
    )


def create_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for dataset, dataug, pretrained, batch_size, seed in product(
        DATASETS, DATAAUG, PRETRAINED, BATCH_SIZES, SEEDS
    ):
        for weight_decay, hess_init in product(WEIGHT_DECAYS, HESS_INITS):
            configs.append(
                _mk_cfg(dataset, dataug, pretrained, batch_size, seed, weight_decay, hess_init)
            )
    return configs


def run(config: Dict[str, Any]) -> None:
    run_training_from_config(config)
