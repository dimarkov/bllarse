from itertools import product
from typing import Any, Dict, List

from bllarse.tools.adapters import run_training_from_config

# -------------------- Sweep axes --------------------
DATASETS = ["cifar10", "cifar100"]
BATCH_SIZES = [4096, 8192, 16384, 20480]
SEEDS = [0, 1, 2, 3]
OPTIMIZERS = ["adamw", "lion", "ivon"]

# Optimizer-specific LR / WD grids (for AdamW & Lion)
LR_GRID = {
    "adamw": [3e-4, 1e-3, 3e-3],
    "lion":  [1e-4, 2e-4, 3e-4],
}
WD_GRID = {
    "adamw": [1e-3, 1e-2, 2e-2],
    "lion":  [2e-2, 3e-2, 5e-2],
}

# IVON-specific grids
IVON_WEIGHT_DECAYS = [1e-7, 1e-6, 1e-5]
IVON_HESS_INITS = [0.01, 0.05, 0.1]

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
    pretrained="in21k",
    num_update_iters=16,    # full network finetuning script defaults to 16 num update iters, just being explicit
)


def _epochs_for(dataset: str) -> int:
    return 20 if dataset == "cifar10" else 40


def _mk_cfg(
    dataset: str,
    batch_size: int,
    optimizer: str,
    seed: int,
    *,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    ivon_weight_decay: float | None = None,
    ivon_hess_init: float | None = None,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = dict(
        **BASE,
        dataset=dataset,
        batch_size=batch_size,
        optimizer=optimizer,
        epochs=_epochs_for(dataset),
        group_id="sweep5a1_optimizer_hyperparameters_fullnetwork_finetuning",
        seed=seed,
    )

    if optimizer in ("adamw", "lion"):
        assert learning_rate is not None
        assert weight_decay is not None
        cfg["learning_rate"] = learning_rate
        cfg["weight_decay"] = weight_decay

    elif optimizer == "ivon":
        assert ivon_weight_decay is not None
        assert ivon_hess_init is not None
        cfg["ivon_weight_decay"] = ivon_weight_decay
        cfg["ivon_hess_init"] = ivon_hess_init

    return cfg


def create_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []

    for dataset, batch_size, seed in product(DATASETS, BATCH_SIZES, SEEDS):
        # AdamW & Lion: sweep learning_rate and weight_decay
        for optimizer in ("adamw", "lion"):
            for lr in LR_GRID[optimizer]:
                for wd in WD_GRID[optimizer]:
                    configs.append(
                        _mk_cfg(
                            dataset=dataset,
                            batch_size=batch_size,
                            optimizer=optimizer,
                            seed=seed,
                            learning_rate=lr,
                            weight_decay=wd,
                        )
                    )

        # IVON: sweep ivon_weight_decay and ivon_hess_init
        optimizer = "ivon"
        for wd in IVON_WEIGHT_DECAYS:
            for hess_init in IVON_HESS_INITS:
                configs.append(
                    _mk_cfg(
                        dataset=dataset,
                        batch_size=batch_size,
                        optimizer=optimizer,
                        seed=seed,
                        ivon_weight_decay=wd,
                        ivon_hess_init=hess_init,
                    )
                )

    return configs


def run(config: Dict[str, Any]) -> None:
    run_training_from_config(config, finetuning_type="full_network")
