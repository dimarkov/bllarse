from itertools import product
from typing import Any, Dict, List

from bllarse.tools.adapters import run_vbll_training_from_config

# Sweep axes
DATASETS = ["cifar10", "cifar100"]
PRETRAINED_SOURCES = ["in21k", "in21k_cifar"]
TUNE_MODES = ["last_layer", "full_network"]
BATCH_SIZES = [512, 1024, 2048, 4096, 8192, 16384]
LEARNING_RATES = [1e-4, 1e-3]
SEEDS = [0]

# Global defaults for VBLL finetuning
BASE = dict(
    epochs=20,
    optimizer="adamw",
    weight_decay=1e-4,
    num_blocks=12,
    embed_dim=1024,
    nodataaug=True,
    parameterization="diagonal",
    vbll_type="discriminative",
    return_ood=False,
    prior_scale=1.0,
    wishart_scale=0.1,
    device="cuda",
    num_workers=4,
    enable_mlflow=True,
    mlflow_experiment="bllarse",
)


def _mk_cfg(
    dataset: str,
    pretrained: str,
    tune_mode: str,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = dict(
        **BASE,
        dataset=dataset,
        pretrained=pretrained,
        tune_mode=tune_mode,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        group_id="sweep9a1_vbll_discriminative_diagonal_fnf_llf_adamw_bs_lr",
    )
    return cfg


def create_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []

    for (
        dataset,
        pretrained,
        tune_mode,
        batch_size,
        learning_rate,
        seed,
    ) in product(
        DATASETS,
        PRETRAINED_SOURCES,
        TUNE_MODES,
        BATCH_SIZES,
        LEARNING_RATES,
        SEEDS,
    ):
        configs.append(
            _mk_cfg(
                dataset=dataset,
                pretrained=pretrained,
                tune_mode=tune_mode,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed,
            )
        )

    return configs


def run(config: Dict[str, Any]) -> None:
    run_vbll_training_from_config(config)
