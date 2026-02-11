from itertools import product
from typing import Any, Dict, List

from bllarse.tools.adapters import run_vbll_training_from_config

# Smoke-test axes (small but still exercises key branches)
DATASETS = ["cifar10", "cifar100"]
PRETRAINED_SOURCES = ["in21k", "in21k_cifar"]
TUNE_MODES = ["last_layer", "full_network"]
BATCH_SIZES = [512]
LEARNING_RATES = [1e-4]
SEEDS = [0]

COV_RANK_BY_DATASET = {
    "cifar10": 52,
    "cifar100": 6,
}

BASE = dict(
    epochs=2,
    optimizer="adamw",
    weight_decay=1e-4,
    num_blocks=12,
    embed_dim=1024,
    nodataaug=True,
    parameterization="lowrank",
    vbll_type="discriminative",
    prior_scale=1.0,
    wishart_scale=0.1,
    device="cuda",
    num_workers=2,
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
        cov_rank=COV_RANK_BY_DATASET[dataset],
        group_id="sweep8a1_smoketest_vbll",
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
