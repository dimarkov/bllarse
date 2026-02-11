from typing import Any, Dict, List

from bllarse.tools.adapters import run_vbll_training_from_config


BASE = dict(
    dataset="cifar10",
    seed=0,
    epochs=3,
    batch_size=32,
    num_workers=2,
    optimizer="adamw",
    learning_rate=1e-3,
    weight_decay=1e-4,
    num_blocks=6,
    embed_dim=512,
    pretrained="in21k_cifar",
    tune_mode="last_layer",
    nodataaug=True,
    prior_scale=1.0,
    wishart_scale=0.1,
    parameterization="diagonal",
    cov_rank=2,
    vbll_type="discriminative",
    device="cuda",
    enable_mlflow=True,
    mlflow_experiment="bllarse",
    group_id="vbll_smoke_sweep",
)


def create_configs() -> List[Dict[str, Any]]:
    return [dict(BASE)]


def run(config: Dict[str, Any]) -> None:
    run_vbll_training_from_config(config)
