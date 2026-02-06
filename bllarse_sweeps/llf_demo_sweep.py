from typing import Any, Dict, List

from bllarse.tools.adapters import run_training_from_config

NUM_UPDATE_ITERS = [1, 8, 16]

# Global defaults for last-layer Bayesian finetuning
BASE = dict(
    dataset="cifar10",
    tune_mode="last_layer",
    loss_fn="IBProbit",
    enable_mlflow=True,
    device="gpu",
    save_every=1,
    epochs=3,
    embed_dim=512,
    num_blocks=6,
    batch_size=64,
    nodataaug=True,
    pretrained="in21k_cifar",
    seed=0,
)


def _mk_cfg(num_update_iters: int) -> Dict[str, Any]:
    cfg: Dict[str, Any] = dict(
        **BASE,
        num_update_iters=num_update_iters,
        group_id="sweepXYZ_numiters_llf",
    )
    return cfg


def create_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for num_iters in NUM_UPDATE_ITERS:
        configs.append(_mk_cfg(num_iters))
    return configs


def run(config: Dict[str, Any]) -> None:
    run_training_from_config(config)
