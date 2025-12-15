"""
A tiny smoke-test sweep that runs quickly locally.
- Keep epochs small
- Tiny batch sizes
- 2-3 configs max
"""
from typing import Dict, Any, List
from bllarse.tools.adapters import run_training_from_config

def create_configs() -> List[Dict[str, Any]]:
    base = dict(
        optimizer="cavi",
        loss_function="IBProbit",
        dataset="cifar10",
        epochs=2,                 # small for smoke test
        save_every=1,             # log each epoch
        embed_dim=512,
        num_blocks=6,
        pretrained="in21k_cifar",
        enable_wandb=True,
        group_id="smoke_local",
        label_smooth=0.0,
        warmup=0,
        device="gpu",             # local test on CPU; switch to 'gpu' remotely
    )
    grid = [
        dict(seed=0, batch_size=64,  num_update_iters=4),
        dict(seed=1, batch_size=128, num_update_iters=8),
    ]
    return [{**base, **g} for g in grid]

def run(config: Dict[str, Any]) -> None:
    run_training_from_config(config)
