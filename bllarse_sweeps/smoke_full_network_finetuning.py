"""
Minimal sweep to smoketest the full-network finetuning path on Slurm.
- Keeps configs tiny so each run finishes quickly.
- Exercises each optimizer once to confirm docker + env setup.
"""
from typing import Any, Dict, List

from bllarse.tools.adapters import run_training_from_config

BASE = dict(
    dataset="cifar10",
    epochs=2,
    save_every=1,
    batch_size=512,
    embed_dim=512,
    num_blocks=6,
    label_smooth=0.0,
    nodataaug=True,
    pretrained="in21k",
    enable_wandb=False,
    device="gpu",
    mc_samples=1,
    ivon_weight_decay=1e-6,
    ivon_hess_init=0.1,
    weight_decay=0.02,
    learning_rate=5e-4,
    group_id="smoke_full_network_finetune",
    num_update_iters=4,
)

GRID = [
    dict(optimizer="adamw", seed=0),
    dict(optimizer="lion", learning_rate=3e-4, seed=1),
    dict(optimizer="ivon", seed=2),
]


def create_configs() -> List[Dict[str, Any]]:
    return [{**BASE, **overrides} for overrides in GRID]


def run(config: Dict[str, Any]) -> None:
    run_training_from_config(config, finetuning_type="full_network")
