from typing import Dict, Any, List
from itertools import product
from bllarse.tools.adapters import run_training_from_config

# -------------------- Sweep axes --------------------
DATASETS    = ["cifar10"]
OPTIMIZERS  = ["lion", "adamw"]

BATCH_SIZES = [512]
SEEDS       = [0]  # 5 seeds


# Global defaults:
BASE = dict(
    loss_function="CrossEntropy",
    label_smooth=0.0,
    enable_wandb=True,
    device="gpu",          # cluster default
    save_every=1,          # per your spec
    embed_dim=1024,
    num_blocks=12,
    pretrained="in21k",
    nodataaug=False,
)

def _mk_cfg(
    dataset: str,
    batch_size: int,
    optimizer: str,
    seed: int,
) -> Dict[str, Any]:
    if dataset == "cifar10":
        epochs = 5
    elif dataset == "cifar100":
        epochs = 5
    return dict(
        **BASE,
        dataset=dataset,
        batch_size=batch_size,
        optimizer=optimizer,
        epochs=epochs,
        group_id="sweep1_batchsize_by_optimizer_smoke",
        seed=seed,
    )

def create_configs() -> List[Dict[str, Any]]:
    config_combos = [
        _mk_cfg(ds, bs, opt, sd)
        for (ds, bs, opt, sd) in product(DATASETS, BATCH_SIZES, OPTIMIZERS, SEEDS)
    ]

    return config_combos

def run(config: Dict[str, Any]) -> None:
    run_training_from_config(config)