from typing import Dict, Any, List
from itertools import product
from bllarse.tools.adapters import run_training_from_config

# -------------------- Sweep axes --------------------
DATASETS    = ["cifar10", "cifar100"]
OPTIMIZERS  = ["lion", "adamw"] 
BATCH_SIZES = [512, 1024, 2048, 4096, 8192, 16384]
SEEDS       = [0, 1, 2, 3]  # 4 seeds

# Optimizer-specific LR / WD grids
LR_GRID = {
    "adamw": [3e-4, 1e-3, 3e-3],
    "lion":  [1e-4, 2e-4, 3e-4],
}
WD_GRID = {
    "adamw": [1e-3, 1e-2, 2e-2],
    "lion":  [2e-2, 3e-2, 5e-2],
}

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
    learning_rate: float,
    weight_decay: float,
) -> Dict[str, Any]:
    if dataset == "cifar10":
        epochs = 20
    elif dataset == "cifar100":
        epochs = 40
    return dict(
        **BASE,
        dataset=dataset,
        batch_size=batch_size,
        optimizer=optimizer,
        epochs=epochs,
        group_id="sweep2_batchsize_by_optimizer_by_lr",
        seed=seed,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

def create_configs() -> List[Dict[str, Any]]:
    config_combos: List[Dict[str, Any]] = []
    for (ds, bs, opt, sd) in product(DATASETS, BATCH_SIZES, OPTIMIZERS, SEEDS):
        for (lr, wd) in product(LR_GRID[opt], WD_GRID[opt]):
            config_combos.append(
                _mk_cfg(ds, bs, opt, sd, lr, wd)
            )
    return config_combos

def run(config: Dict[str, Any]) -> None:
    run_training_from_config(config)
