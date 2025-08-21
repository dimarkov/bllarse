from typing import Dict, Any, List
from itertools import product
from bllarse.tools.adapters import run_training_from_config

# -------------------- Sweep axes --------------------
DATASETS    = ["cifar10", "cifar100"]
PRETRAINED  = ["in21k", "in21k_cifar"]

BATCH_SIZES = [512, 1024, 2048, 4096, 8192, 16384, 30000]

# vb iters:
VB_OFF = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]   # data aug OFF
VB_ON  = [1, 2, 4, 8, 16, 32]                    # data aug ON

# Fixed model bits:
EMBED_DIM  = 1024
NUM_BLOCKS = 12

# Global defaults:
BASE = dict(
    optimizer="cavi",
    loss_function="IBProbit",
    label_smooth=0.0,
    enable_wandb=True,
    device="gpu",          # cluster default
    save_every=1,          # per your spec
)

def _mk_cfg(
    dataset: str,
    pretrained: str,
    batch_size: int,
    num_update_iters: int,
    dataaug_on: bool,
) -> Dict[str, Any]:
    epochs = 20 if dataaug_on else 5
    return dict(
        **BASE,
        dataset=dataset,
        pretrained=pretrained,
        batch_size=batch_size,
        num_update_iters=num_update_iters,
        epochs=epochs,
        nodataaug=(not dataaug_on),   # your script: True == OFF
        embed_dim=EMBED_DIM,
        num_blocks=NUM_BLOCKS,
        group_id=f"sweep2_batchsize_numiters",
    )

def create_configs() -> List[Dict[str, Any]]:
    # OFF (no aug): bigger VB list, epochs=5
    off = [
        _mk_cfg(ds, pt, bs, vbi, dataaug_on=False)
        for (ds, pt, bs, vbi) in product(DATASETS, PRETRAINED, BATCH_SIZES, VB_OFF)
    ]

    # ON (aug): reduced VB list, epochs=20
    on = [
        _mk_cfg(ds, pt, bs, vbi, dataaug_on=True)
        for (ds, pt, bs, vbi) in product(DATASETS, PRETRAINED, BATCH_SIZES, VB_ON)
    ]

    return off + on

def run(config: Dict[str, Any]) -> None:
    run_training_from_config(config)
