#!/usr/bin/env python3
"""
Generate a YAML “hypercube” of configs for last-layer finetuning.

Example
-------
python scripts/generate_configs.py \
       --group vb_sweep_aug07          \  # W&B group name
       --out   ~/bllarse_sweeps/vb.yaml \
       --epochs 10 --save-every 1         # handy for a quick local smoke-test
"""

import argparse, yaml, uuid
from datetime import datetime
from itertools import product
from pathlib import Path

def make_base(args):
    """Base (non-swept) parameters."""
    return dict(
        optimizer      = "cavi",
        loss_function  = "IBProbit",
        dataset        = "cifar10",
        epochs         = 100,
        save_every     = 10,
        embed_dim      = 512,
        num_blocks     = 6,
        pretrained     = "in21k_cifar",
        enable_wandb   = True,
        group_id       = args.group,
    )

GRID = dict(
    seed            = [0, 1, 2, 3, 4, 5],
    batch_size      = [32, 64, 128],
    num_update_iters= [8, 16, 32, 64],
)

def main(args):
    base = make_base(args)
    keys = list(GRID)
    cube = []

    for vals in product(*[GRID[k] for k in keys]):
        cfg = base.copy()
        cfg.update({k: v for k, v in zip(keys, vals)})
        cfg["uid"] = uuid.uuid4().hex[:8]          # short but unique
        cube.append(cfg)

    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(cube))
    print(f"✅  Wrote {len(cube)} configs → {args.out}")

if __name__ == "__main__":
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path,
                   default=Path(f"~/bllarse_sweeps/sweep_configs_{now}.yaml"))
    p.add_argument("--group", type=str,
                   default=f"vb_sweep_{now}",
                   help="W&B group name stored in `group_id`.")
    main(p.parse_args())
