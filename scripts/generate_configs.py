#!/usr/bin/env python3
"""
Run ONCE on the head node:

    python scripts/generate_hypercube.py --out ~/bllarse_sweeps/vb_cube.yaml

Creates a YAML list, one dict per config, incl. a unique 'uid'.
"""

import yaml, uuid, argparse
from datetime import datetime
from itertools import product
from pathlib import Path

BASE = dict(
    optimizer="cavi",
    loss_function="IBProbit",
    dataset="cifar10",
    epochs=100,
    save_every=10,
    embed_dim=512,
    num_blocks=6,
    pretrained="in21k_cifar",
    enable_wandb=True,          # always log
    group_id="vb_sweep_20250807"
)

GRID = dict(
    seed=[0,1,2,3,4,5],
    batch_size=[32,64,128],
    num_update_iters=[8,16,32,64],
)

def main(out_path: Path):
    keys = list(GRID)
    cube = []
    for vals in product(*[GRID[k] for k in keys]):
        cfg = BASE.copy()
        cfg.update({k: v for k, v in zip(keys, vals)})
        cfg["uid"] = uuid.uuid4().hex[:8]
        cube.append(cfg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cube))
    print(f"✅  Wrote {len(cube)} configs → {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p.add_argument("--out", type=Path,
        default=Path(f"~/bllarse_sweeps/sweep1_configs_{ts}.yaml").expanduser())
    main(p.parse_args().out)
