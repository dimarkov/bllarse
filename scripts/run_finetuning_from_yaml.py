#!/usr/bin/env python3
"""
Inside-container entry-point for a SLURM array task or local smoke-test.

Usage
-----
python scripts/run_finetuning_from_yaml.py <yaml_file> <row_idx>
"""

import argparse, yaml, sys
from pathlib import Path

# repo root already on $PYTHONPATH inside container
from last_layer_finetuning import (
    main as train_main,
    build_argparser,
    build_configs,
)
def dict_to_argv(d):
    """Turns {'batch_size':128,'enable_wandb':True} → ['--batch-size','128','--enable-wandb']"""
    argv = []
    for k, v in d.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v: 
                argv.append(flag)
        else:
            argv.extend([flag, str(v)])
    return argv

def main(yaml_path: Path, idx: int):
    cfgs = yaml.safe_load(Path(yaml_path).read_text())
    if idx >= len(cfgs):
        raise IndexError(f"{idx=} out of range (0-{len(cfgs)-1})")
    cfg = cfgs[idx]
    print("🔹 Selected config:", cfg)

    parser = build_argparser()
    args = parser.parse_args(dict_to_argv(cfg))
    m_conf, o_conf = build_configs(args)
    train_main(args, m_conf, o_conf)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_path", type=Path, help="Path to sweep YAML")
    ap.add_argument("idx",       type=int,  help="Row index (e.g. SLURM_ARRAY_TASK_ID)")
    args = ap.parse_args()

    main(args.yaml_path, args.idx)
