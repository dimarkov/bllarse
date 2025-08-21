#!/usr/bin/env python3
import argparse
from bllarse.tools.module import get_module_from_source_path

def run_config(sweep_source: str, config_idx: int):
    sweep = get_module_from_source_path(sweep_source)
    all_configs = sweep.create_configs()
    if config_idx < 0 or config_idx >= len(all_configs):
        raise IndexError(f"{config_idx=} out of range (0..{len(all_configs)-1})")
    cfg = all_configs[config_idx]
    sweep.run(cfg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_source", type=str)
    ap.add_argument("config_idx", type=int)
    args = ap.parse_args()
    run_config(args.sweep_source, args.config_idx)

if __name__ == "__main__":
    main()
