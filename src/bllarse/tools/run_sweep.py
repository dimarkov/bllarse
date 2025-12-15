#!/usr/bin/env python3
"""
Launch a SLURM array sweep from a Python sweep file.

The sweep file must define:
  - create_configs() -> List[object]
  - run(config: object) -> None
"""
import argparse
import os
import subprocess
from bllarse.tools.module import get_module_from_source_path

def run_sweep(sweep_source: str, venv_name: str, max_concurrent: int, job_name: str | None, job_script: str):
    sweep = get_module_from_source_path(sweep_source)
    all_configs = sweep.create_configs()
    n = len(all_configs)
    if n == 0:
        raise ValueError("create_configs() returned an empty list.")
    env = os.environ.copy()
    env["BLLARSE_SWEEP_SOURCE"] = sweep_source
    env["VENV_NAME"] = venv_name
    name = job_name or "bllarse_sweep"
    subprocess.run(
        [
            "sbatch",
            "--array", f"0-{n-1}%{max_concurrent}",
            "--job-name", name,
            job_script,
        ],
        env=env,
        check=True,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_source", type=str)
    ap.add_argument("--venv", type=str, default=".venv")
    ap.add_argument("--max-concurrent", type=int, default=7)
    ap.add_argument("--job-name", type=str, default=None)
    ap.add_argument("--job-script", type=str, default="slurm/jobs/slurm_run_config.sh")
    args = ap.parse_args()
    run_sweep(args.sweep_source, args.venv, args.max_concurrent, args.job_name, args.job_script)
if __name__ == "__main__":
    main()
