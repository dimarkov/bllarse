#!/usr/bin/env python3
"""Download W&B run history as parquet files from logged artifacts.

This script downloads the raw parquet files from W&B artifacts and stores them
in a local results directory along with run configs as JSON files.

Usage:
    python scripts/download_wandb_artifacts.py --group sweep7a1_fnf_dataaug_adamw_batchsize_numiters
    python scripts/download_wandb_artifacts.py --group sweep7a1_fnf_dataaug_adamw_batchsize_numiters --output-dir results/sweep7a1
"""

import os
import json
import argparse
import wandb


def download_artifacts(entity: str, project: str, group: str, output_dir: str):
    """Download parquet artifacts and configs for all runs in a group."""
    
    api = wandb.Api(timeout=1000)
    filters = {"group": group}
    
    print(f"Querying runs from {entity}/{project} with group={group}...")
    runs = api.runs(f"{entity}/{project}", filters=filters)
    
    # Convert to list to get count
    runs_list = list(runs)
    total_runs = len(runs_list)
    print(f"Found {total_runs} runs")
    
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded = 0
    for i, run in enumerate(runs_list):
        run_dir = os.path.join(output_dir, run.id)
        os.makedirs(run_dir, exist_ok=True)
        
        print(f"[{i + 1}/{total_runs}] Processing run: {run.name} ({run.id})")
        
        # Save config
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(run.config, f, indent=2)
        
        # Download history artifact
        found_history = False
        try:
            for artifact in run.logged_artifacts():
                if artifact.type == "wandb-history":
                    print(f"  Downloading history artifact: {artifact.name}")
                    local_path = artifact.download(root=run_dir)
                    print(f"  Downloaded to: {local_path}")
                    found_history = True
                    downloaded += 1
                    break
        except Exception as e:
            print(f"  Warning: could not fetch artifacts: {e}")
        
        if not found_history:
            # Fallback: try to save history directly
            print(f"  No wandb-history artifact found, trying direct history export...")
            try:
                history = run.history(pandas=True)
                if not history.empty:
                    history_path = os.path.join(run_dir, "history.parquet")
                    history.to_parquet(history_path)
                    print(f"  Saved history to: {history_path}")
                    downloaded += 1
            except Exception as e:
                print(f"  Warning: could not export history: {e}")
    
    print(f"\nDone! Downloaded data for {downloaded}/{total_runs} runs to {output_dir}")
    return downloaded


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="verses_ai",
        help="W&B entity (team/user)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="bllarse_experiments",
        help="W&B project name",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="sweep7a1_fnf_dataaug_adamw_batchsize_numiters",
        help="W&B run group name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scripts/results",
        help="Directory to store downloaded artifacts",
    )
    args = parser.parse_args()
    
    download_artifacts(args.entity, args.project, args.group, args.output_dir)


if __name__ == "__main__":
    main()
