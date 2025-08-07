#!/usr/bin/env bash
set -euo pipefail
YAML=$1
IDX=$2

# 0) cd to repo root
cd "$(git rev-parse --show-toplevel)"

# 1) activate venv
source .venv/bin/activate

# 2) run the row
python scripts/run_finetuning_from_yaml.py "$YAML" "$IDX"
