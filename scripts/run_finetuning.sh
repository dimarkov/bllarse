#!/usr/bin/env bash
set -e 

source .venv/bin/activate
python scripts/finetuning.py --epochs=10 --tune-mode last_layer
