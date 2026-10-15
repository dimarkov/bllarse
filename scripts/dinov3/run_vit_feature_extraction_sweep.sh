#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_NAME="bllarse"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Sweep axes ──
# Every model must be a key of EQUIMO_MODELS: precompute_features.py takes its
# --models choices straight from it, so an unknown name fails the whole sweep at
# argparse before a single feature is extracted.
MODELS=(dinov3_small dinov3_big dinov3_large dinov3_huge)
DATASETS=(cifar10 cifar100 oxford_pets food101 flowers102 stanford_cars dtd imagenet1k)

# ── Precompute features for all model/dataset combos ──
echo "Precomputing features (skipping already cached)..."
uv run "$SCRIPT_DIR/precompute_features.py" \
    --models "${MODELS[@]}" \
    --datasets "${DATASETS[@]}"