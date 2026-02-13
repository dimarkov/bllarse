#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_NAME="bllarse"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Shared defaults ──
EPOCHS=1
LOSS_FN="IBProbit"
SEEDS=(137 139 141 143 147 151 153 157 161)

# ── Sweep axes ──
MODELS=(dinov3_small dinov3_big dinov3_large dinov3_huge deepMLP_big deepMLP_large)
DATASETS=(cifar10, cifar100, oxford_pets, food101, flowers102, stanford_cars, dtd, imagenet1k)
BATCH_SIZES=(512 1024 2048 4096 8192 16384)

# Loss-specific hyperparameters
IBPROBIT_NUM_UPDATE_ITERS=(16 32 64)
CE_LRS=(1e-2 5e-3 1e-3 5e-4)
CE_WEIGHT_DECAYS=(1e-4 1e-3)

# ── Precompute features for all model/dataset combos ──
echo "Precomputing features (skipping already cached)..."
uv run "$SCRIPT_DIR/precompute_features.py" \
    --models "${MODELS[@]}" \
    --datasets "${DATASETS[@]}"

# ── Get or create parent run for the loss function ──
PARENT_RUN_ID=$(uv run "$SCRIPT_DIR/get_mlflow_run.py" \
    --experiment-name "$EXPERIMENT_NAME" --run-name "alt-classification-dinov3-$LOSS_FN")
echo "Parent run for $LOSS_FN: $PARENT_RUN_ID"

# ── Sweep ──
for model in "${MODELS[@]}"; do
for dataset in "${DATASETS[@]}"; do
for seed in "${SEEDS[@]}"; do

    if [[ "$LOSS_FN" == "IBProbit" ]]; then
        echo "Running: $model / $dataset / seed=$seed (IBProbit sweep)"
        uv run "$SCRIPT_DIR/vit_classification.py" \
            --model "$model" --dataset "$dataset" --loss-fn "$LOSS_FN" \
            --run-id "$PARENT_RUN_ID" \
            --epochs "$EPOCHS" --seed "$seed" \
            --batch-size "${BATCH_SIZES[@]}" \
            --num-update-iters "${IBPROBIT_NUM_UPDATE_ITERS[@]}"

    elif [[ "$LOSS_FN" == "CrossEntropy" ]]; then
        echo "Running: $model / $dataset / seed=$seed (CrossEntropy sweep)"
        uv run "$SCRIPT_DIR/vit_classification.py" \
            --model "$model" --dataset "$dataset" --loss-fn "$LOSS_FN" \
            --run-id "$PARENT_RUN_ID" \
            --epochs "$EPOCHS" --seed "$seed" \
            --batch-size "${BATCH_SIZES[@]}" \
            --lr "${CE_LRS[@]}" \
            --weight-decay "${CE_WEIGHT_DECAYS[@]}"

    else
        echo "Unknown loss function: $LOSS_FN"
        exit 1
    fi

done
done
done
