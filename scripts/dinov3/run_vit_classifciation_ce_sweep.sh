#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_NAME="bllarse"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Shared defaults ──
EPOCHS=1000
LOSS_FN="CrossEntropy"
SEEDS=(143 145)

# ── Sweep axes ──
MODELS=(dinov3_huge)
DATASETS=(food101 flowers102 stanford_cars dtd imagenet1k)
BATCH_SIZES=(512 1024 2048 4096 8192 16384)

# Loss-specific hyperparameters
IBPROBIT_NUM_UPDATE_ITERS=(16 32 64 128 256)
CE_LRS=(1e-2 5e-3 1e-3)
CE_PATIENCE=5

# ── Get or create parent run for the loss function ──
PARENT_RUN_ID=$(uv run "$SCRIPT_DIR/get_mlflow_run.py" \
    --experiment-name "$EXPERIMENT_NAME" --run-name "sweep10a-alt-classification-dinov3-$LOSS_FN")
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
            --num-update-iters "${IBPROBIT_NUM_UPDATE_ITERS[@]}" \
            --log-mlflow

    elif [[ "$LOSS_FN" == "CrossEntropy" ]]; then
        echo "Running: $model / $dataset / seed=$seed (CrossEntropy sweep)"
        uv run "$SCRIPT_DIR/vit_classification.py" \
            --model "$model" --dataset "$dataset" --loss-fn "$LOSS_FN" \
            --run-id "$PARENT_RUN_ID" \
            --epochs "$EPOCHS" --seed "$seed" \
            --batch-size "${BATCH_SIZES[@]}" \
            --lr "${CE_LRS[@]}" \
            --patience "$CE_PATIENCE" \
            --log-mlflow

    else
        echo "Unknown loss function: $LOSS_FN"
        exit 1
    fi

done
done
done
