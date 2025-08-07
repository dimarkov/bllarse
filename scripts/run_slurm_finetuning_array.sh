#!/usr/bin/env bash
# Submit with:
#   rows=$(yq '. | length' bllarse_sweeps/my_smoke_sweep.yaml)
#   sbatch --array=0-$((rows-1))\
#          scripts/run_slurm_finetuning_array.sh \
#          bllarse_sweeps/my_smoke_sweep.yaml  --job-name my_smoke_sweep

#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH -G 1                             # one GPU, cluster’s preferred flag
#SBATCH --output=slurm_logs/%x.%A_%a.out
#SBATCH --error=slurm_logs/%x.%A_%a.err

set -euo pipefail

# ---------------- inputs ----------------
YAML=$1          # first positional arg

# path to repo root (visible both outside & inside the container)
REPO_ROOT=$(git rev-parse --show-toplevel)

# use the existing container wrapper you already have
scripts/start_docker_sbatch.sh \
  scripts/run_finetuning_array_internal.sh \
  "$YAML" "$SLURM_ARRAY_TASK_ID"
