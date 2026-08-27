#!/bin/bash -l
# Peak GPU memory per arm, at the campaign's own micro batch. Fig 7c.
#SBATCH --time=00:40:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=125000
#SBATCH --output=slurm_logs/%x.%j.out
#SBATCH --error=slurm_logs/%x.%j.err
#
# Peak memory is a property of (arm, model, micro batch) on ONE device, not of
# a trajectory, so this is a single short job rather than anything per-seed --
# and it has to run on the A100s the campaign ran on. A number measured on a
# different card answers a different question.
#
# All three arms in one allocation: two JAX (ce, ibprobit) through
# probe_micro_batch.py and one torch (vbll) through probe_micro_batch_torch.py,
# since jax[cuda] and torch cannot share an environment. Sequential, because
# they would otherwise contend for the same device and each would report the
# other's allocations as its own.
set -euo pipefail

cd "${BLLARSE_FT_REPO:?}"
out="${BLLARSE_FT_MEMDIR:?}"
model="${BLLARSE_FT_MODEL:-dinov3_large}"
mkdir -p "$out"

export PYTHONPATH="${BLLARSE_FT_REPO}/src:${PYTHONPATH:-}"
export HF_HOME="${BLLARSE_HF_HOME:?}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[mem] model=$model node=$(hostname)"

"${BLLARSE_FT_VENV:?}" scripts/finetune/probe_micro_batch.py \
  --arm ce --model "$model" --peak-at "${BLLARSE_MB_CE:-32}" --bf16 \
  --json-out "$out/mem_${model}_ce.json"

"${BLLARSE_FT_VENV}" scripts/finetune/probe_micro_batch.py \
  --arm ibprobit --model "$model" --peak-at "${BLLARSE_MB_IB:-32}" --bf16 \
  --num-update-iters "${BLLARSE_IB_ITERS:-128}" \
  --json-out "$out/mem_${model}_ibprobit.json"

"${BLLARSE_FT_VBLL_VENV:?}" scripts/finetune/probe_micro_batch_torch.py \
  --model "$model" --peak-at "${BLLARSE_MB_VBLL:-64}" --bf16 \
  --prior-scale "${BLLARSE_VBLL_PRIOR_SCALE:-197.71719501241446}" \
  --kl-ratio "${BLLARSE_VBLL_KL_RATIO:-0.0010040873768869787}" \
  --wishart-scale "${BLLARSE_VBLL_WISHART:-0.13}" \
  --parameterization "${BLLARSE_VBLL_PARAM:-diagonal}" \
  --vbll-type "${BLLARSE_VBLL_TYPE:-discriminative}" \
  --json-out "$out/mem_${model}_vbll.json"

echo "[mem] done:"
ls -la "$out"
