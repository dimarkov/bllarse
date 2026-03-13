# bllarse

Fitting Bayesian last layers or Bayesian "heads" with sparse priors to embeddings of deep neural networks, and pruning their parameters with Bayesian model reduction.

## 🚀 Building the Docker image

> **Note:** Build the image on a GPU-equipped node so that CUDA drivers are available.

```bash
# From the project root:
docker build -t bllarse-dev .
```

## 💻 Interactive container usage

Use the interactive startup script to enter a shell (or run a one-off command) inside the container, with your home directory and project workspace mounted:

```bash
# Drop into an interactive bash shell:
scripts/start_docker_interactive.sh

# And then run a specific command, e.g. finetuning script:
>>> source .venv/bin/activate # assumes you've already created a venv and synced dependencies
>>> python scripts/finetuning.py --tune-mode last_layer --epochs 3
```

Once inside the container:

```bash
# (Re)create and synchronize your virtual environment:
uv venv
uv sync

# Activate it:
source .venv/bin/activate

# Run any Python code, e.g.:
python scripts/finetuning.py --tune-mode last_layer --epochs 3
```

## 🔄 Updating Git-based dependencies

Some dependencies in `pyproject.toml` are installed directly from GitHub (e.g. `blrax`, `mlpox`).  
By default, `uv sync` will use the commit pinned in your `uv.lock` file, so you won’t automatically get the latest changes from `main`.

To update one of these packages to the newest commit on its branch:

```bash
uv sync --active --upgrade-package blrax # replace `blrax` with the name of the dependency you want to upgrade.
```

This updates the lockfile to the latest commit and reinstalls the package in your virtual environment.

## 📋 SLURM cluster usage

We provide an sbatch wrapper for non-interactive, GPU-accelerated jobs.

1. **Submit a SLURM job** from the login node of your computing cluster:

   ```bash
   sbatch scripts/run_slurm_finetuning_test.sh
   ```

2. **How it works**:

   * `scripts/run_slurm_finetuning_test.sh` requests a GPU node and then invokes:

     ```bash
     scripts/start_docker_sbatch.sh scripts/run_finetuning.sh
     ```
   * `scripts/start_docker_sbatch.sh` launches the Docker container (with `--gpus`), mounts your home and working directory, and executes the provided script inside.

3. **`scripts/run_finetuning.sh`** (inside the container):

   ```bash
   #!/usr/bin/env bash
   set -e

   # Activate the virtual environment
   source .venv/bin/activate

# Run the finetuning
   python scripts/finetuning.py --epochs=10 --tune-mode last_layer
   ```

## 🧪 SLURM sweeps (MLflow parent/child runs)

Sweeps are submitted from the **login node** using `run_sweep`, which launches a SLURM array.

- By default, `run_sweep` creates a single MLflow parent run for that submission and child runs are nested under it.
- If you pass `--parent-run-id <RUN_ID>`, child runs are attached to that existing parent instead of creating a new one.

```bash
source .<venv_name>/bin/activate
export MLFLOW_TRACKING_URI="https://mycustom.url"  # or use ~/.mlflow/credentials

python -m bllarse.tools.run_sweep \
  bllarse_sweeps/llf_demo_sweep.py \
  --venv .<venv_name> \
  --max-concurrent 3 \
  --cpus-per-task 8 \
  --job-name llf_mlflow_smoke \
  --job-script src/slurm/jobs/slurm_run_config_docker.sh
```

### Attach additional jobs to an existing MLflow parent

Use this when you want multiple sweep submissions (e.g. extra seeds or chunks) grouped under one parent run:

```bash
python -m bllarse.tools.run_sweep \
  bllarse_sweeps/llf_demo_sweep.py \
  --venv .<venv_name> \
  --max-concurrent 3 \
  --cpus-per-task 8 \
  --job-name llf_mlflow_extra \
  --job-script src/slurm/jobs/slurm_run_config_docker.sh \
  --parent-run-id <EXISTING_PARENT_RUN_ID>
```

### Large sweeps (>1000 configs)

Chunk large sweeps with `--index-offset` and `--num-jobs`. This mirrors the `INDEX_OFFSET` logic in `slurm_run_config_docker.sh`:

```bash
python -m bllarse.tools.run_sweep bllarse_sweeps/huge_sweep.py \
  --venv .<venv_name> \
  --max-concurrent 50 \
  --cpus-per-task 8 \
  --job-name huge_part1 \
  --job-script src/slurm/jobs/slurm_run_config_docker.sh \
  --index-offset 0 \
  --num-jobs 1000

python -m bllarse.tools.run_sweep bllarse_sweeps/huge_sweep.py \
  --venv .<venv_name> \
  --max-concurrent 50 \
  --cpus-per-task 8 \
  --job-name huge_part2 \
  --job-script src/slurm/jobs/slurm_run_config_docker.sh \
  --index-offset 1000 \
  --num-jobs 1000
```

Tips:

- `--cpus-per-task` is optional, but recommended when data loaders use `num_workers > 0`.
- Docker shared memory can be tuned with `BLLARSE_DOCKER_SHM_SIZE` (default: `8g`), e.g.:

```bash
export BLLARSE_DOCKER_SHM_SIZE=16g
```

## RoBERTa MNLI Workflow

The RoBERTa/MNLI cached-feature pipeline lives in `scripts/roberta_mnli.py`.
It has three stages:

- `extract`: precompute frozen CLS features and store them under `data/feature_cache`
- `train_eval`: reuse cached features and train the last-layer baseline
- `all`: do extraction first if needed, then train/eval

### Feature extraction

Current canonical extraction config:

- backbone: `FacebookAI/roberta-base`
- max length: `512`
- cache dtype: `float16`
- extract batch size: `256`
- cluster resources: `1x A100 40GB`, `8 CPUs`, `BLLARSE_DOCKER_SHM_SIZE=16g`
- MLflow: disabled for extraction jobs by design

The canonical sweep file is:

- `bllarse_sweeps/mnli_roberta_len512_extract.py`

Typical cluster launch from the login node:

```bash
source .venv_bllarse_new/bin/activate
export HF_TOKEN="<token>"
export BLLARSE_DOCKER_SHM_SIZE=16g

python -m bllarse.tools.run_sweep \
  bllarse_sweeps/mnli_roberta_len512_extract.py \
  --venv .venv_bllarse_new \
  --max-concurrent 1 \
  --cpus-per-task 8 \
  --job-name mnli_len512_extract_bs256 \
  --job-script src/slurm/jobs/slurm_run_config_docker.sh
```

Direct one-off extraction also works:

```bash
python scripts/roberta_mnli.py \
  --stage extract \
  --backbone FacebookAI/roberta-base \
  --reuse-cache \
  --max-length 512 \
  --cache-dtype float16 \
  --extract-batch-size 256 \
  --hf-sync pull_push \
  --hf-repo-id dimarkov/bllarse-features \
  --hf-subdir-prefix roberta_activations/mnli_roberta_cls
```

Notes:

- The canonical HF destination is:
  `roberta_activations/mnli_roberta_cls/FacebookAI__roberta_base_len512_float16_5c31d846204b`
- Cache keys depend on backbone, max length, dtype, and sample caps. They do **not** depend on extraction batch size.
- Use `hf_sync=none` for throughput benchmarking or smoke runs so you do not push temporary caches.

### Adam / AdamW last-layer training

The deterministic head is a cached-feature linear softmax baseline trained on frozen CLS features.
Per-epoch `acc`, `nll`, and `ece` are logged to MLflow for `train_eval` jobs.

Quick len256 sanity sweep:

- `bllarse_sweeps/mnli_roberta_len256_linear_probe_baseline.py`

```bash
python src/bllarse/tools/run_config.py \
  bllarse_sweeps/mnli_roberta_len256_linear_probe_baseline.py 0
```

Full len256 multiseed sweep:

- `bllarse_sweeps/mnli_roberta_len256_linear_probe_multiseed.py`

```bash
source .venv_bllarse_new/bin/activate
export MLFLOW_TRACKING_URI="https://mlflow.markov.icu"
export BLLARSE_DOCKER_SHM_SIZE=16g

python -m bllarse.tools.run_sweep \
  bllarse_sweeps/mnli_roberta_len256_linear_probe_multiseed.py \
  --venv .venv_bllarse_new \
  --max-concurrent 24 \
  --job-name mnli_len256_multiseed \
  --job-script src/slurm/jobs/slurm_run_config_docker.sh
```

The current len256 multiseed sweep covers:

- optimizers: `adam`, `adamw`
- learning rates: `2e-5`, `3e-5`
- batch sizes: `8`, `4`
- seeds: `2022`, `2023`, `2024`
- epochs: `120`
- dropout: `0.1`

Optimizer conventions:

- `adam`: `weight_decay = 0.0`
- `adamw`: `weight_decay = 1e-2`

Direct single-run training example:

```bash
python scripts/roberta_mnli.py \
  --stage train_eval \
  --backbone FacebookAI/roberta-base \
  --reuse-cache \
  --optimizer adam \
  --learning-rate 2e-5 \
  --weight-decay 0.0 \
  --train-batch-size 8 \
  --epochs 120 \
  --dropout-rate 0.1 \
  --seed 2022 \
  --max-length 256 \
  --enable-mlflow
```
