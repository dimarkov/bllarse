# bllarse

Fitting Bayesian last layers or Bayesian "heads" with sparse priors to embeddings of deep neural networks, and pruning their parameters with Bayesian model reduction.


## Running finetuning

We recommend using `uv` to create a virtual environment and install dependencies


```bash
# (Re)create and synchronize your virtual environment:
uv venv
uv sync

# Activate it:
source .venv/bin/activate

# Run any Python code, e.g.:
python scripts/finetuning.py
```

## SLURM sweeps

Python sweep files provide two functions:

```python
def create_configs() -> list[dict]:
    ...

def run(config: dict) -> None:
    ...
```

[`bllarse_sweeps/finetuning_demo.py`](bllarse_sweeps/finetuning_demo.py)
is a small example using the standard finetuning entry point. From a SLURM
login node, launch it as a GPU array:

```bash
source .venv/bin/activate

python -m bllarse.tools.run_sweep \
  bllarse_sweeps/finetuning_demo.py \
  --venv .venv \
  --max-concurrent 4 \
  --cpus-per-task 8 \
  --job-name finetuning_demo
```

The default job runs under Apptainer (`src/slurm/jobs/slurm_run_config_apptainer.sh`),
since the target cluster forbids the Docker daemon. Point it at a built `.sif`
via `BLLARSE_SIF`; the JAX arms (AdamW+CE, IBProbit) and the VBLL/PyTorch arm
use separate images (`bllarse-jax.sif`, `bllarse-torch.sif` — see
`src/slurm/images/*.def`) because they pin conflicting CUDA/cuDNN versions:

```bash
export BLLARSE_SIF=/path/to/bllarse-jax.sif
```

The workstation's `~/.cache/huggingface` is a symlink to external storage that
won't exist on the cluster, so the HF cache path is overridable via
`BLLARSE_HF_HOME` (bound into the container and exported as `HF_HOME` when set).

On this workstation, or anywhere a Docker daemon is available, use
`--job-script src/slurm/jobs/slurm_run_config_docker.sh` instead; it expects
an image named `bllarse-dev` on each allocated node. Override the image and
Docker shared-memory allocation when needed:

```bash
export BLLARSE_DOCKER_IMAGE=bllarse-dev
export BLLARSE_DOCKER_SHM_SIZE=16g
```

Use `--job-script src/slurm/jobs/slurm_run_config.sh` to run directly in the
host virtual environment instead (no container). `--dry-run` validates the
sweep and prints the `sbatch` command without submitting it.

Large sweeps can be split while preserving the original config indices:

```bash
python -m bllarse.tools.run_sweep bllarse_sweeps/my_sweep.py \
  --index-offset 0 --num-jobs 1000 --max-concurrent 50

python -m bllarse.tools.run_sweep bllarse_sweeps/my_sweep.py \
  --index-offset 1000 --num-jobs 1000 --max-concurrent 50
```

When a selected config sets `enable_mlflow=True`, the launcher creates one
MLflow parent run and passes its ID to every array task. Use
`--parent-run-id <RUN_ID>` to attach another submission to an existing parent.
Parent creation is fail-loud: if MLflow cannot create the requested parent,
the array is not submitted.

## 🔄 Updating Git-based dependencies

Some dependencies in `pyproject.toml` are installed directly from GitHub (e.g. `blrax`, `mlpox`).  
By default, `uv sync` will use the commit pinned in your `uv.lock` file, so you won’t automatically get the latest changes from `main`.

To update one of these packages to the newest commit on its branch:

```bash
uv sync --upgrade-package blrax # replace `blrax` with the name of the dependency you want to upgrade.
```

This updates the lockfile to the latest commit and reinstalls the package in your virtual environment.
