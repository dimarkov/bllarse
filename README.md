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

The default job uses `scripts/start_docker_sbatch.sh` and expects an image
named `bllarse-dev` on each allocated node. Override the image and Docker
shared-memory allocation when needed:

```bash
export BLLARSE_DOCKER_IMAGE=bllarse-dev
export BLLARSE_DOCKER_SHM_SIZE=16g
```

Use `--job-script src/slurm/jobs/slurm_run_config.sh` to run directly in the
host virtual environment instead. `--dry-run` validates the sweep and prints
the `sbatch` command without submitting it.

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
