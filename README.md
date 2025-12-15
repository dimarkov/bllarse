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
>>> python scripts/last_layer_finetuning.py
```

Once inside the container:

```bash
# (Re)create and synchronize your virtual environment:
uv venv
uv sync

# Activate it:
source .venv/bin/activate

# Run any Python code, e.g.:
python scripts/last_layer_finetuning.py
```

## 🔄 Updating Git-based dependencies

Some dependencies in `pyproject.toml` are installed directly from GitHub (e.g. `blrax`, `mlpox`).  
By default, `uv sync` will use the commit pinned in your `uv.lock` file, so you won’t automatically get the latest changes from `main`.

To update one of these packages to the newest commit on its branch:

```bash
uv sync --upgrade-package blrax # replace `blrax` with the name of the dependency you want to upgrade.
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
   python scripts/last_layer_finetuning.py
   ```

