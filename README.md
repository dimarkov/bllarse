# bllarse

Fitting Bayesian last layers or Bayesian "heads" with sparse priors to embeddings of deep neural networks, and pruning their parameters with Bayesian model reduction.

## Setup

We recommend using `uv` to create a virtual environment and install dependencies

```bash
# (Re)create and synchronize your virtual environment:
uv venv
uv sync

# Activate it:
source .venv/bin/activate
```

## Layout

| Path | What it holds |
|---|---|
| `src/bllarse/` | The library: Bayesian heads (`losses/`), metrics and OOD detection, logit dumps, the fine-tuning protocol (`finetune/`), Optuna tuning (`tuning/`) and the analysis that turns dumps into curves (`analysis/`) |
| `scripts/dinov3/` | Feature caching and linear probing on DINOv3 backbones |
| `scripts/finetune/` | Full-network ImageNet-1k fine-tuning: the JAX arms (AdamW+CE, IBProbit), the PyTorch VBLL arm, and the OOD worker |
| `scripts/vbll_pytorch/` | The VBLL baseline, built on the reference `vbll` package |
| `bllarse_sweeps/` | The linear-probing sweeps, with the tuned hyperparameters behind Figs 1 and 2 |
| `scripts/analysis/`, `scripts/plotting/` | Scoring dumps into JSON, Table 1, and the paper figures |

## Running a sweep configuration

A sweep file defines `create_configs() -> list[dict]` and `run(config)`. Run one
of its configurations by index:

```bash
python -m bllarse.tools.run_config bllarse_sweeps/probe_vbll.py 0
```

Every training script writes logit dumps; all reported metrics are computed
from them afterwards. MLflow logging is optional: it is on when a config sets
`enable_mlflow=True`, and it reports to whatever `MLFLOW_TRACKING_URI` points
at.

## From dumps to figures

Linear probing (Figs 1 and 2):

```bash
python scripts/analysis/fig1_curves.py --dump-root $DUMPS --out fig1_curves.json
python scripts/plotting/plot_fig1.py --json fig1_curves.json

python scripts/analysis/fig2_detection.py --dump-root $DUMPS --out fig2_detection.json
python scripts/plotting/plot_fig2.py --json fig2_detection.json
```

Full-network fine-tuning (Table 1, Figs 3-5):

```bash
python scripts/analysis/score_final.py --dump-root $DUMPS --group-id $GROUP \
    --model dinov3_huge --out scores.json
python scripts/analysis/wallclock_from_dumps.py --dump-root $DUMPS --group-id $GROUP \
    --model dinov3_huge --out wallclock.json

python scripts/analysis/table1_endpoint.py --scores scores.json
python scripts/plotting/plot_fig3.py --scores scores.json
python scripts/plotting/plot_fig4.py --scores scores.json
python scripts/plotting/plot_fig5.py --scores scores.json --wallclock wallclock.json \
    --target-arm ibprobit --target-epochs 1.0 --fraction 1.0
```

## 🔄 Updating Git-based dependencies

Some dependencies in `pyproject.toml` are installed directly from GitHub (e.g. `blrax`).  
By default, `uv sync` will use the commit pinned in your `uv.lock` file, so you won’t automatically get the latest changes from `main`.

To update one of these packages to the newest commit on its branch:

```bash
uv sync --upgrade-package blrax # replace `blrax` with the name of the dependency you want to upgrade.
```

This updates the lockfile to the latest commit and reinstalls the package in your virtual environment.
