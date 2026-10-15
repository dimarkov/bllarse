# VBLL baseline (PyTorch)

The VBLL arm of the comparison, built on the reference
[VBLL](https://github.com/VectorInstitute/vbll) (Variational Bayesian Last
Layers) package, which is PyTorch-only.

| Path | What it holds |
|---|---|
| `probing_vbll.py` | VBLL linear probing on cached DINOv3 features: the same `.npz` files the JAX arms read, and the same dump format (`bllarse.evaldump`). Driven by `bllarse_sweeps/probe_vbll.py`, `bllarse_sweeps/probe_ood_vbll.py` and the Optuna tuning objective. |
| `src/vbll_experiments/models.py` | `build_vbll_head`, the bare VBLL classification head shared by `probing_vbll.py` and the full-network fine-tuning arm in `scripts/finetune/finetune_vbll.py`. |

## Setup

From the repository root:

```bash
uv sync --group vbll
```

## Usage

Linear probing, one configuration of the sweep:
```bash
python -m bllarse.tools.run_config bllarse_sweeps/probe_vbll.py 0
```
