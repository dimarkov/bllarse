# bllarse

Bayesian last layers ("heads") on top of pretrained vision backbones. This is
the code for comparing an IBProbit head with AdamW + cross-entropy (with
temperature scaling) and with VBLL, in two settings: linear probing on frozen
DINOv3 features, and full-network fine-tuning on ImageNet-1k. Accuracy,
calibration and OOD detection are all computed after training, from saved
logits.

## Setup

```bash
uv sync                 # the JAX library and the AdamW+CE / IBProbit arms
uv sync --group vbll    # adds PyTorch and `vbll` for the VBLL arm
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

## Linear probing

Probing runs on cached DINOv3 features. By default they come from the public
Hugging Face dataset `dimarkov/bllarse-features`. To compute them yourself
instead, run `scripts/dinov3/precompute_features.py` (and
`precompute_ood_features.py` for the OOD sets) with `--no-hf-cache`.

A sweep file defines `create_configs() -> list[dict]` and `run(config)`. To run
one of its configurations, pass its index:

```bash
python -m bllarse.tools.run_config bllarse_sweeps/probe_vbll.py 0
```

The hyperparameters in the sweep files come from Optuna TPE studies, run with
`python -m bllarse.tuning.worker`.

## Full-network fine-tuning

Each script below documents its arguments in its module docstring. In order:

1. `scripts/finetune/build_shards.py` decodes each ImageNet-1k split once into
   a uint8 shard that every run reads.
2. `scripts/finetune/prepare_ood_cache.py` downloads the OOD sets.
3. `python -m bllarse.tuning.finetune_worker` tunes each arm.
4. `scripts/finetune/final_config.py` prints the command line for the final
   runs from a finished study.
5. `scripts/finetune/finetune_jax.py --arm ce|ibprobit` and
   `scripts/finetune/finetune_vbll.py` train the three arms.
6. `scripts/finetune/ood_worker.py` scores the OOD sets from checkpoints.

## Results and logging

Every training script writes logit dumps under `--dump-root`, and every
reported metric is computed from those dumps. MLflow logging is optional. A
config turns it on with `enable_mlflow=True`, and it logs to
`MLFLOW_TRACKING_URI` if that is set, or to a local store if not.

## From dumps to figures

In the commands below, `$DUMPS` is the dump root, and `$GROUP` is the
`--group-id` that the fine-tuning runs used.

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
