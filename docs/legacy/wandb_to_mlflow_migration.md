# Legacy: W&B to MLflow Migration

This note documents the one-time migration workflow used to back up historical
Weights & Biases sweeps into MLflow parent/child runs.

Status:
- Historical only.
- Not part of normal day-to-day experiment flow.
- Script retained at `src/bllarse/tools/migrate_wandb_to_mlflow.py`.

## What The Migration Script Does

- Creates (or reuses) one MLflow parent run per W&B sweep/group.
- Migrates only W&B runs in `finished` state.
- Creates one MLflow nested child run per finished W&B run.
- Copies W&B config fields into MLflow params.
- Copies metric traces for `ece`, `nll`, and `acc`.
- Copies W&B metadata tags including run id/name/url and commit SHA (when available).

## Historical Usage

```bash
./start_docker_locally.sh bash -lc '
  source .venv/bin/activate
  python -m bllarse.tools.migrate_wandb_to_mlflow \
    --wandb-entity verses_ai \
    --wandb-project bllarse_experiments \
    --sweep sweep2a2_batchsize_by_optimizer_by_lr \
    --sweep sweep7a1_fnf_dataaug_adamw_batchsize_numiters
'
```

Optional dry run:

```bash
./start_docker_locally.sh bash -lc '
  source .venv/bin/activate
  python -m bllarse.tools.migrate_wandb_to_mlflow \
    --wandb-entity verses_ai \
    --wandb-project bllarse_experiments \
    --sweep sweep2a2_batchsize_by_optimizer_by_lr \
    --sweep sweep7a1_fnf_dataaug_adamw_batchsize_numiters \
    --dry-run
'
```
