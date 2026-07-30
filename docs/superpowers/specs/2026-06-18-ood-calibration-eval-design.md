# OOD Calibration Evaluation: IBProbit vs CrossEntropy

**Date:** 2026-06-18
**Scripts:**
- `scripts/alternatives/precompute_ood_features.py`
- `scripts/alternatives/ood_evaluation.py`
- `scripts/alternatives/get_best_hparams.py` (helper)
- `scripts/alternatives/run_ood_eval_sweep.sh`

## Goal

Compare the out-of-distribution (OOD) calibration of IBProbit vs CrossEntropy
last-layer heads trained on ImageNet-1k features from a frozen DINOv3 backbone.
Heads are trained once on ImageNet-1k train features (with fixed best
hyperparameters), then evaluated — without retraining — on ImageNet-1k
validation (in-distribution) and a suite of OOD test sets, reporting test ACC,
ECE, and NLL on each.

This complements the in-distribution sweeps (`run_vit_classifciation_*_sweep.sh`)
by measuring how each head's calibration degrades under distribution shift.

## Scope

- **Backbone:** `dinov3_huge` (equimo `dinov3_vith16plus_pretrain_lvd1689m`,
  embed_dim 1280). The only model (besides `dinov3_small`) with cached
  ImageNet-1k features for the planned comparison; `dinov3_small` features
  exist too, so both scripts are parameterized by `--model`.
- **Resolution:** 512px (`res512`) — matches the cached ImageNet-1k features
  the head is trained on (`DINOV3_DATASET_IMG_SIZES["imagenet1k"] == 512`).
  All OOD features MUST be extracted at the same resolution with the same
  backbone so they live in the same feature space.
- **Seeds:** 137, 139, 141, 143, 145 (same as existing sweeps).
- **Two methods:** IBProbit and CrossEntropy.

## OOD datasets

All are ImageNet-derived, evaluation-only, normalized with ImageNet mean/std
(same as the ImageNet-1k training features). Each is reduced to a uniform
`{image, label}` schema where `label` is the **canonical ImageNet-1k class
index** (0–999). The canonical index ordering is the sorted-WNID order, which
equals `ILSVRC/imagenet-1k`'s native label order; it is obtained at runtime
from a Tsomaros ImageNet-C `ClassLabel.names` (1000 sorted WNIDs).

| OOD key | HF source | split | image key | label derivation | classes |
|---|---|---|---|---|---|
| `imagenet_c_brightness` | `Tsomaros/ImageNet-C-brightness-severity_5` | `validation` | `image` | `ClassLabel` index = canonical | 1000 |
| `imagenet_c_fog` | `Tsomaros/ImageNet-C-fog-severity_5` | `validation` | `image` | `ClassLabel` index = canonical | 1000 |
| `imagenet_c_gaussian_noise` | `Tsomaros/ImageNet-C-gaussian_noise-severity_5` | `validation` | `image` | `ClassLabel` index = canonical | 1000 |
| `imagenet_c_jpeg_compression` | `Tsomaros/ImageNet-C-jpeg_compression-severity_5` | `validation` | `image` | `ClassLabel` index = canonical | 1000 |
| `imagenet_c_motion_blur` | `Tsomaros/ImageNet-C-motion_blur-severity_5` | `validation` | `image` | `ClassLabel` index = canonical | 1000 |
| `imagenet_c_pixelate` | `Tsomaros/ImageNet-C-pixelate-severity_5` | `validation` | `image` | `ClassLabel` index = canonical | 1000 |
| `imagenet_c_shot_noise` | `Tsomaros/ImageNet-C-shot_noise-severity_5` | `validation` | `image` | `ClassLabel` index = canonical | 1000 |
| `imagenet_c_zoom_blur` | `Tsomaros/ImageNet-C-zoom_blur-severity_5` | `validation` | `image` | `ClassLabel` index = canonical | 1000 |
| `imagenet_r` | `axiong/imagenet-r` | `test` | `image` | `wnid` string → canonical index | 200-subset |
| `imagenet_a` | `barkermrl/imagenet-a` | `train` | `image` | `ClassLabel` name (WNID) → canonical index | 200-subset |
| `imagenet_v2` | `vaishaal/ImageNetV2` | `train` | `jpeg` | `int(__key__.split("/")[1])` (already canonical) | 1000 |

**ImageNet-C scope:** severity 5 only, the 8 corruptions available as clean
per-corruption HF datasets from `Tsomaros/*`. (The full 15-corruption set only
exists as an opaque webdataset; deferred.)

**200-class subsets (R, A):** ImageNet-R and ImageNet-A cover 200 of the 1000
classes. Their labels are mapped into the canonical 1000-index space. At
evaluation, logits are **masked to the 200-class subset** (non-subset logits
set to `-inf`) before computing ACC/ECE/NLL, the standard protocol. The subset
index set for each is recorded so the eval script can apply the mask.

## Script 1: `precompute_ood_features.py`

Mirrors the structure of `precompute_features.py`. Reuses
`get_pretrained_backbone`, `load_or_compute_features`, `upload_features_to_hf`,
`IMAGENET_MEAN`, `IMAGENET_STD`, and the 3-tier cache (local → HF → compute)
from `vit_classification.py`.

- Defines `OOD_CONFIGS`: for each OOD key, an HF path, split, image key,
  number of classes, and a `prepare(raw_split, wnid_to_idx)` callable that
  returns a HF dataset (or wrapper) exposing a standard `image` column and an
  integer `label` column in canonical index space. The `prepare` step uses
  HF `dataset.map`/`rename_column`/`cast_column` to normalize:
  - ImageNet-C: rename label column if needed; `ClassLabel` int is already
    canonical (verified: `names[i]` is the i-th sorted WNID).
  - ImageNet-R: map `wnid` → `wnid_to_idx[wnid]`.
  - ImageNet-A: map `ClassLabel` int → its WNID name → `wnid_to_idx`.
  - ImageNet-V2: parse `__key__` middle component to int; rename `jpeg`→`image`.
- Loads the canonical `wnid_to_idx` once from a Tsomaros `ClassLabel`.
- Cache path: `.cache/features/{model}_{ood_key}_res{img}_test.npz`.
- HF path: `{model}/{ood_key}_res{img}_test.npz` in `dimarkov/bllarse-features`.
- CLI mirrors `precompute_features.py`: `--model` (default `dinov3_huge`),
  `--ood-keys` (default all), `--cache-dir`, `--batch-size`, `--seed`,
  `--device`, `--no-cache`, `--hf-repo`, `--no-hf-cache`.
- Records the R/A subset index sets to a small JSON sidecar (or computes them
  deterministically from `OOD_CONFIGS`) for the eval script's masking.

This script is **independent of any head hyperparameters** and can be run
immediately to prepare and upload all OOD features.

## Script 2: `ood_evaluation.py`

For the chosen model, for each seed, for each method:

1. **Load ImageNet-1k train + validation features**
   (`{model}/imagenet1k_res512_{train,validation}.npz`) via the existing cache.
2. **Train one head** with fixed best hyperparameters (see below), reusing
   `CrossEntropy`, `IBProbit`, and `iterate_feature_batches` from
   `vit_classification.py` (with local epoch helpers that show a per-batch
   progress bar):
   - CrossEntropy: `eqx.nn.Linear(1280, 1000)` + AdamW.
   - IBProbit: `IBProbit(1280, 1000)`, CAVI updates.
3. **Model selection.** Because the OOD sets are the held-out test data, the
   head trains on the **full** ImageNet-1k train set (no train/val holdout) and
   early stopping — when enabled — selects the lowest-**ImageNet-1k-validation**-NLL
   epoch. Both methods use this identical protocol. Early stopping engages only
   for multi-epoch runs (`--epochs > 1` with `--patience`); the default
   single-pass IBProbit (`epochs=1`) trains once with no selection. Note that
   `imagenet1k_val` therefore serves both selection and the reported
   in-distribution metric (mildly optimistic); all OOD metrics stay unbiased.
4. **Evaluate** the selected head on ImageNet-1k val + all 11 OOD sets via
   `evaluate_features`, loading each feature file from the cache. For R/A, a
   `subset` mask sets out-of-subset logits to a large negative value before
   `evaluate_classification`.
5. **Log** ACC/ECE/NLL per dataset to MLflow (see structure below).

### Fixed best hyperparameters

Read at run time from the existing sweeps via `get_best_hparams.py` (no
hardcoding), so the values auto-update as sweeps complete:

- **IBProbit:** best `(batch_size, num_update_iters)` by mean `test_nll` for
  `(model, imagenet1k)` under `sweep5a-alt-classification-dinov3-IBProbit`.
  For `dinov3_huge` this is currently `batch_size=16384, num_update_iters=256`.
- **CrossEntropy:** best `(batch_size, lr)` by mean `best_test_nll` for
  `(model, imagenet1k)` under `sweep10a-alt-classification-dinov3-CrossEntropy`.
  For `dinov3_huge` this is **pending** — the CE sweep is mid-run and has not
  yet reached `dinov3_huge/imagenet1k`. The shell runner blocks/falls back
  until it is available (see Open items).

CLI flags allow overriding any hyperparameter manually.

## Helper: `get_best_hparams.py`

Small MLflow query script (sibling of `get_mlflow_run.py`). Given a parent run
name, model, dataset, and method, prints the best hyperparameter combo (lowest
mean selection-NLL across seeds) as shell-consumable `KEY=VALUE` lines. Used by
the shell runner to fetch CE and IBProbit hyperparameters.

## Script 3: `run_ood_eval_sweep.sh`

Mirrors `run_vit_classifciation_*_sweep.sh`. For each method:

- Get/create a parent run `ood-eval-{model}-{method}` via `get_mlflow_run.py`.
- Fetch best hyperparameters via `get_best_hparams.py`.
- For each seed, invoke `ood_evaluation.py` with `--run-id <parent>`,
  `--method`, `--model`, `--seed`, and the fetched hyperparameters.

## MLflow logging

- **Experiment:** `bllarse` (existing).
- **One parent run per method:** `ood-eval-{model}-CE`, `ood-eval-{model}-IBProbit`.
- **One child run per seed** (per the chosen structure), tagged with
  `mlflow.parentRunId`. Each child logs params (`model`, `method`, `seed`, the
  fixed hyperparameters) and metrics named `{dataset}_{acc,ece,nll}`, e.g.
  `imagenet1k_val_acc`, `imagenet_c_fog_ece`, `imagenet_r_nll`,
  `imagenet_a_acc`, `imagenet_v2_nll`. Optionally a few aggregates:
  `imagenet_c_mean_{acc,ece,nll}` averaged over the 8 corruptions.

## Open items

- **CE hyperparameters for `dinov3_huge`** depend on the in-progress CE sweep
  reaching `dinov3_huge/imagenet1k`. Feature precompute (Script 1) and the
  IBProbit arm are unblocked and can proceed now; the CE arm runs once the
  sweep result is available (the user is currently running that sweep).
