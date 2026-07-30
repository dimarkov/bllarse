# Full-Network Finetuning Comparison: IBProbit vs CrossEntropy

**Date:** 2026-06-12
**Script:** `scripts/alternatives/full_finetune_comparison.py`

## Goal

Compare full-network finetuning of a pretrained dinov3_small backbone on
imagenet1k with and without the IBProbit Bayesian last layer, measuring test
ACC, ECE, and NLL after **every training step** for a fixed number of steps,
at the largest batch size that fits in GPU memory.

Linear-probing comparisons are already covered by
`scripts/alternatives/sample_efficiency.py`; this experiment finetunes the
**entire backbone**.

## Experimental setup

- **Backbone:** equimo `dinov3_vits16plus_pretrain_lvd1689m` (dinov3_small,
  embed_dim 384), loaded via `get_pretrained_backbone` conventions from
  `vit_classification.py`. Taken **out** of `inference_mode` for training;
  put back in inference mode for evaluation.
- **Dataset:** HF `ILSVRC/imagenet-1k`, 1000 classes, input resolution 224,
  imagenet mean/std normalization, CHW layout.
- **Two arms, run sequentially in one process:**
  1. **IBProbit:** `IBProbit(384, 1000)` head. Per step: extract features →
     CAVI update of the head (`loss_fn.update(features, labels,
     num_iters=--num-update-iters)`) → one AdamW gradient step on the
     backbone parameters through the updated IBProbit loss (`loss_type=3`).
     Mirrors `batch_update(update="both")` in `bllarse/utils.py`.
  2. **CrossEntropy:** `eqx.nn.Linear(384, 1000)` head appended to the
     backbone. Per step: one AdamW gradient step on backbone + head with
     `CrossEntropy(--label-smooth, 1000)`.
- **Fairness:** both arms use the same seed, the same shuffled data stream
  (identical batch order), the same probed batch size, and the same AdamW
  hyperparameters for the backbone. Memory is freed and `clear_caches()`
  called between arms.

## Data pipeline

Streaming, never materializing the dataset in memory: reuse
`iterate_batches` + `prefetch_iterator` from `vit_classification.py`
(thread-pool image decode/resize, per-batch normalization). If
`num_steps × batch_size` exceeds one epoch, reshuffle and continue
(not expected at default settings: 500 × 256 ≈ 10% of an epoch).

## Per-step evaluation

- A fixed random subset of the validation split (`--eval-subset`, default
  5000, seeded) is decoded/resized **once** at startup and held in host
  memory (~3 GB at 224 res float32); per-step eval runs it through the model
  in inference mode in chunks and computes ACC/ECE/NLL via
  `calibration.evaluate_classification`.
- `--eval-every` (default 1) controls evaluation frequency.
- `--final-full-eval` flag: after the last step, evaluate on the full 50k
  validation split (streamed).

## Batch-size auto-probe

At startup, starting from `--batch-size-init` (default 32), run one full
jitted train step of the **IBProbit arm** (the heavier one: forward + CAVI +
backward) on synthetic data, doubling the batch size until an
`XlaRuntimeError` / RESOURCE_EXHAUSTED is raised, then back off to the last
working size. Passing `--batch-size` explicitly skips the probe (preferred
for reproducible reruns).

## Logging

- Console: per-step line with train loss, acc, ece, nll.
- MLflow (`--log-mlflow`): one parent run, two nested child runs (one per
  arm), per-step metrics logged with `step=k`. Params logged: model,
  dataset, batch size, lr, weight decay, num_update_iters, label_smooth,
  seed, num_steps. Follows conventions in `vit_classification.py`
  (`--run-id` for parent grouping, `--experiment-name`).

## CLI

```
--model dinov3_small        --dataset imagenet1k       --img-size 224
--num-steps 500             --eval-every 1             --eval-subset 5000
--batch-size (optional)     --batch-size-init 32       --final-full-eval
--lr 1e-4                   --weight-decay 1e-4        --label-smooth 0.0
--num-update-iters 16       --seed 42                  --device gpu
--log-mlflow                --experiment-name bllarse  --run-id (optional)
--arm {both,ibprobit,ce}    (default both; allows running arms separately)
```

## Error handling

- OOM is only expected during the probe; it is caught and used as the
  stopping signal.
- Evaluation uses a fixed safe chunk size (e.g. 256) independent of the
  probed training batch size.
- equimo import failure raises with the same install hint as existing
  scripts.

## Testing

- Smoke test on CPU with a tiny config (`--num-steps 2 --eval-subset 64
  --batch-size 8`) on cifar10-sized fake data is impractical given imagenet
  download; instead, verify with `--num-steps 2 --eval-subset 256
  --batch-size 16` on the real dataset (already cached locally on the GPU
  machine) before launching the full run.
