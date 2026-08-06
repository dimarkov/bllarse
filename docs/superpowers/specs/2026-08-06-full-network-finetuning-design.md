# Full-network fine-tuning on ImageNet-1k — design

Date: 2026-08-06
Status: approved, not yet implemented
Covers: results-plan.md §2 (Figs 5–7, Table 1) and its Open question 1
("Is VBLL feasible at ImageNet full fine-tuning?")

## 1. Purpose

Prepare the infrastructure to fine-tune the whole DINOv3 trunk on ImageNet-1k
for three comparators — AdamW+CE (+TS), VBLL+AdamW, IBProbit+AdamW — under one
protocol, plus the Optuna tuning campaign that fixes their hyperparameters.

IBProbit's q change is in flight on a separate worktree. Nothing here blocks on
it: the arm is a flag, and the CE arm exercises every shared code path.

### What already exists

`scripts/alternatives/full_finetune_comparison.py` (1132 lines, on this branch
since 726bc78) already implements gradient accumulation with an auto-probed
micro-batch, the IBProbit CAVI-over-macro-batch head update, the CE arm with a
separate head learning rate, bf16 activations, ImageNet OOD evaluation, and the
memory work from cb81798 (donated `_tree_add`, host-parked `backbone_init`).

It is missing everything the Fig 1/2 protocol added afterwards:

- no pinned val/test split, therefore **no temperature scaling** — "+TS" is
  currently impossible
- no `evaldump` writes; metrics are computed inline, so there is no Brier, no
  post-hoc re-scoring at the val-optimal step, and no detection metrics
- `--num-steps` is a fixed count, not an epoch budget
- no VBLL arm, and it cannot have one — VBLL is PyTorch

## 2. Decisions

Recorded because several were contested during design and the reasoning matters
more than the choice.

**D1. VBLL trunk comes from HF `transformers` DINOv3, with a parity check.**
VBLL needs a *trainable* trunk and cannot reuse the JAX/equimo backbone. Same
released weights (`facebook/dinov3-*-pretrain-lvd1689m`), different
implementation, so parity is verified rather than assumed (§7). The alternative
— reimplementing `vbll.DiscClassification` in equinox — was rejected: it
replaces the reference implementation the paper cites with our own, which a
reviewer will ask about.

**D2. Shared module, thin arm scripts.** `full_finetune_comparison.py` keeps its
current job (IVON/EVON benchmarking, inline metrics) and is not rewritten.

**D3. Tuning runs on full train with truncated steps**, not a class-stratified
subset. Every trial sees the real data distribution.

**D4. Trial budget is fixed samples-seen, not fixed steps.** Follows from D3 plus
the 4-epoch final rule. A fixed *step* budget would hand large batches more data,
and the tuned batch size would not transfer to the final run.

**D5. Every hyperparameter is tuned per arm, including batch size.** Batch
tolerance is treated as a method property. Consequence for Fig 7: the arms are
compared at different batch sizes, so **the caption must report batch size per
arm** rather than implying a matched recipe. results-plan §5.4 warns that recipe
asymmetry invalidates the compute claim; the asymmetry it means is
implementation-level (precision, augmentation), which stays matched.

**D6. Head weight decay is 0 on all three arms.** Forced for VBLL — the KL *is*
the head regularizer and adding weight decay double-regularizes. Automatic for
IBProbit, which runs no optimizer on the head. Setting it to 0 for CE therefore
equalizes rather than adding an axis CE alone would carry.

**D7. Fine-tuning runs at img_size 224.** `DINOV3_DATASET_IMG_SIZES` maps
`imagenet1k -> 512`, but that table exists for frozen feature extraction.
Fine-tuning at 512 is ~5x the cost for no benefit. The scripts pin 224 and
**refuse 512 with an explicit error**, because silently inheriting that default
would be an extremely expensive accident.

## 3. Budget arithmetic

`N_train = 1,281,167`. The final run is capped at 4 epochs; steps follow from the
tuned batch size:

```
steps = epochs * N_train // effective_batch
```

| effective batch | steps @ 4 epochs |
| ---: | ---: |
| 512 | 10,009 |
| 1,024 | 5,004 |
| 2,048 | 2,502 |
| 4,096 | 1,251 |

This inverts the probing campaign, which fixed 30,000 steps and derived epochs
(`tuning/budgets.py`). The inversion is what makes Fig 5 valid: results-plan
already puts **samples seen** on its x-axis, so arms tuned to different batch
sizes remain directly comparable — which is also what makes D5 safe for Fig 5,
whatever it implies for Fig 7.

Tuning uses the same shape at a fraction of an epoch (`--epoch-fraction`,
default 0.5 -> 640,583 samples), steps again derived from batch size.

`effective_batch` is the tuned quantity. `micro_batch` is a memory artifact
absorbed by `num_accum = effective_batch / micro_batch` and auto-probed per arm,
so the PyTorch arm may use a different micro-batch than JAX without affecting
comparability.

## 4. Module layout

### `src/bllarse/finetune/`

- **`protocol.py`** — `steps_for_epochs(n_train, epochs, effective_batch)`;
  `accum_plan(target, micro) -> (micro, num_accum, effective)`; the dump-step
  schedule (wrapping `evaldump.dump_schedule`); and
  `shuffled_indices(n, seed, epoch)` built on numpy `default_rng`.

  The RNG choice is load-bearing: `full_finetune_comparison.py` guarantees that
  both arms "consume the identical shuffled data stream (same seed, same batch
  order)". Preserving that across two frameworks requires a permutation neither
  framework owns.

- **`jax_steps.py`** — the jitted kernels lifted out of
  `full_finetune_comparison.py`: `_tree_add`, `apply_grads`, `apply_grads_bayes`,
  `ce_micro_grad`, `ibprobit_micro_forward`, `ibprobit_micro_forward_nograd`,
  the HVP variants, and `forward_features`. `full_finetune_comparison.py` then
  imports them.

  A mechanical move, no behaviour change. Copying instead would leave two
  divergent versions of donation-tuned memory code — the same class of bug the
  shared `bllarse.metrics` import in `bllarse-torch.def` was added to prevent.

- **`imagenet.py`** — the pinned val/test partition over the official 50k
  validation split via `splits.load_or_make_pinned_split` (val_fraction 0.2 ->
  10k val / 40k test), and the OOD key registry.

### `scripts/finetune/`

- **`finetune_jax.py`** — `--arm ce|ibprobit`
- **`finetune_vbll.py`** — PyTorch
- **`check_backbone_parity.py`**, **`probe_throughput.py`** — §7

### `src/bllarse/tuning/`

- **`finetune_space.py`**, **`finetune_worker.py`** — beside the probing ones
- **`objective.py`** gains `seeds=` and `scripts=` kwargs, both defaulted so
  probing behaviour is bit-identical

## 5. Evaluation protocol

Every scheduled step writes val and test dumps through `evaldump.write_dump`,
exactly as the probing arms do. This is what buys:

- **`+TS` at zero GPU cost** — T is fit on the pinned val dump at each step and
  applied to that step's test logits, the same construction as
  `scripts/analysis/fig1_curves.py`
- Brier, and re-scoring at the val-optimal rather than the final step
- detection metrics post hoc, without re-running anything

**OOD at discrete steps.** OOD dumps ride the *same* dump schedule, gated to
every k-th dump (`--ood-every-k-dumps`, default 2 -> ~12 trajectory points).
Disabled entirely during tuning. This is the per-step generalization of the fix
already applied to the probing arms, where OOD was originally dumped only at the
final step and the CIFAR-100-vs-SVHN comparison was confounded by CE's
overfitting at that step.

`OOD_KEYS` currently holds 11 sets: 8 ImageNet-C corruptions plus R, A, V2.
**ImageNet-Sketch and ObjectNet, which results-plan Fig 6 asks for, are not
wired.** Out of scope here; noted so the gap is not discovered at plotting time.

Volume, at `n_dumps = 24`: val 20 MB + test 80 MB per step -> 2.4 GB, plus 11
OOD sets x 2000 images x 4 MB across 12 OOD evals -> ~530 MB. **~2.9 GB/run,
~44 GB for 15 runs** on `/ptmp`. Eval forward passes add ~10% over training
(1.46M eval forwards against 15.4M training forward-equivalents) and are
excluded from reported train time, as the existing script already does for its
OOD path.

## 6. The PyTorch port

Gradient accumulation itself is unremarkable: `(loss / num_accum).backward()` in
the micro loop, one `optimizer.step()` per macro batch, `torch.autocast` in
bfloat16 to match the JAX arm's activation precision. Two hazards deserve tests.

**H1. The KL term must be weighted exactly once.** `vbll`'s loss is
`NLL + regularization_weight * KL` computed *per micro-batch*, and
`regularization_weight = kl_ratio * prior_scale / N`. Averaging over micro-batches
leaves the KL correctly weighted, since it is constant across them. Summing
without dividing multiplies it by `num_accum` — up to 64x — silently retuning
`kl_ratio` by more than the entire range the probing campaign searched.

Pinned by a test asserting that one accumulated step equals one full-batch step
to tolerance.

**H2. Head weight decay stays 0 while the trunk gets `--weight-decay`** (D6). The
existing `finetuning_vbll.py` already splits param groups for this; the port
keeps that structure and adds a per-group `lr`.

`bllarse-torch.def` gains `transformers`. The DINOv3 repos are gated, so the
existing `HF_TOKEN` is required.

## 7. Verification before compute

Both probes run before any real training. Each is minutes; the campaign is
hundreds of A100-hours.

**Backbone parity.** Forward the pinned eval set through the PyTorch DINOv3 and
compare against the JAX `.npz` cached features: per-sample cosine similarity,
plus delta linear-probe accuracy. Fails loudly on divergence. Pooling and
final-LayerNorm differences between implementations are the likely failure and
are cheap to find this way rather than after a 4-hour run.

**Throughput.** Measure decode img/s and train img/s *separately*. The current
data path is PIL in a `ThreadPoolExecutor`; ViT-L at 224 needs ~350 img/s
sustained and it is not yet known whether the loader clears it. If it does not,
the fix is pre-resized 224px shards.

## 8. Tuning

Optuna TPE on `JournalStorage`/`JournalFileBackend`, four workers per study, as
the probing campaign already does. Reused unchanged: `open_study`, the journal
storage, `TPESampler`, `--campaign` namespacing, `summarize_study`'s
"every trial failed is a harness fault" guard, `catch=(Exception,)`, and
`src/slurm/jobs/slurm_tune.sh` (env-var parameterized, SIF selected by
`BLLARSE_SIF`).

The probing campaign's *settings* do not transfer:

| setting | probing | fine-tuning | why |
| --- | --- | --- | --- |
| seeds/trial | 2 (`TUNING_SEEDS`) | 1 | right for a 40-second probe; here it doubles tuning to ~92 A100-h |
| `MedianPruner` startup | 20 trials | 5 | 20 was a fifth of ~80 probing trials; at ~30 trials it would barely engage, and pruning is where the savings are at 30 min/trial |
| `patience` | on | off | existed to stop *converged* trials against a 30k-step cap; a 0.5-epoch trial is truncated by construction, so patience fires on noise or never |
| batch ladder | `batch_cap(N_TRAIN)` | 256…4096 | probing's cap enforces >=2 steps/epoch, a *data*-size constraint that is meaningless at 1.28M images — here the ceiling is memory |

### Search space

Shared: `trunk_lr`, `trunk_weight_decay`, `batch_size` — tuned independently per
arm (D5).

**Learning rates are searched as trunk rate and ratio, not two independent
rates.** `full_finetune_comparison.py`'s own comment records why they are
correlated: the fresh head needs a far higher rate because "AdamW moves weights
by ~lr/step, and the head's random init scale (~0.015) would take thousands of
1e-4 steps to overwrite" — its defaults are 1e-4 / 1e-2, a ratio of 100. Two
independent log-uniform axes spend trials sliding along that diagonal. So:
`trunk_lr in [1e-6, 1e-3]`, `head_lr_ratio in [1, 1e3]`, both log. This is the
same reparameterization `space.py` already applies for `kl_ratio` over
eta/prior_scale.

Per-arm head axes:

- **CE** — `head_lr_ratio`. 4 axes total.
- **VBLL** — `head_lr_ratio`, `kl_ratio`, `prior_scale`, `wishart_scale`. 7 axes.
- **IBProbit** — **no head learning rate exists.** `run_arm` sets `model = None`
  so `optimizer.init` sees only the backbone; the head is assimilated in closed
  form by `loss_fn.update(...)`. Its head knobs are CAVI-side
  (`num_update_iters`, and whatever the q change settles). ~4 axes.

  That absence is a result, not a gap: "no head learning rate to tune" is the
  closed-form-conjugate advantage Fig 7 exists to measure, and belongs in the
  text.

### Trial budget

**VBLL gets ~50 trials; CE and IBProbit ~30.** Spreading one budget over seven
axes under-tunes VBLL, and under-tuning VBLL is the dangerous direction:
REWRITE-PLAN Risk 1 gates submission on a matched-recipe VBLL comparison, so
"you under-tuned the baseline" is the review to preempt. ~10 extra A100-hours.

**VBLL's box is narrowed using the completed probing campaign.** That campaign
measured `kl_ratio` optima spanning 2.3e-7 to 2.5e-3; re-searching `[1e-7, 1e-2]`
blind discards it. Same for `prior_scale` and `wishart_scale`. This is a
documented prior from a finished experiment, and it buys back most of the
7-axis spread.

Capacity knobs stay fixed, following `space.py`'s existing rule: `cov_rank` and
`parameterization` are not searched, because neither CE nor IBProbit has an
equivalent knob and letting TPE spend it would let VBLL win partly by being
larger.

## 9. Cost

At ~40% MFU on one A100, img_size 224:

| backbone | per 4-epoch run | tuning | full sweep (3 arms x 5 seeds) | total |
| --- | ---: | ---: | ---: | ---: |
| dinov3_large (304M) | ~4.1 h | ~56 h | ~61 h | **~117 A100-h** |
| dinov3_huge (840M) | ~11.4 h | ~156 h | ~171 h | **~327 A100-h** |

Tuning assumes 30/50/30 trials at 0.5 epoch. Seeds follow results-plan §0.5
(5 for full fine-tuning).

Backbone rung and seed count are sweep-file constants, not architecture.
Defaults: `dinov3_large`, 5 seeds. Both are a ~3x compute swing and are set at
launch, not here.

## 10. Deferred

- ImageNet-Sketch and ObjectNet OOD sets (results-plan Fig 6)
- The FLOP model (results-plan §4.1), which decides whether Fig 7 runs at all
- Folding `pytest`/`mlflow-skinny` into the image definitions instead of the
  `/ptmp/cheins/pylibs-*` workarounds
