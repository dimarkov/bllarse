# flowers102 pool swap, temperature-fit guard, and a widened VBLL prior

**Date:** 2026-08-03
**Status:** design, approved
**Scope:** Corrections to the 2026-08-02 tuning campaign, and a partial re-run
(7 VBLL studies + AdamW+CE flowers102). The 6 other AdamW+CE studies stand.

## Problem

The campaign completed on 2026-08-03 — 14 studies, 1,425 trials, all 56 array
tasks COMPLETED. Three defects make its output unfit to export as
`tuned_hparams.json`.

**1. `prior_scale` is a boundary solution on all seven VBLL studies.** The search
range is `[1e-2, 1e2]`; the ten best trials of every study sit at 94–100.

| study | prior_scale of the 10 best trials |
|---|---|
| vbll/cifar10 | 65.2, 95.6, 96.0, 96.6, 96.9, 97.6, 97.9, 98.6, 99.1, 99.8 |
| vbll/dtd | 94.5, 96.9, 97.7, 97.7, 97.7, 97.7, 97.9, 98.8, 99.4, 99.7 |
| vbll/food101 | 94.0, 96.3, 96.6, 97.4, 98.0, 98.1, 98.2, 98.3, 98.8, 98.9 |
| vbll/stanford_cars | 97.4, 97.4, 97.9, 98.3, 99.0, 99.3, 99.6, 99.7, 99.7, 100.0 |

TPE is pushing the KL term toward zero — VBLL wants to become CE — and the
ceiling stops it. Every reported VBLL number is therefore limited by the search
space rather than by the method, which is the first thing a reviewer attacks.

**2. flowers102's eval pool is 1,020 images, so its val split is 102 examples
for 102 classes — one per class.** The HF mirror in use, `nelorth/oxford-flowers`,
has `train=7,169` / `test=1,020`, the inverse of the official Flowers-102
protocol (`train=1,020` / `test=6,149`). Every other dataset's eval pool is
≥1,880:

| dataset | train pool | eval pool | val / test |
|---|--:|--:|--:|
| **flowers102** | 7,169 | **1,020** | **102 / 918** |
| dtd | 3,760 | 1,880 | 188 / 1,692 |
| oxford_pets | 3,680 | 3,669 | 367 / 3,302 |
| stanford_cars | 8,144 | 8,041 | 804 / 7,237 |
| cifar10 / cifar100 | 50,000 | 10,000 | 1,000 / 9,000 |
| food101 | 75,750 | 25,250 | 2,525 / 22,725 |

Two failures follow from those 102 points. Selection is noise — the eight best
AdamW+CE trials span `9.3e-6` to `1.4e-4`, all effectively zero. And the
temperature fit degenerates: at the early-stopped step the fit returns
`T = 0.0492`, sharpening logits 20.3x. With `test_acc = 0.9946`, the 0.54% of
test examples that are wrong are charged ~421 nats each, so `test_ts_nll` is
**2.2907** against a raw `test_nll` of 0.1135. That `test_ts_ece` (0.0049) almost
exactly equals the error rate (0.0054) confirms post-scaling confidence is ~1.0
on every example: the mean NLL is entirely the errors.

The same instability hits VBLL on flowers102 — `T = 0.0116` at step 569, which is
the `beta = 100` clip — but its selected step happened to be benign
(`T = 0.362`), so its `test_ts_nll` of 0.0842 looks good. The resulting
"VBLL beats CE by 2.21 nats" in the temperature-scaled column is an artifact of
which step each arm landed on, not a result.

**3. A clipped temperature fit is indistinguishable from a converged one.**
`metrics.fit_temperature` clips `beta` to `[0.01, 100.0]` inside its Newton loop
and returns `1/beta`. When the clip binds, the returned temperature is a
boundary value, not an optimum, and nothing downstream can tell.

## Non-goals

- The `stanford_cars` VBLL collapse (test_acc 0.5168 vs CE's 0.8921, systematic
  across all 79 completed trials). The widened prior range tests whether it is a
  boundary artifact for free. If it survives to `prior_scale = 1e3` it is
  structural and gets its own spec.
- Re-running the 6 AdamW+CE studies that are not flowers102. Neither change
  touches them.
- Task 10's `export.py` and the Stage 4/5 work. They resume once this lands.
- The 1,756 untagged flowers102 runs in the `bllarse` experiment (2026-02-27 to
  2026-06-18). Predate this protocol; left alone.

## Design

### Pool swap for flowers102

`features.eval_split_name(dataset)` is the existing indirection for "which cached
split is the eval pool" (`food101 -> validation`). It gains a counterpart,
because the train pool is currently hardcoded as `"train"` at every call site.

- `features.train_split_name(dataset) -> str`, defaulting to `"train"`, with one
  override: `flowers102 -> "test"`.
- `_EVAL_SPLIT` gains `flowers102 -> "train"`.

The swap is pure relabeling of already-cached files. Both
`dinov3_*_flowers102_res512_train.npz` (7,169) and `_test.npz` (1,020) exist for
every backbone, so **no feature re-extraction is required**.

Downstream constants that must move in lockstep:

| thing | before | after |
|---|--:|--:|
| `space.N_TRAIN["flowers102"]` | 7,169 | 1,020 |
| `batch_cap` | 2,048 | **256** |
| `space.batch_sizes("flowers102")` | 64…2,048 | `[64, 128, 256]` |
| val / test | 102 / 918 | **717 / 6,452** |
| `epoch_budgets.json` batch_size | 512 | 256 |
| `epoch_budgets.json` epochs | 2,143 | 10,000 |

`batch_cap` is 256, not 512: `1020 // 2 = 510` and 512 exceeds it, which would
also put flowers102 at 1 step/epoch, below the `MIN_STEPS_PER_EPOCH = 2` floor.
The step budget stays at 30,000, unchanged from every other dataset — but at
1,020 training images and 3 drop-remainder steps per epoch that is 10,000
epochs, so flowers102 will lean on the patience rule (20 dumps) in a way no
other dataset does.

The pinned split cache is keyed by split name (`{dataset}_{split}_val{f}.npz`),
so the new pool writes `flowers102_train_val0.1.npz` and the stale
`flowers102_test_val0.1.npz` is simply never read. `load_or_make_pinned_split`
already fails loudly on size drift, so there is no silent-reuse path.

### Temperature-fit guard

`fit_temperature`'s semantics do not change. Constraining `T >= 1` (never
sharpen) would be a methodological change to a headline arm, and with a
717-example val set the degenerate fit should stop occurring on its own. The
guard is a tripwire, not a fix.

`fit_temperature`'s signature and return type are unchanged, so no caller
breaks. The clip bounds are promoted to module constants and a predicate reads
the result:

- `metrics.BETA_BOUNDS = (0.01, 100.0)`, with the Newton loop's `jnp.clip` call
  reading from it rather than restating the literals.
- `metrics.is_clipped_temperature(temperature: float) -> bool` — true when the
  returned `T` sits within float tolerance of either `1/beta` bound.

`reporting.step_metrics` is the single production call site
(`reporting.py:31-32`); both arms reach temperature scaling through it. It gains
`out["temperature_clipped"] = float(is_clipped_temperature(temperature))`, so
the flag appears as a per-step MLflow metric next to `temperature` for every run
of both arms, with no change to either training script.

`analysis.curves.run_curve` is the second call site (`curves.py:70`) and adds
the same key to the dict it already builds at `curves.py:78`, so the flag is
available post hoc for runs analysed from dumps.

### Search space

`space.suggest`: `prior_scale` upper bound `1e2 -> 1e3`. `wishart_scale` is
unchanged — its selected values span 0.003–3.2, comfortably interior to
`[1e-3, 1e1]`.

### Campaign namespacing

The re-run must not collide with the first campaign. `study_name()` returns
`f"{arm}_{dataset}"`, the journal file is `{study_name}.log`, and `open_study`
passes `load_if_exists=True`. Re-running as-is would **resume** each existing
study, mixing trials sampled under `prior_scale <= 1e2` with new ones — and for
flowers102, mixing trials whose `batch_size` (512, 1024, 2048) is no longer in
the search space at all. In MLflow the collision is worse still: same
`group_id`, same `run_name` (`trial00000_seed200`), so `search_runs` would match
across campaigns arbitrarily.

A campaign suffix threaded through both names solves all of it in one move:

- `study_name(arm, dataset, campaign=None)` returns `f"{arm}_{dataset}"` when
  `campaign` is None and `f"{arm}_{dataset}_{campaign}"` otherwise.
- `group_id` in the trial config takes the same suffix (`tune_{study_name}`).
- The re-run uses `campaign="r2"`.

A new study name means a new journal file, so no journals need moving and the
first campaign survives intact as a comparison record.

### MLflow cleanup

Before the re-run, delete the 359 flowers102 runs in experiment 13
(`bllarse-probing-tuning`):

| group_id | runs |
|---|--:|
| `tune_vbll_flowers102` | 211 |
| `tune_adamw_ce_flowers102` | 132 |
| `probe_pilot_epochs` (flowers102 only) | 12 |
| parent runs matched by name | 4 |

They are measurements of a val split that no longer exists, and the pilot is
stale in the same way. `delete_run` is a soft delete — `lifecycle_stage` becomes
`"deleted"` and `restore_run` reverses it until someone runs `mlflow gc`
server-side.

The deletion script must select on `group_id`, never on `run_name` alone, and
must be scoped to experiment 13 explicitly. It reports a dry-run count first and
requires an explicit confirmation flag to act.

## Re-run

8 studies, 4 workers each = 32 array tasks:

- 7 VBLL studies (`prior_scale` range changed)
- AdamW+CE flowers102 (pools changed)

The first campaign's other 6 AdamW+CE studies are unaffected and their results
carry forward, so `export.py` will read 6 studies from campaign 1 and 8 from
campaign r2.

## Testing

- `batch_cap(1020) == 256`, and `batch_sizes("flowers102") == [64, 128, 256]`.
- `train_split_name("flowers102") == "test"` and
  `eval_split_name("flowers102") == "train"`; both default to their old values
  for every other dataset.
- The pinned split for the new flowers102 pool is 717 / 6,452.
- `is_clipped_temperature` is true on logits that drive `beta` to either bound
  and false on a well-conditioned fit; `step_metrics` carries the flag through
  to its returned scalars.
- `study_name("vbll", "dtd")` is unchanged; `study_name("vbll", "dtd", "r2")`
  is `vbll_dtd_r2`, and the two resolve to different journal files.
- `suggest` never proposes `prior_scale > 1e3` and can propose `> 1e2`.
- The deletion script, against a fake client, selects exactly the four
  flowers102 group_ids in experiment 13 and nothing in experiment 1.
