# VBLL prior reparameterization, flowers102 pool swap, and a temperature-fit guard

**Date:** 2026-08-03 (revised 2026-08-04 after the prior/eta sweeps)
**Status:** design, approved
**Scope:** Corrections to the 2026-08-02 tuning campaign, and a partial re-run
(7 VBLL studies + AdamW+CE flowers102). The 6 other AdamW+CE studies stand.

## Problem

The campaign completed on 2026-08-03 — 14 studies, 1,425 trials, all 56 array
tasks COMPLETED. Four defects make its output unfit to export as
`tuned_hparams.json`.

### 1. The VBLL search box excluded the useful region by three orders of magnitude

`prior_scale` is a boundary solution on all seven VBLL studies. The search range
is `[1e-2, 1e2]`; the ten best trials of every study sit at 94–100.

| study | prior_scale of the 10 best trials |
|---|---|
| vbll/cifar10 | 65.2, 95.6, 96.0, 96.6, 96.9, 97.6, 97.9, 98.6, 99.1, 99.8 |
| vbll/dtd | 94.5, 96.9, 97.7, 97.7, 97.7, 97.7, 97.9, 98.8, 99.4, 99.7 |
| vbll/food101 | 94.0, 96.3, 96.6, 97.4, 98.0, 98.1, 98.2, 98.3, 98.8, 98.9 |
| vbll/stanford_cars | 97.4, 97.4, 97.9, 98.3, 99.0, 99.3, 99.6, 99.7, 99.7, 100.0 |

Three diagnostic sweeps on Raven (jobs 29117557, 29118348, 29120274; 30,000
steps, each dataset's own best campaign trial, prior axis varied alone) explain
why, and the explanation is sharper than "the ceiling is too low".

vbll's loss (`classification.py:169`) is

```python
total_elbo  = torch.mean(self.softmax_bound(x, y))
total_elbo += self.regularization_weight * (wishart_term - kl_term)
```

with `regularization_weight` pinned at `1.0 / len(train_f)` — call it `eta = 1`,
the textbook ELBO. Writing it as `eta / N` exposes a second axis, and the two
axes turn out to control **different things**:

- `gaussian_kl`'s mean penalty is `‖m‖² · eta / (N · prior_scale)`, so mean
  shrinkage — and therefore fit — depends only on the **ratio** `eta / prior_scale`.
- The KL is stationary in `σ²` at `σ² = prior_scale` whatever `eta` is, so
  `prior_scale` alone sets the **posterior width**, as `sd ∝ √prior_scale`.

The ratio claim is exact in the measurements. Every ratio-matched pair coincides:

| `eta / prior_scale` | config | train | test |
|--:|---|--:|--:|
| 1e-3 | `(q=1, eta=0.001)` | 0.9519 | 0.8224 |
| 1e-3 | `(q=100, eta=0.1)` | 0.9531 | 0.8237 |
| 1e-2 | `(q=1, eta=0.01)` | 0.6426 | 0.5386 |
| 1e-2 | `(q=100, eta=1.0)` | 0.6485 | 0.5408 |

At fixed ratio the two members differ 100× in `prior_scale` and 10× in posterior
sd (0.103 vs 0.991), confirming the width/fit separation.

**The campaign fixed `eta = 1` and searched `prior_scale ∈ [1e-2, 1e2]`, so the
reachable ratio was `[1e-2, 1e2]`. The useful ratio is ~1e-5.** The box's best
corner is ratio 1e-2, which reproduces the campaign's stanford_cars result
exactly (0.5408 in the sweep vs 0.5168 logged). TPE pinned `prior_scale` at the
ceiling on all seven datasets because it was pressing against the wall of a box
that did not contain the answer.

Response to the ratio, on stanford_cars (plain Linear+CE control: train 0.9984 /
test 0.8893):

| ratio | train | test | val_nll |
|--:|--:|--:|--:|
| 1e-5 | 0.9985 | **0.8867** | 0.3767 |
| 1e-4 | 0.9958 | 0.8752 | 0.5029 |
| 1e-3 | 0.9531 | 0.8237 | 1.1405 |
| 1e-2 | 0.6485 | 0.5408 | 2.9631 |

At ratio 1e-5, VBLL matches its CE control on both datasets tested — and
oxford_pets beats it (0.9397 vs 0.9358). `eta` has a genuine interior optimum:
`eta = 0` gives `val_nll` 0.7876, `eta = 0.001` gives 0.3767, `eta = 0.003`
gives 0.4104. A small posterior is buying something real; the campaign's
`eta = 1` was 1,000× too large.

This also closes the stanford_cars investigation. `eta = 0` reproduces the
deterministic control almost exactly (train 0.9985 vs 0.9984, test 0.8819 vs
0.8893, posterior sd 0.001), so the VBLL implementation is sound and the
"collapse" was the operating point, not a bug.

### 2. flowers102's eval pool is 1,020 images, so its val split is 102 examples for 102 classes

The HF mirror in use, `nelorth/oxford-flowers`, has `train=7,169` / `test=1,020`,
the inverse of the official Flowers-102 protocol (`train=1,020` / `test=6,149`).
Every other dataset's eval pool is ≥1,880:

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

### 3. A clipped temperature fit is indistinguishable from a converged one

`metrics.fit_temperature` clips `beta` to `[0.01, 100.0]` inside its Newton loop
and returns `1/beta`. When the clip binds, the returned temperature is a
boundary value, not an optimum, and nothing downstream can tell.

### 4. The logged VBLL predictive is a 20-sample Monte Carlo estimate

`probing_vbll._forward` reads `out.predictive.logits`, and vbll's `forward`
(`classification.py:147`) builds that from `self.predictive(x)` at its default
`n_samples = 20`, with no pass-through. Averaging 20 softmax samples is a noisy
estimate of `E_w[softmax(w·x)]`, and taking an argmax through the noise costs
accuracy in proportion to posterior width relative to the decision margin:

| config | m/sd | test @ n=20 | test @ n=2000 | cost |
|---|--:|--:|--:|--:|
| `q=100, eta=1` (campaign) | 0.44 | 0.4524 | 0.5414 | 8.9 pts |
| `q=1000, eta=1` | 0.55 | 0.6637 | 0.8158 | 15.2 pts |
| `q=100, eta=0.01` | 6.02 | 0.8747 | 0.8758 | 0.1 pts |
| `q=100, eta=0` | 12813 | 0.8819 | 0.8819 | 0 pts |

The cost nearly vanishes at the operating points the new search space will
select, so this is a correctness fix rather than a headline one — but the bias
distorts TPE's model of the space during exploration, where wide-posterior
configs are penalised for the estimator's noise rather than their own merit.

## Non-goals

- Changing the tuning objective. Raw `val_nll` selects a different `eta` than
  temperature-scaled `val_nll` would, but the two picks differ by ≤0.007 nats on
  the scaled metric (see *Recorded for the paper*), so the mismatch has no
  measurable cost. Fitting `T` on an inner split of val to make the objective
  match the reported metric is a larger change than the evidence justifies.
- Re-running the 6 AdamW+CE studies that are not flowers102. None of these
  changes touches them.
- Task 10's `export.py` and the Stage 4/5 work. They resume once this lands.
- The 1,756 untagged flowers102 runs in the `bllarse` experiment (2026-02-27 to
  2026-06-18). Predate this protocol; left alone.
- The 6 non-flowers102 VBLL study records in experiment 13. They are superseded
  by the re-run, but campaign namespacing keeps them from colliding, and they are
  the evidence for the boundary-solution finding above. Retained.

## Design

### VBLL search space

The fit-controlling quantity is a ratio, so searching `prior_scale` and `eta` as
independent axes would spend trials sliding along a diagonal ridge. Searching the
ratio directly makes the axes near-orthogonal: one controls fit, the other
controls posterior width.

This takes the VBLL space from 5 axes to 6 (`learning_rate`, `weight_decay`,
`batch_size`, `kl_ratio`, `prior_scale`, `wishart_scale`). The alternative of
adding `eta` alongside an unchanged `prior_scale` is also 6, but with the
optimum on a ridge rather than at a point, so orthogonality is bought at no cost
in dimension. The budget rises to compensate — see *Re-run*.

`wishart_scale` is a candidate for removal, which would return the space to 5:
it enters only through the `eta`-scaled block, and at the selected ratio that
block is ~1,000× weaker than the campaign's, so it may be nearly inert. That is
the same argument that proved **false** for `prior_scale`, and it is unmeasured
here, so `wishart_scale` stays in the space.

```python
params["kl_ratio"]      = trial.suggest_float("kl_ratio", 1e-7, 1e-2, log=True)
params["prior_scale"]   = trial.suggest_float("prior_scale", 1e-1, 1e3, log=True)
params["wishart_scale"] = trial.suggest_float("wishart_scale", 1e-3, 1e1, log=True)
```

`space.suggest` returns `kl_ratio`; `objective` passes it through unchanged (it
builds args from the config dict, so no mapping code is involved), and
`probing_vbll` multiplies:

- new flag `--kl-ratio`, default `1.0` (which reproduces today's behaviour when
  combined with the default `--prior-scale 1.0`).
- `regularization_weight = args.kl_ratio * args.prior_scale / len(train_f)`,
  replacing the hardcoded `1.0 / len(train_f)` at `probing_vbll.py:94`.

Range justification: measured optimum is ratio ≈1e-5, with 1e-4 already 1.2
points worse and 1e-3 6.3 points worse; `[1e-7, 1e-2]` brackets it by two decades
below and three above. `prior_scale ∈ [1e-1, 1e3]` covers posterior sd from ~0.06
to ~6 at the selected ratio. `wishart_scale` is unchanged — its selected values
span 0.003–3.2, comfortably interior.

`eta` is not itself a search axis; it is `kl_ratio * prior_scale` and is logged
as a derived quantity so the departure from the ELBO is visible per trial.

**This is a tempered posterior, not the ELBO.** `eta = 1` is the principled
setting and the paper must say plainly that the method needs `eta ≈ 1e-3` to be
competitive, rather than presenting it as a tuning detail. vbll 0.4.9 ships no
guidance on `regularization_weight`; `1/N` is simply the standard scaling when
the likelihood term is a per-example mean.

### Predictive estimator

`probing_vbll._forward` stops going through `head(x).predictive` and calls
`head.predictive(x, n_samples=...)` directly, taking `log` of the clamped
probabilities so the returned array stays log-probabilities as before.

- new flag `--predictive-samples`, default **200**.
- `rsample` allocates `[n_samples, chunk, C]`, so the feature chunk is sized from
  `n_samples`: `chunk = max(64, int(2e8 / (n_samples * C)))`. Without this the
  round-1 diagnostic OOM'd at 16 GiB.

200, not 2000, and not 32 or 64. The NLL error is a Jensen bias on
`log E[softmax]` and scales as `1/n`; the accuracy loss comes from argmax errors
through the noise and decays faster still. So `n=200` leaves at most ~10% of the
`n=20` error — under 0.1 points where the new space selects, ~1.5 points at the
campaign's worst corner, which is only visited during exploration. That is an
extrapolation from the two sampled points (`n=20`, `n=2000`), not a measurement,
and it is not worth a job to refine: `predictive` samples logits, `[n, B, C]`,
with no dependence on the 1280-dim feature, so 40 dumps at `n=200` cost ~1 s
against an ~11-minute trial. `n=64` would save under a second.

Two reasons to be generous rather than tight. First, the bias is largest when the
posterior is widest, which is *early* in training (`m/sd ≈ 0` at init) — and
`MedianPruner` cuts at `PRUNER_WARMUP_DUMPS = 10` of 40, so the estimator is
noisiest exactly where the pruning decisions are made. Second, the bias has a
direction: it penalises wide posteriors, and `val_nll` is the objective. Since the
paper's claim is about posterior width collapsing, an objective biased against
width would partly manufacture that claim.

Worth a footnote in the paper: vbll ships `n_samples = 20` as its default, and
that default costs up to 15 accuracy points in the wide-posterior regime. Anyone
using the library as-is on a small-margin problem takes that hit silently.

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
but **neither arm passes the split it actually loaded**: `probing_vbll.py:89` and
`probe.py:149` both hardcode the literal `"test"` while loading
`eval_split_name(dataset)`. Left alone, the swap makes flowers102 load a
7,169-row eval pool and look it up under `flowers102_test_val0.1.npz`, which
holds 1,020 — `load_or_make_pinned_split` raises on the size mismatch. It fails
loudly rather than silently reusing, but it fails.

Both call sites pass `eval_split_name(args.dataset)` instead. This also corrects
a latent mislabel: food101's eval pool is the `validation` split, yet its pinned
split is stored under `food101_test_val0.1.npz`. That file is regenerated under
the correct name; the indices change, so **food101's val/test partition is not
comparable across the rename** and its VBLL study is re-run anyway.

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

### Campaign namespacing

The re-run must not collide with the first campaign. `study_name()` returns
`f"{arm}_{dataset}"`, the journal file is `{study_name}.log`, and `open_study`
passes `load_if_exists=True`. Re-running as-is would **resume** each existing
study, mixing trials sampled under the old parameterization with new ones — and
for flowers102, mixing trials whose `batch_size` (512, 1024, 2048) is no longer
in the search space at all. In MLflow the collision is worse still: same
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

- 7 VBLL studies (search space reparameterized)
- AdamW+CE flowers102 (pools changed)

The VBLL trial budget rises from 24 per worker to **32** (96 → 128 per study) to
cover the sixth axis. AdamW+CE is unchanged at 16 per worker (64 per study); its
space did not change. That is roughly +33% wall-clock on 7 of the 8 studies.

The first campaign's other 6 AdamW+CE studies are unaffected and their results
carry forward, so `export.py` will read 6 studies from campaign 1 and 8 from
campaign r2.

## Recorded for the paper

Findings that this spec does not act on but that the write-up needs:

**Temperature scaling erases the `eta` ranking.** On stanford_cars at
`prior_scale = 100`, raw `test_nll` spans 0.31 nats across `eta ∈ [0.001, 0.03]`
and raw `test_ece` spans 0.22; after scaling they span **0.034** and **0.007**.
The raw objective prefers `eta = 0.001`, the scaled metric prefers `eta = 0.003`
— the ranking flips, at a cost of 0.002 nats. oxford_pets behaves the same way
(raw span 0.163 nats, scaled 0.006, winner flips).

**The posterior is over-dispersed, not correcting overconfidence.** Every fitted
`T` is below 1 (0.872 down to 0.479 as `eta` grows), so the Jensen-bound-trained
predictive is *under*-confident, and raw ECE gets monotonically **worse** with
posterior width (0.0278 → 0.2509). Temperature scaling is what repairs it. This
is the opposite of the usual argument for last-layer Bayes.

**At the selected operating point the posterior is nearly a point mass.** At
`prior_scale = 100, eta = 0.001` the margin-to-sd ratio is 23.6 and posterior sd
is 0.193. Combined with the two findings above, the honest summary is that tuned
VBLL is close to a temperature-scaled deterministic head on these benchmarks.

## Testing

- `batch_cap(1020) == 256`, and `batch_sizes("flowers102") == [64, 128, 256]`.
- `train_split_name("flowers102") == "test"` and
  `eval_split_name("flowers102") == "train"`; both default to their old values
  for every other dataset.
- The pinned split for the new flowers102 pool is 717 / 6,452, and both arms
  request it under the eval split's own name rather than the literal `"test"`.
- `is_clipped_temperature` is true on logits that drive `beta` to either bound
  and false on a well-conditioned fit; `step_metrics` carries the flag through
  to its returned scalars.
- `study_name("vbll", "dtd")` is unchanged; `study_name("vbll", "dtd", "r2")`
  is `vbll_dtd_r2`, and the two resolve to different journal files.
- `suggest` proposes `kl_ratio` within `[1e-7, 1e-2]` and `prior_scale` within
  `[1e-1, 1e3]`, and never returns a bare `kl_weight`.
- `probing_vbll` sets `regularization_weight` to
  `kl_ratio * prior_scale / len(train_f)`; the defaults
  (`kl_ratio=1.0, prior_scale=1.0`) reproduce the pre-change value.
- `_forward` honours `--predictive-samples` and its chunk size falls as
  `n_samples` rises, so a large `n_samples` does not allocate more than the
  budgeted `[n_samples, chunk, C]`.
- The deletion script, against a fake client, selects exactly the four
  flowers102 group_ids in experiment 13 and nothing in experiment 1.
