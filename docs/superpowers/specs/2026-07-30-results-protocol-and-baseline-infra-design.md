# Results Protocol and Baseline Infrastructure

**Date:** 2026-07-30
**Companion to:** `../paper/bllarse/results-plan.md` (figure list),
`../paper/bllarse/q-update-spec.md` (algorithm), `../paper/bllarse/REWRITE-PLAN.md` (§3 and reviewer answers)

## Goal

Build the shared measurement protocol and cluster infrastructure that every
comparator in `results-plan.md` will use, then launch the two comparators that
need no algorithm work: **AdamW+CE** and **VBLL**. The IBProbit `q` change
(`q-update-spec.md` §9) proceeds independently and lands into the same protocol.

The organising claim: **the protocol is the shared dependency, not the
algorithm.** `results-plan.md` §0 fixes the metric set, the comparator set, the
compute accounting and the threshold protocol before any run. Those decisions
bind AdamW, VBLL and IBProbit identically. Wiring them once, first, means the
baseline runs are computed once and never re-run — and it removes the apparent
conflict with `results-plan.md`'s "the `q` change must land first", which gates
*IBProbit* runs, not baseline runs.

## Non-goals

- No IBProbit algorithm changes. That is Phase 2, specified in `q-update-spec.md` §9.
- No runs of any kind in Phase 0 beyond a per-arm smoke test.
- No new figures. The figure list is `results-plan.md`'s and is not revisited here.
- No revival of the excised deep-MLP / `scaling_mlps` experiments
  (`results-plan.md` §3). The existing VBLL harness targets exactly those and is
  being repointed, not extended.

---

## Phase map

| phase | content | gated on |
|---|---|---|
| **0** | shared protocol + cluster infra + VBLL repoint. No runs. | — |
| **1** | AdamW+CE and VBLL runs: probing ladder, then ImageNet full FT | Phase 0 smoke test green |
| **2** | IBProbit `q` change (`q-update-spec.md` §9) | independent of 0/1; only needs 0b's metrics module |
| **3** | IBProbit runs: §12.5 measurement → Fig 4 ablation → ImageNet | Phase 2 + Phase 1 probing (thresholds are set from the strongest baseline) |

Phases 1 and 2 run concurrently. Phase 3 needs both.

---

## Phase 0a — Branch consolidation

The pieces exist in three trees and no tree has all of them.

| tree | ahead / behind `main` | holds |
|---|---|---|
| `main` | — | cleaned `run_sweep.py`, `src/slurm/jobs/*`, `bllarse_sweeps/`, `tools/adapters.py` |
| `ibprobit-process-noise` | 72 / 20 | `scripts/alternatives/*`: `full_finetune_comparison.py`, `vit_classification.py` (model/dataset registries), `precompute_features.py`, `precompute_ood_features.py`, `calibration.py`, `ood_evaluation.py`, `sample_efficiency.py` |
| `experimental_infra_vbll_baseline` | 36 / 29 | `scripts/vbll_pytorch/` (+ `scaling_mlps` submodule), `run_vbll_training_from_config`, `bllarse_sweeps/sweep9_*` |

Both branches have diverged from `main` substantially in both directions, which
is why this is an extraction rather than a merge.

**Target:** new branch `infra/results-protocol` off `main`.

**Method: take the code, not the branches.**

- From `ibprobit-process-noise`, take `scripts/alternatives/*.py` and its
  `docs/superpowers/specs/*`. Do **not** merge the branch — it carries ~5900
  `scripts/results/**/*.parquet` files and `results_ibprobit_large_batch.csv`
  (53k lines). Those are run outputs, not code.
- From `experimental_infra_vbll_baseline`, take `scripts/vbll_pytorch/` and the
  `scaling_mlps` submodule registration.
- **Do not** take that branch's `src/bllarse/tools/adapters.py`. It is the
  pre-cleanup version; `main`'s was rewritten in #48. Port forward only the two
  behaviours it adds, onto `main`'s file:
  1. `run_vbll_training_from_config(config)` → `run_script_from_config("scripts/vbll_pytorch/finetuning_vbll.py", config)`.
     Note the path convention differs: `main`'s `run_script_from_config` takes a
     repo-root-relative path, the old one took a `scripts/`-relative name.
  2. In `config_to_argv`, the `--no-<flag>` case for false booleans. The old
     version special-cased `return_ood` by name; generalise it to any key in an
     explicit `NEGATABLE` set so the intent is visible rather than hardcoded.

**Why this is safe to do before Phase 2:** `full_finetune_comparison.py`'s
IBProbit-specific flags (`--fix-q`, `--hyper-lr`, `--anchor-weight`,
`--q-init`, `--sigma-z2-init`, `--bmr-gate`) are inert under `--arm ce`. The file
comes over unchanged and Phase 2 rewrites that arm in place.

**Also in 0a:** `.gitignore` must exclude the eval-dump tree (0b) so this never
recurs.

---

## Phase 0b — The eval contract

The part that makes Phase 1's runs reusable. Four pieces.

### 1. Dump logits, not probabilities

`results-plan.md` §0.2 makes post-hoc temperature scaling mandatory wherever
calibration is claimed, and TS needs logits. From logits one recovers accuracy,
ECE at any binning, NLL, Brier, TS, and the §0.4 threshold crossings. From
probabilities one recovers all of those *except* TS.

For IBProbit the dumped quantity is the **moderated score**
(`q-update-spec.md` §5.8), i.e. the same array the predictive is formed from —
not the plain `μᵀf`. For VBLL it is the head's predictive logits. Each arm
records which, as a metadata field, so no post-hoc consumer has to guess.

**Format.** `float16`, one file per (run, eval point):

```
$BLLARSE_EVAL_DUMP_ROOT/{group_id}/{uid}/step_{step:07d}.npz
  logits  (N, K) float16
  labels  (N,)   int16
  score   (N,)   float32   # native epistemic score; diagnostic, IBProbit and VBLL only
  meta    dict   {arm, step, samples_seen, split, logit_kind, score_kind, seed, eps_hat}
```

The split index vector is written once per run as `split_indices.npz`, not
repeated per eval point. `eps_hat` is IBProbit's converged ε̂ for that step — a
scalar, needed to reconstruct the floored predictive post-hoc.

**The `score` channel is diagnostic, not load-bearing** — see the Fig 2 scoring
decision below. The primary detection figure is built from `logits` alone for
every arm. `score` carries IBProbit's `x̃ᵀΣᵖʳᵉᵈx̃` (`predictive_variance`,
`bayesian.py:467`) and VBLL's predictive variance so the §5.8 ARD ablation can be
run; `score_kind` records which, and it is absent for AdamW+CE where no such
quantity exists.

**Storage.** MLflow gets the *path* as a param plus the scalar metrics; it does
not get the arrays. ImageNet at K=1000: 10 MB per eval point on the 5k eval
subset, 100 MB for a full 50k final eval. Probing datasets are 1–2 orders
smaller.

**Dump schedule — log-spaced, not every point.** `full_finetune_comparison.py`
defaults to `--eval-every 1` over `--num-steps 500`; dumping all 500 is ~5 GB
per run for no benefit. Fig 5's claim is that *the gap is largest early*, so the
schedule should be dense early and sparse late: log-spaced steps
(`np.unique(np.geomspace(1, num_steps, n_dumps).astype(int))`), ~40 points per
run, ~400 MB, plus always the final point at full eval-set size. In-loop scalar metrics stay at
`--eval-every` and are unaffected — this schedule governs the array dumps only.

⚠ The threshold protocol's time resolution is the dump grid's resolution.
Log spacing is right for a log-x sample-efficiency figure and for
first-crossing detection, but a crossing is reported as the first *dumped* step
at which all three targets hold, and that must be stated in the caption.

### 2. Pin the splits

`results-plan.md` §0.2 fits TS on a held-out split. Today `--val-fraction`
draws a seed-dependent shuffle, so every seed gets a different val/test
partition and TS is not comparable across seeds — nor across arms. With T refit
per dump point (below), the same split is reused ~40 times per run, which makes
this a correctness requirement rather than a tidiness one: a split that moves
with the seed would put ~40 differently-conditioned fits on each curve.

Fix: a fixed, seed-independent index file per (dataset, split), generated once
and cached next to the features:

```
.cache/splits/{dataset}_{split}_val{frac}.npz   # keys: val_idx, test_idx
```

Generated with a fixed dedicated seed (not the run seed), committed to the HF
feature repo alongside the features so cluster and workstation agree.

### 3. One metrics module

`src/bllarse/metrics.py` — accuracy, ECE (20 bins, per §0.1), NLL, Brier, and
temperature scaling; all taking `(logits, labels)`.

- Brier is the **full predictive vector** multiclass form
  `BS = (1/N) Σᵢ Σₖ (p_ik − y_ik)²`, not the top-label variant (§0.1).
- `compute_brier` already exists at `full_finetune_comparison.py:461` — lift it
  here rather than reimplement, and delete the local copy.
- `scripts/alternatives/calibration.py` has accuracy/ECE/NLL but **no Brier**,
  so the probing path currently cannot report it. It becomes a thin wrapper over
  this module.
- **Detection metrics are entirely new**: AUROC, FPR@95TPR and AUPR
  (`results-plan.md` §1 Fig 2). Nothing in the repo computes any of them. What
  exists on `ibprobit-process-noise` — `ood_evaluation.py`,
  `_ood_metrics_from_logits` — measures accuracy and calibration *under shift*,
  which is a different question from *detecting* shift. Both are wanted (Fig 6 is
  the former, Fig 2 the latter) and neither substitutes for the other. Signature
  is `(scores_id, scores_ood) → {auroc, fpr95, aupr}`, with a documented sign
  convention for which direction means "more OOD".
- **MSP and predictive entropy** as score functions over a normalised predictive,
  so the same functional form applies to every arm (see the Fig 2 scoring
  decision below).
- TS is fit by Newton steps on β = 1/T; `CrossEntropy.calibrate`
  (`src/bllarse/losses/classical.py`) already implements exactly this. Move the
  Newton loop here so it applies uniformly to every arm including VBLL, and have
  the loss method call it.

**TS schedule — refit at every dumped step.** Standard TS (Guo et al.) fits T
once, at convergence. Figs 5–6 are trajectories, so a once-fit T would be a
strawman precisely on the early part of the curve where the sample-efficiency
claim lives — IBProbit recalibrates itself continuously, and the baseline should
be allowed to as well. T is therefore refit on the pinned val split at each of
the ~40 dump points, giving AdamW+CE+TS a full curve in Figs 5–7 rather than a
single endpoint. This is non-standard and **the captions must say so**; it makes
the baseline strictly stronger, which is the direction a fairness deviation
should run.

Cost is nil — it is a Newton fit on logits already on disk, with no extra
training.

**Why TS is not optional here.** `results-plan.md` §0.2 makes it R2's stated
bar, and `q-update-spec.md` §5.8 makes it structural: because Σ and Q are
class-independent, `s(f)` carries no class index and *is* an input-dependent
temperature, of which TS is the degenerate constant case. Without a TS curve
that claim has algebra behind it and no measurement.

The linear-probing setting sharpens this rather than weakening it. For a linear
head on frozen features, `softmax(Wf/T) = softmax((W/T)f)` — the temperature is
already inside the model class, and the head simply did not learn that scale
under CE. So TS adds no capacity; it refits the same function class on held-out
data. An IBProbit win that survives it is demonstrably not a rescaling, which is
R2's objection answered on R2's own terms.

### 4. Threshold protocol as post-hoc analysis

`src/bllarse/analysis/thresholds.py`, operating on dumped logits — nothing in
any training loop. Implements `results-plan.md` §0.4 verbatim:

- joint targets on accuracy, NLL and Brier; crossing is the first dumped step at
  which **all three** hold;
- targets are a fraction of the *strongest baseline's* converged value per
  metric, computed from the baseline run;
- non-crossing runs are right-censored and returned flagged, never dropped or
  extrapolated;
- per-seed crossings, not the crossing of the seed-mean curve.

Because targets are read off the strongest baseline, this module cannot be
exercised end-to-end until Phase 1 probing produces one. Phase 0 delivers it
with unit tests over synthetic curves covering: clean crossing, censored run,
accuracy-early/calibration-late (must *not* score as fast — the failure mode
§0.4 exists to prevent), and non-monotone recrossing.

---

## Phase 0c — Apptainer

The committed `scripts/start_docker_sbatch.sh` runs `docker run` on compute
nodes. The target cluster is Apptainer-only.

**Two images, not one.** `jax[cuda]` and `torch` pin CUDA/cuDNN differently;
one env holding both is a recurring tax for no gain, and the two arms never run
in the same process.

| image | contents | used by |
|---|---|---|
| `bllarse-jax.sif` | `jax[cuda]`, equimo, blrax, mlpox, the `bllarse` package | AdamW+CE and IBProbit arms |
| `bllarse-torch.sif` | `torch`, `torchvision`, `vbll`, mlflow | VBLL arm |

Built from the `infra/docker-image` Dockerfile via `apptainer build ... docker-daemon://`
or a def-file, whichever the cluster's build path allows. SIFs live on the
shared filesystem; no per-node pull, and no repeat of the Docker script's
`docker image inspect || exit 125` failure mode.

**New:** `src/slurm/jobs/slurm_run_config_apptainer.sh`, mirroring
`slurm_run_config_docker.sh`, with:

- `apptainer exec --nv "$BLLARSE_SIF" ...`, image selected per sweep;
- binds: repo root, `$HOME`, HF cache, and the eval-dump root. `$HOME`, `$PWD`
  and `/tmp` bind by default, so the Docker script's `passthrough_vars` list
  shrinks to genuine env (MLflow creds, `HF_TOKEN`, `CONFIG_IDX`, `INDEX_OFFSET`,
  `BLLARSE_SWEEP_SOURCE`, `VENV_NAME`);
- no `-u $(id -u):$(id -g)` — Apptainer preserves UID/GID by default;
- `BLLARSE_HF_HOME` env-settable. The workstation's `~/.cache/huggingface` is a
  symlink to an external drive and that path will not exist on the cluster.

`run_sweep.py`'s `DEFAULT_JOB_SCRIPT` moves from the Docker script to this one.
The Docker scripts stay for local multi-GPU use.

---

## Phase 0d — VBLL repoint

The existing harness (`scripts/vbll_pytorch/`) is bound to the excised
experiment: `models.py` hard-requires the `scaling_mlps` submodule and raises if
absent; `data.py` knows only CIFAR-10/100 at 64px; backbones are `B_6-Wi_512` /
`B_12-Wi_1024`. It cannot produce any figure in the current plan.

Split the repoint by what it actually needs.

### 0d-i. Probing (Figs 1–4) — no PyTorch backbone required

`precompute_features.py` caches frozen features as plain `.npz`
(`features`, `labels`), keyed `{model}_{dataset}_res{size}_{split}.npz`,
mirrored to `dimarkov/bllarse-features`. PyTorch reads those arrays as readily
as JAX does.

So probing VBLL is: load npz → `torch.from_numpy` → `vbll.DiscClassification`
head → train → dump logits per 0b. No DINOv3 in torch, no dataloader work, no
image preprocessing.

**This also removes the parity risk from Figs 1–4 entirely** — all four
comparators consume byte-identical features.

New `scripts/vbll_pytorch/probing_vbll.py`; `models.py` gains a backbone-free
head-only path so the `scaling_mlps` import is no longer unconditional.

### 0d-ii. Full fine-tuning (Figs 5–7) — a full PyTorch backend

Backbone gradients flow, so there is no cached-feature shortcut: this arm needs
a genuine PyTorch stack end to end.

| piece | requirement |
|---|---|
| backbone | torch DINOv3 loading the **same checkpoint** equimo does |
| data | ImageNet-1k pipeline at the resolution `DINOV3_DATASET_IMG_SIZES` gives it (512) |
| augmentation | matched to the JAX arm's augmax recipe, transform for transform |
| precision | matched (`--bf16` on the JAX side) |
| optimizer | AdamW with the same betas, weight decay, schedule, and warmup |
| eval | same pinned splits, same dump schedule, same `metrics.py` (0b) |

This is where the accepted parity risk lives. `results-plan.md` §0.4 point 4:
*"same optimizer state, same augmentation, same precision. Any asymmetry here
invalidates the compute claim."* Every row above is a place the two stacks can
silently diverge, and only the first is caught by a feature-parity test.

**Gate — a two-part parity test, before any Phase 1b run counts.**

1. *Forward parity.* Load the same checkpoint into both stacks, run a fixed
   batch through each, assert per-feature agreement to a stated tolerance. The
   tolerance and the achieved max-deviation both go in the paper's appendix —
   Fig 7b is a cross-framework wall-clock comparison and a reader is entitled to
   the number.
2. *Recipe parity.* A config-level assertion that the torch arm's augmentation
   transforms, precision, optimizer hyperparameters, schedule and eval splits
   equal the JAX arm's. Cheap to write, and it is the only thing standing
   between "matched recipe" as a claim and as a fact — forward parity says
   nothing about rows 2–6 of the table above.

**Cost probe, same step:** `results-plan.md` §5.1 asks whether VBLL is feasible
at ImageNet full FT at all — the prior comparison was CIFAR-scale at batch 512.
Measure step time and peak memory for one arm at target batch size here, in
Phase 0, before Phase 1b is committed. If infeasible, `REWRITE-PLAN.md` Risk 1
needs revisiting rather than the figure.

---

## Phase 0e — OOD data for Fig 2

`results-plan.md` §1 Fig 2 evaluates *detection* from frozen features on three
pairs:

| pair | ID | OOD | character |
|---|---|---|---|
| 1 | CIFAR-10 | SVHN | far-OOD |
| 2 | CIFAR-100 | SVHN | far-OOD |
| 3 | CIFAR-10 | CIFAR-100 | **near-OOD** — the harder and more informative pair |

CIFAR-100 appears as ID in pair 2 and as OOD in pair 3, so only SVHN is new data.

**Work required:**

- **Register SVHN** in `DATASET_CONFIGS` (`vit_classification.py`):
  `ufldl-stanford/svhn`, config `cropped_digits`, 10 classes, `test` split. The
  `img_key`/`label_key` need the same check every other entry got — the registry
  is not uniform across sources.
- **`DINOV3_DATASET_IMG_SIZES["svhn"] = 224`**, matching CIFAR's entry. This is
  not a tuning knob: OOD features must be extracted at the *same resolution with
  the same backbone* as the ID features or they do not live in the same feature
  space. `docs/superpowers/specs/2026-06-18-ood-calibration-eval-design.md` makes
  exactly this point for the ImageNet OOD suite and it applies unchanged here.
- **Precompute SVHN features** for the four ladder backbones at res224, into the
  existing `.cache/features` layout and HF mirror. CIFAR-10/100 at 224 are
  already covered by the registry.
- **OOD dumps at the final eval point only.** Fig 2 is a converged-endpoint
  figure, not a trajectory (Fig 6 is the trajectory one, and it is shift rather
  than detection), so the log-spaced schedule of 0b.1 governs the ID curve and
  does not multiply the OOD cost.

**Narrative requirement**, carried from `results-plan.md` §1 Fig 2: this
reintroduces CIFAR as **in-distribution** data, on DINOv3 features. That is not
the excised deep-MLP CIFAR study, and the text must say so explicitly — otherwise
a reviewer who read the submitted version will read the cut as cosmetic.

### Fig 2 scoring — one score function for every arm

⚠ **Deviation from `results-plan.md` §1 Fig 2**, which specifies a *per-arm*
score: `x̃ᵀΣᵖʳᵉᵈx̃` for IBProbit, MSP/max-logit for AdamW, VBLL's own predictive
variance. That is confounded — a win under it cannot be attributed to the model
rather than to the score function, and it makes the weaker claim ("we carry a
signal you structurally lack") in place of the one §3.7 actually asserts ("our
predictive is better off-distribution").

IBProbit has a proper normalised predictive (`q-update-spec.md` §5.8:
`Φ(μₖᵀf/s(f))`, normalised, ε-floored), so MSP and entropy are well-defined for
it and there is no technical reason to score it differently.

| role | score | arms |
|---|---|---|
| **primary** | MSP and predictive entropy on each arm's own normalised predictive | all three, identical functional form |
| **diagnostic** | native epistemic score (`x̃ᵀΣᵖʳᵉᵈx̃`; VBLL predictive variance) | IBProbit, VBLL — labelled not cross-comparable |

The diagnostic row stays because §5.8 makes a claim *about the score itself* —
that per-component `q̂` upgrades the floor from `q‖f‖²` to the learned weighted
norm `Σⱼq̂ⱼfⱼ²`, "strictly a better OOD score than the isotropic version".
Testing that needs the score directly, and it is an ablation, not the comparison.

Two consequences:

- **Max-logit is not a cross-arm score.** IBProbit's scores are on a probit scale
  and normalised; AdamW's are unnormalised softmax logits. Only functions of the
  *normalised* predictive — MSP, entropy — compare across arms. Max-logit stays
  as an AdamW-side extra.
- **The ε floor cannot confound detection.** `p_ε = (1−ε̂)p + ε̂/K` is a strictly
  increasing affine map of MSP, so it preserves per-example ranking exactly and
  leaves AUROC unchanged. Only the input-dependent part of the moderation can
  move detection — which is precisely the mechanism under test. Worth one
  sentence in the paper: it forecloses the reading that the ε machinery is doing
  the OOD work.

The expected mechanism, stated so it can fail visibly: `s(f)` inflates
off-manifold → the predictive flattens → MSP falls. If MSP does *not* express the
epistemic signal, that is a finding about §3.7 worth having before the paper
claims it, and the diagnostic row is where it would show up.

---

## Phase 0f — Sweep definitions

Under `bllarse_sweeps/`, following the existing `create_configs()` / `run(config)`
contract consumed by `run_sweep.py` and `run_config.py`.

**The ladder is `dinov3_small` (384), `dinov3_big` (768), `dinov3_large` (1024),
`dinov3_huge` (1280)** — four models. `dinov3_max` (embed 4096) is out.

| file | arm | grid | figure |
|---|---|---|---|
| `probe_adamw_ce.py` | AdamW+CE | 4 backbones × 6 datasets × 10 seeds | Fig 1 |
| `probe_vbll.py` | VBLL | same grid | Fig 1 |
| `probe_ood_adamw_ce.py` | AdamW+CE | 4 backbones × 3 (ID, OOD) pairs × 10 seeds | Fig 2 |
| `probe_ood_vbll.py` | VBLL | same grid | Fig 2 |
| `fnft_imagenet_adamw_ce.py` | AdamW+CE | ImageNet-1k × 5 seeds | Figs 5–7 |
| `fnft_imagenet_vbll.py` | VBLL | ImageNet-1k × 5 seeds | Figs 5–7 |

Seeds: **10 probing** (deviating from `results-plan.md` §0.5's 25), 5 full FT.
AdamW+CE+TS is **not** a
separate sweep — it is post-hoc from the AdamW dumps, refit per dump point
(0b.3), so it costs no GPU time and adds no array tasks.

Sizing: Fig 1 is 4 × 6 × 10 × 2 = 480 array tasks, Fig 2 is 4 × 3 × 10 × 2 = 240
(three pairs, not two ID datasets — CIFAR-10 appears against both SVHN and
CIFAR-100), so **720 probing tasks** (seconds-to-minutes each, on cached
features), plus 10 long ImageNet full-FT runs.

The Fig 2 sweeps are separate files rather than extra datasets in the Fig 1
sweeps because they differ in what they dump (OOD `score` on a second dataset,
final point only) rather than only in which dataset they train on.
`run_sweep.py`'s `--index-offset` / `--num-jobs` chunking and `--max-concurrent`
cap already handle submitting these against a shared-cluster array-size limit.

---

## Phase 0g — Smoke test

One config per arm on the cluster, verifying end to end:

1. Apptainer launch under `sbatch --array`, both SIFs;
2. MLflow parent/child nesting via `MLFLOW_PARENT_RUN_ID`;
3. logit dumps land at the expected path with the expected shapes and dtypes;
4. **the `score` channel and `eps_hat` are present and non-degenerate** for the
   arms that carry them (0b.1). All-zeros would not break the primary Fig 2,
   which is built from logits, but it would silently cost the §5.8 ablation;
5. **`metrics.py` recomputes the in-loop scalar metrics from the dumps** to
   floating-point tolerance. This is the load-bearing check — it is what
   licenses treating the dumps as the source of truth for every later figure.

The smoke test covers one Fig 1 config and one Fig 2 config per arm, since they
exercise different dump paths.

---

## Alongside Phase 0 — the FLOP model

`results-plan.md` §4.1 puts this first and it needs no code: a paper
calculation of FLOPs per step for all four algorithms, including the head
honestly (§0.3 — IBProbit's two O(D³) Choleskys per batch at D+1=1281, plus
O(BDK) per inner iteration × T).

It decides whether Fig 7 is worth running at all. §2 Fig 7's stated risk is that
if IBProbit's Cholesky cost dominates at the batch sizes where it calibrates
best, *the figure argues against us* — and that is arithmetic, discoverable now,
not after the runs.

---

## Changes to sync back into `results-plan.md`

Settled here, deviating from or resolving something in the plan document. Listed
so the two files do not drift apart silently.

| change | where | status |
|---|---|---|
| Fig 2 scored by **MSP + entropy uniformly across arms**, native epistemic scores demoted to a diagnostic ablation | §1 Fig 2 (currently specifies per-arm scores) | **deviation** — rationale in Phase 0e above |
| Fig 1 ladder is `dinov3_small/big/large/huge`; `dinov3_max` (embed 4096) is out | §5.2 (listed as open) | resolved |
| TS refit at every dumped step rather than once at convergence | §0.2 (schedule unspecified) | refinement; needs a caption note |
| Max-logit demoted to an AdamW-side extra, not a cross-arm score | §1 Fig 2 | deviation, follows from the scoring change |
| Probing runs use **10 seeds**, not 25 | §0.5 | deviation — halves the error-bar resolution on Figs 1–4 and the per-seed censoring picture in §0.4; full-FT stays at 5 |

---

## Deferred decisions

Not blocking Phase 0. Listed so they are not silently forgotten.

| decision | source | needed by |
|---|---|---|
| Fig 5/6 reading (a) — OOD as a second axis over the same trajectory — vs (b) — OOD *is* the sample-efficiency measure | results-plan §2 Fig 6 | Phase 1b |
| ImageNet-C severity: averaged, or per-severity appendix panel | results-plan §5.3 | Phase 1b |
| Per-component vs shared γ | q-update-spec §12.1 | Phase 2 |
| Whether α-forgetting can go (needs the float32 Cholesky conditioning check at ImageNet scale) | q-update-spec §12.3 | Phase 2 |

---

## Risks

1. **Cross-framework wall-clock (Figs 5–7).** Accepted deliberately in favour of
   keeping the reference VBLL implementation. Mitigated by the 0d-ii parity
   tests and by reporting the achieved tolerance. FLOPs — the *primary* axis per
   §0.3 — are framework-independent and unaffected; only 7b is exposed. Note the
   asymmetry across figures: Figs 1–4 have **zero** parity risk (identical
   cached features), Figs 5–7 carry all of it.
2. **Dump-grid resolution bounds threshold precision.** Log spacing is dense
   exactly where Fig 5's claim lives, but crossings are quantised to the grid and
   the caption must say so. The per-step TS refit (0b.3) is on the same grid and
   needs the same caption note.
3. **Branch consolidation drops something.** `ibprobit-process-noise` is +72
   commits and the extraction is selective by design. Mitigation: the Phase 0g
   smoke test exercises both arms end to end, and `full_finetune_comparison.py`
   has its own tests on that branch (`tests/test_full_finetune_*.py`) which come
   across with it.
4. **Cluster array-size limits.** 720 probing tasks, submitted as four arrays of
   240 and 120, sit under the common `MaxArraySize` defaults — but read the
   cluster's value rather than assuming.
   `run_sweep.py` already chunks via `--index-offset`/`--num-jobs`; the limit
   needs reading off the cluster before the first large submission.
