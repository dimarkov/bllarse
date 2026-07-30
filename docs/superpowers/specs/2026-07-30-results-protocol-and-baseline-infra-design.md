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
  meta    dict   {arm, step, samples_seen, split, score_type, seed}
```

The split index vector is written once per run as `split_indices.npz`, not
repeated per eval point.

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
partition and TS is not comparable across seeds — nor across arms.

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
- TS is fit by Newton steps on β; `CrossEntropy.calibrate`
  (`src/bllarse/losses/classical.py`) already implements exactly this. Move the
  Newton loop here so it applies uniformly to every arm including VBLL, and have
  the loss method call it.

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

### 0d-ii. Full fine-tuning (Figs 5–7) — torch DINOv3 required

This is where the accepted parity risk lives. `results-plan.md` §0.4 point 4:
*"same optimizer state, same augmentation, same precision. Any asymmetry here
invalidates the compute claim."* A torch DINOv3 must match equimo's.

**Gate — a parity test, before any Phase 1b run counts:** load the same
checkpoint into both stacks, run a fixed batch through each, assert per-feature
agreement to a stated tolerance. Test lives in `tests/`; the tolerance and the
achieved max-deviation both go in the paper's appendix, because Fig 7b's
wall-clock panel is a cross-framework comparison and a reader is entitled to the
number.

**Cost probe, same step:** `results-plan.md` §5.1 asks whether VBLL is feasible
at ImageNet full FT at all — the prior comparison was CIFAR-scale at batch 512.
Measure step time and peak memory for one arm at target batch size here, in
Phase 0, before Phase 1b is committed. If infeasible, `REWRITE-PLAN.md` Risk 1
needs revisiting rather than the figure.

---

## Phase 0e — Sweep definitions

Under `bllarse_sweeps/`, following the existing `create_configs()` / `run(config)`
contract consumed by `run_sweep.py` and `run_config.py`.

| file | arm | grid |
|---|---|---|
| `probe_adamw_ce.py` | AdamW+CE | backbones × 6 datasets × 25 seeds |
| `probe_vbll.py` | VBLL | same grid |
| `fnft_imagenet_adamw_ce.py` | AdamW+CE | ImageNet-1k × 5 seeds |
| `fnft_imagenet_vbll.py` | VBLL | ImageNet-1k × 5 seeds |

Seeds per `results-plan.md` §0.5: 25 probing, 5 full FT. AdamW+CE+TS is **not** a
separate sweep — it is post-hoc from the AdamW dumps.

Order-of-magnitude, assuming the five-model DINOv3 ladder (the count is a
deferred decision below): probing is 5 × 6 × 25 × 2 ≈ 1500 array tasks
(seconds-to-minutes each, on cached features); ImageNet full FT is 10 long runs.
`run_sweep.py`'s `--index-offset` / `--num-jobs` chunking and `--max-concurrent`
cap already handle submitting these against a shared-cluster array-size limit.

---

## Phase 0f — Smoke test

One config per arm on the cluster, verifying end to end:

1. Apptainer launch under `sbatch --array`, both SIFs;
2. MLflow parent/child nesting via `MLFLOW_PARENT_RUN_ID`;
3. logit dumps land at the expected path with the expected shapes and dtypes;
4. **`metrics.py` recomputes the in-loop scalar metrics from the dumps** to
   floating-point tolerance. This is the load-bearing check — it is what
   licenses treating the dumps as the source of truth for every later figure.

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

## Deferred decisions

Not blocking Phase 0. Listed so they are not silently forgotten.

| decision | source | needed by |
|---|---|---|
| Which DINOv3 checkpoints form the Fig 1 ladder; does `dinov3_max` (embed 4096) enter | results-plan §5.2 | Phase 0e sweep definitions |
| Fig 5/6 reading (a) — OOD as a second axis over the same trajectory — vs (b) — OOD *is* the sample-efficiency measure | results-plan §2 Fig 6 | Phase 1b |
| ImageNet-C severity: averaged, or per-severity appendix panel | results-plan §5.3 | Phase 1b |
| Per-component vs shared γ | q-update-spec §12.1 | Phase 2 |
| Whether α-forgetting can go (needs the float32 Cholesky conditioning check at ImageNet scale) | q-update-spec §12.3 | Phase 2 |

---

## Risks

1. **Cross-framework wall-clock (Figs 5–7).** Accepted deliberately in favour of
   keeping the reference VBLL implementation. Mitigated by the 0d-ii parity test
   and by reporting the achieved tolerance. FLOPs — the *primary* axis per §0.3 —
   are framework-independent and unaffected; only 7b is exposed.
2. **Dump-grid resolution bounds threshold precision.** Log spacing is dense
   exactly where Fig 5's claim lives, but crossings are quantised to the grid and
   the caption must say so.
3. **Branch consolidation drops something.** `ibprobit-process-noise` is +72
   commits and the extraction is selective by design. Mitigation: the Phase 0f
   smoke test exercises both arms end to end, and `full_finetune_comparison.py`
   has its own tests on that branch (`tests/test_full_finetune_*.py`) which come
   across with it.
4. **Cluster array-size limits.** ~1500 probing tasks may exceed `MaxArraySize`.
   `run_sweep.py` already chunks via `--index-offset`/`--num-jobs`; the limit
   needs reading off the cluster before the first large submission.
