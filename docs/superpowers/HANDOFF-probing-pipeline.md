# Probing pipeline — handoff

**Branch:** `infra/results-protocol` · **Date:** 2026-07-31
**Spec:** `specs/2026-07-30-results-protocol-and-baseline-infra-design.md`
**Plan:** `plans/2026-07-30-probing-baseline-pipeline.md` (14 tasks, all complete and reviewed)

Plan A of two. This delivers the measurement protocol and the probing arms
(Figs 1–4). Plan B — the PyTorch full-fine-tuning backend for Figs 5–7 — is not
started.

---

## Blocked on you

### 1. Optimisation budget is unequal across Fig 1's panels

At the configured `batch_size=512, epochs=40`, total gradient steps per dataset:

| dataset | n_train | steps/epoch | total steps |
|---|---|---|---|
| oxford_pets | 3,680 | 7 | 280 |
| dtd | 3,760 | 7 | 280 |
| flowers102 | 7,169 | 14 | 560 |
| stanford_cars | 8,144 | 15 | 600 |
| cifar10 / cifar100 | 50,000 | 97 | 3,880 |
| food101 | 75,750 | 147 | 5,880 |
| imagenet1k | 1,281,167 | 2,502 | 100,080 |

A 357× spread across panels of one figure. **Now measured, and worse than
"plausibly under-trained":** a `dinov3_small` probe on oxford_pets, run to 300
epochs (2,100 steps) with the sweep's own hyperparameters, was *still improving
monotonically at the last step* — val NLL never turned up, so there is no
overfitting to stop early:

| point | test acc | test NLL | val NLL |
|---|---:|---:|---:|
| step 295 (≈ the configured 40-epoch budget) | 0.7865 | 0.6923 | 0.6025 |
| step 2,100 (300 epochs) | 0.8692 | 0.4431 | 0.3574 |

**The configured budget costs 8.3 accuracy points on oxford_pets alone**, and
that is a floor — the curve had not converged at 300 epochs either.

So the fix is not early stopping (nothing overfits within any budget tried);
it is that 40 epochs is simply too few for the small datasets. Options, in
increasing cost: raise `epochs` per dataset so every panel converges; switch to
a fixed gradient-step budget large enough for the smallest dataset; or raise the
learning rate for the small panels. The stopping machinery is now in place
either way (`analysis.thresholds.early_stop`), and it will flag any run that
truncates — `at_last_step=True` is exactly the signal above.

Still your call, because it changes the sweep grid and therefore the cost.

### 2. ~~There is no experiment tracking~~ — done

Both arms open a nested MLflow run per config (`bllarse.mlflow_utils.child_run`),
all five sweeps set `enable_mlflow=True`, and the experiment is
`bllarse-probing`. Verified against a real sqlite-backed tracking server:

- params: the full config plus `steps_per_epoch` / `total_steps`
- tags: `arm`, `group_id`, `uid`, and **`dump_dir`** — spec §0b.1's requirement
  that MLflow carry the dump path
- metrics per dumped step: `test_{acc,ece,nll,brier}`, `val_{...}`,
  `temperature`, `test_ts_{...}`, plus `samples_seen` and `epoch` as metrics so
  MLflow can plot against the sample axis directly
- cross-process nesting confirmed: the parent is created by `run_sweep.py` in
  the submitting process, so children attach via an explicit
  `mlflow.parentRunId` tag rather than `start_run(nested=True)`, which cannot
  work across array tasks. A child run was queried back by parent id.

Tracking failures degrade to untracked rather than killing the job — which is
not hypothetical: MLflow now **raises** on a `file://` store ("maintenance
mode"), and that was caught by the fallback during testing, not by a test.

### 3. Two backbones have no cached features at all

`dinov3_small` and `dinov3_huge` are **fully covered for all four production
sweeps** — 0 missing files, verified by enumerating every config. `dinov3_big`
and `dinov3_large` have only SVHN; between them they need 32 more feature
files:

| backbone | missing | what |
|---|---:|---|
| `dinov3_big` | 16 | oxford_pets, food101, flowers102, stanford_cars, dtd, imagenet1k @res512 (train+eval) · cifar10, cifar100 @res224 (train+test) |
| `dinov3_large` | 16 | same |

ImageNet-1k at 512px is 1.28M images per backbone and dominates that cost.
This is a compute-budget decision, but it **no longer gates Phase 1a** — see
"Half the ladder is runnable today" below.

---

## Verified locally on 2026-07-31

The whole chain was run end-to-end on this workstation, on real cached
features, bypassing only Apptainer and SLURM:

- `python -m bllarse.tools.run_config bllarse_sweeps/probe_smoke.py 1` — the
  exact command the job script execs inside the container. AdamW arm, cifar10 +
  svhn OOD, 4 epochs: acc 0.9296 → 0.9466, 7 dumps written (3 val, 3 test,
  1 ood).
- VBLL arm, same config on CPU: 7 dumps, score channel present and
  **non-constant** (std 1.67e-3) — smoke criterion 3 passes.
- `analysis.curves.run_curve(temperature_scaled=True)`: ECE 0.198 → 0.0062,
  T refit per point (0.423, 0.553, 0.672).
- `analysis.curves.run_detection` on both arms × both scores: AdamW AUROC
  0.969 (MSP) / 0.981 (entropy), VBLL 0.959 / 0.971.
- `thresholds.make_targets` → `first_crossing` → `crossings_per_seed` on those
  curves. This closes the "no production caller" item below.

Still unexercised: the Apptainer build, `--nv` GPU passthrough, and sbatch
array submission. `--dry-run` emits the expected
`sbatch --export=ALL --array=0-59%20 ...`.

### Half the ladder is runnable today

`dinov3_small` and `dinov3_huge` need no new features, and each sweep orders
its configs `product(MODELS, DATASETS, SEEDS)` with MODELS outermost, so each
backbone is a **contiguous index range** — submittable with `--index-offset`
alone, no sweep edits:

| sweep | `dinov3_small` | `dinov3_huge` |
|---|---|---|
| `probe_adamw_ce`, `probe_vbll` | `--index-offset 0 --num-jobs 60` | `--index-offset 180 --num-jobs 60` |
| `probe_ood_adamw_ce`, `probe_ood_vbll` | `--index-offset 0 --num-jobs 30` | `--index-offset 90 --num-jobs 30` |

360 of the 720 configs. Pass `--job-name` per submission, otherwise both
chunks of a sweep share a name and `sacct --name` cannot tell them apart.

---

## Cluster runbook

The container path has **not** been executed — this machine has no `apptainer`
and no SLURM. **Treat the first build as unverified**: the base image tag was
wrong twice before being corrected against the Docker Hub API.

```bash
# 1. Capability check
which apptainer singularity && apptainer --version
scontrol show config | grep -i MaxArraySize   # largest array is 240; four arrays total 720
echo "$SCRATCH"

# 2. Confirm the base image pulls BEFORE building (distinguishes a bad tag
#    from a bad package list)
apptainer build --force /tmp/base-probe.sif \
    docker://nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04

# 3. Build both images
apptainer build bllarse-jax.sif   src/slurm/images/bllarse-jax.def
apptainer build bllarse-torch.sif src/slurm/images/bllarse-torch.def

# 4. Interpreter check FIRST — a broken interpreter masks everything downstream
apptainer exec bllarse-jax.sif python -c "import sys; print(sys.version)"   # expect 3.12
apptainer exec bllarse-jax.sif python -c "import jax, equinox, optax; print(jax.__version__)"

# 5. Then GPU visibility
apptainer exec --nv bllarse-jax.sif python -c "import jax; print(jax.devices())"

# 6. Smoke sweep — two submissions, because the arms need different images
export BLLARSE_EVAL_DUMP_ROOT="$SCRATCH/bllarse/eval_dumps"
BLLARSE_SIF=$PWD/bllarse-jax.sif   uv run python -m bllarse.tools.run_sweep \
    bllarse_sweeps/probe_smoke.py --index-offset 0 --num-jobs 2 --max-concurrent 2
BLLARSE_SIF=$PWD/bllarse-torch.sif uv run python -m bllarse.tools.run_sweep \
    bllarse_sweeps/probe_smoke.py --index-offset 2 --num-jobs 2 --max-concurrent 2
```

Success criteria for the smoke sweep, in order:

1. `sacct -u "$USER" --name probe_smoke --format=JobID,State,ExitCode` — all four
   tasks `COMPLETED`, exit 0.
2. Dumps exist under `$BLLARSE_EVAL_DUMP_ROOT/probe_smoke/{jax_fig1,jax_fig2,torch_fig1,torch_fig2}/`,
   with both `_val` and `_test` files per step and one `_ood` file for the `fig2` uids.
3. The VBLL `score` channel is present **and non-constant** —
   `np.std(record.score) > 0`. A constant score means the epistemic term is dead
   and would silently gut the Fig 2 ablation.
4. `bllarse.metrics.classification_metrics` recomputed from a dump matches the
   value printed in the job's stdout.
5. MLflow nesting — **skip**, structurally not exercised; see blocker 2.

Per-task detail, including full command transcripts and measured tolerances, is
in the (gitignored) working directory
`.superpowers/sdd/2026-07-30-probing-baseline-pipeline/` — one report per task
plus `progress.md`. Safe to delete once read.

---

## What was built

| module | purpose |
|---|---|
| `bllarse.metrics` | accuracy, ECE@20, NLL, full-vector Brier, Newton temperature fit, MSP/entropy OOD scores |
| `bllarse.detection` | AUROC, FPR@95TPR, AUPR (higher = more OOD) |
| `bllarse.splits` | pinned val/test partitions, derived from dataset name alone |
| `bllarse.evaldump` | fp16 dump writer/reader, log-spaced schedule |
| `bllarse.features` | framework-free cached-feature loader, used by both arms |
| `bllarse.analysis.thresholds` | joint acc/NLL/Brier threshold protocol, **post-hoc early stopping** |
| `bllarse.analysis.curves` | dumps → metric curves (test **and val**), with per-dump-point TS refit |
| `bllarse.reporting` | per-step scalars, shared by both arms so neither has its own metric code |
| `bllarse.mlflow_utils.child_run` | nested per-config MLflow run; no-op when off or unreachable |
| `scripts/alternatives/probe.py` | JAX probing arm |
| `scripts/vbll_pytorch/probing_vbll.py` | PyTorch VBLL probing arm |
| `bllarse_sweeps/probe_*.py` | 720 configs across four sweeps |
| `src/slurm/jobs/slurm_run_config_apptainer.sh` | Apptainer array job (now the default) |

Both arms read the same `.npz` feature files, so Figs 1–4 carry no
cross-framework parity risk. A standardisation step that would have broken that
was found and removed.

---

## Deviations from `results-plan.md` to sync into the paper

| change | where | status |
|---|---|---|
| Fig 2 scored by MSP + entropy uniformly across arms; native epistemic scores demoted to a diagnostic | §1 Fig 2 | deviation |
| Probing uses 10 seeds, not 25 | §0.5 | deviation |
| Fig 1 ladder is small/big/large/huge; `dinov3_max` out | §5.2 | resolved |
| TS refit at every dumped step rather than once at convergence | §0.2 | refinement; needs a caption note |
| Max-logit demoted to an AdamW-side extra | §1 Fig 2 | deviation |
| Dumps indexed on **optimiser steps**, not epochs; `samples_seen` exact and logged as the comparison axis | §0.2 | refinement |
| Early stopping applied **post hoc** from val NLL over the dumped grid, never live | §0.4 | deviation; needs a methods sentence |

---

## Known-open, deliberately

- **`crossings_per_seed` has no production caller.** Exercised by hand on real
  smoke dumps (above), so the interface is confirmed; what is still missing is
  a committed figure-building script that walks a whole sweep. Future work.
- **Dump storage is 332 GiB** across the four production sweeps at
  `n_dumps=40`, computed from the real eval-set sizes: 162 GiB per Fig 1 sweep
  (ImageNet alone is 3.7 GiB per run × 80 runs) and 3.8 GiB per Fig 2 sweep.
  Budget `$SCRATCH` accordingly. Raising `n_dumps` scales this linearly —
  per-*step* dumping was considered and is not affordable (ImageNet would be
  ~9 TB per run).
- **Log spacing puts the coarsest resolution late**, which is where a val-NLL
  minimum would sit. Consecutive late points differ by ~13% of the budget at
  `n_dumps=40`. Acceptable because the curve is flat near a minimum, but it
  bounds how precisely `early_stop` can locate one.
- **`segmentation.py` and `sample_efficiency.py` are broken** by a stale equimo
  import, inherited from the extracted code. Off the probing path.
- **Both images pin `mlpox` and `blrax` to `@main`**, a floating ref, while the
  dev venv pins blrax to `c37b1a83`. Neither is on the probing path, so Figs
  1–4 are unaffected — but pin them before Plan B, which does use blrax.
- **VBLL logits are normalised log-probs**, the JAX arm's are raw. Verified not
  to bias any metric (all are shift-invariant, and `softmax(log p) == p`); the
  residue is unmatched fp16 rounding, sub-1e-3.
