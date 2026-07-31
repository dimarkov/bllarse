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

A 357× spread across panels of one figure. For calibration, a VBLL head needed
~600 steps to reach 0.944 on cached CIFAR-10 features in a direct measurement,
so the two smallest datasets are plausibly under-trained.

Options: keep fixed epochs; switch to a fixed gradient-step budget; or early-stop
on the pinned val split. The third is probably right — Fig 1 is a
converged-endpoint comparison and the sample-efficiency claim lives in Fig 5 —
and `ood_evaluation.train_ce_head` already has patience-based early stopping to
model on. It changes `scripts/alternatives/probe.py`, so it is your call.

The table is reproduced as a comment in `bllarse_sweeps/probe_adamw_ce.py`.

### 2. There is no experiment tracking

Verified: `grep -c mlflow` returns 0 for both `scripts/alternatives/probe.py` and
`scripts/vbll_pytorch/probing_vbll.py`; no production sweep sets
`enable_mlflow`; `run_sweep.py`'s parent-run machinery keys off that flag and is
therefore dormant.

Spec §0b.1 ("Storage") calls for MLflow to receive the dump path plus scalar
metrics. This is a gap between spec and plan, not a defect in any task.

Figures are unaffected — every number comes from dumps. Job observability is
not: across ~720 array tasks the only signal is SLURM logs and dump-file
presence. Decide before Phase 1a.

### 3. Two backbones have no cached features at all

`dinov3_small` and `dinov3_huge` have features; **`dinov3_big` and
`dinov3_large` have none, for any dataset**. The four-model ladder needs
extraction for 2 backbones × 6 datasets × 2 splits, including ImageNet-1k at
512px (1.28M images per backbone). Compute-budget decision, gates Phase 1a.

SVHN is done: all four backbones × 2 splits at res224, cached and mirrored to
`dimarkov/bllarse-features`.

---

## Cluster runbook

Nothing here has been executed — this machine has no `apptainer` and no SLURM.
**Treat the first build as unverified**: the base image tag was wrong twice
before being corrected against the Docker Hub API.

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
| `bllarse.analysis.thresholds` | joint acc/NLL/Brier threshold protocol |
| `bllarse.analysis.curves` | dumps → metric curves, with per-dump-point TS refit |
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

---

## Known-open, deliberately

- **`crossings_per_seed` has no production caller.** The interface is complete
  and unit-tested; nothing yet builds `Mapping[int, Curve]` from dumps. Spec
  §0b.4 anticipated this ("cannot be exercised end-to-end until Phase 1
  produces a baseline"). A figure-building script is future work.
- **`samples_seen` overcounts** by the dropped remainder batch — 2.6% on
  oxford_pets, 0.06% on imagenet1k. Symmetric across arms, so no comparison
  bias, but the compute axis is not exactly comparable across panels.
- **`segmentation.py` and `sample_efficiency.py` are broken** by a stale equimo
  import, inherited from the extracted code. Off the probing path.
- **Four test-collection errors** from a `blrax` version drift
  (`blrax.optim.evon` missing). Pre-existing; blocks Plan B, not Plan A.
- **VBLL logits are normalised log-probs**, the JAX arm's are raw. Verified not
  to bias any metric (all are shift-invariant, and `softmax(log p) == p`); the
  residue is unmatched fp16 rounding, sub-1e-3.
