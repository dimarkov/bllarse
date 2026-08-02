# Probing Hyperparameter Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tune `learning_rate`, `weight_decay`, `batch_size` (both arms) and
`prior_scale`, `wishart_scale` (VBLL) per dataset with Optuna TPE, then have the
Fig 1 / Fig 2 sweeps read the tuned values from a committed lookup instead of
restating argparse defaults.

**Architecture:** A new `bllarse.tuning` package holds the search space, the
trial objective, and a worker CLI. The worker is a *separate* submission path
from `run_sweep`: `run_sweep`'s contract is `create_configs() -> list[dict]`,
enumerated before submission, and TPE generates configurations adaptively.
Workers share one Optuna study per `(arm, dataset)` through a
`JournalStorage` file on `/ptmp`, four workers per study. The objective calls
the *existing* training scripts in-process — nothing about the dump protocol,
the metrics, or the stopping rule changes — and computes its value with
`analysis.thresholds.early_stop`, the same function the figures use.

**Tech Stack:** Python 3.12, Optuna >= 4.0 (TPE sampler, MedianPruner,
JournalStorage), JAX/equinox/optax (AdamW+CE arm), PyTorch/vbll (VBLL arm),
Apptainer on MPCDF Raven, SLURM, MLflow.

**Source spec:** `docs/superpowers/specs/2026-08-02-probing-hparam-tuning-design.md`

## Global Constraints

- **Test via Apptainer on a GPU node, never a local venv.** The suite cannot
  even be *collected* on CPU: `tests/losses/test_bayesian.py` calls a test at
  module import and dies with `ValueError: The precision 'TF32_TF32_F32' is not
  supported by dot_general on CPU`. Run the JAX and Torch subsets separately,
  each in its own image.
- **Tuning seeds are `(200, 201)`.** Fig 1 and Fig 2 report `range(137, 147)`.
  Selecting hyperparameters on a reported seed tunes against the same random
  draws the error bars are drawn from.
- **Selection is on `val_nll`; test and OOD never enter tuning.**
  `bllarse.splits` makes val and test disjoint by construction.
- **The objective *is* `analysis.thresholds.early_stop`,** not a
  reimplementation, so the tuning criterion and the reported stopping rule
  cannot drift apart.
- **`parameterization` is fixed to `"diagonal"` in every figure sweep.**
  `cov_rank` is never a tuned axis — it is swept only in the Stage 5 appendix.
- **Batch cap rule:** largest power of two `<= n_train // 2` (floor of 2
  optimiser steps per epoch), minimum 64.
- **MLflow: `experiment_id=`, never `set_experiment`.** See
  `mlflow_utils.resolve_experiment_id` — `set_experiment` does not activate an
  experiment it had to create, and the run silently lands elsewhere.
- **Optimiser-step counts are not comparable across trials.** Trials vary
  `batch_size`, so pruning must report on the *dump index* (0..39), never the
  raw step. See Task 8.
- **Two tasks have a hard cluster-execution dependency** and cannot be completed
  from a laptop: Task 4 (needs the pilot sweep's dumps) and Task 11 (needs the
  tuning campaign's journals). Both say so in their steps.

---

## File Structure

**Created:**

| Path | Responsibility |
|---|---|
| `src/bllarse/tuning/__init__.py` | Public surface: `tuned_config`, `TUNING_SEEDS` |
| `src/bllarse/tuning/space.py` | Dataset metadata, batch caps, the TPE search space |
| `src/bllarse/tuning/objective.py` | Run one config on both seeds, return mean early-stopped `val_nll` |
| `src/bllarse/tuning/worker.py` | Worker CLI: study, storage, pruner, `study.optimize` |
| `src/bllarse/tuning/budgets.py` | Stage 1: read pilot dumps, emit per-dataset epoch budgets |
| `src/bllarse/tuning/export.py` | Stage 3: read study journals, emit `tuned_hparams.json` |
| `bllarse_sweeps/probe_pilot_epochs.py` | Stage 1 pilot sweep (28 configs) |
| `bllarse_sweeps/probe_rank_pareto.py` | Stage 5 appendix sweep (54 configs) |
| `bllarse_sweeps/epoch_budgets.json` | Stage 1 output, committed |
| `bllarse_sweeps/tuned_hparams.json` | Stage 3 output, committed |
| `src/slurm/jobs/slurm_tune.sh` | SLURM array for one study's 4 workers |
| `tests/tuning/test_space.py` | Batch caps and search space |
| `tests/tuning/test_objective.py` | Objective arithmetic and dump cleanup |
| `tests/tuning/test_export.py` | Journal → JSON → loader round trip |

**Modified:**

| Path | Change |
|---|---|
| `scripts/vbll_pytorch/src/vbll_experiments/models.py:272-299` | `build_vbll_head` gains `cov_rank` |
| `scripts/vbll_pytorch/probing_vbll.py` | `--cov-rank`, `head_params`, `on_dump` |
| `scripts/alternatives/probe.py` | `head_params`, `on_dump` |
| `src/bllarse/tools/adapters.py` | Public `load_script_module` |
| `bllarse_sweeps/probe_adamw_ce.py` | Drop `imagenet1k`; read tuned lookup |
| `bllarse_sweeps/probe_vbll.py` | Drop `imagenet1k`; read tuned lookup |
| `bllarse_sweeps/probe_ood_adamw_ce.py` | Read tuned lookup |
| `bllarse_sweeps/probe_ood_vbll.py` | Read tuned lookup |
| `src/slurm/images/bllarse-jax.def` | Add `optuna>=4.0` |
| `src/slurm/images/bllarse-torch.def` | Add `optuna>=4.0` |
| `tests/test_sweep_definitions.py` | 240 → 200 configs |

---

## Execution Order

Code order and cluster order differ. Implement tasks 1-3, then **run the pilot**
before Task 4 can be completed; implement 5-10, then **run the campaign** before
Task 11 can be completed.

```
Tasks 1,2,3  ->  submit pilot  ->  Task 4  ->  Tasks 5,6,7,8,9,10
                                                      |
                                              submit campaign
                                                      |
                                              Tasks 11, 12
```

---

### Task 1: Plumb `cov_rank` through the VBLL probing path

Lowrank is currently unreachable from probing: `build_vbll_head` does not accept
`cov_rank` and `probing_vbll.py` has no flag for it. Stage 5's Pareto front
cannot be produced without this.

**Files:**
- Modify: `scripts/vbll_pytorch/src/vbll_experiments/models.py:272-299`
- Modify: `scripts/vbll_pytorch/probing_vbll.py:69-76`, `:186`
- Test: `tests/test_probing_vbll_smoke.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `build_vbll_head(input_dim, num_classes, *, regularization_weight=1.0,
  prior_scale=1.0, wishart_scale=0.1, parameterization="diagonal", cov_rank=2,
  vbll_type="discriminative")`. `probing_vbll.py` gains `--cov-rank` (int,
  default 2).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_probing_vbll_smoke.py`:

```python
def test_cov_rank_reaches_the_head_and_changes_its_size():
    """Lowrank was unreachable from the probing path: build_vbll_head did not
    accept cov_rank, so `--parameterization lowrank` silently built whatever
    rank vbll defaults to. The appendix Pareto front sweeps rank 2..64 and
    would have produced six identical curves."""
    from vbll_experiments.models import build_vbll_head

    def n_params(**kw):
        head = build_vbll_head(32, 5, **kw)
        return sum(p.numel() for p in head.parameters())

    diag = n_params(parameterization="diagonal")
    r2 = n_params(parameterization="lowrank", cov_rank=2)
    r4 = n_params(parameterization="lowrank", cov_rank=4)

    assert diag < r2 < r4, (
        f"cov_rank is not reaching the head: diagonal={diag} r2={r2} r4={r4}"
    )


def test_cov_rank_flag_is_exposed_by_the_probing_cli():
    import probing_vbll

    args = probing_vbll.build_argparser().parse_args([
        "--model", "dinov3_small", "--dataset", "fake", "--num-classes", "2",
        "--parameterization", "lowrank", "--cov-rank", "4",
    ])
    assert args.cov_rank == 4
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
apptainer exec --nv \
  --env PYTHONPATH=/ptmp/cheins/bllarse/src:/ptmp/cheins/bllarse/scripts/vbll_pytorch/src \
  bllarse-torch.sif \
  python -m pytest tests/test_probing_vbll_smoke.py -k cov_rank -v
```

Expected: FAIL — `TypeError: build_vbll_head() got an unexpected keyword
argument 'cov_rank'` and `unrecognized arguments: --cov-rank`.

- [ ] **Step 3: Add `cov_rank` to `build_vbll_head`**

Replace the body of `build_vbll_head` in
`scripts/vbll_pytorch/src/vbll_experiments/models.py`:

```python
def build_vbll_head(
    input_dim: int,
    num_classes: int,
    *,
    regularization_weight: float = 1.0,
    prior_scale: float = 1.0,
    wishart_scale: float = 0.1,
    parameterization: str = "diagonal",
    cov_rank: int = 2,
    vbll_type: str = "discriminative",
):
    """A bare VBLL classification head, with no backbone.

    Linear probing on cached features needs only this -- the features were
    already extracted by the JAX side, so there is no PyTorch backbone in the
    probing path at all and the two arms consume byte-identical inputs.

    `cov_rank` is accepted only by DiscClassification; GenClassification
    rejects it and supports diagonal only, exactly as VBLLFinetunedMLP already
    handles above. It is inert unless parameterization == "lowrank".
    """
    if vbll_type == "generative":
        if parameterization != "diagonal":
            raise ValueError(
                f"GenClassification supports parameterization='diagonal' only, "
                f"got {parameterization!r}"
            )
        return vbll.GenClassification(
            in_features=input_dim,
            out_features=num_classes,
            regularization_weight=regularization_weight,
            parameterization=parameterization,
            prior_scale=prior_scale,
            wishart_scale=wishart_scale,
            return_ood=True,
        )
    return vbll.DiscClassification(
        in_features=input_dim,
        out_features=num_classes,
        regularization_weight=regularization_weight,
        parameterization=parameterization,
        prior_scale=prior_scale,
        wishart_scale=wishart_scale,
        return_ood=True,
        cov_rank=cov_rank,
    )
```

- [ ] **Step 4: Add the CLI flag and thread it through**

In `scripts/vbll_pytorch/probing_vbll.py`, add to `build_argparser()` directly
after the `--parameterization` line:

```python
    p.add_argument(
        "--cov-rank", type=int, default=2,
        help="Rank of the lowrank posterior. Inert unless "
             "--parameterization lowrank. Head parameters scale as "
             "D*C*(2+cov_rank), so this is a memory dial, not a free knob -- "
             "which is why the figures fix diagonal and only the appendix "
             "Pareto front varies it.",
    )
```

And in `main`, add `cov_rank=args.cov_rank,` to the `build_vbll_head(...)` call
immediately after `parameterization=args.parameterization,`.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
apptainer exec --nv \
  --env PYTHONPATH=/ptmp/cheins/bllarse/src:/ptmp/cheins/bllarse/scripts/vbll_pytorch/src \
  bllarse-torch.sif \
  python -m pytest tests/test_probing_vbll_smoke.py -v
```

Expected: PASS, all tests in the file (the pre-existing ones must still pass —
`build_vbll_head`'s signature is keyword-only past `num_classes`, so no caller
breaks).

- [ ] **Step 6: Commit**

```bash
git add scripts/vbll_pytorch/src/vbll_experiments/models.py \
        scripts/vbll_pytorch/probing_vbll.py \
        tests/test_probing_vbll_smoke.py
git commit -m "feat(vbll): make cov_rank reachable from the probing path"
```

---

### Task 2: Log `head_params` from both arms

The appendix Pareto front plots calibration against head parameter count. That
count must come off the run, not be recomputed by a plotting script — a
recomputation is a second implementation of the thing being compared.

**Files:**
- Modify: `scripts/alternatives/probe.py`
- Modify: `scripts/vbll_pytorch/probing_vbll.py`
- Test: `tests/test_probe_smoke.py`, `tests/test_probing_vbll_smoke.py`

**Interfaces:**
- Consumes: Task 1's `build_vbll_head(..., cov_rank=...)`.
- Produces: `probe.head_params(head) -> int` and
  `probing_vbll.head_params(head) -> int`. Both scripts log `head_params` in
  their `run.log_params({...})` call.

- [ ] **Step 1: Write the failing test for the JAX arm**

Append to `tests/test_probe_smoke.py`:

```python
def test_head_params_counts_every_array_leaf():
    import equinox as eqx
    from jax import random as jr

    import probe

    head = eqx.nn.Linear(16, 3, key=jr.PRNGKey(0))
    # 16*3 weights + 3 biases
    assert probe.head_params(head) == 51


def test_head_params_is_logged_as_a_run_param(fake_features, tmp_path, monkeypatch):
    """The Pareto front plots calibration against head size. Recomputing that
    size in the plotting script would be a second implementation of the exact
    quantity being compared, and it would silently disagree the moment a head
    gains a parameter."""
    import contextlib

    import probe

    logged = {}

    class Recorder:
        active = False

        def log_params(self, params):
            logged.update(params)

        def log_metrics(self, metrics, step=None):
            pass

        def set_tags(self, tags):
            pass

    @contextlib.contextmanager
    def fake_child_run(config, *, run_name, tags=None):
        yield Recorder()

    monkeypatch.setattr(probe, "child_run", fake_child_run)

    args = probe.build_argparser().parse_args([
        "--arm", "adamw_ce", "--model", "dinov3_small", "--dataset", "fake",
        "--num-classes", "2", "--cache-dir", str(fake_features),
        "--split-cache-dir", str(tmp_path / "splits"),
        "--dump-root", str(tmp_path / "dumps"),
        "--group-id", "hp", "--uid", "hp_0",
        "--epochs", "2", "--n-dumps", "2", "--seed", "137", "--no-hf-cache",
    ])
    probe.main(args)

    # 16 feature dims x 2 classes + 2 biases
    assert logged["head_params"] == 34
```

- [ ] **Step 2: Write the failing test for the VBLL arm**

Append to `tests/test_probing_vbll_smoke.py`:

```python
def test_head_params_counts_every_head_parameter():
    from vbll_experiments.models import build_vbll_head

    import probing_vbll

    head = build_vbll_head(32, 5, parameterization="diagonal")
    expected = sum(p.numel() for p in head.parameters())
    assert probing_vbll.head_params(head) == expected
    assert expected > 0


def test_head_params_grows_with_cov_rank():
    """Ties Task 1's plumbing to the number the appendix actually plots: if
    cov_rank stopped reaching the head, head_params would go flat across the
    Pareto front's six ranks and the front would look like a vertical line."""
    from vbll_experiments.models import build_vbll_head

    import probing_vbll

    r2 = probing_vbll.head_params(
        build_vbll_head(32, 5, parameterization="lowrank", cov_rank=2)
    )
    r8 = probing_vbll.head_params(
        build_vbll_head(32, 5, parameterization="lowrank", cov_rank=8)
    )
    assert r8 > r2
```

- [ ] **Step 3: Run both tests to verify they fail**

```bash
apptainer exec --nv --env PYTHONPATH=$PWD/src bllarse-jax.sif \
  python -m pytest tests/test_probe_smoke.py -k head_params -v
apptainer exec --nv \
  --env PYTHONPATH=$PWD/src:$PWD/scripts/vbll_pytorch/src bllarse-torch.sif \
  python -m pytest tests/test_probing_vbll_smoke.py -k head_params -v
```

Expected: FAIL — `AttributeError: module 'probe' has no attribute 'head_params'`.

- [ ] **Step 4: Implement in the JAX arm**

In `scripts/alternatives/probe.py`, add after the `steps_per_epoch` function:

```python
def head_params(head) -> int:
    """Trainable parameter count of the head.

    Logged per run because Fig-appendix's rank/cost Pareto front plots
    calibration against head size: VBLL costs D*C*(2+r) and IBProbit costs
    D*(D-1)/2, independent of C, so the ranks that match IBProbit's budget
    differ by two orders of magnitude across one figure's panels. Reading the
    count off the run keeps that axis a recorded quantity.
    """
    return int(
        sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(head, eqx.is_array)))
    )
```

Add `import jax` to the import block (the file currently imports only
`jax.numpy` and `jax.random`), placed with the other third-party imports:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
```

Then extend the existing `run.log_params` call in `main`:

```python
        run.log_params({
            "steps_per_epoch": per_epoch,
            "total_steps": total_steps,
            "head_params": head_params(head),
        })
```

- [ ] **Step 5: Implement in the VBLL arm**

In `scripts/vbll_pytorch/probing_vbll.py`, add after the `_forward` function:

```python
def head_params(head) -> int:
    """Trainable parameter count of the head. See probe.head_params -- both arms
    must report this identically or the Pareto front's cost axis is not a single
    axis."""
    return int(sum(p.numel() for p in head.parameters()))
```

Then extend the existing `run.log_params` call in `main`:

```python
        run.log_params({
            "steps_per_epoch": n_batches,
            "total_steps": total_steps,
            "head_params": head_params(head),
        })
```

- [ ] **Step 6: Run both tests to verify they pass**

```bash
apptainer exec --nv --env PYTHONPATH=$PWD/src bllarse-jax.sif \
  python -m pytest tests/test_probe_smoke.py -v
apptainer exec --nv \
  --env PYTHONPATH=$PWD/src:$PWD/scripts/vbll_pytorch/src bllarse-torch.sif \
  python -m pytest tests/test_probing_vbll_smoke.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/alternatives/probe.py scripts/vbll_pytorch/probing_vbll.py \
        tests/test_probe_smoke.py tests/test_probing_vbll_smoke.py
git commit -m "feat(probe): log head_params from both arms"
```

---

### Task 3: Drop `imagenet1k` from the Fig 1 sweeps

ImageNet is a full-network-finetuning dataset only. Leaving it in the linear
probing grid means the tuning campaign would either have to cover it (2,502
steps/epoch, ~21x food101) or leave one panel untuned.

**Files:**
- Modify: `bllarse_sweeps/probe_adamw_ce.py:1-25`, `:34-41`
- Modify: `bllarse_sweeps/probe_vbll.py:14-21`
- Test: `tests/test_sweep_definitions.py:39-41`

**Interfaces:**
- Consumes: nothing.
- Produces: both Fig 1 sweeps yield 4 models x 5 datasets x 10 seeds = 200
  configs.

- [ ] **Step 1: Update the failing test**

In `tests/test_sweep_definitions.py`, replace `test_fig1_sweep_sizes_match_the_plan`:

```python
@pytest.mark.parametrize("path", [SWEEPS[0], SWEEPS[1]])
def test_fig1_sweep_sizes_match_the_plan(path):
    # 4 backbones x 5 datasets x 10 seeds. imagenet1k is deliberately absent:
    # linear probing does not cover it, it is a full-network finetuning dataset
    # only, and it will be retuned separately for that.
    assert len(get_module_from_source_path(path).create_configs()) == 200


@pytest.mark.parametrize("path", [SWEEPS[0], SWEEPS[1]])
def test_fig1_sweeps_exclude_imagenet(path):
    for c in get_module_from_source_path(path).create_configs():
        assert c["dataset"] != "imagenet1k"
```

- [ ] **Step 2: Run to verify it fails**

```bash
apptainer exec --nv --env PYTHONPATH=$PWD/src bllarse-jax.sif \
  python -m pytest tests/test_sweep_definitions.py -k fig1 -v
```

Expected: FAIL — `assert 240 == 200`.

- [ ] **Step 3: Remove the dataset from both sweeps**

In `bllarse_sweeps/probe_vbll.py` and `bllarse_sweeps/probe_adamw_ce.py`, delete
the line `"imagenet1k": (1000, 512),` from each `DATASETS` dict.

- [ ] **Step 4: Fix the stale docstring in the AdamW sweep**

Replace the module docstring of `bllarse_sweeps/probe_adamw_ce.py` (lines 1-25)
with:

```python
"""Fig 1: AdamW+CE linear probing across the DINOv3 ladder and five datasets.

AdamW+CE+TS is NOT a separate sweep -- temperature scaling is fit post-hoc from
these dumps at every dump point, so it costs no GPU time.

Measured training-set sizes (cifar10/cifar100 belong to the Fig 2 OOD sweeps,
not this grid, and are listed only for the size spread the project spans):

    | dataset             | n_train    |
    |---------------------|-----------:|
    | oxford_pets         |      3,680 |
    | dtd                 |      3,760 |
    | flowers102          |      7,169 |
    | stanford_cars       |      8,144 |
    | cifar10 / cifar100  |     50,000 |
    | food101             |     75,750 |

imagenet1k is deliberately absent: it is a full-network finetuning dataset only
and will be retuned separately for that.

The steps/epoch and total-step columns this docstring used to carry are gone
because batch_size and epochs are no longer sweep-wide constants -- both are
resolved per dataset from bllarse_sweeps/tuned_hparams.json.
"""
```

- [ ] **Step 5: Run to verify it passes**

```bash
apptainer exec --nv --env PYTHONPATH=$PWD/src bllarse-jax.sif \
  python -m pytest tests/test_sweep_definitions.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add bllarse_sweeps/probe_adamw_ce.py bllarse_sweeps/probe_vbll.py \
        tests/test_sweep_definitions.py
git commit -m "feat(sweeps): drop imagenet1k from linear probing"
```

---

### Task 4: Stage 1 — pilot sweep and per-dataset epoch budgets

Both arms were still improving at the final dump at 40 epochs (CE `val_nll`
0.1895 → 0.1806 between steps 4707 and 5880), so `at_last_step` is true and 40
epochs truncates. Comparing a truncated slow config against a converged fast one
biases the search toward high learning rates, so the budget must be settled
*before* tuning starts.

The pilot runs a fixed **step** budget, not a fixed epoch budget: food101 and
dtd differ 21x in steps/epoch, and a shared epoch count would over-train the
small datasets by that factor.

**Files:**
- Create: `bllarse_sweeps/probe_pilot_epochs.py`
- Create: `src/bllarse/tuning/__init__.py`
- Create: `src/bllarse/tuning/space.py`
- Create: `src/bllarse/tuning/budgets.py`
- Create: `tests/tuning/test_space.py`
- Create: `bllarse_sweeps/epoch_budgets.json` (cluster output)

**Interfaces:**
- Consumes: Task 3's sweep shape.
- Produces:
  - `space.N_TRAIN: dict[str, int]`, `space.DATASETS: dict[str, tuple[int, int]]`
    mapping dataset -> `(num_classes, img_size)`.
  - `space.batch_cap(n_train: int) -> int`, `space.batch_sizes(dataset: str) -> list[int]`
  - `space.steps_per_epoch(dataset: str, batch_size: int) -> int`
  - `budgets.epochs_for_steps(dataset, target_steps, batch_size) -> int`
  - `budgets.derive_budgets(dump_root, group_id, *, margin=1.25, ceiling_steps) -> dict`
  - `bllarse_sweeps/epoch_budgets.json`: `{dataset: {"batch_size": int, "epochs": int, "steps": int}}`

- [ ] **Step 1: Write the failing test for the search-space primitives**

Create `tests/tuning/test_space.py`:

```python
import pytest

from bllarse.tuning import space


# The spec's table, verbatim. Any drift here is a silent change to what the
# campaign searched, so it is pinned rather than recomputed.
EXPECTED_CAPS = {
    "oxford_pets": 1024,
    "dtd": 1024,
    "flowers102": 2048,
    "stanford_cars": 2048,
    "food101": 32768,
    "cifar10": 16384,
    "cifar100": 16384,
}


@pytest.mark.parametrize("dataset,cap", sorted(EXPECTED_CAPS.items()))
def test_batch_cap_matches_the_spec_table(dataset, cap):
    assert space.batch_cap(space.N_TRAIN[dataset]) == cap


@pytest.mark.parametrize("dataset", sorted(EXPECTED_CAPS))
def test_every_cap_leaves_at_least_two_steps_per_epoch(dataset):
    """The floor is 2, not 4, so that 1024 is reachable on dtd and oxford_pets
    -- IBProbit is expected to favour larger batches, and a batch axis that
    stops at a different point per arm would confound the comparison with the
    search space. Below 2, drop-remainder makes an epoch a single batch and the
    permutation stops doing anything."""
    cap = space.batch_cap(space.N_TRAIN[dataset])
    assert space.steps_per_epoch(dataset, cap) >= 2


@pytest.mark.parametrize("dataset", sorted(EXPECTED_CAPS))
def test_batch_ladder_is_powers_of_two_from_64_to_the_cap(dataset):
    ladder = space.batch_sizes(dataset)
    assert ladder[0] == 64
    assert ladder[-1] == space.batch_cap(space.N_TRAIN[dataset])
    assert all(b * 2 == a for b, a in zip(ladder, ladder[1:]))


def test_a_batch_cap_never_exceeds_half_the_training_set():
    for n in (100, 601, 3680, 50_000, 1_281_167):
        assert space.batch_cap(n) <= n // 2


def test_tiny_training_sets_raise_rather_than_returning_an_unusable_cap():
    """32 rows cannot support the 64 minimum. Returning 32 would quietly search
    a ladder the spec does not describe."""
    with pytest.raises(ValueError, match="too small"):
        space.batch_cap(100)
```

- [ ] **Step 2: Run to verify it fails**

```bash
apptainer exec --nv --env PYTHONPATH=$PWD/src bllarse-jax.sif \
  python -m pytest tests/tuning/test_space.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'bllarse.tuning'`.

- [ ] **Step 3: Create the package and the space primitives**

Create `src/bllarse/tuning/__init__.py` with the docstring only — the package's
public surface is filled in by Task 7 (`TUNING_SEEDS`) and Task 10
(`load_tuned`, `tuned_config`), neither of which exists yet:

```python
"""Optuna TPE tuning for the probing arms.

Separate from bllarse.tools.run_sweep by necessity: run_sweep's contract is
`create_configs() -> list[dict]`, enumerated before submission, and TPE
generates configurations adaptively from the trials already reported.
"""
```

Create `src/bllarse/tuning/space.py`:

```python
"""The tuning search space: dataset facts, batch caps, and the TPE suggestions.

Kept as data plus one thin `suggest` adapter so the caps can be tested without
optuna installed, and so the exact space that was searched is inspectable
without reading a worker.
"""

from typing import Any

# Measured training-set sizes. Fig 1's five datasets sit at res512; CIFAR sits
# at res224 by design -- its native resolution is 32x32 and upscaling to 512
# would be meaningless.
N_TRAIN = {
    "oxford_pets": 3_680,
    "dtd": 3_760,
    "flowers102": 7_169,
    "stanford_cars": 8_144,
    "food101": 75_750,
    "cifar10": 50_000,
    "cifar100": 50_000,
}

# dataset -> (num_classes, img_size)
DATASETS = {
    "oxford_pets": (37, 512),
    "dtd": (47, 512),
    "flowers102": (102, 512),
    "stanford_cars": (196, 512),
    "food101": (101, 512),
    "cifar10": (10, 224),
    "cifar100": (100, 224),
}

# The only backbone tuned. The four backbones emit different feature dimensions
# (384/768/1024/1280), so a config tuned here is not guaranteed optimal on the
# others -- accepted deliberately, since the paper's arguments centre on huge,
# and stated in the paper rather than buried.
TUNING_MODEL = "dinov3_huge"

MIN_BATCH = 64
MIN_STEPS_PER_EPOCH = 2


def batch_cap(n_train: int) -> int:
    """Largest power of two leaving at least MIN_STEPS_PER_EPOCH steps/epoch.

    A batch near n_train makes an epoch degenerate under drop-remainder, and
    probing_vbll.py raises outright at len(x_train) // batch_size == 0. The
    floor is 2 rather than 4 so that 1024 is reachable on dtd (3,760) and
    oxford_pets (3,680): IBProbit is expected to favour larger batches, and if
    the ladder stopped at a different point per arm the cross-arm comparison
    would measure the search space rather than the methods.
    """
    limit = n_train // MIN_STEPS_PER_EPOCH
    if limit < MIN_BATCH:
        raise ValueError(
            f"n_train={n_train} is too small: it cannot support the "
            f"{MIN_BATCH}-sample minimum batch at {MIN_STEPS_PER_EPOCH} "
            f"steps per epoch."
        )
    cap = MIN_BATCH
    while cap * 2 <= limit:
        cap *= 2
    return cap


def batch_sizes(dataset: str) -> list[int]:
    """The categorical ladder for `dataset`, 64 up to its cap."""
    cap = batch_cap(N_TRAIN[dataset])
    sizes, size = [], MIN_BATCH
    while size <= cap:
        sizes.append(size)
        size *= 2
    return sizes


def steps_per_epoch(dataset: str, batch_size: int) -> int:
    """Drop-remainder steps in one pass, matching both training scripts."""
    return N_TRAIN[dataset] // batch_size


def suggest(trial: Any, arm: str, dataset: str) -> dict[str, Any]:
    """One TPE-sampled configuration.

    Shared axes apply to both arms because VBLL's head is also optimised by
    AdamW. `parameterization` is fixed to diagonal and `cov_rank` is NOT
    searched: rank buys calibration with memory, and neither AdamW+CE nor
    IBProbit has an equivalent knob, so letting TPE spend it freely would let
    VBLL win Fig 1 partly by being larger. Rank is reported separately as an
    appendix Pareto front.
    """
    if arm not in ("adamw_ce", "vbll"):
        raise ValueError(f"unknown arm {arm!r}")

    params: dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", batch_sizes(dataset)),
    }
    if arm == "vbll":
        params["prior_scale"] = trial.suggest_float("prior_scale", 1e-2, 1e2, log=True)
        params["wishart_scale"] = trial.suggest_float("wishart_scale", 1e-3, 1e1, log=True)
    return params
```

- [ ] **Step 4: Run to verify the space tests pass**

```bash
apptainer exec --nv --env PYTHONPATH=$PWD/src bllarse-jax.sif \
  python -m pytest tests/tuning/test_space.py -v
```

Expected: PASS, 18 tests.

- [ ] **Step 5: Write the pilot sweep**

Create `bllarse_sweeps/probe_pilot_epochs.py`:

```python
"""Stage 1: find a per-dataset epoch budget before tuning starts.

Runs the learning-rate extremes of the tuning range at a generous FIXED STEP
budget. A fixed epoch budget would not do: food101 and dtd differ 21x in
steps/epoch, so one epoch count over-trains the small datasets by that factor.

Both arms were still improving at the final dump at 40 epochs, so the current
budget truncates. Tuning under truncation biases the search toward high
learning rates -- a fast config converges inside the budget while a slower,
possibly better one is scored at a point it had not reached yet.

Output is consumed by `python -m bllarse.tuning.budgets`, which reads the
argmin-val_nll step of every pilot run and sizes each dataset's budget to
cover it with margin.
"""

from itertools import product

from bllarse.tools.adapters import run_script_from_config
from bllarse.tuning import space

GROUP_ID = "probe_pilot_epochs"

# 4,000 steps at the pilot batch size. Generous by construction: the current
# 40-epoch food101 run is 5,880 steps at batch 512 and the argmin sits at the
# last one, but the small datasets are 280 steps under the same setting -- so
# the pilot's job is to be long enough that the argmin lands strictly inside it
# for every dataset. budgets.py reports it loudly if that fails.
PILOT_STEPS = 4_000
PILOT_BATCH = 512
PILOT_SEED = 200  # a tuning seed, never a reported one

LEARNING_RATES = [1e-4, 3e-2]  # the extremes of the tuning range

SCRIPTS = {
    "adamw_ce": "scripts/alternatives/probe.py",
    "vbll": "scripts/vbll_pytorch/probing_vbll.py",
}

BASE = dict(
    enable_mlflow=True,
    mlflow_experiment="bllarse-probing-tuning",
    n_dumps=40,
    batch_size=PILOT_BATCH,
    weight_decay=1e-4,
    val_fraction=0.1,
    model=space.TUNING_MODEL,
    seed=PILOT_SEED,
    group_id=GROUP_ID,
)

VBLL_EXTRA = dict(
    prior_scale=1.0,
    wishart_scale=0.1,
    parameterization="diagonal",
    vbll_type="discriminative",
    device="cuda",
)


def pilot_epochs(dataset: str) -> int:
    """Epochs that reach PILOT_STEPS at PILOT_BATCH, rounded up."""
    per_epoch = space.steps_per_epoch(dataset, PILOT_BATCH)
    return -(-PILOT_STEPS // per_epoch)


def create_configs():
    configs = []
    for arm, dataset, lr in product(SCRIPTS, sorted(space.DATASETS), LEARNING_RATES):
        num_classes, img_size = space.DATASETS[dataset]
        config = dict(
            **BASE,
            arm_script=arm,
            dataset=dataset,
            num_classes=num_classes,
            img_size=img_size,
            learning_rate=lr,
            epochs=pilot_epochs(dataset),
            uid=f"{arm}_{dataset}_lr{lr:g}",
        )
        if arm == "vbll":
            config.update(VBLL_EXTRA)
        else:
            config["arm"] = "adamw_ce"
        configs.append(config)
    return configs


def run(config):
    # `arm_script` selects the entry point and must not reach either argparser:
    # probe.py has --arm but probing_vbll.py has no such flag, so passing it
    # through would make every VBLL task die on an unrecognised argument.
    config = dict(config)
    script = SCRIPTS[config.pop("arm_script")]
    run_script_from_config(script, config)
```

- [ ] **Step 6: Write the budget-derivation module**

Create `src/bllarse/tuning/budgets.py`:

```python
"""Stage 1 analysis: turn pilot dumps into a per-dataset epoch budget.

The rule from the spec is "raise the budget until at_last_step is false for
>=90% of pilot configs, subject to a ceiling". That is an iteration over
submissions; this reads the same answer off ONE generous pilot in a single
pass, because the dumps carry the whole trajectory. If the pilot was itself
too short, `derive_budgets` says so rather than silently returning the
truncated value.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from bllarse.analysis.curves import run_curve
from bllarse.analysis.thresholds import early_stop
from bllarse.tuning import space

DEFAULT_OUTPUT = Path("bllarse_sweeps/epoch_budgets.json")
DEFAULT_MARGIN = 1.25


def epochs_for_steps(dataset: str, target_steps: int, batch_size: int) -> int:
    """Epochs needed to reach `target_steps`, rounded up. Never below 1."""
    per_epoch = space.steps_per_epoch(dataset, batch_size)
    if per_epoch < 1:
        raise ValueError(
            f"batch_size={batch_size} exceeds n_train for {dataset}"
        )
    return max(1, -(-target_steps // per_epoch))


def _uid_dataset(uid: str) -> str:
    """`adamw_ce_food101_lr0.0001` -> `food101`. Longest match wins, so
    cifar100 is never read as cifar10."""
    for dataset in sorted(space.DATASETS, key=len, reverse=True):
        if f"_{dataset}_" in uid:
            return dataset
    raise ValueError(f"no known dataset in pilot uid {uid!r}")


def derive_budgets(
    dump_root: Any,
    group_id: str,
    *,
    batch_size: int,
    margin: float = DEFAULT_MARGIN,
    ceiling_steps: int,
) -> dict[str, dict[str, int]]:
    """Per-dataset step and epoch budget covering every pilot run's argmin.

    Takes the MAX argmin step over that dataset's pilot runs, not the mean: the
    budget has to be adequate for the slowest learning rate in the range, since
    that is exactly the config a too-short budget would unfairly score.
    """
    root = Path(dump_root)
    by_dataset: dict[str, list[int]] = defaultdict(list)
    truncated: dict[str, list[str]] = defaultdict(list)

    group_dir = root / group_id
    if not group_dir.is_dir():
        raise FileNotFoundError(f"no pilot dumps at {group_dir}")

    for uid_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
        uid = uid_dir.name
        dataset = _uid_dataset(uid)
        stop = early_stop(run_curve(root, group_id, uid), "val_nll")
        by_dataset[dataset].append(stop.step)
        if stop.at_last_step:
            truncated[dataset].append(uid)

    budgets: dict[str, dict[str, int]] = {}
    for dataset, steps in sorted(by_dataset.items()):
        if truncated[dataset]:
            raise ValueError(
                f"{dataset}: {len(truncated[dataset])} of {len(steps)} pilot "
                f"runs put their val_nll minimum at the FINAL dumped step "
                f"({', '.join(truncated[dataset])}). The pilot budget itself "
                f"truncated, so the derived budget would be a lower bound of "
                f"unknown looseness. Raise PILOT_STEPS and re-run the pilot."
            )
        target = min(int(max(steps) * margin), ceiling_steps)
        budgets[dataset] = {
            "batch_size": batch_size,
            "steps": target,
            "epochs": epochs_for_steps(dataset, target, batch_size),
        }
    return budgets


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump-root", required=True)
    p.add_argument("--group-id", default="probe_pilot_epochs")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    p.add_argument(
        "--ceiling-steps", type=int, default=8_000,
        help="Hard cap, so a pathological learning rate cannot drive the "
             "budget without bound.",
    )
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    budgets = derive_budgets(
        args.dump_root, args.group_id,
        batch_size=args.batch_size,
        margin=args.margin,
        ceiling_steps=args.ceiling_steps,
    )
    args.output.write_text(json.dumps(budgets, indent=2, sort_keys=True) + "\n")
    for dataset, budget in sorted(budgets.items()):
        print(f"[budgets] {dataset:15s} {budget['steps']:6d} steps "
              f"= {budget['epochs']:5d} epochs @ batch {budget['batch_size']}")
    print(f"[budgets] wrote {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Write the budget test**

Create `tests/tuning/test_budgets.py`:

```python
import numpy as np
import pytest

from bllarse import evaldump
from bllarse.tuning import budgets


def _write_curve(root, group_id, uid, *, argmin_at, n_steps):
    """A synthetic run whose val_nll is minimised at `argmin_at`."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=64)
    for step in range(1, n_steps + 1):
        # Confidence rises to a peak at argmin_at then falls back, so val_nll
        # has its minimum exactly there.
        sharpness = 4.0 - abs(step - argmin_at) * 0.1
        logits = np.zeros((64, 2), dtype=np.float32)
        logits[np.arange(64), labels] = sharpness
        for split in ("val", "test"):
            evaldump.write_dump(
                root, group_id, uid, step, logits, labels,
                arm="adamw_ce", split=split, logit_kind="softmax_logits",
                seed=200, samples_seen=step * 512,
            )


def test_epochs_for_steps_rounds_up():
    # food101: 75,750 // 512 = 147 steps/epoch. 4,000 steps needs 28 epochs
    # (27 would be 3,969).
    assert budgets.epochs_for_steps("food101", 4_000, 512) == 28


def test_budget_covers_the_slowest_run_with_margin(tmp_path):
    """Max over the dataset's pilot runs, not mean: the budget has to be
    adequate for the slowest learning rate in the range, which is exactly the
    config a too-short budget would unfairly score."""
    root = tmp_path / "dumps"
    _write_curve(root, "pilot", "adamw_ce_food101_lr0.0001", argmin_at=40, n_steps=60)
    _write_curve(root, "pilot", "adamw_ce_food101_lr0.03", argmin_at=10, n_steps=60)

    out = budgets.derive_budgets(
        root, "pilot", batch_size=512, margin=1.25, ceiling_steps=10_000
    )
    assert out["food101"]["steps"] == 50          # 40 * 1.25, not 10 or 25
    assert out["food101"]["epochs"] == 1          # 50 steps < 147 per epoch


def test_the_ceiling_bounds_a_pathological_run(tmp_path):
    root = tmp_path / "dumps"
    _write_curve(root, "pilot", "adamw_ce_dtd_lr0.0001", argmin_at=40, n_steps=60)

    out = budgets.derive_budgets(
        root, "pilot", batch_size=512, margin=1.25, ceiling_steps=42
    )
    assert out["dtd"]["steps"] == 42


def test_a_truncated_pilot_raises_instead_of_returning_a_lower_bound(tmp_path):
    """If the argmin is at the final dumped step the curve was still improving,
    so the budget derived from it is a lower bound of unknown looseness. That
    must fail loudly -- it is precisely the truncation the pilot exists to
    rule out."""
    root = tmp_path / "dumps"
    _write_curve(root, "pilot", "adamw_ce_dtd_lr0.03", argmin_at=60, n_steps=60)

    with pytest.raises(ValueError, match="FINAL dumped step"):
        budgets.derive_budgets(
            root, "pilot", batch_size=512, margin=1.25, ceiling_steps=10_000
        )


def test_cifar100_is_not_read_as_cifar10():
    assert budgets._uid_dataset("vbll_cifar100_lr0.001") == "cifar100"
    assert budgets._uid_dataset("vbll_cifar10_lr0.001") == "cifar10"
```

- [ ] **Step 8: Put the pilot sweep under the existing contract tests**

In `tests/test_sweep_definitions.py`, append to the `SWEEPS` list:

```python
SWEEPS = [
    "bllarse_sweeps/probe_adamw_ce.py",
    "bllarse_sweeps/probe_vbll.py",
    "bllarse_sweeps/probe_ood_adamw_ce.py",
    "bllarse_sweeps/probe_ood_vbll.py",
    "bllarse_sweeps/probe_pilot_epochs.py",
]
```

Append, never insert: `test_fig1_sweep_sizes_match_the_plan` and
`test_fig2_sweep_sizes_match_the_plan` index this list positionally
(`SWEEPS[0]`..`SWEEPS[3]`), so an insertion would silently re-point them at the
wrong sweeps.

- [ ] **Step 9: Run the budget and sweep tests**

```bash
apptainer exec --nv --env PYTHONPATH=$PWD/src bllarse-jax.sif \
  python -m pytest tests/tuning/ tests/test_sweep_definitions.py -v
```

Expected: PASS. The pilot sweep inherits the uniqueness, single-group-id and
well-formedness checks; it satisfies them because every uid embeds arm, dataset
and learning rate, and `space.TUNING_MODEL` is `dinov3_huge`, which is in
`LADDER`.

- [ ] **Step 10: Commit the code**

```bash
git add bllarse_sweeps/probe_pilot_epochs.py src/bllarse/tuning/ tests/tuning/ \
        tests/test_sweep_definitions.py
git commit -m "feat(tuning): pilot sweep and per-dataset epoch budgets"
```

- [ ] **Step 11: CLUSTER — submit the pilot**

28 configs, one seed, ~4,000 steps each. At the post-jit rate (~2.2 ms/step for
CE at D=1280, C=101) this is well under an hour of GPU time in total.

```bash
# On Raven, from the staged repo root.
export BLLARSE_SIF=$PWD/bllarse-jax.sif
python -m bllarse.tools.run_sweep bllarse_sweeps/probe_pilot_epochs.py \
  --partition gpu --gres gpu:a100:1 --max-concurrent 7 \
  --parent-run-name probe_pilot_epochs
```

The pilot spans both arms but `run_sweep` submits one image. Submit twice with
`--index-offset`/`--num-jobs` splitting the CE half from the VBLL half, or
submit each arm's slice with its own `BLLARSE_SIF`. `create_configs()` orders
`adamw_ce` before `vbll` (`SCRIPTS` insertion order), so the first 14 configs
are CE and the last 14 are VBLL:

```bash
BLLARSE_SIF=$PWD/bllarse-jax.sif python -m bllarse.tools.run_sweep \
  bllarse_sweeps/probe_pilot_epochs.py --index-offset 0  --num-jobs 14 \
  --partition gpu --gres gpu:a100:1
BLLARSE_SIF=$PWD/bllarse-torch.sif python -m bllarse.tools.run_sweep \
  bllarse_sweeps/probe_pilot_epochs.py --index-offset 14 --num-jobs 14 \
  --partition gpu --gres gpu:a100:1
```

- [ ] **Step 12: CLUSTER — derive and commit the budgets**

```bash
python -m bllarse.tuning.budgets \
  --dump-root "$BLLARSE_EVAL_DUMP_ROOT" \
  --group-id probe_pilot_epochs \
  -o bllarse_sweeps/epoch_budgets.json
```

If it raises `FINAL dumped step`, raise `PILOT_STEPS` in
`bllarse_sweeps/probe_pilot_epochs.py` and re-run the pilot for the affected
datasets. Do not lower `--ceiling-steps` to make the error go away — the error
is about truncation, and the ceiling does not fix truncation.

```bash
git add bllarse_sweeps/epoch_budgets.json
git commit -m "feat(tuning): per-dataset epoch budgets from the Stage 1 pilot"
```

---

### Task 5: Add an `on_dump` callback to both training scripts

Optuna's pruner needs an intermediate value *during* a run: the whole saving is
killing a hopeless config before its second seed. Both scripts already compute
exactly the right scalar at exactly the right moment — `report_step` returns a
dict containing `val_nll` at every dumped step — so this is a hook, not new
measurement code.

**Files:**
- Modify: `scripts/alternatives/probe.py` (`main` signature, dump block)
- Modify: `scripts/vbll_pytorch/probing_vbll.py` (`main` signature, dump block)
- Test: `tests/test_probe_smoke.py`, `tests/test_probing_vbll_smoke.py`

**Interfaces:**
- Consumes: Task 2's edits to the same two functions.
- Produces: `main(args, on_dump=None)` in both scripts, where `on_dump` is
  `Callable[[int, int, dict[str, float]], None]` receiving
  `(dump_index, step, scalars)`. `dump_index` counts from 0 and is the pruner's
  step key; `scalars` is `report_step`'s return value. Exceptions raised by the
  callback propagate — that is how `optuna.TrialPruned` escapes the loop.

- [ ] **Step 1: Write the failing test for the JAX arm**

Append to `tests/test_probe_smoke.py`:

```python
def test_on_dump_is_called_once_per_dump_with_val_nll(fake_features, tmp_path):
    import probe

    seen = []
    args = probe.build_argparser().parse_args([
        "--arm", "adamw_ce", "--model", "dinov3_small", "--dataset", "fake",
        "--num-classes", "2", "--cache-dir", str(fake_features),
        "--split-cache-dir", str(tmp_path / "splits"),
        "--dump-root", str(tmp_path / "dumps"),
        "--group-id", "cb", "--uid", "cb_0",
        "--epochs", "10", "--n-dumps", "5", "--batch-size", "100",
        "--seed", "137", "--no-hf-cache",
    ])
    probe.main(args, on_dump=lambda i, step, scalars: seen.append((i, step, scalars)))

    assert len(seen) == 5, "one callback per dumped step"
    assert [i for i, _, _ in seen] == [0, 1, 2, 3, 4], (
        "dump_index must count dumps, not optimiser steps -- it is the pruner's "
        "step key, and trials vary batch_size so raw steps are not comparable"
    )
    assert all("val_nll" in scalars for _, _, scalars in seen)


def test_an_exception_from_on_dump_propagates(fake_features, tmp_path):
    """optuna.TrialPruned is raised from inside the callback and must escape
    the training loop. Swallowing it would run every pruned trial to
    completion -- the pruner would cost time instead of saving it."""
    import probe

    class Sentinel(Exception):
        pass

    def boom(i, step, scalars):
        raise Sentinel()

    args = probe.build_argparser().parse_args([
        "--arm", "adamw_ce", "--model", "dinov3_small", "--dataset", "fake",
        "--num-classes", "2", "--cache-dir", str(fake_features),
        "--split-cache-dir", str(tmp_path / "splits"),
        "--dump-root", str(tmp_path / "dumps"),
        "--group-id", "cb", "--uid", "cb_boom",
        "--epochs", "4", "--n-dumps", "3", "--batch-size", "100",
        "--seed", "137", "--no-hf-cache",
    ])
    with pytest.raises(Sentinel):
        probe.main(args, on_dump=boom)
```

- [ ] **Step 2: Write the same test for the VBLL arm**

Append to `tests/test_probing_vbll_smoke.py`:

```python
def test_on_dump_is_called_once_per_dump_with_val_nll(fake_features, tmp_path):
    import probing_vbll

    seen = []
    args = probing_vbll.build_argparser().parse_args([
        "--model", "dinov3_small", "--dataset", "fake", "--num-classes", "2",
        "--cache-dir", str(fake_features),
        "--split-cache-dir", str(tmp_path / "splits"),
        "--dump-root", str(tmp_path / "dumps"),
        "--group-id", "cb", "--uid", "cb_0",
        "--epochs", "10", "--n-dumps", "5", "--batch-size", "100",
        "--seed", "137", "--device", "cpu",
    ])
    probing_vbll.main(args, on_dump=lambda i, step, s: seen.append((i, step, s)))

    assert len(seen) == 5
    assert [i for i, _, _ in seen] == [0, 1, 2, 3, 4]
    assert all("val_nll" in scalars for _, _, scalars in seen)


def test_an_exception_from_on_dump_propagates(fake_features, tmp_path):
    import probing_vbll

    class Sentinel(Exception):
        pass

    args = probing_vbll.build_argparser().parse_args([
        "--model", "dinov3_small", "--dataset", "fake", "--num-classes", "2",
        "--cache-dir", str(fake_features),
        "--split-cache-dir", str(tmp_path / "splits"),
        "--dump-root", str(tmp_path / "dumps"),
        "--group-id", "cb", "--uid", "cb_boom",
        "--epochs", "4", "--n-dumps", "3", "--batch-size", "100",
        "--seed", "137", "--device", "cpu",
    ])
    with pytest.raises(Sentinel):
        probing_vbll.main(
            args, on_dump=lambda i, step, s: (_ for _ in ()).throw(Sentinel())
        )
```

- [ ] **Step 3: Run both to verify they fail**

```bash
apptainer exec --nv --env PYTHONPATH=$PWD/src bllarse-jax.sif \
  python -m pytest tests/test_probe_smoke.py -k on_dump -v
apptainer exec --nv \
  --env PYTHONPATH=$PWD/src:$PWD/scripts/vbll_pytorch/src bllarse-torch.sif \
  python -m pytest tests/test_probing_vbll_smoke.py -k on_dump -v
```

Expected: FAIL — `TypeError: main() got an unexpected keyword argument 'on_dump'`.

- [ ] **Step 4: Implement in the JAX arm**

In `scripts/alternatives/probe.py`, change the signature and add a counter:

```python
def main(args, on_dump=None):
    """Train one head. `on_dump(dump_index, step, scalars)` fires after every
    scheduled dump, if given.

    The index counts dumps rather than optimiser steps because it is Optuna's
    pruning key: MedianPruner compares trials at the SAME key, and trials vary
    batch_size, so two configs would share almost no raw step values. n_dumps is
    40 for every run, so the dump index is a common grid by construction.
    """
```

Then in the dump block, immediately after the `print(...)` that reports the
step's scalars:

```python
                if on_dump is not None:
                    on_dump(dump_index, step, scalars)
                dump_index += 1
```

Initialise `dump_index = 0` next to `step = 0`, before the epoch loop:

```python
        step = 0
        dump_index = 0
```

- [ ] **Step 5: Implement in the VBLL arm**

Make the identical change in `scripts/vbll_pytorch/probing_vbll.py`: the
signature becomes `def main(args, on_dump=None):` with the same docstring
addition, `dump_index = 0` beside `step = 0`, and the same two lines after the
`print(...)`.

- [ ] **Step 6: Run both to verify they pass**

```bash
apptainer exec --nv --env PYTHONPATH=$PWD/src bllarse-jax.sif \
  python -m pytest tests/test_probe_smoke.py -v
apptainer exec --nv \
  --env PYTHONPATH=$PWD/src:$PWD/scripts/vbll_pytorch/src bllarse-torch.sif \
  python -m pytest tests/test_probing_vbll_smoke.py -v
```

Expected: PASS. `run_script_from_config` calls `main(args)` positionally and is
unaffected — `on_dump` defaults to `None`.

- [ ] **Step 7: Commit**

```bash
git add scripts/alternatives/probe.py scripts/vbll_pytorch/probing_vbll.py \
        tests/test_probe_smoke.py tests/test_probing_vbll_smoke.py
git commit -m "feat(probe): per-dump callback so a trial can be pruned mid-run"
```

---

### Task 6: Expose a public script loader on `adapters`

The objective must call `main(args, on_dump=...)`, which
`run_script_from_config` cannot do — it calls `main(args)`. Rather than
duplicate its path resolution (which knows about `BLLARSE_REPO_ROOT` and the
`scripts/` prefix), promote the private helper.

**Files:**
- Modify: `src/bllarse/tools/adapters.py:48-62`, `:75-91`
- Test: `tests/tools/test_adapters.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `adapters.load_script_module(script_path: str | Path) -> ModuleType`
  and `adapters.parse_config(module, config: Mapping[str, Any]) -> Namespace`.

- [ ] **Step 1: Write the failing test**

Append to `tests/tools/test_adapters.py`:

```python
def test_load_script_module_resolves_a_repo_relative_script():
    from bllarse.tools import adapters

    module = adapters.load_script_module("scripts/alternatives/probe.py")
    assert callable(module.main)
    assert callable(module.build_argparser)


def test_parse_config_produces_the_same_namespace_as_the_cli():
    from bllarse.tools import adapters

    module = adapters.load_script_module("scripts/alternatives/probe.py")
    args = adapters.parse_config(module, {
        "model": "dinov3_huge", "dataset": "food101", "num_classes": 101,
        "learning_rate": 5e-3,
    })
    assert args.model == "dinov3_huge"
    assert args.learning_rate == 5e-3
```

- [ ] **Step 2: Run to verify it fails**

```bash
apptainer exec --nv --env PYTHONPATH=$PWD/src bllarse-jax.sif \
  python -m pytest tests/tools/test_adapters.py -k "load_script_module or parse_config" -v
```

Expected: FAIL — `AttributeError: module 'bllarse.tools.adapters' has no
attribute 'load_script_module'`.

- [ ] **Step 3: Promote the helpers**

In `src/bllarse/tools/adapters.py`, rename `_resolve_script_path` to
`resolve_script_path` (keeping its body verbatim), and add:

```python
def load_script_module(script_path: str | Path) -> Any:
    """Import a training script as a module, resolving repo-relative paths.

    Public because the tuning objective needs `main(args, on_dump=...)`, which
    run_script_from_config cannot express -- and duplicating the resolution
    would duplicate its knowledge of BLLARSE_REPO_ROOT and the scripts/ prefix.
    """
    return get_module_from_source_path(resolve_script_path(script_path))


def parse_config(module: Any, config: Mapping[str, Any]) -> Namespace:
    """Parse a config mapping with a script's own argparser."""
    return module.build_argparser().parse_args(config_to_argv(config))
```

Then rewrite `run_script_from_config` to use them, so there is one path:

```python
def run_script_from_config(
    script_path: str | Path,
    config: Mapping[str, Any],
) -> None:
    """Parse a config with a script's CLI and invoke its main function."""
    module = load_script_module(script_path)
    args = parse_config(module, config)

    train_main = module.main
    if not hasattr(module, "build_configs"):
        train_main(args)
        return

    model_config, optimizer_config = module.build_configs(args)
    train_main(args, model_config, optimizer_config)
```

- [ ] **Step 4: Run the full adapters suite**

```bash
apptainer exec --nv --env PYTHONPATH=$PWD/src bllarse-jax.sif \
  python -m pytest tests/tools/ -v
```

Expected: PASS, including the pre-existing `test_adapters.py` and
`test_adapters_vbll.py`.

- [ ] **Step 5: Commit**

```bash
git add src/bllarse/tools/adapters.py tests/tools/test_adapters.py
git commit -m "refactor(adapters): expose load_script_module and parse_config"
```

---

### Task 7: The trial objective

One trial = one config run on both tuning seeds, scored by the mean
early-stopped `val_nll`. The scoring function *is*
`analysis.thresholds.early_stop`, so the tuning criterion and the reported
stopping rule cannot drift apart.

**Files:**
- Create: `src/bllarse/tuning/objective.py`
- Modify: `src/bllarse/tuning/__init__.py`
- Test: `tests/tuning/test_objective.py`

**Interfaces:**
- Consumes: `adapters.load_script_module`, `adapters.parse_config` (Task 6);
  `main(args, on_dump=...)` (Task 5); `space.DATASETS`, `space.TUNING_MODEL`
  (Task 4).
- Produces:
  - `objective.TUNING_SEEDS = (200, 201)`
  - `objective.SCRIPTS: dict[str, str]` mapping arm -> script path
  - `objective.score_dumps(dump_root, group_id, uids) -> float`
  - `objective.evaluate(arm, config, *, dump_root, group_id, uid, on_dump=None,
    keep_dumps=False) -> float`

- [ ] **Step 1: Write the failing test**

Create `tests/tuning/test_objective.py`:

```python
import numpy as np
import pytest

from bllarse import evaldump
from bllarse.tuning import objective


def _write_curve(root, group_id, uid, *, best_nll_at, n_steps=6):
    """A synthetic run with a controllable val_nll minimum."""
    labels = np.zeros(64, dtype=np.int64)
    for step in range(1, n_steps + 1):
        sharpness = 5.0 - abs(step - best_nll_at)
        logits = np.zeros((64, 2), dtype=np.float32)
        logits[:, 0] = sharpness
        for split in ("val", "test"):
            evaldump.write_dump(
                root, group_id, uid, step, logits, labels,
                arm="adamw_ce", split=split, logit_kind="softmax_logits",
                seed=200, samples_seen=step * 128,
            )


def test_score_is_the_mean_over_seeds_of_the_early_stopped_val_nll(tmp_path):
    root = tmp_path / "dumps"
    _write_curve(root, "t", "trial0_seed200", best_nll_at=3)
    _write_curve(root, "t", "trial0_seed201", best_nll_at=5)

    from bllarse.analysis.curves import run_curve
    from bllarse.analysis.thresholds import early_stop

    want = np.mean([
        early_stop(run_curve(root, "t", "trial0_seed200"), "val_nll").value,
        early_stop(run_curve(root, "t", "trial0_seed201"), "val_nll").value,
    ])
    got = objective.score_dumps(root, "t", ["trial0_seed200", "trial0_seed201"])
    assert got == pytest.approx(want)


def test_tuning_seeds_are_disjoint_from_the_reported_range():
    """Fig 1 and Fig 2 report range(137, 147). Selecting hyperparameters on a
    reported seed tunes against the same random draws the error bars are drawn
    from. The earlier IBProbit/CE sweeps used 143 and 145 -- inside the range --
    and this deliberately does not repeat that."""
    assert set(objective.TUNING_SEEDS).isdisjoint(range(137, 147))


def test_evaluate_runs_both_seeds_and_removes_its_dumps(tmp_path, monkeypatch):
    """1,120 trials x 2 seeds x 40 dumps x 2 splits is ~179k files. Trial dumps
    have no reporting value once the objective is computed -- Stage 4 re-runs
    the winner on the reported seeds -- so leaving them would fill /ptmp for
    nothing."""
    root = tmp_path / "dumps"
    seeds_seen = []

    class FakeModule:
        @staticmethod
        def build_argparser():
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument("--seed", type=int)
            p.add_argument("--uid")
            p.add_argument("--group-id")
            p.add_argument("--dump-root")
            return p

        @staticmethod
        def main(args, on_dump=None):
            seeds_seen.append(args.seed)
            _write_curve(args.dump_root, args.group_id, args.uid, best_nll_at=2)
            if on_dump is not None:
                on_dump(0, 1, {"val_nll": 0.5})

    monkeypatch.setattr(
        objective.adapters, "load_script_module", lambda path: FakeModule
    )

    value = objective.evaluate(
        "adamw_ce", {}, dump_root=root, group_id="t", uid="trial0"
    )
    assert seeds_seen == list(objective.TUNING_SEEDS)
    assert isinstance(value, float)
    assert not (root / "t" / "trial0_seed200").exists()
    assert not (root / "t" / "trial0_seed201").exists()


def test_on_dump_fires_for_the_first_seed_only(tmp_path, monkeypatch):
    """Pruning a hopeless config before its second seed is where most of the
    saving comes from. Reporting from both seeds would report two different
    trajectories under one trial's step keys."""
    root = tmp_path / "dumps"
    calls = []

    class FakeModule:
        @staticmethod
        def build_argparser():
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument("--seed", type=int)
            p.add_argument("--uid")
            p.add_argument("--group-id")
            p.add_argument("--dump-root")
            return p

        @staticmethod
        def main(args, on_dump=None):
            _write_curve(args.dump_root, args.group_id, args.uid, best_nll_at=2)
            calls.append((args.seed, on_dump is not None))

    monkeypatch.setattr(
        objective.adapters, "load_script_module", lambda path: FakeModule
    )
    objective.evaluate(
        "adamw_ce", {}, dump_root=root, group_id="t", uid="t0",
        on_dump=lambda i, step, scalars: None,
    )
    assert calls == [(200, True), (201, False)]
```

- [ ] **Step 2: Run to verify it fails**

```bash
apptainer exec --nv --env PYTHONPATH=$PWD/src bllarse-jax.sif \
  python -m pytest tests/tuning/test_objective.py -v
```

Expected: FAIL — `ImportError: cannot import name 'objective'`.

- [ ] **Step 3: Implement the objective**

Create `src/bllarse/tuning/objective.py`:

```python
"""One Optuna trial: run a config on both tuning seeds, return its score.

The score IS analysis.thresholds.early_stop, not a reimplementation of it, so
the tuning criterion and the reported stopping rule cannot drift apart.
Selection is on val, reporting is on test, and bllarse.splits makes the two
disjoint by construction -- so this does not leak. The OOD split never enters
tuning at all: for Fig 2, studies are keyed on the ID dataset only, since both
scripts load OOD features after the training loop purely for a final dump.
"""

import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from bllarse.analysis.curves import run_curve
from bllarse.analysis.thresholds import early_stop
from bllarse.tools import adapters

# Disjoint from Fig 1/2's range(137, 147). Selecting hyperparameters on seeds
# that later produce the reported error bars would tune against the same random
# draws being reported.
TUNING_SEEDS = (200, 201)

SCRIPTS = {
    "adamw_ce": "scripts/alternatives/probe.py",
    "vbll": "scripts/vbll_pytorch/probing_vbll.py",
}

OnDump = Callable[[int, int, Mapping[str, float]], None]


def score_dumps(dump_root: Any, group_id: str, uids: Sequence[str]) -> float:
    """Mean over runs of the early-stopped val_nll. Lower is better."""
    values = [
        early_stop(run_curve(dump_root, group_id, uid), "val_nll").value
        for uid in uids
    ]
    if not values:
        raise ValueError("no runs to score")
    return float(sum(values) / len(values))


def evaluate(
    arm: str,
    config: Mapping[str, Any],
    *,
    dump_root: Any,
    group_id: str,
    uid: str,
    on_dump: OnDump | None = None,
    keep_dumps: bool = False,
) -> float:
    """Run `config` on both tuning seeds and return its objective value.

    `on_dump` fires during the FIRST seed only. Pruning a hopeless config before
    its second seed is where most of the saving comes from, and reporting from
    both would push two different trajectories through one trial's step keys --
    MedianPruner compares trials at matching keys, so that would corrupt every
    comparison it makes.

    Dumps are removed unless `keep_dumps`: a full campaign is ~179k files that
    nothing downstream reads, since Stage 4 re-runs the winning config on the
    reported seeds.
    """
    if arm not in SCRIPTS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(SCRIPTS)}")

    module = adapters.load_script_module(SCRIPTS[arm])
    uids = [f"{uid}_seed{seed}" for seed in TUNING_SEEDS]

    try:
        for index, (seed, run_uid) in enumerate(zip(TUNING_SEEDS, uids)):
            args = adapters.parse_config(module, {
                **config,
                "seed": seed,
                "uid": run_uid,
                "group_id": group_id,
                "dump_root": str(dump_root),
            })
            module.main(args, on_dump=on_dump if index == 0 else None)
        return score_dumps(dump_root, group_id, uids)
    finally:
        if not keep_dumps:
            for run_uid in uids:
                shutil.rmtree(Path(dump_root) / group_id / run_uid, ignore_errors=True)
```

- [ ] **Step 4: Export the seeds from the package**

Replace `src/bllarse/tuning/__init__.py` with:

```python
"""Optuna TPE tuning for the probing arms.

Separate from bllarse.tools.run_sweep by necessity: run_sweep's contract is
`create_configs() -> list[dict]`, enumerated before submission, and TPE
generates configurations adaptively from the trials already reported.
"""

from bllarse.tuning.objective import TUNING_SEEDS

__all__ = ["TUNING_SEEDS"]
```

- [ ] **Step 5: Run to verify it passes**

```bash
apptainer exec --nv --env PYTHONPATH=$PWD/src bllarse-jax.sif \
  python -m pytest tests/tuning/ -v
```

Expected: PASS.

- [ ] **Step 6: Add an end-to-end trial test**

Append to `tests/tuning/test_objective.py`:

```python
def test_a_real_trial_runs_end_to_end_on_the_jax_arm(tmp_path):
    """Proves the whole trial path -- adapters resolution, both seeds, the
    on_dump hook, run_curve, early_stop, cleanup -- against the real probe.py
    rather than a fake module."""
    rng = np.random.default_rng(0)
    cache = tmp_path / "features"
    cache.mkdir()
    for split, n in (("train", 600), ("test", 200)):
        labels = rng.integers(0, 2, size=n)
        feats = rng.normal(size=(n, 16)).astype(np.float32)
        feats[labels == 1, 0] += 6.0
        np.savez(cache / f"dinov3_small_fake_res224_{split}.npz",
                 features=feats, labels=labels)

    reported = []
    value = objective.evaluate(
        "adamw_ce",
        {
            "arm": "adamw_ce", "model": "dinov3_small", "dataset": "fake",
            "num_classes": 2, "cache_dir": str(cache),
            "split_cache_dir": str(tmp_path / "splits"),
            "epochs": 5, "n_dumps": 4, "batch_size": 100, "no_hf_cache": True,
        },
        dump_root=tmp_path / "dumps", group_id="e2e", uid="trial0",
        on_dump=lambda i, step, s: reported.append(i),
    )

    assert value > 0.0
    assert reported == [0, 1, 2, 3], "intermediates come from the first seed only"
    assert not (tmp_path / "dumps" / "e2e" / "trial0_seed200").exists()
```

- [ ] **Step 7: Run it**

```bash
apptainer exec --nv --env PYTHONPATH=$PWD/src bllarse-jax.sif \
  python -m pytest tests/tuning/test_objective.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/bllarse/tuning/objective.py src/bllarse/tuning/__init__.py \
        tests/tuning/test_objective.py
git commit -m "feat(tuning): trial objective on the shared early-stopping rule"
```

---

### Task 8: The Optuna worker

Four workers share one study per `(arm, dataset)` through a journal file. This
is the task that first imports optuna, so it also adds the dependency to both
images.

**Files:**
- Create: `src/bllarse/tuning/worker.py`
- Modify: `src/slurm/images/bllarse-jax.def`
- Modify: `src/slurm/images/bllarse-torch.def`
- Test: `tests/tuning/test_worker.py`

**Interfaces:**
- Consumes: `space.suggest`, `space.DATASETS`, `space.TUNING_MODEL` (Task 4);
  `objective.evaluate` (Task 7); `bllarse_sweeps/epoch_budgets.json` (Task 4).
- Produces:
  - `worker.study_name(arm, dataset) -> str`
  - `worker.open_study(arm, dataset, storage_dir) -> optuna.Study`
  - `worker.make_objective(arm, dataset, *, budgets, dump_root, **kw) -> Callable`
  - `python -m bllarse.tuning.worker --arm ARM --dataset DS ...`

- [ ] **Step 1: Add optuna to both images**

In `src/slurm/images/bllarse-jax.def`, extend the pip line:

```
    pip install --no-cache-dir \
        "jax[cuda12]" equinox optax "tfp-nightly[jax]" \
        datasets equimo augmax mlflow python-dotenv tqdm "optuna>=4.0" \
        "mlpox @ git+https://github.com/dimarkov/mlpox.git@main" \
        "blrax @ git+https://github.com/dimarkov/blrax.git@main"
```

In `src/slurm/images/bllarse-torch.def`, extend the first pip line:

```
    pip install --no-cache-dir \
        torch torchvision "vbll>=0.2.0" mlflow numpy tqdm huggingface_hub "optuna>=4.0"
```

`>=4.0` is load-bearing: `JournalFileBackend` moved to
`optuna.storages.journal` in 4.0, and the worker imports it from there.

- [ ] **Step 2: Rebuild both images on the cluster**

```bash
apptainer build bllarse-jax.sif   src/slurm/images/bllarse-jax.def
apptainer build bllarse-torch.sif src/slurm/images/bllarse-torch.def
apptainer exec bllarse-jax.sif   python -c "import optuna; print(optuna.__version__)"
apptainer exec bllarse-torch.sif python -c "import optuna; print(optuna.__version__)"
```

Expected: both print `4.x`.

- [ ] **Step 3: Write the failing test**

Create `tests/tuning/test_worker.py`:

```python
import json

import pytest

optuna = pytest.importorskip("optuna")

from bllarse.tuning import space, worker


def test_study_names_are_unique_per_arm_and_dataset():
    names = {
        worker.study_name(arm, ds)
        for arm in ("adamw_ce", "vbll")
        for ds in space.DATASETS
    }
    assert len(names) == 2 * len(space.DATASETS) == 14


def test_workers_sharing_a_storage_dir_share_one_study(tmp_path):
    """Four workers per study, one journal file per study. SQLite over GPFS with
    concurrent writers is unreliable; the journal backend is built for network
    filesystems, and one file per study means the 14 studies never contend."""
    first = worker.open_study("vbll", "food101", tmp_path)
    first.tell(first.ask(), 1.23)

    second = worker.open_study("vbll", "food101", tmp_path)
    assert len(second.trials) == 1, "the second worker must see the first's trial"
    assert second.study_name == first.study_name


def test_the_study_minimises():
    assert worker.DIRECTION == "minimize"


def test_suggested_batch_sizes_never_exceed_the_dataset_cap(tmp_path):
    study = worker.open_study("adamw_ce", "dtd", tmp_path)
    cap = space.batch_cap(space.N_TRAIN["dtd"])
    for _ in range(20):
        trial = study.ask()
        params = space.suggest(trial, "adamw_ce", "dtd")
        assert params["batch_size"] <= cap == 1024
        study.tell(trial, 1.0)


def test_the_trial_config_carries_the_datasets_epoch_budget(tmp_path):
    budgets = {"dtd": {"batch_size": 512, "steps": 2000, "epochs": 273}}
    config = worker.trial_config(
        "vbll", "dtd",
        params={"learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 256,
                "prior_scale": 1.0, "wishart_scale": 0.1},
        budgets=budgets,
    )
    # 2,000 steps at the SAMPLED batch size, not the pilot's: the budget is a
    # step budget, and epochs is only how the scripts express it.
    # dtd is 3,760 rows: 3760 // 256 = 14 steps/epoch, ceil(2000 / 14) = 143.
    assert config["epochs"] == 143
    assert config["batch_size"] == 256
    assert config["parameterization"] == "diagonal"
    assert config["model"] == space.TUNING_MODEL
    assert config["num_classes"] == 47
    assert config["img_size"] == 512


def test_a_missing_epoch_budget_raises(tmp_path):
    with pytest.raises(KeyError, match="dtd"):
        worker.trial_config(
            "adamw_ce", "dtd",
            params={"learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 256},
            budgets={},
        )
```

- [ ] **Step 4: Run to verify it fails**

```bash
apptainer exec --nv --env PYTHONPATH=$PWD/src bllarse-jax.sif \
  python -m pytest tests/tuning/test_worker.py -v
```

Expected: FAIL — `ImportError: cannot import name 'worker'`.

- [ ] **Step 5: Implement the worker**

Create `src/bllarse/tuning/worker.py`:

```python
"""An Optuna TPE worker for one (arm, dataset) study.

Not a run_sweep submission: run_sweep enumerates create_configs() before
submitting, and TPE generates configurations adaptively from the trials already
reported. Four workers share one study through a journal file.

TPE degrades under parallelism -- each worker samples a posterior that has not
yet been updated by the trials in flight beside it -- so the worker count is
deliberately low.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from bllarse.tuning import budgets as budgets_mod
from bllarse.tuning import objective as objective_mod
from bllarse.tuning import space

DIRECTION = "minimize"
N_DUMPS = 40

# Fires no earlier than 25% of the dump grid, and only once 20 trials have
# completed, so early noise cannot kill a config that starts slowly.
PRUNER_STARTUP_TRIALS = 20
PRUNER_WARMUP_DUMPS = N_DUMPS // 4

VBLL_FIXED = {
    "parameterization": "diagonal",
    "vbll_type": "discriminative",
    "device": "cuda",
}


def study_name(arm: str, dataset: str) -> str:
    return f"{arm}_{dataset}"


def open_study(arm: str, dataset: str, storage_dir: Any) -> optuna.Study:
    """Load or create this (arm, dataset)'s study.

    JournalStorage on JournalFileBackend, NOT SQLite: SQLite over GPFS with
    concurrent writers is unreliable, and the journal backend is built for
    network filesystems. One file per study means the 14 studies never contend
    with one another -- only the four workers within a study do.
    """
    directory = Path(storage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    name = study_name(arm, dataset)
    storage = JournalStorage(JournalFileBackend(str(directory / f"{name}.log")))
    return optuna.create_study(
        study_name=name,
        storage=storage,
        direction=DIRECTION,
        sampler=TPESampler(),
        pruner=MedianPruner(
            n_startup_trials=PRUNER_STARTUP_TRIALS,
            n_warmup_steps=PRUNER_WARMUP_DUMPS,
        ),
        load_if_exists=True,
    )


def trial_config(
    arm: str,
    dataset: str,
    *,
    params: Mapping[str, Any],
    budgets: Mapping[str, Mapping[str, int]],
    cache_dir: str = ".cache/features",
    split_cache_dir: str = ".cache/splits",
    enable_mlflow: bool = True,
    mlflow_experiment: str = "bllarse-probing-tuning",
) -> dict[str, Any]:
    """A full training config for one sampled parameter set.

    The Stage 1 budget is a STEP budget; `epochs` is only how the scripts
    express it. Converting at the SAMPLED batch size rather than the pilot's is
    what keeps trials comparable: two configs with different batch sizes get the
    same number of gradient steps, not the same number of passes.
    """
    if dataset not in budgets:
        raise KeyError(
            f"no epoch budget for {dataset!r}. Run the Stage 1 pilot and "
            f"`python -m bllarse.tuning.budgets` first -- tuning under a "
            f"truncated budget biases the search toward high learning rates."
        )
    num_classes, img_size = space.DATASETS[dataset]
    target_steps = budgets[dataset]["steps"]
    batch_size = params["batch_size"]

    config: dict[str, Any] = {
        "model": space.TUNING_MODEL,
        "dataset": dataset,
        "num_classes": num_classes,
        "img_size": img_size,
        "cache_dir": cache_dir,
        "split_cache_dir": split_cache_dir,
        "epochs": budgets_mod.epochs_for_steps(dataset, target_steps, batch_size),
        "n_dumps": N_DUMPS,
        "val_fraction": 0.1,
        "enable_mlflow": enable_mlflow,
        "mlflow_experiment": mlflow_experiment,
        **params,
    }
    if arm == "vbll":
        config.update(VBLL_FIXED)
    else:
        config["arm"] = "adamw_ce"
    return config


def make_objective(
    arm: str,
    dataset: str,
    *,
    budgets: Mapping[str, Mapping[str, int]],
    dump_root: Any,
    group_id: str,
    **config_kw: Any,
) -> Callable[[optuna.Trial], float]:
    """The optuna objective closure for one study."""

    def run_trial(trial: optuna.Trial) -> float:
        params = space.suggest(trial, arm, dataset)
        config = trial_config(
            arm, dataset, params=params, budgets=budgets, **config_kw
        )
        trial.set_user_attr("epochs", config["epochs"])
        trial.set_user_attr("steps", budgets[dataset]["steps"])

        def on_dump(dump_index: int, step: int, scalars: Mapping[str, float]) -> None:
            # Reported on the DUMP INDEX, never the optimiser step. MedianPruner
            # compares trials at matching keys and trials vary batch_size, so
            # raw steps would rarely even collide -- the pruner would silently
            # never have a cohort to compare against.
            trial.report(float(scalars["val_nll"]), dump_index)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return objective_mod.evaluate(
            arm, config,
            dump_root=dump_root,
            group_id=group_id,
            uid=f"trial{trial.number:05d}",
            on_dump=on_dump,
        )

    return run_trial


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=sorted(objective_mod.SCRIPTS))
    p.add_argument("--dataset", required=True, choices=sorted(space.DATASETS))
    p.add_argument("--n-trials", type=int, required=True,
                   help="Trials THIS worker runs. With 4 workers, one quarter "
                        "of the study's total.")
    p.add_argument("--storage-dir", required=True)
    p.add_argument("--dump-root",
                   default=os.environ.get("BLLARSE_EVAL_DUMP_ROOT", "eval_dumps"))
    p.add_argument("--budgets", type=Path,
                   default=Path("bllarse_sweeps/epoch_budgets.json"))
    p.add_argument("--cache-dir", default=".cache/features")
    p.add_argument("--split-cache-dir", default=".cache/splits")
    p.add_argument("--mlflow-experiment", default="bllarse-probing-tuning")
    p.add_argument("--no-mlflow", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    budgets = json.loads(Path(args.budgets).read_text())
    study = open_study(args.arm, args.dataset, args.storage_dir)
    group_id = f"tune_{args.arm}_{args.dataset}"

    study.optimize(
        make_objective(
            args.arm, args.dataset,
            budgets=budgets,
            dump_root=args.dump_root,
            group_id=group_id,
            cache_dir=args.cache_dir,
            split_cache_dir=args.split_cache_dir,
            enable_mlflow=not args.no_mlflow,
            mlflow_experiment=args.mlflow_experiment,
        ),
        n_trials=args.n_trials,
    )
    print(f"[tune] {study.study_name}: {len(study.trials)} trials, "
          f"best={study.best_value:.5f} params={study.best_params}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the worker tests**

```bash
apptainer exec --nv --env PYTHONPATH=$PWD/src bllarse-jax.sif \
  python -m pytest tests/tuning/ -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/bllarse/tuning/worker.py tests/tuning/test_worker.py \
        src/slurm/images/bllarse-jax.def src/slurm/images/bllarse-torch.def
git commit -m "feat(tuning): Optuna TPE worker on journal storage"
```

---

### Task 9: SLURM job script for the tuning workers

`run_sweep` cannot submit this — it derives the array size from
`create_configs()`. The tuning array is instead fixed at four tasks, all running
the same worker command against the same study.

**Files:**
- Create: `src/slurm/jobs/slurm_tune.sh`
- Test: `tests/tuning/test_tune_job_script.py`

**Interfaces:**
- Consumes: `python -m bllarse.tuning.worker` (Task 8).
- Produces: a job script reading `BLLARSE_TUNE_ARM`, `BLLARSE_TUNE_DATASET`,
  `BLLARSE_TUNE_TRIALS`, `BLLARSE_TUNE_STORAGE`, `BLLARSE_SIF`.

- [ ] **Step 1: Write the failing test**

Create `tests/tuning/test_tune_job_script.py`:

```python
from pathlib import Path

SCRIPT = Path("src/slurm/jobs/slurm_tune.sh")


def test_the_job_script_exists_and_is_a_login_shell():
    """On MPCDF no modules are preloaded and the `module` function only exists
    once /etc/profile.d has been sourced, which a non-login shell skips."""
    text = SCRIPT.read_text()
    assert text.startswith("#!/bin/bash -l")


def test_it_requests_a_gpu_partition_explicitly():
    """Raven's default partition is `interactive`, which has no GPUs, so a bare
    `-G 1` is rejected at submit time and the array never queues."""
    text = SCRIPT.read_text()
    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --gres=gpu:a100:1" in text


def test_it_resolves_the_repo_root_through_symlinks():
    """/ptmp -> /raven/ptmp. Apptainer binds the RESOLVED directory and does not
    recreate the symlink inside the container, so an unresolved path exists on
    the host but not in the container and every task dies on import bllarse."""
    assert "readlink -f" in SCRIPT.read_text()


def test_it_binds_the_repo_and_runs_with_nv():
    text = SCRIPT.read_text()
    assert "--bind" in text
    assert "--nv" in text


def test_it_puts_the_vbll_package_on_pythonpath():
    """vbll_experiments is a LOCAL package under scripts/vbll_pytorch/src that
    is never pip-installed into the image."""
    assert "scripts/vbll_pytorch/src" in SCRIPT.read_text()


def test_it_invokes_the_worker_module():
    assert "bllarse.tuning.worker" in SCRIPT.read_text()
```

- [ ] **Step 2: Run to verify it fails**

```bash
apptainer exec --nv --env PYTHONPATH=$PWD/src bllarse-jax.sif \
  python -m pytest tests/tuning/test_tune_job_script.py -v
```

Expected: FAIL — `FileNotFoundError: src/slurm/jobs/slurm_tune.sh`.

- [ ] **Step 3: Write the job script**

Create `src/slurm/jobs/slurm_tune.sh`:

```bash
#!/bin/bash -l
# Four Optuna workers for ONE (arm, dataset) study. A LOGIN shell (-l), not
# `env bash`: on MPCDF no modules are preloaded and the `module` function only
# exists once /etc/profile.d has been sourced, which a non-login shell skips.
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-3
#SBATCH --output=slurm_logs/%x.%A_%a.out
#SBATCH --error=slurm_logs/%x.%A_%a.err
# The array is fixed at 4 and NOT derived from a config count -- that is the
# whole reason this is not a run_sweep submission. TPE degrades under
# parallelism (each worker samples a posterior not yet updated by the trials in
# flight beside it), so four is a deliberate ceiling, not a resource limit.

set -euo pipefail

if ! command -v apptainer >/dev/null 2>&1; then
  if [[ $(type -t module || true) == function ]]; then
    module purge >/dev/null 2>&1 || true
    module load "${BLLARSE_APPTAINER_MODULE:-apptainer}" >/dev/null 2>&1 || true
  fi
fi
if ! command -v apptainer >/dev/null 2>&1; then
  echo "[bllarse] ERROR: apptainer not found on PATH and no module supplied it." >&2
  exit 2
fi

for var in BLLARSE_SIF BLLARSE_TUNE_ARM BLLARSE_TUNE_DATASET \
           BLLARSE_TUNE_TRIALS BLLARSE_TUNE_STORAGE; do
  if [[ -z "${!var:-}" ]]; then
    echo "[bllarse] ERROR: $var is not set." >&2
    exit 2
  fi
done

# readlink -f, not the raw path: on MPCDF /ptmp and /u are top-level symlinks
# (/ptmp -> /raven/ptmp). Apptainer binds the RESOLVED directory and does NOT
# recreate that symlink inside the container, so an unresolved path exists on
# the host but not in the container and `import bllarse` fails on every task.
repo_root="$(readlink -f "${SLURM_SUBMIT_DIR:-$(pwd)}")"
cd "$repo_root"
storage_dir="$(readlink -f "$BLLARSE_TUNE_STORAGE")"
mkdir -p "$storage_dir"

echo "[bllarse] tune worker=${SLURM_ARRAY_TASK_ID} arm=${BLLARSE_TUNE_ARM}" \
     "dataset=${BLLARSE_TUNE_DATASET} trials=${BLLARSE_TUNE_TRIALS}"

bind_args=(--bind "$repo_root" --bind "$storage_dir")
if [[ -n "${BLLARSE_HF_HOME:-}" ]]; then
  mkdir -p "$BLLARSE_HF_HOME"
  bind_args+=(--bind "$BLLARSE_HF_HOME:$BLLARSE_HF_HOME")
  export HF_HOME="$BLLARSE_HF_HOME"
fi
if [[ -n "${BLLARSE_EVAL_DUMP_ROOT:-}" ]]; then
  mkdir -p "$BLLARSE_EVAL_DUMP_ROOT"
  bind_args+=(--bind "$BLLARSE_EVAL_DUMP_ROOT:$BLLARSE_EVAL_DUMP_ROOT")
fi

env_args=(
  --env "BLLARSE_REPO_ROOT=$repo_root"
  --env "PYTHONPATH=${repo_root}/src:${repo_root}/scripts/vbll_pytorch/src${PYTHONPATH:+:${PYTHONPATH}}"
)
[[ -n "${HF_TOKEN:-}" ]] && env_args+=(--env "HF_TOKEN=$HF_TOKEN")
[[ -n "${HF_HOME:-}" ]] && env_args+=(--env "HF_HOME=$HF_HOME")
[[ -n "${BLLARSE_EVAL_DUMP_ROOT:-}" ]] && env_args+=(--env "BLLARSE_EVAL_DUMP_ROOT=$BLLARSE_EVAL_DUMP_ROOT")
[[ -n "${MLFLOW_TRACKING_URI:-}" ]] && env_args+=(--env "MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI")
[[ -n "${MLFLOW_TRACKING_USERNAME:-}" ]] && env_args+=(--env "MLFLOW_TRACKING_USERNAME=$MLFLOW_TRACKING_USERNAME")
[[ -n "${MLFLOW_TRACKING_PASSWORD:-}" ]] && env_args+=(--env "MLFLOW_TRACKING_PASSWORD=$MLFLOW_TRACKING_PASSWORD")
[[ -n "${MLFLOW_PARENT_RUN_ID:-}" ]] && env_args+=(--env "MLFLOW_PARENT_RUN_ID=$MLFLOW_PARENT_RUN_ID")

exec apptainer exec --nv \
  "${env_args[@]}" \
  "${bind_args[@]}" \
  "$BLLARSE_SIF" \
  python -m bllarse.tuning.worker \
    --arm "$BLLARSE_TUNE_ARM" \
    --dataset "$BLLARSE_TUNE_DATASET" \
    --n-trials "$BLLARSE_TUNE_TRIALS" \
    --storage-dir "$storage_dir" \
    --cache-dir "${BLLARSE_CACHE_DIR:-.cache/features}"
```

- [ ] **Step 4: Make it executable and run the tests**

```bash
chmod +x src/slurm/jobs/slurm_tune.sh
apptainer exec --nv --env PYTHONPATH=$PWD/src bllarse-jax.sif \
  python -m pytest tests/tuning/test_tune_job_script.py -v
```

Expected: PASS.

- [ ] **Step 5: Validate the submission without running it**

```bash
BLLARSE_SIF=$PWD/bllarse-jax.sif \
BLLARSE_TUNE_ARM=adamw_ce BLLARSE_TUNE_DATASET=dtd \
BLLARSE_TUNE_TRIALS=16 BLLARSE_TUNE_STORAGE=/ptmp/cheins/optuna \
sbatch --test-only --export=ALL src/slurm/jobs/slurm_tune.sh
```

Expected: `sbatch: Job N to start at ...` — no `Requested node configuration is
not available`.

- [ ] **Step 6: Commit**

```bash
git add src/slurm/jobs/slurm_tune.sh tests/tuning/test_tune_job_script.py
git commit -m "feat(slurm): tuning worker array job"
```

- [ ] **Step 7: CLUSTER — run the campaign**

| Arm | Free params | Trials/study | Per worker (of 4) | Studies |
|---|---:|---:|---:|---:|
| adamw_ce | 3 (lr, wd, bs) | 64 | 16 | 7 |
| vbll | 5 (+prior, wishart) | 96 | 24 | 7 |

```bash
export BLLARSE_TUNE_STORAGE=/ptmp/cheins/optuna
for ds in oxford_pets dtd flowers102 stanford_cars food101 cifar10 cifar100; do
  BLLARSE_SIF=$PWD/bllarse-jax.sif BLLARSE_TUNE_ARM=adamw_ce \
  BLLARSE_TUNE_DATASET=$ds BLLARSE_TUNE_TRIALS=16 \
    sbatch --export=ALL --job-name=tune_ce_$ds src/slurm/jobs/slurm_tune.sh
  BLLARSE_SIF=$PWD/bllarse-torch.sif BLLARSE_TUNE_ARM=vbll \
  BLLARSE_TUNE_DATASET=$ds BLLARSE_TUNE_TRIALS=24 \
    sbatch --export=ALL --job-name=tune_vbll_$ds src/slurm/jobs/slurm_tune.sh
done
```

---

### Task 10: Stage 3 — export the tuned lookup

Tuned values are read from a committed file, never hand-copied into a sweep's
`BASE`. That keeps submission reproducible without a live MLflow or Optuna
dependency and makes every number's provenance auditable.

**Files:**
- Create: `src/bllarse/tuning/export.py`
- Modify: `src/bllarse/tuning/__init__.py`
- Test: `tests/tuning/test_export.py`

**Interfaces:**
- Consumes: `worker.open_study` (Task 8).
- Produces:
  - `export.collect(storage_dir, budgets) -> dict[str, dict[str, dict]]`
  - `tuning.load_tuned(path=None) -> dict`
  - `tuning.tuned_config(arm, dataset, path=None) -> dict[str, Any]`
  - `bllarse_sweeps/tuned_hparams.json`

- [ ] **Step 1: Write the failing test**

Create `tests/tuning/test_export.py`:

```python
import json

import pytest

optuna = pytest.importorskip("optuna")

from bllarse import tuning
from bllarse.tuning import export, worker


def _seed_study(storage_dir, arm, dataset, values):
    study = worker.open_study(arm, dataset, storage_dir)
    for value, lr in values:
        trial = study.ask({
            "learning_rate": optuna.distributions.FloatDistribution(
                1e-4, 3e-2, log=True
            ),
        })
        study.tell(trial, value)
    return study


def test_export_takes_the_best_trial_per_study(tmp_path):
    _seed_study(tmp_path, "adamw_ce", "dtd", [(0.9, 1e-3), (0.4, 5e-3), (0.7, 1e-2)])
    budgets = {"dtd": {"batch_size": 512, "steps": 2000, "epochs": 273}}

    out = export.collect(tmp_path, budgets)
    assert set(out) == {"adamw_ce"}
    assert set(out["adamw_ce"]) == {"dtd"}
    assert out["adamw_ce"]["dtd"]["value"] == pytest.approx(0.4)
    assert out["adamw_ce"]["dtd"]["steps"] == 2000


def test_tuned_config_round_trips_through_the_json(tmp_path):
    path = tmp_path / "tuned.json"
    path.write_text(json.dumps({
        "adamw_ce": {"dtd": {
            "params": {"learning_rate": 5e-3, "weight_decay": 1e-4,
                       "batch_size": 256},
            "steps": 2000, "epochs": 143, "value": 0.4,
        }},
    }))

    config = tuning.tuned_config("adamw_ce", "dtd", path=path)
    assert config == {
        "learning_rate": 5e-3, "weight_decay": 1e-4,
        "batch_size": 256, "epochs": 143,
    }


def test_a_missing_lookup_raises_rather_than_falling_back_to_defaults(tmp_path):
    """Silently falling back to the argparse defaults is exactly the failure
    this campaign exists to remove -- it would produce a full 200-config figure
    sweep at lr=1e-3 that looks completely normal."""
    with pytest.raises(FileNotFoundError, match="tuned_hparams"):
        tuning.tuned_config("adamw_ce", "dtd", path=tmp_path / "absent.json")


def test_an_untuned_dataset_raises(tmp_path):
    path = tmp_path / "tuned.json"
    path.write_text(json.dumps({"adamw_ce": {"dtd": {
        "params": {"learning_rate": 5e-3, "weight_decay": 1e-4,
                   "batch_size": 256},
        "steps": 2000, "epochs": 143, "value": 0.4,
    }}}))
    with pytest.raises(KeyError, match="food101"):
        tuning.tuned_config("adamw_ce", "food101", path=path)
```

- [ ] **Step 2: Run to verify it fails**

```bash
apptainer exec --nv --env PYTHONPATH=$PWD/src bllarse-jax.sif \
  python -m pytest tests/tuning/test_export.py -v
```

Expected: FAIL — `ImportError: cannot import name 'export'`.

- [ ] **Step 3: Implement the export module**

Create `src/bllarse/tuning/export.py`:

```python
"""Stage 3: read the study journals, emit bllarse_sweeps/tuned_hparams.json.

The sweeps read this file. Tuned values are never hand-copied into a sweep's
BASE -- a copy is a second source of truth that drifts silently, and the whole
point of the campaign is that every reported number's hyperparameters are
auditable back to the trial that chose them.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from bllarse.tuning import budgets as budgets_mod
from bllarse.tuning import objective as objective_mod
from bllarse.tuning import space, worker

DEFAULT_OUTPUT = Path("bllarse_sweeps/tuned_hparams.json")
DEFAULT_BUDGETS = Path("bllarse_sweeps/epoch_budgets.json")


def collect(
    storage_dir: Any, budgets: Mapping[str, Mapping[str, int]]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Best trial per study, as {arm: {dataset: record}}.

    Studies with no completed trial are skipped rather than defaulted -- an
    empty study means that (arm, dataset) was never tuned, and emitting a
    plausible entry for it would hide that.
    """
    directory = Path(storage_dir)
    out: dict[str, dict[str, dict[str, Any]]] = {}

    for arm in sorted(objective_mod.SCRIPTS):
        for dataset in sorted(space.DATASETS):
            journal = directory / f"{worker.study_name(arm, dataset)}.log"
            if not journal.is_file():
                continue
            study = worker.open_study(arm, dataset, directory)
            completed = [t for t in study.trials if t.value is not None]
            if not completed:
                print(f"[export] SKIP {arm}/{dataset}: no completed trial")
                continue

            best = min(completed, key=lambda t: t.value)
            target_steps = budgets[dataset]["steps"]
            out.setdefault(arm, {})[dataset] = {
                "params": dict(best.params),
                "steps": target_steps,
                "epochs": budgets_mod.epochs_for_steps(
                    dataset, target_steps, best.params["batch_size"]
                ),
                "value": float(best.value),
                "trial": best.number,
                "n_trials": len(study.trials),
            }
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--storage-dir", required=True)
    p.add_argument("--budgets", type=Path, default=DEFAULT_BUDGETS)
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    budgets = json.loads(Path(args.budgets).read_text())
    tuned = collect(args.storage_dir, budgets)
    args.output.write_text(json.dumps(tuned, indent=2, sort_keys=True) + "\n")

    for arm, by_dataset in sorted(tuned.items()):
        for dataset, record in sorted(by_dataset.items()):
            print(f"[export] {arm:9s} {dataset:15s} val_nll={record['value']:.5f} "
                  f"({record['n_trials']} trials) {record['params']}")
    print(f"[export] wrote {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement the loader**

Replace `src/bllarse/tuning/__init__.py`:

```python
"""Optuna TPE tuning for the probing arms.

Separate from bllarse.tools.run_sweep by necessity: run_sweep's contract is
`create_configs() -> list[dict]`, enumerated before submission, and TPE
generates configurations adaptively from the trials already reported.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from bllarse.tuning.objective import TUNING_SEEDS

DEFAULT_LOOKUP = Path(__file__).resolve().parents[3] / "bllarse_sweeps" / "tuned_hparams.json"

__all__ = ["TUNING_SEEDS", "load_tuned", "tuned_config"]


@lru_cache(maxsize=4)
def load_tuned(path: Any = None) -> dict[str, Any]:
    """The committed {arm: {dataset: record}} lookup.

    Raises when absent. There is deliberately no fallback to the argparse
    defaults: a silent fallback is exactly the failure this campaign exists to
    remove -- it would submit a full 200-config figure sweep at lr=1e-3 that
    looks entirely normal and is simply untuned.
    """
    lookup = Path(path) if path is not None else DEFAULT_LOOKUP
    if not lookup.is_file():
        raise FileNotFoundError(
            f"tuned_hparams lookup not found at {lookup}. Run the Stage 2 "
            f"campaign and `python -m bllarse.tuning.export` first; the sweeps "
            f"deliberately do not fall back to untuned defaults."
        )
    return json.loads(lookup.read_text())


def tuned_config(arm: str, dataset: str, path: Any = None) -> dict[str, Any]:
    """The training-config fragment for one (arm, dataset): params + epochs."""
    tuned = load_tuned(path)
    if arm not in tuned or dataset not in tuned[arm]:
        raise KeyError(
            f"no tuned hyperparameters for arm={arm!r} dataset={dataset!r}"
        )
    record = tuned[arm][dataset]
    return {**record["params"], "epochs": record["epochs"]}
```

`lru_cache` on a `Path` argument requires hashable input — the tests pass
`Path` objects, which are hashable. Callers passing `None` hit the same cache
entry.

- [ ] **Step 5: Run to verify it passes**

```bash
apptainer exec --nv --env PYTHONPATH=$PWD/src bllarse-jax.sif \
  python -m pytest tests/tuning/ -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/bllarse/tuning/export.py src/bllarse/tuning/__init__.py \
        tests/tuning/test_export.py
git commit -m "feat(tuning): export the tuned hyperparameter lookup"
```

- [ ] **Step 7: CLUSTER — produce and commit the lookup**

Only after the Stage 2 campaign has finished.

```bash
python -m bllarse.tuning.export \
  --storage-dir /ptmp/cheins/optuna \
  --budgets bllarse_sweeps/epoch_budgets.json \
  -o bllarse_sweeps/tuned_hparams.json
```

Check the printed table for any `SKIP` line before committing — a skipped study
means that `(arm, dataset)` was never tuned, and Task 11's sweeps will raise on
it rather than run untuned.

```bash
git add bllarse_sweeps/tuned_hparams.json
git commit -m "feat(tuning): tuned hyperparameters from the Stage 2 campaign"
```

---

### Task 11: Stage 4 — figure sweeps read the lookup

**Blocked on Task 10 Step 7**: the sweeps raise without
`bllarse_sweeps/tuned_hparams.json`, and `tests/test_sweep_definitions.py` calls
`create_configs()` on all of them.

**Files:**
- Modify: `bllarse_sweeps/probe_adamw_ce.py`
- Modify: `bllarse_sweeps/probe_vbll.py`
- Modify: `bllarse_sweeps/probe_ood_adamw_ce.py`
- Modify: `bllarse_sweeps/probe_ood_vbll.py`
- Test: `tests/test_sweep_definitions.py`

**Interfaces:**
- Consumes: `tuning.tuned_config(arm, dataset)` (Task 10).
- Produces: every figure config carries `learning_rate`, `weight_decay`,
  `batch_size`, `epochs` from the lookup, and VBLL additionally carries
  `prior_scale` and `wishart_scale`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sweep_definitions.py`:

```python
TUNED_KEYS = {"learning_rate", "weight_decay", "batch_size", "epochs"}

# Which arm's tuned entry each sweep must resolve. An explicit mapping rather
# than a substring test on the path: Task 12's probe_rank_pareto.py is a VBLL
# sweep whose filename contains no "vbll", and a substring test would silently
# check it against the CE lookup.
ARM_BY_SWEEP = {
    "bllarse_sweeps/probe_adamw_ce.py": "adamw_ce",
    "bllarse_sweeps/probe_vbll.py": "vbll",
    "bllarse_sweeps/probe_ood_adamw_ce.py": "adamw_ce",
    "bllarse_sweeps/probe_ood_vbll.py": "vbll",
}

FIGURE_SWEEPS = sorted(ARM_BY_SWEEP)


@pytest.mark.parametrize("path", FIGURE_SWEEPS)
def test_every_figure_config_carries_tuned_hyperparameters(path):
    """The sweeps used to restate probe.py's argparse defaults, which were
    themselves build_vbll_head's defaults -- nobody had ever chosen them. Every
    hyperparameter now traces to the trial that selected it."""
    from bllarse import tuning

    for c in get_module_from_source_path(path).create_configs():
        assert TUNED_KEYS <= set(c), f"{c['uid']}: missing {TUNED_KEYS - set(c)}"
        expected = tuning.tuned_config(ARM_BY_SWEEP[path], c["dataset"])
        for key, value in expected.items():
            assert c[key] == value, f"{c['uid']}: {key} is not the tuned value"


@pytest.mark.parametrize("path", [SWEEPS[1], SWEEPS[3]])
def test_vbll_sweeps_stay_diagonal(path):
    """cov_rank is a cost/quality dial the other arms do not have. Letting it
    into a figure would let VBLL win partly by being larger -- at stanford_cars
    rank-64 is 16.6M parameters against CE's 251K."""
    for c in get_module_from_source_path(path).create_configs():
        assert c["parameterization"] == "diagonal"
        assert "cov_rank" not in c
```

- [ ] **Step 2: Run to verify it fails**

```bash
apptainer exec --nv --env PYTHONPATH=$PWD/src bllarse-jax.sif \
  python -m pytest tests/test_sweep_definitions.py -k tuned -v
```

Expected: FAIL — the configs still carry the hardcoded `learning_rate=1e-3`.

- [ ] **Step 3: Rewrite the AdamW Fig 1 sweep's config builder**

In `bllarse_sweeps/probe_adamw_ce.py`, delete `epochs`, `batch_size`,
`learning_rate` and `weight_decay` from `BASE`, leaving:

```python
BASE = dict(
    enable_mlflow=True,
    mlflow_experiment="bllarse-probing",
    arm="adamw_ce",
    n_dumps=40,
    val_fraction=0.1,
)
```

Add the import and merge the lookup into every config:

```python
from bllarse import tuning
```

```python
def create_configs():
    configs = []
    for model, (dataset, (num_classes, img_size)), seed in product(
        MODELS, DATASETS.items(), SEEDS
    ):
        configs.append(dict(
            **BASE,
            # Per dataset, not sweep-wide: n_train spans 3,680 to 75,750, so a
            # shared learning rate and batch size cannot be right for both ends.
            **tuning.tuned_config("adamw_ce", dataset),
            model=model,
            dataset=dataset,
            num_classes=num_classes,
            img_size=img_size,
            seed=seed,
            group_id=GROUP_ID,
            uid=f"{model}_{dataset}_seed{seed}",
        ))
    return configs
```

- [ ] **Step 4: Apply the same change to the VBLL Fig 1 sweep**

In `bllarse_sweeps/probe_vbll.py`, reduce `BASE` to:

```python
BASE = dict(
    enable_mlflow=True,
    mlflow_experiment="bllarse-probing",
    n_dumps=40,
    val_fraction=0.1,
    # Fixed, never tuned: rank buys calibration with memory and neither AdamW+CE
    # nor IBProbit has an equivalent knob, so letting it vary would let VBLL win
    # Fig 1 partly by being larger. The rank/cost trade-off is an appendix
    # Pareto front (bllarse_sweeps/probe_rank_pareto.py), which is where it
    # belongs.
    parameterization="diagonal",
    vbll_type="discriminative",
    device="cuda",
)
```

`prior_scale` and `wishart_scale` are removed from `BASE` — they come from the
lookup. Add `from bllarse import tuning` and merge
`**tuning.tuned_config("vbll", dataset),` into the config exactly as in Step 3.

- [ ] **Step 5: Apply to both Fig 2 OOD sweeps**

In `bllarse_sweeps/probe_ood_adamw_ce.py` and
`bllarse_sweeps/probe_ood_vbll.py`, make the same edits, keying the lookup on
the **ID** dataset:

```python
            **tuning.tuned_config("vbll", dataset),   # or "adamw_ce"
```

Add this comment above that line in both files:

```python
            # Keyed on the ID dataset only. Both scripts load OOD features
            # after the training loop, purely for a final dump -- the OOD split
            # never influences training, so cifar10-vs-svhn and
            # cifar10-vs-cifar100 correctly share one tuned config.
```

- [ ] **Step 6: Run the full sweep suite**

```bash
apptainer exec --nv --env PYTHONPATH=$PWD/src bllarse-jax.sif \
  python -m pytest tests/test_sweep_definitions.py -v
```

Expected: PASS — 200 configs for each Fig 1 sweep, 120 for each Fig 2 sweep,
all carrying tuned values.

- [ ] **Step 7: Commit**

```bash
git add bllarse_sweeps/probe_adamw_ce.py bllarse_sweeps/probe_vbll.py \
        bllarse_sweeps/probe_ood_adamw_ce.py bllarse_sweeps/probe_ood_vbll.py \
        tests/test_sweep_definitions.py
git commit -m "feat(sweeps): resolve hyperparameters from the tuned lookup"
```

---

### Task 12: Stage 5 — the rank/cost Pareto front

Takes each dataset's tuned diagonal config and re-runs it at
`parameterization="lowrank"` with `cov_rank` in `{2, 4, 8, 16, 32, 64}`,
everything else held. Reports calibration against **head parameter count**, so
the cost of any calibration gain is visible rather than asserted.

**Files:**
- Create: `bllarse_sweeps/probe_rank_pareto.py`
- Test: `tests/test_sweep_definitions.py`

**Interfaces:**
- Consumes: `tuning.tuned_config` (Task 10); `--cov-rank` (Task 1);
  `head_params` logging (Task 2).
- Produces: 3 datasets x 6 ranks x 3 seeds = 54 configs, VBLL only.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sweep_definitions.py`:

```python
def test_the_rank_pareto_sweep_spans_the_cost_inversion():
    """cifar10 (C=10), food101 (C=101), stanford_cars (C=196) span the regime
    where the cost story inverts: VBLL costs D*C*(2+r) and IBProbit costs
    D*(D-1)/2 independent of C, so at C=10 rank-64 is IBProbit-sized and at
    C=196 it is 20x IBProbit."""
    mod = get_module_from_source_path("bllarse_sweeps/probe_rank_pareto.py")
    configs = mod.create_configs()

    assert len(configs) == 54  # 3 datasets x 6 ranks x 3 seeds
    assert {c["dataset"] for c in configs} == {"cifar10", "food101", "stanford_cars"}
    assert {c["cov_rank"] for c in configs} == {2, 4, 8, 16, 32, 64}
    assert all(c["parameterization"] == "lowrank" for c in configs)


def test_the_pareto_sweep_uses_reported_seeds_not_tuning_seeds():
    """The appendix carries error bars comparable to the figures, so its seeds
    come from the reported range -- unlike the tuning campaign, which must not."""
    from bllarse import tuning

    mod = get_module_from_source_path("bllarse_sweeps/probe_rank_pareto.py")
    seeds = {c["seed"] for c in mod.create_configs()}
    assert seeds <= set(range(137, 147))
    assert seeds.isdisjoint(tuning.TUNING_SEEDS)


def test_the_pareto_sweep_holds_every_other_parameter_at_the_tuned_value():
    """prior_scale and wishart_scale are NOT re-tuned per rank -- the front is a
    lower bound on what each rank could achieve, not a tuned comparison, and the
    appendix says so."""
    from bllarse import tuning

    mod = get_module_from_source_path("bllarse_sweeps/probe_rank_pareto.py")
    for c in mod.create_configs():
        tuned = tuning.tuned_config("vbll", c["dataset"])
        for key, value in tuned.items():
            assert c[key] == value, f"{c['uid']}: {key} drifted from the tuned value"
```

- [ ] **Step 2: Run to verify it fails**

```bash
apptainer exec --nv --env PYTHONPATH=$PWD/src bllarse-jax.sif \
  python -m pytest tests/test_sweep_definitions.py -k pareto -v
```

Expected: FAIL — `FileNotFoundError: bllarse_sweeps/probe_rank_pareto.py`.

- [ ] **Step 3: Write the sweep**

Create `bllarse_sweeps/probe_rank_pareto.py`:

```python
"""Appendix: the VBLL rank/cost Pareto front.

cov_rank buys calibration with memory, and neither AdamW+CE nor IBProbit has an
equivalent knob -- so it is fixed at diagonal in every figure and varied only
here, plotted against head parameter count with CE's and IBProbit's counts
marked on the same axis.

Head parameters at D=1280:

    | dataset       |   C | CE linear | VBLL diag | VBLL r=2 | VBLL r=64 | IBProbit |
    |---------------|----:|----------:|----------:|---------:|----------:|---------:|
    | cifar10       |  10 |     12.8K |     25.6K |    51.2K |      845K |     819K |
    | food101       | 101 |      129K |      259K |     517K |      8.5M |     819K |
    | stanford_cars | 196 |      251K |      502K |    1.00M |     16.6M |     819K |

IBProbit costs D(D-1)/2, INDEPENDENT of C; VBLL costs D*C*(2+r), linear in C.
So the rank matching IBProbit's budget is ~64 at cifar10 and ~1 at
stanford_cars: there is no single fair rank across one figure's panels, which
is exactly why this is an appendix and not a figure.

Two caveats belong in the appendix text:

  - prior_scale and wishart_scale are NOT re-tuned per rank, though changing
    rank changes posterior capacity and so plausibly moves their optimum. The
    front is therefore a lower bound on what each rank could achieve, not a
    tuned comparison. Re-tuning per rank would multiply the cost by the number
    of ranks and is not justified for an appendix.
  - The cost axis is the logged `head_params`, read off each run, not a count
    recomputed by the plotting script.
"""

from itertools import product

from bllarse import tuning
from bllarse.tools.adapters import run_script_from_config
from bllarse.tuning import space

GROUP_ID = "probe_rank_pareto"

# Spans the regime where the cost story inverts -- see the table above.
DATASETS = ["cifar10", "food101", "stanford_cars"]
COV_RANKS = [2, 4, 8, 16, 32, 64]
# From the REPORTED range, so the appendix carries error bars comparable to the
# figures. The tuning campaign's seeds (200, 201) must never appear here.
SEEDS = [137, 138, 139]

BASE = dict(
    enable_mlflow=True,
    mlflow_experiment="bllarse-probing",
    n_dumps=40,
    val_fraction=0.1,
    parameterization="lowrank",
    vbll_type="discriminative",
    device="cuda",
    model=space.TUNING_MODEL,
)


def create_configs():
    configs = []
    for dataset, cov_rank, seed in product(DATASETS, COV_RANKS, SEEDS):
        num_classes, img_size = space.DATASETS[dataset]
        configs.append(dict(
            **BASE,
            **tuning.tuned_config("vbll", dataset),
            dataset=dataset,
            num_classes=num_classes,
            img_size=img_size,
            cov_rank=cov_rank,
            seed=seed,
            group_id=GROUP_ID,
            uid=f"{dataset}_rank{cov_rank}_seed{seed}",
        ))
    return configs


def run(config):
    run_script_from_config("scripts/vbll_pytorch/probing_vbll.py", config)
```

- [ ] **Step 4: Add it to the sweep contract tests**

In `tests/test_sweep_definitions.py`, append
`"bllarse_sweeps/probe_rank_pareto.py"` to the `SWEEPS` list (append, never
insert — `SWEEPS[0]`..`SWEEPS[3]` are indexed positionally), so it inherits the
uniqueness, single-group-id and well-formedness checks.

Add it to `ARM_BY_SWEEP` too, so
`test_every_figure_config_carries_tuned_hyperparameters` covers it:

```python
    "bllarse_sweeps/probe_rank_pareto.py": "vbll",
```

Leave `test_vbll_sweeps_stay_diagonal` parameterised on `[SWEEPS[1], SWEEPS[3]]`
as it is — the Pareto sweep is the one sweep that is deliberately lowrank, and
adding it there would assert the opposite of its purpose.

- [ ] **Step 5: Run the full suite in both images**

```bash
apptainer exec --nv --env PYTHONPATH=$PWD/src bllarse-jax.sif \
  python -m pytest tests/ --ignore=tests/test_probing_vbll_smoke.py -v
apptainer exec --nv \
  --env PYTHONPATH=$PWD/src:$PWD/scripts/vbll_pytorch/src bllarse-torch.sif \
  python -m pytest tests/test_probing_vbll_smoke.py tests/tuning/ -v
```

Expected: PASS in both.

- [ ] **Step 6: Commit**

```bash
git add bllarse_sweeps/probe_rank_pareto.py tests/test_sweep_definitions.py
git commit -m "feat(sweeps): appendix rank/cost Pareto front"
```

- [ ] **Step 7: CLUSTER — run it**

54 runs, VBLL only. Negligible next to Stage 2.

```bash
BLLARSE_SIF=$PWD/bllarse-torch.sif python -m bllarse.tools.run_sweep \
  bllarse_sweeps/probe_rank_pareto.py \
  --partition gpu --gres gpu:a100:1 --max-concurrent 7 \
  --parent-run-name probe_rank_pareto
```

---

## Deviations from the spec

Two, both deliberate, both worth confirming:

1. **`study.optimize`, not Optuna's literal ask-and-tell API.** The spec says
   "a separate ask-and-tell worker path is required" in a paragraph whose
   argument is that TPE generates configurations adaptively and therefore
   cannot go through `create_configs()`. `study.optimize` against shared
   `JournalStorage` is the idiomatic way to run N adaptive workers and satisfies
   that argument; `ask`/`tell` would add a hand-rolled loop with no benefit. The
   tests still use `ask`/`tell` directly, because that is the clean way to seed
   a study with known values.

2. **Tuning runs log to a separate MLflow experiment** (`bllarse-probing-tuning`),
   which the spec does not mention either way. ~2,240 trial runs in
   `bllarse-probing` would bury the 640 figure runs the paper actually reports.
   Set `--no-mlflow` on the worker to turn tracking off entirely.

## Cost

| Stage | Runs | Notes |
|---|---:|---|
| 1 pilot | 28 | 2 LRs x 7 datasets x 2 arms, 1 seed, ~4,000 steps |
| 2 tuning | 1,120 trials | 448 CE + 672 VBLL, x2 seeds, minus pruned |
| 4 figures | 640 | 2 x 200 (Fig 1) + 2 x 120 (Fig 2) |
| 5 appendix | 54 | VBLL only |

At the post-jit rate both arms are at parity (~102 s for 5,880 steps on
food101/dinov3_huge), so the CE arm no longer dominates the critical path.
Pruning removes most of the second seed for hopeless configs, so the campaign's
realised cost is well below 2,240 full runs.
