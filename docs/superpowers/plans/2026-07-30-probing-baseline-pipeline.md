# Probing Baseline Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make AdamW+CE and VBLL linear-probing baselines runnable on the SLURM/Apptainer cluster under a frozen measurement protocol, so Figs 1–4 of `results-plan.md` can be built post-hoc from dumped logits without re-running anything.

**Architecture:** A small pure-library core (`bllarse.metrics`, `bllarse.detection`, `bllarse.splits`, `bllarse.evaldump`, `bllarse.analysis.thresholds`) that knows nothing about training, plus one probing entry point per framework (JAX for CE/IBProbit, PyTorch for VBLL) that loads cached DINOv3 features, trains a head, and writes logit dumps on a log-spaced schedule. Every figure metric — Brier, ECE at any binning, temperature scaling, threshold crossings, OOD detection — is computed from those dumps afterwards, never in the training loop.

**Tech Stack:** JAX/Equinox/Optax, PyTorch (VBLL arm only), TFP-JAX (ECE), NumPy, MLflow, SLURM array jobs, Apptainer.

**Spec:** `docs/superpowers/specs/2026-07-30-results-protocol-and-baseline-infra-design.md` (Phase 0, less 0d-ii)

## Global Constraints

- Python `>=3.11`. Environment is `uv`; run commands as `uv run ...` (see `AGENTS.md`).
- ECE is **20 bins** everywhere, via `tensorflow_probability.substrates.jax.stats.expected_calibration_error`, for continuity with the submitted paper.
- Brier is the **full predictive vector** multiclass form `BS = (1/N) Σᵢ Σₖ (p_ik − y_ik)²`, never the top-label variant.
- OOD score convention is **higher = more OOD**, uniformly, for every score function.
- Dumps are `float16` on disk; every metric upcasts to `float32` before computing.
  The risk is **precision collapse, not overflow**: `jax.nn.softmax` returns fp16 for
  fp16 input, and MSP over 5000 points then collapses from ~4989 distinct values to
  ~381. AUROC gives ties half credit, so Fig 2 would be measurably wrong.
- Seeds for probing are **10**. This deviates from `results-plan.md` §0.5 (which says 25) and is recorded in the spec's sync-back table.
- Backbone ladder is exactly `dinov3_small`, `dinov3_big`, `dinov3_large`, `dinov3_huge`. `dinov3_max` is out of scope.
- Commit after every task. Branch is `infra/results-protocol`.
- `origin` is the private Forgejo remote and takes all branches. Never push feature branches to `github`.

---

## File Structure

| file | responsibility |
|---|---|
| `src/bllarse/metrics.py` | classification metrics + temperature scaling, on `(logits, labels)`. JAX. |
| `src/bllarse/detection.py` | OOD detection metrics on `(scores_id, scores_ood)`. NumPy — post-hoc only. |
| `src/bllarse/splits.py` | seed-independent pinned val/test index files |
| `src/bllarse/evaldump.py` | dump writer/reader + the log-spaced step schedule |
| `src/bllarse/features.py` | cached-feature loader; framework-free, both arms import it |
| `src/bllarse/analysis/thresholds.py` | the §0.4 joint threshold protocol |
| `src/bllarse/analysis/curves.py` | dumps → metric curves; where per-point TS is refit |
| `scripts/alternatives/probe.py` | JAX probing entry point (CE now, IBProbit arm ready for Phase 3) |
| `scripts/vbll_pytorch/probing_vbll.py` | PyTorch probing entry point (VBLL) |
| `src/slurm/jobs/slurm_run_config_apptainer.sh` | Apptainer SLURM array job script |
| `bllarse_sweeps/probe_*.py` | sweep definitions |

---

## Task 1: Branch consolidation

Bring the three trees into one. No new behaviour — this task succeeds when the existing test suite still passes with the extra code present.

**Files:**
- Modify: `src/bllarse/tools/adapters.py`
- Modify: `.gitignore`
- Create (by extraction): `scripts/alternatives/*.py`, `scripts/vbll_pytorch/**`

- [ ] **Step 1: Confirm you are on the right branch**

```bash
cd /home/dima/projects/bllarse
git branch --show-current   # must print: infra/results-protocol
git status --short          # must be empty
```

- [ ] **Step 2: Extract the alternatives code from `ibprobit-process-noise`**

Take the code only. That branch also carries ~5900 `scripts/results/**/*.parquet` files and a 53k-line CSV which are run outputs, not source.

```bash
git checkout ibprobit-process-noise -- scripts/alternatives/
git checkout ibprobit-process-noise -- tests/test_full_finetune_ood.py \
    tests/test_full_finetune_bayes_glue.py tests/test_bayes_optimizer_head_lr.py \
    tests/test_run_training_estimators.py
git checkout ibprobit-process-noise -- docs/superpowers/specs/
git status --short | head -30
```

Verify no `scripts/results/` path appears in the output. If it does, `git rm -r --cached scripts/results` and add it to `.gitignore`.

- [ ] **Step 3: Extract the VBLL subproject**

```bash
git checkout origin/experimental_infra_vbll_baseline -- scripts/vbll_pytorch/
ls scripts/vbll_pytorch/
```

Expected: `README.md  finetuning_vbll.py  notes.md  pyproject.toml  src/`

- [ ] **Step 4: Write the failing test for the VBLL adapter hook**

`src/bllarse/tools/adapters.py` on `main` has no VBLL entry point. Create `tests/tools/test_adapters_vbll.py`:

```python
from bllarse.tools.adapters import config_to_argv


def test_config_to_argv_negates_listed_bools():
    argv = config_to_argv({"return_ood": False, "px": False, "epochs": 3})
    assert "--no-return-ood" in argv
    assert "--px" not in argv
    assert "--no-px" not in argv
    assert argv[argv.index("--epochs") + 1] == "3"


def test_config_to_argv_keeps_true_bools_as_bare_flags():
    argv = config_to_argv({"return_ood": True})
    assert argv == ["--return-ood"]


def test_run_vbll_training_from_config_is_exported():
    from bllarse.tools import adapters
    assert callable(adapters.run_vbll_training_from_config)
```

- [ ] **Step 5: Run it to confirm it fails**

```bash
uv run pytest tests/tools/test_adapters_vbll.py -v
```

Expected: FAIL — `--no-return-ood` not in argv, and `AttributeError: run_vbll_training_from_config`.

- [ ] **Step 6: Port the two behaviours onto `main`'s adapters**

Do **not** copy the old file wholesale — it predates the #48 cleanup. In `src/bllarse/tools/adapters.py`, add the negatable set above `config_to_argv` and the new entry point below `run_training_from_config`:

```python
# Config keys whose False value must be passed explicitly as --no-<flag> rather
# than omitted, because the target script's default is True.
NEGATABLE_FLAGS = frozenset({"return_ood"})
```

In `config_to_argv`, replace the bool branch:

```python
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            elif key in NEGATABLE_FLAGS:
                argv.append("--no-" + key.replace("_", "-"))
            continue
```

And add:

```python
def run_vbll_training_from_config(config: Mapping[str, Any]) -> None:
    """Run the PyTorch VBLL finetuning script from a config."""
    run_script_from_config("scripts/vbll_pytorch/finetuning_vbll.py", config)
```

Note the path is repo-root-relative: `main`'s `run_script_from_config` resolves that way, unlike the old `scripts/`-relative version.

- [ ] **Step 7: Run the test to confirm it passes**

```bash
uv run pytest tests/tools/test_adapters_vbll.py -v
```

Expected: 3 passed.

- [ ] **Step 8: Add the eval-dump tree to `.gitignore`**

Append to `.gitignore`:

```
# Eval-dump artifacts (logits/scores per run) — large, regenerable, never committed
eval_dumps/
scripts/results/
```

- [ ] **Step 9: Run the whole suite to confirm nothing regressed**

```bash
uv run pytest tests/tools tests/losses -v
```

Expected: all pass. Tests needing GPU or network may skip or error on collection — record which, and do not "fix" them in this task.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "chore: consolidate alternatives and vbll_pytorch onto the protocol branch

Extracts code (not run outputs) from ibprobit-process-noise and the VBLL
subproject from experimental_infra_vbll_baseline. Ports the VBLL adapter
entry point and negatable-bool handling forward onto main's cleaned
adapters rather than reverting to the pre-#48 version."
```

---

## Task 2: `bllarse.metrics` — classification metrics and temperature scaling

**Files:**
- Create: `src/bllarse/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `accuracy(logits, labels) -> float`
  - `ece(logits, labels, num_bins=20) -> float`
  - `nll(logits, labels) -> float`
  - `brier(logits, labels) -> float`
  - `fit_temperature(logits, labels, num_iters=16) -> float` — returns **T**, not β
  - `apply_temperature(logits, temperature) -> Array`
  - `msp_ood_score(logits) -> Array` — shape `(N,)`, **higher = more OOD**
  - `entropy_ood_score(logits) -> Array` — shape `(N,)`, **higher = more OOD**
  - `classification_metrics(logits, labels, num_bins=20) -> dict` with keys `acc`, `ece`, `nll`, `brier`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metrics.py`:

```python
import jax.numpy as jnp
import numpy as np
import pytest

from bllarse import metrics


def test_accuracy_perfect_and_zero():
    logits = jnp.array([[5.0, 0.0], [0.0, 5.0]])
    assert metrics.accuracy(logits, jnp.array([0, 1])) == pytest.approx(1.0)
    assert metrics.accuracy(logits, jnp.array([1, 0])) == pytest.approx(0.0)


def test_brier_is_full_vector_not_top_label():
    # Uniform predictive over 4 classes: each p_k = 0.25.
    # Full-vector Brier = (1-0.25)^2 + 3*(0.25)^2 = 0.5625 + 0.1875 = 0.75
    logits = jnp.zeros((1, 4))
    assert metrics.brier(logits, jnp.array([0])) == pytest.approx(0.75, abs=1e-6)


def test_brier_zero_for_perfect_confident_prediction():
    logits = jnp.array([[100.0, -100.0, -100.0]])
    assert metrics.brier(logits, jnp.array([0])) == pytest.approx(0.0, abs=1e-6)


def test_nll_matches_hand_computation_for_uniform():
    # Uniform over 10 classes -> NLL = log(10)
    logits = jnp.zeros((1, 10))
    assert metrics.nll(logits, jnp.array([3])) == pytest.approx(np.log(10.0), abs=1e-6)


def _calibrated_logits_and_labels(seed, n=4000, k=5):
    """Logits that are calibrated at T=1, by SAMPLING labels from softmax(base).

    Do not use labels = base.argmax(-1) here: that makes the model correct on
    every point, so NLL falls monotonically as T -> 0 and there is no interior
    optimum for fit_temperature to find.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, k)).astype(np.float32)
    probs = np.exp(base) / np.exp(base).sum(axis=-1, keepdims=True)
    labels = np.array([rng.choice(k, p=row) for row in probs])
    return base, jnp.asarray(labels)


def test_fit_temperature_recovers_a_known_scaling():
    # base is calibrated at T=1, so base * 4 is overconfident by exactly 4x and
    # the NLL-optimal temperature is 4.
    base, labels = _calibrated_logits_and_labels(0)
    overconfident = jnp.asarray(base * 4.0)

    T = metrics.fit_temperature(overconfident, labels)
    assert 3.0 < T < 5.0, f"expected T near 4, got {T}"

    # Applying it must not change accuracy — temperature is rank-preserving.
    before = metrics.accuracy(overconfident, labels)
    after = metrics.accuracy(metrics.apply_temperature(overconfident, T), labels)
    assert before == pytest.approx(after)


def test_fit_temperature_reduces_nll():
    base, labels = _calibrated_logits_and_labels(1)
    overconfident = jnp.asarray(base * 4.0)
    T = metrics.fit_temperature(overconfident, labels)
    assert metrics.nll(metrics.apply_temperature(overconfident, T), labels) < metrics.nll(
        overconfident, labels
    )


def test_msp_ood_score_is_higher_for_uniform_than_confident():
    confident = jnp.array([[10.0, 0.0, 0.0]])
    uniform = jnp.zeros((1, 3))
    assert metrics.msp_ood_score(uniform)[0] > metrics.msp_ood_score(confident)[0]


def test_entropy_ood_score_is_higher_for_uniform_than_confident():
    confident = jnp.array([[10.0, 0.0, 0.0]])
    uniform = jnp.zeros((1, 3))
    assert metrics.entropy_ood_score(uniform)[0] > metrics.entropy_ood_score(confident)[0]


def test_entropy_ood_score_maximum_is_log_k():
    uniform = jnp.zeros((1, 8))
    assert float(metrics.entropy_ood_score(uniform)[0]) == pytest.approx(np.log(8.0), abs=1e-6)


def test_classification_metrics_returns_all_four_keys():
    rng = np.random.default_rng(2)
    logits = jnp.asarray(rng.normal(size=(500, 10)).astype(np.float32))
    labels = jnp.asarray(rng.integers(0, 10, size=500))
    out = metrics.classification_metrics(logits, labels)
    assert set(out) == {"acc", "ece", "nll", "brier"}
    assert all(isinstance(v, float) for v in out.values())


def test_ood_scores_do_not_collapse_to_ties_on_float16_input():
    """Dumps are fp16, and jax.nn.softmax returns fp16 for fp16 input. Without the
    upcast, MSP over 5000 points collapses to a few hundred distinct values --
    and AUROC gives ties half credit, so Fig 2's detection metrics would be
    measurably wrong. Overflow is NOT the risk here: softmax subtracts the max
    internally, so raw fp16 logits never actually overflow through this path."""
    rng = np.random.default_rng(1)
    logits16 = rng.normal(size=(5000, 100)).astype(np.float16)

    scores = np.asarray(metrics.msp_ood_score(logits16))
    assert scores.dtype == np.float32
    distinct = len(np.unique(scores))
    assert distinct > 4000, f"only {distinct} distinct scores -- fp16 ties will distort AUROC"


def test_metrics_accept_float16_input(): 
    logits = jnp.asarray(np.full((10, 4), 60.0, dtype=np.float16))
    labels = jnp.zeros((10,), dtype=jnp.int32)
    assert np.isfinite(metrics.nll(logits, labels))
    assert np.isfinite(metrics.brier(logits, labels))
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/test_metrics.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'bllarse.metrics'`.

- [ ] **Step 3: Implement `src/bllarse/metrics.py`**

```python
"""Classification metrics and temperature scaling, on (logits, labels).

One implementation, used both in training loops and by post-hoc analysis over
dumped logits. Dumps are float16 on disk, so every entry point upcasts to
float32 first. The risk is precision, not overflow: jax.nn.softmax subtracts the
max internally so fp16 logits never overflow, but it also RETURNS fp16, and an
fp16 MSP over 5000 points collapses to a few hundred distinct values. AUROC
gives ties half credit, so that would distort the detection figures.

OOD score functions in this module follow the project-wide convention:
**higher means more OOD**.
"""

from typing import Any

import jax.numpy as jnp
from jax import lax, nn
from jaxtyping import Array
from tensorflow_probability.substrates.jax.stats import (
    expected_calibration_error as _tfp_ece,
)

DEFAULT_NUM_BINS = 20


def _f32(x: Any) -> Array:
    """Upcast to float32. Dumped logits are fp16; without this, softmax returns
    fp16 and the score functions lose the ranking resolution AUROC depends on."""
    return jnp.asarray(x, dtype=jnp.float32)


def accuracy(logits: Any, labels: Any) -> float:
    logits = _f32(logits)
    return float(jnp.mean(jnp.argmax(logits, axis=-1) == jnp.asarray(labels)))


def ece(logits: Any, labels: Any, num_bins: int = DEFAULT_NUM_BINS) -> float:
    logits = _f32(logits)
    labels = jnp.asarray(labels)
    predictions = jnp.argmax(logits, axis=-1)
    return float(
        _tfp_ece(num_bins, logits=logits, labels_true=labels, labels_predicted=predictions)
    )


def nll(logits: Any, labels: Any) -> float:
    logits = _f32(logits)
    labels = jnp.asarray(labels)
    log_probs = nn.log_softmax(logits, axis=-1)
    return float(-jnp.mean(jnp.take_along_axis(log_probs, labels[:, None], axis=-1)))


def brier(logits: Any, labels: Any) -> float:
    """Multiclass Brier over the FULL predictive vector, in [0, 2].

    Not the top-label variant -- results-plan.md 0.1 is explicit about which.
    """
    logits = _f32(logits)
    p = nn.softmax(logits, axis=-1)
    onehot = nn.one_hot(jnp.asarray(labels), p.shape[-1])
    return float(jnp.mean(jnp.sum((p - onehot) ** 2, axis=-1)))


def classification_metrics(
    logits: Any, labels: Any, num_bins: int = DEFAULT_NUM_BINS
) -> dict[str, float]:
    return {
        "acc": accuracy(logits, labels),
        "ece": ece(logits, labels, num_bins),
        "nll": nll(logits, labels),
        "brier": brier(logits, labels),
    }


def fit_temperature(logits: Any, labels: Any, num_iters: int = 16) -> float:
    """Fit one scalar temperature T > 0 by minimising NLL of softmax(z / T).

    Newton on beta = 1/T with the closed-form gradient and Hessian of softmax
    cross-entropy. NLL is convex in beta, so this reaches the global optimum in a
    handful of steps -- no learning rate to tune, unlike a gradient fit.

    Returns T (the divisor), not beta.
    """
    logits = _f32(logits)
    labels = jnp.asarray(labels)
    target_z = jnp.take_along_axis(logits, labels[:, None], axis=-1).squeeze(-1)

    def newton_step(beta, _):
        p = nn.softmax(beta * logits, axis=-1)
        e_z = jnp.sum(p * logits, axis=-1)
        e_z2 = jnp.sum(p * logits * logits, axis=-1)
        grad = jnp.mean(e_z - target_z)
        hess = jnp.mean(e_z2 - e_z * e_z)
        return jnp.clip(beta - grad / (hess + 1e-8), 0.01, 100.0), None

    beta, _ = lax.scan(newton_step, jnp.ones(()), jnp.arange(num_iters))
    return float(1.0 / beta)


def apply_temperature(logits: Any, temperature: float) -> Array:
    return _f32(logits) / temperature


def msp_ood_score(logits: Any) -> Array:
    """1 - max softmax probability. Higher = more OOD."""
    return 1.0 - jnp.max(nn.softmax(_f32(logits), axis=-1), axis=-1)


def entropy_ood_score(logits: Any) -> Array:
    """Predictive entropy in nats. Higher = more OOD. Maximum is log(K)."""
    logits = _f32(logits)
    log_p = nn.log_softmax(logits, axis=-1)
    return -jnp.sum(jnp.exp(log_p) * log_p, axis=-1)
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_metrics.py -v
```

Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bllarse/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): shared classification metrics, Brier, and temperature scaling

One implementation for in-loop and post-hoc use. Brier is the full-vector
multiclass form. fit_temperature is Newton on beta=1/T (convex, no LR to
tune). OOD score functions follow the higher-means-more-OOD convention."
```

---

## Task 3: `bllarse.detection` — OOD detection metrics

**Files:**
- Create: `src/bllarse/detection.py`
- Test: `tests/test_detection.py`

**Interfaces:**
- Consumes: nothing. Takes raw score arrays, so it is independent of Task 2.
- Produces:
  - `auroc(scores_id, scores_ood) -> float`
  - `fpr_at_tpr(scores_id, scores_ood, tpr_target=0.95) -> float`
  - `aupr(scores_id, scores_ood) -> float`
  - `detection_metrics(scores_id, scores_ood) -> dict` with keys `auroc`, `fpr95`, `aupr`

NumPy rather than JAX: this is post-hoc-only (Fig 2 is built from dumps), gets no
GPU benefit, and rank/tie handling is clearer.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_detection.py`:

```python
import numpy as np
import pytest

from bllarse import detection


def test_auroc_perfect_separation():
    # Convention: higher score = more OOD. Perfectly separated -> 1.0
    assert detection.auroc(np.array([0.0, 0.1, 0.2]), np.array([1.0, 1.1])) == pytest.approx(1.0)


def test_auroc_inverted_separation():
    assert detection.auroc(np.array([1.0, 1.1]), np.array([0.0, 0.1])) == pytest.approx(0.0)


def test_auroc_identical_distributions_is_half():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    b = rng.normal(size=5000)
    assert detection.auroc(a, b) == pytest.approx(0.5, abs=0.02)


def test_auroc_handles_ties_as_half_credit():
    # All scores equal -> no discrimination -> exactly 0.5
    assert detection.auroc(np.zeros(10), np.zeros(10)) == pytest.approx(0.5)


def test_fpr_at_tpr_perfect_separation_is_zero():
    fpr = detection.fpr_at_tpr(np.array([0.0, 0.1, 0.2]), np.array([1.0, 1.1]))
    assert fpr == pytest.approx(0.0)


def test_fpr_at_tpr_captures_required_tpr():
    # 100 OOD at 1.0, 100 ID uniform on [0, 2]. To capture 95% of OOD we need a
    # threshold at or below 1.0, which admits roughly half the ID points.
    rng = np.random.default_rng(1)
    scores_id = rng.uniform(0.0, 2.0, size=10000)
    scores_ood = np.full(1000, 1.0)
    assert detection.fpr_at_tpr(scores_id, scores_ood) == pytest.approx(0.5, abs=0.03)


def test_aupr_perfect_separation():
    assert detection.aupr(np.array([0.0, 0.1]), np.array([1.0, 1.1])) == pytest.approx(1.0)


def test_aupr_baseline_is_positive_rate():
    # With no signal, average precision tends to the positive (OOD) base rate.
    rng = np.random.default_rng(2)
    scores_id = rng.normal(size=9000)
    scores_ood = rng.normal(size=1000)
    assert detection.aupr(scores_id, scores_ood) == pytest.approx(0.1, abs=0.03)


def test_detection_metrics_returns_all_three_keys():
    out = detection.detection_metrics(np.array([0.0, 0.1]), np.array([1.0]))
    assert set(out) == {"auroc", "fpr95", "aupr"}


def test_rejects_empty_input():
    with pytest.raises(ValueError):
        detection.auroc(np.array([]), np.array([1.0]))
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/test_detection.py -v
```

Expected: collection error — no module `bllarse.detection`.

- [ ] **Step 3: Implement `src/bllarse/detection.py`**

```python
"""OOD detection metrics: AUROC, FPR@95TPR, AUPR.

Convention, applied everywhere in this project: **higher score = more OOD**.
The OOD set is the positive class. Callers pass score arrays already in that
orientation (see bllarse.metrics.msp_ood_score / entropy_ood_score); nothing
here inspects or flips the sign.

NumPy rather than JAX deliberately: this is post-hoc analysis over dumped
arrays, gets no GPU benefit, and tie handling reads more clearly.
"""

import numpy as np


def _validate(scores_id: np.ndarray, scores_ood: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores_id = np.asarray(scores_id, dtype=np.float64).ravel()
    scores_ood = np.asarray(scores_ood, dtype=np.float64).ravel()
    if scores_id.size == 0 or scores_ood.size == 0:
        raise ValueError("both score arrays must be non-empty")
    return scores_id, scores_ood


def auroc(scores_id, scores_ood) -> float:
    """Area under the ROC curve, via the Mann-Whitney U identity.

    AUROC = P(score_ood > score_id) + 0.5 * P(score_ood == score_id).
    Average ranks give tied scores half credit, which is why an all-ties input
    returns exactly 0.5 rather than 0 or 1.
    """
    scores_id, scores_ood = _validate(scores_id, scores_ood)
    n_id, n_ood = scores_id.size, scores_ood.size

    combined = np.concatenate([scores_ood, scores_id])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(combined.size, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1, dtype=np.float64)

    # Average the ranks within each group of tied values.
    sorted_scores = combined[order]
    start = 0
    for end in range(1, sorted_scores.size + 1):
        if end == sorted_scores.size or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end

    rank_sum_ood = ranks[:n_ood].sum()
    u = rank_sum_ood - n_ood * (n_ood + 1) / 2.0
    return float(u / (n_id * n_ood))


def fpr_at_tpr(scores_id, scores_ood, tpr_target: float = 0.95) -> float:
    """False positive rate on the ID set at the threshold achieving `tpr_target`.

    The threshold is the largest value that still captures at least
    `tpr_target` of the OOD set; FPR is the fraction of ID points at or above it.
    """
    scores_id, scores_ood = _validate(scores_id, scores_ood)
    if not 0.0 < tpr_target <= 1.0:
        raise ValueError("tpr_target must be in (0, 1]")

    # The threshold capturing >= tpr_target of OOD is the (1 - tpr_target)
    # quantile of the OOD scores, taken from below.
    threshold = np.quantile(scores_ood, 1.0 - tpr_target, method="lower")
    return float(np.mean(scores_id >= threshold))


def aupr(scores_id, scores_ood) -> float:
    """Average precision with OOD as the positive class.

    Step-interpolated (the `sum((R_n - R_{n-1}) * P_n)` form), not trapezoidal,
    which is the standard non-optimistic estimator.
    """
    scores_id, scores_ood = _validate(scores_id, scores_ood)
    scores = np.concatenate([scores_ood, scores_id])
    is_positive = np.concatenate(
        [np.ones(scores_ood.size, dtype=bool), np.zeros(scores_id.size, dtype=bool)]
    )

    order = np.argsort(-scores, kind="mergesort")
    is_positive = is_positive[order]

    tp = np.cumsum(is_positive)
    fp = np.cumsum(~is_positive)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / scores_ood.size

    recall_delta = np.diff(recall, prepend=0.0)
    return float(np.sum(recall_delta * precision))


def detection_metrics(scores_id, scores_ood) -> dict[str, float]:
    return {
        "auroc": auroc(scores_id, scores_ood),
        "fpr95": fpr_at_tpr(scores_id, scores_ood, 0.95),
        "aupr": aupr(scores_id, scores_ood),
    }
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_detection.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bllarse/detection.py tests/test_detection.py
git commit -m "feat(detection): AUROC, FPR@95TPR and AUPR for OOD detection

New capability -- the repo had shift-accuracy metrics only, which answer a
different question from detecting shift. Higher-score-means-more-OOD
convention, ties given half credit."
```

---

## Task 4: `bllarse.splits` — pinned, seed-independent val/test splits

**Files:**
- Create: `src/bllarse/splits.py`
- Test: `tests/test_splits.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `make_pinned_split(n, val_fraction, dataset) -> tuple[np.ndarray, np.ndarray]` returning `(val_idx, test_idx)`. **No seed parameter** — the fixed `SPLIT_SEED` lives as a module constant, because an overridable seed is the exact footgun this module exists to remove.
  - `pinned_split_path(cache_dir, dataset, split, val_fraction) -> Path`
  - `load_or_make_pinned_split(cache_dir, dataset, split, n, val_fraction) -> tuple[np.ndarray, np.ndarray]`

The whole point is that the split does **not** depend on the run seed. Today
`--val-fraction` reshuffles per seed, which would put ~40 differently-conditioned
temperature fits on every curve.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_splits.py`:

```python
import numpy as np
import pytest

from bllarse import splits


def test_split_is_a_partition():
    val_idx, test_idx = splits.make_pinned_split(100, 0.1, "cifar10")
    assert len(val_idx) == 10
    assert len(test_idx) == 90
    assert set(val_idx).isdisjoint(test_idx)
    assert sorted(np.concatenate([val_idx, test_idx])) == list(range(100))


def test_split_is_independent_of_call_order_and_stable():
    a = splits.make_pinned_split(1000, 0.1, "cifar10")
    b = splits.make_pinned_split(1000, 0.1, "cifar10")
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_different_datasets_get_different_splits():
    a, _ = splits.make_pinned_split(1000, 0.1, "cifar10")
    b, _ = splits.make_pinned_split(1000, 0.1, "cifar100")
    assert not np.array_equal(a, b)


def test_split_does_not_depend_on_any_run_seed():
    # There is no seed parameter to pass; the signature itself enforces this.
    import inspect

    params = inspect.signature(splits.make_pinned_split).parameters
    assert "seed" not in params, "run seed must not influence the pinned split"


def test_rejects_degenerate_fractions():
    with pytest.raises(ValueError):
        splits.make_pinned_split(100, 0.0, "cifar10")
    with pytest.raises(ValueError):
        splits.make_pinned_split(100, 1.0, "cifar10")


def test_load_or_make_round_trips_through_disk(tmp_path):
    first = splits.load_or_make_pinned_split(tmp_path, "cifar10", "test", 500, 0.1)
    path = splits.pinned_split_path(tmp_path, "cifar10", "test", 0.1)
    assert path.exists()
    second = splits.load_or_make_pinned_split(tmp_path, "cifar10", "test", 500, 0.1)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_load_or_make_rejects_a_cached_split_of_the_wrong_size(tmp_path):
    splits.load_or_make_pinned_split(tmp_path, "cifar10", "test", 500, 0.1)
    with pytest.raises(ValueError, match="size"):
        splits.load_or_make_pinned_split(tmp_path, "cifar10", "test", 600, 0.1)
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/test_splits.py -v
```

Expected: collection error — no module `bllarse.splits`.

- [ ] **Step 3: Implement `src/bllarse/splits.py`**

```python
"""Pinned, seed-independent val/test splits.

Temperature scaling is fit on a held-out split (results-plan.md 0.2) and, under
the per-dump-point refit schedule, the same split is reused ~40 times per run.
If the split moved with the run seed, every curve would carry 40 differently
conditioned fits and no two seeds would be comparable. So the partition is
derived from the dataset name alone.
"""

import hashlib
from pathlib import Path

import numpy as np

# Fixed project-wide constant. Changing it invalidates every existing dump.
SPLIT_SEED = 20260730


def _dataset_seed(dataset: str) -> int:
    """Stable per-dataset seed. Python's hash() is salted per process, so it
    cannot be used here -- the value must be identical across machines and runs."""
    digest = hashlib.sha256(f"{SPLIT_SEED}:{dataset}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def make_pinned_split(
    n: int, val_fraction: float, dataset: str
) -> tuple[np.ndarray, np.ndarray]:
    """Partition range(n) into (val_idx, test_idx). No run seed, by design."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    n_val = int(round(n * val_fraction))
    if n_val == 0 or n_val == n:
        raise ValueError(
            f"val_fraction={val_fraction} yields a degenerate split for n={n}"
        )

    rng = np.random.default_rng(_dataset_seed(dataset))
    perm = rng.permutation(n)
    return np.sort(perm[:n_val]), np.sort(perm[n_val:])


def pinned_split_path(cache_dir, dataset: str, split: str, val_fraction: float) -> Path:
    return Path(cache_dir) / f"{dataset}_{split}_val{val_fraction}.npz"


def load_or_make_pinned_split(
    cache_dir, dataset: str, split: str, n: int, val_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Load the cached split, or create and cache it. Fails loudly on size drift."""
    path = pinned_split_path(cache_dir, dataset, split, val_fraction)
    if path.exists():
        data = np.load(path)
        val_idx, test_idx = data["val_idx"], data["test_idx"]
        cached_n = val_idx.size + test_idx.size
        if cached_n != n:
            raise ValueError(
                f"cached split at {path} covers {cached_n} items but {n} were "
                f"requested -- the underlying dataset changed; delete the cache "
                f"deliberately if that is intended"
            )
        return val_idx, test_idx

    val_idx, test_idx = make_pinned_split(n, val_fraction, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, val_idx=val_idx, test_idx=test_idx)
    return val_idx, test_idx
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_splits.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bllarse/splits.py tests/test_splits.py
git commit -m "feat(splits): pinned seed-independent val/test partitions

--val-fraction currently reshuffles per run seed, which would make the
per-dump-point temperature fits incomparable across seeds and arms. The
partition is now derived from the dataset name alone."
```

---

## Task 5: `bllarse.evaldump` — logit dump writer, reader, and schedule

**Files:**
- Create: `src/bllarse/evaldump.py`
- Test: `tests/test_evaldump.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `dump_schedule(num_steps, n_dumps=40) -> np.ndarray` — log-spaced, always includes 1 and `num_steps`
  - `dump_dir(root, group_id, uid) -> Path`
  - `write_dump(root, group_id, uid, step, logits, labels, *, arm, split, logit_kind, seed, samples_seen, score=None, score_kind=None, eps_hat=None) -> Path`
  - `load_dump(path) -> DumpRecord` — a dataclass with `logits`, `labels`, `score`, `meta`
  - `iter_dumps(root, group_id, uid) -> Iterator[DumpRecord]` — ordered by step

- [ ] **Step 1: Write the failing tests**

Create `tests/test_evaldump.py`:

```python
import numpy as np
import pytest

from bllarse import evaldump


# Size 67 is deliberate: it is one of the ~2% of sizes where the gap sequence
# dips by exactly one at the boundary between the forced-increment region and
# the geometric region. Asserting strictly non-decreasing gaps would fail there.
@pytest.mark.parametrize(
    "num_steps,n_dumps",
    [(41, 40), (67, 40), (100, 40), (200, 40), (500, 40), (2000, 40), (60, 8), (3000, 16)],
)
def test_dump_schedule_invariants(num_steps, n_dumps):
    """The four properties that actually hold.

    Verified by brute force over n_dumps 2..60 x num_steps up to n_dumps+2000
    (118k combinations): zero violations of any of these. A naive
    round-then-unique returns 25-36 points instead of 40 over this range,
    thinning exactly the early region the schedule exists to resolve.
    """
    sched = evaldump.dump_schedule(num_steps, n_dumps=n_dumps)
    assert len(sched) == n_dumps, "must return exactly the number of points requested"
    assert sched[0] == 1
    assert sched[-1] == num_steps

    gaps = np.diff(sched)
    assert np.all(gaps > 0), f"must be strictly increasing, got {sched}"
    # Gaps trend upward, but dip by at most one where the forced-increment
    # region meets the geometric one. Do NOT tighten this to >= 0: that is false
    # at ~2% of sizes. Do NOT assert an absolute fraction of points in the first
    # tenth either -- arithmetically impossible at moderate num_steps, since at
    # 100 steps only 10 of 40 points can land at or below step 10.
    assert np.all(np.diff(gaps) >= -1), f"gaps dipped by more than one: {gaps}"


def test_dump_schedule_concentrates_early_at_the_operating_point():
    """Early concentration, asserted at the configuration the sweeps use.

    It is not assertable in general: when num_steps approaches n_dumps nearly
    every step is dumped and there is no concentration to be had.
    """
    sched = evaldump.dump_schedule(500, n_dumps=40)
    assert sched[20] < 500 / 4, "half the points should land in the first quarter"


def test_dump_schedule_rejects_a_degenerate_n_dumps():
    with pytest.raises(ValueError):
        evaldump.dump_schedule(500, n_dumps=1)


def test_dump_schedule_handles_fewer_steps_than_dumps():
    sched = evaldump.dump_schedule(5, n_dumps=40)
    assert list(sched) == [1, 2, 3, 4, 5]


def test_write_and_load_round_trip(tmp_path):
    logits = np.random.default_rng(0).normal(size=(64, 10)).astype(np.float32)
    labels = np.arange(64) % 10
    path = evaldump.write_dump(
        tmp_path, "grp", "uid1", step=7, logits=logits, labels=labels,
        arm="adamw_ce", split="test", logit_kind="softmax_logits",
        seed=137, samples_seen=448,
    )
    rec = evaldump.load_dump(path)
    assert rec.logits.dtype == np.float16, "dumps are fp16 on disk"
    np.testing.assert_allclose(rec.logits, logits, atol=1e-2)
    np.testing.assert_array_equal(rec.labels, labels)
    assert rec.meta["arm"] == "adamw_ce"
    assert rec.meta["step"] == 7
    assert rec.meta["samples_seen"] == 448
    assert rec.score is None


def test_write_carries_optional_score_and_eps(tmp_path):
    logits = np.zeros((8, 3), dtype=np.float32)
    path = evaldump.write_dump(
        tmp_path, "grp", "uid2", step=1, logits=logits, labels=np.zeros(8, dtype=int),
        arm="ibprobit", split="test", logit_kind="moderated_score",
        seed=137, samples_seen=64,
        score=np.arange(8, dtype=np.float32), score_kind="predictive_variance",
        eps_hat=0.03,
    )
    rec = evaldump.load_dump(path)
    np.testing.assert_allclose(rec.score, np.arange(8))
    assert rec.meta["score_kind"] == "predictive_variance"
    assert rec.meta["eps_hat"] == pytest.approx(0.03)


def test_labels_and_logits_must_agree_in_length(tmp_path):
    with pytest.raises(ValueError, match="length"):
        evaldump.write_dump(
            tmp_path, "grp", "uid3", step=1,
            logits=np.zeros((8, 3)), labels=np.zeros(7, dtype=int),
            arm="adamw_ce", split="test", logit_kind="softmax_logits",
            seed=1, samples_seen=1,
        )


def test_iter_dumps_returns_steps_in_order(tmp_path):
    for step in (13, 1, 5):
        evaldump.write_dump(
            tmp_path, "grp", "uid4", step=step,
            logits=np.zeros((4, 2)), labels=np.zeros(4, dtype=int),
            arm="adamw_ce", split="test", logit_kind="softmax_logits",
            seed=1, samples_seen=step * 8,
        )
    steps = [rec.meta["step"] for rec in evaldump.iter_dumps(tmp_path, "grp", "uid4")]
    assert steps == [1, 5, 13]
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/test_evaldump.py -v
```

Expected: collection error — no module `bllarse.evaldump`.

- [ ] **Step 3: Implement `src/bllarse/evaldump.py`**

```python
"""Per-eval-point logit dumps -- the source of truth for every reported figure.

Everything downstream (accuracy, ECE at any binning, NLL, Brier, temperature
scaling, threshold crossings, OOD detection) is computed from these files, so a
run is measured once and never re-run to add a metric.

Logits are float16 on disk: at ImageNet's K=1000 a 5k eval subset is 10 MB per
point, and a full 50k final eval is 100 MB. Readers upcast before any
exponentiation (see bllarse.metrics).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


@dataclass(frozen=True)
class DumpRecord:
    logits: np.ndarray
    labels: np.ndarray
    score: np.ndarray | None
    meta: dict[str, Any]


def dump_schedule(num_steps: int, n_dumps: int = 40) -> np.ndarray:
    """Log-spaced eval steps: dense early, sparse late.

    Fig 5's claim is that the calibration gap is largest early, so resolution
    belongs there. Dumping all `num_steps` points would be ~5 GB per run for no
    additional information about the claim.

    Rounding a geometric grid to integers collapses its early points -- at
    num_steps=100 the first twenty all land below 3 -- so a naive unique() drops
    exactly the resolution this schedule exists to provide, and silently returns
    fewer points than were asked for. Each step is therefore advanced to at least
    one past its predecessor, giving exactly min(num_steps, n_dumps) distinct
    steps.

    Gaps trend upward but are not monotone: where the forced-increment region
    meets the geometric one they can dip by exactly one, at roughly 2% of sizes.
    Verified by brute force over 118k (n_dumps, num_steps) combinations -- the
    dip never exceeds one.

    Always includes step 1 and step `num_steps`.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if n_dumps < 2:
        raise ValueError("n_dumps must be >= 2 so the schedule spans 1..num_steps")
    if num_steps <= n_dumps:
        return np.arange(1, num_steps + 1)

    steps: list[int] = []
    previous = 0
    for value in np.geomspace(1, num_steps, n_dumps):
        step = max(int(round(value)), previous + 1)
        if step > num_steps:
            break
        steps.append(step)
        previous = step
    return np.asarray(steps, dtype=int)


def dump_dir(root, group_id: str, uid: str) -> Path:
    return Path(root) / group_id / uid


def write_dump(
    root,
    group_id: str,
    uid: str,
    step: int,
    logits,
    labels,
    *,
    arm: str,
    split: str,
    logit_kind: str,
    seed: int,
    samples_seen: int,
    score=None,
    score_kind: str | None = None,
    eps_hat: float | None = None,
) -> Path:
    """Write one eval point. Returns the path written.

    `logit_kind` records what the array actually is -- "softmax_logits" for CE
    and VBLL, "moderated_score" for IBProbit (q-update-spec.md 5.8) -- so no
    downstream consumer has to infer it.

    `score` is the arm's native epistemic score and is diagnostic only: the
    primary Fig 2 comparison uses MSP and entropy derived from `logits`, which
    every arm has. It is absent for AdamW+CE, which has no such quantity.
    """
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"logits and labels length mismatch: {logits.shape[0]} vs {labels.shape[0]}"
        )
    # Labels are stored int16. This module exists to stop silent, unrecoverable
    # loss, and an out-of-range cast wraps rather than raising.
    if labels.size and (labels.min() < -32768 or labels.max() > 32767):
        raise ValueError(
            f"labels outside int16 range [{labels.min()}, {labels.max()}]; "
            f"storing them would wrap silently"
        )

    meta = {
        "arm": arm,
        "step": int(step),
        "samples_seen": int(samples_seen),
        "split": split,
        "logit_kind": logit_kind,
        "score_kind": score_kind,
        "seed": int(seed),
        "eps_hat": None if eps_hat is None else float(eps_hat),
    }

    payload = {
        "logits": logits.astype(np.float16),
        "labels": labels.astype(np.int16),
        "meta": np.asarray(json.dumps(meta)),
    }
    if score is not None:
        payload["score"] = np.asarray(score, dtype=np.float32)

    directory = dump_dir(root, group_id, uid)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step_{int(step):07d}.npz"
    np.savez_compressed(path, **payload)
    return path


def load_dump(path) -> DumpRecord:
    data = np.load(path, allow_pickle=False)
    return DumpRecord(
        logits=data["logits"],
        labels=data["labels"],
        score=data["score"] if "score" in data.files else None,
        meta=json.loads(str(data["meta"])),
    )


def iter_dumps(root, group_id: str, uid: str) -> Iterator[DumpRecord]:
    """Yield every dump for one run, ordered by step."""
    directory = dump_dir(root, group_id, uid)
    for path in sorted(directory.glob("step_*.npz")):
        yield load_dump(path)
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_evaldump.py -v
```

Expected: 15 passed (the invariants are parametrised over eight size pairs).

- [ ] **Step 5: Commit**

```bash
git add src/bllarse/evaldump.py tests/test_evaldump.py
git commit -m "feat(evaldump): fp16 logit dumps and the log-spaced eval schedule

Dumps are the source of truth for every figure metric, so a run is measured
once. Schedule is dense early where the sample-efficiency claim lives."
```

---

## Task 6: `bllarse.features` — the shared cached-feature loader

**Files:**
- Create: `src/bllarse/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `feature_cache_path(cache_dir, model, dataset, img_size, split) -> Path`
  - `feature_hf_path(model, dataset, img_size, split) -> str`
  - `load_cached_features(cache_dir, model, dataset, img_size, split, hf_repo=None) -> tuple[np.ndarray, np.ndarray]`

Both probing arms read the same `.npz` files, so this loader must be importable
from the PyTorch container as well as the JAX one. That rules out reusing
`ood_evaluation.load_cached_features`, which reaches `vit_classification` and so
drags in `mlpox`, `datasets`, `PIL`, `equinox`, `optax` and `jax` for what is a
`np.load` with an HF fallback.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_features.py`:

```python
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bllarse import features


def test_cache_path_matches_the_precompute_layout():
    path = features.feature_cache_path(".cache/features", "dinov3_huge", "cifar10", 224, "test")
    assert Path(path).name == "dinov3_huge_cifar10_res224_test.npz"


def test_hf_path_matches_the_mirror_layout():
    assert features.feature_hf_path("dinov3_huge", "cifar10", 224, "test") == (
        "dinov3_huge/cifar10_res224_test.npz"
    )


def test_loads_from_the_local_cache(tmp_path):
    feats = np.arange(12, dtype=np.float32).reshape(4, 3)
    labels = np.array([0, 1, 0, 1])
    np.savez(tmp_path / "dinov3_small_fake_res224_train.npz", features=feats, labels=labels)

    got_f, got_l = features.load_cached_features(tmp_path, "dinov3_small", "fake", 224, "train")
    np.testing.assert_array_equal(got_f, feats)
    np.testing.assert_array_equal(got_l, labels)


def test_missing_features_raise_a_message_naming_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="precompute_features"):
        features.load_cached_features(tmp_path, "dinov3_small", "absent", 224, "train")


def test_falls_back_to_hf_when_the_local_cache_misses(tmp_path, monkeypatch):
    remote = tmp_path / "remote.npz"
    feats = np.ones((2, 3), dtype=np.float32)
    np.savez(remote, features=feats, labels=np.zeros(2, dtype=int))

    calls = {}

    def fake_download(repo_id, filename, repo_type):
        calls["repo_id"] = repo_id
        calls["filename"] = filename
        calls["repo_type"] = repo_type
        return str(remote)

    monkeypatch.setattr(features, "_hf_hub_download", fake_download)

    cache = tmp_path / "cache"
    got_f, _ = features.load_cached_features(
        cache, "dinov3_small", "fake", 224, "train", hf_repo="dimarkov/bllarse-features"
    )
    np.testing.assert_array_equal(got_f, feats)
    assert calls["filename"] == "dinov3_small/fake_res224_train.npz"
    assert calls["repo_type"] == "dataset"
    # The download must be cached locally so the next run does not re-fetch.
    assert (cache / "dinov3_small_fake_res224_train.npz").exists()


class _HubHTTPError(Exception):
    """Stands in for huggingface_hub's HfHubHTTPError, which carries .response."""

    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.response = SimpleNamespace(status_code=status)


def _raising(exc):
    def fake_download(repo_id, filename, repo_type):
        raise exc

    return fake_download


def test_404_on_the_hub_means_the_features_were_never_generated(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "_hf_hub_download", _raising(_HubHTTPError(404)))
    with pytest.raises(FileNotFoundError, match="precompute_features"):
        features.load_cached_features(
            tmp_path, "dinov3_small", "fake", 224, "train", hf_repo="r/f"
        )


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failure_does_not_tell_the_user_to_regenerate(tmp_path, monkeypatch, status):
    """A bad token against a private repo must not send someone to burn GPU
    hours regenerating features that are sitting on the Hub."""
    monkeypatch.setattr(features, "_hf_hub_download", _raising(_HubHTTPError(status)))
    with pytest.raises(RuntimeError, match="HF_TOKEN") as excinfo:
        features.load_cached_features(
            tmp_path, "dinov3_small", "fake", 224, "train", hf_repo="r/f"
        )
    assert "precompute_features" not in str(excinfo.value)


def test_connection_failure_does_not_tell_the_user_to_regenerate(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "_hf_hub_download", _raising(ConnectionError("reset")))
    with pytest.raises(RuntimeError, match="network or offline") as excinfo:
        features.load_cached_features(
            tmp_path, "dinov3_small", "fake", 224, "train", hf_repo="r/f"
        )
    assert "precompute_features" not in str(excinfo.value)


def test_classification_matches_the_real_huggingface_hub_hierarchy(tmp_path, monkeypatch):
    """Guard against the library's shape changing under us.

    The previous version of this module matched exception CLASS NAMES and got it
    wrong in both directions: the real 404 is RemoteEntryNotFoundError (plain
    EntryNotFoundError is a bare base class), and GatedRepoError -- an auth
    failure -- subclasses RepositoryNotFoundError. This test constructs the
    genuine article rather than a stand-in.
    """
    errors = pytest.importorskip("huggingface_hub.errors")

    class _Req:
        method, url = "GET", "https://huggingface.co/x"

    def _resp(status):
        return SimpleNamespace(status_code=status, headers={}, text="", request=_Req())

    real_404 = errors.RemoteEntryNotFoundError("nope", response=_resp(404))
    monkeypatch.setattr(features, "_hf_hub_download", _raising(real_404))
    with pytest.raises(FileNotFoundError, match="precompute_features"):
        features.load_cached_features(
            tmp_path, "dinov3_small", "fake", 224, "train", hf_repo="r/f"
        )

    real_gated = errors.GatedRepoError("gated", response=_resp(403))
    monkeypatch.setattr(features, "_hf_hub_download", _raising(real_gated))
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        features.load_cached_features(
            tmp_path, "dinov3_small", "fake", 224, "train", hf_repo="r/f"
        )


def test_module_imports_without_jax_or_torch():
    """The torch container must import this without the JAX stack, and vice
    versa. A heavy import chain here would defeat the point of the module."""
    code = (
        "import sys, bllarse.features; "
        "heavy = [m for m in ('jax', 'torch', 'equimo', 'mlpox', 'datasets') "
        "if m in sys.modules]; "
        "assert not heavy, heavy; print('clean')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/test_features.py -v
```

Expected: collection error — no module `bllarse.features`.

- [ ] **Step 3: Implement `src/bllarse/features.py`**

```python
"""Loading cached backbone features.

Both probing arms read the same .npz files that precompute_features.py writes,
so the backbone is byte-identical across frameworks and Figs 1-4 carry no
cross-framework parity risk.

Deliberately free of JAX and PyTorch. The PyTorch container must import this
without the JAX stack, and the JAX arm must not pull equimo/mlpox/datasets in
just to call np.load. `huggingface_hub` is imported lazily for the same reason.
"""

import shutil
from pathlib import Path

import numpy as np

def _hub_status(exc: BaseException) -> int | None:
    """HTTP status behind a Hub failure, or None if it was not an HTTP error.

    Read by duck-typing on `.response.status_code` rather than by catching
    huggingface_hub's exception classes. Two reasons: importing them would break
    the lazy-import contract this module exists to satisfy, and the class names
    do not mean what they look like -- the 404 for an absent file is
    RemoteEntryNotFoundError (EntryNotFoundError is a bare base class), and
    GatedRepoError subclasses RepositoryNotFoundError, so name-matching
    'not found' silently swallows auth failures. The status code does not lie.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def feature_cache_path(cache_dir, model: str, dataset: str, img_size: int, split: str) -> Path:
    """Local cache path, matching precompute_features.py's layout exactly."""
    return Path(cache_dir) / f"{model}_{dataset}_res{img_size}_{split}.npz"


def feature_hf_path(model: str, dataset: str, img_size: int, split: str) -> str:
    """Path within the HF dataset repo. Note the model is a directory there,
    where it is a filename prefix locally."""
    return f"{model}/{dataset}_res{img_size}_{split}.npz"


def _hf_hub_download(repo_id: str, filename: str, repo_type: str) -> str:
    """Indirection so tests can substitute the download without network access."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)


def load_cached_features(
    cache_dir, model: str, dataset: str, img_size: int, split: str, hf_repo: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (features, labels) from the local cache, falling back to HF Hub.

    A successful HF fetch is copied into the local cache, so a sweep pays the
    download once per node rather than once per task.
    """
    local = feature_cache_path(cache_dir, model, dataset, img_size, split)
    if local.exists():
        data = np.load(local)
        return data["features"], data["labels"]

    if hf_repo:
        remote = feature_hf_path(model, dataset, img_size, split)
        try:
            downloaded = _hf_hub_download(repo_id=hf_repo, filename=remote, repo_type="dataset")
        except Exception as exc:
            status = _hub_status(exc)
            if status == 404:
                raise FileNotFoundError(
                    f"Features not in the local cache ({local}) and "
                    f"{hf_repo}/{remote} does not exist on the Hub. Run "
                    f"precompute_features.py --models {model} --datasets {dataset} "
                    f"to create them."
                ) from exc
            if status in (401, 403):
                raise RuntimeError(
                    f"Features not in the local cache ({local}) and the Hub "
                    f"refused access to {hf_repo}/{remote} (HTTP {status}). Check "
                    f"HF_TOKEN and that you have access to the repo. Do NOT "
                    f"regenerate -- the data may well be there."
                ) from exc
            raise RuntimeError(
                f"Features not in the local cache ({local}) and fetching "
                f"{hf_repo}/{remote} failed with {type(exc).__name__}: {exc}. "
                f"That is a network or offline problem, not missing data -- do "
                f"NOT regenerate before checking connectivity."
            ) from exc
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, local)
        data = np.load(local)
        return data["features"], data["labels"]

    raise FileNotFoundError(
        f"Features not found at {local} and no hf_repo was given. Run "
        f"precompute_features.py --models {model} --datasets {dataset} first."
    )
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_features.py -v
```

Expected: 11 passed. If `test_module_imports_without_jax_or_torch` fails, something
in `bllarse/__init__.py` is importing the heavy stack — check it and make that
import lazy rather than weakening the test.

- [ ] **Step 5: Commit**

```bash
git add src/bllarse/features.py tests/test_features.py
git commit -m "feat(features): framework-free loader for the cached feature files

Both probing arms read the same .npz files, so the loader must import from
the torch container as well as the JAX one. ood_evaluation's version reaches
vit_classification and drags in mlpox/datasets/PIL/jax for what is an
np.load with an HF fallback."
```

---

## Task 7: `bllarse.analysis.thresholds` — the joint threshold protocol

**Files:**
- Create: `src/bllarse/analysis/__init__.py`
- Create: `src/bllarse/analysis/thresholds.py`
- Test: `tests/analysis/test_thresholds.py`

**Interfaces:**
- Consumes: nothing at runtime — operates on metric curves, not dumps, so it is
  testable without any training.
- Produces:
  - `Targets` dataclass with fields `acc`, `nll`, `brier`
  - `make_targets(baseline_curve, fraction=0.95) -> Targets`
  - `first_crossing(curve, targets) -> Crossing` — dataclass with `step`, `censored`
  - `crossings_per_seed(curves_by_seed, targets) -> dict[int, Crossing]`

A `curve` is a list of dicts, each with keys `step`, `acc`, `nll`, `brier`.

Implements `results-plan.md` §0.4 verbatim. Cannot be exercised end-to-end until
Phase 1 produces a baseline, so it ships with synthetic-curve tests covering the
four cases the protocol exists to handle.

- [ ] **Step 1: Write the failing tests**

Create `tests/analysis/test_thresholds.py`:

```python
import pytest

from bllarse.analysis import thresholds


def _curve(rows):
    return [dict(step=s, acc=a, nll=n, brier=b) for s, a, n, b in rows]


def test_targets_are_a_fraction_of_converged_baseline():
    baseline = _curve([(1, 0.10, 3.0, 1.2), (10, 0.80, 0.5, 0.30)])
    t = thresholds.make_targets(baseline, fraction=0.95)
    assert t.acc == pytest.approx(0.76)      # 95% of 0.80
    assert t.nll == pytest.approx(0.5 / 0.95)  # NLL is lower-is-better
    assert t.brier == pytest.approx(0.30 / 0.95)


def test_crossing_requires_all_three_metrics_jointly():
    targets = thresholds.Targets(acc=0.76, nll=0.53, brier=0.32)
    # Accuracy clears at step 2 but calibration does not until step 4.
    curve = _curve([
        (1, 0.10, 3.00, 1.20),
        (2, 0.80, 2.00, 0.90),   # acc only -- must NOT count
        (3, 0.81, 0.60, 0.40),   # still short on nll and brier
        (4, 0.82, 0.50, 0.30),   # all three
    ])
    assert thresholds.first_crossing(curve, targets).step == 4


def test_accuracy_early_calibration_late_does_not_score_as_fast():
    """The exact failure mode results-plan 0.4 exists to prevent."""
    targets = thresholds.Targets(acc=0.76, nll=0.53, brier=0.32)
    fast_but_miscalibrated = _curve([(1, 0.90, 5.0, 1.5), (2, 0.91, 4.0, 1.4)])
    result = thresholds.first_crossing(fast_but_miscalibrated, targets)
    assert result.censored is True
    assert result.step is None


def test_non_crossing_run_is_right_censored_not_dropped():
    targets = thresholds.Targets(acc=0.76, nll=0.53, brier=0.32)
    curve = _curve([(1, 0.10, 3.0, 1.2), (10, 0.50, 1.0, 0.8)])
    result = thresholds.first_crossing(curve, targets)
    assert result.censored is True
    assert result.step is None
    assert result.last_step == 10


def test_non_monotone_curve_takes_the_first_crossing_not_the_last():
    targets = thresholds.Targets(acc=0.76, nll=0.53, brier=0.32)
    curve = _curve([
        (1, 0.80, 0.50, 0.30),   # crosses here
        (2, 0.50, 2.00, 1.00),   # regresses
        (3, 0.85, 0.40, 0.25),   # crosses again
    ])
    assert thresholds.first_crossing(curve, targets).step == 1


def test_crossings_are_reported_per_seed_not_from_the_mean_curve():
    targets = thresholds.Targets(acc=0.76, nll=0.53, brier=0.32)
    curves = {
        0: _curve([(1, 0.80, 0.50, 0.30)]),
        1: _curve([(1, 0.10, 3.00, 1.20), (2, 0.80, 0.50, 0.30)]),
    }
    out = thresholds.crossings_per_seed(curves, targets)
    assert out[0].step == 1
    assert out[1].step == 2


def test_make_targets_rejects_an_empty_curve():
    with pytest.raises(ValueError):
        thresholds.make_targets([], fraction=0.95)
```

Create an empty `tests/analysis/__init__.py` if the suite needs it (check whether
`tests/losses/` has one; mirror that).

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/analysis/test_thresholds.py -v
```

Expected: collection error — no module `bllarse.analysis`.

- [ ] **Step 3: Implement the module**

`src/bllarse/analysis/__init__.py`:

```python
"""Post-hoc analysis over eval dumps. Nothing here runs inside a training loop."""
```

`src/bllarse/analysis/thresholds.py`:

```python
"""The joint compute-to-threshold protocol (results-plan.md 0.4).

Fixed before looking at any results, because the researcher degrees of freedom
here are large. Four rules, all load-bearing:

1. The target is joint over accuracy, NLL and Brier. A method that reaches
   accuracy early with bad calibration must not score as fast -- that is the
   whole point of the figure.
2. Targets come from the strongest *baseline's* converged value, never from
   IBProbit.
3. Runs that never reach threshold are right-censored and reported as such,
   never dropped and never extrapolated.
4. Crossings are per seed, not the crossing of the seed-mean curve.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

Curve = Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class Targets:
    acc: float
    nll: float
    brier: float


@dataclass(frozen=True)
class Crossing:
    step: int | None
    censored: bool
    last_step: int | None


def make_targets(baseline_curve: Curve, fraction: float = 0.95) -> Targets:
    """Targets as a fraction of the baseline's converged value.

    Accuracy is higher-is-better so the target is `fraction * converged`. NLL and
    Brier are lower-is-better, so the target is `converged / fraction` -- a
    proportionally looser bound, not a tighter one.
    """
    if not baseline_curve:
        raise ValueError("baseline_curve is empty; cannot derive targets")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    converged = max(baseline_curve, key=lambda row: row["step"])
    return Targets(
        acc=converged["acc"] * fraction,
        nll=converged["nll"] / fraction,
        brier=converged["brier"] / fraction,
    )


def _meets(row: Mapping[str, Any], targets: Targets) -> bool:
    return (
        row["acc"] >= targets.acc
        and row["nll"] <= targets.nll
        and row["brier"] <= targets.brier
    )


def first_crossing(curve: Curve, targets: Targets) -> Crossing:
    """First step at which ALL THREE targets hold. Censored if never."""
    if not curve:
        return Crossing(step=None, censored=True, last_step=None)

    ordered = sorted(curve, key=lambda row: row["step"])
    for row in ordered:
        if _meets(row, targets):
            return Crossing(step=row["step"], censored=False, last_step=ordered[-1]["step"])
    return Crossing(step=None, censored=True, last_step=ordered[-1]["step"])


def crossings_per_seed(
    curves_by_seed: Mapping[int, Curve], targets: Targets
) -> dict[int, Crossing]:
    """Per-seed crossings. Deliberately not the crossing of the mean curve --
    averaging first would let one fast seed manufacture an early crossing that no
    individual run achieved."""
    return {seed: first_crossing(curve, targets) for seed, curve in curves_by_seed.items()}
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/analysis/test_thresholds.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bllarse/analysis tests/analysis
git commit -m "feat(analysis): joint compute-to-threshold protocol

Implements results-plan 0.4: joint acc/NLL/Brier targets from the strongest
baseline, per-seed crossings, explicit right-censoring. Pure post-hoc --
nothing runs in a training loop."
```

---

## Task 8: Register SVHN and precompute its features

**Files:**
- Modify: `scripts/alternatives/vit_classification.py` (`DATASET_CONFIGS`, `DINOV3_DATASET_IMG_SIZES`)
- Test: `tests/test_dataset_registry.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `DATASET_CONFIGS["svhn"]` and `DINOV3_DATASET_IMG_SIZES["svhn"] == 224`,
  consumed by Task 9's OOD path and Task 12's sweep definitions.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset_registry.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "alternatives"))

import vit_classification as vc


def test_svhn_is_registered():
    assert "svhn" in vc.DATASET_CONFIGS
    cfg = vc.DATASET_CONFIGS["svhn"]
    assert cfg["num_classes"] == 10
    assert {"hf_path", "label_key", "img_key", "test_split"} <= set(cfg)


def test_svhn_resolution_matches_cifar():
    """OOD features must be extracted at the same resolution as the ID features
    or they do not live in the same feature space."""
    assert vc.get_img_size("dinov3_huge", "svhn") == vc.get_img_size("dinov3_huge", "cifar10")
    assert vc.get_img_size("dinov3_huge", "svhn") == 224


def test_fig2_id_and_ood_datasets_all_share_a_resolution():
    sizes = {vc.get_img_size("dinov3_large", d) for d in ("cifar10", "cifar100", "svhn")}
    assert len(sizes) == 1
```

- [ ] **Step 2: Run to confirm it fails**

```bash
uv run pytest tests/test_dataset_registry.py -v
```

Expected: FAIL — `'svhn' not in DATASET_CONFIGS`.

- [ ] **Step 3: Register SVHN**

In `scripts/alternatives/vit_classification.py`, add to `DINOV3_DATASET_IMG_SIZES`
(next to the CIFAR entries):

```python
    "svhn":          224,
```

and to `DATASET_CONFIGS`:

```python
    "svhn": {
        "hf_path": "ufldl-stanford/svhn",
        "hf_config": "cropped_digits",
        "num_classes": 10,
        "label_key": "label",
        "img_key": "image",
        "test_split": "test",
    },
```

- [ ] **Step 4: Verify the HF keys are actually right**

The registry is not uniform across sources — `cifar100` uses `fine_label`,
`cifar10` uses `img` rather than `image`. Confirm rather than assume:

```bash
uv run python -c "
from datasets import load_dataset
ds = load_dataset('ufldl-stanford/svhn', 'cropped_digits', split='test', streaming=True)
row = next(iter(ds))
print('keys:', list(row.keys()))
print('label type:', type(row['label']), row['label'])
print('image:', row['image'])
"
```

If the printed keys differ from what you registered, fix the registry entry to
match and re-run Step 3. If `load_or_compute_features` needs an `hf_config`
argument that it does not currently accept, add it — check how it calls
`load_dataset` first.

- [ ] **Step 5: Run the test**

```bash
uv run pytest tests/test_dataset_registry.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Precompute SVHN features for the four ladder backbones**

This is a GPU job and will take a while. Run it once; the results land in
`.cache/features/` and mirror to HF.

```bash
uv run python scripts/alternatives/precompute_features.py \
    --models dinov3_small dinov3_big dinov3_large dinov3_huge \
    --datasets svhn
```

Then confirm the CIFAR features exist too (they should already):

```bash
ls -la .cache/features/ | grep -E "svhn|cifar" | head -20
```

Expected: `{model}_svhn_res224_{split}.npz` for all four models, alongside the
CIFAR files.

- [ ] **Step 7: Commit**

```bash
git add scripts/alternatives/vit_classification.py tests/test_dataset_registry.py
git commit -m "feat(data): register SVHN at CIFAR resolution for Fig 2

Fig 2 needs CIFAR-10/CIFAR-100 as in-distribution against SVHN (far-OOD) and
CIFAR-100 (near-OOD). SVHN must be extracted at 224 like CIFAR or the
features do not share a space."
```

---

## Task 9: JAX probing entry point with dump wiring

**Files:**
- Create: `scripts/alternatives/probe.py`
- Test: `tests/test_probe_smoke.py`

**Interfaces:**
- Consumes: `bllarse.metrics`, `bllarse.splits`, `bllarse.evaldump`,
  `bllarse.features.load_cached_features` (Task 6). Nothing from
  `scripts/alternatives/` — see Task 6 on why that import chain is avoided.
- Produces: a CLI with `build_argparser()` and `main(args)`, callable through
  `bllarse.tools.adapters.run_script_from_config("scripts/alternatives/probe.py", cfg)`.

This is a new focused entry point rather than an extension of
`vit_classification.py` (800+ lines, also does feature extraction and MLflow) or
`ood_evaluation.py` (ImageNet-specific). It loads cached features, trains one
head, and dumps.

- [ ] **Step 1: Write the failing test**

Create `tests/test_probe_smoke.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pytest

ALT = Path(__file__).resolve().parents[1] / "scripts" / "alternatives"
sys.path.insert(0, str(ALT))

from bllarse import evaldump


@pytest.fixture
def fake_features(tmp_path):
    """Two well-separated Gaussian blobs -- a linear head must reach high accuracy."""
    rng = np.random.default_rng(0)
    cache = tmp_path / "features"
    cache.mkdir()
    for split, n in (("train", 600), ("test", 200)):
        labels = rng.integers(0, 2, size=n)
        feats = rng.normal(size=(n, 16)).astype(np.float32)
        feats[labels == 1, 0] += 6.0
        np.savez(cache / f"dinov3_small_fake_res224_{split}.npz",
                 features=feats, labels=labels)
    return cache


def test_probe_writes_dumps_and_learns(fake_features, tmp_path):
    import probe

    args = probe.build_argparser().parse_args([
        "--arm", "adamw_ce",
        "--model", "dinov3_small",
        "--dataset", "fake",
        "--num-classes", "2",
        "--cache-dir", str(fake_features),
        "--split-cache-dir", str(tmp_path / "splits"),
        "--dump-root", str(tmp_path / "dumps"),
        "--group-id", "smoke",
        "--uid", "smoke_0",
        "--epochs", "8",
        "--n-dumps", "5",
        "--seed", "137",
        "--no-hf-cache",
    ])
    probe.main(args)

    records = list(evaldump.iter_dumps(tmp_path / "dumps", "smoke", "smoke_0"))
    assert len(records) >= 2, "expected several dump points"
    assert records[0].meta["arm"] == "adamw_ce"
    assert records[0].meta["logit_kind"] == "softmax_logits"
    assert records[0].logits.shape[1] == 2

    from bllarse import metrics
    final = records[-1]
    assert metrics.accuracy(final.logits, final.labels) > 0.9


def test_probe_dumps_ood_scores_when_an_ood_dataset_is_given(fake_features, tmp_path):
    import probe

    rng = np.random.default_rng(1)
    for split in ("train", "test"):
        n = 150
        np.savez(fake_features / f"dinov3_small_fakeood_res224_{split}.npz",
                 features=rng.normal(loc=20.0, size=(n, 16)).astype(np.float32),
                 labels=rng.integers(0, 2, size=n))

    args = probe.build_argparser().parse_args([
        "--arm", "adamw_ce",
        "--model", "dinov3_small",
        "--dataset", "fake",
        "--ood-dataset", "fakeood",
        "--num-classes", "2",
        "--cache-dir", str(fake_features),
        "--split-cache-dir", str(tmp_path / "splits"),
        "--dump-root", str(tmp_path / "dumps"),
        "--group-id", "smoke",
        "--uid", "smoke_ood",
        "--epochs", "4",
        "--n-dumps", "3",
        "--seed", "137",
        "--no-hf-cache",
    ])
    probe.main(args)

    ood = [r for r in evaldump.iter_dumps(tmp_path / "dumps", "smoke", "smoke_ood")
           if r.meta["split"] == "ood"]
    assert len(ood) == 1, "OOD is dumped at the final point only"
```

- [ ] **Step 2: Run to confirm it fails**

```bash
uv run pytest tests/test_probe_smoke.py -v
```

Expected: `ModuleNotFoundError: No module named 'probe'`.

- [ ] **Step 3: Implement `scripts/alternatives/probe.py`**

```python
#!/usr/bin/env python3
"""Linear probing on cached DINOv3 features, with logit dumps.

One head, one dataset, one seed. Loads features that precompute_features.py has
already extracted, trains a head, and writes a dump at each scheduled step. All
reported metrics are computed later from those dumps -- nothing here decides
what the figures show.

Arms: adamw_ce now; ibprobit is wired the same way and enters in Phase 3.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax
from jax import random as jr, vmap

from bllarse import evaldump, metrics, splits
from bllarse.features import load_cached_features
from bllarse.losses import CrossEntropy


def _ce_epoch(head, opt_state, optimizer, loss_fn, feats, labels, batch_size, key):
    """One AdamW pass over the features."""
    perm = jr.permutation(key, len(feats))
    n_batches = len(feats) // batch_size
    for i in range(n_batches):
        idx = perm[i * batch_size:(i + 1) * batch_size]
        fb, yb = jnp.asarray(feats[idx]), jnp.asarray(labels[idx])

        def loss(h):
            return loss_fn(vmap(h)(fb), yb).mean()

        grads = eqx.filter_grad(loss)(head)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(head, eqx.is_array)
        )
        head = eqx.apply_updates(head, updates)
    return head, opt_state


def _logits_for(head, loss_fn, feats, labels, chunk=4096):
    out = []
    for start in range(0, len(feats), chunk):
        fb = jnp.asarray(feats[start:start + chunk])
        yb = jnp.asarray(labels[start:start + chunk])
        _, logits = loss_fn(vmap(head)(fb), yb, with_logits=True)
        out.append(np.asarray(logits))
    return np.concatenate(out, axis=0)


def main(args):
    key = jr.PRNGKey(args.seed)
    hf_repo = None if args.no_hf_cache else args.hf_repo
    img_size = args.img_size

    train_f, train_l = load_cached_features(
        args.cache_dir, args.model, args.dataset, img_size, "train", hf_repo=hf_repo
    )
    eval_f, eval_l = load_cached_features(
        args.cache_dir, args.model, args.dataset, img_size, "test", hf_repo=hf_repo
    )

    # Pinned split: val is for post-hoc temperature scaling, test for reporting.
    # Seed-independent by construction (bllarse.splits).
    val_idx, test_idx = splits.load_or_make_pinned_split(
        args.split_cache_dir, args.dataset, "test", len(eval_f), args.val_fraction
    )

    embed_dim = train_f.shape[-1]
    key, head_key = jr.split(key)

    if args.arm != "adamw_ce":
        raise NotImplementedError(
            f"arm '{args.arm}' is not wired yet; adamw_ce only in Phase 1"
        )

    head = eqx.nn.Linear(embed_dim, args.num_classes, key=head_key)
    loss_fn = CrossEntropy(args.label_smooth, args.num_classes)
    optimizer = optax.adamw(args.learning_rate, weight_decay=args.weight_decay)
    opt_state = optimizer.init(eqx.filter(head, eqx.is_array))

    schedule = set(evaldump.dump_schedule(args.epochs, n_dumps=args.n_dumps).tolist())
    print(f"[probe] {args.uid}: {args.epochs} epochs, dumping at {sorted(schedule)}")

    for epoch in range(1, args.epochs + 1):
        key, epoch_key = jr.split(key)
        head, opt_state = _ce_epoch(
            head, opt_state, optimizer, loss_fn,
            train_f, train_l, args.batch_size, epoch_key,
        )
        if epoch not in schedule:
            continue

        logits = _logits_for(head, loss_fn, eval_f, eval_l)
        for split_name, idx in (("val", val_idx), ("test", test_idx)):
            evaldump.write_dump(
                args.dump_root, args.group_id, args.uid, epoch,
                logits[idx], eval_l[idx],
                arm=args.arm, split=split_name, logit_kind="softmax_logits",
                seed=args.seed, samples_seen=epoch * len(train_f),
            )
        scalars = metrics.classification_metrics(logits[test_idx], eval_l[test_idx])
        print(f"[probe] epoch {epoch}: " + " ".join(f"{k}={v:.4f}" for k, v in scalars.items()))

    # OOD dump: final point only. Fig 2 is a converged-endpoint figure.
    if args.ood_dataset:
        ood_f, ood_l = load_cached_features(
            args.cache_dir, args.model, args.ood_dataset, img_size, "test", hf_repo=hf_repo
        )
        ood_logits = _logits_for(head, loss_fn, ood_f, ood_l)
        evaldump.write_dump(
            args.dump_root, args.group_id, args.uid, args.epochs,
            ood_logits, ood_l,
            arm=args.arm, split="ood", logit_kind="softmax_logits",
            seed=args.seed, samples_seen=args.epochs * len(train_f),
        )
        print(f"[probe] wrote OOD dump for {args.ood_dataset} ({len(ood_f)} samples)")


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", default="adamw_ce", choices=["adamw_ce", "ibprobit"])
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--ood-dataset", default=None)
    p.add_argument("--num-classes", type=int, required=True)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--cache-dir", default=".cache/features")
    p.add_argument("--split-cache-dir", default=".cache/splits")
    p.add_argument("--dump-root", default=os.environ.get("BLLARSE_EVAL_DUMP_ROOT", "eval_dumps"))
    p.add_argument("--hf-repo", default="dimarkov/bllarse-features")
    p.add_argument("--no-hf-cache", action="store_true")
    p.add_argument("--group-id", default="probe")
    p.add_argument("--uid", default="probe_0")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--n-dumps", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smooth", type=float, default=0.0)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=137)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_probe_smoke.py -v
```

Expected: 2 passed. If the head does not reach 0.9 accuracy, raise `--epochs` in
the test rather than loosening the assertion — the blobs are separable by
construction and a failure here means the training loop is wrong.

- [ ] **Step 5: Commit**

```bash
git add scripts/alternatives/probe.py tests/test_probe_smoke.py
git commit -m "feat(probe): JAX probing entry point writing logit dumps

Loads cached features, trains one head, dumps at scheduled steps on the
pinned val/test partition. OOD dumped at the final point only, since Fig 2
is a converged-endpoint figure."
```

---

## Task 10: PyTorch VBLL probing entry point

**Files:**
- Create: `scripts/vbll_pytorch/probing_vbll.py`
- Modify: `scripts/vbll_pytorch/src/vbll_experiments/models.py`
- Test: `tests/test_probing_vbll_smoke.py`

**Interfaces:**
- Consumes: `bllarse.evaldump`, `bllarse.splits`,
  `bllarse.features.load_cached_features` (Task 6); the `vbll` package.
- Produces: `build_argparser()` / `main(args)` with the same dump contract as Task 9.

The key simplification: VBLL reads the **same `.npz` feature files** the JAX arm
does, so no PyTorch DINOv3 is needed here and the backbone is byte-identical
across arms. `models.py` currently raises at import if the `scaling_mlps`
submodule is absent — that must become conditional.

- [ ] **Step 1: Write the failing test**

Create `tests/test_probing_vbll_smoke.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vbll")

VBLL_DIR = Path(__file__).resolve().parents[1] / "scripts" / "vbll_pytorch"
sys.path.insert(0, str(VBLL_DIR))

from bllarse import evaldump


@pytest.fixture
def fake_features(tmp_path):
    rng = np.random.default_rng(0)
    cache = tmp_path / "features"
    cache.mkdir()
    for split, n in (("train", 600), ("test", 200)):
        labels = rng.integers(0, 2, size=n)
        feats = rng.normal(size=(n, 16)).astype(np.float32)
        feats[labels == 1, 0] += 6.0
        np.savez(cache / f"dinov3_small_fake_res224_{split}.npz",
                 features=feats, labels=labels)
    return cache


def test_models_import_without_scaling_mlps_submodule():
    """The head-only path must not require the deep-MLP submodule, which belongs
    to the excised experiment."""
    from vbll_experiments import models
    assert hasattr(models, "build_vbll_head")


def test_vbll_probe_writes_dumps_and_learns(fake_features, tmp_path):
    import probing_vbll

    args = probing_vbll.build_argparser().parse_args([
        "--model", "dinov3_small",
        "--dataset", "fake",
        "--num-classes", "2",
        "--cache-dir", str(fake_features),
        "--split-cache-dir", str(tmp_path / "splits"),
        "--dump-root", str(tmp_path / "dumps"),
        "--group-id", "smoke",
        "--uid", "vbll_0",
        "--epochs", "20",
        "--n-dumps", "5",
        "--seed", "137",
        "--device", "cpu",
    ])
    probing_vbll.main(args)

    records = list(evaldump.iter_dumps(tmp_path / "dumps", "smoke", "vbll_0"))
    assert len(records) >= 2
    assert records[0].meta["arm"] == "vbll"
    assert records[0].meta["logit_kind"] == "softmax_logits"
    assert records[0].meta["score_kind"] == "predictive_variance"
    assert records[0].score is not None, "VBLL must carry its native epistemic score"

    from bllarse import metrics
    final = [r for r in records if r.meta["split"] == "test"][-1]
    assert metrics.accuracy(final.logits, final.labels) > 0.9
```

- [ ] **Step 2: Run to confirm it fails**

```bash
uv run pytest tests/test_probing_vbll_smoke.py -v
```

Expected: FAIL — `models` has no `build_vbll_head`, and no module `probing_vbll`.

- [ ] **Step 3: Make the `scaling_mlps` import conditional and add a head factory**

In `scripts/vbll_pytorch/src/vbll_experiments/models.py`, replace the
unconditional submodule check at import time with a lazy one. The current code
raises `FileNotFoundError` at module scope; move that check inside the function
that actually needs the backbone, then add:

```python
def build_vbll_head(
    input_dim: int,
    num_classes: int,
    *,
    regularization_weight: float = 1.0,
    prior_scale: float = 1.0,
    wishart_scale: float = 0.1,
    parameterization: str = "diagonal",
    vbll_type: str = "discriminative",
):
    """A bare VBLL classification head, with no backbone.

    Linear probing on cached features needs only this -- the features were
    already extracted by the JAX side, so there is no PyTorch backbone in the
    probing path at all and the two arms consume byte-identical inputs.
    """
    import vbll

    vbll_class = (
        vbll.DiscClassification if vbll_type == "discriminative" else vbll.GenClassification
    )
    return vbll_class(
        in_features=input_dim,
        out_features=num_classes,
        regularization_weight=regularization_weight,
        parameterization=parameterization,
        prior_scale=prior_scale,
        wishart_scale=wishart_scale,
        return_ood=True,
    )
```

- [ ] **Step 4: Implement `scripts/vbll_pytorch/probing_vbll.py`**

```python
#!/usr/bin/env python3
"""VBLL linear probing on cached DINOv3 features.

Reads the same .npz feature files the JAX arm reads, so the backbone is
byte-identical across arms and Figs 1-4 carry no cross-framework parity risk.
Writes dumps under the shared contract (bllarse.evaldump).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bllarse import evaldump, splits
from bllarse.features import load_cached_features
from vbll_experiments.models import build_vbll_head


def _forward(head, feats, device, chunk=4096):
    """Return (logits, epistemic_score) as numpy arrays."""
    head.eval()
    logits_out, score_out = [], []
    with torch.no_grad():
        for start in range(0, len(feats), chunk):
            x = torch.from_numpy(feats[start:start + chunk]).float().to(device)
            out = head(x)
            logits_out.append(out.predictive.logits.cpu().numpy())
            # VBLL's own predictive variance, summed over classes: the native
            # epistemic score. Diagnostic only -- the primary Fig 2 comparison
            # uses MSP/entropy from the logits, which every arm has.
            score_out.append(out.predictive.covariance.sum(dim=-1).cpu().numpy())
    return np.concatenate(logits_out), np.concatenate(score_out)


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_f, train_l = load_cached_features(
        args.cache_dir, args.model, args.dataset, args.img_size, "train"
    )
    eval_f, eval_l = load_cached_features(
        args.cache_dir, args.model, args.dataset, args.img_size, "test"
    )
    val_idx, test_idx = splits.load_or_make_pinned_split(
        args.split_cache_dir, args.dataset, "test", len(eval_f), args.val_fraction
    )

    head = build_vbll_head(
        train_f.shape[-1], args.num_classes,
        regularization_weight=1.0 / len(train_f),
        prior_scale=args.prior_scale,
        wishart_scale=args.wishart_scale,
        parameterization=args.parameterization,
        vbll_type=args.vbll_type,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    x_train = torch.from_numpy(train_f).float()
    y_train = torch.from_numpy(train_l).long()
    schedule = set(evaldump.dump_schedule(args.epochs, n_dumps=args.n_dumps).tolist())
    print(f"[vbll] {args.uid}: {args.epochs} epochs, dumping at {sorted(schedule)}")

    for epoch in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(len(x_train))
        for i in range(len(x_train) // args.batch_size):
            idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            xb, yb = x_train[idx].to(device), y_train[idx].to(device)
            optimizer.zero_grad()
            head(xb).train_loss_fn(yb).backward()
            optimizer.step()

        if epoch not in schedule:
            continue

        logits, score = _forward(head, eval_f, device)
        for split_name, idx in (("val", val_idx), ("test", test_idx)):
            evaldump.write_dump(
                args.dump_root, args.group_id, args.uid, epoch,
                logits[idx], eval_l[idx],
                arm="vbll", split=split_name, logit_kind="softmax_logits",
                seed=args.seed, samples_seen=epoch * len(train_f),
                score=score[idx], score_kind="predictive_variance",
            )
        print(f"[vbll] epoch {epoch}: dumped {len(val_idx)} val / {len(test_idx)} test")

    if args.ood_dataset:
        ood_f, ood_l = load_cached_features(
            args.cache_dir, args.model, args.ood_dataset, args.img_size, "test"
        )
        ood_logits, ood_score = _forward(head, ood_f, device)
        evaldump.write_dump(
            args.dump_root, args.group_id, args.uid, args.epochs,
            ood_logits, ood_l,
            arm="vbll", split="ood", logit_kind="softmax_logits",
            seed=args.seed, samples_seen=args.epochs * len(train_f),
            score=ood_score, score_kind="predictive_variance",
        )
        print(f"[vbll] wrote OOD dump for {args.ood_dataset} ({len(ood_f)} samples)")


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--ood-dataset", default=None)
    p.add_argument("--num-classes", type=int, required=True)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--cache-dir", default=".cache/features")
    p.add_argument("--split-cache-dir", default=".cache/splits")
    p.add_argument("--dump-root", default=os.environ.get("BLLARSE_EVAL_DUMP_ROOT", "eval_dumps"))
    p.add_argument("--group-id", default="probe_vbll")
    p.add_argument("--uid", default="vbll_0")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--n-dumps", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--prior-scale", type=float, default=1.0)
    p.add_argument("--wishart-scale", type=float, default=0.1)
    p.add_argument("--parameterization", default="diagonal")
    p.add_argument("--vbll-type", default="discriminative")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=137)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
```

- [ ] **Step 5: Run the tests**

```bash
uv run pytest tests/test_probing_vbll_smoke.py -v
```

Expected: 2 passed, or skipped if `torch`/`vbll` are not installed in the active
env. If skipped, install them (`uv sync --group vbll`) and re-run — this test
must actually execute before the task is done.

**If the VBLL output attribute names differ** (`out.predictive.logits`,
`out.predictive.covariance`, `out.train_loss_fn`), inspect the installed version
and fix the two call sites:

```bash
uv run python -c "
import torch, vbll
h = vbll.DiscClassification(4, 3, regularization_weight=1.0, parameterization='diagonal', return_ood=True)
out = h(torch.randn(2, 4))
print(type(out), [a for a in dir(out) if not a.startswith('_')])
print([a for a in dir(out.predictive) if not a.startswith('_')])
"
```

- [ ] **Step 6: Commit**

```bash
git add scripts/vbll_pytorch/probing_vbll.py \
        scripts/vbll_pytorch/src/vbll_experiments/models.py \
        tests/test_probing_vbll_smoke.py
git commit -m "feat(vbll): probing entry point on cached features

VBLL reads the same .npz features the JAX arm reads, so Figs 1-4 have zero
cross-framework parity risk and no PyTorch DINOv3 is needed. Makes the
scaling_mlps submodule import lazy -- it belongs to the excised experiment."
```

---

## Task 11: Apptainer SLURM job script

**Files:**
- Create: `src/slurm/jobs/slurm_run_config_apptainer.sh`
- Modify: `src/bllarse/tools/run_sweep.py` (`DEFAULT_JOB_SCRIPT`)
- Test: `tests/tools/test_run_sweep_apptainer.py`

**Interfaces:**
- Consumes: the sweep contract from `run_sweep.py` / `run_config.py`.
- Produces: a job script selected by `BLLARSE_SIF`, and a changed default in
  `run_sweep.py` that Task 12's sweeps rely on.

- [ ] **Step 1: Check what the cluster actually provides**

Before writing anything, on the cluster login node:

```bash
which apptainer singularity
apptainer --version
scontrol show config | grep -i MaxArraySize
echo "$SCRATCH"
```

Record `MaxArraySize` — Task 12 submits ~1600 tasks and may need chunking via
`--index-offset` / `--num-jobs`. If `apptainer` is absent but `singularity` is
present, use `singularity` in the script; the CLI is compatible for `exec --nv`.

- [ ] **Step 2: Write the failing test**

Create `tests/tools/test_run_sweep_apptainer.py`:

```python
from pathlib import Path

from bllarse.tools import run_sweep


def test_default_job_script_is_the_apptainer_one():
    assert "apptainer" in str(run_sweep.DEFAULT_JOB_SCRIPT)


def test_apptainer_job_script_exists_and_is_executable():
    path = Path("src/slurm/jobs/slurm_run_config_apptainer.sh")
    assert path.is_file()
    assert path.stat().st_mode & 0o111, "job script must be executable"


def test_apptainer_job_script_uses_nv_and_honours_the_sif_variable():
    text = Path("src/slurm/jobs/slurm_run_config_apptainer.sh").read_text()
    assert "--nv" in text, "GPU passthrough requires --nv"
    assert "BLLARSE_SIF" in text
    assert "BLLARSE_HF_HOME" in text, "HF cache path must be overridable on the cluster"
```

- [ ] **Step 3: Run to confirm it fails**

```bash
uv run pytest tests/tools/test_run_sweep_apptainer.py -v
```

Expected: FAIL on all three.

- [ ] **Step 4: Write the job script**

Create `src/slurm/jobs/slurm_run_config_apptainer.sh`:

```bash
#!/usr/bin/env bash
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH -G 1
#SBATCH --output=slurm_logs/%x.%A_%a.out
#SBATCH --error=slurm_logs/%x.%A_%a.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$repo_root"

export BLLARSE_REPO_ROOT="$repo_root"
export INDEX_OFFSET="${INDEX_OFFSET:-0}"
export CONFIG_IDX="$((SLURM_ARRAY_TASK_ID + INDEX_OFFSET))"

sif="${BLLARSE_SIF:-}"
if [[ -z "$sif" ]]; then
  echo "[bllarse] ERROR: BLLARSE_SIF is not set; point it at the .sif to run." >&2
  exit 2
fi
if [[ ! -f "$sif" ]]; then
  echo "[bllarse] ERROR: SIF image not found: $sif" >&2
  exit 2
fi

echo "[bllarse] task=${SLURM_ARRAY_TASK_ID} offset=${INDEX_OFFSET} config=${CONFIG_IDX} sif=${sif}"

# Apptainer binds $HOME, $PWD and /tmp by default and preserves UID/GID, so the
# Docker wrapper's -u and passthrough gymnastics are unnecessary. Only paths
# outside those roots need an explicit bind.
bind_args=()
if [[ -n "${BLLARSE_HF_HOME:-}" ]]; then
  mkdir -p "$BLLARSE_HF_HOME"
  bind_args+=(--bind "$BLLARSE_HF_HOME:$BLLARSE_HF_HOME")
  export HF_HOME="$BLLARSE_HF_HOME"
fi
if [[ -n "${BLLARSE_EVAL_DUMP_ROOT:-}" ]]; then
  mkdir -p "$BLLARSE_EVAL_DUMP_ROOT"
  bind_args+=(--bind "$BLLARSE_EVAL_DUMP_ROOT:$BLLARSE_EVAL_DUMP_ROOT")
fi

exec apptainer exec --nv \
  --env "BLLARSE_REPO_ROOT=$BLLARSE_REPO_ROOT" \
  --env "BLLARSE_SWEEP_SOURCE=$BLLARSE_SWEEP_SOURCE" \
  --env "CONFIG_IDX=$CONFIG_IDX" \
  --env "PYTHONPATH=${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  --env "HF_HOME=${HF_HOME:-}" \
  --env "BLLARSE_EVAL_DUMP_ROOT=${BLLARSE_EVAL_DUMP_ROOT:-}" \
  --env "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-}" \
  --env "MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME:-}" \
  --env "MLFLOW_TRACKING_PASSWORD=${MLFLOW_TRACKING_PASSWORD:-}" \
  --env "MLFLOW_PARENT_RUN_ID=${MLFLOW_PARENT_RUN_ID:-}" \
  --env "MLFLOW_EXPERIMENT_NAME=${MLFLOW_EXPERIMENT_NAME:-}" \
  "${bind_args[@]}" \
  "$sif" \
  python -m bllarse.tools.run_config "$BLLARSE_SWEEP_SOURCE" "$CONFIG_IDX"
```

Make it executable:

```bash
chmod +x src/slurm/jobs/slurm_run_config_apptainer.sh
```

- [ ] **Step 5: Point `run_sweep.py` at it**

In `src/bllarse/tools/run_sweep.py`:

```python
DEFAULT_JOB_SCRIPT = Path("src/slurm/jobs/slurm_run_config_apptainer.sh")
```

The Docker scripts stay in place for local multi-GPU use.

- [ ] **Step 6: Run the tests**

```bash
uv run pytest tests/tools/test_run_sweep_apptainer.py tests/tools/test_run_sweep.py -v
```

Expected: all pass. If `test_run_sweep.py` asserted the old Docker default, update
that assertion — the default genuinely changed.

- [ ] **Step 7: Build the two SIF images on the cluster**

The two images are separate because `jax[cuda]` and `torch` pin CUDA/cuDNN
differently and the two arms never share a process.

If a Docker daemon is reachable on the build host:

```bash
apptainer build bllarse-jax.sif   docker-daemon://bllarse-dev:latest
apptainer build bllarse-torch.sif docker-daemon://bllarse-torch:latest
```

Shared clusters usually have no daemon. Build from def-files instead — create
`src/slurm/images/bllarse-jax.def`:

```
Bootstrap: docker
From: nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

%post
    apt-get update && apt-get install -y python3.11 python3-pip git && rm -rf /var/lib/apt/lists/*
    pip install --no-cache-dir uv
    pip install --no-cache-dir "jax[cuda12]" equinox optax "tfp-nightly[jax]" \
        datasets equimo augmax mlflow python-dotenv tqdm \
        "mlpox @ git+https://github.com/dimarkov/mlpox.git@main" \
        "blrax @ git+https://github.com/dimarkov/blrax.git@main"

%environment
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

and `src/slurm/images/bllarse-torch.def`:

```
Bootstrap: docker
From: nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

%post
    apt-get update && apt-get install -y python3.11 python3-pip git && rm -rf /var/lib/apt/lists/*
    pip install --no-cache-dir torch torchvision "vbll>=0.2.0" mlflow numpy tqdm
```

```bash
apptainer build bllarse-jax.sif   src/slurm/images/bllarse-jax.def
apptainer build bllarse-torch.sif src/slurm/images/bllarse-torch.def
```

Building may require `--fakeroot` or a remote builder depending on site policy;
`apptainer build --help` lists what is permitted.

Verify GPU visibility inside the container before submitting anything:

```bash
apptainer exec --nv bllarse-jax.sif python -c "import jax; print(jax.devices())"
```

Expected: a CUDA device, not CPU.

- [ ] **Step 8: Commit**

```bash
git add src/slurm/jobs/slurm_run_config_apptainer.sh \
        src/bllarse/tools/run_sweep.py tests/tools/test_run_sweep_apptainer.py
git commit -m "feat(slurm): Apptainer job script, now the sweep default

Shared clusters forbid the Docker daemon. Apptainer preserves UID/GID and
binds \$HOME/\$PWD by default, so the passthrough list shrinks to real
secrets and the per-node image-presence failure mode disappears."
```

---

## Task 12: Sweep definitions

**Files:**
- Create: `bllarse_sweeps/probe_adamw_ce.py`
- Create: `bllarse_sweeps/probe_vbll.py`
- Create: `bllarse_sweeps/probe_ood_adamw_ce.py`
- Create: `bllarse_sweeps/probe_ood_vbll.py`
- Test: `tests/test_sweep_definitions.py`

**Interfaces:**
- Consumes: `bllarse.tools.adapters.run_script_from_config`, Task 9's `probe.py`,
  Task 10's `probing_vbll.py`.
- Produces: `create_configs() -> list[dict]` and `run(config) -> None` in each
  file, the contract `run_sweep.py` and `run_config.py` expect.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sweep_definitions.py`:

```python
import pytest

from bllarse.tools.module import get_module_from_source_path

SWEEPS = [
    "bllarse_sweeps/probe_adamw_ce.py",
    "bllarse_sweeps/probe_vbll.py",
    "bllarse_sweeps/probe_ood_adamw_ce.py",
    "bllarse_sweeps/probe_ood_vbll.py",
]

LADDER = {"dinov3_small", "dinov3_big", "dinov3_large", "dinov3_huge"}


@pytest.mark.parametrize("path", SWEEPS)
def test_sweep_exposes_the_expected_contract(path):
    mod = get_module_from_source_path(path)
    assert callable(mod.create_configs)
    assert callable(mod.run)


@pytest.mark.parametrize("path", SWEEPS)
def test_configs_are_unique_and_well_formed(path):
    configs = get_module_from_source_path(path).create_configs()
    assert len(configs) > 0
    uids = [c["uid"] for c in configs]
    assert len(uids) == len(set(uids)), "uids must be unique -- they key the dumps"
    for c in configs:
        assert c["model"] in LADDER, "dinov3_max is out of scope"
        assert "group_id" in c and "seed" in c


@pytest.mark.parametrize("path", SWEEPS)
def test_all_configs_share_one_group_id(path):
    configs = get_module_from_source_path(path).create_configs()
    assert len({c["group_id"] for c in configs}) == 1


def test_fig1_sweep_sizes_match_the_plan():
    # 4 backbones x 6 datasets x 10 seeds
    assert len(get_module_from_source_path(SWEEPS[0]).create_configs()) == 240


def test_fig2_sweep_sizes_match_the_plan():
    # 4 backbones x 3 (ID, OOD) pairs x 10 seeds.
    # Three pairs, not two ID datasets: CIFAR-10 appears twice, against SVHN
    # (far-OOD) and against CIFAR-100 (near-OOD).
    assert len(get_module_from_source_path(SWEEPS[2]).create_configs()) == 120


def test_fig2_sweeps_pair_each_id_dataset_with_an_ood_dataset():
    for c in get_module_from_source_path(SWEEPS[2]).create_configs():
        assert c["ood_dataset"] in {"svhn", "cifar100"}
        assert c["ood_dataset"] != c["dataset"]
```

- [ ] **Step 2: Run to confirm it fails**

```bash
uv run pytest tests/test_sweep_definitions.py -v
```

Expected: FAIL — sweep files do not exist.

- [ ] **Step 3: Write `bllarse_sweeps/probe_adamw_ce.py`**

```python
"""Fig 1: AdamW+CE linear probing across the DINOv3 ladder and six datasets.

AdamW+CE+TS is NOT a separate sweep -- temperature scaling is fit post-hoc from
these dumps at every dump point, so it costs no GPU time.
"""

from itertools import product

from bllarse.tools.adapters import run_script_from_config

GROUP_ID = "probe_adamw_ce_fig1"

MODELS = ["dinov3_small", "dinov3_big", "dinov3_large", "dinov3_huge"]
DATASETS = {
    "oxford_pets": (37, 512),
    "food101": (101, 512),
    "flowers102": (102, 512),
    "stanford_cars": (196, 512),
    "dtd": (47, 512),
    "imagenet1k": (1000, 512),
}
SEEDS = list(range(137, 147))  # 10 seeds

BASE = dict(
    arm="adamw_ce",
    epochs=40,
    n_dumps=40,
    batch_size=512,
    learning_rate=1e-3,
    weight_decay=1e-4,
    val_fraction=0.1,
)


def create_configs():
    configs = []
    for model, (dataset, (num_classes, img_size)), seed in product(
        MODELS, DATASETS.items(), SEEDS
    ):
        configs.append(dict(
            **BASE,
            model=model,
            dataset=dataset,
            num_classes=num_classes,
            img_size=img_size,
            seed=seed,
            group_id=GROUP_ID,
            uid=f"{model}_{dataset}_seed{seed}",
        ))
    return configs


def run(config):
    run_script_from_config("scripts/alternatives/probe.py", config)
```

- [ ] **Step 4: Write `bllarse_sweeps/probe_vbll.py`**

Same grid, VBLL arm. Repeated in full rather than imported, so each sweep file
reads standalone:

```python
"""Fig 1: VBLL linear probing across the DINOv3 ladder and six datasets.

Reads the same cached feature files as the AdamW arm, so the backbone is
byte-identical between them.
"""

from itertools import product

from bllarse.tools.adapters import run_script_from_config

GROUP_ID = "probe_vbll_fig1"

MODELS = ["dinov3_small", "dinov3_big", "dinov3_large", "dinov3_huge"]
DATASETS = {
    "oxford_pets": (37, 512),
    "food101": (101, 512),
    "flowers102": (102, 512),
    "stanford_cars": (196, 512),
    "dtd": (47, 512),
    "imagenet1k": (1000, 512),
}
SEEDS = list(range(137, 147))

BASE = dict(
    epochs=40,
    n_dumps=40,
    batch_size=512,
    learning_rate=1e-3,
    weight_decay=1e-4,
    val_fraction=0.1,
    prior_scale=1.0,
    wishart_scale=0.1,
    parameterization="diagonal",
    vbll_type="discriminative",
    device="cuda",
)


def create_configs():
    configs = []
    for model, (dataset, (num_classes, img_size)), seed in product(
        MODELS, DATASETS.items(), SEEDS
    ):
        configs.append(dict(
            **BASE,
            model=model,
            dataset=dataset,
            num_classes=num_classes,
            img_size=img_size,
            seed=seed,
            group_id=GROUP_ID,
            uid=f"{model}_{dataset}_seed{seed}",
        ))
    return configs


def run(config):
    run_script_from_config("scripts/vbll_pytorch/probing_vbll.py", config)
```

- [ ] **Step 5: Write the two Fig 2 sweeps**

`bllarse_sweeps/probe_ood_adamw_ce.py`:

```python
"""Fig 2: AdamW+CE OOD detection from frozen features.

Pairs: CIFAR-10 vs SVHN, CIFAR-100 vs SVHN (far-OOD), CIFAR-10 vs CIFAR-100
(near-OOD -- the harder and more informative pair). CIFAR-100 is ID in one pair
and OOD in another. All at res224, so ID and OOD features share a space.
"""

from itertools import product

from bllarse.tools.adapters import run_script_from_config

GROUP_ID = "probe_adamw_ce_fig2"

MODELS = ["dinov3_small", "dinov3_big", "dinov3_large", "dinov3_huge"]
# (id_dataset, num_classes, ood_dataset)
PAIRS = [
    ("cifar10", 10, "svhn"),
    ("cifar10", 10, "cifar100"),
    ("cifar100", 100, "svhn"),
]
SEEDS = list(range(137, 147))

BASE = dict(
    arm="adamw_ce",
    epochs=40,
    n_dumps=40,
    batch_size=512,
    learning_rate=1e-3,
    weight_decay=1e-4,
    val_fraction=0.1,
    img_size=224,
)


def create_configs():
    configs = []
    for model, (dataset, num_classes, ood_dataset), seed in product(MODELS, PAIRS, SEEDS):
        configs.append(dict(
            **BASE,
            model=model,
            dataset=dataset,
            num_classes=num_classes,
            ood_dataset=ood_dataset,
            seed=seed,
            group_id=GROUP_ID,
            uid=f"{model}_{dataset}_vs_{ood_dataset}_seed{seed}",
        ))
    return configs


def run(config):
    run_script_from_config("scripts/alternatives/probe.py", config)
```

This yields 4 × 3 × 10 = 120 configs — three pairs, because CIFAR-10 appears
against both SVHN and CIFAR-100.

`bllarse_sweeps/probe_ood_vbll.py`:

```python
"""Fig 2: VBLL OOD detection from frozen features.

Same three pairs as the AdamW arm, on the same cached res224 features.
"""

from itertools import product

from bllarse.tools.adapters import run_script_from_config

GROUP_ID = "probe_vbll_fig2"

MODELS = ["dinov3_small", "dinov3_big", "dinov3_large", "dinov3_huge"]
# (id_dataset, num_classes, ood_dataset)
PAIRS = [
    ("cifar10", 10, "svhn"),
    ("cifar10", 10, "cifar100"),
    ("cifar100", 100, "svhn"),
]
SEEDS = list(range(137, 147))

BASE = dict(
    epochs=40,
    n_dumps=40,
    batch_size=512,
    learning_rate=1e-3,
    weight_decay=1e-4,
    val_fraction=0.1,
    img_size=224,
    prior_scale=1.0,
    wishart_scale=0.1,
    parameterization="diagonal",
    vbll_type="discriminative",
    device="cuda",
)


def create_configs():
    configs = []
    for model, (dataset, num_classes, ood_dataset), seed in product(MODELS, PAIRS, SEEDS):
        configs.append(dict(
            **BASE,
            model=model,
            dataset=dataset,
            num_classes=num_classes,
            ood_dataset=ood_dataset,
            seed=seed,
            group_id=GROUP_ID,
            uid=f"{model}_{dataset}_vs_{ood_dataset}_seed{seed}",
        ))
    return configs


def run(config):
    run_script_from_config("scripts/vbll_pytorch/probing_vbll.py", config)
```

- [ ] **Step 6: Run the tests**

```bash
uv run pytest tests/test_sweep_definitions.py -v
```

Expected: all pass, with the corrected Fig 2 count.

- [ ] **Step 7: Dry-run a submission to check the sbatch line**

```bash
uv run python -m bllarse.tools.run_sweep bllarse_sweeps/probe_adamw_ce.py --dry-run
```

Expected: an `sbatch --export=ALL --array=0-239%7 ... slurm_run_config_apptainer.sh`
line. Compare `239` against the cluster's `MaxArraySize` from Task 11 Step 1; if
it exceeds it, submission must be chunked with `--index-offset` / `--num-jobs`.

- [ ] **Step 8: Commit**

```bash
git add bllarse_sweeps/probe_*.py tests/test_sweep_definitions.py
git commit -m "feat(sweeps): Fig 1 and Fig 2 probing sweep definitions

Four-model ladder, 10 seeds. Fig 2 covers three (ID, OOD) pairs including
the near-OOD CIFAR-10 vs CIFAR-100. TS is post-hoc, so no TS sweep."
```

---

## Task 13: End-to-end smoke test on the cluster

**Files:**
- Create: `tests/test_dump_roundtrip.py`
- Create: `bllarse_sweeps/probe_smoke.py`

**Interfaces:**
- Consumes: everything above.
- Produces: nothing new — this is the gate that licenses treating dumps as the
  source of truth.

- [ ] **Step 1: Write the round-trip test**

This is the load-bearing check: metrics recomputed from a dump must match
metrics computed in-loop, to fp16 round-trip tolerance.

Create `tests/test_dump_roundtrip.py`:

```python
import numpy as np
import pytest

from bllarse import evaldump, metrics


def test_metrics_survive_the_fp16_dump_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    logits = rng.normal(scale=3.0, size=(2000, 50)).astype(np.float32)
    labels = rng.integers(0, 50, size=2000)

    in_loop = metrics.classification_metrics(logits, labels)

    path = evaldump.write_dump(
        tmp_path, "g", "u", step=1, logits=logits, labels=labels,
        arm="adamw_ce", split="test", logit_kind="softmax_logits",
        seed=137, samples_seen=1000,
    )
    record = evaldump.load_dump(path)
    post_hoc = metrics.classification_metrics(record.logits, record.labels)

    assert post_hoc["acc"] == pytest.approx(in_loop["acc"], abs=2e-3)
    assert post_hoc["nll"] == pytest.approx(in_loop["nll"], rel=2e-3)
    assert post_hoc["brier"] == pytest.approx(in_loop["brier"], rel=2e-3)
    assert post_hoc["ece"] == pytest.approx(in_loop["ece"], abs=5e-3)


def test_temperature_fit_survives_the_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    logits = (rng.normal(size=(3000, 10)) * 4.0).astype(np.float32)
    labels = logits.argmax(axis=-1)

    in_loop_T = metrics.fit_temperature(logits, labels)
    path = evaldump.write_dump(
        tmp_path, "g", "u2", step=1, logits=logits, labels=labels,
        arm="adamw_ce", split="val", logit_kind="softmax_logits",
        seed=137, samples_seen=1000,
    )
    rec = evaldump.load_dump(path)
    assert metrics.fit_temperature(rec.logits, rec.labels) == pytest.approx(in_loop_T, rel=1e-2)
```

- [ ] **Step 2: Run it**

```bash
uv run pytest tests/test_dump_roundtrip.py -v
```

Expected: 2 passed. If ECE drifts beyond `5e-3`, that is real information — fp16
is too coarse for 20-bin ECE at this logit scale, and the dump dtype needs
revisiting before any sweep runs. Do not widen the tolerance to make it pass.

- [ ] **Step 3: Write a one-config-per-arm smoke sweep**

Create `bllarse_sweeps/probe_smoke.py`:

```python
"""One config per arm and per figure, for shaking out the cluster path.

Not a results sweep. Exercises: Apptainer launch, MLflow nesting, dump paths,
and both the Fig 1 (no OOD) and Fig 2 (with OOD) code paths.
"""

from bllarse.tools.adapters import run_script_from_config

GROUP_ID = "probe_smoke"

_JAX = dict(arm="adamw_ce", epochs=4, n_dumps=3, batch_size=512,
            learning_rate=1e-3, weight_decay=1e-4, val_fraction=0.1,
            model="dinov3_small", seed=137, group_id=GROUP_ID)
_TORCH = dict(epochs=4, n_dumps=3, batch_size=512, learning_rate=1e-3,
              weight_decay=1e-4, val_fraction=0.1, parameterization="diagonal",
              vbll_type="discriminative", device="cuda",
              model="dinov3_small", seed=137, group_id=GROUP_ID)

CONFIGS = [
    dict(**_JAX, dataset="cifar10", num_classes=10, img_size=224, uid="jax_fig1"),
    dict(**_JAX, dataset="cifar10", num_classes=10, img_size=224,
         ood_dataset="svhn", uid="jax_fig2"),
    dict(**_TORCH, dataset="cifar10", num_classes=10, img_size=224, uid="torch_fig1"),
    dict(**_TORCH, dataset="cifar10", num_classes=10, img_size=224,
         ood_dataset="svhn", uid="torch_fig2"),
]


def create_configs():
    return CONFIGS


def run(config):
    script = ("scripts/vbll_pytorch/probing_vbll.py"
              if "arm" not in config else "scripts/alternatives/probe.py")
    run_script_from_config(script, config)
```

- [ ] **Step 4: Submit the smoke sweep on the cluster**

```bash
export BLLARSE_SIF=$PWD/bllarse-jax.sif
export BLLARSE_EVAL_DUMP_ROOT=$SCRATCH/bllarse/eval_dumps
uv run python -m bllarse.tools.run_sweep bllarse_sweeps/probe_smoke.py \
    --max-concurrent 2 --job-name probe_smoke
squeue -u "$USER"
```

The two torch configs need `BLLARSE_SIF` pointing at `bllarse-torch.sif`; submit
them as a second invocation with `--index-offset 2 --num-jobs 2` and the torch
SIF exported.

- [ ] **Step 5: Verify all five gates**

```bash
# 1. jobs completed
sacct -u "$USER" --name probe_smoke --format=JobID,State,ExitCode

# 2. dumps landed with the right shapes
uv run python -c "
from bllarse import evaldump, metrics
import os
root = os.environ['BLLARSE_EVAL_DUMP_ROOT']
for uid in ('jax_fig1', 'jax_fig2', 'torch_fig1', 'torch_fig2'):
    recs = list(evaldump.iter_dumps(root, 'probe_smoke', uid))
    print(uid, len(recs), [r.meta['split'] for r in recs])
    assert recs, f'no dumps for {uid}'
    f = [r for r in recs if r.meta['split'] == 'test'][-1]
    print('   acc', metrics.accuracy(f.logits, f.labels))
"

# 3. score channel present and non-degenerate for VBLL
uv run python -c "
from bllarse import evaldump
import os, numpy as np
recs = list(evaldump.iter_dumps(os.environ['BLLARSE_EVAL_DUMP_ROOT'], 'probe_smoke', 'torch_fig1'))
s = recs[-1].score
assert s is not None, 'VBLL score channel missing'
assert np.std(s) > 0, 'VBLL score is constant -- the epistemic term is dead'
print('score std', float(np.std(s)))
"

# 4. detection metrics compute on the Fig 2 dumps
uv run python -c "
from bllarse import evaldump, metrics, detection
import os
root = os.environ['BLLARSE_EVAL_DUMP_ROOT']
recs = list(evaldump.iter_dumps(root, 'probe_smoke', 'jax_fig2'))
idr = [r for r in recs if r.meta['split'] == 'test'][-1]
oodr = [r for r in recs if r.meta['split'] == 'ood'][-1]
print(detection.detection_metrics(metrics.msp_ood_score(idr.logits),
                                  metrics.msp_ood_score(oodr.logits)))
"

# 5. MLflow nesting -- children appear under one parent in the UI
```

Gate 3 is the one worth dwelling on: a constant VBLL score means the epistemic
term is not being learned, which would silently gut the §5.8 ablation.

- [ ] **Step 6: Commit**

```bash
git add tests/test_dump_roundtrip.py bllarse_sweeps/probe_smoke.py
git commit -m "test: dump round-trip and cluster smoke sweep

The round-trip test is what licenses treating dumps as the source of truth
for every figure. The smoke sweep exercises Apptainer, MLflow nesting, both
dump paths, and the VBLL score channel before 1600 real tasks are queued."
```

---

## Task 14: `bllarse.analysis.curves` — dumps to metric curves, with per-point TS

**Files:**
- Create: `src/bllarse/analysis/curves.py`
- Test: `tests/analysis/test_curves.py`

**Interfaces:**
- Consumes: `bllarse.evaldump`, `bllarse.metrics`, `bllarse.detection`.
- Produces:
  - `run_curve(root, group_id, uid, *, temperature_scaled=False) -> list[dict]` — rows with `step`, `samples_seen`, `acc`, `ece`, `nll`, `brier`, consumable directly by `thresholds.make_targets` and `thresholds.first_crossing`
  - `run_detection(root, group_id, uid, score_fn) -> dict` — `auroc`/`fpr95`/`aupr` at the final point

This is the figure-assembly layer. It is **not** on the critical path for
launching runs — Tasks 1–13 produce the dumps, and this reads them — but Figs 1–4
cannot be drawn without it, and it is where the per-dump-point temperature refit
actually happens.

- [ ] **Step 1: Write the failing tests**

Create `tests/analysis/test_curves.py`:

```python
import numpy as np
import pytest

from bllarse import evaldump, metrics
from bllarse.analysis import curves


def _write_run(root, uid, steps, *, n=800, k=5, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    for step in steps:
        base = rng.normal(size=(n, k)).astype(np.float32)
        labels = base.argmax(axis=-1)
        logits = base * scale
        for split, sl in (("val", slice(0, n // 2)), ("test", slice(n // 2, n))):
            evaldump.write_dump(
                root, "g", uid, step, logits[sl], labels[sl],
                arm="adamw_ce", split=split, logit_kind="softmax_logits",
                seed=137, samples_seen=step * 100,
            )


def test_run_curve_has_one_row_per_step_with_all_metrics(tmp_path):
    _write_run(tmp_path, "u1", [1, 5, 20])
    curve = curves.run_curve(tmp_path, "g", "u1")
    assert [row["step"] for row in curve] == [1, 5, 20]
    assert set(curve[0]) >= {"step", "samples_seen", "acc", "ece", "nll", "brier"}
    assert curve[0]["samples_seen"] == 100


def test_temperature_scaling_is_refit_at_every_step(tmp_path):
    _write_run(tmp_path, "u2", [1, 5, 20], scale=5.0, seed=1)
    plain = curves.run_curve(tmp_path, "g", "u2", temperature_scaled=False)
    scaled = curves.run_curve(tmp_path, "g", "u2", temperature_scaled=True)
    # A per-point refit must reduce NLL at every point, not just the last.
    for a, b in zip(plain, scaled):
        assert b["nll"] <= a["nll"] + 1e-6
        assert b["temperature"] > 0
    # And the fitted temperatures must not all be identical -- that would mean
    # one fit was reused rather than refit.
    assert len({round(row["temperature"], 6) for row in scaled}) > 1


def test_temperature_scaling_does_not_change_accuracy(tmp_path):
    _write_run(tmp_path, "u3", [1, 5], scale=5.0, seed=2)
    plain = curves.run_curve(tmp_path, "g", "u3", temperature_scaled=False)
    scaled = curves.run_curve(tmp_path, "g", "u3", temperature_scaled=True)
    for a, b in zip(plain, scaled):
        assert a["acc"] == pytest.approx(b["acc"])


def test_curve_feeds_the_threshold_protocol_directly(tmp_path):
    from bllarse.analysis import thresholds

    _write_run(tmp_path, "u4", [1, 5, 20])
    curve = curves.run_curve(tmp_path, "g", "u4")
    targets = thresholds.make_targets(curve, fraction=0.95)
    assert thresholds.first_crossing(curve, targets).step is not None


def test_run_detection_computes_from_id_and_ood_dumps(tmp_path):
    rng = np.random.default_rng(3)
    n, k = 400, 5
    for split, loc in (("test", 6.0), ("ood", 0.0)):
        base = rng.normal(loc=0.0, size=(n, k)).astype(np.float32)
        base[:, 0] += loc  # ID is confident on class 0; OOD is not
        evaldump.write_dump(
            tmp_path, "g", "u5", 1, base, base.argmax(axis=-1),
            arm="adamw_ce", split=split, logit_kind="softmax_logits",
            seed=137, samples_seen=100,
        )
    out = curves.run_detection(tmp_path, "g", "u5", metrics.msp_ood_score)
    assert set(out) == {"auroc", "fpr95", "aupr"}
    assert out["auroc"] > 0.8, "confident ID vs diffuse OOD must be separable"


def test_run_detection_raises_without_an_ood_dump(tmp_path):
    _write_run(tmp_path, "u6", [1])
    with pytest.raises(ValueError, match="ood"):
        curves.run_detection(tmp_path, "g", "u6", metrics.msp_ood_score)
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/analysis/test_curves.py -v
```

Expected: `ImportError: cannot import name 'curves'`.

- [ ] **Step 3: Implement `src/bllarse/analysis/curves.py`**

```python
"""Assemble metric curves from eval dumps. Pure post-hoc -- no training here.

This is where temperature scaling actually happens. Standard TS fits one T at
convergence, but Figs 5-6 are trajectories, and a once-fit T would be a strawman
exactly on the early part of the curve where the sample-efficiency claim lives.
IBProbit recalibrates itself continuously, so the baseline is allowed to as
well: T is refit on the pinned val split at every dumped step.

That is non-standard and makes the baseline strictly stronger, which is the
direction a fairness deviation should run. It needs saying in the caption.
"""

from typing import Any, Callable

from bllarse import detection, evaldump, metrics


def _by_split(root, group_id: str, uid: str) -> dict[str, dict[int, Any]]:
    """Index a run's dumps as {split: {step: record}}."""
    indexed: dict[str, dict[int, Any]] = {}
    for record in evaldump.iter_dumps(root, group_id, uid):
        indexed.setdefault(record.meta["split"], {})[record.meta["step"]] = record
    return indexed


def run_curve(
    root, group_id: str, uid: str, *, temperature_scaled: bool = False
) -> list[dict[str, Any]]:
    """One row per dumped step, on the test split.

    With `temperature_scaled=True`, T is fit on that step's val dump and applied
    to that step's test logits -- a fresh fit per point, never a reused one.
    """
    indexed = _by_split(root, group_id, uid)
    if "test" not in indexed:
        raise ValueError(f"no test-split dumps found for {group_id}/{uid}")

    rows = []
    for step in sorted(indexed["test"]):
        test_record = indexed["test"][step]
        logits, labels = test_record.logits, test_record.labels
        temperature = 1.0

        if temperature_scaled:
            val_record = indexed.get("val", {}).get(step)
            if val_record is None:
                raise ValueError(
                    f"temperature scaling needs a val dump at step {step} for "
                    f"{group_id}/{uid}, but none was written"
                )
            temperature = metrics.fit_temperature(val_record.logits, val_record.labels)
            logits = metrics.apply_temperature(logits, temperature)

        row = {
            "step": step,
            "samples_seen": test_record.meta["samples_seen"],
            "temperature": temperature,
            **metrics.classification_metrics(logits, labels),
        }
        rows.append(row)
    return rows


def run_detection(
    root, group_id: str, uid: str, score_fn: Callable[[Any], Any]
) -> dict[str, float]:
    """Detection metrics at the final dumped point.

    `score_fn` maps logits to a score under the project convention that higher
    means more OOD -- metrics.msp_ood_score or metrics.entropy_ood_score. Passing
    the same function to both sides is what makes the comparison a controlled
    one; a per-arm score would confound model with score function.
    """
    indexed = _by_split(root, group_id, uid)
    if "ood" not in indexed:
        raise ValueError(f"no ood-split dump found for {group_id}/{uid}")
    if "test" not in indexed:
        raise ValueError(f"no test-split dump found for {group_id}/{uid}")

    id_record = indexed["test"][max(indexed["test"])]
    ood_record = indexed["ood"][max(indexed["ood"])]
    return detection.detection_metrics(
        score_fn(id_record.logits), score_fn(ood_record.logits)
    )
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/analysis/test_curves.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Run the whole suite**

```bash
uv run pytest tests/ -v --ignore=tests/test_probing_vbll_smoke.py
```

Expected: everything passes. Run the VBLL test separately if torch is in a
different env.

- [ ] **Step 6: Commit**

```bash
git add src/bllarse/analysis/curves.py tests/analysis/test_curves.py
git commit -m "feat(analysis): metric curves from dumps, with per-point TS refit

The figure-assembly layer. Temperature is refit on the pinned val split at
every dumped step rather than once at convergence, so AdamW+CE+TS has a
trajectory in Figs 5-7 instead of a single endpoint."
```

---

## After this plan

Phase 1a can launch: `probe_adamw_ce`, `probe_vbll`, `probe_ood_adamw_ce`,
`probe_ood_vbll` — 720 array tasks, Figs 1–4 buildable from the dumps.

Then **Plan B**: 0d-ii, the PyTorch full-FT backend (torch DINOv3, ImageNet
pipeline at 512, matched augmentation/precision/optimizer, forward and recipe
parity tests) plus the ImageNet sweeps → Figs 5–7.

`results-plan.md` §4.1's FLOP model is independent of both and gates whether
Fig 7 is worth running at all.
