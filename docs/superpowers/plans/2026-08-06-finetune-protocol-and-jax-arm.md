# Fine-tune Protocol + JAX Arm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the framework-independent fine-tuning protocol (`src/bllarse/finetune/`) and the JAX arm script that fine-tunes the whole DINOv3 trunk on ImageNet-1k for AdamW+CE and IBProbit, writing eval dumps on a log-spaced schedule so temperature scaling, Brier and OOD detection all come post hoc.

**Architecture:** A shared `bllarse.finetune` package holds everything both frameworks must agree on — budget arithmetic, the micro/accumulation plan, the batch-order RNG, the pinned ImageNet eval split, and the uint8 OOD preload. The jitted JAX kernels are lifted out of `scripts/alternatives/full_finetune_comparison.py` into `bllarse.finetune.jax_steps` so that script and the new arm script share one copy. `scripts/finetune/finetune_jax.py` is then thin: resolve the plan, stream batches, accumulate, dump.

**Tech Stack:** JAX + Equinox + Optax, `equimo` for DINOv3, HuggingFace `datasets`, numpy, pytest.

## Global Constraints

- **Phase 1 only.** The PyTorch VBLL arm and the Optuna campaign are a separate plan. Do not add them here.
- **`--img-size` is 224.** `DINOV3_DATASET_IMG_SIZES` maps `imagenet1k -> 512`; that table is for frozen feature extraction. 512 must be *refused with an explicit error*, not silently accepted.
- **Never run training locally.** All GPUs on this machine belong to the user's jobs. Run tests with `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu`. Real training runs on Raven only.
- **`N_train = 1_281_167`**, `N_val = 50_000`, `val_fraction = 0.2` → 10,000 val / 40,000 test.
- **OOD sets are exactly `("imagenet_r", "imagenet_a", "imagenet_v2")`, evaluated in full.** ImageNet-C is deliberately excluded.
- **Default `n_dumps = 24`** for fine-tuning (probing uses 40).
- **Batch order RNG is numpy `default_rng`**, never `jax.random` — the PyTorch arm in Phase 2 must reproduce the identical permutation.
- Commit messages end with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- Branch is `infra/results-protocol`. Origin is the private Forgejo. Never push to GitHub.

---

### Task 1: Budget and accumulation arithmetic

`bllarse.finetune.protocol` is the module both arm scripts import so their step counts, batch plans and shuffle order cannot drift apart.

**Files:**
- Create: `src/bllarse/finetune/__init__.py`
- Create: `src/bllarse/finetune/protocol.py`
- Test: `tests/finetune/test_protocol.py`

**Interfaces:**
- Consumes: `bllarse.evaldump.dump_schedule(num_steps, n_dumps)` (existing)
- Produces:
  - `DEFAULT_N_DUMPS: int = 24`
  - `steps_for_epochs(n_train: int, epochs: float, effective_batch: int) -> int`
  - `accum_plan(target_batch: int, micro_batch: int) -> tuple[int, int, int]` returning `(micro, num_accum, effective_batch)`
  - `shuffled_indices(n: int, seed: int, epoch: int) -> np.ndarray`
  - `dump_steps(num_steps: int, n_dumps: int = DEFAULT_N_DUMPS) -> np.ndarray`
  - `ood_dump_steps(steps: np.ndarray, every_k: int) -> np.ndarray`

- [ ] **Step 1: Create the package directory and write the failing test**

Create `tests/finetune/__init__.py` as an empty file, then `tests/finetune/test_protocol.py`:

```python
"""The arithmetic both arm scripts must agree on.

These are pure integer/RNG functions with no JAX or torch in them, which is the
point: the PyTorch arm in Phase 2 imports the SAME module, so a divergence
between the two arms' step counts or batch order becomes impossible rather than
merely unlikely.
"""

import numpy as np
import pytest

from bllarse.finetune import protocol


def test_steps_for_epochs_matches_the_design_table():
    # The table in docs/superpowers/specs/2026-08-06-full-network-finetuning-design.md
    # section 3, at the fixed 4-epoch cap on ImageNet-1k.
    n_train = 1_281_167
    assert protocol.steps_for_epochs(n_train, 4, 512) == 10_009
    assert protocol.steps_for_epochs(n_train, 4, 1024) == 5_004
    assert protocol.steps_for_epochs(n_train, 4, 2048) == 2_502
    assert protocol.steps_for_epochs(n_train, 4, 4096) == 1_251


def test_steps_for_epochs_accepts_a_fractional_epoch():
    # Tuning trials get a fraction of an epoch, not a whole number of them.
    assert protocol.steps_for_epochs(1_281_167, 0.5, 1024) == 625


def test_steps_for_epochs_rejects_a_batch_larger_than_the_dataset():
    with pytest.raises(ValueError, match="exceeds"):
        protocol.steps_for_epochs(1000, 4, 2000)


def test_steps_for_epochs_rejects_a_budget_yielding_no_steps():
    with pytest.raises(ValueError, match="zero steps"):
        protocol.steps_for_epochs(1000, 0.001, 512)


def test_accum_plan_divides_evenly_when_it_can():
    assert protocol.accum_plan(4096, 64) == (64, 64, 4096)


def test_accum_plan_reports_the_effective_batch_it_actually_achieves():
    # 4096 / 100 rounds to 41 accumulation steps, so the effective batch is
    # 4100, not 4096. Callers log the returned value, never the request --
    # otherwise MLflow records a batch size the run did not use.
    micro, num_accum, effective = protocol.accum_plan(4096, 100)
    assert (micro, num_accum) == (100, 41)
    assert effective == 4100


def test_accum_plan_clamps_micro_to_the_target():
    # A target smaller than what fits in memory needs no accumulation.
    assert protocol.accum_plan(256, 1024) == (256, 1, 256)


def test_accum_plan_rejects_a_nonpositive_micro_batch():
    with pytest.raises(ValueError, match="micro_batch"):
        protocol.accum_plan(4096, 0)


def test_shuffled_indices_is_a_permutation():
    idx = protocol.shuffled_indices(1000, seed=137, epoch=0)
    assert np.array_equal(np.sort(idx), np.arange(1000))


def test_shuffled_indices_is_deterministic_across_calls():
    # This is the property that keeps the JAX and PyTorch arms on the same
    # data stream. If it ever regresses, the two arms silently diverge.
    a = protocol.shuffled_indices(1000, seed=137, epoch=3)
    b = protocol.shuffled_indices(1000, seed=137, epoch=3)
    assert np.array_equal(a, b)


def test_shuffled_indices_differs_across_epochs_and_seeds():
    base = protocol.shuffled_indices(1000, seed=137, epoch=0)
    assert not np.array_equal(base, protocol.shuffled_indices(1000, seed=137, epoch=1))
    assert not np.array_equal(base, protocol.shuffled_indices(1000, seed=138, epoch=0))


def test_dump_steps_spans_one_to_num_steps():
    steps = protocol.dump_steps(5000)
    assert len(steps) == protocol.DEFAULT_N_DUMPS
    assert steps[0] == 1
    assert steps[-1] == 5000
    assert np.all(np.diff(steps) > 0)


def test_ood_dump_steps_subsamples_and_always_keeps_the_last():
    steps = np.array([1, 2, 4, 8, 16, 32, 64])
    chosen = protocol.ood_dump_steps(steps, every_k=2)
    assert chosen[0] == 1
    assert chosen[-1] == 64  # the final step is never dropped by the stride
    assert set(chosen).issubset(set(steps))


def test_ood_dump_steps_disabled_returns_empty():
    steps = np.array([1, 2, 4])
    assert len(protocol.ood_dump_steps(steps, every_k=0)) == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run pytest tests/finetune/test_protocol.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bllarse.finetune'`

- [ ] **Step 3: Write the implementation**

Create `src/bllarse/finetune/__init__.py`:

```python
"""Framework-independent fine-tuning protocol.

The JAX arm (scripts/finetune/finetune_jax.py) and the PyTorch VBLL arm both
import from here, so the two cannot disagree about step counts, effective batch
sizes, the eval schedule or the order the training data arrives in.
"""
```

Create `src/bllarse/finetune/protocol.py`:

```python
"""Budget arithmetic, the accumulation plan, and the shared batch order.

Fine-tuning inverts the probing campaign's convention. Probing fixed 30,000
optimizer STEPS and derived epochs (bllarse/tuning/budgets.py); here the cap is
4 EPOCHS and the step count falls out of the tuned batch size. That inversion is
what makes Fig 5 valid: results-plan puts samples-seen on its x-axis, so arms
tuned to different batch sizes stay comparable even though they take different
numbers of steps.
"""

import numpy as np

from bllarse import evaldump

# Probing uses 40. Fine-tuning runs are 1-2 orders of magnitude shorter in
# steps, and every dump costs a full pass over 50k eval images, so 24 buys the
# same log-spaced resolution at a proportionate price.
DEFAULT_N_DUMPS = 24


def steps_for_epochs(n_train: int, epochs: float, effective_batch: int) -> int:
    """Optimizer steps in `epochs` passes at `effective_batch`, drop-remainder.

    `epochs` is a float so a tuning trial can ask for a fraction of a pass --
    which is the whole tuning budget under this design, since trials are
    truncated by a samples-seen budget rather than run to convergence.
    """
    if effective_batch < 1:
        raise ValueError(f"effective_batch must be >= 1, got {effective_batch}")
    if effective_batch > n_train:
        raise ValueError(
            f"effective_batch={effective_batch} exceeds n_train={n_train}: "
            f"a single batch cannot be drawn"
        )
    steps = int(epochs * n_train) // effective_batch
    if steps < 1:
        raise ValueError(
            f"epochs={epochs} at effective_batch={effective_batch} over "
            f"n_train={n_train} yields zero steps"
        )
    return steps


def accum_plan(target_batch: int, micro_batch: int) -> tuple[int, int, int]:
    """Resolve a target batch into (micro, num_accum, effective).

    `micro_batch` is a memory artifact -- the largest per-forward batch that
    fits -- and differs between arms and frameworks. `target_batch` is the tuned
    quantity. The returned `effective` is what was ACTUALLY achieved and need
    not equal the request: callers must log the returned value, or MLflow will
    record a batch size the run did not use.
    """
    if micro_batch < 1:
        raise ValueError(f"micro_batch must be >= 1, got {micro_batch}")
    if target_batch < 1:
        raise ValueError(f"target_batch must be >= 1, got {target_batch}")
    micro = min(micro_batch, target_batch)
    num_accum = max(1, round(target_batch / micro))
    return micro, num_accum, micro * num_accum


def shuffled_indices(n: int, seed: int, epoch: int) -> np.ndarray:
    """The training permutation for one pass.

    numpy `default_rng`, NOT `jax.random`: the PyTorch arm has to reproduce this
    exact order so both arms consume an identical data stream, which is the
    property full_finetune_comparison.py already guarantees within JAX and which
    a per-framework RNG would quietly break.
    """
    return np.random.default_rng([seed, epoch]).permutation(n)


def dump_steps(num_steps: int, n_dumps: int = DEFAULT_N_DUMPS) -> np.ndarray:
    """Log-spaced eval steps, dense early. Always includes 1 and `num_steps`."""
    return evaldump.dump_schedule(num_steps, n_dumps)


def ood_dump_steps(steps: np.ndarray, every_k: int) -> np.ndarray:
    """Every k-th entry of `steps`, with the final step always retained.

    OOD evaluation forwards 47,500 images, so it rides a stride of the val/test
    schedule rather than every point. The last step is forced back in because
    Table 1's converged endpoint needs it and a stride will drop it whenever
    len(steps) is not 1 mod k.
    """
    steps = np.asarray(steps, dtype=int)
    if every_k <= 0:
        return np.asarray([], dtype=int)
    chosen = steps[::every_k]
    if len(steps) and steps[-1] not in chosen:
        chosen = np.append(chosen, steps[-1])
    return chosen
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run pytest tests/finetune/test_protocol.py -v`
Expected: PASS, 14 tests

- [ ] **Step 5: Commit**

```bash
git add src/bllarse/finetune/ tests/finetune/
git commit -m "feat(finetune): budget arithmetic and the shared batch order

Fine-tuning caps epochs and derives steps, inverting the probing
campaign's fixed-step convention. shuffled_indices uses numpy
default_rng rather than jax.random so the PyTorch arm can reproduce the
identical data stream.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: ImageNet eval split and uint8 OOD preload

**Files:**
- Create: `src/bllarse/finetune/imagenet.py`
- Test: `tests/finetune/test_imagenet.py`

**Interfaces:**
- Consumes: `bllarse.splits.load_or_make_pinned_split(cache_dir, dataset, split, n, val_fraction)` (existing), `protocol` from Task 1
- Produces:
  - `IMAGENET_N_TRAIN: int = 1_281_167`, `IMAGENET_N_VAL: int = 50_000`
  - `DEFAULT_VAL_FRACTION: float = 0.2`
  - `OOD_KEYS: tuple[str, ...] = ("imagenet_r", "imagenet_a", "imagenet_v2")`
  - `pinned_eval_split(cache_dir, n=IMAGENET_N_VAL, val_fraction=DEFAULT_VAL_FRACTION) -> tuple[np.ndarray, np.ndarray]`
  - `decode_uint8(split, img_key, img_size, idx) -> np.ndarray` — one image, `(H, W, 3)` uint8
  - `load_images_uint8(split, img_key, img_size, indices, num_workers=8) -> np.ndarray` — `(N, H, W, 3)` uint8
  - `normalize_chunk(images_uint8, mean, std) -> np.ndarray` — `(N, 3, H, W)` float32

- [ ] **Step 1: Write the failing test**

Create `tests/finetune/test_imagenet.py`:

```python
"""The ImageNet eval split and the uint8 host cache.

The uint8 part is not a micro-optimization. R+A+V2 in full is 47,500 images;
at 3x224x224 float32 that is 28.6 GB against slurm_tune.sh's --mem=40G. It fits
at the old 2,000-per-set cap and does not at full size, so a float32 preload
would OOM several hours into a run. The dtype assertion below is the guard.
"""

import numpy as np
import pytest

from bllarse.finetune import imagenet


class FakeSplit:
    """Minimal stand-in for a HuggingFace split: indexable, returns a dict."""

    def __init__(self, images):
        self._images = images

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        return {"image": self._images[idx]}


def _fake_images(n, size):
    from PIL import Image
    return [
        Image.fromarray(
            np.full((size, size, 3), fill_value=i % 256, dtype=np.uint8)
        )
        for i in range(n)
    ]


def test_ood_keys_are_r_a_v2_and_exclude_imagenet_c():
    assert imagenet.OOD_KEYS == ("imagenet_r", "imagenet_a", "imagenet_v2")
    assert not any("imagenet_c" in key for key in imagenet.OOD_KEYS)


def test_pinned_eval_split_is_disjoint_and_correctly_sized(tmp_path):
    val_idx, test_idx = imagenet.pinned_eval_split(tmp_path)
    assert len(val_idx) == 10_000
    assert len(test_idx) == 40_000
    assert not set(val_idx.tolist()) & set(test_idx.tolist())


def test_pinned_eval_split_is_stable_across_calls(tmp_path):
    # Temperature scaling refits on this split at every one of 24 dumps; if it
    # moved, no two dumps would be comparably conditioned.
    first, _ = imagenet.pinned_eval_split(tmp_path)
    second, _ = imagenet.pinned_eval_split(tmp_path)
    assert np.array_equal(first, second)


def test_load_images_uint8_returns_uint8_not_float():
    split = FakeSplit(_fake_images(4, 32))
    out = imagenet.load_images_uint8(split, "image", 32, np.arange(4), num_workers=2)
    assert out.dtype == np.uint8, "a float32 preload OOMs at full OOD-set size"
    assert out.shape == (4, 32, 32, 3)


def test_load_images_uint8_resizes_to_the_requested_size():
    split = FakeSplit(_fake_images(2, 64))
    out = imagenet.load_images_uint8(split, "image", 32, np.arange(2), num_workers=2)
    assert out.shape == (2, 32, 32, 3)


def test_load_images_uint8_preserves_index_order():
    split = FakeSplit(_fake_images(5, 8))
    out = imagenet.load_images_uint8(split, "image", 8, np.array([3, 1]), num_workers=2)
    assert out[0, 0, 0, 0] == 3
    assert out[1, 0, 0, 0] == 1


def test_normalize_chunk_produces_channels_first_float32():
    images = np.full((2, 8, 8, 3), 255, dtype=np.uint8)
    out = imagenet.normalize_chunk(images, mean=np.zeros(3), std=np.ones(3))
    assert out.shape == (2, 3, 8, 8)
    assert out.dtype == np.float32
    assert out == pytest.approx(np.ones_like(out))


def test_normalize_chunk_applies_mean_and_std():
    images = np.full((1, 4, 4, 3), 255, dtype=np.uint8)
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.25, 0.25, 0.25])
    out = imagenet.normalize_chunk(images, mean=mean, std=std)
    assert out == pytest.approx(np.full_like(out, 2.0))


def test_memory_budget_of_a_full_ood_preload_stays_under_slurm_mem():
    # 47,500 images (R 30k + A 7.5k + V2 10k) at 224px. Documents WHY uint8:
    # float32 would be 28.6 GB against --mem=40G with the model also resident.
    n, size = 47_500, 224
    uint8_gb = n * size * size * 3 / 1024**3
    assert uint8_gb < 8.0
    assert uint8_gb * 4 > 25.0  # the float32 equivalent genuinely does not fit
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run pytest tests/finetune/test_imagenet.py -v`
Expected: FAIL with `ImportError: cannot import name 'imagenet'`

- [ ] **Step 3: Write the implementation**

Create `src/bllarse/finetune/imagenet.py`:

```python
"""ImageNet-1k facts, the pinned eval split, and the uint8 host cache.

The OOD suite is ImageNet-R, -A and -V2 in FULL -- no subsampling. ImageNet-C is
excluded by decision: it measures synthetic corruption robustness, a different
axis from the natural distribution shift Fig 6 is about, and it was the bulk of
the OOD cost. Full sets also take the sampling noise out of every point on the
OOD trajectory.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
from PIL import Image

from bllarse import splits

IMAGENET_N_TRAIN = 1_281_167
IMAGENET_N_VAL = 50_000

# 0.2 of the official 50k validation split -> 10k val / 40k test. Val is used
# ONLY to fit temperature and to select the reporting step; test is never seen
# by either.
DEFAULT_VAL_FRACTION = 0.2

OOD_KEYS = ("imagenet_r", "imagenet_a", "imagenet_v2")

# Per-set sizes, for cost arithmetic and for asserting a load was complete.
OOD_SIZES = {"imagenet_r": 30_000, "imagenet_a": 7_500, "imagenet_v2": 10_000}

# R and A cover 200 ImageNet classes; their out-of-subset logits are masked
# before scoring (full_finetune_comparison._ood_metrics_from_logits).
OOD_SUBSET_KEYS = frozenset({"imagenet_r", "imagenet_a"})


def pinned_eval_split(
    cache_dir, n: int = IMAGENET_N_VAL, val_fraction: float = DEFAULT_VAL_FRACTION
) -> tuple[np.ndarray, np.ndarray]:
    """(val_idx, test_idx) over the official validation split, seed-independent.

    Derived from the dataset name alone, so it is identical across arms, seeds
    and machines -- see bllarse/splits.py for why that matters when T is refit
    at every dumped step.
    """
    return splits.load_or_make_pinned_split(
        cache_dir, "imagenet1k", "validation", n, val_fraction
    )


def decode_uint8(split, img_key: str, img_size: int, idx) -> np.ndarray:
    """Decode, RGB-convert and resize ONE image to (img_size, img_size, 3) uint8.

    Deliberately does NOT divide by 255 or normalize, unlike
    vit_classification._load_and_process. Normalization is deferred to
    `normalize_chunk` so the host cache stays 1 byte per channel.
    """
    img = split[int(idx)][img_key]
    if hasattr(img, "convert"):
        img = img.convert("RGB")
    else:
        img = Image.fromarray(np.asarray(img)).convert("RGB")
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def load_images_uint8(
    split, img_key: str, img_size: int, indices, num_workers: int = 8
) -> np.ndarray:
    """Decode `indices` into one (N, H, W, 3) uint8 array, order preserved.

    At full OOD-set size (47,500 images) this is 7.1 GB. The float32 equivalent
    is 28.6 GB, which does not fit alongside the model under the SLURM memory
    request -- so the dtype here is a requirement, not a tuning choice.
    """
    worker = partial(decode_uint8, split, img_key, img_size)
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        # pool.map preserves input order, which the caller relies on to line
        # images up with their labels.
        return np.stack(list(pool.map(worker, np.asarray(indices))))


def normalize_chunk(images_uint8, mean, std) -> np.ndarray:
    """(N, H, W, 3) uint8 -> (N, 3, H, W) float32, scaled and normalized.

    Called per eval chunk rather than once over the whole cache, so the float32
    expansion only ever exists for `eval_chunk` images at a time.
    """
    x = np.asarray(images_uint8, dtype=np.float32) / 255.0
    x = (x - np.asarray(mean)) / np.asarray(std)
    return np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2)), dtype=np.float32)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run pytest tests/finetune/test_imagenet.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add src/bllarse/finetune/imagenet.py tests/finetune/test_imagenet.py
git commit -m "feat(finetune): pinned ImageNet eval split and uint8 OOD preload

OOD suite is R/A/V2 in full; ImageNet-C excluded by decision. The host
cache is uint8 with normalisation deferred to the eval chunk loop:
47,500 images at 3x224x224 float32 is 28.6 GB against --mem=40G, which
fits at the old 2,000-per-set cap and does not at full size.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Lift the jitted kernels into the shared module

Mechanical move, no behaviour change. Two copies of donation-tuned memory code is the exact drift bug the shared `bllarse.metrics` import in `bllarse-torch.def` was added to prevent.

**Files:**
- Create: `src/bllarse/finetune/jax_steps.py`
- Modify: `scripts/alternatives/full_finetune_comparison.py` (delete the moved definitions, import them instead)
- Test: `tests/finetune/test_jax_steps.py`

**Interfaces:**
- Produces (all moved verbatim from `full_finetune_comparison.py`): `forward_features`, `_tree_add`, `apply_grads`, `apply_grads_bayes`, `ce_micro_grad`, `ce_micro_hvp`, `ibprobit_micro_forward`, `ibprobit_micro_forward_nograd`, `ibprobit_micro_hvp`, `_to_default_device`, `draw_bayes_sample`, `draw_hutchinson_probe`, `_stash_bayes_hessian`, `make_bayes_optimizer`

- [ ] **Step 1: Write the failing test**

Create `tests/finetune/test_jax_steps.py`:

```python
"""The lifted kernels are the SAME objects the comparison script uses.

full_finetune_comparison.py and scripts/finetune/finetune_jax.py both drive the
gradient-accumulation path. These kernels carry buffer-donation contracts tuned
against a measured ~6 GiB peak (commit cb81798); a second, drifting copy would
regress that silently and only on the largest backbone.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "alternatives"))

import full_finetune_comparison as ffc  # noqa: E402

from bllarse.finetune import jax_steps  # noqa: E402


def test_comparison_script_uses_the_shared_kernels_not_its_own_copies():
    for name in ("_tree_add", "apply_grads", "ce_micro_grad",
                 "ibprobit_micro_forward", "forward_features"):
        assert getattr(ffc, name) is getattr(jax_steps, name), (
            f"{name} has diverged: the comparison script holds its own copy"
        )


def test_tree_add_sums_leafwise_and_tolerates_none_leaves():
    a = {"w": jnp.ones((2, 2)), "b": None}
    b = {"w": jnp.full((2, 2), 3.0), "b": None}
    out = jax_steps._tree_add(a, b)
    assert out["w"] == jax.numpy.all(jnp.full((2, 2), 4.0)) or jnp.allclose(
        out["w"], jnp.full((2, 2), 4.0)
    )
    assert out["b"] is None


def test_tree_add_returns_the_present_leaf_when_one_side_is_none():
    out = jax_steps._tree_add({"w": jnp.ones(3)}, {"w": None})
    assert jnp.allclose(out["w"], jnp.ones(3))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run pytest tests/finetune/test_jax_steps.py -v`
Expected: FAIL with `ImportError: cannot import name 'jax_steps'`

- [ ] **Step 3: Create the module by moving the definitions**

Create `src/bllarse/finetune/jax_steps.py` with a module docstring, the imports below, and then **cut** these definitions verbatim from `scripts/alternatives/full_finetune_comparison.py` — do not retype them, and keep every existing comment, especially the donation-contract comments on `_tree_add` and `apply_grads`:

`_to_default_device` (line ~86), `forward_features` (~95), `ibprobit_micro_forward` (~135), `ibprobit_micro_forward_nograd` (~152), `ce_micro_grad` (~163), `_tree_add` (~175), `apply_grads` (~191), `make_bayes_optimizer` (~217), `draw_bayes_sample` (~240), `draw_hutchinson_probe` (~252), `_stash_bayes_hessian` (~280), `ibprobit_micro_hvp` (~324), `ce_micro_hvp` (~342), `apply_grads_bayes` (~357).

Module header:

```python
"""Jitted micro-batch kernels and the gradient-accumulation step.

Lifted out of scripts/alternatives/full_finetune_comparison.py so that script
and scripts/finetune/finetune_jax.py share ONE copy. These carry buffer-donation
contracts tuned against a measured peak (commit cb81798, ~6 GiB off dinov3_huge)
and the argument order of `_tree_add` is load-bearing -- a second copy that
drifted would regress memory silently, and only on the largest backbone.
"""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import random as jr, vmap

from blrax.states import ScaleByEvonState, MatrixEvonLeaf, _is_evon_leaf
from blrax.utils import (
    _evon_sample_leaves,
    _project,
    _project_back,
    precision as bayes_precision,
    rademacher_like,
    sample_posterior,
)
from bllarse.utils import bayes_optimizer
```

- [ ] **Step 4: Replace the definitions in the comparison script with an import**

In `scripts/alternatives/full_finetune_comparison.py`, delete the moved function bodies and add, next to the other `bllarse` imports:

```python
from bllarse.finetune.jax_steps import (  # noqa: F401  -- re-exported for tests
    _stash_bayes_hessian,
    _to_default_device,
    _tree_add,
    apply_grads,
    apply_grads_bayes,
    ce_micro_grad,
    ce_micro_hvp,
    draw_bayes_sample,
    draw_hutchinson_probe,
    forward_features,
    ibprobit_micro_forward,
    ibprobit_micro_forward_nograd,
    ibprobit_micro_hvp,
    make_bayes_optimizer,
)
```

Then remove any imports the script no longer uses (`ScaleByEvonState` is still needed by `run_arm`'s hutchinson cadence check; `MatrixEvonLeaf`, `_is_evon_leaf`, `_project`, `_project_back`, `bayes_precision`, `rademacher_like`, `sample_posterior`, `_evon_sample_leaves` and `bayes_optimizer` are not).

- [ ] **Step 5: Run the new test AND the existing comparison-script tests**

Run: `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run pytest tests/finetune/test_jax_steps.py tests/test_full_finetune_ood.py tests/test_full_finetune_bayes_glue.py tests/test_bayes_optimizer_head_lr.py -v`
Expected: PASS. The three pre-existing files are the regression gate on the move — if any fails, the lift changed behaviour and must be corrected, not worked around.

- [ ] **Step 6: Commit**

```bash
git add src/bllarse/finetune/jax_steps.py scripts/alternatives/full_finetune_comparison.py tests/finetune/test_jax_steps.py
git commit -m "refactor(finetune): lift jitted kernels into bllarse.finetune.jax_steps

Mechanical move, no behaviour change; the comparison script now imports
what it used to define. The new arm script needs the same accumulation
path, and these kernels carry buffer-donation contracts tuned against a
measured peak (cb81798) -- a second copy would regress memory silently
and only on the largest backbone.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: The arm script's CLI, budget resolution and guards

Everything in `finetune_jax.py` that can be tested without a GPU. The training loop lands in Task 5.

**Files:**
- Create: `scripts/finetune/__init__.py` (empty)
- Create: `scripts/finetune/finetune_jax.py`
- Test: `tests/finetune/test_finetune_jax_plan.py`

**Interfaces:**
- Consumes: `protocol.steps_for_epochs`, `protocol.accum_plan`, `protocol.dump_steps`, `protocol.ood_dump_steps`, `imagenet.IMAGENET_N_TRAIN`
- Produces:
  - `build_argparser() -> argparse.ArgumentParser`
  - `resolve_plan(args, micro_batch: int) -> dict` with keys `micro`, `num_accum`, `effective_batch`, `steps`, `dump_steps`, `ood_steps`
  - `check_img_size(args) -> None`
  - `main(args, on_dump=None) -> None` (stub in this task, filled in Task 5)

- [ ] **Step 1: Write the failing test**

Create `tests/finetune/test_finetune_jax_plan.py`:

```python
"""The arm script's pure planning layer: budget, accumulation, guards.

Everything here runs without a GPU or a dataset, so a misconfigured sweep fails
at submit time rather than after a node allocation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "finetune"))

import finetune_jax as fj  # noqa: E402


def _args(**overrides):
    argv = ["--arm", "ce", "--batch-size", "1024"]
    for key, value in overrides.items():
        flag = f"--{key.replace('_', '-')}"
        # store_true flags take no value; passing one is an argparse error.
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    return fj.build_argparser().parse_args(argv)


def test_default_epochs_is_the_four_epoch_cap():
    assert _args().epochs == 4.0


def test_default_img_size_is_224():
    assert _args().img_size == 224


def test_check_img_size_refuses_512():
    # DINOV3_DATASET_IMG_SIZES maps imagenet1k -> 512 for FEATURE EXTRACTION.
    # Inheriting it here would be a ~5x cost accident, so it must be loud.
    with pytest.raises(SystemExit, match="512"):
        fj.check_img_size(_args(img_size=512))


def test_check_img_size_accepts_224():
    fj.check_img_size(_args(img_size=224))  # must not raise


def test_check_img_size_allows_an_explicit_override():
    fj.check_img_size(_args(img_size=512, allow_non_224=True))


def test_resolve_plan_derives_steps_from_the_effective_batch():
    plan = fj.resolve_plan(_args(batch_size=1024), micro_batch=64)
    assert plan["effective_batch"] == 1024
    assert plan["num_accum"] == 16
    assert plan["steps"] == 5_004


def test_resolve_plan_uses_the_achieved_batch_not_the_request():
    # accum_plan may not hit the target exactly; steps must follow what the run
    # will actually do, or samples-seen on Fig 5's x-axis is wrong.
    plan = fj.resolve_plan(_args(batch_size=4096), micro_batch=100)
    assert plan["effective_batch"] == 4100
    assert plan["steps"] == fj.protocol.steps_for_epochs(
        fj.imagenet.IMAGENET_N_TRAIN, 4.0, 4100
    )


def test_resolve_plan_dump_schedule_spans_the_run():
    plan = fj.resolve_plan(_args(batch_size=1024), micro_batch=64)
    assert plan["dump_steps"][0] == 1
    assert plan["dump_steps"][-1] == plan["steps"]
    assert len(plan["dump_steps"]) == 24


def test_resolve_plan_ood_steps_are_a_subset_of_dump_steps():
    plan = fj.resolve_plan(_args(batch_size=1024), micro_batch=64)
    assert set(plan["ood_steps"]).issubset(set(plan["dump_steps"].tolist()))
    assert plan["ood_steps"][-1] == plan["steps"]


def test_resolve_plan_ood_disabled_yields_no_ood_steps():
    plan = fj.resolve_plan(
        _args(batch_size=1024, ood_every_k_dumps=0), micro_batch=64
    )
    assert len(plan["ood_steps"]) == 0


def test_resolve_plan_accepts_a_fractional_epoch_for_tuning():
    plan = fj.resolve_plan(_args(batch_size=1024, epochs=0.5), micro_batch=64)
    assert plan["steps"] == 625


def test_head_lr_is_rejected_for_the_ibprobit_arm():
    # IBProbit runs no optimizer on its head -- optimizer.init sees the backbone
    # alone and the head is assimilated in closed form by loss_fn.update. A
    # head lr silently doing nothing would look like a tuned axis that isn't.
    argv = ["--arm", "ibprobit", "--batch-size", "1024", "--head-lr", "1e-2"]
    args = fj.build_argparser().parse_args(argv)
    with pytest.raises(SystemExit, match="head"):
        fj.check_arm_flags(args)


def test_head_lr_is_accepted_for_the_ce_arm():
    fj.check_arm_flags(_args(head_lr=1e-2))  # must not raise
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run pytest tests/finetune/test_finetune_jax_plan.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'finetune_jax'`

- [ ] **Step 3: Write the implementation**

Create `scripts/finetune/__init__.py` (empty file), then `scripts/finetune/finetune_jax.py`:

```python
"""Full-network fine-tuning of DINOv3 on ImageNet-1k -- JAX arms.

One arm per invocation (`--arm ce|ibprobit`), writing eval dumps on a log-spaced
schedule so temperature scaling, Brier and OOD detection are all computed post
hoc by the same code that scores the probing figures.

This is NOT scripts/alternatives/full_finetune_comparison.py, which stays as the
IVON/EVON benchmarking harness with inline metrics. The two share their jitted
kernels via bllarse.finetune.jax_steps.
"""

import argparse
import os

from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.secrets", override=False)

from bllarse.finetune import imagenet, protocol  # noqa: E402

# Fine-tuning resolution. DINOV3_DATASET_IMG_SIZES maps imagenet1k -> 512, but
# that table exists for FROZEN FEATURE EXTRACTION; fine-tuning at 512 is ~5x the
# cost for no benefit, so inheriting it must be impossible by accident.
FINETUNE_IMG_SIZE = 224


def check_img_size(args) -> None:
    """Refuse a non-224 resolution unless it was asked for explicitly."""
    if args.img_size != FINETUNE_IMG_SIZE and not args.allow_non_224:
        raise SystemExit(
            f"[finetune] --img-size {args.img_size} refused. Fine-tuning runs at "
            f"{FINETUNE_IMG_SIZE}. DINOV3_DATASET_IMG_SIZES maps imagenet1k -> 512 "
            f"for frozen feature extraction, and inheriting that here costs ~5x "
            f"for no benefit. Pass --allow-non-224 if this is deliberate."
        )


def check_arm_flags(args) -> None:
    """Reject flags that the chosen arm cannot act on.

    IBProbit has no head learning rate: run_arm initialises the optimizer on the
    backbone alone and the head is assimilated in closed form by
    `loss_fn.update(...)`. Accepting --head-lr there would present a tuned axis
    that silently does nothing.
    """
    if args.arm == "ibprobit" and args.head_lr is not None:
        raise SystemExit(
            "[finetune] --head-lr is meaningless for --arm ibprobit: the head is "
            "updated in closed form by CAVI, not by the optimizer. Its head-side "
            "knob is --num-update-iters."
        )


def resolve_plan(args, micro_batch: int) -> dict:
    """Turn (target batch, micro batch, epoch cap) into the concrete schedule.

    `steps` is derived from the ACHIEVED effective batch, not the requested one:
    Fig 5's x-axis is samples seen, so a plan built on a batch size the run does
    not use would mislabel every point on it.
    """
    micro, num_accum, effective = protocol.accum_plan(args.batch_size, micro_batch)
    steps = protocol.steps_for_epochs(
        imagenet.IMAGENET_N_TRAIN, args.epochs, effective
    )
    schedule = protocol.dump_steps(steps, args.n_dumps)
    return {
        "micro": micro,
        "num_accum": num_accum,
        "effective_batch": effective,
        "steps": steps,
        "dump_steps": schedule,
        "ood_steps": protocol.ood_dump_steps(schedule, args.ood_every_k_dumps),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=["ce", "ibprobit"])
    p.add_argument("--model", default="dinov3_large")
    p.add_argument("--dataset", default="imagenet1k")
    p.add_argument("--img-size", type=int, default=FINETUNE_IMG_SIZE)
    p.add_argument(
        "--allow-non-224", action="store_true",
        help="Permit a resolution other than 224. See check_img_size.",
    )

    p.add_argument(
        "--epochs", type=float, default=4.0,
        help="Epoch cap. Float so a tuning trial can ask for a fraction of a "
             "pass; steps are derived from this and the effective batch.",
    )
    p.add_argument(
        "--batch-size", type=int, required=True,
        help="TARGET (effective) batch, reached by gradient accumulation. This "
             "is the tuned quantity; --micro-batch-size is a memory artifact.",
    )
    p.add_argument(
        "--micro-batch-size", type=int, default=None,
        help="Per-forward batch. Omit to auto-probe the largest that fits.",
    )
    p.add_argument("--batch-size-init", type=int, default=32)
    p.add_argument("--batch-size-max", type=int, default=8192)

    p.add_argument("--lr", type=float, default=1e-4, help="Trunk learning rate")
    p.add_argument(
        "--head-lr", type=float, default=None,
        help="Head learning rate (CE arm only; IBProbit has no head optimizer).",
    )
    p.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="Trunk weight decay. Head weight decay is pinned to 0 on every arm.",
    )
    p.add_argument("--label-smooth", type=float, default=0.0)
    p.add_argument("--num-update-iters", type=int, default=128,
                   help="CAVI iterations per macro batch (IBProbit arm).")
    p.add_argument("--bf16", action="store_true")

    p.add_argument("--n-dumps", type=int, default=protocol.DEFAULT_N_DUMPS)
    p.add_argument(
        "--ood-every-k-dumps", type=int, default=2,
        help="Write OOD dumps on every k-th scheduled dump. 0 disables OOD "
             "entirely, which is what tuning trials use.",
    )
    p.add_argument("--eval-chunk", type=int, default=256)
    p.add_argument("--val-fraction", type=float, default=imagenet.DEFAULT_VAL_FRACTION)

    p.add_argument("--dump-root",
                   default=os.environ.get("BLLARSE_EVAL_DUMP_ROOT", "eval_dumps"))
    p.add_argument("--group-id", default="finetune")
    p.add_argument("--uid", default=None)
    p.add_argument("--split-cache-dir", default=".cache/splits")
    p.add_argument("--seed", type=int, default=137)
    p.add_argument("--device", default="gpu")

    p.add_argument("--enable-mlflow", action="store_true")
    p.add_argument("--mlflow-experiment", default="bllarse-finetune")
    return p


def main(args, on_dump=None) -> None:
    """Fine-tune one arm. Filled in by Task 5.

    `on_dump(dump_index, step, scalars)` matches the hook
    bllarse.tuning.objective.evaluate passes, so the Phase 2 tuning campaign can
    drive this script without a wrapper.
    """
    check_img_size(args)
    check_arm_flags(args)
    raise NotImplementedError("training loop lands in Task 5")


if __name__ == "__main__":
    from jax import config as jax_config

    parsed = build_argparser().parse_args()
    jax_config.update("jax_platform_name", parsed.device)
    main(parsed)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run pytest tests/finetune/test_finetune_jax_plan.py -v`
Expected: PASS, 13 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/finetune/ tests/finetune/test_finetune_jax_plan.py
git commit -m "feat(finetune): JAX arm CLI, budget resolution and guards

Steps derive from the ACHIEVED effective batch rather than the request,
so Fig 5's samples-seen axis cannot be mislabelled. --img-size 512 is
refused explicitly, and --head-lr is refused on the ibprobit arm, which
runs no optimizer on its head.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Training loop with val/test dumps

**Files:**
- Modify: `scripts/finetune/finetune_jax.py` (replace the `main` stub)
- Test: `tests/finetune/test_finetune_jax_contracts.py`

**Interfaces:**
- Consumes: `jax_steps.*` (Task 3), `resolve_plan` (Task 4), `evaldump.write_dump`, `imagenet.pinned_eval_split`
- Produces: `stream_micro_batches(...)`, `evaluate_and_dump(...)`, a working `main(args, on_dump=None)`

- [ ] **Step 1: Write the contract test**

These pin the invariants the loop must satisfy; they are NOT an end-to-end
exercise of the loop, which needs a real backbone and is verified on Raven per
the exit criteria. Written first so the loop is implemented against a stated
contract rather than checked against itself afterwards.

Create `tests/finetune/test_finetune_jax_contracts.py`:

```python
"""Invariants the training loop must satisfy, checked without a GPU.

Deliberately NOT an end-to-end run: that needs a real DINOv3 forward and is
covered by the Raven smoke in the plan's exit criteria. What is pinned here is
the contract the ANALYSIS layer depends on, because every one of these failures
is silent -- a run that violates them produces files that load fine and figures
that are wrong.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "finetune"))

import finetune_jax as fj  # noqa: E402

from bllarse import evaldump  # noqa: E402


def test_dump_steps_and_ood_steps_never_disagree_about_the_final_step():
    plan = fj.resolve_plan(
        fj.build_argparser().parse_args(["--arm", "ce", "--batch-size", "512"]),
        micro_batch=64,
    )
    # curves._id_ood_pair raises if the final test dump and the final ood dump
    # sit at different steps, so the schedules must agree at the endpoint.
    assert plan["ood_steps"][-1] == plan["dump_steps"][-1]


def test_every_scheduled_step_writes_both_val_and_test(tmp_path):
    """The invariant the analysis layer depends on.

    fig1_curves.score_run intersects the val and test step sets and returns None
    if either is missing, so a run that dumped test-only would be silently
    dropped from the figure rather than reported as broken.
    """
    steps = [1, 3, 10]
    for step in steps:
        for split in ("val", "test"):
            evaldump.write_dump(
                tmp_path, "g", "u", step,
                np.zeros((4, 3), dtype=np.float32), np.zeros(4, dtype=np.int64),
                arm="ce", split=split, logit_kind="softmax_logits",
                seed=137, samples_seen=step * 512,
            )
    by_split = {}
    for record in evaldump.iter_dumps(tmp_path, "g", "u"):
        by_split.setdefault(record.meta["split"], set()).add(record.meta["step"])
    assert by_split["val"] == by_split["test"] == set(steps)


def test_samples_seen_tracks_the_effective_batch(tmp_path):
    """samples_seen is Fig 5's x-axis; it must count the effective batch."""
    evaldump.write_dump(
        tmp_path, "g", "u", 7,
        np.zeros((4, 3), dtype=np.float32), np.zeros(4, dtype=np.int64),
        arm="ce", split="test", logit_kind="softmax_logits",
        seed=137, samples_seen=7 * 1024,
    )
    record = next(iter(evaldump.iter_dumps(tmp_path, "g", "u")))
    assert record.meta["samples_seen"] == 7168
```

- [ ] **Step 2: Run the test**

Run: `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run pytest tests/finetune/test_finetune_jax_contracts.py -v`
Expected: PASS, 3 tests — these exercise `resolve_plan` (Task 4) and `evaldump`, both of which already exist. That is intended: they are the specification the loop in Step 3 must be written against, not a red test to turn green. Read them before writing the loop.

- [ ] **Step 3: Implement the training loop**

Replace the `main` stub in `scripts/finetune/finetune_jax.py`. Add these imports at the top of the file, after the existing ones:

```python
import gc

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random as jr

from bllarse import evaldump, metrics
from bllarse.finetune import jax_steps
from bllarse.losses import CrossEntropy
```

Add the batch streamer and the eval/dump helper:

```python
def stream_micro_batches(train_split, ds_config, micro, img_size, mean, std,
                         total_micro, seed, num_workers=16):
    """Yield exactly `total_micro` micro-batches, reshuffling between passes.

    The permutation comes from protocol.shuffled_indices (numpy), NOT jax.random,
    so the PyTorch arm reproduces this exact order. Short trailing batches are
    skipped to keep the jitted shape constant.
    """
    n = len(train_split)
    labels_all = np.asarray(train_split[ds_config["label_key"]])
    emitted, epoch = 0, 0
    while emitted < total_micro:
        order = protocol.shuffled_indices(n, seed, epoch)
        for start in range(0, n - micro + 1, micro):
            batch_idx = order[start:start + micro]
            images = imagenet.load_images_uint8(
                train_split, ds_config["img_key"], img_size, batch_idx,
                num_workers=num_workers,
            )
            yield (jnp.asarray(imagenet.normalize_chunk(images, mean, std)),
                   jnp.asarray(labels_all[batch_idx]))
            emitted += 1
            if emitted >= total_micro:
                return
        epoch += 1


def evaluate_and_dump(arm, backbone, head_or_loss, images_uint8, labels, *,
                      split, step, samples_seen, args, mean, std, key,
                      subset=None):
    """Forward one preloaded eval set in chunks and write its dump.

    Returns the scalar metrics so the caller can drive `on_dump` and MLflow.
    Normalization happens per chunk (imagenet.normalize_chunk), so the float32
    expansion only ever covers `--eval-chunk` images at a time.
    """
    inference_bb = eqx.nn.inference_mode(backbone, True)
    chunks = []
    for start in range(0, len(labels), args.eval_chunk):
        raw = images_uint8[start:start + args.eval_chunk]
        imgs = jnp.asarray(imagenet.normalize_chunk(raw, mean, std))
        labs = jnp.asarray(labels[start:start + args.eval_chunk])
        if arm == "ibprobit":
            logits = eval_logits_ibprobit(inference_bb, head_or_loss, imgs, labs,
                                            key, bf16=args.bf16)
        else:
            logits = eval_logits_ce(inference_bb, head_or_loss, imgs, key,
                                       bf16=args.bf16)
        chunks.append(np.asarray(logits))
    logits = np.concatenate(chunks, axis=0)

    evaldump.write_dump(
        args.dump_root, args.group_id, args.uid, step,
        logits, labels,
        arm=arm, split=split,
        logit_kind="moderated_score" if arm == "ibprobit" else "softmax_logits",
        seed=args.seed, samples_seen=samples_seen,
    )
    return metrics.classification_metrics(logits, labels)


@eqx.filter_jit
def eval_logits_ce(backbone, head, images, key, bf16=False):
    features = jax_steps.forward_features(backbone, images, key, bf16=bf16)
    return jax.vmap(head)(features)


@eqx.filter_jit
def eval_logits_ibprobit(backbone, loss_fn, images, labels, key, bf16=False):
    features = jax_steps.forward_features(backbone, images, key, bf16=bf16)
    _, logits = loss_fn(features, labels, with_logits=True, loss_type=2)
    return logits
```

Then replace `main`:

```python
def main(args, on_dump=None) -> None:
    check_img_size(args)
    check_arm_flags(args)
    if args.uid is None:
        args.uid = f"{args.model}_{args.arm}_seed{args.seed}"

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "alternatives"))
    from vit_classification import (
        DATASET_CONFIGS, EQUIMO_MODELS, IMAGENET_MEAN, IMAGENET_STD,
        get_pretrained_backbone,
    )
    from datasets import load_dataset

    ds_config = DATASET_CONFIGS[args.dataset]
    embed_dim = EQUIMO_MODELS[args.model]["embed_dim"]
    num_classes = ds_config["num_classes"]
    mean, std = IMAGENET_MEAN, IMAGENET_STD

    key = jr.PRNGKey(args.seed)
    key, model_key, init_key, fwd_key = jr.split(key, 4)

    ds = load_dataset(ds_config["hf_path"], name=ds_config.get("hf_config"))
    train_split = ds["train"]
    eval_split = ds[ds_config["test_split"]]

    backbone = eqx.nn.inference_mode(get_pretrained_backbone(args.model, model_key), True)

    micro = args.micro_batch_size
    if micro is None:
        raise SystemExit(
            "[finetune] --micro-batch-size is required in this task; the "
            "auto-probe is wired in Task 5 step 4."
        )
    plan = resolve_plan(args, micro)
    print(f"[finetune] arm={args.arm} effective_batch={plan['effective_batch']} "
          f"= {plan['num_accum']} x micro {plan['micro']}, steps={plan['steps']}")

    # Pinned val/test partition over the official validation split, preloaded
    # once as uint8 and normalized per chunk at eval time.
    val_idx, test_idx = imagenet.pinned_eval_split(
        args.split_cache_dir, n=len(eval_split), val_fraction=args.val_fraction
    )
    eval_labels_all = np.asarray(eval_split[ds_config["label_key"]])
    eval_sets = {}
    for name, idx in (("val", val_idx), ("test", test_idx)):
        eval_sets[name] = (
            imagenet.load_images_uint8(eval_split, ds_config["img_key"],
                                       args.img_size, idx),
            eval_labels_all[idx],
        )
        print(f"[finetune] preloaded {name}: {len(idx)} images")

    if args.arm == "ce":
        loss_fn = CrossEntropy(args.label_smooth, num_classes)
        head = eqx.nn.Linear(embed_dim, num_classes, key=init_key)
        head_lr = args.head_lr if args.head_lr is not None else args.lr
        param_labels = (
            jax.tree.map(lambda _: "backbone", eqx.filter(backbone, eqx.is_array)),
            jax.tree.map(lambda _: "head", eqx.filter(head, eqx.is_array)),
        )
        optimizer = optax.multi_transform(
            {
                "backbone": optax.adamw(args.lr, weight_decay=args.weight_decay),
                # Head weight decay is 0 on every arm (design D6): VBLL's KL is
                # its head regulariser and IBProbit runs no optimizer on the
                # head at all, so a nonzero value here would be an asymmetry.
                "head": optax.adamw(head_lr, weight_decay=0.0),
            },
            param_labels,
        )
        opt_state = optimizer.init(eqx.filter((backbone, head), eqx.is_array))
    else:
        from bllarse.losses import IBProbit
        loss_fn = IBProbit(embed_dim, num_classes, key=init_key)
        head = None
        optimizer = optax.adamw(args.lr, weight_decay=args.weight_decay)
        opt_state = optimizer.init(eqx.filter(backbone, eqx.is_array))

    dump_set = set(int(s) for s in plan["dump_steps"])
    micro_iter = stream_micro_batches(
        train_split, ds_config, plan["micro"], args.img_size, mean, std,
        plan["steps"] * plan["num_accum"], args.seed,
    )

    dump_index = 0
    for step in range(1, plan["steps"] + 1):
        train_key = jr.fold_in(fwd_key, step)
        grads_acc, all_fts, all_labels = None, [], []

        for _ in range(plan["num_accum"]):
            images, labels = next(micro_iter)
            if args.arm == "ibprobit":
                if step == 1:
                    fts, _ = jax_steps.ibprobit_micro_forward_nograd(
                        backbone, loss_fn, images, labels, train_key, bf16=args.bf16)
                else:
                    grads, fts, _ = jax_steps.ibprobit_micro_forward(
                        backbone, loss_fn, images, labels, train_key, bf16=args.bf16)
                    grads_acc = grads if grads_acc is None else jax_steps._tree_add(
                        grads, grads_acc)
                all_fts.append(fts)
                all_labels.append(labels)
            else:
                grads, _ = jax_steps.ce_micro_grad(
                    (backbone, head), loss_fn, images, labels, train_key,
                    bf16=args.bf16)
                grads_acc = grads if grads_acc is None else jax_steps._tree_add(
                    grads, grads_acc)

        if args.arm == "ibprobit":
            if step > 1:
                backbone, opt_state = jax_steps.apply_grads(
                    backbone, grads_acc, opt_state, optimizer, plan["num_accum"])
            loss_fn = loss_fn.update(
                jnp.concatenate(all_fts), jnp.concatenate(all_labels),
                num_iters=args.num_update_iters)
            head_or_loss = loss_fn
        else:
            (backbone, head), opt_state = jax_steps.apply_grads(
                (backbone, head), grads_acc, opt_state, optimizer, plan["num_accum"])
            head_or_loss = head

        if step in dump_set:
            samples_seen = step * plan["effective_batch"]
            scalars = {}
            for name in ("val", "test"):
                images_uint8, labels = eval_sets[name]
                out = evaluate_and_dump(
                    args.arm, backbone, head_or_loss, images_uint8, labels,
                    split=name, step=step, samples_seen=samples_seen,
                    args=args, mean=mean, std=std, key=fwd_key,
                )
                scalars.update({f"{name}_{k}": v for k, v in out.items()})
            print(f"[finetune] step {step}/{plan['steps']} " + " ".join(
                f"{k}={v:.4f}" for k, v in sorted(scalars.items())))
            if on_dump is not None:
                on_dump(dump_index, step, scalars)
            dump_index += 1
            jax.clear_caches()
            gc.collect()

    print(f"[finetune] done: {dump_index} dumps under "
          f"{args.dump_root}/{args.group_id}/{args.uid}")
```

Add `from pathlib import Path` to the imports.

- [ ] **Step 4: Wire the micro-batch auto-probe**

Replace the `SystemExit` in `main` with a call to the existing prober. Add to the imports inside `main`:

```python
    from full_finetune_comparison import probe_micro_batch_size
```

and replace the block:

```python
    micro = args.micro_batch_size
    if micro is None:
        key, probe_key = jr.split(key)
        print("[finetune] probing maximal micro-batch size...")
        micro = probe_micro_batch_size(
            backbone, args, embed_dim, num_classes, args.img_size,
            args.batch_size_init, args.batch_size_max, probe_key,
        )
        print(f"[finetune] probed micro-batch size: {micro}")
```

`probe_micro_batch_size` reads `args.bf16`, `args.optimizer` and `args.estimator`. Add the two it does not already have to `build_argparser`, defaulted so they never change behaviour here:

```python
    p.add_argument("--optimizer", default="adamw", choices=["adamw"],
                   help="Present because the shared micro-batch prober reads "
                        "it. IVON/EVON live in full_finetune_comparison.py.")
    p.add_argument("--estimator", default="sampling", choices=["sampling"],
                   help="Present for the shared prober; see --optimizer.")
```

- [ ] **Step 5: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run pytest tests/finetune/ -v`
Expected: PASS, all files

- [ ] **Step 6: Commit**

```bash
git add scripts/finetune/finetune_jax.py tests/finetune/test_finetune_jax_contracts.py
git commit -m "feat(finetune): JAX training loop with val/test dumps

Both splits are dumped at every scheduled step, which is the invariant
fig1_curves.score_run relies on -- it intersects the val and test step
sets and returns None if either is missing, so a test-only run would be
dropped from the figure rather than reported as broken. samples_seen
counts the effective batch, since that is Fig 5's x-axis.

Head weight decay is pinned to 0 on the CE arm to match VBLL (KL is its
head regulariser) and IBProbit (no head optimizer).

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: OOD dumps on the schedule

**Files:**
- Modify: `scripts/finetune/finetune_jax.py`
- Test: `tests/finetune/test_finetune_jax_ood.py`

**Interfaces:**
- Consumes: `imagenet.OOD_KEYS`, `imagenet.OOD_SUBSET_KEYS`, `imagenet.OOD_SIZES`, `precompute_ood_features.build_ood_registry`, `precompute_ood_features.load_wnid_to_idx`
- Produces: `load_ood_sets(args, mean, std) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray | None]]`

- [ ] **Step 1: Write the failing test**

Create `tests/finetune/test_finetune_jax_ood.py`:

```python
"""OOD dumps ride the val/test schedule rather than a separate cadence.

The probing arms originally dumped OOD only at the FINAL step, which made the
CIFAR-100-vs-SVHN detection number partly a measurement of CE's overfitting at
that step. Scheduling OOD alongside val/test is the fix, generalized.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "finetune"))

import finetune_jax as fj  # noqa: E402

from bllarse.finetune import imagenet  # noqa: E402


def _plan(**overrides):
    argv = ["--arm", "ce", "--batch-size", "1024"]
    for key, value in overrides.items():
        argv.extend([f"--{key.replace('_', '-')}", str(value)])
    return fj.resolve_plan(fj.build_argparser().parse_args(argv), micro_batch=64)


def test_ood_steps_ride_the_dump_schedule():
    plan = _plan()
    assert set(plan["ood_steps"]).issubset(set(plan["dump_steps"].tolist()))


def test_ood_cadence_of_two_halves_the_dump_count_roughly():
    plan = _plan(ood_every_k_dumps=2)
    # 24 dumps at stride 2 -> 12, and the final step is already on the stride.
    assert 11 <= len(plan["ood_steps"]) <= 13


def test_tuning_disables_ood_entirely():
    # OOD forwards 47,500 images; a tuning trial must never pay for it.
    assert len(_plan(ood_every_k_dumps=0)["ood_steps"]) == 0


def test_only_r_and_a_are_logit_masked_subsets():
    assert imagenet.OOD_SUBSET_KEYS == frozenset({"imagenet_r", "imagenet_a"})
    assert "imagenet_v2" not in imagenet.OOD_SUBSET_KEYS


def test_ood_sizes_sum_to_the_documented_total():
    # 47,500 is the number the uint8 memory argument rests on.
    assert sum(imagenet.OOD_SIZES[k] for k in imagenet.OOD_KEYS) == 47_500
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run pytest tests/finetune/test_finetune_jax_ood.py -v`
Expected: FAIL on `test_only_r_and_a_are_logit_masked_subsets` / `test_ood_sizes_sum_to_the_documented_total` only if Task 2 omitted those constants; the cadence tests pass from Task 4. Fix whatever fails.

- [ ] **Step 3: Implement the OOD loader**

Add to `scripts/finetune/finetune_jax.py`:

```python
def load_ood_sets(args, img_size):
    """Preload every OOD set as uint8 host arrays, once, shared across steps.

    Returns [(name, images_uint8, labels, subset_or_None)]. `subset` is the
    array of classes present in R/A, used to mask out-of-subset logits before
    scoring; V2 spans all 1000 classes and gets None.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "alternatives"))
    from precompute_ood_features import build_ood_registry, load_wnid_to_idx

    wnid_to_idx = load_wnid_to_idx()
    registry = build_ood_registry()
    loaded = []
    for name in imagenet.OOD_KEYS:
        split, img_key = registry[name](wnid_to_idx)
        labels = np.asarray(split["label"])
        subset = np.unique(labels) if name in imagenet.OOD_SUBSET_KEYS else None
        images = imagenet.load_images_uint8(
            split, img_key, img_size, np.arange(len(labels)))
        expected = imagenet.OOD_SIZES[name]
        if len(labels) != expected:
            # A short load means the HF split changed. Scoring it anyway would
            # produce a plausible AUROC over the wrong population.
            raise SystemExit(
                f"[finetune] {name} has {len(labels)} images, expected {expected}"
            )
        loaded.append((name, images, labels, subset))
        print(f"[finetune] preloaded {name}: {len(labels)} images"
              + (f", {len(subset)}-class subset" if subset is not None else ""))
    return loaded
```

In `main`, after the val/test preload:

```python
    ood_sets = load_ood_sets(args, args.img_size) if len(plan["ood_steps"]) else []
    ood_step_set = set(int(s) for s in plan["ood_steps"])
```

and inside the `if step in dump_set:` block, after the val/test loop:

```python
            if step in ood_step_set:
                for name, images_uint8, labels, subset in ood_sets:
                    evaluate_and_dump(
                        args.arm, backbone, head_or_loss, images_uint8, labels,
                        split=f"ood_{name}", step=step, samples_seen=samples_seen,
                        args=args, mean=mean, std=std, key=fwd_key, subset=subset,
                    )
```

`evaluate_and_dump` already accepts `subset`; make it apply the mask before computing metrics, mirroring `full_finetune_comparison._ood_metrics_from_logits`. Insert immediately before the `write_dump` call:

```python
    if subset is not None:
        keep = np.zeros(logits.shape[-1], dtype=bool)
        keep[np.asarray(subset)] = True
        logits = np.where(keep, logits, -1e9)
```

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run pytest tests/finetune/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/finetune/finetune_jax.py tests/finetune/test_finetune_jax_ood.py
git commit -m "feat(finetune): OOD dumps on the val/test schedule

R/A/V2 in full, preloaded once as uint8, scored on every k-th scheduled
dump and disabled entirely for tuning. The probing arms originally
dumped OOD only at the final step, which made their detection numbers
partly a measurement of CE's overfitting at that step; scheduling OOD
alongside val/test generalises the fix.

A short load raises rather than scoring a plausible AUROC over the wrong
population.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Throughput probe

Runs before any real training. ViT-L at 224 needs ~350 img/s sustained and the PIL/ThreadPoolExecutor path may not clear it; learning that now costs minutes, learning it later costs a node allocation.

**Files:**
- Create: `scripts/finetune/probe_throughput.py`
- Test: `tests/finetune/test_probe_throughput.py`

**Interfaces:**
- Produces: `measure_decode_rate(split, ds_config, img_size, n_images, micro, num_workers) -> float`, `verdict(decode_rate: float, train_rate: float) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/finetune/test_probe_throughput.py`:

```python
"""The probe's verdict logic, which is what a human acts on."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "finetune"))

import probe_throughput as pt  # noqa: E402


def test_verdict_flags_the_loader_as_the_bottleneck():
    v = pt.verdict(decode_rate=120.0, train_rate=350.0)
    assert "loader" in v.lower()
    assert "pre-resized" in v.lower()  # names the actual remedy


def test_verdict_reports_compute_bound_when_the_loader_keeps_up():
    v = pt.verdict(decode_rate=900.0, train_rate=350.0)
    assert "compute" in v.lower()
    assert "loader" not in v.lower().split("compute")[0]


def test_verdict_is_explicit_at_the_margin():
    # Within 20% is not "keeping up" -- the loader stalls the GPU intermittently.
    v = pt.verdict(decode_rate=380.0, train_rate=350.0)
    assert "marginal" in v.lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run pytest tests/finetune/test_probe_throughput.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'probe_throughput'`

- [ ] **Step 3: Write the implementation**

Create `scripts/finetune/probe_throughput.py`:

```python
"""Measure decode and train throughput SEPARATELY, before committing compute.

The fine-tuning campaign is ~117 A100-hours at dinov3_large. If the data loader
cannot sustain the model's consumption rate, every one of those hours runs at
the loader's speed instead -- and that is invisible in a training log, which
shows only the combined step time.
"""

import argparse
import time

import numpy as np

from bllarse.finetune import imagenet

# Within this ratio, the loader intermittently stalls the GPU even though its
# mean rate looks adequate.
MARGIN = 1.2


def measure_decode_rate(split, ds_config, img_size, n_images, micro, num_workers):
    """Images per second through decode+resize alone, no model involved."""
    order = np.random.default_rng(0).permutation(len(split))[:n_images]
    start = time.perf_counter()
    decoded = 0
    for offset in range(0, n_images - micro + 1, micro):
        imagenet.load_images_uint8(
            split, ds_config["img_key"], img_size,
            order[offset:offset + micro], num_workers=num_workers,
        )
        decoded += micro
    elapsed = time.perf_counter() - start
    return decoded / elapsed if elapsed > 0 else float("inf")


def verdict(decode_rate: float, train_rate: float) -> str:
    """What to do about the measured pair. This is the probe's actual output."""
    if decode_rate < train_rate:
        return (
            f"LOADER-BOUND: decode {decode_rate:.0f} img/s < train "
            f"{train_rate:.0f} img/s. The GPU will idle. Remedy: pre-resized "
            f"224px shards, or raise --num-workers if cores are free."
        )
    if decode_rate < train_rate * MARGIN:
        return (
            f"MARGINAL: decode {decode_rate:.0f} img/s barely clears train "
            f"{train_rate:.0f} img/s. Expect intermittent stalls; pre-resized "
            f"shards are still worth it."
        )
    return (
        f"COMPUTE-BOUND: decode {decode_rate:.0f} img/s comfortably clears "
        f"train {train_rate:.0f} img/s. The loader is not the ceiling."
    )


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="imagenet1k")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--n-images", type=int, default=4096)
    p.add_argument("--micro", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--train-rate", type=float, default=None,
                   help="Measured train img/s to compare against. Omit to "
                        "report the decode rate alone.")
    return p


def main():
    args = build_argparser().parse_args()
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "alternatives"))
    from vit_classification import DATASET_CONFIGS
    from datasets import load_dataset

    ds_config = DATASET_CONFIGS[args.dataset]
    split = load_dataset(ds_config["hf_path"], name=ds_config.get("hf_config"))["train"]

    rate = measure_decode_rate(split, ds_config, args.img_size, args.n_images,
                               args.micro, args.num_workers)
    print(f"[probe] decode: {rate:.0f} img/s at {args.img_size}px "
          f"with {args.num_workers} workers")
    if args.train_rate is not None:
        print(f"[probe] {verdict(rate, args.train_rate)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run pytest tests/finetune/ -v`
Expected: PASS, all files

- [ ] **Step 5: Commit**

```bash
git add scripts/finetune/probe_throughput.py tests/finetune/test_probe_throughput.py
git commit -m "feat(finetune): decode-vs-train throughput probe

Measures the two rates separately. A loader that cannot sustain the
model's consumption rate runs the whole ~117 A100-hour campaign at the
loader's speed, and that is invisible in a training log, which shows
only the combined step time.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Phase 1 exit criteria

- `uv run pytest tests/finetune/ -v` passes with `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu`
- The three pre-existing comparison-script test files still pass (Task 3 regression gate)
- `scripts/finetune/finetune_jax.py --arm ce --batch-size 1024 --epochs 0.01` completes on Raven and writes val+test dumps at every scheduled step
- `scripts/analysis/fig1_curves.py` can score the resulting run directory without modification

## Deferred to Phase 2

- `scripts/finetune/finetune_vbll.py` (PyTorch), `transformers` in `bllarse-torch.def`
- `scripts/finetune/check_backbone_parity.py`
- `src/bllarse/tuning/finetune_space.py`, `finetune_worker.py`, and the `seeds=`/`scripts=` kwargs on `objective.evaluate`
- Sweep definitions for the 5-seed full run
- The Fig 5/6/7 + Table 1 analysis and plotting scripts. Nothing in Phase 1
  needs them, and they are cheap to write once real dumps exist — but they are
  the reason the dump contract in Task 5 matters, so they are listed rather
  than assumed.
- ImageNet-Sketch and ObjectNet (spec §10)
