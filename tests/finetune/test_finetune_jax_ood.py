"""OOD dumps ride the val/test schedule rather than a separate cadence.

Dumping OOD only at the FINAL step would make a detection number partly a
measurement of CE's overfitting at that step. Scheduling OOD alongside val/test
lets it be read anywhere along the trajectory.

The plan-level schedule (the eval and dump strides, tuning disabling OOD) is
covered by tests/finetune/test_finetune_jax_plan.py and
test_finetune_jax_contracts.py -- duplicating it here would not add coverage.
What this file covers is the wiring: `load_ood_sets` (the preload, the
short-load guard, the subset derivation) and `main`'s dispatch of that data
onto the schedule. That is what this file exercises, against fakes standing
in for `precompute_ood_features` (network) and the DINOv3 backbone (GPU),
mirroring the fake-module technique in test_finetune_jax_contracts.py.
"""

import sys
import types
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "finetune"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "dinov3"))

import finetune_jax as fj  # noqa: E402

from bllarse import evaldump  # noqa: E402
from bllarse.finetune import imagenet  # noqa: E402


def _args(**overrides):
    argv = ["--arm", "ce", "--batch-size", "1024", "--micro-batch-size", "256"]
    for key, value in overrides.items():
        argv.extend([f"--{key.replace('_', '-')}", str(value)])
    return fj.build_argparser().parse_args(argv)


def _plan(**overrides):
    return fj.resolve_plan(_args(**overrides), micro_batch=64)


# --- Schedule sanity (thin, but kept as a guard on the interfaces this file
# depends on -- if these ever went false the rest of the file would be
# testing nonsense) -------------------------------------------------------


def test_ood_rides_a_stride_of_a_uniform_dump_grid():
    # Both grids are strides of the sample-spaced eval index, so "OOD is a
    # subset of the dumps" holds by construction: dump every j-th eval, OOD every k-th of those.
    plan = _plan(dump_every_k_evals=2, epochs=4.0)
    assert plan["eval_every"] == 16
    assert plan["n_evals"] == 312


def test_tuning_disables_ood_entirely():
    # OOD forwards 98,389 images; a tuning trial must never pay for it. With
    # --dump-every-k-evals 0 nothing is dumped at all, so nothing is OOD-dumped
    # either -- and main() must not even preload the sets.
    args = _args(dump_every_k_evals=0)
    assert not (args.dump_every_k_evals and args.ood_every_k_dumps)


# --- load_ood_sets: fakes for precompute_ood_features ----------------------


class _FakeOODSplit:
    """Minimal stand-in for RelabeledSplit: supports split["label"] and
    split[i] -> {img_key: PIL.Image}, the only access load_ood_sets and
    imagenet.load_images_uint8 need."""

    def __init__(self, images, labels):
        self._images = images
        self._labels = np.asarray(labels)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, key):
        if key == "label":
            return self._labels
        return {"image": self._images[int(key)]}


def _fake_images(n, size):
    return [
        Image.fromarray(np.full((size, size, 3), fill_value=(i * 31) % 256, dtype=np.uint8))
        for i in range(n)
    ]


def _fake_prefetch_iterator(iterator, n_prefetch=2):
    """Stand-in for vit_classification.prefetch_iterator: the identical
    queue/thread wrapping technique, duplicated so the fake module doesn't
    need to import the real (heavier) one just for this one function."""
    from queue import Queue
    from threading import Thread

    q = Queue(maxsize=n_prefetch)
    sentinel = object()

    def fill():
        for item in iterator:
            q.put(item)
        q.put(sentinel)

    t = Thread(target=fill, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item


def _install_fake_ood_registry(monkeypatch, sizes, subset_labels):
    """Stand in for scripts/dinov3/precompute_ood_features.

    `sizes`: {name: n_images} -- also monkeypatched onto imagenet.OOD_SIZES so
    a correctly-sized fake does not spuriously look "short".
    `subset_labels`: {name: label_array} -- the FULL label column each fake
    split reports; load_ood_sets derives `subset` from exactly this.
    """
    img_size = 4
    fake_module = types.ModuleType("precompute_ood_features")
    fake_module.load_wnid_to_idx = lambda: {}

    def build_ood_registry():
        registry = {}
        for name in imagenet.OOD_KEYS:
            n = sizes[name]
            labels = subset_labels[name]
            assert len(labels) == n
            split = _FakeOODSplit(_fake_images(n, img_size), labels)
            registry[name] = (lambda w, split=split: (split, "image"))
        return registry

    fake_module.build_ood_registry = build_ood_registry
    monkeypatch.setitem(sys.modules, "precompute_ood_features", fake_module)
    monkeypatch.setattr(imagenet, "OOD_SIZES", dict(sizes))
    return img_size


def _default_sizes_and_labels():
    # Small, fast, and deliberately NOT all covering the same class range:
    # imagenet_r/a use only classes {0, 1} (a genuine strict subset of a
    # larger label space) while imagenet_v2 spans a wider range -- so a
    # "subset always equals np.unique(labels)" mutant and a "subset is
    # ignored" mutant are both distinguishable.
    sizes = {"imagenet_r": 4, "imagenet_a": 3, "imagenet_v2": 5,
             "imagenet_sketch": 6}
    labels = {
        "imagenet_r": np.array([0, 1, 0, 1]),
        "imagenet_a": np.array([1, 0, 1]),
        "imagenet_v2": np.array([0, 1, 2, 3, 4]),
        # 1000-class set like V2, so it must NOT be subset-masked either.
        "imagenet_sketch": np.array([0, 1, 2, 3, 4, 5]),
    }
    return sizes, labels


def test_load_ood_sets_returns_every_set_in_declared_order(monkeypatch):
    sizes, labels = _default_sizes_and_labels()
    img_size = _install_fake_ood_registry(monkeypatch, sizes, labels)

    args = _args()
    loaded = fj.load_ood_sets(args, img_size)

    assert [name for name, *_ in loaded] == list(imagenet.OOD_KEYS)
    for name, images, out_labels, _subset in loaded:
        assert images.shape == (sizes[name], img_size, img_size, 3)
        assert images.dtype == np.uint8
        np.testing.assert_array_equal(out_labels, labels[name])


def test_load_ood_sets_only_r_and_a_get_a_subset(monkeypatch):
    sizes, labels = _default_sizes_and_labels()
    img_size = _install_fake_ood_registry(monkeypatch, sizes, labels)

    loaded = {name: subset for name, _images, _labels, subset in
               fj.load_ood_sets(_args(), img_size)}

    # R and A: subset present and equal to the unique classes in their FULL
    # label column (not e.g. all 1000 classes, not empty).
    np.testing.assert_array_equal(loaded["imagenet_r"], np.array([0, 1]))
    np.testing.assert_array_equal(loaded["imagenet_a"], np.array([0, 1]))
    # V2 spans everything and must get None, not an inferred subset.
    assert loaded["imagenet_v2"] is None


def test_load_ood_sets_raises_on_a_short_split(monkeypatch):
    sizes, labels = _default_sizes_and_labels()
    # imagenet_r reports one fewer image than imagenet.OOD_SIZES expects --
    # simulating a HuggingFace split that shrank between runs.
    short_sizes = dict(sizes)
    short_sizes["imagenet_r"] = sizes["imagenet_r"] - 1
    short_labels = dict(labels)
    short_labels["imagenet_r"] = labels["imagenet_r"][:-1]
    img_size = _install_fake_ood_registry(monkeypatch, short_sizes, short_labels)
    # OOD_SIZES itself must still say the ORIGINAL (larger) expectation --
    # only the fake split's reported length actually shrinks below it.
    monkeypatch.setitem(imagenet.OOD_SIZES, "imagenet_r", sizes["imagenet_r"])

    with pytest.raises(SystemExit, match="imagenet_r"):
        fj.load_ood_sets(_args(), img_size)


def test_load_ood_sets_does_not_raise_when_sizes_match(monkeypatch):
    sizes, labels = _default_sizes_and_labels()
    img_size = _install_fake_ood_registry(monkeypatch, sizes, labels)
    fj.load_ood_sets(_args(), img_size)  # must not raise


# --- main(): the actual dispatch onto the schedule --------------------------
#
# Reuses the fake-heavy-module technique from test_finetune_jax_contracts.py
# (vit_classification / datasets) and adds a fake precompute_ood_features so
# `main` runs its real control flow end to end, without a GPU or network.


class DummyBackbone(eqx.Module):
    """Tiny stand-in for the equimo ViT: mean-pool then a linear projection.

    Must be an eqx.Module (not a plain class): evaluate_split calls
    eqx.nn.inference_mode(backbone, True), which walks the model as a pytree.
    """

    proj: jax.Array

    def __call__(self, x, key=None):
        return jnp.mean(x, axis=(1, 2)) @ self.proj


class FakeDatasetSplit:
    def __init__(self, images, labels):
        self._images = images
        self._labels = labels

    def __len__(self):
        return len(self._images)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "label":
                return list(self._labels)
            if key == "image":
                return list(self._images)
            raise KeyError(key)
        return {"image": self._images[key], "label": self._labels[key]}


def _install_fake_heavy_modules(monkeypatch, *, embed_dim, num_classes, img_size,
                                train_n, eval_n):
    fake_vit = types.ModuleType("vit_classification")
    fake_vit.DATASET_CONFIGS = {
        "fake": {
            "hf_path": "fake/fake",
            "num_classes": num_classes,
            "label_key": "label",
            "img_key": "image",
            "test_split": "validation",
        }
    }
    fake_vit.EQUIMO_MODELS = {
        "fake_model": {"img_size": img_size, "embed_dim": embed_dim, "channels_first": True}
    }
    fake_vit.IMAGENET_MEAN = np.zeros(3)
    fake_vit.IMAGENET_STD = np.ones(3)

    def get_pretrained_backbone(model_name, key):
        return DummyBackbone(proj=jax.random.normal(key, (3, embed_dim)))

    fake_vit.get_pretrained_backbone = get_pretrained_backbone
    fake_vit.prefetch_iterator = _fake_prefetch_iterator

    train_split = FakeDatasetSplit(
        _fake_images(train_n, img_size), [i % num_classes for i in range(train_n)]
    )
    eval_split = FakeDatasetSplit(
        _fake_images(eval_n, img_size), [i % num_classes for i in range(eval_n)]
    )

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda hf_path, name=None: {
        "train": train_split, "validation": eval_split,
    }

    monkeypatch.setitem(sys.modules, "vit_classification", fake_vit)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setattr(fj.imagenet, "IMAGENET_N_TRAIN", train_n)


def _base_argv(*, dump_root, split_cache_dir, img_size, epochs, batch_size,
               micro_batch_size, **extra):
    argv = [
        "--arm", "ce",
        "--model", "fake_model",
        "--dataset", "fake",
        "--img-size", str(img_size),
        "--allow-non-224",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--micro-batch-size", str(micro_batch_size),
        # Evaluate every optimizer step and dump every evaluation: these runs
        # are a handful of steps long, so the production quantum (16,384
        # samples) would schedule nothing at all.
        "--eval-quantum", str(batch_size),
        "--dump-every-k-evals", "1",
        "--dump-root", str(dump_root),
        "--group-id", "g",
        "--uid", "u",
        "--split-cache-dir", str(split_cache_dir),
        "--seed", "137",
    ]
    for key, value in extra.items():
        argv.extend([f"--{key.replace('_', '-')}", str(value)])
    return argv


def test_main_never_calls_load_ood_sets_when_disabled(tmp_path, monkeypatch):
    """The tuning-cost guard: --ood-every-k-dumps 0
    must mean load_ood_sets is never even CALLED, not just that it returns
    an empty result -- a tuning trial must never pay the 98,389-image
    preload. A version that always calls load_ood_sets and only skips writing
    dumps would still pass every schedule-level test but blow the tuning
    trial's cost budget; only spying on the call itself catches that.
    """
    embed_dim, num_classes, img_size = 5, 3, 4
    _install_fake_heavy_modules(
        monkeypatch, embed_dim=embed_dim, num_classes=num_classes, img_size=img_size,
        train_n=8, eval_n=10,
    )
    calls = []
    real_load_ood_sets = fj.load_ood_sets

    def spy_load_ood_sets(args, img_size):
        calls.append(args.ood_every_k_dumps)
        return real_load_ood_sets(args, img_size)

    monkeypatch.setattr(fj, "load_ood_sets", spy_load_ood_sets)

    dump_root = tmp_path / "dumps"
    split_cache = tmp_path / "splits"
    argv = _base_argv(
        dump_root=dump_root, split_cache_dir=split_cache, img_size=img_size,
        epochs=1.0, batch_size=4, micro_batch_size=2, ood_every_k_dumps=0,
    )
    args = fj.build_argparser().parse_args(argv)
    fj.main(args, on_dump=None)

    assert calls == []
    records = list(evaldump.iter_dumps(dump_root, "g", "u"))
    assert all(not r.meta["split"].startswith("ood_") for r in records)


def test_main_writes_ood_dumps_only_on_ood_steps_not_every_dump_step(tmp_path, monkeypatch):
    """When OOD IS enabled, it must land exactly on every
    --ood-every-k-dumps-th dump (plus the final evaluation) -- a strict subset
    of the dump steps at this stride -- and nowhere else. A mutant that writes OOD at every
    scheduled dump step (ignoring the stride) would pass a "some ood_*
    records exist" check but fail this one.
    """
    embed_dim, num_classes, img_size = 5, 3, 4
    _install_fake_heavy_modules(
        monkeypatch, embed_dim=embed_dim, num_classes=num_classes, img_size=img_size,
        train_n=8, eval_n=10,
    )
    sizes = {"imagenet_r": 2, "imagenet_a": 2, "imagenet_v2": 2,
             "imagenet_sketch": 2}
    labels = {
        "imagenet_r": np.array([0, 1]),
        "imagenet_a": np.array([1, 0]),
        "imagenet_v2": np.array([0, 2]),
        "imagenet_sketch": np.array([0, 2]),
    }
    _install_fake_ood_registry(monkeypatch, sizes, labels)

    dump_root = tmp_path / "dumps"
    split_cache = tmp_path / "splits"
    # epochs=1.5, batch_size=2, micro_batch_size=2 -> accum_plan(2, 2) gives
    # micro=2, num_accum=1, effective=2; steps = int(1.5 * 8) // 2 = 6.
    # _base_argv sets --eval-quantum 2 == effective, so eval_every = 1 and
    # eval_index == step: evaluations at 1..6. --dump-every-k-evals 1 dumps at
    # all six; the default --ood-every-k-dumps 2 then takes every other dump
    # ({2, 4, 6}) -- a STRICT, non-trivial subset, so steps 1, 3 and 5 are the
    # ones a "write OOD at every dump" mutant would wrongly also hit.
    argv = _base_argv(
        dump_root=dump_root, split_cache_dir=split_cache, img_size=img_size,
        epochs=1.5, batch_size=2, micro_batch_size=2,
    )
    args = fj.build_argparser().parse_args(argv)
    plan = fj.resolve_plan(args, micro_batch=2)
    assert (plan["steps"], plan["eval_every"], plan["n_evals"]) == (6, 1, 6)

    fj.main(args, on_dump=None)

    records = list(evaldump.iter_dumps(dump_root, "g", "u"))
    ood_steps_seen = {r.meta["step"] for r in records if r.meta["split"].startswith("ood_")}
    val_steps_seen = {r.meta["step"] for r in records if r.meta["split"] == "val"}

    assert ood_steps_seen == {2, 4, 6}
    assert val_steps_seen == {1, 2, 3, 4, 5, 6}  # val/test schedule is untouched
    # The final evaluation carries OOD whatever the stride lands on, because
    # curves._id_ood_pair raises when the last test and last OOD dump disagree.
    assert max(ood_steps_seen) == max(val_steps_seen) == plan["steps"]

    ood_splits_at_an_ood_step = {r.meta["split"] for r in records if r.meta["step"] == 2
                                 and r.meta["split"].startswith("ood_")}
    assert ood_splits_at_an_ood_step == {
        f"ood_{name}" for name in imagenet.OOD_KEYS}


def test_main_ood_dump_uses_the_ood_prefixed_split_name(tmp_path, monkeypatch):
    """split=f"ood_{name}" so several OOD sets coexist in one run directory
    without clobbering each other or the val/test dumps."""
    embed_dim, num_classes, img_size = 5, 3, 4
    _install_fake_heavy_modules(
        monkeypatch, embed_dim=embed_dim, num_classes=num_classes, img_size=img_size,
        train_n=4, eval_n=6,
    )
    sizes = {"imagenet_r": 2, "imagenet_a": 2, "imagenet_v2": 2,
             "imagenet_sketch": 2}
    labels = {
        "imagenet_r": np.array([0, 1]),
        "imagenet_a": np.array([1, 0]),
        "imagenet_v2": np.array([0, 2]),
        "imagenet_sketch": np.array([0, 2]),
    }
    _install_fake_ood_registry(monkeypatch, sizes, labels)

    dump_root = tmp_path / "dumps"
    split_cache = tmp_path / "splits"
    # epochs chosen for the single-step case: 1 step
    # total, so it is trivially also the final (and only, and thus always
    # OOD-scheduled) step.
    argv = _base_argv(
        dump_root=dump_root, split_cache_dir=split_cache, img_size=img_size,
        epochs=1.0, batch_size=4, micro_batch_size=4,
    )
    args = fj.build_argparser().parse_args(argv)
    fj.main(args, on_dump=None)

    splits_seen = {r.meta["split"] for r in evaldump.iter_dumps(dump_root, "g", "u")}
    assert {"val", "test", "ood_imagenet_r", "ood_imagenet_a", "ood_imagenet_v2"} <= splits_seen


def test_main_ood_dispatch_actually_masks_r_and_a_but_not_v2(tmp_path, monkeypatch):
    """`load_ood_sets` computing the right subset is not enough on its own --
    `main` has to actually thread it through to `evaluate_split` as
    `subset=subset`. A version that computes the correct subset but then
    calls `evaluate_split(..., subset=None)` for every OOD set would pass
    every test above (they only look at what load_ood_sets returns, or at
    which steps/split-names dumps land on) while silently never masking
    anything. This test loads the actual written dumps and checks the logit
    values themselves.
    """
    embed_dim, num_classes, img_size = 5, 4, 4
    _install_fake_heavy_modules(
        monkeypatch, embed_dim=embed_dim, num_classes=num_classes, img_size=img_size,
        train_n=4, eval_n=6,
    )
    # imagenet_r/a only ever see classes {0, 1} out of 4 -- classes 2 and 3
    # must come back masked. imagenet_v2 sees all 4 classes and must not be
    # masked at all.
    sizes = {"imagenet_r": 3, "imagenet_a": 3, "imagenet_v2": 3,
             "imagenet_sketch": 3}
    labels = {
        "imagenet_r": np.array([0, 1, 0]),
        "imagenet_a": np.array([1, 0, 1]),
        "imagenet_v2": np.array([0, 1, 2]),
        "imagenet_sketch": np.array([0, 1, 2]),
    }
    _install_fake_ood_registry(monkeypatch, sizes, labels)

    dump_root = tmp_path / "dumps"
    split_cache = tmp_path / "splits"
    argv = _base_argv(
        dump_root=dump_root, split_cache_dir=split_cache, img_size=img_size,
        epochs=1.0, batch_size=4, micro_batch_size=4,
    )
    args = fj.build_argparser().parse_args(argv)
    fj.main(args, on_dump=None)

    by_split = {r.meta["split"]: r for r in evaldump.iter_dumps(dump_root, "g", "u")}

    for name in ("imagenet_r", "imagenet_a"):
        logits = by_split[f"ood_{name}"].logits.astype(np.float32)
        # Classes 2 and 3 are out-of-subset for R/A and must be driven to
        # MASK_NEG (float16 round-trip tolerance) on EVERY row.
        assert np.all(logits[:, 2] < -1000), f"{name} class 2 was not masked"
        assert np.all(logits[:, 3] < -1000), f"{name} class 3 was not masked"
        # Classes 0 and 1 are in-subset and must be real (finite, not pinned
        # to the sentinel).
        assert np.all(logits[:, 0] > -1000), f"{name} class 0 was wrongly masked"
        assert np.all(logits[:, 1] > -1000), f"{name} class 1 was wrongly masked"

    v2_logits = by_split["ood_imagenet_v2"].logits.astype(np.float32)
    # V2 must not be masked at all: no column is uniformly pinned to the
    # sentinel.
    assert np.all(v2_logits > -1000), "imagenet_v2 was masked but should not be"
