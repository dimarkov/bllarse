"""OOD eval helpers ported from ood_evaluation.py into full_finetune_comparison.

Only the PURE pieces are unit-tested here (the backbone forward + HF loading are
integration glue verified separately): the Brier score, the subset-masked
4-metric report (acc/ece/nll/brier) with NO temperature scaling, and the OOD-key
resolution (default = the two cheapest datasets).
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "dinov3"))

import full_finetune_comparison as ffc  # noqa: E402


def test_compute_brier_perfect_and_uniform():
    labels = jnp.array([0, 1])
    confident = jnp.array([[100.0, -100.0], [-100.0, 100.0]])  # p ~ one-hot correct
    assert float(ffc.compute_brier(confident, labels)) == pytest.approx(0.0, abs=1e-6)
    # uniform p=[.5,.5]: each row (.5-1)^2 + (.5-0)^2 = .5
    assert float(ffc.compute_brier(jnp.zeros((2, 2)), labels)) == pytest.approx(0.5, abs=1e-6)


def test_ood_metrics_keys_are_the_four_no_temperature():
    logits = jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    labels = jnp.array([0, 1, 2])
    m = ffc._ood_metrics_from_logits(logits, labels, subset=None)
    assert set(m) == {"acc", "ece", "nll", "brier"}
    assert m["acc"] == pytest.approx(1.0)
    assert all(jnp.isfinite(jnp.asarray(v)) for v in m.values())
    assert not any("ts" in k or "temp" in k for k in m)  # no post-hoc TS leaked in


def test_ood_subset_masking_restricts_argmax_to_subset():
    # Every row most favors class 3, which is OUTSIDE the {0,1,2} subset; the
    # additive mask must drop it so the prediction falls back into the subset.
    logits = jnp.array([
        [5.0, 0.0, 0.0, 9.0],
        [0.0, 5.0, 0.0, 9.0],
        [0.0, 0.0, 5.0, 9.0],
    ])
    labels = jnp.array([0, 1, 2])
    masked = ffc._ood_metrics_from_logits(logits, labels, subset=jnp.array([0, 1, 2]))
    assert masked["acc"] == pytest.approx(1.0)
    unmasked = ffc._ood_metrics_from_logits(logits, labels, subset=None)
    assert unmasked["acc"] == pytest.approx(0.0)  # argmax = class 3 everywhere


def test_ood_key_registry_matches_source():
    # The ported key list must cover the 8 ImageNet-C corruptions + R/A/V2, with
    # R/A flagged as the masked 200-class subsets.
    assert len(ffc.IMAGENET_C_KEYS) == 8
    assert ffc.OOD_KEYS == ffc.IMAGENET_C_KEYS + ["imagenet_r", "imagenet_a", "imagenet_v2"]
    assert ffc.SUBSET_KEYS == {"imagenet_r", "imagenet_a"}


def test_default_ood_keys_are_the_two_cheapest():
    # All 8 ImageNet-C sets are 50k (tied) -> jpeg_compression by choice; shortest
    # size-variant is imagenet_a (7.5k).
    assert ffc.DEFAULT_OOD_KEYS == ["imagenet_c_jpeg_compression", "imagenet_a"]
    assert all(k in ffc.OOD_KEYS for k in ffc.DEFAULT_OOD_KEYS)


def test_resolve_ood_keys():
    assert ffc.resolve_ood_keys(None) == ffc.DEFAULT_OOD_KEYS
    assert ffc.resolve_ood_keys(["all"]) == ffc.OOD_KEYS
    assert ffc.resolve_ood_keys(["imagenet_a", "imagenet_v2"]) == ["imagenet_a", "imagenet_v2"]
    with pytest.raises(ValueError):
        ffc.resolve_ood_keys(["not_a_dataset"])
