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


def test_normalize_chunk_distinguishes_h_and_w_axes():
    # Non-square input (1, H=2, W=3, C=3) with position-dependent values.
    # If H and W axes are swapped in transpose, output shape changes and values differ.
    images = np.zeros((1, 2, 3, 3), dtype=np.uint8)
    for h in range(2):
        for w in range(3):
            images[0, h, w, :] = (10 * h + w) % 256

    out = imagenet.normalize_chunk(images, mean=np.zeros(3), std=np.ones(3))
    assert out.shape == (1, 3, 2, 3)

    # Expected output after transpose(0, 3, 1, 2): (1, 3, 2, 3)
    # Channel dimension expanded from last axis, so all channels have the same values
    # (since all channels were filled with the same position-dependent value).
    expected = np.zeros((1, 3, 2, 3), dtype=np.float32)
    for h in range(2):
        for w in range(3):
            val = float((10 * h + w) % 256) / 255.0
            for c in range(3):
                expected[0, c, h, w] = val

    np.testing.assert_allclose(out, expected)


def test_normalize_chunk_applies_mean_and_std():
    images = np.full((1, 4, 4, 3), 255, dtype=np.uint8)
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.25, 0.25, 0.25])
    out = imagenet.normalize_chunk(images, mean=mean, std=std)
    assert out == pytest.approx(np.full_like(out, 2.0))


def test_ood_sizes_and_keys_define_full_suite():
    # Verify the OOD suite composition: R 30k + A 7.5k + V2 10k = 47,500 images total.
    # OOD_SIZES must match OOD_KEYS and sum to 47,500.
    assert set(imagenet.OOD_SIZES.keys()) == set(imagenet.OOD_KEYS)
    assert imagenet.OOD_SIZES["imagenet_r"] == 30_000
    assert imagenet.OOD_SIZES["imagenet_a"] == 7_500
    assert imagenet.OOD_SIZES["imagenet_v2"] == 10_000
    assert sum(imagenet.OOD_SIZES[k] for k in imagenet.OOD_KEYS) == 47_500


def test_ood_subset_keys_excludes_v2():
    # R and A cover 200 ImageNet classes; logits outside these 200 are masked.
    # V2 spans all 1000 classes and must never be logit-masked.
    assert imagenet.OOD_SUBSET_KEYS == frozenset({"imagenet_r", "imagenet_a"})
    assert "imagenet_v2" not in imagenet.OOD_SUBSET_KEYS


def test_memory_budget_of_the_real_preload_peak_stays_under_slurm_mem():
    # M3: the full OOD suite alone (R 30k + A 7.5k + V2 10k = 47,500 images at
    # 224px, ~6.66 GiB uint8) is NOT the real peak a finetune_jax.py run
    # holds. It also keeps the val+test uint8 cache resident for the whole
    # run (IMAGENET_N_VAL = 50,000 images, same 224px frame, ~7.0 GiB) --
    # load_ood_sets and the val/test preload in main() are BOTH alive at
    # once, never freed against each other. On top of that,
    # imagenet.load_images_uint8's `np.stack(list(pool.map(...)))` holds the
    # source Python list of per-image arrays AND the destination stacked
    # array simultaneously until the list is garbage collected -- a
    # transient DOUBLE copy of whichever set is decoding at that moment. The
    # worst case is the largest single set (imagenet_r, 30k images)
    # decoding last, stacked on top of everything already resident
    # (val+test + every other OOD set already loaded).
    size = 224
    bytes_per_image = size * size * 3  # uint8, one image
    ood_n = sum(imagenet.OOD_SIZES[k] for k in imagenet.OOD_KEYS)
    assert ood_n == 47_500
    val_test_n = imagenet.IMAGENET_N_VAL  # 50,000

    resident_gb = (val_test_n + ood_n) * bytes_per_image / 1024**3
    largest_single_set_n = max(imagenet.OOD_SIZES[k] for k in imagenet.OOD_KEYS)
    transient_double_copy_gb = largest_single_set_n * bytes_per_image / 1024**3
    peak_gb = resident_gb + transient_double_copy_gb

    assert peak_gb < 40.0, (
        f"real preload peak {peak_gb:.1f} GiB would OOM slurm_tune.sh's --mem=40G"
    )
    # Sanity: this is genuinely close to the limit (unlike the OOD-suite-only
    # figure, ~6.66 GiB, which left the real risk invisible), not slack.
    assert peak_gb > 12.0
    # And the float32 equivalent of just resting memory (no double-copy)
    # still does not fit, which is why the cache is uint8 at all.
    assert resident_gb * 4 > 40.0
