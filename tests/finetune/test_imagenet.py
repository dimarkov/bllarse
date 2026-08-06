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
