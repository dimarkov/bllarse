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
