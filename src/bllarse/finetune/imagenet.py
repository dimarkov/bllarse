"""ImageNet-1k facts, the pinned eval split, and the uint8 host cache.

The OOD suite is ImageNet-R, -A, -V2 and -Sketch in FULL -- no subsampling.
Sketch joins the three originals because it is the same KIND of evidence: a
natural-image distribution shift over the full 1000-class label set, which is
the axis Fig 6 is about.

ImageNet-C stays excluded, and the reasoning is unchanged: it measures
synthetic corruption robustness rather than natural shift, and it is the bulk
of the OOD cost. The numbers behind "the bulk" are worth writing down, because
they also rule out the obvious workaround. The only build available is
`Tsomaros/ImageNet-C-{corruption}-severity_5`, i.e. severity 5 ALONE (so
results-plan's "severity-averaged" is not obtainable from it anyway), at 50,000
images x 8 corruptions = 400,000 per OOD pass. As uint8 224x224x3 host arrays
that is ~60 GB against the job's 125 GB allocation, on top of the shard reader
and JAX -- and ~5.7 h of extra evaluation against a 24 h wall. Subsampling
would fit, but subsampling is what this suite is defined to avoid: full sets
are what keep sampling noise out of every point on the OOD trajectory.
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

OOD_KEYS = ("imagenet_r", "imagenet_a", "imagenet_v2", "imagenet_sketch")

# Per-set sizes, for cost arithmetic and for asserting a load was complete.
#
# imagenet_sketch's 50,889 is from the dataset card, NOT verified against a
# local copy -- it is not in the cluster cache yet. If the card is wrong,
# load_ood_sets raises at job start with both numbers rather than scoring a
# short load, which is the behaviour that check exists for.
OOD_SIZES = {"imagenet_r": 30_000, "imagenet_a": 7_500, "imagenet_v2": 10_000,
             "imagenet_sketch": 50_889}

# Host RAM for one OOD pass, at 224x224x3 uint8. Preloaded once and held for
# the whole run (load_ood_sets), so it competes with the shard reader and JAX
# inside the job's --mem. ~14.8 GB for the four sets above.
OOD_PRELOAD_BYTES = sum(OOD_SIZES.values()) * 224 * 224 * 3

# R and A cover 200 ImageNet classes; their out-of-subset logits are masked
# before scoring (full_finetune_comparison._ood_metrics_from_logits).
#
# The mask is a property of the DATASETS, not a choice about evaluation: R and
# A contain images from 200 classes only, so the other 800 cannot occur. It is
# also the protocol the papers introducing them use, which is what makes our
# accuracies comparable to published ones.
#
# It is not free, and the cost lands on exactly what this paper measures.
# Masking renormalises the predictive over 200 classes, so it is an ORACLE
# intervention: it hands the model a fact about the evaluation set it does not
# have at test time, and it changes the distribution whose calibration is being
# scored. NLL, ECE and Brier are all computed on the masked view, and a head
# whose merit is spreading mass sensibly over the full label space has part of
# that merit erased before it is measured. V2 and Sketch span all 1000 classes
# and are scored unmasked, so the suite reports both regimes.
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
