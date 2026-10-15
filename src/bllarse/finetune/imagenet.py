"""ImageNet-1k facts, the pinned eval split, and the uint8 host cache.

The OOD suite is ImageNet-R, -A, -V2 and -Sketch in FULL -- no subsampling.
All four are the same KIND of evidence: natural-image distribution shift over
the ImageNet-1k label set, which is the axis Fig 4 is about.

ImageNet-C is excluded: it measures synthetic corruption robustness rather
than natural shift, and it would be the bulk of the OOD cost. The numbers
behind "the bulk" are worth writing down, because they also rule out the
obvious workaround. The only build available is
`Tsomaros/ImageNet-C-{corruption}-severity_5`, i.e. severity 5 ALONE (so a
severity-averaged score is not obtainable from it anyway), at 50,000
images x 8 corruptions = 400,000 per OOD pass. As uint8 224x224x3 host arrays
that is ~60 GB against a 125 GB job allocation, on top of the shard reader
and JAX -- and ~5.7 h of extra evaluation against a 24 h wall. Subsampling
would fit, but subsampling is what this suite is defined to avoid: full sets
are what keep sampling noise out of every point on the OOD trajectory.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
from PIL import Image

from bllarse import splits

IMAGENET_N_TRAIN = 1_281_167
IMAGENET_N_VAL = 50_000

# 0.2 of the official 50k validation split -> 10k val / 40k test. Val is used
# ONLY to fit temperature and to select the reporting step; test is never seen
# by either.
DEFAULT_VAL_FRACTION = 0.2

# The canonical WNID -> ImageNet-1k index map, as a CHECKED-IN CONSTANT.
#
# Shipped rather than fetched so that loading an OOD set needs no network, and
# `prepare_ood_cache.py --check-only` can report on a cold cache.
#
# No locally cached dataset can replace it: ILSVRC/imagenet-1k carries
# human-readable names rather than WNIDs, the benjamin-paine mirror has 1001
# classes so its indices are not canonical, and barkermrl/imagenet-a has real
# WNIDs but only the 200-class subset. The ordering is sorted-WNID, fixed
# since 2012, and index 0 is n01440764 (tench), matching ImageNet-1k's own
# first label.
WNID_FILE = Path(__file__).with_name("imagenet_wnids.txt")


@lru_cache(maxsize=1)
def canonical_wnids() -> tuple[str, ...]:
    """The 1000 WNIDs in canonical (sorted) order.

    Validated on load rather than trusted. A truncated or reordered file would
    silently relabel every OOD set -- producing a plausible accuracy over
    scrambled classes, which is exactly the failure that is invisible
    downstream.
    """
    wnids = tuple(WNID_FILE.read_text().split())
    if len(wnids) != 1000:
        raise ValueError(f"{WNID_FILE} has {len(wnids)} WNIDs, expected 1000")
    if len(set(wnids)) != 1000:
        raise ValueError(f"{WNID_FILE} contains duplicates")
    if list(wnids) != sorted(wnids):
        raise ValueError(f"{WNID_FILE} is not in sorted-WNID order")
    bad = [w for w in wnids if not (len(w) == 9 and w[0] == "n" and w[1:].isdigit())]
    if bad:
        raise ValueError(f"{WNID_FILE} has malformed entries: {bad[:3]}")
    return wnids


def load_wnid_to_idx() -> dict[str, int]:
    """WNID -> canonical ImageNet-1k index. Offline, no dataset required."""
    return {w: i for i, w in enumerate(canonical_wnids())}


OOD_KEYS = ("imagenet_r", "imagenet_a", "imagenet_v2", "imagenet_sketch")

# Per-set sizes, for cost arithmetic and for asserting a load was complete.
#
# imagenet_sketch's 50,889 is from the dataset card. If the card is wrong,
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

# fp16-safe mask sentinel, shared by both arms so they cannot mask differently.
#
# evaldump.write_dump stores logits as float16 (max finite magnitude 65504) and
# float16(-1e9) OVERFLOWS to -inf: entropy_ood_score and fit_temperature on such
# a row silently return NaN while msp_ood_score still looks healthy, so a figure
# would show a working MSP column beside a silently NaN entropy one. -1e4 is
# ~340 sigma below any real logit and round-trips as a large finite number.
MASK_NEG = -1e4


def load_ood_sets(img_size: int):
    """Preload every OOD set as uint8 host arrays, once.

    Returns [(name, images_uint8, labels, subset_or_None)]. `subset` is that
    set's class list, used to mask out-of-subset logits before scoring; V2 and
    Sketch span all 1000 classes and get None.

    Framework-agnostic and shared BY BOTH ARMS. It lives here rather than in
    finetune_jax so the torch arm can reach it without importing jax -- an arm
    that could not load the suite could not be scored on it, and VBLL would
    silently have no OOD at all.
    """
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[3]
                           / "scripts" / "dinov3"))
    from precompute_ood_features import build_ood_registry

    wnid_to_idx = load_wnid_to_idx()
    registry = build_ood_registry()
    loaded = []
    for name in OOD_KEYS:
        split, img_key = registry[name](wnid_to_idx)
        labels = np.asarray(split["label"])
        expected = OOD_SIZES[name]
        if len(labels) != expected:
            # A short (or long) load means the HF split changed shape. Scoring
            # it anyway produces a plausible AUROC over the wrong population,
            # so this raises rather than warns -- before the expensive decode.
            raise SystemExit(
                f"[ood] {name} has {len(labels)} images, expected {expected}")
        subset = np.unique(labels) if name in OOD_SUBSET_KEYS else None
        images = load_images_uint8(split, img_key, img_size,
                                   np.arange(len(labels)))
        loaded.append((name, images, labels, subset))
        print(f"[ood] preloaded {name}: {len(labels)} images"
              + (f", {len(subset)}-class subset" if subset is not None else ""))
    return loaded


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

    At full OOD-suite size (98,389 images) this is 14.8 GB. The float32
    equivalent is 59 GB, which does not fit alongside the model under a job's
    memory request -- so the dtype here is a requirement, not a tuning choice.
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
