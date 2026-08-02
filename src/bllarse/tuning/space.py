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
