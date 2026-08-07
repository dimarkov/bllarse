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
    # The official Flowers-102 training split, 10 images per class. NOT the
    # 7,169-image pool the upstream config calls "train" -- that is val+test and
    # is now the eval pool. See bllarse.features._TRAIN_SPLIT.
    "flowers102": 1_020,
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

# The tunable arms. Held here rather than inferred from the training scripts so
# the search space stays inspectable without importing one, and pinned against
# objective.SCRIPTS by a test: an arm added to one and not the other gives a
# worker that accepts --arm and then dies inside its first trial.
ARMS = ("adamw_ce", "vbll", "ibprobit")

# IBProbit's CAVI iteration counts. Categorical, not suggest_int: 192 is 1.5x128
# rather than a doubling, so the ladder is not geometric and an integer range
# would spend trials on values nobody asked to compare.
IBPROBIT_UPDATE_ITERS = (32, 64, 128, 192)


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


def _suggest_ibprobit(trial: Any, dataset: str) -> dict[str, Any]:
    """IBProbit's axes -- disjoint from the other two arms' except for the
    shared batch ladder.

    NO learning_rate and NO weight_decay. The head is assimilated in closed
    form by `IBProbit.update` and linear probing freezes the backbone, so this
    arm runs no optimizer at all: those two axes have nothing to act on, and
    sampling them would spend a third of the campaign distinguishing
    configurations that come out bit-identical.

    FIXED, not searched:
      * `kappa` at the class default 0.5, by decision.
      * `px` on -- PX shares the plain CAVI fixed point exactly (see
        `_cavi_scan_px`) and only reaches it in fewer iterations, so it is a
        solver form rather than a model hyperparameter.
      * `learn_q` on -- the Inverse-Wishart recursion is the method under test.
        Turning it off is an ablation, and an ablation belongs in its own runs
        rather than as an axis TPE can quietly collapse.
    """
    return {
        "batch_size": trial.suggest_categorical("batch_size", batch_sizes(dataset)),
        # q0 DIRECTLY, not alpha. The class derives q0 = alpha^2 / B by default,
        # and B is tuned here alongside it -- so a table of tuned alpha is not a
        # table of process noise: two datasets whose alpha agrees sit at
        # different q0 whenever their batch sizes differ, and process noise
        # against (num_classes, num_update_iters) is exactly what this campaign
        # exists to read. alpha stays recoverable afterwards as sqrt(q0 * B).
        #
        # The box brackets 1/D = 7.8e-4 (the pre-Wishart operating point) with
        # two decades of room above and below. Its floor is a decade and a half
        # above the measured-inert q0 = 3.1e-8, where learn_q on and off give
        # bit-identical mu over 40 steps -- trials sampled down there would be
        # comparing configurations that do not differ.
        "q_init": trial.suggest_float("q_init", 1e-6, 1e-1, log=True),
        # Beta(1, 1/eps_init) prior mean on the contamination rate. A genuine
        # model knob: it sets how readily the head discounts a badly-predicted
        # point, and the datasets here span very different label-noise levels.
        "eps_init": trial.suggest_float("eps_init", 1e-3, 1e-1, log=True),
        # A TPE axis rather than four separate studies: every dataset already
        # gets its own study, so K is held fixed within one and a tuned
        # iteration count that VARIES across datasets is itself the
        # K-dependence signal. Read it with care -- TPE's per-level sampling
        # noise can move the winner by one rung on its own.
        "num_update_iters": trial.suggest_categorical(
            "num_update_iters", list(IBPROBIT_UPDATE_ITERS)
        ),
    }


def suggest(trial: Any, arm: str, dataset: str) -> dict[str, Any]:
    """One TPE-sampled configuration.

    The AdamW axes are shared by adamw_ce and vbll because VBLL's head is also
    optimised by AdamW; IBProbit's are disjoint (see `_suggest_ibprobit`).
    `parameterization` is fixed to diagonal and `cov_rank` is NOT searched: rank
    buys calibration with memory, and neither AdamW+CE nor IBProbit has an
    equivalent knob, so letting TPE spend it freely would let VBLL win Fig 1
    partly by being larger. Rank is reported separately as an appendix Pareto
    front.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")

    if arm == "ibprobit":
        return _suggest_ibprobit(trial, dataset)

    # Lower bound raised from 1e-4 to 1e-3 after the pilot: at lr=1e-4, every
    # dataset on both arms still had its val_nll minimum sitting at or within
    # a hair of the final dumped step (still improving 1.5-2.6 nats over the
    # last quarter of the run), while lr=3e-2 converged early everywhere
    # (minima at steps 5-904, mildly overfitting past the optimum). 1e-4 lost
    # to 3e-2 on all 7 datasets on both arms, so keeping it in the range would
    # only inflate the pilot's step budget for a rate the data rules out.
    params: dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", batch_sizes(dataset)),
    }
    if arm == "vbll":
        # kl_ratio, not eta, is the first axis. vbll's loss carries the KL at
        # regularization_weight = eta / N, and the shrinkage of the posterior
        # mean goes as ||m||^2 * eta / (N * prior_scale) -- so eta and
        # prior_scale are degenerate in the mean and differ only in the width
        # they leave behind. Searching them as independent axes spends trials
        # sliding along that diagonal ridge; searching the ratio directly makes
        # the axes near-orthogonal, one for fit and one for width.
        #
        # The first campaign fixed eta = 1 and searched prior_scale in
        # [1e-2, 1e2], which confines the ratio to [1e-2, 1e2]. The measured
        # optimum is ~1e-5, so the box excluded the answer by three orders of
        # magnitude and every one of the seven VBLL studies pinned prior_scale
        # to its ceiling. [1e-7, 1e-2] brackets 1e-5 by two decades below and
        # three above; 1e-4 was already 1.2 points worse and 1e-3 6.3 points.
        params["kl_ratio"] = trial.suggest_float("kl_ratio", 1e-7, 1e-2, log=True)
        # Widened one decade at each end. At a fixed ratio, posterior sd grows
        # as sqrt(prior_scale), so this spans roughly sd 0.06 to 6.
        params["prior_scale"] = trial.suggest_float("prior_scale", 1e-1, 1e3, log=True)
        params["wishart_scale"] = trial.suggest_float("wishart_scale", 1e-3, 1e1, log=True)
    return params
