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
    # serves as the eval pool. See bllarse.features._TRAIN_SPLIT.
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

# Ceiling on the ladder, independent of dataset size. Only food101 (75,750) is
# large enough to reach past it -- its MIN_STEPS_PER_EPOCH bound alone would
# allow 32,768, twice any other dataset's ceiling. Two reasons to cut it:
#
#   * It is the eval quantum (see `eval_quantum`), and the quantum sets how
#     much data passes between two consecutive val_nll measurements. Letting
#     one dataset's quantum run to 32,768 makes its curve twice as coarse as
#     everyone else's for no gain.
#   * At a fixed step budget samples_seen = steps * batch_size, so the top of
#     the ladder is also the most expensive rung AND the one that consumes the
#     most data. Measured on food101, trials at batch 16,384 scored 0.1861
#     against 0.1588 at batch 2,048 -- 17% worse on 8x the data.
MAX_BATCH = 16_384

# The tunable arms. Held here rather than inferred from the training scripts so
# the search space stays inspectable without importing one, and pinned against
# objective.SCRIPTS by a test: an arm added to one and not the other gives a
# worker that accepts --arm and then dies inside its first trial.
ARMS = ("adamw_ce", "vbll", "ibprobit", "ibprobit_noeps")

# The eps-free IBProbit arm. `ibprobit_noeps` is the same head and the same
# training script as `ibprobit`; it differs only in that eps_init is PINNED here rather than
# searched, so the mixture floor is numerically absent and the head is scored
# on its own predictive alone.
#
# 1e-12, not 0.0. `IBProbit._eps_seed` reads
#     return 1.0, (1.0 / self.eps_init if self.eps_init > 0.0 else 100.0)
# so eps_init = 0.0 falls through its own guard to b0 = 100 -- a Beta(1, 100)
# whose mean is 0.0099, i.e. the MIDDLE of the searched box, and measured to
# drift to 0.16 within 32 updates. Zero is also structurally unreachable: the
# recursion is `eps_alpha = a0 + rho*(eps_alpha - a0) + sum(1 - r)` with a0
# pinned at 1.0, so eps > 0 always. 1e-12 gives b0 = 1e12, against which the
# recursion's `+sum(r)` term (at most one batch) is entirely lost in float32 --
# measured to hold eps at exactly 1e-12 over 32 updates on 10%-noisy labels,
# with finite logits and NLL. 1e-16 behaves identically; 1e-12 keeps headroom.
IBPROBIT_NOEPS_INIT = 1e-12

# IBProbit's CAVI iteration counts. Categorical, not suggest_int: 192 is 1.5x128
# rather than a doubling, so the ladder is not geometric and an integer range
# would spend trials on values nobody asked to compare.
IBPROBIT_UPDATE_ITERS = (32, 64, 128, 192)


def batch_cap(n_train: int) -> int:
    """Largest power of two leaving at least MIN_STEPS_PER_EPOCH steps/epoch,
    never above MAX_BATCH.

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
    return min(cap, MAX_BATCH)


def eval_quantum(dataset: str) -> int:
    """Samples between consecutive val_nll evaluations, for every trial.

    THE reason evaluation is scheduled on samples rather than steps: at a fixed
    step budget a step means `batch_size` samples, so a step grid measures
    trials at wildly different points of their own training -- across one
    study's ladder that is a 256x spread in data seen at the same "step". A
    trial at the ladder's top evaluates every step; one at batch = cap/k
    evaluates every k steps; both land on exactly the same samples_seen values.

    The quantum is the ladder's own cap, which makes it the only choice that
    works: `cap / batch` is an integer for every rung because the ladder is
    powers of two, and evaluation can only happen BETWEEN optimiser steps, so a
    quantum that is not a whole number of steps for some rung is unschedulable
    for that rung.

    It is also sub-epoch by construction, which is the whole point of a
    step-based protocol. `batch_cap` leaves at least MIN_STEPS_PER_EPOCH steps
    per epoch, so the quantum is at most half an epoch -- in practice 0.22 to
    0.33 epochs across these seven datasets. A quantum fixed globally at
    MAX_BATCH instead would be 16.1 epochs on flowers102 (1,020 training
    images), coarser even than evaluating once per epoch.
    """
    return batch_cap(N_TRAIN[dataset])


def eval_every_steps(dataset: str, batch_size: int) -> int:
    """Optimiser steps between evaluations for `batch_size` on `dataset`.

    Raises rather than rounding: a non-integer ratio means the requested
    schedule cannot be realised, and silently rounding it would hand back a
    grid whose samples_seen values differ per trial -- exactly the defect
    `eval_quantum` exists to remove, but now invisible.
    """
    quantum = eval_quantum(dataset)
    if quantum % batch_size:
        raise ValueError(
            f"eval quantum {quantum} for {dataset} is not a whole number of "
            f"batch_size={batch_size} steps. The ladder is powers of two, so "
            f"this means batch_size is off-ladder; see batch_sizes()."
        )
    return quantum // batch_size


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


def _suggest_ibprobit(
    trial: Any, dataset: str, *, search_eps: bool = True
) -> dict[str, Any]:
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
      * `learn_q` OFF -- Q is pinned at its q0 seed for this campaign, so the
        search is over a STATIC process noise. Leaving the recursion running
        while tuning `q_init` would have the search chase a moving target: q0
        is the seed the recursion starts from, so the two are not separable
        within a trial. Whether inferring Q beats the tuned static value is a
        follow-up ablation at that value. See worker.IBPROBIT_FIXED.
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
        # The box brackets 1/D = 7.8e-4 (D = 1280 for dinov3_huge) with
        # two decades of room above and below. Its floor is a decade and a half
        # above the measured-inert q0 = 3.1e-8, where learn_q on and off give
        # bit-identical mu over 40 steps -- trials sampled down there would be
        # comparing configurations that do not differ.
        "q_init": trial.suggest_float("q_init", 1e-6, 1e-1, log=True),
        # Beta(1, 1/eps_init) prior mean on the contamination rate. A genuine
        # model knob: it sets how readily the head discounts a badly-predicted
        # point, and the datasets here span very different label-noise levels.
        # Absent on the ibprobit_noeps arm, where it is pinned to
        # IBPROBIT_NOEPS_INIT instead. Dropped from the SEARCH rather than
        # sampled-and-overridden: a sampled axis that does not reach the
        # training script makes TPE model a dimension the runs do not have, and
        # the study's own record of what it searched would be wrong.
        **({"eps_init": trial.suggest_float("eps_init", 1e-3, 1e-1, log=True)}
           if search_eps else {}),
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

    if arm in ("ibprobit", "ibprobit_noeps"):
        return _suggest_ibprobit(trial, dataset, search_eps=(arm == "ibprobit"))

    # Lower bound 1e-3, not 1e-4, set by the pilot: at lr=1e-4, every
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
        # The measured optimum is ~1e-5. Fixing eta = 1 and searching
        # prior_scale in [1e-2, 1e2] instead confines the ratio to [1e-2, 1e2],
        # three orders of magnitude off the answer, and drives prior_scale to
        # its ceiling on all seven datasets. [1e-7, 1e-2] brackets 1e-5 by two
        # decades below and three above; 1e-4 is already 1.2 points worse and
        # 1e-3 6.3 points.
        params["kl_ratio"] = trial.suggest_float("kl_ratio", 1e-7, 1e-2, log=True)
        # At a fixed ratio, posterior sd grows as sqrt(prior_scale), so this
        # spans roughly sd 0.06 to 6.
        params["prior_scale"] = trial.suggest_float("prior_scale", 1e-1, 1e3, log=True)
        params["wishart_scale"] = trial.suggest_float("wishart_scale", 1e-3, 1e1, log=True)
    return params
