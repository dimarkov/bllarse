"""The fine-tuning search space: the batch ladder, the boxes, the TPE suggestions.

Sibling of `space.py`, which does the same job for linear probing. Held apart
rather than parameterised because almost nothing transfers: probing tunes a head
against FROZEN features, and every axis defined relative to feature scale means
something different once the trunk co-adapts.

What DOES transfer from the finished probing campaign is the shape of each
box -- see the per-axis notes below. Narrowing a box with a completed
experiment's optima is evidence; pinning a value to one of them is not, unless
the optima agreed across datasets. Exactly one did.
"""

from typing import Any

from bllarse.finetune import protocol

# The only backbone tuned, matching the probing campaign's choice. The paper's
# arguments centre on huge, and a config tuned here is transplanted across the
# ladder for the figures with that stated rather than buried.
TUNING_MODEL = "dinov3_huge"

# One seed per trial, not probing's two. Right for a 40-second probe; here it
# would double a ~880 A100-hour campaign.
TUNING_SEED = 200

# Disjoint from the reported seeds (137..146), so hyperparameters are not
# selected on the same random draws that later produce the error bars.
REPORTED_SEEDS = tuple(range(137, 147))

ARMS = ("ce", "vbll", "ibprobit", "ibprobit_noeps")

# EFFECTIVE (accumulated) batch. Memory does not bound this -- accumulation
# absorbs it -- so the ladder is bounded by what is worth spending trials on.
#
# Every rung divides protocol.EVAL_QUANTUM, which is what makes evaluation
# schedulable at the same samples_seen values on all of them. A rung that did
# not would raise in protocol.eval_every_steps rather than silently evaluating
# on a per-trial grid.
BATCH_SIZES = (256, 512, 1024, 2048, 4096)

# The GPU MICRO_BATCH was measured on. The constants below are valid for this
# device and nothing else -- a smaller card OOMs on them, a larger one leaves
# memory unused, and neither is visible from a finished run's config.
MICRO_BATCH_DEVICE = "NVIDIA A100-SXM4-40GB"

# Per-forward batch, measured per arm with
# scripts/finetune/probe_micro_batch.py (dinov3_huge, 224, bf16) and pinned.
#
# Both JAX arms measured 64; 128 OOMs on both. They agree because the trunk
# dominates: IBProbit's head state is ~20 MB at D=1281, K=1000, since `Sigma`
# is (d, d) shared across classes rather than per-class.
#
# 64, not the 16 the earlier full_finetune_comparison runs used. That 16 was
# set by hand and never probed, so those runs took 4x more micro-steps per
# optimizer step than the hardware required.
#
# `vbll` is absent until the PyTorch arm exists to measure it against -- a
# different framework and a different activation profile, so the JAX number
# does not transfer. `micro_batch` raises rather than borrowing it.
#
# Values must satisfy probe_micro_batch.floor_to_ladder_divisor: a power of two
# no larger than 256 (the ladder's gcd), so num_accum is exact at every rung.
MICRO_BATCH: dict[str, int] = {
    "ce": 64,
    "ibprobit": 64,
}

# Pinned, not searched -- the ONE head axis the probing campaign transfers as a
# value. Its per-dataset optima spanned 0.076 to 0.204, a 2.7x range across
# five datasets; every other head knob spanned 30x to 2500x, which is wider
# than a narrowed box and therefore evidence that the optimum is a property of
# the dataset rather than of the head.
WISHART_SCALE = 0.13

# The contamination ablation. Same head, same script, same everything -- eps is
# PINNED here instead of searched, so the mixture floor is numerically absent
# and the head is scored on its own predictive alone.
#
# 1e-12, not 0.0: `IBProbit._eps_seed` returns b0 = 100 when eps_init == 0.0,
# which is a Beta(1, 100) with mean 0.0099 -- the middle of the searched box,
# not the absence of one. At 1e-12, b0 = 1e12 swamps the recursion's `+sum(r)`
# term in float32 and eps holds at 1e-12.
#
# Six decades below EPS_INIT_BOX's floor, so "searched and tiny" and
# "numerically absent" stay distinguishable outcomes.
IBPROBIT_NOEPS_INIT = 1e-12

# Beta(1, 1/eps_init) prior mean on the contamination rate.
#
# The floor is three decades below anything the probing campaign selected
# (1.1e-3 to 3.3e-2). That is deliberate extra room downward: if the tuned
# value lands on or near 1e-6, the SEARCHED arm has answered the ablation's
# question by itself -- the head switched contamination off when it was free
# to keep it -- which is a stronger result than the pinned arm winning.
EPS_INIT_BOX = (1e-6, 1e-1)

# CAVI iterations per macro batch.
#
# Shifted UP from probing's (32, 64, 128, 192), which pinned to its ceiling in
# 4 of 5 IBProbit studies -- a ladder-too-low signal, not a value to adopt. 64
# and 32 are dropped because the PX operating point is ~128 iterations (where
# it matches ~1000 plain ones), so rungs below it measure an unconverged head
# rather than the method.
IBPROBIT_UPDATE_ITERS = (128, 192, 256)

# Narrowed from probing's [1e-6, 1e-1] to bracket its measured optima
# (2.4e-5 to 4.2e-3) by roughly a decade either side. q0 = alpha^2 / B, so this
# is batch-dependent by construction and the box, not the point, is what the
# probing campaign licenses.
Q_INIT_BOX = (1e-5, 1e-2)

# Narrowed from probing's [1e-7, 1e-2]. vbll's regularization_weight is
# kl_ratio * prior_scale / N, so N (1,020 to 75,750 in probing, 1.28M here) is
# already divided out and the ratio is the N-normalised quantity -- the one
# VBLL axis whose UNITS transfer even though its value does not. Probing's
# optima spanned 2.4e-7 to 7.1e-5.
KL_RATIO_BOX = (1e-7, 1e-4)

# Unchanged from probing. Its optima spanned 0.14 to 352 -- a 2500x range with
# no pattern -- so there is nothing here to narrow with.
PRIOR_SCALE_BOX = (1e-1, 1e3)

# Trunk learning rate and the head's MULTIPLE of it, not two independent rates.
# full_finetune_comparison.py records why they are correlated: AdamW moves
# weights by ~lr/step, and a fresh head's init scale (~0.015) would take
# thousands of 1e-4 steps to overwrite -- its defaults are 1e-4 / 1e-2, a ratio
# of 100. Two independent log-uniform axes spend trials sliding along that
# diagonal. Same reparameterisation `space.py` applies for kl_ratio.
TRUNK_LR_BOX = (1e-6, 1e-3)
HEAD_LR_RATIO_BOX = (1.0, 1e3)
TRUNK_WD_BOX = (1e-5, 1e-1)


def micro_batch(arm: str) -> int:
    """The pinned per-forward batch for `arm`.

    Raises rather than defaulting: the value is hardware-specific and
    unrecoverable from a finished run's config, so a guess here would either
    OOM mid-trial or waste memory across the whole campaign, invisibly.
    """
    flag_arm = "ibprobit" if arm.startswith("ibprobit") else arm
    if flag_arm not in MICRO_BATCH:
        raise KeyError(
            f"no pinned micro batch for arm {flag_arm!r}. Measure it on the "
            f"target GPU first:\n"
            f"    python scripts/finetune/probe_micro_batch.py "
            f"--arm {flag_arm if flag_arm != 'vbll' else 'ce'} "
            f"--model {TUNING_MODEL} --bf16\n"
            f"then pin the printed value in finetune_space.MICRO_BATCH."
        )
    return MICRO_BATCH[flag_arm]


def eval_count(max_samples: int = protocol.MAX_SAMPLES) -> int:
    """Evaluations per trial. Identical at every rung, which is the point."""
    counts = {
        protocol.steps_for_samples(max_samples, batch)
        // protocol.eval_every_steps(batch)
        for batch in BATCH_SIZES
    }
    if len(counts) != 1:
        raise ValueError(
            f"the ladder does not give a uniform evaluation count: {counts}. "
            f"Every rung must divide protocol.EVAL_QUANTUM."
        )
    return counts.pop()


def _suggest_ibprobit(trial: Any, *, search_eps: bool) -> dict[str, Any]:
    """IBProbit's axes.

    NO head learning rate, on either ibprobit arm. `main` initialises the
    optimizer on the backbone alone and the head is assimilated in closed form
    by `loss_fn.update(...)`; `finetune_jax.check_arm_flags` rejects --head-lr
    outright. Sampling one would present a tuned axis that does nothing, and
    would make TPE model a dimension the runs do not have.

    That absence is a RESULT, not a gap: "no head learning rate to tune" is the
    closed-form-conjugate advantage Fig 7 exists to measure.
    """
    params: dict[str, Any] = {
        "batch_size": trial.suggest_categorical("batch_size", list(BATCH_SIZES)),
        "lr": trial.suggest_float("lr", *TRUNK_LR_BOX, log=True),
        "weight_decay": trial.suggest_float("weight_decay", *TRUNK_WD_BOX, log=True),
        "q_init": trial.suggest_float("q_init", *Q_INIT_BOX, log=True),
        "num_update_iters": trial.suggest_categorical(
            "num_update_iters", list(IBPROBIT_UPDATE_ITERS)
        ),
    }
    if search_eps:
        params["eps_init"] = trial.suggest_float("eps_init", *EPS_INIT_BOX, log=True)
    return params


def suggest(trial: Any, arm: str) -> dict[str, Any]:
    """One TPE-sampled configuration for `arm`.

    Axis counts: ce 4, vbll 6, ibprobit 6, ibprobit_noeps 5.

    `cov_rank` and `parameterization` are NOT searched, following space.py's
    existing rule: rank buys calibration with memory, and neither CE nor
    IBProbit has an equivalent knob -- letting TPE spend it would let VBLL win
    partly by being larger. Rank is reported separately.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")

    if arm.startswith("ibprobit"):
        return _suggest_ibprobit(trial, search_eps=(arm == "ibprobit"))

    params: dict[str, Any] = {
        "batch_size": trial.suggest_categorical("batch_size", list(BATCH_SIZES)),
        "lr": trial.suggest_float("lr", *TRUNK_LR_BOX, log=True),
        "weight_decay": trial.suggest_float("weight_decay", *TRUNK_WD_BOX, log=True),
        "head_lr_ratio": trial.suggest_float(
            "head_lr_ratio", *HEAD_LR_RATIO_BOX, log=True
        ),
    }
    if arm == "vbll":
        params["kl_ratio"] = trial.suggest_float("kl_ratio", *KL_RATIO_BOX, log=True)
        params["prior_scale"] = trial.suggest_float(
            "prior_scale", *PRIOR_SCALE_BOX, log=True
        )
        # wishart_scale is absent by design -- pinned at WISHART_SCALE. Dropped
        # from the SEARCH rather than sampled-and-overridden, so the study's own
        # record of what it searched stays true.
    return params
