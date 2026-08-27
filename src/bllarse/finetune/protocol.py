"""Budget arithmetic, the accumulation plan, and the shared batch order.

Fine-tuning inverts the probing campaign's convention. Probing fixed 30,000
optimizer STEPS and derived epochs (bllarse/tuning/budgets.py); here the cap is
4 EPOCHS and the step count falls out of the tuned batch size. That inversion is
what makes Fig 5 valid: results-plan puts samples-seen on its x-axis, so arms
tuned to different batch sizes stay comparable even though they take different
numbers of steps.
"""

import math
from typing import Any, Mapping

import numpy as np

# The tuning campaign's sample budget. A budget in SAMPLES, not steps: at a
# fixed step budget `samples_seen = steps * batch`, so large batches would be
# handed more data and the tuned batch size would not transfer to the final run.
#
# 2,000,000 / 1,281,167 = 1.561 epochs, and 2e6 / EVAL_QUANTUM = 122
# evaluations at every rung of the ladder.
#
# 1,000,000 / 1,281,167 = 0.781 epochs, and 1e6 / EVAL_QUANTUM = 61
# evaluations at every rung of the ladder.
#
# HALVED from 2,000,000, and the objection this reverses is worth stating
# because it was correct at the time. 2M was chosen over 1M on the grounds that
# the shorter budget carried a DIRECTIONAL bias rather than merely less data:
# truncating earlier favours configs that move fast -- high trunk learning
# rates -- which is exactly the catastrophic-forgetting direction the campaign
# exists to characterise.
#
# That argument assumed there was no way to correct for the budget. There is
# now. The SDE duration rule says the optimal learning rate at a shorter run is
# HIGHER, by sqrt(m_D), and `transfer` scales it back down on the way out. The
# drift toward fast configs is no longer a bias to be avoided by spending more
# compute; it is the rule's prediction, and it is corrected rather than dodged.
#
# What that buys is the beta axes. The betas were previously unsearchable --
# they are a per-step timescale, so pinning the batch was the only thing that
# made the defaults defensible -- and `horizons` makes them transferable, hence
# searchable. Two more axes need more trials, and halving the budget pays for
# them.
#
# 1M is a FLOOR, not a midpoint, and two independent constraints put it here:
#
#   * beta_from_horizons requires H < steps. At 1953 steps the reference
#     H1 = 390.6 can still be explored ~2.5x upward before beta1 falls below
#     0.5; at 500k (976 steps) that headroom collapses to 1.25x, in the one
#     direction a non-stationary objective would want.
#   * The pruner and the patience clock are calibrated in EVALUATIONS. 1M
#     gives 61, so a 10-evaluation warmup and 20-evaluation patience still
#     leave room to fire (earliest stop at 21 of 61). 500k gives 30, where a
#     20-evaluation patience is barely functional.
#
# `truncated` remains the diagnostic, and matters MORE at this budget, not
# less. If it is true almost everywhere then ranking is happening on runs that
# were all still improving -- and the transfer rule maps optima, not rankings,
# so it does not rescue a comparison made before the curves separated.
MAX_SAMPLES = 1_000_000

# Samples between consecutive val_nll evaluations, for every trial.
#
# This replaces a log-spaced grid over STEPS. The old grid was sized as a figure
# artifact for 500-step runs and then reused at 30,000 steps for three jobs it
# was never sized for -- the pruner's key, the patience clock, and the argmin
# search -- where one "count" meant ~150 steps early and ~7,000 late.
#
# The quantum works because every rung of the batch ladder divides it: a trial
# at the top evaluates every 4 steps, one at the bottom every 64, and both land
# on exactly the same samples_seen values. Evaluation can only happen BETWEEN
# optimizer steps, so a quantum that is not a whole number of steps for some
# rung is unschedulable for that rung -- hence `eval_every_steps` raises.
EVAL_QUANTUM = 16_384


def steps_for_epochs(n_train: int, epochs: float, effective_batch: int) -> int:
    """Optimizer steps in `epochs` passes at `effective_batch`, drop-remainder.

    `epochs` is a float so a tuning trial can ask for a fraction of a pass --
    which is the whole tuning budget under this design, since trials are
    truncated by a samples-seen budget rather than run to convergence.
    """
    if effective_batch < 1:
        raise ValueError(f"effective_batch must be >= 1, got {effective_batch}")
    if effective_batch > n_train:
        raise ValueError(
            f"effective_batch={effective_batch} exceeds n_train={n_train}: "
            f"a single batch cannot be drawn"
        )
    steps = int(epochs * n_train) // effective_batch
    if steps < 1:
        raise ValueError(
            f"epochs={epochs} at effective_batch={effective_batch} over "
            f"n_train={n_train} yields zero steps"
        )
    return steps


def steps_for_samples(max_samples: int, effective_batch: int) -> int:
    """Optimizer steps to consume `max_samples` at `effective_batch`.

    The tuning counterpart of `steps_for_epochs`, and the reason both exist:
    the final campaign is capped in EPOCHS (4) while tuning is capped in
    SAMPLES (MAX_SAMPLES), so the two derive their step counts from different
    quantities and neither is expressible as the other without knowing n_train.
    """
    if effective_batch < 1:
        raise ValueError(f"effective_batch must be >= 1, got {effective_batch}")
    steps = int(max_samples) // effective_batch
    if steps < 1:
        raise ValueError(
            f"max_samples={max_samples} at effective_batch={effective_batch} "
            f"yields zero steps"
        )
    return steps


def eval_every_steps(effective_batch: int, quantum: int = EVAL_QUANTUM) -> int:
    """Optimizer steps between evaluations at `effective_batch`.

    Raises rather than rounding: a non-integer ratio means the requested
    schedule cannot be realised, and silently rounding it would hand back a
    grid whose samples_seen values differ per trial -- exactly the defect
    EVAL_QUANTUM exists to remove, but now invisible.
    """
    if effective_batch < 1:
        raise ValueError(f"effective_batch must be >= 1, got {effective_batch}")
    if quantum % effective_batch:
        raise ValueError(
            f"eval quantum {quantum} is not a whole number of "
            f"effective_batch={effective_batch} steps. The ladder is powers of "
            f"two, so this means the batch is off-ladder; see "
            f"tuning.finetune_space.BATCH_SIZES."
        )
    return quantum // effective_batch


def accum_plan(target_batch: int, micro_batch: int) -> tuple[int, int, int]:
    """Resolve a target batch into (micro, num_accum, effective).

    `micro_batch` is a memory artifact -- the largest per-forward batch that
    fits -- and differs between arms and frameworks. `target_batch` is the
    tuned quantity.

    The micro batch must DIVIDE the target exactly. This used to round, so a
    non-dividing pair silently produced an effective batch that was not the
    requested one (4096 over a micro batch of 100 gave 4100), which then
    mislabels every point of a samples-seen axis. Design A8 pins the micro
    batch to a power of two no larger than the ladder's gcd precisely so this
    is always satisfiable; a violation here means the pinned constant and the
    ladder have drifted apart, which is a configuration bug rather than
    something to paper over.
    """
    if micro_batch < 1:
        raise ValueError(f"micro_batch must be >= 1, got {micro_batch}")
    if target_batch < 1:
        raise ValueError(f"target_batch must be >= 1, got {target_batch}")
    micro = min(micro_batch, target_batch)
    if target_batch % micro:
        raise ValueError(
            f"micro batch {micro} does not divide target batch {target_batch}: "
            f"the effective batch would be {micro * round(target_batch / micro)}, "
            f"not the {target_batch} that was tuned."
        )
    num_accum = target_batch // micro
    return micro, num_accum, micro * num_accum


# --------------------------------------------------------------------------
# Hyperparameter transfer across batch size and training duration.
#
# The tuning campaign scores trials at MAX_SAMPLES (1,000,000) and the final
# campaign runs 4 epochs (5,124,668 samples). Nothing made those two
# comparable: a config tuned at one duration was simply RUN at the other, which
# assumes the optimum does not move with the budget. It does.
#
# The rule is from Mlodozeniec et al. 2025 (arXiv:2512.22382, Table 1), which
# derives batch-size and duration transfer from the same AdamW SDE limit. With
# m_B the batch ratio and m_D the data/duration ratio:
#
#     learning rate       x sqrt(m_B / m_D)
#     weight decay        x sqrt(m_B / m_D)     (PyTorch/optax convention)
#     (1 - beta1/beta2)   x  m_B / m_D
#     AdamW eps           x sqrt(m_D / m_B)
#     iterations          proportional to m_D / m_B
#
# The weight-decay direction is worth stating because the paper's own table is
# ambiguous at a glance and the two conventions differ. Their Eq. 1 discretises
# dTheta = (1/(eta*sigma))(g + lambda*sigma*Theta)dt + dW, so holding the drift
# fixed requires eta*sigma invariant AND lambda/eta invariant -- hence
# m_lambda = m_eta, the SAME factor as the learning rate. They confirm it from
# the other side: under the ORIGINAL AdamW (`AdamLH`, where lambda is not
# multiplied by eta) the exponent doubles to m_lambda = m_eta^2, "doubling the
# batch-size means that the weight decay should now be doubled". optax.adamw
# and torch.optim.AdamW are both the PyTorch convention, so m_lambda = m_eta
# here.
#
# Two invariants fall out, and both are worth checking against when this code
# is edited:
#
#   * Total decoupled shrinkage `steps * lr * wd` is preserved. That is the
#     same quantity finetune_space.TRUNK_WD_BOX derives by hand -- this rule is
#     its generalisation, not a new claim.
#   * The beta horizon 1/(1-beta) is a fixed FRACTION of the run rather than a
#     fixed number of steps. At the reference point beta2's 1000-step horizon
#     is 25.6% of 3906 steps; after transfer to 4 epochs it is 2562 of 10,009,
#     which is the same 25.6%.
#
# CAVEATS, since this is applied outside the regime it was validated in. The
# paper's experiments are decoder-only LM pre-training from scratch on
# RedPajama with a cosine schedule, and they explicitly note the duration rule
# "might be specific to the fixed (cosine) schedule that we use". We fine-tune
# a pretrained ViT at a CONSTANT learning rate with no warmup. Their Figure 7
# also shows that the best LR *schedule* at a short horizon is never a prefix
# of the best at a long one -- which we are not exposed to, having no schedule,
# but which is the reason not to add one without re-deriving this.
#
# NOT covered here, and not coverable: the head-side axes. VBLL's kl_ratio and
# prior_scale are loss terms rather than optimizer state, and IBProbit's q_init
# is alpha^2 / B, i.e. already batch-coupled by a different relation. `transfer`
# refuses a batch change in the latter case rather than guessing.

# The batch and betas the boxes in finetune_space were measured at, and the
# point every transfer is expressed relative to.
#
# 512 duplicates finetune_space.FIXED_BATCH_SIZE rather than importing it:
# finetune_space imports THIS module, so the dependency cannot run the other
# way. `test_protocol_reference_matches_the_pinned_batch` keeps them equal.
REFERENCE_BATCH = 512

# optax's and torch's shared AdamW defaults. Recorded as a constant because
# they are now a transferable quantity rather than a library default -- the
# campaign's betas are correct only at the reference step count, and
# `betas_for` is what keeps that true anywhere else.
REFERENCE_BETAS = (0.9, 0.999)

# The step count REFERENCE_BETAS is anchored at: the ft_*_fixb512 campaign's
# operating point, 2,000,000 samples at batch 512.
#
# A literal, NOT steps_for_samples(MAX_SAMPLES, REFERENCE_BATCH), which is what
# it used to be. That derivation was a trap: MAX_SAMPLES is the current tuning
# budget and moves when the campaign is re-costed, and re-deriving the anchor
# from it would silently redefine what "the reference betas" means -- declaring
# (0.9, 0.999) correct at whatever the new budget happens to be. The anchor is
# a fact about a campaign that has already run, so it is frozen as one.
REFERENCE_STEPS = 3906


def reference_steps() -> int:
    """Optimizer steps the reference betas are anchored at: 3906."""
    return REFERENCE_STEPS


def horizons(beta: float, steps: int) -> float:
    """`(1 - beta) * steps`: averaging windows per run. THE searched coordinate.

    beta is a per-step quantity and therefore not comparable across budgets;
    this is its transfer-invariant form, the same role `lr / sqrt(batch)` plays
    on the learning rate. A run of `steps` at `beta` averages over
    1/(1-beta) steps, so H counts how many such windows fit in the run --
    dimensionless, and unchanged by the SDE transfer rule, which scales
    (1 - beta) by exactly the inverse of the change in step count.

    At the reference: H = 390.6 for beta1, 3.906 for beta2.
    """
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"beta must be in [0, 1), got {beta}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    return (1.0 - beta) * steps


def beta_from_horizons(h: float, steps: int) -> float:
    """Invert `horizons`: the beta giving `h` averaging windows in `steps`.

    Raises when H >= steps. That is not an edge case to round away, it is the
    hard limit on what a short run can express: beta >= 0 requires the
    averaging window to be at least one step, so a budget of N steps cannot
    represent more than N windows. The bound is asymmetric in the direction
    that matters -- LONGER memory (small H) is free at any budget, SHORTER
    memory is what a short tuning run cannot reach, and shorter memory is what
    a non-stationary objective like IBProbit's re-assimilated head would want.
    """
    if h <= 0.0:
        raise ValueError(f"horizons must be > 0, got {h}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if h >= steps:
        raise ValueError(
            f"{h:.4g} averaging windows do not fit in a {steps}-step run: "
            f"beta would be {1.0 - h / steps:.4g}, not in [0, 1). A budget of "
            f"N steps cannot express more than N windows -- see "
            f"beta_from_horizons on why this bound is the real constraint on "
            f"shortening the tuning budget."
        )
    return 1.0 - h / steps


def betas_for(
    steps: int,
    *,
    betas: tuple[float, float] = REFERENCE_BETAS,
    from_steps: int | None = None,
) -> tuple[float, float]:
    """AdamW betas at `steps`, holding `(1 - beta) * steps` invariant.

    The momentum horizon is a fixed fraction of the run, not a fixed number of
    steps, so a run that is twice as long wants twice the averaging window.
    Defaults transfer from the tuning campaign's operating point
    (`REFERENCE_BETAS` at `reference_steps()`).

    Raises rather than clamping. A run short enough to drive a beta to zero is
    a run where the SDE limit this rule comes from has stopped applying, and
    silently returning beta1 = 0 would turn AdamW into signSGD-with-decay
    without anything in the config recording it. The binding case is beta1:
    (1 - 0.9) * 3906 = 390.6, so any budget under ~391 steps trips it.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    reference = reference_steps() if from_steps is None else from_steps
    if reference < 1:
        raise ValueError(f"from_steps must be >= 1, got {reference}")

    ratio = reference / steps
    out = []
    for name, beta in zip(("beta1", "beta2"), betas):
        one_minus = (1.0 - beta) * ratio
        if not 0.0 < one_minus < 1.0:
            raise ValueError(
                f"{name} transfers out of range at steps={steps}: "
                f"(1 - {beta}) * {reference} / {steps} = {one_minus:.4g}, "
                f"which is not in (0, 1). The horizon (1-beta)^-1 = "
                f"{1.0 / (1.0 - beta):.0f} steps cannot be held as a fraction "
                f"of a {steps}-step run."
            )
        out.append(1.0 - one_minus)
    return out[0], out[1]


def transfer(
    params: Mapping[str, Any],
    *,
    from_samples: int,
    from_batch: int,
    to_samples: int,
    to_batch: int,
) -> dict[str, Any]:
    """Rescale one tuned config to a new (batch, duration) operating point.

    Scales `lr`, `head_lr` and `weight_decay` by sqrt(m_B / m_D), `eps` by its
    inverse, sets `beta1`/`beta2` from `betas_for`, and sets `batch_size` to
    `to_batch`. Every other key passes through untouched.

    `head_lr_ratio` is deliberately among the untouched: trunk and head rates
    take the SAME factor on the batch axis, the duration axis and (under
    CompleteP's width rule, which gives hidden and unembedding weights the same
    m_N^-1) the width axis, so their ratio is invariant on all three. The
    ratio parameterisation was chosen in finetune_space for an unrelated reason
    and happens to be the transfer-safe coordinate.

    Betas come from whatever `params` already carries, falling back to
    `REFERENCE_BETAS` -- so transferring a tuning-campaign trial, which records
    no betas because it ran at the defaults, is correct without special-casing.
    """
    for name, value in (("from_samples", from_samples), ("to_samples", to_samples)):
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")
    for name, value in (("from_batch", from_batch), ("to_batch", to_batch)):
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")

    out = dict(params)

    # q0 = alpha^2 / B is IBProbit's own batch relation, derived from the CAVI
    # steady state and unrelated to the SDE rule above. Which of the two should
    # win when both apply is genuinely unknown, so a batch change on a config
    # carrying q_init is refused rather than resolved by picking one.
    if "q_init" in out and to_batch != from_batch:
        raise ValueError(
            f"cannot transfer q_init across a batch change "
            f"({from_batch} -> {to_batch}): q0 = alpha^2 / B is IBProbit's own "
            f"batch relation and the SDE rule says nothing about it. Transfer "
            f"at a fixed batch, or drop q_init and let --alpha re-derive it."
        )

    scale = math.sqrt((to_batch / from_batch) / (to_samples / from_samples))
    for key in ("lr", "head_lr", "weight_decay"):
        if key in out:
            out[key] = out[key] * scale
    if "eps" in out:
        out["eps"] = out["eps"] / scale

    betas = (
        float(out.get("beta1", REFERENCE_BETAS[0])),
        float(out.get("beta2", REFERENCE_BETAS[1])),
    )
    out["beta1"], out["beta2"] = betas_for(
        steps_for_samples(to_samples, to_batch),
        betas=betas,
        from_steps=steps_for_samples(from_samples, from_batch),
    )
    out["batch_size"] = to_batch
    return out


def shuffled_indices(n: int, seed: int, epoch: int) -> np.ndarray:
    """The training permutation for one pass.

    numpy `default_rng`, NOT `jax.random`: the PyTorch arm has to reproduce this
    exact order so both arms consume an identical data stream, which is the
    property full_finetune_comparison.py already guarantees within JAX and which
    a per-framework RNG would quietly break.
    """
    return np.random.default_rng([seed, epoch]).permutation(n)
