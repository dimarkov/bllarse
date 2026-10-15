"""Budget arithmetic, the accumulation plan, and the shared batch order.

Fine-tuning inverts the probing convention. Probing fixes 30,000 optimizer
STEPS and derives epochs (bllarse/tuning/budgets.py); here the cap is in EPOCHS
(or, for tuning, in samples) and the step count falls out of the tuned batch
size. That inversion is what makes Fig 3 valid: its x-axis is samples seen, so
arms tuned to different batch sizes stay comparable even though they take
different numbers of steps.
"""

import math
from typing import Any, Mapping

import numpy as np

# The default tuning sample budget (`--max-samples`). A budget in SAMPLES, not
# steps: at a fixed step budget `samples_seen = steps * batch`, so large batches
# would be handed more data and the tuned batch size would not transfer to the
# final run.
#
# 1,000,000 / 1,281,167 = 0.781 epochs, and 1e6 / EVAL_QUANTUM = 61
# evaluations at every rung of the ladder.
#
# A short budget carries a DIRECTIONAL bias rather than merely less data:
# truncating earlier favours configs that move fast -- high trunk learning
# rates -- which is exactly the catastrophic-forgetting direction this
# experiment characterises. The SDE duration rule accounts for it: the optimal
# learning rate at a shorter run is HIGHER, by sqrt(m_D), and `transfer` scales
# it back down on the way out. The drift toward fast configs is the rule's
# prediction, and it is corrected rather than avoided by spending more compute.
#
# The same rule makes the beta axes searchable. The betas are a per-step
# timescale, so without a transfer rule only the library defaults at a pinned
# batch would be defensible; `horizons` makes them transferable.
#
# Two independent constraints bound how short the budget can go:
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
# A fixed quantum of SAMPLES rather than a grid over steps, because the
# evaluation schedule serves three jobs -- the pruner's key, the patience clock,
# and the argmin search -- and each needs one evaluation to mean the same amount
# of training for every trial.
#
# The quantum works because every rung of the batch ladder divides it: a trial
# at batch 4096 evaluates every 4 steps, one at 256 every 64, and both land on
# exactly the same samples_seen values. Evaluation can only happen BETWEEN
# optimizer steps, so a quantum that is not a whole number of steps for some
# rung is unschedulable for that rung -- hence `eval_every_steps` raises.
EVAL_QUANTUM = 16_384


def steps_for_epochs(n_train: int, epochs: float, effective_batch: int) -> int:
    """Optimizer steps in `epochs` passes at `effective_batch`, drop-remainder.

    `epochs` is a float so a run can ask for a fraction of a pass.
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
    the final runs are capped in EPOCHS while tuning is capped in SAMPLES
    (MAX_SAMPLES by default), so the two derive their step counts from different
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

    The micro batch must DIVIDE the target exactly. Rounding would let a
    non-dividing pair silently produce an effective batch that is not the
    requested one (4096 over a micro batch of 100 would give 4100), which
    mislabels every point of a samples-seen axis. finetune_space.MICRO_BATCH
    pins the micro batch to a power of two no larger than the ladder's gcd
    precisely so this is always satisfiable; a violation here means the pinned
    constant and the ladder have drifted apart, which is a configuration bug
    rather than something to paper over.
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
# Tuning scores trials at a short sample budget (MAX_SAMPLES by default) and
# the final runs train for whole epochs (2 epochs = 2,562,334 samples for the
# reported runs). Simply RUNNING a config tuned at one duration at the other
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
#     is 25.6% of 3906 steps; after transfer to 2 epochs at the same batch it
#     is 1281 of 5004, which is the same 25.6%.
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

# Layer-wise learning-rate decay. Standard for transformer fine-tuning (ULMFiT's
# discriminative rates; the MAE/BEiT ViT recipes), and the convention is that
# the multiplier grows with depth: the layers nearest the head adapt at the full
# rate while the layers nearest the input are damped, on the argument that
# generic features should not be disturbed to fit a task.
#
# NOTE which end this damps, because it decides what LLRD can and cannot fix
# here. IBProbit's head consumes the pooled patch-token mean of the FINAL block,
# so the features it fits are produced by the layers LLRD leaves at full rate.
# Late-run drift in the top blocks is not what this instrument targets.
LAYER_DECAY_OFF = 1.0

# LLRD's SPAN: what the bottom of the trunk gets relative to the top block.
# `--layer-decay` is a PER-BLOCK factor, so a fixed value means something
# different at every depth -- 0.75 puts the embeddings at 0.75^12 = 3.2e-2 on
# a 12-block trunk, 1.0e-3 at 24 blocks and 1.0e-4 at 32. Carrying one number
# up the dinov3 ladder would silently freeze the embeddings on the deeper
# backbones: 316x more damping on huge than the value validated on small.
#
# The transferable quantity is therefore the span, not the factor. 0.75^12 is
# the span validated on dinov3_small (0.75 per block over 12 blocks), written as
# the expression that produced it rather than as 0.0317 so the provenance
# survives.
LAYER_DECAY_SPAN = 0.75 ** 12
# The span that means "no decay". Distinct from LAYER_DECAY_OFF only in what it
# names, but they are different quantities and collapsing them reads as a bug.
LAYER_DECAY_SPAN_OFF = 1.0


def layer_decay_for_span(span: float, depth: int) -> float:
    """Per-block decay putting the trunk's BOTTOM at `span` x the top's rate.

    Inverts `llrd_scale`'s `decay^(depth-1-i)` at the deepest damping the trunk
    applies. `llrd_scale` gives a NON-BLOCK parameter (patch embedding, cls
    token, positional embeddings) `decay^depth`, one step below block 0, so the
    span is measured there -- that is the parameter the damping is really
    about, and keying on block 0 instead would leave the embeddings drifting by
    a factor of `decay` across the ladder.
    """
    if span <= 0.0 or span > 1.0:
        raise ValueError(
            f"layer decay span must be in (0, 1], got {span}. It is the "
            f"fraction of the TOP block's rate that the embeddings receive."
        )
    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")
    return float(span ** (1.0 / depth))


def llrd_scale(block: int | None, depth: int, decay: float) -> float:
    """LR multiplier for a parameter in `block` of a `depth`-layer trunk.

    Layers are numbered from the input, so `block=0` is the first transformer
    block and `block=depth-1` the last. `block=None` is a NON-BLOCK parameter --
    patch embedding, cls token, positional embeddings -- which sits BELOW every
    block and takes the strongest damping; the trunk's final norm counts as the
    top and takes 1.0, which is why callers pass `depth - 1` for it rather than
    None.

    The last block gets exactly 1.0, so `decay=1.0` is the identity and
    reproduces a run without LLRD bit-for-bit.
    """
    if not 0.0 < decay <= 1.0:
        raise ValueError(f"decay must be in (0, 1], got {decay}")
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if block is None:
        exponent = depth
    else:
        if not 0 <= block < depth:
            raise ValueError(f"block {block} outside [0, {depth})")
        exponent = depth - 1 - block
    return float(decay ** exponent)


def wsd_multiplier(step: int, steps: int, *, decay_start: float,
                   decay_to: float) -> float:
    """The WSD multiplier on the peak LR at `step`, as plain arithmetic.

    `decay_start` is WHERE THE DECAY BEGINS, as a fraction of the run: 0.25
    holds the peak until 25% of the steps and then ramps down over the
    remaining 75%. It is not the length of the decay phase.

    `decay_start = 1.0` never decays, which is the OFF setting.

    THE definition of the schedule. `wsd_schedule` is its optax realisation for
    the JAX arms and the torch arm reads this directly, so the two frameworks
    cannot drift into running different schedules under one flag --
    `test_the_two_realisations_agree` pins them together.
    """
    if not 0.0 <= decay_start <= 1.0:
        raise ValueError(f"decay_start must be in [0, 1], got {decay_start}")
    if not 0.0 < decay_to <= 1.0:
        raise ValueError(f"decay_to must be in (0, 1], got {decay_to}")
    if decay_start >= 1.0:
        return 1.0
    stable = int(round(steps * decay_start))
    if step <= stable:
        return 1.0
    span = max(1, steps - stable)
    frac = min(1.0, (step - stable) / span)
    return 1.0 + (decay_to - 1.0) * frac


def wsd_schedule(peak: float, steps: int, *, decay_start: float,
                 decay_to: float):
    """Warmup-Stable-Decay learning rate: constant, then a linear tail.

    `decay_start` is the fraction of the run at which the ramp BEGINS, so 0.25
    decays over the last 75% of the steps. Returns the scalar `peak` unchanged
    when `decay_start` is 1.0 (the OFF setting), so the default path is
    bit-identical to a constant-LR run.

    WSD rather than cosine, and the reason is the transfer rule. Every
    hyperparameter was tuned at a CONSTANT learning rate and
    moved by the AdamW SDE rule, whose authors note it "might be specific to
    the fixed (cosine) schedule that we use". A cosine run shares no phase with
    the tuned configuration. A WSD run spends `decay_start` of its steps at
    exactly the tuned constant, so the tuned point remains the operating point
    for most of the run and only the tail is new.

    NOTE what this does to weight decay: optax.adamw scales the decoupled decay
    update BY the learning rate, so decaying lr decays the shrinkage rate with
    it. Total shrinkage `steps x lr x wd` therefore falls by the schedule's own
    mean multiplier -- decay_start + (1 - decay_start) * (1 + decay_to) / 2 --
    and is NOT the value the tuned configuration was selected at.
    """
    if not 0.0 <= decay_start <= 1.0:
        raise ValueError(f"decay_start must be in [0, 1], got {decay_start}")
    if not 0.0 < decay_to <= 1.0:
        raise ValueError(f"decay_to must be in (0, 1], got {decay_to}")
    if decay_start >= 1.0:
        return peak

    import optax

    stable = int(round(steps * decay_start))
    return optax.join_schedules(
        [optax.constant_schedule(peak),
         optax.linear_schedule(peak, peak * decay_to,
                               transition_steps=max(1, steps - stable))],
        boundaries=[stable],
    )


def wsd_mean_multiplier(decay_start: float, decay_to: float) -> float:
    """Mean LR over a WSD run, as a fraction of the peak.

    The factor by which the run's total decoupled shrinkage falls relative to
    the constant-LR configuration the hyperparameters were tuned at.
    """
    if decay_start >= 1.0:
        return 1.0
    decaying = 1.0 - decay_start
    return decay_start + decaying * (1.0 + decay_to) / 2.0


# The batch and betas the boxes in finetune_space were measured at, and the
# point every transfer is expressed relative to.
#
# 512 duplicates finetune_space.FIXED_BATCH_SIZE rather than importing it:
# finetune_space imports THIS module, so the dependency cannot run the other
# way. `test_the_reference_batch_matches_the_pinned_one` keeps them equal.
REFERENCE_BATCH = 512

# optax's and torch's shared AdamW defaults. Recorded as a constant because
# they are a transferable quantity rather than a library default -- these
# betas are correct only at the reference step count, and `betas_for` is what
# keeps that true anywhere else.
REFERENCE_BETAS = (0.9, 0.999)

# The step count REFERENCE_BETAS is anchored at: the operating point the
# finetune_space boxes were measured at, 2,000,000 samples at batch 512.
#
# A literal, NOT steps_for_samples(MAX_SAMPLES, REFERENCE_BATCH): MAX_SAMPLES
# is the tuning budget and moves when tuning is re-costed, and deriving the
# anchor from it would silently redefine what "the reference betas" means --
# declaring (0.9, 0.999) correct at whatever the new budget happens to be. The
# anchor is a fact about runs that have already happened, so it is frozen as
# one.
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
    Defaults transfer from the reference operating point (`REFERENCE_BETAS`
    at `reference_steps()`).

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
    inverse, sets `beta1`/`beta2` from `betas_for`, scales `q_init` by
    (from_steps / to_steps)^2, and sets `batch_size` to `to_batch`. Every other
    key passes through untouched.

    `head_lr_ratio` is deliberately among the untouched: trunk and head rates
    take the SAME factor on the batch axis, the duration axis and (under
    CompleteP's width rule, which gives hidden and unembedding weights the same
    m_N^-1) the width axis, so their ratio is invariant on all three. The
    ratio parameterisation was chosen in finetune_space for an unrelated reason
    and happens to be the transfer-safe coordinate.

    Betas come from whatever `params` already carries, falling back to
    `REFERENCE_BETAS` -- so transferring a tuning trial that records no betas
    because it ran at the defaults is correct without special-casing.
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

    # The DURATION axis for q0, which the batch guard above deliberately left
    # open. MEASURED: a 7-point q_init grid (one decade apart,
    # 1.29e-7 .. 1.29e-2) run at three horizons on dinov3_large with every
    # other coordinate frozen puts the optimum at
    #
    #   781 steps -> 1.29e-4     2502 -> 1.29e-5     5004 -> 1.29e-6
    #
    # i.e. one decade down per 3.2x of horizon. Fitting p in q0 ~ T^-p gives
    # p = 1.98 at the middle point; p = 1 predicts 4.03e-5 there against a
    # measured 1.29e-5, and the curve at that value is 0.008 val_nll worse, so
    # p = 1 is excluded. p = 2 predicts 3.15e-6 at 5004, which falls between the
    # two best grid points there (0.4273 and 0.4285, indistinguishable).
    #
    # The Kalman reading predicts the exponent rather than fitting it: for a
    # random walk plus observation the steady state gives P ~ sqrt(QR), so the
    # effective memory is L ~ 1/sqrt(q0), and holding memory at a fixed fraction
    # of the run gives q0 ~ 1/T^2 directly. alpha ~ 1/T equivalently.
    #
    # Why this is not optional: transferring q_init by IDENTITY leaves it too
    # high by (T_to / T_from)^2 on a longer run. A q0 ~68x above the rule's
    # optimum (5.35e-5 against 7.9e-7 at T = 10008) puts IBProbit in the regime
    # where the val curve peaks at ~45% of the run and then climbs, and makes it
    # trail both baselines on test NLL; at the transferred value it leads them.
    if "q_init" in out:
        out["q_init"] = out["q_init"] * (
            steps_for_samples(from_samples, from_batch)
            / steps_for_samples(to_samples, to_batch)
        ) ** 2

    out["batch_size"] = to_batch
    return out


def shuffled_indices(n: int, seed: int, epoch: int) -> np.ndarray:
    """The training permutation for one pass.

    numpy `default_rng`, NOT `jax.random`: the PyTorch arm has to reproduce this
    exact order so both arms consume an identical data stream, which a
    per-framework RNG would quietly break.
    """
    return np.random.default_rng([seed, epoch]).permutation(n)
