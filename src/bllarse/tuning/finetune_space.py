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

import math
from typing import Any

from bllarse.finetune import protocol

# The backbone tuned. dinov3_large (304M), NOT huge (840M): at the measured
# throughput a huge trial is ~8.4 h against large's ~3 h, and the search budget
# buys roughly three times as many trials per A100-hour on large.
#
# The cost is stated rather than hidden: a config tuned on large is
# TRANSPLANTED to huge for the final runs, so the reported operating point was
# not selected on the backbone it is reported at. Fig 1 already transplants in
# the other direction (huge's probing config across the whole ladder), so the
# paper carries this class of assumption either way -- but it is an assumption,
# and the direction here is the less comfortable one, since large has 24 layers
# at width 1024 against huge's 32 at 1280.
TUNING_MODEL = "dinov3_large"

# One seed per trial, not probing's two. Right for a 40-second probe; here it
# would double a ~880 A100-hour campaign.
TUNING_SEED = 200

# Disjoint from the reported seeds (137..146), so hyperparameters are not
# selected on the same random draws that later produce the error bars.
REPORTED_SEEDS = tuple(range(137, 147))

ARMS = ("ce", "vbll", "ibprobit", "ibprobit_noeps")

# EFFECTIVE (accumulated) batch.
#
# The ceiling is protocol.EVAL_QUANTUM, and that is the ONLY thing bounding it.
# Every rung must divide the quantum or the evaluation grid is unschedulable
# for that rung (protocol.eval_every_steps raises rather than silently
# evaluating on a per-trial grid), and 16,384 divides itself -- so it is the
# largest batch this protocol can carry, and every rung from 256 up yields
# exactly 122 evaluations.
#
# It is NOT bounded by memory. The design doc said "here the ceiling is memory"
# and capped the ladder at 4096, but gradient accumulation absorbs the
# effective batch entirely -- that is the whole reason the micro batch is
# pinned separately (design A8). At the top rung a step is 512 micro-steps of
# 32; nothing about that is memory-limited.
#
# 16,384 also matches the probing campaign's MAX_BATCH and is the batch the
# earlier full_finetune_comparison production runs actually used, so excluding
# it would have excluded the one large-batch operating point this codebase has
# real evidence for.
#
# The top rungs are a genuinely different regime, not just bigger: at 16,384
# the whole 2M-sample budget is 122 optimizer steps and evaluation happens
# after EVERY step. Whether that trains well is exactly what design D5 means by
# treating batch tolerance as a method property -- so it is TPE's to discover,
# not the ladder's to prejudge.
BATCH_SIZES = (256, 512, 1024, 2048, 4096, 8192, 16384)

# PINNED at 512 for the second campaign; the ladder above is kept because
# `eval_count` still validates it and because the first campaign searched it.
#
# The first campaign (studies ft_*_shards2m) sampled the whole ladder and the
# arms disagreed: vbll converged to 512 (all of its top 8), ibprobit_noeps to
# 512 (27 of 36 completions), ce to a 1024/2048 plateau, and ibprobit did not
# converge at all -- TPE drifted to 256 while the best scores sat at 1024.
#
# That drift is the reason this is now pinned rather than re-searched. At the
# pruner's warmup evaluation the median val_nll is MONOTONE in batch (256
# 0.5396, 512 0.5463, 1024 0.5697, 2048 0.5959, 4096 0.6299) because a smaller
# batch takes more optimizer steps per sample and therefore leads early. By the
# end the ordering reverses (1024 wins at 0.4427, 256 is 0.4481). MedianPruner
# scores the early ranking and TPE consumes pruned trials, so the search was
# being pulled toward a rung that loses.
#
# 512 over 1024 because the arms that have a MEASURED preference prefer it, and
# because of the step count. At 2M samples a rung is 2e6/B optimizer steps:
#
#     256 -> 7812   512 -> 3906   1024 -> 1953   2048 -> 976
#     4096 -> 488   8192 -> 244   16384 -> 122
#
# AdamW's default b2 = 0.999 has a 1/(1-b2) = 1000-step averaging horizon. At
# 2048 the ENTIRE run is 976 steps -- shorter than one horizon, so the second
# moment never reaches steady state and the run lives inside bias correction.
# At 8192/16384 there are 244/122 steps, which is the likeliest explanation for
# vbll returning 8.49 and 8.64 there rather than any vbll-specific fragility.
# 512 gives 3906 steps, about four horizons, which is the regime the default
# betas are meant for -- see BETAS below.
#
# What this costs: ce measured its optimum at 1024/2048 and has only one
# completed trial at 512 (val_nll 0.5667, at lr 1.3e-4 -- four times its own
# winning band, so that number is an lr artefact, not a verdict on the rung).
# ce's lr band at 1024 was 4.6e-6 to 3.4e-5, so at 512 it wants roughly
# 2e-6 to 2e-5, comfortably inside TRUNK_LR_BOX. The risk that 512 is simply
# wrong for ce is real but small, and it is what the first campaign's ce trials
# at 512 were never given an lr to test.
#
# Pinning also removes a confound from the arm comparison itself: at a fixed
# SAMPLE budget the batch is the steps-per-sample knob, so the first campaign
# compared ce at 2048 against vbll at 512 -- a factor of four in optimizer
# effort. This is a protocol decision in favour of the most fragile arm and
# belongs in the methods section as one.
FIXED_BATCH_SIZE = 512

# SEARCHED, per arm, as of this campaign. The two constants below are the
# boxes; `BETAS` is retained only as the historical anchor.
#
# The previous reasoning for pinning them was: the betas are a per-STEP
# timescale, so whether the default is appropriate is a question about the step
# count and nothing else -- which is settled by FIXED_BATCH_SIZE rather than by
# a search. That was sound while there was no way to move a beta across step
# counts. protocol.horizons is that way, so the axis stops being
# settled-by-construction and becomes searchable-and-transferable.
#
# It is also what the rule's authors actually do. Mlodozeniec et al. jointly
# optimise per-module learning rate, weight decay, AdamW b1/b2/eps and
# initialisation scale at 50M/1.6B and transfer to 1.3B/26B -- 420x compute --
# for a 27% speed-up. Tuning the betas and scaling them is the VALIDATED path;
# pinning them was us being more conservative than the source.
#
# Per-arm optima are plausible on mechanism rather than merely in principle:
#
#   * ibprobit's trunk gradient is computed against a head re-assimilated in
#     closed form every step, which is a genuinely non-stationary objective and
#     argues for SHORTER memory (larger H).
#   * vbll's KL term collapsed toward zero in the last campaign (the
#     kl_ratio x prior_scale product fell five decades), leaving a loss close
#     to CE through a SAMPLED head -- estimator noise, which argues for longer
#     b2 averaging.
#   * ce has neither.
#
# One shared box rather than three, because those are hypotheses about
# DIRECTION and narrowing per arm before measuring would prejudge them.
BETAS = (0.9, 0.999)

# The searched coordinate is protocol.horizons(beta, steps) = (1-beta)*steps --
# averaging windows per run -- NOT beta. Same reason lr is searched as
# lr/sqrt(B): beta is per-step and therefore incomparable across budgets, while
# H is exactly what the SDE rule holds invariant. A trial records H; the config
# carries the beta that H implies at the trial's own step count.
#
# Boxes are stated in H and their beta ranges quoted at the tuning budget
# (MAX_SAMPLES at FIXED_BATCH_SIZE = 1953 steps), because H is the searched
# thing but beta is the readable one.
#
# b2: two decades centred on the reference H = 3.906. At 1953 steps that is
# b2 in [0.9795, 0.99980] -- wider on both sides than the 0.95-0.9999 range
# modern practice spans, so the box does not bind.
BETA2_HORIZONS_BOX = (0.4, 40.0)

# b1: the reference is H = 390.6, and this box is deliberately NOT symmetric
# about it. At 1953 steps it spans b1 in [0.600, 0.980], and the ceiling is set
# by protocol.beta_from_horizons' hard limit rather than by taste: H must stay
# below the step count, and 780 already puts b1 at 0.6. Pushing further would
# buy range that the FINAL budget can express (10,009 steps) but the tuning
# budget cannot, which is the wrong way round -- the search has to be able to
# visit whatever it transfers.
#
# This is the axis that sets the floor under MAX_SAMPLES; see the note there.
BETA1_HORIZONS_BOX = (39.0, 780.0)

# The ceiling is not really a constant -- it is `b1 >= 0.6`, which is
# H1 <= 0.4 x steps, and 780 is that rule evaluated at the 1953-step tuning
# budget (0.4 x 1953 = 781). Stated as a rule because a SHORTER budget makes
# the constant infeasible rather than merely wide: at 781 steps (100k samples
# at batch 128) `beta_from_horizons` raises for any H >= 781, so the sampler
# would draw points that cannot be realised, on a boundary, and the trial would
# die rather than be scored.
BETA1_MAX_H_FRACTION = 0.4


def beta1_horizons_box(steps: int) -> tuple[float, float]:
    """The b1 horizon box at a given tuning budget.

    Identical to BETA1_HORIZONS_BOX at the 1953-step budget every completed
    campaign used, so nothing already searched is re-scoped; it narrows only
    where the constant would be unreachable.
    """
    lo, hi = BETA1_HORIZONS_BOX
    capped = min(hi, BETA1_MAX_H_FRACTION * steps)
    if capped <= lo:
        raise ValueError(
            f"{steps} steps leaves no b1 horizon box: the ceiling "
            f"{capped:.1f} is at or below the floor {lo}. The tuning budget is "
            f"too short to search b1 at all -- raise it, or lower the batch."
        )
    return (lo, capped)

# The GPU MICRO_BATCH was measured on. The constants below are valid for this
# device and nothing else -- a smaller card OOMs on them, a larger one leaves
# memory unused, and neither is visible from a finished run's config.
MICRO_BATCH_DEVICE = "NVIDIA A100-SXM4-40GB"

# Per-forward batch, keyed by (MODEL, arm), at 224 with bf16 on an A100-40GB.
#
# Keyed by MODEL as well as arm, which is a CORRECTION. It used to be arm-only,
# so micro_batch("vbll") returned 64 whatever backbone was being trained -- a
# value measured on dinov3_large. The final campaign switched to dinov3_huge
# and all five vbll runs died at model load. The constant was not wrong; it was
# answering a question with two arguments while being given one.
#
# 32 on both arms; 64 OOMs on both. Measured by
# scripts/finetune/probe_micro_batch.py running a FULL optimizer step --
# optimizer state allocated, two micro-batches accumulated, apply_grads, then a
# jitted eval chunk -- because a bare micro-grad measurement is not the peak the
# loop reaches.
#
# It cross-checks against two independent data points. Eleven
# full_finetune_comparison runs completed at micro=16 on a 32 GB card, and the
# fixed state here is ~13.5 GB (AdamW's two moment buffers at 2 x 3.4 GB for
# 840M fp32 params, plus the grads_acc buffer). Activations scale about
# linearly in the micro batch, so 13.5 + A(16) fits 32 GB and 13.5 + A(32) fits
# 40 GB while A(64) does not -- which is exactly what the probe found.
#
# Two earlier answers were wrong and are recorded so they are not re-derived:
# 64 came from measuring `ce_micro_grad` in ISOLATION (3.4 GB of backbone
# instead of 13.5 GB of resident state -- 26 GB fits in 40, 36 GB does not, and
# a smoke trial OOMed on exactly that); 4 came from a probe whose eval leg ran
# OUTSIDE jit, materialising a bf16 copy of the whole backbone and every
# layer's activations instead of letting XLA fuse them.
#
# `vbll` is absent until the PyTorch arm exists to measure it against -- a
# different framework and a different activation profile, so the JAX number
# does not transfer. `micro_batch` raises rather than borrowing it.
#
# Values must satisfy probe_micro_batch.floor_to_ladder_divisor: a power of two
# no larger than 256 (the ladder's gcd), so num_accum is exact at every rung.
MICRO_BATCH: dict[str, dict[str, int]] = {
    # dinov3_small at 128 is the whole target batch in ONE forward
    # (num_accum = 1), measured at 7.18 / 7.17 / 8.81 GiB on the 40 GB card --
    # a quarter of it, so the ceiling is nowhere near. 128 is pinned rather
    # than probed higher because the batch ladder needs micro to DIVIDE the
    # target and 128 is the target.
    "dinov3_small": {"ce": 128, "ibprobit": 128, "vbll": 128},
    # dinov3_big (ViT-B/16, 768 dim, 12 blocks). Measured 2026-08-31 by a true
    # ceiling SEARCH: the JAX arms take 128 at 16.94 / 16.97 GiB and OOM at
    # 256, while vbll fits 256 at 26.38 GiB reserved. So the torch arm needs no
    # accumulation at the target batch of 256 and the JAX arms accumulate 2.
    #
    # The asymmetry is the same shape as dinov3_large's (vbll 64 against 32)
    # and has the same cause: the JAX arms hold optimizer state for the whole
    # trunk in one jitted step, so their peak lands inside the step rather than
    # across it.
    #
    # An earlier probe reported 64 for both JAX arms. That measurement was
    # void: it was invoked with --peak-at 64, which pins batch_size_init AND
    # batch_size_max to its argument (probe_micro_batch.py), so it measured
    # exactly where it was told to and reported the input back. 9.08 GiB on a
    # 40 GB card should have been the tell.
    "dinov3_big": {"ce": 128, "ibprobit": 128, "vbll": 256},
    # Not a fresh measurement: this is what the whole betasearch1m tuning
    # campaign ran at, 934 trials with no OOM -- a stronger guarantee than a
    # probe. vbll's 64 belongs to THIS row and always did.
    "dinov3_large": {"ce": 32, "ibprobit": 32, "vbll": 64},
    # vbll is 32 here, not the 64 it takes on large. Measured after the
    # borrowed value killed five reported runs with
    # `torch.OutOfMemoryError: tried to allocate 32.00 MiB` on a 39.49 GiB
    # card that already held 36.5 GiB: a real 32,768-sample run completed at
    # both 32 and 16, and 32 is taken because it needs half the accumulation
    # steps of 16 for the same memory headroom.
    #
    # The two settings scored val_nll 0.8486 and 0.8489 on identical data,
    # which is the check that matters: the micro batch is a MEMORY artefact
    # and must not move the arithmetic. A gap larger than accumulation-order
    # noise would mean it does.
    #
    # The JAX arms are 16, not 32, MEASURED 2026-09-04 by running the real
    # training script on an A100-40GB rather than the probe (which builds a
    # two-group optimizer and has no --layer-decay-span, so it no longer
    # measures what the run executes):
    #
    #   micro 32, LLRD on  -> OOM, 13.77 GiB in jit_ce_micro_grad
    #   micro 32, LLRD OFF -> OOM, 13.77 GiB -- the SAME failure
    #   micro 16, LLRD on  -> runs, 76.60 img/s
    #   micro  8, LLRD on  -> runs, 61.51 img/s
    #
    # The LLRD-off control is why 32 is retired rather than made conditional:
    # it was the obvious suspect (33 optax groups on a 32-block trunk) and it
    # is not the cause. 32 simply does not fit on this card.
    #
    # 16 over 8 because it is 25% faster and the arithmetic is identical --
    # val_nll 6.3554 against 6.3555 on the same data, which is the check that
    # matters: the micro batch is a MEMORY artefact and must not move the
    # numbers. Accumulation absorbs the change (512 = 32 x 16), so the FLOPs
    # per optimizer step are unchanged; only the launch count moves.
    #
    # UNEXPLAINED, and worth a bisect when the cluster is quiet: MLflow records
    # the ftfinal huge campaign -- 3 arms x 5 seeds, 4 full epochs -- running at
    # micro 32 / batch 512 with layer_decay=None, on this same hardware. That
    # configuration is exactly the LLRD-off control above, and it now OOMs. If
    # something regressed, dinov3_large's 32 may be nearer its edge than it
    # looks.
    "dinov3_huge": {"ce": 16, "ibprobit": 16, "vbll": 32},
}

# Trunk depth, in transformer blocks. Needed for exactly one thing: converting
# a LEGACY per-block layer decay into the depth-agnostic SPAN that transfers.
#
# Trials from before 2026-08-31 recorded `layer_decay` (a per-block factor) and
# no span, because the span did not exist yet. A per-block 0.75 means 0.75^12 at
# depth 12 and 0.75^32 at depth 32 -- 316x more damping -- so replaying such a
# trial's config on a different backbone silently trains a different model than
# the one that was scored. `final_config.best_config` converts through this
# table instead.
#
# Read off the backbones themselves; `submit_finetune_tune.sh`'s pre-flight
# prints the same numbers ("OK dinov3_small 12 blocks") on every launch, which
# is the check that keeps this table honest. It is NOT consulted at training
# time -- finetune_jax reads the depth from the trunk it actually loaded.
TRUNK_DEPTH: dict[str, int] = {
    "dinov3_small": 12,
    "dinov3_big": 12,
    "dinov3_large": 24,
    "dinov3_huge": 32,
}


def trunk_depth(model: str) -> int:
    """The block count for `model`, or a refusal naming what is missing."""
    try:
        return TRUNK_DEPTH[model]
    except KeyError:
        raise KeyError(
            f"no trunk depth recorded for {model!r}; known: "
            f"{sorted(TRUNK_DEPTH)}. Add it from the launcher's pre-flight "
            f"line ('OK <model> <n> blocks')."
        ) from None

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
# Four decades below EPS_INIT_BOX's floor (1e-8), so "searched and tiny" and
# "numerically absent" stay distinguishable outcomes. The gap narrowed when the
# searched floor was lowered; four decades is still far more than the recursion
# can traverse, but it is the quantity to re-check if that floor drops again.
IBPROBIT_NOEPS_INIT = 1e-12

# Beta(1, 1/eps_init) prior mean on the contamination rate.
#
# The floor is FIVE decades below anything the probing campaign selected
# (1.1e-3 to 3.3e-2). That is deliberate room downward: if the tuned value
# lands near 1e-8, the SEARCHED arm has answered the ablation's question by
# itself -- the head switched contamination off when it was free to keep it --
# which is a stronger result than a separately-tuned pinned arm winning, and it
# costs nothing but box width on an axis TPE explores in log space.
EPS_INIT_BOX = (1e-8, 1e-1)

# CAVI iterations per macro batch.
#
# Shifted UP from probing's (32, 64, 128, 192), which pinned to its ceiling in
# 4 of 5 IBProbit studies -- a ladder-too-low signal, not a value to adopt. 64
# and 32 are dropped because the PX operating point is ~128 iterations (where
# it matches ~1000 plain ones), so rungs below it measure an unconverged head
# rather than the method.
#
# Extended again for the second campaign, for the same reason it was shifted up
# the first time: 256 was selected in 5 of the top 6 ibprobit trials, i.e. it
# pinned to its ceiling exactly as probing's 192 did.
#
# This is the one widening here that is NOT free. CAVI iterations are compute
# per optimizer step, so 512 doubles the head's share of a trial against 256.
# Watch trial duration against the ~2.8 h median at 256 and drop the top rung
# if it binds.
#
# It is worth the risk because the answer is genuinely unknown: the PX
# operating point is ~128 iterations (it matches ~1000 plain ones), so a pin at
# 256 is ALREADY above convergence and the extra iterations must be buying
# something other than a better-fitted head -- most likely acting as a
# regulariser through the number of assimilation passes. Rungs above 256
# separate "the ladder is still too low" from "the pin is an artefact of where
# it stopped".
# PINNED at 256, not searched. The ladder rung above asked whether the extra
# iterations past the ~128 PX operating point buy a better-fitted head or act as
# a regulariser. MEASURED 2026-09-08/09, and the answer is NEITHER: they REPAIR A
# WRONG q0, and once q0 is right they do essentially nothing.
#
# The q grid at the tuning horizon, run at both 128 and 256 (job 30079741):
#
#   q_init     nui=128   nui=256      q_init     nui=128   nui=256
#   1.29e-07   0.48420   0.48510      1.29e-03   0.48910   0.49010
#   1.29e-06   0.48180   0.48280      4.09e-03   0.68610   0.53910
#   1.29e-05   0.47770   0.47750      1.29e-02   1.28380   0.55660
#   1.29e-04   0.47450   0.47380
#
# At every q in the good regime the two are within 0.001 -- a wash. In the bad
# regime 256 degrades gracefully where 128 blows up (1.284 -> 0.557). The same
# thing explains the 2-epoch nui sweep, where 256 beat 128 by 0.069 val_nll:
# that sweep ran at the tuned q of 1.292e-3, which is two decades high.
#
# So 256 is chosen for the SEARCH, not for the optimum. `no_learn_q` is on, so
# TPE will visit bad q, and at nui=128 those trials return catastrophic values
# (1.28 against a 0.47 optimum) that distort the surrogate far out of proportion
# to how bad the config is. 256 keeps the objective landscape smooth. It is free:
# across the 8-point large sweep, wall-clock over nui 16..512 was 7:52 to 8:37
# with no monotone trend, i.e. inside node-to-node noise.
#
# Searching it was also costing an axis for nothing -- the first campaign
# collapsed it to a single value across every one of its 20 best trials.
IBPROBIT_UPDATE_ITERS = 256

# Narrowed from probing's [1e-6, 1e-1] to bracket its measured optima
# (2.4e-5 to 4.2e-3) by roughly a decade either side. q0 = alpha^2 / B, so this
# is batch-dependent by construction and the box, not the point, is what the
# probing campaign licenses.
# Floor dropped from 1e-5 to 1e-7 for the second campaign. The first campaign
# drove this axis INTO its floor: rho(val_nll, q_init) = +0.31 on ibprobit and
# +0.49 on ibprobit_noeps (smaller is better on both), and the top-6 q_init
# values span 1.8e-5 to 1.3e-4 against a floor of 1e-5. Two decades of new room
# downward, on an axis TPE explores in log space.
Q_INIT_BOX = (1e-7, 1e-2)

# Narrowed from probing's [1e-7, 1e-2]. vbll's regularization_weight is
# kl_ratio * prior_scale / N, so N (1,020 to 75,750 in probing, 1.28M here) is
# already divided out and the ratio is the N-normalised quantity -- the one
# VBLL axis whose UNITS transfer even though its value does not. Probing's
# optima spanned 2.4e-7 to 7.1e-5.
#
# Ceiling raised from 1e-4 to 1e-2 for the second campaign: the first one
# CLIPPED here. rho(val_nll, kl_ratio) = -0.66 (larger is better) and the top-6
# values are 5.2e-5 to 9.5e-5 -- the whole winning cluster is inside the last
# half-decade below the old ceiling. Probing's optima (2.4e-7 to 7.1e-5) are
# what set that ceiling, and this is the axis where the probing evidence
# transferred worst.
KL_RATIO_BOX = (1e-7, 1e-2)

# Unchanged from probing. Its optima spanned 0.14 to 352 -- a 2500x range with
# no pattern -- so there is nothing here to narrow with.
#
# Floor dropped from 1e-1 to 1e-3. Probing found no pattern, but fine-tuning
# did: rho(val_nll, prior_scale) = +0.79, the strongest correlation on any axis
# in any arm, and the top-6 sit at 0.72 to 2.6 -- heading for the floor. Two
# decades of new room, matching the two added at KL_RATIO_BOX's ceiling.
#
# The two axes multiply (regularization_weight = kl_ratio * prior_scale / N),
# so widening both in OPPOSITE directions leaves the product's reachable range
# roughly where it was while letting TPE find the split. That the search wants
# large kl_ratio AND small prior_scale is itself the finding: the product is
# already near its optimum and the first campaign was spending trials sliding
# along that diagonal into two walls.
PRIOR_SCALE_BOX = (1e-3, 1e3)

# Trunk learning rate and the head's MULTIPLE of it, not two independent rates.
# full_finetune_comparison.py records why they are correlated: AdamW moves
# weights by ~lr/step, and a fresh head's init scale (~0.015) would take
# thousands of 1e-4 steps to overwrite -- its defaults are 1e-4 / 1e-2, a ratio
# of 100. Two independent log-uniform axes spend trials sliding along that
# diagonal. Same reparameterisation `space.py` applies for kl_ratio.
#
# Ceiling narrowed from 1e-3 to 3e-4 for the second campaign. This is the axis
# the first campaign converged HARDEST on: every arm, every top trial, 2e-5 to
# 4e-5, out of a box three decades wide. It is also the axis with the WEAKEST
# rank correlation to the score in all four arms (ce +0.20, ibprobit -0.13,
# noeps -0.18, vbll -0.09) -- which is restriction of range, not unimportance:
# TPE had already collapsed it, so among survivors there was little left to
# correlate.
#
# Only half a decade comes off, and the cut is off the TOP, where the highest
# observed survivor was 9e-5. Deliberately modest: three other axes in this
# file were just found clipped, and a narrow lr box would be a fourth. The
# floor is untouched -- ce at 512 is expected to want roughly half its 1024
# value (2e-6 to 2e-5), near the bottom of the box, not the top.
TRUNK_LR_BOX = (1e-6, 1e-3)

# Weight decay is now PER ARM, because the first campaign found three different
# behaviours on one axis. Best val_nll by wd decade, over all completed trials:
#
#             1e-5..1e-4   1e-4..1e-3   1e-3..1e-2   1e-2..1e-1
#   ce           0.44408      0.44480      0.44887      0.45315   monotone
#   ibprobit     0.44515      0.44273      0.44702      0.44555   flat
#   noeps        0.44695      0.44150      0.44493      0.45155   flat-ish
#   vbll         0.49682      0.48824      0.49117      (0 of 13) interior
#
# So a single shared box is wrong in two directions at once, and a single
# PINNED value would be worse: vbll's interior optimum is worth 0.009 against
# the neighbouring decade, which is three times the entire spread between the
# JAX arms.
#
# Pinning was considered and rejected on a second ground as well. AdamW's decay
# is decoupled, so per-step shrinkage is lr * wd and total shrinkage over a run
# is N * lr * wd. Pinning wd while re-searching lr pins one FACTOR of the
# regulariser, not the regulariser -- the effective decay would then move as a
# side effect of the lr search. Searching the invariant N * lr * wd directly
# was the alternative, and the data does not support it either: the top-5 of
# that quantity spans 100x on ce, 500x on ibprobit and 30x on noeps. Only vbll
# is tight (5x), which is not enough arms to reparameterise on.
#
# What the invariance DOES buy is the batch change: N * lr * wd is unchanged
# when lr scales with B, so these boxes transfer from the first campaign's
# rungs to FIXED_BATCH_SIZE without adjustment.
TRUNK_WD_BOX = (1e-5, 1e-1)

# ce presses the FLOOR: rho(val_nll, wd) = +0.44 over 33 completed trials, the
# best decade is the lowest one, and the winner sits at 2.2e-5 against a floor
# of 1e-5. A decade of new room downward, and the top two decades dropped --
# nothing there survived well (0.44887 and 0.45315 against 0.44408).
CE_WD_BOX = (1e-6, 1e-3)

# vbll has a real interior optimum at 1e-4..1e-3 and nothing survived above
# 1e-2 (0 of 13 sampled). Narrowed to bracket that by a decade either side
# rather than pinned, since rho = -0.43 at n = 16 is suggestive, not settled.
VBLL_WD_BOX = (1e-5, 1e-2)

# Both IBProbit arms cap the trunk rate half a decade lower than CE and VBLL.
#
# The head is assimilated in CLOSED FORM once per optimizer step, from the
# features the trunk produced during that step's accumulation window. A trunk
# moving fast enough leaves those sufficient statistics describing a feature
# distribution the trunk has already left, so the head is always fitting a
# representation one step stale. The optimiser-driven arms have no equivalent
# failure: their head is updated by the same gradient step as the trunk and
# tracks it by construction.
#
# So this is not a shared budget cut applied unevenly -- it is a property of
# the conjugate update, and the asymmetry is reported rather than hidden. The
# lower bound is unchanged, so the arm can still reach every trunk rate the
# others can except the top half-decade.
#
# Ceiling lowered from 5e-4 to 1e-4 alongside TRUNK_LR_BOX's 1e-3 -> 3e-4, so
# the half-decade asymmetry this constant exists to express is preserved rather
# than quietly closed by narrowing only the other box.
IBPROBIT_TRUNK_LR_BOX = (1e-6, 1e-3)

# One ceiling on every arm, and the half-decade asymmetry that used to live
# here is GONE. It was the claim that IBProbit's closed-form head cannot track
# a trunk past some rate, so its box was capped 5x below the others'. Two
# measurements retired it:
#
#   * small100k (ceiling 1e-4): all 8 of IBProbit's top trials drew 91.7-99.9%
#     of the ceiling. The BOX was choosing the rate, not the objective.
#   * small100klr (ceiling 1e-3): the top 8 cluster at 1.28e-4..1.37e-4 --
#     just OUTSIDE the old cap -- and best val NLL improved 1.0131 -> 1.0071.
#
# So the optimum sat a hair above the old ceiling and the cap was costing real
# performance. At 1e-3 the top trials sit at ~13% of the ceiling on every arm,
# i.e. the box binds nowhere and each arm's rate is now the objective's choice.
# Kept as its own name rather than aliased to TRUNK_LR_BOX: the arms having
# equal boxes is a measured RESULT, and a future measurement may separate them
# again.


# The head rate is ALWAYS at least 5x the trunk rate. Because the RATIO is the
# searched axis, this is a floor on the box rather than a rejected-sample
# constraint: every config TPE draws satisfies it by construction, and no trial
# is spent discovering that it should not.
#
# The floor was 1.0, which admitted ratio == 1 -- one uniform learning rate
# across a pretrained 840M-parameter trunk and a randomly initialised head.
# Those two need opposite things: any rate fast enough to overwrite the head's
# ~0.015 init scale inside 1.5 epochs is far too fast for the trunk, and any
# rate gentle enough for the trunk leaves the head near its init. That
# opposition is the whole reason this is parameterised as a ratio, so allowing
# the ratio to collapse to 1 contradicted the parameterisation.
#
# 5 is a floor, not a target: the box still spans to 1e3, and
# full_finetune_comparison's own default sits at 100.
#
# Ceiling raised from 1e3 to 1e4 for the second campaign: vbll clipped here.
# rho(val_nll, head_lr_ratio) = -0.78 (larger is better, the second-strongest
# correlation in the whole campaign) and its top-6 are 378 to 872 -- the
# winning cluster is the top half-decade of the old box. ce is unaffected and
# shares the box anyway: its winners are 20 to 104, nowhere near either bound.
MIN_HEAD_LR_RATIO = 5.0
HEAD_LR_RATIO_BOX = (MIN_HEAD_LR_RATIO, 1e4)


# The SEARCHED coordinate for lr and weight decay is the batch-invariant one:
# lr / sqrt(B), not lr. Same for weight decay, which takes the same exponent
# under the PyTorch AdamW convention (protocol.transfer documents why).
#
# The boxes above stay in ABSOLUTE units at protocol.REFERENCE_BATCH, and the
# division happens here. That is deliberate: every box comment cites measured
# optima in absolute lr, and rewriting them into lr/sqrt(B) units would make
# the evidence unreadable while changing nothing about the search.
#
# At FIXED_BATCH_SIZE == protocol.REFERENCE_BATCH == 512 this is an EXACT
# relabelling -- the same log-uniform draw over the same interval, under a
# different parameter name. It buys two things anyway:
#
#   * The study records the transferable quantity. A finished trial can be
#     read at any batch without knowing which one it ran at.
#   * It is the precondition for un-pinning FIXED_BATCH_SIZE. The reason the
#     batch is pinned is the low-fidelity bias documented there: at the
#     pruner's warmup evaluation the median val_nll is monotone in batch,
#     because a smaller batch takes more optimizer steps per sample and leads
#     early. Under this coordinate that bias should not exist -- batch kappa*B
#     takes kappa fewer steps at sqrt(kappa) larger lr, and the trajectory in
#     normalised time is invariant -- so the pruner would be comparing
#     batch-equivalent configs.
#
# That second claim is NOT yet established and the batch stays pinned until it
# is. The test costs no GPU time: the first campaign (ft_*_shards2m) searched
# the whole ladder and recorded lr, batch and every evaluation, so the question
# "does the eval-10 median flatten in batch once conditioned on lr/sqrt(B)?" is
# a re-analysis of journals that already exist. Note that our own first
# campaign is not obviously consistent with the rule: within ce the best
# observed lr ran 9.1e-6 at 1024, 3.1e-5 at 2048 and 8.7e-5 at 4096, i.e. ~3x
# per doubling against the predicted sqrt(2) = 1.41x. That is one noisy draw
# per rung out of a heavily pruned search rather than a refutation, but it is
# the number the re-analysis has to explain before the pin comes off.
#
# Changing a parameter's NAME changes the search space, so this needs a new
# BLLARSE_FT_CAMPAIGN. Optuna's failure here is the quiet one: `load_if_exists`
# on the old study would leave TPE modelling `lr` from prior trials while the
# objective samples `lr_per_sqrt_batch`, and every old trial would look like it
# simply never set the new axis.
PER_SQRT_BATCH_SUFFIX = "_per_sqrt_batch"

# What `_suggest_betas` appends. Named for the same reason the other suffix is:
# `final_config.absolute_params` has to undo this rename to recover a study's
# betas, and two independent spellings would drift.
HORIZONS_SUFFIX = "_horizons"


def _suggest_per_sqrt_batch(
    trial: Any, name: str, box: tuple[float, float], batch: int
) -> float:
    """Suggest `name` in batch-invariant units; return the absolute value.

    The trial records `<name>_per_sqrt_batch`; the caller gets the quantity the
    training script's flag actually takes.
    """
    root = math.sqrt(batch)
    scaled = trial.suggest_float(
        f"{name}{PER_SQRT_BATCH_SUFFIX}", box[0] / root, box[1] / root, log=True
    )
    return scaled * root


def _suggest_betas(trial: Any, steps: int) -> tuple[float, float]:
    """Draw both AdamW betas in horizon units; return them as betas.

    The trial records `beta1_horizons` / `beta2_horizons`, which transfer; the
    caller gets the values the scripts' --beta1/--beta2 flags take, which do
    not. `protocol.transfer` reproduces this exactly -- moving a config to a
    new step count preserves H by construction -- so a searched beta and a
    transferred one are the same quantity seen at two budgets.
    """
    h1 = trial.suggest_float(f"beta1{HORIZONS_SUFFIX}",
                             *beta1_horizons_box(steps), log=True)
    h2 = trial.suggest_float(f"beta2{HORIZONS_SUFFIX}", *BETA2_HORIZONS_BOX, log=True)
    return (protocol.beta_from_horizons(h1, steps),
            protocol.beta_from_horizons(h2, steps))


def micro_batch(arm: str, model: str = TUNING_MODEL) -> int:
    """The pinned per-forward batch for (`model`, `arm`).

    Raises rather than defaulting OR borrowing another backbone's value. The
    number is hardware- AND model-specific and unrecoverable from a finished
    run's config, so a guess either OOMs mid-run or wastes memory across a
    whole campaign, invisibly. Borrowing is the failure that actually happened.
    """
    flag_arm = "ibprobit" if arm.startswith("ibprobit") else arm
    per_model = MICRO_BATCH.get(model)
    if per_model is None:
        raise KeyError(
            f"no pinned micro batches for model {model!r}; known: "
            f"{sorted(MICRO_BATCH)}")
    if flag_arm not in per_model:
        raise KeyError(
            f"no pinned micro batch for arm {flag_arm!r} on {model!r}. "
            f"Measure it on the target GPU first:\n"
            f"    python scripts/finetune/probe_micro_batch.py "
            f"--arm {flag_arm if flag_arm != 'vbll' else 'ce'} "
            f"--model {model} --bf16\n"
            f"then pin the printed value in finetune_space.MICRO_BATCH."
        )
    return per_model[flag_arm]


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


def _suggest_ibprobit(
    trial: Any, *, search_eps: bool, steps: int,
    batch_size: int = FIXED_BATCH_SIZE,
) -> dict[str, Any]:
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
        # Constant, NOT suggest_categorical over a one-element list: a sampled
        # axis with one value still becomes a dimension TPE models and reports.
        # See FIXED_BATCH_SIZE. The value reaches the study record through the
        # trial's `effective_batch` user attr either way.
        "batch_size": batch_size,
        # IBPROBIT_TRUNK_LR_BOX, not TRUNK_LR_BOX: the closed-form head refits
        # from features the trunk has already moved past once the trunk is fast
        # enough. See the constant.
        "lr": _suggest_per_sqrt_batch(
            trial, "lr", IBPROBIT_TRUNK_LR_BOX, batch_size),
        # Both ibprobit arms keep the wide shared box: wd measured FLAT on this
        # head (rho +0.17 and +0.03, best-of-decade spread 0.004 with no
        # ordering). "Does not matter" is a result here, so it is left free to
        # be re-measured rather than narrowed on noise.
        "weight_decay": _suggest_per_sqrt_batch(
            trial, "weight_decay", TRUNK_WD_BOX, batch_size),
        # q_init is searched at the TUNING horizon and moved to the reported
        # one by protocol.transfer's q0 ~ T^-2 rule, the same way lr and the
        # betas are. Before that rule existed this axis transferred by identity
        # and the tuned value was ~68x too high at 4 epochs.
        "q_init": trial.suggest_float("q_init", *Q_INIT_BOX, log=True),
        # Constant, NOT suggest_categorical over a one-element list -- a sampled
        # axis with one value still becomes a dimension TPE models and reports.
        "num_update_iters": IBPROBIT_UPDATE_ITERS,
    }
    params["beta1"], params["beta2"] = _suggest_betas(trial, steps)
    if search_eps:
        params["eps_init"] = trial.suggest_float("eps_init", *EPS_INIT_BOX, log=True)
    return params


def suggest(
    trial: Any, arm: str, *, max_samples: int = protocol.MAX_SAMPLES,
    batch_size: int = FIXED_BATCH_SIZE,
) -> dict[str, Any]:
    """One TPE-sampled configuration for `arm`.

    Axis counts: ce 3, vbll 5, ibprobit 5, ibprobit_noeps 4 -- one lower than
    the first campaign on every arm, since batch_size is now pinned.

    The RETURNED keys are what the training scripts' flags take (`lr`,
    `weight_decay`, ...). The SEARCHED names differ for two of them: lr and
    weight decay are drawn as `lr_per_sqrt_batch` and
    `weight_decay_per_sqrt_batch`, the batch-invariant coordinates. See
    `_suggest_per_sqrt_batch`. Anything reading a finished study's params has
    to go through that rename.

    `cov_rank` and `parameterization` are NOT searched, following space.py's
    existing rule: rank buys calibration with memory, and neither CE nor
    IBProbit has an equivalent knob -- letting TPE spend it would let VBLL win
    partly by being larger. Rank is reported separately.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")

    # The betas are drawn in horizon units and converted at the trial's OWN
    # step count, so a shortened smoke budget stays self-consistent instead of
    # inheriting the campaign's betas at a budget they are wrong for.
    steps = protocol.steps_for_samples(max_samples, batch_size)

    if arm.startswith("ibprobit"):
        return _suggest_ibprobit(
            trial, search_eps=(arm == "ibprobit"), steps=steps,
            batch_size=batch_size)

    params: dict[str, Any] = {
        # See FIXED_BATCH_SIZE, and _suggest_ibprobit on why it is a constant
        # rather than a one-element categorical.
        "batch_size": batch_size,
        "lr": _suggest_per_sqrt_batch(
            trial, "lr", TRUNK_LR_BOX, batch_size),
        # Per arm: ce presses the floor, vbll has an interior optimum. See
        # CE_WD_BOX / VBLL_WD_BOX.
        "weight_decay": _suggest_per_sqrt_batch(
            trial, "weight_decay",
            VBLL_WD_BOX if arm == "vbll" else CE_WD_BOX, batch_size),
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
    params["beta1"], params["beta2"] = _suggest_betas(trial, steps)
    return params
