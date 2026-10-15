"""The fine-tuning search space.

These are the constraints a misconfigured campaign would otherwise reveal three
hours into its first trial, or -- worse -- never reveal at all, by quietly
searching a different space than the one recorded.
"""

import math

import pytest

from bllarse.finetune import protocol
from bllarse.tuning import finetune_space as fs

# lr and weight decay are SEARCHED in batch-invariant units and RETURNED in
# absolute ones, so a test asserting on the space has to know both names. The
# boxes stay written in absolute units; `searched_box` converts back so every
# assertion below still reads against the constant it is about.
SEARCHED = {
    "lr": "lr" + fs.PER_SQRT_BATCH_SUFFIX,
    "weight_decay": "weight_decay" + fs.PER_SQRT_BATCH_SUFFIX,
    # Betas are searched in HORIZON units -- (1-beta)*steps -- for the same
    # reason: beta is per-step and therefore incomparable across budgets.
    "beta1": "beta1_horizons",
    "beta2": "beta2_horizons",
}
# The subset whose rename also carries a sqrt(batch) rescaling. The beta axes
# are renamed but not rescaled by the batch, so they must not go through it.
PER_SQRT_BATCH = ("lr", "weight_decay")


def searched_box(trial, name, batch=None):
    """`trial`'s recorded axis for `name`, in the box's own absolute units."""
    kind, low, high, log = trial.axes[SEARCHED.get(name, name)]
    if name in PER_SQRT_BATCH:
        root = math.sqrt(fs.FIXED_BATCH_SIZE if batch is None else batch)
        low, high = low * root, high * root
    return kind, low, high, log


class RecordingTrial:
    """Records what was suggested, so a test can assert on the SPACE rather
    than on sampled values. Returns the low end of each range, which keeps the
    returned config usable as a config."""

    def __init__(self):
        self.axes: dict[str, tuple] = {}

    def suggest_float(self, name, low, high, log=False):
        self.axes[name] = ("float", low, high, log)
        return low

    def suggest_categorical(self, name, choices):
        self.axes[name] = ("categorical", tuple(choices))
        return choices[0]


# --- The ladder and the evaluation grid -------------------------------------


def test_every_rung_divides_the_eval_quantum():
    # Evaluation happens BETWEEN optimizer steps, so a quantum that is not a
    # whole number of steps for some rung is unschedulable for that rung --
    # protocol.eval_every_steps raises rather than rounding, which would turn
    # a config error into a silently per-trial evaluation grid.
    for batch in fs.BATCH_SIZES:
        assert protocol.EVAL_QUANTUM % batch == 0, batch


def test_every_rung_gets_the_same_number_of_evaluations():
    # 1,000,000 / 16,384. THE property the sample-spaced grid exists for: eval
    # index i means i * 16,384 samples on every rung, so MedianPruner compares
    # trials at matching amounts of data rather than matching steps.
    assert fs.eval_count() == 61


def test_the_ladder_is_powers_of_two():
    for batch in fs.BATCH_SIZES:
        assert batch & (batch - 1) == 0, batch


# --- The pinned micro batch -------------------------------------------------


def test_micro_batch_refuses_to_guess():
    # Hardware-specific and unrecoverable from a finished run's config, so an
    # unmeasured arm must fail loudly at submit time rather than OOM three
    # hours into a trial or silently leave memory unused all campaign.
    #
    # Checked with an arm that does not exist, so the test asserts the
    # invariant rather than which arms happen to be unmeasured.
    with pytest.raises(KeyError, match="probe_micro_batch"):
        fs.micro_batch("ivon", "dinov3_huge")


def test_the_pinned_micro_batch_matches_the_full_step_measurement():
    # Every value here came from running a FULL step -- optimizer state,
    # accumulation, the update, then an eval chunk -- not a bare micro-grad,
    # which over-reports: an isolated micro-grad measurement says 64 for the
    # JAX arms where the real loop OOMs at 32.
    #
    # The JAX arms on dinov3_huge are 16. Extrapolating ("16 fits a 32 GB
    # card, ~13.5 GB of fixed state plus roughly linear activations, so 32 fits
    # 40 GB") predicts 32, but run directly on the 40 GB card it does not fit:
    #
    #   micro 32, LLRD on  -> OOM, 13.77 GiB in jit_ce_micro_grad
    #   micro 32, LLRD OFF -> OOM, 13.77 GiB -- the SAME failure, which is what
    #                         rules LLRD out as the cause
    #   micro 16, LLRD on  -> runs, 76.60 img/s
    #   micro  8, LLRD on  -> runs, 61.51 img/s
    #
    # 16 over 8 because it is 25% faster and the arithmetic is identical:
    # val_nll 6.3554 against 6.3555 on the same data.
    assert fs.MICRO_BATCH["dinov3_huge"]["ce"] == 16
    assert fs.MICRO_BATCH["dinov3_huge"]["ibprobit"] == 16
    # vbll: 64 on dinov3_large on the same A100, measured the same way
    # (132.1 img/s against 130.0 at 32 and 87.6 at 16), and tuning trials ran
    # at it without an OOM. Twice the JAX value because PyTorch frees
    # activations eagerly as the backward consumes them, where the JAX arms
    # hold a jitted graph's live set plus an explicit grads_acc buffer.
    assert fs.MICRO_BATCH["dinov3_large"]["vbll"] == 64
    # HALF that on huge, where 64 OOMs at model load. The two backbones
    # needing different values is the whole reason this dict is keyed on the
    # model.
    assert fs.MICRO_BATCH["dinov3_huge"]["vbll"] == 32
    assert fs.MICRO_BATCH["dinov3_huge"]["vbll"] < fs.MICRO_BATCH["dinov3_large"]["vbll"]
    assert "A100" in fs.MICRO_BATCH_DEVICE


@pytest.mark.parametrize("model,arm", [
    (m, a) for m, per in sorted(fs.MICRO_BATCH.items()) for a in sorted(per)])
def test_every_pinned_micro_batch_divides_every_rung_of_the_ladder(model, arm):
    """Otherwise accum_plan raises and the arm cannot run at that batch size.

    A power of two no larger than the ladder's gcd, so num_accum is exact
    everywhere.
    """
    micro = fs.MICRO_BATCH[model][arm]
    assert micro & (micro - 1) == 0, f"{model}/{arm}: {micro} is not a power of two"
    for batch in fs.BATCH_SIZES:
        assert batch % micro == 0, f"{model}/{arm}: {micro} does not divide {batch}"


@pytest.mark.parametrize("arm", ["ce", "ibprobit", "ibprobit_noeps"])
def test_the_pinned_values_divide_every_rung(arm):
    micro = fs.micro_batch(arm, "dinov3_large")
    for rung in fs.BATCH_SIZES:
        assert rung % micro == 0, (arm, rung, micro)


def test_any_pinned_micro_batch_divides_every_rung(monkeypatch):
    # protocol.accum_plan raises unless the micro batch divides the target
    # exactly. This pins the invariant so a hand-edited MICRO_BATCH cannot
    # break the bottom rung.
    monkeypatch.setattr(fs, "MICRO_BATCH",
                        {"dinov3_large": {"ce": 64, "ibprobit": 32, "vbll": 128}})
    for arm in fs.ARMS:
        micro = fs.micro_batch(arm, "dinov3_large")
        for rung in fs.BATCH_SIZES:
            assert rung % micro == 0, (arm, rung, micro)


def test_both_ibprobit_arms_share_one_micro_batch(monkeypatch):
    # `ibprobit_noeps` is an ablation of the ibprobit head, not a third
    # training step: same script, same memory profile, one measurement.
    monkeypatch.setattr(fs, "MICRO_BATCH", {"dinov3_large": {"ibprobit": 32}})
    assert fs.micro_batch("ibprobit_noeps", "dinov3_large") == fs.micro_batch(
        "ibprobit", "dinov3_large") == 32


# --- Per-arm axes -----------------------------------------------------------


@pytest.mark.parametrize("arm,expected", [
    ("ce", {"lr", "weight_decay", "head_lr_ratio", "beta1", "beta2"}),
    ("vbll", {"lr", "weight_decay", "head_lr_ratio", "kl_ratio",
              "prior_scale", "beta1", "beta2"}),
    ("ibprobit", {"lr", "weight_decay", "q_init",
                  "eps_init", "beta1", "beta2"}),
    ("ibprobit_noeps", {"lr", "weight_decay", "q_init",
                        "beta1", "beta2"}),
])
def test_each_arm_searches_exactly_its_own_axes(arm, expected):
    trial = RecordingTrial()
    params = fs.suggest(trial, arm)
    # `expected` names the RETURNED keys, which are the training scripts'
    # flags. Two of them are searched under a different name -- see SEARCHED.
    assert set(trial.axes) == {SEARCHED.get(name, name) for name in expected}
    # batch_size is emitted but not SEARCHED, so params carries exactly one key
    # the axes do not. The ibprobit arms' pinned num_update_iters is NOT here:
    # it arrives through finetune_worker.IBPROBIT_FIXED, because a value
    # returned from `suggest` never reaches `trial.params` and so would be lost
    # when final_config rebuilds the reported run.
    assert set(params) == expected | {"batch_size"}


@pytest.mark.parametrize("arm", fs.ARMS)
def test_the_batch_is_pinned_rather_than_sampled(arm):
    """A constant, not a one-element categorical.

    The distinction is not cosmetic: a suggested axis with a single choice is
    still a dimension TPE models and still appears in `trial.params` as though
    it had been searched, which is exactly the misreading FIXED_BATCH_SIZE
    exists to prevent.
    """
    trial = RecordingTrial()
    params = fs.suggest(trial, arm)
    assert params["batch_size"] == fs.FIXED_BATCH_SIZE == 512
    assert "batch_size" not in trial.axes


def test_the_pinned_batch_is_a_rung_the_protocol_can_schedule():
    # Not merely on the ladder: it has to divide the quantum, or every trial
    # dies in protocol.eval_every_steps.
    assert fs.FIXED_BATCH_SIZE in fs.BATCH_SIZES
    assert protocol.EVAL_QUANTUM % fs.FIXED_BATCH_SIZE == 0


@pytest.mark.parametrize("arm", fs.ARMS)
def test_the_pinned_batch_divides_by_every_arms_micro_batch(arm):
    # protocol.accum_plan raises otherwise, at the top of a ~3 h trial -- and
    # with the batch pinned there is no other rung for TPE to escape to, so a
    # mismatch would kill the whole campaign rather than one trial.
    assert fs.FIXED_BATCH_SIZE % fs.micro_batch(arm, "dinov3_large") == 0


def test_the_betas_are_searched_rather_than_left_at_the_defaults():
    """Searched in horizon units, the transfer-invariant coordinate.

    protocol.horizons moves a beta across step counts, so a drawn beta means
    the same thing at the tuning budget and at the final one.
    """
    trial = RecordingTrial()
    params = fs.suggest(trial, "ce")
    assert "beta1_horizons" in trial.axes and "beta2_horizons" in trial.axes
    assert "beta1" not in trial.axes and "beta2" not in trial.axes
    steps = protocol.steps_for_samples(protocol.MAX_SAMPLES, fs.FIXED_BATCH_SIZE)
    for name, box in (("beta1", fs.BETA1_HORIZONS_BOX),
                      ("beta2", fs.BETA2_HORIZONS_BOX)):
        # RecordingTrial returns the low end, so the drawn beta is the one the
        # box floor implies -- and inverting it recovers the floor exactly.
        assert params[name] == pytest.approx(
            protocol.beta_from_horizons(box[0], steps))
        assert protocol.horizons(params[name], steps) == pytest.approx(box[0])


def test_neither_ibprobit_arm_gets_a_head_learning_rate():
    # There is no head optimizer to receive one -- finetune_jax.check_arm_flags
    # rejects --head-lr for this arm outright. Sampling it would present a
    # tuned axis that silently does nothing AND make TPE model a dimension the
    # runs do not have.
    for arm in ("ibprobit", "ibprobit_noeps"):
        assert "head_lr_ratio" not in fs.suggest(RecordingTrial(), arm)


def test_the_noeps_arm_never_samples_eps():
    # Dropped from the SEARCH rather than sampled-and-overridden: a sampled
    # axis that does not reach the training script makes the study's own record
    # of what it searched wrong.
    assert "eps_init" not in fs.suggest(RecordingTrial(), "ibprobit_noeps")


def test_vbll_does_not_search_wishart_scale():
    # Pinned at WISHART_SCALE -- the one head knob whose probing optima agreed
    # across datasets (0.076-0.204, a 2.7x spread).
    assert "wishart_scale" not in fs.suggest(RecordingTrial(), "vbll")
    assert fs.WISHART_SCALE == pytest.approx(0.13)


def test_vbll_does_not_search_capacity():
    # cov_rank and parameterization stay fixed: rank buys calibration with
    # memory and neither CE nor IBProbit has an equivalent knob, so letting TPE
    # spend it would let VBLL win Fig 1 partly by being larger.
    axes = set(fs.suggest(RecordingTrial(), "vbll"))
    assert "cov_rank" not in axes
    assert "parameterization" not in axes


# --- The boxes --------------------------------------------------------------


def test_eps_box_reaches_far_below_what_probing_selected():
    # Probing's optima spanned 1.1e-3 to 3.3e-2. The floor is FIVE decades
    # lower so that "the head switched contamination off when free to keep it"
    # is a reachable, readable outcome of the SEARCHED arm -- a stronger result
    # than a separately-tuned pinned arm winning, and free on a log-space axis.
    trial = RecordingTrial()
    fs.suggest(trial, "ibprobit")
    kind, low, high, log = trial.axes["eps_init"]
    assert (low, high, log) == (1e-8, 1e-1, True)


def test_the_pinned_and_searched_eps_stay_distinguishable():
    # At least three decades apart (four as pinned), so a trial that lands on
    # the searched floor is still clearly not the numerically-absent arm.
    assert fs.IBPROBIT_NOEPS_INIT < fs.EPS_INIT_BOX[0] / 1e3


def test_the_cavi_iterations_are_pinned_not_searched():
    # The q grid run at both 128 and 256 puts them within 0.001 val_nll at
    # every q in the good regime, so the axis buys nothing there -- and a
    # search over it collapsed to one value across all 20 of its best trials
    # anyway.
    assert fs.IBPROBIT_UPDATE_ITERS == 256
    trial = RecordingTrial()
    params = fs.suggest(trial, "ibprobit")
    # Pinned means ABSENT from the search space, not a one-element categorical:
    # a sampled axis with one value is still a dimension TPE models.
    assert "num_update_iters" not in trial.axes
    # And absent from the RETURNED params too. A value returned here reaches the
    # tuning run but never `trial.params`, so `final_config` would rebuild
    # without it -- reporting at finetune_jax's default of 128 a config selected
    # at 256. It arrives instead through finetune_worker.IBPROBIT_FIXED.
    assert "num_update_iters" not in params


def test_the_pinned_cavi_value_is_at_or_above_the_px_operating_point():
    # PX matches ~1000 plain iterations at ~128, so a pin below that would
    # measure an unconverged head rather than the method. 256 is deliberately
    # above it: in the bad-q regime the extra passes degrade gracefully
    # (0.5566 at q=1.29e-2) where 128 blows up (1.2838), which keeps TPE's
    # surrogate from being distorted by trials that are merely mis-seeded.
    assert fs.IBPROBIT_UPDATE_ITERS >= 128


def test_learning_rate_is_searched_as_trunk_rate_and_ratio():
    # Not two independent rates: AdamW moves weights by ~lr/step and a fresh
    # head needs ~100x the trunk rate, so independent log-uniform axes spend
    # trials sliding along that diagonal.
    trial = RecordingTrial()
    fs.suggest(trial, "ce")
    assert searched_box(trial, "lr") == ("float", *fs.TRUNK_LR_BOX, True)
    assert trial.axes["head_lr_ratio"] == ("float", *fs.HEAD_LR_RATIO_BOX, True)
    assert "head_lr" not in trial.axes


@pytest.mark.parametrize("arm", ["ce", "vbll"])
def test_the_head_rate_is_always_at_least_five_times_the_trunk_rate(arm):
    """A floor on the BOX, so every drawn config satisfies it by construction.

    Ratio == 1 is one uniform rate across a pretrained 840M-parameter trunk and
    a randomly initialised head, which need opposite things: fast enough to
    overwrite the head's ~0.015 init scale within a fine-tuning run is far too
    fast for the trunk. Allowing the ratio to collapse to 1 would contradict
    the reason this is parameterised as a ratio at all.
    """
    trial = RecordingTrial()
    fs.suggest(trial, arm)
    _kind, low, high, _log = trial.axes["head_lr_ratio"]
    assert low >= fs.MIN_HEAD_LR_RATIO == 5.0
    assert high > low
    # RecordingTrial returns the low end, so the worst case a trial can draw
    # is exactly the floor -- and even that keeps the head 5x faster.
    assert low * 1.0 >= 5.0


def test_tuning_seed_is_disjoint_from_the_reported_seeds():
    # Selecting hyperparameters on the seeds that later produce the error bars
    # would tune against the same random draws being reported.
    assert fs.TUNING_SEED not in fs.REPORTED_SEEDS


def test_suggest_rejects_an_unknown_arm():
    with pytest.raises(ValueError, match="unknown arm"):
        fs.suggest(RecordingTrial(), "ivon")


def test_the_ladder_reaches_the_eval_quantum():
    """The ceiling is the quantum, and nothing else.

    A rung larger than EVAL_QUANTUM cannot divide it, so the evaluation grid
    would be unschedulable there. 16,384 divides itself, so it is the largest
    batch this protocol can carry -- and memory does not enter into it, since
    gradient accumulation absorbs the effective batch entirely.
    """
    assert max(fs.BATCH_SIZES) == protocol.EVAL_QUANTUM == 16_384


def test_the_top_rung_evaluates_after_every_optimizer_step():
    # At the ceiling a step IS a quantum, so eval_every collapses to 1 and the
    # MAX_SAMPLES budget is 61 steps. A genuinely different training regime,
    # and one TPE is allowed to explore rather than have prejudged for it:
    # batch tolerance is a property of the method.
    assert protocol.eval_every_steps(16_384) == 1
    assert protocol.steps_for_samples(protocol.MAX_SAMPLES, 16_384) == 61


def test_the_pinned_micro_batch_divides_the_new_top_rungs(monkeypatch):
    monkeypatch.setattr(fs, "MICRO_BATCH", {"dinov3_large": {"ce": 32, "ibprobit": 32}})
    for rung in (8192, 16384):
        assert rung % fs.micro_batch("ce", "dinov3_large") == 0
        protocol.accum_plan(rung, fs.micro_batch("ce", "dinov3_large"))


# --- The batch-invariant search coordinate -----------------------------------


def test_the_reference_batch_matches_the_pinned_one():
    """protocol duplicates 512 rather than importing it -- finetune_space
    imports protocol, so the dependency cannot run the other way. This is the
    check that keeps the duplicate honest: if they diverge, every transferred
    config is scaled from a batch the campaign never ran at."""
    assert protocol.REFERENCE_BATCH == fs.FIXED_BATCH_SIZE == 512


@pytest.mark.parametrize("arm", fs.ARMS)
def test_lr_and_weight_decay_are_searched_in_batch_invariant_units(arm):
    # The SEARCHED name carries the suffix; the RETURNED key is the flag the
    # training scripts take. Anything reading a study's params has to know it.
    trial = RecordingTrial()
    params = fs.suggest(trial, arm)
    for name in ("lr", "weight_decay"):
        assert SEARCHED[name] in trial.axes
        assert name not in trial.axes
        assert name in params


@pytest.mark.parametrize("arm", fs.ARMS)
def test_the_returned_value_is_absolute_not_the_searched_coordinate(arm):
    # RecordingTrial returns the low end, so the returned lr must be the box's
    # floor -- not the floor divided by sqrt(512), which is what would ship if
    # the multiply-back were dropped.
    trial = RecordingTrial()
    params = fs.suggest(trial, arm)
    box = fs.IBPROBIT_TRUNK_LR_BOX if arm.startswith("ibprobit") else fs.TRUNK_LR_BOX
    assert params["lr"] == pytest.approx(box[0])


def test_at_the_pinned_batch_the_reparameterisation_is_an_exact_relabelling():
    """Which is the point: it changes the recorded coordinate, not the search.

    FIXED_BATCH_SIZE equals protocol.REFERENCE_BATCH, so the interval TPE draws
    from is the absolute box divided by a constant -- the same log-uniform
    measure.
    What it buys is a study whose params can be read at any batch, and the
    precondition for un-pinning the batch at all.
    """
    trial = RecordingTrial()
    fs.suggest(trial, "ce")
    _kind, low, high, log = searched_box(trial, "lr")
    assert (low, high, log) == (*fs.TRUNK_LR_BOX, True)
    raw = trial.axes[SEARCHED["lr"]]
    assert raw[1] == pytest.approx(fs.TRUNK_LR_BOX[0] / math.sqrt(512))


def test_the_coordinate_scales_the_absolute_rate_with_the_batch():
    """The invariance the rule actually asserts: one tilde, sqrt(B) absolute.

    A fixed searched value must map to a 2x larger learning rate at 4x the
    batch. This is what makes the axis transferable, and it is checked through
    the helper rather than by arithmetic so a change to the exponent fails.
    """
    at_512 = fs._suggest_per_sqrt_batch(RecordingTrial(), "lr", (1e-6, 1e-3), 512)
    at_2048 = fs._suggest_per_sqrt_batch(RecordingTrial(), "lr", (1e-6, 1e-3), 2048)
    # Same box floor in absolute units either way -- the box IS absolute.
    assert at_512 == pytest.approx(1e-6)
    assert at_2048 == pytest.approx(1e-6)
    # ... and a fixed SEARCHED value maps to sqrt(4) = 2x the rate.
    trial_512 = RecordingTrial()
    fs._suggest_per_sqrt_batch(trial_512, "lr", (1e-6, 1e-3), 512)
    tilde = trial_512.axes[SEARCHED["lr"]][1]
    assert tilde * math.sqrt(2048) == pytest.approx(1e-6 * 2.0)


def test_the_beta_anchor_does_not_move_with_the_tuning_budget():
    """REFERENCE_STEPS is a literal, and this is why.

    Derived as steps_for_samples(MAX_SAMPLES, REFERENCE_BATCH) it would move
    with the tuning budget: at the current one "the reference betas" would
    mean (0.9, 0.999) at 1953 steps instead of 3906 -- a different H on both
    axes, and every transfer anchored to it would move with no diff to show
    for it. The anchor is a fixed operating point (2M samples at batch 512),
    so it is frozen as one.
    """
    assert protocol.REFERENCE_STEPS == protocol.reference_steps() == 3906
    assert protocol.steps_for_samples(
        protocol.MAX_SAMPLES, fs.FIXED_BATCH_SIZE) == 1953
    assert protocol.betas_for(3906) == protocol.REFERENCE_BETAS


def test_the_top_rungs_cannot_carry_a_transferred_beta():
    """A property of those rungs, recorded rather than worked around.

    At batch 16,384 the MAX_SAMPLES budget is 61 optimizer steps, so beta1's
    10-step averaging horizon is 16% of the whole run against 0.5% at the
    pinned rung. Holding the fraction would need beta1 < 0. This is the
    breakdown the SDE rule's own authors predict at coarse discretisation, and
    it is a further reason -- independent of the low-fidelity bias in
    FIXED_BATCH_SIZE -- that the top of the ladder is not simply a bigger
    version of the bottom.
    """
    # At MAX_SAMPLES the breakdown reaches down to 4096.
    for rung in (4096, 8192, 16384):
        steps = protocol.steps_for_samples(protocol.MAX_SAMPLES, rung)
        with pytest.raises(ValueError, match="beta1 transfers out of range"):
            protocol.betas_for(steps)
    # ... while the pinned rung and everything below it is fine.
    for rung in (256, 512, 1024, 2048):
        protocol.betas_for(protocol.steps_for_samples(protocol.MAX_SAMPLES, rung))


def test_beta1_box_is_unchanged_at_the_campaign_budget():
    """1953 steps is the default tuning budget, and the box must be reachable
    there exactly as stated."""
    steps = protocol.steps_for_samples(protocol.MAX_SAMPLES, fs.FIXED_BATCH_SIZE)
    assert steps == 1953
    assert fs.beta1_horizons_box(steps) == fs.BETA1_HORIZONS_BOX


def test_beta1_box_narrows_where_the_constant_would_be_unreachable():
    # 100k samples at batch 128 = 781 steps. beta_from_horizons raises for
    # H >= steps, so the 780 ceiling sits ON the boundary -- the sampler could
    # draw a point that cannot be realised and kill the trial.
    lo, hi = fs.beta1_horizons_box(781)

    assert lo == fs.BETA1_HORIZONS_BOX[0]
    assert hi < 781
    assert hi == pytest.approx(0.4 * 781)


def test_beta1_box_refuses_a_budget_too_short_to_search_b1_at_all():
    with pytest.raises(ValueError, match="too short"):
        fs.beta1_horizons_box(50)


def test_suggest_honours_a_non_default_batch():
    import optuna

    study = optuna.create_study()
    params = fs.suggest(study.ask(), "ce", max_samples=100_000, batch_size=128)

    assert params["batch_size"] == 128


# --- One trunk-rate ceiling on every arm ------------------------------------


@pytest.mark.parametrize("arm", ["ce", "vbll", "ibprobit", "ibprobit_noeps"])
def test_every_arm_shares_the_same_trunk_rate_box(arm):
    """No lower cap on IBProbit.

    A cap below the others' would encode the claim that the closed-form head
    cannot track a trunk past some rate, and measurement refutes it: at
    ceiling 1e-4 all 8 of IBProbit's top trials drew 91.7-99.9% of the cap,
    and at 1e-3 the top 8 land at 1.28e-4..1.37e-4 -- just outside the lower
    cap -- improving best val NLL from 1.0131 to 1.0071. A 1e-4 box, not the
    objective, chooses the rate.
    """
    trial = RecordingTrial()
    fs.suggest(trial, arm)
    _kind, low, high, log = searched_box(trial, "lr")
    assert (low, high, log) == (1e-6, 1e-3, True)


def test_the_arms_are_equal_by_measurement_not_by_aliasing():
    # Equal boxes are a RESULT, so the two names stay distinct: a future
    # measurement may separate them again, and aliasing would hide that the
    # question was ever asked.
    assert fs.IBPROBIT_TRUNK_LR_BOX == fs.TRUNK_LR_BOX
    # Referenced through its OWN name at the draw site, so separating the arms
    # again is a one-line constant change rather than a re-plumbing.
    import inspect
    assert "IBPROBIT_TRUNK_LR_BOX" in inspect.getsource(fs._suggest_ibprobit)


def test_the_ceiling_is_not_binding_for_any_arm():
    """1e-3 is above every optimum measured so far, with room to spare.

    The ceiling is there so the box does not choose the rate. Recorded
    here as the fact that justifies the number: the best trunk rates measured
    are CE 4.4e-5, VBLL 2.9e-5 and IBProbit 1.3e-4, all at most 13% of it.
    """
    for measured in (4.4e-5, 2.9e-5, 1.3e-4):
        assert measured < 0.2 * fs.TRUNK_LR_BOX[1]
