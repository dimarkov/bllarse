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
}


def searched_box(trial, name, batch=None):
    """`trial`'s recorded axis for `name`, in the box's own absolute units."""
    kind, low, high, log = trial.axes[SEARCHED.get(name, name)]
    if name in SEARCHED:
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
    # 2,000,000 / 16,384. THE property the sample-spaced grid exists for: eval
    # index i means i * 16,384 samples on every rung, so MedianPruner compares
    # trials at matching amounts of data rather than matching steps.
    assert fs.eval_count() == 122


def test_the_ladder_is_powers_of_two():
    for batch in fs.BATCH_SIZES:
        assert batch & (batch - 1) == 0, batch


# --- The pinned micro batch -------------------------------------------------


def test_micro_batch_refuses_to_guess():
    # Hardware-specific and unrecoverable from a finished run's config, so an
    # unmeasured arm must fail loudly at submit time rather than OOM three
    # hours into a trial or silently leave memory unused all campaign.
    #
    # Checked with an arm that does not exist rather than with `vbll`, which
    # this used to use: vbll HAS been measured now, so pinning it as the live
    # example made the test assert "vbll is unmeasured" -- a fact, not an
    # invariant, and one that expired the moment the probe ran.
    with pytest.raises(KeyError, match="probe_micro_batch"):
        fs.micro_batch("ivon")


def test_the_pinned_micro_batch_matches_the_full_step_measurement():
    # Every value here came from running a FULL step -- optimizer state,
    # accumulation, the update, then an eval chunk -- not a bare micro-grad.
    # That distinction cost a campaign launch once: an isolated micro-grad
    # measurement reported 64 for the JAX arms and the real loop OOMed at 32.
    #
    # JAX arms: 32, 64 OOMs. Cross-checks against micro=16 having run on a
    # 32 GB card -- ~13.5 GB of fixed state plus activations scaling about
    # linearly means 16 fits 32 GB and 32 fits 40 GB while 64 does not.
    assert fs.MICRO_BATCH["ce"] == 32
    assert fs.MICRO_BATCH["ibprobit"] == 32
    # vbll: 64 on the same A100, measured the same way (132.1 img/s against
    # 130.0 at 32 and 87.6 at 16). Twice the JAX value because PyTorch frees
    # activations eagerly as the backward consumes them, where the JAX arms
    # hold a jitted graph's live set plus an explicit grads_acc buffer.
    assert fs.MICRO_BATCH["vbll"] == 64
    assert "A100" in fs.MICRO_BATCH_DEVICE


@pytest.mark.parametrize("arm", sorted(fs.MICRO_BATCH))
def test_every_pinned_micro_batch_divides_every_rung_of_the_ladder(arm):
    """Otherwise accum_plan raises and the arm cannot run at that batch size.

    Design A8: a power of two no larger than the ladder's gcd, so num_accum is
    exact everywhere. 64 satisfies this as readily as 32 -- which is what made
    the vbll measurement adoptable without touching the ladder.
    """
    micro = fs.MICRO_BATCH[arm]
    assert micro & (micro - 1) == 0, f"{arm}: {micro} is not a power of two"
    for batch in fs.BATCH_SIZES:
        assert batch % micro == 0, f"{arm}: {micro} does not divide {batch}"


@pytest.mark.parametrize("arm", ["ce", "ibprobit", "ibprobit_noeps"])
def test_the_pinned_values_divide_every_rung(arm):
    micro = fs.micro_batch(arm)
    for rung in fs.BATCH_SIZES:
        assert rung % micro == 0, (arm, rung, micro)


def test_any_pinned_micro_batch_divides_every_rung(monkeypatch):
    # protocol.accum_plan raises unless the micro batch divides the target
    # exactly. This pins the invariant so a hand-edited MICRO_BATCH cannot
    # break the bottom rung.
    monkeypatch.setattr(fs, "MICRO_BATCH", {"ce": 64, "ibprobit": 32, "vbll": 128})
    for arm in fs.ARMS:
        micro = fs.micro_batch(arm)
        for rung in fs.BATCH_SIZES:
            assert rung % micro == 0, (arm, rung, micro)


def test_both_ibprobit_arms_share_one_micro_batch(monkeypatch):
    # `ibprobit_noeps` is an ablation of the ibprobit head, not a third
    # training step: same script, same memory profile, one measurement.
    monkeypatch.setattr(fs, "MICRO_BATCH", {"ibprobit": 32})
    assert fs.micro_batch("ibprobit_noeps") == fs.micro_batch("ibprobit") == 32


# --- Per-arm axes -----------------------------------------------------------


@pytest.mark.parametrize("arm,expected", [
    ("ce", {"lr", "weight_decay", "head_lr_ratio"}),
    ("vbll", {"lr", "weight_decay", "head_lr_ratio", "kl_ratio",
              "prior_scale"}),
    ("ibprobit", {"lr", "weight_decay", "q_init", "num_update_iters",
                  "eps_init"}),
    ("ibprobit_noeps", {"lr", "weight_decay", "q_init", "num_update_iters"}),
])
def test_each_arm_searches_exactly_its_own_axes(arm, expected):
    trial = RecordingTrial()
    params = fs.suggest(trial, arm)
    # `expected` names the RETURNED keys, which are the training scripts'
    # flags. Two of them are searched under a different name -- see SEARCHED.
    assert set(trial.axes) == {SEARCHED.get(name, name) for name in expected}
    # batch_size is emitted but not SEARCHED, so params carries exactly one
    # key the axes do not.
    assert set(params) == expected | {"batch_size"}


@pytest.mark.parametrize("arm", fs.ARMS)
def test_the_batch_is_pinned_rather_than_sampled(arm):
    """A constant, not a one-element categorical.

    The distinction is not cosmetic: a suggested axis with a single choice is
    still a dimension TPE models and still appears in `trial.params` as though
    it had been searched, which is exactly the misreading FIXED_BATCH_SIZE
    exists to prevent for the second campaign's studies.
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
    # with the batch pinned there is no longer another rung for TPE to escape
    # to, so a mismatch would kill the whole campaign rather than one trial.
    assert fs.FIXED_BATCH_SIZE % fs.micro_batch(arm) == 0


def test_the_pinned_batch_leaves_the_default_betas_in_their_regime():
    """Why fixing the batch is what makes fixing the betas safe.

    b2 = 0.999 averages over 1/(1-b2) = 1000 steps. At 2M samples the run is
    MAX_SAMPLES / batch steps, so the first campaign's ce optimum of 2048 gave
    976 -- less than one horizon, i.e. the second moment never left bias
    correction. This asserts the chosen rung clears the horizon severalfold.
    """
    steps = protocol.MAX_SAMPLES / fs.FIXED_BATCH_SIZE
    horizon = 1.0 / (1.0 - fs.BETAS[1])
    assert steps / horizon >= 3.0


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
    # Six decades apart, so a trial that lands on the searched floor is still
    # clearly not the numerically-absent arm.
    assert fs.IBPROBIT_NOEPS_INIT < fs.EPS_INIT_BOX[0] / 1e3


def test_cavi_ladder_starts_at_the_px_operating_point():
    # 32 and 64 are dropped: PX matches ~1000 plain iterations at ~128, so
    # rungs below it measure an unconverged head rather than the method.
    assert fs.IBPROBIT_UPDATE_ITERS[0] == 128


def test_the_cavi_ladder_extends_past_where_it_last_pinned():
    # 256 was selected in 5 of the first campaign's top 6 ibprobit trials --
    # the same pinned-to-its-ceiling signal that moved this ladder up once
    # already (probing pinned at 192 in 4 of 5 studies). A ladder whose top
    # rung keeps winning has not been given room to lose.
    assert max(fs.IBPROBIT_UPDATE_ITERS) > 256


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
    overwrite the head's ~0.015 init scale inside 1.5 epochs is far too fast
    for the trunk. Allowing the ratio to collapse to 1 contradicted the reason
    this is parameterised as a ratio at all.
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


@pytest.mark.parametrize("arm", ["ibprobit", "ibprobit_noeps"])
def test_ibprobit_caps_the_trunk_rate_half_a_decade_below_the_others(arm):
    """The closed-form head cannot track a fast trunk.

    IBProbit assimilates its head once per optimizer step from the features
    that step's accumulation window produced. Past some trunk rate those
    sufficient statistics describe a representation the trunk has already left,
    so the head is permanently one step stale. The optimiser-driven arms have
    no equivalent failure -- their head is updated by the same gradient step as
    the trunk.
    """
    trial = RecordingTrial()
    fs.suggest(trial, arm)
    _kind, low, high, log = searched_box(trial, "lr")
    assert (low, high, log) == (*fs.IBPROBIT_TRUNK_LR_BOX, True)


@pytest.mark.parametrize("arm", ["ce", "vbll"])
def test_the_optimiser_driven_arms_keep_the_full_trunk_range(arm):
    # The cap is a property of the conjugate update, not a budget cut applied
    # unevenly -- so it must NOT leak onto the arms that do not need it.
    trial = RecordingTrial()
    fs.suggest(trial, arm)
    _kind, low, high, log = searched_box(trial, "lr")
    assert (low, high, log) == (*fs.TRUNK_LR_BOX, True)


def test_only_the_top_of_the_trunk_range_differs_between_arms():
    # Same floor on every arm: the cap removes the top half-decade and nothing
    # else, so IBProbit can still reach every trunk rate the others can bar
    # that slice.
    assert fs.IBPROBIT_TRUNK_LR_BOX[0] == fs.TRUNK_LR_BOX[0]
    assert fs.IBPROBIT_TRUNK_LR_BOX[1] < fs.TRUNK_LR_BOX[1]


def test_the_ladder_reaches_the_eval_quantum():
    """The ceiling is the quantum, and nothing else.

    A rung larger than EVAL_QUANTUM cannot divide it, so the evaluation grid
    would be unschedulable there. 16,384 divides itself, so it is the largest
    batch this protocol can carry -- and memory does not enter into it, since
    gradient accumulation absorbs the effective batch entirely (design A8).
    The design doc's "here the ceiling is memory" capped this at 4096, which
    also excluded the one large-batch operating point the codebase has real
    evidence for: the earlier production runs used 16,384.
    """
    assert max(fs.BATCH_SIZES) == protocol.EVAL_QUANTUM == 16_384


def test_the_top_rung_evaluates_after_every_optimizer_step():
    # At the ceiling a step IS a quantum, so eval_every collapses to 1 and the
    # 2M budget is 122 steps. A genuinely different training regime, and one
    # TPE is allowed to explore rather than have prejudged for it (design D5:
    # batch tolerance is a method property).
    assert protocol.eval_every_steps(16_384) == 1
    assert protocol.steps_for_samples(protocol.MAX_SAMPLES, 16_384) == 122


def test_the_pinned_micro_batch_divides_the_new_top_rungs(monkeypatch):
    monkeypatch.setattr(fs, "MICRO_BATCH", {"ce": 32, "ibprobit": 32})
    for rung in (8192, 16384):
        assert rung % fs.micro_batch("ce") == 0
        protocol.accum_plan(rung, fs.micro_batch("ce"))


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
    from is the old one divided by a constant -- the same log-uniform measure.
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


def test_the_tuning_budget_is_the_beta_reference_point():
    """Why the campaign can leave AdamW's betas at the library defaults.

    The betas are a per-STEP timescale, so (0.9, 0.999) is not a neutral choice
    -- it is the correct one at 3906 steps and wrong elsewhere. The campaign
    passes no --beta1/--beta2 because MAX_SAMPLES at FIXED_BATCH_SIZE lands
    exactly on the reference. Move either constant and this fails, which is the
    signal that finetune_worker.trial_config has to start emitting them.
    """
    steps = protocol.steps_for_samples(protocol.MAX_SAMPLES, fs.FIXED_BATCH_SIZE)
    assert steps == protocol.reference_steps() == 3906
    assert protocol.betas_for(steps) == protocol.REFERENCE_BETAS


def test_the_top_rungs_cannot_carry_a_transferred_beta():
    """A property of those rungs, recorded rather than worked around.

    At batch 16,384 the 2M budget is 122 optimizer steps, so beta1's 10-step
    averaging horizon is 8% of the whole run against 0.26% at the pinned rung.
    Holding the fraction would need beta1 < 0. This is the breakdown the SDE
    rule's own authors predict at coarse discretisation, and it is a further
    reason -- independent of the low-fidelity bias in FIXED_BATCH_SIZE -- that
    the top of the ladder is not simply a bigger version of the bottom.
    """
    for rung in (8192, 16384):
        steps = protocol.steps_for_samples(protocol.MAX_SAMPLES, rung)
        with pytest.raises(ValueError, match="beta1 transfers out of range"):
            protocol.betas_for(steps)
    # ... while every rung the campaign can actually reach is fine.
    for rung in (256, 512, 1024, 2048, 4096):
        protocol.betas_for(protocol.steps_for_samples(protocol.MAX_SAMPLES, rung))
