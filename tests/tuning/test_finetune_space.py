"""The fine-tuning search space.

These are the constraints a misconfigured campaign would otherwise reveal three
hours into its first trial, or -- worse -- never reveal at all, by quietly
searching a different space than the one recorded.
"""

import pytest

from bllarse.finetune import protocol
from bllarse.tuning import finetune_space as fs


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
    ("ce", {"batch_size", "lr", "weight_decay", "head_lr_ratio"}),
    ("vbll", {"batch_size", "lr", "weight_decay", "head_lr_ratio",
              "kl_ratio", "prior_scale"}),
    ("ibprobit", {"batch_size", "lr", "weight_decay", "q_init",
                  "num_update_iters", "eps_init"}),
    ("ibprobit_noeps", {"batch_size", "lr", "weight_decay", "q_init",
                        "num_update_iters"}),
])
def test_each_arm_searches_exactly_its_own_axes(arm, expected):
    trial = RecordingTrial()
    params = fs.suggest(trial, arm)
    assert set(trial.axes) == expected
    assert set(params) == expected


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
    assert fs.IBPROBIT_UPDATE_ITERS == (128, 192, 256)


def test_learning_rate_is_searched_as_trunk_rate_and_ratio():
    # Not two independent rates: AdamW moves weights by ~lr/step and a fresh
    # head needs ~100x the trunk rate, so independent log-uniform axes spend
    # trials sliding along that diagonal.
    trial = RecordingTrial()
    fs.suggest(trial, "ce")
    assert trial.axes["lr"] == ("float", 1e-6, 1e-3, True)
    assert trial.axes["head_lr_ratio"] == ("float", 5.0, 1e3, True)
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
    _kind, low, high, log = trial.axes["lr"]
    assert (low, high, log) == (1e-6, 5e-4, True)


@pytest.mark.parametrize("arm", ["ce", "vbll"])
def test_the_optimiser_driven_arms_keep_the_full_trunk_range(arm):
    # The cap is a property of the conjugate update, not a budget cut applied
    # unevenly -- so it must NOT leak onto the arms that do not need it.
    trial = RecordingTrial()
    fs.suggest(trial, arm)
    _kind, low, high, log = trial.axes["lr"]
    assert (low, high, log) == (1e-6, 1e-3, True)


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
