import pytest

from bllarse.tuning import space


# The spec's table, verbatim. Any drift here is a silent change to what the
# campaign searched, so it is pinned rather than recomputed.
EXPECTED_CAPS = {
    "oxford_pets": 1024,
    "dtd": 1024,
    # 256, not 2048: flowers102 trains on the official 1,020-image split, not
    # the 7,169-image val+test pool the upstream repo calls "train". See
    # bllarse.features._TRAIN_SPLIT.
    "flowers102": 256,
    "stanford_cars": 2048,
    # 16384, not the 32768 its 75,750 training images would otherwise allow:
    # MAX_BATCH caps the ladder globally. The cap is also the eval quantum, and
    # a quantum of 32768 would make food101's val_nll curve twice as coarse as
    # every other dataset's.
    "food101": 16384,
    "cifar10": 16384,
    "cifar100": 16384,
}


@pytest.mark.parametrize("dataset,cap", sorted(EXPECTED_CAPS.items()))
def test_batch_cap_matches_the_spec_table(dataset, cap):
    assert space.batch_cap(space.N_TRAIN[dataset]) == cap


@pytest.mark.parametrize("dataset", sorted(EXPECTED_CAPS))
def test_every_cap_leaves_at_least_two_steps_per_epoch(dataset):
    """The floor is 2, not 4, so that 1024 is reachable on dtd and oxford_pets
    -- IBProbit is expected to favour larger batches, and a batch axis that
    stops at a different point per arm would confound the comparison with the
    search space. Below 2, drop-remainder makes an epoch a single batch and the
    permutation stops doing anything."""
    cap = space.batch_cap(space.N_TRAIN[dataset])
    assert space.steps_per_epoch(dataset, cap) >= 2


@pytest.mark.parametrize("dataset", sorted(EXPECTED_CAPS))
def test_batch_ladder_is_powers_of_two_from_64_to_the_cap(dataset):
    ladder = space.batch_sizes(dataset)
    assert ladder[0] == 64
    assert ladder[-1] == space.batch_cap(space.N_TRAIN[dataset])
    assert all(b * 2 == a for b, a in zip(ladder, ladder[1:]))


def test_a_batch_cap_never_exceeds_half_the_training_set():
    for n in (601, 3680, 50_000, 1_281_167):
        assert space.batch_cap(n) <= n // 2


def test_tiny_training_sets_raise_rather_than_returning_an_unusable_cap():
    """32 rows cannot support the 64 minimum. Returning 32 would quietly search
    a ladder the spec does not describe."""
    with pytest.raises(ValueError, match="too small"):
        space.batch_cap(32)


class _FakeTrial:
    """Records what `suggest` asks for without depending on optuna.

    `space.suggest` takes `trial: Any` specifically so this module -- and its
    tests -- never need optuna installed; neither container image has it yet.
    """

    def __init__(self):
        self.floats: dict[str, tuple[float, float, bool]] = {}
        self.categoricals: dict[str, list] = {}

    def suggest_float(self, name, low, high, *, log=False):
        self.floats[name] = (low, high, log)
        return low

    def suggest_categorical(self, name, choices):
        self.categoricals[name] = list(choices)
        return choices[0]


def test_learning_rate_lower_bound_is_1em3_not_1em4():
    """Pinned per the pilot: at lr=1e-4, every dataset on both arms had its
    val_nll minimum sitting at or within a hair of the final dumped step
    (still improving 1.5-2.6 nats over the last quarter of the run) --
    nowhere near converged at 4,000 steps. lr=3e-2 converged early everywhere
    (minima at steps 5-904, mildly overfitting past the optimum) and beat
    1e-4 on all 7 datasets on both arms. Sizing the campaign's budget so
    1e-4 converges would have multiplied its compute 5-12x for a rate the
    data shows is not competitive anywhere, so the floor is 1e-3 -- an
    evidence-backed decision, not an arbitrary default, and this test is here
    so it cannot silently drift back to 1e-4.
    """
    trial = _FakeTrial()
    space.suggest(trial, "adamw_ce", "dtd")
    assert trial.floats["learning_rate"] == (1e-3, 3e-2, True)


@pytest.mark.parametrize("arm", ["adamw_ce", "vbll"])
def test_suggest_works_with_a_duck_typed_trial(arm):
    """`suggest` must work with any object shaped like a trial -- not
    specifically optuna.Trial -- since optuna is not installed in either
    container image yet."""
    trial = _FakeTrial()
    params = space.suggest(trial, arm, "dtd")
    assert params["learning_rate"] == 1e-3


def test_vbll_searches_the_kl_ratio_not_a_bare_eta():
    """The quantity that controls fit is eta / prior_scale, so it is the axis.

    The first campaign fixed eta = 1 and searched prior_scale in [1e-2, 1e2],
    confining the ratio to [1e-2, 1e2] when the measured optimum is ~1e-5 --
    the box excluded the answer by three orders of magnitude, and all seven
    VBLL studies pinned prior_scale to its ceiling as a result. This pins the
    replacement so it cannot drift back to a range that cannot express 1e-5.
    """
    trial = _FakeTrial()
    space.suggest(trial, "vbll", "dtd")
    low, high, log = trial.floats["kl_ratio"]
    assert (low, high, log) == (1e-7, 1e-2, True)
    assert low < 1e-5 < high, "the measured optimum must be interior"
    assert "eta" not in trial.floats, "eta is derived from the ratio, not searched"


def test_vbll_prior_scale_is_widened_a_decade_at_each_end():
    trial = _FakeTrial()
    space.suggest(trial, "vbll", "dtd")
    assert trial.floats["prior_scale"] == (1e-1, 1e3, True)


def test_adamw_ce_gains_no_vbll_axes():
    trial = _FakeTrial()
    params = space.suggest(trial, "adamw_ce", "dtd")
    for axis in ("kl_ratio", "prior_scale", "wishart_scale"):
        assert axis not in params
        assert axis not in trial.floats


def test_ibprobit_has_no_optimizer_axes():
    """The decisive difference between this arm and the other two.

    IBProbit's head is assimilated in closed form by CAVI, and linear probing
    freezes the backbone -- so the arm runs no optimizer at all and there is
    nothing for a learning rate or a weight decay to act on. Sampling them
    anyway would present two tuned axes that do nothing, and would spend a
    third of the campaign's trials distinguishing configurations that are
    bit-identical. This is the test that fails if the adamw_ce block is ever
    copied into the ibprobit branch.
    """
    trial = _FakeTrial()
    params = space.suggest(trial, "ibprobit", "dtd")
    for axis in ("learning_rate", "weight_decay"):
        assert axis not in params
        assert axis not in trial.floats


def test_ibprobit_searches_q_init_directly_rather_than_alpha():
    """q0 = alpha^2 / B, and B is tuned alongside it.

    The campaign's artifact is a per-dataset table of tuned PROCESS NOISE read
    against the number of classes and the CAVI iteration count. A table of
    tuned `alpha` is not that table: two datasets whose tuned alpha agrees sit
    at different q0 whenever their tuned batch sizes differ, so the quantity
    the comparison is about would not be the quantity recorded. alpha stays
    derivable after the fact as sqrt(q0 * B).
    """
    trial = _FakeTrial()
    params = space.suggest(trial, "ibprobit", "dtd")
    assert "q_init" in params
    assert "alpha" not in params and "alpha" not in trial.floats
    assert trial.floats["q_init"] == (1e-6, 1e-1, True)


def test_the_q_init_range_brackets_the_pre_wishart_operating_point():
    """1/D = 7.8e-4 (D = 1280 for dinov3_huge) is where the head ran before the
    Inverse-Wishart recursion existed, so the box must be able to express it --
    and must not merely touch it at an endpoint, where TPE cannot refine.

    The floor sits above the MEASURED-inert point: at q0 = 3.1e-8 the whole
    recursion is provably dead (learn_q on and off give bit-identical mu over
    40 steps), so trials sampled there would compare configurations that do not
    differ. A decade of margin keeps the search out of that region without
    pretending the boundary is sharp.
    """
    trial = _FakeTrial()
    space.suggest(trial, "ibprobit", "dtd")
    low, high, log = trial.floats["q_init"]
    assert log, "process noise spans decades; a linear axis would sample only the top one"
    assert low < 7.8e-4 < high, "1/D must be interior, not an endpoint"
    assert low >= 10 * 3.1e-8, "the measured-inert regime must be below the floor"


def test_ibprobit_does_not_search_kappa():
    """Fixed at the class default of 0.5 by decision, not searched. It is the
    Inverse-Wishart prior pseudo-count against the K the likelihood contributes
    per step; the merge moved its default off `num_classes` precisely because a
    K-weight prior pins Q at its q0 seed."""
    trial = _FakeTrial()
    params = space.suggest(trial, "ibprobit", "dtd")
    assert "kappa" not in params
    assert "kappa" not in trial.floats and "kappa" not in trial.categoricals


def test_ibprobit_searches_the_four_requested_cavi_iteration_counts():
    """Categorical, not an integer range: 192 is 1.5x128, so the ladder is not
    geometric and suggest_int would sample values never asked for.

    Kept as a TPE axis rather than split across four separate studies because
    each dataset already has its own study: K is held fixed within a study, so
    a tuned iteration count that VARIES across datasets is itself the
    K-dependence signal.
    """
    trial = _FakeTrial()
    params = space.suggest(trial, "ibprobit", "dtd")
    assert trial.categoricals["num_update_iters"] == [32, 64, 128, 192]
    assert params["num_update_iters"] == 32


def test_ibprobit_searches_eps_init():
    trial = _FakeTrial()
    space.suggest(trial, "ibprobit", "dtd")
    assert trial.floats["eps_init"] == (1e-3, 1e-1, True)


@pytest.mark.parametrize("dataset", sorted(EXPECTED_CAPS))
def test_every_arm_shares_one_batch_ladder(dataset):
    """A per-arm ladder would make the cross-arm comparison partly a comparison
    of search spaces -- the reason batch_cap's floor is what it is."""
    ladders = {
        arm: _FakeTrial() for arm in ("adamw_ce", "vbll", "ibprobit")
    }
    for arm, trial in ladders.items():
        space.suggest(trial, arm, dataset)
    seen = {tuple(t.categoricals["batch_size"]) for t in ladders.values()}
    assert len(seen) == 1
    assert list(seen.pop()) == space.batch_sizes(dataset)


def test_an_unknown_arm_still_raises():
    with pytest.raises(ValueError, match="unknown arm"):
        space.suggest(_FakeTrial(), "not_an_arm", "dtd")


def test_every_tunable_arm_has_a_training_script():
    """space.ARMS and objective.SCRIPTS are two lists of the same thing, and
    worker's --arm choices come from the second while its search space comes
    from the first. An arm in one and not the other is a worker that accepts
    the flag at submit time and then dies inside its first trial -- on a
    cluster, several hours after the mistake was made."""
    from bllarse.tuning import objective

    assert set(space.ARMS) == set(objective.SCRIPTS)


def test_flowers102_trains_on_the_official_split():
    """1,020 images, not the 7,169-image val+test pool the upstream repo names
    "train". The whole batch ladder follows from this number, so it is pinned
    here as well as in bllarse.features."""
    assert space.N_TRAIN["flowers102"] == 1_020
    assert space.batch_sizes("flowers102") == [64, 128, 256]


@pytest.mark.parametrize("dataset", sorted(EXPECTED_CAPS))
def test_eval_quantum_is_a_whole_number_of_steps_at_every_rung(dataset):
    """The property that makes the quantum schedulable at all.

    Evaluation happens BETWEEN optimiser steps, so a quantum that is not an
    integer number of steps for some rung cannot be realised for that rung --
    and silently rounding it would give each trial a slightly different sample
    grid, which is precisely the defect the quantum exists to remove, but now
    invisible. It holds because the ladder is powers of two and the quantum is
    its cap.
    """
    quantum = space.eval_quantum(dataset)
    for batch_size in space.batch_sizes(dataset):
        assert quantum % batch_size == 0
        assert space.eval_every_steps(dataset, batch_size) == quantum // batch_size


@pytest.mark.parametrize("dataset", sorted(EXPECTED_CAPS))
def test_eval_quantum_is_sub_epoch(dataset):
    """The point of a step-based protocol is catching the minimum INSIDE an
    epoch. The quantum is at most half an epoch by construction -- it is
    batch_cap, which leaves MIN_STEPS_PER_EPOCH steps per pass -- so this
    cannot regress without MIN_STEPS_PER_EPOCH changing.

    A quantum fixed globally at MAX_BATCH would fail this on four of the seven:
    16,384 samples is 16.1 epochs of flowers102's 1,020 training images.
    """
    quantum = space.eval_quantum(dataset)
    epochs_per_eval = quantum / space.N_TRAIN[dataset]
    assert epochs_per_eval <= 1.0 / space.MIN_STEPS_PER_EPOCH


def test_no_ladder_exceeds_the_global_batch_ceiling():
    """MAX_BATCH caps every dataset, not just the ones small enough to hit
    MIN_STEPS_PER_EPOCH first. Without it food101 reaches 32768, which is both
    the most expensive rung and -- since the cap IS the eval quantum -- twice
    as coarse a val_nll curve as every other dataset gets."""
    for dataset in EXPECTED_CAPS:
        assert max(space.batch_sizes(dataset)) <= space.MAX_BATCH


def test_eval_every_steps_rejects_an_off_ladder_batch():
    """Raises rather than rounding. A rounded interval would hand back a grid
    whose samples_seen differ per trial, and the pruner would go on comparing
    trials at indices that no longer mean the same thing."""
    with pytest.raises(ValueError, match="whole number"):
        space.eval_every_steps("cifar10", 100)


def test_the_noeps_arm_drops_eps_from_the_search_rather_than_overriding_it():
    """`ibprobit_noeps` pins eps_init; it must not also SAMPLE it.

    A sampled axis that never reaches the training script would have TPE model
    a dimension the runs do not have, and the study's own record of what it
    searched would be a fiction. Everything else must match `ibprobit` exactly,
    or the ablation measures the search space instead of the contamination
    mixture.
    """
    free, pinned = _FakeTrial(), _FakeTrial()
    a = space.suggest(free, "ibprobit", "dtd")
    b = space.suggest(pinned, "ibprobit_noeps", "dtd")

    assert "eps_init" in a and "eps_init" in free.floats
    assert "eps_init" not in b and "eps_init" not in pinned.floats
    assert set(a) - {"eps_init"} == set(b), (
        "the two arms must differ in the eps axis and nothing else"
    )
    assert free.floats["q_init"] == pinned.floats["q_init"]
    assert free.categoricals == pinned.categoricals


def test_the_pinned_eps_is_not_zero_and_clears_the_heads_own_guard():
    """0.0 does NOT disable the mixture. `IBProbit._eps_seed` reads

        return 1.0, (1.0 / self.eps_init if self.eps_init > 0.0 else 100.0)

    so zero falls through to b0 = 100 -- a Beta(1, 100) with mean 0.0099, the
    MIDDLE of the searched box, measured to drift to 0.16 within 32 updates.
    An ablation set to 0.0 would silently re-run the baseline.
    """
    assert space.IBPROBIT_NOEPS_INIT > 0.0
    assert space.IBPROBIT_NOEPS_INIT < 1e-9, (
        "must be far below the searched box's 1e-3 floor to make the mixture "
        "floor numerically absent"
    )


def test_both_ibprobit_arms_are_registered():
    assert "ibprobit" in space.ARMS and "ibprobit_noeps" in space.ARMS
