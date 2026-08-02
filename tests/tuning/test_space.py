import pytest

from bllarse.tuning import space


# The spec's table, verbatim. Any drift here is a silent change to what the
# campaign searched, so it is pinned rather than recomputed.
EXPECTED_CAPS = {
    "oxford_pets": 1024,
    "dtd": 1024,
    "flowers102": 2048,
    "stanford_cars": 2048,
    "food101": 32768,
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
