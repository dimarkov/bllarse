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
