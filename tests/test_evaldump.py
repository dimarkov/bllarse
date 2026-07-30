import numpy as np
import pytest

from bllarse import evaldump


# Size 67 is deliberate: it is one of the ~2% of sizes where the gap sequence
# dips by exactly one at the boundary between the forced-increment region and
# the geometric region. Asserting strictly non-decreasing gaps would fail there.
@pytest.mark.parametrize(
    "num_steps,n_dumps",
    [(41, 40), (67, 40), (100, 40), (200, 40), (500, 40), (2000, 40), (60, 8), (3000, 16)],
)
def test_dump_schedule_invariants(num_steps, n_dumps):
    """The four properties that actually hold.

    Verified by brute force over n_dumps 2..60 x num_steps up to n_dumps+2000
    (118k combinations): zero violations of any of these. A naive
    round-then-unique returns 25-36 points instead of 40 over this range,
    thinning exactly the early region the schedule exists to resolve.
    """
    sched = evaldump.dump_schedule(num_steps, n_dumps=n_dumps)
    assert len(sched) == n_dumps, "must return exactly the number of points requested"
    assert sched[0] == 1
    assert sched[-1] == num_steps

    gaps = np.diff(sched)
    assert np.all(gaps > 0), f"must be strictly increasing, got {sched}"
    # Gaps trend upward, but dip by at most one where the forced-increment
    # region meets the geometric one. Do NOT tighten this to >= 0: that is false
    # at ~2% of sizes. Do NOT assert an absolute fraction of points in the first
    # tenth either -- arithmetically impossible at moderate num_steps, since at
    # 100 steps only 10 of 40 points can land at or below step 10.
    assert np.all(np.diff(gaps) >= -1), f"gaps dipped by more than one: {gaps}"


def test_dump_schedule_concentrates_early_at_the_operating_point():
    """Early concentration, asserted at the configuration the sweeps use.

    It is not assertable in general: when num_steps approaches n_dumps nearly
    every step is dumped and there is no concentration to be had.
    """
    sched = evaldump.dump_schedule(500, n_dumps=40)
    assert sched[20] < 500 / 4, "half the points should land in the first quarter"


def test_dump_schedule_rejects_a_degenerate_n_dumps():
    with pytest.raises(ValueError):
        evaldump.dump_schedule(500, n_dumps=1)


def test_dump_schedule_handles_fewer_steps_than_dumps():
    sched = evaldump.dump_schedule(5, n_dumps=40)
    assert list(sched) == [1, 2, 3, 4, 5]


def test_write_and_load_round_trip(tmp_path):
    logits = np.random.default_rng(0).normal(size=(64, 10)).astype(np.float32)
    labels = np.arange(64) % 10
    path = evaldump.write_dump(
        tmp_path, "grp", "uid1", step=7, logits=logits, labels=labels,
        arm="adamw_ce", split="test", logit_kind="softmax_logits",
        seed=137, samples_seen=448,
    )
    rec = evaldump.load_dump(path)
    assert rec.logits.dtype == np.float16, "dumps are fp16 on disk"
    np.testing.assert_allclose(rec.logits, logits, atol=1e-2)
    np.testing.assert_array_equal(rec.labels, labels)
    assert rec.meta["arm"] == "adamw_ce"
    assert rec.meta["step"] == 7
    assert rec.meta["samples_seen"] == 448
    assert rec.score is None


def test_write_carries_optional_score_and_eps(tmp_path):
    logits = np.zeros((8, 3), dtype=np.float32)
    path = evaldump.write_dump(
        tmp_path, "grp", "uid2", step=1, logits=logits, labels=np.zeros(8, dtype=int),
        arm="ibprobit", split="test", logit_kind="moderated_score",
        seed=137, samples_seen=64,
        score=np.arange(8, dtype=np.float32), score_kind="predictive_variance",
        eps_hat=0.03,
    )
    rec = evaldump.load_dump(path)
    np.testing.assert_allclose(rec.score, np.arange(8))
    assert rec.meta["score_kind"] == "predictive_variance"
    assert rec.meta["eps_hat"] == pytest.approx(0.03)


def test_labels_and_logits_must_agree_in_length(tmp_path):
    with pytest.raises(ValueError, match="length"):
        evaldump.write_dump(
            tmp_path, "grp", "uid3", step=1,
            logits=np.zeros((8, 3)), labels=np.zeros(7, dtype=int),
            arm="adamw_ce", split="test", logit_kind="softmax_logits",
            seed=1, samples_seen=1,
        )


def test_iter_dumps_returns_steps_in_order(tmp_path):
    for step in (13, 1, 5):
        evaldump.write_dump(
            tmp_path, "grp", "uid4", step=step,
            logits=np.zeros((4, 2)), labels=np.zeros(4, dtype=int),
            arm="adamw_ce", split="test", logit_kind="softmax_logits",
            seed=1, samples_seen=step * 8,
        )
    steps = [rec.meta["step"] for rec in evaldump.iter_dumps(tmp_path, "grp", "uid4")]
    assert steps == [1, 5, 13]
