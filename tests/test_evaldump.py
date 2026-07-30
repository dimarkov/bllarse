import numpy as np
import pytest

from bllarse import evaldump


def test_dump_schedule_is_log_spaced_and_bounded():
    sched = evaldump.dump_schedule(500, n_dumps=40)
    assert sched[0] == 1
    assert sched[-1] == 500
    assert len(sched) <= 40
    assert list(sched) == sorted(set(sched)), "must be strictly increasing, no repeats"
    # Dense early: more than half the points fall in the first tenth of training.
    assert (sched <= 50).sum() > len(sched) // 2


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
