import numpy as np
import pytest

from bllarse import evaldump


# Size 67 is deliberate: it is one of the ~2% of sizes where the gap sequence
# dips by exactly one at the boundary between the forced-increment region and
# the geometric region. Asserting strictly non-decreasing gaps would fail there.
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


def test_write_dump_keeps_multiple_splits_at_the_same_step_separate(tmp_path):
    """A single run dumps val and test (and, at the final step, ood) from the
    same forward pass at the same step number. step alone must not be the
    on-disk key, or the second write silently clobbers the first."""
    for split in ("val", "test", "ood"):
        evaldump.write_dump(
            tmp_path, "grp", "uid5", step=9,
            logits=np.full((4, 2), fill_value=hash(split) % 7, dtype=np.float32),
            labels=np.zeros(4, dtype=int),
            arm="adamw_ce", split=split, logit_kind="softmax_logits",
            seed=1, samples_seen=9 * 8,
        )
    records = list(evaldump.iter_dumps(tmp_path, "grp", "uid5"))
    splits = {rec.meta["split"] for rec in records}
    assert splits == {"val", "test", "ood"}, (
        f"expected all three splits to survive at the same step, got {splits}"
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
