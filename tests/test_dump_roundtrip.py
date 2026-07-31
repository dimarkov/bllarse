import numpy as np
import pytest

from bllarse import evaldump, metrics


def test_metrics_survive_the_fp16_dump_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    logits = rng.normal(scale=3.0, size=(2000, 50)).astype(np.float32)
    labels = rng.integers(0, 50, size=2000)

    in_loop = metrics.classification_metrics(logits, labels)

    path = evaldump.write_dump(
        tmp_path, "g", "u", step=1, logits=logits, labels=labels,
        arm="adamw_ce", split="test", logit_kind="softmax_logits",
        seed=137, samples_seen=1000,
    )
    record = evaldump.load_dump(path)
    post_hoc = metrics.classification_metrics(record.logits, record.labels)

    assert post_hoc["acc"] == pytest.approx(in_loop["acc"], abs=2e-3)
    assert post_hoc["nll"] == pytest.approx(in_loop["nll"], rel=2e-3)
    assert post_hoc["brier"] == pytest.approx(in_loop["brier"], rel=2e-3)
    assert post_hoc["ece"] == pytest.approx(in_loop["ece"], abs=5e-3)


def test_temperature_fit_survives_the_round_trip(tmp_path):
    # Labels are SAMPLED from softmax(base), so base is calibrated at T=1 and
    # base*4 has an NLL-optimal temperature of 4. Using labels = logits.argmax(-1)
    # instead would make the model correct everywhere, pinning fit_temperature to
    # its clip bound (T = 0.01) on BOTH sides -- the assertion would then compare
    # 0.01 to 0.01 and pass even if fp16 had destroyed the logits entirely.
    rng = np.random.default_rng(1)
    base = rng.normal(size=(3000, 10)).astype(np.float32)
    probs = np.exp(base) / np.exp(base).sum(axis=-1, keepdims=True)
    labels = np.array([rng.choice(10, p=row) for row in probs])
    logits = base * 4.0

    in_loop_T = metrics.fit_temperature(logits, labels)
    assert 3.0 < in_loop_T < 5.0, f"fixture should give T near 4, got {in_loop_T}"
    path = evaldump.write_dump(
        tmp_path, "g", "u2", step=1, logits=logits, labels=labels,
        arm="adamw_ce", split="val", logit_kind="softmax_logits",
        seed=137, samples_seen=1000,
    )
    rec = evaldump.load_dump(path)
    assert metrics.fit_temperature(rec.logits, rec.labels) == pytest.approx(in_loop_T, rel=1e-2)
