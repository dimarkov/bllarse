"""The val_nll curve written alongside the argmin dump.

The probing scripts keep one logit dump per run (the argmin), which alone
leaves questions about the stopping rule unanswerable without a re-run. The
curve must not change what gets selected -- only record what selection had to
choose from.
"""
import numpy as np
import pytest

from bllarse import evaldump


def test_roundtrip_preserves_the_curve(tmp_path):
    evaldump.write_trajectory(
        tmp_path, "g", "u", eval_index=[1, 2, 3], step=[16, 32, 48],
        samples_seen=[1024, 2048, 3072], val_nll=[0.5, 0.4, 0.45],
        arm="adamw_ce", seed=137, evals_planned=1875,
    )
    got = evaldump.load_trajectory(tmp_path, "g", "u")
    assert got["val_nll"] == pytest.approx([0.5, 0.4, 0.45])
    assert got["step"].tolist() == [16, 32, 48]
    assert got["meta"] == {"arm": "adamw_ce", "seed": 137,
                           "evals_planned": 1875, "n_evals": 3}


def test_a_missing_trajectory_raises_rather_than_returning_empty(tmp_path):
    """A run that wrote no trajectory has no curve. Handing back an empty
    one would let a caller average over runs that contributed nothing."""
    with pytest.raises(FileNotFoundError):
        evaldump.load_trajectory(tmp_path, "g", "nope")


def test_it_is_cheap_next_to_the_logit_dumps(tmp_path):
    """Scalars only. The whole point is that it costs nothing where the logits
    cost everything -- measured at 2.7 KB for 180 evaluations against 1.3 MB
    for that run's two logit dumps."""
    n = 30_000
    path = evaldump.write_trajectory(
        tmp_path, "g", "u", eval_index=list(range(n)), step=list(range(n)),
        samples_seen=list(range(n)), val_nll=np.linspace(2.0, 0.2, n).tolist(),
        arm="adamw_ce", seed=137, evals_planned=n,
    )
    assert path.stat().st_size < 2_000_000, "a full-length curve must stay ~1 MB"


@pytest.mark.parametrize("script", [
    "scripts/dinov3/probe.py", "scripts/vbll_pytorch/probing_vbll.py",
])
def test_both_training_scripts_write_the_curve_from_the_finally(script):
    """From the same `finally` as the argmin dump, and BEFORE it: `_write_best`
    raises when no evaluation ran, and the curve is exactly the evidence needed
    to see why. Ordering it after would lose the curve on the one failure it
    would most help diagnose."""
    from pathlib import Path

    text = Path(script).read_text()
    assert "evaldump.write_trajectory(" in text

    # Comment lines stripped: the prose explaining the ordering names
    # `_write_best` before the call it describes, and matching that instead of
    # the code would pass or fail on how the comment is worded.
    live = "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("#")
    )
    tail = live[live.rindex("finally:"):]
    assert tail.index("write_trajectory") < tail.index("_write_best("), (
        "the curve must be written before _write_best can raise"
    )
