"""The pinned micro-batch rule.

`probe_ceiling` needs a GPU and a real DINOv3 trunk, so it is exercised on
Raven rather than here. What IS testable -- and what actually decides whether
a campaign's effective batch matches its tuned batch -- is the rule that turns
a measured ceiling into the pinned constant.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "finetune"))

import probe_micro_batch as pmb  # noqa: E402

LADDER = (256, 512, 1024, 2048, 4096)


@pytest.mark.parametrize("probed,expected", [
    (64, 64),      # already a power of two, under the cap
    (100, 64),     # floors down, never up
    (128, 128),
    (200, 128),
    (256, 256),
    (300, 256),    # capped at the ladder gcd
    (1000, 256),   # a roomy GPU still runs the bottom rung un-accumulated
    (1, 1),
])
def test_floor_to_ladder_divisor(probed, expected):
    assert pmb.floor_to_ladder_divisor(probed) == expected


@pytest.mark.parametrize("probed", [1, 7, 64, 100, 255, 256, 999, 8192])
def test_the_result_divides_every_rung_of_the_ladder(probed):
    """The property the whole rule exists for.

    protocol.accum_plan raises unless the micro batch divides the target
    exactly, because rounding produced an effective batch that was not the
    tuned one and mislabelled every point of a samples-seen axis. A pinned
    value that fails here would make the bottom rung unrunnable.
    """
    micro = pmb.floor_to_ladder_divisor(probed)
    for rung in LADDER:
        assert rung % micro == 0, (rung, micro)


def test_floor_to_ladder_divisor_never_exceeds_the_measurement():
    # Rounding UP would pin a value that OOMs on the hardware it was measured
    # on -- the one failure mode that costs a whole campaign rather than a run.
    for probed in range(1, 600):
        assert pmb.floor_to_ladder_divisor(probed) <= probed


def test_floor_to_ladder_divisor_rejects_a_nonpositive_measurement():
    with pytest.raises(ValueError, match="probed"):
        pmb.floor_to_ladder_divisor(0)
