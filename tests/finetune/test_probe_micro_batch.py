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


# --- build_optimizer: exercised on CPU so a missing import cannot reach the
# cluster. The first corrected probe run died on `NameError: name 'optax' is
# not defined` after queueing behind a busy GPU partition -- a 45-minute
# round trip for a one-line mistake that this catches in milliseconds.


def _tiny_pair():
    import equinox as eqx
    import jax

    key = jax.random.PRNGKey(0)
    backbone = eqx.nn.Linear(4, 3, key=key)
    head = eqx.nn.Linear(3, 2, key=key)
    return backbone, head


@pytest.mark.parametrize("arm", ["ce", "ibprobit"])
def test_build_optimizer_returns_an_initialised_state(arm):
    import equinox as eqx
    import jax

    backbone, head = _tiny_pair()
    optimizer, state = pmb.build_optimizer(
        arm, backbone, head, lr=1e-4, head_lr=1e-2, weight_decay=1e-2)
    assert optimizer is not None
    # The state must actually hold arrays -- that allocation IS the thing the
    # probe exists to account for (AdamW's two moment buffers are ~2x params,
    # ~6.7 GB on dinov3_huge, and omitting them is what made the first probe
    # report a micro-batch the real run OOMs on).
    leaves = [x for x in jax.tree.leaves(state) if eqx.is_array(x)]
    assert leaves, f"{arm}: optimizer state allocated no arrays"


def test_the_ce_optimizer_gives_the_head_zero_weight_decay():
    # Design D6: head weight decay is 0 on every arm, so the probe's memory
    # profile matches finetune_jax's rather than merely resembling it.
    import inspect

    source = inspect.getsource(pmb.build_optimizer)
    assert "weight_decay=0.0" in source
    assert "multi_transform" in source
