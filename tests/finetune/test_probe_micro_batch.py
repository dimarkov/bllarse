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


def test_probe_ceiling_builds_a_fresh_optimizer_state_per_attempt():
    """apply_grads is @eqx.filter_jit(donate="all-except-first").

    It DONATES opt_state, so a state reused across attempts is a freed buffer
    by the second one -- which surfaces as `Array has been deleted with
    shape=int32[]` (the AdamW step counter), not as an OOM, and so gets
    mistaken for the ceiling. This asserts the factory is called once per
    attempt rather than hoisted.
    """
    import inspect

    source = inspect.getsource(pmb.probe_ceiling)
    assert "make_optimizer()" in source
    # Inside the attempt body, not before the loop.
    body = source.split("while bs <= max_bs:", 1)[1]
    assert "make_optimizer()" in body


def test_the_probe_evaluates_through_the_training_script_jitted_path():
    """The eval leg must be the SAME compiled function the run uses.

    Calling jax_steps.forward_features directly is not equivalent: outside jit,
    bf16=True materialises a full bf16 copy of the backbone (~1.7 GB on
    dinov3_huge) alongside the fp32 master weights instead of fusing the cast,
    and all 32 layers' activations stay live with no early free. That inflates
    the eval leg enough to depress the reported ceiling below what the run
    needs -- the probe would measure its own un-jitted eval, not the step.
    """
    import inspect

    source = inspect.getsource(pmb.probe_ceiling)
    # Code only -- the comment above the call explains WHY the un-jitted path
    # is wrong and would otherwise match the negative assertion below.
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    assert "finetune_jax.eval_logits_ce" in code
    assert "finetune_jax.eval_logits_ibprobit" in code
    assert "jax_steps.forward_features" not in code


def test_both_jitted_eval_entry_points_exist_on_the_training_script():
    # Pins the import contract: a rename in finetune_jax must fail here, on
    # CPU, rather than on a compute node after a queue wait.
    import finetune_jax

    for name in ("eval_logits_ce", "eval_logits_ibprobit"):
        assert callable(getattr(finetune_jax, name)), name
