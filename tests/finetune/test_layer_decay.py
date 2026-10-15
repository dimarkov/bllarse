"""Layer-wise LR decay. Default off, and the block index must be the LAYER."""

import re
import sys
from pathlib import Path

import pytest

from bllarse.finetune import protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "finetune"))


def test_off_by_default_is_the_exact_identity():
    for block in (None, 0, 5, 11):
        assert protocol.llrd_scale(block, 12, protocol.LAYER_DECAY_OFF) == 1.0


def test_the_top_block_is_one_and_the_embeddings_are_damped_hardest():
    d, decay = 12, 0.75
    assert protocol.llrd_scale(d - 1, d, decay) == pytest.approx(1.0)
    assert protocol.llrd_scale(0, d, decay) == pytest.approx(decay ** (d - 1))
    # Non-block sits BELOW block 0, so one more factor of decay.
    assert protocol.llrd_scale(None, d, decay) == pytest.approx(decay ** d)
    assert protocol.llrd_scale(None, d, decay) < protocol.llrd_scale(0, d, decay)


def test_scales_increase_monotonically_towards_the_head():
    scales = [protocol.llrd_scale(i, 12, 0.75) for i in range(12)]
    assert scales == sorted(scales)


@pytest.mark.parametrize("block,depth,decay", [(12, 12, 0.75), (-1, 12, 0.75),
                                               (0, 0, 0.75), (0, 12, 0.0),
                                               (0, 12, 1.5)])
def test_out_of_range_arguments_are_refused(block, depth, decay):
    with pytest.raises(ValueError):
        protocol.llrd_scale(block, depth, decay)


def test_the_layer_index_is_the_SECOND_blocks_in_an_equimo_path():
    """The regression this is most likely to hit.

    equimo nests stages: `.blocks[0].blocks[7]...`. Matching the FIRST index
    labels every parameter block 0, which silently disables LLRD while the run
    still trains and still logs a decay value.
    """
    key = ".blocks[0].blocks[7].attn.qkv.weight"
    found = re.findall(r"blocks\[(\d+)\]", key)

    assert len(found) == 2
    assert int(found[1]) == 7
    assert int(found[0]) == 0        # the stage, not the layer


def test_trunk_groups_cover_every_array_leaf_of_a_real_backbone():
    equimo = pytest.importorskip("equimo")
    import equinox as eqx
    import jax
    from bllarse.finetune import equimo_vision_models
    import finetune_jax

    bb = equimo_vision_models().dinov3_vits16plus_pretrain_lvd1689m(pretrained=False)
    depth = len(bb.blocks[0].blocks)
    scales = finetune_jax._trunk_lr_scales(depth, 0.75)
    labels = finetune_jax._trunk_labels_of(bb, depth)

    names = [v for v in jax.tree_util.tree_leaves(
        labels, is_leaf=lambda x: isinstance(x, str)) if isinstance(v, str)]

    # one group per block plus the embeddings, and every leaf labelled into one
    assert len(scales) == depth + 1
    assert set(names) == set(scales)
    assert scales[f"trunk_b{depth - 1:03d}"] == pytest.approx(1.0)
    assert scales["trunk_embed"] == pytest.approx(0.75 ** depth)


def test_multi_transform_actually_applies_the_per_block_rates():
    """The check the arithmetic tests cannot make.

    Every earlier test here asserts what `llrd_scale` RETURNS. This asserts
    that optax applies it: with decay 0.75 over 12 blocks the top block must
    move 0.75**-12 = 31.6x further than the embeddings under identical
    gradients. It also pins the labels-as-CALLABLE form -- handing
    multi_transform a labels PYTREE built from an equinox Module makes it treat
    the Module as a leaf and reach for `.shape`, a failure only a real
    backbone shows.
    """
    pytest.importorskip("equimo")
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax
    from bllarse.finetune import equimo_vision_models
    import finetune_jax

    bb = equimo_vision_models().dinov3_vits16plus_pretrain_lvd1689m(pretrained=False)
    depth = len(bb.blocks[0].blocks)
    scales = finetune_jax._trunk_lr_scales(depth, 0.75)
    assert len(scales) == depth + 1

    tx = {name: optax.adamw(1e-4 * s) for name, s in scales.items()}
    opt = optax.multi_transform(
        tx, lambda p: finetune_jax._trunk_labels_of(p, depth))

    params = eqx.filter(bb, eqx.is_array)
    state = opt.init(params)
    grads = jax.tree.map(lambda a: jnp.ones_like(a) * 1e-3, params)
    updates, _ = opt.update(grads, state, params)

    flat = {jax.tree_util.keystr(p): v
            for p, v in jax.tree_util.tree_flatten_with_path(updates)[0]}
    mean = lambda pred: float(jnp.mean(jnp.stack([  # noqa: E731
        jnp.mean(jnp.abs(v)) for k, v in flat.items() if pred(k)])))
    embeddings = mean(lambda k: k.startswith(".patch_embed"))
    top = mean(lambda k: f".blocks[{depth - 1}]" in k)

    assert top / embeddings == pytest.approx(0.75 ** -(depth), rel=0.05)


def test_the_ce_arm_tuple_shape_also_builds_and_steps():
    """CE labels a (backbone, head) TUPLE, a different pytree shape from
    IBProbit's bare trunk. Both go through multi_transform, so both need
    exercising -- the tuple form is where a path-prefix assumption would break."""
    pytest.importorskip("equimo")
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax
    from bllarse.finetune import equimo_vision_models
    import finetune_jax

    bb = equimo_vision_models().dinov3_vits16plus_pretrain_lvd1689m(pretrained=False)
    depth = len(bb.blocks[0].blocks)
    head = eqx.nn.Linear(bb.dim, 10, key=jax.random.PRNGKey(0))
    tx = {name: optax.adamw(1e-4 * s)
          for name, s in finetune_jax._trunk_lr_scales(depth, 0.75).items()}
    tx["head"] = optax.adamw(1e-2)

    opt = optax.multi_transform(tx, lambda p: (
        finetune_jax._trunk_labels_of(p[0], depth),
        jax.tree.map(lambda _: "head", eqx.filter(p[1], eqx.is_array))))
    params = (eqx.filter(bb, eqx.is_array), eqx.filter(head, eqx.is_array))
    state = opt.init(params)
    grads = jax.tree.map(lambda a: jnp.ones_like(a) * 1e-3, params)

    updates, _ = opt.update(grads, state, params)
    assert updates is not None


# --- The span, which is what transfers across backbones ---------------------


def test_the_span_is_invariant_across_the_dinov3_ladder():
    """One number, three depths, the same trunk shape.

    This is the defect the span exists to prevent: `--layer-decay 0.75` carried
    from dinov3_small to dinov3_huge puts the embeddings at 0.75^32 = 1.0e-4
    instead of 0.75^12 = 3.2e-2 -- 316x more damping, frozen in all but name,
    and nothing in the code would object.
    """
    span = protocol.LAYER_DECAY_SPAN
    for depth in (12, 24, 32):
        decay = protocol.layer_decay_for_span(span, depth)
        assert decay ** depth == pytest.approx(span, rel=1e-9)


def test_it_reproduces_the_validated_small_campaign_exactly():
    # LAYER_DECAY_SPAN was validated as 0.75 per block over dinov3_small's 12
    # blocks, so the span form must return that value unchanged or those runs
    # become unreproducible.
    assert protocol.layer_decay_for_span(
        protocol.LAYER_DECAY_SPAN, 12) == pytest.approx(0.75, rel=1e-12)


def test_the_deeper_backbones_need_a_weaker_per_block_factor():
    # Stated as the direction, not just the arithmetic: more blocks compound
    # the same span, so each block must decay LESS.
    d12 = protocol.layer_decay_for_span(protocol.LAYER_DECAY_SPAN, 12)
    d24 = protocol.layer_decay_for_span(protocol.LAYER_DECAY_SPAN, 24)
    d32 = protocol.layer_decay_for_span(protocol.LAYER_DECAY_SPAN, 32)
    assert d12 < d24 < d32 < 1.0
    assert d24 == pytest.approx(0.8660, abs=5e-5)
    assert d32 == pytest.approx(0.8977, abs=5e-5)


def test_a_span_of_one_is_no_decay_at_any_depth():
    for depth in (12, 24, 32):
        assert protocol.layer_decay_for_span(
            protocol.LAYER_DECAY_SPAN_OFF, depth) == protocol.LAYER_DECAY_OFF


@pytest.mark.parametrize("span", [0.0, -0.1, 1.5])
def test_a_span_outside_the_unit_interval_is_refused(span):
    # It is a FRACTION of the top block's rate. A value above 1 would mean the
    # embeddings train faster than the head, which is the opposite of LLRD.
    with pytest.raises(ValueError, match="span"):
        protocol.layer_decay_for_span(span, 12)


def test_the_span_matches_what_llrd_scale_gives_the_embeddings():
    """The span is measured at the NON-BLOCK params, not at block 0.

    `llrd_scale(None, ...)` is `decay^depth` -- one step below block 0 -- and
    that is the parameter the damping is really about. Keying the span on
    block 0 instead would let the embeddings drift by a factor of `decay`
    across the ladder, which is the drift the span exists to prevent.
    """
    span = protocol.LAYER_DECAY_SPAN
    for depth in (12, 24, 32):
        decay = protocol.layer_decay_for_span(span, depth)
        assert protocol.llrd_scale(None, depth, decay) == pytest.approx(span)
        assert protocol.llrd_scale(depth - 1, depth, decay) == pytest.approx(1.0)
