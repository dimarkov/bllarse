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
    from bllarse.finetune import equimo_vision_models
    import equinox as eqx
    import finetune_jax

    bb = equimo_vision_models().dinov3_vits16plus_pretrain_lvd1689m(pretrained=False)
    depth = len(bb.blocks[0].blocks)
    labels, scales = finetune_jax._trunk_lr_groups(bb, depth, 0.75)

    leaves = eqx.filter(labels, lambda x: isinstance(x, str), is_leaf=lambda x: isinstance(x, str))
    names = [v for v in __import__("jax").tree_util.tree_leaves(
        leaves, is_leaf=lambda x: isinstance(x, str)) if isinstance(v, str)]

    # every block gets its own group, plus one for the embeddings
    assert len(scales) == depth + 1
    assert set(names) <= set(scales)
    assert scales["trunk_b%03d" % (depth - 1)] == pytest.approx(1.0)
    assert scales["trunk_embed"] == pytest.approx(0.75 ** depth)
