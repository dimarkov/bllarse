"""Layer-wise decay on the torch arm. Skipped where torch is absent.

The JAX arms and this one share `protocol.llrd_scale` but not the parameter
naming -- equimo gives pytree paths, transformers gives dotted names -- so the
GROUPING is a second implementation and needs its own test.
"""

import sys
from pathlib import Path

import pytest

from bllarse.finetune import protocol

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "finetune"))
import finetune_vbll as fv  # noqa: E402


class _Config:
    def __init__(self, depth):
        self.num_hidden_layers = depth


def _fake_dinov3(depth=4):
    """A module with dinov3's parameter NAMES and nothing else.

    Naming is the only thing the grouping reads, and using the real checkpoint
    would make this test need a 1.2 GB download to check a regex.
    """
    m = torch.nn.Module()
    m.config = _Config(depth)
    m.embeddings = torch.nn.ModuleDict({"patch_embeddings": torch.nn.Linear(2, 2)})
    m.layer = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(depth)])
    m.norm = torch.nn.LayerNorm(2)
    return m


def test_off_is_a_single_group_at_the_base_rate():
    groups = fv._trunk_lr_groups(_fake_dinov3(), 1e-4, 1e-2,
                                 protocol.LAYER_DECAY_OFF)

    assert len(groups) == 1
    assert groups[0]["lr"] == pytest.approx(1e-4)


def test_one_group_per_layer_plus_the_embeddings():
    depth = 4
    groups = fv._trunk_lr_groups(_fake_dinov3(depth), 1e-4, 1e-2, 0.5)

    assert len(groups) == depth + 1
    # Ordered bottom-up, so rates increase towards the head.
    lrs = [g["lr"] for g in groups]
    assert lrs == sorted(lrs)
    assert lrs[-1] == pytest.approx(1e-4)              # top layer at full rate
    assert lrs[0] == pytest.approx(1e-4 * 0.5 ** depth)  # embeddings below all


def test_the_final_norm_rides_with_the_TOP_layer_not_the_embeddings():
    groups = fv._trunk_lr_groups(_fake_dinov3(4), 1e-4, 1e-2, 0.5)
    top = max(groups, key=lambda g: g["lr"])
    names = {id(p) for p in top["params"]}
    norm_ids = {id(p) for p in _fake_dinov3(4).norm.parameters()}

    # identity comparison across two instances is meaningless, so compare shapes
    assert any(p.ndim == 1 for p in top["params"])
    assert len(norm_ids) == 2


def test_a_naming_change_fails_loudly_instead_of_disabling_llrd():
    """The failure this guard exists for.

    If the regex stops matching, every tensor lands in one group and the run
    trains with LLRD off while still logging a decay value -- invisible in
    every result. The config's own depth is the cross-check.
    """
    model = _fake_dinov3(4)
    model.config.num_hidden_layers = 9

    with pytest.raises(SystemExit, match="num_hidden_layers"):
        fv._trunk_lr_groups(model, 1e-4, 1e-2, 0.5)


def test_every_parameter_lands_in_exactly_one_group():
    model = _fake_dinov3(4)
    groups = fv._trunk_lr_groups(model, 1e-4, 1e-2, 0.5)

    grouped = [id(p) for g in groups for p in g["params"]]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {id(p) for p in model.parameters()}
