"""The lifted kernels are the SAME objects the comparison script uses.

full_finetune_comparison.py and scripts/finetune/finetune_jax.py both drive the
gradient-accumulation path. These kernels carry buffer-donation contracts tuned
against a measured ~6 GiB peak (commit cb81798); a second, drifting copy would
regress that silently and only on the largest backbone.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "alternatives"))

import full_finetune_comparison as ffc  # noqa: E402

from bllarse.finetune import jax_steps  # noqa: E402


def test_comparison_script_uses_the_shared_kernels_not_its_own_copies():
    for name in ("_tree_add", "apply_grads", "ce_micro_grad",
                 "ibprobit_micro_forward", "forward_features"):
        assert getattr(ffc, name) is getattr(jax_steps, name), (
            f"{name} has diverged: the comparison script holds its own copy"
        )


def test_tree_add_sums_leafwise_and_tolerates_none_leaves():
    a = {"w": jnp.ones((2, 2)), "b": None}
    b = {"w": jnp.full((2, 2), 3.0), "b": None}
    out = jax_steps._tree_add(a, b)
    assert jnp.allclose(out["w"], jnp.full((2, 2), 4.0))
    assert out["b"] is None


def test_tree_add_returns_the_present_leaf_when_one_side_is_none():
    out = jax_steps._tree_add({"w": jnp.ones(3)}, {"w": None})
    assert jnp.allclose(out["w"], jnp.ones(3))
