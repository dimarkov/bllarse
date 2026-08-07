"""Guards the API contract between IBProbit and its consuming scripts.

726bc78 landed ood_evaluation.py, full_finetune_comparison.py and
vit_classification.py on this branch without the IBProbit they call, and the
suite did not notice. These assertions are cheap and would have.

ood_evaluation.py has since been removed -- its OOD helpers were ported into
full_finetune_comparison.py (see tests/test_full_finetune_ood.py) and its
detection metrics are now produced from evaldump dumps by
bllarse.analysis.curves -- so the contract now covers the two survivors.
"""
import jax.numpy as jnp
import jax.random as jr
import pytest

from bllarse.losses.bayesian import IBProbit


def _kwargs():
    """The union of constructor kwargs the consuming scripts pass.

    Every kwarg here is passed at construction by at least one of them.
    `alpha` must track the scripts' argparse default (currently 0.1) -- not the
    IBProbit constructor default -- because the scripts pass alpha=args.alpha
    explicitly and so bypass the constructor's own default entirely.
    """
    return dict(q_init=None, alpha=0.1, batch_size=32, kappa=None,
                learn_q=True, px=True, warmup=1, eps_init=0.01, rho=0.9)


def test_scripts_kwargs_construct_a_head():
    lf = IBProbit(16, 4, key=jr.key(0), **_kwargs())
    assert lf.mu.shape == (17, 4)
    assert lf.Sigma.shape == (17, 17)


def test_scripts_kwargs_survive_a_streaming_pass():
    lf = IBProbit(16, 4, key=jr.key(0), **_kwargs()).reset(jr.key(9))
    keys = jr.split(jr.key(1), 2)
    fts = jr.normal(keys[0], (24, 16))
    y = jr.randint(keys[1], (24,), 0, 4)
    for _ in range(3):
        lf = lf.update(fts, y, num_iters=4)
    assert jnp.isfinite(lf.mu).all()
    assert jnp.isfinite(lf.Sigma).all()
    assert lf.step == 3
    loss = lf(fts, y)
    assert loss.shape == (24,) and jnp.isfinite(loss).all()


def test_predictive_variance_is_positive_and_per_input():
    lf = IBProbit(16, 4, key=jr.key(0), **_kwargs())
    v = lf.predictive_variance(jr.normal(jr.key(2), (7, 16)))
    assert v.shape == (7,) and (v > 0).all()


def test_px_can_be_passed_at_update_time():
    """px is settable per-update, not only at construction: `update(..., px=px)`
    overrides the constructor value. The now-removed ood_evaluation.py was the
    caller that used this, but the override is public API and `_cavi_scan_px`
    shares the plain fixed point exactly, so a solver chosen per-call stays a
    solver choice. Guard the path so it does not rot unnoticed."""
    ctor = {k: v for k, v in _kwargs().items() if k != "px"}
    lf = IBProbit(16, 4, key=jr.key(0), **ctor).reset(jr.key(9))
    keys = jr.split(jr.key(3), 2)
    fts = jr.normal(keys[0], (24, 16))
    y = jr.randint(keys[1], (24,), 0, 4)
    out = lf.update(fts, y, num_iters=4, px=True)
    assert jnp.isfinite(out.mu).all() and jnp.isfinite(out.Sigma).all()
    assert out.step == 1


def test_kwargs_alpha_matches_the_scripts_argparse_default():
    """_kwargs() claims to mirror the scripts; pin that for `alpha` specifically.

    alpha is the one value where a CLI/constructor mismatch is silent AND harmful:
    the scripts pass alpha=args.alpha explicitly, so the constructor default is
    bypassed, and at alpha=1e-3 the q recursion is inert (learn_q on/off give
    identical mu). A drift here would put the whole q study in the dead regime.
    """
    import importlib
    import sys
    from pathlib import Path
    alt_dir = str(Path(__file__).resolve().parents[1] / "scripts" / "dinov3")
    sys.path.insert(0, alt_dir)
    try:
        for mod in ("full_finetune_comparison", "vit_classification"):
            M = importlib.import_module(mod)
            p = M.build_argparser() if hasattr(M, "build_argparser") else M.get_parser()
            default = next(a.default for a in p._actions if a.dest == "alpha")
            assert default == _kwargs()["alpha"], (
                f"{mod} --alpha default {default} != _kwargs() alpha {_kwargs()['alpha']}")
    finally:
        sys.path.remove(alt_dir)
