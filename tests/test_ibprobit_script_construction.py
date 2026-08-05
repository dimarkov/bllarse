"""Guards the API contract between IBProbit and its three consuming scripts.

726bc78 landed ood_evaluation.py, full_finetune_comparison.py and
vit_classification.py on this branch without the IBProbit they call, and the
suite did not notice. These assertions are cheap and would have.
"""
import jax.numpy as jnp
import jax.random as jr
import pytest

from bllarse.losses.bayesian import IBProbit


def _kwargs():
    """Exactly the kwarg set the three scripts construct with."""
    return dict(q_init=1e-3, sigma_z2_init=1.0, learn_q=True, hyper_lr=0.1,
                anchor_weight=1e-3, px=True, bmr_gate=True, bmr_warmup=1,
                learn_eps=True, eps_init=0.01, rho=0.9)


def test_scripts_kwargs_construct_a_head():
    lf = IBProbit(16, 4, key=jr.key(0), **_kwargs())
    assert lf.mu.shape == (17, 4)
    assert lf.Sigma.shape == (17, 17)


def test_scripts_kwargs_survive_a_streaming_pass():
    lf = IBProbit(16, 4, key=jr.key(0), **_kwargs()).reset_posterior()
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
