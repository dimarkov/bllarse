"""run_training must expose the blrax gradient estimator (sampling vs hutchinson)
for the IVON/EVON bayesian optimizers, while default behaviour stays unchanged.

CPU-friendly: uses the classical CrossEntropy loss (no TF32 matmuls) with a tiny
MLP; ivon/evon are pure JAX and run on any backend.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from blrax.optim import evon, ivon

from bllarse.losses import CrossEntropy
from bllarse.utils import run_training

N, D, H, K = 32, 6, 8, 4
BATCH = 8


class TinyNet(eqx.Module):
    lin: eqx.nn.Linear
    fc: eqx.nn.Linear

    def __init__(self, key):
        k1, k2 = jr.split(key)
        self.lin = eqx.nn.Linear(D, H, key=k1)
        self.fc = eqx.nn.Linear(H, K, key=k2)

    def __call__(self, x):
        return self.fc(jax.nn.relu(self.lin(x)))


def _identity_aug(img, key=None, **kwargs):
    return img


def _toy_ds(key):
    k_x, k_y = jr.split(key)
    images = jr.normal(k_x, (N, D))
    labels = jr.randint(k_y, (N,), 0, K)
    return {"image": images, "label": labels}


@pytest.mark.parametrize(
    "make_optim,estimator",
    [
        (lambda: ivon(1e-2, ess=float(N), hess_init=1.0), "sampling"),
        (lambda: ivon(1e-2, ess=float(N), hess_init=1.0), "hutchinson"),
        (lambda: evon(1e-2, ess=float(N), hess_init=1.0), "sampling"),
        (lambda: evon(1e-2, ess=float(N), hess_init=1.0), "hutchinson"),
    ],
)
def test_run_training_accepts_grad_estimator(make_optim, estimator):
    key = jr.PRNGKey(0)
    ds = _toy_ds(jr.fold_in(key, 1))
    nnet = TinyNet(jr.fold_in(key, 2))
    loss = CrossEntropy(0.0, K)

    trained_loss, trained_nnet, opt_state, metrics = run_training(
        key,
        nnet,
        loss,
        _identity_aug,
        ds,
        ds,
        optimizer=make_optim(),
        tune_last_layer_only=False,
        num_epochs=2,
        batch_size=BATCH,
        mc_samples=1,
        grad_estimator=estimator,
    )

    assert jnp.isfinite(metrics["loss"]).all()
    assert jnp.isfinite(metrics["nll"]).all()
    # the network must actually have moved
    assert not jnp.allclose(trained_nnet.fc.weight, nnet.fc.weight)


def test_run_training_default_estimator_unchanged():
    """Callers that never pass grad_estimator keep the sampling behaviour."""
    key = jr.PRNGKey(3)
    ds = _toy_ds(jr.fold_in(key, 1))
    nnet = TinyNet(jr.fold_in(key, 2))
    loss = CrossEntropy(0.0, K)

    run = lambda **kw: run_training(
        key, nnet, loss, _identity_aug, ds, ds,
        optimizer=ivon(1e-2, ess=float(N), hess_init=1.0),
        tune_last_layer_only=False, num_epochs=1, batch_size=BATCH,
        mc_samples=1, **kw,
    )

    _, nnet_default, _, m_default = run()
    _, nnet_sampling, _, m_sampling = run(grad_estimator="sampling")

    assert jnp.allclose(m_default["loss"], m_sampling["loss"])
    assert jnp.allclose(nnet_default.fc.weight, nnet_sampling.fc.weight)
