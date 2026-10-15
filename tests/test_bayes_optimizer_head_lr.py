"""bayes_optimizer must build IVON/EVON with an optional separate head lr while
keeping the blrax state at opt_state[0] (required by noisy_value_and_grad and the
comparison-script glue, which introspect that slot).

Per-part lr is only a rescaling of the final update: with identical grads the
head part must match a plain ivon/evon built with head_lr and the backbone part
one built with lr (the scale_by state evolution is lr-independent).
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from blrax.optim import evon, ivon
from blrax.states import ScaleByEvonState, ScaleByIvonState, _is_evon_leaf
from blrax.utils import noisy_value_and_grad

from bllarse.utils import bayes_optimizer

LR, HEAD_LR = 1e-2, 1e-1
CONF = dict(ess=100.0, hess_init=1.0, weight_decay=1e-4)

LABELS = {"backbone": {"w": "backbone", "b": "backbone"}, "head": "head"}


def _params(key):
    kw, kh = jr.split(key)
    return {
        "backbone": {"w": jr.normal(kw, (5, 3)), "b": jnp.zeros(3)},
        "head": jr.normal(kh, (3, 2)),
    }


def _stash(opt_state, params):
    """Minimal per-step hessian stash so update() can run outside blrax's flow."""
    st = opt_state[0]
    if isinstance(st, ScaleByIvonState):
        st = st._replace(h_bar=jax.tree.map(jnp.zeros_like, st.hess))
    else:
        leaves = jax.tree.leaves(st.leaves, is_leaf=_is_evon_leaf)
        _, treedef = jax.tree.flatten(params)
        st = st._replace(
            leaves=jax.tree.unflatten(treedef, [l._replace(h_hat=l.H) for l in leaves]))
    return (st,) + tuple(opt_state[1:])


def _one_update(optimizer, params, grads):
    state = _stash(optimizer.init(params), params)
    updates, _ = optimizer.update(grads, state, params)
    return updates


@pytest.mark.parametrize("name,plain", [("ivon", ivon), ("evon", evon)])
def test_state_first_and_per_part_lr(name, plain):
    params = _params(jr.PRNGKey(0))
    grads = jax.tree.map(lambda p: 0.1 * jnp.ones_like(p), params)

    opt = bayes_optimizer(name, LR, head_learning_rate=HEAD_LR,
                          param_labels=LABELS, clip_radius=1e3, **CONF)
    state = opt.init(params)
    expected = ScaleByIvonState if name == "ivon" else ScaleByEvonState
    assert isinstance(state[0], expected)

    u = _one_update(opt, params, grads)
    u_lr = _one_update(plain(LR, clip_radius=1e3, **CONF), params, grads)
    u_head = _one_update(plain(HEAD_LR, clip_radius=1e3, **CONF), params, grads)

    assert jnp.allclose(u["head"], u_head["head"])
    assert jnp.allclose(u["backbone"]["w"], u_lr["backbone"]["w"])
    assert jnp.allclose(u["backbone"]["b"], u_lr["backbone"]["b"])
    # sanity: the two parts really differ (head 10x larger)
    assert not jnp.allclose(u["head"], u_lr["head"])


@pytest.mark.parametrize("name,plain", [("ivon", ivon), ("evon", evon)])
def test_single_lr_path_matches_plain_constructor(name, plain):
    params = _params(jr.PRNGKey(1))
    grads = jax.tree.map(lambda p: 0.1 * jnp.ones_like(p), params)
    u = _one_update(bayes_optimizer(name, LR, clip_radius=1e3, **CONF), params, grads)
    u_ref = _one_update(plain(LR, clip_radius=1e3, **CONF), params, grads)
    jax.tree.map(lambda a, b: _assert_close(a, b), u, u_ref)


def test_noisy_value_and_grad_still_routes_to_ivon():
    """The flat chain must keep blrax's estimator dispatch working."""
    params = _params(jr.PRNGKey(2))
    opt = bayes_optimizer("ivon", LR, head_learning_rate=HEAD_LR,
                          param_labels=LABELS, clip_radius=1e3, **CONF)
    state = opt.init(params)

    def loss(p, key):
        return jnp.sum(p["head"] ** 2) + jnp.sum(p["backbone"]["w"] ** 2)

    _, grads, state = noisy_value_and_grad(loss, state, params, jr.PRNGKey(0))
    assert state[0].h_bar is not None  # the ivon sampling branch ran
    updates, _ = opt.update(grads, state, params)
    assert all(jnp.isfinite(u).all() for u in jax.tree.leaves(updates))


def test_head_lr_requires_labels():
    with pytest.raises(ValueError, match="param_labels"):
        bayes_optimizer("ivon", LR, head_learning_rate=HEAD_LR, **CONF)


@pytest.mark.parametrize("name", ["ivon", "evon"])
def test_run_training_with_fc_labeled_head_lr_chain(name):
    """End-to-end: run_training + flat head-lr chain + the .fc-labeling callable
    used by the full-network finetuning path (classical loss)."""
    import equinox as eqx
    import jax.tree as jtu
    from bllarse.losses import CrossEntropy
    from bllarse.utils import run_training

    class TinyNet(eqx.Module):
        lin: eqx.nn.Linear
        fc: eqx.nn.Linear

        def __init__(self, key):
            k1, k2 = jr.split(key)
            self.lin = eqx.nn.Linear(6, 8, key=k1)
            self.fc = eqx.nn.Linear(8, 4, key=k2)

        def __call__(self, x):
            return self.fc(jax.nn.relu(self.lin(x)))

    def _labels(params):
        labels = jtu.map(lambda _: "backbone", params)
        return eqx.tree_at(
            lambda m: m.fc, labels,
            replace_fn=lambda fc: jtu.map(lambda _: "head", fc))

    key = jr.PRNGKey(0)
    n = 32
    ds = {
        "image": jr.normal(jr.fold_in(key, 1), (n, 6)),
        "label": jr.randint(jr.fold_in(key, 2), (n,), 0, 4),
    }
    nnet = TinyNet(jr.fold_in(key, 3))
    optim = bayes_optimizer(
        name, LR, head_learning_rate=HEAD_LR, param_labels=_labels,
        clip_radius=1e3, ess=float(n), hess_init=1.0, weight_decay=1e-4)

    _, trained, _, metrics = run_training(
        key, nnet, CrossEntropy(0.0, 4), lambda img, key=None, **kw: img,
        ds, ds, optimizer=optim, tune_last_layer_only=False,
        num_epochs=2, batch_size=8, mc_samples=1)

    assert jnp.isfinite(metrics["loss"]).all()
    assert not jnp.allclose(trained.fc.weight, nnet.fc.weight)
    assert not jnp.allclose(trained.lin.weight, nnet.lin.weight)


def _assert_close(a, b):
    assert jnp.allclose(a, b), f"max diff {jnp.max(jnp.abs(a - b))}"
