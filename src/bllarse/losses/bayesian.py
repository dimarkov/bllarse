import equinox as eqx
import optax
import jax.numpy as jnp
from jax import nn, vmap, random as jr, lax
from jax.scipy.stats import norm
from jax.scipy.linalg import lu_factor, lu_solve
from jaxtyping import Array, PRNGKeyArray as PRNGKey
from typing import Optional, Tuple
from functools import partial

const = jnp.sqrt(2 / jnp.pi)

def approx_cdf(x):
    return 0.5 * ( 1 + jnp.tanh(const * (x + 0.044715 * x ** 3)))

def apporx_pdf(x):
    v = const * (x + 0.044715 * x ** 3)
    return 0.5 * (1 - jnp.tanh(v) ** 2) * ( const * (1 + 3 * 0.044715 * x ** 2))


class BayesianMultinomialProbit(eqx.Module):
    """
    A stateful loss function that implements a Bayesian final layer using
    variational inference for a multinomial probit model.
    """
    eta: Array
    Sigma: Array
    num_classes: int
    embed_dim: int
    use_bias: bool
    loss_type: int
    num_iters: int

    def __init__(self, key: PRNGKey, embed_dim: int, num_classes: int, use_bias: bool = True, num_iters: int = 8, loss_type: int = 0):
        self.eta = jr.normal(key, shape=(embed_dim + int(use_bias), num_classes)) * 1e-3
        self.Sigma = jnp.eye(embed_dim + int(use_bias))
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_bias = use_bias
        self.num_iters = num_iters
        self.loss_type = loss_type

    def reset(self, key: PRNGKey) -> "BayesianMultinomialProbit":
        d, num_classes = self.eta.shape
        eta = jr.normal(key, shape=(d, num_classes)) * 0.01
        Sigma = jnp.eye(d)
        return eqx.tree_at(lambda x: (x.eta, x.Sigma), self, (eta, Sigma))

    def update(self, features: Array, y: Array) -> "BayesianMultinomialProbit":
        fts = jnp.pad(lax.stop_gradient(features), [(0, 0), (0, int(self.use_bias))], constant_values=1.0)

        luf = lu_factor(jnp.eye(fts.shape[-1]) + self.Sigma @ (fts.T @ fts))
        Sigma_new = lu_solve(luf, self.Sigma)
        y_one_hot = nn.one_hot(y, self.num_classes)
        
        x = fts @ Sigma_new

        def step_fn(carry, *args):
            # One step of Coordinate Ascent Variational Inference (CAVI) for the probit model.
            eta = carry

            pred = x @ eta

            # Compute E_q[z_ik]
            phi = norm.pdf(-pred)
            Phi = norm.cdf(-pred)

            # Update variational mean
            E_q_z = pred + phi * (y_one_hot + Phi - 1) / (Phi * (1 - Phi) + 1e-8)
            eta = self.eta + fts.T @ E_q_z
            return eta, None

        eta_new, _ = lax.scan(step_fn, self.eta, jnp.arange(self.num_iters - 1))

        return eqx.tree_at(lambda x: (x.eta, x.Sigma), self, (eta_new, Sigma_new))

    def __call__(self, features: Array, y: Array, *, with_logits: bool = False) -> Tuple[Array, Array, "BayesianMultinomialProbit"]:
        if self.use_bias:
            fts = jnp.pad(lax.stop_gradient(features), [(0, 0), (0, 1)], constant_values=1.0)
        else:
            fts = lax.stop_gradient(features)

        luf = lu_factor(jnp.eye(fts.shape[-1]) + self.Sigma @ (fts.T @ fts))
        Sigma_new = lu_solve(luf, self.Sigma)
        y_one_hot = nn.one_hot(y, self.num_classes)

        def step_fn(carry, *args):
            # One step of Coordinate Ascent Variational Inference (CAVI) for the probit model.
            eta = carry

            pred = fts @ (Sigma_new @ eta)

            # Compute E_q[z_ik]
            phi = norm.pdf(-pred)
            Phi = norm.cdf(-pred)

            # Update variational mean
            E_q_z = pred + phi * (y_one_hot + Phi - 1) / (Phi * (1 - Phi) + 1e-8)
            eta = self.eta + fts.T @ E_q_z
            return eta, None

        eta_new, _ = lax.scan(step_fn, self.eta, jnp.arange(self.num_iters - 1))
        params = Sigma_new @ eta_new

        # compute loss
        if self.use_bias:
            weights = params[:-1]
            bias = params[-1]
        else:
            weights = params
            bias = 0.0

        logits = features @ weights + bias

        # note: computation of loss is relevant for gradients over model parameters
        if self.loss_type == 0:
            # binary loss
            loss = optax.sigmoid_binary_cross_entropy(logits, y_one_hot).sum(-1).mean()
        elif self.loss_type == 1:
            # CBM Loss
            log_probs = jnp.log(norm.cdf(logits))
            loss = optax.softmax_cross_entropy(log_probs, y_one_hot).mean()
        elif self.loss_type == 2:
            # CBC Loss
            probs = norm.cdf(logits)
            log_probs = jnp.log(probs) - jnp.log(1 - probs)
            loss = optax.softmax_cross_entropy(log_probs, y_one_hot).mean()
        elif self.loss_type == 3:
            # mixed loss
            probs = norm.cdf(logits)
            log_probs1 = jnp.log(probs)
            log_probs2 = log_probs1 - jnp.log(1 - probs)

            loss_cbm = optax.softmax_cross_entropy(log_probs1, y_one_hot).mean()
            loss_cbc = optax.softmax_cross_entropy(log_probs2, y_one_hot).mean()

            loss = loss_cbm + nn.softplus(loss_cbc - loss_cbm)

        new_self = eqx.tree_at(lambda x: (x.eta, x.Sigma), self, (eta_new, Sigma_new))

        if with_logits:
            return loss, logits, new_self
        else:
            return loss, new_self
