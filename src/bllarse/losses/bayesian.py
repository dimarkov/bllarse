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

class MultinomialPolyaGamma(eqx.Module):
    num_classes: int
    input_dim: int
    use_bias: bool
    loss_type: int
    mu: Array
    Sigma: Array

    def __init__(
        self,
        input_dim: int,
        num_classes:int,
        *,
        key: PRNGKey,
        use_bias: bool = False,
        loss_type: int = 0
    ):

        self.num_classes = num_classes
        self.input_dim = input_dim
        self.use_bias = use_bias
        self.loss_type = loss_type

        d = input_dim + int(use_bias)
        k = num_classes - 1
        self.mu = jr.normal((d, k)) * 1e-3
        self.Sigma = jnp.eye(d)[None].repeat(k, axis=0)

    def reset(self, key: PRNGKey) -> "MultinomialPolyaGamma":
        d = self.input_dim + int(self.use_bias)
        k = self.num_classes - 1
        mu = jr.normal((d, k)) * 1e-3
        Sigma = jnp.eye(d)[None].repeat(k, axis=0)

        return eqx.tree_at(lambda x: (x.mu, x.Sigma), self, (mu, Sigma))

    def get_b_kappa(self, y_one_hot: Array):
        delta_ky = y_one_hot[..., :-1]
        b = 1 - jnp.pad(y_one_hot, [(0, 0), (1, 0)])[..., :-2].cumsum(-1)
        kappa = delta_ky - b / 2

        return b, kappa

    def __call__(self, features: Array, y: Array, *, with_logits: bool = False):
        
        x = jnp.pad(features, [(0, 0), (0, int(self.use_bias))], constant_values=1.0)
        M_x = x[..., None, :] * x[..., None]

        y_onehots = nn.one_hot(y, self.num_classes)
        b, kappa = self.get_b_kappa(y_onehots)
        lam = jnp.sum(lax.stop_gradient(x)[..., None] * kappa[..., None, :], axis=0)

        def step_fn(carry, _):

            mu, Sigma, M_x = carry

            M_beta = Sigma + mu @ mu.mT
            psi = jnp.sqrt(jnp.sum(M_beta * jnp.expand_dims(M_x, -3), axis=(-1, -2)))

            rho = b * jnp.tanh(psi / 2) / (psi + 1e-8) / 2
            F = jnp.sum(rho[:, None, None] * M_x, axis=0) / 2
            tmp = jnp.eye(x.shape[-1]) + self.Sigma @ F 
            Sigma_new = lu_solve(lu_factor(tmp), self.Sigma)
            mu_new = Sigma_new @ lam

            return (mu_new, Sigma_new, M_x), None

        init = (self.mu, self.Sigma, lax.stop_gradient(M_x))
        (params, covariance, _), _ = lax.scan(step_fn, init, jnp.arange(self.num_iters))

        # compute loss as  E_{q(\beta)q(\omega)}[\ln q(\omega) - \log p(y, \omega|\beta, x)]
        weights = params[:-1] if self.use_bias else params
        bias = params[-1] if self.use_bias else 0.0

        logits = features @ weights + bias
        
        if self.loss_type == 0:
            logits = jnp.pad(logits, [(0, 0), (0, 1)]) - jnp.sum(b * nn.softplus(logits), -1)
            loss = - (y_onehots * logits).sum(-1)

        elif self.loss_type == 1:
            M_beta = covariance + params @ params.mT
            psi = jnp.sqrt(
                jnp.sum(M_beta * jnp.expand_dims(M_x, -3), axis=(-1, -2))
            )
            logits = jnp.pad(logits, [(0, 0), (0, 1)]) \
                + jnp.sum(b * ((psi - logits) / 2  - nn.softplus(psi)) )
            # loss = jnp.sum(logits * kappa + b * (psi / 2 - nn.softplus(psi) ), -1)
            loss = - (y_onehots * logits).sum(-1)

        return loss, logits if with_logits else loss
