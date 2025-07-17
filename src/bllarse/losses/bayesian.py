import equinox as eqx
import optax
import jax.numpy as jnp
from jax import nn, vmap, random as jr, lax
from jax.scipy.stats import norm
from jax.scipy.linalg import lu_factor, lu_solve, solve
from jaxtyping import Array, PRNGKeyArray as PRNGKey
from typing import Optional, Tuple, Callable
from functools import partial

const = jnp.sqrt(2 / jnp.pi)

def approx_cdf(x):
    return 0.5 * ( 1 + jnp.tanh(const * (x + 0.044715 * x ** 3)))

class IBProbit(eqx.Module):
    """
    A stateful loss function that implements a Bayesian final layer using
    variational inference for a multinomial probit model.
    """
    eta: Array  # natural parameter Sigma_inv @ mu
    Sigma: Array
    use_bias: bool
    cdf: Callable

    def __init__(
        self,
        input_dim: int,
        num_classes: int, 
        *,
        key: PRNGKey,
        use_bias: bool = True,
        use_approx_cdf: bool = True
    ):

        self.eta = jr.normal(key, shape=(input_dim + int(use_bias), num_classes)) * 1e-3
        self.Sigma = jnp.eye(input_dim + int(use_bias))
        self.use_bias = use_bias
        self.cdf = approx_cdf if use_approx_cdf else norm.cdf

    def reset(self, key: PRNGKey) -> "IBProbit":
        d, num_classes = self.eta.shape
        eta = jr.normal(key, shape=(d, num_classes)) * 0.01
        Sigma = jnp.eye(d)
        return eqx.tree_at(lambda x: (x.eta, x.Sigma), self, (eta, Sigma))

    def update(self, features: Array, y: Array, *, num_iters: int = 32) -> "IBProbit":
        fts = jnp.pad(lax.stop_gradient(features), [(0, 0), (0, int(self.use_bias))], constant_values=1.0)

        luf = lu_factor(jnp.eye(fts.shape[-1]) + self.Sigma @ (fts.T @ fts))
        Sigma_new = lu_solve(luf, self.Sigma)
        y_one_hot = nn.one_hot(y, self.eta.shape[-1])
        
        x = fts @ Sigma_new

        def step_fn(carry, *args):
            # One step of Coordinate Ascent Variational Inference (CAVI) for the probit model.
            eta = carry

            pred = x @ eta

            # Compute E_q[z_ik]
            phi = norm.pdf(-pred)
            Phi = self.cdf(-pred)

            # Update variational mean
            E_q_z = pred + phi * (y_one_hot + Phi - 1) / (Phi * (1 - Phi) + 1e-8)
            eta = self.eta + fts.T @ E_q_z
            return eta, None

        eta_new, _ = lax.scan(step_fn, self.eta, jnp.arange(num_iters))

        return eqx.tree_at(lambda x: (x.eta, x.Sigma), self, (eta_new, Sigma_new))
    
    @property
    def params(self) -> Tuple[Array, Array]:
        params = self.Sigma @ self.eta
        return (params[:-1], params[-1]) if self.use_bias else (params, None)

    def __call__(self, features: Array, y: Array, *, with_logits: bool = False, loss_type: int = 3) -> Tuple[Array, Optional[Array]]:
        weights, bias = self.params

        y_one_hot = nn.one_hot(y, weights.shape[-1])

        # compute loss
        logits = features @ weights + bias if self.use_bias else features @ weights

        # note: computation of loss is relevant for gradients over model parameters
        if loss_type == 0:
            # binary loss
            loss = optax.sigmoid_binary_cross_entropy(logits, y_one_hot).sum(-1)

        elif loss_type == 1:
            # CBM Loss
            logits = jnp.log(self.cdf(logits))
            loss = optax.safe_softmax_cross_entropy(logits, y_one_hot)

        elif loss_type == 2:
            # CBC Loss
            probs = self.cdf(logits)
            logits = jnp.nan_to_num(jnp.log(probs) - jnp.log(1 - probs))
            loss = optax.safe_softmax_cross_entropy(logits, y_one_hot)

        elif loss_type == 3:
            # mixed loss
            probs = self.cdf(logits)
            logits = jnp.log(probs)
            logits2 = jnp.nan_to_num(logits - jnp.log(1 - probs))

            loss_cbm = optax.safe_softmax_cross_entropy(jnp.nan_to_num(logits), y_one_hot)
            loss_cbc = optax.safe_softmax_cross_entropy(logits2, y_one_hot)

            loss = loss_cbm - nn.softplus(loss_cbm - loss_cbc) + jnp.log(2)

        if with_logits:
            return loss, logits
        else:
            return loss

class MultinomialPolyaGamma(eqx.Module):

    mu: Array
    Sigma: Array
    use_bias: bool

    def __init__(
        self,
        input_dim: int,
        num_classes:int,
        *,
        use_bias: bool = False,
        key: PRNGKey
    ):

        self.use_bias = use_bias
        d = input_dim + int(use_bias)
        k = num_classes - 1
        self.mu = jr.normal(key, (d, k)) * 1e-3
        self.Sigma = jnp.eye(d)[None].repeat(k, axis=0)

    def reset(self, key: PRNGKey) -> "MultinomialPolyaGamma":
        d, k = self.mu.shape
        mu = jr.normal(key, (d, k)) * 1e-3
        Sigma = jnp.eye(d)[None].repeat(k, axis=0)

        return eqx.tree_at(lambda x: (x.mu, x.Sigma), self, (mu, Sigma))

    def __get_b_ky(self):
        nc = self.mu.shape[-1] + 1
        return 1 - jnp.pad(jnp.eye(nc)[..., :-2], [(0, 0), (1, 0)]).cumsum(-1)

    def get_b_kappa(self, y: Array):
        y_onehot = nn.one_hot(y, self.mu.shape[-1] + 1)
        b = y_onehot @ self.__get_b_ky()
        kappa = y_onehot[..., :-1] - b / 2

        return b, kappa

    def update(self, features: Array, y: Array, *, num_iters: int = 32) -> "MultinomialPolyaGamma":

        x = jnp.pad(lax.stop_gradient(features), [(0, 0), (0, int(self.use_bias))], constant_values=1.0)
        xxT = jnp.expand_dims(x[..., None, :] * x[..., None], -3)
        b, kappa = self.get_b_kappa(y)

        # get first nat parameter for beta posterior
        lam = jnp.expand_dims(jnp.sum(x[..., None, :] * kappa[..., None], axis=0), -1)

        def step_fn(carry, _):

            mu, Sigma = carry

            E_betabetaT = Sigma + mu[..., None, :] * mu[..., None]
            psi = jnp.sqrt(jnp.sum(E_betabetaT * xxT, axis=(-1, -2)))

            rho = 0.5 * b * jnp.tanh(psi / 2) / (psi + 1e-8)
            F = jnp.sum(rho[..., None, None] * xxT, axis=0)
            tmp = jnp.eye(x.shape[-1]) + self.Sigma @ F 
            Sigma_new = lu_solve(lu_factor(tmp), self.Sigma)
            mu_new = (Sigma_new @ lam).squeeze(-1)

            return (mu_new, Sigma_new), None

        init = (self.mu.mT, self.Sigma)
        (mu_trans, Sigma), _ = lax.scan(step_fn, init, jnp.arange(num_iters))

        return eqx.tree_at(lambda x: (x.mu, x.Sigma), self, (mu_trans.mT, Sigma))

    def __call__(self, features: Array, y: Array, *, with_logits: bool = False, loss_type: int = 0):

        x = jnp.pad(features, [(0, 0), (0, int(self.use_bias))], constant_values=1.0)
        logits = x @ self.mu
        b_ky = self.__get_b_ky().mT
        
        if loss_type == 0:
            logits = jnp.pad(logits, [(0, 0), (0, 1)]) - nn.softplus(logits) @ b_ky
            loss = - jnp.sum(logits * nn.one_hot(y, self.mu.shape[-1] + 1), -1)
            return (loss, logits) if with_logits else loss

        elif loss_type == 1:
            params = self.mu.mT
            E_betabetaT = self.Sigma + params[:, None, :] * params[..., None]
            
            xxT = jnp.expand_dims(x[..., None, :] * x[..., None], -3)
            psi = jnp.sqrt(
                jnp.sum(E_betabetaT * xxT, axis=(-1, -2))
            )

            if with_logits:
                logits = jnp.pad(logits, [(0, 0), (0, 1)]) + ((psi - logits) / 2  - nn.softplus(psi)) @ b_ky
                loss = - jnp.sum(logits * nn.one_hot(y, self.mu.shape[-1] + 1), -1)
                return loss, logits
            else:
                b, kappa = self.get_b_kappa(y)
                loss = - jnp.sum(logits * kappa + b * (psi / 2 - nn.softplus(psi) ), -1)
                return loss


class IBPolyaGamma(eqx.Module):

    mu: Array
    Sigma: Array
    use_bias: bool

    def __init__(
        self,
        input_dim: int,
        num_classes:int,
        *,
        use_bias: bool = False,
        key: PRNGKey
    ):

        self.use_bias = use_bias
        d = input_dim + int(use_bias)
        self.mu = jr.normal(key, (d, num_classes)) * 1e-3
        self.Sigma = jnp.eye(d)[None].repeat(num_classes, axis=0)

    def reset(self, key: PRNGKey) -> "IBPolyaGamma":
        d, k = self.mu.shape
        mu = jr.normal(key, (d, k)) * 1e-3
        Sigma = jnp.eye(d)[None].repeat(k, axis=0)

        return eqx.tree_at(lambda x: (x.mu, x.Sigma), self, (mu, Sigma))

    def get_kappa(self, y: Array):
        return nn.one_hot(y, self.mu.shape[-1]) - 0.5

    def update(self, features: Array, y: Array, *, num_iters: int = 32) -> "IBPolyaGamma":

        x = jnp.pad(lax.stop_gradient(features), [(0, 0), (0, int(self.use_bias))], constant_values=1.0)
        xxT = jnp.expand_dims(x[..., None, :] * x[..., None], -3)
        # b is always 1 for binary case
        kappa = self.get_kappa(y)

        # get first nat parameter for beta posterior
        eta = solve(self.Sigma, self.mu.T[..., None], assume_a='pos')
        lam = jnp.expand_dims(jnp.sum(x[..., None, :] * kappa[..., None], axis=0), -1) + eta

        def step_fn(carry, _):

            mu, Sigma = carry

            E_betabetaT = Sigma + mu[..., None, :] * mu[..., None]
            psi = jnp.sqrt(jnp.sum(E_betabetaT * xxT, axis=(-1, -2)))

            # as b is one always we do not need it here
            rho = 0.5 * jnp.tanh(psi / 2) / (psi + 1e-8)
            F = jnp.sum(rho[..., None, None] * xxT, axis=0)
            tmp = jnp.eye(x.shape[-1]) + self.Sigma @ F 
            Sigma_new = lu_solve(lu_factor(tmp), self.Sigma)
            mu_new = (Sigma_new @ lam).squeeze(-1)

            return (mu_new, Sigma_new), None

        init = (self.mu.mT, self.Sigma)
        (final_mu_trans, final_Sigma), _ = lax.scan(step_fn, init, jnp.arange(num_iters))

        return eqx.tree_at(lambda x: (x.mu, x.Sigma), self, (final_mu_trans.T, final_Sigma))

    @property
    def params(self):
        return (self.mu[:-1], self.mu[-1]) if self.use_bias else self.mu

    def __call__(self, features: Array, y: Array, *, with_logits: bool = False, loss_type: int = 3) -> Tuple[Array, Optional[Array]]:
        x = jnp.pad(features, [(0, 0), (0, int(self.use_bias))], constant_values=1.0)
        y_one_hot = nn.one_hot(y, self.mu.shape[-1])

        # compute loss
        pred = x @ self.mu

        # note: computation of loss is relevant for gradients over model parameters
        if loss_type == 0:
            # binary loss
            logits = pred
            loss = optax.sigmoid_binary_cross_entropy(logits, y_one_hot).sum(-1)

        elif loss_type == 1:
            # CBM Loss
            # compute log p(y=1|params, x)
            logits = pred - nn.softplus(pred)
            # alternative but slower to compute
            # params = self.mu.mT
            # E_betabetaT = self.Sigma + params[:, None, :] * params[..., None]
            # xxT = jnp.expand_dims(x[..., None, :] * x[..., None], -3)
            # psi = jnp.sqrt(jnp.sum(E_betabetaT * xxT, axis=(-1, -2)))
            # logits = (psi - pred) / 2  - nn.softplus(psi)

            loss = optax.safe_softmax_cross_entropy(logits, y_one_hot)

        elif loss_type == 2:
            # CBC Loss
            logits = pred
            loss = optax.safe_softmax_cross_entropy(logits, y_one_hot)

        elif loss_type == 3:
            # mixed loss
            logits = pred
            loss_cbm = optax.safe_softmax_cross_entropy(logits - nn.softplus(pred), y_one_hot)
            loss_cbc = optax.safe_softmax_cross_entropy(logits, y_one_hot)

            loss = loss_cbm - nn.softplus(loss_cbm - loss_cbc) + jnp.log(2)

        if with_logits:
            return loss, logits
        else:
            return loss