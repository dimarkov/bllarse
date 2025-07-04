import equinox as eqx
import jax
from jax import numpy as jnp, random as jr
from jax import nn
from jaxtyping import Array
from bllarse.utils import stable_inverse

class MVN_NatParams_Fast(eqx.Module):
    dim: int
    eta: Array
    Lambda_chol: Array

    def __init__(self, eta, Lambda_chol):
        self.eta = eta
        self.Lambda_chol = Lambda_chol
        self.dim = eta.shape[-1]   

    def mu(self):
        # solve Lᵀ μ = L⁻¹ η   ⇒   μ = (Lᵀ)⁻¹ (L⁻¹ η)
        y  = jax.lax.linalg.triangular_solve(self.Lambda_chol, self.eta, left_side=True,
                                            lower=True, transpose_a=False)
        mu = jax.lax.linalg.triangular_solve(self.Lambda_chol, y,left_side=True,
                                            lower=True, transpose_a=True)
        return mu  
    
class MVN_NatParams(eqx.Module):
    dim: int
    inv_sigma_mu: Array
    inv_sigma: Array

    def __init__(self, inv_sigma_mu, inv_sigma):
        self.inv_sigma_mu = inv_sigma_mu
        self.inv_sigma = inv_sigma
        self.dim = inv_sigma_mu.shape[-1]   

    def sigma(self):
        """ Compute covariance matrix from the precision matrix. """
        return stable_inverse(self.inv_sigma)
    
    def _to_expectations(self):
        """ Convert natural parameters to expectations."""
        sigma = self.sigma()
        expected_x = (sigma @ self.inv_sigma_mu[...,None])
        expected_xx = sigma + (expected_x @ expected_x.mT)
        return MVN_Expectations(e_x=expected_x.squeeze(-1), e_xx=expected_xx)

    def _to_moments(self):
        """ Convert natural parameters to moments."""
        sigma = self.sigma()
        mu = sigma @ self.inv_sigma_mu[..., None]
        return MVN_Moments(mu=mu.squeeze(-1), sigma=sigma)

class MVN_Expectations(eqx.Module):
    dim: int
    e_x: Array
    e_xx: Array

    def __init__(self, e_x, e_xx):
        self.e_x = e_x
        self.e_xx = e_xx
        self.dim = e_x.shape[-1]
    
    def to_nat_params(self):
        """
        Convert expectations to natural parameters.
        """
        sigma = self.e_xx - (self.e_x[..., None] @ self.e_x[..., None].mT)
        inv_sigma = stable_inverse(sigma)
        inv_sigma_mu = inv_sigma @ self.e_x[..., None]
        return MVN_NatParams(inv_sigma_mu=inv_sigma_mu.squeeze(-1), inv_sigma=inv_sigma)
    
    def to_moments(self):
        """
        Convert expectations to moments.
        """
        sigma = self.e_xx - (self.e_x[..., None] @ self.e_x[..., None].mT)
        return MVN_Moments(mu=self.e_x, sigma=sigma)

class MVN_Moments(eqx.Module):
    dim: int
    mu: Array
    sigma: Array
    
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.dim = mu.shape[-1]
    
    def to_nat_params(self):
        """
        Convert moments to natural parameters.
        """
        inv_sigma = stable_inverse(self.sigma)
        inv_sigma_mu = inv_sigma @ self.mu[..., None]
        return MVN_NatParams(inv_sigma_mu=inv_sigma_mu.squeeze(-1), inv_sigma=inv_sigma)

    def to_expectations(self):
        """
        Convert moments to expectations.
        """
        e_x = self.mu
        e_xx = self.sigma + (e_x[..., None] @ e_x[..., None].mT)
        return MVN_Expectations(e_x=e_x, e_xx=e_xx)



class MultinomialLogistic_Polyagamma(eqx.Module):
    """
    Multinomial logistic regression layer that uses polyagamma augmentation to lower-bound of the stick-breaking parameterization
    of the multinomial logistic likelihood.
    """
    num_classes: int
    stick_breaking_dim: int # where in the shapes of the parameters the output dimension is stored, counting from the right
    sb_weights: eqx.Module # this will be one of the instances of the MVN_NatParams, MVN_Expectations, or MVN_Moments classes

    def __init__(self, args):
        pass

    def __call__(self, x):
        """ Forward pass that predicts logits given inputs x"""

        beta_mu = self.sb_weights.compute_mu()
        batch_dimensions = len(x.shape[:-1])  # batch dimensions of the input
        class_onehots = jnp.expand_dims(jnp.eye(self.num_classes, dtype=x.dtype), tuple(range(1, len(batch_dimensions)+1)))  # (1, ..., 1, C, C)

        pg_b = self.compute_y_stats(class_onehots)

        kappa = class_onehots[...,:-1] - 0.5 * pg_b
        pg_c = jnp.sqrt(self.compute_psipsi(beta_mu, self.sb_weights.Lambda_chol, x))
        psi = (jnp.expand_dims(x, -2) * beta_mu).sum(-1)

        logits = jnp.sum(kappa * psi + 0.5 * (pg_b * pg_c) - pg_b * nn.softplus(pg_c), -1)
        return jnp.moveaxis(logits, 0, -1)  # move the class dimension to the end
        
    def compute_psipsi(self, mu, L, x):
        """ Efficient form of computation of psipsi which takes advantage of rank-1 form of E[xx^T].
        
        Details: If the input regressors x are a Delta distribution, then we know E[xxᵀ] = xxᵀ is a rank-1 matrix.
        We can take advantage of this to compute the quadratic form Σx and the outer product μ·x to compute the psipsi (φφ) term
        in the polyagamma lower bound faster than using a dense matrix multiplication
        """

        # x.shape = (sample_shape, batch_shape, d)
        # L.shape = (batch_shape, C, d, d)
        # we need both to be broadcastable to (sample_shape, batch_shape, C, d, d)

        C = L.shape[-self.stick_breaking_dim]
        d = x.shape[-1]

        x_transposed = jnp.moveaxis(x, -1, 0) # (d, ...)
        rhs   = jnp.broadcast_to(x_transposed, (C,) + x_transposed.shape)  # (C, d, ...)
        # if rhs has shape (C, d, N, M, ...) then we need to reshape it to (C, d, N*M*...)
        rhs = jnp.reshape(rhs, (C, d, -1))
        y     = jax.lax.linalg.triangular_solve(
                                     L, rhs, left_side=True, lower=True)             # (C, d, N*M*...)
        quad  = jnp.moveaxis((y ** 2).sum(1), 0, -1)                                 # (C, d, N*M*...) --> (C, N*M*...) --> (N*M*..., C)
        outer = (x[...,None] * mu.mT).sum(-2) ** 2                                   # (N, M, ..., C)
        return quad.reshape(x.shape[:-1] + (C,)) + outer                             # (N, M, ..., C)




