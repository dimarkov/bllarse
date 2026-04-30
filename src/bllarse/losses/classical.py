import equinox as eqx
import jax.numpy as jnp

from functools import partial
from typing import Optional
from jaxtyping import Array, PRNGKeyArray as PRNGKey
from optax import smooth_labels, safe_softmax_cross_entropy, l2_loss
from jax import vmap, nn, lax


class Classical(eqx.Module):
    num_classes: int

    def update(self, *args, **kwargs):
        """Classical losses have no learnable parameters, return unchanged."""
        return self


class MSE(Classical):

    def __init__(self, num_classes: int):
        self.beta = jnp.ones((), dtype=jnp.float32)
        super().__init__(num_classes)

    def __call__(self, logits: Array, y: Array, *, with_logits: bool = False, loss_type = None):
        _y = nn.one_hot(y, self.num_classes)
        logits = self.beta * logits

        if with_logits:
            return l2_loss(logits, _y), logits
        else:
            return l2_loss(logits, _y)

    def calibrate(self, logits: Array, y: Array, *, num_iters: int = 16) -> "MSE":
        """Temperature scaling via Newton steps on beta (= 1/T) using closed-form
        gradient and Hessian of L2 loss w.r.t. beta.
        """
        target_z = (y * logits).sum(-1)
        var_z = jnp.square(logits).sum(-1)

        def newton_step(b, _):
            g = (2 * b * var_z - target_z).mean()
            h = var_z.mean()
            return jnp.clip(b - g / (h + 1e-8), 0.01, 100.0), None

        beta_new, _ = lax.scan(newton_step, self.beta, jnp.arange(num_iters))
        return eqx.tree_at(lambda m: m.beta, self, beta_new)

class CrossEntropy(Classical):
    alpha: float
    beta: Array

    def __init__(self, alpha: float, num_classes: int):
        self.alpha = alpha
        self.beta = jnp.ones((), dtype=jnp.float32)
        super().__init__(num_classes)

    def __call__(self, logits: Array, y: Array, *, with_logits: bool = False, loss_type = None):
        _y = smooth_labels(nn.one_hot(y, self.num_classes), alpha=self.alpha)
        scaled = self.beta * logits

        if with_logits:
            return safe_softmax_cross_entropy(scaled, _y), scaled
        else:
            return safe_softmax_cross_entropy(scaled, _y)

    def calibrate(self, logits: Array, y: Array, *, num_iters: int = 16) -> "CrossEntropy":
        """Temperature scaling via Newton steps on beta (= 1/T) using closed-form
        gradient and Hessian of softmax cross-entropy w.r.t. beta.
        """
        _y = smooth_labels(nn.one_hot(y, self.num_classes), alpha=self.alpha)
        target_z = (_y * logits).sum(-1)

        def newton_step(b, _):
            p = nn.softmax(b * logits, axis=-1)
            E_z = (p * logits).sum(-1)
            E_z2 = (p * logits * logits).sum(-1)
            var_z = E_z2 - E_z * E_z
            g = (E_z - target_z).mean()
            h = var_z.mean()
            return jnp.clip(b - g / (h + 1e-8), 0.01, 100.0), None

        beta_new, _ = lax.scan(newton_step, self.beta, jnp.arange(num_iters))
        return eqx.tree_at(lambda m: m.beta, self, beta_new)
