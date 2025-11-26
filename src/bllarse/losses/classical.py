import equinox as eqx

from functools import partial
from typing import Optional
from jaxtyping import Array, PRNGKeyArray as PRNGKey
from optax import smooth_labels, safe_softmax_cross_entropy, l2_loss
from jax import vmap, nn


class Classical(eqx.Module):
    num_classes: int

    def update(self, *args, **kwargs):
        """Classical losses have no learnable parameters, return unchanged."""
        return self


class MSE(Classical):

    def __call__(self, logits: Array, y: Array, *, with_logits: bool = False, loss_type = None):
        _y = nn.one_hot(y, self.num_classes)
        
        if with_logits:
            return l2_loss(logits, _y), logits
        else:
            return l2_loss(logits, _y)

class CrossEntropy(Classical):
    alpha: float

    def __init__(self, alpha: float, num_classes: int):
        self.alpha = alpha
        super().__init__(num_classes)

    def __call__(self, logits: Array, y: Array, *, with_logits: bool = False, loss_type = None):
        _y = smooth_labels(nn.one_hot(y, self.num_classes), alpha=self.alpha)

        if with_logits:
            return safe_softmax_cross_entropy(logits, _y), logits
        else:
            return safe_softmax_cross_entropy(logits, _y)
