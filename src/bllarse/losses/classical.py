import equinox as eqx

from functools import partial
from typing import Optional
from jaxtyping import Array, PRNGKeyArray as PRNGKey
from optax import smooth_labels, safe_softmax_cross_entropy, l2_loss
from jax import vmap, nn


class MSE(eqx.Module):
    num_classes: int

    def __call__(self, params: eqx.Module, static:eqx.Module, x: Array, y: Array, *, key: Optional[PRNGKey] = None, with_logits: bool = False):
        model = eqx.combine(params, static)
        pred = vmap(partial(model, key=key))
        _y = nn.one_hot(y, self.num_classes)
        if with_logits:
            return l2_loss(pred, y).mean(), pred
        else:
            return l2_loss(pred, y).mean()

class CrossEntropy(eqx.Module):
    alpha: float
    num_classes: int

    def __init__(self, alpha: float, num_classes: int):
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, model: eqx.Module, x: Array, y: Array, *, key: Optional[PRNGKey] = None, with_logits: bool = False):
        logits = vmap(partial(model, key=key))(x)
        _y = smooth_labels(nn.one_hot(y, self.num_classes), alpha=self.alpha)

        if with_logits:
            return safe_softmax_cross_entropy(logits, _y).mean(), logits
        else:
            return safe_softmax_cross_entropy(logits, _y).mean()


