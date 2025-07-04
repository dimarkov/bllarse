import equinox as eqx

from functools import partial
from typing import Optional
from jaxtyping import Array, PRNGKeyArray as PRNGKey
from optax import smooth_labels, safe_softmax_cross_entropy, l2_loss
from jax import vmap, nn


class ELBO_multinomial_pg(eqx.Module):
    num_classes: int

    def __call__(self, params: eqx.Module, static:eqx.Module, x: Array, y: Array, *, key: Optional[PRNGKey] = None, with_logits: bool = False):
        model = eqx.combine(params, static)
        log_py = vmap(partial(model, key=key))
        py = nn.softmax(log_py, axis=-1)

        """ compute VFE """

        ## compute negative energy term or expected log likelihood E_{q(sb_weights)q(\omega)q(x)}[\log p(y|sb_weights, \omega, x)]

        ell = 0.0

        ## compute KL divergence terms

        # 1. compute KL(q(sb_weights) || p(sb_weights))

        kl_sb_weights = 0.0

        # 2. compute KL(q(\omega) || p(\omega))       

        kl_omega = 0.0

        return -ell + kl_sb_weights + kl_omega