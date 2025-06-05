import equinox as eqx

from typing import Optional
from jaxtyping import Array, PRNGKeyArray as PRNGKey

identity = eqx.nn.Identity()

class LastLayer(eqx.Module):
    last_layer: eqx.Module

    def __init__(self, last_layer):
        self.last_layer = last_layer

    def __call__(self, input_nnet: eqx.Module, x: Array, *, key: Optional[PRNGKey] = None):

        # remove final layer from input neural network
        headless_nnet = eqx.tree_at(lambda m: m.fc, input_nnet, identity)

        if key is None:
            x = headless_nnet(x)
            return self.last_layer(x)
        else:
            key1, key2 = jr.split(key)
            x = headless_nnet(x, key=key1)
            return self.last_layer(x, key=key2)
