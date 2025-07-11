import jax.numpy as jnp
import jax.random as jr
from jax import nn
from bllarse.losses.bayesian import BayesianMultinomialProbit

def test_bayesian_multinomial_probit():
    key = jr.PRNGKey(0)
    D, C, N = 10, 3, 10_000

    key, W_key, b_key, x_key, noise_key = jr.split(key, 5)
    W = jr.normal(W_key, (D, C))
    b = jr.normal(b_key, (C,))
    x = jr.normal(x_key, (N, D))
    
    logits = x @ W + b
    probs = nn.softmax(logits)
    y = jr.categorical(noise_key, logits)
    
    for loss_type in range(4):
        key, model_key = jr.split(key)
        bmp = BayesianMultinomialProbit(model_key, embed_dim=D, num_classes=C, use_bias=True, loss_type=loss_type)

        loss, logits, new_bmp = bmp(x, y, with_logits=True)

        print(jnp.mean(logits.argmax(-1) == y))

        params = new_bmp.Sigma @ new_bmp.eta
        W_recovered = params[:-1]
        b_recovered = params[-1]

        assert jnp.allclose(W, W_recovered, atol=1.0)
        assert jnp.allclose(b, b_recovered, atol=1.0)


test_bayesian_multinomial_probit()
