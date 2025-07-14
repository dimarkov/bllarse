import jax.numpy as jnp
import jax.random as jr
from jax import nn
from bllarse.losses import IBProbit, MultinomialPolyaGamma

def test_independent_binary_probit_loss():
    key = jr.PRNGKey(0)
    D, C, N = 10, 5, 1_000

    key, W_key, b_key, x_key, noise_key = jr.split(key, 5)
    W = jr.normal(W_key, (D, C))
    b = jr.normal(b_key, (C,))
    x = jr.normal(x_key, (N, D))
    
    logits = x @ W + b
    probs = nn.softmax(logits)
    y = jr.categorical(noise_key, logits)

    key, model_key = jr.split(key)
    bmp = IBProbit(input_dim=D, num_classes=C, key=model_key, use_bias=True)
    bmp = bmp.update(x, y, num_iters=64)

    W_recovered, b_recovered = bmp.params

    assert jnp.allclose(W, W_recovered, atol=10.0)  # try to improve atol
    assert jnp.allclose(b, b_recovered, atol=10.0)  # try to imporve atol
    
    for loss_type in range(4):
        loss, logits = bmp(x, y, with_logits=True, loss_type=loss_type)
        print(loss_type, loss.mean())
    
    acc = jnp.mean(logits.argmax(-1) == y)
    print('acc = ', acc)

def test_independent_multinomial_polya_gamma_loss():
    key = jr.PRNGKey(0)
    D, C, N = 10, 5, 1_000

    key, W_key, b_key, x_key, noise_key = jr.split(key, 5)
    W = jr.normal(W_key, (D, C))
    b = jr.normal(b_key, (C,))
    x = jr.normal(x_key, (N, D))
    
    logits = x @ W + b
    probs = nn.softmax(logits)
    y = jr.categorical(noise_key, logits)

    key, model_key = jr.split(key)
    mnpg = MultinomialPolyaGamma(input_dim=D, num_classes=C, use_bias=True, key=model_key)

    mnpg = mnpg.update(x, y, num_iters=64)
    
    for loss_type in range(2):
        loss, logits = mnpg(x, y, with_logits=True, loss_type=loss_type)
        print(loss.mean())

        print(jnp.mean(logits.argmax(-1) == y))


test_independent_binary_probit_loss()
test_independent_multinomial_polya_gamma_loss()
