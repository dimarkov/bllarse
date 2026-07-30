"""Classification metrics and temperature scaling, on (logits, labels).

One implementation, used both in training loops and by post-hoc analysis over
dumped logits. Dumps are float16 on disk, so every entry point upcasts to
float32 before any exponentiation -- softmax of a raw fp16 logit overflows
around |z| > 11.

OOD score functions in this module follow the project-wide convention:
**higher means more OOD**.
"""

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax, nn
from jaxtyping import Array
from tensorflow_probability.substrates.jax.stats import (
    expected_calibration_error as _tfp_ece,
)

DEFAULT_NUM_BINS = 20


def _f32(x: Any) -> Array:
    """Upcast to float32. Dumped logits are fp16 and would overflow in softmax."""
    return jnp.asarray(x, dtype=jnp.float32)


def accuracy(logits: Any, labels: Any) -> float:
    logits = _f32(logits)
    return float(jnp.mean(jnp.argmax(logits, axis=-1) == jnp.asarray(labels)))


def ece(logits: Any, labels: Any, num_bins: int = DEFAULT_NUM_BINS) -> float:
    logits = _f32(logits)
    labels = jnp.asarray(labels)
    predictions = jnp.argmax(logits, axis=-1)
    return float(
        _tfp_ece(num_bins, logits=logits, labels_true=labels, labels_predicted=predictions)
    )


def nll(logits: Any, labels: Any) -> float:
    logits = _f32(logits)
    labels = jnp.asarray(labels)
    log_probs = nn.log_softmax(logits, axis=-1)
    return float(-jnp.mean(jnp.take_along_axis(log_probs, labels[:, None], axis=-1)))


def brier(logits: Any, labels: Any) -> float:
    """Multiclass Brier over the FULL predictive vector, in [0, 2].

    Not the top-label variant -- results-plan.md 0.1 is explicit about which.
    """
    logits = _f32(logits)
    p = nn.softmax(logits, axis=-1)
    onehot = nn.one_hot(jnp.asarray(labels), p.shape[-1])
    return float(jnp.mean(jnp.sum((p - onehot) ** 2, axis=-1)))


def classification_metrics(
    logits: Any, labels: Any, num_bins: int = DEFAULT_NUM_BINS
) -> dict[str, float]:
    return {
        "acc": accuracy(logits, labels),
        "ece": ece(logits, labels, num_bins),
        "nll": nll(logits, labels),
        "brier": brier(logits, labels),
    }


def fit_temperature(logits: Any, labels: Any, num_iters: int = 16) -> float:
    """Fit one scalar temperature T > 0 by minimising NLL of softmax(z / T).

    Newton on beta = 1/T with the closed-form gradient and Hessian of softmax
    cross-entropy. NLL is convex in beta, so this reaches the global optimum in a
    handful of steps -- no learning rate to tune, unlike a gradient fit.

    Returns T (the divisor), not beta.
    """
    logits = _f32(logits)
    labels = jnp.asarray(labels)
    target_z = jnp.take_along_axis(logits, labels[:, None], axis=-1).squeeze(-1)

    def newton_step(beta, _):
        p = nn.softmax(beta * logits, axis=-1)
        e_z = jnp.sum(p * logits, axis=-1)
        e_z2 = jnp.sum(p * logits * logits, axis=-1)
        grad = jnp.mean(e_z - target_z)
        hess = jnp.mean(e_z2 - e_z * e_z)
        return jnp.clip(beta - grad / (hess + 1e-8), 0.01, 100.0), None

    beta, _ = lax.scan(newton_step, jnp.ones(()), jnp.arange(num_iters))
    return float(1.0 / beta)


def apply_temperature(logits: Any, temperature: float) -> Array:
    return _f32(logits) / temperature


def msp_ood_score(logits: Any) -> Array:
    """1 - max softmax probability. Higher = more OOD."""
    return 1.0 - jnp.max(nn.softmax(_f32(logits), axis=-1), axis=-1)


def entropy_ood_score(logits: Any) -> Array:
    """Predictive entropy in nats. Higher = more OOD. Maximum is log(K)."""
    logits = _f32(logits)
    log_p = nn.log_softmax(logits, axis=-1)
    return -jnp.sum(jnp.exp(log_p) * log_p, axis=-1)
