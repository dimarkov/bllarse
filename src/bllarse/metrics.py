"""Classification metrics and temperature scaling, on (logits, labels).

One implementation, used both in training loops and by post-hoc analysis over
dumped logits. Dumps are float16 on disk, so every entry point upcasts to
float32 first. The risk is precision, not overflow: jax.nn.softmax subtracts the
max internally so fp16 logits never overflow, but it also RETURNS fp16, and an
fp16 MSP over 5000 points collapses to a few hundred distinct values. AUROC
gives ties half credit, so that would distort the detection figures.

OOD score functions in this module follow the project-wide convention:
**higher means more OOD**.
"""

from typing import Any

import jax.numpy as jnp
from jax import lax, nn
from jaxtyping import Array
from tensorflow_probability.substrates.jax.stats import (
    expected_calibration_error as _tfp_ece,
)

DEFAULT_NUM_BINS = 20


def _f32(x: Any) -> Array:
    """Upcast to float32. Dumped logits are fp16; without this, softmax returns
    fp16 and the score functions lose the ranking resolution AUROC depends on."""
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


# Newton's iterate is clipped into this interval so a flat Hessian cannot throw
# beta to infinity. The bounds are a safety rail, not a prior: when one binds,
# the returned temperature is a boundary value rather than an optimum, and every
# temperature-scaled metric downstream is conditioned on a fit that did not
# converge. is_clipped_temperature exists so that is visible instead of silent.
BETA_BOUNDS = (0.01, 100.0)


def is_clipped_temperature(temperature: float, tol: float = 1e-6) -> bool:
    """Did `fit_temperature` return a clip boundary rather than an optimum?"""
    return any(
        abs(temperature - 1.0 / bound) < tol for bound in BETA_BOUNDS
    )


def fit_temperature(logits: Any, labels: Any, num_iters: int = 16) -> float:
    """Fit one scalar temperature T > 0 by minimising NLL of softmax(z / T).

    Newton on beta = 1/T with the closed-form gradient and Hessian of softmax
    cross-entropy. NLL is convex in beta, so this reaches the global optimum in a
    handful of steps -- no learning rate to tune, unlike a gradient fit.

    Returns T (the divisor), not beta. The result may be a BETA_BOUNDS boundary
    rather than an optimum -- check is_clipped_temperature.
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
        return jnp.clip(beta - grad / (hess + 1e-8), *BETA_BOUNDS), None

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


# --- Magnitude-based scores -------------------------------------------------
#
# msp/entropy read only the normalized predictive, so they discard the overall
# level of the logit vector: softmax(z + c) = softmax(z). That level is not
# noise. Writing log p(k|x) = z_k - logsumexp(z), the softmax keeps the
# discriminative term and throws away logsumexp(z), which for a cross-entropy
# classifier behaves like an unnormalized log-density (Grathwohl et al. 2019;
# Liu et al. 2020). Concretely for a linear probe, z_k = w_k . phi(x) + b_k, so
# the level tracks ||phi(x)|| and how well the features align with ANY class
# direction -- OOD inputs often produce a sharply peaked but small logit vector,
# which MSP scores as confident and these two score as OOD.
#
# Both are defined ONLY on unnormalized logits, which is why they are guarded.


def _reject_normalized(logits: Array, score_name: str) -> None:
    """Refuse an already-normalized array. Fails loudly rather than plausibly.

    AdamW+CE and VBLL both dump under logit_kind="softmax_logits", but they are
    not the same kind of array: CE writes raw pre-softmax logits, while VBLL
    writes log(predictive) -- already normalized, because VBLL's predictive is a
    Monte Carlo average of softmax samples and has no single logit vector behind
    it. On a normalized array logsumexp is identically 0, so `energy_ood_score`
    is constant (AUROC exactly 0.5) and `max_logit_ood_score` collapses to a
    monotone transform of MSP (AUROC identical to it). Both return a
    figure-shaped number instead of an error, so this is checked, not documented.
    """
    level = nn.logsumexp(logits, axis=-1)
    if float(jnp.max(jnp.abs(level))) < 1e-3:
        raise ValueError(
            f"{score_name} needs unnormalized logits, but logsumexp is 0 to "
            f"within 1e-3 on every row -- this array is already a normalized "
            f"log-predictive (e.g. the VBLL arm's dump). {score_name} is "
            f"degenerate on it; use msp_ood_score or entropy_ood_score, which "
            f"are well defined for every arm."
        )


def energy_ood_score(logits: Any) -> Array:
    """Negative free energy, -logsumexp(z). Higher = more OOD.

    Liu et al. (2020). Unnormalized logits only -- see `_reject_normalized`.
    """
    logits = _f32(logits)
    _reject_normalized(logits, "energy_ood_score")
    return -nn.logsumexp(logits, axis=-1)


def max_logit_ood_score(logits: Any) -> Array:
    """Negated maximum logit. Higher = more OOD.

    Hendrycks et al. (2022). Unnormalized logits only -- see `_reject_normalized`.
    """
    logits = _f32(logits)
    _reject_normalized(logits, "max_logit_ood_score")
    return -jnp.max(logits, axis=-1)
