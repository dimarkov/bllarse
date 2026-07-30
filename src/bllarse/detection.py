"""OOD detection metrics: AUROC, FPR@95TPR, AUPR.

Convention, applied everywhere in this project: **higher score = more OOD**.
The OOD set is the positive class. Callers pass score arrays already in that
orientation (see bllarse.metrics.msp_ood_score / entropy_ood_score); nothing
here inspects or flips the sign.

NumPy rather than JAX deliberately: this is post-hoc analysis over dumped
arrays, gets no GPU benefit, and tie handling reads more clearly.
"""

import numpy as np


def _validate(scores_id: np.ndarray, scores_ood: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores_id = np.asarray(scores_id, dtype=np.float64).ravel()
    scores_ood = np.asarray(scores_ood, dtype=np.float64).ravel()
    if scores_id.size == 0 or scores_ood.size == 0:
        raise ValueError("both score arrays must be non-empty")
    return scores_id, scores_ood


def auroc(scores_id, scores_ood) -> float:
    """Area under the ROC curve, via the Mann-Whitney U identity.

    AUROC = P(score_ood > score_id) + 0.5 * P(score_ood == score_id).
    Average ranks give tied scores half credit, which is why an all-ties input
    returns exactly 0.5 rather than 0 or 1.
    """
    scores_id, scores_ood = _validate(scores_id, scores_ood)
    n_id, n_ood = scores_id.size, scores_ood.size

    combined = np.concatenate([scores_ood, scores_id])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(combined.size, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1, dtype=np.float64)

    # Average the ranks within each group of tied values.
    sorted_scores = combined[order]
    start = 0
    for end in range(1, sorted_scores.size + 1):
        if end == sorted_scores.size or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end

    rank_sum_ood = ranks[:n_ood].sum()
    u = rank_sum_ood - n_ood * (n_ood + 1) / 2.0
    return float(u / (n_id * n_ood))


def fpr_at_tpr(scores_id, scores_ood, tpr_target: float = 0.95) -> float:
    """False positive rate on the ID set at the threshold achieving `tpr_target`.

    The threshold is the largest value that still captures at least
    `tpr_target` of the OOD set; FPR is the fraction of ID points at or above it.
    """
    scores_id, scores_ood = _validate(scores_id, scores_ood)
    if not 0.0 < tpr_target <= 1.0:
        raise ValueError("tpr_target must be in (0, 1]")

    # The threshold capturing >= tpr_target of OOD is the (1 - tpr_target)
    # quantile of the OOD scores, taken from below.
    threshold = np.quantile(scores_ood, 1.0 - tpr_target, method="lower")
    return float(np.mean(scores_id >= threshold))


def aupr(scores_id, scores_ood) -> float:
    """Average precision with OOD as the positive class.

    Step-interpolated (the `sum((R_n - R_{n-1}) * P_n)` form), not trapezoidal,
    which is the standard non-optimistic estimator.
    """
    scores_id, scores_ood = _validate(scores_id, scores_ood)
    scores = np.concatenate([scores_ood, scores_id])
    is_positive = np.concatenate(
        [np.ones(scores_ood.size, dtype=bool), np.zeros(scores_id.size, dtype=bool)]
    )

    order = np.argsort(-scores, kind="mergesort")
    is_positive = is_positive[order]

    tp = np.cumsum(is_positive)
    fp = np.cumsum(~is_positive)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / scores_ood.size

    recall_delta = np.diff(recall, prepend=0.0)
    return float(np.sum(recall_delta * precision))


def detection_metrics(scores_id, scores_ood) -> dict[str, float]:
    return {
        "auroc": auroc(scores_id, scores_ood),
        "fpr95": fpr_at_tpr(scores_id, scores_ood, 0.95),
        "aupr": aupr(scores_id, scores_ood),
    }
