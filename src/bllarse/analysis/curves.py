"""Assemble metric curves from eval dumps. Pure post-hoc -- no training here.

This is where temperature scaling actually happens. Standard TS fits one T at
convergence, but Figs 5-6 are trajectories, and a once-fit T would be a strawman
exactly on the early part of the curve where the sample-efficiency claim lives.
IBProbit recalibrates itself continuously, so the baseline is allowed to as
well: T is refit on the pinned val split at every dumped step.

That is non-standard and makes the baseline strictly stronger, which is the
direction a fairness deviation should run. It needs saying in the caption.
"""

from typing import Any, Callable

from bllarse import detection, evaldump, metrics

# `metrics.classification_metrics` softmaxes the dumped array unconditionally.
# That is correct for "softmax_logits" (AdamW+CE and VBLL), but a future arm
# dumping a differently-scaled score under a different logit_kind -- e.g.
# IBProbit's "moderated_score" (q-update-spec.md 5.8) -- would be silently
# mis-metricised rather than rejected.
_METRICISABLE_LOGIT_KINDS = frozenset({"softmax_logits"})


def _by_split(root, group_id: str, uid: str) -> dict[str, dict[int, Any]]:
    """Index a run's dumps as {split: {step: record}}."""
    indexed: dict[str, dict[int, Any]] = {}
    for record in evaldump.iter_dumps(root, group_id, uid):
        indexed.setdefault(record.meta["split"], {})[record.meta["step"]] = record
    return indexed


def run_curve(
    root, group_id: str, uid: str, *, temperature_scaled: bool = False
) -> list[dict[str, Any]]:
    """One row per dumped step, on the test split.

    With `temperature_scaled=True`, T is fit on that step's val dump and applied
    to that step's test logits -- a fresh fit per point, never a reused one.

    Each row carries `seed` and `arm` from the dump's own meta, so a caller
    grouping runs by seed (e.g. for `thresholds.crossings_per_seed`) can read
    them straight off the row instead of re-parsing the uid string.
    """
    indexed = _by_split(root, group_id, uid)
    if "test" not in indexed:
        raise ValueError(f"no test-split dumps found for {group_id}/{uid}")

    rows = []
    for step in sorted(indexed["test"]):
        test_record = indexed["test"][step]
        logit_kind = test_record.meta["logit_kind"]
        if logit_kind not in _METRICISABLE_LOGIT_KINDS:
            raise ValueError(
                f"{group_id}/{uid} step {step}: run_curve softmaxes the dumped "
                f"array and only knows how to do that for "
                f"{sorted(_METRICISABLE_LOGIT_KINDS)}, but found "
                f"logit_kind={logit_kind!r}"
            )
        logits, labels = test_record.logits, test_record.labels
        temperature = 1.0

        if temperature_scaled:
            val_record = indexed.get("val", {}).get(step)
            if val_record is None:
                raise ValueError(
                    f"temperature scaling needs a val dump at step {step} for "
                    f"{group_id}/{uid}, but none was written"
                )
            temperature = metrics.fit_temperature(val_record.logits, val_record.labels)
            logits = metrics.apply_temperature(logits, temperature)

        row = {
            "step": step,
            "samples_seen": test_record.meta["samples_seen"],
            "seed": test_record.meta["seed"],
            "arm": test_record.meta["arm"],
            "temperature": temperature,
            **metrics.classification_metrics(logits, labels),
        }

        # Val metrics ride along so the stopping rule can be applied post hoc:
        # `thresholds.early_stop` takes an argmin over val_nll, and without it
        # on the row there is nothing to take the argmin of. Scaled by the same
        # T as the test side when temperature_scaled, so the two are comparable.
        val_record = indexed.get("val", {}).get(step)
        if val_record is not None:
            val_logits = val_record.logits
            if temperature_scaled:
                val_logits = metrics.apply_temperature(val_logits, temperature)
            row.update({
                f"val_{k}": v
                for k, v in metrics.classification_metrics(
                    val_logits, val_record.labels
                ).items()
            })
        rows.append(row)
    return rows


def run_detection(
    root, group_id: str, uid: str, score_fn: Callable[[Any], Any]
) -> dict[str, float]:
    """Detection metrics at the final dumped point.

    `score_fn` maps logits to a score under the project convention that higher
    means more OOD -- metrics.msp_ood_score or metrics.entropy_ood_score. Passing
    the same function to both sides is what makes the comparison a controlled
    one; a per-arm score would confound model with score function.
    """
    indexed = _by_split(root, group_id, uid)
    if "ood" not in indexed:
        raise ValueError(f"no ood-split dump found for {group_id}/{uid}")
    if "test" not in indexed:
        raise ValueError(f"no test-split dump found for {group_id}/{uid}")

    id_step, ood_step = max(indexed["test"]), max(indexed["ood"])
    if id_step != ood_step:
        # Pairing scores from two different checkpoints would produce a
        # plausible-looking AUROC that means nothing. A partial write or a
        # changed eval schedule is the likely cause.
        raise ValueError(
            f"{group_id}/{uid}: final test dump is at step {id_step} but the "
            f"final ood dump is at step {ood_step}; detection would compare "
            f"different checkpoints"
        )

    return detection.detection_metrics(
        score_fn(indexed["test"][id_step].logits),
        score_fn(indexed["ood"][ood_step].logits),
    )
