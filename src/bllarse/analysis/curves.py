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
            # A boundary T is a clip, not an optimum, so the scaled metrics on
            # this row rest on a fit that did not converge. Carried onto the row
            # so a figure built from these curves can exclude or flag them
            # rather than averaging them in silently. False when
            # temperature_scaled is off, where T is the identity by definition.
            "temperature_clipped": (
                temperature_scaled and metrics.is_clipped_temperature(temperature)
            ),
            # None on dumps written before this was recorded; `early_stop`
            # falls back to inferring it from the curve there. See the note in
            # evaldump.write_dump for why the inference cannot be trusted on a
            # run that writes only its argmin.
            "at_last_eval": test_record.meta.get("at_last_eval"),
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


def _id_ood_pair(
    root, group_id: str, uid: str, step: int | None = None, ood_split: str = "ood",
) -> tuple[Any, Any]:
    """The (test, `ood_split`) dump records at `step`, or the final step if None.

    `step` exists because the endpoint is not automatically the right place to
    score. These runs carry no early stopping, and an arm that overfits is a
    different model at the last step than at the one its hyperparameters were
    selected at -- so a caller comparing arms should be able to ask for the
    val-optimal step instead. Runs written before the OOD dump moved into the
    eval loop have only the final step available, and asking for any other
    raises rather than silently falling back to it.

    `ood_split` exists because not every writer uses the plain "ood" split
    name: scripts/finetune/finetune_jax.py writes one dump per OOD set as
    split=f"ood_{name}" (e.g. "ood_imagenet_r"), so those dumps are invisible
    under the default. probe.py and probing_vbll.py both write a single
    split="ood" dump, which the default keeps finding unchanged.
    """
    indexed = _by_split(root, group_id, uid)
    if ood_split not in indexed:
        raise ValueError(f"no {ood_split}-split dump found for {group_id}/{uid}")
    if "test" not in indexed:
        raise ValueError(f"no test-split dump found for {group_id}/{uid}")

    if step is None:
        id_step, ood_step = max(indexed["test"]), max(indexed[ood_split])
        if id_step != ood_step:
            # Pairing scores from two different checkpoints would produce a
            # plausible-looking AUROC that means nothing. A partial write or a
            # changed eval schedule is the likely cause.
            raise ValueError(
                f"{group_id}/{uid}: final test dump is at step {id_step} but the "
                f"final {ood_split} dump is at step {ood_step}; detection would "
                f"compare different checkpoints"
            )
        return indexed["test"][id_step], indexed[ood_split][ood_step]

    for split in ("test", ood_split):
        if step not in indexed[split]:
            available = sorted(indexed[split])
            raise ValueError(
                f"{group_id}/{uid}: no {split} dump at step {step}. Available: "
                f"{available[:3]}...{available[-1] if available else '-'} "
                f"({len(available)} steps). A run written before the OOD dump "
                f"moved into the eval loop has only its final step."
            )
    return indexed["test"][step], indexed[ood_split][step]


def run_detection(
    root, group_id: str, uid: str, score_fn: Callable[[Any], Any],
    step: int | None = None, ood_split: str = "ood",
) -> dict[str, float]:
    """Detection metrics from a shared score function, at `step` (default: last).

    `score_fn` maps logits to a score under the project convention that higher
    means more OOD -- metrics.msp_ood_score or metrics.entropy_ood_score. Passing
    the same function to both sides is what makes the comparison a controlled
    one; a per-arm score would confound model with score function.

    `ood_split` selects which OOD dump to pair against test -- see
    `_id_ood_pair`; the default "ood" matches probe.py / probing_vbll.py.

    This is Fig 2's PRIMARY comparison. `run_native_detection` answers the
    different question of whether each method's own epistemic quantity works.
    """
    id_record, ood_record = _id_ood_pair(root, group_id, uid, step, ood_split=ood_split)
    return detection.detection_metrics(
        score_fn(id_record.logits), score_fn(ood_record.logits)
    )


def run_native_detection(
    root, group_id: str, uid: str, step: int | None = None, ood_split: str = "ood",
) -> dict[str, float]:
    """Detection metrics from the arm's OWN dumped epistemic score.

    Not a substitute for `run_detection` -- it deliberately breaks the
    controlled comparison, because the two answer different questions. A shared
    score function asks "which model orders its confidence better"; this asks
    "does the arm's own epistemic quantity detect OOD", which is the claim
    section 3.7 actually makes when it names the epistemic term as the
    mechanism. Report both, labelled.

    The score is read off the dump rather than recomputed: it is the arm's
    native quantity (VBLL's predictive variance, IBProbit's epistemic term) and
    only the training script can produce it. `ood_split` -- see `_id_ood_pair`.
    """
    id_record, ood_record = _id_ood_pair(root, group_id, uid, step, ood_split=ood_split)
    for record, side in ((id_record, "test"), (ood_record, ood_split)):
        if record.score is None:
            raise ValueError(
                f"{group_id}/{uid}: the {side} dump carries no native score "
                f"(score_kind is unset). AdamW+CE has no such quantity by "
                f"construction -- use run_detection with a shared score "
                f"function for that arm."
            )
    if id_record.meta["score_kind"] != ood_record.meta["score_kind"]:
        raise ValueError(
            f"{group_id}/{uid}: score_kind differs between the test dump "
            f"({id_record.meta['score_kind']!r}) and the ood dump "
            f"({ood_record.meta['score_kind']!r}); these are not comparable."
        )
    return detection.detection_metrics(id_record.score, ood_record.score)
