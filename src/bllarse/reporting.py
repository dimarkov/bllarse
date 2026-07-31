"""Live per-dump-point reporting, shared by both probing arms.

Exists so the AdamW and VBLL arms compute and name their scalars identically --
a second ECE implementation in the PyTorch arm would be a silent comparability
bug, so both call `bllarse.metrics` and both go through this function.

None of this decides a published number. Every figure is rebuilt post hoc from
the dumps by `bllarse.analysis.curves`; this is observability, and it is
deliberately allowed to be lossy (MLflow may be down) in a way the dumps are
not.
"""

from typing import Any

from bllarse import metrics


def step_metrics(
    test_logits: Any, test_labels: Any, val_logits: Any, val_labels: Any
) -> dict[str, float]:
    """Scalars for one dumped step, on both splits.

    Val metrics are logged, not just used for the temperature fit, because the
    stopping rule is applied post hoc: `analysis.thresholds.early_stop` picks
    argmin val NLL over the dumped grid. Without val NLL on the curve there is
    nothing to take the argmin of.
    """
    out = {f"test_{k}": v for k, v in metrics.classification_metrics(test_logits, test_labels).items()}
    out.update({f"val_{k}": v for k, v in metrics.classification_metrics(val_logits, val_labels).items()})

    temperature = metrics.fit_temperature(val_logits, val_labels)
    out["temperature"] = temperature
    scaled = metrics.apply_temperature(test_logits, temperature)
    out.update({f"test_ts_{k}": v for k, v in metrics.classification_metrics(scaled, test_labels).items()})
    return out


def report_step(
    run: Any,
    step: int,
    samples_seen: int,
    epoch: int,
    test_logits: Any,
    test_labels: Any,
    val_logits: Any,
    val_labels: Any,
) -> dict[str, float]:
    """Compute the step's scalars, push them to MLflow, and return them.

    `samples_seen` is logged as a metric, not only as the dump's metadata, so
    MLflow can plot against it directly -- it is the axis that stays comparable
    when batch size changes, which step and epoch do not.
    """
    scalars = step_metrics(test_logits, test_labels, val_logits, val_labels)
    run.log_metrics({**scalars, "samples_seen": samples_seen, "epoch": epoch}, step=step)
    return scalars
