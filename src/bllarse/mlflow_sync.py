"""Push scored results onto the runs that produced them, and read them back.

The campaign's dumps live on Raven; the tracking server has the runs but knows
almost nothing about how they turned out. This closes that gap in the direction
that is actually possible: every child run gets the numbers the analysis scored
it at, so the server becomes self-describing and figures can be built from it
alone.

WHAT CANNOT BE RECOVERED, and is not attempted here: the per-step metric
history. `report_step` computed those scalars transiently during training and
they were never written to disk -- the dumps hold one evaluation per run, the
val_nll argmin. Whole (arm, dataset) cells logged nothing at all (see the
geometric-grid bug fixed in 8ec351f), and that data is simply gone. Backfilling
a trajectory would mean inventing one.

So what lands on each run is a SCORED SNAPSHOT, prefixed `scored_` and tagged
with its provenance, deliberately distinct from whatever the training loop
logged live under bare names. Nothing overwrites a live metric.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Iterator, Mapping

# Metrics are floats and stay numerically queryable in the UI; tags carry the
# categorical facts. Params are deliberately unused: MLflow refuses to change a
# param once set, so a re-sync after a corrected analysis would fail on every
# run it had already touched.
METRIC_PREFIX = "scored_"
TAG_PREFIX = "bllarse."

# Row keys that identify the run rather than describe its result. They are not
# synced as metrics -- `arm`/`uid` are already tags, and `model`/`dataset`/`seed`
# are recoverable from the uid, so writing them again would be a second copy
# free to disagree with the first.
_IDENTITY_KEYS = frozenset({"arm", "uid", "model", "dataset", "seed", "id", "ood"})


def flatten_row(row: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, str]]:
    """Split one analysis row into (numeric metrics, categorical tags).

    Fig 2 rows nest one level -- `row["msp"]["auroc"]` -- so those flatten to
    `msp__auroc`. The separator is doubled because single underscores already
    appear inside score and metric names (`max_logit`, `at_best_nll`), and a
    single-underscore join could not be undone.
    """
    metrics: dict[str, float] = {}
    tags: dict[str, str] = {}

    for key, value in row.items():
        if key in _IDENTITY_KEYS:
            continue
        if value is None:
            # Unknown is not zero. A dump too old to carry `evals_run` must not
            # come back as "0 evaluations ran".
            continue
        if isinstance(value, bool):
            tags[key] = "true" if value else "false"
        elif isinstance(value, (int, float)):
            metrics[key] = float(value)
        elif isinstance(value, Mapping):
            for sub, sub_value in value.items():
                if isinstance(sub_value, bool):
                    tags[f"{key}__{sub}"] = "true" if sub_value else "false"
                elif isinstance(sub_value, (int, float)):
                    metrics[f"{key}__{sub}"] = float(sub_value)
                elif isinstance(sub_value, str):
                    tags[f"{key}__{sub}"] = sub_value[:500]
        elif isinstance(value, str):
            tags[key] = value[:500]

    return metrics, tags


def unflatten_row(metrics: Mapping[str, float], tags: Mapping[str, str]) -> dict[str, Any]:
    """Rebuild an analysis row from what `flatten_row` wrote.

    The inverse must be exact, or a figure built from the server silently
    differs from the same figure built from the JSON -- which is the one failure
    this whole path must not have.
    """
    row: dict[str, Any] = {}

    def _assign(key: str, value: Any) -> None:
        if "__" in key:
            outer, inner = key.split("__", 1)
            row.setdefault(outer, {})[inner] = value
        else:
            row[key] = value

    for key, value in metrics.items():
        _assign(key, value)
    for key, value in tags.items():
        if value in ("true", "false"):
            _assign(key, value == "true")
        else:
            _assign(key, value)
    return row


def batch_payload(
    metrics: Mapping[str, float], tags: Mapping[str, str], *,
    step: int, timestamp_ms: int,
) -> tuple[list, list]:
    """One run's writes as MLflow entities, for a single `log_batch` call.

    Split out so the entity construction is testable without a server, and
    because the alternative is what this replaced: `log_metric` / `set_tag`
    per key, which is one HTTP round trip each. At ~26 keys across 600 runs
    that is ~15,000 requests, and the first attempt at this sync was killed by
    a 30-minute timeout two thirds of the way through. Batched it is one
    request per run.
    """
    from mlflow.entities import Metric, RunTag

    return (
        [Metric(key, value, timestamp_ms, step) for key, value in metrics.items()],
        [RunTag(key, value) for key, value in tags.items()],
    )


def sync_rows(
    client: Any,
    experiment_id: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    group_ids: Mapping[str, str],
    provenance: str,
    excluded: Mapping[tuple[str, str], str] | None = None,
    dry_run: bool = True,
) -> list[str]:
    """Write each row onto the run with the matching (group_id, uid).

    Returns a human-readable log of what was done, or would be done.

    `excluded` records runs the analysis DROPPED -- they get a tag saying so and
    no scored metrics. That is the point of syncing them at all: a crashed run
    whose dump looks normal is exactly what nothing downstream could detect, so
    the server should carry the verdict rather than leaving the next reader to
    rediscover it.
    """
    import time

    # One stamp for the whole sync: these are post-hoc scores, not a series
    # of observations, so spreading them over the wall-clock of the sync
    # would invent a chronology they do not have.
    timestamp_ms = int(time.time() * 1000)
    log: list[str] = []
    by_uid: dict[tuple[str, str], str] = {}

    for arm, group_id in group_ids.items():
        for run in client.search_runs(
            [experiment_id], filter_string=f"tags.group_id = '{group_id}'",
            max_results=2000,
        ):
            uid = run.data.tags.get("uid")
            if uid:
                by_uid[(arm, uid)] = run.info.run_id

    excluded = dict(excluded or {})

    for row in rows:
        key = (row["arm"], row["uid"])
        if key in excluded:
            # The exclusion list wins over a row that scored the same run. An
            # analysis JSON produced BEFORE the exclusions existed still carries
            # rows for the crashed runs, and syncing those would leave the
            # server holding scored metrics for a run it also marks excluded --
            # two answers to one question, with nothing to say which is current.
            log.append(f"skip     {key[0]}/{key[1]}: excluded, not scored")
            continue
        run_id = by_uid.get(key)
        if run_id is None:
            log.append(f"MISSING  {key[0]}/{key[1]}: scored, but no MLflow run")
            continue
        metrics, tags = flatten_row(row)
        log.append(f"sync     {key[0]}/{key[1]}: {len(metrics)} metrics, {len(tags)} tags")
        if dry_run:
            continue
        payload_metrics, payload_tags = batch_payload(
            {f"{METRIC_PREFIX}{k}": v for k, v in metrics.items()},
            {**{f"{TAG_PREFIX}{k}": v for k, v in tags.items()},
             f"{TAG_PREFIX}scored_by": provenance},
            step=int(row.get("step") or 0),
            timestamp_ms=timestamp_ms,
        )
        client.log_batch(run_id, metrics=payload_metrics, tags=payload_tags)

    for (arm, uid), reason in excluded.items():
        run_id = by_uid.get((arm, uid))
        if run_id is None:
            log.append(f"MISSING  {arm}/{uid}: excluded, but no MLflow run")
            continue
        log.append(f"exclude  {arm}/{uid}: {reason}")
        if not dry_run:
            _, payload_tags = batch_payload(
                {},
                {f"{TAG_PREFIX}excluded": reason,
                 f"{TAG_PREFIX}scored_by": provenance},
                step=0, timestamp_ms=timestamp_ms,
            )
            client.log_batch(run_id, metrics=[], tags=payload_tags)

    return log


def fetch_rows(
    client: Any, experiment_id: str, group_ids: Mapping[str, str], uid_re: re.Pattern
) -> Iterator[dict[str, Any]]:
    """Rebuild analysis rows from synced runs -- the inverse of `sync_rows`.

    Runs tagged excluded are skipped, and so are runs never synced. A run with
    no scored metrics is not a run that scored zero; it is a run this has no
    information about, and letting it through as a row of empty values would
    quietly shrink every cell it belongs to.
    """
    for arm, group_id in group_ids.items():
        for run in client.search_runs(
            [experiment_id], filter_string=f"tags.group_id = '{group_id}'",
            max_results=2000,
        ):
            uid = run.data.tags.get("uid")
            if not uid or run.data.tags.get(f"{TAG_PREFIX}excluded"):
                continue
            metrics = {
                k[len(METRIC_PREFIX):]: v
                for k, v in run.data.metrics.items()
                if k.startswith(METRIC_PREFIX)
            }
            if not metrics:
                continue
            tags = {
                k[len(TAG_PREFIX):]: v
                for k, v in run.data.tags.items()
                if k.startswith(TAG_PREFIX) and k != f"{TAG_PREFIX}scored_by"
            }
            match = uid_re.match(uid)
            if not match:
                continue
            yield {
                "arm": arm, "uid": uid, **match.groupdict(),
                **_drop_inapplicable(unflatten_row(metrics, tags), arm),
            }


def _drop_inapplicable(row: dict, arm: str) -> dict:
    """Strip values this arm must not report, whatever the server holds.

    Only temperature scaling so far, and it is not hypothetical bookkeeping: an
    earlier sync wrote `scored_ts_*` and `scored_temperature` onto every VBLL and
    IBProbit run, before the analysis was changed to compute those for AdamW+CE
    alone. MLflow's API has no way to delete a metric, so those values are on the
    server permanently and would otherwise flow straight back into any figure
    built from it -- making the server route disagree with the JSON route, which
    is the one failure this path must not have.
    """
    from bllarse.analysis.curves import applies_temperature_scaling

    if applies_temperature_scaling(arm):
        return row
    return {
        k: v for k, v in row.items()
        if not k.startswith("ts_") and not k.startswith("temperature")
    }
