"""Scored results on the runs, and figures built back out of them.

The property that matters is the ROUND TRIP: a figure built from the tracking
server must be the same figure built from the JSON. If those diverge, nothing
looks broken -- both render perfectly and one of them is wrong.

Network-free; the client is a stub that records what it was told.
"""

import json
import re
from pathlib import Path

import pytest

from bllarse import mlflow_sync


class _StubRun:
    def __init__(self, run_id, tags, metrics=None):
        self.info = type("info", (), {"run_id": run_id})()
        self.data = type("data", (), {"tags": dict(tags), "metrics": dict(metrics or {})})()


class _StubClient:
    """Records writes; serves runs back with those writes applied."""

    def __init__(self, runs):
        self._runs = runs
        self.metrics, self.tags = [], []
        self.calls = 0

    def search_runs(self, _ids, filter_string, max_results=None):
        group = filter_string.split("'")[1]
        return [r for r in self._runs if r.data.tags.get("group_id") == group]

    def log_batch(self, run_id, metrics=(), tags=()):
        """One call per run -- the real client's batching API.

        Deliberately does NOT implement log_metric/set_tag: if sync_rows ever
        falls back to per-key writes, these tests should fail loudly rather
        than quietly pass while the real sync takes 15,000 round trips.
        """
        self.calls += 1
        for m in metrics:
            self.metrics.append((run_id, m.key, m.value, m.step))
            for r in self._runs:
                if r.info.run_id == run_id:
                    r.data.metrics[m.key] = m.value
        for t in tags:
            self.tags.append((run_id, t.key, t.value))
            for r in self._runs:
                if r.info.run_id == run_id:
                    r.data.tags[t.key] = t.value


UID_RE = re.compile(r"^(?P<model>dinov3_[a-z0-9]+)_(?P<dataset>.+)_seed(?P<seed>\d+)$")
GROUPS = {"adamw_ce": "probe_adamw_ce_fig1"}


def _row(uid="dinov3_big_dtd_seed137", **over):
    row = {
        "arm": "adamw_ce", "uid": uid, "model": "dinov3_big",
        "dataset": "dtd", "seed": "137",
        "step": 1088, "n_dumps": 1, "acc": 0.8439, "ece": 0.0213,
        "nll": 0.5149, "brier": 0.2661, "ts_acc": 0.8439, "ts_nll": 0.5102,
        "temperature": 1.0213, "temperature_clipped": 0.0,
        "at_last_eval": False, "truncated": False, "incomplete": False,
        "budget_fraction": 0.036, "evals_run": None, "evals_planned": 1877,
    }
    row.update(over)
    return row


def test_flatten_unflatten_is_exact():
    row = _row()
    metrics, tags = mlflow_sync.flatten_row(row)
    back = mlflow_sync.unflatten_row(metrics, tags)
    for key, value in row.items():
        if key in {"arm", "uid", "model", "dataset", "seed"} or value is None:
            continue
        assert back[key] == pytest.approx(value) if isinstance(value, float) else back[key] == value


def test_booleans_survive_as_booleans_not_as_ones_and_zeros():
    """`truncated` drives a figure marker and `summarize` averages it into a
    per-cell rate. If it came back as the string "false" the marker would fire
    on every cell, since a non-empty string is truthy."""
    _, tags = mlflow_sync.flatten_row(_row(truncated=True, at_last_eval=False))
    back = mlflow_sync.unflatten_row({}, tags)
    assert back["truncated"] is True
    assert back["at_last_eval"] is False


def test_unknown_is_not_written_as_zero():
    """A dump too old to carry `evals_run` must not come back claiming that
    zero evaluations ran -- that reads as a crashed run."""
    metrics, tags = mlflow_sync.flatten_row(_row(evals_run=None))
    assert "evals_run" not in metrics and "evals_run" not in tags
    assert "evals_run" not in mlflow_sync.unflatten_row(metrics, tags)


def test_nested_detection_scores_round_trip():
    """Fig 2 rows nest: row["msp"]["auroc"]. Score and metric names contain
    single underscores (`max_logit`, `fpr95`), so the join must not be one."""
    row = {"arm": "vbll", "uid": "u", "msp": {"auroc": 0.99, "fpr95": 0.27},
           "max_logit": {"auroc": 0.5}}
    metrics, tags = mlflow_sync.flatten_row(row)
    back = mlflow_sync.unflatten_row(metrics, tags)
    assert back["msp"] == {"auroc": pytest.approx(0.99), "fpr95": pytest.approx(0.27)}
    assert back["max_logit"] == {"auroc": pytest.approx(0.5)}


def test_sync_then_fetch_reproduces_the_rows():
    """The whole point: the server route and the JSON route must agree."""
    runs = [_StubRun("r1", {"group_id": "probe_adamw_ce_fig1",
                            "uid": "dinov3_big_dtd_seed137"})]
    client = _StubClient(runs)
    rows = [_row()]
    mlflow_sync.sync_rows(client, "11", rows, group_ids=GROUPS,
                          provenance="test", dry_run=False)

    fetched = list(mlflow_sync.fetch_rows(client, "11", GROUPS, UID_RE))
    assert len(fetched) == 1
    got, want = fetched[0], rows[0]
    for key in ("arm", "uid", "model", "dataset", "seed", "truncated"):
        assert got[key] == want[key]
    for key in ("acc", "ece", "nll", "brier", "temperature", "step"):
        assert got[key] == pytest.approx(want[key])


def test_dry_run_writes_nothing():
    runs = [_StubRun("r1", {"group_id": "probe_adamw_ce_fig1",
                            "uid": "dinov3_big_dtd_seed137"})]
    client = _StubClient(runs)
    log = mlflow_sync.sync_rows(client, "11", [_row()], group_ids=GROUPS,
                                provenance="test", dry_run=True)
    assert log and not client.metrics and not client.tags


def test_excluded_runs_are_marked_and_never_fetched_back():
    """A crashed run whose dump looks normal is what nothing downstream could
    detect. The server should carry the verdict, and must not then serve that
    run back into a figure."""
    runs = [_StubRun("r1", {"group_id": "probe_adamw_ce_fig1",
                            "uid": "dinov3_big_food101_seed143"})]
    client = _StubClient(runs)
    mlflow_sync.sync_rows(
        client, "11", [], group_ids=GROUPS, provenance="test",
        excluded={("adamw_ce", "dinov3_big_food101_seed143"): "FAILED after 39 s"},
        dry_run=False,
    )
    assert any(k == "bllarse.excluded" for _, k, _ in client.tags)
    assert list(mlflow_sync.fetch_rows(client, "11", GROUPS, UID_RE)) == []


def test_an_unsynced_run_is_skipped_not_returned_empty():
    """A run with no scored metrics is one we know nothing about. Returning it
    as a row of empty values would quietly shrink the cell it belongs to."""
    runs = [_StubRun("r2", {"group_id": "probe_adamw_ce_fig1",
                            "uid": "dinov3_big_dtd_seed138"})]
    client = _StubClient(runs)
    assert list(mlflow_sync.fetch_rows(client, "11", GROUPS, UID_RE)) == []


def test_params_are_not_used_for_synced_values():
    """MLflow refuses to change a param once set, so a re-sync after a corrected
    analysis would fail on every run it had already touched. This pins that the
    sync path never calls log_param."""
    source = (Path(mlflow_sync.__file__)).read_text()
    assert "log_param" not in source


def test_an_excluded_run_is_never_also_scored():
    """An analysis JSON produced before the exclusions existed still carries
    rows for the crashed runs. Syncing those would leave the server holding
    scored metrics for a run it simultaneously marks excluded."""
    uid = "dinov3_big_food101_seed143"
    runs = [_StubRun("r1", {"group_id": "probe_adamw_ce_fig1", "uid": uid})]
    client = _StubClient(runs)
    log = mlflow_sync.sync_rows(
        client, "11", [_row(uid=uid)], group_ids=GROUPS, provenance="test",
        excluded={("adamw_ce", uid): "FAILED after 39 s"}, dry_run=False,
    )
    assert any("skip" in line for line in log)
    assert not client.metrics, "an excluded run must carry no scored metrics"
    assert any(k == "bllarse.excluded" for _, k, _ in client.tags)


def test_one_request_per_run_not_one_per_key():
    """The first real sync did ~26 round trips per run and was killed by a
    30-minute timeout two thirds through 600 runs. Batching is not a nicety."""
    runs = [_StubRun(f"r{i}", {"group_id": "probe_adamw_ce_fig1",
                               "uid": f"dinov3_big_dtd_seed{137+i}"})
            for i in range(3)]
    client = _StubClient(runs)
    rows = [_row(uid=f"dinov3_big_dtd_seed{137+i}") for i in range(3)]
    mlflow_sync.sync_rows(client, "11", rows, group_ids=GROUPS,
                          provenance="test", dry_run=False)
    assert client.calls == 3, f"expected one batch per run, got {client.calls}"
    assert len(client.metrics) > 3, "and the batch must actually carry the keys"


def test_every_scored_metric_carries_the_prefix_and_the_argmin_step():
    runs = [_StubRun("r1", {"group_id": "probe_adamw_ce_fig1",
                            "uid": "dinov3_big_dtd_seed137"})]
    client = _StubClient(runs)
    mlflow_sync.sync_rows(client, "11", [_row()], group_ids=GROUPS,
                          provenance="test", dry_run=False)
    assert client.metrics, "nothing was written"
    for _, key, _, step in client.metrics:
        assert key.startswith(mlflow_sync.METRIC_PREFIX), key
        assert step == 1088, "metrics are stamped at the run's argmin step"


def test_temperature_scaling_is_stripped_for_the_bayesian_arms_on_read():
    """MLflow cannot delete a metric.

    An earlier sync wrote scored_ts_* onto every VBLL and IBProbit run, before
    the analysis was changed to compute TS for AdamW+CE alone. Those values are
    on the server permanently, so the read path has to refuse them -- otherwise
    a figure built from MLflow shows TS series the JSON route does not.
    """
    for arm, group in (("vbll", "probe_vbll_fig1"), ("ibprobit", "probe_ib_fig1")):
        runs = [_StubRun("r1", {"group_id": group, "uid": "dinov3_big_dtd_seed137"},
                         metrics={"scored_nll": 0.5, "scored_ts_nll": 0.49,
                                  "scored_temperature": 1.08,
                                  "scored_ts_ece": 0.01})]
        client = _StubClient(runs)
        rows = list(mlflow_sync.fetch_rows(client, "11", {arm: group}, UID_RE))
        assert len(rows) == 1
        row = rows[0]
        assert row["nll"] == pytest.approx(0.5), "real metrics must survive"
        assert not any(k.startswith("ts_") for k in row), f"{arm} kept {sorted(row)}"
        assert "temperature" not in row


def test_adamw_ce_keeps_its_temperature_scaling():
    """It is the benchmark baseline -- results-plan 0.2 requires it beside CE."""
    runs = [_StubRun("r1", {"group_id": "probe_adamw_ce_fig1",
                            "uid": "dinov3_big_dtd_seed137"},
                     metrics={"scored_nll": 0.5, "scored_ts_nll": 0.49,
                              "scored_temperature": 1.02})]
    client = _StubClient(runs)
    row = list(mlflow_sync.fetch_rows(client, "11", GROUPS, UID_RE))[0]
    assert row["ts_nll"] == pytest.approx(0.49)
    assert row["temperature"] == pytest.approx(1.02)
