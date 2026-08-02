"""The per-config MLflow run used by both probing arms.

The behaviour that matters here is what happens when tracking is OFF or BROKEN:
these arms run as ~720 unattended SLURM array tasks, and a tracking-server
outage must not cost a GPU-hour. Dumps are the source of truth; MLflow is
observability.
"""

import sys
from types import SimpleNamespace

import pytest

from bllarse import mlflow_utils


def test_disabled_config_yields_a_no_op_run():
    with mlflow_utils.child_run({}, run_name="x") as run:
        assert run.active is False
        run.log_params({"a": 1})
        run.log_metrics({"b": 2.0}, step=3)
        run.set_tags({"c": "d"})


def test_absent_enable_flag_does_not_import_mlflow(monkeypatch):
    """The disabled path must not pay MLflow's import cost, and must not fail
    on an image where MLflow is not installed at all."""
    monkeypatch.setitem(sys.modules, "mlflow", None)  # importing it would raise
    with mlflow_utils.child_run({"enable_mlflow": False}, run_name="x") as run:
        assert run.active is False


def test_a_broken_tracking_backend_does_not_take_the_job_down(monkeypatch, capsys):
    """Measured against the real failure: MLflow now raises on a file:// store
    ('maintenance mode'). That must degrade to untracked, not to a dead job."""

    class Boom:
        def set_tracking_uri(self, uri):
            raise RuntimeError("tracking backend is in maintenance mode")

    monkeypatch.setitem(sys.modules, "mlflow", Boom())
    config = {"enable_mlflow": True, "mlflow_tracking_uri": "file:///nope"}
    with mlflow_utils.child_run(config, run_name="x") as run:
        assert run.active is False
        run.log_metrics({"acc": 1.0}, step=1)
    assert "continuing untracked" in capsys.readouterr().out


def test_parent_run_id_is_attached_as_a_tag(monkeypatch):
    """Cross-process nesting: the parent is created by run_sweep.py in the
    submitting process, so there is no active parent on this process's run
    stack and start_run(nested=True) would not attach. Verified end-to-end
    against a sqlite backend; this pins the mechanism."""
    seen = {}

    class FakeRun:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class FakeClient:
        """Models the two calls resolve_experiment_id makes."""

        def get_experiment_by_name(self, name):
            seen["experiment"] = name
            return SimpleNamespace(experiment_id="exp-id-7")

        def create_experiment(self, name):  # pragma: no cover - must not happen
            raise AssertionError("experiment exists; must not be re-created")

    class FakeMlflow:
        MlflowClient = FakeClient

        def set_tracking_uri(self, uri):
            seen["uri"] = uri

        def set_experiment(self, name):  # pragma: no cover - must not be used
            raise AssertionError(
                "set_experiment does not activate a newly created experiment; "
                "child_run must resolve the id and pass experiment_id instead"
            )

        def start_run(self, experiment_id=None, run_name=None, tags=None):
            seen["experiment_id"] = experiment_id
            seen["run_name"] = run_name
            seen["tags"] = tags
            return FakeRun()

        def log_params(self, params):
            seen.setdefault("params", {}).update(params)

        def log_metrics(self, metrics, step=None):
            seen.setdefault("metrics", []).append((step, metrics))

        def set_tags(self, tags):
            pass

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())
    monkeypatch.setenv("MLFLOW_PARENT_RUN_ID", "parent123")
    config = {"enable_mlflow": True, "mlflow_experiment": "exp", "seed": 137}

    with mlflow_utils.child_run(config, run_name="uid_0", tags={"arm": "vbll"}) as run:
        assert run.active is True
        run.log_metrics({"acc": 0.5, "ignored": "not a number"}, step=7)

    assert seen["tags"]["mlflow.parentRunId"] == "parent123"
    assert seen["tags"]["arm"] == "vbll"
    assert seen["experiment"] == "exp"
    # The run must be filed by resolved id, not by whatever experiment happened
    # to be active -- see resolve_experiment_id in mlflow_utils.
    assert seen["experiment_id"] == "exp-id-7"
    assert seen["params"]["seed"] == "137"
    assert seen["metrics"] == [(7, {"acc": 0.5})]


def test_a_missing_experiment_is_created_and_its_id_used(monkeypatch):
    """mlflow.set_experiment() creates a missing experiment but does NOT make it
    active, so the next start_run() silently lands in the previously active one
    (Default, on a fresh process). Observed against the real tracking server:
    set_experiment("brand_new") resolved id 7 while the run landed in id 5.
    The sweep parent then sits in a different experiment from its children and
    the UI shows no nesting at all. Pin the resolve-then-pass-id behaviour.
    """
    seen = {}

    class FakeClient:
        def get_experiment_by_name(self, name):
            return None if "created" not in seen else SimpleNamespace(
                experiment_id="new-id-9"
            )

        def create_experiment(self, name):
            seen["created"] = name
            return "new-id-9"

    class FakeMlflow:
        MlflowClient = FakeClient

        def set_tracking_uri(self, uri):
            pass

        def start_run(self, experiment_id=None, run_name=None, tags=None):
            seen["experiment_id"] = experiment_id

            class R:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

            return R()

        def log_params(self, params):
            pass

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())
    config = {"enable_mlflow": True, "mlflow_experiment": "brand_new"}
    with mlflow_utils.child_run(config, run_name="x") as run:
        assert run.active is True

    assert seen["created"] == "brand_new"
    assert seen["experiment_id"] == "new-id-9"


def test_non_numeric_metrics_are_dropped_not_raised(monkeypatch):
    """log_metrics is called with whatever the arm computed; a stray string
    must not abort a run mid-sweep."""
    logged = []

    class FakeMlflow:
        # set_tracking_uri is reachable whenever MLFLOW_TRACKING_URI is set --
        # and load_mlflow_env_defaults() sets it process-wide from
        # ~/.mlflow/credentials, so whether it is called depends on which tests
        # ran first. Omitting it made this test pass alone and fail in the full
        # suite, by silently degrading to the _NullRun path.
        def set_tracking_uri(self, uri):
            pass

        class MlflowClient:
            def get_experiment_by_name(self, name):
                return SimpleNamespace(experiment_id="0")

        def start_run(self, experiment_id=None, run_name=None, tags=None):
            class R:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

            return R()

        def log_params(self, params):
            pass

        def log_metrics(self, metrics, step=None):
            logged.append(metrics)

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())
    with mlflow_utils.child_run({"enable_mlflow": True}, run_name="x") as run:
        run.log_metrics({"good": 1.0, "bad": None, "flag": True, "text": "x"}, step=1)
    assert logged == [{"good": 1.0}]


class _FakeRun:
    """Models the object mlflow.start_run() returns: a context manager that
    records what it was exited with, the way the real ActiveRun marks the
    run FINISHED on a clean exit and FAILED/errored otherwise."""

    def __init__(self):
        self.exit_args = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exit_args = (exc_type, exc, tb)
        return False  # never swallow -- exactly like the real ActiveRun


class _FakeClient:
    def get_experiment_by_name(self, name):
        return SimpleNamespace(experiment_id="0")


def test_a_body_exception_propagates_as_itself_when_mlflow_is_enabled(monkeypatch):
    """The bug this whole file exists to catch: @contextmanager throws the
    body's exception INTO the generator at the `yield` point. The old code
    caught it with `except Exception: ... yield _NullRun()`, and yielding a
    second time after catching a thrown exception makes CPython raise
    RuntimeError("generator didn't stop after throw()") instead -- the real
    exception (here, what would be optuna.TrialPruned in production) is
    destroyed. enable_mlflow=True is what every production sweep config
    uses, so this path is the one that matters."""

    class Sentinel(Exception):
        pass

    class FakeMlflow:
        MlflowClient = _FakeClient

        def set_tracking_uri(self, uri):
            pass

        def start_run(self, experiment_id=None, run_name=None, tags=None):
            return _FakeRun()

        def log_params(self, params):
            pass

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())
    sentinel = Sentinel("prune me")
    with pytest.raises(Sentinel) as excinfo:
        with mlflow_utils.child_run({"enable_mlflow": True}, run_name="x"):
            raise sentinel
    # Not just the same type -- the same object, unchanged.
    assert excinfo.value is sentinel


def test_the_run_is_recorded_as_failed_not_finished_when_the_body_raises(monkeypatch):
    """`mlflow.start_run(...)` used as a context manager marks the run
    FINISHED on clean exit and FAILED when the body raises. child_run must
    preserve that distinction even though it can no longer use a plain
    `with` statement (see the bug this file is named for) -- a crashed or
    pruned run must never show up as a normal FINISHED one."""

    created = {}

    class FakeMlflow:
        MlflowClient = _FakeClient

        def set_tracking_uri(self, uri):
            pass

        def start_run(self, experiment_id=None, run_name=None, tags=None):
            created["run"] = _FakeRun()
            return created["run"]

        def log_params(self, params):
            pass

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())

    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        with mlflow_utils.child_run({"enable_mlflow": True}, run_name="x"):
            raise Boom("crashed")

    exc_type, exc, _ = created["run"].exit_args
    assert exc_type is Boom
    assert isinstance(exc, Boom)


def test_a_setup_failure_after_start_run_still_degrades_to_a_no_op_run(monkeypatch, capsys):
    """A failure between a successful start_run() and the first log_params
    (still "setup") must degrade to untracked like every other setup
    failure -- not raise, and not leave the half-opened run stuck RUNNING
    on the server forever."""

    created = {}

    class FakeMlflow:
        MlflowClient = _FakeClient

        def set_tracking_uri(self, uri):
            pass

        def start_run(self, experiment_id=None, run_name=None, tags=None):
            created["run"] = _FakeRun()
            return created["run"]

        def log_params(self, params):
            raise RuntimeError("tracking backend rejected the params")

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())
    with mlflow_utils.child_run({"enable_mlflow": True}, run_name="x") as run:
        assert run.active is False

    assert "continuing untracked" in capsys.readouterr().out
    # The half-opened run must be closed out as failed, not left dangling.
    assert created["run"].exit_args[0] is RuntimeError


def test_a_broken_log_metrics_call_during_the_body_does_not_propagate(monkeypatch, capsys):
    """The per-call tolerance that used to live in the outer catch-all now
    has to live in _LiveRun itself: a tracking outage discovered mid-run
    (not at setup) must still not cost a GPU job."""

    class FakeMlflow:
        MlflowClient = _FakeClient

        def set_tracking_uri(self, uri):
            pass

        def start_run(self, experiment_id=None, run_name=None, tags=None):
            return _FakeRun()

        def log_params(self, params):
            pass

        def log_metrics(self, metrics, step=None):
            raise RuntimeError("tracking backend hiccup")

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())
    with mlflow_utils.child_run({"enable_mlflow": True}, run_name="x") as run:
        assert run.active is True
        run.log_metrics({"acc": 1.0}, step=1)  # must not raise

    assert "continuing untracked" in capsys.readouterr().out
