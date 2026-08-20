from __future__ import annotations

import configparser
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def load_mlflow_env_defaults(credentials_path: Optional[Path] = None) -> Dict[str, str]:
    """
    Load MLflow credentials/tracking URI from ~/.mlflow/credentials (if present)
    and set environment defaults when variables are missing.
    """
    creds_path = credentials_path or (Path.home() / ".mlflow" / "credentials")
    if not creds_path.exists():
        return {}

    parser = configparser.ConfigParser()
    try:
        parser.read(creds_path)
    except Exception:
        return {}

    if not parser.has_section("mlflow"):
        return {}

    username = parser.get("mlflow", "mlflow_tracking_username", fallback="").strip()
    password = parser.get("mlflow", "mlflow_tracking_password", fallback="").strip()

    tracking_uri = ""
    for key in ("mlflow_tracking_uri", "tracking_uri"):
        tracking_uri = parser.get("mlflow", key, fallback="").strip()
        if tracking_uri:
            break

    env_defaults: Dict[str, str] = {}
    if username:
        env_defaults["MLFLOW_TRACKING_USERNAME"] = username
    if password:
        env_defaults["MLFLOW_TRACKING_PASSWORD"] = password
    if tracking_uri:
        env_defaults["MLFLOW_TRACKING_URI"] = tracking_uri

    for key, value in env_defaults.items():
        os.environ.setdefault(key, value)

    return env_defaults


class _NullRun:
    """Stand-in used when tracking is off or unreachable.

    Every method is a no-op, so an arm can call `run.log_metrics(...)`
    unconditionally. A tracking outage must never take a GPU job down with it --
    the dumps, not MLflow, are the source of truth for every published number.
    """

    active = False
    # The OOD worker resumes the trainer's run by id, so it has to travel in
    # the checkpoint. None here means "untracked": the worker then logs
    # nothing rather than opening a stray run of its own.
    run_id = None

    def log_params(self, params: Mapping[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        pass

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        pass


def _clean_params(params: Mapping[str, Any]) -> Dict[str, str]:
    # MLflow rejects a param over 500 chars and freezes params after the
    # first write, so stringify defensively and log once per key.
    return {k: str(v)[:500] for k, v in params.items()}


class _LiveRun:
    active = True

    def __init__(self, mlflow_module: Any, run_id: str | None = None) -> None:
        self._mlflow = mlflow_module
        # Exposed so a SECOND process can attach to this same run --
        # start_run(run_id=...) from the OOD worker, so its metrics land on the
        # training run rather than beside it.
        self.run_id = run_id
        # Once the tracking backend starts rejecting calls it usually stays
        # broken for the rest of the run (a dead connection, a full disk on
        # the server, ...), and a training loop can call log_metrics on the
        # order of hundreds of times. Printing the same "continuing
        # untracked" line at every single one of those would drown the real
        # training log for no extra information, so we report the failure
        # once per run and go quiet after that -- the dumps are still the
        # source of truth regardless.
        self._warned = False

    def _report(self, action: str, exc: Exception) -> None:
        if not self._warned:
            print(
                f"[bllarse] MLflow {action} failed "
                f"({type(exc).__name__}: {exc}); continuing untracked."
            )
            self._warned = True

    def log_params(self, params: Mapping[str, Any]) -> None:
        try:
            self._mlflow.log_params(_clean_params(params))
        except Exception as exc:  # pragma: no cover - network dependent
            self._report("log_params", exc)

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        clean = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if not clean:
            return
        try:
            self._mlflow.log_metrics(clean, step=step)
        except Exception as exc:
            self._report("log_metrics", exc)

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        try:
            self._mlflow.set_tags(_clean_params(tags))
        except Exception as exc:  # pragma: no cover - network dependent
            self._report("set_tags", exc)


def resolve_experiment_id(mlflow_module: Any, name: str) -> str:
    """Return the id of experiment `name`, creating it if absent.

    NOT `mlflow.set_experiment(name)`. When that call has to CREATE the
    experiment it does not make the new one active, so the very next
    `start_run()` silently lands in whatever experiment was active before
    (Default, on a fresh process). Verified against the tracking server:

        set_experiment("existing") -> id 5, run landed in 5   OK
        set_experiment("brand_new") -> id 7, run landed in 5   WRONG

    That is invisible until you notice the sweep parent sitting in Default
    while its children are in the real experiment -- at which point the UI
    shows no nesting at all, because it only nests within one experiment.
    Resolving the id explicitly and passing `experiment_id=` to start_run
    removes the dependence on that active-experiment state entirely.
    """
    client = mlflow_module.MlflowClient()
    existing = client.get_experiment_by_name(name)
    if existing is not None:
        return existing.experiment_id
    try:
        return client.create_experiment(name)
    except Exception:
        # A concurrent array task won the race; re-read rather than fail. If it
        # is still missing the error is real, so let this second call raise.
        return client.get_experiment_by_name(name).experiment_id


# Exception types that mean a run ended early BY DESIGN rather than broke.
#
# Without this distinction every pruned trial lands on the server as FAILED,
# indistinguishable from a genuine harness fault. That is not hypothetical: the
# `bllarse-probing-tuning` experiment holds 22,374 FAILED runs out of 24,492,
# and essentially all of them are trials the MedianPruner killed on purpose.
# The search's own record is therefore unreadable -- you cannot filter the
# campaign that worked from the campaign that was misconfigured -- which is
# exactly the state a three-day fine-tuning failure went unnoticed in.
#
# Matched by TYPE NAME, not by class, and that is a deliberate trade. The two
# that matter are `optuna.TrialPruned` and `tuning.objective.EarlyConvergence`;
# a class-based check would make this module import optuna, and this module is
# imported by training scripts that run standalone on GPU nodes. A name match
# costs precision no one here can spend and buys zero coupling.
BY_DESIGN_EXCEPTIONS = frozenset({"TrialPruned", "EarlyConvergence"})


@contextmanager
def child_run(
    config: Mapping[str, Any],
    *,
    run_name: str,
    tags: Mapping[str, Any] | None = None,
    by_design_exceptions: frozenset[str] = BY_DESIGN_EXCEPTIONS,
):
    """A per-config MLflow run, nested under the sweep parent. No-op when off.

    Nesting is done by setting `mlflow.parentRunId` explicitly rather than by
    `start_run(nested=True)`: the parent is created by run_sweep.py in the
    submitting process and every child lives in a different SLURM array task,
    so there is no active parent on this process's run stack to nest under.

    Yields a run object whose log_* methods are no-ops when `enable_mlflow` is
    unset or when MLflow cannot be reached, so callers need no branching.
    """
    if not config.get("enable_mlflow", False):
        yield _NullRun()
        return

    try:
        load_mlflow_env_defaults()
        import mlflow
    except Exception as exc:  # pragma: no cover - depends on the install
        print(f"[bllarse] MLflow unavailable ({type(exc).__name__}: {exc}); continuing untracked.")
        yield _NullRun()
        return

    tracking_uri = config.get("mlflow_tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI")
    experiment = (
        config.get("mlflow_experiment")
        or os.environ.get("MLFLOW_EXPERIMENT_NAME")
        or "bllarse"
    )
    parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID")

    all_tags = dict(tags or {})
    if parent_run_id:
        all_tags["mlflow.parentRunId"] = parent_run_id

    # Setup: everything up to and including the initial log_params call is
    # guarded together, exactly as before -- any failure here degrades the
    # whole run to untracked. This is deliberately NOT a `with` statement:
    # the body has to run *outside* any `except` block that could see its
    # exception, or a body exception thrown into this generator would be
    # caught here and the generator would try to yield a second time, which
    # CPython turns into "RuntimeError: generator didn't stop after throw()"
    # -- destroying the real exception (this was the bug). So start_run's
    # context manager is driven by hand: __enter__ during setup, __exit__
    # during teardown, with the body's yield in between, guarded by nothing.
    run_ctx = None
    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        experiment_id = resolve_experiment_id(mlflow, experiment)
        run_ctx = mlflow.start_run(experiment_id=experiment_id, run_name=run_name, tags=all_tags)
        run_ctx.__enter__()
        mlflow.log_params(_clean_params(config))
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[bllarse] MLflow run failed ({type(exc).__name__}: {exc}); continuing untracked.")
        if run_ctx is not None:
            # start_run succeeded but something after it (log_params, ...)
            # did not -- close out the half-opened run as failed rather than
            # leaving it stuck RUNNING on the server forever. Best effort:
            # this is already the "tracking is broken" path.
            try:
                run_ctx.__exit__(type(exc), exc, exc.__traceback__)
            except Exception:
                pass
        yield _NullRun()
        return

    run = _LiveRun(mlflow, run_id=getattr(
        getattr(run_ctx, "info", None), "run_id", None))
    try:
        # Body: an exception here (including optuna.TrialPruned) must reach
        # the caller as itself. Nothing above this line can be an `except`
        # that wraps this `yield`.
        yield run
    except BaseException as body_exc:
        # Teardown on the failure path. Re-raises the SAME exception object
        # untouched either way; a teardown failure must not replace or mask it.
        #
        # The two branches differ only in what the SERVER is told. A pruned or
        # patience-stopped trial is a normal outcome of the search, so marking
        # it FAILED -- which is what `run_ctx.__exit__` does for any exception
        # -- destroys the distinction between "the pruner did its job" and "the
        # harness is broken". KILLED is MLflow's status for stopped-on-purpose,
        # and `run_outcome` records which stopper fired.
        outcome = type(body_exc).__name__
        try:
            if outcome in by_design_exceptions:
                run.set_tags({"run_outcome": outcome})
                mlflow.end_run(status="KILLED")
            else:
                run_ctx.__exit__(*sys.exc_info())
        except Exception as exc:  # pragma: no cover - network dependent
            print(
                f"[bllarse] MLflow end_run failed ({type(exc).__name__}: {exc}); "
                "continuing untracked."
            )
        raise
    else:
        # Teardown on the success path: mark the run finished. A failure
        # here is a tracking-only problem and must not raise.
        try:
            run.set_tags({"run_outcome": "completed"})
            run_ctx.__exit__(None, None, None)
        except Exception as exc:  # pragma: no cover - network dependent
            print(
                f"[bllarse] MLflow end_run failed ({type(exc).__name__}: {exc}); "
                "continuing untracked."
            )


# --- Analysis artifacts ------------------------------------------------------
#
# The dumps stay the source of truth, but the JSON the analysis distils out of
# them does not have to live only on the cluster that produced it. Publishing it
# as an MLflow artifact is what lets plotting run anywhere: without this, every
# figure needs Raven mounted, because the JSON is written next to the dumps and
# nothing carries it off.
#
# Deliberately NOT the lossy path the training loop uses. `_LiveRun` swallows
# failures because a tracking outage must never kill a GPU job; here a failure
# means the artifact does not exist, and a caller that then plots from a stale
# local copy would publish the wrong figure. So these raise.

ANALYSIS_TAG = "bllarse.analysis"


def _analysis_client(tracking_uri: Optional[str] = None):
    load_mlflow_env_defaults()
    import mlflow

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError(
            "no MLflow tracking URI: set MLFLOW_TRACKING_URI (or pass one). "
            "Publishing an analysis artifact is not optional bookkeeping -- "
            "silently skipping it would leave the figures reading a stale file."
        )
    return mlflow, uri


def publish_analysis(
    path: Path | str,
    name: str,
    *,
    experiment: str = "bllarse-probing",
    tracking_uri: Optional[str] = None,
    tags: Optional[Mapping[str, Any]] = None,
) -> str:
    """Publish an analysis JSON as an artifact. Returns the run id.

    One run per `name`, REUSED across republishes rather than a new run each
    time: the point is a stable address that `fetch_analysis` can resolve, and a
    fresh run per invocation would turn "the fig1 JSON" into a guess about which
    of forty runs was meant. MLflow has no upsert, so the run is found by tag.
    """
    # The local check comes FIRST, before anything touches the network: a
    # mistyped --publish path is a local mistake and should say so without
    # needing a reachable tracking server to find out.
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"nothing to publish at {path}")

    mlflow, uri = _analysis_client(tracking_uri)
    mlflow.set_tracking_uri(uri)
    experiment_id = resolve_experiment_id(mlflow, experiment)
    client = mlflow.MlflowClient()

    existing = client.search_runs(
        [experiment_id], filter_string=f"tags.`{ANALYSIS_TAG}` = '{name}'",
        max_results=1, order_by=["attributes.start_time DESC"],
    )
    run_id = existing[0].info.run_id if existing else None
    if run_id is None:
        run_id = client.create_run(
            experiment_id, run_name=name, tags={ANALYSIS_TAG: name}
        ).info.run_id

    client.log_artifact(run_id, str(path))
    for key, value in (tags or {}).items():
        client.set_tag(run_id, key, str(value)[:500])
    # Terminated explicitly: created runs start RUNNING, and a run left RUNNING
    # forever reads in the UI as a job that hung.
    client.set_terminated(run_id, "FINISHED")
    return run_id


def fetch_analysis(
    name: str,
    *,
    experiment: str = "bllarse-probing",
    tracking_uri: Optional[str] = None,
    dest_dir: Path | str | None = None,
    artifact: str | None = None,
) -> Path:
    """Download a published analysis artifact. Returns the local path.

    Raises when the artifact is absent rather than falling back to a local file:
    a plot built from a stale JSON that the author believed came from the server
    is exactly the failure this whole mechanism exists to prevent.
    """
    mlflow, uri = _analysis_client(tracking_uri)
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()

    experiment_obj = client.get_experiment_by_name(experiment)
    if experiment_obj is None:
        raise FileNotFoundError(f"no MLflow experiment named {experiment!r}")

    runs = client.search_runs(
        [experiment_obj.experiment_id],
        filter_string=f"tags.`{ANALYSIS_TAG}` = '{name}'",
        max_results=1, order_by=["attributes.start_time DESC"],
    )
    if not runs:
        raise FileNotFoundError(
            f"no published analysis named {name!r} in experiment {experiment!r}. "
            f"Run the analysis with --publish where the dumps are."
        )
    run_id = runs[0].info.run_id

    if artifact is None:
        files = [a.path for a in client.list_artifacts(run_id) if not a.is_dir]
        if not files:
            raise FileNotFoundError(
                f"analysis run {name!r} ({run_id}) exists but holds no artifact"
            )
        artifact = files[0]

    local = client.download_artifacts(
        run_id, artifact, str(dest_dir) if dest_dir else None
    )
    return Path(local)


def resolve_analysis_path(spec: str, *, dest_dir: Path | str | None = None,
                          experiment: str = "bllarse-probing") -> Path:
    """A local path, or `mlflow:<name>` downloaded to one.

    Lets every plotting script take its input from either side of the cluster
    boundary without each one growing its own copy of the branch.
    """
    if spec.startswith("mlflow:"):
        return fetch_analysis(spec[len("mlflow:"):], experiment=experiment,
                              dest_dir=dest_dir)
    return Path(spec)
