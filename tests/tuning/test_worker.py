import json
import sys

import pytest

optuna = pytest.importorskip("optuna")

from bllarse.tuning import space, worker


def test_study_names_are_unique_per_arm_and_dataset():
    names = {
        worker.study_name(arm, ds)
        for arm in ("adamw_ce", "vbll")
        for ds in space.DATASETS
    }
    assert len(names) == 2 * len(space.DATASETS) == 14


def test_workers_sharing_a_storage_dir_share_one_study(tmp_path):
    """Four workers per study, one journal file per study. SQLite over GPFS with
    concurrent writers is unreliable; the journal backend is built for network
    filesystems, and one file per study means the 14 studies never contend."""
    first = worker.open_study("vbll", "food101", tmp_path)
    first.tell(first.ask(), 1.23)

    second = worker.open_study("vbll", "food101", tmp_path)
    assert len(second.trials) == 1, "the second worker must see the first's trial"
    assert second.study_name == first.study_name


def test_the_study_minimises():
    assert worker.DIRECTION == "minimize"


def test_suggested_batch_sizes_never_exceed_the_dataset_cap(tmp_path):
    study = worker.open_study("adamw_ce", "dtd", tmp_path)
    cap = space.batch_cap(space.N_TRAIN["dtd"])
    for _ in range(20):
        trial = study.ask()
        params = space.suggest(trial, "adamw_ce", "dtd")
        assert params["batch_size"] <= cap == 1024
        study.tell(trial, 1.0)


def test_the_trial_config_carries_the_datasets_epoch_budget(tmp_path):
    budgets = {"dtd": {"batch_size": 512, "steps": 2000, "epochs": 273}}
    config = worker.trial_config(
        "vbll", "dtd",
        params={"learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 256,
                "prior_scale": 1.0, "wishart_scale": 0.1},
        budgets=budgets,
    )
    # 2,000 steps at the SAMPLED batch size, not the pilot's: the budget is a
    # step budget, and epochs is only how the scripts express it.
    # dtd is 3,760 rows: 3760 // 256 = 14 steps/epoch, ceil(2000 / 14) = 143.
    assert config["epochs"] == 143
    assert config["batch_size"] == 256
    assert config["parameterization"] == "diagonal"
    assert config["model"] == space.TUNING_MODEL
    assert config["num_classes"] == 47
    assert config["img_size"] == 512


def test_a_missing_epoch_budget_raises(tmp_path):
    with pytest.raises(KeyError, match="dtd"):
        worker.trial_config(
            "adamw_ce", "dtd",
            params={"learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 256},
            budgets={},
        )


def test_main_passes_catch_to_study_optimize(tmp_path, monkeypatch):
    """Pins the fix for the incident that killed 13 of the 56 worker tasks
    (14 studies x 4 workers): a single trial's
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED exception propagated straight out of
    study.optimize and killed the whole worker process, abandoning every trial
    it still had left to run. worker.main() must pass Optuna's `catch`
    argument -- covering at least Exception -- so a failing trial is recorded
    FAILED instead and the worker moves on to the next one."""
    budgets_path = tmp_path / "budgets.json"
    budgets_path.write_text("{}")

    captured: dict[str, object] = {}

    def fake_optimize(self, func, n_trials=None, catch=(), callbacks=None, **kw):
        captured["catch"] = catch
        captured["callbacks"] = callbacks
        # A real trial so main()'s closing `study.best_value` print does not
        # itself raise on an empty study.
        self.tell(self.ask(), 1.0)

    monkeypatch.setattr(optuna.Study, "optimize", fake_optimize)
    monkeypatch.setattr(
        sys, "argv",
        [
            "worker",
            "--arm", "adamw_ce",
            "--dataset", "food101",
            "--n-trials", "1",
            "--storage-dir", str(tmp_path),
            "--budgets", str(budgets_path),
            "--no-mlflow",
        ],
    )

    worker.main()

    catch = captured["catch"]
    assert isinstance(catch, tuple)
    assert Exception in catch
    # optuna.TrialPruned must NOT be caught here: Optuna handles pruning via
    # its own separate except-clause before `catch` is ever consulted, and a
    # prune re-caught as a plain failure would be misrecorded as FAILED instead
    # of PRUNED. KeyboardInterrupt/SystemExit derive from BaseException, not
    # Exception, so a scope of exactly (Exception,) already excludes them --
    # no explicit exclusion is needed for either.
    assert optuna.TrialPruned not in catch


def test_a_failing_trial_does_not_abort_the_remaining_trials(tmp_path):
    """The behaviour `catch` buys, pinned directly against a live study rather
    than against the wiring in main(): a trial that raises must not stop the
    ones after it, and must be recorded FAILED rather than silently vanishing
    or crashing the run."""
    study = worker.open_study("adamw_ce", "food101", tmp_path)
    calls = {"n": 0}

    def flaky_objective(trial: optuna.Trial) -> float:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED")
        return 1.0

    study.optimize(flaky_objective, n_trials=2, catch=(Exception,))

    states = [t.state for t in study.trials]
    assert states.count(optuna.trial.TrialState.FAIL) == 1
    assert states.count(optuna.trial.TrialState.COMPLETE) == 1
