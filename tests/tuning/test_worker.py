import json
import sys

import pytest

optuna = pytest.importorskip("optuna")

from bllarse.tuning import space, worker


def test_study_names_are_unique_per_arm_and_dataset():
    names = {
        worker.study_name(arm, ds)
        for arm in space.ARMS
        for ds in space.DATASETS
    }
    assert len(names) == 3 * len(space.DATASETS) == 21


def test_the_ibprobit_trial_config_reaches_probe_pys_cli():
    """The end the unit tests cannot see: a config that `trial_config` builds
    but `probe.build_argparser` rejects fails the trial AFTER its GPU time is
    spent, and `study.optimize(catch=(Exception,))` records it as an ordinary
    failure rather than the harness fault it is.
    """
    from bllarse.tools import adapters

    budgets = {"dtd": {"batch_size": 512, "steps": 2000, "epochs": 273}}
    config = worker.trial_config(
        "ibprobit", "dtd",
        params={"batch_size": 256, "q_init": 3e-4, "eps_init": 0.02,
                "num_update_iters": 64},
        budgets=budgets,
    )
    assert config["arm"] == "ibprobit"
    assert config["px"] is True

    module = adapters.load_script_module("scripts/dinov3/probe.py")
    args = adapters.parse_config(module, config)
    assert args.arm == "ibprobit"
    assert args.q_init == 3e-4
    assert args.num_update_iters == 64
    assert args.px is True
    # --no-learn-q PRESENT: Q is pinned at its q0 seed for this campaign, so
    # the search is over a static process noise. q_init is the seed the
    # recursion would start from, so leaving it running would have the search
    # chase a moving target within each trial. This is the assertion that
    # catches the flag silently flipping back on.
    assert args.no_learn_q is True
    # kappa unset, so IBProbit's own default of 0.5 applies. Fixed by decision.
    assert args.kappa is None


def test_the_ibprobit_config_carries_no_optimizer_knobs():
    """probe.py's argparser still DEFINES --learning-rate (the adamw_ce arm
    needs it), so passing one would parse cleanly and silently suggest the arm
    has a tuned learning rate. The config must simply not carry one."""
    budgets = {"dtd": {"batch_size": 512, "steps": 2000, "epochs": 273}}
    config = worker.trial_config(
        "ibprobit", "dtd",
        params={"batch_size": 256, "q_init": 3e-4, "eps_init": 0.02,
                "num_update_iters": 64},
        budgets=budgets,
    )
    assert "learning_rate" not in config
    assert "weight_decay" not in config


def test_the_adamw_ce_config_gains_no_ibprobit_knobs():
    budgets = {"dtd": {"batch_size": 512, "steps": 2000, "epochs": 273}}
    config = worker.trial_config(
        "adamw_ce", "dtd",
        params={"learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 256},
        budgets=budgets,
    )
    assert config["arm"] == "adamw_ce"
    for knob in ("px", "q_init", "eps_init", "num_update_iters"):
        assert knob not in config


def test_workers_sharing_a_storage_dir_share_one_study(tmp_path):
    """Four workers per study, one journal file per study. SQLite over GPFS with
    concurrent writers is unreliable; the journal backend is built for network
    filesystems, and one file per study means the 14 studies never contend."""
    first = worker.open_study("vbll", "food101", tmp_path)
    first.tell(first.ask(), 1.23)

    second = worker.open_study("vbll", "food101", tmp_path)
    assert len(second.trials) == 1, "the second worker must see the first's trial"
    assert second.study_name == first.study_name


def test_an_unnamed_campaign_reproduces_the_original_study_names():
    """The suffix is opt-in, so replaying the first campaign stays possible."""
    assert worker.study_name("vbll", "food101") == "vbll_food101"
    assert worker.study_name("vbll", "food101", None) == "vbll_food101"


def test_a_campaign_suffix_makes_a_separate_study(tmp_path):
    """open_study passes load_if_exists=True, which is what lets four workers
    share one study -- and exactly why a re-run under a CHANGED search space
    must not reuse the name. The old trials carry a prior_scale distribution of
    [1e-2, 1e2] and no kl_ratio at all, so resuming would have TPE model a
    space that no longer exists."""
    old = worker.open_study("vbll", "food101", tmp_path)
    old.tell(old.ask(), 1.23)

    new = worker.open_study("vbll", "food101", tmp_path, "kl_ratio")
    assert new.study_name == "vbll_food101_kl_ratio"
    assert new.study_name != old.study_name
    assert len(new.trials) == 0, "a new campaign must not inherit old trials"

    # ...and the second worker of the NEW campaign still shares its study.
    peer = worker.open_study("vbll", "food101", tmp_path, "kl_ratio")
    peer.tell(peer.ask(), 0.5)
    assert len(worker.open_study("vbll", "food101", tmp_path, "kl_ratio").trials) == 1


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


def test_a_study_with_no_completed_trials_reports_the_harness_fault(tmp_path):
    """`catch` keeps the worker alive through a failing trial, which is right
    -- but if EVERY trial fails, study.best_value raises "No trials are
    completed yet" and that traceback becomes the last thing in the SLURM log,
    burying the real cause far above it. That is how a misresolved --cache-dir
    presented: 32 trials dying on Errno 30, reported as an opaque optuna error.
    The worker must say plainly that it produced nothing, and exit non-zero so
    the task is not mistaken for a bad-scoring configuration."""
    study = worker.open_study("adamw_ce", "dtd", tmp_path)

    def always_fails(trial: optuna.Trial) -> float:
        raise OSError(30, "Read-only file system")

    study.optimize(always_fails, n_trials=3, catch=(Exception,))
    assert all(t.state == optuna.trial.TrialState.FAIL for t in study.trials)

    with pytest.raises(SystemExit) as excinfo:
        worker.summarize_study(study)
    message = str(excinfo.value)
    assert "NONE completed" in message
    assert "3 failed" in message
    assert "harness fault" in message


def test_a_study_with_results_summarises_normally(tmp_path):
    study = worker.open_study("adamw_ce", "cifar10", tmp_path)
    study.optimize(lambda t: 0.25, n_trials=2, catch=(Exception,))

    summary = worker.summarize_study(study)
    assert "2 completed" in summary
    assert "0.25000" in summary


def test_a_partly_failed_study_still_summarises(tmp_path):
    """One bad trial must NOT be escalated to a harness fault -- that is the
    whole point of catch, and conflating the two would make a single flaky
    trial kill an otherwise good study."""
    study = worker.open_study("adamw_ce", "cifar100", tmp_path)
    calls = {"n": 0}

    def flaky(trial: optuna.Trial) -> float:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return 0.5

    study.optimize(flaky, n_trials=3, catch=(Exception,))
    summary = worker.summarize_study(study)
    assert "2 completed" in summary
