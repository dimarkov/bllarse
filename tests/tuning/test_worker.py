import json

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
