"""The fine-tuning launcher.

Checked as text rather than executed: it submits to SLURM, and every property
worth guarding is a property of what it would submit.

The defect this launcher exists to correct is documented in its own header --
the first campaign gave every arm 10 trials per worker and, because the prune
rate ran 83%-92%, landed at 36 completed trials on ibprobit_noeps and 16 on
vbll. Those arms were then compared. These tests guard the correction.
"""
import os
from pathlib import Path

import pytest

from bllarse.tuning import finetune_space as fs

SCRIPT = Path("src/slurm/submit_finetune_tune.sh")
TEXT = SCRIPT.read_text()
JOB = Path("src/slurm/jobs/slurm_finetune_tune.sh").read_text()


def _default(name: str) -> str:
    """The value of a `: "${NAME:=...}"` line."""
    body = TEXT[TEXT.index(f"{name}:="):]
    return body[len(name) + 2:body.index("}")]


def test_it_is_executable():
    assert os.access(SCRIPT, os.X_OK)


def test_it_passes_a_completed_trial_target_not_only_a_trial_count():
    # --n-trials counts PRUNED trials too, so on its own it converts into a
    # search depth nobody chose. Both are needed: the target sets the depth,
    # the count keeps a mostly-pruned study off a GPU for the full 24 h wall.
    assert "BLLARSE_FT_TARGET_COMPLETE" in TEXT
    assert "BLLARSE_FT_TRIALS" in TEXT
    assert "--target-complete" in JOB


def test_the_target_is_forwarded_by_the_job_script_it_submits():
    # A launcher that exports a variable no job script reads is worse than one
    # that does not export it: the submission LOOKS depth-controlled.
    assert 'BLLARSE_FT_TARGET_COMPLETE' in JOB


def test_it_names_a_campaign_so_a_rerun_cannot_resume_the_old_study():
    """open_study loads by name with load_if_exists.

    The second campaign pins batch_size and moves five boxes, so resuming
    ft_*_shards2m would have TPE model a space that no longer exists -- and for
    IBPROBIT_UPDATE_ITERS, which gained two rungs, optuna raises
    `CategoricalDistribution does not support dynamic value space` and fails
    every trial rather than failing quietly.
    """
    campaign = _default("BLLARSE_FT_CAMPAIGN")
    assert campaign != ""
    # The DEFAULT, not the file: the header cites ft_*_shards2m by name as the
    # campaign whose findings set these boxes, and should keep doing so.
    assert campaign != "shards2m"


def test_the_default_arms_are_a_subset_of_the_search_space():
    arms = _default("BLLARSE_FT_ARMS").split()
    assert arms, "no default arms"
    for arm in arms:
        assert arm in fs.ARMS, f"{arm} is not in finetune_space.ARMS"


def test_the_eps_searched_arm_is_droppable_but_still_supported():
    """`ibprobit` is out of the default set, not out of the code.

    Its question is answered: across the first campaign's top six trials
    eps_init spanned five decades (8.4e-7 to 2.1e-2) with every score inside
    0.004, and the arm with eps numerically absent won outright. But the
    ABLATION still has to be runnable -- the finding is only reportable while
    the arm that produced it can be re-run.
    """
    assert "ibprobit" in fs.ARMS
    assert "ibprobit_noeps" in _default("BLLARSE_FT_ARMS")
    # Named in the script, so putting it back does not require reading suggest().
    assert "BLLARSE_FT_ARMS" in TEXT


def test_the_two_frameworks_get_separate_interpreters():
    # jax[cuda] and torch each ship a full CUDA stack; installing both into one
    # environment is what broke the vbll smoke runs.
    assert "BLLARSE_FT_VBLL_VENV" in TEXT
    assert 'arm" == "vbll" ]] && venv=' in TEXT


def test_it_supports_a_dry_run_that_submits_nothing():
    assert "BLLARSE_FT_DRY_RUN" in TEXT
    dry = TEXT[TEXT.index('if [[ -n "$BLLARSE_FT_DRY_RUN" ]]'):]
    # printf, not sbatch, on the taken branch.
    assert dry.index("printf") < dry.index("sbatch") if "sbatch" in dry else True


def test_it_exports_all_not_only_the_named_variables():
    # --export replaces the environment wholesale rather than adding to it, so
    # without ALL the job loses PATH and HOME and cannot find `module` or the
    # venv.
    assert '--export "ALL,' in TEXT


def test_it_throttles_the_array_so_one_arm_cannot_take_the_whole_queue():
    assert "BLLARSE_FT_THROTTLE" in TEXT
    assert "%${BLLARSE_FT_THROTTLE}" in TEXT
    assert int(_default("BLLARSE_FT_THROTTLE")) >= 1


@pytest.mark.parametrize("var", ["BLLARSE_FT_STORAGE", "BLLARSE_FT_SHARDS",
                                 "BLLARSE_FT_CAMPAIGN", "BLLARSE_FT_VENV"])
def test_every_variable_the_job_requires_is_exported(var):
    # slurm_finetune_tune.sh exits 2 on any of these being unset, which would
    # burn a queue slot per worker to discover.
    assert f"{var}=" in TEXT
