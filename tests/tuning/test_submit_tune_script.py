"""The launcher that tops studies up to a completed-trial target.

Checked as text rather than executed: it submits to SLURM, and every property
worth guarding here is a property of what it would submit.
"""
from pathlib import Path

import pytest

from bllarse.tuning import space

SCRIPT = Path("src/slurm/submit_tune.sh")
TEXT = SCRIPT.read_text()


def test_it_is_executable():
    import os
    assert os.access(SCRIPT, os.X_OK)


def test_it_covers_every_dataset_in_the_search_space():
    """A dataset left out of the launcher is a column of Fig 1 tuned to a
    different depth than the ones beside it -- the defect this submission exists
    to correct, reintroduced by omission."""
    datasets_line = TEXT[TEXT.index('BLLARSE_TUNE_DATASETS:='):]
    datasets_line = datasets_line[:datasets_line.index("}")]
    for dataset in space.DATASETS:
        assert dataset in datasets_line


def test_it_defaults_to_the_arms_the_figures_actually_read():
    """NOT every arm in space.ARMS. Both figures read the eps-free sweep --
    fig1_curves.SWEEP_MODULES maps "ibprobit" to the `probe_ibprobit_noeps`
    module and fig2_detection reads `probe_ibprobit_noeps_fig2` -- so tuning the
    with-eps arm yields a config nothing downstream reads.

    Guarded because the cost is not just wasted GPU: the submission loop claims
    GPUs in arm order, so a with-eps array runs AHEAD of the eps-free one. On
    2026-08-12 that left all 7 `tune_noeps_*` arrays PENDING behind 20 running
    `tune_ibprobit_*` tasks -- the unused arm blocking the needed one."""
    arms_line = TEXT[TEXT.index('BLLARSE_TUNE_ARMS:='):]
    arms_line = arms_line[:arms_line.index("}")]
    arms = arms_line.split(":=", 1)[1].split()

    assert "ibprobit_noeps" in arms
    assert "adamw_ce" in arms and "vbll" in arms
    assert "ibprobit" not in arms, (
        "the with-eps arm must not be a default; it is an appendix ablation "
        "and it delays the arm the figures read"
    )
    # Still reachable deliberately, so the eps ablation remains runnable.
    assert "BLLARSE_TUNE_ARMS" in TEXT


def test_the_figures_read_the_eps_free_sweep():
    """Pins the fact the default above rests on. If a figure is ever repointed
    at the with-eps sweep, this fails rather than leaving the launcher tuning
    an arm nothing reads."""
    fig1 = Path("scripts/analysis/fig1_curves.py").read_text()
    fig2 = Path("scripts/analysis/fig2_detection.py").read_text()
    assert '"ibprobit": "probe_ibprobit_noeps"' in fig1
    assert '"ibprobit": "probe_ibprobit_noeps_fig2"' in fig2


@pytest.mark.parametrize("arm,sif", [
    ("vbll", "bllarse-torch.sif"),
    ("adamw_ce", "bllarse-jax.sif"),
    ("ibprobit", "bllarse-jax.sif"),
    ("ibprobit_noeps", "bllarse-jax.sif"),
])
def test_each_arm_selects_the_image_its_training_script_runs_under(arm, sif):
    """VBLL is the torch arm; the other three all run scripts/dinov3/probe.py
    under JAX. The wrong image does not fail at submit -- it fails inside the
    first trial, once the GPU is already held."""
    case = TEXT[TEXT.index(f"    {arm})"):]
    assert sif in case[:case.index(";;")]


def test_it_passes_the_completed_trial_target_rather_than_a_trial_count_alone():
    """The point of the script. --n-trials counts trials STARTED, and the
    pruner kills 8%-72% of them depending on the study, so a per-worker count
    cannot express "search every arm to the same depth"."""
    assert "BLLARSE_TUNE_TARGET_COMPLETE" in TEXT
    assert "BLLARSE_TUNE_TARGET_COMPLETE:=100" in TEXT
    # Exported, or --export=ALL would not carry it to the job.
    exported = TEXT[TEXT.index("export BLLARSE_TUNE_STORAGE"):]
    assert "BLLARSE_TUNE_TARGET_COMPLETE" in exported[:exported.index("\n\n")]
    assert "--export=ALL" in TEXT


def test_it_defaults_to_resuming_the_existing_campaign():
    """Studies are resumed, not restarted: the worker opens them with
    load_if_exists, so an unchanged campaign name keeps all 1,836 trials
    already in the journals. A DIFFERENT default here would silently discard
    them."""
    assert "BLLARSE_TUNE_CAMPAIGN:=samples40" in TEXT


def test_it_supports_a_dry_run_that_submits_nothing():
    assert "BLLARSE_TUNE_DRY_RUN" in TEXT
    dry = TEXT[TEXT.index('if [[ -n "$BLLARSE_TUNE_DRY_RUN" ]]; then'):]
    assert "continue" in dry[:dry.index("fi")]


def test_it_fails_loudly_on_a_missing_image_rather_than_at_trial_time():
    assert 'if [[ ! -f "$sif_path" ]]; then' in TEXT
    assert "exit 2" in TEXT


def test_it_refuses_to_append_a_changed_objective_to_an_existing_campaign():
    """Smoothing redefines the objective. The worker opens studies with
    load_if_exists, so reusing the campaign name would append windowed-mean
    trials to a study of single-point-minimum trials and hand TPE a value column
    meaning two different things. The damage is silent and lands in a shared
    journal, so this must be a hard refusal, not a warning."""
    assert "BLLARSE_TUNE_SMOOTH_WINDOW" in TEXT
    guard = TEXT[TEXT.index("for _bad in"):]
    guard = guard[:guard.index("done")]
    # samples40 scored the single-point minimum; smoothed PRUNED on raw val_nll
    # while scoring the windowed mean, so its stored intermediate values are on
    # a different scale from anything reported now.
    assert "samples40" in guard and "smoothed" in guard and "smoothed2" in guard
    assert "exit 2" in guard
    # Exported, or --export=ALL would not carry it into the job.
    exported = TEXT[TEXT.index("export BLLARSE_TUNE_STORAGE"):]
    assert "BLLARSE_TUNE_SMOOTH_WINDOW" in exported[:exported.index("\n\n")]


def test_the_job_script_forwards_the_smoothing_fraction():
    job = Path("src/slurm/jobs/slurm_tune.sh").read_text()
    assert 'if [[ -n "${BLLARSE_TUNE_SMOOTH_WINDOW:-}" ]]; then' in job
    assert "--smooth-window" in job
    required = job[job.index("for var in"):job.index("done", job.index("for var in"))]
    assert "BLLARSE_TUNE_SMOOTH_WINDOW" not in required
