from pathlib import Path

SCRIPT = Path("src/slurm/jobs/slurm_tune.sh")


def test_the_job_script_exists_and_is_a_login_shell():
    """On MPCDF no modules are preloaded and the `module` function only exists
    once /etc/profile.d has been sourced, which a non-login shell skips."""
    text = SCRIPT.read_text()
    assert text.startswith("#!/bin/bash -l")


def test_it_requests_a_gpu_partition_explicitly():
    """Raven's default partition is `interactive`, which has no GPUs, so a bare
    `-G 1` is rejected at submit time and the array never queues."""
    text = SCRIPT.read_text()
    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --gres=gpu:a100:1" in text


def test_it_resolves_the_repo_root_through_symlinks():
    """/ptmp -> /raven/ptmp. Apptainer binds the RESOLVED directory and does not
    recreate the symlink inside the container, so an unresolved path exists on
    the host but not in the container and every task dies on import bllarse."""
    assert "readlink -f" in SCRIPT.read_text()


def test_it_binds_the_repo_and_runs_with_nv():
    text = SCRIPT.read_text()
    assert "--bind" in text
    assert "--nv" in text


def test_it_puts_the_vbll_package_on_pythonpath():
    """vbll_experiments is a LOCAL package under scripts/vbll_pytorch/src that
    is never pip-installed into the image."""
    assert "scripts/vbll_pytorch/src" in SCRIPT.read_text()


def test_it_invokes_the_worker_module():
    assert "bllarse.tuning.worker" in SCRIPT.read_text()


def test_it_passes_a_patience_of_20_by_default():
    """20 is the smallest patience at which none of the 28 real pilot curves
    stop before their own val_nll argmin (15 stops one early, 8 stops two
    early). A trial stopped before its argmin is scored at the wrong point,
    which would change which hyperparameters the campaign selects."""
    text = SCRIPT.read_text()
    assert "--patience" in text
    assert "BLLARSE_TUNE_PATIENCE:-20" in text
