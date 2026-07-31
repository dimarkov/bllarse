from pathlib import Path

from bllarse.tools import run_sweep


def test_default_job_script_is_the_apptainer_one():
    assert "apptainer" in str(run_sweep.DEFAULT_JOB_SCRIPT)


def test_apptainer_job_script_exists_and_is_executable():
    path = Path("src/slurm/jobs/slurm_run_config_apptainer.sh")
    assert path.is_file()
    assert path.stat().st_mode & 0o111, "job script must be executable"


def test_apptainer_job_script_uses_nv_and_honours_the_sif_variable():
    text = Path("src/slurm/jobs/slurm_run_config_apptainer.sh").read_text()
    assert "--nv" in text, "GPU passthrough requires --nv"
    assert "BLLARSE_SIF" in text
    assert "BLLARSE_HF_HOME" in text, "HF cache path must be overridable on the cluster"
