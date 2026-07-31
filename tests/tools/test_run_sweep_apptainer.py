import os
import stat
import subprocess
from pathlib import Path

from bllarse.tools import run_sweep

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "src/slurm/jobs/slurm_run_config_apptainer.sh"

# Every env var the job script conditionally forwards into the container.
# Popped from the subprocess environment before each run so ambient values on
# the machine running the test (e.g. a real HF_TOKEN in the dev shell) cannot
# contaminate the "unset" case.
CANDIDATE_VARS = [
    "HF_TOKEN", "HF_HOME", "BLLARSE_HF_HOME", "BLLARSE_EVAL_DUMP_ROOT",
    "MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD",
    "MLFLOW_PARENT_RUN_ID", "MLFLOW_EXPERIMENT_NAME",
]


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


def _make_stub_apptainer(bin_dir: Path, capture_file: Path) -> None:
    """A fake `apptainer` that records its argv (one per line) and exits 0,
    so the real job script can run end-to-end without an actual container
    runtime or GPU."""
    stub = bin_dir / "apptainer"
    stub.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "{capture_file}"\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _run_job_script(tmp_path: Path, set_vars: dict | None = None) -> list[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture_file = tmp_path / "argv.txt"
    _make_stub_apptainer(bin_dir, capture_file)

    sif = tmp_path / "fake.sif"
    sif.touch()

    env = dict(os.environ)
    for key in CANDIDATE_VARS:
        env.pop(key, None)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["SLURM_SUBMIT_DIR"] = str(REPO_ROOT)
    env["SLURM_ARRAY_TASK_ID"] = "0"
    env["BLLARSE_SIF"] = str(sif)
    env["BLLARSE_SWEEP_SOURCE"] = "bllarse_sweeps/probe_smoke.py"
    if set_vars:
        env.update(set_vars)

    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return capture_file.read_text().splitlines()


def test_a_set_env_var_is_forwarded_to_apptainer(tmp_path):
    argv = _run_job_script(tmp_path, {"HF_TOKEN": "secret-token"})
    assert "HF_TOKEN=secret-token" in argv
    assert argv[argv.index("HF_TOKEN=secret-token") - 1] == "--env"


def test_an_unset_env_var_is_never_forwarded_not_even_empty(tmp_path):
    """Regression for `--env "VAR=${VAR:-}"`: with the key present but empty,
    os.environ.get("VAR", default) returns '' instead of falling back to
    `default`, defeating every Python-side default. The fix must omit the
    flag entirely rather than pass an empty value."""
    argv = _run_job_script(tmp_path, {"HF_TOKEN": "secret-token"})
    for var in CANDIDATE_VARS:
        if var == "HF_TOKEN":
            continue
        assert not any(a.startswith(f"{var}=") for a in argv), (
            f"{var} was never set and must not be passed, not even as '{var}='"
        )


def test_all_candidate_vars_unset_means_none_of_them_are_forwarded(tmp_path):
    argv = _run_job_script(tmp_path, {})
    for var in CANDIDATE_VARS:
        assert not any(a.startswith(f"{var}=") for a in argv)
    # The always-on vars must still be present.
    assert any(a.startswith("BLLARSE_REPO_ROOT=") for a in argv)
    assert any(a.startswith("BLLARSE_SWEEP_SOURCE=") for a in argv)
