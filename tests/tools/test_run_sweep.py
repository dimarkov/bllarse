import subprocess

import pytest

import bllarse.tools.run_sweep as run_sweep_module


def _write_sweep(tmp_path, configs):
    source = tmp_path / "sweep.py"
    source.write_text(
        "def create_configs():\n"
        f"    return {configs!r}\n"
        "\n"
        "def run(config):\n"
        "    return None\n"
    )
    return source


def _write_job_script(tmp_path):
    script = tmp_path / "job.sh"
    script.write_text("#!/usr/bin/env bash\n")
    return script


def test_run_sweep_submits_chunk_with_expected_environment(tmp_path, monkeypatch):
    source = _write_sweep(tmp_path, [{"seed": 0}, {"seed": 1}, {"seed": 2}])
    script = _write_job_script(tmp_path)
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_sweep_module.subprocess, "run", fake_run)

    run_sweep_module.run_sweep(
        source,
        job_script=script,
        index_offset=1,
        num_jobs=2,
        max_concurrent=3,
        cpus_per_task=8,
        exclude_nodes="node4",
        job_name="demo",
    )

    assert captured["command"] == [
        "sbatch",
        "--export=ALL",
        "--array=0-1%3",
        "--job-name=demo",
        "--cpus-per-task=8",
        "--exclude=node4",
        str(script.resolve()),
    ]
    env = captured["kwargs"]["env"]
    assert env["BLLARSE_SWEEP_SOURCE"] == str(source.resolve())
    assert env["INDEX_OFFSET"] == "1"
    assert env["VENV_NAME"] == ".venv"
    assert (tmp_path / "slurm_logs").is_dir()


def test_run_sweep_reuses_mlflow_parent_without_creating_one(tmp_path, monkeypatch):
    source = _write_sweep(tmp_path, [{"enable_mlflow": True}])
    script = _write_job_script(tmp_path)
    captured = {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        run_sweep_module,
        "_create_mlflow_parent",
        lambda *args, **kwargs: pytest.fail("parent should be reused"),
    )

    def fake_run(command, **kwargs):
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_sweep_module.subprocess, "run", fake_run)

    run_sweep_module.run_sweep(
        source,
        job_script=script,
        parent_run_id="parent-123",
    )

    assert captured["env"]["MLFLOW_PARENT_RUN_ID"] == "parent-123"


def test_run_sweep_does_not_submit_when_mlflow_parent_creation_fails(
    tmp_path,
    monkeypatch,
):
    source = _write_sweep(tmp_path, [{"enable_mlflow": True}])
    script = _write_job_script(tmp_path)
    monkeypatch.chdir(tmp_path)

    def fail_parent(*args, **kwargs):
        raise RuntimeError("tracking unavailable")

    monkeypatch.setattr(run_sweep_module, "_create_mlflow_parent", fail_parent)
    monkeypatch.setattr(
        run_sweep_module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("sbatch must not be called"),
    )

    with pytest.raises(RuntimeError, match="tracking unavailable"):
        run_sweep_module.run_sweep(source, job_script=script)


def test_dry_run_has_no_mlflow_or_submission_side_effects(tmp_path, monkeypatch):
    source = _write_sweep(tmp_path, [{"enable_mlflow": True}])
    script = _write_job_script(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        run_sweep_module,
        "_create_mlflow_parent",
        lambda *args, **kwargs: pytest.fail("dry-run must not create a parent"),
    )
    monkeypatch.setattr(
        run_sweep_module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("dry-run must not call sbatch"),
    )

    result = run_sweep_module.run_sweep(source, job_script=script, dry_run=True)

    assert result is None
    assert not (tmp_path / "slurm_logs").exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"max_concurrent": 0}, "max_concurrent"),
        ({"cpus_per_task": 0}, "cpus_per_task"),
        ({"index_offset": 3}, "index_offset"),
        ({"num_jobs": 0}, "num_jobs"),
        ({"index_offset": 2, "num_jobs": 2}, "exceeds"),
    ),
)
def test_run_sweep_validates_array_bounds(tmp_path, kwargs, message):
    source = _write_sweep(tmp_path, [{"seed": 0}, {"seed": 1}, {"seed": 2}])
    script = _write_job_script(tmp_path)

    with pytest.raises(ValueError, match=message):
        run_sweep_module.run_sweep(source, job_script=script, **kwargs)
