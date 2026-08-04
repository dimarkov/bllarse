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


def test_it_resolves_the_cache_dir_through_symlinks():
    """/ptmp is a host symlink to /raven/ptmp. Apptainer binds the RESOLVED
    target and does not recreate the symlink inside the container, so an
    unresolved /ptmp/... cache dir is a read-only stub in there. That does not
    fail at startup: load_cached_features just misses the cache, falls back to
    the Hub, and dies on mkdir with Errno 30 once per trial -- a whole campaign
    of tasks that run, burn GPU time and complete zero trials. The script must
    resolve it itself rather than trusting whatever the launcher exported."""
    text = SCRIPT.read_text()
    assert 'cache_dir="$(readlink -f' in text
    assert '--cache-dir "$cache_dir"' in text
    # The raw variable must not reach the worker directly.
    assert '--cache-dir "${BLLARSE_CACHE_DIR' not in text


def test_it_forwards_an_optional_campaign_suffix():
    """The worker opens studies with load_if_exists=True, so a re-run under a
    CHANGED search space must namespace its study or it silently resumes the
    old one. The suffix has to be OPTIONAL -- an unset variable must reproduce
    the first campaign's unsuffixed names rather than passing an empty string,
    which would create a study called `vbll_food101_`."""
    text = SCRIPT.read_text()
    assert "BLLARSE_TUNE_CAMPAIGN" in text
    assert "--campaign" in text
    # Guarded by a non-empty test, not interpolated unconditionally.
    assert 'if [[ -n "${BLLARSE_TUNE_CAMPAIGN:-}" ]]; then' in text
    # ...and it must NOT be in the required-variable list, which exits 2.
    required = text[text.index("for var in"):text.index("done", text.index("for var in"))]
    assert "BLLARSE_TUNE_CAMPAIGN" not in required


def test_it_passes_a_patience_of_20_by_default():
    """20 is the smallest patience at which none of the 28 real pilot curves
    stop before their own val_nll argmin (15 stops one early, 8 stops two
    early). A trial stopped before its argmin is scored at the wrong point,
    which would change which hyperparameters the campaign selects."""
    text = SCRIPT.read_text()
    assert "--patience" in text
    assert "BLLARSE_TUNE_PATIENCE:-20" in text


def test_it_disables_cuda_graph_command_buffers_for_the_jax_arm():
    """13/14 tuning studies died on large-batch trials (batch 8192, 2048) with
    `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED: operation not permitted when stream
    is capturing` inside jit__ce_step, followed by a fatal cublasLtMatmul
    INTERNAL error -- XLA's CUDA-graph command-buffer capture colliding with
    `XLA_PYTHON_CLIENT_ALLOCATOR=platform` (set in bllarse-jax.def). The flag
    must be present in the script's live `apptainer exec` invocation, not just
    mentioned in a comment, and it must be unconditional -- it is harmless for
    the torch/VBLL image, which never touches XLA."""
    text = SCRIPT.read_text()

    # Strip comment lines so a mention in prose (e.g. this file's own header
    # explaining the fix) cannot satisfy the assertion -- the flag must appear
    # in code that actually runs.
    live_lines = [
        line for line in text.splitlines() if not line.strip().startswith("#")
    ]
    live_text = "\n".join(live_lines)

    assert '--env "XLA_FLAGS=--xla_gpu_enable_command_buffer="' in live_text
    # Must land in the unconditional part of env_args() -- not behind one of
    # the `[[ -n ... ]] &&` guards used for optional vars below it -- so split
    # on comment-stripped text where parens genuinely close the array.
    assert "env_args=(" in live_text
    env_args_block = live_text.split("env_args=(", 1)[1].split(")", 1)[0]
    assert "XLA_FLAGS=--xla_gpu_enable_command_buffer=" in env_args_block
