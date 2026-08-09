"""The fine-tuning SLURM entry point.

A submit-time typo here costs a whole array of A100 allocations that die on
their first line, so the contract is pinned rather than trusted.
"""

import os
import stat
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "src/slurm/jobs/slurm_finetune_tune.sh"


@pytest.fixture(scope="module")
def text():
    return SCRIPT.read_text()


def test_the_script_exists_and_is_executable():
    assert SCRIPT.is_file()
    assert os.stat(SCRIPT).st_mode & stat.S_IXUSR


def test_it_is_a_login_shell(text):
    # On MPCDF the `module` function only exists once /etc/profile.d has been
    # sourced, which a non-login shell skips.
    assert text.startswith("#!/bin/bash -l")


def test_it_requests_a_gpu_by_gres_not_dash_g(text):
    # Raven's default partition is `interactive`, which has no GPUs, so a bare
    # `-G 1` is rejected at submit time and the array never queues.
    assert "--gres=gpu:a100:1" in text
    assert "\n#SBATCH -G " not in text


def test_it_asks_for_a_full_day(text):
    # Sized against the FULL trial cost (~6.3 h), not the hoped-for average: a
    # 24 h job fits 3 trials.
    assert "--time=24:00:00" in text


def test_it_passes_patience_twenty_by_default(text):
    assert '--patience "${BLLARSE_FT_PATIENCE:-20}"' in text


def test_it_runs_the_finetuning_worker_not_the_probing_one(text):
    assert "bllarse.tuning.finetune_worker" in text
    assert "bllarse.tuning.worker " not in text


def test_it_puts_the_repo_src_ahead_of_the_venv(text):
    # The venv carries an editable install of an OLDER checkout under the same
    # package name, so without this shadowing every worker silently imports
    # the wrong bllarse.
    assert 'export PYTHONPATH="${repo_root}/src' in text


def test_it_does_not_disable_xla_command_buffers(text):
    """The opposite of the probing job script, on purpose.

    slurm_tune.sh must disable command buffers because the probing IMAGE pins
    XLA_PYTHON_CLIENT_ALLOCATOR=platform and CUDA-graph capture collides with
    it. This path runs in a venv with no such allocator, and disabling them
    measured 18.29 img/s against 40.27 img/s enabled -- a 2.2x tax for a
    collision that cannot happen here.

    The flag is a property of the ALLOCATOR, not of the model: if this ever
    moves into a container that pins that allocator, it has to come back.
    """
    active = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    assert "xla_gpu_enable_command_buffer" not in active


def test_it_refuses_to_start_without_its_required_variables(text):
    for var in ("BLLARSE_FT_ARM", "BLLARSE_FT_TRIALS",
                "BLLARSE_FT_STORAGE", "BLLARSE_FT_VENV"):
        assert var in text


def test_it_resolves_symlinked_paths(text):
    # /ptmp is a top-level symlink to /raven/ptmp; an unresolved path has
    # repeatedly cost a job by existing on the host and not where the process
    # actually looks.
    assert 'readlink -f' in text


def test_it_unbuffers_python_stdout(text):
    # Python block-buffers stdout to a file, so progress lines sit in an 8 KB
    # buffer instead of the log. On a ~6 h trial that makes a hung job
    # indistinguishable from a slow one.
    assert "PYTHONUNBUFFERED=1" in text


def test_it_configures_a_persistent_compilation_cache(text):
    # Compiling the ViT-H fwd+bwd graph measured ~19 min, and finetune_jax
    # clears caches in its `finally`, so each of a worker's 3 trials pays it
    # again. Every trial has identical shapes (pinned micro batch, fixed eval
    # chunk), so one cache entry serves the whole campaign.
    assert "JAX_COMPILATION_CACHE_DIR" in text
    assert "mkdir -p" in text
