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
                "BLLARSE_FT_STORAGE", "BLLARSE_FT_VENV", "BLLARSE_FT_SHARDS"):
        assert var in text


def test_it_checks_the_training_shard_before_taking_a_gpu(text):
    """The one failure in this campaign that is slow rather than wrong.

    Without a shard finetune_jax decodes JPEGs and every trial still completes
    and is still scored -- at ~40 img/s against ~220, so a 24 h worker finishes
    1 trial instead of 7 and the campaign silently takes five times as long.
    Nothing in the results would look off.
    """
    assert "shard_meta=" in text
    assert 'if [[ ! -f "$shard_meta" ]]; then' in text
    assert "exit 2" in text


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


def test_the_job_forwards_the_layer_decay_span():
    """A campaign spanning backbones passes a span, not a per-block factor.

    Without this the span reaches the launcher and stops there, and the run
    trains with no LLRD at all -- which looks like a successful campaign and
    answers a different question.
    """
    body = SCRIPT.read_text()
    assert "BLLARSE_FT_LAYER_DECAY_SPAN" in body
    assert "--layer-decay-span" in body


def test_span_and_per_block_decay_are_forwarded_independently():
    # They are mutually exclusive at the script, so the job must not collapse
    # one into the other or a caller setting only the span would get both.
    body = SCRIPT.read_text()
    span_at = body.index("--layer-decay-span")
    decay_at = body.index('--layer-decay "')
    assert span_at != decay_at


# --- The arena that fragments across a long campaign ------------------------


def test_it_runs_the_worker_in_chunks_rather_than_one_long_process(text):
    """The OOM that cost 28 of ft_ce_small100k's 273 trials.

    Every trial in a campaign asks for identical memory -- the micro batch is
    pinned and the eval chunk is fixed -- so a trial that cannot allocate at #32
    when #1 could is a fragmented arena, not a bigger trial. All six workers of
    that study died, on six different nodes, at wall-clock times two hours
    apart, after 32/33/39 trials, on the same 6.55 GiB in jit_ce_micro_grad.
    Only a per-process cause fits that pattern, and only a fresh process clears
    it.
    """
    assert "BLLARSE_FT_TRIALS_PER_PROC" in text
    # The whole worker budget must NOT reach one process any more.
    assert '--n-trials "$BLLARSE_FT_TRIALS"' not in text
    assert '--n-trials "$n"' in text
    assert "while (( remaining > 0 )); do" in text


def test_the_chunk_loop_cannot_be_an_exec(text):
    # `exec` replaces the shell: with it there is no loop to come back to, and
    # the restart silently becomes a no-op.
    assert "exec srun" not in text


def test_it_reads_the_chunk_exit_code_from_pipestatus(text):
    # The worker is piped through tee to keep the log live during a ~22 min
    # trial, so `$?` is tee's status and would report every failed chunk as a
    # success.
    assert "PIPESTATUS[0]" in text


def test_it_stops_starting_chunks_once_the_study_is_at_depth(text):
    """Otherwise every remaining chunk pays a process start to learn nothing.

    The token is printed by finetune_worker.main; the two are coupled on
    purpose and there is a test on the worker side naming this one.
    """
    assert 'grep -q "TARGET-MET"' in text
    assert "break" in text


def test_two_consecutive_failed_chunks_end_the_job(text):
    """A restart is the remedy for fragmentation, not for a broken arm.

    Two fresh arenas in a row cannot both be fragmented, and
    abort_on_consecutive_failures has already burned a run of trials proving
    it, so the budget must not be spent confirming it a third time.
    """
    assert "failed_chunks >= 2" in text
