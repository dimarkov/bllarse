"""The CUDA graph-capture flag must be on the SWEEP job script, not only the
tuning one.

`slurm_tune.sh` gained this fix after it killed 13 of 56 tuning tasks. It was
never carried across to `slurm_run_config_apptainer.sh`, which is what Fig 1 and
Fig 2 actually run under, so every probing campaign stayed exposed -- it killed
2 of 128 tasks in the win5 submission on 2026-08-20.
"""
from pathlib import Path

import pytest

SCRIPTS = [
    Path("src/slurm/jobs/slurm_run_config_apptainer.sh"),
    Path("src/slurm/jobs/slurm_tune.sh"),
]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_command_buffers_are_disabled_in_live_code(script):
    """In the live `apptainer exec` invocation, not merely mentioned in prose:
    comments explaining the fix must not be able to satisfy the assertion."""
    text = script.read_text()
    live = "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("#")
    )
    assert '--env "XLA_FLAGS=--xla_gpu_enable_command_buffer="' in live, (
        f"{script.name} is missing the CUDA graph-capture workaround"
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_the_flag_is_unconditional(script):
    """Must sit in the unconditional part of env_args(), not behind one of the
    `[[ -n ... ]] &&` guards used for optional variables below it."""
    live = "\n".join(
        line for line in script.read_text().splitlines()
        if not line.strip().startswith("#")
    )
    block = live.split("env_args=(", 1)[1].split("\n)", 1)[0]
    assert "XLA_FLAGS=--xla_gpu_enable_command_buffer=" in block
