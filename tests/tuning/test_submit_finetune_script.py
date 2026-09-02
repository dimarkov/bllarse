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


class _Recorder:
    """Minimal optuna Trial stand-in: records the names actually suggested."""

    def __init__(self):
        self.names: list[str] = []

    def suggest_float(self, name, low, high, log=False):
        self.names.append(name)
        return low

    def suggest_categorical(self, name, choices):
        self.names.append(name)
        return choices[0]


def _recorded_axes(arm: str) -> list[str]:
    trial = _Recorder()
    fs.suggest(trial, arm)
    return trial.names


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
    #
    # fixb512 joins it for the same reason one step later. lr and weight decay
    # are now searched as `lr_per_sqrt_batch` and `weight_decay_per_sqrt_batch`
    # -- the batch-invariant coordinates -- and a RENAMED float axis is
    # optuna's quiet failure rather than its loud one: the study loads, TPE
    # keeps modelling `lr` from trials that carry it, and every new trial looks
    # like it simply never set the axis.
    assert campaign not in ("shards2m", "fixb512", "sdelr512", "betas1m")


def test_the_campaign_name_is_bumped_whenever_the_searched_names_change():
    """The guard above, tied to the thing that actually forces a rename.

    Asserting a denylist of old names rots: the next search-space change adds a
    third and nothing reminds anyone. This pins the CURRENT default against the
    space it belongs to, so renaming an axis without renaming the campaign
    fails here rather than three hours into the first trial.
    """
    searched = set()
    for arm in fs.ARMS:
        searched |= set(_recorded_axes(arm))
    assert {"lr_per_sqrt_batch", "weight_decay_per_sqrt_batch",
            "beta1_horizons", "beta2_horizons"} <= searched
    assert _default("BLLARSE_FT_CAMPAIGN") == "betasearch1m"


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


def test_it_refuses_a_venv_directory_before_taking_a_node():
    """`BLLARSE_FT_VENV` is the INTERPRETER, not the venv.

    The job script runs "$BLLARSE_FT_VENV" -m bllarse.tuning.finetune_worker,
    so a directory fails with `Is a directory` and exit 126 -- after SLURM has
    allocated a GPU. One bad path multiplies by workers x arms: it took out all
    48 tasks of a submission in under 3 seconds, and left the arrays exhausted
    so the campaign had to be resubmitted from scratch.
    """
    assert "check_interpreter" in TEXT
    assert 'check_interpreter "$BLLARSE_FT_VENV"' in TEXT
    assert 'check_interpreter "$BLLARSE_FT_VBLL_VENV"' in TEXT
    # Names the fix, not just the fault.
    assert "bin/python" in TEXT


def test_it_checks_the_shard_before_taking_a_node_too():
    # Same class: slurm_finetune_tune.sh exits 2 on a missing shard, but only
    # once it already holds a GPU.
    assert "imagenet1k_train_224.json" in TEXT


def test_it_resolves_the_dataset_cache_before_taking_a_node():
    """The check whose absence cost the betas1m campaign.

    BLLARSE_HF_HOME pointed at a cache with no PREPARED imagenet-1k, so all 18
    workers called load_dataset at once, raced to build it into the same
    directory and destroyed each other's `.incomplete` staging dir. 48 array
    tasks, 410 GPU-hours, 9 usable trials out of 2,860.

    The existing shard check could not catch it: the shard covers TRAINING, and
    finetune_jax still calls load_dataset for the eval split and the label
    metadata, so a valid shard says nothing about whether the dataset resolves.
    """
    assert "check_dataset" in TEXT
    # OFFLINE is the load-bearing part. A cache that would have to DOWNLOAD is
    # exactly the one that must not be handed to 18 concurrent workers.
    assert "HF_DATASETS_OFFLINE=1" in TEXT
    assert "HF_HUB_OFFLINE=1" in TEXT
    # No escape hatch: the campaign it protects costs orders of magnitude more
    # than the ~3 minutes it takes.
    assert "SKIP_DATASET" not in TEXT


def test_the_dataset_check_covers_every_interpreter_in_use():
    """`datasets` can differ in version between the two venvs.

    The prepared-dataset path is keyed by a builder revision hash, so one cache
    can resolve for the JAX interpreter and trigger a rebuild for the torch one.
    vbll runs its own venv and is the arm that took 0 completions out of 960, so
    checking only BLLARSE_FT_VENV would have missed exactly the worst case.
    """
    block = TEXT[TEXT.index("declare -A _checked"):]
    assert 'BLLARSE_FT_VBLL_VENV' in block
    assert 'check_dataset "$interp"' in block


def test_the_dataset_check_reads_the_path_from_the_registry():
    # Hardcoding "ILSVRC/imagenet-1k" here would be a second place for the
    # dataset identity to live, and the two would drift.
    assert "DATASET_CONFIGS" in TEXT
    assert "hf_path" in TEXT
    assert "ILSVRC/imagenet-1k" not in TEXT


def test_it_forwards_the_per_process_trial_chunk():
    """The size of the fresh-process chunk is a campaign setting, not a default.

    The job script defaults it to 10, which is right for the ~22 min trials the
    small rungs measured; a rung whose trials run hours wants a smaller chunk,
    and a launcher that cannot set it would have to edit the job script to say
    so.
    """
    assert "BLLARSE_FT_TRIALS_PER_PROC" in TEXT
