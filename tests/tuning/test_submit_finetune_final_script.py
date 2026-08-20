"""The REPORTED campaign's launcher.

Checked as text, like its tuning sibling: it submits to SLURM, and every
property worth guarding is a property of what it would submit. The stakes are
higher here than for a search -- these 15 runs are the paper's Figs 5-7 and
Table 1, so a wrong flag is a wrong result rather than a wasted trial.
"""
import os
from pathlib import Path

import pytest

from bllarse.finetune import protocol
from bllarse.tuning import final_config as fc
from bllarse.tuning import finetune_space as fs

SCRIPT = Path("src/slurm/submit_finetune_final.sh")
TEXT = SCRIPT.read_text()
JOB = Path("src/slurm/jobs/slurm_finetune_final.sh").read_text()


def _default(name: str) -> str:
    body = TEXT[TEXT.index(f"{name}:="):]
    return body[len(name) + 2:body.index("}")]


def test_both_scripts_are_executable():
    assert os.access(SCRIPT, os.X_OK)
    assert os.access(Path("src/slurm/jobs/slurm_finetune_final.sh"), os.X_OK)


def test_it_reads_a_finished_campaign_rather_than_starting_one():
    # This launcher CONSUMES a search. Naming a fresh campaign would create
    # empty studies and report from whatever the first trial happened to be.
    assert _default("BLLARSE_FT_CAMPAIGN") == "betasearch1m"
    assert "check_study" in TEXT


def test_it_refuses_a_study_that_has_not_reached_its_target():
    """A half-finished search still has a leader, and freezing it is silent.

    Nothing downstream distinguishes "best of 90 completions" from "best of 12
    so far" -- the config looks identical -- so the refusal has to happen here.
    """
    assert "under the target" in TEXT
    assert "n_completed" in TEXT


def test_one_task_per_seed_and_the_index_is_not_clamped():
    """A clamped index would silently run a duplicate seed.

    Five identical runs reported as an error bar is exactly the failure the
    seed plumbing was added to prevent, and clamping would reintroduce it one
    layer down.
    """
    assert "SLURM_ARRAY_TASK_ID" in JOB
    assert "exceeds" in JOB and "seeds" in JOB
    assert '--array "0-$((n_seeds - 1))' in TEXT


def test_the_seed_list_matches_the_results_plan_not_the_seed_pool():
    """results-plan 0.5 says FIVE seeds for full fine-tuning.

    finetune_space.REPORTED_SEEDS holds ten; that is a pool to draw from, and
    the difference is 320 vs 640 GPU-hours on the plan's largest line item.
    """
    seeds = _default("BLLARSE_FT_SEEDS").split()
    assert len(seeds) == 5
    assert [int(s) for s in seeds] == list(fs.REPORTED_SEEDS[:5])


def test_the_evaluation_grid_is_coarser_than_tunings():
    """Wall clock, not preference.

    val+test is 50k forwards per evaluation against tuning's 10k, over 2.56x
    the samples. At protocol.EVAL_QUANTUM that is 312 evaluations and roughly
    doubles the run, putting a dinov3_huge job near 36 h against a 24 h limit.
    """
    q = int(_default("BLLARSE_FT_EVAL_QUANTUM"))
    assert q > protocol.EVAL_QUANTUM
    assert q % fs.FIXED_BATCH_SIZE == 0, "unschedulable: not a whole number of steps"
    final_steps = protocol.steps_for_samples(int(4 * 1_281_167), fs.FIXED_BATCH_SIZE)
    n_evals = final_steps // (q // fs.FIXED_BATCH_SIZE)
    assert 40 <= n_evals <= 160, f"{n_evals} evaluations is a bad curve or a long job"


def test_it_reports_on_huge_for_four_epochs():
    assert _default("BLLARSE_FT_MODEL") == "dinov3_huge"
    assert float(_default("BLLARSE_FT_EPOCHS")) == fc.FINAL_EPOCHS == 4.0


def test_the_task_derives_its_own_config_instead_of_being_templated():
    # The transfer must not be hand-copied into a shell command; that is the
    # confound bllarse.tuning.final_config exists to remove.
    assert "final_config.py" in JOB
    assert "--format exec" in JOB
    assert "--seed" in JOB


def test_the_default_arms_are_the_three_that_actually_train():
    # CE + temperature scaling is a POST-HOC fit on the CE run's dumps, so it
    # is a comparator without being a training run.
    arms = _default("BLLARSE_FT_ARMS").split()
    assert arms == ["ce", "ibprobit_noeps", "vbll"]
    for a in arms:
        assert a in fs.ARMS


def test_it_keeps_the_tuning_launchers_pre_flights():
    # Same failure modes, same guards: a venv directory and an unresolvable
    # dataset cache each cost a whole submission once already.
    for token in ("check_interpreter", 'check_interpreter "$BLLARSE_FT_VENV"',
                  "HF_DATASETS_OFFLINE=1", "imagenet1k_train_224.json"):
        assert token in TEXT


def test_the_two_frameworks_keep_separate_interpreters():
    assert "BLLARSE_FT_VBLL_VENV" in TEXT
    assert 'arm" == "vbll" ]] && venv=' in TEXT


def test_it_supports_a_dry_run():
    assert "BLLARSE_FT_DRY_RUN" in TEXT


def test_it_exports_all_not_only_the_named_variables():
    assert '--export "ALL,' in TEXT


# --- OOD ---------------------------------------------------------------------


def test_the_reported_campaign_actually_enables_dumps_and_ood():
    """Both, because finetune_jax gates on both being nonzero.

    trial_config sets each to 0 -- right for tuning, where a trial writes one
    argmin dump and must never pay the OOD preload. Inheriting that silently
    gave the reported campaign no trajectory (Fig 5) and no OOD (Fig 6) at all.
    """
    assert int(_default("BLLARSE_FT_DUMP_EVERY_K")) >= 1
    assert int(_default("BLLARSE_FT_OOD_EVERY_K")) >= 1
    assert "--ood-every-k-dumps" in JOB and "--dump-every-k-evals" in JOB


def test_the_trainer_pays_no_ood_forwards_at_all():
    """What delegation is FOR, stated as the arithmetic it replaces.

    At the current stride a 4-epoch run has 78 evaluations and 39 OOD passes.
    Inline, at the measured ~0.87M forwards/hour, that is ~4.4 h of the suite
    on the TRAINING gpu -- on a job already near 23 h against a 24 h wall. The
    trainer now writes a checkpoint instead, and the only cost it keeps is that
    write.
    """
    from bllarse.finetune import imagenet as im

    q = int(_default("BLLARSE_FT_EVAL_QUANTUM"))
    final_steps = protocol.steps_for_samples(int(4 * 1_281_167), fs.FIXED_BATCH_SIZE)
    n_evals = final_steps // (q // fs.FIXED_BATCH_SIZE)
    n_ood = n_evals // int(_default("BLLARSE_FT_OOD_EVERY_K"))
    inline_hours = n_ood * sum(im.OOD_SIZES.values()) / 0.87e6
    assert inline_hours > 2.0, (
        "this stride would fit inline, so delegation is not carrying its weight"
    )
    # Delegated instead: the flag that moves it off the training gpu.
    assert "--ood-checkpoint-root" in JOB
    assert "BLLARSE_FT_CKPT_ROOT" in TEXT


def test_the_checkpoint_write_the_trainer_keeps_is_negligible():
    """The cost delegation ADDS, bounded rather than assumed away.

    A dinov3_huge checkpoint is ~3.4 GB (840M fp32 params) and the stride
    writes one per OOD pass. At GPFS write speeds that is minutes against a
    ~22 h job -- but peak DISK is what needs bounding, and --ood-max-pending
    does it: at most that many slots per run are ever resident.
    """
    q = int(_default("BLLARSE_FT_EVAL_QUANTUM"))
    final_steps = protocol.steps_for_samples(int(4 * 1_281_167), fs.FIXED_BATCH_SIZE)
    n_ood = (final_steps // (q // fs.FIXED_BATCH_SIZE)) // int(
        _default("BLLARSE_FT_OOD_EVERY_K"))
    gb_per_ckpt = 840e6 * 4 / 1e9
    write_minutes = n_ood * gb_per_ckpt / 1.0 / 60      # ~1 GB/s
    assert write_minutes < 15, f"{write_minutes:.0f} min of checkpoint writes"

    max_pending = int(_default("BLLARSE_FT_OOD_MAX_PENDING"))
    n_runs = len(_default("BLLARSE_FT_SEEDS").split()) * len(
        _default("BLLARSE_FT_ARMS").split())
    peak_gb = max_pending * gb_per_ckpt * n_runs
    assert peak_gb < 200, f"peak checkpoint residency {peak_gb:.0f} GB"


def test_the_ood_preload_fits_the_jobs_memory():
    # load_ood_sets holds every set as uint8 host arrays for the whole run,
    # competing with the shard reader and JAX inside --mem=125000.
    from bllarse.finetune import imagenet as im

    assert im.OOD_PRELOAD_BYTES / 1e9 < 40, "preload would crowd the allocation"
    assert "--mem=125000" in JOB


def test_imagenet_c_stays_out_and_says_why():
    """A recorded decision, not an omission.

    Only severity 5 is available (so "severity-averaged" is unobtainable), and
    8 corruptions x 50,000 is 400,000 images per pass -- ~60 GB of host arrays
    and ~5.7 h of evaluation. Subsampling would fit but is what this suite is
    defined to avoid.
    """
    from bllarse.finetune import imagenet as im

    assert not any(k.startswith("imagenet_c") for k in im.OOD_KEYS)
    assert "ImageNet-C stays excluded" in im.__doc__
    assert "severity 5 ALONE" in im.__doc__


def test_sketch_is_in_the_suite_and_is_not_subset_masked():
    # 1000 classes, so it is scored on the full label space -- unlike R and A,
    # whose 200-class mask is an oracle intervention on the very distribution
    # whose calibration is being measured.
    from bllarse.finetune import imagenet as im

    assert "imagenet_sketch" in im.OOD_KEYS
    assert "imagenet_sketch" not in im.OOD_SUBSET_KEYS
    assert im.OOD_SUBSET_KEYS == {"imagenet_r", "imagenet_a"}


# --- The decoupled OOD worker ------------------------------------------------

WORKER = Path("src/slurm/jobs/slurm_ood_worker.sh").read_text()


def test_ood_is_delegated_to_workers_not_run_on_the_training_gpu():
    """The whole point: OOD costs the trainer a checkpoint write, nothing more.

    Inline it was ~1 h per run and pushed a dinov3_huge 4-epoch job past its
    24 h wall.
    """
    assert "BLLARSE_FT_CKPT_ROOT" in TEXT
    assert "--ood-checkpoint-root" in JOB
    assert "--ood-max-pending" in JOB
    assert "ood_worker.py" in WORKER


def test_the_stride_is_finer_now_that_it_is_off_the_critical_path():
    # 78 evaluations at the default quantum -> 39 OOD passes. Inline that would
    # have been ~4.5 h per run on a job already near 23 h.
    assert int(_default("BLLARSE_FT_OOD_EVERY_K")) == 2


def test_one_worker_per_framework_because_the_venvs_cannot_be_merged():
    """jax[cuda] and torch each ship a full CUDA stack.

    Each worker polls the same root and consumes only slots whose metadata
    names its own framework, so neither can eat the other's checkpoints.
    """
    assert "BLLARSE_OOD_FRAMEWORK" in WORKER
    assert "--framework" in WORKER
    assert "submit_workers jax" in TEXT
    assert "submit_workers torch" in TEXT
    assert 'BLLARSE_FT_VBLL_VENV" "$BLLARSE_OOD_WORKERS_TORCH' in TEXT


def test_workers_are_submitted_before_the_training_arrays():
    # So they are draining by the time the first checkpoint lands, rather than
    # queueing behind 15 training jobs.
    assert TEXT.index("submit_workers jax") < TEXT.index('exports="BLLARSE_FT_ARM=')


def test_no_workers_are_submitted_when_ood_is_off():
    # An idle gpu is worse than no gpu.
    block = TEXT[TEXT.index("if (( BLLARSE_FT_OOD_EVERY_K"):]
    assert "BLLARSE_FT_DUMP_EVERY_K > 0" in block.split("\n")[0]


def test_the_torch_worker_is_skipped_when_vbll_is_not_in_the_arms():
    assert "grep -qw vbll" in TEXT


def test_the_launcher_gates_on_the_ood_cache_being_ready():
    """Only ImageNet-A was ever cached.

    15 trainers plus their workers racing to download the rest is the exact
    shape of the betas1m loss, and the existing dataset pre-flight only covers
    imagenet-1k.
    """
    assert "prepare_ood_cache.py" in TEXT
    assert "--check-only" in TEXT
    gate = TEXT[TEXT.index("OOD cache:"):]
    assert gate.index("exit 2") < gate.index("submit_workers")


def test_the_worker_gets_the_same_memory_as_a_trainer():
    # It holds the 98,389-image suite as uint8 host arrays (~14.8 GB) for its
    # whole lifetime, same as an inline trainer would have.
    assert "--mem=125000" in WORKER
    assert "gres=gpu:a100:1" in WORKER


def test_the_worker_exits_rather_than_idling_forever():
    assert "--idle-exit-seconds" in WORKER
    assert "BLLARSE_OOD_IDLE_EXIT" in WORKER
