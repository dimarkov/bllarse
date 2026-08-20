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
    assert int(_default("BLLARSE_FT_OOD_EVERY_K")) == 8
    assert "--ood-every-k-dumps" in JOB and "--dump-every-k-evals" in JOB


def test_the_ood_stride_leaves_the_job_inside_the_wall():
    """Arithmetic, not taste. One OOD pass is the whole suite preloaded.

    At the default quantum a 4-epoch run has 78 evaluations; dumping every one
    and striding OOD by 8 gives 9 passes. Measured eval throughput is ~0.87M
    forwards/hour, and the job's wall is 24 h with ~17.9 h of training and
    ~4.5 h of val+test already spent.
    """
    from bllarse.finetune import imagenet as im

    q = int(_default("BLLARSE_FT_EVAL_QUANTUM"))
    final_steps = protocol.steps_for_samples(int(4 * 1_281_167), fs.FIXED_BATCH_SIZE)
    n_evals = final_steps // (q // fs.FIXED_BATCH_SIZE)
    n_ood = n_evals // int(_default("BLLARSE_FT_OOD_EVERY_K"))
    hours = n_ood * sum(im.OOD_SIZES.values()) / 0.87e6
    assert hours < 2.0, f"OOD would add {hours:.1f} h; the wall has ~1.6 h spare"


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
