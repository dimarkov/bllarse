"""The arm script's pure planning layer: budget, accumulation, guards.

Everything here runs without a GPU or a dataset, so a misconfigured sweep fails
at submit time rather than after a node allocation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "finetune"))

import finetune_jax as fj  # noqa: E402


def _args(**overrides):
    argv = ["--arm", "ce", "--batch-size", "1024", "--micro-batch-size", "256"]
    for key, value in overrides.items():
        flag = f"--{key.replace('_', '-')}"
        # store_true flags take no value; passing one is an argparse error.
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    return fj.build_argparser().parse_args(argv)


def test_default_epochs_is_the_four_epoch_cap():
    assert _args().epochs == 4.0


def test_default_img_size_is_224():
    assert _args().img_size == 224


def test_check_img_size_refuses_512():
    # DINOV3_DATASET_IMG_SIZES maps imagenet1k -> 512 for FEATURE EXTRACTION.
    # Inheriting it here would be a ~5x cost accident, so it must be loud.
    # The pattern pins the interpolated size in its actual context, not just the
    # boilerplate "imagenet1k -> 512", so if interpolation breaks the test fails.
    with pytest.raises(SystemExit, match=r"--img-size 512 refused"):
        fj.check_img_size(_args(img_size=512))


def test_check_img_size_refuses_384():
    # A different non-224 size, proving the guard is not hardcoded to 512 and
    # that the guard actually interpolates the supplied size in the message.
    with pytest.raises(SystemExit, match=r"--img-size 384 refused"):
        fj.check_img_size(_args(img_size=384))


def test_check_img_size_accepts_224():
    fj.check_img_size(_args(img_size=224))  # must not raise


def test_check_img_size_allows_an_explicit_override():
    fj.check_img_size(_args(img_size=512, allow_non_224=True))


def test_resolve_plan_derives_steps_from_the_effective_batch():
    plan = fj.resolve_plan(_args(batch_size=1024), micro_batch=64)
    assert plan["effective_batch"] == 1024
    assert plan["num_accum"] == 16
    assert plan["steps"] == 5_004


def test_resolve_plan_refuses_a_micro_batch_that_does_not_divide_the_target():
    # This used to round -- 4096 over a micro batch of 100 became an effective
    # batch of 4100 -- which silently mislabels samples-seen on Fig 5's x-axis.
    # Design A8 pins the micro batch to a power of two no larger than the
    # ladder's gcd precisely so this is always satisfiable.
    with pytest.raises(ValueError, match="does not divide"):
        fj.resolve_plan(_args(batch_size=4096), micro_batch=100)


def test_resolve_plan_evaluates_on_a_uniform_sample_grid():
    plan = fj.resolve_plan(_args(batch_size=1024, max_samples=2_000_000), micro_batch=256)
    assert plan["eval_quantum"] == 16_384
    assert plan["eval_every"] == 16
    assert plan["steps"] == 1953
    assert plan["n_evals"] == 122


def test_every_ladder_rung_gets_the_same_number_of_evaluations():
    # The whole point of spacing evaluation in SAMPLES: eval index i is the
    # same amount of training on every rung, so MedianPruner compares trials
    # at matching data rather than matching steps. On the log-spaced step grid
    # this replaces, index i on a batch-4096 trial had seen 16x the data of a
    # batch-256 one and the pruner compared them anyway.
    counts = {
        batch: fj.resolve_plan(
            _args(batch_size=batch, max_samples=2_000_000), micro_batch=256
        )["n_evals"]
        for batch in (256, 512, 1024, 2048, 4096)
    }
    assert set(counts.values()) == {122}, counts


def test_resolve_plan_refuses_a_budget_shorter_than_one_evaluation():
    # Without this the run trains to completion and writes nothing, which
    # surfaces downstream as a missing run rather than a misconfiguration.
    with pytest.raises(SystemExit, match="shorter than one evaluation"):
        # 10 steps of 1024, but one evaluation is 16 steps.
        fj.resolve_plan(
            _args(batch_size=1024, max_samples=10_240), micro_batch=256
        )


def test_max_samples_overrides_the_epoch_cap():
    # Tuning is capped in samples so the tuned batch size transfers; the final
    # campaign is capped at 4 epochs. --max-samples selects the former.
    plan = fj.resolve_plan(
        _args(batch_size=1024, epochs=4.0, max_samples=2_000_000), micro_batch=256
    )
    assert plan["steps"] == 1953  # not steps_for_epochs(...) == 5004


def test_resolve_plan_accepts_a_fractional_epoch_for_tuning():
    plan = fj.resolve_plan(_args(batch_size=1024, epochs=0.5), micro_batch=256)
    assert plan["steps"] == 625


def test_head_lr_is_rejected_for_the_ibprobit_arm():
    # IBProbit runs no optimizer on its head -- optimizer.init sees the backbone
    # alone and the head is assimilated in closed form by loss_fn.update. A
    # head lr silently doing nothing would look like a tuned axis that isn't.
    argv = ["--arm", "ibprobit", "--batch-size", "1024", "--micro-batch-size", "256", "--head-lr", "1e-2"]
    args = fj.build_argparser().parse_args(argv)
    with pytest.raises(SystemExit, match="head"):
        fj.check_arm_flags(args)


def test_head_lr_is_accepted_for_the_ce_arm():
    fj.check_arm_flags(_args(head_lr=1e-2))  # must not raise


@pytest.mark.parametrize("arm", ["ce", "ibprobit"])
def test_micro_batch_size_is_required_on_every_arm(arm):
    # Design A8: the micro batch is measured once per arm and per GPU
    # (scripts/finetune/probe_micro_batch.py) and pinned, so no arm auto-probes
    # any more. That also removes the asymmetry where only --arm ibprobit had
    # to pass it, which existed purely because the CE-only prober could not
    # build an IBProbit head.
    with pytest.raises(SystemExit):
        fj.build_argparser().parse_args(["--arm", arm, "--batch-size", "1024"])


def test_the_auto_probe_is_gone():
    assert not hasattr(fj, "probe_micro_batch_size_ce")
    assert not hasattr(fj, "check_micro_batch_required")


def test_enable_mlflow_flags_were_removed():
    # M1: --enable-mlflow / --mlflow-experiment were parsed but never read in
    # main() -- a silently-ignored logging flag loses a run's telemetry
    # without any error. Removed rather than wired, so passing them is now a
    # loud argparse error instead of a silent no-op.
    with pytest.raises(SystemExit):
        fj.build_argparser().parse_args(
            ["--arm", "ce", "--batch-size", "1024", "--enable-mlflow"]
        )
    with pytest.raises(SystemExit):
        fj.build_argparser().parse_args(
            ["--arm", "ce", "--batch-size", "1024", "--mlflow-experiment", "x"]
        )


