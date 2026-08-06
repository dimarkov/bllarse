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
    argv = ["--arm", "ce", "--batch-size", "1024"]
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
    with pytest.raises(SystemExit, match="512"):
        fj.check_img_size(_args(img_size=512))


def test_check_img_size_accepts_224():
    fj.check_img_size(_args(img_size=224))  # must not raise


def test_check_img_size_allows_an_explicit_override():
    fj.check_img_size(_args(img_size=512, allow_non_224=True))


def test_resolve_plan_derives_steps_from_the_effective_batch():
    plan = fj.resolve_plan(_args(batch_size=1024), micro_batch=64)
    assert plan["effective_batch"] == 1024
    assert plan["num_accum"] == 16
    assert plan["steps"] == 5_004


def test_resolve_plan_uses_the_achieved_batch_not_the_request():
    # accum_plan may not hit the target exactly; steps must follow what the run
    # will actually do, or samples-seen on Fig 5's x-axis is wrong.
    plan = fj.resolve_plan(_args(batch_size=4096), micro_batch=100)
    assert plan["effective_batch"] == 4100
    assert plan["steps"] == fj.protocol.steps_for_epochs(
        fj.imagenet.IMAGENET_N_TRAIN, 4.0, 4100
    )


def test_resolve_plan_dump_schedule_spans_the_run():
    plan = fj.resolve_plan(_args(batch_size=1024), micro_batch=64)
    assert plan["dump_steps"][0] == 1
    assert plan["dump_steps"][-1] == plan["steps"]
    assert len(plan["dump_steps"]) == 24


def test_resolve_plan_ood_steps_are_a_subset_of_dump_steps():
    plan = fj.resolve_plan(_args(batch_size=1024), micro_batch=64)
    assert set(plan["ood_steps"]).issubset(set(plan["dump_steps"].tolist()))
    assert plan["ood_steps"][-1] == plan["steps"]


def test_resolve_plan_ood_disabled_yields_no_ood_steps():
    plan = fj.resolve_plan(
        _args(batch_size=1024, ood_every_k_dumps=0), micro_batch=64
    )
    assert len(plan["ood_steps"]) == 0


def test_resolve_plan_accepts_a_fractional_epoch_for_tuning():
    plan = fj.resolve_plan(_args(batch_size=1024, epochs=0.5), micro_batch=64)
    assert plan["steps"] == 625


def test_head_lr_is_rejected_for_the_ibprobit_arm():
    # IBProbit runs no optimizer on its head -- optimizer.init sees the backbone
    # alone and the head is assimilated in closed form by loss_fn.update. A
    # head lr silently doing nothing would look like a tuned axis that isn't.
    argv = ["--arm", "ibprobit", "--batch-size", "1024", "--head-lr", "1e-2"]
    args = fj.build_argparser().parse_args(argv)
    with pytest.raises(SystemExit, match="head"):
        fj.check_arm_flags(args)


def test_head_lr_is_accepted_for_the_ce_arm():
    fj.check_arm_flags(_args(head_lr=1e-2))  # must not raise
