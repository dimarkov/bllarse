"""The arm script's pure planning layer: budget, accumulation, guards.

Everything here runs without a GPU or a dataset, so a misconfigured sweep fails
at submit time rather than after a node allocation.
"""

import contextlib
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


def test_default_epochs_is_the_reported_two_epochs():
    assert _args().epochs == 2.0


def test_default_model_is_the_reported_dinov3_huge():
    assert _args().model == "dinov3_huge"


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
    plan = fj.resolve_plan(_args(batch_size=1024, epochs=4.0), micro_batch=64)
    assert plan["effective_batch"] == 1024
    assert plan["num_accum"] == 16
    assert plan["steps"] == 5_004


def test_resolve_plan_refuses_a_micro_batch_that_does_not_divide_the_target():
    # Rounding would turn 4096 over a micro batch of 100 into an effective
    # batch of 4100, which silently mislabels samples-seen on Fig 3's x-axis.
    # finetune_space.MICRO_BATCH pins the micro batch to a power of two no
    # larger than the ladder's gcd precisely so this is always satisfiable.
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
    # at matching data rather than matching steps. On a log-spaced step grid,
    # index i on a batch-4096 trial would have seen 16x the data of a
    # batch-256 one.
    counts = {
        batch: fj.resolve_plan(
            _args(batch_size=batch, max_samples=2_000_000), micro_batch=256
        )["n_evals"]
        for batch in (256, 512, 1024, 2048, 4096, 8192, 16384)
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
    # Tuning is capped in samples so the tuned batch size transfers; a final
    # run is capped in epochs. --max-samples selects the former.
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
    # The micro batch is measured once per arm and per GPU
    # (scripts/finetune/probe_micro_batch.py) and pinned in
    # finetune_space.MICRO_BATCH, so no arm auto-probes and every arm must
    # pass it.
    with pytest.raises(SystemExit):
        fj.build_argparser().parse_args(["--arm", arm, "--batch-size", "1024"])


def test_the_auto_probe_is_gone():
    assert not hasattr(fj, "probe_micro_batch_size_ce")
    assert not hasattr(fj, "check_micro_batch_required")


def test_the_mlflow_flags_reach_the_run(monkeypatch):
    # A logging flag that parses but never reaches child_run loses a run's
    # telemetry without any error. main() hands vars(args) to child_run as the
    # run's config, so both flags must arrive there.
    args = fj.build_argparser().parse_args(
        ["--arm", "ce", "--batch-size", "1024", "--micro-batch-size", "32",
         "--enable-mlflow", "--mlflow-experiment", "x"])
    seen = {}

    @contextlib.contextmanager
    def fake_child_run(config, **_kwargs):
        seen.update(config)
        yield None

    monkeypatch.setattr(fj, "child_run", fake_child_run)
    monkeypatch.setattr(fj, "_train", lambda args, on_dump=None, run=None: None)
    fj.main(args)

    assert seen["enable_mlflow"] is True
    assert seen["mlflow_experiment"] == "x"


def test_dump_index_is_read_only_where_it_is_bound():
    """`dump_index` is bound only inside the trajectory-dump branch, which a
    tuning trial (--dump-every-k-evals 0) never enters. A read anywhere else
    would let the run train correctly, write its dump, and THEN die with
    UnboundLocalError -- Optuna would record a FAILED trial for a run that
    succeeded. So every read must sit inside the body of an `if` that binds it.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(fj._train)))
    binding_bodies = [
        node.body for node in ast.walk(tree) if isinstance(node, ast.If)
        and any(isinstance(stmt, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "dump_index"
                        for t in stmt.targets)
                for stmt in node.body)
    ]
    assert binding_bodies, "dump_index is no longer bound in _train"
    guarded = {id(n) for body in binding_bodies for stmt in body
               for n in ast.walk(stmt)}
    reads = [n for n in ast.walk(tree) if isinstance(n, ast.Name)
             and n.id == "dump_index" and isinstance(n.ctx, ast.Load)]
    assert reads
    assert all(id(n) in guarded for n in reads)
