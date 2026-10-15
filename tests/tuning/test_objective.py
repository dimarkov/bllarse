import numpy as np
import pytest

from bllarse import evaldump, metrics
from bllarse.tuning import objective


# The fake training script's batch, so `samples_seen = step * FAKE_BATCH`.
# Patience thresholds in these tests are written as `n * FAKE_BATCH` to keep
# them readable as "n evaluations" while still exercising the real sample
# arithmetic the rule uses.
FAKE_BATCH = 100


def _make_patience_module(profiles, seed_log=None):
    """A fake training script driven by a per-seed sharpness PROFILE: a list
    of one sharpness value per evaluation. Higher sharpness means lower val_nll
    (a single-class 2-logit setup, same trick as `_write_curve` below), so a
    plateaued profile (a repeated value) is a plateaued val_nll curve.

    Unlike `_write_curve`, this fires `on_dump` once per evaluation -- like
    probe.py's real loop -- with a val_nll ACTUALLY COMPUTED (via
    bllarse.metrics.nll) from the logits just written, so what patience sees
    can never drift from what early_stop later reads back off disk. If
    `on_dump` raises (patience's EarlyConvergence, or a caller's own
    exception), main stops immediately.

    It writes one dump PER EVALUATION where the real scripts buffer and
    write only their argmin. That divergence is deliberate and local to this
    double: `_n_val_dumps` is how these tests observe how far a run got before
    patience cut it, which a single end-of-run dump cannot show. The real
    single-dump path is covered by
    `test_the_real_probe_writes_exactly_one_dump_at_its_val_optimum`.
    """

    class FakeModule:
        @staticmethod
        def build_argparser():
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument("--seed", type=int)
            p.add_argument("--uid")
            p.add_argument("--group-id")
            p.add_argument("--dump-root")
            return p

        @staticmethod
        def main(args, on_dump=None):
            if seed_log is not None:
                seed_log.append(args.seed)
            labels = np.zeros(64, dtype=np.int64)
            for i, sharpness in enumerate(profiles[args.seed]):
                step = i + 1
                samples_seen = step * FAKE_BATCH
                # 1-based, and equal to samples_seen // quantum with a
                # quantum of FAKE_BATCH -- the real scripts' definition.
                eval_index = step
                logits = np.zeros((64, 2), dtype=np.float32)
                logits[:, 0] = sharpness
                for split in ("val", "test"):
                    evaldump.write_dump(
                        args.dump_root, args.group_id, args.uid, step, logits, labels,
                        arm="adamw_ce", split=split, logit_kind="softmax_logits",
                        seed=args.seed, samples_seen=samples_seen,
                    )
                if on_dump is not None:
                    on_dump(eval_index, step, samples_seen,
                            {"val_nll": metrics.nll(logits, labels)})

    return FakeModule


def _n_val_dumps(root, group_id, uid):
    return len(list((root / group_id / uid).glob("step_*_val.npz")))


def _write_curve(root, group_id, uid, *, best_nll_at, n_steps=6):
    """A synthetic run with a controllable val_nll minimum."""
    labels = np.zeros(64, dtype=np.int64)
    for step in range(1, n_steps + 1):
        sharpness = 5.0 - abs(step - best_nll_at)
        logits = np.zeros((64, 2), dtype=np.float32)
        logits[:, 0] = sharpness
        for split in ("val", "test"):
            evaldump.write_dump(
                root, group_id, uid, step, logits, labels,
                arm="adamw_ce", split=split, logit_kind="softmax_logits",
                seed=200, samples_seen=step * 128,
            )


def test_score_is_the_mean_over_seeds_of_the_early_stopped_val_nll(tmp_path):
    root = tmp_path / "dumps"
    _write_curve(root, "t", "trial0_seed200", best_nll_at=3)
    _write_curve(root, "t", "trial0_seed201", best_nll_at=5)

    from bllarse.analysis.curves import run_curve
    from bllarse.analysis.thresholds import early_stop

    want = np.mean([
        early_stop(run_curve(root, "t", "trial0_seed200"), "val_nll").value,
        early_stop(run_curve(root, "t", "trial0_seed201"), "val_nll").value,
    ])
    got = objective.score_dumps(root, "t", ["trial0_seed200", "trial0_seed201"])
    assert got == pytest.approx(want)


def test_tuning_seeds_are_disjoint_from_the_reported_range():
    """Fig 1 and Fig 2 report range(137, 147). Selecting hyperparameters on a
    reported seed tunes against the same random draws the error bars are drawn
    from."""
    assert set(objective.TUNING_SEEDS).isdisjoint(range(137, 147))


def test_evaluate_runs_both_seeds_and_removes_its_dumps(tmp_path, monkeypatch):
    """Trial dumps have no reporting value once the objective is computed --
    the reported sweeps re-run the winner on the reported seeds -- so over
    thousands of trials, leaving them would fill scratch storage for nothing."""
    root = tmp_path / "dumps"
    seeds_seen = []

    class FakeModule:
        @staticmethod
        def build_argparser():
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument("--seed", type=int)
            p.add_argument("--uid")
            p.add_argument("--group-id")
            p.add_argument("--dump-root")
            return p

        @staticmethod
        def main(args, on_dump=None):
            seeds_seen.append(args.seed)
            _write_curve(args.dump_root, args.group_id, args.uid, best_nll_at=2)
            if on_dump is not None:
                on_dump(0, 1, 100, {"val_nll": 0.5})

    monkeypatch.setattr(
        objective.adapters, "load_script_module", lambda path: FakeModule
    )

    value = objective.evaluate(
        "adamw_ce", {}, dump_root=root, group_id="t", uid="trial0"
    )
    assert seeds_seen == list(objective.TUNING_SEEDS)
    assert isinstance(value, float)
    assert not (root / "t" / "trial0_seed200").exists()
    assert not (root / "t" / "trial0_seed201").exists()


def test_on_dump_fires_for_the_first_seed_only(tmp_path, monkeypatch):
    """Pruning a hopeless config before its second seed is where most of the
    saving comes from. Reporting from both seeds would report two different
    trajectories under one trial's step keys."""
    root = tmp_path / "dumps"
    calls = []

    class FakeModule:
        @staticmethod
        def build_argparser():
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument("--seed", type=int)
            p.add_argument("--uid")
            p.add_argument("--group-id")
            p.add_argument("--dump-root")
            return p

        @staticmethod
        def main(args, on_dump=None):
            _write_curve(args.dump_root, args.group_id, args.uid, best_nll_at=2)
            calls.append((args.seed, on_dump is not None))

    monkeypatch.setattr(
        objective.adapters, "load_script_module", lambda path: FakeModule
    )
    objective.evaluate(
        "adamw_ce", {}, dump_root=root, group_id="t", uid="t0",
        on_dump=lambda i, step, samples, scalars: None,
    )
    assert calls == [(200, True), (201, False)]


def test_a_real_trial_runs_end_to_end_on_the_jax_arm(tmp_path):
    """Proves the whole trial path -- adapters resolution, both seeds, the
    on_dump hook, run_curve, early_stop, cleanup -- against the real probe.py
    rather than a fake module."""
    rng = np.random.default_rng(0)
    cache = tmp_path / "features"
    cache.mkdir()
    for split, n in (("train", 600), ("test", 200)):
        labels = rng.integers(0, 2, size=n)
        feats = rng.normal(size=(n, 16)).astype(np.float32)
        feats[labels == 1, 0] += 6.0
        np.savez(cache / f"dinov3_small_fake_res224_{split}.npz",
                 features=feats, labels=labels)

    reported = []
    value = objective.evaluate(
        "adamw_ce",
        {
            "arm": "adamw_ce", "model": "dinov3_small", "dataset": "fake",
            "num_classes": 2, "cache_dir": str(cache),
            "split_cache_dir": str(tmp_path / "splits"),
            # 600 train / batch 100 = 6 steps per epoch, 5 epochs = 30 steps.
            # batch_cap(600) is 256, not a whole number of 100-sample steps, so
            # _derive_quantum falls back to the batch itself: one evaluation per
            # step, 30 of them, indexed 1..30 on the sample grid.
            "epochs": 5, "report_every": 1, "batch_size": 100, "no_hf_cache": True,
        },
        dump_root=tmp_path / "dumps", group_id="e2e", uid="trial0",
        on_dump=lambda i, step, samples, s: reported.append((i, samples)),
    )

    assert value > 0.0
    assert reported == [(i, i * 100) for i in range(1, 31)], (
        "intermediates come from the first seed only, indexed on the sample grid"
    )
    assert not (tmp_path / "dumps" / "e2e" / "trial0_seed200").exists()


def test_a_plateaued_run_stops_early_and_scores_what_was_dumped(tmp_path, monkeypatch):
    """The whole point of the feature: a config whose optimum arrives early
    should not keep paying for dumps to the cap. `score_dumps` is the same
    scoring path evaluate() uses internally, so if the two agree, the
    returned value really is the argmin over whatever got written -- not a
    stale or fabricated number."""
    root = tmp_path / "dumps"
    # Improves for 3 dumps, then plateaus for as many as 20 more -- without
    # patience this would write all 23 dumps for both seeds.
    profile = [1.0, 2.0, 3.0] + [3.0] * 20
    monkeypatch.setattr(
        objective.adapters, "load_script_module",
        lambda path: _make_patience_module({200: profile, 201: profile}),
    )

    value = objective.evaluate(
        "adamw_ce", {}, dump_root=root, group_id="t", uid="trial0",
        patience=3, warmup_evals=2, keep_dumps=True,
    )

    uids = [f"trial0_seed{seed}" for seed in objective.TUNING_SEEDS]
    for uid in uids:
        n_written = _n_val_dumps(root, "t", uid)
        assert 0 < n_written < len(profile), (
            "patience should have cut the run short of its full profile"
        )
    assert value == pytest.approx(objective.score_dumps(root, "t", uids))


def test_patience_does_not_fire_before_the_warmup_threshold(tmp_path, monkeypatch):
    """A flat profile plateaus from dump 1 onward -- with a low warmup this
    would stop almost immediately, which is exactly the failure mode the
    warmup guard exists to prevent: a config that is merely slow to start must
    not be killed for it."""
    root = tmp_path / "dumps"
    profile = [3.0, 3.0, 3.0, 3.0, 3.0]
    monkeypatch.setattr(
        objective.adapters, "load_script_module",
        lambda path: _make_patience_module({200: profile, 201: profile}),
    )

    objective.evaluate(
        "adamw_ce", {}, dump_root=root, group_id="t", uid="trial0",
        patience=2, warmup_evals=10, keep_dumps=True,
    )

    uids = [f"trial0_seed{seed}" for seed in objective.TUNING_SEEDS]
    for uid in uids:
        assert _n_val_dumps(root, "t", uid) == len(profile), (
            "warmup_dumps (10) exceeds the profile length, so patience "
            "(which would otherwise fire almost immediately) must never fire"
        )


def test_patience_applies_to_the_second_seed_too(tmp_path, monkeypatch):
    """The part most likely to be got wrong: `on_dump` (Optuna's reporting
    hook) fires for the first tuning seed only, but patience is a different
    concern and must cover seed 201 as well -- otherwise half the saving this
    feature exists for evaporates."""
    root = tmp_path / "dumps"
    # Seed 200 keeps improving every dump -- patience must never fire for it.
    improving = [1.0 + i for i in range(8)]
    # Seed 201 plateaus after 3 dumps.
    plateaued = [1.0, 2.0, 3.0] + [3.0] * 5
    monkeypatch.setattr(
        objective.adapters, "load_script_module",
        lambda path: _make_patience_module({200: improving, 201: plateaued}),
    )

    converged = []
    objective.evaluate(
        "adamw_ce", {}, dump_root=root, group_id="t", uid="trial0",
        on_dump=lambda i, step, samples, scalars: None,
        patience=3, warmup_evals=2, keep_dumps=True,
        on_converged=lambda seed, eval_index, step: converged.append(seed),
    )

    assert converged == [201], "only the plateaued seed should have converged early"
    assert _n_val_dumps(root, "t", "trial0_seed200") == len(improving)
    assert _n_val_dumps(root, "t", "trial0_seed201") < len(plateaued)


def test_patience_none_leaves_evaluate_byte_for_byte_unchanged(tmp_path, monkeypatch):
    """Default is None, and must leave the run loop EXACTLY as without it: every
    dump in the profile gets written and reported, even though this profile
    is deliberately flat -- a plateau that patience, if enabled, would cut
    short."""
    root = tmp_path / "dumps"
    profile = [3.0, 3.0, 3.0, 3.0]
    monkeypatch.setattr(
        objective.adapters, "load_script_module",
        lambda path: _make_patience_module({200: profile, 201: profile}),
    )

    calls = []
    objective.evaluate(
        "adamw_ce", {}, dump_root=root, group_id="t", uid="trial0",
        on_dump=lambda i, step, samples, scalars: calls.append((i, step)),
        patience=None, keep_dumps=True,
    )

    assert calls == [(i, i) for i in range(1, len(profile) + 1)], (
        "on_dump must still fire once per evaluation, for the full profile"
    )
    for seed in objective.TUNING_SEEDS:
        assert _n_val_dumps(root, "t", f"trial0_seed{seed}") == len(profile)


def test_an_on_dump_exception_still_propagates_through_patience(tmp_path, monkeypatch):
    """Stands in for optuna.TrialPruned, which objective.py must never import
    or know about. The convergence handler catches ONLY its own signal --
    pruning a losing trial and stopping a converged one are different events,
    and this must not conflate them by swallowing the wrong one."""

    class FakePrune(Exception):
        pass

    root = tmp_path / "dumps"
    profile = [1.0, 2.0, 3.0, 3.0, 3.0, 3.0]
    monkeypatch.setattr(
        objective.adapters, "load_script_module",
        lambda path: _make_patience_module({200: profile, 201: profile}),
    )

    def pruning_on_dump(eval_index, step, samples_seen, scalars):
        if eval_index == 2:
            raise FakePrune("pruned")

    with pytest.raises(FakePrune):
        objective.evaluate(
            "adamw_ce", {}, dump_root=root, group_id="t", uid="trial0",
            on_dump=pruning_on_dump, patience=2, warmup_evals=1,
        )

    # The try/finally cleanup must still have run despite the exception.
    assert not (root / "t" / "trial0_seed200").exists()


def _fake_feature_cache(tmp_path, n_train=600, n_test=200, dim=16):
    """Cached features for a 2-class synthetic dataset, laid out the way
    load_cached_features expects."""
    rng = np.random.default_rng(0)
    cache = tmp_path / "features"
    cache.mkdir()
    for split, n in (("train", n_train), ("test", n_test)):
        labels = rng.integers(0, 2, size=n)
        feats = rng.normal(size=(n, dim)).astype(np.float32)
        feats[labels == 1, 0] += 6.0
        np.savez(cache / f"dinov3_small_fake_res224_{split}.npz",
                 features=feats, labels=labels)
    return cache


def test_the_real_probe_writes_exactly_one_dump_at_its_val_optimum(tmp_path):
    """The run's whole disk output is the argmin evaluation, written once at the
    end -- not one dump per evaluation.

    This is what makes a sample-spaced grid affordable at all: the grid has up
    to 30,000 points, and streaming an array pair to disk at each would be a
    flood of I/O for data nothing reads. Everything downstream selects the
    argmin anyway (analysis.thresholds.early_stop), so the rest would be
    written and then deleted unread.
    """
    from bllarse.analysis.curves import run_curve
    from bllarse.analysis.thresholds import early_stop
    from bllarse.tools import adapters

    cache = _fake_feature_cache(tmp_path)
    module = adapters.load_script_module(objective.SCRIPTS["adamw_ce"])
    args = adapters.parse_config(module, {
        "arm": "adamw_ce", "model": "dinov3_small", "dataset": "fake",
        "num_classes": 2, "cache_dir": str(cache),
        "split_cache_dir": str(tmp_path / "splits"),
        "epochs": 5, "batch_size": 100, "no_hf_cache": True,
        "seed": 200, "uid": "run0", "group_id": "g",
        "dump_root": str(tmp_path / "dumps"),
    })

    seen = []
    module.main(args, on_dump=lambda i, step, samples, s: seen.append(s["val_nll"]))

    run_dir = tmp_path / "dumps" / "g" / "run0"
    assert len(list(run_dir.glob("step_*_val.npz"))) == 1
    assert len(list(run_dir.glob("step_*_test.npz"))) == 1
    assert len(seen) == 30, "one evaluation per step at a batch-sized quantum"

    # The dump that landed is the argmin of what on_dump reported -- the two
    # must agree or the objective scores a different point than patience and
    # the pruner were watching.
    #
    # Not bit-equal, and cannot be: on_dump sees the fp32 val_nll computed in
    # the loop, while this reads it back off a dump whose logits were cast to
    # float16 (evaldump.write_dump). The gap is ~3e-5 relative, which is fp16
    # round-trip and nothing else -- and is why the tolerance is loose rather
    # than exact.
    stop = early_stop(run_curve(tmp_path / "dumps", "g", "run0"), "val_nll")
    assert stop.value == pytest.approx(min(seen), rel=1e-3)


def test_the_best_dump_is_written_even_when_patience_stops_the_run(tmp_path):
    """The buffered dump is written from a `finally`.

    Because the dump is buffered, an EarlyConvergence escaping `main` would
    otherwise leave the directory EMPTY and score a converged trial as a
    missing run -- so the write has to survive the exception, and this is the
    test that says so.
    """
    from bllarse.tools import adapters

    cache = _fake_feature_cache(tmp_path)
    module = adapters.load_script_module(objective.SCRIPTS["adamw_ce"])
    args = adapters.parse_config(module, {
        "arm": "adamw_ce", "model": "dinov3_small", "dataset": "fake",
        "num_classes": 2, "cache_dir": str(cache),
        "split_cache_dir": str(tmp_path / "splits"),
        "epochs": 5, "batch_size": 100, "no_hf_cache": True,
        "seed": 200, "uid": "run0", "group_id": "g",
        "dump_root": str(tmp_path / "dumps"),
    })

    def stop_at_eval_3(eval_index, step, samples_seen, scalars):
        if eval_index == 3:
            raise objective.EarlyConvergence(eval_index, step)

    with pytest.raises(objective.EarlyConvergence):
        module.main(args, on_dump=stop_at_eval_3)

    run_dir = tmp_path / "dumps" / "g" / "run0"
    assert len(list(run_dir.glob("step_*_val.npz"))) == 1, (
        "the argmin dump must survive the exception that stopped the run"
    )


def test_patience_counts_evaluations_at_any_scale(tmp_path, monkeypatch):
    """A count is meaningful because the evaluation grid is UNIFORM in samples.

    N evaluations is always N x eval_quantum samples, so the same count cuts a
    plateau wherever it sits in the run -- which a count on a geometric grid
    would not.

    Two runs whose plateau begins at very different points, one threshold.
    """
    early = [1.0, 2.0, 3.0] + [3.0] * 20        # plateaus at eval 3
    late = [1.0] * 20 + [2.0, 3.0] + [3.0] * 20  # improves late, then plateaus

    for name, profile in (("early", early), ("late", late)):
        root = tmp_path / name
        monkeypatch.setattr(
            objective.adapters, "load_script_module",
            lambda path, p=profile: _make_patience_module({200: p, 201: p}),
        )
        objective.evaluate(
            "adamw_ce", {}, dump_root=root, group_id="t", uid="trial0",
            patience=5, warmup_evals=2, keep_dumps=True,
        )
        n = _n_val_dumps(root, "t", "trial0_seed200")
        assert 0 < n < len(profile), (
            f"{name}: patience 5 must cut a plateau at either position, "
            f"but wrote {n} of {len(profile)}"
        )


def _evals_before_patience(curve, min_relative_improvement, patience=20,
                           warmup_evals=10):
    """How many evaluations of `curve` (val_nll values) run before patience.

    Drives `_patience_on_dump` directly rather than going through a fake
    training script: the question is about a THRESHOLD on val_nll, and the
    script double is driven by a sharpness profile whose relationship to
    val_nll is monotone but not linear -- fine for "does a plateau get cut",
    useless for "is 3e-4 per evaluation above the threshold".
    """
    hook = objective._patience_on_dump(
        None, patience=patience, warmup_evals=warmup_evals,
        min_relative_improvement=min_relative_improvement)
    for index, value in enumerate(curve, start=1):
        try:
            hook(index, index, index * 16_384, {"val_nll": value})
        except objective.EarlyConvergence:
            return index - 1
    return len(curve)


def test_slow_but_real_improvement_survives_patience():
    """The threshold has to sit BELOW the effect the campaign measures.

    Patience needs `patience` CONSECUTIVE stalls, so on a curve falling at rate
    r the best refreshes every ceil(theta * best / r) evaluations and patience
    fires only when r < theta * best / patience -- a factor of `patience`
    smaller than theta * best. At val_nll ~= 0.44 with patience 20 that
    critical rate is 2.2e-5 per evaluation at theta = 1e-3.

    So a curve improving at 1e-5 per evaluation is still learning, and would
    give up ~5e-4 over the remaining 50 evaluations. The arms are separated by
    0.0026, which makes that a fifth of the effect -- discarded from whichever
    arm happens to sit on the wrong side.
    """
    assert objective.DEFAULT_MIN_RELATIVE_IMPROVEMENT == 1e-4
    curve = [0.44 - 1e-5 * i for i in range(60)]

    assert _evals_before_patience(curve, 1e-4) == len(curve), (
        "a curve still improving at 1e-5 per evaluation was cut short")
    # 1e-3, shown failing on the same curve.
    assert _evals_before_patience(curve, 1e-3) < len(curve)


def test_a_genuine_plateau_is_still_cut():
    """The 1e-4 threshold must not disable patience in practice.

    1e-4 still has to catch the fourth-decimal drift the constant exists to
    ignore, or every trial runs the full budget and the economy is gone.
    """
    drift = [0.44 - 1e-7 * i for i in range(60)]
    cut = _evals_before_patience(drift, 1e-4)
    assert cut < len(drift), "patience never fired on a drifting curve"
    # Stalls accumulate from evaluation 1 and `warmup_evals` gates only the
    # FIRING, so with patience 20 the earliest possible stop is evaluation 21,
    # not warmup + patience = 30.
    assert cut >= 20


def test_patience_is_disabled_unless_a_threshold_is_given(tmp_path, monkeypatch):
    """Defaults to None. Unset, the callback must be the caller's own hook
    untouched, so a run trains to its full budget -- the behaviour any figure
    run depends on."""
    root = tmp_path / "dumps"
    profile = [3.0] * 8
    monkeypatch.setattr(
        objective.adapters, "load_script_module",
        lambda path: _make_patience_module({200: profile, 201: profile}),
    )
    objective.evaluate(
        "adamw_ce", {}, dump_root=root, group_id="t", uid="trial0", keep_dumps=True,
    )
    for seed in objective.TUNING_SEEDS:
        assert _n_val_dumps(root, "t", f"trial0_seed{seed}") == len(profile)


def test_a_nonfinite_first_evaluation_does_not_latch_the_patience_counter(
    tmp_path, monkeypatch
):
    """Without the `isfinite` guard, `best is None or value < best * (...)`
    adopts a nan as the best when it arrives FIRST, because the `best is None`
    arm short-circuits. Every later comparison is then `v < nan` -> False, so
    the counter never resets and patience fires at exactly `1 + patience`
    evaluations regardless of how well the run went (signature:
    patience_stopped_at_eval of patience + 1).

    A run that recovers must be scored on the values it recovered to.
    """
    root = tmp_path / "dumps"
    # nan first, then a genuinely improving curve.
    profile = [float("nan"), 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    monkeypatch.setattr(
        objective.adapters, "load_script_module",
        lambda path: _make_patience_module({200: profile, 201: profile}),
    )

    value = objective.evaluate(
        "adamw_ce", {}, dump_root=root, group_id="t", uid="trial0",
        patience=3, warmup_evals=2, keep_dumps=True,
    )

    assert np.isfinite(value), (
        "a run that produced finite values after a nan must score finitely"
    )
    for seed in objective.TUNING_SEEDS:
        assert _n_val_dumps(root, "t", f"trial0_seed{seed}") == len(profile), (
            "the improving tail must not be cut short by a latched counter"
        )


def test_a_nonfinite_evaluation_still_counts_as_a_stall(tmp_path, monkeypatch):
    """Not adopted as the best, but not free either. A run producing nothing
    but nan has converged to nothing and must still stop, rather than paying
    its full budget."""
    root = tmp_path / "dumps"
    profile = [1.0, 2.0] + [float("nan")] * 20
    monkeypatch.setattr(
        objective.adapters, "load_script_module",
        lambda path: _make_patience_module({200: profile, 201: profile}),
    )
    objective.evaluate(
        "adamw_ce", {}, dump_root=root, group_id="t", uid="trial0",
        patience=3, warmup_evals=2, keep_dumps=True,
    )
    for seed in objective.TUNING_SEEDS:
        assert _n_val_dumps(root, "t", f"trial0_seed{seed}") < len(profile)


# --- seeds/scripts/score: the fine-tuning campaign shares this run loop ------


def test_evaluate_runs_one_run_per_seed(tmp_path, monkeypatch):
    """`seeds` controls how many runs a trial costs.

    Fine-tuning uses ONE. Two is right for a 40-second probe; at ~6 hours a
    trial it would double a ~880 A100-hour campaign.
    """
    seen = []

    class FakeModule:
        @staticmethod
        def main(args, on_dump=None):
            seen.append(args["seed"])

    monkeypatch.setattr(
        objective.adapters, "load_script_module", lambda path: FakeModule)
    monkeypatch.setattr(objective.adapters, "parse_config", lambda mod, cfg: dict(cfg))

    value = objective.evaluate(
        "ce", {}, dump_root=tmp_path, group_id="g", uid="t",
        seeds=(200,), scripts={"ce": "fake.py"},
        score=lambda root, group, uids: float(len(uids)),
    )
    assert seen == [200]
    assert value == 1.0


def test_evaluate_defaults_reproduce_the_probing_behaviour(tmp_path, monkeypatch):
    # Both probing seeds, in order, off the unmodified SCRIPTS table.
    seen = []

    class FakeModule:
        @staticmethod
        def main(args, on_dump=None):
            seen.append(args["seed"])

    monkeypatch.setattr(
        objective.adapters, "load_script_module", lambda path: FakeModule)
    monkeypatch.setattr(objective.adapters, "parse_config", lambda mod, cfg: dict(cfg))
    monkeypatch.setattr(objective, "score_dumps", lambda *a: 0.0)

    objective.evaluate(
        "adamw_ce", {}, dump_root=tmp_path, group_id="g", uid="t",
        score=lambda *a: 0.0,
    )
    assert seen == list(objective.TUNING_SEEDS)


def test_evaluate_uses_the_supplied_script_table(tmp_path, monkeypatch):
    # An arm valid in the fine-tuning table but absent from the probing one
    # must be accepted, and vice versa -- otherwise the two campaigns' arm
    # names would have to be kept globally unique.
    loaded = []
    monkeypatch.setattr(
        objective.adapters, "load_script_module",
        lambda path: loaded.append(path) or type("M", (), {
            "main": staticmethod(lambda args, on_dump=None: None)}))
    monkeypatch.setattr(objective.adapters, "parse_config", lambda mod, cfg: dict(cfg))

    objective.evaluate(
        "ce", {}, dump_root=tmp_path, group_id="g", uid="t",
        seeds=(200,), scripts={"ce": "scripts/finetune/finetune_jax.py"},
        score=lambda *a: 0.0,
    )
    assert loaded == ["scripts/finetune/finetune_jax.py"]

    with pytest.raises(ValueError, match="unknown arm"):
        objective.evaluate(
            "adamw_ce", {}, dump_root=tmp_path, group_id="g", uid="t",
            seeds=(200,), scripts={"ce": "scripts/finetune/finetune_jax.py"},
            score=lambda *a: 0.0,
        )


def test_best_window_mean_needs_a_level_held_not_a_single_dip():
    """The whole point. A lone spike down cannot carry a window, so a curve that
    only ever touches a low value scores worse than one that holds it."""
    from bllarse.tuning.objective import best_window_mean

    dips = [0.5] * 20 + [0.10] + [0.5] * 20      # touches 0.10 once
    holds = [0.30] * 41                           # never better than 0.30
    assert best_window_mean(dips, 1) < best_window_mean(holds, 1)      # min: dips wins
    assert best_window_mean(holds, 10) < best_window_mean(dips, 10)    # window: holds wins


def test_best_window_mean_penalises_a_window_containing_a_nonfinite():
    """A run that diverges inside a window has not held that level. Dropping the
    point instead would let a config launder a NaN excursion via good
    neighbours."""
    import math

    from bllarse.tuning.objective import best_window_mean

    clean = [0.2] * 10
    spoiled = [0.2] * 5 + [float("nan")] + [0.2] * 4
    assert best_window_mean(clean, 10) == pytest.approx(0.2)
    assert math.isinf(best_window_mean(spoiled, 10))
    # ...but a window that clears the excursion is still scored.
    assert best_window_mean([0.2] * 10 + [float("inf")], 5) == pytest.approx(0.2)


def test_best_window_mean_clamps_a_window_longer_than_the_run():
    from bllarse.tuning.objective import best_window_mean

    assert best_window_mean([0.4, 0.2], 50) == pytest.approx(0.3)


def test_the_smoothing_window_is_a_FIXED_COUNT_not_a_fraction_of_the_plan():
    """A window sized as a fraction of the planned grid overruns the data.

    At 10% of `evals_planned`, patience truncates runs far below their budget,
    so the window OVERRUNS the trajectory -- dtd's best config planned 3,750
    evaluations, ran 79, and would get a window of 375, which
    `best_window_mean` clamps to the trajectory length. The objective silently
    becomes "mean of the whole run", charging a config for every point after
    its minimum, and ranks the genuinely best config 88th of 100.
    """
    from bllarse.tuning.objective import DEFAULT_SMOOTH_WINDOW, best_window_mean

    assert isinstance(DEFAULT_SMOOTH_WINDOW, int)
    # Must fit inside the SHORTEST trajectory actually observed (71 on dtd),
    # or it degenerates into the mean of the whole run for that trial.
    assert 1 < DEFAULT_SMOOTH_WINDOW <= 20

    # The shape at risk: a real minimum mid-run, followed by mild overfitting.
    curve = [1.14, 0.48, 0.43, 0.411, 0.415, 0.427, 0.430, 0.428] * 10
    whole_run = best_window_mean(curve, len(curve))
    windowed = best_window_mean(curve, DEFAULT_SMOOTH_WINDOW)
    assert windowed < whole_run, (
        "a window spanning the whole trajectory cannot see the minimum at all"
    )
    assert windowed < 0.43, "the real minimum must still be visible"


def test_smoothing_off_leaves_the_scoring_path_untouched():
    """The fine-tuning campaign shares `evaluate` without smoothing, so the
    default must not acquire a per-evaluation side effect."""
    import inspect

    from bllarse.tuning import objective

    src = inspect.getsource(objective.evaluate)
    assert "if smooth_window is not None:" in src
    assert objective.evaluate.__kwdefaults__["smooth_window"] is None


def test_running_window_matches_best_window_mean_on_complete_windows():
    """The O(1) pruner view must agree with the O(n) objective wherever both
    are defined -- once a full window exists, the running minimum IS the lowest
    complete-window mean."""
    import random

    from bllarse.tuning.objective import RunningWindow, best_window_mean

    rng = random.Random(20260812)
    values = [rng.uniform(0.1, 0.9) for _ in range(200)]
    w = 12
    run = RunningWindow(w)
    for v in values:
        run.push(v)
    assert run.best == pytest.approx(best_window_mean(values, w))


def test_running_window_reports_a_prefix_mean_before_the_window_fills():
    """The pruner needs a comparable number from the first evaluation. Trials
    are only ever compared at equal eval index, so a deterministic function of
    the prefix is sufficient."""
    from bllarse.tuning.objective import RunningWindow

    run = RunningWindow(10)
    assert run.push(0.4) == pytest.approx(0.4)
    assert run.push(0.2) == pytest.approx(0.3)   # mean, not min


def test_running_window_does_not_latch_onto_a_lucky_first_evaluation():
    """A running MINIMUM over prefixes would let evaluation 1 set a floor for
    the whole run -- the dip-sensitivity this class removes, reintroduced at the
    start of every trial. The prefix mean must rise when the run gets worse."""
    from bllarse.tuning.objective import RunningWindow

    run = RunningWindow(50)          # never fills, so this is the prefix regime
    first = run.push(0.01)           # one lucky evaluation
    later = [run.push(0.9) for _ in range(9)]
    assert first == pytest.approx(0.01)
    assert later[-1] > 0.5, "a lucky first point must not pin the report"


def test_running_window_rejects_a_window_spanning_a_nonfinite():
    """Same rule as best_window_mean: a run that diverged has not held that
    level, so the window scores +inf rather than skipping the point."""
    import math

    from bllarse.tuning.objective import RunningWindow

    run = RunningWindow(3)
    run.push(0.2); run.push(0.2); run.push(0.2)
    assert run.best == pytest.approx(0.2)
    run.push(float("nan"))          # window now spans the nan
    assert run.best == pytest.approx(0.2)   # the earlier clean window survives
    clean = RunningWindow(3)
    clean.push(float("nan")); clean.push(0.2); clean.push(0.2)
    assert math.isinf(clean.best)   # never held a clean window


def test_the_windowed_pruner_view_prefers_a_held_level_over_a_lone_dip():
    """MedianPruner compares what trials REPORT. Under the raw statistic a
    spiky config reports a low value the moment it dips and survives, while a
    steady config reports its honest level and is killed -- the pruner
    selecting for exactly what the objective penalises. Measured on a campaign
    with raw reporting and windowed scoring: 11 of 21 winners came from the
    un-prunable first 20 trials (see objective.RunningWindow)."""
    from bllarse.tuning.objective import RunningWindow

    spiky = [0.5] * 8 + [0.10] + [0.5] * 8
    steady = [0.30] * 17

    raw_spiky, raw_steady = min(spiky), min(steady)
    assert raw_spiky < raw_steady, "raw reporting favours the dip"

    a, b = RunningWindow(8), RunningWindow(8)
    for v in spiky:
        a.push(v)
    for v in steady:
        b.push(v)
    assert b.best < a.best, "windowed reporting must favour the held level"


def test_jit_caches_are_cleared_after_every_run(tmp_path, monkeypatch):
    """Both arms reach JAX -- VBLL trains in PyTorch but scores through
    bllarse.metrics, whose fit_temperature is a jitted lax.scan. Trials vary
    batch_size, so the compilation cache grows monotonically over a worker's
    hundreds of trials until the CPU JIT cannot allocate (measured: VBLL
    workers on flowers102 die at ~177 trials without clearing).

    Asserted per SEED, not per trial: a two-seed trial would otherwise carry the
    first seed's kernels into the second.
    """
    calls = []
    monkeypatch.setattr(objective, "clear_jit_caches", lambda: calls.append(1))

    class FakeModule:
        @staticmethod
        def build_argparser():
            import argparse
            p = argparse.ArgumentParser()
            for f in ("--seed", "--uid", "--group-id", "--dump-root"):
                p.add_argument(f, type=int if f == "--seed" else str)
            return p

        @staticmethod
        def main(args, on_dump=None):
            _write_curve(args.dump_root, args.group_id, args.uid, best_nll_at=2)

    monkeypatch.setattr(
        objective.adapters, "load_script_module", lambda path: FakeModule
    )
    objective.evaluate("adamw_ce", {}, dump_root=tmp_path, group_id="t", uid="t0")
    assert len(calls) == len(objective.TUNING_SEEDS)


def test_clearing_is_best_effort_and_never_fails_a_trial(monkeypatch):
    """A jax that is missing or too old to expose clear_caches must not turn
    cleanup into a trial failure."""
    import builtins

    real_import = builtins.__import__

    def boom(name, *a, **k):
        if name == "jax":
            raise ImportError("no jax here")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    objective.clear_jit_caches()   # must not raise
