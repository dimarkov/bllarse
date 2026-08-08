import numpy as np
import pytest

from bllarse import evaldump, metrics
from bllarse.tuning import objective


# The fake training script's batch, so `samples_seen = step * FAKE_BATCH`.
# Patience thresholds in these tests are written as `n * FAKE_BATCH` to keep
# them readable as "n evaluations" while still exercising the real sample
# arithmetic the rule now uses.
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

    It writes one dump PER EVALUATION where the real scripts now buffer and
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
    from. The earlier IBProbit/CE sweeps used 143 and 145 -- inside the range --
    and this deliberately does not repeat that."""
    assert set(objective.TUNING_SEEDS).isdisjoint(range(137, 147))


def test_evaluate_runs_both_seeds_and_removes_its_dumps(tmp_path, monkeypatch):
    """1,120 trials x 2 seeds x 40 dumps x 2 splits is ~179k files. Trial dumps
    have no reporting value once the objective is computed -- Stage 4 re-runs
    the winner on the reported seeds -- so leaving them would fill /ptmp for
    nothing."""
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
    warmup guard exists to prevent: the dump schedule is log-spaced, so the
    earliest dumps are a single optimiser step apart."""
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
    """Default is None, and must reproduce today's behaviour EXACTLY: every
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

    This is what makes a sample-spaced grid affordable at all: the grid now has
    up to 30,000 points where the old geometric one had 40, and streaming an
    array pair to GPFS at each would be ~750x the I/O for data nothing reads.
    Everything downstream selects the argmin anyway (analysis.thresholds.
    early_stop), so the rest was written and then deleted unread.
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
    # float16 (evaldump.py:130). The gap is ~3e-5 relative, which is fp16
    # round-trip and nothing else -- it predates this protocol and is why the
    # tolerance is loose rather than exact.
    stop = early_stop(run_curve(tmp_path / "dumps", "g", "run0"), "val_nll")
    assert stop.value == pytest.approx(min(seen), rel=1e-3)


def test_the_best_dump_is_written_even_when_patience_stops_the_run(tmp_path):
    """The buffered dump is written from a `finally`.

    With streaming writes an EarlyConvergence escaping `main` still left every
    dump so far on disk. Buffered, the same escape would leave the directory
    EMPTY and score a converged trial as a missing run -- so the write has to
    survive the exception, and this is the test that says so.
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
    plateau wherever it sits in the run -- which a count on the old geometric
    grid did not, and which is why `--patience 20` could only fire when a curve
    turned in its first half.

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
    """The bug that failed two real ibprobit trials.

    `best is None or value < best * (...)` adopts a nan as the best when it
    arrives FIRST, because the `best is None` arm short-circuits. Every later
    comparison is then `v < nan` -> False, so the counter never resets and
    patience fires at exactly `1 + patience` evaluations regardless of how well
    the run went. Both failures showed the signature precisely:
    patience_stopped_at_eval {'201': 41} against a patience of 40.

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
