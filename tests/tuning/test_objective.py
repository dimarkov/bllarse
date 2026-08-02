import numpy as np
import pytest

from bllarse import evaldump, metrics
from bllarse.tuning import objective


def _make_patience_module(profiles, seed_log=None):
    """A fake training script driven by a per-seed sharpness PROFILE: a list
    of one sharpness value per dump. Higher sharpness means lower val_nll (a
    single-class 2-logit setup, same trick as `_write_curve` below), so a
    plateaued profile (a repeated value) is a plateaued val_nll curve.

    Unlike `_write_curve`, this fires `on_dump` once per dump -- exactly like
    probe.py's real loop -- with a val_nll ACTUALLY COMPUTED (via
    bllarse.metrics.nll) from the logits just written, so what patience sees
    can never drift from what early_stop later reads back off disk. If
    `on_dump` raises (patience's EarlyConvergence, or a caller's own
    exception), main stops immediately -- mirroring probe.py, which does not
    catch exceptions from on_dump.
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
            for dump_index, sharpness in enumerate(profiles[args.seed]):
                step = dump_index + 1
                logits = np.zeros((64, 2), dtype=np.float32)
                logits[:, 0] = sharpness
                for split in ("val", "test"):
                    evaldump.write_dump(
                        args.dump_root, args.group_id, args.uid, step, logits, labels,
                        arm="adamw_ce", split=split, logit_kind="softmax_logits",
                        seed=args.seed, samples_seen=step * 128,
                    )
                if on_dump is not None:
                    on_dump(dump_index, step, {"val_nll": metrics.nll(logits, labels)})

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
                on_dump(0, 1, {"val_nll": 0.5})

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
        on_dump=lambda i, step, scalars: None,
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
            "epochs": 5, "n_dumps": 4, "batch_size": 100, "no_hf_cache": True,
        },
        dump_root=tmp_path / "dumps", group_id="e2e", uid="trial0",
        on_dump=lambda i, step, s: reported.append(i),
    )

    assert value > 0.0
    assert reported == [0, 1, 2, 3], "intermediates come from the first seed only"
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
        patience=3, warmup_dumps=2, keep_dumps=True,
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
        patience=2, warmup_dumps=10, keep_dumps=True,
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
        on_dump=lambda i, step, scalars: None,
        patience=3, warmup_dumps=2, keep_dumps=True,
        on_converged=lambda seed, dump_index, step: converged.append(seed),
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
        on_dump=lambda i, step, scalars: calls.append((i, step)),
        patience=None, keep_dumps=True,
    )

    assert calls == [(i, i + 1) for i in range(len(profile))], (
        "on_dump must still fire once per dump, for the full profile"
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

    def pruning_on_dump(dump_index, step, scalars):
        if dump_index == 1:
            raise FakePrune("pruned")

    with pytest.raises(FakePrune):
        objective.evaluate(
            "adamw_ce", {}, dump_root=root, group_id="t", uid="trial0",
            on_dump=pruning_on_dump, patience=2, warmup_dumps=1,
        )

    # The try/finally cleanup must still have run despite the exception.
    assert not (root / "t" / "trial0_seed200").exists()
