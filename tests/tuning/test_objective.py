import numpy as np
import pytest

from bllarse import evaldump
from bllarse.tuning import objective


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
