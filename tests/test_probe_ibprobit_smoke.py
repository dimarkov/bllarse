"""End-to-end cover for probe.py's ibprobit arm.

The point of these is that the arm's dumps have to survive the REAL scoring
path, not merely be written: `curves.run_curve` refuses a logit_kind it cannot
metricise, and `objective.score_dumps` is what a tuning trial's value comes
from. A run that trains perfectly and dumps something run_curve rejects fails
every trial of a campaign after all its GPU time has been spent.

These need a GPU: bllarse.losses.bayesian.mm_loss pins the TF32_TF32_F32 dot
algorithm, which the JAX CPU backend rejects outright.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ALT = Path(__file__).resolve().parents[1] / "scripts" / "dinov3"
sys.path.insert(0, str(ALT))

from bllarse import evaldump
from bllarse.analysis.curves import run_curve
from bllarse.tuning import objective


@pytest.fixture
def fake_features(tmp_path):
    """Three separated blobs. Three classes rather than two so a bug that
    collapses the class axis (e.g. dumping a (n,) score where (n,K) logits
    belong) cannot pass by coincidence on a binary problem."""
    rng = np.random.default_rng(0)
    cache = tmp_path / "features"
    cache.mkdir()
    for split, n in (("train", 600), ("test", 300)):
        labels = rng.integers(0, 3, size=n)
        feats = rng.normal(size=(n, 16)).astype(np.float32)
        feats[labels == 1, 0] += 6.0
        feats[labels == 2, 1] += 6.0
        np.savez(cache / f"dinov3_small_fake_res224_{split}.npz",
                 features=feats, labels=labels)
    return cache


def _run(tmp_path, cache, uid, extra=()):
    import probe

    args = probe.build_argparser().parse_args([
        "--arm", "ibprobit",
        "--model", "dinov3_small",
        "--dataset", "fake",
        "--num-classes", "3",
        "--cache-dir", str(cache),
        "--split-cache-dir", str(tmp_path / "splits"),
        "--dump-root", str(tmp_path / "dumps"),
        "--group-id", "smoke",
        "--uid", uid,
        "--batch-size", "64",
        "--epochs", "12",
        "--n-dumps", "5",
        "--num-update-iters", "32",
        "--px",
        "--seed", "137",
        "--no-hf-cache",
        *extra,
    ])
    probe.main(args)
    return list(evaldump.iter_dumps(tmp_path / "dumps", "smoke", uid))


def test_the_ibprobit_arm_trains_and_dumps(fake_features, tmp_path):
    records = _run(tmp_path, fake_features, "ib_0")

    assert records, "the arm wrote no dumps at all"
    assert {r.meta["split"] for r in records} >= {"val", "test"}
    assert all(r.meta["arm"] == "ibprobit" for r in records)
    assert all(r.logits.shape[1] == 3 for r in records)

    from bllarse import metrics
    final = [r for r in records if r.meta["split"] == "test"][-1]
    assert metrics.accuracy(final.logits, final.labels) > 0.9


def test_its_dumps_survive_the_real_scoring_path(fake_features, tmp_path):
    """run_curve softmaxes and only accepts logit kinds it knows how to. Under
    the old "moderated_score" label this raised, which would have failed every
    trial of a campaign whose training all succeeded."""
    _run(tmp_path, fake_features, "ib_1")

    rows = run_curve(tmp_path / "dumps", "smoke", "ib_1")
    assert rows, "run_curve found no scoreable rows"
    assert all(np.isfinite(row["val_nll"]) for row in rows)
    assert all(row["arm"] == "ibprobit" for row in rows)

    # ...and the value a tuning trial would actually report.
    value = objective.score_dumps(tmp_path / "dumps", "smoke", ["ib_1"])
    assert np.isfinite(value)


def test_the_dumped_logits_are_a_normalized_log_predictive(fake_features, tmp_path):
    """Which is WHY "softmax_logits" is the right label despite the name.

    IBProbit.__call__ returns _mixture_logp, and (1-eps)*1 + K*(eps/K) == 1, so
    each row sums to 1 in probability space and softmaxing it is a no-op. If
    that ever stops holding, classification_metrics silently re-normalizes a
    second time and every NLL in the figures shifts.
    """
    records = _run(tmp_path, fake_features, "ib_2")
    logits = [r for r in records if r.meta["split"] == "test"][-1].logits
    total = np.exp(np.asarray(logits, dtype=np.float64)).sum(axis=-1)
    np.testing.assert_allclose(total, 1.0, atol=2e-3)


def test_it_dumps_a_native_score_and_the_contamination_estimate(fake_features, tmp_path):
    """Both are IBProbit-only. curves.run_native_detection raises without the
    score, so Fig 2's native column would be unavailable for the arm whose
    epistemic term the paper names as the mechanism."""
    records = _run(tmp_path, fake_features, "ib_3")

    for record in records:
        assert record.score is not None, f"{record.meta['split']} dump carries no score"
        assert record.meta["score_kind"] == "predictive_variance"
        assert record.score.shape == record.labels.shape
        # 1 + f'(Qhat + Sigma)f, and both matrices are PSD.
        assert np.all(record.score >= 1.0 - 1e-4)
        assert 0.0 < record.meta["eps_hat"] < 1.0


def test_the_score_matches_the_split_it_was_dumped_with(fake_features, tmp_path):
    """val and test are disjoint index sets taken from ONE forward pass, so the
    score has to be sliced by the same indices as the logits. Slicing one and
    not the other pairs each point's logits with a stranger's uncertainty --
    which still produces a plausible AUROC."""
    records = _run(tmp_path, fake_features, "ib_4")
    by_split = {}
    for record in records:
        by_split.setdefault(record.meta["split"], []).append(record)

    val = by_split["val"][-1]
    test = by_split["test"][-1]
    assert len(val.score) == len(val.labels)
    assert len(test.score) == len(test.labels)
    assert len(val.labels) != len(test.labels), "the fixture must not split evenly"


def test_a_whole_optuna_trial_runs_end_to_end(fake_features, tmp_path, monkeypatch):
    """The seam the unit tests each see only half of.

    `objective.evaluate` looks the arm up in SCRIPTS, imports that script,
    re-parses a config through its argparser, calls main(on_dump=...) once per
    tuning seed, and scores the dumps. Every one of those can break
    independently, and all of them break AFTER the trial has spent its GPU
    time -- at which point `study.optimize(catch=(Exception,))` records an
    ordinary failed trial and the campaign keeps going, so a broken seam looks
    like a bad search space rather than a harness fault.
    """
    from bllarse.tools import adapters
    from bllarse.tuning import worker

    monkeypatch.setenv("BLLARSE_REPO_ROOT", str(Path(__file__).resolve().parents[1]))

    config = worker.trial_config(
        "ibprobit", "dtd",
        params={"batch_size": 64, "q_init": 3e-4, "eps_init": 0.02,
                "num_update_iters": 32},
        budgets={"dtd": {"batch_size": 512, "steps": 100, "epochs": 1}},
        cache_dir=str(fake_features),
        split_cache_dir=str(tmp_path / "splits"),
        enable_mlflow=False,
    )
    # The study's dataset is dtd (that is what sets the budget); the fixture
    # stands in for its cached features, so the loaded names have to match it.
    config.update({"model": "dinov3_small", "dataset": "fake",
                   "num_classes": 3, "img_size": 224, "epochs": 12,
                   "n_dumps": 5, "no_hf_cache": True})
    # `arm` must survive: adapters.config_to_argv turns it into --arm, and
    # dropping it would silently run the adamw_ce arm under an ibprobit study.
    assert config["arm"] == "ibprobit"

    seen = []
    value = objective.evaluate(
        "ibprobit", config,
        dump_root=tmp_path / "dumps", group_id="trial", uid="t0",
        on_dump=lambda i, step, samples, scalars: seen.append((i, step)),
        keep_dumps=True,
    )
    assert np.isfinite(value), "the trial produced no scoreable objective value"
    assert seen, "on_dump never fired, so pruning would never engage"

    # Both tuning seeds ran and each wrote its own dumps.
    for seed in objective.TUNING_SEEDS:
        assert list(evaldump.iter_dumps(tmp_path / "dumps", "trial", f"t0_seed{seed}"))

    del adapters  # imported for the module-resolution side of the seam only


@pytest.mark.parametrize("flag,expected_learn_q", [([], True), (["--no-learn-q"], False)])
def test_the_learn_q_flag_reaches_the_constructor(
    fake_features, tmp_path, monkeypatch, flag, expected_learn_q
):
    """`learn_q=not args.no_learn_q` is one expression, and getting it backwards
    is invisible downstream: the dumps carry no Qhat, and `_q_recursion`'s own
    MEASURED LIMITATION says the node barely moves at these operating points --
    so an inverted flag would produce a run that trains, dumps, scores and looks
    entirely healthy while pinning Q when it should infer it, or the reverse.

    The tuning campaign pins it OFF (worker.IBPROBIT_FIXED), which makes this
    the only place the wiring is checked at all.
    """
    import probe

    captured = {}
    real = probe.IBProbit

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(probe, "IBProbit", spy)
    _run(tmp_path, fake_features, f"ib_lq_{expected_learn_q}", extra=flag)

    assert captured["learn_q"] is expected_learn_q
    # ...and the neighbouring fixed knobs are not being crossed with it.
    assert captured["px"] is True
    assert captured["kappa"] is None


def test_learning_rate_is_inert_on_this_arm(fake_features, tmp_path):
    """The arm runs no optimizer, so --learning-rate must not reach anything.

    This is the behavioural counterpart to space.suggest not sampling it: if a
    future edit gave the ibprobit branch an optimizer, the search space would
    silently stop covering a live axis.
    """
    slow = _run(tmp_path, fake_features, "ib_lr_slow", extra=["--learning-rate", "1e-8"])
    fast = _run(tmp_path, fake_features, "ib_lr_fast", extra=["--learning-rate", "1e-1"])

    slow_final = [r for r in slow if r.meta["split"] == "test"][-1]
    fast_final = [r for r in fast if r.meta["split"] == "test"][-1]
    np.testing.assert_array_equal(slow_final.logits, fast_final.logits)
