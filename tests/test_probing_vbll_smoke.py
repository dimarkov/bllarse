import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vbll")

VBLL_DIR = Path(__file__).resolve().parents[1] / "scripts" / "vbll_pytorch"
# `probing_vbll.py` lives flat in VBLL_DIR; `vbll_experiments` lives under its
# `src/` layout (mirroring the package's own pyproject.toml). Both need to be
# on sys.path -- the first test below imports vbll_experiments directly,
# without going through probing_vbll.py's own sys.path setup.
sys.path.insert(0, str(VBLL_DIR))
sys.path.insert(0, str(VBLL_DIR / "src"))

from bllarse import evaldump


@pytest.fixture
def fake_features(tmp_path):
    rng = np.random.default_rng(0)
    cache = tmp_path / "features"
    cache.mkdir()
    for split, n in (("train", 600), ("test", 200)):
        labels = rng.integers(0, 2, size=n)
        feats = rng.normal(size=(n, 16)).astype(np.float32)
        # Class-symmetric offset (+2.0 for label 1, -2.0 for label 0), not a
        # one-sided `feats[labels == 1, 0] += 4.0`. An earlier version of this
        # fixture used a one-sided offset and it made VBLL specifically (not
        # AdamW+CE) plateau at ~0.70 test accuracy no matter the epoch count,
        # batch size, or regularization weight -- traced to VBLL's Jensen
        # bound, which adds a term scaled by E[x**2] per feature. A one-sided
        # shift drags that feature's pooled *mean* away from 0, inflating
        # E[x**2] by a squared-bias term unrelated to the actual class
        # separation, and that spurious inflation dominates the gradient on
        # the feature's posterior variance rather than on its
        # class-discriminative mean weight. A symmetric shift keeps the
        # pooled mean at 0 -- as real cached backbone features naturally are,
        # since a feature's marginal statistics come from the backbone, not
        # from any downstream label -- and resolves it completely with no
        # other change (verified: 0.98 accuracy at 1000 epochs here, vs. 0.70
        # for the one-sided version). It also keeps per-dimension std
        # well-conditioned (max/min ratio ~2.2 here, vs. ~3.2 one-sided for
        # the same separation), matching how real dinov3_small features
        # looked when checked directly (per-dim std range 0.115-1.331, no
        # single dimension dominating the rest).
        feats[labels == 1, 0] += 2.0
        feats[labels == 0, 0] -= 2.0
        np.savez(cache / f"dinov3_small_fake_res224_{split}.npz",
                 features=feats, labels=labels)
    return cache


def test_models_import_without_scaling_mlps_submodule():
    """The head-only path must not require the deep-MLP submodule, which belongs
    to the excised experiment."""
    from vbll_experiments import models
    assert hasattr(models, "build_vbll_head")


def test_vbll_probe_writes_dumps_and_learns(fake_features, tmp_path):
    import probing_vbll

    args = probing_vbll.build_argparser().parse_args([
        "--model", "dinov3_small",
        "--dataset", "fake",
        "--num-classes", "2",
        "--cache-dir", str(fake_features),
        "--split-cache-dir", str(tmp_path / "splits"),
        "--dump-root", str(tmp_path / "dumps"),
        "--group-id", "smoke",
        "--uid", "vbll_0",
        # NOTE: raised from the brief's --epochs 20, mirroring the same fix
        # tests/test_probe_smoke.py needed for the identical row/batch-size
        # combination: at batch-size 512 with 600 training rows there is 1
        # gradient step per epoch, and VBLL's ELBO converges slower than
        # plain CE on top of that. Verified directly (scripts/vbll_pytorch,
        # this exact fixture and seed, no transform of any kind applied to
        # the features): 0.98 test accuracy by epoch 1000.
        "--epochs", "1000",
        "--n-dumps", "5",
        "--seed", "137",
        "--device", "cpu",
    ])
    probing_vbll.main(args)

    records = list(evaldump.iter_dumps(tmp_path / "dumps", "smoke", "vbll_0"))
    assert len(records) >= 2
    assert records[0].meta["arm"] == "vbll"
    assert records[0].meta["logit_kind"] == "softmax_logits"
    assert records[0].meta["score_kind"] == "predictive_variance"
    assert records[0].score is not None, "VBLL must carry its native epistemic score"

    from bllarse import metrics
    final = [r for r in records if r.meta["split"] == "test"][-1]
    assert metrics.accuracy(final.logits, final.labels) > 0.9


@pytest.mark.parametrize("no_hf_cache", [False, True])
def test_hf_repo_flag_threads_through_every_load_cached_features_call(
    fake_features, tmp_path, monkeypatch, no_hf_cache
):
    """Cross-cutting fix: probing_vbll.py must expose --hf-repo/--no-hf-cache
    and thread hf_repo through the train, eval, and OOD load_cached_features
    call sites -- matching scripts/alternatives/probe.py. Before this fix, the
    VBLL arm had no way to fetch features from the Hub at all, so any cluster
    node without a pre-staged .cache/features would fail with
    FileNotFoundError on every VBLL task while the JAX arm succeeded."""
    import probing_vbll

    calls = []
    real_load = probing_vbll.load_cached_features

    def spy(cache_dir, model, dataset, img_size, split, hf_repo=None):
        calls.append(hf_repo)
        return real_load(cache_dir, model, dataset, img_size, split, hf_repo=hf_repo)

    monkeypatch.setattr(probing_vbll, "load_cached_features", spy)

    argv = [
        "--model", "dinov3_small",
        "--dataset", "fake",
        "--ood-dataset", "fake",
        "--num-classes", "2",
        "--cache-dir", str(fake_features),
        "--split-cache-dir", str(tmp_path / "splits"),
        "--dump-root", str(tmp_path / "dumps"),
        "--group-id", "smoke",
        "--uid", "vbll_hf",
        "--epochs", "1",
        "--n-dumps", "2",
        "--seed", "137",
        "--device", "cpu",
    ]
    if no_hf_cache:
        argv.append("--no-hf-cache")
    args = probing_vbll.build_argparser().parse_args(argv)
    probing_vbll.main(args)

    assert len(calls) == 3, "train, eval, and ood loads must all go through load_cached_features"
    expected = None if no_hf_cache else "dimarkov/bllarse-features"
    assert calls == [expected] * 3
