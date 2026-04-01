import numpy as np
import optax
from jax import nn
import jax.numpy as jnp
import importlib.util
import json
from pathlib import Path


def _load_script_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "roberta_mnli.py"
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script module at {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_make_cache_key_is_deterministic():
    mod = _load_script_module()
    key1 = mod.make_cache_key("FacebookAI/roberta-base", 256, "float16")
    key2 = mod.make_cache_key("FacebookAI/roberta-base", 256, "float16")
    key3 = mod.make_cache_key("FacebookAI/roberta-large", 256, "float16")
    key4 = mod.make_cache_key(
        "FacebookAI/roberta-base",
        256,
        "float16",
        max_train_samples=128,
        max_val_samples=64,
    )
    assert key1 == key2
    assert key1 != key3
    assert key1 != key4


def test_build_hf_cache_prefix_normalises_slashes():
    mod = _load_script_module()
    assert mod.build_hf_cache_prefix("mnli_roberta_cls/", "abc123") == "mnli_roberta_cls/abc123"
    assert mod.build_hf_cache_prefix("/mnli_roberta_cls", "abc123") == "mnli_roberta_cls/abc123"


def test_cache_satisfies_request_rejects_underfilled_full_cache(tmp_path):
    mod = _load_script_module()

    for name, filename in mod._CACHE_FILES.items():
        path = tmp_path / filename
        if name == "metadata":
            path.write_text(
                json.dumps(
                    {
                        "rows": {
                            "train": 32,
                            "validation_matched": 16,
                            "validation_mismatched": 16,
                        }
                    }
                ),
                encoding="utf-8",
            )
        else:
            np.savez_compressed(path, data=np.zeros((1, 1), dtype=np.float32))

    assert mod._cache_satisfies_request(
        tmp_path,
        max_train_samples=32,
        max_val_samples=16,
    )
    assert not mod._cache_satisfies_request(
        tmp_path,
        max_train_samples=None,
        max_val_samples=None,
    )


def test_hf_sync_flags():
    mod = _load_script_module()
    assert mod.hf_sync_flags("none") == (False, False)
    assert mod.hf_sync_flags("pull") == (True, False)
    assert mod.hf_sync_flags("push") == (False, True)
    assert mod.hf_sync_flags("pull_push") == (True, True)


def test_compute_metrics_from_logits_matches_expected_nll_and_acc():
    mod = _load_script_module()
    logits = jnp.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    labels = jnp.array([0, 1, 2, 1], dtype=jnp.int32)

    metrics = mod.compute_metrics_from_logits(logits, labels, num_bins=10)
    expected_nll = float(
        optax.softmax_cross_entropy(logits, nn.one_hot(labels, 3)).mean()
    )

    assert np.isclose(metrics["acc"], 0.75)
    assert np.isclose(metrics["nll"], expected_nll)
    assert 0.0 <= metrics["ece"] <= 1.0


def test_checkpoint_is_better_prefers_acc_then_nll_then_ece():
    mod = _load_script_module()

    best = {"acc": 0.8, "nll": 0.5, "ece": 0.2}
    assert mod.checkpoint_is_better({"acc": 0.81, "nll": 10.0, "ece": 1.0}, best)
    assert mod.checkpoint_is_better({"acc": 0.8, "nll": 0.4, "ece": 0.3}, best)
    assert mod.checkpoint_is_better({"acc": 0.8, "nll": 0.5, "ece": 0.1}, best)
    assert not mod.checkpoint_is_better({"acc": 0.79, "nll": 0.1, "ece": 0.0}, best)


def test_argparser_accepts_adam_and_dropout_rate():
    mod = _load_script_module()
    parser = mod.build_argparser()
    args = parser.parse_args(["--optimizer", "adam", "--dropout-rate", "0.1"])

    assert args.optimizer == "adam"
    assert np.isclose(args.dropout_rate, 0.1)


def test_argparser_accepts_eval_every_batches_and_subset_seed():
    mod = _load_script_module()
    parser = mod.build_argparser()
    args = parser.parse_args(
        ["--optimizer", "cavi", "--eval-every-batches", "1", "--train-subset-seed", "7"]
    )

    assert args.optimizer == "cavi"
    assert args.eval_every_batches == 1
    assert args.train_subset_seed == 7


def test_truncate_samples_uses_seeded_nested_train_prefix():
    mod = _load_script_module()
    arrays = {
        "X_train": np.arange(20, dtype=np.float32).reshape(10, 2),
        "y_train": np.arange(10, dtype=np.int32),
        "X_val_m": np.arange(12, dtype=np.float32).reshape(6, 2),
        "y_val_m": np.arange(6, dtype=np.int32),
        "X_val_mm": np.arange(12, dtype=np.float32).reshape(6, 2),
        "y_val_mm": np.arange(6, dtype=np.int32),
    }

    first = mod._truncate_samples(
        arrays,
        max_train_samples=4,
        max_val_samples=None,
        train_subset_seed=3,
    )
    second = mod._truncate_samples(
        arrays,
        max_train_samples=4,
        max_val_samples=None,
        train_subset_seed=3,
    )
    third = mod._truncate_samples(
        arrays,
        max_train_samples=4,
        max_val_samples=None,
        train_subset_seed=5,
    )

    assert np.array_equal(first["X_train"], second["X_train"])
    assert np.array_equal(first["y_train"], second["y_train"])
    assert not np.array_equal(first["X_train"], third["X_train"])
    assert not np.array_equal(first["y_train"], third["y_train"])


def test_train_eval_cache_root_ignores_subset_sizes():
    mod = _load_script_module()
    parser = mod.build_argparser()
    args = parser.parse_args(
        [
            "--stage",
            "train_eval",
            "--backbone",
            "FacebookAI/roberta-large",
            "--max-length",
            "512",
            "--cache-dtype",
            "float16",
            "--max-train-samples",
            "16384",
            "--max-val-samples",
            "1024",
        ]
    )
    cache_root = mod._resolve_cache_root(args)
    cache_key = cache_root.name

    assert "tr16384" not in cache_key
    assert "val1024" not in cache_key
