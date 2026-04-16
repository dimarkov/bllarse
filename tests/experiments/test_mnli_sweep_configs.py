import importlib.util
from pathlib import Path
from typing import Any


def _load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_optimizer_sweep_config_expansion():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = repo_root / "bllarse_sweeps" / "mnli_roberta_optimizer_sweep.py"
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) > 0

    optimizers = {cfg["optimizer"] for cfg in configs}
    assert optimizers == {"adamw", "lion", "cavi"}

    for cfg in configs:
        if cfg["optimizer"] == "cavi":
            assert cfg["num_update_iters"] in {1, 2, 4, 8, 16, 32}
        else:
            assert cfg["num_update_iters"] == 0


def test_smoke_sweep_contains_adamw_and_cavi():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = repo_root / "bllarse_sweeps" / "mnli_roberta_optimizer_smoke.py"
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) >= 2
    optimizers = {cfg["optimizer"] for cfg in configs}
    assert "adamw" in optimizers
    assert "cavi" in optimizers


def test_gluex_linear_probe_sweep_matches_expected_grid():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = repo_root / "bllarse_sweeps" / "mnli_roberta_gluex_linear_probe_sweep.py"
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 20

    optimizers = {cfg["optimizer"] for cfg in configs}
    assert optimizers == {"adam", "adamw"}

    for cfg in configs:
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 120
        assert cfg["dropout_rate"] == 0.1
        assert cfg["seed"] == 2022
        if cfg["optimizer"] == "adam":
            assert cfg["weight_decay"] == 0.0
        else:
            assert cfg["weight_decay"] in {0.0, 1e-4, 1e-3, 1e-2}


def test_gluex_linear_probe_smoke_contains_adam_and_adamw():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = repo_root / "bllarse_sweeps" / "mnli_roberta_gluex_linear_probe_smoke.py"
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 2
    optimizers = {cfg["optimizer"] for cfg in configs}
    assert optimizers == {"adam", "adamw"}

    for cfg in configs:
        assert cfg["stage"] == "all"
        assert cfg["max_length"] == 512
        assert cfg["train_batch_size"] == 8


def test_len256_linear_probe_baseline_configs():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = repo_root / "bllarse_sweeps" / "mnli_roberta_len256_linear_probe_baseline.py"
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 4
    optimizers = {cfg["optimizer"] for cfg in configs}
    assert optimizers == {"adam", "adamw"}

    for cfg in configs:
        assert cfg["max_length"] == 256
        assert cfg["epochs"] == 5
        assert cfg["train_batch_size"] == 8
        assert cfg["seed"] == 2022


def test_len256_linear_probe_multiseed_configs():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = repo_root / "bllarse_sweeps" / "mnli_roberta_len256_linear_probe_multiseed.py"
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 24
    optimizers = {cfg["optimizer"] for cfg in configs}
    assert optimizers == {"adam", "adamw"}
    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024}
    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {4, 8}

    for cfg in configs:
        assert cfg["max_length"] == 256
        assert cfg["epochs"] == 120
        if cfg["optimizer"] == "adam":
            assert cfg["weight_decay"] == 0.0
        else:
            assert cfg["weight_decay"] == 1e-2


def test_len512_extract_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = repo_root / "bllarse_sweeps" / "mnli_roberta_len512_extract.py"
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 1
    cfg = configs[0]
    assert cfg["stage"] == "extract"
    assert cfg["max_length"] == 512
    assert cfg["hf_sync"] == "pull_push"
    assert cfg["extract_batch_size"] == 256
    assert cfg["enable_mlflow"] is False


def test_len512_extract_batchsize_benchmark_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_len512_extract_batchsize_benchmark.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 3
    batch_sizes = {cfg["extract_batch_size"] for cfg in configs}
    assert batch_sizes == {32, 64, 128}

    for cfg in configs:
        assert cfg["stage"] == "extract"
        assert cfg["max_length"] == 512
        assert cfg["hf_sync"] == "none"
        assert cfg["reuse_cache"] is False
        assert cfg["enable_mlflow"] is False
        assert cfg["max_train_samples"] == 32768
        assert cfg["max_val_samples"] == 2048


def test_len512_largebatch_adam_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_len512_linear_probe_largebatch_adam.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 6

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {1024, 2048, 4096}

    learning_rates = {cfg["learning_rate"] for cfg in configs}
    assert learning_rates == {3e-5, 1e-4}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "adam"
        assert cfg["weight_decay"] == 0.0
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 40
        assert cfg["dropout_rate"] == 0.1
        assert cfg["seed"] == 2022


def test_len512_largebatch_adam_long_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_len512_linear_probe_largebatch_adam_long.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 6

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {2048, 4096}

    learning_rates = {cfg["learning_rate"] for cfg in configs}
    assert learning_rates == {1e-4, 3e-4, 1e-3}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "adam"
        assert cfg["weight_decay"] == 0.0
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 120
        assert cfg["dropout_rate"] == 0.1
        assert cfg["seed"] == 2022


def test_len512_bs2048_refine_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_len512_linear_probe_bs2048_refine.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 15

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {2048}

    learning_rates = {cfg["learning_rate"] for cfg in configs}
    assert learning_rates == {7e-4, 1e-3, 1.3e-3}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "adam"
        assert cfg["weight_decay"] == 0.0
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 100
        assert cfg["dropout_rate"] == 0.1


def test_roberta_large_len512_extract_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_extract.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 1
    cfg = configs[0]
    assert cfg["stage"] == "extract"
    assert cfg["backbone"] == "FacebookAI/roberta-large"
    assert cfg["max_length"] == 512
    assert cfg["hf_sync"] == "pull_push"
    assert cfg["extract_batch_size"] == 64
    assert cfg["enable_mlflow"] is False


def test_roberta_large_len512_extract_batchsize_benchmark_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_extract_batchsize_benchmark.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 4
    batch_sizes = {cfg["extract_batch_size"] for cfg in configs}
    assert batch_sizes == {32, 64, 96, 128}

    for cfg in configs:
        assert cfg["stage"] == "extract"
        assert cfg["backbone"] == "FacebookAI/roberta-large"
        assert cfg["max_length"] == 512
        assert cfg["hf_sync"] == "none"
        assert cfg["reuse_cache"] is False
        assert cfg["enable_mlflow"] is False
        assert cfg["max_train_samples"] == 32768
        assert cfg["max_val_samples"] == 2048


def test_roberta_large_len512_bs2048_refine_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_linear_probe_bs2048_refine.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 15

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {2048}

    learning_rates = {cfg["learning_rate"] for cfg in configs}
    assert learning_rates == {7e-4, 1e-3, 1.3e-3}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "adam"
        assert cfg["weight_decay"] == 0.0
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 100
        assert cfg["dropout_rate"] == 0.1


def test_roberta_large_len512_regime_search_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_linear_probe_regime_search.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 24

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {256, 512, 1024, 2048}

    learning_rates = {cfg["learning_rate"] for cfg in configs}
    assert learning_rates == {3e-4, 7e-4, 1e-3}

    dropout_rates = {cfg["dropout_rate"] for cfg in configs}
    assert dropout_rates == {0.0, 0.1}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "adam"
        assert cfg["weight_decay"] == 0.0
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 120


def test_roberta_large_len512_regime_refine_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_linear_probe_regime_refine.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 24

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {128, 256, 384, 512}

    learning_rates = {cfg["learning_rate"] for cfg in configs}
    assert learning_rates == {8e-4, 1e-3, 1.2e-3}

    dropout_rates = {cfg["dropout_rate"] for cfg in configs}
    assert dropout_rates == {0.0, 0.02}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "adam"
        assert cfg["weight_decay"] == 0.0
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 140


def test_roberta_large_len512_bs128_confirm_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_linear_probe_bs128_confirm.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 10

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {128}

    learning_rates = {cfg["learning_rate"] for cfg in configs}
    assert learning_rates == {1e-3, 1.2e-3}

    dropout_rates = {cfg["dropout_rate"] for cfg in configs}
    assert dropout_rates == {0.0}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "adam"
        assert cfg["weight_decay"] == 0.0
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 140


def test_roberta_large_len512_largebatch_adam_adamw_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_linear_probe_largebatch_adam_adamw.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 60

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {512, 1024, 2048, 4096, 8192, 16384}

    optimizers = {cfg["optimizer"] for cfg in configs}
    assert optimizers == {"adam", "adamw"}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 140
        assert cfg["learning_rate"] == 1.2e-3
        assert cfg["dropout_rate"] == 0.0
        assert cfg["num_update_iters"] == 0
        if cfg["optimizer"] == "adam":
            assert cfg["weight_decay"] == 0.0
        else:
            assert cfg["weight_decay"] == 1e-2


def test_roberta_large_len512_ibprobit_initial_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_ibprobit_initial.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 6

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {1024, 2048, 4096}

    num_iters = {cfg["num_update_iters"] for cfg in configs}
    assert num_iters == {16, 32}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "cavi"
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 20
        assert cfg["dropout_rate"] == 0.0


def test_roberta_large_len512_ibprobit_singlepass_multiseed_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_ibprobit_singlepass_multiseed.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 105

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {2048, 4096, 8192}

    num_iters = {cfg["num_update_iters"] for cfg in configs}
    assert num_iters == {1, 4, 8, 16, 32, 64, 128}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "cavi"
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 1
        assert cfg["dropout_rate"] == 0.0
        assert cfg["reset_loss_per_epoch"] is False


def test_roberta_large_len512_ibprobit_singlepass_multiseed_expanded_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_ibprobit_singlepass_multiseed_expanded.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 280

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {128, 256, 512, 1024, 2048, 4096, 8192, 16384}

    num_iters = {cfg["num_update_iters"] for cfg in configs}
    assert num_iters == {1, 4, 8, 16, 32, 64, 128}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "cavi"
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 1
        assert cfg["dropout_rate"] == 0.0
        assert cfg["reset_loss_per_epoch"] is False


def test_roberta_large_len512_ibprobit_singlepass_multiseed_expanded_seed0to4_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_ibprobit_singlepass_multiseed_expanded_seed0to4.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 280

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {128, 256, 512, 1024, 2048, 4096, 8192, 16384}

    num_iters = {cfg["num_update_iters"] for cfg in configs}
    assert num_iters == {1, 4, 8, 16, 32, 64, 128}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {0, 1, 2, 3, 4}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "cavi"
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 1
        assert cfg["dropout_rate"] == 0.0
        assert cfg["reset_loss_per_epoch"] is False


def test_roberta_large_len512_ibprobit_singlepass_multiseed_expanded_alpha1e2_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_ibprobit_singlepass_multiseed_expanded_alpha1e2.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 280

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {128, 256, 512, 1024, 2048, 4096, 8192, 16384}

    alphas = {cfg["ibprobit_alpha"] for cfg in configs}
    assert alphas == {1e-2}

    num_iters = {cfg["num_update_iters"] for cfg in configs}
    assert num_iters == {1, 4, 8, 16, 32, 64, 128}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "cavi"
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 1
        assert cfg["dropout_rate"] == 0.0
        assert cfg["reset_loss_per_epoch"] is False


def test_roberta_large_len512_ibprobit_singlepass_multiseed_expanded_alpha1e3_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_ibprobit_singlepass_multiseed_expanded_alpha1e3.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 280

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {128, 256, 512, 1024, 2048, 4096, 8192, 16384}

    alphas = {cfg["ibprobit_alpha"] for cfg in configs}
    assert alphas == {1e-3}

    num_iters = {cfg["num_update_iters"] for cfg in configs}
    assert num_iters == {1, 4, 8, 16, 32, 64, 128}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "cavi"
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 1
        assert cfg["dropout_rate"] == 0.0
        assert cfg["reset_loss_per_epoch"] is False


def test_roberta_large_len512_ibprobit_singlepass_multiseed_expanded_alpha1e4_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_ibprobit_singlepass_multiseed_expanded_alpha1e4.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 280

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {128, 256, 512, 1024, 2048, 4096, 8192, 16384}

    alphas = {cfg["ibprobit_alpha"] for cfg in configs}
    assert alphas == {1e-4}

    num_iters = {cfg["num_update_iters"] for cfg in configs}
    assert num_iters == {1, 4, 8, 16, 32, 64, 128}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "cavi"
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 1
        assert cfg["dropout_rate"] == 0.0
        assert cfg["reset_loss_per_epoch"] is False


def test_roberta_large_len512_ibprobit_singlepass_multiseed_expanded_alpha1e2_iters256_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_ibprobit_singlepass_multiseed_expanded_alpha1e2_iters256.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 240

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {512, 1024, 2048, 4096, 8192, 16384}

    alphas = {cfg["ibprobit_alpha"] for cfg in configs}
    assert alphas == {1e-2}

    num_iters = {cfg["num_update_iters"] for cfg in configs}
    assert num_iters == {1, 4, 8, 16, 32, 64, 128, 256}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "cavi"
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 1
        assert cfg["dropout_rate"] == 0.0
        assert cfg["reset_loss_per_epoch"] is False


def test_roberta_large_len512_ibprobit_singlepass_multiseed_expanded_alpha1e2_iters6_only_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_ibprobit_singlepass_multiseed_expanded_alpha1e2_iters6_only.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 30

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {512, 1024, 2048, 4096, 8192, 16384}

    alphas = {cfg["ibprobit_alpha"] for cfg in configs}
    assert alphas == {1e-2}

    num_iters = {cfg["num_update_iters"] for cfg in configs}
    assert num_iters == {6}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "cavi"
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 1
        assert cfg["dropout_rate"] == 0.0
        assert cfg["reset_loss_per_epoch"] is False


def test_roberta_large_len512_linear_probe_largebatch_adam_lr_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_linear_probe_largebatch_adam_lr_sweep.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 120

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {512, 1024, 2048, 4096, 8192, 16384}

    learning_rates = {cfg["learning_rate"] for cfg in configs}
    assert learning_rates == {1e-4, 3e-4, 6e-4, 1e-3}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "adam"
        assert cfg["weight_decay"] == 0.0
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 140
        assert cfg["dropout_rate"] == 0.0
        assert cfg["num_update_iters"] == 0


def test_roberta_large_len512_linear_probe_largebatch_adam_lr_dropout_refine_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_linear_probe_largebatch_adam_lr_dropout_refine.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 180

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {4096, 8192, 16384}

    learning_rates = {cfg["learning_rate"] for cfg in configs}
    assert learning_rates == {8e-4, 1e-3, 1.2e-3, 1.5e-3}

    dropout_rates = {cfg["dropout_rate"] for cfg in configs}
    assert dropout_rates == {0.0, 0.02, 0.05}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "adam"
        assert cfg["weight_decay"] == 0.0
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 140
        assert cfg["num_update_iters"] == 0


def test_roberta_large_len512_linear_probe_largebatch_adam_lr_high_refine_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_linear_probe_largebatch_adam_lr_high_refine.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 75

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {4096, 8192, 16384}

    learning_rates = {cfg["learning_rate"] for cfg in configs}
    assert learning_rates == {1.5e-3, 1.8e-3, 2.0e-3, 2.5e-3, 3.0e-3}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "adam"
        assert cfg["weight_decay"] == 0.0
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 140
        assert cfg["dropout_rate"] == 0.0
        assert cfg["num_update_iters"] == 0


def test_roberta_large_len512_linear_probe_largebatch_adam_lr_upper_refine_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_linear_probe_largebatch_adam_lr_upper_refine.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 75

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {4096, 8192, 16384}

    learning_rates = {cfg["learning_rate"] for cfg in configs}
    assert learning_rates == {3.0e-3, 4.0e-3, 5.0e-3, 6.0e-3, 8.0e-3}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "adam"
        assert cfg["weight_decay"] == 0.0
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 140
        assert cfg["dropout_rate"] == 0.0
        assert cfg["num_update_iters"] == 0


def test_roberta_large_len512_linear_probe_largebatch_adam_lr_extreme_refine_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_linear_probe_largebatch_adam_lr_extreme_refine.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 60

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {4096, 8192, 16384}

    learning_rates = {cfg["learning_rate"] for cfg in configs}
    assert learning_rates == {8.0e-3, 1.0e-2, 1.2e-2, 1.5e-2}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "adam"
        assert cfg["weight_decay"] == 0.0
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 140
        assert cfg["dropout_rate"] == 0.0
        assert cfg["num_update_iters"] == 0


def test_roberta_large_len512_ibprobit_bs1024_recheck_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_ibprobit_bs1024_recheck.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 35

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {1024}

    num_iters = {cfg["num_update_iters"] for cfg in configs}
    assert num_iters == {1, 4, 8, 16, 32, 64, 128}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "cavi"
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 1
        assert cfg["dropout_rate"] == 0.0
        assert cfg["reset_loss_per_epoch"] is False


def test_roberta_large_len512_ibprobit_bs1024_altseeds_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_ibprobit_bs1024_altseeds.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 35

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {1024}

    num_iters = {cfg["num_update_iters"] for cfg in configs}
    assert num_iters == {1, 4, 8, 16, 32, 64, 128}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2027, 2028, 2029, 2030, 2031}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "cavi"
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 1
        assert cfg["dropout_rate"] == 0.0
        assert cfg["reset_loss_per_epoch"] is False


def test_roberta_large_len512_ibprobit_bs1024_alpha_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_ibprobit_bs1024_alpha_sweep.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 140

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {1024}

    alphas = {cfg["ibprobit_alpha"] for cfg in configs}
    assert alphas == {1e-2, 3e-3, 1e-3, 3e-4}

    num_iters = {cfg["num_update_iters"] for cfg in configs}
    assert num_iters == {1, 4, 8, 16, 32, 64, 128}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "cavi"
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 1
        assert cfg["dropout_rate"] == 0.0
        assert cfg["reset_loss_per_epoch"] is False


def test_roberta_large_len512_ibprobit_data_efficiency_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_ibprobit_data_efficiency.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 5

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {16384}

    num_iters = {cfg["num_update_iters"] for cfg in configs}
    assert num_iters == {16}

    alphas = {cfg["ibprobit_alpha"] for cfg in configs}
    assert alphas == {1e-2}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "cavi"
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 1
        assert cfg["dropout_rate"] == 0.0
        assert cfg["eval_every_batches"] == 1
        assert cfg["reset_loss_per_epoch"] is False


def test_roberta_large_len512_ibprobit_data_efficiency_alpha1e3_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_ibprobit_data_efficiency_alpha1e3.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 5

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {16384}

    num_iters = {cfg["num_update_iters"] for cfg in configs}
    assert num_iters == {16}

    alphas = {cfg["ibprobit_alpha"] for cfg in configs}
    assert alphas == {1e-3}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "cavi"
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 1
        assert cfg["dropout_rate"] == 0.0
        assert cfg["eval_every_batches"] == 1
        assert cfg["reset_loss_per_epoch"] is False


def test_roberta_large_len512_ibprobit_data_efficiency_iters64_alpha1e2_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_ibprobit_data_efficiency_iters64_alpha1e2.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 5

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {16384}

    num_iters = {cfg["num_update_iters"] for cfg in configs}
    assert num_iters == {64}

    alphas = {cfg["ibprobit_alpha"] for cfg in configs}
    assert alphas == {1e-2}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "cavi"
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 1
        assert cfg["dropout_rate"] == 0.0
        assert cfg["eval_every_batches"] == 1
        assert cfg["reset_loss_per_epoch"] is False


def test_roberta_large_len512_ibprobit_data_efficiency_iters64_alpha1e3_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_ibprobit_data_efficiency_iters64_alpha1e3.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 5

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {16384}

    num_iters = {cfg["num_update_iters"] for cfg in configs}
    assert num_iters == {64}

    alphas = {cfg["ibprobit_alpha"] for cfg in configs}
    assert alphas == {1e-3}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "cavi"
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 1
        assert cfg["dropout_rate"] == 0.0
        assert cfg["eval_every_batches"] == 1
        assert cfg["reset_loss_per_epoch"] is False


def test_roberta_large_len512_linear_probe_bs512_refine_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_linear_probe_bs512_refine.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 20

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {512}

    learning_rates = {cfg["learning_rate"] for cfg in configs}
    assert learning_rates == {8e-4, 1e-3, 1.2e-3, 1.5e-3}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "adam"
        assert cfg["weight_decay"] == 0.0
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 140
        assert cfg["dropout_rate"] == 0.0
        assert cfg["num_update_iters"] == 0


def test_roberta_large_len512_linear_probe_bs512_full_tune_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_linear_probe_bs512_full_tune.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 120

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {512}

    learning_rates = {cfg["learning_rate"] for cfg in configs}
    assert learning_rates == {1e-3, 1.2e-3, 1.5e-3, 1.8e-3, 2.2e-3, 2.8e-3}

    dropout_rates = {cfg["dropout_rate"] for cfg in configs}
    assert dropout_rates == {0.0, 0.01, 0.02, 0.05}

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "adam"
        assert cfg["weight_decay"] == 0.0
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 140
        assert cfg["num_update_iters"] == 0


def test_roberta_large_len512_linear_probe_bs512_data_efficiency_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_linear_probe_bs512_data_efficiency.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 120

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {512}

    learning_rates = {cfg["learning_rate"] for cfg in configs}
    assert learning_rates == {1.2e-3}

    subset_sizes = {cfg["max_train_samples"] for cfg in configs}
    assert min(subset_sizes) == 16384
    assert max(subset_sizes) == 392702
    assert len(subset_sizes) == 24

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    subset_seed_values = {cfg["train_subset_seed"] for cfg in configs}
    assert subset_seed_values == seeds

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "adam"
        assert cfg["weight_decay"] == 0.0
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 140
        assert cfg["dropout_rate"] == 0.0
        assert cfg["num_update_iters"] == 0


def test_roberta_large_len512_linear_probe_bs512_data_efficiency_retuned_sweep_config():
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = (
        repo_root
        / "bllarse_sweeps"
        / "mnli_roberta_large_len512_linear_probe_bs512_data_efficiency_retuned.py"
    )
    sweep = _load_module(sweep_path)

    configs = sweep.create_configs()
    assert len(configs) == 120

    backbones = {cfg["backbone"] for cfg in configs}
    assert backbones == {"FacebookAI/roberta-large"}

    batch_sizes = {cfg["train_batch_size"] for cfg in configs}
    assert batch_sizes == {512}

    learning_rates = {cfg["learning_rate"] for cfg in configs}
    assert learning_rates == {2.8e-3}

    subset_sizes = {cfg["max_train_samples"] for cfg in configs}
    assert min(subset_sizes) == 16384
    assert max(subset_sizes) == 392702
    assert len(subset_sizes) == 24

    seeds = {cfg["seed"] for cfg in configs}
    assert seeds == {2022, 2023, 2024, 2025, 2026}

    subset_seed_values = {cfg["train_subset_seed"] for cfg in configs}
    assert subset_seed_values == seeds

    for cfg in configs:
        assert cfg["stage"] == "train_eval"
        assert cfg["optimizer"] == "adam"
        assert cfg["weight_decay"] == 0.0
        assert cfg["max_length"] == 512
        assert cfg["epochs"] == 140
        assert cfg["dropout_rate"] == 0.0
        assert cfg["num_update_iters"] == 0
