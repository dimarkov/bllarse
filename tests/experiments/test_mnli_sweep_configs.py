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
