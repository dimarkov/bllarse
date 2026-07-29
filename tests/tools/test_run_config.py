from types import SimpleNamespace

import pytest

import bllarse.tools.run_config as run_config_module


def test_run_config_selects_requested_config(monkeypatch):
    seen = []
    sweep = SimpleNamespace(
        create_configs=lambda: [{"seed": 0}, {"seed": 1}],
        run=seen.append,
    )
    monkeypatch.setattr(
        run_config_module,
        "get_module_from_source_path",
        lambda _: sweep,
    )

    run_config_module.run_config("unused.py", 1)

    assert seen == [{"seed": 1}]


def test_run_config_rejects_invalid_index(monkeypatch):
    sweep = SimpleNamespace(create_configs=lambda: [{"seed": 0}], run=lambda _: None)
    monkeypatch.setattr(
        run_config_module,
        "get_module_from_source_path",
        lambda _: sweep,
    )

    with pytest.raises(IndexError, match="out of range"):
        run_config_module.run_config("unused.py", 1)
