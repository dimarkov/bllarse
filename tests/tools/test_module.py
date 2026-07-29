import pytest

from bllarse.tools.module import get_module_from_source_path


def test_get_module_from_source_path_loads_module(tmp_path):
    source = tmp_path / "example.py"
    source.write_text("VALUE = 42\n")

    module = get_module_from_source_path(source)

    assert module.VALUE == 42


def test_get_module_from_source_path_supports_dataclasses(tmp_path):
    source = tmp_path / "config.py"
    source.write_text(
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass\n"
        "class Config:\n"
        "    seed: int\n"
    )

    module = get_module_from_source_path(source)

    assert module.Config(seed=3).seed == 3


def test_get_module_from_source_path_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Python source file not found"):
        get_module_from_source_path(tmp_path / "missing.py")
