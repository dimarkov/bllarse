from bllarse.tools.adapters import config_to_argv, run_script_from_config


def test_config_to_argv_handles_values_and_boolean_flags():
    assert config_to_argv(
        {
            "batch_size": 64,
            "nodataaug": True,
            "disabled_flag": False,
            "optional": None,
        }
    ) == ["--batch-size", "64", "--nodataaug"]


def test_run_script_from_config_supports_single_argument_main(tmp_path, monkeypatch):
    output = tmp_path / "result.txt"
    script = tmp_path / "runner.py"
    script.write_text(
        "import argparse\n"
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "def build_argparser():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--value', type=int)\n"
        "    return parser\n"
        "\n"
        "def main(args):\n"
        "    Path(os.environ['ADAPTER_TEST_OUTPUT']).write_text(str(args.value))\n"
    )
    monkeypatch.setenv("ADAPTER_TEST_OUTPUT", str(output))

    run_script_from_config(script, {"value": 17})

    assert output.read_text() == "17"


def test_run_script_from_config_supports_build_configs(tmp_path, monkeypatch):
    output = tmp_path / "result.txt"
    script = tmp_path / "runner_with_configs.py"
    script.write_text(
        "import argparse\n"
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "def build_argparser():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--value', type=int)\n"
        "    return parser\n"
        "\n"
        "def build_configs(args):\n"
        "    return {'model': args.value}, {'optimizer': args.value + 1}\n"
        "\n"
        "def main(args, model_config, optimizer_config):\n"
        "    result = model_config['model'] + optimizer_config['optimizer']\n"
        "    Path(os.environ['ADAPTER_TEST_OUTPUT']).write_text(str(result))\n"
    )
    monkeypatch.setenv("ADAPTER_TEST_OUTPUT", str(output))

    run_script_from_config(script, {"value": 4})

    assert output.read_text() == "9"


def test_load_script_module_resolves_a_repo_relative_script():
    from bllarse.tools import adapters

    module = adapters.load_script_module("scripts/alternatives/probe.py")
    assert callable(module.main)
    assert callable(module.build_argparser)


def test_parse_config_produces_the_same_namespace_as_the_cli():
    from bllarse.tools import adapters

    module = adapters.load_script_module("scripts/alternatives/probe.py")
    args = adapters.parse_config(module, {
        "model": "dinov3_huge", "dataset": "food101", "num_classes": 101,
        "learning_rate": 5e-3,
    })
    assert args.model == "dinov3_huge"
    assert args.learning_rate == 5e-3
