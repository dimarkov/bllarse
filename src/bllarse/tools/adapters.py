import os
from argparse import Namespace
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from bllarse.tools.module import get_module_from_source_path


def config_to_argv(config: Mapping[str, Any]) -> list[str]:
    """Convert a config mapping to conventional argparse flags."""
    argv: list[str] = []
    for key, value in config.items():
        if value is None:
            continue

        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue

        argv.extend((flag, str(value)))
    return argv


def _resolve_repo_root(script_path: str | Path) -> Path:
    env_root = os.environ.get("BLLARSE_REPO_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    relative_script = Path(script_path)
    for parent in Path(__file__).resolve().parents:
        if (parent / relative_script).is_file():
            return parent
        if (parent / "scripts" / relative_script).is_file():
            return parent

    return Path.cwd().resolve()


def _resolve_script_path(script_path: str | Path) -> Path:
    requested = Path(script_path).expanduser()
    if requested.is_absolute():
        resolved = requested.resolve()
    else:
        repo_root = _resolve_repo_root(requested)
        direct = repo_root / requested
        resolved = direct if direct.is_file() else repo_root / "scripts" / requested

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Could not find training script {resolved}. "
            "Set BLLARSE_REPO_ROOT when launching outside the repository."
        )
    return resolved


def run_training_from_config(config: Mapping[str, Any]) -> None:
    """Run the repository's standard finetuning script from a config."""
    run_script_from_config("scripts/finetuning.py", config)


def run_script_from_config(
    script_path: str | Path,
    config: Mapping[str, Any],
) -> None:
    """Parse a config with a script's CLI and invoke its main function."""
    module = get_module_from_source_path(_resolve_script_path(script_path))
    build_argparser: Callable[[], Any] = module.build_argparser
    parser = build_argparser()
    args: Namespace = parser.parse_args(config_to_argv(config))

    train_main = module.main
    if not hasattr(module, "build_configs"):
        train_main(args)
        return

    model_config, optimizer_config = module.build_configs(args)
    train_main(args, model_config, optimizer_config)
