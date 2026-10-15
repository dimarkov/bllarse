import os
from argparse import Namespace
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from bllarse.tools.module import get_module_from_source_path

# Config keys whose False value must be passed explicitly as --no-<flag> rather
# than omitted, because the target script's default is True.
NEGATABLE_FLAGS = frozenset({"return_ood"})


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
            elif key in NEGATABLE_FLAGS:
                argv.append("--no-" + key.replace("_", "-"))
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


def resolve_script_path(script_path: str | Path) -> Path:
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


def load_script_module(script_path: str | Path) -> Any:
    """Import a training script as a module, resolving repo-relative paths.

    Public because the tuning objective needs `main(args, on_dump=...)`, which
    run_script_from_config cannot express -- and duplicating the resolution
    would duplicate its knowledge of BLLARSE_REPO_ROOT and the scripts/ prefix.
    """
    return get_module_from_source_path(resolve_script_path(script_path))


def parse_config(module: Any, config: Mapping[str, Any]) -> Namespace:
    """Parse a config mapping with a script's own argparser."""
    return module.build_argparser().parse_args(config_to_argv(config))


def run_script_from_config(
    script_path: str | Path,
    config: Mapping[str, Any],
) -> None:
    """Parse a config with a script's CLI and invoke its main function."""
    module = load_script_module(script_path)
    args = parse_config(module, config)

    train_main = module.main
    if not hasattr(module, "build_configs"):
        train_main(args)
        return

    model_config, optimizer_config = module.build_configs(args)
    train_main(args, model_config, optimizer_config)
