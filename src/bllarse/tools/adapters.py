from argparse import Namespace
from pathlib import Path
from typing import Dict, Any, List, Callable, Tuple
import os

from bllarse.tools.module import get_module_from_source_path

def _dict_to_argv(d: Dict[str, Any]) -> List[str]:
    argv: List[str] = []
    for k, v in d.items():
        flag = "--" + k.replace("_", "-")
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                argv.append(flag)
            elif k == "return_ood":
                argv.append("--no-return-ood")
        else:
            argv.extend([flag, str(v)])
    return argv

def _resolve_repo_root(script_name) -> Path:
    """
    Resolve the repo root. Priority:
      1) env var BLLARSE_REPO_ROOT, if set
      2) walk up from this file until a 'scripts/<script_name>' exists
      3) fallback to current working directory
    """
    env_root = os.environ.get("BLLARSE_REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()

    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        candidate = parent / "scripts" / script_name
        if candidate.exists():
            return parent

    return Path.cwd().resolve()

def _load_module(script_name) -> Any:
    repo_root = _resolve_repo_root(script_name)
    llf_path = repo_root / "scripts" / script_name
    if not llf_path.exists():
        raise FileNotFoundError(f"Could not find {llf_path}. "
                                "Set BLLARSE_REPO_ROOT or check your repo layout.")
    return get_module_from_source_path(str(llf_path))

def run_training_from_config(config: Dict[str, Any]) -> None:
    run_script_from_config("finetuning.py", config)

def run_vbll_training_from_config(config: Dict[str, Any]) -> None:
    run_script_from_config("vbll_pytorch/finetuning_vbll.py", config)

def run_script_from_config(script_name: str, config: Dict[str, Any]) -> None:
    """
    Adapter: Dict config -> argparse args -> main,
    implemented by importing a training script under `scripts/`.
    """

    finetuning_module = _load_module(script_name)

    # Fetch callables from the target script.
    build_argparser: Callable[[], Any] = getattr(finetuning_module, "build_argparser")
    parser = build_argparser()
    argv = _dict_to_argv(config)
    args: Namespace = parser.parse_args(argv)

    # JAX finetuning script expects (args, model_cfg, opt_cfg), while other
    # scripts (e.g. VBLL PyTorch) expect just (args).
    if hasattr(finetuning_module, "build_configs"):
        build_configs: Callable[[Any], Tuple[dict, dict]] = getattr(finetuning_module, "build_configs")
        train_main: Callable[[Any, dict, dict], None] = getattr(finetuning_module, "main")
        m_conf, o_conf = build_configs(args)
        train_main(args, m_conf, o_conf)
        return

    train_main: Callable[[Any], None] = getattr(finetuning_module, "main")
    train_main(args)
