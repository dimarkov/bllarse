from argparse import Namespace
from pathlib import Path
from typing import Dict, Any, List, Callable, Tuple
import os

from bllarse.tools.module import get_module_from_source_path

def _dict_to_argv(d: Dict[str, Any]) -> List[str]:
    argv: List[str] = []
    for k, v in d.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv.extend([flag, str(v)])
    return argv

def _resolve_repo_root() -> Path:
    """
    Resolve the repo root. Priority:
      1) env var BLLARSE_REPO_ROOT, if set
      2) walk up from this file until a 'scripts/last_layer_finetuning.py' exists
      3) fallback to current working directory
    """
    env_root = os.environ.get("BLLARSE_REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()

    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        candidate = parent / "scripts" / "last_layer_finetuning.py"
        if candidate.exists():
            return parent

    return Path.cwd().resolve()

def _load_llf_module() -> Any:
    repo_root = _resolve_repo_root()
    llf_path = repo_root / "scripts" / "last_layer_finetuning.py"
    if not llf_path.exists():
        raise FileNotFoundError(f"Could not find {llf_path}. "
                                "Set BLLARSE_REPO_ROOT or check your repo layout.")
    return get_module_from_source_path(str(llf_path))

def run_training_from_config(config: Dict[str, Any]) -> None:
    """
    Adapter: Dict config -> argparse args -> build_configs -> main
    implemented by dynamically importing scripts/last_layer_finetuning.py.
    """
    llf = _load_llf_module()

    # fetch required callables from the training script
    build_argparser: Callable[[], Any] = getattr(llf, "build_argparser")
    build_configs: Callable[[Any], Tuple[dict, dict]] = getattr(llf, "build_configs")
    train_main: Callable[[Any, dict, dict], None] = getattr(llf, "main")

    parser = build_argparser()
    argv = _dict_to_argv(config)
    args: Namespace = parser.parse_args(argv)
    m_conf, o_conf = build_configs(args)
    train_main(args, m_conf, o_conf)
