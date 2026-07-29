import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def get_module_from_source_path(source_path: str | Path) -> ModuleType:
    """Load a Python module from a source file."""
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Python source file not found: {source}")

    digest = hashlib.sha256(str(source).encode()).hexdigest()[:12]
    module_name = f"_bllarse_dynamic_{source.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {source}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module
