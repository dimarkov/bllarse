import importlib.util as iu
from pathlib import Path
from types import ModuleType

def get_module_from_source_path(source_path: str) -> ModuleType:
    source = Path(source_path)
    modname = source.stem
    spec = iu.spec_from_file_location(modname, source)
    module = iu.module_from_spec(spec)
    assert spec and spec.loader, f"Could not load module from {source_path}"
    spec.loader.exec_module(module)
    return module
