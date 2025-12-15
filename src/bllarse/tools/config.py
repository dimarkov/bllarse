import itertools
from typing import Callable, Dict, List, Any

def get_config_grid(create_config: Callable[..., Any], values_dict: Dict[str, list]) -> List[Any]:
    """Return [create_config(...)] over the cartesian product of values_dict."""
    keys = list(values_dict.keys())
    out = []
    for vals in itertools.product(*[values_dict[k] for k in keys]):
        kwargs = {k: v for k, v in zip(keys, vals)}
        out.append(create_config(**kwargs))
    return out
