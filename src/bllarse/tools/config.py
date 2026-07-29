import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def get_config_grid(
    create_config: Callable[..., Any],
    values: Mapping[str, Sequence[Any]],
) -> list[Any]:
    """Build configs over the Cartesian product of the supplied values."""
    keys = list(values)
    return [
        create_config(**dict(zip(keys, combination, strict=True)))
        for combination in itertools.product(*(values[key] for key in keys))
    ]
