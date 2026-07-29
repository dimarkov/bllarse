from bllarse.tools.config import get_config_grid


def test_get_config_grid_builds_cartesian_product_in_key_order():
    configs = get_config_grid(
        lambda seed, batch_size: (seed, batch_size),
        {
            "seed": [0, 1],
            "batch_size": [64, 128],
        },
    )

    assert configs == [(0, 64), (0, 128), (1, 64), (1, 128)]
