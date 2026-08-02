"""Fig 1: VBLL linear probing across the DINOv3 ladder and six datasets.

Reads the same cached feature files as the AdamW arm, so the backbone is
byte-identical between them.
"""

from itertools import product

from bllarse.tools.adapters import run_script_from_config

GROUP_ID = "probe_vbll_fig1"

MODELS = ["dinov3_small", "dinov3_big", "dinov3_large", "dinov3_huge"]
DATASETS = {
    "oxford_pets": (37, 512),
    "food101": (101, 512),
    "flowers102": (102, 512),
    "stanford_cars": (196, 512),
    "dtd": (47, 512),
}
SEEDS = list(range(137, 147))

BASE = dict(
    enable_mlflow=True,
    mlflow_experiment="bllarse-probing",
    epochs=40,
    n_dumps=40,
    batch_size=512,
    learning_rate=1e-3,
    weight_decay=1e-4,
    val_fraction=0.1,
    prior_scale=1.0,
    wishart_scale=0.1,
    parameterization="diagonal",
    vbll_type="discriminative",
    device="cuda",
)


def create_configs():
    configs = []
    for model, (dataset, (num_classes, img_size)), seed in product(
        MODELS, DATASETS.items(), SEEDS
    ):
        configs.append(dict(
            **BASE,
            model=model,
            dataset=dataset,
            num_classes=num_classes,
            img_size=img_size,
            seed=seed,
            group_id=GROUP_ID,
            uid=f"{model}_{dataset}_seed{seed}",
        ))
    return configs


def run(config):
    run_script_from_config("scripts/vbll_pytorch/probing_vbll.py", config)
