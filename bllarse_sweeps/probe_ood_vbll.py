"""Fig 2: VBLL OOD detection from frozen features.

Same three pairs as the AdamW arm, on the same cached res224 features.
"""

from itertools import product

from bllarse.tools.adapters import run_script_from_config

GROUP_ID = "probe_vbll_fig2"

MODELS = ["dinov3_small", "dinov3_big", "dinov3_large", "dinov3_huge"]
# (id_dataset, num_classes, ood_dataset)
PAIRS = [
    ("cifar10", 10, "svhn"),
    ("cifar10", 10, "cifar100"),
    ("cifar100", 100, "svhn"),
]
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
    img_size=224,
    prior_scale=1.0,
    wishart_scale=0.1,
    parameterization="diagonal",
    vbll_type="discriminative",
    device="cuda",
)


def create_configs():
    configs = []
    for model, (dataset, num_classes, ood_dataset), seed in product(MODELS, PAIRS, SEEDS):
        configs.append(dict(
            **BASE,
            model=model,
            dataset=dataset,
            num_classes=num_classes,
            ood_dataset=ood_dataset,
            seed=seed,
            group_id=GROUP_ID,
            uid=f"{model}_{dataset}_vs_{ood_dataset}_seed{seed}",
        ))
    return configs


def run(config):
    run_script_from_config("scripts/vbll_pytorch/probing_vbll.py", config)
