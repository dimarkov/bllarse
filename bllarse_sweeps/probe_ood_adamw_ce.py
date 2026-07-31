"""Fig 2: AdamW+CE OOD detection from frozen features.

Pairs: CIFAR-10 vs SVHN, CIFAR-100 vs SVHN (far-OOD), CIFAR-10 vs CIFAR-100
(near-OOD -- the harder and more informative pair). CIFAR-100 is ID in one pair
and OOD in another. All at res224, so ID and OOD features share a space.
"""

from itertools import product

from bllarse.tools.adapters import run_script_from_config

GROUP_ID = "probe_adamw_ce_fig2"

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
    arm="adamw_ce",
    epochs=40,
    n_dumps=40,
    batch_size=512,
    learning_rate=1e-3,
    weight_decay=1e-4,
    val_fraction=0.1,
    img_size=224,
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
    run_script_from_config("scripts/alternatives/probe.py", config)
