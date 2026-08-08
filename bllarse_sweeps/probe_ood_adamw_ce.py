"""Fig 2: AdamW+CE OOD detection from frozen features.

Pairs: CIFAR-10 vs SVHN, CIFAR-100 vs SVHN (far-OOD), CIFAR-10 vs CIFAR-100
(near-OOD -- the harder and more informative pair). CIFAR-100 is ID in one pair
and OOD in another. All at res224, so ID and OOD features share a space.

Hyperparameters are per-ID-dataset, taken from the val_nll-optimal trial of the
samples40 tuning campaign rather than shared defaults: batch size alone spans
128-1024 between the two ID sets, so one BASE would put at least one of them far
off its own optimum and the comparison against VBLL would measure the defaults.
"""

from itertools import product

from bllarse.tools.adapters import run_script_from_config

GROUP_ID = "probe_adamw_ce_fig2"

# Only dinov3_huge, because it is the only TUNED backbone -- not for want of
# features: the full 4-backbone x 9-dataset cache exists on Raven. space.py
# accepts transplanting huge's config across the ladder for Fig 1, where the
# backbone axis IS the point. Fig 2 has no backbone axis, so there is nothing to
# buy here by paying 4x the compute to run three untuned rungs.
MODELS = ["dinov3_huge"]

# (id_dataset, num_classes, ood_dataset)
PAIRS = [
    ("cifar10", 10, "svhn"),
    ("cifar10", 10, "cifar100"),
    ("cifar100", 100, "svhn"),
]
SEEDS = list(range(137, 147))

# adamw_ce_<dataset>_samples40, best trial by val_nll. `epochs` carries the campaign's
# fixed 30,000-step budget at that trial's batch size; it is not a free knob.
TUNED = {
    "cifar10": dict(
        batch_size=128, learning_rate=0.0034581670852852956,
        weight_decay=0.09462300712138334, epochs=77
    ),
    "cifar100": dict(
        batch_size=256, learning_rate=0.0019153485721736003,
        weight_decay=0.0004941138658043405, epochs=154
    ),
}

BASE = dict(
    enable_mlflow=True,
    mlflow_experiment="bllarse-probing",
    arm="adamw_ce",
    n_dumps=40,
    val_fraction=0.1,
    img_size=224,
)


def create_configs():
    configs = []
    for model, (dataset, num_classes, ood_dataset), seed in product(MODELS, PAIRS, SEEDS):
        configs.append(dict(
            **BASE,
            **TUNED[dataset],
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
    run_script_from_config("scripts/dinov3/probe.py", config)
