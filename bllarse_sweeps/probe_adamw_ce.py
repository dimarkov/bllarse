"""Fig 1: AdamW+CE linear probing across the DINOv3 ladder and six datasets.

AdamW+CE+TS is NOT a separate sweep -- temperature scaling is fit post-hoc from
these dumps at every dump point, so it costs no GPU time.

Measured training-set sizes and the resulting optimisation budgets at this
sweep's batch_size=512, epochs=40 (cifar10/cifar100 are not part of this
sweep's own grid -- they belong to the Fig 2 OOD sweeps -- but are included
here for the full picture of the dataset-size spread the project spans):

    | dataset             | n_train    | steps/epoch | total steps |
    |---------------------|-----------:|------------:|-------------:|
    | oxford_pets         |      3,680 |           7 |          280 |
    | dtd                 |      3,760 |           7 |          280 |
    | flowers102          |      7,169 |          14 |          560 |
    | stanford_cars       |      8,144 |          15 |          600 |
    | cifar10 / cifar100  |     50,000 |          97 |        3,880 |
    | food101             |     75,750 |         147 |        5,880 |
    | imagenet1k          |  1,281,167 |       2,502 |      100,080 |

The epoch budget (40) is uniform across datasets, but the resulting
gradient-step budget is not -- it spans roughly 280 to 100,080 steps, a 357x
range across the panels of a single figure -- and this is unresolved pending
a decision from the project owner.
"""

from itertools import product

from bllarse.tools.adapters import run_script_from_config

GROUP_ID = "probe_adamw_ce_fig1"

MODELS = ["dinov3_small", "dinov3_big", "dinov3_large", "dinov3_huge"]
DATASETS = {
    "oxford_pets": (37, 512),
    "food101": (101, 512),
    "flowers102": (102, 512),
    "stanford_cars": (196, 512),
    "dtd": (47, 512),
    "imagenet1k": (1000, 512),
}
SEEDS = list(range(137, 147))  # 10 seeds

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
    run_script_from_config("scripts/alternatives/probe.py", config)
