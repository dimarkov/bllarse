"""Fig 1: AdamW+CE linear probing across the DINOv3 ladder and five datasets.

AdamW+CE+TS is NOT a separate sweep -- temperature scaling is fit post-hoc from
these dumps at every dump point, so it costs no GPU time.

Measured training-set sizes (cifar10/cifar100 belong to the Fig 2 OOD sweeps,
not this grid, and are listed only for the size spread the project spans):

    | dataset             | n_train    |
    |---------------------|-----------:|
    | oxford_pets         |      3,680 |
    | dtd                 |      3,760 |
    | flowers102          |      7,169 |
    | stanford_cars       |      8,144 |
    | cifar10 / cifar100  |     50,000 |
    | food101             |     75,750 |

imagenet1k is deliberately absent: it is a full-network finetuning dataset only
and will be retuned separately for that.

The steps/epoch and total-step columns this docstring used to carry are gone
because batch_size and epochs are no longer sweep-wide constants -- both are
resolved per dataset from bllarse_sweeps/tuned_hparams.json.
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
