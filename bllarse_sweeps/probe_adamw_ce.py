"""Fig 1: AdamW+CE linear probing across the DINOv3 ladder and five datasets.

AdamW+CE+TS is NOT a separate sweep -- temperature scaling is fit post-hoc from
these dumps at every dump point, so it costs no GPU time.

Measured training-set sizes (cifar10/cifar100 belong to the Fig 2 OOD sweeps,
not this grid, and are listed only for the size spread the project spans):

    | dataset             | n_train    |
    |---------------------|-----------:|
    | flowers102          |      1,020 |
    | oxford_pets         |      3,680 |
    | dtd                 |      3,760 |
    | stanford_cars       |      8,144 |
    | cifar10 / cifar100  |     50,000 |
    | food101             |     75,750 |

flowers102 is 1,020, NOT the 7,169 this table used to say: that figure was the
upstream config's "train" split, which is really val+test and is now the eval
pool. See bllarse.features._TRAIN_SPLIT.

imagenet1k is deliberately absent: it is a full-network finetuning dataset only
and will be retuned separately for that.

batch_size and epochs are per dataset, not sweep-wide, so there is no meaningful
steps/epoch column for the whole grid. (An earlier version of this docstring
promised they were read from bllarse_sweeps/tuned_hparams.json; that file was
never written, and the values live in TUNED below. Fig 1's datasets and Fig 2's
are disjoint, so nothing is duplicated by holding them here.)
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

# adamw_ce_<dataset>_samples40, best trial by val_nll. `epochs` expresses the
# campaign's fixed 30,000-step budget at that trial's batch size -- it is a
# restatement of the budget, not an independent knob, so epochs and batch_size
# must be changed together or not at all.
#
# Tuned at dinov3_huge only and applied across all four rungs. space.py accepts
# this deliberately (the backbones emit 384/768/1024/1280-dim features, so a
# config optimal at one is not guaranteed optimal at another) on the grounds
# that the paper's arguments centre on huge. It has to be stated in the paper
# rather than buried -- the non-huge rungs are not independently tuned.
TUNED = {
    "oxford_pets": dict(
        batch_size=64, learning_rate=0.018152101741656677,
        weight_decay=0.008278246675974297, epochs=527
    ),
    "food101": dict(
        batch_size=512, learning_rate=0.002826281219835646,
        weight_decay=7.725356544062722e-05, epochs=205
    ),
    "flowers102": dict(
        batch_size=256, learning_rate=0.019652435709019516,
        weight_decay=1.0092039298773214e-05, epochs=10000
    ),
    "stanford_cars": dict(
        batch_size=64, learning_rate=0.0020402667218283985,
        weight_decay=1.5563704047949982e-05, epochs=237
    ),
    "dtd": dict(
        batch_size=64, learning_rate=0.0035828430208080723,
        weight_decay=1.8311539230944342e-05, epochs=518
    ),
}

BASE = dict(
    enable_mlflow=True,
    mlflow_experiment="bllarse-probing",
    arm="adamw_ce",
    report_every=10,
    val_fraction=0.1,
)


def create_configs():
    configs = []
    for model, (dataset, (num_classes, img_size)), seed in product(
        MODELS, DATASETS.items(), SEEDS
    ):
        configs.append(dict(
            **BASE,
            **TUNED[dataset],
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
    run_script_from_config("scripts/dinov3/probe.py", config)
