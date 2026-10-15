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

flowers102 trains on the official 1,020-image split, NOT the 7,169-image pool
the upstream config calls "train": that pool is really val+test and serves as
the eval pool here. See bllarse.features._TRAIN_SPLIT.

imagenet1k is deliberately absent: it is a full-network finetuning dataset only
and is tuned separately for that (bllarse.tuning.finetune_worker).

batch_size and epochs are per dataset (see TUNED below), not sweep-wide, so
there is no meaningful steps/epoch column for the whole grid.
"""

import os
from itertools import product

from bllarse.evaldump import campaign_root
from bllarse.tools.adapters import run_script_from_config

GROUP_ID = "probe_adamw_ce_fig1"

# The tuning campaign these TUNED values came from, and therefore the
# subdirectory the dumps go in. Declared beside the numbers it describes
# because the two MUST move together: re-tuning changes TUNED, and a re-run
# under the previous campaign's root would overwrite the earlier campaign's
# dumps -- same group_id, same uid, different hyperparameters, no error. Flip
# this in the same edit that replaces TUNED. See evaldump.campaign_root.
CAMPAIGN = "win2"
DUMP_ROOT = campaign_root(CAMPAIGN)

MODELS = ["dinov3_small", "dinov3_big", "dinov3_large", "dinov3_huge"]
DATASETS = {
    "oxford_pets": (37, 512),
    "food101": (101, 512),
    "flowers102": (102, 512),
    "stanford_cars": (196, 512),
    "dtd": (47, 512),
}
def _seeds():
    """The reported seeds, overridable for a staged rollout.

    `BLLARSE_SWEEP_SEEDS="137 138"` runs a two-seed slice first; a later
    submission with the remaining eight ADDS to the same group_id, because each
    run's uid carries its seed and nothing keys on the seed count. So the
    sweep can be validated cheaply and then completed, rather than committed
    to in one block.

    Unset reproduces all ten -- the value the figures are specified on.
    """
    raw = os.environ.get("BLLARSE_SWEEP_SEEDS", "").split()
    return [int(s) for s in raw] if raw else list(range(137, 147))


SEEDS = _seeds()


# Best trial per dataset from the CAMPAIGN tuning studies, scored on the
# windowed val_nll objective (best mean over DEFAULT_SMOOTH_WINDOW consecutive
# evaluations, see tuning.objective) rather than the single-point minimum,
# which selects unstable high-lr configurations. Each entry's inline comment
# gives its campaign, the study's trial count and the realised step count.
#
# `epochs` restates the fixed 30,000-step budget AT THAT TRIAL'S BATCH SIZE.
# It is not an independent knob: batch and epochs must change together.
# Recomputed by budgets.epochs_for_steps (scripts/analysis/export_tuned.py
# writes this block) and checked by tests/test_sweep_budgets.py, never by hand.
#
# Tuned at dinov3_huge only and applied across all four rungs -- stated in
# the paper rather than buried; the non-huge rungs are not independently tuned.
TUNED = {
    "oxford_pets": dict(batch_size=64, learning_rate=0.003949689430975156, weight_decay=2.640766599613012e-05, epochs=527),  # win2, 102 trials, 30039 steps
    "food101": dict(batch_size=512, learning_rate=0.0011318003515322155, weight_decay=0.00022270768247029422, epochs=205),  # win2, 100 trials, 30135 steps
    "flowers102": dict(batch_size=256, learning_rate=0.021022436227844634, weight_decay=0.0016820134637137957, epochs=10000),  # win2, 103 trials, 30000 steps
    "stanford_cars": dict(batch_size=64, learning_rate=0.0010102482092427694, weight_decay=1.5982106955727333e-05, epochs=237),  # win2, 102 trials, 30099 steps
    "dtd": dict(batch_size=64, learning_rate=0.002165822230859914, weight_decay=0.0010307788585123925, epochs=518),  # win2, 100 trials, 30044 steps
}

BASE = dict(
    dump_root=DUMP_ROOT,
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
