"""Fig 2: AdamW+CE OOD detection from frozen features.

Pairs: CIFAR-10 vs SVHN, CIFAR-100 vs SVHN (far-OOD), CIFAR-10 vs CIFAR-100
(near-OOD -- the harder and more informative pair). CIFAR-100 is ID in one pair
and OOD in another. All at res224, so ID and OOD features share a space.

Hyperparameters are per-ID-dataset, taken from the best trial of that dataset's
tuning study rather than shared defaults: one BASE would put at least one of
the two ID sets off its own optimum and the comparison against the other arms
would measure the defaults.
"""

import os
from itertools import product

from bllarse.evaldump import campaign_root
from bllarse.tools.adapters import run_script_from_config

GROUP_ID = "probe_adamw_ce_fig2"

# The tuning campaign these TUNED values came from, and therefore the
# subdirectory the dumps go in. Declared beside the numbers it describes
# because the two MUST move together: re-tuning changes TUNED, and a re-run
# under the previous campaign's root would overwrite the earlier campaign's
# dumps -- same group_id, same uid, different hyperparameters, no error. Flip
# this in the same edit that replaces TUNED. See evaldump.campaign_root.
CAMPAIGN = "win2"
DUMP_ROOT = campaign_root(CAMPAIGN)

# Only dinov3_huge, because it is the only TUNED backbone. Fig 1 transplants
# huge's config across the ladder (see space.TUNING_MODEL) because the backbone
# axis IS its point. Fig 2 has no backbone axis, so there is nothing to buy here
# by paying 4x the compute to run three untuned rungs.
MODELS = ["dinov3_huge"]

# (id_dataset, num_classes, ood_dataset)
PAIRS = [
    ("cifar10", 10, "svhn"),
    ("cifar10", 10, "cifar100"),
    ("cifar100", 100, "svhn"),
]
def _seeds():
    """The reported seeds, overridable for a staged rollout.

    `BLLARSE_SWEEP_SEEDS="137 138"` runs a two-seed slice first; a later
    submission with the remaining eight ADDS to the same group_id, because each
    run's uid carries its seed and nothing keys on the seed count.

    Unset reproduces all ten -- the value the figures are specified on.
    """
    raw = os.environ.get("BLLARSE_SWEEP_SEEDS", "").split()
    return [int(s) for s in raw] if raw else list(range(137, 147))


SEEDS = _seeds()

# Best trial of the adamw_ce_<dataset>_<CAMPAIGN> study, by the windowed val_nll
# objective (see tuning.objective). `epochs` carries the campaign's fixed
# 30,000-step budget at that trial's batch size; it is not a free knob.
TUNED = {
    "cifar10": dict(batch_size=512, learning_rate=0.014570293902991813, weight_decay=0.07198300389818668, epochs=310),  # win2, 79 trials, 30070 steps
    "cifar100": dict(batch_size=512, learning_rate=0.0016688334907619896, weight_decay=0.00013159281599845325, epochs=310),  # win2, 101 trials, 30070 steps
}

BASE = dict(
    dump_root=DUMP_ROOT,
    enable_mlflow=True,
    mlflow_experiment="bllarse-probing",
    arm="adamw_ce",
    report_every=10,
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
