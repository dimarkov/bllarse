"""Fig 1: IBProbit (eps pinned) linear probing across the DINOv3 ladder and five datasets.

eps_init is PINNED at 1e-12 rather than tuned, so the contamination-mixture
floor is numerically absent and the head is scored on its own predictive
alone. NOT 0.0 -- `IBProbit._eps_seed` falls through its own guard at zero to
b0 = 100, a Beta(1, 100) whose mean is 0.0099, and which drifts to 0.16 within
32 updates. See tuning.space.IBPROBIT_NOEPS_INIT.

IBProbit takes NO learning_rate and NO weight_decay: the head is assimilated in
closed form by `IBProbit.update` and linear probing freezes the backbone, so
this arm runs no optimizer at all. `px` and `no_learn_q` mirror
tuning.worker.IBPROBIT_FIXED exactly -- these runs must sit at the same
operating point the search scored, or the tuned values do not describe them.

`arm="ibprobit"`, not the tuning arm name `ibprobit_noeps`: probe.py's
argparser accepts only {adamw_ce, ibprobit}, and `ibprobit_noeps` is the same
head with eps pinned rather than a third training step. It is identified by
GROUP_ID and eps_init, not by --arm.

Tuned at dinov3_huge only and applied across all four rungs, matching the other
Fig 1 sweeps. The backbones emit 384/768/1024/1280-dim features, so a config
optimal at one is not guaranteed optimal at another; this is accepted because
the paper's arguments centre on huge, and the paper states it.

`epochs` expresses the campaign's fixed 30,000-step budget at that trial's batch
size -- a restatement of the budget, not an independent knob, so epochs and
batch_size must change together or not at all.
"""

import os
from itertools import product

from bllarse.evaldump import campaign_root
from bllarse.tools.adapters import run_script_from_config

GROUP_ID = "probe_ibprobit_noeps_fig1"

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
    "oxford_pets": dict(batch_size=1024, q_init=0.001388032637079427, num_update_iters=192, epochs=10000),  # win2, 73 trials (30 diverged), 30000 steps
    "food101": dict(batch_size=1024, q_init=1.932378111864763e-06, num_update_iters=64, epochs=411),  # win2, 103 trials, 30003 steps
    "flowers102": dict(batch_size=256, q_init=9.515865087582528e-05, num_update_iters=32, epochs=10000),  # win2, 103 trials, 30000 steps
    "stanford_cars": dict(batch_size=256, q_init=2.5778298869708678e-05, num_update_iters=192, epochs=968),  # win2, 86 trials, 30008 steps
    "dtd": dict(batch_size=64, q_init=0.0010137220102958532, num_update_iters=192, epochs=518),  # win2, 102 trials, 30044 steps
}

BASE = dict(
    dump_root=DUMP_ROOT,
    enable_mlflow=True,
    mlflow_experiment="bllarse-probing",
    arm="ibprobit",
    px=True,
    no_learn_q=True,
    eps_init=1e-12,
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
