"""Fig 1: VBLL linear probing across the DINOv3 ladder and five datasets.

Reads the same cached feature files as the AdamW arm, so the backbone is
byte-identical between them. Five datasets: imagenet1k is a full-network
finetuning dataset only and is absent from bllarse.tuning.space entirely, so it
has no tuned probing configuration for any arm.
"""

import os
from itertools import product

from bllarse.evaldump import campaign_root
from bllarse.tools.adapters import run_script_from_config

GROUP_ID = "probe_vbll_fig1"

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
    "oxford_pets": dict(batch_size=128, learning_rate=0.01586914808068211, weight_decay=7.610736500372484e-05, kl_ratio=2.3642297454643804e-07, prior_scale=1.4542579046177186, wishart_scale=0.5463178291117708, epochs=1072),  # win2, 101 trials, 30016 steps
    "food101": dict(batch_size=128, learning_rate=0.0012699478434900565, weight_decay=0.0005661671880335932, kl_ratio=1.1458983575649299e-06, prior_scale=0.10332805599040151, wishart_scale=0.005553463368393326, epochs=51),  # win2, 100 trials, 30141 steps
    "flowers102": dict(batch_size=64, learning_rate=0.0295296765376387, weight_decay=0.00597094638409956, kl_ratio=3.805487024635167e-07, prior_scale=11.132284683837032, wishart_scale=0.37082749281945177, epochs=2000),  # win2, 102 trials, 30000 steps
    "stanford_cars": dict(batch_size=64, learning_rate=0.003399168168667544, weight_decay=0.00012578257430452753, kl_ratio=1.4178776037314004e-07, prior_scale=12.630885774154905, wishart_scale=0.020098946638004454, epochs=237),  # win2, 100 trials, 30099 steps
    "dtd": dict(batch_size=64, learning_rate=0.007203249368895867, weight_decay=0.037689415896219505, kl_ratio=1.665571611791085e-05, prior_scale=16.494654519963444, wishart_scale=0.04633612033189494, epochs=518),  # win2, 103 trials, 30044 steps
}

BASE = dict(
    dump_root=DUMP_ROOT,
    enable_mlflow=True,
    mlflow_experiment="bllarse-probing",
    report_every=10,
    val_fraction=0.1,
    parameterization="diagonal",
    vbll_type="discriminative",
    predictive_samples=200,
    device="cuda",
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
    run_script_from_config("scripts/vbll_pytorch/probing_vbll.py", config)
