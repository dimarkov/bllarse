"""Fig 2: VBLL OOD detection from frozen features.

Same three pairs as the AdamW arm, on the same cached res224 features, so the
backbone and the ID/OOD splits are byte-identical between the two arms.

Two scores come out of each run. The primary Fig 2 comparison uses MSP and
entropy derived from the dumped logits, because those are defined for every arm;
VBLL's own predictive variance is dumped alongside as `score` and read by
`curves.run_native_detection`. Reporting only the native score would confound
model with score function; reporting only the shared one would never test the
epistemic-term claim. Both, labelled.
"""

import os
from itertools import product

from bllarse.evaldump import campaign_root
from bllarse.tools.adapters import run_script_from_config

GROUP_ID = "probe_vbll_fig2"

# The tuning campaign these TUNED values came from, and therefore the
# subdirectory the dumps go in. Declared beside the numbers it describes
# because the two MUST move together: re-tuning changes TUNED, and a re-run
# under the previous campaign's root would overwrite the earlier campaign's
# dumps -- same group_id, same uid, different hyperparameters, no error. Flip
# this in the same edit that replaces TUNED. See evaldump.campaign_root.
CAMPAIGN = "win2"
DUMP_ROOT = campaign_root(CAMPAIGN)

# Matches the AdamW arm: dinov3_huge only, because it is the only TUNED
# backbone. Features for all four rungs exist; see the AdamW arm's note.
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

# tune_vbll_<dataset>_kl_ratio, best trial by val_nll. `epochs` carries the
# campaign's fixed 30,000-step budget at that trial's batch size.
#
# kl_ratio = eta / prior_scale is what the head is actually parameterised on:
# the shrinkage of the posterior mean goes as ||m||^2 * eta / (N * prior_scale),
# so eta and prior_scale are degenerate in the mean and only their ratio
# controls fit. prior_scale then sets the posterior width at fixed ratio -- which
# is exactly the quantity Fig 2's epistemic score depends on, so it is NOT a
# nuisance parameter here even though it barely moves the Fig 1 metrics.
#
# NOTE cifar10's prior_scale sits at 881 against a search ceiling of 1e3 -- the
# one axis in the whole campaign that did not come back interior. cifar10 is ID
# in two of these three pairs, so if the native-score results look odd this is
# the first thing to re-tune under a wider box.
TUNED = {
    "cifar10": dict(batch_size=1024, learning_rate=0.014199224084588319, weight_decay=0.002541292245333371, kl_ratio=0.00395546379800891, prior_scale=0.1244772177142675, wishart_scale=0.37123517198062456, epochs=625),  # win2, 89 trials, 30000 steps
    "cifar100": dict(batch_size=512, learning_rate=0.0017891011901586562, weight_decay=0.07754943204905308, kl_ratio=9.854065829990097e-06, prior_scale=6.15774616459221, wishart_scale=6.941868478640677, epochs=310),  # win2, 102 trials, 30070 steps
}

BASE = dict(
    dump_root=DUMP_ROOT,
    enable_mlflow=True,
    mlflow_experiment="bllarse-probing",
    report_every=10,
    val_fraction=0.1,
    img_size=224,
    parameterization="diagonal",
    vbll_type="discriminative",
    predictive_samples=200,
    device="cuda",
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
    run_script_from_config("scripts/vbll_pytorch/probing_vbll.py", config)
