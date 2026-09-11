"""Fig 2: IBProbit (eps pinned) OOD detection from frozen features.

Pairs: CIFAR-10 vs SVHN, CIFAR-100 vs SVHN (far-OOD), CIFAR-10 vs CIFAR-100
(near-OOD -- the harder and more informative pair). CIFAR-100 is ID in one pair
and OOD in another. All at res224, so ID and OOD features share a space.

The contamination ablation: eps_init is PINNED at 1e-12 rather than tuned, so
the mixture floor is numerically absent and the head is scored on its own
predictive alone. NOT 0.0 -- `IBProbit._eps_seed` falls through its own guard
at zero to b0 = 100, a Beta(1, 100) whose mean is 0.0099, and which drifts to
0.16 within 32 updates. See tuning.space.IBPROBIT_NOEPS_INIT.

IBProbit takes NO learning_rate and NO weight_decay: the head is assimilated in
closed form by `IBProbit.update` and linear probing freezes the backbone, so
this arm runs no optimizer at all. `px` and `no_learn_q` mirror
tuning.worker.IBPROBIT_FIXED exactly -- the Stage 4 runs must sit at the same
operating point the search scored, or the tuned values do not describe them.

`arm="ibprobit"` for BOTH IBProbit sweeps: probe.py's argparser accepts only
{adamw_ce, ibprobit}, and `ibprobit_noeps` is an ablation of that head rather
than a third training step. The two are separated by GROUP_ID, not by --arm.

Hyperparameters are per-ID-dataset rather than shared defaults: batch size alone
differs between the two ID sets, so one BASE would put at least one of them off
its own optimum and the cross-arm comparison would measure the defaults.

WHY THE CAMPAIGN WAS RE-TUNED AT w=2. The CIFAR-10 entry below sat on the
q_init box floor (1.0e-6) under win5, and that operating point is what cost
IBProbit its detection numbers between the samples40 figures and the win5 ones.
Four tuning studies of the same cell say the floor is a property of the
SMOOTHING WINDOW rather than of the data or of an unlucky draw:

    w=1 (samples40)  q_init 5.4e-3    rho(q_init, value) = -0.66
    w=2 (win2)       q_init 5.8e-3    rho(q_init, value) = -0.66
    w=5 (win5)       q_init 1.0e-6    rho(q_init, value) = +0.79
    w=5 (win5b)      q_init 1.0e-6    rho(q_init, value) = +0.94   <- fresh draw

The floor configurations are flat -- every window scores them ~0.0318 -- while
the q ~ 5e-3 ones carry a sharp early dip that a 2-point mean still sees
(0.0290) and a 5-point mean averages away (0.033). The window alone decides
which family wins, and w=2 does NOT reintroduce the noise-mining the smoothing
was added to stop: re-tuned at w=2, adamw_ce/oxford_pets lands at lr 0.00395
against win5's 0.00341, not samples40's 0.0182.

A 20-run control of this sweep's CIFAR-10 cell at the w=2 configuration
recovered samples40's detection within 1 sd on both OOD pairs (MSP FPR@95
0.0081 vs 0.0080 against SVHN, 0.0163 vs 0.0159 against CIFAR-100; win5 gave
0.0261 and 0.0199), with the ID metrics unchanged. Its numbers are in
scripts/results/fig2_w2check.json; the one-cell sweep module that produced them
was deleted once the full w=2 campaign superseded it.
"""

import os
from itertools import product

from bllarse.evaldump import campaign_root
from bllarse.tools.adapters import run_script_from_config

GROUP_ID = "probe_ibprobit_noeps_fig2"

# The tuning campaign these TUNED values came from, and therefore the
# subdirectory the dumps go in. Declared beside the numbers it describes
# because the two MUST move together: re-tuning changes TUNED, and a re-run
# under the previous campaign's root would overwrite the earlier campaign's
# dumps -- same group_id, same uid, different hyperparameters, no error. Flip
# this in the same edit that replaces TUNED. See evaldump.campaign_root.
CAMPAIGN = "win2"
DUMP_ROOT = campaign_root(CAMPAIGN)

# Only dinov3_huge, because it is the only TUNED backbone. Fig 2 has no backbone
# axis, so there is nothing to buy by paying 4x the compute for untuned rungs.
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

# ibprobit_noeps_<dataset>_samples40, best trial by val_nll.
TUNED = {
    "cifar10": dict(batch_size=64, q_init=0.005772726541613569, num_update_iters=128, epochs=39),  # win2, 102 trials, 30459 steps
    "cifar100": dict(batch_size=128, q_init=0.0005294035477354173, num_update_iters=32, epochs=77),  # win2, 102 trials, 30030 steps
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
