"""Fig 1: IBProbit (eps pinned) linear probing across the DINOv3 ladder and five datasets.

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

Tuned at dinov3_huge only and applied across all four rungs, matching the other
Fig 1 sweeps. The backbones emit 384/768/1024/1280-dim features, so a config
optimal at one is not guaranteed optimal at another; this is accepted because
the paper's arguments centre on huge, and it has to be stated in the paper
rather than buried.

`epochs` expresses the campaign's fixed 30,000-step budget at that trial's batch
size -- a restatement of the budget, not an independent knob, so epochs and
batch_size must change together or not at all.
"""

from itertools import product

from bllarse.tools.adapters import run_script_from_config

GROUP_ID = "probe_ibprobit_noeps_fig1"

MODELS = ["dinov3_small", "dinov3_big", "dinov3_large", "dinov3_huge"]
DATASETS = {
    "oxford_pets": (37, 512),
    "food101": (101, 512),
    "flowers102": (102, 512),
    "stanford_cars": (196, 512),
    "dtd": (47, 512),
}
SEEDS = list(range(137, 147))  # 10 seeds

# ibprobit_noeps_<dataset>_samples40, best trial by val_nll.
TUNED = {
    "oxford_pets": dict(batch_size=1024, q_init=0.0011889176805104206, num_update_iters=192, epochs=10000),
    "food101": dict(batch_size=64, q_init=0.00016280090390799026, num_update_iters=32, epochs=26),
    "flowers102": dict(batch_size=256, q_init=7.304878668598381e-05, num_update_iters=64, epochs=10000),
    "stanford_cars": dict(batch_size=256, q_init=2.4074162260449082e-05, num_update_iters=32, epochs=968),
    "dtd": dict(batch_size=128, q_init=0.0028571519043595144, num_update_iters=192, epochs=1035),
}

BASE = dict(
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
