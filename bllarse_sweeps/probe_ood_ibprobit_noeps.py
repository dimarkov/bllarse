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
"""

from itertools import product

from bllarse.tools.adapters import run_script_from_config

GROUP_ID = "probe_ibprobit_noeps_fig2"

# Only dinov3_huge, because it is the only TUNED backbone. Fig 2 has no backbone
# axis, so there is nothing to buy by paying 4x the compute for untuned rungs.
MODELS = ["dinov3_huge"]

# (id_dataset, num_classes, ood_dataset)
PAIRS = [
    ("cifar10", 10, "svhn"),
    ("cifar10", 10, "cifar100"),
    ("cifar100", 100, "svhn"),
]
SEEDS = list(range(137, 147))

# ibprobit_noeps_<dataset>_samples40, best trial by val_nll.
TUNED = {
    "cifar10": dict(batch_size=64, q_init=0.004571160669381488, num_update_iters=128, epochs=39),
    "cifar100": dict(batch_size=256, q_init=0.0013706794216466608, num_update_iters=128, epochs=154),
}

BASE = dict(
    enable_mlflow=True,
    mlflow_experiment="bllarse-probing",
    arm="ibprobit",
    px=True,
    no_learn_q=True,
    eps_init=1e-12,
    n_dumps=40,
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
