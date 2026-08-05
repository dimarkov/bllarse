"""Fig 1: VBLL linear probing across the DINOv3 ladder and five datasets.

Reads the same cached feature files as the AdamW arm, so the backbone is
byte-identical between them. Five, not six: imagenet1k is a full-network
finetuning dataset only and is absent from bllarse.tuning.space entirely, so it
has no tuned configuration for either arm.
"""

from itertools import product

from bllarse.tools.adapters import run_script_from_config

GROUP_ID = "probe_vbll_fig1"

MODELS = ["dinov3_small", "dinov3_big", "dinov3_large", "dinov3_huge"]
DATASETS = {
    "oxford_pets": (37, 512),
    "food101": (101, 512),
    "flowers102": (102, 512),
    "stanford_cars": (196, 512),
    "dtd": (47, 512),
}
SEEDS = list(range(137, 147))

# tune_vbll_<dataset>_kl_ratio, best trial by val_nll. See the AdamW arm on why
# `epochs` is a restatement of the 30,000-step budget and on the single-backbone
# tuning caveat, both of which apply identically here.
#
# The head is parameterised on kl_ratio = eta / prior_scale because the shrinkage
# of the posterior mean goes as ||m||^2 * eta / (N * prior_scale): eta and
# prior_scale are degenerate in the mean, so only their ratio controls fit, and
# prior_scale then sets posterior width at fixed ratio. The optima are strongly
# dataset-dependent -- kl_ratio spans 2.3e-7 to 1.4e-4 across these five, and the
# implied eta spans 9.1e-8 to 8.4e-3, all far below the textbook ELBO's eta = 1.
TUNED = {
    "oxford_pets": dict(batch_size=1024, learning_rate=0.021259684564184235,
                        weight_decay=0.008158209386171941, epochs=10000,
                        kl_ratio=6.4735824277955436e-06,
                        prior_scale=0.4298669857164471,
                        wishart_scale=2.737879517972118),
    "food101": dict(batch_size=4096, learning_rate=0.010595072734930645,
                    weight_decay=0.07900144106396743, epochs=1667,
                    kl_ratio=2.2512894655707186e-07,
                    prior_scale=411.05835707428395,
                    wishart_scale=0.1590377899575667),
    "flowers102": dict(batch_size=256, learning_rate=0.018170132266055055,
                       weight_decay=0.005306021933757882, epochs=10000,
                       kl_ratio=5.43962896641966e-07,
                       prior_scale=0.16656790376525496,
                       wishart_scale=0.08255546902844318),
    "stanford_cars": dict(batch_size=1024, learning_rate=0.005651911780129438,
                          weight_decay=0.07777972642213599, epochs=4286,
                          kl_ratio=3.6319498187756327e-07,
                          prior_scale=54.24415107235277,
                          wishart_scale=0.18030325427989444),
    "dtd": dict(batch_size=128, learning_rate=0.010010089218628395,
                weight_decay=8.054055700615195e-05, epochs=1035,
                kl_ratio=0.00013507277261844218,
                prior_scale=62.32624360321586,
                wishart_scale=0.03342071539821254),
}

BASE = dict(
    enable_mlflow=True,
    mlflow_experiment="bllarse-probing",
    n_dumps=40,
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
