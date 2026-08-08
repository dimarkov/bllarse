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
    "oxford_pets": dict(
        batch_size=128, learning_rate=0.014582332608481552,
        weight_decay=0.0029003502496597238, kl_ratio=1.3067604587856238e-06,
        prior_scale=351.7862885242714, wishart_scale=0.12465977620739462,
        epochs=1072
    ),
    "food101": dict(
        batch_size=256, learning_rate=0.0017523056959308691,
        weight_decay=0.0006983980274893595, kl_ratio=1.9699169406162906e-06,
        prior_scale=1.9329868444045457, wishart_scale=0.13019640570783467,
        epochs=102
    ),
    "flowers102": dict(
        batch_size=64, learning_rate=0.027772152286249085,
        weight_decay=3.569490434699052e-05, kl_ratio=6.581564108994321e-07,
        prior_scale=0.14469407662365807, wishart_scale=0.09564180809225105,
        epochs=2000
    ),
    "stanford_cars": dict(
        batch_size=64, learning_rate=0.0015496634275213618,
        weight_decay=0.0003441784909185116, kl_ratio=2.363217816602551e-07,
        prior_scale=0.37977399310692955, wishart_scale=0.2037317330205334,
        epochs=237
    ),
    "dtd": dict(
        batch_size=64, learning_rate=0.011639615162990528,
        weight_decay=0.0010661667809165759, kl_ratio=7.110021785552001e-05,
        prior_scale=0.3666703835471811, wishart_scale=0.07576231746200533,
        epochs=518
    ),
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
