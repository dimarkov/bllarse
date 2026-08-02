"""Stage 1: find a per-dataset epoch budget before tuning starts.

Runs the learning-rate extremes of the tuning range at a generous FIXED STEP
budget. A fixed epoch budget would not do: food101 and dtd differ 21x in
steps/epoch, so one epoch count over-trains the small datasets by that factor.

Both arms were still improving at the final dump at 40 epochs, so the current
budget truncates. Tuning under truncation biases the search toward high
learning rates -- a fast config converges inside the budget while a slower,
possibly better one is scored at a point it had not reached yet.

Output is consumed by `python -m bllarse.tuning.budgets`, which reads the
argmin-val_nll step of every pilot run and sizes each dataset's budget to
cover it with margin.
"""

from itertools import product

from bllarse.tools.adapters import run_script_from_config
from bllarse.tuning import space

GROUP_ID = "probe_pilot_epochs"

# 4,000 steps at the pilot batch size. Generous by construction: the current
# 40-epoch food101 run is 5,880 steps at batch 512 and the argmin sits at the
# last one, but the small datasets are 280 steps under the same setting -- so
# the pilot's job is to be long enough that the argmin lands strictly inside it
# for every dataset. budgets.py reports it loudly if that fails.
PILOT_STEPS = 4_000
PILOT_BATCH = 512
PILOT_SEED = 200  # a tuning seed, never a reported one

LEARNING_RATES = [1e-4, 3e-2]  # the extremes of the tuning range

SCRIPTS = {
    "adamw_ce": "scripts/alternatives/probe.py",
    "vbll": "scripts/vbll_pytorch/probing_vbll.py",
}

BASE = dict(
    enable_mlflow=True,
    mlflow_experiment="bllarse-probing-tuning",
    n_dumps=40,
    batch_size=PILOT_BATCH,
    weight_decay=1e-4,
    val_fraction=0.1,
    model=space.TUNING_MODEL,
    seed=PILOT_SEED,
    group_id=GROUP_ID,
)

VBLL_EXTRA = dict(
    prior_scale=1.0,
    wishart_scale=0.1,
    parameterization="diagonal",
    vbll_type="discriminative",
    device="cuda",
)


def pilot_epochs(dataset: str) -> int:
    """Epochs that reach PILOT_STEPS at PILOT_BATCH, rounded up."""
    per_epoch = space.steps_per_epoch(dataset, PILOT_BATCH)
    return -(-PILOT_STEPS // per_epoch)


def create_configs():
    configs = []
    for arm, dataset, lr in product(SCRIPTS, sorted(space.DATASETS), LEARNING_RATES):
        num_classes, img_size = space.DATASETS[dataset]
        config = dict(
            **BASE,
            arm_script=arm,
            dataset=dataset,
            num_classes=num_classes,
            img_size=img_size,
            learning_rate=lr,
            epochs=pilot_epochs(dataset),
            uid=f"{arm}_{dataset}_lr{lr:g}",
        )
        if arm == "vbll":
            config.update(VBLL_EXTRA)
        else:
            config["arm"] = "adamw_ce"
        configs.append(config)
    return configs


def run(config):
    # `arm_script` selects the entry point and must not reach either argparser:
    # probe.py has --arm but probing_vbll.py has no such flag, so passing it
    # through would make every VBLL task die on an unrecognised argument.
    config = dict(config)
    script = SCRIPTS[config.pop("arm_script")]
    run_script_from_config(script, config)
