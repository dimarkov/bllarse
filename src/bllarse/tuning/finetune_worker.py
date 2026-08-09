"""An Optuna TPE worker for one fine-tuning arm's study.

Sibling of `worker.py`. The probing campaign's settings do not transfer -- the
trial is ~6 hours here rather than ~40 seconds -- so seeds, pruner warmup and
patience are all set independently; see the constants below.

One study per ARM, not per (arm, dataset): the fine-tuning campaign is
ImageNet-1k only.
"""

import argparse
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from bllarse.finetune import protocol
from bllarse.tuning import finetune_space as space
from bllarse.tuning import objective as objective_mod

DIRECTION = "minimize"

SCRIPTS = {
    "ce": "scripts/finetune/finetune_jax.py",
    # Same script as ce: the JAX arms share the data stream, the pinned split
    # and the evaluation grid, and differ only in the training step. `--arm`
    # selects between them.
    "ibprobit": "scripts/finetune/finetune_jax.py",
    # The contamination ablation: same head, same script, eps_init pinned at
    # space.IBPROBIT_NOEPS_INIT instead of searched.
    "ibprobit_noeps": "scripts/finetune/finetune_jax.py",
    "vbll": "scripts/finetune/finetune_vbll.py",
}

# 5, not probing's 20. Twenty was a fifth of ~80 probing trials; at ~30 trials
# it would barely engage, and pruning is where the saving is at ~6 h a trial.
PRUNER_STARTUP_TRIALS = 5
PRUNER_WARMUP_EVALS = 10

# Consecutive evaluations with no 0.1% relative improvement in val_nll.
#
# 20 evaluations = 327,680 samples (0.256 epochs); with WARMUP_EVALS the
# earliest possible stop is eval 30, so a doomed config costs ~1.5 h instead of
# ~6.3 h. What it is aimed at is trunk learning rates high enough to cause
# catastrophic forgetting, which plateau or diverge well inside the first epoch.
#
# This is NOT redundant with MedianPruner and the two are not interchangeable:
# the pruner kills trials that are LOSING against their cohort, patience stops
# trials that have FINISHED CONVERGING -- whose score is a perfectly good
# result, which is why EarlyConvergence is deliberately not TrialPruned.
#
# The probing campaign's `--patience 20` was structurally unable to fire (it
# needed the argmin at dump <= 19 on a bounded 40-point geometric grid). This
# one runs on 122 uniformly sample-spaced evaluations, so a count is a
# well-defined amount of training. Read `patience_stopped_at_eval` and
# `truncated` off the journal after the first worker batch: many stops at eval
# 30 means 20 is too tight, `truncated` everywhere means the 2M budget binds
# instead, and the two call for opposite corrections.
PATIENCE = 20
WARMUP_EVALS = 10

# Every trial evaluates val ONLY. The objective is val_nll, and forwarding the
# test partition at every evaluation would cost as much again for a number
# nothing reads.
#
# 0.2 over the official 50k validation split gives 10k val / 40k test -- the
# same split imagenet.DEFAULT_VAL_FRACTION uses. Halved from 0.4 (20k val)
# because evaluation was ~29% of a trial at the measured throughput: 61
# evaluations x 10k images is ~1.4 h against ~7 h of training.
#
# 10k is still a large selection set. Its standard error on val_nll is ~1.4x
# that of 20k, which matters only if trials are separated by less than that --
# and configs that close cannot be distinguished by a longer budget either.
VAL_FRACTION = 0.2
EVAL_SPLITS = "val"

# No trajectory dumps and no OOD during tuning: the run writes exactly one
# dump, its buffered argmin, and never pays the 47,500-image OOD preload.
DUMP_EVERY_K_EVALS = 0
OOD_EVERY_K_DUMPS = 0

# Where the pre-decoded uint8 training shard lives (bllarse.finetune.shards).
#
# Read from the environment because the path is a property of the cluster, not
# of the campaign -- 193 GB of ImageNet-1k train that no repo constant should
# name. `finetune_jax`'s argparser reads the same variable as its own default,
# so the shard would be picked up even if this were absent; it is carried
# through the CONFIG anyway so the study records which trials used it. An
# unset variable is not a silent wrong answer, it is a silent SLOW one -- the
# JPEG path measures ~40 img/s against the shard's ~220, which turns a 3 h
# trial into 14 h and a 24 h worker's 7 trials into 1. The SLURM entry point
# refuses to start without it for that reason.
TRAIN_SHARDS_ENV = "BLLARSE_FT_SHARDS"

VBLL_FIXED = {
    "parameterization": "diagonal",
    "vbll_type": "discriminative",
    "device": "cuda",
    # Pinned, not searched -- the one head knob whose probing optima agreed
    # across datasets. See space.WISHART_SCALE.
    "wishart_scale": space.WISHART_SCALE,
}

# px on for every IBProbit trial: PX shares the plain CAVI fixed point exactly
# and only reaches it in fewer iterations, so it is a solver form rather than a
# model hyperparameter -- searching it would let TPE trade compute against fit
# and make trials incomparable in cost.
#
# learn_q OFF: q0 is the seed the recursion starts from, so tuning q_init while
# the recursion also runs measures a moving target. Whether inferring Q beats
# the tuned static value is a follow-up ablation at that value.
IBPROBIT_FIXED = {
    "px": True,
    "no_learn_q": True,
}


def study_name(arm: str, campaign: str | None = None) -> str:
    """Study identity, optionally namespaced by campaign.

    `open_study` passes `load_if_exists=True`, so a name that collides with an
    earlier campaign's RESUMES it -- right for the workers of one campaign and
    wrong across campaigns, since TPE would model a search space that no longer
    exists.
    """
    base = f"ft_{arm}"
    return base if campaign is None else f"{base}_{campaign}"


def open_study(arm: str, storage_dir: Any, campaign: str | None = None) -> optuna.Study:
    """Load or create this (arm, campaign)'s study.

    JournalStorage on JournalFileBackend, not SQLite: SQLite over GPFS with
    concurrent writers is unreliable, and the journal backend is built for
    network filesystems.
    """
    directory = Path(storage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    name = study_name(arm, campaign)
    storage = JournalStorage(JournalFileBackend(str(directory / f"{name}.log")))
    return optuna.create_study(
        study_name=name,
        storage=storage,
        direction=DIRECTION,
        sampler=TPESampler(),
        pruner=MedianPruner(
            n_startup_trials=PRUNER_STARTUP_TRIALS,
            n_warmup_steps=PRUNER_WARMUP_EVALS,
        ),
        load_if_exists=True,
    )


def summarize_study(study: optuna.Study) -> str:
    """One line describing what this worker achieved.

    Raises SystemExit when NOTHING completed: `catch=(Exception,)` keeps the
    worker alive through a failing trial, but if EVERY trial fails then
    `study.best_value` raises and THAT traceback becomes the last thing in the
    SLURM log, burying the real cause. A study where every trial failed is a
    harness fault -- paths, container, missing weights -- not a search outcome.
    """
    states = [t.state for t in study.trials]
    n_complete = states.count(optuna.trial.TrialState.COMPLETE)
    if n_complete == 0:
        raise SystemExit(
            f"[ft-tune] {study.study_name}: {len(states)} trials, NONE "
            f"completed ({states.count(optuna.trial.TrialState.FAIL)} failed, "
            f"{states.count(optuna.trial.TrialState.PRUNED)} pruned). Read the "
            f"FIRST traceback in this log, not the last. Every trial failing "
            f"is a harness fault, not a search outcome."
        )
    return (f"[ft-tune] {study.study_name}: {len(states)} trials, "
            f"{n_complete} completed, best={study.best_value:.5f} "
            f"params={study.best_params}")


def trial_config(
    arm: str,
    *,
    params: Mapping[str, Any],
    max_samples: int = protocol.MAX_SAMPLES,
    dump_root: Any = None,
    bf16: bool = True,
    train_shards: Any = None,
) -> dict[str, Any]:
    """A full training config for one sampled parameter set.

    `train_shards` defaults to $BLLARSE_FT_SHARDS; pass `""` to force the JPEG
    path (which is what a machine without a built shard has to do).
    """
    if arm not in SCRIPTS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(SCRIPTS)}")

    if train_shards is None:
        train_shards = os.environ.get(TRAIN_SHARDS_ENV, "")

    params = dict(params)
    # head_lr is a MULTIPLE of the trunk rate, not an independent axis -- the
    # scripts take an absolute rate, so the product is formed here rather than
    # searched. See space.HEAD_LR_RATIO_BOX.
    head_lr_ratio = params.pop("head_lr_ratio", None)

    config: dict[str, Any] = {
        "model": space.TUNING_MODEL,
        "dataset": "imagenet1k",
        "img_size": 224,
        "max_samples": max_samples,
        "eval_quantum": protocol.EVAL_QUANTUM,
        "eval_splits": EVAL_SPLITS,
        "dump_every_k_evals": DUMP_EVERY_K_EVALS,
        "ood_every_k_dumps": OOD_EVERY_K_DUMPS,
        "val_fraction": VAL_FRACTION,
        "micro_batch_size": space.micro_batch(arm),
        "bf16": bf16,
        **params,
    }
    if dump_root is not None:
        config["dump_root"] = str(dump_root)
    if train_shards:
        config["train_shards"] = str(train_shards)

    if arm == "vbll":
        config.update(VBLL_FIXED)
        config["head_lr"] = params["lr"] * head_lr_ratio
    else:
        # The JAX arms run scripts/finetune/finetune_jax.py and select the
        # training step with --arm. The STUDY arm name is not always the flag
        # value: `ibprobit_noeps` is an ablation of the ibprobit head, not a
        # third training step, and finetune_jax's argparser accepts only
        # {ce, ibprobit} -- passing the study name through would abort every
        # trial at argparse.
        config["arm"] = "ibprobit" if arm.startswith("ibprobit") else arm
        if arm.startswith("ibprobit"):
            config.update(IBPROBIT_FIXED)
            # No head_lr on either ibprobit arm: check_arm_flags rejects it
            # outright, because the head has no optimizer to receive one.
            if arm == "ibprobit_noeps":
                # Pinned, not searched. space.suggest omits the eps axis for
                # this arm, so nothing in `params` overwrites this.
                config["eps_init"] = space.IBPROBIT_NOEPS_INIT
        else:
            config["head_lr"] = params["lr"] * head_lr_ratio
    return config


def make_objective(
    arm: str,
    *,
    dump_root: Any,
    group_id: str,
    max_samples: int = protocol.MAX_SAMPLES,
    patience: int | None = PATIENCE,
    warmup_evals: int = WARMUP_EVALS,
    keep_dumps: bool = True,
) -> Callable[[optuna.Trial], float]:
    """The optuna objective closure for one arm's study."""

    def run_trial(trial: optuna.Trial) -> float:
        params = space.suggest(trial, arm)
        config = trial_config(
            arm, params=params, max_samples=max_samples, dump_root=dump_root)
        trial.set_user_attr("max_samples", max_samples)
        trial.set_user_attr("eval_quantum", protocol.EVAL_QUANTUM)
        trial.set_user_attr("micro_batch_size", config["micro_batch_size"])
        trial.set_user_attr("effective_batch", config["batch_size"])
        # Recorded so a trial's wall-clock is interpretable after the fact: the
        # same config costs ~3 h off the shard and ~14 h off JPEGs, and nothing
        # else in the trial distinguishes the two.
        trial.set_user_attr("train_shards", config.get("train_shards", ""))

        # The objective IS this minimum -- see objective.evaluate's `score`
        # note. Tracked here rather than read back off the dump because the
        # run evaluates val only, so there is no test dump for run_curve to
        # read, and because the in-loop value is fp32 while a dump round-trips
        # through fp16.
        seen = {"total": 0, "argmin": 0, "best": math.inf}

        def on_dump(
            eval_index: int, step: int, samples_seen: int,
            scalars: Mapping[str, float],
        ) -> None:
            seen["total"] = eval_index
            value = float(scalars["val_nll"])
            # `isfinite`, not a bare `<`: a nan adopted as the best would make
            # every later comparison False and latch the argmin forever.
            if math.isfinite(value) and value < seen["best"]:
                seen["best"], seen["argmin"] = value, eval_index
            # Reported on the EVAL INDEX, never the optimizer step. Trials vary
            # batch_size, so raw steps would rarely collide and MedianPruner
            # would silently never have a cohort. The eval index is uniformly
            # sample-spaced, so matching keys also means matching samples_seen.
            trial.report(value, eval_index)
            if trial.should_prune():
                raise optuna.TrialPruned()

        stopped_at: dict[int, int] = {}

        def on_converged(seed: int, eval_index: int, step: int) -> None:
            stopped_at[seed] = eval_index

        value = objective_mod.evaluate(
            arm, config,
            dump_root=dump_root,
            group_id=group_id,
            uid=f"trial{trial.number:05d}",
            on_dump=on_dump,
            patience=patience,
            warmup_evals=warmup_evals,
            on_converged=on_converged,
            keep_dumps=keep_dumps,
            seeds=(space.TUNING_SEED,),
            scripts=SCRIPTS,
            score=lambda root, group, uids: seen["best"],
        )

        trial.set_user_attr("patience_stopped", bool(stopped_at))
        trial.set_user_attr(
            "patience_stopped_at_eval",
            {str(seed): index for seed, index in stopped_at.items()},
        )
        trial.set_user_attr("argmin_eval", seen["argmin"])
        trial.set_user_attr("total_evals", seen["total"])
        # A trial whose argmin IS its last evaluation never converged: it was
        # still improving when the budget ran out. Many of these means
        # MAX_SAMPLES is too small -- the opposite correction to a patience
        # that is firing too eagerly, which is why both are recorded.
        trial.set_user_attr(
            "truncated", seen["total"] > 0 and seen["argmin"] == seen["total"]
        )
        return value

    return run_trial


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=sorted(SCRIPTS))
    p.add_argument("--n-trials", type=int, required=True,
                   help="Trials THIS worker runs. Size it against the FULL "
                        "trial cost (~6.3 h), not the hoped-for average: a "
                        "24 h job fits 3.")
    p.add_argument("--storage-dir", required=True)
    p.add_argument("--dump-root",
                   default=os.environ.get("BLLARSE_EVAL_DUMP_ROOT", "eval_dumps"))
    p.add_argument("--max-samples", type=int, default=protocol.MAX_SAMPLES)
    p.add_argument(
        "--campaign", default=None,
        help="Namespace suffix for the study, its journal file and its dump "
             "directory. Required whenever the search space has changed, since "
             "open_study loads an existing study of the same name rather than "
             "creating a fresh one.",
    )
    p.add_argument(
        "--patience", type=int, default=PATIENCE,
        help="Consecutive evaluations with no relative val_nll improvement "
             "that stop a trial early. Evaluations are uniformly spaced in "
             "samples, so N of them is always the same amount of training.",
    )
    p.add_argument("--warmup-evals", type=int, default=WARMUP_EVALS)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    study = open_study(args.arm, args.storage_dir, args.campaign)
    group_id = f"fttune_{study_name(args.arm, args.campaign)}"

    def log_failed_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.FAIL:
            print(f"[ft-tune] WARNING: trial {trial.number} FAILED "
                  f"(arm={args.arm})", flush=True)

    study.optimize(
        make_objective(
            args.arm,
            dump_root=args.dump_root,
            group_id=group_id,
            max_samples=args.max_samples,
            patience=args.patience,
            warmup_evals=args.warmup_evals,
        ),
        n_trials=args.n_trials,
        # Scoped to `Exception`, deliberately NOT `BaseException`:
        # optuna.TrialPruned is itself an Exception subclass but is
        # special-cased by Optuna before this tuple is consulted, and
        # KeyboardInterrupt/SystemExit derive from BaseException so a job
        # timeout still tears the worker down instead of being swallowed.
        catch=(Exception,),
        callbacks=[log_failed_trial],
    )
    print(summarize_study(study))


if __name__ == "__main__":
    main()
