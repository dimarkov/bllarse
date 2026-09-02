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
# Imported rather than reimplemented: both take a study and read its journal,
# and nothing in either is specific to the probing search space. The probing
# campaign is where the defect they fix was first diagnosed.
from bllarse.tuning.worker import n_completed, stop_at_completed

DIRECTION = "minimize"

# Abort the worker after this many CONSECUTIVE failed trials.
#
# The failure this bounds is a LAUNCH fault -- a bad HF_HOME, an unreadable
# shard, a package missing from the venv -- which makes every trial fail in
# seconds while nothing stops the worker from grinding through its whole
# --n-trials budget. The betas1m campaign did exactly that: 950 of 954 trials
# failed on ce, 935 of 946 on ibprobit_noeps and 960 of 960 on vbll, across 48
# array tasks and 410 GPU-hours over three days, because 18 workers raced to
# prepare the same uncached ImageNet-1k and tore down each other's `.incomplete`
# staging directory.
#
# `summarize_study` already DETECTED that and printed "Every trial failing is a
# harness fault, not a search outcome" on the first worker within minutes. The
# gap was that nothing ACTED on it. This does.
#
# 5, not 1: a single trial can fail transiently -- a node OOM, a filesystem
# hiccup -- and killing an array over one would be worse than the disease. Five
# in a row with nothing succeeding in between is not transient. A completed
# trial takes ~1.4 h and a launch fault takes seconds, so five of them cost
# minutes rather than days.
CONSECUTIVE_FAILURE_LIMIT = 5


def abort_on_consecutive_failures(
    limit: int = CONSECUTIVE_FAILURE_LIMIT,
) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    """Raise once `limit` trials have failed back to back.

    RAISES rather than calling `study.stop()`, and the difference is the whole
    point: stop() exits the worker 0, which SLURM records as success, so the
    next array task starts and repeats the fault. An exception propagates out
    of `study.optimize` and the task exits non-zero, so the failure is visible
    in `sacct` and the array does not quietly launder it.

    Not caught by optimize's `catch=(Exception,)`: that applies to the
    objective, inside `_run_trial`. Callbacks are invoked outside it, so this
    escapes. `test_the_abort_guard_escapes_optimizes_catch` pins that, since it
    is a property of Optuna's internals rather than of this function.

    The streak resets on any non-FAIL trial, PRUNED included. A pruned trial
    means the harness works and the pruner is doing its job.
    """
    state = {"streak": 0}

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.FAIL:
            state["streak"] += 1
        else:
            state["streak"] = 0
        if state["streak"] >= limit:
            raise RuntimeError(
                f"[ft-tune] ABORTING {study.study_name}: {state['streak']} "
                f"consecutive trials failed. This is a harness fault, not a "
                f"search outcome -- read the FIRST traceback in this log, not "
                f"the last. Common causes: HF_HOME pointing at a cache without "
                f"the prepared dataset, an unreadable training shard, or a "
                f"package missing from the venv."
            )

    return callback


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

# Consecutive evaluations with no relative improvement in val_nll (the
# threshold is objective.DEFAULT_MIN_RELATIVE_IMPROVEMENT, now 1e-4).
#
# 20 evaluations = 327,680 samples (0.256 epochs). The earliest possible stop is
# eval 21: stalls accumulate from evaluation 1 and WARMUP_EVALS gates only the
# FIRING, not the counting -- so a doomed config costs ~1 h instead of ~6.3 h.
# What it aims at is trunk learning rates high enough to cause catastrophic
# forgetting, which plateau or diverge well inside the first epoch.
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
# `truncated` off the journal after each worker batch: many stops at eval 21
# means 20 is too tight, `truncated` everywhere means the 2M budget binds
# instead, and the two call for opposite corrections.
#
# That reading has already been done once and changed a constant. On the first
# 67 completed trials patience fired on 57 of them, at a median evaluation 88
# of 122 with the argmin 20 earlier -- while trials patience did NOT stop put
# their argmin at 91-100% of the budget. The threshold, not this count, was
# what was too tight; see DEFAULT_MIN_RELATIVE_IMPROVEMENT.
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

# The tuning experiment. Mirrors the probing side's `bllarse-probing-tuning` /
# `bllarse-probing` split, so the four experiments read as two campaigns x two
# phases rather than four unrelated names.
#
# Fine-tuning logged nothing at all before this. Everything the campaign had
# ever produced lived only as Optuna journals on /ptmp -- scratch, with a
# retention policy -- and the consequence was concrete: a campaign burned 410
# GPU-hours failing 2,851 of 2,860 trials for three days, and the only way to
# see it was to ssh in and read slurm logs.
MLFLOW_EXPERIMENT = "bllarse-finetuning-tuning"

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

    model = params.pop("model", None) or space.TUNING_MODEL
    config: dict[str, Any] = {
        "model": model,
        "dataset": "imagenet1k",
        "img_size": 224,
        "max_samples": max_samples,
        "eval_quantum": protocol.EVAL_QUANTUM,
        "eval_splits": EVAL_SPLITS,
        # Every trial is tracked, PRUNED ones included -- they are the shape of
        # the search, not noise. What made the probing experiment unreadable was
        # not that pruned trials were logged but that child_run marked them
        # FAILED, indistinguishable from a harness fault; that is fixed at the
        # source (mlflow_utils.BY_DESIGN_EXCEPTIONS), so they now land as KILLED
        # with a `run_outcome` tag and filter cleanly.
        "enable_mlflow": True,
        "mlflow_experiment": MLFLOW_EXPERIMENT,
        "dump_every_k_evals": DUMP_EVERY_K_EVALS,
        "ood_every_k_dumps": OOD_EVERY_K_DUMPS,
        "val_fraction": VAL_FRACTION,
        # Keyed on the MODEL too: vbll fits 64 on large and OOMs on huge.
        "micro_batch_size": space.micro_batch(arm, model),
        "bf16": bf16,
        **params,
    }
    # `beta1` and `beta2` arrive through **params, not from a default here.
    # space.suggest draws them in HORIZON units and converts at the trial's own
    # step count, so trial_config never has to know the rule -- and must not
    # re-derive them, since a trial's betas are a searched result rather than a
    # function of the budget.
    #
    # Deriving them here was tried and reverted for a separate reason worth
    # keeping: protocol.beta_from_horizons rightly refuses a horizon longer than
    # the run, and at batch 16,384 the whole budget is a few dozen optimizer
    # steps. That is a fact about those rungs, not a reason trial_config should
    # be unable to build a config for them.
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
    model: str | None = None,
    batch_size: int = space.FIXED_BATCH_SIZE,
    eval_quantum: int = protocol.EVAL_QUANTUM,
    layer_decay: float = protocol.LAYER_DECAY_OFF,
    layer_decay_span: float = protocol.LAYER_DECAY_SPAN_OFF,
) -> Callable[[optuna.Trial], float]:
    """The optuna objective closure for one arm's study.

    `batch_size`, `eval_quantum` and `layer_decay` are campaign-level and are
    recorded as trial user attrs, not searched. They have to travel together
    with `max_samples`: the beta axes are drawn in horizon units and converted
    at the trial's own STEP count, which is samples/batch, so a campaign that
    changed one without the other would search a space it did not record.
    """

    def run_trial(trial: optuna.Trial) -> float:
        # max_samples, not the default: the beta axes are drawn in horizon
        # units and converted at this trial's step count, so a shortened smoke
        # budget must convert against ITS budget rather than the campaign's.
        params = space.suggest(trial, arm, max_samples=max_samples,
                               batch_size=batch_size)
        # Campaign-level settings ride in through params, which trial_config
        # splats LAST -- so they override its defaults without it needing to
        # know about them.
        params["eval_quantum"] = eval_quantum
        if layer_decay != protocol.LAYER_DECAY_OFF:
            params["layer_decay"] = layer_decay
        # The SPAN is what transfers across backbones -- a per-block decay
        # means a different trunk shape at every depth. Passed through to the
        # script, which resolves it against the loaded backbone's real depth.
        if layer_decay_span != protocol.LAYER_DECAY_SPAN_OFF:
            params["layer_decay_span"] = layer_decay_span
        if model is not None:
            params["model"] = model
        config = trial_config(
            arm, params=params, max_samples=max_samples, dump_root=dump_root)
        trial.set_user_attr("max_samples", max_samples)
        trial.set_user_attr("eval_quantum", eval_quantum)
        trial.set_user_attr("layer_decay", layer_decay)
        trial.set_user_attr("layer_decay_span", layer_decay_span)
        trial.set_user_attr("model", config["model"])
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
                   help="Trials THIS worker runs, PRUNED ONES INCLUDED. A cap "
                        "on the job, not a search depth -- use "
                        "--target-complete for that. Size it against the FULL "
                        "trial cost (~3 h at the pinned batch), not the "
                        "hoped-for average.")
    p.add_argument(
        "--target-complete", type=int, default=None,
        help="Stop once the STUDY has at least this many COMPLETED trials, "
             "counting every earlier submission's. Invariant to the prune "
             "rate, which --n-trials is not.",
    )
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
    p.add_argument("--model", default=None,
                   help="Backbone to tune on. Defaults to space.TUNING_MODEL.")
    p.add_argument("--batch-size", type=int, default=space.FIXED_BATCH_SIZE,
                   help="EFFECTIVE batch for every trial. Not searched. Moves "
                        "the step count, and therefore what the beta horizon "
                        "axes can express -- see space.beta1_horizons_box.")
    p.add_argument("--eval-quantum", type=int, default=protocol.EVAL_QUANTUM,
                   help="Samples between evaluations. The pruner and the "
                        "patience clock are counted in EVALUATIONS, so this "
                        "sets whether they can fire at all: max_samples / "
                        "quantum must exceed warmup + patience.")
    p.add_argument("--layer-decay-span", type=float,
                   default=protocol.LAYER_DECAY_SPAN_OFF,
                   help="Campaign-level LLRD span: the fraction of the top "
                        "block's rate the embeddings receive, resolved at each "
                        "backbone's real depth. Use this rather than "
                        "--layer-decay when a campaign spans backbones.")
    p.add_argument("--layer-decay", type=float,
                   default=protocol.LAYER_DECAY_OFF,
                   help="Layer-wise LR decay for the trunk, applied to every "
                        "trial. Recipe, not a searched axis.")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    # Refused HERE, not discovered after a campaign of trials that could never
    # be stopped. Both clocks are counted in EVALUATIONS, so a coarse quantum
    # silently disables them: at 100k samples and quantum 16384 there are 6
    # evaluations against a warmup of 10.
    n_evals = args.max_samples // args.eval_quantum
    need = args.warmup_evals + (args.patience or 0) + 1
    if n_evals <= args.warmup_evals:
        raise SystemExit(
            f"[ft-tune] {args.max_samples:,} samples at quantum "
            f"{args.eval_quantum} gives {n_evals} evaluations, at or below the "
            f"pruner warmup of {args.warmup_evals}: the pruner can never fire. "
            f"Lower --eval-quantum or raise --max-samples."
        )
    if args.patience and n_evals < need:
        raise SystemExit(
            f"[ft-tune] {n_evals} evaluations cannot reach the earliest "
            f"patience stop at {need} (warmup {args.warmup_evals} + patience "
            f"{args.patience}). Lower --eval-quantum, or --patience."
        )

    study = open_study(args.arm, args.storage_dir, args.campaign)
    group_id = f"fttune_{study_name(args.arm, args.campaign)}"

    # Checked BEFORE the first trial, so re-running a finished submission costs
    # a queue slot rather than a 3 h trial. Makes the submit script idempotent:
    # arms already at depth exit immediately, arms short of it top up.
    if args.target_complete is not None:
        already = n_completed(study)
        if already >= args.target_complete:
            # TARGET-MET is a machine token, not decoration: the SLURM job
            # script runs this worker in CHUNKS of trials -- a fresh process per
            # chunk, to clear the arena that fragments across a long campaign --
            # and greps for it to stop starting chunks that would each pay a
            # process start to rediscover the study is done. Keep the two in
            # step; slurm_finetune_tune.sh names this line.
            print(f"[ft-tune] TARGET-MET {study.study_name}: {already} "
                  f"completed >= target {args.target_complete}, nothing to do.")
            return
        print(f"[ft-tune] {study.study_name}: {already} completed, "
              f"target {args.target_complete}, cap {args.n_trials} this worker.")

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
            model=args.model,
            batch_size=args.batch_size,
            eval_quantum=args.eval_quantum,
            layer_decay=args.layer_decay,
            layer_decay_span=args.layer_decay_span,
        ),
        n_trials=args.n_trials,
        # Scoped to `Exception`, deliberately NOT `BaseException`:
        # optuna.TrialPruned is itself an Exception subclass but is
        # special-cased by Optuna before this tuple is consulted, and
        # KeyboardInterrupt/SystemExit derive from BaseException so a job
        # timeout still tears the worker down instead of being swallowed.
        catch=(Exception,),
        # abort_on_consecutive_failures is FIRST and unconditional: it has to
        # run even when --target-complete is absent, and a study that is failing
        # every trial must die before anything else gets a say.
        callbacks=(
            [abort_on_consecutive_failures(), log_failed_trial]
            + ([] if args.target_complete is None
               else [stop_at_completed(args.target_complete)])
        ),
    )
    print(summarize_study(study))


if __name__ == "__main__":
    main()
