"""An Optuna TPE worker for one (arm, dataset) study.

Not a run_sweep submission: run_sweep enumerates create_configs() before
submitting, and TPE generates configurations adaptively from the trials already
reported. Four workers share one study through a journal file.

TPE degrades under parallelism -- each worker samples a posterior that has not
yet been updated by the trials in flight beside it -- so the worker count is
deliberately low.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from bllarse.tuning import budgets as budgets_mod
from bllarse.tuning import objective as objective_mod
from bllarse.tuning import space

DIRECTION = "minimize"

# Points on the geometric grid that throttles MLflow and stdout in the training
# scripts. NOT the evaluation grid: evaluation is spaced in samples (see
# space.eval_quantum), and the pruner's key is that sample-spaced index.
N_DUMPS = 40

# Fires only once 20 trials have completed, and never in a trial's first
# PRUNER_WARMUP_EVALS evaluations, so early noise cannot kill a config that
# starts slowly. In samples the warmup is 10 * eval_quantum -- at most five
# epochs, since the quantum is at most half an epoch.
PRUNER_STARTUP_TRIALS = 20
PRUNER_WARMUP_EVALS = 10

VBLL_FIXED = {
    "parameterization": "diagonal",
    "vbll_type": "discriminative",
    "device": "cuda",
}

# px on for every IBProbit trial. PX shares the plain CAVI fixed point exactly
# and only reaches it in fewer iterations, so it is a solver form rather than a
# model hyperparameter -- searching it would let TPE trade compute against fit
# and make trials incomparable in cost.
#
# learn_q OFF for this campaign, by decision: Q is pinned at its q0 seed and q0
# is the tuned axis. The two would otherwise confound each other -- q_init is
# the seed the recursion starts from, so tuning it while the recursion is also
# running measures a moving target, and `_q_recursion`'s own MEASURED
# LIMITATION says the node barely moves at the operating points tested anyway
# (bit-identical mu at q0 = 3.1e-8; "does not make the mechanism responsive" at
# 3.1e-4). Fixing it makes this campaign a clean search over a STATIC process
# noise; whether inferring Q beats the tuned static value is a follow-up
# ablation at that value, not a knob inside the search.
#
# `kappa` stays absent, taking the class default of 0.5.
IBPROBIT_FIXED = {
    "px": True,
    "no_learn_q": True,
}


def study_name(arm: str, dataset: str, campaign: str | None = None) -> str:
    """Study identity, optionally namespaced by campaign.

    `open_study` passes `load_if_exists=True`, so a study name that collides
    with an earlier campaign's RESUMES it. That is the desired behaviour for the
    four workers of one campaign and wrong across campaigns: the trials of the
    first VBLL campaign carry a `prior_scale` distribution of [1e-2, 1e2] and no
    `kl_ratio` at all, so resuming would have TPE model a search space that no
    longer exists. The campaign suffix keeps the old journals readable as
    evidence rather than forcing them to be deleted.
    """
    base = f"{arm}_{dataset}"
    return base if campaign is None else f"{base}_{campaign}"


def open_study(
    arm: str, dataset: str, storage_dir: Any, campaign: str | None = None
) -> optuna.Study:
    """Load or create this (arm, dataset, campaign)'s study.

    JournalStorage on JournalFileBackend, NOT SQLite: SQLite over GPFS with
    concurrent writers is unreliable, and the journal backend is built for
    network filesystems. One file per study means the 14 studies never contend
    with one another -- only the four workers within a study do.
    """
    directory = Path(storage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    name = study_name(arm, dataset, campaign)
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

    Raises SystemExit when NOTHING completed. `catch=(Exception,)` keeps the
    worker alive through a failing trial, which is right -- but if EVERY trial
    fails, `study.best_value` raises ValueError("No trials are completed yet.")
    and THAT traceback becomes the last thing in the SLURM log, burying the
    real cause a hundred lines above it. A misresolved --cache-dir presented
    exactly this way: 32 trials each dying on `OSError: [Errno 30] Read-only
    file system`, surfacing as an opaque optuna error.

    A study where every trial failed is a harness fault -- paths, container,
    missing features -- not a search-space outcome, and it must be
    distinguishable from a configuration that merely scored badly.
    """
    states = [t.state for t in study.trials]
    n_complete = states.count(optuna.trial.TrialState.COMPLETE)
    if n_complete == 0:
        raise SystemExit(
            f"[tune] {study.study_name}: {len(states)} trials, NONE completed "
            f"({states.count(optuna.trial.TrialState.FAIL)} failed, "
            f"{states.count(optuna.trial.TrialState.PRUNED)} pruned). The "
            f"worker ran but produced no usable result -- read the FIRST "
            f"traceback in this log, not the last. Every trial failing is a "
            f"harness fault, not a search outcome."
        )
    return (f"[tune] {study.study_name}: {len(states)} trials, "
            f"{n_complete} completed, best={study.best_value:.5f} "
            f"params={study.best_params}")


def trial_config(
    arm: str,
    dataset: str,
    *,
    params: Mapping[str, Any],
    budgets: Mapping[str, Mapping[str, int]],
    cache_dir: str = ".cache/features",
    split_cache_dir: str = ".cache/splits",
    enable_mlflow: bool = True,
    mlflow_experiment: str = "bllarse-probing-tuning",
) -> dict[str, Any]:
    """A full training config for one sampled parameter set.

    The Stage 1 budget is a STEP budget; `epochs` is only how the scripts
    express it. Converting at the SAMPLED batch size rather than the pilot's is
    what keeps trials comparable: two configs with different batch sizes get the
    same number of gradient steps, not the same number of passes.
    """
    if dataset not in budgets:
        raise KeyError(
            f"no epoch budget for {dataset!r}. Run the Stage 1 pilot and "
            f"`python -m bllarse.tuning.budgets` first -- tuning under a "
            f"truncated budget biases the search toward high learning rates."
        )
    num_classes, img_size = space.DATASETS[dataset]
    target_steps = budgets[dataset]["steps"]
    batch_size = params["batch_size"]

    config: dict[str, Any] = {
        "model": space.TUNING_MODEL,
        "dataset": dataset,
        "num_classes": num_classes,
        "img_size": img_size,
        "cache_dir": cache_dir,
        "split_cache_dir": split_cache_dir,
        "epochs": budgets_mod.epochs_for_steps(dataset, target_steps, batch_size),
        # Passed explicitly rather than left to the script's default so the
        # campaign's evaluation grid is recorded in the trial config and cannot
        # drift if the ladder is ever re-capped mid-campaign.
        "eval_quantum": space.eval_quantum(dataset),
        "n_dumps": N_DUMPS,
        "val_fraction": 0.1,
        "enable_mlflow": enable_mlflow,
        "mlflow_experiment": mlflow_experiment,
        **params,
    }
    if arm == "vbll":
        config.update(VBLL_FIXED)
    else:
        # Both JAX arms run scripts/dinov3/probe.py and select the
        # training step with --arm, so the study's arm name IS the flag value.
        config["arm"] = arm
        if arm == "ibprobit":
            config.update(IBPROBIT_FIXED)
    return config


def make_objective(
    arm: str,
    dataset: str,
    *,
    budgets: Mapping[str, Mapping[str, int]],
    dump_root: Any,
    group_id: str,
    patience: int | None = None,
    warmup_evals: int = objective_mod.DEFAULT_WARMUP_EVALS,
    min_relative_improvement: float = objective_mod.DEFAULT_MIN_RELATIVE_IMPROVEMENT,
    **config_kw: Any,
) -> Callable[[optuna.Trial], float]:
    """The optuna objective closure for one study.

    `patience` is the search-only counterpart to MedianPruner: the pruner
    kills trials that are LOSING, patience stops trials that have FINISHED
    CONVERGING, so the step cap becomes a true maximum rather than a
    fixed price every trial pays. It applies to both tuning seeds -- unlike
    `on_dump`/pruning below, which is deliberately first-seed-only.
    """

    def run_trial(trial: optuna.Trial) -> float:
        params = space.suggest(trial, arm, dataset)
        config = trial_config(
            arm, dataset, params=params, budgets=budgets, **config_kw
        )
        trial.set_user_attr("epochs", config["epochs"])
        trial.set_user_attr("steps", budgets[dataset]["steps"])
        trial.set_user_attr("eval_quantum", config["eval_quantum"])

        # Where this trial's val_nll bottomed out, and how far it got. Recorded
        # from the callback rather than from the dumps because tuning dumps are
        # deleted once the objective is computed -- the journal is all that
        # outlives a trial, so a diagnostic that lives only on disk is no
        # diagnostic at all.
        #
        # This is what says whether `--patience` is set sensibly. A trial with
        # argmin_eval == total_evals never converged: it was still improving
        # when the step budget ran out, and patience had nothing to cut. Many
        # of those at small batch means the BUDGET is too small, not that
        # patience is too large -- and the two call for opposite corrections.
        # First seed only, like the rest of on_dump.
        seen = {"total": 0, "argmin": 0, "best": float("inf")}

        def on_dump(
            eval_index: int, step: int, samples_seen: int,
            scalars: Mapping[str, float],
        ) -> None:
            seen["total"] = eval_index
            value = float(scalars["val_nll"])
            if value < seen["best"]:
                seen["best"], seen["argmin"] = value, eval_index
            # Reported on the EVAL INDEX, never the optimiser step. MedianPruner
            # compares trials at matching keys and trials vary batch_size, so
            # raw steps would rarely even collide -- the pruner would silently
            # never have a cohort to compare against. The eval index is a
            # sample-spaced grid (space.eval_quantum), so matching keys now also
            # means matching samples_seen, which the old geometric step grid did
            # not: at index i a batch-16384 trial had seen 256x the data of a
            # batch-64 one, and the pruner compared them anyway.
            trial.report(float(scalars["val_nll"]), eval_index)
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
            min_relative_improvement=min_relative_improvement,
            on_converged=on_converged,
        )

        # Recorded so a campaign can be audited afterwards for how much
        # patience actually saved, and to notice if it is misfiring.
        trial.set_user_attr("patience_stopped", bool(stopped_at))
        trial.set_user_attr(
            "patience_stopped_at_eval",
            {str(seed): eval_index for seed, eval_index in stopped_at.items()},
        )
        trial.set_user_attr("argmin_eval", seen["argmin"])
        trial.set_user_attr("total_evals", seen["total"])
        trial.set_user_attr(
            "truncated", seen["total"] > 0 and seen["argmin"] == seen["total"]
        )
        return value

    return run_trial


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=sorted(objective_mod.SCRIPTS))
    p.add_argument("--dataset", required=True, choices=sorted(space.DATASETS))
    p.add_argument("--n-trials", type=int, required=True,
                   help="Trials THIS worker runs. With 4 workers, one quarter "
                        "of the study's total.")
    p.add_argument("--storage-dir", required=True)
    p.add_argument("--dump-root",
                   default=os.environ.get("BLLARSE_EVAL_DUMP_ROOT", "eval_dumps"))
    p.add_argument("--budgets", type=Path,
                   default=Path("bllarse_sweeps/epoch_budgets.json"))
    p.add_argument("--cache-dir", default=".cache/features")
    p.add_argument("--split-cache-dir", default=".cache/splits")
    p.add_argument(
        "--campaign", default=None,
        help="Namespace suffix for the study, its journal file, its dump "
             "directory and its MLflow group_id tag. Omit to reproduce the "
             "unsuffixed names of the first campaign. Required whenever the "
             "search space has changed, since open_study loads an existing "
             "study of the same name rather than creating a fresh one.",
    )
    p.add_argument("--mlflow-experiment", default="bllarse-probing-tuning")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument(
        "--patience", type=int, default=None,
        help="Consecutive evaluations with no relative val_nll improvement "
             "that stop a trial's run early. Evaluations are uniformly spaced "
             "in samples (space.eval_quantum), so N of them is always the same "
             "amount of training -- unlike the geometric dump grid this count "
             "used to run on. Default: disabled.",
    )
    p.add_argument(
        "--warmup-evals", type=int,
        default=objective_mod.DEFAULT_WARMUP_EVALS,
        help="Minimum evaluations seen before --patience may fire.",
    )
    p.add_argument(
        "--min-relative-improvement", type=float,
        default=objective_mod.DEFAULT_MIN_RELATIVE_IMPROVEMENT,
        help="Relative val_nll decrease required to reset the patience counter.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    budgets = json.loads(Path(args.budgets).read_text())
    study = open_study(args.arm, args.dataset, args.storage_dir, args.campaign)
    # The group_id carries the campaign too, so the dumps and the MLflow runs
    # of two campaigns never land in the same directory or the same tag value.
    group_id = f"tune_{study_name(args.arm, args.dataset, args.campaign)}"

    def log_failed_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        # Printed unconditionally on any non-COMPLETE, non-PRUNED outcome so a
        # study that is quietly failing every trial is visible in the SLURM log
        # instead of looking like ordinary slow progress.
        if trial.state == optuna.trial.TrialState.FAIL:
            print(f"[tune] WARNING: trial {trial.number} FAILED "
                  f"(arm={args.arm} dataset={args.dataset})")

    study.optimize(
        make_objective(
            args.arm, args.dataset,
            budgets=budgets,
            dump_root=args.dump_root,
            group_id=group_id,
            patience=args.patience,
            warmup_evals=args.warmup_evals,
            min_relative_improvement=args.min_relative_improvement,
            cache_dir=args.cache_dir,
            split_cache_dir=args.split_cache_dir,
            enable_mlflow=not args.no_mlflow,
            mlflow_experiment=args.mlflow_experiment,
        ),
        n_trials=args.n_trials,
        # `catch=(Exception,)`, not the default (propagate): this is exactly
        # how 13/56 worker tasks died on the CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED
        # crash -- each one abandoned the 15+ trials it still had left. A
        # caught trial is recorded FAILED and the worker moves on. Scoped to
        # `Exception`, deliberately NOT `BaseException`: optuna.TrialPruned is
        # itself an Exception subclass but must reach Optuna's own handling
        # inside `study.optimize`, not be caught here and misrecorded as a
        # failure -- Optuna special-cases TrialPruned before this `catch` tuple
        # is consulted. `KeyboardInterrupt`/`SystemExit` derive from
        # BaseException, not Exception, so this scope already excludes them
        # without needing to say so explicitly -- a Ctrl-C or job-timeout SIGTERM
        # still tears the worker down instead of being swallowed as a failure.
        catch=(Exception,),
        callbacks=[log_failed_trial],
    )
    print(summarize_study(study))


if __name__ == "__main__":
    main()
