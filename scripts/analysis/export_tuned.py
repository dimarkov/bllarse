"""Write each sweep's TUNED block from the tuning journals it was searched in.

WHY THIS EXISTS. Every TUNED entry the figures have ever been built on was
transcribed by hand out of a study summary -- 21 entries across six modules, up
to six searched parameters each, at 17 significant figures. `test_sweep_budgets`
catches a wrong `epochs` (it recomputes the step budget at that entry's batch)
and a batch off the ladder, but nothing catches a mistyped `learning_rate`,
`q_init`, `kl_ratio`, `prior_scale` or `wishart_scale`: a plausible number in a
plausible place, and the run that reports it is simply not the run that was
selected. Hand-transcription is the one step in the pipeline with no check on
it, so it should not be a step.

What it emits is exactly what `worker.trial_config` builds a trial from -- the
best trial's own params, plus `epochs` recomputed by `budgets.epochs_for_steps`
at that trial's batch size. `epochs` is a restatement of the step budget, never
an independent knob, which is why it is derived here rather than copied.

The campaign is written with the numbers. `CAMPAIGN` decides which directory the
dumps go in (see evaldump.campaign_root), so a TUNED block updated without it
would overwrite the previous campaign's dumps in place -- same group_id, same
uid, different hyperparameters, no error. One tool writes both.

Usage:

    # audit artefact only
    python scripts/analysis/export_tuned.py --campaign win2 \\
        --storage-dir .cache/optuna --out scripts/results/tuned_win2.json

    # and rewrite the six sweep modules
    python scripts/analysis/export_tuned.py --campaign win2 \\
        --storage-dir .cache/optuna --write
"""

import argparse
import json
import os
import re
import sys

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from bllarse.tuning import budgets as budgets_mod  # noqa: E402
from bllarse.tuning import space  # noqa: E402

# Which study feeds which module. The Fig 1 and Fig 2 sweeps of one arm read
# the SAME study per dataset -- studies are keyed on the ID dataset only, since
# both scripts load OOD features after the training loop purely for a final
# dump (see tuning.objective). Nothing here picks the datasets: each module's
# own TUNED keys do, so this can never invent a cell the sweep does not run.
MODULES = {
    "bllarse_sweeps/probe_adamw_ce.py": "adamw_ce",
    "bllarse_sweeps/probe_ood_adamw_ce.py": "adamw_ce",
    "bllarse_sweeps/probe_vbll.py": "vbll",
    "bllarse_sweeps/probe_ood_vbll.py": "vbll",
    "bllarse_sweeps/probe_ibprobit_noeps.py": "ibprobit_noeps",
    "bllarse_sweeps/probe_ood_ibprobit_noeps.py": "ibprobit_noeps",
}

# Emission order, so a re-export produces a minimal diff rather than a reshuffle
# that hides which number moved. `epochs` is appended last by `entry_source`.
PARAM_ORDER = {
    "adamw_ce": ["batch_size", "learning_rate", "weight_decay"],
    "vbll": ["batch_size", "learning_rate", "weight_decay",
             "kl_ratio", "prior_scale", "wishart_scale"],
    "ibprobit_noeps": ["batch_size", "q_init", "num_update_iters"],
}

TUNED_RE = re.compile(r"^TUNED = \{.*?^\}", re.MULTILINE | re.DOTALL)
CAMPAIGN_RE = re.compile(r'^CAMPAIGN = "[^"]*"', re.MULTILINE)


def study_name(arm: str, dataset: str, campaign: str) -> str:
    return f"{arm}_{dataset}_{campaign}"


def read_study(storage_dir: str, arm: str, dataset: str, campaign: str) -> dict:
    """The best trial of one study, with the counts that describe its depth."""
    name = study_name(arm, dataset, campaign)
    path = os.path.join(storage_dir, f"{name}.log")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no journal at {path}")
    study = optuna.load_study(
        study_name=name, storage=JournalStorage(JournalFileBackend(path))
    )
    states = [t.state for t in study.trials]
    best = study.best_trial

    window = best.user_attrs.get("smooth_window")
    return {
        "params": dict(best.params),
        "value": best.value,
        "trial_number": best.number,
        "smooth_window": window,
        "completed": sum(s == optuna.trial.TrialState.COMPLETE for s in states),
        "pruned": sum(s == optuna.trial.TrialState.PRUNED for s in states),
        "failed": sum(s == optuna.trial.TrialState.FAIL for s in states),
    }


def tuned_entry(arm: str, dataset: str, record: dict, target_steps: int) -> dict:
    """The sweep's TUNED value: the trial's own params plus a derived `epochs`.

    Recomputed, never carried over from the study: `epochs` expresses the step
    budget AT THAT TRIAL'S BATCH SIZE, so a re-tune that moves the batch and
    keeps the epochs silently gives one cell a different budget from its
    neighbours -- invisible in the figure, fatal to the comparison.
    """
    params = record["params"]
    missing = [k for k in PARAM_ORDER[arm] if k not in params]
    if missing:
        raise KeyError(f"{arm}/{dataset}: study has no {missing}")
    batch = params["batch_size"]
    entry = {k: params[k] for k in PARAM_ORDER[arm]}
    entry["epochs"] = budgets_mod.epochs_for_steps(dataset, target_steps, batch)
    return entry


def entry_source(dataset: str, entry: dict, record: dict, campaign: str,
                 target_steps: int) -> str:
    """One TUNED line, with the provenance the reviewer needs in the comment."""
    args = ", ".join(f"{k}={v!r}" for k, v in entry.items())
    steps = entry["epochs"] * space.steps_per_epoch(dataset, entry["batch_size"])
    depth = f"{record['completed']} trials"
    if record["failed"]:
        # Named, not smoothed over: on ibprobit_noeps/oxford_pets 30 trials
        # diverged to a non-finite val_nll, all in the same (batch, iters)
        # corner the winner came from. A reader comparing this cell's depth
        # with its neighbours' should see why it is short.
        depth += f" ({record['failed']} diverged)"
    return (f'    "{dataset}": dict({args}),'
            f'  # {campaign}, {depth}, {steps} steps')


def render_tuned(arm: str, datasets, records, campaign: str, budgets) -> str:
    lines = ["TUNED = {"]
    for dataset in datasets:
        record = records[dataset]
        target_steps = budgets[dataset]["steps"]
        entry = tuned_entry(arm, dataset, record, target_steps)
        lines.append(entry_source(dataset, entry, record, campaign, target_steps))
    lines.append("}")
    return "\n".join(lines)


def module_datasets(source: str) -> list[str]:
    """The datasets a module already declares, in its own order."""
    block = TUNED_RE.search(source)
    if block is None:
        raise ValueError("no TUNED block")
    return re.findall(r'^    "([^"]+)":', block.group(0), re.MULTILINE)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", required=True)
    p.add_argument("--storage-dir", default=".cache/optuna")
    p.add_argument("--budgets", default="bllarse_sweeps/epoch_budgets.json")
    p.add_argument("--out", default=None, help="write the audit JSON here")
    p.add_argument("--write", action="store_true",
                   help="rewrite each module's TUNED block and CAMPAIGN")
    args = p.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    budgets = json.loads(open(args.budgets).read())

    audit: dict = {"campaign": args.campaign, "arms": {}}
    for path, arm in MODULES.items():
        source = open(path).read()
        datasets = module_datasets(source)

        records = {}
        for dataset in datasets:
            record = read_study(args.storage_dir, arm, dataset, args.campaign)
            if record["smooth_window"] is not None:
                # The campaign name is not evidence of the objective; the trial
                # is. A journal resumed under the wrong window would otherwise
                # be exported without comment.
                expected = int(args.campaign[3:]) if args.campaign.startswith("win") else None
                if expected is not None and record["smooth_window"] != expected:
                    raise ValueError(
                        f"{arm}/{dataset}: best trial was scored at window "
                        f"{record['smooth_window']}, not {expected}"
                    )
            records[dataset] = record

        block = render_tuned(arm, datasets, records, args.campaign, budgets)
        arm_audit = audit["arms"].setdefault(arm, {})
        for dataset in datasets:
            arm_audit[dataset] = {
                **tuned_entry(arm, dataset, records[dataset],
                              budgets[dataset]["steps"]),
                "value": records[dataset]["value"],
                "completed": records[dataset]["completed"],
                "pruned": records[dataset]["pruned"],
                "failed": records[dataset]["failed"],
                "trial_number": records[dataset]["trial_number"],
            }

        if args.write:
            updated = TUNED_RE.sub(lambda _: block, source, count=1)
            updated, n = CAMPAIGN_RE.subn(
                f'CAMPAIGN = "{args.campaign}"', updated, count=1
            )
            if n != 1:
                raise ValueError(f"{path}: no CAMPAIGN line to update")
            open(path, "w").write(updated)
            print(f"[export] rewrote {path} ({len(datasets)} datasets)")
        else:
            print(f"# {path}\n{block}\n")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(audit, fh, indent=2, sort_keys=True)
        print(f"[export] wrote {args.out}")


if __name__ == "__main__":
    main()
