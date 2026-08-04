"""Stage 1 analysis: turn pilot dumps into a per-dataset epoch budget.

The rule from the spec is "raise the budget until at_last_step is false for
>=90% of pilot configs, subject to a ceiling". That is an iteration over
submissions; this reads the same answer off ONE generous pilot in a single
pass, because the dumps carry the whole trajectory. If the pilot was itself
too short, `derive_budgets` says so rather than silently returning the
truncated value.

The 90% is a GLOBAL rate over every pilot run in the group, not a per-dataset
zero-tolerance check -- one truncated run out of dozens is ordinary pilot
noise, and treating any single truncation as disqualifying is stricter than
the spec asks for.

FIXED-BUDGET MODE: at 12,000 pilot steps, the slow end of the learning-rate
range (lr=1e-3) had still not converged on flowers102, food101 and
stanford_cars, while it was already *outperforming* the fast end there.
Rather than chase a fourth pilot to find where lr=1e-3 actually bottoms out,
the project owner decided to fix the campaign's step budget at a generous
30,000 steps, uniformly across all seven datasets. `fixed_budgets` (and the
`--fixed-steps` CLI flag) produce that budget directly, bypassing
`derive_budgets` and the pilot dumps entirely. Every emitted record carries a
`"provenance"` field so a fixed, decided number is never later mistaken for a
measured one.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from bllarse.analysis.curves import run_curve
from bllarse.analysis.thresholds import early_stop
from bllarse.tuning import space

DEFAULT_OUTPUT = Path("bllarse_sweeps/epoch_budgets.json")
DEFAULT_MARGIN = 1.25

# The spec's rule: raise the budget until at_last_step is false for >=90% of
# ALL pilot configs (a rate over the whole pilot group, not per dataset).
DEFAULT_TRUNCATION_TOLERANCE = 0.90


def epochs_for_steps(dataset: str, target_steps: int, batch_size: int) -> int:
    """Epochs needed to reach `target_steps`, rounded up. Never below 1."""
    per_epoch = space.steps_per_epoch(dataset, batch_size)
    if per_epoch < 1:
        raise ValueError(
            f"batch_size={batch_size} exceeds n_train for {dataset}"
        )
    return max(1, -(-target_steps // per_epoch))


def _uid_dataset(uid: str) -> str:
    """`adamw_ce_food101_lr0.0001` -> `food101`. Longest match wins, so
    cifar100 is never read as cifar10."""
    for dataset in sorted(space.DATASETS, key=len, reverse=True):
        if f"_{dataset}_" in uid:
            return dataset
    raise ValueError(f"no known dataset in pilot uid {uid!r}")


def derive_budgets(
    dump_root: Any,
    group_id: str,
    *,
    batch_size: int,
    margin: float = DEFAULT_MARGIN,
    ceiling_steps: int,
    truncation_tolerance: float = DEFAULT_TRUNCATION_TOLERANCE,
) -> dict[str, dict[str, int]]:
    """Per-dataset step and epoch budget covering every pilot run's argmin.

    Takes the MAX argmin step over that dataset's pilot runs, not the mean: the
    budget has to be adequate for the slowest learning rate in the range, since
    that is exactly the config a too-short budget would unfairly score.

    Truncation is checked as a GLOBAL rate across every run in the pilot
    group, matching the spec ("raise the budget until at_last_step is false
    for >=90% of pilot configs"): this raises only when the non-truncated
    fraction over ALL runs falls below `truncation_tolerance`, not when any
    single dataset has a truncated run.
    """
    root = Path(dump_root)
    by_dataset: dict[str, list[int]] = defaultdict(list)
    truncated: dict[str, list[str]] = defaultdict(list)
    total_runs = 0

    group_dir = root / group_id
    if not group_dir.is_dir():
        raise FileNotFoundError(f"no pilot dumps at {group_dir}")

    for uid_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
        uid = uid_dir.name
        dataset = _uid_dataset(uid)
        stop = early_stop(run_curve(root, group_id, uid), "val_nll")
        by_dataset[dataset].append(stop.step)
        total_runs += 1
        if stop.at_last_step:
            truncated[dataset].append(uid)

    all_truncated = sorted(uid for uids in truncated.values() for uid in uids)
    non_truncated_fraction = (
        1.0 - len(all_truncated) / total_runs if total_runs else 1.0
    )
    if non_truncated_fraction < truncation_tolerance:
        raise ValueError(
            f"only {non_truncated_fraction:.1%} of {total_runs} pilot runs "
            f"put their val_nll minimum before the FINAL dumped step "
            f"(need >= {truncation_tolerance:.1%}). Truncated runs: "
            f"{', '.join(all_truncated)}. The pilot budget itself truncated "
            f"too often, so the derived budget would be a lower bound of "
            f"unknown looseness. Raise PILOT_STEPS and re-run the pilot."
        )
    if all_truncated:
        print(
            f"[budgets] WARNING: {len(all_truncated)} of {total_runs} pilot "
            f"runs were truncated (val_nll minimum at the FINAL dumped step) "
            f"but the {non_truncated_fraction:.1%} non-truncated rate still "
            f"clears the {truncation_tolerance:.1%} threshold. A silently "
            f"dropped truncation looks exactly like convergence when it is "
            f"not -- truncated runs: {', '.join(all_truncated)}"
        )

    budgets: dict[str, dict[str, int]] = {}
    for dataset, steps in sorted(by_dataset.items()):
        target = min(int(max(steps) * margin), ceiling_steps)
        budgets[dataset] = {
            "batch_size": batch_size,
            "steps": target,
            "epochs": epochs_for_steps(dataset, target, batch_size),
            "provenance": "derived from pilot",
        }
    return budgets


def fixed_budgets(fixed_steps: int, *, batch_size: int) -> dict[str, dict[str, int]]:
    """A uniform step budget for every dataset in `space.DATASETS`, set by
    decision rather than derived from the pilot.

    See the module docstring: the pilot showed lr=1e-3 still improving on
    flowers102, food101 and stanford_cars at 12,000 steps while already
    beating faster learning rates, so 30,000 was chosen as a generous budget
    rather than chased further with another pilot.

    `batch_size` is a reference for the steps -> epochs conversion only;
    `worker.trial_config` re-derives epochs at each trial's SAMPLED batch size,
    so `steps` is the field that binds. It is clamped per dataset to that
    dataset's own cap, because a reference above the cap documents a config the
    search cannot sample: flowers102 trains on 1,020 images, capping it at 256.
    """
    return {
        dataset: {
            "batch_size": (reference := min(batch_size, space.batch_cap(space.N_TRAIN[dataset]))),
            "steps": fixed_steps,
            "epochs": epochs_for_steps(dataset, fixed_steps, reference),
            "provenance": "fixed by decision",
        }
        for dataset in space.DATASETS
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dump-root", default=None,
        help="Pilot dump root. Required unless --fixed-steps bypasses the "
             "pilot entirely.",
    )
    p.add_argument("--group-id", default="probe_pilot_epochs")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    p.add_argument(
        "--ceiling-steps", type=int, default=8_000,
        help="Hard cap, so a pathological learning rate cannot drive the "
             "budget without bound.",
    )
    p.add_argument(
        "--truncation-tolerance", type=float, default=DEFAULT_TRUNCATION_TOLERANCE,
        help="Minimum GLOBAL non-truncated fraction across all pilot runs "
             "required to accept the derived budget.",
    )
    p.add_argument(
        "--fixed-steps", type=int, default=None,
        help="Bypass derive_budgets and the pilot dumps entirely: emit a "
             "uniform STEPS budget for every dataset in space.DATASETS, "
             "marked with provenance 'fixed by decision'.",
    )
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.fixed_steps is not None:
        budgets = fixed_budgets(args.fixed_steps, batch_size=args.batch_size)
    else:
        if args.dump_root is None:
            raise SystemExit("--dump-root is required unless --fixed-steps is given")
        budgets = derive_budgets(
            args.dump_root, args.group_id,
            batch_size=args.batch_size,
            margin=args.margin,
            ceiling_steps=args.ceiling_steps,
            truncation_tolerance=args.truncation_tolerance,
        )
    args.output.write_text(json.dumps(budgets, indent=2, sort_keys=True) + "\n")
    for dataset, budget in sorted(budgets.items()):
        print(f"[budgets] {dataset:15s} {budget['steps']:6d} steps "
              f"= {budget['epochs']:5d} epochs @ batch {budget['batch_size']} "
              f"({budget['provenance']})")
    print(f"[budgets] wrote {args.output}")


if __name__ == "__main__":
    main()
