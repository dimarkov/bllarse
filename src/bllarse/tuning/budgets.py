"""Stage 1 analysis: turn pilot dumps into a per-dataset epoch budget.

The rule from the spec is "raise the budget until at_last_step is false for
>=90% of pilot configs, subject to a ceiling". That is an iteration over
submissions; this reads the same answer off ONE generous pilot in a single
pass, because the dumps carry the whole trajectory. If the pilot was itself
too short, `derive_budgets` says so rather than silently returning the
truncated value.
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
) -> dict[str, dict[str, int]]:
    """Per-dataset step and epoch budget covering every pilot run's argmin.

    Takes the MAX argmin step over that dataset's pilot runs, not the mean: the
    budget has to be adequate for the slowest learning rate in the range, since
    that is exactly the config a too-short budget would unfairly score.
    """
    root = Path(dump_root)
    by_dataset: dict[str, list[int]] = defaultdict(list)
    truncated: dict[str, list[str]] = defaultdict(list)

    group_dir = root / group_id
    if not group_dir.is_dir():
        raise FileNotFoundError(f"no pilot dumps at {group_dir}")

    for uid_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
        uid = uid_dir.name
        dataset = _uid_dataset(uid)
        stop = early_stop(run_curve(root, group_id, uid), "val_nll")
        by_dataset[dataset].append(stop.step)
        if stop.at_last_step:
            truncated[dataset].append(uid)

    budgets: dict[str, dict[str, int]] = {}
    for dataset, steps in sorted(by_dataset.items()):
        if truncated[dataset]:
            raise ValueError(
                f"{dataset}: {len(truncated[dataset])} of {len(steps)} pilot "
                f"runs put their val_nll minimum at the FINAL dumped step "
                f"({', '.join(truncated[dataset])}). The pilot budget itself "
                f"truncated, so the derived budget would be a lower bound of "
                f"unknown looseness. Raise PILOT_STEPS and re-run the pilot."
            )
        target = min(int(max(steps) * margin), ceiling_steps)
        budgets[dataset] = {
            "batch_size": batch_size,
            "steps": target,
            "epochs": epochs_for_steps(dataset, target, batch_size),
        }
    return budgets


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump-root", required=True)
    p.add_argument("--group-id", default="probe_pilot_epochs")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    p.add_argument(
        "--ceiling-steps", type=int, default=8_000,
        help="Hard cap, so a pathological learning rate cannot drive the "
             "budget without bound.",
    )
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    budgets = derive_budgets(
        args.dump_root, args.group_id,
        batch_size=args.batch_size,
        margin=args.margin,
        ceiling_steps=args.ceiling_steps,
    )
    args.output.write_text(json.dumps(budgets, indent=2, sort_keys=True) + "\n")
    for dataset, budget in sorted(budgets.items()):
        print(f"[budgets] {dataset:15s} {budget['steps']:6d} steps "
              f"= {budget['epochs']:5d} epochs @ batch {budget['batch_size']}")
    print(f"[budgets] wrote {args.output}")


if __name__ == "__main__":
    main()
