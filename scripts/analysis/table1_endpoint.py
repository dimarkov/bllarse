#!/usr/bin/env python3
"""Table 1 -- the fine-tuning endpoint, as LaTeX, from score_final's JSON.

    python scripts/analysis/table1_endpoint.py --scores large.json

Emitted rather than typed, because the numbers move every time the campaign is
re-run and a hand-copied table is the one artefact that silently keeps the old
ones. Print it, paste it, regenerate it when the runs change.

MATCHED BUDGET, not each arm's own last evaluation. Arms finish at different
points -- a wall clock, a preemption, a different throughput -- and comparing
one arm's step 78 against another's step 65 is a comparison of run lengths. The
budget used is the largest evaluation present in EVERY run, and it is printed so
the caption can state it.

ECE is omitted deliberately, matching the figures: NLL and Brier are proper and
carry the calibration claim, while ECE depends on a binning choice the table
would not show.

The shift columns average the four sets in `final_json.OOD_SETS` at the last OOD
evaluation at or before the budget -- the OOD grid is a stride of the trajectory
grid, so it rarely lands on the same step.
"""

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bllarse.analysis import final_json  # noqa: E402


def _matched_budget(data) -> int:
    ends = []
    for (arm, pass_key), _label, _c, _m in final_json.ARMS:
        for uid in (final_json.uids_by_arm(data, pass_key).get(arm) or {}).values():
            rows = final_json.rows(data, uid, pass_key)
            if rows:
                ends.append(max(r["samples_seen"] for r in rows))
    if not ends:
        raise SystemExit("[table1] no runs in the scores")
    return min(ends)


def _at(rows, budget):
    eligible = [r for r in rows if r["samples_seen"] <= budget]
    return max(eligible, key=lambda r: r["samples_seen"]) if eligible else None


def _pm(vals, fmt="{:.4f}"):
    if not vals:
        return "---"
    if len(vals) == 1:
        return fmt.format(vals[0])
    return (fmt.format(statistics.fmean(vals)) + r" $\pm$ "
            + fmt.format(statistics.stdev(vals)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", required=True)
    args = p.parse_args()

    data = final_json.load(args.scores)
    budget = _matched_budget(data)
    print(f"% {final_json.provenance(data)}")
    print(f"% matched budget: {budget:,} samples")

    rows_out = []
    for (arm, pass_key), label, _c, _m in final_json.ARMS:
        uids = final_json.uids_by_arm(data, pass_key).get(arm)
        if not uids:
            continue
        idv = {k: [] for k in ("acc", "nll", "brier")}
        shv = {k: [] for k in ("acc", "nll", "brier", "auroc")}
        shift_at = []
        for seed, uid in sorted(uids.items()):
            row = _at(final_json.rows(data, uid, pass_key), budget)
            if row:
                for k in idv:
                    idv[k].append(row[k])
            # shift: average the four sets at their own last point <= budget
            per_metric = {k: [] for k in shv}
            ood_steps = []
            block = (data["runs"][uid].get(
                "ood_ts" if pass_key == "ts" else "ood") or {})
            for split, _title in final_json.OOD_SETS:
                steps = block.get(split) or {}
                elig = [int(s) for s in steps if int(s) <= budget]
                if not elig:
                    continue
                ood_steps.append(max(elig))
                r = steps[str(max(elig))]
                for k in shv:
                    if k in r:
                        per_metric[k].append(r[k])
            for k in shv:
                if per_metric[k]:
                    shv[k].append(statistics.fmean(per_metric[k]))
            if ood_steps:
                shift_at.append(min(ood_steps))
        rows_out.append((label, idv, shv, shift_at))

    # Decide BEFORE printing whether the shift half is comparable. Emitting it
    # under a warning invites the warning to be stripped with the comment.
    used = {label: min(at) for label, _i, _s, at in rows_out if at}
    shift_ok = bool(used) and (max(used.values()) - min(used.values())) <= 0.1 * budget
    if not shift_ok:
        print("% shift columns OMITTED: the arms' OOD grids do not line up.")
        for label, at in sorted(used.items(), key=lambda kv: kv[1]):
            print(f"%   {label:14} would be read at {at:>10,} samples "
                  f"({at / budget:.0%} of the in-distribution budget)")
        print("%   Averaging an early model against late ones under one header "
              "is not a comparison. Fix the sparse arm's OOD coverage.")

    print(r"\begin{tabular}{lccc" + ("|cccc}" if shift_ok else "}"))
    print(r"\toprule")
    if shift_ok:
        print(r"& \multicolumn{3}{c|}{In distribution} "
              r"& \multicolumn{4}{c}{Shift (mean of four sets)} \\")
    print(r"Method & Acc $\uparrow$ & NLL $\downarrow$ & Brier $\downarrow$"
          + (r" & Acc $\uparrow$ & NLL $\downarrow$ & Brier $\downarrow$ "
             r"& AUROC $\uparrow$ \\" if shift_ok else r" \\"))
    print(r"\midrule")
    # Bold the winner per column, computed rather than typed: the ordering
    # changes between campaigns and a hand-bolded cell is how a table ends up
    # crediting the wrong arm.
    cols = [("acc", idv_key := "id", True), ("nll", "id", False),
            ("brier", "id", False)]
    if shift_ok:
        cols += [("acc", "sh", True), ("nll", "sh", False),
                 ("brier", "sh", False), ("auroc", "sh", True)]
    means = []
    for metric, where, higher in cols:
        vals = []
        for _label, idv, shv, _ in rows_out:
            src = idv if where == "id" else shv
            vals.append(statistics.fmean(src[metric]) if src[metric] else None)
        live = [v for v in vals if v is not None]
        best = (max(live) if higher else min(live)) if live else None
        means.append((vals, best))

    for i, (label, idv, shv, _) in enumerate(rows_out):
        cells = []
        for j, (metric, where, _h) in enumerate(cols):
            src = idv if where == "id" else shv
            text = _pm(src[metric])
            vals, best = means[j]
            if best is not None and vals[i] is not None and vals[i] == best:
                text = r"$\bm{" + text.replace(r" $\pm$ ", r" \pm ").replace("$", "") + "}$"
            cells.append(text)
        print(f"{label} & " + " & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
