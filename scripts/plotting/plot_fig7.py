#!/usr/bin/env python3
"""Fig 7 -- compute and memory to reach the joint threshold.

    python scripts/plotting/plot_fig7.py --scores /ptmp/$USER/large_all.json

The claim is *faster to calibrated*, not just faster to accurate, so the x-axis
is cost and the bar is the cost of reaching a target that includes calibration.

Threshold protocol (results-plan 0.4, implemented in analysis.thresholds, fixed
before any of these numbers were looked at). An earlier version of this figure
took the target from IBProbit's own optimum instead, on the argument that the
question is "how long does each arm take to reach the quality IBProbit
converges to". That is self-referential in a way 0.4 exists to prevent: the
target moves with the proposed method's quality, so the same rule understated
IBProbit when its process noise was mis-scaled and would flatter it now that it
is not. The target comes from the baseline; only the BAR NORMALISATION is
relative to IBProbit, and that rescales every bar equally.

  - the target is JOINT on accuracy, NLL and Brier -- the first step where all
    three are at or better than target, never the first step where one is;
  - targets come from the STRONGEST BASELINE's converged value, never from
    IBProbit. The baseline here is whichever of CE and CE+TS converges to the
    lower NLL, chosen by that rule and printed, not hand-picked;
  - crossings are PER SEED. Crossing the seed-mean curve would let one fast
    seed manufacture a crossing no individual run achieved;
  - a run that never reaches the target is RIGHT-CENSORED and drawn as such --
    an open bar with an arrow at its budget -- never dropped and never
    extrapolated.

There is NO FLOPs panel, and its absence is deliberate. results-plan v2 asks
for one, derived as `samples-to-threshold x per-step cost from bench_step.py`.
But `bench_step.step_flops` is a BACKBONE model -- depth, width, token count --
and is therefore identical across arms by construction. Multiplying every arm
by one constant makes that panel a rescaled copy of samples-to-threshold with
the same ranking, presented as though it measured compute.

The arms do not cost the same. VBLL carries a covariance head and IBProbit runs
128 CAVI iterations per step; neither appears in a trunk-only FLOP count. The
measured wall clock says how much that matters: over the same 78-eval budget on
dinov3_large, CE takes 10.2-10.3 h, IBProbit 10.7-10.9 h and VBLL 12.7-12.9 h.
A model that assigns all three the same per-sample cost is wrong by ~25% on
VBLL, and wrong in the direction that flatters it.

Reinstating the panel needs a per-arm FLOP model -- head cost and, for
IBProbit, the Cholesky term at its own iteration count -- not a shared scalar.
`panel_cost` is general over a per-(arm, seed) cost function and will take one.

Both panels come from separate measurements -- the dump-mtime timeline and
`probe_micro_batch{,_torch}.py` -- passed in as JSON. A panel with no input
draws as empty and says so; it is never filled in from the trajectory, which
cannot support either number.

Peak memory is a property of (arm, model, batch), not of the trajectory: one
measurement per arm is the honest object and panel c carries no seed spread.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import paper_style  # noqa: E402
from bllarse.finetune import imagenet
from bllarse.analysis import final_json, thresholds  # noqa: E402

BASELINES = [("ce", "id"), ("ce", "ts")]


def _mean_curve(data, uids, pass_key="id"):
    """Seed-mean rows for one arm, on the x values any seed reached.

    The TARGETS come from this, so it has to be the mean for every baseline
    variant alike. Taking one arbitrary seed for the TS baseline and the mean
    for the raw one -- which an earlier version did -- makes the target depend
    on which run happened to be first in a dict.
    """
    per_metric = {}
    for metric in ("acc", "nll", "brier", "val_nll"):
        per_seed = {seed: [(x, y) for x, y in zip(
            data["runs"][uid][pass_key]["samples_seen"],
            data["runs"][uid][pass_key][metric])] for seed, uid in uids.items()}
        xs, mean, _lo, _hi, _n = final_json.band(per_seed)
        per_metric[metric] = dict(zip(xs, mean))
    # Intersection, not union: the reference point is chosen by val_nll and
    # scored on acc/nll/brier, so a step missing any one of them cannot be the
    # reference and must not enter the curve half-formed.
    xs = sorted(set.intersection(*(set(v) for v in per_metric.values())))
    return [{"step": x, "samples_seen": x,
             **{m: per_metric[m][x] for m in per_metric}} for x in xs]


def _is_truncated(data, uid, pass_key) -> bool:
    """Was this run's val minimum its LAST evaluation?

    thresholds.early_stop calls that `at_last_step`: the curve had not turned
    when the budget ran out, so the run says nothing about where its optimum is.
    """
    rows = final_json.rows(data, uid, pass_key)
    if not rows:
        return False
    return bool(thresholds.early_stop(rows).at_last_step)


def pick_targets(data, fraction, reference="early_stop"):
    """Targets from the strongest of the CE variants, at its reference point.

    "Strongest" is decided at the same reference point the targets come from,
    not at the endpoint: with a non-monotone curve the two can disagree about
    which CE variant is stronger.
    """
    best = None
    for arm, pass_key in BASELINES:
        by_arm = final_json.uids_by_arm(data, pass_key)
        uids = by_arm.get(arm)
        if not uids:
            continue
        curve = _mean_curve(data, uids, pass_key)
        if not curve:
            continue
        if reference == "early_stop":
            if "val_nll" not in curve[0]:
                raise SystemExit(
                    "[fig7] --reference early_stop needs val_nll on the scored "
                    "curves; re-score with a score_final that emits it.")
            chosen = thresholds.early_stop(curve).row
        else:
            chosen = max(curve, key=lambda r: r["step"])
        if best is None or chosen["nll"] < best[0]["nll"]:
            best = (chosen, f"{arm}/{pass_key}", curve)
    if best is None:
        raise SystemExit("[fig7] no CE baseline in the scored JSON")
    chosen, name, curve = best
    targets = thresholds.make_targets(curve, fraction=fraction,
                                      reference=reference)
    print(f"[fig7] baseline {name} at {reference} "
          f"({chosen['samples_seen']:,} samples): acc={chosen['acc']:.4f} "
          f"nll={chosen['nll']:.4f} brier={chosen['brier']:.4f}")
    # A reference taken from a curve that was still improving when the budget
    # ran out is not the baseline's operating point, it is where the baseline
    # happened to be cut off -- and every bar on this figure is timed against
    # it, so a target that is too loose moves all of them, and not by the same
    # amount. Loud, because nothing else on the figure shows it.
    if reference == "early_stop":
        truncated = [uid for uid in final_json.uids_by_arm(data, "id").get("ce", {}).values()
                     if _is_truncated(data, uid, "id")]
        if truncated:
            print(f"[fig7] WARNING: the baseline's own curve is TRUNCATED on "
                  f"{len(truncated)} seed(s) -- its minimum is the last "
                  f"evaluation, so this target is an artefact of where the run "
                  f"stopped. Every multiple below will move once the runs are "
                  f"complete.")
    print(f"[fig7] targets at {fraction:g}: acc>={targets.acc:.4f} "
          f"nll<={targets.nll:.4f} brier<={targets.brier:.4f}")
    return targets, name



def joint_crossings(data, targets):
    """[(label, colour, arm, {seed: (Crossing, row)})] under results-plan 0.4.

    The crossing is JOINT -- the first step at which accuracy, NLL and Brier are
    all at or better than target -- and PER SEED, never the crossing of the
    seed-mean curve, which would let one fast seed manufacture a crossing no
    individual run achieved. A run that never meets all three comes back
    censored and is drawn as such.
    """
    entries = []
    for (arm, pass_key), label, colour, _marker in final_json.ARMS:
        uids = final_json.uids_by_arm(data, pass_key).get(arm)
        if not uids:
            continue
        per_seed = {}
        for seed, uid in uids.items():
            rows = final_json.rows(data, uid, pass_key)
            crossing = thresholds.first_crossing(rows, targets)
            row = next((r for r in rows
                        if r["samples_seen"] == crossing.samples_seen), None)
            per_seed[seed] = (crossing, row)
        entries.append((label, colour, arm, per_seed))
    return entries



def crossings(data, targets):
    """[(label, colour, arm, {seed: Crossing})] in ARMS order."""
    entries = []
    for (arm, pass_key), label, colour, _marker in final_json.ARMS:
        uids = final_json.uids_by_arm(data, pass_key).get(arm)
        if not uids:
            continue
        curves = {seed: final_json.rows(data, uid, pass_key)
                  for seed, uid in uids.items()}
        entries.append((label, colour, arm,
                        thresholds.crossings_per_seed(curves, targets)))
    return entries


def panel_cost(ax, entries, cost, title, ylabel):
    """Bars of mean cost to threshold, per-seed dots, censored runs as bounds.

    `cost(arm, seed, crossing) -> (value, is_bound)`. A censored run has no
    crossing, so its value is the cost of its WHOLE budget and `is_bound` says
    to draw it as a lower bound rather than as a measurement. Returning None
    drops the seed, for panels whose measurement is missing for that run.
    """
    costed = []
    for label, colour, arm, cross in entries:
        reached, censored = [], []
        for seed, c in cross.items():
            got = cost(arm, seed, c)
            if got is None:
                continue
            value, is_bound = got
            (censored if is_bound else reached).append(value)
        costed.append((label, colour, reached, censored, len(cross)))

    finite = [v for _, _, r, c, _ in costed for v in (r + c)]
    if not finite:
        ax.set_title(title, fontsize=10)
        ax.text(0.5, 0.5, "no measurement for these runs", ha="center",
                va="center", fontsize=8, color="#777", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    # Bars on a log axis are drawn from the axis floor, so the floor has to be
    # set from the data. Left to matplotlib it lands at a decade boundary far
    # below the smallest bar and every bar reads as full height.
    floor = min(finite) / 2.0

    for i, (label, colour, reached, censored, n_seeds) in enumerate(costed):
        if reached:
            # height is TOP MINUS BOTTOM. Passing the value itself with a
            # non-zero `bottom` draws the bar to floor+value, which on a log
            # axis put every bar ~1.5x above its own per-seed dots.
            mean = sum(reached) / len(reached)
            ax.bar(i, mean - floor, width=0.6, bottom=floor,
                   color=colour, alpha=0.75, zorder=2)
            ax.plot([i] * len(reached), reached, "o", ms=5, color="k",
                    mfc="white", mew=1.0, zorder=4)
        # A censored run is not a fast run and not a missing one: it is a lower
        # bound at its own budget. Open bar, arrow off the top, and a count.
        # The arrow and the count are placed in POINTS off the bar, never in
        # data units -- a multiplicative offset on a log axis rescales the
        # whole panel and squashes the bars it was meant to annotate.
        for x in censored:
            ax.bar(i, x - floor, width=0.6, bottom=floor, facecolor="none",
                   edgecolor=colour, hatch="///", lw=1.2, zorder=3)
            ax.annotate("", xy=(i, x), xytext=(0, 16),
                        textcoords="offset points",
                        arrowprops=dict(arrowstyle="<|-", color=colour, lw=1.2))
        if censored:
            ax.annotate(f"{len(censored)}/{n_seeds} censored",
                        xy=(i, max(censored)), xytext=(0, 20),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7, color=colour)

    ax.set_xticks(range(len(costed)))
    ax.set_xticklabels([e[0] for e in costed], rotation=20, ha="right",
                       fontsize=8)
    ax.set_yscale("log")
    # Headroom for the censored arrow and its count, which sit ~20 points above
    # the tallest bar. Without it matplotlib clips the label at the axes edge.
    any_censored = any(c for _, _, _, c, _ in costed)
    ax.set_ylim(floor, max(finite) * (1.9 if any_censored else 1.25))
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, lw=0.6, axis="y")
    ax.set_axisbelow(True)


def panel_relative(ax, entries, timelines, baseline_arm):
    """Wall clock to the val-NLL target, RELATIVE to the baseline arm.

    Absolute seconds are a property of the A100 this ran on; the ratio is the
    comparison the paper actually makes, and it is the same number on any
    device the arms share. The baseline sits at 1.0 by construction, so its bar
    is a reference line rather than a result -- its per-seed dots still carry
    its own spread.
    """
    costed = {}
    for label, colour, arm, per_seed in entries:
        reached, censored, accs = [], [], []
        for seed, (crossing, row) in per_seed.items():
            timeline = timelines.get((arm, seed))
            if not timeline:
                continue
            if crossing.censored:
                censored.append(_elapsed_at(timeline, crossing.last_step))
                continue
            reached.append(_elapsed_at(timeline, crossing.samples_seen))
            if row is not None:
                accs.append(row["acc"])
        costed[arm] = (label, colour, reached, censored, accs, len(per_seed))

    base = costed.get(baseline_arm)
    if not base or not base[2]:
        raise SystemExit(f"[fig7] {baseline_arm} never reaches the target; "
                         f"there is nothing to normalise by")
    unit = sum(base[2]) / len(base[2])
    base_acc = sum(base[4]) / len(base[4]) if base[4] else None
    print(f"[fig7] normalising by {baseline_arm}: {unit:.0f} s = 1.0")

    order = [arm for (arm, _p), _l, _c, _m in final_json.ARMS if arm in costed]
    note_size = plt.rcParams["xtick.labelsize"]
    # Headroom set from the data before anything is drawn: the notes sit above
    # the tallest DOT, not the tallest bar, and a mean bar well below a
    # spread-out seed put them outside the axes.
    tallest = max((v / unit for _l, _c, r, c, _a, _n in costed.values()
                   for v in (r + c)), default=1.0)
    ax.set_ylim(0, tallest * 1.34)

    for i, arm in enumerate(order):
        label, colour, reached, censored, accs, n_seeds = costed[arm]
        rel = [v / unit for v in reached]
        rel_censored = [v / unit for v in censored]
        if rel:
            ax.bar(i, sum(rel) / len(rel), width=0.6, color=colour, alpha=0.75,
                   lw=0, zorder=2)
            ax.plot([i] * len(rel), rel, "o", ms=4.5, color="k", mfc="white",
                    mew=0.9, zorder=4)
        # A censored run is not a fast run and not a missing one: it is a lower
        # bound at its own budget. Open hatched bar per seed, one arrow off the
        # tallest, and a count.
        for v in rel_censored:
            ax.bar(i, v, width=0.6, facecolor="none", edgecolor=colour,
                   hatch="///", lw=1.0, zorder=3)
        if rel_censored:
            ax.annotate("", xy=(i, max(rel_censored)), xytext=(0, 14),
                        textcoords="offset points",
                        arrowprops=dict(arrowstyle="<|-", color=colour, lw=1.2,
                                        shrinkA=0, shrinkB=0))
        # What ACCURACY the run had when its val NLL crossed. The panel gates
        # on one metric, so the other one has to be stated rather than assumed
        # to have come along -- an arm can cross first and still be behind here.
        top = max(rel + rel_censored) if (rel or rel_censored) else 0
        if accs:
            acc = sum(accs) / len(accs)
            # The difference to the normalising arm, which for that arm itself
            # is zero by construction and is not printed.
            delta = ("" if base_acc is None or arm == baseline_arm
                     else f"\n({acc - base_acc:+.3f})")
            ax.annotate(f"acc {acc:.3f}{delta}", xy=(i, top), xytext=(0, 6),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=note_size, color="0.25")
        if censored:
            ax.annotate(f"{len(censored)}/{n_seeds} censored", xy=(i, top),
                        xytext=(0, 16), textcoords="offset points",
                        ha="center", va="bottom", fontsize=note_size,
                        color=colour)

    ax.axhline(1.0, color="0.4", ls=":", lw=1.0, zorder=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([costed[a][0] for a in order])
    ax.set_xlim(-0.55, len(order) - 0.45)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(f"Wall-clock time / {costed[baseline_arm][0]}")



def _elapsed_at(timeline, samples):
    """Wall-clock seconds at `samples`, from a measured [(samples, s)] list.

    Interpolated between the two bracketing eval points rather than snapped to
    one: a crossing lands at an eval point by construction, but the LAST point
    of a censored run is asked for its own budget and must return it exactly.
    """
    if not timeline:
        return None
    for x, secs in timeline:
        if x >= samples:
            return secs
    return timeline[-1][1]


def make_figure(data, out_path, *, targets, baseline_name, fraction,
                reference, normalise_by, wallclock):
    # A single panel, drawn at 0.65 of the text width and included at that.
    # The target, the baseline it comes from and the normalisation are stated
    # in the caption; `main` prints them.
    paper_style.use()
    fig, axes = paper_style.figure(1, 1, width=0.65 * paper_style.TEXT_WIDTH,
                                   height=2.5)
    ax = axes[0][0]
    entries = joint_crossings(data, targets)

    # {(arm, seed): [(samples, elapsed_s)]} -- a MEASURED timeline per run, so
    # the panel reads the clock at the crossing instead of assuming throughput
    # is flat across a run that pays compilation once and evaluation per dump.
    timelines = {}
    for run in (wallclock.get("runs") or {}).values():
        timelines[(run["arm"], int(run["seed"]))] = run["elapsed_s"]
    if not timelines:
        raise SystemExit("[fig7] --wallclock is required: this panel is a time")

    panel_relative(ax, entries, timelines, normalise_by)
    print(f"[fig7] joint target: acc >= {targets.acc:.4f}, "
          f"NLL <= {targets.nll:.4f}, Brier <= {targets.brier:.4f} "
          + (f"({fraction:g}x {baseline_name})" if " at " in baseline_name
             else f"({fraction:g}x {baseline_name} at its {reference})")
          + f", relative to {normalise_by}")
    paper_style.save(fig, out_path)


# The arm whose time is measured in the baseline-endpoint mode.
METHOD = ("ibprobit", "id")


def endpoint_targets(data, arm, pass_key):
    """One arm's seed-mean values at the end of its run, as joint targets.

    "The end" is the largest evaluation present in EVERY seed of the arm, so
    the mean is over all of them. Returns (targets, samples_seen, uids).
    """
    uids = final_json.uids_by_arm(data, pass_key).get(arm)
    if not uids:
        return None
    budget = min(max(r["samples_seen"] for r in final_json.rows(data, uid, pass_key))
                 for uid in uids.values())
    targets, row = thresholds.targets_at_samples(
        _mean_curve(data, uids, pass_key), budget, 1.0)
    return targets, row["samples_seen"], uids


def time_to_baseline_endpoints(data, timelines):
    """How long METHOD needs to reach what each baseline has at its endpoint.

    The reverse of `--target-arm ibprobit`, and the comparison that can be
    made when the baselines never reach the method's level: the target comes
    from the BASELINE, at the end of its own run, and it is the method that is
    timed against it. Each baseline sets its own target, so one baseline being
    weaker than the other censors nothing.

    The crossing is joint and per seed of the method, as everywhere else in
    this file. A seed that never reaches a target is kept as censored, at the
    time its own run ended.

    Returns [(label, colour, targets, baseline_seconds, samples,
              {seed: (seconds, samples_seen, censored)})].
    """
    method_arm, method_pass = METHOD
    method_uids = final_json.uids_by_arm(data, method_pass).get(method_arm)
    if not method_uids:
        raise SystemExit(f"[fig7] no {method_arm!r} runs in the scores")

    out = []
    for (arm, pass_key), label, colour, _marker in final_json.ARMS:
        if (arm, pass_key) == METHOD:
            continue
        found = endpoint_targets(data, arm, pass_key)
        if found is None:
            continue
        targets, samples, uids = found
        totals = [_elapsed_at(timelines[(arm, seed)], samples)
                  for seed in uids if timelines.get((arm, seed))]
        if not totals:
            continue
        baseline_seconds = sum(totals) / len(totals)

        per_seed = {}
        for seed, uid in sorted(method_uids.items()):
            timeline = timelines.get((method_arm, seed))
            if not timeline:
                continue
            crossing = thresholds.first_crossing(
                final_json.rows(data, uid, method_pass), targets)
            at = crossing.last_step if crossing.censored else crossing.samples_seen
            per_seed[seed] = (_elapsed_at(timeline, at), at, crossing.censored)
        out.append((label, colour, targets, baseline_seconds, samples, per_seed))
    return out


def make_endpoint_figure(data, out_path, *, wallclock):
    """The method's time to each baseline's endpoint, over that baseline's time.

    One bar per baseline. The bar is the mean over the method's seeds and the
    circles are the seeds, both as a fraction of the baseline's own training
    time, so the dotted line at 1 is the baseline. The number above a bar is
    the speed-up, the inverse of the bar.
    """
    timelines = _timelines(wallclock)
    entries = time_to_baseline_endpoints(data, timelines)
    if not entries:
        raise SystemExit("[fig7] no baseline with both scores and a timeline")
    n_train = imagenet.IMAGENET_N_TRAIN

    paper_style.use()
    fig, axes = paper_style.figure(1, 1, width=0.5 * paper_style.TEXT_WIDTH,
                                   height=2.4)
    ax = axes[0][0]
    note_size = plt.rcParams["xtick.labelsize"]

    for i, (label, colour, targets, base_s, samples, per_seed) in enumerate(entries):
        reached = [t for t, _x, censored in per_seed.values() if not censored]
        bounds = [t for t, _x, censored in per_seed.values() if censored]
        print(f"[fig7] target from {label} at {samples:,} samples "
              f"({samples / n_train:.2f} epochs, {base_s / 3600:.2f} h): "
              f"acc >= {targets.acc:.4f}, NLL <= {targets.nll:.4f}, "
              f"Brier <= {targets.brier:.4f}")
        for seed, (t, x, censored) in per_seed.items():
            print(f"[fig7]   seed {seed}: "
                  + ("NEVER reached, run ended at " if censored else "reached at ")
                  + f"{x:,} samples ({x / n_train:.2f} epochs), "
                  f"{t / 3600:.2f} h, {t / base_s:.3f} of the baseline's time")
        top = 0.0
        if reached:
            mean = sum(reached) / len(reached)
            epochs = [x / n_train for _t, x, c in per_seed.values() if not c]
            print(f"[fig7]   mean {mean / 3600:.2f} h and "
                  f"{sum(epochs) / len(epochs):.2f} epochs, "
                  f"{mean / base_s:.3f} of the baseline's time, "
                  f"speed-up {base_s / mean:.2f}x "
                  f"(per seed {base_s / max(reached):.2f}x to "
                  f"{base_s / min(reached):.2f}x)")
            ax.bar(i, mean / base_s, width=0.6, color=colour, alpha=0.75,
                   lw=0, zorder=2)
            ax.plot([i] * len(reached), [t / base_s for t in reached], "o",
                    ms=4.5, color="k", mfc="white", mew=0.9, zorder=4)
            top = max(reached) / base_s
            ax.annotate(f"{base_s / mean:.1f}$\\times$", xy=(i, top),
                        xytext=(0, 5), textcoords="offset points", ha="center",
                        va="bottom", fontsize=note_size, color="0.25")
        # A seed that never reaches the target is a LOWER bound on the time,
        # at the end of its own run: an open marker with an arrow, and a count.
        if bounds:
            ax.plot([i] * len(bounds), [t / base_s for t in bounds], "^",
                    ms=5.0, color=colour, mfc="white", mew=1.0, zorder=4)
            ax.annotate(f"{len(bounds)}/{len(per_seed)} never",
                        xy=(i, max(max(bounds) / base_s, top)), xytext=(0, 16),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=note_size, color=colour)

    ax.axhline(1.0, color="0.4", ls=":", lw=1.0, zorder=1)
    ax.set_xticks(range(len(entries)))
    ax.set_xticklabels([e[0] for e in entries])
    ax.set_xlim(-0.6, len(entries) - 0.4)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, max(1.12, ax.get_ylim()[1]))
    ax.set_xlabel("Baseline that sets the target")
    ax.set_ylabel("IBProbit time / baseline time")
    paper_style.save(fig, out_path)


def _timelines(wallclock):
    timelines = {}
    for run in (wallclock.get("runs") or {}).values():
        timelines[(run["arm"], int(run["seed"]))] = run["elapsed_s"]
    if not timelines:
        raise SystemExit("[fig7] --wallclock is required: this figure is a time")
    return timelines


def cost_to_endpoint(data, timelines, target_arm):
    """Training each arm needs to reach `target_arm`'s performance at its endpoint.

    ONE target for every arm: the seed-mean accuracy, NLL and Brier of
    `target_arm` at the end of its own run. The target arm reaches it at that
    point by construction, so its cost is the length of its run and is not
    read off a per-seed crossing -- about half of its seeds end below their
    own mean and would otherwise count as never reaching a level that is
    defined as where the arm ends.

    Every other arm is timed at its first joint crossing, per seed. A seed that
    never crosses is censored at the end of its own run: a lower bound.

    Returns (targets, samples, [(arm, label, colour,
                                 {seed: (hours, epochs, censored)})]).
    """
    spec = next(((arm, pass_key) for (arm, pass_key), _l, _c, _m in final_json.ARMS
                 if arm == target_arm), None)
    found = spec and endpoint_targets(data, *spec)
    if not found:
        raise SystemExit(f"[fig7] no {target_arm!r} runs in the scores")
    targets, samples, _uids = found
    n_train = imagenet.IMAGENET_N_TRAIN

    entries = []
    for (arm, pass_key), label, colour, _marker in final_json.ARMS:
        uids = final_json.uids_by_arm(data, pass_key).get(arm)
        if not uids:
            continue
        per_seed = {}
        for seed, uid in sorted(uids.items()):
            timeline = timelines.get((arm, seed))
            if not timeline:
                continue
            if arm == target_arm:
                at, censored = samples, False
            else:
                crossing = thresholds.first_crossing(
                    final_json.rows(data, uid, pass_key), targets)
                censored = crossing.censored
                at = crossing.last_step if censored else crossing.samples_seen
            per_seed[seed] = (_elapsed_at(timeline, at) / 3600.0, at / n_train,
                              censored)
        entries.append((arm, label, colour, per_seed))
    return targets, samples, entries


def make_cost_to_endpoint_figure(data, out_path, *, wallclock, target_arm):
    """Two panels, wall-clock hours and epochs, to one baseline's endpoint.

    A filled bar is the mean over the seeds that reach the target, with the
    seeds as circles. A seed that never reaches it is an open hatched bar at
    the length of its own run, with one arrow: a lower bound, not a
    measurement. The number above a bar is the speed-up against the target arm.
    """
    timelines = _timelines(wallclock)
    targets, samples, entries = cost_to_endpoint(data, timelines, target_arm)
    n_train = imagenet.IMAGENET_N_TRAIN
    print(f"[fig7] target from {target_arm} at {samples:,} samples "
          f"({samples / n_train:.2f} epochs): acc >= {targets.acc:.4f}, "
          f"NLL <= {targets.nll:.4f}, Brier <= {targets.brier:.4f}")

    paper_style.use()
    fig, axes = paper_style.figure(1, 2, width=0.85 * paper_style.TEXT_WIDTH,
                                   height=2.35)
    note_size = plt.rcParams["xtick.labelsize"]
    panels = [(0, "Wall-clock time (hours)", "h"), (1, "Epochs", "epochs")]
    base = next(per_seed for arm, _l, _c, per_seed in entries if arm == target_arm)

    for ax, (k, ylabel, unit) in zip(axes[0], panels):
        base_mean = sum(v[k] for v in base.values()) / len(base)
        tallest = 0.0
        for i, (arm, label, colour, per_seed) in enumerate(entries):
            reached = [v[k] for v in per_seed.values() if not v[2]]
            bounds = [v[k] for v in per_seed.values() if v[2]]
            tallest = max([tallest] + reached + bounds)
            if reached:
                mean = sum(reached) / len(reached)
                ax.bar(i, mean, width=0.6, color=colour, alpha=0.75, lw=0,
                       zorder=2)
                ax.plot([i] * len(reached), reached, "o", ms=4.5, color="k",
                        mfc="white", mew=0.9, zorder=4)
                if arm != target_arm:
                    ax.annotate(f"{base_mean / mean:.1f}$\\times$",
                                xy=(i, max(reached)), xytext=(0, 5),
                                textcoords="offset points", ha="center",
                                va="bottom", fontsize=note_size, color="0.25")
            for v in bounds:
                ax.bar(i, v, width=0.6, facecolor="none", edgecolor=colour,
                       hatch="///", lw=1.0, zorder=3)
            if bounds:
                ax.annotate("", xy=(i, max(bounds)), xytext=(0, 14),
                            textcoords="offset points",
                            arrowprops=dict(arrowstyle="<|-", color=colour,
                                            lw=1.2, shrinkA=0, shrinkB=0))
                # The count only when it is partial; an arm in which no seed
                # reaches the target is described in the caption.
                if reached:
                    ax.annotate(f"{len(bounds)}/{len(per_seed)} censored",
                                xy=(i, max(bounds)), xytext=(0, 16),
                                textcoords="offset points", ha="center",
                                va="bottom", fontsize=note_size, color=colour)
            if k == 0:
                n_c = len(bounds)
                print(f"[fig7] {label}: "
                      + (f"{sum(reached) / len(reached):.2f} {unit} "
                         f"({min(reached):.2f}-{max(reached):.2f})"
                         if reached else "never reached")
                      + (f"; {n_c}/{len(per_seed)} censored at "
                         f"{sum(bounds) / n_c:.2f} {unit}" if n_c else "")
                      + (f"; epochs {sum(v[1] for v in per_seed.values() if not v[2]) / len(reached):.2f}"
                         if reached else ""))
        ax.set_xticks(range(len(entries)))
        ax.set_xticklabels([e[1].replace("AdamW+", "") for e in entries])
        ax.set_xlim(-0.6, len(entries) - 0.4)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, tallest * 1.2)
        ax.set_ylabel(ylabel)
        paper_style.few_ticks(ax, y=5)
    paper_style.save(fig, out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", required=True)
    p.add_argument("--out-dir", default="scripts/figures")
    p.add_argument("--name", default="fig7_compute_to_threshold")
    p.add_argument("--fraction", type=float, default=0.95)
    p.add_argument("--target-arm", default=None,
                   help="Anchor the target on THIS arm's value at "
                        "--target-epochs of training, instead of on the "
                        "strongest baseline's early-stopping point. Pinned by "
                        "budget rather than by the arm's own argmin, so the bar "
                        "does not move when that arm gets better or worse -- "
                        "which is what makes anchoring on the method under test "
                        "legitimate here and not in results-plan 0.4's sense.")
    p.add_argument("--target-epochs", type=float, default=1.0,
                   help="Budget, in epochs of the training set, at which "
                        "--target-arm's value is read.")
    p.add_argument("--reference", default="early_stop",
                   choices=["early_stop", "endpoint"],
                   help="Which point of the STRONGEST CE variant sets the "
                        "target: its val-NLL early-stopping point, or its last "
                        "evaluation.")
    p.add_argument("--normalise-by", default="ibprobit",
                   help="Display only: whose time the bars are expressed as a "
                        "multiple of. This does NOT set the target -- "
                        "results-plan 0.4 fixes that at the strongest baseline "
                        "-- and changing it rescales every bar equally.")
    p.add_argument("--target-endpoint", default=None, metavar="ARM",
                   help="ONE target for every arm: what ARM (e.g. ce) has at "
                        "the end of its own run. Two panels, hours and "
                        "epochs; an arm that never reaches the target is "
                        "drawn right-censored at the length of its run. "
                        "Ignores --fraction, --target-arm and --reference.")
    p.add_argument("--baseline-endpoints", action="store_true",
                   help="The reverse comparison: for each baseline, the target "
                        "is what that baseline has at the end of its own run, "
                        "and the figure is the time IBProbit needs to reach it "
                        "as a fraction of the baseline's training time. For "
                        "when the baselines never reach IBProbit's level. "
                        "Ignores --fraction, --target-arm and --reference.")
    p.add_argument("--wallclock", default=None,
                   help="scripts/analysis/wallclock_from_dumps.py output: a "
                        "measured (samples, elapsed) timeline per run.")
    args = p.parse_args()

    data = final_json.load(args.scores)
    print(f"[fig7] {final_json.provenance(data)}")
    if args.target_endpoint or args.baseline_endpoints:
        wallclock = (json.loads(Path(args.wallclock).read_text())
                     if args.wallclock else {})
        os.makedirs(args.out_dir, exist_ok=True)
        for ext in ("pdf", "png"):
            out_path = os.path.join(args.out_dir, f"{args.name}.{ext}")
            if args.target_endpoint:
                make_cost_to_endpoint_figure(data, out_path, wallclock=wallclock,
                                             target_arm=args.target_endpoint)
            else:
                make_endpoint_figure(data, out_path, wallclock=wallclock)
        return
    if args.target_arm:
        samples = int(round(args.target_epochs * imagenet.IMAGENET_N_TRAIN))
        uids = final_json.uids_by_arm(data, "id").get(args.target_arm)
        if not uids:
            raise SystemExit(f"[fig7] no {args.target_arm!r} runs in the scores")
        curve = _mean_curve(data, uids, "id")
        targets, ref_row = thresholds.targets_at_samples(
            curve, samples, args.fraction)
        baseline_name = (f"{args.target_arm} at {args.target_epochs:g} epoch"
                         f"{'' if args.target_epochs == 1 else 's'}")
        print(f"[fig7] target from {args.target_arm} at "
              f"{ref_row['samples_seen']:,} samples "
              f"({args.target_epochs:g} epochs = {samples:,}): "
              f"acc={ref_row['acc']:.4f} nll={ref_row['nll']:.4f} "
              f"brier={ref_row['brier']:.4f}")
        print(f"[fig7] targets at {args.fraction:g}: acc>={targets.acc:.4f} "
              f"nll<={targets.nll:.4f} brier<={targets.brier:.4f}")
    else:
        targets, baseline_name = pick_targets(data, args.fraction,
                                              args.reference)
    wallclock = json.loads(Path(args.wallclock).read_text()) if args.wallclock else {}

    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        make_figure(data, os.path.join(args.out_dir, f"{args.name}.{ext}"),
                    targets=targets, baseline_name=baseline_name,
                    fraction=args.fraction, reference=args.reference,
                    normalise_by=args.normalise_by, wallclock=wallclock)


if __name__ == "__main__":
    main()
