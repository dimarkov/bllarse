"""Read `scripts/analysis/score_final.py`'s output for the fine-tuning figures.

Scoring one backbone reads ~116 GB and takes hours, so it writes its JSON after
every run and marks it `complete: false` until the last one lands. Everything
here therefore treats PARTIAL input as the normal case: `load` reports how much
of the campaign is present, and `band` records the per-x seed count so a figure
can say that its early points average five seeds and its late points two.

Figs 5, 6 and 7 all read the same file. The arm styling lives here rather than
in one of them so an arm cannot change colour between figures -- the colours are
`scripts/plotting/plot_fig1.LINES`', which is where the paper's palette is set.
"""

import json
from pathlib import Path

# (key, label, colour, marker). `key` is (arm, pass) where pass is "id" for the
# raw curve and "ts" for the temperature-scaled one. CE+TS is a comparator on CE
# ALONE -- score_final does not compute a TS pass for VBLL or IBProbit, because
# their claim is calibration by construction.
#
# Raw CE is NOT drawn. CE+TS is the strong form of the baseline: a practitioner
# with a val split gets it for the cost of a one-parameter fit, so beating raw
# CE on calibration is not the claim worth making. The raw curve is still
# scored and still in the JSON.
ARMS = [
    (("ce", "ts"), "AdamW+CE+TS", "#55A868", "s"),
    (("vbll", "id"), "VBLL", "#C44E52", "^"),
    (("ibprobit", "id"), "IBProbit", "#8172B2", "D"),
]


def ood_arms(data):
    """ARMS as they can be drawn off-distribution, given what was scored.

    T is fit ONCE on the in-distribution val split and applied unchanged to the
    shifted logits -- `score_final` stores that as `ood_ts`. No OOD label is
    read to obtain it. Refitting T on the shifted set would be test-time
    adaptation, and the symmetric move would be to run IBProbit's CAVI update
    or VBLL's posterior update on the same data; every arm here is fit
    in-distribution and then evaluated under shift, full stop.

    A JSON scored before `ood_ts` existed carries no such curve, and dropping
    the entry would delete the baseline from the figure. There the raw curve
    stands in, labelled as raw -- which UNDERSTATES the baseline, since the
    Bayesian arms carry their ID-fitted machinery across while CE would be left
    at T=1.
    """
    has_ood_ts = any(run.get("ood_ts") for run in data["runs"].values())
    out = []
    for (arm, pass_key), label, colour, marker in ARMS:
        if pass_key == "ts" and not has_ood_ts:
            label = label.replace("+TS", "") + " (raw — no TS scored)"
            pass_key = "id"
        out.append(((arm, pass_key), label, colour, marker))
    return out


def load(path):
    data = json.loads(Path(path).read_text())
    if not data.get("runs"):
        raise SystemExit(f"[fig] {path} has no scored runs yet")
    return data


def provenance(data) -> str:
    """One line naming exactly what the figure was drawn from.

    Every figure built off a partial scoring must carry this, unedited. A curve
    over three of fifteen runs looks identical to one over fifteen.
    """
    scored, total = data.get("runs_scored"), data.get("runs_total")
    state = "complete" if data.get("complete") else "PARTIAL"
    line = (f"{data.get('model', '?')} / {data.get('group_id', '?')}: "
            f"{scored} of {total} runs scored ({state})")
    n_corrupt = len(data.get("corrupt_dumps") or ())
    if n_corrupt:
        line += f"; {n_corrupt} unreadable dump(s) skipped"
    return line


def _arm_of(uid, run):
    stored = (run.get("id") or {}).get("arm") or (run.get("ts") or {}).get("arm")
    if stored:
        return stored
    return uid.rsplit("_seed", 1)[0].split("_")[-1]


def _seed_of(uid, run):
    stored = (run.get("id") or {}).get("seed")
    return int(stored) if stored is not None else int(uid.rsplit("_seed", 1)[1])


def id_series(data, arm, pass_key, metric):
    """{seed: [(samples_seen, value), ...]} for one arm on the ID split."""
    out = {}
    for uid, run in data["runs"].items():
        if _arm_of(uid, run) != arm:
            continue
        block = run.get(pass_key) or {}
        xs, ys = block.get("samples_seen"), block.get(metric)
        if not xs or not ys:
            continue
        out[_seed_of(uid, run)] = list(zip(xs, ys))
    return out


def ood_series(data, arm, split, metric, pass_key="id"):
    """{seed: [(samples_seen, value), ...]} for one arm on one shift set.

    `pass_key="ts"` reads the temperature-scaled variant, where that step's
    ID-fit T was applied to the shifted logits.

    The OOD grid is a stride of the dump grid and -- on dinov3_large -- an
    irregular one, because OOD workers were under-provisioned and the trainer
    skipped scheduled checkpoints under backpressure. Returned on its own x
    values, never resampled onto the ID grid.
    """
    out = {}
    for uid, run in data["runs"].items():
        if _arm_of(uid, run) != arm:
            continue
        field = "ood_ts" if pass_key == "ts" else "ood"
        points = (run.get(field) or {}).get(split) or {}
        if not points:
            continue
        out[_seed_of(uid, run)] = sorted(
            (int(x), m[metric]) for x, m in points.items() if metric in m)
    return out


def band(per_seed):
    """(xs, mean, lo, hi, n) over seeds, on the x values any seed reached.

    lo/hi are min and max across seeds, not a standard error: at five seeds the
    spread IS the interesting object and a 1-sigma ribbon hides which runs are
    outliers. `n` is per-x, because a corrupt dump or a censored run leaves some
    x values with fewer seeds than others.
    """
    by_x = {}
    for points in per_seed.values():
        for x, y in points:
            by_x.setdefault(x, []).append(y)
    xs = sorted(by_x)
    mean = [sum(by_x[x]) / len(by_x[x]) for x in xs]
    lo = [min(by_x[x]) for x in xs]
    hi = [max(by_x[x]) for x in xs]
    n = [len(by_x[x]) for x in xs]
    return xs, mean, lo, hi, n


def rows(data, uid, pass_key="id"):
    """One run's curve as threshold-shaped rows.

    `analysis.thresholds` orders on `step` and reports `samples_seen`. The
    scored JSON keeps only samples_seen -- it is the compute axis every figure
    actually uses, and it is monotone in step -- so it stands in for both.
    """
    block = (data["runs"][uid].get(pass_key) or {})
    xs = block.get("samples_seen") or []
    val_nll = block.get("val_nll") or [None] * len(xs)
    rows = []
    for i, x in enumerate(xs):
        row = {"step": x, "samples_seen": x, "acc": block["acc"][i],
               "nll": block["nll"][i], "brier": block["brier"][i],
               "ece": block["ece"][i]}
        # val_nll rides along so a caller can apply a SELECTION rule
        # (thresholds.early_stop, and Fig 7's threshold reference) without
        # reading the test curve it is about to report on. Omitted where the
        # val dump at that step was unreadable, since early_stop refuses a
        # curve whose criterion is missing rather than silently ranking None.
        if val_nll[i] is not None:
            row["val_nll"] = val_nll[i]
        rows.append(row)
    return rows


def uids_by_arm(data, pass_key="id"):
    """{arm: {seed: uid}} for runs that have the requested pass."""
    out = {}
    for uid, run in data["runs"].items():
        if not (run.get(pass_key) or {}).get("samples_seen"):
            continue
        out.setdefault(_arm_of(uid, run), {})[_seed_of(uid, run)] = uid
    return out
