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


# Row order for Figs 5 and 6. `lower_is_better` drives the threshold direction
# in Fig 7 and the "↓"/"↑" in the axis label; `log` is a display choice.
#
# ECE is SCORED but not plotted. It stays in the JSON -- `rows` carries it and
# any analysis can read it -- and it is off the reported figures because NLL
# and Brier are proper scoring rules that already carry the calibration claim,
# while ECE's value depends on a binning choice the figure would not show.
#
# The arrows are mathtext, not the Unicode characters: the figures' text font
# is not guaranteed to have those glyphs.
METRICS = [
    ("acc", r"Accuracy $\uparrow$", False, False),
    ("nll", r"NLL $\downarrow$", True, True),
    ("brier", r"Brier $\downarrow$", True, False),
]

# Where the hyperparameter search actually looked. Everything to the RIGHT of
# this line is the transfer extrapolating past its selection budget rather than
# a tuned operating point -- the first thing to check when a curve turns after
# it. Which makes a WRONG line worse than no line, and there is no honest
# default: the budget is a property of the STUDY a config came from, and none of
# it survives into the scored JSON (which carries model, group_id and per-step
# metrics) or into the dumps.
#
# It used to return protocol.MAX_SAMPLES unconditionally. That is right only for
# the campaigns tuned at a fixed 1M samples; the dinov3 LADDER holds STEPS at
# 781 and lets samples follow the rung's batch, so at dinov3_large it is
# 781 x 512 = 399,872 and the constant overstated it by 2.5x -- drawing the
# extrapolation boundary a decade and a half into territory that was in fact
# searched. Callers now pass the value they know; None draws no line.
def tuning_budget() -> int | None:
    """No default. Pass --tuning-budget, or get no annotation."""
    return None


def format_samples(n: float) -> str:
    """Axis-style sample count: 399872 -> '400k', 5004000 -> '5.0M'.

    The previous `f"{n / 1e6:.0f}M"` rendered every per-rung budget as "0M".
    """
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.1f}M".replace(".0M", "M")
    if n >= 1e3:
        return f"{n / 1e3:.0f}k"
    return f"{n:.0f}"

# Fig 6 gets a fourth row that Fig 5 cannot have: there is no detection score
# without a second distribution to detect against. MSP only, matching the
# probing figure -- it reads the normalised predictive, so it is the one score
# all three arms produce identically, and IBProbit's moderation reaches it
# through the class-independent rescaling rather than being discarded.
#
# Higher is better, unlike every other row here, so `lower_is_better` is False
# and the axis label carries the arrow.
#
# Note what this row is and is not. All four sets share ImageNet's label space,
# so it measures whether the model can tell SHIFTED inputs from clean ones --
# covariate shift. It is not the semantic OOD task of Fig 2 (CIFAR-10 vs SVHN,
# disjoint labels), and the two numbers must not be read across.
OOD_METRICS = METRICS + [("auroc", r"MSP AUROC $\uparrow$", False, False)]

# OOD panel order. Matches finetune.imagenet.OOD_KEYS; R and A cover only 200 of
# the 1000 classes and are evaluated under the subset mask, which the caption
# has to say out loud -- the panel headers name the set and nothing else.
OOD_SETS = [
    ("ood_imagenet_r", "ImageNet-R"),
    ("ood_imagenet_a", "ImageNet-A"),
    ("ood_imagenet_v2", "ImageNet-V2"),
    ("ood_imagenet_sketch", "ImageNet-Sketch"),
]


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
