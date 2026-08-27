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
ARMS = [
    (("ce", "id"), "AdamW+CE", "#4C72B0", "o"),
    (("ce", "ts"), "AdamW+CE+TS", "#55A868", "s"),
    (("vbll", "id"), "VBLL", "#C44E52", "^"),
    (("ibprobit", "id"), "IBProbit", "#8172B2", "D"),
]

# Row order for Figs 5 and 6. `lower_is_better` drives the threshold direction
# in Fig 7 and the "↓"/"↑" in the axis label; `log` is a display choice.
METRICS = [
    ("acc", "accuracy ↑", False, False),
    ("ece", "ECE ↓", True, True),
    ("nll", "NLL ↓", True, True),
    ("brier", "Brier ↓", True, False),
]

# OOD panel order. Matches finetune.imagenet.OOD_KEYS; R and A cover only 200 of
# the 1000 classes and are evaluated under the subset mask, which the caption
# has to say out loud.
OOD_SETS = [
    ("ood_imagenet_r", "ImageNet-R (200c)"),
    ("ood_imagenet_a", "ImageNet-A (200c)"),
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


def ood_series(data, arm, split, metric):
    """{seed: [(samples_seen, value), ...]} for one arm on one shift set.

    The OOD grid is a stride of the dump grid and -- on dinov3_large -- an
    irregular one, because OOD workers were under-provisioned and the trainer
    skipped scheduled checkpoints under backpressure. Returned on its own x
    values, never resampled onto the ID grid.
    """
    out = {}
    for uid, run in data["runs"].items():
        if _arm_of(uid, run) != arm:
            continue
        points = (run.get("ood") or {}).get(split) or {}
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
    return [
        {"step": x, "samples_seen": x, "acc": block["acc"][i],
         "nll": block["nll"][i], "brier": block["brier"][i],
         "ece": block["ece"][i]}
        for i, x in enumerate(xs)
    ]


def uids_by_arm(data, pass_key="id"):
    """{arm: {seed: uid}} for runs that have the requested pass."""
    out = {}
    for uid, run in data["runs"].items():
        if not (run.get(pass_key) or {}).get("samples_seen"):
            continue
        out.setdefault(_arm_of(uid, run), {})[_seed_of(uid, run)] = uid
    return out
