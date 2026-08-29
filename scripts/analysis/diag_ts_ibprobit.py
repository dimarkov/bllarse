#!/usr/bin/env python3
"""DIAGNOSTIC: what a per-step temperature would do to IBProbit's late rise.

    scripts/analysis/diag_ts_ibprobit.py --dump-root /ptmp/$USER/eval_dumps \\
        --group-id ftfinal --model dinov3_large --arm ibprobit

NOT a reported figure, and it must not become one. results-plan 0.2 fixes
temperature scaling as a comparator on CE ALONE: scaling an arm whose claim is
calibration by construction measures how much a one-parameter rescaling can
rescue a method that says it does not need one. `score_final` keeps that rule
and this script does not change it.

It exists to answer ONE question. Past its NLL minimum at ~50% of the budget,
IBProbit ends +34.2% on dinov3_large while CE raw ends +9.5% and CE+TS +4.6%
-- so half of CE's own degradation is removed by refitting one scalar per step.
If the same scalar removes most of IBProbit's, the deficit is a SCALAR
miscalibration that the drifting trunk induces and IBProbit has no mechanism to
track (Qhat is pinned by no_learn_q, and the q sweep moved nothing across 15x).
If it does not, the damage is in the SHAPE of the predictive and no rescaling
reaches it -- which points away from the optimiser and back at the head.

T is fit on that step's val dump and applied to that step's test dump, never
carried across steps, exactly as the CE+TS curve is built. IBProbit's dumped
array is a normalised log-predictive, so this is the canonical temperature
family on a categorical, p_i^(1/T) / sum_j p_j^(1/T).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bllarse.analysis import curves  # noqa: E402


def summarise(rows, label):
    xs = [r["samples_seen"] for r in rows]
    nll = [r["nll"] for r in rows]
    lo = min(nll)
    i = nll.index(lo)
    rise = (nll[-1] / lo - 1.0) * 100.0
    print(f"    {label:10s} min {lo:.4f} at {xs[i]:>9,} "
          f"({xs[i] / xs[-1]:3.0%})  end {nll[-1]:.4f}  rise {rise:+6.1f}%")
    return {"samples_seen": xs, "nll": nll, "min": lo, "end": nll[-1],
            "rise_pct": rise,
            "temperature": [r.get("temperature") for r in rows]}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump-root", required=True)
    p.add_argument("--group-id", default="ftfinal")
    p.add_argument("--model", default="dinov3_large")
    p.add_argument("--arm", default="ibprobit")
    p.add_argument("--seeds", nargs="*", type=int, default=[137])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out = {}
    for seed in args.seeds:
        uid = f"{args.model}_imagenet1k_{args.arm}_seed{seed}"
        print(f"[ts-diag] {uid}", flush=True)
        raw = curves.run_curve(args.dump_root, args.group_id, uid,
                               skip_corrupt=True)
        ts = curves.run_curve(args.dump_root, args.group_id, uid,
                              temperature_scaled=True, skip_corrupt=True)
        out[uid] = {"raw": summarise(raw, "raw"), "ts": summarise(ts, "TS")}
        temps = [r["temperature"] for r in ts]
        # T over the run. Over-confidence shows up as T climbing above 1: the
        # fit is flattening a predictive that has grown too sharp.
        n = len(temps)
        picks = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        print("    T: " + "  ".join(
            f"{ts[i]['samples_seen'] / 1e6:.1f}M={temps[i]:.3f}" for i in picks),
            flush=True)
        clipped = sum(bool(r.get("temperature_clipped")) for r in ts)
        if clipped:
            # A boundary T is a failed fit, not an optimum, so a rise that
            # survives only because T hit its clip is not evidence either way.
            print(f"    WARNING: {clipped}/{n} temperatures hit the clip",
                  flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(out))
        print(f"[ts-diag] wrote {args.out}")


if __name__ == "__main__":
    main()
