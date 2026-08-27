#!/usr/bin/env python3
"""Score a finished final-campaign dump tree into one compact JSON.

    scripts/analysis/score_final.py --dump-root /ptmp/$USER/eval_dumps \\
        --group-id ftfinal --model dinov3_large --out large.json

Runs WHERE THE DUMPS ARE. The tree is ~116 GB for one backbone, so the scoring
happens on the cluster and only the distilled curves travel -- a few hundred KB,
enough to plot Fig 5 anywhere.

Emits one row per (arm, seed, step) on the test split, plus -- for the CE arm
ONLY -- the temperature-scaled variant of the same points. results-plan 0.2
fixes CE+TS as a comparator, and it is a POST-HOC fit on the CE run's own dumps
rather than a separate training run. T is refit at every step against that
step's val dump, never reused across steps.

TS is NOT applied to VBLL or IBProbit. Their claim is calibration by
construction, so scaling them would measure how much a one-parameter rescaling
can rescue a method that says it does not need one -- and it is a different
transform there in any case, since those arms dump a normalised log-predictive
rather than logits.

OOD is summarised separately and by STEP, since the OOD grid is a stride of the
dump grid and -- on the dinov3_large campaign -- an irregular one, because
workers were under-provisioned and the trainer skipped scheduled checkpoints
under backpressure. Reporting the two grids together would silently imply an
OOD point exists wherever an ID point does.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bllarse import evaldump, metrics  # noqa: E402
from bllarse.analysis import curves  # noqa: E402


def _uids(dump_root, group_id, model):
    base = Path(dump_root) / group_id
    return sorted(p.name for p in base.iterdir()
                  if p.is_dir() and p.name.startswith(model + "_"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump-root", required=True)
    p.add_argument("--group-id", default="ftfinal")
    p.add_argument("--model", default="dinov3_large")
    p.add_argument("--out", required=True)
    p.add_argument("--skip-ts", action="store_true",
                   help="Skip the temperature-scaled pass. It roughly doubles "
                        "the work -- a fresh Newton fit per step against that "
                        "step's val dump, on top of reading the val dumps at "
                        "all -- and the ID curves are readable without it.")
    p.add_argument("--seeds", nargs="*", type=int, default=None,
                   help="Score only these seeds. The full tree is ~116 GB per "
                        "backbone, and one seed per arm is enough to see the "
                        "curves and validate the pipeline while the cluster's "
                        "CPU allocation is saturated by the GPU campaign.")
    args = p.parse_args()

    import time as _time
    t0 = _time.time()
    uids = _uids(args.dump_root, args.group_id, args.model)
    if args.seeds:
        keep = tuple(f"_seed{s}" for s in args.seeds)
        uids = [u for u in uids if u.endswith(keep)]
    print(f"[score] {len(uids)} runs under {args.group_id}/{args.model}*")

    out = {"model": args.model, "group_id": args.group_id, "runs": {}}
    for uid in uids:
        run = {"id": {}, "ts": {}, "ood": defaultdict(dict)}
        # Temperature scaling is a comparator on CE ALONE, not a per-arm
        # variant. VBLL and IBProbit are reported RAW because their claim is
        # calibration by construction -- temperature-scaling them would measure
        # how much a one-parameter rescaling can rescue a method that says it
        # does not need one. plot_fig1.LINES already encodes this; doing it here
        # too also halves the work on two arms out of three.
        wants_ts = (not args.skip_ts) and "_ce_" in uid
        passes = [(False, "id")] + ([(True, "ts")] if wants_ts else [])
        for scaled, key in passes:
            try:
                rows = curves.run_curve(args.dump_root, args.group_id, uid,
                                        temperature_scaled=scaled,
                                        skip_corrupt=True)
            except Exception as exc:
                print(f"[score] {uid} ({key}): {type(exc).__name__}: {exc}")
                continue
            run[key] = {
                "samples_seen": [r["samples_seen"] for r in rows],
                "acc": [r["acc"] for r in rows],
                "nll": [r["nll"] for r in rows],
                "ece": [r["ece"] for r in rows],
                "brier": [r["brier"] for r in rows],
                "seed": rows[0]["seed"] if rows else None,
                "arm": rows[0]["arm"] if rows else None,
                # A boundary T is a clip, not an optimum. Carried through so a
                # figure can exclude those points rather than average them in.
                "t_clipped": sum(bool(r.get("temperature_clipped")) for r in rows),
            }

        # OOD, on its own (sparser, irregular) grid.
        for rec in evaldump.iter_dumps(args.dump_root, args.group_id, uid,
                                       skip_corrupt=True):
            split = rec.meta.get("split", "")
            if not split.startswith("ood_"):
                continue
            m = metrics.classification_metrics(rec.logits, rec.labels)
            run["ood"][split][rec.meta["samples_seen"]] = {
                k: float(v) for k, v in m.items()}
        run["ood"] = {k: dict(sorted(v.items())) for k, v in run["ood"].items()}
        out["runs"][uid] = run
        n_ood = sum(len(v) for v in run["ood"].values())
        print(f"[score] {uid:46s} id={len(run['id'].get('acc', []))} "
              f"ts={len(run['ts'].get('acc', []))} ood={n_ood} "
              f"[{_time.time() - t0:.0f}s elapsed]", flush=True)

    # `write_dump` is not atomic, so a run killed mid-write leaves a truncated
    # npz behind. Those are skipped rather than fatal (one bad file must not
    # cost the other thousands), but they are NOT silent: the paths ride in the
    # output so a figure built from this JSON can say what is missing, and the
    # count is printed where the launcher's log will show it.
    corrupt = sorted(set(evaldump.CORRUPT_DUMPS))
    out["corrupt_dumps"] = corrupt
    if corrupt:
        print(f"[score] {len(corrupt)} CORRUPT dumps skipped:")
        for path in corrupt:
            print(f"[score]   {path}")

    Path(args.out).write_text(json.dumps(out))
    print(f"[score] wrote {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
