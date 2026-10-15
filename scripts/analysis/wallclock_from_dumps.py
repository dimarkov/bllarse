#!/usr/bin/env python3
"""Wall-clock elapsed at every eval point, per run, for Fig 5.

    scripts/analysis/wallclock_from_dumps.py --dump-root $SCRATCH/eval_dumps \\
        --group-id ftfinal --model dinov3_huge --out huge_wallclock.json

Fig 5 is time to a threshold, not time to the end, so a single run duration is
not enough: the crossing lands mid-run and the cost has to be read off at that
point. Each test dump is written immediately after its evaluation, so the
file's mtime IS the wall clock at that many samples seen -- a measured
timeline, not `duration x samples/total`, which would assume a throughput that
is flat and is not (the first step pays compilation, and evaluation is a fixed
cost per dump rather than a fraction of one).

samples_seen comes from the dump's own meta rather than step x batch, so a run
whose effective batch differs cannot silently mislabel its own x axis. Only the
meta is read; the logits are never decompressed.

The zero point is each run's start time when `--starts` gives one, which is
when the training process began -- including model construction and the first
compile, which differ between the JAX and torch arms and belong in a comparison
of cost. Without it the zero is the FIRST dump, which excludes all of that; the
file records which was used.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402


def _meta(path):
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["meta"]))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump-root", required=True)
    p.add_argument("--group-id", default="ftfinal")
    p.add_argument("--model", default="dinov3_huge")
    p.add_argument("--out", required=True)
    p.add_argument("--starts", default=None,
                   help='JSON {uid: start_epoch_seconds}: when each training '
                        'process began. Without it the zero is the first dump.')
    args = p.parse_args()

    base = Path(args.dump_root) / args.group_id
    uids = sorted(d.name for d in base.iterdir()
                  if d.is_dir() and d.name.startswith(args.model + "_"))
    if args.starts:
        starts = {k: float(v) for k, v in
                  json.loads(Path(args.starts).read_text()).items()}
    else:
        starts = {}

    out = {"model": args.model, "group_id": args.group_id, "runs": {}}
    for uid in uids:
        points = []
        for path in sorted((base / uid).glob("step_*_test.npz")):
            try:
                meta = _meta(path)
            except Exception as exc:
                print(f"[wall] skipping {path}: {type(exc).__name__}: {exc}")
                continue
            points.append([int(meta["samples_seen"]), path.stat().st_mtime])
        if not points:
            continue
        points.sort()
        zero = starts.get(uid)
        source = "given_start"
        if zero is None:
            zero, source = points[0][1], "first_dump"
        out["runs"][uid] = {
            "arm": _meta(sorted((base / uid).glob("step_*_test.npz"))[0])["arm"],
            "seed": int(uid.rsplit("_seed", 1)[1]),
            "zero_source": source,
            "elapsed_s": [[x, round(ts - zero, 3)] for x, ts in points],
            "total_s": round(points[-1][1] - zero, 3),
        }
        print(f"[wall] {uid:46s} {len(points):3d} points "
              f"total {out['runs'][uid]['total_s'] / 3600:.2f} h ({source})")

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"[wall] wrote {args.out} "
          f"({os.path.getsize(args.out) / 1e3:.0f} kB, {len(out['runs'])} runs)")


if __name__ == "__main__":
    main()
