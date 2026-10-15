#!/usr/bin/env python3
"""Score a finished fine-tuning dump tree into one compact JSON.

    scripts/analysis/score_final.py --dump-root $SCRATCH/eval_dumps \\
        --group-id ftfinal --model dinov3_huge --out huge.json

Runs WHERE THE DUMPS ARE. The tree is ~116 GB for one backbone, so the scoring
happens on the cluster and only the distilled curves travel -- a few hundred KB,
enough to plot Fig 3 anywhere.

Emits one row per (arm, seed, step) on the test split, plus -- for the CE arm
ONLY -- the temperature-scaled variant of the same points. CE+TS is reported as
a comparator in its own right, and it is a POST-HOC fit on the CE run's own
dumps rather than a separate training run. T is refit at every step against that
step's val dump, never reused across steps.

TS is NOT applied to VBLL or IBProbit. Their claim is calibration by
construction, so scaling them would measure how much a one-parameter rescaling
can rescue a method that says it does not need one -- and it is a different
transform there in any case, since those arms dump a normalised log-predictive
rather than logits.

OOD is summarised separately and by STEP, since the OOD grid is a stride of the
dump grid and can be an irregular one: when OOD evaluation is delegated to
separate workers, the trainer skips a scheduled OOD checkpoint while too many
are already pending. Reporting the two grids together would silently imply an
OOD point exists wherever an ID point does.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bllarse import detection, evaldump, metrics  # noqa: E402
from bllarse.analysis import curves  # noqa: E402


# The fp16-safe sentinel finetune_jax.evaluate_split writes into the classes an
# OOD subset does not cover. Recovering the subset from the dump rather than
# from a registry keeps the two in step: whatever was masked at eval time is
# exactly what is masked here.
MASK_SENTINEL = -9.0e3


def _subset_of(logits) -> "np.ndarray | None":
    """Columns this dump actually scores, or None when it scores all of them.

    ImageNet-R and -A cover 200 of the 1000 classes and are dumped with the
    other 800 driven to the sentinel; V2 and Sketch span all 1000 and are not
    masked.
    """
    kept = ~(np.asarray(logits) <= MASK_SENTINEL).all(axis=0)
    return None if kept.all() else np.flatnonzero(kept)


def _msp(logits, subset=None):
    """1 - max softmax probability, over `subset` when one is given.

    Restricting by driving the other columns to the sentinel rather than by
    slicing, so the ID reference goes through the arithmetic the OOD dump
    already went through at eval time.
    """
    x = np.asarray(logits, dtype=np.float32)
    if subset is not None:
        keep = np.zeros(x.shape[-1], dtype=bool)
        keep[subset] = True
        x = np.where(keep, x, MASK_SENTINEL)
    return np.asarray(metrics.msp_ood_score(x))


def _seed_of(uid: str) -> int:
    return int(uid.rsplit("_seed", 1)[1])


def _uids(dump_root, group_id, model):
    """Every run for one backbone, ordered SEED-MAJOR.

    Name order would be arm-major -- all five CE seeds, then all five IBProbit,
    then all five VBLL -- so a scorer interrupted or read early would hold five
    seeds of one arm and nothing to compare them against. Seed-major means the
    first three runs are already a complete cross-arm comparison at seed 137,
    which is what a preliminary figure needs. Scoring one backbone reads ~116 GB
    and takes hours; partial output is the normal case, not the exception.
    """
    base = Path(dump_root) / group_id
    names = [p.name for p in base.iterdir()
             if p.is_dir() and p.name.startswith(model + "_")]
    return sorted(names, key=lambda u: (_seed_of(u), u))


def _write(out_path, out, uids) -> None:
    """Serialise the scoring so far, atomically.

    Atomically because this is called after every run while a plotting script
    may be reading the same path -- a half-written JSON is a crash at the other
    end, and the whole reason this function exists is that partial output gets
    consumed.
    """
    out["runs_scored"] = len(out["runs"])
    out["runs_total"] = len(uids)
    out["complete"] = len(out["runs"]) == len(uids)
    out["corrupt_dumps"] = sorted(set(evaldump.CORRUPT_DUMPS))
    path = Path(out_path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(out))
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump-root", required=True)
    p.add_argument("--group-id", default="ftfinal")
    p.add_argument("--model", default="dinov3_huge")
    p.add_argument("--out", required=True)
    p.add_argument("--skip-ts", action="store_true",
                   help="Skip the temperature-scaled pass. It roughly doubles "
                        "the work -- a fresh Newton fit per step against that "
                        "step's val dump, on top of reading the val dumps at "
                        "all -- and the ID curves are readable without it.")
    p.add_argument("--resume", action="store_true",
                   help="Keep the runs already in --out and score only the "
                        "rest. Scoring is hours long and writes after every "
                        "run, so a crash mid-campaign should cost the run it "
                        "died on, not the ones already paid for.")
    p.add_argument("--seeds", nargs="*", type=int, default=None,
                   help="Score only these seeds. The full tree is ~116 GB per "
                        "backbone, and one seed per arm is enough to see the "
                        "curves and validate the pipeline when CPU time is "
                        "scarce.")
    args = p.parse_args()

    import time as _time
    t0 = _time.time()
    uids = _uids(args.dump_root, args.group_id, args.model)
    if args.seeds:
        keep = tuple(f"_seed{s}" for s in args.seeds)
        uids = [u for u in uids if u.endswith(keep)]
    print(f"[score] {len(uids)} runs under {args.group_id}/{args.model}*")

    out = {"model": args.model, "group_id": args.group_id, "runs": {}}
    if args.resume and Path(args.out).exists():
        prior = json.loads(Path(args.out).read_text())
        if prior.get("model") != args.model or \
                prior.get("group_id") != args.group_id:
            raise SystemExit(
                f"[score] --resume refused: {args.out} is "
                f"{prior.get('group_id')}/{prior.get('model')}, not "
                f"{args.group_id}/{args.model}")
        out["runs"] = prior.get("runs", {})
        evaldump.CORRUPT_DUMPS.extend(prior.get("corrupt_dumps") or ())
        print(f"[score] resuming: {len(out['runs'])} runs already scored")

    for uid in uids:
        prior_run = out["runs"].get(uid)
        if prior_run is not None:
            # Keep it only if it carries everything THIS scorer would write. A
            # record from before a field existed is not "already scored" -- it
            # is missing that field, and silently keeping it would leave the
            # figure short of a line, or a selection rule short of its
            # criterion, with no error anywhere.
            wanted_ts = (not args.skip_ts) and "_ce_" in uid
            missing = [f for f in ("ood_ts",) if wanted_ts and not prior_run.get(f)]
            if "val_nll" not in (prior_run.get("id") or {}):
                missing.append("id.val_nll")
            if not missing:
                continue
            print(f"[score] {uid}: re-scoring, prior record lacks "
                  f"{', '.join(missing)}")
        run = {"id": {}, "ts": {}, "ood": defaultdict(dict),
               "ood_ts": defaultdict(dict)}
        temperatures: dict[int, float] = {}
        # Temperature scaling is a comparator on CE ALONE, not a per-arm
        # variant. VBLL and IBProbit are reported RAW because their claim is
        # calibration by construction -- temperature-scaling them would measure
        # how much a one-parameter rescaling can rescue a method that says it
        # does not need one (see curves.TS_ARMS). Skipping it also halves the
        # work on two arms out of three.
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
                # The VAL side of the same rows. Selection is on val and
                # reporting is on test -- disjoint by construction -- so
                # anything that picks an operating point (early stopping, and
                # Fig 5's threshold reference) must read these and never the
                # test curve it is about to report. None where the val dump at
                # that step was unreadable.
                "val_nll": [r.get("val_nll") for r in rows],
                "val_acc": [r.get("val_acc") for r in rows],
                "val_brier": [r.get("val_brier") for r in rows],
                "seed": rows[0]["seed"] if rows else None,
                "arm": rows[0]["arm"] if rows else None,
                # A boundary T is a clip, not an optimum. Carried through so a
                # figure can exclude those points rather than average them in.
                "t_clipped": sum(bool(r.get("temperature_clipped")) for r in rows),
            }
            if key == "ts":
                # T at each eval point, keyed by samples_seen, so the OOD pass
                # below can apply THAT step's temperature to that step's
                # shifted logits.
                temperatures = {r["samples_seen"]: r["temperature"]
                                for r in rows}

        # DETECTION needs both sides of a pair at the same step, so collect
        # the in-distribution MSP first. Scores, not logits: 40k floats a step
        # against 40k x 1000 fp16, which would not fit for a whole trajectory.
        #
        # A masked column needs its OWN id reference -- the same 200 classes,
        # and only the id samples whose label falls in them. Comparing a
        # 200-class max against a 1000-class max would measure the class count,
        # not the shift: renormalising over fewer classes raises the max
        # mechanically, and every R/A point would look confidently in
        # distribution.
        # One dump per OOD split, by direct path, NOT a sweep: `iter_dumps`
        # decompresses every archive it yields, so recovering four subsets
        # through it would cost a third pass over the whole tree -- on a run
        # with 200 dumps that measured 26 minutes against 7.
        subsets = {}
        for path in sorted(evaldump.dump_dir(
                args.dump_root, args.group_id, uid).glob("step_*_ood_*.npz")):
            split = path.stem.split("_", 2)[2]
            if split in subsets:
                continue
            try:
                subsets[split] = _subset_of(evaldump.load_dump(path).logits)
            except Exception as exc:                      # a truncated npz
                print(f"[score] {uid}: {path.name}: {type(exc).__name__}: {exc}")
        distinct = {None: None}
        for sub in subsets.values():
            if sub is not None:
                distinct[tuple(sub)] = sub

        # Raw and temperature-scaled, because the CE+TS series is drawn from
        # `ood_ts` and would otherwise have no detection row at all. T is the
        # one fit on the in-distribution val split at this step -- the same
        # scalar Fig 3 uses -- applied to BOTH sides of the pair, never refit
        # on the shifted set.
        id_scores = defaultdict(dict)      # {subset_key: {samples: scores}}
        id_scores_ts = defaultdict(dict)
        for rec in evaldump.iter_dumps(args.dump_root, args.group_id, uid,
                                       skip_corrupt=True):
            if rec.meta.get("split") != "test":
                continue
            seen_id = rec.meta["samples_seen"]
            labels = np.asarray(rec.labels)
            temp = temperatures.get(seen_id)
            for key, sub in distinct.items():
                keep = slice(None) if sub is None else np.isin(labels, sub)
                logits = rec.logits[keep]
                id_scores[key][seen_id] = _msp(logits, sub)
                if temp is not None:
                    id_scores_ts[key][seen_id] = _msp(
                        metrics.apply_temperature(logits, temp), sub)

        # OOD, on its own (sparser, irregular) grid.
        for rec in evaldump.iter_dumps(args.dump_root, args.group_id, uid,
                                       skip_corrupt=True):
            split = rec.meta.get("split", "")
            if not split.startswith("ood_"):
                continue
            seen = rec.meta["samples_seen"]
            m = metrics.classification_metrics(rec.logits, rec.labels)
            row = {k: float(v) for k, v in m.items()}

            # MSP detection against the id split at the SAME step. MSP only,
            # matching the probing figure: it reads the normalised predictive,
            # so it is the one score all three arms produce identically, and
            # IBProbit's moderation enters it through the class-independent
            # rescaling rather than being discarded.
            sub = subsets.get(split)
            key = None if sub is None else tuple(sub)
            ref = id_scores.get(key, {}).get(seen)
            if ref is not None and len(ref):
                det = detection.detection_metrics(ref, _msp(rec.logits, sub))
                row.update({k: float(v) for k, v in det.items()})
            run["ood"][split][seen] = row

            # The SAME T that Fig 3 uses at this step, fit on the
            # in-distribution val split and applied unchanged to the shifted
            # logits. No OOD label is read to obtain it: refitting T here would
            # be test-time adaptation, and the symmetric move would be to run
            # IBProbit's CAVI update or VBLL's posterior update on the shifted
            # set. Every arm is fit in-distribution and evaluated under shift.
            temperature = temperatures.get(seen)
            if temperature is None:
                continue
            scaled_logits = metrics.apply_temperature(rec.logits, temperature)
            m_ts = metrics.classification_metrics(scaled_logits, rec.labels)
            row_ts = {k: float(v) for k, v in m_ts.items()}
            ref_ts = id_scores_ts.get(key, {}).get(seen)
            if ref_ts is not None and len(ref_ts):
                det_ts = detection.detection_metrics(
                    ref_ts, _msp(scaled_logits, sub))
                row_ts.update({k: float(v) for k, v in det_ts.items()})
            run["ood_ts"][split][seen] = row_ts

        for field in ("ood", "ood_ts"):
            run[field] = {k: dict(sorted(v.items()))
                          for k, v in run[field].items()}
        out["runs"][uid] = run
        n_ood = sum(len(v) for v in run["ood"].values())
        n_ood_ts = sum(len(v) for v in run["ood_ts"].values())
        print(f"[score] {uid:46s} id={len(run['id'].get('acc', []))} "
              f"ts={len(run['ts'].get('acc', []))} ood={n_ood}"
              f"{f'+{n_ood_ts}ts' if n_ood_ts else ''} "
              f"[{_time.time() - t0:.0f}s elapsed]", flush=True)
        # Written after EVERY run, not once at the end. A full backbone takes
        # hours, and a plot wants the seed-137 comparison as soon as it exists
        # rather than after seed 141. `runs_scored`/`runs_total` let a figure
        # say out loud that it is drawn from a partial scoring.
        _write(args.out, out, uids)

    # `write_dump` is not atomic, so a run killed mid-write leaves a truncated
    # npz behind. Those are skipped rather than fatal (one bad file must not
    # cost the other thousands), but they are NOT silent: the paths ride in the
    # output so a figure built from this JSON can say what is missing, and the
    # count is printed where the launcher's log will show it.
    corrupt = sorted(set(evaldump.CORRUPT_DUMPS))
    if corrupt:
        print(f"[score] {len(corrupt)} CORRUPT dumps skipped:")
        for path in corrupt:
            print(f"[score]   {path}")

    _write(args.out, out, uids)
    print(f"[score] wrote {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
