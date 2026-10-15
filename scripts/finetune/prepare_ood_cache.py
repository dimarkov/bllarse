#!/usr/bin/env python3
"""Download and prepare every OOD set into the HF cache, ONCE, serially.

    HF_HOME=$SCRATCH/hf_cache scripts/finetune/prepare_ood_cache.py

Concurrent jobs that call load_dataset on an unprepared cache race to build
it into one directory and tear down each other's `.incomplete` staging dirs,
so the cache has to be prepared once, serially, before they are submitted.

It walks the SAME path the trainer walks (`build_ood_registry()[name](...)`,
then the label column), so a set that prepares here is one `load_ood_sets` can
open offline. It does not decode images to uint8 -- that is the trainer's job
and needs no network.

Idempotent: an already-prepared set resolves from cache in seconds, so this is
safe to re-run and safe to point at a partially-filled cache.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dinov3"))

from bllarse.finetune import imagenet  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--keys", nargs="*", default=list(imagenet.OOD_KEYS),
                   help="OOD sets to prepare (default: the whole suite).")
    p.add_argument("--check-only", action="store_true",
                   help="Resolve OFFLINE and report, downloading nothing. Run "
                        "it before submitting concurrent jobs.")
    args = p.parse_args()

    if args.check_only:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    from precompute_ood_features import build_ood_registry, load_wnid_to_idx

    print(f"[ood-cache] HF_HOME={os.environ.get('HF_HOME', '<unset>')}")

    # The canonical WNID -> index map is read from the checked-in
    # imagenet.WNID_FILE, which needs no network. It is still loaded before any
    # set is reported, so a failure here is caught and answered as "not ready"
    # rather than as a traceback that looks like a different fault.
    try:
        wnid_to_idx = load_wnid_to_idx()
    except Exception as exc:
        raise SystemExit(
            f"[ood-cache] the canonical WNID map could not be read "
            f"({type(exc).__name__}: {exc}). It is the checked-in "
            f"bllarse.finetune.imagenet.WNID_FILE."
        )
    registry = build_ood_registry()

    missing, ok = [], []
    for name in args.keys:
        if name not in registry:
            raise SystemExit(f"[ood-cache] {name!r} is not in the registry; "
                             f"known: {sorted(registry)}")
        t0 = time.time()
        try:
            split, _img_key = registry[name](wnid_to_idx)
            n = len(split["label"])
        except Exception as exc:
            # In --check-only this is the answer, not a crash: the set is not
            # cached and the jobs must not be submitted yet.
            print(f"[ood-cache] {name:18s} UNAVAILABLE ({type(exc).__name__})")
            missing.append(name)
            continue

        expected = imagenet.OOD_SIZES.get(name)
        flag = ""
        if expected is not None and n != expected:
            # Loud, not a warning. A short load scored anyway produces a
            # plausible AUROC over the wrong population, and load_ood_sets
            # would refuse it at job start regardless -- better to learn here.
            flag = f"  MISMATCH: OOD_SIZES says {expected}"
            missing.append(name)
        else:
            ok.append(name)
        print(f"[ood-cache] {name:18s} {n:>7,} images  {time.time() - t0:6.1f}s{flag}")

    print(f"[ood-cache] ready: {len(ok)}/{len(args.keys)}")
    if missing:
        raise SystemExit(
            f"[ood-cache] NOT ready: {', '.join(missing)}. Run without "
            f"--check-only (once, serially) before submitting the campaign."
        )


if __name__ == "__main__":
    main()
