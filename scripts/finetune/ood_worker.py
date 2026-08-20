#!/usr/bin/env python3
"""Evaluate OOD from training checkpoints, on a gpu of its own.

    scripts/finetune/ood_worker.py --checkpoint-root /ptmp/$USER/ft_ckpt \\
        --dump-root /ptmp/$USER/eval_dumps

The reported campaign scores four OOD sets (98,389 images) on a stride of the
dump grid. Inline that costs the TRAINING gpu about an hour per run and pushes a
dinov3_huge 4-epoch job past its 24 h wall. Here it costs the trainer only a
checkpoint write, and OOD can run on a finer stride than the trainer could ever
have afforded -- the eval gpu's wall clock is its own.

WHAT IT GUARANTEES. Each checkpoint names the STEP it was written at, and the
dumps this writes carry that same step. That is not tidiness:
`analysis.curves._id_ood_pair` looks up the test dump and the OOD dump at the
same step and raises when they differ, so a worker that evaluated "the latest
weights" would produce dumps that pair with nothing.

The model is rebuilt through `finetune_jax.build_model` -- the same function
training used -- from the args stored in the checkpoint. A worker with its own
copy of that construction would drift silently: a different head init or a
different IBProbit prior deserialises without error and scores a model that
never existed.

Metrics land on the TRAINING run, via start_run(run_id=...) from the checkpoint
metadata, so Fig 6's OOD numbers sit beside Fig 5's ID numbers in one run rather
than in a sibling nobody joins.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bllarse import evaldump, metrics  # noqa: E402
from bllarse.finetune import checkpoint, imagenet  # noqa: E402


def _args_namespace(payload: dict) -> argparse.Namespace:
    """The training args, as the trainer saw them."""
    return argparse.Namespace(**payload)


def evaluate_checkpoint(stem: Path, ood_sets, *, dump_root, mlflow_module=None):
    """Score one checkpoint against every OOD set and write its dumps."""
    import equinox as eqx
    import jax.random as jr

    import finetune_jax as fj
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dinov3"))
    from vit_classification import (
        DATASET_CONFIGS, EQUIMO_MODELS, IMAGENET_MEAN, IMAGENET_STD,
        get_pretrained_backbone,
    )

    meta = checkpoint.read_meta(stem)
    args = _args_namespace(meta["args"])
    step, samples_seen = meta["step"], meta["samples_seen"]

    ds_config = DATASET_CONFIGS[args.dataset]
    embed_dim = EQUIMO_MODELS[args.model]["embed_dim"]
    num_classes = ds_config["num_classes"]

    # The SAME keys training used, so the template pytree has the same shapes
    # and the same static fields as the leaves being loaded into it.
    key = jr.PRNGKey(args.seed)
    _key, model_key, init_key, fwd_key = jr.split(key, 4)
    backbone = eqx.nn.inference_mode(
        get_pretrained_backbone(args.model, model_key), True)
    loss_fn, head = fj.build_model(
        args, embed_dim=embed_dim, num_classes=num_classes,
        init_key=init_key, effective_batch=meta.get("effective_batch")
        or args.batch_size)
    template = (backbone, head if args.arm == "ce" else loss_fn)
    backbone, head_or_loss = eqx.tree_deserialise_leaves(
        checkpoint.blob_path(stem), template)

    scalars = {}
    for name, images_uint8, labels, subset in ood_sets:
        logits, out = fj.evaluate_split(
            args.arm, backbone, head_or_loss, images_uint8, labels,
            args=args, mean=IMAGENET_MEAN, std=IMAGENET_STD, key=fwd_key,
            subset=subset,
        )
        evaldump.write_dump(
            dump_root, args.group_id, args.uid, step, logits, labels,
            arm=args.arm, split=f"ood_{name}", logit_kind="softmax_logits",
            seed=args.seed, samples_seen=samples_seen,
        )
        scalars.update({f"ood_{name}_{k}": v for k, v in out.items()})

    run_id = meta.get("mlflow_run_id")
    if mlflow_module is not None and run_id and scalars:
        # Attach to the TRAINING run. nested=False and an explicit run_id, so
        # this is the same run rather than a child of it.
        with mlflow_module.start_run(run_id=run_id):
            mlflow_module.log_metrics(
                {k: float(v) for k, v in scalars.items()
                 if isinstance(v, (int, float))}, step=samples_seen)
    return step, scalars


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-root", required=True)
    p.add_argument("--dump-root", required=True)
    p.add_argument("--poll-seconds", type=float, default=30.0)
    p.add_argument("--idle-exit-seconds", type=float, default=1800.0,
                   help="Exit after this long with nothing to do. The trainers "
                        "have finished, or something is wrong -- either way a "
                        "gpu should not idle indefinitely.")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--once", action="store_true",
                   help="Drain what is pending and exit; for smoke runs.")
    args = p.parse_args()

    mlflow_module = None
    if not args.no_mlflow:
        try:
            from bllarse.mlflow_utils import load_mlflow_env_defaults
            load_mlflow_env_defaults()
            import mlflow as mlflow_module  # noqa: F811
        except Exception as exc:
            print(f"[ood-worker] MLflow unavailable ({type(exc).__name__}); "
                  f"dumps still written, metrics not logged.", flush=True)

    # Loaded ONCE and held: 98,389 images is ~14.8 GB of uint8 host arrays and
    # re-reading it per checkpoint would dominate the worker's runtime.
    ood_sets = None
    idle_since = time.time()

    while True:
        slots = checkpoint.pending(args.checkpoint_root)
        if not slots:
            if args.once:
                print("[ood-worker] nothing pending; exiting.", flush=True)
                return
            if time.time() - idle_since > args.idle_exit_seconds:
                print(f"[ood-worker] idle {args.idle_exit_seconds:.0f}s; "
                      f"exiting.", flush=True)
                return
            time.sleep(args.poll_seconds)
            continue

        idle_since = time.time()
        if ood_sets is None:
            ns = argparse.Namespace(img_size=args.img_size)
            print(f"[ood-worker] preloading {sum(imagenet.OOD_SIZES.values()):,} "
                  f"OOD images", flush=True)
            import finetune_jax as fj
            ood_sets = fj.load_ood_sets(ns, args.img_size)

        for stem in slots:
            t0 = time.time()
            try:
                step, scalars = evaluate_checkpoint(
                    stem, ood_sets, dump_root=args.dump_root,
                    mlflow_module=mlflow_module)
            except Exception as exc:
                # Consume it anyway. A slot that fails forever blocks the queue
                # and, through backpressure, stalls the trainer's OOD schedule
                # for the rest of the run.
                print(f"[ood-worker] FAILED on {stem.name} "
                      f"({type(exc).__name__}: {exc}); dropping the slot.",
                      flush=True)
                checkpoint.consume(stem)
                continue
            checkpoint.consume(stem)
            print(f"[ood-worker] step {step}: {len(scalars)} metrics in "
                  f"{time.time() - t0:.0f}s", flush=True)

        if args.once:
            return


if __name__ == "__main__":
    main()
