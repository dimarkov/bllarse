#!/usr/bin/env python3
"""Evaluate OOD from training checkpoints, on a gpu of its own.

    scripts/finetune/ood_worker.py --checkpoint-root $SCRATCH/ft_ckpt \\
        --dump-root $SCRATCH/eval_dumps

The reported runs score four OOD sets (98,389 images) on a stride of the dump
grid. Inline that costs the TRAINING gpu about an hour per run and can push a
long dinov3_huge job past a 24 h wall-clock limit. Here it costs the trainer
only a checkpoint write, and OOD can run on a finer stride than the trainer
could ever have afforded -- the eval gpu's wall clock is its own.

WHAT IT GUARANTEES. Each checkpoint names the STEP it was written at, and the
dumps this writes carry that same step. That is not tidiness:
`analysis.curves._id_ood_pair` looks up the test dump and the OOD dump at the
same step and raises when they differ, so a worker that evaluated "the latest
weights" would produce dumps that pair with nothing.

The model is rebuilt through `finetune_jax.build_model` (or
`finetune_vbll.build_model` for the torch arm) -- the same function training
used -- from the args stored in the checkpoint. A worker with its own
copy of that construction would drift silently: a different head init or a
different IBProbit prior deserialises without error and scores a model that
never existed.

Metrics land on the TRAINING run, via start_run(run_id=...) from the checkpoint
metadata, so Fig 4's OOD numbers sit beside Fig 3's ID numbers in one run rather
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


def evaluate_checkpoint_torch(stem: Path, ood_sets, *, dump_root, mlflow_module=None):
    """The VBLL arm. Same contract as the JAX path, different framework.

    Runs in the TORCH venv: jax[cuda] and torch each ship a full CUDA stack and
    fight over the device, which is why the two arms have separate environments
    and therefore separate workers. A worker only consumes slots whose metadata
    names its own framework, so both can poll the same root.
    """
    import torch

    import finetune_vbll as fv
    from bllarse.finetune.registry import (
        DATASET_CONFIGS, EQUIMO_MODELS, IMAGENET_MEAN, IMAGENET_STD)

    meta = checkpoint.read_meta(stem)
    args = _args_namespace(meta["args"])
    step, samples_seen = meta["step"], meta["samples_seen"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = EQUIMO_MODELS[args.model]["embed_dim"]
    num_classes = DATASET_CONFIGS[args.dataset]["num_classes"]

    model, head = fv.build_model(
        args, embed_dim=embed_dim, num_classes=num_classes, device=device)
    state = torch.load(checkpoint.blob_path(stem), map_location=device)
    model.load_state_dict(state["model"])
    head.load_state_dict(state["head"])
    model.eval()
    head.eval()

    scalars = {}
    for name, images_uint8, labels, subset in ood_sets:
        # (logits, score, scalars) -- NOT (logits, scalars, score). The two
        # arms' evaluate_split return different orders.
        logits, score, out = fv.evaluate_split(
            model, head, images_uint8, labels,
            args=args, mean=IMAGENET_MEAN, std=IMAGENET_STD, subset=subset)
        evaldump.write_dump(
            dump_root, args.group_id, args.uid, step, logits, labels,
            arm="vbll", split=f"ood_{name}", logit_kind="softmax_logits",
            seed=args.seed, samples_seen=samples_seen,
            score=score, score_kind="predictive_variance",
        )
        scalars.update({f"ood_{name}_{k}": v for k, v in out.items()})
    _log_to_training_run(meta, scalars, samples_seen, mlflow_module)
    return step, scalars


def _log_to_training_run(meta, scalars, samples_seen, mlflow_module):
    """Attach to the TRAINING run rather than opening a sibling."""
    run_id = meta.get("mlflow_run_id")
    if mlflow_module is None or not run_id or not scalars:
        return
    try:
        with mlflow_module.start_run(run_id=run_id):
            mlflow_module.log_metrics(
                {k: float(v) for k, v in scalars.items()
                 if isinstance(v, (int, float)) and not isinstance(v, bool)},
                step=samples_seen)
    except Exception as exc:
        # Tracking is observability; the dumps are the source of truth. A
        # tracking outage must not cost the evaluation that was just computed.
        print(f"[ood-worker] MLflow log failed ({type(exc).__name__}: {exc}); "
              f"dumps kept.", flush=True)


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

    _log_to_training_run(meta, scalars, samples_seen, mlflow_module)
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
    p.add_argument("--framework", default="jax", choices=["jax", "torch"],
                   help="Consume only slots written by this framework. The two "
                        "arms live in separate venvs (jax[cuda] and torch each "
                        "ship a CUDA stack), so each needs its own worker; both "
                        "can poll the same root.")
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
        slots = [s for s in checkpoint.pending(args.checkpoint_root)
                 if checkpoint.read_meta(s).get("framework") == args.framework]
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
            if args.framework == "jax":
                import finetune_jax as loader
            else:
                import finetune_vbll as loader
            ood_sets = loader.load_ood_sets(ns, args.img_size)

        for stem in slots:
            # Exactly one worker evaluates a slot. Losing the race is normal
            # with several workers polling one root, not an error.
            if not checkpoint.claim(stem):
                continue
            t0 = time.time()
            try:
                evaluate = (evaluate_checkpoint if args.framework == "jax"
                            else evaluate_checkpoint_torch)
                step, scalars = evaluate(
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
