"""Full-network fine-tuning of DINOv3 on ImageNet-1k -- JAX arms.

One arm per invocation (`--arm ce|ibprobit`), writing eval dumps on a log-spaced
schedule so temperature scaling, Brier and OOD detection are all computed post
hoc by the same code that scores the probing figures.

This is NOT scripts/alternatives/full_finetune_comparison.py, which stays as the
IVON/EVON benchmarking harness with inline metrics. The two share their jitted
kernels via bllarse.finetune.jax_steps.
"""

import argparse
import gc
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.secrets", override=False)

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from jax import random as jr  # noqa: E402

from bllarse import evaldump, metrics  # noqa: E402
from bllarse.finetune import imagenet, jax_steps, protocol  # noqa: E402
from bllarse.losses import CrossEntropy  # noqa: E402

# Fine-tuning resolution. DINOV3_DATASET_IMG_SIZES maps imagenet1k -> 512, but
# that table exists for FROZEN FEATURE EXTRACTION; fine-tuning at 512 is ~5x the
# cost for no benefit, so inheriting it must be impossible by accident.
FINETUNE_IMG_SIZE = 224

# fp16-safe mask sentinel for evaluate_and_dump's `subset` masking.
# evaldump.write_dump stores logits as float16 (max finite magnitude 65504),
# and float16(-1e9) OVERFLOWS to -inf: entropy_ood_score and fit_temperature
# on such a row silently return NaN while msp_ood_score still looks healthy,
# so a downstream figure would show a working MSP column next to a silently
# NaN entropy column. -1e4 is ~340 sigma below any real logit and comfortably
# inside float16's -65504 floor, so it round-trips as a large but finite
# number. full_finetune_comparison.MASK_NEG = -1e9 is safe there ONLY because
# that mask is applied at fp32 and used purely in-memory (softmax/argmax) --
# it never round-trips through this fp16 dump path.
MASK_NEG = -1e4


def check_img_size(args) -> None:
    """Refuse a non-224 resolution unless it was asked for explicitly."""
    if args.img_size != FINETUNE_IMG_SIZE and not args.allow_non_224:
        raise SystemExit(
            f"[finetune] --img-size {args.img_size} refused. Fine-tuning runs at "
            f"{FINETUNE_IMG_SIZE}. DINOV3_DATASET_IMG_SIZES maps imagenet1k -> 512 "
            f"for frozen feature extraction, and inheriting that here costs ~5x "
            f"for no benefit. Pass --allow-non-224 if this is deliberate."
        )


def check_arm_flags(args) -> None:
    """Reject flags that the chosen arm cannot act on.

    IBProbit has no head learning rate: run_arm initialises the optimizer on the
    backbone alone and the head is assimilated in closed form by
    `loss_fn.update(...)`. Accepting --head-lr there would present a tuned axis
    that silently does nothing.
    """
    if args.arm == "ibprobit" and args.head_lr is not None:
        raise SystemExit(
            "[finetune] --head-lr is meaningless for --arm ibprobit: the head is "
            "updated in closed form by CAVI, not by the optimizer. Its head-side "
            "knob is --num-update-iters."
        )


def resolve_plan(args, micro_batch: int) -> dict:
    """Turn (target batch, micro batch, epoch cap) into the concrete schedule.

    `steps` is derived from the ACHIEVED effective batch, not the requested one:
    Fig 5's x-axis is samples seen, so a plan built on a batch size the run does
    not use would mislabel every point on it.
    """
    micro, num_accum, effective = protocol.accum_plan(args.batch_size, micro_batch)
    steps = protocol.steps_for_epochs(
        imagenet.IMAGENET_N_TRAIN, args.epochs, effective
    )
    schedule = protocol.dump_steps(steps, args.n_dumps)
    return {
        "micro": micro,
        "num_accum": num_accum,
        "effective_batch": effective,
        "steps": steps,
        "dump_steps": schedule,
        "ood_steps": protocol.ood_dump_steps(schedule, args.ood_every_k_dumps),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=["ce", "ibprobit"])
    p.add_argument("--model", default="dinov3_large")
    p.add_argument("--dataset", default="imagenet1k")
    p.add_argument("--img-size", type=int, default=FINETUNE_IMG_SIZE)
    p.add_argument(
        "--allow-non-224", action="store_true",
        help="Permit a resolution other than 224. See check_img_size.",
    )

    p.add_argument(
        "--epochs", type=float, default=4.0,
        help="Epoch cap. Float so a tuning trial can ask for a fraction of a "
             "pass; steps are derived from this and the effective batch.",
    )
    p.add_argument(
        "--batch-size", type=int, required=True,
        help="TARGET (effective) batch, reached by gradient accumulation. This "
             "is the tuned quantity; --micro-batch-size is a memory artifact.",
    )
    p.add_argument(
        "--micro-batch-size", type=int, default=None,
        help="Per-forward batch. Omit to auto-probe the largest that fits.",
    )
    p.add_argument("--batch-size-init", type=int, default=32)
    p.add_argument("--batch-size-max", type=int, default=8192)

    p.add_argument("--lr", type=float, default=1e-4, help="Trunk learning rate")
    p.add_argument(
        "--head-lr", type=float, default=None,
        help="Head learning rate (CE arm only; IBProbit has no head optimizer).",
    )
    p.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="Trunk weight decay. Head weight decay is pinned to 0 on every arm.",
    )
    p.add_argument("--label-smooth", type=float, default=0.0)
    p.add_argument("--num-update-iters", type=int, default=128,
                   help="CAVI iterations per macro batch (IBProbit arm).")
    p.add_argument("--bf16", action="store_true")

    # --- IBProbit head knobs (mirror scripts/alternatives/ood_evaluation.py) ---
    # Copied VERBATIM (flag names, types, defaults, help text) from
    # full_finetune_comparison.build_argparser: probe_micro_batch_size ->
    # build_ibprobit reads all ten of these regardless of --arm, since the
    # prober always builds an IBProbit head to find the heavier micro-step's
    # memory ceiling. Do not invent defaults here -- copying keeps this script
    # and full_finetune_comparison.py at one operating point.
    p.add_argument("--px", action="store_true",
                   help="parameter-expanded CAVI scan (collapses the slow scale mode; far fewer iters)")
    p.add_argument("--fix-q", action="store_true",
                   help="fix q at its init (disable the in-update self-tuning q gradient)")
    p.add_argument("--hyper-lr", type=float, default=0.1,
                   help="in-update prequential SGD step size for log_q")
    p.add_argument("--anchor-weight", type=float, default=1e-3,
                   help="Inverse-Gamma anchor strength on q (mode at its init); 0 = off")
    p.add_argument("--q-init", type=float, default=None,
                   help="process-noise q init / fixed value (default 1/D, the LayerNorm-scale floor)")
    p.add_argument("--sigma-z2-init", type=float, default=1.0,
                   help="fixed sigma_z^2 (the moderation floor; not a learnable parameter)")
    p.add_argument("--bmr-gate", action="store_true",
                   help="enable the per-data-point BMR smoothing gate in the CAVI update")
    p.add_argument("--bmr-warmup", type=int, default=0,
                   help="skip the BMR gate for the first N update steps (the mu=0 seed gate is useless)")
    p.add_argument("--learn-eps", action="store_true",
                   help="self-tune the eps-contamination weight in-update via conjugate Beta(1, 1/eps_init) updating")
    p.add_argument("--eps-init", type=float, default=0.0,
                   help="with --learn-eps sets the Beta(1, 1/eps_init) prior (mean ~ eps_init); else the fixed eps (>0 activates the mixture). Forgetting rho = 1 - B/N set automatically.")

    p.add_argument("--n-dumps", type=int, default=protocol.DEFAULT_N_DUMPS)
    p.add_argument(
        "--ood-every-k-dumps", type=int, default=2,
        help="Write OOD dumps on every k-th scheduled dump. 0 disables OOD "
             "entirely, which is what tuning trials use.",
    )
    p.add_argument("--eval-chunk", type=int, default=256)
    p.add_argument("--val-fraction", type=float, default=imagenet.DEFAULT_VAL_FRACTION)

    p.add_argument("--dump-root",
                   default=os.environ.get("BLLARSE_EVAL_DUMP_ROOT", "eval_dumps"))
    p.add_argument("--group-id", default="finetune")
    p.add_argument("--uid", default=None)
    p.add_argument("--split-cache-dir", default=".cache/splits")
    p.add_argument("--seed", type=int, default=137)
    p.add_argument("--device", default="gpu")

    p.add_argument("--enable-mlflow", action="store_true")
    p.add_argument("--mlflow-experiment", default="bllarse-finetune")

    p.add_argument("--optimizer", default="adamw", choices=["adamw"],
                   help="Present because the shared micro-batch prober reads "
                        "it. IVON/EVON live in full_finetune_comparison.py.")
    p.add_argument("--estimator", default="sampling", choices=["sampling"],
                   help="Present for the shared prober; see --optimizer.")
    return p


def stream_micro_batches(train_split, ds_config, micro, img_size, mean, std,
                         total_micro, seed, num_workers=16):
    """Yield exactly `total_micro` micro-batches, reshuffling between passes.

    The permutation comes from protocol.shuffled_indices (numpy), NOT jax.random,
    so the PyTorch arm reproduces this exact order. Short trailing batches are
    skipped to keep the jitted shape constant.
    """
    n = len(train_split)
    labels_all = np.asarray(train_split[ds_config["label_key"]])
    emitted, epoch = 0, 0
    while emitted < total_micro:
        order = protocol.shuffled_indices(n, seed, epoch)
        for start in range(0, n - micro + 1, micro):
            batch_idx = order[start:start + micro]
            images = imagenet.load_images_uint8(
                train_split, ds_config["img_key"], img_size, batch_idx,
                num_workers=num_workers,
            )
            yield (jnp.asarray(imagenet.normalize_chunk(images, mean, std)),
                   jnp.asarray(labels_all[batch_idx]))
            emitted += 1
            if emitted >= total_micro:
                return
        epoch += 1


def evaluate_and_dump(arm, backbone, head_or_loss, images_uint8, labels, *,
                      split, step, samples_seen, args, mean, std, key,
                      subset=None):
    """Forward one preloaded eval set in chunks and write its dump.

    Returns the scalar metrics so the caller can drive `on_dump` and MLflow.
    Normalization happens per chunk (imagenet.normalize_chunk), so the float32
    expansion only ever covers `--eval-chunk` images at a time.

    `subset`, if given, restricts scoring to those class indices: every other
    logit is driven to MASK_NEG before the dump is written and before metrics
    are computed, so both share the same masked view (OOD evaluation over
    ImageNet-R/-A, which cover only 200 of the 1000 classes).
    """
    inference_bb = eqx.nn.inference_mode(backbone, True)
    chunks = []
    for start in range(0, len(labels), args.eval_chunk):
        raw = images_uint8[start:start + args.eval_chunk]
        imgs = jnp.asarray(imagenet.normalize_chunk(raw, mean, std))
        labs = jnp.asarray(labels[start:start + args.eval_chunk])
        if arm == "ibprobit":
            logits = eval_logits_ibprobit(inference_bb, head_or_loss, imgs, labs,
                                            key, bf16=args.bf16)
        else:
            logits = eval_logits_ce(inference_bb, head_or_loss, imgs, key,
                                       bf16=args.bf16)
        chunks.append(np.asarray(logits))
    logits = np.concatenate(chunks, axis=0)

    if subset is not None:
        keep = np.zeros(logits.shape[-1], dtype=bool)
        keep[np.asarray(subset)] = True
        logits = np.where(keep, logits, MASK_NEG)

    evaldump.write_dump(
        args.dump_root, args.group_id, args.uid, step,
        logits, labels,
        arm=arm, split=split,
        logit_kind="moderated_score" if arm == "ibprobit" else "softmax_logits",
        seed=args.seed, samples_seen=samples_seen,
    )
    return metrics.classification_metrics(logits, labels)


@eqx.filter_jit
def eval_logits_ce(backbone, head, images, key, bf16=False):
    features = jax_steps.forward_features(backbone, images, key, bf16=bf16)
    return jax.vmap(head)(features)


@eqx.filter_jit
def eval_logits_ibprobit(backbone, loss_fn, images, labels, key, bf16=False):
    features = jax_steps.forward_features(backbone, images, key, bf16=bf16)
    _, logits = loss_fn(features, labels, with_logits=True, loss_type=2)
    return logits


def main(args, on_dump=None) -> None:
    """Fine-tune one arm: gradient accumulation to the target batch, eval dumps
    of both val and test on a log-spaced step schedule.

    `on_dump(dump_index, step, scalars)` matches the hook
    bllarse.tuning.objective.evaluate passes, so the Phase 2 tuning campaign can
    drive this script without a wrapper. Any exception `on_dump` raises (e.g.
    objective.EarlyConvergence) propagates out of this loop uncaught -- the
    caller is responsible for catching it.
    """
    check_img_size(args)
    check_arm_flags(args)
    if args.uid is None:
        args.uid = f"{args.model}_{args.arm}_seed{args.seed}"

    # Heavy, GPU-oriented imports stay inside main(): this module is imported
    # by the CPU-only test suite (and by --help), and equimo/datasets pull in
    # a lot of machinery neither needs.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "alternatives"))
    from vit_classification import (
        DATASET_CONFIGS, EQUIMO_MODELS, IMAGENET_MEAN, IMAGENET_STD,
        get_pretrained_backbone,
    )
    from datasets import load_dataset

    ds_config = DATASET_CONFIGS[args.dataset]
    embed_dim = EQUIMO_MODELS[args.model]["embed_dim"]
    num_classes = ds_config["num_classes"]
    mean, std = IMAGENET_MEAN, IMAGENET_STD

    key = jr.PRNGKey(args.seed)
    key, model_key, init_key, fwd_key = jr.split(key, 4)

    ds = load_dataset(ds_config["hf_path"], name=ds_config.get("hf_config"))
    train_split = ds["train"]
    eval_split = ds[ds_config["test_split"]]

    backbone = eqx.nn.inference_mode(get_pretrained_backbone(args.model, model_key), True)

    micro = args.micro_batch_size
    if micro is None:
        from full_finetune_comparison import probe_micro_batch_size

        key, probe_key = jr.split(key)
        print("[finetune] probing maximal micro-batch size...")
        micro = probe_micro_batch_size(
            backbone, args, embed_dim, num_classes, args.img_size,
            args.batch_size_init, args.batch_size_max, probe_key,
        )
        print(f"[finetune] probed micro-batch size: {micro}")
    plan = resolve_plan(args, micro)
    print(f"[finetune] arm={args.arm} effective_batch={plan['effective_batch']} "
          f"= {plan['num_accum']} x micro {plan['micro']}, steps={plan['steps']}")

    # Pinned val/test partition over the official validation split, preloaded
    # once as uint8 and normalized per chunk at eval time.
    val_idx, test_idx = imagenet.pinned_eval_split(
        args.split_cache_dir, n=len(eval_split), val_fraction=args.val_fraction
    )
    eval_labels_all = np.asarray(eval_split[ds_config["label_key"]])
    eval_sets = {}
    for name, idx in (("val", val_idx), ("test", test_idx)):
        eval_sets[name] = (
            imagenet.load_images_uint8(eval_split, ds_config["img_key"],
                                       args.img_size, idx),
            eval_labels_all[idx],
        )
        print(f"[finetune] preloaded {name}: {len(idx)} images")

    if args.arm == "ce":
        loss_fn = CrossEntropy(args.label_smooth, num_classes)
        head = eqx.nn.Linear(embed_dim, num_classes, key=init_key)
        head_lr = args.head_lr if args.head_lr is not None else args.lr
        param_labels = (
            jax.tree.map(lambda _: "backbone", eqx.filter(backbone, eqx.is_array)),
            jax.tree.map(lambda _: "head", eqx.filter(head, eqx.is_array)),
        )
        optimizer = optax.multi_transform(
            {
                "backbone": optax.adamw(args.lr, weight_decay=args.weight_decay),
                # Head weight decay is 0 on every arm (design D6): VBLL's KL is
                # its head regulariser and IBProbit runs no optimizer on the
                # head at all, so a nonzero value here would be an asymmetry.
                "head": optax.adamw(head_lr, weight_decay=0.0),
            },
            param_labels,
        )
        opt_state = optimizer.init(eqx.filter((backbone, head), eqx.is_array))
    else:
        from bllarse.losses import IBProbit
        loss_fn = IBProbit(embed_dim, num_classes, key=init_key)
        head = None
        optimizer = optax.adamw(args.lr, weight_decay=args.weight_decay)
        opt_state = optimizer.init(eqx.filter(backbone, eqx.is_array))

    dump_set = set(int(s) for s in plan["dump_steps"])
    micro_iter = stream_micro_batches(
        train_split, ds_config, plan["micro"], args.img_size, mean, std,
        plan["steps"] * plan["num_accum"], args.seed,
    )

    dump_index = 0
    for step in range(1, plan["steps"] + 1):
        train_key = jr.fold_in(fwd_key, step)
        grads_acc, all_fts, all_labels = None, [], []

        for _ in range(plan["num_accum"]):
            images, labels = next(micro_iter)
            if args.arm == "ibprobit":
                if step == 1:
                    fts, _ = jax_steps.ibprobit_micro_forward_nograd(
                        backbone, loss_fn, images, labels, train_key, bf16=args.bf16)
                else:
                    grads, fts, _ = jax_steps.ibprobit_micro_forward(
                        backbone, loss_fn, images, labels, train_key, bf16=args.bf16)
                    grads_acc = grads if grads_acc is None else jax_steps._tree_add(
                        grads, grads_acc)
                all_fts.append(fts)
                all_labels.append(labels)
            else:
                grads, _ = jax_steps.ce_micro_grad(
                    (backbone, head), loss_fn, images, labels, train_key,
                    bf16=args.bf16)
                grads_acc = grads if grads_acc is None else jax_steps._tree_add(
                    grads, grads_acc)

        if args.arm == "ibprobit":
            if step > 1:
                backbone, opt_state = jax_steps.apply_grads(
                    backbone, grads_acc, opt_state, optimizer, plan["num_accum"])
            loss_fn = loss_fn.update(
                jnp.concatenate(all_fts), jnp.concatenate(all_labels),
                num_iters=args.num_update_iters)
            head_or_loss = loss_fn
        else:
            (backbone, head), opt_state = jax_steps.apply_grads(
                (backbone, head), grads_acc, opt_state, optimizer, plan["num_accum"])
            head_or_loss = head

        if step in dump_set:
            samples_seen = step * plan["effective_batch"]
            scalars = {}
            for name in ("val", "test"):
                images_uint8, labels = eval_sets[name]
                out = evaluate_and_dump(
                    args.arm, backbone, head_or_loss, images_uint8, labels,
                    split=name, step=step, samples_seen=samples_seen,
                    args=args, mean=mean, std=std, key=fwd_key,
                )
                scalars.update({f"{name}_{k}": v for k, v in out.items()})
            print(f"[finetune] step {step}/{plan['steps']} " + " ".join(
                f"{k}={v:.4f}" for k, v in sorted(scalars.items())))
            if on_dump is not None:
                on_dump(dump_index, step, scalars)
            dump_index += 1
            # gc.collect() only -- NOT jax.clear_caches() here. Dropping the
            # compiled fwd+bwd graph plus the eval-chunk shapes on every dump
            # (up to --n-dumps times a run) forces a full cold XLA recompile
            # each time: this repo has no persistent compilation cache
            # configured, so at --n-dumps 24 that is an estimated 20-70
            # minutes of a 4-hour run. full_finetune_comparison.py only clears
            # once, at the very end of a run, for the same reason -- see below.
            gc.collect()

    jax.clear_caches()
    print(f"[finetune] done: {dump_index} dumps under "
          f"{args.dump_root}/{args.group_id}/{args.uid}")


if __name__ == "__main__":
    from jax import config as jax_config

    parsed = build_argparser().parse_args()
    jax_config.update("jax_platform_name", parsed.device)
    main(parsed)
