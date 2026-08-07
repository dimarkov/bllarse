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
import time
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

# full_finetune_comparison.py's --head-lr default: the fresh head needs ~100x
# the trunk rate -- AdamW moves weights by ~lr/step, and the head's random
# init scale (~0.015) would take thousands of 1e-4 steps to overwrite
# otherwise. Kept as a fallback CONSTANT rather than argparse's `default=` so
# check_arm_flags can still use `args.head_lr is not None` to tell "user
# explicitly passed --head-lr" apart from "omitted" -- an ibprobit run that
# explicitly passes even this same value must still be rejected, since its
# head has no optimizer to receive it.
DEFAULT_HEAD_LR = 1e-2


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


def check_micro_batch_required(args) -> None:
    """The auto-probe is not wired for --arm ibprobit.

    Omitting --micro-batch-size makes main() auto-probe the largest micro batch
    that fits. Doing that for the ibprobit arm needs a real IBProbit head, and
    this script has no ibprobit prober -- only probe_micro_batch_size_ce, which
    never touches IBProbit at all. (full_finetune_comparison.py does have one,
    probe_micro_batch_size; wiring an equivalent here is unfinished work, not a
    blocked dependency: the constructor knobs it needs landed with the
    process-noise merge.) Until then --arm ibprobit must pass
    --micro-batch-size explicitly.
    """
    if args.arm == "ibprobit" and args.micro_batch_size is None:
        raise SystemExit(
            "[finetune] --micro-batch-size is required for --arm ibprobit: this "
            "script has no auto-probe for that arm (only the ce one). Pass "
            "--micro-batch-size explicitly."
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
        help="Head learning rate (CE arm only; IBProbit has no head "
             f"optimizer). Defaults to DEFAULT_HEAD_LR={DEFAULT_HEAD_LR:g} "
             "when omitted -- 100x the trunk --lr, matching "
             "full_finetune_comparison.py: AdamW moves weights by ~lr/step, "
             "and the head's random init scale (~0.015) would take "
             "thousands of 1e-4 steps to overwrite otherwise.",
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
    p.add_argument("--no-learn-q", action="store_true",
                   help="pin Qhat at its q0 seed (skip the Inverse-Wishart recursion on Q)")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="process-noise scale: q0 = alpha^2 / effective_batch. Default 0.1; "
                        "at 1e-3 the recursion is measurably inert (design spec sec 4.8)")
    p.add_argument("--q-init", type=float, default=None,
                   help="override q0 directly (bypasses alpha^2 / B; 1/D is the pre-Wishart value)")
    p.add_argument("--kappa", type=float, default=None,
                   help="Inverse-Wishart prior pseudo-count (default: 0.5, i.e. half a pseudo-observation against the K the likelihood adds per step)")
    p.add_argument("--warmup", type=int, default=1,
                   help="hold the contamination gate at 1 for the first N update steps")
    p.add_argument("--eps-init", type=float, default=0.01,
                   help="Beta(1, 1/eps_init) prior on the contamination rate (mean ~ eps_init). "
                        "Forgetting rho = 1 - B/N set automatically.")

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

    Each epoch's decode+resize is wrapped in vit_classification.prefetch_iterator
    (n_prefetch=10, same as full_finetune_comparison.stream_batches), so JPEG
    decode for the NEXT micro-batch runs on a background thread while the
    current one trains -- without it, decode is inline on the main thread and
    overlap is one micro-batch deep. The wrap is per-epoch (a fresh generator
    and a fresh prefetch_iterator each time `order` is reshuffled), mirroring
    full_finetune_comparison.py exactly, rather than one prefetch_iterator over
    an unbounded stream.
    """
    from vit_classification import prefetch_iterator

    n = len(train_split)
    labels_all = np.asarray(train_split[ds_config["label_key"]])
    emitted, epoch = 0, 0
    while emitted < total_micro:
        order = protocol.shuffled_indices(n, seed, epoch)

        def _epoch_batches(order=order):
            for start in range(0, n - micro + 1, micro):
                batch_idx = order[start:start + micro]
                images = imagenet.load_images_uint8(
                    train_split, ds_config["img_key"], img_size, batch_idx,
                    num_workers=num_workers,
                )
                yield (jnp.asarray(imagenet.normalize_chunk(images, mean, std)),
                       jnp.asarray(labels_all[batch_idx]))

        for images, labels in prefetch_iterator(_epoch_batches(), n_prefetch=10):
            yield images, labels
            emitted += 1
            if emitted >= total_micro:
                return
        epoch += 1


def load_ood_sets(args, img_size):
    """Preload every OOD set as uint8 host arrays, once, shared across steps.

    Returns [(name, images_uint8, labels, subset_or_None)]. `subset` is the
    array of classes present in R/A (derived from that set's FULL label
    column), used by `evaluate_and_dump` to mask out-of-subset logits before
    scoring; V2 spans all 1000 classes and gets None. Only called when
    `plan["ood_steps"]` is non-empty -- a tuning trial with
    --ood-every-k-dumps 0 must never pay this preload cost (47,500 images).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "alternatives"))
    from precompute_ood_features import build_ood_registry, load_wnid_to_idx

    wnid_to_idx = load_wnid_to_idx()
    registry = build_ood_registry()
    loaded = []
    for name in imagenet.OOD_KEYS:
        split, img_key = registry[name](wnid_to_idx)
        labels = np.asarray(split["label"])
        expected = imagenet.OOD_SIZES[name]
        if len(labels) != expected:
            # A short (or long) load means the HF split changed shape.
            # Scoring it anyway would produce a plausible-looking AUROC over
            # the wrong population, so this must raise rather than warn --
            # checked before the (expensive) image decode below.
            raise SystemExit(
                f"[finetune] {name} has {len(labels)} images, expected "
                f"{expected}"
            )
        subset = np.unique(labels) if name in imagenet.OOD_SUBSET_KEYS else None
        images = imagenet.load_images_uint8(
            split, img_key, img_size, np.arange(len(labels)))
        loaded.append((name, images, labels, subset))
        print(f"[finetune] preloaded {name}: {len(labels)} images"
              + (f", {len(subset)}-class subset" if subset is not None else ""))
    return loaded


def evaluate_and_dump(arm, backbone, head_or_loss, images_uint8, labels, *,
                      split, step, samples_seen, args, mean, std, key,
                      subset=None):
    """Forward one preloaded eval set in chunks and write its dump.

    Returns the scalar metrics so the caller can drive `on_dump`.
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
        # "softmax_logits" on BOTH arms, and for IBProbit that is not a
        # convenience: `IBProbit.__call__(with_logits=True)` returns
        # `_mixture_logp`, which is normalized by construction --
        # (1-eps)*1 + K*(eps/K) == 1 -- so softmaxing it is a no-op that
        # recovers the mixture exactly. That is the same kind of array the VBLL
        # arm dumps (log of its predictive) under the same label, and
        # metrics.classification_metrics is correct on it. Labelling it
        # "moderated_score" instead made run_curve refuse every IBProbit dump.
        logit_kind="softmax_logits",
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


def probe_micro_batch_size_ce(backbone, args, embed_dim, num_classes, img_size,
                              init_bs, max_bs, key):
    """Double the micro-batch size until the CE micro-step OOMs; return the
    last size that fits. This is the per-forward micro-batch -- the target
    (effective) batch is reached by gradient accumulation over several of
    these.

    Modelled on full_finetune_comparison.probe_micro_batch_size (same
    doubling structure, OOM-substring detection, per-attempt
    jax.clear_caches()+gc.collect(), and "even the smallest does not fit"
    RuntimeError), but built on jax_steps.ce_micro_grad with a throwaway
    eqx.nn.Linear head and CrossEntropy loss instead of build_ibprobit --
    so, unlike that function, this one never constructs an IBProbit head and
    does not depend on its constructor at all. See check_micro_batch_required
    for why the ibprobit arm cannot use this (or any) auto-probe yet.
    """
    bf16 = args.bf16
    loss_fn = CrossEntropy(args.label_smooth, num_classes)
    bs = init_bs
    last_ok = None
    while bs <= max_bs:
        # Initialised before the try, same reasoning as the fix to
        # full_finetune_comparison.probe_micro_batch_size: a `finally: del
        # head` must never raise UnboundLocalError over a real exception.
        head = None
        try:
            key, head_key = jr.split(key)
            head = eqx.nn.Linear(embed_dim, num_classes, key=head_key)
            images = jnp.zeros((bs, 3, img_size, img_size), dtype=jnp.float32)
            labels = jnp.zeros((bs,), dtype=jnp.int32)
            _, loss = jax_steps.ce_micro_grad(
                (backbone, head), loss_fn, images, labels, key, bf16=bf16)
            jax.block_until_ready(loss)
            last_ok = bs
            print(f"[finetune] probe micro_batch_size={bs} OK")
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower() or "OOM" in msg:
                print(f"[finetune] probe micro_batch_size={bs} OOM, backing off")
                break
            raise
        finally:
            del head
            jax.clear_caches()
            gc.collect()
        bs *= 2

    if last_ok is None:
        raise RuntimeError(f"Even micro_batch_size={init_bs} does not fit in memory.")
    return last_ok


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
    check_micro_batch_required(args)
    if args.uid is None:
        # fig1_curves.UID_RE parses a uid as {model}_{dataset}_seed{seed}. A
        # uid of just {model}_{arm}_seed{seed} (e.g. dinov3_large_ce_seed137)
        # parses CLEANLY under that pattern with dataset="ce" -- a silent
        # mislabel, not a parse failure. Folding {dataset} in ahead of {arm}
        # keeps that field correct; {arm} still rides along in the uid string
        # (and, independently, in every dump's own meta["arm"]).
        args.uid = f"{args.model}_{args.dataset}_{args.arm}_seed{args.seed}"

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
        # check_micro_batch_required (above) already refused --arm ibprobit
        # with no --micro-batch-size, so reaching here means --arm ce: the
        # local CE-only prober never constructs an IBProbit head.
        key, probe_key = jr.split(key)
        print("[finetune] probing maximal micro-batch size...")
        micro = probe_micro_batch_size_ce(
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

    # Guarded on len(...) rather than truthiness of --ood-every-k-dumps
    # itself: ood_dump_steps is what actually decides whether any step is
    # scheduled, and a tuning trial (--ood-every-k-dumps 0) must never pay
    # the 47,500-image preload cost.
    ood_sets = load_ood_sets(args, args.img_size) if len(plan["ood_steps"]) else []

    if args.arm == "ce":
        loss_fn = CrossEntropy(args.label_smooth, num_classes)
        head = eqx.nn.Linear(embed_dim, num_classes, key=init_key)
        head_lr = args.head_lr if args.head_lr is not None else DEFAULT_HEAD_LR
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
        # B is the EFFECTIVE batch, not the micro batch: the head assimilates the
        # whole accumulated batch in one CAVI update per step, so that is the B the
        # q0 = alpha^2 / B steady-state relation is derived against. Matches
        # run_arm in full_finetune_comparison.py. --q-init overrides q0 when set.
        eps_rho = max(0.0, 1.0 - plan["effective_batch"] / imagenet.IMAGENET_N_TRAIN)
        loss_fn = IBProbit(
            embed_dim, num_classes, key=init_key,
            q_init=args.q_init, alpha=args.alpha, batch_size=plan["effective_batch"],
            kappa=args.kappa, learn_q=not args.no_learn_q, px=args.px,
            warmup=args.warmup, eps_init=args.eps_init, rho=eps_rho,
        )
        head = None
        optimizer = optax.adamw(args.lr, weight_decay=args.weight_decay)
        opt_state = optimizer.init(eqx.filter(backbone, eqx.is_array))

    dump_set = set(int(s) for s in plan["dump_steps"])
    ood_step_set = set(int(s) for s in plan["ood_steps"])
    micro_iter = stream_micro_batches(
        train_split, ds_config, plan["micro"], args.img_size, mean, std,
        plan["steps"] * plan["num_accum"], args.seed,
    )

    dump_index = 0
    train_time = 0.0
    for step in range(1, plan["steps"] + 1):
        step_t0 = time.perf_counter()
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

        # Excludes eval/dump time below (banked before that block starts) and
        # step 1 (JIT compilation, not steady-state throughput) -- mirrors
        # full_finetune_comparison.run_arm's warmup exclusion.
        step_time = time.perf_counter() - step_t0
        if step > 1:
            train_time += step_time

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
            if not np.isfinite(scalars["val_nll"]):
                # A diverged run would otherwise train silently to the full
                # step budget, burning the rest of the wall-clock/compute
                # budget on a broken model.
                raise RuntimeError(
                    f"[finetune] step {step}: val_nll={scalars['val_nll']!r} "
                    f"is not finite -- the run has diverged."
                )
            if step in ood_step_set:
                for name, images_uint8, labels, subset in ood_sets:
                    evaluate_and_dump(
                        args.arm, backbone, head_or_loss, images_uint8, labels,
                        split=f"ood_{name}", step=step, samples_seen=samples_seen,
                        args=args, mean=mean, std=std, key=fwd_key, subset=subset,
                    )
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

    # Steady-state train throughput: probe_throughput.verdict needs a train
    # img/s to compare against the loader's decode rate, and nothing else in
    # this script produces one. n_timed excludes step 1 (JIT compilation);
    # with plan["steps"] == 1 there is no post-compile step to time at all.
    n_timed = plan["steps"] - 1
    if n_timed > 0 and train_time > 0:
        mean_step_time = train_time / n_timed
        img_per_sec = plan["effective_batch"] / mean_step_time
        print(f"[finetune] train time: {train_time:.3f}s total, "
              f"{mean_step_time * 1000:.2f}ms/step (excl. step 1, compile), "
              f"{img_per_sec:.2f} img/s")
    else:
        print(f"[finetune] train time: {train_time:.3f}s total "
              f"(no post-compile step to report a rate for)")

    print(f"[finetune] done: {dump_index} dumps under "
          f"{args.dump_root}/{args.group_id}/{args.uid}")


if __name__ == "__main__":
    from jax import config as jax_config

    parsed = build_argparser().parse_args()
    jax_config.update("jax_platform_name", parsed.device)
    main(parsed)
