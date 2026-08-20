"""Full-network fine-tuning of DINOv3 on ImageNet-1k -- JAX arms.

One arm per invocation (`--arm ce|ibprobit`). val_nll is evaluated every
`--eval-quantum` SAMPLES -- uniformly, not on a log-spaced step grid -- so the
eval index means the same amount of training at every rung of the batch ladder
and is usable as a pruner key and a patience clock. Dumps are written in the
same format the probing figures are scored from, so temperature scaling, Brier
and OOD detection are all computed post hoc by that same code.

This is NOT scripts/dinov3/full_finetune_comparison.py, which stays as the
IVON/EVON benchmarking harness with inline metrics. The two share their jitted
kernels via bllarse.finetune.jax_steps.
"""

import argparse
import gc
import math
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
from bllarse.mlflow_utils import child_run  # noqa: E402
from bllarse.finetune import checkpoint, imagenet, jax_steps, protocol  # noqa: E402
from bllarse.losses import CrossEntropy  # noqa: E402

# Fine-tuning resolution. DINOV3_DATASET_IMG_SIZES maps imagenet1k -> 512, but
# that table exists for FROZEN FEATURE EXTRACTION; fine-tuning at 512 is ~5x the
# cost for no benefit, so inheriting it must be impossible by accident.
FINETUNE_IMG_SIZE = 224

# fp16-safe mask sentinel for evaluate_split's `subset` masking.
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


def _write_best(args, best, last_eval_index):
    """Write the buffered argmin-val_nll evaluation.

    Called from a `finally`, so it runs whether the loop finished its budget or
    was stopped by an EarlyConvergence raised out of `on_dump`. With patience
    enabled that is the ORDINARY path, not an edge case: a converged trial
    leaves `main` by exception, and a `_write_best` outside the `finally` would
    silently produce no dump for exactly the trials that converged.

    `best` is None for TWO reasons that need telling apart, because they call
    for opposite responses. Either no evaluation ran (a budget shorter than one
    eval interval, which `resolve_plan` already refuses), or evaluations ran
    and every one was non-finite -- a diverged run, which is a result. On the
    probing side the single-cause message sent a reader hunting a budget bug on
    a run that had completed 30,000 evaluations of pure nan.
    """
    if best is None:
        if not last_eval_index:
            raise ValueError(
                f"{args.uid}: no evaluation ran, so there is no dump to write. "
                f"The step budget is shorter than one eval interval."
            )
        raise ValueError(
            f"{args.uid}: {last_eval_index} evaluations ran and EVERY ONE had "
            f"a non-finite val_nll, so there is no dump to write. This is a "
            f"diverged run, not a misconfigured budget -- the argmin buffer "
            f"only ever accepts finite values."
        )
    for split_name, logits, labels in best["splits"]:
        evaldump.write_dump(
            args.dump_root, args.group_id, args.uid, best["step"],
            logits, labels,
            arm=args.arm, split=split_name, logit_kind="softmax_logits",
            seed=args.seed, samples_seen=best["samples_seen"],
            # Recorded here because only this loop knows how many evaluations
            # there were. A run writes one dump, so nothing downstream can tell
            # a converged optimum from a budget that ran out -- and those two
            # call for opposite corrections to the campaign.
            at_last_eval=best["eval_index"] == last_eval_index,
        )
    print(
        f"[finetune] {args.uid}: best val_nll={best['val_nll']:.4f} at step "
        f"{best['step']} (eval {best['eval_index']} of {last_eval_index}, "
        f"{best['samples_seen']} samples)"
        + (" AT LAST EVAL -- truncated, not converged"
           if best["eval_index"] == last_eval_index else "")
        + f"; wrote {len(best['splits'])} dumps",
        flush=True,
    )


def resolve_plan(args, micro_batch: int) -> dict:
    """Turn (target batch, micro batch, budget) into the concrete schedule.

    `steps` is derived from the ACHIEVED effective batch, not the requested one:
    Fig 5's x-axis is samples seen, so a plan built on a batch size the run does
    not use would mislabel every point on it. Since design A8 pins the micro
    batch to a divisor of every ladder rung, `accum_plan` now guarantees the two
    agree -- but the derivation stays on the achieved value, because the pinned
    constant and the ladder are separately editable.

    The budget is EITHER --max-samples (tuning) or --epochs (the final
    campaign), never both: tuning is capped in samples so the tuned batch size
    transfers, and the final run is capped at 4 epochs.
    """
    micro, num_accum, effective = protocol.accum_plan(args.batch_size, micro_batch)
    if args.max_samples:
        steps = protocol.steps_for_samples(args.max_samples, effective)
    else:
        steps = protocol.steps_for_epochs(
            imagenet.IMAGENET_N_TRAIN, args.epochs, effective
        )
    eval_every = protocol.eval_every_steps(effective, args.eval_quantum)
    n_evals = steps // eval_every
    if n_evals < 1:
        raise SystemExit(
            f"[finetune] budget of {steps} steps at effective batch {effective} "
            f"is shorter than one evaluation interval ({eval_every} steps = "
            f"{args.eval_quantum} samples). The run would produce no dump."
        )
    return {
        "micro": micro,
        "num_accum": num_accum,
        "effective_batch": effective,
        "steps": steps,
        # Evaluation is scheduled on SAMPLES, not steps: `eval_quantum` samples
        # pass between consecutive val_nll measurements no matter which rung of
        # the ladder this run sits on, so eval index i is the same samples_seen
        # for every trial in a study. That is what makes the index usable as a
        # MedianPruner key and as a patience clock -- neither of which the
        # log-spaced step grid this replaces could support.
        "eval_every": eval_every,
        "n_evals": n_evals,
        "eval_quantum": args.eval_quantum,
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
        help="Epoch cap for the final campaign. Ignored when --max-samples is "
             "given; steps are derived from this and the effective batch.",
    )
    p.add_argument(
        "--max-samples", type=int, default=0,
        help="Sample budget, overriding --epochs when nonzero. This is what "
             "tuning uses (protocol.MAX_SAMPLES = 2,000,000): at a fixed STEP "
             "budget samples_seen = steps * batch, so large batches would be "
             "handed more data and the tuned batch size would not transfer to "
             "the final run.",
    )
    p.add_argument(
        "--batch-size", type=int, required=True,
        help="TARGET (effective) batch, reached by gradient accumulation. This "
             "is the tuned quantity; --micro-batch-size is a memory artifact.",
    )
    p.add_argument(
        "--micro-batch-size", type=int, required=True,
        help="Per-forward batch, a memory artifact absorbed by "
             "num_accum = batch_size / micro_batch_size. REQUIRED and must "
             "divide --batch-size exactly. Measure it once per arm and per GPU "
             "with scripts/finetune/probe_micro_batch.py and pin the result "
             "(design A8) -- the old per-run auto-probe made every trial pay "
             "a doubling-until-OOM search whose answer never changed.",
    )

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
    # Exposed rather than left at optax's defaults because they are a per-STEP
    # timescale, and this script is run at two different step counts: ~3906 for
    # a tuning trial and ~10,009 for a 4-epoch final run. Holding
    # (1 - beta) * steps invariant is what makes those two comparable -- see
    # protocol.betas_for, which is what computes the value to pass here.
    #
    # The defaults ARE optax's, so an invocation that omits them is unchanged.
    p.add_argument(
        "--beta1", type=float, default=protocol.REFERENCE_BETAS[0],
        help="AdamW beta1. Transfers with the run length; see "
             "protocol.betas_for.",
    )
    p.add_argument(
        "--beta2", type=float, default=protocol.REFERENCE_BETAS[1],
        help="AdamW beta2. Transfers with the run length; see "
             "protocol.betas_for.",
    )
    # MLflow, mirroring scripts/dinov3/probe.py's flags. Off unless asked, so a
    # standalone invocation stays offline and a tracking outage can never take a
    # GPU job down (child_run degrades to a no-op run object).
    #
    # These are on the SCRIPT, not on the tuning worker, and that is what makes
    # the same integration serve both campaigns: the tuning trials reach here
    # through finetune_worker.trial_config and the final 4-epoch runs through
    # tuning.final_config, and both render a config dict into these flags.
    p.add_argument("--enable-mlflow", action="store_true")
    p.add_argument("--mlflow-experiment", default=None)
    p.add_argument("--mlflow-tracking-uri", default=None)
    p.add_argument("--label-smooth", type=float, default=0.0)
    p.add_argument("--num-update-iters", type=int, default=128,
                   help="CAVI iterations per macro batch (IBProbit arm).")
    p.add_argument("--bf16", action="store_true")

    # --- IBProbit head knobs (mirror scripts/dinov3/ood_evaluation.py) ---
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

    p.add_argument(
        "--eval-quantum", type=int, default=protocol.EVAL_QUANTUM,
        help="Samples between consecutive val_nll evaluations. Must be a whole "
             "number of --batch-size steps, which every rung of the ladder "
             "satisfies by construction.",
    )
    p.add_argument(
        "--eval-splits", default="val+test", choices=["val", "val+test"],
        help="Which pinned splits to forward at each evaluation. Tuning passes "
             "`val`: the objective is val_nll alone, and forwarding the 30k "
             "test partition 122 times would roughly double the campaign.",
    )
    p.add_argument(
        "--dump-every-k-evals", type=int, default=0,
        help="Write a full dump on every k-th evaluation, giving Fig 5 its "
             "trajectory. 0 (the default, and what tuning uses) writes exactly "
             "ONE dump per run: the buffered argmin-val_nll evaluation. At "
             "1000 classes a val dump is 40 MB, so 122 of them would be 4.9 GB "
             "per run.",
    )
    # Hand OOD to a separate process on a separate gpu instead of evaluating
    # it here. The schedule is unchanged -- --ood-every-k-dumps still decides
    # WHICH steps -- but at those steps this run writes a checkpoint and moves
    # on, and scripts/finetune/ood_worker.py does the 98,389-image evaluation.
    # That takes ~1 h per run off the TRAINING gpu, which is what keeps a
    # dinov3_huge 4-epoch job inside its 24 h wall, and it lets OOD run on a
    # finer stride than the trainer could otherwise afford.
    p.add_argument("--ood-checkpoint-root", default=None,
                   help="Write checkpoints for the OOD worker instead of "
                        "evaluating OOD inline.")
    p.add_argument("--ood-max-pending", type=int, default=2,
                   help="Skip a scheduled checkpoint when this many are already "
                        "waiting. A lagging worker must not fill /ptmp; a "
                        "dinov3_huge checkpoint is ~3.4 GB.")
    p.add_argument(
        "--ood-every-k-dumps", type=int, default=2,
        help="Write OOD dumps on every k-th scheduled dump. 0 disables OOD "
             "entirely, which is what tuning trials use. Requires "
             "--dump-every-k-evals, since it strides that grid.",
    )
    p.add_argument("--eval-chunk", type=int, default=256)
    p.add_argument("--val-fraction", type=float, default=imagenet.DEFAULT_VAL_FRACTION)

    p.add_argument("--dump-root",
                   default=os.environ.get("BLLARSE_EVAL_DUMP_ROOT", "eval_dumps"))
    p.add_argument("--group-id", default="finetune")
    p.add_argument("--uid", default=None)
    p.add_argument("--split-cache-dir", default=".cache/splits")
    p.add_argument(
        "--train-shards", default=os.environ.get("BLLARSE_FT_SHARDS", ""),
        help="Directory holding a pre-decoded uint8 shard (see "
             "scripts/finetune/build_shards.py). When set, training reads the "
             "shard instead of decoding JPEGs -- measured ~5x on the step, "
             "because decode contends with JAX dispatch for the GIL rather "
             "than being slow in itself. Batch order is unchanged.",
    )
    p.add_argument("--seed", type=int, default=137)
    p.add_argument("--device", default="gpu")

    p.add_argument("--optimizer", default="adamw", choices=["adamw"],
                   help="Present because the shared micro-batch prober reads "
                        "it. IVON/EVON live in full_finetune_comparison.py.")
    p.add_argument("--estimator", default="sampling", choices=["sampling"],
                   help="Present for the shared prober; see --optimizer.")
    return p


def stream_micro_batches(train_split, ds_config, micro, img_size, mean, std,
                         total_micro, seed, num_workers=16, shard=None):
    """Yield exactly `total_micro` micro-batches, reshuffling between passes.

    The permutation comes from protocol.shuffled_indices (numpy), NOT jax.random,
    so the PyTorch arm reproduces this exact order. Short trailing batches are
    skipped to keep the jitted shape constant.

    `shard`, when given, is a shards.ShardReader and replaces per-image JPEG
    decode with one gather out of a memmap. The batch ORDER is identical either
    way -- row i of a shard is row i of the split, which is exactly why the
    shard is invisible to `shuffled_indices` and therefore to the PyTorch arm.
    That equivalence is the load-bearing property; a shard that permuted its
    rows would desynchronise the two arms silently.

    Sharding matters because the step is not the bottleneck. On an A100 one
    dinov3_large step runs at 220 img/s with synthetic inputs and ~40 through
    this function's JPEG path -- decode contends with JAX dispatch for the GIL,
    and a fresh ThreadPoolExecutor per micro-batch makes it worse.

    Each epoch is wrapped in vit_classification.prefetch_iterator (n_prefetch=10)
    so the NEXT micro-batch is prepared while the current one trains. Kept for
    the shard path too: the gather is far cheaper than decode but still not
    free, and overlapping it costs nothing.
    """
    from vit_classification import prefetch_iterator

    n = len(shard) if shard is not None else len(train_split)
    labels_all = (np.asarray(shard.labels) if shard is not None
                  else np.asarray(train_split[ds_config["label_key"]]))
    emitted, epoch = 0, 0
    while emitted < total_micro:
        order = protocol.shuffled_indices(n, seed, epoch)

        def _epoch_batches(order=order):
            for start in range(0, n - micro + 1, micro):
                batch_idx = order[start:start + micro]
                if shard is not None:
                    images = shard.take(batch_idx)
                else:
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dinov3"))
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


def evaluate_split(arm, backbone, head_or_loss, images_uint8, labels, *,
                   args, mean, std, key, subset=None):
    """Forward one preloaded eval set in chunks. Returns (logits, scalars).

    Deliberately does NOT write. Evaluation now happens every `eval_quantum`
    samples -- 122 times in a tuning run -- while at most a stride of those
    points is ever dumped, so computing and writing had to come apart. The
    caller buffers the argmin's logits in memory and writes once, from
    `_write_best`.

    Normalization happens per chunk (imagenet.normalize_chunk), so the float32
    expansion only ever covers `--eval-chunk` images at a time.

    `subset`, if given, restricts scoring to those class indices: every other
    logit is driven to MASK_NEG before metrics are computed, so a later dump
    and the scalars reported here share the same masked view (OOD evaluation
    over ImageNet-R/-A, which cover only 200 of the 1000 classes).
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

    # Dumps carry logit_kind="softmax_logits" on BOTH arms, and for IBProbit
    # that is not a convenience: `IBProbit.__call__(with_logits=True)` returns
    # `_mixture_logp`, which is normalized by construction --
    # (1-eps)*1 + K*(eps/K) == 1 -- so softmaxing it is a no-op that recovers
    # the mixture exactly. That is the same kind of array the VBLL arm dumps
    # (log of its predictive) under the same label, and
    # metrics.classification_metrics is correct on it. Labelling it
    # "moderated_score" instead made run_curve refuse every IBProbit dump.
    return logits, metrics.classification_metrics(logits, labels)


@eqx.filter_jit
def eval_logits_ce(backbone, head, images, key, bf16=False):
    features = jax_steps.forward_features(backbone, images, key, bf16=bf16)
    return jax.vmap(head)(features)


@eqx.filter_jit
def eval_logits_ibprobit(backbone, loss_fn, images, labels, key, bf16=False):
    features = jax_steps.forward_features(backbone, images, key, bf16=bf16)
    _, logits = loss_fn(features, labels, with_logits=True, loss_type=2)
    return logits



def build_model(args, *, embed_dim, num_classes, init_key, effective_batch):
    """(loss_fn, head) for `args.arm`. The ONE place either is constructed.

    Extracted so the OOD worker rebuilds exactly what training built. A worker
    with its own copy of this would drift silently -- a different head init
    scale or a different IBProbit prior deserialises without error and scores a
    model that never existed.

    `head` is None on the IBProbit arms: the head IS the loss_fn there,
    assimilated in closed form rather than by an optimizer.
    """
    if args.arm == "ce":
        return CrossEntropy(args.label_smooth, num_classes), eqx.nn.Linear(
            embed_dim, num_classes, key=init_key)

    from bllarse.losses import IBProbit
    # B is the EFFECTIVE batch, not the micro batch: the head assimilates the
    # whole accumulated batch in one CAVI update per step, so that is the B the
    # q0 = alpha^2 / B steady-state relation is derived against.
    eps_rho = max(0.0, 1.0 - effective_batch / imagenet.IMAGENET_N_TRAIN)
    return IBProbit(
        embed_dim, num_classes, key=init_key,
        q_init=args.q_init, alpha=args.alpha, batch_size=effective_batch,
        kappa=args.kappa, learn_q=not args.no_learn_q, px=args.px,
        warmup=args.warmup, eps_init=args.eps_init, rho=eps_rho,
    ), None


def default_uid(args) -> str:
    """The run's identity: model, dataset, arm and seed."""
    return f"{args.model}_{args.dataset}_{args.arm}_seed{args.seed}"


def resolve_uid(args) -> str:
    """Set `args.uid` from the run's identity if it was not passed explicitly.

    Called from `main` BEFORE the MLflow run opens, not from `_train`. That
    ordering is the whole point: `child_run` names the run from `args.uid`, and
    when the default was computed inside `_train` the run had already been
    created with uid=None. Tuning never showed it -- finetune_worker passes
    uid=trialNNNNN -- but the final campaign passes no uid at all, so every seed
    of every arm would have landed under one nameless run.

    Idempotent, so `_train` may still be called directly (the contract tests do)
    without the default being computed twice or differently.
    """
    if getattr(args, "uid", None) is None:
        args.uid = default_uid(args)
    return args.uid


def _train(args, on_dump=None, run=None) -> None:
    """Fine-tune one arm: gradient accumulation to the target batch, val_nll
    evaluated every `--eval-quantum` samples.

    `on_dump(eval_index, step, samples_seen, scalars)` matches the hook
    bllarse.tuning.objective.evaluate passes, so the tuning campaign drives this
    script without a wrapper. Any exception `on_dump` raises (e.g.
    objective.EarlyConvergence, optuna.TrialPruned) propagates out of this loop
    uncaught -- the caller is responsible for catching it. The buffered argmin
    dump is still written on the way out, from a `finally`.
    """
    check_img_size(args)
    check_arm_flags(args)
    if args.uid is None:
        # fig1_curves.UID_RE is {model}_(?P<dataset>.+)_seed{seed}, and that
        # dataset group is GREEDY -- so this parses as dataset="imagenet1k_ce",
        # not "imagenet1k". This comment used to claim that folding {dataset}
        # in ahead of {arm} kept the field correct. It does not; it only keeps
        # the dataset at the FRONT of it.
        #
        # Left as is because it is harmless for this campaign -- imagenet1k is
        # the only dataset, and the arm is independently in group_id and in
        # every dump's meta["arm"] -- and because scripts/finetune/
        # finetune_vbll.py must produce the same shape to stay groupable. Worth
        # revisiting if a fine-tuning figure ever keys on the parsed dataset.
        # Tuning trials never reach here: finetune_worker passes uid=trialNNNNN.
        args.uid = default_uid(args)

    # Heavy, GPU-oriented imports stay inside main(): this module is imported
    # by the CPU-only test suite (and by --help), and equimo/datasets pull in
    # a lot of machinery neither needs.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dinov3"))
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

    plan = resolve_plan(args, args.micro_batch_size)
    print(f"[finetune] arm={args.arm} effective_batch={plan['effective_batch']} "
          f"= {plan['num_accum']} x micro {plan['micro']}, steps={plan['steps']}, "
          f"eval every {plan['eval_every']} steps ({plan['eval_quantum']} "
          f"samples) = {plan['n_evals']} evals")

    # Pinned val/test partition over the official validation split, preloaded
    # once as uint8 and normalized per chunk at eval time.
    val_idx, test_idx = imagenet.pinned_eval_split(
        args.split_cache_dir, n=len(eval_split), val_fraction=args.val_fraction
    )
    eval_labels_all = np.asarray(eval_split[ds_config["label_key"]])
    # `val` alone during tuning: the objective is val_nll, and forwarding the
    # 30k test partition at all 122 evaluations would roughly double the
    # campaign for a number nothing reads.
    split_names = args.eval_splits.split("+")
    eval_sets = {}
    for name, idx in (("val", val_idx), ("test", test_idx)):
        if name not in split_names:
            continue
        eval_sets[name] = (
            imagenet.load_images_uint8(eval_split, ds_config["img_key"],
                                       args.img_size, idx),
            eval_labels_all[idx],
        )
        print(f"[finetune] preloaded {name}: {len(idx)} images")

    # OOD rides a stride of the dump grid, which is itself a stride of the eval
    # grid -- so with --dump-every-k-evals 0 (tuning) nothing is scheduled and
    # the 47,500-image preload must never be paid.
    ood_enabled = bool(args.dump_every_k_evals and args.ood_every_k_dumps)
    # Delegated: the schedule still fires, but this process neither preloads
    # the suite (14.8 GB of host arrays) nor forwards it.
    ood_delegated = bool(ood_enabled and args.ood_checkpoint_root)
    # Travels in the checkpoint so the worker attaches to THIS run instead of
    # opening a sibling. None when tracking is off, which the worker honours.
    mlflow_run_id = getattr(run, "run_id", None) if run is not None else None
    ood_sets = load_ood_sets(args, args.img_size) if (
        ood_enabled and not ood_delegated) else []
    if ood_delegated:
        print(f"[finetune] OOD delegated to the worker via "
              f"{args.ood_checkpoint_root} (every {args.ood_every_k_dumps} dumps)")

    loss_fn, head = build_model(
        args, embed_dim=embed_dim, num_classes=num_classes,
        init_key=init_key, effective_batch=plan["effective_batch"])

    if args.arm == "ce":
        head_lr = args.head_lr if args.head_lr is not None else DEFAULT_HEAD_LR
        param_labels = (
            jax.tree.map(lambda _: "backbone", eqx.filter(backbone, eqx.is_array)),
            jax.tree.map(lambda _: "head", eqx.filter(head, eqx.is_array)),
        )
        optimizer = optax.multi_transform(
            {
                "backbone": optax.adamw(
                    args.lr, b1=args.beta1, b2=args.beta2,
                    weight_decay=args.weight_decay),
                # Head weight decay is 0 on every arm (design D6): VBLL's KL is
                # its head regulariser and IBProbit runs no optimizer on the
                # head at all, so a nonzero value here would be an asymmetry.
                #
                # The betas are NOT similarly split: they set the optimizer's
                # averaging horizon in steps, and both groups take the same
                # number of steps.
                "head": optax.adamw(
                    head_lr, b1=args.beta1, b2=args.beta2, weight_decay=0.0),
            },
            param_labels,
        )
        opt_state = optimizer.init(eqx.filter((backbone, head), eqx.is_array))
    else:
        optimizer = optax.adamw(
            args.lr, b1=args.beta1, b2=args.beta2,
            weight_decay=args.weight_decay)
        opt_state = optimizer.init(eqx.filter(backbone, eqx.is_array))

    shard = None
    if args.train_shards:
        from bllarse.finetune import shards as shards_mod
        # open_shard REFUSES an incomplete shard: a partial 193 GB file is
        # mostly zeros, and zeros train as valid black images -- the run would
        # look merely bad rather than broken.
        shard = shards_mod.open_shard(
            args.train_shards, args.dataset, "train", args.img_size)
        print(f"[finetune] training from shard: {len(shard)} rows "
              f"({args.train_shards})")

    micro_iter = stream_micro_batches(
        train_split, ds_config, plan["micro"], args.img_size, mean, std,
        plan["steps"] * plan["num_accum"], args.seed, shard=shard,
    )

    # The argmin-val_nll evaluation, held in memory and written once from the
    # `finally` below. Buffering rather than writing every point is what keeps a
    # tuning run to one 40 MB dump instead of 122 of them.
    best = None
    eval_index = 0
    train_time = 0.0
    try:
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

            # Evaluation is spaced in SAMPLES, so this fires every `eval_every`
            # steps and eval index i means i * eval_quantum samples on every rung
            # of the ladder.
            if step % plan["eval_every"]:
                continue

            samples_seen = step * plan["effective_batch"]
            eval_index = samples_seen // plan["eval_quantum"]
            payload, scalars = [], {}
            for name, (images_uint8, labels) in eval_sets.items():
                logits, out = evaluate_split(
                    args.arm, backbone, head_or_loss, images_uint8, labels,
                    args=args, mean=mean, std=std, key=fwd_key,
                )
                payload.append((name, logits, labels))
                scalars.update({f"{name}_{k}": v for k, v in out.items()})

            # `isfinite` guards the `best is None` arm specifically. Without it a
            # non-finite FIRST value becomes `best`, every later `v < nan` is
            # False, and the run keeps its diverged evaluation as the argmin for
            # the rest of the budget. A non-finite value carries no information
            # about improvement and is never adopted.
            val_nll = float(scalars["val_nll"])
            if math.isfinite(val_nll) and (best is None or val_nll < best["val_nll"]):
                best = {
                    "val_nll": val_nll,
                    "step": step,
                    "eval_index": eval_index,
                    "samples_seen": samples_seen,
                    "splits": payload,
                }

            # Checked AFTER the buffer update so one bad evaluation cannot discard
            # a good dump -- but still fatal, because a diverged run would
            # otherwise train silently to the full budget on a broken model.
            if not np.isfinite(val_nll):
                raise RuntimeError(
                    f"[finetune] step {step}: val_nll={val_nll!r} is not finite "
                    f"-- the run has diverged."
                )

            # Trajectory dumps for Fig 5. With --dump-every-k-evals 0 (tuning)
            # nothing is written here and the run's only disk write is the
            # buffered argmin, from _write_best.
            # The final evaluation always dumps, whatever the stride lands on.
            # curves._id_ood_pair raises when the last test dump and the last
            # OOD dump sit at different steps, and Table 1's converged endpoint
            # is read from exactly that point -- so a stride that misses the
            # end (len(evals) not 1 mod k) would break scoring rather than
            # merely thinning the trajectory. The geometric schedule this
            # replaces force-appended the last step for the same reason.
            is_last_eval = eval_index >= plan["n_evals"]
            if args.dump_every_k_evals and (
                is_last_eval or eval_index % args.dump_every_k_evals == 0
            ):
                dump_index = eval_index // args.dump_every_k_evals
                for name, logits, labels in payload:
                    evaldump.write_dump(
                        args.dump_root, args.group_id, args.uid, step,
                        logits, labels, arm=args.arm, split=name,
                        logit_kind="softmax_logits", seed=args.seed,
                        samples_seen=samples_seen,
                    )
                # The `and` is load-bearing: --ood-every-k-dumps 0 disables OOD,
                # and the old `if ood_sets and (...)` short-circuited before the
                # modulo. Hoisting the condition out lost that guard and divided
                # by zero on every tuning trial.
                ood_due = bool(args.ood_every_k_dumps) and (
                    is_last_eval or dump_index % args.ood_every_k_dumps == 0)
                if ood_delegated and ood_due:
                    stem = checkpoint.write(
                        args.ood_checkpoint_root,
                        group_id=args.group_id, uid=args.uid, step=step,
                        blob_suffix=".eqx",
                        write_blob=lambda path: eqx.tree_serialise_leaves(
                            path, (backbone, head_or_loss)),
                        meta={
                            "framework": "jax",
                            "samples_seen": samples_seen,
                            "eval_index": eval_index,
                            "is_last_eval": bool(is_last_eval),
                            "mlflow_run_id": mlflow_run_id,
                            "args": vars(args),
                        },
                        max_pending=args.ood_max_pending,
                    )
                    # Logged either way: a silently dropped OOD point is
                    # indistinguishable from one the schedule never asked for.
                    print(f"[finetune] ood checkpoint step {step}: "
                          + ("written" if stem else
                             f"SKIPPED, {args.ood_max_pending} already pending"),
                          flush=True)
                if ood_sets and ood_due:
                    for name, images_uint8, labels, subset in ood_sets:
                        logits, _ = evaluate_split(
                            args.arm, backbone, head_or_loss, images_uint8, labels,
                            args=args, mean=mean, std=std, key=fwd_key, subset=subset,
                        )
                        evaldump.write_dump(
                            args.dump_root, args.group_id, args.uid, step,
                            logits, labels, arm=args.arm, split=f"ood_{name}",
                            logit_kind="softmax_logits", seed=args.seed,
                            samples_seen=samples_seen,
                        )

            print(f"[finetune] eval {eval_index}/{plan['n_evals']} "
                  f"step {step}/{plan['steps']} " + " ".join(
                      f"{k}={v:.4f}" for k, v in sorted(scalars.items())))
            # BEFORE on_dump, deliberately: on_dump is where the pruner and
            # the patience clock raise, so logging after it would drop the very
            # evaluation that ended the trial -- the one that explains why.
            if run is not None:
                run.log_metrics(dict(scalars), step=samples_seen)
            if on_dump is not None:
                on_dump(eval_index, step, samples_seen, scalars)
            # gc.collect() only -- NOT jax.clear_caches() here. Dropping the
            # compiled fwd+bwd graph plus the eval-chunk shapes on every
            # evaluation forces a full cold XLA recompile each time: this repo
            # has no persistent compilation cache configured, and at 122
            # evaluations that would dwarf the run itself.
            # full_finetune_comparison.py only clears once, at the very end of a
            # run, for the same reason -- see below.
            gc.collect()
    finally:
        # Reached whether the loop finished its budget, was stopped by an
        # EarlyConvergence raised out of `on_dump` (the ordinary path once
        # patience is on), or died -- so the argmin dump exists either way
        # and the run stays scoreable.
        #
        # Only when no trajectory was written. With --dump-every-k-evals set,
        # the trajectory IS the record and `early_stop` selects the reporting
        # point from it, exactly as the probing Fig 1 path does; adding a
        # buffered argmin on top would mix two dump semantics in one run --
        # one row carrying `at_last_eval` and the rest not.
        if not args.dump_every_k_evals:
            _write_best(args, best, eval_index)
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

    # eval_index, not dump_index: dump_index is bound only inside the
    # trajectory branch, so with --dump-every-k-evals 0 (every tuning trial) it
    # never exists and this line raised UnboundLocalError AFTER the run had
    # trained correctly and written its dump -- turning a good trial into a
    # FAILED one at the very last statement.
    print(f"[finetune] done: {eval_index} evaluations under "
          f"{args.dump_root}/{args.group_id}/{args.uid}")



def main(args, on_dump=None) -> None:
    """Public entry point: the training run, wrapped in one MLflow run.

    A thin wrapper so `_train` keeps its indentation and its `finally`. The
    signature is unchanged -- `bllarse.tuning.objective.evaluate` calls
    `module.main(args, on_dump=dump_cb)` -- so tuning is unaffected.

    `vars(args)` is the config, which is what makes `enable_mlflow` and friends
    reach `child_run` without a second plumbing path, and what gets logged as
    the run's params. The run is a no-op object unless --enable-mlflow is set,
    so a standalone invocation stays offline.

    Exceptions raised out of `_train` -- optuna.TrialPruned from the pruner,
    objective.EarlyConvergence from patience -- propagate untouched. child_run
    marks those KILLED rather than FAILED; see BY_DESIGN_EXCEPTIONS.
    """
    resolve_uid(args)
    with child_run(
        vars(args),
        run_name=args.uid,
        tags={
            "arm": getattr(args, "arm", "vbll"),
            "group_id": args.group_id,
            "uid": args.uid,
            "model": args.model,
            "dump_dir": str(Path(args.dump_root) / args.group_id / str(args.uid)),
        },
    ) as run:
        _train(args, on_dump=on_dump, run=run)

if __name__ == "__main__":
    from jax import config as jax_config

    parsed = build_argparser().parse_args()
    jax_config.update("jax_platform_name", parsed.device)
    main(parsed)
