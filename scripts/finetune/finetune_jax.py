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
import re
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
MASK_NEG = imagenet.MASK_NEG

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
    and every one was non-finite -- a diverged run, which is a result. A
    single-cause message would send a reader hunting a budget bug on a run that
    completed every one of its evaluations as nan.
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
            # call for opposite corrections to the budget.
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
    Fig 3's x-axis is samples seen, so a plan built on a batch size the run does
    not use would mislabel every point on it. Since finetune_space.MICRO_BATCH
    pins the micro batch to a divisor of every ladder rung, `accum_plan`
    guarantees the two agree -- but the derivation stays on the achieved value,
    because the pinned constant and the ladder are separately editable.

    The budget is EITHER --max-samples (tuning) or --epochs (a final run),
    never both: tuning is capped in samples so the tuned batch size transfers,
    and a final run is capped in epochs.
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
        # MedianPruner key and as a patience clock.
        "eval_every": eval_every,
        "n_evals": n_evals,
        "eval_quantum": args.eval_quantum,
    }


def _scaled(schedule, scale: float):
    """`schedule` times a constant, whether it is a float or an optax callable."""
    if scale == 1.0:
        return schedule
    if callable(schedule):
        return lambda count: scale * schedule(count)
    return scale * schedule


def _resolve_layer_decay(args, depth: int) -> float:
    """`--layer-decay`, or `--layer-decay-span` converted at the REAL depth.

    Resolved here rather than in the launcher because this is where the depth
    is authoritative -- it comes off the loaded backbone. A depth table in the
    launcher would be one more thing to keep in sync with the checkpoints, and
    getting it wrong is silent: the run trains, it just damps the trunk by a
    factor nobody chose.
    """
    span_on = args.layer_decay_span != protocol.LAYER_DECAY_SPAN_OFF
    decay_on = args.layer_decay != protocol.LAYER_DECAY_OFF
    if span_on and decay_on:
        raise SystemExit(
            "[finetune] --layer-decay and --layer-decay-span are mutually "
            "exclusive: they set the same quantity, one per block and one "
            "across the whole trunk. Pass exactly one."
        )
    if not span_on:
        return args.layer_decay
    decay = protocol.layer_decay_for_span(args.layer_decay_span, depth)
    print(f"[finetune] LLRD span {args.layer_decay_span:.4g} over {depth} "
          f"blocks -> per-block decay {decay:.4f}.", flush=True)
    return decay


def _trunk_block_of(path, depth: int) -> int | None:
    """Which transformer block a trunk leaf belongs to, from its pytree path.

    equimo nests its layers as `.blocks[<stage>].blocks[<layer>]`, so the LAYER
    index is the SECOND `blocks[...]` -- taking the first labels every parameter
    as block 0 and silently disables LLRD while the run still trains.

    The trunk's final `.norm` is the TOP and rides with the last block; the
    patch embedding, cls / register / mask tokens and positional embeddings sit
    below every block and return None, the strongest damping.
    """
    key = jax.tree_util.keystr(path)
    found = re.findall(r"blocks\[(\d+)\]", key)
    if len(found) >= 2:
        return int(found[1])
    if key.startswith(".norm."):
        return depth - 1
    return None


def _trunk_lr_scales(depth: int, decay: float) -> dict[str, float]:
    """Every trunk group's LR multiplier. Derived from depth, not from a walk,
    so the transform dict exists before any pytree is touched."""
    scales = {"trunk_embed": protocol.llrd_scale(None, depth, decay)}
    for block in range(depth):
        scales[f"trunk_b{block:03d}"] = protocol.llrd_scale(block, depth, decay)
    return scales


def _trunk_labels_of(tree, depth: int):
    name = lambda path, _leaf: (  # noqa: E731
        "trunk_embed" if (b := _trunk_block_of(path, depth)) is None
        else f"trunk_b{b:03d}")
    return jax.tree_util.tree_map_with_path(name, eqx.filter(tree, eqx.is_array))


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
        "--epochs", type=float, default=2.0,
        help="Epoch cap for a final run. Ignored when --max-samples is "
             "given; steps are derived from this and the effective batch.",
    )
    p.add_argument(
        "--max-samples", type=int, default=0,
        help="Sample budget, overriding --epochs when nonzero. This is what "
             "tuning uses (protocol.MAX_SAMPLES by default): at a fixed STEP "
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
             "(finetune_space.MICRO_BATCH) -- a per-run auto-probe would make "
             "every trial pay a doubling-until-OOM search whose answer never "
             "changes.",
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
    # timescale, and this script runs at very different step counts for a
    # tuning trial and a final run (at batch 512: 1953 steps for 1M samples,
    # 5004 for 2 epochs). Holding (1 - beta) * steps invariant is what makes
    # those comparable -- see
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
    # the same integration serve both tuning and final runs: the tuning trials
    # reach here through finetune_worker.trial_config and the final runs through
    # tuning.final_config, and both render a config dict into these flags.
    p.add_argument("--enable-mlflow", action="store_true")
    p.add_argument("--mlflow-experiment", default=None)
    p.add_argument("--mlflow-tracking-uri", default=None)
    p.add_argument("--label-smooth", type=float, default=0.0)
    p.add_argument("--num-update-iters", type=int, default=128,
                   help="CAVI iterations per macro batch (IBProbit arm).")
    p.add_argument("--bf16", action="store_true")

    # --- IBProbit head knobs ---
    # Same flag names, types and defaults as
    # full_finetune_comparison.build_argparser: probe_micro_batch_size ->
    # build_ibprobit reads these regardless of --arm, since the prober always
    # builds an IBProbit head to find the heavier micro-step's memory ceiling.
    # Do not invent defaults here -- matching keeps this script and
    # full_finetune_comparison.py at one operating point.
    p.add_argument("--px", action="store_true",
                   help="parameter-expanded CAVI scan (collapses the slow scale mode; far fewer iters)")
    p.add_argument("--no-learn-q", action="store_true",
                   help="pin Qhat at its q0 seed (skip the Inverse-Wishart recursion on Q)")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="process-noise scale: q0 = alpha^2 / effective_batch. Default 0.1; "
                        "at 1e-3 the recursion is measurably inert")
    p.add_argument("--q-init", type=float, default=None,
                   help="override q0 directly (bypasses alpha^2 / B; 1/D is the pre-Wishart value)")
    p.add_argument("--kappa", type=float, default=None,
                   help="Inverse-Wishart prior pseudo-count (default: 0.5, i.e. half a pseudo-observation against the K the likelihood adds per step)")
    p.add_argument("--layer-decay", type=float, default=protocol.LAYER_DECAY_OFF,
                   help="Layer-wise LR decay for the TRUNK: block i runs at "
                        "decay^(depth-1-i), so the top block is 1.0 and the "
                        "embeddings are damped hardest. 1.0, the default, is "
                        "off. Standard for "
                        "transformer fine-tuning; 0.65-0.75 is usual for a ViT.")
    p.add_argument("--layer-decay-span", type=float,
                   default=protocol.LAYER_DECAY_SPAN_OFF,
                   help="Layer-wise LR decay expressed as the SPAN: the fraction of the top block's rate that the EMBEDDINGS receive. Resolved against this backbone's actual depth, so one value is comparable across the dinov3 ladder -- unlike --layer-decay, whose per-block factor means 0.75^12 on a 12-block trunk but 0.75^32 on a 32-block one. Mutually exclusive with --layer-decay. 1.0 is off.")
    p.add_argument("--lr-decay-start", type=float, default=1.0,
                   help="Fraction of the run at which the linear LR decay "
                        "BEGINS (Warmup-Stable-Decay): 0.25 holds the peak to "
                        "25%% of the steps then ramps down over the remaining "
                        "75%%. NOT the length of the decay phase. 1.0, the "
                        "default, never decays (a constant LR). WSD "
                        "rather than cosine because the transfer rule's "
                        "hyperparameters were tuned at a constant LR: a WSD "
                        "run keeps that exact operating point for its stable "
                        "phase, a cosine run shares no phase with it.")
    p.add_argument("--lr-decay-to", type=float, default=0.5,
                   help="LR multiplier at the END of the decay, relative to "
                        "the peak. Applied to the head as well as the trunk.")
    p.add_argument("--warmup", type=int, default=1,
                   help="hold the contamination gate at 1 and defer the q recursion for the first N update steps")
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
             "test partition at every evaluation would roughly double a "
             "trial's cost.",
    )
    p.add_argument(
        "--dump-every-k-evals", type=int, default=0,
        help="Write a full dump on every k-th evaluation, giving Fig 3 its "
             "trajectory. 0 (the default, and what tuning uses) writes exactly "
             "ONE dump per run: the buffered argmin-val_nll evaluation. At "
             "1000 classes a val dump is 40 MB, so one per evaluation would "
             "be several GB per run.",
    )
    # Hand OOD to a separate process on a separate gpu instead of evaluating
    # it here. The schedule is unchanged -- --ood-every-k-dumps still decides
    # WHICH steps -- but at those steps this run writes a checkpoint and moves
    # on, and scripts/finetune/ood_worker.py does the 98,389-image evaluation.
    # That takes ~1 h per run off the TRAINING gpu, which keeps a long
    # dinov3_huge job inside a 24 h wall-clock limit, and it lets OOD run on a
    # finer stride than the trainer could otherwise afford.
    p.add_argument("--ood-checkpoint-root", default=None,
                   help="Write checkpoints for the OOD worker instead of "
                        "evaluating OOD inline.")
    p.add_argument("--ood-max-pending", type=int, default=2,
                   help="Skip a scheduled checkpoint when this many are already "
                        "waiting. A lagging worker must not fill scratch; a "
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
    """Delegates to imagenet.load_ood_sets, which both arms share.

    Kept as a thin wrapper because the tests and the worker call it by this
    name; `args` is accepted for their sake and not used by the loader.
    """
    return imagenet.load_ood_sets(img_size)


def evaluate_split(arm, backbone, head_or_loss, images_uint8, labels, *,
                   args, mean, std, key, subset=None):
    """Forward one preloaded eval set in chunks. Returns (logits, scalars).

    Deliberately does NOT write. Evaluation happens every `eval_quantum`
    samples -- dozens of times in a tuning run -- while at most a stride of
    those points is ever dumped, so computing and writing are kept apart. The
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
    # metrics.classification_metrics is correct on it. Under a
    # "moderated_score" label run_curve would refuse every IBProbit dump.
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
    ordering is the whole point: `child_run` names the run from `args.uid`, so
    a default computed inside `_train` would come too late and the run would be
    created with uid=None. Tuning passes uid=trialNNNNN, but a final run passes
    no uid at all, so every seed of every arm would land under one nameless
    run.

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
    bllarse.tuning.objective.evaluate passes, so the tuning loop drives this
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
        # not "imagenet1k". Folding {dataset} in ahead of {arm} does not keep
        # the field correct; it only keeps the dataset at the FRONT of it.
        #
        # Left as is because it is harmless here -- imagenet1k is the only
        # dataset, and the arm is independently in group_id and in
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
    # 30k test partition at every evaluation would roughly double a trial's
    # cost for a number nothing reads.
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
    # the 98,389-image preload must never be paid.
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

    # Read unconditionally: the span has to be resolved against the real depth
    # before we know whether LLRD is on at all. A stub backbone in a contract
    # test has no `.blocks` and must not be forced to grow one, hence getattr.
    _blocks = getattr(backbone, "blocks", None)
    trunk_depth = len(_blocks[0].blocks) if _blocks is not None else 0
    args.layer_decay = _resolve_layer_decay(args, trunk_depth)

    # Warmup-Stable-Decay. `--lr-decay-start 1.0` (the default) returns the scalar
    # unchanged, so the default path is exactly a constant-LR run.
    #
    # The SAME schedule object goes on the head as on the trunk: the head is
    # fitted to the trunk's features, and annealing one while the other keeps
    # moving would leave the head chasing a target at full speed with a
    # shrinking step. Both groups also take the same number of steps, which is
    # the same reason the betas are not split between them.
    lr_sched = protocol.wsd_schedule(
        args.lr, plan["steps"], decay_start=args.lr_decay_start,
        decay_to=args.lr_decay_to)
    if args.lr_decay_start < 1.0:
        mult = protocol.wsd_mean_multiplier(args.lr_decay_start, args.lr_decay_to)
        print(f"[finetune] WSD: constant for "
              f"{args.lr_decay_start * 100:.0f}% of {plan['steps']} steps, "
              f"then linear to {args.lr_decay_to:g}x. Mean lr is {mult:.3f}x the "
              f"peak, so total shrinkage steps*lr*wd is {mult:.3f}x the tuned "
              f"configuration's.", flush=True)

    # Layer-wise decay groups. At --layer-decay 1.0 every scale is 1.0 and this
    # collapses to the plain two-group (backbone, head) split; the extra optax
    # transforms are then all identical.
    llrd_on = args.layer_decay != protocol.LAYER_DECAY_OFF
    trunk_scales = (_trunk_lr_scales(trunk_depth, args.layer_decay) if llrd_on
                    else {"backbone": 1.0})
    if args.layer_decay != protocol.LAYER_DECAY_OFF:
        lo, hi = min(trunk_scales.values()), max(trunk_scales.values())
        print(f"[finetune] LLRD: decay={args.layer_decay:g} over {trunk_depth} "
              f"blocks, {len(trunk_scales)} groups, lr scale {lo:.4f}..{hi:.4f}. "
              f"The TOP block runs at 1.0 -- this damps the layers furthest "
              f"from the head.", flush=True)
    # With LLRD off the trunk is ONE group and ONE adamw -- built once, not
    # once per path. Two constructions would be invisible at runtime but make
    # the optimizer count a lie, which is what the CE contract test checks.
    trunk_tx = ({name: optax.adamw(_scaled(lr_sched, scale), b1=args.beta1,
                                   b2=args.beta2, weight_decay=args.weight_decay)
                 for name, scale in trunk_scales.items()}
                if llrd_on else
                {"backbone": optax.adamw(lr_sched, b1=args.beta1, b2=args.beta2,
                                         weight_decay=args.weight_decay)})

    if args.arm == "ce":
        head_lr = args.head_lr if args.head_lr is not None else DEFAULT_HEAD_LR
        head_sched = protocol.wsd_schedule(
            head_lr, plan["steps"], decay_start=args.lr_decay_start,
            decay_to=args.lr_decay_to)
        # A CALLABLE, not a pytree. optax's multi_transform accepts either,
        # but handing it a labels PYTREE built from an equinox Module makes it
        # treat the Module itself as a leaf and reach for `.shape` --
        # `AttributeError: 'VisionTransformer' object has no attribute 'shape'`.
        # Called with the real params, the structure matches by construction.
        param_labels = lambda p: (  # noqa: E731
            (_trunk_labels_of(p[0], trunk_depth) if llrd_on
             else jax.tree.map(lambda _: "backbone",
                               eqx.filter(p[0], eqx.is_array))),
            jax.tree.map(lambda _: "head", eqx.filter(p[1], eqx.is_array)),
        )
        optimizer = optax.multi_transform(
            {
                **trunk_tx,
                # Head weight decay is 0 on every arm: VBLL's KL is
                # its head regulariser and IBProbit runs no optimizer on the
                # head at all, so a nonzero value here would be an asymmetry.
                #
                # The betas are NOT similarly split: they set the optimizer's
                # averaging horizon in steps, and both groups take the same
                # number of steps.
                "head": optax.adamw(
                    head_sched, b1=args.beta1, b2=args.beta2, weight_decay=0.0),
            },
            param_labels,
        )
        opt_state = optimizer.init(eqx.filter((backbone, head), eqx.is_array))
    else:
        # multi_transform partitions by mapping over a labels pytree, which for
        # an equinox Module is itself a Module carrying strings -- fine for a
        # real equimo backbone, but reconstructing one trips the stubs the
        # contract tests drive. With LLRD off there is nothing to partition, so
        # the bare optimizer is used.
        optimizer = (
            optax.multi_transform(
                trunk_tx, lambda p: _trunk_labels_of(p, trunk_depth))
            if llrd_on else trunk_tx["backbone"])
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
    # tuning run to one 40 MB dump instead of one per evaluation.
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

            # Trajectory dumps for Fig 3. With --dump-every-k-evals 0 (tuning)
            # nothing is written here and the run's only disk write is the
            # buffered argmin, from _write_best.
            # The final evaluation always dumps, whatever the stride lands on.
            # curves._id_ood_pair raises when the last test dump and the last
            # OOD dump sit at different steps, and Table 1's converged endpoint
            # is read from exactly that point -- so a stride that misses the
            # end (len(evals) not 1 mod k) would break scoring rather than
            # merely thinning the trajectory.
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
                # and without the short-circuit the modulo would divide by zero
                # on every tuning trial.
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
            # has no persistent compilation cache configured, and at dozens of
            # evaluations per run that would dwarf the run itself.
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
    # never exists and this line would raise UnboundLocalError AFTER the run had
    # trained correctly and written its dump -- turning a good trial into a
    # FAILED one at the very last statement.
    print(f"[finetune] done: {eval_index} evaluations under "
          f"{args.dump_root}/{args.group_id}/{args.uid}")



def main(args, on_dump=None) -> None:
    """Public entry point: the training run, wrapped in one MLflow run.

    A thin wrapper so `_train` keeps its own `finally`. The signature is part
    of the tuning contract: `bllarse.tuning.objective.evaluate` calls
    `module.main(args, on_dump=dump_cb)`.

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
