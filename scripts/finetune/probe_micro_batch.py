"""Measure the largest per-forward micro-batch that fits, per arm, per GPU.

Run this ONCE on the target hardware and pin the answer in
`bllarse.tuning.finetune_space.MICRO_BATCH`. The per-run auto-probe it replaces
made every trial pay a doubling-until-OOM search whose answer never changed,
and returned a value that could not be reproduced from the config afterwards.

The result is not the raw ceiling. `accum_plan` requires the micro batch to
DIVIDE the target batch exactly -- rounding used to yield an effective batch
that was not the tuned one (4096 over micro 100 gave 4100), silently
mislabelling every point of a samples-seen axis -- so the measurement is
floored to a power of two and capped at the batch ladder's gcd. See
`floor_to_ladder_divisor`.

The number is only valid for the GPU it was measured on, so the run prints the
device and its memory alongside.
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
import optax  # noqa: E402
from jax import random as jr  # noqa: E402

from bllarse.finetune import jax_steps  # noqa: E402
from bllarse.losses import CrossEntropy  # noqa: E402

# The training script itself, for its jitted eval entry points. Imported rather
# than reimplemented: a local copy would drift, and the whole value of this
# probe is that it measures the SAME compiled functions the run executes.
# This file lives beside it, so sys.path[0] already covers the import.
import finetune_jax  # noqa: E402

# The batch ladder's gcd. Every rung (256, 512, 1024, 2048, 4096) is a multiple
# of 256, so a micro batch that divides 256 divides all of them -- and one that
# does not is unusable at the bottom rung whatever it measures.
LADDER_GCD = 256

# Substrings JAX/XLA uses to report an out-of-memory allocation failure. Matched
# rather than caught by type because XLA raises several unrelated classes for
# the same condition depending on the backend.
_OOM_MARKERS = ("RESOURCE_EXHAUSTED", "out of memory", "OOM")


def floor_to_ladder_divisor(probed: int, gcd: int = LADDER_GCD) -> int:
    """The largest usable micro batch no greater than `probed`.

    Floored to a power of two and capped at the ladder's gcd, so
    `num_accum = target / micro` is an exact integer at every rung.

    Capping rather than raising when the ceiling exceeds the gcd is deliberate:
    a GPU that fits 512 images per forward simply runs the bottom rung as a
    single un-accumulated step, which is correct, not a configuration to
    reject.
    """
    if probed < 1:
        raise ValueError(f"probed must be >= 1, got {probed}")
    power = 1
    while power * 2 <= probed:
        power *= 2
    return min(power, gcd)


def build_optimizer(arm, backbone, head, *, lr, head_lr, weight_decay):
    """The optimizer and its initialised state, exactly as finetune_jax builds them.

    Allocating this is the whole point: AdamW carries two moment buffers, so
    `optimizer.init` puts 2 x params on the device (~6.7 GB for dinov3_huge's
    840M fp32 parameters) BEFORE the training loop runs a single micro-step.
    """
    if arm == "ibprobit":
        optimizer = optax.adamw(lr, weight_decay=weight_decay)
        return optimizer, optimizer.init(eqx.filter(backbone, eqx.is_array))

    param_labels = (
        jax.tree.map(lambda _: "backbone", eqx.filter(backbone, eqx.is_array)),
        jax.tree.map(lambda _: "head", eqx.filter(head, eqx.is_array)),
    )
    optimizer = optax.multi_transform(
        {
            "backbone": optax.adamw(lr, weight_decay=weight_decay),
            # Head weight decay is 0 on every arm (design D6).
            "head": optax.adamw(head_lr, weight_decay=0.0),
        },
        param_labels,
    )
    return optimizer, optimizer.init(eqx.filter((backbone, head), eqx.is_array))


def probe_ceiling(arm, backbone, loss_fn, head, make_optimizer, *,
                  img_size, init_bs, max_bs, eval_chunk, key, bf16,
                  num_update_iters) -> int:
    """Double the micro-batch until a FULL optimizer step OOMs.

    Returns the last size that fit.

    A full step, not a bare micro-grad. The first version of this probe
    measured `ce_micro_grad` alone and reported 64 on a card where the real
    loop OOMs at 64: by the time the loop calls that kernel it is already
    holding AdamW's two moment buffers and a `grads_acc` accumulation buffer,
    ~13.5 GB of resident state for dinov3_huge that the isolated kernel never
    allocated. The probe fit 26 GB into 40; the run needed 36.

    So this reproduces the peak the loop actually reaches:

      * `opt_state` is live throughout (the caller allocates it),
      * TWO micro-batches are accumulated, so `grads_acc` exists and
        `_tree_add` runs against a real second gradient tree,
      * `apply_grads` runs, which is where the update's transient copies land,
      * an eval chunk is forwarded, since evaluation happens inside the same
        process with all of the above still resident.

    `num_accum = 2` rather than the real ladder value: `grads_acc` is ONE
    param-sized buffer no matter how many micro-steps fold into it, so two is
    enough to allocate it and to exercise the add.

    `make_optimizer` is a factory called ONCE PER ATTEMPT, not a prebuilt
    state. `apply_grads` is `@eqx.filter_jit(donate="all-except-first")`, so it
    DONATES `opt_state` -- reusing one across attempts hands the second attempt
    a freed buffer and fails with `Array has been deleted with shape=int32[]`
    (the AdamW step counter) rather than with an OOM. The model is not donated,
    so backbone and head are safe to reuse.
    """
    bs = init_bs
    last_ok = None
    while bs <= max_bs:
        # Rebound each attempt so `finally` can drop them even if the try body
        # raised before assigning -- a bare `del` on an unbound name would
        # mask the real exception with UnboundLocalError.
        grads_acc = trial_backbone = trial_head = trial_loss = trial_state = None
        try:
            images = jnp.zeros((bs, 3, img_size, img_size), dtype=jnp.float32)
            labels = jnp.zeros((bs,), dtype=jnp.int32)
            trial_backbone, trial_head = backbone, head
            optimizer, trial_state = make_optimizer()
            trial_loss = loss_fn
            fts = []

            for _ in range(2):
                if arm == "ibprobit":
                    grads, f, _ = jax_steps.ibprobit_micro_forward(
                        trial_backbone, trial_loss, images, labels, key, bf16=bf16)
                    fts.append(f)
                else:
                    grads, _ = jax_steps.ce_micro_grad(
                        (trial_backbone, trial_head), trial_loss, images, labels,
                        key, bf16=bf16)
                grads_acc = grads if grads_acc is None else jax_steps._tree_add(
                    grads, grads_acc)

            if arm == "ibprobit":
                trial_backbone, trial_state = jax_steps.apply_grads(
                    trial_backbone, grads_acc, trial_state, optimizer, 2)
                trial_loss = trial_loss.update(
                    jnp.concatenate(fts), jnp.concatenate([labels, labels]),
                    num_iters=num_update_iters)
                head_or_loss = trial_loss
            else:
                (trial_backbone, trial_head), trial_state = jax_steps.apply_grads(
                    (trial_backbone, trial_head), grads_acc, trial_state,
                    optimizer, 2)
                head_or_loss = trial_head

            # Evaluation runs in the same process with all of the above live.
            #
            # Through finetune_jax's OWN jitted entry points, not a local
            # re-implementation. Calling jax_steps.forward_features directly is
            # not equivalent: outside jit, `bf16=True` materialises a full
            # bf16 copy of the 840M-param backbone (~1.7 GB) alongside the fp32
            # master weights instead of fusing the cast, and every one of the
            # 32 layers' activations for `eval_chunk` images stays live with no
            # fusion and no early free. That inflates the eval leg enough to
            # depress the reported ceiling below what the run actually needs --
            # the probe would be measuring its own un-jitted eval rather than
            # the training step.
            eval_images = jnp.zeros(
                (eval_chunk, 3, img_size, img_size), dtype=jnp.float32)
            inference_bb = eqx.nn.inference_mode(trial_backbone, True)
            if arm == "ibprobit":
                logits = finetune_jax.eval_logits_ibprobit(
                    inference_bb, head_or_loss, eval_images,
                    jnp.zeros((eval_chunk,), dtype=jnp.int32), key, bf16=bf16)
            else:
                logits = finetune_jax.eval_logits_ce(
                    inference_bb, head_or_loss, eval_images, key, bf16=bf16)

            jax.block_until_ready(logits)
            last_ok = bs
            peak = jax.local_devices()[0].memory_stats() or {}
            peak_gib = peak.get("peak_bytes_in_use", 0) / 2**30
            print(f"[probe] {arm} micro_batch_size={bs} OK "
                  f"(full step + eval chunk {eval_chunk}), "
                  f"peak {peak_gib:.2f} GiB", flush=True)
        except Exception as exc:
            message = str(exc)
            if any(m in message for m in _OOM_MARKERS) or \
                    "out of memory" in message.lower():
                print(f"[probe] {arm} micro_batch_size={bs} OOM, backing off",
                      flush=True)
                break
            raise
        finally:
            del grads_acc, trial_backbone, trial_head, trial_loss, trial_state
            jax.clear_caches()
            gc.collect()
        bs *= 2

    if last_ok is None:
        raise RuntimeError(
            f"Even micro_batch_size={init_bs} does not fit for arm={arm}."
        )
    return last_ok


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=["ce", "ibprobit"])
    p.add_argument("--model", default="dinov3_huge")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size-init", type=int, default=8)
    p.add_argument("--batch-size-max", type=int, default=2048)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--label-smooth", type=float, default=0.0)
    p.add_argument("--num-update-iters", type=int, default=256,
                   help="CAVI iterations, at the ladder's TOP rung: the whole "
                        "point is to measure the worst case that tuning can "
                        "sample, not a typical one.")
    p.add_argument("--eval-chunk", type=int, default=256,
                   help="Eval forward width, matching finetune_jax's default. "
                        "Evaluation happens in the same process with the "
                        "optimizer state still resident, so it belongs in the "
                        "measurement.")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Trunk rate. Irrelevant to memory, but the optimizer "
                        "must be built to allocate its moment buffers.")
    p.add_argument("--head-lr", type=float, default=1e-1)
    p.add_argument("--weight-decay", type=float, default=1e-1)
    p.add_argument("--seed", type=int, default=137)
    return p


def main(args) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dinov3"))
    from vit_classification import EQUIMO_MODELS, get_pretrained_backbone

    device = jax.local_devices()[0]
    print(f"[probe] device={device.device_kind} platform={device.platform}")

    embed_dim = EQUIMO_MODELS[args.model]["embed_dim"]
    key = jr.PRNGKey(args.seed)
    key, model_key, head_key, fwd_key = jr.split(key, 4)
    backbone = get_pretrained_backbone(args.model, model_key)

    head, loss_fn = None, None
    if args.arm == "ce":
        loss_fn = CrossEntropy(args.label_smooth, args.num_classes)
        head = eqx.nn.Linear(embed_dim, args.num_classes, key=head_key)
    else:
        from bllarse.losses import IBProbit
        # batch_size here only seeds q0 = alpha^2 / B; the probe measures
        # memory, not fit, so any ladder-legal value gives the same footprint.
        loss_fn = IBProbit(embed_dim, args.num_classes, key=head_key,
                           batch_size=LADDER_GCD)

    # A FACTORY, not a prebuilt state: apply_grads donates opt_state, so each
    # attempt needs its own. The allocation itself is the point -- AdamW's two
    # moment buffers are ~2x params of resident state (~6.7 GB on dinov3_huge)
    # that the training loop never hands back, and omitting them is what made
    # the first version of this probe report a micro-batch the real run OOMs on.
    def make_optimizer():
        return build_optimizer(
            args.arm, backbone, head,
            lr=args.lr, head_lr=args.head_lr, weight_decay=args.weight_decay)

    probed = probe_ceiling(
        args.arm, backbone, loss_fn, head, make_optimizer,
        img_size=args.img_size, init_bs=args.batch_size_init,
        max_bs=args.batch_size_max, eval_chunk=args.eval_chunk,
        key=fwd_key, bf16=args.bf16, num_update_iters=args.num_update_iters,
    )
    pinned = floor_to_ladder_divisor(probed)
    print(
        f"\n[probe] arm={args.arm} model={args.model} img_size={args.img_size} "
        f"bf16={args.bf16}\n"
        f"[probe] measured ceiling: {probed}\n"
        f"[probe] PIN THIS: MICRO_BATCH[{args.arm!r}] = {pinned}\n"
        f"[probe] (floored to a power of two, capped at the ladder gcd "
        f"{LADDER_GCD}; valid only for {device.device_kind})"
    )


if __name__ == "__main__":
    from jax import config as jax_config

    parsed = build_argparser().parse_args()
    jax_config.update("jax_platform_name", os.environ.get("JAX_PLATFORM_NAME", "gpu"))
    main(parsed)
