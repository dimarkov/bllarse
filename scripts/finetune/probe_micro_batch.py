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
from jax import random as jr  # noqa: E402

from bllarse.finetune import jax_steps  # noqa: E402
from bllarse.losses import CrossEntropy  # noqa: E402

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


def probe_ceiling(arm, backbone, loss_fn, head, *, embed_dim, num_classes,
                  img_size, init_bs, max_bs, key, bf16) -> int:
    """Double the micro-batch until the arm's micro-step OOMs.

    Returns the last size that fit. Each attempt clears the compilation cache
    and collects, so a failed attempt's buffers do not count against the next.
    """
    bs = init_bs
    last_ok = None
    while bs <= max_bs:
        try:
            images = jnp.zeros((bs, 3, img_size, img_size), dtype=jnp.float32)
            labels = jnp.zeros((bs,), dtype=jnp.int32)
            if arm == "ibprobit":
                _, _, out = jax_steps.ibprobit_micro_forward(
                    backbone, loss_fn, images, labels, key, bf16=bf16)
            else:
                _, out = jax_steps.ce_micro_grad(
                    (backbone, head), loss_fn, images, labels, key, bf16=bf16)
            jax.block_until_ready(out)
            last_ok = bs
            print(f"[probe] {arm} micro_batch_size={bs} OK", flush=True)
        except Exception as exc:
            if any(marker in str(exc) for marker in _OOM_MARKERS) or \
                    "out of memory" in str(exc).lower():
                print(f"[probe] {arm} micro_batch_size={bs} OOM, backing off",
                      flush=True)
                break
            raise
        finally:
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
    p.add_argument("--num-update-iters", type=int, default=192)
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

    probed = probe_ceiling(
        args.arm, backbone, loss_fn, head,
        embed_dim=embed_dim, num_classes=args.num_classes,
        img_size=args.img_size, init_bs=args.batch_size_init,
        max_bs=args.batch_size_max, key=fwd_key, bf16=args.bf16,
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
