"""Full-network fine-tuning of DINOv3 on ImageNet-1k -- JAX arms.

One arm per invocation (`--arm ce|ibprobit`), writing eval dumps on a log-spaced
schedule so temperature scaling, Brier and OOD detection are all computed post
hoc by the same code that scores the probing figures.

This is NOT scripts/alternatives/full_finetune_comparison.py, which stays as the
IVON/EVON benchmarking harness with inline metrics. The two share their jitted
kernels via bllarse.finetune.jax_steps.
"""

import argparse
import os

from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.secrets", override=False)

from bllarse.finetune import imagenet, protocol  # noqa: E402

# Fine-tuning resolution. DINOV3_DATASET_IMG_SIZES maps imagenet1k -> 512, but
# that table exists for FROZEN FEATURE EXTRACTION; fine-tuning at 512 is ~5x the
# cost for no benefit, so inheriting it must be impossible by accident.
FINETUNE_IMG_SIZE = 224


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
    return p


def main(args, on_dump=None) -> None:
    """Fine-tune one arm. Filled in by Task 5.

    `on_dump(dump_index, step, scalars)` matches the hook
    bllarse.tuning.objective.evaluate passes, so the Phase 2 tuning campaign can
    drive this script without a wrapper.
    """
    check_img_size(args)
    check_arm_flags(args)
    raise NotImplementedError("training loop lands in Task 5")


if __name__ == "__main__":
    from jax import config as jax_config

    parsed = build_argparser().parse_args()
    jax_config.update("jax_platform_name", parsed.device)
    main(parsed)
