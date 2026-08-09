"""Time ONE fine-tuning optimizer step, and say what fraction of the GPU it uses.

Built to run anywhere, including a workstation with no dataset and no HF cache:
the backbone is constructed with `pretrained=False`, so the weights are random
but every SHAPE is identical to a real run. Memory and FLOPs do not depend on
the values in the weights.

Why it exists: the campaign's first real trial measured 40 img/s on an A100,
which is ~13% of the device's dense bf16 throughput. That is not a plausible
number for a well-formed ViT step, and finding out why on the cluster costs a
queue wait per hypothesis. This makes the same measurement locally in minutes.

Absolute img/s does not transfer between GPUs. What transfers is the RATIO to
the device's own measured matmul ceiling, and whether a change moves it.
"""

import argparse
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import random as jr

from bllarse.finetune import equimo_vision_models, jax_steps
from bllarse.losses import CrossEntropy

# equimo factory per backbone name, and the FLOP-relevant shape facts. Depth
# and width are recorded so the FLOP estimate is auditable rather than a magic
# constant; token count is (img_size/patch)^2 + cls + registers.
MODELS = {
    "dinov3_large": dict(factory="dinov3_vitl16_pretrain_lvd1689m",
                         depth=24, width=1024, patch=16, extra_tokens=5),
    "dinov3_huge": dict(factory="dinov3_vith16plus_pretrain_lvd1689m",
                        depth=32, width=1280, patch=16, extra_tokens=5),
}


def device_matmul_ceiling(dtype=jnp.bfloat16, n=4096, iters=20) -> float:
    """Measured dense matmul TFLOP/s -- the honest denominator for MFU.

    Vendor peak numbers assume conditions a real workload never sees. The
    fraction that matters is 'of what this device actually delivers on a big
    square matmul', so that is what is measured.
    """
    x = jnp.ones((n, n), dtype)
    (x @ x).block_until_ready()
    start = time.perf_counter()
    for _ in range(iters):
        y = x @ x
    y.block_until_ready()
    elapsed = time.perf_counter() - start
    return iters * 2 * n**3 / elapsed / 1e12


def step_flops(spec, micro: int, img_size: int) -> float:
    """FLOPs in one fwd+bwd over `micro` images.

    2 * params_applied_per_token * tokens for the forward, and backward is
    conventionally 2x forward, giving the 3x. Attention's quadratic term is
    included separately: at 201 tokens it is small, but omitting it silently
    would flatter the MFU number.
    """
    tokens = (img_size // spec["patch"]) ** 2 + spec["extra_tokens"]
    width, depth = spec["width"], spec["depth"]
    # Per layer: qkv (3 w^2), attn out (w^2), MLP (8 w^2 at ratio 4).
    per_token = depth * 12 * width**2
    attn = depth * 2 * 2 * tokens**2 * width  # scores + weighted sum
    return 3.0 * micro * (2 * per_token * tokens + attn)


def build(model: str, num_classes: int, key):
    # Via the shared shim: equimo 1.1.1 moved `equimo.models` to
    # `equimo.vision.models`, and hardcoding either path breaks on the machine
    # with the other version. See bllarse.finetune.equimo_vision_models.
    equimo_models = equimo_vision_models()
    spec = MODELS[model]
    # pretrained=False: random weights, identical shapes. The whole point --
    # this runs with no HF cache and no network.
    backbone = getattr(equimo_models, spec["factory"])(pretrained=False)
    backbone = eqx.nn.inference_mode(backbone, True)
    head = eqx.nn.Linear(spec["width"], num_classes, key=key)
    return backbone, head, spec


def bench(model, micro, num_accum, img_size, num_classes, steps, bf16, seed=0):
    key = jr.PRNGKey(seed)
    key, head_key = jr.split(key)
    backbone, head, spec = build(model, num_classes, head_key)

    loss_fn = CrossEntropy(0.0, num_classes)
    param_labels = (
        jax.tree.map(lambda _: "backbone", eqx.filter(backbone, eqx.is_array)),
        jax.tree.map(lambda _: "head", eqx.filter(head, eqx.is_array)),
    )
    optimizer = optax.multi_transform(
        {"backbone": optax.adamw(1e-4, weight_decay=1e-2),
         "head": optax.adamw(1e-2, weight_decay=0.0)},
        param_labels,
    )
    opt_state = optimizer.init(eqx.filter((backbone, head), eqx.is_array))

    images = jnp.zeros((micro, 3, img_size, img_size), jnp.float32)
    labels = jnp.zeros((micro,), jnp.int32)

    def one_step(backbone, head, opt_state, key):
        grads_acc = None
        for _ in range(num_accum):
            grads, _ = jax_steps.ce_micro_grad(
                (backbone, head), loss_fn, images, labels, key, bf16=bf16)
            grads_acc = grads if grads_acc is None else jax_steps._tree_add(
                grads, grads_acc)
        return jax_steps.apply_grads(
            (backbone, head), grads_acc, opt_state, optimizer, num_accum)

    # Warmup: compiles, and is excluded from the timing exactly as the training
    # loop excludes its own step 1.
    (backbone, head), opt_state = one_step(backbone, head, opt_state, key)
    jax.block_until_ready(opt_state)

    start = time.perf_counter()
    for i in range(steps):
        (backbone, head), opt_state = one_step(backbone, head, opt_state, key)
    jax.block_until_ready(opt_state)
    elapsed = time.perf_counter() - start

    per_step = elapsed / steps
    images_per_step = micro * num_accum
    return {
        "img_per_s": images_per_step / per_step,
        "ms_per_step": per_step * 1e3,
        "tflops": step_flops(spec, images_per_step, img_size) / per_step / 1e12,
    }


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="dinov3_large", choices=sorted(MODELS))
    p.add_argument("--micro", type=int, default=32)
    p.add_argument("--num-accum", type=int, default=4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--sweep-micro", default="",
                   help="Comma-separated micro batches to sweep, e.g. 16,32,64. "
                        "num_accum is adjusted to hold the effective batch fixed.")
    return p


def main():
    args = build_argparser().parse_args()
    ceiling = device_matmul_ceiling()
    device = jax.local_devices()[0]
    print(f"[bench] device      : {device.device_kind}")
    print(f"[bench] matmul ceil : {ceiling:.1f} TFLOP/s (bf16, 4096^3)")
    print(f"[bench] model       : {args.model} @ {args.img_size}px, "
          f"bf16={not args.no_bf16}")

    micros = ([int(m) for m in args.sweep_micro.split(",")]
              if args.sweep_micro else [args.micro])
    effective = args.micro * args.num_accum
    print(f"[bench] effective batch held at {effective}\n")
    print(f"  {'micro':>6s} {'accum':>6s} {'ms/step':>10s} {'img/s':>9s} "
          f"{'TFLOP/s':>9s} {'of ceiling':>11s}")
    for micro in micros:
        accum = max(1, effective // micro)
        try:
            out = bench(args.model, micro, accum, args.img_size,
                        args.num_classes, args.steps, not args.no_bf16)
        except Exception as exc:
            marker = "OOM" if "RESOURCE_EXHAUSTED" in str(exc) else type(exc).__name__
            print(f"  {micro:6d} {accum:6d} {marker:>10s}")
            continue
        print(f"  {micro:6d} {accum:6d} {out['ms_per_step']:10.1f} "
              f"{out['img_per_s']:9.1f} {out['tflops']:9.1f} "
              f"{100 * out['tflops'] / ceiling:10.1f}%")


if __name__ == "__main__":
    main()
