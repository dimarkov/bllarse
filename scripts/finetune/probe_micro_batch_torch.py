#!/usr/bin/env python3
"""Peak GPU memory for the VBLL arm, at the micro batch the campaign runs.

    python scripts/finetune/probe_micro_batch_torch.py --model dinov3_large \
        --peak-at 64 --json-out vbll_mem.json

The JAX companion (`probe_micro_batch.py`) covers `ce` and `ibprobit`. VBLL is
the torch arm and shares none of that machinery -- torch.optim.AdamW instead of
optax, `transformers` instead of equimo -- so it needs its own measurement, and
Fig 7c is incomplete without it.

Faithful to the training loop in the ways that move the number:

  * model and head come from `finetune_vbll.build_model`, the ONE place either
    is constructed, so a changed cov_rank or prior_scale cannot silently make
    this measure a model that never trained;
  * the optimizer is built and STEPPED, because AdamW's two moment buffers are
    ~2x params of resident state that the loop never gives back;
  * TWO micro-batches accumulate before the step, so the gradient buffers are
    live at the same time the update's transients are;
  * an eval chunk is forwarded afterwards with all of the above resident, since
    evaluation happens in the same process;
  * the head runs in fp32 even under --bf16, exactly as `evaluate_split` does.

Weights are the real pretrained checkpoint rather than random: unlike the JAX
`bench_step`, `build_model` validates hidden_size against the registry, and
there is no pretrained=False path through `AutoModel.from_pretrained` that
keeps that check meaningful.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import finetune_vbll  # noqa: E402


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="dinov3_large")
    p.add_argument("--peak-at", type=int, required=True,
                   help="Micro batch to measure. Use the campaign's value from "
                        "finetune_space.MICRO_BATCH[model]['vbll'].")
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--eval-chunk", type=int, default=256)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    # Head shape knobs. They change the head's parameter count, so they change
    # the number -- defaults mirror finetune_vbll's own.
    p.add_argument("--kl-ratio", type=float, default=1.0)
    p.add_argument("--prior-scale", type=float, default=1.0)
    p.add_argument("--wishart-scale", type=float, default=1.0)
    p.add_argument("--parameterization", default="dense")
    p.add_argument("--cov-rank", type=int, default=1)
    p.add_argument("--vbll-type", default="disc")
    p.add_argument("--json-out", default=None)
    return p


def main(args):
    if not torch.cuda.is_available():
        raise SystemExit("[probe-torch] no CUDA device; the number is "
                         "meaningless off the target hardware")
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    print(f"[probe-torch] device={name}")

    from bllarse.finetune.registry import EQUIMO_MODELS
    embed_dim = EQUIMO_MODELS[args.model]["embed_dim"]

    model, head = finetune_vbll.build_model(
        args, embed_dim=embed_dim, num_classes=args.num_classes, device=device)
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": args.lr,
         "weight_decay": args.weight_decay},
        {"params": head.parameters(), "lr": args.head_lr, "weight_decay": 0.0},
    ], betas=(args.beta1, args.beta2))

    micro = args.peak_at
    images = torch.zeros((micro, 3, args.img_size, args.img_size),
                         device=device)
    labels = torch.zeros((micro,), dtype=torch.long, device=device)

    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    for _ in range(2):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            feats = finetune_vbll.trunk_features(model, images)
        loss = head(feats.float()).train_loss_fn(labels) / 2
        loss.backward()
    optimizer.step()

    eval_images = torch.zeros(
        (args.eval_chunk, 3, args.img_size, args.img_size), device=device)
    model.eval()
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            eval_feats = finetune_vbll.trunk_features(model, eval_images)
        # fp32 head, matching evaluate_split: the predictive covariance path is
        # delicate and bf16 would run it at a precision training never used.
        head(eval_feats.float())
    torch.cuda.synchronize()

    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    print(f"[probe-torch] arm=vbll model={args.model} micro_batch={micro} "
          f"peak {peak_gib:.2f} GiB")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "model": args.model, "arm": "vbll", "micro_batch": micro,
            "peak_gib": peak_gib, "device": name, "bf16": bool(args.bf16),
            "img_size": args.img_size, "eval_chunk": args.eval_chunk,
        }, indent=1))
        print(f"[probe-torch] wrote {args.json_out}")


if __name__ == "__main__":
    main(build_argparser().parse_args())
