"""Do the JAX and PyTorch DINOv3 trunks produce the same features?

The gate the VBLL fine-tuning arm has to pass before it is allowed to spend an
A100-hour. The VBLL arm is PyTorch (the reference `vbll` package is PyTorch, and
reimplementing a baseline is not something a paper should do), while CE and
IBProbit are JAX/equimo. That puts a SECOND trunk implementation in the
comparison, and if the two disagree then a difference the paper attributes to
the head is really a difference in the backbone.

Both sides derive from the same Meta release -- equimo converts the official
checkpoint to `weights.eqx`, transformers converts it to its own format -- so
they SHOULD agree to floating-point noise. "Should" is the reason this script
exists.

The two frameworks cannot share an environment (jax[cuda] and torch each ship a
full CUDA stack and fight over it), so this runs in three steps:

    # in the jax venv
    python scripts/finetune/trunk_parity.py emit --framework jax  --out /tmp/jax.npy
    # in the torch venv
    python scripts/finetune/trunk_parity.py emit --framework torch --out /tmp/torch.npy
    # either one
    python scripts/finetune/trunk_parity.py compare /tmp/jax.npy /tmp/torch.npy

Inputs come from the built uint8 shard, so both sides see byte-identical
images -- no decode, no resize, no colour-space difference to argue about.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# The rows compared. Fixed, small, and taken from the front of the shard so the
# check is reproducible and costs one page-in rather than a scan.
DEFAULT_ROWS = 32

# equimo's ViT returns the CLS token (jax_steps.forward_features takes out[0]
# when the backbone returns a tuple). Anything else on the torch side -- mean
# patch token, pooler output -- is a different feature and would show up here as
# a total mismatch rather than a subtle one.
POOLING = "cls"

HF_CHECKPOINTS = {
    "dinov3_small": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "dinov3_big": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3_huge": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
}


def _images(args):
    """The same uint8 rows, normalized the same way, for both frameworks.

    Normalisation happens HERE rather than in each framework's branch so a
    mean/std transcription error cannot masquerade as a trunk mismatch.
    """
    # bllarse.finetune.registry, NOT vit_classification: the latter imports jax
    # at module scope, and the torch side of this comparison runs in an
    # environment that deliberately has none.
    from bllarse.finetune import imagenet, shards
    from bllarse.finetune.registry import IMAGENET_MEAN, IMAGENET_STD

    reader = shards.open_shard(args.shards, args.dataset, "train", args.img_size)
    raw = reader.take(np.arange(args.rows))
    reader.close()
    return imagenet.normalize_chunk(raw, IMAGENET_MEAN, IMAGENET_STD)


def emit_jax(args, images):
    import equinox as eqx
    import jax.random as jr

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dinov3"))
    from vit_classification import get_pretrained_backbone

    from bllarse.finetune import jax_steps

    key = jr.PRNGKey(0)
    backbone = eqx.nn.inference_mode(get_pretrained_backbone(args.model, key), True)
    # bf16=False: this compares the TRUNKS, and bf16 would fold ~3 decimal
    # digits of cast noise into the very number being measured. The campaign
    # runs bf16 on both arms, so it is a shared choice rather than a difference.
    features = jax_steps.forward_features(backbone, images, key, bf16=False)
    return np.asarray(features, dtype=np.float32)


def emit_torch(args, images):
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(HF_CHECKPOINTS[args.model]).eval()
    if torch.cuda.is_available():
        model = model.cuda()

    # Already NCHW: imagenet.normalize_chunk transposes NHWC -> (N, 3, H, W)
    # itself, because equimo's models are channels_first. Transposing again
    # here fed torch a (32, 224, 3, 224) tensor and it said so -- the good
    # failure, and the reason both sides share one normalisation path.
    batch = torch.from_numpy(np.ascontiguousarray(images))
    if torch.cuda.is_available():
        batch = batch.cuda()

    with torch.no_grad():
        out = model(pixel_values=batch)

    # MEAN OF THE PATCH TOKENS -- not the CLS token, and not pooler_output.
    #
    # Measured, not assumed. Against the JAX reference on the same 32 rows:
    # mean patch gives cosine 1.000000 and max relative difference 2.2e-6,
    # while CLS and pooler_output (which are the same vector) give cosine
    # 0.416 and mean-over-all-tokens gives 0.9995. Only one of those is the
    # same feature.
    #
    # It is worth stating because the JAX side LOOKS like it takes CLS --
    # jax_steps.forward_features indexes out[0] and used to call it the CLS
    # token in a comment. It is equimo's pooled output, and equimo pools by
    # averaging patch tokens.
    n_skip = 1 + int(getattr(model.config, "num_register_tokens", 0) or 0)
    features = out.last_hidden_state[:, n_skip:].mean(dim=1)
    return features.float().cpu().numpy()


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Absolute, relative and directional agreement.

    Three numbers because they fail differently. A scale difference (one side
    missing a final norm) leaves cosine at 1.0 while abs error explodes; a
    wrong token leaves all three bad; genuine float noise leaves all three
    good. Reporting only one of them would hide two of the three cases.
    """
    if a.shape != b.shape:
        raise SystemExit(f"shape mismatch: {a.shape} vs {b.shape}")
    diff = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b)).max()
    cos = (
        (a * b).sum(axis=1)
        / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)
    )
    return {
        "shape": list(a.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_rel_diff": float(diff.max() / (scale + 1e-30)),
        "min_cosine": float(cos.min()),
        "mean_cosine": float(cos.mean()),
        "jax_norm": float(np.linalg.norm(a, axis=1).mean()),
        "torch_norm": float(np.linalg.norm(b, axis=1).mean()),
    }


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    emit = sub.add_parser("emit", help="forward the fixed rows in one framework")
    emit.add_argument("--framework", required=True, choices=["jax", "torch"])
    emit.add_argument("--model", default="dinov3_large")
    emit.add_argument("--dataset", default="imagenet1k")
    emit.add_argument("--img-size", type=int, default=224)
    emit.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    emit.add_argument("--shards", required=True)
    emit.add_argument("--out", required=True)

    cmp_ = sub.add_parser("compare", help="report agreement between two emissions")
    cmp_.add_argument("jax_npy")
    cmp_.add_argument("torch_npy")
    # 1e-3 on the RELATIVE difference, with cosine above 0.9999. Both sides run
    # fp32 here, so the residual is accumulation order across two different
    # kernel libraries -- which lands near 1e-5 when the implementations agree
    # and nowhere near it when they do not.
    cmp_.add_argument("--max-rel-diff", type=float, default=1e-3)
    cmp_.add_argument("--min-cosine", type=float, default=0.9999)
    return p


def main():
    args = build_argparser().parse_args()

    if args.command == "emit":
        images = _images(args)
        emit = emit_jax if args.framework == "jax" else emit_torch
        features = emit(args, images)
        np.save(args.out, features)
        print(f"[parity] {args.framework}: {features.shape} -> {args.out}")
        print(f"[parity] mean L2 norm {np.linalg.norm(features, axis=1).mean():.4f}")
        return

    report = compare(np.load(args.jax_npy), np.load(args.torch_npy))
    print(json.dumps(report, indent=2))
    failures = []
    if report["max_rel_diff"] > args.max_rel_diff:
        failures.append(
            f"max_rel_diff {report['max_rel_diff']:.2e} > {args.max_rel_diff:.0e}")
    if report["min_cosine"] < args.min_cosine:
        failures.append(
            f"min_cosine {report['min_cosine']:.6f} < {args.min_cosine}")
    if failures:
        raise SystemExit(
            "[parity] TRUNKS DISAGREE: " + "; ".join(failures) + "\n"
            "The VBLL arm would not be comparable to CE and IBProbit. Do not "
            "run the campaign until this passes -- a head-level conclusion "
            "drawn across disagreeing backbones is not a head-level conclusion."
        )
    print("[parity] OK: the two trunks agree within tolerance.")


if __name__ == "__main__":
    main()
