"""Full-network fine-tuning with a VBLL head, in PyTorch.

The fourth arm of the fine-tuning campaign, and the only one not in JAX: the
reference `vbll` package is PyTorch, and a paper should not reimplement the
baseline it compares against. The cost is a second trunk implementation, which
is why `scripts/finetune/trunk_parity.py` exists and must pass before this runs
-- it scores this file's backbone against equimo's on identical inputs
(currently max relative difference 2.1e-06, cosine 0.9999999).

WHAT IS SHARED WITH THE JAX ARMS, and why each one has to be:

  * the batch order -- protocol.shuffled_indices over row positions, so this
    arm and finetune_jax consume an identical stream at the same seed. numpy,
    not torch.Generator, for exactly this reason.
  * the training data -- the same pre-decoded uint8 shard, read through the
    same ShardReader, so neither arm can differ by a decode or a resize.
  * the evaluation partition -- imagenet.pinned_eval_split, so val and test
    mean the same rows here as there.
  * the budget arithmetic -- protocol.steps_for_samples / eval_every_steps, so
    eval index i is the same samples_seen in every arm of every study, which
    is what makes it usable as a MedianPruner key.
  * the dump format -- bllarse.evaldump, so one scoring path reads all four
    arms.

None of that is convenience. The campaign's claim is that the arms differ in
their HEAD, so every other axis has to be pinned to the point where a
difference cannot come from anywhere else.

WHAT IS NECESSARILY DIFFERENT: the trunk is transformers' DINOv3 rather than
equimo's, the optimizer is torch.optim.AdamW rather than optax, and the micro
batch is measured separately (finetune_space.MICRO_BATCH has no vbll entry
until it is, and micro_batch() raises rather than borrowing the JAX value --
a different framework has a different activation profile).
"""

import argparse
import gc
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from bllarse import evaldump, metrics
from bllarse.mlflow_utils import child_run
from bllarse.finetune import checkpoint, imagenet, protocol
from bllarse.finetune.registry import (
    DATASET_CONFIGS,
    EQUIMO_MODELS,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

# bllarse.finetune.registry, NOT scripts/dinov3/vit_classification: that module
# imports jax at module scope and this arm runs in an environment with no jax in
# it (jax[cuda] and torch each ship a full CUDA stack and fight over the
# device). The constants are identical because there is one definition.

# The resolution every fine-tuning run uses. Mirrors finetune_jax; stated here
# rather than imported because importing it would pull jax in.
FINETUNE_IMG_SIZE = 224

# equimo model name -> the HF repo carrying the same Meta checkpoint. Both
# libraries convert the same released weights, which is what lets trunk_parity
# hold the two implementations to float noise.
HF_CHECKPOINTS = {
    "dinov3_small": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "dinov3_big": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3_huge": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
}


def trunk_features(model, pixel_values):
    """The pooled trunk feature: MEAN OVER PATCH TOKENS.

    Not the CLS token and not `pooler_output` -- which are the same vector as
    each other, and a different one from this. equimo's `out[0]`, which every
    JAX arm and every cached probing feature uses, is the patch-token mean;
    measured against it, mean-patch matches at cosine 1.000000 while CLS
    matches at 0.416 (scripts/finetune/trunk_parity.py).

    This is the single most dangerous line in the file. An arm built on CLS
    trains perfectly well, produces a plausible loss curve, and is quietly
    comparing a different representation against CE and IBProbit. Nothing
    downstream would look wrong.
    """
    out = model(pixel_values=pixel_values)
    n_skip = 1 + int(getattr(model.config, "num_register_tokens", 0) or 0)
    return out.last_hidden_state[:, n_skip:].mean(dim=1)


def predictive_logits(head, features, n_samples):
    """(log predictive, epistemic score) for one batch of features.

    `head.predictive(x, n_samples=...)`, not `head(x).predictive`: vbll's
    `forward` builds that Categorical at its own default n_samples with no
    pass-through, so reading it would silently ignore --predictive-samples.
    The twin of `_forward` in scripts/vbll_pytorch/probing_vbll.py.

    Dumped as `logit_kind="softmax_logits"` because the log of a normalised
    predictive is exactly what metrics.classification_metrics expects, and it
    is the same kind of array the JAX arms dump under that label.
    """
    import torch

    probs = head.predictive(features, n_samples=n_samples)
    # clamp_min guards the log: predictive() clips to [0, 1] and a zero would
    # send one class to -inf and every metric to nan.
    logits = torch.log(probs.clamp_min(1e-12))
    score = head.logit_predictive(features).covariance_diagonal.sum(dim=-1)
    return logits, score


def check_required_flags(args) -> None:
    """Reject configurations that would run but mean something else.

    Called before torch is imported, so the failure is a one-line message
    rather than a CUDA-initialised process dying three minutes in -- and so it
    is testable on a machine with no torch.
    """
    if not args.train_shards:
        raise SystemExit(
            "[vbll-ft] --train-shards is required. Without it this arm decodes "
            "JPEGs: ~40 img/s against the shard's ~220, which turns a 3 h "
            "trial into 14 h. That is a silent SLOWDOWN, not a visible "
            "failure -- every trial still completes and is still scored. "
            "Build one with scripts/finetune/build_shards.py."
        )
    if args.max_samples and args.epochs and args.epochs != 4.0:
        raise SystemExit(
            "[vbll-ft] --max-samples and --epochs are alternatives: tuning is "
            "capped in samples so the tuned batch size transfers, the final "
            "campaign is capped in epochs. Pass one."
        )


def resolve_plan(args, micro_batch: int) -> dict:
    """(target batch, micro batch, budget) -> the concrete schedule.

    protocol, not a local reimplementation: this arm has to land on the same
    samples_seen grid as the JAX ones or the eval index stops being comparable
    across a study, and MedianPruner compares trials at matching keys.
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
            f"[vbll-ft] budget of {steps} steps at effective batch {effective} "
            f"is shorter than one evaluation interval ({eval_every} steps = "
            f"{args.eval_quantum} samples). The run would produce no dump."
        )
    return {
        "micro": micro, "num_accum": num_accum, "effective_batch": effective,
        "steps": steps, "eval_every": eval_every, "n_evals": n_evals,
        "eval_quantum": args.eval_quantum,
    }


def stream_micro_batches(shard, micro, mean, std, total_micro, seed):
    """Yield exactly `total_micro` micro-batches, reshuffling between passes.

    The permutation is protocol.shuffled_indices -- numpy, seeded identically
    to the JAX arm -- so both arms see the same images in the same order. A
    torch.Generator here would break that silently: the runs would still train
    and still report the same seed.
    """
    import torch

    n = len(shard)
    labels_all = np.asarray(shard.labels)
    emitted, epoch = 0, 0
    while emitted < total_micro:
        order = protocol.shuffled_indices(n, seed, epoch)
        for start in range(0, n - micro + 1, micro):
            batch_idx = order[start:start + micro]
            images = imagenet.normalize_chunk(shard.take(batch_idx), mean, std)
            yield (torch.from_numpy(images),
                   torch.from_numpy(labels_all[batch_idx]).long())
            emitted += 1
            if emitted >= total_micro:
                return
        epoch += 1


def evaluate_split(model, head, images_uint8, labels, *, args, mean, std,
                   subset=None):
    """Forward one preloaded eval set in chunks. Returns (logits, score, scalars).

    Deliberately does NOT write, mirroring finetune_jax.evaluate_split: the
    caller buffers the argmin's logits and writes once from `_write_best`, so a
    122-evaluation tuning run produces one dump rather than 122.
    """
    import torch

    device = torch.device(args.device)
    model.eval()
    head.eval()
    logit_chunks, score_chunks = [], []
    with torch.no_grad():
        for start in range(0, len(labels), args.eval_chunk):
            raw = images_uint8[start:start + args.eval_chunk]
            imgs = torch.from_numpy(
                imagenet.normalize_chunk(raw, mean, std)).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                feats = trunk_features(model, imgs)
            # float() before the head: vbll's covariance algebra is numerically
            # delicate and autocast's bf16 would run the Cholesky-ish paths at
            # 8 mantissa bits. The trunk is the expensive part and keeps bf16.
            logits, score = predictive_logits(
                head, feats.float(), args.predictive_samples)
            logit_chunks.append(logits.float().cpu().numpy())
            score_chunks.append(score.float().cpu().numpy())
    model.train()
    head.train()
    logits = np.concatenate(logit_chunks, axis=0)
    if subset is not None:
        # R and A cover 200 classes, so the other 800 cannot occur. The JAX
        # arms mask before scoring; without this the VBLL arm would be scored
        # on the full 1000-way predictive over the same images and its OOD
        # numbers would not be comparable with theirs.
        keep = np.zeros(logits.shape[-1], dtype=bool)
        keep[np.asarray(subset)] = True
        logits = np.where(keep, logits, imagenet.MASK_NEG)
    return (logits, np.concatenate(score_chunks, axis=0),
            metrics.classification_metrics(logits, labels))


def load_ood_sets(args, img_size):
    """Same suite, same order, same masking as the JAX arms."""
    return imagenet.load_ood_sets(img_size)


def _write_best(args, best, last_eval_index):
    """Write the buffered argmin-val_nll evaluation, from a `finally`.

    So a trial stopped by EarlyConvergence -- the ORDINARY path once patience
    is on -- still leaves a scoreable dump. The twin of `_write_best` in
    finetune_jax.py and probing_vbll.py; see the note there for why `best is
    None` distinguishes two cases.
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
    for split_name, logits, labels, score in best["splits"]:
        evaldump.write_dump(
            args.dump_root, args.group_id, args.uid, best["step"],
            logits, labels,
            arm="vbll", split=split_name, logit_kind="softmax_logits",
            seed=args.seed, samples_seen=best["samples_seen"],
            score=score, score_kind="predictive_variance",
            at_last_eval=best["eval_index"] == last_eval_index,
        )
    print(
        f"[vbll-ft] {args.uid}: best val_nll={best['val_nll']:.4f} at step "
        f"{best['step']} (eval {best['eval_index']} of {last_eval_index}, "
        f"{best['samples_seen']} samples)"
        + (" AT LAST EVAL -- truncated, not converged"
           if best["eval_index"] == last_eval_index else ""),
        flush=True,
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="dinov3_large", choices=sorted(HF_CHECKPOINTS))
    p.add_argument("--dataset", default="imagenet1k")
    p.add_argument("--img-size", type=int, default=FINETUNE_IMG_SIZE)

    p.add_argument("--batch-size", type=int, default=1024,
                   help="EFFECTIVE (accumulated) batch -- the tuned quantity.")
    p.add_argument(
        "--micro-batch-size", type=int, required=True,
        help="Per-forward batch. REQUIRED, never auto-probed: design A8 pins "
             "it so num_accum is exact at every rung of the ladder. Measure it "
             "with scripts/finetune/probe_micro_batch.py on the target GPU.",
    )
    p.add_argument("--max-samples", type=int, default=0,
                   help="Tuning budget in samples. Mutually exclusive with "
                        "--epochs; see resolve_plan.")
    p.add_argument("--epochs", type=float, default=4.0)
    p.add_argument("--eval-quantum", type=int, default=protocol.EVAL_QUANTUM)
    p.add_argument("--eval-splits", default="val",
                   help="'+'-separated. Tuning uses val only; forwarding test "
                        "at all 122 evaluations would roughly double the run.")
    p.add_argument("--eval-chunk", type=int, default=256)
    p.add_argument("--val-fraction", type=float, default=imagenet.DEFAULT_VAL_FRACTION)
    p.add_argument("--dump-every-k-evals", type=int, default=0)
    p.add_argument("--ood-every-k-dumps", type=int, default=0)
    # See finetune_jax: the schedule is unchanged, but at those steps this run
    # writes a checkpoint and ood_worker.py evaluates it on another gpu.
    p.add_argument("--ood-checkpoint-root", default=None)
    p.add_argument("--ood-max-pending", type=int, default=2)

    # Two rates, and the head's is NOT optional. A fresh VBLL head at the
    # trunk's rate would still be near its init after a whole epoch; see
    # finetune_space.HEAD_LR_RATIO_BOX for why the ratio is what gets searched.
    p.add_argument("--lr", type=float, default=1e-4, help="Trunk learning rate.")
    p.add_argument("--head-lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="Trunk only. The head's is 0 on every arm (design D6): "
                        "VBLL's KL is its head regulariser, and IBProbit runs "
                        "no optimizer on the head at all, so a nonzero value "
                        "here would be an asymmetry between arms.")
    # See the same pair in finetune_jax.py: a per-STEP timescale, and this
    # script runs at ~3906 steps when tuning and ~10,009 for a 4-epoch final
    # run. protocol.betas_for holds (1 - beta) * steps invariant across that.
    # The defaults are torch's, so an invocation that omits them is unchanged.
    p.add_argument("--beta1", type=float, default=protocol.REFERENCE_BETAS[0],
                   help="AdamW beta1. Transfers with the run length; see "
                        "protocol.betas_for.")
    p.add_argument("--beta2", type=float, default=protocol.REFERENCE_BETAS[1],
                   help="AdamW beta2. Transfers with the run length; see "
                        "protocol.betas_for.")

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
    p.add_argument("--kl-ratio", type=float, default=1e-6)
    p.add_argument("--prior-scale", type=float, default=1.0)
    p.add_argument("--wishart-scale", type=float, default=0.13)
    p.add_argument("--parameterization", default="diagonal",
                   choices=["diagonal", "dense", "lowrank"])
    p.add_argument("--cov-rank", type=int, default=2)
    p.add_argument("--vbll-type", default="discriminative",
                   choices=["discriminative"])
    p.add_argument("--predictive-samples", type=int, default=20,
                   help="MC samples for E_w[softmax(w.x)] at eval time.")

    p.add_argument("--train-shards", default=os.environ.get("BLLARSE_FT_SHARDS", ""),
                   help="Pre-decoded uint8 shard directory. Without it this "
                        "arm decodes JPEGs, which measured ~5x slower.")
    p.add_argument("--dump-root",
                   default=os.environ.get("BLLARSE_EVAL_DUMP_ROOT", "eval_dumps"))
    p.add_argument("--group-id", default="finetune_vbll")
    p.add_argument("--uid", default=None)
    p.add_argument("--split-cache-dir", default=".cache/splits")
    p.add_argument("--seed", type=int, default=137)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action="store_true")
    return p


def build_model(args, *, embed_dim, num_classes, device):
    """(model, head) for the VBLL arm. The ONE place either is constructed.

    Extracted so the OOD worker rebuilds exactly what training built, from the
    args stored in the checkpoint. A worker with its own copy would drift
    silently -- a different regularization_weight or cov_rank loads a
    state_dict without complaint and scores a model that never existed.

    Mirrors finetune_jax.build_model, which does the same job for the JAX arms.
    """
    from transformers import AutoModel

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "vbll_pytorch" / "src"))
    from vbll_experiments.models import build_vbll_head

    model = AutoModel.from_pretrained(HF_CHECKPOINTS[args.model]).to(device)
    if model.config.hidden_size != embed_dim:
        # The registry's embed_dim drives the head's input width, and a
        # mismatch would surface as a shape error deep in vbll rather than
        # here. It also means the two libraries disagree about which
        # checkpoint this is.
        raise SystemExit(
            f"[vbll-ft] {args.model}: transformers reports hidden_size="
            f"{model.config.hidden_size} but registry.EQUIMO_MODELS says "
            f"{embed_dim}. The two libraries disagree about this checkpoint."
        )
    model.train()

    head = build_vbll_head(
        embed_dim, num_classes,
        # eta / N, not 1 / N. eta = kl_ratio * prior_scale is the KL
        # multiplier; shrinkage of the posterior mean goes as eta / prior_scale
        # = kl_ratio, so searching the ratio keeps the two axes near-orthogonal.
        # N is the FULL training set, matching the probing arm's len(train_f) --
        # not the samples actually seen, which varies with the budget.
        regularization_weight=(
            args.kl_ratio * args.prior_scale / imagenet.IMAGENET_N_TRAIN),
        prior_scale=args.prior_scale,
        wishart_scale=args.wishart_scale,
        parameterization=args.parameterization,
        cov_rank=args.cov_rank,
        vbll_type=args.vbll_type,
    ).to(device)
    return model, head


def default_uid(args) -> str:
    """The run's identity: model, dataset, arm and seed."""
    return f"{args.model}_{args.dataset}_vbll_seed{args.seed}"


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
    check_required_flags(args)

    import torch
    from transformers import AutoModel

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "vbll_pytorch" / "src"))
    from vbll_experiments.models import build_vbll_head

    if args.uid is None:
        # Byte-identical in shape to finetune_jax's default, because the two
        # have to be groupable by the same code.
        #
        # Note what this does NOT do: fig1_curves.UID_RE is
        # {model}_(?P<dataset>.+)_seed{seed} with a GREEDY dataset group, so
        # this parses as dataset="imagenet1k_vbll", not "imagenet1k". The arm
        # ends up inside the dataset field on every arm, including the JAX
        # ones, whose comment claims otherwise. It is harmless for this
        # campaign -- imagenet1k is the only dataset, and the arm is also in
        # group_id and in every dump's meta["arm"] -- but a table keyed on the
        # parsed dataset would print "imagenet1k_vbll" as a dataset name.
        # Tuning trials never hit this path; finetune_worker passes
        # uid=f"trial{n:05d}".
        args.uid = default_uid(args)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    ds_config = DATASET_CONFIGS[args.dataset]
    embed_dim = EQUIMO_MODELS[args.model]["embed_dim"]
    num_classes = ds_config["num_classes"]
    mean, std = IMAGENET_MEAN, IMAGENET_STD

    plan = resolve_plan(args, args.micro_batch_size)
    print(f"[vbll-ft] effective_batch={plan['effective_batch']} = "
          f"{plan['num_accum']} x micro {plan['micro']}, steps={plan['steps']}, "
          f"eval every {plan['eval_every']} steps ({plan['eval_quantum']} "
          f"samples) = {plan['n_evals']} evals", flush=True)

    model, head = build_model(
        args, embed_dim=embed_dim, num_classes=num_classes, device=device)
    model.train()

    # Two param groups, two rates. weight_decay 0 on the head: see --weight-decay.
    # betas are set on the optimizer rather than per group: they are an
    # averaging horizon in STEPS, and both groups take the same steps.
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": args.lr,
         "weight_decay": args.weight_decay},
        {"params": head.parameters(), "lr": args.head_lr, "weight_decay": 0.0},
    ], betas=(args.beta1, args.beta2))

    from datasets import load_dataset
    ds = load_dataset(ds_config["hf_path"], name=ds_config.get("hf_config"))
    eval_split = ds[ds_config["test_split"]]

    val_idx, test_idx = imagenet.pinned_eval_split(
        args.split_cache_dir, n=len(eval_split), val_fraction=args.val_fraction)
    eval_labels_all = np.asarray(eval_split[ds_config["label_key"]])
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
        print(f"[vbll-ft] preloaded {name}: {len(idx)} images", flush=True)

    from bllarse.finetune import shards as shards_mod
    shard = shards_mod.open_shard(
        args.train_shards, args.dataset, "train", args.img_size)
    print(f"[vbll-ft] training from shard: {len(shard)} rows "
          f"({args.train_shards})", flush=True)

    micro_iter = stream_micro_batches(
        shard, plan["micro"], mean, std,
        plan["steps"] * plan["num_accum"], args.seed)

    best = None
    eval_index = 0
    train_time = 0.0
    try:
        for step in range(1, plan["steps"] + 1):
            step_t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)

            for _ in range(plan["num_accum"]):
                images, labels = next(micro_iter)
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=args.bf16):
                    feats = trunk_features(model, images)
                # Divided by num_accum so the accumulated gradient is the MEAN
                # over the effective batch, matching what a single unaccumulated
                # step of that size would produce -- and matching jax_steps.
                # apply_grads, which divides by num_accum on the JAX side.
                loss = head(feats.float()).train_loss_fn(labels) / plan["num_accum"]
                loss.backward()

            optimizer.step()

            step_time = time.perf_counter() - step_t0
            if step > 1:
                train_time += step_time

            if step % plan["eval_every"]:
                continue

            samples_seen = step * plan["effective_batch"]
            eval_index = samples_seen // plan["eval_quantum"]
            payload, scalars = [], {}
            for name, (images_uint8, labels) in eval_sets.items():
                logits, score, out = evaluate_split(
                    model, head, images_uint8, labels,
                    args=args, mean=mean, std=std)
                payload.append((name, logits, labels, score))
                scalars.update({f"{name}_{k}": v for k, v in out.items()})

            val_nll = float(scalars["val_nll"])
            if math.isfinite(val_nll) and (best is None or val_nll < best["val_nll"]):
                best = {
                    "val_nll": val_nll, "step": step, "eval_index": eval_index,
                    "samples_seen": samples_seen, "splits": payload,
                }

            # After the buffer update, so one bad evaluation cannot discard a
            # good dump -- but still fatal, because a diverged run would
            # otherwise train silently to the full budget on a broken model.
            if not np.isfinite(val_nll):
                raise RuntimeError(
                    f"[vbll-ft] step {step}: val_nll={val_nll!r} is not finite "
                    f"-- the run has diverged."
                )

            if args.dump_every_k_evals and (
                eval_index >= plan["n_evals"]
                or eval_index % args.dump_every_k_evals == 0
            ):
                for name, logits, labels, score in payload:
                    evaldump.write_dump(
                        args.dump_root, args.group_id, args.uid, step,
                        logits, labels, arm="vbll", split=name,
                        logit_kind="softmax_logits", seed=args.seed,
                        samples_seen=samples_seen,
                        score=score, score_kind="predictive_variance",
                    )

            # Hand OOD to the worker: write a slot at the scheduled steps and
            # move on. Slots rather than a rolling latest because
            # curves._id_ood_pair pairs ID and OOD dumps by STEP.
            if args.ood_checkpoint_root and args.dump_every_k_evals and (
                    args.ood_every_k_dumps):
                _di = eval_index // args.dump_every_k_evals
                if eval_index >= plan["n_evals"] or (
                        _di % args.ood_every_k_dumps == 0):
                    _stem = checkpoint.write(
                        args.ood_checkpoint_root,
                        group_id=args.group_id, uid=args.uid, step=step,
                        blob_suffix=".pt",
                        write_blob=lambda path: torch.save(
                            {"model": model.state_dict(),
                             "head": head.state_dict()}, path),
                        meta={
                            "framework": "torch",
                            "samples_seen": samples_seen,
                            "eval_index": eval_index,
                            "mlflow_run_id": getattr(run, "run_id", None),
                            "args": vars(args),
                        },
                        max_pending=args.ood_max_pending,
                    )
                    print(f"[vbll-ft] ood checkpoint step {step}: "
                          + ("written" if _stem else
                             f"SKIPPED, {args.ood_max_pending} already pending"),
                          flush=True)

            print(f"[vbll-ft] eval {eval_index}/{plan['n_evals']} "
                  f"step {step}/{plan['steps']} " + " ".join(
                      f"{k}={v:.4f}" for k, v in sorted(scalars.items())),
                  flush=True)
            # BEFORE on_dump: see finetune_jax. on_dump is where the pruner
            # and the patience clock raise.
            if run is not None:
                run.log_metrics(dict(scalars), step=samples_seen)
            if on_dump is not None:
                on_dump(eval_index, step, samples_seen, scalars)
            gc.collect()
    finally:
        # See finetune_jax: only when no trajectory was written, so a run never
        # mixes buffered-argmin and trajectory dump semantics.
        if not args.dump_every_k_evals:
            _write_best(args, best, eval_index)
        shard.close()

    n_timed = plan["steps"] - 1
    if n_timed > 0 and train_time > 0:
        mean_step_time = train_time / n_timed
        print(f"[vbll-ft] train time: {train_time:.3f}s total, "
              f"{mean_step_time * 1000:.2f}ms/step (excl. step 1), "
              f"{plan['effective_batch'] / mean_step_time:.2f} img/s", flush=True)

    print(f"[vbll-ft] done: {eval_index} evaluations under "
          f"{args.dump_root}/{args.group_id}/{args.uid}", flush=True)



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
    main(build_argparser().parse_args())
