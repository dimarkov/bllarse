"""Turn a finished tuning study into the final campaign's run configuration.

The tuning campaign scores trials at `protocol.MAX_SAMPLES` (2,000,000 samples,
1.56 epochs); the final campaign runs 4 epochs (5,124,668 samples). Until this
module existed the winning config was simply RUN at the longer budget, which
assumes the optimum does not move with the duration. It does -- see
`protocol.transfer` for the rule and its provenance.

Two jobs, and they are separate on purpose:

  * `absolute_params` undoes the search-space reparameterisation, so a study's
    recorded params become the quantities the training scripts' flags take.
  * `final_config` applies the batch/duration transfer and assembles the full
    command, so what the final campaign ran is derived from what the search
    found rather than transcribed from it.

The second is the one that has to be reportable. "Tuned at 2M samples,
transferred to 4 epochs by the AdamW SDE rule" is a methods sentence; "tuned at
2M samples and then run at 4 epochs" is a confound.
"""

import math
from typing import Any, Mapping

from bllarse.finetune import imagenet, protocol
from bllarse.tuning import finetune_space as space
from bllarse.tuning import finetune_worker

# The final campaign's budget, from the design doc. `protocol.MAX_SAMPLES` is
# the TUNING budget and the two are deliberately different quantities: tuning
# is capped in samples so the batch ladder stays comparable, the final run is
# capped in epochs.
FINAL_EPOCHS = 4.0

# What `finetune_space._suggest_per_sqrt_batch` appends to a searched name.
# Aliased rather than re-spelled: the rename is a property of the search space,
# and two independent spellings of it would drift the moment one is edited.
SEARCH_SUFFIX = space.PER_SQRT_BATCH_SUFFIX
HORIZONS_SUFFIX = space.HORIZONS_SUFFIX

# Reporting flags the final run needs and a tuning trial does not. Tuning
# forwards `val` alone -- the objective is val_nll and forwarding the 30k test
# partition 122 times would roughly double the campaign -- and writes exactly
# one dump. The final run is what the figures are made from.
FINAL_EVAL_SPLITS = "val+test"

# The main campaign's experiment, distinct from the tuning one so a search
# trial and a reported run are never averaged together by accident. The
# integration itself is shared: both campaigns reach MLflow through the same
# --enable-mlflow flag on the same training scripts, because both render a
# config dict into that script's argv.
MLFLOW_EXPERIMENT = "bllarse-finetuning"

# One group for the whole final campaign, not one per arm. The uid already
# carries model, dataset, arm and seed, so a single group puts every reported
# run under one dump directory -- which is what Fig 5 reads, since it plots the
# arms against each other rather than one at a time.
FINAL_GROUP_ID = "ftfinal"


def absolute_params(
    params: Mapping[str, Any], *, effective_batch: int, steps: int | None = None
) -> dict[str, Any]:
    """Undo BOTH search reparameterisations: per-sqrt-batch and horizons.

    A study records `lr_per_sqrt_batch` and `beta1_horizons`; the training
    scripts take `--lr` and `--beta1`. This multiplies the first back up by
    sqrt(effective_batch) and converts the second at `steps`, the step count the
    trial actually ran.

    Handling the horizons here is not cosmetic. `protocol.transfer` reads a
    config's betas by KEY, so a study whose betas are recorded under
    `*_horizons` looked to it like a study with no betas at all -- and it
    silently fell back to REFERENCE_BETAS. Every arm then transferred to the
    same (0.9, 0.999)-derived pair, discarding the entire beta search: the one
    thing this campaign was re-run to measure.

    `steps` is required once any horizon key is present, and refused otherwise
    rather than defaulted -- a beta converted at the wrong step count is a
    plausible number with no error attached to it.

    Tolerates studies that have NEITHER form of the rename, which is not
    defensive coding but a requirement: the campaigns that exist today
    (ft_*_shards2m and ft_*_fixb512) both searched `lr` directly, and they are
    the studies this is first going to be pointed at. A key without the suffix
    passes through untouched.
    """
    if effective_batch < 1:
        raise ValueError(f"effective_batch must be >= 1, got {effective_batch}")

    if any(k.endswith(HORIZONS_SUFFIX) for k in params) and steps is None:
        raise ValueError(
            "params carry a horizon coordinate but `steps` was not given; the "
            "beta it implies is meaningless without the step count it was "
            "searched at."
        )

    root = math.sqrt(effective_batch)
    out: dict[str, Any] = {}
    for key, value in params.items():
        if key.endswith(HORIZONS_SUFFIX):
            bare = key[: -len(HORIZONS_SUFFIX)]
            if bare in params:
                raise ValueError(
                    f"study records both {key!r} and {bare!r}; the "
                    f"reparameterisation is ambiguous."
                )
            out[bare] = protocol.beta_from_horizons(value, steps)
        elif key.endswith(SEARCH_SUFFIX):
            bare = key[: -len(SEARCH_SUFFIX)]
            if bare in params:
                raise ValueError(
                    f"study records both {key!r} and {bare!r}; the "
                    f"reparameterisation is ambiguous. A study should carry "
                    f"one or the other, and a campaign that changed the search "
                    f"space should have taken a new name."
                )
            out[bare] = value * root
        else:
            out[key] = value
    return out


def final_config(
    arm: str,
    params: Mapping[str, Any],
    *,
    effective_batch: int,
    epochs: float = FINAL_EPOCHS,
    from_samples: int = protocol.MAX_SAMPLES,
    to_batch: int | None = None,
    n_train: int = imagenet.IMAGENET_N_TRAIN,
    dump_root: Any = None,
    seed: int = space.REPORTED_SEEDS[0],
    group_id: str = FINAL_GROUP_ID,
    eval_splits: str = FINAL_EVAL_SPLITS,
    eval_quantum: int | None = None,
    enable_mlflow: bool = True,
    mlflow_experiment: str = MLFLOW_EXPERIMENT,
    dump_every_k_evals: int | None = None,
    ood_every_k_dumps: int | None = None,
    ood_checkpoint_root: str | None = None,
    ood_max_pending: int | None = None,
    model: str | None = None,
    lr_decay_start: float = 1.0,
    lr_decay_to: float = 0.5,
    layer_decay: float = protocol.LAYER_DECAY_OFF,
    layer_decay_span: float = protocol.LAYER_DECAY_SPAN_OFF,
) -> dict[str, Any]:
    """The final campaign's config for `arm`, transferred from a tuned trial.

    `params` are a study's recorded params (either reparameterisation -- see
    `absolute_params`). `effective_batch` is the batch they were tuned at,
    which a trial carries in its `effective_batch` user attr.

    `to_batch` defaults to the tuned batch, i.e. a pure duration transfer. A
    different value additionally applies the batch rule, and will raise on the
    IBProbit arms: q0 = alpha^2 / B is a second, unrelated batch relation and
    `protocol.transfer` refuses to guess which one wins.

    `model` overrides `space.TUNING_MODEL`. Left alone by default, and it is
    worth being explicit about why: the final campaign transplants a config
    tuned on dinov3_large onto dinov3_huge, and NOTHING here corrects for that.
    The width rule that would (CompleteP's lr x m_N^-1, so x0.8 for
    1024 -> 1280) is unvalidated for fine-tuning a pretrained backbone, and its
    depth companion -- a m_L^-alpha multiplier on every residual branch -- is
    an architecture change that cannot be applied to a pretrained ViT at all.
    So the transplant stays an assumption that the paper reports, exactly as it
    was before this module existed. This function narrows the batch/duration
    gap and leaves the width gap visible.

    `lr_decay_start`/`lr_decay_to` carry the WSD schedule. `lr_decay_start` is
    the fraction of the run at which the decay BEGINS, not its length. They
    default to OFF (start 1.0),
    reproducing the constant-LR runs exactly, and they are emitted into the
    config unconditionally so a sweep can reach them through
    `final_config.py --set` -- which refuses a key the config does not already
    have, and so would refuse a schedule that only existed as a script flag.
    Turning the schedule on changes what the transferred hyperparameters mean:
    they were tuned at a constant LR, and optax scales decoupled weight decay
    by the learning rate, so the run's total shrinkage falls by
    `protocol.wsd_mean_multiplier`.
    """
    to_samples = int(epochs * n_train)
    batch = effective_batch if to_batch is None else to_batch

    absolute = absolute_params(
        params, effective_batch=effective_batch,
        steps=protocol.steps_for_samples(from_samples, effective_batch))
    transferred = protocol.transfer(
        absolute,
        from_samples=from_samples,
        from_batch=effective_batch,
        to_samples=to_samples,
        to_batch=batch,
    )

    # The model goes IN, not on afterwards. trial_config derives
    # micro_batch_size from it, and overriding config["model"] after the fact
    # left the micro batch pinned to TUNING_MODEL's -- which is precisely how
    # five vbll runs got dinov3_large's 64 on dinov3_huge and OOMed at model
    # load. Setting it here means one model decides both.
    if model is not None:
        transferred["model"] = model
    config = finetune_worker.trial_config(
        arm, params=transferred, max_samples=from_samples, dump_root=dump_root
    )

    # The budget swaps from samples to epochs. Both scripts take these as
    # alternatives and `--max-samples` wins when nonzero, so it has to be
    # cleared rather than merely joined by `--epochs`.
    config["max_samples"] = 0
    config["epochs"] = epochs
    config["eval_splits"] = eval_splits
    # A coarser grid than tuning's, and it is a WALL-CLOCK constraint rather
    # than a preference. The final run forwards val+test (50k) at every
    # evaluation instead of val alone (10k), over 2.56x the samples. At
    # protocol.EVAL_QUANTUM that is 312 evaluations x 50k = 15.6M forward
    # passes against ~15.4M fwd-equivalents of training -- evaluation would
    # roughly DOUBLE the run, putting a dinov3_huge 4-epoch job near 36 h
    # against a 24 h partition limit. Passing a multiple of the quantum trades
    # curve resolution for fitting the wall; see submit_finetune_final.sh.
    if eval_quantum is not None:
        if eval_quantum % batch:
            raise ValueError(
                f"eval quantum {eval_quantum} is not a whole number of "
                f"batch-{batch} steps; protocol.eval_every_steps would refuse "
                f"it and every run would die at startup."
            )
        config["eval_quantum"] = eval_quantum
    # The final campaign is 5 seeds per arm, and NOTHING upstream supplies one:
    # trial_config never sets a seed because objective.evaluate passes it for
    # tuning trials, and the final runs have no such wrapper. Without this every
    # seed of every arm would run at the scripts' default 137 -- five identical
    # runs reported as an error bar -- and would collide on one MLflow run name,
    # because the uid the scripts derive is keyed on the seed.
    config["seed"] = seed
    config["group_id"] = group_id
    # Overrides trial_config's tuning experiment: same flag, same scripts, but
    # a reported 4-epoch run does not belong in the search's experiment.
    # Always emitted, both of them, even at the default. --set refuses a key
    # the config lacks, so a schedule that appeared only when already enabled
    # could never be enabled through it.
    config["lr_decay_start"] = float(lr_decay_start)
    config["lr_decay_to"] = float(lr_decay_to)
    # Emitted unconditionally, like the WSD pair above and for the same reason:
    # --set refuses a key the config lacks, so a decay that only appeared when
    # already enabled could never be turned on -- or OFF -- through it.
    config["layer_decay"] = float(layer_decay)
    # Both are emitted, and the scripts refuse to receive both as non-default.
    # A study tuned by span therefore transfers by span -- which is the only
    # form that stays meaningful if the final run uses a different backbone
    # from the one the search ran on.
    config["layer_decay_span"] = float(layer_decay_span)
    config["enable_mlflow"] = enable_mlflow
    config["mlflow_experiment"] = mlflow_experiment
    if dump_every_k_evals is not None:
        config["dump_every_k_evals"] = dump_every_k_evals
    # OOD rides a stride of the dump grid, which is itself a stride of the eval
    # grid, and finetune_jax gates on BOTH being nonzero. trial_config sets each
    # to 0 -- right for tuning, where a trial writes one argmin dump and must
    # never pay the 98,389-image preload -- so the reported campaign has to set
    # them explicitly or it silently produces no trajectory and no OOD at all.
    if ood_every_k_dumps is not None:
        config["ood_every_k_dumps"] = ood_every_k_dumps
    # With a checkpoint root the trainer writes slots instead of forwarding the
    # suite, and scripts/finetune/ood_worker.py scores them on another gpu.
    if ood_checkpoint_root is not None:
        config["ood_checkpoint_root"] = str(ood_checkpoint_root)
    if ood_max_pending is not None:
        config["ood_max_pending"] = ood_max_pending
    # Check the INVARIANT, not the formula. The betas reach `config` through
    # trial_config's `**params`; what has to survive is the horizon
    # (1-beta)*steps, which is the searched coordinate and the thing the rule
    # actually holds fixed. Re-deriving the betas here instead would only
    # restate protocol.transfer's arithmetic -- and did, wrongly: it compared
    # against the transfer of REFERENCE_BETAS rather than of the trial's own
    # searched values, so every study that tuned its betas failed this check
    # while carrying entirely correct numbers.
    tune_steps = protocol.steps_for_samples(from_samples, effective_batch)
    final_steps = protocol.steps_for_samples(to_samples, batch)
    for name, fallback in zip(("beta1", "beta2"), protocol.REFERENCE_BETAS):
        # The fallback is for studies predating the searched axes, which ran at
        # the library defaults and recorded no betas.
        before = protocol.horizons(float(absolute.get(name, fallback)), tune_steps)
        after = protocol.horizons(float(config[name]), final_steps)
        if abs(after - before) > 1e-6 * max(1.0, abs(before)):
            raise AssertionError(
                f"{name} horizon not preserved across the transfer: "
                f"{before:.6g} at {tune_steps} steps became {after:.6g} at "
                f"{final_steps}. The searched coordinate is (1-beta)*steps and "
                f"it must come out unchanged."
            )
    return config


def best_config(study: Any, arm: str, **kwargs: Any) -> dict[str, Any]:
    """`final_config` for a study's best COMPLETED trial.

    Reads the batch off the trial's `effective_batch` user attr rather than its
    params, because the batch is pinned and therefore no longer a param. Trials
    from the first campaign, where it WAS searched, carry the same attr -- the
    worker has always written it.
    """
    trial = study.best_trial
    batch = trial.user_attrs.get("effective_batch")
    if batch is None:
        batch = trial.params.get("batch_size")
    if batch is None:
        batch = space.FIXED_BATCH_SIZE
    kwargs.setdefault("from_samples", trial.user_attrs.get(
        "max_samples", protocol.MAX_SAMPLES))
    # LLRD is a CAMPAIGN setting, not a searched param, so it lives in the
    # trial's user attrs and never appears in `trial.params`. Without this it
    # is silently dropped by the transfer and the final run trains every block
    # at 1.0x -- on hyperparameters selected under a 0.032x-1.0x trunk. On
    # dinov3_small at 0.75 that is the embeddings moving 31x faster than
    # anything the search ever scored.
    #
    # A trial records ONE of the two. New trials carry `layer_decay_span`, which
    # is depth-agnostic and transfers as-is. Trials from before 2026-08-31 carry
    # only `layer_decay`, a PER-BLOCK factor, and that number does not survive a
    # change of backbone: 0.75 is a span of 0.032 on a 12-block trunk and of
    # 1.0e-4 on a 32-block one. Emitting it unchanged for a final run on a
    # deeper model would train the embeddings 316x slower than anything the
    # search ever scored -- and nothing would error.
    #
    # So a legacy per-block factor is converted through the depth of the model
    # it was TUNED at, which the trial records.
    span = trial.user_attrs.get("layer_decay_span")
    decay = trial.user_attrs.get("layer_decay")
    if span is None and decay is not None and float(decay) != protocol.LAYER_DECAY_OFF:
        tuned_at = trial.user_attrs.get("model", space.TUNING_MODEL)
        span = float(decay) ** space.trunk_depth(tuned_at)
        decay = protocol.LAYER_DECAY_OFF
    kwargs.setdefault("layer_decay", float(
        protocol.LAYER_DECAY_OFF if decay is None else decay))
    kwargs.setdefault("layer_decay_span", float(
        protocol.LAYER_DECAY_SPAN_OFF if span is None else span))
    return final_config(arm, trial.params, effective_batch=int(batch), **kwargs)
