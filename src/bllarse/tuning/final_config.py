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

# Reporting flags the final run needs and a tuning trial does not. Tuning
# forwards `val` alone -- the objective is val_nll and forwarding the 30k test
# partition 122 times would roughly double the campaign -- and writes exactly
# one dump. The final run is what the figures are made from.
FINAL_EVAL_SPLITS = "val+test"


def absolute_params(
    params: Mapping[str, Any], *, effective_batch: int
) -> dict[str, Any]:
    """Undo the per-sqrt-batch reparameterisation.

    A study records `lr_per_sqrt_batch`; the training scripts take `--lr`. This
    multiplies the searched coordinate back up by sqrt(effective_batch) and
    strips the suffix.

    Tolerates studies that have NEITHER form of the rename, which is not
    defensive coding but a requirement: the campaigns that exist today
    (ft_*_shards2m and ft_*_fixb512) both searched `lr` directly, and they are
    the studies this is first going to be pointed at. A key without the suffix
    passes through untouched.
    """
    if effective_batch < 1:
        raise ValueError(f"effective_batch must be >= 1, got {effective_batch}")

    root = math.sqrt(effective_batch)
    out: dict[str, Any] = {}
    for key, value in params.items():
        if key.endswith(SEARCH_SUFFIX):
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
    eval_splits: str = FINAL_EVAL_SPLITS,
    dump_every_k_evals: int | None = None,
    model: str | None = None,
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
    """
    to_samples = int(epochs * n_train)
    batch = effective_batch if to_batch is None else to_batch

    transferred = protocol.transfer(
        absolute_params(params, effective_batch=effective_batch),
        from_samples=from_samples,
        from_batch=effective_batch,
        to_samples=to_samples,
        to_batch=batch,
    )

    config = finetune_worker.trial_config(
        arm, params=transferred, max_samples=from_samples, dump_root=dump_root
    )

    # The budget swaps from samples to epochs. Both scripts take these as
    # alternatives and `--max-samples` wins when nonzero, so it has to be
    # cleared rather than merely joined by `--epochs`.
    config["max_samples"] = 0
    config["epochs"] = epochs
    config["eval_splits"] = eval_splits
    if dump_every_k_evals is not None:
        config["dump_every_k_evals"] = dump_every_k_evals
    if model is not None:
        config["model"] = model

    # The betas reach `config` through trial_config's `**params`, since
    # `transfer` put them in `transferred`. Re-derived and compared rather than
    # trusted: this is the one value the tuning campaign never sets, so nothing
    # upstream would notice it going missing, and shipping the library defaults
    # at the final duration is the exact defect this module exists to remove.
    expected = protocol.betas_for(
        protocol.steps_for_samples(to_samples, batch),
        from_steps=protocol.steps_for_samples(from_samples, effective_batch),
    )
    got = (config.get("beta1"), config.get("beta2"))
    if got != expected:
        raise AssertionError(
            f"betas {got} do not match the transfer rule's {expected} at "
            f"{to_samples} samples, batch {batch}"
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
    return final_config(arm, trial.params, effective_batch=int(batch), **kwargs)
