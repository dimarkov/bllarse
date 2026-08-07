"""Per-eval-point logit dumps -- the source of truth for every reported figure.

Everything downstream (accuracy, ECE at any binning, NLL, Brier, temperature
scaling, threshold crossings, OOD detection) is computed from these files, so a
run is measured once and never re-run to add a metric.

Logits are float16 on disk: at ImageNet's K=1000 a 5k eval subset is 10 MB per
point, and a full 50k final eval is 100 MB. Readers upcast before any
exponentiation (see bllarse.metrics).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


@dataclass(frozen=True)
class DumpRecord:
    logits: np.ndarray
    labels: np.ndarray
    score: np.ndarray | None
    meta: dict[str, Any]


def dump_schedule(num_steps: int, n_dumps: int = 40) -> np.ndarray:
    """Log-spaced eval steps: dense early, sparse late.

    Fig 5's claim is that the calibration gap is largest early, so resolution
    belongs there. Dumping all `num_steps` points would be ~5 GB per run for no
    additional information about the claim.

    Rounding a geometric grid to integers collapses its early points -- at
    num_steps=100 the first twenty all land below 3 -- so a naive unique() drops
    exactly the resolution this schedule exists to provide, and silently returns
    fewer points than were asked for. Each step is therefore advanced to at least
    one past its predecessor, giving exactly min(num_steps, n_dumps) distinct
    steps.

    Gaps trend upward but are not monotone: where the forced-increment region
    meets the geometric one they can dip by exactly one, at roughly 2% of sizes.
    Verified by brute force over 118k (n_dumps, num_steps) combinations -- the
    dip never exceeds one.

    Always includes step 1 and step `num_steps`.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if n_dumps < 2:
        raise ValueError("n_dumps must be >= 2 so the schedule spans 1..num_steps")
    if num_steps <= n_dumps:
        return np.arange(1, num_steps + 1)

    steps: list[int] = []
    previous = 0
    for value in np.geomspace(1, num_steps, n_dumps):
        step = max(int(round(value)), previous + 1)
        if step > num_steps:
            break
        steps.append(step)
        previous = step
    return np.asarray(steps, dtype=int)


def dump_dir(root, group_id: str, uid: str) -> Path:
    return Path(root) / group_id / uid


def write_dump(
    root,
    group_id: str,
    uid: str,
    step: int,
    logits,
    labels,
    *,
    arm: str,
    split: str,
    logit_kind: str,
    seed: int,
    samples_seen: int,
    score=None,
    score_kind: str | None = None,
    eps_hat: float | None = None,
) -> Path:
    """Write one eval point. Returns the path written.

    `logit_kind` records what the array actually is, so no downstream consumer
    has to infer it. All three arms currently write "softmax_logits", but they
    do not write the same THING: AdamW+CE dumps raw pre-softmax logits, while
    VBLL and IBProbit both dump an already-normalized log-predictive (VBLL's is
    a Monte Carlo average of softmax samples; IBProbit's is `_mixture_logp`,
    normalized because (1-eps)*1 + K*(eps/K) == 1). Softmaxing any of the three
    is correct -- it is a no-op on the latter two -- which is why they share a
    label; but scores that need an unnormalized level are not, and
    `metrics._reject_normalized` is what guards those.

    `score` is the arm's native epistemic score and is diagnostic only: the
    primary Fig 2 comparison uses MSP and entropy derived from `logits`, which
    every arm has. It is absent for AdamW+CE, which has no such quantity.
    """
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"logits and labels length mismatch: {logits.shape[0]} vs {labels.shape[0]}"
        )
    # Labels are stored int16. This module exists to stop silent, unrecoverable
    # loss, and an out-of-range cast wraps rather than raising.
    if labels.size and (labels.min() < -32768 or labels.max() > 32767):
        raise ValueError(
            f"labels outside int16 range [{labels.min()}, {labels.max()}]; "
            f"storing them would wrap silently"
        )

    meta = {
        "arm": arm,
        "step": int(step),
        "samples_seen": int(samples_seen),
        "split": split,
        "logit_kind": logit_kind,
        "score_kind": score_kind,
        "seed": int(seed),
        "eps_hat": None if eps_hat is None else float(eps_hat),
    }

    payload = {
        "logits": logits.astype(np.float16),
        "labels": labels.astype(np.int16),
        "meta": np.asarray(json.dumps(meta)),
    }
    if score is not None:
        payload["score"] = np.asarray(score, dtype=np.float32)

    directory = dump_dir(root, group_id, uid)
    directory.mkdir(parents=True, exist_ok=True)
    # `split` is part of the filename, not just `step`: a single training run
    # dumps val and test from the same forward pass at every scheduled step,
    # and OOD lands on the same step number as the final val/test point. All
    # of them share one uid directory, so step alone is not a unique key --
    # without the split suffix, the second write_dump call at a given step
    # would silently clobber the first (e.g. the val dump lost under the test
    # dump written right after it).
    path = directory / f"step_{int(step):07d}_{split}.npz"
    np.savez_compressed(path, **payload)
    return path


def load_dump(path) -> DumpRecord:
    data = np.load(path, allow_pickle=False)
    return DumpRecord(
        logits=data["logits"],
        labels=data["labels"],
        score=data["score"] if "score" in data.files else None,
        meta=json.loads(str(data["meta"])),
    )


def iter_dumps(root, group_id: str, uid: str) -> Iterator[DumpRecord]:
    """Yield every dump for one run, ordered by step."""
    directory = dump_dir(root, group_id, uid)
    for path in sorted(directory.glob("step_*.npz")):
        yield load_dump(path)
