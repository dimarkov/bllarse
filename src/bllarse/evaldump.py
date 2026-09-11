"""Per-eval-point logit dumps -- the source of truth for every reported figure.

Everything downstream (accuracy, ECE at any binning, NLL, Brier, temperature
scaling, threshold crossings, OOD detection) is computed from these files, so a
run is measured once and never re-run to add a metric.

Logits are float16 on disk: at ImageNet's K=1000 a 5k eval subset is 10 MB per
point, and a full 50k final eval is 100 MB. Readers upcast before any
exponentiation (see bllarse.metrics).
"""

import json
import os
import sys
import zipfile
import zlib
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


def dump_dir(root, group_id: str, uid: str) -> Path:
    return Path(root) / group_id / uid


def campaign_root(campaign: str, root=None) -> str:
    """The dump root for one TUNING CAMPAIGN, under the campaign-free root.

    A dump's path is root/group_id/uid, and none of those three carries the
    hyperparameters the run used. `group_id` is a property of the SWEEP
    ("probe_adamw_ce_fig1") and `uid` of the cell (model, dataset, seed), so
    re-running the same sweep with a re-tuned TUNED block writes a DIFFERENT
    configuration to exactly the same files. Nothing errors; the figures simply
    start reporting a mixture, or the earlier campaign is gone.

    That was live: the win5 dumps behind the current Figs 1 and 2 sit under
    `eval_dumps_win5` only because the submitter happened to export
    BLLARSE_EVAL_DUMP_ROOT that way, and the win2 re-run with the variable
    unset or stale would have landed on top of them.

    So the campaign is appended HERE rather than typed into an environment
    variable at submit time: BLLARSE_EVAL_DUMP_ROOT names the campaign-free
    root and each sweep declares which campaign its numbers came from, next to
    the numbers. Forgetting the variable can no longer merge two campaigns --
    it can only put both under ./eval_dumps/<campaign>.

    Analysis is unaffected and stays campaign-agnostic: it is pointed at one
    campaign with `--dump-root eval_dumps/win5`, exactly as before.
    """
    base = root if root is not None else os.environ.get(
        "BLLARSE_EVAL_DUMP_ROOT", "eval_dumps"
    )
    return os.path.join(base, campaign)


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
    at_last_eval: bool | None = None,
    eval_index: int | None = None,
    evals_run: int | None = None,
    evals_planned: int | None = None,
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
        # Whether this point was the run's LAST evaluation, recorded by the
        # training script rather than inferred later.
        #
        # `thresholds.early_stop` used to infer truncation from the curve --
        # "is the argmin the last row?" -- which worked when a run dumped its
        # whole trajectory. A run now writes only its argmin, so that inference
        # is trivially True for every run and the budget rule built on it
        # ("raise the budget until at_last_step is false for >=90% of runs",
        # tuning.budgets) would reject every budget. Only the training loop
        # knows how many evaluations there were, so it is the only thing that
        # can answer this.
        #
        # None on dumps written without it -- older runs, and any caller that
        # does not know -- and early_stop falls back to the curve inference
        # there.
        "at_last_eval": None if at_last_eval is None else bool(at_last_eval),
        # How many evaluations RAN, against how many the budget SCHEDULED.
        #
        # `at_last_eval` answers "was the argmin the last point?", which
        # separates a converged run from a budget-truncated one -- but only
        # among runs that reached the end of their budget. It cannot see the
        # third case: a run that DIED. `_write_best` is called from a `finally`
        # in both probing scripts, so a crashed run still writes its
        # best-so-far dump, and that dump is byte-indistinguishable from a
        # completed run's. Two Fig 1 runs went into the published figure that
        # way (MLflow status FAILED at 39 s and 415 s against sibling medians
        # of 253 s and 1893 s); nothing downstream could tell.
        #
        # evals_run < evals_planned is the signal, and only the training loop
        # can record it: the planned count is a function of the budget, which
        # the dump otherwise never sees.
        #
        # None on dumps written without them. Consumers must treat None as
        # "unknown", never as "complete".
        "evals_run": None if evals_run is None else int(evals_run),
        "evals_planned": None if evals_planned is None else int(evals_planned),
        # WHERE on the evaluation grid the argmin fell. `at_last_eval` is the
        # boolean version of this and is too blunt on the fine grids: a trial
        # whose batch size equals the eval quantum evaluates every single step,
        # so flowers102 runs a 30,000-point grid and an argmin at index 29,992
        # sets at_last_eval FALSE while being, obviously, a run that never
        # turned. All ten adamw_ce/flowers102/dinov3_small seeds sit there.
        #
        # With the index, "how far into its budget did this run peak" is a
        # ratio rather than a boolean, and the near-boundary case stops hiding
        # behind an exact comparison. See analysis.curves.completeness.
        "eval_index": None if eval_index is None else int(eval_index),
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
    # Written to a sibling temp file and renamed, never straight to `path`.
    # np.savez_compressed streams a zip: a job killed part-way through -- a
    # SLURM wall clock, a preemption -- leaves a truncated archive AT THE FINAL
    # NAME, which reads as a real dump until something decompresses it. That
    # cost the dinov3_large scoring run twice, once as BadZipFile and once as
    # zlib.error, and the four casualties are unrecoverable without re-running
    # the job. os.replace is atomic within a directory, so an interrupted write
    # now leaves a stray .tmp and no dump, which is the honest outcome.
    #
    # The temp name ENDS IN .npz. savez_compressed appends `.npz` to any target
    # that does not already end in it, so a `.tmp` suffix would put the archive
    # at `<tmp>.npz` and leave the rename pointing at a file that was never
    # written. The pid is in the name because OOD workers for several runs
    # share a directory and a fixed suffix would have two of them writing one
    # file; the leading dot keeps it out of the `step_*.npz` glob readers use.
    tmp = directory / f".{path.stem}.{os.getpid()}.tmp.npz"
    try:
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


# Paths `iter_dumps(skip_corrupt=True)` refused to read, in the order it hit
# them. Module-level because the scorer needs the tally across every run it
# walks, not per call.
CORRUPT_DUMPS: list[str] = []


def load_dump(path) -> DumpRecord:
    data = np.load(path, allow_pickle=False)
    return DumpRecord(
        logits=data["logits"],
        labels=data["labels"],
        score=data["score"] if "score" in data.files else None,
        meta=json.loads(str(data["meta"])),
    )


def iter_dumps(
    root, group_id: str, uid: str, *, skip_corrupt: bool = False
) -> Iterator[DumpRecord]:
    """Yield every dump for one run, ordered by step.

    `write_dump` hands the path straight to `np.savez_compressed`, so a run
    killed mid-write (a SLURM wall clock, a preemption) leaves a truncated npz
    behind. Reading one raises out of numpy as `zipfile.BadZipFile` when the
    archive's own CRC catches it and as `zlib.error` when the deflate stream is
    damaged in a way the CRC never gets to check -- both were seen on the same
    dinov3_large run. By default either propagates -- a corrupt file is a real defect and a
    single-run analysis should not paper over it. `skip_corrupt=True` is for
    the batch scorer, where one bad file out of thousands must not cost the
    other 3,311: the path is reported on stderr and appended to
    `CORRUPT_DUMPS` so the caller can account for what it lost.
    """
    directory = dump_dir(root, group_id, uid)
    for path in sorted(directory.glob("step_*.npz")):
        if not skip_corrupt:
            yield load_dump(path)
            continue
        try:
            record = load_dump(path)
        except (zipfile.BadZipFile, zlib.error, EOFError, ValueError,
                OSError) as exc:
            CORRUPT_DUMPS.append(str(path))
            print(f"[evaldump] SKIPPING corrupt dump {path}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            continue
        yield record


TRAJECTORY_NAME = "trajectory.npz"


def trajectory_path(root, group_id: str, uid: str) -> Path:
    return dump_dir(root, group_id, uid) / TRAJECTORY_NAME


def write_trajectory(
    root,
    group_id: str,
    uid: str,
    *,
    eval_index,
    step,
    samples_seen,
    val_nll,
    arm: str,
    seed: int,
    evals_planned: int | None = None,
) -> Path:
    """The val_nll curve for one run: every evaluation, not just the argmin.

    WHY THIS EXISTS. `write_dump` stores ONE evaluation per run -- the val_nll
    argmin -- because logits are large. That made every question about the
    stopping rule unanswerable without re-running the campaign, and this project
    hit four of them: whether a run was truncated, whether the val split is big
    enough to select on, whether a different criterion (Brier, accuracy, a
    smoothed minimum) would choose a different checkpoint, and whether patience
    was set sensibly. Each cost either a re-run or an argument from indirect
    evidence.

    Scalars only, so it is cheap where the logits are not: a 30,000-evaluation
    run is ~1 MB here against ~100 MB for a single ImageNet-scale logit dump.
    Written from the same `finally` as the argmin dump, so a crashed or
    early-stopped run still leaves whatever curve it produced.

    This does NOT change which checkpoint is selected or reported -- it records
    what the selection had to choose from.
    """
    path = trajectory_path(root, group_id, uid)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"arm": arm, "seed": seed, "evals_planned": evals_planned,
            "n_evals": int(len(val_nll))}
    np.savez_compressed(
        path,
        eval_index=np.asarray(eval_index, dtype=np.int64),
        step=np.asarray(step, dtype=np.int64),
        samples_seen=np.asarray(samples_seen, dtype=np.int64),
        val_nll=np.asarray(val_nll, dtype=np.float32),
        meta=json.dumps(meta),
    )
    return path


def load_trajectory(root, group_id: str, uid: str) -> dict[str, Any]:
    """Read back what `write_trajectory` wrote, or raise FileNotFoundError.

    Raises rather than returning None on a missing file: a run predating
    trajectory dumping has no curve, and silently handing back an empty one
    would let a caller average over runs that contributed nothing.
    """
    path = trajectory_path(root, group_id, uid)
    if not path.exists():
        raise FileNotFoundError(
            f"no trajectory at {path}; this run predates trajectory dumping"
        )
    with np.load(path, allow_pickle=False) as data:
        out = {k: data[k] for k in ("eval_index", "step", "samples_seen", "val_nll")}
        out["meta"] = json.loads(str(data["meta"]))
    return out
