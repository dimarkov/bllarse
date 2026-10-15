"""A dump does not prove its run finished, and the analysis must know that.

`_write_best` is called from a `finally` in both probing scripts, so a run that
crashes mid-budget still writes a normal-looking argmin dump. These pin the
machinery that makes that detectable, and the guards that stop a degenerate
quantity being plotted as a measured one.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from bllarse import evaldump
from bllarse.analysis.curves import completeness
from bllarse.tuning import space

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "plotting"))

import fig1_curves  # noqa: E402
import plot_fig1  # noqa: E402
import plot_fig2  # noqa: E402


def _write(root, uid, steps, *, group="g", arm="adamw_ce", **meta):
    """One run's dumps. `steps` may be a single step (what the probing scripts
    write) or several (a trajectory of dumps, as budget pilots produce)."""
    rng = np.random.default_rng(0)
    for step in steps:
        logits = rng.normal(size=(64, 5)).astype(np.float32)
        labels = np.arange(64) % 5
        for split in ("val", "test"):
            evaldump.write_dump(
                root, group, uid, step, logits, labels,
                arm=arm, split=split, logit_kind="softmax_logits",
                seed=137, samples_seen=step * 8, **meta,
            )


def test_completeness_separates_died_from_truncated_from_converged():
    assert completeness({"at_last_eval": False, "eval_index": 40,
                         "evals_run": 100, "evals_planned": 100})["incomplete"] is False
    # Reached the end of its budget, but the argmin IS the end: still improving.
    truncated = completeness({"at_last_eval": True, "eval_index": 100,
                              "evals_run": 100, "evals_planned": 100})
    assert truncated["incomplete"] is False and truncated["truncated"] is True
    # Died: fewer evaluations ran than were scheduled.
    assert completeness({"at_last_eval": True, "eval_index": 12,
                         "evals_run": 12, "evals_planned": 100})["incomplete"] is True


def test_truncation_catches_the_near_boundary_case_the_exact_flag_misses():
    """The defect this constant exists for.

    On a fine grid -- batch size equal to the eval quantum, so one evaluation
    per step -- flowers102 runs a 30,000-point grid. An argmin at index 29,992
    sets at_last_eval FALSE, because 29,992 != 30,000, while being a run that
    plainly never turned. All ten adamw_ce/flowers102/dinov3_small seeds sit
    there, and keying the figure's marker on the exact flag would leave that
    entire cell unmarked: 1 run of 599 flagged where 18 belong.
    """
    near = completeness({"at_last_eval": False, "eval_index": 29_992,
                         "evals_run": 30_000, "evals_planned": 30_000})
    assert near["at_last_eval"] is False, "the exact flag genuinely does not fire"
    assert near["truncated"] is True, "but the run is still improving"
    assert near["budget_fraction"] == pytest.approx(0.99973, abs=1e-4)

    # And a run that peaked early is not swept up by the looser rule.
    early = completeness({"at_last_eval": False, "eval_index": 3_000,
                          "evals_run": 30_000, "evals_planned": 30_000})
    assert early["truncated"] is False


def test_truncation_falls_back_to_the_exact_flag_without_an_index():
    """A coarse budget -- food101/ibprobit schedules 26 evaluations -- has no
    meaningful 'last tenth', and a dump may carry no index at all. The exact flag
    is then the only signal, and it must still be honoured."""
    assert completeness({"at_last_eval": True})["truncated"] is True
    assert completeness({"at_last_eval": False})["truncated"] is False


def test_the_budget_is_reconstructed_from_the_sweep_not_from_the_dump():
    """No re-run is needed to place an existing dump on its own budget.

    A dump records the numerator (`step`, `samples_seen`) and never the
    denominator, so "is step 3,759 early or terminal?" cannot be answered from
    the dump alone. But the denominator is fixed by the sweep config, which is
    in the repo.

    Asserted as the INVARIANT rather than a literal: `budget` counts
    evaluations, and evaluations are `eval_quantum` samples apart, so
    `budget * eval_every_steps` must be the step budget whatever batch size the
    sweep happens to hold. A literal such as 30,000 evaluations for flowers102
    is true only at "batch 256, evaluating every step"; at batch 64 the same
    30,000 steps is 7,500 evaluations, and the test would fail on a config
    change rather than on a defect.
    """
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, "bllarse_sweeps")
    import probe_adamw_ce

    from bllarse.tuning import space

    budgets = json.loads(Path("bllarse_sweeps/epoch_budgets.json").read_text())
    for dataset, cfg in probe_adamw_ce.TUNED.items():
        evals = fig1_curves.budget("adamw_ce", dataset)
        eval_every = space.eval_every_steps(dataset, cfg["batch_size"])
        steps = cfg["epochs"] * space.steps_per_epoch(dataset, cfg["batch_size"])
        assert evals == steps // eval_every, dataset
        assert evals * eval_every >= budgets[dataset]["steps"] - eval_every, dataset
    # Unknown, never guessed -- a wrong denominator silently reclassifies runs.
    assert fig1_curves.budget("adamw_ce", "not_a_dataset") is None
    assert fig1_curves.budget("no_such_arm", "flowers102") is None


def test_reconstruction_marks_the_campaigns_real_truncated_cell():
    """adamw_ce/flowers102/dinov3_small, all ten seeds, from dump data alone.

    Their optima land around step 29,950+ of 30,000. With a 30,000-point eval
    grid the exact flag reads False, which alone would leave this cell
    unmarked; the reconstructed fraction places it correctly without touching
    the GPU.
    """
    planned = fig1_curves.budget("adamw_ce", "flowers102")
    index = 29_950 * 256 // space.eval_quantum("flowers102")  # samples_seen // quantum
    assert completeness({}, eval_index=index, evals_planned=planned)["truncated"]

    # A run that peaked a fifth of the way in is left alone.
    early = 6_000 * 256 // space.eval_quantum("flowers102")
    assert not completeness({}, eval_index=early, evals_planned=planned)["truncated"]


def test_the_dumps_own_values_win_over_the_reconstruction():
    """The fallback is for dumps that lack these fields. A dump that records
    them knows better than any after-the-fact arithmetic, so it must not be
    overridden by it."""
    c = completeness({"eval_index": 10, "evals_planned": 1000},
                     eval_index=999, evals_planned=1000)
    assert c["eval_index"] == 10 and c["truncated"] is False


def test_completeness_reports_unknown_rather_than_complete_on_old_dumps():
    """The dangerous default. A dump written without these fields knows
    nothing about whether its run finished, and returning False ("not
    incomplete") would let every such dump pass a completeness filter."""
    assert completeness({})["incomplete"] is None
    assert completeness({"at_last_eval": True})["incomplete"] is None


def test_write_dump_round_trips_the_evaluation_counts(tmp_path):
    path = evaldump.write_dump(
        tmp_path, "g", "u", 5, np.zeros((4, 3)), np.zeros(4, dtype=int),
        arm="adamw_ce", split="test", logit_kind="softmax_logits",
        seed=1, samples_seen=40, at_last_eval=True,
        eval_index=7, evals_run=7, evals_planned=100,
    )
    meta = evaldump.load_dump(path).meta
    assert (meta["evals_run"], meta["evals_planned"]) == (7, 100)
    assert completeness(meta)["incomplete"] is True


def test_a_crashed_run_is_dropped_not_scored(tmp_path):
    """The whole point: its dump is readable and looks entirely normal."""
    _write(tmp_path, "dinov3_big_food101_seed137", [100],
           at_last_eval=True, eval_index=11, evals_run=11, evals_planned=940)
    _write(tmp_path, "dinov3_big_food101_seed138", [100],
           at_last_eval=False, eval_index=300, evals_run=940, evals_planned=940)

    monkey = dict(fig1_curves.GROUPS)
    fig1_curves.GROUPS.clear()
    fig1_curves.GROUPS["adamw_ce"] = "g"
    try:
        rows, skipped, dropped = fig1_curves.collect(tmp_path)
    finally:
        fig1_curves.GROUPS.update(monkey)

    assert [r["uid"] for r in rows] == ["dinov3_big_food101_seed138"]
    assert len(dropped) == 1 and "seed137" in dropped[0]
    assert "11 of 940" in dropped[0], "the reason must name the actual counts"
    assert not skipped, "a crashed run is dropped, not skipped -- they differ"


def test_the_named_casualties_are_dropped_without_any_counts(tmp_path):
    """The runs in INCOMPLETE_RUNS were written without the counts, so only
    the explicit list catches them. If an entry is removed without its run
    being re-run, this fails."""
    uid = "dinov3_big_food101_seed143"
    _write(tmp_path, uid, [100])
    assert ("adamw_ce", uid) in fig1_curves.INCOMPLETE_RUNS

    monkey = dict(fig1_curves.GROUPS)
    fig1_curves.GROUPS.clear()
    fig1_curves.GROUPS["adamw_ce"] = "g"
    try:
        rows, _, dropped = fig1_curves.collect(tmp_path)
    finally:
        fig1_curves.GROUPS.update(monkey)
    assert rows == [] and len(dropped) == 1


def test_single_dump_runs_emit_no_endpoint_metrics(tmp_path):
    """`final_nll - nll` on a one-dump run is zero by construction. Emitting it
    turns "not measured" into the much stronger "measured: no drift"."""
    _write(tmp_path, "dinov3_big_dtd_seed137", [100])
    scored = fig1_curves.score_run(tmp_path, "g", "dinov3_big_dtd_seed137")
    assert "nll" in scored
    assert "final_nll" not in scored


def test_trajectory_runs_still_emit_endpoint_metrics(tmp_path):
    _write(tmp_path, "dinov3_big_dtd_seed138", [100, 200, 300])
    scored = fig1_curves.score_run(tmp_path, "g", "dinov3_big_dtd_seed138")
    assert "final_nll" in scored and scored["n_dumps"] == 3


def test_summarize_counts_seeds_by_rows_not_by_metric_lists():
    """Keys are not on every row -- `final_*` only on trajectory dumps, the
    completeness fields only on dumps that record them -- so taking the length of
    whichever metric list came first under-counts a mixed cell."""
    rows = [
        {"arm": "a", "model": "m", "dataset": "d", "seed": "1", "nll": 1.0},
        {"arm": "a", "model": "m", "dataset": "d", "seed": "2", "nll": 2.0,
         "final_nll": 9.0},
    ]
    cell = fig1_curves.summarize(rows)["a|m|d"]
    assert cell["n_seeds"] == 2
    assert cell["nll"]["n"] == 2
    assert cell["final_nll"]["n"] == 1, "and the partial key reports its own n"


def test_truncation_aggregates_into_a_per_cell_rate():
    rows = [
        {"arm": "a", "model": "m", "dataset": "d", "seed": "1", "nll": 1.0,
         "truncated": True},
        {"arm": "a", "model": "m", "dataset": "d", "seed": "2", "nll": 2.0,
         "truncated": False},
    ]
    assert fig1_curves.summarize(rows)["a|m|d"]["truncated"]["mean"] == 0.5


def test_the_endpoint_penalty_figure_refuses_a_campaign_with_no_trajectories(tmp_path):
    summary = {"adamw_ce|dinov3_small|dtd": {"n_seeds": 1, "nll": {"mean": 0.5, "std": 0.0}}}
    out = tmp_path / "penalty.png"
    plot_fig1.make_overfit_figure(summary, out)
    assert not out.exists(), (
        "it drew a flat line through zero across every panel, which reads as "
        "the measured result 'no arm overfits'"
    )


def test_the_id_control_figure_refuses_to_silently_drop_a_real_comparison(tmp_path):
    """The single-point form is correct only while the two steps coincide. If
    the summary carries at_best_* keys, plotting one point would report the
    endpoint alone."""
    id_summary = {
        "adamw_ce|cifar10": {
            "n_runs": 10,
            "acc": {"mean": 0.99, "std": 0.001},
            "at_best_acc": {"mean": 0.99, "std": 0.001},
        }
    }
    with pytest.raises(NotImplementedError, match="at_best"):
        plot_fig2.make_id_figure(id_summary, tmp_path / "control.png")
