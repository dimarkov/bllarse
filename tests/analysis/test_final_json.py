"""The reader that Figs 5-7 share. Its job is to survive PARTIAL input."""

import json

import pytest

from bllarse.analysis import final_json


def _run(arm, seed, xs, values, ood=None):
    return {
        "id": {"samples_seen": list(xs), "arm": arm, "seed": seed,
               **{m: list(v) for m, v in values.items()}},
        "ts": {},
        "ood": ood or {},
    }


def _doc(runs, *, scored=None, total=15):
    return {"model": "dinov3_large", "group_id": "ftfinal", "runs": runs,
            "runs_scored": scored if scored is not None else len(runs),
            "runs_total": total, "complete": len(runs) == total,
            "corrupt_dumps": []}


def test_band_counts_seeds_per_x_not_per_curve():
    # Seed 138 is missing the middle point -- a corrupt dump, which is exactly
    # what iter_dumps(skip_corrupt=True) leaves behind. The x survives on the
    # seeds that have it rather than being dropped from the band entirely.
    data = _doc({
        "m_d_ce_seed137": _run("ce", 137, [1, 2, 3], {"nll": [3.0, 2.0, 1.0]}),
        "m_d_ce_seed138": _run("ce", 138, [1, 3], {"nll": [5.0, 3.0]}),
    })
    xs, mean, lo, hi, n = final_json.band(
        final_json.id_series(data, "ce", "id", "nll"))

    assert xs == [1, 2, 3]
    assert n == [2, 1, 2]
    assert mean == pytest.approx([4.0, 2.0, 2.0])
    assert lo == pytest.approx([3.0, 2.0, 1.0])
    assert hi == pytest.approx([5.0, 2.0, 3.0])


def test_provenance_says_partial_and_names_the_corrupt_count():
    data = _doc({"m_d_ce_seed137": _run("ce", 137, [1], {"nll": [1.0]})})
    data["corrupt_dumps"] = ["/ptmp/a.npz", "/ptmp/b.npz"]

    line = final_json.provenance(data)

    assert "PARTIAL" in line
    assert "1 of 15" in line
    assert "2 unreadable" in line


def test_ood_series_keeps_its_own_irregular_grid():
    # OOD lands on a stride of the dump grid, and on dinov3_large an irregular
    # one. The reader must not resample it onto the ID x values.
    ood = {"ood_imagenet_r": {"20": {"nll": 4.0}, "60": {"nll": 3.0}}}
    data = _doc({"m_d_vbll_seed137": _run(
        "vbll", 137, [10, 20, 30, 40, 50, 60], {"nll": [1.0] * 6}, ood=ood)})

    series = final_json.ood_series(data, "vbll", "ood_imagenet_r", "nll")

    assert series == {137: [(20, 4.0), (60, 3.0)]}


def test_rows_are_threshold_shaped_with_samples_seen_standing_in_for_step():
    data = _doc({"m_d_ce_seed137": _run(
        "ce", 137, [10, 20],
        {"acc": [0.1, 0.2], "nll": [2.0, 1.0], "brier": [0.9, 0.8],
         "ece": [0.3, 0.2]})})

    rows = final_json.rows(data, "m_d_ce_seed137")

    assert [r["step"] for r in rows] == [10, 20]
    assert [r["samples_seen"] for r in rows] == [10, 20]
    assert {"acc", "nll", "brier", "ece"} <= set(rows[0])


def test_load_refuses_a_json_with_no_scored_runs(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(_doc({})))

    with pytest.raises(SystemExit):
        final_json.load(path)


def test_ood_series_reads_the_scaled_variant_when_asked():
    run = _run("ce", 137, [10], {"nll": [1.0]},
               ood={"ood_imagenet_r": {"10": {"nll": 4.0}}})
    # T fit on the ID val split, applied unchanged to the shifted logits.
    run["ood_ts"] = {"ood_imagenet_r": {"10": {"nll": 2.5}}}
    data = _doc({"m_d_ce_seed137": run})

    assert final_json.ood_series(data, "ce", "ood_imagenet_r", "nll") == \
        {137: [(10, 4.0)]}
    assert final_json.ood_series(data, "ce", "ood_imagenet_r", "nll", "ts") == \
        {137: [(10, 2.5)]}


def test_ood_arms_falls_back_to_raw_when_no_scaled_ood_was_scored():
    # A JSON from before ood_ts existed must not silently lose the baseline.
    data = _doc({"m_d_ce_seed137": _run("ce", 137, [10], {"nll": [1.0]})})

    keys = {key: label for key, label, _c, _m in final_json.ood_arms(data)}

    assert ("ce", "id") in keys
    assert "raw" in keys[("ce", "id")]


def test_ood_arms_keeps_the_scaled_entry_once_it_exists():
    run = _run("ce", 137, [10], {"nll": [1.0]})
    run["ood_ts"] = {"ood_imagenet_r": {"10": {"nll": 2.5}}}
    data = _doc({"m_d_ce_seed137": run})

    keys = [key for key, _l, _c, _m in final_json.ood_arms(data)]

    assert ("ce", "ts") in keys
    assert ("ce", "id") not in keys


def test_the_module_still_exports_what_the_figure_scripts_import():
    """A guard against silent deletion.

    plot_fig5 and plot_fig6 are scripts with no test of their own, so when an
    edit to this module dropped METRICS, OOD_SETS and tuning_budget the whole
    suite stayed green and only rendering a figure showed it. These four names
    are the module's contract with those scripts.
    """
    for name in ("ARMS", "METRICS", "OOD_SETS", "tuning_budget",
                 "format_samples"):
        assert hasattr(final_json, name), f"final_json.{name} is gone"

    assert [m[0] for m in final_json.METRICS] == ["acc", "nll", "brier"]
    assert len(final_json.OOD_SETS) == 4
    # None, DELIBERATELY: the selection budget is a property of the study a
    # config came from and reaches neither the scored JSON nor the dumps. It
    # used to return protocol.MAX_SAMPLES, which is right only for the
    # fixed-1M campaigns and overstated the per-rung ladder's 399,872 by 2.5x
    # -- drawing the extrapolation boundary inside territory that was searched.
    assert final_json.tuning_budget() is None
    # and the label formatter that replaced `f"{n/1e6:.0f}M"`, which rendered
    # every per-rung budget as "0M".
    assert final_json.format_samples(399872) == "400k"
    assert final_json.format_samples(2_562_334) == "2.6M"
