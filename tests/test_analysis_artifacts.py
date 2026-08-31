"""Publishing the analysis JSON, so figures can be rebuilt off-cluster.

The dumps stay the source of truth, but they live on Raven and the JSON the
analysis distils out of them was written beside them -- so every figure needed
the cluster mounted. These helpers move that JSON to the tracking server.

Network-free: the round trip against the live server was verified by hand. What
is pinned here is the behaviour that must NOT be lenient, because the failure
mode is a figure built from a stale local file that its author believed came
from the server.
"""

from pathlib import Path

import pytest

from bllarse import mlflow_utils


def test_a_local_path_resolves_to_itself(tmp_path):
    local = tmp_path / "fig1_curves.json"
    local.write_text("{}")
    assert mlflow_utils.resolve_analysis_path(str(local)) == local


def test_publishing_a_missing_file_raises(tmp_path):
    """Not a warning. `--publish` on a path that does not exist means the
    analysis did not produce what the caller thinks it did."""
    with pytest.raises(FileNotFoundError, match="nothing to publish"):
        mlflow_utils.publish_analysis(tmp_path / "absent.json", "whatever")


def test_no_tracking_uri_raises_rather_than_skipping(monkeypatch, tmp_path):
    """The lenient path is the dangerous one here.

    `_LiveRun` swallows tracking failures on purpose -- an outage must never
    kill a GPU job, and the dumps still hold the result. Publishing is the
    opposite case: if it silently no-ops, the next plot reads whatever stale
    JSON is lying around locally and nothing anywhere says so.
    """
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(mlflow_utils, "load_mlflow_env_defaults", lambda *a, **k: {})
    target = tmp_path / "fig1_curves.json"
    target.write_text("{}")
    with pytest.raises(RuntimeError, match="tracking URI"):
        mlflow_utils.publish_analysis(target, "fig1_curves")
    with pytest.raises(RuntimeError, match="tracking URI"):
        mlflow_utils.fetch_analysis("fig1_curves")


def test_the_analysis_tag_is_stable():
    """`fetch_analysis` finds a run by this tag, and `publish_analysis` reuses
    the run it finds. Renaming the tag orphans every previously published
    artifact -- fetch would report "no published analysis" for something that is
    sitting right there."""
    assert mlflow_utils.ANALYSIS_TAG == "bllarse.analysis"


def test_the_plotting_scripts_accept_an_mlflow_source():
    """The flag is only useful if the plot scripts actually consult it."""
    root = Path(__file__).resolve().parents[1] / "scripts" / "plotting"
    for name in ("plot_fig1.py", "plot_fig2_draft.py"):
        source = (root / name).read_text()
        assert "resolve_analysis_path" in source, name
        assert "open(args.json)" not in source, (
            f"{name} still opens args.json directly, so `--json mlflow:NAME` "
            f"would be treated as a filename"
        )


def test_fig2_divides_arm_groups_and_only_when_there_are_groups():
    """The divider was hardcoded at x=1.5 -- right for the two arms the figure
    had when it was written, wrong the moment IBProbit made it three. Derived
    from SERIES, it tracks however many arms the figure grows to.

    It divides GROUPS, not bars. Since Fig 2 dropped entropy and native and
    reports MSP alone, every x position is its own arm, and a rule between each
    adjacent pair would separate nothing while adding two lines per panel. The
    dividers return the moment any arm contributes a second entry -- another
    score, or a second campaign -- which is exactly when a reader could
    misattribute one arm's bar to its neighbour."""
    import sys
    root = Path(__file__).resolve().parents[1] / "scripts" / "plotting"
    sys.path.insert(0, str(root))
    import plot_fig2_draft

    arms = [s[0] for s in plot_fig2_draft.SERIES]
    grouped = len(set(arms)) < len(arms)
    expected = sum(1 for a, b in zip(arms, arms[1:]) if a != b) if grouped else 0
    boundaries = plot_fig2_draft._arm_boundaries()
    assert len(boundaries) == expected, (
        f"{len(boundaries)} dividers for {len(arms)} entries over "
        f"{len(set(arms))} arms in SERIES"
    )
    # Each boundary must fall between two entries of different arms.
    for b in boundaries:
        left, right = arms[int(b - 0.5)], arms[int(b + 0.5)]
        assert left != right, f"divider at {b} sits inside arm {left!r}"

    # And the grouped case still divides, whatever SERIES happens to hold today.
    original = plot_fig2_draft.SERIES
    try:
        plot_fig2_draft.SERIES = [
            ("adamw_ce", "ts_msp", "", ""), ("adamw_ce", "msp", "", ""),
            ("vbll", "msp", "", ""),
        ]
        assert plot_fig2_draft._arm_boundaries() == [1.5]
    finally:
        plot_fig2_draft.SERIES = original
