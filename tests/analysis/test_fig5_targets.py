"""Fig 5's threshold targets, read off the strongest CE baseline."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "plotting"))

import plot_fig5  # noqa: E402

from bllarse.analysis import thresholds  # noqa: E402


def _ce_run(seed):
    # The val NLL bottoms out at the middle evaluation, so the early-stopping
    # point and the last evaluation are different rows.
    return {
        "id": {"samples_seen": [100, 200, 300], "arm": "ce", "seed": seed,
               "acc": [0.70, 0.80, 0.82], "nll": [1.0, 0.6, 0.7],
               "brier": [0.40, 0.30, 0.32], "ece": [0.05, 0.03, 0.04],
               "val_nll": [1.1, 0.5, 0.8]},
        "ts": {},
        "ood": {},
    }


DATA = {"model": "dinov3_huge", "group_id": "g",
        "runs": {f"m_d_ce_seed{s}": _ce_run(s) for s in (137, 138)},
        "runs_scored": 2, "runs_total": 2, "complete": True, "corrupt_dumps": []}


@pytest.mark.parametrize("reference", thresholds.REFERENCES)
def test_every_reference_the_cli_offers_produces_targets(reference):
    # The --reference choices ARE thresholds.REFERENCES, so this covers every
    # value the command line accepts.
    targets, _name = plot_fig5.pick_targets(DATA, 1.0, reference=reference)

    expected = {"early_stop": (0.80, 0.6, 0.30), "last": (0.82, 0.7, 0.32)}[reference]
    assert (targets.acc, targets.nll, targets.brier) == pytest.approx(expected)
