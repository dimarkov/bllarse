"""Which sweep each figure arm reports.

The one mapping in this repo where the label and the group_id deliberately
disagree, so it is pinned rather than left to reading comprehension. A silent
swap here changes which experiment every published number came from, and
nothing downstream would look wrong.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "analysis"))

import fig1_curves  # noqa: E402
import fig2_detection  # noqa: E402

FIGURES = [("fig1", fig1_curves), ("fig2", fig2_detection)]


@pytest.mark.parametrize("suffix,module", FIGURES)
def test_ibprobit_reports_the_eps_free_sweep(suffix, module):
    """The presented method has no contamination parameter.

    So the arm labelled "ibprobit" must read probe_ibprobit_noeps_*, not
    probe_ibprobit_*. The two sweeps differ ONLY in whether eps_init was tuned
    or pinned at 1e-12, so pointing at the wrong one produces a figure that
    renders perfectly and reports the other experiment.
    """
    assert module.GROUPS["ibprobit"] == f"probe_ibprobit_noeps_{suffix}"


@pytest.mark.parametrize("suffix,module", FIGURES)
def test_the_eps_tuned_sweep_is_not_plotted(suffix, module):
    # It is retained as an ablation (scripts/analysis/eps_ablation.py), which
    # reads its group_ids directly and does not go through GROUPS.
    assert f"probe_ibprobit_{suffix}" not in set(module.GROUPS.values())


@pytest.mark.parametrize("suffix,module", FIGURES)
def test_no_arm_is_labelled_as_an_ablation(suffix, module):
    # "ibprobit_noeps" as a LABEL would put a qualifier in the legend for the
    # method the paper actually presents.
    assert "ibprobit_noeps" not in module.GROUPS


@pytest.mark.parametrize("suffix,module", FIGURES)
def test_every_arm_maps_to_a_distinct_sweep(suffix, module):
    assert len(set(module.GROUPS.values())) == len(module.GROUPS)


@pytest.mark.parametrize("suffix,module", FIGURES)
def test_the_three_compared_arms_are_present(suffix, module):
    assert set(module.GROUPS) == {"adamw_ce", "vbll", "ibprobit"}
