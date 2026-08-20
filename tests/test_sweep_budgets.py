"""`epochs` in each sweep must still encode the 30,000-step budget.

It is a restatement of the budget at that entry's batch size, not a free knob,
so a batch change without an epochs change silently gives one cell a different
budget from its neighbours -- invisible in the figure, fatal to the comparison.
The re-tune changed several batch sizes, which is exactly when this breaks.
"""
import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "bllarse_sweeps")
from bllarse.tuning import space  # noqa: E402

BUDGETS = json.loads(Path("bllarse_sweeps/epoch_budgets.json").read_text())
# Fig 1 and Fig 2 sweeps alike: both carry TUNED blocks whose `epochs`
# encodes the step budget at that entry's batch size, and the Fig 2 (OOD)
# modules were missed when the Fig 1 ones were first re-exported.
MODULES = ["probe_adamw_ce", "probe_vbll", "probe_ibprobit_noeps",
           "probe_ood_adamw_ce", "probe_ood_vbll", "probe_ood_ibprobit_noeps"]


@pytest.mark.parametrize("name", MODULES)
def test_every_entry_realises_the_full_step_budget(name):
    mod = importlib.import_module(name)
    for dataset, cfg in mod.TUNED.items():
        want = BUDGETS[dataset]["steps"]
        per_epoch = space.steps_per_epoch(dataset, cfg["batch_size"])
        got = cfg["epochs"] * per_epoch
        assert got >= want, (
            f"{name}/{dataset}: batch={cfg['batch_size']} epochs={cfg['epochs']} "
            f"gives {got} steps, short of the {want}-step budget"
        )
        # Rounded UP to whole epochs, so never more than one epoch of overshoot.
        assert got - want < per_epoch, (
            f"{name}/{dataset}: {got} steps overshoots {want} by more than one epoch"
        )


@pytest.mark.parametrize("name", MODULES)
def test_the_batch_size_is_on_the_datasets_own_ladder(name):
    """A batch off the powers-of-two ladder makes `eval_quantum / batch` a
    non-integer, and the sample-spaced evaluation grid becomes unschedulable."""
    mod = importlib.import_module(name)
    for dataset, cfg in mod.TUNED.items():
        assert cfg["batch_size"] in space.batch_sizes(dataset), (
            f"{name}/{dataset}: batch {cfg['batch_size']} is off the ladder "
            f"{space.batch_sizes(dataset)}"
        )
        assert space.eval_quantum(dataset) % cfg["batch_size"] == 0


@pytest.mark.parametrize("name", MODULES)
def test_seeds_default_to_the_ten_the_figures_are_specified_on(name, monkeypatch):
    monkeypatch.delenv("BLLARSE_SWEEP_SEEDS", raising=False)
    mod = importlib.reload(importlib.import_module(name))
    assert mod._seeds() == list(range(137, 147))
    monkeypatch.setenv("BLLARSE_SWEEP_SEEDS", "137 138")
    assert mod._seeds() == [137, 138]
