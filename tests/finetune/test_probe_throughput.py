"""The probe's verdict logic, which is what a human acts on."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "finetune"))

import probe_throughput as pt  # noqa: E402


def test_verdict_flags_the_loader_as_the_bottleneck():
    v = pt.verdict(decode_rate=120.0, train_rate=350.0)
    assert "loader" in v.lower()
    assert "pre-resized" in v.lower()  # names the actual remedy


def test_verdict_reports_compute_bound_when_the_loader_keeps_up():
    v = pt.verdict(decode_rate=900.0, train_rate=350.0)
    assert "compute" in v.lower()
    assert "compute-bound" in v.lower()  # Verify we're in the COMPUTE-BOUND branch, not LOADER-BOUND


def test_verdict_is_explicit_at_the_margin():
    # Within 20% is not "keeping up" -- the loader stalls the GPU intermittently.
    v = pt.verdict(decode_rate=380.0, train_rate=350.0)
    assert "marginal" in v.lower()
