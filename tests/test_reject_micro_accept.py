"""Test the optional gate that rejects improvements below the meaningful-gain
threshold.

Accepted results replace the seed used by later combinations, so small gains
can alter subsequent search. The gate is disabled by default; these tests cover
switch behavior and threshold arithmetic in ns.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FLAG = "FPL26_ILS_REJECT_MICRO_ACCEPT"


@pytest.fixture
def ils(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    import optimizer.ils_polish as m
    return importlib.reload(m)


def test_default_is_off_so_the_ship_path_is_unchanged(ils):
    assert ils.reject_micro_accept_enabled() is False, (
        "this changes the ILS accept rule on one design's evidence; it must not "
        "be armed by default before an A/B")


@pytest.mark.parametrize("val,expected", [
    ("1", True), ("true", True), ("on", True), ("yes", True),
    ("0", False), ("", False), ("no", False),
])
def test_flag_parses_like_its_siblings(monkeypatch, val, expected):
    monkeypatch.setenv(FLAG, val)
    import optimizer.ils_polish as m
    m = importlib.reload(m)
    assert m.reject_micro_accept_enabled() is expected


def test_the_fir_numbers_are_what_the_gate_would_have_caught(ils):
    """Verify that a gain between the accept margin and meaningful-gain threshold
    is accepted normally but rejected by the micro-accept gate.
    """
    cfg = ils.ILSPolishConfig()
    shipref_best_before = -0.247      # recipe baseline that run inherited
    cycle6_wns = -0.243               # identical result in BOTH runs
    gain = cycle6_wns - shipref_best_before
    assert gain == pytest.approx(0.004, abs=0.0005)
    # Accepted under the shipped rule...
    assert gain > cfg.accept_margin_ns, (
        "if the accept margin now exceeds this gain, the shipped rule already "
        "rejects it and this flag is redundant")
    # ...and refused under the gate.
    assert gain < cfg.meaningful_accept_ns


def test_the_other_run_never_faced_the_decision(ils):
    """night16c's better baseline meant cycle 6 was not an improvement at all, so
    no accept rule could have fired. The two runs differ in INHERITED STATE, not
    in the rule — which is what makes this a natural experiment rather than a
    configuration difference."""
    cfg = ils.ILSPolishConfig()
    night16c_best_before = -0.216
    cycle6_wns = -0.243
    assert cycle6_wns < night16c_best_before + cfg.accept_margin_ns


def test_the_trade_is_asymmetric(ils):
    """Worst case forgone is one meaningful threshold; observed upside is 12.23 MHz.

    I first wrote 0.05 MHz for the downside here and the test caught it: on fir
    (T = 2.5 ns) 0.010 ns of WNS is **1.32 MHz**, 26x what I estimated. The trade
    is still favourable — about 9:1 — but it is not the near-free bet the first
    version of this comment claimed, and a 1.32 MHz worst case on a column worth
    0.128 mean-rank points per MHz is a real cost, not a rounding error.

    Pinned as a RATIO so it fails if either side moves enough to change the bet.
    """
    cfg = ils.ILSPolishConfig()
    T = 2.5
    base = 0.243
    forgone_mhz = 1000.0 / (T + base) - 1000.0 / (T + base + cfg.meaningful_accept_ns)
    observed_upside_mhz = 1000.0 / (T + 0.154) - 1000.0 / (T + 0.243)
    assert forgone_mhz == pytest.approx(1.32, abs=0.05)
    assert observed_upside_mhz == pytest.approx(12.23, abs=0.05)
    assert observed_upside_mhz / forgone_mhz > 5.0, (
        f"upside {observed_upside_mhz:.2f} MHz vs worst-case forgone "
        f"{forgone_mhz:.2f} MHz — the asymmetry this gate relies on no longer holds")
