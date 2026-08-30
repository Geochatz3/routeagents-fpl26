"""Verify the effective scope of the first-stage band gate.

The affordability override bypasses the band when the measured cost fits the
wall time remaining after the tail reserve. Otherwise, the band may reject work
outside its extreme-failure criteria. The threshold is pinned by the
tail-reserve clamp and deep-replacement cost margin; split-aware retries use
their remaining wall time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimizer.deep_replace_sibling import (  # noqa: E402
    DEEP_REPLACE_COST_MARGIN, deep_replace_should_run,
)
from optimizer.recipe_router import (  # noqa: E402
    R1_FAILING_ENDPOINTS_MIN, R1_ROUTE_FIRST_WNS_NS,
)

EVAL_WALL_S = 3500.0

# Tuple fields are design, failing endpoints, |WNS| in ns, and place-and-route cost in s.
# One fixture lacks a WNS sample and uses a placeholder; its verdict depends only
# on the failing-endpoint count.
CORPUS = [
    ("boom_soc_v2",       220_131, 11.39, 1650),   # MEASURED
    ("corescore_500_mod",  39_008,  1.44, 1099),   # PARTIAL — |WNS| not banked
    ("finn_radioml",       46_438,  1.91,  684),   # MEASURED
    ("rosetta_optical",     7_866,  1.08,  367),   # MEASURED
    ("vtr_mcml_v2",        18_773, 12.88,  343),   # MEASURED
    ("rosetta_digit",      22_946,  1.02,  213),   # MEASURED
    ("logicnets_jscl",      1_529,  0.98,  161),   # MEASURED
    ("rosetta_spam",        6_786,  0.69,  114),   # MEASURED
    ("fir_systolic",          252,  0.31,   54),   # MEASURED
    ("amd_mini-isp",        4_887,  1.69,   37),   # MEASURED (shipdef_a)
    ("vexriscv_v2",         2_933,  0.95,   18),   # MEASURED
    ("vexriscv",            1_937,  1.65,   15),   # MEASURED
]

# The one design where band ON and band OFF give different first-stage decisions.
BAND_SENSITIVE = {"corescore_500_mod"}


def _first_stage_budget(wall_s: float) -> float:
    from dcp_optimizer import TAIL_RESERVE_WALL_CLAMP_FRAC
    return wall_s * (1.0 - TAIL_RESERVE_WALL_CLAMP_FRAC)


def _override_fires(anchor_s: float, wall_s: float = EVAL_WALL_S) -> bool:
    """The affordability override: a stage that fits the leftover budget."""
    return anchor_s * DEEP_REPLACE_COST_MARGIN <= _first_stage_budget(wall_s)


def test_threshold_is_846_seconds_on_the_eval_wall():
    budget = _first_stage_budget(EVAL_WALL_S)
    assert budget == pytest.approx(1100.0, abs=1.0), (
        "the first-stage budget moved; the band gate's live scope moved with it")
    assert budget / DEEP_REPLACE_COST_MARGIN == pytest.approx(846.0, abs=1.0), (
        "the anchor threshold moved; re-derive which designs the band gate can "
        "reach before trusting any prior band A/B")


@pytest.mark.parametrize("name,failing,wns,anchor", CORPUS)
def test_band_gate_changes_nothing_outside_the_sensitive_set(name, failing, wns, anchor):
    """Run the real gate both ways and compare the decision, not the alpha."""
    override = _override_fires(anchor)
    verdicts = {}
    for band_on in (True, False):
        # The override runs BEFORE the band is consulted, so an affordable stage
        # reaches deep_replace_should_run with require_physics_band False either way.
        require_band = band_on and not override
        run, _why = deep_replace_should_run(
            enabled=True,
            pristine_dcp="/nonexistent/pristine.dcp",
            failing_endpoint_count=failing,
            wns_magnitude_ns=wns,
            remaining_s=EVAL_WALL_S,
            cost_basis_s=float(anchor),
            finalize_reserve_s=300.0,
            failing_endpoints_min=R1_FAILING_ENDPOINTS_MIN,
            wns_min_ns=R1_ROUTE_FIRST_WNS_NS,
            require_physics_band=require_band,
        )
        verdicts[band_on] = run

    differs = verdicts[True] != verdicts[False]
    if name in BAND_SENSITIVE:
        assert differs, (
            f"{name} was the ONLY design the band gate could change; it no longer "
            "does, so the gate now has no measured effect anywhere in the corpus")
    else:
        assert not differs, (
            f"the band gate now changes {name}'s first-stage decision "
            f"(anchor {anchor}s, override={override}). Any band A/B measured on "
            "this design before this change was measuring nothing; re-run it.")


def test_spam_cannot_inform_the_band_decision():
    """The design all six band rows were measured on is a no-op case."""
    assert _override_fires(114), (
        "spam's 114s anchor no longer clears the affordability override — the "
        "spam rows would now be informative, which they were not")


def test_only_one_corpus_design_is_band_sensitive():
    sensitive = set()
    for name, failing, wns, anchor in CORPUS:
        if _override_fires(anchor):
            continue
        deep = failing >= R1_FAILING_ENDPOINTS_MIN and wns >= R1_ROUTE_FIRST_WNS_NS
        if not deep:
            sensitive.add(name)
    assert sensitive == BAND_SENSITIVE, (
        f"the band gate's live scope changed: {sorted(sensitive)} vs "
        f"{sorted(BAND_SENSITIVE)}. Re-price the ship/revert decision.")
