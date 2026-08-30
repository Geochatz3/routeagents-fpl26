"""Verify that measurable WNS always produces routing guidance.

Calibrated rules do not cover every feature combination, and feasibility checks
may reject a matching rule under wall-time, memory, or size pressure. In those
cases the out-of-band safety floor selects an existing route-first or
placement-preserving plan and blocks `place_design -unplace`. An empty plan is
permitted only when WNS is unavailable, preventing unguided routing.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimizer.recipe_router import (  # noqa: E402
    PhaseOneFeatures, decide_recipe_path,
)


def _is_silent(plan) -> bool:
    """Exactly the suppression test in dcp_optimizer.py::_build_recipe_router_block."""
    return plan.rule_id == "FALLBACK" and not plan.blocks


def _plan(wns, period, failing, wall=3200.0):
    return decide_recipe_path(PhaseOneFeatures(
        wns_ns=wns, clock_period_ns=period,
        failing_endpoint_count=failing, remaining_wall_budget_s=wall))


# Deterministic: a fresh seed per run would make this flaky in exactly the way
# that hides a rare hole.
SEED = 20260729
SAMPLES = 20_000


def test_never_silent_when_wns_is_known():
    rng = random.Random(SEED)
    violations = []
    for _ in range(SAMPLES):
        wns = -math.exp(rng.uniform(math.log(0.01), math.log(60.0)))
        period = math.exp(rng.uniform(math.log(0.5), math.log(40.0)))
        failing = int(math.exp(rng.uniform(math.log(1), math.log(1_000_000))))
        if rng.random() < 0.15:
            period = None
        if rng.random() < 0.15:
            failing = None
        wall = rng.choice([0.0, 600.0, 3200.0])
        plan = _plan(wns, period, failing, wall)
        if _is_silent(plan):
            violations.append((wns, period, failing, wall))
            if len(violations) >= 5:
                break
    assert not violations, (
        "the router returns NOTHING for a design whose WNS is known, so the LLM "
        f"would freelance on it: {violations}")


@pytest.mark.parametrize("wns,period,failing", [
    (-9.0, 2.0, 500),        # deep miss, too few failing for R1
    (-20.0, 1.5, 1_000),     # huge miss, low failing
    (-3.0, 4.0, 10_000),     # mid miss, mid ratio — squarely between islands
    (-0.4, 10.0, 100),       # near-met but a very low ratio
    (-12.0, None, 50_000),   # deep miss, ratio unknowable
    (-6.0, 3.0, None),       # failing count unmeasurable
])
def test_named_out_of_band_shapes_get_guidance(wns, period, failing):
    plan = _plan(wns, period, failing)
    assert not _is_silent(plan), f"silent on {(wns, period, failing)}"
    assert plan.actions or plan.blocks


def test_an_unmeasurable_wns_is_the_only_way_to_get_silence():
    """The residual hole, stated rather than hidden.

    With no WNS there is no region to reason about, and the design note is
    explicit that dressing that up as a decision would be dishonest. So silence
    here is intended — but it must remain the ONLY route to silence.
    """
    plan = _plan(None, None, None)
    assert _is_silent(plan), (
        "a totally unmeasured design no longer reaches FALLBACK — if that is "
        "deliberate, this test should be updated to say what it reaches instead")


def test_out_of_band_plans_never_unplace():
    """The OOB floor must stay never-worse: it may not throw away the placement.

    `place_design -unplace` is the expensive, non-recoverable move. The floor
    exists to be safe on designs nothing was calibrated on, so it routes to
    already-proven builders and blocks the unplace outright.
    """
    rng = random.Random(SEED + 1)
    checked = 0
    for _ in range(4_000):
        wns = -math.exp(rng.uniform(math.log(0.01), math.log(60.0)))
        period = math.exp(rng.uniform(math.log(0.5), math.log(40.0)))
        failing = int(math.exp(rng.uniform(math.log(1), math.log(1_000_000))))
        plan = _plan(wns, period, failing)
        if plan.rule_id != "OOB":
            continue
        checked += 1
        assert "place_design -unplace" in (plan.blocks or ()), (
            f"an OOB plan failed to block unplace: {(wns, period, failing)}")
        names = {getattr(a, "name", str(a)) for a in (plan.actions or ())}
        assert names, "an OOB plan with no actions is the fail-open shape"
    assert checked > 100, f"only {checked} OOB samples — the sweep stopped covering it"


# R3 routing is conditional on affordability.
# Wall pressure, memory dominance, or excessive cell count sends an
# otherwise matching design to the out-of-band floor.

# (name, measured initial Fmax MHz, measured clock period ns, failing endpoints)
R3_DESIGNS = [
    ("finn_radioml", 284.90, 1.600, 46_438),
    ("amd_mini-isp", 307.13, 1.566, 4_887),
    ("vexriscv", 310.17, 1.570, 1_937),
]


def _rule(name_fmax_period_failing, **over):
    _, f0, period, failing = name_fmax_period_failing
    wns = 1000.0 / f0 - period
    kw = dict(wns_ns=-wns, clock_period_ns=period,
              failing_endpoint_count=failing, remaining_wall_budget_s=3200.0)
    kw.update(over)
    return decide_recipe_path(PhaseOneFeatures(**kw)).rule_id


@pytest.mark.parametrize("design", R3_DESIGNS, ids=lambda d: d[0])
def test_r3_designs_are_in_band_only_while_the_wall_is_healthy(design):
    assert _rule(design) == "R3"
    assert _rule(design, remaining_wall_budget_s=1200.0) == "OOB", (
        f"{design[0]} no longer falls to the OOB floor under wall pressure — if R3 "
        "became affordable at 1200s, the cheapest live confirmation of the OOB path "
        "(MAX_WALL=1200 on mini-ISP) no longer works and needs re-deriving")


@pytest.mark.parametrize("design", R3_DESIGNS, ids=lambda d: d[0])
def test_wall_pressure_still_yields_a_usable_plan(design):
    """Falling to the floor must never mean falling silent."""
    _, f0, period, failing = design
    wns = 1000.0 / f0 - period
    plan = _plan(-wns, period, failing, wall=1200.0)
    assert not _is_silent(plan)
    assert "place_design -unplace" in (plan.blocks or ()), (
        "a wall-pressured design got a plan that may discard its placement")
