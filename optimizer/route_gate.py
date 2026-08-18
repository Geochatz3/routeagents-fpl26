"""Pure route-time predictor + feasibility assessment for destructive
re-routes (R-D1-2, the D1 fix).

Predicts how long a routed-state-DESTROYING re-route (post `-unroute` /
post `place_design`) will take on THIS design on THIS box, and assesses
whether it — plus the banking that must follow it — fits in the remaining
wall budget. Feature-keyed only: prior heavy-op timings this run (via
optimizer.ils_polish.derive_cost_anchors) and the design's primitive cell
count. NO design-name conditionals, NO IO, NO Vivado — a pure function
mirroring the PhaseOneFeatures / decide_recipe_path contract in
optimizer/recipe_router.py so it can be unit-tested exhaustively.

The failure this prevents (official beta eval, 2026-07-14, 379,380-cell
design, effective window 3199s): the LLM ran `route_design -unroute` with
1650s remaining, then a full AggressiveExplore re-route that was
budget-killed at its 1632.9s allowance → design left unrouted → α=0
despite a completed 1387.8s improving phys_opt. The old flat guard
(DEFAULT_RISKY_RUNTIME_S = 600) was off by ≥2.7x; this predictor SCALES.

Locked bias (01-CONTEXT.md): over-refusing costs a gamble, under-refusing
costs a zero — PREFER REFUSING. Hence max() over both candidates and a
reserve+margin subtraction. Do NOT weaken K or the cells rate without new
cross-design evidence.

All calibration numbers below are from the eval-box op-timing ADDENDUM
(01-RESEARCH.md, mined from all five official 2026-07-14 harness logs).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from optimizer.ils_polish import derive_cost_anchors


# ---------------------------------------------------------------------------
# Constants (evidence-commented, recipe_router-style)
# ---------------------------------------------------------------------------

# K=2.0 is VALIDATED against the single complete full-reroute data point in
# the ADDENDUM: 84,422-cell design, largest completed phys_opt 47.6s →
# predicted 2.0 × 47.6 = 95.2s vs 94.95s observed full route after
# place_design. It is NOT a uniform upper bound on the route/phys_opt ratio
# (observed span: 12k-cell design ≈ 3.0, 84k-cell ≈ 2–4, all above 2.0; the
# 379k-cell D1 design gives only an incomplete ≥1.18) — the cells-scaled
# floor below is the backstop for ratios above 2. On the D1 failure itself
# the K-branch refuses regardless: 2.0 × 1387.8 = 2775.6s > 1650s remaining.
HEAVY_OP_MULTIPLIER_K = 2.0

# Conservative full-re-route floor in seconds per primitive cell.
# Derivation: the D1 design's killed AggressiveExplore re-route gives
# ≥1632.9s on 379,380 cells = ≥0.0043 s/cell as a LOWER BOUND (the run was
# killed incomplete; the true cost is unknown and higher). 0.006 ≈ 1.4×
# that lower bound, per the locked "prefer refusing" bias. A floor at or
# below 0.0043 would still allow the cold-start gamble: e.g. 0.004 gives a
# 1517.5s floor that PASSES at ~1700–2000s remaining while the real need is
# ≥1633s and likely higher. At 0.006 the floor is 379,380 × 0.006 = 2276.3s
# and cold-start destroying ops on designs of that class are refused below
# ~2426s remaining (floor + reserve + margin). The 84k-cell design's
# measured 0.0011 s/cell (default-class directive) confirms small/mid
# designs stay comfortably feasible under this rate (size × directive
# scaling is super-linear; the rate must cover the aggressive-directive
# large-design corner, not the average).
CELLS_SCALED_RATE_S = 0.006

# Post-re-route banking envelope: WNS measure + report_route_status +
# write_checkpoint + EDIF mirror. The eval-box timing report on the D1
# design took ~52s alone but the DESTROYING-op decision point only needs
# the incremental banking tail after a successful route (report + mirror,
# ~25–30s observed on the mirror machinery's [mirror] enter/exit spans).
# The WNS-measure cost on big designs is already inside SAFETY_MARGIN_S.
BANKING_RESERVE_S = 30.0

# Conservative margin subtracted from the remaining wall before comparing.
# Covers the finalize/lifecycle tail and measurement slop on large designs
# (~52s timing report on the D1 design) — same spirit as the existing
# finalize reserve and R3_REMAINING_WALL_MIN_S wall-feasibility precedents
# (recipe_router.py), but SCALED prediction + margin instead of a flat
# constant, which CONTEXT.md flags as insufficient when unscaled.
SAFETY_MARGIN_S = 120.0


# ---------------------------------------------------------------------------
# Bare re-route (state-PRESERVING) cost basis — drill jul21 D1-null:
# +0.094 boom_v2 deterministic; meta: undirected beats directed.
# ---------------------------------------------------------------------------
# Measured preserving ratio (drill1_vivado_full.log, boom_v2, same box &
# session): bare route_design from the banked ROUTED state = 1199s
# (line 1570) vs the full AggressiveExplore re-route from unrouted =
# 3871s (line 1061) → ratio 0.31.  0.5 ships ≈1.6× headroom over that
# single-design measurement.  A factor ≥ ~0.7 makes the gate refuse on
# the leg-2 eval trace (route sample 1620s, ~1300s remaining post-bank)
# and the lever becomes dead code at eval — the K=2 destructive basis
# refused BOTH there (2×1620=3240) and at DEBUG-WALL (7200 vs 2375).
PRESERVING_ROUTE_FACTOR = 0.5

# Slimmer margin stack than the destructive gate ON PURPOSE (risk
# posture, coordinator-sanctioned jul21): the destructive gate's
# failure mode is α:=0 (unrouted design shipped), so it stacks 30s
# banking + 120s margin on a doubled prediction.  The bare re-route is
# INSURED — state-preserving, banked disk mirror untouched, an overrun
# is budget-timeout-killed and finalize ships the banked best via the
# emergency path — so failure costs only the tail wall (γ penalty,
# −0.1·α·γ scale), while success is a deterministic +0.094-class α
# gain.  30s banking (measure + mirror tail, same as destructive) +
# 60s margin (measurement slop only; no finalize tail here — the
# _budget_deadline already excludes the 300s finalize reserve).
PRESERVING_BANKING_RESERVE_S = 30.0
PRESERVING_SAFETY_MARGIN_S = 60.0


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RerouteAssessment:
    """Outcome of the destructive-re-route feasibility check.

    feasible: True iff predicted_reroute_s + banking reserve fits in
      remaining_wall_s minus the safety margin.
    predicted_reroute_s: conservative predicted full re-route seconds.
    remaining_wall_s: the remaining wall the caller passed (echoed for
      logs/steering messages).
    reason: human-readable explanation naming the predicted time vs the
      effective budget (and the shortfall when infeasible).
    """
    feasible: bool
    predicted_reroute_s: float
    remaining_wall_s: float
    reason: str


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def predict_reroute_seconds(
    tool_call_details: List[dict],
    input_cell_count: Optional[int],
    k: float = HEAVY_OP_MULTIPLIER_K,
    cells_rate_s: float = CELLS_SCALED_RATE_S,
) -> float:
    """Predict a destructive full re-route's duration on this design/box.

    predicted = max(k × largest completed heavy-op time this run,
                    input_cell_count × cells_rate_s)

    The K-branch uses the run's own completed place/route/phys_opt timings
    as an on-box speed signal (via derive_cost_anchors — reused, not
    reimplemented). The cells-scaled branch is a conservative floor that
    covers COLD-START (no heavy op completed yet this run) — the exact
    gamble the K-branch cannot see. The result never drops below the
    cells-scaled floor and never returns 0 for a real design.

    Malformed history entries (elapsed_time None / missing keys / non-dict)
    are skipped without raising. Inputs are not mutated. Pure: no IO.
    """
    safe_history = [tc for tc in (tool_call_details or [])
                    if isinstance(tc, dict)]
    _, _, maxes = derive_cost_anchors(safe_history)
    largest_heavy_op_s = max(maxes.values()) if maxes else 0.0
    cells = float(input_cell_count or 0)
    if cells < 0:
        cells = 0.0
    if largest_heavy_op_s <= 0.0 and cells <= 0.0:
        # Double-blind case (jul20 external review S2): cell-count
        # measurement failed AND no heavy op has completed — max(0, 0)
        # would declare every destructive re-route "feasible" exactly
        # when we know nothing. Prefer-refusing bias: no data, no gamble.
        return float("inf")
    return max(k * largest_heavy_op_s, cells * cells_rate_s)


def predict_preserving_reroute_seconds(
    tool_call_details: List[dict],
    preserve_factor: float = PRESERVING_ROUTE_FACTOR,
) -> float:
    """Predict a state-PRESERVING bare re-route's duration on this
    design/box (drill jul21 D1-null: +0.094 boom_v2 deterministic;
    meta: undirected beats directed).

    predicted = observed single-stage route_design sample this run
                (derive_cost_anchors maxes — the same sample source the
                destructive predictor's K-branch uses) × preserve_factor

    NO K=2 doubling and NO cells-scaled floor: those exist because a
    DESTROYED state must be fully rebuildable in-window (overrun there
    = unrouted design = α:=0).  A bare route_design on a routed design
    is insured — see PRESERVING_ROUTE_FACTOR above for the measured
    basis and risk posture.

    No route sample this run → inf (honest skip: without an on-box
    route sample the op cannot be sized — same prefer-refusing
    double-blind posture as the destructive predictor; the designs this
    lever targets have banked a route by construction, so a missing
    sample means the history itself is broken).
    """
    safe_history = [tc for tc in (tool_call_details or [])
                    if isinstance(tc, dict)]
    _, _, maxes = derive_cost_anchors(safe_history)
    route_sample_s = float(maxes.get("route_design", 0.0) or 0.0)
    if route_sample_s <= 0.0:
        return float("inf")
    return route_sample_s * preserve_factor


def assess_preserving_reroute(
    remaining_wall_s: float,
    tool_call_details: List[dict],
    preserve_factor: float = PRESERVING_ROUTE_FACTOR,
    banking_reserve_s: float = PRESERVING_BANKING_RESERVE_S,
    safety_margin_s: float = PRESERVING_SAFETY_MARGIN_S,
) -> RerouteAssessment:
    """Assess whether ONE state-preserving bare re-route + banking fits
    the remaining wall.

    feasible ⇔ predicted + banking_reserve_s
               ≤ remaining_wall_s − safety_margin_s

    Same result shape as assess_destructive_reroute but with the
    preserving cost basis (no K=2, no cells floor) and the slimmer
    margin stack — the failure mode costs tail-γ, not α (constants
    above).  The caller (_run_bare_reroute_polish) has already charged
    the banked-best re-open cost against remaining_wall_s.
    """
    predicted = predict_preserving_reroute_seconds(
        tool_call_details, preserve_factor=preserve_factor)
    effective_budget_s = remaining_wall_s - safety_margin_s
    need_s = predicted + banking_reserve_s
    if predicted == float("inf"):
        feasible = False
        reason = (
            "REFUSE: no route_design sample in this run's history — "
            "cannot size the bare re-route (honest skip, no gamble)"
        )
    elif need_s <= effective_budget_s:
        feasible = True
        reason = (
            f"OK: predicted bare re-route {predicted:.1f}s + banking "
            f"reserve {banking_reserve_s:.0f}s = {need_s:.1f}s fits "
            f"effective budget {effective_budget_s:.1f}s (remaining "
            f"{remaining_wall_s:.1f}s - safety margin "
            f"{safety_margin_s:.0f}s)"
        )
    else:
        feasible = False
        shortfall_s = need_s - effective_budget_s
        reason = (
            f"REFUSE: predicted bare re-route {predicted:.1f}s + banking "
            f"reserve {banking_reserve_s:.0f}s = {need_s:.1f}s exceeds "
            f"effective budget {effective_budget_s:.1f}s (remaining "
            f"{remaining_wall_s:.1f}s - safety margin "
            f"{safety_margin_s:.0f}s) by {shortfall_s:.1f}s"
        )
    return RerouteAssessment(
        feasible=feasible,
        predicted_reroute_s=predicted,
        remaining_wall_s=remaining_wall_s,
        reason=reason,
    )


def assess_destructive_reroute(
    remaining_wall_s: float,
    tool_call_details: List[dict],
    input_cell_count: Optional[int],
    banking_reserve_s: float = BANKING_RESERVE_S,
    safety_margin_s: float = SAFETY_MARGIN_S,
    k: float = HEAVY_OP_MULTIPLIER_K,
    cells_rate_s: float = CELLS_SCALED_RATE_S,
) -> RerouteAssessment:
    """Assess whether a routed-state-destroying re-route can complete AND
    be banked within the remaining wall budget.

    feasible ⇔ predicted_reroute_s + banking_reserve_s
               ≤ remaining_wall_s − safety_margin_s

    Biased toward refusal by construction (max() predictor, reserve and
    margin on the budget side). The caller (gate wiring, Plan 03) turns an
    infeasible verdict into a steering message toward bankable incremental
    alternatives; this function only judges.
    """
    predicted = predict_reroute_seconds(
        tool_call_details, input_cell_count, k=k, cells_rate_s=cells_rate_s)
    effective_budget_s = remaining_wall_s - safety_margin_s
    need_s = predicted + banking_reserve_s
    feasible = need_s <= effective_budget_s
    if feasible:
        reason = (
            f"OK: predicted re-route {predicted:.1f}s + banking reserve "
            f"{banking_reserve_s:.0f}s = {need_s:.1f}s fits effective budget "
            f"{effective_budget_s:.1f}s (remaining {remaining_wall_s:.1f}s "
            f"- safety margin {safety_margin_s:.0f}s)"
        )
    else:
        shortfall_s = need_s - effective_budget_s
        reason = (
            f"REFUSE: predicted re-route {predicted:.1f}s + banking reserve "
            f"{banking_reserve_s:.0f}s = {need_s:.1f}s exceeds effective "
            f"budget {effective_budget_s:.1f}s (remaining "
            f"{remaining_wall_s:.1f}s - safety margin {safety_margin_s:.0f}s)"
            f" by {shortfall_s:.1f}s — a destroyed routed state could not be "
            f"re-routed and banked in time"
        )
    return RerouteAssessment(
        feasible=feasible,
        predicted_reroute_s=predicted,
        remaining_wall_s=remaining_wall_s,
        reason=reason,
    )
