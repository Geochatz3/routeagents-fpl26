"""Pure route-time predictor + feasibility assessment for destructive
re-routes.

Predicts how long a routed-state-DESTROYING re-route (post `-unroute` /
post `place_design`) will take on THIS design on THIS machine, and assesses
whether it — plus the banking that must follow it — fits in the remaining
wall budget. Feature-keyed only: prior heavy-op timings this run (via
optimizer.ils_polish.derive_cost_anchors) and the design's primitive cell
count. NO design-name conditionals, NO IO, NO Vivado — a pure function
mirroring the PhaseOneFeatures / decide_recipe_path contract in
optimizer/recipe_router.py so it can be unit-tested exhaustively.

The failure this prevents, observed on an evaluation run of a
379,380-cell design with an effective 3199s window: the agent ran
`route_design -unroute` with 1650s remaining, then a full
AggressiveExplore re-route that was budget-killed at its 1632.9s
allowance, leaving the design unrouted (α=0) despite a completed
1387.8s improving phys_opt. A flat guard (DEFAULT_RISKY_RUNTIME_S =
600) was off by ≥2.7x there; this predictor scales instead.

Locked bias: over-refusing costs a gamble, under-refusing
costs a zero — PREFER REFUSING. Hence max() over both candidates and a
reserve+margin subtraction. Do NOT weaken K or the cells rate without new
cross-design evidence.

All calibration numbers below come from heavy-op timings mined from five
evaluation-harness logs, so the evidence is thin: treat them as
conservative floors, not as a characterisation of the tool.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from optimizer.ils_polish import derive_cost_anchors


# Tunables. Each constant is followed by the measurement that set it.

# Estimate a destructive full reroute as twice the longest completed heavy
# operation. This is not an upper bound; the cell-scaled floor covers cases
# where routing grows more expensive than that ratio.
HEAVY_OP_MULTIPLIER_K = 2.0

# Apply a conservative full-reroute floor of 0.006 seconds per primitive cell.
# The rate targets large designs with aggressive directives, where runtime
# scales superlinearly, and biases uncertain feasibility toward refusal.
CELLS_SCALED_RATE_S = 0.006

# Reserve 30 seconds after rerouting for timing and route checks, checkpoint
# banking, and EDIF mirroring. Large-design WNS measurement variability is
# covered separately by the safety margin.
BANKING_RESERVE_S = 30.0

# Subtract 120 seconds for finalization, lifecycle overhead, and timing
# measurement variability before testing feasibility. The reroute prediction
# remains size-scaled because a flat wall-time threshold is insufficient.
SAFETY_MARGIN_S = 120.0


# A bare reroute starts from the banked routed state and preserves recoverability.
# Estimate it at half the destructive reroute cost, leaving headroom relative
# to the expected routed-state runtime.
PRESERVING_ROUTE_FACTOR = 0.5

# State-preserving reroutes use a smaller margin than destructive operations.
# The banked disk artifact remains untouched, and a timeout falls back to that
# candidate rather than leaving finalization with an unrouted design.
# Reserve 30 seconds for measurement and mirroring plus 60 seconds for runtime
# variability; the budget deadline already excludes the finalization reserve.
PRESERVING_BANKING_RESERVE_S = 30.0
PRESERVING_SAFETY_MARGIN_S = 60.0


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


def predict_reroute_seconds(
    tool_call_details: List[dict],
    input_cell_count: Optional[int],
    k: float = HEAVY_OP_MULTIPLIER_K,
    cells_rate_s: float = CELLS_SCALED_RATE_S,
) -> float:
    """Predict a destructive full re-route's duration on this design/machine.

    predicted = max(k × largest completed heavy-op time this run,
                    input_cell_count × cells_rate_s)

    The K-branch uses the run's own completed place/route/phys_opt timings
    as an on-machine speed signal (via derive_cost_anchors — reused, not
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
        # If both cell count and completed-operation history are unavailable,
        # return an infinite estimate so destructive reroutes fail closed.
        return float("inf")
    return max(k * largest_heavy_op_s, cells * cells_rate_s)


def predict_preserving_reroute_seconds(
    tool_call_details: List[dict],
    preserve_factor: float = PRESERVING_ROUTE_FACTOR,
) -> float:
    """Estimate the duration in seconds of a state-preserving bare reroute.

    The estimate multiplies the current run's single-stage routing sample by
    `PRESERVING_ROUTE_FACTOR`. It applies neither the destructive predictor's
    two-cycle multiplier nor its cell-scaled floor because the existing routed
    state remains recoverable after an overrun. Returns infinity when no
    routing sample is available, causing the budget gate to skip an operation
    that cannot be sized for the current environment.
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
    margin on the budget side). The caller (the gate wiring) turns an
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
