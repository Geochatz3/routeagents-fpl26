"""Implements stagnation-triggered iterated local search as a placement polish
step.

Each cycle unplaces and replaces the design with another directive, accepting
only a strict improvement over the current best. Stagnation triggering
preserves budget while the main optimization loop remains productive. All
implementation work runs through tool Tcl calls without LLM involvement.
Failures leave the existing best result unchanged.
"""
from __future__ import annotations
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional, List, Tuple

# Proven directive rotation (winners first; keep-best filters the rest).
# Each tuple = (place_directive, route_directive, phys_opt_directive).
# Sentinel place-directive: this combo does targeted ruin (the worst paths'
# cells only) instead of a full unplace.
PARTIAL_RUIN_PD = "__PARTIAL_RUIN__"
# Sentinel for the UG906 last-mile cycle (phys_opt clock_opt/retime/lut ->
# place_design -directive LastMile -> phys_opt Explore -> route). See the
# ILS_COMBOS entry for the probe evidence.
LASTMILE_PD = "__LASTMILE__"
# Sentinel for a route-only cycle: keep the placement, unroute, and re-route
# with the combo's route directive.  Every other combo re-places first, so
# this is the only operator class that leaves placement untouched.
ROUTE_ONLY_PD = "__ROUTE_ONLY__"
# Sentinel for a route re-roll: a full `route_design -unroute` followed by a
# from-scratch `route_design -directive AggressiveExplore`, and nothing else
# (no phys_opt tail).  Solving the routing from scratch on a good placement
# re-rolls the route lottery instead of inheriting the incumbent solution's
# compromises; ripping up and repairing in place yields far less.  The picker
# gates this to near-met states — designs with deep negative slack are served
# by the bare-reroute loop in the main controller instead.
# Kill switch: cfg.route_reroll_enabled / --no-route-reroll /
# FPL26_NO_ROUTE_REROLL=1.
ROUTE_REROLL_PD = "__ROUTE_REROLL__"
# Sentinel for an incremental escalated re-route: route-only without the
# `-unroute`, so the solver refines the incumbent routing under a more
# aggressive directive rather than re-solving from scratch.
#
# The escalation is the mechanism, not the repetition: re-routing with the
# SAME directive gains almost nothing, and running the aggressive directive
# first and the milder one second is measurably worse than the other order.
# Both other sentinels unroute first, so neither can express this.
# Armed by FPL26_ILS_INCR_ROUTE=1; appended late, and keep-best means a
# design it does not suit pays one cheap cycle rather than losing Fmax.
INCR_ROUTE_PD = "__INCR_ROUTE__"
# Scope of the targeted ruin, in worst paths.  Swept on a large near-met
# design: gain is unimodal in scope and peaks here, on total gain and on
# gain per second.  Overridable via FPL26_PARTIAL_RUIN_SCOPE for further
# sweeps; an invalid or sub-1 value keeps the default.
PARTIAL_RUIN_SCOPE_DEFAULT = 200


def partial_ruin_scope() -> int:
    env = os.environ.get("FPL26_PARTIAL_RUIN_SCOPE", "").strip()
    if env:
        try:
            v = int(float(env))
            if v >= 1:
                return v
        except ValueError:
            pass
    return PARTIAL_RUIN_SCOPE_DEFAULT


def partial_ruin_tcl(scope: int = PARTIAL_RUIN_SCOPE_DEFAULT) -> str:
    return (
        f"set paths [get_timing_paths -quiet -max_paths {int(scope)} "
        "-nworst 1 -setup]; "
        "set pcells [get_cells -quiet -of_objects "
        "[get_pins -quiet -of_objects $paths]]; "
        "set cells [filter -quiet $pcells {IS_PRIMITIVE && "
        "(PRIMITIVE_GROUP == \"REGISTER\" || PRIMITIVE_GROUP == \"LUT\" || "
        "PRIMITIVE_GROUP == \"CARRY\" || PRIMITIVE_GROUP == \"MUXF\" || "
        "PRIMITIVE_GROUP == \"FLOP_LATCH\" || PRIMITIVE_GROUP == \"CLB\")}]; "
        "if {[llength $cells] > 0} {unplace_cell $cells}; "
        "puts \"RUIN_CELLS=[llength $cells]\""
    )


# Back-compat constant (default scope) — existing imports/tests unchanged.
PARTIAL_RUIN_TCL = partial_ruin_tcl()

ILS_COMBOS: List[Tuple[str, str, str]] = [
    # Ordered by measured accept counts and cycle cost across a multi-run
    # audit: directives that accept often and cheaply come first, and the
    # costliest directive with a single accept was demoted below them.
    ("Explore", "Explore", "AlternateFlowWithRetiming"),
    # Last-mile at index 1: the strongest accept evidence of any combo on
    # plateaued states, which is exactly where ruin combos stop moving.  It
    # must sit inside the futility window — with the K=2 stop, combos at
    # index >= 2 are unreachable on a run whose first two cycles do not
    # accept — and it is a different operator class from ruin (netlist-level
    # phys_opt plus an incremental re-place), so both classes get a try
    # before the search gives up.
    (LASTMILE_PD, "Explore", "Explore"),
    # Route re-roll at index 2: unroute plus a from-scratch aggressive route,
    # two operations only (the phys_opt slot is deliberately unused; its
    # budget basis is a full route x1.3, see route_reroll_cost_basis).
    # The picker's near-met eligibility gate makes this slot
    # class-conditional: on near-met designs it sits next to last-mile inside
    # the early window, and on deep-negative-slack designs the gate skips it
    # inline within the same pick, consuming no cycle, so their effective
    # rotation is unchanged by its presence.
    (ROUTE_REROLL_PD, "AggressiveExplore", "Explore"),
    # Route-only: re-route the current placement with the aggressive
    # directive.  The cheapest cycle in the set (no place step) and the
    # broadest single lever measured — a third operator class, with
    # placement untouched, distinct from both ruin and last-mile.  It sits at
    # the edge of the futility window so a stuck run tries all three classes
    # before the K=2 stop.  Hold gate: the standard accept floor (-0.001),
    # which matches the official scorecard gate.
    (ROUTE_ONLY_PD, "AggressiveExplore", "Explore"),
    ("ExtraTimingOpt", "AggressiveExplore", "AggressiveExplore"),
    # Partial ruin is promoted above the two expensive spread/net-delay
    # directives so that the cheap (0.7x) validated finisher executes on any
    # budget rather than only under budget pressure: given a roomy window the
    # expensive combos run first, do poorly on the stuck class, and burn the
    # K=2 futility budget before partial ruin can fire.
    # It unplaces only the worst paths' fabric cells and re-places
    # incrementally, at about two thirds of a full-ruin cycle's cost.
    (PARTIAL_RUIN_PD, "Explore", "AggressiveExplore"),
    ("AltSpreadLogic_high", "AggressiveExplore", "AggressiveExplore"),
    ("ExtraNetDelay_high", "Explore", "AlternateFlowWithRetiming"),
    ("SSI_SpreadLogic_high", "Explore", "AlternateFlowWithRetiming"),
    ("EarlyBlockPlacement", "AggressiveExplore", "AggressiveExplore"),
]

# ---- Place-retry extension.  Default off = byte-identical behaviour ----
#
# The rotation above was ordered by an audit of accept counts, which cannot
# see a directive that was never in the set.  Neither directive below has
# been: only their `_high` siblings, which are ordered late because they cost
# 2-3x a full-ruin cycle.  That cost objection does not apply to these two —
# both come in at roughly the price of a plain Explore cycle, and both beat
# it on the designs where the plain cycle regresses.
PLACE_RETRY_COMBOS: List[Tuple[str, str, str]] = [
    ("AltSpreadLogic_medium", "Explore", "AlternateFlowWithRetiming"),
    ("ExtraNetDelay_low", "Explore", "AlternateFlowWithRetiming"),
]

# Which directive the regression trigger forces.  Among the candidates, this
# is the only one measured positive on every design known to satisfy the
# trigger; another candidate is worth more on one of them but regresses the
# other.  Forcing a substitute that regresses a triggering design is not
# merely a wasted cycle — it burns that design's second futility strike and
# ends the search before the rotation can reach anything better.
#
# It is already in ILS_COMBOS, so forcing it adds no combo and does not
# change the rotation's length.  The extension above stays appended so the
# higher-value alternative remains reachable later, on designs with the
# budget to get there.
PLACE_RETRY_TARGET = "ExtraNetDelay_high"

# ---- Probe ladder.  Strictly additive to the single forced pick. ----
#
# The ladder's first rung is PLACE_RETRY_TARGET itself, so an armed ladder
# does everything the single retry did, in the same order, before it
# continues.  Nothing that works today can be displaced by arming it.
#
# One substitute cannot serve every design: the best directive is
# design-specific and the runner-up costs real Fmax.  A ladder does not have
# to choose — keep-best rejects the loser at the cost of one cycle.  The
# single retry could not do this because a rejected probe burned a futility
# strike, so the ladder is only sound together with the futility exemption
# below, and only affordable with FPL26_ILS_MEASURED_BASIS, since its own
# first rung is the 3.0x directive that the inflated cold-start anchor
# prices out.
#
# Ordered by evidence rather than cost, so a design that stops early stops
# on the best-supported rung.
PLACE_RETRY_LADDER = ["ExtraNetDelay_high", "AltSpreadLogic_medium", "ExtraNetDelay_low"]


# The rung a NEAR-MET design wants first. Kept as a name, not an index, because the
# ladder is a list of directives and the rotation's indices move.
LADDER_NEAR_MET_FIRST = "AltSpreadLogic_medium"


def ladder_order_by_wns_enabled() -> bool:
    """Reports whether WNS-based ILS ladder ordering is enabled.

    The feature is enabled by default and disabled by
    ``FPL26_ILS_LADDER_ORDER_BY_WNS=0``. The 0.7 ns magnitude threshold is
    inherited unchanged from ``cfg.route_reroll_max_wns_mag`` so related
    routing decisions share one near-met boundary.
    """
    return os.environ.get(
        "FPL26_ILS_LADDER_ORDER_BY_WNS", "1").strip().lower() in ("1", "true", "on", "yes")


def ladder_rungs(baseline_wns: Optional[float] = None,
                 near_met_mag: Optional[float] = None) -> List[str]:
    """Returns the ILS directive ladder in execution order.

    ``FPL26_ILS_LADDER_ORDER`` provides a comma-separated explicit order and
    takes precedence over automatic ordering. The first rung runs from the
    unmodified seed; later rungs run from the latest accepted state, so
    reordering changes their base placement. Automatic ordering starts with
    fine spreading for near-met timing and stronger net-delay weighting for
    deep negative slack. The near-met threshold is shared with the route-reroll
    gate rather than fitted independently.
    """
    raw = os.environ.get("FPL26_ILS_LADDER_ORDER", "").strip()
    if raw:
        # Accept "|" as well as ",": the A/B driver splits its ENV spec on
        # commas, so a comma-separated value cannot survive that transport.
        got = [x.strip() for x in raw.replace("|", ",").split(",") if x.strip()]
        if got:
            return got
    order = list(PLACE_RETRY_LADDER)
    if (ladder_order_by_wns_enabled()
            and baseline_wns is not None and near_met_mag
            and abs(baseline_wns) < near_met_mag
            and LADDER_NEAR_MET_FIRST in order):
        order.remove(LADDER_NEAR_MET_FIRST)
        order.insert(0, LADDER_NEAR_MET_FIRST)
    return order


def place_retry_ladder_enabled() -> bool:
    """Default on. Disable with FPL26_ILS_PLACE_RETRY_LADDER=0.

    Was opt-in, which meant the eval path never armed it — see the module note above.
    """
    return os.environ.get(
        "FPL26_ILS_PLACE_RETRY_LADDER", "1").strip().lower() in ("1", "true", "on", "yes")


# This optional gate narrows the retry trigger: a fresh placement must regress
# against the untouched input, not only against the polished incumbent.
# If baseline timing is unavailable or invalid, the gate is bypassed and the
# original incumbent-relative trigger remains active.
def retry_baseline_gate_enabled() -> bool:
    """Armed only by FPL26_ILS_RETRY_BASELINE_GATE."""
    return os.environ.get(
        "FPL26_ILS_RETRY_BASELINE_GATE", "0").strip().lower() in ("1", "true", "on", "yes")


def _retry_baseline_gate_active(cfg) -> bool:
    """Reports whether the retry baseline gate is active.

    The gate is active when either its global environment flag or
    ``cfg.retry_baseline_gate_scoped`` is enabled. The scoped setting may limit
    the gate to mid-band designs with input WNS magnitude between 1.05 ns and
    8.0 ns. OR composition preserves the global override while leaving the
    default disabled when neither source is armed.
    """
    return retry_baseline_gate_enabled() or bool(
        getattr(cfg, "retry_baseline_gate_scoped", False))


# Optional ladder refinements are independently gated.
# Later rungs require enough budget for the rung and one additional full cycle.
# The first rung remains exempt from the reserve gate.
def ladder_reserve_enabled() -> bool:
    """Armed only by FPL26_ILS_LADDER_RESERVE."""
    return os.environ.get(
        "FPL26_ILS_LADDER_RESERVE", "0").strip().lower() in ("1", "true", "on", "yes")


# Unaffordable ladder rungs are skipped within the cycle already claimed.
# Ladder order is unchanged; only refusal advances to the next rung.
# _rungs_popped counts pops, not executions, so a skipped rung cannot inherit
# the first rung's reserve exemption.
def ladder_skip_unaffordable_enabled() -> bool:
    """Default on. Kill switch: FPL26_ILS_LADDER_SKIP_UNAFFORDABLE=0."""
    return os.environ.get(
        "FPL26_ILS_LADDER_SKIP_UNAFFORDABLE", "1").strip().lower() in (
            "1", "true", "on", "yes")


# Ladder rungs may optionally borrow from the wrapper window beyond the fenced
# ILS deadline while preserving the finalize reserve.
# Borrowing is capped at 500 seconds, the size of the polish reserve.
# All non-ladder gates and loop exit conditions continue using the fenced deadline.
# A missing or malformed wrapper deadline yields zero borrow and fails closed.
# Borrowed time may reduce a live polish stage, so this feature remains opt-in.
WALL_FENCE_MAX_BORROW_S_DEFAULT = 500.0


def ils_wall_fence_enabled() -> bool:
    """Default off. Enable: FPL26_ILS_WALL_FENCE=1;
    FPL26_NO_ILS_WALL_FENCE=1 force-disables and wins."""
    if os.environ.get("FPL26_NO_ILS_WALL_FENCE", "").strip().lower() in (
            "1", "true", "on", "yes"):
        return False
    return os.environ.get("FPL26_ILS_WALL_FENCE", "").strip().lower() in (
        "1", "true", "on", "yes")


# The high net-delay directive is gated by failing-endpoint density:
# failing_endpoints / critical_path_avg_spread_tiles.
# The threshold separates concentrated failures from sparse, dispersed failures;
# missing or non-finite features fail open.
# The redraw reserve protects 1200 seconds of wrapper budget for another attempt.
# A meaningful ILS accept disarms the reserve; malformed wrapper timing fails open.
# The reserve cannot fire during the initial cycle floor, allowing ILS time to earn it.
ENDHIGH_PD = "ExtraNetDelay_high"
ENDHIGH_DENSITY_MIN_DEFAULT = 363.0
# Reject the high-cost directive when its estimate exceeds 60% of the remaining window;
# the limit prevents one speculative cycle from consuming nearly all available time.
ENDHIGH_SHARE_MIN_DEFAULT = 0.60
# The multi-restart attempt floor. Kept as a named constant so the reserve and
# the wrapper cannot drift apart into a reserve that funds nothing.
REDRAW_RESERVE_S_DEFAULT = 1200.0
# Do not classify ILS as unproductive until five cycles have started.
# This floor allows a useful fourth-cycle result to disarm the redraw reserve on merit.
REDRAW_MIN_CYCLES_DEFAULT = 5


def endhigh_density_gate_enabled() -> bool:
    """Reports whether the high-density end-stage ILS gate is enabled.

    The gate is enabled by default and disabled by
    ``FPL26_ILS_ENDHIGH_DENSITY_GATE=0``. The default lives in Python because
    build-layer settings override Python defaults, while this policy applies
    unless explicitly disabled.
    """
    return os.environ.get(
        "FPL26_ILS_ENDHIGH_DENSITY_GATE", "1").strip().lower() in (
            "1", "true", "on", "yes")


def endhigh_density_min() -> float:
    """Threshold, env-overridable via FPL26_ILS_ENDHIGH_DENSITY_MIN.

    A non-numeric or non-positive override is IGNORED (falls back to the
    default) rather than silently disabling the gate -- a
    broken probe must never read as a pass.
    """
    raw = os.environ.get("FPL26_ILS_ENDHIGH_DENSITY_MIN", "").strip()
    if raw:
        try:
            v = float(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return ENDHIGH_DENSITY_MIN_DEFAULT


def redraw_reserve_enabled() -> bool:
    """Armed only by FPL26_ILS_REDRAW_RESERVE. Default off."""
    return os.environ.get(
        "FPL26_ILS_REDRAW_RESERVE", "0").strip().lower() in (
            "1", "true", "on", "yes")


def redraw_reserve_s() -> float:
    """Seconds to leave for a wrapper re-draw, env FPL26_ILS_REDRAW_RESERVE_S.

    Default 1200 = the multi-restart attempt floor (scripts/multi_restart_
    optimize.py). Leaving LESS than the floor funds nothing, which is exactly the
    fir failure; leaving more just donates wall the wrapper cannot use. A
    non-numeric or non-positive override is IGNORED rather than silently
    disabling the reserve.
    """
    raw = os.environ.get("FPL26_ILS_REDRAW_RESERVE_S", "").strip()
    if raw:
        try:
            v = float(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return REDRAW_RESERVE_S_DEFAULT


def endhigh_share_min() -> float:
    """Budget-share ceiling, env-overridable via FPL26_ILS_ENDHIGH_SHARE_MIN.

    Garbage or out-of-range (<=0 or >1) is IGNORED in favour of the default.
    """
    raw = os.environ.get("FPL26_ILS_ENDHIGH_SHARE_MIN", "").strip()
    if raw:
        try:
            v = float(raw)
            if 0.0 < v <= 1.0:
                return v
        except ValueError:
            pass
    return ENDHIGH_SHARE_MIN_DEFAULT


def endhigh_density_blocked(cfg, est_s=None, remaining_s=None) -> Tuple[bool, str]:
    """Returns whether ``ExtraNetDelay_high`` is blocked and the reason for that
    decision.

    Blocking requires both conditions: density indicates that the directive is
    unlikely to help, and its estimated cost would consume a large share of the
    remaining window. Density estimates expected benefit, while cost share
    limits expensive speculative cycles. The gate fails open when disabled or
    when density, spread, estimated cost, or remaining time is missing,
    non-finite, or non-positive.
    """
    if not endhigh_density_gate_enabled():
        return False, ""
    spread = getattr(cfg, "critical_path_avg_spread_tiles", None)
    failing = getattr(cfg, "phase1_failing_endpoints", None)
    if spread is None or failing is None:
        return False, ("endhigh-density: UNMEASURED "
                       f"(spread={spread}, failing={failing}) -> fail-open")
    try:
        spread = float(spread)
        failing = float(failing)
    except (TypeError, ValueError):
        return False, "endhigh-density: unparseable features -> fail-open"
    # Check finiteness before comparisons: comparisons with NaN are false, while
    # infinite spread produces a valid-looking zero density.
    # Either case must fail open rather than silently blocking the directive.
    if not (math.isfinite(spread) and math.isfinite(failing)):
        return False, (f"endhigh-density: non-finite features "
                       f"(spread={spread}, failing={failing}) -> fail-open")
    if spread <= 0:
        return False, f"endhigh-density: spread={spread} <= 0 -> fail-open"
    dens = failing / spread
    thr = endhigh_density_min()
    if dens >= thr:
        return False, (f"endhigh-gate: density {dens:.0f} >= {thr:.0f} -> ARM "
                       f"(failing={failing:.0f} / spread={spread:.1f})")
    # Density alone does not block a cycle; combine it with the budget-risk gate.
    # This avoids suppressing cheap cycles that still fit the remaining window.
    try:
        est = float(est_s)
        rem = float(remaining_s)
    except (TypeError, ValueError):
        return False, (f"endhigh-gate: density {dens:.0f} < {thr:.0f} but cycle "
                       f"unpriceable (est={est_s}, remaining={remaining_s}) "
                       f"-> fail-open")
    if not (math.isfinite(est) and math.isfinite(rem)) or rem <= 0 or est <= 0:
        return False, (f"endhigh-gate: density {dens:.0f} < {thr:.0f} but "
                       f"est/remaining non-finite or non-positive "
                       f"(est={est}, remaining={rem}) -> fail-open")
    share = est / rem
    smin = endhigh_share_min()
    if share < smin:
        return False, (f"endhigh-gate: density {dens:.0f} < {thr:.0f} but share "
                       f"{share:.0%} < {smin:.0%} -> ARM (small bet; every "
                       f"accept this gate would otherwise lose sits below here)")
    return True, (f"endhigh-gate: density {dens:.0f} < {thr:.0f} AND share "
                  f"{share:.0%} >= {smin:.0%} -> BLOCK (est {est:.0f}s of "
                  f"{rem:.0f}s remaining; corpus accept rate below the density "
                  f"line is 0-15% at ~1300s a cycle)")


# Stop the placement-family ladder after its first accepted rung; the accept
# identifies a viable family and later rungs would spend full cycles against it.
# Incremental routing requires an incumbent not yet routed at the target aggression.
# When eligible, schedule it before from-scratch reroute sentinels, which destroy
# that precondition. Do not unroute or run full placement before the escalation.
# The priority move remains a route-only cycle with no placement step.
def incr_route_first_enabled() -> bool:
    """Report whether incremental routing receives first priority during
    escalation.

    The switch defaults on and is disabled with FPL26_ILS_INCR_ROUTE_FIRST=0;
    FPL26_ILS_INCR_ROUTE must also be enabled. The displacement guard bounds
    any queue jump.
    """
    return os.environ.get(
        "FPL26_ILS_INCR_ROUTE_FIRST", "1").strip().lower() in ("1", "true", "on", "yes")


def shallow_escalation_guard_enabled() -> bool:
    """Report whether displacement-free incremental-route escalation is enabled.

    FPL26_SHALLOW_ESCALATION_GUARD defaults off. When enabled, escalation
    precedes a reroll only if the remaining budget covers both operations,
    preserving the later work already admitted by the guard. Keep-best prevents
    checkpoint regression, although the added cycle may exclude an unrelated
    later combination.
    """
    return os.environ.get(
        "FPL26_SHALLOW_ESCALATION_GUARD", "0").strip().lower() in (
            "1", "true", "on", "yes")


def ladder_stop_on_accept_enabled() -> bool:
    """Armed only by FPL26_ILS_LADDER_STOP_ON_ACCEPT."""
    return os.environ.get("FPL26_ILS_LADDER_STOP_ON_ACCEPT",
                          "0").strip().lower() in ("1", "true", "on", "yes")


# Incremental routing is terminal: run it only near the futility stop or when no
# full-placement combo remains affordable.
# This preserves full ruin cycles while allowing one final inexpensive route move.
def incr_route_terminal_enabled() -> bool:
    """Armed only by FPL26_ILS_INCR_ROUTE_TERMINAL."""
    return os.environ.get(
        "FPL26_ILS_INCR_ROUTE_TERMINAL", "0").strip().lower() in ("1", "true", "on", "yes")


# The stuck-gain threshold selects a raw seed for small recipe gains and the
# recipe-best seed otherwise.
# The environment override permits tuning without changing the configured default.
# The raw-first path retains the better result, so a poor threshold costs only time.
def stuck_gain_threshold(cfg) -> float:
    """cfg.stuck_recipe_gain_ns unless FPL26_ILS_STUCK_GAIN_NS overrides it."""
    raw = os.environ.get("FPL26_ILS_STUCK_GAIN_NS", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return float(getattr(cfg, "stuck_recipe_gain_ns", 0.15))


# ---- INCREMENTAL ESCALATED RE-ROUTE (integration of the probe above) --------
INCR_ROUTE_COMBOS: List[Tuple[str, str, str]] = [
    (INCR_ROUTE_PD, "AggressiveExplore", "AlternateFlowWithRetiming"),
]


def incr_route_enabled() -> bool:
    """Default on. Disable with FPL26_ILS_INCR_ROUTE=0 for the shipped rotation."""
    return os.environ.get(
        "FPL26_ILS_INCR_ROUTE", "1").strip().lower() in ("1", "true", "on", "yes")


def place_retry_enabled() -> bool:
    """Armed only by FPL26_ILS_PLACE_RETRY. Unset/0 -> rotation exactly as shipped."""
    return os.environ.get(
        "FPL26_ILS_PLACE_RETRY", "0").strip().lower() in ("1", "true", "on", "yes")


def reject_micro_accept_enabled() -> bool:
    """Armed only by FPL26_ILS_REJECT_MICRO_ACCEPT. Default off.

    Refuses an accept whose gain is below ``cfg.meaningful_accept_ns`` so the seed
    stays clean for later combos. See the long note at the accept site: on
    fir_systolic a 0.0040 ns micro-accept replaced the seed and turned a -0.154
    ExtraTimingOpt cycle into -0.327, costing 12.23 MHz.

    OFF by default because that is ONE natural experiment on ONE design and this is
    the ILS's core accept rule.
    """
    return os.environ.get(
        "FPL26_ILS_REJECT_MICRO_ACCEPT", "0").strip().lower() in (
            "1", "true", "on", "yes")


# These sentinels reuse an existing placement rather than running full placement.
# Exclude them from place-retry diagnosis and from the full-place cost basis because
# neither their regressions nor their runtimes characterize a placement cycle.
PLACE_SENTINELS = (LASTMILE_PD, ROUTE_ONLY_PD, PARTIAL_RUIN_PD, ROUTE_REROLL_PD,
                   INCR_ROUTE_PD)

# The cold-start anchor can overestimate ILS cost because it comes from an earlier
# tool session. After a full-place cycle completes, normalize its runtime by the
# directive prior and use the maximum observed basis.
# The maximum remains conservative; the 1.15 margin covers runtime variation.
MEASURED_BASIS_MARGIN = 1.15


def measured_basis_enabled() -> bool:
    """Report whether cycle pricing uses an observed duration instead of the
    cold-start anchor.

    Measured-basis pricing defaults on and is disabled with
    FPL26_ILS_MEASURED_BASIS=0.
    """
    return os.environ.get(
        "FPL26_ILS_MEASURED_BASIS", "1").strip().lower() in ("1", "true", "on", "yes")


def cold_start_basis(cfg, measured_basis_s: float) -> float:
    """Return the cycle-cost basis in seconds for an unseen directive combination.

    When measured-basis pricing is enabled and a full-place cycle has been
    observed, the observed wall time is returned. Otherwise, the function
    returns cfg.expected_heavy_cycle_s unchanged.
    """
    if measured_basis_s > 0 and measured_basis_enabled():
        return measured_basis_s * MEASURED_BASIS_MARGIN
    return float(getattr(cfg, "expected_heavy_cycle_s", 0.0) or 0.0)


def active_combos() -> List[Tuple[str, str, str]]:
    """Rotation in use. Extended ONLY when explicitly armed.

    Extensions APPEND, never reorder, so ILS_COMBOS[0..3] keep their indices and
    every unarmed rotation-exhaustion assertion still holds.
    """
    combos = list(ILS_COMBOS)
    if place_retry_enabled() or place_retry_ladder_enabled():
        combos += list(PLACE_RETRY_COMBOS)
    if incr_route_enabled():
        combos += list(INCR_ROUTE_COMBOS)
    return combos


# Relative cycle-cost priors use a full-ruin Explore cycle as 1.0.
# They apply only to cold-start affordability when an absolute anchor is available.
# The optional measured-prior map substitutes median values for overstated priors.
# Median pricing limits false refusals that can forfeit the remaining optimization path.
MEASURED_COMBO_COST_PRIOR: dict = {
    "ExtraNetDelay_high": 2.10,   # corpus p50 (shipped 3.0 = p90)
}


def measured_priors_enabled() -> bool:
    """Default off. FPL26_ILS_MEASURED_PRIORS=1 arms the measured priors."""
    return os.environ.get(
        "FPL26_ILS_MEASURED_PRIORS", "0").strip().lower() in ("1", "true", "on", "yes")


def combo_cost_prior(pd: str) -> float:
    """Relative cycle-cost prior for `pd`, in Explore=1.0 units.

    THE ONLY read path. Every affordability site must go through this so the
    override cannot be half-applied -- a half-applied constant is the same
    failure mode as a half-deployed module.
    """
    if measured_priors_enabled():
        _m = MEASURED_COMBO_COST_PRIOR.get(pd)
        if _m is not None:
            return _m
    return COMBO_COST_PRIOR.get(pd, 1.0)


COMBO_COST_PRIOR: dict = {
    "Explore": 1.0,
    # route-only skips the place step entirely; measured per-design route+physopt
    # times were ~0.4-0.7x of a full-ruin cycle on the same design/machine.
    ROUTE_ONLY_PD: 0.6,
    # route-reroll = route step only, no phys_opt tail (ROUTE_ONLY minus its
    # phys_opt): slightly under the ROUTE_ONLY prior. Used only when no route
    # sample exists to build the x1.3 basis (route_reroll_cost_basis == 0).
    ROUTE_REROLL_PD: 0.55,
    "ExtraTimingOpt": 1.4,
    "AltSpreadLogic_high": 2.0,
    "ExtraNetDelay_high": 3.0,
    PARTIAL_RUIN_PD: 0.7,
    LASTMILE_PD: 2.5,
    "SSI_SpreadLogic_high": 1.1,
    "EarlyBlockPlacement": 1.1,
    # Medium- and low-effort directives are costed below their high-effort variants.
    "AltSpreadLogic_medium": 1.05,
    "ExtraNetDelay_low": 1.2,
    # Incremental route-plus-polish assumes an already-routed design: do not
    # unroute or place first. Its factor reflects a fraction of a full reroute.
    INCR_ROUTE_PD: 0.5,
}

# Defaults (overridable). Tuned from the 13-design sweep.
DEFAULT_MAX_CELLS = 300_000          # size gate: above this, <2 cycles fit
# Require a real execution window because the dispatcher skips risky implementation
# tools below roughly 5–8 minutes. A 1500-second minimum funds about two full cycles.
DEFAULT_MIN_REMAINING_S = 1500.0
DEFAULT_ACCEPT_MARGIN_NS = 0.002     # strictly-better threshold
DEFAULT_MAX_CYCLES = 60              # hard backstop vs runaway loop
DEFAULT_MIN_CYCLE_SECONDS = 20.0     # a real cycle takes minutes; faster = skipped
# Measure stagnation in wall time because one LLM iteration spans many tool calls
# and is too coarse to preempt a stalled search within budget.
DEFAULT_STAGNATION_SECONDS = 600.0   # ~10 min w/o improvement -> stalled


@dataclass
class ILSPolishConfig:
    enabled: bool = True
    max_cells: int = DEFAULT_MAX_CELLS
    min_remaining_s: float = DEFAULT_MIN_REMAINING_S
    accept_margin_ns: float = DEFAULT_ACCEPT_MARGIN_NS
    stagnation_seconds: float = DEFAULT_STAGNATION_SECONDS
    max_cycles: int = DEFAULT_MAX_CYCLES
    min_cycle_seconds: float = DEFAULT_MIN_CYCLE_SECONDS
    # Loop-exit trigger floor: lower than the mid-loop min (it's "use whatever
    # budget is left after the recipe finished"), enough for >=1 cycle.
    exit_min_remaining_s: float = 600.0
    # Restart Vivado before global replacement because recipe-phase process state can
    # slow or degrade later placement. Reopening a checkpoint does not clear that state.
    restart_vivado_before_ils: bool = True
    # Seed global ruin-and-recreate from the raw checkpoint to obtain a true
    # from-scratch placement; keep-best selection prevents output regression.
    seed_from_raw: bool = True
    # Treat recipe gains below 0.15 ns as negligible and seed from raw.
    # Otherwise preserve the recipe-best seed; unknown gain also uses recipe-best.
    # This threshold separates marginal movement from material recipe improvement.
    stuck_recipe_gain_ns: float = 0.15
    # Measure timing after routing and before physical optimization. Skip
    # physical optimization if routing is incomplete because such a cycle
    # cannot pass. Also skip when WNS trails the best beyond the configured
    # recovery margin. The 0.5 ns default exceeds expected phys-opt recovery; 0
    # disables this margin check.
    physopt_skip_margin_ns: float = 0.5
    # Acceptance checks worst hold slack after setup and routedness pass.
    # A missing or unparsable hold result fails open.
    # The -0.001 ns floor tolerates report noise near zero.
    accept_requires_hold_clean: bool = True
    hold_slack_floor_ns: float = -0.001
    # Optional exploration may start the next cycle from a routed regression
    # within the configured WNS distance of the global best.
    # The best checkpoint remains untouched, so exploration cannot degrade output.
    explore_from_current: bool = False
    explore_regression_floor_ns: float = 0.15
    # For a design classified as stuck, try the raw seed first and then use
    # remaining budget for a recipe-best corrective seed.
    # Other designs use only recipe-best.
    # Global keep-best selection prevents either seed from degrading output.
    dual_seed: bool = True
    # Stop a primary seed after this many consecutive real cycles without acceptance,
    # leaving budget for the corrective seed. The final or sole seed has its own limit.
    # A value of 0 disables this early stop.
    no_improve_stop_cycles: int = 2
    # Stop the final or sole seed after consecutive cycles without meaningful
    # progress. Two cycles limit plateau cost while allowing one retry; 0 runs
    # to the budget limit.
    final_seed_no_improve_stop: int = 2
    # Attempt the LastMile combination only when the current best WNS is near
    # closure; the placement directive may fail outright on deeply negative
    # slack. Accepted gains below 0.010 ns remain saved but count toward the
    # no-improvement streak. This keeps noise-scale gains from extending an
    # otherwise unproductive search.
    meaningful_accept_ns: float = 0.010
    lastmile_min_wns_ns: float = -1.0
    # Estimated wall time, in seconds, for one full-ruin cycle on this design
    # and machine. The caller derives it from recipe-phase placement, routing,
    # and phys-opt durations. It prevents starting an expensive cycle unlikely
    # to finish within the budget. Zero means unknown and retains optimistic
    # cold-start selection.
    expected_heavy_cycle_s: float = 0.0
    # PRISTINE design baseline WNS (dcp_optimizer's self.initial_wns), NOT the
    # recipe-best that run_ils_polish receives as baseline_wns. Only consumed by the
    # retry baseline gate. None = unmeasured -> that gate does not apply.
    design_baseline_wns: Optional[float] = None
    # Scope retry-baseline gating to the moderate-slack band.
    # The caller enables this only when retry gating is active and
    # 1.05 < |input WNS| < 8.0 ns; false leaves the gate unscoped.
    retry_baseline_gate_scoped: bool = False
    # Allow one cycle beyond the futility stop when its scoring benefit exceeds
    # the projected wall-time cost. Banked performance is supplied in MHz.
    # The hurdle scales as alpha * 0.1 * (cycle_seconds / 3600) / P.
    # Zero disables continuation; the maximum-MHz setting caps its threshold.
    hurdle_continue_alpha_mhz: float = 0.0
    hurdle_continue_max_mhz: float = 1.0
    # Apply this timeout to heavy placement, routing, and phys-opt commands.
    # Each command is also capped by the remaining overall budget.
    # A timeout makes the cycle incomplete, and keep-best discards its result.
    # The 30-minute default accommodates slow stages without relaxing the wall deadline.
    heavy_cmd_timeout_s: float = 1800.0
    # Per-command timeout for the LIGHT steps (open_checkpoint, WNS/route_status
    # queries). Fast normally, but slow on a huge design; well above 300s avoids a
    # false no-op when the design is merely large, not stuck.
    measure_cmd_timeout_s: float = 600.0
    # Treat the post-transform cell-count band as a sanity check, not a
    # validity criterion. Counts are compared with the input design to detect
    # wholesale netlist corruption. A missing reference count or failed
    # measurement fails open.
    golden_cell_count: Optional[int] = None
    cell_floor_ratio: float = 0.5
    cell_ceil_ratio: float = 3.0
    # Run one AggressiveFanoutOpt pass on the final best after combo rotation,
    # so the normal plateau stop cannot suppress this dedicated polish.
    # Replication can reduce hold slack, so candidates must meet the stricter
    # 0.010 ns floor. Cost gating skips slow designs; rejected or incomplete
    # candidates leave the best unchanged.
    fanout_polish_enabled: bool = True
    fanout_hold_slack_floor_ns: float = 0.010
    # Fanout exploration requires a known cold-start cost anchor below this limit.
    # Unknown anchors and slow designs are excluded because another cycle may not fit.
    fanout_max_cycle_s: float = 600.0
    # Estimate fanout-polish cost as one phys-opt pass plus one possible
    # reroute, using single-stage durations collected during the recipe phase.
    # When a route sample exists, this anchor takes precedence over the full-
    # cycle estimate. Otherwise use the conservative full-cycle anchor when
    # available; zero means unknown.
    fanout_cost_anchor_s: float = 0.0
    fanout_anchor_has_route: bool = False
    # Let the corrective seed accept stepping stones against its own local baseline;
    # global keep-best selection still controls the emitted checkpoint.
    # Its combo rotation starts at the primary seed's first accepted position.
    # If the primary seed has no accepts, rotation continues from its ending position.
    corrective_local_climb: bool = True
    # Reserve this many seconds after fanout affordability checks for final
    # publication. The best checkpoint is published eagerly, so this covers
    # finalization rather than a full exit path. Polish commands remain
    # deadline-capped; 300 seconds exceeds expected finalization latency.
    fanout_finalize_reserve_s: float = 300.0
    # Applies last-mile optimization to the final global best before fanout polish.
    # A candidate is accepted only if timing improves, routing is complete, hold
    # slack is at least -0.001 ns, and the cell-count guard passes.
    # The banked checkpoint preserves never-worse behavior.
    lastmile_polish_enabled: bool = True
    # Allows ILS to improve designs that already meet timing when budget remains.
    # Positive setup slack increases the derived maximum frequency, so the met
    # checkpoint remains the acceptance floor while normal hold and futility
    # guards continue to apply.
    met_surplus_ils: bool = True
    # Skips partial-ruin moves when the critical path is tightly co-located, where
    # unplacing several path cells is unlikely to help and only consumes budget.
    # The 30-tile floor separates compact paths from paths with useful placement
    # spread. Missing spread data fails open; other combo families remain eligible.
    partial_ruin_spread_gate: bool = True
    partial_ruin_spread_min_tiles: float = 30.0
    # Route re-roll is eligible only for near-met states with enough budget for a
    # complete route using the sampled route cost and safety margin.
    # Unknown slack fails closed because eligibility requires confirmed shallow
    # negative slack. The operation may unroute the in-session state, but the
    # banked best checkpoint remains routed and is replaced only after acceptance;
    # a timeout therefore discards the cycle without corrupting the saved result.
    route_reroll_enabled: bool = True
    # The 0.7 ns band limits route re-rolls to shallow negative slack, where a
    # routing-only perturbation remains likely to affect the critical path.
    route_reroll_max_wns_mag: float = 0.7
    # Stores the recipe-phase route wall time in seconds, including caller margin.
    # The affordability gate falls back to a route-bearing fanout anchor, then to
    # 0.55 of the estimated full-cycle cost when no direct sample is available.
    route_cost_anchor_s: float = 0.0
    # Phase-1 RapidWright critical-path avg spread (tiles), plumbed by the
    # caller from critical_path_spread_info["avg_distance"]. None = unmeasured.
    critical_path_avg_spread_tiles: Optional[float] = None
    # Absolute wrapper deadline in epoch seconds, before the ILS polish reserve is
    # deducted. Re-draw reserves must use this deadline rather than the shorter
    # ILS deadline. Missing deadline data disables the reserve check and fails open.
    wrapper_deadline_ts: Optional[float] = None
    # Phase-1 failing-endpoint count used with critical-path spread to estimate
    # endpoint density for placement gating. Missing data makes the gate fail open.
    phase1_failing_endpoints: Optional[int] = None
    # Optional terminal re-placement stage that compares each draw with the banked
    # best and adopts only a sufficiently better valid result.
    # Enable with --replace-gamble or FPL26_REPLACE_GAMBLE=1; the environment
    # variable FPL26_NO_REPLACE_GAMBLE=1 takes precedence as a kill switch.
    replace_gamble_enabled: bool = False
    # Adopt bar over the chain-best: +0.15 ns.
    # NOT the 0.002 polish margin — a full re-place discards the chain's
    # accumulated phys_opt polish, so a marginal win is likely noise.
    replace_gamble_adopt_margin_ns: float = 0.15
    # Draw count. Variants
    # beyond this index in REPLACE_GAMBLE_VARIANTS are never attempted.
    replace_gamble_max_draws: int = 2
    # Reserves terminal-finalization time in addition to the margin-adjusted full
    # place-and-route estimate for each draw. The 300-second allowance covers
    # checkpoint publication and final validation.
    replace_gamble_finalize_reserve_s: float = 300.0
    # Gates re-placement on the routing-delay fraction of the critical path.
    # The 0.6 threshold selects paths dominated by routing rather than logic.
    # Missing phase-1 data fails open so the optional stage remains eligible.
    replace_gamble_min_route_delay_frac: float = 0.6
    critical_path_route_delay_frac: Optional[float] = None

    # Optional deep-slack sibling stage that re-places from the pristine input rather
    # than from the banked best. Its output is an insured comparison candidate, so
    # a losing result is discarded and router decisions remain unchanged.
    # Enable with --deep-replace; FPL26_NO_DEEP_REPLACE=1 is the kill switch.
    deep_replace_enabled: bool = False
    # Runs the deep-replacement recipe before the LLM loop so it can consume the
    # full placement-and-routing budget; a tail invocation may not fit.
    # The banked baseline preserves never-worse output, but the LLM loop receives
    # only the remaining time. This option requires deep replacement to be enabled.
    deep_replace_first_enabled: bool = False
    # Terminal finalize reserve on top of the x1.3-margined full place+route
    # anchor (same 300s convention as the other terminal stages).
    deep_replace_finalize_reserve_s: float = 300.0
    # Adopt bar over the chain-best. Same reasoning as replace_gamble's
    # +0.15: a full re-place from PRISTINE discards every bit of chain
    # polish, so a marginal win is almost certainly noise.
    deep_replace_adopt_margin_ns: float = 0.15


@dataclass
class ILSPolishResult:
    triggered: bool = False
    improved: bool = False
    skip_reason: str = ""
    baseline_wns: Optional[float] = None
    best_wns: Optional[float] = None
    cycles: int = 0
    accepted: int = 0
    physopt_skipped: int = 0   # cycles whose phys_opt was provably futile
    # Rotation position after the last combo attempted on the seed's pristine state.
    # A sibling seed may resume there because those attempts share an equivalent
    # starting state. Attempts made after the first acceptance are not included,
    # since acceptance changes the state and leaves those combos fresh for siblings.
    pristine_rot: int = 0
    # Carries observed per-combo cycle costs into sibling seeds on the same machine
    # and design. Reusing these estimates avoids pricing later attempts from cold
    # priors.
    combo_cost: dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.triggered:
            return f"ILS-polish SKIPPED ({self.skip_reason})"
        d = None if (self.best_wns is None or self.baseline_wns is None) \
            else round(self.best_wns - self.baseline_wns, 4)
        return (f"ILS-polish {'IMPROVED' if self.improved else 'no-gain'}: "
                f"wns {self.baseline_wns} -> {self.best_wns} (d={d}) "
                f"cycles={self.cycles} accepted={self.accepted}")


CallTool = Callable[[str, dict], Awaitable[str]]


def should_trigger(*, cells: Optional[int], remaining_s: float,
                   seconds_since_improve: float, best_wns: Optional[float],
                   baseline_wns: Optional[float], cfg: ILSPolishConfig
                   ) -> Tuple[bool, str]:
    """Pure gate (unit-testable, no I/O). Returns (trigger, reason).

    seconds_since_improve = wall-time since best_wns last improved. A productive
    loop keeps this small (no preempt); a stalled loop lets it grow past the
    threshold (preempt to ILS)."""
    if not cfg.enabled:
        return False, "disabled"
    if cells is not None and cells > cfg.max_cells:
        return False, f"too_large(cells={cells}>{cfg.max_cells})"
    if remaining_s < cfg.min_remaining_s:
        return False, f"insufficient_budget({remaining_s:.0f}s<{cfg.min_remaining_s:.0f}s)"
    if seconds_since_improve < cfg.stagnation_seconds:
        return False, (f"loop_still_productive(stall={seconds_since_improve:.0f}s"
                       f"<{cfg.stagnation_seconds:.0f}s)")
    # Headroom: only worth it if not already at/above timing closure.
    if best_wns is not None and best_wns >= 0.0:
        return False, "timing_already_met(wns>=0)"
    return True, "stalled+size-ok+budget-ok"


def should_trigger_at_exit(*, cells: Optional[int], remaining_s: float,
                           best_wns: Optional[float], cfg: ILSPolishConfig
                           ) -> Tuple[bool, str]:
    """Loop-EXIT trigger (generalizable): the agent's main loop ended but timing
    is still unmet (wns<0) and there's leftover budget + the design is small
    enough — so spend the rest on ILS ('if nothing closed it, ruin-and-recreate;
    leave good/closed designs alone'). Independent of mid-loop stagnation, so it
    also catches early LLM 'done' exits. keep-best => never-worse."""
    if not cfg.enabled:
        return False, "disabled"
    if best_wns is None:
        return False, "wns_unknown"
    if best_wns >= 0.0 and not cfg.met_surplus_ils:
        return False, "timing_met_or_unknown"      # closed timing -> leave it
    if cells is not None and cells > cfg.max_cells:
        return False, f"too_large(cells={cells})"
    if remaining_s < cfg.exit_min_remaining_s:
        return False, f"insufficient_budget({remaining_s:.0f}s<{cfg.exit_min_remaining_s:.0f}s)"
    if best_wns >= 0.0:
        # Met with surplus budget: positive slack still buys alpha
        # (fmax = 1/(T - wns)); keep-best floor is the met DCP.
        return True, "loop-exit+met-surplus+budget+size-ok"
    return True, "loop-exit+timing-unmet+budget+size-ok"


def choose_ils_seed(*, initial_wns: Optional[float], recipe_wns: Optional[float],
                    cfg: ILSPolishConfig) -> Tuple[str, Optional[float]]:
    """Pure seed-selection gate. Returns ("raw"|"recipe_best", recipe_gain).

    Seed the ruin-and-recreate from RAW only for STUCK designs (recipe gained ~0
    over the initial design — global re-place from scratch wins, e.g. v2 -> -0.84).
    Where the recipe genuinely improved timing, raw discards that gain and lands
    worse, so seed from recipe-best. Unknown gain -> conservative recipe-best."""
    if not cfg.seed_from_raw:
        return "recipe_best", None
    if (initial_wns is None or recipe_wns is None
            or recipe_wns == float("-inf")):
        return "recipe_best", None
    gain = recipe_wns - initial_wns
    return ("raw" if gain < stuck_gain_threshold(cfg) else "recipe_best"), gain


def fanout_polish_accept(*, new_wns: Optional[float], best_wns: Optional[float],
                         unrouted: Optional[int], whs: Optional[float],
                         base_whs: Optional[float],
                         cfg: ILSPolishConfig) -> Tuple[bool, str]:
    """Decide whether to accept the post-ILS aggressive fanout polish.

    Acceptance requires a fully routed result whose setup slack exceeds the
    current best by accept_margin_ns, whose hold slack is at least
    fanout_hold_slack_floor_ns, and whose hold slack does not worsen from the
    pre-polish best. The strict hold floor protects against hold erosion from
    fanout replication and is intentionally stronger than the general
    combination floor. Rejection preserves the existing best checkpoint.
    """
    if unrouted is None or unrouted != 0:
        return False, f"unrouted={unrouted}"
    if new_wns is None or best_wns is None:
        return False, f"wns unavailable (new={new_wns} best={best_wns})"
    if not (new_wns > best_wns + cfg.accept_margin_ns):
        return False, f"no setup gain (w={new_wns} <= best {best_wns}+{cfg.accept_margin_ns})"
    floor = cfg.fanout_hold_slack_floor_ns
    if whs is None or whs < floor:
        return False, f"hold below strict floor (whs={whs} < {floor})"
    if base_whs is not None and whs < base_whs - 0.001:
        return False, f"hold worsened (whs {base_whs} -> {whs})"
    return True, f"accept w={new_wns} (best {best_wns}) whs={whs}"


def lastmile_polish_accept(*, new_wns: Optional[float], best_wns: Optional[float],
                           unrouted: Optional[int], whs: Optional[float],
                           cell_count: Optional[int],
                           cfg: ILSPolishConfig) -> Tuple[bool, str]:
    """Pure accept for the post-ILS LASTMILE polish. Never-worse on
    setup + fully routed + hold >= -0.001 (matches the official scorecard
    gate, per preview evidence) + cell-count sanity band (LASTMILE's
    -lut_opt can shrink the netlist; the validator's hard Check-4 gate was
    removed upstream — band is sanity-only now)."""
    if unrouted is None or unrouted != 0:
        return False, f"unrouted={unrouted}"
    if new_wns is None or best_wns is None:
        return False, f"wns unavailable (new={new_wns} best={best_wns})"
    if not (new_wns > best_wns + cfg.accept_margin_ns):
        return False, f"no setup gain (w={new_wns} <= best {best_wns}+{cfg.accept_margin_ns})"
    if whs is None or whs < -0.001:
        return False, f"hold below floor (whs={whs} < -0.001)"
    if cfg.golden_cell_count:
        if cell_count is None:
            return False, "cell-count unmeasurable with golden set"
        lo = cfg.golden_cell_count * cfg.cell_floor_ratio
        hi = cfg.golden_cell_count * cfg.cell_ceil_ratio
        if not (lo <= cell_count <= hi):
            return False, (f"cell-count {cell_count} outside sanity band "
                           f"[{lo:.0f},{hi:.0f}]")
    return True, f"accept w={new_wns} (best {best_wns}) whs={whs}"


def derive_cost_anchors(tool_call_details) -> Tuple[float, float, dict]:
    """Derive cycle, fanout-pass, and per-stage cost anchors from observed
    heavy-tool durations.

    Records are classified by tool_name or cmd_head because stages may appear
    as granular tool calls or inside vivado_run_tcl commands. A record matching
    at least two stage keys represents a combined place-to-route sequence and
    contributes one direct full-cycle sample rather than duplicated per-stage
    time.

    The full-cycle anchor is one place-and-route ruin cycle, or 0.0 when no
    placement sample exists. The fanout-pass anchor covers one
    physical-optimization pass plus one reroute using only single-stage
    samples, or 0.0 when neither sample exists. The singles mapping contains
    the maximum observed duration for each stage.

    The function is pure and applies no scheduling margins.
    """
    maxes: dict = {}
    full_cycle = 0.0
    for tc in tool_call_details:
        tn = (str(tc.get("tool_name") or "") + " "
              + str(tc.get("cmd_head") or ""))
        el = tc.get("elapsed_time")
        if el is None:
            continue
        hits = [k for k in ("place_design", "route_design",
                            "phys_opt_design") if k in tn]
        if len(hits) >= 2:
            full_cycle = max(full_cycle, float(el))
        elif len(hits) == 1:
            maxes[hits[0]] = max(maxes.get(hits[0], 0.0), float(el))
    est = 0.0
    if "place_design" in maxes and "route_design" in maxes:
        est = sum(maxes.values())
    if full_cycle > 0:
        est = max(est, full_cycle)
    fan = maxes.get("phys_opt_design", 0.0) + maxes.get("route_design", 0.0)
    return est, fan, maxes


# Aggressive routing can cost more than the recipe's route sample.
# The 1.3 multiplier provides an affordability margin for that directive overhead.
ROUTE_REROLL_COST_MARGIN = 1.3


def route_reroll_cost_basis(cfg: "ILSPolishConfig") -> float:
    """FULL-route cost basis (seconds) for the __ROUTE_REROLL__ affordability
    gate: the best available route sample from the anchors x1.3.

    Preference ladder:
      1. route_cost_anchor_s — the recipe phase's route_design single-stage
         sample (the true analog of the re-roll's one heavy op);
      2. fanout_cost_anchor_s WITH a route sample — overstates by one
         phys_opt pass (conservative: over-refusing a re-roll costs nothing,
         the rotation continues);
      3. 0.0 (unknown) — the picker falls back to the scaled combo prior /
         optimistic cold start; overruns are bounded by the deadline-capped
         heavy timeouts and the never-worse banked mirror (an incomplete
         cycle is discarded, the disk best is untouched).

    NO K=2 doubling (contrast optimizer/route_gate.py's destructive-basis
    LLM gate): here the unroute happens on the in-session copy only; the
    banked best DCP is written only on accept, so the worst case of an
    unaffordable overrun is wasted wall, never an unrouted ship."""
    if cfg.route_cost_anchor_s > 0:
        return cfg.route_cost_anchor_s * ROUTE_REROLL_COST_MARGIN
    if cfg.fanout_cost_anchor_s > 0 and cfg.fanout_anchor_has_route:
        return cfg.fanout_cost_anchor_s * ROUTE_REROLL_COST_MARGIN
    return 0.0


def _tool_ok(resp: object) -> bool:
    """call_tool reports failures as an error-envelope STRING (it never raises):
    a client-side timeout returns {"error": "tool_timed_out_budget", ...}. A cycle
    step that 'returned' such an envelope did NOT run in Vivado, so treating it as
    success would let the cycle continue on stale state — worst case a timed-out
    write_checkpoint getting ACCEPTed with the old/partial file on disk
    (never-worse violation). Same idiom as dcp_optimizer's restart check.

    ALSO: the MCP server wraps every run_tcl command in
    `catch` and reports a Vivado-side failure as plain output "TCL ERROR: <msg>"
    — NOT a client error envelope. Before this check, an instantly-erroring
    place/route/write_checkpoint passed as success: the v2 LASTMILE accept
    recorded a 40ms write_checkpoint (physically impossible) = PHANTOM accept
    with a stale file on disk. Treat TCL ERROR output as failure."""
    if not isinstance(resp, str):
        return True
    # The tool server reports transport failures as plain text beginning with
    # "Error: " rather than a JSON error or Tcl marker. Treat that prefix as
    # failure so a timed-out or dead tool process cannot be accepted as success.
    return not ('"error"' in resp
                or "TCL ERROR:" in resp
                or resp.strip().startswith("Error: "))


async def _measure_hold(call_tool: CallTool, timeout_s: float = 600.0) -> Optional[float]:
    """Worst HOLD slack (WHS), or None if unmeasurable (caller fails open)."""
    try:
        res = await call_tool("vivado_run_tcl", {
            "command": ("set tp [get_timing_paths -quiet -hold -max_paths 1 "
                        "-slack_lesser_than 999]; if {[llength $tp] > 0} "
                        "{get_property SLACK [lindex $tp 0]} else {puts 99.0}"),
            "timeout": timeout_s})
        if not _tool_ok(res):
            return None
        for tok in (res or "").strip().split("\n"):
            tok = tok.strip()
            if not tok or tok.startswith(("ERROR", "WARNING")):
                continue
            try:
                return float(tok)
            except ValueError:
                continue
    except Exception:
        pass
    return None


async def _measure(call_tool: CallTool, wns_tcl: str,
                   timeout_s: float = 600.0) -> Tuple[Optional[float], int]:
    """Return (wns, effective_unrouted). unrouted==0 => clean & fully routed.
    Mirrors the validated standalone parse: 2025.1 report_route_status has NO
    'unrouted nets' line; use routable/fully-routed/errors."""
    wns = None
    try:
        res = await call_tool("vivado_run_tcl", {"command": wns_tcl, "timeout": timeout_s})
        for tok in (res or "").strip().split("\n"):
            tok = tok.strip()
            if not tok or tok.startswith(("ERROR", "WARNING")):
                continue
            try:
                wns = float(tok); break
            except ValueError:
                continue
    except Exception:
        wns = None
    ur = -1
    try:
        rs = await call_tool("vivado_run_tcl", {"command": "report_route_status -return_string",
                                                "timeout": timeout_s})
        def num(label):
            m = re.search(re.escape(label) + r"[^\d]*(\d+)", rs or "")
            return int(m.group(1)) if m else None
        routable = num("# of routable nets")
        fully = num("# of fully routed nets")
        errors = num("# of nets with routing errors")
        if routable is not None and fully is not None:
            ur = (routable - fully) + (errors or 0)
    except Exception:
        ur = -1
    return wns, ur


async def run_ils_polish(
    call_tool: CallTool,
    *,
    best_dcp_path: str,
    baseline_wns: Optional[float],
    deadline_ts: float,
    wns_tcl: str,
    cfg: ILSPolishConfig,
    log: Callable[[str], None],
    no_improve_stop: int = 0,
    combo_offset: int = 0,
    combo_cost_seed: Optional[dict] = None,
) -> ILSPolishResult:
    """Run ruin-and-recreate cycles until the deadline while preserving the best
    routed checkpoint.

    The starting checkpoint must be the agent's current routed best. It is
    overwritten only by a strict improvement.

    A positive no_improve_stop ends the run after that many consecutive real
    cycles without a gain of at least cfg.meaningful_accept_ns. Smaller
    accepted gains remain saved but count as non-improving; zero runs until the
    deadline.

    The combo_offset control continues directive rotation across seeds. It must
    carry forward the preceding seed's cycle count because deterministic
    placement can otherwise repeat the same failed combinations for nearly
    identical checkpoints.
    """
    r = ILSPolishResult(triggered=True, baseline_wns=baseline_wns)
    best = baseline_wns
    r.best_wns = best
    cyc = 0
    noop_streak = 0      # consecutive cycles that didn't actually execute Vivado
    no_improve_streak = 0  # consecutive REAL cycles without a MEANINGFUL accept
    # Tracks wall time per combo because placement directives can have substantially
    # different costs. Each candidate is gated by its own estimate; unseen combos
    # use the cheapest observed cost, with reserve-protected timeouts bounding risk.
    # Costs transfer across sibling seeds because machine and design are unchanged.
    combo_cost: dict = dict(combo_cost_seed) if combo_cost_seed else {}
    r.combo_cost = combo_cost   # live reference; mutated per observed cycle
    tried_since_accept: set = set()  # deterministic placer: same combo + same
    #                                  best DCP => same result; never repeat.
    rot = combo_offset
    # SA exploration state: the DCP each cycle STARTS from. Stays best_dcp_path
    # in greedy mode; in explore mode it may point at the last near-miss state.
    work_path = best_dcp_path
    explore_path = best_dcp_path + ".explore.dcp"
    _prev_lastmile = False   # session-poisoning fix (see in-loop comment)
    _spread_gate_noted = False   # spread gate: one result-note per run
    _rr_budget_noted = False     # route-reroll budget gate: one note per run
    _irf_yield_noted = False     # incr-route priority yielded to an affordable route
    # SHALLOW-GUARD firing audit: "armed but never
    # eligible" must be distinguishable from "not reached" in an A/B grep.
    _seg_armed_ever = False      # guard armed on >= 1 decision cycle
    _seg_logged_ever = False     # >= 1 "SHALLOW-GUARD:" line emitted
    # Rotation actually in use. Identical to ILS_COMBOS unless the
    # place-retry extension is armed (FPL26_ILS_PLACE_RETRY=1).
    _COMBOS = active_combos()
    _retry_jumped = False    # place-retry fires at most once per seed
    _forced_queue = []       # place-retry picks, in order; never advance rot
    _rungs_popped = 0        # ladder rungs consumed from the queue this seed
    # Log the density verdict ONCE per seed. It is a per-design constant, so
    # emitting it every rotation pass would bury the log without adding signal.
    _endhigh_gate_logged = set()
    # Has THIS ILS invocation produced an accept at or above meaningful_accept_ns?
    # Gates the re-draw reserve: a micro-accept does not unlock the reserve, for
    # the same reason it does not reset the futility streak.
    _had_meaningful_accept = False
    _rung_no = 0             # index of the rung running this cycle (0 = shipped slot)
    _meas_basis = 0.0        # observed full-place cycle wall, normalised to Explore=1.0
    _meas_basis_noted = False
    _last_rd = None          # route directive that produced the current incumbent
    # The optional wall fence gives ladder rungs a seed-wide extension in
    # seconds. It is an absolute ceiling at deadline plus the extension,
    # so consecutive rungs cannot compound the borrowed time.
    _wf_extra = 0.0
    if ils_wall_fence_enabled():
        _wf_wdl = getattr(cfg, "wrapper_deadline_ts", None)
        if _wf_wdl:
            try:
                _wf_extra = max(0.0, min(float(_wf_wdl) - deadline_ts,
                                         WALL_FENCE_MAX_BORROW_S_DEFAULT))
            except (TypeError, ValueError):
                _wf_extra = 0.0
        if _wf_extra > 0:
            log(f"ILS wall-fence: ARMED — ladder rungs may judge affordability "
                f"against the wrapper window (+{_wf_extra:.0f}s over the fenced "
                f"deadline); rotation, gates and exits unchanged")
        elif _wf_wdl:
            # A known wrapper deadline equal to the fenced deadline means the
            # polish reserve is zero or released, not that configuration is malformed.
            log("ILS wall-fence: enabled, wrapper window equals the fenced "
                "deadline (polish reserve released/0) — borrow 0s")
        else:
            log("ILS wall-fence: enabled but wrapper_deadline_ts is unknown — "
                "borrow 0s (fail-closed, shipped behaviour)")
    while cyc < cfg.max_cycles:
        remaining = deadline_ts - time.time()
        _wf_admit = False    # this cycle runs a fence-admitted rung (set below)
        # Before the first meaningful accept, preserve enough wall time for a
        # wrapper redraw. Check at the loop head using the measured cycle
        # basis so the cycle about to start cannot consume that reserve.
        _wdl = getattr(cfg, "wrapper_deadline_ts", None)
        if (redraw_reserve_enabled() and not _had_meaningful_accept
                and cyc >= REDRAW_MIN_CYCLES_DEFAULT and _wdl):
            _rr = redraw_reserve_s()
            _next_cost = cold_start_basis(cfg, _meas_basis)
            # Budget this decision against the wrapper deadline, which includes the
            # polish reserve; the fenced ILS remainder is intentionally not used.
            _wrem = _wdl - time.time()
            if _rr > 0 and (_wrem - _next_cost) < _rr:
                remaining = _wrem   # report the budget we actually judged
                log(f"ILS redraw-reserve: no meaningful accept in {cyc} cycles "
                    f"and the next cycle (~{_next_cost:.0f}s) would leave "
                    f"{remaining - _next_cost:.0f}s < the {_rr:.0f}s attempt "
                    f"floor — stopping so the wrapper can re-draw instead. "
                    f"(An unproven stage does not spend a proven alternative's "
                    f"budget; a measured run burned 2000s for +0.006ns and starved a "
                    f"re-draw worth ~6 MHz in expectation.)")
                r.notes.append(f"redraw-reserve: stopped at cycle {cyc}, "
                               f"{remaining:.0f}s returned for a re-draw")
                break
        n = len(_COMBOS)
        pick = None
        # Incremental routing has a perishable precondition: a from-scratch
        # aggressive route destroys the weaker routed incumbent it needs.
        # Consider it first, but let forced placement retries and normal gates
        # win. Jump reroute sentinels only when a priced full reroute cannot
        # fit; unknown cost fails closed.
        _fr_need = route_reroll_cost_basis(cfg)
        _route_still_fits = (_fr_need <= 0.0) or (_fr_need < remaining)
        # The optional shallow-escalation guard permits an early incremental
        # route only when both it and the subsequent full reroute fit.
        # A known positive reroute basis is required to prove this; missing
        # cost data fails closed and preserves the normal yield order.
        _seg_try = (shallow_escalation_guard_enabled() and _route_still_fits
                    and _fr_need > 0.0)
        if _seg_try:
            _seg_armed_ever = True
        if (incr_route_first_enabled() and incr_route_enabled()
                and _last_rd is not None and _route_still_fits
                and not _seg_try
                and not _irf_yield_noted):
            _irf_yield_noted = True
            r.notes.append(
                f"incr-route priority yielded: full-route need {_fr_need:.0f}s "
                f"< remaining {remaining:.0f}s, jump would displace it")
            log(f"[ils] incr-route PRIORITY yielded: a from-scratch route still "
                f"fits (need {_fr_need:.0f}s < remaining {remaining:.0f}s), so the "
                f"jump would DISPLACE it rather than use a dead cycle; 3d measured "
                f"that trade at -5.96 MHz (ROUTE_ONLY +0.079ns vs escalation "
                f"+0.005ns). Escalation stays available on the ordinary path.")
        if (incr_route_first_enabled() and incr_route_enabled()
                and _last_rd is not None
                and (not _route_still_fits or _seg_try)):
            for _ir, _c in enumerate(_COMBOS):
                if (_c[0] == INCR_ROUTE_PD and _ir not in tried_since_accept
                        and _last_rd != _c[1] and not _forced_queue):
                    _ir_est = combo_cost.get(_ir)
                    if _ir_est is None:
                        _b = cold_start_basis(cfg, _meas_basis)
                        _ir_est = (combo_cost_prior(INCR_ROUTE_PD) * _b
                                   if _b > 0 else None)
                    if _seg_try:
                        # Escalation is displacement-free only if its estimated cost
                        # plus the full-reroute cost fits in the remaining window.
                        # An unknown incremental-route cost fails closed.
                        if (_ir_est is not None
                                and remaining >= _ir_est + _fr_need):
                            pick = _ir
                            _seg_logged_ever = True
                            log(f"SHALLOW-GUARD: displacement-free incr-route "
                                f"PRIORITY — incumbent is {_last_rd}-routed, "
                                f"remaining {remaining:.0f}s >= incr "
                                f"{_ir_est:.0f}s + reroll need "
                                f"{_fr_need:.0f}s; the from-scratch route "
                                f"still fits AFTER this cycle, so nothing "
                                f"is displaced (keep-best; S6-eligible).")
                        else:
                            _seg_logged_ever = True
                            log(f"SHALLOW-GUARD: yielded — cannot prove the "
                                f"reroll still fits after the escalation "
                                f"(remaining {remaining:.0f}s vs incr "
                                f"{'unknown' if _ir_est is None else format(_ir_est, '.0f') + 's'}"
                                f" + reroll need {_fr_need:.0f}s); measured "
                                f"yield behaviour stands (3d's -5.96 MHz "
                                f"case is exactly this branch).")
                            # Record the one-time yield outcome here because
                            # the armed guard suppresses the equivalent note in
                            # the earlier branch.
                            if not _irf_yield_noted:
                                _irf_yield_noted = True
                                r.notes.append(
                                    f"incr-route priority yielded: "
                                    f"full-route need {_fr_need:.0f}s < "
                                    f"remaining {remaining:.0f}s, jump "
                                    f"would displace it")
                    elif _ir_est is None or _ir_est < remaining:
                        pick = _ir
                        log(f"[ils] incr-route PRIORITY: incumbent is "
                            f"{_last_rd}-routed and the escalation is eligible now; "
                            f"a from-scratch route is priced out (need "
                            f"{_fr_need:.0f}s >= remaining {remaining:.0f}s), so "
                            f"this cycle displaces nothing (S6)")
                    break
        # Forced placement retries are consumed even when unusable, preventing
        # an invalid entry from wedging the rotation. The default single-rung
        # configuration therefore preserves single-retry behavior.
        _use_forced = False
        _skip_unaff = ladder_skip_unaffordable_enabled()
        # Evaluate ladder-rung affordability against the fenced window.
        # Recomputing from remaining time keeps the total seed-wide borrow
        # bounded by the fixed extension rather than granting it per rung.
        _wf_rem = remaining + _wf_extra
        while _forced_queue:
            _fp = _forced_queue.pop(0)
            # Count queue pops rather than executed runs. The first popped rung
            # owns the unreserved slot even if it is invalid or unaffordable;
            # otherwise a later rung could silently inherit that budget.
            # Each rung examined while skipping unaffordable entries is one pop.
            _rung_no = _rungs_popped
            _rungs_popped += 1
            if not (_fp < n and _fp not in tried_since_accept):
                # Already tried, or out of range. Unarmed this ends the attempt
                # (shipped behaviour); armed, look at the next rung.
                if _skip_unaff and _forced_queue:
                    continue
                break
            # Apply the density policy to forced high-net-delay placement as well
            # as the base rotation. A rejected rung is consumed and logged;
            # skip-unaffordable mode may continue to the next rung.
            if _COMBOS[_fp][0] == ENDHIGH_PD:
                _eh_est = combo_cost.get(_fp)
                _eh_basis = cold_start_basis(cfg, _meas_basis)
                if _eh_est is None and _eh_basis > 0:
                    _eh_est = combo_cost_prior(ENDHIGH_PD) * _eh_basis
                _blk, _why = endhigh_density_blocked(cfg, _eh_est, remaining)
                if _blk:
                    log(f"ILS place-retry: {ENDHIGH_PD} density-gated; {_why}")
                    if _skip_unaff and _forced_queue:
                        log(f"ILS place-retry: skip-unaffordable ARMED — "
                            f"advancing to rung {_rungs_popped + 1} "
                            f"{_COMBOS[_forced_queue[0]][0]} in the SAME cycle "
                            f"rather than surrendering it to rotation")
                        continue
                    break
            _est = combo_cost.get(_fp)
            _fp_basis = cold_start_basis(cfg, _meas_basis)
            if _est is None and _fp_basis > 0:
                _est = (combo_cost_prior(_COMBOS[_fp][0])
                        * _fp_basis)
            # Every rung after the first must leave enough time for one full
            # follow-up cycle, preventing exploratory retries from consuming
            # the productive rotation's budget. The first popped rung is exempt.
            _res = (_fp_basis if (ladder_reserve_enabled() and _rung_no > 0
                                  and _fp_basis > 0) else 0.0)
            # Test the follow-up reserve against fenced remaining time, because
            # that cycle cannot use ladder-only borrowed time. Consequently,
            # when reserve gating applies, later rungs cannot benefit from the fence.
            if _res > 0 and _est is not None and _est + _res >= remaining:
                log(f"ILS place-retry: rung {_rung_no + 1} "
                    f"{_COMBOS[_fp][0]} reserve-gated (est {_est:.0f}s + "
                    f"reserve {_res:.0f}s >= remaining {remaining:.0f}s); "
                    f"ladder stops here")
                # Reserve-gating stops the LADDER, not just this rung: the window
                # cannot pay for a rung plus a follow-up cycle, so no later rung
                # can qualify either. Unchanged by skip-unaffordable.
                _forced_queue.clear()
                break
            elif _est is None or _est + _res < _wf_rem:
                pick, _use_forced = _fp, True
                # Fence-admitted: this rung fits the wrapper window but NOT the
                # fenced one. Recorded on the result and logged with a stable
                # key so a firing audit can grep the treatment, not a proxy.
                if (_wf_extra > 0 and _est is not None
                        and _est + _res >= remaining):
                    _wf_admit = True
                    log(f"ILS wall-fence: ADMITTED rung {_rung_no + 1} "
                        f"{_COMBOS[_fp][0]} using the wrapper window "
                        f"(est {_est:.0f}s vs fenced {remaining:.0f}s, "
                        f"fence {_wf_rem:.0f}s) — spending the polish "
                        f"reserve on a claimed ladder cycle")
                    r.notes.append(
                        f"wall-fence: admitted {_COMBOS[_fp][0]} "
                        f"(est {_est:.0f}s > fenced {remaining:.0f}s)")
                break
            else:
                log(f"ILS place-retry: {_COMBOS[_fp][0]} unaffordable "
                    f"(est {_est:.0f}s > {_wf_rem:.0f}s remaining); "
                    f"continuing normal rotation")
                if _skip_unaff and _forced_queue:
                    log(f"ILS place-retry: skip-unaffordable ARMED — advancing to "
                        f"rung {_rungs_popped + 1} "
                        f"{_COMBOS[_forced_queue[0]][0]} in the SAME cycle rather "
                        f"than surrendering it to rotation")
                    continue
                break
        for k in range(n if pick is None else 0):
            idx = (rot + k) % n
            if idx in tried_since_accept:
                continue
            # Last-mile placement can fail outright when timing is far from
            # closure. Vendor guidance uses WNS >= -0.25 ns; the configured
            # -0.30 ns gate allows downstream physical optimization and routing
            # to close a small gap. Gated entries count as tried; any accept
            # clears the set and re-enables them.
            if (_COMBOS[idx][0] == LASTMILE_PD and best is not None
                    and best < cfg.lastmile_min_wns_ns):
                tried_since_accept.add(idx)
                continue
            # Apply the high-net-delay density gate in the base rotation as well
            # as the forced-retry ladder. Gated entries count as tried so
            # rotation exhaustion remains correct until an accept clears the set.
            if _COMBOS[idx][0] == ENDHIGH_PD:
                # Price the cycle the same way the forced-rung site does, so the
                # two paths can never disagree about what this combo costs.
                _eh_est = combo_cost.get(idx)
                _eh_basis = cold_start_basis(cfg, _meas_basis)
                if _eh_est is None and _eh_basis > 0:
                    _eh_est = combo_cost_prior(ENDHIGH_PD) * _eh_basis
                _blk, _why = endhigh_density_blocked(cfg, _eh_est, remaining)
                if _blk:
                    if ENDHIGH_PD not in _endhigh_gate_logged:
                        _endhigh_gate_logged.add(ENDHIGH_PD)
                        log(f"[ils] {_why}")
                    tried_since_accept.add(idx)
                    continue
            # Skip partial-ruin placement when critical-path cells are co-located;
            # average spread is measured in placement tiles. Missing spread
            # fails closed for this combo family only. Skips count as tried
            # until an accept clears the rotation state.
            if (_COMBOS[idx][0] == PARTIAL_RUIN_PD
                    and cfg.partial_ruin_spread_gate
                    and (cfg.critical_path_avg_spread_tiles is None
                         or (cfg.critical_path_avg_spread_tiles
                             < cfg.partial_ruin_spread_min_tiles))):
                tried_since_accept.add(idx)
                _sp = cfg.critical_path_avg_spread_tiles
                _sp_s = f"{_sp:.1f}" if _sp is not None else "UNMEASURED"
                if not _spread_gate_noted:
                    _spread_gate_noted = True
                    r.notes.append(
                        f"partial-ruin spread-gated: "
                        f"spread={_sp_s} < "
                        f"{cfg.partial_ruin_spread_min_tiles:.0f} tiles")
                log(f"[ils] partial-ruin skipped: "
                    f"spread={_sp_s} < "
                    f"{cfg.partial_ruin_spread_min_tiles:.0f} "
                    f"(evidence corpus 26/26 negative; None fails closed)")
                continue
            # Route rerolls are eligible only near met timing and when the remaining
            # window covers a full route using the configured cost margin.
            # This gate affects only the reroll family and composes with other gates.
            if _COMBOS[idx][0] == ROUTE_REROLL_PD:
                if not cfg.route_reroll_enabled:
                    # Kill switch: account it as tried so rotation
                    # exhaustion still terminates, exactly like a gate skip.
                    tried_since_accept.add(idx)
                    continue
                if best is None or best < -cfg.route_reroll_max_wns_mag:
                    # Deep-negative or unmeasured WNS is ineligible for this near-met
                    # reroll; deeper failures are handled by the tail reroute loop.
                    # Mark the entry tried so a later accept can clear the set and
                    # re-enable it after timing improves.
                    tried_since_accept.add(idx)
                    log(f"[ils] route-reroll skipped: wns={best} below "
                        f"near-met floor -{cfg.route_reroll_max_wns_mag} "
                        f"(probe evidence is near-met only, fir "
                        f"-0.195 +0.070ns; deep-WNS owned by bare-reroute "
                        f"tail)")
                    continue
                _rr_need = max(route_reroll_cost_basis(cfg),
                               combo_cost.get(idx, 0.0))
                if _rr_need > 0 and _rr_need >= remaining:
                    # (b) budget-unfit: NOT tried-marked (affordability is
                    # not a deterministic-replay fact), the generic picker
                    # just moves on; note/log once per run.
                    if not _rr_budget_noted:
                        _rr_budget_noted = True
                        r.notes.append(
                            f"route-reroll budget-gated: need "
                            f"{_rr_need:.0f}s (route anchor x"
                            f"{ROUTE_REROLL_COST_MARGIN}) >= remaining "
                            f"{remaining:.0f}s")
                        log(f"[ils] route-reroll skipped: full-route need "
                            f"{_rr_need:.0f}s >= remaining {remaining:.0f}s "
                            f"(destructive op costed at route anchor x"
                            f"{ROUTE_REROLL_COST_MARGIN}; banked mirror = "
                            f"never-worse, no K=2 doubling)")
                    continue
            # Incremental re-route refines an existing routing with a stronger
            # directive. It requires a known incumbent router, and ladder
            # ordering must keep the preceding directive weaker than this rung.
            if _COMBOS[idx][0] == INCR_ROUTE_PD:
                if _last_rd is None or _last_rd == _COMBOS[idx][1]:
                    tried_since_accept.add(idx)
                    log(f"[ils] incr-route skipped: incumbent router "
                        f"{_last_rd or 'unknown'} is not weaker than "
                        f"{_COMBOS[idx][1]} (S6: escalation order is causal)")
                    continue
                # Run this terminal probe only when the futility stop is imminent
                # or the remaining window cannot fund another full-place cycle.
                # Budget and streak misses do not mark it tried because the gate
                # can become eligible later.
                if incr_route_terminal_enabled():
                    _basis_now = cold_start_basis(cfg, _meas_basis)
                    _about_to_stop = (no_improve_stop > 0
                                      and no_improve_streak >= no_improve_stop - 1)
                    _cant_afford_full = (_basis_now > 0 and _basis_now >= remaining)
                    if not (_about_to_stop or _cant_afford_full):
                        log(f"[ils] incr-route deferred: loop still productive "
                            f"(futility streak {no_improve_streak}/"
                            f"{no_improve_stop}) and a full cycle still fits "
                            f"({_basis_now:.0f}s <= {remaining:.0f}s) — the "
                            f"escalation is the LAST cheap move, not a competitor "
                            f"to full ruin")
                        continue
            est = combo_cost.get(idx)
            if est is None:
                # Before cycle costs are observed, scale relative combo priors by
                # the caller's place-route-phys_opt wall-time anchor.
                # This prevents an unaffordable full cycle from consuming the
                # window while allowing cheaper partial work to remain eligible.
                if (_COMBOS[idx][0] == ROUTE_REROLL_PD
                        and route_reroll_cost_basis(cfg) > 0):
                    # Route-reroll: cost it from THIS design's observed
                    # route sample x1.3 (the eligibility gate above used
                    # the same basis, so pass/fail here is consistent).
                    est = route_reroll_cost_basis(cfg)
                elif (_COMBOS[idx][0] == ROUTE_ONLY_PD
                        and cfg.fanout_cost_anchor_s > 0
                        and cfg.fanout_anchor_has_route):
                    # Route-only cycles reuse placement, so estimate them from the
                    # design's observed route-plus-phys_opt anchor rather than a
                    # place-dominated full-cycle prior.
                    est = cfg.fanout_cost_anchor_s
                elif cold_start_basis(cfg, _meas_basis) > 0:
                    # Measured-on-THIS-design basis when armed and available,
                    # else the caller's derived anchor exactly as shipped.
                    est = (combo_cost_prior(_COMBOS[idx][0])
                           * cold_start_basis(cfg, _meas_basis))
                else:
                    est = min(combo_cost.values()) if combo_cost else 0.0
            if est < remaining:
                pick = idx
                break
        if pick is None:
            if len(tried_since_accept) >= n:
                r.notes.append("rotation exhausted on current best; stopped")
                log("ILS: all combos tried since last accept (deterministic "
                    "placer) — stopping")
            else:
                log(f"ILS: stopping — no affordable combo for the remaining "
                    f"{remaining:.0f}s")
            break
        # A forced place-retry pick is deliberately OUT of rotation order, so it
        # must not advance rot -- and therefore must not move pristine_rot, which
        # the corrective sibling seed uses to skip genuine replays.
        if not _use_forced:
            rot = pick + 1
        if r.accepted == 0 and not _use_forced:
            # still on the pristine start state: this combo is a genuine
            # replay for a near-identical sibling seed (see pristine_rot).
            r.pristine_rot = rot
        pd, rd, od = _COMBOS[pick]
        # Forced probes are exempt from futility accounting in every mode.
        # A directed probe must not consume the final strike and prevent the
        # normal rotation from reaching another eligible combo.
        _futility_exempt = _use_forced
        # Vivado may reject subsequent full placement after a LastMile placement,
        # even when a fresh checkpoint is opened in the same session.
        # Restart before the next cycle. Restart is best-effort; the Tcl error
        # guard fails closed by skipping work in a still-poisoned session.
        if _prev_lastmile:
            _prev_lastmile = False
            try:
                _rr = await call_tool("vivado_restart_vivado", {})
                if _tool_ok(_rr):
                    log("ILS: Vivado restarted after LASTMILE cycle "
                        "(full-place session poisoning, reproduced)")
                else:
                    log(f"ILS: post-LASTMILE restart returned error "
                        f"({str(_rr)[:100]}); poisoned full-place cycles "
                        f"will be skipped by the error guard.")
            except Exception as e:
                log(f"ILS: post-LASTMILE restart failed ({e!r}); continuing.")
        if pd == LASTMILE_PD:
            _prev_lastmile = True
        cyc += 1
        r.cycles = cyc
        _t0 = time.time()
        # Command timeouts are capped by the cycle deadline.
        # Ordinary cycles retain the 300 s minimum used for large design steps.
        # Wall-fence admission extends both the cycle budget and command window;
        # otherwise the admitted rung would be killed by the original window.
        # For admitted cycles, clamp each timeout to end at least 30 s before the
        # wrapper deadline so the managed kill precedes outer cancellation.
        # This may abort a hopelessly late step quickly but avoids compromising
        # the tool session through wrapper-level cancellation.
        _cyc_dl = deadline_ts + (_wf_extra if _wf_admit else 0.0)
        def _heavy_to() -> float:
            if _wf_admit:
                return max(1.0, min(cfg.heavy_cmd_timeout_s,
                                    _cyc_dl - 30.0 - time.time()))
            return max(300.0, min(cfg.heavy_cmd_timeout_s, deadline_ts - time.time()))
        def _light_to() -> float:
            if _wf_admit:
                return max(1.0, min(cfg.measure_cmd_timeout_s,
                                    _cyc_dl - 30.0 - time.time()))
            return max(120.0, min(cfg.measure_cmd_timeout_s, deadline_ts - time.time()))
        async def _step(cmd: str, to: float) -> None:
            # Abort the cycle on an error envelope (timeout/failure) instead of
            # running the NEXT step on whatever state the failed one left behind.
            resp = await call_tool("vivado_run_tcl", {"command": cmd, "timeout": to})
            if not _tool_ok(resp):
                # Keep the TCL ERROR line itself if present — the first 120
                # chars are usually license/INFO preamble (measured: the real
                # error text was once truncated away while debugging LASTMILE).
                _txt = str(resp)
                _eline = next((l for l in _txt.splitlines() if "TCL ERROR" in l), "")
                raise RuntimeError(
                    f"{cmd.split()[0]} failed: {(_eline or _txt)[:400]}")
        try:
            await _step(f"open_checkpoint {{{work_path}}}", _light_to())
            if pd == LASTMILE_PD:
                # Last-mile polish operates on the current routed state; preserve
                # the netlist optimization, incremental placement, pre-route
                # optimization, and routing order. Standard post-route phys_opt
                # runs below and must not be duplicated here.
                await _step("phys_opt_design -clock_opt -retime -lut_opt", _heavy_to())
                await _step("place_design -directive LastMile", _heavy_to())
                await _step("phys_opt_design -directive Explore", _heavy_to())
                await _step(f"route_design -directive {rd}", _heavy_to())
            elif pd == ROUTE_REROLL_PD:
                # Route re-roll discards the current routing and solves from scratch.
                # Keep this as exactly unroute plus route: its cost model excludes
                # the standard phys_opt tail. The on-disk best checkpoint remains
                # unchanged unless the new routing passes the accept gate.
                await _step("route_design -unroute", _light_to())
                await _step(f"route_design -directive {rd}", _heavy_to())
            elif pd == ROUTE_ONLY_PD:
                # Route-only: keep the placement, re-route with
                # a stronger directive. No place step -> cheapest cycle; the
                # post-route phys_opt comes from the standard block below (od).
                await _step("route_design -unroute", _light_to())
                await _step(f"route_design -directive {rd}", _heavy_to())
            elif pd == INCR_ROUTE_PD:
                # Incremental escalated re-route refines the incumbent routing.
                # Do not unroute between the incumbent and this step; doing so
                # changes the operator into a from-scratch route-only cycle.
                # Standard post-route phys_opt supplies the polish tail below.
                await _step(f"route_design -directive {rd}", _heavy_to())
            elif pd == PARTIAL_RUIN_PD:
                # Targeted ruin unplaces only fabric cells on the worst paths, then
                # incrementally places them around the preserved majority.
                # Resolve FPL26_PARTIAL_RUIN_SCOPE at execution time; the default
                # target set is 200 cells to limit placement disruption.
                await _step(partial_ruin_tcl(partial_ruin_scope()),
                            _light_to())
                await _step("place_design", _heavy_to())
                await _step(f"route_design -directive {rd}", _heavy_to())
            else:
                await _step("place_design -unplace", _heavy_to())
                await _step(f"place_design -directive {pd}", _heavy_to())
                await _step(f"route_design -directive {rd}", _heavy_to())
            # Measure BEFORE phys_opt. Unrouted nets can never be accepted
            # (ur==0 required) and a WNS gap beyond physopt_skip_margin_ns can't
            # be closed by phys_opt (typ. +0.05-0.15ns) — skip the 200-790s step.
            w, ur = await _measure(call_tool, wns_tcl, timeout_s=_light_to())
            _skip = None
            if ur != 0:
                _skip = f"unrouted={ur} (can never be accepted)"
            elif (cfg.physopt_skip_margin_ns > 0 and w is not None
                    and best is not None
                    and w < best - cfg.physopt_skip_margin_ns):
                _skip = (f"post-route wns={w} trails best {best} by > "
                         f"{cfg.physopt_skip_margin_ns}ns")
            if _skip is None and pd != ROUTE_REROLL_PD:
                # ROUTE_REROLL never runs the phys_opt tail: its protocol
                # (and its route-x1.3 budget basis) is exactly the two route
                # ops of the plateau probe.
                await _step(f"phys_opt_design -directive {od}", _heavy_to())
                w, ur = await _measure(call_tool, wns_tcl, timeout_s=_light_to())
            elif _skip is not None:
                r.physopt_skipped += 1
                log(f"ILS cycle {cyc}: phys_opt SKIPPED — {_skip}")
            cycle_dt = time.time() - _t0
            tried_since_accept.add(pick)
            combo_cost[pick] = max(combo_cost.get(pick, 0.0), cycle_dt * 1.15)
            # A completed full-place cycle updates the cost basis for other
            # full-place combos. Divide by the directive prior to express the
            # basis in Explore=1.0 units, and retain the maximum as a
            # conservative estimate across cycles.
            if pd not in PLACE_SENTINELS:
                _pr = combo_cost_prior(pd)
                if _pr > 0 and cycle_dt > 0:
                    _meas_basis = max(_meas_basis, cycle_dt / _pr)
                    if not _meas_basis_noted and measured_basis_enabled():
                        _meas_basis_noted = True
                        log(f"[ils] measured cost basis ARMED: {_meas_basis:.0f}s "
                            f"per Explore-equivalent cycle (from {pd} "
                            f"dt={cycle_dt:.0f}s / prior {_pr}); cold-start "
                            f"anchor was {cfg.expected_heavy_cycle_s:.0f}s")
            log(f"ILS cycle {cyc} place={pd}: wns={w} unrouted={ur} "
                f"dt={cycle_dt:.0f}s (best {best})")
            # Missing WNS or a cycle shorter than the configured minimum indicates
            # that budget gating prevented substantive tool work.
            # Stop after two consecutive no-ops to avoid spinning to the deadline
            # while tolerating one transient measurement failure.
            if w is None or cycle_dt < cfg.min_cycle_seconds:
                noop_streak += 1
                if noop_streak >= 2:
                    r.notes.append("commands not executing (budget-gated); stopped")
                    log("ILS: cycles not executing (budget-gated/skip) — stopping")
                    break
                continue
            noop_streak = 0
            # A worse full-placement result can schedule the opt-in retry family
            # immediately, before the futility limit makes its ladder rung
            # unreachable.
            # The retry fires at most once per seed and excludes sentinels because
            # they reuse placement rather than testing a placement family.
            # When enabled, the baseline gate additionally requires the result to
            # trail the pristine input. An unknown baseline fails open.
            _base_ok = True
            if _retry_baseline_gate_active(cfg) and cfg.design_baseline_wns is not None:
                _base_ok = (w is not None and w <= cfg.design_baseline_wns)
            if ((place_retry_enabled() or place_retry_ladder_enabled())
                    and not _retry_jumped
                    and pd not in PLACE_SENTINELS
                    and best is not None and w is not None and w <= best
                    and not _base_ok):
                log(f"ILS place-retry: cycle {cyc} place={pd} regressed vs incumbent "
                    f"({w} <= {best}) but BEATS the pristine baseline "
                    f"{cfg.design_baseline_wns} — baseline gate holds the trigger "
                    f"(the recipe is good here; the placement family is not wrong)")
            if ((place_retry_enabled() or place_retry_ladder_enabled())
                    and not _retry_jumped
                    and pd not in PLACE_SENTINELS
                    and best is not None and w is not None and w <= best
                    and _base_ok):
                # RUNGS. Single-rung = [PLACE_RETRY_TARGET]. The ladder
                # keeps that as rung 1 and appends the rest, so arming it can only
                # ADD probes after the behaviour that already ships.
                _rungs = (ladder_rungs(baseline_wns,
                                       cfg.route_reroll_max_wns_mag)
                          if place_retry_ladder_enabled()
                          else [PLACE_RETRY_TARGET])
                _names = []
                for _want in _rungs:
                    for _j, _c in enumerate(_COMBOS):
                        if (_c[0] == _want and _j not in tried_since_accept
                                and _j not in _forced_queue):
                            _forced_queue.append(_j)
                            _names.append(f"{_c[0]} (idx {_j})")
                            break
                if _forced_queue:
                    _retry_jumped = True
                    log(f"ILS place-retry: cycle {cyc} place={pd} regressed "
                        f"(wns={w} <= best {best}); next cycle forced to "
                        f"{_names[0]}"
                        + (f"; ladder queued {' -> '.join(_names[1:])}"
                           if len(_names) > 1 else ""))
            _accept = (ur == 0 and (best is None or w > best + cfg.accept_margin_ns))
            # Accepted results become the seed for later combos, so a marginal gain
            # can alter subsequent search even when its immediate value is small.
            # This optional guard preserves the current seed by rejecting gains
            # below meaningful_accept_ns. It can discard a real improvement and
            # therefore remains off unless FPL26_ILS_REJECT_MICRO_ACCEPT is set.
            if (_accept and reject_micro_accept_enabled()
                    and best is not None and cfg.meaningful_accept_ns > 0):
                _micro_gain = w - best
                if _micro_gain < cfg.meaningful_accept_ns:
                    _accept = False
                    r.notes.append(
                        f"cycle {cyc}: micro-accept gain {_micro_gain:.4f} < "
                        f"meaningful {cfg.meaningful_accept_ns} — REJECTED to "
                        f"keep the seed clean (measured evidence)")
                    log(f"ILS cycle {cyc}: micro-accept REJECTED (gain "
                        f"{_micro_gain:.4f} < {cfg.meaningful_accept_ns}) — seed "
                        f"kept clean for later combos")
            if _accept and cfg.accept_requires_hold_clean:
                _whs = await _measure_hold(call_tool, timeout_s=_light_to())
                if _whs is not None and _whs < cfg.hold_slack_floor_ns:
                    _accept = False
                    r.notes.append(f"cycle {cyc}: hold-dirty (whs={_whs}); "
                                   f"setup improvement wns={w} REJECTED")
                    log(f"ILS cycle {cyc}: HOLD-DIRTY (whs={_whs}) — accept "
                        f"rejected (validator gates hold_passed)")
            # LastMile may change the netlist through LUT optimization, so accepted
            # results receive an informational cell-count sanity check.
            # A missing golden count or unparsable measurement fails open.
            if _accept and pd == LASTMILE_PD and cfg.golden_cell_count:
                try:
                    _cc = await call_tool("vivado_run_tcl", {
                        "command": "llength [get_cells -quiet -hierarchical "
                                   "-filter {IS_PRIMITIVE}]",
                        "timeout": _light_to()})
                    _m = re.search(r"(\d+)", _cc or "")
                    _n = int(_m.group(1)) if _m else None
                except Exception:
                    _n = None
                if _n is not None:
                    _lo = cfg.golden_cell_count * cfg.cell_floor_ratio
                    _hi = cfg.golden_cell_count * cfg.cell_ceil_ratio
                    if not (_lo <= _n <= _hi):
                        _accept = False
                        r.notes.append(
                            f"cycle {cyc}: cell-count {_n} outside sanity "
                            f"band [{_lo:.0f},{_hi:.0f}] (golden "
                            f"{cfg.golden_cell_count}); accept REJECTED")
                        log(f"ILS cycle {cyc}: CELL-COUNT {_n} outside "
                            f"[{_lo:.0f},{_hi:.0f}] — accept rejected "
                            f"(netlist likely mangled; sanity band)")
            if _accept:
                # Retry checkpoint persistence once to tolerate transient filesystem
                # failures. If both writes fail, discard the candidate so the
                # on-disk best checkpoint remains the never-worse state.
                _wr = await call_tool("vivado_run_tcl",
                                      {"command": f"write_checkpoint -force {{{best_dcp_path}}}",
                                       "timeout": _heavy_to()})
                if not _tool_ok(_wr):
                    log(f"ILS cycle {cyc}: write_checkpoint failed once "
                        f"({str(_wr)[:80]}); retrying before discard")
                    _wr = await call_tool("vivado_run_tcl",
                                          {"command": f"write_checkpoint -force {{{best_dcp_path}}}",
                                           "timeout": _heavy_to()})
                if not _tool_ok(_wr):
                    # The improved design was NOT persisted — the file on disk is
                    # still the previous best (or partial). Accepting would record
                    # a WNS the shipped DCP doesn't have. Discard the improvement.
                    _accept = False
                    r.notes.append(f"cycle {cyc}: write_checkpoint failed (x2); "
                                   f"improvement wns={w} discarded")
                    log(f"ILS cycle {cyc}: write_checkpoint error/timeout (x2) — "
                        f"improvement wns={w} DISCARDED (file unchanged)")
            if _accept:
                _gain = (w - best) if best is not None else None
                best = w
                r.best_wns = best
                r.accepted += 1
                r.improved = True
                tried_since_accept.clear()  # new best DCP: all combos fresh again
                work_path = best_dcp_path   # current state IS the new best
                # The incumbent is now THIS cycle's output, so its router is this
                # cycle's rd — the state fact the incr-route eligibility reads.
                _last_rd = rd
                if _use_forced and _forced_queue and ladder_stop_on_accept_enabled():
                    _dropped = [_COMBOS[j][0] for j in _forced_queue]
                    _forced_queue.clear()
                    log(f"ILS place-retry: rung {_rung_no + 1} ACCEPTED — the "
                        f"placement family question is answered; dropping the "
                        f"remaining rungs {' -> '.join(_dropped)} so the window "
                        f"goes to the rotation instead of to directives that "
                        f"must now beat this result")
                log(f"ILS *** ACCEPT new best wns={best} ***")
                if (_gain is not None and cfg.meaningful_accept_ns > 0
                        and _gain < cfg.meaningful_accept_ns
                        and not _futility_exempt):
                    # Micro-accept: kept (never-worse), but it doesn't pay
                    # for continuation (see meaningful_accept_ns evidence).
                    no_improve_streak += 1
                    r.notes.append(f"cycle {cyc}: accept gain {_gain:.4f} < "
                                   f"meaningful {cfg.meaningful_accept_ns} — "
                                   f"kept, futility streak {no_improve_streak}")
                    log(f"ILS cycle {cyc}: micro-accept (gain {_gain:.4f} < "
                        f"{cfg.meaningful_accept_ns}) — kept, futility "
                        f"streak {no_improve_streak}")
                    if no_improve_stop and no_improve_streak >= no_improve_stop:
                        r.notes.append(f"no meaningful improvement in "
                                       f"{no_improve_streak} cycles; stopped")
                        log(f"ILS: no meaningful improvement in "
                            f"{no_improve_streak} real cycles — stopping "
                            f"(yield budget)")
                        break
                else:
                    no_improve_streak = 0
                    # A meaningful accept removes the redraw reserve for the rest of
                    # the invocation, allowing later cycles to use the full wall.
                    _had_meaningful_accept = True
            else:
                # SA exploration: a routed near-miss (within the regression
                # floor of best) becomes the next cycle's starting state —
                # the best DCP stays the untouched ship floor.
                if (cfg.explore_from_current and ur == 0 and w is not None
                        and best is not None
                        and w >= best - cfg.explore_regression_floor_ns):
                    _ew = await call_tool("vivado_run_tcl",
                                          {"command": f"write_checkpoint -force {{{explore_path}}}",
                                           "timeout": _heavy_to()})
                    if _tool_ok(_ew):
                        work_path = explore_path
                        tried_since_accept.clear()  # new state: combos fresh
                        log(f"ILS explore: continuing from near-miss wns={w} "
                            f"(best {best} preserved)")
                    else:
                        work_path = best_dcp_path
                else:
                    work_path = best_dcp_path
                if _futility_exempt:
                    # Each ladder rung is a directed probe of a placement
                    # family, not a restart. Keep-best makes regressions
                    # harmless, so forced probes do not advance futility. At
                    # most len(PLACE_RETRY_LADDER) cycles are exempt for each
                    # seed.
                    log(f"ILS cycle {cyc}: ladder rung (forced probe) — futility "
                        f"streak held at {no_improve_streak}")
                else:
                    no_improve_streak += 1
                if no_improve_stop and no_improve_streak >= no_improve_stop:
                    # The optional hurdle override uses the scoring model to
                    # price one more cycle. The hurdle in MHz is alpha * 0.1 *
                    # (cycle_s / 3600) / P. Each decision authorizes only one
                    # additional cycle; deadline, budget, cycle-cap, and
                    # maximum-hurdle limits still apply.
                    _hurdle_alpha = float(getattr(cfg, "hurdle_continue_alpha_mhz", 0.0) or 0.0)
                    _hurdle_cap = float(getattr(cfg, "hurdle_continue_max_mhz", 1.0) or 1.0)
                    _cycle_s = float(cycle_dt or 0.0)
                    _remaining = (deadline_ts - time.time()) if deadline_ts else 0.0
                    _override = False
                    if _hurdle_alpha > 0 and _cycle_s > 0 and _remaining > _cycle_s * 2:
                        _hurdle = (_hurdle_alpha * 0.1 * (_cycle_s / 3600.0)) / 0.94
                        if _hurdle < _hurdle_cap:
                            _override = True
                            log(f"ILS: futility K={no_improve_stop} reached but "
                                f"one more cycle costs {_cycle_s:.0f}s => hurdle "
                                f"{_hurdle:.3f} MHz (< {_hurdle_cap} "
                                f"cap) with {_remaining:.0f}s left; CONTINUING "
                                f"(scoring function, not the counter, decides)")
                            no_improve_streak = 0
                    if not _override:
                        r.notes.append(f"no improvement in {no_improve_streak} cycles; stopped")
                        log(f"ILS: no improvement in {no_improve_streak} real cycles "
                            f"— stopping (yield budget)")
                        break
        except Exception as e:  # never let ILS crash the agent's finalize
            tried_since_accept.add(pick)  # don't deterministically re-fail it
            work_path = best_dcp_path     # abandon any exploration state
            r.notes.append(f"cycle {cyc} error: {e!r}")
            log(f"ILS cycle {cyc} error: {e!r}")
            try:
                _rec = await call_tool("vivado_run_tcl",
                                       {"command": f"open_checkpoint {{{best_dcp_path}}}",
                                        "timeout": max(120.0, min(cfg.measure_cmd_timeout_s,
                                                                  deadline_ts - time.time()))})
                if not _tool_ok(_rec):
                    log("ILS: recovery open_checkpoint failed — ending this seed")
                    break
            except Exception:
                break
    if _seg_armed_ever and not _seg_logged_ever:
        log("SHALLOW-GUARD: armed, no eligible site (no S6-eligible "
            "incr-route combo materialized this run)")
    if _prev_lastmile:
        # Don't hand a LASTMILE-poisoned session to whatever runs next (the
        # corrective seed, finalize, winner-polish): full placement is broken
        # in this session until restart (reproduced).
        try:
            await call_tool("vivado_restart_vivado", {})
            log("ILS: Vivado restarted on exit (last cycle was LASTMILE).")
        except Exception as e:
            log(f"ILS: exit restart failed ({e!r}); downstream guards apply.")
    return r
