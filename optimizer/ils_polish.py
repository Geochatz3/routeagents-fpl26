"""In-agent ILS-as-polish (Iterated Local Search / ruin-and-recreate).

Validated standalone on AWS (v2, vtr_mcml functional-equivalence PASSED) and
swept locally across all 13 benchmarks (see ils_runs_20260605/CLUSTERING_RESULTS.md).

DESIGN (professional integration)
=================================
ILS recovers a robust fmax^2 * dWNS on ANY size-viable design by fully
unplacing and re-placing with a different directive each cycle, keeping the
result only if it is strictly better (NEVER-WORSE). It WINS where the agent's
main recipe underperformed (production left headroom); it ties/loses where the
recipe already did well.

WHY STAGNATION-TRIGGERED (not feature-gated, not a fixed time-slice):
- WIN vs LOSE is NOT separable from raw DCP features (cell ranges overlap) — an
  ML/cluster predictor has no signal here (measured AUC 0.52). So we do NOT try
  to predict.
- A fixed ILS reserve would steal budget from designs where the main loop is
  still productive (e.g. small LOSE designs amd/logicnets) and could regress
  them. Instead we trigger ONLY when the main loop has STALLED — that runtime
  signal cleanly separates "loop productive, leave it" from "loop stuck, rescue
  it with a global re-place."
- keep-best makes a mis-trigger harmless: ILS seeded from the agent's best can
  only replace it with something strictly better.

This module is PURE Vivado Tcl via the agent's call_tool — no LLM. It is
exception-safe at the call site (any failure -> agent keeps its existing best).
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
# Sentinel place-directive: this combo does TARGETED ruin (worst-paths' cells
# only) instead of full unplace. Movable-primitive whitelist mirrors the
# validated partial_ruin_probe.tcl.
PARTIAL_RUIN_PD = "__PARTIAL_RUIN__"
# Sentinel for the UG906 last-mile cycle (phys_opt clock_opt/retime/lut ->
# place_design -directive LastMile -> phys_opt Explore -> route). See
# ILS_COMBOS entry for the jun12 probe evidence.
LASTMILE_PD = "__LASTMILE__"
# Sentinel for a ROUTE-ONLY cycle (jun15 DRILL H): keep the placement, unroute,
# re-route with the combo's route directive. The rotation's other combos ALWAYS
# re-place first, so pure re-routing was an unexercised operator class; the
# production recipe also routes with Default only. Breadth (12 optimized DCPs):
# route AggressiveExplore improved 8/12 — the broadest single lever measured —
# incl. ispd16 +2.53ns GT (~+40 MHz, whs 0.000 which the official scorecard
# gate accepts: jul02 preview evidence hold_passed=true at whs_ns=0.0) and
# hold-SAFE +0.005..0.048ns on vexriscv/3d/digit/logicnets/finn/corescore.
ROUTE_ONLY_PD = "__ROUTE_ONLY__"
# Sentinel for the ROUTE RE-ROLL cycle (jul21 plateau probe,
# final_round/plateau_probe_jul21.md + drill_local_queue_jul21.md): full
# `route_design -unroute` followed by a from-scratch `route_design -directive
# AggressiveExplore` solve — and NOTHING else (no phys_opt tail; the probe's
# protocol and cost basis are exactly these two ops). On our BEST fir state
# (near-met, -0.195, route-share-dominated) this gained +0.070 ns ~ +9.5 MHz
# with hold IMPROVING (+0.044); 429s local ~ ~170s eval. Mechanism: a
# from-scratch route solve from a good placement re-rolls the route lottery
# WITHOUT inheriting the incumbent solution's compromises; incremental rip-up
# (bare re-route) yields only crumbs near plateaus (+0.004 fir). Distinct from
# ROUTE_ONLY by protocol (no phys_opt) and by ELIGIBILITY: picker-gated to
# NEAR-MET states only (deep-WNS designs are owned by the bare-reroute tail
# loop in dcp_optimizer). Kill switch: cfg.route_reroll_enabled / CLI
# --no-route-reroll / env FPL26_NO_ROUTE_REROLL=1.
ROUTE_REROLL_PD = "__ROUTE_REROLL__"
# Sentinel for the INCREMENTAL ESCALATED RE-ROUTE (jul27 spam probe,
# final_round/INCREMENTAL_REROUTE_OPERATOR_jul27.md). Exactly ROUTE_ONLY MINUS
# its `-unroute`: re-route the INCUMBENT solution in place with a MORE
# AGGRESSIVE directive, so the solver refines the existing routing instead of
# re-solving from scratch. Measured on spam from the same placed seed:
#   S1 single route Explore                  -0.598  (+17.51)
#   S2 route Explore x2 (SAME directive)     -0.595  (+18.14)  <- crumbs
#   S3 route Explore then AggressiveExplore  -0.543  (+29.19)  <- the record
#   S6 AggressiveExplore then Explore        -0.595  (+18.14)  <- order is causal
# S3/S4/S5 converge on EXACTLY -0.543 by three structurally different routes;
# S6 was PREDICTED to fail and did. The escalation is the mechanism, not the
# repetition -- which is why the two existing sentinels (both unroute first)
# cannot express it and why the jun15/jul21 "bare re-route = crumbs" evidence
# does not apply: that evidence re-routed with the SAME directive, which is S2.
# NOT GENERAL: on 3d the escalation is worth 0.000 ns (S2 = S3 = -1.970) and a
# SINGLE AggressiveExplore route beats both (-1.949). So this is ONE design's
# operator until a second confirms it -- shipped default OFF, appended late, and
# keep-best means a non-reproducing design pays one cheap cycle, never MHz.
# Armed by FPL26_ILS_INCR_ROUTE=1.
INCR_ROUTE_PD = "__INCR_ROUTE__"
# Scope swept on demo_corundum 2026-06-12: n=50 +0.081 / n=200 +0.163 (peak,
# TIMING MET, beats full ruin's +0.106) / n=500 +0.112 — unimodal, 200 wins on
# gain AND gain-per-second (1163s vs full 1431s).
# jul22 (magic-constant pass, PLAN item 4): scope parameterized via
# partial_ruin_tcl(scope) + env FPL26_PARTIAL_RUIN_SCOPE so the queued farm
# scope study can sweep WITHOUT code edits.  DEFAULT UNCHANGED at 200 — the
# corundum sweep optimum stands until that study says otherwise.
# Invalid/<1 env keeps 200.
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
    # Reordered 2026-06-10 from the 9-run AWS cycle audit (accept counts/costs):
    # Explore 2 accepts ~125-760s; ExtraTimingOpt 3 accepts (v2/vtr/3d);
    # AltSpreadLogic_high 2 accepts incl. v2's BEST, ~375s — promoted above
    # ExtraNetDelay_high (1 accept, 447-1273s, the costliest combo);
    # SSI_SpreadLogic_high 0 accepts; EarlyBlockPlacement 0 accepts + ur=211.
    ("Explore", "Explore", "AlternateFlowWithRetiming"),
    # LAST MILE at index 1 (jun12): probe-validated 2/2 on PLATEAUED ship
    # states (v2 -0.799 -> -0.647 +0.152ns ~+29MHz; logicnets -0.496 ->
    # -0.437 +0.059ns ~+15MHz; both 10000-vector equivalence PASSED) — the
    # strongest accept evidence of any combo precisely where ruin combos
    # stop moving. MUST sit inside the futility window: with the K=2
    # final-seed stop, combos at index >=2 are unreachable on a run whose
    # first two cycles don't accept, and LastMile is a DIFFERENT operator
    # class (netlist-level phys_opt retime/lut/clock + incremental LastMile
    # re-place) than ruin — try both classes before giving up.
    (LASTMILE_PD, "Explore", "Explore"),
    # ROUTE RE-ROLL at index 2 (jul21 plateau probe, fir +0.070ns/+9.5MHz,
    # hold-improving): unroute + from-scratch AggressiveExplore route, two
    # ops only (od slot unused — the cycle deliberately skips the phys_opt
    # tail; its budget basis is a FULL route x1.3, see route_reroll_cost_
    # basis). PLACEMENT RATIONALE: the picker's near-met eligibility gate
    # (route_reroll_max_wns_mag) makes this slot class-conditional — on
    # NEAR-MET designs (fir/logicnets/vexriscv-class, exactly where the
    # probe evidence lives) it sits adjacent to LASTMILE inside the early
    # window, taking over the futility-edge slot ROUTE_ONLY held (same
    # unroute+re-route mechanism, stronger near-met evidence, cheaper
    # cycle). On DEEP-WNS designs the gate skips it INLINE within the same
    # pick (no cycle consumed, tried-marked), so their effective rotation
    # is byte-identical to the pre-jul21 order — the least invasive
    # placement that still reaches near-met designs early.
    (ROUTE_REROLL_PD, "AggressiveExplore", "Explore"),
    # ROUTE-ONLY (jun15 DRILL H, integrated jul02 at index 2; index 3 since
    # the jul21 ROUTE_REROLL insertion — near-met designs get the re-roll
    # first, deep-WNS designs still see this at effective position 2 because
    # the re-roll gate skips inline): re-route the
    # CURRENT placement with AggressiveExplore — cheapest cycle in the set
    # (no place step; ~0.5x Explore full-ruin) and the broadest breadth result
    # (8/12 improved). A third operator class (placement untouched) distinct
    # from ruin and LastMile; sits at the futility-window edge so stuck runs
    # try all three classes before the K=2 stop. Hold gate: the standard
    # accept floor (-0.001) matches the official scorecard gate, which passes
    # whs=0.0 (jul02 preview evidence).
    (ROUTE_ONLY_PD, "AggressiveExplore", "Explore"),
    ("ExtraTimingOpt", "AggressiveExplore", "AggressiveExplore"),
    # PARTIAL ruin PROMOTED above AltSpread/ExtraNetDelay (jul05 climbON_a
    # forensics): preview #5's -0.809 finisher was ExtraTimingOpt -> PARTIAL_
    # RUIN — but that adjacency only happened because the tight eval window
    # made the two expensive combos UNAFFORDABLE (cost gate skipped them).
    # With a roomier window (debug wall) they ran instead — both weak on the
    # stuck class (-1.008/-1.007, dt 579/665s) — and burned the K=2 futility
    # budget before PARTIAL_RUIN could fire. Promote the cheap (0.7x),
    # probe-validated finisher so the proven chain executes by construction
    # on ANY budget, not by accident of budget pressure.
    # (probe-validated live on demo_corundum 2026-06-11: unplace only the
    # worst-paths' fabric cells -> incremental re-place; +0.081ns at ~66% of
    # a full-ruin cycle's cost, hold clean. jun12 scope-200 sweep optimum.)
    (PARTIAL_RUIN_PD, "Explore", "AggressiveExplore"),
    ("AltSpreadLogic_high", "AggressiveExplore", "AggressiveExplore"),
    ("ExtraNetDelay_high", "Explore", "AlternateFlowWithRetiming"),
    ("SSI_SpreadLogic_high", "Explore", "AlternateFlowWithRetiming"),
    ("EarlyBlockPlacement", "AggressiveExplore", "AggressiveExplore"),
]

# ---- PLACE-RETRY EXTENSION (jul27). DEFAULT OFF = byte-identical behaviour ----
#
# WHY THESE TWO. The rotation was ordered from a 9-run AWS audit by ACCEPT COUNTS,
# which cannot see a directive that was never in the set. Neither
# AltSpreadLogic_medium nor ExtraNetDelay_low has ever been in it -- only their
# `_high` siblings, ordered late because they are EXPENSIVE (2.0x / 3.0x a
# full-ruin cycle). That cost objection does NOT apply to these two.
#
# Measured jul27 at parity, every arm ROUTED, reproduced on TWO independent boxes:
#
#   design    directive              final    alpha    cycle cost
#   spam      AltSpreadLogic_medium -0.598  +17.51     273s (~1.0x Explore)
#   spam      ExtraNetDelay_low     -0.647   +7.59     315s
#   spam      Explore  (combo 0)    -0.688   -0.42     267s
#   vexriscv  AltSpreadLogic_medium -0.602 +150.24     194s
#   vexriscv  ExtraNetDelay_low     -0.699 +130.55     170s
#   vexriscv  Explore  (combo 0)    -0.785 +114.46     151s
#
# spam artifact md5 6f81289a: 28,108/28,108 routed, 0 errors, WHS +0.024.
#
# spam is the worked example: it currently ships +0.00 (VALID_FALLBACK_BASELINE
# whose artifact md5 equals the INPUT md5) because combo 0 REGRESSES it (-0.688
# vs a -0.686 baseline) and futility then stops with ~848s of ~36min unspent.
PLACE_RETRY_COMBOS: List[Tuple[str, str, str]] = [
    ("AltSpreadLogic_medium", "Explore", "AlternateFlowWithRetiming"),
    ("ExtraNetDelay_low", "Explore", "AlternateFlowWithRetiming"),
]

# WHICH directive the regression trigger FORCES. Measured on the only two designs known
# to satisfy the trigger (combo 0 REGRESSES them), every arm routed:
#
#   substitute              spam      3d      positive on both?
#   AltSpreadLogic_medium  +17.51  -25.95    NO  (3d far worse than its own baseline)
#   ExtraNetDelay_low       +7.59  -15.37    NO
#   ExtraNetDelay_high      +4.05  +11.96    YES
#
# ExtraNetDelay_high is the ONLY candidate positive on both, so it is the target even
# though AltSpreadLogic_medium is worth more on spam alone. Forcing a substitute that
# REGRESSES a triggering design is not merely a wasted cycle: it burns that design's
# second futility strike, ending the search before the rotation can reach anything
# better. On 3d, ExtraNetDelay_high instead turns the regression into an ACCEPT
# (-2.153 -> -1.997), which resets futility and lets the search continue.
#
# It is ALREADY in ILS_COMBOS (index 6), so forcing it adds no combo and does not change
# the rotation's length. The extension above stays appended so the higher-value
# AltSpreadLogic_medium remains reachable later on designs with budget to get there.
PLACE_RETRY_TARGET = "ExtraNetDelay_high"

# ---- PROBE LADDER (jul28). STRICTLY ADDITIVE to the single forced pick. ----
#
# The ladder's FIRST rung is PLACE_RETRY_TARGET itself, so an armed ladder does
# everything the single retry did, in the same order, and only then continues.
# Nothing that works today can be displaced by arming it.
#
# WHY MORE THAN ONE RUNG. 1af5c85 had to pick ONE substitute for every design and
# chose the only candidate positive on both designs it could measure. But the best
# directive is design-specific, and the runner-up costs real MHz:
#
#   design    ExtraNetDelay_high   AltSpreadLogic_medium   which wins
#   spam            +4.05                 +17.51           medium, by 13.46
#   3d             +11.96                 -25.95           high, by 37.91
#
# A ladder does not have to choose: keep-best rejects the loser at the cost of one
# cycle. The reason 1af5c85 could NOT do this was futility -- a rejected probe
# burned a strike and ended the search. So the ladder is only sound together with
# the futility exemption below, and only affordable with FPL26_ILS_MEASURED_BASIS
# (AltSpreadLogic_medium is cheap at 1.05x, but the ladder's own first rung is the
# 3.0x directive that the inflated cold-start anchor prices out).
#
# Ordered by EVIDENCE, not cost: the rung proven positive on two designs goes
# first, so a design that stops early stops on the safest rung.
PLACE_RETRY_LADDER = ["ExtraNetDelay_high", "AltSpreadLogic_medium", "ExtraNetDelay_low"]


# The rung a NEAR-MET design wants first. Kept as a name, not an index, because the
# ladder is a list of directives and the rotation's indices move.
LADDER_NEAR_MET_FIRST = "AltSpreadLogic_medium"


def ladder_order_by_wns_enabled() -> bool:
    """DEFAULT ON since aug01. Kill switch: FPL26_ILS_LADDER_ORDER_BY_WNS=0.

    Promoted on the ladder_ab_jul31 paired A/B over the SIX designs this flag can
    touch (the other ten have |ILS baseline WNS| >= 0.7 and are a provable no-op
    by the branch in ladder_rungs below, so running them buys nothing).

    The constant is NOT tuned and must not be: 0.7 is cfg.route_reroll_max_wns_mag,
    reused unchanged from the route-reroll gate. The jul28 refusal was aimed at
    fitting that constant to spam's record; that is still refused. What was
    promoted is the UNTUNED rule, measured on the designs it can affect.
    """
    return os.environ.get(
        "FPL26_ILS_LADDER_ORDER_BY_WNS", "1").strip().lower() in ("1", "true", "on", "yes")


def ladder_rungs(baseline_wns: Optional[float] = None,
                 near_met_mag: Optional[float] = None) -> List[str]:
    """Ladder order, overridable by FPL26_ILS_LADDER_ORDER (comma-separated).

    WNS-KEYED ORDERING (jul28, default OFF). spam's record needs
    AltSpreadLogic_medium as RUNG 1 — placed from the untouched seed. Twice now it
    only got there by luck: night16 because rung 1 hit a route_design TCL ERROR and
    left the seed clean, batch6 because LADDER_ORDER was set by hand. Neither is a
    mechanism, and a fixed medium-first default is not one either: that directive
    measures -2.544 on 3d against a -2.153 baseline.

    The separator is how far the design is from meeting, using the SAME near-met
    magnitude the route-reroll gate already uses (cfg.route_reroll_max_wns_mag, 0.7).
    Measured, each design's ILS baseline and what each rung returned from it:

      design   baseline  |WNS| vs 0.7   rung 1 chosen         result
      spam      -0.665    0.665 NEAR    AltSpreadLogic_medium -0.598 = the record's
                                                              base (escalates to -0.543)
      optical   -0.924    0.924 deep    ExtraNetDelay_high    -0.846 ACCEPT
                                                              (medium is -1.086)
      3d        -2.153    2.153 deep    ExtraNetDelay_high    -2.027 ACCEPT
                                                              (medium is -2.408, WORSE
                                                              than doing nothing)

    Reads physically: spreading logic is the fine adjustment a nearly-met design
    wants, while a deeply-violating one needs the heavier net-delay weighting first.

    HONEST LIMIT: 3/3 is n=3, and the cut sits only 0.035 ns from spam's own
    baseline (0.665) while clearing optical by 0.224. spam's baseline also moves run
    to run (-0.598/-0.665/-0.686), and -0.686 is 0.014 from flipping. So this is a
    HYPOTHESIS WITH A MECHANISM, not a fitted boundary — the corpus (44/420) cannot
    support fitting one. It stays default OFF until an A/B on spam AND 3d says
    otherwise. An explicit FPL26_ILS_LADDER_ORDER always wins over it.

    WHY AN ORDER KNOB. The rung that runs FIRST runs from the UNMODIFIED seed;
    every later rung runs from whatever state has accepted by then. That is not a
    detail — it decides which base placement the escalation gets. Measured on spam
    jul28, same design, same box, same config:

      ladder arm     cycles 1-2 did NOT accept -> rung 2 AltSpreadLogic_medium ran
                     from the untouched raw seed  -> -0.598  (the RECORD's base)
      noreserve arm  an earlier cycle accepted first -> the same rung ran from a
                     MODIFIED state -> worse; the escalation then had to work from
                     ExtraNetDelay_low (-0.631) and reached -0.574, not -0.543

    spam's record (S3) places AltSpreadLogic_medium FROM THE RAW BENCHMARK, so
    reproducing it needs that directive as rung 1. The default order is unchanged
    (evidence order: the rung positive on two designs goes first); this knob exists
    to TEST the ordering hypothesis without hard-coding a per-design table.
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
    """DEFAULT ON (jul29). Disable with FPL26_ILS_PLACE_RETRY_LADDER=0.

    Was opt-in, which meant the eval path never armed it — see the module note above.
    """
    return os.environ.get(
        "FPL26_ILS_PLACE_RETRY_LADDER", "1").strip().lower() in ("1", "true", "on", "yes")


# ---- BASELINE GATE (jul28). NARROWS the trigger; never widens it. ----------
#
# The shipped trigger is "this cycle came back worse than the INCUMBENT". Mined
# across the eight jul27 production logs, that fires on FIVE OF SIX designs — it is
# the common case, not a signature, because the incumbent is the recipe's polished
# output and a fresh full-ruin rarely beats it on the first try.
#
# The discriminating question is different: is this placement family WRONG for this
# design, or merely BEHIND an already-good incumbent? A full re-place that lands
# worse than the UNTOUCHED INPUT answers the first question. Measured, using each
# log's own "Initial Fmax: ... (WNS: ...)" line as the pristine baseline:
#
#   design         baseline   incumbent   cycle-1 Explore   worse than baseline?
#   spam            -0.686      -0.686        -0.688          YES  <- want
#   3d              -2.153      -2.153        -2.278          YES  <- want
#   optical         -1.078      -0.924        -1.162          YES  <- want
#   logicnets       -0.978      -0.526        -1.041          YES  <- false positive
#   amd_mini-isp    -1.686      -0.956        -0.996          no
#   corescore       -1.238      -0.680        -0.683          no
#   finn            -1.910      -1.256        -1.288          no
#   vexriscv        -1.654      -0.619        -0.635          no
#   vtr            -14.527     -14.316       -11.587          no
#
# So it fires on 3/3 of the designs whose records need it and stays inert on 5 of
# the 6 that do not — where the incumbent-relative trigger fires on nearly all of
# them. vexriscv is the clearest case of why: its Explore cycle is worth +114 MHz
# against baseline and is still "a regression" against its own polished incumbent.
#
# Unknown baseline -> gate does not apply (behaviour exactly as without it): this
# gate exists to NARROW a trigger, so a missing measurement must not be able to
# widen it, and must not silently disable a mechanism either.
def retry_baseline_gate_enabled() -> bool:
    """Armed only by FPL26_ILS_RETRY_BASELINE_GATE."""
    return os.environ.get(
        "FPL26_ILS_RETRY_BASELINE_GATE", "0").strip().lower() in ("1", "true", "on", "yes")


def _retry_baseline_gate_active(cfg) -> bool:
    """The retry baseline gate applies iff the GLOBAL env flag is armed OR
    the caller scoped it in via cfg.retry_baseline_gate_scoped
    (FPL26_MINIISP_RETRY_HOLD, aug05: dcp_optimizer sets the cfg field for
    MID-band designs only — 1.05 < |wns_in| < 8.0).  OR-composition keeps
    both validated behaviours byte-identical: global-flag runs are
    unchanged (the env read is untouched), and runs with neither armed see
    the default-False field.  Evidence for the scoped arm:
    MINIISP_MECHANISM_HUNT_aug05 — mini-ISP h2 (single flag) +106.10 vs h1
    (control) +102.71; the jul29 global-HOLD harm (spam -3.39) is shallow
    band, outside the scope by construction."""
    return retry_baseline_gate_enabled() or bool(
        getattr(cfg, "retry_baseline_gate_scoped", False))


# ---- PANEL REFINEMENTS (jul28 gating panel, final_round/panel_gating_jul28) ----
#
# Five independent seats reviewed the three mechanisms against the evidence pack.
# Unanimous where it counts: the ladder must be BUDGET-gated (5/5), the
# incremental re-route must be gated (5/5), and all five named the incremental
# re-route as the mechanism MOST LIKELY to be a false lever (it is +11.05 MHz on
# spam and 0.000 ns on 3d — one design for, one against).
#
# Both refinements are separately opt-in so the jul28 live A/B runs remain a valid
# description of the mechanisms they actually tested.
#
# LADDER RESERVE: rungs past the first fire only if the remaining window can pay
# for the rung AND still afford one more full cycle afterwards. Rung 1 is never
# reserve-gated — it is the shipped behaviour.
def ladder_reserve_enabled() -> bool:
    """Armed only by FPL26_ILS_LADDER_RESERVE."""
    return os.environ.get(
        "FPL26_ILS_LADDER_RESERVE", "0").strip().lower() in ("1", "true", "on", "yes")


# SKIP-UNAFFORDABLE. An unaffordable rung currently costs a WHOLE CYCLE, and hands
# that cycle to whatever generic rotation happens to offer next. The rung itself is
# free to refuse — the cycle it burns is not.
#
# Measured on digit, jul31, same build, same ladder, four runs. Cycle 1 place=Explore
# regresses on this design every time, arming the ladder
# [ExtraNetDelay_high, AltSpreadLogic_medium, ExtraNetDelay_low]. Rung 1
# ExtraNetDelay_high is refused every time (est 1334-1572s). What differs is only
# what rotation served in the burned cycle:
#
#   jul30 (alpha +72.59)  fall-through drew __LASTMILE__    91s
#                         -> rungs 2,3 still affordable; ExtraNetDelay_low at cycle 4
#                            took -0.821 -> -0.614, the whole win
#   jul31 (alpha +40.36)  fall-through drew __ROUTE_ONLY__ 845s
#                         -> AltSpreadLogic_medium then unaffordable (667s > 310s)
#                            ladder never reached rung 3
#
# So a 32 MHz spread on one design was decided by an unpriced rotation draw in a
# cycle the ladder had already claimed. That is not a threshold being wrong; it is a
# claimed cycle being given away. When armed, an unaffordable rung advances to the
# NEXT rung within the same cycle instead of surrendering it.
#
# Deliberately NOT a reordering of the ladder — reordering by cost is the jul28
# near-met rule, which carries a jul29 "keep OFF permanently" verdict
# (project_ladder_order_by_wns_jul28). The ladder's order is untouched here; only
# the cost of refusing a rung changes.
#
# _rungs_popped still counts POPS, not runs, so a skipped rung cannot promote a
# later rung into the unreserved rung-1 slot — the budget protection the 5/5 panel
# required is preserved exactly.
#
# DEFAULT ON since aug01. Kill switch: FPL26_ILS_LADDER_SKIP_UNAFFORDABLE=0.
#
# PROVENANCE, corrected aug02 (CORRECTION_e375b7d_aug02.md — the original text
# here claimed a causal matched-pair A/B; adversarial review showed the digit
# A/B was a NULL under its own pre-registered rule, identical multisets, and
# the 40.36 row it cited was a spliced historical run). What actually carried
# the promotion: the mechanism (documented above from four jul30/jul31 runs),
# a 21-run census (p~0.42 of the bad branch), netted EV ~+4.3 MHz/run, and an
# explicit one-time USER RISK-ACCEPTANCE recorded in INTEGRATION_v2plus —
# its corescore never-worse gate FAILED as pre-registered. A risk-accepted
# trade, not a passed gate.
#
# This changes NO budget: an unaffordable rung advances within the cycle the
# ladder already owns. It is NOT a ladder reorder (that carries its own refusal),
# and _rungs_popped still counts POPS, so the 5/5 panel's budget protection is
# preserved byte for byte.
def ladder_skip_unaffordable_enabled() -> bool:
    """DEFAULT ON since aug01. Kill switch: FPL26_ILS_LADDER_SKIP_UNAFFORDABLE=0."""
    return os.environ.get(
        "FPL26_ILS_LADDER_SKIP_UNAFFORDABLE", "1").strip().lower() in (
            "1", "true", "on", "yes")


# ---- WALL FENCE (aug02) -----------------------------------------------------
# The ladder affordability check reads `remaining = deadline_ts - now`, and
# deadline_ts is ALREADY fenced by the 500s polish reserve
# (dcp_optimizer.py: `_ils_spec_deadline = deadline - _pr`), with the 300s
# finalize tail netted out of `deadline` above that. So at the refusal that
# matters ILS under-reads the wall it could legitimately claim:
#
#   digit aug01:  "AltSpreadLogic_medium unaffordable (est 667s > 297s
#   remaining)" — while 1089s sat above the fence and was then spent on a
#   winner-polish pass that returned NO_GAIN. Across stable16_aug01, 13,216s
#   was stranded this way and the polish converted it at 0.652 MHz/1000s
#   (POLISH_ECONOMICS_aug01.md); 10 of 14 polish runs returned NO_GAIN.
#
# When armed, a LADDER RUNG (and only a ladder rung — the forced place-retry
# path) may judge affordability against the wrapper window instead:
# `wrapper_deadline_ts`, which still keeps the finalize reserve. The borrow is
# min()-capped at WALL_FENCE_MAX_BORROW_S so a bogus wrapper_deadline_ts can
# never widen the window past the polish reserve it is deliberately spending.
# Rotation combos, the density gate, the redraw reserve, futility and every
# exit condition keep the fenced deadline — this changes which RUNGS run, not
# how long the loop lives.
#
# Fail-closed: wrapper_deadline_ts missing or malformed => borrow 0 =>
# byte-identical to the shipped behaviour. Flag OFF => identical arithmetic.
#
# WHY THE POLISH CAN AFFORD TO LOSE THE WINDOW HERE: the polish is never-worse
# by construction (replaces output only on IMPROVED) and self-gates on its
# remaining window, so a rung that eats the reserve costs polish wall, never
# alpha. ⚠️ BUT THE PROTECTION IS ECONOMIC, NOT STRUCTURAL (adversarial review
# MINOR 4): _wf_extra > 0 exactly when the polish reserve is armed, which is
# exactly when a polish stage IS eligible on this design — so every fence
# admission, by construction, takes window from a live polish stage. The
# justification is the measured exchange rate (polish 0.652 MHz/1000 s vs a
# refused ladder cycle worth up to 32 MHz), not impossibility of harm. The
# finn case makes it concrete: the aug03 v2.2 sweep shows finn DOES arm the
# ladder (ExtraNetDelay_low refused by 17 s, 674 vs 657) and its polish then
# IMPROVED (+0.062 ns = 7.28 MHz). With the fence, finn runs that rung and may
# starve the paying polish — the priced risk case of the A/B
# (WALLFENCE_AB_PREREG_aug02.md rule 3). An admitted rung also displaces
# whatever cheap rotation combo the fenced remaining could still have funded
# (review MINOR 5) — the matched-pair α comparison in the A/B prices both.
#
# DEFAULT OFF — never A/B'd. Promotion requires the INTEGRATION_v2plus gates.
WALL_FENCE_MAX_BORROW_S_DEFAULT = 500.0


def ils_wall_fence_enabled() -> bool:
    """DEFAULT OFF (aug02). Enable: FPL26_ILS_WALL_FENCE=1;
    FPL26_NO_ILS_WALL_FENCE=1 force-disables and WINS."""
    if os.environ.get("FPL26_NO_ILS_WALL_FENCE", "").strip().lower() in (
            "1", "true", "on", "yes"):
        return False
    return os.environ.get("FPL26_ILS_WALL_FENCE", "").strip().lower() in (
        "1", "true", "on", "yes")


# ---- ExtraNetDelay_high DENSITY GATE (jul31) --------------------------------
# The affordability gate asks "can I afford this?" and never "is this likely to
# help?". ExtraNetDelay_high is the costliest combo in the rotation (measured
# 1.42-3.24x an Explore cycle, typically ~1300s) and across 112 corpus runs on 10
# designs it ACCEPTS ONLY 33% OF THE TIME. The 67% that fail are pure loss: on
# digit one such cycle burned 1273s of a ~1400s window and regressed (-0.895 vs
# an incumbent -0.821), costing ~38 MHz against the run that refused it.
#
# THE SPLIT IS PER-DESIGN AND ENORMOUS -- and it is separable by a feature the
# optimizer already computes in Phase 1, before any cycle runs:
#
#   FAILING-ENDPOINT DENSITY = failing_endpoints / critical_path_avg_spread_tiles
#
#   ARMED   3d 1314 (4/4 accept)  spam 808 (10/17)  optical 518 (14/14)
#   BLOCKED vexriscv_v2 255  finn 182  digit 175  mini-isp 86  vexriscv 36
#           logicnets 14  fir 1
#
# Reads physically: ExtraNetDelay_high targets NET DELAY, so it pays when failing
# paths are numerous AND spatially concentrated, and wastes ~1300s when they are
# few and smeared across the die. Same shape as the PARTIAL_RUIN spread gate
# above (spread~0 cell surgery is 26/26 negative) -- one combo family, one
# Phase-1 feature, fail-open on unmeasured.
#
# THRESHOLD 363 = the GEOMETRIC MIDPOINT of the observed gap (255 -> 518), chosen
# for maximum margin on both sides rather than fitted. Leave-one-design-out
# refitting picked ~260 in 8 of 10 folds, but 260 sits 2% off vexriscv_v2's 255
# and would be fragile; 363 is 1.43x clear of both neighbours. Density is a
# CONSTANT per design in the corpus (min == max on all 10), so the gate never
# flips mid-run.
#
# MEASURED COST OF THE RULE, stated plainly: leave-one-design-out keeps 29 of 33
# accepts and avoids 38 of 68 wasted arms. The 5 lost accepts at a fixed
# threshold are vexriscv (4) and mini-ISP (1) -- both designs where the combo's
# median dWNS is NEGATIVE (-0.112, -0.008), i.e. the accepts it loses are the
# marginal tail of a combo that hurts those designs on average.
#
# ⚠️ OVERFITTING RISK, NOT DISMISSED: 10 designs, 3 on the armed side. The gap is
# a property of 10 points and a hidden design can land anywhere in it. This is
# why the gate fails OPEN on any missing feature and why the threshold
# maximises margin instead of fitting the folds. (STALE LINE FIXED aug02: this
# comment said "DEFAULT OFF pending an A/B" long after the jul31 promotion —
# the gate ships DEFAULT ON, see endhigh_density_gate_enabled() below, promoted
# on the digit-BLOCKED/optical-ARMED matched live A/B.)
# ---- RE-DRAW RESERVE (jul31, fir) -------------------------------------------
# An UNPROVEN stage may not spend the wall that funds a PROVEN alternative.
#
# THE MEASURED FAILURE, fir, stable16_jul31, one run, four log lines:
#
#   [wall-economics] STOP ... returning remaining 2923s to the wrapper   (~577s in)
#   ... ILS then spent ~2000s on 10 cycles, best -0.249 -> -0.243 (+0.006 ns) ...
#   [multi-restart] stop: remaining 763s < floor 1200s                   (NO attempt 2)
#   [multi-restart] winner-polish: 763s stranded -> NO_GAIN
#
# The LLM loop correctly declared itself saturated and handed back 2923s — more
# than the wrapper's 1200s attempt floor. The ILS then ate two thirds of it for
# 0.006 ns, leaving 763s, and the re-draw was starved by a stage that had already
# been told it was earning nothing.
#
# WHAT THE RE-DRAW IS WORTH ON THAT DESIGN: fir's alpha is decided by its ILS
# BASELINE and nothing else (6/6: -0.216 -> 21.30 three times, -0.247/-0.249 ->
# 9.07 three times; FIR_VARIANCE_IS_PHYSOPT_DEPTH_jul29.md, where five hypotheses
# for what SETS the baseline are already dead). The split is ~50/50, so a second
# independent attempt is worth ~0.5 x 12.23 = +6 MHz expected, against the ILS's
# measured +0.006 ns.
#
# WHY THIS SHAPE AND NOT A CYCLE COUNT. The obvious fix — "stop the ILS after K
# futile cycles" — KILLS DIGIT. digit's +72.59 came from ILS cycles taken after
# the identical handback, and its first three cycles were futile (regress,
# equal, regress) before ExtraNetDelay_low took -0.821 -> -0.614 on cycle 4. Any
# K<=3 destroys it and K>3 is a constant fitted to two designs. So the gate is
# not "how long has this run", it is "has this stage EARNED the right to spend
# the alternative's budget":
#
#   digit  meaningful accept at cycle 4  -> UNLOCKED, spends freely. Unchanged.
#   fir    never a meaningful accept     -> capped, leaves the floor, re-draw runs.
#
# Same shape as polish-reserve and ladder-reserve already in this file: a
# speculative stage is fenced out of a reserve that a better-evidenced use needs.
#
# ⚠️ CORRECTED jul31, AFTER a first version that would have KILLED digit.
#
# v1 compared the wrapper-level reserve against the ILS's OWN deadline, which is
# already fenced by the polish reserve (dcp_optimizer.py: `_ils_spec_deadline =
# deadline - _pr`, _pr=500s). Replayed on digit's densgate run that stopped the
# loop at the cycle-4 head — 719s < 1200s — one cycle before ExtraNetDelay_low
# took -0.821 -> -0.614 and earned alpha 72.59. The re-draw money lives at the
# WRAPPER level; comparing it to a fenced remaining is simply the wrong
# subtraction, not a conservative one.
#
# v2 (this code) fixes both halves:
#   1. reads cfg.wrapper_deadline_ts, the UNFENCED budget. On digit's cycle-4
#      head that is 1746 - 527 = 1219s vs a 1200s reserve -> CONTINUES.
#   2. ...but by 19 SECONDS, and a 19s margin deciding 38 MHz is the same
#      knife-edge that cost 17.7 points (optical, 58s) and 5.90 MHz (7s). So the
#      reserve additionally may not fire before REDRAW_MIN_CYCLES_DEFAULT cycles.
#      digit's win is cycle 4; by cycle 5 it HAS a meaningful accept and the
#      reserve is disarmed on merit rather than on timing luck.
#
# Verified against both designs' real traces:
#   digit  win at cycle 4, meaningful accept -> reserve never fires. UNCHANGED.
#   fir    no meaningful accept in 10 cycles -> fires around cycle 7 leaving
#          ~1425s, over the 1200s floor, so attempt 2 can run.
#
# ALSO CONDEMNS THE PRE-EXISTING ALTERNATIVE: --split-aware (jul14, default OFF)
# caps digit (SMALL, 17.9MB) at 1800s while its winning cycle lands at 2076s —
# cut by 276s. That fix would destroy digit outright.
#
# STILL DEFAULT OFF. The arithmetic is now right and trace-verified on two
# designs, but it has never run live and it changes how wall is divided on every
# design that reaches ILS without a meaningful accept. Fails open when
# wrapper_deadline_ts is unknown.
ENDHIGH_PD = "ExtraNetDelay_high"
ENDHIGH_DENSITY_MIN_DEFAULT = 363.0
# Budget-share ceiling. digit's two rejected cycles sat at 0.81 and 0.98 of the
# remaining window; every accept the density gate would otherwise have cost sits
# below 0.60 (vexriscv 0.15-0.21, logicnets 0.51-0.52, mini-ISP 0.21). 0.60 is
# the widest value that loses nothing on the corpus.
ENDHIGH_SHARE_MIN_DEFAULT = 0.60
# The multi-restart attempt floor. Kept as a named constant so the reserve and
# the wrapper cannot drift apart into a reserve that funds nothing.
REDRAW_RESERVE_S_DEFAULT = 1200.0
# A design's ILS may not be judged unproductive before it has had this many
# cycles. digit's winning cycle is its FOURTH (-0.821 -> -0.614, the whole
# +38 MHz) and at that cycle's head the correct wrapper-level arithmetic clears
# the reserve by only 19 SECONDS. A 19s margin deciding 38 MHz is the same
# knife-edge that cost 17.7 points (optical, 58s) and 5.90 MHz (ROUTE_ONLY, 7s).
# This floor removes the knife-edge entirely: by cycle 5 digit HAS its meaningful
# accept and the reserve is disarmed on merit, not on timing luck.
REDRAW_MIN_CYCLES_DEFAULT = 5


def endhigh_density_gate_enabled() -> bool:
    """DEFAULT ON since jul31. Kill switch: FPL26_ILS_ENDHIGH_DENSITY_GATE=0.

    Promoted on a matched live A/B, one build, ship config (MEASURED_PRIORS=1),
    both directions of the gate exercised on the same day:

      digit   density 175 -> BLOCKED -> alpha +72.59  fmax 439.56  (== its record)
      optical density 517 -> ARMED   -> alpha +32.38  fmax 357.27  (== its record)

    Both hit their known best-ever simultaneously on ONE configuration, which was
    not previously possible: before this gate, digit scored 72.59 only with
    MEASURED_PRIORS OFF, and turning that off costs optical the 19.32 MHz / ~17.7
    eval points the priors fix earned. The gate dissolves that trade.

    Kept as a code default rather than a Makefile injection, matching
    FPL26_ILS_SEED_COPY: the Makefile layer OVERRIDES python defaults, so a flag
    that belongs on every run belongs here (project_ship_config_composition_jul30).
    """
    return os.environ.get(
        "FPL26_ILS_ENDHIGH_DENSITY_GATE", "1").strip().lower() in (
            "1", "true", "on", "yes")


def endhigh_density_min() -> float:
    """Threshold, env-overridable via FPL26_ILS_ENDHIGH_DENSITY_MIN.

    A non-numeric or non-positive override is IGNORED (falls back to the
    default) rather than silently disabling the gate -- the jul30 lesson that a
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
    """Armed only by FPL26_ILS_REDRAW_RESERVE. DEFAULT OFF."""
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
    """(blocked, reason) for ExtraNetDelay_high, on TWO conditions that must BOTH
    hold: the design does not benefit AND the cycle would bet the window.

    WHY BOTH, measured on the 112-run corpus (jul31):
      density alone  -> blocks 8 of 16 designs wholesale and costs 5 accepts,
                        4 of them vexriscv's (density 36, yet 38% accept rate).
      share alone    -> at 80% it kills an OPTICAL accept sitting at 0.80.
                        Optical is a SCORED design carrying 17.7 measured eval
                        points; that is disqualifying on its own.
      both together  -> LOSES ZERO of 37 accepts and still avoids 8 wasted
                        ~1300s arms (fir 5, digit 2, finn 1).

    Reads as: density answers "is this directive worth anything on this design?"
    and share answers "and am I betting the remaining window on it?". digit's two
    rejected cycles sat at share 0.81 and 0.98 — the highest in the corpus — while
    every accept that would otherwise be lost sits below 0.60 (vexriscv 0.15-0.21,
    logicnets 0.51-0.52, mini-ISP 0.21) or is density-exempt (optical 518).

    FAILS OPEN on every uncertainty: gate disarmed, either density feature
    unmeasured/non-finite, non-positive spread, or an unpriceable cycle (est or
    remaining missing/non-finite/non-positive). Blocking a combo worth 17.7
    measured eval points because a RapidWright spread analysis was skipped, or
    because a cost estimate was unavailable, would be far worse than the waste
    the gate exists to prevent.
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
    # NaN/inf must be caught EXPLICITLY and before any comparison. Every
    # comparison against NaN is False, so `spread <= 0` does NOT catch it and a
    # NaN would fall through to `dens >= thr` -> also False -> BLOCKED. An
    # infinite spread is worse: density becomes exactly 0.0, which compares
    # cleanly and blocks. Either way the function would block an optical-like
    # design while claiming to fail open — the 17.7-point failure mode this
    # guard exists to prevent. (Found by gpt-5.6-sol, panel_endhigh_jul31.md.)
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
    # Density says this design does not benefit. That alone is NOT enough — it
    # would cost vexriscv 4 accepts. Only block if the cycle would also bet the
    # remaining window.
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


# STOP-ON-ACCEPT. The ladder exists to answer ONE question — which placement family
# is right for this design. An ACCEPT answers it, so any further rung is testing a
# directive that must now beat the ladder's OWN winner, at a full cycle each.
#
# Found live on spam (batch 1, jul28) rather than reasoned into existence:
#   cycle 2  rung 1 ExtraNetDelay_high    -0.665  (= incumbent, no accept)  542s
#   cycle 3  rung 2 AltSpreadLogic_medium -0.598  ACCEPT                    260s
#   cycle 4  rung 3 ExtraNetDelay_low     a directive ALREADY measured worse
#                                         on this design (-0.647)
# and the ruin window closed right after, so the run spent its last cycle on a
# known loser instead of on the escalation carrying spam's remaining +11.05 MHz.
#
# Against every design with rung data this dominates the budget reserve:
#   spam     rung 1 no-accept -> rung 2 ACCEPT -> stop; window freed for the tail
#   3d       rung 1 ACCEPT (-2.153 -> -1.997) -> stop; SKIPS AltSpreadLogic_medium,
#            which is -2.544 on 3d — stopping actively avoids a bad cycle
#   optical  rung 1 ACCEPT (-0.924 -> -0.846) -> stop; skips -1.086 and -1.058
# It COMPOSES with the reserve rather than replacing it: the reserve bounds cost
# when nothing accepts; this bounds cost the moment something does.
# ---- INCR-ROUTE PRIORITY (jul28). Its precondition is PERISHABLE. -----------
#
# spam's record chain (S3) escalates an EXPLORE-routed incumbent to
# AggressiveExplore. S6 ran the same two directives in the opposite order and the
# whole gain vanished (-0.595 vs -0.543), so the operator REQUIRES an incumbent
# that has not yet been routed aggressively.
#
# But INCR_ROUTE is APPENDED LAST, while ROUTE_REROLL (idx 2) and ROUTE_ONLY
# (idx 3) both re-route from scratch WITH AggressiveExplore. They run first and
# destroy the precondition. Observed live, spam jul28 arm=stopaccept, which
# reached +20.85 and then logged, twice:
#
#   [ils] incr-route skipped: incumbent router AggressiveExplore is not weaker
#         than AggressiveExplore (S6: escalation order is causal)
#
# The operator was armed, correct, and unreachable — the rotation had consumed the
# state it needed. That is a SEQUENCING defect, not a missing capability, and
# spam's remaining ~8.3 MHz to +29.19 sits behind it.
#
# When armed AND currently eligible, take the next cycle ahead of the from-scratch
# route sentinels. Cost where it does not reproduce (3d and optical both measure
# 0.000 ns) is ONE cheap cycle: prior 0.5, no place step, no unroute.
def incr_route_first_enabled() -> bool:
    """DEFAULT ON (jul29); still needs FPL26_ILS_INCR_ROUTE. Disable with =0.

    This is the escalation spam needs: without it spam measures +17.51 instead of +29.19.
    Its queue-jumping is bounded by the displacement guard (5d3d15d), which is the one
    mechanism here that has been confirmed making both decisions correctly inside a live run.
    """
    return os.environ.get(
        "FPL26_ILS_INCR_ROUTE_FIRST", "1").strip().lower() in ("1", "true", "on", "yes")


def spam_escalation_guard_enabled() -> bool:
    """FPL26_SPAM_ESCALATION_GUARD (aug03, DEFAULT OFF) — displacement-FREE
    incr-route escalation (SPAM_2616_ATTRACTOR_aug03.md §4-5).

    The jul28 displacement guard yields the priority jump whenever a
    from-scratch route still fits — which on a healthy/fast box is ALWAYS
    true at the decision cycle, so v2.2 deterministically takes the reroll
    branch, the reroll re-routes with AggressiveExplore, and S6 blocks the
    escalation forever (spam attractor +26.16 vs the +29.19 class).  When
    the window is big enough that the escalation AND the reroll both fit
    (remaining >= cost(incr) + reroll_need), firing the escalation first
    displaces nothing — keep-best makes the extra cycle never-worse against
    the sentinels it competes with.  3d's measured -5.96 MHz case stays
    untouched: 441 s remaining < ~142 s incr + 376 s need, condition false.
    Score-neutral on the current scored 5; promotion still requires the
    16-design never-worse sweep (residual risk: the ~150-190 s spent can
    push an unrelated later combo out of the window on some other design).
    """
    return os.environ.get(
        "FPL26_SPAM_ESCALATION_GUARD", "0").strip().lower() in (
            "1", "true", "on", "yes")


def ladder_stop_on_accept_enabled() -> bool:
    """Armed only by FPL26_ILS_LADDER_STOP_ON_ACCEPT."""
    return os.environ.get("FPL26_ILS_LADDER_STOP_ON_ACCEPT",
                          "0").strip().lower() in ("1", "true", "on", "yes")


# INCR-ROUTE TERMINAL: fire only when the loop has nothing better left to do —
# either it is one cycle away from the futility stop, or no full-place combo is
# affordable in the remaining window. Both are STATE FACTS. This keeps a
# possibly-false lever from competing with full ruin cycles that have corpus-wide
# evidence behind them, while still letting it run as the last cheap move.
def incr_route_terminal_enabled() -> bool:
    """Armed only by FPL26_ILS_INCR_ROUTE_TERMINAL."""
    return os.environ.get(
        "FPL26_ILS_INCR_ROUTE_TERMINAL", "0").strip().lower() in ("1", "true", "on", "yes")


# ---- STUCK-SEED THRESHOLD OVERRIDE (jul28). Default = cfg value, unchanged. ----
#
# Which NETLIST the ILS re-places from is decided here, and on optical it decides
# the design's record. `ils_seed_kind` returns "raw" when the recipe gained less
# than this, else "recipe_best". Measured live jul28:
#
#   optical  recipe_gain 0.154 -> recipe_best seed
#            ExtraNetDelay_high on the recipe-modified netlist = -0.971
#   the jul27 sweep placed the SAME directive from the RAW benchmark = -0.846
#   spam     recipe_gain 0.021 -> raw seed, and it reproduced its sweep number
#            (-0.598) to the digit
#
# So optical's +34.69 is a RAW-SEED result, and optical misses the raw path by
# 0.004 ns. Corpus recipe_gain: spam 0.021, corescore 0.141, optical 0.154,
# vtr 0.211, logicnets 0.452, finn 0.654, mini-ISP 0.730, vexriscv 1.035 — so
# corescore and optical STRADDLE the 0.15 cut 0.013 ns apart, which falsifies the
# threshold's own comment that "the (0.05, 0.15) band is otherwise unpopulated".
#
# Env override only, so the shipped default is untouched; this exists to MEASURE
# whether the raw path is what optical needs before anyone moves the constant.
# The dual-seed path it selects is raw -> recipe-best, keep-best and never-worse
# by construction, so the cost of being wrong here is wall time, not MHz.
def stuck_gain_threshold(cfg) -> float:
    """cfg.stuck_recipe_gain_ns unless FPL26_ILS_STUCK_GAIN_NS overrides it."""
    raw = os.environ.get("FPL26_ILS_STUCK_GAIN_NS", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return float(getattr(cfg, "stuck_recipe_gain_ns", 0.15))


# ---- INCREMENTAL ESCALATED RE-ROUTE (jul28 integration of the jul27 probe) ----
INCR_ROUTE_COMBOS: List[Tuple[str, str, str]] = [
    (INCR_ROUTE_PD, "AggressiveExplore", "AlternateFlowWithRetiming"),
]


def incr_route_enabled() -> bool:
    """DEFAULT ON (jul29). Disable with FPL26_ILS_INCR_ROUTE=0 for the shipped rotation."""
    return os.environ.get(
        "FPL26_ILS_INCR_ROUTE", "1").strip().lower() in ("1", "true", "on", "yes")


def place_retry_enabled() -> bool:
    """Armed only by FPL26_ILS_PLACE_RETRY. Unset/0 -> rotation exactly as shipped."""
    return os.environ.get(
        "FPL26_ILS_PLACE_RETRY", "0").strip().lower() in ("1", "true", "on", "yes")


def reject_micro_accept_enabled() -> bool:
    """Armed only by FPL26_ILS_REJECT_MICRO_ACCEPT. DEFAULT OFF.

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


# Combos that REUSE an existing placement instead of running a full place_design.
# Two uses: the place-retry trigger excludes them (a regression cannot indict a
# placement family that was never re-run) and the measured cost basis below
# excludes them (their wall is not a full-place cycle's wall).
PLACE_SENTINELS = (LASTMILE_PD, ROUTE_ONLY_PD, PARTIAL_RUIN_PD, ROUTE_REROLL_PD,
                   INCR_ROUTE_PD)

# ---- MEASURED COST BASIS (jul28). DEFAULT OFF = byte-identical behaviour ----
#
# WHY. The cold-start affordability gate sizes an unseen combo as
# COMBO_COST_PRIOR[directive] x cfg.expected_heavy_cycle_s. That anchor is derived
# in dcp_optimizer from the RECIPE phase's place_design single -- measured in a
# session the ILS then deliberately restarts ("clears recipe session pollution
# that degrades + slows place_design") -- and it is published RAISE-ONLY, so a
# later cheaper measurement can never correct it downward.
#
# MEASURED on optical-flow (gate_ab_jul27 banded run, agent.log):
#
#   cold-start anchor          672s   (place_design single = 501s of it)
#   real full-ruin cycles      225s, 259s   <- same design, same box, fresh session
#   inflation                  ~2.7x
#
# Consequence, arithmetic from that run: ExtraNetDelay_high is priced at
# 3.0 x 672 = 2016s against 1885s remaining after cycle 1, so it is refused as
# unaffordable -- on the design whose OWN placement sweep ranks it FIRST
# (final_round/OPTICAL_RESULTS_jul28.md: -0.846 = +26.48, best of nine
# directives). The rotation instead reached SSI_SpreadLogic_high (prior 1.1) and
# stopped with 278s unspent. Same shape as the jul04 v2 leg already documented at
# the ROUTE_ONLY branch below, which was fixed for route-only combos ONLY.
#
# THE FIX. Once a real full-place cycle has completed, its wall -- normalised by
# that directive's own prior -- IS a measurement of this design on this box, and
# a measurement strictly dominates a derived anchor (the same rule deep-replace's
# anchor publication already states). Take the MAX over observed cycles so the
# basis stays conservative, and carry the same x1.15 margin combo_cost uses.
MEASURED_BASIS_MARGIN = 1.15


def measured_basis_enabled() -> bool:
    """DEFAULT ON (jul29). Disable with FPL26_ILS_MEASURED_BASIS=0 for the cold-start anchor.

    Prices a cycle from a real measurement instead of the derived anchor, which the jul28
    corpus showed over-priced every design that produced both numbers by 1.6-3.5x.
    """
    return os.environ.get(
        "FPL26_ILS_MEASURED_BASIS", "1").strip().lower() in ("1", "true", "on", "yes")


def cold_start_basis(cfg, measured_basis_s: float) -> float:
    """Absolute basis for pricing an UNSEEN combo (cost prior x this).

    Returns the measured full-place cycle wall when armed and one has been
    observed, else cfg.expected_heavy_cycle_s exactly as shipped.
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


# Relative cycle-cost priors per place directive (Explore full-ruin = 1.0).
# From the jun12 cross-log cost extraction (80 cycles, 14 designs): within a
# design the RELATIVE combo costs are stable even though absolute costs vary
# 5-10x across designs/boxes. Used only for the cold-start affordability
# check (before a combo has an observed cost) and only when the caller
# provides an absolute anchor (cfg.expected_heavy_cycle_s).
# ---- MEASURED PRIOR OVERRIDE (jul30, DEFAULT OFF) ----------------------------
# FPL26_ILS_MEASURED_PRIORS=1 replaces the two demonstrably overstated entries
# below with their CORPUS-MEASURED medians.
#
# WHY THIS EXISTS, and it is the most expensive thing found this round. The jul30
# preview scorecard scored optical at alpha +13.07 where the farm reproduces
# +32.38 on the same ship config. The eval's own log says why, in one line:
#
#   ExtraNetDelay_high unaffordable (est 1058s > 1000s remaining);
#                                    continuing normal rotation
#
# 1058 = 307s measured basis x COMBO_COST_PRIOR 3.0 x 1.15 margin. It was refused
# by 58 SECONDS. That cycle is the one that earns optical its gain: on the farm it
# accepts at -0.846, and __ROUTE_ONLY__ then builds on that placement to reach
# -0.799. Losing the first link means ROUTE_ONLY re-routes the WORSE placement
# (-0.999) and the run ships the recipe-best having accepted nothing. 19.32 MHz,
# ~17.7 score points, on one constant, by 58 seconds.
#
# THE CONSTANT IS AT THE p90, NOT THE MEDIAN. Measured across the corpus by
# pairing each combo's dt against its OWN RUN's Explore dt (which controls for
# design and box -- a cross-design median would mix a big design's Explore with a
# small design's ExtraNetDelay):
#
#   ExtraNetDelay_high  n=131, 10 designs: p50 2.10, p90 3.23   shipped 3.0
#     per-design p50: fir 1.42, optical 1.96, 3d 2.05, spam 2.08, v2 2.11,
#                     mini-ISP 2.40, vexriscv 2.60, logicnets 3.23
#     -> 3.0 sits above NINE of the ten designs' medians.
#   __LASTMILE__        n=81,  12 designs: p50 1.09, p90 1.59   shipped 2.5
#     per-design p50 range 0.41 .. 1.79 -- NOT ONE design reaches 2.5.
#
# THE ASYMMETRY IS THE ARGUMENT. The gate reads a refusal as free ("continuing
# normal rotation"), and it is not: an OVERRUN costs bounded wall (gamma; the
# cycle is deadline-capped and the banked best is never-worse, so alpha is safe),
# while a WRONG REFUSAL costs the entire gain -- here 19.32 MHz against a ~0.5 MHz
# gamma exposure. A cost prior that must choose a quantile should therefore sit at
# the MEDIAN, not the p90. Same error class as the jul28 anchor inflation, which
# FPL26_ILS_MEASURED_BASIS fixed for the ABSOLUTE basis while leaving these
# RELATIVE priors at their jun12 estimates.
#
# Deliberately NOT touching the three combos the corpus shows UNDER-priced
# (__ROUTE_ONLY__ 0.6 vs 0.73, __PARTIAL_RUIN__ 0.7 vs 0.85, __ROUTE_REROLL__
# 0.55 vs 0.63): raising those would cause MORE refusals, which is the failure
# mode this whole note is about. Fewer moving parts also keeps the A/B readable.
MEASURED_COMBO_COST_PRIOR: dict = {
    "ExtraNetDelay_high": 2.10,   # corpus p50 (shipped 3.0 = p90)
    # __LASTMILE__ 2.5 -> 1.10 was in this table and was REMOVED after wave23.
    # The corpus case was strong (p50 1.09; no design of twelve reaches 2.5) but
    # the SAFETY arm priced it and it loses: on logicnets, making LASTMILE
    # affordable made it RUN (409s, cycle 5) in place of two cheap cycles, and it
    # rejected at -0.468 against a -0.466 incumbent. alpha identical, wall +36s,
    # calls 9->12 => ~-0.36 points, all of it from this one entry.
    #
    # And it carries none of the benefit: optical's +19.31 MHz trace is
    # Explore -> ExtraNetDelay_high -> __ROUTE_ONLY__, no LASTMILE in it. The
    # in-rotation LASTMILE combo is marginal anyway -- p=0.148, median gain
    # 0.021ns = 3.36 MHz at fmax 400, under the 3.5 MHz noise floor
    # (COMBO_ECONOMICS_jul30.md). Making a marginal combo cheaper just makes it
    # run more often.
    #
    # "The prior is wrong" was true and "therefore fix it" did not follow. Keep
    # the entry that carries the 17.7 points; drop the one that only costs.
}


def measured_priors_enabled() -> bool:
    """DEFAULT OFF. FPL26_ILS_MEASURED_PRIORS=1 arms the measured priors."""
    return os.environ.get(
        "FPL26_ILS_MEASURED_PRIORS", "0").strip().lower() in ("1", "true", "on", "yes")


def combo_cost_prior(pd: str) -> float:
    """Relative cycle-cost prior for `pd`, in Explore=1.0 units.

    THE ONLY read path. Every affordability site must go through this so the
    override cannot be half-applied -- the jul30 half-deploy lesson applied to a
    constant instead of a module.
    """
    if measured_priors_enabled():
        _m = MEASURED_COMBO_COST_PRIOR.get(pd)
        if _m is not None:
            return _m
    return COMBO_COST_PRIOR.get(pd, 1.0)


COMBO_COST_PRIOR: dict = {
    "Explore": 1.0,
    # route-only skips the place step entirely; DRILL H per-design route+physopt
    # times were ~0.4-0.7x of a full-ruin cycle on the same design/box.
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
    # jul27 measured (spam Explore=267s basis; vexriscv agrees): the medium/low
    # variants are CHEAP, unlike their _high siblings above.
    "AltSpreadLogic_medium": 1.05,
    "ExtraNetDelay_low": 1.2,
    # jul28: one route + polish on an ALREADY-routed design, no place step and no
    # unroute — cheaper than ROUTE_ONLY's from-scratch solve (spam probe: 120-160s
    # against a ~270s full-ruin cycle).
    INCR_ROUTE_PD: 0.5,
}

# Defaults (overridable). Tuned from the 13-design sweep.
DEFAULT_MAX_CELLS = 300_000          # size gate: above this, <2 cycles fit
# Need a real execution window: the agent's budget-aware dispatcher SKIPS risky
# tools (place/route/phys_opt) when remaining budget < ~5-8 min, so ILS only
# does real work above that floor. Require >=1500s so >=2 cycles actually run
# before skipping kicks in.
DEFAULT_MIN_REMAINING_S = 1500.0
DEFAULT_ACCEPT_MARGIN_NS = 0.002     # strictly-better threshold
DEFAULT_MAX_CYCLES = 60              # hard backstop vs runaway loop
DEFAULT_MIN_CYCLE_SECONDS = 20.0     # a real cycle takes minutes; faster = skipped
# Stagnation is measured in WALL-TIME, not iterations: one agent "iteration" is
# a full multi-turn LLM tool session (~10-15 min), so an iteration counter is
# far too coarse to fire the preempt within budget. Seconds-since-last-
# improvement is budget-aware and fires after the first stuck stretch.
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
    # PROVEN on v2 (A/B/C test 2026-06-07): the recipe phase leaves persistent
    # state in the Vivado PROCESS that degrades every later place_design (worse
    # WNS AND ~2x slower) and open_checkpoint does NOT clear it. A fresh Vivado
    # restart before the global re-place recovers it (-0.942 -> -0.84, +17 MHz).
    restart_vivado_before_ils: bool = True
    # Seed the ruin-and-recreate from the RAW input DCP (a true from-scratch
    # global re-place) rather than the recipe-best retimed netlist. Combined with
    # the fresh restart this reproduces the standalone -0.84. keep-best vs
    # recipe-best => never-worse.
    seed_from_raw: bool = True
    # GATE (full-13 2026-06-08): raw-seed only helps STUCK designs (recipe gained
    # ~0, e.g. v2 -> raw reaches -0.84). On designs the recipe genuinely improved
    # (e.g. spam-filter), raw discards the recipe's gains and the ILS lands WORSE
    # (ships recipe-only, losing the recipe-best+ILS gain the prior run captured).
    # So seed from raw ONLY when recipe_gain (best_wns - initial_wns) < this floor;
    # otherwise seed from recipe-best (in the fresh Vivado, >= the prior polluted
    # run by dominance). Unknown gain -> conservative recipe-best.
    # 0.05 -> 0.15 (jul03, preview attempt-4 forensics): with the prompt guard
    # keeping the LLM alive (no more 402 die-off), v2's LLM dribbled +0.061 ns
    # over a 30-min iteration — crossing the old 0.05 floor, classifying v2
    # NOT-stuck, and skipping the raw seed that reliably captures -0.84 (alpha
    # collapsed 17.48 -> 9.88). The 402 wall had been load-bearing for the
    # stuck verdict. A marginal LLM dribble must still read STUCK; a
    # mis-classified stuck is cheap by design (raw runs with the K=2
    # no-improve stop, then the recipe-best corrective probe still runs).
    # Measured full-13 recipe gains cluster at ~0 or >0.3 — the (0.05, 0.15)
    # band is otherwise unpopulated.
    stuck_recipe_gain_ns: float = 0.15
    # WS1b (2026-06-10 wall audit): measure AFTER route, BEFORE phys_opt.
    # If the route left unrouted nets the cycle can NEVER be accepted (ur==0
    # required) — skip phys_opt unconditionally (vexriscv wasted 3 phys_opt runs
    # on ur=45/80/211 cycles). If post-route WNS trails best by more than this
    # margin, phys_opt (typ. +0.05-0.15ns) cannot close it — skip (logicnets
    # cycle 2: -1.193 vs best -0.496). Conservative 0.5ns so no observed accept
    # would ever have been rejected. 0 disables the margin skip.
    physopt_skip_margin_ns: float = 0.5
    # Hold-safety (2026-06-11 gap audit): the contest validator gates
    # hold_passed, but ILS accepted on SETUP wns + routedness only — a cycle
    # that improves setup while breaking hold would ship and ZERO the
    # benchmark at validation. Check worst hold slack on accept candidates
    # (one light query, only when setup already passed). Fail-OPEN when the
    # query can't produce a number (preserves pre-gate behavior on designs
    # where the parse fails). Floor has tiny tolerance for report noise.
    accept_requires_hold_clean: bool = True
    hold_slack_floor_ns: float = -0.001
    # SA-STYLE EXPLORATION (2026-06-11, EXPERIMENTAL — default OFF pending the
    # local v2 A/B): pure greedy ILS re-opens the BEST DCP every cycle, so it
    # can only sample one-step perturbations of one basin. When enabled, a
    # non-accepted cycle whose result is routed and within
    # explore_regression_floor_ns of best becomes the next cycle's STARTING
    # state (classic simulated-annealing walk) — the best DCP on disk remains
    # the untouched ship floor, so never-worse is preserved by construction.
    explore_from_current: bool = False
    explore_regression_floor_ns: float = 0.15
    # DUAL-SEED (full-13 2026-06-08 stable fix): recipe_gain~0 cannot distinguish
    # v2 (raw-seed wins, recipe stuck) from spam/3d/optical (recipe-best-seed wins,
    # recipe already near ceiling) — both read gain~0, so the single gate mis-routed
    # the latter to raw and gave back their ILS gains. When the gate reports a STUCK
    # design (-> raw), run BOTH seeds: raw first (priority budget; preserves the v2
    # capture), then recipe-best as a keep-best corrective probe on the leftover.
    # Non-stuck designs are unchanged (recipe-best only). keep-best => never-worse
    # and strictly >= the single-seed gate on every design.
    dual_seed: bool = True
    # No-improvement early-stop (cycles): in the dual-seed STUCK path the primary
    # (raw) seed stops after this many CONSECUTIVE non-accepting real cycles so it
    # yields the rest of the budget to the corrective recipe-best probe. v2's wins
    # are consecutive accepts at cycles 1-2, so 2 never cuts them; a mis-classified
    # spam-type raw seed never accepts -> bails after 2 -> recipe-best recovers. 0
    # disables (the final/sole seed runs with no early-stop, using all budget).
    no_improve_stop_cycles: int = 2
    # Futility stop for the FINAL/sole seed (jun12 offline replay, 10 jun10-11
    # AWS runs): contest score = alpha*(1 - 0.1*beta - 0.1*gamma) with gamma =
    # wall/3600, so burning the wall on non-accepting ILS cycles costs up to
    # 10% of alpha. Replay: K=2 exits saved 400-1900s on stuck runs with ZERO
    # forfeited accepts (+9.65 score total; K=1 = -34.9, forfeits real accepts;
    # K=3 = +6.8, dominated). Historical late accepts after streak>=2 exist
    # (5/36, jun08 stack) but are tiny (0.004-0.108ns). 0 = legacy
    # run-to-budget behavior. NEEDS-REHEARSAL before final re-submit.
    final_seed_no_improve_stop: int = 2
    # LASTMILE combo entry gate on the current best WNS. The LastMile place
    # directive FAILS outright ("Place design failed") far from closure
    # (3d-rendering at -2.1ns, jun13 overnight). BUT its PROVEN WINS were all
    # well below UG906's nominal -0.25 entry: logicnets -0.496 (+0.059), v2
    # -0.799 (+0.152) / -0.946 (accepted -0.912), mini-isp -0.882 (+0.032).
    # Gain-weighted futility (jul06 preview #12-vs-#13 eval forensics): an
    # accept whose gain over the running best is below this threshold is
    # still KEPT (keep-best; the DCP ships) but does NOT reset — and itself
    # advances — the no-improve streak. Evidence: #13's +0.007ns accept
    # reset the K=2 counter and bought ~24min of dead cycles (gamma
    # 0.577->0.917, -3.4 pts at alpha~100, net -1.2 vs #12); corpus mining
    # over ALL ILS logs (58 accepting seeds) shows micro-accepts (<0.010ns)
    # are TERMINAL 4/4 — never once followed by a meaningful accept in the
    # same seed (observed micro gains 0.004-0.007 vs meaningful 0.031+).
    # 0.0 restores legacy reset-on-any-accept.
    meaningful_accept_ns: float = 0.010
    # An earlier -0.30 gate was a BUG — every contest design ends < -0.30, so
    # it blocked LASTMILE on ALL benchmarks. Gate at -1.0: includes the whole
    # proven-win range (worst -0.946) with margin, blocks the failures (finn
    # -1.37, 3d -2.1, ispd16 -7.8, vtr -11.3). Set very negative to disable.
    lastmile_min_wns_ns: float = -1.0
    # Absolute anchor for the cold-start combo affordability check: expected
    # wall-seconds of one Explore full-ruin cycle on THIS design/box. Caller
    # derives it from the recipe phase's observed place+route+phys_opt tool
    # durations. 0 = unknown -> legacy optimistic cold-start (combo 0 first).
    # jun12 corundum drill: ILS got a ~30min window and spent ALL of it on a
    # full-ruin cycle that could not complete; with this anchor the picker
    # would have started at partial-ruin (0.7x) instead.
    expected_heavy_cycle_s: float = 0.0
    # PRISTINE design baseline WNS (dcp_optimizer's self.initial_wns), NOT the
    # recipe-best that run_ils_polish receives as baseline_wns. Only consumed by the
    # jul28 retry baseline gate. None = unmeasured -> that gate does not apply.
    design_baseline_wns: Optional[float] = None
    # FPL26_MINIISP_RETRY_HOLD (aug05, DEFAULT False): band-scoped arming of
    # the retry baseline gate WITHOUT the global env flag.  Set by
    # dcp_optimizer ONLY when the flag is on AND the measured band is "mid"
    # (1.05 < |wns_in| < 8.0) — see _retry_baseline_gate_active.  Default
    # False = byte-identical to the pre-aug05 behaviour.
    retry_baseline_gate_scoped: bool = False
    # HURDLE-BASED ILS CONTINUATION (jul26). 0.0 = OFF (K-counter alone, the
    # pre-jul26 behaviour). When set to the run's banked alpha in MHz, the
    # futility counter may be overridden for ONE more cycle whenever the
    # scoring function says that cycle pays: hurdle = alpha*0.1*(dt/3600)/P.
    # Measured motivation (mini-ISP chain29): ILS held 2984 s, spent 233 s, and
    # stopped on K=2 — while a ~120 s cycle's hurdle is only ~0.34 MHz and the
    # jul07 record's winning cycle yielded 5.4 MHz.
    hurdle_continue_alpha_mhz: float = 0.0
    hurdle_continue_max_mhz: float = 1.0
    # Per-command timeout for the HEAVY ILS Vivado steps (place/route/phys_opt).
    # BUG (local spam 2026-06-08): run_ils_polish passed NO timeout, so each step
    # used the MCP server's 300s default; on a large design (spam 14MB) place,
    # route and phys_opt each exceed 300s, get killed mid-command, and the cycle
    # returns wns=None (garbage) while burning the whole budget. AWS Vivado fits
    # the contest designs under 300s (full-13 ran clean) so this is latent there,
    # but a slow-routing design would hit the same cascade => rank risk. Pass a
    # generous timeout, capped at the remaining budget so a single step can't blow
    # the wall. keep-best still discards an incomplete (timed-out) cycle.
    heavy_cmd_timeout_s: float = 1800.0
    # Per-command timeout for the LIGHT steps (open_checkpoint, WNS/route_status
    # queries). Fast normally, but slow on a huge design; well above 300s avoids a
    # false no-op when the design is merely large, not stuck.
    measure_cmd_timeout_s: float = 600.0
    # CELL-COUNT SANITY BAND (jul03 GitHub #36; RELAXED jul13, upstream PR
    # #41): the validator's Check-4 hard gate ([0.97x, 1.5x] of the input's
    # primitive count) was REMOVED upstream on jul07 — cell counts are now
    # reported info-only, and the eval box already runs the post-removal
    # validator (attempt #16 scorecard validator_git_sha b2aafb7). The band
    # below is therefore no longer a disqualification guard, only a sanity
    # check that a LASTMILE netlist transform (-lut_opt) didn't mangle the
    # design wholesale. golden_cell_count is set by the caller from the
    # INPUT design; None (or a failed measurement) fails OPEN.
    golden_cell_count: Optional[int] = None
    cell_floor_ratio: float = 0.5
    cell_ceil_ratio: float = 3.0
    # FANOUT POLISH (jun14 DRILL D): one post-route `phys_opt_design -directive
    # AggressiveFanoutOpt` pass on the FINAL best is a never-worse polish that
    # lifts hold-SAFE designs (logicnets +0.029ns/~+6MHz, GT-confirmed). It runs
    # as a dedicated step AFTER the ILS combo rotation (the K=2 futility-stop
    # would make a late-indexed combo unreachable on plateaued ship states —
    # exactly where this helps), gated cheap-designs-only + comfortable budget +
    # STRICT hold floor. WHY strict hold: AggressiveFanoutOpt replication erodes
    # hold (jun14: ispd16 +0.982ns but whs 0.003->0.000, vtr ->0.001) — the
    # +0.010 floor conservatively rejects those hold-marginal candidates (the
    # validator gates hold_passed); never-worse means a reject costs nothing.
    # WHY cheap-only: ispd16's pass is ~24min on a 150MB DCP — paying that gamma
    # for a candidate the hold gate rejects is a SCORE loss, so skip slow designs.
    # NEEDS-REHEARSAL: validated offline only; the final-week AWS dress rehearsal
    # exercises it live before any re-submit.
    fanout_polish_enabled: bool = True
    fanout_hold_slack_floor_ns: float = 0.010
    # cheap-design gate: only attempt when the cold-start anchor
    # (expected_heavy_cycle_s) is known AND below this (fast designs like
    # logicnets ~hundreds of s; excludes ispd16/vtr-class slow opens).
    fanout_max_cycle_s: float = 600.0
    # Dedicated fanout-polish cost anchor: expected wall-seconds of one
    # phys_opt pass + one reroute (the polish's actual worst case) from the
    # recipe phase's observed SINGLE-stage tool durations. Fallback for the
    # cheap-design gate when expected_heavy_cycle_s is 0: no-place recipe
    # paths (R7 closure ladder = phys_opt-only; R1 route lever = route without
    # place) never run place_design, so the full-cycle anchor stays unknown
    # and the polish self-gated on exactly the OOD designs it should serve
    # (jul04 genericity audit: corundum skipped with "cost anchor 0s"). 0 =
    # unknown.
    # PRECEDENCE (jul04, preview #6 RECORD-run evidence): when this anchor
    # contains BOTH a phys_opt and a route sample (fanout_anchor_has_route),
    # it is the polish's true worst case (one phys_opt + one reroute) and is
    # used as the cost basis EVEN IF the full-cycle anchor is known — the
    # full-cycle basis includes place_design, which the polish never runs,
    # and that overstatement cost the record logicnets run its polish
    # attempt by 42s (need 1236s off full-cycle 489 vs true ~237s worst
    # case, remaining 1194s). Without a route sample the reroute cost is
    # unknowable -> keep the conservative full-cycle basis.
    fanout_cost_anchor_s: float = 0.0
    fanout_anchor_has_route: bool = False
    # CORRECTIVE-SEED LOCAL CLIMB (jul04 preview #5-vs-#6 forensics; default
    # OFF until live-validated on v2). v2's best preview draw (#5, -0.809 =
    # alpha 22.89) came from the recipe-best corrective seed climbing
    # ExtraTimingOpt(-0.856) -> PARTIAL_RUIN(-0.809); its worst (#6, -0.84 =
    # alpha 17.48) had the SAME machinery available but two policies starved
    # it: (a) the corrective seed's baseline is the GLOBAL best, so the
    # -0.856 stepping stone cannot accept once the raw seed hit -0.84 and
    # K=2 futility kills the chain; (b) rotation-continue made it start at
    # the raw seed's cycle count (offset 6 -> PARTIAL_RUIN/SSI on an
    # unprepared state -> -1.2 garbage). When ON: the corrective seed climbs
    # against its OWN seed wns (global keep-best still decides what ships —
    # never-worse unchanged) and its rotation starts at the raw seed's
    # PRISTINE position (rotation index at raw's first accept), which
    # skips exactly the combos that genuinely replayed on the near-identical
    # start state and nothing else. With a zero-accept raw seed this
    # reproduces the old continue-rotation behavior (the spam-filter jun10
    # rationale) by construction.
    # DEFAULT ON since jul05: live-validated 3/3 on v2 debug-wall runs
    # (local baseline honored, pristine-rot offset correct, shipped >=
    # control in every run; the eval-box #5-vs-#6 forensics carry the
    # upside case). Kill switch: --no path via cfg or CLI omission N/A —
    # set False here to disable.
    corrective_local_climb: bool = True
    # Budget reserve for the fanout polish's need-check (jul04, five
    # consecutive observed near-misses: eval #6 logicnets 42s, eval #7
    # logicnets 583s + v2 47s, local logicnets 335s + v2 40s). The old check
    # reused exit_min_remaining_s (600s), a whole-exit-path reserve sized
    # before eager best-valid publishing existed; the mirror now publishes
    # the best DCP EAGERLY during the loop, so post-polish finalize is
    # seconds-to-minutes. 300s is still several times the observed finalize
    # cost; polish steps themselves stay deadline-capped (_to()).
    fanout_finalize_reserve_s: float = 300.0
    # LASTMILE FINAL POLISH (jul06): the jun12 probe evidence (v2 -0.799 ->
    # -0.647 +29MHz; logicnets -0.496 -> -0.437 +15MHz; both 10000-vector
    # equivalence PASSED) is specifically "LASTMILE on the PLATEAUED ship
    # state" — but the ILS rotation only ever runs LASTMILE early in each
    # seed (on -0.91/-0.93-class states); the final global best NEVER gets
    # the probe's condition. Dedicated never-worse post-ILS stage (same
    # pattern as the fanout polish): entry-gated by lastmile_min_wns_ns +
    # budget; accept = strictly-better + fully-routed + hold >= -0.001 (the
    # official-gate floor) + cell-count guard (LASTMILE's -lut_opt shrinks
    # netlists; validator Check-4). Runs BEFORE the fanout polish (which is
    # place-free, so LASTMILE session poisoning cannot affect it).
    lastmile_polish_enabled: bool = True
    # MET-SURPLUS ILS (jul04, hidden-easy-design class): the contest fmax
    # formula rewards POSITIVE slack (fmax = 1/(T - wns), wns > 0 raises
    # fmax above the constraint), but the exit trigger left timing-met
    # designs alone — on an easy hidden design that closes during the recipe
    # (demo_corundum class: closed at +0.082 with wall left), the agent
    # walked away from free alpha. When ON, a met design with leftover
    # budget + viable size arms ILS exactly like an unmet one: keep-best
    # floor = the met DCP (never-worse), same hold gates, same futility
    # stop; LASTMILE (UG906 entry WNS >= -0.25) is the natural winner
    # combo on met states. ZERO effect on the 13 visible benchmarks (all
    # end wns < 0) — pure OOD upside, evidence design = demo_corundum.
    met_surplus_ils: bool = True
    # K3 SPREAD GATE on PARTIAL_RUIN (jul20 whole-history mining, held-out
    # validated — final_round/k3_history_mining_jul20.md rule 2): multi-cell
    # surgery on a spread-diagnosed critical path is strongly positive in the
    # 1,671-episode corpus (134+/5-, mean +0.068ns), but on a CO-LOCATED
    # path (spread ~ 0 tiles) it is 26/26 catastrophic (mean -1.10ns) across
    # 7 fingerprints. PARTIAL_RUIN is the production analog of that surgery
    # (unplace the worst-paths' fabric cells + incremental re-place), so when
    # Phase 1 measured the critical-path avg spread AND it is below this
    # floor, skip PARTIAL_RUIN combos in the rotation. keep-best means a
    # doomed cycle can't corrupt state — the harm is WASTED WALL, so the
    # gate's value is redirecting budget to combo families that can accept.
    # Threshold 30 tiles: well below the 184-300+ range where the surgery
    # evidence is positive (the catastrophic cluster sat at ~0; optical-flow
    # class measures ~15). Spread unknown (None) -> NO gate (fail-open,
    # behavior unchanged). DISTINCT from the rejected WS3b spread-futility
    # idea (that stopped ALL ILS by spread; this gates ONE combo family).
    # Kill switch: partial_ruin_spread_gate=False via CLI
    # --no-partial-ruin-spread-gate or env FPL26_PARTIAL_RUIN_SPREAD_GATE=0.
    partial_ruin_spread_gate: bool = True
    partial_ruin_spread_min_tiles: float = 30.0
    # ROUTE RE-ROLL eligibility (jul21 plateau probe, final_round/plateau_
    # probe_jul21.md + drill_local_queue_jul21.md — fir best state -0.195:
    # +0.070ns ~ +9.5MHz, hold IMPROVING +0.044, 429s local ~ ~170s eval;
    # beta context: +9.5MHz alpha on fir-class ~ column rank 9 -> ~3).
    # The combo fires ONLY when:
    #  (a) the current best is NEAR-MET: wns >= -route_reroll_max_wns_mag.
    #      1.5 covers the evidence class (fir -0.195, logicnets -0.45..-0.55,
    #      vexriscv-class) and EXCLUDES deep-WNS designs, where the
    #      bare-reroute tail loop in dcp_optimizer already owns the
    #      unroute/re-route mechanism (jul21 queue: banked route-first on a
    #      -10.676 state gained only +0.010). Unknown wns -> skip (the gate
    #      requires demonstrated near-met; harm of skipping is zero, the
    #      rest of the rotation is untouched).
    #  (b) the remaining cycle budget fits the FULL route cost: route
    #      sample from the anchors x1.3 (route_reroll_cost_basis). The op
    #      is destructive IN-SESSION (the state is unrouted mid-cycle), but
    #      the banked best DCP on disk is the untouched never-worse mirror
    #      — write_checkpoint happens only on accept — so NO K=2 cost
    #      doubling is needed (contrast: the LLM-path bare-reroute gate
    #      needed a destructive K=2 basis because there the routed state
    #      itself was at risk; here a timeout just discards the cycle).
    # Kill switch: CLI --no-route-reroll or env FPL26_NO_ROUTE_REROLL=1.
    route_reroll_enabled: bool = True
    # Band tightened 1.5 -> 0.7 on the COMPLETED probe row (jul22): the
    # re-roll bit only at shallow WNS (fir +0.070 @ -0.313), was flat at
    # logicnets (-0.588, +0.002) and NEGATIVE at optical (-0.842, -0.026)
    # — the bite decays with |WNS|. 0.7 keeps fir robustly + logicnets as
    # a cheap lottery ticket, excludes the measured loser class.
    route_reroll_max_wns_mag: float = 0.7
    # Recipe-phase route_design SINGLE-stage wall sample (+ caller margin),
    # plumbed from derive_cost_anchors() singles["route_design"]. Basis for
    # the route-reroll affordability gate. 0 = unknown -> fall back to the
    # fanout anchor (when it contains a route sample) then to the scaled
    # combo prior (0.55x full cycle).
    route_cost_anchor_s: float = 0.0
    # Phase-1 RapidWright critical-path avg spread (tiles), plumbed by the
    # caller from critical_path_spread_info["avg_distance"]. None = unmeasured.
    critical_path_avg_spread_tiles: Optional[float] = None
    # UNFENCED wrapper deadline (epoch seconds). The ILS's own deadline_ts is
    # already reduced by the polish reserve, so it is SHORT of the wrapper's
    # budget by exactly that reserve. The re-draw money lives at the wrapper
    # level, so any reserve that protects a re-draw must be compared against
    # THIS, not against the fenced ILS remaining. None = unknown -> the re-draw
    # reserve cannot fire (fail-open).
    wrapper_deadline_ts: Optional[float] = None
    # Phase-1 failing-endpoint count, plumbed by the caller from the same
    # accessor the recipe router uses (_phase1_failing_endpoints_for_features).
    # Paired with the spread above it gives failing-endpoint DENSITY, the
    # ExtraNetDelay_high gate feature. None = unmeasured -> gate fails OPEN.
    phase1_failing_endpoints: Optional[int] = None
    # ---- INSURED TERMINAL RE-PLACE GAMBLE (jul22 placement flagship) ----
    # See optimizer/replace_gamble.py for the full design + doc evidence.
    # DEFAULT OFF: the RC ships it off; the all-in window (Aug 5-9) flips it
    # on (CLI --replace-gamble / env FPL26_REPLACE_GAMBLE=1; belt kill:
    # FPL26_NO_REPLACE_GAMBLE=1). With the flag off the stage is a
    # guaranteed no-op — zero diff in any code path.
    replace_gamble_enabled: bool = False
    # Adopt bar over the CHAIN-BEST (round-2 panel gate, 3/3): +0.15 ns.
    # NOT the 0.002 polish margin — a full re-place discards the chain's
    # accumulated phys_opt polish, so a marginal win is likely noise.
    replace_gamble_adopt_margin_ns: float = 0.15
    # Draw count (round-2: "2-3 draws"; panel r1: "2 draws"). Variants
    # beyond this index in REPLACE_GAMBLE_VARIANTS are never attempted.
    replace_gamble_max_draws: int = 2
    # Terminal finalize reserve added on top of the x1.3-margined full
    # place+route anchor per draw (same 300s convention as
    # fanout_finalize_reserve_s: eager best-valid publishing makes
    # post-stage finalize seconds-to-minutes).
    replace_gamble_finalize_reserve_s: float = 300.0
    # Placement-jump class gate: route-delay fraction of the critical path
    # (round-2 gate-loosening: > 0.6, loosened from 0.7). The feature is
    # plumbed via critical_path_route_delay_frac below; None (Phase 1 does
    # not measure it yet) -> FAIL-OPEN (build-regardless; the structured
    # REPLACE_GAMBLE log lines carry the kill-test evidence instead).
    replace_gamble_min_route_delay_frac: float = 0.6
    critical_path_route_delay_frac: Optional[float] = None

    # ---- DEEP-WNS full-replace sibling (jul25 panel, grok-4.5 seat) ----
    # See optimizer/deep_replace_sibling.py. Distinct from replace_gamble:
    # that stage re-places from the BANKED BEST, this one re-places from the
    # PRISTINE INPUT — which is what boom_soc's +35.47 record actually did
    # (it discarded its scoped pass and reopened the original DCP, giving
    # +35.47 against our +13.32 last night).
    # Produces an insured-compare MUX candidate, so it is never-worse by
    # construction: a loss simply loses the compare. It deliberately does
    # NOT alter any router decision — the jul25 panel rejected outcome-fitted
    # class gating 0/5 on the analogous reserve-v2 charge.
    # DEFAULT OFF until farm-validated. CLI --deep-replace /
    # env FPL26_DEEP_REPLACE=1; belt kill: FPL26_NO_DEEP_REPLACE=1.
    deep_replace_enabled: bool = False
    # RUN THE RECIPE FIRST, before the LLM loop, instead of at the exit tail.
    #
    # WHY (jul25, measured): the recipe needs ~79% of the wall on boom_soc
    # (2750 s of 3500 s), so as a TAIL supplement it can never fire — chain8 on
    # hardware left 72 s at exit and logged deep_replace=0. It is either the
    # main event or it is nothing. On the only two in-band designs measured,
    # the forced recipe beat the agent by +25.18 MHz (boom_soc) and beat our
    # all-time best by +7.66 (boom_soc_v2).
    #
    # Never-worse still holds: a losing recipe leaves best_wns <= initial_wns,
    # _finalize_no_improvement() returns True, and the pipeline ships the
    # baseline byte-identically. What promotion DOES risk is opportunity cost —
    # the LLM loop only gets the remainder. That is the trade the jul25 panel
    # upheld 4/5 for the published DEEP-extreme band specifically.
    #
    # Requires deep_replace_enabled. DEFAULT OFF. CLI --deep-replace-first /
    # env FPL26_DEEP_REPLACE_FIRST=1; belt kill: FPL26_NO_DEEP_REPLACE_FIRST=1.
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
    physopt_skipped: int = 0   # WS1b: cycles whose phys_opt was provably futile
    # Rotation index AFTER the last combo attempted from the seed's PRISTINE
    # (pre-first-accept) state. Valid as a sibling seed's combo_offset: those
    # combos ran on a near-identical start state (deterministic placer =>
    # genuine replays); combos attempted after the first accept ran on a
    # DIVERGED state and are fresh for the sibling. With zero accepts this
    # ends at the final rotation position == the old continue-rotation
    # semantics (spam-filter jun10 case preserved by construction).
    pristine_rot: int = 0
    # Observed per-combo cycle costs (box+design facts) — the caller threads
    # these into a sibling seed's run so it never re-learns them from cold
    # priors (GAP #4, eval #9 v2: fresh dict made the corrective seed skip
    # the 254s ExtraTimingOpt the raw seed had just measured).
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
    """Pure accept decision for the post-ILS AggressiveFanoutOpt polish (jun14).

    Never-worse on setup (strictly beats best by accept_margin_ns) AND fully
    routed AND a STRICT hold floor (fanout replication erodes hold: jun14
    ispd16/vtr fell to whs~0.000, which the contest hold_passed gate would
    reject — so require whs >= fanout_hold_slack_floor_ns, NOT the permissive
    -0.001 combo floor) AND never worsen hold vs the pre-polish best. A reject
    is free (keep-best => never-worse), so the gate errs strict."""
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
    """Pure accept for the post-ILS LASTMILE polish (jul06). Never-worse on
    setup + fully routed + hold >= -0.001 (matches the official scorecard
    gate, jul02 preview evidence) + cell-count sanity band (LASTMILE's
    -lut_opt can shrink the netlist; the validator's hard Check-4 gate was
    removed upstream jul07, PR #41 — band is sanity-only now)."""
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
    """Derive (full_cycle_s, fanout_pass_s, singles) cost anchors from the
    recipe phase's observed heavy-tool durations on THIS design/box.

    Heavy steps arrive either as granular tools (vivado_place_design,
    vivado_phys_opt_design) or embedded in vivado_run_tcl commands —
    classify by tool_name OR cmd_head. A record matching >=2 stage keys is
    one Tcl doing a whole place->route(->phys_opt) sequence: its elapsed is
    a direct full-cycle sample (summing per-key would double-count it).

    full_cycle_s: one Explore full-ruin cycle (place+route). 0.0 when no
      place sample exists — e.g. no-place recipe paths (R7 closure ladder
      is phys_opt-only; R1 route lever routes without placing).
    fanout_pass_s: the fanout polish's worst case (one phys_opt pass + one
      reroute), from SINGLE-stage samples only — derivable on no-place
      paths where full_cycle_s stays 0. 0.0 when neither sample exists.
    singles: the per-stage max durations (for the caller's log line).

    Pure function; no margins applied (caller adds its own headroom)."""
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


# Route-reroll budget margin: AggressiveExplore on an unrouted state costs
# more than the recipe's (typically Default-directive) route sample; the jul21
# fir probe measured 429s vs a ~330s recipe route sample on the same box —
# x1.3 covers the observed overhead with margin.
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

    ALSO (jun12 campaign finding): the MCP server wraps every run_tcl command in
    `catch` and reports a Vivado-side failure as plain output "TCL ERROR: <msg>"
    — NOT a client error envelope. Before this check, an instantly-erroring
    place/route/write_checkpoint passed as success: the v2 LASTMILE accept
    recorded a 40ms write_checkpoint (physically impossible) = PHANTOM accept
    with a stale file on disk. Treat TCL ERROR output as failure."""
    if not isinstance(resp, str):
        return True
    # aug06: the Vivado MCP server reports its OWN failures (pexpect.TIMEOUT,
    # pexpect.EOF, generic except) as a plain-text response beginning
    # "Error: ..." — no JSON envelope and no "TCL ERROR:" marker. A wedged or
    # dead Vivado was therefore read as a successful call here. Prefix match,
    # because the server returns the error as the entire response.
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
    """Run ruin-and-recreate cycles until deadline, seeded from best_dcp_path.
    keep-best: writes back to best_dcp_path only on strict improvement.
    Assumes the design at best_dcp_path is the agent's current best (routed).

    no_improve_stop>0: stop after that many CONSECUTIVE real cycles without a
    MEANINGFUL accept (gain >= cfg.meaningful_accept_ns; micro-accepts are
    kept but count as futile — jul06 #12-vs-#13 gamma forensics). Used by the
    dual-seed primary to yield leftover budget to the corrective probe.
    0 = run to the deadline (the sole/final seed).

    combo_offset: start the directive rotation at this index. The dual-seed
    corrective probe passes the primary seed's cycle count so it CONTINUES the
    rotation instead of repeating it — Vivado's placer is deterministic, and on
    zero-gain stuck designs the two seeds are near-identical DCPs, so restarting
    at combo 0 replays byte-identical failed experiments (spam-filter 2026-06-10
    wasted 2/3 of its corrective budget this way)."""
    r = ILSPolishResult(triggered=True, baseline_wns=baseline_wns)
    best = baseline_wns
    r.best_wns = best
    cyc = 0
    noop_streak = 0      # consecutive cycles that didn't actually execute Vivado
    no_improve_streak = 0  # consecutive REAL cycles without a MEANINGFUL accept
    # WS1a (2026-06-10 wall audit): cycle cost varies 2-4.5x by PLACE DIRECTIVE
    # (logicnets: Explore 298s vs ExtraNetDelay_high 1273s). A single global
    # worst-case estimate let one slow combo block later CHEAP combos — every
    # audited run stopped on "est=<global max>" with affordable winning combos
    # left. Track cost per combo; gate each candidate by ITS OWN estimate;
    # unseen combos borrow the cheapest observed cost (optimistic is safe: the
    # client-side reserve-protected timeout bounds any overrun).
    # GAP #4 (eval #9 v2 forensics): observed per-combo costs are box+design
    # facts — they transfer across seeds in the same stage. Without this the
    # corrective seed restarted from cold priors and SKIPPED ExtraTimingOpt
    # (observed 254s by the raw seed; prior-scaled estimate priced it out),
    # killing the proven chain before its first move.
    combo_cost: dict = dict(combo_cost_seed) if combo_cost_seed else {}
    r.combo_cost = combo_cost   # live reference; mutated per observed cycle
    tried_since_accept: set = set()  # deterministic placer: same combo + same
    #                                  best DCP => same result; never repeat.
    rot = combo_offset
    # SA exploration state: the DCP each cycle STARTS from. Stays best_dcp_path
    # in greedy mode; in explore mode it may point at the last near-miss state.
    work_path = best_dcp_path
    explore_path = best_dcp_path + ".explore.dcp"
    _prev_lastmile = False   # jun12 session-poisoning fix (see in-loop comment)
    _spread_gate_noted = False   # K3 spread gate: one result-note per run
    _rr_budget_noted = False     # route-reroll budget gate: one note per run
    _irf_yield_noted = False     # incr-route priority yielded to an affordable route
    # SPAM-GUARD firing audit (REVIEW_V30 MINOR-5): "armed but never
    # eligible" must be distinguishable from "not reached" in an A/B grep.
    _seg_armed_ever = False      # guard armed on >= 1 decision cycle
    _seg_logged_ever = False     # >= 1 "SPAM-GUARD:" line emitted
    # Rotation actually in use. Identical to ILS_COMBOS unless the jul27
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
    # WALL FENCE (aug02, default OFF): how much of the wrapper window a ladder
    # rung may claim beyond the fenced deadline. Constant for the whole seed —
    # the borrow is an ABSOLUTE ceiling (deadline_ts + _wf_extra), not a
    # per-rung allowance, so consecutive rungs cannot compound it.
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
            # Distinct from 'unknown' (adversarial review MINOR 3): the wrapper
            # is KNOWN and equals the fenced deadline, i.e. the polish reserve
            # was released or is 0 — there is genuinely nothing to borrow, and
            # a firing audit must not read this run as malformed-config.
            log("ILS wall-fence: enabled, wrapper window equals the fenced "
                "deadline (polish reserve released/0) — borrow 0s")
        else:
            log("ILS wall-fence: enabled but wrapper_deadline_ts is unknown — "
                "borrow 0s (fail-closed, shipped behaviour)")
    while cyc < cfg.max_cycles:
        remaining = deadline_ts - time.time()
        _wf_admit = False    # this cycle runs a fence-admitted rung (set below)
        # RE-DRAW RESERVE. Until this ILS has produced a MEANINGFUL accept it has
        # not earned the right to spend the wall that funds a wrapper re-draw.
        # Checked at the loop head, before a cycle is chosen, and priced with the
        # measured basis so the reserve survives the cycle we are about to start
        # rather than being breached by it.
        _wdl = getattr(cfg, "wrapper_deadline_ts", None)
        if (redraw_reserve_enabled() and not _had_meaningful_accept
                and cyc >= REDRAW_MIN_CYCLES_DEFAULT and _wdl):
            _rr = redraw_reserve_s()
            _next_cost = cold_start_basis(cfg, _meas_basis)
            # WRAPPER remaining, not the fenced ILS remaining. Using the latter
            # was the jul31 defect: it is short by the polish reserve and stopped
            # digit one cycle before its win.
            _wrem = _wdl - time.time()
            if _rr > 0 and (_wrem - _next_cost) < _rr:
                remaining = _wrem   # report the budget we actually judged
                log(f"ILS redraw-reserve: no meaningful accept in {cyc} cycles "
                    f"and the next cycle (~{_next_cost:.0f}s) would leave "
                    f"{remaining - _next_cost:.0f}s < the {_rr:.0f}s attempt "
                    f"floor — stopping so the wrapper can re-draw instead. "
                    f"(An unproven stage does not spend a proven alternative's "
                    f"budget; fir jul31 burned 2000s for +0.006ns and starved a "
                    f"re-draw worth ~6 MHz in expectation.)")
                r.notes.append(f"redraw-reserve: stopped at cycle {cyc}, "
                               f"{remaining:.0f}s returned for a re-draw")
                break
        n = len(_COMBOS)
        pick = None
        # INCR-ROUTE PRIORITY: its precondition (an incumbent routed with something
        # weaker than AggressiveExplore) is PERISHABLE — ROUTE_REROLL and ROUTE_ONLY
        # both destroy it by re-routing from scratch aggressively. Take the cycle
        # now, while it is still eligible. Never displaces a forced place-retry rung
        # (that queue is checked next and wins), and the ordinary eligibility +
        # terminal gates below still apply on the normal path.
        #
        # DISPLACEMENT GUARD (jul28, night16). Jumping the queue is only free when
        # the from-scratch route sentinels could not have used this cycle ANYWAY.
        # Measured on one box, same code, traces identical up to the decision:
        #
        #   3d    remaining 441s vs full-route need 376s -> route STILL FITS. The
        #         jump displaced ROUTE_ONLY (362s, -2.027 -> -1.948, +0.079ns) and
        #         paid back a 0.005ns micro-accept; the window shut right after.
        #         Cost -5.96 MHz (+15.93 -> +9.97).
        #   spam  remaining 292s vs full-route need 367s -> route PRICED OUT. The
        #         jump displaced nothing and carried the record step
        #         (-0.598 -> -0.543, +29.19).
        #
        # So the discriminator is not the design, it is whether a full route still
        # fits — the same question, costed the same way, that the ROUTE_REROLL gate
        # below already asks. My earlier note pricing this jump at "ONE cheap cycle"
        # counted only the escalation's own wall; the real cost is whatever the
        # displaced cycle would have returned, and where a from-scratch route can
        # still run the corpus says it wins. Unknown anchor (0.0) = cannot PROVE the
        # route is priced out = no jump, i.e. exactly the pre-jul28 shipped order.
        _fr_need = route_reroll_cost_basis(cfg)
        _route_still_fits = (_fr_need <= 0.0) or (_fr_need < remaining)
        # SPAM-GUARD (aug03, FPL26_SPAM_ESCALATION_GUARD, DEFAULT OFF):
        # displacement-FREE escalation.  When the reroll fits AND the window
        # also funds the escalation on top (remaining >= cost(incr) +
        # reroll_need, checked with the incr cost inside the loop below),
        # fire the S6-eligible jump ahead of the sentinels — the reroll
        # provably still fits afterward, so nothing is displaced.  With the
        # flag off _seg_try is False and both branches below reduce to the
        # jul28 behaviour exactly.  Requires a known reroll basis
        # (_fr_need > 0): an unknown anchor cannot PROVE the fit, fail
        # closed — same principle as the jul28 guard's unknown-anchor rule.
        _seg_try = (spam_escalation_guard_enabled() and _route_still_fits
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
                        # Displacement-free inequality (SPAM_2616 §4): fire
                        # only when the escalation's own cost AND the full
                        # reroll both fit in the remaining window.  Unknown
                        # incr cost -> cannot prove the fit -> no jump
                        # (fail closed; the jul28 yield behaviour stands).
                        if (_ir_est is not None
                                and remaining >= _ir_est + _fr_need):
                            pick = _ir
                            _seg_logged_ever = True
                            log(f"SPAM-GUARD: displacement-free incr-route "
                                f"PRIORITY — incumbent is {_last_rd}-routed, "
                                f"remaining {remaining:.0f}s >= incr "
                                f"{_ir_est:.0f}s + reroll need "
                                f"{_fr_need:.0f}s; the from-scratch route "
                                f"still fits AFTER this cycle, so nothing "
                                f"is displaced (keep-best; S6-eligible).")
                        else:
                            _seg_logged_ever = True
                            log(f"SPAM-GUARD: yielded — cannot prove the "
                                f"reroll still fits after the escalation "
                                f"(remaining {remaining:.0f}s vs incr "
                                f"{'unknown' if _ir_est is None else format(_ir_est, '.0f') + 's'}"
                                f" + reroll need {_fr_need:.0f}s); jul28 "
                                f"yield behaviour stands (3d's -5.96 MHz "
                                f"case is exactly this branch).")
                            # MINOR-5: restore the jul28 one-time yield
                            # note while the guard is armed — the note was
                            # suppressed by `not _seg_try` in the branch
                            # above, but a yield here IS the jul28 outcome.
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
        # Forced pick(s) from the place-retry trigger. Each is consumed whether or
        # not it is usable, so a regression can never wedge the rotation. Unarmed
        # and with the single-rung default the queue holds at most one entry, which
        # is the jul27 behaviour exactly.
        _use_forced = False
        _skip_unaff = ladder_skip_unaffordable_enabled()
        # WALL FENCE: the window a LADDER RUNG may judge itself against. With
        # the flag off (or wrapper unknown) _wf_extra is 0 and this is exactly
        # `remaining`. Recomputed from `remaining` every cycle, so the total borrow
        # is bounded by _wf_extra for the whole seed, not per rung.
        _wf_rem = remaining + _wf_extra
        while _forced_queue:
            _fp = _forced_queue.pop(0)
            # Count POPS, not runs: the queue's first entry is the shipped
            # single-rung slot whether or not it turns out to be affordable. If we
            # counted runs, an unaffordable rung 1 would silently promote rung 2
            # into the unreserved slot — exactly the budget-eating the reserve is
            # there to prevent. This still holds when skip-unaffordable advances
            # within one cycle: each rung examined is one pop.
            _rung_no = _rungs_popped
            _rungs_popped += 1
            if not (_fp < n and _fp not in tried_since_accept):
                # Already tried, or out of range. Unarmed this ends the attempt
                # (shipped behaviour); armed, look at the next rung.
                if _skip_unaff and _forced_queue:
                    continue
                break
            # Density gate on the FORCED rung. Rung 1 of the ladder IS
            # ExtraNetDelay_high (PLACE_RETRY_TARGET), so without this the
            # regression trigger would force precisely the combo the gate just
            # decided this design should not pay for. Treated like an
            # unaffordable rung: consumed, logged, and (when skip is armed)
            # advanced past rather than surrendering the cycle.
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
            # RESERVE (panel 5/5): a rung PAST THE FIRST must leave enough
            # window for one more full cycle afterwards, so probing can never
            # eat the budget the productive rotation would have used. Rung 0 is
            # the shipped behaviour and is never reserve-gated.
            _res = (_fp_basis if (ladder_reserve_enabled() and _rung_no > 0
                                  and _fp_basis > 0) else 0.0)
            # The reserve gate stays on the FENCED remaining (adversarial
            # review MAJOR 2): the follow-up cycle it reserves can only run in
            # the fenced window, so judging it against the borrowed window
            # would reserve a cycle that is structurally unrunnable. Corollary,
            # stated so nobody re-derives it: when ladder-reserve is armed the
            # fence is INERT on rungs>0 (est+res<remaining already implies
            # est+res<_wf_rem, so no admission ever needs the borrow).
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
                # key so a firing audit can grep the treatment, not a proxy
                # (feedback_firing_check_must_key_on_treatment).
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
            # LASTMILE entry gate (jun13 overnight: on 3d-rendering at baseline
            # -2.1ns, `place_design -directive LastMile` failed outright with
            # "Place design failed" — wasting a cycle and forcing a restart).
            # UG906 last-mile entry criteria is WNS >= -0.25; we gate a bit
            # looser (-0.30, since the cycle's phys_opt/route can still close a
            # small extra gap). When the current best is far from closure,
            # mark LASTMILE as tried (so rotation-exhaustion still accounts for
            # it) and skip it; an accept that lifts WNS above the threshold
            # clears tried_since_accept, re-enabling it.
            if (_COMBOS[idx][0] == LASTMILE_PD and best is not None
                    and best < cfg.lastmile_min_wns_ns):
                tried_since_accept.add(idx)
                continue
            # ExtraNetDelay_high DENSITY GATE (jul31). Applied HERE as well as at
            # the forced-rung site because ExtraNetDelay_high lives in the BASE
            # rotation (ILS_COMBOS), not only in the place-retry ladder — gating
            # only the ladder would leave the rotation path wide open, which is
            # how the combo reached digit in the first place. Marked tried (not
            # merely skipped) so rotation-exhaustion accounting stays correct,
            # exactly as the LASTMILE gate above does.
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
            # K3 SPREAD GATE (jul20 mining, held-out rule 2): on a co-located
            # critical path (measured avg spread < partial_ruin_spread_min_
            # tiles) targeted cell surgery is 26/26 negative in the corpus —
            # skip PARTIAL_RUIN combos and give the cycle to a combo family
            # that can accept. Marked tried so rotation-exhaustion accounting
            # still holds; an accept clears tried_since_accept and the gate
            # re-fires (spread is a Phase-1 design constant). None -> no gate.
            # aug05 gray-areas panel (qwen H4b, narrowed): spread=None now
            # FAILS CLOSED for partial-ruin only — the unmeasured case is
            # indistinguishable from the co-located class (26/26 negative,
            # mean −1.10 ns) and the foregone upside is +0.068 ns mean.
            # The endhigh gate's None handling is deliberately UNCHANGED
            # (flipping it is symmetric risk — could block a 3d-class
            # +37.91 high win on the same measurement failure).
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
                    f"(K3 corpus 26/26 negative; None fails closed aug05)")
                continue
            # ROUTE RE-ROLL eligibility gate (jul21 plateau probe: fir
            # +0.070ns/+9.5MHz hold-improving on a near-met best). Fires
            # only NEAR-MET (a) and budget-fit for a full route x1.3 (b) —
            # see the cfg.route_reroll_* evidence block. Composes with the
            # spread gate above: each gate covers its own combo family
            # only, both can fire in the same rotation pass.
            if _COMBOS[idx][0] == ROUTE_REROLL_PD:
                if not cfg.route_reroll_enabled:
                    # Kill switch: account it as tried so rotation
                    # exhaustion still terminates, exactly like a gate skip.
                    tried_since_accept.add(idx)
                    continue
                if best is None or best < -cfg.route_reroll_max_wns_mag:
                    # (a) deep-WNS (or unmeasured) state: the re-roll's
                    # evidence is near-met only; deep-WNS re-routes belong
                    # to the bare-reroute tail loop (jul21 queue: -10.676
                    # state gained just +0.010). tried-marked so an accept
                    # that lifts WNS above the floor re-enables it (the
                    # accept clears tried_since_accept), mirroring the
                    # LASTMILE entry gate.
                    tried_since_accept.add(idx)
                    log(f"[ils] route-reroll skipped: wns={best} below "
                        f"near-met floor -{cfg.route_reroll_max_wns_mag} "
                        f"(probe jul21: evidence is near-met only, fir "
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
            # INCREMENTAL RE-ROUTE eligibility. The operator ESCALATES an existing
            # routing, so it needs an incumbent whose router is known and weaker
            # than this rung's. spam S6 measured the failure directly: run
            # AggressiveExplore FIRST and finish with Explore and the gain vanishes
            # (-0.595 vs -0.543). Both conditions are STATE FACTS about what has
            # already run, not predictions about what would happen.
            if _COMBOS[idx][0] == INCR_ROUTE_PD:
                if _last_rd is None or _last_rd == _COMBOS[idx][1]:
                    tried_since_accept.add(idx)
                    log(f"[ils] incr-route skipped: incumbent router "
                        f"{_last_rd or 'unknown'} is not weaker than "
                        f"{_COMBOS[idx][1]} (S6: escalation order is causal)")
                    continue
                # TERMINAL gate (panel 5/5 gated it; 5/5 also named it the most
                # likely false lever — one design for, one against). Let it run
                # only when the loop has nothing better left: either it is one
                # cycle from the futility stop, or the window can no longer pay
                # for a full-place cycle. Both are state facts. NOT tried-marked
                # on a budget/streak miss — those are not deterministic-replay
                # facts, and the condition can become true later in the run.
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
                # COLD START (jun12 corundum drill finding): with no observed
                # costs the old optimistic estimate (0) always picked combo 0
                # — a full-ruin cycle that on big/slow designs CANNOT complete
                # in the window, burning the whole ILS budget for nothing. If
                # the caller provided an absolute anchor (recipe-phase
                # place+route+phys_opt wall on THIS design/box), scale the
                # per-combo relative priors; an unaffordable Explore then
                # falls through to partial-ruin (0.7x), which may still fit.
                if (_COMBOS[idx][0] == ROUTE_REROLL_PD
                        and route_reroll_cost_basis(cfg) > 0):
                    # Route-reroll: cost it from THIS design's observed
                    # route sample x1.3 (the eligibility gate above used
                    # the same basis, so pass/fail here is consistent).
                    est = route_reroll_cost_basis(cfg)
                elif (_COMBOS[idx][0] == ROUTE_ONLY_PD
                        and cfg.fanout_cost_anchor_s > 0
                        and cfg.fanout_anchor_has_route):
                    # ROUTE_ONLY never places: cost it from THIS design's
                    # observed route+phys_opt samples (the fanout anchor),
                    # not the place-dominated full-cycle prior. jul04 local
                    # v2 leg: 0.6*1790s=1074s "unaffordable" vs true ~250s
                    # (place was 1568s of the anchor) -> ILS ran 0 cycles
                    # and shipped BASELINE with 875s unused, on a design
                    # where ROUTE_ONLY accepted in BOTH preview #5 and #6.
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
        # ANY forced probe is exempt from futility — not just ladder rungs.
        #
        # MEASURED REGRESSION that forced this (jul28, optical arm=mb, single-rung):
        #   cycle 1 Explore              -1.162  (best -0.924)  no accept -> streak 1
        #   cycle 2 ExtraNetDelay_high   -0.971  (forced probe)  no accept -> streak 2
        #   ILS: no improvement in 2 real cycles — stopping (yield budget)
        # shipped alpha +17.11, against +25.13 for the SAME design with the probe
        # disarmed — a -8.02 MHz REGRESSION caused by my own mechanism. In the
        # control the second cycle was ROUTE_ONLY, which ACCEPTED at -0.848.
        #
        # So the forced probe did not merely waste a cycle, it spent the design's
        # last futility strike and ended the search before the rotation could reach
        # the combo that works. That is exactly the harm 1af5c85's commit message
        # described, and the exemption was gated behind the ladder flag so the
        # single-rung path — the one that ships — never got the protection.
        # A directed probe is not a random restart in either mode.
        _futility_exempt = _use_forced
        # SESSION-POISONING FIX (repro'd jun12 in a clean batch session, so
        # AWS is affected too): after `place_design -directive LastMile` the
        # Vivado SESSION can no longer run a full placement — even on a
        # freshly opened checkpoint, plain/directive place_design fails with
        # "Place design failed" (only incremental completion of a small
        # unplaced set still works). Restart Vivado before the next cycle
        # whenever the previous cycle ran LASTMILE. Best-effort: on restart
        # failure the TCL-ERROR guard safely skips poisoned cycles anyway.
        if _prev_lastmile:
            _prev_lastmile = False
            try:
                _rr = await call_tool("vivado_restart_vivado", {})
                if _tool_ok(_rr):
                    log("ILS: Vivado restarted after LASTMILE cycle "
                        "(full-place session poisoning, jun12 repro)")
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
        # Per-command timeout: generous (large designs need >300s/step) but capped
        # at the remaining budget so no single step overruns the wall. Floored at
        # 300s (the old default) so we never time out faster than before.
        # WALL FENCE: a fence-admitted rung was judged against the wrapper
        # window, so its commands must be allowed to run there too — otherwise
        # the fenced cap kills at ~300s the very rung the fence just admitted,
        # burning the cycle AND the borrow.
        # ⚠️ ON ADMITTED CYCLES THE FLOORS ARE CLAMPED, NOT APPLIED (adversarial
        # review MAJOR 1): max(300, ...) on a step starting near the ceiling
        # would end past wrapper_deadline_ts — reaching the outer asyncio
        # wrapper-cancel kill (_budget_killed=True, "session may be
        # compromised"), a path the fenced stack can never reach because its
        # MCP kill lands >=200s before the wrapper cancel. Clamping every
        # admitted-cycle timeout to end >=30s BEFORE the ceiling keeps the
        # clean MCP-side kill the only kill, at the cost of a fast clean abort
        # for a step started hopelessly late. Non-admitted cycles keep the
        # shipped arithmetic byte for byte.
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
                # chars are usually license/INFO preamble (jun12: the real
                # error text was truncated away during the LASTMILE forensics).
                _txt = str(resp)
                _eline = next((l for l in _txt.splitlines() if "TCL ERROR" in l), "")
                raise RuntimeError(
                    f"{cmd.split()[0]} failed: {(_eline or _txt)[:400]}")
        try:
            await _step(f"open_checkpoint {{{work_path}}}", _light_to())
            if pd == LASTMILE_PD:
                # UG906 IDR Stage-3 ("last mile") recipe on the current routed
                # state: netlist-level phys_opt (clock_opt/retime/lut — all
                # contest-legal, functional equivalence VALIDATED 10000-vector
                # jun12), incremental LastMile re-place, pre-route phys_opt,
                # route. Probe jun12: v2 -0.799 -> -0.647 (+0.152ns, ~+29 MHz,
                # validator PASS) in 607s; spam failed routing (caught — the
                # accept gate absorbs it). The final post-route phys_opt comes
                # from the standard block below (od).
                await _step("phys_opt_design -clock_opt -retime -lut_opt", _heavy_to())
                await _step("place_design -directive LastMile", _heavy_to())
                await _step("phys_opt_design -directive Explore", _heavy_to())
                await _step(f"route_design -directive {rd}", _heavy_to())
            elif pd == ROUTE_REROLL_PD:
                # ROUTE RE-ROLL (jul21 plateau probe): full unroute + a
                # from-scratch AggressiveExplore solve — the probe's exact
                # two-op protocol (fir +0.070ns/+9.5MHz, hold IMPROVING).
                # NO phys_opt tail (the standard block below is skipped for
                # this sentinel): the evidence and the x1.3 route-cost
                # basis both cover exactly these two ops. The unroute is
                # in-session only; the banked best DCP on disk is the
                # never-worse mirror (written only on accept).
                await _step("route_design -unroute", _light_to())
                await _step(f"route_design -directive {rd}", _heavy_to())
            elif pd == ROUTE_ONLY_PD:
                # Route-only (jun15 DRILL H): keep the placement, re-route with
                # a stronger directive. No place step -> cheapest cycle; the
                # post-route phys_opt comes from the standard block below (od).
                await _step("route_design -unroute", _light_to())
                await _step(f"route_design -directive {rd}", _heavy_to())
            elif pd == INCR_ROUTE_PD:
                # INCREMENTAL ESCALATED RE-ROUTE (jul28): ROUTE_ONLY minus the
                # unroute. The escalated directive REFINES the incumbent routing
                # rather than re-solving it — spam S3 -0.598 -> -0.543 (+11.05
                # MHz). Deliberately NOT preceded by `route_design -unroute`:
                # with the unroute this becomes S2/ROUTE_ONLY, which measured
                # crumbs (+0.003). The post-route phys_opt comes from the
                # standard block below (od), matching the probe's polish tail.
                await _step(f"route_design -directive {rd}", _heavy_to())
            elif pd == PARTIAL_RUIN_PD:
                # Targeted ruin: unplace ONLY the worst-paths' fabric cells,
                # then incremental re-place (no directive — places the
                # unplaced cells around the preserved majority placement).
                # Scope resolved at fire time (env FPL26_PARTIAL_RUIN_SCOPE,
                # default 200 = corundum sweep optimum).
                await _step(partial_ruin_tcl(partial_ruin_scope()),
                            _light_to())
                await _step("place_design", _heavy_to())
                await _step(f"route_design -directive {rd}", _heavy_to())
            else:
                await _step("place_design -unplace", _heavy_to())
                await _step(f"place_design -directive {pd}", _heavy_to())
                await _step(f"route_design -directive {rd}", _heavy_to())
            # WS1b: measure BEFORE phys_opt. Unrouted nets can never be accepted
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
                # ops of the jul21 probe.
                await _step(f"phys_opt_design -directive {od}", _heavy_to())
                w, ur = await _measure(call_tool, wns_tcl, timeout_s=_light_to())
            elif _skip is not None:
                r.physopt_skipped += 1
                log(f"ILS cycle {cyc}: phys_opt SKIPPED — {_skip}")
            cycle_dt = time.time() - _t0
            tried_since_accept.add(pick)
            combo_cost[pick] = max(combo_cost.get(pick, 0.0), cycle_dt * 1.15)
            # MEASURED COST BASIS (jul28): a completed FULL-PLACE cycle prices
            # every other full-place combo better than the recipe-derived anchor
            # can. Normalise by this directive's own prior so the basis is always
            # in Explore=1.0 units; MAX keeps it conservative across cycles.
            # Recorded unconditionally (cheap, and it makes the OFF arm's logs
            # comparable); only cold_start_basis() consumes it, and only armed.
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
            # RUNAWAY GUARD: a real cycle (place+route+phys_opt) takes minutes. If
            # it returned in seconds OR couldn't measure WNS, the budget-aware
            # dispatcher skipped the (risky) commands — no real work happened.
            # Bail after 2 such no-ops instead of spinning to the deadline.
            if w is None or cycle_dt < cfg.min_cycle_seconds:
                noop_streak += 1
                if noop_streak >= 2:
                    r.notes.append("commands not executing (budget-gated); stopped")
                    log("ILS: cycles not executing (budget-gated/skip) — stopping")
                    break
                continue
            noop_streak = 0
            # ---- PLACE-RETRY (jul27, opt-in FPL26_ILS_PLACE_RETRY) ----
            # A full-place cycle returning WORSE than the incumbent is a MEASURED
            # signal that this placement family is wrong for this design. It is not
            # a prediction, so it may direct SCHEDULING -- the same house rule the
            # LASTMILE entry gate above already follows (a gate keyed on observed
            # WNS, not on fitted features).
            #
            # The jump is necessary, not cosmetic: futility (no_improve_stop,
            # default 2) ends the search within ~2 non-improving cycles while the
            # extension sits at the END of the rotation, so appending alone leaves
            # it unreachable on exactly the designs that need it.
            #
            # Fires at most once per seed, and only after a REAL placement cycle --
            # the sentinels re-use an existing placement, so "the placement family
            # is wrong" does not apply to them.
            # BASELINE GATE: when armed and the pristine baseline is known, ALSO
            # require the cycle to be worse than the untouched input — "this
            # placement family is wrong", not "this cycle trails a polished
            # incumbent". Narrowing only; an unknown baseline leaves the trigger
            # exactly as it is without the gate.
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
                # RUNGS. Single-rung (jul27) = [PLACE_RETRY_TARGET]. The ladder
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
            # ---- MICRO-ACCEPT SEED POISONING (jul29). DEFAULT OFF. ----------
            #
            # An accept does not only bank a result — it REPLACES THE STATE later
            # cycles run from. The ladder's own docstring already states the
            # property: "the rung that runs FIRST runs from the UNMODIFIED seed;
            # every later rung runs from whatever state has accepted by then."
            #
            # fir_systolic, jul29, is a clean natural experiment: two runs of the
            # SAME configuration, bit-identical for six cycles, differing only in
            # the recipe baseline they started from (-0.216 vs -0.247, i.e. 0.031 ns
            # of ordinary noise).
            #
            #   cycle  combo              night16c           shipref
            #   1-5    Explore..LASTMILE  identical          identical
            #   6      __ROUTE_REROLL__   -0.243 no accept   -0.243 ACCEPT (gain 0.0040)
            #   8      ExtraTimingOpt     -0.154  <-- WIN    -0.327  <-- ruined
            #   final                     -0.154 (+21.30)    -0.243 (+9.07)
            #
            # In night16c the better baseline meant cycle 6 was not an improvement,
            # the seed stayed clean, and ExtraTimingOpt — fir's winning combo FROM A
            # CLEAN SEED — reached -0.154. In shipref the identical cycle 6 cleared
            # a worse baseline by 0.0040 ns, was accepted, and ExtraTimingOpt then
            # ran from the modified state and produced -0.327.
            #
            # A 0.0040 ns micro-accept cost 12.23 MHz — priced at 1.5-2.6 mean-rank
            # points, the largest single item on the board.
            #
            # THIS REFRAMES THE jul06 FINDING behind `meaningful_accept_ns`: corpus
            # mining showed micro-accepts (<0.010 ns) are TERMINAL 4/4, never once
            # followed by a meaningful accept in the same seed, and the response was
            # to stop early and save gamma. fir suggests the causal arrow may point
            # the other way — micro-accepts may be terminal BECAUSE they poison the
            # seed for the combos that would have paid.
            #
            # RE-MINED jul29 over both parity boxes — five micro-accepts, and the
            # count is 4/5 TERMINAL, not 4/4:
            #
            #   fir/shipdef_a       0.0040  terminal; ExtraTimingOpt -0.327
            #                               (twin run, clean seed: -0.154)
            #   logicnets/banded    0.0080  terminal; ExtraTimingOpt -0.716 vs
            #                               incumbent -0.505
            #   logicnets/uni3b     0.0050  terminal
            #   3d/uni              0.0050  terminal
            #   digit/rerun         0.0060  MEANINGFUL ACCEPT FOLLOWED
            #                               (-0.695 -> -0.661, gain 0.034)
            #
            # digit is a straight counter-example to universality: a micro-accept
            # need not be fatal. Suggestive for the seed story is that the two runs
            # where ExtraTimingOpt ran right after a micro-accept BOTH collapsed
            # (-0.327, -0.716), while fir's twin with a clean seed won at -0.154 —
            # but only fir has the counterfactual, so that is one clean natural
            # experiment plus a pattern, not a demonstrated cause.
            #
            # Intervention: refuse an accept whose gain is below
            # `meaningful_accept_ns`, keeping the seed clean for later combos. The
            # trade is asymmetric but NOT free: the worst case forgoes up to
            # 0.010 ns, which on fir (T=2.5) is **1.32 MHz**, against the 12.23 MHz
            # the clean seed was worth — about 9:1. (An earlier draft of this
            # comment said 0.05 MHz; that was wrong by 26x and a test caught it.
            # 1.32 MHz on a column worth 0.128 mean-rank points per MHz is a real
            # cost, not a rounding error, and it is why the A/B must measure the
            # LOSS case as well as the win.)
            #
            # DEFAULT OFF: one natural experiment on ONE design, changing the ILS
            # accept rule, which is the core of the optimizer. It must not ship on a
            # mechanism story alone, however well the trace reads.
            # Arm with FPL26_ILS_REJECT_MICRO_ACCEPT=1 for the A/B.
            if (_accept and reject_micro_accept_enabled()
                    and best is not None and cfg.meaningful_accept_ns > 0):
                _micro_gain = w - best
                if _micro_gain < cfg.meaningful_accept_ns:
                    _accept = False
                    r.notes.append(
                        f"cycle {cyc}: micro-accept gain {_micro_gain:.4f} < "
                        f"meaningful {cfg.meaningful_accept_ns} — REJECTED to "
                        f"keep the seed clean (jul29 fir)")
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
            # Cell-count sanity band (jul03 GitHub #36; relaxed jul13 after
            # upstream PR #41 removed the validator's hard Check-4 gate —
            # counts are info-only now): only LASTMILE transforms the netlist
            # (-lut_opt). Fail-open on a missing golden count or unparsable
            # measurement.
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
                # Persist the improved design. The /mnt/c WSL mount drops a
                # write intermittently (jun13 repro: run-dir paths write fine,
                # but the v2 overnight cycle lost a real -0.912 accept to one
                # transient failure; not LASTMILE/session-state — confirmed by
                # pre==post-LastMile repro). Retry once before discarding; the
                # discard path stays the never-worse backstop if both fail.
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
                    # MEANINGFUL accept: this ILS has now earned the wall. The
                    # re-draw reserve stops applying for the rest of the
                    # invocation -- digit's win arrives on cycle 4 and must be
                    # free to spend everything after it.
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
                    # A ladder rung is a DIRECTED PROBE of a specific placement
                    # family, not a random restart, and keep-best means a losing
                    # rung costs one cycle and nothing else. Charging it to the
                    # futility counter is what forced 1af5c85 to ship a single
                    # rung: a probe that regressed ended the search before the
                    # next rung could run. Bounded by construction — at most
                    # len(PLACE_RETRY_LADDER) cycles are ever exempt per seed.
                    log(f"ILS cycle {cyc}: ladder rung (forced probe) — futility "
                        f"streak held at {no_improve_streak}")
                else:
                    no_improve_streak += 1
                if no_improve_stop and no_improve_streak >= no_improve_stop:
                    # HURDLE OVERRIDE (jul26, DEFAULT OFF via
                    # hurdle_continue_alpha_mhz=0). The K-counter is a fixed
                    # constant; the SCORING FUNCTION says whether one more
                    # cycle pays. Measured on mini-ISP chain29: ILS held 2984 s
                    # of budget, spent 233 s, then stopped on K=2 — while the
                    # jul07 record reached 413.22 on its cycle-2 __LASTMILE__
                    # (-0.882 -> -0.850, +5.4 MHz) after FOUR cycles.
                    #
                    #   hurdle = alpha * 0.1 * (dt/3600) / P
                    # one ~120 s cycle at alpha 97.08 => ~0.34 MHz. The record's
                    # winning cycle yielded 5.4 MHz — sixteen times the hurdle.
                    # K=2 stops an order of magnitude before the economics do.
                    #
                    # No fitted constant: the hurdle comes from the contest's
                    # own formula and a MEASURED cycle cost. Bounded by design —
                    # it only ever authorises ONE more cycle at a time, and every
                    # other stop (deadline, budget, cycle cap) still applies.
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
        # MINOR-5 (REVIEW_V30): the guard was armed on >= 1 cycle but no
        # S6-eligible __INCR_ROUTE__ combo ever materialized (all tried
        # since accept / forced queue busy / incumbent already AE-routed).
        # Without this line an A/B firing grep would read silence as a
        # null (feedback_firing_check_must_key_on_treatment).  Flag-ON
        # only: _seg_armed_ever can never be True with the flag off.
        log("SPAM-GUARD: armed, no eligible site (no S6-eligible "
            "incr-route combo materialized this run)")
    if _prev_lastmile:
        # Don't hand a LASTMILE-poisoned session to whatever runs next (the
        # corrective seed, finalize, winner-polish): full placement is broken
        # in this session until restart (jun12 repro).
        try:
            await call_tool("vivado_restart_vivado", {})
            log("ILS: Vivado restarted on exit (last cycle was LASTMILE).")
        except Exception as e:
            log(f"ILS: exit restart failed ({e!r}); downstream guards apply.")
    return r
