"""DEEP-WNS full-replace sibling — a never-worse MUX candidate producer.

WHY THIS EXISTS (jul25 panel, grok-4.5 seat, the only Scale-1 killer; E[alpha
hidden] +10, the highest number on the board):

On boom_soc our validated all-time best is **alpha +35.47** (48.24 -> 83.70 MHz,
WNS -10.378, post-route confirmed, routed DCP written --
``live_tests_archive/boom_soc_cap55/run.log``).  Last night we scored **+13.32**.
The gap is **-22.15 MHz**, the largest in the portfolio.

It is NOT a missing mechanism.  The router sends boom_soc to R1 with the
DEEP-extreme rationale *"failing_endpoints=217,988 (>= 100,000), |WNS|=19.16 ns
(>= 10.0 -- DEEP-extreme sub-class: retiming is the slow step on this)"* and then
executes **one** ``place_design`` and **zero** ``place_design -unplace``.  The run
that won +35.47 did the opposite: it made a scoped/route-first pass, **threw it
away, reopened the pristine input DCP, and did a full re-placement**:

    open_checkpoint <ORIGINAL benchmark dcp>     <- discard the scoped pass
    place_design -unplace
    place_design    -directive Explore
    route_design
    phys_opt_design -directive AlternateFlowWithRetiming

So the mechanism exists in-tree (Class-G / R4 place-Explore + retiming); it is
merely **unreachable** for that feature band.

WHY NOT JUST WIDEN R1's DECISION BOUNDARY:
The jul25 panel rejected outcome-fitted class gating **0/5** on the exactly
analogous reserve-v2 charge -- gating on two observed outcomes is precisely the
boundary-fitting the standing methodology invariant forbids (n~16 designs vs ~20
thresholds = memorisation).  Two seats put "size-gating" on their kill list.
So this module does NOT change any routing decision.  It **adds a candidate** to
the existing insured-compare MUX, which picks the better of the two by measured
WNS.  That is never-worse by construction: if the sibling is worse it simply
loses the compare, and if it cannot be afforded it never starts.

RELATIONSHIP TO ``replace_gamble.py``:
That stage re-places from the **banked best** (``best_dcp_path`` -> read-only
``open_checkpoint``).  This one re-places from the **pristine input**, which is
the actual distinguishing feature of the boom_soc winner -- it deliberately
discards accumulated pipeline state rather than refining it.  Different seed,
different mechanism, so this is a sibling stage rather than a variant.

FREE PARAMETERS: two (``enabled``, ``finalize_reserve_s``).  The physics gate
REUSES the router's existing published constants ``R1_FAILING_ENDPOINTS_MIN``
and ``R1_ROUTE_FIRST_WNS_NS`` -- no new cut is fitted here.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

# Telemetry only — emit() never raises and is a no-op unless FPL26_GATE_LOG
# is set, so it can never alter a decision. See optimizer/gate_log.py.
from optimizer import gate_log as _gl
from typing import Awaitable, Callable, Optional, Tuple

CallTool = Callable[[str, dict], Awaitable[str]]

# The exact winning sequence from live_tests_archive/boom_soc_cap55/run.log.
# Directives are the logged ones; they are NOT tuned here.
DEEP_REPLACE_PLACE_DIRECTIVE = "Explore"
DEEP_REPLACE_PHYSOPT_DIRECTIVE = "AlternateFlowWithRetiming"

# Affordability margin on the measured cost anchor. Same constant and meaning
# as replace_gamble's, kept separate only so the two stages can be reasoned
# about independently.
DEEP_REPLACE_COST_MARGIN = 1.3

# Post-route phys_opt allowance, as a fraction of the MEASURED place+route
# time — used by the second gate (deep_replace_physopt_affordable), never to
# pre-size the arm decision.
#
# Measured twice, on two boxes, on the same recipe:
#   boom_soc record (live_tests_archive/boom_soc_cap55/run.log, 2 core):
#     place Explore 33 min + route 24 min = 57 min; phys_opt 39 min -> 0.68
#   jul25 8-vCPU replay (~/drill/recipe_replay_jul25/boom_soc_replay.log):
#     unplace 33 s + place 865 s + route 746 s = 1644 s; phys_opt 1106 s -> 0.673
# Rounded 0.68 -> 0.7 (toward caution: over-estimating cost only ever declines
# a phys_opt extension, which is the safe direction).
#
# WHY THIS IS NO LONGER PART OF THE ARM GATE (jul25):
# It used to multiply the arm requirement (basis x1.7 x1.3). Combined with a
# basis that is itself x1.15-margined at record time (ils_polish.py combo_cost)
# that demanded ~2.54x the true place+route cost: boom_soc's real 2750 s recipe
# had to clear ~4176 s + reserve against a 3500 s wall, so the stage could not
# arm even at t=0 with the whole budget free. The recipe was never too
# expensive; the estimate was. Margins do not compose — each one is only sound
# against the uncertainty it was calibrated for.
#
# This is a COEFFICIENT calibrated from measured runtimes, not a decision
# boundary fitted to an outcome — the distinction the standing methodology
# invariant draws.
DEEP_REPLACE_PHYSOPT_FRAC = 0.7

# Checkpoint write cost, charged to the phys_opt gate's reserve.
#
# jul25 panel, gemini seat (the lone C2 dissent) argued the B1/B2 split is not
# never-worse because "if B2 regresses we must reload the B1 DCP from disk, and
# if we hit the wall during the reload the run scores 0". kimi (upholding)
# named the same mechanism from the other side: the tail can squeeze validation
# at the wall boundary.
#
# THE RELOAD DOES NOT EXIST in this implementation, and that is deliberate:
# B1's checkpoint is written BEFORE B2 starts. If B2 regresses or throws, we
# simply DO NOT overwrite that file — the banked candidate is already on disk
# and the caller registers it by PATH (register_final_candidate(verify=False)),
# never from the live session. So the only I/O the gate must fund is the write
# itself, which happens while there is still budget, plus one possible
# overwrite if B2 wins.
#
# Measured: write_checkpoint on the 141 MB boom_soc DCP, 8 vCPU, elapsed 15 s
# (~/drill/recipe_replay_jul25/boom_soc_replay.log). Doubled to 30 s for the
# overwrite case and to stay conservative on larger designs. Over-reserving
# only ever declines the tail, which is the safe direction.
DEEP_REPLACE_WRITE_IO_S = 30.0

# B2 fresh-session gate (jul31). B2 ran `phys_opt AlternateFlowWithRetiming` in
# the same Vivado process that had just unplaced, re-placed and routed the
# design, so peak memory was B1's placer+router state plus a retiming pass, on a
# 31 GB swapless box. boom_soc died there three times in one night, the last
# alone on an idle box after reproducing B1's wns=-10.256 exactly.
#
# MIN_PR_S: only designs whose MEASURED place+route is this expensive are at
# risk, and only they pay the reopen. The threshold is not guessed — it sits in
# a MEASURED GAP. Every deep-replace B1 / published place+route anchor in the
# corpus, ~150 observations across both boxes:
#
#   boom_soc          1507 .. 1525   (n=5)    <- fires
#   boom_soc_v2       1398 .. 1528   (n=10)   <- fires
#   ---------------------------------------- 1200 s gate
#   corescore                  539   (n=1)    <- highest NON-boom in the corpus
#   vtr_mcml_v2        401 ..  415   (n=9)
#   vtr_mcml           397 ..  399   (n=6)
#   finn               375 ..  378   (n=6)
#   all 11 others     <=      239
#
# So the boom cluster is separated from the next design by 859 s — any cut in
# (540, 1398) is equivalent, and 1200 is placed nearer the boom end so the
# error direction is "fails to fire" rather than "fires on a design that never
# needed it". The 15 designs that completed the jul31 sweep are ALL excluded,
# by a factor of 2.2x or more.
# REOPEN_S: budgeted cost of restart + open_checkpoint on a design that large;
# if it does not fit, B2 proceeds in the current session exactly as before.
DEEP_REPLACE_B2_RESTART_MIN_PR_S = 1200.0
DEEP_REPLACE_B2_REOPEN_S = 120.0


def b2_restart_min_pr_s() -> float:
    """Threshold, env-overridable (FPL26_DEEP_REPLACE_B2_RESTART_MIN_PR_S).

    Overridable so it can be re-tuned from a farm result without a code change,
    and so tests can drive BOTH sides of the gate — a threshold that only ever
    takes one branch in the suite is untested, not proven.
    """
    raw = os.environ.get("FPL26_DEEP_REPLACE_B2_RESTART_MIN_PR_S", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return DEEP_REPLACE_B2_RESTART_MIN_PR_S

# Place+route seconds per routable net, 8-vCPU eval parity.
#
# WHY THIS EXISTS: as a TAIL stage the recipe could size itself from this run's
# own observed ILS cycle. Run FIRST there is no such anchor — no heavy step has
# executed yet — and the affordability gate fails closed on cost_basis_s <= 0.
# A size-based prediction is the only thing available at t=0.
#
# Measured, both on the Dev Cloud 8-vCPU box, jul25:
#   boom_soc     273,573 nets -> 1644 s  => 0.006009 s/net
#   boom_soc_v2  273,764 nets -> 1492 s  => 0.005450 s/net
# (~/drill/recipe_replay_jul25/boom_soc_replay.log and
#  ~/drill/boomv2_transfer_jul25/armA_deepreplace.log, RPL_STAGE timestamps.)
#
# We take the MAX observed per-net cost, not the mean: over-estimating only
# ever declines a run, which is the safe direction, and the two samples differ
# by 9% at essentially identical size — that spread IS the run-to-run variance.
#
# ⚠ HONEST LIMIT: both samples sit at ~273.7k nets, so they pin the coefficient
# but do NOT constrain the slope's shape away from that size. Extrapolating to
# a much smaller design is unvalidated until the vtr_mcml probe (70,431 nets,
# predicted 423 s) reports. Treat a prediction far from ~273k as provisional,
# and note the x1.3 margin in the arm gate sits on top of this.
#
# This is a cost model fitted to measured RUNTIMES — explicitly permitted by
# the methodology invariant, which forbids only fitting decision boundaries to
# outcomes. One free parameter.
DEEP_REPLACE_PR_S_PER_NET = 0.00601   # rounded UP from 0.0060094 (boom_soc)

# Same model keyed on PRIMITIVE CELL COUNT.
#
# WHY BOTH: routable_nets is what the replay logs report, but the optimizer
# does NOT have it at t=0 — Phase 1 records `_input_cell_count` (primitive
# cells) and `lut_count`, and never a net count. A cost model the caller cannot
# feed is a model that fails closed forever, which is how deep_replace ended up
# at 0 in the first place. So the cell-keyed form is the one recipe-first
# actually uses; the net-keyed form stays for reasoning against replay logs.
#
# Calibrated on the same two runs, using PLACED_PRIMITIVES from the jul25 gate
# sweep (the same extractor validate_ship uses, so the units match what
# Phase 1 measures):
#   boom_soc     377,972 cells -> 1644 s  => 0.0043495 s/cell
#   boom_soc_v2  378,372 cells -> 1492 s  => 0.0039432 s/cell
# Max, rounded up, for the same reason as above.
#
# ⚠ THE TWO FORMS DISAGREE OUT OF SAMPLE, and that is informative rather than a
# defect: for vtr_mcml the net-keyed form predicts 423 s and the cell-keyed
# form 339 s, because the cell:net ratio is design-specific (vtr 78,033 cells /
# 70,431 nets = 1.11; boom 377,972 / 273,573 = 1.38). The queued chain10 probe
# measures vtr_mcml's true place+route and settles which feature generalizes.
# Until then the caller takes the MAX of whichever forms are available.
DEEP_REPLACE_PR_S_PER_CELL = 0.004350

# Minimum useful B2 slice, as a fraction of the MEASURED place+route time.
#
# WHY A FLOOR RATHER THAN A PREDICTION (jul25, n=3 on hardware): the old B2 gate
# predicted the tail's cost as measured_place_route x PHYSOPT_FRAC x COST_MARGIN
# and refused if that did not fit. It was wrong every time we have data for --
# the tail would have COMPLETED in all three:
#
#   run                B2 "need"   had     actual phys_opt   would have fit by
#   chain12 boom_soc     1724 s   1481 s        1106 s            +375 s
#   chain15 att.1        1702 s   1595 s         805 s            +460 s
#   chain15 att.2        1702 s   1148 s         805 s             +13 s
#
# ~1.5x over-estimate, costing ~6.5 MHz on the SCORED boom_soc_v2 benchmark
# (+7.56 shipped vs +14.06 with the tail).
#
# The deeper error was structural, not numeric. B1 is measured, gated and
# WRITTEN before B2 starts, and nothing reloads it -- so B2 cannot cost the
# banked gain, only wall. A predictive gate was therefore solving a problem the
# staging had already removed. Tuning COST_MARGIN would have papered over that.
#
# So B2 now runs whenever there is a slice worth starting, bounded by a deadline
# that protects the finalize reserve. The floor is calibrated from measured
# phys_opt/place+route ratios -- 0.673 (boom_soc), 0.539 (boom_soc_v2), 0.227
# (vtr_mcml), 0.206 (vtr_mcml_v2). Set BELOW the smallest of those, so we never
# decline a slice in which an observed pass has actually completed -- a 0.25
# floor would have refused vtr_mcml_v2's 88 s tail on a 427 s place+route,
# repeating the very error this replaces. Because B2 cannot cost the banked
# gain, the safe rounding direction here is DOWN (permissive), the opposite of
# the arm gate's. One parameter, fitted to measured RUNTIMES, which the
# methodology invariant permits; it is not a boundary fitted to outcomes.
DEEP_REPLACE_B2_MIN_SLICE_FRAC = 0.20

# ---------------------------------------------------------------------------
# B3 "small-floor" leg (aug09) — third sibling candidate, from the PRISTINE
# input:  place ExtraNetDelay_low -> phys_opt AggressiveExplore ->
#         route AggressiveExplore
#
# WHY: on mini-ISP the shipped -0.850 (alpha 106.10) is a hard logic floor
# (V42_EVIDENCE aug06 §path-anatomy) that the ILS reaches only via a 2-accept
# draw (2/21 cycles; the eval previews shipped the -0.904/-0.870 misses:
# P3/P4 = 102.71, the aug09 leaderboard row = 97.08-class). The mg5/od4 drills
# reach the SAME -0.850 deterministically in 149-160 s from pristine
# (RETIME_TRANSFER_DRILLS_aug05, PREREG_V42_RECUT_aug06: n=4, one box class —
# cross-box determinism checked in the v5.3 gate). B3 turns that draw into a
# banked MUX candidate.
#
# WHY THE COST CAP + PHYSICS ADMISSION (the generalization contract): the od4
# chain as a MID-BAND recipe replacement was KILLED (MIDBAND_EGG_KILLED_aug07,
# net -8.95) because the bare band admits corescore/finn/vex1, where the
# chain's pre-loop wall starves the LLM loop. Two gates keep that from
# recurring (v5.5.3): (1) AFFORDABILITY — est from B1's just-measured
# place+route must fit DEEP_REPLACE_B3_MAX_COST_S (550 s => pr <= ~209 s);
# on dev measurements this admits mini-ISP (75-78 s), vexriscv (92-97 s) and
# optical (~183 s) while 3d (260 s), finn (461 s), corescore (~640 s)
# decline by arithmetic; (2) PHYSICS ADMISSION (the b3_admission hook) —
# only a B1 solve whose worst paths are hard-macro-dominated at a near-floor
# bound admits, which refuses vex1/optical-class fabric on any hardware.
# A hidden tiny floor-bound mid-band design gets the same insurance
# candidate; fabric-bound or big ones never pay. Like B2, B3 runs AFTER the
# banked candidate is on disk, so it can only ever cost its own bounded
# slice — never a result.
#
# COST MODEL: od4 measured 149-160 s total against B1 place+route 75-76 s on
# the same box  =>  ~2.05x. Rounded UP to 2.2 plus a fixed 90 s for
# open/unplace/measure/write. Over-estimating only declines the leg (safe
# direction — same rounding rule as DEEP_REPLACE_PR_S_PER_NET).
DEEP_REPLACE_B3_PLACE_DIRECTIVE = "ExtraNetDelay_low"
DEEP_REPLACE_B3_PHYSOPT_DIRECTIVE = "AggressiveExplore"
DEEP_REPLACE_B3_ROUTE_DIRECTIVE = "AggressiveExplore"
DEEP_REPLACE_B3_COST_MULT = 2.2
DEEP_REPLACE_B3_FIXED_OVERHEAD_S = 90.0
# v5.5.3 (AWS eval-parity, aug10): 300 -> 550. The contest instance runs the
# same B1 probe ~1.9x slower than the dev boxes' MEASUREMENTS (mini-ISP:
# 145 s measured on m7a.2xlarge vs 75-78 s measured on every archived dev
# draw jul26->aug08 — the oft-quoted 37 s is the arm-time cell-count ANCHOR,
# not a measurement; r1 finding 2). wns -0.904 is identical on both, so the
# 300 s cap priced the eval box out of a leg whose physics are unchanged
# (est 2.2x145+90 = 409 s). 550 s admits that with margin while the formula
# still bounds the class: est <= 550 implies measured pr <= ~209 s, and the
# floor-exit payoff on an admitted design (~2000+ s of skipped LLM
# loop/polish) dwarfs the spend.
# The old DEEP_REPLACE_B3_MAX_PR_S = 85.0 wall-clock anchor cap is REMOVED:
# on eval hardware the dev-calibrated separation INVERTS (mini-ISP 145 s vs
# vexriscv 130 s, while dev measures 75-78 s vs 92-97 s), so no wall-clock
# cap can express the class. Its job moved to the PHYSICS admission
# attestation (logic_floor.evaluate_b1_admission), which asks the design,
# not the stopwatch — see the b3_admission hook. CONSEQUENCE on dev boxes:
# vexriscv (est 292-303 s) and optical (est ~493 s) now PASS affordability
# and are decided by the admission — the v5.5.3 gate's D-2 census is the
# measured evidence that they refuse there.
DEEP_REPLACE_B3_MAX_COST_S = 550.0


def deep_replace_b3_affordable(
    *,
    measured_place_route_s: float,
    remaining_s: float,
    finalize_reserve_s: float,
) -> Tuple[bool, str]:
    """AFFORDABILITY gate for the B3 small-floor leg, sized from B1's
    MEASUREMENT. v5.5.3: affordability only — the CLASS decision (is this a
    floor-bound macro-dominated design?) moved to the physics admission
    attestation (b3_admission hook), because wall-clock class caps do not
    transfer across hardware (AWS aug10: the dev-calibrated separation
    inverts on the eval box).

    Two conditions, both required:
      1. est <= DEEP_REPLACE_B3_MAX_COST_S — bounds the spend (and implies
         measured pr <= ~209 s);
      2. est fits the usable slice (remaining minus the finalize reserve and
         write I/O) — the same budget discipline as the B2 gate.
    Fails closed on a missing/absurd measurement.
    """
    if measured_place_route_s is None or measured_place_route_s <= 0:
        return False, "B3 declined: no measured B1 place+route (fail closed)"
    est = (measured_place_route_s * DEEP_REPLACE_B3_COST_MULT
           + DEEP_REPLACE_B3_FIXED_OVERHEAD_S)
    if est > DEEP_REPLACE_B3_MAX_COST_S:
        return False, (
            f"B3 declined: est {est:.0f}s = measured place+route "
            f"{measured_place_route_s:.0f}s x {DEEP_REPLACE_B3_COST_MULT} + "
            f"{DEEP_REPLACE_B3_FIXED_OVERHEAD_S:.0f}s exceeds small-design "
            f"cap {DEEP_REPLACE_B3_MAX_COST_S:.0f}s")
    usable = remaining_s - finalize_reserve_s - DEEP_REPLACE_WRITE_IO_S
    if usable < est:
        return False, (f"B3 declined: usable slice {usable:.0f}s < est "
                       f"{est:.0f}s")
    return True, (f"B3 armed: est {est:.0f}s (measured place+route "
                  f"{measured_place_route_s:.0f}s) fits usable {usable:.0f}s "
                  f"within cap {DEEP_REPLACE_B3_MAX_COST_S:.0f}s")


VERDICT_ADOPTED = "ADOPTED"
VERDICT_REJECTED = "REJECTED"
VERDICT_ERROR = "ERROR"
VERDICT_UNAFFORDABLE = "UNAFFORDABLE"


@dataclass
class DeepReplaceResult:
    """Outcome of one deep-replace sibling attempt."""
    attempted: bool = False
    skip_reason: str = ""
    verdict: str = ""
    reason: str = ""
    pre_wns: Optional[float] = None
    post_wns: Optional[float] = None
    post_whs: Optional[float] = None
    unrouted: Optional[int] = None
    candidate_path: Optional[str] = None
    place_s: float = 0.0
    route_s: float = 0.0
    registered: bool = False
    # --- staging (jul25): which leg produced the DCP currently on disk ---
    b1_wns: Optional[float] = None      # routed place+route result
    stage_banked: str = ""              # "" | "routed" | "physopt_retime" | "small_floor"
    b2_reason: str = ""                 # why the retiming tail ran / did not
    b3_wns: Optional[float] = None      # small-floor leg result (measured)
    b3_reason: str = ""                 # why the small-floor leg ran / did not

    def log_line(self) -> str:
        return (f"deep-replace: {self.verdict} {self.reason} "
                f"pre_wns={self.pre_wns} post_wns={self.post_wns} "
                f"whs={self.post_whs} unrouted={self.unrouted} "
                f"place={self.place_s:.0f}s route={self.route_s:.0f}s "
                f"banked={self.stage_banked or 'none'}")


def deep_replace_should_run(
    *,
    enabled: bool,
    pristine_dcp: Optional[str],
    failing_endpoint_count: Optional[int],
    wns_magnitude_ns: Optional[float],
    remaining_s: float,
    cost_basis_s: float,
    finalize_reserve_s: float,
    failing_endpoints_min: int,
    wns_min_ns: float,
    require_physics_band: bool = True,
    reserve_already_in_deadline: bool = False,
) -> Tuple[bool, str]:
    """Pure firing gate — unit-testable, no I/O. Returns ``(run, reason)``.

    ``require_physics_band`` (jul26, DEFAULT True = unchanged behaviour):
    when False, the DEEP-extreme band is not used to REFUSE the stage.

    WHY THE OPTION EXISTS. The band (failing >= 100k AND |WNS| >= 10 ns) is a
    PREDICTION OF BENEFIT being used as a VETO, and it is measurably mis-scoped:

        design      fmax_ratio  failing   |WNS|   operate   regen-from-pristine
        logicnets   0.605         1,529   0.978   +6.45     +100.23
        digit       0.624             -   1.025   +0.00     +15.44
        optical     0.650             -   1.074   +0.42     +12.64
        FIR         0.889           252   0.313   +13.65    LOSES

    logicnets sits three orders of magnitude outside the band and regeneration
    is still worth +93.77 MHz over operating (chain22, jul26).

    The house rule this restores: A PREDICTION MAY SCHEDULE WORK; IT MAY NEVER
    REFUSE IT. Refusal is legitimate only against a hard bound — wall clock,
    legality, a measured cost. Those gates all remain in force below; only the
    benefit-prediction is dropped. Never-worse is unaffected: the stage
    registers an insured-compare MUX candidate, so a regeneration that loses is
    discarded by measurement rather than pre-empted by a guess.

    NOTE this deliberately does NOT widen the band to a new fitted number.
    Choosing a boundary from six designs, on a corpus with 44 alpha-trustworthy
    rows and a 3.5 MHz noise floor, is the threshold-fitting the methodology
    invariant forbids (and a jul25 panel rejected widening 0/5). Replacing a
    guessed veto with "afford it, then measure it" needs no boundary at all.

    Order matters: the kill switch and state checks precede the physics and
    budget gates so a disabled stage is a guaranteed zero-diff no-op.

    Fails CLOSED on every unmeasurable input. An unmeasured physics feature
    must not arm a stage that costs a full place+route — the cost of a wrong
    ARM here is a wasted heavy move inside the eval hour.
    """
    if not enabled:
        return False, "disabled (kill switch; default OFF)"
    if not pristine_dcp:
        return False, "no pristine input dcp recorded"
    # ---- physics gate: the DEEP-extreme sub-class, router's own constants ----
    if failing_endpoint_count is None or wns_magnitude_ns is None:
        return False, (f"physics unmeasurable "
                       f"(failing={failing_endpoint_count} "
                       f"wns={wns_magnitude_ns}) — fail closed")
    if require_physics_band:
        if failing_endpoint_count < failing_endpoints_min:
            return False, (f"failing_endpoints={failing_endpoint_count:,} < "
                           f"{failing_endpoints_min:,} (not DEEP-extreme)")
        if wns_magnitude_ns < wns_min_ns:
            return False, (f"|WNS|={wns_magnitude_ns:.2f} < {wns_min_ns} "
                           f"(not DEEP-extreme)")
    # ---- affordability: never start what cannot finish ----
    if cost_basis_s <= 0:
        return False, "no measured cost anchor (fail closed)"
    # Size the PLACE+ROUTE leg only. The post-route phys_opt retiming pass is
    # gated separately by deep_replace_physopt_affordable() once this leg has
    # actually run, so its cost is sized from a measurement of THIS design on
    # THIS box rather than from a proxy anchor. See the two-gate rationale on
    # DEEP_REPLACE_PHYSOPT_FRAC.
    # DOUBLE-RESERVE FIX (jul30, DEFAULT OFF — ships OFF until farm-validated).
    #
    # `remaining_s` arrives from BOTH call sites already reserve-adjusted:
    #   dcp_optimizer.py:10380 (tail)  passes self._budget_deadline
    #   dcp_optimizer.py:9145  (first) passes time.time() + _budget_remaining()
    # and `_budget_deadline = start + max_wall - _finalize_reserve_seconds`.
    # So adding `+ finalize_reserve_s` here subtracts the SAME 300s a second
    # time: the gate withholds 600s for a finalize that needs 300s.
    #
    # This is the jul26 pattern that earned boom_soc +6.5 MHz — "prediction was
    # solving a problem the staging had already removed" — left unapplied to
    # this gate. Corpus mined jul30 over 59 agent.logs since jul26: of 30
    # distinct terminal-reserve declines, **22 (73%) arm once the double count
    # is removed**, several with 2-3x the measured work time already available
    # (one had 458s remaining for a 125s x1.3 = 162s job). The 8 that remain
    # declined genuinely cannot finish (need 780-2153s), so the correction is
    # DISCRIMINATING rather than merely permissive — it does not arm the
    # impossible cases, which is what makes it safe to test.
    #
    # Never-worse is unaffected either way: the stage registers an
    # insured-compare MUX candidate, so an overrun costs wall (gamma), never a
    # result — the banked best is already mirrored to disk before this runs.
    #
    # WHY THIS IS NOT A STAGE-SPECIFIC RESERVE THAT MERELY SHARES THE VALUE 300.
    # It is fed `cfg.deep_replace_finalize_reserve_s`, a distinct config field
    # that also happens to be 300.0, so the objection is worth answering. Two
    # pieces of in-tree evidence say it stands for the GLOBAL finalize reserve:
    #   * its own comment calls it "the same 300s convention as the other
    #     terminal stages" — a copied convention, not a measured stage need;
    #   * `deep_replace_physopt_affordable()` below computes
    #     `reserve = finalize_reserve_s + DEEP_REPLACE_WRITE_IO_S`, i.e. this
    #     stage's own write cost is accounted by a SEPARATE named constant. So
    #     `finalize_reserve_s` is not standing in for stage I/O.
    #
    # HONEST NUANCE: the ambiguity is WHICH END is wrong, not whether the
    # reserve is applied twice. Either the gate should not add it, or the call
    # sites should pass a raw-wall deadline. This flag fixes the gate end, which
    # is the smaller and opt-in change; both ends produce the same predicate.
    _reserve_term = 0.0 if reserve_already_in_deadline else finalize_reserve_s
    need = cost_basis_s * DEEP_REPLACE_COST_MARGIN + _reserve_term
    if remaining_s < need:
        _how = (f"place+route anchor {cost_basis_s:.0f}s x"
                f"{DEEP_REPLACE_COST_MARGIN}"
                + ("" if reserve_already_in_deadline else
                   f" + reserve {finalize_reserve_s:.0f}s"))
        return False, (f"insufficient terminal reserve (remaining "
                       f"{remaining_s:.0f}s < need {need:.0f}s = {_how})")
    return True, (f"armed ({'DEEP-extreme' if require_physics_band else 'UNBANDED'}"
                  f": failing={failing_endpoint_count:,}, "
                  f"|WNS|={wns_magnitude_ns:.2f}; place+route anchor "
                  f"{cost_basis_s:.0f}sx{DEEP_REPLACE_COST_MARGIN}"
                  + ("" if reserve_already_in_deadline else
                     f" + reserve {finalize_reserve_s:.0f}s")
                  + f" fits {remaining_s:.0f}s"
                  + (" [reserve already in deadline]"
                     if reserve_already_in_deadline else "")
                  + "; phys_opt gated separately post-route)")


def predict_place_route_s(routable_nets: Optional[int] = None,
                          primitive_cells: Optional[int] = None,
                          ) -> Tuple[float, str]:
    """Size-based place+route cost estimate, for sizing the recipe at t=0.

    Accepts either size feature. ``primitive_cells`` is the one the optimizer
    actually has at Phase 1 (``_input_cell_count``); ``routable_nets`` is what
    the replay logs report. When both are given we take the MAX estimate —
    over-estimating only ever declines a run, which is the safe direction, and
    the two forms are known to disagree out of sample (see the constants).

    Returns ``(seconds, why)``. Returns ``0.0`` when no size is measurable,
    which makes the affordability gate fail closed — an unmeasured design must
    not arm a stage that costs a full place+route.

    Use the MEASURED anchor whenever one exists (``replace_gamble_cost_basis``);
    this is strictly the no-anchor fallback for the recipe-first path.
    """
    ests: list = []
    if routable_nets and routable_nets > 0:
        e = float(routable_nets) * DEEP_REPLACE_PR_S_PER_NET
        w = (f"{routable_nets:,} nets x {DEEP_REPLACE_PR_S_PER_NET} = "
             f"{e:.0f}s")
        if routable_nets < 150_000 or routable_nets > 500_000:
            w += " [EXTRAPOLATED, calibrated ~273.7k nets]"
        ests.append((e, w))
    if primitive_cells and primitive_cells > 0:
        e = float(primitive_cells) * DEEP_REPLACE_PR_S_PER_CELL
        w = (f"{primitive_cells:,} cells x {DEEP_REPLACE_PR_S_PER_CELL} = "
             f"{e:.0f}s")
        if primitive_cells < 200_000 or primitive_cells > 700_000:
            w += " [EXTRAPOLATED, calibrated ~378k cells]"
        ests.append((e, w))
    if not ests:
        return 0.0, (f"no size feature measurable (nets={routable_nets}, "
                     f"cells={primitive_cells}) — fail closed")
    best, why = max(ests, key=lambda t: t[0])
    prefix = "size model" if len(ests) == 1 else "size model (max of 2)"
    return best, f"{prefix}: {' | '.join(w for _, w in ests)} -> {best:.0f}s"


def deep_replace_physopt_affordable(
    *,
    measured_place_route_s: float,
    remaining_s: float,
    finalize_reserve_s: float,
) -> Tuple[bool, str]:
    """Second gate: may we extend the banked routed result into phys_opt?

    Called only AFTER place+route has completed and its candidate is on disk,
    so ``measured_place_route_s`` is an observation of this design on this box
    -- not an anchor, not a margined proxy. That is the whole point of
    splitting the gate: the expensive decision (boom_soc: another 1106 s for
    the last +5.10 MHz) is the one made with real data.

    Declining is cheap and safe: the routed candidate is already written and
    registered, so a refusal here costs only the phys_opt delta, never the
    place+route gain.
    """
    if measured_place_route_s <= 0:
        return False, "no measured place+route cost (fail closed)"
    reserve = finalize_reserve_s + DEEP_REPLACE_WRITE_IO_S
    usable = remaining_s - reserve
    if usable <= 0:
        return False, (f"phys_opt declined (remaining {remaining_s:.0f}s does "
                       f"not even cover finalize {finalize_reserve_s:.0f}s + "
                       f"write {DEEP_REPLACE_WRITE_IO_S:.0f}s); routed "
                       f"candidate kept")
    floor = measured_place_route_s * DEEP_REPLACE_B2_MIN_SLICE_FRAC
    if usable < floor:
        return False, (f"phys_opt declined (usable {usable:.0f}s < minimum "
                       f"useful slice {floor:.0f}s = measured place+route "
                       f"{measured_place_route_s:.0f}s x"
                       f"{DEEP_REPLACE_B2_MIN_SLICE_FRAC}); no observed "
                       f"phys_opt pass has ever completed in less; routed "
                       f"candidate kept")
    return True, (f"phys_opt armed with a {usable:.0f}s slice "
                  f"(remaining {remaining_s:.0f}s - finalize "
                  f"{finalize_reserve_s:.0f}s - write "
                  f"{DEEP_REPLACE_WRITE_IO_S:.0f}s); >= minimum useful "
                  f"{floor:.0f}s. B1 is already on disk, so an unfinished B2 "
                  f"costs wall only")


async def run_deep_replace_sibling(
    call_tool: CallTool,
    *,
    pristine_dcp: str,
    chain_best_wns: Optional[float],
    deadline_ts: float,
    wns_tcl: str,
    out_dcp: str,
    log: Callable[[str], None],
    measure: Callable[..., Awaitable[Tuple[Optional[float], int]]],
    measure_hold: Callable[..., Awaitable[Optional[float]]],
    tool_ok: Callable[[object], bool],
    hold_slack_floor_ns: float = 0.0,
    heavy_timeout_s: float = 1800.0,
    light_timeout_s: float = 600.0,
    adopt_margin_ns: float = 0.0,
    finalize_reserve_s: float = 300.0,
    b3_enabled: bool = False,
    b3_admission: Optional[
        Callable[[Optional[float]], Awaitable[Tuple[bool, str]]]] = None,
) -> DeepReplaceResult:
    """Replay the boom_soc winner from the PRISTINE input and bank a candidate.

    STAGED (jul25): the recipe runs as two legs.

      B1  ``open -> unplace -> place Explore -> route``  — measured, gated,
          and WRITTEN to ``out_dcp`` immediately. On boom_soc this is +36.33
          of the recipe's +41.43, reached at 47% of the wall.
      B2  ``phys_opt -directive AlternateFlowWithRetiming`` — entered only if
          ``deep_replace_physopt_affordable`` says the MEASURED B1 cost leaves
          room. It overwrites ``out_dcp`` only on a strict improvement.

    Why staged: the old all-or-nothing form produced no candidate at all
    unless the full ~2750 s sequence completed, so a tail overrun threw away
    a banked, routed, admissible gain. B2 can now fail, regress, or be
    declined without costing B1 — and no reload is ever required, because the
    B1 file is written before B2 starts and is simply not overwritten.

    The pristine DCP is only ever passed to ``open_checkpoint`` — never a
    write target.  The candidate is written to ``out_dcp``, which the caller
    owns and must keep stable until finalize.

    Returns a result carrying the measured post-WNS; the CALLER registers it
    with the insured-compare MUX (this module deliberately does no MUX I/O so
    it stays unit-testable).
    """
    r = DeepReplaceResult(pre_wns=chain_best_wns)
    r.attempted = True

    def _heavy_to() -> float:
        return max(300.0, min(heavy_timeout_s, deadline_ts - time.time()))

    def _light_to() -> float:
        return max(120.0, min(light_timeout_s, deadline_ts - time.time()))

    async def _step(cmd: str, to: float) -> None:
        resp = await call_tool("vivado_run_tcl", {"command": cmd, "timeout": to})
        if not tool_ok(resp):
            txt = str(resp)
            eline = next((l for l in txt.splitlines() if "TCL ERROR" in l), "")
            raise RuntimeError(f"{cmd.split()[0]} failed: {(eline or txt)[:400]}")

    # Fresh session before a full re-place: a LastMile place poisons the
    # session so that a later full place_design fails outright (jun12 repro),
    # and a fresh session places both better and faster (jun07 A/B/C).
    try:
        rr = await call_tool("vivado_restart_vivado", {})
        if tool_ok(rr):
            log("deep-replace: Vivado restarted fresh (session hygiene).")
        else:
            log(f"deep-replace: restart returned error ({str(rr)[:100]}); "
                f"continuing in current session.")
    except Exception as e:  # never fatal — a poisoned place surfaces as ERROR
        log(f"deep-replace: restart failed ({e!r}); continuing.")

    async def _measure_and_gate(label: str, heavy_to=None):
        """Measure the CURRENT session state and apply the accept gates.

        Returns ``(wns_or_None, whs, unrouted, why_rejected)``. Shared by all
        legs so B1/B2/B3 are held to identical standards — the same standards
        register_final_candidate applies, which is what lets the caller enroll
        with verify=False. ``heavy_to`` lets a budget-capped leg (B3) bound the
        incremental-repair route by ITS budget instead of the run-level
        heavy timeout (review-1 MAJOR: the repair could otherwise draw up to
        heavy_timeout_s from a leg the gate priced at ~255s).
        """
        to_fn = heavy_to or _heavy_to
        w, ur = await measure(call_tool, wns_tcl, timeout_s=_light_to())
        if ur != 0:
            # Post-route phys_opt can leave a few connections open; one
            # incremental completion pass is the established pattern.
            await call_tool("vivado_run_tcl",
                            {"command": "route_design", "timeout": to_fn()})
            w, ur = await measure(call_tool, wns_tcl, timeout_s=_light_to())
        whs = await measure_hold(call_tool, timeout_s=_light_to())
        if w is None or ur != 0:
            return None, whs, ur, f"{label} not fully routed (unrouted={ur}, wns={w})"
        if whs is not None and whs < hold_slack_floor_ns:
            return None, whs, ur, (f"{label} hold violation whs={whs} < floor "
                                   f"{hold_slack_floor_ns}")
        if (chain_best_wns is not None
                and w <= chain_best_wns + adopt_margin_ns):
            # Ledger the hurdle. Before jul27 this rejection was log-only, so the
            # gate ledger showed a clean run while this quietly discarded the
            # recipe's result. On jul26 corescore B1 measured 0.020 ns against a
            # 0.15 ns margin, was dropped here, and the stall that followed
            # triggered the ILS preempt at iteration 1 with 0 LLM calls.
            _gl.emit("deep_replace_hurdle", _gl.VERDICT_REFUSE,
                     reason_code="does_not_beat_chain_best",
                     wns_ns=w, best_wns_ns=chain_best_wns,
                     threshold_s=float(adopt_margin_ns),
                     site=f"deep_replace_sibling.{label}")
            return None, whs, ur, (f"{label} wns {w:.3f} does not beat chain-best "
                                   f"{chain_best_wns:.3f} by +{adopt_margin_ns}ns")
        _gl.emit("deep_replace_hurdle", _gl.VERDICT_ALLOW,
                 reason_code="beats_chain_best",
                 wns_ns=w, best_wns_ns=chain_best_wns,
                 threshold_s=float(adopt_margin_ns),
                 site=f"deep_replace_sibling.{label}")
        return w, whs, ur, ""

    async def _run_b3() -> None:
        """B3 small-floor leg — rationale at the DEEP_REPLACE_B3_* constants.

        Runs on BOTH exits (B2 declined / B2 completed), only when the caller
        armed it (band=mid) and the anchor-sized gate passes. Never raises:
        the banked candidate is already on disk, so like B2 this leg can only
        ever cost its own bounded slice. It deliberately does NOT touch
        r.place_s / r.route_s — those are B1's measurements and feed the
        published cost anchor; ExtraNetDelay_low is slower than Explore and
        must not inflate it.
        """
        if not b3_enabled:
            return
        try:
            pr_b1 = r.place_s + r.route_s
            go3, why3 = deep_replace_b3_affordable(
                measured_place_route_s=pr_b1,
                remaining_s=deadline_ts - time.time(),
                finalize_reserve_s=finalize_reserve_s)
            r.b3_reason = why3
            log(f"deep-replace[B3]: {why3}")
            if not go3:
                return
            # v5.5.3 PHYSICS ADMISSION (replaces the removed 85s wall-clock
            # anchor cap): the caller supplies an attestor that inspects the
            # session's B1/B2 solve — hard-macro-dominated near-floor paths
            # admit, anything else refuses. No attestor => fail closed (a
            # B3 armed purely by affordability was review-1's blocker).
            if b3_admission is None:
                r.b3_reason = ("B3 declined: no physics admission attestor "
                               "provided (fail closed)")
                log(f"deep-replace[B3]: {r.b3_reason}")
                return
            ok_admit, why_admit = await b3_admission(r.post_wns)
            r.b3_reason = why_admit
            log(f"deep-replace[B3]: {why_admit}")
            if not ok_admit:
                return
            w_prev = r.post_wns  # best of B1/B2, already on disk
            est3 = (pr_b1 * DEEP_REPLACE_B3_COST_MULT
                    + DEEP_REPLACE_B3_FIXED_OVERHEAD_S)
            b3_deadline = time.time() + max(240.0, est3 * 1.5)

            def _b3_to() -> float:
                # Reserve-aware (review-1 MAJOR fix): every grant is bounded by
                # the B3 deadline AND the run deadline minus the finalize
                # reserve + write I/O — the same discipline as _b2_to. When the
                # budget is exhausted we RAISE (caught below, banked candidate
                # intact) instead of granting a 60s floor past the reserve.
                to = min(_heavy_to(),
                         b3_deadline - time.time(),
                         (deadline_ts - time.time()) - finalize_reserve_s
                         - DEEP_REPLACE_WRITE_IO_S)
                if to < 30.0:
                    raise RuntimeError(
                        "B3 budget exhausted (reserve-aware cap)")
                return to

            await _step(f"open_checkpoint {{{pristine_dcp}}}", _b3_to())
            await _step("place_design -unplace", _b3_to())
            await _step(f"place_design -directive "
                        f"{DEEP_REPLACE_B3_PLACE_DIRECTIVE}", _b3_to())
            await _step(f"phys_opt_design -directive "
                        f"{DEEP_REPLACE_B3_PHYSOPT_DIRECTIVE}", _b3_to())
            await _step(f"route_design -directive "
                        f"{DEEP_REPLACE_B3_ROUTE_DIRECTIVE}", _b3_to())
            w3, whs3, ur3, why_m3 = await _measure_and_gate(
                "B3(small_floor)", heavy_to=_b3_to)
            r.b3_wns = w3
            # v5.4 (panel k3): require a MEASURED hold value to adopt. The
            # shared gate only rejects when whs is present and below floor,
            # so a hold measurement that returned None could otherwise let
            # B3 overwrite a hold-measured B1 candidate with an unmeasured
            # one — on a hidden design with a real hold violation that is an
            # alpha=0 tail. B1/B2 keep their validated behavior.
            if whs3 is None and w3 is not None:
                log("deep-replace[B3]: not adopted — hold measurement "
                    "returned None (unmeasured hold is not adoptable for "
                    "B3); banked candidate on disk is UNTOUCHED")
            elif w3 is not None and (w_prev is None or w3 > w_prev):
                # Write-to-temp + atomic rename (review-2 MAJOR): an in-place
                # `write_checkpoint -force {out_dcp}` that dies mid-write
                # leaves TRUNCATED B3 bytes at the registered path while the
                # exception log claims the banked candidate is intact — the
                # caller registers by path with verify=False, so nothing
                # downstream would catch it (possible alpha=0 on the treated
                # design). With the temp write, a failure at any point leaves
                # out_dcp holding the banked B1/B2 bytes and the log stays
                # true. (B2's in-place write is the same pre-existing class,
                # deliberately untouched here — validated ship surface.)
                if out_dcp.endswith(".dcp"):
                    tmp_dcp = out_dcp[:-4] + ".b3tmp.dcp"
                else:
                    tmp_dcp = out_dcp + ".b3tmp"
                await _step(f"write_checkpoint -force {{{tmp_dcp}}}",
                            _light_to())
                os.replace(tmp_dcp, out_dcp)
                r.post_wns, r.post_whs, r.unrouted = w3, whs3, ur3
                r.stage_banked = "small_floor"
                r.reason = (f"B3 small-floor improved {w_prev} -> {w3:.3f} "
                            f"(candidate overwritten)")
                log(f"deep-replace[B3]: ADOPTED wns={w3:.3f} "
                    f"(previous banked {w_prev})")
            else:
                detail = why_m3 or f"B3 wns {w3} did not beat banked {w_prev}"
                log(f"deep-replace[B3]: not adopted — {detail}; banked "
                    f"candidate on disk is UNTOUCHED")
        except Exception as e:
            log(f"deep-replace[B3]: FAILED ({e!r}); banked candidate intact")

    try:
        # ================= B1: place + route — the money leg =================
        # boom_soc: +36.33 of the recipe's +41.43 lands here, at 1644 s (47% of
        # the wall). Everything below exists so that gain survives whatever the
        # retiming tail does.
        await _step(f"open_checkpoint {{{pristine_dcp}}}", _light_to())
        t_place = time.time()
        await _step("place_design -unplace", _heavy_to())
        await _step(f"place_design -directive {DEEP_REPLACE_PLACE_DIRECTIVE}",
                    _heavy_to())
        r.place_s = time.time() - t_place
        t_route = time.time()
        await _step("route_design", _heavy_to())
        r.route_s = time.time() - t_route

        w1, whs1, ur1, why1 = await _measure_and_gate("B1(place+route)")
        r.post_wns, r.post_whs, r.unrouted = w1, whs1, ur1
        if w1 is None:
            r.verdict = VERDICT_REJECTED
            r.reason = why1
            log(r.log_line())
            return r

        # BANK IT NOW, before anything risky runs. The caller registers by
        # PATH, so from here on the gain is on disk and independent of the
        # live session.
        await _step(f"write_checkpoint -force {{{out_dcp}}}", _light_to())
        r.candidate_path = out_dcp
        r.b1_wns = w1
        r.stage_banked = "routed"
        r.verdict = VERDICT_ADOPTED
        r.reason = f"B1 routed wns={w1:.3f} banked"
        log(f"deep-replace: B1 BANKED wns={w1:.3f} whs={whs1} "
            f"place={r.place_s:.0f}s route={r.route_s:.0f}s -> {out_dcp}")

        # ================= B2 gate: sized from a MEASUREMENT =================
        measured_pr = r.place_s + r.route_s
        go, why2 = deep_replace_physopt_affordable(
            measured_place_route_s=measured_pr,
            remaining_s=deadline_ts - time.time(),
            finalize_reserve_s=finalize_reserve_s)
        r.b2_reason = why2
        if not go:
            r.reason = f"B1 routed wns={w1:.3f} banked; {why2}"
            log(f"deep-replace: {why2}")
            await _run_b3()
            log(r.log_line())
            return r

        # ================= B2: phys_opt retiming extension ===================
        # Its own try: a failure here must not discard B1. There is NO reload
        # on regress — out_dcp already holds B1 and we simply decline to
        # overwrite it (see DEEP_REPLACE_WRITE_IO_S for why that matters).
        try:
            log(f"deep-replace: B2 armed — {why2}")
            # ---- FRESH SESSION BEFORE B2 (jul31) --------------------------
            # boom_soc died THREE times tonight, and the third — alone on a
            # freshly-idle box, full ship build — reproduced B1 bit-for-bit
            # (wns=-10.256, the value chain33 banked on jul26 en route to
            # +41.43) and then vanished inside this phys_opt: log stops,
            # driver and multi_restart gone, no traceback. The OOM signature.
            #
            # WHY HERE. B2 runs `phys_opt -directive AlternateFlowWithRetiming`
            # in the SAME session that just unplaced, re-placed and routed a
            # 250k-cell design, so peak memory is B1's full placer+router state
            # PLUS a retiming pass on top, against 31 GB with swap=0. The
            # pre-B1 restart above already establishes a fresh session as the
            # right tool for exactly this; B2 never got one.
            #
            # WHY IT IS SAFE, in this function's own words: "out_dcp already
            # holds B1". The candidate is on disk and registered before we get
            # here, so a restart cannot cost a RESULT — only wall. phys_opt
            # re-reads the identical checkpoint either way, so B2's INPUT is
            # unchanged; only the memory high-water mark moves.
            #
            # Gated on the design being expensive enough to be at risk (boom
            # measures ~1500-1650 s of place+route; logicnets ~270 s), so small
            # designs never pay the reopen. Budget-checked: if the reopen does
            # not fit, proceed exactly as before rather than risk the slice.
            _b2_restart = os.environ.get(
                "FPL26_DEEP_REPLACE_B2_RESTART", "1").strip().lower() in (
                    "1", "true", "on", "yes")
            _reopen_budget = ((deadline_ts - time.time()) - finalize_reserve_s
                              - DEEP_REPLACE_WRITE_IO_S
                              - DEEP_REPLACE_B2_REOPEN_S)
            _b2_min_pr = b2_restart_min_pr_s()
            if (_b2_restart
                    and measured_pr >= _b2_min_pr
                    and _reopen_budget > 0):
                try:
                    rr2 = await call_tool("vivado_restart_vivado", {})
                    if tool_ok(rr2):
                        await _step(f"open_checkpoint {{{out_dcp}}}", _light_to())
                        log(f"deep-replace: fresh session before B2 "
                            f"(place+route measured {measured_pr:.0f}s >= "
                            f"{_b2_min_pr:.0f}s); reopened "
                            f"the banked B1 so phys_opt starts from a clean "
                            f"process.")
                    else:
                        log(f"deep-replace: B2 restart returned error "
                            f"({str(rr2)[:100]}); continuing in current session.")
                except Exception as _e2:
                    # Never fatal: B1 is banked, and the pre-jul31 behaviour was
                    # to run B2 in this session anyway.
                    log(f"deep-replace: B2 restart/reopen failed ({_e2!r}); "
                        f"continuing in the current session.")
            # Cap B2 at the USABLE slice, not the run deadline. This is what
            # makes running it unconditionally safe: it can never eat the
            # finalize reserve, and if the cap cuts it short we simply keep the
            # B1 file already on disk. Floor at 60 s so the tool call is
            # well-formed even in a degenerate window.
            _b2_to = max(60.0, min(
                heavy_timeout_s,
                (deadline_ts - time.time())
                - finalize_reserve_s - DEEP_REPLACE_WRITE_IO_S))
            await _step(f"phys_opt_design -directive "
                        f"{DEEP_REPLACE_PHYSOPT_DIRECTIVE}", _b2_to)
            w2, whs2, ur2, why_b2 = await _measure_and_gate("B2(physopt_retime)")
            if w2 is not None and w2 > w1:
                await _step(f"write_checkpoint -force {{{out_dcp}}}", _light_to())
                r.post_wns, r.post_whs, r.unrouted = w2, whs2, ur2
                r.stage_banked = "physopt_retime"
                r.reason = (f"B2 improved {w1:.3f} -> {w2:.3f} "
                            f"(candidate overwritten)")
            else:
                detail = why_b2 or f"B2 wns {w2} did not beat B1 {w1:.3f}"
                r.reason = (f"B1 routed wns={w1:.3f} kept; B2 declined "
                            f"({detail})")
                log(f"deep-replace: B2 not adopted — {detail}; "
                    f"B1 candidate on disk is UNTOUCHED")
        except Exception as e:
            # B1 is already written and gated; the run is still a success.
            r.reason = (f"B1 routed wns={w1:.3f} kept; B2 raised "
                        f"{type(e).__name__}: {e!r}"[:300])
            log(f"deep-replace: B2 FAILED ({e!r}); B1 candidate intact")
        await _run_b3()
        log(r.log_line())
        return r
    except Exception as e:
        # A throw before B1 banked leaves no candidate; after it, keep it.
        if r.candidate_path:
            r.reason = (f"B1 routed wns={r.b1_wns} kept; later stage raised "
                        f"{e!r}")[:300]
        else:
            r.verdict = VERDICT_ERROR
            r.reason = f"{e!r}"[:300]
        log(r.log_line())
        return r
