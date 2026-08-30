"""Produces a full-replacement candidate from the pristine input for insured
comparison.

The stage discards accumulated pipeline state, unplaces and explores a fresh
placement, routes it, and applies retiming-oriented physical optimization. The
insured MUX selects the candidate only when its measured WNS is better, so a
worse candidate cannot replace the banked result.

This stage differs from replacement based on the banked-best checkpoint because
its pristine starting point can reach otherwise inaccessible placement basins.
Admission reuses the router's physics constants; the only free parameters are
`enabled` and `finalize_reserve_s`.
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

# The exact winning sequence from the archived record run's log.
# Directives are the logged ones; they are NOT tuned here.
DEEP_REPLACE_PLACE_DIRECTIVE = "Explore"
DEEP_REPLACE_PHYSOPT_DIRECTIVE = "AlternateFlowWithRetiming"

# Affordability margin on the measured cost anchor. Same constant and meaning
# as replace_gamble's, kept separate only so the two stages can be reasoned
# about independently.
DEEP_REPLACE_COST_MARGIN = 1.3

# Estimated post-route phys_opt cost as a fraction of measured
# place-and-route time; the observed ratio, rounded upward.
#
# RETIRED, kept as a reference value. It once stacked onto the arm gate, and
# the comment here still claimed it was "used only by the second
# affordability gate" long after that stopped being true: the arm gate now
# sizes only the place-and-route leg, and B2's affordability floor is
# DEEP_REPLACE_B2_MIN_SLICE_FRAC below. Nothing in optimizer/ reads this any
# more -- its only reader is test_deep_replace_sibling.py, which reconstructs
# the retired stacked estimate to show it could not fit the wall.
DEEP_REPLACE_PHYSOPT_FRAC = 0.7

# Reserve checkpoint I/O for the optional second stage.
# The baseline checkpoint must be written before B2; a failed or regressing
# B2 leaves it untouched, and registration uses its path. Thirty seconds
# covers the initial write plus a possible winning overwrite.
DEEP_REPLACE_WRITE_IO_S = 30.0

# Run B2 in a fresh tool session after unusually expensive place-and-route
# to reduce accumulated peak memory. The 1,200 s cutoff identifies the
# high-runtime class, and 120 s reserves restart and checkpoint reopen.
# If reopening does not fit, B2 proceeds in the existing session.
DEEP_REPLACE_B2_RESTART_MIN_PR_S = 1200.0
DEEP_REPLACE_B2_REOPEN_S = 120.0


def b2_restart_min_pr_s() -> float:
    """Threshold, env-overridable (FPL26_DEEP_REPLACE_B2_RESTART_MIN_PR_S).

    Overridable so it can be re-tuned from a new measurement without a code change,
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

# Startup cost estimate for place-and-route when no measured cycle is available.
# Units are seconds per routable net at 8-vCPU evaluation capacity.
# The conservative estimate fails closed by declining work that may not fit.
# The coefficient is calibrated near 274k nets and is provisional far from that size.
DEEP_REPLACE_PR_S_PER_NET = 0.00601   # rounded UP from the measured 0.0060094

# Cell-count cost model used at startup, when primitive count is available but
# routable net count is not. Units are seconds per primitive cell.
# The caller uses the maximum available estimate because cell-to-net ratios vary.
# The coefficient is calibrated from place-and-route runtime near 378k cells.
DEEP_REPLACE_PR_S_PER_CELL = 0.004350

# Minimum worthwhile B2 budget as a fraction of measured place-and-route time.
# B1 is banked before B2 starts, so B2 can consume wall time but cannot lose it.
# The deadline preserves the finalize reserve.
# The 0.20 floor is below completed-pass ratios so short feasible tails are admitted.
DEEP_REPLACE_B2_MIN_SLICE_FRAC = 0.20

# B3 is an independent candidate built from the pristine checkpoint for
# hard-macro-dominated paths near a timing floor.
# It requires both bounded estimated cost and physics-based admission; B1/B2 is
# banked first. Cost is 2.2 times measured place-and-route plus fixed overhead.
DEEP_REPLACE_B3_PLACE_DIRECTIVE = "ExtraNetDelay_low"
DEEP_REPLACE_B3_PHYSOPT_DIRECTIVE = "AggressiveExplore"
DEEP_REPLACE_B3_ROUTE_DIRECTIVE = "AggressiveExplore"
DEEP_REPLACE_B3_COST_MULT = 2.2
DEEP_REPLACE_B3_FIXED_OVERHEAD_S = 90.0
# Maximum estimated B3 wall cost, in seconds. The 550 s cap accommodates slower
# evaluation hosts while bounding optional work.
# Physics attestation, rather than raw runtime, determines design eligibility.
DEEP_REPLACE_B3_MAX_COST_S = 550.0


def deep_replace_b3_affordable(
    *,
    measured_place_route_s: float,
    remaining_s: float,
    finalize_reserve_s: float,
) -> Tuple[bool, str]:
    """Determine whether the small-floor replacement leg is affordable.

    The cost estimate derives from the earlier placement measurement and must
    not exceed `DEEP_REPLACE_B3_MAX_COST_S`. It must also fit within the
    remaining time after the finalization reserve and write I/O allowance.

    This gate checks budget only; `b3_admission` decides class eligibility.
    Missing or implausible measurements fail closed.
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
    # --- staging: which leg produced the DCP currently on disk ---
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
    """Decide whether the deep replacement stage may run without performing I/O.

    Returns the decision and a reason. Setting `require_physics_band=False`
    prevents the predicted-benefit band from vetoing the stage; hard legality,
    state, and budget gates still apply. Predicted benefit may schedule work
    but must not refuse affordable work, because insured comparison discards
    regressions after measurement. The kill switch and state checks precede the
    physics and budget gates, so disabling the stage guarantees a no-op. Any
    unmeasurable required input fails closed to avoid starting a full
    place-and-route operation without a cost basis.
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
    # remaining_s may already exclude the finalize reserve at the call site.
    # reserve_already_in_deadline prevents subtracting that reserve twice.
    # This gate sizes only place-and-route; post-route phys_opt has its own gate.
    # The candidate is banked before optional work, so timeout loses only wall time.
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
    """Estimate place-and-route cost in seconds from available design-size
    features.

    The estimate may use primitive-cell count, routable-net count, or both;
    when both are present, the larger estimate is used so uncertainty fails
    toward declining the run. Returns `0.0` when neither size is measurable,
    causing the affordability gate to fail closed. A measured
    `replace_gamble_cost_basis` takes precedence; this estimate is only the
    fallback used during initial recipe sizing.
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
    """Determine whether the routed deep replacement candidate can afford physical
    optimization.

    This gate runs only after place and route has completed and the routed
    candidate has been persisted. The measured place-and-route duration is an
    observation from the current design and host, allowing the remaining budget
    to be evaluated without a proxy estimate. Declining preserves the routed
    candidate and avoids only the additional physical-optimization cost.
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
    """Regenerate placement from the pristine checkpoint and persist a routed
    candidate.

    The first leg opens the input, unplaces the design, performs exploratory
    placement and routing, measures the result, and immediately writes
    `out_dcp`. The optional second leg runs retiming-aware physical
    optimization only when the measured first-leg cost leaves sufficient
    budget, and overwrites `out_dcp` only on strict improvement. Declining or
    failing the second leg leaves the routed first-leg candidate intact and
    requires no reload. The pristine checkpoint is read-only and is used only
    with `open_checkpoint`. The caller owns `out_dcp` and must keep it stable
    until finalization. The result includes measured post-route WNS; the caller
    registers it for insured comparison, keeping this function free of
    candidate-multiplexer I/O.
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

    # Restart before full placement because LastMile placement can leave session
    # state that causes a subsequent full place_design to fail.
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
        heavy timeout (review finding: the repair could otherwise draw up to
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
            # Record hurdle rejection in the gate ledger so discarded candidates are
            # represented as refusals rather than apparently clean runs.
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
            # B3 requires an attestor to confirm hard-macro-dominated, near-floor paths.
            # Missing attestation fails closed; affordability alone never admits B3.
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
                # Bound each grant by the B3 deadline and the run budget after finalize
                # and checkpoint-write reserves. Exhaustion raises into the handler,
                # leaving the banked candidate intact rather than extending the timeout.
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
            # B3 adoption requires measured hold slack; missing data fails
            # closed. This prevents an unmeasured result from replacing a hold-
            # validated candidate.
            if whs3 is None and w3 is not None:
                log("deep-replace[B3]: not adopted — hold measurement "
                    "returned None (unmeasured hold is not adoptable for "
                    "B3); banked candidate on disk is UNTOUCHED")
            elif w3 is not None and (w_prev is None or w3 > w_prev):
                # Write B3 to a temporary checkpoint and atomically rename it
                # into place. A failed in-place write could truncate the
                # registered banked checkpoint. Any write failure therefore
                # leaves the B1/B2 checkpoint unchanged.
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
        # B1 produces the primary place-and-route candidate, which is banked before
        # optional retiming extensions run.
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

        # Isolate B2 failures so the banked B1 checkpoint remains available.
        # On regression, do not reload or overwrite out_dcp.
        try:
            log(f"deep-replace: B2 armed — {why2}")
            # Restarting before B2 releases placer/router state that can
            # increase peak memory during retiming on large designs. B1 is
            # already banked. Restart only for expensive designs when reopening
            # fits the protected budget; otherwise continue B2 in the current
            # session.
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
                    # Never fatal: B1 is banked, and the previous behaviour was
                    # to run B2 in this session anyway.
                    log(f"deep-replace: B2 restart/reopen failed ({_e2!r}); "
                        f"continuing in the current session.")
            # Limit B2 to the budget remaining after finalize and checkpoint-
            # write reserves. If interrupted, the banked B1 checkpoint remains
            # available. The 60 s timeout floor keeps the tool call valid but
            # can cross the protected reserve if this code is reached with less
            # than 60 s usable.
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
