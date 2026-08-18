"""Logic-floor attestation (v5.5) — the physics termination predicate.

WHY (mechanism, k3 seat aug09): achieved period = logic (frozen by the
netlist and the silicon) + net (placement-dependent) + uncertainty − skew.
When the worst paths are measurably at that state — logic-dominated, net at
the per-hop routing floor, the residual harvestable bound small — no
placement/route/phys_opt move can pay for the wall it costs, so the run
should finalize immediately. The 16-design STOP-honor census (aug09) shows
why this must be a PHYSICS predicate and not a timer: the timer signal loses
on 10/16 designs (digit tail +54 alpha, vexriscv +42.9) because their
mid-run paths are net/congestion-dominated; this predicate refuses there by
construction.

SCOPE: evaluated ONLY after the B3 small-floor sibling was ADOPTED. Since
v5.5.3 the B3 funnel is gated by affordability (cost cap 550s) plus the
PHYSICS admission below — not the removed 85s wall-clock pr cap. Every
guard failure is NO-FIRE: the run falls through to normal v5.4.2 behavior.

Thresholds (k3, fixed — an EV*probability form was rejected as
unidentifiable from n=1):
  - bound_alpha <= 10.0 MHz over the top-32 setup paths, net floor 45 ps/hop
    (mini-ISP's own 3-router-converged routing measures 51.4 ps/hop at zero
    congestion; 45 sits 12% below anything achieved => the bound
    OVER-estimates harvestable alpha, the fail-safe direction).
    mini-ISP: 7.83 MHz (28% margin). corescore-class mid-run: >= 14.5 (1.45x)
    — and corescore can never reach this predicate (B3 does not arm there).
  - worst-path logic fraction >= 0.80 (mini-ISP 84.2%).
  - hard-macro share of the worst path's logic delay >= 0.50: 84% logic in a
    LUT chain is retimeable and ILS would earn there — only DSP/BRAM/URAM
    internal delay is unbreakable. mini-ISP's 1.911 ns is DSP-internal.
  - two-independent-solve agreement: |wns_B1 − wns_B3| <= 0.08 ns (the live
    analog of the 3-router convergence; geometry alone must not attest).
  - coverage: the 32nd path's slack >= WNS + 0.100 ns, else the near-critical
    population is too large to attest from a 32-path window.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional, Tuple

LOGIC_FLOOR_MAX_BOUND_ALPHA_MHZ = 10.0
LOGIC_FLOOR_MIN_LOGIC_FRAC = 0.80
LOGIC_FLOOR_MIN_MACRO_FRAC = 0.50
LOGIC_FLOOR_TWO_SOLVE_MAX_DELTA_NS = 0.08
LOGIC_FLOOR_NET_FLOOR_NS_PER_HOP = 0.045
LOGIC_FLOOR_COVERAGE_MARGIN_NS = 0.100
LOGIC_FLOOR_NPATHS = 32

# One Tcl call, emitting one LFPATH line per path plus LFMETA. Every value
# the python side needs is printed explicitly; any missing/garbled line is a
# parse miss => NO-FIRE.
# ONE LINE, semicolon-joined — the vivado_run_tcl transport sendline/expects a
# single prompt; a multi-line payload returns at the first embedded newline
# and leaves the remaining commands' prompts in the pexpect buffer, desyncing
# every subsequent call (v5.5 review-1 BLOCKER 1). Cell classification uses
# the RESOURCE TYPE token that precedes the (Prop_...) arc name — real 2025.1
# grammar is `LUT3 (Prop_A6LUT_SLICEL_I2_O)` / `DSP_ALU (Prop_DSP_ALU_...)`
# with the Incr/Path numbers on the FOLLOWING line (\s crosses the newline)
# (review-1 BLOCKER 2, fixed against captured real report text).
LOGIC_FLOOR_TCL = (
    'set lf_clk [get_clocks -quiet {%CLK%}]; '
    'if {[llength $lf_clk] == 0} { set lf_clk [lindex [get_clocks] 0] }; '
    'set lf_paths [get_timing_paths -quiet -setup -sort_by slack '
    '-max_paths %N% -nworst 1 -unique_pins -group $lf_clk]; '
    'puts "LFMETA npaths=[llength $lf_paths]"; '
    'set lf_i 0; '
    'foreach lf_p $lf_paths { '
    'set lf_slack [get_property -quiet SLACK $lf_p]; '
    'set lf_rpt [report_timing -of_objects $lf_p -return_string]; '
    'set lf_dp -1; set lf_lg -1; set lf_rt -1; '
    r'regexp {Data Path Delay:\s+([0-9.]+)ns\s+\(logic\s+([0-9.]+)ns[^r]*'
    r'route\s+([0-9.]+)ns} $lf_rpt -> lf_dp lf_lg lf_rt; '
    'set lf_hops [llength [get_nets -quiet -of_objects $lf_p]]; '
    'set lf_lut 0.0; set lf_macro 0.0; '
    'foreach {lf_all lf_cell lf_d} [regexp -all -inline '
    r'{([A-Za-z0-9_]+)\s+\(Prop_[^)]*\)\s+([0-9.]+)} $lf_rpt] { '
    'if {[string match LUT* $lf_cell]} '
    '{ set lf_lut [expr {$lf_lut + $lf_d}] }; '
    'if {[string match DSP* $lf_cell] || [string match RAMB* $lf_cell] '
    '|| [string match URAM* $lf_cell]} '
    '{ set lf_macro [expr {$lf_macro + $lf_d}] } }; '
    'puts "LFPATH i=$lf_i slack=$lf_slack dp=$lf_dp lg=$lf_lg rt=$lf_rt '
    'hops=$lf_hops lut=$lf_lut macro=$lf_macro"; '
    'incr lf_i }; '
    'puts "LFDONE"'
)


def _tool_error_text(resp_text: str) -> bool:
    """True iff the tool response looks like a failure, matched against the
    REAL error grammar this codebase documents (r2 finding F1: a bare
    case-sensitive "Error" substring matches none of it): JSON '{"error"...}',
    'TCL ERROR:', 'ERROR: [Common 17-...', 'ERROR: [Vivado ...', and the
    transport's 'Error: Command timed out'. 'error:' is matched
    case-insensitively; benign strings like '0 errors' do not contain it."""
    t = resp_text.lower()
    return '"error"' in t or "tcl error" in t or "error:" in t


@dataclass
class LFPath:
    slack: float
    dp: float
    lg: float
    rt: float
    hops: int
    lut: float
    macro: float


@dataclass
class LogicFloorVerdict:
    fire: bool
    reason: str
    bound_alpha_mhz: Optional[float] = None
    logic_frac: Optional[float] = None
    macro_frac: Optional[float] = None
    details: List[str] = field(default_factory=list)


def parse_lf_output(text: str) -> Tuple[Optional[bool], List[LFPath]]:
    """Returns (routed_or_None, paths). Any malformed LFPATH line is DROPPED
    (the caller treats short coverage as NO-FIRE, so a drop is fail-safe)."""
    routed = None
    m = re.search(r"LFMETA routed=(\d)", text)
    if m:
        routed = m.group(1) == "1"
    paths: List[LFPath] = []
    for pm in re.finditer(
            r"LFPATH i=\d+ slack=(-?[0-9.]+) dp=(-?[0-9.]+) lg=(-?[0-9.]+) "
            r"rt=(-?[0-9.]+) hops=(\d+) lut=([0-9.]+) macro=([0-9.]+)", text):
        try:
            p = LFPath(slack=float(pm.group(1)), dp=float(pm.group(2)),
                       lg=float(pm.group(3)), rt=float(pm.group(4)),
                       hops=int(pm.group(5)), lut=float(pm.group(6)),
                       macro=float(pm.group(7)))
        except ValueError:
            continue
        if p.dp <= 0 or p.lg < 0 or p.rt < 0 or p.hops <= 0:
            continue
        paths.append(p)
    return routed, paths


def evaluate_logic_floor(
    *,
    routed: Optional[bool],
    paths: List[LFPath],
    period_ns: float,
    wns_b1: Optional[float],
    wns_b3: Optional[float],
    npaths_requested: int = LOGIC_FLOOR_NPATHS,
) -> LogicFloorVerdict:
    """Pure decision core (unit-testable). Every gate failure => NO-FIRE."""
    d: List[str] = []
    if not routed:
        return LogicFloorVerdict(False, "not fully routed (fail-safe)")
    if wns_b1 is None or wns_b3 is None:
        return LogicFloorVerdict(False, "missing two-solve values")
    if abs(wns_b1 - wns_b3) > LOGIC_FLOOR_TWO_SOLVE_MAX_DELTA_NS:
        return LogicFloorVerdict(
            False, f"two-solve disagreement |{wns_b1:.3f}-{wns_b3:.3f}| > "
                   f"{LOGIC_FLOOR_TWO_SOLVE_MAX_DELTA_NS}")
    if len(paths) < npaths_requested:
        return LogicFloorVerdict(
            False, f"coverage: only {len(paths)}/{npaths_requested} paths "
                   f"parsed (fail-safe)")
    wns = paths[0].slack
    # Consistency guard (review-1 finding 7): the measured worst slack must
    # BE the B3 value we are attesting — a wrong-group or stale-timing read
    # must never attest silently.
    if abs(wns - wns_b3) > 0.010:
        return LogicFloorVerdict(
            False, f"worst-path slack {wns:.3f} disagrees with B3 "
                   f"{wns_b3:.3f} (wrong group / stale timing — fail-safe)")
    # TRUNCATED-WINDOW TREATMENT (v5.5.2, decided on the live box6
    # measurement): the original plan FAILED when path[N-1] sat within 100ps
    # of WNS ("population too large to attest"). Measured reality on the
    # lever's own design: mini-ISP's top-32 are ALL within 16 mils of WNS
    # (the DSP family is wide), yet every path bounds at -0.805 +/- 0.001 —
    # the old gate no-fired on the exact state it exists for. And the fear
    # was directionally wrong: an UNSEEN path has slack >= paths[-1].slack,
    # and bound_i >= slack_i always, so unseen paths can only DRAG the
    # achievable WNS DOWN (less harvest), never above the window's min
    # bound. The window min-bound therefore remains an OVER-estimate of
    # harvestable alpha — exactly what a <=10 MHz fire test needs. The
    # dense-window FAIL is removed; the short-parse FAIL above stays (a
    # window we could not even read is still distrusted).
    # min-over-window bound: slack each path could reach if its net delay
    # were driven to hops * 45ps (an over-estimate of harvestable gain).
    bound_slack = None
    for p in paths:
        b = p.slack + max(
            0.0, p.rt - LOGIC_FLOOR_NET_FLOOR_NS_PER_HOP * p.hops)
        if bound_slack is None or b < bound_slack:
            bound_slack = b
    f_now = 1000.0 / (period_ns - wns)
    f_bound = 1000.0 / (period_ns - bound_slack)
    bound_alpha = f_bound - f_now
    worst = paths[0]
    logic_frac = worst.lg / worst.dp if worst.dp > 0 else 0.0
    macro_frac = worst.macro / worst.lg if worst.lg > 0 else 0.0
    d.append(f"bound_alpha={bound_alpha:.2f}MHz logic_frac={logic_frac:.3f} "
             f"macro_frac={macro_frac:.3f} wns={wns:.3f} "
             f"bound_slack={bound_slack:.3f}")
    if bound_alpha > LOGIC_FLOOR_MAX_BOUND_ALPHA_MHZ:
        return LogicFloorVerdict(
            False, f"bound_alpha {bound_alpha:.2f} > "
                   f"{LOGIC_FLOOR_MAX_BOUND_ALPHA_MHZ} MHz — net residual "
                   f"still harvestable", bound_alpha, logic_frac, macro_frac, d)
    if logic_frac < LOGIC_FLOOR_MIN_LOGIC_FRAC:
        return LogicFloorVerdict(
            False, f"logic_frac {logic_frac:.3f} < "
                   f"{LOGIC_FLOOR_MIN_LOGIC_FRAC}", bound_alpha, logic_frac,
            macro_frac, d)
    if macro_frac < LOGIC_FLOOR_MIN_MACRO_FRAC:
        return LogicFloorVerdict(
            False, f"macro_frac {macro_frac:.3f} < "
                   f"{LOGIC_FLOOR_MIN_MACRO_FRAC} — logic is retimeable "
                   f"fabric, ILS could earn here", bound_alpha, logic_frac,
            macro_frac, d)
    return LogicFloorVerdict(
        True, f"logic floor attested: bound {bound_alpha:.2f} MHz <= "
              f"{LOGIC_FLOOR_MAX_BOUND_ALPHA_MHZ}, logic {logic_frac:.0%}, "
              f"hard-macro {macro_frac:.0%}, two-solve agree",
        bound_alpha, logic_frac, macro_frac, d)


async def run_logic_floor_attestation(
    call_tool: Callable[..., Awaitable[object]],
    *,
    clock_name: str,
    period_ns: float,
    wns_b1: Optional[float],
    wns_b3: Optional[float],
    log: Callable[[str], None],
    recover_dcp: Optional[str] = None,
    timeout_s: float = 120.0,
) -> LogicFloorVerdict:
    """One Tcl round-trip + the pure evaluation. Never raises: any tool
    error/timeout/parse failure is NO-FIRE.

    routed=True is supplied to the evaluator BY CONTRACT: the caller invokes
    this only on a state the B3 adoption gate just measured fully routed
    (unrouted==0 + hold), and nothing runs in the session between adoption
    and this call. (The Tcl-side route check was dropped in v5.5.1 — the
    naive `*fully routed*` match hits report_route_status's label line
    regardless of errors, review-1 finding 4.)

    Recovery (review-1 finding 5): if the Tcl did not complete — timeout or
    a transport desync — the session may hold a wedged command or stale
    buffered output that would poison the NEXT tool calls. Best-effort:
    restart Vivado and reopen the banked candidate (recover_dcp) so the
    session again holds exactly the state downstream stages expect.
    """
    incomplete = False
    try:
        cmd = (LOGIC_FLOOR_TCL
               .replace("%CLK%", clock_name or "*")
               .replace("%N%", str(LOGIC_FLOOR_NPATHS)))
        resp = await call_tool("vivado_run_tcl",
                               {"command": cmd, "timeout": timeout_s})
        text = str(resp)
        if "LFDONE" not in text:
            incomplete = True
            return LogicFloorVerdict(False, "attestation Tcl did not "
                                            "complete (fail-safe)")
        routed, paths = parse_lf_output(text)
        v = evaluate_logic_floor(routed=True, paths=paths,
                                 period_ns=period_ns,
                                 wns_b1=wns_b1, wns_b3=wns_b3)
        log(f"[logic-floor] {'FIRE' if v.fire else 'no-fire'}: {v.reason}"
            + (f" | {v.details[0]}" if v.details else ""))
        return v
    except Exception as e:
        incomplete = True
        return LogicFloorVerdict(False, f"attestation raised {e!r} "
                                        f"(fail-safe)")
    finally:
        if incomplete:
            try:
                log("[logic-floor] attestation incomplete — restarting "
                    "Vivado to clear any wedged/desynced session state.")
                rr = await call_tool("vivado_restart_vivado", {})
                if recover_dcp and not _tool_error_text(str(rr)):
                    await call_tool("vivado_run_tcl", {
                        "command": "open_checkpoint {%s}" % recover_dcp,
                        "timeout": 600.0})
                    log("[logic-floor] session recovered: banked candidate "
                        "reopened.")
            except Exception as re_:
                log(f"[logic-floor] session recovery failed ({re_!r}) — "
                    f"downstream stages fall back to their own "
                    f"wedge handling.")


# ---------------------------------------------------------------------------
# v5.5.3 — B3 ADMISSION by physics (replaces the wall-clock anchor-class cap)
#
# WHY (AWS eval-parity, aug10): the 85s measured-pr cap was calibrated on the
# dev-cloud boxes (mini-ISP MEASURED 75-78s across every archived draw
# jul26→aug08; vexriscv 92-97s). On the CONTEST instance (m7a.2xlarge) the
# same probes measure mini-ISP 145s (~1.9x the dev measurement) and vexriscv
# 130s — the separation INVERTS, so no wall-clock cap can admit the
# floor-bound design and refuse the re-place design on the hardware that
# scores. Wall time is a property of the box; the class we mean is a
# property of the DESIGN. The admission therefore asks the design directly:
# are the B1 solve's worst paths already hard-macro-dominated and near the
# structural floor? mini-ISP at B1 (-0.904, measured IDENTICALLY on dev and
# AWS): logic 84%, macro ~0.94+, bound ~7.5 MHz => ADMIT (any hardware).
# vexriscv at B1 (-0.785, LUT re-place fabric): macro << 0.50 => REFUSE
# (any hardware). Fail-closed everywhere: a refusal is exactly the
# pre-v5.5.3 decline path.
#
# Thresholds are RELAXED vs the exit attestation (B1 sits ~54ps above the
# B3 floor, so its harvestable bound is larger) but keep the same fail-safe
# direction: bound over-estimates harvest; macro/logic ask for the same
# structural dominance the exit test proves at the floor.
# BOUND THRESHOLD 18 (corrected by the AWS gate's own first measurement,
# aug10 04:26Z): the original 12 was derived from the FLOOR-state path shape
# (bound ~7.5 MHz), but at B1 the design sits ~54ps ABOVE its floor and the
# harvest B3 itself recovers is part of the B1 bound — the aug10 AWS gate
# measured bound_alpha = 14.38 MHz at B1 (logic 0.828, macro 0.974,
# wns -0.904, the predicted values) and correctly-but-wrongly refused.
# 18 = measured 14.38 + ~25% margin; with the window min-bound fixed at the
# measured -0.819, admission holds for B1 draws down to ~-0.925 (-0.904 ->
# 14.38, -0.92 -> 16.98; -0.93 -> 18.6 REFUSES). The EXIT attestation
# (<=10 MHz at the floor) is unchanged and still solely decides termination;
# an over-admitted design costs one MUX-protected B3 leg (acknowledged R3).
# ---------------------------------------------------------------------------
LOGIC_FLOOR_B1_MAX_BOUND_ALPHA_MHZ = 18.0
LOGIC_FLOOR_B1_MIN_LOGIC_FRAC = 0.70
LOGIC_FLOOR_B1_MIN_MACRO_FRAC = 0.50
LOGIC_FLOOR_B1_MIN_NPATHS = 16
# Coverage floor is 16 (vs the exit test's strict 32): the admission only
# needs the WINDOW MINIMUM bound, and dropping parsed paths can only
# raise bound_slack => raise bound_alpha => make admission HARDER (the
# fail-safe direction); the exit test keeps 32 because it terminates the
# run. A garbled worst line additionally trips the wrong-state sanity
# check against the banked WNS. (r2 finding F5 documentation.)
# Wrong-state guard: the session's worst slack must be NEAR the banked value
# (B2 may have left a slightly different-but-equivalent solve open; 0.10 ns
# tolerates that while still refusing to attest an unrelated design state).
# On a wrong-state refusal the ASYNC wrapper reopens the banked candidate
# (a known fully-routed artifact) and re-attests ONCE — so B2 drift can
# never silently disarm the admission on its target design (r1 finding 4).
LOGIC_FLOOR_B1_WORST_SANITY_NS = 0.10
_B1_WRONG_STATE_MARK = "wrong state"


def evaluate_b1_admission(
    *,
    paths: List[LFPath],
    period_ns: float,
    wns_banked: Optional[float],
) -> LogicFloorVerdict:
    """Pure B3-admission core. fire == ADMIT. Every failure => REFUSE.

    State integrity contract (r1 finding 3 — no false 'routed' claim): this
    core cannot see route status. Integrity is enforced by (a) the
    worst-slack sanity window against the BANKED measured value, (b) the
    coverage minimum, and (c) the async wrapper's reopen-banked-and-reattest
    path, which attests the banked artifact itself — a state the B3
    adoption gate measured fully routed before writing."""
    if wns_banked is None:
        return LogicFloorVerdict(False, "no banked WNS to sanity-check "
                                        "against (fail closed)")
    if len(paths) < LOGIC_FLOOR_B1_MIN_NPATHS:
        return LogicFloorVerdict(
            False, f"coverage: only {len(paths)}/"
                   f"{LOGIC_FLOOR_B1_MIN_NPATHS} paths parsed (fail closed)")
    worst = paths[0]
    if abs(worst.slack - wns_banked) > LOGIC_FLOOR_B1_WORST_SANITY_NS:
        return LogicFloorVerdict(
            False, f"session worst slack {worst.slack:.3f} is not the banked "
                   f"solve {wns_banked:.3f} ({_B1_WRONG_STATE_MARK} — "
                   f"fail closed)")
    bound_slack = None
    for p in paths:
        b = p.slack + max(
            0.0, p.rt - LOGIC_FLOOR_NET_FLOOR_NS_PER_HOP * p.hops)
        if bound_slack is None or b < bound_slack:
            bound_slack = b
    f_now = 1000.0 / (period_ns - worst.slack)
    f_bound = 1000.0 / (period_ns - bound_slack)
    bound_alpha = f_bound - f_now
    logic_frac = worst.lg / worst.dp if worst.dp > 0 else 0.0
    macro_frac = worst.macro / worst.lg if worst.lg > 0 else 0.0
    detail = (f"bound_alpha={bound_alpha:.2f}MHz logic_frac={logic_frac:.3f} "
              f"macro_frac={macro_frac:.3f} wns={worst.slack:.3f}")
    if bound_alpha > LOGIC_FLOOR_B1_MAX_BOUND_ALPHA_MHZ:
        return LogicFloorVerdict(
            False, f"bound_alpha {bound_alpha:.2f} > "
                   f"{LOGIC_FLOOR_B1_MAX_BOUND_ALPHA_MHZ} MHz — net residual "
                   f"harvestable, not floor-class", bound_alpha, logic_frac,
            macro_frac, [detail])
    if logic_frac < LOGIC_FLOOR_B1_MIN_LOGIC_FRAC:
        return LogicFloorVerdict(
            False, f"logic_frac {logic_frac:.3f} < "
                   f"{LOGIC_FLOOR_B1_MIN_LOGIC_FRAC}", bound_alpha,
            logic_frac, macro_frac, [detail])
    if macro_frac < LOGIC_FLOOR_B1_MIN_MACRO_FRAC:
        return LogicFloorVerdict(
            False, f"macro_frac {macro_frac:.3f} < "
                   f"{LOGIC_FLOOR_B1_MIN_MACRO_FRAC} — retimeable fabric, "
                   f"not floor-class", bound_alpha, logic_frac, macro_frac,
            [detail])
    return LogicFloorVerdict(
        True, f"floor-class attested at B1: bound {bound_alpha:.2f} MHz <= "
              f"{LOGIC_FLOOR_B1_MAX_BOUND_ALPHA_MHZ}, logic "
              f"{logic_frac:.0%}, hard-macro {macro_frac:.0%}",
        bound_alpha, logic_frac, macro_frac, [detail])


async def run_b1_admission_attestation(
    call_tool: Callable[..., Awaitable[object]],
    *,
    clock_name: str,
    period_ns: float,
    wns_banked: Optional[float],
    log: Callable[[str], None],
    recover_dcp: Optional[str] = None,
    timeout_s: float = 120.0,
) -> LogicFloorVerdict:
    """Tcl round-trip(s) + evaluate_b1_admission. Never raises; any tool
    error/timeout/parse failure REFUSES admission (pre-v5.5.3 behavior).

    WRONG-STATE RECOVERY (r1 finding 4): if the first attestation refuses
    because the session's worst slack is not the banked value (B2 left a
    drifted solve open), reopen the BANKED candidate (recover_dcp — a state
    the adoption gate measured fully routed before writing) and re-attest
    ONCE. B2 drift therefore cannot silently disarm the admission; only a
    genuinely non-floor-class design (or a tool failure) refuses.

    Wedge-recovery contract on an incomplete Tcl matches
    run_logic_floor_attestation: restart Vivado, best-effort reopen the
    banked candidate so downstream stages see the state they expect."""
    incomplete = False

    async def _attest_once() -> LogicFloorVerdict:
        nonlocal incomplete
        cmd = (LOGIC_FLOOR_TCL
               .replace("%CLK%", clock_name or "*")
               .replace("%N%", str(LOGIC_FLOOR_NPATHS)))
        resp = await call_tool("vivado_run_tcl",
                               {"command": cmd, "timeout": timeout_s})
        text = str(resp)
        if "LFDONE" not in text:
            incomplete = True
            return LogicFloorVerdict(False, "admission Tcl did not complete "
                                            "(fail closed)")
        _routed, paths = parse_lf_output(text)
        return evaluate_b1_admission(paths=paths, period_ns=period_ns,
                                     wns_banked=wns_banked)

    try:
        v = await _attest_once()
        if (not v.fire and _B1_WRONG_STATE_MARK in v.reason
                and not incomplete and recover_dcp):
            log(f"[b3-admit] session state is not the banked solve "
                f"({v.reason}) — reopening the banked candidate to attest "
                f"the artifact itself.")
            rr = await call_tool("vivado_run_tcl", {
                "command": "open_checkpoint {%s}" % recover_dcp,
                "timeout": 600.0})
            if _tool_error_text(str(rr)):
                v = LogicFloorVerdict(
                    False, "banked-candidate reopen failed (fail closed)")
            else:
                v = await _attest_once()
        log(f"[b3-admit] {'ADMIT' if v.fire else 'refuse'}: {v.reason}"
            + (f" | {v.details[0]}" if v.details else ""))
        return v
    except Exception as e:
        incomplete = True
        return LogicFloorVerdict(False, f"admission raised {e!r} "
                                        f"(fail closed)")
    finally:
        if incomplete:
            try:
                log("[b3-admit] admission incomplete — restarting Vivado to "
                    "clear any wedged/desynced session state.")
                rr = await call_tool("vivado_restart_vivado", {})
                if recover_dcp and not _tool_error_text(str(rr)):
                    await call_tool("vivado_run_tcl", {
                        "command": "open_checkpoint {%s}" % recover_dcp,
                        "timeout": 600.0})
                    log("[b3-admit] session recovered: banked candidate "
                        "reopened.")
            except Exception as re_:
                log(f"[b3-admit] session recovery failed ({re_!r}) — "
                    f"downstream stages fall back to their own wedge "
                    f"handling.")
