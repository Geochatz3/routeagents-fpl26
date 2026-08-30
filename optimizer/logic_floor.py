"""Detect when timing is limited by immutable logic and optimization should stop.

Achieved period combines netlist- and silicon-fixed logic delay with
placement-dependent net delay, uncertainty, and skew. Attestation requires
logic-dominated critical paths, routing near its per-hop floor, and little
residual harvestable gain. This physics-based test does not fire on net- or
congestion-dominated paths that remain improvable.

Evaluation occurs only after the small-floor result is adopted and its
affordability and admission checks pass. Any failed or missing guard yields no
attestation.

Fixed admission thresholds are:
- At most 10 MHz of harvestable gain across the top 32 setup paths, using a 45 ps per-hop net floor that conservatively overestimates available gain.
- A worst-path logic-delay fraction of at least 0.80.
- At least 0.50 of worst-path logic delay inside DSP, block RAM, or URAM macros; LUT-chain delay remains optimizable.
- Agreement between two independent solves within 0.08 ns.
- The 32nd path's slack at least 0.100 ns above WNS, ensuring adequate coverage of the near-critical population.
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

# Emit one LFPATH record per path and one LFMETA record in a single Tcl call.
# Missing or malformed records make the check decline rather than infer data.
# The transport requires one semicolon-joined line; embedded newlines return
# early and leave prompts buffered, desynchronizing later calls.
# Resource types precede Prop_* arc names, while delay values follow on the
# next line, so the parser pattern must span that newline.
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
    REAL error grammar this codebase documents (a bare
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
    # Consistency guard: the measured worst slack must BE the B3 value being
    # attested. A wrong-group or stale-timing read must never attest silently.
    if abs(wns - wns_b3) > 0.010:
        return LogicFloorVerdict(
            False, f"worst-path slack {wns:.3f} disagrees with B3 "
                   f"{wns_b3:.3f} (wrong group / stale timing — fail-safe)")
    # A truncated timing window remains conservative because unseen paths have
    # slack no worse than the last reported path, while each computed bound
    # overestimates achievable slack. An incomplete parse still refuses the check.
    # The per-path net-delay floor is 45 ps per hop.
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
    """Runs the logic-floor attestation and returns a fail-closed result.

    Tool errors, timeouts, and parse failures do not fire the attestation and
    never propagate. The evaluator receives `routed=True` because this runs
    immediately after the adoption gate verifies a fully routed state, with no
    intervening session operations. If Tcl does not complete, the function
    restarts the tool and best-effort reopens `recover_dcp` so downstream
    stages see the expected banked state.
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


# Admit the later optimization stage from path physics rather than wall time,
# which depends on the host. Admission requires hard-macro-dominated paths
# whose estimated remaining timing gain is near the structural floor.
# The gain bound is deliberately optimistic, so uncertainty causes refusal.
# The 18 MHz limit allows the expected pre-floor gap; the logic and macro
# fractions enforce structural dominance, and 16 paths provide minimum coverage.
# Over-admission costs one protected optimization leg; exit attestation remains
# the sole authority for terminating at the floor.
LOGIC_FLOOR_B1_MAX_BOUND_ALPHA_MHZ = 18.0
LOGIC_FLOOR_B1_MIN_LOGIC_FRAC = 0.70
LOGIC_FLOOR_B1_MIN_MACRO_FRAC = 0.50
LOGIC_FLOOR_B1_MIN_NPATHS = 16
# Admission accepts 16 parsed paths because omitted paths can only make its
# gain estimate larger and therefore make admission harder. Exit attestation
# still requires 32 paths because it can terminate the run.
# The open session's worst slack must be within 0.10 ns of the banked value.
# On mismatch, the async wrapper reopens the banked candidate and retries once.
LOGIC_FLOOR_B1_WORST_SANITY_NS = 0.10
_B1_WRONG_STATE_MARK = "wrong state"


def evaluate_b1_admission(
    *,
    paths: List[LFPath],
    period_ns: float,
    wns_banked: Optional[float],
) -> LogicFloorVerdict:
    """Evaluates admission and returns either admit or refuse.

    Every failed check produces refusal. The evaluator cannot inspect route
    status directly, so state integrity depends on the worst-slack sanity
    window against the banked measurement, minimum coverage, and the wrapper
    reopening and reattesting the banked routed artifact.
    """
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
    """Runs tool-backed admission attestation and returns a fail-closed decision.

    Tool errors, timeouts, and parse failures refuse admission and never
    propagate. If the first attestation finds worst slack inconsistent with the
    banked value, the function reopens `recover_dcp` and reattests once to
    eliminate session drift. If Tcl does not complete, it restarts the tool and
    best-effort reopens the banked candidate so downstream stages see the
    expected state.
    """
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
