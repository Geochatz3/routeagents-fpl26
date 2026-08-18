"""INSURED TERMINAL RE-PLACE GAMBLE (jul22 placement flagship, P3+P2 merged).

Panel provenance (build-regardless, 3/3 round-2 flip + 5/5 life-or-death on
-net_delay_weight): late in the wall, from the BANKED best state, gamble on a
full re-place using the last untried placement-layer knob family —
`place_design -directive <X> -net_delay_weight {medium,high}` (+ an optional
`place_design -post_place_opt` variant axis) — then route + measure.

INSURED means:
  - the on-disk banked best DCP is NEVER touched: every draw re-opens it
    read-only into the session and banks its own candidate to a dedicated
    draw-indexed output file, written ONLY on adoption;
  - a failed/unfinished/rejected draw costs nothing but its reserved wall
    slice (never-worse by construction, same discipline as the LASTMILE /
    fanout polish stages and the ILS banked mirror);
  - adoption requires beating the chain-best by >= replace_gamble_adopt_
    margin_ns (+0.15 ns, the round-2 panel gate), fully routed, hold clean
    (whs >= hold_slack_floor_ns — the official scorecard gate passes
    whs=0.0), and a verified (non-error-envelope) write_checkpoint.

FAIL-CLOSED BUDGET: the stage runs only when the terminal reserve affords a
full place+route for THIS design's measured cost anchors (observed Explore
full-ruin ILS cycle cost, else the recipe-phase full-cycle anchor) x1.3
margin + a finalize reserve. NO optimistic cold start (contrast the ILS
picker): an unknown anchor SKIPS the stage — a terminal re-place that cannot
finish has no right tail (terra's condition, adopted by the round-2 panel).

KILL-CRITERION INSTRUMENTATION (measure, don't decide): every draw emits one
structured line

  REPLACE_GAMBLE attempt=<n> variant=<label> pre_wns=<w> place_s=<s>
  route_s=<s> post_wns=<w> post_whs=<h> verdict=<ADOPTED|REJECTED|
  UNAFFORDABLE|ERROR> reason=<...>

plus a stage summary with completed_valid counts. The panel kills the FIRING
policy (not the build) if completion-to-valid < 80% within the terminal
reserve across >= 3 states.

KNOB DOC EVIDENCE (2025.1, UG835 place_design + UG904):
  - -net_delay_weight {low,medium,high} (default low): 2025.1 syntax line +
    UG904 "Using the -net_delay_weight Option"; UG835 lists it as compatible
    with -directive. KEPT (the 5/5 panel knob).
  - -post_place_opt: 2025.1 syntax line; "run optimization after placement
    ... any placement changes will result in unrouted connections, so
    route_design will need to be run after". KEPT as a separate post-place
    step in one variant (it is NOT a per-directive modifier).
  - -clock_vtree_type {balanced,intraSLR,interSLR}: SLR-skew oriented
    ("default option for Versal SSIT devices"; intraSLR/interSLR minimize
    skew within/between SLRs). DROPPED: the contest device is xcvu3p — a
    single-SLR UltraScale+ part — and 2025.1's -directive compatibility
    list does not include it.

This module is PURE Vivado Tcl via the agent's call_tool — no LLM. Exception
-safe at the call site (any failure -> agent keeps its existing best).
Default OFF (`ILSPolishConfig.replace_gamble_enabled = False`): the RC ships
it off; the all-in window (Aug 5-9) flips it on via CLI --replace-gamble or
env FPL26_REPLACE_GAMBLE=1.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional, Tuple

from optimizer.ils_polish import (
    ILS_COMBOS,
    ILSPolishConfig,
    _measure,
    _measure_hold,
    _tool_ok,
)

# ---------------------------------------------------------------------------
# Draw variants
# ---------------------------------------------------------------------------
# (label, place_directive, net_delay_weight, post_place_opt_step)
# Order = draw priority. Rationale:
#  1. Explore + ndw high — the proven strongest full-ruin place directive in
#     the ILS rotation (most historical accepts) crossed with the maximum
#     setting of the untried knob (the 5/5 panel pick: kimi/gemini/grok #1).
#  2. Explore + ndw medium — the round-2 "{med,high}" second draw: same
#     directive, softer pessimism (high can over-congest, UG904 warns).
#  3. ExtraTimingOpt + ndw high — the rotation's second-strongest ruin
#     directive (3 historical accepts) x the new knob.
#  4. Explore + ndw high + -post_place_opt — adds the incremental
#     post-placement optimizer as an extra step before routing (2025.1
#     UG835: run route_design after). Only reached when max_draws is
#     raised above the default 2 and the wall keeps affording draws.
REPLACE_GAMBLE_VARIANTS: List[Tuple[str, str, str, bool]] = [
    ("Explore+ndw_high", "Explore", "high", False),
    ("Explore+ndw_medium", "Explore", "medium", False),
    ("ExtraTimingOpt+ndw_high", "ExtraTimingOpt", "high", False),
    ("Explore+ndw_high+ppo", "Explore", "high", True),
]

# Route directive for every draw: Explore — the combo-0 route directive the
# cost anchor (an observed Explore full-ruin cycle) was measured with, so
# the affordability estimate and the executed protocol stay consistent.
REPLACE_GAMBLE_ROUTE_DIRECTIVE = "Explore"

# Same margin convention as ROUTE_REROLL_COST_MARGIN (x1.3): covers the
# knob/directive cost spread around the measured Explore-cycle sample.
REPLACE_GAMBLE_COST_MARGIN = 1.3

VERDICT_ADOPTED = "ADOPTED"
VERDICT_REJECTED = "REJECTED"
VERDICT_UNAFFORDABLE = "UNAFFORDABLE"
VERDICT_ERROR = "ERROR"


def replace_gamble_cost_basis(cfg: ILSPolishConfig,
                              observed_costs: Optional[dict] = None) -> float:
    """FULL place+route cycle basis (seconds) for the gamble's affordability
    gate. Preference ladder (measured-on-THIS-design/box first):

      1. this run's observed ILS Explore full-ruin cycle cost (combo index
         0 in observed_costs, as recorded by run_ils_polish's combo_cost —
         already x1.15-margined at record time);
      2. cfg.expected_heavy_cycle_s — the recipe-phase full place+route(+
         phys_opt) anchor from derive_cost_anchors();
      3. 0.0 (unknown) -> the stage FAILS CLOSED (skip). No optimistic
         cold start: a terminal re-place that cannot finish has no right
         tail (terra's wall-reserve condition, adopted round-2).
    """
    try:
        explore_idx = [c[0] for c in ILS_COMBOS].index("Explore")
    except ValueError:                                    # pragma: no cover
        explore_idx = 0
    obs = (observed_costs or {}).get(explore_idx)
    if obs and obs > 0:
        return float(obs)
    if cfg.expected_heavy_cycle_s > 0:
        return float(cfg.expected_heavy_cycle_s)
    return 0.0


def replace_gamble_accept(*, new_wns: Optional[float],
                          best_wns: Optional[float],
                          unrouted: Optional[int],
                          whs: Optional[float],
                          cfg: ILSPolishConfig) -> Tuple[bool, str]:
    """Pure adopt decision for a re-place gamble draw.

    Adopt ONLY on: fully routed AND setup beats the chain-best by >=
    replace_gamble_adopt_margin_ns (+0.15, the round-2 panel gate — NOT the
    polish stages' 0.002 never-worse margin: a full re-place discards the
    chain's accumulated phys_opt polish, so a marginal win is likely noise)
    AND hold clean (whs >= hold_slack_floor_ns; the official scorecard gate
    passes whs=0.0 — jul02 preview evidence). A reject is free: the banked
    best on disk was never touched."""
    if unrouted is None or unrouted != 0:
        return False, f"unrouted={unrouted}"
    if new_wns is None or best_wns is None:
        return False, f"wns unavailable (new={new_wns} best={best_wns})"
    margin = cfg.replace_gamble_adopt_margin_ns
    if not (new_wns >= best_wns + margin):
        return False, (f"below adopt margin (w={new_wns} < best {best_wns} "
                       f"+ {margin})")
    if whs is None or whs < cfg.hold_slack_floor_ns:
        return False, (f"hold below floor (whs={whs} < "
                       f"{cfg.hold_slack_floor_ns})")
    return True, f"adopt w={new_wns} (chain-best {best_wns}) whs={whs}"


@dataclass
class ReplaceGambleDraw:
    """Per-draw record — one structured log line each (kill-test evidence)."""
    variant: str = ""
    pre_wns: Optional[float] = None
    place_s: Optional[float] = None
    route_s: Optional[float] = None
    post_wns: Optional[float] = None
    post_whs: Optional[float] = None
    verdict: str = VERDICT_ERROR
    reason: str = ""
    # completion-to-valid (the panel kill criterion's numerator): the draw
    # produced a MEASURED, fully-routed, hold-clean state within the
    # reserve — independent of whether it beat the adopt margin.
    completed_valid: bool = False

    def log_line(self, attempt: int) -> str:
        def _f(v):
            return "None" if v is None else (f"{v:.3f}"
                                             if isinstance(v, float) else str(v))
        return (f"REPLACE_GAMBLE attempt={attempt} variant={self.variant} "
                f"pre_wns={_f(self.pre_wns)} place_s={_f(self.place_s)} "
                f"route_s={_f(self.route_s)} post_wns={_f(self.post_wns)} "
                f"post_whs={_f(self.post_whs)} verdict={self.verdict} "
                f"reason={self.reason}")


@dataclass
class ReplaceGambleResult:
    attempted: bool = False
    skip_reason: str = ""
    adopted: bool = False
    best_wns: Optional[float] = None      # chain-best, ratcheted on adopt
    best_path: Optional[str] = None       # adopted DCP path (None = no adopt)
    # jul23 INSURED-COMPARE MUX wiring: the best VERIFIED (fully routed +
    # hold-clean) draw REGARDLESS of the +0.15 adopt ratchet. A draw that
    # improved on the chain-best but sat BELOW the re-adopt bar is discarded
    # by the in-run adopt logic (a full re-place forfeits the chain's
    # accumulated polish, so a marginal win is likely noise), yet it is
    # still a legitimate FINAL candidate — the finalize MUX's 0.005 argmax
    # decides it against the pipeline. Persisted to a draw-indexed
    # _cand_d{n}.dcp (the banked best on disk is STILL never touched). None
    # when no draw produced a verified routed+hold-clean state.
    best_draw_path: Optional[str] = None
    best_draw_wns: Optional[float] = None
    best_draw_whs: Optional[float] = None
    draws: List[ReplaceGambleDraw] = field(default_factory=list)

    @property
    def completed_valid(self) -> int:
        return sum(1 for d in self.draws if d.completed_valid)

    def summary(self) -> str:
        if not self.attempted:
            return f"REPLACE_GAMBLE SKIPPED ({self.skip_reason})"
        return (f"REPLACE_GAMBLE summary attempts={len(self.draws)} "
                f"completed_valid={self.completed_valid} "
                f"adopted={int(self.adopted)} best_wns={self.best_wns}")


CallTool = Callable[[str, dict], Awaitable[str]]


def replace_gamble_should_run(*, cfg: ILSPolishConfig,
                              best_wns: Optional[float],
                              best_path: Optional[str],
                              remaining_s: float,
                              cost_basis_s: float) -> Tuple[bool, str]:
    """Pure firing gate (unit-testable, no I/O). Returns (run, reason).

    Order matters: the kill switch and state checks precede the class/budget
    gates so a disabled stage is a guaranteed zero-diff no-op."""
    if not cfg.replace_gamble_enabled:
        return False, "disabled (kill switch; RC default OFF)"
    if best_wns is None or not best_path:
        return False, f"no banked best (wns={best_wns} path={best_path})"
    # Class gate (round-2 gate-loosening: route-delay fraction > 0.6,
    # loosened from 0.7). Phase 1 does not measure this feature yet, so the
    # plumbed value is normally None -> FAIL-OPEN (build-regardless: the
    # all-in window judges by the measured completion/adopt log lines).
    frac = cfg.critical_path_route_delay_frac
    if frac is not None and frac < cfg.replace_gamble_min_route_delay_frac:
        return False, (f"route_delay_frac={frac:.2f} < "
                       f"{cfg.replace_gamble_min_route_delay_frac} "
                       f"(placement-jump class gate)")
    if cost_basis_s <= 0:
        return False, "no measured cost anchor (fail closed)"
    need = (cost_basis_s * REPLACE_GAMBLE_COST_MARGIN
            + cfg.replace_gamble_finalize_reserve_s)
    if remaining_s < need:
        return False, (f"insufficient terminal reserve (remaining "
                       f"{remaining_s:.0f}s < need {need:.0f}s)")
    return True, (f"armed (anchor {cost_basis_s:.0f}s x"
                  f"{REPLACE_GAMBLE_COST_MARGIN} + reserve "
                  f"{cfg.replace_gamble_finalize_reserve_s:.0f}s fits "
                  f"{remaining_s:.0f}s)")


async def run_replace_gamble(
    call_tool: CallTool,
    *,
    best_dcp_path: str,
    best_wns: Optional[float],
    deadline_ts: float,
    wns_tcl: str,
    cfg: ILSPolishConfig,
    log: Callable[[str], None],
    out_dcp_base: str,
    observed_costs: Optional[dict] = None,
) -> ReplaceGambleResult:
    """Run up to cfg.replace_gamble_max_draws insured re-place draws.

    best_dcp_path is the banked best DCP on disk — it is ONLY ever passed to
    open_checkpoint, never written. Each adopted draw banks to
    f"{out_dcp_base}_d{n}.dcp" (draw-indexed: a later failed write can never
    clobber an earlier adopted file). The adopt bar starts at the chain-best
    and RATCHETS to each adopted draw's wns (monotone, never-worse)."""
    r = ReplaceGambleResult(best_wns=best_wns)
    basis = replace_gamble_cost_basis(cfg, observed_costs)
    run, why = replace_gamble_should_run(
        cfg=cfg, best_wns=best_wns, best_path=best_dcp_path,
        remaining_s=deadline_ts - time.time(), cost_basis_s=basis)
    if not run:
        r.skip_reason = why
        log(f"replace-gamble: skipped ({why})")
        return r
    r.attempted = True
    log(f"replace-gamble: {why}; chain-best wns={best_wns}, adopt margin "
        f"+{cfg.replace_gamble_adopt_margin_ns}ns, max_draws="
        f"{cfg.replace_gamble_max_draws}")

    def _heavy_to() -> float:
        return max(300.0, min(cfg.heavy_cmd_timeout_s,
                              deadline_ts - time.time()))

    def _light_to() -> float:
        return max(120.0, min(cfg.measure_cmd_timeout_s,
                              deadline_ts - time.time()))

    async def _step(cmd: str, to: float) -> None:
        resp = await call_tool("vivado_run_tcl",
                               {"command": cmd, "timeout": to})
        if not _tool_ok(resp):
            _txt = str(resp)
            _eline = next((l for l in _txt.splitlines() if "TCL ERROR" in l),
                          "")
            raise RuntimeError(
                f"{cmd.split()[0]} failed: {(_eline or _txt)[:400]}")

    # Fresh Vivado before the first draw (best-effort): (a) the LASTMILE
    # polish stage may have poisoned the session — after a LastMile place,
    # full place_design fails outright until restart (jun12 repro); (b) a
    # fresh session places better AND faster (jun07 A/B/C). On failure we
    # proceed — a poisoned place surfaces as a caught ERROR draw.
    try:
        _rr = await call_tool("vivado_restart_vivado", {})
        if _tool_ok(_rr):
            log("replace-gamble: Vivado restarted fresh (session hygiene "
                "before full re-place).")
        else:
            log(f"replace-gamble: restart returned error "
                f"({str(_rr)[:100]}); continuing in current session.")
    except Exception as e:
        log(f"replace-gamble: restart failed ({e!r}); continuing.")

    error_streak = 0
    for i, (label, pd, ndw, ppo) in enumerate(
            REPLACE_GAMBLE_VARIANTS[:max(0, cfg.replace_gamble_max_draws)],
            start=1):
        d = ReplaceGambleDraw(variant=label, pre_wns=r.best_wns)
        # Per-draw affordability re-check against the SAME measured basis:
        # a draw that cannot finish inside the terminal reserve is never
        # started (fail closed), it does not "try and see".
        need = (basis * REPLACE_GAMBLE_COST_MARGIN
                + cfg.replace_gamble_finalize_reserve_s)
        remaining = deadline_ts - time.time()
        if remaining < need:
            d.verdict = VERDICT_UNAFFORDABLE
            d.reason = (f"remaining {remaining:.0f}s < need {need:.0f}s "
                        f"(anchor {basis:.0f}s x{REPLACE_GAMBLE_COST_MARGIN} "
                        f"+ reserve)")
            r.draws.append(d)
            log(d.log_line(i))
            break
        try:
            # The banked best is opened READ-ONLY into the session; it is
            # never a write_checkpoint target anywhere in this stage.
            await _step(f"open_checkpoint {{{best_dcp_path}}}", _light_to())
            _t_place = time.time()
            await _step("place_design -unplace", _heavy_to())
            await _step(f"place_design -directive {pd} "
                        f"-net_delay_weight {ndw}", _heavy_to())
            if ppo:
                # 2025.1 UG835: incremental post-commit optimizer; placement
                # changes unroute connections -> the route step below covers
                # the required re-route.
                await _step("place_design -post_place_opt", _heavy_to())
            d.place_s = time.time() - _t_place
            _t_route = time.time()
            await _step(f"route_design -directive "
                        f"{REPLACE_GAMBLE_ROUTE_DIRECTIVE}", _heavy_to())
            d.route_s = time.time() - _t_route
            w, ur = await _measure(call_tool, wns_tcl, timeout_s=_light_to())
            if ur != 0:
                # One incremental completion pass (LASTMILE-stage pattern:
                # fresh routes sometimes leave a few nets; a bare
                # route_design usually finishes them). Still never-worse —
                # a second failure just rejects below.
                await call_tool("vivado_run_tcl",
                                {"command": "route_design",
                                 "timeout": _heavy_to()})
                w, ur = await _measure(call_tool, wns_tcl,
                                       timeout_s=_light_to())
            whs = await _measure_hold(call_tool, timeout_s=_light_to())
            d.post_wns, d.post_whs = w, whs
            d.completed_valid = (
                w is not None and ur == 0
                and whs is not None and whs >= cfg.hold_slack_floor_ns)
            ok, why = replace_gamble_accept(
                new_wns=w, best_wns=r.best_wns, unrouted=ur, whs=whs,
                cfg=cfg)
            if ok:
                out_dcp = f"{out_dcp_base}_d{i}.dcp"
                _wr = await call_tool(
                    "vivado_run_tcl",
                    {"command": f"write_checkpoint -force {{{out_dcp}}}",
                     "timeout": _heavy_to()})
                if not _tool_ok(_wr):
                    # NEVER bank on a Tcl error: the file on disk is stale/
                    # partial; adopting would ship a WNS the DCP doesn't
                    # have (jun12 phantom-accept lesson).
                    d.verdict = VERDICT_ERROR
                    d.reason = (f"write_checkpoint failed "
                                f"({str(_wr)[:80]}); draw discarded")
                else:
                    d.verdict = VERDICT_ADOPTED
                    d.reason = why
                    r.adopted = True
                    r.best_wns = w
                    r.best_path = out_dcp
                    # jul23 MUX: an adopted draw is trivially the best
                    # VERIFIED draw so far (it cleared the ratchet). Track
                    # it as the finalize-MUX candidate too — ADDITIVE, the
                    # adopt/mirror-repoint above is unchanged, and no extra
                    # write (the _d{i} file was just banked).
                    if r.best_draw_wns is None or w > r.best_draw_wns:
                        r.best_draw_path = out_dcp
                        r.best_draw_wns = w
                        r.best_draw_whs = whs
            else:
                d.verdict = VERDICT_REJECTED
                d.reason = why
                # jul23 MUX: a VERIFIED (routed + hold-clean) draw that fell
                # BELOW the +0.15 re-adopt ratchet is still a legitimate
                # FINAL candidate. Persist the best such draw to a dedicated
                # _cand_d{n} file (the banked best on disk is STILL never a
                # write target) so the finalize MUX can ship it via its
                # 0.005 argmax — the panel's "loosen selection, insure with
                # the MUX" intent. Written only while it is the running-best
                # verified draw (one checkpoint per genuine improvement), in
                # this iteration BEFORE the next draw re-opens the banked
                # best and clobbers the session.
                if (d.completed_valid
                        and (r.best_draw_wns is None or w > r.best_draw_wns)):
                    cand_dcp = f"{out_dcp_base}_cand_d{i}.dcp"
                    _wc = await call_tool(
                        "vivado_run_tcl",
                        {"command": f"write_checkpoint -force {{{cand_dcp}}}",
                         "timeout": _heavy_to()})
                    if _tool_ok(_wc):
                        r.best_draw_path = cand_dcp
                        r.best_draw_wns = w
                        r.best_draw_whs = whs
                    else:
                        log(f"replace-gamble: MUX-candidate write_checkpoint "
                            f"failed ({str(_wc)[:80]}); draw {i} not surfaced "
                            f"to the finalize MUX (banked best untouched).")
            error_streak = 0
        except Exception as e:
            d.verdict = VERDICT_ERROR
            d.reason = f"{e!r}"[:200]
            error_streak += 1
        r.draws.append(d)
        log(d.log_line(i))
        if error_streak >= 2:
            log("replace-gamble: two consecutive draw errors — stopping "
                "(banked best untouched).")
            break
    # Leave the SESSION holding the shipping state: disk truth is what
    # finalize ships, but downstream stages assume session == best.
    try:
        _final = r.best_path if r.adopted else best_dcp_path
        await call_tool("vivado_run_tcl",
                        {"command": f"open_checkpoint {{{_final}}}",
                         "timeout": _light_to()})
    except Exception:
        pass
    log(r.summary())
    return r
