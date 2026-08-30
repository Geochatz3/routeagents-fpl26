"""Provide an optional, insured terminal full-re-placement stage.

The stage is disabled by default. Each variant reopens the banked checkpoint
read-only, applies a placement directive and net-delay weight with optional
post-placement optimization, then routes and measures the result. Candidates
use dedicated draw-indexed files, and only adopted candidates are written as
outputs. Adoption requires exceeding the configured margin, completing routing,
passing hold checks, and verifying the checkpoint write. The budget gate
requires measured placement and routing costs, safety margin, and finalization
reserve; unknown costs skip the stage. Structured telemetry records each
variant, timing, duration, and verdict. The implementation uses Tcl through the
tool interface without model calls, and exceptions leave the existing best
checkpoint untouched.
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

# Variants are ordered by expected timing benefit under the draw budget.
# High net-delay weighting is tried before medium weighting but may increase
# congestion. The final variant adds post-placement optimization and must
# still be followed by routing.
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
    """Return the full placement-and-routing cost basis in seconds for the
    affordability gate.

    Use `observed_costs` entry 0 when available; it is the current
    environment's observed full-cycle cost and already includes the 1.15
    overrun allowance. Otherwise use `cfg.expected_heavy_cycle_s`, the derived
    heavy-cycle estimate. Return `0.0` when neither anchor is known; this
    sentinel makes the terminal stage fail closed.
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
    replace_gamble_adopt_margin_ns (+0.15 — NOT the
    polish stages' 0.002 never-worse margin: a full re-place discards the
    chain's accumulated phys_opt polish, so a marginal win is likely noise)
    AND hold clean (whs >= hold_slack_floor_ns; the official scorecard gate
    passes whs=0.0 — preview evidence). A reject is free: the banked
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
    # completion-to-valid (the kill criterion's numerator): the draw
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
    # Track the best fully routed, hold-clean draw independently of the in-run
    # adoption threshold. A marginal draw may be unsuitable for continuing the
    # chain but remain a valid final candidate. Store it in a draw-specific
    # checkpoint without overwriting the banked best; finalization compares both.
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
    # Apply the route-delay gate only when Phase 1 provides the metric;
    # a missing value fails open.
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

    # Restart Vivado before the first draw because late-stage placement can
    # leave the session unable to run full placement. Restart failure is
    # non-fatal; subsequent draw validation rejects tool failures.
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
                # A completion pass can route residual nets left by a fresh route.
                # Reject the draw if completion or validation still fails.
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
                    # Never bank on a Tcl error: the file on disk is stale or
                    # partial, and adopting would ship a WNS the DCP does not
                    # have — the phantom-accept failure mode.
                    d.verdict = VERDICT_ERROR
                    d.reason = (f"write_checkpoint failed "
                                f"({str(_wr)[:80]}); draw discarded")
                else:
                    d.verdict = VERDICT_ADOPTED
                    d.reason = why
                    r.adopted = True
                    r.best_wns = w
                    r.best_path = out_dcp
                    # An adopted draw is also eligible for final selection. Its
                    # checkpoint is already banked, so no additional write is
                    # needed.
                    if r.best_draw_wns is None or w > r.best_draw_wns:
                        r.best_draw_path = out_dcp
                        r.best_draw_wns = w
                        r.best_draw_whs = whs
            else:
                d.verdict = VERDICT_REJECTED
                d.reason = why
                # Routed, hold-clean draws remain final candidates even when
                # they miss the re-adoption threshold. Store only each
                # improving candidate in a dedicated checkpoint; never
                # overwrite the banked best. Write it before the next draw
                # reopens the banked checkpoint and replaces session state.
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
