"""Adaptive banked tail controller — pure policy (jul22, PLAN item 1).

Upgrades the fixed bare-reroute tail loop to a measured-Δns/s PORTFOLIO
policy over the PROVEN move menu, with per-class harvest-to-wall on
deep-WNS designs.  This module is pure (no IO, no Vivado, no optimizer
state) so the policy is exhaustively unit-testable; execution lives in
DCPOptimizer._run_tail_controller.

Evidence base (plateau probe jul21 + chain cell, boom −11.177,
final_round/plateau_probe_jul21.md):

  move            Δns     wall s   prior rate ns/s
  m1_route       +0.404   2082     1.94e-4  (live: +0.215 banked eval,
                                             DEBUG-WALL loop iterated
                                             −11.309→−10.557 jul21/22)
  m4_ladder      +0.431   1668     2.58e-4  (monotone per-pass)
  m3_fanout      +0.369   1772     2.08e-4
  m2_physopt_ae  +0.595   4012     1.48e-4
  CHAIN M2→M1→M3 = +0.726 cumulative (hold +0.010): moves RE-BITE after
  another move changes the state — hence accept ⇒ un-retire the others.

Near-met designs: every move decays to no-op/marginal (fir/logicnets/
optical rows) → the controller only ARMS on deep-WNS states
(best_wns ≤ deep_wns_threshold; round-2 loosened gate −1.0).  There the
wall policy is HARVEST-TO-WALL: γ-thrift is anti-P(top-5) when the
gradient is ~0.1 ns per 15 min and γ costs ≤3% of α (fable-maxthink
jul22) — the loop stops only on wall floor, plateau (all moves retired
with nothing un-retiring them), or the runaway caps.

Fail-closed: any controller-internal error makes the caller fall back to
the plain shipped M1 bare-reroute loop (banked best intact either way).

Hold discipline: M1 rides the auto-bank hook unchanged (bare route_design
was hold-neutral in every probe cell and live run).  phys_opt moves are
banked ONLY through the stage's own gate: WNS improved AND fully routed
AND hold_accept() below (h0-relative floor, plateau-probe discipline) —
the auto-bank hook is SUPPRESSED around them because it does not check
hold, and banking a hold-dirty state would fail the validator (α=0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# Per-move re-execution cap (mirrors BARE_REROUTE_MAX_ITERS_DEFAULT=4):
# bounds the accept→un-retire→fail alternation; the observed decay and the
# global max_moves cap normally stop the loop first.
TAIL_CTRL_MAX_EXECS_PER_MOVE = 4

# h0-relative hold epsilon (plateau probe jul21 discipline:
# whs1 ≥ min(whs0, 0) − 0.001 — never make hold meaningfully worse, and
# never gate on pre-existing positive margin a move did not create).
TAIL_CTRL_HOLD_EPS_NS = 0.001


@dataclass(frozen=True)
class TailMove:
    """One proven tail move.

    cost_factor_vs_route: fallback cost predictor as a multiple of the
    PRESERVING bare-route prediction when the move has not yet been
    observed this run (probe ratios on boom: ladder 1668/2082≈0.80,
    fanout 1772/2082≈0.85, phys_opt AE 4012/2082≈1.93 — rounded UP,
    prefer-refusing).
    """
    key: str
    cmds: Tuple[str, ...]
    prior_gain_ns: float
    prior_cost_s: float
    cost_factor_vs_route: float
    rides_autobank: bool  # True only for the bare route move


TAIL_MENU: Tuple[TailMove, ...] = (
    TailMove("m1_route", ("route_design",),
             0.404, 2082.0, 1.0, True),
    TailMove("m4_ladder", ("phys_opt_design -critical_pin_opt",
                           "phys_opt_design -placement_opt",
                           "phys_opt_design -restruct_opt",
                           "phys_opt_design -critical_cell_opt"),
             0.431, 1668.0, 0.9, False),
    TailMove("m3_fanout", ("phys_opt_design -directive AggressiveFanoutOpt",),
             0.369, 1772.0, 0.9, False),
    TailMove("m2_physopt_ae", ("phys_opt_design -directive AggressiveExplore",),
             0.595, 4012.0, 2.0, False),
)


@dataclass
class MoveState:
    executed: int = 0
    last_gain_ns: Optional[float] = None
    last_cost_s: Optional[float] = None
    retired: bool = False
    errors: int = 0


def new_states() -> Dict[str, MoveState]:
    return {m.key: MoveState() for m in TAIL_MENU}


def expected_rate(move: TailMove, st: MoveState) -> float:
    """Expected Δns/s for the NEXT execution of this move.

    Before the first observation this run: the plateau-probe prior.
    After: last observed gain/cost (state-local — the probe showed rates
    shift as the state moves; the freshest sample is the best predictor).
    An un-retired move has last_gain_ns reset to None (re-explore from
    the prior: its stale below-min observation predates the state change
    that un-retired it).
    """
    if st.last_gain_ns is not None and st.last_cost_s:
        return st.last_gain_ns / max(st.last_cost_s, 1.0)
    return move.prior_gain_ns / max(move.prior_cost_s, 1.0)


def pick_next(states: Dict[str, MoveState],
              affordable: Dict[str, bool],
              max_execs: int = TAIL_CTRL_MAX_EXECS_PER_MOVE,
              ) -> Optional[str]:
    """argmax expected-rate over non-retired, affordable, under-cap moves.

    Ties break by TAIL_MENU order (M1 first — the proven backbone).
    Returns None when no move is eligible (caller stops: plateau or wall).
    """
    best_key: Optional[str] = None
    best_rate = float("-inf")
    for m in TAIL_MENU:
        st = states[m.key]
        if st.retired or st.executed >= max_execs:
            continue
        if not affordable.get(m.key, False):
            continue
        r = expected_rate(m, st)
        if r > best_rate:
            best_rate = r
            best_key = m.key
    return best_key


def record_result(states: Dict[str, MoveState], key: str,
                  gain_ns: float, cost_s: float,
                  min_gain_ns: float,
                  errored: bool = False) -> None:
    """Update the ledger after one execution.

    Retirement: gain < min_gain (or an error) retires the move.
    Un-retirement: an ACCEPTED move (gain ≥ min_gain) changed the state —
    chain evidence (M2→M1→M3 additive) says previously-dead moves may
    re-bite, so every OTHER move is un-retired with its stale observation
    cleared (falls back to prior rate).
    """
    st = states[key]
    st.executed += 1
    st.last_gain_ns = gain_ns
    st.last_cost_s = cost_s if cost_s > 0 else st.last_cost_s
    if errored:
        st.errors += 1
        st.retired = True
        return
    if gain_ns < min_gain_ns:
        st.retired = True
        return
    # Accepted: un-retire the rest (state changed).
    for other_key, other in states.items():
        if other_key == key:
            continue
        if other.retired:
            other.retired = False
            other.last_gain_ns = None


def hold_accept(new_whs: Optional[float], base_whs: Optional[float],
                eps_ns: float = TAIL_CTRL_HOLD_EPS_NS) -> bool:
    """h0-relative hold floor: whs_after ≥ min(whs_before, 0) − eps.

    Unmeasurable hold (None) REJECTS — fail-closed: the validator gates
    hold_passed and a blind bank risks α=0 on the whole design.
    """
    if new_whs is None:
        return False
    base = base_whs if base_whs is not None else 0.0
    return new_whs >= min(base, 0.0) - eps_ns


def predict_move_cost_s(move: TailMove, st: MoveState,
                        route_prediction_s: float) -> float:
    """Cost predictor for the NEXT execution: observed-this-run beats the
    route-scaled fallback (same freshest-sample principle as the M1
    loop's refresh-with-observed-cost rule)."""
    if st.last_cost_s:
        return st.last_cost_s
    return route_prediction_s * move.cost_factor_vs_route


def move_by_key(key: str) -> TailMove:
    for m in TAIL_MENU:
        if m.key == key:
            return m
    raise KeyError(key)
