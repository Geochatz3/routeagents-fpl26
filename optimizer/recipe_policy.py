"""Define feature-based admission policy for deterministic optimization recipes.

Routes designs to timing bands and sub-bands using measured WNS magnitude,
failing endpoints, achievable-Fmax ratio, and critical-path spread. Admission
depends only on measured features, never on design identity.
"""
from __future__ import annotations

import logging
import os

# Log through the orchestrator's logger: this code still belongs to the same
# run, and a module-named logger would change every line it emits.
logger = logging.getLogger("dcp_optimizer")


ETO_RETIME_FF_DRIFT_MAX_FRAC = 0.01


ETO_RETIME_WNS_MAG_MAX_NS = 1.05  # == RECIPE_PASS_SHALLOW_WNS_MAG_MAX_NS


ETO_RETIME_WNS_MAG_MIN_NS = 0.60


FRESH_PRESWEEP_COST_SAFETY = 1.3        # observed-cost inflation gating 2..K


FRESH_PRESWEEP_DRAWS_MAX = 3            # K clamp 1..3


OWNFRONT_RETIME_SPLIT_NS = 1.00         # [min, split) -> ETO; [split, max] -> WLD


OWNFRONT_RETIME_WNS_MAG_MAX_NS = 1.05   # == RECIPE_PASS_SHALLOW_WNS_MAG_MAX_NS


OWNFRONT_RETIME_WNS_MAG_MIN_NS = 0.90   # raised floor (was 0.60): parity


RECIPE_PASS_DEEP_WNS_MAG_MIN_NS = 8.0


RECIPE_PASS_SHALLOW_WNS_MAG_MAX_NS = 1.05


SHALLOW_DET_WNS_MAG_MAX_NS = 0.90          # EXCLUSIVE — the ownfront raised


SHALLOW_DET_WNS_MAG_MIN_NS = 0.60          # == ETO_RETIME_WNS_MAG_MIN_NS


SUBBAND_CARVEOUT_WNS_MAG_MAX_NS = 0.50


SUBBAND_FLOOR_MAX_FAILING_ENDPOINTS = 1_000


SUBBAND_FLOOR_WNS_MAG_MAX_NS = 0.45


SUBBAND_FLOOR_WNS_MAG_MIN_NS = 0.10


FRESH_PRESWEEP_DRAWS_DEFAULT = 0        # OFF


from typing import Optional


def eto_retime_parse_ffcount(res):
    """Parse the FFCOUNT= sentinel token out of the probe's stdout.
    Returns int or None; anything without an anchored
    FFCOUNT=<int> token is unmeasurable — the audit caller fails CLOSED.
    Unanchored numbers in interleaved Vivado chatter never bind."""
    if not isinstance(res, str):
        return None
    for token in res.strip().split():
        token = token.strip()
        if token.startswith("FFCOUNT="):
            try:
                return int(token[len("FFCOUNT="):])
            except ValueError:
                return None
    return None


def deep_first_sizegated_enabled() -> bool:
    """FPL26_DEEP_FIRST_SIZEGATED master flag (DEFAULT OFF).

    Kill switch FPL26_NO_DEEP_FIRST_SIZEGATED wins, per house convention.
    With the flag unset every path below is unreachable and deep-replace's
    FIRST decision is byte-identical to the shipped baseline."""
    _truthy = ("1", "true", "on", "yes")
    if (os.environ.get("FPL26_NO_DEEP_FIRST_SIZEGATED", "")
            .strip().lower() in _truthy):
        return False
    return (os.environ.get("FPL26_DEEP_FIRST_SIZEGATED", "")
            .strip().lower() in _truthy)


def positive_slack_continue_enabled() -> bool:
    """FPL26_POSITIVE_SLACK_CONTINUE master flag (DEFAULT OFF).

    Kill switch FPL26_NO_POSITIVE_SLACK_CONTINUE wins, per house convention.
    With the flag unset both zero-crossing exits behave byte-identically to
    the shipped baseline."""
    _truthy = ("1", "true", "on", "yes")
    if (os.environ.get("FPL26_NO_POSITIVE_SLACK_CONTINUE", "")
            .strip().lower() in _truthy):
        return False
    return (os.environ.get("FPL26_POSITIVE_SLACK_CONTINUE", "")
            .strip().lower() in _truthy)


def positive_slack_entry_decision(initial_wns, flag_on: bool) -> str:
    """Classify the entry action when input timing may already be met.

    `not_met` selects the normal path for negative or unmeasured WNS.
    `early_exit` writes the input, finalizes, and returns when timing is met
    but optimization is disabled. `optimize` continues through the normal
    pipeline when timing is met and optimization is enabled.
    """
    if initial_wns is None:
        return "not_met"
    try:
        if float(initial_wns) < 0.0:
            return "not_met"
    except (TypeError, ValueError):
        return "not_met"
    return "optimize" if flag_on else "early_exit"


def plateau_armed_from_loop_start(plateau_preloop_fix: bool, initial_wns,
                                  best_wns, positive_slack_entry_met: bool
                                  ) -> bool:
    """Should the >=3-iteration plateau exit be armed from iteration 1?

    The plateau guard needs `last_improvement_iter > 0` OR this. On a
    MET-TIMING ENTRY nothing ever improves off a positive baseline, so
    `last_improvement_iter` stays 0 forever and the guard is STRUCTURALLY DEAD
    — a review proved the loop's real bound on that class was
    max_iterations=50 plus the $0.75 LLM exit, not the "handful of iterations"
    the feature's own commit message claimed. A met-timing entry is exactly the
    shape `_pre_loop_best_banked` was invented for ("the best is already banked,
    last_improvement_iter is pinned at 0"), so it arms the same exit.

    The pre-existing FPL26_PLATEAU_PRELOOP_FIX branch is unchanged.
    """
    if positive_slack_entry_met:
        return True
    return bool(plateau_preloop_fix
                and initial_wns is not None
                and best_wns is not None
                and best_wns > initial_wns)


def ownfront_retime_enabled() -> bool:
    """FPL26_OWNFRONT_RETIME_CANDIDATE master flag (DEFAULT OFF).
    Kill switch FPL26_NO_OWNFRONT_RETIME_CANDIDATE wins, per house
    convention.  Only consulted inside the shallow RECIPE_PASS, so the
    master FPL26_RECIPE_PASS / FPL26_NO_RECIPE_PASS pair implicitly
    gates it.  When True the ETO candidate DEFERS (one retime
    candidate per run); when False/killed the run reverts to the prior
    shipped behavior byte-identically."""
    _truthy = ("1", "true", "on", "yes")
    if (os.environ.get("FPL26_NO_OWNFRONT_RETIME_CANDIDATE", "")
            .strip().lower() in _truthy):
        return False
    return (os.environ.get("FPL26_OWNFRONT_RETIME_CANDIDATE", "")
            .strip().lower() in _truthy)


def ownfront_retime_front(wns_in):
    """Select the retiming strategy for a failing input in the own-front band.

    Returns `eto` for |WNS| in [0.60, 1.00) ns and `wld` for [1.00, 1.05] ns.
    Returns `None` for missing or non-negative WNS and for values outside the band.
    """
    if wns_in is None:
        return None
    try:
        w = float(wns_in)
    except (TypeError, ValueError):
        return None
    if w >= 0.0:
        return None
    mag = abs(w)
    if OWNFRONT_RETIME_WNS_MAG_MIN_NS <= mag < OWNFRONT_RETIME_SPLIT_NS:
        return "eto"
    if OWNFRONT_RETIME_SPLIT_NS <= mag <= OWNFRONT_RETIME_WNS_MAG_MAX_NS:
        return "wld"
    return None


def shallow_determinizer_enabled() -> bool:
    """FPL26_SHALLOW_DETERMINIZER_CANDIDATE master flag (DEFAULT OFF).
    Kill switch FPL26_NO_SHALLOW_DETERMINIZER_CANDIDATE wins, per house
    convention.  Only consulted inside the shallow RECIPE_PASS, so the
    master FPL26_RECIPE_PASS / FPL26_NO_RECIPE_PASS pair implicitly
    gates it.  Off/killed = zero new log lines, zero new IO —
    byte-identical to the prior revision."""
    _truthy = ("1", "true", "on", "yes")
    if (os.environ.get("FPL26_NO_SHALLOW_DETERMINIZER_CANDIDATE", "")
            .strip().lower() in _truthy):
        return False
    return (os.environ.get("FPL26_SHALLOW_DETERMINIZER_CANDIDATE", "")
            .strip().lower() in _truthy)


def shallow_det_subband_match(wns_in) -> bool:
    """Report whether measured failing WNS lies in the determinizer sub-band.

    The lower magnitude bound is inclusive and the upper bound is exclusive,
    making this band adjacent to the raised own-front band without a gap or
    overlap. Missing or non-negative WNS returns `False`. Eligibility depends
    only on measured characteristics, never design identifiers.
    """
    if wns_in is None:
        return False
    try:
        w = float(wns_in)
    except (TypeError, ValueError):
        return False
    if w >= 0.0:
        return False
    return SHALLOW_DET_WNS_MAG_MIN_NS <= abs(w) < SHALLOW_DET_WNS_MAG_MAX_NS


def fir_subband_floor_enabled() -> bool:
    """FPL26_SUBBAND_PHYSOPT_FLOOR master flag (DEFAULT OFF).  Kill switch
    FPL26_NO_SUBBAND_PHYSOPT_FLOOR wins, per house convention."""
    _truthy = ("1", "true", "on", "yes")
    if (os.environ.get("FPL26_NO_SUBBAND_PHYSOPT_FLOOR", "")
            .strip().lower() in _truthy):
        return False
    return (os.environ.get("FPL26_SUBBAND_PHYSOPT_FLOOR", "")
            .strip().lower() in _truthy)


def fir_subband_match(wns_in) -> bool:
    """Report whether measured failing WNS lies in the bounded shallow sub-band.

    Both magnitude bounds are inclusive and expressed in nanoseconds by the
    corresponding constants. Missing or non-negative WNS returns `False`.
    """
    if wns_in is None:
        return False
    try:
        w = float(wns_in)
    except (TypeError, ValueError):
        return False
    if w >= 0.0:
        return False
    return SUBBAND_FLOOR_WNS_MAG_MIN_NS <= abs(w) <= SUBBAND_FLOOR_WNS_MAG_MAX_NS


def fir_like_failing_ok(failing_endpoints) -> bool:
    """Check whether the failing-endpoint count permits specialized sub-band
    handling.

    Returns `True` only for a present, parseable count at or below
    `SUBBAND_FLOOR_MAX_FAILING_ENDPOINTS`. Missing or unparseable values return
    `False`, preserving the normal path when eligibility is uncertain.
    """
    if failing_endpoints is None:
        return False
    try:
        return int(failing_endpoints) <= SUBBAND_FLOOR_MAX_FAILING_ENDPOINTS
    except (TypeError, ValueError):
        return False


def subband_carveout_match(wns_in, failing_endpoints) -> bool:
    """Carve-out band: failing design, |wns_in| <
    SUBBAND_CARVEOUT_WNS_MAG_MAX_NS (0.50 — wider than the floor band), AND
    not measurably NON-fir-like.

    None-direction: at the CARVE-OUT an unmeasured
    failing count keeps the PROTECTION (treat as fir-like) — the carve-out
    guards against the shallow pass's reset, the one proven
    mechanism for scoring below the baseline, and stripping a protection on
    missing data would make the code MORE exposed than its
    predecessor exactly on a failing-parse miss
    (static_parsers.py leaves failing=None with wns intact).  Contrast the
    FLOOR gate, where None correctly refuses the monoculture treatment."""
    if failing_endpoints is not None and not fir_like_failing_ok(
            failing_endpoints):
        return False
    if wns_in is None:
        return False
    try:
        w = float(wns_in)
    except (TypeError, ValueError):
        return False
    if w >= 0.0:
        return False
    return abs(w) < SUBBAND_CARVEOUT_WNS_MAG_MAX_NS


def eto_retime_candidate_enabled() -> bool:
    """FPL26_ETO_RETIME_CANDIDATE master flag (DEFAULT OFF).  Kill switch
    FPL26_NO_ETO_RETIME_CANDIDATE wins, per house convention.  Only
    consulted inside the shallow RECIPE_PASS, so the master
    FPL26_RECIPE_PASS / FPL26_NO_RECIPE_PASS pair implicitly gates it."""
    _truthy = ("1", "true", "on", "yes")
    if (os.environ.get("FPL26_NO_ETO_RETIME_CANDIDATE", "")
            .strip().lower() in _truthy):
        return False
    return (os.environ.get("FPL26_ETO_RETIME_CANDIDATE", "")
            .strip().lower() in _truthy)


def eto_retime_subband_match(wns_in) -> bool:
    """Report whether measured failing WNS lies in the retiming-candidate
    sub-band.

    Both magnitude bounds are inclusive and expressed in nanoseconds by the
    corresponding constants. Missing or non-negative WNS returns `False`.
    """
    if wns_in is None:
        return False
    try:
        w = float(wns_in)
    except (TypeError, ValueError):
        return False
    if w >= 0.0:
        return False
    return ETO_RETIME_WNS_MAG_MIN_NS <= abs(w) <= ETO_RETIME_WNS_MAG_MAX_NS


def eto_retime_ff_drift_ok(ff_before, ff_after) -> bool:
    """Protect retiming registration against excessive flip-flop count drift.

    Returns `True` only when both counts are parseable, the initial count is
    positive, and relative drift is at most 1% inclusive. Any missing, invalid,
    or unsafe value returns `False` and therefore aborts registration. The 1%
    threshold is the latency-protection tolerance.
    """
    if ff_before is None or ff_after is None:
        return False
    try:
        b = int(ff_before)
        a = int(ff_after)
    except (TypeError, ValueError):
        return False
    if b <= 0 or a < 0:
        return False
    return abs(a - b) / float(b) <= ETO_RETIME_FF_DRIFT_MAX_FRAC


def v40_flag_env_present(*names) -> bool:
    """True iff ANY of the given env vars is PRESENT (any value, incl. "0").
    Gates ONLY the 'skipped reason=disabled' audit lines: a run with
    none of these knobs in the environment must stay log-byte
    identical to the baseline.  Behavior gates keep using the *_enabled() helpers."""
    return any(os.environ.get(n) is not None for n in names)


def midband_retry_hold_enabled() -> bool:
    """FPL26_MIDBAND_RETRY_HOLD master flag (DEFAULT OFF).  Kill switch
    FPL26_NO_MIDBAND_RETRY_HOLD wins, per house convention."""
    _truthy = ("1", "true", "on", "yes")
    if (os.environ.get("FPL26_NO_MIDBAND_RETRY_HOLD", "")
            .strip().lower() in _truthy):
        return False
    return (os.environ.get("FPL26_MIDBAND_RETRY_HOLD", "")
            .strip().lower() in _truthy)


def midband_route_rung_enabled() -> bool:
    """FPL26_MIDBAND_ROUTE_RUNG master flag (DEFAULT OFF).  Kill switch
    FPL26_NO_MIDBAND_ROUTE_RUNG wins, per house convention."""
    _truthy = ("1", "true", "on", "yes")
    if (os.environ.get("FPL26_NO_MIDBAND_ROUTE_RUNG", "")
            .strip().lower() in _truthy):
        return False
    return (os.environ.get("FPL26_MIDBAND_ROUTE_RUNG", "")
            .strip().lower() in _truthy)


def recipe_pass_band(wns_in) -> Optional[str]:
    """Classify failing input timing by absolute WNS for recipe selection.

    Uses the initial WNS captured when the design is opened and does not
    remeasure it. Band boundaries are inclusive: 1.05 ns belongs to the shallow
    band and 8.0 ns belongs to the deep band. Missing or non-negative WNS
    produces no band.
    """
    if wns_in is None:
        return None
    try:
        w = float(wns_in)
    except (TypeError, ValueError):
        return None
    # Non-finite timing values disable this band predicate.
    # The owning class initializes best WNS to negative infinity, so this is reachable.
    # Returning None prevents malformed data from arming a timing-band recipe.
    if w != w or w in (float("inf"), float("-inf")):
        return None
    if w >= 0.0:
        # Met designs are out of scope.
        # Currently unreachable at the call site (optimize() early-returns
        # on initial_wns >= 0 before the pass) — belt-and-braces so abs()
        # can never fold POSITIVE slack into a band if the slot ever moves.
        return None
    mag = abs(w)
    if mag <= RECIPE_PASS_SHALLOW_WNS_MAG_MAX_NS:
        return "shallow"
    if mag >= RECIPE_PASS_DEEP_WNS_MAG_MIN_NS:
        return "deep"
    return "mid"


def recipe_pass_enabled() -> bool:
    """FPL26_RECIPE_PASS master flag (DEFAULT OFF).  The kill switch
    FPL26_NO_RECIPE_PASS wins over the enable, per house convention."""
    _truthy = ("1", "true", "on", "yes")
    if os.environ.get("FPL26_NO_RECIPE_PASS", "").strip().lower() in _truthy:
        return False
    return os.environ.get("FPL26_RECIPE_PASS", "").strip().lower() in _truthy


def recipe_first_deep_enabled() -> bool:
    """FPL26_RECIPE_FIRST_DEEP (DEFAULT OFF): size-anchored PRE-LLM
    deep gate variant.  Opt-in on TOP of the master flag — BOTH
    FPL26_RECIPE_PASS and FPL26_RECIPE_FIRST_DEEP must be truthy, and
    FPL26_NO_RECIPE_PASS kills both (it already zeroes
    recipe_pass_enabled, which this requires)."""
    if not recipe_pass_enabled():
        return False
    _truthy = ("1", "true", "on", "yes")
    return (os.environ.get("FPL26_RECIPE_FIRST_DEEP", "")
            .strip().lower() in _truthy)


def resolve_fresh_presweep_draws(
        cli_value, default: int = FRESH_PRESWEEP_DRAWS_DEFAULT) -> int:
    """FRESH-PRESWEEP draw count K (DEFAULT 0 = OFF = zero diff).

    CLI (--fresh-presweep-draws) wins over env
    (FPL26_FRESH_PRESWEEP_DRAWS); 0/unset = OFF; unparseable/negative
    keeps the default (OFF); values above FRESH_PRESWEEP_DRAWS_MAX are
    clamped to it (K clamp 1..3).  Same convention as
    resolve_deep_wns_tail_reserve_s."""
    v = cli_value
    if v is None:
        env = os.environ.get("FPL26_FRESH_PRESWEEP_DRAWS", "").strip()
        if env:
            try:
                v = int(float(env))
            except ValueError:
                logger.warning(
                    f"FPL26_FRESH_PRESWEEP_DRAWS={env!r} not an int; "
                    f"keeping default {default} (pre-sweep OFF).")
                return default
    if v is None:
        return default
    try:
        v = int(v)
    except (TypeError, ValueError):
        return default
    if v < 0:
        logger.warning(
            f"fresh_presweep_draws={v} negative; keeping default {default} "
            "(pre-sweep OFF).")
        return default
    if v > FRESH_PRESWEEP_DRAWS_MAX:
        logger.warning(
            f"fresh_presweep_draws={v} above clamp; using "
            f"{FRESH_PRESWEEP_DRAWS_MAX} (K clamp 1..{FRESH_PRESWEEP_DRAWS_MAX}).")
        return FRESH_PRESWEEP_DRAWS_MAX
    return v


def _presweep_draw_allowance(
        draw_index: int,
        spent_s: float,
        max_observed_cost_s: Optional[float],
        budget_total_s: float,
        first_draw_timeout_s: float) -> tuple[bool, float, str]:
    """Budget gate for the k-th pre-sweep draw (1-based).  Pure function.

    First-draw-measures protocol: at step 0 there are NO
    cost samples yet, so draw 1 runs under a hard timeout of
    first_draw_timeout_s (0.12 x wall) and its OBSERVED cost gates
    draws 2..K — the next draw is allowed only if
        spent + max_observed * FRESH_PRESWEEP_COST_SAFETY <= budget_total.

    Returns (allowed, timeout_s, reason).  timeout_s for draws 2..K is
    the inflated prediction bounded by the remaining budget."""
    if draw_index <= 1:
        timeout = min(first_draw_timeout_s, budget_total_s)
        if timeout <= 0.0:
            return (False, 0.0, "no_budget")
        return (True, timeout, "first_draw_measures")
    if max_observed_cost_s is None or max_observed_cost_s <= 0.0:
        # Draw 1 left no cost observation (should not happen — even a
        # timed-out draw is timed) — fail CLOSED, never guess.
        return (False, 0.0, "no_cost_sample")
    predicted = max_observed_cost_s * FRESH_PRESWEEP_COST_SAFETY
    remaining = budget_total_s - spent_s
    if spent_s + predicted > budget_total_s:
        return (False, 0.0,
                f"budget_gate spent={spent_s:.0f}s "
                f"predicted={predicted:.0f}s budget={budget_total_s:.0f}s")
    return (True, min(predicted, remaining), "cost_gated")


def polish_gate_reserve_s(cfg, budget_deadline) -> tuple[float, str]:
    """Select the additional finalize reserve for post-ILS polish gates.

    The polish deadline already excludes `_finalize_reserve_seconds`, so
    corrected accounting must not add the same reserve to `need` again. Returns
    the reserve in seconds and a tag describing the selected accounting mode
    for logs.
    """
    _cfg_reserve = float(getattr(cfg, "fanout_finalize_reserve_s", 300.0))
    _armed = (os.environ.get("FPL26_POLISH_NO_DOUBLE_RESERVE", "0")
              .strip().lower() in ("1", "true", "on", "yes"))
    # HONESTY CONDITION: only the `_budget_deadline` path has the reserve baked
    # in.  `_ils_polish_body` falls back to `time.time() + 1200` when no wall
    # cap is set, and THAT deadline never had a reserve removed — dropping the
    # term there would be a real over-spend, not a correction.
    if _armed and budget_deadline is not None:
        return 0.0, "reserve already in deadline"
    return _cfg_reserve, f"reserve {_cfg_reserve:.0f}s"


def _fanout_polish_cost_basis(cfg) -> tuple[float, str]:
    """Compute the shared cost basis for fanout-polish eligibility and budgeting.

    Prefer a complete dedicated anchor containing physical-optimization and
    routing samples; the stage never places, so one physical-optimization pass
    plus one reroute is its worst case. If the routing sample is unavailable,
    use the full-cycle anchor when present and the incomplete fanout anchor
    only as a fallback. The eligibility gate and budget precheck must use this
    same estimate to avoid accounting drift. The returned estimate is in
    seconds.
    """
    full = getattr(cfg, "expected_heavy_cycle_s", 0.0) or 0.0
    fan = getattr(cfg, "fanout_cost_anchor_s", 0.0) or 0.0
    has_route = bool(getattr(cfg, "fanout_anchor_has_route", False))
    if fan > 0 and (has_route or full <= 0):
        return fan, "granular"
    return full, "full-cycle"
