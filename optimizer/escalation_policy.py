"""Model-escalation policy — feature-based triggers for when to upgrade
the executor LLM from a cheap/fast default to a stronger reasoning model.

DESIGN INTENT
=============

Most iterations of the optimizer flow are routine tool calls (phys_opt,
route, get_wns, write_checkpoint). A cheap/fast model handles these
fine. The expensive cases — diagnosis, strategy pivot, failure recovery
— are rare. Globally switching to an expensive model for every
iteration wastes cost on the routine 95% to slightly improve the 5%.

This module provides PURE feature-based trigger detection. A run-state
snapshot goes in; an EscalationDecision comes out. The actual model
swap is a separate, opt-in concern wired up in dcp_optimizer.

CURRENT IMPLEMENTATION PHASE: detection + logging only.
The trigger functions fire and log; dcp_optimizer collects the
EscalationDecision but does NOT swap models in this phase. Telemetry
from real runs will inform whether the swap actually improves outcomes
before we commit to the higher cost.

FUTURE PHASE: gated model swap.
A new --escalation-model CLI flag (and per-call model_override) wires
the decision into the chat-completions call. Default off; opt-in.

TRIGGER POLICY (feature-based, no benchmark names)
==================================================

A trigger fires when ALL of its preconditions hold. Triggers are checked
in priority order; first matching wins. Inputs are derived from
optimizer run state (iteration counts, slope, budget, error counters)
— never from design names.

T1: STAGNATION
    - last_improvement_iter > 0 (i.e., we know what gain looks like)
    - current_iter - last_improvement_iter ≥ 2 (2+ iters since last gain)
    - remaining_wall_budget_s ≥ 600 (10 min — enough for a single
      reasoning-heavy call to be useful)
    - best_wns better than initial_wns (we're not just at baseline)
    Rationale: cheap model has tried 2+ moves without improvement; a
    smarter pivot decision could find a new path.

T2: BAD_TCL_BURST
    - bad_tcl_count_recent ≥ 2 across last 3 iters
      (cheap model is generating Tcl that Vivado/RapidWright rejects)
    - remaining_wall_budget_s ≥ 300
    Rationale: cheap model is making syntactic/semantic errors that
    waste tool-call budget.

T3: HIGH_UPSIDE_STUCK
    - achievable_fmax_ratio < 0.30 (fundamental closure gap remains)
    - iters_since_improve ≥ 1
    - remaining_wall_budget_s ≥ 900 (15 min)
    Rationale: large WNS gap remains but cheap model isn't progressing;
    a stronger reasoning pass might identify an unexplored angle.

T4: ROUTER_PLAN_SKIP
    - router plan recommends a heavy step (vivado_place_design) that
      has not been attempted in this run
    - LLM has signalled stop (is_done = True)
    - remaining_wall_budget_s ≥ 300
    Rationale: this overlaps with the BETA-CTRL-V0.4 router-step-aware
    continuation; the escalation hook is an alternative — instead of
    just nudging the cheap model with text, escalate to a smarter model
    that's more likely to actually execute the recommended step.
    Currently logged but not acted on; V0.4 nudge runs first.

ANTI-TRIGGER (do NOT escalate)
------------------------------

X1: ROUTINE_PROGRESS — last_improvement_iter == current_iter
    The cheap model just improved WNS; reward the working strategy.

X2: NEAR_CLOSURE — best_wns >= -0.5 ns
    Design is very close to closure; cheap fast iterations are likely
    enough to finish.

X3: BUDGET_EXHAUSTED — remaining_wall_budget_s < 300
    Not enough time for the escalation call to deliver value before
    finalize.

X4: ALREADY_ESCALATED_THIS_RUN — escalated_iter > 0
    One-shot per run for now; further escalations need explicit evidence
    that they help (not in this phase).

POLICY SUMMARY
==============

Trigger checks happen at iteration boundaries AFTER WNS measurement.
The result is recorded in self._escalation_telemetry for the run
summary and (in a future phase) used to gate the next LLM call.

All decisions log a single INFO line with: trigger_id, rationale,
feature snapshot. This makes post-run analysis trivial — grep for
`escalation_policy:` in the run.log.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds — exposed for tests + future tuning.
# ---------------------------------------------------------------------------

T1_STAGNATION_ITERS = 2
T1_MIN_BUDGET_S = 600.0

T2_BAD_TCL_RECENT = 2
T2_BAD_TCL_WINDOW = 3
T2_MIN_BUDGET_S = 300.0

T3_FMAX_RATIO_MAX = 0.30
T3_MIN_BUDGET_S = 900.0

T4_MIN_BUDGET_S = 300.0

X1_NEAR_CLOSURE_WNS_NS = -0.5


# ---------------------------------------------------------------------------
# Inputs / Outputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSnapshot:
    """Pure-data snapshot of optimizer state at an iteration boundary.

    Constructed by dcp_optimizer from self.* fields; the policy module
    has no knowledge of the optimizer class or its internals.
    """
    current_iter: int
    last_improvement_iter: int
    initial_wns_ns: Optional[float]
    best_wns_ns: Optional[float]
    achievable_fmax_ratio: Optional[float]
    remaining_wall_budget_s: Optional[float]
    bad_tcl_count_recent: int = 0          # over the last T2_BAD_TCL_WINDOW iters
    is_done_signal: bool = False           # LLM signalled stop this iter
    router_unattempted_heavy_step: Optional[str] = None  # from optimizer helper
    already_escalated_this_run: bool = False


@dataclass(frozen=True)
class EscalationDecision:
    """The policy module's answer for a snapshot."""
    should_escalate: bool
    trigger_id: str          # "T1".."T4" when firing; "NONE" / "X*" otherwise
    rationale: str           # one-line explanation for logging
    snapshot: RunSnapshot    # echoed back for telemetry capture


# ---------------------------------------------------------------------------
# Anti-trigger checks (return reason string when escalation should be
# blocked; None when not blocking).
# ---------------------------------------------------------------------------

def _x1_routine_progress(s: RunSnapshot) -> Optional[str]:
    if s.last_improvement_iter == s.current_iter:
        return ("X1_ROUTINE_PROGRESS — improvement just landed this iter "
                "(iter=%d); cheap model is working" % s.current_iter)
    return None


def _x2_near_closure(s: RunSnapshot) -> Optional[str]:
    if s.best_wns_ns is not None and s.best_wns_ns >= X1_NEAR_CLOSURE_WNS_NS:
        return ("X2_NEAR_CLOSURE — best_wns=%.3f ns >= %.1f ns; cheap model "
                "should close" % (s.best_wns_ns, X1_NEAR_CLOSURE_WNS_NS))
    return None


def _x3_budget_exhausted(s: RunSnapshot) -> Optional[str]:
    if (s.remaining_wall_budget_s is not None
            and s.remaining_wall_budget_s < T4_MIN_BUDGET_S):
        return ("X3_BUDGET_EXHAUSTED — remaining=%.0fs < %.0fs floor; no "
                "time for escalation to land" %
                (s.remaining_wall_budget_s, T4_MIN_BUDGET_S))
    return None


def _x4_already_escalated(s: RunSnapshot) -> Optional[str]:
    if s.already_escalated_this_run:
        return "X4_ALREADY_ESCALATED — one-shot per run in current phase"
    return None


# ---------------------------------------------------------------------------
# Trigger checks (return rationale when firing; None when not firing).
# ---------------------------------------------------------------------------

def _t1_stagnation(s: RunSnapshot) -> Optional[str]:
    if s.last_improvement_iter <= 0:
        return None
    iters_since = s.current_iter - s.last_improvement_iter
    if iters_since < T1_STAGNATION_ITERS:
        return None
    if (s.remaining_wall_budget_s is None
            or s.remaining_wall_budget_s < T1_MIN_BUDGET_S):
        return None
    if (s.best_wns_ns is None or s.initial_wns_ns is None
            or s.best_wns_ns <= s.initial_wns_ns):
        return None
    return ("T1_STAGNATION — %d iters since improvement (best=%.3f ns, "
            "budget=%.0fs)" %
            (iters_since, s.best_wns_ns, s.remaining_wall_budget_s))


def _t2_bad_tcl_burst(s: RunSnapshot) -> Optional[str]:
    if s.bad_tcl_count_recent < T2_BAD_TCL_RECENT:
        return None
    if (s.remaining_wall_budget_s is None
            or s.remaining_wall_budget_s < T2_MIN_BUDGET_S):
        return None
    return ("T2_BAD_TCL_BURST — %d bad-Tcl events in last %d iters "
            "(budget=%.0fs)" %
            (s.bad_tcl_count_recent, T2_BAD_TCL_WINDOW,
             s.remaining_wall_budget_s))


def _t3_high_upside_stuck(s: RunSnapshot) -> Optional[str]:
    if s.achievable_fmax_ratio is None or s.achievable_fmax_ratio >= T3_FMAX_RATIO_MAX:
        return None
    iters_since = (s.current_iter - s.last_improvement_iter
                   if s.last_improvement_iter > 0
                   else s.current_iter)
    if iters_since < 1:
        return None
    if (s.remaining_wall_budget_s is None
            or s.remaining_wall_budget_s < T3_MIN_BUDGET_S):
        return None
    return ("T3_HIGH_UPSIDE_STUCK — fmax_ratio=%.0f%% < %.0f%%, "
            "iters_since_improve=%d, budget=%.0fs" %
            (100 * s.achievable_fmax_ratio,
             100 * T3_FMAX_RATIO_MAX,
             iters_since,
             s.remaining_wall_budget_s))


def _t4_router_plan_skip(s: RunSnapshot) -> Optional[str]:
    if not s.is_done_signal:
        return None
    if not s.router_unattempted_heavy_step:
        return None
    if (s.remaining_wall_budget_s is None
            or s.remaining_wall_budget_s < T4_MIN_BUDGET_S):
        return None
    return ("T4_ROUTER_PLAN_SKIP — LLM stopping while %s unattempted "
            "(budget=%.0fs)" %
            (s.router_unattempted_heavy_step,
             s.remaining_wall_budget_s))


# Triggers in priority order. T1 (stagnation) is the most evidence-backed
# case; T4 overlaps with BETA-CTRL-V0.4 router nudge.
_TRIGGERS = (
    ("T4", _t4_router_plan_skip),   # First — most specific
    ("T2", _t2_bad_tcl_burst),
    ("T1", _t1_stagnation),
    ("T3", _t3_high_upside_stuck),
)


def evaluate(snapshot: RunSnapshot) -> EscalationDecision:
    """Pure policy entry point. Returns an EscalationDecision.

    Logs one INFO line whether escalation fires or not so post-run greps
    can audit policy behaviour without re-running anything.
    """
    # Anti-triggers — short-circuit BEFORE any positive trigger check.
    for x in (_x4_already_escalated, _x3_budget_exhausted,
              _x2_near_closure, _x1_routine_progress):
        msg = x(snapshot)
        if msg is not None:
            logger.info(f"escalation_policy: BLOCKED ({msg})")
            return EscalationDecision(
                should_escalate=False,
                trigger_id=msg.split("_", 1)[0],  # "X1".."X4"
                rationale=msg,
                snapshot=snapshot,
            )

    for trigger_id, check in _TRIGGERS:
        msg = check(snapshot)
        if msg is not None:
            logger.info(f"escalation_policy: FIRE {msg}")
            return EscalationDecision(
                should_escalate=True,
                trigger_id=trigger_id,
                rationale=msg,
                snapshot=snapshot,
            )

    logger.info(
        "escalation_policy: NONE (iter=%d, last_improve=%d, "
        "wns=%s, ratio=%s, budget=%s)",
        snapshot.current_iter,
        snapshot.last_improvement_iter,
        snapshot.best_wns_ns,
        snapshot.achievable_fmax_ratio,
        snapshot.remaining_wall_budget_s,
    )
    return EscalationDecision(
        should_escalate=False,
        trigger_id="NONE",
        rationale="no trigger and no block",
        snapshot=snapshot,
    )
