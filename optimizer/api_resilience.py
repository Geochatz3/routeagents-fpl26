"""Pure API-error classifier + deadline-aware backoff scheduler
(R-D2-1 exponential wall-aware backoff, R-D2-4 no spurious model fallback).

The failure this prevents (TWO live reproductions, 02-CONTEXT.md):
official eval mini-isp 2026-07-14 18:09:02 and local boom_debugwall_run1
2026-07-19 18:53:02 — a KEY-level 401 storm (AuthenticationError, body
{'error': {'message': 'User not found.', 'code': 401}}) was string-matched
as a MODEL-unavailable error and instantly switched the run to the fallback
model — but the fallback shares the same dead key and also 401s, so the run
pinned a useless fallback and burned 8+ iterations in <1 second.  The only
backoff was a single flat 2s sleep.

This module is PURE (recipe_router / route_gate style): no clock module, no
IO, no SDK, no reference to the optimizer — fully unit-testable with a
mocked clock and zero real sleeps.  The caller (dcp_optimizer's
_create_completion_with_fallback) owns the loop, the sleeping, and the
budget plumbing; this module only classifies and computes.
"""
from __future__ import annotations

from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# Constants (evidence-commented, route_gate-style)
# ---------------------------------------------------------------------------

# Exponential per-episode schedule (~15/30/60/120/240s), saturating at the
# last step.  Sum = 465s, so the 600s cap below allows the full ladder plus
# part of one saturated step before giving up.
API_BACKOFF_SCHEDULE_S = (15.0, 30.0, 60.0, 120.0, 240.0)

# Per-episode total backoff cap.  02-CONTEXT.md: wall cost of backoff is
# cheap insurance — 10 min of 401-storm backoff costs gamma ~= 0.17h ~= 1.7%
# of alpha, vs losing the whole LLM path (the mini-isp eval key self-
# recovered minutes later; a run that kept retrying scored 89.316 by luck).
# After the cap the caller falls through to the existing "LLM dead"
# propagation — never loops backoff forever (T-02-03).
API_BACKOFF_EPISODE_CAP_S = 600.0


# Signature tables.  Classification lowercases once and checks in STRICT
# precedence order: prompt_limit -> key_auth -> transient ->
# model_unavailable -> other.  key_auth MUST precede model_unavailable:
# "user not found" contains the bare substring "not found" (a model
# signature), which is exactly how the eval-day 401 storm was misrouted to
# the fallback model (T-02-02).
_PROMPT_LIMIT_SIGS = ("prompt tokens limit exceeded",)

_KEY_AUTH_SIGS = (
    "user not found",
    "401",
    "403",
    "unauthorized",
    "invalid api key",
    "no auth credentials",
    "permission",
)

# Mirrors dcp_optimizer._is_transient_api_error's signature list (kept in
# sync manually; that staticmethod is intentionally unchanged).
_TRANSIENT_SIGS = (
    "429", "rate limit", "rate_limit", "500", "502", "503", "504",
    "timeout", "timed out", "overloaded", "connection error",
    "connection reset", "temporarily", "service unavailable",
    "apiconnectionerror", "internal server error",
)

# Model-level unavailability: these justify a one-shot fallback-model
# switch.  The bare "not found" only fires when key_auth did NOT already
# match (precedence order guarantees this).  The auth signatures
# ("unauthorized", "401", "403", "permission") were REMOVED from this list
# on 2026-07-20 — they are key-level, not model-level (R-D2-4).
_MODEL_UNAVAILABLE_SIGS = (
    "not found", "404", "model_not_found", "no endpoints",
    "no allowed providers", "is not a valid model", "does not exist",
    "deprecat",
)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def classify_api_error(exc_text: str) -> str:
    """Classify an API/SDK exception rendering into one of:
    "key_auth" | "model_unavailable" | "transient" | "prompt_limit" |
    "other".

    `exc_text` is the caller-rendered f"{type(e).__name__}: {e}" string
    (safe to pass any string; None is treated as empty).  Precedence:
    prompt_limit -> key_auth -> transient -> model_unavailable -> other.
    """
    s = (exc_text or "").lower()
    if any(x in s for x in _PROMPT_LIMIT_SIGS) or (
            "402" in s and "prompt token" in s):
        return "prompt_limit"
    if any(x in s for x in _KEY_AUTH_SIGS):
        return "key_auth"
    if any(x in s for x in _TRANSIENT_SIGS):
        return "transient"
    if any(x in s for x in _MODEL_UNAVAILABLE_SIGS):
        return "model_unavailable"
    return "other"


def compute_backoff_sleep(
    attempt: int,
    remaining_budget_s: float,
    finalize_guard_s: float,
    backoff_used_s: float,
    schedule: Sequence[float] = API_BACKOFF_SCHEDULE_S,
    episode_cap_s: float = API_BACKOFF_EPISODE_CAP_S,
) -> Optional[float]:
    """Deadline-aware backoff decision for retry number `attempt` (0-based).

    Returns the number of seconds to sleep before the next retry, or None
    when the caller must GIVE UP and propagate the error (episode cap
    reached, or no room left before the finalize reserve).

    - step = schedule[min(attempt, len-1)] (saturates, never IndexError);
    - episode cap: if backoff_used_s + step would exceed episode_cap_s,
      return None — fall through to the existing LLM-dead path (T-02-03);
    - deadline clamp: room = remaining_budget_s - finalize_guard_s; if no
      positive room, return None (the finalize path must always survive,
      T-02-01); otherwise the sleep is min(step, room), clamped so it never
      pushes past remaining-wall-minus-reserve;
    - result is always >= 0 and finite (an inf budget yields the plain
      step; a NaN budget fails CLOSED with None).

    Pure: no clock, no IO, no optimizer reference.
    """
    if not schedule:
        return None
    idx = min(max(int(attempt), 0), len(schedule) - 1)
    step = float(schedule[idx])
    if backoff_used_s + step > episode_cap_s:
        return None
    room = remaining_budget_s - finalize_guard_s
    if not (room > 0.0):  # catches <= 0 AND NaN (fail closed)
        return None
    return max(0.0, min(step, room))
