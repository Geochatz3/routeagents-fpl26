# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (C) 2026, Georgios Chatzitsompanis.
# Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

"""LLM runtime mechanisms: error classification, prompt-size guard, cost ledger.

Extracted verbatim from dcp_optimizer.py (no behavior change): the pure,
self-state-free mechanisms under the LLM call path —

- API-error classification (`is_model_unavailable_error`,
  `is_key_level_auth_error`, `is_transient_api_error`,
  `is_prompt_limit_error`): the disjoint outage classes that decide
  retry-same-model vs model-fallback vs prune-and-retry.  Delegates to
  optimizer.api_resilience.classify_api_error when available (soft
  import), with the historical signature lists as the fallback;
- prompt-size guard (`estimate_message_tokens`, `prune_conversation`):
  conservative chars/3 token estimation and the head+suffix conversation
  prune that defuses the provider's per-request 402 prompt-limit doom
  loop;
- per-request SDK timeout (`_llm_timeout_s`) and the in-attempt LLM cost
  exit (`LLM_COST_EXIT_USD`, `resolve_llm_cost_exit`);
- crash-safe spend ledger (`write_cost_ledger`): atomic
  cost_ledger.json rewrite after every cost accrual, so a crash can
  never bypass the cumulative beta circuit-breaker.

DCPOptimizer keeps thin delegating methods with their original names and
signatures (e.g. `_is_transient_api_error`, `_prune_conversation`,
`_write_cost_ledger`), so call sites, subclass overrides, and the test
suite are unchanged.  The completion-with-fallback loop itself
(`_create_completion_with_fallback`), model constants, and backoff wiring
stay in dcp_optimizer.py — they are run-state-coupled orchestration;
only the mechanisms live here (backoff scheduling is already in
optimizer/api_resilience.py).
"""

import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Same soft-import pattern as dcp_optimizer.py: when api_resilience is
# unavailable the classifiers fall back to the historical signature lists.
try:
    from optimizer.api_resilience import classify_api_error as _classify_api_error
except Exception:  # pragma: no cover — defensive
    _classify_api_error = None


def _llm_timeout_s() -> float:
    """Return the per-request SDK timeout in seconds.

    FPL26_LLM_TIMEOUT_S overrides the 300-second default. Missing, malformed,
    or non-positive values fall back to that default instead of failing client
    construction; the default is pinned to conservative headroom for
    synchronous SDK latency.
    """
    try:
        v = float(os.environ.get("FPL26_LLM_TIMEOUT_S", "") or 300.0)
    except (TypeError, ValueError):
        return 300.0
    return v if v > 0 else 300.0


# End LLM calls before the run's dollar cap, then execute the zero-LLM-cost
# polish and finalization tail. The $0.75 default leaves margin under a $1 cap.
# A separate $0.85 between-attempt guard applies cumulatively across restarts.
LLM_COST_EXIT_USD = 0.75


def resolve_llm_cost_exit(budget, default: float = LLM_COST_EXIT_USD) -> float:
    """β circuit-breaker: effective in-attempt LLM cost exit ($).

    The multi-restart wrapper passes attempt N a budget of
    (cumulative ceiling − prior attempts' spend) so a later attempt can
    never re-spend the full $0.75 allowance on top of what earlier
    attempts already billed (the eval zeroes a benchmark at
    $1.00 cumulative LLM spend; a rehearsal run hit $0.76 cumulative with
    no such accounting). Tightening-only: a budget can lower the exit below the
    default but never raise it; None/<=0/unparseable (unset, or the
    wrapper's breaker switched off) keeps the default.
    """
    try:
        b = float(budget) if budget is not None else None
    except (TypeError, ValueError):
        return default
    if b is None or b <= 0:
        return default
    return min(default, b)


def is_model_unavailable_error(e: Exception) -> bool:
    """True iff the error indicates the MODEL is unavailable (not a transient
    rate-limit/timeout/5xx).  Only these warrant a model fallback.

    Reclassification note: the auth signatures
    ("unauthorized", "401", "403", "permission") were removed from this
    classifier — they are key-level outages (see
    is_key_level_auth_error), not model outages.  The eval-day defect:
    a key-level 401 ("User not found") matched here and spuriously
    pinned the fallback model, which shares the same dead key.
    Delegates to optimizer.api_resilience.classify_api_error, whose
    key_auth check precedes the model check so "user not found" can
    never match the bare "not found" model signature."""
    s = f"{type(e).__name__}: {e}"
    if _classify_api_error is not None:
        return _classify_api_error(s) == "model_unavailable"
    s = s.lower()
    sigs = ("not found", "404", "model_not_found", "no endpoints",
            "no allowed providers", "is not a valid model", "does not exist",
            "deprecat")
    return any(x in s for x in sigs)


def is_key_level_auth_error(e: Exception) -> bool:
    """True iff the error is a KEY-level auth outage (the eval-day 401
    'User not found' storm class, observed twice in the same
    outage).  These must backoff-retry the SAME primary model —
    switching models cannot fix a dead key."""
    s = f"{type(e).__name__}: {e}"
    if _classify_api_error is not None:
        return _classify_api_error(s) == "key_auth"
    s = s.lower()
    sigs = ("user not found", "401", "403", "unauthorized",
            "invalid api key", "no auth credentials", "permission")
    return any(x in s for x in sigs)


def is_transient_api_error(e: Exception) -> bool:
    """True for retryable provider/API conditions (rate-limit, 5xx, network
    timeouts).  Disjoint from is_model_unavailable_error: that one means
    the MODEL is gone (fall back immediately); this one means the provider
    is having a moment (retry, then fall back if it persists)."""
    s = f"{type(e).__name__}: {e}".lower()
    sigs = ("429", "rate limit", "rate_limit", "500", "502", "503", "504",
            "timeout", "timed out", "overloaded", "connection error",
            "connection reset", "temporarily", "service unavailable",
            "apiconnectionerror", "internal server error")
    return any(x in s for x in sigs)


def is_prompt_limit_error(e: Exception) -> bool:
    """True for the provider's per-request prompt-size rejection (the
    contest key's 402 'Prompt tokens limit exceeded')."""
    s = f"{type(e).__name__}: {e}".lower()
    return ("prompt tokens limit exceeded" in s
            or ("402" in s and "prompt token" in s))


def estimate_message_tokens(m) -> int:
    """Conservative token estimate (chars/3 — JSON-heavy tool output runs
    denser than prose's chars/4; overestimating is the safe direction)."""
    try:
        return max(1, len(json.dumps(m, default=str)) // 3)
    except Exception:
        return max(1, len(str(m)) // 3)


def prune_conversation(messages, target_tokens: int, *, head_keep: int,
                       best_wns=None, estimate_fn=None):
    """Drop middle messages so the conversation fits ~target_tokens.

    Keeps the head (system prompt + initial analysis/recipe message,
    ``head_keep``) and a contiguous SUFFIX of recent messages,
    replacing the dropped middle with one marker message carrying the
    essential state (best WNS). A contiguous suffix can never orphan a
    tool response from a LATER parent, but it may START with tool
    messages whose assistant parent was dropped — those are popped so
    the API never sees an orphaned tool_call_id.

    Pure: returns ``(new_messages, dropped)`` and never mutates the
    input list; the caller assigns the result (DCPOptimizer's
    ``_prune_conversation`` delegate does, preserving its return-#dropped
    contract).  ``estimate_fn`` defaults to :func:`estimate_message_tokens`.
    """
    if estimate_fn is None:
        estimate_fn = estimate_message_tokens
    head = messages[:head_keep]
    rest = messages[head_keep:]
    if not rest:
        return messages, 0
    head_tokens = sum(estimate_fn(m) for m in head)
    tail_budget = max(2_000, target_tokens - head_tokens - 200)
    kept: list = []
    used = 0
    for m in reversed(rest):
        t = estimate_fn(m)
        if used + t > tail_budget and kept:
            break
        kept.append(m)
        used += t
    kept.reverse()
    while kept and isinstance(kept[0], dict) and kept[0].get("role") == "tool":
        kept.pop(0)
    dropped = len(rest) - len(kept)
    if dropped <= 0:
        return messages, 0
    best = best_wns
    best_txt = (f"{best:.3f} ns" if isinstance(best, (int, float))
                and best != float("-inf") else "unknown")
    marker = {
        "role": "user",
        "content": (
            f"[CONTEXT PRUNED: {dropped} earlier conversation messages were "
            f"removed to fit the provider's per-request prompt-token limit. "
            f"State: current best WNS on the scored clock is {best_txt}; the "
            f"best checkpoint is preserved on disk by the keep-best mirror. "
            f"Continue optimizing from the most recent messages below.]"
        ),
    }
    return head + [marker] + kept, dropped


def write_cost_ledger(run_dir, total_cost, calls) -> None:
    """Persist cumulative LLM cost in a crash-safe ledger.

    cost_ledger.json is initialized with a zero-cost record and rewritten after
    every API-call cost accrual so restarts can enforce the cumulative circuit
    breaker even when the normal summary is absent. Writes use a temporary file
    followed by os.replace to avoid exposing partial data. Ledger failures are
    logged and swallowed because auditing must not terminate optimization.
    """
    try:
        payload = json.dumps({
            "total_cost": round(float(total_cost or 0.0), 6),
            "calls": int(calls or 0),
        })
        path = Path(run_dir) / "cost_ledger.json"
        fd, tmp = tempfile.mkstemp(
            prefix="cost_ledger.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(payload)
            os.replace(tmp, str(path))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as e:
        try:
            logger.warning(f"cost-ledger write failed (non-fatal): {e}")
        except Exception:
            pass
