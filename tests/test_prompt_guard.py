"""Unit tests for the prompt-size guard (eval-key 402 doom loop, jul03).

Preview attempt-3 forensics: the contest's provisioned OpenRouter key enforces
a per-request prompt-token limit (observed 57,489). Once the conversation
crossed it every LLM call failed 402 and the error handler grew the prompt
further. These tests cover the estimator, the pruner (head kept, contiguous
suffix kept, no orphaned tool messages, marker inserted), the 402 classifier,
and the reactive prune-and-retry path. No network."""
import pytest
from dcp_optimizer import (DCPOptimizer, DEFAULT_MODEL,
                           PROMPT_TOKEN_SOFT_LIMIT, PROMPT_PRUNE_TARGET,
                           PROMPT_PRUNE_HEAD_KEEP)


PROMPT_402 = Exception(
    "Error code: 402 - {'error': {'message': \"Prompt tokens limit exceeded: "
    "110991 > 57489. To increase, visit https://openrouter.ai/...\", "
    "'code': 402}}")


def _opt():
    o = DCPOptimizer(api_key="dummy", model=DEFAULT_MODEL)
    o.tools = []
    return o


def _mk_messages(n_pairs, content_chars=3000):
    """system + initial-user head, then n_pairs of (assistant+tool_calls,
    tool, user) triples — the real conversation shape."""
    msgs = [{"role": "system", "content": "SYSTEM PROMPT " + "s" * 200},
            {"role": "user", "content": "INITIAL ANALYSIS + RECIPE " + "r" * 500}]
    for i in range(n_pairs):
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": f"call_{i}", "type": "function",
                                     "function": {"name": "vivado_run_tcl",
                                                  "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}",
                     "name": "vivado_run_tcl",
                     "content": "TOOL OUTPUT " + "x" * content_chars})
        msgs.append({"role": "user", "content": f"continue {i}"})
    return msgs


def test_classifier_prompt_limit_true():
    assert DCPOptimizer._is_prompt_limit_error(PROMPT_402) is True


def test_classifier_prompt_limit_false_on_other_errors():
    for msg in ("Error code: 429 - rate limit", "connection timed out",
                "Error code: 402 - insufficient credits"):
        assert DCPOptimizer._is_prompt_limit_error(Exception(msg)) is False


def test_estimator_monotonic():
    o = _opt()
    o.messages = _mk_messages(2)
    small = o._estimate_messages_tokens()
    o.messages = _mk_messages(20)
    assert o._estimate_messages_tokens() > small > 0


def test_prune_keeps_head_and_recent_suffix():
    o = _opt()
    o.messages = _mk_messages(40)          # ~40*3k chars >> target
    o.best_wns = -0.457
    before = list(o.messages)
    dropped = o._prune_conversation(PROMPT_PRUNE_TARGET)
    assert dropped > 0
    # head preserved verbatim
    assert o.messages[:PROMPT_PRUNE_HEAD_KEEP] == before[:PROMPT_PRUNE_HEAD_KEEP]
    # marker present right after the head, carrying the state
    marker = o.messages[PROMPT_PRUNE_HEAD_KEEP]
    assert marker["role"] == "user" and "CONTEXT PRUNED" in marker["content"]
    assert "-0.457" in marker["content"]
    # the kept tail is a contiguous suffix of the original
    tail = o.messages[PROMPT_PRUNE_HEAD_KEEP + 1:]
    assert tail == before[len(before) - len(tail):]
    # and the estimate actually fits the target now
    assert o._estimate_messages_tokens() <= PROMPT_PRUNE_TARGET + 2_000


def test_prune_never_orphans_tool_messages():
    o = _opt()
    o.messages = _mk_messages(40)
    o._prune_conversation(6_000)           # aggressive target
    tail = o.messages[PROMPT_PRUNE_HEAD_KEEP + 1:]
    # a tool message may only appear after its assistant parent in the tail
    seen_assistant_ids = set()
    for m in tail:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                seen_assistant_ids.add(tc["id"])
        elif m.get("role") == "tool":
            assert m["tool_call_id"] in seen_assistant_ids, \
                "orphaned tool message after prune"


def test_prune_noop_when_small():
    o = _opt()
    o.messages = _mk_messages(1, content_chars=100)
    before = list(o.messages)
    assert o._prune_conversation(PROMPT_PRUNE_TARGET) == 0
    assert o.messages == before


class _PromptLimitedCompletions:
    """Rejects any request whose serialized messages exceed a char budget —
    a faithful model of the provider-side prompt cap."""
    def __init__(self, char_limit):
        self.char_limit = char_limit
        self.calls = 0
    def create(self, **kw):
        import json as _j
        self.calls += 1
        if len(_j.dumps(kw["messages"], default=str)) > self.char_limit:
            raise type(PROMPT_402)(str(PROMPT_402))
        return {"ok": True}


def test_reactive_prune_and_retry_recovers():
    o = _opt()
    o.messages = _mk_messages(40)          # far above the fake cap
    comp = _PromptLimitedCompletions(char_limit=60_000)
    o.openai = type("F", (), {"chat": type("C", (), {"completions": comp})()})()
    resp = o._create_completion_with_fallback()
    assert resp == {"ok": True}
    assert comp.calls >= 2                 # failed at least once, then recovered
    # model unchanged (prompt-limit is not a model failure)
    assert o.model == DEFAULT_MODEL


def test_llm_cost_exit_constant_below_contest_budget():
    """The in-attempt cost exit must sit below multi-restart's cumulative cap
    (0.85) and the contest's $1.00/benchmark budget, with metering slop room."""
    from dcp_optimizer import LLM_COST_EXIT_USD
    assert 0.5 <= LLM_COST_EXIT_USD <= 0.80


def test_run_dir_base_env_override(tmp_path, monkeypatch):
    """FPL26_RUN_DIR_BASE redirects run-artifact dirs (local ops, C-drive
    space); unset -> CWD (contest/eval behavior unchanged)."""
    from dcp_optimizer import _run_dir_base
    from pathlib import Path
    monkeypatch.delenv("FPL26_RUN_DIR_BASE", raising=False)
    assert _run_dir_base() == Path.cwd()
    monkeypatch.setenv("FPL26_RUN_DIR_BASE", str(tmp_path / "runs"))
    p = _run_dir_base()
    assert p == tmp_path / "runs" and p.is_dir()
