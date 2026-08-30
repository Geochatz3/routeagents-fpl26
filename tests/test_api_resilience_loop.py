"""Test loop behavior around retryable and permanent API failures.

API retries remain inside the completion layer and retain the selected model.
If an exhausted retry episode reaches the optimization loop:
- failed LLM calls do not consume optimization iterations;
- failures do not append one conversation message per attempt;
- a resolved episode adds at most one compact summary;
- non-API exceptions retain ordinary exception handling;
- disabling the kill switch restores ordinary handling;
- consecutive permanent episodes trigger a bounded termination condition;
- final statistics report episode count and total backoff time.

Integration replays use scripted completion calls and recorded backoff
requests, with no network access, FPGA tools, or real sleeps.
"""
from __future__ import annotations

import asyncio
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import (
    DCPOptimizer,
    FALLBACK_MODEL,
    PERMANENT_API_FAILURE_LIMIT,
)


def _async(coro):
    return asyncio.run(coro)


# The classifier receives a formatted exception type and message, so the
# serialized 401 body carries the authentication signature without an SDK exception.
_AUTH_401 = ("Error code: 401 - {'error': {'message': 'User not found.', "
             "'code': 401}}")


class _FakeAuthError(Exception):
    """Mimics openai.AuthenticationError by message content."""


def _make_loop_optimizer(tmp_path: Path, mode: str = "anchor",
                         max_wall: float | None = None) -> DCPOptimizer:
    """DCPOptimizer wired for OFFLINE loop tests.

    - rag_seed off: no strategy-memory disk lookups;
    - contest_mode on: skips the design-name-keyed recipe gate;
    - ILS stays disabled (constructor default) so the exit tail goes
      straight to finalize.
    """
    opt = DCPOptimizer(api_key="test", run_dir=tmp_path, mode=mode)
    opt.rag_seed = False
    opt.contest_mode = True
    opt.max_wall_seconds = max_wall
    return opt


def _run_optimize(opt: DCPOptimizer, tmp_path: Path,
                  get_completion_side_effect) -> tuple[bool, Path]:
    """Drive the REAL optimize() loop offline with a scripted get_completion.

    perform_initial_analysis is mocked (sets a failing initial WNS so the
    loop is entered); call_tool returns a benign string; the strategy-memory
    persist hook is silenced.  No sessions, no network, no sleeps.
    """
    input_dcp = tmp_path / "baseline.dcp"
    input_dcp.write_bytes(b"BASELINE_DCP_BYTES_FOR_LOOP_TEST")
    output_dcp = tmp_path / "optimized.dcp"

    async def fake_analysis(_input_dcp):
        opt.initial_wns = -2.0
        opt.clock_period = 4.0
        return "ANALYSIS: initial WNS -2.000 ns"

    with mock.patch("dcp_optimizer.load_system_prompt",
                    return_value="SYS PROMPT (loop test)"), \
         mock.patch.object(opt, "perform_initial_analysis",
                           side_effect=fake_analysis), \
         mock.patch.object(opt, "get_completion",
                           new=mock.AsyncMock(
                               side_effect=get_completion_side_effect)), \
         mock.patch.object(opt, "call_tool",
                           new=mock.AsyncMock(return_value="ok")), \
         mock.patch.object(opt, "_persist_to_strategy_memory"), \
         redirect_stdout(io.StringIO()):
        result = _async(opt.optimize(input_dcp, output_dcp))
    return result, output_dcp


def _error_appends(opt: DCPOptimizer) -> list[dict]:
    return [m for m in opt.messages
            if isinstance(m.get("content"), str)
            and m["content"].startswith("An error occurred")]


def _summaries(opt: DCPOptimizer) -> list[dict]:
    return [m for m in opt.messages
            if isinstance(m.get("content"), str)
            and m["content"].startswith("[api-status]")]


class ExceptHandlerHygieneTests(unittest.TestCase):
    """R-D2-2/R-D2-3: the optimize() except handler on API-error episodes."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_single_api_failure_no_append_no_iteration_burn(self):
        # One propagated key-401 then a clean done — the failure must not
        # grow the conversation (R-D2-3) and must not consume an iteration
        # (R-D2-2): final iteration is 1 (the successful call only).
        opt = _make_loop_optimizer(self.tmp_path)
        result, _ = _run_optimize(opt, self.tmp_path, [
            _FakeAuthError(_AUTH_401),
            ("optimization complete", True),
        ])
        self.assertTrue(result)
        self.assertEqual(opt.iteration, 1,
                         "failed API call must not consume an iteration")
        self.assertEqual(_error_appends(opt), [],
                         "no per-failure 'An error occurred' append")
        # No episode was recorded on self (get_completion is mocked, so
        # the call-path counter never moved) => no summary either.  The
        # conversation is exactly the system + iter-1 user message.
        self.assertEqual(len(opt.messages), 2)
        self.assertEqual(opt._api_error_episodes, 0)

    def test_multi_failure_episode_appends_at_most_one_summary(self):
        # Two cap-exhausted failures precede recovery. Failure episodes add at
        # most one compact status summary, and only successful calls increment
        # the iteration count.
        opt = _make_loop_optimizer(self.tmp_path)

        calls = {"n": 0}

        async def scripted():
            calls["n"] += 1
            if calls["n"] <= 2:
                # The real permanent path increments the episode counter
                # inside _create_completion_with_fallback before raising.
                opt._api_error_episodes += 1
                raise _FakeAuthError(_AUTH_401)
            return ("optimization complete", True)

        result, _ = _run_optimize(opt, self.tmp_path, scripted)
        self.assertTrue(result)
        self.assertEqual(opt.iteration, 1)
        self.assertEqual(_error_appends(opt), [])
        self.assertEqual(len(_summaries(opt)), 1,
                         "exactly one episode summary after recovery")
        self.assertEqual(opt._last_summarized_episode, 2)

    def test_summary_never_duplicated_after_resolution(self):
        # One episode, then TWO successful iterations — the summary must
        # be appended once and never repeated on later successes.
        opt = _make_loop_optimizer(self.tmp_path)

        calls = {"n": 0}

        async def scripted():
            calls["n"] += 1
            if calls["n"] == 1:
                opt._api_error_episodes += 1
                raise _FakeAuthError(_AUTH_401)
            if calls["n"] == 2:
                return ("continuing analysis", False)
            return ("optimization complete", True)

        result, _ = _run_optimize(opt, self.tmp_path, scripted)
        self.assertTrue(result)
        self.assertEqual(len(_summaries(opt)), 1,
                         "identical summary text must never repeat")
        self.assertEqual(opt.iteration, 2)

    def test_non_api_exception_keeps_legacy_behavior(self):
        # A real logic/tool error is NOT an API episode: the legacy
        # "An error occurred" append and the iteration burn both stay
        # (the LLM needs to see genuine tool failures).
        opt = _make_loop_optimizer(self.tmp_path)
        result, _ = _run_optimize(opt, self.tmp_path, [
            ValueError("tool argument schema broke"),
            ("optimization complete", True),
        ])
        self.assertTrue(result)
        self.assertEqual(len(_error_appends(opt)), 1,
                         "non-API exceptions keep the legacy append")
        self.assertEqual(opt.iteration, 2,
                         "non-API failures still consume the iteration")
        self.assertEqual(opt._consecutive_permanent_api_failures, 0)

    def test_kill_switch_off_restores_legacy_behavior(self):
        # _api_resilience_enabled = False must reproduce the pre-change
        # loop behavior even for key-401s: append + iteration burn.
        opt = _make_loop_optimizer(self.tmp_path)
        opt._api_resilience_enabled = False
        result, _ = _run_optimize(opt, self.tmp_path, [
            _FakeAuthError(_AUTH_401),
            ("optimization complete", True),
        ])
        self.assertTrue(result)
        self.assertEqual(len(_error_appends(opt)), 1)
        self.assertEqual(opt.iteration, 2)

    def test_permanent_death_breaks_loop_after_bound(self):
        # Permanent failures do not increment the iteration count, so neither
        # the iteration guard nor an absent wall-time limit terminates this
        # scenario. The consecutive-failure bound exits as `api_dead` and
        # continues into finalization.
        opt = _make_loop_optimizer(self.tmp_path, max_wall=None)

        calls = {"n": 0}

        async def always_fail():
            calls["n"] += 1
            opt._api_error_episodes += 1
            raise _FakeAuthError(_AUTH_401)

        result, output_dcp = _run_optimize(opt, self.tmp_path, always_fail)
        self.assertEqual(calls["n"], PERMANENT_API_FAILURE_LIMIT,
                         "loop must break after the bounded episode count")
        self.assertEqual(opt.iteration, 0,
                         "all failed calls compensated — zero consumed")
        self.assertEqual(_error_appends(opt), [])
        self.assertEqual(len(_summaries(opt)), 0,
                         "no summary on unresolved (dead-LLM) episodes")
        # Finalize still ran and shipped a valid artifact (baseline here —
        # nothing was banked in this unit test; the banked-ship replay is
        # the Task-2 integration test).
        self.assertTrue(result)
        self.assertTrue(output_dcp.exists())
        self.assertTrue(str(opt.final_status).startswith(
            "VALID_FALLBACK_BASELINE"))

    def test_stats_block_prints_episode_and_backoff_counters(self):
        # R-D2-4: run-end forensics — the ITERATION STATS block must
        # surface both counters so an eval-log storm is reconstructable
        # from the summary alone.
        opt = _make_loop_optimizer(self.tmp_path)
        opt._api_error_episodes = 2
        opt._total_backoff_s = 105.0
        buf = io.StringIO()
        with mock.patch.object(opt, "_persist_to_strategy_memory"), \
             redirect_stdout(buf):
            opt._print_optimization_summary()
        out = buf.getvalue()
        self.assertIn("API error episodes:", out)
        self.assertIn("2", out)
        self.assertIn("Total backoff:", out)
        self.assertIn("105s", out)


# Integration tests exercise the complete completion and backoff path.
# Only the API boundary is scripted; sleeps and the FPGA-tool session are
# inert stubs, so no network access, real sessions, or real delays occur.


class _FakeMessage:
    """Tool-less assistant message digestible by process_response."""

    def __init__(self, content: str):
        self.content = content
        self.tool_calls = None

    def model_dump(self, exclude_none: bool = False) -> dict:
        return {"role": "assistant", "content": self.content}


class _FakeChoice:
    def __init__(self, message: _FakeMessage):
        self.message = message


class _FakeCompletion:
    """Minimal OpenAI-SDK-shaped completion: choices[0].message present,
    usage/error None so get_completion's accounting branches no-op."""

    usage = None
    error = None

    def __init__(self, content: str):
        self.choices = [_FakeChoice(_FakeMessage(content))]


class _ScriptedChat:
    """Stand-in for DCPOptimizer._chat_create (test_api_resilience.py
    pattern).  Pops the next script item per call: Exception instances
    raise, anything else is returned.  Records the model of every call.
    Over-calling fails loudly instead of looping."""

    def __init__(self, script):
        self.script = list(script)
        self.models: list[str] = []

    def __call__(self, model):
        self.models.append(model)
        if not self.script:
            raise AssertionError("unexpected extra _chat_create call")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _AlwaysAuthFail:
    """_chat_create stand-in: every call raises the eval-shape key-401."""

    def __init__(self):
        self.models: list[str] = []

    def __call__(self, model):
        self.models.append(model)
        raise _FakeAuthError(_AUTH_401)


def _run_optimize_real_call_path(opt: DCPOptimizer, tmp_path: Path,
                                 chat) -> tuple[bool, Path]:
    """Drive the REAL optimize() loop + REAL get_completion offline.

    Unlike _run_optimize, get_completion is NOT mocked — the storm is
    absorbed (or exhausted) by the genuine 02-01 backoff path; only the
    SDK boundary (_chat_create) is scripted.
    """
    input_dcp = tmp_path / "baseline.dcp"
    input_dcp.write_bytes(b"BASELINE_DCP_BYTES_FOR_REPLAY")
    output_dcp = tmp_path / "optimized.dcp"

    async def fake_analysis(_input_dcp):
        opt.initial_wns = -10.0
        opt.clock_period = 4.0
        return "ANALYSIS: initial WNS -10.000 ns"

    opt._chat_create = chat  # instance attr shadows the bound method
    with mock.patch("dcp_optimizer.load_system_prompt",
                    return_value="SYS PROMPT (replay test)"), \
         mock.patch.object(opt, "perform_initial_analysis",
                           side_effect=fake_analysis), \
         mock.patch.object(opt, "call_tool",
                           new=mock.AsyncMock(return_value="ok")), \
         mock.patch.object(opt, "_persist_to_strategy_memory"), \
         redirect_stdout(io.StringIO()):
        result = _async(opt.optimize(input_dcp, output_dcp))
    return result, output_dcp


class RecoveryMidStormReplayTests(unittest.TestCase):
    """Replay a transient authentication-error episode that recovers.

    The retry sequence remains on the primary model and consumes backoff time
    without consuming optimization iterations. Processing continues after
    recovery, with at most one compact conversation summary for the episode.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_recovery_mid_storm_same_model_wall_only(self):
        opt = _make_loop_optimizer(self.tmp_path, mode="anchor",
                                   max_wall=7200.0)
        model_before = opt.model
        sleeps: list[tuple[float, float]] = []  # (dur, remaining-at-issue)
        opt._backoff_sleep = (
            lambda d: sleeps.append((d, opt._budget_remaining())))

        chat = _ScriptedChat(
            [_FakeAuthError(_AUTH_401)] * 3
            + [_FakeCompletion("Optimization complete — final design saved.")]
        )
        result, _output = _run_optimize_real_call_path(
            opt, self.tmp_path, chat)

        self.assertTrue(result)
        # R-D2-4: the run NEVER switched models — a key-level outage must
        # not pin the fallback (it shares the same dead key).
        self.assertEqual(opt.model, model_before)
        self.assertNotEqual(opt.model, FALLBACK_MODEL)
        self.assertEqual(set(chat.models), {model_before})
        # One episode, absorbed entirely inside the call path.
        self.assertEqual(opt._api_error_episodes, 1)
        # Backoff counter equals the injected-sleep sum (15+30+60), and
        # every sleep respected the remaining budget at issue time.
        self.assertEqual([d for d, _r in sleeps], [15.0, 30.0, 60.0])
        self.assertEqual(opt._total_backoff_s, sum(d for d, _r in sleeps))
        for d, remaining in sleeps:
            self.assertLessEqual(d, remaining,
                                 "no sleep may exceed remaining budget")
        # R-D2-2: the storm consumed NO iterations — one get_completion,
        # one counted iteration.
        self.assertEqual(opt.iteration, 1)
        self.assertEqual(opt.llm_call_count, 1)
        # R-D2-3: at most one compact episode summary, zero junk appends.
        self.assertEqual(_error_appends(opt), [])
        self.assertLessEqual(len(_summaries(opt)), 1)
        self.assertEqual(len(_summaries(opt)), 1)


class PermanentDeathReplayTests(unittest.TestCase):
    """Local repro shape (boom_debugwall_run1 2026-07-19 18:53:02): the key
    NEVER recovers.  The run must exhaust bounded backoff episodes on the
    SAME model, terminate, and finalize by shipping the Phase-1 BANKED
    best_valid — not an instant baseline death."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_permanent_death_finalizes_banked_best_valid(self):
        opt = _make_loop_optimizer(self.tmp_path, mode="v0_3",
                                   max_wall=7200.0)
        model_before = opt.model
        # Pre-seed Phase-1 banked state (FinalizeFastPathTests pattern):
        # a routed best_valid mirror on disk + fresh WNS markers, so the
        # finalize fast path is exercisable without Vivado.
        best_dcp = self.tmp_path / "best_valid.dcp"
        best_edf = self.tmp_path / "best_valid.edf"
        best_dcp.write_bytes(b"BANKED_BEST_VALID_DCP_CONTENT")
        best_edf.write_bytes(b"BANKED_BEST_VALID_EDIF_CONTENT")
        opt._best_valid_dcp = best_dcp
        opt._best_valid_edif = best_edf
        opt._best_valid_dcp_wns = -1.0
        opt._best_valid_edif_wns = -1.0
        opt.best_wns = -1.0  # improved over initial (-10.0, set by analysis)

        sleeps: list[tuple[float, float]] = []
        opt._backoff_sleep = (
            lambda d: sleeps.append((d, opt._budget_remaining())))

        chat = _AlwaysAuthFail()
        result, output_dcp = _run_optimize_real_call_path(
            opt, self.tmp_path, chat)

        # Model never switched — key-level death must not pin fallback.
        self.assertEqual(opt.model, model_before)
        self.assertEqual(set(chat.models), {model_before})
        # Bounded termination: PERMANENT_API_FAILURE_LIMIT episodes, each
        # a full in-call backoff schedule (5 sleeps + cap give-up on the
        # 6th failure) — 6 SDK calls and 465s absorbed backoff per episode.
        self.assertEqual(opt._api_error_episodes,
                         PERMANENT_API_FAILURE_LIMIT)
        self.assertEqual(len(chat.models), 6 * PERMANENT_API_FAILURE_LIMIT)
        self.assertEqual([d for d, _r in sleeps],
                         [15.0, 30.0, 60.0, 120.0, 240.0]
                         * PERMANENT_API_FAILURE_LIMIT)
        self.assertEqual(opt._total_backoff_s, sum(d for d, _r in sleeps))
        # Deadline safety: every recorded sleep fit the remaining budget
        # at issue time (T-02-01).
        for d, remaining in sleeps:
            self.assertLessEqual(d, remaining)
        # R-D2-2: zero iterations consumed by the storm.
        self.assertEqual(opt.iteration, 0)
        # R-D2-3: no junk appends, no summary (episodes never resolved —
        # the dead LLM would never read one).
        self.assertEqual(_error_appends(opt), [])
        self.assertLessEqual(len(_summaries(opt)), 1)
        self.assertEqual(len(_summaries(opt)), 0)
        # THE combined D1+D2 truth: finalize shipped the Phase-1 BANKED
        # best_valid (routed state), not a baseline instant-death.  Same
        # lifecycle/status fields the Phase-1 FinalizeFastPath tests pin.
        self.assertTrue(result)
        self.assertEqual(opt.final_status, "VALID_OPTIMIZED")
        self.assertTrue(output_dcp.exists())
        self.assertEqual(output_dcp.read_bytes(), best_dcp.read_bytes())
        self.assertEqual(output_dcp.with_suffix(".edf").read_bytes(),
                         best_edf.read_bytes())
        events = [e["event"] for e in opt.lifecycle_log]
        self.assertIn("fast_path_best_valid_copy", events)


if __name__ == "__main__":
    unittest.main()
