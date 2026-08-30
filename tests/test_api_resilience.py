"""Test API error classification and deadline-aware backoff scheduling.

The tests use pure functions without network access, optimizer execution, clock
advancement, or real sleeps. Authentication failures remain key-level errors
and must not trigger model-unavailable fallback behavior.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizer.api_resilience import (
    API_BACKOFF_EPISODE_CAP_S,
    API_BACKOFF_SCHEDULE_S,
    classify_api_error,
    compute_backoff_sleep,
)

import tempfile
import time as _time

from dcp_optimizer import (
    DCPOptimizer,
    DEFAULT_MODEL,
    FALLBACK_MODEL,
    TRANSIENT_RETRY_BACKOFF_S,
)


# classify_api_error — precedence: prompt_limit → key_auth → transient →
# model_unavailable → other

class ClassifyApiErrorTests(unittest.TestCase):

    def test_key_auth_user_not_found_401(self):
        self.assertEqual(
            classify_api_error(
                "AuthenticationError: Error code: 401 - {'error': "
                "{'message': 'User not found.', 'code': 401}}"),
            "key_auth")

    def test_key_auth_plain_401(self):
        self.assertEqual(
            classify_api_error("AuthenticationError: User not found. 401"),
            "key_auth")

    def test_key_auth_403_unauthorized(self):
        # 403/unauthorized is KEY-level, NOT model-unavailable (R-D2-4).
        self.assertEqual(
            classify_api_error("PermissionDeniedError: 403 unauthorized"),
            "key_auth")

    def test_key_auth_invalid_api_key(self):
        self.assertEqual(
            classify_api_error("AuthenticationError: invalid api key"),
            "key_auth")

    def test_key_auth_no_auth_credentials(self):
        self.assertEqual(
            classify_api_error("AuthenticationError: no auth credentials"),
            "key_auth")

    def test_ordering_guard_user_not_found_is_key_auth(self):
        # Key-auth classification must precede the broader "not found"
        # model-unavailable check to avoid misclassifying account errors.
        self.assertEqual(classify_api_error("User not found."), "key_auth")

    def test_model_unavailable_404(self):
        self.assertEqual(
            classify_api_error(
                "NotFoundError: model x-ai/grok-4.3 is not a valid model 404"),
            "model_unavailable")

    def test_model_unavailable_no_endpoints(self):
        self.assertEqual(
            classify_api_error("NotFoundError: No endpoints found for model"),
            "model_unavailable")

    def test_model_unavailable_deprecated(self):
        self.assertEqual(
            classify_api_error("BadRequestError: model has been deprecated"),
            "model_unavailable")

    def test_transient_429(self):
        self.assertEqual(
            classify_api_error("RateLimitError: 429 rate limit"), "transient")

    def test_transient_connection_reset(self):
        self.assertEqual(
            classify_api_error("APIConnectionError: connection reset"),
            "transient")

    def test_transient_5xx_and_timeout(self):
        for msg in ("InternalServerError: 500 internal server error",
                    "APIStatusError: 503 service unavailable",
                    "APITimeoutError: Request timed out"):
            self.assertEqual(classify_api_error(msg), "transient", msg)

    def test_prompt_limit_402(self):
        self.assertEqual(
            classify_api_error("Error code: 402 Prompt tokens limit exceeded"),
            "prompt_limit")

    def test_other(self):
        self.assertEqual(
            classify_api_error("ValueError: something else"), "other")

    def test_other_bad_tool_schema(self):
        self.assertEqual(
            classify_api_error("invalid request: bad tool schema"), "other")


# compute_backoff_sleep — schedule, saturation, episode cap, deadline clamp

class ComputeBackoffSleepTests(unittest.TestCase):

    def _sleep(self, attempt, remaining=10_000.0, guard=60.0, used=0.0):
        return compute_backoff_sleep(
            attempt=attempt, remaining_budget_s=remaining,
            finalize_guard_s=guard, backoff_used_s=used)

    def test_schedule_steps(self):
        self.assertEqual(self._sleep(0), 15.0)
        self.assertEqual(self._sleep(1), 30.0)
        self.assertEqual(self._sleep(2), 60.0)
        self.assertEqual(self._sleep(3), 120.0)
        self.assertEqual(self._sleep(4), 240.0)

    def test_schedule_saturates_past_last_step(self):
        # attempt >= len(schedule) returns the last step, not IndexError.
        self.assertEqual(self._sleep(5), 240.0)
        self.assertEqual(self._sleep(50), 240.0)

    def test_episode_cap_returns_none(self):
        # Full schedule sums to 465s; the saturated next step (240) would
        # push past the 600s cap → give up → propagate to the LLM-dead path.
        self.assertEqual(sum(API_BACKOFF_SCHEDULE_S), 465.0)
        self.assertIsNone(self._sleep(5, used=465.0))
        # Boundary: exactly reaching the cap is allowed (>= only past it).
        self.assertEqual(
            self._sleep(5, used=API_BACKOFF_EPISODE_CAP_S - 240.0), 240.0)
        self.assertIsNone(
            self._sleep(5, used=API_BACKOFF_EPISODE_CAP_S - 239.0))

    def test_deadline_clamp(self):
        # remaining - guard < scheduled step → clamp to the positive
        # remainder, never sleep past the finalize reserve (T-02-01).
        self.assertEqual(self._sleep(0, remaining=65.0, guard=60.0), 5.0)
        self.assertEqual(self._sleep(3, remaining=100.0, guard=60.0), 40.0)

    def test_deadline_exhausted_returns_none(self):
        # No room to sleep AND still finalize → None (propagate).
        self.assertIsNone(self._sleep(0, remaining=60.0, guard=60.0))
        self.assertIsNone(self._sleep(0, remaining=10.0, guard=60.0))
        self.assertIsNone(self._sleep(0, remaining=0.0, guard=60.0))

    def test_uncapped_budget_returns_finite_step(self):
        # remaining inf (no wall cap set) → plain schedule step; the
        # returned sleep is always >= 0 and never NaN/inf.
        val = self._sleep(0, remaining=float("inf"))
        self.assertEqual(val, 15.0)
        val = self._sleep(9, remaining=float("inf"))
        self.assertEqual(val, 240.0)

    def test_nan_budget_returns_none(self):
        # A NaN remaining budget must fail CLOSED (None), never return NaN.
        self.assertIsNone(self._sleep(0, remaining=float("nan")))

    def test_never_negative(self):
        for attempt in range(6):
            v = self._sleep(attempt, remaining=61.0, guard=60.0)
            if v is not None:
                self.assertGreaterEqual(v, 0.0)


# These tests script the API boundary and replace backoff sleeps with a
# recorder. A controlled deadline drives budget calculations without
# network access or real delays.

_AUTH_401 = ("Error code: 401 - {'error': {'message': 'User not found.', "
             "'code': 401}}")
_MODEL_404 = "404 No endpoints found for model x-ai/grok-4.3"
_TRANSIENT_429 = "429 rate limit exceeded"
_PROMPT_402 = "Error code: 402 - Prompt tokens limit exceeded"


class _ScriptedChat:
    """Stand-in for DCPOptimizer._chat_create.  Pops the next script item
    per call: Exception instances raise, anything else is returned.
    Records the model of every call.  Over-calling the script fails the
    test loudly instead of looping."""

    def __init__(self, script):
        self.script = list(script)
        self.models = []

    def __call__(self, model):
        self.models.append(model)
        if not self.script:
            raise AssertionError("unexpected extra _chat_create call")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _AlwaysFail:
    """chat_create stand-in that fails every call with `error`."""

    def __init__(self, error_text):
        self.error_text = error_text
        self.models = []

    def __call__(self, model):
        self.models.append(model)
        raise Exception(self.error_text)


class _FailOnModels:
    """Fails while `model` is in fail_models, succeeds otherwise."""

    def __init__(self, fail_models, error_text):
        self.fail_models = fail_models
        self.error_text = error_text
        self.models = []

    def __call__(self, model):
        self.models.append(model)
        if model in self.fail_models:
            raise Exception(self.error_text)
        return {"ok": model}


class ResilienceCallPathTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = DCPOptimizer(api_key="test", run_dir=Path(self.tmp.name))
        self.opt.messages = [{"role": "user", "content": "hi"}]
        self.opt.tools = []
        self.sleeps = []
        self.opt._backoff_sleep = self.sleeps.append  # recorder, no real sleep

    def tearDown(self):
        self.tmp.cleanup()

    def test_kill_switch_defaults_on_and_sleep_injectable(self):
        self.assertTrue(self.opt._api_resilience_enabled)
        fresh = DCPOptimizer(api_key="test", run_dir=Path(self.tmp.name))
        self.assertIs(fresh._backoff_sleep, _time.sleep)
        self.assertEqual(fresh._api_error_episodes, 0)
        self.assertEqual(fresh._total_backoff_s, 0.0)
        self.assertEqual(fresh._api_backoff_used_s, 0.0)

    def test_key_401_retries_same_model_never_fallback(self):
        # R-D2-4 core truth: a key-level 401 storm that recovers on the
        # 3rd attempt keeps the PRIMARY model; sleeps follow 15, 30.
        chat = _ScriptedChat([Exception(_AUTH_401), Exception(_AUTH_401),
                              {"ok": DEFAULT_MODEL}])
        self.opt._chat_create = chat
        res = self.opt._create_completion_with_fallback()
        self.assertEqual(res, {"ok": DEFAULT_MODEL})
        self.assertEqual(self.opt.model, DEFAULT_MODEL)       # NEVER switched
        self.assertEqual(chat.models, [DEFAULT_MODEL] * 3)    # same model
        self.assertEqual(self.sleeps, [15.0, 30.0])
        self.assertEqual(self.opt._api_error_episodes, 1)
        self.assertEqual(self.opt._total_backoff_s, 45.0)
        self.assertEqual(self.opt._api_backoff_used_s, 0.0)   # reset on success
        self.assertEqual(self.opt._consecutive_transient_failures, 0)

    def test_model_404_still_switches_to_fallback(self):
        # REGRESSION for the R-D2-4 distinction: a genuine model-level 404
        # keeps the existing one-shot fallback behavior.
        chat = _ScriptedChat([Exception(_MODEL_404), {"ok": FALLBACK_MODEL}])
        self.opt._chat_create = chat
        res = self.opt._create_completion_with_fallback()
        self.assertEqual(res, {"ok": FALLBACK_MODEL})
        self.assertEqual(self.opt.model, FALLBACK_MODEL)
        self.assertEqual(chat.models, [DEFAULT_MODEL, FALLBACK_MODEL])
        self.assertEqual(self.sleeps, [])                     # no backoff

    def test_transient_backoff_then_pin_fallback_at_threshold(self):
        # Transient keeps the existing pin-fallback-after-threshold escape:
        # 2 backoff retries on primary (15, 30), 3rd consecutive failure
        # hits TRANSIENT_FALLBACK_THRESHOLD=3 -> pin fallback (succeeds).
        chat = _FailOnModels({DEFAULT_MODEL}, _TRANSIENT_429)
        self.opt._chat_create = chat
        res = self.opt._create_completion_with_fallback()
        self.assertEqual(res, {"ok": FALLBACK_MODEL})
        self.assertEqual(self.opt.model, FALLBACK_MODEL)
        self.assertEqual(chat.models,
                         [DEFAULT_MODEL] * 3 + [FALLBACK_MODEL])
        self.assertEqual(self.sleeps, [15.0, 30.0])

    def test_key_auth_never_pins_fallback_cap_then_propagate(self):
        # Key-auth failures exhaust the retry schedule without triggering the
        # transient-model escape. Once the episode cap admits no further
        # delay, the error propagates to the API-dead path.
        chat = _AlwaysFail(_AUTH_401)
        self.opt._chat_create = chat
        with self.assertRaises(Exception):
            self.opt._create_completion_with_fallback()
        self.assertEqual(self.opt.model, DEFAULT_MODEL)       # NEVER switched
        self.assertEqual(self.sleeps, [15.0, 30.0, 60.0, 120.0, 240.0])
        self.assertEqual(set(chat.models), {DEFAULT_MODEL})
        self.assertEqual(self.opt._consecutive_transient_failures, 0)

    def test_deadline_aware_single_clamped_sleep_then_propagate(self):
        # With 65 s left, backoff may consume only the roughly 5 s above the
        # 60 s finalization reserve. Further backoff then fails closed and
        # propagates, preserving the reserve for finalization.
        self.opt._budget_deadline = _time.time() + 65.0

        def rec(d):
            self.sleeps.append(d)
            self.opt._budget_deadline -= d   # sleeping consumes budget

        self.opt._backoff_sleep = rec
        chat = _AlwaysFail(_AUTH_401)
        self.opt._chat_create = chat
        with self.assertRaises(Exception):
            self.opt._create_completion_with_fallback()
        self.assertEqual(len(self.sleeps), 1)                 # exactly one
        self.assertAlmostEqual(self.sleeps[0], 5.0, delta=0.5)
        self.assertEqual(self.opt.model, DEFAULT_MODEL)

    def test_prompt_limit_402_path_unchanged(self):
        # 402 prompt-cap handling (the prompt guard) never enters
        # backoff: prune-and-retry as before.
        chat = _ScriptedChat([Exception(_PROMPT_402), {"ok": DEFAULT_MODEL}])
        self.opt._chat_create = chat
        res = self.opt._create_completion_with_fallback()
        self.assertEqual(res, {"ok": DEFAULT_MODEL})
        self.assertEqual(self.opt.model, DEFAULT_MODEL)
        self.assertEqual(self.sleeps, [])                     # no backoff

    def test_episode_counters_across_two_episodes(self):
        for _ in range(2):
            chat = _ScriptedChat([Exception(_AUTH_401), {"ok": DEFAULT_MODEL}])
            self.opt._chat_create = chat
            self.opt._create_completion_with_fallback()
        self.assertEqual(self.opt._api_error_episodes, 2)
        self.assertEqual(self.opt._total_backoff_s, 30.0)     # 15 + 15
        self.assertEqual(self.opt._api_backoff_used_s, 0.0)

    # ---- kill switch OFF: byte-for-byte legacy behavior ----

    def test_kill_switch_off_legacy_auth_as_model_unavailable(self):
        # Legacy (pre-R-D2-4) lumped 401 into model-unavailable and
        # switched to the fallback model, which 401s too (same dead key).
        self.opt._api_resilience_enabled = False
        chat = _AlwaysFail(_AUTH_401)
        self.opt._chat_create = chat
        with self.assertRaises(Exception):
            self.opt._create_completion_with_fallback()
        self.assertEqual(self.opt.model, FALLBACK_MODEL)      # legacy switch
        self.assertEqual(chat.models, [DEFAULT_MODEL, FALLBACK_MODEL])
        self.assertEqual(self.sleeps, [])

    def test_kill_switch_off_legacy_single_2s_transient_retry(self):
        self.opt._api_resilience_enabled = False
        chat = _AlwaysFail(_TRANSIENT_429)
        self.opt._chat_create = chat
        with self.assertRaises(Exception):
            self.opt._create_completion_with_fallback()
        self.assertEqual(self.opt.model, DEFAULT_MODEL)       # below threshold
        self.assertEqual(chat.models, [DEFAULT_MODEL, DEFAULT_MODEL])
        self.assertEqual(self.sleeps, [TRANSIENT_RETRY_BACKOFF_S])
        self.assertEqual(self.opt._consecutive_transient_failures, 1)


if __name__ == "__main__":
    unittest.main()
