"""Unit tests for the grok-4.3 -> gemini model fallback (no network)."""
import pytest
from dcp_optimizer import DCPOptimizer, DEFAULT_MODEL, FALLBACK_MODEL


def _opt():
    o = DCPOptimizer(api_key="dummy", model=DEFAULT_MODEL)
    o.messages = [{"role": "user", "content": "hi"}]
    o.tools = []
    return o


class _FakeCompletions:
    def __init__(self, fail_models, error):
        self.fail_models = fail_models
        self.error = error
        self.calls = []
    def create(self, **kw):
        self.calls.append(kw["model"])
        if kw["model"] in self.fail_models:
            raise self.error
        return {"ok": kw["model"]}


class _FakeOpenAI:
    def __init__(self, completions):
        self.chat = type("C", (), {"completions": completions})()


def test_classifier_model_errors_true():
    """R-D2-4 reclassification (2026-07-20): '401 Unauthorized' was REMOVED
    from this list — key-level auth errors are no longer model-unavailable
    (they spuriously pinned the fallback model during the eval-day 401
    storm; see tests/test_api_resilience.py).  They now classify as
    key_auth via _is_key_level_auth_error."""
    for msg in ["404 model not found", "No endpoints found for x-ai/grok-4.3",
                "model_not_found", "is not a valid model",
                "model has been deprecated"]:
        assert DCPOptimizer._is_model_unavailable_error(Exception(msg)) is True
    # key-level auth: NOT model-unavailable anymore (R-D2-4)
    for msg in ["401 Unauthorized", "User not found.", "403 permission"]:
        assert DCPOptimizer._is_model_unavailable_error(Exception(msg)) is False
        assert DCPOptimizer._is_key_level_auth_error(Exception(msg)) is True


def test_classifier_transient_errors_false():
    for msg in ["429 rate limit exceeded", "Request timed out",
                "500 internal server error", "connection reset"]:
        assert DCPOptimizer._is_model_unavailable_error(Exception(msg)) is False


def test_fallback_on_model_unavailable():
    o = _opt()
    fc = _FakeCompletions(fail_models={DEFAULT_MODEL},
                          error=Exception("404 No endpoints found"))
    o.openai = _FakeOpenAI(fc)
    res = o._create_completion_with_fallback()
    assert res == {"ok": FALLBACK_MODEL}
    assert o.model == FALLBACK_MODEL            # pinned for rest of run
    assert fc.calls == [DEFAULT_MODEL, FALLBACK_MODEL]


def test_transient_error_retries_once_no_switch_below_threshold(monkeypatch):
    """LEGACY path (kill switch OFF).  R-D2-1 (2026-07-20): with resilience
    ON (default) a persistent transient is absorbed inside one call — it
    backoff-retries on the schedule and pins the fallback at the threshold
    (covered in tests/test_api_resilience.py).  This test pins the
    pre-change single-2s-retry behavior behind _api_resilience_enabled."""
    import dcp_optimizer as d
    monkeypatch.setattr(d.time, "sleep", lambda s: None)
    o = _opt()
    o._api_resilience_enabled = False
    fc = _FakeCompletions(fail_models={DEFAULT_MODEL},
                          error=Exception("429 rate limit"))
    o.openai = _FakeOpenAI(fc)
    with pytest.raises(Exception):
        o._create_completion_with_fallback()
    assert o.model == DEFAULT_MODEL             # below threshold: no switch
    assert fc.calls == [DEFAULT_MODEL, DEFAULT_MODEL]   # one backoff retry
    assert o._consecutive_transient_failures == 1


class _FlakyCompletions:
    """Fails the first `fail_first_n` calls with `error`, then succeeds."""
    def __init__(self, fail_first_n, error):
        self.fail_first_n = fail_first_n
        self.error = error
        self.calls = []
    def create(self, **kw):
        self.calls.append(kw["model"])
        if len(self.calls) <= self.fail_first_n:
            raise self.error
        return {"ok": kw["model"]}


def test_transient_retry_recovers(monkeypatch):
    import dcp_optimizer as d
    monkeypatch.setattr(d.time, "sleep", lambda s: None)
    o = _opt()
    fc = _FlakyCompletions(1, Exception("503 service unavailable"))
    o.openai = _FakeOpenAI(fc)
    res = o._create_completion_with_fallback()
    assert res == {"ok": DEFAULT_MODEL}
    assert o.model == DEFAULT_MODEL
    assert o._consecutive_transient_failures == 0   # reset on success


def test_transient_persists_pins_fallback_at_threshold(monkeypatch):
    """LEGACY path (kill switch OFF).  R-D2-1 (2026-07-20): with resilience
    ON the threshold pin happens INSIDE a single call after schedule
    backoffs (tests/test_api_resilience.py).  This pins the pre-change
    across-calls threshold behavior behind _api_resilience_enabled."""
    import dcp_optimizer as d
    monkeypatch.setattr(d.time, "sleep", lambda s: None)
    o = _opt()
    o._api_resilience_enabled = False
    fc = _FakeCompletions(fail_models={DEFAULT_MODEL},
                          error=Exception("Request timed out"))
    o.openai = _FakeOpenAI(fc)
    for _ in range(d.TRANSIENT_FALLBACK_THRESHOLD - 1):
        with pytest.raises(Exception):
            o._create_completion_with_fallback()
        assert o.model == DEFAULT_MODEL
    # threshold-th consecutive transient failure -> pin the fallback model
    res = o._create_completion_with_fallback()
    assert res == {"ok": FALLBACK_MODEL}
    assert o.model == FALLBACK_MODEL


def test_non_transient_non_model_error_raises_immediately():
    o = _opt()
    fc = _FakeCompletions(fail_models={DEFAULT_MODEL},
                          error=Exception("invalid request: bad tool schema"))
    o.openai = _FakeOpenAI(fc)
    with pytest.raises(Exception):
        o._create_completion_with_fallback()
    assert fc.calls == [DEFAULT_MODEL]          # no retry, no switch
    assert o.model == DEFAULT_MODEL


def test_primary_success_no_fallback():
    o = _opt()
    fc = _FakeCompletions(fail_models=set(), error=Exception("x"))
    o.openai = _FakeOpenAI(fc)
    res = o._create_completion_with_fallback()
    assert res == {"ok": DEFAULT_MODEL}
    assert fc.calls == [DEFAULT_MODEL]
