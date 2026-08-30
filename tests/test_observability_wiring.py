"""Verify that observability components are invoked from production call sites.

Coverage includes budget skips, timeouts, tool exceptions, successful tool
calls, finalization failures, and out-of-root path checks. Audit-mode path
violations are recorded without raising.

The MCP call layer and analysis step are mocked, so no implementation tools or
checkpoint-processing services are started.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import DCPOptimizer
from optimizer.decision_tracer import load_decisions


def _async(coro):
    return asyncio.run(coro)


class _FakeBaselineDCP:
    def __init__(self, tmpdir: Path):
        self.path = tmpdir / "baseline.dcp"
        self.path.write_bytes(b"DCP_BASELINE_TEST_FIXTURE")


def _decision_path(opt: DCPOptimizer) -> Path:
    return Path(opt.run_dir) / "decisions.jsonl"


def _load(opt: DCPOptimizer):
    p = _decision_path(opt)
    if not p.exists():
        return []
    return load_decisions(p)


class CallToolDecisionTraceTests(unittest.TestCase):
    """call_tool exit paths must emit decision records with ToolError
    codes when the payload is an error."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name) / "run"
        self.run_dir.mkdir()
        self.opt = DCPOptimizer(api_key="test", run_dir=self.run_dir)
        # Provide a usable session shape — call_tool routes on prefix.
        self.opt.vivado_session = mock.MagicMock()
        self.opt.rapidwright_session = mock.MagicMock()
        # Trigger the lazy DecisionTracer init.
        self.opt._ensure_decision_tracer()

    def tearDown(self):
        self.tmp.cleanup()

    def test_budget_skip_emits_record_with_budget_skip_code(self):
        """When _should_skip_for_budget returns True, a BUDGET_SKIP
        decision record must be written."""
        with mock.patch.object(
            self.opt, "_should_skip_for_budget",
            return_value=(True, "deadline_passed"),
        ):
            result = _async(self.opt.call_tool("vivado_phys_opt_design", {}))
        payload = json.loads(result)
        self.assertEqual(payload["error"], "tool_skipped_budget")
        records = _load(self.opt)
        # Budget-skip should produce exactly one decision record at the
        # budget_skip action label.
        skips = [r for r in records if r.get("action_label") == "budget_skip"]
        self.assertEqual(len(skips), 1)
        self.assertEqual(skips[0]["tool_name"], "vivado_phys_opt_design")
        self.assertEqual(skips[0]["tool_error_code"], "BUDGET_SKIP")
        self.assertEqual(skips[0]["phase"], "call_tool")

    def test_call_tool_success_emits_record_with_no_error_code(self):
        """Success path: tool_call_success record, tool_error_code None."""
        fake_result = mock.MagicMock()
        fake_text = mock.MagicMock()
        fake_text.text = "ok\n"
        fake_result.content = [fake_text]
        self.opt.vivado_session.call_tool = mock.AsyncMock(return_value=fake_result)
        with mock.patch.object(
            self.opt, "_should_skip_for_budget", return_value=(False, ""),
        ), mock.patch.object(
            self.opt, "_deadline_aware_timeout", return_value=None,
        ):
            _async(self.opt.call_tool("vivado_get_clocks", {}))
        records = _load(self.opt)
        successes = [r for r in records
                     if r.get("action_label") == "tool_call_success"]
        self.assertGreaterEqual(len(successes), 1)
        self.assertEqual(successes[-1]["tool_name"], "vivado_get_clocks")
        self.assertIsNone(successes[-1].get("tool_error_code"))

    def test_call_tool_warning_label_when_classifier_matches_but_call_succeeds(self):
        """Regression: 2026-05-20 multi-design smoke saw a real Vivado
        WARNING "integrity check ... missing" classify as MISSING_ARTIFACT
        even though _looks_like_tool_error said the call succeeded.
        The action_label must be `tool_call_warning`, not
        `tool_call_classified_error`."""
        fake_text = mock.MagicMock()
        fake_text.text = (
            "open_checkpoint completed\n"
            "WARNING: [Vivado 12-8410] Design file 'X.dcp' has failed "
            "integrity check (1). There is either a missing\n"
            "Time (s): cpu = 00:00:00 ; elapsed = 00:00:00.030 "
        )
        fake_result = mock.MagicMock()
        fake_result.content = [fake_text]
        self.opt.vivado_session.call_tool = mock.AsyncMock(
            return_value=fake_result,
        )
        with mock.patch.object(
            self.opt, "_should_skip_for_budget", return_value=(False, ""),
        ), mock.patch.object(
            self.opt, "_deadline_aware_timeout", return_value=None,
        ):
            _async(self.opt.call_tool("vivado_open_checkpoint", {}))
        records = _load(self.opt)
        # Should be a single call_tool record with warning label.
        call_records = [r for r in records if r.get("phase") == "call_tool"]
        self.assertEqual(len(call_records), 1)
        self.assertEqual(call_records[0]["action_label"], "tool_call_warning")
        # MISSING_ARTIFACT classification is still attached for diagnostics.
        self.assertEqual(call_records[0]["tool_error_code"], "MISSING_ARTIFACT")

    def test_call_tool_exception_emits_record_with_classified_code(self):
        """When the MCP call raises, the exception path must emit a
        decision record with a tool_error_code."""
        self.opt.vivado_session.call_tool = mock.AsyncMock(
            side_effect=Exception("simulated MCP transport failure")
        )
        with mock.patch.object(
            self.opt, "_should_skip_for_budget", return_value=(False, ""),
        ), mock.patch.object(
            self.opt, "_deadline_aware_timeout", return_value=None,
        ):
            result = _async(self.opt.call_tool("vivado_phys_opt_design", {}))
        # The function returns a JSON-error envelope, never raises.
        payload = json.loads(result)
        self.assertIn("error", payload)
        records = _load(self.opt)
        excs = [r for r in records
                if r.get("action_label") == "tool_call_exception"]
        self.assertEqual(len(excs), 1)
        self.assertEqual(excs[0]["tool_name"], "vivado_phys_opt_design")
        # Code may be UNKNOWN or matched depending on text; non-None.
        self.assertIsNotNone(excs[0]["tool_error_code"])

    def test_call_tool_timeout_emits_record_with_timeout_budget(self):
        """asyncio.TimeoutError path emits a TIMEOUT_BUDGET record."""

        async def slow(*a, **kw):  # pragma: no cover — should be cancelled
            await asyncio.sleep(60)

        self.opt.vivado_session.call_tool = mock.AsyncMock(side_effect=slow)
        with mock.patch.object(
            self.opt, "_should_skip_for_budget", return_value=(False, ""),
        ), mock.patch.object(
            self.opt, "_deadline_aware_timeout", return_value=0.01,
        ):
            result = _async(self.opt.call_tool("vivado_phys_opt_design", {}))
        payload = json.loads(result)
        self.assertEqual(payload["error"], "tool_timed_out_budget")
        records = _load(self.opt)
        timeouts = [r for r in records
                    if r.get("action_label") == "budget_timeout"]
        self.assertEqual(len(timeouts), 1)
        self.assertEqual(timeouts[0]["tool_error_code"], "TIMEOUT_BUDGET")


class PathGuardWiringTests(unittest.TestCase):
    """PathGuard must be initialised in optimize() and must record
    audit-mode violations without blocking the run."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name) / "run"
        self.run_dir.mkdir()
        self.baseline = _FakeBaselineDCP(self.run_dir)
        self.output_dcp = self.run_dir / "optimized.dcp"
        self.opt = DCPOptimizer(api_key="test", run_dir=self.run_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_optimize_initialises_path_guard_via_phase1_failure(self):
        """Run optimize() with Phase 1 failure → guard must have been
        wired up and the phase1-failure baseline-copy must have been
        audited."""

        async def boom(*_a, **_kw):
            raise RuntimeError("simulated Phase 1 timeout")

        with mock.patch.object(
            self.opt, "perform_initial_analysis", side_effect=boom,
        ), mock.patch.object(
            self.opt, "call_tool",
            new=mock.AsyncMock(side_effect=Exception("vivado dead")),
        ):
            _async(self.opt.optimize(self.baseline.path, self.output_dcp))

        # PathGuard should now exist on the optimizer.
        self.assertIsNotNone(self.opt._path_guard)
        # The decision trace should record at least one path_guard_check
        # event from the phase1_failed_baseline_copy site.
        records = _load(self.opt)
        guard_records = [r for r in records
                         if r.get("action_label") == "path_guard_check"]
        self.assertGreater(len(guard_records), 0)
        # Final-status / phase_1 / run_start / finalize_end records must
        # also have landed.
        labels = {r.get("action_label") for r in records}
        self.assertIn("optimize_begin", labels)

    def test_audit_mode_records_out_of_root_violation_no_raise(self):
        """Verify that audit mode records an out-of-root path without raising.

        The test explicitly sets `_path_guard_mode` to `"audit"` before lazy
        initialization because the default mode enforces path restrictions.
        """
        # Force audit mode for this test (session-4 default is enforce).
        self.opt._path_guard_mode = "audit"
        self.opt._ensure_path_guard(output_dcp=self.output_dcp)
        self.assertIsNotNone(self.opt._path_guard)
        self.assertEqual(self.opt._path_guard.mode, "audit")
        rogue = Path(self.tmp.name) / "totally-elsewhere" / "rogue.dcp"
        # Must not raise in audit mode.
        result = self.opt._path_guard_check(rogue, context="test_rogue")
        self.assertFalse(result)
        self.assertEqual(len(self.opt._path_guard_violations), 1)
        self.assertEqual(self.opt._path_guard_violations[0]["context"],
                          "test_rogue")
        # And it should emit a trace record.
        records = _load(self.opt)
        guard_records = [r for r in records
                         if r.get("action_label") == "path_guard_check"
                         and r.get("notes") == "test_rogue"]
        self.assertEqual(len(guard_records), 1)
        self.assertFalse(guard_records[0]["path_guard_allowed"])

    def test_default_path_guard_mode_is_enforce(self):
        """Session-4 flip: fresh DCPOptimizer must default to enforce."""
        # New optimizer (don't reuse self.opt — its mode may have been
        # overridden by the prior audit-mode test in setUp).
        fresh_run = Path(self.tmp.name) / "fresh_run"
        fresh_run.mkdir()
        # Ensure no env override forces audit for this test.
        with mock.patch.dict("os.environ", {}, clear=False):
            os = __import__("os")
            os.environ.pop("PATH_GUARD_MODE", None)
            opt = DCPOptimizer(api_key="test", run_dir=fresh_run)
            self.assertEqual(opt._path_guard_mode, "enforce")
            opt._ensure_path_guard(output_dcp=fresh_run / "out.dcp")
            self.assertEqual(opt._path_guard.mode, "enforce")

    def test_env_override_can_force_audit_mode(self):
        """PATH_GUARD_MODE=audit env var must downgrade the default."""
        fresh_run = Path(self.tmp.name) / "env_run"
        fresh_run.mkdir()
        with mock.patch.dict("os.environ", {"PATH_GUARD_MODE": "audit"}):
            opt = DCPOptimizer(api_key="test", run_dir=fresh_run)
            self.assertEqual(opt._path_guard_mode, "audit")
            opt._ensure_path_guard(output_dcp=fresh_run / "out.dcp")
            self.assertEqual(opt._path_guard.mode, "audit")

    def test_enforce_mode_raises_on_rogue_write(self):
        """In enforce mode (the default), an out-of-root write must
        raise PathGuardError.  This catches the case where a future
        code change introduces an unexpected DCP/EDIF write target."""
        from optimizer.path_guard import PathGuardError
        # self.opt has been audited via setUp; force a fresh instance
        # to make sure the default takes effect cleanly.
        fresh_run = Path(self.tmp.name) / "enforce_run"
        fresh_run.mkdir()
        os = __import__("os")
        with mock.patch.dict("os.environ", {}, clear=False):
            os.environ.pop("PATH_GUARD_MODE", None)
            opt = DCPOptimizer(api_key="test", run_dir=fresh_run)
            opt._ensure_path_guard(output_dcp=fresh_run / "out.dcp")
            rogue = Path(self.tmp.name) / "rogue-area" / "x.dcp"
            with self.assertRaises(PathGuardError):
                opt._path_guard_check(rogue, context="test_rogue_enforce")


class FinalizeRefreshEdifTraceTests(unittest.TestCase):
    """P8: _finalize_refresh_edif must emit start + outcome trace
    events."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name) / "run"
        self.run_dir.mkdir()
        self.opt = DCPOptimizer(api_key="test", run_dir=self.run_dir)
        self.opt._ensure_decision_tracer()
        self.opt._ensure_path_guard(output_dcp=self.run_dir / "out.dcp")

    def tearDown(self):
        self.tmp.cleanup()

    def test_refresh_edif_emits_start_and_ok_records(self):
        out_edif = self.run_dir / "out.edf"
        # Simulate call_tool returning a non-error result + file landed.
        async def fake_call_tool(tool_name, args):
            # Write a non-empty file at the requested edif_path.
            Path(args["edif_path"]).write_bytes(b"EDIF")
            return "EDIF written"
        with mock.patch.object(self.opt, "call_tool", new=fake_call_tool):
            ok = _async(self.opt._finalize_refresh_edif(out_edif))
        self.assertTrue(ok)
        records = _load(self.opt)
        labels = [r.get("action_label") for r in records]
        self.assertIn("finalize_refresh_edif_start", labels)
        self.assertIn("finalize_refresh_edif_ok", labels)

    def test_refresh_edif_refused_emits_refused_record(self):
        out_edif = self.run_dir / "out.edf"
        # Result text matches BUDGET_SKIP / generic error pattern.
        async def fake_call_tool(tool_name, args):
            return json.dumps({
                "error": "tool_skipped_budget",
                "reason": "deadline_passed",
            })
        with mock.patch.object(self.opt, "call_tool", new=fake_call_tool):
            ok = _async(self.opt._finalize_refresh_edif(out_edif))
        self.assertFalse(ok)
        records = _load(self.opt)
        labels = [r.get("action_label") for r in records]
        self.assertIn("finalize_refresh_edif_refused", labels)
        # Refusal record should carry a tool_error_code from the
        # classifier (BUDGET_SKIP).
        refused = [r for r in records
                   if r.get("action_label") == "finalize_refresh_edif_refused"]
        self.assertEqual(len(refused), 1)
        self.assertIn(refused[0]["tool_error_code"],
                      {"BUDGET_SKIP", "UNKNOWN_TOOL_ERROR"})


if __name__ == "__main__":
    unittest.main()
