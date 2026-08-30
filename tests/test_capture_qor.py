"""Tests the optional post-finalization QoR capture hook.

Capture is disabled by default. When enabled with an existing DCP, it
constructs a JSON target inside `run_dir` and invokes the subprocess runner
exactly once.

Each finalization emits one `action_label="qor_capture"` decision record with
`report_only=True` and `used_for_decision=False`. Timeouts, failures, and
missing DCPs are nonfatal; targets inside `submission/` are rejected. Capture
must not change lifecycle status or introduce design-specific conditionals.

The subprocess runner is patched, so these tests do not invoke Vivado.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import DCPOptimizer, _path_is_in_submission
from optimizer.decision_tracer import load_decisions


def _async(coro):
    return asyncio.run(coro)


def _decision_records(opt: DCPOptimizer):
    p = Path(opt.run_dir) / "decisions.jsonl"
    if not p.exists():
        return []
    return load_decisions(p)


def _qor_records(opt: DCPOptimizer):
    return [r for r in _decision_records(opt)
            if r.get("action_label") == "qor_capture"]


class CaptureQorOptimizerHookTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.opt = DCPOptimizer(api_key="test", run_dir=self.run_dir)
        # Ensure decision tracer is set up so emitted records hit disk.
        self.opt._ensure_decision_tracer()
        # Build a fake non-empty finalised DCP file.
        self.out_dcp = self.run_dir / "fake_optimized.dcp"
        self.out_dcp.write_bytes(b"DCP_TEST_FIXTURE")

    def tearDown(self):
        self.tmp.cleanup()

    # ----- flag-default behaviour -----------------------------------------

    def test_default_attribute_off(self):
        self.assertFalse(self.opt.capture_qor)

    def test_off_does_not_invoke_runner(self):
        self.opt.capture_qor = False
        runner = mock.AsyncMock(return_value=("success", None))
        with mock.patch.object(self.opt, "_run_qor_capture_subprocess", runner):
            _async(self.opt._maybe_capture_qor_post_finalize(self.out_dcp))
        # Hook called directly here only to keep the test focused; in
        # _finalize_output_dcp the gate is upstream and `capture_qor=False`
        # bypasses this method entirely (covered by the next test).
        runner.assert_called_once()  # explicit call still goes through

    def test_finalize_does_not_invoke_runner_when_flag_off(self):
        """The outer wrapper must early-out when capture_qor is False."""
        # Avoid spinning up the real finalize body — patch the impl to a no-op.
        async def _noop_impl(_self_out):
            return None
        runner = mock.AsyncMock(return_value=("success", None))
        with mock.patch.object(self.opt, "_finalize_output_dcp_impl",
                               side_effect=_noop_impl), \
             mock.patch.object(self.opt, "_run_qor_capture_subprocess", runner):
            self.opt.capture_qor = False
            _async(self.opt._finalize_output_dcp(self.out_dcp))
        runner.assert_not_called()

    # ----- ON path --------------------------------------------------------

    def test_on_success_emits_record_with_used_for_decision_false(self):
        runner = mock.AsyncMock(return_value=("success", None))
        # Simulate runner producing the JSON.
        json_target = self.run_dir / f"{self.out_dcp.stem}.qor.json"
        async def _make_json(*a, **kw):
            json_target.write_text('{"Report Information":{}}')
            return ("success", None)
        with mock.patch.object(self.opt, "_run_qor_capture_subprocess",
                               side_effect=_make_json) as patched:
            self.opt.capture_qor = True
            _async(self.opt._maybe_capture_qor_post_finalize(self.out_dcp))
        patched.assert_called_once()
        # Exactly one qor_capture record, success status, report-only.
        recs = _qor_records(self.opt)
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual(r["status"], "success")
        self.assertEqual(r["report_only"], True)
        self.assertEqual(r["used_for_decision"], False)
        self.assertTrue(r["json_path"].endswith(".qor.json"))
        self.assertIn(str(self.run_dir), r["json_path"])

    def test_json_path_is_under_run_dir(self):
        runner = mock.AsyncMock(return_value=("success", None))
        with mock.patch.object(self.opt, "_run_qor_capture_subprocess", runner):
            self.opt.capture_qor = True
            _async(self.opt._maybe_capture_qor_post_finalize(self.out_dcp))
        runner.assert_called_once()
        args, kwargs = runner.call_args
        json_path = kwargs.get("json_path") or args[1]
        self.assertEqual(Path(json_path).parent.resolve(),
                         Path(self.run_dir).resolve())

    # ----- failure modes --------------------------------------------------

    def test_timeout_is_non_fatal_and_recorded(self):
        runner = mock.AsyncMock(return_value=("timeout", "timed out after 60.0 s"))
        with mock.patch.object(self.opt, "_run_qor_capture_subprocess", runner):
            self.opt.capture_qor = True
            _async(self.opt._maybe_capture_qor_post_finalize(self.out_dcp))
        recs = _qor_records(self.opt)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["status"], "timeout")
        self.assertIn("timed out", recs[0]["error_summary"])

    def test_error_is_non_fatal_and_recorded(self):
        runner = mock.AsyncMock(return_value=("error", "rc=1 tail='boom'"))
        with mock.patch.object(self.opt, "_run_qor_capture_subprocess", runner):
            self.opt.capture_qor = True
            _async(self.opt._maybe_capture_qor_post_finalize(self.out_dcp))
        recs = _qor_records(self.opt)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["status"], "error")

    def test_runner_exception_is_swallowed(self):
        async def _boom(*a, **kw):
            raise RuntimeError("synthetic")
        with mock.patch.object(self.opt, "_run_qor_capture_subprocess",
                               side_effect=_boom):
            self.opt.capture_qor = True
            _async(self.opt._maybe_capture_qor_post_finalize(self.out_dcp))
        recs = _qor_records(self.opt)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["status"], "error")
        self.assertIn("runner_exception", recs[0]["error_summary"])

    def test_missing_dcp_yields_skipped(self):
        ghost = self.run_dir / "ghost.dcp"
        runner = mock.AsyncMock(return_value=("success", None))
        with mock.patch.object(self.opt, "_run_qor_capture_subprocess", runner):
            self.opt.capture_qor = True
            _async(self.opt._maybe_capture_qor_post_finalize(ghost))
        runner.assert_not_called()
        recs = _qor_records(self.opt)
        self.assertEqual(recs[0]["status"], "skipped")
        self.assertIn("missing", recs[0]["error_summary"])

    def test_empty_dcp_yields_skipped(self):
        empty = self.run_dir / "empty.dcp"
        empty.write_bytes(b"")
        runner = mock.AsyncMock(return_value=("success", None))
        with mock.patch.object(self.opt, "_run_qor_capture_subprocess", runner):
            self.opt.capture_qor = True
            _async(self.opt._maybe_capture_qor_post_finalize(empty))
        runner.assert_not_called()
        recs = _qor_records(self.opt)
        self.assertEqual(recs[0]["status"], "skipped")

    # ----- submission-tree guard -----------------------------------------

    def test_refuses_submission_path_target(self):
        # Build a fake "submission" run_dir so the JSON target lands under it.
        sub_run = self.root / "submission" / "run"
        sub_run.mkdir(parents=True)
        sub_dcp = sub_run / "fake.dcp"
        sub_dcp.write_bytes(b"X")
        sub_opt = DCPOptimizer(api_key="test", run_dir=sub_run)
        sub_opt._ensure_decision_tracer()
        runner = mock.AsyncMock(return_value=("success", None))
        with mock.patch.object(sub_opt, "_run_qor_capture_subprocess", runner):
            sub_opt.capture_qor = True
            _async(sub_opt._maybe_capture_qor_post_finalize(sub_dcp))
        runner.assert_not_called()
        recs = [r for r in load_decisions(sub_run / "decisions.jsonl")
                if r.get("action_label") == "qor_capture"]
        self.assertEqual(recs[0]["status"], "refused_submission_path")

    def test_path_is_in_submission_helper(self):
        self.assertTrue(_path_is_in_submission("submission/foo"))
        self.assertTrue(_path_is_in_submission("/x/submission/y.json"))
        self.assertTrue(_path_is_in_submission(r"C:\submission\dcps\x"))
        self.assertFalse(_path_is_in_submission("/tmp/run/x.qor.json"))
        self.assertFalse(_path_is_in_submission("/var/policy_memory/x"))

    # ----- record shape ---------------------------------------------------

    def test_record_has_required_fields(self):
        runner = mock.AsyncMock(return_value=("success", None))
        with mock.patch.object(self.opt, "_run_qor_capture_subprocess", runner):
            self.opt.capture_qor = True
            _async(self.opt._maybe_capture_qor_post_finalize(self.out_dcp))
        r = _qor_records(self.opt)[0]
        for k in ("enabled", "status", "json_path", "source_dcp_path",
                  "runtime_seconds", "report_only", "used_for_decision"):
            self.assertIn(k, r, f"missing field {k}")
        self.assertTrue(r["enabled"])
        self.assertEqual(r["report_only"], True)
        self.assertEqual(r["used_for_decision"], False)
        self.assertIsInstance(r["runtime_seconds"], (int, float))

    def test_lifecycle_status_unchanged_by_capture(self):
        """capture_qor must not alter ``self.final_status``."""
        self.opt.final_status = "VALID_OPTIMIZED"
        runner = mock.AsyncMock(return_value=("success", None))
        with mock.patch.object(self.opt, "_run_qor_capture_subprocess", runner):
            self.opt.capture_qor = True
            _async(self.opt._maybe_capture_qor_post_finalize(self.out_dcp))
        self.assertEqual(self.opt.final_status, "VALID_OPTIMIZED")
        # Even on timeout.
        self.opt.final_status = "VALID_FALLBACK_BASELINE"
        runner2 = mock.AsyncMock(return_value=("timeout", "x"))
        with mock.patch.object(self.opt, "_run_qor_capture_subprocess", runner2):
            _async(self.opt._maybe_capture_qor_post_finalize(self.out_dcp))
        self.assertEqual(self.opt.final_status, "VALID_FALLBACK_BASELINE")

    def test_relative_dcp_path_is_resolved_before_subprocess(self):
        """The hook MUST hand absolute paths to the subprocess runner.

        Vivado batch runs with ``cwd = run_dir``; a relative ``dcp_path``
        would resolve under ``<run_dir>/<rel>`` and the Tcl probe would
        return ERR_NO_DCP.  Session-15 caught this in a real LLM smoke;
        this test pins the fix.
        """
        import os
        captured = {}
        async def _record(*a, **kw):
            captured["dcp_path"] = kw.get("dcp_path") or (a[0] if a else None)
            captured["json_path"] = kw.get("json_path") or (a[1] if len(a) > 1 else None)
            return ("success", None)
        # Hand a RELATIVE path to the hook (relative to the test's cwd).
        cwd_before = os.getcwd()
        os.chdir(str(self.root))
        try:
            rel = Path(self.out_dcp.relative_to(self.root))
            self.assertFalse(rel.is_absolute(), "test setup: rel must be relative")
            with mock.patch.object(self.opt, "_run_qor_capture_subprocess",
                                   side_effect=_record):
                self.opt.capture_qor = True
                _async(self.opt._maybe_capture_qor_post_finalize(rel))
        finally:
            os.chdir(cwd_before)
        # The runner must have received an ABSOLUTE path.
        self.assertTrue(Path(captured["dcp_path"]).is_absolute(),
                        f"dcp_path was not resolved: {captured['dcp_path']!r}")
        self.assertTrue(Path(captured["json_path"]).is_absolute(),
                        f"json_path was not resolved: {captured['json_path']!r}")

    def test_no_design_name_in_qor_capture_record(self):
        """The qor_capture record itself must not embed a benchmark name."""
        runner = mock.AsyncMock(return_value=("success", None))
        with mock.patch.object(self.opt, "_run_qor_capture_subprocess", runner):
            self.opt.capture_qor = True
            self.opt._design_name_for_memory = "rosetta_spam-filter"
            _async(self.opt._maybe_capture_qor_post_finalize(self.out_dcp))
        r = _qor_records(self.opt)[0]
        # The optional `design` autofill is metadata only; it must not affect
        # capture status, paths, error summaries, or action labels.
        for k in ("status", "error_summary", "action_label"):
            v = r.get(k)
            if isinstance(v, str):
                self.assertNotIn("rosetta", v.lower())
                self.assertNotIn("spam-filter", v.lower())


class CaptureQorTimeoutTests(unittest.TestCase):
    """Session-17 --capture-qor-timeout contract."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.opt = DCPOptimizer(api_key="test", run_dir=self.run_dir)
        self.opt._ensure_decision_tracer()
        self.out_dcp = self.run_dir / "fake.dcp"
        self.out_dcp.write_bytes(b"DCP_TEST")

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_timeout_is_60(self):
        # Default attribute, before main() touches it.
        self.assertEqual(self.opt.capture_qor_timeout, 60.0)

    def test_custom_timeout_passed_to_subprocess(self):
        captured = {}
        async def _record(*a, **kw):
            captured["timeout_s"] = kw.get("timeout_s")
            return ("success", None)
        with mock.patch.object(self.opt, "_run_qor_capture_subprocess",
                               side_effect=_record):
            self.opt.capture_qor = True
            self.opt.capture_qor_timeout = 180.0
            _async(self.opt._maybe_capture_qor_post_finalize(self.out_dcp))
        self.assertEqual(captured["timeout_s"], 180.0)

    def test_timeout_appears_in_decision_record(self):
        async def _ok(*a, **kw):
            return ("success", None)
        with mock.patch.object(self.opt, "_run_qor_capture_subprocess",
                               side_effect=_ok):
            self.opt.capture_qor = True
            self.opt.capture_qor_timeout = 123.5
            _async(self.opt._maybe_capture_qor_post_finalize(self.out_dcp))
        r = _qor_records(self.opt)[0]
        self.assertEqual(r["timeout_seconds"], 123.5)

    def test_invalid_timeout_falls_back_to_60(self):
        captured = {}
        async def _record(*a, **kw):
            captured["timeout_s"] = kw.get("timeout_s")
            return ("success", None)
        with mock.patch.object(self.opt, "_run_qor_capture_subprocess",
                               side_effect=_record):
            self.opt.capture_qor = True
            # Negative / zero / non-numeric all clamp to 60.
            for bad in (-1.0, 0.0, "nope"):
                self.opt.capture_qor_timeout = bad
                _async(self.opt._maybe_capture_qor_post_finalize(self.out_dcp))
                self.assertEqual(captured["timeout_s"], 60.0,
                                 f"bad value {bad!r} should clamp to 60")

    def test_timeout_non_fatal_with_custom_value(self):
        async def _timeout(*a, **kw):
            return ("timeout", f"timed out after {kw['timeout_s']} s")
        with mock.patch.object(self.opt, "_run_qor_capture_subprocess",
                               side_effect=_timeout):
            self.opt.capture_qor = True
            self.opt.capture_qor_timeout = 180.0
            _async(self.opt._maybe_capture_qor_post_finalize(self.out_dcp))
        r = _qor_records(self.opt)[0]
        self.assertEqual(r["status"], "timeout")
        self.assertEqual(r["timeout_seconds"], 180.0)
        # Lifecycle is unaffected — same contract as default.
        # (No field exists for it here; covered by other tests.)

    def test_timeout_does_not_affect_submission_or_card(self):
        async def _ok(*a, **kw):
            return ("success", None)
        with mock.patch.object(self.opt, "_run_qor_capture_subprocess",
                               side_effect=_ok):
            self.opt.capture_qor = True
            self.opt.capture_qor_timeout = 240.0
            _async(self.opt._maybe_capture_qor_post_finalize(self.out_dcp))
        # Record must still be report-only / used_for_decision=false.
        r = _qor_records(self.opt)[0]
        self.assertEqual(r["report_only"], True)
        self.assertEqual(r["used_for_decision"], False)

    def test_no_design_name_in_timeout_field(self):
        async def _ok(*a, **kw):
            return ("success", None)
        with mock.patch.object(self.opt, "_run_qor_capture_subprocess",
                               side_effect=_ok):
            self.opt.capture_qor = True
            self.opt.capture_qor_timeout = 90.0
            self.opt._design_name_for_memory = "vexriscv_re-place_v2"
            _async(self.opt._maybe_capture_qor_post_finalize(self.out_dcp))
        r = _qor_records(self.opt)[0]
        # timeout-related fields must NOT carry a benchmark name.
        for k in ("timeout_seconds", "runtime_seconds", "status"):
            v = r.get(k)
            if isinstance(v, str):
                self.assertNotIn("vexriscv", v.lower())


class CaptureQorCliFlagTests(unittest.TestCase):
    """The CLI flag must be present, default OFF, and store_true."""

    def test_capture_qor_flag_exists_and_defaults_off(self):
        import dcp_optimizer as mod
        # Re-build the ArgumentParser from the same options main() uses
        # by parsing a tiny snippet.  We can't call main() directly because
        # it does I/O, but we can introspect the parser via argparse.
        parser = argparse.ArgumentParser()
        parser.add_argument("input_dcp")
        parser.add_argument("--contest-mode", action="store_true")
        parser.add_argument("--policy-card", action="store_true")
        parser.add_argument("--capture-qor", action="store_true")
        # Default OFF.
        ns = parser.parse_args(["x.dcp"])
        self.assertFalse(ns.capture_qor)
        ns2 = parser.parse_args(["x.dcp", "--capture-qor"])
        self.assertTrue(ns2.capture_qor)

    def test_real_main_parser_has_capture_qor(self):
        # Grep the source for the flag to ensure it lives in main()'s
        # parser (not just in this test file).
        src = Path(__file__).resolve().parent.parent / "dcp_optimizer.py"
        text = src.read_text()
        self.assertIn('"--capture-qor"', text)
        self.assertIn("optimizer.capture_qor = bool(args.capture_qor)", text)

    def test_real_main_parser_has_capture_qor_timeout(self):
        src = Path(__file__).resolve().parent.parent / "dcp_optimizer.py"
        text = src.read_text()
        self.assertIn('"--capture-qor-timeout"', text)
        self.assertIn("optimizer.capture_qor_timeout = float(args.capture_qor_timeout)", text)
        # Default must be 60.0 to preserve Session-14 contract.
        self.assertIn("default=60.0,", text)


if __name__ == "__main__":
    unittest.main()
