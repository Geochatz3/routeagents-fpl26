"""Tests for optimizer.tool_errors — structured ToolError taxonomy.

Phase 1: classification + log-only.  These tests pin the classifier's
behavior on every code in TOOL_ERROR_CODES.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizer.tool_errors import (
    TOOL_ERROR_CODES,
    ErrorPolicy,
    ToolError,
    classify_tool_error,
    get_policy,
    is_error_payload,
)
from optimizer.decision_tracer import DecisionTracer, load_decisions


class ToolErrorTaxonomyTests(unittest.TestCase):

    def test_all_18_codes_present(self):
        expected = {
            "VIVADO_TIMEOUT", "BUDGET_SKIP", "TIMEOUT_BUDGET",
            "INVALID_DCP", "WRONG_CLOCK", "ROUTE_FAILED", "PLACE_FAILED",
            "REGRESSION_DETECTED", "STALE_MIRROR",
            "TCL_SYNTAX_ERROR", "TCL_RUNTIME_ERROR", "MCP_UNAVAILABLE",
            "PARSER_FALSE_POSITIVE", "VALIDATOR_MISMATCH",
            "MISSING_ARTIFACT", "RQA_PARSE_FAILED", "RQS_GENERATOR_REFUSED",
            "UNKNOWN_TOOL_ERROR",
        }
        missing = expected - TOOL_ERROR_CODES
        self.assertEqual(set(), missing)

    def test_every_code_has_policy_entry(self):
        for code in TOOL_ERROR_CODES:
            pol = get_policy(code)
            self.assertIsInstance(pol, ErrorPolicy)
            self.assertIn(pol.severity, ("low", "medium", "high", "critical"))
            self.assertIn(pol.rollback, ("none", "mirror", "baseline"))

    def test_tool_error_to_dict_caps_payload(self):
        long_text = "x" * 5000
        err = ToolError(
            code="UNKNOWN_TOOL_ERROR", severity="high", retryable=False,
            rollback="none", reason="test", raw_payload=long_text,
        )
        d = err.to_dict()
        self.assertLessEqual(len(d["raw_payload"]), 500)


class ClassifierBehaviorTests(unittest.TestCase):
    """One test per recognized code — assert the classifier returns
    the right ToolError on a representative input."""

    def test_budget_skip_envelope(self):
        err = classify_tool_error(
            '{"error": "tool_skipped_budget", "reason": "estimated_300s_exceeds_remaining_60s"}'
        )
        self.assertIsNotNone(err)
        self.assertEqual(err.code, "BUDGET_SKIP")
        self.assertFalse(err.retryable)

    def test_timeout_budget(self):
        err = classify_tool_error("tool_timed_out_budget after 30s of 30s allowance")
        self.assertIsNotNone(err)
        self.assertEqual(err.code, "TIMEOUT_BUDGET")

    def test_vivado_timeout(self):
        err = classify_tool_error("Vivado tool call timed out (asyncio.TimeoutError)")
        self.assertIsNotNone(err)
        self.assertEqual(err.code, "VIVADO_TIMEOUT")

    def test_tcl_syntax_error(self):
        err = classify_tool_error(
            "invalid command name \"frobnicate\" while executing run_tcl"
        )
        self.assertIsNotNone(err)
        self.assertEqual(err.code, "TCL_SYNTAX_ERROR")

    def test_tcl_runtime_error(self):
        # Use a Common-subsystem error that does NOT collide with the
        # more-specific PLACE_FAILED / ROUTE_FAILED patterns.
        err = classify_tool_error(
            "ERROR: [Common 17-1234] unknown property on object"
        )
        self.assertIsNotNone(err)
        self.assertEqual(err.code, "TCL_RUNTIME_ERROR")

    def test_place_failed_takes_precedence_over_generic_tcl_runtime(self):
        # Specific failure shapes win over generic ERROR.
        err = classify_tool_error(
            "ERROR: [Place 30-58] cannot place cell foo at SLICE_X10Y20"
        )
        self.assertIsNotNone(err)
        self.assertEqual(err.code, "PLACE_FAILED")

    def test_route_failed(self):
        err = classify_tool_error("route_design failed: 5 unrouted nets remain")
        self.assertIsNotNone(err)
        self.assertEqual(err.code, "ROUTE_FAILED")

    def test_place_failed(self):
        err = classify_tool_error("placer failed during global placement phase")
        self.assertIsNotNone(err)
        self.assertEqual(err.code, "PLACE_FAILED")

    def test_invalid_dcp(self):
        err = classify_tool_error("DCP missing or zero-byte: /tmp/out.dcp")
        self.assertIsNotNone(err)
        self.assertEqual(err.code, "INVALID_DCP")

    def test_wrong_clock(self):
        err = classify_tool_error("CSWNS:CLOCK_NAME=NOT_FOUND")
        self.assertIsNotNone(err)
        self.assertEqual(err.code, "WRONG_CLOCK")

    def test_stale_mirror(self):
        err = classify_tool_error(
            "best_valid mirror is STALE (mirror_wns=-0.946, best=-0.926)"
        )
        self.assertIsNotNone(err)
        self.assertEqual(err.code, "STALE_MIRROR")

    def test_regression_detected(self):
        err = classify_tool_error("WNS regressed: -0.971 < initial -0.946")
        self.assertIsNotNone(err)
        self.assertEqual(err.code, "REGRESSION_DETECTED")

    def test_mcp_unavailable(self):
        err = classify_tool_error(
            "Failed to parse JSONRPC message from server (ValidationError)"
        )
        self.assertIsNotNone(err)
        self.assertEqual(err.code, "MCP_UNAVAILABLE")

    def test_validator_mismatch(self):
        err = classify_tool_error("FAIL_DELTA_5.62 claimed=10.0 validated=4.38")
        self.assertIsNotNone(err)
        self.assertEqual(err.code, "VALIDATOR_MISMATCH")
        self.assertEqual(err.severity, "critical")

    def test_missing_artifact(self):
        err = classify_tool_error(
            "vivado_write_checkpoint returned but file missing/empty"
        )
        self.assertIsNotNone(err)
        self.assertEqual(err.code, "MISSING_ARTIFACT")

    def test_missing_artifact_requires_error_context(self):
        """Regression: smoke 2026-05-20 saw the prior `file.*missing`
        substring match benign Vivado log chatter like "Optional
        constraint file is missing — using defaults".  The tightened
        pattern requires either an ERROR/WARNING tag or a specific
        envelope phrase.  Plain "file is missing" without context
        must NOT classify."""
        for benign in (
            "Optional constraint file is missing — using defaults",
            "Note: pin file is missing for some IO; continuing anyway",
            "Tip: extra constraint file is missing optional features",
        ):
            self.assertIsNone(
                classify_tool_error(benign),
                f"benign log line should not classify as MISSING_ARTIFACT: {benign!r}",
            )

    def test_missing_artifact_with_error_tag_still_classifies(self):
        """Verifies that missing non-DCP artifacts with error tags remain
        classified as errors.

        The `dcp missing` pattern is claimed earlier by `INVALID_DCP`, so these
        cases use other artifact paths. Missing-DCP behavior is covered by the
        invalid-DCP tests.
        """
        for hit in (
            "ERROR: [Common 17-69] expected file out.json does not exist",
            "WARNING: [Vivado 12-1234] file mirror.edf is missing on disk",
            "MISSING_ARTIFACT: dispatcher saw no output artifact",
            "vivado_write_checkpoint returned but file missing/empty",
            "output_dcp missing after finalize",  # routes to INVALID_DCP
        )[:-1]:  # exclude the last (intentionally INVALID_DCP)
            err = classify_tool_error(hit)
            self.assertIsNotNone(
                err, f"expected MISSING_ARTIFACT match for: {hit!r}",
            )
            self.assertEqual(err.code, "MISSING_ARTIFACT")

    def test_rqa_parse_failed(self):
        err = classify_tool_error("RQA parse failed: no score in output")
        self.assertIsNotNone(err)
        self.assertEqual(err.code, "RQA_PARSE_FAILED")

    def test_rqs_generator_refused(self):
        err = classify_tool_error(
            "ML Strategy Not Available — design not implemented with Default/Explore"
        )
        self.assertIsNotNone(err)
        self.assertEqual(err.code, "RQS_GENERATOR_REFUSED")

    def test_unknown_tool_error(self):
        err = classify_tool_error("ERROR: something weird happened that we don't recognize")
        self.assertIsNotNone(err)
        self.assertEqual(err.code, "UNKNOWN_TOOL_ERROR")
        self.assertTrue(get_policy(err.code).reviewer_escalation)

    def test_success_payloads_return_none(self):
        for ok in (None, "", "OK", "Wrote checkpoint: /tmp/out.dcp",
                   {"status": "OK"}, {}, '{"status":"ok"}'):
            self.assertIsNone(classify_tool_error(ok),
                              f"unexpected error classification for {ok!r}")

    def test_parser_false_positive_constructor(self):
        # PARSER_FALSE_POSITIVE has no auto-classifier pattern; callers
        # construct it explicitly via get_policy when a parser misfires.
        pol = get_policy("PARSER_FALSE_POSITIVE")
        self.assertEqual(pol.severity, "low")
        self.assertFalse(pol.retryable)

    def test_is_error_payload_convenience(self):
        self.assertTrue(is_error_payload("ERROR: weird"))
        self.assertFalse(is_error_payload("OK"))
        self.assertFalse(is_error_payload(None))


class ClassifierIntegrationWithTracerTests(unittest.TestCase):
    """Simulated tool error landing in a decision-tracer JSONL record."""

    def test_classified_error_writes_to_decisions_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracer = DecisionTracer(Path(tmp) / "run")
            payload = "ERROR: [Place 30-58] something broke"
            err = classify_tool_error(payload, context="call_tool")
            self.assertIsNotNone(err)
            ok = tracer.emit({
                "design": "vexriscv_re-place_v2",
                "iteration": 3,
                "decision_source": "executor",
                "tool_name": "vivado_place_design",
                "tool_error_code": err.code,
                "notes": err.reason,
            })
            self.assertTrue(ok)
            records = load_decisions(tracer.path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["tool_error_code"], "TCL_RUNTIME_ERROR")


if __name__ == "__main__":
    unittest.main()
