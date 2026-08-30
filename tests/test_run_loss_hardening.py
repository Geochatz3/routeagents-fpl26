"""Test that completed valid runs remain rankable and recoverable when reporting
or tool queries fail.

Frequency metadata must have a fallback beyond `token_usage.json` so an
interrupted summary cannot discard an artifact. Primitive-cell parsing rejects
tool-error text before extracting digits, and tool-result validation recognizes
JSON envelopes, Tcl errors, and plain-text server timeouts.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import DCPOptimizer, _looks_like_tool_error
from optimizer.ils_polish import _tool_ok
from tests.source_corpus import dcp_source_lines, dcp_source_text


# The MCP dispatcher's three self-reported failure shapes.
SERVER_ERRORS = [
    "Error: Command timed out. Vivado may be stuck. Use restart_vivado to recover.",
    "Error: Vivado process terminated unexpectedly. Use restart_vivado to restart.",
    "Error: [Errno 32] Broken pipe",
]

# call_tool's budget envelopes — note each carries a plausible-looking integer.
BUDGET_ENVELOPES = [
    json.dumps({"error": "tool_skipped_budget",
                "reason": "only_23s_remain_below_min_useful_30s"}),
    json.dumps({"error": "tool_timed_out_budget",
                "elapsed_seconds": 45.0, "allowance_seconds": 120.0}),
]


class ServerErrorDetectionTests(unittest.TestCase):
    """C4 — a wedged Vivado must not read as success."""

    def test_server_errors_are_detected(self):
        for txt in SERVER_ERRORS:
            with self.subTest(txt=txt[:40]):
                self.assertTrue(
                    _looks_like_tool_error(txt),
                    "MCP server error string read as a successful result")

    def test_ils_tool_ok_rejects_server_errors(self):
        for txt in SERVER_ERRORS:
            with self.subTest(txt=txt[:40]):
                self.assertFalse(_tool_ok(txt))

    def test_known_shapes_still_detected(self):
        self.assertTrue(_looks_like_tool_error('{"error": "boom"}'))
        self.assertTrue(_looks_like_tool_error("TCL ERROR: bad command"))
        for env in BUDGET_ENVELOPES:
            self.assertTrue(_looks_like_tool_error(env))

    def test_clean_output_is_not_flagged(self):
        """The prefix match must not swallow legitimate reports."""
        clean = [
            "532160",
            "# of nets with routing errors : 0\n",
            "Design is fully routed",
            # 'Error:' appearing MID-text is not a failed call
            "Route Status: 0 nets\nError: 0 unrouted\n",
            "",
        ]
        for txt in clean:
            with self.subTest(txt=txt[:40]):
                self.assertFalse(_looks_like_tool_error(txt),
                                 "clean output misread as an error")
        self.assertTrue(_tool_ok("532160"))


class CellCountEnvelopeTests(unittest.TestCase):
    """C3 — an error envelope must never parse into a cell count."""

    def test_every_cellcount_site_guards_before_parsing(self):
        """All four copies of the query must guard; :7307 is the template."""
        src = dcp_source_text(errors="ignore")
        blocks = src.split("IS_PRIMITIVE}]")
        # Each post-preamble block begins after a query and must contain a guard
        # before any regular expression attempts to match digits.
        checked = 0
        for blk in blocks[1:]:
            head = blk[:1400]
            if 'search(r"(\\d+)"' not in head:
                continue          # not a cell-count parse site
            checked += 1
            guard = head.find("_looks_like_tool_error")
            parse = head.find('search(r"(\\d+)"')
            self.assertNotEqual(guard, -1,
                                "cell-count site parses digits with no "
                                "_looks_like_tool_error guard")
            self.assertLess(guard, parse,
                            "guard must come BEFORE the digit parse")
        self.assertGreaterEqual(checked, 3,
                                "expected at least 3 cell-count parse sites")

    def test_envelope_first_integer_would_be_a_bogus_count(self):
        """Why the guard matters: the envelopes really do yield small ints."""
        import re
        for env in BUDGET_ENVELOPES:
            m = re.search(r"(\d+)", env)
            self.assertIsNotNone(m)
            bogus = int(m.group(1))
            input_cells = 532_160          # ispd16
            self.assertFalse(0.5 * input_cells <= bogus <= 3.0 * input_cells,
                             "envelope integer must be out-of-band (that is "
                             "the silent-rejection mechanism)")


class TokenUsageLastResortTests(unittest.TestCase):
    """C1 — the wrapper's only usability signal must survive a failed print."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_report_is_writable_without_the_summary_printer(self):
        """The last-resort path calls this directly; it must stand alone."""
        opt = DCPOptimizer(api_key="test", run_dir=self.tmp_path)
        opt.initial_wns = -2.0
        opt.best_wns = -1.0
        opt.clock_period = 5.0
        opt.start_time = 1000.0
        opt.end_time = 1100.0
        # A constraint-guard bookkeeping entry, i.e. the shape.
        opt.tool_call_details = [
            {"tool_name": "vivado_place_design", "elapsed_time": 3.0},
            {"tool_name": "constraint_guard", "status": "CONSTRAINTS_CHANGED",
             "detail": "clock count 1 -> 0"},
        ]
        out = self.tmp_path / "token_usage.json"
        opt.save_token_usage_report(out)
        self.assertTrue(out.exists())
        summary = json.loads(out.read_text())["summary"]
        # The wrapper reads exactly this key to decide the attempt is usable.
        self.assertIsNotNone(summary.get("best_fmax_mhz"),
                             "best_fmax_mhz is the wrapper's ONLY fmax source")

    def _mk_opt(self):
        opt = DCPOptimizer(api_key="test", run_dir=self.tmp_path)
        opt.initial_wns = -2.0
        opt.best_wns = -1.0
        opt.clock_period = 5.0
        opt.start_time = 1000.0
        opt.end_time = 1100.0
        opt.tool_call_details = []
        return opt

    def test_report_uses_shipped_wns_when_worse_than_tracked(self):
        """(panel B-3): a finalize branch that ships an older mirror
        records _shipped_wns_ns; the report — select_best's ONLY fmax
        source — must describe the artifact, not the in-memory claim."""
        opt = self._mk_opt()
        opt._shipped_wns_ns = -1.5   # shipped mirror holds LESS than tracked
        out = self.tmp_path / "token_usage.json"
        opt.save_token_usage_report(out)
        got = json.loads(out.read_text())["summary"]["best_fmax_mhz"]
        want = opt.calculate_fmax(-1.5, 5.0)
        self.assertAlmostEqual(got, want, places=3,
                               msg="report must carry the SHIPPED WNS")

    def test_report_never_worsened_by_a_better_shipped_wns(self):
        # min() semantics: shipped better-or-equal than tracked -> tracked.
        opt = self._mk_opt()
        opt._shipped_wns_ns = -0.5
        out = self.tmp_path / "token_usage.json"
        opt.save_token_usage_report(out)
        got = json.loads(out.read_text())["summary"]["best_fmax_mhz"]
        self.assertAlmostEqual(got, opt.calculate_fmax(-1.0, 5.0), places=3)

    def test_report_write_is_atomic_no_partial_file_on_crash(self):
        """: the report is written tmp+os.replace; a crash mid-write
        must leave either the OLD complete file or none — never truncated
        JSON (which reads as fmax=None and drops a VALID artifact)."""
        opt = self._mk_opt()
        out = self.tmp_path / "token_usage.json"
        out.write_text('{"summary": {"best_fmax_mhz": 123.0}}')
        import unittest.mock as _mock
        with _mock.patch("json.dump", side_effect=OSError("disk full")):
            try:
                opt.save_token_usage_report(out)
            except Exception:
                pass
        data = json.loads(out.read_text())  # old file intact, still valid
        self.assertEqual(data["summary"]["best_fmax_mhz"], 123.0)

    def test_emergency_path_writes_the_report_when_missing(self):
        """The fix must live on the every-exit path, not the print path."""
        src = dcp_source_text(errors="ignore")
        i = src.find("def _emergency_baseline_copy")
        self.assertNotEqual(i, -1)
        # Bound the body at the next sibling def rather than a fixed slice —
        # this function is ~10 KB and a guessed window silently misses the end.
        j = src.find("\n    def ", i + 1)
        self.assertNotEqual(j, -1, "could not find the end of the function")
        body = src[i:j]
        self.assertIn("save_token_usage_report", body,
                      "_emergency_baseline_copy must write token_usage.json "
                      "when nothing else did — otherwise a failed summary "
                      "silently costs the whole attempt")
        self.assertIn("token_usage.json", body)

    def test_emergency_ladder_falls_through_not_out(self):
        """Verify that finalization falls through each recovery path instead of
        escaping to the outer hard-failure handler.

        The finalized latch is set before copying, so each path must contain
        its own copy failures. If the preferred mirror copy fails, including
        from insufficient disk space, later paths must still produce a valid
        checkpoint.
        """
        src = dcp_source_text(errors="ignore")
        i = src.find("def _emergency_baseline_copy")
        self.assertNotEqual(i, -1)
        j = src.find("\n    def ", i + 1)
        self.assertNotEqual(j, -1, "could not find the end of the function")
        body = src[i:j]
        self.assertIn("EMERGENCY_BEST_VALID_COPY FAILED", body,
                      "Path 1 needs its own except that logs and falls "
                      "through to the rest of the ladder")
        self.assertIn("falling through to the", body)
        self.assertIn("baseline guarantee", body)
        self.assertIn("elif chosen is None:", body,
                      "Path 3 must be guarded on `chosen is None` — a bare "
                      "`else` after the restructure would overwrite a "
                      "SUCCESSFUL best_valid copy with the baseline")
        self.assertIn("if chosen is None and out.exists()", body,
                      "Path 2 must also key on `chosen is None`")

    def test_wrapper_still_has_no_other_fmax_source(self):
        """Pins WHY the fix has to be agent-side.

        If a fallback is ever added to _read_run_metrics, revisit this.
        """
        wrapper = (Path(__file__).resolve().parent.parent
                   / "scripts" / "multi_restart_optimize.py").read_text(
                       errors="ignore")
        self.assertIn('out["fmax"] = s.get("best_fmax_mhz")', wrapper)
        self.assertIn('a.get("fmax") is not None', wrapper)


if __name__ == "__main__":
    unittest.main()
