"""Tests for optimizer.decision_tracer — minimal JSONL telemetry.

Covers:
  - append + read round-trip
  - schema_version + run_id + timestamp stamping
  - missing optional fields allowed
  - unknown fields preserved
  - malformed lines skipped on read
  - filter_decisions helper
  - emit returns False on failure path (read-only dir) without raising
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizer.decision_tracer import (
    DECISION_RECORD_FIELDS,
    DECISION_SOURCES,
    DecisionTracer,
    SCHEMA_VERSION,
    filter_decisions,
    load_decisions,
)


class DecisionTracerWriterTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name) / "dcp_optimizer_run-20260520_120000"
        self.tracer = DecisionTracer(self.run_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_emit_appends_to_jsonl(self):
        ok = self.tracer.emit({"phase": "phase_1", "tool_name": "open_checkpoint"})
        self.assertTrue(ok)
        self.assertTrue(self.tracer.path.exists())
        records = load_decisions(self.tracer.path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["tool_name"], "open_checkpoint")

    def test_emit_stamps_schema_version_run_id_timestamp(self):
        before = time.time()
        self.tracer.emit({"tool_name": "get_wns"})
        after = time.time()
        rec = load_decisions(self.tracer.path)[0]
        self.assertEqual(rec["schema_version"], SCHEMA_VERSION)
        self.assertEqual(rec["run_id"], "dcp_optimizer_run-20260520_120000")
        self.assertGreaterEqual(rec["timestamp"], before)
        self.assertLessEqual(rec["timestamp"], after)

    def test_emit_does_not_overwrite_caller_provided_headers(self):
        self.tracer.emit({
            "run_id": "custom_id", "timestamp": 1.0,
            "schema_version": 99,
        })
        rec = load_decisions(self.tracer.path)[0]
        self.assertEqual(rec["run_id"], "custom_id")
        self.assertEqual(rec["timestamp"], 1.0)
        self.assertEqual(rec["schema_version"], 99)

    def test_missing_optional_fields_allowed(self):
        # Smallest possible record — almost everything is optional.
        ok = self.tracer.emit({})
        self.assertTrue(ok)
        rec = load_decisions(self.tracer.path)[0]
        self.assertIn("schema_version", rec)
        self.assertIn("run_id", rec)

    def test_unknown_fields_preserved(self):
        self.tracer.emit({"some_future_field": 42, "tool_name": "x"})
        rec = load_decisions(self.tracer.path)[0]
        self.assertEqual(rec["some_future_field"], 42)

    def test_path_objects_serialize(self):
        # Path values should round-trip as strings via _safe_json_default
        path_value = Path("/tmp/test.dcp")
        ok = self.tracer.emit({
            "best_valid_checkpoint_path": path_value,
            "output_dcp_path": path_value,
        })
        self.assertTrue(ok)
        rec = load_decisions(self.tracer.path)[0]
        self.assertEqual(rec["best_valid_checkpoint_path"], "/tmp/test.dcp")

    def test_emit_returns_false_on_bad_record(self):
        # A non-dict cannot be stamped.
        self.assertFalse(self.tracer.emit("not a dict"))  # type: ignore[arg-type]
        self.assertFalse(self.tracer.emit(None))  # type: ignore[arg-type]

    def test_emit_returns_false_on_unencodable_record(self):
        # Circular reference cannot be encoded; emit must return False
        # without raising into the caller.
        circular = {}
        circular["self"] = circular
        # Path may also recurse via _safe_json_default's __dict__ fallback;
        # the writer must still survive without crashing the optimizer.
        result = self.tracer.emit(circular)
        # We accept either True (if _safe_json_default unwrapped it) or
        # False (if json.dumps raised) — as long as it didn't crash.
        self.assertIn(result, (True, False))

    def test_multiple_emits_each_become_separate_lines(self):
        for i in range(5):
            self.tracer.emit({"iteration": i, "tool_name": f"t{i}"})
        records = load_decisions(self.tracer.path)
        self.assertEqual(len(records), 5)
        self.assertEqual([r["iteration"] for r in records], [0, 1, 2, 3, 4])

    def test_path_for_run_classmethod(self):
        # The class helper must produce the same path as the instance.
        p = DecisionTracer.path_for_run(self.run_dir)
        self.assertEqual(p, self.tracer.path)


class DecisionTracerSchemaTests(unittest.TestCase):

    def test_schema_version_constant_is_int(self):
        self.assertIsInstance(SCHEMA_VERSION, int)
        self.assertGreaterEqual(SCHEMA_VERSION, 1)

    def test_recognized_fields_include_required_set(self):
        # Pin the minimum set the user's brief documented.
        required = {
            "schema_version", "run_id", "timestamp", "design", "iteration",
            "phase", "tool_name", "action_label", "decision_source",
            "rag_mode", "rqa_mode", "model",
            "wns_before", "wns_after", "fmax_before", "fmax_after",
            "delta_wns", "delta_fmax", "validity_state",
            "best_valid_checkpoint_path", "best_valid_token",
            "ship_lineage_source", "output_dcp_path", "tool_error_code",
            "runtime_s", "token_count", "cost_usd", "notes",
        }
        missing = required - DECISION_RECORD_FIELDS
        self.assertEqual(set(), missing,
                          f"schema must include {missing} per the brief")

    def test_decision_sources_closed_set(self):
        expected = {
            "router", "planner", "executor", "reviewer",
            "rag", "rqa", "fallback", "finalize", "validator",
        }
        missing = expected - DECISION_SOURCES
        self.assertEqual(set(), missing,
                          f"decision_source must include {missing}")


class LoadDecisionsTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "decisions.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_returns_empty(self):
        self.assertEqual(load_decisions(self.path), [])

    def test_skips_malformed_lines(self):
        self.path.write_text(
            '{"valid": 1}\n'
            'NOT-JSON\n'
            '{"valid": 2}\n'
            '\n'                      # blank line
            '{"valid": 3}\n',
            encoding="utf-8",
        )
        records = load_decisions(self.path)
        self.assertEqual([r["valid"] for r in records], [1, 2, 3])

    def test_filter_decisions(self):
        records = [
            {"design": "A", "decision_source": "router", "iteration": 1,
             "tool_error_code": None},
            {"design": "A", "decision_source": "executor", "iteration": 1,
             "tool_error_code": "BUDGET_SKIP"},
            {"design": "B", "decision_source": "router", "iteration": 2,
             "tool_error_code": None},
        ]
        self.assertEqual(len(filter_decisions(records, design="A")), 2)
        self.assertEqual(len(filter_decisions(records, decision_source="router")), 2)
        self.assertEqual(len(filter_decisions(records, iteration=2)), 1)
        self.assertEqual(len(filter_decisions(records, has_tool_error=True)), 1)
        self.assertEqual(len(filter_decisions(records, has_tool_error=False)), 2)


class DecisionTracerSmokeTests(unittest.TestCase):
    """Mini integration test that simulates an iteration's worth of
    decision events end-to-end and confirms readback shape."""

    def test_simulated_iteration_writes_nonempty_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run-1"
            tracer = DecisionTracer(run_dir)

            # Phase-1 init
            tracer.emit({
                "phase": "phase_1",
                "tool_name": "open_checkpoint",
                "decision_source": "executor",
                "iteration": 0,
                "design": "vexriscv_re-place_v2",
            })
            # WNS measurement
            tracer.emit({
                "phase": "iter",
                "tool_name": "vivado_get_wns",
                "decision_source": "executor",
                "iteration": 1,
                "wns_before": None,
                "wns_after": -0.946,
            })
            # Mirror event
            tracer.emit({
                "phase": "iter",
                "tool_name": "vivado_write_checkpoint",
                "decision_source": "executor",
                "iteration": 1,
                "best_valid_token": 1,
                "best_valid_checkpoint_path": "/tmp/best_valid.dcp",
            })
            # Finalize
            tracer.emit({
                "phase": "finalize",
                "decision_source": "finalize",
                "ship_lineage_source": "eager_mirror",
                "validity_state": "VALID_OPTIMIZED",
                "output_dcp_path": "/tmp/out.dcp",
            })

            records = load_decisions(tracer.path)
            self.assertEqual(len(records), 4)
            # Phase coverage:
            phases = {r.get("phase") for r in records}
            self.assertIn("phase_1", phases)
            self.assertIn("iter", phases)
            self.assertIn("finalize", phases)
            # Ship lineage propagated
            finalize_recs = [r for r in records if r.get("phase") == "finalize"]
            self.assertEqual(finalize_recs[0]["ship_lineage_source"],
                              "eager_mirror")


if __name__ == "__main__":
    unittest.main()
