"""Verify persistence of completed-run strategy memory.

A mocked optimizer state simulates completion and confirms that the resulting
record is written to the configured memory file without starting external tools
or services.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import DCPOptimizer
from optimizer.strategy_memory import RunRecord


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory_path = Path(self.tmp.name) / "test_memory.jsonl"
        self._env = mock.patch.dict(
            os.environ,
            {"STRATEGY_MEMORY_PATH": str(self.memory_path)},
            clear=False,
        )
        self._env.start()

    def tearDown(self):
        self._env.stop()
        self.tmp.cleanup()

    def _make_fake_optimizer(self, **overrides):
        """Build a DCPOptimizer-like object with just the fields
        _persist_to_strategy_memory needs."""
        import time
        class Fake:
            pass
        f = Fake()
        f.iteration = 5
        f.initial_wns = -0.978
        f.best_wns = -0.450
        f.clock_period = 2.5
        f.total_cost = 0.135
        f.start_time = time.time() - 600
        f.end_time = time.time()
        f.tool_call_details = [
            {"tool_name": "vivado_open_checkpoint", "wns": None},
            {"tool_name": "vivado_report_timing_summary", "wns": -0.978},
            {"tool_name": "vivado_create_and_apply_pblock", "wns": None},
            {"tool_name": "vivado_place_design", "wns": None},
            {"tool_name": "vivado_route_design", "wns": None},
            {"tool_name": "vivado_report_timing_summary", "wns": -0.700},
            {"tool_name": "vivado_phys_opt_design", "wns": None},
            {"tool_name": "vivado_report_timing_summary", "wns": -0.450},
        ]
        f._design_name_for_memory = "test_design"
        f._initial_fmax_for_memory = 1000.0 / (2.5 - (-0.978))  # ≈ 287.36
        f.critical_path_spread_info = {"max_distance": 80, "avg_distance": 65.0,
                                       "paths_analyzed": 12}
        f.mode = "v0_3"
        f.force_continue_count = 2
        f.lut_count = 12000
        f.calculate_fmax = DCPOptimizer.calculate_fmax.__get__(f, type(f))
        for k, v in overrides.items():
            setattr(f, k, v)
        return f

    def test_persists_winning_record(self):
        fake = self._make_fake_optimizer()
        DCPOptimizer._persist_to_strategy_memory(fake)
        self.assertTrue(self.memory_path.exists())
        records = [json.loads(l) for l in self.memory_path.read_text().splitlines()]
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual(r["design"], "test_design")
        self.assertEqual(r["candidate"], "v0_3")
        self.assertEqual(r["iterations"], 5)
        self.assertEqual(r["force_continues"], 2)
        self.assertEqual(r["lut_count"], 12000)
        self.assertEqual(r["critical_path_spread"], 65.0)
        # winning_tools should include place + route + phys_opt (the
        # transformative tools attributed to WNS improvements)
        self.assertIn("vivado_route_design", r["winning_tools"])
        self.assertIn("vivado_phys_opt_design", r["winning_tools"])
        # completed must be True since delta > 0
        self.assertTrue(r["completed"])

    def test_no_design_skips_persistence(self):
        fake = self._make_fake_optimizer(_design_name_for_memory=None)
        DCPOptimizer._persist_to_strategy_memory(fake)
        # File should not exist (nothing persisted)
        self.assertFalse(self.memory_path.exists())

    def test_regression_persisted_but_marked_uncompleted(self):
        # best_wns < initial_wns: this is a regression
        fake = self._make_fake_optimizer(best_wns=-1.2)
        DCPOptimizer._persist_to_strategy_memory(fake)
        records = [json.loads(l) for l in self.memory_path.read_text().splitlines()]
        r = records[0]
        # completed must be False for regressions
        self.assertFalse(r["completed"])

    def test_handles_missing_attributes_gracefully(self):
        fake = self._make_fake_optimizer()
        del fake.lut_count
        del fake.force_continue_count
        # Should not raise — these are optional via getattr
        DCPOptimizer._persist_to_strategy_memory(fake)
        records = [json.loads(l) for l in self.memory_path.read_text().splitlines()]
        # lut_count + force_continues missing → not in serialized dict
        r = records[0]
        self.assertNotIn("lut_count", r)
        self.assertNotIn("force_continues", r)


if __name__ == "__main__":
    unittest.main()
