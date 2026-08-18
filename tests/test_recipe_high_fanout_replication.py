"""Unit tests for recipe_high_fanout_timing_replication."""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import DCPOptimizer


def _async(coro):
    return asyncio.run(coro)


def _make_opt(tmp_path: Path) -> DCPOptimizer:
    opt = DCPOptimizer(api_key="test", run_dir=tmp_path)
    opt.clock_period = 1.570
    opt.best_wns = -1.000
    opt.initial_wns = -1.000
    # Pre-stage best_valid mirror so revert has somewhere to go.
    bv = tmp_path / "best_valid.dcp"
    bv.write_bytes(b"BEST_VALID_DCP_BYTES")
    opt._best_valid_dcp = bv
    return opt


HF_NETS_RESPONSE = """net_a/sub_net
net_b/q
clk_en[3]
net_c/something
"""


class HighFanoutReplicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.opt = _make_opt(self.tmp_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dispatch_routes_to_recipe(self):
        with mock.patch.object(
            self.opt, "_recipe_high_fanout_timing_replication",
            new=mock.AsyncMock(return_value='{"status": "stub"}')
        ) as stub:
            result = _async(self.opt.call_tool(
                "recipe_high_fanout_timing_replication", {}
            ))
            self.assertEqual(result, '{"status": "stub"}')
            stub.assert_awaited_once()

    def test_no_initial_wns_returns_error(self):
        self.opt.best_wns = float("-inf")
        out = _async(self.opt._recipe_high_fanout_timing_replication({}))
        r = json.loads(out)
        self.assertEqual(r["status"], "error")
        self.assertIn("best_wns not yet established", r["error"])

    def test_no_candidates_short_circuits(self):
        async def fake(name, args):
            if name == "vivado_get_critical_high_fanout_nets":
                return ""  # no nets
            raise AssertionError(f"unexpected tool call {name}")
        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake)):
            out = _async(self.opt._recipe_high_fanout_timing_replication({}))
        r = json.loads(out)
        self.assertEqual(r["status"], "no_candidates")
        self.assertEqual(r["candidates_total"], 0)

    def test_commit_on_improvement(self):
        async def fake(name, args):
            if name == "vivado_get_critical_high_fanout_nets":
                return HF_NETS_RESPONSE
            if name == "vivado_phys_opt_design":
                self.opt.best_wns = -0.900
                return "phys_opt_design ok"
            if name == "vivado_report_route_status":
                return "# of nets with routing errors that are routable: 0"
            if name == "vivado_get_wns":
                return "-0.900"
            return "ok"

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake)):
            out = _async(self.opt._recipe_high_fanout_timing_replication({}))
        r = json.loads(out)
        self.assertEqual(r["status"], "committed")
        self.assertAlmostEqual(r["delta_wns_ns"], 0.100, places=3)
        self.assertGreater(r["delta_fmax_mhz"], 0.0)
        self.assertGreater(r["candidates_total"], 0)
        # Top-N nets were forwarded to phys_opt.
        self.assertTrue(len(r["nets_replicated"]) >= 1)

    def test_regression_triggers_revert(self):
        async def fake(name, args):
            if name == "vivado_get_critical_high_fanout_nets":
                return HF_NETS_RESPONSE
            if name == "vivado_phys_opt_design":
                self.opt.best_wns = -1.200  # regression
                return "ok"
            if name == "vivado_report_route_status":
                return "# of nets with routing errors that are routable: 0"
            if name == "vivado_get_wns":
                return "-1.200"
            if name == "vivado_open_checkpoint":
                return "ok"
            return "ok"

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake)):
            out = _async(self.opt._recipe_high_fanout_timing_replication({}))
        r = json.loads(out)
        self.assertEqual(r["status"], "reverted_regression")
        self.assertEqual(r["revert_status"], "reopened_best_valid")

    def test_routing_errors_trigger_revert(self):
        async def fake(name, args):
            if name == "vivado_get_critical_high_fanout_nets":
                return HF_NETS_RESPONSE
            if name == "vivado_phys_opt_design":
                return "ok"
            if name == "vivado_report_route_status":
                # 5 unroutable nets after replication.
                return "# of nets with routing errors that are routable: 5"
            if name == "vivado_open_checkpoint":
                return "ok"
            return "ok"

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake)):
            out = _async(self.opt._recipe_high_fanout_timing_replication({}))
        r = json.loads(out)
        self.assertEqual(r["status"], "route_errors_revert")
        self.assertEqual(r["revert_status"], "reopened_best_valid")
        self.assertEqual(r["route_errors"], 5)

    def test_max_nets_to_replicate_caps_list(self):
        # 4 candidates in the fixture; we cap to 2.
        async def fake(name, args):
            if name == "vivado_get_critical_high_fanout_nets":
                return HF_NETS_RESPONSE
            if name == "vivado_phys_opt_design":
                self.opt.best_wns = -0.950
                return "ok"
            if name == "vivado_report_route_status":
                return "# of nets with routing errors that are routable: 0"
            if name == "vivado_get_wns":
                return "-0.950"
            return "ok"

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake)):
            out = _async(self.opt._recipe_high_fanout_timing_replication(
                {"max_nets_to_replicate": 2}
            ))
        r = json.loads(out)
        self.assertLessEqual(len(r["nets_replicated"]), 2)


if __name__ == "__main__":
    unittest.main()
