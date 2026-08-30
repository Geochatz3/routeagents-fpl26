"""Integration tests: feature-based recipe router output reaches the
iter-1 user message via DCPOptimizer._build_recipe_router_block.

The router was added on top of the existing pathology classifier as a
second-stage routing layer. These tests pin the wiring: given the same
Phase-1 state the optimizer collects (wns, clock_period, failing-
endpoint count, spread, wall budget), _build_recipe_router_block emits
a structured FEATURE-BASED RECIPE ROUTING block with rule_id + actions
+ blocks.
"""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import DCPOptimizer


def _make_opt(tmp_path: Path) -> DCPOptimizer:
    opt = DCPOptimizer(api_key="test", run_dir=tmp_path)
    return opt


class RouterIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = _make_opt(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_block_empty_when_no_phase1_data(self):
        # No WNS, no clock period → no rule can fire and no blocks apply.
        self.opt.initial_wns = None
        self.opt.clock_period = None
        self.opt.initial_failing_endpoints = None
        self.opt.critical_path_spread_info = None
        block = self.opt._build_recipe_router_block()
        self.assertEqual(block, [])

    def test_boom_soc_phase1_state_triggers_r1_block(self):
        # Extreme negative slack and a very large failing set select this path.
        self.opt.initial_wns = -19.16
        self.opt.clock_period = 1.666
        self.opt.initial_failing_endpoints = 217_988
        self.opt.critical_path_spread_info = {"avg_distance": 302.0}
        # Generous wall budget so R3 wouldn't fire (not applicable anyway).
        self.opt._budget_deadline = time.time() + 60 * 60
        block = self.opt._build_recipe_router_block()
        self.assertGreater(len(block), 0)
        text = "\n".join(block)
        self.assertIn("rule R1", text)
        self.assertIn("vivado_phys_opt_design", text)
        self.assertIn("AlternateFlowWithRetiming", text)
        # Plan stashed on optimizer for downstream consumers.
        self.assertIsNotNone(self.opt.recipe_router_plan)
        self.assertEqual(self.opt.recipe_router_plan.rule_id, "R1")

    def test_ispd16_phase1_state_triggers_r2_with_block(self):
        # High path spread, large negative slack, and a medium failing set
        # select this path.
        self.opt.initial_wns = -7.75
        self.opt.clock_period = 1.587
        self.opt.initial_failing_endpoints = 30_000
        self.opt.critical_path_spread_info = {"avg_distance": 100.0}
        self.opt._budget_deadline = time.time() + 60 * 60
        block = self.opt._build_recipe_router_block()
        text = "\n".join(block)
        self.assertIn("rule R2", text)
        self.assertIn("recipe_post_route_phys_opt_sweep", text)
        # cell_replacement must be blocked.
        self.assertIn("BLOCKED as iter-1", text)
        self.assertIn("recipe_cell_replacement", text)

    def test_finn_phase1_state_triggers_r3_when_budget_allows(self):
        # This moderate-WNS, medium-failing-set case requires at least 25 min for R3.
        self.opt.initial_wns = -1.91
        self.opt.clock_period = 1.628
        self.opt.initial_failing_endpoints = 30_000
        self.opt.critical_path_spread_info = {"avg_distance": 60.0}
        self.opt._budget_deadline = time.time() + 30 * 60
        block = self.opt._build_recipe_router_block()
        text = "\n".join(block)
        self.assertIn("rule R3", text)
        self.assertIn("Explore", text)

    def test_finn_phase1_state_does_not_trigger_r3_when_budget_tight(self):
        # With the same timing profile and only 20 min remaining, R3 stays disabled.
        self.opt.initial_wns = -1.91
        self.opt.clock_period = 1.628
        self.opt.initial_failing_endpoints = 30_000
        self.opt.critical_path_spread_info = {"avg_distance": 60.0}
        self.opt._budget_deadline = time.time() + 20 * 60
        block = self.opt._build_recipe_router_block()
        text = "\n".join(block)
        self.assertNotIn("rule R3", text)

    def test_corescore_phase1_state_triggers_r4(self):
        self.opt.initial_wns = -1.24
        self.opt.clock_period = 1.644
        self.opt.initial_failing_endpoints = 3_000
        # An average critical-path distance of 232 indicates placement-limited timing
        # and selects the R4 Explore path.
        self.opt.critical_path_spread_info = {"avg_distance": 232.0}
        self.opt._budget_deadline = time.time() + 30 * 60
        block = self.opt._build_recipe_router_block()
        text = "\n".join(block)
        self.assertIn("rule R4", text)
        self.assertIn("Explore", text)


if __name__ == "__main__":
    unittest.main()
