"""Test the pure route-time predictor with direct fixtures and no optimizer or
mocks.

Fixtures span small designs, route and physical-optimization timings, and a
very large design whose full reroute exceeds the remaining budget. The gate
must reject that infeasible reroute at cold start and mid-run without
over-refusing small designs.
"""
from __future__ import annotations

import unittest

from optimizer.route_gate import (
    BANKING_RESERVE_S,
    CELLS_SCALED_RATE_S,
    HEAVY_OP_MULTIPLIER_K,
    SAFETY_MARGIN_S,
    RerouteAssessment,
    assess_destructive_reroute,
    predict_reroute_seconds,
)


# Fixture helpers

BOOM_CELLS = 379_380
BOOM_PHYS_OPT_S = 1387.79
BOOM_REMAINING_S = 1650.0

def _boom_history():
    """tool_call_details as seen at boom's -unroute moment: one completed
    heavy op (phys_opt AlternateFlowWithRetiming, 1387.79s)."""
    return [
        {"tool_name": "vivado_phys_opt_design", "iteration": 1,
         "elapsed_time": BOOM_PHYS_OPT_S, "wns": None, "cmd_head": ""},
    ]


# predict_reroute_seconds

class TestPredictRerouteSeconds(unittest.TestCase):

    def test_k_branch_dominates_on_boom_history(self):
        """2.0 × 1387.79 = 2775.58 dominates the cells floor (2276.28)."""
        pred = predict_reroute_seconds(_boom_history(), BOOM_CELLS)
        self.assertGreaterEqual(pred, 2775.5)

    def test_cold_start_uses_cells_floor(self):
        """Empty history → the 0.006 s/cell floor, never 0."""
        pred = predict_reroute_seconds([], BOOM_CELLS)
        self.assertGreater(pred, 0.0)
        self.assertAlmostEqual(pred, BOOM_CELLS * CELLS_SCALED_RATE_S,
                               places=3)
        self.assertGreaterEqual(pred, 2276.0)

    def test_small_design_floor_not_inflated(self):
        """mini-isp class (8,414 cells) floor ≈ 50.5s — small stays small."""
        pred = predict_reroute_seconds([], 8_414)
        self.assertAlmostEqual(pred, 50.484, places=2)

    def test_prediction_never_below_cells_floor(self):
        """Even with a tiny completed heavy op, the floor holds."""
        hist = [{"tool_name": "vivado_route_design", "elapsed_time": 10.0}]
        pred = predict_reroute_seconds(hist, BOOM_CELLS)
        self.assertGreaterEqual(pred, BOOM_CELLS * CELLS_SCALED_RATE_S)


# assess_destructive_reroute — the boom calibration cases

class TestBoomScenario(unittest.TestCase):

    def test_boom_scenario_calibration_refuses(self):
        """THE fix: mid-run boom (completed 1387.79s phys_opt, 1650s
        remaining) must be refused — predicted 2775.6s cannot fit."""
        r = assess_destructive_reroute(
            remaining_wall_s=BOOM_REMAINING_S,
            tool_call_details=_boom_history(),
            input_cell_count=BOOM_CELLS,
        )
        self.assertIsInstance(r, RerouteAssessment)
        self.assertFalse(r.feasible)
        self.assertGreaterEqual(r.predicted_reroute_s, 2775.0)
        # reason must name the predicted-vs-remaining shortfall
        self.assertIn("REFUSE", r.reason)
        self.assertIn("exceeds", r.reason)

    def test_cold_start_large_design_floor_refuses(self):
        """COLD-START gamble the K-branch cannot see: 379,380 cells, empty
        history, 2200s remaining. Floor 2276.3 + reserve 30 > 2200 − 120."""
        r = assess_destructive_reroute(
            remaining_wall_s=2200.0,
            tool_call_details=[],
            input_cell_count=BOOM_CELLS,
        )
        self.assertFalse(r.feasible)
        self.assertGreaterEqual(r.predicted_reroute_s, 2276.0)


class TestNoOverRefusal(unittest.TestCase):

    def test_cold_start_small_design_feasible(self):
        """fir-class 12,367 cells, empty history, 600s remaining:
        floor 74.2 + 30 ≤ 600 − 120 → feasible (no over-refusal)."""
        r = assess_destructive_reroute(
            remaining_wall_s=600.0,
            tool_call_details=[],
            input_cell_count=12_367,
        )
        self.assertTrue(r.feasible)

    def test_small_design_not_over_refused(self):
        """mini-isp class (8,414 cells) with ample budget → feasible."""
        r = assess_destructive_reroute(
            remaining_wall_s=600.0,
            tool_call_details=[],
            input_cell_count=8_414,
        )
        self.assertTrue(r.feasible)
        self.assertIn("OK", r.reason)

    def test_optical_class_incremental_headroom_feasible(self):
        """84,422 cells with a completed 90s route and 3000s remaining:
        predicted = max(2×90, 506.5) = 506.5 → fits comfortably."""
        r = assess_destructive_reroute(
            remaining_wall_s=3000.0,
            tool_call_details=[
                {"tool_name": "vivado_route_design", "elapsed_time": 90.0},
            ],
            input_cell_count=84_422,
        )
        self.assertTrue(r.feasible)
        # cells floor (506.5) dominates the K-branch (180.0) here
        self.assertAlmostEqual(
            r.predicted_reroute_s, 84_422 * CELLS_SCALED_RATE_S, places=2)


# Boundary correctness — thresholds derived from the module constants so
# the pair stays valid if a constant is ever re-calibrated.

class TestBoundary(unittest.TestCase):

    CELLS = 100_000  # cold-start → predicted = 100,000 × 0.006 = 600.0s

    def _threshold(self) -> float:
        predicted = self.CELLS * CELLS_SCALED_RATE_S
        return predicted + BANKING_RESERVE_S + SAFETY_MARGIN_S

    def test_boundary_just_infeasible(self):
        r = assess_destructive_reroute(
            remaining_wall_s=self._threshold() - 0.1,
            tool_call_details=[],
            input_cell_count=self.CELLS,
        )
        self.assertFalse(r.feasible)

    def test_boundary_just_feasible(self):
        r = assess_destructive_reroute(
            remaining_wall_s=self._threshold() + 0.1,
            tool_call_details=[],
            input_cell_count=self.CELLS,
        )
        self.assertTrue(r.feasible)


# Robustness — malformed history must be skipped, never raise; inputs
# must not be mutated (purity contract).

class TestMalformedHistory(unittest.TestCase):

    def test_malformed_entries_skipped(self):
        """elapsed_time None / missing keys / non-dict entries are ignored
        — prediction falls back to the cells-scaled floor, no exception."""
        history = [
            {"tool_name": "vivado_phys_opt_design", "elapsed_time": None},
            {"iteration": 3},                      # missing everything
            {"elapsed_time": 50.0},                # no tool_name/cmd_head
            "not-a-dict",
            {"tool_name": "vivado_route_design"},  # missing elapsed_time
        ]
        pred = predict_reroute_seconds(history, BOOM_CELLS)
        self.assertAlmostEqual(pred, BOOM_CELLS * CELLS_SCALED_RATE_S,
                               places=3)

    def test_inputs_not_mutated(self):
        history = [
            {"tool_name": "vivado_phys_opt_design",
             "elapsed_time": BOOM_PHYS_OPT_S},
        ]
        snapshot = [dict(tc) for tc in history]
        predict_reroute_seconds(history, BOOM_CELLS)
        assess_destructive_reroute(
            remaining_wall_s=1650.0,
            tool_call_details=history,
            input_cell_count=BOOM_CELLS,
        )
        self.assertEqual(history, snapshot)

    def test_assessment_uses_constant_defaults(self):
        """Default reserve/margin/k/rate wired to the module constants."""
        self.assertEqual(HEAVY_OP_MULTIPLIER_K, 2.0)
        self.assertEqual(CELLS_SCALED_RATE_S, 0.006)
        self.assertEqual(BANKING_RESERVE_S, 30.0)
        self.assertEqual(SAFETY_MARGIN_S, 120.0)


class DoubleBlindColdStartTests(unittest.TestCase):
    """external review S2: cell-count None + empty history must
    REFUSE (predict inf), never return 0/'feasible' with zero data."""

    def test_no_cells_no_history_predicts_inf(self):
        self.assertEqual(
            predict_reroute_seconds(
                tool_call_details=[], input_cell_count=None),
            float("inf"))

    def test_no_cells_no_history_assessment_infeasible(self):
        a = assess_destructive_reroute(
            tool_call_details=[], input_cell_count=None,
            remaining_wall_s=100000.0)
        self.assertFalse(a.feasible)


if __name__ == "__main__":
    unittest.main()
