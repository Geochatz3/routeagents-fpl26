"""Tests size-aware fallback runtime estimates for risky tools without history.

The fallback may reduce the blind constant when size evidence supports it, but
it must never exceed that constant. This keeps budget gating no more
conservative than the history-free behavior.
"""
import os
import unittest
from unittest import mock

import dcp_optimizer
from dcp_optimizer import DCPOptimizer


def _opt(cells):
    """A DCPOptimizer with only what the estimator touches, no __init__ side effects."""
    o = DCPOptimizer.__new__(DCPOptimizer)
    o._tool_runtime_history = {}
    o._input_cell_count = cells
    return o


class SizeAwareToolEstimateTests(unittest.TestCase):

    # ---- the defect, and that it is fixed -------------------------------

    def test_mini_isp_sized_design_is_no_longer_refused_at_600s(self):
        """8,414 cells: the estimate must drop far below the blind constant."""
        est = _opt(8414)._no_history_risky_estimate_s()
        self.assertLess(est, dcp_optimizer.DEFAULT_RISKY_RUNTIME_S)
        # 8,414 x 0.00435 = 36.6s; x1.3 = 47.6s
        self.assertAlmostEqual(est, 8414 * 0.004350 * 1.3, places=3)

    def test_the_exact_chain40_refusal_no_longer_happens(self):
        """Reproduce that run's numbers: 277s remaining must now be enough."""
        o = _opt(8414)
        est = o._no_history_risky_estimate_s()
        self.assertGreater(277.0, est,
                           "277s remaining must now clear the estimate (chain40 "
                           "refused here with the 600s constant)")

    # ---- the safety invariant: never MORE conservative -------------------

    def test_large_design_is_unchanged(self):
        """532,160 cells (ispd16): size model exceeds the constant -> unchanged."""
        self.assertEqual(_opt(532160)._no_history_risky_estimate_s(),
                         dcp_optimizer.DEFAULT_RISKY_RUNTIME_S)

    def test_never_exceeds_the_blind_constant_across_the_whole_corpus(self):
        """The min() invariant, swept over every benchmark scale we ship."""
        for cells in (1, 100, 8414, 13980, 37019, 100000, 137727,
                      377000, 532160, 2000000):
            with self.subTest(cells=cells):
                self.assertLessEqual(_opt(cells)._no_history_risky_estimate_s(),
                                     dcp_optimizer.DEFAULT_RISKY_RUNTIME_S)

    def test_tiny_design_cannot_produce_an_absurd_estimate(self):
        """A 1-cell design must not yield a sub-MIN_USEFUL estimate."""
        self.assertGreaterEqual(_opt(1)._no_history_risky_estimate_s(),
                                dcp_optimizer.MIN_USEFUL_TOOL_SECONDS)

    # ---- every fallback lands on CURRENT behaviour -----------------------

    def test_unknown_cell_count_falls_back_to_the_constant(self):
        for bad in (None, 0, -5):
            with self.subTest(cells=bad):
                self.assertEqual(_opt(bad)._no_history_risky_estimate_s(),
                                 dcp_optimizer.DEFAULT_RISKY_RUNTIME_S)

    def test_missing_attribute_falls_back_to_the_constant(self):
        o = DCPOptimizer.__new__(DCPOptimizer)
        o._tool_runtime_history = {}
        self.assertEqual(o._no_history_risky_estimate_s(),
                         dcp_optimizer.DEFAULT_RISKY_RUNTIME_S)

    def test_size_model_returning_zero_falls_back_to_the_constant(self):
        with mock.patch("optimizer.deep_replace_sibling.predict_place_route_s",
                        return_value=(0.0, "unmeasurable")):
            self.assertEqual(_opt(8414)._no_history_risky_estimate_s(),
                             dcp_optimizer.DEFAULT_RISKY_RUNTIME_S)

    def test_exception_falls_back_to_the_constant(self):
        with mock.patch("optimizer.deep_replace_sibling.predict_place_route_s",
                        side_effect=RuntimeError("boom")):
            self.assertEqual(_opt(8414)._no_history_risky_estimate_s(),
                             dcp_optimizer.DEFAULT_RISKY_RUNTIME_S)

    @mock.patch.dict(os.environ, {"FPL26_SIZE_AWARE_TOOL_EST": "0"})
    def test_kill_switch_restores_the_old_behaviour_exactly(self):
        self.assertEqual(_opt(8414)._no_history_risky_estimate_s(),
                         dcp_optimizer.DEFAULT_RISKY_RUNTIME_S)

    # ---- measured history still wins over any model ----------------------

    def test_history_takes_precedence_and_is_unaffected(self):
        o = _opt(8414)
        o._tool_runtime_history = {"vivado_place_design": [12.0, 900.0, 31.0]}
        self.assertEqual(o._estimate_tool_runtime("vivado_place_design",
                                                  is_risky=True), 900.0)

    def test_cheap_tools_untouched(self):
        self.assertEqual(
            _opt(8414)._estimate_tool_runtime("vivado_run_tcl", is_risky=False),
            dcp_optimizer.DEFAULT_CHEAP_RUNTIME_S)


if __name__ == "__main__":
    unittest.main()
