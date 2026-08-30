"""Wiring tests for the derived wall-economics stop rule.

optimizer/wall_economics.py is unit-tested in test_wall_economics.py (the
arithmetic).  THIS file tests the INTEGRATION into dcp_optimizer:

  - the observation window is sized from COMPLETED heavy moves only, so a
    cheap `get_property` query cannot shrink it (that would make the rule
    fire after seconds instead of after a real move's worth of wall);
  - the rule arms the EXISTING wall-handback signal, inheriting its locked
    guards (never before a banked accept, first-reason-wins);
  - it is OBSERVE-ONLY only while --wall-handback is off.  The Makefile
    ship targets pass --wall-handback by default (WALL_HANDBACK=0 disables),
    so on the ship path the signal is acted on; with the flag off, wiring
    it is zero behavior change.

These tests are the evidence that adding a fourth arming site did not alter
flag-off behavior.
"""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dcp_optimizer import DCPOptimizer  # noqa: E402


def _make_optimizer(tmp_path: Path) -> DCPOptimizer:
    opt = DCPOptimizer(api_key="test", run_dir=tmp_path)
    # A banked accept exists unless a test says otherwise (the arm guard is
    # covered by test_wall_handback.py; here it must not mask our signal).
    opt._best_valid_dcp = tmp_path / "best_valid.dcp"
    opt.clock_period = 10.0          # 100 MHz target
    opt.initial_wns = -10.0          # init fmax = 1000/20 = 50 MHz
    opt.best_wns = -5.0              # best fmax = 1000/15 ≈ 66.67 MHz → α ≈ +16.67
    opt.start_time = time.time() - 5000.0
    opt._budget_deadline = time.time() + 1200.0   # 1200 s remaining
    return opt


class ObservationWindowTests(unittest.TestCase):
    """The window is a MEASUREMENT — cheapest completed heavy move."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = _make_optimizer(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_heavy_move_yet_is_unmeasurable(self):
        self.assertIsNone(self.opt._wall_economics_window_s())

    def test_window_is_the_cheapest_completed_heavy_move(self):
        self.opt._heavy_move_seconds = [900.0, 300.0, 1500.0]
        self.assertEqual(self.opt._wall_economics_window_s(), 300.0)

    def test_rule_cannot_fire_without_a_measured_window(self):
        """Fail-safe: no heavy move completed => keep going, never stop."""
        self.opt.last_improvement_time = time.time() - 99999.0
        self.opt._heavy_move_seconds = []
        self.opt._maybe_arm_wall_economics_stop()
        self.assertIsNone(self.opt._exit_early_reason)


class ArmingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = _make_optimizer(Path(self.tmp.name))
        self.opt._heavy_move_seconds = [600.0]

    def tearDown(self):
        self.tmp.cleanup()

    def test_arms_after_a_full_heavy_move_with_no_gain(self):
        self.opt.last_improvement_time = time.time() - 900.0   # > 600 s window
        self.opt._maybe_arm_wall_economics_stop()
        self.assertIsNotNone(self.opt._exit_early_reason)
        self.assertTrue(self.opt._exit_early_reason.startswith("wall_economics:"))

    def test_does_not_arm_before_one_full_heavy_move(self):
        """Not yet a fair chance — less than one heavy move since the gain."""
        self.opt.last_improvement_time = time.time() - 120.0    # < 600 s
        self.opt._maybe_arm_wall_economics_stop()
        self.assertIsNone(self.opt._exit_early_reason)

    def test_does_not_arm_without_a_banked_accept(self):
        """Inherits the locked never-trim-before-banked-accept guard."""
        self.opt._best_valid_dcp = None
        self.opt.last_improvement_time = time.time() - 900.0
        self.opt._maybe_arm_wall_economics_stop()
        self.assertIsNone(self.opt._exit_early_reason)

    def test_does_not_arm_with_no_alpha(self):
        """alpha <= 0 => nothing banked worth protecting; fail open."""
        self.opt.best_wns = self.opt.initial_wns
        self.opt.last_improvement_time = time.time() - 900.0
        self.opt._maybe_arm_wall_economics_stop()
        self.assertIsNone(self.opt._exit_early_reason)

    def test_first_reason_wins(self):
        self.opt._exit_early_reason = "ils_no_improve"
        self.opt.last_improvement_time = time.time() - 900.0
        self.opt._maybe_arm_wall_economics_stop()
        self.assertEqual(self.opt._exit_early_reason, "ils_no_improve")


class DefaultOffTests(unittest.TestCase):
    """The ship path does not pass --wall-handback: arming must not act."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = _make_optimizer(Path(self.tmp.name))
        self.opt._heavy_move_seconds = [600.0]
        self.opt.last_improvement_time = time.time() - 900.0

    def tearDown(self):
        self.tmp.cleanup()

    def test_armed_but_does_not_break_when_switch_off(self):
        self.opt._maybe_arm_wall_economics_stop()
        self.assertIsNotNone(self.opt._exit_early_reason)   # observed
        self.assertFalse(self.opt._wall_handback_break_due())  # but inert

    def test_breaks_only_when_switch_on(self):
        self.opt._wall_handback_enabled = True
        self.opt._maybe_arm_wall_economics_stop()
        self.assertTrue(self.opt._wall_handback_break_due())


if __name__ == "__main__":
    unittest.main()
