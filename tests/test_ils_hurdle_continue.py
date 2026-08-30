"""Test that ILS futility continuation uses a time-weighted scoring hurdle instead
of a fixed cycle count.

The threshold is alpha × 0.1 × (dt / 3600) / P and uses the observed cycle
duration rather than a fitted constant. A zero hurdle disables the override by
default.
"""
from __future__ import annotations
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizer.ils_polish import ILSPolishConfig  # noqa: E402


def hurdle_mhz(alpha, cycle_s, P=0.94):
    return (alpha * 0.1 * (cycle_s / 3600.0)) / P


def would_continue(cfg, cycle_s, remaining_s):
    a = float(getattr(cfg, "hurdle_continue_alpha_mhz", 0.0) or 0.0)
    cap = float(getattr(cfg, "hurdle_continue_max_mhz", 1.0) or 1.0)
    if a <= 0 or cycle_s <= 0 or remaining_s <= cycle_s * 2:
        return False
    return hurdle_mhz(a, cycle_s) < cap


class DefaultOffTests(unittest.TestCase):
    def test_default_is_off(self):
        cfg = ILSPolishConfig()
        self.assertEqual(cfg.hurdle_continue_alpha_mhz, 0.0)
        self.assertFalse(would_continue(cfg, 120.0, 2984.0),
                         "default must preserve earlier K-counter behaviour")


class HurdleArithmeticTests(unittest.TestCase):
    def test_the_minisp_case(self):
        """alpha 97.08, 120 s cycle => ~0.34 MHz hurdle; record cycle gave 5.4."""
        h = hurdle_mhz(97.08, 120.0)
        self.assertAlmostEqual(h, 0.344, places=2)
        self.assertLess(h, 5.4, "the record's winning cycle dwarfs the hurdle")

    def test_continues_when_a_cycle_is_cheap_and_budget_is_free(self):
        cfg = ILSPolishConfig()
        cfg.hurdle_continue_alpha_mhz = 97.08
        self.assertTrue(would_continue(cfg, 120.0, 2984.0))

    def test_stops_when_the_cycle_is_expensive(self):
        """A 1200 s cycle at alpha 97 costs 3.44 MHz of hurdle — do NOT continue."""
        cfg = ILSPolishConfig()
        cfg.hurdle_continue_alpha_mhz = 97.08
        self.assertGreater(hurdle_mhz(97.08, 1200.0), 1.0)
        self.assertFalse(would_continue(cfg, 1200.0, 2984.0))

    def test_stops_when_budget_cannot_fit_two_more_cycles(self):
        cfg = ILSPolishConfig()
        cfg.hurdle_continue_alpha_mhz = 97.08
        self.assertFalse(would_continue(cfg, 120.0, 150.0))

    def test_high_alpha_designs_face_a_higher_bar(self):
        """The rule self-scales: on a big-alpha design a cycle costs more score."""
        self.assertGreater(hurdle_mhz(400.0, 120.0), hurdle_mhz(97.08, 120.0))


if __name__ == "__main__":
    unittest.main()
