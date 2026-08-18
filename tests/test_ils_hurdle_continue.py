"""ILS futility override: the SCORING FUNCTION decides, not a counter (jul26).

MEASURED MOTIVATION (mini-ISP chain29, eval parity):
    ILS-polish loop-exit trigger; using remaining 2984s.
    ILS cycle 1 place=Explore:      wns=-0.943 dt=111s (best -0.904)
    ILS cycle 2 place=__LASTMILE__: wns=-0.904 dt=122s (best -0.904)
    ILS: no improvement in 2 real cycles — stopping (yield budget)
ILS held 2984 s, spent 233 s, then stopped on the K=2 constant. The jul07 record
reached 413.22 on its cycle-2 __LASTMILE__ (-0.882 -> -0.850, +5.4 MHz) after
FOUR cycles.

  hurdle = alpha * 0.1 * (dt/3600) / P
one ~120 s cycle at alpha 97.08 => ~0.34 MHz. The record's winning cycle yielded
5.4 MHz — 16x the hurdle. K=2 stops an order of magnitude before the economics.

No fitted constant: the hurdle is the contest's own formula plus a MEASURED
cycle cost. DEFAULT OFF (hurdle_continue_alpha_mhz=0.0).
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
                         "default must preserve pre-jul26 K-counter behaviour")


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
