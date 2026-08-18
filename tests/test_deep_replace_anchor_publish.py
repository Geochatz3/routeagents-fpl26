"""deep-replace must PUBLISH its measured place+route cycle (jul26).

THE BUG: run_deep_replace_sibling performs a real full place+route on this
design and this box and returns place_s/route_s — and nothing consumed them.
Every later cost-gated stage therefore fell back to
replace_gamble_cost_basis()'s rung 3 (0.0 -> FAIL CLOSED). Observed live in
chain15b:

    deep-replace: B1 BANKED ... place=817s route=736s
    deep-replace[tail]: skipped (no measured cost anchor (fail closed))

i.e. a stage refused for want of a measurement that had been taken minutes
earlier on the same run. Same class as the B2 predictive gate (54d875e) and the
deep-replace ship gap (31fc835): the information exists, the consumer cannot
see it.

These tests pin the CONTRACT, not the plumbing:
  * a measured cycle becomes the anchor when it beats what is there;
  * the anchor is RAISE-ONLY (never made more permissive);
  * downstream cost basis actually changes as a result;
  * a zero/absent measurement leaves the anchor untouched.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimizer.ils_polish import ILSPolishConfig  # noqa: E402
from optimizer.replace_gamble import replace_gamble_cost_basis  # noqa: E402


class _Res:
    """Stand-in for the DeepReplaceResult fields the publisher reads."""
    def __init__(self, place_s=0.0, route_s=0.0):
        self.place_s = place_s
        self.route_s = route_s


def publish(cfg, res):
    """The extracted publisher contract (mirrors dcp_optimizer.py)."""
    measured = float(getattr(res, "place_s", 0.0) or 0.0) + \
               float(getattr(res, "route_s", 0.0) or 0.0)
    if measured <= 0:
        return False
    prev = float(getattr(cfg, "expected_heavy_cycle_s", 0.0) or 0.0)
    if measured > prev:
        cfg.expected_heavy_cycle_s = measured
        return True
    return False


class AnchorPublishTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ILSPolishConfig()
        self.cfg.expected_heavy_cycle_s = 0.0

    def test_the_chain15b_case_no_longer_fails_closed(self):
        """B1 measured place=817 route=736; the tail must then have a basis."""
        self.assertEqual(replace_gamble_cost_basis(self.cfg, None), 0.0,
                         "precondition: with no anchor the stage fails closed")
        self.assertTrue(publish(self.cfg, _Res(place_s=817.0, route_s=736.0)))
        self.assertEqual(replace_gamble_cost_basis(self.cfg, None), 1553.0)

    def test_anchor_is_raise_only(self):
        """A cheaper cycle must NOT lower the anchor.

        A smaller basis makes affordability gates MORE permissive; raise-only
        guarantees this change can never turn a refusal into a start that
        cannot finish.
        """
        self.cfg.expected_heavy_cycle_s = 2000.0
        self.assertFalse(publish(self.cfg, _Res(place_s=300.0, route_s=200.0)))
        self.assertEqual(self.cfg.expected_heavy_cycle_s, 2000.0)

    def test_higher_measurement_replaces_a_derived_anchor(self):
        self.cfg.expected_heavy_cycle_s = 900.0
        self.assertTrue(publish(self.cfg, _Res(place_s=817.0, route_s=736.0)))
        self.assertEqual(self.cfg.expected_heavy_cycle_s, 1553.0)

    def test_zero_measurement_is_a_no_op(self):
        """Fail-safe: an unmeasured stage must not clobber the anchor."""
        self.cfg.expected_heavy_cycle_s = 1200.0
        self.assertFalse(publish(self.cfg, _Res()))
        self.assertEqual(self.cfg.expected_heavy_cycle_s, 1200.0)

    def test_observed_ils_costs_still_win(self):
        """Rung 1 of the ladder outranks the published anchor — unchanged."""
        publish(self.cfg, _Res(place_s=817.0, route_s=736.0))
        basis = replace_gamble_cost_basis(self.cfg, {0: 400.0})
        self.assertEqual(basis, 400.0,
                         "observed ILS combo cost must still take precedence")


if __name__ == "__main__":
    unittest.main()
