"""Verify that deep-replacement affordability reserves finalization exactly once.

The supplied remaining time already excludes the finalization reserve, so
adding that reserve to the estimated need would count it twice. Corrected
calculations must admit feasible work while continuing to reject work that
cannot finish within the available budget.
"""
from __future__ import annotations

import unittest

from optimizer.deep_replace_sibling import (
    DEEP_REPLACE_COST_MARGIN,
    deep_replace_should_run,
)
from tests.source_corpus import dcp_source_text, optimizer_class_source

RESERVE = 300.0


def gate(remaining, anchor, *, fixed, banded=False):
    return deep_replace_should_run(
        enabled=True,
        pristine_dcp="/tmp/pristine.dcp",
        failing_endpoint_count=250_000,
        wns_magnitude_ns=12.0,
        remaining_s=remaining,
        cost_basis_s=anchor,
        finalize_reserve_s=RESERVE,
        failing_endpoints_min=100_000,
        wns_min_ns=10.0,
        require_physics_band=banded,
        reserve_already_in_deadline=fixed,
    )


# (label, remaining_s, anchor_s) — real declines, corrected verdict = ARM.
CORPUS_NEAR_MISSES = [
    ("chain35",            458.0, 125.0),   # 2.8x the work time, declined by 4s
    ("chain35b",           457.0, 125.0),
    ("logicnets_gated",    639.0, 272.0),
    ("optical_shipdef_a",  638.0, 275.0),
    ("fir_hb_on_a",        567.0, 225.0),
    ("spam_bandoff1",      660.0, 305.0),
    ("spam_ladder",        642.0, 298.0),
    ("optical_uni3",       598.0, 291.0),
    ("digit_rerun",        557.0, 262.0),
    ("fir_uni3",           465.0, 237.0),
    ("fir_shipdef_a",      455.0, 231.0),
    ("spam_noreserve",     570.0, 341.0),
    ("optical_uni",        474.0, 282.0),
    ("big_one",            796.0, 600.0),
]

# Real declines that must STAY declined — genuinely cannot finish.
CORPUS_TRUE_NEGATIVES = [
    ("boomv2_uni3",        273.0, 1418.0),
    ("digit_uni3",          91.0, 1656.0),
    ("heavy_a",            927.0,  770.0),
    ("heavy_b",            737.0,  666.0),
    ("heavy_c",            655.0,  607.0),
    ("heavy_d",            881.0,  801.0),
    ("heavy_e",            704.0,  767.0),
    ("heavy_f",            492.0,  605.0),
    ("heavy_g",            721.0,  797.0),
]


class DoubleCountTests(unittest.TestCase):
    def test_the_double_count_exists_in_the_default_path(self):
        """Pin the defect itself: 458s remaining for a 125s job is refused."""
        run, why = gate(458.0, 125.0, fixed=False)
        self.assertFalse(run)
        self.assertIn("insufficient terminal reserve", why)
        self.assertIn("reserve 300s", why)
        # need = 125*1.3 + 300 = 462.5, i.e. the reserve is the entire refusal.
        self.assertLess(125.0 * DEEP_REPLACE_COST_MARGIN, 458.0)

    def test_default_is_byte_identical(self):
        """reserve_already_in_deadline defaults False — no behaviour change."""
        explicit = gate(458.0, 125.0, fixed=False)
        implicit = deep_replace_should_run(
            enabled=True, pristine_dcp="/tmp/p.dcp",
            failing_endpoint_count=250_000, wns_magnitude_ns=12.0,
            remaining_s=458.0, cost_basis_s=125.0, finalize_reserve_s=RESERVE,
            failing_endpoints_min=100_000, wns_min_ns=10.0,
            require_physics_band=False,
        )
        self.assertEqual(explicit, implicit)


class CorpusTests(unittest.TestCase):
    def test_near_misses_arm_once_corrected(self):
        for label, rem, anchor in CORPUS_NEAR_MISSES:
            with self.subTest(label):
                self.assertFalse(gate(rem, anchor, fixed=False)[0],
                                 f"{label}: expected the defect to refuse it")
                run, why = gate(rem, anchor, fixed=True)
                self.assertTrue(run, f"{label}: should arm once corrected ({why})")
                self.assertIn("reserve already in deadline", why)

    def test_infeasible_cases_still_declined(self):
        """THE SAFETY PROPERTY. If the correction armed these it would be
        recklessly permissive rather than a bug fix."""
        for label, rem, anchor in CORPUS_TRUE_NEGATIVES:
            with self.subTest(label):
                run, why = gate(rem, anchor, fixed=True)
                self.assertFalse(run, f"{label}: must stay declined")
                self.assertIn("insufficient terminal reserve", why)
                self.assertNotIn("+ reserve", why,
                                 "corrected path must not cite the reserve term")

    def test_corrected_threshold_is_exactly_the_work_estimate(self):
        anchor = 300.0
        need = anchor * DEEP_REPLACE_COST_MARGIN
        self.assertFalse(gate(need - 1.0, anchor, fixed=True)[0])
        self.assertTrue(gate(need + 1.0, anchor, fixed=True)[0])

    def test_correction_can_only_widen_never_narrow(self):
        """It removes a term from `need`, so anything armed before stays armed."""
        for rem in (100.0, 300.0, 458.0, 700.0, 1500.0, 3000.0):
            for anchor in (50.0, 125.0, 275.0, 600.0, 1418.0):
                if gate(rem, anchor, fixed=False)[0]:
                    self.assertTrue(
                        gate(rem, anchor, fixed=True)[0],
                        f"correction suppressed an arm at rem={rem} anchor={anchor}",
                    )


class HardGatesStillApplyTests(unittest.TestCase):
    """The correction touches ONE term. Every other refusal must survive it —
    otherwise it is not a bug fix, it is a hole."""

    def test_kill_switch_still_wins(self):
        run, why = deep_replace_should_run(
            enabled=False, pristine_dcp="/tmp/p.dcp",
            failing_endpoint_count=250_000, wns_magnitude_ns=12.0,
            remaining_s=99_999.0, cost_basis_s=10.0, finalize_reserve_s=RESERVE,
            failing_endpoints_min=100_000, wns_min_ns=10.0,
            require_physics_band=False, reserve_already_in_deadline=True)
        self.assertFalse(run)
        self.assertIn("disabled", why)

    def test_missing_pristine_still_refuses(self):
        run, why = deep_replace_should_run(
            enabled=True, pristine_dcp=None,
            failing_endpoint_count=250_000, wns_magnitude_ns=12.0,
            remaining_s=99_999.0, cost_basis_s=10.0, finalize_reserve_s=RESERVE,
            failing_endpoints_min=100_000, wns_min_ns=10.0,
            require_physics_band=False, reserve_already_in_deadline=True)
        self.assertFalse(run)

    def test_unmeasured_cost_anchor_still_fails_closed(self):
        run, why = gate(99_999.0, 0.0, fixed=True)
        self.assertFalse(run)
        self.assertIn("no measured cost anchor", why)

    def test_physics_band_still_refuses_when_required(self):
        run, why = deep_replace_should_run(
            enabled=True, pristine_dcp="/tmp/p.dcp",
            failing_endpoint_count=10, wns_magnitude_ns=0.5,
            remaining_s=99_999.0, cost_basis_s=10.0, finalize_reserve_s=RESERVE,
            failing_endpoints_min=100_000, wns_min_ns=10.0,
            require_physics_band=True, reserve_already_in_deadline=True)
        self.assertFalse(run)
        self.assertIn("not DEEP-extreme", why)

    def test_unmeasurable_physics_still_fails_closed(self):
        run, why = deep_replace_should_run(
            enabled=True, pristine_dcp="/tmp/p.dcp",
            failing_endpoint_count=None, wns_magnitude_ns=None,
            remaining_s=99_999.0, cost_basis_s=10.0, finalize_reserve_s=RESERVE,
            failing_endpoints_min=100_000, wns_min_ns=10.0,
            require_physics_band=False, reserve_already_in_deadline=True)
        self.assertFalse(run)
        self.assertIn("fail closed", why)


class WiringTests(unittest.TestCase):
    """Pin that the flag is read with an OFF default and reaches the gate — a
    correction nobody can turn on is the drift in miniature."""

    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        cls.text = dcp_source_text()

    def test_flag_default_off(self):
        self.assertIn(
            'os.environ.get("FPL26_DEEP_REPLACE_NO_DOUBLE_RESERVE", "0")',
            self.text)

    def test_flag_is_passed_to_the_gate(self):
        # `_no_double_reserve` must reach the gate as the global opt-in.
        # Check expression membership because other valid conditions may share the gate.
        import re
        m = re.search(r"reserve_already_in_deadline=\(?([^)\n]*)", self.text)
        self.assertIsNotNone(m, "argument no longer passed to the gate")
        self.assertIn("_no_double_reserve", m.group(1))


if __name__ == "__main__":
    unittest.main()
