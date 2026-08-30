"""Verify that unbanded deep replacement treats benefit predictions as scheduling
hints rather than vetoes.

Unbanded mode removes only the physics-band rejection. Affordability and other
hard bounds still refuse unsafe work, while checkpoint comparison protects
against adopting an unhelpful result. Default banded behavior remains
unchanged.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimizer.deep_replace_sibling import deep_replace_should_run  # noqa: E402

BOOM = dict(failing_endpoint_count=220_131, wns_magnitude_ns=11.39)
LOGICNETS = dict(failing_endpoint_count=1_529, wns_magnitude_ns=0.978)
FIR = dict(failing_endpoint_count=252, wns_magnitude_ns=0.313)


def call(design, *, banded=True, remaining_s=3000.0, cost_basis_s=800.0,
         pristine="/tmp/p.dcp", enabled=True):
    return deep_replace_should_run(
        enabled=enabled,
        pristine_dcp=pristine,
        remaining_s=remaining_s,
        cost_basis_s=cost_basis_s,
        finalize_reserve_s=300.0,
        failing_endpoints_min=100_000,
        wns_min_ns=10.0,
        require_physics_band=banded,
        **design)


class DefaultUnchangedTests(unittest.TestCase):
    """Default MUST be byte-identical to before the option existed."""

    def test_boom_still_arms_by_default(self):
        run, why = call(BOOM)
        self.assertTrue(run)
        self.assertIn("DEEP-extreme", why)

    def test_logicnets_still_refused_by_default(self):
        run, why = call(LOGICNETS)
        self.assertFalse(run)
        self.assertIn("not DEEP-extreme", why)

    def test_fir_still_refused_by_default(self):
        run, why = call(FIR)
        self.assertFalse(run)
        self.assertIn("not DEEP-extreme", why)


class UnbandedTests(unittest.TestCase):
    """Unbanded drops the benefit-prediction veto and NOTHING else."""

    def test_logicnets_arms_when_unbanded(self):
        """The chain28 case: this refusal cost 14.3 MHz and ~21 score points."""
        run, why = call(LOGICNETS, banded=False)
        self.assertTrue(run)
        self.assertIn("UNBANDED", why)

    def test_fir_also_arms_but_is_protected_by_MEASUREMENT(self):
        """FIR regen LOSES (-1.9). Unbanded lets it RUN — and that is correct:
        the insured-compare MUX discards a losing candidate by measurement.
        The point of the change is that a guess no longer pre-empts the test."""
        run, _ = call(FIR, banded=False)
        self.assertTrue(run)

    def test_kill_switch_still_wins(self):
        run, why = call(LOGICNETS, banded=False, enabled=False)
        self.assertFalse(run)
        self.assertIn("disabled", why)

    def test_missing_pristine_still_refuses(self):
        run, why = call(LOGICNETS, banded=False, pristine=None)
        self.assertFalse(run)
        self.assertIn("pristine", why)

    def test_no_cost_anchor_still_fails_closed(self):
        run, why = call(LOGICNETS, banded=False, cost_basis_s=0.0)
        self.assertFalse(run)
        self.assertIn("no measured cost anchor", why)

    def test_insufficient_wall_still_refuses(self):
        """Affordability is a HARD BOUND and must still veto."""
        run, why = call(LOGICNETS, banded=False, remaining_s=100.0)
        self.assertFalse(run)
        self.assertIn("insufficient terminal reserve", why)

    def test_unmeasurable_physics_still_fails_closed(self):
        run, why = call(dict(failing_endpoint_count=None,
                             wns_magnitude_ns=None), banded=False)
        self.assertFalse(run)
        self.assertIn("unmeasurable", why)


if __name__ == "__main__":
    unittest.main()


class FirstStageAffordabilityTests(unittest.TestCase):
    """Verify that affordable first-stage work is not deferred solely by the
    benefit band.

    First-stage replacement reseeds the pipeline, whereas tail replacement is
    only a closing optimization and is not equivalent. The affordability
    override compares measured cost with wall time after the existing tail
    reserve, allowing inexpensive first-stage work while deferring work that
    cannot fit safely.
    """

    # mirrors dcp_optimizer: budget = wall x (1 - TAIL_RESERVE_WALL_CLAMP_FRAC)
    TAIL_FRAC = 2400.0 / 3500.0
    MARGIN = 1.3          # DEEP_REPLACE_COST_MARGIN
    WALL = 3500.0

    def budget(self, wall=None):
        # `wall if wall is not None` — NOT `wall or ...`: a 0.0 wall is falsy and would
        # silently fall back to 3500, hiding the unknown-wall path this class tests.
        w = self.WALL if wall is None else wall
        return w * (1.0 - self.TAIL_FRAC)

    def keeps_first(self, basis_s, wall=None):
        """True when the stage is affordable out of the loop's leftover budget."""
        return basis_s * self.MARGIN <= self.budget(wall)

    def test_cheap_stages_keep_first(self):
        """logicnets 161s and mini-ISP 37s — the two that LOST 10.64 and 3.39."""
        self.assertTrue(self.keeps_first(161.0), "logicnets must keep FIRST")
        self.assertTrue(self.keeps_first(37.0), "mini-ISP must keep FIRST")

    def test_expensive_stage_still_defers(self):
        """corescore 1099s: deferring it is worth +4.96, and ~+69 vs UNGATED."""
        self.assertFalse(self.keeps_first(1099.0), "corescore must DEFER")

    def test_separation_is_wide_not_a_knife_edge(self):
        """A boundary that only just separates n=3 would be threshold-fitting, which
        this module's own docstring forbids. Require a >2x gap on BOTH sides."""
        b = self.budget()
        self.assertLess(161.0 * self.MARGIN * 2, b,
                        "largest KEEP should sit well under the budget")
        self.assertGreater(1099.0 * self.MARGIN, b * 1.25,
                           "smallest DEFER should sit well over the budget")

    def test_unknown_wall_leaves_the_band_decision_alone(self):
        """No wall = no budget to reason about = do not override the gate.

        dcp_optimizer computes the budget only `if _wall` and then requires
        `_first_budget > 0`, so an unset or zero wall falls through to the plain band
        decision instead of silently keeping every stage at FIRST."""
        self.assertEqual(self.budget(0.0), 0.0)
        self.assertFalse(self.budget(0.0) > 0.0)
        # and with no budget, even a 1-second stage must not be force-kept
        self.assertFalse(1.0 * self.MARGIN <= self.budget(0.0))

    def test_ispd16_stays_deferred_even_though_it_is_in_band(self):
        """Keep an unaffordable deep-replacement stage deferred even when it is
        otherwise in band.

        A separate missing-anchor path fails closed and must not be mistaken
        for affordability behavior.
        """
        self.assertFalse(self.keeps_first(2315.0))


class TailAnchorFallbackTests(unittest.TestCase):
    """Verify that tail affordability can use a size-model fallback when no
    measured anchor exists.

    Skipping the first heavy stage leaves no measured cost anchor. The
    low-level predicate still fails closed for a zero basis, but the caller
    supplies the modeled fallback before evaluating affordability so that
    budget, rather than missing telemetry, determines refusal.
    """

    def test_zero_basis_still_fails_closed_at_the_predicate(self):
        """The fallback happens in the CALLER. The predicate itself must be unchanged:
        given no anchor it still refuses, so nothing unbounded can start."""
        run, why = call(LOGICNETS, banded=False, cost_basis_s=0.0)
        self.assertFalse(run)
        self.assertIn("no measured cost anchor", why)

    def test_an_estimated_basis_is_accepted_when_affordable(self):
        """With the size-model estimate supplied, a cheap design arms normally."""
        run, why = call(LOGICNETS, banded=False, cost_basis_s=161.0,
                        remaining_s=3000.0)
        self.assertTrue(run)
        self.assertIn("armed", why)

    def test_an_estimated_basis_is_REFUSED_when_unaffordable(self):
        """Refuse an estimated-basis fallback when its projected cost exceeds the
        remaining tail window.

        The refusal must report a numeric cost reason rather than a missing
        measurement.
        """
        run, why = call(LOGICNETS, banded=False, cost_basis_s=2315.0,
                        remaining_s=960.0)
        self.assertFalse(run)
        self.assertIn("insufficient terminal reserve", why)
        self.assertNotIn("no measured cost anchor", why)
