"""Tests for optimizer.cross_model_steering.

The public tree ships with an EMPTY hint registry (per-benchmark steering
data is excluded from the release — see docs/PROVENANCE.md), so
these tests exercise the mechanism with synthetic hints injected into
RECIPE_HINTS.

Covers:
  - Default OFF (ENV_FLAG unset → no hint returned for any design).
  - ENV_FLAG=1 enables iter-1 hints for registered designs.
  - Unknown designs return None even when enabled.
  - get_continue_hint conditions: fraction-recovered threshold, budget
    threshold, design-applicability.
  - The 50%-recovery and 15-min-budget gates do not fire spuriously.
  - The shipped registry is empty.
"""
from __future__ import annotations

import logging
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizer import cross_model_steering as cms

_SYNTH_HINTS = {
    "synthetic_design_a": cms.RecipeHint(
        design_name="synthetic_design_a",
        preferred_first_action=(
            "recipe_post_route_phys_opt_sweep (invokes "
            "vivado_phys_opt_design -critical_cell_opt)"
        ),
        rationale=(
            "The prior model committed the critical_cell_opt sub-flag for "
            "+19.44 MHz on this design class."
        ),
        avoid=(
            "recipe_cell_replacement — the analyze_net_detour step starved "
            "the subsequent route_design in the prior trial. Net result: 0 MHz."
        ),
        historical_target_fmax_mhz=19.44,
    ),
    "synthetic_design_b": cms.RecipeHint(
        design_name="synthetic_design_b",
        preferred_first_action=(
            "vivado_run_tcl 'place_design -unplace' then vivado_place_design "
            "directive=Explore, then route Default, then phys_opt "
            "AlternateFlowWithRetiming"
        ),
        rationale=(
            "The prior model's dominant contribution was a full re-place "
            "Explore + route + retime chain (+51.46 MHz)."
        ),
        avoid=(
            "declaring optimization complete while wall budget remains and "
            "place_design -directive Explore has not been attempted."
        ),
        historical_target_fmax_mhz=51.46,
    ),
}


def _with_synth_registry():
    """Patch the (empty) shipped registry with synthetic hints."""
    return mock.patch.dict(cms.RECIPE_HINTS, _SYNTH_HINTS, clear=True)


class GatingTests(unittest.TestCase):
    def test_steering_disabled_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(cms.ENV_FLAG, None)
            self.assertFalse(cms.is_steering_enabled())

    def test_steering_enabled_only_on_literal_1(self):
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            self.assertTrue(cms.is_steering_enabled())
        # Any other truthy-looking value must NOT enable steering.
        for v in ("true", "yes", "on", "0", "", "True", "TRUE"):
            with mock.patch.dict(os.environ, {cms.ENV_FLAG: v}):
                self.assertFalse(cms.is_steering_enabled(),
                                 f"value {v!r} should not enable steering")

    def test_steering_whitespace_trimmed(self):
        # "1\n" or "  1  " from a misconfigured shell should still enable.
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: " 1 "}):
            self.assertTrue(cms.is_steering_enabled())


class ShippedRegistryTests(unittest.TestCase):
    """The public tree must ship with no per-benchmark steering data."""

    def test_shipped_registry_is_empty(self):
        self.assertEqual(cms.RECIPE_HINTS, {})

    def test_enabled_with_empty_registry_returns_none(self):
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            self.assertIsNone(cms.get_iter1_hint("any_design"))
            self.assertIsNone(cms.get_continue_hint("any_design", 1.0, 2000.0))


class Iter1HintTests(unittest.TestCase):
    def setUp(self):
        self._reg = _with_synth_registry()
        self._reg.start()
        self.addCleanup(self._reg.stop)

    def test_disabled_returns_none_for_known_design(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(cms.ENV_FLAG, None)
            self.assertIsNone(cms.get_iter1_hint("synthetic_design_a"))

    def test_enabled_unknown_design_returns_none(self):
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            self.assertIsNone(cms.get_iter1_hint("not_in_registry"))

    def test_enabled_none_design_returns_none(self):
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            self.assertIsNone(cms.get_iter1_hint(None))

    def test_enabled_returns_text_for_registered_designs(self):
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            for design in _SYNTH_HINTS:
                hint = cms.get_iter1_hint(design)
                self.assertIsNotNone(hint, f"missing hint for {design}")
                self.assertIn(design, hint)
                # Hint must mention historical-target context and disclaim
                # guarantees.
                self.assertIn("Historical pre-migration", hint)
                self.assertIn("NOT a guarantee", hint)

    def test_hint_carries_registry_content(self):
        # The wrapper must carry the actual steering content, not just
        # boilerplate.
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            hint = cms.get_iter1_hint("synthetic_design_a")
            self.assertIn("recipe_post_route_phys_opt_sweep", hint)
            self.assertIn("critical_cell_opt", hint)
            self.assertIn("19.44", hint)
            # Must warn against the trap.
            self.assertIn("recipe_cell_replacement", hint)
            self.assertIn("analyze_net_detour", hint)

    def test_iter1_hint_logs_at_info(self):
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            with self.assertLogs(cms.logger, level=logging.INFO) as cap:
                cms.get_iter1_hint("synthetic_design_a")
            joined = "\n".join(cap.output)
            self.assertIn("cross_model_steering", joined)
            self.assertIn("synthetic_design_a", joined)
            self.assertIn("19.44", joined)


class ContinueHintTests(unittest.TestCase):
    def setUp(self):
        self._reg = _with_synth_registry()
        self._reg.start()
        self.addCleanup(self._reg.stop)
        self.design = "synthetic_design_b"
        self.target = cms.RECIPE_HINTS[self.design].historical_target_fmax_mhz
        self.assertAlmostEqual(self.target, 51.46, places=2)

    def test_disabled_returns_none(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(cms.ENV_FLAG, None)
            self.assertIsNone(cms.get_continue_hint(self.design, 5.0, 2000.0))

    def test_unknown_design_returns_none(self):
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            self.assertIsNone(cms.get_continue_hint("unknown", 1.0, 2000.0))

    def test_above_50pct_recovered_skips(self):
        # 60% recovered → no override; honor the LLM's stop signal.
        gain = 0.6 * self.target
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            self.assertIsNone(cms.get_continue_hint(self.design, gain, 2000.0))

    def test_below_50pct_with_low_budget_skips(self):
        # 30% recovered but only 600s remaining (<= 900s threshold) →
        # no override; honor the LLM's stop because budget is tight.
        gain = 0.3 * self.target
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            self.assertIsNone(cms.get_continue_hint(self.design, gain, 600.0))

    def test_below_50pct_with_high_budget_fires(self):
        # ~31% recovered (+15.94 of +51.46) AND ~1400s budget remaining
        # → should fire.
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            msg = cms.get_continue_hint(self.design, 15.94, 1400.0)
            self.assertIsNotNone(msg)
            self.assertIn("ANTI-PREMATURE-EXIT", msg)
            self.assertIn("synthetic_design_b", msg)
            self.assertIn("15.94", msg)
            self.assertIn("51.46", msg)
            # Must contain the suggested next move from the registry.
            self.assertIn("place_design -unplace", msg)

    def test_continue_hint_logs_trigger_at_info(self):
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            with self.assertLogs(cms.logger, level=logging.INFO) as cap:
                cms.get_continue_hint(self.design, 15.94, 1400.0)
            joined = "\n".join(cap.output)
            self.assertIn("TRIGGERED", joined)
            self.assertIn("synthetic_design_b", joined)

    def test_continue_hint_logs_skip_for_above_threshold(self):
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            with self.assertLogs(cms.logger, level=logging.INFO) as cap:
                cms.get_continue_hint(self.design, 30.0, 2000.0)
            joined = "\n".join(cap.output)
            self.assertIn("SKIPPED", joined)

    def test_nonpositive_target_returns_none(self):
        # A target of 0 makes "fraction recovered" undefined → no hint.
        broken = cms.RecipeHint(
            design_name="zero_target", preferred_first_action="x",
            rationale="y", avoid="z", historical_target_fmax_mhz=0.0,
        )
        with mock.patch.dict(cms.RECIPE_HINTS, {"zero_target": broken}):
            with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
                self.assertIsNone(
                    cms.get_continue_hint("zero_target", 0.0, 2000.0))


if __name__ == "__main__":
    unittest.main()
