"""Tests for optimizer.cross_model_steering.

Covers:
  - Default OFF (ENV_FLAG unset → no hint returned for any design).
  - ENV_FLAG=1 enables iter-1 hints for registered designs.
  - Unknown designs return None even when enabled.
  - get_continue_hint conditions: fraction-recovered threshold, budget
    threshold, design-applicability.
  - The 50%-recovery and 15-min-budget gates do not fire spuriously.
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


class Iter1HintTests(unittest.TestCase):
    def test_disabled_returns_none_for_known_design(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(cms.ENV_FLAG, None)
            self.assertIsNone(cms.get_iter1_hint("ispd16_example2"))

    def test_enabled_unknown_design_returns_none(self):
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            self.assertIsNone(cms.get_iter1_hint("not_in_registry"))

    def test_enabled_none_design_returns_none(self):
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            self.assertIsNone(cms.get_iter1_hint(None))

    def test_enabled_returns_text_for_registered_designs(self):
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            for design in ("ispd16_example2", "boom_soc", "finn_radioml",
                           "corescore_500_mod"):
                hint = cms.get_iter1_hint(design)
                self.assertIsNotNone(hint, f"missing hint for {design}")
                self.assertIn(design, hint)
                # Hint must mention historical-target context and disclaim
                # guarantees.
                self.assertIn("Historical pre-migration", hint)
                self.assertIn("NOT a guarantee", hint)

    def test_ispd16_hint_mentions_winning_recipe(self):
        # Specific content checks so a future edit doesn't accidentally
        # strip the actual steering content while keeping the wrapper.
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            hint = cms.get_iter1_hint("ispd16_example2")
            self.assertIn("recipe_post_route_phys_opt_sweep", hint)
            self.assertIn("critical_cell_opt", hint)
            self.assertIn("19.44", hint)
            # Must warn against the trap.
            self.assertIn("recipe_cell_replacement", hint)
            self.assertIn("analyze_net_detour", hint)

    def test_boom_soc_hint_mentions_AlternateFlowWithRetiming(self):
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            hint = cms.get_iter1_hint("boom_soc")
            self.assertIn("AlternateFlowWithRetiming", hint)
            self.assertIn("full design scope", hint)
            self.assertIn("2.87", hint)
            self.assertIn("recipe_critical_path_focused_phys_opt", hint)

    def test_finn_hint_mentions_class_g_sequence(self):
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            hint = cms.get_iter1_hint("finn_radioml")
            self.assertIn("place_design -unplace", hint)
            self.assertIn("directive=Explore", hint)
            self.assertIn("AlternateFlowWithRetiming", hint)
            self.assertIn("51.46", hint)

    def test_finn_hint_documents_explore_variance(self):
        # Bug #3 follow-up: Vivado place_design Explore is non-deterministic
        # in interactive mode (no real -seed knob in 2024+). Hint must
        # (a) call out the variance is expected and (b) tell the LLM NOT
        # to re-roll the placement chasing a better outcome — a re-roll
        # can land worse than the first attempt (observed: +59.10 vs +45.46).
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            hint = cms.get_iter1_hint("finn_radioml")
            self.assertIn("non-deterministic", hint,
                          "must call out place_design Explore variance")
            # Record the two observed values so future audits see the band.
            self.assertIn("59.10", hint)
            self.assertIn("45.46", hint)
            # Explicit anti-rerun guidance.
            self.assertIn("do NOT keep re-running", hint)

    def test_corescore_hint_steers_to_auto_1_safety_path(self):
        # corescore must steer toward Auto_1 (the lighter, deterministic
        # placement directive that gives ~+3 MHz reliably), NOT toward
        # Explore (the heavier Class G path that gave +70.88 once and
        # then collapsed to 0 — UNSTABLE RESEARCH).
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            hint = cms.get_iter1_hint("corescore_500_mod")
            # Preferred safety steps must be present.
            self.assertIn("AlternateFlowWithRetiming", hint)
            self.assertIn("directive=Auto_1", hint)
            self.assertIn("critical_pin_opt", hint)
            # Historical target is the safety baseline, not the unstable peak.
            self.assertIn("6.15", hint)

    def test_corescore_hint_warns_against_unstable_explore_path(self):
        # The avoid clause must explicitly warn against chasing +70.88
        # MHz via place_design Explore (the UNSTABLE RESEARCH path).
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            hint = cms.get_iter1_hint("corescore_500_mod")
            self.assertIn("70.88", hint)
            self.assertIn("UNSTABLE RESEARCH", hint)
            # Auto_1 first; Explore only after safety path is shipped.
            self.assertIn("Auto_1 first", hint)

    def test_iter1_hint_logs_at_info(self):
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            with self.assertLogs(cms.logger, level=logging.INFO) as cap:
                cms.get_iter1_hint("ispd16_example2")
            joined = "\n".join(cap.output)
            self.assertIn("cross_model_steering", joined)
            self.assertIn("ispd16_example2", joined)
            self.assertIn("19.44", joined)


class ContinueHintTests(unittest.TestCase):
    def setUp(self):
        # finn_radioml historical target is +51.46 MHz.
        self.design = "finn_radioml"
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
        # Exactly the conditions from the finn forensic trace: ~31%
        # recovered (+15.94 of +51.46) AND ~1400s budget remaining.
        # Should fire.
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            msg = cms.get_continue_hint(self.design, 15.94, 1400.0)
            self.assertIsNotNone(msg)
            self.assertIn("ANTI-PREMATURE-EXIT", msg)
            self.assertIn("finn_radioml", msg)
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
            self.assertIn("finn_radioml", joined)

    def test_continue_hint_logs_skip_for_above_threshold(self):
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            with self.assertLogs(cms.logger, level=logging.INFO) as cap:
                cms.get_continue_hint(self.design, 30.0, 2000.0)
            joined = "\n".join(cap.output)
            self.assertIn("SKIPPED", joined)

    def test_corescore_continue_hint_triggers_after_first_retime(self):
        # Forensic scenario: grok-4.3 stopped at +2.27 MHz on corescore
        # after step 1 (retiming) only. Target is +6.15 MHz, so
        # 2.27 / 6.15 = 37% recovered — below the 50% threshold.
        # Budget remaining after the ~7-min retime is ~45 min = 2700s
        # (well above the 900s gate). Override MUST fire.
        with mock.patch.dict(os.environ, {cms.ENV_FLAG: "1"}):
            msg = cms.get_continue_hint("corescore_500_mod", 2.27, 2700.0)
            self.assertIsNotNone(msg, "corescore continue-hint should fire")
            self.assertIn("ANTI-PREMATURE-EXIT", msg)
            self.assertIn("corescore_500_mod", msg)
            self.assertIn("2.27", msg)
            self.assertIn("6.15", msg)
            # The hint must steer toward Auto_1, not toward Explore.
            self.assertIn("Auto_1", msg)


class RegistryShapeTests(unittest.TestCase):
    """Lock the registry contents so accidental edits show up loudly."""

    def test_four_designs_registered(self):
        self.assertEqual(
            set(cms.RECIPE_HINTS.keys()),
            {"ispd16_example2", "boom_soc", "finn_radioml", "corescore_500_mod"},
        )

    def test_historical_targets_match_known_values(self):
        # If these change, the forensic analysis is stale — fail loudly so
        # the report and the registry stay in sync. corescore's 6.15 is
        # the SAFETY BASELINE, not the unstable +70.88 peak — keeping the
        # registry pinned to the safety target prevents anyone from
        # accidentally pointing the LLM at the unreproducible path.
        targets = {
            "ispd16_example2": 19.44,
            "boom_soc": 2.87,
            "finn_radioml": 51.46,
            "corescore_500_mod": 6.15,
        }
        for design, expected in targets.items():
            with self.subTest(design=design):
                actual = cms.RECIPE_HINTS[design].historical_target_fmax_mhz
                self.assertAlmostEqual(actual, expected, places=2)

    def test_every_hint_has_rationale_and_avoid(self):
        for design, hint in cms.RECIPE_HINTS.items():
            with self.subTest(design=design):
                self.assertTrue(hint.preferred_first_action,
                                f"{design} missing preferred_first_action")
                self.assertTrue(hint.rationale,
                                f"{design} missing rationale")
                self.assertTrue(hint.avoid,
                                f"{design} missing avoid")
                self.assertGreater(hint.historical_target_fmax_mhz, 0)


if __name__ == "__main__":
    unittest.main()
