"""Tests for scheduler/dispatch.py.  Pure offline; no Vivado."""
from __future__ import annotations

import unittest
from pathlib import Path

from scheduler.dispatch import (
    KNOWN_DESIGNS,
    RECIPE_APPLICABILITY,
    DesignFingerprint,
    candidates_for,
    candidates_with_recipe,
    design_name_from_dcp,
    fingerprint_for_unknown,
    recipe_safe_for,
)


class KnownDesignTableTests(unittest.TestCase):
    def test_anchor_winners(self):
        # Small + low-spread designs win on anchor — anchor first
        for d in ("amd_mini-isp", "rosetta_optical-flow", "vexriscv_re-place"):
            self.assertEqual(candidates_for(d)[0], "anchor",
                             f"{d} should prefer anchor first")

    def test_v0_3_winners(self):
        # All other portfolio designs prefer v0_3 first
        for d in ("rosetta_spam-filter", "logicnets_jscl",
                  "rosetta_digit-recognition", "vtr_mcml",
                  "rosetta_3d-rendering", "finn_radioml", "corescore_500_mod"):
            self.assertEqual(candidates_for(d)[0], "v0_3",
                             f"{d} should prefer v0_3 first")

    def test_all_known_have_both_candidates(self):
        # Every entry should produce 2 candidates (or be a known special case)
        for d, cands in KNOWN_DESIGNS.items():
            self.assertEqual(len(set(cands)), 2,
                             f"{d}: expected 2 distinct candidates, got {cands}")
            self.assertEqual(set(cands), {"anchor", "v0_3"},
                             f"{d}: expected anchor+v0_3, got {set(cands)}")


class FingerprintTests(unittest.TestCase):
    def test_small_low_spread_picks_anchor(self):
        fp = DesignFingerprint(lut_count=3000, critical_path_spread=17)
        self.assertEqual(fingerprint_for_unknown(fp)[0], "anchor")

    def test_large_picks_v0_3(self):
        fp = DesignFingerprint(lut_count=100000, critical_path_spread=248)
        self.assertEqual(fingerprint_for_unknown(fp)[0], "v0_3")

    def test_high_spread_picks_v0_3(self):
        fp = DesignFingerprint(lut_count=2000, critical_path_spread=63)
        # vexriscv_re-place territory — fingerprint says v0_3 (anchor cluster
        # is < 5k LUT AND < 30 spread; this is small but high-spread)
        self.assertEqual(fingerprint_for_unknown(fp)[0], "v0_3")

    def test_no_fingerprint_falls_through(self):
        fp = DesignFingerprint()
        self.assertEqual(fingerprint_for_unknown(fp), ["v0_3", "anchor"])

    def test_unknown_design_no_fingerprint_uses_default(self):
        result = candidates_for("brand_new_benchmark")
        self.assertEqual(result, ["v0_3", "anchor"])

    def test_unknown_design_with_fingerprint_uses_fingerprint(self):
        fp = DesignFingerprint(lut_count=4000, critical_path_spread=14)
        result = candidates_for("brand_new_small_benchmark", fp)
        self.assertEqual(result[0], "anchor")


class DesignNameFromDCPTests(unittest.TestCase):
    def test_strips_vivado_version(self):
        self.assertEqual(
            design_name_from_dcp("fpl26_contest_benchmarks/amd_mini-isp_2025.1.dcp"),
            "amd_mini-isp",
        )

    def test_strips_vivado_version_path_variant(self):
        self.assertEqual(
            design_name_from_dcp(Path("/some/path/corescore_500_mod_2025.1.dcp")),
            "corescore_500_mod",
        )

    def test_no_version_suffix(self):
        self.assertEqual(
            design_name_from_dcp("amd_mini-isp.dcp"),
            "amd_mini-isp",
        )


class IntegrationTests(unittest.TestCase):
    def test_lookup_priority_known_over_fingerprint(self):
        """If both name and fingerprint match contradictorily, name wins
        (table is the highest-confidence source)."""
        # amd_mini-isp is anchor-first in the table.  Pass a fingerprint
        # that would otherwise vote for v0_3 — table should prevail.
        fp = DesignFingerprint(lut_count=999999, critical_path_spread=999)
        self.assertEqual(candidates_for("amd_mini-isp", fp)[0], "anchor")


class RecipeApplicabilityTests(unittest.TestCase):
    def test_harmful_designs_blocked(self):
        # corescore was confirmed HARMFUL in the 9-design sweep — recipe
        # moved cells but route_design left routing errors → score 0.
        self.assertFalse(recipe_safe_for("corescore_500_mod"))

    def test_error_designs_blocked(self):
        # digit-recognition hard-errored on AWS (40-min hang + crash).
        # Block until the root cause is understood.
        self.assertFalse(recipe_safe_for("rosetta_digit-recognition"))

    def test_applicable_designs_allowed(self):
        for d in ("amd_mini-isp", "logicnets_jscl",
                  "rosetta_3d-rendering", "vexriscv_re-place"):
            self.assertTrue(recipe_safe_for(d), f"{d} should be recipe-safe")

    def test_no_op_designs_allowed(self):
        # The recipe's no-candidates short-circuit handles these cleanly,
        # so we don't block them — they just return early.
        for d in ("finn_radioml", "rosetta_optical-flow"):
            self.assertTrue(recipe_safe_for(d), f"{d} no_op should not be blocked")

    def test_unknown_designs_allowed(self):
        # For unswept designs, lean conservative-permissive: the recipe's
        # own gates catch route errors before any DCP is written.
        self.assertTrue(recipe_safe_for("rosetta_spam-filter"))
        self.assertTrue(recipe_safe_for("brand_new_benchmark"))
        self.assertTrue(recipe_safe_for(None))

    def test_table_covers_all_known_designs(self):
        # Every name in KNOWN_DESIGNS must have a recipe applicability verdict
        # so the LLM-side hint never sees a missing entry on a known design.
        missing = set(KNOWN_DESIGNS) - set(RECIPE_APPLICABILITY)
        self.assertFalse(missing, f"missing recipe applicability for: {missing}")


class CandidatesWithRecipeTests(unittest.TestCase):
    def test_applicable_design_gets_recipe_first(self):
        # vexriscv_re-place is in the applicable list — recipe should
        # come first, then the existing 2-candidate ordering.
        out = candidates_with_recipe("vexriscv_re-place")
        self.assertEqual(out[0], "recipe")
        self.assertEqual(out[1:], candidates_for("vexriscv_re-place"))

    def test_harmful_design_no_recipe(self):
        # corescore is harmful — recipe MUST NOT be added.
        out = candidates_with_recipe("corescore_500_mod")
        self.assertNotIn("recipe", out)
        self.assertEqual(out, candidates_for("corescore_500_mod"))

    def test_no_op_design_no_recipe(self):
        # finn_radioml is no_op — adding recipe would burn budget for nothing.
        out = candidates_with_recipe("finn_radioml")
        self.assertNotIn("recipe", out)

    def test_unknown_design_no_recipe(self):
        # spam-filter is `unknown` in the table — conservative-permissive
        # for the LLM-side gate but DO NOT auto-schedule.
        self.assertNotIn("recipe", candidates_with_recipe("rosetta_spam-filter"))
        self.assertNotIn("recipe", candidates_with_recipe("brand_new_benchmark"))

    def test_error_design_no_recipe(self):
        # digit-recog hard-errored — never auto-schedule.
        self.assertNotIn("recipe", candidates_with_recipe("rosetta_digit-recognition"))


if __name__ == "__main__":
    unittest.main()
