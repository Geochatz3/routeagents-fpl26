"""Tests for optimizer.pathology classifier."""
from __future__ import annotations

import unittest

from optimizer.pathology import (
    classify_path, classify_design,
    PLACEMENT_DETOUR, ROUTE_DETOUR, LUT_DEPTH, HIGH_FANOUT_DRIVER,
    HARDBLOCK_DISTANCE, RETIMING_CANDIDATE, PIN_SWAP_CANDIDATE,
    CELL_SPREAD, ROUTE_DOMINATED_NO_SAFE_MOVE, NO_CLEAR_LOCAL_RECIPE,
    ALL_LABELS, PATHOLOGY_TO_RECIPES,
)


class SinglePathClassifierTests(unittest.TestCase):
    def test_placement_detour_fires(self):
        r = classify_path({"avg_detour_ratio": 2.5})
        labels = [p.label for p in r]
        self.assertIn(PLACEMENT_DETOUR, labels)

    def test_placement_detour_high_confidence_at_very_high(self):
        r = classify_path({"avg_detour_ratio": 3.5})
        p = next(p for p in r if p.label == PLACEMENT_DETOUR)
        self.assertGreater(p.confidence, 0.80)

    def test_lut_depth_fires(self):
        r = classify_path({"logic_levels": 8})
        labels = [p.label for p in r]
        self.assertIn(LUT_DEPTH, labels)

    def test_high_fanout_driver_fires(self):
        r = classify_path({"high_fanout_nets": [("net_a", 200)]})
        labels = [p.label for p in r]
        self.assertIn(HIGH_FANOUT_DRIVER, labels)

    def test_high_fanout_driver_skipped_below_threshold(self):
        r = classify_path({"high_fanout_nets": [("net_a", 50)]})
        labels = [p.label for p in r]
        self.assertNotIn(HIGH_FANOUT_DRIVER, labels)

    def test_route_detour_distinguished_from_placement_detour(self):
        # max >> avg → ROUTE_DETOUR (one bad net), not PLACEMENT_DETOUR.
        r = classify_path({"avg_detour_ratio": 1.2, "max_detour_ratio": 4.0})
        labels = [p.label for p in r]
        self.assertIn(ROUTE_DETOUR, labels)
        self.assertNotIn(PLACEMENT_DETOUR, labels)

    def test_hardblock_distance_fires_on_dsp_endpoint(self):
        r = classify_path({"dsp_or_bram_endpoint": True, "lut_count": 5})
        labels = [p.label for p in r]
        self.assertIn(HARDBLOCK_DISTANCE, labels)

    def test_retiming_candidate_fires_on_balanced_ff_path(self):
        # 4 FFs, slack -1.0 ns → 0.25 ns/FF < 0.5 threshold.
        r = classify_path({"ff_count": 4, "slack_ns": -1.0})
        labels = [p.label for p in r]
        self.assertIn(RETIMING_CANDIDATE, labels)

    def test_retiming_skipped_on_few_ffs(self):
        r = classify_path({"ff_count": 1, "slack_ns": -0.5})
        labels = [p.label for p in r]
        self.assertNotIn(RETIMING_CANDIDATE, labels)

    def test_cell_spread_fires(self):
        r = classify_path({"cell_spread": 80.0})
        labels = [p.label for p in r]
        self.assertIn(CELL_SPREAD, labels)

    def test_pin_swap_candidate_explicit(self):
        r = classify_path({"has_pin_swap_opportunity": True})
        labels = [p.label for p in r]
        self.assertIn(PIN_SWAP_CANDIDATE, labels)

    def test_route_dominated_residual_only_when_nothing_else_fires(self):
        # Route-bound path with no other pathology.
        r = classify_path({
            "route_delay_fraction": 0.85,
            "logic_levels": 2,  # below LUT_DEPTH threshold
        })
        labels = [p.label for p in r]
        self.assertIn(ROUTE_DOMINATED_NO_SAFE_MOVE, labels)

    def test_route_dominated_suppressed_when_other_pathology_fires(self):
        # High route-delay AND high logic levels → LUT_DEPTH wins; we
        # don't add the residual "no safe move" label.
        r = classify_path({
            "route_delay_fraction": 0.85,
            "logic_levels": 10,
        })
        labels = [p.label for p in r]
        self.assertIn(LUT_DEPTH, labels)
        self.assertNotIn(ROUTE_DOMINATED_NO_SAFE_MOVE, labels)

    def test_no_clear_recipe_when_empty(self):
        r = classify_path({})
        labels = [p.label for p in r]
        self.assertEqual(labels, [NO_CLEAR_LOCAL_RECIPE])

    def test_multiple_pathologies_can_coexist(self):
        # LUT_DEPTH + HIGH_FANOUT_DRIVER + PLACEMENT_DETOUR all fire.
        r = classify_path({
            "avg_detour_ratio": 2.3,
            "logic_levels": 8,
            "high_fanout_nets": [("clk_en", 600)],
        })
        labels = {p.label for p in r}
        self.assertIn(LUT_DEPTH, labels)
        self.assertIn(HIGH_FANOUT_DRIVER, labels)
        self.assertIn(PLACEMENT_DETOUR, labels)

    def test_every_label_has_recipe_mapping(self):
        # Sanity contract: each label maps to a (possibly empty) tuple.
        for label in ALL_LABELS:
            self.assertIn(label, PATHOLOGY_TO_RECIPES,
                          f"{label} missing from PATHOLOGY_TO_RECIPES")

    def test_suggested_recipes_include_known_recipe(self):
        # PLACEMENT_DETOUR should always recommend recipe_cell_replacement
        # (the contest-published recipe for this exact pathology).
        r = classify_path({"avg_detour_ratio": 3.0})
        p = next(p for p in r if p.label == PLACEMENT_DETOUR)
        self.assertIn("recipe_cell_replacement", p.suggested_recipes)


class DesignClassifierTests(unittest.TestCase):
    def test_primary_label_is_most_frequent_nontrivial(self):
        paths = [
            {"logic_levels": 8},
            {"logic_levels": 9},
            {"logic_levels": 10},
            {"avg_detour_ratio": 2.5},
        ]
        d = classify_design(paths)
        self.assertEqual(d.primary_label, LUT_DEPTH)
        self.assertEqual(d.counts.get(LUT_DEPTH), 3)

    def test_no_clear_recipe_excluded_from_primary_when_others_fire(self):
        paths = [
            {},  # NO_CLEAR_LOCAL_RECIPE
            {},
            {},
            {"avg_detour_ratio": 2.3},  # PLACEMENT_DETOUR
        ]
        d = classify_design(paths)
        # PLACEMENT_DETOUR fires once, NO_CLEAR fires three times,
        # but the primary should be PLACEMENT_DETOUR (nontrivial wins).
        self.assertEqual(d.primary_label, PLACEMENT_DETOUR)

    def test_no_clear_recipe_when_all_paths_empty(self):
        d = classify_design([{}, {}, {}])
        self.assertEqual(d.primary_label, NO_CLEAR_LOCAL_RECIPE)

    def test_design_notes_flag_global_high_fanout(self):
        d = classify_design(
            [{"high_fanout_nets": [("path_net", 50)]}],
            design_context={"high_fanout_nets": [("big_net", 1500)]},
        )
        self.assertTrue(any("big_net" in n for n in d.notes))

    def test_design_notes_flag_global_spread(self):
        d = classify_design(
            [{}],
            design_context={"critical_path_spread_info": {"avg_distance": 120.0}},
        )
        self.assertTrue(any("avg spread" in n for n in d.notes))

    def test_suggested_recipes_ordered_by_primary_then_others(self):
        paths = [{"avg_detour_ratio": 2.5}]  # PLACEMENT_DETOUR primary
        d = classify_design(paths)
        # First recipes should be PLACEMENT_DETOUR's preferred ones.
        self.assertEqual(d.suggested_recipes_ordered[0], "recipe_cell_replacement")
        # And the list deduplicates (no recipe appears twice).
        self.assertEqual(len(d.suggested_recipes_ordered),
                          len(set(d.suggested_recipes_ordered)))


if __name__ == "__main__":
    unittest.main()
