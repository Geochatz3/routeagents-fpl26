"""Tests for the offline pathology classifier prototype.

The classifier is REPORT-ONLY today; tests pin behaviour so we can wire
it into retrieval safely later.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizer.pathology_features import (
    LABEL_CONGESTION_LOW,
    LABEL_GLOBAL_WNS_BOUND_LIKELY,
    LABEL_INSUFFICIENT_DATA,
    LABEL_NO_CONGESTION_SIGNAL,
    LABEL_ROUTE_BOUND_LIKELY,
    LABEL_ROUTE_BOUND_MILD,
    classify_pathology,
)


class PathologyClassifierTests(unittest.TestCase):

    def test_spam_filter_like_classified_route_bound(self):
        # rosetta_spam-filter probe: route_bound_score ≈ 0.5714,
        # has_congestion_signal True, initial WNS small.
        labels = classify_pathology({
            "qor_route_bound_score": 0.5714,
            "qor_has_congestion_signal": True,
            "initial_wns": -0.686,
        })
        self.assertIn(LABEL_ROUTE_BOUND_LIKELY, labels)
        self.assertNotIn(LABEL_GLOBAL_WNS_BOUND_LIKELY, labels)

    def test_boom_soc_like_no_congestion_signal(self):
        # boom_soc probe: route_bound_score 0.0 with all global = None and
        # huge initial WNS.  Should flag WNS-bound + no congestion signal.
        labels = classify_pathology({
            "qor_route_bound_score": 0.0,
            "qor_has_congestion_signal": True,
            "initial_wns": -19.162,
        })
        # Score 0 → congestion_low.  WNS very negative → WNS bound.
        self.assertIn(LABEL_CONGESTION_LOW, labels)
        self.assertIn(LABEL_GLOBAL_WNS_BOUND_LIKELY, labels)

    def test_no_signal_with_wns_bound(self):
        labels = classify_pathology({
            "qor_route_bound_score": None,
            "qor_has_congestion_signal": False,
            "initial_wns": -10.0,
        })
        self.assertIn(LABEL_NO_CONGESTION_SIGNAL, labels)
        self.assertIn(LABEL_GLOBAL_WNS_BOUND_LIKELY, labels)

    def test_no_signal_no_wns(self):
        labels = classify_pathology({
            "qor_route_bound_score": None,
            "qor_has_congestion_signal": False,
            "initial_wns": -0.5,
        })
        self.assertIn(LABEL_NO_CONGESTION_SIGNAL, labels)
        self.assertIn(LABEL_CONGESTION_LOW, labels)
        self.assertNotIn(LABEL_GLOBAL_WNS_BOUND_LIKELY, labels)

    def test_missing_qor_returns_insufficient_data(self):
        labels = classify_pathology({})
        self.assertEqual(labels, [LABEL_INSUFFICIENT_DATA])

    def test_missing_qor_but_huge_wns_still_emits_wns_label(self):
        labels = classify_pathology({"initial_wns": -10.0})
        self.assertIn(LABEL_INSUFFICIENT_DATA, labels)
        self.assertIn(LABEL_GLOBAL_WNS_BOUND_LIKELY, labels)

    def test_no_design_name_dependency_in_input(self):
        """Even if a design_name key sneaks in, classifier refuses to use it."""
        labels = classify_pathology({
            "design_name": "rosetta_spam-filter",
            "qor_route_bound_score": 0.7,
            "qor_has_congestion_signal": True,
        })
        # The presence of design_name forces INSUFFICIENT_DATA — safer to
        # refuse than to silently classify with a leak in scope.
        self.assertEqual(labels, [LABEL_INSUFFICIENT_DATA])

    def test_mild_band(self):
        labels = classify_pathology({
            "qor_route_bound_score": 0.3,
            "qor_has_congestion_signal": True,
            "initial_wns": -1.0,
        })
        self.assertIn(LABEL_ROUTE_BOUND_MILD, labels)
        self.assertNotIn(LABEL_ROUTE_BOUND_LIKELY, labels)
        self.assertNotIn(LABEL_CONGESTION_LOW, labels)

    def test_boundary_above_strong(self):
        labels = classify_pathology({
            "qor_route_bound_score": 0.50,
            "qor_has_congestion_signal": True,
        })
        self.assertIn(LABEL_ROUTE_BOUND_LIKELY, labels)

    def test_boundary_at_mild(self):
        labels = classify_pathology({
            "qor_route_bound_score": 0.20,
            "qor_has_congestion_signal": True,
        })
        self.assertIn(LABEL_ROUTE_BOUND_MILD, labels)

    def test_boundary_below_mild(self):
        labels = classify_pathology({
            "qor_route_bound_score": 0.19,
            "qor_has_congestion_signal": True,
        })
        self.assertIn(LABEL_CONGESTION_LOW, labels)

    def test_non_mapping_input(self):
        self.assertEqual(classify_pathology(None), [LABEL_INSUFFICIENT_DATA])  # type: ignore
        self.assertEqual(classify_pathology([1, 2, 3]), [LABEL_INSUFFICIENT_DATA])  # type: ignore

    def test_deterministic(self):
        x = {
            "qor_route_bound_score": 0.57,
            "qor_has_congestion_signal": True,
            "initial_wns": -0.7,
        }
        self.assertEqual(classify_pathology(x), classify_pathology(x))

    def test_no_recipe_or_tool_label_leaks(self):
        """Pathology labels must not look like recipe names or tool flags.

        They should be strictly axis labels, not action labels.
        """
        labels = classify_pathology({
            "qor_route_bound_score": 0.6,
            "qor_has_congestion_signal": True,
            "initial_wns": -7.0,
        })
        for lbl in labels:
            for forbidden_token in (
                "phys_opt", "place_design", "route_design",
                "phase2", "pblock", "directive",
            ):
                self.assertNotIn(forbidden_token, lbl,
                                 f"label {lbl!r} contains recipe-like token")


if __name__ == "__main__":
    unittest.main()
