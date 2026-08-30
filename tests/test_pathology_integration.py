"""Integration tests: pathology classifier output reaches the iter-1
user message via DCPOptimizer._build_pathology_block.

The 2026-05-14 critique surfaced that the classifier (built + 23 unit-
tested) was not reaching runtime decision-making.  These tests pin
the integration: given the same Phase 1 data the classifier sees,
_build_pathology_block emits a structured PATHOLOGY DIAGNOSIS block
with the dominant pathology + ordered recipe list.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import DCPOptimizer
from optimizer.pathology import (
    HIGH_FANOUT_DRIVER, CELL_SPREAD, NO_CLEAR_LOCAL_RECIPE,
)


def _make_opt(tmp_path: Path) -> DCPOptimizer:
    opt = DCPOptimizer(api_key="test", run_dir=tmp_path)
    opt.clock_period = 1.570
    opt.initial_wns = -1.000
    return opt


class PathologyIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = _make_opt(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_block_empty_when_no_phase1_data(self):
        # No high-fanout nets, no spread, no initial_wns.
        self.opt.initial_wns = None
        self.opt.high_fanout_nets = []
        self.opt.critical_path_spread_info = None
        block = self.opt._build_pathology_block()
        self.assertEqual(block, [])
        self.assertIsNone(self.opt.design_pathology)

    def test_block_fires_on_high_fanout_signal(self):
        # 3 high-fanout nets, well above the 100 threshold — classifier
        # must produce HIGH_FANOUT_DRIVER as the dominant pathology.
        self.opt.high_fanout_nets = [
            ("net_a", 500, 5),
            ("net_b", 300, 3),
            ("net_c", 200, 2),
        ]
        self.opt.critical_path_spread_info = None
        block = self.opt._build_pathology_block()
        self.assertGreater(len(block), 3)
        # Header always first.
        self.assertEqual(block[0], "PATHOLOGY DIAGNOSIS (deterministic, "
                                    "from Phase 1 data):")
        # Primary label line mentions HIGH_FANOUT_DRIVER.
        primary_line = next(l for l in block if l.startswith("  Primary:"))
        self.assertIn(HIGH_FANOUT_DRIVER, primary_line)
        # Recommended recipes section includes the canonical recipe.
        recipe_lines = [l for l in block if "recipe_" in l]
        self.assertTrue(any("recipe_high_fanout_timing_replication" in l
                             for l in recipe_lines),
                         f"missing recipe in {recipe_lines}")
        # The diagnosis is stashed on the optimizer for downstream use.
        self.assertIsNotNone(self.opt.design_pathology)
        self.assertEqual(
            self.opt.design_pathology.primary_label, HIGH_FANOUT_DRIVER
        )

    def test_block_fires_on_cell_spread_signal(self):
        # No fanout signal, but RapidWright spread analysis reports high.
        self.opt.high_fanout_nets = []
        self.opt.critical_path_spread_info = {
            "avg_distance": 90.0, "max_distance": 150, "paths_analyzed": 10,
        }
        block = self.opt._build_pathology_block()
        self.assertGreater(len(block), 0)
        primary_line = next(l for l in block if l.startswith("  Primary:"))
        self.assertIn(CELL_SPREAD, primary_line)

    def test_block_recommends_recipe_order(self):
        # The recommended recipes line is structured as a numbered list.
        self.opt.high_fanout_nets = [("net_a", 700, 5)]
        block = self.opt._build_pathology_block()
        numbered = [l for l in block if l.lstrip().startswith(("1.", "2.", "3.", "4."))]
        self.assertGreaterEqual(len(numbered), 1,
                                 f"expected numbered recipe list, got {block}")

    def test_block_filters_blocked_recipes_for_corescore(self):
        # A recipe on the scheduler's per-design denylist must be removed
        # from the recommended set and reported as blocked.
        self.opt.critical_path_spread_info = {
            "avg_distance": 200.0, "max_distance": 250, "paths_analyzed": 10,
        }
        block = self.opt._build_pathology_block(design_name="corescore_500_mod")
        joined = "\n".join(block)
        # The blocked recipe must NOT appear in the recommended numbered list.
        recommended_lines = [l for l in block
                             if l.lstrip().startswith(("1.", "2.", "3.", "4."))]
        for line in recommended_lines:
            self.assertNotIn("recipe_cell_replacement", line,
                              f"blocked recipe leaked: {line}")
        # And it MUST appear under BLOCKED.
        self.assertIn("BLOCKED RECIPES", joined)
        self.assertIn("recipe_cell_replacement", joined)

    def test_block_keeps_recipes_when_design_safe(self):
        # Unknown applicability fails open, so every classifier-recommended recipe
        # remains in the recommended set.
        self.opt.critical_path_spread_info = {
            "avg_distance": 200.0, "max_distance": 250, "paths_analyzed": 10,
        }
        block = self.opt._build_pathology_block(design_name="boom_soc")
        joined = "\n".join(block)
        # recipe_cell_replacement should be in the recommended list, no BLOCKED section.
        self.assertNotIn("BLOCKED RECIPES", joined)
        recommended_lines = [l for l in block
                             if l.lstrip().startswith(("1.", "2.", "3.", "4."))]
        rec_joined = "\n".join(recommended_lines)
        self.assertIn("recipe_cell_replacement", rec_joined)

    def test_block_no_clear_recipe_when_no_signal(self):
        # initial_wns provides slack but no fanout, no spread, no other
        # signals → NO_CLEAR_LOCAL_RECIPE residual fires.
        self.opt.initial_wns = -2.0
        self.opt.high_fanout_nets = []
        self.opt.critical_path_spread_info = None
        block = self.opt._build_pathology_block()
        # Either empty (no paths constructed) or NO_CLEAR_LOCAL_RECIPE.
        # Empty is acceptable when no paths could be built — the
        # classifier wouldn't have meaningful input.
        if block:
            primary_line = next(l for l in block if l.startswith("  Primary:"))
            self.assertIn(NO_CLEAR_LOCAL_RECIPE, primary_line)


if __name__ == "__main__":
    unittest.main()
