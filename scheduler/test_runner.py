"""Tests for scheduler/runner.py — log parsing + extra_args dispatch.

Pure offline; doesn't spawn subprocesses.  We just verify the parser
extracts headline metrics from a representative recipe stdout dump and
that _extra_args_for routes correctly per candidate name.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scheduler.runner import (
    _extra_args_for,
    parse_recipe_log,
    parse_run_log,
)


class ExtraArgsTests(unittest.TestCase):
    def test_anchor(self):
        self.assertEqual(_extra_args_for("anchor"), ["--mode", "anchor"])

    def test_v0_3(self):
        self.assertEqual(_extra_args_for("v0_3"), ["--mode", "v0_3"])

    def test_v0_3_seed_falls_through_to_v0_3(self):
        self.assertEqual(_extra_args_for("v0_3_seed2"), ["--mode", "v0_3"])

    def test_recipe_no_args(self):
        # recipe binary has its own argparse defaults — runner shouldn't
        # inject any --mode flag.
        self.assertEqual(_extra_args_for("recipe"), [])


class ParseRecipeLogTests(unittest.TestCase):
    def test_extracts_headline_metrics_from_recipe_json(self):
        # recipes/cell_replacement.py prints a JSON blob via json.dumps
        # (default=str) at the end.  We need to recover delta_fmax_mhz +
        # final_fmax_mhz at minimum.
        recipe_blob = {
            "status": "success",
            "input_dcp": "/in.dcp",
            "output_dcp": "/out.dcp",
            "baseline_wns_ns": -0.946,
            "baseline_fmax_mhz": 310.17,
            "final_wns_ns": -0.683,
            "final_fmax_mhz": 386.84,
            "delta_fmax_mhz": 76.67,
            "candidates_total": 5,
            "candidates_targeted": ["base/lut2"],
            "cells_moved": ["base/lut2"],
            "cells_unmoved": [],
            "route_errors": 0,
            "wall_time_s": 306.0,
            "error": None,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
            fh.write("[INFO] Starting recipe ...\n")
            fh.write("Some progress output\n\n")
            fh.write(json.dumps(recipe_blob, indent=2))
            fh.write("\n")
            log_path = Path(fh.name)

        out = parse_recipe_log(log_path)
        self.assertEqual(out["initial_fmax_mhz"], 310.17)
        self.assertEqual(out["final_fmax_mhz"], 386.84)
        self.assertEqual(out["delta_fmax_mhz"], 76.67)
        self.assertEqual(out["iterations"], 1)
        self.assertEqual(out["cost_usd"], 0.0)
        log_path.unlink()

    def test_handles_no_candidates_blob(self):
        recipe_blob = {
            "status": "no_candidates",
            "candidates_total": 0,
            "candidates_targeted": [],
            "cells_moved": [],
            "cells_unmoved": [],
            "delta_fmax_mhz": None,
            "final_fmax_mhz": None,
            "baseline_fmax_mhz": None,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
            fh.write(json.dumps(recipe_blob, indent=2))
            log_path = Path(fh.name)
        out = parse_recipe_log(log_path)
        self.assertIsNone(out["delta_fmax_mhz"])
        self.assertIsNone(out["final_fmax_mhz"])
        # iterations + cost still fill in — recipe ran (one attempt, $0)
        self.assertEqual(out["iterations"], 1)
        self.assertEqual(out["cost_usd"], 0.0)
        log_path.unlink()

    def test_missing_log_returns_all_none(self):
        out = parse_recipe_log(Path("/nonexistent/recipe.log"))
        self.assertIsNone(out["delta_fmax_mhz"])
        self.assertIsNone(out["final_fmax_mhz"])

    def test_malformed_log_returns_all_none(self):
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
            fh.write("not json at all, no braces here\n")
            log_path = Path(fh.name)
        out = parse_recipe_log(log_path)
        self.assertIsNone(out["delta_fmax_mhz"])
        log_path.unlink()


class ParseRunLogTests(unittest.TestCase):
    """Sanity check the existing dcp_optimizer log parser still works."""
    def test_extracts_dcp_optimizer_summary(self):
        log = """
        ... lots of optimizer output ...
        Initial Fmax: 310.17 MHz
        Best Fmax: 386.84 MHz
        Fmax Improvement: +76.67 MHz
        Total iterations: 5
        Total cost: $0.1130
        """
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
            fh.write(log)
            log_path = Path(fh.name)
        out = parse_run_log(log_path)
        self.assertEqual(out["initial_fmax_mhz"], 310.17)
        self.assertEqual(out["final_fmax_mhz"], 386.84)
        self.assertEqual(out["delta_fmax_mhz"], 76.67)
        self.assertEqual(out["iterations"], 5)
        self.assertEqual(out["cost_usd"], 0.113)
        log_path.unlink()


if __name__ == "__main__":
    unittest.main()
