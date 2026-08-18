"""Pure-Python tests for recipes/cell_replacement.py — exercise the
candidate filtering and result-dict shape without spawning Vivado or
RapidWright.  Run with `python3 -m recipes.test_cell_replacement` from
the repo root."""
from __future__ import annotations

import unittest

from recipes.cell_replacement import _select_target_cells


class SelectTargetCellsTests(unittest.TestCase):
    def test_single_path(self):
        cands = [
            {"cell": "design/lut1", "path": 1, "max_detour_ratio": 3.5},
            {"cell": "design/lut2", "path": 1, "max_detour_ratio": 2.1},
        ]
        result = _select_target_cells(cands, max_path=2)
        self.assertEqual(set(result), {"design/lut1", "design/lut2"})

    def test_dedupes_repeated_cells(self):
        cands = [
            {"cell": "design/lut1", "path": 1, "max_detour_ratio": 3.5},
            {"cell": "design/lut1", "path": 2, "max_detour_ratio": 4.0},
        ]
        result = _select_target_cells(cands, max_path=2)
        self.assertEqual(result, ["design/lut1"])

    def test_filters_to_worst_N_paths(self):
        cands = [
            {"cell": "design/lut1", "path": 1, "max_detour_ratio": 3.5},
            {"cell": "design/lut2", "path": 5, "max_detour_ratio": 4.0},
            {"cell": "design/lut3", "path": 2, "max_detour_ratio": 2.5},
        ]
        result = _select_target_cells(cands, max_path=2)
        self.assertEqual(set(result), {"design/lut1", "design/lut3"})

    def test_skips_missing_path(self):
        cands = [
            {"cell": "design/lut1", "max_detour_ratio": 3.5},  # no path
            {"cell": "design/lut2", "path": 1, "max_detour_ratio": 4.0},
        ]
        result = _select_target_cells(cands, max_path=2)
        self.assertEqual(result, ["design/lut2"])

    def test_empty_candidates(self):
        self.assertEqual(_select_target_cells([], max_path=2), [])

    def test_all_filtered_out_by_max_path(self):
        cands = [
            {"cell": "design/lut1", "path": 5, "max_detour_ratio": 3.5},
            {"cell": "design/lut2", "path": 9, "max_detour_ratio": 4.0},
        ]
        result = _select_target_cells(cands, max_path=2)
        self.assertEqual(result, [])

    def test_max_path_default_inclusive(self):
        cands = [
            {"cell": "design/lut1", "path": 2, "max_detour_ratio": 3.5},
        ]
        result = _select_target_cells(cands, max_path=2)
        self.assertEqual(result, ["design/lut1"])  # path == max is included

    def test_preserves_input_order(self):
        # analyze_net_detour returns candidates sorted by detour ratio
        # descending — the recipe must preserve that order so the
        # highest-detour cell is tried first.
        cands = [
            {"cell": "design/high_detour_lut",   "path": 1, "max_detour_ratio": 5.5},
            {"cell": "design/medium_detour_lut", "path": 1, "max_detour_ratio": 3.2},
            {"cell": "design/low_detour_lut",    "path": 2, "max_detour_ratio": 2.1},
        ]
        result = _select_target_cells(cands, max_path=2)
        self.assertEqual(result, [
            "design/high_detour_lut",
            "design/medium_detour_lut",
            "design/low_detour_lut",
        ])

    def test_dedup_keeps_first_occurrence(self):
        # If a cell appears on multiple paths, keep the FIRST occurrence
        # (which has the highest detour ratio in the input ordering).
        cands = [
            {"cell": "design/lut_X", "path": 1, "max_detour_ratio": 5.0},  # keep
            {"cell": "design/lut_Y", "path": 1, "max_detour_ratio": 4.0},
            {"cell": "design/lut_X", "path": 2, "max_detour_ratio": 3.5},  # drop
        ]
        result = _select_target_cells(cands, max_path=2)
        self.assertEqual(result, ["design/lut_X", "design/lut_Y"])


if __name__ == "__main__":
    unittest.main()
