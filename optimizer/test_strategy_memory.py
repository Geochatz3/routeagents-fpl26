"""Tests for optimizer.strategy_memory.  Pure offline; no Vivado.

Uses tempfile JSONL fixtures to avoid touching the real campaign data
or the developer's local memory file.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from optimizer.strategy_memory import (
    DEFAULT_MEMORY_BASENAME,
    RunRecord,
    append_run,
    fingerprint_match,
    format_for_prompt,
    global_aggregate,
    load_memory,
    seed_prompt_for,
    winning_tools_from_call_details,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


class RunRecordTests(unittest.TestCase):
    def test_from_dict_ignores_extras(self):
        r = RunRecord.from_dict({"design": "x", "delta_fmax_mhz": 10.0, "extra": "ignored"})
        self.assertEqual(r.design, "x")
        self.assertEqual(r.delta_fmax_mhz, 10.0)

    def test_to_dict_drops_none(self):
        r = RunRecord(design="x", delta_fmax_mhz=10.0)
        d = r.to_dict()
        self.assertEqual(d, {"design": "x", "delta_fmax_mhz": 10.0})


class LoadMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "mem.jsonl"
        _write_jsonl(self.path, [
            {"design": "amd_mini-isp", "candidate": "anchor",
             "delta_fmax_mhz": 87.35, "iterations": 5, "completed": True},
            {"design": "amd_mini-isp", "candidate": "v0_3",
             "delta_fmax_mhz": 68.25, "iterations": 6, "completed": True},
            {"design": "logicnets_jscl", "candidate": "v0_3",
             "delta_fmax_mhz": 69.04, "force_continues": 1, "completed": True},
            {"malformed": "row"},  # ignored — no usable fields
        ])
        # Empty out the env so default discovery doesn't pick up dev's local memory
        self._patcher = mock.patch.dict(os.environ, {"STRATEGY_MEMORY_PATH": str(self.path)},
                                        clear=False)
        self._patcher.start()
        # Force cwd to a clean tmp so cwd/strategy_memory.jsonl is not present
        self._cwd_patcher = mock.patch("optimizer.strategy_memory.Path.cwd",
                                       return_value=Path(self.tmp.name))
        self._cwd_patcher.start()

    def tearDown(self):
        self._cwd_patcher.stop()
        self._patcher.stop()
        self.tmp.cleanup()

    def test_load_picks_up_env_path(self):
        records = load_memory()
        # The malformed row is permissively kept (RunRecord with all-None
        # fields); downstream consumers filter on usable fields.
        self.assertGreaterEqual(len(records), 3)


class FingerprintMatchTests(unittest.TestCase):
    def test_returns_closest(self):
        records = [
            RunRecord(design="small_a", lut_count=2000,
                      critical_path_spread=20, delta_fmax_mhz=50, completed=True),
            RunRecord(design="big_b", lut_count=100000,
                      critical_path_spread=200, delta_fmax_mhz=80, completed=True),
        ]
        m = fingerprint_match(records, lut_count=2500, spread=22)
        self.assertEqual(m.design, "small_a")

    def test_no_fingerprint_returns_none(self):
        records = [RunRecord(design="x", delta_fmax_mhz=10, completed=True)]
        self.assertIsNone(fingerprint_match(records, None, None))

    def test_no_records_with_fingerprint_returns_none(self):
        records = [RunRecord(design="x", delta_fmax_mhz=10, completed=True)]
        self.assertIsNone(fingerprint_match(records, lut_count=1000, spread=10))


class FormatForPromptTests(unittest.TestCase):
    def test_none_returns_empty(self):
        self.assertEqual(format_for_prompt(None, "x"), "")

    def test_anchor_winner_includes_pblock_hint(self):
        r = RunRecord(design="amd_mini-isp", candidate="anchor",
                      delta_fmax_mhz=87.35, iterations=5)
        out = format_for_prompt(r, "amd_mini-isp")
        self.assertIn("anchor", out)
        self.assertIn("87.35", out)
        self.assertIn("PBLOCK-first", out)

    def test_v0_3_winner_with_force_continue(self):
        r = RunRecord(design="logicnets_jscl", candidate="v0_3",
                      delta_fmax_mhz=69.04, force_continues=1, iterations=5)
        out = format_for_prompt(r, "logicnets_jscl")
        self.assertIn("v0_3", out)
        self.assertIn("force-continue", out)

    def test_fingerprint_match_marks_as_closest(self):
        r = RunRecord(design="logicnets_jscl", candidate="v0_3", delta_fmax_mhz=69.04)
        out = format_for_prompt(r, "brand_new_design")
        self.assertIn("closest fingerprint match", out)
        self.assertIn("logicnets_jscl", out)

    def test_narrow_gap_hint(self):
        r = RunRecord(design="rosetta_3d-rendering", candidate="v0_3",
                      delta_fmax_mhz=2.0)
        out = format_for_prompt(r, "rosetta_3d-rendering")
        self.assertIn("narrow", out)

    def test_short_run_hint(self):
        r = RunRecord(design="x", candidate="v0_3", delta_fmax_mhz=80, iterations=2)
        out = format_for_prompt(r, "x")
        self.assertIn("over-iterate", out)


class AppendRunTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mem.jsonl"
            r = RunRecord(design="x", candidate="v0_3", delta_fmax_mhz=10.0,
                          completed=True)
            written = append_run(r, path=path)
            self.assertEqual(written, path)
            # Read back and verify
            records = list(path.read_text().splitlines())
            self.assertEqual(len(records), 1)
            parsed = RunRecord.from_dict(json.loads(records[0]))
            self.assertEqual(parsed.design, "x")

    def test_append_missing_directory_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deep" / "nested" / "mem.jsonl"
            r = RunRecord(design="x", delta_fmax_mhz=10)
            self.assertEqual(append_run(r, path=path), path)
            self.assertTrue(path.exists())


class SeedPromptForTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "mem.jsonl"
        _write_jsonl(self.path, [
            {"design": "amd_mini-isp", "candidate": "anchor",
             "delta_fmax_mhz": 87.35, "completed": True,
             "lut_count": 24_000, "critical_path_spread": 42.0},
        ])
        self._env = mock.patch.dict(os.environ,
                                    {"STRATEGY_MEMORY_PATH": str(self.path)},
                                    clear=False)
        self._env.start()
        self._cwd = mock.patch("optimizer.strategy_memory.Path.cwd",
                               return_value=Path(self.tmp.name))
        self._cwd.start()

    def tearDown(self):
        self._cwd.stop()
        self._env.stop()
        self.tmp.cleanup()

    def test_fingerprint_query_returns_matched_snippet(self):
        snippet = seed_prompt_for(lut_count=24_500,
                                  critical_path_spread=41.0)
        self.assertIn("anchor", snippet)
        self.assertIn("87.35", snippet)

    def test_design_name_is_ignored_for_retrieval(self):
        # Retrieval is fingerprint-only: passing the exact stored design
        # name without a fingerprint must NOT retrieve that record.
        snippet = seed_prompt_for("amd_mini-isp")
        self.assertIn("PRIOR-CAMPAIGN HISTORY:", snippet)
        self.assertIn("prior campaigns", snippet)  # global aggregate

    def test_no_fingerprint_falls_back_to_global_aggregate(self):
        # No lut/spread given → fingerprint match returns None → fall
        # back to global_aggregate so the LLM still sees *some* prior.
        snippet = seed_prompt_for()
        self.assertIn("PRIOR-CAMPAIGN HISTORY:", snippet)
        self.assertIn("prior campaigns", snippet)


class GlobalAggregateTests(unittest.TestCase):
    def test_empty_memory_returns_empty(self):
        self.assertEqual(global_aggregate([]), "")

    def test_counts_per_design_winners(self):
        memory = [
            RunRecord(design="d1", candidate="anchor", delta_fmax_mhz=80, completed=True),
            RunRecord(design="d1", candidate="v0_3", delta_fmax_mhz=70, completed=True),
            RunRecord(design="d2", candidate="v0_3", delta_fmax_mhz=50, completed=True),
            RunRecord(design="d3", candidate="v0_3_seed2", delta_fmax_mhz=40, completed=True),
        ]
        out = global_aggregate(memory)
        self.assertIn("3 prior campaigns", out)
        self.assertIn("v0_3 won 2×", out)  # d2 + d3 (seed dedupes to v0_3)
        self.assertIn("anchor won 1×", out)  # d1
        self.assertIn("global mean ΔFmax", out)

    def test_excludes_completed_false(self):
        memory = [
            RunRecord(design="d1", candidate="anchor", delta_fmax_mhz=80, completed=True),
            RunRecord(design="d2", candidate="anchor", delta_fmax_mhz=10, completed=False),
        ]
        out = global_aggregate(memory)
        self.assertIn("1 prior campaigns", out)


class SeedPromptFallbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "mem.jsonl"
        _write_jsonl(self.path, [
            {"design": "known_a", "candidate": "anchor",
             "delta_fmax_mhz": 50.0, "completed": True,
             "lut_count": 5_000, "critical_path_spread": 12.0},
            {"design": "known_b", "candidate": "v0_3",
             "delta_fmax_mhz": 70.0, "completed": True},
        ])
        self._env = mock.patch.dict(os.environ,
                                    {"STRATEGY_MEMORY_PATH": str(self.path)},
                                    clear=False)
        self._env.start()
        self._cwd = mock.patch("optimizer.strategy_memory.Path.cwd",
                               return_value=Path(self.tmp.name))
        self._cwd.start()

    def tearDown(self):
        self._cwd.stop()
        self._env.stop()
        self.tmp.cleanup()

    def test_unknown_fingerprint_falls_back_to_global_aggregate(self):
        snippet = seed_prompt_for("brand_new")
        self.assertIn("PRIOR-CAMPAIGN HISTORY:", snippet)
        self.assertIn("prior campaigns", snippet)

    def test_fingerprint_uses_match_not_global(self):
        snippet = seed_prompt_for(lut_count=5_200,
                                  critical_path_spread=13.0)
        self.assertIn("anchor", snippet)
        # Should NOT show the "Across N prior campaigns" aggregate header
        self.assertNotIn("Across", snippet)


class WinningToolsTests(unittest.TestCase):
    def test_empty_returns_empty(self):
        self.assertEqual(winning_tools_from_call_details([], -0.5), [])
        self.assertEqual(winning_tools_from_call_details([], None), [])

    def test_attributes_improvement_to_prior_transformative_tool(self):
        tcd = [
            {"tool_name": "vivado_open_checkpoint", "wns": None},
            {"tool_name": "vivado_report_timing_summary", "wns": -0.978},
            {"tool_name": "vivado_route_design", "wns": None},
            {"tool_name": "vivado_report_timing_summary", "wns": -0.450},  # better
        ]
        out = winning_tools_from_call_details(tcd, -0.978)
        self.assertEqual(out, ["vivado_route_design"])

    def test_skips_regressions(self):
        tcd = [
            {"tool_name": "vivado_route_design", "wns": None},
            {"tool_name": "vivado_report_timing_summary", "wns": -0.450},
            {"tool_name": "vivado_phys_opt_design", "wns": None},
            {"tool_name": "vivado_report_timing_summary", "wns": -0.500},  # worse → skip
        ]
        out = winning_tools_from_call_details(tcd, -0.978)
        self.assertEqual(out, ["vivado_route_design"])

    def test_skips_errored_calls(self):
        tcd = [
            {"tool_name": "vivado_create_and_apply_pblock", "wns": None, "error": True},
            {"tool_name": "vivado_route_design", "wns": None},
            {"tool_name": "vivado_report_timing_summary", "wns": -0.500},
        ]
        out = winning_tools_from_call_details(tcd, -0.978)
        # The errored pblock call must NOT be attributed; route_design wins.
        self.assertEqual(out, ["vivado_route_design"])

    def test_dedups_adjacent_duplicates(self):
        tcd = [
            {"tool_name": "vivado_route_design", "wns": None},
            {"tool_name": "vivado_report_timing_summary", "wns": -0.5},
            {"tool_name": "vivado_route_design", "wns": None},
            {"tool_name": "vivado_report_timing_summary", "wns": -0.4},  # better
        ]
        out = winning_tools_from_call_details(tcd, -0.978)
        # Adjacent route_design duplicates collapse to one entry.
        self.assertEqual(out, ["vivado_route_design"])

    def test_caps_at_12_entries(self):
        tcd = []
        for i in range(20):
            tcd.append({"tool_name": f"transform_{i}", "wns": None})
            tcd.append({"tool_name": "vivado_report_timing_summary",
                        "wns": -0.978 + 0.01 * (i + 1)})
        out = winning_tools_from_call_details(tcd, -0.978)
        self.assertLessEqual(len(out), 12)

    def test_recipe_call_in_chain(self):
        # Recipe call shows up as the transformative tool
        tcd = [
            {"tool_name": "recipe_cell_replacement", "wns": None},
            {"tool_name": "vivado_report_timing_summary", "wns": -0.4},
        ]
        out = winning_tools_from_call_details(tcd, -0.978)
        self.assertEqual(out, ["recipe_cell_replacement"])


class HintsIncludeWinningToolsTests(unittest.TestCase):
    def test_winning_tools_emit_hint(self):
        r = RunRecord(
            design="x", candidate="v0_3", delta_fmax_mhz=50,
            winning_tools=["vivado_route_design", "vivado_phys_opt_design"],
        )
        out = format_for_prompt(r, "x")
        self.assertIn("prior winning tool sequence", out)
        self.assertIn("vivado_route_design", out)
        self.assertIn("vivado_phys_opt_design", out)


if __name__ == "__main__":
    unittest.main()
