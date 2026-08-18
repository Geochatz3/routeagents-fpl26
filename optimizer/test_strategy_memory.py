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
    DESIGN_NOTES,
    RunRecord,
    all_for_design,
    append_run,
    best_for_design,
    design_note_for,
    fingerprint_match,
    format_for_prompt,
    format_top_n_for_prompt,
    global_aggregate,
    load_memory,
    seed_prompt_for,
    top_n_for_design,
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
        # The malformed row is permissively kept (RunRecord with all-None fields),
        # but it has no design — best_for_design will skip it.
        self.assertGreaterEqual(len(records), 3)

    def test_best_for_design_picks_highest_delta(self):
        records = load_memory()
        best = best_for_design(records, "amd_mini-isp")
        self.assertIsNotNone(best)
        self.assertEqual(best.candidate, "anchor")
        self.assertAlmostEqual(best.delta_fmax_mhz, 87.35)

    def test_best_for_design_unknown_returns_none(self):
        records = load_memory()
        self.assertIsNone(best_for_design(records, "nonexistent_design"))

    def test_best_for_design_none_name_returns_none(self):
        records = load_memory()
        self.assertIsNone(best_for_design(records, None))


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
             "delta_fmax_mhz": 87.35, "completed": True},
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

    def test_known_design_returns_snippet(self):
        snippet = seed_prompt_for("amd_mini-isp")
        self.assertIn("anchor", snippet)
        self.assertIn("87.35", snippet)

    def test_unknown_design_falls_back_to_global_aggregate(self):
        # Records have no lut/spread → fingerprint match returns None.
        # New behaviour: fall back to global_aggregate so the LLM still
        # sees *some* prior, not nothing.
        snippet = seed_prompt_for("nonexistent")
        self.assertIn("PRIOR-CAMPAIGN HISTORY:", snippet)
        self.assertIn("prior campaigns", snippet)


class TopNForDesignTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            RunRecord(design="x", candidate="anchor", delta_fmax_mhz=20, completed=True),
            RunRecord(design="x", candidate="v0_3", delta_fmax_mhz=50, completed=True),
            RunRecord(design="x", candidate="v0_3_seed2", delta_fmax_mhz=53, completed=True),
            RunRecord(design="x", candidate="v0_3_seed3", delta_fmax_mhz=46, completed=True),
            RunRecord(design="other", candidate="anchor", delta_fmax_mhz=99, completed=True),
        ]

    def test_filters_by_design(self):
        out = top_n_for_design(self.records, "x", n=10)
        for r in out:
            self.assertEqual(r.design, "x")

    def test_dedups_seeds_by_candidate_class(self):
        # v0_3, v0_3_seed2, v0_3_seed3 should collapse to one entry (highest)
        out = top_n_for_design(self.records, "x", n=10)
        candidate_classes = [(r.candidate or "").split("_seed")[0] for r in out]
        self.assertEqual(len(candidate_classes), len(set(candidate_classes)))

    def test_returns_highest_per_class(self):
        out = top_n_for_design(self.records, "x", n=10)
        # The single v0_3 representative should be seed2 (highest delta=53)
        v03 = next(r for r in out if (r.candidate or "").startswith("v0_3"))
        self.assertEqual(v03.candidate, "v0_3_seed2")
        self.assertEqual(v03.delta_fmax_mhz, 53)

    def test_caps_at_n(self):
        out = top_n_for_design(self.records, "x", n=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].candidate, "v0_3_seed2")  # highest

    def test_empty_for_unknown_design(self):
        self.assertEqual(top_n_for_design(self.records, "nonexistent"), [])
        self.assertEqual(top_n_for_design(self.records, None), [])


class FormatTopNForPromptTests(unittest.TestCase):
    def test_empty_returns_empty_string(self):
        self.assertEqual(format_top_n_for_prompt([], "x"), "")

    def test_single_record_uses_single_format(self):
        # When only one record, falls back to format_for_prompt's single style
        r = RunRecord(design="x", candidate="anchor", delta_fmax_mhz=10)
        out = format_top_n_for_prompt([r], "x")
        single = format_for_prompt(r, "x")
        self.assertEqual(out, single)

    def test_multi_record_lists_all(self):
        r1 = RunRecord(design="x", candidate="anchor", delta_fmax_mhz=20, iterations=3)
        r2 = RunRecord(design="x", candidate="v0_3", delta_fmax_mhz=50, iterations=5,
                       force_continues=2)
        out = format_top_n_for_prompt([r2, r1], "x")
        self.assertIn("anchor", out)
        self.assertIn("v0_3", out)
        self.assertIn("+50.00", out)
        self.assertIn("+20.00", out)
        # Hints should come from the best (first) record
        self.assertIn("force-continue", out)

    def test_fingerprint_match_relabels(self):
        r = RunRecord(design="other_design", candidate="v0_3", delta_fmax_mhz=50)
        out = format_top_n_for_prompt([r, r], "brand_new")
        self.assertIn("closest fingerprint match", out)
        self.assertIn("other_design", out)


class SeedPromptTopNTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "mem.jsonl"
        _write_jsonl(self.path, [
            {"design": "x", "candidate": "anchor", "delta_fmax_mhz": 20.0, "completed": True},
            {"design": "x", "candidate": "v0_3", "delta_fmax_mhz": 50.0, "completed": True},
            {"design": "x", "candidate": "v0_3_seed2", "delta_fmax_mhz": 55.0, "completed": True},
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

    def test_default_top_n_3_includes_multiple_candidates(self):
        snippet = seed_prompt_for("x")
        # Both anchor and v0_3 (deduped from seeds) should appear
        self.assertIn("anchor", snippet)
        self.assertIn("v0_3", snippet)

    def test_top_n_1_returns_single_format(self):
        snippet = seed_prompt_for("x", top_n=1)
        # Should not have the "top candidates" header
        self.assertNotIn("top candidates", snippet)


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
             "delta_fmax_mhz": 50.0, "completed": True},
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

    def test_unknown_design_falls_back_to_global_aggregate(self):
        snippet = seed_prompt_for("brand_new")
        self.assertIn("PRIOR-CAMPAIGN HISTORY:", snippet)
        self.assertIn("prior campaigns", snippet)

    def test_known_design_uses_match_not_global(self):
        snippet = seed_prompt_for("known_a")
        self.assertIn("anchor", snippet)
        # Should NOT show the "Across N prior campaigns" aggregate header
        self.assertNotIn("Across", snippet)


class DesignNoteTests(unittest.TestCase):
    def test_known_loss_design_returns_note(self):
        out = design_note_for("corescore_500_mod")
        self.assertTrue(out.startswith("DESIGN-SPECIFIC NOTE:"))
        self.assertIn("LOSS", out)
        self.assertIn("retiming", out.lower())

    def test_known_tight_design_returns_note(self):
        out = design_note_for("vexriscv_re-place_v2")
        self.assertTrue(out.startswith("DESIGN-SPECIFIC NOTE:"))
        self.assertIn("tight", out.lower())

    def test_unknown_design_returns_empty(self):
        self.assertEqual(design_note_for("brand_new_design"), "")
        self.assertEqual(design_note_for(None), "")

    def test_all_table_entries_have_substantive_notes(self):
        # Sanity — every entry should be a non-trivial sentence
        for design, note in DESIGN_NOTES.items():
            self.assertGreater(len(note), 50,
                               f"DESIGN_NOTES[{design!r}] is too short")

    def test_note_appears_in_seed_prompt_when_known(self):
        # corescore appears in DESIGN_NOTES — should land in the iter-1 snippet
        # even when we use the canonical seed memory (which has corescore data).
        snippet = seed_prompt_for("corescore_500_mod")
        self.assertIn("DESIGN-SPECIFIC NOTE:", snippet)
        # The campaign-history section should also be there (memory has corescore)
        self.assertIn("PRIOR-CAMPAIGN HISTORY", snippet)


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
