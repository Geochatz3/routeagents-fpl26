"""Fixture-based parser tests for optimizer.qor_json_features.

These tests exercise the tolerant reader against minimised copies of the
five real Session-12 ship-DCP JSONs (Report Information stripped of host /
date / path; design names anonymised to design_a … design_e).  Goal: pin
down behaviour on the actual schemas Vivado emits, not just hand-written
stubs.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizer.qor_json_features import (
    QOR_SCHEMA_VERSION,
    parse_qor_json,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "qor_json"
FIXTURES = sorted(FIXTURE_DIR.glob("design_*.qor.json"))


EXPECTED_KEYS = {
    "schema_version", "schema_ok",
    "qor_steps", "runtime_per_step", "directives_observed",
    "per_step_wns", "per_step_tns",
    "global_cong_level_NESW", "global_cong_tile_NESW",
    "long_cong_level_NESW", "long_cong_tile_NESW",
    "short_cong_level_NESW", "short_cong_tile_NESW",
    "qor_tool_version", "qor_design_state",
    "source_path",
}


class RealFixtureTests(unittest.TestCase):

    def test_fixture_directory_populated(self):
        self.assertGreaterEqual(
            len(FIXTURES), 5,
            f"expected ≥5 fixtures in {FIXTURE_DIR}, got {len(FIXTURES)}",
        )

    def test_all_fixtures_parse_with_schema_ok(self):
        for fx in FIXTURES:
            d = parse_qor_json(fx)
            self.assertTrue(d["schema_ok"], f"{fx.name} failed to parse")

    def test_all_fixtures_have_stable_key_set(self):
        for fx in FIXTURES:
            d = parse_qor_json(fx)
            self.assertEqual(
                set(d.keys()), EXPECTED_KEYS,
                f"key drift in {fx.name}",
            )

    def test_schema_version_unchanged(self):
        for fx in FIXTURES:
            d = parse_qor_json(fx)
            self.assertEqual(d["schema_version"], QOR_SCHEMA_VERSION)

    def test_all_fixtures_emit_at_least_one_step(self):
        for fx in FIXTURES:
            d = parse_qor_json(fx)
            self.assertGreaterEqual(len(d["qor_steps"]), 1, f"{fx.name}")

    def test_tool_version_present(self):
        for fx in FIXTURES:
            d = parse_qor_json(fx)
            self.assertIsInstance(d["qor_tool_version"], str)
            self.assertIn("Vivado", d["qor_tool_version"])

    def test_design_state_present(self):
        for fx in FIXTURES:
            d = parse_qor_json(fx)
            self.assertIsInstance(d["qor_design_state"], str)
            self.assertGreater(len(d["qor_design_state"]), 0)

    def test_congestion_lists_length_4(self):
        for fx in FIXTURES:
            d = parse_qor_json(fx)
            for k in ("global_cong_level_NESW", "global_cong_tile_NESW",
                     "long_cong_level_NESW", "long_cong_tile_NESW",
                     "short_cong_level_NESW", "short_cong_tile_NESW"):
                self.assertEqual(len(d[k]), 4, f"{fx.name} {k}")

    def test_no_design_name_leak_in_parsed_output(self):
        """Parser output must never contain a benchmark name.

        We anonymised fixtures so the input itself doesn't carry names,
        but this test enforces the invariant *over the parser output*
        regardless of input.
        """
        forbidden = {
            "logicnets", "vexriscv", "rosetta", "corescore", "boom_soc",
            "finn", "ispd16", "amd_mini", "vtr", "spam-filter",
        }
        for fx in FIXTURES:
            d = parse_qor_json(fx)
            flat = json.dumps(d).lower()
            for n in forbidden:
                self.assertNotIn(n, flat, f"{fx.name} leaked {n}")

    def test_runtime_per_step_values_are_ints(self):
        for fx in FIXTURES:
            d = parse_qor_json(fx)
            for k, v in d["runtime_per_step"].items():
                self.assertIsInstance(k, str)
                self.assertIsInstance(v, int, f"{fx.name} {k}={v!r}")

    def test_per_step_wns_tns_are_float_or_none(self):
        for fx in FIXTURES:
            d = parse_qor_json(fx)
            for v in d["per_step_wns"] + d["per_step_tns"]:
                self.assertTrue(v is None or isinstance(v, float),
                                f"{fx.name} got {v!r}")


class StabilityAndMalformedTests(unittest.TestCase):
    """Extra coverage: missing fields, malformed numerics, empty JSON,
    unexpected keys, route_bound determinism, design-name independence."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = Path(self.tmp.name) / "x.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, obj):
        self.p.write_text(json.dumps(obj))

    def test_missing_design_qor_summary_returns_default(self):
        self._write({"Report Information": {"Tool Version": "Vivado v.2025.1"}})
        d = parse_qor_json(self.p)
        # Top-level shape OK, but no rows → schema_ok=False.
        self.assertFalse(d["schema_ok"])
        self.assertEqual(d["qor_steps"], [])

    def test_missing_report_information_tolerated(self):
        self._write({"Design QoR Summary": [{"Task Name": "synth_design"}]})
        d = parse_qor_json(self.p)
        self.assertTrue(d["schema_ok"])
        self.assertIsNone(d["qor_tool_version"])
        self.assertIsNone(d["qor_design_state"])

    def test_malformed_numeric_wns_returns_none(self):
        self._write({
            "Report Information": {"Tool Version": "Vivado v.2025.1"},
            "Design QoR Summary": [{
                "Task Name": "route_design", "WNS(ns)": "not-a-number",
                "TNS(ns)": "nan-ish",
            }],
        })
        d = parse_qor_json(self.p)
        self.assertTrue(d["schema_ok"])
        # "not-a-number" → None
        self.assertEqual(d["per_step_wns"], [None])

    def test_empty_design_qor_summary_list(self):
        self._write({"Report Information": {"Tool Version": "X"},
                     "Design QoR Summary": []})
        d = parse_qor_json(self.p)
        self.assertTrue(d["schema_ok"])
        self.assertEqual(d["qor_steps"], [])
        self.assertEqual(d["global_cong_level_NESW"], [None, None, None, None])

    def test_unexpected_extra_top_level_keys_ignored(self):
        self._write({
            "Report Information": {"Tool Version": "X"},
            "Design QoR Summary": [{"Task Name": "route_design"}],
            "FutureFeature": {"will_we_break": True},
            "AnotherTable": [],
        })
        d = parse_qor_json(self.p)
        self.assertTrue(d["schema_ok"])
        # Output key-set unchanged.
        self.assertEqual(set(d.keys()), EXPECTED_KEYS)

    def test_unexpected_extra_row_keys_ignored(self):
        self._write({
            "Report Information": {"Tool Version": "X"},
            "Design QoR Summary": [{
                "Task Name": "route_design",
                "New2026Metric": "57", "Some Other Field": "foo",
                "WNS(ns)": "-1.0",
            }],
        })
        d = parse_qor_json(self.p)
        self.assertTrue(d["schema_ok"])
        self.assertEqual(d["per_step_wns"], [-1.0])

    def test_route_bound_score_deterministic_via_parser(self):
        """Same input must produce same per_step_wns / congestion list."""
        self._write({
            "Report Information": {"Tool Version": "X"},
            "Design QoR Summary": [{
                "Task Name": "route_design",
                "Long Cong Level N-E-S-W": "4 1 1 2",
                "Global Cong Level N-E-S-W": "3 3 2 2",
            }],
        })
        d1 = parse_qor_json(self.p)
        d2 = parse_qor_json(self.p)
        self.assertEqual(d1["long_cong_level_NESW"],
                         d2["long_cong_level_NESW"])
        self.assertEqual(d1["global_cong_level_NESW"],
                         d2["global_cong_level_NESW"])

    def test_parser_has_no_design_name_input_dependence(self):
        """The parser's output for a given JSON must NOT depend on the
        file path that contains the design name."""
        body = {
            "Report Information": {"Tool Version": "X"},
            "Design QoR Summary": [{
                "Task Name": "route_design",
                "Long Cong Level N-E-S-W": "4 1 1 2",
            }],
        }
        tmp1 = Path(self.tmp.name) / "rosetta_spam-filter.qor.json"
        tmp2 = Path(self.tmp.name) / "boom_soc.qor.json"
        tmp3 = Path(self.tmp.name) / "anon.qor.json"
        tmp1.write_text(json.dumps(body))
        tmp2.write_text(json.dumps(body))
        tmp3.write_text(json.dumps(body))
        d1 = parse_qor_json(tmp1)
        d2 = parse_qor_json(tmp2)
        d3 = parse_qor_json(tmp3)
        # source_path differs by file name; everything else must match.
        for d in (d1, d2, d3):
            d_no_src = {k: v for k, v in d.items() if k != "source_path"}
        d1k = {k: v for k, v in d1.items() if k != "source_path"}
        d2k = {k: v for k, v in d2.items() if k != "source_path"}
        d3k = {k: v for k, v in d3.items() if k != "source_path"}
        self.assertEqual(d1k, d2k)
        self.assertEqual(d2k, d3k)


if __name__ == "__main__":
    unittest.main()
