"""Exercise the complete recipe-as-tool flow with mocked tool responses.

The tests cover no-candidate, no-move, routing-error, and success branches
without invoking Vivado. They also validate error-envelope handling,
routing-error parsing, and candidate-pin filtering.
"""
from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import DCPOptimizer

_recipe = DCPOptimizer._recipe_cell_replacement
_lut_recipe = DCPOptimizer._recipe_lut_optimization
_retime_recipe = DCPOptimizer._recipe_register_retiming


class _MockOptimizer:
    """Provide lookup-driven canned responses for optimizer tests.

    Unconfigured tools return '{}' and each instance uses a fresh temporary
    directory for isolation.
    """
    def __init__(self, design_name="amd_mini-isp", responses=None, temp_dir=None):
        import tempfile
        self._design_name_for_memory = design_name
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="recipe_test_")
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        self.initial_wns = -0.5
        self.clock_period = 2.0
        self.best_wns = -0.5
        self.responses = responses or {}
        self.call_log = []

    async def call_tool(self, name, args):
        self.call_log.append((name, args))
        if name in self.responses:
            v = self.responses[name]
            return v(args) if callable(v) else v
        # Default: empty success
        return "{}"

    def calculate_fmax(self, wns, period):
        if wns is None or period is None:
            return None
        return 1000.0 / (period - wns)


def _run(method, fake, args=None):
    return asyncio.run(method(fake, args or {}))


def _critical_paths_jsonish(paths):
    """Generate the JSON shape extract_critical_path_pins writes to file."""
    return json.dumps(paths)


class CellReplacementFullFlowTests(unittest.TestCase):
    def setUp(self):
        # Pre-seed analyze_net_detour with one valid candidate
        self.detour_response = json.dumps({
            "cells_analyzed": 12,
            "candidates": [
                {"cell": "design/inst/lut3", "max_detour_ratio": 3.5, "path": 1},
                {"cell": "design/inst/ff_x", "max_detour_ratio": 2.8, "path": 2},
            ],
        })

    def test_full_success_path(self):
        responses = {
            "rapidwright_analyze_net_detour": self.detour_response,
            "rapidwright_optimize_cell_placement": json.dumps({
                "results": [{"cell": "design/inst/lut3", "status": "success",
                             "message": "moved successfully"}]
            }),
            "vivado_report_route_status": "...\n# of nets with routing errors    :  0\n...",
        }
        # write_checkpoint must "create" the file for rw_out.exists() check
        rw_out = Path("/tmp/recipe_test/recipe_rw_optimized.dcp")
        def fake_write(args):
            Path(args["dcp_path"]).touch()
            return "Wrote checkpoint OK"
        responses["rapidwright_write_checkpoint"] = fake_write

        fake = _MockOptimizer(responses=responses)
        out = _run(_recipe, fake)
        parsed = json.loads(out)

        self.assertEqual(parsed["status"], "success", parsed)
        self.assertEqual(parsed["route_errors"], 0)
        self.assertIn("design/inst/lut3", parsed["cells_moved"])
        # Cleanup
        rw_out.unlink(missing_ok=True)

    def test_route_errors_short_circuits(self):
        # 5 routing errors → status=route_errors, no success
        responses = {
            "rapidwright_analyze_net_detour": self.detour_response,
            "rapidwright_optimize_cell_placement": json.dumps({
                "results": [{"cell": "design/inst/lut3", "status": "success", "message": "moved"}]
            }),
            "vivado_report_route_status": "# of nets with routing errors    :  5\n",
            "rapidwright_write_checkpoint": lambda a: (Path(a["dcp_path"]).touch(), "OK")[1],
        }
        fake = _MockOptimizer(responses=responses)
        out = _run(_recipe, fake)
        parsed = json.loads(out)
        self.assertEqual(parsed["status"], "route_errors")
        self.assertEqual(parsed["route_errors"], 5)

    def test_no_cells_moved_when_all_fail(self):
        responses = {
            "rapidwright_analyze_net_detour": self.detour_response,
            "rapidwright_optimize_cell_placement": json.dumps({
                "results": [{"cell": "design/inst/lut3", "status": "skipped",
                             "message": "cell type not movable"}]
            }),
        }
        fake = _MockOptimizer(responses=responses)
        out = _run(_recipe, fake)
        parsed = json.loads(out)
        self.assertEqual(parsed["status"], "no_cells_moved")
        self.assertEqual(len(parsed["cells_moved"]), 0)

    def test_no_candidates_when_detour_empty(self):
        responses = {
            "rapidwright_analyze_net_detour": json.dumps({
                "cells_analyzed": 12, "candidates": []
            }),
        }
        fake = _MockOptimizer(responses=responses)
        out = _run(_recipe, fake)
        parsed = json.loads(out)
        self.assertEqual(parsed["status"], "no_candidates")

    def test_write_checkpoint_failure_propagates_clean_error(self):
        responses = {
            "rapidwright_analyze_net_detour": self.detour_response,
            "rapidwright_optimize_cell_placement": json.dumps({
                "results": [{"cell": "design/inst/lut3", "status": "success", "message": "moved"}]
            }),
            # call_tool error envelope
            "rapidwright_write_checkpoint": json.dumps({"error": "RapidWright JVM died"}),
        }
        fake = _MockOptimizer(responses=responses)
        out = _run(_recipe, fake)
        parsed = json.loads(out)
        self.assertEqual(parsed["status"], "error")
        self.assertIn("write_checkpoint failed", parsed["error"])

    def test_open_checkpoint_failure_propagates(self):
        responses = {
            "rapidwright_analyze_net_detour": self.detour_response,
            "rapidwright_optimize_cell_placement": json.dumps({
                "results": [{"cell": "design/inst/lut3", "status": "success", "message": "moved"}]
            }),
            "rapidwright_write_checkpoint": lambda a: (Path(a["dcp_path"]).touch(), "OK")[1],
            "vivado_open_checkpoint": json.dumps({"error": "DCP corrupt"}),
        }
        fake = _MockOptimizer(responses=responses)
        out = _run(_recipe, fake)
        parsed = json.loads(out)
        self.assertEqual(parsed["status"], "error")
        self.assertIn("open_checkpoint failed", parsed["error"])


class LutRecipeFullFlowTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp_dir = tempfile.mkdtemp(prefix="lut_recipe_test_")
        # Pin format: "cell/pin_name".  LUT inputs match /I[0-5]$
        self.critical_paths = [
            ["src_ff/Q", "lut1/I2", "lut1/O", "lut2/I0", "lut2/O", "dst_ff/D"],
            ["a_ff/Q", "lut3/I1", "lut3/O", "b_ff/D"],
        ]

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _seed_pins_file(self, args):
        """Mock for vivado_extract_critical_path_pins: writes the pins file."""
        Path(args["output_file"]).write_text(json.dumps(self.critical_paths))
        return "Wrote pins to file"

    def _make(self, responses):
        return _MockOptimizer(responses=responses, temp_dir=self.tmp_dir)

    def test_extracts_lut_input_pins_only(self):
        responses = {
            "vivado_extract_critical_path_pins": self._seed_pins_file,
            # Pretend optimization succeeds for one pin
            "rapidwright_optimize_lut_input_cone": json.dumps({
                "results": [
                    {"pin": "lut1/I2", "status": "optimized", "message": "merged"},
                    {"pin": "lut2/I0", "status": "no_optimization", "message": "single LUT"},
                ]
            }),
            "vivado_report_route_status": "# of nets with routing errors    :  0\n",
            "rapidwright_write_checkpoint": lambda a: (Path(a["dcp_path"]).touch(), "OK")[1],
        }
        fake = self._make(responses)
        out = _run(_lut_recipe, fake)
        parsed = json.loads(out)
        # Should have filtered to LUT-input pins (I0..I5)
        self.assertIn("lut1/I2", parsed["pins_targeted"])
        self.assertEqual(parsed["status"], "success")
        self.assertEqual(parsed["pins_optimized"], ["lut1/I2"])

    def test_no_candidates_when_no_lut_inputs(self):
        # Override critical_paths with paths that have no /I[0-5]$ pins
        self.critical_paths = [["a/Q", "b/D"], ["c/Q", "d/D"]]
        responses = {
            "vivado_extract_critical_path_pins": self._seed_pins_file,
        }
        fake = self._make(responses)
        out = _run(_lut_recipe, fake)
        parsed = json.loads(out)
        self.assertEqual(parsed["status"], "no_candidates")

    def test_route_errors_short_circuits(self):
        responses = {
            "vivado_extract_critical_path_pins": self._seed_pins_file,
            "rapidwright_optimize_lut_input_cone": json.dumps({
                "results": [{"pin": "lut1/I2", "status": "optimized", "message": "merged"}]
            }),
            "vivado_report_route_status": "# of nets with routing errors    :  3\n",
            "rapidwright_write_checkpoint": lambda a: (Path(a["dcp_path"]).touch(), "OK")[1],
        }
        fake = self._make(responses)
        out = _run(_lut_recipe, fake)
        parsed = json.loads(out)
        self.assertEqual(parsed["status"], "route_errors")
        self.assertEqual(parsed["route_errors"], 3)

    def test_no_optimization_possible(self):
        # All pins return no_optimization
        responses = {
            "vivado_extract_critical_path_pins": self._seed_pins_file,
            "rapidwright_optimize_lut_input_cone": json.dumps({
                "results": [
                    {"pin": "lut1/I2", "status": "no_optimization", "message": "single"},
                    {"pin": "lut2/I0", "status": "no_optimization", "message": "single"},
                ]
            }),
        }
        fake = self._make(responses)
        out = _run(_lut_recipe, fake)
        parsed = json.loads(out)
        self.assertEqual(parsed["status"], "no_optimization_possible")


class RetimingRecipeTests(unittest.TestCase):
    def test_rejects_invalid_directive(self):
        fake = _MockOptimizer()
        out = _run(_retime_recipe, fake, {"directive": "Bogus"})
        parsed = json.loads(out)
        self.assertEqual(parsed["status"], "error")
        self.assertIn("directive must be", parsed["error"])

    def test_success_path(self):
        responses = {
            "vivado_phys_opt_design": "phys_opt completed OK",
            "vivado_route_design": "Route OK",
            "vivado_report_route_status": "# of nets with routing errors    :  0\n",
        }
        fake = _MockOptimizer(responses=responses)
        # Bump self.best_wns so the recipe sees improvement
        fake.best_wns = -0.2
        out = _run(_retime_recipe, fake)
        parsed = json.loads(out)
        self.assertEqual(parsed["status"], "success")
        self.assertEqual(parsed["directive_used"], "AlternateFlowWithRetiming")
        self.assertEqual(parsed["route_errors"], 0)

    def test_route_errors_short_circuits(self):
        responses = {
            "vivado_phys_opt_design": "phys_opt completed OK",
            "vivado_route_design": "Route OK",
            "vivado_report_route_status": "# of nets with routing errors    :  4\n",
        }
        fake = _MockOptimizer(responses=responses)
        out = _run(_retime_recipe, fake)
        parsed = json.loads(out)
        self.assertEqual(parsed["status"], "route_errors")
        self.assertEqual(parsed["route_errors"], 4)

    def test_phys_opt_error_propagates(self):
        responses = {
            "vivado_phys_opt_design": json.dumps({"error": "incompatible directive"}),
        }
        fake = _MockOptimizer(responses=responses)
        out = _run(_retime_recipe, fake)
        parsed = json.loads(out)
        self.assertEqual(parsed["status"], "error")
        self.assertIn("phys_opt_design failed", parsed["error"])


if __name__ == "__main__":
    unittest.main()
