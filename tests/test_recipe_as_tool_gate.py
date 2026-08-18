"""Tests for the recipe-as-tool synthetic call gate in dcp_optimizer.

The full recipe path needs running Vivado + RapidWright MCP servers, so
those branches are integration-tested separately.  This file covers the
pre-flight gate: when the design is marked harmful or error in
scheduler.dispatch.RECIPE_APPLICABILITY, the synthetic tool must refuse
without calling any MCP tool.
"""
from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

# Make repo root importable when tests run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _FakeOptimizer:
    """Minimal stand-in exposing the same surface _recipe_cell_replacement uses.

    Each instance gets a fresh tempdir so tests don't share state via the
    on-disk pins file the LUT recipe writes/reads.
    """
    def __init__(self, design_name):
        import tempfile
        self._design_name_for_memory = design_name
        self.call_log = []
        self.temp_dir = tempfile.mkdtemp(prefix="gate_test_")
        self.initial_wns = -0.5
        self.clock_period = 2.0
        self.best_wns = -0.5

    async def call_tool(self, name, args):
        self.call_log.append((name, args))
        return "{}"

    def calculate_fmax(self, wns, period):
        if wns is None or period is None:
            return None
        return 1000.0 / (period - wns)


# Pull the unbound coroutines from DCPOptimizer so we can call them on _FakeOptimizer.
from dcp_optimizer import DCPOptimizer
_recipe_method = DCPOptimizer._recipe_cell_replacement
_lut_method = DCPOptimizer._recipe_lut_optimization


class RecipeGateTests(unittest.TestCase):
    def _run(self, fake, args=None):
        return asyncio.run(_recipe_method(fake, args or {}))

    def test_blocked_on_harmful_design(self):
        fake = _FakeOptimizer("corescore_500_mod")
        out = self._run(fake)
        parsed = json.loads(out)
        self.assertEqual(parsed["status"], "error")
        self.assertIn("BLOCKED", parsed["error"])
        # Crucially: no MCP calls were made
        self.assertEqual(fake.call_log, [])

    def test_blocked_on_error_design(self):
        fake = _FakeOptimizer("rosetta_digit-recognition")
        out = self._run(fake)
        parsed = json.loads(out)
        self.assertEqual(parsed["status"], "error")
        self.assertIn("BLOCKED", parsed["error"])
        self.assertEqual(fake.call_log, [])

    def test_allowed_on_applicable_design_attempts_call(self):
        fake = _FakeOptimizer("amd_mini-isp")
        # We don't care about the eventual outcome — just that it tries
        # to call MCP tools (i.e., the gate didn't pre-empt).
        try:
            self._run(fake)
        except Exception:
            pass
        self.assertGreater(len(fake.call_log), 0)
        # First call must be extract_critical_path_pins
        self.assertEqual(fake.call_log[0][0], "vivado_extract_critical_path_pins")

    def test_allowed_on_unknown_design(self):
        fake = _FakeOptimizer("brand_new_benchmark_v999")
        try:
            self._run(fake)
        except Exception:
            pass
        self.assertGreater(len(fake.call_log), 0)


class LutRecipeBasicTests(unittest.TestCase):
    """recipe_lut_optimization has no per-design applicability gate (the
    underlying optimize_lut_input_cone short-circuits cleanly on non-LUT
    pins), so we just verify the call structure."""
    def _run(self, fake, args=None):
        return asyncio.run(_lut_method(fake, args or {}))

    def test_attempts_extract_pins_first(self):
        fake = _FakeOptimizer("amd_mini-isp")
        try:
            self._run(fake)
        except Exception:
            pass
        # First MCP call must be extract_critical_path_pins
        self.assertGreater(len(fake.call_log), 0)
        self.assertEqual(fake.call_log[0][0], "vivado_extract_critical_path_pins")

    def test_no_candidates_short_circuits(self):
        # If the pins file doesn't exist (extract_critical_path_pins is a
        # no-op in our fake), the recipe must report no_candidates rather
        # than crashing.
        fake = _FakeOptimizer("amd_mini-isp")
        out = self._run(fake)
        parsed = json.loads(out)
        # Either no_candidates or error (file-read failure) — both are OK,
        # what matters is that we don't reach the optimize step.
        self.assertIn(parsed["status"], ("no_candidates", "error"))


class LooksLikeToolErrorTests(unittest.TestCase):
    """Verify _looks_like_tool_error catches both JSON and TCL-error shapes."""
    def setUp(self):
        from dcp_optimizer import _looks_like_tool_error
        self.detect = _looks_like_tool_error

    def test_json_error_envelope(self):
        self.assertTrue(self.detect('{"error": "oops"}'))

    def test_json_success(self):
        self.assertFalse(self.detect('{"status": "success", "data": 1}'))

    def test_free_form_text_no_error(self):
        self.assertFalse(self.detect("Routing complete.\nINFO: [Vivado 12-1] All good."))

    def test_vivado_tcl_error_caught(self):
        # The actual error string that bit us on the WSL2 path-translation bug
        msg = ("Starting open_checkpoint Task\n"
               "TCL ERROR: ERROR: [Common 17-69] Command failed: File "
               "'C:/tmp/.../recipe_rw_optimized.dcp' does not exist")
        self.assertTrue(self.detect(msg))

    def test_user_exception_caught(self):
        msg = ("Command: route_design -directive Default\n"
               "ERROR: [Common 17-53] User Exception: No open project. "
               "Please create or open a project before executing this command.")
        self.assertTrue(self.detect(msg))

    def test_empty_string(self):
        self.assertFalse(self.detect(""))
        self.assertFalse(self.detect(None))

    def test_malformed_json_with_error_text_still_caught(self):
        # Even if JSON parsing fails, if the body says TCL ERROR, catch it.
        self.assertTrue(self.detect("{bad json with TCL ERROR: something"))


if __name__ == "__main__":
    unittest.main()
