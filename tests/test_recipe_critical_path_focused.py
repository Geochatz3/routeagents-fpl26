"""Unit tests for recipe_critical_path_focused_phys_opt."""
from __future__ import annotations

import asyncio
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import DCPOptimizer


def _async(coro):
    return asyncio.run(coro)


def _make_opt(tmp_path: Path) -> DCPOptimizer:
    opt = DCPOptimizer(api_key="test", run_dir=tmp_path)
    opt.clock_period = 1.570
    opt.best_wns = -1.000
    opt.initial_wns = -1.000
    bv = tmp_path / "best_valid.dcp"
    bv.write_bytes(b"BEST_VALID_DCP_BYTES")
    opt._best_valid_dcp = bv
    return opt


ENDPOINT_EXTRACT_RESPONSE = """ENDPOINTS:3
EP:system/foo/reg_a/D
EP:system/foo/reg_b/D
EP:system/bar/reg_c/D
"""


class CriticalPathFocusedDispatchTests(unittest.TestCase):
    def test_dispatch_routes_to_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            opt = _make_opt(Path(tmp))
            with mock.patch.object(
                opt, "_recipe_critical_path_focused_phys_opt",
                new=mock.AsyncMock(return_value='{"status": "stub"}'),
            ) as stub:
                result = _async(opt.call_tool(
                    "recipe_critical_path_focused_phys_opt", {}
                ))
                self.assertEqual(result, '{"status": "stub"}')
                stub.assert_awaited_once()


class CriticalPathFocusedBehaviourTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.opt = _make_opt(self.tmp_path)
        # Capture the Tcl commands actually issued so we can assert
        # the recipe builds the right group_path / phys_opt sequence.
        self.tcl_calls: list[str] = []
        self.phys_opt_calls: list[dict] = []

    def tearDown(self):
        self.tmp.cleanup()

    def _make_fake_call(self, post_wns: str = "-1.000"):
        async def fake(name, args):
            if name == "vivado_run_tcl":
                cmd = args.get("command", "")
                self.tcl_calls.append(cmd)
                if "get_timing_paths" in cmd and "ENDPOINTS:" in cmd:
                    return ENDPOINT_EXTRACT_RESPONSE
                if "group_path" in cmd:
                    return "PATH_GROUP:pg_critical_focus"
                return "ok"
            if name == "vivado_phys_opt_design":
                self.phys_opt_calls.append(args)
                # Simulate the requested sub_flag committing some improvement.
                self.opt.best_wns = float(post_wns)
                return "phys_opt_design completed"
            if name == "vivado_get_wns":
                return post_wns
            if name == "vivado_report_route_status":
                return "# of nets with routing errors that are routable: 0"
            if name == "vivado_open_checkpoint":
                return "checkpoint opened"
            return "ok"
        return fake

    def test_no_initial_wns_returns_error(self):
        self.opt.best_wns = float("-inf")
        out = _async(self.opt._recipe_critical_path_focused_phys_opt({}))
        r = json.loads(out)
        self.assertEqual(r["status"], "error")
        self.assertIn("best_wns not yet established", r["error"])

    def test_invalid_sub_flag_rejected(self):
        # Stray sub_flag like "AggressiveExplore" must NOT pass through —
        # would otherwise let the LLM smuggle a full directive into the
        # phys_opt call.
        out = _async(self.opt._recipe_critical_path_focused_phys_opt(
            {"sub_flag": "AggressiveExplore"}
        ))
        r = json.loads(out)
        self.assertEqual(r["status"], "error")
        self.assertIn("invalid sub_flag", r["error"])

    def test_no_endpoints_short_circuits(self):
        async def fake(name, args):
            if name == "vivado_run_tcl":
                # Tcl returned ENDPOINTS:0 (no critical paths).
                return "ENDPOINTS:0\n"
            raise AssertionError(f"unexpected tool call {name}")
        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake)):
            out = _async(self.opt._recipe_critical_path_focused_phys_opt({}))
        r = json.loads(out)
        self.assertEqual(r["status"], "no_endpoints")
        self.assertEqual(r["endpoints_captured"], 0)

    def test_commit_on_improvement(self):
        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(
                                    side_effect=self._make_fake_call(post_wns="-0.900"))):
            out = _async(self.opt._recipe_critical_path_focused_phys_opt(
                {"num_paths": 20, "sub_flag": "critical_cell_opt"}
            ))
        r = json.loads(out)
        self.assertEqual(r["status"], "committed")
        self.assertEqual(r["endpoints_captured"], 3)
        self.assertAlmostEqual(r["delta_wns_ns"], 0.100, places=3)
        # phys_opt was called with our sub_flag AND path_groups arg.
        self.assertEqual(len(self.phys_opt_calls), 1)
        po_args = self.phys_opt_calls[0]
        self.assertTrue(po_args.get("critical_cell_opt"))
        self.assertEqual(po_args.get("path_groups"), "pg_critical_focus")
        # Tcl extracted endpoints AND defined a path group.
        tcl_joined = "\n".join(self.tcl_calls)
        self.assertIn("get_timing_paths", tcl_joined)
        self.assertIn("group_path -name pg_critical_focus", tcl_joined)

    def test_regression_triggers_revert(self):
        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(
                                    side_effect=self._make_fake_call(post_wns="-1.200"))):
            out = _async(self.opt._recipe_critical_path_focused_phys_opt(
                {"num_paths": 20}
            ))
        r = json.loads(out)
        self.assertEqual(r["status"], "reverted_regression")
        self.assertEqual(r["revert_status"], "reopened_best_valid")

    def test_sub_epsilon_noise_reverts(self):
        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(
                                    side_effect=self._make_fake_call(post_wns="-0.995"))):
            out = _async(self.opt._recipe_critical_path_focused_phys_opt(
                {"num_paths": 20, "epsilon_ns": 0.010}
            ))
        r = json.loads(out)
        self.assertEqual(r["status"], "reverted_no_gain")

    def test_phys_opt_failure_triggers_revert(self):
        async def fake(name, args):
            if name == "vivado_run_tcl":
                return ENDPOINT_EXTRACT_RESPONSE
            if name == "vivado_phys_opt_design":
                return json.dumps({"error": "phys_opt internal error"})
            if name == "vivado_open_checkpoint":
                return "ok"
            return "ok"
        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake)):
            out = _async(self.opt._recipe_critical_path_focused_phys_opt({}))
        r = json.loads(out)
        self.assertEqual(r["status"], "error")
        self.assertIn("scoped phys_opt failed", r["error"])
        self.assertEqual(r["revert_status"], "reopened_best_valid")

    def test_measure_failure_reverts(self):
        # phys_opt succeeded, but get_wns returned non-numeric data
        # (Vivado state inconsistent after the optimization).
        async def fake(name, args):
            if name == "vivado_run_tcl":
                return ENDPOINT_EXTRACT_RESPONSE
            if name == "vivado_phys_opt_design":
                return "ok"
            if name == "vivado_get_wns":
                return "WARNING: no current timing"
            if name == "vivado_open_checkpoint":
                return "ok"
            return "ok"
        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake)):
            out = _async(self.opt._recipe_critical_path_focused_phys_opt({}))
        r = json.loads(out)
        self.assertEqual(r["status"], "measure_failed_parse")
        self.assertEqual(r["revert_status"], "reopened_best_valid")


# These predicates flag forbidden Vivado Tcl combinations that mocked tool
# calls cannot validate. In particular, `-setup` and `-delay_type max`
# must not appear together in `get_timing_paths` commands.
TCL_LINT_RULES = [
    (
        lambda s: "get_timing_paths" in s
                  and re.search(r"-setup\b", s)
                  and re.search(r"-delay_type\b", s),
        "get_timing_paths: -setup conflicts with -delay_type (Vivado 12-608)",
    ),
    (
        lambda s: "get_timing_paths" in s
                  and re.search(r"-hold\b", s)
                  and re.search(r"-delay_type\b", s),
        "get_timing_paths: -hold conflicts with -delay_type (Vivado 12-608)",
    ),
]


def _lint_tcl(cmd: str) -> list[str]:
    return [msg for pred, msg in TCL_LINT_RULES if pred(cmd)]


class CriticalPathFocusedTclShapeTests(unittest.TestCase):
    """Verify the exact Tcl commands issued by the critical-path recipe.

    Exact string assertions protect command scoping and option compatibility
    that keyword-only checks cannot detect.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.opt = _make_opt(self.tmp_path)
        self.tcl_calls: list[str] = []

    def tearDown(self):
        self.tmp.cleanup()

    def _capture_tcl_fake(self, post_wns: str = "-1.000"):
        async def fake(name, args):
            if name == "vivado_run_tcl":
                cmd = args.get("command", "")
                self.tcl_calls.append(cmd)
                if "get_timing_paths" in cmd and "ENDPOINTS:" in cmd:
                    return ENDPOINT_EXTRACT_RESPONSE
                if "group_path" in cmd:
                    return "PATH_GROUP:pg_critical_focus"
                return "ok"
            if name == "vivado_phys_opt_design":
                self.opt.best_wns = float(post_wns)
                return "phys_opt_design completed"
            if name == "vivado_get_wns":
                return post_wns
            if name == "vivado_open_checkpoint":
                return "ok"
            return "ok"
        return fake

    def _run_recipe(self):
        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(
                                    side_effect=self._capture_tcl_fake())):
            _async(self.opt._recipe_critical_path_focused_phys_opt(
                {"num_paths": 20, "sub_flag": "critical_cell_opt"}
            ))

    def test_extract_tcl_passes_lint(self):
        self._run_recipe()
        extract = next((c for c in self.tcl_calls
                        if "get_timing_paths" in c and "ENDPOINTS:" in c), None)
        self.assertIsNotNone(extract,
                             "endpoint-extract Tcl not issued")
        violations = _lint_tcl(extract)
        self.assertEqual(violations, [],
                         f"endpoint-extract Tcl failed lint:\n  Tcl: {extract!r}\n  Violations: {violations}")

    def test_extract_tcl_uses_setup_without_delay_type(self):
        # The exact regression: this combo triggered Vivado 12-608.
        self._run_recipe()
        extract = next((c for c in self.tcl_calls
                        if "get_timing_paths" in c and "ENDPOINTS:" in c), None)
        self.assertIsNotNone(extract)
        self.assertIn("-setup", extract,
                      "expected -setup flag for max-delay analysis")
        self.assertNotIn("-delay_type", extract,
                         "must NOT use -delay_type with -setup (Vivado 12-608)")

    def test_extract_tcl_pins_full_command_shape(self):
        # This snapshot pins the Tcl protocol consumed by the parser:
        # sorted setup paths, one endpoint pin per path, and stable
        # `ENDPOINTS:<count>` and `EP:<pin>` output markers.
        self._run_recipe()
        extract = next((c for c in self.tcl_calls
                        if "get_timing_paths" in c and "ENDPOINTS:" in c), None)
        self.assertIsNotNone(extract)
        self.assertRegex(
            extract,
            r"get_timing_paths\s+-max_paths\s+\d+\s+-setup\s+-sort_by\s+slack",
            "endpoint-extract Tcl shape changed; verify deliberately and update test",
        )
        self.assertIn("get_property ENDPOINT_PIN", extract)
        self.assertIn("ENDPOINTS:", extract)

    def test_group_path_tcl_uses_named_path_group(self):
        # group_path Tcl must define a named path group containing the
        # extracted endpoints.  Without this, phys_opt_design
        # -path_groups pg_critical_focus would fail at runtime.
        self._run_recipe()
        pg_cmd = next((c for c in self.tcl_calls if "group_path -name" in c), None)
        self.assertIsNotNone(pg_cmd, "group_path Tcl not issued")
        self.assertRegex(
            pg_cmd,
            r"group_path\s+-name\s+pg_critical_focus\s+-to\s+\[get_pins",
            "group_path shape changed; phys_opt -path_groups would fail",
        )

    def test_lint_rule_self_check(self):
        # Sanity: the lint rules actually fire on the known-bad combo.
        bad = ("set tps [get_timing_paths -max_paths 20 "
               "-setup -delay_type max -sort_by slack]")
        self.assertTrue(_lint_tcl(bad),
                        "lint rule failed to detect the regression shape")
        bad_hold = "set tps [get_timing_paths -max_paths 5 -hold -delay_type min]"
        self.assertTrue(_lint_tcl(bad_hold))
        # Good combos (from elsewhere in the codebase) must NOT fire.
        good_setup_only = "get_timing_paths -max_paths 1 -setup"
        self.assertEqual(_lint_tcl(good_setup_only), [])
        good_delay_only = "report_timing -delay_type max -sort_by slack"
        self.assertEqual(_lint_tcl(good_delay_only), [])


if __name__ == "__main__":
    unittest.main()
