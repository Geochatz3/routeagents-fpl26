"""Tests the budget gate for operations that destroy routed state.

Before a destructive operation starts, the predicted mandatory full reroute and
artifact banking must fit the remaining wall time. A refusal steers toward
bankable incremental physical optimization.

Only the three supported destructive command forms classify as destructive.
Incremental reroutes preserve routed state and must never be blocked by this
gate.

Coverage includes routed-state transitions through open, route, unroute, and
place operations, plus bypasses for finalization, ILS, and the kill switch.
External tool sessions are mocked.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import (
    DCPOptimizer,
    _looks_like_tool_error,
    _routed_state_transition,
    is_routed_state_destroying,
)


def _async(coro):
    return asyncio.run(coro)


class _FakeContent:
    """Mimics MCP CallToolResult content items — has .text attribute."""

    def __init__(self, text: str):
        self.text = text


class _FakeResult:
    """Mimics the MCP CallToolResult shape (result.content list of items)."""

    def __init__(self, text: str):
        self.content = [_FakeContent(text)]


class _FakeSession:
    """Stand-in for the MCP ClientSession used by DCPOptimizer.call_tool."""

    def __init__(self, response_text: str = "{}", sleep_s: float = 0.0):
        self.response_text = response_text
        self.sleep_s = sleep_s
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        if self.sleep_s > 0:
            await asyncio.sleep(self.sleep_s)
        return _FakeResult(self.response_text)


def _make_optimizer(tmp_path: Path) -> DCPOptimizer:
    """Real DCPOptimizer instance, mocked sessions — no servers spawn."""
    opt = DCPOptimizer(api_key="test", run_dir=tmp_path)
    opt.vivado_session = _FakeSession()
    opt.rapidwright_session = _FakeSession()
    return opt


class ClassifierTests(unittest.TestCase):
    """Pure-function contract of is_routed_state_destroying().

    Exactly three destroying cases (01-03-PLAN scope):
      (1) route_design whose command text contains -unroute,
      (2) place_design while the design is currently routed,
      (3) a full re-route while unrouted (no incremental/preserve flag).
    Everything else — notably incremental routed-state-preserving
    re-routes — must return False.
    """

    # ---- case 1: route_design -unroute ----------------------------

    def test_classifier_unroute_on_routed_design_is_destroying(self):
        self.assertTrue(is_routed_state_destroying(
            "vivado_run_tcl", {"command": "route_design -unroute"},
            currently_routed=True))

    def test_classifier_unroute_case_insensitive(self):
        self.assertTrue(is_routed_state_destroying(
            "vivado_run_tcl", {"command": "ROUTE_DESIGN -UNROUTE"},
            currently_routed=True))

    def test_classifier_unroute_reroute_script_is_destroying(self):
        # Unroute + full re-route in ONE script still commits the design
        # to the destroy path — the full re-route must fit.
        self.assertTrue(is_routed_state_destroying(
            "vivado_run_tcl",
            {"command": "route_design -unroute; "
                        "route_design -directive AggressiveExplore"},
            currently_routed=True))

    # ---- case 2: place_design on a routed design ------------------

    def test_classifier_place_design_on_routed_design_is_destroying(self):
        # place_design unroutes: the routed state is destroyed.
        self.assertTrue(is_routed_state_destroying(
            "vivado_place_design", {}, currently_routed=True))

    def test_classifier_run_tcl_place_on_routed_is_destroying(self):
        self.assertTrue(is_routed_state_destroying(
            "vivado_run_tcl", {"command": "place_design -directive Explore"},
            currently_routed=True))

    def test_classifier_full_ruin_script_on_routed_is_destroying(self):
        # place+route full-ruin cycle on a routed design destroys the
        # current routed state (case 2 fires on the place step).
        self.assertTrue(is_routed_state_destroying(
            "vivado_run_tcl",
            {"command": "place_design -directive Explore; "
                        "route_design -directive Explore"},
            currently_routed=True))

    def test_classifier_place_while_unrouted_not_destroying(self):
        # No routed state exists to destroy; the mandatory route AFTER
        # the place is gated separately (case 3) when it is dispatched.
        self.assertFalse(is_routed_state_destroying(
            "vivado_place_design", {}, currently_routed=False))

    # ---- case 3: full re-route while unrouted ---------------------

    def test_classifier_full_reroute_while_unrouted_is_destroying(self):
        self.assertTrue(is_routed_state_destroying(
            "vivado_route_design", {"directive": "AggressiveExplore"},
            currently_routed=False))

    def test_classifier_run_tcl_full_reroute_while_unrouted_is_destroying(self):
        self.assertTrue(is_routed_state_destroying(
            "vivado_run_tcl",
            {"command": "route_design -directive AggressiveExplore"},
            currently_routed=False))

    def test_classifier_preserve_flag_route_while_unrouted_not_destroying(self):
        self.assertFalse(is_routed_state_destroying(
            "vivado_run_tcl",
            {"command": "route_design -preserve -directive Explore"},
            currently_routed=False))

    # ---- incremental routed-state-preserving re-routes: NEVER -----

    def test_classifier_incremental_route_on_routed_design_not_destroying(self):
        # THE critical exclusion: a directive route on a currently-routed
        # design preserves routed state (cheap, ≤101s on ≤85k cells per
        # the eval-box ADDENDUM) — must NEVER be gated.
        self.assertFalse(is_routed_state_destroying(
            "vivado_route_design", {"directive": "Explore"},
            currently_routed=True))

    def test_classifier_run_tcl_incremental_route_on_routed_not_destroying(self):
        self.assertFalse(is_routed_state_destroying(
            "vivado_run_tcl",
            {"command": "route_design -directive NoTimingRelaxation"},
            currently_routed=True))

    # ---- cheap / non-route/place tools: never destroying ----------

    def test_classifier_cheap_tools_never_destroying(self):
        for tool, args in (
            ("vivado_report_timing_summary", {}),
            ("vivado_get_wns", {}),
            ("vivado_write_checkpoint", {"file_path": "/tmp/x.dcp"}),
            ("vivado_write_edif", {"file_path": "/tmp/x.edf"}),
            ("vivado_phys_opt_design", {"directive": "AggressiveExplore"}),
            ("vivado_run_tcl",
             {"command": "report_route_status -return_string"}),
            ("vivado_run_tcl",
             {"command": "puts [get_property PERIOD [get_clocks]]"}),
        ):
            for routed in (True, False):
                self.assertFalse(
                    is_routed_state_destroying(tool, args,
                                               currently_routed=routed),
                    f"{tool} {args} routed={routed} wrongly destroying")


class RoutedStateTrackingTests(unittest.TestCase):
    """_design_routed_state harness tracking across open/route/unroute/
    place, updated after each SUCCESSFUL op in call_tool."""

    OK_TEXT = "command completed successfully"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = _make_optimizer(Path(self.tmp.name))
        self.opt.vivado_session.response_text = self.OK_TEXT
        # Isolate state tracking from the R-D1-1 auto-bank hook (its own
        # suite covers it) — tracking must work regardless.
        self.opt._auto_bank_enabled = False

    def tearDown(self):
        self.tmp.cleanup()

    def _call(self, tool_name: str, arguments: dict | None = None) -> str:
        return _async(self.opt.call_tool(tool_name, arguments or {}))

    def test_state_initializes_true_for_contest_input(self):
        # Contest input DCPs enter placed+routed (WNS<0 on a routed
        # design) — the harness starts from routed.
        self.assertTrue(self.opt._design_routed_state)

    def test_compound_unroute_then_reroute_ends_routed(self):
        # A Tcl command may unroute and then reroute the design.
        # Routed-state tracking must follow the final route_design statement.
        self._call("vivado_run_tcl", {
            "command": "route_design -unroute; "
                       "route_design -directive AggressiveExplore"})
        self.assertTrue(self.opt._design_routed_state)

    def test_state_false_after_successful_unroute(self):
        self._call("vivado_run_tcl", {"command": "route_design -unroute"})
        self.assertFalse(self.opt._design_routed_state)

    def test_state_true_after_successful_route(self):
        self.opt._design_routed_state = False
        self._call("vivado_route_design", {"directive": "Explore"})
        self.assertTrue(self.opt._design_routed_state)

    def test_state_true_after_run_tcl_route(self):
        self.opt._design_routed_state = False
        self._call("vivado_run_tcl",
                   {"command": "route_design -directive Explore"})
        self.assertTrue(self.opt._design_routed_state)

    def test_state_false_after_place_design(self):
        self._call("vivado_place_design", {"directive": "Explore"})
        self.assertFalse(self.opt._design_routed_state)

    def test_state_false_after_run_tcl_place(self):
        self._call("vivado_run_tcl",
                   {"command": "place_design -directive Explore"})
        self.assertFalse(self.opt._design_routed_state)

    def test_state_unchanged_after_failed_op(self):
        # A failed unroute did not destroy anything — state must hold.
        self.opt.vivado_session.response_text = (
            "TCL ERROR: route_design -unroute failed")
        self._call("vivado_run_tcl", {"command": "route_design -unroute"})
        self.assertTrue(self.opt._design_routed_state)

    def test_state_true_after_open_checkpoint(self):
        # Re-opening a checkpoint restores a routed design (contest DCPs
        # and best_valid mirrors are routed-gated).
        self.opt._design_routed_state = False
        self._call("vivado_open_checkpoint", {"file_path": "/tmp/x.dcp"})
        self.assertTrue(self.opt._design_routed_state)

    def test_state_tracking_runs_with_gate_kill_switch_off(self):
        # Tracking itself is NOT behind the kill switch — only the
        # refusal is.  OFF must still keep the state accurate.
        self.opt._unroute_gate_enabled = False
        self._call("vivado_run_tcl", {"command": "route_design -unroute"})
        self.assertFalse(self.opt._design_routed_state)


class GateIntegrationTests(unittest.TestCase):
    """call_tool pre-flight gate: destroying + infeasible → steering
    refusal envelope, no dispatch; feasible/incremental/bypassed →
    normal dispatch."""

    OK_TEXT = "command completed successfully"
    # Eval-box boom figures (2026-07-14 harness log / RESEARCH ADDENDUM).
    BOOM_CELLS = 379_380
    BOOM_PHYS_OPT_S = 1387.79
    BOOM_REMAINING_S = 1650.0

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = _make_optimizer(Path(self.tmp.name))
        self.opt.vivado_session.response_text = self.OK_TEXT
        # Isolate the gate from the R-D1-1 auto-bank hook (own suite).
        self.opt._auto_bank_enabled = False

    def tearDown(self):
        self.tmp.cleanup()

    def _boom_setup(self):
        """The exact boom scenario: routed design, one completed 1387.79s
        phys_opt in history, 379,380 cells, ~1650s remaining."""
        self.opt._design_routed_state = True
        self.opt._input_cell_count = self.BOOM_CELLS
        self.opt.tool_call_details = [{
            "tool_name": "vivado_phys_opt_design",
            "iteration": 1,
            "elapsed_time": self.BOOM_PHYS_OPT_S,
            "wns": None,
            "error": False,
            "cmd_head": "",
        }]
        self.opt._budget_deadline = time.time() + self.BOOM_REMAINING_S

    def _call(self, tool_name: str, arguments: dict | None = None) -> str:
        return _async(self.opt.call_tool(tool_name, arguments or {}))

    # ---- the boom scenario: REFUSED with steering ------------------

    def test_boom_scenario_unroute_refused_and_steers(self):
        self._boom_setup()
        result = self._call("vivado_run_tcl",
                            {"command": "route_design -unroute"})
        parsed = json.loads(result)
        self.assertEqual(parsed["error"], "unroute_gate_refused")
        self.assertIn("REFUSE", parsed["reason"])
        # Steering names the bankable incremental phys_opt alternative.
        self.assertIn("phys_opt", parsed["steering"])
        self.assertIn("incremental", parsed["steering"].lower())
        # The underlying Vivado op was NEVER dispatched.
        self.assertEqual(self.opt.vivado_session.calls, [])

    def test_refusal_envelope_is_error_classified_with_fields(self):
        self._boom_setup()
        result = self._call("vivado_run_tcl",
                            {"command": "route_design -unroute"})
        # Must satisfy the error classifier so telemetry/tracer never
        # mis-report the refusal as a tool success.
        self.assertTrue(_looks_like_tool_error(result))
        parsed = json.loads(result)
        # predicted = max(2.0*1387.79, 379380*0.006) = 2775.58s
        self.assertGreaterEqual(parsed["predicted_reroute_s"], 2775.0)
        self.assertLessEqual(parsed["remaining_seconds"],
                             self.BOOM_REMAINING_S + 1.0)

    def test_refusal_does_not_flip_budget_killed(self):
        # A gate refusal is NOT budget death — the run must continue
        # with bankable incremental alternatives, not finalize.
        self._boom_setup()
        self._call("vivado_run_tcl", {"command": "route_design -unroute"})
        self.assertFalse(self.opt._budget_killed)

    def test_refusal_recorded_in_tool_call_details(self):
        self._boom_setup()
        self._call("vivado_run_tcl", {"command": "route_design -unroute"})
        last = self.opt.tool_call_details[-1]
        self.assertTrue(last["error"])
        self.assertIn("unroute_gate_refused", last["error_message"])

    # ---- other destroying shapes gated the same way ----------------

    def test_place_design_on_routed_design_refused_when_infeasible(self):
        self._boom_setup()
        result = self._call("vivado_place_design", {"directive": "Explore"})
        self.assertIn("unroute_gate_refused", result)
        self.assertEqual(self.opt.vivado_session.calls, [])

    def test_full_reroute_while_unrouted_refused_when_infeasible(self):
        self._boom_setup()
        self.opt._design_routed_state = False
        result = self._call("vivado_route_design",
                            {"directive": "AggressiveExplore"})
        self.assertIn("unroute_gate_refused", result)
        self.assertEqual(self.opt.vivado_session.calls, [])

    # ---- feasible / incremental: never refused ---------------------

    def test_feasible_destroying_op_dispatches(self):
        # Small design, ample budget: predicted = max(0, 12000*0.006=72)
        # + 30 reserve fits 5000-120 → dispatch proceeds.
        self.opt._design_routed_state = True
        self.opt._input_cell_count = 12_000
        self.opt._budget_deadline = time.time() + 5000.0
        result = self._call("vivado_run_tcl",
                            {"command": "route_design -unroute"})
        self.assertEqual(result, self.OK_TEXT)
        self.assertEqual(len(self.opt.vivado_session.calls), 1)

    def test_incremental_route_never_gated_even_when_tight(self):
        # A routed, very large design permits a state-preserving directive route.
        # The unroute gate does not reject it, and the 600 s estimate fits within
        # the 700 s remaining budget.
        self.opt._design_routed_state = True
        self.opt._input_cell_count = self.BOOM_CELLS
        self.opt._budget_deadline = time.time() + 700.0
        result = self._call("vivado_route_design", {"directive": "Explore"})
        self.assertEqual(result, self.OK_TEXT)
        self.assertEqual(len(self.opt.vivado_session.calls), 1)

    # ---- bypasses --------------------------------------------------

    def test_finalize_bypasses_gate(self):
        self._boom_setup()
        self.opt._in_finalize = True
        result = self._call("vivado_run_tcl",
                            {"command": "route_design -unroute"})
        self.assertEqual(result, self.OK_TEXT)
        self.assertEqual(len(self.opt.vivado_session.calls), 1)

    def test_ils_stage_bypasses_gate(self):
        # ILS caps its own unroute/reroute cycles (ils_polish _heavy_to)
        # — the gate must not add a second refusal layer there.
        self._boom_setup()
        self.opt._in_ils_stage = True
        result = self._call("vivado_run_tcl",
                            {"command": "route_design -unroute"})
        self.assertEqual(result, self.OK_TEXT)
        self.assertEqual(len(self.opt.vivado_session.calls), 1)

    def test_kill_switch_off_bypasses_gate(self):
        self._boom_setup()
        self.opt._unroute_gate_enabled = False
        result = self._call("vivado_run_tcl",
                            {"command": "route_design -unroute"})
        self.assertEqual(result, self.OK_TEXT)
        self.assertEqual(len(self.opt.vivado_session.calls), 1)

    def test_kill_switch_defaults_on(self):
        self.assertTrue(self.opt._unroute_gate_enabled)

    # ---- write/status ops must never be gated (RESEARCH Landmine 4:
    # ---- the mirror recurses into call_tool for its writes) ---------

    def test_write_checkpoint_not_gated_even_when_unrouted_and_tight(self):
        # _mirror_best_valid_now / finalize recurse through call_tool
        # with write_checkpoint/write_edif — classifying those as
        # destroying would deadlock banking exactly when it matters.
        self.opt._design_routed_state = False
        self.opt._input_cell_count = self.BOOM_CELLS
        self.opt._budget_deadline = time.time() + 200.0
        for tool in ("vivado_write_checkpoint", "vivado_write_edif"):
            result = self._call(tool, {"file_path": "/tmp/x"})
            self.assertEqual(result, self.OK_TEXT, f"{tool} was refused")
        self.assertEqual(len(self.opt.vivado_session.calls), 2)

    def test_route_status_query_not_gated_while_unrouted(self):
        # _routed_ok_for_best's report_route_status run_tcl must pass the
        # gate untouched (it contains 'route' but not route_design).
        self.opt._design_routed_state = False
        self.opt._input_cell_count = self.BOOM_CELLS
        self.opt._budget_deadline = time.time() + 200.0
        result = self._call("vivado_run_tcl",
                            {"command": "report_route_status -return_string"})
        self.assertEqual(result, self.OK_TEXT)
        self.assertEqual(len(self.opt.vivado_session.calls), 1)

    # ---- state + gate end-to-end sequence --------------------------

    def test_sequence_feasible_unroute_then_infeasible_reroute_refused(self):
        # A feasible unroute is allowed and flips the tracked state;
        # when the wall then shrinks, the mandatory full re-route is
        # refused (case 3) instead of being budget-killed mid-flight.
        self.opt._design_routed_state = True
        self.opt._input_cell_count = self.BOOM_CELLS
        self.opt._budget_deadline = time.time() + 5000.0
        result = self._call("vivado_run_tcl",
                            {"command": "route_design -unroute"})
        self.assertEqual(result, self.OK_TEXT)
        self.assertFalse(self.opt._design_routed_state)
        # Wall shrinks to the boom window — the rebuild no longer fits.
        self.opt._budget_deadline = time.time() + self.BOOM_REMAINING_S
        result = self._call("vivado_route_design",
                            {"directive": "AggressiveExplore"})
        self.assertIn("unroute_gate_refused", result)
        self.assertEqual(len(self.opt.vivado_session.calls), 1)  # unroute only


class TclStatementParsingTests(unittest.TestCase):
    """Review regressions: statement-level parsing.

    Filenames embedding route_design, Tcl comments, newline separators,
    and trailing place_design must not desync _design_routed_state or
    the destroy gate (pure-function tests, no optimizer instance)."""

    def test_filename_containing_route_design_is_no_transition(self):
        args = {"command": "write_checkpoint post_route_design.dcp"}
        self.assertIsNone(_routed_state_transition("vivado_run_tcl", args))
        self.assertFalse(is_routed_state_destroying(
            "vivado_run_tcl", args, currently_routed=False))

    def test_commented_out_reroute_does_not_mask_unroute(self):
        args = {"command": "route_design -unroute; "
                           "# route_design -directive AggressiveExplore"}
        self.assertIs(
            _routed_state_transition("vivado_run_tcl", args), False)
        self.assertTrue(is_routed_state_destroying(
            "vivado_run_tcl", args, currently_routed=True))

    def test_newline_separated_unroute_then_reroute_ends_routed(self):
        args = {"command": "route_design -unroute\n"
                           "route_design -directive AggressiveExplore"}
        self.assertIs(
            _routed_state_transition("vivado_run_tcl", args), True)
        # -unroute commits the destroy path even with the re-route.
        self.assertTrue(is_routed_state_destroying(
            "vivado_run_tcl", args, currently_routed=True))

    def test_route_then_place_ends_unrouted(self):
        args = {"command": "route_design -directive Explore; "
                           "place_design -directive Explore"}
        self.assertIs(
            _routed_state_transition("vivado_run_tcl", args), False)
        self.assertTrue(is_routed_state_destroying(
            "vivado_run_tcl", args, currently_routed=True))

    def test_place_then_route_while_unrouted_destroying_ends_routed(self):
        args = {"command": "place_design\nroute_design -directive Explore"}
        self.assertIs(
            _routed_state_transition("vivado_run_tcl", args), True)
        self.assertTrue(is_routed_state_destroying(
            "vivado_run_tcl", args, currently_routed=False))

    def test_incremental_route_on_routed_design_not_destroying(self):
        args = {"command": "route_design -directive Explore"}
        self.assertFalse(is_routed_state_destroying(
            "vivado_run_tcl", args, currently_routed=True))


if __name__ == "__main__":
    unittest.main()
