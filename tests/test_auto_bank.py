"""R-D1-1 auto-bank-after-heavy-op tests (2026-07-19).

The boom_soc_2025.1_v2 eval run (2026-07-14, budget 3499s) completed a
1387.79s vivado_phys_opt_design that was NEVER measured and NEVER banked —
the LLM skipped the measurement call, ran `route_design -unroute`, the
re-route was budget-killed, and finalize shipped baseline (alpha=0,
rank 20/21).  The auto-bank hook in call_tool closes that hole: after ANY
heavy mutating op returns OK, the harness itself measures contest-clock WNS
(cheap SLACK query) and, if improved AND routed, mirrors best_valid — no
LLM measurement call required.

We never spawn Vivado/RapidWright/MCP — sessions are mocked, and the
measurement / routedness / mirror primitives are stubbed per test so each
gate of the hook is exercised in isolation.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import DCPOptimizer


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


class AutoBankTests(unittest.TestCase):
    """Every R-D1-1 invariant of the post-heavy-op auto-bank hook."""

    HEAVY_TOOL = "vivado_phys_opt_design"
    OK_TEXT = "phys_opt_design completed successfully"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = _make_optimizer(Path(self.tmp.name))
        self.opt.vivado_session.response_text = self.OK_TEXT
        # Baseline measurement state: a prior best exists at -2.0 ns.
        self.opt.best_wns = -2.0
        # Ample budget so the pre-flight risky-estimate gate passes
        # (default risky estimate 600s < 5000s remaining).
        self.opt._budget_deadline = time.time() + 5000.0

        # --- instrumented stubs -------------------------------------
        self.measure_calls: list[object] = []
        self.mirror_calls: list[bool] = []
        self.routed_result = True
        self.measured_wns: float | None = -1.0  # improved vs -2.0

        async def fake_measure(call_tool_fn):
            self.measure_calls.append(call_tool_fn)
            return self.measured_wns

        async def fake_routed_ok():
            return self.routed_result

        async def fake_mirror(eager: bool = False):
            self.mirror_calls.append(eager)

        self.opt.get_wns_for_target_clock = fake_measure
        self.opt._routed_ok_for_best = fake_routed_ok
        self.opt._mirror_best_valid_now = fake_mirror

    def tearDown(self):
        self.tmp.cleanup()

    def _call_heavy(self, tool_name: str | None = None, arguments: dict | None = None) -> str:
        return _async(self.opt.call_tool(tool_name or self.HEAVY_TOOL,
                                         arguments if arguments is not None else {}))

    # ---- the load-bearing path ------------------------------------

    def test_bank_fires_on_improved_and_routed(self):
        # Heavy op OK + WNS improved + routed → best_wns updated, mirror
        # invoked eagerly, no LLM measurement call needed.
        result = self._call_heavy()
        self.assertEqual(result, self.OK_TEXT)
        self.assertEqual(len(self.measure_calls), 1)
        self.assertEqual(self.opt.best_wns, -1.0)
        self.assertTrue(self.opt._pending_best_mirror)
        self.assertEqual(self.mirror_calls, [True])  # eager=True

    def test_no_bank_when_unrouted(self):
        # must-not-bank-unrouted invariant (phantom-best guard): improved
        # WNS on an unrouted design must NOT touch best state.
        self.routed_result = False
        self._call_heavy()
        self.assertEqual(len(self.measure_calls), 1)
        self.assertEqual(self.opt.best_wns, -2.0)
        self.assertFalse(self.opt._pending_best_mirror)
        self.assertEqual(self.mirror_calls, [])
        self.assertIsNone(self.opt._best_valid_dcp)

    def test_no_bank_when_not_improved(self):
        self.measured_wns = -3.0  # worse than best -2.0
        self._call_heavy()
        self.assertEqual(len(self.measure_calls), 1)
        self.assertEqual(self.opt.best_wns, -2.0)
        self.assertEqual(self.mirror_calls, [])

    # ---- budget guard ---------------------------------------------

    def test_measurement_skipped_when_budget_tight(self):
        # Remaining wall below the cheap-measurement floor AT HOOK TIME →
        # the hook must skip the measurement ENTIRELY (no get_wns call),
        # no mirror, no crash.  The boom shape: pre-flight passes with
        # ample budget, then the heavy op itself consumes almost all of
        # it — simulated by the session shrinking the deadline mid-call.
        opt = self.opt

        class _BudgetBurningSession(_FakeSession):
            async def call_tool(self, name, arguments):
                opt._budget_deadline = time.time() + 10.0
                return await super().call_tool(name, arguments)

        self.opt.vivado_session = _BudgetBurningSession(
            response_text=self.OK_TEXT)
        result = self._call_heavy()
        self.assertEqual(result, self.OK_TEXT)
        self.assertEqual(self.measure_calls, [])
        self.assertEqual(self.mirror_calls, [])

    # ---- no double-measurement ------------------------------------

    def test_no_double_measure_on_measurement_tools(self):
        # The existing WNS side-channel already handles these two tool
        # names — the hook must not re-measure after them.
        # target_clock stays None so the side-channel itself never calls
        # get_wns_for_target_clock either.
        self.opt.target_clock = None
        for tool in ("vivado_report_timing_summary", "vivado_get_wns"):
            self._call_heavy(tool_name=tool)
        self.assertEqual(self.measure_calls, [])
        self.assertEqual(self.mirror_calls, [])

    # ---- lifecycle bypasses ---------------------------------------

    def test_noop_in_finalize(self):
        self.opt._in_finalize = True
        self._call_heavy()
        self.assertEqual(self.measure_calls, [])
        self.assertEqual(self.mirror_calls, [])

    def test_noop_in_ils_stage(self):
        self.opt._in_ils_stage = True
        self._call_heavy()
        self.assertEqual(self.measure_calls, [])
        self.assertEqual(self.mirror_calls, [])

    # ---- kill switch ----------------------------------------------

    def test_kill_switch_defaults_on(self):
        fresh = _make_optimizer(Path(self.tmp.name))
        self.assertTrue(fresh._auto_bank_enabled)

    def test_kill_switch_off_is_noop(self):
        # OFF must restore byte-identical pre-change behavior: no
        # measurement, no mirror, result passthrough untouched.
        self.opt._auto_bank_enabled = False
        result = self._call_heavy()
        self.assertEqual(result, self.OK_TEXT)
        self.assertEqual(self.measure_calls, [])
        self.assertEqual(self.mirror_calls, [])
        self.assertEqual(self.opt.best_wns, -2.0)

    # ---- scope of the heavy-op classifier -------------------------

    def test_non_heavy_tool_does_not_fire(self):
        # write_checkpoint is not in RISKY_VIVADO_TOOLS — the hook must
        # not fire (this is also what makes the mirror's own recursive
        # write_checkpoint/write_edif calls safe from re-entry).
        self._call_heavy(tool_name="vivado_write_checkpoint",
                         arguments={"dcp_path": str(Path(self.tmp.name) / "x.dcp")})
        self.assertEqual(self.measure_calls, [])
        self.assertEqual(self.mirror_calls, [])

    def test_risky_run_tcl_fires_hook(self):
        # Heavy work embedded in Tcl (the dominant recipe path) must be
        # auto-banked too — classification reuses _is_risky.
        self._call_heavy(tool_name="vivado_run_tcl",
                         arguments={"command": "route_design -directive AggressiveExplore"})
        self.assertEqual(len(self.measure_calls), 1)
        self.assertEqual(self.mirror_calls, [True])

    # ---- failure containment --------------------------------------

    def test_tool_error_result_skips_hook(self):
        # A failed heavy op (error envelope / TCL ERROR text) must not be
        # measured or banked.
        self.opt.vivado_session.response_text = (
            'TCL ERROR: ERROR: [Route 35-7] Router failed')
        self._call_heavy()
        self.assertEqual(self.measure_calls, [])
        self.assertEqual(self.mirror_calls, [])

    def test_budget_skipped_dispatch_never_reaches_hook(self):
        # Pre-flight budget skip returns the skip envelope before the op
        # ever runs — "heavy op returned OK" is false, so no measurement.
        self.opt._budget_deadline = time.time() - 1.0  # deadline passed
        result = self._call_heavy()
        self.assertIn("tool_skipped_budget", result)
        self.assertEqual(self.measure_calls, [])
        self.assertEqual(self.mirror_calls, [])

    def test_exception_dispatch_never_reaches_hook(self):
        # A raising session takes the exception path (error envelope
        # return) — the hook must not fire on a failed op.
        class _RaisingSession(_FakeSession):
            async def call_tool(self, name, arguments):
                raise RuntimeError("session died mid-call")

        self.opt.vivado_session = _RaisingSession()
        result = self._call_heavy()
        self.assertIn("error", result)
        self.assertEqual(self.measure_calls, [])
        self.assertEqual(self.mirror_calls, [])

    def test_all_dedicated_heavy_tools_fire_hook(self):
        # place/route/phys_opt are ALL heavy mutating ops (R-D1-1 says
        # "ANY heavy mutating tool"); each must trigger a measurement.
        for tool in ("vivado_place_design", "vivado_route_design",
                     "vivado_phys_opt_design"):
            self.measure_calls.clear()
            # S2 fix (jul20): double-blind gate now refuses with no cell
            # data — provide a small count so place_design stays feasible
            # (this test's subject is the hook, not the gate).
            self.opt._input_cell_count = 10_000
            self._call_heavy(tool_name=tool)
            self.assertEqual(len(self.measure_calls), 1,
                             f"{tool} must trigger the auto-bank measurement")

    def test_hook_exception_never_crashes_call(self):
        # Banking must never crash the run — a raising measurement is
        # swallowed and the tool result still returns.
        async def boom(call_tool_fn):
            raise RuntimeError("dead session")

        self.opt.get_wns_for_target_clock = boom
        result = self._call_heavy()
        self.assertEqual(result, self.OK_TEXT)
        self.assertEqual(self.mirror_calls, [])
        self.assertEqual(self.opt.best_wns, -2.0)


if __name__ == "__main__":
    unittest.main()
