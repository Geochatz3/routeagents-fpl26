"""Test wall-clock reservation for pending post-route polish.

The reserve is armed after a routed best is banked while enabled polish stages
remain pending. Speculative operations that destroy routed state must fit
within the remaining time minus this reserve, while polishing, banking, and
finalization retain the full remaining window. The reserve is released after
polish runs or is correctly skipped, and always during finalization. All
calculations occur inside the existing 300-second finalization reserve and
remain independent of monetary cost limits.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import (
    DCPOptimizer,
    POLISH_RESERVE_S_DEFAULT,
    resolve_polish_reserve_s,
)
from optimizer.ils_polish import ILSPolishConfig, ILSPolishResult


def _async(coro):
    return asyncio.run(coro)


class _FakeContent:
    def __init__(self, text: str):
        self.text = text


class _FakeResult:
    def __init__(self, text: str):
        self.content = [_FakeContent(text)]


class _FakeSession:
    """Stand-in for the MCP ClientSession used by DCPOptimizer.call_tool."""

    def __init__(self, response_text: str = "{}"):
        self.response_text = response_text
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        return _FakeResult(self.response_text)


def _make_optimizer(tmp_path: Path) -> DCPOptimizer:
    opt = DCPOptimizer(api_key="test", run_dir=tmp_path)
    opt.vivado_session = _FakeSession()
    opt.rapidwright_session = _FakeSession()
    return opt


def _arm(opt: DCPOptimizer, tmp: Path, remaining_s: float,
         reserve_s: float = POLISH_RESERVE_S_DEFAULT) -> Path:
    """Put the optimizer into the canonical ARMED state: contest budget,
    routed banked best, polish stages enabled+pending."""
    opt.max_wall_seconds = 3300.0
    opt._budget_deadline = time.time() + remaining_s
    opt._polish_reserve_s = reserve_s
    banked = tmp / "best_valid.dcp"
    banked.write_bytes(b"dcp")
    opt._best_valid_dcp = banked
    opt._best_valid_dcp_wns = -0.5
    opt._ils_polish_cfg = ILSPolishConfig(enabled=True)
    return banked


class ResolverTests(unittest.TestCase):
    """resolve_polish_reserve_s — CLI wins over env; 0 = off; invalid
    keeps default."""

    def test_unset_returns_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FPL26_POLISH_RESERVE_S", None)
            self.assertEqual(resolve_polish_reserve_s(None),
                             POLISH_RESERVE_S_DEFAULT)

    def test_default_is_500(self):
        self.assertEqual(POLISH_RESERVE_S_DEFAULT, 500.0)

    def test_cli_wins_over_env(self):
        with mock.patch.dict(os.environ,
                             {"FPL26_POLISH_RESERVE_S": "250"}):
            self.assertEqual(resolve_polish_reserve_s(400.0), 400.0)

    def test_env_fallback(self):
        with mock.patch.dict(os.environ,
                             {"FPL26_POLISH_RESERVE_S": "250"}):
            self.assertEqual(resolve_polish_reserve_s(None), 250.0)

    def test_zero_disables(self):
        self.assertEqual(resolve_polish_reserve_s(0), 0.0)
        with mock.patch.dict(os.environ, {"FPL26_POLISH_RESERVE_S": "0"}):
            self.assertEqual(resolve_polish_reserve_s(None), 0.0)

    def test_negative_keeps_default(self):
        self.assertEqual(resolve_polish_reserve_s(-100.0),
                         POLISH_RESERVE_S_DEFAULT)

    def test_garbage_env_keeps_default(self):
        with mock.patch.dict(os.environ,
                             {"FPL26_POLISH_RESERVE_S": "banana"}):
            self.assertEqual(resolve_polish_reserve_s(None),
                             POLISH_RESERVE_S_DEFAULT)

    def test_garbage_cli_keeps_default(self):
        self.assertEqual(resolve_polish_reserve_s("banana"),
                         POLISH_RESERVE_S_DEFAULT)


class ArmingGuardTests(unittest.TestCase):
    """Never-worse: the reserve arms ONLY when a polish opportunity
    exists — every unarmed case must behave byte-identically to today."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.opt = _make_optimizer(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_armed_when_all_conditions_hold(self):
        _arm(self.opt, self.tmp, remaining_s=2000.0)
        self.assertEqual(self.opt._polish_reserve_armed_s(), 500.0)

    def test_unarmed_without_budget_deadline(self):
        _arm(self.opt, self.tmp, remaining_s=2000.0)
        self.opt._budget_deadline = None  # dev mode
        self.assertEqual(self.opt._polish_reserve_armed_s(), 0.0)

    def test_unarmed_without_routed_banked_best(self):
        _arm(self.opt, self.tmp, remaining_s=2000.0)
        self.opt._best_valid_dcp = None
        self.assertEqual(self.opt._polish_reserve_armed_s(), 0.0)

    def test_unarmed_when_reserve_disabled(self):
        _arm(self.opt, self.tmp, remaining_s=2000.0, reserve_s=0.0)
        self.assertEqual(self.opt._polish_reserve_armed_s(), 0.0)

    def test_unarmed_when_ils_stage_disabled(self):
        # No --ils-polish -> the LASTMILE/fanout stages can never run;
        # reserving for them would strand wall time.
        _arm(self.opt, self.tmp, remaining_s=2000.0)
        self.opt._ils_polish_cfg = ILSPolishConfig(enabled=False)
        self.assertEqual(self.opt._polish_reserve_armed_s(), 0.0)

    def test_unarmed_when_both_polish_ladder_flag_disabled(self):
        _arm(self.opt, self.tmp, remaining_s=2000.0)
        self.opt._ils_polish_cfg = ILSPolishConfig(
            enabled=True,
            lastmile_polish_enabled=False,
            fanout_polish_enabled=False)
        self.assertEqual(self.opt._polish_reserve_armed_s(), 0.0)

    def test_unarmed_after_release(self):
        _arm(self.opt, self.tmp, remaining_s=2000.0)
        self.opt._release_polish_reserve("test")
        self.assertEqual(self.opt._polish_reserve_armed_s(), 0.0)

    def test_release_is_one_way_first_reason_wins(self):
        _arm(self.opt, self.tmp, remaining_s=2000.0)
        self.opt._release_polish_reserve("first")
        self.opt._release_polish_reserve("second")
        self.assertEqual(self.opt._polish_reserve_release_reason, "first")


class BoomShapeReplayTests(unittest.TestCase):
    """Exercise a late-budget refusal scenario with and without polish
    reservation.

    Without a reserve, a speculative operation may consume the time needed for
    final polish. With the reserve armed, that operation is refused non-fatally
    while state-preserving polish remains affordable against the full remaining
    window.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.opt = _make_optimizer(self.tmp)
        # Isolate the reserve gate from the (orthogonal) unroute gate.
        self.opt._unroute_gate_enabled = False
        self.opt._auto_bank_enabled = False

    def tearDown(self):
        self._tmp.cleanup()

    def test_without_reserve_polish_refused_at_443s(self):
        # Today's boom shape: est 600s (DEFAULT_RISKY_RUNTIME_S, no
        # history) vs 443s remaining -> budget skip, run budget-killed.
        _arm(self.opt, self.tmp, remaining_s=443.0, reserve_s=0.0)
        payload = _async(self.opt.call_tool("vivado_phys_opt_design", {}))
        data = json.loads(payload)
        self.assertEqual(data["error"], "tool_skipped_budget")
        self.assertIn("estimated_600s_exceeds_remaining_44", data["reason"])
        self.assertTrue(self.opt._budget_killed)
        self.assertEqual(self.opt.vivado_session.calls, [])

    def test_with_reserve_speculative_gamble_refused_polish_affordable(self):
        # Reserve armed earlier (banked best exists), 1050s remaining.
        _arm(self.opt, self.tmp, remaining_s=1050.0, reserve_s=500.0)
        # A destructive operation that exceeds the speculative window is
        # refused before launch, preserving the polish reserve.
        payload = _async(self.opt.call_tool("vivado_place_design", {}))
        data = json.loads(payload)
        self.assertEqual(data["error"], "polish_reserve_refused")
        self.assertIn("exceeds_speculative_window", data["reason"])
        self.assertIn("steering", data)
        self.assertFalse(self.opt._budget_killed)
        self.assertEqual(self.opt.vivado_session.calls, [])
        # 2) The post-route phys_opt polish (state-PRESERVING) sees the
        #    FULL remaining window: est 600 <= 1050 -> dispatched.
        _async(self.opt.call_tool("vivado_phys_opt_design", {}))
        called = [name for name, _ in self.opt.vivado_session.calls]
        self.assertIn("phys_opt_design", called)

    def test_speculative_op_fitting_outside_reserve_still_runs(self):
        # est 600 <= (1200 - 500): the reserve never blocks a gamble
        # that leaves the polish window intact.
        _arm(self.opt, self.tmp, remaining_s=1200.0, reserve_s=500.0)
        _async(self.opt.call_tool("vivado_place_design", {}))
        called = [name for name, _ in self.opt.vivado_session.calls]
        self.assertIn("place_design", called)

    def test_unarmed_destroying_op_runs_exactly_as_today(self):
        # Never-worse guard: same op, same window, but NO banked best ->
        # reserve unarmed -> dispatched exactly as today.
        _arm(self.opt, self.tmp, remaining_s=1050.0, reserve_s=500.0)
        self.opt._best_valid_dcp = None
        _async(self.opt.call_tool("vivado_place_design", {}))
        called = [name for name, _ in self.opt.vivado_session.calls]
        self.assertIn("place_design", called)

    def test_released_reserve_no_longer_blocks(self):
        _arm(self.opt, self.tmp, remaining_s=1050.0, reserve_s=500.0)
        self.opt._release_polish_reserve("ils_polish_ladder_reached")
        _async(self.opt.call_tool("vivado_place_design", {}))
        called = [name for name, _ in self.opt.vivado_session.calls]
        self.assertIn("place_design", called)

    def test_ils_stage_bypasses_reserve_gate(self):
        # The polish stages themselves (ILS bypass) must never be
        # refused by the reserve they are consuming.
        _arm(self.opt, self.tmp, remaining_s=1050.0, reserve_s=500.0)
        self.opt._in_ils_stage = True
        try:
            _async(self.opt.call_tool("vivado_place_design", {}))
        finally:
            self.opt._in_ils_stage = False
        called = [name for name, _ in self.opt.vivado_session.calls]
        self.assertIn("place_design", called)

    def test_finalize_bypasses_reserve_gate(self):
        _arm(self.opt, self.tmp, remaining_s=1050.0, reserve_s=500.0)
        self.opt._in_finalize = True
        try:
            _async(self.opt.call_tool("vivado_place_design", {}))
        finally:
            self.opt._in_finalize = False
        called = [name for name, _ in self.opt.vivado_session.calls]
        self.assertIn("place_design", called)


class DeadlineTimeoutCapTests(unittest.TestCase):
    """Runtime enforcement: an armed reserve caps the asyncio timeout of
    speculative destroying ops at (remaining − reserve); everything else
    keeps the full remaining window."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.opt = _make_optimizer(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_destroying_op_capped_at_fence(self):
        _arm(self.opt, self.tmp, remaining_s=1000.0, reserve_s=500.0)
        t = self.opt._deadline_aware_timeout("vivado_place_design", None)
        self.assertLessEqual(t, 501.0)
        self.assertGreater(t, 480.0)

    def test_preserving_op_keeps_full_window(self):
        _arm(self.opt, self.tmp, remaining_s=1000.0, reserve_s=500.0)
        t = self.opt._deadline_aware_timeout("vivado_phys_opt_design", None)
        self.assertGreater(t, 980.0)

    def test_anonymous_call_keeps_full_window(self):
        # ILS-stage call sites pass tool_name=None (they are fenced by
        # the reduced deadline_ts instead) — never capped here.
        _arm(self.opt, self.tmp, remaining_s=1000.0, reserve_s=500.0)
        t = self.opt._deadline_aware_timeout()
        self.assertGreater(t, 980.0)

    def test_unarmed_destroying_op_keeps_full_window(self):
        _arm(self.opt, self.tmp, remaining_s=1000.0, reserve_s=500.0)
        self.opt._best_valid_dcp = None
        t = self.opt._deadline_aware_timeout("vivado_place_design", None)
        self.assertGreater(t, 980.0)


class FinalizeReserveLayeringTests(unittest.TestCase):
    """The polish reserve sits ON TOP of the 300s finalize reserve:
    _budget_deadline already excludes the finalize tail, and the
    speculative window subtracts the polish reserve from THAT."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.opt = _make_optimizer(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_speculative_window_stacks_both_reserves(self):
        # Emulate optimize(): soft deadline = start + max_wall − 300.
        self.opt.max_wall_seconds = 3300.0
        start = time.time()
        self.opt._budget_deadline = (
            start + self.opt.max_wall_seconds
            - self.opt._finalize_reserve_seconds)
        _arm(self.opt, self.tmp,
             remaining_s=self.opt._budget_deadline - time.time(),
             reserve_s=500.0)
        self.opt._budget_deadline = (
            start + self.opt.max_wall_seconds
            - self.opt._finalize_reserve_seconds)  # _arm overwrote it
        remaining = self.opt._budget_remaining()
        # remaining is already finalize-protected (3000s, not 3300s)…
        self.assertLess(remaining, 3001.0)
        self.assertGreater(remaining, 2980.0)
        # …and the destroying-op window subtracts the polish reserve
        # from that: ~2500s, i.e. 3300 − 300 (finalize) − 500 (polish).
        t = self.opt._deadline_aware_timeout("vivado_place_design", None)
        self.assertLess(t, 2501.0)
        self.assertGreater(t, 2480.0)

    def test_finalize_entry_releases_reserve(self):
        _arm(self.opt, self.tmp, remaining_s=1000.0, reserve_s=500.0)
        with mock.patch.object(self.opt, "_finalize_output_dcp_impl",
                               new=mock.AsyncMock(return_value=None)):
            _async(self.opt._finalize_output_dcp(self.tmp / "out.dcp"))
        self.assertEqual(self.opt._polish_reserve_release_reason, "finalize")


class IlsStageFencingTests(unittest.TestCase):
    """Inside the ILS stage: new ruin cycles (speculative) run against
    (deadline − reserve); the reserve releases when the polish
    sub-stages are reached, or immediately when the design class is
    dynamically polish-INELIGIBLE (LASTMILE wns entry gate + fanout
    cheap-design anchor gate — the stages' own gates, reused)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.opt = _make_optimizer(self.tmp)
        self.opt.max_wall_seconds = 3300.0
        self.opt._budget_deadline = time.time() + 3000.0
        self.opt._polish_reserve_s = 500.0
        banked = self.tmp / "best_valid.dcp"
        banked.write_bytes(b"dcp")
        self.opt._best_valid_dcp = banked
        self.opt._best_valid_dcp_wns = -0.5
        self.opt.initial_wns = -0.6
        self.opt.best_wns = -0.5
        self.opt._ils_polish_cfg = ILSPolishConfig(
            enabled=True, restart_vivado_before_ils=False)
        self.captured: list[float] = []

    def tearDown(self):
        self._tmp.cleanup()

    def _fake_run_ils_polish(self):
        captured = self.captured

        async def fake(call_tool, *, best_dcp_path, baseline_wns,
                       deadline_ts, wns_tcl, cfg, log, no_improve_stop,
                       combo_offset, combo_cost_seed):
            captured.append(deadline_ts)
            return ILSPolishResult(triggered=True, improved=False,
                                   cycles=1)
        return fake

    def test_armed_ils_cycles_fenced_and_release_at_polish_ladder(self):
        # LASTMILE plausible (wns −0.5 ≥ −1.0 entry gate) -> reserve
        # holds through the ruin cycles.
        _async(self.opt._ils_polish_body(self._fake_run_ils_polish()))
        self.assertEqual(len(self.captured), 1)
        fenced = self.opt._budget_deadline - 500.0
        self.assertAlmostEqual(self.captured[0], fenced, delta=5.0)
        self.assertEqual(self.opt._polish_reserve_release_reason,
                         "ils_polish_ladder_reached")

    def test_polish_ineligible_design_releases_before_cycles(self):
        # boom-class: wns −2.5 fails the LASTMILE entry gate AND no
        # affordable fanout anchor -> reserve released up front, ruin
        # cycles keep the FULL deadline (unreserved behavior, no stranding).
        self.opt.best_wns = -2.5
        self.opt._best_valid_dcp_wns = -2.5
        _async(self.opt._ils_polish_body(self._fake_run_ils_polish()))
        self.assertEqual(len(self.captured), 1)
        self.assertAlmostEqual(self.captured[0],
                               self.opt._budget_deadline, delta=5.0)
        self.assertTrue(str(self.opt._polish_reserve_release_reason)
                        .startswith("ils_polish_ineligible"))

    def test_disabled_reserve_never_fences_ils(self):
        self.opt._polish_reserve_s = 0.0
        _async(self.opt._ils_polish_body(self._fake_run_ils_polish()))
        self.assertEqual(len(self.captured), 1)
        self.assertAlmostEqual(self.captured[0],
                               self.opt._budget_deadline, delta=5.0)


if __name__ == "__main__":
    unittest.main()
