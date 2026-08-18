"""DEEP-WNS TAIL RESERVE tests (jul22 PLAN item 5 + jul23 V2 predicate).

Mechanism under test (dcp_optimizer.py): when --deep-wns-tail-reserve /
FPL26_DEEP_WNS_TAIL_RESERVE > 0 AND the design is STILL deep-WNS at the
boundary (CURRENT best_wns <= _tail_ctrl_deep_wns_ns, default -1.0), the
main LLM loop breaks once remaining wall <= reserve with
loop_exit_reason=tail_reserve, exiting through the SHARED
_exit_with_ils_polish tail like every other loop exit.

Evidence being encoded (jul22 farm wave, leg_boom at eval speed): the
LLM/recipe loop ran the ENTIRE wall to budget_exhausted (-35 s remaining
at exit) so the deterministic exit tail got ZERO seconds (wall-fit gate
refused 886 s vs -95 s effective); the only eval tail fire ever (+0.215
banked, jul21) came from an accidental early loop exit.

V2 (jul23 panel Q1b, 3-2 V2_BOOM_ONLY; farm waves 2+3 evidence):
  - WALL CLAMP: effective reserve = min(requested,
    TAIL_RESERVE_WALL_CLAMP_FRAC x wall) when the wall is known
    (0.686 = 2400/3500, the wave-2 proven arm; wave-3: reserve >= wall
    truncated ispd16/boom recipes to baseline), loud WARNING on clamp;
  - STAGNATION GUARD: break only when no best_wns improvement in the
    last FPL26_TAIL_RESERVE_STAGNANT_S seconds (default 240; 0 = off)
    — wave-3 failed by exiting MID-recipe-improvement;
  - SIZE-GATED-ONLY ARMING: _input_cell_count must exceed the LIVE ILS
    max_cells gate (wave-2 L3-vs-L5: ILS-path control beat the reserve
    on vtr); unknown cell count -> NOT armed (fail-safe).

Load-bearing invariants:
  - DEFAULT OFF (0/unset): zero behavior change on every existing path;
  - resolver follows the house convention (CLI wins over env;
    unparseable/negative keeps the default = OFF);
  - deep-WNS check uses the CURRENT best_wns at the boundary, not entry
    WNS (recipe-fixed near-met designs must NOT early-exit);
  - the early exit goes through the SHARED exit tail
    (_exit_with_ils_polish) with a distinct "[tail-reserve]" reason line;
  - a value in (0, 1) is a fraction of --max-wall-seconds.

Harness style mirrors tests/test_tail_controller.py +
tests/test_bare_reroute_polish.py (+ the _run_optimize loop driver from
tests/test_api_resilience_loop.py): stubbed sessions / scripted
get_completion, NO Vivado, NO RapidWright, NO network, NO real sleeps.
"""
from __future__ import annotations

import asyncio
import io
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import (
    DEEP_WNS_TAIL_RESERVE_S_DEFAULT,
    DEEP_WNS_TAIL_RESERVE_S_RECOMMENDED,
    TAIL_CTRL_DEEP_WNS_NS_DEFAULT,
    TAIL_RESERVE_STAGNANT_S_DEFAULT,
    TAIL_RESERVE_WALL_CLAMP_FRAC,
    DCPOptimizer,
    resolve_deep_wns_tail_reserve_s,
    resolve_tail_reserve_stagnant_s,
)

_ENV = "FPL26_DEEP_WNS_TAIL_RESERVE"
_ENV_STAGNANT = "FPL26_TAIL_RESERVE_STAGNANT_S"

# ILS-size-gated fixture cell counts (live gate default is 300_000 in
# optimizer/ils_polish.py DEFAULT_MAX_CELLS — read from the cfg object,
# never hardcoded in the predicate).
_CELLS_SIZE_GATED = 400_000   # > max_cells: boom-class, reserve may arm
_CELLS_ILS_PATH = 100_000     # <= max_cells: ILS-path, reserve must NOT arm


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


# ---------------------------------------------------------------------------
# Resolver convention (house style: CLI wins over env; invalid keeps default)
# ---------------------------------------------------------------------------

class ResolverTests(unittest.TestCase):

    def setUp(self):
        # Isolate from any ambient env configuration.
        self._env_patcher = mock.patch.dict(os.environ)
        self._env_patcher.start()
        os.environ.pop(_ENV, None)

    def tearDown(self):
        self._env_patcher.stop()

    def test_default_is_off(self):
        self.assertEqual(DEEP_WNS_TAIL_RESERVE_S_DEFAULT, 0.0)
        self.assertEqual(resolve_deep_wns_tail_reserve_s(None), 0.0)

    def test_recommended_constant_documents_two_tail_moves(self):
        # ~2 tail moves at eval speed: bare route 850-1600 s + phys_opt
        # 400-770 s + banking margins (jul22_loop_dw op costs).
        self.assertEqual(DEEP_WNS_TAIL_RESERVE_S_RECOMMENDED, 2400.0)

    def test_cli_seconds(self):
        self.assertEqual(resolve_deep_wns_tail_reserve_s(2400), 2400.0)

    def test_cli_zero_is_explicit_off(self):
        self.assertEqual(resolve_deep_wns_tail_reserve_s(0), 0.0)

    def test_env_seconds(self):
        os.environ[_ENV] = "1800"
        self.assertEqual(resolve_deep_wns_tail_reserve_s(None), 1800.0)

    def test_cli_wins_over_env(self):
        os.environ[_ENV] = "1800"
        self.assertEqual(resolve_deep_wns_tail_reserve_s(600.0), 600.0)

    def test_fraction_passes_through(self):
        self.assertEqual(resolve_deep_wns_tail_reserve_s(0.5), 0.5)
        os.environ[_ENV] = "0.25"
        self.assertEqual(resolve_deep_wns_tail_reserve_s(None), 0.25)

    def test_negative_keeps_default_off(self):
        self.assertEqual(resolve_deep_wns_tail_reserve_s(-5), 0.0)
        os.environ[_ENV] = "-100"
        self.assertEqual(resolve_deep_wns_tail_reserve_s(None), 0.0)

    def test_unparseable_env_keeps_default_off(self):
        os.environ[_ENV] = "junk"
        self.assertEqual(resolve_deep_wns_tail_reserve_s(None), 0.0)

    def test_unparseable_cli_keeps_default_off(self):
        self.assertEqual(resolve_deep_wns_tail_reserve_s("junk"), 0.0)

    def test_empty_env_keeps_default_off(self):
        os.environ[_ENV] = ""
        self.assertEqual(resolve_deep_wns_tail_reserve_s(None), 0.0)


# ---------------------------------------------------------------------------
# V2 resolver: stagnation-guard window (env-only knob, house convention)
# ---------------------------------------------------------------------------

class StagnantResolverTests(unittest.TestCase):

    def setUp(self):
        self._env_patcher = mock.patch.dict(os.environ)
        self._env_patcher.start()
        os.environ.pop(_ENV_STAGNANT, None)

    def tearDown(self):
        self._env_patcher.stop()

    def test_default_240(self):
        self.assertEqual(TAIL_RESERVE_STAGNANT_S_DEFAULT, 240.0)
        self.assertEqual(resolve_tail_reserve_stagnant_s(), 240.0)

    def test_env_overrides(self):
        os.environ[_ENV_STAGNANT] = "120"
        self.assertEqual(resolve_tail_reserve_stagnant_s(), 120.0)

    def test_env_zero_disables_guard(self):
        os.environ[_ENV_STAGNANT] = "0"
        self.assertEqual(resolve_tail_reserve_stagnant_s(), 0.0)

    def test_unparseable_keeps_default(self):
        os.environ[_ENV_STAGNANT] = "junk"
        self.assertEqual(resolve_tail_reserve_stagnant_s(), 240.0)

    def test_negative_keeps_default(self):
        os.environ[_ENV_STAGNANT] = "-10"
        self.assertEqual(resolve_tail_reserve_stagnant_s(), 240.0)

    def test_constructor_picks_up_env(self):
        os.environ[_ENV_STAGNANT] = "90"
        with tempfile.TemporaryDirectory() as tmp:
            opt = _make_optimizer(Path(tmp))
            self.assertEqual(opt._tail_reserve_stagnant_s, 90.0)


# ---------------------------------------------------------------------------
# Predicate: _tail_reserve_break_due / _deep_wns_tail_reserve_effective_s
# ---------------------------------------------------------------------------

class PredicateTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = _make_optimizer(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def _arm(self, reserve: float, *, remaining: float = 500.0,
             best_wns: float = -2.0, max_wall: float | None = 3600.0,
             cells: int | None = _CELLS_SIZE_GATED,
             stagnant: bool = True):
        """V2 fixture defaults: size-gated design (cells > ILS max_cells)
        and a STAGNANT loop (last improvement far in the past) so the
        pre-V2 predicate dimensions stay independently testable."""
        self.opt._deep_wns_tail_reserve_s = reserve
        self.opt.max_wall_seconds = max_wall
        self.opt._budget_deadline = time.time() + remaining
        self.opt.best_wns = best_wns
        self.opt._input_cell_count = cells
        self.opt.last_improvement_time = (
            time.time() - 9999.0 if stagnant else time.time())

    def test_constructor_default_off(self):
        self.assertEqual(self.opt._deep_wns_tail_reserve_s, 0.0)
        self.assertFalse(self.opt._tail_reserve_break_due())

    def test_off_never_breaks_even_at_boundary(self):
        # Deep WNS + boundary hit, but reserve OFF (default): no break —
        # zero behavior change with the flag off.
        self._arm(0.0, remaining=100.0, best_wns=-5.0)
        self.assertFalse(self.opt._tail_reserve_break_due())

    def test_on_deep_at_boundary_breaks(self):
        self._arm(2400.0, remaining=1000.0, best_wns=-2.0)
        self.assertTrue(self.opt._tail_reserve_break_due())

    def test_on_deep_with_room_does_not_break(self):
        self._arm(2400.0, remaining=3000.0, best_wns=-2.0)
        self.assertFalse(self.opt._tail_reserve_break_due())

    def test_shallow_wns_never_breaks(self):
        # CURRENT best_wns owns the decision: a design recipe-fixed to
        # near-met (-0.5 > -1.0 threshold) stays with the LLM/ILS even
        # though the boundary was hit.
        self._arm(2400.0, remaining=1000.0, best_wns=-0.5)
        self.assertFalse(self.opt._tail_reserve_break_due())

    def test_threshold_is_inclusive(self):
        self._arm(2400.0, remaining=1000.0,
                  best_wns=TAIL_CTRL_DEEP_WNS_NS_DEFAULT)
        self.assertTrue(self.opt._tail_reserve_break_due())

    def test_unmeasured_wns_never_breaks(self):
        # best_wns == -inf: nothing banked/measured — the tail would
        # no-op and the early exit would waste the window.
        self._arm(2400.0, remaining=1000.0, best_wns=float("-inf"))
        self.assertFalse(self.opt._tail_reserve_break_due())

    def test_no_deadline_never_breaks(self):
        self._arm(2400.0, best_wns=-2.0)
        self.opt._budget_deadline = None
        self.assertFalse(self.opt._tail_reserve_break_due())

    def test_effective_seconds_absolute(self):
        self._arm(2400.0)
        self.assertEqual(self.opt._deep_wns_tail_reserve_effective_s(),
                         2400.0)

    def test_effective_seconds_fraction_of_wall(self):
        self._arm(0.5, max_wall=3600.0)
        self.assertEqual(self.opt._deep_wns_tail_reserve_effective_s(),
                         1800.0)

    def test_fraction_break_at_boundary(self):
        self._arm(0.5, remaining=1000.0, best_wns=-2.0, max_wall=3600.0)
        self.assertTrue(self.opt._tail_reserve_break_due())

    def test_fraction_no_break_with_room(self):
        self._arm(0.5, remaining=2500.0, best_wns=-2.0, max_wall=3600.0)
        self.assertFalse(self.opt._tail_reserve_break_due())

    def test_fraction_without_wall_cap_is_off(self):
        # No max_wall_seconds -> the fraction has no base; reserve is
        # inert even if a deadline was set some other way.
        self._arm(0.5, remaining=100.0, best_wns=-2.0, max_wall=None)
        self.assertEqual(self.opt._deep_wns_tail_reserve_effective_s(), 0.0)
        self.assertFalse(self.opt._tail_reserve_break_due())

    # --- V2 wall clamp -----------------------------------------------------

    def test_clamp_math_absolute_reserve(self):
        # 5000s requested on a 3500s wall -> clamped to 0.686 x 3500 =
        # exactly 2400 (the wave-2 proven arm, by construction of the
        # fraction 2400/3500).
        self._arm(5000.0, max_wall=3500.0)
        self.assertAlmostEqual(
            self.opt._deep_wns_tail_reserve_effective_s(), 2400.0, places=6)

    def test_clamp_math_fractional_reserve(self):
        # 0.9 x wall exceeds the clamp fraction -> clamped identically.
        self._arm(0.9, max_wall=3500.0)
        self.assertAlmostEqual(
            self.opt._deep_wns_tail_reserve_effective_s(), 2400.0, places=6)

    def test_no_clamp_below_cap(self):
        self._arm(2400.0, max_wall=3500.0)
        self.assertEqual(self.opt._deep_wns_tail_reserve_effective_s(),
                         2400.0)

    def test_clamp_unknown_wall_absolute_unclamped(self):
        # Wall unknown: absolute-seconds reserve stays as-is (the
        # predicate is already inert without a deadline).
        self._arm(5000.0, max_wall=None)
        self.assertEqual(self.opt._deep_wns_tail_reserve_effective_s(),
                         5000.0)

    def test_clamp_logs_loud_warning_once(self):
        self._arm(5000.0, max_wall=3500.0)
        with self.assertLogs("dcp_optimizer", level="WARNING") as logs:
            self.opt._deep_wns_tail_reserve_effective_s()
            self.opt._deep_wns_tail_reserve_effective_s()  # 2nd poll: quiet
        clamp_lines = [l for l in logs.output
                       if "[tail-reserve] WALL CLAMP" in l]
        self.assertEqual(len(clamp_lines), 1)
        self.assertIn("clamping", clamp_lines[0])

    def test_no_clamp_no_warning(self):
        self._arm(2400.0, max_wall=3500.0)
        with self.assertNoLogs("dcp_optimizer", level="WARNING"):
            self.opt._deep_wns_tail_reserve_effective_s()

    def test_clamped_reserve_still_breaks_inside_clamped_window(self):
        # Requested 5000s (>= wall would have broken IMMEDIATELY after
        # the first bank — the wave-3 recipe-truncation failure); with
        # the clamp the boundary is 2400s: remaining 3000 must NOT
        # break, remaining 2000 must.
        self._arm(5000.0, remaining=3000.0, max_wall=3500.0)
        self.assertFalse(self.opt._tail_reserve_break_due())
        self._arm(5000.0, remaining=2000.0, max_wall=3500.0)
        self.assertTrue(self.opt._tail_reserve_break_due())

    # --- V2 size gate (ILS-size-gated class only) --------------------------

    def test_size_gate_blocks_ils_path_design(self):
        # cells <= ILS max_cells -> ILS-path class: wave-2 L3-vs-L5 (vtr
        # control beat the reserve) — never arm.
        self._arm(2400.0, remaining=1000.0, cells=_CELLS_ILS_PATH)
        self.assertFalse(self.opt._tail_reserve_break_due())

    def test_size_gate_exactly_at_max_cells_blocks(self):
        # Gate is strictly greater-than (mirrors ILS's own "too_large"
        # cells > max_cells test).
        self._arm(2400.0, remaining=1000.0,
                  cells=self.opt._ils_polish_cfg.max_cells)
        self.assertFalse(self.opt._tail_reserve_break_due())

    def test_unknown_cell_count_blocks(self):
        # Fail-safe: no measured cell count -> NOT armed.
        self._arm(2400.0, remaining=1000.0, cells=None)
        self.assertFalse(self.opt._tail_reserve_break_due())

    def test_size_gate_reads_live_cfg_not_hardcoded(self):
        # The predicate must consult the LIVE ILS cfg object: raising
        # max_cells above the design size flips the verdict.
        self._arm(2400.0, remaining=1000.0, cells=_CELLS_SIZE_GATED)
        self.assertTrue(self.opt._tail_reserve_break_due())
        self.opt._ils_polish_cfg.max_cells = _CELLS_SIZE_GATED + 1
        self.assertFalse(self.opt._tail_reserve_break_due())

    # --- V2 stagnation guard ------------------------------------------------

    def test_recent_improvement_blocks_exit(self):
        # Loop improved just now (mid-recipe-improvement, the wave-3
        # failure mode): boundary hit but NO break.
        self._arm(2400.0, remaining=1000.0, stagnant=False)
        self.assertFalse(self.opt._tail_reserve_break_due())

    def test_stagnant_loop_allows_exit(self):
        self._arm(2400.0, remaining=1000.0, stagnant=True)
        self.assertTrue(self.opt._tail_reserve_break_due())

    def test_stagnation_window_boundary(self):
        # Just inside the default 240s window -> blocked; just past it
        # -> stagnant -> break.
        self._arm(2400.0, remaining=1000.0)
        self.opt.last_improvement_time = time.time() - 100.0
        self.assertFalse(self.opt._tail_reserve_break_due())
        self.opt.last_improvement_time = (
            time.time() - TAIL_RESERVE_STAGNANT_S_DEFAULT - 1.0)
        self.assertTrue(self.opt._tail_reserve_break_due())

    def test_stagnant_zero_disables_guard(self):
        self._arm(2400.0, remaining=1000.0, stagnant=False)
        self.opt._tail_reserve_stagnant_s = 0.0
        self.assertTrue(self.opt._tail_reserve_break_due())

    def test_no_improvement_clock_blocks(self):
        # Neither last_improvement_time nor start_time available:
        # stagnation unprovable -> fail-safe NO break.
        self._arm(2400.0, remaining=1000.0)
        self.opt.last_improvement_time = None
        self.opt.start_time = None
        self.assertFalse(self.opt._tail_reserve_break_due())

    def test_start_time_is_stagnation_fallback_clock(self):
        # No banked improvement yet: start_time anchors the window
        # (matches the ILS preempt convention).
        self._arm(2400.0, remaining=1000.0)
        self.opt.last_improvement_time = None
        self.opt.start_time = time.time() - 9999.0
        self.assertTrue(self.opt._tail_reserve_break_due())


# ---------------------------------------------------------------------------
# Loop integration: the break exits through the SHARED exit tail
# (_run_optimize driver mirrored from tests/test_api_resilience_loop.py)
# ---------------------------------------------------------------------------

class LoopIntegrationTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _run_optimize(self, *, reserve: float, max_wall: float,
                      best_wns: float,
                      cells: int | None = _CELLS_SIZE_GATED,
                      stagnant_s: float = 0.0):
        """Drive the REAL optimize() loop offline.

        perform_initial_analysis is mocked (seeds initial/best WNS AND
        the Phase-1 cell count the way the real method does);
        get_completion is scripted to signal done immediately (anchor
        mode -> clean one-iteration exit when no break fires);
        _exit_with_ils_polish is spied so the break path's use of the
        SHARED tail is observable without running the polish stages.

        V2 defaults: size-gated cell count; stagnant_s=0 DISABLES the
        stagnation guard (optimize() resets last_improvement_time to
        start_time = now, so a real-time 240s window can never elapse
        inside an offline one-iteration drive).
        """
        opt = DCPOptimizer(api_key="test", run_dir=self.tmp_path,
                           mode="anchor")
        opt.rag_seed = False
        opt.contest_mode = True
        opt.max_wall_seconds = max_wall
        opt._deep_wns_tail_reserve_s = reserve
        opt._tail_reserve_stagnant_s = stagnant_s

        input_dcp = self.tmp_path / "baseline.dcp"
        input_dcp.write_bytes(b"BASELINE_DCP_BYTES_FOR_TAIL_RESERVE_TEST")
        output_dcp = self.tmp_path / "optimized.dcp"

        async def fake_analysis(_input_dcp):
            opt.initial_wns = best_wns
            opt.best_wns = best_wns  # real analysis seeds best from initial
            opt.clock_period = 4.0
            opt._input_cell_count = cells  # Phase 1 measures this
            return f"ANALYSIS: initial WNS {best_wns:.3f} ns"

        get_completion = mock.AsyncMock(return_value=("done", True))
        exit_tail = mock.AsyncMock()
        with mock.patch("dcp_optimizer.load_system_prompt",
                        return_value="SYS PROMPT (tail-reserve test)"), \
             mock.patch.object(opt, "perform_initial_analysis",
                               side_effect=fake_analysis), \
             mock.patch.object(opt, "get_completion", new=get_completion), \
             mock.patch.object(opt, "call_tool",
                               new=mock.AsyncMock(return_value="ok")), \
             mock.patch.object(opt, "_exit_with_ils_polish", new=exit_tail), \
             mock.patch.object(opt, "_finalize_output_dcp",
                               new=mock.AsyncMock()), \
             mock.patch.object(opt, "_print_optimization_summary"), \
             mock.patch.object(opt, "_persist_to_strategy_memory"), \
             redirect_stdout(io.StringIO()), \
             self.assertLogs("dcp_optimizer", level="INFO") as logs:
            _async(opt.optimize(input_dcp, output_dcp))
        return opt, get_completion, exit_tail, "\n".join(logs.output)

    def test_on_deep_at_boundary_exits_through_shared_tail(self):
        # Wall 900s: deadline = start + (900 - 300 finalize reserve) ->
        # remaining ~600s <= CLAMPED reserve (0.686 x 900 = 617s; the
        # raw 5000s request exceeds the wall) at iteration 1 -> the
        # loop breaks BEFORE any LLM call and exits via the SHARED
        # tail, with the loud clamp warning in the logs.
        opt, get_completion, exit_tail, logs = self._run_optimize(
            reserve=5000.0, max_wall=900.0, best_wns=-2.0)
        get_completion.assert_not_awaited()
        exit_tail.assert_awaited_once()
        self.assertIn("[tail-reserve] WALL CLAMP", logs)
        self.assertIn("[tail-reserve] deep-WNS reserve reached", logs)
        self.assertIn("exited via tail_reserve", logs)

    def test_off_no_early_exit(self):
        # Identical wall/WNS shape, reserve OFF (default 0): the loop
        # runs its LLM iteration — zero behavior change.
        opt, get_completion, exit_tail, logs = self._run_optimize(
            reserve=0.0, max_wall=900.0, best_wns=-2.0)
        get_completion.assert_awaited()
        self.assertNotIn("[tail-reserve]", logs)

    def test_on_shallow_no_early_exit(self):
        # Boundary hit but the design is near-met (-0.5 > -1.0): owned
        # by the LLM/ILS — no reserve exit (CURRENT-WNS requirement).
        # (The clamp warning may still log — it is a config observation,
        # not a break.)
        opt, get_completion, exit_tail, logs = self._run_optimize(
            reserve=5000.0, max_wall=900.0, best_wns=-0.5)
        get_completion.assert_awaited()
        self.assertNotIn("[tail-reserve] deep-WNS reserve reached", logs)
        self.assertNotIn("exited via tail_reserve", logs)

    def test_on_deep_with_room_no_early_exit(self):
        # Deep WNS but remaining (~8700s) > reserve (100s): no exit.
        opt, get_completion, exit_tail, logs = self._run_optimize(
            reserve=100.0, max_wall=9000.0, best_wns=-2.0)
        get_completion.assert_awaited()
        self.assertNotIn("[tail-reserve]", logs)

    def test_ils_path_design_no_early_exit(self):
        # V2 SIZE GATE at loop level: same boundary shape as the break
        # test but the design is ILS-path (cells <= max_cells) -> the
        # reserve never arms and the LLM iteration runs.
        opt, get_completion, exit_tail, logs = self._run_optimize(
            reserve=5000.0, max_wall=900.0, best_wns=-2.0,
            cells=_CELLS_ILS_PATH)
        get_completion.assert_awaited()
        self.assertNotIn("[tail-reserve] deep-WNS reserve reached", logs)

    def test_unknown_cells_no_early_exit(self):
        # V2 fail-safe at loop level: Phase 1 could not measure a cell
        # count -> NOT armed.
        opt, get_completion, exit_tail, logs = self._run_optimize(
            reserve=5000.0, max_wall=900.0, best_wns=-2.0, cells=None)
        get_completion.assert_awaited()
        self.assertNotIn("[tail-reserve] deep-WNS reserve reached", logs)

    def test_recent_improvement_no_early_exit(self):
        # V2 STAGNATION GUARD at loop level: optimize() seeds
        # last_improvement_time = start_time = now, so with the default
        # 240s window the loop is NOT stagnant at iteration 1 -> no
        # break even though every other condition holds.
        opt, get_completion, exit_tail, logs = self._run_optimize(
            reserve=5000.0, max_wall=900.0, best_wns=-2.0,
            stagnant_s=TAIL_RESERVE_STAGNANT_S_DEFAULT)
        get_completion.assert_awaited()
        self.assertNotIn("[tail-reserve] deep-WNS reserve reached", logs)


if __name__ == "__main__":
    unittest.main()
