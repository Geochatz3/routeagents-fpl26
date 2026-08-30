"""Tests a preserving bare reroute as an exit-tail polish step.

The step runs only when local search did not arm, a routed banked best exists,
and the estimated reroute cost fits the remaining wall time. The estimate uses
half the route sample plus 30 seconds for banking and 60 seconds of margin,
reflecting the lower cost of preserving reroutes.

Measurement and banking use the normal auto-bank path, so a worse result cannot
overwrite the disk mirror. External tool calls are mocked.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import (
    BARE_REROUTE_MAX_ITERS_DEFAULT,
    BARE_REROUTE_MIN_GAIN_NS_DEFAULT,
    DCPOptimizer,
    resolve_bare_reroute_max_iters,
    resolve_bare_reroute_min_gain_ns,
    resolve_bare_reroute_polish_enabled,
)
from optimizer.ils_polish import ILSPolishConfig


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


BOOM_CELLS = 379_380          # the official-beta ILS-size-gated shape
BOOM_BANKED_WNS = -10.676     # measured banked route-first state
LEG2_ROUTE_SAMPLE_S = 1620.0  # leg-2 eval trace: recipe AE route sample
LEG2_REMAINING_S = 1300.0     # leg-2 eval trace: remaining post-bank


def _make_optimizer(tmp_path: Path) -> DCPOptimizer:
    opt = DCPOptimizer(api_key="test", run_dir=tmp_path)
    opt.vivado_session = _FakeSession()
    opt.rapidwright_session = _FakeSession()
    # These fixtures exercise the plain reroute loop used for shallow states,
    # the controller kill switch, and the fail-closed fallback.
    # The adaptive tail controller is disabled so it does not claim them.
    opt._tail_controller_enabled = False
    return opt


def _boom_shape(opt: DCPOptimizer, tmp: Path, remaining_s: float = 3000.0,
                open_elapsed_s: float = 300.0,
                route_sample_s: float = LEG2_ROUTE_SAMPLE_S) -> Path:
    """Boom-shaped fixture: ILS size-gated (379k cells), routed banked
    best on disk, contest wall with `remaining_s` left, a recorded
    phase-1 open_checkpoint cost (the re-open charge) and a recipe
    route_design sample (the preserving cost basis)."""
    opt.max_wall_seconds = 3300.0
    opt._budget_deadline = time.time() + remaining_s
    opt._input_cell_count = BOOM_CELLS
    opt._design_cells = BOOM_CELLS
    banked = tmp / "best_valid.dcp"
    banked.write_bytes(b"dcp")
    opt._best_valid_dcp = banked
    opt._best_valid_dcp_wns = BOOM_BANKED_WNS
    opt.best_wns = BOOM_BANKED_WNS
    opt.initial_wns = -13.0
    opt._ils_preempt_requested = False
    opt.tool_call_details.append({
        "tool_name": "vivado_open_checkpoint",
        "iteration": 0,
        "elapsed_time": open_elapsed_s,
        "wns": None,
        "error": False,
    })
    if route_sample_s > 0:
        # The R1 route-first recipe's AE route as call_tool records it
        # (run_tcl-embedded heavy op — the dominant recipe path; feeds
        # derive_cost_anchors' route_design single-stage sample).
        opt.tool_call_details.append({
            "tool_name": "vivado_run_tcl",
            "iteration": 1,
            "elapsed_time": route_sample_s,
            "wns": None,
            "error": False,
            "cmd_head": "route_design -directive AggressiveExplore",
        })
    return banked


class ResolverTests(unittest.TestCase):
    """resolve_bare_reroute_polish_enabled — default ON; CLI flag or env
    FPL26_NO_BARE_REROUTE_POLISH kill switch disables."""

    def test_default_on(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FPL26_NO_BARE_REROUTE_POLISH", None)
            self.assertTrue(resolve_bare_reroute_polish_enabled(False))

    def test_cli_flag_disables(self):
        self.assertFalse(resolve_bare_reroute_polish_enabled(True))

    def test_env_disables(self):
        for val in ("1", "true", "YES", "on"):
            with mock.patch.dict(
                    os.environ, {"FPL26_NO_BARE_REROUTE_POLISH": val}):
                self.assertFalse(
                    resolve_bare_reroute_polish_enabled(False),
                    f"env {val!r} should disable")

    def test_garbage_env_keeps_on(self):
        with mock.patch.dict(
                os.environ, {"FPL26_NO_BARE_REROUTE_POLISH": "banana"}):
            self.assertTrue(resolve_bare_reroute_polish_enabled(False))

    def test_env_zero_keeps_on(self):
        with mock.patch.dict(
                os.environ, {"FPL26_NO_BARE_REROUTE_POLISH": "0"}):
            self.assertTrue(resolve_bare_reroute_polish_enabled(False))


class FiringGateTests(unittest.TestCase):
    """Fires only when ILS never armed + routed bank exists + wall fits;
    every other combination skips honestly with zero tool calls."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.opt = _make_optimizer(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_fires_on_boom_shape(self):
        # Preserving basis: route sample 1620s × 0.5 = 810s predicted;
        # 3000s remaining − 300s re-open charge = 2700s effective input;
        # 810 + 30 banking ≤ 2700 − 60 margin -> fires.
        banked = _boom_shape(self.opt, self.tmp)
        with mock.patch.object(self.opt, "get_wns_for_target_clock",
                               new=mock.AsyncMock(return_value=None)):
            _async(self.opt._run_bare_reroute_polish())
        calls = self.opt.vivado_session.calls
        names = [n for n, _ in calls]
        self.assertIn("open_checkpoint", names)
        self.assertIn("run_tcl", names)
        # Re-open targets the BANKED best…
        open_args = calls[names.index("open_checkpoint")][1]
        self.assertEqual(open_args["dcp_path"], str(banked.resolve()))
        # …and the route is UNDIRECTED (meta: undirected beats directed):
        # exactly `route_design`, no directive, no -unroute.
        route_args = calls[names.index("run_tcl")][1]
        self.assertEqual(route_args["command"], "route_design")
        self.assertGreaterEqual(route_args["timeout"], 600.0)
        # open before route
        self.assertLess(names.index("open_checkpoint"),
                        names.index("run_tcl"))

    def test_never_fires_when_ils_ran(self):
        _boom_shape(self.opt, self.tmp)
        self.opt._ils_preempt_requested = True
        _async(self.opt._run_bare_reroute_polish())
        self.assertEqual(self.opt.vivado_session.calls, [])

    def test_skips_without_routed_banked_best(self):
        _boom_shape(self.opt, self.tmp)
        self.opt._best_valid_dcp = None
        _async(self.opt._run_bare_reroute_polish())
        self.assertEqual(self.opt.vivado_session.calls, [])

    def test_skips_when_banked_file_missing(self):
        banked = _boom_shape(self.opt, self.tmp)
        banked.unlink()
        _async(self.opt._run_bare_reroute_polish())
        self.assertEqual(self.opt.vivado_session.calls, [])

    def test_wall_unfit_skips_honestly(self):
        # 500s remaining − 300s re-open charge = 200s effective cannot
        # fit predicted 810 + 30 banking (vs 200 − 60 margin) — honest
        # skip, zero tool calls, reserve untouched.
        _boom_shape(self.opt, self.tmp, remaining_s=500.0)
        _async(self.opt._run_bare_reroute_polish())
        self.assertEqual(self.opt.vivado_session.calls, [])
        self.assertIsNone(self.opt._polish_reserve_release_reason)

    def test_no_route_sample_skips(self):
        # History has an open sample but NO route_design sample -> the
        # preserving predictor returns inf (cannot size the op) -> skip
        # even with wall to spare.
        _boom_shape(self.opt, self.tmp, route_sample_s=0.0)
        _async(self.opt._run_bare_reroute_polish())
        self.assertEqual(self.opt.vivado_session.calls, [])

    def test_double_blind_no_data_skips(self):
        # Empty history entirely -> predictor inf -> honest skip.
        _boom_shape(self.opt, self.tmp)
        self.opt._input_cell_count = None
        self.opt.tool_call_details.clear()
        _async(self.opt._run_bare_reroute_polish())
        self.assertEqual(self.opt.vivado_session.calls, [])

    def test_kill_switch_skips(self):
        _boom_shape(self.opt, self.tmp)
        self.opt._bare_reroute_polish_enabled = False
        _async(self.opt._run_bare_reroute_polish())
        self.assertEqual(self.opt.vivado_session.calls, [])

    def test_auto_bank_disabled_skips(self):
        # Without auto-bank the re-route would mutate state with no
        # measurement/banking ride-along — pure waste, so skip.
        _boom_shape(self.opt, self.tmp)
        self.opt._auto_bank_enabled = False
        _async(self.opt._run_bare_reroute_polish())
        self.assertEqual(self.opt.vivado_session.calls, [])


class KeepBestTests(unittest.TestCase):
    """Verifies that reroute polishing preserves the best banked result.

    A worse reroute is measured but leaves the mirror unchanged; an improved,
    routed result is banked through the eager-mirror path.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.opt = _make_optimizer(self.tmp)
        self.banked = _boom_shape(self.opt, self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_worse_reroute_not_banked(self):
        mirror = mock.AsyncMock()
        with mock.patch.object(
                self.opt, "get_wns_for_target_clock",
                new=mock.AsyncMock(return_value=BOOM_BANKED_WNS - 0.2)), \
             mock.patch.object(self.opt, "_mirror_best_valid_now",
                               new=mirror):
            _async(self.opt._run_bare_reroute_polish())
        self.assertEqual(self.opt.best_wns, BOOM_BANKED_WNS)
        self.assertEqual(self.opt._best_valid_dcp, self.banked)
        self.assertEqual(self.opt._best_valid_dcp_wns, BOOM_BANKED_WNS)
        mirror.assert_not_awaited()

    def test_improved_routed_reroute_banked(self):
        # Measured shape: banked −10.676 -> bare re-route −10.582 (+0.094).
        improved = BOOM_BANKED_WNS + 0.094
        mirror = mock.AsyncMock()
        with mock.patch.object(
                self.opt, "get_wns_for_target_clock",
                new=mock.AsyncMock(return_value=improved)), \
             mock.patch.object(self.opt, "_routed_ok_for_best",
                               new=mock.AsyncMock(return_value=True)), \
             mock.patch.object(self.opt, "_mirror_best_valid_now",
                               new=mirror):
            _async(self.opt._run_bare_reroute_polish())
        self.assertAlmostEqual(self.opt.best_wns, improved)
        mirror.assert_awaited()

    def test_improved_but_unrouted_rejected_phantom_guard(self):
        # Phantom-best guard: a better WNS on a not-fully-routed state
        # must NOT be banked (estimated-timing mirage).
        improved = BOOM_BANKED_WNS + 0.094
        mirror = mock.AsyncMock()
        with mock.patch.object(
                self.opt, "get_wns_for_target_clock",
                new=mock.AsyncMock(return_value=improved)), \
             mock.patch.object(self.opt, "_routed_ok_for_best",
                               new=mock.AsyncMock(return_value=False)), \
             mock.patch.object(self.opt, "_mirror_best_valid_now",
                               new=mirror):
            _async(self.opt._run_bare_reroute_polish())
        self.assertEqual(self.opt.best_wns, BOOM_BANKED_WNS)
        mirror.assert_not_awaited()

    def test_route_error_keeps_best(self):
        self.opt.vivado_session.response_text = '{"error": "TCL ERROR: boom"}'
        _async(self.opt._run_bare_reroute_polish())
        self.assertEqual(self.opt.best_wns, BOOM_BANKED_WNS)
        self.assertEqual(self.opt._best_valid_dcp, self.banked)
        # Failed step never releases the reserve (finalize backstops).
        self.assertIsNone(self.opt._polish_reserve_release_reason)


class ReserveReleaseTests(unittest.TestCase):
    """Composition: the step counts as a POLISH stage — it may
    spend the armed reserve (a bare route on a routed design is
    state-preserving, so neither reserve gate nor fence cap applies)
    and its completion releases it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.opt = _make_optimizer(self.tmp)
        _boom_shape(self.opt, self.tmp)
        # Arm the reserve: contest budget + banked best + polish stages
        # enabled/pending (test_polish_reserve.py _arm shape).
        self.opt._polish_reserve_s = 500.0
        self.opt._ils_polish_cfg = ILSPolishConfig(enabled=True)

    def tearDown(self):
        self._tmp.cleanup()

    def test_completion_releases_reserve(self):
        self.assertEqual(self.opt._polish_reserve_armed_s(), 500.0)
        with mock.patch.object(self.opt, "get_wns_for_target_clock",
                               new=mock.AsyncMock(return_value=None)):
            _async(self.opt._run_bare_reroute_polish())
        self.assertEqual(self.opt._polish_reserve_release_reason,
                         "bare_reroute_polish_done")
        self.assertEqual(self.opt._polish_reserve_armed_s(), 0.0)

    def test_armed_reserve_does_not_refuse_the_bare_route(self):
        # The reserved window belongs to polish — and this IS polish:
        # the state-preserving bare route must dispatch even while the
        # reserve is armed.
        with mock.patch.object(self.opt, "get_wns_for_target_clock",
                               new=mock.AsyncMock(return_value=None)):
            _async(self.opt._run_bare_reroute_polish())
        cmds = [a.get("command") for n, a in self.opt.vivado_session.calls
                if n == "run_tcl"]
        self.assertIn("route_design", cmds)


class PreservingCostBasisTests(unittest.TestCase):
    """Pure-function pins for the preserving basis (route_gate)."""

    def test_predict_is_half_the_route_sample(self):
        from optimizer.route_gate import predict_preserving_reroute_seconds
        details = [{"tool_name": "vivado_run_tcl", "elapsed_time": 1620.0,
                    "cmd_head": "route_design -directive AggressiveExplore"}]
        self.assertAlmostEqual(
            predict_preserving_reroute_seconds(details), 810.0)

    def test_predict_inf_without_route_sample(self):
        from optimizer.route_gate import predict_preserving_reroute_seconds
        self.assertEqual(predict_preserving_reroute_seconds([]),
                         float("inf"))
        # A place/phys_opt-only history is NOT a route sample.
        details = [{"tool_name": "vivado_phys_opt_design",
                    "elapsed_time": 900.0}]
        self.assertEqual(predict_preserving_reroute_seconds(details),
                         float("inf"))

    def test_factor_is_headroom_over_measured_ratio(self):
        # The preserving-route factor retains conservative cost headroom.
        # Changing it requires re-deriving the leg-2 replay expectation.
        from optimizer.route_gate import PRESERVING_ROUTE_FACTOR
        measured_ratio = 1199.0 / 3871.0
        self.assertAlmostEqual(measured_ratio, 0.31, places=2)
        self.assertGreaterEqual(PRESERVING_ROUTE_FACTOR,
                                measured_ratio * 1.5)
        self.assertLess(PRESERVING_ROUTE_FACTOR, 0.7,
                        "factor >= ~0.7 refuses the leg-2 eval shape "
                        "-> lever becomes dead code at eval")


class EvalShapeReplayTests(unittest.TestCase):
    """Replay the two real traces the gate was judged against.

    Leg-2 eval trace (coordinator,): route sample 1620s, ~1300s
    remaining post-bank.  DEBUG-WALL local trace: predicted 7200s under
    the old K=2 destructive basis vs 2375s remaining (route sample
    3600s, open 68s).  The destructive basis refused BOTH; the shipped
    preserving posture (0.5 × sample + 30s banking + 60s margin) fires
    on both — pinned here so any posture change re-derives the eval-
    shape outcome explicitly.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.opt = _make_optimizer(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def _route_cmds(self):
        return [a.get("command") for n, a in self.opt.vivado_session.calls
                if n == "run_tcl"]

    def test_leg2_eval_shape_fires(self):
        # 1620 × 0.5 = 810 + 30 = 840 ≤ (1300 − 60 open) − 60 = 1180.
        _boom_shape(self.opt, self.tmp, remaining_s=LEG2_REMAINING_S,
                    open_elapsed_s=60.0,
                    route_sample_s=LEG2_ROUTE_SAMPLE_S)
        with mock.patch.object(self.opt, "get_wns_for_target_clock",
                               new=mock.AsyncMock(return_value=None)):
            _async(self.opt._run_bare_reroute_polish())
        self.assertIn("route_design", self._route_cmds())

    def test_leg2_old_destructive_basis_would_refuse(self):
        # Documents WHY the basis changed: on the same leg-2 shape the
        # destructive K=2 predictor (2×1620=3240s; cells floor 2276s)
        # refuses — the lever would never have fired at eval.
        from optimizer.route_gate import assess_destructive_reroute
        details = [{"tool_name": "vivado_run_tcl", "elapsed_time": 1620.0,
                    "cmd_head": "route_design -directive AggressiveExplore"}]
        a = assess_destructive_reroute(
            LEG2_REMAINING_S - 60.0, details, BOOM_CELLS)
        self.assertFalse(a.feasible)
        self.assertAlmostEqual(a.predicted_reroute_s, 3240.0)

    def test_debug_wall_local_shape_fires(self):
        # DEBUG-WALL trace: route sample 3600s, remaining 2375s, open
        # 68s.  Preserving: 1800 + 30 = 1830 ≤ (2375−68) − 60 = 2247.
        # (Old basis: predicted 7200s -> refused, as observed live.)
        _boom_shape(self.opt, self.tmp, remaining_s=2375.0,
                    open_elapsed_s=68.0, route_sample_s=3600.0)
        with mock.patch.object(self.opt, "get_wns_for_target_clock",
                               new=mock.AsyncMock(return_value=None)):
            _async(self.opt._run_bare_reroute_polish())
        self.assertIn("route_design", self._route_cmds())
        # Timeout posture: 2× prediction ≈ the full-route sample,
        # clamped to remaining (insured overrun costs tail-γ only).
        to = [a["timeout"] for n, a in self.opt.vivado_session.calls
              if n == "run_tcl" and a.get("command") == "route_design"][0]
        self.assertAlmostEqual(to, 2375.0, delta=5.0)


class ExitTailWiringTests(unittest.TestCase):
    """_exit_with_ils_polish: the bare re-route polish runs exactly when
    the ILS stage does NOT (size-gated / never armed), never after ILS."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.opt = _make_optimizer(self.tmp)
        _boom_shape(self.opt, self.tmp)
        # ILS enabled but SIZE-GATED: 379k cells > max_cells 300k, so
        # the loop-exit trigger declines with too_large(...).
        self.opt._ils_polish_cfg = ILSPolishConfig(enabled=True)

    def tearDown(self):
        self._tmp.cleanup()

    def _run_exit_tail(self):
        with mock.patch.object(self.opt, "_finalize_output_dcp",
                               new=mock.AsyncMock()), \
             mock.patch.object(self.opt, "_print_optimization_summary",
                               new=mock.Mock()), \
             mock.patch.object(self.opt, "_run_ils_polish_stage",
                               new=mock.AsyncMock()) as ils_stage, \
             mock.patch.object(self.opt, "_run_bare_reroute_polish",
                               new=mock.AsyncMock()) as bare:
            _async(self.opt._exit_with_ils_polish(self.tmp / "out.dcp"))
        return ils_stage, bare

    def test_size_gated_design_gets_bare_reroute_not_ils(self):
        ils_stage, bare = self._run_exit_tail()
        ils_stage.assert_not_awaited()   # too_large(cells=379380>300000)
        bare.assert_awaited_once()

    def test_ils_ran_no_bare_reroute(self):
        self.opt._ils_preempt_requested = True
        ils_stage, bare = self._run_exit_tail()
        ils_stage.assert_awaited_once()
        bare.assert_not_awaited()

    def test_bare_reroute_failure_does_not_break_finalize(self):
        finalize = mock.AsyncMock()
        with mock.patch.object(self.opt, "_finalize_output_dcp",
                               new=finalize), \
             mock.patch.object(self.opt, "_print_optimization_summary",
                               new=mock.Mock()), \
             mock.patch.object(
                 self.opt, "_run_bare_reroute_polish",
                 new=mock.AsyncMock(side_effect=RuntimeError("kaboom"))):
            _async(self.opt._exit_with_ils_polish(self.tmp / "out.dcp"))
        finalize.assert_awaited_once()


class LoopKnobResolverTests(unittest.TestCase):
    """resolve_bare_reroute_min_gain_ns / resolve_bare_reroute_max_iters
    — CLI wins over env; invalid/negative/<1 keeps the defaults."""

    def _clean_env(self):
        os.environ.pop("FPL26_BARE_REROUTE_MIN_GAIN", None)
        os.environ.pop("FPL26_BARE_REROUTE_MAX_ITERS", None)

    def test_min_gain_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            self._clean_env()
            self.assertAlmostEqual(resolve_bare_reroute_min_gain_ns(None),
                                   BARE_REROUTE_MIN_GAIN_NS_DEFAULT)
            self.assertAlmostEqual(BARE_REROUTE_MIN_GAIN_NS_DEFAULT, 0.020)

    def test_min_gain_cli_wins_over_env(self):
        with mock.patch.dict(os.environ,
                             {"FPL26_BARE_REROUTE_MIN_GAIN": "0.5"}):
            self.assertAlmostEqual(
                resolve_bare_reroute_min_gain_ns(0.1), 0.1)

    def test_min_gain_env_fallback(self):
        with mock.patch.dict(os.environ,
                             {"FPL26_BARE_REROUTE_MIN_GAIN": "0.05"}):
            self.assertAlmostEqual(
                resolve_bare_reroute_min_gain_ns(None), 0.05)

    def test_min_gain_invalid_or_negative_keeps_default(self):
        with mock.patch.dict(os.environ,
                             {"FPL26_BARE_REROUTE_MIN_GAIN": "banana"}):
            self.assertAlmostEqual(resolve_bare_reroute_min_gain_ns(None),
                                   BARE_REROUTE_MIN_GAIN_NS_DEFAULT)
        with mock.patch.dict(os.environ, {}, clear=False):
            self._clean_env()
            self.assertAlmostEqual(resolve_bare_reroute_min_gain_ns(-0.1),
                                   BARE_REROUTE_MIN_GAIN_NS_DEFAULT)

    def test_max_iters_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            self._clean_env()
            self.assertEqual(resolve_bare_reroute_max_iters(None),
                             BARE_REROUTE_MAX_ITERS_DEFAULT)
            self.assertEqual(BARE_REROUTE_MAX_ITERS_DEFAULT, 4)

    def test_max_iters_cli_wins_over_env(self):
        with mock.patch.dict(os.environ,
                             {"FPL26_BARE_REROUTE_MAX_ITERS": "9"}):
            self.assertEqual(resolve_bare_reroute_max_iters(2), 2)

    def test_max_iters_env_fallback(self):
        with mock.patch.dict(os.environ,
                             {"FPL26_BARE_REROUTE_MAX_ITERS": "6"}):
            self.assertEqual(resolve_bare_reroute_max_iters(None), 6)

    def test_max_iters_below_one_or_invalid_keeps_default(self):
        with mock.patch.dict(os.environ,
                             {"FPL26_BARE_REROUTE_MAX_ITERS": "zero"}):
            self.assertEqual(resolve_bare_reroute_max_iters(None),
                             BARE_REROUTE_MAX_ITERS_DEFAULT)
        with mock.patch.dict(os.environ, {}, clear=False):
            self._clean_env()
            self.assertEqual(resolve_bare_reroute_max_iters(0),
                             BARE_REROUTE_MAX_ITERS_DEFAULT)

    def test_max_iters_one_restores_one_shot(self):
        # 1 is a VALID setting (pre-loop one-shot behavior), not clamped.
        self.assertEqual(resolve_bare_reroute_max_iters(1), 1)


PROBE_START_WNS = -11.392     # plateau probe: pre-loop boom state
PROBE_WNS_SEQ = (-11.177,     # pass 1: +0.215
                 -10.773,     # pass 2: +0.404 (the re-bite)
                 -10.769)     # pass 3: +0.004 (plateau decay < 0.020)


class IterationLoopTests(unittest.TestCase):
    """Plateau probe: the rip-up re-roll COMPOUNDS while WNS is
    deep (boom 2nd pass +0.404, cumulative +0.62 over two passes from
    −11.392) and decays near plateaus (fir +0.004).  The loop continues
    while gain >= min-gain AND the next pass wall-fits AND iters < max;
    each stop reason is logged distinctly and ALL stops after >=1
    successful pass release the one reserve window."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.opt = _make_optimizer(self.tmp)
        self.banked = _boom_shape(self.opt, self.tmp)
        self.opt.best_wns = PROBE_START_WNS
        self.opt._best_valid_dcp_wns = PROBE_START_WNS

    def tearDown(self):
        self._tmp.cleanup()

    def _route_calls(self):
        return [a for n, a in self.opt.vivado_session.calls
                if n == "run_tcl" and a.get("command") == "route_design"]

    def _run(self, wns_side_effect):
        mirror = mock.AsyncMock()
        with mock.patch.object(
                self.opt, "get_wns_for_target_clock",
                new=mock.AsyncMock(side_effect=wns_side_effect)), \
             mock.patch.object(self.opt, "_routed_ok_for_best",
                               new=mock.AsyncMock(return_value=True)), \
             mock.patch.object(self.opt, "_mirror_best_valid_now",
                               new=mirror), \
             self.assertLogs(level="INFO") as cm:
            _async(self.opt._run_bare_reroute_polish())
        return mirror, "\n".join(cm.output)

    def test_loop_continues_while_gain_above_min(self):
        # Probe replay: +0.215 -> +0.404 (re-bite) -> +0.004 (decay).
        mirror, logs = self._run(list(PROBE_WNS_SEQ))
        self.assertEqual(len(self._route_calls()), 3)
        # Every improved+routed pass banked (incl. the +0.004 tail —
        # never-worse keeps it; only the LOOP stops on it)…
        self.assertAlmostEqual(self.opt.best_wns, PROBE_WNS_SEQ[-1])
        self.assertEqual(mirror.await_count, 3)
        # …and the stop reason is the plateau-decay one.
        self.assertIn("gain_below_min", logs)
        self.assertIn("stop=gain_below_min", logs)
        self.assertEqual(self.opt._polish_reserve_release_reason,
                         "bare_reroute_polish_done")

    def test_first_iteration_equivalence_single_pass(self):
        # Regression pin: a single non-improving pass has EXACTLY the
        # shipped one-shot shape — one re-open + one bare route, then
        # measure/no-bank/release.  The loop wrapper adds nothing.
        with mock.patch.object(self.opt, "get_wns_for_target_clock",
                               new=mock.AsyncMock(return_value=None)), \
             self.assertLogs(level="INFO") as cm:
            _async(self.opt._run_bare_reroute_polish())
        names = [n for n, _ in self.opt.vivado_session.calls]
        # The POLISH itself is still EXACTLY one re-open + one bare route.
        self.assertEqual(names[:2], ["open_checkpoint", "run_tcl"])
        # A completed tail contributes its best valid state to final selection
        # after polish and reserve release. Verification reopens and probes
        # the candidate without adding another bare route.
        self.assertEqual(names[2], "open_checkpoint")
        self.assertEqual(len(self._route_calls()), 1)
        self.assertEqual(self.opt.best_wns, PROBE_START_WNS)
        self.assertEqual(self.opt._polish_reserve_release_reason,
                         "bare_reroute_polish_done")
        self.assertIn("stop=gain_below_min", "\n".join(cm.output))

    def test_stops_on_wall_unfit(self):
        # Pass 1 improves but burns the wall (deadline shrunk to 50s <
        # the 60s safety margin) -> the NEXT pass cannot fit -> stop
        # wall_unfit after exactly one route; reserve still released.
        def _improve_and_burn_wall(*_a, **_k):
            self.opt._budget_deadline = time.time() + 50.0
            return PROBE_WNS_SEQ[0]
        _, logs = self._run(_improve_and_burn_wall)
        self.assertEqual(len(self._route_calls()), 1)
        self.assertAlmostEqual(self.opt.best_wns, PROBE_WNS_SEQ[0])
        self.assertIn("stop=wall_unfit", logs)
        self.assertEqual(self.opt._polish_reserve_release_reason,
                         "bare_reroute_polish_done")

    def test_stops_at_max_iters(self):
        # Runaway guard: gains keep clearing the min but the iteration
        # cap fences the loop.
        self.opt._bare_reroute_max_iters = 2
        _, logs = self._run(lambda *_a, **_k: self.opt.best_wns + 0.2)
        self.assertEqual(len(self._route_calls()), 2)
        self.assertIn("stop=max_iters", logs)
        self.assertEqual(self.opt._polish_reserve_release_reason,
                         "bare_reroute_polish_done")

    def test_stops_on_route_error_second_iteration(self):
        # Pass 1 improves; pass 2's route_design errors.  The loop stops
        # (route_error) but the polish DID complete a successful pass —
        # reserve releases and the banked pass-1 best is intact.
        def _improve_then_arm_error(*_a, **_k):
            self.opt.vivado_session.response_text = (
                '{"error": "TCL ERROR: boom"}')
            return PROBE_WNS_SEQ[0]
        _, logs = self._run(_improve_then_arm_error)
        self.assertEqual(len(self._route_calls()), 2)
        self.assertAlmostEqual(self.opt.best_wns, PROBE_WNS_SEQ[0])
        self.assertIn("stop=route_error", logs)
        self.assertEqual(self.opt._polish_reserve_release_reason,
                         "bare_reroute_polish_done")

    def test_first_pass_route_error_keeps_shipped_no_release(self):
        # First-iteration route error = shipped semantics: return with
        # NO reserve release (finalize backstops), zero extra passes.
        # (Open succeeds; only the bare route errors.)
        sess = self.opt.vivado_session

        async def _route_errors(name, arguments):
            sess.calls.append((name, arguments))
            if (name == "run_tcl"
                    and arguments.get("command") == "route_design"):
                return _FakeResult('{"error": "TCL ERROR: x"}')
            return _FakeResult("{}")

        sess.call_tool = _route_errors
        _async(self.opt._run_bare_reroute_polish())
        self.assertEqual(len(self._route_calls()), 1)
        self.assertEqual(self.opt.best_wns, PROBE_START_WNS)
        self.assertIsNone(self.opt._polish_reserve_release_reason)

    def test_observed_cost_refreshes_next_prediction(self):
        # The second iteration uses the first iteration's observed bare-route
        # cost for wall-fit sizing. Because it is already a bare-route sample,
        # no full-to-bare discount is applied.
        import optimizer.route_gate as rg
        real_assess = rg.assess_preserving_reroute
        spy = mock.Mock(side_effect=real_assess)

        def _improve_and_stamp_cost(*_a, **_k):
            # The fake session has negligible runtime, so append a synthetic
            # route record after the real call. The observed-cost scan uses
            # the maximum elapsed time in this post-mark slice.
            self.opt.tool_call_details.append({
                "tool_name": "vivado_run_tcl",
                "iteration": 99,
                "elapsed_time": 1234.0,
                "wns": None,
                "error": False,
                "cmd_head": "route_design",
            })
            return self.opt.best_wns + 0.2

        with mock.patch.object(rg, "assess_preserving_reroute", new=spy):
            self.opt._bare_reroute_max_iters = 2
            self._run(_improve_and_stamp_cost)
        self.assertEqual(spy.call_count, 2)
        # Call 1: the shipped pre-open assessment on the FULL history.
        first_args, first_kwargs = spy.call_args_list[0]
        self.assertIs(first_args[1], self.opt.tool_call_details)
        self.assertNotIn("preserve_factor", first_kwargs)
        # Call 2: refreshed single-entry sample = observed cost, 1.0.
        second_args, second_kwargs = spy.call_args_list[1]
        refreshed = second_args[1]
        self.assertEqual(len(refreshed), 1)
        self.assertEqual(refreshed[0]["cmd_head"], "route_design")
        self.assertAlmostEqual(refreshed[0]["elapsed_time"], 1234.0)
        self.assertEqual(second_kwargs.get("preserve_factor"), 1.0)

    def test_observed_cost_missing_falls_back_to_prediction(self):
        # If the route record vanishes (no observable elapsed), the next
        # wall-fit falls back to the prediction the pass just ran under
        # — never an inf-skip masquerading as wall_unfit.
        import optimizer.route_gate as rg
        real_assess = rg.assess_preserving_reroute
        spy = mock.Mock(side_effect=real_assess)

        def _improve_and_lose_history(*_a, **_k):
            self.opt.tool_call_details.clear()
            return self.opt.best_wns + 0.2

        with mock.patch.object(rg, "assess_preserving_reroute", new=spy):
            self.opt._bare_reroute_max_iters = 2
            _, logs = self._run(_improve_and_lose_history)
        self.assertEqual(spy.call_count, 2)
        second_args, _ = spy.call_args_list[1]
        # Fallback = the iteration-1 prediction (1620 x 0.5 = 810s).
        self.assertAlmostEqual(second_args[1][0]["elapsed_time"], 810.0)
        self.assertEqual(len(self._route_calls()), 2)
        self.assertIn("stop=max_iters", logs)


if __name__ == "__main__":
    unittest.main()
