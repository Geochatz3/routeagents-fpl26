"""Adaptive banked tail controller tests.

Policy layer (optimizer/tail_controller.py) is pure — tested directly.
Execution layer (DCPOptimizer._run_tail_controller + the dispatch inside
_run_bare_reroute_polish + the auto-bank suppression bracket) is tested
stub-driven in the test_auto_bank.py / test_bare_reroute_polish.py style:
no Vivado / RapidWright / MCP.

Load-bearing invariants:
  - controller arms ONLY on deep-WNS (best_wns <= threshold, default
    -1.0); shallow states keep the shipped plain M1 loop verbatim;
  - phys_opt moves NEVER bank without the stage's hold-gated accept
    (the auto-bank hook has no hold gate — banking a hold-dirty state
    would fail the validator, α=0);
  - the suppression bracket is set during phys_opt moves and ALWAYS
    restored (try/finally);
  - fail-closed: a controller crash falls back to the plain M1 loop and
    never escapes _run_bare_reroute_polish;
  - accepted moves un-retire the rest (chain evidence M2→M1→M3 +0.726);
  - kill switch / knob resolvers follow the house convention.
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
    TAIL_CTRL_DEEP_WNS_NS_DEFAULT,
    TAIL_CTRL_MAX_MOVES_DEFAULT,
    DCPOptimizer,
    resolve_tail_controller_enabled,
    resolve_tail_ctrl_deep_wns_ns,
    resolve_tail_ctrl_max_moves,
    resolve_tail_ctrl_m1_echo,
)
from optimizer.tail_controller import (
    TAIL_CTRL_MAX_EXECS_PER_MOVE,
    TAIL_MENU,
    MoveState,
    expected_rate,
    hold_accept,
    move_by_key,
    new_states,
    pick_next,
    predict_move_cost_s,
    record_result,
)


def _async(coro):
    return asyncio.run(coro)


class _FakeContent:
    def __init__(self, text: str):
        self.text = text


class _FakeResult:
    def __init__(self, text: str):
        self.content = [_FakeContent(text)]


class _FakeSession:
    def __init__(self, response_text: str = "{}"):
        self.response_text = response_text
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        return _FakeResult(self.response_text)


ROUTE_SAMPLE = {"tool_name": "vivado_run_tcl", "elapsed_time": 1000.0,
                "error": False,
                "cmd_head": "route_design -directive AggressiveExplore"}


# ---------------------------------------------------------------------------
# Pure policy
# ---------------------------------------------------------------------------

class PolicyTests(unittest.TestCase):

    def test_priors_pick_ladder_first(self):
        # Probe rates: m4 2.58e-4 > m3 2.08e-4 > m1 1.94e-4 > m2 1.48e-4.
        states = new_states()
        afford = {m.key: True for m in TAIL_MENU}
        self.assertEqual(pick_next(states, afford), "m4_ladder")

    def test_pick_skips_unaffordable_and_retired(self):
        states = new_states()
        afford = {m.key: True for m in TAIL_MENU}
        afford["m4_ladder"] = False
        states["m3_fanout"].retired = True
        self.assertEqual(pick_next(states, afford), "m1_route")

    def test_pick_respects_exec_cap(self):
        states = new_states()
        afford = {m.key: True for m in TAIL_MENU}
        for st in states.values():
            st.executed = TAIL_CTRL_MAX_EXECS_PER_MOVE
        self.assertIsNone(pick_next(states, afford))

    def test_pick_none_when_all_unaffordable(self):
        states = new_states()
        self.assertIsNone(pick_next(states, {m.key: False for m in TAIL_MENU}))

    def test_observed_rate_replaces_prior(self):
        m = move_by_key("m1_route")
        st = MoveState()
        prior = expected_rate(m, st)
        record_result({"m1_route": st, **{k: MoveState() for k in
                       ("m4_ladder", "m3_fanout", "m2_physopt_ae")}},
                      "m1_route", gain_ns=0.4, cost_s=100.0,
                      min_gain_ns=0.02)
        self.assertAlmostEqual(expected_rate(m, st), 0.004)
        self.assertGreater(expected_rate(m, st), prior)

    def test_below_min_gain_retires(self):
        states = new_states()
        record_result(states, "m1_route", gain_ns=0.001, cost_s=100.0,
                      min_gain_ns=0.02)
        self.assertTrue(states["m1_route"].retired)
        # A below-min result never un-retires others.
        self.assertFalse(states["m2_physopt_ae"].retired)

    def test_error_retires(self):
        states = new_states()
        record_result(states, "m3_fanout", gain_ns=0.0, cost_s=50.0,
                      min_gain_ns=0.02, errored=True)
        self.assertTrue(states["m3_fanout"].retired)
        self.assertEqual(states["m3_fanout"].errors, 1)

    def test_accept_unretires_others_and_resets_observation(self):
        # Chain evidence M2→M1→M3: a state change makes dead moves
        # re-bite — un-retire with the stale observation cleared.
        states = new_states()
        record_result(states, "m1_route", gain_ns=0.001, cost_s=100.0,
                      min_gain_ns=0.02)          # retires m1
        self.assertTrue(states["m1_route"].retired)
        record_result(states, "m2_physopt_ae", gain_ns=0.5, cost_s=1000.0,
                      min_gain_ns=0.02)          # accepted
        self.assertFalse(states["m1_route"].retired)
        self.assertIsNone(states["m1_route"].last_gain_ns)
        self.assertFalse(states["m2_physopt_ae"].retired)

    def test_hold_accept_h0_relative(self):
        # Clean base: floor is -0.001.
        self.assertTrue(hold_accept(0.010, 0.010))
        self.assertTrue(hold_accept(0.000, 0.100))   # positive margin may shrink
        self.assertFalse(hold_accept(-0.010, 0.010))
        # Dirty base: may not get meaningfully worse.
        self.assertTrue(hold_accept(-0.020, -0.020))
        self.assertFalse(hold_accept(-0.050, -0.020))
        # Unmeasurable hold rejects (fail-closed).
        self.assertFalse(hold_accept(None, 0.010))

    def test_predict_cost_observed_beats_fallback(self):
        m = move_by_key("m2_physopt_ae")
        st = MoveState()
        self.assertAlmostEqual(predict_move_cost_s(m, st, 1000.0), 2000.0)
        st.last_cost_s = 500.0
        self.assertAlmostEqual(predict_move_cost_s(m, st, 1000.0), 500.0)

    def test_menu_shape(self):
        self.assertEqual([m.key for m in TAIL_MENU],
                         ["m1_route", "m4_ladder", "m3_fanout",
                          "m2_physopt_ae"])
        self.assertTrue(move_by_key("m1_route").rides_autobank)
        for k in ("m4_ladder", "m3_fanout", "m2_physopt_ae"):
            self.assertFalse(move_by_key(k).rides_autobank)


# Resolvers

class ResolverTests(unittest.TestCase):

    def test_enabled_default_on(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FPL26_NO_TAIL_CONTROLLER", None)
            self.assertTrue(resolve_tail_controller_enabled(False))

    def test_cli_disables(self):
        self.assertFalse(resolve_tail_controller_enabled(True))

    def test_env_disables(self):
        with mock.patch.dict(os.environ,
                             {"FPL26_NO_TAIL_CONTROLLER": "1"}):
            self.assertFalse(resolve_tail_controller_enabled(False))

    def test_deep_wns_default_and_bounds(self):
        self.assertEqual(resolve_tail_ctrl_deep_wns_ns(None),
                         TAIL_CTRL_DEEP_WNS_NS_DEFAULT)
        self.assertEqual(resolve_tail_ctrl_deep_wns_ns(-3.0), -3.0)
        # Positive threshold is nonsense — keep default.
        self.assertEqual(resolve_tail_ctrl_deep_wns_ns(0.5),
                         TAIL_CTRL_DEEP_WNS_NS_DEFAULT)

    def test_deep_wns_env(self):
        with mock.patch.dict(os.environ,
                             {"FPL26_TAIL_CTRL_DEEP_WNS": "-2.5"}):
            self.assertEqual(resolve_tail_ctrl_deep_wns_ns(None), -2.5)
        with mock.patch.dict(os.environ,
                             {"FPL26_TAIL_CTRL_DEEP_WNS": "bogus"}):
            self.assertEqual(resolve_tail_ctrl_deep_wns_ns(None),
                             TAIL_CTRL_DEEP_WNS_NS_DEFAULT)

    def test_max_moves(self):
        self.assertEqual(resolve_tail_ctrl_max_moves(None),
                         TAIL_CTRL_MAX_MOVES_DEFAULT)
        self.assertEqual(resolve_tail_ctrl_max_moves(6), 6)
        self.assertEqual(resolve_tail_ctrl_max_moves(0),
                         TAIL_CTRL_MAX_MOVES_DEFAULT)
        with mock.patch.dict(os.environ,
                             {"FPL26_TAIL_CTRL_MAX_MOVES": "3"}):
            self.assertEqual(resolve_tail_ctrl_max_moves(None), 3)

    def test_m1_echo_default_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FPL26_TAIL_CTRL_M1_ECHO", None)
            self.assertFalse(resolve_tail_ctrl_m1_echo(False))

    def test_m1_echo_cli_enables(self):
        self.assertTrue(resolve_tail_ctrl_m1_echo(True))

    def test_m1_echo_env_enables(self):
        for val in ("1", "true", "yes", "on"):
            with mock.patch.dict(os.environ,
                                 {"FPL26_TAIL_CTRL_M1_ECHO": val}):
                self.assertTrue(resolve_tail_ctrl_m1_echo(False))

    def test_m1_echo_env_bogus_stays_off(self):
        for val in ("0", "false", "off", "bogus", ""):
            with mock.patch.dict(os.environ,
                                 {"FPL26_TAIL_CTRL_M1_ECHO": val}):
                self.assertFalse(resolve_tail_ctrl_m1_echo(False))


# ---------------------------------------------------------------------------
# Execution layer
# ---------------------------------------------------------------------------

def _make_optimizer(tmp_path: Path) -> DCPOptimizer:
    opt = DCPOptimizer(api_key="test", run_dir=tmp_path)
    opt.vivado_session = _FakeSession()
    opt.rapidwright_session = _FakeSession()
    return opt


class _ControllerHarness(unittest.TestCase):
    """Shared stub harness for _run_tail_controller (no Vivado).  First
    pick under priors is m4_ladder (NOT rides_autobank) — the hold-gate
    paths are exercised on the very first move."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = _make_optimizer(Path(self.tmp.name))
        self.opt.best_wns = -2.0
        self.opt._budget_deadline = time.time() + 50000.0
        self.opt._tail_ctrl_max_moves = 1
        self.opt.tool_call_details.append(dict(ROUTE_SAMPLE))
        banked = Path(self.tmp.name) / "best_valid.dcp"
        banked.write_bytes(b"x" * 64)
        self.opt._best_valid_dcp = banked

        self.call_log: list[tuple[str, dict]] = []
        self.suppress_seen: list[bool] = []
        self.mirror_calls: list[bool] = []
        self.measured_wns: float | None = -1.5   # improved vs -2.0
        self.routed_result = True
        self.hold_values: list[float | None] = [0.010, 0.010]  # base, after

        opt = self.opt

        async def fake_call_tool(name, args):
            self.call_log.append((name, args))
            self.suppress_seen.append(opt._tail_ctrl_suppress_autobank)
            return "ok"

        async def fake_measure(call_tool_fn):
            return self.measured_wns

        async def fake_routed_ok():
            return self.routed_result

        async def fake_mirror(eager: bool = False):
            self.mirror_calls.append(eager)

        opt.call_tool = fake_call_tool
        opt.get_wns_for_target_clock = fake_measure
        opt._routed_ok_for_best = fake_routed_ok
        opt._mirror_best_valid_now = fake_mirror

        async def fake_hold(call_tool_fn, timeout_s=120.0):
            return self.hold_values.pop(0) if self.hold_values else None

        self._hold_patch = mock.patch(
            "optimizer.ils_polish._measure_hold", new=fake_hold)
        self._hold_patch.start()

    def tearDown(self):
        self._hold_patch.stop()
        self.tmp.cleanup()

    def _run(self):
        _async(self.opt._run_tail_controller(min_gain_ns=0.02))

    def _route_design_calls(self):
        return [(n, a) for n, a in self.call_log
                if n == "vivado_run_tcl"
                and str(a.get("command", "")) == "route_design"]


class ControllerExecutionTests(_ControllerHarness):
    """_run_tail_controller behavior with the echo flag at its DEFAULT
    (OFF) — the pre-echo controller contract, unchanged."""

    def test_hold_clean_improvement_adopts(self):
        self._run()
        self.assertEqual(self.mirror_calls, [True])
        self.assertEqual(self.opt.best_wns, -1.5)

    def test_hold_dirty_rejects(self):
        self.hold_values = [0.010, -0.500]
        self._run()
        self.assertEqual(self.mirror_calls, [])
        self.assertEqual(self.opt.best_wns, -2.0)

    def test_unrouted_rejects(self):
        self.routed_result = False
        self._run()
        self.assertEqual(self.mirror_calls, [])
        self.assertEqual(self.opt.best_wns, -2.0)

    def test_unmeasurable_wns_rejects(self):
        self.measured_wns = None
        self._run()
        self.assertEqual(self.mirror_calls, [])
        self.assertEqual(self.opt.best_wns, -2.0)

    def test_suppression_bracket_set_and_restored(self):
        self._run()
        # The 4 ladder phys_opt commands ran under suppression.
        physopt_flags = [
            s for (n, a), s in zip(self.call_log, self.suppress_seen)
            if n == "vivado_run_tcl"
            and str(a.get("command", "")).startswith("phys_opt_design")]
        self.assertEqual(len(physopt_flags), 4)
        self.assertTrue(all(physopt_flags))
        self.assertFalse(self.opt._tail_ctrl_suppress_autobank)

    def test_suppression_restored_on_error(self):
        opt = self.opt

        async def erroring_call_tool(name, args):
            self.call_log.append((name, args))
            return 'TCL ERROR: boom'

        opt.call_tool = erroring_call_tool
        self._run()
        self.assertFalse(opt._tail_ctrl_suppress_autobank)
        self.assertEqual(self.mirror_calls, [])
        self.assertEqual(opt.best_wns, -2.0)

    def test_harvest_continues_after_accept(self):
        # max_moves=3: accept (m4) -> next move runs -> etc.  With every
        # phys_opt measuring -1.5 the 2nd accept shows no further gain
        # (w == best) and the loop keeps going until caps/retire.
        self.opt._tail_ctrl_max_moves = 3
        self.hold_values = [0.010, 0.010] * 6
        self._run()
        self.assertGreaterEqual(len(self.mirror_calls), 1)

    def test_release_called(self):
        released = []
        self.opt._release_polish_reserve = lambda why: released.append(why)
        self._run()
        self.assertEqual(released, ["tail_controller_done"])


class M1EchoTests(_ControllerHarness):
    """POST-ACCEPT M1 ECHO (panel fantasy #2, terra; DEFAULT OFF).

    Load-bearing invariants:
      - flag OFF (constructor default) => zero diff: no route_design
        ever runs off an adopted phys_opt move;
      - flag ON + ADOPTED phys_opt move => exactly ONE bare route_design
        echo immediately after it, riding the auto-bank hook (NOT under
        the suppression bracket), COUNTED toward moves_done/max_moves;
      - the echo needs its own affordability (same margin stack) and
        skips with a log line when unaffordable or cap-blocked;
      - the echo result is recorded into the m1_route MoveState (a
        below-min echo retires m1 exactly like a picked M1 — the echo
        IS an M1 execution).
    """

    def _record_result_spy(self):
        """Patch optimizer.tail_controller.record_result with a spy that
        logs (key, gain) and delegates to the real policy fn.  Works
        because _run_tail_controller from-imports it at call time."""
        import optimizer.tail_controller as _tc
        real = _tc.record_result
        calls: list[tuple[str, float]] = []

        def spy(states, key, gain_ns, cost_s, min_gain_ns, errored=False):
            calls.append((key, gain_ns))
            return real(states, key, gain_ns, cost_s, min_gain_ns,
                        errored=errored)

        return mock.patch("optimizer.tail_controller.record_result",
                          new=spy), calls

    def test_default_off_no_echo(self):
        # Constructor default is OFF; an adopted m4 move gets NO echo.
        self.assertFalse(self.opt._tail_ctrl_m1_echo)
        self.opt._tail_ctrl_max_moves = 2
        self.hold_values = [0.010] * 4
        self._run()
        self.assertEqual(self._route_design_calls(), [])
        self.assertEqual(self.mirror_calls, [True])  # m4 still adopts

    def test_on_adopted_physopt_exactly_one_echo_counted(self):
        # With max_moves=2, the adopted m4 and its m1 echo consume both move
        # slots. Expect four ladder phys_opt calls and one bare route_design;
        # no later move may run.
        self.opt._tail_ctrl_m1_echo = True
        self.opt._tail_ctrl_max_moves = 2
        self._run()
        route_calls = self._route_design_calls()
        self.assertEqual(len(route_calls), 1)
        tcl_cmds = [str(a.get("command", "")) for n, a in self.call_log
                    if n == "vivado_run_tcl"]
        self.assertEqual(len(tcl_cmds), 5)
        self.assertTrue(all(c.startswith("phys_opt_design")
                            for c in tcl_cmds[:4]))
        self.assertEqual(tcl_cmds[4], "route_design")
        # The echo rides the auto-bank hook: NOT under suppression.
        echo_flags = [s for (n, a), s in zip(self.call_log,
                                             self.suppress_seen)
                      if n == "vivado_run_tcl"
                      and str(a.get("command", "")) == "route_design"]
        self.assertEqual(echo_flags, [False])

    def test_echo_result_recorded_into_m1_state(self):
        # The call_tool stub leaves best_wns unchanged, making the echo a zero-gain
        # m1_route result immediately after its parent m4.
        # The below-minimum result retires m1 because an echo counts as an M1 execution.
        self.opt._tail_ctrl_m1_echo = True
        self.opt._tail_ctrl_max_moves = 2
        patcher, calls = self._record_result_spy()
        with patcher:
            self._run()
        self.assertEqual([k for k, _ in calls], ["m4_ladder", "m1_route"])
        self.assertEqual(calls[1], ("m1_route", 0.0))

    def test_echo_gain_recorded_and_banks_via_hook(self):
        # An improving echo (auto-bank hook moves best_wns) is recorded
        # into the m1 state with its positive gain.
        self.opt._tail_ctrl_m1_echo = True
        self.opt._tail_ctrl_max_moves = 2
        opt = self.opt

        async def bumping_call_tool(name, args):
            self.call_log.append((name, args))
            self.suppress_seen.append(opt._tail_ctrl_suppress_autobank)
            if str(args.get("command", "")) == "route_design":
                opt.best_wns = -1.2   # hook banked an improvement
            return "ok"

        opt.call_tool = bumping_call_tool
        patcher, calls = self._record_result_spy()
        with patcher:
            self._run()
        self.assertEqual([k for k, _ in calls], ["m4_ladder", "m1_route"])
        self.assertAlmostEqual(calls[1][1], 0.3, places=6)
        self.assertEqual(opt.best_wns, -1.2)

    def test_unaffordable_echo_skips_with_log(self):
        # Remaining wall lies between the parent move and echo estimates.
        # The parent is adopted, but the echo is skipped without routing.
        from optimizer.route_gate import assess_preserving_reroute
        import dcp_optimizer as dcp_mod
        self.opt._tail_ctrl_m1_echo = True
        self.opt._tail_ctrl_max_moves = 2
        self.hold_values = [0.010] * 4
        rp = assess_preserving_reroute(
            10000.0, self.opt.tool_call_details).predicted_reroute_s
        self.assertNotEqual(rp, float("inf"))
        m4_need = 0.9 * rp * 1.3 + 30.0 + 120.0
        m1_need = 1.0 * rp * 1.3 + 30.0 + 120.0
        remaining = (m4_need + m1_need) / 2.0
        self.opt._budget_remaining = lambda: remaining
        with self.assertLogs(dcp_mod.logger, level="INFO") as cm:
            self._run()
        self.assertEqual(self._route_design_calls(), [])
        self.assertEqual(self.mirror_calls, [True])  # parent adopted
        self.assertTrue(any("echo after m4_ladder SKIPPED (unaffordable"
                            in line for line in cm.output))

    def test_echo_blocked_by_max_moves_cap(self):
        # max_moves=1: the adopted parent consumes the whole cap — the
        # echo must NOT run (no free lunch).
        self.opt._tail_ctrl_m1_echo = True
        self.opt._tail_ctrl_max_moves = 1
        import dcp_optimizer as dcp_mod
        with self.assertLogs(dcp_mod.logger, level="INFO") as cm:
            self._run()
        self.assertEqual(self._route_design_calls(), [])
        self.assertEqual(self.mirror_calls, [True])
        self.assertTrue(any("echo after m4_ladder SKIPPED (max_moves"
                            in line for line in cm.output))

    def test_no_echo_after_rejected_move(self):
        # A REJECTED parent (hold-dirty) mints no echo even with the
        # flag ON — the echo rides only ADOPTED states.
        self.opt._tail_ctrl_m1_echo = True
        self.opt._tail_ctrl_max_moves = 1
        self.hold_values = [0.010, -0.500]
        self._run()
        self.assertEqual(self._route_design_calls(), [])
        self.assertEqual(self.mirror_calls, [])


class DispatchTests(unittest.TestCase):
    """_run_bare_reroute_polish routes deep-WNS states to the controller,
    keeps shallow states on the plain M1 loop, and fails closed."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = _make_optimizer(Path(self.tmp.name))
        self.opt._budget_deadline = time.time() + 50000.0
        self.opt._ils_preempt_requested = False
        self.opt.tool_call_details.append(dict(ROUTE_SAMPLE))
        banked = Path(self.tmp.name) / "best_valid.dcp"
        banked.write_bytes(b"x" * 64)
        self.opt._best_valid_dcp = banked

        self.call_log: list[tuple[str, dict]] = []
        opt = self.opt

        async def fake_call_tool(name, args):
            self.call_log.append((name, args))
            return "ok"

        opt.call_tool = fake_call_tool

        self.ctrl_calls: list[float] = []

        async def fake_controller(min_gain_ns):
            self.ctrl_calls.append(min_gain_ns)

        opt._run_tail_controller = fake_controller

    def tearDown(self):
        self.tmp.cleanup()

    def _route_calls(self):
        return [a for (n, a) in self.call_log
                if n == "vivado_run_tcl"
                and str(a.get("command", "")) == "route_design"]

    def test_deep_wns_dispatches_controller(self):
        self.opt.best_wns = -5.0
        _async(self.opt._run_bare_reroute_polish())
        self.assertEqual(len(self.ctrl_calls), 1)
        self.assertEqual(self._route_calls(), [])  # plain loop did NOT run

    def test_shallow_wns_keeps_plain_loop(self):
        self.opt.best_wns = -0.4
        _async(self.opt._run_bare_reroute_polish())
        self.assertEqual(self.ctrl_calls, [])
        self.assertEqual(len(self._route_calls()), 1)

    def test_kill_switch_keeps_plain_loop_on_deep_wns(self):
        self.opt.best_wns = -5.0
        self.opt._tail_controller_enabled = False
        _async(self.opt._run_bare_reroute_polish())
        self.assertEqual(self.ctrl_calls, [])
        self.assertEqual(len(self._route_calls()), 1)

    def test_boundary_arms_at_threshold(self):
        self.opt.best_wns = TAIL_CTRL_DEEP_WNS_NS_DEFAULT  # exactly -1.0
        _async(self.opt._run_bare_reroute_polish())
        self.assertEqual(len(self.ctrl_calls), 1)

    def test_fail_closed_falls_back_to_plain_loop(self):
        self.opt.best_wns = -5.0

        async def crashing_controller(min_gain_ns):
            raise RuntimeError("controller bug")

        self.opt._run_tail_controller = crashing_controller
        _async(self.opt._run_bare_reroute_polish())   # must not raise
        # Fail-closed recovery and the plain routing loop account for two opens.
        # Insured comparison opens the banked best again for final selection.
        opens = [a for (n, a) in self.call_log
                 if n == "vivado_open_checkpoint"]
        self.assertEqual(len(opens), 3)
        self.assertEqual(len(self._route_calls()), 1)


class AutoBankSuppressionTests(unittest.TestCase):
    """The one-line hook edit: _tail_ctrl_suppress_autobank gates the
    R-D1-1 auto-bank hook (test_auto_bank.py harness style)."""

    HEAVY_TOOL = "vivado_phys_opt_design"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = _make_optimizer(Path(self.tmp.name))
        self.opt.vivado_session.response_text = "phys_opt_design done"
        self.opt.best_wns = -2.0
        self.opt._budget_deadline = time.time() + 5000.0
        self.measure_calls: list[object] = []
        self.mirror_calls: list[bool] = []

        async def fake_measure(call_tool_fn):
            self.measure_calls.append(call_tool_fn)
            return -1.0

        async def fake_routed_ok():
            return True

        async def fake_mirror(eager: bool = False):
            self.mirror_calls.append(eager)

        self.opt.get_wns_for_target_clock = fake_measure
        self.opt._routed_ok_for_best = fake_routed_ok
        self.opt._mirror_best_valid_now = fake_mirror

    def tearDown(self):
        self.tmp.cleanup()

    def test_suppressed_hook_does_not_bank(self):
        self.opt._tail_ctrl_suppress_autobank = True
        _async(self.opt.call_tool(self.HEAVY_TOOL, {}))
        self.assertEqual(self.measure_calls, [])
        self.assertEqual(self.mirror_calls, [])
        self.assertEqual(self.opt.best_wns, -2.0)

    def test_unsuppressed_hook_banks(self):
        self.opt._tail_ctrl_suppress_autobank = False
        _async(self.opt.call_tool(self.HEAVY_TOOL, {}))
        self.assertEqual(len(self.mirror_calls), 1)
        self.assertEqual(self.opt.best_wns, -1.0)

    def test_hook_hold_guard_rejects_dirty_bank(self):
        # OOD stress finding 1: an improvement event with measurable
        # whs < 0 must NOT bank (validator gates hold_passed -> alpha=0).
        from unittest import mock as _mock

        async def dirty_hold(call_tool_fn, timeout_s=600.0):
            return -0.05

        with _mock.patch("optimizer.ils_polish._measure_hold",
                         new=dirty_hold):
            _async(self.opt.call_tool(self.HEAVY_TOOL, {}))
        self.assertEqual(self.mirror_calls, [])
        self.assertEqual(self.opt.best_wns, -2.0)

    def test_hook_hold_guard_fails_open_on_unmeasurable(self):
        # Unmeasurable hold (None) banks exactly as before — a measurement
        # gap can never re-open the boom alpha=0 hole the hook closes.
        from unittest import mock as _mock

        async def none_hold(call_tool_fn, timeout_s=600.0):
            return None

        with _mock.patch("optimizer.ils_polish._measure_hold",
                         new=none_hold):
            _async(self.opt.call_tool(self.HEAVY_TOOL, {}))
        self.assertEqual(len(self.mirror_calls), 1)
        self.assertEqual(self.opt.best_wns, -1.0)

    def test_hook_hold_probe_reaches_session_with_correct_tool_name(self):
        # Exercise the real dispatch path because it prefixes tool names.
        # The assertion verifies that hold-analysis Tcl reaches the session.
        _async(self.opt.call_tool(self.HEAVY_TOOL, {}))
        hold_calls = [
            (n, a) for (n, a) in self.opt.vivado_session.calls
            if "-hold" in str(a.get("command", ""))]
        assert hold_calls, "hold probe never reached the session"
        # call_tool strips the "vivado_" prefix at dispatch: correct
        # session-level name is "run_tcl"; the bug arrived as
        # "vivado_run_tcl" (only the first prefix stripped).
        for n, _a in hold_calls:
            self.assertEqual(n, "run_tcl")
        bogus = [n for (n, _a) in self.opt.vivado_session.calls
                 if n.startswith("vivado_")]
        self.assertEqual(bogus, [])

    def test_hook_cell_count_guard_rejects_gutted_design(self):
        # panel Q2: logic deletion (remove_cell) is tracker-
        # invisible and can improve WNS while failing equivalence.
        # A session reporting a cell count below 0.5x entry must not bank.
        self.opt._input_cell_count = 100_000
        self.opt.vivado_session.response_text = "42"  # llength -> 42 cells
        _async(self.opt.call_tool(self.HEAVY_TOOL, {}))
        self.assertEqual(self.mirror_calls, [])
        self.assertEqual(self.opt.best_wns, -2.0)

    def test_hook_cell_count_guard_passes_in_band(self):
        self.opt._input_cell_count = 100_000
        self.opt.vivado_session.response_text = "101000"
        _async(self.opt.call_tool(self.HEAVY_TOOL, {}))
        self.assertEqual(len(self.mirror_calls), 1)
        self.assertEqual(self.opt.best_wns, -1.0)

    def test_hook_cell_count_guard_fails_open_without_entry_count(self):
        self.opt._input_cell_count = None
        _async(self.opt.call_tool(self.HEAVY_TOOL, {}))
        self.assertEqual(len(self.mirror_calls), 1)

    def test_hook_hold_guard_clean_hold_banks(self):
        from unittest import mock as _mock

        async def clean_hold(call_tool_fn, timeout_s=600.0):
            return 0.012

        with _mock.patch("optimizer.ils_polish._measure_hold",
                         new=clean_hold):
            _async(self.opt.call_tool(self.HEAVY_TOOL, {}))
        self.assertEqual(len(self.mirror_calls), 1)
        self.assertEqual(self.opt.best_wns, -1.0)


if __name__ == "__main__":
    unittest.main()
