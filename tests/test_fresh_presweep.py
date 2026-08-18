"""FRESH-STATE ROUTE-LOTTERY PRE-SWEEP tests (jul23 panel Q3 fantasy #1).

Mechanism under test (dcp_optimizer.py): when --fresh-presweep-draws /
FPL26_FRESH_PRESWEEP_DRAWS resolves to K in 1..3, perform_initial_analysis
takes K banked route re-rolls on the PRISTINE input state at step 0
(after the entry WNS is measured, before all remaining Phase-1 feature
capture and the recipe/LLM loop) and keeps the best draw — never-worse
vs entry — as the pipeline entry state.

Evidence being encoded (farm_validation_jul22/report.md §3): a bare
`route_design -unroute` + `route_design -directive AggressiveExplore`
re-roll from the PRISTINE organizer state gains +0.070 (fir) / +0.306
(vtr) / +0.075 (optical) deterministically, 10/10 hold-clean; the
re-roll decays (or regresses) on optimized states — hence step 0 only.

Load-bearing invariants:
  - DEFAULT OFF (0/unset): ZERO tool calls, zero behavior change;
  - resolver follows the house convention (CLI wins over env;
    unparseable/negative keeps the default OFF; K clamped 1..3);
  - budget gate: first-draw-measures — draw 1 capped at 0.12 x wall,
    its OBSERVED cost gates draws 2..K
    (spent + observed * 1.3 <= 0.2 x wall);
  - banking discipline: tracker-first routed gate + hold gate
    (whs < 0 rejects; whs None fails OPEN loud, auto-bank precedent);
    adoption goes through the exact eager-mirror path;
  - a worse/failed draw re-opens the PRISTINE input DCP; the pipeline
    is NEVER handed a worse-than-entry state (hand-off re-opens the
    banked best_valid mirror when a later draw was rejected after an
    adoption);
  - initial_wns stays PRISTINE (improvement accounting); the recipe
    router / pathology feature view sees the POST-sweep WNS.

Harness style mirrors tests/test_tail_reserve.py /
tests/test_tail_controller.py: stubbed sessions / scripted draws, NO
Vivado, NO RapidWright, NO network, NO real sleeps.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import (
    FRESH_PRESWEEP_BUDGET_FRAC,
    FRESH_PRESWEEP_COST_SAFETY,
    FRESH_PRESWEEP_DRAWS_DEFAULT,
    FRESH_PRESWEEP_DRAWS_MAX,
    FRESH_PRESWEEP_FIRST_DRAW_FRAC,
    DCPOptimizer,
    _presweep_draw_allowance,
    resolve_fresh_presweep_draws,
)

_ENV = "FPL26_FRESH_PRESWEEP_DRAWS"


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

    def __init__(self, response_text: str = "ok"):
        self.response_text = response_text
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        return _FakeResult(self.response_text)


def _make_optimizer(tmp_path: Path, *, draws: int = 0,
                    initial_wns: float = -0.5,
                    max_wall: float | None = 3300.0) -> DCPOptimizer:
    opt = DCPOptimizer(api_key="test", run_dir=tmp_path)
    opt.vivado_session = _FakeSession()
    opt.rapidwright_session = _FakeSession()
    opt._fresh_presweep_draws = draws
    opt.initial_wns = initial_wns
    opt.best_wns = initial_wns
    opt.max_wall_seconds = max_wall
    # jul23 panel #2: production default is CANDIDATE mode (adopt-entry is
    # DEAD 5/5).  These legacy tests pin the adopt-entry KILL-SWITCH path so
    # they keep exercising the preserved hand-off/bank mechanics unchanged;
    # candidate-mode behaviour is covered by CandidateModeTests below +
    # tests/test_mux_compare.py.
    opt._presweep_adopt_entry = True
    return opt


# ---------------------------------------------------------------------------
# Resolver convention (house style: CLI wins over env; invalid keeps default)
# ---------------------------------------------------------------------------

class ResolverTests(unittest.TestCase):

    def setUp(self):
        self._env_patcher = mock.patch.dict(os.environ)
        self._env_patcher.start()
        os.environ.pop(_ENV, None)

    def tearDown(self):
        self._env_patcher.stop()

    def test_default_is_off(self):
        self.assertEqual(FRESH_PRESWEEP_DRAWS_DEFAULT, 0)
        self.assertEqual(resolve_fresh_presweep_draws(None), 0)

    def test_cli_value(self):
        self.assertEqual(resolve_fresh_presweep_draws(2), 2)

    def test_cli_zero_is_explicit_off(self):
        self.assertEqual(resolve_fresh_presweep_draws(0), 0)

    def test_env_value(self):
        os.environ[_ENV] = "3"
        self.assertEqual(resolve_fresh_presweep_draws(None), 3)

    def test_cli_wins_over_env(self):
        os.environ[_ENV] = "3"
        self.assertEqual(resolve_fresh_presweep_draws(1), 1)

    def test_clamp_above_max(self):
        self.assertEqual(FRESH_PRESWEEP_DRAWS_MAX, 3)
        self.assertEqual(resolve_fresh_presweep_draws(9), 3)
        os.environ[_ENV] = "12"
        self.assertEqual(resolve_fresh_presweep_draws(None), 3)

    def test_negative_keeps_default_off(self):
        self.assertEqual(resolve_fresh_presweep_draws(-1), 0)
        os.environ[_ENV] = "-2"
        self.assertEqual(resolve_fresh_presweep_draws(None), 0)

    def test_unparseable_env_keeps_default_off(self):
        os.environ[_ENV] = "junk"
        self.assertEqual(resolve_fresh_presweep_draws(None), 0)

    def test_unparseable_cli_keeps_default_off(self):
        self.assertEqual(resolve_fresh_presweep_draws("junk"), 0)

    def test_empty_env_keeps_default_off(self):
        os.environ[_ENV] = ""
        self.assertEqual(resolve_fresh_presweep_draws(None), 0)


# ---------------------------------------------------------------------------
# Budget gate: first-draw-measures (pure function)
# ---------------------------------------------------------------------------

class DrawAllowanceTests(unittest.TestCase):
    # Shape for a 3300 s wall: budget 660 s, first-draw timeout 396 s.
    BUDGET = FRESH_PRESWEEP_BUDGET_FRAC * 3300.0
    FIRST = FRESH_PRESWEEP_FIRST_DRAW_FRAC * 3300.0

    def test_first_draw_allowed_with_first_draw_timeout(self):
        allowed, timeout, reason = _presweep_draw_allowance(
            1, 0.0, None, self.BUDGET, self.FIRST)
        self.assertTrue(allowed)
        self.assertEqual(timeout, self.FIRST)
        self.assertEqual(reason, "first_draw_measures")

    def test_first_draw_timeout_bounded_by_budget(self):
        allowed, timeout, _ = _presweep_draw_allowance(
            1, 0.0, None, 100.0, 396.0)
        self.assertTrue(allowed)
        self.assertEqual(timeout, 100.0)

    def test_zero_budget_refuses_first_draw(self):
        allowed, _, reason = _presweep_draw_allowance(1, 0.0, None, 0.0, 0.0)
        self.assertFalse(allowed)
        self.assertEqual(reason, "no_budget")

    def test_second_draw_gated_by_observed_cost(self):
        # spent 100, observed 100 -> predicted 130; 100+130 <= 660: allowed
        allowed, timeout, reason = _presweep_draw_allowance(
            2, 100.0, 100.0, self.BUDGET, self.FIRST)
        self.assertTrue(allowed)
        self.assertAlmostEqual(timeout, 100.0 * FRESH_PRESWEEP_COST_SAFETY)
        self.assertEqual(reason, "cost_gated")

    def test_second_draw_refused_when_prediction_overruns(self):
        # spent 600, observed 100 -> predicted 130; 600+130 > 660: refused
        allowed, _, reason = _presweep_draw_allowance(
            2, 600.0, 100.0, self.BUDGET, self.FIRST)
        self.assertFalse(allowed)
        self.assertIn("budget_gate", reason)

    def test_second_draw_exact_boundary_allowed(self):
        # spent + observed*1.3 == budget exactly: allowed (<=)
        allowed, _, _ = _presweep_draw_allowance(
            2, 530.0, 100.0, 660.0, self.FIRST)
        self.assertTrue(allowed)

    def test_second_draw_timeout_is_inflated_prediction(self):
        # observed 400 -> predicted 520; remaining 660 -> timeout = 520.
        allowed, timeout, _ = _presweep_draw_allowance(
            2, 500.0, 400.0, 1160.0, self.FIRST)
        self.assertTrue(allowed)
        self.assertEqual(timeout, 400.0 * FRESH_PRESWEEP_COST_SAFETY)

    def test_second_draw_timeout_bounded_by_remaining_budget(self):
        # observed 400 -> predicted 520; spent 100 of 620 -> remaining
        # 520 == predicted; boundary allowed and timeout == remaining.
        allowed, timeout, _ = _presweep_draw_allowance(
            2, 100.0, 400.0, 620.0, self.FIRST)
        self.assertTrue(allowed)
        self.assertEqual(timeout, 520.0)

    def test_no_cost_sample_fails_closed(self):
        allowed, _, reason = _presweep_draw_allowance(
            2, 0.0, None, self.BUDGET, self.FIRST)
        self.assertFalse(allowed)
        self.assertEqual(reason, "no_cost_sample")
        allowed, _, reason = _presweep_draw_allowance(
            2, 0.0, 0.0, self.BUDGET, self.FIRST)
        self.assertFalse(allowed)
        self.assertEqual(reason, "no_cost_sample")


# ---------------------------------------------------------------------------
# OFF = zero calls / entry gating
# ---------------------------------------------------------------------------

class OffAndGatingTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.input_dcp = self.tmp_path / "pristine.dcp"
        self.input_dcp.write_bytes(b"PRISTINE")

    def tearDown(self):
        self.tmp.cleanup()

    def _spy(self, opt):
        spy = mock.AsyncMock(return_value="ok")
        return mock.patch.object(opt, "call_tool", new=spy), spy

    def test_constructor_default_off(self):
        opt = _make_optimizer(self.tmp_path)
        self.assertEqual(opt._fresh_presweep_draws, 0)
        self.assertFalse(opt._presweep_adopted)

    def test_off_zero_tool_calls(self):
        opt = _make_optimizer(self.tmp_path, draws=0, initial_wns=-2.0)
        patcher, spy = self._spy(opt)
        with patcher:
            _async(opt._run_fresh_presweep(self.input_dcp))
        spy.assert_not_awaited()
        self.assertFalse(opt._presweep_adopted)
        self.assertEqual(opt._presweep_draw_records, [])

    def test_unmeasured_entry_wns_skips_with_zero_calls(self):
        opt = _make_optimizer(self.tmp_path, draws=2, initial_wns=-2.0)
        opt.initial_wns = None
        patcher, spy = self._spy(opt)
        with patcher, self.assertLogs("dcp_optimizer", level="INFO") as logs:
            _async(opt._run_fresh_presweep(self.input_dcp))
        spy.assert_not_awaited()
        self.assertIn("cannot gate never-worse", "\n".join(logs.output))

    def test_timing_met_entry_skips_with_zero_calls(self):
        opt = _make_optimizer(self.tmp_path, draws=2, initial_wns=0.1)
        patcher, spy = self._spy(opt)
        with patcher, self.assertLogs("dcp_optimizer", level="INFO") as logs:
            _async(opt._run_fresh_presweep(self.input_dcp))
        spy.assert_not_awaited()
        self.assertIn("timing already met", "\n".join(logs.output))


# ---------------------------------------------------------------------------
# Sweep driver: adopt / reject / error, budget, never-hand-off-worse
# ---------------------------------------------------------------------------

class _SweepHarness(unittest.TestCase):
    """Drives _run_fresh_presweep with a scripted _presweep_execute_draw
    and a scripted clock so per-draw costs are deterministic."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.input_dcp = self.tmp_path / "pristine.dcp"
        self.input_dcp.write_bytes(b"PRISTINE")

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, *, draws, initial_wns=-2.0, max_wall=3300.0,
             script=None, draw_cost=100.0, best_valid_exists=False,
             candidate_mode=False):
        """script: list of (wns, whs, routed, err) tuples returned by
        successive _presweep_execute_draw calls.  Each draw advances the
        scripted clock by draw_cost seconds.

        candidate_mode=True flips the optimizer to the production
        INSURED-COMPARE default (adopt-entry OFF) and stubs
        register_final_candidate so the branch can be asserted without
        Vivado."""
        opt = _make_optimizer(self.tmp_path, draws=draws,
                              initial_wns=initial_wns, max_wall=max_wall)
        if candidate_mode:
            opt._presweep_adopt_entry = False
        self.register_spy = mock.AsyncMock(return_value=True)
        opt.register_final_candidate = self.register_spy
        clock = {"t": 1000.0}

        def fake_now():
            return clock["t"]

        script = list(script or [])
        self.ops_drawn: list[str] = []

        async def fake_draw(op, timeout_s):
            self.ops_drawn.append(op)
            clock["t"] += draw_cost
            if not script:
                raise AssertionError("draw script exhausted")
            return script.pop(0)

        self.reopened: list[str] = []

        async def fake_reopen(dcp_path):
            self.reopened.append(Path(dcp_path).name)
            return True

        mirror = mock.MagicMock()
        if best_valid_exists:
            bv = self.tmp_path / "best_valid.dcp"
            bv.write_bytes(b"BEST")
            opt._best_valid_dcp = bv

        async def fake_mirror(eager=False):
            # Mimic the real mirror's side effect the hand-off relies on.
            if opt._best_valid_dcp is None:
                bv = self.tmp_path / "best_valid.dcp"
                bv.write_bytes(b"BEST")
                opt._best_valid_dcp = bv
            mirror(eager=eager)

        refresh_spy = mock.AsyncMock(return_value="err")  # non-parse -> fail-open
        with mock.patch.object(opt, "_presweep_now", side_effect=fake_now), \
             mock.patch.object(opt, "_presweep_execute_draw",
                               side_effect=fake_draw), \
             mock.patch.object(opt, "_presweep_reopen",
                               side_effect=fake_reopen), \
             mock.patch.object(opt, "_mirror_best_valid_now",
                               side_effect=fake_mirror), \
             mock.patch.object(opt, "call_tool", new=refresh_spy), \
             self.assertLogs("dcp_optimizer", level="INFO") as logs:
            _async(opt._run_fresh_presweep(self.input_dcp))
        self.mirror = mirror
        self.refresh_spy = refresh_spy
        return opt, "\n".join(logs.output)


class AdoptRejectErrorTests(_SweepHarness):

    def test_adopted_draw_banks_and_updates_best(self):
        opt, logs = self._run(draws=1, initial_wns=-2.0,
                              script=[(-1.7, 0.05, True, None)])
        self.assertEqual(opt.best_wns, -1.7)
        self.assertTrue(opt._presweep_adopted)
        self.assertEqual(opt._presweep_post_wns, -1.7)
        # initial_wns stays PRISTINE (accounting invariant).
        self.assertEqual(opt.initial_wns, -2.0)
        self.mirror.assert_called_once_with(eager=True)
        # In-memory state IS the adopted state: no pristine re-open.
        self.assertEqual(self.reopened, [])
        self.assertIn("verdict=ADOPTED", logs)
        self.assertIn("wns -2.000 -> -1.700", logs)
        self.assertIn("whs=0.050", logs)
        # Requirement-5 shape.
        self.assertIn("draw=1/1", logs)
        self.assertRegex(logs, r"cost=\d+s remaining=\d+s")

    def test_rejected_draw_reopens_pristine_no_bank(self):
        opt, logs = self._run(draws=1, initial_wns=-2.0,
                              script=[(-2.3, 0.05, True, None)])
        self.assertEqual(opt.best_wns, -2.0)
        self.assertFalse(opt._presweep_adopted)
        self.mirror.assert_not_called()
        self.assertEqual(self.reopened, ["pristine.dcp"])
        self.assertIn("verdict=REJECTED", logs)
        self.assertIn("no strict improvement", logs)

    def test_equal_wns_is_rejected_never_worse_is_strict(self):
        opt, logs = self._run(draws=1, initial_wns=-2.0,
                              script=[(-2.0, 0.05, True, None)])
        self.assertFalse(opt._presweep_adopted)
        self.assertIn("verdict=REJECTED", logs)

    def test_error_draw_reopens_pristine(self):
        opt, logs = self._run(draws=1, initial_wns=-2.0,
                              script=[(None, None, False, "op_failed:ERROR")])
        self.assertEqual(opt.best_wns, -2.0)
        self.mirror.assert_not_called()
        self.assertEqual(self.reopened, ["pristine.dcp"])
        self.assertIn("verdict=ERROR", logs)

    def test_unrouted_draw_rejected_by_tracker_gate(self):
        # WNS improves but routed gate says NO (phantom class): reject.
        opt, logs = self._run(draws=1, initial_wns=-2.0,
                              script=[(-1.5, 0.05, False, None)])
        self.assertEqual(opt.best_wns, -2.0)
        self.mirror.assert_not_called()
        self.assertIn("verdict=REJECTED", logs)
        self.assertIn("not_routed", logs)

    def test_hold_dirty_draw_rejected(self):
        opt, logs = self._run(draws=1, initial_wns=-2.0,
                              script=[(-1.5, -0.02, True, None)])
        self.assertEqual(opt.best_wns, -2.0)
        self.mirror.assert_not_called()
        self.assertIn("verdict=REJECTED", logs)
        self.assertIn("hold_dirty", logs)

    def test_hold_unmeasurable_fails_open_loud(self):
        # whs None on an otherwise-adoptable draw: bank (auto-bank
        # precedent — a measurement gap must not strand a real gain)
        # with a LOUD warning.
        opt, logs = self._run(draws=1, initial_wns=-2.0,
                              script=[(-1.5, None, True, None)])
        self.assertTrue(opt._presweep_adopted)
        self.assertEqual(opt.best_wns, -1.5)
        self.assertEqual(opt._presweep_hold_failopen_count, 1)
        self.assertIn("HOLD UNMEASURABLE", logs)

    def test_first_draw_uses_measured_two_op_form(self):
        self._run(draws=1, initial_wns=-2.0,
                  script=[(-1.7, 0.05, True, None)])
        self.assertEqual(self.ops_drawn, ["unroute_ae"])

    def test_chained_draw_after_adoption_uses_bare_route(self):
        self._run(draws=2, initial_wns=-2.0,
                  script=[(-1.7, 0.05, True, None),
                          (-1.6, 0.05, True, None)])
        self.assertEqual(self.ops_drawn, ["unroute_ae", "bare"])

    def test_second_adoption_compounds_best(self):
        opt, _ = self._run(draws=2, initial_wns=-2.0,
                           script=[(-1.7, 0.05, True, None),
                                   (-1.6, 0.05, True, None)])
        self.assertEqual(opt.best_wns, -1.6)
        self.assertEqual(opt._presweep_post_wns, -1.6)
        self.assertEqual(self.mirror.call_count, 2)

    def test_deterministic_duplicate_stops_sweep(self):
        # Draw 1 (unroute_ae from pristine) rejected -> pristine re-open;
        # draw 2 switches to bare-on-pristine; draw 3 would repeat
        # (pristine, bare) -> deterministic duplicate -> stop.
        opt, logs = self._run(draws=3, initial_wns=-2.0,
                              script=[(-2.5, 0.05, True, None),
                                      (-2.4, 0.05, True, None)])
        self.assertEqual(self.ops_drawn, ["unroute_ae", "bare"])
        self.assertIn("deterministic duplicate", logs)
        self.assertEqual(len(opt._presweep_draw_records), 2)


class BudgetGatingTests(_SweepHarness):

    def test_first_draw_cost_gates_second_draw(self):
        # Wall 3300 -> budget 660. Draw 1 costs 400: 400 + 400*1.3=920
        # > 660 -> draw 2 refused by the budget gate.
        opt, logs = self._run(draws=2, initial_wns=-2.0, draw_cost=400.0,
                              script=[(-1.7, 0.05, True, None)])
        self.assertEqual(len(self.ops_drawn), 1)
        self.assertIn("budget_gate", logs)
        self.assertIn("stopping sweep", logs)

    def test_cheap_first_draw_allows_full_k(self):
        # Draw cost 100: 100+130 <= 660, 200+130 <= 660 -> all 3 draws.
        opt, _ = self._run(draws=3, initial_wns=-2.0, draw_cost=100.0,
                           script=[(-1.9, 0.05, True, None),
                                   (-1.8, 0.05, True, None),
                                   (-1.7, 0.05, True, None)])
        self.assertEqual(len(self.ops_drawn), 3)
        self.assertEqual(opt.best_wns, -1.7)

    def test_every_budget_decision_is_logged(self):
        _, logs = self._run(draws=2, initial_wns=-2.0, draw_cost=400.0,
                            script=[(-1.7, 0.05, True, None)])
        self.assertIn("first-draw", logs)          # first-draw-measures shape
        self.assertIn("budget=660s", logs)
        self.assertRegex(logs, r"draw=2/2 skipped: budget_gate")

    def test_global_wall_floor_refuses_draw(self):
        # Deadline nearly exhausted: even though the pre-sweep budget
        # allows the draw, the global wall floor refuses it.
        opt = _make_optimizer(self.tmp_path, draws=1, initial_wns=-2.0,
                              max_wall=3300.0)
        import time as _time
        opt._budget_deadline = _time.time() + 50.0  # < timeout + 120 margin
        draw = mock.AsyncMock()
        with mock.patch.object(opt, "_presweep_execute_draw", new=draw), \
             self.assertLogs("dcp_optimizer", level="INFO") as logs:
            _async(opt._run_fresh_presweep(self.input_dcp))
        draw.assert_not_awaited()
        self.assertIn("global wall floor", "\n".join(logs.output))


class HandOffTests(_SweepHarness):

    def test_adopt_then_reject_hands_off_best_valid(self):
        # Draw 1 adopted (mirror written), draw 2 rejected -> in-memory
        # is pristine -> hand-off must re-open the banked best mirror.
        opt, logs = self._run(draws=2, initial_wns=-2.0,
                              script=[(-1.7, 0.05, True, None),
                                      (-1.9, 0.05, True, None)])
        self.assertTrue(opt._presweep_adopted)
        self.assertEqual(opt.best_wns, -1.7)
        # Rejected draw re-opened pristine, then hand-off re-opened best.
        self.assertEqual(self.reopened, ["pristine.dcp", "best_valid.dcp"])
        self.assertIn("hand-off: re-opened banked best draw", logs)

    def test_all_rejected_hands_off_pristine(self):
        opt, _ = self._run(draws=2, initial_wns=-2.0,
                           script=[(-2.5, 0.05, True, None),
                                   (-2.4, 0.05, True, None)])
        self.assertFalse(opt._presweep_adopted)
        self.assertEqual(opt.best_wns, -2.0)
        # Every rejection re-opened pristine; no best_valid re-open.
        self.assertEqual(self.reopened, ["pristine.dcp", "pristine.dcp"])

    def test_never_hand_off_worse_when_mirror_reopen_fails(self):
        # Adoption banked, later draw rejected, best_valid re-open FAILS:
        # the pipeline enters at PRISTINE (never worse-than-entry) and
        # the feature view falls back to the pristine WNS.
        opt = _make_optimizer(self.tmp_path, draws=2, initial_wns=-2.0)
        script = [(-1.7, 0.05, True, None), (-1.9, 0.05, True, None)]
        clock = {"t": 0.0}

        async def fake_draw(op, timeout_s):
            clock["t"] += 100.0
            return script.pop(0)

        reopened = []

        async def fake_reopen(dcp_path):
            reopened.append(Path(dcp_path).name)
            # pristine re-opens succeed; the best_valid mirror fails
            return Path(dcp_path).name != "best_valid.dcp"

        bv = self.tmp_path / "best_valid.dcp"
        bv.write_bytes(b"BEST")
        with mock.patch.object(opt, "_presweep_now",
                               side_effect=lambda: clock["t"]), \
             mock.patch.object(opt, "_presweep_execute_draw",
                               side_effect=fake_draw), \
             mock.patch.object(opt, "_presweep_reopen",
                               side_effect=fake_reopen), \
             mock.patch.object(opt, "_mirror_best_valid_now",
                               new=mock.AsyncMock(
                                   side_effect=lambda eager=False:
                                   setattr(opt, "_best_valid_dcp", bv))), \
             mock.patch.object(opt, "call_tool",
                               new=mock.AsyncMock(return_value="err")), \
             self.assertLogs("dcp_optimizer", level="INFO") as logs:
            _async(opt._run_fresh_presweep(self.input_dcp))
        text = "\n".join(logs.output)
        self.assertIn("pipeline enters at PRISTINE", text)
        # best_wns keeps the banked value (the mirror's contract) but the
        # FEATURE VIEW describes the pristine state the LLM will work.
        self.assertEqual(opt.best_wns, -1.7)
        self.assertFalse(opt._presweep_adopted)
        self.assertEqual(opt._phase1_wns_for_features(), -2.0)


# ---------------------------------------------------------------------------
# INSURED-COMPARE candidate mode (jul23 panel #2, drill #1) — the
# production DEFAULT: the pre-sweep REGISTERS its best draw as a FINAL
# candidate and hands the pipeline a PRISTINE entry (never adopts).
# ---------------------------------------------------------------------------

class CandidateModeTests(_SweepHarness):

    def test_registers_not_adopts_pipeline_sees_pristine(self):
        # Adopted-quality draw (+0.3): in candidate mode the sweep
        # REGISTERS it and RESETS the pipeline to pristine — best_wns
        # back to entry, mirror pointer cleared, adopted flag False.
        opt, logs = self._run(draws=1, initial_wns=-2.0, max_wall=9000.0,
                              script=[(-1.7, 0.05, True, None)],
                              candidate_mode=True)
        # Candidate registered exactly once as "presweep".
        self.register_spy.assert_awaited_once()
        args, kwargs = self.register_spy.await_args
        self.assertEqual(args[1], -1.7)          # candidate WNS
        self.assertEqual(args[2], "presweep")    # source label
        # Pipeline sees PRISTINE: lineage reset, not the perturbed draw.
        self.assertEqual(opt.best_wns, -2.0)
        self.assertIsNone(opt._best_valid_dcp)
        self.assertIsNone(opt._best_valid_dcp_wns)
        self.assertFalse(opt._presweep_adopted)
        self.assertIsNone(opt._presweep_post_wns)
        # Final action re-opens pristine for the pipeline entry.
        self.assertEqual(self.reopened[-1], "pristine.dcp")
        self.assertIn("COMPLETE (candidate mode)", logs)
        self.assertIn("pipeline enters PRISTINE", logs)

    def test_cost_gate_skips_small_wall_zero_draws(self):
        # Candidate mode + wall below the 4500s gate: no draws at all.
        opt, logs = self._run(draws=2, initial_wns=-2.0, max_wall=3600.0,
                              script=[(-1.7, 0.05, True, None)],
                              candidate_mode=True)
        self.assertEqual(len(self.ops_drawn), 0)
        self.register_spy.assert_not_awaited()
        self.assertIn("skip (candidate mode)", logs)
        self.assertIn("cost gate", logs)

    def test_cost_gate_arms_large_wall(self):
        # Candidate mode + wall above the gate: the sweep runs.
        opt, _ = self._run(draws=1, initial_wns=-2.0, max_wall=9000.0,
                           script=[(-1.7, 0.05, True, None)],
                           candidate_mode=True)
        self.assertEqual(len(self.ops_drawn), 1)
        self.register_spy.assert_awaited_once()

    def test_all_rejected_registers_nothing_pipeline_pristine(self):
        # No draw beats entry -> nothing to register; pipeline pristine.
        opt, logs = self._run(draws=2, initial_wns=-2.0, max_wall=9000.0,
                              script=[(-2.5, 0.05, True, None),
                                      (-2.4, 0.05, True, None)],
                              candidate_mode=True)
        self.register_spy.assert_not_awaited()
        self.assertEqual(opt.best_wns, -2.0)
        self.assertFalse(opt._presweep_adopted)
        self.assertIn("registered=False", logs)


# ---------------------------------------------------------------------------
# Feature view (requirement 4) + autobank suppression + allowance shift
# ---------------------------------------------------------------------------

class FeatureViewTests(_SweepHarness):

    def test_feature_view_pristine_when_off(self):
        opt = _make_optimizer(self.tmp_path, initial_wns=-2.0)
        opt.initial_failing_endpoints = 42
        self.assertEqual(opt._phase1_wns_for_features(), -2.0)
        self.assertEqual(opt._phase1_failing_endpoints_for_features(), 42)

    def test_feature_view_post_sweep_when_adopted(self):
        opt, _ = self._run(draws=1, initial_wns=-2.0,
                           script=[(-1.7, 0.05, True, None)])
        self.assertEqual(opt._phase1_wns_for_features(), -1.7)
        # initial_wns pristine for accounting.
        self.assertEqual(opt.initial_wns, -2.0)

    def test_feature_view_endpoints_fall_back_when_refresh_fails(self):
        opt, _ = self._run(draws=1, initial_wns=-2.0,
                           script=[(-1.7, 0.05, True, None)])
        opt.initial_failing_endpoints = 42
        # refresh returned an error envelope in the harness -> fall back.
        self.assertIsNone(opt._presweep_post_failing_endpoints)
        self.assertEqual(opt._phase1_failing_endpoints_for_features(), 42)

    def test_adoption_triggers_timing_refresh_call(self):
        self._run(draws=1, initial_wns=-2.0,
                  script=[(-1.7, 0.05, True, None)])
        called = [c.args[0] for c in self.refresh_spy.await_args_list]
        self.assertIn("vivado_report_timing_summary", called)

    def test_no_refresh_when_nothing_adopted(self):
        self._run(draws=1, initial_wns=-2.0,
                  script=[(-2.5, 0.05, True, None)])
        called = [c.args[0] for c in self.refresh_spy.await_args_list]
        self.assertNotIn("vivado_report_timing_summary", called)

    def test_phase1_allowance_shifted_by_sweep_elapsed(self):
        # Pre-sweep spend must NOT consume the C1-T4a Phase-1 allowance:
        # the anchor shifts forward by the sweep's elapsed time.
        opt = _make_optimizer(self.tmp_path, draws=1, initial_wns=-2.0)
        opt._phase1_start_ts = 500.0
        clock = {"t": 1000.0}

        async def fake_draw(op, timeout_s):
            clock["t"] += 123.0
            return (-2.5, 0.05, True, None)  # rejected

        with mock.patch.object(opt, "_presweep_now",
                               side_effect=lambda: clock["t"]), \
             mock.patch.object(opt, "_presweep_execute_draw",
                               side_effect=fake_draw), \
             mock.patch.object(opt, "_presweep_reopen",
                               new=mock.AsyncMock(return_value=True)), \
             mock.patch.object(opt, "call_tool",
                               new=mock.AsyncMock(return_value="ok")):
            _async(opt._run_fresh_presweep(self.input_dcp))
        self.assertEqual(opt._phase1_start_ts, 500.0 + 123.0)

    def test_autobank_suppressed_during_draw_and_restored(self):
        opt = _make_optimizer(self.tmp_path, draws=1, initial_wns=-2.0)
        seen = {}

        async def fake_draw(op, timeout_s):
            seen["suppressed"] = opt._tail_ctrl_suppress_autobank
            return (-2.5, 0.05, True, None)

        with mock.patch.object(opt, "_presweep_execute_draw",
                               side_effect=fake_draw), \
             mock.patch.object(opt, "_presweep_reopen",
                               new=mock.AsyncMock(return_value=True)), \
             mock.patch.object(opt, "call_tool",
                               new=mock.AsyncMock(return_value="ok")):
            _async(opt._run_fresh_presweep(self.input_dcp))
        self.assertTrue(seen["suppressed"])
        self.assertFalse(opt._tail_ctrl_suppress_autobank)


# ---------------------------------------------------------------------------
# Draw executor: op forms, error envelopes, hold probe dispatch (94e53bd)
# ---------------------------------------------------------------------------

class ExecuteDrawTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _opt(self):
        return _make_optimizer(self.tmp_path, draws=1, initial_wns=-2.0)

    def test_unroute_ae_sends_measured_two_op_command(self):
        opt = self._opt()
        calls = []

        async def fake_call_tool(name, args):
            calls.append((name, args))
            return "ok"

        with mock.patch.object(opt, "call_tool", side_effect=fake_call_tool), \
             mock.patch.object(opt, "get_wns_for_target_clock",
                               new=mock.AsyncMock(return_value=-1.7)), \
             mock.patch.object(opt, "_routed_ok_for_best",
                               new=mock.AsyncMock(return_value=True)), \
             mock.patch("optimizer.ils_polish._measure_hold",
                        new=mock.AsyncMock(return_value=0.05)):
            wns, whs, routed, err = _async(
                opt._presweep_execute_draw("unroute_ae", 396.0))
        self.assertIsNone(err)
        self.assertEqual((wns, whs, routed), (-1.7, 0.05, True))
        op_call = calls[0]
        self.assertEqual(op_call[0], "vivado_run_tcl")
        self.assertEqual(
            op_call[1]["command"],
            "route_design -unroute; route_design -directive AggressiveExplore")
        self.assertEqual(op_call[1]["timeout"], 396.0)

    def test_bare_form_sends_plain_route_design(self):
        opt = self._opt()
        calls = []

        async def fake_call_tool(name, args):
            calls.append((name, args))
            return "ok"

        with mock.patch.object(opt, "call_tool", side_effect=fake_call_tool), \
             mock.patch.object(opt, "get_wns_for_target_clock",
                               new=mock.AsyncMock(return_value=-1.7)), \
             mock.patch.object(opt, "_routed_ok_for_best",
                               new=mock.AsyncMock(return_value=True)), \
             mock.patch("optimizer.ils_polish._measure_hold",
                        new=mock.AsyncMock(return_value=0.05)):
            _async(opt._presweep_execute_draw("bare", 130.0))
        self.assertEqual(calls[0][1]["command"], "route_design")

    def test_op_error_envelope_returns_error_no_measurement(self):
        opt = self._opt()
        wns_probe = mock.AsyncMock()
        with mock.patch.object(
                opt, "call_tool",
                new=mock.AsyncMock(
                    return_value='{"error": "tool_skipped_budget"}')), \
             mock.patch.object(opt, "get_wns_for_target_clock",
                               new=wns_probe):
            wns, whs, routed, err = _async(
                opt._presweep_execute_draw("unroute_ae", 396.0))
        self.assertIsNone(wns)
        self.assertIsNotNone(err)
        self.assertTrue(err.startswith("op_failed"))
        wns_probe.assert_not_awaited()

    def test_op_exception_returns_error(self):
        opt = self._opt()
        with mock.patch.object(
                opt, "call_tool",
                new=mock.AsyncMock(side_effect=RuntimeError("boom"))):
            wns, whs, routed, err = _async(
                opt._presweep_execute_draw("unroute_ae", 396.0))
        self.assertIsNone(wns)
        self.assertEqual(err, "op_raised:RuntimeError")

    def test_unmeasurable_wns_returns_error(self):
        opt = self._opt()
        with mock.patch.object(opt, "call_tool",
                               new=mock.AsyncMock(return_value="ok")), \
             mock.patch.object(opt, "get_wns_for_target_clock",
                               new=mock.AsyncMock(return_value=None)):
            wns, whs, routed, err = _async(
                opt._presweep_execute_draw("unroute_ae", 396.0))
        self.assertEqual(err, "wns_unmeasurable")

    def test_hold_probe_uses_call_tool_not_prefixed_dispatch(self):
        # 94e53bd regression guard: _measure_hold must receive
        # self.call_tool (full tool names) — the prefixing
        # _call_vivado_tool dispatched the probe to the nonexistent
        # "vivado_vivado_run_tcl" and silently failed open.
        opt = self._opt()
        received = {}

        async def fake_measure_hold(call_tool, timeout_s=600.0):
            received["fn"] = call_tool
            return 0.05

        with mock.patch.object(opt, "call_tool",
                               new=mock.AsyncMock(return_value="ok")), \
             mock.patch.object(opt, "get_wns_for_target_clock",
                               new=mock.AsyncMock(return_value=-1.7)), \
             mock.patch.object(opt, "_routed_ok_for_best",
                               new=mock.AsyncMock(return_value=True)), \
             mock.patch("optimizer.ils_polish._measure_hold",
                        side_effect=fake_measure_hold):
            _async(opt._presweep_execute_draw("bare", 130.0))
            # Assert INSIDE the patch context (opt.call_tool is the
            # patched-in mock here; on exit it reverts to the bound
            # method).  The probe must have received exactly the
            # object bound at opt.call_tool — never a name-prefixing
            # wrapper like _call_vivado_tool.
            self.assertIs(received["fn"], opt.call_tool)

    def test_hold_probe_failure_fails_open_to_none(self):
        opt = self._opt()
        with mock.patch.object(opt, "call_tool",
                               new=mock.AsyncMock(return_value="ok")), \
             mock.patch.object(opt, "get_wns_for_target_clock",
                               new=mock.AsyncMock(return_value=-1.7)), \
             mock.patch.object(opt, "_routed_ok_for_best",
                               new=mock.AsyncMock(return_value=True)), \
             mock.patch("optimizer.ils_polish._measure_hold",
                        new=mock.AsyncMock(side_effect=RuntimeError("x"))):
            wns, whs, routed, err = _async(
                opt._presweep_execute_draw("bare", 130.0))
        self.assertIsNone(err)
        self.assertIsNone(whs)
        self.assertEqual(wns, -1.7)


if __name__ == "__main__":
    unittest.main()
