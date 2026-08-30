"""Test cumulative LLM-spend circuit breakers without external services.

The suite covers the cumulative ceiling, shrinking per-attempt budgets, and the
pre-call gate within an attempt. Stubs replace network access, Vivado, and
sleeping.
"""
from __future__ import annotations

import asyncio
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import (
    DCPOptimizer,
    LLM_COST_EXIT_USD,
    resolve_llm_cost_exit,
)
from scripts.multi_restart_optimize import (
    COST_CEILING_DEFAULT,
    _attempt_cmd,
    _default_cost_ceiling,
    cost_gate,
)
import scripts.multi_restart_optimize as mro


def _async(coro):
    return asyncio.run(coro)


# Part A — pure cost_gate (wrapper pre-launch predictive gate)

def _att(i, cost, fmax=100.0):
    return {"i": i, "fmax": fmax, "status": "VALID_OPTIMIZED",
            "cost": cost, "exists": True, "output": f"/tmp/mr_{i}.dcp"}


class CostGateTests(unittest.TestCase):
    def test_first_attempt_no_priors_launches_with_full_allowance(self):
        launch, allowance, _ = cost_gate([], 0.0, 0.80)
        self.assertTrue(launch)
        self.assertEqual(allowance, 0.80)

    def test_15_zeroing_shape_refused(self):
        # #15 replay shape: one expensive attempt ($0.55); launching another
        # like it would predict $1.10 cumulative — past the ceiling and past
        # the eval's $1.00 zero-cap.  Must NOT launch.
        launch, _, why = cost_gate([_att(1, 0.55)], 0.55, 0.80)
        self.assertFalse(launch)
        self.assertIn("#15", why)

    def test_retrospective_hole_closed(self):
        # The exact hole this closes: $0.84 spent < 0.85 cap used to launch a
        # fresh $0.75 allowance (worst case ~$1.59).  Ceiling refuses.
        launch, _, _ = cost_gate([_att(1, 0.44), _att(2, 0.40)], 0.84, 0.80)
        self.assertFalse(launch)

    def test_reh1_fir_trace_never_crosses_ceiling(self):
        # reh-1 fir-like accrual, replayed through the gate: cumulative KNOWN
        # spend must never cross the 0.80 ceiling and the predictive estimate
        # must stop the trace early ($0.54 + max-prior $0.28 > $0.80).
        costs = [0.28, 0.26, 0.22]
        attempts, cum = [], 0.0
        launched = 0
        for i, c in enumerate(costs, start=1):
            launch, allowance, _ = cost_gate(attempts, cum, 0.80)
            if not launch:
                break
            launched += 1
            self.assertLessEqual(cum + allowance, 0.80 + 1e-9,
                                 "allowance may never fund a ceiling cross")
            attempts.append(_att(i, c))
            cum += c
        self.assertEqual(launched, 2)          # third attempt refused
        self.assertLessEqual(cum, 0.80)        # 0.54, not reh-1's 0.76+

    def test_allowance_shrinks_with_spend(self):
        launch, allowance, _ = cost_gate([_att(1, 0.30)], 0.30, 0.80)
        self.assertTrue(launch)
        self.assertEqual(allowance, 0.50)      # ceiling − spent, cent-floored

    def test_allowance_truncated_down_to_cent(self):
        launch, allowance, _ = cost_gate([_att(1, 0.111)], 0.111, 0.80)
        self.assertTrue(launch)
        self.assertEqual(allowance, 0.68)      # 0.689 -> 0.68, conservative

    def test_spent_at_or_over_ceiling_refused(self):
        self.assertFalse(cost_gate([], 0.80, 0.80)[0])
        self.assertFalse(cost_gate([], 0.95, 0.80)[0])

    def test_tiny_allowance_refused(self):
        # < $0.01 left: an LLM-less attempt only burns wall the keep-best
        # would discard.  Prior cost unknown (None) so the predictive branch
        # cannot fire — this isolates the allowance floor.
        launch, _, why = cost_gate([_att(1, None)], 0.795, 0.80)
        self.assertFalse(launch)
        self.assertIn("$0.01", why)

    def test_exact_fit_launches(self):
        # 0.40 + 0.40 = 0.80 does not CROSS the ceiling (the ceiling itself
        # already carries the $0.20 margin under the $1.00 zero-cap).
        launch, allowance, _ = cost_gate([_att(1, 0.40)], 0.40, 0.80)
        self.assertTrue(launch)
        self.assertEqual(allowance, 0.40)

    def test_kill_switch_ceiling_zero_or_none(self):
        for ceiling in (0, 0.0, -1.0, None):
            launch, allowance, why = cost_gate([_att(1, 0.99)], 0.99, ceiling)
            self.assertTrue(launch)
            self.assertIsNone(allowance)       # no budget passed downstream
            self.assertEqual(why, "breaker_off")

    def test_unknown_costs_do_not_estimate(self):
        # Attempts that died before writing token_usage.json have cost None;
        # they contribute no estimate (allowance still bounds the next draw).
        atts = [_att(1, None), _att(2, 0.0)]
        launch, allowance, _ = cost_gate(atts, 0.0, 0.80)
        self.assertTrue(launch)
        self.assertEqual(allowance, 0.80)

    # ---- FIX 4 (S5, C1 review): first-attempt sub-cent clamp ----

    def test_first_attempt_subcent_ceiling_clamps_up_to_one_cent(self):
        # A sub-cent ceiling must yield ONE minimal $0.01 LLM-lean attempt,
        # not zero attempts (one lean draw beats shipping nothing).
        launch, allowance, why = cost_gate([], 0.0, 0.005)
        self.assertTrue(launch)
        self.assertEqual(allowance, 0.01)
        self.assertEqual(why, "")

    def test_first_attempt_subcent_remainder_clamps_up(self):
        launch, allowance, _ = cost_gate([], 0.796, 0.80)
        self.assertTrue(launch)
        self.assertEqual(allowance, 0.01)

    def test_subcent_refusal_still_applies_after_first_attempt(self):
        # With an incumbent banked, an LLM-less redraw only burns wall the
        # keep-best would discard — the strict floor stays for attempts 2+.
        launch, _, why = cost_gate([_att(1, None)], 0.796, 0.80)
        self.assertFalse(launch)
        self.assertIn("$0.01", why)

    def test_clamp_does_not_override_at_or_over_ceiling_refusal(self):
        # cost_so_far >= ceiling refuses regardless of attempt count.
        self.assertFalse(cost_gate([], 0.80, 0.80)[0])


class DefaultCeilingEnvTests(unittest.TestCase):
    def test_default_without_env(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FPL26_COST_CEILING", None)
            self.assertEqual(_default_cost_ceiling(), COST_CEILING_DEFAULT)

    def test_env_override(self):
        with mock.patch.dict(os.environ, {"FPL26_COST_CEILING": "0.70"}):
            self.assertEqual(_default_cost_ceiling(), 0.70)

    def test_env_kill_switch_zero(self):
        with mock.patch.dict(os.environ, {"FPL26_COST_CEILING": "0"}):
            self.assertEqual(_default_cost_ceiling(), 0.0)

    def test_bad_env_falls_back_never_crashes(self):
        with mock.patch.dict(os.environ, {"FPL26_COST_CEILING": "cheap"}):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(_default_cost_ceiling(), COST_CEILING_DEFAULT)


class AttemptCmdBudgetTests(unittest.TestCase):
    def test_budget_appended_as_make_var(self):
        cmd = _attempt_cmd(Path("in.dcp"), Path("/tmp/o.dcp"), 1800.0, False,
                           llm_cost_budget=0.52)
        self.assertIn("LLM_COST_BUDGET=0.52", cmd)

    def test_none_budget_omits_var(self):
        cmd = _attempt_cmd(Path("in.dcp"), Path("/tmp/o.dcp"), 1800.0, False,
                           llm_cost_budget=None)
        self.assertFalse(any(a.startswith("LLM_COST_BUDGET=") for a in cmd))


# Part B — run() integration (stubbed subprocess, TestIncrementalRefresh style)

class TestRunCumulativeCeiling:
    def _drive(self, tmp_path, monkeypatch, per_attempt_cost, fmaxes,
               max_attempts=4, **run_kwargs):
        inp = tmp_path / "bench.dcp"; inp.write_bytes(b"x" * 64)
        final = tmp_path / "bench_optimized.dcp"
        cmds = []
        calls = {"n": 0}

        def fake_attempt(cmd, repo=None, budget_s=None, attempt_i=None, **kw):
            calls["n"] += 1
            cmds.append(list(cmd))
            out = [a.split("=", 1)[1] for a in cmd if a.startswith("OUTPUT=")][0]
            Path(out).write_bytes(b"dcp-attempt-%d" % calls["n"])
            return 0

        # Give each attempt a fresh run directory, matching agent behavior.
        # Cost attribution deduplicates directories, so reuse would collapse
        # distinct per-attempt costs.
        def fake_attribute(base, before):
            d = tmp_path / f"dcp_optimizer_run-{calls['n']}"
            d.mkdir(exist_ok=True)
            return d

        fm = iter(fmaxes)
        monkeypatch.setattr(mro, "_run_attempt_process", fake_attempt)
        monkeypatch.setattr(mro, "_attribute_run_dir", fake_attribute)
        monkeypatch.setattr(mro, "_read_run_metrics",
                            lambda rd: {"fmax": next(fm),
                                        "status": "VALID_OPTIMIZED",
                                        "cost": per_attempt_cost})
        s = mro.run(inp, final, total_wall=100_000, attempt_floor=1.0,
                    max_attempts=max_attempts, repo=tmp_path, cost_cap=0.85,
                    polish=False, **run_kwargs)
        return s, cmds, final

    def test_stops_before_crossing_ceiling(self, tmp_path, monkeypatch):
        # 0.30/attempt, ceiling 0.80: a1 (cum .30), a2 (cum .60), a3 refused
        # (predicted .60+.30=.90 > .80).  Distinct fmaxes defeat early-stop.
        s, cmds, final = self._drive(tmp_path, monkeypatch, 0.30,
                                     [100.0, 50.0, 25.0, 10.0])
        assert len(cmds) == 2
        assert s["llm_cost_total"] == 0.60          # never crossed 0.80
        assert final.exists()                       # banked best shipped
        assert final.read_bytes() == b"dcp-attempt-1"

    def test_15_shape_single_expensive_attempt(self, tmp_path, monkeypatch):
        s, cmds, final = self._drive(tmp_path, monkeypatch, 0.55,
                                     [100.0, 50.0])
        assert len(cmds) == 1                       # attempt 2 not launched
        assert final.exists()                       # exits WITH banked output
        assert s["chosen"] is not None
        assert s["llm_cost_total"] == 0.55

    def test_shrinking_llm_budget_passed_down(self, tmp_path, monkeypatch):
        _, cmds, _ = self._drive(tmp_path, monkeypatch, 0.30,
                                 [100.0, 50.0, 25.0])
        def budget(cmd):
            return [a for a in cmd if a.startswith("LLM_COST_BUDGET=")]
        assert budget(cmds[0]) == ["LLM_COST_BUDGET=0.80"]   # ceiling − 0
        assert budget(cmds[1]) == ["LLM_COST_BUDGET=0.50"]   # ceiling − 0.30

    def test_breaker_off_omits_budget_and_does_not_gate(self, tmp_path,
                                                        monkeypatch):
        s, cmds, _ = self._drive(tmp_path, monkeypatch, 0.45,
                                 [100.0, 50.0], max_attempts=2,
                                 cost_ceiling=0)
        assert len(cmds) == 2                       # cum 0.90 > 0.80 allowed
        assert all(not a.startswith("LLM_COST_BUDGET=")
                   for cmd in cmds for a in cmd)


class TestRunEstimateCharging:
    """Verify conservative charging when an attempt has no readable cost record.

    A budgeted attempt uses the highest known prior cost or, if none exists,
    $0.35 as the configured prior-band ceiling. Each run directory is charged
    at most once.
    """

    def _drive(self, tmp_path, monkeypatch, costs, fmaxes, rd_present=True,
               same_dir=False, max_attempts=4, **run_kwargs):
        inp = tmp_path / "bench.dcp"; inp.write_bytes(b"x" * 64)
        final = tmp_path / "bench_optimized.dcp"
        calls = {"n": 0}

        def fake_attempt(cmd, repo=None, budget_s=None, attempt_i=None, **kw):
            calls["n"] += 1
            out = [a.split("=", 1)[1] for a in cmd if a.startswith("OUTPUT=")][0]
            Path(out).write_bytes(b"dcp-attempt-%d" % calls["n"])
            return 0

        def fake_attribute(base, before):
            if not rd_present:
                return None
            n = 1 if same_dir else calls["n"]
            d = tmp_path / f"dcp_optimizer_run-{n}"
            d.mkdir(exist_ok=True)
            return d

        cs, fs = iter(costs), iter(fmaxes)
        monkeypatch.setattr(mro, "_run_attempt_process", fake_attempt)
        monkeypatch.setattr(mro, "_attribute_run_dir", fake_attribute)
        monkeypatch.setattr(mro, "_read_run_metrics",
                            lambda rd: {"fmax": next(fs),
                                        "status": "VALID_OPTIMIZED",
                                        "elapsed": None, "cost": next(cs)})
        return mro.run(inp, final, total_wall=100_000, attempt_floor=1.0,
                       max_attempts=max_attempts, repo=tmp_path,
                       cost_cap=0.85, polish=False, **run_kwargs)

    def test_no_priors_charges_fixed_prior(self, tmp_path, monkeypatch):
        s = self._drive(tmp_path, monkeypatch, [None], [100.0],
                        max_attempts=1)
        assert s["llm_cost_total"] == mro.COST_ESTIMATE_FALLBACK_USD
        assert s["attempts"][0]["cost"] == mro.COST_ESTIMATE_FALLBACK_USD
        assert s["attempts"][0]["cost_estimated"] is True

    def test_priors_charge_max_prior_cost(self, tmp_path, monkeypatch):
        # Attempt 1 measured $0.20; attempt 2 crashes without a record ->
        # charged the max prior ($0.20), not $0 and not the fixed prior.
        s = self._drive(tmp_path, monkeypatch, [0.20, None], [100.0, 50.0],
                        max_attempts=2)
        assert s["llm_cost_total"] == 0.40
        assert s["attempts"][1]["cost"] == 0.20
        assert s["attempts"][1]["cost_estimated"] is True

    def test_no_new_run_dir_still_charges_estimate(self, tmp_path,
                                                   monkeypatch):
        # FIX 3: attribution None (attempt crashed pre-run-dir) must not
        # reuse an old dir NOR skip the charge.
        s = self._drive(tmp_path, monkeypatch, [None], [100.0],
                        rd_present=False, max_attempts=1)
        assert s["llm_cost_total"] == mro.COST_ESTIMATE_FALLBACK_USD
        assert s["attempts"][0]["run_dir"] is None

    def test_breaker_off_still_charges_the_estimate(self, tmp_path,
                                                     monkeypatch):
        # The ceiling kill switch (--cost-ceiling 0) does NOT mean the
        # attempt spent nothing: cost_gate's "breaker_off" path hands down no
        # budget precisely so the agent keeps its own $0.75 default exit.
        # Charging $0 for an unreadable record therefore left cost_so_far
        # frozen -- and cost_so_far is also the meter the INDEPENDENT
        # $1.00/benchmark cost_cap guard reads, so disabling the ceiling used
        # to blind both spend guards at once.
        s = self._drive(tmp_path, monkeypatch, [None, None], [100.0, 50.0],
                        max_attempts=2, cost_ceiling=0)
        assert s["llm_cost_total"] == 2 * mro.COST_ESTIMATE_FALLBACK_USD
        assert s["attempts"][0]["cost_estimated"] is True

    def test_breaker_off_cost_cap_still_stops_the_loop(self, tmp_path,
                                                       monkeypatch):
        # With the ceiling off and no readable records, the retrospective
        # cap must still fire: 3 x $0.35 = $1.05 >= the $0.85 cap passed by
        # _drive, so the 4th attempt never launches.
        s = self._drive(tmp_path, monkeypatch, [None] * 4,
                        [100.0, 50.0, 50.0, 50.0],
                        max_attempts=4, cost_ceiling=0)
        assert len(s["attempts"]) == 3
        assert s["llm_cost_total"] == round(
            3 * mro.COST_ESTIMATE_FALLBACK_USD, 4)

    def test_zero_ledger_charged_zero_not_estimate(self, tmp_path,
                                                   monkeypatch):
        # FIX 1a's $0 seed read back: a benign no-LLM crash costs $0.
        s = self._drive(tmp_path, monkeypatch, [0.0], [100.0],
                        max_attempts=1)
        assert s["llm_cost_total"] == 0.0
        assert s["attempts"][0]["cost_estimated"] is False

    def test_same_run_dir_never_charged_twice(self, tmp_path, monkeypatch):
        # FIX 3 guard: if attribution ever yields the same dir again, its
        # cost must not be double-counted.
        s = self._drive(tmp_path, monkeypatch, [0.30, 0.30], [100.0, 50.0],
                        same_dir=True, max_attempts=2)
        assert s["llm_cost_total"] == 0.30

    def test_estimate_feeds_predictive_gate(self, tmp_path, monkeypatch):
        # Estimated charges accumulate toward the ceiling exactly like
        # measured ones: 0.35-estimates stop the loop before crossing 0.80
        # (0.35 + 0.35 = 0.70; a third would predict 1.05 > 0.80).
        s = self._drive(tmp_path, monkeypatch, [None, None, None, None],
                        [100.0, 50.0, 25.0, 10.0])
        assert len(s["attempts"]) == 2
        assert s["llm_cost_total"] == 0.70


# Part C — agent side (dcp_optimizer): effective exit + pre-call gate +
# deterministic finalize-with-banked on breach in BOTH controller modes.

class ResolveLlmCostExitTests(unittest.TestCase):
    def test_unset_keeps_default(self):
        self.assertEqual(resolve_llm_cost_exit(None), LLM_COST_EXIT_USD)

    def test_budget_tightens(self):
        self.assertEqual(resolve_llm_cost_exit(0.50), 0.50)

    def test_budget_never_raises_above_default(self):
        self.assertEqual(resolve_llm_cost_exit(2.0), LLM_COST_EXIT_USD)

    def test_zero_negative_or_garbage_keep_default(self):
        for bad in (0, 0.0, -1, "cheap"):
            self.assertEqual(resolve_llm_cost_exit(bad), LLM_COST_EXIT_USD)


def _make_opt(tmp_path: Path, mode: str = "v0_3") -> DCPOptimizer:
    opt = DCPOptimizer(api_key="test", run_dir=tmp_path, mode=mode)
    opt.rag_seed = False
    opt.contest_mode = True
    opt.max_wall_seconds = None
    return opt


class PreCallGateTests(unittest.TestCase):
    """The gate must refuse to ISSUE another LLM call once breached — this is
    the choke point every LLM path shares (outer loop iterations AND the
    process_message tool-round recursion that used to chain calls unchecked
    between iterations)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_breached_returns_without_api_call(self):
        opt = _make_opt(self.tmp_path)
        opt.total_cost = 0.80
        opt.llm_cost_exit_usd = 0.75
        with mock.patch.object(
                opt, "_create_completion_with_fallback",
                side_effect=AssertionError("API must not be called")):
            text, is_done = _async(opt.get_completion())
        self.assertIn("[cost-exit]", text)
        self.assertFalse(is_done)
        self.assertEqual(opt.llm_call_count, 0)

    def test_under_threshold_still_calls_api(self):
        opt = _make_opt(self.tmp_path)
        opt.total_cost = 0.10
        sentinel = RuntimeError("api-was-called")
        with mock.patch.object(opt, "_create_completion_with_fallback",
                               side_effect=sentinel):
            with self.assertRaises(RuntimeError):
                _async(opt.get_completion())
        self.assertEqual(opt.llm_call_count, 1)

    def test_tightened_budget_gates_earlier(self):
        # Wrapper passed ceiling − prior spend = $0.20; $0.25 already spent
        # in-attempt must gate even though it is far below the $0.75 default.
        opt = _make_opt(self.tmp_path)
        opt.llm_cost_exit_usd = resolve_llm_cost_exit(0.20)
        opt.total_cost = 0.25
        with mock.patch.object(
                opt, "_create_completion_with_fallback",
                side_effect=AssertionError("API must not be called")):
            text, is_done = _async(opt.get_completion())
        self.assertIn("[cost-exit]", text)

    def test_breached_predicate_disabled_at_nonpositive_threshold(self):
        opt = _make_opt(self.tmp_path)
        opt.total_cost = 99.0
        opt.llm_cost_exit_usd = 0.0
        self.assertFalse(opt._llm_cost_breached())


class LoopBreachFinalizeTests(unittest.TestCase):
    """In-attempt breach -> deterministic finalize-with-banked in BOTH
    controller modes (anchor previously had NO cost exit at all)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _drive(self, opt: DCPOptimizer, scripted) -> bool:
        input_dcp = self.tmp_path / "baseline.dcp"
        input_dcp.write_bytes(b"BASELINE_DCP_BYTES")
        output_dcp = self.tmp_path / "optimized.dcp"

        async def fake_analysis(_input_dcp):
            opt.initial_wns = -2.0
            opt.clock_period = 4.0
            return "ANALYSIS: initial WNS -2.000 ns"

        with mock.patch("dcp_optimizer.load_system_prompt",
                        return_value="SYS PROMPT (cost-breaker test)"), \
             mock.patch.object(opt, "perform_initial_analysis",
                               side_effect=fake_analysis), \
             mock.patch.object(opt, "get_completion",
                               new=mock.AsyncMock(side_effect=scripted)), \
             mock.patch.object(opt, "call_tool",
                               new=mock.AsyncMock(return_value="ok")), \
             mock.patch.object(opt, "_persist_to_strategy_memory"), \
             redirect_stdout(io.StringIO()):
            return _async(opt.optimize(input_dcp, output_dcp))

    def test_v03_breach_exits_via_ils_polish_tail(self):
        # The #15 spend shape inside one attempt: iteration 1 lands with
        # $0.90 accrued -> the loop must hand over to the zero-cost tail
        # (ILS polish + finalize with the banked best), not keep iterating.
        opt = _make_opt(self.tmp_path, mode="v0_3")

        async def scripted():
            opt.total_cost = 0.90
            return ("optimizing...", False)

        exit_tail = mock.AsyncMock()
        with mock.patch.object(opt, "_exit_with_ils_polish", new=exit_tail):
            result = self._drive(opt, scripted)
        self.assertTrue(result)
        self.assertEqual(exit_tail.await_count, 1)
        self.assertEqual(opt.iteration, 1)

    def test_anchor_breach_finalizes_with_banked(self):
        # Gap closed: anchor mode previously ran to is_done or
        # max_iterations with NO cost exit.
        opt = _make_opt(self.tmp_path, mode="anchor")

        async def scripted():
            opt.total_cost = 0.90
            return ("optimizing...", False)

        finalize = mock.AsyncMock()
        with mock.patch.object(opt, "_finalize_output_dcp", new=finalize):
            result = self._drive(opt, scripted)
        self.assertTrue(result)
        self.assertEqual(finalize.await_count, 1)
        self.assertEqual(opt.iteration, 1)

    def test_no_breach_no_cost_exit(self):
        # Control: under-threshold spend must not trigger the breaker —
        # the LLM's own done signal ends the run (unchanged behavior).
        opt = _make_opt(self.tmp_path, mode="anchor")

        async def scripted():
            opt.total_cost = 0.10
            return ("optimization complete", True)

        finalize = mock.AsyncMock()
        with mock.patch.object(opt, "_finalize_output_dcp", new=finalize):
            result = self._drive(opt, scripted)
        self.assertTrue(result)
        self.assertEqual(finalize.await_count, 1)   # via is_done, not breach


if __name__ == "__main__":
    unittest.main()
