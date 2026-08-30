"""C1-phase adversarial-review fixes — verified diagnoses, pinned.

Covers (see the review's exact fix list):
  FIX 1a (C1/S7 BLOCKER) — incremental crash-safe cost_ledger.json written
      by the agent after EVERY API-call cost accrual (+ $0 seed at
      construction); atomic tmp+rename; a write failure must never crash
      the run.
  FIX 1b — the wrapper falls back token_usage.json -> cost_ledger.json;
      when BOTH are unreadable an LLM-budgeted attempt is charged the
      predictive estimate (max prior known cost, else the $0.35 band-top
      prior), never a silent $0.  (run()-level charging tests live in
      tests/test_cost_breaker.py next to the ceiling suite.)
  FIX 2 (S2) — the budget-aware last-resort single run moved INSIDE the
      wrapper (LLM_COST_BUDGET = max($0.01, ceiling − spent)); the
      Makefile `||` stays as a pre-Python safety net with a fixed $0.10.
  FIX 3 (S6/S8) — snapshot-diff run-dir attribution in the agent's real
      base dir (FPL26_RUN_DIR_BASE honored); no new dir -> None, never
      reuse a pre-existing dir.
  FIX 5 (S3, observability only) — distinct
      loop_exit_reason="polish_reserve_fence" when the fence-capped
      timeout (not genuine budget exhaustion) caused the budget kill.

Stub-driven per tests/test_budget_enforcement.py — no network, no Vivado.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import DCPOptimizer
from optimizer.ils_polish import ILSPolishConfig
import scripts.multi_restart_optimize as mro


def _async(coro):
    return asyncio.run(coro)


def _make_opt(tmp_path: Path, mode: str = "v0_3") -> DCPOptimizer:
    opt = DCPOptimizer(api_key="test", run_dir=tmp_path, mode=mode)
    opt.rag_seed = False
    opt.contest_mode = True
    opt.max_wall_seconds = None
    return opt


def _ledger(run_dir: Path) -> dict:
    return json.loads((run_dir / "cost_ledger.json").read_text())


# FIX 1a — incremental cost ledger (agent side)

class CostLedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_init_seeds_zero_ledger(self):
        # A benign crash BEFORE any LLM call must leave an explicit $0
        # record (wrapper charges $0), not NO record (wrapper would charge
        # the predictive estimate).
        _make_opt(self.tmp_path)
        led = _ledger(self.tmp_path)
        self.assertEqual(led["total_cost"], 0.0)
        self.assertEqual(led["calls"], 0)

    def test_ledger_reflects_accrued_spend(self):
        opt = _make_opt(self.tmp_path)
        opt.total_cost = 0.1234
        opt.llm_call_count = 3
        opt._write_cost_ledger()
        led = _ledger(self.tmp_path)
        self.assertEqual(led["total_cost"], 0.1234)
        self.assertEqual(led["calls"], 3)

    def test_get_completion_writes_ledger_after_accrual(self):
        # The single accrual point (response.usage.cost) must be followed
        # by a ledger rewrite on EVERY call — this is what survives a
        # crash that never reaches _print_optimization_summary.
        opt = _make_opt(self.tmp_path)
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                                total_tokens=15, cost=0.02)
        resp = SimpleNamespace(usage=usage, error=None)
        with mock.patch.object(opt, "_create_completion_with_fallback",
                               return_value=resp), \
             mock.patch.object(opt, "process_response",
                               new=mock.AsyncMock(return_value=("ok", False))), \
             redirect_stdout(io.StringIO()):
            _async(opt.get_completion())
        led = _ledger(self.tmp_path)
        self.assertEqual(led["total_cost"], 0.02)
        self.assertEqual(led["calls"], 1)

    def test_write_failure_never_raises_and_keeps_prior_ledger(self):
        # Atomicity discipline: a failing rename must neither crash the
        # run nor corrupt the previous ledger (tmp+os.replace, like
        # _atomic_copy) nor strand tmp files.
        opt = _make_opt(self.tmp_path)
        opt.total_cost = 0.50
        opt.llm_call_count = 7
        with mock.patch("dcp_optimizer.os.replace",
                        side_effect=OSError("disk full")):
            opt._write_cost_ledger()   # must not raise
        led = _ledger(self.tmp_path)   # $0 seed from __init__ survives
        self.assertEqual(led["total_cost"], 0.0)
        self.assertEqual(list(self.tmp_path.glob("cost_ledger.*.tmp")), [])

    def test_missing_run_dir_never_raises(self):
        opt = _make_opt(self.tmp_path)
        opt.run_dir = self.tmp_path / "gone" / "deeper"
        opt._write_cost_ledger()       # must not raise


# FIX 1b — wrapper-side metric read: token_usage -> cost_ledger fallback

class ReadRunMetricsFallbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rd = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_ledger(self, cost=0.42, calls=6):
        (self.rd / "cost_ledger.json").write_text(
            json.dumps({"total_cost": cost, "calls": calls}))

    def test_ledger_used_when_token_usage_missing(self):
        # The crash shape: run died mid-flight, token_usage.json never
        # written, incremental ledger present.
        self._write_ledger(0.42)
        self.assertEqual(mro._read_run_metrics(self.rd)["cost"], 0.42)

    def test_ledger_used_when_token_usage_unparseable(self):
        (self.rd / "token_usage.json").write_text("{truncated-by-kill")
        self._write_ledger(0.31)
        self.assertEqual(mro._read_run_metrics(self.rd)["cost"], 0.31)

    def test_token_usage_wins_when_readable(self):
        (self.rd / "token_usage.json").write_text(
            json.dumps({"summary": {"total_cost": 0.50,
                                    "best_fmax_mhz": 100.0}}))
        self._write_ledger(0.20)       # stale mid-run snapshot
        m = mro._read_run_metrics(self.rd)
        self.assertEqual(m["cost"], 0.50)
        self.assertEqual(m["fmax"], 100.0)

    def test_zero_ledger_reads_zero_not_none(self):
        # FIX 1a's $0 seed: a no-LLM crash is charged $0, NOT the estimate.
        self._write_ledger(0.0, calls=0)
        self.assertEqual(mro._read_run_metrics(self.rd)["cost"], 0.0)

    def test_both_unreadable_cost_stays_none(self):
        (self.rd / "token_usage.json").write_text("not json")
        (self.rd / "cost_ledger.json").write_text("also not json")
        self.assertIsNone(mro._read_run_metrics(self.rd)["cost"])

    def test_nothing_readable_cost_none(self):
        self.assertIsNone(mro._read_run_metrics(self.rd)["cost"])


# FIX 3 — snapshot-diff run-dir attribution

class RunDirAttributionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _mkrun(self, name, mtime=None):
        d = self.base / name
        d.mkdir()
        if mtime is not None:
            os.utime(d, (mtime, mtime))
        return d

    def test_no_new_dir_returns_none_never_the_old_one(self):
        # The S8 bug: a crashed attempt used to re-attribute (and
        # double-count) the newest PRE-EXISTING dir.
        self._mkrun("dcp_optimizer_run-old")
        before = mro._snapshot_run_dirs(self.base)
        self.assertIsNone(mro._attribute_run_dir(self.base, before))

    def test_new_dir_is_attributed(self):
        self._mkrun("dcp_optimizer_run-old")
        before = mro._snapshot_run_dirs(self.base)
        new = self._mkrun("dcp_optimizer_run-new")
        self.assertEqual(mro._attribute_run_dir(self.base, before), new)

    def test_foreign_concurrent_dirs_in_snapshot_are_ignored(self):
        # A concurrent wrapper's dir that existed at snapshot time must
        # never be attributed to THIS attempt.
        self._mkrun("dcp_optimizer_run-foreign")
        before = mro._snapshot_run_dirs(self.base)
        self.assertIsNone(mro._attribute_run_dir(self.base, before))

    def test_newest_of_multiple_new_dirs_wins(self):
        before = mro._snapshot_run_dirs(self.base)
        now = time.time()
        self._mkrun("dcp_optimizer_run-a", mtime=now - 100)
        newest = self._mkrun("dcp_optimizer_run-b", mtime=now)
        self.assertEqual(mro._attribute_run_dir(self.base, before), newest)

    def test_empty_base_returns_none(self):
        self.assertIsNone(mro._attribute_run_dir(self.base, set()))

    def test_base_honors_fpl26_run_dir_base(self):
        # Pre-existing bug pinned: dcp_optimizer._run_dir_base honors the
        # env var; the wrapper's glob did not — attribution silently missed
        # every run dir on local ops boxes.
        target = self.base / "artifacts"
        with mock.patch.dict(os.environ,
                             {"FPL26_RUN_DIR_BASE": str(target)}):
            self.assertEqual(mro._wrapper_run_dir_base(Path("/some/repo")),
                             target)
        self.assertTrue(target.is_dir())   # created like the agent does

    def test_base_defaults_to_repo_without_env(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FPL26_RUN_DIR_BASE", None)
            repo = self.base / "repo"
            self.assertEqual(mro._wrapper_run_dir_base(repo), repo)


# FIX 2 — wrapper-internal budgeted fallback

class InternalFallbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.inp = self.tmp_path / "bench.dcp"
        self.inp.write_bytes(b"x")
        self.final = self.tmp_path / "bench_optimized.dcp"

    def tearDown(self):
        self.tmp.cleanup()

    def _fake_subprocess(self, create_output=True):
        calls = []

        def fake_run(cmd, cwd=None, **kw):
            calls.append(list(cmd))
            if create_output:
                out = [a.split("=", 1)[1] for a in cmd
                       if a.startswith("OUTPUT=")][0]
                Path(out).write_bytes(b"fallback-dcp")

            class R:
                returncode = 0
            return R()

        return calls, fake_run

    def _fb(self, cost_ceiling, cost_so_far, elapsed_s=0.0,
            total_wall=3500.0, create_output=True):
        calls, fake_run = self._fake_subprocess(create_output)
        with mock.patch.object(mro.subprocess, "run", fake_run), \
             redirect_stdout(io.StringIO()):
            rc = mro._run_internal_fallback(
                self.inp, self.final, total_wall, False, cost_ceiling,
                cost_so_far, elapsed_s, self.tmp_path)
        return rc, calls

    def test_remaining_budget_passed_down(self):
        # ceiling 0.80, spent 0.55 -> the final attempt gets exactly the
        # remainder, NOT a fresh unbudgeted default $0.75 (the S2 hole).
        rc, calls = self._fb(0.80, 0.55)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)
        self.assertIn("LLM_COST_BUDGET=0.25", calls[0])
        self.assertTrue(self.final.exists())

    def test_subcent_remainder_clamps_to_minimal_budget(self):
        # An LLM-lean attempt beats nothing: sub-cent remainder -> $0.01.
        rc, calls = self._fb(0.80, 0.799)
        self.assertIn("LLM_COST_BUDGET=0.01", calls[0])
        self.assertEqual(rc, 0)

    def test_overspent_still_gets_minimal_budget(self):
        rc, calls = self._fb(0.80, 0.95)
        self.assertIn("LLM_COST_BUDGET=0.01", calls[0])

    def test_breaker_off_passes_no_budget(self):
        for ceiling in (0, 0.0, None):
            rc, calls = self._fb(ceiling, 0.55)
            self.assertFalse(any(a.startswith("LLM_COST_BUDGET=")
                                 for a in calls[0]))

    def test_wall_slice_is_total_minus_elapsed(self):
        _, calls = self._fb(0.80, 0.10, elapsed_s=3000.0, total_wall=3500.0)
        self.assertIn("MAX_WALL=500", calls[0])

    def test_exhausted_wall_floors_at_60s(self):
        _, calls = self._fb(0.80, 0.10, elapsed_s=3499.0, total_wall=3500.0)
        self.assertIn("MAX_WALL=60", calls[0])

    def test_nonzero_exit_only_when_no_output(self):
        rc, _ = self._fb(0.80, 0.55, create_output=False)
        self.assertEqual(rc, 1)
        self.assertFalse(self.final.exists())

    def test_launch_failure_never_raises(self):
        def boom(cmd, cwd=None, **kw):
            raise OSError("make not found")
        with mock.patch.object(mro.subprocess, "run", boom), \
             redirect_stdout(io.StringIO()):
            rc = mro._run_internal_fallback(
                self.inp, self.final, 3500.0, False, 0.80, 0.55, 0.0,
                self.tmp_path)
        self.assertEqual(rc, 1)


class MainRoutesIntoInternalFallbackTests(unittest.TestCase):
    def test_chosen_none_calls_fallback_with_known_spend(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            inp = tmp_path / "b.dcp"
            inp.write_bytes(b"x")
            captured = {}

            def fake_fb(input_dcp, final_output, total_wall, ils_polish,
                        cost_ceiling, cost_so_far, elapsed_s, repo):
                captured.update(cost_so_far=cost_so_far,
                                elapsed_s=elapsed_s,
                                cost_ceiling=cost_ceiling)
                return 0

            with mock.patch.object(mro, "run", return_value={
                        "chosen": None, "llm_cost_total": 0.55,
                        "wall_elapsed_s": 3000.0}), \
                 mock.patch.object(mro, "_install_signal_publisher",
                                   lambda: None), \
                 mock.patch.object(mro, "_run_internal_fallback", fake_fb):
                rc = mro.main([str(inp), "--cost-ceiling", "0.80"])
            self.assertEqual(rc, 0)
            self.assertEqual(captured["cost_so_far"], 0.55)
            self.assertEqual(captured["elapsed_s"], 3000.0)
            self.assertEqual(captured["cost_ceiling"], 0.80)

    def test_chosen_present_skips_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            inp = tmp_path / "b.dcp"
            inp.write_bytes(b"x")
            with mock.patch.object(mro, "run", return_value={
                        "chosen": {"i": 1}, "llm_cost_total": 0.2,
                        "wall_elapsed_s": 100.0}), \
                 mock.patch.object(mro, "_install_signal_publisher",
                                   lambda: None), \
                 mock.patch.object(
                     mro, "_run_internal_fallback",
                     side_effect=AssertionError("must not fire")):
                self.assertEqual(mro.main([str(inp)]), 0)


# FIX 5 — polish-reserve fence exit-reason label (observability only)

class FenceCapFlagTests(unittest.TestCase):
    """_deadline_aware_timeout must remember whether it fence-capped the
    window, so the TimeoutError handler can label the kill cause."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.opt = _make_opt(self.tmp_path)
        # Canonical ARMED state (mirrors tests/test_polish_reserve.py):
        # contest budget + routed banked best + polish stages enabled.
        self.opt.max_wall_seconds = 3300.0
        self.opt._budget_deadline = time.time() + 2000.0
        self.opt._polish_reserve_s = 500.0
        banked = self.tmp_path / "best_valid.dcp"
        banked.write_bytes(b"dcp")
        self.opt._best_valid_dcp = banked
        self.opt._ils_polish_cfg = ILSPolishConfig(enabled=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fence_capped_sets_flag_and_reduces_window(self):
        t = self.opt._deadline_aware_timeout("vivado_place_design", {})
        self.assertTrue(self.opt._last_timeout_fence_capped)
        self.assertLess(t, 1600.0)           # ~2000 − 500 reserve
        self.assertGreater(t, 1400.0)

    def test_non_destroying_call_leaves_flag_clear(self):
        # First set it via a fence-capped computation, then confirm the
        # next (non-destroying) computation RESETS it — the handler must
        # only ever see the flag of its own call.
        self.opt._deadline_aware_timeout("vivado_place_design", {})
        t = self.opt._deadline_aware_timeout(
            "vivado_report_timing_summary", {})
        self.assertFalse(self.opt._last_timeout_fence_capped)
        self.assertGreater(t, 1900.0)        # full window

    def test_reserve_released_leaves_flag_clear(self):
        self.opt._release_polish_reserve("test")
        self.opt._deadline_aware_timeout("vivado_place_design", {})
        self.assertFalse(self.opt._last_timeout_fence_capped)


class _FakeContent:
    def __init__(self, text):
        self.text = text


class _FakeResult:
    def __init__(self, text="{}"):
        self.content = [_FakeContent(text)]


class FenceTimeoutCauseTests(unittest.TestCase):
    """A fence-capped asyncio timeout labels _budget_kill_cause; a plain
    budget timeout does not (behavior identical either way)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.opt = _make_opt(self.tmp_path)
        self.opt.max_wall_seconds = 3300.0
        self.opt._budget_deadline = time.time() + 1000.0

        async def slow(*a, **kw):  # pragma: no cover — always cancelled
            await asyncio.sleep(60)

        self.opt.vivado_session = SimpleNamespace(
            call_tool=mock.AsyncMock(side_effect=slow))

    def tearDown(self):
        self.tmp.cleanup()

    def _timeout_call(self, fence_capped: bool):
        def fake_timeout(tool_name=None, arguments=None):
            self.opt._last_timeout_fence_capped = fence_capped
            return 0.01

        with mock.patch.object(self.opt, "_should_skip_for_budget",
                               return_value=(False, "")), \
             mock.patch.object(self.opt, "_deadline_aware_timeout",
                               side_effect=fake_timeout):
            result = _async(
                self.opt.call_tool("vivado_phys_opt_design", {}))
        return json.loads(result)

    def test_fence_capped_timeout_labels_cause(self):
        payload = self._timeout_call(fence_capped=True)
        self.assertEqual(payload["error"], "tool_timed_out_budget")
        self.assertTrue(self.opt._budget_killed)
        self.assertEqual(self.opt._budget_kill_cause, "polish_reserve_fence")

    def test_plain_budget_timeout_keeps_cause_none(self):
        payload = self._timeout_call(fence_capped=False)
        self.assertEqual(payload["error"], "tool_timed_out_budget")
        self.assertTrue(self.opt._budget_killed)
        self.assertIsNone(self.opt._budget_kill_cause)

    def test_first_cause_wins(self):
        self.opt._budget_kill_cause = "polish_reserve_fence"
        self._timeout_call(fence_capped=False)
        self.assertEqual(self.opt._budget_kill_cause, "polish_reserve_fence")


class FenceLoopExitReasonTests(unittest.TestCase):
    """The iteration loop maps _budget_kill_cause into a DISTINCT
    loop_exit_reason (forensics only — same finalize tail either way)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _drive(self, cause):
        opt = _make_opt(self.tmp_path, mode="v0_3")
        input_dcp = self.tmp_path / "baseline.dcp"
        input_dcp.write_bytes(b"BASELINE_DCP_BYTES")
        output_dcp = self.tmp_path / "optimized.dcp"

        async def fake_analysis(_input_dcp):
            opt.initial_wns = -2.0
            opt.clock_period = 4.0
            return "ANALYSIS: initial WNS -2.000 ns"

        async def scripted():
            # iteration 1 "runs a tool" that gets budget-killed; the loop
            # detects the flag at the top of iteration 2.
            opt._budget_killed = True
            opt._budget_kill_cause = cause
            return ("optimizing...", False)

        exit_tail = mock.AsyncMock()
        with mock.patch("dcp_optimizer.load_system_prompt",
                        return_value="SYS PROMPT (fence test)"), \
             mock.patch.object(opt, "perform_initial_analysis",
                               side_effect=fake_analysis), \
             mock.patch.object(opt, "get_completion",
                               new=mock.AsyncMock(side_effect=scripted)), \
             mock.patch.object(opt, "call_tool",
                               new=mock.AsyncMock(return_value="ok")), \
             mock.patch.object(opt, "_persist_to_strategy_memory"), \
             mock.patch.object(opt, "_exit_with_ils_polish", new=exit_tail), \
             redirect_stdout(io.StringIO()), \
             self.assertLogs("dcp_optimizer", level="INFO") as logs:
            _async(opt.optimize(input_dcp, output_dcp))
        self.assertEqual(exit_tail.await_count, 1)   # same tail either way
        return "\n".join(logs.output)

    def test_fence_cause_gets_distinct_exit_reason(self):
        out = self._drive("polish_reserve_fence")
        self.assertIn("exited via polish_reserve_fence", out)

    def test_no_cause_keeps_budget_killed_reason(self):
        out = self._drive(None)
        self.assertIn("exited via budget_killed", out)


if __name__ == "__main__":
    unittest.main()
