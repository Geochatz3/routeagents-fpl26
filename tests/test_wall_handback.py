"""04-01 Task 2 (D3 wall handback) tests — R-D3, threat T-04-02.

The D3 mechanism composes the three EXISTING saturation signals (ILS
no-improve stop, LASTMILE reject verdict, _should_skip_for_budget
budget-kill) into a single agent-side early-exit reason
(_exit_early_reason), gated by:
  * a --wall-handback kill switch (DEFAULT OFF -> byte-identical behavior),
  * a never-trim-before-banked-accept guard (_best_valid_dcp must be set:
    "a trimmed zero is worse than a slow zero"),
and per the locked OQ1 resolution the LASTMILE/fanout polish stages STILL
RUN ONCE and must NOT consume/reset the signal.

No Vivado/RapidWright/MCP servers are ever spawned — sessions are mocked
(test_auto_bank.py pattern) and the ILS inner loop (run_ils_polish) is
monkeypatched where the composition test needs a synthetic verdict.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dcp_optimizer as dcp_mod
from dcp_optimizer import DCPOptimizer
from tests.source_corpus import dcp_source_lines, dcp_source_text


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


class ArmGuardTests(unittest.TestCase):
    """Never-trim-before-banked-accept guard on _maybe_arm_wall_handback."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = _make_optimizer(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_signal_without_banked_accept_not_armed(self):
        # A trimmed zero is worse than a slow zero: no accept banked ->
        # the reason must stay None even though a saturation signal fired.
        self.assertIsNone(self.opt._best_valid_dcp)
        self.opt._maybe_arm_wall_handback("ils_no_improve_stop(seed=raw)")
        self.assertIsNone(self.opt._exit_early_reason)

    def test_signal_with_banked_accept_armed(self):
        banked = Path(self.tmp.name) / "best_valid.dcp"
        banked.write_bytes(b"dcp")
        self.opt._best_valid_dcp = banked
        self.opt._maybe_arm_wall_handback("lastmile_reject:unrouted=-1")
        self.assertEqual(self.opt._exit_early_reason,
                         "lastmile_reject:unrouted=-1")

    def test_first_reason_wins(self):
        self.opt._best_valid_dcp = Path(self.tmp.name) / "b.dcp"
        self.opt._maybe_arm_wall_handback("first")
        self.opt._maybe_arm_wall_handback("second")
        self.assertEqual(self.opt._exit_early_reason, "first")


class DefaultOffTests(unittest.TestCase):
    """Kill switch DEFAULT OFF: the reason is never acted on (T-04-02)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = _make_optimizer(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_defaults(self):
        self.assertFalse(self.opt._wall_handback_enabled)
        self.assertIsNone(self.opt._exit_early_reason)
        self.assertFalse(self.opt._wall_handback_break_due())

    def test_reason_never_acted_on_when_disabled(self):
        # Parity: even a validly-armed reason must not break the loop
        # while the kill switch is OFF.
        self.opt._best_valid_dcp = Path(self.tmp.name) / "b.dcp"
        self.opt._maybe_arm_wall_handback("ils_no_improve_stop(seed=recipe_best)")
        self.assertIsNotNone(self.opt._exit_early_reason)
        self.assertFalse(self.opt._wall_handback_break_due())

    def test_break_due_when_enabled_and_armed(self):
        self.opt._wall_handback_enabled = True
        self.opt._best_valid_dcp = Path(self.tmp.name) / "b.dcp"
        self.opt._maybe_arm_wall_handback("budget_skip:deadline_passed")
        self.assertTrue(self.opt._wall_handback_break_due())

    def test_enabled_without_signal_does_not_break(self):
        self.opt._wall_handback_enabled = True
        self.assertFalse(self.opt._wall_handback_break_due())

    def test_default_off_never_acts_on_real_signal_site(self):
        # When wall handback is disabled, a sanctioned budget-expiry signal does
        # not break the loop, even when a banked checkpoint satisfies its guard.
        self.assertFalse(self.opt._wall_handback_enabled)
        self.opt._best_valid_dcp = Path(self.tmp.name) / "b.dcp"
        self.opt.max_wall_seconds = 100.0
        self.opt._budget_deadline = time.time() - 10.0
        _async(self.opt.call_tool("vivado_run_tcl", {"command": "puts hi"}))
        self.assertIsNotNone(self.opt._exit_early_reason)  # observed...
        self.assertFalse(self.opt._wall_handback_break_due())  # ...never acted


class BudgetSkipSignalTests(unittest.TestCase):
    """Signal (c): _should_skip_for_budget budget-kill arms the reason."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = _make_optimizer(Path(self.tmp.name))
        # Force the "deadline passed" budget skip on the next tool call.
        self.opt.max_wall_seconds = 100.0
        self.opt._budget_deadline = time.time() - 10.0

    def tearDown(self):
        self.tmp.cleanup()

    def test_budget_skip_arms_with_banked_accept(self):
        self.opt._best_valid_dcp = Path(self.tmp.name) / "b.dcp"
        _async(self.opt.call_tool("vivado_run_tcl", {"command": "puts hi"}))
        self.assertTrue(self.opt._budget_killed)
        self.assertIsNotNone(self.opt._exit_early_reason)
        self.assertIn("budget_skip", self.opt._exit_early_reason)

    def test_budget_skip_does_not_arm_without_banked_accept(self):
        self.assertIsNone(self.opt._best_valid_dcp)
        _async(self.opt.call_tool("vivado_run_tcl", {"command": "puts hi"}))
        self.assertTrue(self.opt._budget_killed)
        self.assertIsNone(self.opt._exit_early_reason)


class _FakeIlsResult:
    """Duck-typed ILSPolishResult stand-in for the monkeypatched inner loop."""

    def __init__(self, notes):
        self.triggered = True
        self.improved = False
        self.skip_reason = ""
        self.baseline_wns = -0.5
        self.best_wns = None
        self.cycles = 2
        self.accepted = 0
        self.physopt_skipped = 0
        self.pristine_rot = 0
        self.combo_cost = {}
        self.notes = list(notes)

    def summary(self):
        return "ILS-polish no-gain (fake)"


class CompositionThroughPolishTests(unittest.TestCase):
    """OQ1 locked behavior: signals arm during/after the ILS stage, the
    LASTMILE + fanout polish stages still run ONCE, and the armed reason
    SURVIVES them (no polish stage may clear/reset it)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.opt = _make_optimizer(tmp)
        # Banked accept exists (guard satisfied) and is on disk so the ILS
        # body skips its recipe-best write_checkpoint branch.
        banked = tmp / "recipe_best.dcp"
        banked.write_bytes(b"dcp")
        self.opt._best_valid_dcp = banked
        # Non-stuck design (gain 0.3 >= stuck_recipe_gain_ns 0.15) -> single
        # recipe_best seed, no raw-seed copies needed.
        self.opt.initial_wns = -0.8
        self.opt.best_wns = -0.5
        # Roomy deadline so the real LASTMILE budget gate passes.
        self.opt._budget_deadline = time.time() + 5000.0
        self.opt._wall_handback_enabled = True

    def tearDown(self):
        self.tmp.cleanup()

    def _run_stage(self, ils_notes):
        import optimizer.ils_polish as ils_mod

        async def fake_run_ils_polish(call_tool, **kwargs):
            return _FakeIlsResult(ils_notes)

        orig = ils_mod.run_ils_polish
        ils_mod.run_ils_polish = fake_run_ils_polish
        # Count polish invocations while still running the REAL methods —
        # the reason must survive the real code paths.
        counts = {"lastmile": 0, "fanout": 0}
        real_lm = self.opt._lastmile_polish_after_ils
        real_fp = self.opt._fanout_polish_after_ils

        async def lm(*a, **k):
            counts["lastmile"] += 1
            return await real_lm(*a, **k)

        async def fp(*a, **k):
            counts["fanout"] += 1
            return await real_fp(*a, **k)

        self.opt._lastmile_polish_after_ils = lm
        self.opt._fanout_polish_after_ils = fp
        try:
            _async(self.opt._run_ils_polish_stage())
        finally:
            ils_mod.run_ils_polish = orig
        return counts

    def test_ils_no_improve_arms_and_survives_polish(self):
        counts = self._run_stage(["no improvement in 2 cycles; stopped"])
        self.assertIsNotNone(self.opt._exit_early_reason)
        self.assertIn("ils_no_improve_stop", self.opt._exit_early_reason)
        # OQ1: both polish stages still ran exactly once...
        self.assertEqual(counts, {"lastmile": 1, "fanout": 1})
        # ...and the composed break predicate holds after them.
        self.assertTrue(self.opt._wall_handback_break_due())

    def test_meaningful_variant_note_also_arms(self):
        self._run_stage(["no meaningful improvement in 2 cycles; stopped"])
        self.assertIsNotNone(self.opt._exit_early_reason)
        self.assertIn("ils_no_improve_stop", self.opt._exit_early_reason)

    def test_lastmile_reject_arms_when_ils_had_no_signal(self):
        # No ILS futility note -> the REAL LASTMILE runs against the mocked
        # session, cannot measure a routed improvement, REJECTS (never-worse)
        # and that reject verdict is signal (b).
        self._run_stage([])
        self.assertIsNotNone(self.opt._exit_early_reason)
        self.assertIn("lastmile_reject", self.opt._exit_early_reason)

    def test_no_arming_without_banked_accept_even_in_stage(self):
        self.opt._best_valid_dcp = None
        self.opt.best_wns = -0.5
        self._run_stage(["no improvement in 2 cycles; stopped"])
        # The stage wrote its own recipe-best snapshot but the guard keys on
        # the banked-accept state at signal time.
        self.assertIsNone(self.opt._exit_early_reason)


class LoopGateWiringTests(unittest.TestCase):
    """The iteration loop must consult the break predicate; the CLI must
    expose --wall-handback and wire it to _wall_handback_enabled. Source
    scan (the loop itself needs a live LLM session to execute)."""

    SRC = dcp_source_text()

    def test_loop_breaks_via_predicate(self):
        self.assertIn("_wall_handback_break_due()", self.SRC)
        self.assertIn('loop_exit_reason = "wall_handback"', self.SRC)

    def test_cli_flag_and_wiring_present(self):
        self.assertIn('"--wall-handback"', self.SRC)
        self.assertIn("_wall_handback_enabled = bool(", self.SRC)

    def test_polish_ladder_never_write_the_reason(self):
        # No polish stage may clear/reset _exit_early_reason: the ONLY
        # assignments allowed are the __init__ default (annotated) and the
        # arm helper.  Comparisons (==, !=, `is`) don't count.
        import re
        assigns = re.findall(
            r"_exit_early_reason(?:\s*:\s*Optional\[str\])?\s*=\s*(?!=)",
            self.SRC)
        self.assertEqual(len(assigns), 2, assigns)


class WrapperPassthroughTests(unittest.TestCase):
    """Wrapper side: WALL_HANDBACK=1 make-var passthrough + --wall-handback."""

    def test_attempt_cmd_appends_wall_handback_var(self):
        from scripts.multi_restart_optimize import _attempt_cmd
        cmd = _attempt_cmd(Path("in.dcp"), Path("/tmp/o.dcp"), 1800.0,
                           False, wall_handback=True)
        self.assertIn("WALL_HANDBACK=1", cmd)

    def test_attempt_cmd_default_off_and_legacy_call(self):
        from scripts.multi_restart_optimize import _attempt_cmd
        # Legacy 4-arg positional call must keep working (default OFF).
        cmd = _attempt_cmd(Path("in.dcp"), Path("/tmp/o.dcp"), 1800.0, True)
        self.assertNotIn("WALL_HANDBACK=1", cmd)

    def test_main_threads_wall_handback_into_run(self):
        import scripts.multi_restart_optimize as mr
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "b.dcp"
            inp.write_bytes(b"x")
            seen = {}

            def fake_run(input_dcp, final_output, total_wall, attempt_floor,
                         max_attempts, repo, cost_cap=0.85, **kw):
                seen.clear()
                seen.update(kw)
                return {"chosen": None}

            orig_run = mr.run
            orig_sig = mr._install_signal_publisher
            mr.run = fake_run
            mr._install_signal_publisher = lambda: None
            try:
                mr.main([str(inp), "--wall-handback"])
                self.assertIs(seen.get("wall_handback"), True)
                mr.main([str(inp)])
                self.assertIs(seen.get("wall_handback"), False)
            finally:
                mr.run = orig_run
                mr._install_signal_publisher = orig_sig


if __name__ == "__main__":
    unittest.main()
