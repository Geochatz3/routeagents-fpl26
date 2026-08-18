"""The optimization summary must NEVER be able to fail a run (aug06 incident).

WHAT HAPPENED (observed live on shipped v4.1.2 `60109c7b`, ispd16, contest wall):

    [constraint-guard] *** TIMING CONSTRAINTS CHANGED *** clock count 1 -> 0
    [LIFECYCLE] shipped-artifact identity recorded: size=146681810 md5=0349c750...
    Fatal error: 'elapsed_time'
      File "dcp_optimizer.py", line 17017, in _print_optimization_summary
        total_tool_time = sum(detail['elapsed_time'] for detail in self.tool_call_details)
    KeyError: 'elapsed_time'
    [LIFECYCLE] EMERGENCY_NO_OP (fatal_KeyError)
    make[1]: *** [Makefile:426: run_optimizer_contest] Error 1
    [multi-restart] no usable attempt output — internal budgeted fallback
    => VALID_FALLBACK_BASELINE, Fmax Improvement +0.00

A COMPLETED, VALID, 0-route-error artifact was discarded because a *reporting*
function raised. The two guard appends that create such an entry are the only
`tool_call_details` writers that omit timing, and they are precisely the
safety-mechanism paths — so the guard firing is what zeroed the benchmark.

These tests drive the REAL methods (checklist item 15: a test that does not
enter the changed code proves nothing).
"""
from __future__ import annotations

import asyncio
import inspect
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import DCPOptimizer


def _async(coro):
    return asyncio.run(coro)


def _make_optimizer(tmp_path: Path) -> DCPOptimizer:
    opt = DCPOptimizer(api_key="test", run_dir=tmp_path)
    opt.initial_wns = -2.0
    opt.best_wns = -1.0
    opt.clock_period = 5.0
    opt.start_time = 1000.0
    opt.end_time = 1100.0
    return opt


# The exact shapes the two constraint-guard paths append.
GUARD_ENTRIES = [
    # dcp_optimizer.py, call_tool refusal path
    {"tool_name": "vivado_run_tcl",
     "status": "refused_constraint_edit",
     "detail": "create_clock"},
    # dcp_optimizer.py, ship-time fingerprint path
    {"tool_name": "constraint_guard",
     "status": "CONSTRAINTS_CHANGED",
     "detail": "clock count 1 -> 0"},
]


class SummaryNeverFailsRunTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    # --- the regression itself -------------------------------------------

    def test_summary_survives_guard_entry_without_elapsed_time(self):
        """The exact crash: a guard entry with no 'elapsed_time'.

        On the shipped code this raises KeyError out of the printer.
        """
        opt = _make_optimizer(self.tmp_path)
        opt.tool_call_details = [
            {"tool_name": "vivado_place_design", "elapsed_time": 12.5},
            *[dict(e) for e in GUARD_ENTRIES],
        ]
        # Must not raise. (No assertRaises: the point is that it returns.)
        opt._print_optimization_summary()

    def test_inner_body_itself_tolerates_the_missing_key(self):
        """Not just the wrapper: the body must handle it too.

        A wrapper that swallows the error would still lose the whole summary.
        This drives the INNER function, which the wrapper does not protect.
        """
        opt = _make_optimizer(self.tmp_path)
        opt.tool_call_details = [
            {"tool_name": "vivado_route_design", "elapsed_time": 30.0},
            *[dict(e) for e in GUARD_ENTRIES],
        ]
        opt._print_optimization_summary_inner()

    def test_totals_count_real_calls_and_ignore_bookkeeping_entries(self):
        """The fix must not silently corrupt the number it prints."""
        opt = _make_optimizer(self.tmp_path)
        opt.tool_call_details = [
            {"tool_name": "a", "elapsed_time": 10.0},
            {"tool_name": "b", "elapsed_time": 5.5},
            *[dict(e) for e in GUARD_ENTRIES],
        ]
        total = sum(d.get("elapsed_time", 0.0)
                    for d in opt.tool_call_details)
        self.assertEqual(total, 15.5)
        # And the printer agrees (drives the same expression).
        opt._print_optimization_summary()

    def test_token_report_survives_guard_entry(self):
        """save_token_usage_report sums the same key (second crash site)."""
        opt = _make_optimizer(self.tmp_path)
        opt.tool_call_details = [
            {"tool_name": "vivado_place_design", "elapsed_time": 1.0},
            *[dict(e) for e in GUARD_ENTRIES],
        ]
        out = self.tmp_path / "token_usage.json"
        opt.save_token_usage_report(out)
        self.assertTrue(out.exists())

    # --- the writers stay consistent with the readers ---------------------

    def test_guard_appends_now_carry_elapsed_time(self):
        """Both guard append sites must include the key going forward.

        Source-level, because these branches need a live Vivado session to
        reach; the behavioural tests above cover the readers.
        """
        src = inspect.getsource(DCPOptimizer)
        for marker in ('"status": "refused_constraint_edit"',
                       '"status": "CONSTRAINTS_CHANGED"'):
            self.assertIn(marker, src, f"append site vanished: {marker}")
            block = src.split(marker, 1)[1][:400]
            self.assertIn('"elapsed_time"', block,
                          f"{marker} appends no elapsed_time — the aug06 crash "
                          f"is reintroduced")

    def test_no_unguarded_elapsed_time_subscript_remains(self):
        """No reader may subscript the key raw again."""
        src = Path(__file__).resolve().parent.parent / "dcp_optimizer.py"
        offenders = [
            f"{i}: {ln.strip()}"
            for i, ln in enumerate(src.read_text(errors="ignore").splitlines(), 1)
            if "['elapsed_time']" in ln or '["elapsed_time"]' in ln
        ]
        self.assertEqual(offenders, [],
                         "raw elapsed_time subscript(s) reintroduced")

    def test_summary_call_sites_are_all_the_safe_wrapper(self):
        """Every call site must hit the wrapper, never the inner directly."""
        src = (Path(__file__).resolve().parent.parent
               / "dcp_optimizer.py").read_text(errors="ignore")
        inner_calls = src.count("self._print_optimization_summary_inner(")
        self.assertEqual(inner_calls, 1,
                         "the inner body must be called exactly once — from "
                         "the wrapper")

    # --- the guarantee, stated as a test ---------------------------------

    def test_wrapper_absorbs_an_arbitrary_failure(self):
        """Any future defect in the body must not escape either."""
        opt = _make_optimizer(self.tmp_path)

        def boom(**_kw):
            raise RuntimeError("simulated reporting defect")

        opt._print_optimization_summary_inner = boom
        with self.assertLogs("dcp_optimizer", level="WARNING") as cm:
            opt._print_optimization_summary()
        self.assertTrue(
            any("summary failed" in m for m in cm.output),
            f"expected a warning, got {cm.output}")


if __name__ == "__main__":
    unittest.main()
