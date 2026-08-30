"""Tests Phase 1 resource limits and unmeasured-baseline finalization.

Optional analysis steps stop launching when Phase 1 reaches `phase1_wall_frac`,
which defaults to 15% of `max_wall_seconds`. Each optional step's scaled
timeout is clamped to the remaining allowance.

Mandatory checkpoint-opening and timing-report steps are not capped. If
`initial_wns` is `None`, improvement is unverifiable and finalization must copy
the baseline.

The tests use stubs and do not invoke FPGA tools or external analysis services.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import (
    DCPOptimizer,
    PHASE1_WALL_FRAC_DEFAULT,
    resolve_phase1_wall_frac,
)


def _async(coro):
    return asyncio.run(coro)


def _make_optimizer(tmp_path: Path) -> DCPOptimizer:
    return DCPOptimizer(api_key="test", run_dir=tmp_path)


class Phase1WallFracResolverTests(unittest.TestCase):
    """resolve_phase1_wall_frac — CLI wins over env; invalid keeps default."""

    def test_unset_returns_default(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("FPL26_PHASE1_WALL_FRAC", None)
            self.assertEqual(resolve_phase1_wall_frac(None),
                             PHASE1_WALL_FRAC_DEFAULT)

    def test_cli_value_wins(self):
        with mock.patch.dict("os.environ",
                             {"FPL26_PHASE1_WALL_FRAC": "0.30"}):
            self.assertEqual(resolve_phase1_wall_frac(0.10), 0.10)

    def test_env_fallback_used_when_cli_unset(self):
        with mock.patch.dict("os.environ",
                             {"FPL26_PHASE1_WALL_FRAC": "0.25"}):
            self.assertEqual(resolve_phase1_wall_frac(None), 0.25)

    def test_unparseable_env_keeps_default(self):
        with mock.patch.dict("os.environ",
                             {"FPL26_PHASE1_WALL_FRAC": "banana"}):
            self.assertEqual(resolve_phase1_wall_frac(None),
                             PHASE1_WALL_FRAC_DEFAULT)

    def test_out_of_range_keeps_default(self):
        # Must be in (0, 1]: 0, negatives, and >1 all fall back.
        for bad in (0.0, -0.5, 1.5):
            self.assertEqual(resolve_phase1_wall_frac(bad),
                             PHASE1_WALL_FRAC_DEFAULT,
                             f"value {bad} should keep the default")

    def test_boundary_one_is_valid(self):
        self.assertEqual(resolve_phase1_wall_frac(1.0), 1.0)


class Phase1WallCapTests(unittest.TestCase):
    """_phase1_call must skip/clamp OPTIONAL steps under the cumulative cap
    and never touch MANDATORY steps."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = _make_optimizer(Path(self.tmp.name))
        # Capture the arguments _phase1_call forwards (incl. the injected
        # per-call timeout) without touching any session dispatch logic.
        self.seen: list[tuple[str, dict]] = []

        async def fake_call_tool(name, args):
            self.seen.append((name, args))
            return "ok: analysis text"

        self._patch = mock.patch.object(
            self.opt, "call_tool", new=mock.AsyncMock(side_effect=fake_call_tool))
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self.tmp.cleanup()

    def test_no_cap_when_wall_budget_unset(self):
        # Legacy/dev mode (max_wall_seconds=None) must behave exactly as
        # before the hardening: full scaled timeout, no skips.
        self.opt.max_wall_seconds = None
        self.opt.phase1_timeout_scale = 3.0
        self.opt._phase1_start_ts = time.time() - 10_000  # huge elapsed, irrelevant
        result = _async(self.opt._phase1_call(
            "vivado_get_critical_high_fanout_nets", {}, 600.0,
            mandatory=False, label="get_critical_high_fanout_nets"))
        self.assertIsNotNone(result)
        self.assertEqual(self.seen[0][1]["timeout"], 1800.0)
        self.assertEqual(self.opt.phase1_skipped, [])

    def test_no_cap_before_phase1_clock_started(self):
        self.opt.max_wall_seconds = 3300.0
        self.opt.phase1_timeout_scale = 3.0
        self.opt._phase1_start_ts = None
        result = _async(self.opt._phase1_call(
            "vivado_get_critical_high_fanout_nets", {}, 600.0,
            mandatory=False))
        self.assertIsNotNone(result)
        self.assertEqual(self.seen[0][1]["timeout"], 1800.0)

    def test_optional_step_skipped_when_allowance_exhausted(self):
        # Allowance = 0.15 × 1000 = 150 s; Phase 1 already burned ~200 s
        # → the optional step must be skipped WITHOUT calling the tool,
        # with reason "phase1_wall_cap".
        self.opt.max_wall_seconds = 1000.0
        self.opt.phase1_wall_frac = 0.15
        self.opt._phase1_start_ts = time.time() - 200.0
        result = _async(self.opt._phase1_call(
            "rapidwright_read_checkpoint", {"dcp_path": "x.dcp"}, 900.0,
            mandatory=False, label="rapidwright_read_checkpoint"))
        self.assertIsNone(result)
        self.assertEqual(self.seen, [], "capped step must not launch the tool")
        self.assertIn("rapidwright_read_checkpoint", self.opt.phase1_skipped)
        self.assertEqual(
            self.opt.phase1_skip_reasons["rapidwright_read_checkpoint"],
            "phase1_wall_cap")

    def test_mandatory_step_never_capped(self):
        # Same exhausted-allowance state — a MANDATORY step must still run
        # with its full scaled timeout (open_checkpoint/report_timing are
        # the LLM's lifeline; capping them guarantees alpha=0).
        self.opt.max_wall_seconds = 1000.0
        self.opt.phase1_wall_frac = 0.15
        self.opt.phase1_timeout_scale = 3.0
        self.opt._phase1_start_ts = time.time() - 200.0
        result = _async(self.opt._phase1_call(
            "vivado_open_checkpoint", {"dcp_path": "x.dcp"}, 600.0,
            mandatory=True, label="open_checkpoint"))
        self.assertIsNotNone(result)
        self.assertEqual(self.seen[0][1]["timeout"], 1800.0)
        self.assertNotIn("open_checkpoint", self.opt.phase1_skipped)

    def test_optional_timeout_clamped_to_remaining_allowance(self):
        # Allowance = 0.15 × 10000 = 1500 s; ~1400 s already burned →
        # remaining ~100 s.  Scaled timeout 600 × 3.0 = 1800 must be
        # clamped down to the remaining allowance, not multiplied past it.
        self.opt.max_wall_seconds = 10_000.0
        self.opt.phase1_wall_frac = 0.15
        self.opt.phase1_timeout_scale = 3.0
        self.opt._phase1_start_ts = time.time() - 1400.0
        result = _async(self.opt._phase1_call(
            "vivado_extract_critical_path_cells", {}, 600.0,
            mandatory=False, label="extract_critical_path_cells"))
        self.assertIsNotNone(result)
        injected = self.seen[0][1]["timeout"]
        self.assertLessEqual(injected, 101.0,
                             f"timeout {injected} not clamped to ~100 s allowance")
        self.assertGreater(injected, 50.0)

    def test_ample_allowance_keeps_full_scaled_timeout(self):
        # Happy path (known designs): Phase 1 just started, allowance is
        # far larger than any scaled step timeout → behavior unchanged.
        self.opt.max_wall_seconds = 100_000.0
        self.opt.phase1_timeout_scale = 3.0
        self.opt._phase1_start_ts = time.time()
        _async(self.opt._phase1_call(
            "vivado_get_critical_high_fanout_nets", {}, 600.0,
            mandatory=False))
        self.assertEqual(self.seen[0][1]["timeout"], 1800.0)
        self.assertEqual(self.opt.phase1_skipped, [])

    def test_skip_reason_rendered_in_display(self):
        self.opt.max_wall_seconds = 1000.0
        self.opt._phase1_start_ts = time.time() - 500.0
        _async(self.opt._phase1_call(
            "vivado_run_tcl", {"command": "report_qor_assessment"}, 240.0,
            mandatory=False, label="report_qor_assessment"))
        display = self.opt._phase1_skipped_display()
        self.assertIn("report_qor_assessment (phase1_wall_cap)", display)

    def test_ordinary_failure_skips_without_wall_cap_reason(self):
        # A non-capped optional failure (tool raises) keeps the plain
        # label — reasons dict is exclusively for wall-cap skips.
        self.opt.max_wall_seconds = None
        self._patch.stop()
        self._patch = mock.patch.object(
            self.opt, "call_tool",
            new=mock.AsyncMock(side_effect=RuntimeError("boom")))
        self._patch.start()
        result = _async(self.opt._phase1_call(
            "vivado_get_critical_high_fanout_nets", {}, 5.0,
            mandatory=False, label="get_critical_high_fanout_nets"))
        self.assertIsNone(result)
        self.assertIn("get_critical_high_fanout_nets", self.opt.phase1_skipped)
        self.assertNotIn("get_critical_high_fanout_nets",
                         self.opt.phase1_skip_reasons)
        self.assertEqual(self.opt._phase1_skipped_display(),
                         ["get_critical_high_fanout_nets"])


class FinalizeInitialWnsNoneTests(unittest.TestCase):
    """initial_wns=None ⇒ no_improvement=True (ship baseline).

    Why unconditional (no routed-scored-clock exception): the best-valid
    lineage records mirror mechanics (eager_mirror/piggyback/backstop),
    not measurement provenance — there is no cheap field distinguishing a
    scored-clock routed measurement, and even one would be uncomparable
    against an unmeasured baseline.  See _finalize_no_improvement.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.opt = _make_optimizer(self.tmp_path)
        # Stage a FRESH best_valid mirror at -1.0 ns (the shape the old
        # code would have shipped as VALID_OPTIMIZED).
        self.best_dcp = self.tmp_path / "best_valid.dcp"
        self.best_edf = self.tmp_path / "best_valid.edf"
        self.best_dcp.write_bytes(b"BEST_VALID_DCP_FROM_UNVERIFIED_IMPROVEMENT")
        self.best_edf.write_bytes(b"BEST_VALID_EDIF_FROM_UNVERIFIED_IMPROVEMENT")
        self.opt._best_valid_dcp = self.best_dcp
        self.opt._best_valid_edif = self.best_edf
        self.opt._best_valid_dcp_wns = -1.0
        self.opt._best_valid_edif_wns = -1.0
        self.baseline = self.tmp_path / "baseline.dcp"
        self.baseline.write_bytes(b"BASELINE_DCP_BYTES")
        self.opt.input_dcp_path = self.baseline

    def tearDown(self):
        self.tmp.cleanup()

    # --- helper truth table ----------------------------------------------

    def test_no_improvement_truth_table(self):
        cases = [
            # (initial_wns, best_wns, expected_no_improvement)
            (None, -1.0, True),    # unmeasured baseline
            (None, 5.0, True),     # even timing-met "best" is unverifiable
            (None, float("-inf"), True),
            (-10.0, -1.0, False),  # measured improvement — unchanged
            (-5.0, -5.0, True),    # equal — unchanged
            (-1.0, -5.0, True),    # regression — unchanged
            (-1.0, None, True),    # never measured anything
        ]
        for initial, best, want in cases:
            self.opt.initial_wns = initial
            self.opt.best_wns = best
            self.assertEqual(
                self.opt._finalize_no_improvement(), want,
                f"initial={initial} best={best}: expected "
                f"no_improvement={want}")

    # --- end-to-end finalize behavior -------------------------------------

    def test_initial_wns_none_ships_baseline_not_best(self):
        # If baseline WNS is unknown, a later mirrored result cannot be proven
        # to improve it. Finalization must ship the baseline fallback rather
        # than report the mirror as VALID_OPTIMIZED.
        self.opt.initial_wns = None
        self.opt.best_wns = -1.0
        out = self.tmp_path / "optimized.dcp"

        # best_valid.edf exists → the no-improvement branch needs no
        # Vivado at all; any call_tool invocation is a bug.
        with mock.patch.object(self.opt, "call_tool",
                               new=mock.AsyncMock(
                                   side_effect=AssertionError(
                                       "initial_wns=None finalize must not "
                                       "touch Vivado"))):
            _async(self.opt._finalize_output_dcp(out))

        self.assertTrue(out.exists())
        self.assertEqual(out.read_bytes(), self.baseline.read_bytes(),
                         "must ship the BASELINE, not the unverifiable best")
        self.assertIn(self.opt.final_status,
                      ("VALID_FALLBACK_BASELINE",
                       "VALID_FALLBACK_BASELINE_NO_EDIF"))
        events = [e["event"] for e in self.opt.lifecycle_log]
        self.assertNotIn("fast_path_best_valid_copy", events)
        self.assertIn("no_improvement_baseline_copy", events)

    def test_measured_improvement_still_ships_fast_path(self):
        # Guard: the fix must not disturb the normal improvement path.
        self.opt.initial_wns = -10.0
        self.opt.best_wns = -1.0
        out = self.tmp_path / "optimized.dcp"
        with mock.patch.object(self.opt, "call_tool",
                               new=mock.AsyncMock(
                                   side_effect=AssertionError(
                                       "fast path must not call Vivado"))):
            _async(self.opt._finalize_output_dcp(out))
        self.assertEqual(out.read_bytes(), self.best_dcp.read_bytes())
        self.assertEqual(self.opt.final_status, "VALID_OPTIMIZED")

    def test_ensure_output_write_noops_on_initial_wns_none(self):
        # _ensure_output_dcp_written shares the verdict: with an
        # unmeasured baseline it must not copy the mirror to the output
        # path (Step 2 would overwrite it with baseline anyway).
        self.opt.initial_wns = None
        self.opt.best_wns = -1.0
        out = self.tmp_path / "optimized.dcp"
        _async(self.opt._ensure_output_dcp_written(out))
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
