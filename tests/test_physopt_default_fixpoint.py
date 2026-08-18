"""`phys_opt_design -directive Default` to fixpoint, pre-loop. (jul31)

WHY. `VivadoMCP/vivado_mcp_server.py` builds its phys_opt command with an
if/else: when `directive` is set, every other argument is DISCARDED, including
`path_groups`. So the LLM calls that FIR_VARIANCE_IS_PHYSOPT_DEPTH_jul29 §4 read
as "aimed at the re-grouped critical path" actually ran a full-design
`phys_opt_design -directive Default` with the group thrown away. The variable is
directive-vs-sub-option, not scope.

Corpus, 833 phys_opt tool calls in 199 agent.logs, replicating on BOTH boxes:
    directive=Default   26/30  = 86.7%
    sub-option only     70/479 = 15%
fir's alpha is decided by how many Default calls the LLM happens to emit —
nDefault>=2 selected the 21.30 mode 8/8; nDefault<=1 gave 9.07. 12.23 MHz.

Every branch that can cost something is pinned here in BOTH directions, because
a gate that only ever takes one branch in the suite is untested, not proven:
  * OFF by default — the flag is one design's evidence (fir, all 30 calls);
  * stops on the FIRST non-gain, and REVERTS to best_valid.dcp so the LLM loop
    never inherits a degraded placement;
  * the wall gate is a MEASUREMENT — call 1 sized from the design-aware model,
    calls 2+ from the OBSERVED duration of the one before, so a big design
    (ispd16/boom, est 600 s) stops before spending a second call;
  * caps at 4 calls (streak gains: 10/11, 8/8, 6/6, 1/4);
  * fails OPEN on a tool error or an exception.
"""
from __future__ import annotations

import asyncio
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dcp_optimizer  # noqa: E402
from dcp_optimizer import DCPOptimizer  # noqa: E402

PODF = "FPL26_PHYSOPT_DEFAULT_FIXPOINT"


class _Stub:
    """Duck-typed `self` — the real optimizer is far too heavy to construct."""

    _PODF_MAX_CALLS = DCPOptimizer._PODF_MAX_CALLS
    _PODF_EPSILON_NS = DCPOptimizer._PODF_EPSILON_NS
    _PODF_MAX_SHARE = DCPOptimizer._PODF_MAX_SHARE
    _physopt_default_fixpoint = DCPOptimizer._physopt_default_fixpoint

    def __init__(self, gains, est_s=70.0, remaining=3100.0, best=-0.313,
                 call_s=25.0, tool_error_at=None, raise_at=None,
                 best_valid=None):
        self._gains = list(gains)      # per-call WNS delta the tool "achieves"
        self._est = est_s
        self._remaining = remaining
        self.best_wns = best
        self._call_s = call_s
        self._tool_error_at = tool_error_at
        self._raise_at = raise_at
        self._best_valid_dcp = best_valid
        self.calls = []                # (tool, args)
        self.n_physopt = 0
        self.clock = 1_000_000.0       # fake monotonic clock, advanced per call

    def _estimate_tool_runtime(self, name, is_risky=None):
        return self._est

    def _budget_remaining(self):
        return self._remaining

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if name != "vivado_phys_opt_design":
            return "OK"
        self.n_physopt += 1
        n = self.n_physopt
        if self._raise_at == n:
            raise RuntimeError("synthetic vivado death")
        if self._tool_error_at == n:
            return "TCL ERROR: synthetic phys_opt failure"
        # Simulate the auto-bank hook: a gain moves best_wns, a non-gain does not.
        d = self._gains[n - 1] if n - 1 < len(self._gains) else 0.0
        if d > 0:
            self.best_wns += d
        # Advance the FAKE clock so the stage reads a realistic observed_s.
        self._remaining -= self._call_s
        self.clock += self._call_s
        return "Physical optimization complete."

    @property
    def physopt_args(self):
        return [a for t, a in self.calls if t == "vivado_phys_opt_design"]

    @property
    def reopened(self):
        return [a for t, a in self.calls if t == "vivado_open_checkpoint"]


def run(stub):
    """Drive the stage against the stub's FAKE clock.

    The measured-cost gate is the whole point of the stage, so the suite must
    not read a real ~0 s elapsed for a call the test declares takes 400 s.
    """
    real = dcp_optimizer.time.time
    dcp_optimizer.time.time = lambda: stub.clock
    try:
        asyncio.run(stub._physopt_default_fixpoint())
    finally:
        dcp_optimizer.time.time = real
    return stub


class PhysoptDefaultFixpointTests(unittest.TestCase):
    def setUp(self):
        self._prev = dcp_optimizer.os.environ.get(PODF)
        dcp_optimizer.os.environ[PODF] = "1"

    def tearDown(self):
        if self._prev is None:
            dcp_optimizer.os.environ.pop(PODF, None)
        else:
            dcp_optimizer.os.environ[PODF] = self._prev

    # ---- the flag itself -------------------------------------------------
    def test_default_off_makes_it_a_no_op(self):
        dcp_optimizer.os.environ[PODF] = "0"
        s = run(_Stub(gains=[0.05, 0.05, 0.05]))
        self.assertEqual(s.n_physopt, 0, "OFF must not call Vivado at all")

    # ---- the fir case it exists for --------------------------------------
    def test_fir_shape_walks_the_baseline_and_stops_at_the_fixpoint(self):
        # jul30's fir: -0.313 -> -0.249 -> -0.223 -> -0.218, then dry.
        s = run(_Stub(gains=[0.064, 0.026, 0.005, 0.0]))
        self.assertEqual(s.n_physopt, 4)
        self.assertAlmostEqual(s.best_wns, -0.218, places=3)
        self.assertTrue(all(a["directive"] == "Default"
                            for a in s.physopt_args))
        self.assertTrue(all("path_groups" not in a for a in s.physopt_args),
                        "must not send path_groups — the server drops it "
                        "silently whenever a directive is set")

    def test_caps_at_four_calls_even_when_still_gaining(self):
        s = run(_Stub(gains=[0.05] * 10))
        self.assertEqual(s.n_physopt, self._Stub_max())

    def _Stub_max(self):
        return DCPOptimizer._PODF_MAX_CALLS

    # ---- never-worse -----------------------------------------------------
    def test_first_non_gain_stops_and_reverts_to_best_valid(self):
        dcp = Path(__file__)          # any existing path
        s = run(_Stub(gains=[0.064, 0.0, 0.05], best_valid=dcp))
        self.assertEqual(s.n_physopt, 2, "must stop on the FIRST non-gain")
        self.assertEqual(len(s.reopened), 1, "must revert the Vivado state")
        self.assertAlmostEqual(s.best_wns, -0.249, places=3,
                               msg="best_wns must be restored to pre-call")

    def test_a_regression_is_reverted_not_kept(self):
        dcp = Path(__file__)
        s = run(_Stub(gains=[-0.100], best_valid=dcp))
        self.assertEqual(s.n_physopt, 1)
        self.assertAlmostEqual(s.best_wns, -0.313, places=3)
        self.assertEqual(len(s.reopened), 1)

    def test_sub_epsilon_gain_counts_as_no_gain(self):
        dcp = Path(__file__)
        s = run(_Stub(gains=[0.001], best_valid=dcp))
        self.assertEqual(s.n_physopt, 1, "0.001 ns is noise, not a gain")
        self.assertEqual(len(s.reopened), 1)

    def test_missing_best_valid_does_not_raise(self):
        s = run(_Stub(gains=[0.0], best_valid=None))
        self.assertEqual(s.n_physopt, 1)
        self.assertEqual(s.reopened, [])

    # ---- the wall gate, BOTH directions ----------------------------------
    def test_big_design_is_blocked_before_the_first_call(self):
        # ispd16 / boom: size model returns the 600s cap. 600*1.3=780 >
        # 15% of 2900s = 435s.
        s = run(_Stub(gains=[0.05] * 4, est_s=600.0, remaining=2900.0))
        self.assertEqual(s.n_physopt, 0,
                         "a 600s-estimate design must not spend even one call")

    def test_small_design_is_allowed(self):
        # fir: 70*1.3=91 <= 15% of 3100s = 465s.
        s = run(_Stub(gains=[0.05, 0.0], est_s=70.0, remaining=3100.0))
        self.assertGreaterEqual(s.n_physopt, 1)

    def test_second_call_is_gated_on_the_MEASURED_first(self):
        # Call 1 fires: est 70*1.3=91 <= 15% of 1000s = 150s. It then MEASURES
        # 400s, and 400*1.3=520 > 15% of the 600s left = 90s -> no call 2.
        s = _Stub(gains=[0.05, 0.05], est_s=70.0, remaining=1000.0,
                  call_s=400.0)
        run(s)
        self.assertEqual(s.n_physopt, 1,
                         "the measured cost of call 1 must gate call 2")

    def test_late_in_the_window_it_does_not_fire(self):
        s = run(_Stub(gains=[0.05], est_s=70.0, remaining=200.0))
        self.assertEqual(s.n_physopt, 0)

    # ---- fail open -------------------------------------------------------
    def test_tool_error_stops_without_raising(self):
        s = run(_Stub(gains=[0.05, 0.05], tool_error_at=1))
        self.assertEqual(s.n_physopt, 1)
        self.assertAlmostEqual(s.best_wns, -0.313, places=3)

    def test_exception_stops_without_propagating(self):
        s = run(_Stub(gains=[0.05, 0.05], raise_at=1))
        self.assertEqual(s.n_physopt, 1)

    def test_timing_already_met_is_a_no_op(self):
        s = run(_Stub(gains=[0.05], best=0.120))
        self.assertEqual(s.n_physopt, 0)

    def test_no_baseline_is_a_no_op(self):
        s = run(_Stub(gains=[0.05], best=float("-inf")))
        self.assertEqual(s.n_physopt, 0)


if __name__ == "__main__":
    unittest.main()


class CallSiteIsPostLoopTests(unittest.TestCase):
    """v2's whole fix is WHERE it is called. Pin that, or a refactor undoes it.

    v1 ran pre-loop and its own all-16 A/B killed it: on digit, from an identical
    -1.025 start, ship default let the LLM's own phys_opt bank +0.204 (alpha
    72.59) while v1's stage banked +0.110 first and the loop then found nothing
    (alpha 59.65). The stage competes with the LLM rather than adding to it.
    Post-loop there is no call left to pre-empt.

    A behavioural test cannot see this — both placements produce identical stage
    logs. The ordering is only visible in the source, so that is what is asserted.
    """

    def setUp(self):
        self.src = (Path(__file__).resolve().parents[1] / "dcp_optimizer.py").read_text()

    def test_called_exactly_once(self):
        self.assertEqual(self.src.count("await self._physopt_default_fixpoint()"), 1)

    def test_call_is_inside_the_loop_exit_tail_not_before_the_loop(self):
        call = self.src.index("await self._physopt_default_fixpoint()")
        tail = self.src.index("async def _exit_with_ils_polish")
        loop = self.src.index("while self.iteration < max_iterations:")
        self.assertGreater(call, tail,
                           "the stage must be called from _exit_with_ils_polish")
        self.assertLess(call, loop,
                        "_exit_with_ils_polish is defined above the loop; if this "
                        "fails the call has drifted out of the exit tail")

    def test_it_runs_before_the_ILS_trigger_reads_best_wns(self):
        # The gain must be visible to the ILS entry baseline, which is the whole
        # mechanism by which fir converts -0.249 into -0.218.
        call = self.src.index("await self._physopt_default_fixpoint()")
        trig = self.src.index("ILS-polish LOOP-EXIT trigger (generalizable)")
        self.assertLess(call, trig,
                        "the stage must bank BEFORE the ILS-polish trigger reads "
                        "best_wns, or fir's mechanism is lost")
