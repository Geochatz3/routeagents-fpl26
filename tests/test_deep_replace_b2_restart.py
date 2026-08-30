"""Restart Vivado before the second sibling physical-optimization pass on
expensive designs.

A fresh process avoids retaining placer and router memory during physical
optimization. The restart is gated by measured place-and-route cost; the saved
checkpoint is registered by path beforehand and is reopened afterward. Tests
cover both sides of the cost gate.
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimizer.deep_replace_sibling import (  # noqa: E402
    DEEP_REPLACE_B2_RESTART_MIN_PR_S,
    b2_restart_min_pr_s,
    run_deep_replace_sibling,
)

B2_CMD = "phys_opt_design -directive AlternateFlowWithRetiming"


class _Vivado:
    def __init__(self, wns=-10.378, unrouted=0, whs=0.05, restart_fails=False):
        self.cmds = []
        self.wns, self.unrouted, self.whs = wns, unrouted, whs
        self.restart_fails = restart_fails

    async def call_tool(self, name, args):
        self.cmds.append(args.get("command", name))
        if name == "vivado_restart_vivado" and self.restart_fails:
            return "TCL ERROR: synthetic restart failure"
        return "OK"


def _run(v, **over):
    async def measure(call_tool, tcl, timeout_s=None):
        return v.wns, v.unrouted

    async def measure_hold(call_tool, timeout_s=None):
        return v.whs

    import time as _t
    kw = dict(pristine_dcp="/in/boom.dcp", chain_best_wns=-10.399,
              deadline_ts=_t.time() + 5000, wns_tcl="get_wns",
              out_dcp="/out/cand.dcp", log=lambda m: None,
              measure=measure, measure_hold=measure_hold,
              tool_ok=lambda r: "ERROR" not in str(r))
    kw.update(over)
    return asyncio.run(run_deep_replace_sibling(v.call_tool, **kw))


def _restarts_before_b2(cmds):
    """How many restarts precede the B2 phys_opt call."""
    if B2_CMD not in cmds:
        return None
    return cmds[:cmds.index(B2_CMD)].count("vivado_restart_vivado")


class B2RestartTests(unittest.TestCase):
    ENVS = ("FPL26_DEEP_REPLACE_B2_RESTART",
            "FPL26_DEEP_REPLACE_B2_RESTART_MIN_PR_S")

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in self.ENVS}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # ---------------------------------------------------------------- gate ON
    def test_expensive_design_gets_a_fresh_session_and_reopens_B1(self):
        os.environ["FPL26_DEEP_REPLACE_B2_RESTART_MIN_PR_S"] = "0"
        v = _Vivado()
        _run(v)
        n = _restarts_before_b2(v.cmds)
        self.assertIsNotNone(n, "B2 never ran")
        self.assertEqual(n, 2, "expected the pre-B1 restart AND a new pre-B2 one")
        # and it must reopen the BANKED candidate, not the pristine input
        i_b2 = v.cmds.index(B2_CMD)
        self.assertIn("open_checkpoint {/out/cand.dcp}", v.cmds[:i_b2],
                      "B2 must resume from the banked B1, not from pristine")

    # --------------------------------------------------------------- gate OFF
    def test_cheap_design_does_NOT_pay_the_reopen(self):
        """The negative control. Without it, a gate that always fires looks fine."""
        os.environ["FPL26_DEEP_REPLACE_B2_RESTART_MIN_PR_S"] = "999999"
        v = _Vivado()
        _run(v)
        self.assertEqual(_restarts_before_b2(v.cmds), 1,
                         "cheap designs must keep the earlier single restart")
        i_b2 = v.cmds.index(B2_CMD)
        self.assertNotIn("open_checkpoint {/out/cand.dcp}", v.cmds[:i_b2])

    def test_kill_switch_restores_the_old_behaviour(self):
        os.environ["FPL26_DEEP_REPLACE_B2_RESTART_MIN_PR_S"] = "0"
        os.environ["FPL26_DEEP_REPLACE_B2_RESTART"] = "0"
        v = _Vivado()
        _run(v)
        self.assertEqual(_restarts_before_b2(v.cmds), 1)

    # ------------------------------------------------------------ never fatal
    def test_a_failed_restart_still_runs_B2(self):
        """B1 is banked; a restart problem must never cost the phys_opt."""
        os.environ["FPL26_DEEP_REPLACE_B2_RESTART_MIN_PR_S"] = "0"
        v = _Vivado(restart_fails=True)
        r = _run(v)
        self.assertIn(B2_CMD, v.cmds,
                      "a failed restart must fall through to B2, not skip it")
        self.assertIsNotNone(r)

    # -------------------------------------------------------------- threshold
    def test_default_threshold_selects_the_boom_class_only(self):
        """1200 s sits above every non-boom measured place+route in the corpus."""
        self.assertEqual(b2_restart_min_pr_s(),
                         DEEP_REPLACE_B2_RESTART_MIN_PR_S)
        boom_pr = 798 + 719          # the run that died; another measured 875+781
        logicnets_pr = 161 + 107     # measured anchors
        self.assertGreater(boom_pr, DEEP_REPLACE_B2_RESTART_MIN_PR_S)
        self.assertLess(logicnets_pr, DEEP_REPLACE_B2_RESTART_MIN_PR_S)

    def test_env_override_parses_and_bad_values_fall_back(self):
        os.environ["FPL26_DEEP_REPLACE_B2_RESTART_MIN_PR_S"] = "900"
        self.assertEqual(b2_restart_min_pr_s(), 900.0)
        os.environ["FPL26_DEEP_REPLACE_B2_RESTART_MIN_PR_S"] = "not-a-number"
        self.assertEqual(b2_restart_min_pr_s(),
                         DEEP_REPLACE_B2_RESTART_MIN_PR_S)


if __name__ == "__main__":
    unittest.main()
