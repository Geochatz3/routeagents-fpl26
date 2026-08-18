"""Typed gate ledger (jul26 panel #1 signal).

The invariants that matter are not "it writes JSON" — they are:
  1. with FPL26_GATE_LOG unset, behaviour is byte-identical to before (default OFF);
  2. emit() can NEVER raise, whatever it is handed, because a telemetry failure must not
     be able to change an optimization decision;
  3. a refusal records the PROVENANCE of the estimate that drove it, which is the field
     that separates a legitimate bound (HISTORY) from the jul26 defect family (CONSTANT).
"""
import json
import os
import tempfile
import unittest
from unittest import mock

import dcp_optimizer
from dcp_optimizer import DCPOptimizer
from optimizer import gate_log


def _opt(cells=8414, remaining=277.0, history=None):
    o = DCPOptimizer.__new__(DCPOptimizer)
    o._tool_runtime_history = history or {}
    o._input_cell_count = cells
    o._last_estimate_provenance = None
    o._budget_deadline = 1.0            # non-None so the gate engages
    o.best_wns = -0.85
    o.iteration = 7
    o._design_routed_state = None
    o._budget_remaining = lambda: remaining
    return o


class GateLogDefaultOffTests(unittest.TestCase):

    def setUp(self):
        gate_log.reset_for_test()

    def tearDown(self):
        gate_log.reset_for_test()

    @mock.patch.dict(os.environ, {}, clear=False)
    def test_disabled_when_env_unset(self):
        os.environ.pop("FPL26_GATE_LOG", None)
        gate_log.reset_for_test()
        self.assertFalse(gate_log.enabled())
        gate_log.emit("g", gate_log.VERDICT_REFUSE)   # must be a silent no-op

    def test_emit_never_raises_on_hostile_input(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ,
                                 {"FPL26_GATE_LOG": os.path.join(d, "g.jsonl")}):
                gate_log.reset_for_test()
                class Boom:
                    def __repr__(self): raise RuntimeError("nope")
                # unserialisable values, bad numerics, None gate — none may propagate
                gate_log.emit(None, None, predicted_s="abc", threshold_s=object(),
                              observed_s=0, extra={"x": Boom()})
                gate_log.emit("g", "v", extra="not-a-dict")

    def test_unwritable_path_is_silent(self):
        with mock.patch.dict(os.environ, {"FPL26_GATE_LOG": "/proc/cannot/write.jsonl"}):
            gate_log.reset_for_test()
            gate_log.emit("g", gate_log.VERDICT_REFUSE)   # must not raise


class GateLogContentTests(unittest.TestCase):

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.d.name, "sub", "gate.jsonl")
        self.env = mock.patch.dict(os.environ, {"FPL26_GATE_LOG": self.path})
        self.env.start()
        gate_log.reset_for_test()

    def tearDown(self):
        self.env.stop()
        gate_log.reset_for_test()
        self.d.cleanup()

    def _rows(self):
        with open(self.path) as f:
            return [json.loads(l) for l in f if l.strip()]

    def test_creates_parent_dir_and_writes_typed_row(self):
        gate_log.emit("budget_skip", gate_log.VERDICT_REFUSE,
                      predicted_s=600.0, threshold_s=277.0,
                      provenance=gate_log.PROV_CONSTANT, tool="vivado_phys_opt_design")
        r = self._rows()[0]
        self.assertEqual(r["gate"], "budget_skip")
        self.assertEqual(r["verdict"], "REFUSE")
        self.assertEqual(r["provenance"], "CONSTANT")
        self.assertAlmostEqual(r["margin_s"], 277.0 - 600.0)   # negative => refused

    def test_over_estimate_ratio_is_computed(self):
        """The field kimi-k3's 'absurd-constant' rule sorts on."""
        gate_log.emit("tool_runtime", gate_log.VERDICT_OBSERVE,
                      predicted_s=600.0, observed_s=20.0)
        self.assertAlmostEqual(self._rows()[0]["over_estimate_ratio"], 30.0)

    # ---- the wired call sites ----

    def test_budget_gate_records_CONSTANT_provenance_when_blind(self):
        """Kill switch on => blind 600s => the refusal must be marked CONSTANT."""
        with mock.patch.dict(os.environ, {"FPL26_SIZE_AWARE_TOOL_EST": "0"}):
            o = _opt(remaining=277.0)
            skip, why = o._should_skip_for_budget("vivado_phys_opt_design")
        self.assertTrue(skip)
        row = [r for r in self._rows() if r["gate"] == "budget_skip"][0]
        self.assertEqual(row["verdict"], "REFUSE")
        self.assertEqual(row["provenance"], "CONSTANT")
        self.assertEqual(row["predicted_s"], dcp_optimizer.DEFAULT_RISKY_RUNTIME_S)

    def test_budget_gate_records_MODEL_and_ALLOWS_after_the_fix(self):
        """Size-aware estimate on => 48s => ALLOW, provenance MODEL."""
        o = _opt(cells=8414, remaining=277.0)
        skip, why = o._should_skip_for_budget("vivado_phys_opt_design")
        self.assertFalse(skip, "277s must now clear the design-sized estimate")
        row = [r for r in self._rows() if r["gate"] == "budget_skip"][0]
        self.assertEqual(row["verdict"], "ALLOW")
        self.assertEqual(row["provenance"], "MODEL")

    def test_measured_history_is_marked_HISTORY(self):
        """A refusal bounded by a real measurement is legitimate — must be labelled."""
        o = _opt(remaining=50.0,
                 history={"vivado_place_design": [61.0, 60.0, 62.0]})
        skip, why = o._should_skip_for_budget("vivado_place_design")
        self.assertTrue(skip)
        row = [r for r in self._rows() if r["gate"] == "budget_skip"][0]
        self.assertEqual(row["provenance"], "HISTORY")
        self.assertEqual(row["predicted_s"], 62.0)

    def test_observed_runtime_leg_is_recorded(self):
        o = _opt()
        o._record_tool_runtime("vivado_route_design", 20.9)
        row = [r for r in self._rows() if r["gate"] == "tool_runtime"][0]
        self.assertEqual(row["verdict"], "OBSERVE")
        self.assertEqual(row["observed_s"], 20.9)

    def test_subsecond_runtimes_still_not_recorded(self):
        """Pre-existing behaviour: <1s calls are skipped. Must be unchanged."""
        o = _opt()
        o._record_tool_runtime("vivado_run_tcl", 0.4)
        self.assertFalse(os.path.exists(self.path) and
                         [r for r in self._rows() if r["gate"] == "tool_runtime"])


if __name__ == "__main__":
    unittest.main()
