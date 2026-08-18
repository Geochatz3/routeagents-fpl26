"""Tests for optimizer.escalation_policy — model-escalation triggers.

Pure policy module: no Vivado, no LLM calls. Each test constructs a
RunSnapshot and asserts the EscalationDecision matches the brief's
trigger semantics.
"""
from __future__ import annotations

import unittest

from optimizer.escalation_policy import (
    RunSnapshot,
    EscalationDecision,
    evaluate,
    T1_STAGNATION_ITERS,
    T1_MIN_BUDGET_S,
    T2_BAD_TCL_RECENT,
    T3_FMAX_RATIO_MAX,
    T3_MIN_BUDGET_S,
    T4_MIN_BUDGET_S,
    X1_NEAR_CLOSURE_WNS_NS,
)


def _base_snapshot(**overrides) -> RunSnapshot:
    """Build a benign snapshot — no triggers fire, no blocks apply."""
    base = dict(
        current_iter=3,
        last_improvement_iter=2,
        initial_wns_ns=-2.0,
        best_wns_ns=-1.5,
        achievable_fmax_ratio=0.50,
        remaining_wall_budget_s=2000.0,
        bad_tcl_count_recent=0,
        is_done_signal=False,
        router_unattempted_heavy_step=None,
        already_escalated_this_run=False,
    )
    base.update(overrides)
    return RunSnapshot(**base)


class AntiTriggerTests(unittest.TestCase):
    """Anti-triggers must block escalation even when positive triggers
    would otherwise fire."""

    def test_x1_routine_progress_blocks(self):
        # Just improved this iter → reward cheap model.
        snap = _base_snapshot(
            current_iter=5, last_improvement_iter=5,
            # And set up a T1-eligible profile too — anti-trigger wins.
            best_wns_ns=-1.5, initial_wns_ns=-2.0,
            remaining_wall_budget_s=1200.0,
        )
        d = evaluate(snap)
        self.assertFalse(d.should_escalate)
        self.assertIn("ROUTINE_PROGRESS", d.rationale)

    def test_x2_near_closure_blocks(self):
        snap = _base_snapshot(best_wns_ns=-0.1)  # within X1 threshold
        d = evaluate(snap)
        self.assertFalse(d.should_escalate)
        self.assertIn("NEAR_CLOSURE", d.rationale)

    def test_x3_budget_exhausted_blocks(self):
        # Budget below T4 floor → no escalation can land.
        snap = _base_snapshot(remaining_wall_budget_s=T4_MIN_BUDGET_S - 1)
        d = evaluate(snap)
        self.assertFalse(d.should_escalate)
        self.assertIn("BUDGET_EXHAUSTED", d.rationale)

    def test_x4_already_escalated_blocks(self):
        # One-shot per run in current phase.
        snap = _base_snapshot(
            already_escalated_this_run=True,
            # Make T1 eligible to confirm the block wins.
            current_iter=10, last_improvement_iter=5,
            remaining_wall_budget_s=2000.0,
        )
        d = evaluate(snap)
        self.assertFalse(d.should_escalate)
        self.assertIn("ALREADY_ESCALATED", d.rationale)


class T1StagnationTests(unittest.TestCase):
    def test_fires_when_iters_since_improve_meets_threshold(self):
        snap = _base_snapshot(
            current_iter=7,
            last_improvement_iter=7 - T1_STAGNATION_ITERS,
            best_wns_ns=-1.5, initial_wns_ns=-2.0,
            remaining_wall_budget_s=T1_MIN_BUDGET_S + 1.0,
        )
        d = evaluate(snap)
        self.assertTrue(d.should_escalate)
        self.assertEqual(d.trigger_id, "T1")

    def test_does_not_fire_without_prior_improvement(self):
        # last_improvement_iter == 0 — never improved → can't measure stagnation
        snap = _base_snapshot(
            current_iter=10, last_improvement_iter=0,
            remaining_wall_budget_s=2000.0,
        )
        d = evaluate(snap)
        self.assertFalse(d.should_escalate)

    def test_does_not_fire_below_iters_threshold(self):
        snap = _base_snapshot(
            current_iter=5, last_improvement_iter=4,  # only 1 iter
            best_wns_ns=-1.5, initial_wns_ns=-2.0,
            remaining_wall_budget_s=1200.0,
        )
        d = evaluate(snap)
        self.assertFalse(d.should_escalate)

    def test_does_not_fire_below_budget_threshold(self):
        snap = _base_snapshot(
            current_iter=7, last_improvement_iter=5,
            best_wns_ns=-1.5, initial_wns_ns=-2.0,
            remaining_wall_budget_s=T1_MIN_BUDGET_S - 1.0,
        )
        d = evaluate(snap)
        # T4 floor (X3) blocks first when budget is severely low.
        self.assertFalse(d.should_escalate)


class T2BadTclBurstTests(unittest.TestCase):
    def test_fires_at_threshold(self):
        # Need to avoid X1 routine progress trigger (last_improvement_iter != current_iter)
        snap = _base_snapshot(
            current_iter=5, last_improvement_iter=2,
            bad_tcl_count_recent=T2_BAD_TCL_RECENT,
            remaining_wall_budget_s=1000.0,
        )
        d = evaluate(snap)
        self.assertTrue(d.should_escalate)
        self.assertEqual(d.trigger_id, "T2")

    def test_does_not_fire_below_threshold(self):
        snap = _base_snapshot(
            bad_tcl_count_recent=T2_BAD_TCL_RECENT - 1,
            current_iter=5, last_improvement_iter=4,  # avoid T1 cross-fire
        )
        d = evaluate(snap)
        self.assertFalse(d.should_escalate)


class T3HighUpsideStuckTests(unittest.TestCase):
    def test_fires_on_large_gap_with_budget(self):
        snap = _base_snapshot(
            achievable_fmax_ratio=T3_FMAX_RATIO_MAX - 0.01,
            current_iter=5, last_improvement_iter=3,
            best_wns_ns=-5.0, initial_wns_ns=-6.0,
            remaining_wall_budget_s=T3_MIN_BUDGET_S + 1.0,
        )
        d = evaluate(snap)
        self.assertTrue(d.should_escalate)
        # T1 may fire first depending on order; either T1 or T3 acceptable.
        self.assertIn(d.trigger_id, ("T1", "T3"))

    def test_does_not_fire_when_near_target(self):
        snap = _base_snapshot(
            achievable_fmax_ratio=0.80,  # close to target
            current_iter=5, last_improvement_iter=3,
            remaining_wall_budget_s=2000.0,
        )
        # T1 might fire here (iters_since=2), so disable it by making best == initial.
        snap = _base_snapshot(
            achievable_fmax_ratio=0.80,
            current_iter=5, last_improvement_iter=0,
            best_wns_ns=-2.0, initial_wns_ns=-2.0,
            remaining_wall_budget_s=2000.0,
        )
        d = evaluate(snap)
        self.assertFalse(d.should_escalate)


class T4RouterPlanSkipTests(unittest.TestCase):
    def test_fires_when_done_signal_and_unattempted_step(self):
        snap = _base_snapshot(
            is_done_signal=True,
            router_unattempted_heavy_step="vivado_place_design directive=Auto_1",
            current_iter=5, last_improvement_iter=3,
            remaining_wall_budget_s=600.0,
        )
        d = evaluate(snap)
        self.assertTrue(d.should_escalate)
        # T4 has the highest priority of the positive triggers — should win.
        self.assertEqual(d.trigger_id, "T4")

    def test_does_not_fire_without_done_signal(self):
        snap = _base_snapshot(
            is_done_signal=False,
            router_unattempted_heavy_step="vivado_place_design directive=Auto_1",
        )
        d = evaluate(snap)
        # Without done signal, T4 doesn't fire — but T1 might.
        # Verify T4 specifically isn't the reason.
        self.assertNotEqual(d.trigger_id, "T4")


class TelemetryShapeTests(unittest.TestCase):
    """The returned decision must carry enough info for post-run audit."""

    def test_decision_includes_snapshot_for_audit(self):
        snap = _base_snapshot()
        d = evaluate(snap)
        self.assertIsInstance(d, EscalationDecision)
        # The snapshot is echoed back so telemetry consumers can capture
        # the feature values that drove the decision.
        self.assertEqual(d.snapshot, snap)

    def test_no_trigger_returns_NONE(self):
        snap = _base_snapshot()  # nothing fires
        d = evaluate(snap)
        self.assertFalse(d.should_escalate)
        # Either NONE (no trigger no block) or X* (block). With base
        # snapshot last_improvement_iter==2, current_iter==3 → iters_since=1
        # → T1 doesn't fire. No block applies. Should be NONE.
        self.assertEqual(d.trigger_id, "NONE")


if __name__ == "__main__":
    unittest.main()
