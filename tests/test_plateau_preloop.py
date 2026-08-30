"""Test plateau detection when the best result is banked before the optimization
loop.

The iteration-zero sentinel can mean either that nothing has succeeded yet or
that a result was banked before iteration one. The former must not trigger an
early exit, while the latter must arm plateau detection when the loop makes no
progress.
"""
from __future__ import annotations

import unittest


TRUTHY = ("1", "true", "on", "yes")


def arm_flag(env_value: str | None) -> bool:
    """Mirror of the flag resolver in dcp_optimizer.py."""
    return (env_value or "0").strip().lower() in TRUTHY


def pre_loop_banked(flag_on: bool, initial_wns, best_wns) -> bool:
    """Mirror of the `_pre_loop_best_banked` computation at loop start."""
    return bool(flag_on and initial_wns is not None and best_wns > initial_wns)


def plateau_exits(last_improvement_iter: int, iteration: int,
                  pre_loop_best_banked: bool) -> bool:
    """Mirror of the armed plateau condition."""
    iters_since_improve = iteration - last_improvement_iter
    return ((last_improvement_iter > 0 or pre_loop_best_banked)
            and iters_since_improve >= 3)


class FlagDefaultTests(unittest.TestCase):
    def test_ships_off_by_default(self):
        """House rule: a mechanism ships OFF until farm-validated."""
        self.assertFalse(arm_flag(None))
        self.assertFalse(arm_flag(""))
        self.assertFalse(arm_flag("0"))

    def test_accepts_the_usual_truthy_spellings(self):
        for v in TRUTHY + ("ON", "True", " yes "):
            self.assertTrue(arm_flag(v), v)


class ArmingTests(unittest.TestCase):
    def test_not_armed_when_flag_off_even_with_preloop_best(self):
        """Default path must be byte-identical to pre-fix behaviour."""
        self.assertFalse(pre_loop_banked(False, -2.0, -1.5))

    def test_armed_when_preloop_improved_over_baseline(self):
        self.assertTrue(pre_loop_banked(True, -2.0, -1.5))

    def test_not_armed_when_best_equals_baseline(self):
        """No pre-loop IMPROVEMENT means state (a): the LLM is still ramping and
        the ramp-up protection must hold."""
        self.assertFalse(pre_loop_banked(True, -2.0, -2.0))

    def test_not_armed_when_best_is_worse_than_baseline(self):
        self.assertFalse(pre_loop_banked(True, -2.0, -2.5))

    def test_not_armed_when_baseline_unknown(self):
        """initial_wns is None on designs where the baseline probe did not
        report — cannot conclude an improvement, so do not arm."""
        self.assertFalse(pre_loop_banked(True, None, -1.5))


class ExitBehaviourTests(unittest.TestCase):
    def test_regression_state_a_never_exits_early(self):
        """Verify that an empty initial state never triggers the plateau exit.

        When no result has been banked, iteration zero indicates that the agent
        is still ramping up; pre-loop plateau handling must preserve this
        protection.
        """
        for iteration in range(1, 8):
            for armed in (False, True):
                self.assertFalse(
                    plateau_exits(0, iteration, pre_loop_best_banked=False),
                    f"exited at iter {iteration} with nothing banked "
                    f"(armed={armed}) — ramp-up protection broken",
                )

    def test_state_b_exits_at_iteration_3(self):
        """Best banked pre-loop: iters_since_improve == iteration, so 3 empty
        iterations on top of an already-banked best hands over to the tail."""
        self.assertFalse(plateau_exits(0, 1, True))
        self.assertFalse(plateau_exits(0, 2, True))
        self.assertTrue(plateau_exits(0, 3, True))

    def test_state_b_unarmed_runs_to_the_empty_spin_backstop(self):
        """Documents the defect being fixed: unarmed, state (b) does not exit
        even at iteration 10 — only the iter>=8 + 5-empty backstop can."""
        self.assertFalse(plateau_exits(0, 10, pre_loop_best_banked=False))

    def test_normal_in_loop_plateau_is_unchanged(self):
        """An improvement at iteration 4 still exits 3 iterations later,
        identically to pre-fix behaviour, whether or not the fix is armed."""
        for armed in (False, True):
            self.assertFalse(plateau_exits(4, 6, armed))
            self.assertTrue(plateau_exits(4, 7, armed))

    def test_arming_never_delays_an_exit_that_would_already_fire(self):
        """The fix can only ADD exits, never remove one — it is an `or`."""
        for lii in range(0, 6):
            for it in range(1, 12):
                if plateau_exits(lii, it, False):
                    self.assertTrue(
                        plateau_exits(lii, it, True),
                        f"arming suppressed an exit at lii={lii} iter={it}",
                    )


class SourceIntegrationTests(unittest.TestCase):
    """The mirrors above test the logic; these pin that the real call site uses
    it, so the mirrors cannot drift away from the code they describe."""

    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        src = Path(__file__).resolve().parent.parent / "dcp_optimizer.py"
        cls.text = src.read_text(encoding="utf-8", errors="replace")

    def test_flag_is_read_with_off_default(self):
        self.assertIn('os.environ.get("FPL26_PLATEAU_PRELOOP_FIX", "0")', self.text)

    def test_plateau_condition_is_armed_by_the_preloop_flag(self):
        self.assertIn('or getattr(self, "_pre_loop_best_banked", False))', self.text)

    def test_exit_still_routes_through_the_zero_cost_tail(self):
        """When the fix is wrong the cost must be forgone LLM iterations, never a
        lost result — so the exit must still run ILS polish + finalize."""
        i = self.text.index('or getattr(self, "_pre_loop_best_banked", False))')
        self.assertIn("_exit_with_ils_polish", self.text[i:i + 900])


if __name__ == "__main__":
    unittest.main()
