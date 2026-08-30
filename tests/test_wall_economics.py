"""Test the intra-attempt wall-economics stop rule.

The rule is derived from the scoring function, so these tests pin its
arithmetic to hand-computed values rather than a tuned threshold.
"""
import pytest

from optimizer.wall_economics import (
    MIN_PENALTY_FACTOR,
    PENALTY_PER_HOUR,
    marginal_hurdle_mhz,
    should_stop_for_wall,
)


# ----------------------------------------------------------- the arithmetic
def test_hurdle_matches_the_scoring_formula_by_hand():
    """alpha 36.33 over 1730 s at P=0.9:
       36.33 * 0.1 * (1730/3600) / 0.9 = 1.939 MHz."""
    h = marginal_hurdle_mhz(36.33, 1730.0, 0.9)
    assert h == pytest.approx(36.33 * 0.1 * (1730.0 / 3600.0) / 0.9, rel=1e-9)
    assert h == pytest.approx(1.939, abs=0.001)


def test_hurdle_scales_linearly_in_alpha_and_time():
    a = marginal_hurdle_mhz(50.0, 900.0)
    assert marginal_hurdle_mhz(100.0, 900.0) == pytest.approx(2 * a)
    assert marginal_hurdle_mhz(50.0, 1800.0) == pytest.approx(2 * a)


def test_hurdle_is_zero_without_alpha_or_horizon():
    assert marginal_hurdle_mhz(0.0, 900.0) == 0.0
    assert marginal_hurdle_mhz(-5.0, 900.0) == 0.0
    assert marginal_hurdle_mhz(36.0, 0.0) == 0.0


def test_penalty_factor_is_clamped():
    """A pathological P must not make the rule arbitrarily eager."""
    assert (marginal_hurdle_mhz(36.0, 1800.0, 0.001)
            == pytest.approx(marginal_hurdle_mhz(36.0, 1800.0,
                                                 MIN_PENALTY_FACTOR)))


# ------------------------------------------------------------- fail-safe
@pytest.mark.parametrize("alpha", [None, 0.0, -3.0])
def test_never_stops_before_anything_is_banked(alpha):
    stop, why = should_stop_for_wall(
        alpha_mhz=alpha, remaining_s=1800.0, since_last_gain_s=9999.0,
        observation_window_s=100.0)
    assert stop is False and "forfeit" in why


def test_fails_open_when_gain_history_is_unmeasurable():
    for kw in ({"since_last_gain_s": None, "observation_window_s": 100.0},
               {"since_last_gain_s": 500.0, "observation_window_s": None}):
        stop, why = should_stop_for_wall(
            alpha_mhz=36.33, remaining_s=1800.0, **kw)
        assert stop is False and "fail open" in why


def test_fails_open_without_a_completed_heavy_move():
    stop, why = should_stop_for_wall(
        alpha_mhz=36.33, remaining_s=1800.0, since_last_gain_s=900.0,
        observation_window_s=0.0)
    assert stop is False and "fail open" in why


def test_no_remaining_wall_is_not_a_stop():
    stop, _ = should_stop_for_wall(
        alpha_mhz=36.33, remaining_s=0.0, since_last_gain_s=900.0,
        observation_window_s=100.0)
    assert stop is False


# ------------------------------------------------------ the fair-chance rule
def test_does_not_stop_before_one_full_heavy_move_has_passed():
    """The window is a MEASURED cost, not a tuned constant: until a heavy
    move's worth of wall has elapsed without gain, we have no evidence."""
    stop, why = should_stop_for_wall(
        alpha_mhz=36.33, remaining_s=1800.0,
        since_last_gain_s=700.0, observation_window_s=1106.0)
    assert stop is False and "fair chance" in why


def test_stops_once_a_full_heavy_move_passed_with_no_gain():
    stop, why = should_stop_for_wall(
        alpha_mhz=36.33, remaining_s=1800.0,
        since_last_gain_s=1200.0, observation_window_s=1106.0,
        gain_in_window_mhz=0.0)
    assert stop is True and "STOP" in why


def test_keeps_going_when_the_realized_rate_clears_the_hurdle():
    """Still gaining fast enough to pay for the wall it consumes."""
    stop, why = should_stop_for_wall(
        alpha_mhz=36.33, remaining_s=1800.0,
        since_last_gain_s=1200.0, observation_window_s=1106.0,
        gain_in_window_mhz=8.0)          # 0.00667 MHz/s -> 12 MHz projected
    assert stop is False and "keep going" in why


# ------------------------------------------------ the run that motivated it
def test_chain12_boom_soc_would_have_stopped():
    """Verify the stop rule halts after a banked gain when further expensive
    candidates cannot repay their wall-time cost.

    The scenario models a long optimization window followed by candidates that
    leave timing unchanged.
    """
    stop, why = should_stop_for_wall(
        alpha_mhz=36.33, remaining_s=1730.0,
        since_last_gain_s=1730.0, observation_window_s=1106.0,
        gain_in_window_mhz=0.0, penalty_factor=0.906)
    assert stop is True
    # It should also state what stopping is worth: 36.33*0.1*(1730/3600) = 1.746
    assert "1.75" in why or "1.74" in why


def test_a_run_still_climbing_is_not_stopped():
    """Counter-case from the same family: if the tail were still finding
    ~5 MHz per heavy move, the rule must let it run."""
    stop, _ = should_stop_for_wall(
        alpha_mhz=14.06, remaining_s=1200.0,
        since_last_gain_s=805.0, observation_window_s=805.0,
        gain_in_window_mhz=6.50)     # boom_v2's measured retime gain
    assert stop is False


def test_high_alpha_designs_face_a_stiffer_hurdle():
    """mini-ISP shape: alpha 97 makes idle wall much more expensive than it is
    on a deep design, which is why the penalty term dominates there."""
    h_hi = marginal_hurdle_mhz(97.07, 1587.0)
    h_lo = marginal_hurdle_mhz(3.03, 1587.0)
    assert h_hi > 30 * h_lo
    stop, _ = should_stop_for_wall(
        alpha_mhz=97.07, remaining_s=1587.0,
        since_last_gain_s=1587.0, observation_window_s=112.0,
        gain_in_window_mhz=0.0)
    assert stop is True
