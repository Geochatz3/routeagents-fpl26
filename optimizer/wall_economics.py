"""Intra-attempt wall economics — stop paying gamma for zero gain.

WHY THIS EXISTS (measured, chain12 jul25, boom_soc through `make run_optimizer`):
the recipe banked wns=-10.256 (84.5666 MHz, alpha +36.33) at 18:09:25 and the
attempt then ran until 18:37 -- 1730 s during which BOTH non-recipe MUX
candidates measured the untouched baseline (-19.162). Zero gain, 1730 s of
gamma:

    stop point       wall      gamma penalty   score at alpha 36.33
    actual         3380 s          9.4%              32.92
    at the bank    ~1650 s         4.6%              34.66

1.74 points paid for 29 minutes of nothing. The same pathology on mini-ISP costs
7.8 points (alpha 97.07 -> score 89.316).

WHAT THE WRAPPER ALREADY DOES, AND WHY IT CANNOT HELP HERE:
scripts/multi_restart_optimize.py stops BETWEEN attempts (should_skip_truncated,
the cost gates, should_stop_early). Those cannot see inside a running attempt,
which is exactly where the 1730 s was spent. Note also PITFALL 5 in
ab_wall_economics.py: gamma is the WHOLE wrapper's wall, so an intra-attempt
stop only converts into score if the wrapper then declines to start another
attempt -- which its existing floor/truncation gates handle.

THE RULE IS DERIVED, NOT FITTED. From the published scoring function
    score = alpha * (1 - 0.1*(beta + gamma)),    gamma = wall_hours
continuing for dt seconds is worth it only when the marginal alpha beats the
marginal penalty:
    d(score)/dt > 0
    alpha'(t) * P  >  alpha(t) * 0.1/3600          (P = current penalty factor)
    alpha'(t)      >  alpha(t) * 0.1 / (3600 * P)
Both constants come from the contest's own formula. There is NO tuned threshold
here, which matters: the methodology invariant forbids fitting decision
boundaries to outcomes, and three panels have flagged it.

THE ONE JUDGEMENT CALL is estimating alpha'(t). We use the run's OWN realized
history: if a full heavy move's worth of wall has passed with no gain, the
maximum-likelihood forward rate is 0, which loses to any positive hurdle. The
observation window is itself a MEASURED quantity (the cheapest completed heavy
move this run), not a tuned constant -- so the design stays free of fitted
parameters.

Fails SAFE in every unmeasurable case: unknown alpha, unknown window, or no
elapsed evidence all return "keep going". Stopping early on bad data would
forfeit real gain; continuing merely costs gamma we were already spending.
"""
from __future__ import annotations

from typing import Optional, Tuple

# From the contest scoring function, not tuned here.
PENALTY_PER_HOUR = 0.1
SECONDS_PER_HOUR = 3600.0

# Floor on the penalty factor used in the hurdle denominator. P can only shrink
# the hurdle; clamping it stops a pathological beta+gamma from making the rule
# arbitrarily eager. 0.5 corresponds to beta+gamma = 5.0, far outside anything
# the eval can produce (1 h wall, $1 budget => ~1.1).
MIN_PENALTY_FACTOR = 0.5


def marginal_hurdle_mhz(alpha_mhz: float, horizon_s: float,
                        penalty_factor: float = 0.9) -> float:
    """Alpha that must be gained over ``horizon_s`` to break even on gamma.

    Pure arithmetic from score = alpha*(1 - 0.1*(beta+gamma)).
    """
    if alpha_mhz <= 0 or horizon_s <= 0:
        return 0.0
    p = max(float(penalty_factor), MIN_PENALTY_FACTOR)
    return (alpha_mhz * PENALTY_PER_HOUR * (horizon_s / SECONDS_PER_HOUR)) / p


def should_stop_for_wall(
    *,
    alpha_mhz: Optional[float],
    remaining_s: float,
    since_last_gain_s: Optional[float],
    observation_window_s: Optional[float],
    gain_in_window_mhz: float = 0.0,
    penalty_factor: float = 0.9,
) -> Tuple[bool, str]:
    """Should this attempt finalize NOW rather than spend ``remaining_s``?

    ``observation_window_s`` should be the cheapest COMPLETED heavy move this
    run (a measurement). Returns ``(stop, reason)`` and fails safe -- any
    unmeasurable input yields ``(False, ...)``.
    """
    if alpha_mhz is None or alpha_mhz <= 0:
        return False, ("nothing banked yet (alpha=%s) — stopping could only "
                       "forfeit gain" % alpha_mhz)
    if remaining_s <= 0:
        return False, "no remaining wall to save"
    if since_last_gain_s is None or observation_window_s is None:
        return False, ("gain history unmeasurable (since_last_gain=%s "
                       "window=%s) — fail open"
                       % (since_last_gain_s, observation_window_s))
    if observation_window_s <= 0:
        return False, "no completed heavy move to size the window — fail open"
    if since_last_gain_s < observation_window_s:
        return False, (f"only {since_last_gain_s:.0f}s since the last gain, "
                       f"less than one heavy move ({observation_window_s:.0f}s)"
                       f" — not yet a fair chance")

    hurdle = marginal_hurdle_mhz(alpha_mhz, remaining_s, penalty_factor)
    # Forward rate estimated from realized history over the observed window.
    rate = max(0.0, gain_in_window_mhz) / since_last_gain_s
    expected = rate * remaining_s
    if expected >= hurdle:
        return False, (f"expected {expected:.3f} MHz over {remaining_s:.0f}s "
                       f">= hurdle {hurdle:.3f} MHz — keep going")
    return True, (f"STOP: {since_last_gain_s:.0f}s since last gain "
                  f"(window {observation_window_s:.0f}s), realized "
                  f"{gain_in_window_mhz:.3f} MHz => expected "
                  f"{expected:.3f} MHz over the remaining {remaining_s:.0f}s, "
                  f"below the gamma hurdle {hurdle:.3f} MHz "
                  f"(alpha {alpha_mhz:.2f} x 0.1 x {remaining_s/3600:.3f} h "
                  f"/ P {max(penalty_factor, MIN_PENALTY_FACTOR):.2f}). "
                  f"Finalizing saves {alpha_mhz * PENALTY_PER_HOUR * (remaining_s/3600):.2f} "
                  f"score points.")
