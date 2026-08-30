"""Stops an attempt when additional wall time is unlikely to offset its score
penalty.

The break-even test follows score = alpha * (1 - 0.1 * (beta + gamma)), where
gamma is wall time in hours; the 0.1 factor is fixed by the scoring formula
rather than fitted. The improvement rate is estimated from the attempt's
realized history over the cheapest completed heavy move. Unknown gain, an
unknown observation window, or insufficient elapsed evidence always keeps the
attempt running, preventing missing data from discarding a possible gain.
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
    """Return the minimum Fmax gain required to offset a wall-time horizon.

    The result is in MHz, and the horizon is converted from seconds to gamma
    hours using score = alpha * (1 - 0.1 * (beta + gamma)).
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
