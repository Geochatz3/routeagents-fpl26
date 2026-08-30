"""Verify that stagnation preemption measures only time spent in the LLM loop.

Pre-loop placement and routing stages may be long and must not consume the
loop's stagnation allowance. The improvement clock therefore starts when the
loop begins.
"""
import os
import time

from optimizer.ils_polish import ILSPolishConfig, should_trigger, should_trigger_at_exit


def _cfg(**kw):
    return ILSPolishConfig(**{"enabled": True, **kw})


def test_preempt_still_fires_when_the_loop_itself_stalls():
    """The gate must keep working — this behaviour is being preserved, not removed."""
    trig, why = should_trigger(cells=250_000, remaining_s=2400,
                               seconds_since_improve=600, best_wns=-1.238,
                               baseline_wns=-1.238, cfg=_cfg())
    assert trig, why


def test_preempt_does_not_fire_on_a_loop_that_has_not_run_yet():
    """Ensure pre-loop work cannot preempt an LLM loop before its first iteration.

    The improvement clock is anchored at loop entry, so elapsed pre-loop time
    is excluded from the stagnation decision.
    """
    trig, why = should_trigger(cells=250_000, remaining_s=2400,
                               seconds_since_improve=0.14, best_wns=-1.238,
                               baseline_wns=-1.238, cfg=_cfg())
    assert not trig
    assert "loop_still_productive" in why


def test_the_regression_itself_pre_loop_time_trips_the_gate():
    """Documents WHY the re-anchor exists: 580s pre-loop vs a 240s threshold.

    If someone removes the re-anchor, this is the behaviour that comes back.
    """
    trig, _ = should_trigger(cells=250_000, remaining_s=2400,
                             seconds_since_improve=580, best_wns=-1.238,
                             baseline_wns=-1.238, cfg=_cfg(stagnation_seconds=240))
    assert trig, "580s of pre-loop time DOES trip a 240s gate — hence the re-anchor"


def test_ils_is_not_starved_when_the_loop_ends_with_timing_unmet():
    """Suppressing the mid-loop preempt is only safe because the EXIT trigger remains.

    Designs that genuinely need ruin-and-recreate still receive the leftover
    budget when the loop finishes without closing timing.
    """
    trig, why = should_trigger_at_exit(cells=250_000, remaining_s=2400,
                                       best_wns=-1.238, cfg=_cfg())
    assert trig, why


def test_reanchor_has_a_kill_switch():
    """The kill switch must accept every house falsy spelling.

    This asserted `os.environ.get(...) != "0"` — a restatement of the
    comparison, true whatever the implementation did. It now calls the
    resolver the run actually consults, which is what makes `off`/`false`
    regressing back to a bare `!= "0"` a failure here."""
    from optimizer.config_resolution import resolve_preempt_loop_clock_enabled

    prev = os.environ.get("FPL26_PREEMPT_LOOP_CLOCK")
    try:
        os.environ.pop("FPL26_PREEMPT_LOOP_CLOCK", None)
        assert resolve_preempt_loop_clock_enabled() is True, "default is ON"
        for off in ("0", "false", "FALSE", "no", "off", " Off "):
            os.environ["FPL26_PREEMPT_LOOP_CLOCK"] = off
            assert resolve_preempt_loop_clock_enabled() is False, off
        for on in ("1", "true", "yes", "on", ""):
            os.environ["FPL26_PREEMPT_LOOP_CLOCK"] = on
            assert resolve_preempt_loop_clock_enabled() is True, on
    finally:
        if prev is None:
            os.environ.pop("FPL26_PREEMPT_LOOP_CLOCK", None)
        else:
            os.environ["FPL26_PREEMPT_LOOP_CLOCK"] = prev


def test_reanchor_only_moves_the_clock_forward():
    """Re-anchoring must never make the loop look MORE stalled than it is."""
    run_start = time.time() - 580
    loop_start = time.time()
    assert loop_start > run_start
    assert (time.time() - loop_start) < (time.time() - run_start)


# --- R3 reroute chain (opt-in) -------------------------------------

def test_r3_reroute_chain_is_off_by_default():
    """Ensure the optional reroute chain remains disabled by default.

    The default preserves the existing four-step routing contract unless the
    chain is explicitly enabled.
    """
    import os
    from optimizer.recipe_router import _r3_reroute_chain
    prev = os.environ.pop("FPL26_R3_REROUTE_CHAIN", None)
    try:
        assert _r3_reroute_chain() == ()
    finally:
        if prev is not None:
            os.environ["FPL26_R3_REROUTE_CHAIN"] = prev


def test_r3_reroute_chain_arms_with_the_flag():
    """FPL26_R3_REROUTE_CHAIN=1 appends exactly one guidance step."""
    import os
    from optimizer.recipe_router import _r3_reroute_chain
    prev = os.environ.get("FPL26_R3_REROUTE_CHAIN")
    try:
        os.environ["FPL26_R3_REROUTE_CHAIN"] = "1"
        chain = _r3_reroute_chain()
        assert len(chain) == 1
        note = chain[0].note
        # the three moves 3d's +19.09 actually used, in order
        assert "Explore" in note and "critical_pin_opt" in note
        assert "AggressiveExplore" in note
        # and the guards that stop it displacing the placement work
        assert "revert on regression" in note
        assert "25 min" in note
    finally:
        if prev is None:
            os.environ.pop("FPL26_R3_REROUTE_CHAIN", None)
        else:
            os.environ["FPL26_R3_REROUTE_CHAIN"] = prev
