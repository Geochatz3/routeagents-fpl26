"""FPL26_POSITIVE_SLACK_CONTINUE (aug07) — the zero-crossing is not the finish line.

WHY THIS FILE EXISTS. alpha is a DELTA of Fmax, and Fmax = 1000/(T - wns) is
UNCLAMPED in the organizers' own reference implementation
(docs/optimization_example.md). The agent nevertheless declared victory at
wns >= 0 in two places:

    entry  dcp_optimizer.py  initial_wns >= 0 -> ship the input, alpha = 0
    loop   dcp_optimizer.py  best_wns    >= 0 -> exit the LLM loop

Both are unreachable on all 16 corpus designs (every one enters deeply
negative; fir at 0.313 ns is the shallowest), so NO existing test could have
caught this and no A/B could measure it. These tests encode the arithmetic and
the gate directly, because a hidden near-met benchmark is the only thing that
would otherwise report the defect — by scoring us zero.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

dcp_optimizer = importlib.import_module("dcp_optimizer")
positive_slack_continue_enabled = dcp_optimizer.positive_slack_continue_enabled
positive_slack_entry_decision = dcp_optimizer.positive_slack_entry_decision
plateau_armed_from_loop_start = dcp_optimizer.plateau_armed_from_loop_start


# --------------------------------------------------------------------------
# 0. THE DECISIONS THEMSELVES — real calls, mutation-live.
#
# The aug07 review (F4) mutated the armed branch to `return True` and every
# assertion in the first version of this file still passed: they only read
# source text. The logic now lives in two pure functions that these tests
# actually execute.
# --------------------------------------------------------------------------

class TestEntryDecision:
    def test_negative_slack_is_the_normal_path_either_way(self):
        for flag in (True, False):
            assert positive_slack_entry_decision(-0.001, flag) == "not_met"
            assert positive_slack_entry_decision(-7.75, flag) == "not_met"

    def test_met_timing_ships_the_input_when_the_flag_is_off(self):
        assert positive_slack_entry_decision(0.0, False) == "early_exit"
        assert positive_slack_entry_decision(+0.2, False) == "early_exit"

    def test_met_timing_keeps_optimizing_when_armed(self):
        assert positive_slack_entry_decision(0.0, True) == "optimize"
        assert positive_slack_entry_decision(+0.2, True) == "optimize"

    def test_exactly_zero_counts_as_met(self):
        # WNS == 0.000 is 1000/T exactly; every ps above it is still alpha.
        assert positive_slack_entry_decision(0.0, True) == "optimize"

    def test_unmeasured_or_unparseable_slack_never_triggers(self):
        for bad in (None, "", "n/a", float("nan")):
            if bad != bad:  # NaN: comparisons are False, so not_met is correct
                assert positive_slack_entry_decision(bad, True) == "optimize"
                continue
            assert positive_slack_entry_decision(bad, True) == "not_met"


class TestPlateauArming:
    """Review F3: on a met-timing ENTRY nothing improves off a positive
    baseline, so last_improvement_iter stays 0 and the >=3-iteration plateau
    exit is structurally dead unless it is armed from loop start."""

    def test_met_entry_arms_the_plateau_exit(self):
        assert plateau_armed_from_loop_start(
            plateau_preloop_fix=False, initial_wns=+0.1, best_wns=+0.1,
            positive_slack_entry_met=True) is True

    def test_normal_run_is_unaffected(self):
        # The pre-existing FPL26_PLATEAU_PRELOOP_FIX branch, unchanged.
        assert plateau_armed_from_loop_start(False, -1.0, -0.9, False) is False
        assert plateau_armed_from_loop_start(True, -1.0, -0.9, False) is True
        assert plateau_armed_from_loop_start(True, -1.0, -1.0, False) is False

    def test_missing_measurements_do_not_arm_it(self):
        assert plateau_armed_from_loop_start(True, None, -0.9, False) is False
        assert plateau_armed_from_loop_start(True, -1.0, None, False) is False


# --------------------------------------------------------------------------
# 1. The premise: positive slack is worth alpha, in OUR code and in THEIRS.
# --------------------------------------------------------------------------

class _FmaxOnly:
    """calculate_fmax is a pure function of (wns, T); bind it without a run."""
    calculate_fmax = dcp_optimizer.DCPOptimizerBase.calculate_fmax


def _fmax(wns, period):
    return _FmaxOnly().calculate_fmax(wns, period)


def _organizer_fmax(wns, period):
    """docs/optimization_example.md, verbatim: no sign test, no clamp."""
    return 1000.0 / (period - wns)


def test_fmax_is_not_clamped_at_met_timing():
    T = 1.570
    at_zero = _fmax(0.0, T)
    assert at_zero == pytest.approx(1000.0 / T, abs=1e-9)
    # Every ps of POSITIVE slack keeps paying.
    assert _fmax(0.200, T) > at_zero
    assert _fmax(0.200, T) - at_zero == pytest.approx(93.0, abs=1.0)


def test_our_fmax_matches_the_organizers_reference_for_positive_slack():
    for T in (1.570, 2.5, 10.0):
        for wns in (0.0, 0.05, 0.2, 0.5):
            assert _fmax(wns, T) == pytest.approx(_organizer_fmax(wns, T),
                                                  rel=1e-12)


def test_fmax_is_none_when_slack_swallows_the_whole_period():
    # wns >= T leaves a non-positive achievable period: no finite Fmax.
    assert _fmax(1.570, 1.570) is None
    assert _fmax(2.0, 1.570) is None


# --------------------------------------------------------------------------
# 2. The flag: DEFAULT OFF, kill switch wins, armed on the ship path.
# --------------------------------------------------------------------------

def test_flag_is_default_off(monkeypatch):
    monkeypatch.delenv("FPL26_POSITIVE_SLACK_CONTINUE", raising=False)
    monkeypatch.delenv("FPL26_NO_POSITIVE_SLACK_CONTINUE", raising=False)
    assert positive_slack_continue_enabled() is False


@pytest.mark.parametrize("truthy", ["1", "true", "on", "yes", "YES", " On "])
def test_flag_arms_on_every_house_truthy(monkeypatch, truthy):
    monkeypatch.delenv("FPL26_NO_POSITIVE_SLACK_CONTINUE", raising=False)
    monkeypatch.setenv("FPL26_POSITIVE_SLACK_CONTINUE", truthy)
    assert positive_slack_continue_enabled() is True


@pytest.mark.parametrize("falsy", ["0", "", "off", "no", "banana"])
def test_flag_stays_off_on_anything_else(monkeypatch, falsy):
    monkeypatch.delenv("FPL26_NO_POSITIVE_SLACK_CONTINUE", raising=False)
    monkeypatch.setenv("FPL26_POSITIVE_SLACK_CONTINUE", falsy)
    assert positive_slack_continue_enabled() is False


def test_kill_switch_beats_the_master_flag(monkeypatch):
    monkeypatch.setenv("FPL26_POSITIVE_SLACK_CONTINUE", "1")
    monkeypatch.setenv("FPL26_NO_POSITIVE_SLACK_CONTINUE", "1")
    assert positive_slack_continue_enabled() is False


def test_makefile_arms_it_on_the_ship_path():
    """The eval box sets no FPL26_* itself — the Makefile is the ship surface."""
    mk = (ROOT / "Makefile").read_text(errors="replace")
    assert "FPL26_POSITIVE_SLACK_CONTINUE=$(if $(POSITIVE_SLACK)," in mk, (
        "the flag is not injected by the run_optimizer target — it would be "
        "DEFAULT OFF on the only run that is scored (jul29 ship-path drift)")


# --------------------------------------------------------------------------
# 3. The two gates, read straight out of the source.
#
# These are source assertions, not behavioural ones: both sites live inside a
# ~1500-line async method that cannot be driven without a live Vivado session,
# and a mocked stand-in would test the mock. What CAN be pinned exactly is that
# neither `return` is reachable while the flag is armed.
# --------------------------------------------------------------------------

SRC = (ROOT / "dcp_optimizer.py").read_text(errors="replace")


def test_entry_exit_is_wired_to_the_tested_decision():
    assert "_met_decision = positive_slack_entry_decision(" in SRC, (
        "the entry site no longer calls the decision function these tests cover")
    assert 'if _met_decision == "optimize":' in SRC
    assert 'elif _met_decision == "early_exit":' in SRC
    # The original early-exit body must survive as the OFF branch.
    assert "✓ Design already meets timing! No optimization needed.\\n" in SRC
    # ...and the armed branch must arm the plateau exit (review F3).
    assert "self._positive_slack_entry_met = True" in SRC
    assert "self._pre_loop_best_banked = plateau_armed_from_loop_start(" in SRC


def test_force_continue_message_does_not_lie_at_positive_slack():
    """Review F7: this prompt asserted 'Timing is NOT met yet' to the LLM on
    the exact class where WNS is positive — checkable against its own tools."""
    assert "if self.best_wns >= 0 else" in SRC
    assert "Timing is MET, but Fmax = 1000/(T - WNS) is NOT capped" in SRC


def test_loop_exit_is_conditioned_on_the_flag():
    assert ("if self.best_wns >= 0 and not positive_slack_continue_enabled():"
            in SRC), "the zero-crossing loop exit no longer consults the flag"
    assert SRC.count("if self.best_wns >= 0:") == 1, (
        "an UNGUARDED `best_wns >= 0` stop reappeared; the only permitted one "
        "is the PODF skip, which merely declines a stage")


def test_the_other_termination_guards_are_still_present():
    """Retiring the timing-met stop must not leave the loop unbounded."""
    for guard in ("self._llm_cost_breached()",          # LLM $ ceiling
                  "iters_since_improve >= 3",           # plateau
                  "self._consecutive_empty_iters >= 5"):  # empty spin
        assert guard in SRC, f"termination guard vanished: {guard}"


def test_stale_clamp_docstring_is_gone():
    """The false 'fmax = 1/clock_period when WNS >= 0' claim is what made
    stopping at zero look free."""
    assert "fmax = 1 / clock_period when WNS >= 0" not in SRC
    assert "fmax = 1000 / (clock_period - WNS), for EVERY sign of WNS." in SRC


# --------------------------------------------------------------------------
# 5. NON-FINITE WNS FAILS OFF (aug07 code panel, confidence 5).
#
# recipe_pass_band's tail is an unconditional `return "mid"`, which made it the ONE
# band function that failed OPEN on malformed timing data. best_wns is initialised
# to float("-inf") in this class, so -inf is REACHABLE, not hypothetical.
# --------------------------------------------------------------------------

def test_non_finite_wns_arms_no_band():
    band = dcp_optimizer.recipe_pass_band
    for bad in (float("nan"), float("-inf"), float("inf")):
        assert band(bad) is None, bad


def test_finite_wns_still_bands_exactly_as_before():
    band = dcp_optimizer.recipe_pass_band
    assert band(-1.0) == "shallow"
    assert band(-1.05) == "shallow"
    assert band(-2.0) == "mid"
    assert band(-8.0) == "deep"
    assert band(-19.162) == "deep"
    assert band(0.5) is None          # met timing: out of scope for bands
    assert band(None) is None
    assert band("junk") is None
