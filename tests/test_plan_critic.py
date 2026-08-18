"""Tests for the cheap pre-flight plan critic (jul25 panel, 4/5 seats).

The critic must be (a) advisory-only — never able to block, (b) cheap — gated
on genuine beta headroom, (c) inert when malformed.
"""
import pytest

from optimizer.plan_critic import (
    VERDICT_APPROVE,
    VERDICT_CONCERN,
    VERDICT_UNPARSED,
    build_critic_prompt,
    is_heavy_action,
    parse_critic_verdict,
    should_critique,
)


def _gate(**over):
    kw = dict(enabled=True, tool_name="vivado_place_design", arguments={},
              calls_made=0, max_calls=4, spent_usd=0.30, budget_usd=1.00,
              beta_headroom_frac=0.7, remaining_s=2000.0,
              min_remaining_s=600.0)
    kw.update(over)
    return should_critique(**kw)


# ------------------------------------------------------------------- gating
def test_disabled_is_a_hard_noop():
    run, why = _gate(enabled=False)
    assert run is False and "disabled" in why


def test_boom_soc_profile_arms():
    """boom_soc spent $0.357 of ~$1 — exactly the underuse the panel flagged."""
    run, why = _gate(spent_usd=0.357)
    assert run is True and "armed" in why


def test_light_action_is_not_reviewed():
    run, why = _gate(tool_name="vivado_get_wns")
    assert run is False and "not a heavy action" in why


def test_call_cap_is_respected():
    run, why = _gate(calls_made=4, max_calls=4)
    assert run is False and "cap reached" in why


def test_beta_headroom_exhausted_stops_critiquing():
    # rend3d spent 69% of budget — at frac 0.7 it is essentially done
    run, why = _gate(spent_usd=0.71)
    assert run is False and "headroom exhausted" in why


def test_unknown_budget_fails_closed():
    run, why = _gate(budget_usd=None)
    assert run is False and "fail closed" in why


def test_low_wall_spends_on_vivado_not_on_talk():
    run, why = _gate(remaining_s=300.0)
    assert run is False and "spend it on Vivado" in why


# ------------------------------------------------------------ heavy actions
@pytest.mark.parametrize("tool", ["vivado_place_design", "vivado_route_design",
                                  "vivado_phys_opt_design"])
def test_named_heavy_tools(tool):
    assert is_heavy_action(tool, None) is True


@pytest.mark.parametrize("cmd,expected", [
    ("place_design -directive Explore", True),
    ("route_design -directive AggressiveExplore", True),
    ("phys_opt_design -retime", True),
    ("report_route_status -return_string", False),
    ("get_timing_paths -max_paths 1", False),
])
def test_heavy_detection_inside_raw_tcl(cmd, expected):
    assert is_heavy_action("vivado_run_tcl", {"command": cmd}) is expected


# ----------------------------------------------------------------- parsing
def test_parses_a_well_formed_concern():
    v = parse_critic_verdict(
        "VERDICT: CONCERN\n"
        "WHY: a full re-place discards banked polish with 900s left.\n"
        "BETTER: route_design -directive AggressiveExplore")
    assert v.verdict == VERDICT_CONCERN and v.is_concern
    assert "discards banked polish" in v.why
    assert "AggressiveExplore" in v.better
    note = v.advisory_note()
    assert "advisory" in note and "you may" in note.lower()


def test_parses_approve():
    v = parse_critic_verdict("VERDICT: APPROVE\nWHY: fine.\nBETTER: none")
    assert v.verdict == VERDICT_APPROVE and not v.is_concern
    # 'none' alternative is not surfaced as noise
    assert "alternative" not in v.advisory_note()


@pytest.mark.parametrize("text", [None, "", "the model rambled with no verdict"])
def test_malformed_reply_is_inert(text):
    v = parse_critic_verdict(text)
    assert v.verdict == VERDICT_UNPARSED
    # CRITICAL: an unparseable second opinion must not perturb planner context
    assert v.advisory_note() == ""


def test_advisory_note_never_reads_as_a_veto():
    v = parse_critic_verdict("VERDICT: CONCERN\nWHY: risky.\nBETTER: none")
    note = v.advisory_note().lower()
    for banned in ("do not", "must not", "forbidden", "blocked", "abort"):
        assert banned not in note


# ------------------------------------------------------------------ prompt
def test_prompt_carries_the_decision_relevant_state():
    p = build_critic_prompt(
        tool_name="vivado_run_tcl",
        arguments={"command": "place_design -directive Explore"},
        initial_wns=-19.162, current_wns=-14.675, failing_endpoints=217988,
        router_rule="R1", router_why="DEEP-extreme: retiming is the slow step",
        tried=["phys_opt AlternateFlowWithRetiming", "route_design Default"],
        remaining_s=1800.0)
    for token in ("place_design -directive Explore", "-19.162", "217988",
                  "R1", "DEEP-extreme", "route_design Default", "1800"):
        assert token in p


def test_prompt_truncates_long_tried_lists():
    p = build_critic_prompt(
        tool_name="vivado_place_design", arguments=None,
        initial_wns=-1.0, current_wns=-0.9, failing_endpoints=10,
        router_rule=None, router_why=None,
        tried=[f"action_{i}" for i in range(40)])
    assert "action_39" in p and "action_0\n" not in p
