"""Tests for the XDC / timing-constraint integrity guard.

Editing timing constraints is DISQUALIFYING. Two properties matter most:
  * read-only timing analysis must NEVER be blocked (a false positive here
    would break ordinary optimisation);
  * an unmeasurable fingerprint must report UNVERIFIED, never "unchanged".
"""
import pytest

from optimizer.constraint_guard import (
    build_fingerprint,
    deny_reason,
    is_constraint_mutating,
    parse_clock_report,
    parse_exception_report,
)

CLOCKS_BEFORE = """
Clock              Period  Waveform      Attributes
clk_fpl26contest    2.000   0.000 1.000  P
"""
EXC_BEFORE = ""
EXC_AFTER_CHEAT = """
set_false_path -from [get_pins a/CLK] -to [get_pins b/D]
"""


# ------------------------------------------------- prevention (deny-list)
@pytest.mark.parametrize("cmd", [
    "set_false_path -from [get_pins x/C] -to [get_pins y/D]",
    "set_multicycle_path 2 -setup -from [get_clocks clk_fpl26contest]",
    "set_clock_uncertainty 0.0 [get_clocks clk_fpl26contest]",
    "create_clock -period 4.0 -name clk_fpl26contest [get_ports clk]",
    "set_max_delay 5.0 -from [get_pins a/C]",
    "set_disable_timing [get_cells foo]",
    "remove_clock [get_clocks clk_fpl26contest]",
    "read_xdc /tmp/cheat.xdc",
])
def test_constraint_mutating_commands_are_caught(cmd):
    hit, which = is_constraint_mutating(cmd)
    assert hit is True and which


@pytest.mark.parametrize("cmd", [
    "report_timing_summary",
    "report_clocks",
    "get_clocks -quiet clk_fpl26contest",
    "report_exceptions",
    "get_timing_paths -max_paths 1 -setup",
    "place_design -directive Explore",
    "route_design -directive AggressiveExplore",
    "phys_opt_design -directive AlternateFlowWithRetiming",
    "report_route_status -return_string",
    "write_checkpoint -force {/tmp/x.dcp}",
])
def test_readonly_and_physical_commands_are_allowed(cmd):
    hit, _ = is_constraint_mutating(cmd)
    assert hit is False, f"{cmd!r} must not be blocked"


def test_word_boundary_prevents_false_positives():
    # 'report_clocks' contains 'clocks' but must not trip 'create_clock'
    assert is_constraint_mutating("report_clocks")[0] is False
    # a variable named like a command is not a command
    assert is_constraint_mutating("set my_set_false_path_note 1")[0] is False


def test_commented_out_command_is_not_a_command():
    assert is_constraint_mutating("# set_false_path -from a -to b")[0] is False


def test_multiline_payload_is_scanned():
    cmd = "place_design -directive Explore\nset_false_path -from a -to b"
    hit, which = is_constraint_mutating(cmd)
    assert hit is True and which == "set_false_path"


def test_deny_reason_explains_the_legal_alternative():
    msg = deny_reason("set_false_path")
    assert "DISQUALIFYING" in msg
    assert "physically" in msg or "physical" in msg


# ------------------------------------------------------ detection (diff)
def test_identical_constraints_do_not_trip():
    a = build_fingerprint(CLOCKS_BEFORE, EXC_BEFORE)
    b = build_fingerprint(CLOCKS_BEFORE, EXC_BEFORE)
    changed, why = a.differs_from(b)
    assert changed is False and "unchanged" in why


def test_added_false_path_is_detected():
    before = build_fingerprint(CLOCKS_BEFORE, EXC_BEFORE)
    after = build_fingerprint(CLOCKS_BEFORE, EXC_AFTER_CHEAT)
    changed, why = after and before.differs_from(after)
    assert changed is True and "exception" in why


def test_changed_clock_period_is_detected():
    slower = CLOCKS_BEFORE.replace("2.000", "4.000")
    before = build_fingerprint(CLOCKS_BEFORE, EXC_BEFORE)
    after = build_fingerprint(slower, EXC_BEFORE)
    changed, why = before.differs_from(after)
    assert changed is True and "clock" in why


def test_removed_clock_is_detected():
    before = build_fingerprint(CLOCKS_BEFORE, EXC_BEFORE)
    after = build_fingerprint("Clock Period Waveform\n", EXC_BEFORE)
    changed, why = before.differs_from(after)
    assert changed is True and "clock count" in why


def test_reordered_reports_are_not_a_false_alarm():
    e1 = "set_false_path -from a -to b\nset_multicycle_path 2 -from c -to d"
    e2 = "set_multicycle_path 2 -from c -to d\nset_false_path -from a -to b"
    a = build_fingerprint(CLOCKS_BEFORE, e1)
    b = build_fingerprint(CLOCKS_BEFORE, e2)
    changed, _ = a.differs_from(b)
    assert changed is False


def test_whitespace_only_difference_is_not_a_false_alarm():
    e1 = "set_false_path -from a  -to b"
    e2 = "set_false_path   -from a -to b"
    a = build_fingerprint(CLOCKS_BEFORE, e1)
    b = build_fingerprint(CLOCKS_BEFORE, e2)
    assert a.differs_from(b)[0] is False


def test_unmeasurable_fingerprint_reports_unverified_not_unchanged():
    good = build_fingerprint(CLOCKS_BEFORE, EXC_BEFORE)
    bad = build_fingerprint(None, None)
    changed, why = good.differs_from(bad)
    # must NOT claim a change (that would brick a run) but must say UNVERIFIED
    assert changed is False
    assert "UNVERIFIED" in why
    assert "unchanged" not in why


# ------------------------------------------------------------- parsing
def test_clock_report_parsing():
    assert parse_clock_report(CLOCKS_BEFORE) == ["clk_fpl26contest 2.000"]


def test_exception_report_parsing_picks_up_exceptions():
    got = parse_exception_report(EXC_AFTER_CHEAT)
    assert len(got) == 1 and "false_path" in got[0]


def test_empty_reports_parse_to_empty():
    assert parse_clock_report("") == []
    assert parse_exception_report(None) == []


# Partial captures are unverified: if either clocks or exceptions is missing,
# the fingerprint must not infer a change from an empty parsed report.

def test_partial_capture_missing_clocks_is_unverified_not_changed():
    """The exact production shape: clocks timed out, exceptions returned."""
    good = build_fingerprint(CLOCKS_BEFORE, EXC_BEFORE)
    partial = build_fingerprint(None, EXC_BEFORE)
    assert partial.captured is False, (
        "a partial fingerprint must never claim to be captured — a missing "
        "report parses to 0 constraints, which reads as 'all deleted'")
    changed, why = good.differs_from(partial)
    assert changed is False, "must NOT raise a false CONSTRAINTS CHANGED alarm"
    assert "UNVERIFIED" in why
    assert "clock count" not in why


def test_partial_capture_missing_exceptions_is_unverified_not_changed():
    good = build_fingerprint(CLOCKS_BEFORE, EXC_BEFORE)
    partial = build_fingerprint(CLOCKS_BEFORE, None)
    assert partial.captured is False
    changed, why = good.differs_from(partial)
    assert changed is False
    assert "UNVERIFIED" in why


def test_partial_capture_note_names_the_missing_report():
    assert "clocks" in build_fingerprint(None, EXC_BEFORE).note
    assert "exceptions" in build_fingerprint(CLOCKS_BEFORE, None).note


def test_a_real_constraint_edit_is_still_detected():
    """The fix must not blunt the guard: full captures still compare."""
    before = build_fingerprint(CLOCKS_BEFORE, EXC_BEFORE)
    after = build_fingerprint(CLOCKS_BEFORE, EXC_AFTER_CHEAT)
    changed, why = before.differs_from(after)
    assert changed is True
    assert "UNVERIFIED" not in why
