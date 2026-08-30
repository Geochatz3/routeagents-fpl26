"""Execute scripts/winner_polish.tcl under tclsh with stubbed Vivado commands.

The script's never-worse contract rests entirely on its two timing probes.
An empty `get_timing_paths -quiet` used to read as 0.0 ns setup / 99.0 ns
hold, which beats any negative baseline and passes the hold guard
vacuously -- the wrapper then replaces the SCORED artifact on a verdict no
measurement supports. These tests drive the ladder through tclsh so the
verdict text itself is the assertion.
"""
import os
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLISH_TCL = os.path.join(REPO, "scripts", "winner_polish.tcl")

pytestmark = pytest.mark.skipif(shutil.which("tclsh") is None,
                                reason="tclsh not installed")

# Stubs for the Vivado commands the script calls. Timing paths are served
# from two queues; the literal EMPTY stands for a query that came back with
# no paths. `get_property SLACK <obj>` echoes the object, so a queue entry
# is both the path handle and its slack.
PRELUDE = """\
set ::setup_q {%(setup)s}
set ::hold_q {%(hold)s}

proc get_clocks {args} { return {} }

proc get_timing_paths {args} {
    if {[lsearch -exact $args "-hold"] >= 0} {
        upvar #0 ::hold_q q
    } else {
        upvar #0 ::setup_q q
    }
    if {[llength $q] == 0} { return {} }
    set v [lindex $q 0]
    set q [lrange $q 1 end]
    if {$v eq "EMPTY"} { return {} }
    return [list $v]
}

proc get_property {prop obj} { return $obj }

proc report_route_status {args} {
    return "# of routable nets : 100\\n# of fully routed nets : 100\\n\\
# of nets with routing errors : 0"
}

proc open_checkpoint {args} {}
proc phys_opt_design {args} {}
proc write_checkpoint {args} { puts "STUB_WROTE=[lindex $args end]" }

source %(script)s
"""


def run_polish(tmp_path, setup, hold, max_passes=4):
    """Source winner_polish.tcl under tclsh against stubbed timing queues."""
    harness = tmp_path / "harness.tcl"
    harness.write_text(PRELUDE % {
        "setup": " ".join(str(v) for v in setup),
        "hold": " ".join(str(v) for v in hold),
        "script": POLISH_TCL,
    })
    p = subprocess.run(
        ["tclsh", str(harness), "/tmp/in.dcp", "/tmp/out.dcp",
         "AggressiveExplore", str(max_passes), "0"],
        capture_output=True, text=True, cwd=REPO)
    assert p.returncode == 0, p.stderr
    return p.stdout


def verdict_of(out):
    for line in out.splitlines():
        if line.startswith("POLISH_VERDICT="):
            return line.split("=", 1)[1]
    raise AssertionError(f"no POLISH_VERDICT in output:\n{out}")


class TestEmptyTimingQuery:
    def test_empty_setup_query_is_not_an_improvement(self, tmp_path):
        # Baseline -0.5 ns, then a pass whose setup query comes back empty.
        # Reading that as 0.0 would publish IMPROVED delta=0.5.
        out = run_polish(tmp_path, setup=[-0.5, "EMPTY"], hold=[-0.1])
        assert verdict_of(out).startswith("NO_GAIN")
        assert "STUB_WROTE" not in out
        assert "unmeasurable" in out

    def test_empty_hold_query_does_not_pass_the_hold_guard(self, tmp_path):
        # Setup improved, but the hold probe came back empty. Reading that
        # as 99.0 ns would clear the guard on no measurement at all.
        out = run_polish(tmp_path, setup=[-0.5, -0.2], hold=[0.05, "EMPTY"])
        assert verdict_of(out).startswith("NO_GAIN")
        assert "STUB_WROTE" not in out
        assert "hold_unmeasurable" in out

    def test_unreadable_baseline_setup_is_invalid(self, tmp_path):
        out = run_polish(tmp_path, setup=["EMPTY"], hold=[0.05])
        assert verdict_of(out) == "INVALID reason=no_baseline_setup_timing"

    def test_unreadable_baseline_hold_is_invalid(self, tmp_path):
        out = run_polish(tmp_path, setup=[-0.5], hold=["EMPTY"])
        assert verdict_of(out) == "INVALID reason=no_baseline_hold_timing"


class TestLadderStillWorks:
    def test_real_improvement_writes_and_reports_the_delta(self, tmp_path):
        out = run_polish(tmp_path, setup=[-0.5, -0.2, -0.2],
                         hold=[0.05, 0.05, 0.05])
        assert verdict_of(out) == "IMPROVED delta=0.3"
        assert "STUB_WROTE=/tmp/out.dcp" in out

    def test_no_setup_gain_writes_nothing(self, tmp_path):
        out = run_polish(tmp_path, setup=[-0.5, -0.6], hold=[0.05, 0.05])
        assert verdict_of(out).startswith("NO_GAIN")
        assert "STUB_WROTE" not in out

    def test_hold_regression_is_not_persisted(self, tmp_path):
        # Setup improves but hold goes negative and worse than baseline.
        out = run_polish(tmp_path, setup=[-0.5, -0.2], hold=[0.05, -0.3])
        assert verdict_of(out).startswith("NO_GAIN")
        assert "hold_dirty" in out
        assert "STUB_WROTE" not in out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
