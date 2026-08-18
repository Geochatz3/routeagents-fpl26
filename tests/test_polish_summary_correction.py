"""The summary must describe the artifact that SHIPPED, not the one before polish.

`winner_polish` replaces the scored DCP after the agent has already printed its
summary block. On a POLISH_VERDICT=IMPROVED that block understates what shipped.
No contest score is lost — the better DCP is what gets scored — but every A/B we
run parses that block, so six rows of the jul28/29 corpus were read low, logicnets
by 4.50 MHz. That is above the 3.5 MHz noise floor and it turned a win into what
was recorded as the cohort's only loss.

These tests pin the correction arithmetic and the fact that the corrected values
are emitted with the SAME labels our drivers grep, so a `tail -1` finds them.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "mr_polish_corr", ROOT / "scripts" / "multi_restart_optimize.py")
mr = importlib.util.module_from_spec(_spec)
sys.modules["mr_polish_corr"] = mr
_spec.loader.exec_module(mr)


# (logged fmax MHz, logged polish delta ns, artifact-measured fmax MHz) — the two runs
# whose shipped DCP was re-opened in Vivado on jul29 (`report_route_status`). Every input
# is logged; none is back-solved from the answer. Kept identical to the corpus corrector's
# ARTIFACT_TRUTH in final_round/tools/tests/test_polish_correct.py, so the in-run
# correction and the after-the-fact one cannot drift apart.
ARTIFACT_TRUTH = [
    (498.01, 0.018000000000000016, 502.5125628),
    (79.18, 0.0009999999999994458, 79.18910357),
]


@pytest.mark.parametrize("logged,delta,measured", ARTIFACT_TRUTH)
def test_correction_matches_the_re_opened_artifacts(logged, delta, measured):
    """Within 0.01 MHz of what Vivado reported for the DCP that actually shipped."""
    assert mr.polish_corrected_fmax(logged, delta) == pytest.approx(measured, abs=0.01)


def test_correction_is_an_increase_and_uses_the_period_identity():
    """fmax = 1000/(T+|WNS|); removing delta ns of WNS must raise fmax."""
    base = 250.0
    corrected = mr.polish_corrected_fmax(base, 0.05)
    assert corrected > base
    # Round trip: going back by the same delta returns the original.
    total = 1000.0 / corrected
    assert 1000.0 / (total + 0.05) == pytest.approx(base, abs=1e-9)


@pytest.mark.parametrize("fmax,delta", [
    (None, 0.02),      # no logged fmax
    (0.0, 0.02),       # degenerate
    (250.0, None),     # polish reported no delta
    (250.0, 0.0),      # NO_GAIN must not be reported as a correction
    (250.0, -0.01),    # a regression is not a polish gain
    (250.0, 4.0),      # delta exceeds the whole period -> underivable
])
def test_refuses_to_invent_a_number(fmax, delta):
    """Return None rather than print a value that was not derived."""
    assert mr.polish_corrected_fmax(fmax, delta) is None


def test_winner_polish_reports_the_delta(tmp_path, monkeypatch):
    """The report dict is how run() learns the WNS gain; without it there is
    nothing to correct with."""
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts" / "winner_polish.tcl").write_text("# stub")
    final = tmp_path / "design_optimized.dcp"
    final.write_bytes(b"PK\x03\x04original")

    monkeypatch.setattr(mr, "_resolve_vivado", lambda: "/usr/bin/true")

    class _P:
        stdout = ("# POLISH_VERDICT= usage echo line\n"
                  "POLISH_VERDICT=IMPROVED delta=0.018\n")
        stderr = ""

    def _fake_run(*a, **k):
        # The polish writes its candidate to the staged /tmp path (argv[-3]).
        Path(a[0][a[0].index("-tclargs") + 2]).write_bytes(b"PK\x03\x04polished")
        return _P()

    monkeypatch.setattr(mr.subprocess, "run", _fake_run)

    report: dict = {}
    assert mr.winner_polish(final, 900.0, tmp_path, report=report) is True
    assert report["delta_ns"] == pytest.approx(0.018)
    # And the signature stays backwards compatible for callers that don't care.
    assert mr.winner_polish(final, 900.0, tmp_path) is True


def test_corrected_fmax_and_improvement_are_emitted_together(capsys, tmp_path,
                                                             monkeypatch):
    """Never a corrected Fmax beside a stale improvement — a parser would read
    that pair as consistent."""
    calls = {}

    def _fake_polish(final_output, remaining, repo, report=None):
        if report is not None:
            report["delta_ns"] = 0.018
        return True

    monkeypatch.setattr(mr, "winner_polish", _fake_polish)
    monkeypatch.setattr(mr, "_atomic_publish", lambda a, b: True)
    monkeypatch.setattr(mr, "select_best", lambda att: calls["best"])

    for initial, expect_labels in ((250.0, True), (None, False)):
        calls["best"] = {"i": 1, "fmax": 498.00796812749, "output": "/tmp/x.dcp",
                         "initial_fmax": initial}
        mr.run(tmp_path / "in.dcp", tmp_path / "out.dcp", total_wall=10.0,
               attempt_floor=1e9, max_attempts=0, repo=tmp_path, polish=True)
        out = capsys.readouterr().out
        assert ("Best Fmax:" in out) is expect_labels, out
        assert ("Fmax Improvement:" in out) is expect_labels, out
        if not expect_labels:
            assert "cannot be restated consistently" in out, out


def test_no_delta_reported_when_polish_did_not_improve(tmp_path, monkeypatch):
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts" / "winner_polish.tcl").write_text("# stub")
    final = tmp_path / "design_optimized.dcp"
    final.write_bytes(b"PK\x03\x04original")
    monkeypatch.setattr(mr, "_resolve_vivado", lambda: "/usr/bin/true")

    class _P:
        stdout = "POLISH_VERDICT=NO_GAIN\n"
        stderr = ""

    monkeypatch.setattr(mr.subprocess, "run", lambda *a, **k: _P())
    report: dict = {}
    assert mr.winner_polish(final, 900.0, tmp_path, report=report) is False
    assert "delta_ns" not in report, (
        "a NO_GAIN polish must not report a delta — 'no polish happened' and "
        "'polish gained nothing' are different facts")
