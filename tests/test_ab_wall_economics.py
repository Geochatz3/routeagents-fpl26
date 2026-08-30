"""Test wall-time attribution, draw counting, and paired-run grading.

Fixtures preserve the summary schema written by the multi-restart optimizer,
including attempt timing, status, cost, output, and selection fields.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.ab_wall_economics import (  # noqa: E402
    MEANINGFUL_FMAX_DELTA_MHZ,
    draw_count,
    grade_pair,
    parse_attempt_wall,
    variant_make_args,
)

# Fixture for a wrapper log in which the second attempt runs.
# It preserves the line structure consumed by the attribution parser.
MINI_ISP_LOG = """\
[multi-restart] attempt 1: budget 3500s -> /tmp/mr_input_3540_1.dcp
[multi-restart] attempt 1 -> fmax=404.20371867421176 status=VALID_OPTIMIZED cost=$0.0 cum_cost=$0.00 exists=True
[multi-restart] refreshed scored output (attempt 1, fmax=404.20371867421176) -> /tmp/fpl26contest-validation/benchmarks/amd_mini-isp_2025.1/input_optimized.dcp
[multi-restart] attempt 2: budget 2285s -> /tmp/mr_input_3540_2.dcp
[multi-restart] attempt 2 -> fmax=404.20371867421176 status=VALID_OPTIMIZED cost=$0.22802504999999998 cum_cost=$0.23 exists=True
[multi-restart] refreshed scored output (attempt 1, fmax=404.20371867421176) -> /tmp/fpl26contest-validation/benchmarks/amd_mini-isp_2025.1/input_optimized.dcp
[multi-restart] stop: strong result confirmed by >=2 attempts after 2 runs
[multi-restart] BEST = attempt 1 fmax=404.20371867421176 -> /tmp/fpl26contest-validation/benchmarks/amd_mini-isp_2025.1/input_optimized.dcp
[multi-restart] winner-polish: 1488s stranded -> phys_opt pass on input_optimized.dcp
[multi-restart] polish: POLISH_VERDICT=NO_GAIN
"""

# Total wrapper wall time includes all attempts, winner polish, and setup.
# Per-attempt attribution must not use this aggregate value.
MINI_ISP_SCORECARD_WALL_S = 2058.17
MINI_ISP_FMAX = 404.20371867421176

# The fixture matches the run-summary schema.
# `elapsed` is agent runtime in seconds and excludes wrapper setup and
# subprocess overhead, so it remains below budget-delta wall time.
MINI_ISP_MR_SUMMARY = {
    "input": "/tmp/mr_input.dcp",
    "final_output": "/tmp/fpl26contest-validation/benchmarks/amd_mini-isp_2025.1/input_optimized.dcp",
    "attempts": [
        {"i": 1, "fmax": MINI_ISP_FMAX, "status": "VALID_OPTIMIZED",
         "cost": 0.0, "elapsed": 1198.4, "output": "/tmp/mr_input_3540_1.dcp",
         "exists": True, "run_dir": "/repo/dcp_optimizer_run-1"},
        {"i": 2, "fmax": MINI_ISP_FMAX, "status": "VALID_OPTIMIZED",
         "cost": 0.22802504999999998, "elapsed": 785.2,
         "output": "/tmp/mr_input_3540_2.dcp",
         "exists": True, "run_dir": "/repo/dcp_optimizer_run-2"},
    ],
    "chosen": {"i": 1, "fmax": MINI_ISP_FMAX, "status": "VALID_OPTIMIZED",
               "cost": 0.0, "elapsed": 1198.4,
               "output": "/tmp/mr_input_3540_1.dcp", "exists": True,
               "run_dir": "/repo/dcp_optimizer_run-1"},
    "total_wall": 3500.0,
}

# Fixture for a wrapper log in which the floor gate suppresses attempt 2.
BOOM_BETA_LOG = """\
[multi-restart] attempt 1: budget 3500s -> /tmp/mr_input_148356_1.dcp
[multi-restart] attempt 1 -> fmax=77.15454054471107 status=VALID_FALLBACK_BASELINE cost=$0.06782515 cum_cost=$0.07 exists=True
[multi-restart] refreshed scored output (attempt 1, fmax=77.15454054471107) -> /tmp/fpl26contest-validation/benchmarks/boom_soc_2025.1_v2/input_optimized.dcp
[multi-restart] stop: remaining 286s < floor 1200s
[multi-restart] BEST = attempt 1 fmax=77.15454054471107 -> /tmp/fpl26contest-validation/benchmarks/boom_soc_2025.1_v2/input_optimized.dcp
"""

BOOMLEG_LOG = """\
[multi-restart] attempt 1: budget 3500s -> /tmp/mr_boom_soc_2025.1_v2_4082_1.dcp
[multi-restart] attempt 1 -> fmax=79.91688643810436 status=VALID_OPTIMIZED cost=$0.061444 cum_cost=$0.06 exists=True
[multi-restart] refreshed scored output (attempt 1, fmax=79.91688643810436) -> /tmp/fpl26contest-validation/benchmarks/boom_soc_2025.1_v2_optimized.dcp
[multi-restart] attempt 2: budget 1542s -> /tmp/mr_boom_soc_2025.1_v2_4082_2.dcp
[multi-restart] attempt 2 -> fmax=77.15454054471107 status=VALID_FALLBACK_BASELINE cost=$0.04087455 cum_cost=$0.10 exists=True
[multi-restart] refreshed scored output (attempt 1, fmax=79.91688643810436) -> /tmp/fpl26contest-validation/benchmarks/boom_soc_2025.1_v2_optimized.dcp
[multi-restart] stop: remaining 287s < floor 1200s
[multi-restart] BEST = attempt 1 fmax=79.91688643810436 -> /tmp/fpl26contest-validation/benchmarks/boom_soc_2025.1_v2_optimized.dcp
"""

# A split-aware cap replaces the normal remaining-budget value.
# Budget-delta inference is therefore invalid for capped attempts.
SPLIT_CAP_LOG = """\
[multi-restart] split-aware: attempt 1 capped to 1800s (size class small; D4 restart split)
[multi-restart] attempt 1: budget 1800s -> /tmp/mr_x_1.dcp
[multi-restart] attempt 1 -> fmax=400.0 status=VALID_OPTIMIZED cost=$0.1 cum_cost=$0.10 exists=True
[multi-restart] attempt 2: budget 1650s -> /tmp/mr_x_2.dcp
[multi-restart] attempt 2 -> fmax=401.0 status=VALID_OPTIMIZED cost=$0.1 cum_cost=$0.20 exists=True
"""


def _rows_by_attempt(rows):
    return {r["attempt"]: r for r in rows}


class TestParseAttemptWall:
    def test_budgets_parsed_from_real_mini_isp_log(self):
        rows = _rows_by_attempt(
            parse_attempt_wall(MINI_ISP_MR_SUMMARY, MINI_ISP_LOG))
        assert rows[1]["budget_s"] == 3500.0
        assert rows[2]["budget_s"] == 2285.0

    def test_summary_elapsed_preferred_over_log_delta(self):
        rows = _rows_by_attempt(
            parse_attempt_wall(MINI_ISP_MR_SUMMARY, MINI_ISP_LOG))
        assert rows[1]["elapsed_s"] == 1198.4
        assert rows[1]["elapsed_source"] == "summary"
        assert rows[2]["elapsed_s"] == 785.2
        assert rows[2]["elapsed_source"] == "summary"

    def test_log_only_fallback_budget_delta_mini_isp(self):
        # No summary at all: elapsed from consecutive budget lines
        # (3500-2285=1215s) and the stranded winner-polish line for the
        # last attempt (2285-1488=797s).
        rows = _rows_by_attempt(parse_attempt_wall({}, MINI_ISP_LOG))
        assert rows[1]["elapsed_s"] == 1215.0
        assert rows[1]["elapsed_source"] == "log_delta"
        assert rows[2]["elapsed_s"] == 797.0
        assert rows[2]["elapsed_source"] == "log_delta"

    def test_log_only_fallback_stop_line_boom_beta(self):
        # Single attempt, floor-gated: elapsed from the stop line
        # (3500-286=3214s) — the RESEARCH wall-breakdown number.
        rows = _rows_by_attempt(parse_attempt_wall({}, BOOM_BETA_LOG))
        assert rows[1]["elapsed_s"] == 3214.0
        assert rows[1]["elapsed_source"] == "log_delta"

    def test_log_only_fmax_status_boomleg(self):
        rows = _rows_by_attempt(parse_attempt_wall({}, BOOMLEG_LOG))
        assert rows[1]["fmax"] == 79.91688643810436
        assert rows[1]["status"] == "VALID_OPTIMIZED"
        assert rows[2]["fmax"] == 77.15454054471107
        assert rows[2]["status"] == "VALID_FALLBACK_BASELINE"
        # attempt 1: 3500-1542=1958s; attempt 2: 1542-287=1255s
        assert rows[1]["elapsed_s"] == 1958.0
        assert rows[2]["elapsed_s"] == 1255.0

    def test_pitfall4_diverges_from_scorecard_wall(self):
        # THE Pitfall-4 pin: no per-attempt attribution (nor their sum)
        # may equal the scorecard whole-wrapper wall_time_seconds.
        for summary in (MINI_ISP_MR_SUMMARY, {}):
            rows = parse_attempt_wall(summary, MINI_ISP_LOG)
            elapsed = [r["elapsed_s"] for r in rows if r["elapsed_s"]]
            assert elapsed, "attribution produced no per-attempt wall"
            for e in elapsed:
                assert abs(e - MINI_ISP_SCORECARD_WALL_S) > 500.0
            # Attributed attempt time excludes wrapper setup and winner-polish
            # overhead, so its sum remains below total wrapper wall time.
            assert abs(sum(elapsed) - MINI_ISP_SCORECARD_WALL_S) > 10.0

    def test_split_cap_invalidates_budget_delta(self):
        # Capped attempt 1: budget line != remaining -> log-delta would be
        # wrong (150s); must return None instead of a fabricated number.
        rows = _rows_by_attempt(parse_attempt_wall({}, SPLIT_CAP_LOG))
        assert rows[1]["budget_s"] == 1800.0
        assert rows[1]["elapsed_s"] is None
        # attempt 2 has no following marker line either -> None
        assert rows[2]["elapsed_s"] is None

    def test_summary_only_no_log(self):
        rows = _rows_by_attempt(parse_attempt_wall(MINI_ISP_MR_SUMMARY, ""))
        assert rows[1]["elapsed_s"] == 1198.4
        assert rows[1]["budget_s"] is None
        assert rows[1]["fmax"] == MINI_ISP_FMAX
        assert rows[2]["status"] == "VALID_OPTIMIZED"

    def test_empty_everything(self):
        assert parse_attempt_wall({}, "") == []


class TestDrawCount:
    def test_mini_isp_two_draws(self):
        assert draw_count(MINI_ISP_MR_SUMMARY) == 2

    def test_empty_summary_zero(self):
        assert draw_count({}) == 0


def _mk_summary(fmax, n_attempts=1, elapsed=None, status="VALID_OPTIMIZED"):
    """Minimal schema-shaped summary for grading tests."""
    elapsed = elapsed if elapsed is not None else [1000.0] * n_attempts
    attempts = [{"i": i + 1, "fmax": fmax, "status": status,
                 "cost": 0.1, "elapsed": elapsed[i],
                 "output": f"/tmp/mr_g_{i+1}.dcp", "exists": True,
                 "run_dir": None}
                for i in range(n_attempts)]
    chosen = dict(attempts[0]) if fmax is not None else None
    return {"input": "/tmp/in.dcp", "final_output": "/tmp/out.dcp",
            "attempts": attempts, "chosen": chosen, "total_wall": 3500.0}


class TestGradePair:
    def test_threshold_constant(self):
        # Prior-sessions meaningfulness convention (S21: |0.25| < 0.5 ->
        # CARD_NEUTRAL; S22: -4.83 -> HARMFUL).
        assert MEANINGFUL_FMAX_DELTA_MHZ == 0.5

    def test_identical_is_neutral(self):
        v = grade_pair(_mk_summary(404.2), _mk_summary(404.2))
        assert v["label"] == "NEUTRAL"
        assert v["never_worse"] is True
        assert v["fmax_delta_mhz"] == 0.0

    def test_s21_sub_threshold_delta_is_neutral(self):
        # A 0.25 MHz paired delta is below the 0.5 MHz meaningful-change threshold.
        v = grade_pair(_mk_summary(65.57), _mk_summary(65.32))
        assert v["label"] == "NEUTRAL"
        assert v["never_worse"] is True

    def test_helps_at_exact_threshold(self):
        v = grade_pair(_mk_summary(100.0), _mk_summary(100.5))
        assert v["label"] == "HELPS"
        assert v["never_worse"] is True

    def test_s22_regression_is_harmful(self):
        # Real S22 route_bound_v1 pair: A 442.28 / B 437.45 -> -4.83.
        v = grade_pair(_mk_summary(442.28), _mk_summary(437.45))
        assert v["label"] == "HARMFUL"
        assert v["never_worse"] is False

    def test_harmful_at_exact_negative_threshold(self):
        # Boundary tie breaks toward HARMFUL (prefer-refusing bias).
        v = grade_pair(_mk_summary(100.0), _mk_summary(99.5))
        assert v["label"] == "HARMFUL"
        assert v["never_worse"] is False

    def test_wall_reclaimed_and_attempts_delta(self):
        off = _mk_summary(80.0, n_attempts=1, elapsed=[3214.0])
        on = _mk_summary(80.0, n_attempts=2, elapsed=[1900.0, 900.0])
        v = grade_pair(off, on)
        assert v["off_attempt_wall_s"] == 3214.0
        assert v["on_attempt_wall_s"] == 2800.0
        assert v["wall_reclaimed_s"] == 414.0
        assert v["attempts_delta"] == 1
        assert v["label"] == "NEUTRAL"

    def test_on_lost_its_output_is_harmful(self):
        off = _mk_summary(100.0)
        on = _mk_summary(None)
        v = grade_pair(off, on)
        assert v["label"] == "HARMFUL"
        assert v["never_worse"] is False

    def test_both_none_is_neutral(self):
        v = grade_pair(_mk_summary(None), _mk_summary(None))
        assert v["label"] == "NEUTRAL"

    def test_behavior_only_scope_recorded(self):
        # T-04-03 mitigation: local numbers are BEHAVIOR evidence only.
        v = grade_pair(_mk_summary(100.0), _mk_summary(100.0))
        assert "BEHAVIOR" in v["evidence_scope"]

    def test_empty_off_summary_refuses_to_grade(self):
        # run_variant returns {} when the OFF run left no readable
        # mr_summary. Falling through to the `off_fmax is None` branch
        # would label every mechanism HELPS/never_worse on no evidence.
        with pytest.raises(ValueError, match="empty OFF summary"):
            grade_pair({}, _mk_summary(100.0))

    def test_off_that_ran_but_produced_no_fmax_still_grades(self):
        # A real OFF run that found no usable fmax is a genuine ON win --
        # that branch stays.
        v = grade_pair(_mk_summary(None), _mk_summary(100.0))
        assert v["label"] == "HELPS"
        assert v["never_worse"] is True


class TestFailedBaselineSkipsGrading:
    """main() must not emit a verdict when the OFF baseline never ran."""

    def _run(self, tmp_path, monkeypatch, off_summary):
        import scripts.ab_wall_economics as abw

        def fake_run_variant(dcp, variant, max_wall, repo, log_path):
            summary = off_summary if variant == "off" else _mk_summary(100.0)
            return summary, ""

        monkeypatch.setattr(abw, "run_variant", fake_run_variant)
        out = tmp_path / "evidence.md"
        rc = abw.main([str(tmp_path / "design.dcp"),
                       "--variants", "off", "split",
                       "--out", str(out)])
        return rc, out.read_text()

    def test_empty_off_summary_writes_no_verdict(self, tmp_path, monkeypatch):
        rc, text = self._run(tmp_path, monkeypatch, {})
        assert rc == 1
        assert "grade skipped" in text
        assert "HELPS" not in text
        assert "never_worse" not in text

    def test_readable_off_summary_still_grades(self, tmp_path, monkeypatch):
        rc, text = self._run(tmp_path, monkeypatch, _mk_summary(100.0))
        assert rc == 0
        assert "grade off-vs-split" in text
        assert "grade skipped" not in text


class TestVariantMatrix:
    def test_make_var_spelling(self):
        # Pin the exact make-variable names the ship targets thread.
        assert variant_make_args("off") == []
        assert variant_make_args("handback") == ["WALL_HANDBACK=1"]
        assert variant_make_args("split") == ["SPLIT_AWARE=1"]
        assert sorted(variant_make_args("both")) == [
            "SPLIT_AWARE=1", "WALL_HANDBACK=1"]
