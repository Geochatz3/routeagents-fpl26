"""
Tests for the scheduler replay logic.  Pure offline; no Vivado or LLM
involvement.  Run with `python3 -m scheduler.test_replay` from the repo
root.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from scheduler.replay import simulate_scheduler, replay_from_jsonl, _row_to_result
from scheduler.runner import CandidateResult, SchedulerConfig, select_best


def _mk(name, exit_code=0, fmax=None, wall=600, output_exists=True):
    """Build a CandidateResult.  Default valid: exit 0, dcp exists, fmax set."""
    # is_valid gates on output_dcp.exists() — point at /tmp for simulated
    # validity, /nonexistent for invalid.
    if output_exists and fmax is not None:
        f = tempfile.NamedTemporaryFile(suffix=".dcp", delete=False)
        f.close()
        path = Path(f.name)
    else:
        path = Path("/nonexistent")
    return CandidateResult(
        candidate_name=name,
        output_dcp=path,
        exit_code=exit_code,
        wall_time_s=wall,
        final_fmax_mhz=fmax,
    )


class SelectBestTests(unittest.TestCase):
    def test_picks_highest_fmax(self):
        results = [_mk("anchor", fmax=100.0), _mk("v0_3", fmax=110.0)]
        best = select_best(results)
        self.assertIsNotNone(best)
        self.assertEqual(best.candidate_name, "v0_3")

    def test_skips_invalid(self):
        results = [_mk("anchor", exit_code=1, fmax=200.0, output_exists=False),
                   _mk("v0_3", fmax=50.0)]
        best = select_best(results)
        self.assertIsNotNone(best)
        self.assertEqual(best.candidate_name, "v0_3")

    def test_returns_none_when_all_invalid(self):
        results = [_mk("anchor", exit_code=1, fmax=None, output_exists=False),
                   _mk("v0_3", exit_code=1, fmax=None, output_exists=False)]
        self.assertIsNone(select_best(results))

    def test_tie_break_prefers_lower_wall_time(self):
        results = [_mk("anchor", fmax=100.0, wall=1500),
                   _mk("v0_3", fmax=100.0, wall=600)]
        best = select_best(results)
        self.assertEqual(best.candidate_name, "v0_3")


class SimulateSchedulerTests(unittest.TestCase):
    def test_runs_all_candidates_under_budget(self):
        results = [_mk("anchor", fmax=100.0, wall=600),
                   _mk("v0_3", fmax=110.0, wall=900)]
        config = SchedulerConfig(total_budget_s=3600)
        sim = simulate_scheduler(results, config)
        self.assertEqual(sim["scheduled_count"], 2)
        self.assertEqual(sim["selected"].candidate_name, "v0_3")

    def test_stops_scheduling_when_budget_low(self):
        # First candidate eats most of the budget — second can't fit
        results = [_mk("anchor", fmax=100.0, wall=3300),
                   _mk("v0_3", fmax=110.0, wall=900)]
        config = SchedulerConfig(total_budget_s=3600, min_first_candidate_s=600)
        sim = simulate_scheduler(results, config)
        self.assertEqual(sim["scheduled_count"], 1)
        self.assertEqual(sim["selected"].candidate_name, "anchor")

    def test_skips_to_oracle_when_first_invalid(self):
        # First candidate failed; second succeeded
        results = [_mk("anchor", exit_code=1, fmax=None, output_exists=False, wall=500),
                   _mk("v0_3", fmax=80.0, wall=900)]
        config = SchedulerConfig(total_budget_s=3600)
        sim = simulate_scheduler(results, config)
        self.assertEqual(sim["selected"].candidate_name, "v0_3")


class ReplayFromJsonlTests(unittest.TestCase):
    def _write_jsonl(self, rows):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.close()
        return Path(f.name)

    def _row(self, design, candidate, exit_code=0, fmax=100.0, wall=600,
             with_dcp=True):
        out = None
        if with_dcp:
            t = tempfile.NamedTemporaryFile(suffix=".dcp", delete=False)
            t.close()
            out = t.name
        return {
            "design": design, "candidate": candidate, "commit": "deadbeef",
            "exit_code": exit_code, "wall_time_s": wall,
            "log_path": "/tmp/x.log", "optimized_dcp": out,
            "final_fmax_mhz": fmax, "final_wns_ns": -0.5,
            "delta_fmax_mhz": 5.0, "iterations": 4,
            "tool_calls": 30, "force_continues": 1, "total_cost_usd": 0.1,
            "completed": exit_code == 0,
        }

    def test_basic_two_design_match(self):
        # On both designs, v0_3 is the higher-fmax winner.
        jsonl = self._write_jsonl([
            self._row("alpha", "anchor", fmax=100.0),
            self._row("alpha", "v0_3", fmax=110.0),
            self._row("beta",  "anchor", fmax=200.0),
            self._row("beta",  "v0_3", fmax=220.0),
        ])
        config = SchedulerConfig(total_budget_s=3600)
        rep = replay_from_jsonl(jsonl, ["anchor", "v0_3"], config)
        self.assertEqual(rep["matches_oracle"], 2)
        self.assertEqual(rep["designs_with_results"], 2)

    def test_anchor_wins_tracked(self):
        # Design where anchor beats v0_3; oracle and scheduler should agree.
        jsonl = self._write_jsonl([
            self._row("amd_mini-isp", "anchor", fmax=394.48, wall=595),
            self._row("amd_mini-isp", "v0_3",   fmax=375.38, wall=495),
        ])
        config = SchedulerConfig(total_budget_s=3600)
        rep = replay_from_jsonl(jsonl, ["anchor", "v0_3"], config)
        self.assertEqual(rep["designs"]["amd_mini-isp"]["selected"], "anchor")
        self.assertEqual(rep["designs"]["amd_mini-isp"]["oracle"], "anchor")
        self.assertTrue(rep["designs"]["amd_mini-isp"]["match"])


if __name__ == "__main__":
    unittest.main()
