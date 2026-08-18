"""Unit tests for recipe_post_route_phys_opt_sweep.

Validates:
- per-flag iteration order matches the default sweep
- improvements are kept; regressions trigger best_valid revert
- budget pre-flight aborts the sweep gracefully
- tool errors on one flag don't poison subsequent flags
- final JSON carries cumulative deltas + per-flag breakdown
- invalid flag arguments are filtered out (no full-directive smuggling)
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import DCPOptimizer


def _async(coro):
    return asyncio.run(coro)


def _make_opt(tmp_path: Path) -> DCPOptimizer:
    opt = DCPOptimizer(api_key="test", run_dir=tmp_path)
    opt.clock_period = 1.570
    opt.best_wns = -1.000
    opt.initial_wns = -1.000
    return opt


class PhysOptSweepDispatchTests(unittest.TestCase):
    """High-level: the new recipe is registered in call_tool dispatch."""

    def test_dispatch_routes_to_recipe_handler(self):
        with tempfile.TemporaryDirectory() as tmp:
            opt = _make_opt(Path(tmp))
            with mock.patch.object(
                opt, "_recipe_post_route_phys_opt_sweep",
                new=mock.AsyncMock(return_value='{"status": "stub"}'),
            ) as stub:
                result = _async(opt.call_tool(
                    "recipe_post_route_phys_opt_sweep", {"flags": ["critical_cell_opt"]}
                ))
                self.assertEqual(result, '{"status": "stub"}')
                stub.assert_awaited_once()


class PhysOptSweepBehaviourTests(unittest.TestCase):
    """Behaviour: per-flag iteration + commit-or-revert logic."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.opt = _make_opt(self.tmp_path)
        # Pre-stage a best_valid mirror so revert has somewhere to go.
        self.best_valid = self.tmp_path / "best_valid.dcp"
        self.best_valid.write_bytes(b"BEST_VALID_DCP_BYTES")
        self.opt._best_valid_dcp = self.best_valid

    def tearDown(self):
        self.tmp.cleanup()

    def test_invalid_flags_filtered_out(self):
        """A stray non-sub-flag argument (or a full-directive name)
        must not slip through.  No flags survive → error envelope."""
        async def stub_call(*_a, **_kw):
            raise AssertionError("should not be reached when no valid flags")
        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=stub_call)):
            out = _async(self.opt._recipe_post_route_phys_opt_sweep(
                {"flags": ["AggressiveExplore", "synth_design", "foo_bar"]}
            ))
        result = json.loads(out)
        self.assertEqual(result["status"], "error")
        self.assertIn("no valid flags", result["error"])

    def test_no_initial_wns_returns_error(self):
        """Recipe refuses to run before Phase 1 timing analysis."""
        self.opt.best_wns = float("-inf")
        out = _async(self.opt._recipe_post_route_phys_opt_sweep(
            {"flags": ["critical_cell_opt"]}
        ))
        result = json.loads(out)
        self.assertEqual(result["status"], "error")
        self.assertIn("best_wns not yet established", result["error"])

    def test_committed_flag_keeps_improvement(self):
        """One sub-flag improves WNS by 0.05 ns → committed entry."""
        # Sequence: phys_opt_design → get_wns (improved) → no revert.
        async def fake_call(name, args):
            if name == "vivado_phys_opt_design":
                # Simulate that phys_opt ran and improved internal state.
                self.opt.best_wns = -0.950  # +0.050 from -1.000
                return "phys_opt_design completed"
            if name == "vivado_get_wns":
                return "-0.950"
            return "ok"

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake_call)):
            out = _async(self.opt._recipe_post_route_phys_opt_sweep(
                {"flags": ["critical_cell_opt"]}
            ))
        result = json.loads(out)
        self.assertEqual(result["status"], "success")
        self.assertIn("critical_cell_opt", result["flags_committed"])
        self.assertEqual(result["flags_reverted"], [])
        self.assertAlmostEqual(result["delta_wns_ns"], 0.050, places=3)
        self.assertGreater(result["delta_fmax_mhz"], 0.0)

    def test_regression_triggers_best_valid_revert(self):
        """One sub-flag regresses WNS → recipe reverts via
        vivado_open_checkpoint on best_valid.dcp."""
        revert_calls: list = []

        async def fake_call(name, args):
            if name == "vivado_phys_opt_design":
                # Regression: -1.000 → -1.200 (worse by 0.2 ns).
                self.opt.best_wns = -1.200
                return "phys_opt_design completed"
            if name == "vivado_get_wns":
                return "-1.200"
            if name == "vivado_open_checkpoint":
                revert_calls.append(args.get("dcp_path"))
                return "checkpoint opened"
            return "ok"

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake_call)):
            out = _async(self.opt._recipe_post_route_phys_opt_sweep(
                {"flags": ["critical_cell_opt"]}
            ))
        result = json.loads(out)
        self.assertIn("critical_cell_opt", result["flags_reverted"])
        entry = result["per_flag"][0]
        self.assertEqual(entry["status"], "reverted_regression")
        self.assertEqual(entry["revert_status"], "reopened_best_valid")
        # Vivado was told to re-open the mirror.
        self.assertEqual(len(revert_calls), 1)
        self.assertIn("best_valid.dcp", revert_calls[0])
        # best_wns must have been restored to the pre-flag value so
        # the next flag in a multi-flag sweep starts from the right place.
        self.assertAlmostEqual(self.opt.best_wns, -1.000, places=3)

    def test_sub_epsilon_noise_does_not_commit(self):
        """Tiny WNS change within +/- epsilon is reverted as noise."""
        async def fake_call(name, args):
            if name == "vivado_phys_opt_design":
                self.opt.best_wns = -0.995  # +0.005 ns < epsilon 0.010
                return "ok"
            if name == "vivado_get_wns":
                return "-0.995"
            if name == "vivado_open_checkpoint":
                return "ok"
            return "ok"

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake_call)):
            out = _async(self.opt._recipe_post_route_phys_opt_sweep(
                {"flags": ["critical_cell_opt"], "epsilon_ns": 0.010}
            ))
        result = json.loads(out)
        self.assertIn("critical_cell_opt", result["flags_reverted"])
        self.assertEqual(result["per_flag"][0]["status"], "reverted_no_gain")

    def test_per_flag_tool_error_continues_sweep(self):
        """One sub-flag's tool error does not abort the whole sweep."""
        call_log: list = []

        async def fake_call(name, args):
            call_log.append((name, args))
            if name == "vivado_phys_opt_design":
                # First flag errors; second succeeds.
                flag = next(k for k in args.keys() if k != "directive")
                if flag == "critical_cell_opt":
                    return json.dumps({"error": "simulated tool failure"})
                # equ_drivers_opt: improvement.
                self.opt.best_wns = -0.940
                return "ok"
            if name == "vivado_get_wns":
                return "-0.940"
            return "ok"

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake_call)):
            out = _async(self.opt._recipe_post_route_phys_opt_sweep(
                {"flags": ["critical_cell_opt", "equ_drivers_opt"]}
            ))
        result = json.loads(out)
        # Both flags attempted; first errored, second committed.
        self.assertEqual(result["flags_attempted"],
                          ["critical_cell_opt", "equ_drivers_opt"])
        self.assertEqual(result["flags_committed"], ["equ_drivers_opt"])
        statuses = [e["status"] for e in result["per_flag"]]
        self.assertIn("tool_error", statuses)
        self.assertIn("committed", statuses)

    def test_budget_pre_flight_aborts_remaining_sweep(self):
        """When budget remaining < single-flag estimate, the recipe
        records skipped_budget and stops at the next iteration."""
        # Set a tight budget AND a low default risky estimate to keep
        # the test deterministic.
        self.opt._budget_deadline = time.time() + 5.0  # 5s remaining
        self.opt.max_wall_seconds = 5.0

        call_count = {"phys_opt": 0}

        async def fake_call(name, args):
            if name == "vivado_phys_opt_design":
                call_count["phys_opt"] += 1
                self.opt.best_wns = -0.950
                return "ok"
            if name == "vivado_get_wns":
                return "-0.950"
            return "ok"

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake_call)):
            out = _async(self.opt._recipe_post_route_phys_opt_sweep(
                {"flags": ["critical_cell_opt", "equ_drivers_opt", "placement_opt"]}
            ))
        result = json.loads(out)
        # First flag should NOT have started — 5s remaining < ~120s estimate.
        self.assertEqual(call_count["phys_opt"], 0)
        # Last per_flag entry must be a skipped_budget marker.
        self.assertTrue(
            any(e.get("status") == "skipped_budget" for e in result["per_flag"]),
            f"expected skipped_budget entry, got {result['per_flag']}"
        )

    def test_early_exit_on_first_gain_default(self):
        """Default early_exit_on_first_gain=True: recipe returns after
        the first commit; remaining flags are not attempted.  Leaves
        wall time for the next-layer recipe in the timing-onion loop."""
        attempted: list = []

        async def fake_call(name, args):
            if name == "vivado_phys_opt_design":
                flag = next(k for k in args.keys() if k != "directive")
                attempted.append(flag)
                # First flag improves; the rest would too if reached.
                self.opt.best_wns = -0.950
                return "ok"
            if name == "vivado_get_wns":
                return "-0.950"
            return "ok"

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake_call)):
            out = _async(self.opt._recipe_post_route_phys_opt_sweep({}))
        result = json.loads(out)
        # Only the first flag should have been attempted.
        self.assertEqual(len(attempted), 1)
        self.assertEqual(result["flags_committed"], [attempted[0]])
        self.assertIn("early_exit_reason", result)
        self.assertIn("early_exit_on_first_gain", result["early_exit_reason"])

    def test_early_exit_disabled_runs_all_flags(self):
        """early_exit_on_first_gain=False reverts to the original
        sweep-everything behaviour."""
        attempted: list = []

        async def fake_call(name, args):
            if name == "vivado_phys_opt_design":
                flag = next(k for k in args.keys() if k != "directive")
                attempted.append(flag)
                # Each flag commits a small improvement.  best_wns
                # rises monotonically; epsilon=0.010 default.
                self.opt.best_wns = -1.000 + 0.020 * len(attempted)
                return "ok"
            if name == "vivado_get_wns":
                return f"{-1.000 + 0.020 * len(attempted):.3f}"
            return "ok"

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake_call)):
            out = _async(self.opt._recipe_post_route_phys_opt_sweep(
                {"early_exit_on_first_gain": False}
            ))
        result = json.loads(out)
        # All 6 default flags attempted.
        self.assertEqual(len(attempted), 6)
        self.assertNotIn("early_exit_reason", result)

    def test_max_successful_subpasses_allows_chained_commits(self):
        """With max_successful_subpasses=2 and early_exit_on_first_gain
        true, recipe stops after the second commit (not the first)."""
        attempted: list = []

        async def fake_call(name, args):
            if name == "vivado_phys_opt_design":
                flag = next(k for k in args.keys() if k != "directive")
                attempted.append(flag)
                self.opt.best_wns = -1.000 + 0.020 * len(attempted)
                return "ok"
            if name == "vivado_get_wns":
                return f"{-1.000 + 0.020 * len(attempted):.3f}"
            return "ok"

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake_call)):
            out = _async(self.opt._recipe_post_route_phys_opt_sweep(
                {"max_successful_subpasses": 2}
            ))
        result = json.loads(out)
        self.assertEqual(len(attempted), 2)
        self.assertEqual(len(result["flags_committed"]), 2)
        self.assertIn("early_exit_reason", result)

    def test_min_remaining_time_for_next_layer_triggers_early_exit(self):
        """When remaining budget drops below the floor after a commit,
        recipe stops to preserve room for a follow-up recipe.

        Uses MIN_USEFUL_TOOL_SECONDS patch so the test can use a tight
        budget; the next-layer floor is set high so a single commit
        triggers the exit."""
        attempted: list = []

        async def fake_call(name, args):
            if name == "vivado_phys_opt_design":
                attempted.append(next(k for k in args.keys() if k != "directive"))
                self.opt.best_wns = -0.950
                return "ok"
            if name == "vivado_get_wns":
                return "-0.950"
            return "ok"

        # Budget 400s clears the recipe's internal pre-flight floor
        # (single_flag_estimate = max(120, 0.5*600) = 300s).  After the
        # first commit remaining≈400s, which is below the 600s
        # next-layer floor → exit fires.
        self.opt._budget_deadline = time.time() + 400.0
        self.opt.max_wall_seconds = 400.0
        import dcp_optimizer as _mod
        with mock.patch.object(_mod, "MIN_USEFUL_TOOL_SECONDS", 0.05), \
             mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake_call)):
            out = _async(self.opt._recipe_post_route_phys_opt_sweep(
                # Disable first-gain exit so the floor check is the
                # only trigger; raise max so we don't hit it.
                {"early_exit_on_first_gain": False,
                 "max_successful_subpasses": 99,
                 "min_remaining_time_for_next_layer_s": 500.0}
            ))
        result = json.loads(out)
        self.assertEqual(len(attempted), 1)
        self.assertIn("early_exit_reason", result)
        self.assertIn("min_remaining_for_next_layer", result["early_exit_reason"])

    def test_default_flag_order(self):
        """Default sweep iterates the canonical 6-flag order (no args)."""
        attempted: list = []

        async def fake_call(name, args):
            if name == "vivado_phys_opt_design":
                flag = next(k for k in args.keys() if k != "directive")
                attempted.append(flag)
                # No improvement → every flag reverts.
                self.opt.best_wns = -1.000
                return "ok"
            if name == "vivado_get_wns":
                return "-1.000"
            if name == "vivado_open_checkpoint":
                return "ok"
            return "ok"

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake_call)):
            _async(self.opt._recipe_post_route_phys_opt_sweep({}))

        self.assertEqual(attempted, list(self.opt._DEFAULT_PHYS_OPT_SWEEP))


if __name__ == "__main__":
    unittest.main()
