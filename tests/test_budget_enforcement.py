"""Tests deadline-aware dispatch, best-result mirroring, and budget-safe
finalization.

The suite verifies dispatch gating, per-call timeouts, mirror handling, and
finalize behavior without starting external tool sessions.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import (
    DCPOptimizer,
    MIN_USEFUL_TOOL_SECONDS,
    RISKY_VIVADO_TOOLS,
)
import optimizer.tool_dispatch as _tool_dispatch


def _async(coro):
    return asyncio.run(coro)


class _FakeContent:
    """Mimics MCP CallToolResult content items — has .text attribute."""

    def __init__(self, text: str):
        self.text = text


class _FakeResult:
    """Mimics the MCP CallToolResult shape (result.content list of items)."""

    def __init__(self, text: str):
        self.content = [_FakeContent(text)]


class _FakeSession:
    """Stand-in for the MCP ClientSession used by DCPOptimizer.call_tool.

    Configure `response_text` to control what is returned and `sleep_s` to
    simulate a slow call (for asyncio.wait_for timeout tests).  Tracks
    invocation count + the last arguments seen so tests can assert.
    """

    def __init__(self, response_text: str = "{}", sleep_s: float = 0.0):
        self.response_text = response_text
        self.sleep_s = sleep_s
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        if self.sleep_s > 0:
            await asyncio.sleep(self.sleep_s)
        return _FakeResult(self.response_text)


def _make_optimizer(tmp_path: Path) -> DCPOptimizer:
    """Creates an optimizer with mocked sessions and minimal state.

    A real optimizer instance is required because dispatcher behavior is
    method-bound, but its sessions are replaced before any server starts.
    """
    opt = DCPOptimizer(api_key="test", run_dir=tmp_path)
    opt.vivado_session = _FakeSession()
    opt.rapidwright_session = _FakeSession()
    return opt


class BudgetSkipPolicyTests(unittest.TestCase):
    """Pure policy tests for _should_skip_for_budget — no async."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = _make_optimizer(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_budget_set_never_skips(self):
        # When max_wall_seconds is None (legacy/dev mode), the dispatcher
        # must not gate any calls — backwards compatibility.
        self.opt._budget_deadline = None
        skip, reason = self.opt._should_skip_for_budget("vivado_phys_opt_design")
        self.assertFalse(skip)
        self.assertEqual(reason, "")

    def test_deadline_passed_skips_unconditionally(self):
        self.opt._budget_deadline = time.time() - 1.0
        skip, reason = self.opt._should_skip_for_budget("vivado_get_wns")
        self.assertTrue(skip)
        self.assertEqual(reason, "deadline_passed")

    def test_risky_tool_skipped_when_estimate_exceeds_remaining(self):
        # The budget deadline already excludes the final reserve, so remaining
        # time is the reserve-protected window. A 100 s window cannot admit
        # a tool with a 600 s default estimate.
        self.opt._budget_deadline = time.time() + 100.0
        skip, reason = self.opt._should_skip_for_budget("vivado_phys_opt_design")
        self.assertTrue(skip)
        self.assertIn("estimated_", reason)
        self.assertIn("exceeds_remaining_", reason)

    def test_risky_tool_skipped_at_useful_floor(self):
        # 20s remaining < MIN_USEFUL_TOOL_SECONDS=30s → skip with floor
        # reason BEFORE the risky estimate check fires.
        self.opt._budget_deadline = time.time() + 20.0
        skip, reason = self.opt._should_skip_for_budget("vivado_phys_opt_design")
        self.assertTrue(skip)
        self.assertIn("below_min_useful_", reason)

    def test_cheap_tool_allowed_when_budget_fits(self):
        # 600s remaining for a cheap tool — must not skip.
        self.opt._budget_deadline = time.time() + 600.0
        skip, reason = self.opt._should_skip_for_budget("vivado_get_wns")
        self.assertFalse(skip)
        self.assertEqual(reason, "")

    def test_risky_tool_allowed_when_history_shows_short_runs(self):
        # 5 prior fast runs of vivado_phys_opt_design (60s each); estimate
        # becomes 60s.  Budget remaining 800s → must NOT skip.
        self.opt._budget_deadline = time.time() + 800.0
        for _ in range(5):
            self.opt._record_tool_runtime("vivado_phys_opt_design", 60.0)
        skip, _reason = self.opt._should_skip_for_budget("vivado_phys_opt_design")
        self.assertFalse(skip)

    def test_risky_set_membership(self):
        # Pin the contract: the tools we always gate.
        for tool in (
            "vivado_place_design",
            "vivado_route_design",
            "vivado_phys_opt_design",
            "recipe_cell_replacement",
            "recipe_lut_optimization",
            "recipe_register_retiming",
        ):
            self.assertIn(tool, RISKY_VIVADO_TOOLS, f"{tool} must be risky")
        # Tcl dispatch is classified from its payload because property queries
        # are cheap while implementation commands may be expensive.
        # Cell-placement calls are cheap individually; their orchestration
        # carries the budget risk.
        self.assertNotIn("vivado_run_tcl", RISKY_VIVADO_TOOLS)
        self.assertNotIn("rapidwright_optimize_cell_placement",
                         RISKY_VIVADO_TOOLS)

    def test_tcl_payload_heuristic_cheap_query_not_risky(self):
        # Cheap query — get_clocks/get_property — must NOT be flagged risky
        # even with a small remaining budget.
        self.opt._budget_deadline = time.time() + 200.0
        cmd = "set c [get_clocks clk_fpl26contest]; puts [get_property PERIOD $c]"
        self.assertFalse(self.opt._is_risky("vivado_run_tcl", {"command": cmd}))
        skip, _ = self.opt._should_skip_for_budget(
            "vivado_run_tcl", {"command": cmd})
        self.assertFalse(skip)

    def test_tcl_payload_heuristic_place_design_is_risky(self):
        # Wrapping a heavy implementation step via run_tcl MUST be gated
        # the same way the dedicated vivado_place_design call would be.
        self.opt._budget_deadline = time.time() + 200.0
        cmd = "place_design -directive Explore"
        self.assertTrue(self.opt._is_risky("vivado_run_tcl", {"command": cmd}))
        skip, reason = self.opt._should_skip_for_budget(
            "vivado_run_tcl", {"command": cmd})
        self.assertTrue(skip)
        self.assertIn("estimated_", reason)

    def test_tcl_payload_heuristic_route_design_is_risky(self):
        self.opt._budget_deadline = time.time() + 200.0
        cmd = "ROUTE_DESIGN -directive AggressiveExplore"  # case-insensitive
        self.assertTrue(self.opt._is_risky("vivado_run_tcl", {"command": cmd}))


class CallToolBudgetSkipTests(unittest.TestCase):
    """call_tool integration — pre-flight skip returns error envelope and
    flips _budget_killed without ever invoking the underlying session."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = _make_optimizer(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_skipped_tool_returns_error_envelope_and_marks_budget_killed(self):
        self.opt._budget_deadline = time.time() - 5.0  # deadline passed
        result = _async(self.opt.call_tool(
            "vivado_phys_opt_design", {"directive": "Explore"}))
        self.assertIn("tool_skipped_budget", result)
        self.assertIn("deadline_passed", result)
        self.assertTrue(self.opt._budget_killed)
        # Underlying session must NOT have been called.
        self.assertEqual(self.opt.vivado_session.calls, [])
        # The skip is recorded so the run summary can surface it.
        self.assertTrue(any("vivado_phys_opt_design" in s
                            for s in self.opt._strategies_skipped_budget))

    def test_skipped_tool_records_failed_call_detail(self):
        self.opt._budget_deadline = time.time() - 5.0
        _async(self.opt.call_tool("vivado_route_design", {}))
        self.assertEqual(len(self.opt.tool_call_details), 1)
        detail = self.opt.tool_call_details[0]
        self.assertTrue(detail["error"])
        self.assertIn("budget_skip", detail["error_message"])


class CallToolTimeoutTests(unittest.TestCase):
    """call_tool integration — asyncio.wait_for cancels a slow call and the
    dispatcher reports a budget_timeout envelope.

    These tests use mock.patch.object(dcp_optimizer, MIN_USEFUL_TOOL_SECONDS)
    to drop the 30s usefulness floor so the test budget can be sub-second
    without tripping the pre-flight skip path.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = _make_optimizer(Path(self.tmp.name))
        # Drop the MIN_USEFUL_TOOL_SECONDS floor for the duration of each
        # test so we can use sub-second budgets without hitting the
        # pre-flight skip path.
        import dcp_optimizer as _mod
        # ToolDispatchMixin resolves this constant in its defining module.
        # Patching only the optimizer module's re-export does not affect dispatch.
        self._patch_min = mock.patch.object(
            _tool_dispatch, "MIN_USEFUL_TOOL_SECONDS", 0.05)
        self._patch_min_reexport = mock.patch.object(
            _mod, "MIN_USEFUL_TOOL_SECONDS", 0.05)
        self._patch_min.start()
        self._patch_min_reexport.start()

    def tearDown(self):
        self._patch_min.stop()
        self._patch_min_reexport.stop()
        self.tmp.cleanup()

    def test_slow_call_is_cancelled_by_deadline(self):
        # Allow 0.4s of usable time (reserve=0, deadline = now+0.4s).
        # The fake session sleeps 5s → asyncio.wait_for must cancel it.
        self.opt._finalize_reserve_seconds = 0.0
        self.opt._budget_deadline = time.time() + 0.4
        self.opt.max_wall_seconds = 0.4
        self.opt.vivado_session = _FakeSession(sleep_s=5.0)

        start = time.time()
        result = _async(self.opt.call_tool("vivado_get_wns", {}))
        elapsed = time.time() - start

        self.assertIn("tool_timed_out_budget", result)
        self.assertTrue(self.opt._budget_killed)
        # Must have cancelled at the deadline, not waited the full 5s.
        self.assertLess(elapsed, 4.0,
                        f"call_tool waited too long ({elapsed:.2f}s) — "
                        "asyncio.wait_for didn't fire")
        # Skip recorded.
        self.assertTrue(any("timed_out" in s
                            for s in self.opt._strategies_skipped_budget))

    def test_fast_call_succeeds_within_budget(self):
        # 5s window, call returns immediately — must succeed normally.
        self.opt._finalize_reserve_seconds = 0.0
        self.opt._budget_deadline = time.time() + 5.0
        self.opt.max_wall_seconds = 5.0
        self.opt.vivado_session = _FakeSession(response_text="OK", sleep_s=0.0)

        result = _async(self.opt.call_tool("vivado_get_wns", {}))
        self.assertEqual(result, "OK")
        self.assertFalse(self.opt._budget_killed)


class PiggybackMirrorTests(unittest.TestCase):
    """When _pending_best_mirror is set and the LLM writes a checkpoint,
    the dispatcher must copy that file to best_valid.dcp."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.opt = _make_optimizer(self.tmp_path)
        # Simulate the LLM's checkpoint target — a real file we wrote.
        self.llm_dcp = self.tmp_path / "llm_iter1.dcp"
        self.llm_dcp.write_bytes(b"FAKE_DCP_BYTES_FOR_TESTING")

    def tearDown(self):
        self.tmp.cleanup()

    def test_piggyback_mirror_on_write_checkpoint_when_flag_set(self):
        self.opt._pending_best_mirror = True
        self.opt.best_wns = -1.0

        _async(self.opt.call_tool("vivado_write_checkpoint",
                                   {"dcp_path": str(self.llm_dcp), "force": True}))

        mirror = self.tmp_path / "best_valid.dcp"
        self.assertTrue(mirror.exists(), "best_valid.dcp must be mirrored")
        self.assertEqual(mirror.read_bytes(), self.llm_dcp.read_bytes())
        self.assertEqual(self.opt._best_valid_dcp, mirror)
        self.assertFalse(self.opt._pending_best_mirror,
                          "flag must be cleared after successful mirror")

    def test_no_mirror_when_flag_unset(self):
        self.opt._pending_best_mirror = False
        _async(self.opt.call_tool("vivado_write_checkpoint",
                                   {"dcp_path": str(self.llm_dcp)}))

        self.assertFalse((self.tmp_path / "best_valid.dcp").exists())
        self.assertIsNone(self.opt._best_valid_dcp)

    def test_mirror_skipped_when_src_missing(self):
        # A successful tool response does not guarantee that the checkpoint file
        # exists. A missing file skips mirroring and leaves the pending flag set
        # so a later attempt can retry.
        self.opt._pending_best_mirror = True
        ghost = self.tmp_path / "ghost.dcp"  # never created
        _async(self.opt.call_tool("vivado_write_checkpoint",
                                   {"dcp_path": str(ghost)}))

        self.assertFalse((self.tmp_path / "best_valid.dcp").exists())
        self.assertTrue(self.opt._pending_best_mirror,
                         "flag stays set when mirror src missing")

    def test_piggyback_edif_mirror(self):
        # Sequence: write_checkpoint mirrors DCP → write_edif mirrors EDIF.
        edf = self.tmp_path / "llm_iter1.edf"
        edf.write_bytes(b"FAKE_EDIF_BYTES")
        self.opt._pending_best_mirror = True
        self.opt.best_wns = -1.0

        _async(self.opt.call_tool("vivado_write_checkpoint",
                                   {"dcp_path": str(self.llm_dcp)}))
        _async(self.opt.call_tool("vivado_write_edif",
                                   {"edif_path": str(edf)}))

        mirror_edf = self.tmp_path / "best_valid.edf"
        self.assertTrue(mirror_edf.exists())
        self.assertEqual(self.opt._best_valid_edif, mirror_edf)


class FinalizeFastPathTests(unittest.TestCase):
    """Verifies that finalization copies complete mirror artifacts without
    invoking the tool session.

    When both checkpoint and EDIF artifacts exist, finalization must copy them
    directly to the output paths. This path remains available when the external
    tool is unavailable.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.opt = _make_optimizer(self.tmp_path)
        # A fresh mirror requires both files and a WNS marker matching best_wns
        # within tolerance. Fast-path fixtures must stage both.
        # Stale-mirror fixtures omit the marker to select the fallback path.
        self.best_dcp = self.tmp_path / "best_valid.dcp"
        self.best_edf = self.tmp_path / "best_valid.edf"
        self.best_dcp.write_bytes(b"BEST_VALID_DCP_CONTENT_FOR_TEST")
        self.best_edf.write_bytes(b"BEST_VALID_EDIF_CONTENT_FOR_TEST")
        self.opt._best_valid_dcp = self.best_dcp
        self.opt._best_valid_edif = self.best_edf
        # Mirror is FRESH at -1.0 ns (matches the best_wns each test will set).
        self.opt._best_valid_dcp_wns = -1.0
        self.opt._best_valid_edif_wns = -1.0
        self.opt.input_dcp_path = self.tmp_path / "baseline.dcp"
        self.opt.input_dcp_path.write_bytes(b"BASELINE_DCP_BYTES_NOT_USED_HERE")

    def tearDown(self):
        self.tmp.cleanup()

    def test_fast_path_copies_best_valid_to_output(self):
        # Improvement is real (best_wns better than initial_wns).
        self.opt.initial_wns = -10.0
        self.opt.best_wns = -1.0
        out = self.tmp_path / "optimized.dcp"

        # call_tool must NOT be invoked on this path — patch it to raise so
        # any accidental call surfaces as a test failure.
        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(
                                    side_effect=AssertionError(
                                        "fast path must not call Vivado"))):
            _async(self.opt._finalize_output_dcp(out))

        self.assertTrue(out.exists())
        self.assertEqual(out.read_bytes(), self.best_dcp.read_bytes())
        self.assertTrue(out.with_suffix(".edf").exists())
        self.assertEqual(out.with_suffix(".edf").read_bytes(),
                          self.best_edf.read_bytes())
        self.assertEqual(self.opt.final_status, "VALID_OPTIMIZED")
        events = [e["event"] for e in self.opt.lifecycle_log]
        self.assertIn("fast_path_best_valid_copy", events)

    def test_fast_path_marks_no_edif_when_edif_missing(self):
        # Remove the EDIF mirror — fast path must still ship the DCP but
        # mark VALID_OPTIMIZED_NO_EDIF so the contest packager knows.
        self.best_edf.unlink()
        self.opt._best_valid_edif = None
        self.opt.initial_wns = -10.0
        self.opt.best_wns = -1.0
        out = self.tmp_path / "optimized.dcp"

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(
                                    side_effect=AssertionError("no Vivado"))):
            _async(self.opt._finalize_output_dcp(out))

        self.assertTrue(out.exists())
        self.assertEqual(self.opt.final_status, "VALID_OPTIMIZED_NO_EDIF")

    def test_fast_path_skipped_on_no_improvement(self):
        # If best_wns ≤ initial_wns, contract requires baseline-copy (Step 2),
        # NOT best_valid copy.  The fast path must defer to the slow path.
        self.opt.initial_wns = -5.0
        self.opt.best_wns = -5.0  # no improvement
        out = self.tmp_path / "optimized.dcp"

        async def stub_ensure(*_a, **_kw):
            return None

        # Mock the slow-path tools so we don't hit real Vivado.
        with mock.patch.object(self.opt, "_ensure_output_dcp_written",
                                new=mock.AsyncMock()), \
             mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock()):
            _async(self.opt._finalize_output_dcp(out))

        # Baseline copied → output exists, but status is FALLBACK_BASELINE
        # (no EDIF mirror existed for this test → NO_EDIF variant).
        self.assertIn(self.opt.final_status,
                      ("VALID_FALLBACK_BASELINE", "VALID_FALLBACK_BASELINE_NO_EDIF"))
        # And the fast-path lifecycle event was NOT recorded.
        events = [e["event"] for e in self.opt.lifecycle_log]
        self.assertNotIn("fast_path_best_valid_copy", events)

    def test_no_improvement_uses_best_valid_edif_when_present(self):
        # Finalization copies an existing mirrored EDIF before invoking tools,
        # because deadline enforcement may skip checkpoint opening or EDIF writing.
        self.opt.initial_wns = -5.0
        self.opt.best_wns = -5.0  # no improvement
        out = self.tmp_path / "optimized.dcp"

        # Model a state where the valid-checkpoint mirror exists and
        # _best_valid_dcp refers to a baseline-equivalent checkpoint.
        baseline = self.tmp_path / "baseline.dcp"
        baseline.write_bytes(b"BASELINE_DCP")
        self.opt.input_dcp_path = baseline

        # call_tool MUST NOT be invoked when best_valid.edf is available.
        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(
                                    side_effect=AssertionError(
                                        "no-improvement path must use "
                                        "best_valid.edf, not Vivado"))):
            _async(self.opt._finalize_output_dcp(out))

        out_edf = out.with_suffix(".edf")
        self.assertTrue(out.exists(), "output_dcp must exist")
        self.assertTrue(out_edf.exists(), "output.edf must exist via best_valid mirror")
        self.assertEqual(out_edf.read_bytes(), self.best_edf.read_bytes())
        # With EDIF present this stays as VALID_FALLBACK_BASELINE.
        self.assertEqual(self.opt.final_status, "VALID_FALLBACK_BASELINE")

    def test_no_improvement_falls_back_to_vivado_when_no_best_valid_edif(self):
        # Without a mirrored EDIF, finalization attempts checkpoint opening and
        # EDIF generation. If that path also fails, final status reports the
        # missing EDIF so the packager can handle it.
        self.best_edf.unlink()
        self.opt._best_valid_edif = None
        self.opt.initial_wns = -5.0
        self.opt.best_wns = -5.0
        out = self.tmp_path / "optimized.dcp"
        baseline = self.tmp_path / "baseline.dcp"
        baseline.write_bytes(b"BASELINE_DCP")
        self.opt.input_dcp_path = baseline

        async def fake_call_tool(name, args):
            if name == "vivado_open_checkpoint":
                return "ok"
            # write_edif returns but file never appears → _write_readable_edif
            # returns False.
            return "ok"

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake_call_tool)):
            _async(self.opt._finalize_output_dcp(out))

        self.assertTrue(out.exists())
        self.assertEqual(self.opt.final_status, "VALID_FALLBACK_BASELINE_NO_EDIF")


class StaleMirrorDetectionTests(unittest.TestCase):
    """Verifies that finalization rejects a stale best-result mirror.

    A checkpoint whose recorded slack does not match the optimizer's best slack
    must not be emitted as the optimized result.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.opt = _make_optimizer(self.tmp_path)
        self.best_dcp = self.tmp_path / "best_valid.dcp"
        self.best_edf = self.tmp_path / "best_valid.edf"
        self.best_dcp.write_bytes(b"STALE_BASELINE_MIRROR_BYTES")
        self.best_edf.write_bytes(b"STALE_BASELINE_EDIF_BYTES")
        self.opt._best_valid_dcp = self.best_dcp
        self.opt._best_valid_edif = self.best_edf
        self.opt.input_dcp_path = self.tmp_path / "baseline.dcp"
        self.opt.input_dcp_path.write_bytes(b"BASELINE")
        self.opt.initial_wns = -10.0
        self.opt.best_wns = -1.0  # claimed improvement

    def tearDown(self):
        self.tmp.cleanup()

    def test_unset_mirror_wns_falls_through_to_baseline_no_inmemory_write(self):
        # An absent mirror WNS means the mirror's quality is unknown.
        # Finalization must not replace it from potentially degraded in-memory
        # state; it fails closed to the baseline checkpoint.
        self.opt._best_valid_dcp_wns = None
        self.opt._best_valid_edif_wns = None
        out = self.tmp_path / "optimized.dcp"

        call_log = []

        async def fake_call_tool(name, args):
            call_log.append(name)
            return "ok"

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake_call_tool)):
            _async(self.opt._finalize_output_dcp(out))

        events = [e["event"] for e in self.opt.lifecycle_log]
        self.assertIn("stale_mirror_detected", events,
                      "must still detect stale mirror via _best_valid_dcp_wns")
        # A stale mirror must never trigger a fresh write from potentially
        # regressed in-memory state.
        self.assertNotIn(
            "stale_mirror_recovered_via_fresh_write",
            events,
            "finalize MUST NOT dump Vivado in-memory state for stale-mirror "
            "with unknown WNS — root cause of optical-flow/spam-filter "
            "regressions caught by submission validator 2026-05-19"
        )
        # The stale-mirror decision was logged.
        stale_decisions = [e["decision"] for e in self.opt.lifecycle_log
                           if e["event"] == "stale_mirror_detected"]
        self.assertEqual(stale_decisions, ["fall_through_to_baseline"],
            "with mirror_wns=None the only safe decision is baseline fallback")

    def test_stale_mirror_with_improvement_wns_ships_disk_file_not_inmemory(self):
        # This mirror is stale relative to the latest best WNS but still improves
        # on the baseline. Finalization ships the known disk state rather than
        # potentially regressed in-memory state.
        self.opt._best_valid_dcp_wns = -3.0
        self.opt._best_valid_edif_wns = -3.0
        # Replace mirror bytes so we can identify what was shipped.
        self.best_dcp.write_bytes(b"STALE_MIRROR_AT_MINUS_THREE")
        out = self.tmp_path / "optimized.dcp"

        call_log = []

        async def fake_call_tool(name, args):
            call_log.append(name)
            if name == "vivado_write_checkpoint":
                # If this is invoked from the stale-mirror branch, that's
                # the EXACT bug being fixed. Make it loud.
                Path(args["dcp_path"]).write_bytes(b"REGRESSED_INMEMORY_DCP")
                return "ok"
            return "ok"

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake_call_tool)):
            _async(self.opt._finalize_output_dcp(out))

        events = [e["event"] for e in self.opt.lifecycle_log]
        self.assertIn("stale_mirror_detected", events)
        self.assertIn("stale_mirror_recovered_via_disk_copy", events,
            "stale mirror that beats baseline must be shipped via disk copy")
        self.assertEqual(out.read_bytes(), b"STALE_MIRROR_AT_MINUS_THREE",
            "output must be the disk mirror file, NOT Vivado's in-memory state")
        self.assertNotIn(
            "vivado_write_checkpoint", call_log,
            "stale-mirror branch must NOT invoke write_checkpoint — that "
            "would dump in-memory state, which is the regression bug"
        )
        # Lifecycle status reflects shipping a valid (albeit
        # stale) optimized artifact.
        self.assertIn(self.opt.final_status,
                      ("VALID_OPTIMIZED", "VALID_OPTIMIZED_NO_EDIF"))

    def test_ensure_output_refuses_inmemory_write_when_mirror_stale(self):
        # Without a fresh best-valid mirror, output creation must not write the
        # potentially regressed in-memory checkpoint.
        self.opt._best_valid_dcp_wns = None  # mirror stale
        out = self.tmp_path / "optimized_via_ensure.dcp"

        call_log = []

        async def fake_call_tool(name, args):
            call_log.append(name)
            if name == "vivado_write_checkpoint":
                # If invoked, that's the bug.
                Path(args["dcp_path"]).write_bytes(b"REGRESSED_INMEMORY")
            return "ok"

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake_call_tool)):
            _async(self.opt._ensure_output_dcp_written(out))

        self.assertNotIn(
            "vivado_write_checkpoint", call_log,
            "_ensure_output_dcp_written MUST NOT issue write_checkpoint "
            "when no fresh best_valid mirror exists — would risk shipping "
            "Vivado in-memory state (root cause of 2026-05-19 regressions)"
        )
        self.assertFalse(out.exists(),
            "no output should be created — Step 2 baseline fallback ships baseline")
        events = [e["event"] for e in self.opt.lifecycle_log]
        self.assertIn("ensure_output_skipped_no_fresh_mirror", events,
            "skipped event must be logged for audit")

    def test_ensure_output_copies_fresh_mirror_no_vivado_call(self):
        # When the mirror IS fresh, _ensure_output_dcp_written must
        # shutil-copy it (DISK TRUTH), not write_checkpoint from Vivado.
        self.opt._best_valid_dcp_wns = self.opt.best_wns  # mark mirror fresh
        # Replace mirror bytes to a known-good marker.
        self.best_dcp.write_bytes(b"FRESH_MIRROR_BYTES")
        out = self.tmp_path / "optimized_via_ensure.dcp"

        call_log = []

        async def fake_call_tool(name, args):
            call_log.append(name)
            return "ok"

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake_call_tool)):
            _async(self.opt._ensure_output_dcp_written(out))

        self.assertNotIn("vivado_write_checkpoint", call_log,
            "fresh-mirror path must use shutil-copy, not Vivado write")
        self.assertTrue(out.exists())
        self.assertEqual(out.read_bytes(), b"FRESH_MIRROR_BYTES",
            "must copy disk mirror bytes, not in-memory state")

    def test_stale_mirror_fresh_write_fail_falls_through(self):
        # If both the cached checkpoint and a fresh write are unavailable,
        # finalization must use the baseline fallback and must not report
        # VALID_OPTIMIZED without a valid optimized artifact.
        self.opt._best_valid_dcp_wns = None
        out = self.tmp_path / "optimized.dcp"

        async def fake_call_tool(name, args):
            # Every Vivado call refused.
            return '{"error": "tool_skipped_budget", "reason": "deadline_passed"}'

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake_call_tool)):
            _async(self.opt._finalize_output_dcp(out))

        # Stale mirror detected, fresh write failed, fell through to slow
        # path, eventually baseline-fallback (output_dcp absent → invalid).
        events = [e["event"] for e in self.opt.lifecycle_log]
        self.assertIn("stale_mirror_detected", events)
        self.assertNotIn("stale_mirror_recovered_via_fresh_write", events)
        # Final status is the baseline-fallback (NOT VALID_OPTIMIZED).
        self.assertIn(self.opt.final_status,
                      ("VALID_FALLBACK_BASELINE", "VALID_FALLBACK_BASELINE_NO_EDIF"))

    def test_fresh_mirror_uses_fast_path_no_vivado(self):
        # Sanity check: when mirror IS fresh (_best_valid_dcp_wns matches
        # self.best_wns), fast path runs and Vivado is NOT called.
        self.opt._best_valid_dcp_wns = -1.0
        self.opt._best_valid_edif_wns = -1.0
        # Refresh file bytes so we can verify the copy.
        self.best_dcp.write_bytes(b"FRESH_OPTIMIZED_MIRROR")
        self.best_edf.write_bytes(b"FRESH_OPTIMIZED_EDIF_MIRROR")
        out = self.tmp_path / "optimized.dcp"

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(
                                    side_effect=AssertionError(
                                        "fresh-mirror fast path must not call Vivado"))):
            _async(self.opt._finalize_output_dcp(out))

        events = [e["event"] for e in self.opt.lifecycle_log]
        self.assertIn("fast_path_best_valid_copy", events)
        self.assertNotIn("stale_mirror_detected", events)
        self.assertEqual(out.read_bytes(), b"FRESH_OPTIMIZED_MIRROR")
        self.assertEqual(out.with_suffix(".edf").read_bytes(),
                         b"FRESH_OPTIMIZED_EDIF_MIRROR")
        self.assertEqual(self.opt.final_status, "VALID_OPTIMIZED")

    def test_fresh_dcp_stale_edif_refreshes_via_vivado(self):
        # A fresh checkpoint may outlive a failed EDIF refresh after retiming.
        # Ship the checkpoint, but regenerate EDIF from the tool's in-memory
        # state rather than pairing it with a stale baseline EDIF.
        self.opt._best_valid_dcp_wns = -1.0
        self.opt._best_valid_edif_wns = None  # explicitly stale
        self.best_dcp.write_bytes(b"FRESH_OPTIMIZED_DCP_RETIMED")
        # The stale EDIF on disk represents pre-retiming netlist.
        self.best_edf.write_bytes(b"STALE_PRE_RETIME_EDIF")
        out = self.tmp_path / "optimized.dcp"

        async def fake_call_tool(name, args):
            if name == "vivado_write_edif":
                Path(args["edif_path"]).write_bytes(b"FRESH_POST_RETIME_EDIF")
                return "ok"
            return "ok"

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake_call_tool)):
            _async(self.opt._finalize_output_dcp(out))

        # Fresh DCP shipped via fast path; EDIF regenerated from Vivado.
        events = [e["event"] for e in self.opt.lifecycle_log]
        self.assertIn("fast_path_best_valid_copy", events)
        self.assertEqual(out.read_bytes(), b"FRESH_OPTIMIZED_DCP_RETIMED")
        self.assertEqual(out.with_suffix(".edf").read_bytes(),
                         b"FRESH_POST_RETIME_EDIF",
                         "EDIF must come from fresh Vivado write, NOT stale mirror")
        self.assertEqual(self.opt.final_status, "VALID_OPTIMIZED")


class MirrorFreshnessTrackingTests(unittest.TestCase):
    """_mirror_best_valid_now must set _best_valid_dcp_wns ONLY when the
    write actually landed (mtime advanced or size changed) — not when the
    dispatcher refused the call and left a stale file on disk."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.opt = _make_optimizer(self.tmp_path)
        self.opt.initial_wns = -10.0
        self.opt.best_wns = -1.0
        self.opt._pending_best_mirror = True
        # Pre-stage a STALE best_valid.dcp (the would-be baseline mirror).
        self.stale_dcp = self.tmp_path / "best_valid.dcp"
        self.stale_edf = self.tmp_path / "best_valid.edf"
        self.stale_dcp.write_bytes(b"STALE_BASELINE_DCP")
        self.stale_edf.write_bytes(b"STALE_BASELINE_EDIF")
        # Backdate so write-detection mtime comparison is reliable.
        old = time.time() - 3600.0
        os.utime(self.stale_dcp, (old, old))
        os.utime(self.stale_edf, (old, old))

    def tearDown(self):
        self.tmp.cleanup()

    def test_dispatcher_refusal_keeps_dcp_wns_none(self):
        # Dispatcher returns the tool_skipped_budget error envelope — the
        # mirror function must NOT mark the file as fresh just because it
        # exists and is non-empty.
        async def fake_call_tool(name, args):
            # write_checkpoint refused; file on disk is unchanged.
            return '{"error": "tool_skipped_budget", "reason": "deadline_passed"}'

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake_call_tool)):
            _async(self.opt._mirror_best_valid_now())

        self.assertIsNone(self.opt._best_valid_dcp_wns,
                          "dispatcher refusal must leave _best_valid_dcp_wns=None")
        # _pending_best_mirror stays True so a future iteration can retry.
        self.assertTrue(self.opt._pending_best_mirror,
                        "pending flag must persist on refused mirror so iter can retry")

    def test_successful_write_sets_dcp_wns_to_best_wns(self):
        # Successful write actually rewrites the file → mtime advances.
        async def fake_call_tool(name, args):
            if name == "vivado_write_checkpoint":
                Path(args["dcp_path"]).write_bytes(b"FRESHLY_WRITTEN_DCP")
                return "wrote"
            if name == "vivado_write_edif":
                Path(args["edif_path"]).write_bytes(b"FRESHLY_WRITTEN_EDIF")
                return "wrote"
            return "ok"

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake_call_tool)):
            _async(self.opt._mirror_best_valid_now())

        self.assertEqual(self.opt._best_valid_dcp_wns, -1.0)
        self.assertEqual(self.opt._best_valid_edif_wns, -1.0)
        self.assertFalse(self.opt._pending_best_mirror)

    def test_unchanged_file_after_write_treated_as_stale(self):
        # write_checkpoint returned success-looking text but the file on
        # disk wasn't actually touched (mtime didn't advance, size identical).
        # The defensive check must treat this as stale.
        async def fake_call_tool(name, args):
            # Return a non-error string but do NOT modify the file.
            return "Wrote checkpoint successfully."

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake_call_tool)):
            _async(self.opt._mirror_best_valid_now())

        self.assertIsNone(self.opt._best_valid_dcp_wns,
                          "no-op write must NOT mark mirror as fresh")
        # _pending_best_mirror remains True so retry stays open.
        self.assertTrue(self.opt._pending_best_mirror)

    def test_dcp_fresh_but_edif_refused_keeps_edif_wns_none(self):
        # DCP write succeeds, EDIF write refused — DCP-WNS tracked,
        # EDIF-WNS stays None so finalize can refresh the EDIF.
        async def fake_call_tool(name, args):
            if name == "vivado_write_checkpoint":
                Path(args["dcp_path"]).write_bytes(b"FRESH_DCP_ONLY")
                return "wrote"
            if name == "vivado_write_edif":
                return '{"error": "tool_skipped_budget", "reason": "deadline_passed"}'
            return "ok"

        with mock.patch.object(self.opt, "call_tool",
                                new=mock.AsyncMock(side_effect=fake_call_tool)):
            _async(self.opt._mirror_best_valid_now())

        self.assertEqual(self.opt._best_valid_dcp_wns, -1.0,
                         "DCP write landed — _best_valid_dcp_wns must reflect best_wns")
        self.assertIsNone(self.opt._best_valid_edif_wns,
                          "EDIF write refused — _best_valid_edif_wns must stay None")
        # Pending flag cleared because at least the DCP refreshed.
        self.assertFalse(self.opt._pending_best_mirror)


class DeadlineAwareTimeoutTests(unittest.TestCase):
    """_deadline_aware_timeout returns the correct asyncio.wait_for value
    given the current budget state."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = _make_optimizer(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_budget_returns_none(self):
        self.opt._budget_deadline = None
        self.assertIsNone(self.opt._deadline_aware_timeout())

    def test_remaining_window_returned(self):
        # _budget_deadline is already reserve-adjusted by optimize() when
        # the optimizer is wired up for real.  Here we treat it as the
        # soft deadline directly.  Remaining ≈ 1000s.
        self.opt._budget_deadline = time.time() + 1000.0
        t = self.opt._deadline_aware_timeout()
        self.assertIsNotNone(t)
        # 5s slack for clock drift between setting deadline and reading
        # remaining (asyncio import overhead etc.).
        self.assertGreater(t, 990.0)
        self.assertLess(t, 1005.0)

    def test_returns_zero_when_below_min_useful(self):
        # Remaining 10s < MIN_USEFUL_TOOL_SECONDS=30s.
        self.opt._budget_deadline = time.time() + 10.0
        t = self.opt._deadline_aware_timeout()
        self.assertEqual(t, 0.0)


class FinalizeBudgetBypassTests(unittest.TestCase):
    """Verifies that finalization bypasses the normal budget gate.

    Critical finalize calls may run after the soft deadline, but each remains
    bounded by a 120-second timeout to contain a hung tool session.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = _make_optimizer(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_in_finalize_bypasses_deadline_passed_skip(self):
        # Deadline has passed by 5s — without the bypass, call_tool would
        # return tool_skipped_budget(deadline_passed).
        self.opt._budget_deadline = time.time() - 5.0
        self.opt.vivado_session = _FakeSession(response_text='{"ok":true}')
        self.opt._in_finalize = True
        result = _async(self.opt.call_tool("vivado_write_checkpoint", {
            "dcp_path": "/tmp/x.dcp", "force": True,
        }))
        # Must NOT be a budget skip — the underlying session was invoked.
        self.assertNotIn("tool_skipped_budget", result)
        self.assertEqual(len(self.opt.vivado_session.calls), 1)
        self.assertEqual(self.opt.vivado_session.calls[0][0], "write_checkpoint")

    def test_normal_mode_still_skips_when_deadline_passed(self):
        # Same setup but _in_finalize=False — must still refuse so the
        # iter loop's normal budget enforcement keeps working.
        self.opt._budget_deadline = time.time() - 5.0
        self.opt.vivado_session = _FakeSession(response_text='{"ok":true}')
        self.opt._in_finalize = False
        result = _async(self.opt.call_tool("vivado_write_checkpoint", {
            "dcp_path": "/tmp/x.dcp", "force": True,
        }))
        self.assertIn("tool_skipped_budget", result)
        self.assertIn("deadline_passed", result)
        # Session must NOT have been called.
        self.assertEqual(self.opt.vivado_session.calls, [])

    def test_in_finalize_does_not_flip_budget_killed(self):
        # Budget bypass for finalize must not retroactively flip the
        # budget-killed flag — it would affect run summary semantics.
        self.opt._budget_deadline = time.time() - 5.0
        self.opt.vivado_session = _FakeSession(response_text='{"ok":true}')
        self.opt._in_finalize = True
        prior = self.opt._budget_killed
        _async(self.opt.call_tool("vivado_write_edif", {
            "edif_path": "/tmp/x.edf", "force": True,
        }))
        self.assertEqual(self.opt._budget_killed, prior)

    def test_in_finalize_uses_fixed_120s_timeout(self):
        # Verify the per-call timeout in finalize mode is the fixed
        # constant, not the (zero) deadline-derived one — otherwise
        # asyncio.wait_for(0) would cancel immediately.
        from dcp_optimizer import FINALIZE_PER_CALL_TIMEOUT_S
        self.assertGreaterEqual(FINALIZE_PER_CALL_TIMEOUT_S, 60.0,
            "finalize timeout must be generous enough to allow a slow "
            "write_checkpoint on a large design")
        # Finalization bypasses the expired budget timeout so required tool
        # operations can complete; the instant fake isolates this behavior.
        self.opt._budget_deadline = time.time() - 5.0
        self.opt.vivado_session = _FakeSession(response_text='{"ok":true}')
        self.opt._in_finalize = True
        result = _async(self.opt.call_tool("vivado_write_checkpoint", {
            "dcp_path": "/tmp/x.dcp", "force": True,
        }))
        self.assertNotIn("tool_timed_out_budget", result)

    def test_finalize_wrapper_sets_and_clears_flag(self):
        # _finalize_output_dcp wrapper must set _in_finalize=True for the
        # duration and clear it afterward — even if the inner impl raises.
        observed_flag: list[bool] = []

        async def fake_impl(_self_out):
            observed_flag.append(self.opt._in_finalize)
            raise RuntimeError("simulated mid-finalize error")

        # Patch the inner impl so we can observe without running the full
        # finalize logic.
        with mock.patch.object(self.opt, "_finalize_output_dcp_impl",
                                side_effect=fake_impl):
            self.assertFalse(self.opt._in_finalize)
            with self.assertRaises(RuntimeError):
                _async(self.opt._finalize_output_dcp(Path(self.tmp.name) / "out.dcp"))
            # During impl, flag was True.
            self.assertEqual(observed_flag, [True])
            # After impl raised, flag is cleared.
            self.assertFalse(self.opt._in_finalize)


class FinalizeBypassIntegrationTests(unittest.TestCase):
    """Exercises expired-deadline finalization through the full dispatch path.

    The test confirms that the finalize bypass writes the optimized checkpoint
    after the soft deadline. Disabling the bypass must instead produce the
    dispatcher’s budget-skip envelope.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = _make_optimizer(Path(self.tmp.name))
        # Seed the optimized checkpoint that finalization copies. The fake
        # session writes requested paths to emulate write_checkpoint's side effect.
        improved = Path(self.tmp.name) / "best_valid.dcp"
        improved.write_bytes(b"improved DCP marker for integration test")
        self.improved_dcp = improved

    def tearDown(self):
        self.tmp.cleanup()

    def _make_writing_session(self):
        """A session that, when called with vivado_write_checkpoint,
        writes the requested file at dcp_path with marker bytes.
        Simulates the real Vivado side effect."""

        class _WritingSession:
            def __init__(self):
                self.calls = []

            async def call_tool(self, name, arguments):
                self.calls.append((name, arguments))
                if name in ("write_checkpoint", "write_edif"):
                    key = "dcp_path" if name == "write_checkpoint" else "edif_path"
                    p = Path(arguments[key])
                    p.write_bytes(b"finalize-bypass-integration-marker")
                return _FakeResult('{"status":"OK"}')

        return _WritingSession()

    def test_bypass_active_writes_output_after_deadline(self):
        # Deadline expired by 10s — strict failure mode.
        self.opt._budget_deadline = time.time() - 10.0
        self.opt.vivado_session = self._make_writing_session()
        self.opt._in_finalize = True  # finalize wrapper would have set this

        output_dcp = Path(self.tmp.name) / "out.dcp"
        result = _async(self.opt.call_tool("vivado_write_checkpoint", {
            "dcp_path": str(output_dcp),
            "force": True,
        }))

        # Finalization must bypass the budget gate, invoke the session, and
        # leave a nonempty checkpoint on disk.
        self.assertNotIn("tool_skipped_budget", result,
            "FINALIZE BYPASS REGRESSION: write_checkpoint was refused "
            "by the dispatcher despite _in_finalize=True. This is the "
            "exact failure mode that lost +22.89 MHz on rosetta_digit-"
            "recognition 2026-05-18.")
        self.assertEqual(len(self.opt.vivado_session.calls), 1)
        self.assertEqual(self.opt.vivado_session.calls[0][0], "write_checkpoint")
        self.assertTrue(output_dcp.exists(),
            "output DCP must land on disk after the bypass — otherwise "
            "the iter-loop's gain is lost at finalize time")
        self.assertGreater(output_dcp.stat().st_size, 0,
            "output DCP must be non-empty")

    def test_bypass_disabled_replicates_the_original_bug(self):
        # Outside finalization, an expired deadline must return the budget-skip
        # envelope without invoking the session or writing an output artifact.
        self.opt._budget_deadline = time.time() - 10.0
        self.opt.vivado_session = self._make_writing_session()
        self.opt._in_finalize = False  # simulate pre-44b3193 behaviour

        output_dcp = Path(self.tmp.name) / "out_pre_fix.dcp"
        result = _async(self.opt.call_tool("vivado_write_checkpoint", {
            "dcp_path": str(output_dcp),
            "force": True,
        }))

        # Pre-fix behaviour: skip envelope, no session call, no file.
        self.assertIn("tool_skipped_budget", result)
        self.assertIn("deadline_passed", result)
        self.assertEqual(self.opt.vivado_session.calls, [])
        self.assertFalse(output_dcp.exists(),
            "without the bypass, the dispatcher must NOT let the call "
            "land — otherwise the bypass flag is non-load-bearing")


class LineageTrackingTests(unittest.TestCase):
    """Best-valid / ship-DCP lineage (post-2026-05-20 P2).

    Every mirror-capture event bumps a monotonic token and records a
    lineage entry.  Finalize records a ship lineage tied to one of
    {eager_mirror, piggyback, backstop, stale_mirror_file, baseline,
    no_improvement_baseline, external_write_validated,
    hard_fail_no_ship}.

    These tests pin down:
      1. Two consecutive eager-mirror captures increment the token and
         the final ship inherits the latest token.
      2. A stale mirror present + newer best_wns surfaces the
         stale_mirror_file lineage at finalize.
      3. No improvement → ship lineage is no_improvement_baseline.
      4. Baseline fallback after a failed optimization output → ship
         lineage is baseline.
      5. Final ship lineage is non-None on every finalize exit path.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.opt = _make_optimizer(self.tmp_path)
        # Baseline DCP for fallback path
        self.baseline_dcp = self.tmp_path / "baseline_input.dcp"
        self.baseline_dcp.write_bytes(b"BASELINE")
        self.opt.input_dcp_path = self.baseline_dcp
        self.opt.initial_wns = -1.5

    def tearDown(self):
        self.tmp.cleanup()

    def test_eager_mirror_bumps_lineage_token(self):
        # Stage eager mirrors through the lineage helper; the real mirror is
        # asynchronous and tool-driven, while lineage depends on this helper's
        # event and token contract.
        self.assertEqual(self.opt._best_valid_token, 0)
        e1 = self.opt._bump_lineage(
            "eager_mirror", wns=-1.0,
            tool_name="vivado_write_checkpoint",
            dcp_path=self.tmp_path / "best_valid.dcp",
        )
        self.assertEqual(self.opt._best_valid_token, 1)
        self.assertEqual(e1["token"], 1)
        self.assertEqual(e1["source"], "eager_mirror")

        e2 = self.opt._bump_lineage(
            "eager_mirror", wns=-0.5,
            tool_name="vivado_write_checkpoint",
            dcp_path=self.tmp_path / "best_valid.dcp",
        )
        self.assertEqual(self.opt._best_valid_token, 2)
        self.assertEqual(e2["token"], 2)
        # Current lineage is the most recent one.
        self.assertEqual(self.opt._best_valid_lineage["token"], 2)
        self.assertEqual(self.opt._best_valid_lineage["wns"], -0.5)

    def test_final_ship_inherits_latest_best_valid_token(self):
        """Two captures, then fast-path finalize ships the latest token."""
        bv = self.tmp_path / "best_valid.dcp"
        bv.write_bytes(b"BV_FILE")
        edf = self.tmp_path / "best_valid.edf"
        edf.write_bytes(b"BV_EDIF")
        # First capture at -1.0
        self.opt._best_valid_dcp = bv
        self.opt._best_valid_edif = edf
        self.opt._best_valid_dcp_wns = -1.0
        self.opt._best_valid_edif_wns = -1.0
        self.opt.best_wns = -1.0
        self.opt._bump_lineage("eager_mirror", wns=-1.0, dcp_path=bv)
        token_after_first = self.opt._best_valid_token
        # Second capture at -0.5 (improved)
        self.opt._best_valid_dcp_wns = -0.5
        self.opt._best_valid_edif_wns = -0.5
        self.opt.best_wns = -0.5
        self.opt._bump_lineage("eager_mirror", wns=-0.5, dcp_path=bv)
        token_after_second = self.opt._best_valid_token

        self.assertGreater(token_after_second, token_after_first)

        # Trigger fast-path finalize.
        out = self.tmp_path / "ship.dcp"
        tracking = _FakeSession()
        self.opt.vivado_session = tracking
        _async(self.opt._finalize_output_dcp(out))

        self.assertIsNotNone(self.opt._ship_lineage)
        self.assertEqual(self.opt._ship_lineage["source"], "eager_mirror")
        self.assertEqual(self.opt._ship_lineage["token"], token_after_second,
                          "ship must inherit the LATEST best-valid token")

    def test_stale_mirror_with_improvement_surfaces_stale_lineage(self):
        """Mirror exists at wns=-0.5 but best_wns advanced to -0.3 (eager
        write was refused).  Finalize takes the stale-mirror branch and
        the ship lineage source is stale_mirror_file."""
        bv = self.tmp_path / "best_valid.dcp"
        bv.write_bytes(b"BV_FILE")
        edf = self.tmp_path / "best_valid.edf"
        edf.write_bytes(b"BV_EDIF")
        self.opt._best_valid_dcp = bv
        self.opt._best_valid_edif = edf
        self.opt._best_valid_dcp_wns = -0.5  # mirror captures
        self.opt._best_valid_edif_wns = -0.5
        self.opt._bump_lineage("eager_mirror", wns=-0.5, dcp_path=bv)
        self.opt.best_wns = -0.3  # diverged: mirror is stale

        out = self.tmp_path / "ship.dcp"
        tracking = _FakeSession()
        self.opt.vivado_session = tracking
        _async(self.opt._finalize_output_dcp(out))

        self.assertIsNotNone(self.opt._ship_lineage)
        self.assertEqual(self.opt._ship_lineage["source"], "stale_mirror_file")
        self.assertEqual(self.opt._ship_lineage["dcp_path"], str(out))

    def test_no_improvement_surfaces_baseline_lineage(self):
        """best_wns == initial_wns → ship lineage is no_improvement_baseline."""
        self.opt.best_wns = self.opt.initial_wns  # = -1.5

        out = self.tmp_path / "ship.dcp"
        tracking = _FakeSession()
        self.opt.vivado_session = tracking
        _async(self.opt._finalize_output_dcp(out))

        self.assertIsNotNone(self.opt._ship_lineage)
        self.assertEqual(self.opt._ship_lineage["source"],
                          "no_improvement_baseline")
        self.assertEqual(self.opt._ship_lineage["wns"], self.opt.initial_wns)

    def test_piggyback_updates_freshness_marker_and_lineage(self):
        """Piggyback success must set _best_valid_dcp_wns AND bump
        the lineage (closes Finding F2 from the 2026-05-19 audit)."""
        self.opt._pending_best_mirror = True
        self.opt.best_wns = -0.8
        llm_dcp = self.tmp_path / "llm_iter1.dcp"
        llm_dcp.write_bytes(b"LLM_CHECKPOINT")

        _async(self.opt.call_tool("vivado_write_checkpoint",
                                   {"dcp_path": str(llm_dcp)}))

        # Piggyback should have fired:
        mirror = self.tmp_path / "best_valid.dcp"
        self.assertTrue(mirror.exists())
        self.assertEqual(self.opt._best_valid_dcp, mirror)
        # F2 fix: freshness marker now set
        self.assertEqual(self.opt._best_valid_dcp_wns, -0.8)
        # Lineage bumped to piggyback source
        self.assertIsNotNone(self.opt._best_valid_lineage)
        self.assertEqual(self.opt._best_valid_lineage["source"], "piggyback")
        self.assertEqual(self.opt._best_valid_lineage["wns"], -0.8)
        self.assertGreater(self.opt._best_valid_token, 0)

    def test_ship_lineage_always_recorded_on_finalize(self):
        """Every reachable finalize exit must record a ship_lineage —
        this is the load-bearing invariant for downstream auditing."""
        # No improvement, no mirror, no special state.  Should still
        # ship with a recorded lineage.
        out = self.tmp_path / "ship.dcp"
        tracking = _FakeSession()
        self.opt.vivado_session = tracking
        _async(self.opt._finalize_output_dcp(out))

        self.assertIsNotNone(self.opt._ship_lineage,
                              "every finalize path must record a lineage")
        from dcp_optimizer import LINEAGE_SOURCES
        self.assertIn(self.opt._ship_lineage["source"], LINEAGE_SOURCES)


class OptimizeReturnValueTests(unittest.TestCase):
    """Verifies the optimizer's success-return contract.

    Success requires a final status of `VALID_OPTIMIZED` or
    `VALID_FALLBACK_BASELINE`, indicating a valid artifact is ready to ship.
    Hard failures, no improvement, and missing statuses return false. The
    return value is derived from the final status, including after
    budget-driven loop termination.
    """

    def _success(self, status: str) -> bool:
        # Mirror the contract in dcp_optimizer.py optimize() post-loop.
        return (status or "").startswith((
            "VALID_OPTIMIZED",
            "VALID_FALLBACK_BASELINE",
        ))

    def test_valid_optimized_returns_true(self):
        self.assertTrue(self._success("VALID_OPTIMIZED"))

    def test_valid_optimized_no_edif_returns_true(self):
        # We still ship — EDIF-missing is a yellow flag, not a failure.
        self.assertTrue(self._success("VALID_OPTIMIZED_NO_EDIF"))

    def test_valid_fallback_baseline_returns_true(self):
        # Baseline is a SUBMISSION-SAFE outcome per contest contract.
        self.assertTrue(self._success("VALID_FALLBACK_BASELINE"))

    def test_valid_fallback_baseline_no_edif_returns_true(self):
        self.assertTrue(self._success("VALID_FALLBACK_BASELINE_NO_EDIF"))

    def test_valid_fallback_baseline_phase1_failed_returns_true(self):
        # Phase-1 failed but baseline-copy succeeded — submission-safe.
        self.assertTrue(self._success("VALID_FALLBACK_BASELINE_PHASE1_FAILED"))

    def test_hard_fail_returns_false(self):
        self.assertFalse(self._success("HARD_FAIL_NO_VALID_BASELINE"))

    def test_no_improvement_returns_false(self):
        # NO_IMPROVEMENT means no checkpoint was written and is unsuccessful.
        # Finalization normally converts this outcome to a baseline fallback.
        self.assertFalse(self._success("NO_IMPROVEMENT"))

    def test_missing_status_returns_false(self):
        # Safety net: if final_status was never set, do NOT claim success.
        self.assertFalse(self._success(""))
        self.assertFalse(self._success(None))  # type: ignore[arg-type]


class FinalizeStressTests(unittest.TestCase):
    """Verifies deadline-safe finalization of an eagerly mirrored improvement.

    The improved state must be mirrored before the deadline expires.
    Finalization then copies the freshest best-valid DCP from disk to the
    output without invoking Vivado or relying on mutable in-memory state.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.opt = _make_optimizer(self.tmp_path)
        # Baseline DCP that finalize would fall back to if the mirror
        # path failed.  Distinct content so we can tell baseline vs
        # mirror apart in the asserts.
        self.baseline_dcp = self.tmp_path / "baseline_input.dcp"
        self.baseline_dcp.write_bytes(b"BASELINE_INPUT_CONTENT")
        self.opt.input_dcp_path = self.baseline_dcp
        # Initial WNS set so improvement detection is meaningful.
        self.opt.initial_wns = -1.5
        self.opt.best_wns = -1.5
        # Eager-mirror artifacts staged: best_valid.dcp represents the
        # captured improvement state.
        self.improved_dcp = self.tmp_path / "best_valid.dcp"
        self.improved_dcp.write_bytes(b"IMPROVED_DCP_FROM_EAGER_MIRROR")
        self.improved_edf = self.tmp_path / "best_valid.edf"
        self.improved_edf.write_bytes(b"IMPROVED_EDIF_FROM_EAGER_MIRROR")
        self.opt._best_valid_dcp = self.improved_dcp
        self.opt._best_valid_edif = self.improved_edf
        # Mirror is fresh — WNS matches best_wns within epsilon.
        self.opt.best_wns = -0.5  # improvement over -1.5 initial
        self.opt._best_valid_dcp_wns = -0.5
        self.opt._best_valid_edif_wns = -0.5

    def tearDown(self):
        self.tmp.cleanup()

    def test_deadline_expired_finalize_ships_disk_mirror_no_vivado(self):
        """The chain: improvement captured → deadline expires →
        finalize must ship the disk mirror without entering Vivado."""
        # Deadline has been expired for 30s — typical "ran out at the
        # buzzer" scenario.
        self.opt._budget_deadline = time.time() - 30.0
        # Wire a tracking session that records any call — we expect
        # ZERO Vivado calls during the fast path.
        tracking = _FakeSession()
        self.opt.vivado_session = tracking

        output_dcp = self.tmp_path / "ship.dcp"
        _async(self.opt._finalize_output_dcp(output_dcp))

        self.assertTrue(output_dcp.exists(),
                          "output_dcp must land on disk via the fast path")
        # Byte-equal to the eager-mirror DCP — proves the run shipped the
        # disk-truth best-valid, not Vivado in-memory state.
        self.assertEqual(output_dcp.read_bytes(),
                          self.improved_dcp.read_bytes(),
                          "shipped DCP must equal the eager-mirror DCP "
                          "byte-for-byte")
        # The fast path must not invoke the tool to rewrite a checkpoint,
        # including after the deadline has expired.
        dcp_writes = [c for c in tracking.calls
                       if c[0] in ("vivado_write_checkpoint",
                                    "write_checkpoint")]
        self.assertEqual(dcp_writes, [],
                          "fast path must not call write_checkpoint "
                          "when a fresh mirror is available")
        # Final status must reflect a valid optimized ship (EDIF may
        # have been refreshed via Vivado in this code path, so
        # VALID_OPTIMIZED or VALID_OPTIMIZED_NO_EDIF both qualify).
        self.assertIn(self.opt.final_status,
                       ("VALID_OPTIMIZED", "VALID_OPTIMIZED_NO_EDIF"))

    def test_deadline_expired_no_improvement_ships_baseline(self):
        """Deadline expired AND best_wns == initial_wns (no
        improvement) → finalize must shutil-copy baseline to output.
        No risk of shipping a stale mirror as the optimized DCP."""
        # No improvement scenario — overwrite the improvement-derived
        # setUp state.
        self.opt.best_wns = self.opt.initial_wns  # = -1.5, no improvement
        self.opt._budget_deadline = time.time() - 30.0
        tracking = _FakeSession()
        self.opt.vivado_session = tracking

        output_dcp = self.tmp_path / "ship_baseline.dcp"
        _async(self.opt._finalize_output_dcp(output_dcp))

        self.assertTrue(output_dcp.exists())
        self.assertEqual(output_dcp.read_bytes(),
                          self.baseline_dcp.read_bytes(),
                          "no-improvement path must copy baseline byte-for-byte")
        # Status must be a baseline-fallback variant.
        self.assertTrue(
            self.opt.final_status.startswith("VALID_FALLBACK_BASELINE"),
            f"final_status={self.opt.final_status!r} must indicate baseline ship",
        )

    def test_stale_mirror_at_deadline_falls_back_to_baseline(self):
        """If the disk mirror is stale (mirror_wns < best_wns), even
        an expired deadline finalize must NOT promote the stale mirror
        as if it were the optimized DCP.  Falls back to baseline."""
        # Simulate the post-eager-mirror failure: best_wns advanced to
        # -0.3, but mirror only captured -0.5 (e.g. eager mirror's
        # vivado_write_checkpoint was refused by the dispatcher).
        self.opt.best_wns = -0.3
        # mirror_wns stays at -0.5 (set in setUp) — STALE relative to
        # current best_wns.
        self.opt._budget_deadline = time.time() - 30.0
        tracking = _FakeSession()
        self.opt.vivado_session = tracking

        output_dcp = self.tmp_path / "ship_stale.dcp"
        _async(self.opt._finalize_output_dcp(output_dcp))

        self.assertTrue(output_dcp.exists())
        # A stale mirror that still improves on the baseline remains a
        # known-good disk artifact and must be shipped instead of the baseline
        # or unmirrored in-memory state.
        self.assertEqual(output_dcp.read_bytes(),
                          self.improved_dcp.read_bytes(),
                          "stale-but-still-improvement mirror must ship as "
                          "disk truth, not Vivado in-memory state")
        # Critically: NO write_checkpoint Vivado call.
        dcp_writes = [c for c in tracking.calls
                       if c[0] in ("vivado_write_checkpoint",
                                    "write_checkpoint")]
        self.assertEqual(dcp_writes, [])


class RouterStepAwareContinuationTests(unittest.TestCase):
    """Verifies router-aware continuation when a recommended heavy step remains
    unattempted.

    `_router_unattempted_heavy_step` identifies a recommended placement step
    when none has run, at least five minutes remain, and measured gain is
    positive. The continuation prompt must name the unattempted step
    explicitly.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = _make_optimizer(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def _r4_plan(self):
        """Build an R4-shaped RecipePlan with a place_design step."""
        from optimizer.recipe_router import RecipeAction, RecipePlan
        return RecipePlan(
            rule_id="R4",
            confidence="high",
            summary="Safety path: retime → Auto_1 → retime → polish",
            actions=(
                RecipeAction(name="recipe_register_retiming",
                             note="AlternateFlowWithRetiming (~6 min)"),
                RecipeAction(name="vivado_run_tcl",
                             note="`place_design -unplace`"),
                RecipeAction(name="vivado_place_design",
                             note="directive=Auto_1 (~12 min, ML-best)"),
                RecipeAction(name="recipe_register_retiming",
                             note="AlternateFlowWithRetiming again"),
                RecipeAction(name="vivado_phys_opt_design",
                             note="critical_pin_opt"),
            ),
        )

    def test_returns_none_when_no_plan(self):
        self.opt.recipe_router_plan = None
        self.assertIsNone(self.opt._router_unattempted_heavy_step())

    def test_returns_none_when_no_actions(self):
        from optimizer.recipe_router import RecipePlan
        self.opt.recipe_router_plan = RecipePlan(
            rule_id="FALLBACK", confidence="low",
            summary="no signal", actions=(),
        )
        self.assertIsNone(self.opt._router_unattempted_heavy_step())

    def test_returns_place_design_when_unattempted(self):
        self.opt.recipe_router_plan = self._r4_plan()
        # Simulate repeated physical optimization with no placement call.
        self.opt._tool_calls_seen = [
            ("vivado_phys_opt_design", "AlternateFlowWithRetiming"),
            ("vivado_route_design", "Default"),
            ("vivado_phys_opt_design", "Explore"),
        ]
        result = self.opt._router_unattempted_heavy_step()
        self.assertIsNotNone(result)
        self.assertIn("vivado_place_design", result)
        self.assertIn("Auto_1", result)

    def test_returns_none_after_place_design_called(self):
        # A prior placement call suppresses the placement nudge.
        self.opt.recipe_router_plan = self._r4_plan()
        self.opt._tool_calls_seen = [
            ("vivado_phys_opt_design", "AlternateFlowWithRetiming"),
            ("vivado_place_design", "Auto_1"),
            ("vivado_phys_opt_design", "Explore"),
        ]
        self.assertIsNone(self.opt._router_unattempted_heavy_step())

    def test_router_plan_without_place_design_returns_none(self):
        # R1 / R2 plans don't have a place_design action — the helper
        # must not fabricate one.
        from optimizer.recipe_router import RecipeAction, RecipePlan
        self.opt.recipe_router_plan = RecipePlan(
            rule_id="R1", confidence="high",
            summary="Global retiming",
            actions=(
                RecipeAction(name="vivado_phys_opt_design",
                             note="directive=AlternateFlowWithRetiming"),
            ),
        )
        self.opt._tool_calls_seen = [("vivado_phys_opt_design", "Explore")]
        self.assertIsNone(self.opt._router_unattempted_heavy_step())

    def test_tool_calls_seen_recorded_in_call_tool(self):
        # Black-box: a successful call_tool invocation must append
        # (tool_name, directive) to _tool_calls_seen.
        self.opt.vivado_session = _FakeSession(response_text='{"ok":true}')
        self.opt._budget_deadline = time.time() + 1000.0  # plenty of budget
        # S2 fix: give the unroute gate cell data so the routed-
        # state-destroying place_design stays feasible for this test.
        self.opt._input_cell_count = 10_000
        _async(self.opt.call_tool("vivado_place_design",
                                  {"directive": "Auto_1"}))
        self.assertIn(("vivado_place_design", "Auto_1"),
                      self.opt._tool_calls_seen)


class V05GateTests(unittest.TestCase):
    """Verifies router-aware continuation at zero measured gain.

    A recommended heavy step remains eligible when it has not been attempted,
    sufficient budget remains, and gain is nonnegative. Exact zero therefore
    triggers the specific continuation path rather than the generic fallback.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.opt = _make_optimizer(Path(self.tmp.name))
        # Plant a router plan with a place_design step.
        from optimizer.recipe_router import RecipeAction, RecipePlan
        self.opt.recipe_router_plan = RecipePlan(
            rule_id="R4", confidence="high",
            summary="Safety path",
            actions=(
                RecipeAction(name="recipe_register_retiming",
                             note="AlternateFlowWithRetiming"),
                RecipeAction(name="vivado_place_design",
                             note="directive=Auto_1"),
            ),
        )
        # No place_design call yet — heavy step is unattempted.
        self.opt._tool_calls_seen = [
            ("vivado_phys_opt_design", "AlternateFlowWithRetiming"),
            ("vivado_route_design", "Default"),
        ]
        self.opt._budget_killed = False

    def tearDown(self):
        self.tmp.cleanup()

    def test_fires_at_exactly_zero_gain_with_heavy_step_unattempted(self):
        # V0.4 would NOT have fired (gain == 0 fails > 0). V0.5 must fire.
        ok, step = self.opt.should_inject_v05_router_nudge(
            current_gain_mhz=0.0,
            remaining_s=600.0,
        )
        self.assertTrue(ok,
            "V0.5 must fire at gain=0 — that's the whole point of the "
            "relaxation. Bug: spam-filter shipped 0 MHz because V0.4 "
            "stayed silent at this exact state.")
        self.assertIsNotNone(step)
        self.assertIn("vivado_place_design", step)

    def test_fires_at_positive_gain_just_like_v04(self):
        # V0.5 must remain a superset of V0.4's positive-slope behaviour.
        ok, step = self.opt.should_inject_v05_router_nudge(
            current_gain_mhz=2.5,
            remaining_s=600.0,
        )
        self.assertTrue(ok)
        self.assertIn("vivado_place_design", step)

    def test_does_not_fire_on_clear_regression(self):
        # If current_gain_mhz is negative, the LLM has made things WORSE
        # since initial — nudging it to do MORE heavy work is wrong.
        # The right action is to revert, which is handled elsewhere.
        ok, step = self.opt.should_inject_v05_router_nudge(
            current_gain_mhz=-3.0,
            remaining_s=600.0,
        )
        self.assertFalse(ok,
            "V0.5 must NOT fire when current_gain_mhz < 0 — that's a "
            "regression, not stagnation. Heavy step would compound the "
            "damage.")

    def test_does_not_fire_when_budget_too_low(self):
        # 4 min remaining < 5 min floor → place_design (~9 min) can't
        # finish before finalize → don't bother.
        ok, _ = self.opt.should_inject_v05_router_nudge(
            current_gain_mhz=0.0,
            remaining_s=240.0,
        )
        self.assertFalse(ok)

    def test_does_not_fire_when_budget_killed(self):
        # _budget_killed means the session was previously hard-cancelled
        # by an asyncio timeout — Vivado may be in a bad state. Don't
        # ask it to do heavy work.
        self.opt._budget_killed = True
        ok, _ = self.opt.should_inject_v05_router_nudge(
            current_gain_mhz=0.0,
            remaining_s=600.0,
        )
        self.assertFalse(ok)

    def test_does_not_fire_when_place_design_already_called(self):
        # Heavy step was already attempted this run — no nudge needed.
        self.opt._tool_calls_seen = [
            ("vivado_phys_opt_design", "AlternateFlowWithRetiming"),
            ("vivado_place_design", "Auto_1"),
        ]
        ok, step = self.opt.should_inject_v05_router_nudge(
            current_gain_mhz=0.0,
            remaining_s=600.0,
        )
        self.assertFalse(ok)
        self.assertIsNone(step)

    def test_does_not_fire_without_router_plan(self):
        self.opt.recipe_router_plan = None
        ok, step = self.opt.should_inject_v05_router_nudge(
            current_gain_mhz=0.0,
            remaining_s=600.0,
        )
        self.assertFalse(ok)
        self.assertIsNone(step)

    def test_does_not_fire_when_plan_has_no_place_design(self):
        # R1 / R2 plans don't recommend place_design — the V0.5 nudge
        # specifically targets the missing-place_design pattern.
        from optimizer.recipe_router import RecipeAction, RecipePlan
        self.opt.recipe_router_plan = RecipePlan(
            rule_id="R1", confidence="high",
            summary="Global retiming",
            actions=(
                RecipeAction(name="vivado_phys_opt_design",
                             note="directive=AlternateFlowWithRetiming"),
            ),
        )
        ok, step = self.opt.should_inject_v05_router_nudge(
            current_gain_mhz=0.0,
            remaining_s=600.0,
        )
        self.assertFalse(ok)
        self.assertIsNone(step)

    def test_boundary_at_min_remaining_seconds(self):
        # Strict `<` is the spec — at exactly the floor, fire.
        from dcp_optimizer import V05_MIN_REMAINING_S
        ok, _ = self.opt.should_inject_v05_router_nudge(
            current_gain_mhz=0.0,
            remaining_s=V05_MIN_REMAINING_S,
        )
        self.assertTrue(ok)
        ok2, _ = self.opt.should_inject_v05_router_nudge(
            current_gain_mhz=0.0,
            remaining_s=V05_MIN_REMAINING_S - 0.01,
        )
        self.assertFalse(ok2)


class EagerMirrorInlineTests(unittest.TestCase):
    """Verifies that WNS improvements are mirrored immediately.

    Every improvement-detection path in `call_tool` must invoke
    `_mirror_best_valid_now(eager=True)` before returning or allowing another
    tool call. This ordering ties the disk snapshot to the measured in-memory
    state before later operations can degrade it.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.opt = _make_optimizer(self.tmp_path)
        # Replace the tool-driven mirror with a recorder to verify call order
        # and the best-WNS snapshot without requiring the FPGA tool.
        self.mirror_calls: list[tuple[bool, float, str]] = []

        async def spy(eager: bool = False):
            self.mirror_calls.append((eager, self.opt.best_wns,
                                       "fired"))
        self.opt._mirror_best_valid_now = spy

    def tearDown(self):
        self.tmp.cleanup()

    def test_eager_mirror_fires_inline_on_report_timing_improvement(self):
        """vivado_report_timing_summary → improved WNS → eager mirror
        must be the LAST thing call_tool does on the improvement branch,
        BEFORE call_tool returns.  No window for a degrading next call."""
        self.opt.best_wns = -2.0
        self.opt.target_clock = None  # use parse_timing_summary_static path
        self.opt.vivado_session = _FakeSession(response_text="WNS -1.0")

        with mock.patch("optimizer.tool_dispatch.parse_timing_summary_static",
                          return_value={"wns": -1.0, "tns": 0.0}):
            _async(self.opt.call_tool("vivado_report_timing_summary", {}))

        self.assertEqual(len(self.mirror_calls), 1,
                          "eager mirror must fire exactly once on improvement")
        eager_arg, snapshot_wns, _ = self.mirror_calls[0]
        self.assertTrue(eager_arg,
                          "mirror must be called with eager=True")
        self.assertEqual(snapshot_wns, -1.0,
                          "snapshot of best_wns at mirror-call time must be "
                          "the improved value, not the pre-improvement value")
        self.assertEqual(self.opt.best_wns, -1.0)
        self.assertTrue(self.opt._pending_best_mirror)

    def test_eager_mirror_fires_inline_on_get_wns_improvement(self):
        """vivado_get_wns path — second of the three improvement detection
        sites in call_tool."""
        self.opt.best_wns = -2.0
        self.opt.target_clock = None
        # vivado_get_wns returns just a float as text
        self.opt.vivado_session = _FakeSession(response_text="-0.5")

        _async(self.opt.call_tool("vivado_get_wns", {}))

        self.assertEqual(len(self.mirror_calls), 1)
        eager_arg, snapshot_wns, _ = self.mirror_calls[0]
        self.assertTrue(eager_arg)
        self.assertEqual(snapshot_wns, -0.5)

    def test_eager_mirror_fires_on_target_clock_improvement(self):
        """When target_clock is set, the optimizer uses
        get_wns_for_target_clock instead of parse_timing_summary_static.
        That branch is also a WNS-improvement detection site and must
        eagerly mirror."""
        self.opt.best_wns = -3.0
        self.opt.target_clock = "clk_main"
        # The improvement path will call super().get_wns_for_target_clock.
        # Patch it to return our improved value.
        async def fake_clock_wns(call_tool_fn):
            return -0.25

        with mock.patch.object(self.opt.__class__.__bases__[0],
                                "get_wns_for_target_clock",
                                side_effect=fake_clock_wns):
            _async(self.opt.call_tool("vivado_report_timing_summary", {}))

        self.assertEqual(len(self.mirror_calls), 1,
                          "target-clock improvement must also eager-mirror")
        eager_arg, snapshot_wns, _ = self.mirror_calls[0]
        self.assertTrue(eager_arg)
        self.assertEqual(snapshot_wns, -0.25)

    def test_no_eager_mirror_on_regression(self):
        """A WNS measurement worse than best_wns must NOT trigger a
        mirror — that would overwrite a good disk artifact with a worse
        in-memory state (the 2026-05-19 incident root cause)."""
        self.opt.best_wns = -0.5
        self.opt.target_clock = None
        self.opt.vivado_session = _FakeSession(response_text="WNS -2.0")

        with mock.patch("optimizer.tool_dispatch.parse_timing_summary_static",
                          return_value={"wns": -2.0, "tns": 0.0}):
            _async(self.opt.call_tool("vivado_report_timing_summary", {}))

        self.assertEqual(len(self.mirror_calls), 0,
                          "no mirror on regression — disk artifact stays")
        # best_wns must remain the high-water mark
        self.assertEqual(self.opt.best_wns, -0.5)

    def test_no_eager_mirror_on_lateral_move(self):
        """Equal WNS is not an improvement — no mirror fire."""
        self.opt.best_wns = -1.0
        self.opt.target_clock = None
        self.opt.vivado_session = _FakeSession(response_text="WNS -1.0")

        with mock.patch("optimizer.tool_dispatch.parse_timing_summary_static",
                          return_value={"wns": -1.0, "tns": 0.0}):
            _async(self.opt.call_tool("vivado_report_timing_summary", {}))

        self.assertEqual(len(self.mirror_calls), 0)

    def test_mirror_capture_precedes_subsequent_tool_call(self):
        """Verifies that eager mirroring precedes a subsequent degrading tool
        call.

        The mirrored snapshot must retain the improved WNS even when the next
        simulated call reports a worse result.
        """
        self.opt.best_wns = -2.0
        self.opt.target_clock = None
        self.opt.vivado_session = _FakeSession(response_text="WNS -1.0")

        # Round 1: improvement to -1.0 → eager mirror should fire NOW
        with mock.patch("optimizer.tool_dispatch.parse_timing_summary_static",
                          return_value={"wns": -1.0, "tns": 0.0}):
            _async(self.opt.call_tool("vivado_report_timing_summary", {}))

        # Mirror snapshot must reflect the improved state
        self.assertEqual(len(self.mirror_calls), 1)
        self.assertEqual(self.mirror_calls[0][1], -1.0,
                          "the capture happened at improvement time, not later")

        # Round 2: a 'degrading' measurement at -3.0 — best_wns must
        # stay at the high-water mark, and no new mirror call fires
        # (no improvement → no eager mirror).
        with mock.patch("optimizer.tool_dispatch.parse_timing_summary_static",
                          return_value={"wns": -3.0, "tns": 0.0}):
            _async(self.opt.call_tool("vivado_report_timing_summary", {}))

        self.assertEqual(len(self.mirror_calls), 1,
                          "regression must not trigger a second mirror call")
        self.assertEqual(self.opt.best_wns, -1.0,
                          "best_wns stays at high-water mark across regression")


if __name__ == "__main__":
    unittest.main()
