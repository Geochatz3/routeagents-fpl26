"""Submission-safety tests for the optimizer's failure-mode handling.

The contest contract: every benchmark must produce a valid output DCP, even
when optimization fails entirely.  These tests pin down the recovery paths in
`dcp_optimizer.py` so they don't regress.

We don't spawn Vivado/RapidWright/MCP — we mock the call surface and verify
the lifecycle decisions.
"""
from __future__ import annotations

import asyncio
import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import DCPOptimizer


def _async(coro):
    """Run an async coroutine and return the result (test-helper).

    Use asyncio.run() rather than get_event_loop() — Python 3.12 deprecates
    get_event_loop() outside an active loop, which trips on `unittest
    discover` where earlier tests leave the implicit loop closed.
    """
    return asyncio.run(coro)


class _FakeBaselineDCP:
    """Builds a tiny 'baseline' DCP file we can copy from in tests.

    Real DCPs are PKZip archives; for the submission-safety paths we test
    here, the optimizer only does shutil.copy on these files — content is
    irrelevant.  We write a non-empty byte string so the size check passes.
    """

    def __init__(self, tmpdir: Path):
        self.path = tmpdir / "baseline.dcp"
        self.path.write_bytes(b"DCP_BASELINE_TEST_FIXTURE")


class PhaseOneFailureTests(unittest.TestCase):
    """Phase 1 mandatory step fails (e.g., open_checkpoint times out)
    → optimize() must still leave a baseline-copy DCP at output_dcp.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name)
        self.baseline = _FakeBaselineDCP(self.run_dir)
        self.output_dcp = self.run_dir / "optimized.dcp"
        self.opt = DCPOptimizer(api_key="test", run_dir=self.run_dir)
        self.opt.input_dcp_path = self.baseline.path

    def tearDown(self):
        self.tmp.cleanup()

    def test_phase1_failure_emits_baseline_copy_via_optimize(self):
        """When perform_initial_analysis raises, optimize() must still copy
        the baseline DCP to output_dcp before returning False.
        """
        async def boom(*_a, **_kw):
            raise RuntimeError("simulated Phase 1 timeout")

        # Replace the analysis step with one that always raises.  Also stub
        # call_tool so the EDIF best-effort attempt doesn't blow up the test.
        with mock.patch.object(self.opt, "perform_initial_analysis", side_effect=boom), \
             mock.patch.object(self.opt, "call_tool",
                               new=mock.AsyncMock(side_effect=Exception("vivado dead"))):
            result = _async(self.opt.optimize(self.baseline.path, self.output_dcp))

        self.assertFalse(result, "optimize() should return False on Phase 1 failure")
        self.assertTrue(self.output_dcp.exists(),
                        "output_dcp MUST exist even when Phase 1 fails")
        self.assertGreater(self.output_dcp.stat().st_size, 0,
                           "output_dcp must not be zero-byte")
        self.assertEqual(self.opt.final_status, "VALID_FALLBACK_BASELINE_PHASE1_FAILED")
        # Lifecycle log should record the failure.
        events = [e["event"] for e in self.opt.lifecycle_log]
        self.assertIn("phase1_failed_baseline_copied", events)

    def test_phase1_failure_with_missing_baseline_hard_fails_cleanly(self):
        """If even the baseline is missing, final_status must be HARD_FAIL
        and we must not produce a zero-byte phantom DCP.
        """
        # Point at a baseline that doesn't exist.
        self.opt.input_dcp_path = self.run_dir / "no_such_baseline.dcp"

        async def boom(*_a, **_kw):
            raise RuntimeError("simulated phase 1 failure")

        with mock.patch.object(self.opt, "perform_initial_analysis", side_effect=boom), \
             mock.patch.object(self.opt, "call_tool",
                               new=mock.AsyncMock(side_effect=Exception("nope"))):
            result = _async(self.opt.optimize(self.opt.input_dcp_path, self.output_dcp))

        self.assertFalse(result)
        self.assertEqual(self.opt.final_status, "HARD_FAIL_NO_VALID_BASELINE")
        # No phantom output should be left behind.
        self.assertFalse(self.output_dcp.exists())


class LifecycleStatusEnumTests(unittest.TestCase):
    """Pin the lifecycle status strings — downstream tooling and the
    submission report rely on these exact identifiers.
    """

    def test_known_status_strings(self):
        opt = DCPOptimizer(api_key="test")
        # Just touch the attribute path to ensure the field exists.
        self.assertIsNone(opt.final_status)
        self.assertEqual(opt.lifecycle_log, [])

        known = {
            "VALID_OPTIMIZED",
            "VALID_OPTIMIZED_NO_EDIF",
            "VALID_FALLBACK_BASELINE",
            "VALID_FALLBACK_BASELINE_PHASE1_FAILED",
            "HARD_FAIL_NO_VALID_BASELINE",
        }
        # The optimizer must accept any of these strings without complaint
        # (the values are written by various recovery paths in optimize() /
        # _finalize_output_dcp() — we test them set/read here).
        for s in known:
            opt.final_status = s
            self.assertEqual(opt.final_status, s)


if __name__ == "__main__":
    unittest.main()
