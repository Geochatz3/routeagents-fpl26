"""Tests that ILS seeds are detached from the banked best-valid checkpoint.

The banked mirror must remain byte-for-byte unchanged while ILS runs because
emergency recovery validates it against its recorded size. Both seed paths
therefore pass a writable copy to `run_ils_polish`.
"""
from __future__ import annotations

import asyncio
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dcp_optimizer as dopt  # noqa: E402
from optimizer.ils_polish import ILSPolishConfig  # noqa: E402

MIRROR_BYTES = b"PK\x03\x04" + b"banked-best-valid-mirror" * 512
ACCEPT_BYTES = MIRROR_BYTES + b"an-ILS-accept-rewrote-this-file" * 64


class _Stop(Exception):
    """Raised by the stubbed run_ils_polish to end the body at the call site."""


class _Stub:
    """Only the attributes `_ils_polish_body` touches before it seeds."""

    def __init__(self, tmp: Path, mirror: Path):
        self._best_valid_dcp = mirror
        self._ils_polish_cfg = ILSPolishConfig()
        self.run_dir = tmp
        self.best_wns = -0.90
        self.initial_wns = -1.20
        self.tool_call_details = []
        self.target_clock = None

    def __getattr__(self, name):
        # The seed decision completes before these stubs are reached.
        # Later state remains unset so the fixture cannot influence the path
        # under test.
        return None

    # Reserve plumbing the body calls after seeding; neutral values so the
    # deadline arithmetic runs without steering the seed decision.
    def _polish_reserve_armed_s(self):
        return 0.0

    def _release_polish_reserve(self, *_a, **_kw):
        return None

    async def call_tool(self, *_a, **_kw):
        # No Vivado here: the body only uses call_tool for the pre-ILS restart
        # and an optional snapshot write, neither of which this test exercises.
        return {"content": [{"type": "text", "text": "ok"}]}


def _run_body_and_capture(tmp: Path, mirror: Path, seed_copy_env):
    """Run the REAL body; return the seed path it handed to run_ils_polish."""
    captured = {}

    async def _fake_run_ils_polish(*_a, **kw):
        captured["seed"] = kw.get("best_dcp_path")
        raise _Stop()

    stub = _Stub(tmp, mirror)
    body = types.MethodType(dopt.DCPOptimizer._ils_polish_body, stub)
    try:
        asyncio.run(body(_fake_run_ils_polish))
    except _Stop:
        pass
    return captured.get("seed")


class SeedCopyTests(unittest.TestCase):
    def setUp(self):
        self._saved = dopt.os.environ.get("FPL26_ILS_SEED_COPY")

    def tearDown(self):
        if self._saved is None:
            dopt.os.environ.pop("FPL26_ILS_SEED_COPY", None)
        else:
            dopt.os.environ["FPL26_ILS_SEED_COPY"] = self._saved

    def _mirror(self, tmp: Path) -> Path:
        m = tmp / "best_valid.dcp"
        m.write_bytes(MIRROR_BYTES)
        return m

    def test_default_on_never_hands_the_ils_the_banked_mirror(self):
        """THE FIX. The seed must be a different file from the banked mirror."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            mirror = self._mirror(tmp)
            dopt.os.environ.pop("FPL26_ILS_SEED_COPY", None)   # default
            seed = _run_body_and_capture(tmp, mirror, None)
            self.assertIsNotNone(seed, "body never reached run_ils_polish")
            self.assertNotEqual(
                str(seed), str(mirror),
                "the ILS was handed the banked mirror itself — its accept path "
                "would rewrite it in place")
            # Simulate an ILS accept writing to the seed it was given.
            Path(seed).write_bytes(ACCEPT_BYTES)
            self.assertEqual(
                mirror.read_bytes(), MIRROR_BYTES,
                "the banked mirror changed under an ILS accept; the emergency "
                "size-integrity gate would distrust it and ship the baseline")

    def test_kill_switch_restores_the_old_in_place_behaviour(self):
        """The defect itself, pinned — so the fix is shown to be load-bearing.

        A test that passes with and without the change proves nothing; this is
        the negative control for the one above.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            mirror = self._mirror(tmp)
            dopt.os.environ["FPL26_ILS_SEED_COPY"] = "0"
            seed = _run_body_and_capture(tmp, mirror, "0")
            self.assertIsNotNone(seed, "body never reached run_ils_polish")
            self.assertEqual(
                str(seed), str(mirror),
                "kill switch did not restore the earlier in-place path")
            Path(seed).write_bytes(ACCEPT_BYTES)
            self.assertNotEqual(
                mirror.read_bytes(), MIRROR_BYTES,
                "with the kill switch on, an accept MUST corrupt the banked "
                "size reference — that is the defect this flag exists to undo")

    def test_seed_copy_is_byte_identical_to_the_mirror(self):
        """The copy must be the same design, or the ILS starts from elsewhere."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            mirror = self._mirror(tmp)
            dopt.os.environ.pop("FPL26_ILS_SEED_COPY", None)
            seed = _run_body_and_capture(tmp, mirror, None)
            self.assertEqual(Path(seed).read_bytes(), MIRROR_BYTES)


if __name__ == "__main__":
    unittest.main()
