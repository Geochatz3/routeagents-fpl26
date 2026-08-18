"""Tests for `_atomic_copy` — the never-fail output-write hardening.

The contest scores the output DCP.  A SIGKILL / budget-kill / power loss
mid-copy must never leave a truncated DCP at the scored path: a partial
DCP is an invalid submission (score 0).  `_atomic_copy` writes to a temp
file in the destination's own directory and `os.replace`s it onto the
target, so a reader observes EITHER the prior contents OR the complete
new contents — never a partial one.

These tests pin the contract so it doesn't silently regress to a direct
`shutil.copy2` (the pre-hardening behaviour that left the window open).

Design-property-agnostic: the helper has no design-name/feature branching,
so these tests don't reference any benchmark.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import _atomic_copy


def _leftover_tmps(d: Path) -> list[str]:
    """Temp artifacts the helper might have left behind in `d`."""
    return [p.name for p in d.iterdir() if ".tmp" in p.name]


class AtomicCopyTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp(prefix="atomic_copy_test_")
        self.dir = Path(self._dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._dir, ignore_errors=True)

    def _write(self, name: str, data: bytes) -> Path:
        p = self.dir / name
        p.write_bytes(data)
        return p

    def test_basic_copy_preserves_content(self):
        src = self._write("src.dcp", b"OPTIMIZED-DCP-BYTES")
        dst = self.dir / "out.dcp"
        _atomic_copy(src, dst)
        self.assertTrue(dst.exists())
        self.assertEqual(dst.read_bytes(), b"OPTIMIZED-DCP-BYTES")
        self.assertEqual(_leftover_tmps(self.dir), [],
                         "no temp file should remain after a successful copy")

    def test_overwrite_replaces_old_content_fully(self):
        src = self._write("src.dcp", b"NEW-COMPLETE-CONTENT-1234567890")
        dst = self._write("out.dcp", b"OLD")
        _atomic_copy(src, dst)
        self.assertEqual(dst.read_bytes(), b"NEW-COMPLETE-CONTENT-1234567890")
        self.assertEqual(_leftover_tmps(self.dir), [])

    def test_uses_os_replace_for_atomicity(self):
        # The atomic guarantee comes from os.replace.  Assert we actually
        # route through it on the happy path (not a plain copy onto dst).
        src = self._write("src.dcp", b"abc")
        dst = self.dir / "out.dcp"
        real_replace = os.replace
        seen = {}

        def _spy(a, b):
            seen["called"] = (str(a), str(b))
            return real_replace(a, b)

        with mock.patch("dcp_optimizer.os.replace", side_effect=_spy):
            _atomic_copy(src, dst)
        self.assertIn("called", seen)
        self.assertEqual(seen["called"][1], str(dst))
        self.assertEqual(dst.read_bytes(), b"abc")

    def test_fallback_when_replace_fails_still_copies(self):
        # If os.replace raises (e.g. cross-filesystem), the helper must
        # fall back to a direct copy and still deliver correct content
        # without leaking the temp file.
        src = self._write("src.dcp", b"FALLBACK-CONTENT")
        dst = self.dir / "out.dcp"
        with mock.patch("dcp_optimizer.os.replace",
                        side_effect=OSError("cross-device link")):
            _atomic_copy(src, dst)
        self.assertEqual(dst.read_bytes(), b"FALLBACK-CONTENT")
        self.assertEqual(_leftover_tmps(self.dir), [],
                         "temp file must be cleaned up on the fallback path")

    def test_fsync_failure_is_tolerated(self):
        src = self._write("src.dcp", b"DATA")
        dst = self.dir / "out.dcp"
        with mock.patch("dcp_optimizer.os.fsync",
                        side_effect=OSError("fsync unsupported")):
            _atomic_copy(src, dst)
        self.assertEqual(dst.read_bytes(), b"DATA")
        self.assertEqual(_leftover_tmps(self.dir), [])

    def test_missing_source_raises_and_preserves_existing_dst(self):
        # Atomicity property: a failed copy must NOT clobber a good dst.
        # The pre-existing (e.g. baseline) output stays intact and no temp
        # file is leaked.
        dst = self._write("out.dcp", b"PRIOR-GOOD-DCP")
        missing = self.dir / "does_not_exist.dcp"
        with self.assertRaises(Exception):
            _atomic_copy(missing, dst)
        self.assertEqual(dst.read_bytes(), b"PRIOR-GOOD-DCP",
                         "a failed copy must leave the prior dst untouched")
        self.assertEqual(_leftover_tmps(self.dir), [])


if __name__ == "__main__":
    unittest.main()
