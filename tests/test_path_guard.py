"""Tests for optimizer.path_guard — DCP write-path validation."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizer.path_guard import PathGuard, PathGuardError, is_under


class IsUnderTests(unittest.TestCase):

    def test_descendant_path_is_under(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertTrue(is_under(root / "a" / "b.dcp", root))

    def test_root_itself_is_under(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertTrue(is_under(root, root))

    def test_outside_path_is_not_under(self):
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            self.assertFalse(is_under(Path(t2) / "x.dcp", Path(t1)))


class PathGuardAuditModeTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name) / "run"
        self.run_dir.mkdir()
        self.output_parent = Path(self.tmp.name) / "out"
        self.output_parent.mkdir()
        self.subm = Path(self.tmp.name) / "submission" / "dcps"
        self.subm.mkdir(parents=True)
        self.guard = PathGuard(
            [self.run_dir, self.output_parent, self.subm],
            mode="audit",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_allowed_run_dir_passes(self):
        ok = self.guard.check(self.run_dir / "best_valid.dcp",
                              context="mirror_write")
        self.assertTrue(ok)
        self.assertEqual(self.guard.violations, [])

    def test_allowed_output_path_passes(self):
        self.assertTrue(self.guard.check(self.output_parent / "ship.dcp"))

    def test_allowed_submission_path_passes(self):
        self.assertTrue(self.guard.check(self.subm / "amd_mini-isp" / "amd_mini-isp.dcp"))

    def test_outside_path_audit_returns_false_and_logs(self):
        outside = Path(self.tmp.name) / "rogue.dcp"
        ok = self.guard.check(outside, context="rogue_write")
        self.assertFalse(ok)
        self.assertEqual(len(self.guard.violations), 1)
        self.assertEqual(self.guard.violations[0]["context"], "rogue_write")
        # Audit mode must NOT raise.

    def test_allow_root_extends_policy(self):
        new_root = Path(self.tmp.name) / "extra"
        new_root.mkdir()
        target = new_root / "x.dcp"
        self.assertFalse(self.guard.check(target))
        self.guard.allow_root(new_root)
        self.guard.reset_violations()
        self.assertTrue(self.guard.check(target))


class PathGuardEnforceModeTests(unittest.TestCase):

    def test_enforce_raises_on_outside_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            guard = PathGuard([run_dir], mode="enforce")
            self.assertTrue(guard.check(run_dir / "ok.dcp"))
            with self.assertRaises(PathGuardError):
                guard.check(Path(tmp) / "rogue.dcp", context="rogue")


if __name__ == "__main__":
    unittest.main()
