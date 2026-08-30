"""Tests the strict submission archive packager.

The archive must exclude host-specific environment files, virtual environments,
credentials, and local tool configuration so it remains portable. Tests skip
when `tar` or `bash` is unavailable, allowing the suite to run on non-POSIX
systems.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "build_submission.sh"


@unittest.skipUnless(shutil.which("bash") and shutil.which("tar"),
                     "needs bash + tar")
class BuildSubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="build_sub_test_")
        cls.out = os.path.join(cls._tmp, "sub.tar.gz")
        cls.proc = subprocess.run(
            ["bash", str(SCRIPT), cls.out],
            capture_output=True, text=True, cwd=str(REPO),
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def _names(self):
        with tarfile.open(self.out, "r:gz") as t:
            return t.getnames()

    def test_script_succeeds(self):
        self.assertEqual(self.proc.returncode, 0,
                         f"build_submission.sh failed:\n{self.proc.stdout}\n{self.proc.stderr}")
        self.assertTrue(os.path.exists(self.out), "archive not produced")

    def test_no_secret_or_env_leaks(self):
        names = self._names()
        bad = [n for n in names
               if n.endswith("/.env") or "/.env." in n
               or "/.venv/" in n or n.endswith(".pem")
               or "fpl26contest-key" in n or "/.git/" in n]
        self.assertEqual(bad, [], f"submission archive leaks secrets/venv/git: {bad}")

    def test_required_files_present(self):
        names = set(self._names())
        for rel in ["Makefile", "dcp_optimizer.py", "requirements.txt",
                    "prompts/system_prompt_scored.txt", "optimizer/recipe_router.py",
                    "VivadoMCP/vivado_mcp_server.py",
                    "scripts/multi_restart_optimize.py"]:
            self.assertIn(f"fpl26_optimization_contest/{rel}", names,
                          f"required file missing from submission: {rel}")

    def test_archive_root_is_forced_not_derived(self):
        """Verifies that every archive member uses the required fixed root
        directory.

        All members must reside under `fpl26_optimization_contest/`,
        independent of the checkout directory name, because the extraction
        harness enters that exact path.
        """
        names = self._names()
        self.assertTrue(names, "archive is empty")
        stray = sorted({n.split("/", 1)[0] for n in names} - {"fpl26_optimization_contest"})
        self.assertEqual(
            stray, [],
            "archive has members outside fpl26_optimization_contest/ "
            f"(built from a checkout named {REPO.name!r}): {stray}")

    def test_imports_cleanly_reported(self):
        # The script's own import-check must have passed (it exits non-zero
        # otherwise, covered by test_script_succeeds, but assert the signal too).
        self.assertIn("imports cleanly", self.proc.stdout,
                      "build script did not confirm a clean import of the archive")


@unittest.skipUnless(shutil.which("bash") and shutil.which("tar")
                     and shutil.which("git"), "needs bash + tar + git")
class ProvenanceGateTests(unittest.TestCase):
    """Refuse to package a tree that a newer branch strictly supersedes.

    The archive-root bug was invisible from the MAIN checkout because that
    directory is named `fpl26_optimization_contest`. The mirror-image hazard is
    worse and was live: the main checkout sat on a branch that was a
    STRICT ANCESTOR, 140 commits behind final-round-dev. Packaging it would have
    shipped a tree with none of the final round in it — including the fix that
    put the uniform ILS stack on the ship path — and every other check in the
    script would have passed, because the dirname is right and the files are all
    there. A correct-looking archive of the wrong code.
    """

    def _repo(self, tmp):
        root = os.path.join(tmp, "fpl26_optimization_contest")
        shutil.copytree(str(REPO / "scripts"), os.path.join(root, "scripts"))
        for f in ("Makefile", "dcp_optimizer.py", "requirements.txt",
                  "prompts/system_prompt_scored.txt"):
            path = os.path.join(root, f)
            os.makedirs(os.path.dirname(path) or root, exist_ok=True)
            open(path, "w").write("# stub\n")
        for d in ("optimizer", "VivadoMCP"):
            os.makedirs(os.path.join(root, d), exist_ok=True)
        open(os.path.join(root, "optimizer", "recipe_router.py"), "w").write("x=1\n")
        open(os.path.join(root, "VivadoMCP", "vivado_mcp_server.py"), "w").write("x=1\n")
        run = lambda *a: subprocess.run(a, cwd=root, capture_output=True, text=True)
        run("git", "init", "-q", "-b", "release")
        run("git", "config", "user.email", "t@t")
        run("git", "config", "user.name", "t")
        run("git", "add", "-A")
        run("git", "commit", "-qm", "base")
        return root, run

    def test_refuses_when_a_newer_branch_contains_this_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, run = self._repo(tmp)
            # A branch that is a strict DESCENDANT of what we would package.
            run("git", "branch", "newer")
            run("git", "checkout", "-q", "newer")
            open(os.path.join(root, "later.py"), "w").write("x=2\n")
            run("git", "add", "-A")
            run("git", "commit", "-qm", "later work")
            run("git", "checkout", "-q", "release")

            p = subprocess.run(["bash", os.path.join(root, "scripts",
                                                     "build_submission.sh"),
                                os.path.join(tmp, "s.tar.gz")],
                               capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0,
                                "packaged a tree superseded by a newer branch")
            self.assertIn("superseded by a newer branch", p.stderr)
            self.assertIn("newer is 1 commits ahead", p.stderr)
            self.assertFalse(os.path.exists(os.path.join(tmp, "s.tar.gz")),
                             "archive was written despite the refusal")

    def test_allows_when_nothing_supersedes_it(self):
        """A branch whose tip IS this commit is not 'newer' — no false alarm."""
        with tempfile.TemporaryDirectory() as tmp:
            root, run = self._repo(tmp)
            run("git", "branch", "same-commit")
            p = subprocess.run(["bash", os.path.join(root, "scripts",
                                                     "build_submission.sh"),
                                os.path.join(tmp, "s.tar.gz")],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0,
                             f"false alarm on a current tree:\n{p.stderr}")
            self.assertIn("SUBMISSION READY", p.stdout)

    def test_override_is_explicit_and_announced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, run = self._repo(tmp)
            run("git", "branch", "newer")
            run("git", "checkout", "-q", "newer")
            open(os.path.join(root, "later.py"), "w").write("x=2\n")
            run("git", "add", "-A")
            run("git", "commit", "-qm", "later work")
            run("git", "checkout", "-q", "release")

            env = dict(os.environ, SUBMISSION_ALLOW_STALE="1")
            p = subprocess.run(["bash", os.path.join(root, "scripts",
                                                     "build_submission.sh"),
                                os.path.join(tmp, "s.tar.gz")],
                               capture_output=True, text=True, env=env)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("superseded tree packaged on purpose", p.stdout)


if __name__ == "__main__":
    sys.exit(unittest.main())
