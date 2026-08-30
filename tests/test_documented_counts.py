"""The test counts printed in the docs match the suite that prints them.

ARTIFACT.md quotes what `make test` reports. It used to be quoted in the
README as well, and the two drifted apart the moment tests were added: README
said 2473/69 while ARTIFACT said 2477/78, which is exactly the failure the rest
of this repository's guards exist to prevent. One file states it now, so there
is one number to keep true.

Collection is static and takes under two seconds, so the test count is pinned
directly. Subtest counts are a runtime property of `pytest-subtests` and cannot
be collected, so that number is only checked for being present and plausible.
"""
from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The same path list the `test` target in the Makefile passes to pytest.
MAKE_TEST_PATHS = ["tests/", "optimizer/", "scheduler/", "recipes/",
                   "tests/test_validate_dcps.py"]

README = REPO / "README.md"
ARTIFACT = REPO / "ARTIFACT.md"


def collected() -> int:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", *MAKE_TEST_PATHS, "--collect-only", "-q",
         "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True, timeout=300).stdout
    m = re.search(r"(\d+) tests collected", out)
    assert m, f"could not parse a collection count from:\n{out[-800:]}"
    return int(m.group(1))


def documented() -> dict[str, tuple[int, int]]:
    """(tests, subtests) as each file states them."""
    found = {}
    for path, pattern in (
        (ARTIFACT, r"`([\d,]+) passed, (\d+) subtests passed`"),
    ):
        m = re.search(pattern, path.read_text())
        assert m, f"{path.name} no longer states a test count in the expected form"
        found[path.name] = (int(m.group(1).replace(",", "")), int(m.group(2)))
    return found


class DocumentedCountTests(unittest.TestCase):
    def test_the_subtest_count_is_stated_and_plausible(self):
        (_tests, subtests), = documented().values()
        self.assertGreater(subtests, 0, "ARTIFACT.md states no subtest count")

    def test_the_documented_test_count_is_the_real_one(self):
        real = collected()
        for name, (tests, _) in documented().items():
            with self.subTest(doc=name):
                self.assertEqual(
                    tests, real,
                    f"{name} says `make test` runs {tests} tests; it collects "
                    f"{real}. Update both files, or the number is a claim "
                    f"nobody checked.",
                )


if __name__ == "__main__":
    unittest.main()
