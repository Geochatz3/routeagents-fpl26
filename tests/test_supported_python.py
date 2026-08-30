"""The Python versions the README promises are the ones CI actually runs.

README.md states `Python >= 3.10`. CI tested 3.11 and nothing else, so the
floor was a claim no one had checked -- the same shape as a test count nobody
re-ran or a citation nobody opened.

Two things are pinned here: the matrix covers the stated floor, and the source
tree still parses under it. The second is a static check, so it catches syntax
that a newer interpreter accepts (`except*`, PEP 695 `type` aliases) without
needing an old interpreter installed. It cannot catch a stdlib API that only
exists on a newer version -- the CI matrix is what covers that.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "tests.yml"
README = REPO / "README.md"


def _readme_floor() -> tuple[int, int]:
    m = re.search(r"\|\s*Python\s*\|\s*>=\s*(\d+)\.(\d+)\s*\|", README.read_text())
    assert m, "README.md no longer states a Python floor in the expected form"
    return int(m.group(1)), int(m.group(2))


def _matrix() -> list[tuple[int, int]]:
    m = re.search(r"python-version:\s*\[([^\]]+)\]", WORKFLOW.read_text())
    assert m, "the workflow no longer declares a python-version matrix"
    out = []
    for chunk in m.group(1).split(","):
        v = chunk.strip().strip('"').strip("'")
        major, minor = v.split(".")
        out.append((int(major), int(minor)))
    return out


def test_ci_runs_the_version_the_readme_promises():
    floor = _readme_floor()
    matrix = _matrix()
    assert floor in matrix, (
        f"README.md promises Python >= {floor[0]}.{floor[1]} but CI runs "
        f"{['.'.join(map(str, v)) for v in matrix]}. Either test the floor or "
        f"stop promising it.")


def test_the_matrix_tests_more_than_one_version():
    assert len(_matrix()) >= 2, (
        "a single-version matrix cannot detect a version-specific break")


# Names that exist on a newer interpreter than the 3.10 floor. A static parse
# cannot see these -- `self.enterContext(...)` is valid syntax everywhere, it
# just raises AttributeError below 3.11, which is how it reached main and was
# caught only once CI actually ran 3.10.
#
# Matched through the AST, as an imported name or an attribute access, never
# as raw text: a first attempt grepped for the word and flagged six English
# sentences containing "override". Same lesson as the docstring that was
# minting flag names in ship_path_audit.py -- prose is not code.
#
# A hand-maintained list is a FAST LOCAL SIGNAL, never the guarantee; it will
# always be incomplete. The CI matrix is what actually proves the floor.
TOO_NEW = {
    "enterContext": (3, 11),
    "enterAsyncContext": (3, 11),
    "tomllib": (3, 11),
    "ExceptionGroup": (3, 11),
    "BaseExceptionGroup": (3, 11),
    "StrEnum": (3, 11),
    "TaskGroup": (3, 11),
    "file_digest": (3, 11),
    "batched": (3, 12),
    "override": (3, 12),
}


def _version_gated_uses(path: Path, banned: set[str]):
    """(lineno, name) for each banned name imported or accessed on this file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in banned:
            out.append((node.lineno, node.attr))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                base = a.name.split(".")[0]
                for cand in (a.name, base):
                    if cand in banned:
                        out.append((node.lineno, cand))
    return out


def test_no_source_uses_an_api_newer_than_the_floor():
    floor = _readme_floor()
    banned = {n for n, v in TOO_NEW.items() if v > floor}
    if not banned:
        pytest.skip(f"nothing in the list postdates {floor[0]}.{floor[1]}")
    files = subprocess.run(
        ["git", "ls-files", "*.py"], cwd=REPO,
        capture_output=True, text=True, timeout=120).stdout.split()
    hits = [f"{rel}:{ln}: {name}"
            for rel in files
            for ln, name in _version_gated_uses(REPO / rel, banned)]
    assert hits == [], (
        f"these use an API newer than the stated floor "
        f"{floor[0]}.{floor[1]}:\n  " + "\n  ".join(hits[:10]))


def test_every_source_file_parses_on_the_stated_floor():
    floor = _readme_floor()
    files = subprocess.run(
        ["git", "ls-files", "*.py"], cwd=REPO,
        capture_output=True, text=True, timeout=120).stdout.split()
    assert len(files) > 50, "expected the tracked Python corpus, got %d" % len(files)
    broken = []
    for rel in files:
        src = (REPO / rel).read_text(encoding="utf-8", errors="ignore")
        try:
            ast.parse(src, feature_version=floor)
        except SyntaxError as e:
            broken.append(f"{rel}:{e.lineno}: {e.msg}")
    assert broken == [], (
        f"syntax not valid on Python {floor[0]}.{floor[1]}:\n  "
        + "\n  ".join(broken[:10]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
