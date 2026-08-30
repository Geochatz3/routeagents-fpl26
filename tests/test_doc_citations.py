"""Every `path/to/file.py:SYMBOL` citation in the docs names a symbol that exists.

docs/PLAYBOOK.md sources each play to the constant or function that implements
it -- 38 citations, plus one in CONFIGURATION.md. Nothing checked them, and one
was wrong for a while: the B2 row sourced its ">= 0.20x slice" rule to
DEEP_REPLACE_PHYSOPT_FRAC, which is 0.7 and means something else. The rule
lives in DEEP_REPLACE_B2_MIN_SLICE_FRAC. Both names exist, so no link checker
or import could catch it; only reading the constant could.

Two checks, because existence alone would NOT have caught that bug:
DEEP_REPLACE_PHYSOPT_FRAC exists, it is just the wrong constant.

  1. Every cited symbol exists in the file it is sourced to. Cheap and total:
     a rename, move, or deletion turns a stale citation into a failing test
     instead of a silently wrong doc.
  2. When a citation names a module-level NUMERIC constant and the same doc
     line quotes numbers, the constant's value has to be one of them. That is
     what fails on a row reading "\u2265 0.20x" next to a constant equal to 0.7.

Check 2 currently guards exactly one citation -- it is the only one in the
repository that cites a numeric constant on a line that also states a number.
Measured, not assumed: across every doc it fires once and produces no false
positives. It costs little and covers the precise shape that went wrong.

Neither check can tell you a citation points at the *conceptually* right
symbol when no number is involved. That still needs a reader.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# `optimizer/tail_controller.py:TAIL_MENU`, `dcp_optimizer.py:SHALLOW_DET_TCL`,
# and the one wildcard form, `...:DEEP_REPLACE_B3_*`.
CITATION = re.compile(r"`([A-Za-z0-9_./]+\.(?:py|tcl|txt)):([A-Za-z0-9_.*]+)`")


def _citations():
    found = []
    for md in sorted(REPO.glob("**/*.md")):
        if ".git" in md.parts:
            continue
        rel = md.relative_to(REPO)
        for n, line in enumerate(md.read_text().splitlines(), 1):
            for m in CITATION.finditer(line):
                found.append((str(rel), n, m.group(1), m.group(2)))
    return found


def _numeric_constants(path: Path) -> dict[str, float]:
    """Module-level `NAME = <int|float>` assignments. Booleans are excluded:
    `True` is an int subclass and would compare equal to a 1 in the prose."""
    out: dict[str, float] = {}
    if path.suffix != ".py":
        return out
    for node in ast.parse(path.read_text(errors="ignore")).body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, (int, float))
                and not isinstance(node.value.value, bool)):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = float(node.value.value)
    return out


# A bare number, not one glued to a word or a version dotted-triple.
NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)")


def _symbols(path: Path) -> set[str]:
    """Top-level names a Python file defines, plus Class.method pairs.

    Non-Python targets (.tcl, .txt) have no parse tree, so they fall back to
    a word-boundary search of the text.
    """
    text = path.read_text(errors="ignore")
    if path.suffix != ".py":
        return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    names: set[str] = set()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            names.add(node.attr)
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for sub in cls.body:
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(f"{cls.name}.{sub.name}")
    return names


CITATIONS = _citations()

# Citations that name a module-level numeric constant AND sit on a line that
# quotes a number -- the only ones check 2 can say anything about. Selected up
# front so the suite reports real assertions instead of a wall of skips.
NUMERIC_CITATIONS = []
for _doc, _line, _target, _sym in CITATIONS:
    _path = REPO / _target
    if not _path.exists():
        continue
    _val = _numeric_constants(_path).get(_sym)
    if _val is None:
        continue
    _text = (REPO / _doc).read_text().splitlines()[_line - 1]
    if re.search(r"(?<![\w.])\d", _text):
        NUMERIC_CITATIONS.append((_doc, _line, _target, _sym))


def test_the_docs_actually_carry_citations():
    # A regex that silently stops matching would make every case below vacuous.
    assert len(CITATIONS) >= 30, (
        f"only {len(CITATIONS)} file:SYMBOL citations found; the pattern "
        f"probably stopped matching")
    assert NUMERIC_CITATIONS, (
        "no citation names a numeric constant next to a stated number; the "
        "value check below has gone vacuous")


@pytest.mark.parametrize(
    "doc,line,target,symbol", CITATIONS,
    ids=[f"{d}:{n}:{t}:{s}" for d, n, t, s in CITATIONS])
def test_cited_symbol_exists(doc, line, target, symbol):
    path = REPO / target
    assert path.exists(), f"{doc}:{line} cites {target}, which does not exist"
    names = _symbols(path)
    if symbol.endswith("*"):
        prefix = symbol[:-1]
        assert any(n.startswith(prefix) for n in names), (
            f"{doc}:{line} cites {target}:{symbol}, but nothing in that file "
            f"starts with {prefix!r}")
    else:
        assert symbol in names, (
            f"{doc}:{line} cites {target}:{symbol}, which that file does not "
            f"define")


@pytest.mark.parametrize(
    "doc,line,target,symbol", NUMERIC_CITATIONS,
    ids=[f"{d}:{n}:{t}:{s}" for d, n, t, s in NUMERIC_CITATIONS])
def test_cited_constant_value_matches_the_prose(doc, line, target, symbol):
    """A row that states a threshold must cite the constant holding it."""
    value = _numeric_constants(REPO / target)[symbol]
    text = (REPO / doc).read_text().splitlines()[line - 1]
    quoted = {float(n) for n in NUMBER.findall(text)}
    assert any(abs(q - value) < 1e-9 for q in quoted), (
        f"{doc}:{line} states {sorted(quoted)} and cites {target}:{symbol}, "
        f"which is {value}. Either the prose or the citation is wrong.")
