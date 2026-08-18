"""No function may load a NAME that resolves to nothing. (aug08)

THE BUG THIS PINS: dcp_optimizer.py:13623 tested
``res.verdict == VERDICT_ERROR`` inside the deep-replace wedged-session
recovery, but the in-function import one screen up brought in only
``DEEP_REPLACE_COST_MARGIN, VERDICT_ADOPTED, deep_replace_should_run,
run_deep_replace_sibling``. The resulting NameError was swallowed by the
enclosing ``except Exception`` and logged as "(ignored)" — so a guard written
to stop a wedged Vivado session from finalizing a whole benchmark at alpha 0
fired in 17 of 17 gate-16 runs and its body executed ZERO times.

An unresolved constant inside a try/except is not a runtime condition; it is
code that has never run. This test is the static census for that class: every
uppercase Name loaded in any function of the audited files must resolve to a
local binding, an in-function import, a module global, or a builtin.

Validated against the shipped #13 tree (31e9f53): exactly one hit, the
VERDICT_ERROR line — and zero false positives across dcp_optimizer.py,
optimizer/ and scripts/.
"""
from __future__ import annotations

import ast
import builtins
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AUDITED = [ROOT / "dcp_optimizer.py", ROOT / "optimizer", ROOT / "scripts"]


def _function_defined_names(fn: ast.AST) -> set:
    out = set()
    for node in ast.walk(fn):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            for arg in (a.posonlyargs + a.args + a.kwonlyargs +
                        ([a.vararg] if a.vararg else []) +
                        ([a.kwarg] if a.kwarg else [])):
                out.add(arg.arg)
            if node is not fn:
                out.add(node.name)
        elif isinstance(node, ast.ClassDef):
            out.add(node.name)
        elif isinstance(node, ast.ImportFrom):
            for al in node.names:
                out.add(al.asname or al.name)
        elif isinstance(node, ast.Import):
            for al in node.names:
                out.add((al.asname or al.name).split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            out.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            out.update(node.names)
    return out


def _module_defined_names(tree: ast.Module) -> set:
    out = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            out.add(node.name)
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                out.add(sub.id)
            elif isinstance(sub, ast.ImportFrom):
                for al in sub.names:
                    out.add(al.asname or al.name)
            elif isinstance(sub, ast.Import):
                for al in sub.names:
                    out.add((al.asname or al.name).split(".")[0])
    return out


def _audit(path: Path) -> list:
    tree = ast.parse(path.read_text(errors="replace"))
    mod_names = _module_defined_names(tree)
    hits = []
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        defined = (_function_defined_names(fn) | mod_names
                   | set(dir(builtins)))
        for node in ast.walk(fn):
            if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                    and node.id.isupper() and len(node.id) > 1
                    and node.id not in defined):
                hits.append(f"{path.name}:{node.lineno} {fn.name}() "
                            f"UNRESOLVED {node.id}")
    return hits


class NoUnresolvedNamesTests(unittest.TestCase):
    def test_no_unresolved_uppercase_names_anywhere(self):
        hits = []
        for target in AUDITED:
            files = [target] if target.is_file() else \
                sorted(target.rglob("*.py"))
            for f in files:
                hits += _audit(f)
        self.assertEqual(hits, [],
                         "unresolved constants (NameError at runtime):\n"
                         + "\n".join(hits))

    def test_verdict_error_is_imported_where_the_guard_uses_it(self):
        """The specific aug08 fix: the recovery guard's import list must
        carry VERDICT_ERROR, in the same method that compares against it."""
        import inspect

        import dcp_optimizer
        src = inspect.getsource(
            dcp_optimizer.DCPOptimizer._deep_replace_sibling_after_polish)
        self.assertIn("VERDICT_ERROR", src)
        tree = ast.parse("async " + src.strip() if not
                         src.lstrip().startswith(("async", "def"))
                         else __import__("textwrap").dedent(src))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for al in node.names:
                    imported.add(al.asname or al.name)
        self.assertIn("VERDICT_ERROR", imported,
                      "the guard at ~:13623 compares against VERDICT_ERROR; "
                      "it must appear in the in-function import list or the "
                      "recovery body dies on a NameError as it did 17/17 in "
                      "the shipped #13 gate")

    def test_verdict_error_value_matches_the_producer(self):
        """The guard compares res.verdict (set by deep_replace_sibling) to
        the imported constant — same module, same value, by construction."""
        from optimizer.deep_replace_sibling import VERDICT_ERROR
        self.assertEqual(VERDICT_ERROR, "ERROR")


if __name__ == "__main__":
    unittest.main()
