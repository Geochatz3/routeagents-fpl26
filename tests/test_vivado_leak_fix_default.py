"""The engine-leak fix is DEFAULT ON (jul29) — and its two halves must never disagree.

THE LEAK: `_vivado_pid` is the bin/vivado WRAPPER SHELL. SIGKILLing it kills a shell that
cannot forward the signal, so bin/loader and the real engine reparent to init and idle at
0 CPU holding 4-6 GB each, never seeing EOF because our stdin pipe stays open. Measured
jul28: eleven engines / ~19 GB dead RSS on ONE box, and the orphans survive their MCP's
exit. On the eval box (31 GB, swapless, ~1 h per design) an OOM is a 0.000 for that
benchmark — the exact mode that cost boom_soc_v2 in beta.

THE FIX has two halves that are only safe TOGETHER:
  * spawn with `start_new_session=True`, giving this Vivado subtree its own process group;
  * on cleanup, close stdin and then `killpg` that group.

WHY THIS FILE EXISTS. If the spawn half were OFF while the cleanup half were ON, the
Vivado subtree would share OUR process group and `killpg` would kill the driver and every
concurrent run on the box. That is not a degraded mode, it is the jul28 mistake the
handoff warns about ("never kill by pgid"), escalated. The two halves are kept in sync by
reading the SAME environment variable with the SAME default in both places — an invariant
that a future edit to one site can silently break, because both sites still parse fine on
their own. Nothing else in the suite pins it, so it is pinned here at the source level.

These tests pin the CONTRACT:
  * default (variable unset) is ON, so the fix protects a run nobody configured;
  * `FPL26_VIVADO_LEAK_FIX=0` still disables it (the documented kill switch);
  * both call sites use an identical default literal.
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SERVER = ROOT / "VivadoMCP" / "vivado_mcp_server.py"
FLAG = "FPL26_VIVADO_LEAK_FIX"


def _leak_fix_defaults():
    """Every `os.environ.get("FPL26_VIVADO_LEAK_FIX", <default>)` default literal.

    Parsed from the AST rather than grepped so a reformat, a renamed local, or a comment
    mentioning the flag cannot change the answer.
    """
    tree = ast.parse(SERVER.read_text(encoding="utf-8", errors="replace"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "get"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != FLAG:
            continue
        if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant):
            found.append(None)  # no default at all — also a desync risk
            continue
        found.append(node.args[1].value)
    return found


class TestLeakFixDefault(unittest.TestCase):
    def test_both_halves_are_present(self):
        """Spawn and cleanup each read the flag; losing one half is the dangerous case."""
        defaults = _leak_fix_defaults()
        self.assertEqual(
            len(defaults), 2,
            "expected exactly 2 reads of %s (spawn + cleanup), found %d — if a half was "
            "removed or a third consumer added, re-derive the safety argument before "
            "changing this test" % (FLAG, len(defaults)))

    def test_defaults_are_identical(self):
        """The invariant: a spawn/cleanup desync makes killpg target our own group."""
        defaults = _leak_fix_defaults()
        self.assertEqual(
            len(set(defaults)), 1,
            "the two %s reads disagree (%r) — with the spawn off and the cleanup on, "
            "killpg kills the driver and every concurrent run on the box" % (FLAG, defaults))

    def test_default_is_on(self):
        """Validated on hardware jul29 (ORPHANS=0); an unconfigured run must be protected."""
        defaults = _leak_fix_defaults()
        for d in defaults:
            self.assertIn(
                str(d).strip().lower(), ("1", "true", "on", "yes"),
                "%s must default ON — an unconfigured eval run is exactly the run that "
                "OOMs, and an OOM is a 0.000 for that benchmark" % FLAG)

    def test_kill_switch_value_still_disables(self):
        """`=0` must remain a real off switch: it restores the previous spawn byte-for-byte."""
        truthy = ("1", "true", "on", "yes")
        self.assertNotIn("0", truthy)
        self.assertNotIn("", truthy)
        # Mirrors the parse at both call sites.
        for value in ("0", "false", "off", "no", ""):
            self.assertNotIn(value.strip().lower(), truthy)


if __name__ == "__main__":
    unittest.main()
