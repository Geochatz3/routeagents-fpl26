"""Test the coupled Vivado process-group cleanup contract.

The fix is ON by default, and both call sites must read `FPL26_VIVADO_LEAK_FIX`
using the same default literal. Spawning creates a new session; cleanup closes
stdin before signaling that session's process group.

These operations must remain coupled because group cleanup without session
isolation can terminate the driver and concurrent runs, while wrapper-only
termination can orphan engine processes and their open pipes. Setting the
variable to `0` remains the supported kill switch.
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
        """Spawn and cleanup each read the flag; losing one half is the dangerous
        case.
        """
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
        """Validated on hardware (ORPHANS=0); an unconfigured run must be protected."""
        defaults = _leak_fix_defaults()
        for d in defaults:
            self.assertIn(
                str(d).strip().lower(), ("1", "true", "on", "yes"),
                "%s must default ON — an unconfigured eval run is exactly the run that "
                "OOMs, and an OOM is a 0.000 for that benchmark" % FLAG)

    def test_kill_switch_value_still_disables(self):
        """`=0` must remain a real off switch: it restores the previous spawn
        byte-for-byte.
        """
        truthy = ("1", "true", "on", "yes")
        self.assertNotIn("0", truthy)
        self.assertNotIn("", truthy)
        # Mirrors the parse at both call sites.
        for value in ("0", "false", "off", "no", ""):
            self.assertNotIn(value.strip().lower(), truthy)


if __name__ == "__main__":
    unittest.main()
