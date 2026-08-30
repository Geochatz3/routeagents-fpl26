"""`optimizer/README.md` maps every module in `optimizer/`.

A directory listing is the first thing a reader opens after the top-level
README, and a map that silently stops covering new files is worse than no map:
the six modules added for this project sat unmentioned for the whole contest,
the same way `RapidWrightMCP/README.md` documented 11 of its 17 tools.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "optimizer"
DOC = PKG / "README.md"

# Flagged in the doc as shipped-but-inactive; the list is CONFIGURATION.md's.
DEFAULT_OFF = {
    "plan_critic", "cross_model_steering", "replace_gamble", "gate_log",
    "policy_card",
}


def mentioned() -> set[str]:
    return set(re.findall(r"`([a-z_0-9]+)\.py`", DOC.read_text()))


def on_disk() -> set[str]:
    return {p.stem for p in PKG.glob("*.py") if p.stem != "__init__"}


class ModuleMapTests(unittest.TestCase):
    def test_every_module_is_mapped(self):
        missing = sorted(on_disk() - mentioned())
        self.assertEqual(
            missing, [],
            f"optimizer/README.md does not mention: {missing}",
        )

    def test_the_map_names_no_module_that_is_gone(self):
        listed = mentioned() - {"dcp_optimizer"}
        stale = sorted(listed - on_disk())
        self.assertEqual(stale, [], f"optimizer/README.md still names: {stale}")

    def test_the_default_off_set_matches_the_configuration_doc(self):
        """Two files name the same seven modules; neither may drift alone."""
        conf = (REPO / "docs" / "CONFIGURATION.md").read_text()
        # Only the sentence that lists them: the table below it contrasts
        # `replace_gamble` with `deep_replace_sibling`, which ships ON.
        section = conf.split("## Modules that ship off", 1)[1]
        section = section.split("Two of them re-place", 1)[0]
        named = set(re.findall(r"`([a-z_0-9]+)`", section))
        self.assertEqual(
            named & on_disk(), DEFAULT_OFF,
            "docs/CONFIGURATION.md and this test disagree on which modules "
            "ship default-OFF",
        )
        for mod in DEFAULT_OFF:
            with self.subTest(module=mod):
                self.assertIn(f"`{mod}.py`", DOC.read_text())


class NoUnreachableModuleTests(unittest.TestCase):
    """Nothing in optimizer/ is importable-but-never-imported.

    Two modules used to be: default-OFF *and* with no caller outside their own
    tests. They were deleted rather than documented (docs/PROVENANCE.md). This
    keeps the property rather than the exception — a default-OFF module is fine
    when something can reach it, and dead weight when nothing can.
    """

    def test_every_module_has_a_caller_outside_its_own_test(self):
        sources = [p for p in REPO.rglob("*.py")
                   if ".git" not in p.parts and "tests" not in p.parts]
        orphans = []
        for mod in sorted(on_disk() - {"test_strategy_memory"}):
            importers = [
                p for p in sources
                if p.stem != mod
                and re.search(rf"\b(import|from)\s+\S*\b{mod}\b", p.read_text())
            ]
            if not importers:
                orphans.append(mod)
        self.assertEqual(
            orphans, [],
            f"optimizer/ modules nothing imports: {orphans}. A public repo "
            f"should not ship code no caller can reach — delete it, and "
            f"disclose the deletion in docs/PROVENANCE.md.",
        )


if __name__ == "__main__":
    unittest.main()
