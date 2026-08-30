"""The starter-kit comparison is about this tree, not a memory.

The README claims eight mechanism names appear nowhere in the contest starter
kit's agent, and that the tool surface is unchanged from it. The first half is
checkable offline against our own tree — if a mechanism the README names as
ours stops existing here, the comparison is stale and says so.

The other half (that the starter kit lacks them) is a network fact and is not
asserted here; the README gives the diff command for a reader to run.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"
# The starter-kit comparison moved off the front page; it lives with the rest
# of the contest record now.
PROVENANCE = REPO / "docs" / "PROVENANCE.md"

# Named in the README as absent from the starter kit and present here.
OURS = ("recipe_router", "deep_replace", "tail_controller", "wall_economics",
        "logic_floor", "strategy_memory", "multi_restart", "ils")

PLAY_COUNT = 42


class StarterKitDeltaTests(unittest.TestCase):
    def test_every_mechanism_the_readme_claims_still_exists(self):
        here = {p.stem for p in (REPO / "optimizer").glob("*.py")}
        here |= {p.stem for p in (REPO / "scripts").glob("*.py")}
        for name in OURS:
            with self.subTest(mechanism=name):
                self.assertTrue(
                    any(name in stem for stem in here),
                    f"the delta table names {name!r} as this project's contribution, "
                    f"but no module here carries that name",
                )

    def test_the_readme_and_the_playbook_agree_on_the_play_count(self):
        for path in (README, REPO / "docs" / "PLAYBOOK.md"):
            with self.subTest(doc=path.name):
                self.assertRegex(
                    path.read_text(), rf"\b{PLAY_COUNT} plays\b",
                    f"{path.name} no longer says '{PLAY_COUNT} plays'",
                )

    def test_the_tool_total_in_the_delta_table_matches_the_pinned_surface(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_tool_surface import TOTAL, MCP_SERVERS  # noqa: E402
        mcp = sum(n for _, n in MCP_SERVERS.values())
        self.assertEqual(mcp, 34)
        self.assertEqual(TOTAL, 40)
        self.assertIn(f"| MCP tools | {mcp} ", PROVENANCE.read_text())


if __name__ == "__main__":
    unittest.main()
