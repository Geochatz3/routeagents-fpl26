"""Every Markdown link in the repository resolves — path *and* `#anchor`.

This exists because a path-only check reported green while a link in the
README pointed at a heading that had been reworded away. The file still
existed, so the link "resolved"; the anchor did not.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import check_docs  # noqa: E402


class DocLinkTests(unittest.TestCase):
    def test_every_link_resolves(self):
        self.assertEqual(
            check_docs.main([]), 0,
            "broken Markdown links — see the BROKEN lines above",
        )

    def test_a_dead_anchor_is_actually_caught(self):
        """Guard the guard. If the anchor half stops working, the check above
        would keep passing on a repository whose links had rotted."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.md").write_text(
                "# Title\n\n"
                "[live](#the-heading) [dead](#the-hedaing) "
                "[cross](b.md#other) [gone](nope.md)\n\n"
                "## The heading\n"
            )
            (root / "b.md").write_text("# B\n\n## Other\n")
            self.assertEqual(check_docs.main(["--root", str(root)]), 1)

            failures = check_docs.collect(root)
        joined = "\n".join(failures)
        self.assertIn("the-hedaing", joined, "dead anchor not reported")
        self.assertIn("nope.md", joined, "missing path not reported")
        self.assertNotIn("the-heading'", joined, "live anchor reported as dead")
        self.assertNotIn("b.md#other", joined, "live cross-file anchor reported")


if __name__ == "__main__":
    unittest.main()
