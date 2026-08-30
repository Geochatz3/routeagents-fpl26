#!/usr/bin/env python3
"""Check every Markdown link in the repository: file paths *and* `#anchors`.

The path half is the obvious half. The anchor half is the one that bit us: a
heading was reworded, the link that pointed at it kept resolving to a file
that exists, and a path-only checker reported green while the link was dead.
A checker that passes because it is not looking is worse than no checker.

Exit code 0 when every link resolves, 1 otherwise.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# [label](target) — skip image embeds, which start with '!'.
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$", re.M)
FENCE = re.compile(r"^```.*?^```", re.M | re.S)


def slug(heading: str) -> str:
    """GitHub's heading -> anchor rule, as far as we rely on it.

    Inline markup is stripped, the text is lowercased, spaces become hyphens
    and anything that is not alphanumeric, hyphen or underscore is dropped.
    """
    text = re.sub(r"`([^`]*)`", r"\1", heading)
    text = re.sub(r"\*\*?([^*]*)\*\*?", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.strip().lower().replace(" ", "-")
    return "".join(c for c in text if c.isalnum() or c in "-_")


def anchors(path: Path) -> set[str]:
    """Every anchor a Markdown file offers, including duplicate suffixes."""
    body = FENCE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    seen: dict[str, int] = {}
    out = set()
    for _, heading in HEADING.findall(body):
        s = slug(heading)
        if not s:
            continue
        n = seen.get(s, 0)
        out.add(s if n == 0 else f"{s}-{n}")
        seen[s] = n + 1
    # Explicit <a name="..."> / id="..." targets count too.
    out |= set(re.findall(r'<a[^>]+(?:name|id)="([^"]+)"', body))
    return out


def collect(root: Path) -> list[str]:
    """Every broken link under `root`, as human-readable lines."""
    root = root.resolve()

    def hidden(p: Path) -> bool:
        # Relative to `root`: an absolute path may sit under a dot-directory
        # (a checkout inside ~/.cache, a temporary tree) without any of its
        # own parts being hidden. Testing p.parts silently checks nothing.
        return any(part.startswith(".") or part == "__pycache__"
                   for part in p.relative_to(root).parts)

    docs = sorted(p for p in root.rglob("*.md") if not hidden(p))
    cache: dict[Path, set[str]] = {}
    bad: list[str] = []

    for doc in docs:
        body = FENCE.sub("", doc.read_text(encoding="utf-8", errors="replace"))
        rel = doc.relative_to(root)
        for target in LINK.findall(body):
            if target.startswith(("http://", "https://", "mailto:")):
                continue

            path_part, _, anchor = target.partition("#")
            if path_part:
                dest = (doc.parent / path_part).resolve()
                if not dest.exists():
                    bad.append(f"{rel}: missing path -> {target}")
                    continue
                if dest.is_dir():
                    continue
            else:
                dest = doc

            if not anchor or dest.suffix != ".md":
                continue
            if dest not in cache:
                cache[dest] = anchors(dest)
            if anchor.lower() not in cache[dest]:
                where = "this file" if dest == doc else dest.relative_to(root)
                bad.append(f"{rel}: no heading '#{anchor}' in {where}")

    collect.checked = len(docs)  # type: ignore[attr-defined]
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=REPO, type=Path, metavar="DIR",
                    help="tree to check (default: this repository)")
    bad = collect(ap.parse_args(argv).root)
    for line in bad:
        print(f"BROKEN  {line}")
    print(f"{collect.checked} markdown files checked, "  # type: ignore[attr-defined]
          f"{len(bad)} broken links")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
