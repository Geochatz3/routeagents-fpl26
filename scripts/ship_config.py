#!/usr/bin/env python3
"""Compute the effective configuration produced by a clean evaluation invocation.

Configuration composes Makefile environment injection with Python environment
defaults. Values exported by `make run_optimizer` take precedence; Python
defaults apply only when no value is exported, such as during direct execution.

`FPL26_NO_X` variables are kill switches, so an unset value enables the
feature. Makefile predicates that emit a CLI flag are opt-ins, so an unset
value omits the flag.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

PY_GET = re.compile(
    r"""os\.(?:environ\.get|getenv)\(\s*["'](FPL26_[A-Z0-9_]+)["']\s*,\s*["']([^"']*)["']""",
    re.S,
)
MK_IF = re.compile(r"(FPL26_[A-Z0-9_]+)=\$\(if\s*\$\(([A-Z0-9_]+)\),\s*\$\(\2\),\s*([^)]*)\)")
MK_CLI = re.compile(
    r"\$\(if\s*\$\(filter\s+1 true yes on,\s*\$\(([A-Z0-9_]+)\)\),\s*(--[a-z0-9-]+)\)"
)

TRUTHY = {"1", "true", "on", "yes"}
FALSY = {"0", "false", "off", "no", ""}


def is_on(value: str) -> bool:
    return value.strip().lower() in TRUTHY


def _as_root(root) -> Path:
    """Accept str or Path. A CLI-shaped tool gets handed strings; silently
    breaking on one turns a config audit into an AttributeError."""
    return REPO if root is None else Path(root)


def _iter_py(root: Path):
    for f in root.rglob("*.py"):
        if set(f.parts) & {"tests", "RapidWright", ".venv", "__pycache__", "node_modules"}:
            continue
        try:
            yield f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue


def python_defaults(root: Path | None = None) -> dict[str, set[str]]:
    """{FPL26_NAME: {default strings seen at call sites}}."""
    root = _as_root(root)
    out: dict[str, set[str]] = {}
    for src in _iter_py(root):
        for name, dflt in PY_GET.findall(src):
            out.setdefault(name, set()).add(dflt)
    return out


def _recipe(root: Path) -> str:
    root = _as_root(root)
    text = (root / "Makefile").read_text(encoding="utf-8", errors="replace")
    # Only the target the contest evaluator invokes.
    m = re.search(r"^run_optimizer:.*?(?=^\w[\w-]*:)", text, re.S | re.M)
    return m.group(0) if m else text


def makefile_injections(root: Path | None = None) -> dict[str, str]:
    """{FPL26_NAME: value shipped when the corresponding make var is unset}."""
    root = _as_root(root)
    return {n: d.strip() for n, _v, d in MK_IF.findall(_recipe(root))}


def cli_optins(root: Path | None = None) -> dict[str, str]:
    """{MAKE_VAR: cli flag it gates} — not passed unless the make var is set."""
    root = _as_root(root)
    return dict(MK_CLI.findall(_recipe(root)))


def effective(root: Path | None = None) -> dict[str, bool]:
    """The composed answer: {FPL26_NAME: ships_on} for a clean eval box.

    Makefile injection wins over the python default. Names appearing in neither
    layer are absent (never read ⇒ nothing to assert).
    """
    root = _as_root(root)
    py = python_defaults(root)
    mk = makefile_injections(root)
    out: dict[str, bool] = {}
    for name, vals in py.items():
        out[name] = any(is_on(v) for v in vals)
    for name, val in mk.items():
        out[name] = is_on(val)
    return out


def valued_injections(root: Path | None = None) -> dict[str, str]:
    """Names the Makefile injects with a non-boolean value, e.g. a number.

    `is_on` only recognises the boolean spellings, so a numeric setting would
    otherwise be classified as "ships off" — which reads as "this mechanism is
    inactive" when it is in fact armed with a specific value.
    """
    root = _as_root(root)
    out = {}
    for name, val in makefile_injections(root).items():
        v = val.strip()
        if v and not is_on(v) and v.strip().lower() not in FALSY:
            out[name] = v
    return out


def layer_disagreements(root: Path | None = None) -> dict[str, tuple[list[str], str]]:
    """Names where the python default and the Makefile injection disagree.

    Not necessarily a bug — the Makefile is allowed to be the ship decision — but
    every entry is a place where reading the python default alone misleads you,
    so it must be a deliberate, reviewed list.
    """
    root = _as_root(root)
    py, mk = python_defaults(root), makefile_injections(root)
    out = {}
    for name in sorted(set(py) & set(mk)):
        if any(is_on(v) for v in py[name]) != is_on(mk[name]):
            out[name] = (sorted(py[name]), mk[name])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--root", default=None, metavar="DIR",
        help="read the configuration off another checkout (e.g. a worktree of "
             "the fpl26-final-submission tag) instead of this one",
    )
    root = ap.parse_args(argv).root

    eff = effective(root)
    on = sorted(n for n, v in eff.items() if v)
    off = sorted(n for n, v in eff.items() if not v)
    print("=== SHIPS ON (clean eval box, no variables set)")
    for n in on:
        print(f"  {n}")
    valued = valued_injections(root)
    print("\n=== ships with a value (armed, but not a boolean flag)")
    for n, v in sorted(valued.items()):
        print(f"  {n} = {v}")
    if not valued:
        print("  none")
    print("\n=== ships off")
    for n in off:
        if n in valued:
            continue
        print(f"  {n}")
    print("\n=== CLI opt-ins (unset ⇒ flag not passed)")
    for var, flag in sorted(cli_optins(root).items()):
        print(f"  {var} ⇒ {flag}")
    dis = layer_disagreements(root)
    print("\n=== layer disagreements (python default vs Makefile injection)")
    for n, (p, m) in dis.items():
        print(f"  {n}: python={p} Makefile={m!r}")
    if not dis:
        print("  none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
