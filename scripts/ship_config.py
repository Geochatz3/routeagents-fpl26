#!/usr/bin/env python3
"""Compute the EFFECTIVE configuration a clean eval box produces.

The contest evaluator unpacks our archive onto a fresh instance and runs
`make run_optimizer DCP=<bench>.dcp` with **no FPL26_* variables set**. So the
only configuration that ever scores us is the one this module reports.

Why this exists — two failures, three days apart, both the same class:

  * jul29: the uniform ILS stack (`PLACE_RETRY_LADDER`/`MEASURED_BASIS`/
    `INCR_ROUTE`/`INCR_ROUTE_FIRST`) was default-OFF and not on the eval path
    at all. A campaign's worth of measured gains would not have shipped.
  * jul30: `FPL26_DEEP_REPLACE_UNBANDED` ships **ON** even though its own code
    comment says "DEFAULT OFF ... Ships OFF until farm-validated, per house
    rule" — because a *different layer* sets it.

The root cause both times: the effective config is a **composition**, and no
single file states it.

    Makefile `run_optimizer` recipe injection     (wins — set before python starts)
        overrides
    python `os.environ.get("FPL26_X", "default")` (applies only when the Makefile
                                                   is bypassed, e.g. running
                                                   dcp_optimizer.py directly)

Reading either layer alone gives a confidently wrong answer. `tests/
test_ship_config.py` pins the composed result so drift fails the suite instead of
waiting for someone to audit it.

Two conventions that read backwards if you skim:
  * `FPL26_NO_X` is a KILL SWITCH — unset means feature X is ENABLED.
  * `$(if $(filter 1 true yes on,$(VAR)),--flag)` is a CLI opt-in — unset means
    the flag is NOT passed.
"""
from __future__ import annotations

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


def main() -> int:
    eff = effective()
    on = sorted(n for n, v in eff.items() if v)
    off = sorted(n for n, v in eff.items() if not v)
    print("=== SHIPS ON (clean eval box, no variables set)")
    for n in on:
        print(f"  {n}")
    print("\n=== ships off")
    for n in off:
        print(f"  {n}")
    print("\n=== CLI opt-ins (unset ⇒ flag not passed)")
    for var, flag in sorted(cli_optins().items()):
        print(f"  {var} ⇒ {flag}")
    dis = layer_disagreements()
    print("\n=== layer disagreements (python default vs Makefile injection)")
    for n, (p, m) in dis.items():
        print(f"  {n}: python={p} Makefile={m!r}")
    if not dis:
        print("  none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
