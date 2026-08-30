#!/usr/bin/env python3
"""Audit the feature flags enabled by a bare `make run_optimizer` invocation.

The audit reconstructs the environment passed to the optimizer and reports each
`FPL26_*` predicate as:
- `MAKEFILE`: the exported value, or `-` when absent.
- `BARE`: the code default without an export.
- `EFFECTIVE`: the value received on the normal execution path.

A flag is `OVERRIDDEN` when its Makefile value disagrees with the code default.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def makefile_exports() -> dict[str, str]:
    """Parse the FPL26_* assignments on run_optimizer's command line.

    Uses `make -n` rather than a regex over the recipe so that $(if ...) defaults are
    resolved by make itself — the lesson from the original drift was that reading
    intent off a source file is how it went unnoticed.
    """
    try:
        out = subprocess.run(
            ["make", "-n", "run_optimizer", "DCP=/dev/null"],
            cwd=ROOT, capture_output=True, text=True, timeout=60).stdout
    except Exception:
        out = ""
    if "FPL26_" not in out:
        # make refuses without a real DCP; fall back to expanding the recipe text.
        recipe = (ROOT / "Makefile").read_text()
        out = recipe
        # resolve $(if $(X),$(X),DEFAULT) with no X set -> DEFAULT
        out = re.sub(r"\$\(if \$\([A-Z_0-9]+\),\$\([A-Z_0-9]+\),([^)]*)\)", r"\1", out)
        out = re.sub(r"\$\(if \$\([A-Z_0-9]+\),[^)]*\)", "", out)
    return dict(re.findall(r"\b(FPL26_[A-Z0-9_]+)=([^\s\\]*)", out))


FLAG_RE = re.compile(r"FPL26_[A-Z0-9_]+\Z")


def all_flags() -> set[str]:
    """Every FPL26_* name that appears as a COMPLETE string literal in code.

    A regex over raw file text also matched prose. `scripts/ship_config.py`
    documents its return shape as `{FPL26_NAME: ...}`, so the audit printed a
    `FPL26_NAME` row for a flag that does not exist, next to the real ones —
    and `FPL26_X`, `FPL26_NO_X` from the same docstring habit. Reading the
    literals out of the parsed AST keeps the two apart without an exclusion
    list that would need its own maintenance.

    Requiring the WHOLE literal to be a flag name also drops
    `"===FPL26_ROUTE_STATUS==="`, a Tcl output marker, which was never an
    environment flag either. (The old trailing `len(f) > len("FPL26_")` guard
    caught none of this: the regex's `+` already guaranteed the length, so it
    read as protection while removing nothing.)
    """
    flags: set[str] = set()
    for path in _source_files():
        try:
            tree = ast.parse(path.read_text(errors="ignore"))
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and FLAG_RE.match(node.value)):
                flags.add(node.value)
    return flags


def _source_files():
    """Every directory that can define or read an FPL26_* flag.

    The MCP servers were missing, so FPL26_VIVADO_LEAK_FIX — which is read
    in VivadoMCP and decides whether Vivado gets its own process group —
    printed its code default as `?`."""
    roots = ("optimizer", "scripts", "VivadoMCP", "RapidWrightMCP",
             "scheduler", "recipes")
    files = list(ROOT.glob("*.py"))
    for d in roots:
        files += list((ROOT / d).glob("*.py"))
    return files


# Quote-agnostic: FPL26_WSL2_WIN_CWD is read as
# `os.environ.get('FPL26_WSL2_WIN_CWD') or '/mnt/c/'`, and a double-quote-only
# pattern reported its code default as `?` — an unknown default in the one
# table whose job is to show defaults.
_Q = r"[\"']"


def code_defaults() -> dict[str, str]:
    """The literal default at each `os.environ.get("FPL26_X", "<default>")` site."""
    defaults: dict[str, str] = {}
    for path in _source_files():
        text = path.read_text(errors="ignore")
        for flag, dflt in re.findall(
                rf'os\.environ\.get\(\s*{_Q}(FPL26_[A-Z0-9_]+){_Q}\s*,\s*{_Q}([^"\']*){_Q}',
                text):
            defaults.setdefault(flag, dflt)
        for flag in re.findall(
                rf'os\.environ\.get\(\s*{_Q}(FPL26_[A-Z0-9_]+){_Q}\s*\)', text):
            defaults.setdefault(flag, "")
        # multi-line form: get("FLAG",\n   "default")
        for flag, dflt in re.findall(
                rf'os\.environ\.get\(\s*{_Q}(FPL26_[A-Z0-9_]+){_Q}\s*,\s*\n\s*{_Q}([^"\']*){_Q}',
                text):
            defaults.setdefault(flag, dflt)
    return defaults


PREDICATES = [
    "place_retry_ladder_enabled", "measured_basis_enabled", "incr_route_enabled",
    "incr_route_first_enabled", "incr_route_terminal_enabled", "place_retry_enabled",
    "ladder_order_by_wns_enabled", "ladder_reserve_enabled",
    "ladder_stop_on_accept_enabled", "retry_baseline_gate_enabled",
]

TRUTHY = ("1", "true", "on", "yes")

# Ship decisions that are MEANT to defeat their code default, as
# (makefile, code-default) pairs. Each is the "DEFAULT OFF docstring, shipped
# ON" class documented under "Honesty notes" in docs/CONFIGURATION.md, and
# each was settled before the scored run.
#
# Without this list the audit returned 1 on a clean tree, every run, forever
# — so the check docs/CONFIGURATION.md tells readers to run was permanently
# red and its exit code carried no signal at all. That is the same failure
# the comment in main() set out to fix by suppressing four benign flags;
# four deliberate ones simply took their place. Declaring them moves the
# exit code back to meaning "something CHANGED".
#
# The pair, not just the name, is what is acknowledged: if the Makefile later
# injects a different value, the override stops matching and fires again.
ACKNOWLEDGED_OVERRIDES = {
    "FPL26_B3_FLOOR_EXIT": ("1", "0"),
    "FPL26_DEEP_REPLACE_B3": ("1", "0"),
    "FPL26_ILS_MEASURED_PRIORS": ("1", "0"),
    "FPL26_LOGIC_FLOOR_EXIT": ("1", "0"),
}


def classify_overrides(overrides, acknowledged=None):
    """Split observed overrides into (declared, undeclared, stale).

    `overrides` is the [(flag, makefile, code_default)] the table produced.
    Declared means the exact (makefile, code-default) PAIR is acknowledged;
    stale means an acknowledgement matches nothing observed. Only undeclared
    and stale are conflicts.
    """
    ack = ACKNOWLEDGED_OVERRIDES if acknowledged is None else acknowledged
    declared = [(f, mk, cd) for f, mk, cd in overrides
                if ack.get(f) == (mk.strip(), cd.strip())]
    undeclared = [t for t in overrides if t not in declared]
    stale = sorted(set(ack) - {f for f, _, _ in overrides})
    return declared, undeclared, stale


def main() -> int:
    exports = makefile_exports()
    defaults = code_defaults()
    flags = sorted(all_flags())

    print("=" * 86)
    print("SHIP-PATH FLAG AUDIT — bare `make run_optimizer`")
    print("=" * 86)
    print(f"{'flag':44s} {'makefile':>10s} {'code-dflt':>10s} {'effective':>10s}  note")
    print("-" * 86)
    overrides = []
    for f in flags:
        mk = exports.get(f)
        cd = defaults.get(f)
        eff = mk if mk is not None else cd
        note = ""
        # Only a NON-EMPTY code default can be "overridden". An empty default means
        # "unset = off", so the Makefile is that flag's only source — the intended
        # design, not drift. Flagging those too made the alarm fire on four benign
        # flags every single run, and an audit that always screams is one nobody
        # reads, which is how the original drift survived in the first place.
        if (mk is not None and cd is not None and cd.strip() != ""
                and mk.strip() != cd.strip()):
            mk_t = mk.strip().lower() in TRUTHY
            cd_t = cd.strip().lower() in TRUTHY
            if mk_t != cd_t:
                note = "OVERRIDDEN — Makefile defeats a deliberate code default"
                overrides.append((f, mk, cd))
        elif mk is not None and (cd is None or cd.strip() == ""):
            note = "makefile-only (its only source; not drift)"
        print(f"{f:44s} {str(mk if mk is not None else '-'):>10s} "
              f"{str(cd if cd is not None else '?'):>10s} "
              f"{str(eff if eff is not None else '?'):>10s}  {note}")

    print()
    print("-" * 86)
    print("PREDICATE STATE in the bare eval environment (no FPL26_* set at all)")
    print("-" * 86)
    for key in [k for k in os.environ if k.startswith("FPL26_")]:
        del os.environ[key]
    import optimizer.ils_polish as ils
    for name in PREDICATES:
        fn = getattr(ils, name, None)
        if fn is None:
            print(f"  {name:36s} MISSING")
            continue
        try:
            print(f"  {name:36s} {'ARMED' if fn() else 'off'}")
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"  {name:36s} ERROR {exc}")

    print()
    print("-" * 86)
    print("PREDICATE STATE with the Makefile's exports applied (the REAL ship path)")
    print("-" * 86)
    os.environ.update(exports)
    import importlib
    ils = importlib.reload(ils)
    for name in PREDICATES:
        fn = getattr(ils, name, None)
        if fn is None:
            continue
        try:
            print(f"  {name:36s} {'ARMED' if fn() else 'off'}")
        except Exception as exc:  # pragma: no cover
            print(f"  {name:36s} ERROR {exc}")

    # An acknowledgement that no longer describes anything is not harmless:
    # left in place it would silently absorb the flag the day it is
    # overridden again — so `stale` is a conflict too.
    declared, undeclared, stale = classify_overrides(overrides)

    if declared:
        print()
        print("-" * 86)
        print("DECLARED ship-path overrides (documented in docs/CONFIGURATION.md,")
        print("\"Honesty notes\") — the Makefile injection IS the shipped decision:")
        for f, mk, cd in declared:
            print(f"  {f}: Makefile={mk!r} over code default={cd!r}")

    if undeclared:
        print()
        print("!" * 86)
        print("THE MAKEFILE OVERRIDES A CODE DEFAULT — flipping the default in code is a")
        print("NO-OP on the ship path for each of these:")
        for f, mk, cd in undeclared:
            print(f"  {f}: Makefile={mk!r} beats code default={cd!r}")
        print("!" * 86)

    if stale:
        print()
        print("!" * 86)
        print("STALE ACKNOWLEDGED_OVERRIDES — these no longer override anything, so")
        print("the entry would absorb a REAL override if one reappeared. Remove them:")
        for f in stale:
            print(f"  {f}")
        print("!" * 86)

    # Non-zero on a genuine conflict so this is usable as a check, not just a
    # report someone has to remember to read — the failure mode that let the
    # original drift live for a whole campaign. Declared overrides are not
    # conflicts; an undeclared override, or a declaration that has gone
    # stale, is.
    return 1 if (undeclared or stale) else 0


if __name__ == "__main__":
    raise SystemExit(main())
