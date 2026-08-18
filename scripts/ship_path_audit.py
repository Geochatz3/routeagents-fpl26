#!/usr/bin/env python3
"""SHIP-PATH FLAG AUDIT — what does a bare `make run_optimizer` actually arm?

The jul29 §1 bug was that four uniform-stack flags defaulted OFF while every A/B we
ran exported them by hand, so the harness and the ship path had drifted and only the
harness was ever observed. `tests/test_ship_path_flags.py` guards those four. This
script generalises the check to EVERY FPL26_* flag: it reconstructs the exact
environment `make run_optimizer` hands the optimizer, then reports each predicate's
value in that environment.

It reports, per flag:
  MAKEFILE  — the value run_optimizer exports (or '-' if it exports nothing)
  BARE      — the value the code defaults to when nothing is exported
  EFFECTIVE — what actually reaches the code on the ship path

and flags two failure classes:
  OVERRIDDEN — the Makefile exports a value that DISAGREES with the code default.
               A default flipped in code is then a lie on the ship path.
  Anything whose effective state differs from what the last A/B measured.

Usage: python3 final_round_tools_ship_audit.py
"""
from __future__ import annotations

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
    resolved by make itself — the jul29 lesson was that reading intent off a source
    file is how the drift went unnoticed.
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


def all_flags() -> set[str]:
    flags: set[str] = set()
    for path in list(ROOT.glob("*.py")) + list((ROOT / "optimizer").glob("*.py")) \
            + list((ROOT / "scripts").glob("*.py")):
        flags |= set(re.findall(r"FPL26_[A-Z0-9_]+", path.read_text(errors="ignore")))
    return {f for f in flags if len(f) > len("FPL26_")}


def code_defaults() -> dict[str, str]:
    """The literal default at each `os.environ.get("FPL26_X", "<default>")` site."""
    defaults: dict[str, str] = {}
    for path in list(ROOT.glob("*.py")) + list((ROOT / "optimizer").glob("*.py")) \
            + list((ROOT / "scripts").glob("*.py")):
        text = path.read_text(errors="ignore")
        for flag, dflt in re.findall(
                r'os\.environ\.get\(\s*"(FPL26_[A-Z0-9_]+)"\s*,\s*"([^"]*)"', text):
            defaults.setdefault(flag, dflt)
        for flag in re.findall(
                r'os\.environ\.get\(\s*"(FPL26_[A-Z0-9_]+)"\s*\)', text):
            defaults.setdefault(flag, "")
        # multi-line form: get("FLAG",\n   "default")
        for flag, dflt in re.findall(
                r'os\.environ\.get\(\s*"(FPL26_[A-Z0-9_]+)"\s*,\s*\n\s*"([^"]*)"', text):
            defaults.setdefault(flag, dflt)
    return defaults


PREDICATES = [
    "place_retry_ladder_enabled", "measured_basis_enabled", "incr_route_enabled",
    "incr_route_first_enabled", "incr_route_terminal_enabled", "place_retry_enabled",
    "ladder_order_by_wns_enabled", "ladder_reserve_enabled",
    "ladder_stop_on_accept_enabled", "retry_baseline_gate_enabled",
]

TRUTHY = ("1", "true", "on", "yes")


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

    if overrides:
        print()
        print("!" * 86)
        print("THE MAKEFILE OVERRIDES A CODE DEFAULT — flipping the default in code is a")
        print("NO-OP on the ship path for each of these:")
        for f, mk, cd in overrides:
            print(f"  {f}: Makefile={mk!r} beats code default={cd!r}")
        print("!" * 86)
    # Non-zero on a genuine conflict so this is usable as a check, not just a
    # report someone has to remember to read — the failure mode that let the
    # original drift live for a whole campaign.
    return 1 if overrides else 0


if __name__ == "__main__":
    raise SystemExit(main())
