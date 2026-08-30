"""Test feature defaults and kill switches on the production entry path.

The runtime environment does not set `FPL26_ILS_*`, so required features must
be enabled by code defaults rather than a separate harness. Tests assert
effective entry-point behavior: each feature remains active in an unset
environment, and each kill switch can still disable it explicitly.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
from tests.source_corpus import dcp_source_lines, dcp_source_text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The flags a scored eval run depends on, and why each one matters.
SHIP_ON = {
    "FPL26_ILS_PLACE_RETRY_LADDER": "place_retry_ladder_enabled",
    "FPL26_ILS_MEASURED_BASIS": "measured_basis_enabled",
    "FPL26_ILS_INCR_ROUTE": "incr_route_enabled",
    "FPL26_ILS_INCR_ROUTE_FIRST": "incr_route_first_enabled",
}


@pytest.fixture
def eval_env(monkeypatch):
    """An environment like the eval box: no FPL26_ILS_* set at all."""
    for key in list(os.environ):
        if key.startswith("FPL26_ILS_"):
            monkeypatch.delenv(key, raising=False)
    import optimizer.ils_polish as ils
    return importlib.reload(ils)


@pytest.mark.parametrize("flag,fn_name", sorted(SHIP_ON.items()))
def test_armed_on_a_bare_eval_environment(eval_env, flag, fn_name):
    """No env, no flags — exactly what `make run_optimizer` gives the optimizer."""
    fn = getattr(eval_env, fn_name)
    assert fn() is True, (
        "%s is OFF with a clean environment. The eval box exports no FPL26_ILS_* "
        "variables, so this feature would not run on the one measured run — while every "
        "A/B we do exports it by hand and therefore still looks correct." % flag)


@pytest.mark.parametrize("flag,fn_name", sorted(SHIP_ON.items()))
def test_kill_switch_still_disarms(monkeypatch, flag, fn_name):
    """Default-ON must not cost the ability to turn a feature off for an A/B."""
    monkeypatch.setenv(flag, "0")
    import optimizer.ils_polish as ils
    ils = importlib.reload(ils)
    assert getattr(ils, fn_name)() is False, "%s=0 must remain a real kill switch" % flag


def test_makefile_ship_path_does_not_silently_rely_on_exported_ils_flags():
    """If the Makefile ever starts exporting these, the defaults above are what protects
    every OTHER invocation path (multi_restart, the direct fallback, a manual run). This
    records the observed state so a future edit is a deliberate one."""
    mk = (ROOT / "Makefile").read_text(errors="replace")
    exported = {f for f in SHIP_ON if f + "=" in mk}
    assert not exported or exported == set(SHIP_ON), (
        "Makefile exports SOME uniform-stack flags (%s) but not all. Partial exporting is "
        "how the harness and the ship path drifted apart in the first place — either "
        "export all of them or rely on the defaults for all of them." % sorted(exported))


def test_band_gate_and_leak_fix_are_armed_on_the_ship_path(monkeypatch):
    """The two ship decisions, pinned at the same level: both are default-ON and
    both are reachable with no environment at all."""
    for key in ("FPL26_DEEP_REPLACE_FIRST_BANDED", "FPL26_VIVADO_LEAK_FIX"):
        monkeypatch.delenv(key, raising=False)
    src = dcp_source_text()
    assert '"FPL26_DEEP_REPLACE_FIRST_BANDED", "1"' in src, \
        "band gate must default ON — it prevents corescore losing ~63 MHz to deep-replace"
    mcp = (ROOT / "VivadoMCP" / "vivado_mcp_server.py").read_text(errors="replace")
    assert mcp.count('"FPL26_VIVADO_LEAK_FIX", "1"') == 2, (
        "both leak-fix sites must default ON and agree; a spawn/cleanup split makes "
        "killpg target our own process group")
