"""The eval box sets NO FPL26_ILS_* variables. What is armed there is what ships.

WHY THIS FILE EXISTS. On jul29 all four uniform-stack flags were found to be DEFAULT OFF
while every measured gain of the campaign had been produced with them explicitly exported
by our A/B driver:

    FPL26_ILS_PLACE_RETRY_LADDER   default "0"
    FPL26_ILS_MEASURED_BASIS       default "0"
    FPL26_ILS_INCR_ROUTE           default "0"
    FPL26_ILS_INCR_ROUTE_FIRST     default "0"

The Makefile's run_optimizer target — the path the organizers actually invoke — exports
FPL26_DEEP_REPLACE, FPL26_DEEP_REPLACE_FIRST, FPL26_DEEP_REPLACE_UNBANDED,
FPL26_DEEP_WNS_TAIL_RESERVE, FPL26_ILS_HURDLE_CONTINUE and FPL26_POLISH_RESERVE_S, and
none of the four above. So the submission would have run WITHOUT the stack that produced
+23.91 MHz over the same-gate control, and spam would have shipped +17.51 instead of
+29.19.

Nothing detected this, because every A/B we ran exported the flags itself. The measurement
harness and the ship path had drifted apart, and only the harness was ever observed. This
is the config-level form of the jul28 lesson that dcp_optimizer.py had never been deployed
to a box, which is why a gate "never ran live" while still producing numbers we believed.

These tests therefore assert the SHIP PATH, not the code's internals: given an environment
with no FPL26_ILS_* set, the features we rely on must be armed, and each kill switch must
still work. If someone flips a default back to opt-in, this fails loudly rather than
silently costing MHz on the one run that is scored.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

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
    """Default-ON must not cost us the ability to turn a feature off for an A/B."""
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
    """The two jul29 ship decisions, pinned at the same level: both are default-ON and
    both are reachable with no environment at all."""
    for key in ("FPL26_DEEP_REPLACE_FIRST_BANDED", "FPL26_VIVADO_LEAK_FIX"):
        monkeypatch.delenv(key, raising=False)
    src = (ROOT / "dcp_optimizer.py").read_text(errors="replace")
    assert '"FPL26_DEEP_REPLACE_FIRST_BANDED", "1"' in src, \
        "band gate must default ON — it prevents corescore losing ~63 MHz to deep-replace"
    mcp = (ROOT / "VivadoMCP" / "vivado_mcp_server.py").read_text(errors="replace")
    assert mcp.count('"FPL26_VIVADO_LEAK_FIX", "1"') == 2, (
        "both leak-fix sites must default ON and agree; a spawn/cleanup split makes "
        "killpg target our own process group")
