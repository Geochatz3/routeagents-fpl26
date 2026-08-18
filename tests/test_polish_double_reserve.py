"""FPL26_POLISH_NO_DOUBLE_RESERVE — the deep-replace double-reserve defect
(2acb427) found by the same corpus method in the two post-ILS polish gates.

Every budget number below is a REAL row mined from the jul30 decline census
(124 agent.logs, both Dev Cloud boxes) — see
final_round/POLISH_DOUBLE_RESERVE_jul30.md.  Pinning the corpus rows means a
change to the 1.3 margin, the 300s reserve, or the predicate shape fails in the
corpus's own terms rather than against a hand-picked example.
"""
import os
import types

import pytest

import dcp_optimizer


# --- REAL corpus rows: (design_arm, remaining_s, need_s_as_logged) ----------
# `need` as the shipped code logged it = est*1.3 + 300.
FANOUT_FLIPS = [                    # arm once the double count is removed
    ("fir_systolic__uni3",            465.0, 470.0),
    ("vtr_mcml_v2__shipdef_c",        796.0, 803.0),
    ("fir_systolic__box2ctl",         460.0, 469.0),
    ("rosetta_spam__rec_a_spam",      636.0, 650.0),
    ("fir_systolic__shipdef",         455.0, 470.0),
    ("rosetta_3d__uni",               686.0, 710.0),
    ("vtr_mcml_v2__stock",            721.0, 751.0),
    ("chain44_jul26",                 391.0, 428.0),
    ("vexriscv__uni3b",               397.0, 444.0),
    ("vtr_mcml__hb_on",               760.0, 810.0),
]
LASTMILE_FLIPS = [                  # the high-fmax ones that carry the value
    ("logicnets_jscl__banded",        733.0, 746.0),
    ("rosetta_spam__shipdef2",        736.0, 764.0),
    ("rosetta_spam__shipdefault",     681.0, 766.0),
    ("rosetta_optical__mb",          1070.0, 1210.0),
    ("rosetta_optical__banded",      1040.0, 1210.0),
]
# Genuinely infeasible even after the correction — the discrimination check.
STAY_DECLINED = [
    ("lastmile_worst",                100.0, 1209.0),   # shortfall 809s
    ("fanout_worst",                  120.0,  846.0),   # shortfall 426s
    ("lastmile_median",               300.0,  872.0),   # shortfall 272s
]


class _Stub:
    """Minimal carrier exercising the BOUND shim, which is what the two
    production call sites use.  A stub without `_budget_deadline` must resolve
    to None and keep the reserve (fail-safe) — pinned below."""

    _polish_gate_reserve_s = dcp_optimizer.DCPOptimizer._polish_gate_reserve_s

    def __init__(self, budget_deadline=1_000_000.0):
        self._budget_deadline = budget_deadline


class _BareStub:
    """No `_budget_deadline` at all — the shape every pre-existing polish
    stage test in tests/test_ils_polish.py uses."""

    _polish_gate_reserve_s = dcp_optimizer.DCPOptimizer._polish_gate_reserve_s


def _cfg(reserve=300.0):
    return types.SimpleNamespace(fanout_finalize_reserve_s=reserve)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("FPL26_POLISH_NO_DOUBLE_RESERVE", raising=False)


def _need(remaining_unused, logged_need, reserve):
    """Recompute `need` from the logged one: est*1.3 is (logged - 300)."""
    return (logged_need - 300.0) + reserve


# --------------------------------------------------------------------------
# DEFAULT OFF = byte-identical behaviour
# --------------------------------------------------------------------------

def test_default_off_keeps_the_full_reserve():
    rsv, tag = _Stub()._polish_gate_reserve_s(_cfg())
    assert rsv == 300.0
    assert tag == "reserve 300s"


def test_default_off_reproduces_every_mined_decline():
    rsv, _ = _Stub()._polish_gate_reserve_s(_cfg())
    for name, remaining, logged_need in (FANOUT_FLIPS + LASTMILE_FLIPS
                                         + STAY_DECLINED):
        assert remaining < _need(remaining, logged_need, rsv), (
            f"{name}: shipped default must still decline this corpus row")


@pytest.mark.parametrize("value", ["0", "off", "false", "no", "", "  "])
def test_only_truthy_values_arm_it(monkeypatch, value):
    monkeypatch.setenv("FPL26_POLISH_NO_DOUBLE_RESERVE", value)
    assert _Stub()._polish_gate_reserve_s(_cfg())[0] == 300.0


# --------------------------------------------------------------------------
# ARMED — the correction, priced on the corpus
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["1", "true", "on", "yes", "TRUE", " On "])
def test_armed_drops_the_double_counted_term(monkeypatch, value):
    monkeypatch.setenv("FPL26_POLISH_NO_DOUBLE_RESERVE", value)
    rsv, tag = _Stub()._polish_gate_reserve_s(_cfg())
    assert rsv == 0.0
    assert tag == "reserve already in deadline"


@pytest.mark.parametrize("name,remaining,logged_need", FANOUT_FLIPS)
def test_mined_fanout_declines_arm_once_corrected(monkeypatch, name,
                                                  remaining, logged_need):
    monkeypatch.setenv("FPL26_POLISH_NO_DOUBLE_RESERVE", "1")
    rsv, _ = _Stub()._polish_gate_reserve_s(_cfg())
    assert remaining >= _need(remaining, logged_need, rsv), (
        f"{name}: mined as a flip, must arm after the correction")


@pytest.mark.parametrize("name,remaining,logged_need", LASTMILE_FLIPS)
def test_mined_lastmile_declines_arm_once_corrected(monkeypatch, name,
                                                    remaining, logged_need):
    monkeypatch.setenv("FPL26_POLISH_NO_DOUBLE_RESERVE", "1")
    rsv, _ = _Stub()._polish_gate_reserve_s(_cfg())
    assert remaining >= _need(remaining, logged_need, rsv), (
        f"{name}: mined as a flip, must arm after the correction")


@pytest.mark.parametrize("name,remaining,logged_need", STAY_DECLINED)
def test_correction_is_discriminating_not_permissive(monkeypatch, name,
                                                     remaining, logged_need):
    """The whole safety case: it must NOT arm the genuinely infeasible."""
    monkeypatch.setenv("FPL26_POLISH_NO_DOUBLE_RESERVE", "1")
    rsv, _ = _Stub()._polish_gate_reserve_s(_cfg())
    assert remaining < _need(remaining, logged_need, rsv), (
        f"{name}: cannot finish even corrected — must stay declined")


# --------------------------------------------------------------------------
# The honesty condition and the monotonicity property
# --------------------------------------------------------------------------

def test_no_wall_cap_keeps_the_reserve_even_when_armed(monkeypatch):
    """`_ils_polish_body` falls back to `time.time() + 1200` when there is no
    wall cap, and THAT deadline never had the reserve removed — dropping the
    term there would be a real over-spend, not a correction."""
    monkeypatch.setenv("FPL26_POLISH_NO_DOUBLE_RESERVE", "1")
    rsv, tag = _Stub(budget_deadline=None)._polish_gate_reserve_s(_cfg())
    assert rsv == 300.0
    assert tag == "reserve 300s"


@pytest.mark.parametrize("reserve", [0.0, 120.0, 300.0, 600.0])
def test_correction_can_only_widen_never_narrow(monkeypatch, reserve):
    off, _ = _Stub()._polish_gate_reserve_s(_cfg(reserve))
    monkeypatch.setenv("FPL26_POLISH_NO_DOUBLE_RESERVE", "1")
    on, _ = _Stub()._polish_gate_reserve_s(_cfg(reserve))
    assert on <= off, "the armed gate must never demand MORE than the shipped one"


def test_tag_names_the_treatment_so_a_firing_check_can_key_on_it(monkeypatch):
    """feedback_firing_check_must_key_on_treatment: the A/B is read by
    grepping the log, so the log must record the treatment, not the flag."""
    off_tag = _Stub()._polish_gate_reserve_s(_cfg())[1]
    monkeypatch.setenv("FPL26_POLISH_NO_DOUBLE_RESERVE", "1")
    on_tag = _Stub()._polish_gate_reserve_s(_cfg())[1]
    assert "already in deadline" in on_tag
    assert on_tag != off_tag, "the two arms must be distinguishable in the log"


def test_reserve_is_read_from_cfg_not_hardcoded(monkeypatch):
    """A cfg override must still be honoured when the flag is OFF."""
    rsv, tag = _Stub()._polish_gate_reserve_s(_cfg(450.0))
    assert rsv == 450.0 and tag == "reserve 450s"


def test_module_level_function_is_the_patch_target():
    """Refactor policy jul26: keep patch targets module-global."""
    assert callable(dcp_optimizer.polish_gate_reserve_s)
    assert dcp_optimizer.polish_gate_reserve_s(_cfg(), None)[0] == 300.0


def test_stub_without_budget_deadline_fails_safe(monkeypatch):
    """Every pre-existing polish-stage test drives a stub with no
    `_budget_deadline`.  Those must keep the shipped reserve even when the
    flag is armed, or the fix would silently change unrelated tests."""
    monkeypatch.setenv("FPL26_POLISH_NO_DOUBLE_RESERVE", "1")
    rsv, tag = _BareStub()._polish_gate_reserve_s(_cfg())
    assert rsv == 300.0
    assert tag == "reserve 300s"


# --------------------------------------------------------------------------
# INTEGRATION — the real _fanout_polish_after_ils call site, replaying a mined
# corpus row.  The pure-function tests above cannot catch a call site that
# forgets to consult the helper; this one can.
# --------------------------------------------------------------------------

def _fanout_stage_probe(remaining_s, fan_anchor, budget_deadline_set=True):
    """Drive the REAL stage up to (at most) its first Vivado call.  Returns the
    number of Vivado calls: 0 = gated out, >0 = the gate armed."""
    import asyncio
    import time as _t
    from optimizer.ils_polish import ILSPolishConfig

    calls = []

    class _StageStub:
        run_dir = None

        def __init__(self):
            self._ils_polish_cfg = ILSPolishConfig(
                expected_heavy_cycle_s=0.0,
                fanout_cost_anchor_s=fan_anchor,
                fanout_anchor_has_route=True)
            if budget_deadline_set:
                self._budget_deadline = _t.time() + remaining_s

        async def call_tool(self, name, args):
            calls.append(name)
            return '{"error": "stub"}'

    asyncio.run(dcp_optimizer.DCPOptimizer._fanout_polish_after_ils(
        _StageStub(), "/tmp/best.dcp", -0.5, _t.time() + remaining_s, "SLACK"))
    return len(calls)


# fir_systolic__shipdef, mined: remaining 455s, logged need 470s
# => est*1.3 = 170s, so the fanout anchor that produced it is ~131s.
_FIR_REMAINING_S = 455.0
_FIR_ANCHOR_S = 170.0 / 1.3


def test_integration_shipped_default_still_declines_the_mined_row():
    assert _fanout_stage_probe(_FIR_REMAINING_S, _FIR_ANCHOR_S) == 0


def test_integration_armed_flag_arms_the_mined_row(monkeypatch):
    """455s in hand for a 170s job — refused by the shipped gate for want of a
    reserve the deadline had already taken."""
    monkeypatch.setenv("FPL26_POLISH_NO_DOUBLE_RESERVE", "1")
    assert _fanout_stage_probe(_FIR_REMAINING_S, _FIR_ANCHOR_S) > 0


def test_integration_armed_still_declines_the_genuinely_infeasible(monkeypatch):
    """120s in hand for a 546s job stays declined — discrimination, at the
    real call site rather than only in the pure function."""
    monkeypatch.setenv("FPL26_POLISH_NO_DOUBLE_RESERVE", "1")
    assert _fanout_stage_probe(120.0, 420.0) == 0


def test_integration_no_wall_cap_keeps_the_reserve(monkeypatch):
    """No `_budget_deadline` attribute => fail-safe: the shipped reserve
    stands even armed, so the mined row stays declined."""
    monkeypatch.setenv("FPL26_POLISH_NO_DOUBLE_RESERVE", "1")
    assert _fanout_stage_probe(_FIR_REMAINING_S, _FIR_ANCHOR_S,
                               budget_deadline_set=False) == 0
