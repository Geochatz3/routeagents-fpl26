"""v4.0 lever bundle (aug05, branch v40-levers) — unit tests for every new
predicate, the flag discipline, and the gate arithmetic invariants.

THE CONTRACT UNDER TEST (PREREG_V40_LEVERS_aug05.md):
  * all three flags are DEFAULT OFF in python and armed ONLY via the
    Makefile, on BOTH launch branches (jul30 "Makefile not in the ship
    surface" lesson);
  * each FPL26_NO_* kill switch wins over its enable;
  * feature keys are measured-characteristic only (|wns_in| bands, NO
    md5/name keys), with the same None/positive-slack fail-OFF discipline
    as recipe_pass_band;
  * the latency audit (DQ protection) fails ON — unmeasured FF counts
    abort the registration;
  * the primary shallow RECIPE_PASS gate arithmetic is UNTOUCHED (2450 s)
    and the new gates carry their own recomputed fail-closed arithmetic.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dcp_optimizer as d  # noqa: E402
from optimizer.ils_polish import (  # noqa: E402
    ILSPolishConfig,
    _retry_baseline_gate_active,
    retry_baseline_gate_enabled,
)

ALL_V40_ENVS = (
    "FPL26_VEX2_RETIME_CANDIDATE", "FPL26_NO_VEX2_RETIME_CANDIDATE",
    "FPL26_MINIISP_RETRY_HOLD", "FPL26_NO_MINIISP_RETRY_HOLD",
    "FPL26_CORESCORE_ROUTE_RUNG", "FPL26_NO_CORESCORE_ROUTE_RUNG",
    "FPL26_ILS_RETRY_BASELINE_GATE",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Eval-box-like env: none of the v4.0 flags set."""
    for k in ALL_V40_ENVS:
        monkeypatch.delenv(k, raising=False)


FLAG_FNS = [
    ("FPL26_VEX2_RETIME_CANDIDATE", "FPL26_NO_VEX2_RETIME_CANDIDATE",
     d.vex2_retime_candidate_enabled),
    ("FPL26_MINIISP_RETRY_HOLD", "FPL26_NO_MINIISP_RETRY_HOLD",
     d.miniisp_retry_hold_enabled),
    ("FPL26_CORESCORE_ROUTE_RUNG", "FPL26_NO_CORESCORE_ROUTE_RUNG",
     d.corescore_route_rung_enabled),
]


@pytest.mark.parametrize("env,kill,fn", FLAG_FNS,
                         ids=[f[0] for f in FLAG_FNS])
class TestFlagDiscipline:
    def test_default_off(self, env, kill, fn):
        """No env set -> OFF.  The python default must never arm a lever;
        arming is Makefile-only (never-touch-validated-paths rule)."""
        assert fn() is False

    @pytest.mark.parametrize("val", ["1", "true", "on", "yes", " 1 ", "TRUE"])
    def test_armed_by_truthy(self, env, kill, fn, val, monkeypatch):
        monkeypatch.setenv(env, val)
        assert fn() is True

    @pytest.mark.parametrize("val", ["0", "false", "off", "", "no", "2"])
    def test_non_truthy_stays_off(self, env, kill, fn, val, monkeypatch):
        monkeypatch.setenv(env, val)
        assert fn() is False

    def test_kill_switch_wins(self, env, kill, fn, monkeypatch):
        monkeypatch.setenv(env, "1")
        monkeypatch.setenv(kill, "1")
        assert fn() is False

    def test_kill_switch_alone_is_off(self, env, kill, fn, monkeypatch):
        monkeypatch.setenv(kill, "1")
        assert fn() is False


class TestVex2RetimeSubband:
    """[0.60, 1.05] inclusive; None / positive / garbage fail OFF."""

    @pytest.mark.parametrize("wns", [
        -0.946,   # vexriscv_v2 — the evidence design
        -0.686,   # spam (free upside under the MUX)
        -0.978,   # logicnets
        -1.025,   # digit
        -0.60,    # lower bound inclusive
        -1.05,    # upper bound inclusive (== shallow band max)
    ])
    def test_in_band(self, wns):
        assert d.vex2_retime_subband_match(wns) is True

    @pytest.mark.parametrize("wns", [
        -0.313,   # fir — carve-out zone, must never see the chain
        -0.50,    # fir carve-out edge
        -0.599,   # just below the band
        -1.078,   # optical — excluded (deterministic shallow-pass harm)
        -1.051,   # just past the shallow bound
        -8.0,     # deep
        None,     # unmeasured -> treatments fail OFF
        0.0,      # met
        0.5,      # positive slack
        "garbage",
    ])
    def test_out_of_band(self, wns):
        assert d.vex2_retime_subband_match(wns) is False

    def test_band_top_matches_shallow_band(self):
        """The sub-band may never exceed the shallow band it lives in."""
        assert (d.VEX2_RETIME_WNS_MAG_MAX_NS
                == d.RECIPE_PASS_SHALLOW_WNS_MAG_MAX_NS == 1.05)
        assert d.VEX2_RETIME_WNS_MAG_MIN_NS > d.FIR_CARVEOUT_WNS_MAG_MAX_NS


class TestVex2LatencyAudit:
    """DQ protection: protections fail ON (unmeasured -> abort)."""

    def test_q07_measured_pair_passes(self):
        # q07 FF audit: 1678 -> 1682 (+0.24%, replication-class moves).
        assert d.vex2_retime_ff_drift_ok(1678, 1682) is True

    def test_exact_one_percent_inclusive(self):
        assert d.vex2_retime_ff_drift_ok(1000, 1010) is True
        assert d.vex2_retime_ff_drift_ok(1000, 990) is True

    def test_beyond_one_percent_aborts(self):
        assert d.vex2_retime_ff_drift_ok(1000, 1011) is False
        assert d.vex2_retime_ff_drift_ok(1000, 989) is False

    @pytest.mark.parametrize("before,after", [
        (None, 1682), (1678, None), (None, None),
        ("x", 1682), (1678, "x"),
        (0, 0), (-5, 100), (100, -1),
    ])
    def test_unmeasured_or_degenerate_fails_closed(self, before, after):
        assert d.vex2_retime_ff_drift_ok(before, after) is False

    def test_identity_passes(self):
        assert d.vex2_retime_ff_drift_ok(1678, 1678) is True


class TestVex2FrozenChain:
    """The chain is FROZEN (feedback_same_build_not_just_same_config):
    the q07 effective recipe, place-first (retime from an unrouted basin
    measured actively harmful, -0.874 x3)."""

    def test_frozen_steps_verbatim(self):
        assert d.VEX2_RETIME_TCL == (
            "route_design -unroute",
            "place_design -unplace",
            "place_design -directive ExtraTimingOpt",
            "phys_opt_design -retime",
            "phys_opt_design -directive AggressiveExplore",
            "route_design -directive AggressiveExplore",
        )

    def test_retime_after_place(self):
        steps = list(d.VEX2_RETIME_TCL)
        assert (steps.index("phys_opt_design -retime")
                > steps.index("place_design -directive ExtraTimingOpt"))


class TestGateArithmetic:
    """Recomputed budgets — the numbers the prereg cites."""

    def test_primary_shallow_gate_untouched(self):
        """need = 1.5x300 + 900 + 1100 = 2450 s, exactly as validated."""
        assert (d.RECIPE_PASS_TIMEOUT_FACTOR * d.RECIPE_PASS_SHALLOW_EXPECTED_S
                + d.RECIPE_PASS_OVERHEAD_RESERVE_S + 1100.0) == 2450.0

    def test_vex2_second_candidate_need(self):
        """need2 = 1.5x460 + 540 + 1100 = 2330 s <= 3200 s max remaining;
        the second candidate is fundable on the ship wall after a typical
        primary pass, and fail-closed otherwise."""
        cap = d.RECIPE_PASS_TIMEOUT_FACTOR * d.VEX2_RETIME_EXPECTED_S
        assert cap == 690.0
        need2 = cap + d.VEX2_RETIME_OVERHEAD_RESERVE_S + 1100.0
        assert need2 == 2330.0
        assert need2 <= 3200.0
        # no reset term in the second candidate's overheads — the caller's
        # single post-pass reset covers both candidates.
        assert d.VEX2_RETIME_OVERHEAD_RESERVE_S == 540.0

    def test_route_rung_fits_stranded_wall(self):
        """need = 500 + 540 = 1040 s — inside the measured 750-1300 s
        stranded band's upper half and far below the deep postloop slot's
        1890 s; hard pass deadline 1.5x500 = 750 s."""
        need = (d.CORESCORE_ROUTE_RUNG_EXPECTED_S
                + d.RECIPE_PASS_POSTLOOP_OVERHEAD_RESERVE_S)
        assert need == 1040.0
        assert need <= 1300.0
        deep_need = (d.RECIPE_PASS_TIMEOUT_FACTOR
                     * d.RECIPE_PASS_DEEP_EXPECTED_S
                     + d.RECIPE_PASS_POSTLOOP_OVERHEAD_RESERVE_S)
        assert deep_need == 1890.0
        assert need < deep_need
        assert (d.RECIPE_PASS_TIMEOUT_FACTOR
                * d.CORESCORE_ROUTE_RUNG_EXPECTED_S) == 750.0

    def test_route_rung_tcl_frozen(self):
        assert d.CORESCORE_ROUTE_RUNG_TCL == "route_design -directive Explore"


class TestMiniispRetryHoldScope:
    """Band scoping facts the arming site relies on (recipe_pass_band on
    the PRISTINE initial_wns)."""

    @pytest.mark.parametrize("wns,band", [
        (-1.686, "mid"),      # mini-ISP — the evidence design
        (-1.238, "mid"),      # corescore
        (-1.078, "mid"),      # optical — IN scope; disclosed prereg risk
        (-0.686, "shallow"),  # spam — the jul29 clean harm, OUT by scope
        (-0.313, "shallow"),  # fir
        (-1.05, "shallow"),   # boundary inclusive to shallow
        (-8.0, "deep"),       # boundary inclusive to deep
        (-14.527, "deep"),    # vtr
        (None, None),
        (0.5, None),
    ])
    def test_band(self, wns, band):
        assert d.recipe_pass_band(wns) == band


class TestRetryBaselineGateActive:
    """OR-composition: global env flag untouched, scoped field additive."""

    def test_default_config_field_off(self):
        assert ILSPolishConfig().retry_baseline_gate_scoped is False

    def test_neither_armed(self):
        cfg = ILSPolishConfig()
        assert retry_baseline_gate_enabled() is False
        assert _retry_baseline_gate_active(cfg) is False

    def test_scoped_alone_arms(self):
        cfg = ILSPolishConfig()
        cfg.retry_baseline_gate_scoped = True
        assert _retry_baseline_gate_active(cfg) is True
        # the GLOBAL flag read stays untouched
        assert retry_baseline_gate_enabled() is False

    def test_global_alone_arms(self, monkeypatch):
        monkeypatch.setenv("FPL26_ILS_RETRY_BASELINE_GATE", "1")
        cfg = ILSPolishConfig()
        assert _retry_baseline_gate_active(cfg) is True

    def test_missing_field_fails_off(self):
        """A cfg object without the field (older pickle / duck type) must
        read as unscoped, not raise."""
        class _Bare:
            pass
        assert _retry_baseline_gate_active(_Bare()) is False


class TestManifestRecords:
    """Every decision-changing flag must be readable back from a banked
    row (the _MANIFEST_FLAGS omission class bit twice)."""

    @pytest.mark.parametrize("flag", [e for e in ALL_V40_ENVS
                                      if e != "FPL26_ILS_RETRY_BASELINE_GATE"])
    def test_flag_in_manifest(self, flag):
        assert flag in d.DCPOptimizer._MANIFEST_FLAGS


class TestMakefileArming:
    """Both launch branches (wrapper + `||` safety net) must arm all
    three flags — one branch armed alone measures nothing (jul30)."""

    @pytest.fixture(scope="class")
    def branches(self):
        text = (ROOT / "Makefile").read_text()
        wrapper = [ln for ln in text.splitlines()
                   if "multi_restart_optimize.py" in ln
                   and "FPL26_FIR_SUBBAND_FLOOR" in ln]
        fallback = [ln for ln in text.splitlines()
                    if "dcp_optimizer.py" in ln and ln.strip().startswith("||")
                    and "FPL26_FIR_SUBBAND_FLOOR" in ln]
        assert len(wrapper) == 1 and len(fallback) == 1
        return wrapper[0], fallback[0]

    @pytest.mark.parametrize("var,knob", [
        ("FPL26_VEX2_RETIME_CANDIDATE", "VEX2_RETIME"),
        ("FPL26_MINIISP_RETRY_HOLD", "MINIISP_RETRY_HOLD"),
        ("FPL26_CORESCORE_ROUTE_RUNG", "CORESCORE_ROUTE_RUNG"),
    ])
    def test_both_branches_armed(self, branches, var, knob):
        expected = f"{var}=$(if $({knob}),$({knob}),1)"
        for branch in branches:
            assert expected in branch


class TestPreShipAcks:
    """Post-review pre-ship fixes (PREREG post-review additions):
    FF-probe sentinel anchor + disabled-audit-line log-byte parity."""

    def test_ff_probe_tcl_carries_sentinel(self):
        assert d.VEX2_RETIME_FF_COUNT_TCL.startswith("puts FFCOUNT=")
        assert "PRIMITIVE_TYPE =~ REGISTER.*" in d.VEX2_RETIME_FF_COUNT_TCL

    @pytest.mark.parametrize("res,expect", [
        ("FFCOUNT=1678", 1678),
        ("INFO: [Common 17-206] blah 42\nFFCOUNT=1678\n", 1678),
        ("1678", None),                      # unanchored number never binds
        ("INFO 99 nothing here", None),
        ("FFCOUNT=abc", None),
        ("", None),
        (None, None),
        (1678, None),                        # non-str fails closed
    ])
    def test_sentinel_parse_fail_closed(self, res, expect):
        assert d.vex2_retime_parse_ffcount(res) == expect

    def test_env_present_gates_audit_line(self, monkeypatch):
        # absent -> False (log-byte parity with v3.3)
        assert d.v40_flag_env_present(
            "FPL26_VEX2_RETIME_CANDIDATE",
            "FPL26_NO_VEX2_RETIME_CANDIDATE") is False
        # present-but-off ("0" knob, the A/B OFF arm) -> True (line audits)
        monkeypatch.setenv("FPL26_VEX2_RETIME_CANDIDATE", "0")
        assert d.v40_flag_env_present(
            "FPL26_VEX2_RETIME_CANDIDATE",
            "FPL26_NO_VEX2_RETIME_CANDIDATE") is True
        # kill switch alone also audits
        monkeypatch.delenv("FPL26_VEX2_RETIME_CANDIDATE")
        monkeypatch.setenv("FPL26_NO_VEX2_RETIME_CANDIDATE", "1")
        assert d.v40_flag_env_present(
            "FPL26_VEX2_RETIME_CANDIDATE",
            "FPL26_NO_VEX2_RETIME_CANDIDATE") is True


class TestRetryHoldSubbandFloor:
    """aug05 re-scope (pre-registered prereg-risk-#2 fallback): the retry-hold
    floor at |wns_in| >= 1.20 — optical-class [1.05, 1.20) reverts to v3.3."""

    def test_floor_constant(self):
        assert d.MINIISP_RETRY_HOLD_WNS_MAG_MIN_NS == 1.20

    @pytest.mark.parametrize("wns,held", [
        (-1.078, False),   # optical — parity-measured harm, MUST be out
        (-1.05, False),    # band boundary itself is shallow anyway
        (-1.051, False),   # mid but below floor
        (-1.199, False),   # just below floor
        (-1.20, True),     # inclusive floor
        (-1.238, True),    # corescore
        (-1.686, True),    # mini-ISP (the evidence design)
        (-1.91, True),     # finn
    ])
    def test_floor_scope(self, wns, held):
        band = d.recipe_pass_band(wns)
        in_scope = (band == "mid"
                    and abs(wns) >= d.MINIISP_RETRY_HOLD_WNS_MAG_MIN_NS)
        assert in_scope is held
