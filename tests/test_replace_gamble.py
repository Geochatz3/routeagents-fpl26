"""Test the insured terminal replacement-placement stage without invoking the FPGA
tool.

The stage is disabled by default and skips when no anchor or reserve is
available. Adoption requires the configured 0.15 ns margin over chain-best and
a passing hold check. Tool errors never bank results, banked-best artifacts
remain protected, thresholds ratchet across draws, and termination depends on
whether completed draws produce valid results.
"""
import asyncio
import time as _time

from optimizer.ils_polish import ILSPolishConfig
from optimizer.replace_gamble import (
    REPLACE_GAMBLE_COST_MARGIN,
    REPLACE_GAMBLE_VARIANTS,
    ReplaceGambleResult,
    replace_gamble_accept,
    replace_gamble_cost_basis,
    replace_gamble_should_run,
    run_replace_gamble,
)


# Config defaults (RC ships OFF; panel constants)

def test_default_off():
    # The RC must ship with the gamble OFF; the all-in window flips it on.
    assert ILSPolishConfig().replace_gamble_enabled is False


def test_config_panel_constants():
    cfg = ILSPolishConfig()
    assert cfg.replace_gamble_adopt_margin_ns == 0.15     # round-2 gate
    assert cfg.replace_gamble_max_draws == 2              # panel draw count
    assert cfg.replace_gamble_finalize_reserve_s == 300.0
    assert cfg.replace_gamble_min_route_delay_frac == 0.6  # loosened 0.7->0.6
    assert cfg.critical_path_route_delay_frac is None     # Phase 1: unmeasured


def test_variants_axes():
    # net_delay_weight axis is {medium, high} only (round-2: "{med,high}").
    for label, pd, ndw, ppo in REPLACE_GAMBLE_VARIANTS:
        assert ndw in ("medium", "high")
        assert pd in ("Explore", "ExtraTimingOpt")
        # -clock_vtree_type axis DROPPED (xcvu3p = single-SLR UltraScale+;
        # the knob is SLR/SSIT-oriented per 2025.1 UG835).
        assert "vtree" not in label
    # First draw = the 5/5 panel pick: Explore + ndw high.
    assert REPLACE_GAMBLE_VARIANTS[0][:3] == ("Explore+ndw_high", "Explore",
                                              "high")
    # -post_place_opt is a separate step on exactly one variant, last in
    # the ladder (beyond the default 2-draw window).
    ppo_idx = [i for i, v in enumerate(REPLACE_GAMBLE_VARIANTS) if v[3]]
    assert ppo_idx == [len(REPLACE_GAMBLE_VARIANTS) - 1]


# Cost basis (fail closed on unknown)

def test_cost_basis_prefers_observed_explore_cycle():
    cfg = ILSPolishConfig(expected_heavy_cycle_s=1500.0)
    # combo index 0 is the Explore full-ruin cycle in ILS_COMBOS.
    assert replace_gamble_cost_basis(cfg, {0: 800.0}) == 800.0


def test_cost_basis_falls_back_to_full_cycle_anchor():
    cfg = ILSPolishConfig(expected_heavy_cycle_s=1500.0)
    assert replace_gamble_cost_basis(cfg, {}) == 1500.0
    assert replace_gamble_cost_basis(cfg, None) == 1500.0
    # a non-Explore observed cost does not stand in for the Explore cycle
    assert replace_gamble_cost_basis(cfg, {3: 200.0}) == 1500.0


def test_cost_basis_unknown_is_zero():
    assert replace_gamble_cost_basis(ILSPolishConfig(), {}) == 0.0


# Firing gate (pure)

def _should_run(**kw):
    base = dict(cfg=ILSPolishConfig(replace_gamble_enabled=True),
                best_wns=-0.8, best_path="/tmp/best.dcp",
                remaining_s=5000.0, cost_basis_s=1000.0)
    base.update(kw)
    return replace_gamble_should_run(**base)


def test_gate_disabled_is_noop():
    run, why = _should_run(cfg=ILSPolishConfig())
    assert run is False and "disabled" in why


def test_gate_requires_banked_best():
    assert _should_run(best_wns=None)[0] is False
    assert _should_run(best_path=None)[0] is False
    assert _should_run(best_path="")[0] is False


def test_gate_fails_closed_without_anchor():
    run, why = _should_run(cost_basis_s=0.0)
    assert run is False and "fail closed" in why


def test_gate_terminal_reserve_boundary():
    # need = 1000*1.3 + 300 = 1600
    assert _should_run(remaining_s=1599.0)[0] is False
    assert _should_run(remaining_s=1601.0)[0] is True


def test_gate_route_delay_frac():
    cfg_lo = ILSPolishConfig(replace_gamble_enabled=True,
                             critical_path_route_delay_frac=0.5)
    cfg_hi = ILSPolishConfig(replace_gamble_enabled=True,
                             critical_path_route_delay_frac=0.7)
    assert _should_run(cfg=cfg_lo)[0] is False
    assert _should_run(cfg=cfg_hi)[0] is True
    # None (unmeasured) -> fail-open, per the build-regardless directive.
    assert _should_run()[0] is True


# Adopt gate (pure)

def _accept(**kw):
    base = dict(new_wns=-0.5, best_wns=-0.8, unrouted=0, whs=0.02,
                cfg=ILSPolishConfig())
    base.update(kw)
    return replace_gamble_accept(**base)


def test_adopt_at_panel_margin():
    ok, _ = _accept(new_wns=-0.65)          # +0.15 exactly -> adopt (>=)
    assert ok is True
    ok, why = _accept(new_wns=-0.651)       # +0.149 -> below the bar
    assert ok is False and "adopt margin" in why


def test_adopt_rejects_unrouted():
    assert _accept(unrouted=3)[0] is False
    assert _accept(unrouted=None)[0] is False
    assert _accept(unrouted=-1)[0] is False


def test_adopt_rejects_missing_wns():
    assert _accept(new_wns=None)[0] is False
    assert _accept(best_wns=None)[0] is False


def test_adopt_hold_gate_official_floor():
    # official scorecard gate passes whs=0.0
    assert _accept(whs=0.0)[0] is True
    assert _accept(whs=-0.0005)[0] is True   # cfg floor -0.001
    assert _accept(whs=-0.01)[0] is False
    assert _accept(whs=None)[0] is False


# Runner (fake call_tool; instant, so all timings are ~0)

def _cfg_run(**kw):
    base = dict(replace_gamble_enabled=True, expected_heavy_cycle_s=100.0)
    base.update(kw)
    return ILSPolishConfig(**base)


def _fake_vivado(wns_seq, *, hold="0.05", place_resp="ok", write_resp="ok",
                 unrouted=0):
    """Fake Vivado in the test_ils_polish idiom: SLACK queries pop wns_seq
    (last value repeats); records every (tool, command) call."""
    calls = []
    state = {"i": 0}

    async def fake(tool, args):
        cmd = (args or {}).get("command", "")
        calls.append((tool, cmd))
        if tool == "vivado_restart_vivado":
            return "ok"
        if "write_checkpoint" in cmd:
            return write_resp
        if "place_design -directive" in cmd:
            return place_resp
        if "-hold" in cmd:
            return hold
        if "report_route_status" in cmd:
            n_ok = 10 - unrouted
            return (f"# of routable nets : 10\n"
                    f"# of fully routed nets : {n_ok}\n"
                    f"# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            i = min(state["i"], len(wns_seq) - 1)
            state["i"] += 1
            return str(wns_seq[i])
        return "ok"
    return fake, calls


def _run(fake, cfg, best_wns=-0.8, deadline_offset=5000.0):
    logs = []
    res = asyncio.run(run_replace_gamble(
        fake, best_dcp_path="/tmp/banked_best.dcp", best_wns=best_wns,
        deadline_ts=_time.time() + deadline_offset, wns_tcl="SLACK",
        cfg=cfg, log=logs.append, out_dcp_base="/tmp/rg"))
    return res, logs


def test_runner_kill_switch_no_vivado():
    fake, calls = _fake_vivado([-0.5])
    res, logs = _run(fake, _cfg_run(replace_gamble_enabled=False))
    assert res.attempted is False and "disabled" in res.skip_reason
    assert calls == []           # zero diff with the flag off
    assert res.adopted is False


def test_runner_fails_closed_without_anchor():
    fake, calls = _fake_vivado([-0.5])
    res, _ = _run(fake, _cfg_run(expected_heavy_cycle_s=0.0))
    assert res.attempted is False and "fail closed" in res.skip_reason
    assert calls == []


def test_runner_adopt_happy_path():
    # chain-best -0.8; draw measures -0.5 (+0.3 > +0.15) -> ADOPTED
    fake, calls = _fake_vivado([-0.5])
    res, logs = _run(fake, _cfg_run())
    assert res.attempted and res.adopted
    assert res.best_wns == -0.5
    assert res.best_path == "/tmp/rg_d1.dcp"
    assert res.draws[0].verdict == "ADOPTED"
    assert res.draws[0].completed_valid is True
    # INSURANCE: the banked best is only ever an open_checkpoint target.
    for _tool, cmd in calls:
        if "write_checkpoint" in cmd:
            assert "banked_best" not in cmd
    # first draw uses the 5/5 knob combination
    assert any("place_design -directive Explore -net_delay_weight high"
               in cmd for _t, cmd in calls)
    # structured kill-test line
    assert any("REPLACE_GAMBLE attempt=1 variant=Explore+ndw_high" in m
               and "verdict=ADOPTED" in m for m in logs)


def test_runner_reject_below_margin_counts_completed_valid():
    # +0.05 improvement: real, valid, but below the +0.15 panel bar.
    fake, calls = _fake_vivado([-0.75])
    res, logs = _run(fake, _cfg_run(replace_gamble_max_draws=1))
    assert res.attempted and not res.adopted
    assert res.draws[0].verdict == "REJECTED"
    assert res.draws[0].completed_valid is True    # kill-test numerator
    assert not any("write_checkpoint" in cmd and "rg_d" in cmd
                   for _t, cmd in calls)
    assert res.best_wns == -0.8                    # bar unchanged
    assert any("verdict=REJECTED" in m for m in logs)


def test_runner_hold_dirty_rejected_not_valid():
    fake, _ = _fake_vivado([-0.4], hold="-0.05")
    res, _ = _run(fake, _cfg_run(replace_gamble_max_draws=1))
    assert res.draws[0].verdict == "REJECTED"
    assert res.draws[0].completed_valid is False   # hold-dirty != valid
    assert not res.adopted


def test_runner_place_error_envelope():
    fake, calls = _fake_vivado([-0.5], place_resp='{"error": "timeout"}')
    res, logs = _run(fake, _cfg_run(replace_gamble_max_draws=1))
    assert res.draws[0].verdict == "ERROR"
    assert res.draws[0].completed_valid is False
    assert not res.adopted and res.best_wns == -0.8
    assert not any("write_checkpoint" in cmd for _t, cmd in calls)
    assert any("verdict=ERROR" in m for m in logs)


def test_runner_two_errors_stop():
    fake, _ = _fake_vivado([-0.5], place_resp="TCL ERROR: Place design failed")
    res, logs = _run(fake, _cfg_run(replace_gamble_max_draws=4))
    # error streak of 2 stops the stage before draws 3-4
    assert len(res.draws) == 2
    assert all(d.verdict == "ERROR" for d in res.draws)
    assert any("two consecutive draw errors" in m for m in logs)


def test_runner_never_banks_on_write_tcl_error():
    # accept-worthy wns but the bank write itself errors -> NOT adopted
    # (phantom-accept lesson: never bank on a Tcl error).
    fake, _ = _fake_vivado([-0.4], write_resp="TCL ERROR: disk full")
    res, _ = _run(fake, _cfg_run(replace_gamble_max_draws=1))
    assert res.draws[0].verdict == "ERROR"
    assert "write_checkpoint failed" in res.draws[0].reason
    assert not res.adopted and res.best_path is None


def test_runner_bar_ratchets_after_adopt():
    # draw 1 adopts at -0.5 (chain-best -0.8); draw 2 measures -0.45 —
    # only +0.05 over the NEW bar -> rejected against the ratcheted bar.
    fake, _ = _fake_vivado([-0.5, -0.45])
    res, _ = _run(fake, _cfg_run(replace_gamble_max_draws=2))
    assert res.draws[0].verdict == "ADOPTED"
    assert res.draws[1].pre_wns == -0.5            # ratcheted bar
    assert res.draws[1].verdict == "REJECTED"
    assert res.best_wns == -0.5 and res.best_path == "/tmp/rg_d1.dcp"


def test_runner_second_adopt_uses_new_draw_file():
    # both draws clear the +0.15 bar -> the second banks to _d2 (a later
    # failed write can never clobber an earlier adopted file).
    fake, _ = _fake_vivado([-0.5, -0.3])
    res, _ = _run(fake, _cfg_run(replace_gamble_max_draws=2))
    assert [d.verdict for d in res.draws] == ["ADOPTED", "ADOPTED"]
    assert res.best_wns == -0.3 and res.best_path == "/tmp/rg_d2.dcp"


def test_runner_unroute_completion_pass():
    # first measure sees 3 unrouted nets -> one bare route_design
    # completion pass, then a clean re-measure adopts.
    calls_seen = []
    state = {"i": 0, "rs": 0}

    async def fake(tool, args):
        cmd = (args or {}).get("command", "")
        calls_seen.append(cmd)
        if tool == "vivado_restart_vivado":
            return "ok"
        if "write_checkpoint" in cmd:
            return "ok"
        if "-hold" in cmd:
            return "0.05"
        if "report_route_status" in cmd:
            state["rs"] += 1
            ur = 3 if state["rs"] == 1 else 0
            return (f"# of routable nets : 10\n"
                    f"# of fully routed nets : {10 - ur}\n"
                    f"# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-0.5"
        return "ok"

    logs = []
    res = asyncio.run(run_replace_gamble(
        fake, best_dcp_path="/tmp/banked_best.dcp", best_wns=-0.8,
        deadline_ts=_time.time() + 5000.0, wns_tcl="SLACK",
        cfg=_cfg_run(replace_gamble_max_draws=1), log=logs.append,
        out_dcp_base="/tmp/rg"))
    assert res.draws[0].verdict == "ADOPTED"
    # the bare completion pass ran (a route_design with no -directive)
    assert any(c.strip() == "route_design" for c in calls_seen)


def test_runner_mid_stage_unaffordable(monkeypatch):
    """Draw 2 becomes UNAFFORDABLE after draw 1 consumes the reserve —
    the per-draw re-check fails closed instead of starting a re-place
    that cannot finish."""
    import optimizer.replace_gamble as rg

    clock = {"t": 1_000_000.0}

    class _FakeTime:
        @staticmethod
        def time():
            return clock["t"]

    monkeypatch.setattr(rg, "time", _FakeTime)
    # need = 100*1.3 + 300 = 430; give 500s total; draw 1 burns 400s.
    deadline = clock["t"] + 500.0

    state = {"i": 0}

    async def fake(tool, args):
        cmd = (args or {}).get("command", "")
        if tool == "vivado_restart_vivado":
            return "ok"
        if "place_design -directive" in cmd:
            clock["t"] += 400.0            # draw 1 eats the window
            return "ok"
        if "-hold" in cmd:
            return "0.05"
        if "report_route_status" in cmd:
            return ("# of routable nets : 10\n# of fully routed nets : 10\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-0.75"                 # valid but below adopt bar
        return "ok"

    logs = []
    res = asyncio.run(run_replace_gamble(
        fake, best_dcp_path="/tmp/banked_best.dcp", best_wns=-0.8,
        deadline_ts=deadline, wns_tcl="SLACK",
        cfg=_cfg_run(replace_gamble_max_draws=2), log=logs.append,
        out_dcp_base="/tmp/rg"))
    assert res.draws[0].verdict == "REJECTED"
    assert res.draws[1].verdict == "UNAFFORDABLE"
    assert res.draws[1].completed_valid is False
    assert any("verdict=UNAFFORDABLE" in m for m in logs)


# MUX wiring: the best VERIFIED draw is surfaced (best_draw_*)
# REGARDLESS of the +0.15 adopt gate, persisted to a _cand_d file
# (banked best on disk still never touched).

def test_runner_surfaces_below_margin_verified_draw():
    # +0.05 draw: real, routed, hold-clean, but below the +0.15 adopt bar.
    # NOT adopted, yet SURFACED as best_draw_* for the finalize MUX and
    # written to a dedicated _cand_d file.
    fake, calls = _fake_vivado([-0.75])
    res, _ = _run(fake, _cfg_run(replace_gamble_max_draws=1))
    assert res.adopted is False
    assert res.best_wns == -0.8                        # adopt bar unchanged
    assert res.best_draw_path == "/tmp/rg_cand_d1.dcp"  # MUX candidate
    assert res.best_draw_wns == -0.75
    assert res.best_draw_whs == 0.05
    assert any("write_checkpoint" in cmd and "rg_cand_d1" in cmd
               for _t, cmd in calls)
    # the banked best on disk is STILL never a write target
    for _t, cmd in calls:
        if "write_checkpoint" in cmd:
            assert "banked_best" not in cmd


def test_runner_adopt_also_surfaces_best_draw():
    # An adopted draw is trivially the best verified draw — surfaced as the
    # MUX candidate too, reusing the _d1 adopt file (no extra write).
    fake, calls = _fake_vivado([-0.5])
    res, _ = _run(fake, _cfg_run(replace_gamble_max_draws=1))
    assert res.adopted is True
    assert res.best_draw_path == "/tmp/rg_d1.dcp"
    assert res.best_draw_wns == -0.5
    assert not any("rg_cand_d" in cmd for _t, cmd in calls)


def test_runner_hold_dirty_draw_not_surfaced():
    fake, _ = _fake_vivado([-0.4], hold="-0.05")
    res, _ = _run(fake, _cfg_run(replace_gamble_max_draws=1))
    assert res.best_draw_path is None      # hold-dirty != verified


def test_runner_unrouted_draw_not_surfaced():
    fake, _ = _fake_vivado([-0.4], unrouted=3)
    res, _ = _run(fake, _cfg_run(replace_gamble_max_draws=1))
    assert res.draws[0].completed_valid is False
    assert res.best_draw_path is None      # unrouted != verified


def test_runner_best_draw_prefers_higher_wns_verified_draw():
    # draw1 adopts -0.5; draw2 measures -0.45 (better WNS, but only +0.05
    # over the RATCHETED bar -> NOT re-adopted). The MUX candidate tracks
    # the genuinely-best verified draw: -0.45 via a _cand_d2 file.
    fake, _ = _fake_vivado([-0.5, -0.45])
    res, _ = _run(fake, _cfg_run(replace_gamble_max_draws=2))
    assert res.adopted is True and res.best_wns == -0.5   # adopt bar
    assert res.best_draw_path == "/tmp/rg_cand_d2.dcp"     # MUX candidate
    assert res.best_draw_wns == -0.45


def test_runner_cand_write_error_leaves_draw_unsurfaced():
    # A below-margin verified draw whose _cand write errors is NOT surfaced
    # (never point the MUX at a stale/partial file — phantom lesson).
    fake, _ = _fake_vivado([-0.75], write_resp="TCL ERROR: disk full")
    res, logs = _run(fake, _cfg_run(replace_gamble_max_draws=1))
    assert res.draws[0].completed_valid is True
    assert res.best_draw_path is None
    assert any("MUX-candidate write_checkpoint failed" in m for m in logs)


def test_result_summary_lines():
    r = ReplaceGambleResult(skip_reason="disabled (kill switch)")
    assert "SKIPPED" in r.summary()
    fake, _ = _fake_vivado([-0.5])
    res, logs = _run(fake, _cfg_run(replace_gamble_max_draws=1))
    assert any("REPLACE_GAMBLE summary attempts=1 completed_valid=1 "
               "adopted=1" in m for m in logs)


# DCPOptimizer stage wrapper (stub, _fanout_gate_probe idiom)

def _stage_probe(cfg, best_wns=-0.8, wns_seq=("-0.5",)):
    import dcp_optimizer as do
    calls = []
    seq = {"i": 0}

    _bw = best_wns

    class _Stub:
        run_dir = None
        _best_valid_dcp = None
        _best_valid_mirror_size = None
        _ils_observed_costs = {}

        def __init__(self):
            self._ils_polish_cfg = cfg
            self.best_wns = _bw
            self._best_valid_dcp_wns = _bw

        async def call_tool(self, name, args):
            cmd = (args or {}).get("command", "")
            calls.append((name, cmd))
            if name == "vivado_restart_vivado":
                return "ok"
            if "-hold" in cmd:
                return "0.05"
            if "report_route_status" in cmd:
                return ("# of routable nets : 10\n"
                        "# of fully routed nets : 10\n"
                        "# of nets with routing errors : 0\n")
            if "SLACK" in cmd:
                i = min(seq["i"], len(wns_seq) - 1)
                seq["i"] += 1
                return str(wns_seq[i])
            return "ok"

    stub = _Stub()
    asyncio.run(do.DCPOptimizer._replace_gamble_after_polish(
        stub, "/tmp/banked_best.dcp", best_wns,
        _time.time() + 5000.0, "SLACK"))
    return stub, calls


def test_stage_flag_off_is_noop():
    # Zero diff with the flag off: not a single Vivado call, no state change.
    stub, calls = _stage_probe(ILSPolishConfig())
    assert calls == []
    assert stub.best_wns == -0.8 and stub._best_valid_dcp is None


def test_stage_fails_closed_without_anchor():
    stub, calls = _stage_probe(ILSPolishConfig(replace_gamble_enabled=True))
    assert calls == []
    assert stub._best_valid_dcp is None


def test_stage_adopt_repoints_bank():
    from pathlib import Path
    cfg = ILSPolishConfig(replace_gamble_enabled=True,
                          expected_heavy_cycle_s=100.0,
                          replace_gamble_max_draws=1)
    stub, calls = _stage_probe(cfg, wns_seq=("-0.5",))
    assert len(calls) > 0
    assert stub.best_wns == -0.5
    assert stub._best_valid_dcp == Path("ils_replace_gamble_d1.dcp")
    assert stub._best_valid_dcp_wns == -0.5


def test_stage_reject_keeps_bank():
    cfg = ILSPolishConfig(replace_gamble_enabled=True,
                          expected_heavy_cycle_s=100.0,
                          replace_gamble_max_draws=1)
    stub, calls = _stage_probe(cfg, wns_seq=("-0.75",))   # +0.05 < +0.15
    assert len(calls) > 0
    assert stub.best_wns == -0.8 and stub._best_valid_dcp is None
