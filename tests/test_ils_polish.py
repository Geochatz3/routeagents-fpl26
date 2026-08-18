"""Unit tests for the ILS-polish trigger gate + result summary (no Vivado)."""
import asyncio
import pytest
from optimizer.ils_polish import (
    should_trigger, ILSPolishConfig, ILSPolishResult, run_ils_polish, ILS_COMBOS,
    fanout_polish_accept,
)

CFG = ILSPolishConfig(enabled=True, max_cells=300_000, min_remaining_s=1100.0,
                      stagnation_seconds=600.0)


@pytest.fixture(autouse=True)
def _default_rotation(monkeypatch):
    """Pin every test in this module to the DEFAULT rotation.

    These tests assert properties of the shipped combo rotation (indices,
    exhaustion, pristine_rot == cycles). The jul27 place-retry extension
    (FPL26_ILS_PLACE_RETRY) deliberately breaks the cycles==rotation-advance
    invariant with one out-of-order forced pick, so a test of DEFAULT behaviour
    must not depend on whether that variable happens to be set in the ambient
    environment. The extension's own tests arm it explicitly.
    """
    monkeypatch.delenv("FPL26_ILS_PLACE_RETRY", raising=False)
    # jul28: same reason, for every later extension. A test of DEFAULT behaviour
    # must not depend on which flags happen to be exported in the shell that runs
    # it — the armed suite is run deliberately, arm by arm, by the release check.
    # jul29: four of these flipped to DEFAULT ON, because the eval path never armed
    # them and every gain we had measured was therefore not in the submission (see
    # optimizer/ils_polish.py). This fixture's INTENT has always been "start disarmed";
    # that used to be implied by the default and must now be stated. Setting "0" keeps
    # every pre-jul29 test in this file meaning exactly what it meant when written.
    monkeypatch.setenv("FPL26_ILS_PLACE_RETRY_LADDER", "0")
    monkeypatch.setenv("FPL26_ILS_INCR_ROUTE", "0")
    monkeypatch.setenv("FPL26_ILS_MEASURED_BASIS", "0")
    monkeypatch.setenv("FPL26_ILS_INCR_ROUTE_FIRST", "0")
    monkeypatch.delenv("FPL26_ILS_RETRY_BASELINE_GATE", raising=False)
    monkeypatch.delenv("FPL26_ILS_LADDER_RESERVE", raising=False)
    monkeypatch.delenv("FPL26_ILS_INCR_ROUTE_TERMINAL", raising=False)
    monkeypatch.delenv("FPL26_ILS_LADDER_STOP_ON_ACCEPT", raising=False)
    monkeypatch.delenv("FPL26_ILS_LADDER_ORDER", raising=False)




def _trig(**kw):
    base = dict(cells=10_000, remaining_s=1500.0, seconds_since_improve=900.0,
                best_wns=-0.9, baseline_wns=-1.0, cfg=CFG)
    base.update(kw)
    return should_trigger(**base)


def test_disabled():
    t, why = _trig(cfg=ILSPolishConfig(enabled=False))
    assert t is False and why == "disabled"


def test_too_large():
    t, why = _trig(cells=400_000)
    assert t is False and "too_large" in why


def test_insufficient_budget():
    t, why = _trig(remaining_s=500.0)
    assert t is False and "insufficient_budget" in why


def test_loop_still_productive():
    t, why = _trig(seconds_since_improve=120.0)
    assert t is False and "loop_still_productive" in why


def test_timing_already_met():
    t, why = _trig(best_wns=0.05)
    assert t is False and "timing_already_met" in why


def test_fires_when_stalled_small_budget_negative_wns():
    t, why = _trig()
    assert t is True and "stalled" in why


def test_unknown_size_fails_open():
    # cells=None -> size gate skipped (fail-open; keep-best protects correctness)
    t, _ = _trig(cells=None)
    assert t is True


def test_large_design_at_boundary():
    assert _trig(cells=300_000)[0] is True       # == threshold passes
    assert _trig(cells=300_001)[0] is False      # just over fails


def test_result_summary_skip_vs_improved():
    r = ILSPolishResult(triggered=False, skip_reason="disabled")
    assert "SKIPPED" in r.summary()
    r2 = ILSPolishResult(triggered=True, improved=True, baseline_wns=-1.0,
                         best_wns=-0.8, cycles=4, accepted=2)
    assert "IMPROVED" in r2.summary() and "0.2" in r2.summary()


def test_combos_winners_first():
    # Explore (most historical accepts) leads; LASTMILE (2/2 validated
    # accepts on plateaued states, jun12) second so the K=2 futility window
    # covers both operator classes; ROUTE_REROLL (jul21 plateau probe, fir
    # +0.070ns/+9.5MHz hold-improving) third — near-met designs reach it
    # early, deep-WNS designs gate-skip it INLINE so their effective order
    # is unchanged; ROUTE_ONLY (jul02: broadest breadth result, 8/12)
    # fourth; ExtraTimingOpt next.
    from optimizer.ils_polish import LASTMILE_PD, ROUTE_ONLY_PD, ROUTE_REROLL_PD
    assert ILS_COMBOS[0][0] == "Explore"
    assert ILS_COMBOS[1][0] == LASTMILE_PD
    assert ILS_COMBOS[2][0] == ROUTE_REROLL_PD
    assert ILS_COMBOS[3][0] == ROUTE_ONLY_PD
    assert ILS_COMBOS[4][0] == "ExtraTimingOpt"


# For the run-loop tests the fake call_tool returns instantly, so disable the
# min-cycle-duration no-op guard (set min_cycle_seconds=0); cap cycles small.
CFG_RUN = ILSPolishConfig(enabled=True, max_cells=300_000, min_remaining_s=1100.0,
                          stagnation_seconds=600.0, min_cycle_seconds=0.0,
                          max_cycles=5)


def test_run_ils_polish_keep_best_never_worse():
    """Every re-place yields a WORSE wns -> never accepted, best stays baseline."""
    async def fake_call_tool(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "0.05"   # hold clean (gate added 2026-06-11)
        if "report_route_status" in cmd:
            return ("# of routable nets...... : 100\n"
                    "# of fully routed nets.. : 100\n"
                    "# of nets with routing errors.. : 0\n")
        if "get_property SLACK" in cmd or "get_timing_paths" in cmd:
            return "-2.0"   # always worse than baseline -0.5
        return "ok"
    import time as _t
    res = asyncio.run(run_ils_polish(
        fake_call_tool, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 5.0, wns_tcl="get_property SLACK x",
        cfg=CFG_RUN, log=lambda m: None))
    assert res.triggered is True
    assert res.improved is False            # never accepted a worse result
    assert res.best_wns == -0.5             # baseline preserved (never-worse)
    assert res.accepted == 0


def test_run_ils_polish_accepts_improvement():
    """Re-place yields a BETTER, fully-routed wns -> accepted."""
    async def fake_call_tool(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "0.05"   # hold clean (gate added 2026-06-11)
        if "report_route_status" in cmd:
            return ("# of routable nets : 50\n# of fully routed nets : 50\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd or "get_timing_paths" in cmd:
            return "-0.2"   # better than baseline -0.5
        return "ok"
    import time as _t
    res = asyncio.run(run_ils_polish(
        fake_call_tool, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 5.0, wns_tcl="SLACK", cfg=CFG_RUN, log=lambda m: None))
    assert res.improved is True
    assert res.best_wns == -0.2
    assert res.accepted >= 1


def test_run_ils_polish_bails_on_budget_skip_no_runaway():
    """RUNAWAY GUARD: if commands are skipped (instant return, wns unparseable),
    ILS must bail after 2 no-op cycles, NOT spin to the deadline."""
    async def fake_skip(tool, args):
        # report_route_status returns garbage (no routable/fully lines) -> ur=-1;
        # SLACK returns non-numeric -> wns=None. Both instant.
        return "Budget skip: not started"
    import time as _t
    res = asyncio.run(run_ils_polish(
        fake_skip, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 30.0,   # generous deadline; guard must stop early
        wns_tcl="SLACK", cfg=ILSPolishConfig(min_cycle_seconds=20.0, max_cycles=60),
        log=lambda m: None))
    assert res.cycles <= 2                  # bailed, did NOT run to 60 / deadline
    assert res.improved is False
    assert res.best_wns == -0.5             # never-worse preserved


def test_run_ils_polish_no_improve_early_stop():
    """no_improve_stop: a never-accepting seed bails after K real cycles (yields
    budget to the dual-seed corrective probe) instead of running to max_cycles."""
    async def fake_worse(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "0.05"   # hold clean (gate added 2026-06-11)
        if "report_route_status" in cmd:
            return ("# of routable nets : 10\n# of fully routed nets : 10\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd or "get_timing_paths" in cmd:
            return "-2.0"   # always worse than baseline -0.5 -> never accepts
        return "ok"
    import time as _t
    res = asyncio.run(run_ils_polish(
        fake_worse, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 30.0, wns_tcl="SLACK", cfg=CFG_RUN,
        log=lambda m: None, no_improve_stop=2))
    assert res.cycles == 2                  # stopped at K=2, not max_cycles=5
    assert res.improved is False
    assert res.best_wns == -0.5             # never-worse preserved
    assert any("no improvement" in n for n in res.notes)


def test_run_ils_polish_no_improve_stop_resets_on_accept():
    """An accept resets the no-improve streak, so a seed that wins early keeps
    going past K consecutive-non-accept stop (v2-style: accept then plateau)."""
    async def fake_one_accept(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "0.05"   # hold clean (gate added 2026-06-11)
        if "report_route_status" in cmd:
            return ("# of routable nets : 10\n# of fully routed nets : 10\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd or "get_timing_paths" in cmd:
            return "-0.2"   # better than -0.5 once; then ties (no further accept)
        return "ok"
    import time as _t
    res = asyncio.run(run_ils_polish(
        fake_one_accept, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 30.0, wns_tcl="SLACK", cfg=CFG_RUN,
        log=lambda m: None, no_improve_stop=2))
    # cyc1 accept (streak->0), cyc2 non (1), cyc3 non (2) -> stop at 3, not 2.
    assert res.cycles == 3
    assert res.accepted == 1 and res.improved is True
    assert res.best_wns == -0.2


def test_dual_seed_config_defaults():
    c = ILSPolishConfig()
    assert c.dual_seed is True
    assert c.no_improve_stop_cycles == 2
    assert c.heavy_cmd_timeout_s == 1800.0
    assert c.measure_cmd_timeout_s == 600.0


def test_final_seed_futility_stop_default():
    """jun12 wall-penalty fix: the FINAL/sole seed defaults to a K=2 futility
    stop (score = alpha*(1-0.1*gamma): non-accepting tail cycles cost up to
    10% of alpha; replay over 10 AWS runs = +9.65 score, zero forfeited
    accepts). 0 must remain expressible as the legacy kill switch."""
    c = ILSPolishConfig()
    assert c.final_seed_no_improve_stop == 2
    c2 = ILSPolishConfig(final_seed_no_improve_stop=0)
    assert c2.final_seed_no_improve_stop == 0


def test_run_ils_polish_heavy_cmds_carry_large_timeout():
    """Regression (local spam 2026-06-08): place/route/phys_opt must pass a
    timeout >300s, else the MCP server's 300s default kills each step on a large
    design and the cycle returns wns=None. Capture the timeouts actually sent."""
    seen = {}
    async def fake(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "0.05"   # hold clean (gate added 2026-06-11)
        if cmd.startswith("place_design -directive"):
            seen["place"] = args.get("timeout")
        elif cmd.startswith("route_design"):
            seen["route"] = args.get("timeout")
        elif cmd.startswith("phys_opt_design"):
            seen["phys_opt"] = args.get("timeout")
        if "report_route_status" in cmd:
            return ("# of routable nets : 10\n# of fully routed nets : 10\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd or "get_timing_paths" in cmd:
            return "-0.6"   # within physopt_skip margin of best -0.5 -> phys_opt runs
        return "ok"
    import time as _t
    asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 3000.0, wns_tcl="SLACK", cfg=CFG_RUN,
        log=lambda m: None, no_improve_stop=1))
    assert seen.get("place", 0) > 300.0
    assert seen.get("route", 0) > 300.0
    assert seen.get("phys_opt", 0) > 300.0


def test_should_trigger_at_exit():
    from optimizer.ils_polish import should_trigger_at_exit as ex
    c = ILSPolishConfig(enabled=True, max_cells=300_000, exit_min_remaining_s=600.0)
    # stuck + budget + small -> fire
    assert ex(cells=4000, remaining_s=1300, best_wns=-0.9, cfg=c)[0] is True
    # timing MET + surplus budget -> fire in met-surplus mode (jul04:
    # positive slack still buys alpha; fmax = 1/(T - wns))
    t_met, why_met = ex(cells=4000, remaining_s=1300, best_wns=0.1, cfg=c)
    assert t_met is True and "met-surplus" in why_met
    # ... unless the kill switch is off -> old leave-it-alone policy
    c_off = ILSPolishConfig(enabled=True, max_cells=300_000,
                            exit_min_remaining_s=600.0, met_surplus_ils=False)
    assert ex(cells=4000, remaining_s=1300, best_wns=0.1, cfg=c_off)[0] is False
    # met but too large / no budget -> same gates apply
    assert ex(cells=400_000, remaining_s=1300, best_wns=0.1, cfg=c)[0] is False
    assert ex(cells=4000, remaining_s=300, best_wns=0.1, cfg=c)[0] is False
    # unknown wns -> never arm
    assert ex(cells=4000, remaining_s=1300, best_wns=None, cfg=c)[0] is False
    # too large -> skip
    assert ex(cells=400_000, remaining_s=1300, best_wns=-0.9, cfg=c)[0] is False
    # no budget left -> skip
    assert ex(cells=4000, remaining_s=300, best_wns=-0.9, cfg=c)[0] is False
    # disabled -> skip
    assert ex(cells=4000, remaining_s=1300, best_wns=-0.9,
              cfg=ILSPolishConfig(enabled=False))[0] is False


# --- seed gate (full-13 2026-06-08) ---
from optimizer.ils_polish import choose_ils_seed

def _cfg_seed(seed_from_raw=True, stuck=0.05):
    return ILSPolishConfig(enabled=True, seed_from_raw=seed_from_raw,
                           stuck_recipe_gain_ns=stuck)

def test_seed_gate_stuck_design_uses_raw():
    # v2: recipe gained ~0 over initial -> raw
    kind, gain = choose_ils_seed(initial_wns=-0.946, recipe_wns=-0.946, cfg=_cfg_seed())
    assert kind == "raw" and abs(gain) < 1e-9

def test_seed_gate_recipe_helped_uses_recipe_best():
    # spam-filter: recipe gained 0.214 -> recipe_best
    kind, gain = choose_ils_seed(initial_wns=-0.9, recipe_wns=-0.686, cfg=_cfg_seed())
    assert kind == "recipe_best" and round(gain, 3) == 0.214

def test_seed_gate_threshold_boundary():
    # just below floor -> raw; just above -> recipe_best
    assert choose_ils_seed(initial_wns=-1.0, recipe_wns=-0.96, cfg=_cfg_seed())[0] == "raw"     # gain 0.04
    assert choose_ils_seed(initial_wns=-1.0, recipe_wns=-0.94, cfg=_cfg_seed())[0] == "recipe_best"  # gain 0.06

def test_seed_gate_default_covers_llm_dribble():
    # jul03 preview attempt-4: with the prompt guard keeping the LLM alive, v2's
    # LLM dribbled +0.061 over 30 min, crossing the old 0.05 floor and skipping
    # the raw seed (-0.84 capture lost; alpha 17.48 -> 9.88). The DEFAULT floor
    # must classify a marginal dribble as STUCK.
    c = ILSPolishConfig()
    assert c.stuck_recipe_gain_ns == 0.15
    kind, gain = choose_ils_seed(initial_wns=-0.946, recipe_wns=-0.885, cfg=c)
    assert kind == "raw" and round(gain, 3) == 0.061


def test_seed_gate_unknown_gain_conservative_recipe_best():
    assert choose_ils_seed(initial_wns=None, recipe_wns=-0.8, cfg=_cfg_seed())[0] == "recipe_best"
    assert choose_ils_seed(initial_wns=-0.9, recipe_wns=float("-inf"), cfg=_cfg_seed())[0] == "recipe_best"

def test_seed_gate_disabled_uses_recipe_best():
    assert choose_ils_seed(initial_wns=-0.946, recipe_wns=-0.946, cfg=_cfg_seed(seed_from_raw=False))[0] == "recipe_best"


def test_run_ils_polish_step_error_envelope_aborts_cycle():
    """A heavy step returning call_tool's error ENVELOPE (timeout/skip — call_tool
    never raises) must abort the cycle, not run the next step on stale state."""
    import time as _t
    calls = []
    async def fake(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "0.05"   # hold clean (gate added 2026-06-11)
        calls.append(cmd.split()[0])
        if cmd.startswith("route_design"):
            return '{"error": "tool_timed_out_budget", "elapsed_seconds": 300}'
        if cmd.startswith("open_checkpoint") and calls.count("open_checkpoint") > 1:
            return '{"error": "tool_timed_out_budget"}'  # recovery open fails too
        return "ok"
    res = asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 5.0, wns_tcl="get_property SLACK x",
        cfg=CFG_RUN, log=lambda m: None))
    assert res.improved is False and res.accepted == 0
    assert res.cycles == 1                    # ended on the first failed cycle
    assert "phys_opt_design" not in calls     # aborted before the later steps
    assert any("route_design failed" in n for n in res.notes)
    assert res.best_wns == -0.5               # baseline preserved (never-worse)


def test_run_ils_polish_write_checkpoint_failure_discards_improvement():
    """write_checkpoint timing out (error envelope) must NOT be accepted: the
    file on disk is stale/partial, so recording the better wns would ship a DCP
    that doesn't have it (never-worse violation caught 2026-06-10)."""
    import time as _t
    async def fake(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "0.05"   # hold clean (gate added 2026-06-11)
        if "report_route_status" in cmd:
            return ("# of routable nets...... : 100\n"
                    "# of fully routed nets.. : 100\n"
                    "# of nets with routing errors.. : 0\n")
        if "get_property SLACK" in cmd:
            return "-0.1"        # better than baseline -0.5 -> would accept
        if cmd.startswith("write_checkpoint"):
            return '{"error": "tool_timed_out_budget"}'
        return "ok"
    res = asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 5.0, wns_tcl="get_property SLACK x",
        cfg=CFG_RUN, no_improve_stop=1, log=lambda m: None))
    assert res.improved is False
    assert res.accepted == 0
    assert res.best_wns == -0.5               # NOT updated to the unwritten -0.1
    assert any("discarded" in n for n in res.notes)


def test_ws1b_physopt_skipped_when_unrouted():
    """Post-route unrouted nets can never be accepted -> phys_opt must be skipped."""
    import time as _t
    calls = []
    async def fake(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "0.05"   # hold clean (gate added 2026-06-11)
        calls.append(cmd.split()[0])
        if "report_route_status" in cmd:
            return ("# of routable nets : 100\n# of fully routed nets : 90\n"
                    "# of nets with routing errors : 0\n")   # ur=10
        if "SLACK" in cmd:
            return "-0.4"
        return "ok"
    res = asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 5.0, wns_tcl="SLACK", cfg=CFG_RUN,
        log=lambda m: None, no_improve_stop=1))
    assert "phys_opt_design" not in calls
    assert res.physopt_skipped >= 1
    assert res.accepted == 0 and res.best_wns == -0.5


def test_ws1b_physopt_skipped_when_hopeless_margin():
    """Post-route wns far below best (-2.0 vs -0.5, margin 0.5) -> skip phys_opt."""
    import time as _t
    calls = []
    async def fake(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "0.05"   # hold clean (gate added 2026-06-11)
        calls.append(cmd.split()[0])
        if "report_route_status" in cmd:
            return ("# of routable nets : 100\n# of fully routed nets : 100\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-2.0"
        return "ok"
    res = asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 5.0, wns_tcl="SLACK", cfg=CFG_RUN,
        log=lambda m: None, no_improve_stop=1))
    assert "phys_opt_design" not in calls
    assert res.physopt_skipped >= 1


def test_ws2_combo_offset_continues_rotation():
    """The corrective probe must NOT replay the primary seed's combos: with
    combo_offset=2 the first cycle must use ILS_COMBOS[2], not [0]."""
    import time as _t
    placed = []
    async def fake(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "0.05"   # hold clean (gate added 2026-06-11)
        if cmd.startswith("place_design -directive"):
            placed.append(cmd.split()[-1])
        if "report_route_status" in cmd:
            return ("# of routable nets : 100\n# of fully routed nets : 100\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-0.6"
        return "ok"
    asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 5.0, wns_tcl="SLACK", cfg=CFG_RUN,
        log=lambda m: None, no_improve_stop=2, combo_offset=4))
    # offset=4 (first place-directive combo past the ROUTE_REROLL/ROUTE_ONLY
    # sentinels since the jul21 insertion): rotation must continue from
    # there, not restart at 0.
    assert placed[0] == ILS_COMBOS[4][0]
    # jul05 reorder (+jul21 shift): idx5 is now the PARTIAL_RUIN sentinel
    # (noops under this fake and advances). The invariant under test is
    # offset CONTINUATION — the rotation must not restart at combo 0.
    assert ILS_COMBOS[0][0] not in placed[:2]


def test_ws1a_rotation_exhausted_stops_no_repeat():
    """Deterministic placer: once every combo was tried on the current best with
    no accept, ILS must stop rather than replay identical experiments."""
    import time as _t
    placed = []
    async def fake(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "0.05"   # hold clean (gate added 2026-06-11)
        if cmd.startswith("place_design -directive"):
            placed.append(cmd.split()[-1])
        if "report_route_status" in cmd:
            return ("# of routable nets : 100\n# of fully routed nets : 100\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-0.6"   # never better than baseline (-0.2, near closure so
            #                 the LASTMILE combo is NOT gated -> full rotation)
        return "ok"
    cfg = ILSPolishConfig(enabled=True, min_cycle_seconds=0.0, max_cycles=60)
    res = asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.2,
        deadline_ts=_t.time() + 30.0, wns_tcl="SLACK", cfg=cfg,
        log=lambda m: None))
    # PARTIAL_RUIN places WITHOUT a directive and ROUTE_ONLY/ROUTE_REROLL
    # never place (baseline -0.2 is near-met, so the jul21 re-roll gate lets
    # ROUTE_REROLL run — it consumes a cycle but adds no directive place),
    # so directive-style places = all combos minus three; rotation must
    # still exhaust exactly once.
    assert len(placed) == len(ILS_COMBOS) - 3      # each directive combo once
    assert len(set(placed)) == len(ILS_COMBOS) - 3 # no repeats
    assert any("rotation exhausted" in n for n in res.notes)


def test_ws1b_measure_failure_does_not_block_physopt():
    """If the post-route measure can't produce a WNS (None), the skip logic must
    NOT fire on garbage — phys_opt still runs (pre-WS1b behavior preserved)."""
    import time as _t
    calls = []
    state = {"n": 0}
    async def fake(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "0.05"   # hold clean (gate added 2026-06-11)
        calls.append(cmd.split()[0])
        if "report_route_status" in cmd:
            return ("# of routable nets : 100\n# of fully routed nets : 100\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            state["n"] += 1
            return "garbage" if state["n"] == 1 else "-0.6"  # 1st measure fails
        return "ok"
    res = asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 5.0, wns_tcl="SLACK", cfg=CFG_RUN,
        log=lambda m: None, no_improve_stop=1))
    assert "phys_opt_design" in calls
    assert res.physopt_skipped == 0


def test_hold_dirty_accept_rejected():
    """Setup improves + fully routed BUT hold went negative -> the accept must
    be rejected (validator gates hold_passed; a hold-dirty ship zeroes the
    benchmark)."""
    import time as _t
    writes = []
    async def fake(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "-0.12"        # hold DIRTY
        if "report_route_status" in cmd:
            return ("# of routable nets : 100\n# of fully routed nets : 100\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-0.1"         # setup improved vs baseline -0.5
        if cmd.startswith("write_checkpoint"):
            writes.append(cmd)
        return "ok"
    res = asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 5.0, wns_tcl="SLACK", cfg=CFG_RUN,
        log=lambda m: None, no_improve_stop=1))
    assert res.accepted == 0 and res.improved is False
    assert res.best_wns == -0.5
    assert writes == []                       # nothing persisted
    assert any("hold-dirty" in n for n in res.notes)


def test_hold_unmeasurable_fails_open():
    """If the hold query can't produce a number, the gate must FAIL OPEN
    (accept proceeds — preserves pre-gate behavior on odd designs)."""
    import time as _t
    async def fake(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "no timing paths matched"   # unparseable
        if "report_route_status" in cmd:
            return ("# of routable nets : 100\n# of fully routed nets : 100\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-0.1"
        return "ok"
    res = asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 5.0, wns_tcl="SLACK", cfg=CFG_RUN,
        log=lambda m: None, no_improve_stop=1))
    assert res.accepted == 1 and res.best_wns == -0.1


def test_partial_ruin_combo_dispatch():
    """The __PARTIAL_RUIN__ sentinel combo must run targeted unplace_cell +
    incremental place_design — never full `place_design -unplace`.

    Needs an explicit HIGH spread: CFG_RUN leaves
    critical_path_avg_spread_tiles at its None default, and since aug05 that
    fails closed for partial-ruin, so the sentinel would never dispatch and
    this test would assert against an empty command list. The subject here is
    the DISPATCH SHAPE, not the gate — the gate has its own tests above.
    """
    import dataclasses
    import time as _t
    from optimizer.ils_polish import PARTIAL_RUIN_PD, ILS_COMBOS as _C
    offset = next(i for i, c in enumerate(_C) if c[0] == PARTIAL_RUIN_PD)
    cfg = dataclasses.replace(CFG_RUN, critical_path_avg_spread_tiles=120.0)
    cmds = []
    async def fake(tool, args):
        cmd = args.get("command", "")
        cmds.append(cmd)
        if "-hold" in cmd:
            return "0.05"
        if "report_route_status" in cmd:
            return ("# of routable nets : 100\n# of fully routed nets : 100\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-0.6"
        return "ok"
    asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 5.0, wns_tcl="SLACK", cfg=cfg,
        log=lambda m: None, no_improve_stop=1, combo_offset=offset))
    joined = "\n".join(cmds)
    assert "unplace_cell" in joined                 # targeted ruin ran
    assert "place_design -unplace" not in joined    # full ruin did NOT
    assert any(c.strip() == "place_design" for c in cmds)  # incremental place
    assert "get_timing_paths" in joined             # extraction ran


def test_route_only_combo_dispatch():
    """The __ROUTE_ONLY__ sentinel combo must unroute + re-route the EXISTING
    placement — never touch place_design in any form (jul02 DRILL H lever)."""
    import time as _t
    from optimizer.ils_polish import ROUTE_ONLY_PD, ILS_COMBOS as _C
    offset = next(i for i, c in enumerate(_C) if c[0] == ROUTE_ONLY_PD)
    cmds = []
    async def fake(tool, args):
        cmd = args.get("command", "")
        cmds.append(cmd)
        if "-hold" in cmd:
            return "0.05"
        if "report_route_status" in cmd:
            return ("# of routable nets : 100\n# of fully routed nets : 100\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-0.6"
        return "ok"
    asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 5.0, wns_tcl="SLACK", cfg=CFG_RUN,
        log=lambda m: None, no_improve_stop=1, combo_offset=offset))
    joined = "\n".join(cmds)
    assert "route_design -unroute" in joined
    assert "route_design -directive AggressiveExplore" in joined
    assert "place_design" not in joined      # placement must stay untouched


def test_route_only_accepts_exactly_zero_hold():
    """whs=0.0 must be ACCEPTED by the combo hold gate (floor -0.001): the
    official scorecard gate passes hold at exactly zero (jul02 preview,
    hold_passed=true at whs_ns=0.0) — a stricter floor here would forfeit
    ispd16-class route-only wins the validator would have scored."""
    import time as _t
    from optimizer.ils_polish import ROUTE_ONLY_PD, ILS_COMBOS as _C
    offset = next(i for i, c in enumerate(_C) if c[0] == ROUTE_ONLY_PD)
    writes = []
    async def fake(tool, args):
        cmd = args.get("command", "")
        if "write_checkpoint" in cmd:
            writes.append(cmd)
        if "-hold" in cmd:
            return "0.0"    # exactly-zero hold, the DRILL H ispd16 signature
        if "report_route_status" in cmd:
            return ("# of routable nets : 100\n# of fully routed nets : 100\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-0.2"   # better than baseline -0.5
        return "ok"
    res = asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 5.0, wns_tcl="SLACK", cfg=CFG_RUN,
        log=lambda m: None, no_improve_stop=1, combo_offset=offset))
    assert res.improved is True and res.best_wns == -0.2
    assert writes, "accept at whs=0.0 must persist the improved DCP"


def test_partial_ruin_combo_present_before_tail():
    """Rotation order: partial ruin sits above the never-accepting tail combos."""
    from optimizer.ils_polish import PARTIAL_RUIN_PD, ILS_COMBOS as _C
    names = [c[0] for c in _C]
    assert PARTIAL_RUIN_PD in names
    # jul05 promotion (climbON_a forensics): the cheap probe-validated
    # finisher sits IMMEDIATELY after ExtraTimingOpt so the proven
    # ExtraTimingOpt->PARTIAL_RUIN chain executes on any budget, ahead of
    # the expensive low-hit AltSpread/ExtraNetDelay pair.
    assert names.index(PARTIAL_RUIN_PD) == names.index("ExtraTimingOpt") + 1
    assert names.index(PARTIAL_RUIN_PD) < names.index("AltSpreadLogic_high")


def test_sa_explore_continues_from_near_miss():
    """explore_from_current: a routed near-miss becomes the next cycle's
    starting state; the best DCP is never overwritten by exploration."""
    import time as _t
    opens, writes = [], []
    async def fake(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "0.05"
        if cmd.startswith("open_checkpoint"):
            opens.append(cmd)
        if cmd.startswith("write_checkpoint"):
            writes.append(cmd)
        if "report_route_status" in cmd:
            return ("# of routable nets : 100\n# of fully routed nets : 100\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-0.55"   # near-miss: worse than best -0.5 but within 0.15
        return "ok"
    cfg = ILSPolishConfig(enabled=True, min_cycle_seconds=0.0, max_cycles=3,
                          explore_from_current=True)
    res = asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 5.0, wns_tcl="SLACK", cfg=cfg,
        log=lambda m: None))
    assert res.accepted == 0 and res.best_wns == -0.5      # floor untouched
    assert any(".explore.dcp" in w for w in writes)        # explore state saved
    assert not any("write_checkpoint -force {/tmp/x.dcp}" in w for w in writes)
    assert any(".explore.dcp" in o for o in opens[1:])     # later cycles start there


def test_sa_explore_off_by_default_keeps_greedy():
    """Default config: every cycle re-opens the BEST DCP (greedy), no explore
    writes."""
    import time as _t
    opens, writes = [], []
    async def fake(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "0.05"
        if cmd.startswith("open_checkpoint"):
            opens.append(cmd)
        if cmd.startswith("write_checkpoint"):
            writes.append(cmd)
        if "report_route_status" in cmd:
            return ("# of routable nets : 100\n# of fully routed nets : 100\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-0.55"
        return "ok"
    res = asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 5.0, wns_tcl="SLACK", cfg=CFG_RUN,
        log=lambda m: None, no_improve_stop=2))
    assert writes == []                                    # nothing persisted
    assert all("/tmp/x.dcp" in o and ".explore" not in o for o in opens)


def test_lastmile_combo_in_rotation():
    """jun12: LASTMILE must sit INSIDE the K=2 futility window (index 1) —
    at index >=2 it would be unreachable on a run whose first two cycles
    don't accept, and it has the strongest accept evidence of any combo on
    plateaued states (v2 +0.152ns, logicnets +0.059ns, both validated)."""
    from optimizer.ils_polish import LASTMILE_PD
    pds = [c[0] for c in ILS_COMBOS]
    assert pds.index(LASTMILE_PD) == 1
    assert pds[0] == "Explore"


def test_lastmile_cycle_dispatches_recipe_not_ruin():
    """The LASTMILE combo must run the UG906 sequence (clock_opt/retime/lut
    phys_opt + place -directive LastMile) and must NOT unplace the design."""
    from optimizer.ils_polish import LASTMILE_PD
    cmds = []
    async def fake_call_tool(tool, args):
        cmd = args.get("command", "")
        cmds.append(cmd)
        if "-hold" in cmd:
            return "0.05"
        if "report_route_status" in cmd:
            return ("# of routable nets : 50\n# of fully routed nets : 50\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-2.0"   # never accept; we only inspect dispatch
        return "ok"
    import time as _t
    # combo_offset puts the rotation directly on the LASTMILE combo.
    lm_idx = [c[0] for c in ILS_COMBOS].index(LASTMILE_PD)
    cfg = ILSPolishConfig(enabled=True, max_cells=300_000,
                          min_remaining_s=1100.0, stagnation_seconds=600.0,
                          min_cycle_seconds=0.0, max_cycles=1)
    asyncio.run(run_ils_polish(
        fake_call_tool, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.2,
        deadline_ts=_t.time() + 5.0, wns_tcl="SLACK", cfg=cfg,
        log=lambda m: None, combo_offset=lm_idx))
    joined = "\n".join(cmds)
    assert "phys_opt_design -clock_opt -retime -lut_opt" in joined
    assert "place_design -directive LastMile" in joined
    assert "place_design -unplace" not in joined
    assert "route_design -directive Explore" in joined


def test_cold_start_anchor_skips_unaffordable_full_ruin():
    """jun12 corundum drill: with no observed combo costs the picker chose
    combo 0 (full ruin) even when it could not complete in the window. With
    an expected_heavy_cycle_s anchor, an unaffordable Explore cycle must fall
    through to an affordable combo — since jul02 the cheapest affordable at
    the earliest rotation index is ROUTE_ONLY (0.6x prior, no place step)."""
    first_heavy = []
    async def fake_call_tool(tool, args):
        cmd = args.get("command", "")
        if ("place_design" in cmd or "unplace_cell" in cmd
                or "route_design -unroute" in cmd) and len(first_heavy) == 0:
            first_heavy.append(cmd)
        if "-hold" in cmd:
            return "0.05"
        if "report_route_status" in cmd:
            return ("# of routable nets : 50\n# of fully routed nets : 50\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-2.0"
        return "ok"
    import time as _t
    # Window 1500s; anchor 1800s -> Explore est 1800 (unaffordable),
    # LASTMILE 4500, ExtraTimingOpt 2520, AltSpread 3600, ExtraNetDelay
    # 5400 (all out); affordable: ROUTE_REROLL 0.55*1800=990 (index 2,
    # eligible at near-met baseline -0.5), ROUTE_ONLY 0.6*1800=1080
    # (index 3) and partial-ruin 0.7*1800=1260 -> first cycle MUST be an
    # unroute + re-route (ROUTE_REROLL, the earliest affordable).
    cfg = ILSPolishConfig(enabled=True, max_cells=300_000,
                          min_remaining_s=1100.0, stagnation_seconds=600.0,
                          min_cycle_seconds=0.0, max_cycles=1,
                          expected_heavy_cycle_s=1800.0)
    res = asyncio.run(run_ils_polish(
        fake_call_tool, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 1500.0, wns_tcl="SLACK", cfg=cfg,
        log=lambda m: None))
    assert res.cycles == 1
    assert first_heavy and "route_design -unroute" in first_heavy[0]


def test_cold_start_no_anchor_keeps_legacy_order():
    """Without an anchor (expected_heavy_cycle_s=0) the cold-start stays
    optimistic: combo 0 (Explore full ruin) is picked first (legacy)."""
    first_place = []
    async def fake_call_tool(tool, args):
        cmd = args.get("command", "")
        if "place_design -directive" in cmd and not first_place:
            first_place.append(cmd)
        if "-hold" in cmd:
            return "0.05"
        if "report_route_status" in cmd:
            return ("# of routable nets : 50\n# of fully routed nets : 50\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-2.0"
        return "ok"
    import time as _t
    cfg = ILSPolishConfig(enabled=True, max_cells=300_000,
                          min_remaining_s=1100.0, stagnation_seconds=600.0,
                          min_cycle_seconds=0.0, max_cycles=1)
    asyncio.run(run_ils_polish(
        fake_call_tool, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 1500.0, wns_tcl="SLACK", cfg=cfg,
        log=lambda m: None))
    assert first_place and "-directive Explore" in first_place[0]


def test_cold_start_anchor_affordable_keeps_explore_first():
    """With an anchor that FITS the window, Explore (cheapest full ruin and
    rotation head) must still be picked first — the anchor only reorders when
    full ruin is unaffordable."""
    first_place = []
    async def fake_call_tool(tool, args):
        cmd = args.get("command", "")
        if "place_design -directive" in cmd and not first_place:
            first_place.append(cmd)
        if "-hold" in cmd:
            return "0.05"
        if "report_route_status" in cmd:
            return ("# of routable nets : 50\n# of fully routed nets : 50\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-2.0"
        return "ok"
    import time as _t
    cfg = ILSPolishConfig(enabled=True, max_cells=300_000,
                          min_remaining_s=1100.0, stagnation_seconds=600.0,
                          min_cycle_seconds=0.0, max_cycles=1,
                          expected_heavy_cycle_s=400.0)
    asyncio.run(run_ils_polish(
        fake_call_tool, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 1500.0, wns_tcl="SLACK", cfg=cfg,
        log=lambda m: None))
    assert first_place and "-directive Explore" in first_place[0]


def test_tool_ok_rejects_tcl_error_output():
    """jun12 campaign: the MCP server reports Vivado-side failures as plain
    'TCL ERROR: <msg>' output, not a client error envelope — an instantly
    erroring write_checkpoint passed _tool_ok and produced a PHANTOM accept
    (recorded wns with a stale file on disk)."""
    from optimizer.ils_polish import _tool_ok
    assert _tool_ok("ok") is True
    assert _tool_ok("checkpoint_written") is True
    assert _tool_ok('{"error": "tool_timed_out_budget"}') is False
    assert _tool_ok("TCL ERROR: ERROR: [Common 17-69] Command failed") is False
    assert _tool_ok(None) is True   # non-string envelopes fail open (legacy)


def test_phantom_accept_blocked_by_tcl_error_write():
    """A cycle that improves WNS but whose write_checkpoint returns
    'TCL ERROR: ...' must NOT be accepted (file on disk is stale)."""
    async def fake_call_tool(tool, args):
        cmd = args.get("command", "")
        if "write_checkpoint" in cmd:
            return "TCL ERROR: ERROR: [Common 17-69] write failed"
        if "-hold" in cmd:
            return "0.05"
        if "report_route_status" in cmd:
            return ("# of routable nets : 50\n# of fully routed nets : 50\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-0.2"   # better than baseline -0.5 -> would-be accept
        return "ok"
    import time as _t
    res = asyncio.run(run_ils_polish(
        fake_call_tool, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 5.0, wns_tcl="SLACK", cfg=CFG_RUN,
        log=lambda m: None))
    assert res.accepted == 0
    assert res.best_wns == -0.5     # baseline preserved (never-worse)


def test_lastmile_session_poisoning_restart():
    """jun12 batch repro: after `place_design -directive LastMile`, the
    Vivado SESSION can no longer run a full placement (even on a freshly
    opened checkpoint) — affects AWS too. The loop must restart Vivado
    between a LASTMILE cycle and the next cycle, and on exit if the last
    cycle was LASTMILE."""
    from optimizer.ils_polish import LASTMILE_PD
    calls = []
    async def fake_call_tool(tool, args):
        calls.append((tool, args.get("command", "")))
        cmd = args.get("command", "")
        if tool == "vivado_restart_vivado":
            return "ok"
        if "-hold" in cmd:
            return "0.05"
        if "report_route_status" in cmd:
            return ("# of routable nets : 50\n# of fully routed nets : 50\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-2.0"   # never accept
        return "ok"
    import time as _t
    lm_idx = [c[0] for c in ILS_COMBOS].index(LASTMILE_PD)
    cfg = ILSPolishConfig(enabled=True, max_cells=300_000,
                          min_remaining_s=1100.0, stagnation_seconds=600.0,
                          min_cycle_seconds=0.0, max_cycles=2,
                          final_seed_no_improve_stop=0)
    asyncio.run(run_ils_polish(
        fake_call_tool, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.2,
        deadline_ts=_t.time() + 50.0, wns_tcl="SLACK", cfg=cfg,
        log=lambda m: None, combo_offset=lm_idx))
    # cycle 1 = LASTMILE, cycle 2 = next combo: a restart must occur between
    # them (i.e., after the first LastMile place and before the second cycle's
    # open_checkpoint).
    restart_idx = [i for i, (t, _) in enumerate(calls)
                   if t == "vivado_restart_vivado"]
    lastmile_place_idx = [i for i, (_, c) in enumerate(calls)
                          if "directive LastMile" in c]
    opens = [i for i, (_, c) in enumerate(calls) if "open_checkpoint" in c]
    assert restart_idx, "no vivado_restart_vivado call after LASTMILE cycle"
    assert lastmile_place_idx[0] < restart_idx[0]
    second_open = [i for i in opens if i > lastmile_place_idx[0]]
    assert second_open and restart_idx[0] < second_open[0]


def test_lastmile_skipped_far_from_closure():
    """jun13 overnight: place_design -directive LastMile FAILS on
    far-from-closure designs (3d at -2.1ns: 'Place design failed'). The
    LASTMILE combo must be skipped when the current best is below
    lastmile_min_wns_ns, and the cycle must run a real (non-LASTMILE)
    combo instead."""
    from optimizer.ils_polish import LASTMILE_PD
    placed = []
    async def fake_call_tool(tool, args):
        cmd = args.get("command", "")
        # real work = any place OR route directive (the combo after LASTMILE
        # is ROUTE_ONLY since jul02, which never places)
        if ("place_design -directive" in cmd or "directive LastMile" in cmd
                or cmd.startswith("route_design -directive")):
            placed.append(cmd)
        if tool == "vivado_restart_vivado":
            return "ok"
        if "-hold" in cmd:
            return "0.05"
        if "report_route_status" in cmd:
            return ("# of routable nets : 50\n# of fully routed nets : 50\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-2.0"   # far from closure, never accept
        return "ok"
    import time as _t
    lm_idx = [c[0] for c in ILS_COMBOS].index(LASTMILE_PD)
    cfg = ILSPolishConfig(enabled=True, max_cells=300_000,
                          min_remaining_s=1100.0, stagnation_seconds=600.0,
                          min_cycle_seconds=0.0, max_cycles=1,
                          final_seed_no_improve_stop=0)
    # baseline -2.0 (< -0.30) and rotation starts AT lastmile -> must skip it.
    asyncio.run(run_ils_polish(
        fake_call_tool, best_dcp_path="/tmp/x.dcp", baseline_wns=-2.0,
        deadline_ts=_t.time() + 50.0, wns_tcl="SLACK", cfg=cfg,
        log=lambda m: None, combo_offset=lm_idx))
    assert placed, "no real combo work ran"
    assert not any("directive LastMile" in c for c in placed), \
        "LASTMILE ran despite far-from-closure baseline"


def test_lastmile_allowed_near_closure():
    """Near closure (best > -0.30) the LASTMILE combo is NOT gated out."""
    from optimizer.ils_polish import LASTMILE_PD
    placed = []
    async def fake_call_tool(tool, args):
        cmd = args.get("command", "")
        if "place_design -directive" in cmd or "directive LastMile" in cmd:
            placed.append(cmd)
        if tool == "vivado_restart_vivado":
            return "ok"
        if "-hold" in cmd:
            return "0.05"
        if "report_route_status" in cmd:
            return ("# of routable nets : 50\n# of fully routed nets : 50\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-0.2"   # near closure, but make it never-better to avoid accept churn
        return "ok"
    import time as _t
    lm_idx = [c[0] for c in ILS_COMBOS].index(LASTMILE_PD)
    cfg = ILSPolishConfig(enabled=True, max_cells=300_000,
                          min_remaining_s=1100.0, stagnation_seconds=600.0,
                          min_cycle_seconds=0.0, max_cycles=1,
                          final_seed_no_improve_stop=0, accept_margin_ns=0.002)
    asyncio.run(run_ils_polish(
        fake_call_tool, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.25,
        deadline_ts=_t.time() + 50.0, wns_tcl="SLACK", cfg=cfg,
        log=lambda m: None, combo_offset=lm_idx))
    assert any("directive LastMile" in c for c in placed), \
        "LASTMILE was gated out despite near-closure baseline"


def test_write_checkpoint_retry_recovers_transient_failure():
    """jun13: a single transient write_checkpoint failure (WSL /mnt/c
    flakiness) must NOT lose a real accept — retry once before discard."""
    state = {"wc": 0}
    async def fake_call_tool(tool, args):
        cmd = args.get("command", "")
        if "write_checkpoint" in cmd:
            state["wc"] += 1
            if state["wc"] == 1:
                return '{"error": "transient write failure"}'  # first fails
            return "ok"                                          # retry succeeds
        if "-hold" in cmd:
            return "0.05"
        if "report_route_status" in cmd:
            return ("# of routable nets : 50\n# of fully routed nets : 50\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-0.2"   # better than baseline -0.5
        return "ok"
    import time as _t
    res = asyncio.run(run_ils_polish(
        fake_call_tool, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 5.0, wns_tcl="SLACK", cfg=CFG_RUN,
        log=lambda m: None))
    assert res.accepted >= 1           # retry rescued the accept
    assert res.best_wns == -0.2
    assert state["wc"] >= 2            # the retry actually happened


def test_write_checkpoint_double_failure_still_discards():
    """If BOTH the write and its retry fail, the accept is discarded
    (never-worse backstop intact)."""
    async def fake_call_tool(tool, args):
        cmd = args.get("command", "")
        if "write_checkpoint" in cmd:
            return '{"error": "persistent write failure"}'
        if "-hold" in cmd:
            return "0.05"
        if "report_route_status" in cmd:
            return ("# of routable nets : 50\n# of fully routed nets : 50\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-0.2"
        return "ok"
    import time as _t
    res = asyncio.run(run_ils_polish(
        fake_call_tool, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 5.0, wns_tcl="SLACK", cfg=CFG_RUN,
        log=lambda m: None))
    assert res.accepted == 0
    assert res.best_wns == -0.5        # baseline preserved


def test_lastmile_allowed_at_proven_win_baselines():
    """REGRESSION (jun13): the gate must NOT block LASTMILE at its proven-win
    baselines (logicnets -0.496, v2 -0.799/-0.946, mini-isp -0.882) — an
    earlier -0.30 gate blocked LASTMILE on ALL contest designs (every one
    ends < -0.30). Gate is -1.0; -0.9 (worst proven win class) must pass."""
    from optimizer.ils_polish import LASTMILE_PD
    placed = []
    async def fake_call_tool(tool, args):
        cmd = args.get("command", "")
        if "place_design -directive" in cmd or "directive LastMile" in cmd:
            placed.append(cmd)
        if tool == "vivado_restart_vivado":
            return "ok"
        if "-hold" in cmd:
            return "0.05"
        if "report_route_status" in cmd:
            return ("# of routable nets : 50\n# of fully routed nets : 50\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-1.5"   # never better; we only inspect dispatch
        return "ok"
    import time as _t
    lm_idx = [c[0] for c in ILS_COMBOS].index(LASTMILE_PD)
    cfg = ILSPolishConfig(enabled=True, max_cells=300_000,
                          min_remaining_s=1100.0, stagnation_seconds=600.0,
                          min_cycle_seconds=0.0, max_cycles=1,
                          final_seed_no_improve_stop=0)
    asyncio.run(run_ils_polish(
        fake_call_tool, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.9,
        deadline_ts=_t.time() + 50.0, wns_tcl="SLACK", cfg=cfg,
        log=lambda m: None, combo_offset=lm_idx))
    assert any("directive LastMile" in c for c in placed), \
        "LASTMILE gated out at -0.9 (a proven-win baseline) — gate too tight"


# ---- fanout_polish_accept (jun14 DRILL D): strict-hold never-worse gate ----
FCFG = ILSPolishConfig()   # defaults: accept_margin_ns=0.002, fanout floor=0.010


def test_fanout_accept_hold_safe_gain():
    # logicnets-class: real setup gain, hold safe (whs 0.068) -> ACCEPT
    ok, why = fanout_polish_accept(new_wns=-0.572, best_wns=-0.601, unrouted=0,
                                   whs=0.068, base_whs=0.068, cfg=FCFG)
    assert ok, why


def test_fanout_reject_hold_marginal():
    # ispd16-class: big setup gain BUT hold eroded to ~0.000 (< 0.010 floor)
    # -> REJECT (validator gates hold_passed; a reject is free/never-worse).
    ok, why = fanout_polish_accept(new_wns=-6.770, best_wns=-7.752, unrouted=0,
                                   whs=0.000, base_whs=0.003, cfg=FCFG)
    assert not ok and "strict floor" in why


def test_fanout_reject_no_setup_gain():
    ok, why = fanout_polish_accept(new_wns=-0.601, best_wns=-0.601, unrouted=0,
                                   whs=0.080, base_whs=0.080, cfg=FCFG)
    assert not ok and "no setup gain" in why


def test_fanout_reject_unrouted():
    ok, why = fanout_polish_accept(new_wns=-0.50, best_wns=-0.601, unrouted=7,
                                   whs=0.080, base_whs=0.080, cfg=FCFG)
    assert not ok and "unrouted" in why
    # parse failure (ur == -1 from _measure) is also rejected
    ok2, _ = fanout_polish_accept(new_wns=-0.50, best_wns=-0.601, unrouted=-1,
                                  whs=0.080, base_whs=0.080, cfg=FCFG)
    assert not ok2


def test_fanout_reject_hold_worsened():
    # setup gain + above floor, but hold worse than pre-polish best -> REJECT
    ok, why = fanout_polish_accept(new_wns=-0.55, best_wns=-0.601, unrouted=0,
                                   whs=0.012, base_whs=0.090, cfg=FCFG)
    assert not ok and "hold worsened" in why


def test_fanout_reject_missing_wns():
    ok, _ = fanout_polish_accept(new_wns=None, best_wns=-0.601, unrouted=0,
                                 whs=0.080, base_whs=0.080, cfg=FCFG)
    assert not ok


def test_fanout_floor_boundary():
    # exactly at the floor accepts; just below rejects
    ok_at, _ = fanout_polish_accept(new_wns=-0.55, best_wns=-0.601, unrouted=0,
                                    whs=0.010, base_whs=0.010, cfg=FCFG)
    assert ok_at
    ok_below, _ = fanout_polish_accept(new_wns=-0.55, best_wns=-0.601, unrouted=0,
                                       whs=0.009, base_whs=0.009, cfg=FCFG)
    assert not ok_below


# ---- cell-count guard (jul03, GitHub #36): LASTMILE accepts vs validator band ----

def _lastmile_accept_calltool(cell_count):
    """LASTMILE cycle that would accept (-0.2 vs baseline -0.5, hold clean);
    the cell query returns cell_count."""
    async def fake(tool, args):
        cmd = args.get("command", "")
        if "get_cells" in cmd and "llength" in cmd:
            return str(cell_count)
        if "-hold" in cmd:
            return "0.05"
        if "report_route_status" in cmd:
            return ("# of routable nets : 10\n# of fully routed nets : 10\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-0.2"
        return "ok"
    return fake


def _run_lastmile(cell_count, golden):
    import time as _t
    from optimizer.ils_polish import LASTMILE_PD
    lm = [c[0] for c in ILS_COMBOS].index(LASTMILE_PD)
    cfg = ILSPolishConfig(enabled=True, min_cycle_seconds=0.0, max_cycles=1,
                          final_seed_no_improve_stop=0,
                          golden_cell_count=golden)
    return asyncio.run(run_ils_polish(
        _lastmile_accept_calltool(cell_count), best_dcp_path="/tmp/x.dcp",
        baseline_wns=-0.5, deadline_ts=_t.time() + 50.0, wns_tcl="SLACK",
        cfg=cfg, log=lambda m: None, combo_offset=lm))


def test_cell_guard_rejects_mangled_netlist():
    # golden 100k, LASTMILE result 40k (0.4x < 0.5 sanity floor) -> REJECTED
    res = _run_lastmile(cell_count=40_000, golden=100_000)
    assert res.accepted == 0 and res.best_wns == -0.5
    assert any("cell-count" in n for n in res.notes)


def test_cell_guard_allows_in_band():
    # 99k (-1%) is inside the band -> accept proceeds
    res = _run_lastmile(cell_count=99_000, golden=100_000)
    assert res.accepted == 1 and res.best_wns == -0.2


def test_cell_guard_allows_legit_shrink_post_pr41():
    # jul13 relaxation (upstream PR #41 removed the validator's hard Check-4
    # gate): a legitimate -lut_opt shrink to 90k (-10%, would have FAILED the
    # old [0.975x, 1.45x] guard) is now accepted — counts are info-only
    # upstream, our band is sanity-only.
    res = _run_lastmile(cell_count=90_000, golden=100_000)
    assert res.accepted == 1 and res.best_wns == -0.2


def test_cell_guard_fails_open_without_golden():
    res = _run_lastmile(cell_count=1, golden=None)
    assert res.accepted == 1                 # guard disabled, accept proceeds


# ---- derive_cost_anchors (jul04 genericity audit): cold-start cost anchors ----
# The full-cycle anchor (place+route) gates the ILS combo picker; the dedicated
# fanout anchor (phys_opt+route = the polish's actual worst case) is the
# fallback for the polish's cheap-design gate on route-only (R7) recipe paths,
# which never place and thus left the polish self-gated on exactly the OOD
# designs it should serve (corundum, jul04).
from optimizer.ils_polish import derive_cost_anchors


def test_anchor_granular_place_route():
    # granular tools: per-stage max summed for the full cycle; fanout anchor
    # is phys_opt + route only (no place — the polish never places).
    tcs = [
        {"tool_name": "vivado_place_design", "elapsed_time": 200.0},
        {"tool_name": "vivado_route_design", "elapsed_time": 300.0},
        {"tool_name": "vivado_phys_opt_design", "elapsed_time": 100.0},
        {"tool_name": "vivado_place_design", "elapsed_time": 150.0},
    ]
    est, fan, singles = derive_cost_anchors(tcs)
    assert est == 600.0
    assert fan == 400.0
    assert singles["place_design"] == 200.0


def test_anchor_full_cycle_sample_not_double_counted():
    # one Tcl doing place->route is a direct full-cycle sample; it yields NO
    # single-stage samples, so the fanout anchor stays unknown.
    tcs = [{"tool_name": "vivado_run_tcl",
            "cmd_head": "place_design -directive Explore; route_design",
            "elapsed_time": 500.0}]
    est, fan, _ = derive_cost_anchors(tcs)
    assert est == 500.0
    assert fan == 0.0


def test_anchor_r1_route_lever_path():
    # R1 route-lever shape (retiming -> unroute -> route via separate
    # run_tcl calls, NEVER place_design): full-cycle anchor underivable,
    # fanout anchor IS (phys_opt + route singles).
    tcs = [
        {"tool_name": "vivado_run_tcl",
         "cmd_head": "phys_opt_design -directive AggressiveExplore",
         "elapsed_time": 120.0},
        {"tool_name": "vivado_run_tcl",
         "cmd_head": "route_design -directive AggressiveExplore",
         "elapsed_time": 250.0},
    ]
    est, fan, _ = derive_cost_anchors(tcs)
    assert est == 0.0
    assert fan == 370.0


def test_anchor_r7_closure_ladder_path():
    # R7 closure-ladder shape, replayed from the REAL jul04 corundum OOD run
    # (corundum_freeze_ood.log): granular vivado_phys_opt_design calls ONLY —
    # no place, no route in the recipe phase. Old derivation -> anchor 0 ->
    # "fanout-polish: skipped (cost anchor 0s ...)" live at 02:32:59. New:
    # fanout anchor = max phys_opt sample (68s), well under the 600s gate.
    tcs = [
        {"tool_name": "vivado_phys_opt_design", "elapsed_time": 64.0},
        {"tool_name": "vivado_phys_opt_design", "elapsed_time": 57.0},
        {"tool_name": "vivado_phys_opt_design", "elapsed_time": 68.0},
        {"tool_name": "vivado_phys_opt_design", "elapsed_time": 29.0},
        {"tool_name": "vivado_run_tcl",
         "cmd_head": "report_qor_assessment -exclude_methodology_checks",
         "elapsed_time": 51.0},
    ]
    est, fan, _ = derive_cost_anchors(tcs)
    assert est == 0.0
    assert fan == 68.0


def test_anchor_empty_and_none_elapsed():
    assert derive_cost_anchors([]) == (0.0, 0.0, {})
    est, fan, _ = derive_cost_anchors(
        [{"tool_name": "vivado_route_design", "elapsed_time": None}])
    assert est == 0.0 and fan == 0.0


def test_anchor_route_only_single():
    est, fan, _ = derive_cost_anchors(
        [{"tool_name": "vivado_route_design", "elapsed_time": 250.0}])
    assert est == 0.0 and fan == 250.0


def test_fanout_cost_anchor_default_unknown():
    assert ILSPolishConfig().fanout_cost_anchor_s == 0.0


# ---- fanout-polish cheap-design gate: granular-anchor fallback (jul04) ----

def _fanout_gate_probe(expected, fan_anchor, deadline_offset=5000.0,
                       has_route=False):
    """Drive DCPOptimizer._fanout_polish_after_ils on a stub up to (at most)
    the first Vivado call; the stub's call_tool returns error envelopes so
    the method aborts safely right after the gate. Returns #Vivado calls:
    0 = gated out before touching Vivado, >0 = gate passed."""
    import time as _t
    import dcp_optimizer as do
    calls = []

    class _Stub:
        run_dir = None

        def __init__(self):
            self._ils_polish_cfg = ILSPolishConfig(
                expected_heavy_cycle_s=expected,
                fanout_cost_anchor_s=fan_anchor,
                fanout_anchor_has_route=has_route)

        async def call_tool(self, name, args):
            calls.append(name)
            return '{"error": "stub"}'

    stub = _Stub()
    asyncio.run(do.DCPOptimizer._fanout_polish_after_ils(
        stub, "/tmp/best.dcp", -0.5, _t.time() + deadline_offset, "SLACK"))
    return len(calls)


def test_fanout_gate_skips_with_no_anchor_at_all():
    assert _fanout_gate_probe(expected=0.0, fan_anchor=0.0) == 0


def test_fanout_gate_falls_back_to_granular_anchor():
    # R7 path: full-cycle anchor unknown, granular anchor cheap -> gate PASSES
    # (reaches Vivado; stub error envelope then aborts it, never-worse).
    assert _fanout_gate_probe(expected=0.0, fan_anchor=300.0) > 0


def test_fanout_gate_granular_anchor_still_respects_cost_cap():
    # granular anchor above fanout_max_cycle_s (600) -> still gated out.
    assert _fanout_gate_probe(expected=0.0, fan_anchor=700.0) == 0
    # ... even when complete (ispd16-class: route sample alone is huge).
    assert _fanout_gate_probe(expected=2000.0, fan_anchor=700.0,
                              has_route=True) == 0


def test_fanout_gate_incomplete_fan_anchor_keeps_full_cycle_basis():
    # Without a route sample the potential reroute cost is unknowable: the
    # full-cycle anchor stays the cost basis when it is known — a cheap
    # phys_opt-only fan anchor must NOT resurrect the polish on a design
    # whose full cycle is slow (>600s).
    assert _fanout_gate_probe(expected=1200.0, fan_anchor=300.0) == 0


def test_fanout_gate_record_run_logicnets_replay():
    # Preview #6 (RECORD, 2026-07-03) eval-box numbers: full-cycle anchor
    # 489s (incl. 252s place the polish never runs), fan anchor 237s
    # (phys_opt 85 + route 93 + 60) WITH route sample, 1194s remaining.
    # Frozen code: need = 489*1.3+600 = 1236s > 1194 -> polish MISSED BY
    # 42s on the record run. New basis: need = 237*1.3+600 = 908s -> FIRES.
    assert _fanout_gate_probe(expected=489.0, fan_anchor=237.0,
                              deadline_offset=1194.0, has_route=True) > 0


def test_fanout_gate_v2_eval_replay():
    # Preview #6 v2: full-cycle 718s (581s place!) hard-skipped the cheap
    # gate; true polish cost 137s (phys 34 + route 43 + 60) -> now attempts.
    # Never-worse + strict-hold accept make a reject free.
    assert _fanout_gate_probe(expected=718.0, fan_anchor=137.0,
                              has_route=True) > 0


# ---- corrective-seed local climb (jul04 preview #5-vs-#6 forensics) ----

def _wns_sequence_calltool(wns_seq):
    """Fake Vivado: each SLACK query pops the next WNS from wns_seq (the last
    value repeats). Routing always clean, hold safe, cells stable."""
    state = {"i": 0}

    async def fake(tool, args):
        cmd = args.get("command", "")
        if "get_cells" in cmd and "llength" in cmd:
            return "100000"
        if "-hold" in cmd:
            return "0.05"
        if "report_route_status" in cmd:
            return ("# of routable nets : 10\n# of fully routed nets : 10\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            i = min(state["i"], len(wns_seq) - 1)
            state["i"] += 1
            return str(wns_seq[i])
        return "ok"
    return fake


def _run_seq(wns_seq, max_cycles, baseline=-0.9, offset=0):
    import time as _t
    cfg = ILSPolishConfig(enabled=True, min_cycle_seconds=0.0,
                          max_cycles=max_cycles, final_seed_no_improve_stop=0,
                          lastmile_min_wns_ns=-99.0)
    return asyncio.run(run_ils_polish(
        _wns_sequence_calltool(wns_seq), best_dcp_path="/tmp/x.dcp",
        baseline_wns=baseline, deadline_ts=_t.time() + 200.0, wns_tcl="SLACK",
        cfg=cfg, log=lambda m: None, combo_offset=offset))


def test_pristine_rot_freezes_at_first_accept():
    # cycle 1 (idx0) measures -0.95: no accept vs -0.9; cycle 2 (idx1)
    # measures -0.5: ACCEPT; cycle 3 (idx2) measures -1.2: no accept.
    # NOTE each cycle issues ~2 SLACK queries (post-route + post-phys_opt);
    # keep values pairwise identical so cycle outcomes are unambiguous.
    res = _run_seq([-0.95, -0.95, -0.5, -0.5, -1.2, -1.2], max_cycles=3)
    assert res.accepted >= 1
    # first accept happened on the cycle picked at rotation index 2 ->
    # pristine_rot frozen at 2 (idx0 and the accepting idx1 are the only
    # genuine replays for a near-identical sibling seed).
    assert res.pristine_rot == 2


def test_pristine_rot_zero_accepts_equals_continue_semantics():
    # no cycle accepts -> pristine_rot ends at the final rotation position,
    # reproducing the old continue-rotation offset (spam-filter jun10 case).
    # baseline in the tightened route-reroll band (jul22, 0.7) so no
    # combo inline-skips and pristine_rot tracks executed cycles 1:1.
    res = _run_seq([-1.5, -1.5, -1.4, -1.4, -1.3, -1.3], max_cycles=3,
                   baseline=-0.5)
    assert res.accepted == 0
    assert res.pristine_rot == 3 == res.cycles


def test_corrective_local_climb_default_on():
    # jul05: default ON after 3/3 live validation (v2 debug-wall runs,
    # shipped >= control every time); eval #5-vs-#6 forensics carry the
    # upside. False remains the kill switch.
    assert ILSPolishConfig().corrective_local_climb is True


# ---- ROUTE_ONLY granular cold-start estimate (jul04 local v2 leg) ----

def test_cold_start_route_only_uses_granular_estimate():
    """jul04 local-campaign v2 leg: place-dominated anchor (place 1568s of
    expected_heavy_cycle_s=1790s) priced ROUTE_ONLY at 0.6*1790=1074s in an
    875s window -> 'no affordable combo' -> ILS ran 0 cycles and shipped
    BASELINE, on the design where ROUTE_ONLY accepted in previews #5 AND #6.
    With the granular route+phys anchor (221s, has_route) an unroute +
    re-route combo must be affordable and picked (since jul21 the first
    such combo at near-met baseline -0.5 is ROUTE_REROLL at idx2, basis
    221*1.3=287s; ROUTE_ONLY at idx3 keeps the 221s granular estimate)."""
    first_heavy = []
    async def fake_call_tool(tool, args):
        cmd = args.get("command", "")
        if ("place_design" in cmd or "route_design -unroute" in cmd) \
                and len(first_heavy) == 0:
            first_heavy.append(cmd)
        if "-hold" in cmd:
            return "0.05"
        if "report_route_status" in cmd:
            return ("# of routable nets : 50\n# of fully routed nets : 50\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-2.0"
        return "ok"
    import time as _t
    cfg = ILSPolishConfig(enabled=True, min_cycle_seconds=0.0, max_cycles=1,
                          expected_heavy_cycle_s=1790.0,
                          fanout_cost_anchor_s=221.0,
                          fanout_anchor_has_route=True)
    res = asyncio.run(run_ils_polish(
        fake_call_tool, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 875.0, wns_tcl="SLACK", cfg=cfg,
        log=lambda m: None))
    assert res.cycles == 1
    assert first_heavy and "route_design -unroute" in first_heavy[0]


def test_cold_start_route_only_without_route_sample_keeps_prior():
    """Without a route sample the granular anchor is untrustworthy for a
    reroute: the scaled full-cycle prior stays (0.6*1790=1074 > 875 window
    -> no cycle runs)."""
    async def fake_call_tool(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "0.05"
        if "SLACK" in cmd:
            return "-2.0"
        return "ok"
    import time as _t
    cfg = ILSPolishConfig(enabled=True, min_cycle_seconds=0.0, max_cycles=1,
                          expected_heavy_cycle_s=1790.0,
                          fanout_cost_anchor_s=221.0,
                          fanout_anchor_has_route=False)
    res = asyncio.run(run_ils_polish(
        fake_call_tool, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 875.0, wns_tcl="SLACK", cfg=cfg,
        log=lambda m: None))
    assert res.cycles == 0


# ---- fanout-polish finalize reserve (jul04: five consecutive near-misses) ----

def test_fanout_gate_eval7_v2_replay_fires_with_finalize_reserve():
    # Eval #7 v2: granular anchor 137 (no place sample in recipe), 731s
    # remaining. Old need = 137*1.3 + 600 = 778 -> missed by 47s. New need =
    # 137*1.3 + 300 = 478 -> FIRES.
    assert _fanout_gate_probe(expected=0.0, fan_anchor=137.0,
                              deadline_offset=731.0, has_route=True) > 0


def test_fanout_gate_reserve_still_skips_truly_tight():
    # Local digit leg: 233s remaining cannot fit any attempt (need >= 478).
    assert _fanout_gate_probe(expected=0.0, fan_anchor=137.0,
                              deadline_offset=233.0, has_route=True) == 0


# ---- combo-cost carry across seeds (GAP #4, eval #9 v2 forensics) ----

def test_combo_cost_seed_makes_observed_cheap_combo_affordable():
    """Eval #9 v2: the corrective seed's fresh cost table priced
    ExtraTimingOpt off cold priors (1.4x anchor) and skipped it — even though
    the raw seed had JUST measured it at 254s. With the observed table
    threaded in, the combo must be affordable and picked."""
    import time as _t
    placed = []
    async def fake(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "0.05"
        if cmd.startswith("place_design -directive"):
            placed.append(cmd.split()[-1])
        if "report_route_status" in cmd:
            return ("# of routable nets : 10\n# of fully routed nets : 10\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-2.0"
        return "ok"
    eto = [c[0] for c in ILS_COMBOS].index("ExtraTimingOpt")
    cfg = ILSPolishConfig(enabled=True, min_cycle_seconds=0.0, max_cycles=1,
                          expected_heavy_cycle_s=1790.0,
                          lastmile_min_wns_ns=-99.0)
    # window 500s: prior estimate 1.4*1790=2506 -> skipped without carry
    res_cold = asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 500.0, wns_tcl="SLACK", cfg=cfg,
        log=lambda m: None, combo_offset=eto))
    cold_first = placed[0] if placed else None
    placed.clear()
    # with the observed 254s carried in -> ExtraTimingOpt picked first
    res_carry = asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 500.0, wns_tcl="SLACK", cfg=cfg,
        log=lambda m: None, combo_offset=eto,
        combo_cost_seed={eto: 254.0}))
    assert placed and placed[0] == "ExtraTimingOpt"
    assert cold_first != "ExtraTimingOpt"


def test_result_exports_observed_combo_costs():
    import time as _t
    async def fake(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "0.05"
        if "report_route_status" in cmd:
            return ("# of routable nets : 10\n# of fully routed nets : 10\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-2.0"
        return "ok"
    res = asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.5,
        deadline_ts=_t.time() + 200.0, wns_tcl="SLACK",
        cfg=ILSPolishConfig(enabled=True, min_cycle_seconds=0.0, max_cycles=1,
                            lastmile_min_wns_ns=-99.0),
        log=lambda m: None))
    assert res.cycles == 1 and len(res.combo_cost) == 1


# ---- LASTMILE final polish (jul06): accept gate ----
from optimizer.ils_polish import lastmile_polish_accept

LCFG = ILSPolishConfig(golden_cell_count=100_000)


def test_lastmile_polish_accepts_probe_case():
    # jun12 probe: v2 -0.799 -> -0.647, hold clean, cells in band
    ok, why = lastmile_polish_accept(new_wns=-0.647, best_wns=-0.799,
                                     unrouted=0, whs=0.02, cell_count=101_000,
                                     cfg=LCFG)
    assert ok, why


def test_lastmile_polish_rejects_hold_below_official_gate():
    ok, why = lastmile_polish_accept(new_wns=-0.647, best_wns=-0.799,
                                     unrouted=0, whs=-0.01, cell_count=100_000,
                                     cfg=LCFG)
    assert not ok and "hold" in why


def test_lastmile_polish_rejects_cell_band_violation():
    # netlist mangled far below the sanity floor (0.3x < 0.5x)
    ok, why = lastmile_polish_accept(new_wns=-0.647, best_wns=-0.799,
                                     unrouted=0, whs=0.02, cell_count=30_000,
                                     cfg=LCFG)
    assert not ok and "sanity band" in why


def test_lastmile_polish_allows_legit_shrink_post_pr41():
    # jul13: -10% shrink is legal now (upstream PR #41; band is sanity-only)
    ok, why = lastmile_polish_accept(new_wns=-0.647, best_wns=-0.799,
                                     unrouted=0, whs=0.02, cell_count=90_000,
                                     cfg=LCFG)
    assert ok, why


def test_lastmile_polish_rejects_unrouted_and_no_gain():
    assert not lastmile_polish_accept(new_wns=-0.5, best_wns=-0.799, unrouted=3,
                                      whs=0.02, cell_count=100_000, cfg=LCFG)[0]
    assert not lastmile_polish_accept(new_wns=-0.799, best_wns=-0.799, unrouted=0,
                                      whs=0.02, cell_count=100_000, cfg=LCFG)[0]


def test_lastmile_polish_fails_open_without_golden():
    ok, _ = lastmile_polish_accept(new_wns=-0.647, best_wns=-0.799, unrouted=0,
                                   whs=0.02, cell_count=None,
                                   cfg=ILSPolishConfig())
    assert ok


def test_lastmile_polish_default_on():
    assert ILSPolishConfig().lastmile_polish_enabled is True


# ---- LASTMILE final-polish stage wrapper (jul06, c46e62e+bff156a) ----
# Direct tests for DCPOptimizer._lastmile_polish_after_ils — the integrated
# stage (entry/budget gates, step-error abort, the bff156a incremental
# reroute retry, accept bookkeeping). Until now only the pure accept
# function was covered; the retry branch shipped in #13 with zero direct
# coverage (live jul06 v2 run predates bff156a).

def _lastmile_stage_stub(*, route_status_seq, slack_seq, hold="0.05",
                         cells="100000", fail_cmd_substr=None,
                         write_fails=False, cfg=None):
    """Stub DCPOptimizer host + scripted fake Vivado for the LASTMILE stage.

    route_status_seq / slack_seq: successive return payloads for
    report_route_status / the wns_tcl query (last value repeats).
    fail_cmd_substr: first polish step whose command contains this substring
    returns a TCL ERROR envelope. write_fails: write_checkpoint errors.
    Returns (stub, calls) where calls = list of issued Tcl commands."""
    import dcp_optimizer as do
    calls = []
    state = {"rs": 0, "sl": 0}

    class _Stub:
        run_dir = None
        best_wns = None
        _best_valid_dcp = None
        _best_valid_dcp_wns = None

        def __init__(self):
            self._ils_polish_cfg = cfg or ILSPolishConfig()

        def _maybe_arm_wall_handback(self, reason):
            # D3 (04-01): the LASTMILE reject branch reports itself as a
            # saturation signal; the stub just records it (arm-guard logic
            # is covered in tests/test_wall_handback.py, not here).
            self.wall_handback_reasons = getattr(
                self, "wall_handback_reasons", []) + [reason]

        async def call_tool(self, name, args):
            cmd = args.get("command", "") if isinstance(args, dict) else ""
            calls.append(cmd or name)
            if fail_cmd_substr and fail_cmd_substr in cmd:
                return "TCL ERROR: injected step failure"
            if "write_checkpoint" in cmd:
                return ("TCL ERROR: injected write failure" if write_fails
                        else "ok")
            if "-hold" in cmd:
                return hold
            if "report_route_status" in cmd:
                i = min(state["rs"], len(route_status_seq) - 1)
                state["rs"] += 1
                return route_status_seq[i]
            if "LMSLACK" in cmd:
                i = min(state["sl"], len(slack_seq) - 1)
                state["sl"] += 1
                return slack_seq[i]
            if "llength" in cmd and "get_cells" in cmd:
                return cells
            return "ok"

    return _Stub(), calls


_RS_CLEAN = ("# of routable nets : 49\n# of fully routed nets : 49\n"
             "# of nets with routing errors : 0\n")
_RS_UR39 = ("# of routable nets : 49\n# of fully routed nets : 10\n"
            "# of nets with routing errors : 0\n")


def _drive_lastmile(stub, best_wns=-0.5, deadline_offset=5000.0):
    import time as _t
    import dcp_optimizer as do
    asyncio.run(do.DCPOptimizer._lastmile_polish_after_ils(
        stub, "/tmp/best.dcp", best_wns, _t.time() + deadline_offset,
        "LMSLACK"))


def test_lastmile_stage_accept_clean_first_measure():
    # Clean route on the first measure, better wns, hold safe -> ACCEPT with
    # NO retry reroute; best bookkeeping updated to the polish output.
    stub, calls = _lastmile_stage_stub(route_status_seq=[_RS_CLEAN],
                                       slack_seq=["-0.2"])
    _drive_lastmile(stub, best_wns=-0.5)
    assert stub.best_wns == -0.2
    assert stub._best_valid_dcp is not None
    assert str(stub._best_valid_dcp).endswith("ils_lastmile_polish.dcp")
    assert not any(c.strip() == "route_design" for c in calls)  # no retry
    assert any("write_checkpoint" in c for c in calls)


def test_lastmile_stage_retry_reroute_then_accept():
    # jul06 v2 case + the bff156a fix: first measure sees 39 unrouted ->
    # ONE bare incremental route_design -> second measure clean -> ACCEPT
    # judged on the SECOND measurement.
    stub, calls = _lastmile_stage_stub(
        route_status_seq=[_RS_UR39, _RS_CLEAN],
        slack_seq=["-0.9", "-0.2"])
    _drive_lastmile(stub, best_wns=-0.5)
    assert [c for c in calls if c.strip() == "route_design"], \
        "incremental reroute retry must fire on unrouted>0"
    assert stub.best_wns == -0.2          # second (post-retry) wns wins
    assert any("write_checkpoint" in c for c in calls)


def test_lastmile_stage_retry_fails_rejects_never_worse():
    # Reroute retry still leaves unrouted nets -> reject, keep prior best,
    # and NEVER loop (exactly one bare route_design).
    stub, calls = _lastmile_stage_stub(
        route_status_seq=[_RS_UR39, _RS_UR39],
        slack_seq=["-0.2"])
    _drive_lastmile(stub, best_wns=-0.5)
    assert sum(1 for c in calls if c.strip() == "route_design") == 1
    assert stub.best_wns is None          # untouched
    assert stub._best_valid_dcp is None
    assert not any("write_checkpoint" in c for c in calls)


def test_lastmile_stage_entry_gate_below_min_wns():
    # wns below lastmile_min_wns_ns (UG906 closure gate) -> zero Vivado calls.
    stub, calls = _lastmile_stage_stub(route_status_seq=[_RS_CLEAN],
                                       slack_seq=["-0.2"])
    _drive_lastmile(stub, best_wns=-5.0)
    assert calls == []


def test_lastmile_stage_budget_gate():
    # remaining < est*1.3 + finalize reserve -> zero Vivado calls.
    stub, calls = _lastmile_stage_stub(route_status_seq=[_RS_CLEAN],
                                       slack_seq=["-0.2"])
    _drive_lastmile(stub, best_wns=-0.5, deadline_offset=200.0)
    assert calls == []


def test_lastmile_stage_kill_switch():
    stub, calls = _lastmile_stage_stub(
        route_status_seq=[_RS_CLEAN], slack_seq=["-0.2"],
        cfg=ILSPolishConfig(lastmile_polish_enabled=False))
    _drive_lastmile(stub, best_wns=-0.5)
    assert calls == []


def test_lastmile_stage_step_error_aborts_before_measure():
    # A polish step erroring (e.g. place_design) -> early return: no
    # measurement, no accept, best untouched (never-worse).
    stub, calls = _lastmile_stage_stub(
        route_status_seq=[_RS_CLEAN], slack_seq=["-0.2"],
        fail_cmd_substr="place_design")
    _drive_lastmile(stub, best_wns=-0.5)
    assert stub.best_wns is None
    assert not any("report_route_status" in c for c in calls)
    assert not any("write_checkpoint" in c for c in calls)


def test_lastmile_stage_write_failure_keeps_best():
    # Accept passes but write_checkpoint errors -> best NOT updated (the
    # jun12 phantom-accept lesson applied to this stage).
    stub, calls = _lastmile_stage_stub(
        route_status_seq=[_RS_CLEAN], slack_seq=["-0.2"], write_fails=True)
    _drive_lastmile(stub, best_wns=-0.5)
    assert stub.best_wns is None
    assert stub._best_valid_dcp is None


# ---- gain-weighted futility counter (jul06 #12-vs-#13 gamma forensics) ----
# A micro-accept (gain < meaningful_accept_ns) is KEPT but counts as a futile
# cycle: preview #13's +0.007ns accept reset the K=2 counter and bought 24min
# of dead cycles (gamma 0.577->0.917, net -1.2 score vs #12). Corpus mining
# (58 accepting seeds): micro-accepts are terminal 4/4 — never followed by a
# meaningful accept in the same seed.

def _run_futility_seq(wns_seq, *, no_improve_stop=2, meaningful=None,
                      baseline=-0.5, max_cycles=6):
    import time as _t
    kw = {}
    if meaningful is not None:
        kw["meaningful_accept_ns"] = meaningful
    cfg = ILSPolishConfig(enabled=True, min_cycle_seconds=0.0,
                          max_cycles=max_cycles,
                          final_seed_no_improve_stop=0,
                          lastmile_min_wns_ns=-99.0, **kw)
    return asyncio.run(run_ils_polish(
        _wns_sequence_calltool(wns_seq), best_dcp_path="/tmp/x.dcp",
        baseline_wns=baseline, deadline_ts=_t.time() + 200.0, wns_tcl="SLACK",
        cfg=cfg, log=lambda m: None, no_improve_stop=no_improve_stop))


def test_meaningful_accept_default():
    assert ILSPolishConfig().meaningful_accept_ns == 0.010


def test_micro_accept_kept_but_counts_futile():
    # cycle 1: -0.493 = ACCEPT with gain 0.007 (< 0.010) -> kept, streak 1;
    # cycle 2: -0.4945 = no accept -> streak 2 -> STOP. Legacy would have
    # reset at cycle 1 and kept cycling.
    res = _run_futility_seq([-0.493, -0.493, -0.4945, -0.4945])
    assert res.accepted == 1
    assert res.best_wns == -0.493          # micro gain is STILL kept
    assert res.cycles == 2                 # stopped early, gamma saved
    assert any("meaningful" in n for n in res.notes)


def test_two_micro_accepts_stop_inside_accept_branch():
    # cycle 1: gain 0.007 (micro, streak 1); cycle 2: -0.485 = ACCEPT gain
    # 0.008 (micro, streak 2) -> stop fires IN the accept branch. Both
    # improvements kept.
    res = _run_futility_seq([-0.493, -0.493, -0.485, -0.485])
    assert res.accepted == 2
    assert res.best_wns == -0.485
    assert res.cycles == 2
    assert any("no meaningful improvement" in n for n in res.notes)


def test_meaningful_accept_still_resets_streak():
    # cycle 1: -0.55 no accept (streak 1); cycle 2: -0.45 ACCEPT gain 0.05
    # (meaningful -> reset); cycles 3-4: no accept -> stop at streak 2.
    # Total 4 cycles proves the reset happened after the prior futile cycle.
    res = _run_futility_seq([-0.55, -0.55, -0.45, -0.45,
                             -0.452, -0.452, -0.453, -0.453])
    assert res.accepted == 1
    assert res.best_wns == -0.45
    assert res.cycles == 4


def test_meaningful_zero_restores_legacy_reset_on_any_accept():
    # kill switch: meaningful_accept_ns=0.0 -> the same micro accept resets
    # the counter (cycle 1), so the stop needs two MORE futile cycles.
    res = _run_futility_seq([-0.493, -0.493, -0.4945, -0.4945,
                             -0.4946, -0.4946], meaningful=0.0)
    assert res.accepted == 1
    assert res.best_wns == -0.493
    assert res.cycles == 3


# ---- K3 spread gate on PARTIAL_RUIN (jul20 whole-history mining) ----
# Held-out rule 2 (final_round/k3_history_mining_jul20.md): multi-cell
# surgery on a co-located critical path (spread ~ 0) is 26/26 negative
# (mean -1.10ns); on spread-diagnosed paths it is strongly positive.
# The gate skips ONLY the PARTIAL_RUIN combo family; rotation continues.

def _spread_gate_calltool(cmds):
    async def fake(tool, args):
        cmd = args.get("command", "")
        cmds.append(cmd)
        if "get_cells" in cmd and "llength" in cmd:
            return "100000"
        if "-hold" in cmd:
            return "0.05"
        if "report_route_status" in cmd:
            return ("# of routable nets : 10\n# of fully routed nets : 10\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-2.0"     # always worse than baseline -> never accept
        return "ok"
    return fake


def _run_spread(spread, *, gate=True, threshold=30.0, max_cycles=20):
    import time as _t
    cmds: list = []
    logs: list = []
    cfg = ILSPolishConfig(enabled=True, min_cycle_seconds=0.0,
                          max_cycles=max_cycles, final_seed_no_improve_stop=0,
                          lastmile_min_wns_ns=-99.0,
                          partial_ruin_spread_gate=gate,
                          partial_ruin_spread_min_tiles=threshold,
                          critical_path_avg_spread_tiles=spread)
    res = asyncio.run(run_ils_polish(
        _spread_gate_calltool(cmds), best_dcp_path="/tmp/x.dcp",
        baseline_wns=-0.5, deadline_ts=_t.time() + 200.0, wns_tcl="SLACK",
        cfg=cfg, log=logs.append))
    return res, cmds, logs


def _ran_partial_ruin(cmds):
    # PARTIAL_RUIN_TCL is the only command emitting the RUIN_CELLS marker.
    return any("RUIN_CELLS" in c for c in cmds)


def test_spread_gate_defaults():
    cfg = ILSPolishConfig()
    assert cfg.partial_ruin_spread_gate is True          # corpus 26/26 -> ON
    assert cfg.partial_ruin_spread_min_tiles == 30.0
    assert cfg.critical_path_avg_spread_tiles is None    # unmeasured default


def test_spread_gate_fires_on_known_low_spread():
    # optical-flow class measures ~15 tiles -> PARTIAL_RUIN never executes,
    # every OTHER combo still runs (rotation continues around the skip).
    res, cmds, logs = _run_spread(15.2)
    assert not _ran_partial_ruin(cmds)
    assert res.cycles == len(ILS_COMBOS) - 1     # all non-gated combos ran
    assert any("place_design -unplace" in c for c in cmds)  # full-ruin ran
    assert any("rotation exhausted" in n for n in res.notes)
    assert any("partial-ruin spread-gated" in n for n in res.notes)
    assert any("partial-ruin skipped: spread=15.2 < 30 "
               "(K3 corpus 26/26 negative" in l for l in logs)


def test_spread_gate_fails_closed_when_spread_unknown():
    """spread=None FAILS CLOSED for partial-ruin (aug05 gray-areas panel).

    Was `test_spread_gate_inert_when_spread_unknown`, asserting the opposite.
    The aug05 panel (qwen H4b, narrowed) deliberately inverted this: an
    unmeasured spread is indistinguishable from the co-located class, which is
    26/26 negative in the corpus (mean -1.10 ns), against a foregone upside of
    +0.068 ns mean. See the K3 SPREAD GATE block in optimizer/ils_polish.py.
    The ENDHIGH gate's None handling is deliberately NOT symmetric.
    """
    res, cmds, logs = _run_spread(None)
    assert not _ran_partial_ruin(cmds), (
        "spread=None must gate partial-ruin, not run it")
    assert any("UNMEASURED" in l for l in logs)
    assert any("None fails closed" in l for l in logs)


def test_spread_gate_inert_when_spread_is_high():
    """The gate is a LOW-spread rule: a comfortably high spread must not gate."""
    res, cmds, _ = _run_spread(120.0)
    assert _ran_partial_ruin(cmds)
    assert res.cycles == len(ILS_COMBOS)
    assert not any("spread-gated" in n for n in res.notes)


def test_spread_gate_inert_on_high_spread():
    # 302 tiles (vexriscv-class): surgery corpus-positive -> gate must not fire.
    res, cmds, _ = _run_spread(302.0)
    assert _ran_partial_ruin(cmds)
    assert res.cycles == len(ILS_COMBOS)


def test_spread_gate_kill_switch():
    res, cmds, _ = _run_spread(15.2, gate=False)
    assert _ran_partial_ruin(cmds)
    assert res.cycles == len(ILS_COMBOS)
    assert not any("spread-gated" in n for n in res.notes)


def test_spread_gate_boundary_is_strictly_below():
    # spread == threshold is NOT below -> runs; just under -> skipped.
    _, cmds_at, _ = _run_spread(30.0)
    assert _ran_partial_ruin(cmds_at)
    _, cmds_under, _ = _run_spread(29.9)
    assert not _ran_partial_ruin(cmds_under)


# ---- __ROUTE_REROLL__ near-met route lottery re-roll (jul21 plateau probe) ----
# Evidence (final_round/plateau_probe_jul21.md + drill_local_queue_jul21.md):
# full `route_design -unroute` + `route_design -directive AggressiveExplore`
# on the banked BEST fir state (near-met -0.195, route-share-dominated)
# gained +0.070ns ~ +9.5MHz with hold IMPROVING (+0.044); 429s local ~ ~170s
# eval. Deep-WNS states are owned by the bare-reroute tail loop (queue:
# -10.676 state gained only +0.010) -> picker gate is near-met only.

def _rr_calltool(cmds, *, hold="0.05", slack="-2.0"):
    async def fake(tool, args):
        cmd = args.get("command", "")
        cmds.append(cmd)
        if "get_cells" in cmd and "llength" in cmd:
            return "100000"
        if "-hold" in cmd:
            return hold
        if "report_route_status" in cmd:
            return ("# of routable nets : 10\n# of fully routed nets : 10\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return slack
        return "ok"
    return fake


def _rr_idx():
    from optimizer.ils_polish import ROUTE_REROLL_PD
    return [c[0] for c in ILS_COMBOS].index(ROUTE_REROLL_PD)


def _run_rr(*, baseline, window_s=1000.0, max_cycles=1, offset=None,
            enabled=True, route_anchor=0.0, hold="0.05", slack="-2.0",
            spread=None, spread_gate=True, wns_mag=None):
    import time as _t
    cmds: list = []
    logs: list = []
    kw = {}
    if wns_mag is not None:
        kw["route_reroll_max_wns_mag"] = wns_mag
    cfg = ILSPolishConfig(enabled=True, min_cycle_seconds=0.0,
                          max_cycles=max_cycles, final_seed_no_improve_stop=0,
                          route_reroll_enabled=enabled,
                          route_cost_anchor_s=route_anchor,
                          partial_ruin_spread_gate=spread_gate,
                          critical_path_avg_spread_tiles=spread, **kw)
    res = asyncio.run(run_ils_polish(
        _rr_calltool(cmds, hold=hold, slack=slack),
        best_dcp_path="/tmp/x.dcp", baseline_wns=baseline,
        deadline_ts=_t.time() + window_s, wns_tcl="SLACK", cfg=cfg,
        log=logs.append,
        combo_offset=_rr_idx() if offset is None else offset))
    return res, cmds, logs


def _rr_fired(logs):
    # the standard per-cycle log line names the sentinel place directive.
    return any("place=__ROUTE_REROLL__" in l for l in logs)


def test_route_reroll_defaults():
    cfg = ILSPolishConfig()
    assert cfg.route_reroll_enabled is True          # probe-validated -> ON
    assert cfg.route_reroll_max_wns_mag == 0.7       # probe jul22: bite only shallow; optical -0.842 measured negative
    assert cfg.route_cost_anchor_s == 0.0            # unknown until plumbed


def test_route_reroll_cost_basis_ladder():
    from optimizer.ils_polish import (route_reroll_cost_basis,
                                      ROUTE_REROLL_COST_MARGIN)
    assert ROUTE_REROLL_COST_MARGIN == 1.3
    # 1) dedicated route anchor wins.
    cfg = ILSPolishConfig(route_cost_anchor_s=300.0,
                          fanout_cost_anchor_s=500.0,
                          fanout_anchor_has_route=True)
    assert route_reroll_cost_basis(cfg) == 300.0 * 1.3
    # 2) fanout anchor WITH route sample is the fallback (conservative:
    # overstates by one phys_opt pass).
    cfg = ILSPolishConfig(fanout_cost_anchor_s=221.0,
                          fanout_anchor_has_route=True)
    assert route_reroll_cost_basis(cfg) == 221.0 * 1.3
    # 3) no route sample anywhere -> unknown (0), picker uses the prior.
    cfg = ILSPolishConfig(fanout_cost_anchor_s=221.0,
                          fanout_anchor_has_route=False)
    assert route_reroll_cost_basis(cfg) == 0.0


def test_route_reroll_dispatch_exactly_two_ops():
    """The re-roll cycle is EXACTLY the probe protocol: unroute + one
    from-scratch AggressiveExplore route. No place step in any form, and —
    unlike ROUTE_ONLY — no phys_opt tail (the x1.3 route basis covers two
    ops only)."""
    res, cmds, logs = _run_rr(baseline=-0.195)   # the fir probe state
    assert _rr_fired(logs)
    joined = "\n".join(cmds)
    assert "route_design -unroute" in joined
    assert "route_design -directive AggressiveExplore" in joined
    assert "place_design" not in joined          # placement untouched
    assert "phys_opt_design" not in joined       # two-op protocol, no tail
    assert "unplace_cell" not in joined


def test_route_reroll_fires_near_met_budget_fit():
    """Both gate legs green (wns -0.195 >= -1.5; need 300*1.3=390 < 1000s
    remaining) -> the combo fires."""
    res, cmds, logs = _run_rr(baseline=-0.195, route_anchor=300.0,
                              window_s=1000.0)
    assert _rr_fired(logs)
    assert res.cycles == 1


def test_route_reroll_deep_wns_skip_logged():
    """boom-shape wns -10 -> the re-roll is SKIPPED with the evidence-cited
    reason and the rotation moves on (next combo runs in the same pick, no
    cycle consumed by the gate)."""
    res, cmds, logs = _run_rr(baseline=-10.0, max_cycles=1)
    assert not _rr_fired(logs)
    assert any("route-reroll skipped" in l and "near-met floor" in l
               for l in logs)
    # rotation continued: the pick fell through to ROUTE_ONLY (next index),
    # which DID run an unroute + re-route WITH its phys_opt tail.
    assert res.cycles == 1
    assert any("place=__ROUTE_ONLY__" in l for l in logs)
    assert "phys_opt_design" in "\n".join(cmds)


def test_route_reroll_near_met_boundary():
    """Gate is wns >= -route_reroll_max_wns_mag: exactly -0.7 is ELIGIBLE,
    just below is skipped (deep-WNS)."""
    _, _, logs_at = _run_rr(baseline=-0.7)
    assert _rr_fired(logs_at)
    _, _, logs_under = _run_rr(baseline=-0.701)
    assert not _rr_fired(logs_under)


def test_route_reroll_unknown_wns_skips():
    """baseline None (unmeasured) -> near-met is NOT demonstrated -> skip
    (conservative: the mechanism's evidence is near-met only)."""
    _, _, logs = _run_rr(baseline=None)
    assert not _rr_fired(logs)
    assert any("route-reroll skipped" in l for l in logs)


def test_route_reroll_budget_unfit_skip():
    """Full-route need (800*1.3=1040s) >= remaining (~900s) -> skipped with
    the no-K=2-doubling reasoning logged; the run notes it once."""
    res, cmds, logs = _run_rr(baseline=-0.195, route_anchor=800.0,
                              window_s=900.0, max_cycles=1)
    assert not _rr_fired(logs)
    assert any("route-reroll skipped" in l and "no K=2 doubling" in l
               for l in logs)
    assert any("route-reroll budget-gated" in n for n in res.notes)


def test_route_reroll_kill_switch():
    """route_reroll_enabled=False -> never fires even on the probe state;
    the rest of the rotation is untouched (next combo still runs)."""
    res, cmds, logs = _run_rr(baseline=-0.195, enabled=False, max_cycles=1)
    assert not _rr_fired(logs)
    assert res.cycles == 1
    assert any("place=__ROUTE_ONLY__" in l for l in logs)


def test_route_reroll_hold_gate_applies():
    """The GENERIC combo accept path (accept_requires_hold_clean, floor
    -0.001) must gate the re-roll like every other combo: setup improves
    but hold is dirty -> REJECTED, nothing persisted. (The fir probe showed
    hold IMPROVING, so no special-casing — but the gate must still hold.)"""
    res, cmds, logs = _run_rr(baseline=-0.5, slack="-0.1", hold="-0.12")
    assert _rr_fired(logs)
    assert res.accepted == 0 and res.best_wns == -0.5
    assert any("hold-dirty" in n for n in res.notes)
    assert not any(c.startswith("write_checkpoint") for c in cmds)


def test_route_reroll_accepts_and_banks_on_clean_hold():
    """Happy path on the probe shape: setup gain + clean (improving) hold
    -> accepted and persisted to the banked best (never-worse mirror)."""
    res, cmds, logs = _run_rr(baseline=-0.195, slack="-0.125", hold="0.044")
    assert _rr_fired(logs)
    assert res.accepted == 1 and res.best_wns == -0.125
    assert any(c.startswith("write_checkpoint") for c in cmds)


def test_route_reroll_spread_gate_composition():
    """The K3 spread gate (PARTIAL_RUIN family) and the re-roll gate are
    INDEPENDENT: low spread + near-met -> partial-ruin skipped while the
    re-roll fires; low spread + deep WNS -> both skipped, the rest of the
    rotation still runs."""
    # near-met + co-located path: re-roll fires, partial-ruin never does.
    res, cmds, logs = _run_rr(baseline=-0.5, spread=15.2, offset=0,
                              max_cycles=20)
    assert _rr_fired(logs)
    assert not any("RUIN_CELLS" in c for c in cmds)
    assert any("partial-ruin skipped" in l for l in logs)
    # deep WNS + co-located path: both families skipped, others still run.
    res2, cmds2, logs2 = _run_rr(baseline=-10.0, spread=15.2, offset=0,
                                 max_cycles=20)
    assert not _rr_fired(logs2)
    assert not any("RUIN_CELLS" in c for c in cmds2)
    assert any("route-reroll skipped" in l for l in logs2)
    assert any("partial-ruin skipped" in l for l in logs2)
    assert any("place_design -unplace" in c for c in cmds2)  # full ruin ran


def test_route_reroll_rotation_intact_for_other_combos():
    """Freeze-style: the jul21 insertion must not reorder anything else —
    full expected order, sentinels included."""
    from optimizer.ils_polish import (LASTMILE_PD, ROUTE_ONLY_PD,
                                      ROUTE_REROLL_PD, PARTIAL_RUIN_PD)
    assert [c[0] for c in ILS_COMBOS] == [
        "Explore", LASTMILE_PD, ROUTE_REROLL_PD, ROUTE_ONLY_PD,
        "ExtraTimingOpt", PARTIAL_RUIN_PD, "AltSpreadLogic_high",
        "ExtraNetDelay_high", "SSI_SpreadLogic_high", "EarlyBlockPlacement",
    ]
    # route/phys directives of the pre-existing combos are untouched.
    assert ILS_COMBOS[3] == (ROUTE_ONLY_PD, "AggressiveExplore", "Explore")
    assert ILS_COMBOS[2][1] == "AggressiveExplore"   # the probe's directive


# ---- PLACE-RETRY extension (jul27, opt-in FPL26_ILS_PLACE_RETRY) ----
#
# spam ships +0.00 at parity (VALID_FALLBACK_BASELINE whose artifact md5 equals the
# INPUT md5) because ILS_COMBOS[0]=Explore REGRESSES it (-0.688 vs -0.686 baseline)
# and futility stops the search with budget unspent. AltSpreadLogic_medium reaches
# -0.598 (+17.51) at the same cycle cost and is absent from the rotation.

def test_place_retry_default_off_is_byte_identical():
    """Unarmed the rotation must be EXACTLY the shipped one -- same list, same
    length, same indices -- so rotation-exhaustion accounting is untouched."""
    from optimizer.ils_polish import active_combos, place_retry_enabled
    assert place_retry_enabled() is False
    assert active_combos() == list(ILS_COMBOS)


def test_place_retry_armed_appends_without_touching_head(monkeypatch):
    from optimizer.ils_polish import active_combos, PLACE_RETRY_COMBOS
    monkeypatch.setenv("FPL26_ILS_PLACE_RETRY", "1")
    act = active_combos()
    assert len(act) == len(ILS_COMBOS) + len(PLACE_RETRY_COMBOS)
    assert [c[0] for c in act[:4]] == [c[0] for c in ILS_COMBOS[:4]]
    assert [c[0] for c in act[len(ILS_COMBOS):]] == [c[0] for c in PLACE_RETRY_COMBOS]
    assert len({c[0] for c in act}) == len(act)   # no duplicate directives


def _capture_placed(monkeypatch, armed, wns="-0.95", baseline=-0.9, cycles=3):
    import time as _t
    if armed:
        monkeypatch.setenv("FPL26_ILS_PLACE_RETRY", "1")
    placed = []
    async def fake(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "0.05"
        if cmd.startswith("place_design -directive"):
            placed.append(cmd.split()[-1])
        if "report_route_status" in cmd:
            return ("# of routable nets : 100\n# of fully routed nets : 100\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return wns          # always WORSE than baseline -> every cycle regresses
        return "ok"
    cfg = ILSPolishConfig(enabled=True, min_cycle_seconds=0.0, max_cycles=cycles,
                          final_seed_no_improve_stop=0, lastmile_min_wns_ns=-99.0)
    res = asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=baseline,
        deadline_ts=_t.time() + 200.0, wns_tcl="SLACK", cfg=cfg, log=lambda m: None))
    return placed, res


def test_place_retry_fires_on_regression(monkeypatch):
    """Cycle 1 regresses -> the NEXT cycle uses the measured-better directive.
    Without the jump the extension sits at the end of the rotation and futility
    stops the search before ever reaching it."""
    from optimizer.ils_polish import PLACE_RETRY_TARGET
    placed, _ = _capture_placed(monkeypatch, armed=True)
    assert placed[0] == ILS_COMBOS[0][0] == "Explore"
    assert PLACE_RETRY_TARGET in placed, placed
    assert placed.index(PLACE_RETRY_TARGET) == 1, placed


def test_place_retry_does_not_fire_when_disarmed(monkeypatch):
    from optimizer.ils_polish import PLACE_RETRY_TARGET
    placed, _ = _capture_placed(monkeypatch, armed=False)
    # disarmed, the forced jump must not happen: the target may still appear via the
    # NORMAL rotation, so assert it is not reached EARLY (index 1 is the forced slot).
    assert PLACE_RETRY_TARGET not in placed[:2], placed


def test_place_retry_does_not_corrupt_pristine_rot(monkeypatch):
    """REGRESSION TEST for a real defect found while building this.

    The first implementation assigned rot directly. That advanced pristine_rot past
    combos that were never tried -- and the corrective sibling seed starts its
    rotation FROM pristine_rot to skip genuine replays, so it would have silently
    skipped up to eight untried combos. The retry must be a one-off forced pick
    that never advances rotation state.
    """
    placed, res = _capture_placed(monkeypatch, armed=True)
    assert res.cycles == 3
    assert res.pristine_rot < len(ILS_COMBOS), (res.pristine_rot, placed)
    assert res.pristine_rot <= res.cycles


# ---- MEASURED COST BASIS (jul28) -------------------------------------------
# Replays optical-flow's LIVE numbers from gate_ab_jul27 (agent.log):
#   cold-start anchor 672s | ILS window 2110s | cycle 1 Explore dt=225s
#   baseline -0.924 | cycle-1 result -1.162 (a REGRESSION -> place-retry fires)
# Unarmed, ExtraNetDelay_high prices at 3.0 x 672 = 2016s > 1885s remaining and
# the forced pick is REFUSED -- which is why optical never reached the directive
# its own sweep ranks first (+26.48). Armed, the basis becomes the MEASURED
# 225s cycle and the same pick costs 776s, comfortably affordable.
OPTICAL_ANCHOR_S = 672.0
OPTICAL_WINDOW_S = 2110.0
OPTICAL_CYCLE_S = 225.0


class _FakeClock:
    def __init__(self, start=1_000_000.0):
        self.t = start

    def time(self):
        return self.t


def _capture_optical(monkeypatch, measured_basis):
    """Replay optical's cycle 1 with a controlled clock. Returns (placed, logs)."""
    monkeypatch.setenv("FPL26_ILS_PLACE_RETRY", "1")
    if measured_basis:
        monkeypatch.setenv("FPL26_ILS_MEASURED_BASIS", "1")
    else:
        monkeypatch.setenv("FPL26_ILS_MEASURED_BASIS", "0")
    clock = _FakeClock()
    monkeypatch.setattr("optimizer.ils_polish.time.time", clock.time)
    placed, logs = [], []

    async def fake(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "0.05"
        if cmd.startswith("place_design -directive"):
            placed.append(cmd.split()[-1])
            clock.t += OPTICAL_CYCLE_S      # a real full-place cycle's wall
        if "report_route_status" in cmd:
            return ("# of routable nets : 100\n# of fully routed nets : 100\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-1.162"                  # worse than the -0.924 incumbent
        return "ok"

    cfg = ILSPolishConfig(enabled=True, min_cycle_seconds=0.0, max_cycles=2,
                          final_seed_no_improve_stop=0, lastmile_min_wns_ns=-99.0,
                          expected_heavy_cycle_s=OPTICAL_ANCHOR_S)
    asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.924,
        deadline_ts=clock.t + OPTICAL_WINDOW_S, wns_tcl="SLACK", cfg=cfg,
        log=logs.append))
    return placed, logs


def test_optical_replay_cold_anchor_refuses_the_forced_pick(monkeypatch):
    """CONTROL: shipped behaviour. The place-retry trigger fires and is then
    priced out by the recipe-derived anchor -- the live failure mode."""
    from optimizer.ils_polish import PLACE_RETRY_TARGET
    placed, logs = _capture_optical(monkeypatch, measured_basis=False)
    assert placed[0] == "Explore"
    assert PLACE_RETRY_TARGET not in placed, placed
    assert any("unaffordable" in m for m in logs), logs


def test_optical_replay_measured_basis_admits_the_forced_pick(monkeypatch):
    """ARMED: the measured 225s cycle replaces the 672s anchor, so the SAME
    trigger now reaches ExtraNetDelay_high on the very next cycle."""
    from optimizer.ils_polish import PLACE_RETRY_TARGET
    placed, logs = _capture_optical(monkeypatch, measured_basis=True)
    assert placed[0] == "Explore"
    assert placed[1] == PLACE_RETRY_TARGET, placed
    assert not any("unaffordable" in m for m in logs), logs
    assert any("measured cost basis ARMED" in m for m in logs), logs


def test_measured_basis_normalises_by_the_directive_prior():
    """The basis is in Explore=1.0 units, so an expensive directive's wall does
    not inflate it: a 2.0x directive taking 400s implies a 200s Explore cycle."""
    from optimizer.ils_polish import (cold_start_basis, COMBO_COST_PRIOR,
                                      MEASURED_BASIS_MARGIN)
    assert COMBO_COST_PRIOR["AltSpreadLogic_high"] == 2.0

    class _C:
        expected_heavy_cycle_s = OPTICAL_ANCHOR_S
    import os
    os.environ["FPL26_ILS_MEASURED_BASIS"] = "1"
    try:
        assert cold_start_basis(_C, 400.0 / 2.0) == 200.0 * MEASURED_BASIS_MARGIN
        # No measurement yet -> the shipped anchor, unchanged.
        assert cold_start_basis(_C, 0.0) == OPTICAL_ANCHOR_S
    finally:
        del os.environ["FPL26_ILS_MEASURED_BASIS"]


def test_measured_basis_default_off_is_byte_identical():
    """Unarmed, a measurement can never change what an unseen combo costs."""
    from optimizer.ils_polish import cold_start_basis

    class _C:
        expected_heavy_cycle_s = OPTICAL_ANCHOR_S
    assert cold_start_basis(_C, 225.0) == OPTICAL_ANCHOR_S
    assert cold_start_basis(_C, 0.0) == OPTICAL_ANCHOR_S


# ---- PROBE LADDER + INCREMENTAL RE-ROUTE (jul28) ---------------------------

def _capture_cmds(monkeypatch, env, wns_seq, cycles=5, baseline=-0.9):
    """Run the loop with a scripted WNS sequence; capture every issued command."""
    import time as _t
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    cmds, placed, seq = [], [], list(wns_seq)
    state = {"i": 0}

    async def fake(tool, args):
        cmd = args.get("command", "")
        cmds.append(cmd)
        if "-hold" in cmd:
            return "0.05"
        if cmd.startswith("place_design -directive"):
            placed.append(cmd.split()[-1])
        if "report_route_status" in cmd:
            return ("# of routable nets : 100\n# of fully routed nets : 100\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            i = min(state["i"], len(seq) - 1)
            state["i"] += 1
            return seq[i]
        return "ok"

    cfg = ILSPolishConfig(enabled=True, min_cycle_seconds=0.0, max_cycles=cycles,
                          final_seed_no_improve_stop=0, lastmile_min_wns_ns=-99.0)
    res = asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=baseline,
        deadline_ts=_t.time() + 400.0, wns_tcl="SLACK", cfg=cfg, log=lambda m: None))
    return cmds, placed, res


def test_ladder_first_rung_is_exactly_the_shipped_target():
    """STRICT ADDITIVITY: arming the ladder cannot change what fires first."""
    from optimizer.ils_polish import PLACE_RETRY_LADDER, PLACE_RETRY_TARGET
    assert PLACE_RETRY_LADDER[0] == PLACE_RETRY_TARGET


def test_ladder_probes_every_rung_in_evidence_order(monkeypatch):
    """Each rung is a separate forced cycle, in the declared order.

    spam's best directive (AltSpreadLogic_medium, +17.51) is rung 2; it is
    unreachable with the single-rung retry because futility stops the search.
    """
    from optimizer.ils_polish import PLACE_RETRY_LADDER
    _, placed, _ = _capture_cmds(
        monkeypatch,
        {"FPL26_ILS_PLACE_RETRY_LADDER": "1"},
        ["-0.95"] * 12, cycles=4)
    assert placed[0] == "Explore"
    assert placed[1:4] == PLACE_RETRY_LADDER, placed


def test_ladder_disarmed_still_fires_only_the_single_rung(monkeypatch):
    from optimizer.ils_polish import PLACE_RETRY_TARGET
    monkeypatch.setenv("FPL26_ILS_PLACE_RETRY_LADDER", "0")
    _, placed, _ = _capture_cmds(
        monkeypatch, {"FPL26_ILS_PLACE_RETRY": "1"}, ["-0.95"] * 12, cycles=4)
    assert placed[1] == PLACE_RETRY_TARGET
    # rung 2 must NOT be forced when only the single-rung retry is armed
    assert placed[2] != "AltSpreadLogic_medium", placed


def test_incr_route_absent_from_rotation_when_disarmed(monkeypatch):
    """jul29: renamed from '..._unless_armed'. INCR_ROUTE is now DEFAULT ON, so the
    property under test is that =0 restores the shipped rotation exactly."""
    from optimizer.ils_polish import active_combos, INCR_ROUTE_PD, ILS_COMBOS
    monkeypatch.setenv("FPL26_ILS_INCR_ROUTE", "0")
    monkeypatch.delenv("FPL26_ILS_PLACE_RETRY", raising=False)
    assert active_combos() == list(ILS_COMBOS)
    monkeypatch.setenv("FPL26_ILS_INCR_ROUTE", "1")
    assert active_combos()[-1][0] == INCR_ROUTE_PD
    assert active_combos()[:len(ILS_COMBOS)] == list(ILS_COMBOS)


def test_incr_route_never_unroutes(monkeypatch):
    """THE mechanism. ROUTE_ONLY/ROUTE_REROLL both `route_design -unroute` first;
    doing that here turns S3 (-0.543) back into S2 (-0.595). An improving first
    cycle sets the incumbent's router to Explore, which makes the rung eligible."""
    from optimizer.ils_polish import INCR_ROUTE_PD
    import time as _t
    monkeypatch.setenv("FPL26_ILS_INCR_ROUTE", "1")
    trace, seq, state = [], ["-0.80", "-0.80"] + ["-0.85"] * 12, {"i": 0}

    async def fake(tool, args):
        cmd = args.get("command", "")
        trace.append(("cmd", cmd))
        if "-hold" in cmd:
            return "0.05"
        if "report_route_status" in cmd:
            return ("# of routable nets : 100\n# of fully routed nets : 100\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            i = min(state["i"], len(seq) - 1)
            state["i"] += 1
            return seq[i]
        return "ok"

    cfg = ILSPolishConfig(enabled=True, min_cycle_seconds=0.0, max_cycles=12,
                          final_seed_no_improve_stop=0, lastmile_min_wns_ns=-99.0)
    asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.9,
        deadline_ts=_t.time() + 400.0, wns_tcl="SLACK", cfg=cfg,
        log=lambda m: trace.append(("log", m))))

    # Locate the cycle the rung actually ran, then read the commands AFTER that
    # marker up to the next cycle marker. Without this the assertion could pass on
    # an AggressiveExplore issued by ExtraTimingOpt, which also routes with it.
    starts = [i for i, (k, v) in enumerate(trace)
              if k == "log" and f"place={INCR_ROUTE_PD}" in v]
    assert starts, [v for k, v in trace if k == "log"]
    # The cycle's commands are the ones BEFORE its summary log line, back to the
    # previous cycle's summary.
    prev = max([i for i, (k, v) in enumerate(trace)
                if k == "log" and "ILS cycle " in v and i < starts[0]] or [-1])
    window = [v for k, v in trace[prev + 1:starts[0]] if k == "cmd"]
    assert "route_design -directive AggressiveExplore" in window, window
    assert "route_design -unroute" not in window, window
    assert not any(c.startswith("place_design") for c in window), window


# ---- RETRY BASELINE GATE (jul28) -------------------------------------------
# Each row is (design, pristine baseline WNS, incumbent WNS, cycle-1 Explore WNS,
# should the trigger fire). Baselines are read from each production log's own
# "Initial Fmax: ... (WNS: ...)" line; spam and 3d come from their record docs.
# The point of the gate: fire on the three designs whose records need it, stay
# inert on designs where the RECIPE is good and Explore merely trails it.
BASELINE_GATE_CORPUS = [
    ("spam",        -0.686,  -0.686,  -0.688,  True),
    ("3d",          -2.153,  -2.153,  -2.278,  True),
    ("optical",     -1.078,  -0.924,  -1.162,  True),
    ("logicnets",   -0.978,  -0.526,  -1.041,  True),    # known false positive
    ("mini-isp",    -1.686,  -0.956,  -0.996,  False),
    ("corescore",   -1.238,  -0.680,  -0.683,  False),
    ("finn",        -1.910,  -1.256,  -1.288,  False),
    ("vexriscv",    -1.654,  -0.619,  -0.635,  False),
    ("vtr",        -14.527, -14.316, -11.587,  False),
]


def _fires(monkeypatch, baseline, incumbent, cycle1, gate_armed):
    """Return True if the place-retry trigger fires for these measured numbers."""
    import time as _t
    from optimizer.ils_polish import PLACE_RETRY_TARGET
    monkeypatch.setenv("FPL26_ILS_PLACE_RETRY", "1")
    if gate_armed:
        monkeypatch.setenv("FPL26_ILS_RETRY_BASELINE_GATE", "1")
    else:
        monkeypatch.delenv("FPL26_ILS_RETRY_BASELINE_GATE", raising=False)
    placed = []

    async def fake(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "0.05"
        if cmd.startswith("place_design -directive"):
            placed.append(cmd.split()[-1])
        if "report_route_status" in cmd:
            return ("# of routable nets : 100\n# of fully routed nets : 100\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return str(cycle1)
        return "ok"

    cfg = ILSPolishConfig(enabled=True, min_cycle_seconds=0.0, max_cycles=2,
                          final_seed_no_improve_stop=0, lastmile_min_wns_ns=-99.0,
                          design_baseline_wns=baseline)
    asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=incumbent,
        deadline_ts=_t.time() + 400.0, wns_tcl="SLACK", cfg=cfg, log=lambda m: None))
    return PLACE_RETRY_TARGET in placed


@pytest.mark.parametrize("name,base,inc,c1,want", BASELINE_GATE_CORPUS)
def test_baseline_gate_matches_the_measured_corpus(monkeypatch, name, base, inc, c1, want):
    got = _fires(monkeypatch, base, inc, c1, gate_armed=True)
    assert got is want, f"{name}: gate fired={got}, expected {want}"


@pytest.mark.parametrize("name,base,inc,c1,want", BASELINE_GATE_CORPUS)
def test_without_the_gate_the_trigger_fires_on_nearly_everything(
        monkeypatch, name, base, inc, c1, want):
    """The gate's reason to exist: unarmed, the incumbent-relative trigger fires on
    every design whose cycle 1 trails the incumbent — 8 of the 9 rows, including
    the four strongest designs on the board."""
    got = _fires(monkeypatch, base, inc, c1, gate_armed=False)
    assert got is (c1 <= inc), f"{name}: unarmed trigger should track incumbent only"


# ---- PANEL REFINEMENTS (jul28): ladder reserve + incr-route terminal --------

def _run_loop(monkeypatch, env, wns="-0.95", cycles=8, window=400.0,
              no_improve_stop=0, anchor=0.0, spread=None, failing=None,
              wrapper_window=None):
    """Minimal driver returning (placed, logs) with explicit budget/futility.

    spread/failing default to None so every pre-existing caller keeps the
    unmeasured -> fail-open behaviour it was written against.
    """
    import time as _t
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    placed, logs = [], []

    async def fake(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "0.05"
        if cmd.startswith("place_design -directive"):
            placed.append(cmd.split()[-1])
        if "report_route_status" in cmd:
            return ("# of routable nets : 100\n# of fully routed nets : 100\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return wns
        return "ok"

    cfg = ILSPolishConfig(enabled=True, min_cycle_seconds=0.0, max_cycles=cycles,
                          final_seed_no_improve_stop=0, lastmile_min_wns_ns=-99.0,
                          expected_heavy_cycle_s=anchor,
                          critical_path_avg_spread_tiles=spread,
                          phase1_failing_endpoints=failing)
    # The UNFENCED wrapper budget. Left None by default so every pre-existing
    # caller keeps the fail-open behaviour it was written against.
    if wrapper_window is not None:
        cfg.wrapper_deadline_ts = _t.time() + wrapper_window
    asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.9,
        deadline_ts=_t.time() + window, wns_tcl="SLACK", cfg=cfg,
        log=logs.append, no_improve_stop=no_improve_stop))
    return placed, logs


def test_ladder_reserve_stops_the_ladder_when_the_window_is_tight(monkeypatch):
    """A rung past the first must leave room for one more full cycle afterwards.
    With a 250s basis and a 400s window, rung 2 costs 1.05x250=262s and cannot
    also reserve 250s, so the LADDER stops there.

    Note what this does NOT claim: AltSpreadLogic_medium stays in the rotation
    (arming the ladder appends it), so it can still be reached later on its own
    merits. The reserve governs the FORCED probe, not the combo's existence.
    """
    _, logs_on = _run_loop(
        monkeypatch,
        {"FPL26_ILS_PLACE_RETRY_LADDER": "1", "FPL26_ILS_LADDER_RESERVE": "1"},
        window=400.0, anchor=250.0)
    assert any("reserve-gated" in m for m in logs_on), logs_on
    # Same window, reserve disarmed: nothing is reserve-gated, so the gate is
    # demonstrably what stopped the ladder rather than the budget alone.
    _, logs_off = _run_loop(
        monkeypatch,
        {"FPL26_ILS_PLACE_RETRY_LADDER": "1", "FPL26_ILS_LADDER_RESERVE": "0"},
        window=400.0, anchor=250.0)
    assert not any("reserve-gated" in m for m in logs_off), logs_off


def test_ladder_reserve_never_gates_rung_one(monkeypatch):
    """Rung 1 IS the shipped single-rung behaviour and must stay unconditional.
    It may still be refused on plain affordability — but never by the reserve."""
    _, logs = _run_loop(
        monkeypatch,
        {"FPL26_ILS_PLACE_RETRY_LADDER": "1", "FPL26_ILS_LADDER_RESERVE": "1"},
        window=400.0, anchor=250.0)
    assert not any("rung 1" in m and "reserve-gated" in m for m in logs), logs


def test_incr_route_terminal_defers_while_the_loop_is_productive(monkeypatch):
    """Gated, the escalation must not compete with full-ruin cycles that have
    corpus-wide evidence behind them — it is the LAST cheap move."""
    _, logs = _run_loop(
        monkeypatch,
        {"FPL26_ILS_INCR_ROUTE": "1", "FPL26_ILS_INCR_ROUTE_TERMINAL": "1"},
        wns="-0.80", window=4000.0, anchor=100.0, no_improve_stop=0, cycles=24)
    assert any("incr-route deferred" in m for m in logs), logs


def test_incr_route_terminal_allows_it_when_no_full_cycle_fits(monkeypatch):
    """The other half of the terminal condition: when the window can no longer
    pay for a full-place cycle, the escalation is exactly the move to make, so
    the gate must NOT defer it."""
    _, logs = _run_loop(
        monkeypatch,
        {"FPL26_ILS_INCR_ROUTE": "1", "FPL26_ILS_INCR_ROUTE_TERMINAL": "1"},
        wns="-0.80", window=400.0, anchor=500.0, no_improve_stop=0, cycles=24)
    assert not any("incr-route deferred" in m for m in logs), logs


def test_panel_refinements_are_default_off(monkeypatch):
    """Neither refinement may change anything unless explicitly armed."""
    from optimizer.ils_polish import (ladder_reserve_enabled,
                                      incr_route_terminal_enabled)
    monkeypatch.delenv("FPL26_ILS_LADDER_RESERVE", raising=False)
    monkeypatch.delenv("FPL26_ILS_INCR_ROUTE_TERMINAL", raising=False)
    assert ladder_reserve_enabled() is False
    assert incr_route_terminal_enabled() is False


# ---- SKIP-UNAFFORDABLE (jul31, digit) --------------------------------------
# An unaffordable rung costs a whole cycle and hands it to generic rotation.
# On digit that draw decided a 32 MHz spread: __LASTMILE__ (91s) kept the ladder
# alive to ExtraNetDelay_low and +72.59; __ROUTE_ONLY__ (845s) ended it at +40.36.
#
# Window/anchor arithmetic used below: anchor 250s, window 400s.
#   rung 1 ExtraNetDelay_high    prior 2.415 (measured) / 3.447 (legacy)
#                                -> est >= 604s  > 400s remaining  UNAFFORDABLE
#   rung 2 AltSpreadLogic_medium prior 1.05  -> est  262s  < 400s   AFFORDABLE
# so rung 1 always refuses and rung 2 is always reachable, under either prior.

def test_skip_unaffordable_is_default_on_with_kill_switch(monkeypatch):
    """aug02: PROMOTED to default-ON. Pins the NEW contract.

    ⚠️ RECORD, verbatim, because this promotion is unusual: the never-worse gate
    on corescore FAILED as pre-registered (treatment below control in 2 of 3
    fired pairs, gaps 0.73 and 3.09 MHz). It was shipped anyway as an EXPLICIT
    RISK-ACCEPTED DECISION BY THE USER, on the grounds that (a) both gaps are at
    or under the measured 3.5 MHz noise floor, (b) the treatment values
    {83.12, 83.12, 80.03} sit entirely inside corescore's historical flag-OFF
    distribution — v2.0's own sweep drew 80.03 with no flag — so the control
    column being lucky explains the pairs, and (c) the digit benefit removes a
    17-28% chance of a -32 MHz collapse (EV ~ +5.4 MHz). This is a documented
    trade, not a passed gate. finn was 6/6 identical at its MAX.
    """
    from optimizer.ils_polish import ladder_skip_unaffordable_enabled
    monkeypatch.delenv("FPL26_ILS_LADDER_SKIP_UNAFFORDABLE", raising=False)
    assert ladder_skip_unaffordable_enabled() is True
    monkeypatch.setenv("FPL26_ILS_LADDER_SKIP_UNAFFORDABLE", "0")
    assert ladder_skip_unaffordable_enabled() is False, "=0 must stay a real kill switch"


# ---- ExtraNetDelay_high DENSITY GATE (jul31) --------------------------------
# Corpus densities (failing/spread), constant per design across 101 runs:
#   ARMED   3d 1314   spam 808   optical 518        (accept 100% / 59% / 94%)
#   BLOCKED vexriscv_v2 255  digit 175  mini-isp 86  vexriscv 36  logicnets 14
# Threshold 363 = geometric midpoint of the 255->518 gap.

def _cfg_with(spread, failing):
    return ILSPolishConfig(enabled=True,
                           critical_path_avg_spread_tiles=spread,
                           phase1_failing_endpoints=failing)


def test_endhigh_density_gate_is_default_on_with_a_kill_switch(monkeypatch):
    """DEFAULT ON since jul31 (digit +72.59 and optical +32.38 on one build).
    The kill switch must still fully disarm it."""
    from optimizer.ils_polish import (endhigh_density_gate_enabled,
                                      endhigh_density_blocked)
    monkeypatch.delenv("FPL26_ILS_ENDHIGH_DENSITY_GATE", raising=False)
    assert endhigh_density_gate_enabled() is True
    # default-on: a low-density design making a big bet IS blocked
    assert endhigh_density_blocked(_cfg_with(332.3, 252), 900.0, 1000.0)[0] is True
    # kill switch: nothing blocks, whatever the features say
    monkeypatch.setenv("FPL26_ILS_ENDHIGH_DENSITY_GATE", "0")
    assert endhigh_density_gate_enabled() is False
    assert endhigh_density_blocked(_cfg_with(332.3, 252), 900.0, 1000.0)[0] is False


def test_endhigh_density_arms_the_winners_and_blocks_the_losers(monkeypatch):
    """The whole point: optical must stay ARMED (it is worth 17.7 measured eval
    points) while digit/fir/logicnets get blocked."""
    from optimizer.ils_polish import endhigh_density_blocked
    monkeypatch.setenv("FPL26_ILS_ENDHIGH_DENSITY_GATE", "1")
    monkeypatch.delenv("FPL26_ILS_ENDHIGH_DENSITY_MIN", raising=False)
    armed = {"optical": (15.2, 7866), "spam": (8.4, 6786), "3d": (65.8, 86488)}
    blocked = {"digit": (131.4, 22946), "fir": (332.3, 252),
               "logicnets": (111.9, 1529), "mini_isp": (56.9, 4887),
               "vexriscv": (53.5, 1937), "vexriscv_v2": (11.5, 2933)}
    # BIG BET (share 0.9): only the low-density designs may block.
    for name, (sp, fa) in armed.items():
        blk, why = endhigh_density_blocked(_cfg_with(sp, fa), 900.0, 1000.0)
        assert blk is False, f"{name} must stay ARMED even on a big bet: {why}"
    for name, (sp, fa) in blocked.items():
        blk, why = endhigh_density_blocked(_cfg_with(sp, fa), 900.0, 1000.0)
        assert blk is True, f"{name} must be BLOCKED on a big bet: {why}"
    # SMALL BET (share 0.2): NOTHING blocks -- this is what saves vexriscv's
    # 4 accepts (share 0.15-0.21) and logicnets' 2 (0.51-0.52).
    for name, (sp, fa) in {**armed, **blocked}.items():
        blk, why = endhigh_density_blocked(_cfg_with(sp, fa), 200.0, 1000.0)
        assert blk is False, f"{name} must ARM on a small bet: {why}"


def test_endhigh_density_fails_open_on_every_unknown(monkeypatch):
    """THE GUARD THAT MUST STILL FAIL.

    Blocking ExtraNetDelay_high because a RapidWright spread analysis was
    skipped would cost optical 17.7 points to avoid a 1300s waste. Every
    uncertainty must therefore ARM, never block."""
    from optimizer.ils_polish import endhigh_density_blocked
    monkeypatch.setenv("FPL26_ILS_ENDHIGH_DENSITY_GATE", "1")
    nan, inf = float("nan"), float("inf")
    for cfg in (_cfg_with(None, 7866),      # spread unmeasured
                _cfg_with(15.2, None),      # failing unmeasured
                _cfg_with(None, None),      # neither
                _cfg_with(0.0, 7866),       # zero spread (would divide by zero)
                _cfg_with(-3.0, 7866),      # nonsense spread
                _cfg_with("x", 7866),       # unparseable
                # NaN/inf regression guard (gpt-5.6-sol, panel_endhigh_jul31.md):
                # EVERY comparison against NaN is False, so `spread <= 0` does
                # not catch it and it falls through to `dens >= thr` -> False ->
                # BLOCKED. An infinite spread gives density exactly 0.0, which
                # compares cleanly and also blocks. Both would block an
                # optical-like design while claiming to fail open.
                _cfg_with(nan, 7866),
                _cfg_with(15.2, nan),
                _cfg_with(inf, 7866),
                _cfg_with(15.2, inf)):
        blk, why = endhigh_density_blocked(cfg, 900.0, 1000.0)
        assert blk is False, f"must fail OPEN on {cfg}: {why}"


def test_endhigh_density_threshold_override_rejects_garbage(monkeypatch):
    """A broken override must fall back to the default, never silently disable
    the gate (jul30: a probe that returns nothing is not a pass)."""
    from optimizer.ils_polish import (endhigh_density_min, endhigh_density_blocked,
                                      ENDHIGH_DENSITY_MIN_DEFAULT)
    monkeypatch.setenv("FPL26_ILS_ENDHIGH_DENSITY_GATE", "1")
    for bad in ("not-a-number", "", "0", "-5"):
        monkeypatch.setenv("FPL26_ILS_ENDHIGH_DENSITY_MIN", bad)
        assert endhigh_density_min() == ENDHIGH_DENSITY_MIN_DEFAULT
        # and digit is still blocked rather than let through by the bad value
        assert endhigh_density_blocked(_cfg_with(131.4, 22946), 900.0, 1000.0)[0] is True
    monkeypatch.setenv("FPL26_ILS_ENDHIGH_DENSITY_MIN", "100")
    assert endhigh_density_min() == 100.0
    # at 100, digit (175) now ARMS -- proving the threshold is really consulted
    assert endhigh_density_blocked(_cfg_with(131.4, 22946), 900.0, 1000.0)[0] is False


def test_endhigh_density_blocks_the_rotation_path_not_just_the_ladder(monkeypatch):
    """ExtraNetDelay_high lives in the BASE rotation (ILS_COMBOS), so gating only
    the place-retry ladder would leave the path it actually reached digit by wide
    open. With the gate armed on a low-density design it must never be placed."""
    placed, logs = _run_loop(
        monkeypatch,
        {"FPL26_ILS_PLACE_RETRY_LADDER": "0",     # ladder OFF: rotation path only
         "FPL26_ILS_ENDHIGH_DENSITY_GATE": "1"},
        window=1000.0, anchor=250.0, cycles=24,
        spread=131.4, failing=22946)   # digit: density 175 < 363, share ~0.86
    assert "ExtraNetDelay_high" not in placed, placed
    assert any("endhigh-gate" in m and "BLOCK" in m for m in logs), logs


def test_endhigh_density_disarmed_still_reaches_it(monkeypatch):
    """The within-build control: same window, gate off, the combo IS reached.
    Without this, 'not placed' above could just mean the rotation never got
    there."""
    placed, _ = _run_loop(
        monkeypatch,
        {"FPL26_ILS_PLACE_RETRY_LADDER": "0",
         "FPL26_ILS_ENDHIGH_DENSITY_GATE": "0"},
        window=1000.0, anchor=250.0, cycles=24,
        spread=131.4, failing=22946)   # same design+window, gate disarmed
    assert "ExtraNetDelay_high" in placed, placed


def test_skip_unaffordable_advances_to_the_next_rung_in_the_same_cycle(monkeypatch):
    """Armed: an unaffordable rung 1 advances to rung 2 WITHIN the cycle instead
    of surrendering it to rotation."""
    placed, logs = _run_loop(
        monkeypatch,
        {"FPL26_ILS_PLACE_RETRY_LADDER": "1",
         "FPL26_ILS_LADDER_SKIP_UNAFFORDABLE": "1"},
        window=400.0, anchor=250.0)
    assert any("unaffordable" in m for m in logs), logs
    assert any("skip-unaffordable ARMED" in m for m in logs), logs
    assert "AltSpreadLogic_medium" in placed, placed


def test_skip_unaffordable_disarmed_surrenders_the_cycle(monkeypatch):
    """The control. Same window, same anchor, flag off: rung 1 still refuses, but
    nothing advances — which is what let an 845s rotation draw end digit's run.

    This is the within-build control the mechanism claim rests on: without it,
    'AltSpreadLogic_medium ran' proves nothing, since arming the ladder appends
    it to the rotation anyway (see test_ladder_reserve_stops_the_ladder...)."""
    _, logs = _run_loop(
        monkeypatch,
        {"FPL26_ILS_PLACE_RETRY_LADDER": "1",
         "FPL26_ILS_LADDER_SKIP_UNAFFORDABLE": "0"},
        window=400.0, anchor=250.0)
    assert any("unaffordable" in m for m in logs), logs
    assert not any("skip-unaffordable ARMED" in m for m in logs), logs


def test_skip_unaffordable_cannot_bypass_the_ladder_reserve(monkeypatch):
    """THE GUARD THAT MUST STILL FAIL.

    The 5/5 panel required that a rung past the first leave room for one more
    full cycle. Skip-unaffordable advances THROUGH rungs, so the obvious way for
    it to be wrong is to walk past a rung the reserve would have stopped.

    With reserve armed at anchor 250s / window 400s, rung 2 costs 262s and cannot
    also reserve 250s. Skip-unaffordable brought us to rung 2; the reserve must
    still gate it and clear the queue, so the ladder must NOT advance to rung 3.

    NOTE what this deliberately does NOT assert. My first version checked that
    ExtraNetDelay_low never appears in `placed`, and it failed — correctly.
    Arming the ladder APPENDS its combos to the rotation, so rung 3's directive
    can still run later on its own merits; test_ladder_reserve_stops_the_ladder_
    when_the_window_is_tight already says so in as many words. The reserve
    governs the FORCED probe, not the combo's existence. Asserting on `placed`
    tests the rotation, not the gate."""
    _, logs = _run_loop(
        monkeypatch,
        {"FPL26_ILS_PLACE_RETRY_LADDER": "1",
         "FPL26_ILS_LADDER_SKIP_UNAFFORDABLE": "1",
         "FPL26_ILS_LADDER_RESERVE": "1"},
        window=400.0, anchor=250.0)
    assert any("reserve-gated" in m for m in logs), logs
    # The gate fired AND the walk stopped there: no advance past the gated rung.
    assert not any("advancing to rung 3" in m for m in logs), logs


def test_skip_unaffordable_does_not_promote_a_rung_into_the_unreserved_slot(monkeypatch):
    """POPS, not runs. Rung 1 is the only never-reserve-gated slot; a skipped
    rung 1 must not hand that exemption to rung 2."""
    _, logs = _run_loop(
        monkeypatch,
        {"FPL26_ILS_PLACE_RETRY_LADDER": "1",
         "FPL26_ILS_LADDER_SKIP_UNAFFORDABLE": "1",
         "FPL26_ILS_LADDER_RESERVE": "1"},
        window=400.0, anchor=250.0)
    # The rung the reserve stops must be reported as rung 2, never rung 1.
    assert any("rung 2" in m and "reserve-gated" in m for m in logs), logs
    assert not any("rung 1" in m and "reserve-gated" in m for m in logs), logs


def test_baseline_gate_inert_when_baseline_unmeasured(monkeypatch):
    """Unknown baseline must not WIDEN the trigger and must not silently disable
    the mechanism — it leaves behaviour exactly as if the gate were absent."""
    assert _fires(monkeypatch, None, -0.619, -0.635, gate_armed=True) is True


def test_incr_route_skipped_when_incumbent_router_is_unknown(monkeypatch):
    """No accept yet -> the incumbent's router is unknown -> the rung is skipped
    rather than guessed. S6 proved escalation ORDER is causal, so firing blind
    would be a prediction, not a state fact."""
    logs = []
    import time as _t

    monkeypatch.setenv("FPL26_ILS_INCR_ROUTE", "1")

    async def fake(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "0.05"
        if "report_route_status" in cmd:
            return ("# of routable nets : 100\n# of fully routed nets : 100\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-0.95"        # never accepts -> _last_rd stays None
        return "ok"

    cfg = ILSPolishConfig(enabled=True, min_cycle_seconds=0.0, max_cycles=12,
                          final_seed_no_improve_stop=0, lastmile_min_wns_ns=-99.0)
    asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.9,
        deadline_ts=_t.time() + 400.0, wns_tcl="SLACK", cfg=cfg, log=logs.append))
    assert any("incr-route skipped" in m and "unknown" in m for m in logs), logs


def test_ladder_stops_probing_once_a_rung_accepts(monkeypatch):
    """Found live on spam: rung 2 ACCEPTED and rung 3 — a directive already measured
    WORSE on that design — then consumed the last of the ruin window. An accept
    answers the placement-family question, so the remaining rungs are dropped.

    WNS script: cycle 1 regresses (queues the ladder), rung 1 also fails to beat
    the incumbent, rung 2 accepts. Two readings per cycle (post-route, post-physopt).
    """
    seq = ["-0.95", "-0.95",     # cycle 1 Explore: regression -> ladder queued
           "-0.95", "-0.95",     # rung 1: no accept
           "-0.80", "-0.80"] + ["-0.80"] * 12   # rung 2: ACCEPT
    _, placed, _ = _capture_cmds(
        monkeypatch, {"FPL26_ILS_PLACE_RETRY_LADDER": "1",
                      "FPL26_ILS_LADDER_STOP_ON_ACCEPT": "1"},
        seq, cycles=8)
    assert placed[:3] == ["Explore", "ExtraNetDelay_high", "AltSpreadLogic_medium"], placed
    assert "ExtraNetDelay_low" not in placed, placed


def test_ladder_without_stop_on_accept_keeps_probing(monkeypatch):
    """Control for the test above: disarmed, rung 3 still runs — which is what spent
    spam's last ruin cycle on a directive already known to be worse."""
    seq = ["-0.95", "-0.95", "-0.95", "-0.95", "-0.80", "-0.80"] + ["-0.80"] * 12
    _, placed, _ = _capture_cmds(
        monkeypatch, {"FPL26_ILS_PLACE_RETRY_LADDER": "1",
                      "FPL26_ILS_LADDER_STOP_ON_ACCEPT": "0"},
        seq, cycles=8)
    assert placed[:4] == ["Explore", "ExtraNetDelay_high", "AltSpreadLogic_medium",
                          "ExtraNetDelay_low"], placed


def test_stop_on_accept_is_default_off(monkeypatch):
    from optimizer.ils_polish import ladder_stop_on_accept_enabled
    monkeypatch.delenv("FPL26_ILS_LADDER_STOP_ON_ACCEPT", raising=False)
    assert ladder_stop_on_accept_enabled() is False


# ---- STUCK-SEED THRESHOLD OVERRIDE (jul28) ---------------------------------
# Which NETLIST the ILS re-places from decides optical's record: its recipe_gain is
# 0.154 against a 0.15 cut, so it takes the recipe_best seed and lands -0.971 where
# the raw-seeded sweep lands -0.846. These pin the override, not the constant.
STUCK_CORPUS = [("spam", 0.021, "raw"), ("corescore", 0.141, "raw"),
                ("optical", 0.154, "recipe_best"), ("vtr", 0.211, "recipe_best"),
                ("vexriscv", 1.035, "recipe_best")]


@pytest.mark.parametrize("name,gain,want", STUCK_CORPUS)
def test_shipped_seed_choice_matches_the_measured_corpus(monkeypatch, name, gain, want):
    """Default threshold reproduces which seed each design actually took."""
    from optimizer.ils_polish import choose_ils_seed
    monkeypatch.delenv("FPL26_ILS_STUCK_GAIN_NS", raising=False)
    cfg = ILSPolishConfig(enabled=True)
    kind, g = choose_ils_seed(initial_wns=-1.0, recipe_wns=-1.0 + gain, cfg=cfg)
    assert kind == want, f"{name}: gain {gain} -> {kind}, expected {want}"


def test_override_moves_optical_to_the_raw_seed(monkeypatch):
    """optical misses the raw path by 0.004 ns; the override is how we MEASURE
    whether raw is what it needs, without touching the shipped constant."""
    from optimizer.ils_polish import choose_ils_seed
    cfg = ILSPolishConfig(enabled=True)
    monkeypatch.setenv("FPL26_ILS_STUCK_GAIN_NS", "0.20")
    assert choose_ils_seed(initial_wns=-1.0, recipe_wns=-0.846, cfg=cfg)[0] == "raw"
    # vexriscv (1.035) must STAY on recipe-best even at the raised threshold
    assert choose_ils_seed(initial_wns=-1.0, recipe_wns=0.035, cfg=cfg)[0] == "recipe_best"


def test_stuck_threshold_override_default_and_garbage_are_inert(monkeypatch):
    from optimizer.ils_polish import stuck_gain_threshold
    cfg = ILSPolishConfig(enabled=True)
    monkeypatch.delenv("FPL26_ILS_STUCK_GAIN_NS", raising=False)
    assert stuck_gain_threshold(cfg) == cfg.stuck_recipe_gain_ns
    monkeypatch.setenv("FPL26_ILS_STUCK_GAIN_NS", "not-a-number")
    assert stuck_gain_threshold(cfg) == cfg.stuck_recipe_gain_ns


def test_any_forced_probe_is_futility_exempt_not_just_ladder_rungs(monkeypatch):
    """REGRESSION TEST for a measured -8.02 MHz loss.

    optical arm=mb (single-rung place-retry): the forced ExtraNetDelay_high probe
    returned -0.971, took the SECOND futility strike, and the loop stopped at
    cycles=2 accepted=0 -> +17.11, against +25.13 with the probe disarmed. In the
    control the second cycle was ROUTE_ONLY and it ACCEPTED. The exemption had been
    gated behind the ladder flag, so the single-rung path never got it.

    Replays that shape with the REAL K=2 futility rule and every cycle regressing:
    without the exemption the run ends at cycles=2 (cycle 1 = strike 1, the forced
    probe = strike 2); with it, the probe does not count and a third cycle runs.
    """
    import time as _t
    monkeypatch.setenv("FPL26_ILS_PLACE_RETRY", "1")

    async def fake(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "0.05"
        if "report_route_status" in cmd:
            return ("# of routable nets : 100\n# of fully routed nets : 100\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            return "-0.95"          # always worse than the -0.9 incumbent
        return "ok"

    cfg = ILSPolishConfig(enabled=True, min_cycle_seconds=0.0, max_cycles=12,
                          final_seed_no_improve_stop=0, lastmile_min_wns_ns=-99.0)
    res = asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.9,
        deadline_ts=_t.time() + 4000.0, wns_tcl="SLACK", cfg=cfg,
        log=lambda m: None, no_improve_stop=2))
    assert res.cycles >= 3, (
        "forced probe still burns a futility strike: cycles=%s" % res.cycles)


def test_incr_route_priority_takes_the_cycle_before_the_route_sentinels(monkeypatch):
    """spam jul28 arm=stopaccept reached +20.85 and then logged the escalation as
    skipped TWICE — ROUTE_REROLL/ROUTE_ONLY had already re-routed the incumbent
    with AggressiveExplore, destroying the precondition S3 needs. Armed, the
    escalation must take its cycle while it is still eligible."""
    from optimizer.ils_polish import INCR_ROUTE_PD
    seq = ["-0.80", "-0.80"] + ["-0.85"] * 20   # cycle 1 accepts -> _last_rd=Explore
    _, _, _ = _capture_cmds(
        monkeypatch, {"FPL26_ILS_INCR_ROUTE": "1", "FPL26_ILS_INCR_ROUTE_FIRST": "1"},
        seq, cycles=6)
    # behavioural assertion lives in the log-based test below; here just ensure the
    # armed rotation still contains the operator and the run completes.
    from optimizer.ils_polish import active_combos
    assert any(c[0] == INCR_ROUTE_PD for c in active_combos())


def _run_incr_route_priority(monkeypatch, route_anchor_s, window_s):
    """Drive the armed priority jump with a given route anchor and window.

    The jump's displacement guard compares the FULL-ROUTE need
    (route_cost_anchor_s x ROUTE_REROLL_COST_MARGIN) against the remaining
    window, so those two numbers are the whole experiment."""
    logs = []
    import time as _t
    monkeypatch.setenv("FPL26_ILS_INCR_ROUTE", "1")
    monkeypatch.setenv("FPL26_ILS_INCR_ROUTE_FIRST", "1")
    seq = ["-0.80", "-0.80"] + ["-0.85"] * 20
    state = {"i": 0}

    async def fake(tool, args):
        cmd = args.get("command", "")
        if "-hold" in cmd:
            return "0.05"
        if "report_route_status" in cmd:
            return ("# of routable nets : 100\n# of fully routed nets : 100\n"
                    "# of nets with routing errors : 0\n")
        if "SLACK" in cmd:
            i = min(state["i"], len(seq) - 1); state["i"] += 1
            return seq[i]
        return "ok"

    cfg = ILSPolishConfig(enabled=True, min_cycle_seconds=0.0, max_cycles=6,
                          final_seed_no_improve_stop=0, lastmile_min_wns_ns=-99.0,
                          route_cost_anchor_s=route_anchor_s)
    asyncio.run(run_ils_polish(
        fake, best_dcp_path="/tmp/x.dcp", baseline_wns=-0.9,
        deadline_ts=_t.time() + window_s, wns_tcl="SLACK", cfg=cfg,
        log=logs.append))
    return logs


def test_incr_route_priority_fires_when_a_full_route_is_priced_out(monkeypatch):
    """spam jul28: remaining 292s vs full-route need 367s. Nothing else could have
    used that cycle, so the escalation takes it and carries the record step
    (-0.598 -> -0.543, +29.19)."""
    logs = _run_incr_route_priority(monkeypatch, route_anchor_s=400.0,
                                    window_s=300.0)
    assert any("incr-route PRIORITY:" in m for m in logs), logs
    assert any("displaces nothing" in m for m in logs), logs


def test_incr_route_priority_yields_when_a_full_route_still_fits(monkeypatch):
    """3d jul28 night16: remaining 441s vs full-route need 376s. The jump displaced
    ROUTE_ONLY (+0.079ns) for a 0.005ns micro-accept and cost the design 5.96 MHz.
    With the guard the jump must YIELD, leaving the cycle to the from-scratch
    route."""
    logs = _run_incr_route_priority(monkeypatch, route_anchor_s=290.0,
                                    window_s=4000.0)
    assert any("incr-route PRIORITY yielded" in m for m in logs), logs
    assert not any("incr-route PRIORITY:" in m for m in logs), logs


def test_incr_route_priority_yields_when_the_route_anchor_is_unknown(monkeypatch):
    """No anchor = cannot PROVE a full route is priced out. Fail safe to the
    pre-jul28 shipped order rather than jump on an unmeasured guess."""
    logs = _run_incr_route_priority(monkeypatch, route_anchor_s=0.0,
                                    window_s=4000.0)
    assert not any("incr-route PRIORITY:" in m for m in logs), logs


def test_incr_route_priority_default_on_and_disarms_on_zero(monkeypatch):
    """jul29: this flag flipped to DEFAULT ON because the eval path never armed it —
    spam ships +17.51 instead of +29.19 without it. Both directions are pinned here,
    since the ON default is now a SHIP property and not merely a convenience."""
    from optimizer.ils_polish import incr_route_first_enabled
    monkeypatch.delenv("FPL26_ILS_INCR_ROUTE_FIRST", raising=False)
    assert incr_route_first_enabled() is True, "eval runs set no flags; this must be on"
    monkeypatch.setenv("FPL26_ILS_INCR_ROUTE_FIRST", "0")
    assert incr_route_first_enabled() is False, "=0 must remain a real kill switch"


def test_ladder_order_by_wns_is_default_on_with_kill_switch(monkeypatch):
    """aug01: PROMOTED to default-ON, so this pins the NEW contract.

    It used to assert default-OFF, with the rationale "the shipped evidence order
    stands until an A/B moves it". `ladder_ab_jul31` is that A/B: 6 designs x
    (treatment, control), one build, ship surface a828b63f on both boxes. The
    reorder FIRED on three — spam 29.19 vs ctrl 10.78, fir 21.30 vs 13.38,
    logicnets 105.61 vs 105.61 — with ZERO firing designs below their mode, and
    was a provable no-op on the other three (|ILS baseline| >= 0.7). The
    pre-registered rule (spam >= 26.16 AND no firing design below mode) is met.

    Kept as an inversion rather than deleted: a promoted flag still needs its
    default AND its escape hatch pinned, or a later refactor silently un-ships it.
    """
    from optimizer.ils_polish import ladder_rungs, PLACE_RETRY_LADDER
    monkeypatch.delenv("FPL26_ILS_LADDER_ORDER", raising=False)
    monkeypatch.delenv("FPL26_ILS_LADDER_ORDER_BY_WNS", raising=False)
    # default-ON: a near-met baseline now reorders with nothing set.
    assert ladder_rungs(-0.665, 0.7)[0] == "AltSpreadLogic_medium"
    # ...and a DEEP baseline is still untouched, flag or no flag. This is the
    # property that made optical (-0.959, measured) safe without ever being run.
    assert ladder_rungs(-0.959, 0.7) == list(PLACE_RETRY_LADDER)
    # kill switch restores the pre-aug01 order exactly.
    monkeypatch.setenv("FPL26_ILS_LADDER_ORDER_BY_WNS", "0")
    assert ladder_rungs(-0.665, 0.7) == list(PLACE_RETRY_LADDER)


def test_ladder_order_by_wns_promotes_medium_only_for_near_met(monkeypatch):
    """The three designs with rung data, at their measured ILS baselines.

    spam (-0.665) is near-met and its record needs AltSpreadLogic_medium from the
    untouched seed. optical (-0.924) and 3d (-2.153) are deep, and medium measures
    -1.086 / -2.408 on them — on 3d that is worse than its own baseline."""
    from optimizer.ils_polish import ladder_rungs, PLACE_RETRY_LADDER
    monkeypatch.delenv("FPL26_ILS_LADDER_ORDER", raising=False)
    monkeypatch.setenv("FPL26_ILS_LADDER_ORDER_BY_WNS", "1")

    assert ladder_rungs(-0.665, 0.7)[0] == "AltSpreadLogic_medium"   # spam
    assert ladder_rungs(-0.924, 0.7) == list(PLACE_RETRY_LADDER)     # optical
    assert ladder_rungs(-2.153, 0.7) == list(PLACE_RETRY_LADDER)     # 3d
    # promotion REORDERS, never drops or duplicates a rung
    assert sorted(ladder_rungs(-0.665, 0.7)) == sorted(PLACE_RETRY_LADDER)


def test_ladder_order_by_wns_needs_both_inputs(monkeypatch):
    """No baseline (or no near-met magnitude) = no evidence to key on = shipped order."""
    from optimizer.ils_polish import ladder_rungs, PLACE_RETRY_LADDER
    monkeypatch.delenv("FPL26_ILS_LADDER_ORDER", raising=False)
    monkeypatch.setenv("FPL26_ILS_LADDER_ORDER_BY_WNS", "1")
    assert ladder_rungs(None, 0.7) == list(PLACE_RETRY_LADDER)
    assert ladder_rungs(-0.665, None) == list(PLACE_RETRY_LADDER)
    assert ladder_rungs() == list(PLACE_RETRY_LADDER)


def test_explicit_ladder_order_beats_the_wns_rule(monkeypatch):
    """An operator naming the order is stating a fact about this run; the heuristic
    must not silently override it."""
    from optimizer.ils_polish import ladder_rungs
    monkeypatch.setenv("FPL26_ILS_LADDER_ORDER_BY_WNS", "1")
    monkeypatch.setenv("FPL26_ILS_LADDER_ORDER",
                       "ExtraNetDelay_low|ExtraNetDelay_high")
    assert ladder_rungs(-0.665, 0.7) == ["ExtraNetDelay_low", "ExtraNetDelay_high"]


def test_ladder_order_knob_default_and_override(monkeypatch):
    """The FIRST rung runs from the UNMODIFIED seed; later rungs run from whatever
    has accepted. spam's record places AltSpreadLogic_medium from the RAW
    benchmark, so reproducing it needs that rung first."""
    from optimizer.ils_polish import ladder_rungs, PLACE_RETRY_LADDER
    monkeypatch.delenv("FPL26_ILS_LADDER_ORDER", raising=False)
    assert ladder_rungs() == list(PLACE_RETRY_LADDER)
    monkeypatch.setenv("FPL26_ILS_LADDER_ORDER",
                       "AltSpreadLogic_medium,ExtraNetDelay_high")
    assert ladder_rungs() == ["AltSpreadLogic_medium", "ExtraNetDelay_high"]
    monkeypatch.setenv("FPL26_ILS_LADDER_ORDER", "   ")
    assert ladder_rungs() == list(PLACE_RETRY_LADDER)


def test_ladder_order_knob_changes_the_probe_sequence(monkeypatch):
    _, placed, _ = _capture_cmds(
        monkeypatch, {"FPL26_ILS_PLACE_RETRY_LADDER": "1",
                      "FPL26_ILS_LADDER_ORDER": "AltSpreadLogic_medium,ExtraNetDelay_high"},
        ["-0.95"] * 16, cycles=4)
    assert placed[1] == "AltSpreadLogic_medium", placed


# ---- RE-DRAW RESERVE (jul31, fir) ------------------------------------------
# fir handed back 2923s, the ILS ate ~2000s for +0.006ns, and the wrapper then
# refused attempt 2 at "remaining 763s < floor 1200s". The re-draw was worth
# ~0.5 x 12.23 = +6 MHz in expectation. An unproven stage must not spend it.

def test_redraw_reserve_is_default_off(monkeypatch):
    from optimizer.ils_polish import redraw_reserve_enabled
    monkeypatch.delenv("FPL26_ILS_REDRAW_RESERVE", raising=False)
    assert redraw_reserve_enabled() is False


def test_redraw_reserve_s_rejects_garbage(monkeypatch):
    """A broken override must fall back to the attempt floor, never to 0 --
    a 0 reserve silently reinstates the exact fir failure."""
    from optimizer.ils_polish import redraw_reserve_s, REDRAW_RESERVE_S_DEFAULT
    for bad in ("", "not-a-number", "0", "-1"):
        monkeypatch.setenv("FPL26_ILS_REDRAW_RESERVE_S", bad)
        assert redraw_reserve_s() == REDRAW_RESERVE_S_DEFAULT
    monkeypatch.setenv("FPL26_ILS_REDRAW_RESERVE_S", "900")
    assert redraw_reserve_s() == 900.0


def test_redraw_reserve_stops_an_unproductive_ils(monkeypatch):
    """fir's case: cycles run, nothing meaningful accepted, so the ILS must stop
    while the attempt floor is still intact instead of spending it."""
    # window 600s, anchor 100 -> next cycle priced 115s, so remaining-next = 485.
    # A 550s reserve makes 485 < 550 true and the guard must fire.
    # wrapper_window 600s, next cycle priced 115s -> 485 < 550 reserve, and the
    # loop runs past REDRAW_MIN_CYCLES_DEFAULT, so the guard must fire.
    _, logs = _run_loop(
        monkeypatch,
        {"FPL26_ILS_REDRAW_RESERVE": "1", "FPL26_ILS_REDRAW_RESERVE_S": "550"},
        wns="-0.95", window=600.0, anchor=100.0, cycles=24, wrapper_window=600.0)
    assert any("redraw-reserve" in m for m in logs), logs


def test_redraw_reserve_disarmed_keeps_spending(monkeypatch):
    """The within-build control: identical window/anchor, reserve off, the ILS
    runs on. Without this, 'it stopped' proves nothing -- the loop could have
    ended on budget or futility."""
    _, logs = _run_loop(
        monkeypatch,
        {"FPL26_ILS_REDRAW_RESERVE": "0", "FPL26_ILS_REDRAW_RESERVE_S": "550"},
        wns="-0.95", window=600.0, anchor=100.0, cycles=24, wrapper_window=600.0)
    assert not any("redraw-reserve" in m for m in logs), logs


def test_redraw_reserve_never_fires_before_the_first_cycle(monkeypatch):
    """cyc>0 guard: a tight window must not stop the ILS before it has run a
    single cycle -- that would hand the wrapper a re-draw of nothing."""
    _, logs = _run_loop(
        monkeypatch,
        {"FPL26_ILS_REDRAW_RESERVE": "1", "FPL26_ILS_REDRAW_RESERVE_S": "100000"},
        wns="-0.95", window=400.0, anchor=250.0, cycles=24, wrapper_window=400.0)
    assert not any("redraw-reserve: no meaningful accept in 0 cycles" in m
                   for m in logs), logs


def test_redraw_reserve_v1_arithmetic_would_have_killed_digit(monkeypatch):
    """⛔ Pins WHY v1 was wrong, so the fenced-deadline bug cannot come back.

    v1 compared the reserve against the ILS's fenced remaining:
        1246s (fenced) - 527s = 719s  <  1200s  -> STOP at digit's cycle-4 head
    v2 compares against the WRAPPER's remaining:
        1746s (unfenced) - 527s = 1219s > 1200s -> CONTINUE
    Cycle 4 is ExtraNetDelay_low -> -0.614 -> alpha 72.59, the whole +38 MHz.
    """
    from optimizer.ils_polish import redraw_reserve_s, MEASURED_BASIS_MARGIN
    monkeypatch.delenv("FPL26_ILS_REDRAW_RESERVE_S", raising=False)
    next_cost = 458 * MEASURED_BASIS_MARGIN          # digit's measured basis
    fenced, wrapper = 1246.0, 1746.0                 # 500s polish reserve apart
    assert (fenced - next_cost) < redraw_reserve_s(), "v1 bug must stay pinned"
    assert (wrapper - next_cost) >= redraw_reserve_s(), (
        "v2 must clear the reserve on the WRAPPER budget — if this fails, digit "
        "is being stopped one cycle before its win again")


def test_redraw_reserve_min_cycles_covers_digits_19_second_margin():
    """The wrapper-budget fix clears digit by only 19s. A 19s margin deciding
    38 MHz is the knife-edge that already cost 17.7 points elsewhere, so the
    reserve must also refuse to fire before digit's winning cycle (its 4th)."""
    from optimizer.ils_polish import REDRAW_MIN_CYCLES_DEFAULT
    DIGIT_WINNING_CYCLE = 4
    assert REDRAW_MIN_CYCLES_DEFAULT > DIGIT_WINNING_CYCLE, (
        "the floor must sit ABOVE digit's winning cycle so timing luck cannot "
        "decide a 38 MHz outcome")


def test_redraw_reserve_fails_open_without_a_wrapper_deadline(monkeypatch):
    """Unknown wrapper budget -> the reserve must not fire at all. Guessing it
    from the fenced deadline is exactly the v1 defect."""
    _, logs = _run_loop(
        monkeypatch,
        {"FPL26_ILS_REDRAW_RESERVE": "1", "FPL26_ILS_REDRAW_RESERVE_S": "550"},
        wns="-0.95", window=600.0, anchor=100.0, cycles=24)
    assert not any("redraw-reserve" in m for m in logs), logs
