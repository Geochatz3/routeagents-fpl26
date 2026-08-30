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
    exhaustion, pristine_rot == cycles). The place-retry extension
    (FPL26_ILS_PLACE_RETRY) deliberately breaks the cycles==rotation-advance
    invariant with one out-of-order forced pick, so a test of DEFAULT behaviour
    must not depend on whether that variable happens to be set in the ambient
    environment. The extension's own tests arm it explicitly.
    """
    monkeypatch.delenv("FPL26_ILS_PLACE_RETRY", raising=False)
    # Force optional ILS features off so these tests are independent of the
    # caller's environment and exercise the disarmed baseline.
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
    # Explore and LASTMILE lead so the two-cycle futility window covers both
    # operator classes. ROUTE_REROLL follows but is skipped for deep negative slack.
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
    """Verify that the final or sole seed uses a two-cycle futility stop by
    default.

    Non-accepting tail cycles incur a wall-time penalty, while 0 remains an
    explicit immediate-stop setting.
    """
    c = ILSPolishConfig()
    assert c.final_seed_no_improve_stop == 2
    c2 = ILSPolishConfig(final_seed_no_improve_stop=0)
    assert c2.final_seed_no_improve_stop == 0


def test_run_ils_polish_heavy_cmds_carry_large_timeout():
    """Verify that heavy implementation commands receive timeouts longer than 300
    seconds.

    The explicit timeout must exceed the server's 300-second default, which can
    terminate long-running steps and leave WNS unavailable.
    """
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
    # timing MET + surplus budget -> fire in met-surplus mode (:
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
    # The 0.15 ns floor treats small recipe gains as stuck rather than replacing
    # the raw seed.
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
    # sentinels since the insertion): rotation must continue from
    # there, not restart at 0.
    assert placed[0] == ILS_COMBOS[4][0]
    # reorder (+shift): idx5 is now the PARTIAL_RUIN sentinel
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
    # PARTIAL_RUIN places without a directive; ROUTE_ONLY and ROUTE_REROLL do not place.
    # The near-met baseline admits ROUTE_REROLL, so it consumes a cycle without
    # adding a directive. Rotation must still exhaust all remaining combinations once.
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
    critical_path_avg_spread_tiles at its None default, and since that
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
    placement — never touch place_design in any form."""
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
    """Accept exactly zero hold slack through the combined hold gate.

    The gate uses a -0.001 ns floor, matching score validation and preserving
    valid route-only improvements.
    """
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
    # PARTIAL_RUIN must immediately follow ExtraTimingOpt so the finisher chain
    # runs before more expensive operators, even under tight budgets.
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
    """Keep LASTMILE at index 1, inside the two-cycle futility window.

    Placing it later makes the plateau-oriented combination unreachable when
    neither of the first two cycles is accepted.
    """
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
    """Verify cold-start selection skips an unaffordable full-ruin cycle.

    When observed costs are unavailable, the heavy-cycle anchor estimates
    affordability and selection falls through to the earliest affordable combo.
    """
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
    # With a 1,500 s window and 1,800 s anchor, earlier operators are unaffordable.
    # At near-met WNS, ROUTE_REROLL costs 990 s and is the first affordable
    # eligible operator, so the single cycle performs an unroute and reroute.
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
    """campaign: the MCP server reports Vivado-side failures as plain
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
    """batch repro: after `place_design -directive LastMile`, the
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
    """Verify LastMile is skipped when timing is far from closure.

    Below `lastmile_min_wns_ns`, the cycle must select and run a non-LastMile combo.
    """
    from optimizer.ils_polish import LASTMILE_PD
    placed = []
    async def fake_call_tool(tool, args):
        cmd = args.get("command", "")
        # real work = any place OR route directive (the combo after LASTMILE
        # is ROUTE_ONLY, which never places)
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
    """: a single transient write_checkpoint failure (WSL /mnt/c
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
    """Verify LastMile remains eligible when WNS is above -1.0 ns.

    The cutoff restricts the directive to designs sufficiently close to timing
    closure without excluding viable near-threshold cases.
    """
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


# ---- fanout_polish_accept: strict-hold never-worse gate ----
FCFG = ILSPolishConfig()   # defaults: accept_margin_ns=0.002, fanout floor=0.010


def test_fanout_accept_hold_safe_gain():
    # A setup improvement with 0.068 ns hold slack is accepted.
    ok, why = fanout_polish_accept(new_wns=-0.572, best_wns=-0.601, unrouted=0,
                                   whs=0.068, base_whs=0.068, cfg=FCFG)
    assert ok, why


def test_fanout_reject_hold_marginal():
    # A setup gain is rejected when hold slack falls below the 0.010 ns strict floor.
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


# ---- cell-count guard: LASTMILE accepts vs validator band ----

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
    # This pins acceptance of a legitimate 10% cell-count reduction; the bound
    # is a sanity check rather than a strict preservation requirement.
    res = _run_lastmile(cell_count=90_000, golden=100_000)
    assert res.accepted == 1 and res.best_wns == -0.2


def test_cell_guard_fails_open_without_golden():
    res = _run_lastmile(cell_count=1, golden=None)
    assert res.accepted == 1                 # guard disabled, accept proceeds


# Full-cycle place-and-route cost gates ILS combination selection.
# Fanout polish falls back to a phys-opt-plus-route anchor for route-only
# recipes, which provide no placement sample.
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
    # This fixture provides only granular phys-opt samples, with no place or route cost.
    # The fallback fanout anchor is the maximum sample, 68 s, below the 600 s gate.
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


# ---- fanout-polish cheap-design gate: granular-anchor fallback ----

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
    # A complete cost sample still blocks fanout when routing alone exceeds the gate.
    assert _fanout_gate_probe(expected=2000.0, fan_anchor=700.0,
                              has_route=True) == 0


def test_fanout_gate_incomplete_fan_anchor_keeps_full_cycle_basis():
    # Without a route sample, reroute cost is unknown.
    # When a full-cycle anchor exists, it remains the cost basis; a cheap phys-opt
    # anchor cannot admit fanout polish for a slow full cycle.
    assert _fanout_gate_probe(expected=1200.0, fan_anchor=300.0) == 0


def test_fanout_gate_record_run_logicnets_replay():
    # The gate prices a route-sampled probe from the 237 s fanout anchor,
    # not the 489 s full-cycle estimate, leaving it affordable within 1,194 s.
    assert _fanout_gate_probe(expected=489.0, fan_anchor=237.0,
                              deadline_offset=1194.0, has_route=True) > 0


def test_fanout_gate_v2_eval_replay():
    # Preview #6 v2: full-cycle 718s (581s place!) hard-skipped the cheap
    # gate; true polish cost 137s (phys 34 + route 43 + 60) -> now attempts.
    # Never-worse + strict-hold accept make a reject free.
    assert _fanout_gate_probe(expected=718.0, fan_anchor=137.0,
                              has_route=True) > 0


# ---- corrective-seed local climb ----

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
    # Each cycle performs two slack queries, so values are paired to keep
    # the reject, accept, reject outcomes unambiguous.
    # The second-cycle acceptance resets rotation; the final rejection advances it to 2.
    res = _run_seq([-0.95, -0.95, -0.5, -0.5, -1.2, -1.2], max_cycles=3)
    assert res.accepted >= 1
    # first accept happened on the cycle picked at rotation index 2 ->
    # pristine_rot frozen at 2 (idx0 and the accepting idx1 are the only
    # genuine replays for a near-identical sibling seed).
    assert res.pristine_rot == 2


def test_pristine_rot_zero_accepts_equals_continue_semantics():
    # With no accepted cycles, pristine_rot equals the number of executed cycles.
    # The baseline avoids inline combo skips so each cycle advances rotation once.
    res = _run_seq([-1.5, -1.5, -1.4, -1.4, -1.3, -1.3], max_cycles=3,
                   baseline=-0.5)
    assert res.accepted == 0
    assert res.pristine_rot == 3 == res.cycles


def test_corrective_local_climb_default_on():
    # default ON after 3/3 live validation (v2 debug-wall runs,
    # shipped >= control every time); eval #5-vs-#6 forensics carry the
    # upside. False remains the kill switch.
    assert ILSPolishConfig().corrective_local_climb is True


# ---- ROUTE_ONLY granular cold-start estimate ----

def test_cold_start_route_only_uses_granular_estimate():
    """local-campaign v2 leg: place-dominated anchor (place 1568s of
    expected_heavy_cycle_s=1790s) priced ROUTE_ONLY at 0.6*1790=1074s in an
    875s window -> 'no affordable combo' -> ILS ran 0 cycles and shipped
    BASELINE, on the design where ROUTE_ONLY accepted in previews #5 AND #6.
    With the granular route+phys anchor (221s, has_route) an unroute +
    re-route combo must be affordable and picked (since the first
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


# ---- fanout-polish finalize reserve (: five consecutive near-misses) ----

def test_fanout_gate_eval7_v2_replay_fires_with_finalize_reserve():
    # Eval #7 v2: granular anchor 137 (no place sample in recipe), 731s
    # remaining. Old need = 137*1.3 + 600 = 778 -> missed by 47s. New need =
    # 137*1.3 + 300 = 478 -> FIRES.
    assert _fanout_gate_probe(expected=0.0, fan_anchor=137.0,
                              deadline_offset=731.0, has_route=True) > 0


def test_fanout_gate_reserve_still_skips_truly_tight():
    # With 233 s remaining, no attempt fits the 478 s minimum budget.
    assert _fanout_gate_probe(expected=0.0, fan_anchor=137.0,
                              deadline_offset=233.0, has_route=True) == 0


# ---- combo-cost carry across seeds (GAP #4, eval #9 v2 forensics) ----

def test_combo_cost_seed_makes_observed_cheap_combo_affordable():
    """Verify a freshly observed combo cost determines affordability during
    corrective seeding.

    Observed costs take precedence over cold-prior estimates so a measured
    affordable combo remains selectable.
    """
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


# ---- LASTMILE final polish: accept gate ----
from optimizer.ils_polish import lastmile_polish_accept

LCFG = ILSPolishConfig(golden_cell_count=100_000)


def test_lastmile_polish_accepts_probe_case():
    # probe: v2 -0.799 -> -0.647, hold clean, cells in band
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
    # -10% shrink is legal now (upstream PR #41; band is sanity-only)
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


# Integrated final-polish wrapper coverage includes entry and budget gates,
# step-error aborts, incremental reroute retry, and acceptance bookkeeping.

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
    # v2 case + the bff156a fix: first measure sees 39 unrouted ->
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
    # phantom-accept lesson applied to this stage).
    stub, calls = _lastmile_stage_stub(
        route_status_seq=[_RS_CLEAN], slack_seq=["-0.2"], write_fails=True)
    _drive_lastmile(stub, best_wns=-0.5)
    assert stub.best_wns is None
    assert stub._best_valid_dcp is None


# A gain below meaningful_accept_ns remains accepted but counts as futile,
# preventing negligible gains from resetting the no-improvement counter.

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


# Near-zero critical-path spread blocks only the PARTIAL_RUIN combo family.
# Other combo families remain eligible and rotation continues.

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
    # A ~15-tile spread gates partial ruin while rotation runs every other combination.
    res, cmds, logs = _run_spread(15.2)
    assert not _ran_partial_ruin(cmds)
    assert res.cycles == len(ILS_COMBOS) - 1     # all non-gated combos ran
    assert any("place_design -unplace" in c for c in cmds)  # full-ruin ran
    assert any("rotation exhausted" in n for n in res.notes)
    assert any("partial-ruin spread-gated" in n for n in res.notes)
    assert any("partial-ruin skipped: spread=15.2 < 30 "
               "(evidence corpus 26/26 negative" in l for l in logs)


def test_spread_gate_fails_closed_when_spread_unknown():
    """Verify partial ruin is blocked when placement spread is unknown.

    Unknown spread fails closed because it cannot be distinguished from the
    risky co-located condition. The end-high gate intentionally handles unknown
    values differently.
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
    # A wide ~302-tile spread keeps partial ruin enabled.
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


# Route reroll is limited to near-met, route-dominated states.
# Deep negative slack remains assigned to the tail reroute loop.

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
    assert cfg.route_reroll_max_wns_mag == 0.7       # probe: bite only shallow; optical -0.842 measured negative
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
    """Freeze-style: the insertion must not reorder anything else —
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


# Place retry is opt-in and must not change the default combo ordering,
# list length, or indices used by rotation-exhaustion accounting.

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
    """Verify a regressing cycle schedules the measured-better directive next.

    Immediate promotion prevents futility termination from making the directive
    unreachable at the end of the rotation.
    """
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


# A regressing first cycle activates place retry.
# Retry pricing then uses the measured 225 s cycle instead of the 672 s
# cold-start anchor, making the forced pick affordable within a 2,110 s window.
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


# ---- PROBE LADDER + INCREMENTAL RE-ROUTE ---------------------------

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
    """Verify every retry-ladder rung runs as a separate forced cycle in declared
    order.

    Preserving evidence order keeps later rungs reachable despite normal
    futility termination.
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
    """: renamed from '..._unless_armed'. INCR_ROUTE is now DEFAULT ON, so the
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


# Rows contain pristine baseline, incumbent, first-cycle WNS, and expected trigger.
# Cases distinguish retry-worthy regressions from cycles that merely trail an incumbent.
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


# ---- PANEL REFINEMENTS: ladder reserve + incr-route terminal --------

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


# With a 250 s anchor and 400 s window, the first rung costs at least 604 s
# under either prior, while the second costs 262 s.
# Skipping the unaffordable rung must leave the affordable rung reachable.

def test_skip_unaffordable_is_default_on_with_kill_switch(monkeypatch):
    """Verify unaffordable-combo skipping defaults to enabled and supports an
    explicit kill switch.
    """
    from optimizer.ils_polish import ladder_skip_unaffordable_enabled
    monkeypatch.delenv("FPL26_ILS_LADDER_SKIP_UNAFFORDABLE", raising=False)
    assert ladder_skip_unaffordable_enabled() is True
    monkeypatch.setenv("FPL26_ILS_LADDER_SKIP_UNAFFORDABLE", "0")
    assert ladder_skip_unaffordable_enabled() is False, "=0 must stay a real kill switch"


# Density is the failing-endpoint count divided by path spread.
# Threshold 363 is the geometric midpoint between blocked 255 and admitted 518.

def _cfg_with(spread, failing):
    return ILSPolishConfig(enabled=True,
                           critical_path_avg_spread_tiles=spread,
                           phase1_failing_endpoints=failing)


def test_endhigh_density_gate_is_default_on_with_a_kill_switch(monkeypatch):
    """DEFAULT ON (digit +72.59 and optical +32.38 on one build).
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
    # The small-bet path bypasses density filtering at a 0.2 budget share.
    for name, (sp, fa) in {**armed, **blocked}.items():
        blk, why = endhigh_density_blocked(_cfg_with(sp, fa), 200.0, 1000.0)
        assert blk is False, f"{name} must ARM on a small bet: {why}"


def test_endhigh_density_fails_open_on_every_unknown(monkeypatch):
    """Verify the end-high density guard fails open when any required input is unknown.

    Missing optional analysis must not block a potentially useful directive.
    """
    from optimizer.ils_polish import endhigh_density_blocked
    monkeypatch.setenv("FPL26_ILS_ENDHIGH_DENSITY_GATE", "1")
    nan, inf = float("nan"), float("inf")
    for cfg in (_cfg_with(None, 7866),      # spread unmeasured
                _cfg_with(15.2, None),      # failing unmeasured
                _cfg_with(None, None),      # neither
                _cfg_with(0.0, 7866),       # zero spread (would divide by zero)
                _cfg_with(-3.0, 7866),      # nonsense spread
                _cfg_with("x", 7866),       # unparseable
                # Non-finite spread or endpoint inputs must fail open. Explicit
                # checks are required because NaN comparisons otherwise fall
                # through.
                _cfg_with(nan, 7866),
                _cfg_with(15.2, nan),
                _cfg_with(inf, 7866),
                _cfg_with(15.2, inf)):
        blk, why = endhigh_density_blocked(cfg, 900.0, 1000.0)
        assert blk is False, f"must fail OPEN on {cfg}: {why}"


def test_endhigh_density_threshold_override_rejects_garbage(monkeypatch):
    """A broken override must fall back to the default, never silently disable
    the gate (: a probe that returns nothing is not a pass)."""
    from optimizer.ils_polish import (endhigh_density_min, endhigh_density_blocked,
                                      ENDHIGH_DENSITY_MIN_DEFAULT)
    monkeypatch.setenv("FPL26_ILS_ENDHIGH_DENSITY_GATE", "1")
    for bad in ("not-a-number", "", "0", "-5"):
        monkeypatch.setenv("FPL26_ILS_ENDHIGH_DENSITY_MIN", bad)
        assert endhigh_density_min() == ENDHIGH_DENSITY_MIN_DEFAULT
        # A malformed override fails closed for a high-density, ~23k-cell design.
        assert endhigh_density_blocked(_cfg_with(131.4, 22946), 900.0, 1000.0)[0] is True
    monkeypatch.setenv("FPL26_ILS_ENDHIGH_DENSITY_MIN", "100")
    assert endhigh_density_min() == 100.0
    # Lowering the threshold admits the high-density path for this ~23k-cell fixture.
    assert endhigh_density_blocked(_cfg_with(131.4, 22946), 900.0, 1000.0)[0] is False


def test_endhigh_density_blocks_the_rotation_path_not_just_the_ladder(monkeypatch):
    """Verify the density gate removes the high-delay directive from the base rotation.

    The gate applies to every scheduling path, not only the place-retry ladder.
    """
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
    """Verify that a disarmed, unaffordable probe leaves the cycle unused.

    Rejecting the first rung must not advance the rotation when the mechanism
    is disabled.
    """
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
    also reserve 250s. Skip-unaffordable advanced to rung 2; the reserve must
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
    """Verify that the ladder stops probing after a rung accepts.

    An acceptance resolves the placement-family choice, so all remaining rungs
    are dropped.
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


# These cases pin seed selection around the recipe-gain cutoff:
# gains below 0.15 use the raw seed; gains above it use recipe_best.
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
    # A 1.035 ns recipe gain remains on the recipe-best seed at the raised threshold.
    assert choose_ils_seed(initial_wns=-1.0, recipe_wns=0.035, cfg=cfg)[0] == "recipe_best"


def test_stuck_threshold_override_default_and_garbage_are_inert(monkeypatch):
    from optimizer.ils_polish import stuck_gain_threshold
    cfg = ILSPolishConfig(enabled=True)
    monkeypatch.delenv("FPL26_ILS_STUCK_GAIN_NS", raising=False)
    assert stuck_gain_threshold(cfg) == cfg.stuck_recipe_gain_ns
    monkeypatch.setenv("FPL26_ILS_STUCK_GAIN_NS", "not-a-number")
    assert stuck_gain_threshold(cfg) == cfg.stuck_recipe_gain_ns


def test_any_forced_probe_is_futility_exempt_not_just_ladder_rungs(monkeypatch):
    """Verify that every forced probe is exempt from futility strikes.

    The exemption applies to single-rung probes as well as ladder rungs,
    preserving eligibility for later cycles.
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
    """Verify that incremental-route escalation runs before route fallback
    sentinels.

    Fallback reroutes can invalidate escalation eligibility, so an eligible
    escalation must claim the cycle first.
    """
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
    """Verify that incremental-route escalation runs when the remaining budget
    cannot cover a full route.

    The escalation claims an otherwise unusable cycle.
    """
    logs = _run_incr_route_priority(monkeypatch, route_anchor_s=400.0,
                                    window_s=300.0)
    assert any("incr-route PRIORITY:" in m for m in logs), logs
    assert any("displaces nothing" in m for m in logs), logs


def test_incr_route_priority_yields_when_a_full_route_still_fits(monkeypatch):
    """Verify that incremental-route escalation yields when the remaining budget
    can cover a full route.

    A from-scratch route takes priority while it remains affordable.
    """
    logs = _run_incr_route_priority(monkeypatch, route_anchor_s=290.0,
                                    window_s=4000.0)
    assert any("incr-route PRIORITY yielded" in m for m in logs), logs
    assert not any("incr-route PRIORITY:" in m for m in logs), logs


def test_incr_route_priority_yields_when_the_route_anchor_is_unknown(monkeypatch):
    """No anchor = cannot PROVE a full route is priced out. Fail safe to the
    earlier shipped order rather than jump on an unmeasured guess."""
    logs = _run_incr_route_priority(monkeypatch, route_anchor_s=0.0,
                                    window_s=4000.0)
    assert not any("incr-route PRIORITY:" in m for m in logs), logs


def test_incr_route_priority_default_on_and_disarms_on_zero(monkeypatch):
    """Verify that incremental-route priority is enabled by default and disabled
    by a zero-valued setting.

    Both the default and the explicit opt-out are part of the configuration
    contract.
    """
    from optimizer.ils_polish import incr_route_first_enabled
    monkeypatch.delenv("FPL26_ILS_INCR_ROUTE_FIRST", raising=False)
    assert incr_route_first_enabled() is True, "eval runs set no flags; this must be on"
    monkeypatch.setenv("FPL26_ILS_INCR_ROUTE_FIRST", "0")
    assert incr_route_first_enabled() is False, "=0 must remain a real kill switch"


def test_ladder_order_by_wns_is_default_on_with_kill_switch(monkeypatch):
    """Verify that WNS-based ladder ordering is enabled by default and retains an
    opt-out.

    Both the default behavior and the kill switch are part of the configuration
    contract.
    """
    from optimizer.ils_polish import ladder_rungs, PLACE_RETRY_LADDER
    monkeypatch.delenv("FPL26_ILS_LADDER_ORDER", raising=False)
    monkeypatch.delenv("FPL26_ILS_LADDER_ORDER_BY_WNS", raising=False)
    # default-ON: a near-met baseline now reorders with nothing set.
    assert ladder_rungs(-0.665, 0.7)[0] == "AltSpreadLogic_medium"
    # Deep negative slack preserves the default ladder order regardless of the flag.
    assert ladder_rungs(-0.959, 0.7) == list(PLACE_RETRY_LADDER)
    # kill switch restores the earlier order exactly.
    monkeypatch.setenv("FPL26_ILS_LADDER_ORDER_BY_WNS", "0")
    assert ladder_rungs(-0.665, 0.7) == list(PLACE_RETRY_LADDER)


def test_ladder_order_by_wns_promotes_medium_only_for_near_met(monkeypatch):
    """Verify that WNS-based ladder ordering promotes medium effort only for
    near-met timing.

    Deep negative slack retains the normal rung order because medium effort is
    not appropriate there.
    """
    from optimizer.ils_polish import ladder_rungs, PLACE_RETRY_LADDER
    monkeypatch.delenv("FPL26_ILS_LADDER_ORDER", raising=False)
    monkeypatch.setenv("FPL26_ILS_LADDER_ORDER_BY_WNS", "1")

    assert ladder_rungs(-0.665, 0.7)[0] == "AltSpreadLogic_medium"   # spam
    assert ladder_rungs(-0.924, 0.7) == list(PLACE_RETRY_LADDER)     # optical
    assert ladder_rungs(-2.153, 0.7) == list(PLACE_RETRY_LADDER)     # 3d
    # promotion REORDERS, never drops or duplicates a rung
    assert sorted(ladder_rungs(-0.665, 0.7)) == sorted(PLACE_RETRY_LADDER)


def test_ladder_order_by_wns_needs_both_inputs(monkeypatch):
    """No baseline (or no near-met magnitude) = no evidence to key on = shipped
    order.
    """
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
    """Verify that the ladder-order setting controls which rung receives the
    unmodified seed.

    The first rung starts from raw input; later rungs may inherit the latest
    accepted checkpoint.
    """
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


# Redraw reserve is opt-in and prevents an exploratory ILS stage from
# consuming time reserved for another wrapper attempt.

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
    # A 600 s window minus the 115 s next-cycle estimate leaves 485 s,
    # below the 550 s reserve; after the minimum cycle count, the guard fires.
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
    """Verify redraw reserve uses the wrapper's unfenced remaining budget.

    The reserve may stop only when wrapper time minus predicted cycle cost
    falls below the threshold, preventing the internal ILS fence from being
    counted twice.
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
    """Verify redraw reserve honors the four-cycle minimum.

    The floor ensures a shallow failing set reaches its fourth candidate before
    time-based reserve logic can stop exploration.
    """
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
