"""Tests for the DEEP-WNS full-replace sibling (jul25 panel, grok-4.5 seat).

The gate is the safety-critical part: it decides whether to spend a full
place+route inside the eval hour. Every unmeasurable input must fail CLOSED.
"""
import asyncio

import pytest

from optimizer.deep_replace_sibling import (
    DEEP_REPLACE_PHYSOPT_FRAC,
    DEEP_REPLACE_COST_MARGIN,
    DEEP_REPLACE_PHYSOPT_DIRECTIVE,
    DEEP_REPLACE_PLACE_DIRECTIVE,
    DEEP_REPLACE_B2_MIN_SLICE_FRAC,
    DEEP_REPLACE_PR_S_PER_NET,
    DEEP_REPLACE_WRITE_IO_S,
    VERDICT_ADOPTED,
    VERDICT_ERROR,
    VERDICT_REJECTED,
    deep_replace_physopt_affordable,
    deep_replace_should_run,
    predict_place_route_s,
    run_deep_replace_sibling,
)

# boom_soc's real Phase-1 physics (rebaseline_jul24/boom_soc_2025.1/run.log)
BOOM = dict(failing_endpoint_count=217_988, wns_magnitude_ns=19.16)
GATE_DEFAULTS = dict(failing_endpoints_min=100_000, wns_min_ns=10.0,
                     finalize_reserve_s=300.0)


def _gate(**over):
    kw = dict(enabled=True, pristine_dcp="/in/boom.dcp",
              remaining_s=3000.0, cost_basis_s=900.0,
              **BOOM, **GATE_DEFAULTS)
    kw.update(over)
    return deep_replace_should_run(**kw)


# ---------------------------------------------------------------- kill switch
def test_disabled_is_a_hard_noop():
    run, why = _gate(enabled=False)
    assert run is False and "disabled" in why


def test_disabled_precedes_every_other_check():
    # even with everything else broken, a disabled stage reports 'disabled'
    run, why = _gate(enabled=False, pristine_dcp=None,
                     failing_endpoint_count=None, cost_basis_s=0)
    assert run is False and "disabled" in why


# ------------------------------------------------------------- physics gating
def test_boom_soc_arms():
    run, why = _gate()
    assert run is True and "DEEP-extreme" in why


def test_below_failing_endpoint_floor_does_not_arm():
    run, why = _gate(failing_endpoint_count=99_999)
    assert run is False and "not DEEP-extreme" in why


def test_below_wns_floor_does_not_arm():
    # ispd16-like: huge failing set but |WNS| 7.75 < 10.0
    run, why = _gate(failing_endpoint_count=242_906, wns_magnitude_ns=7.75)
    assert run is False and "not DEEP-extreme" in why


@pytest.mark.parametrize("failing,wns", [(None, 19.16), (217_988, None),
                                         (None, None)])
def test_unmeasurable_physics_fails_closed(failing, wns):
    run, why = _gate(failing_endpoint_count=failing, wns_magnitude_ns=wns)
    assert run is False and "fail closed" in why


def test_no_pristine_dcp_fails_closed():
    run, why = _gate(pristine_dcp=None)
    assert run is False and "pristine" in why


# --------------------------------------------------------------- affordability
def test_no_cost_anchor_fails_closed():
    run, why = _gate(cost_basis_s=0.0)
    assert run is False and "fail closed" in why


def _need_b1(basis, reserve=300.0):
    """The ARM requirement — the place+route leg ONLY.

    The retiming tail is deliberately not sized here; it gets its own gate
    (deep_replace_physopt_affordable) fed by the MEASURED B1 elapsed time.
    """
    return basis * DEEP_REPLACE_COST_MARGIN + reserve


def test_insufficient_reserve_does_not_arm():
    run, why = _gate(remaining_s=_need_b1(900.0) - 1.0)
    assert run is False and "insufficient terminal reserve" in why


def test_exactly_sufficient_reserve_arms():
    run, _ = _gate(remaining_s=_need_b1(900.0))
    assert run is True


def test_boom_soc_real_measured_recipe_can_arm():
    """REGRESSION — the arithmetic that kept deep_replace=0 on hardware.

    Measured on the 8-vCPU eval-parity box, jul25
    (~/drill/recipe_replay_jul25/boom_soc_replay.log, RPL_STAGE timestamps):

        unplace 33s + place Explore 865s + route 746s = 1644s  (place+route)
        phys_opt AlternateFlowWithRetiming            = 1106s
        total                                         = 2750s of a 3500s wall

    The OLD arm gate demanded ``basis x1.7 x1.3 + reserve``, on a basis that is
    ITSELF x1.15-margined at record time (ils_polish.py combo_cost) -- roughly
    2.54x the true place+route cost, i.e. ~4176s + reserve. Against a 3500s
    wall that is unsatisfiable even at t=0 with the entire budget free, which
    is precisely why the stage never fired on hardware.

    Three margins were each sound in isolation and nonsense composed. Sizing
    the ARM on place+route alone, the real recipe fits with room to spare.
    """
    run, why = _gate(remaining_s=3500.0, cost_basis_s=1644.0)
    assert run is True, why
    assert "place+route anchor" in why
    # The retired stacked figure could not fit the whole wall, let alone a tail.
    old_stacked = (1644.0 * (1.0 + DEEP_REPLACE_PHYSOPT_FRAC)
                   * DEEP_REPLACE_COST_MARGIN + 300.0)
    assert old_stacked > 3500.0


# ------------------------------- size model (recipe-first has no anchor)
def test_size_model_fails_closed_without_a_size():
    for bad in (None, 0, -1):
        est, why = predict_place_route_s(bad)
        assert est == 0.0 and "fail closed" in why


@pytest.mark.parametrize("nets,measured", [(273_573, 1644.0),   # boom_soc
                                           (273_764, 1492.0)])  # boom_soc_v2
def test_size_model_reproduces_its_calibration_points(nets, measured):
    """Both jul25 8-vCPU measurements must land within the run-to-run spread
    the two samples themselves exhibit (9% at identical size). The model is
    deliberately the MAX per-net cost, so it may over- but never under-shoot
    the cheaper sample by more than that spread."""
    est, _ = predict_place_route_s(nets)
    assert est >= measured * 0.90
    assert est <= measured * 1.15


def test_size_model_is_conservative_by_construction():
    """max-of-observed, not mean: over-estimating only declines a run."""
    assert DEEP_REPLACE_PR_S_PER_NET >= 1492.0 / 273_764
    assert DEEP_REPLACE_PR_S_PER_NET >= 1644.0 / 273_573


def test_size_model_flags_extrapolation_away_from_calibration():
    """vtr_mcml is 3.9x smaller than anything we calibrated on; a prediction
    there must SAY it is extrapolating so a surprise is traceable."""
    est, why = predict_place_route_s(70_431)
    assert 380.0 < est < 460.0, est
    assert "EXTRAPOLATED" in why
    # ...and the calibrated range must NOT carry the warning
    _, why_in = predict_place_route_s(273_573)
    assert "EXTRAPOLATED" not in why_in


def test_size_model_feeds_the_arm_gate_at_t0():
    """End-to-end: with no measured anchor, the size model alone must let
    boom_soc arm inside a 3500s wall — the recipe-first precondition."""
    est, _ = predict_place_route_s(273_573)
    run, why = _gate(remaining_s=3500.0, cost_basis_s=est)
    assert run is True, why


# ------------------------------------------ second gate: the retiming tail
def test_physopt_gate_fails_closed_without_a_measurement():
    ok, why = deep_replace_physopt_affordable(
        measured_place_route_s=0.0, remaining_s=9999.0,
        finalize_reserve_s=300.0)
    assert ok is False and "fail closed" in why


@pytest.mark.parametrize("tag,pr,remaining,actual_physopt", [
    ("chain12_boom_soc", 1532.0, 1481.0, 1106.0),
    ("chain15_att1",     1508.0, 1595.0,  805.0),
    ("chain15_att2",     1508.0, 1148.0,  805.0),
])
def test_all_three_wrongly_declined_runs_now_arm(tag, pr, remaining,
                                                 actual_physopt):
    """REGRESSION, n=3 on hardware. The old predictive gate refused the tail in
    every run we have data for, and in every one the tail would have COMPLETED:

        run                 old "need"   had      actual   would have fit by
        chain12 boom_soc      1724 s   1481 s    1106 s        +375 s
        chain15 att.1         1702 s   1595 s     805 s        +460 s
        chain15 att.2         1702 s   1148 s     805 s         +13 s

    ~1.5x over-estimate, worth ~6.5 MHz on the SCORED boom_soc_v2 benchmark.
    Each must now arm AND leave room for the observed cost.
    """
    ok, why = deep_replace_physopt_affordable(
        measured_place_route_s=pr, remaining_s=remaining,
        finalize_reserve_s=300.0)
    assert ok is True, f"{tag}: {why}"
    usable = remaining - 300.0 - DEEP_REPLACE_WRITE_IO_S
    assert usable >= actual_physopt, (
        f"{tag}: usable {usable:.0f}s must cover the observed "
        f"{actual_physopt:.0f}s phys_opt")


def test_physopt_declines_when_reserves_alone_exceed_the_wall():
    ok, why = deep_replace_physopt_affordable(
        measured_place_route_s=1644.0, remaining_s=200.0,
        finalize_reserve_s=300.0)
    assert ok is False and "does not even cover finalize" in why


def test_physopt_declines_a_slice_too_small_to_complete_a_pass():
    """The floor is calibrated from measured phys_opt/place+route ratios
    (0.673, 0.539, 0.227, 0.206). Below 0.25 no observed pass has completed."""
    pr = 1600.0
    reserve = 300.0 + DEEP_REPLACE_WRITE_IO_S
    just_under = pr * DEEP_REPLACE_B2_MIN_SLICE_FRAC - 1.0
    ok, why = deep_replace_physopt_affordable(
        measured_place_route_s=pr, remaining_s=just_under + reserve,
        finalize_reserve_s=300.0)
    assert ok is False and "minimum useful slice" in why
    ok, _ = deep_replace_physopt_affordable(
        measured_place_route_s=pr,
        remaining_s=pr * DEEP_REPLACE_B2_MIN_SLICE_FRAC + reserve,
        finalize_reserve_s=300.0)
    assert ok is True


def test_the_floor_sits_below_every_measured_ratio():
    """Guards the calibration: the floor must never exceed the smallest ratio
    at which a real phys_opt pass has been observed to complete."""
    measured_ratios = [0.673, 0.539, 0.227, 0.206]
    assert DEEP_REPLACE_B2_MIN_SLICE_FRAC <= min(measured_ratios)


# ------------------------------------------------------------------ execution
class _Vivado:
    """Minimal fake: records commands, returns scripted measurements."""

    def __init__(self, wns, unrouted=0, whs=0.05, fail_on=None):
        self.cmds = []
        self.wns, self.unrouted, self.whs = wns, unrouted, whs
        self.fail_on = fail_on

    async def call_tool(self, name, args):
        cmd = args.get("command", name)
        self.cmds.append(cmd)
        if self.fail_on and self.fail_on in cmd:
            return "TCL ERROR: synthetic failure"
        return "OK"


def _run(v, chain_best=-10.399, **over):
    async def measure(call_tool, tcl, timeout_s=None):
        return v.wns, v.unrouted

    async def measure_hold(call_tool, timeout_s=None):
        return v.whs

    kw = dict(pristine_dcp="/in/boom.dcp", chain_best_wns=chain_best,
              deadline_ts=__import__("time").time() + 5000,
              wns_tcl="get_wns", out_dcp="/out/cand.dcp",
              log=lambda m: None, measure=measure, measure_hold=measure_hold,
              tool_ok=lambda r: "ERROR" not in str(r))
    kw.update(over)
    return asyncio.run(run_deep_replace_sibling(v.call_tool, **kw))


def test_replays_the_logged_winning_sequence_from_pristine():
    v = _Vivado(wns=-10.378)
    r = _run(v)
    joined = " | ".join(v.cmds)
    # opens the PRISTINE input, not a banked best — the distinguishing feature
    assert "open_checkpoint {/in/boom.dcp}" in joined
    assert "place_design -unplace" in joined
    assert f"place_design -directive {DEEP_REPLACE_PLACE_DIRECTIVE}" in joined
    assert "route_design" in joined
    assert f"phys_opt_design -directive {DEEP_REPLACE_PHYSOPT_DIRECTIVE}" in joined
    # ordering: unplace before place before route before phys_opt
    assert (v.cmds.index("place_design -unplace")
            < v.cmds.index(f"place_design -directive {DEEP_REPLACE_PLACE_DIRECTIVE}"))
    assert r.verdict == VERDICT_ADOPTED


def test_pristine_dcp_is_never_a_write_target():
    v = _Vivado(wns=-10.378)
    _run(v)
    writes = [c for c in v.cmds if "write_checkpoint" in c]
    assert writes and all("/in/boom.dcp" not in c for c in writes)


def test_unrouted_result_is_rejected():
    v = _Vivado(wns=-10.0, unrouted=4211)
    r = _run(v)
    assert r.verdict == VERDICT_REJECTED and "not fully routed" in r.reason
    assert not any("write_checkpoint" in c for c in v.cmds)


def test_hold_violation_is_rejected():
    v = _Vivado(wns=-10.0, whs=-0.03)
    r = _run(v)
    assert r.verdict == VERDICT_REJECTED and "hold" in r.reason


def test_worse_than_chain_best_is_rejected():
    v = _Vivado(wns=-12.0)          # worse than chain best -10.399
    r = _run(v)
    assert r.verdict == VERDICT_REJECTED and "does not beat" in r.reason
    assert not any("write_checkpoint" in c for c in v.cmds)


def test_better_than_chain_best_is_adopted_and_written():
    # boom_soc: chain best -10.399 (our +13.32) vs record -10.378 (+35.47)
    v = _Vivado(wns=-10.378)
    r = _run(v)
    assert r.verdict == VERDICT_ADOPTED
    assert r.candidate_path == "/out/cand.dcp"
    assert any("write_checkpoint -force {/out/cand.dcp}" in c for c in v.cmds)


def test_tcl_failure_is_caught_not_raised():
    v = _Vivado(wns=-10.0, fail_on="place_design -unplace")
    r = _run(v)
    assert r.verdict == VERDICT_ERROR and not r.registered


def test_incremental_reroute_on_leftover_nets():
    """Post-route phys_opt can leave nets open; one completion pass runs."""
    class _V(_Vivado):
        def __init__(self):
            super().__init__(wns=-10.378)
            self.calls = 0

    v = _V()
    seq = [(-10.0, 17), (-10.378, 0)]

    async def measure(call_tool, tcl, timeout_s=None):
        out = seq[min(v.calls, len(seq) - 1)]
        v.calls += 1
        return out

    async def measure_hold(call_tool, timeout_s=None):
        return 0.05

    r = _run(v, measure=measure, measure_hold=measure_hold)
    assert r.verdict == VERDICT_ADOPTED
    assert v.cmds.count("route_design") >= 2


# ------------------------------- self-capped-stage bypass (unroute gate)
def test_stage_runs_under_the_self_capped_bypass_and_restores_it(tmp_path,
                                                                 monkeypatch):
    """REGRESSION — the first live E2E run (jul25) died here.

    The unroute gate refused the recipe's own route_design:
        unroute_gate_refused: predicted re-route 2274.0s ... exceeds effective
        budget 2094.8s ... place=827s route=0s banked=none
    The gate exists to stop a ROUTED state being destroyed unrecoverably, but
    the recipe has deliberately unplaced by then (nothing routed to protect),
    and its size-scaled estimate said 2274 s for a route that MEASURES 746 s on
    that design. The stage is self-capped by deadline_ts, which is exactly the
    condition the gate's own comment exempts for ILS.

    The flag must also be RESTORED, not cleared: the tail call site can already
    be inside a real ILS stage.
    """
    import optimizer.deep_replace_sibling as drs
    from dcp_optimizer import DCPOptimizer
    from optimizer.ils_polish import ILSPolishConfig

    seen = {}

    async def _fake_run(call_tool, **kw):
        seen["flag_during"] = stub._in_ils_stage
        from optimizer.deep_replace_sibling import DeepReplaceResult
        return DeepReplaceResult(attempted=True, verdict=VERDICT_REJECTED,
                                 reason="stubbed")

    monkeypatch.setattr(drs, "run_deep_replace_sibling", _fake_run)

    dcp = tmp_path / "pristine.dcp"
    dcp.write_bytes(b"x")

    class _Stub:
        def __init__(self):
            self._ils_polish_cfg = ILSPolishConfig()
            self._ils_polish_cfg.deep_replace_enabled = True
            self.input_dcp_path = dcp
            self._ils_observed_costs = None
            self.run_dir = tmp_path
            self.best_wns = -19.162
            self._in_ils_stage = False
            self.call_tool = None

        def _phase1_wns_for_features(self):
            return -19.162

        def _phase1_failing_endpoints_for_features(self):
            return 217_988

    stub = _Stub()
    # Pre-set to False; the stage must set True during and restore False after.
    asyncio.run(DCPOptimizer._deep_replace_sibling_after_polish(
        stub, __import__("time").time() + 3500, "get_wns",
        stage_label="first", cost_basis_override=1644.0,
        basis_note="size model"))
    assert seen.get("flag_during") is True, "stage must run under the bypass"
    assert stub._in_ils_stage is False, "flag must be restored after"


def test_self_capped_bypass_restores_a_prior_true_flag(tmp_path, monkeypatch):
    """Tail call site: already inside a real ILS stage => must stay True."""
    import optimizer.deep_replace_sibling as drs
    from dcp_optimizer import DCPOptimizer
    from optimizer.ils_polish import ILSPolishConfig

    async def _fake_run(call_tool, **kw):
        from optimizer.deep_replace_sibling import DeepReplaceResult
        return DeepReplaceResult(attempted=True, verdict=VERDICT_REJECTED,
                                 reason="stubbed")

    monkeypatch.setattr(drs, "run_deep_replace_sibling", _fake_run)
    dcp = tmp_path / "pristine.dcp"
    dcp.write_bytes(b"x")

    class _Stub:
        def __init__(self):
            self._ils_polish_cfg = ILSPolishConfig()
            self._ils_polish_cfg.deep_replace_enabled = True
            self.input_dcp_path = dcp
            self._ils_observed_costs = None
            self.run_dir = tmp_path
            self.best_wns = -19.162
            self._in_ils_stage = True          # already inside ILS
            self.call_tool = None

        def _phase1_wns_for_features(self):
            return -19.162

        def _phase1_failing_endpoints_for_features(self):
            return 217_988

    stub = _Stub()
    asyncio.run(DCPOptimizer._deep_replace_sibling_after_polish(
        stub, __import__("time").time() + 3500, "get_wns",
        stage_label="tail", cost_basis_override=1644.0))
    assert stub._in_ils_stage is True, "must restore the prior value, not False"


# --------------------------------------- state hand-forward to the pipeline
def _promotion_stub(tmp_path, monkeypatch, *, verdict, post_wns,
                    prior_best=-19.162):
    """Drive _deep_replace_sibling_after_polish with a stubbed stage result."""
    import optimizer.deep_replace_sibling as drs
    from dcp_optimizer import DCPOptimizer
    from optimizer.deep_replace_sibling import DeepReplaceResult
    from optimizer.ils_polish import ILSPolishConfig

    cand = tmp_path / "deep_replace_candidate.dcp"
    cand.write_bytes(b"candidate-bytes")

    async def _fake_run(call_tool, **kw):
        return DeepReplaceResult(attempted=True, verdict=verdict,
                                 post_wns=post_wns, post_whs=0.01,
                                 unrouted=0, candidate_path=str(cand),
                                 stage_banked="routed", reason="stubbed")

    monkeypatch.setattr(drs, "run_deep_replace_sibling", _fake_run)
    dcp = tmp_path / "pristine.dcp"
    dcp.write_bytes(b"x")

    class _Stub:
        def __init__(self):
            self._ils_polish_cfg = ILSPolishConfig()
            self._ils_polish_cfg.deep_replace_enabled = True
            self.input_dcp_path = dcp
            self._ils_observed_costs = None
            self.run_dir = tmp_path
            self.best_wns = prior_best
            self._best_valid_dcp = None
            self._best_valid_dcp_wns = None
            self._best_valid_mirror_size = None
            self._in_ils_stage = False
            self.call_tool = None

        def _phase1_wns_for_features(self):
            return -19.162

        def _phase1_failing_endpoints_for_features(self):
            return 217_988

        async def register_final_candidate(self, path, wns, label, **kw):
            return True

    stub = _Stub()
    asyncio.run(DCPOptimizer._deep_replace_sibling_after_polish(
        stub, __import__("time").time() + 3500, "get_wns",
        stage_label="first", cost_basis_override=1644.0))
    return stub, cand


def test_adopted_result_is_promoted_to_the_pipeline_best(tmp_path, monkeypatch):
    """REGRESSION — chain12 (jul25), the first live E2E run.

    The recipe banked wns=-10.256 (84.57 MHz) into the MUX, but the pipeline's
    best stayed at the baseline, so the log then read:
        "running finalize with best_wns=-19.162"
        bare-reroute polish: opening best_valid...   <- the BASELINE DCP
        [tail-ctrl] ARMED: entry wns=-19.162
        verdict=ERROR gain=+0.000ns   (after 925 s)
    925 s of tail work on a design already beaten by 36 MHz. The MUX shipped
    the right DCP so the run looked fine — the waste was invisible.

    replace_gamble, this stage's declared sibling, has always promoted. Now
    deep-replace does too, so the tail compounds on the recipe's result.
    """
    stub, cand = _promotion_stub(tmp_path, monkeypatch,
                                 verdict=VERDICT_ADOPTED, post_wns=-10.256)
    assert stub.best_wns == -10.256
    assert stub._best_valid_dcp == cand
    assert stub._best_valid_dcp_wns == -10.256
    assert stub._best_valid_mirror_size == len(b"candidate-bytes")


def test_rejected_result_never_touches_the_pipeline_best(tmp_path, monkeypatch):
    stub, _ = _promotion_stub(tmp_path, monkeypatch,
                              verdict=VERDICT_REJECTED, post_wns=-25.0)
    assert stub.best_wns == -19.162
    assert stub._best_valid_dcp is None


def test_promotion_requires_a_strict_improvement(tmp_path, monkeypatch):
    """The stage's adopt gate compared against chain_best_wns captured at CALL
    time; something else may have banked better since. Re-check before
    overwriting the pipeline best."""
    stub, _ = _promotion_stub(tmp_path, monkeypatch,
                              verdict=VERDICT_ADOPTED, post_wns=-12.0,
                              prior_best=-9.0)      # already better
    assert stub.best_wns == -9.0, "must not regress the pipeline best"
    assert stub._best_valid_dcp is None


# ------------------------------------------- recipe-first hook (task #13)
class _HookStub:
    """Minimal stand-in for DCPOptimizer: the hook only touches these."""

    def __init__(self, *, enabled, first, cells=377_972, raises=False):
        from optimizer.ils_polish import ILSPolishConfig
        self._ils_polish_cfg = ILSPolishConfig()
        self._ils_polish_cfg.deep_replace_enabled = enabled
        self._ils_polish_cfg.deep_replace_first_enabled = first
        self._input_cell_count = cells
        self.calls = []
        self._raises = raises

    def _budget_remaining(self):
        return 3500.0

    def _wns_tcl_for_stages(self):
        return "get_wns"

    async def _deep_replace_sibling_after_polish(self, deadline, wns_tcl, **kw):
        if self._raises:
            raise RuntimeError("synthetic stage failure")
        self.calls.append(kw)


def _run_hook(stub):
    from dcp_optimizer import DCPOptimizer
    asyncio.run(DCPOptimizer._maybe_run_recipe_first(stub))
    return stub


@pytest.mark.parametrize("enabled,first", [(False, False), (False, True),
                                           (True, False)])
def test_recipe_first_is_a_hard_noop_unless_both_flags_set(enabled, first):
    """'first' only REORDERS an enabled stage — it must never enable one."""
    stub = _run_hook(_HookStub(enabled=enabled, first=first))
    assert stub.calls == []


def test_recipe_first_runs_with_both_flags_and_passes_the_size_basis():
    stub = _run_hook(_HookStub(enabled=True, first=True))
    assert len(stub.calls) == 1
    kw = stub.calls[0]
    assert kw["stage_label"] == "first"
    # boom_soc's 377,972 cells must reproduce the measured 1644 s place+route
    assert 1600.0 < kw["cost_basis_override"] < 1700.0
    assert "cells" in kw["basis_note"]


def test_recipe_first_fails_closed_without_a_cell_count():
    """No size feature => no basis => the stage must not start. This is the
    t=0 analogue of 'no measured cost anchor'."""
    for bad in (None, 0):
        stub = _run_hook(_HookStub(enabled=True, first=True, cells=bad))
        assert stub.calls == []


def test_recipe_first_never_sinks_the_run():
    """The LLM loop still has the whole wall behind this hook, so a stage
    failure must be swallowed, not propagated."""
    stub = _HookStub(enabled=True, first=True, raises=True)
    _run_hook(stub)          # must not raise
    assert stub.calls == []


# --------------------------------------------------------- staging B1 / B2
def test_b1_is_banked_before_the_retiming_tail_runs():
    """The whole point of the split: the routed gain reaches disk BEFORE the
    tail is allowed to touch anything."""
    v = _Vivado(wns=-10.378)
    _run(v)
    first_write = next(i for i, c in enumerate(v.cmds)
                       if "write_checkpoint" in c)
    physopt = next(i for i, c in enumerate(v.cmds) if "phys_opt_design" in c)
    assert first_write < physopt


def test_b2_failure_keeps_the_b1_candidate():
    """gemini's C2 dissent, pinned: an erroring tail must not cost the routed
    gain, and must not require a reload to recover it."""
    v = _Vivado(wns=-10.378, fail_on="phys_opt_design")
    r = _run(v)
    assert r.verdict == VERDICT_ADOPTED
    assert r.candidate_path == "/out/cand.dcp"
    assert r.stage_banked == "routed"
    assert r.b1_wns == -10.378
    assert "B2 raised" in r.reason
    # exactly one write — B1's. No reload, no rewrite.
    assert len([c for c in v.cmds if "write_checkpoint" in c]) == 1
    assert not any("open_checkpoint {/out/cand.dcp}" in c for c in v.cmds)


def test_b2_improvement_overwrites_the_candidate():
    """The real boom_soc numbers: routed -10.256 then retimed -9.583."""
    v = _Vivado(wns=-10.378)
    seq = [(-10.256, 0), (-9.583, 0)]
    calls = {"n": 0}

    async def measure(call_tool, tcl, timeout_s=None):
        out = seq[min(calls["n"], len(seq) - 1)]
        calls["n"] += 1
        return out

    async def measure_hold(call_tool, timeout_s=None):
        return 0.05

    r = _run(v, measure=measure, measure_hold=measure_hold)
    assert r.stage_banked == "physopt_retime"
    assert r.b1_wns == -10.256 and r.post_wns == -9.583
    assert len([c for c in v.cmds if "write_checkpoint" in c]) == 2


def test_b2_regression_keeps_b1_and_does_not_rewrite():
    """A tail that measures WORSE than B1 loses silently; B1 stays shipped."""
    v = _Vivado(wns=-10.378)
    seq = [(-10.256, 0), (-11.900, 0)]      # B2 regressed
    calls = {"n": 0}

    async def measure(call_tool, tcl, timeout_s=None):
        out = seq[min(calls["n"], len(seq) - 1)]
        calls["n"] += 1
        return out

    async def measure_hold(call_tool, timeout_s=None):
        return 0.05

    r = _run(v, measure=measure, measure_hold=measure_hold)
    assert r.stage_banked == "routed"
    assert r.post_wns == -10.256
    assert len([c for c in v.cmds if "write_checkpoint" in c]) == 1


def test_b2_declined_when_unaffordable_keeps_b1():
    """No wall for the tail => B1 ships and phys_opt never runs at all."""
    import time as _t
    v = _Vivado(wns=-10.378)
    r = _run(v, deadline_ts=_t.time() + 1.0)
    assert r.verdict == VERDICT_ADOPTED and r.stage_banked == "routed"
    assert not any("phys_opt_design" in c for c in v.cmds)
    assert "declined" in r.b2_reason
