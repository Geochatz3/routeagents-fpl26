"""Tests for the B3 small-floor sibling leg (aug09, v5.3).

Discipline notes:
- The execution tests drive the REAL run_deep_replace_sibling with a fake
  Vivado whose heavy steps take real (small) wall time, so the B3 arm gate
  passes through its own arithmetic — nothing injects a synthetic trigger
  (the null-by-construction rule from the wedge-guard episode).
- The affordability tests pin the arithmetic to the measured anchors that
  justify the lever: mini-ISP B1 place+route ~75 s arms; the next-smallest
  mid-band knowns (3d 260 s, finn 461 s, corescore ~640 s) decline, which is
  the whole generalization contract.
"""
import asyncio
import time

import pytest

from optimizer.deep_replace_sibling import (
    DEEP_REPLACE_B3_COST_MULT,
    DEEP_REPLACE_B3_FIXED_OVERHEAD_S,
    DEEP_REPLACE_B3_MAX_COST_S,
    DEEP_REPLACE_B3_PLACE_DIRECTIVE,
    DEEP_REPLACE_B3_PHYSOPT_DIRECTIVE,
    DEEP_REPLACE_B3_ROUTE_DIRECTIVE,
    DEEP_REPLACE_WRITE_IO_S,
    deep_replace_b3_affordable,
    run_deep_replace_sibling,
)


def _est(pr):
    return pr * DEEP_REPLACE_B3_COST_MULT + DEEP_REPLACE_B3_FIXED_OVERHEAD_S


# ------------------------------------------------------------- affordability
# v5.5.3: affordability ONLY. The class decision moved to the physics
# admission attestation (tests in test_logic_floor.py) after the AWS
# eval-parity run showed the dev-calibrated wall-clock separation INVERTS on
# the contest box (mini-ISP 145s vs vexriscv 130s there; 75-78s vs 92-97s measured on dev).
def test_miniisp_dev_anchor_passes_affordability():
    ok, why = deep_replace_b3_affordable(
        measured_place_route_s=75.0, remaining_s=3000.0,
        finalize_reserve_s=300.0)
    assert ok is True and "B3 armed" in why


def test_miniisp_EVAL_anchor_passes_affordability():
    """THE aug10 fix: the contest instance measured mini-ISP's B1 at 145s
    (est 409); the old 300s cap priced the eval box out of an unchanged
    physics. 145s must now pass with margin."""
    assert _est(145.0) <= DEEP_REPLACE_B3_MAX_COST_S
    ok, why = deep_replace_b3_affordable(
        measured_place_route_s=145.0, remaining_s=3000.0,
        finalize_reserve_s=300.0)
    assert ok is True and "B3 armed" in why


@pytest.mark.parametrize("pr,tag", [
    # MEASURED values: these pass AFFORDABILITY now — the physics admission
    # (not tested here) is what refuses them. The assert is that the reason
    # string no longer claims a class decision.
    (92.0, "vexriscv-v50"), (93.0, "vexriscv-v52"),
    (130.0, "vexriscv-AWS-eval"), (183.0, "optical-v50"),
])
def test_mid_anchors_pass_affordability_class_moved_to_physics(pr, tag):
    ok, why = deep_replace_b3_affordable(
        measured_place_route_s=pr, remaining_s=3400.0,
        finalize_reserve_s=300.0)
    assert ok is True, f"{tag} passes affordability (physics decides class)"
    assert "anchor class" not in why


@pytest.mark.parametrize("pr,tag", [
    (260.0, "3d-v50"), (368.0, "finn-v50"), (2021.0, "ispd16-v52"),
])
def test_large_designs_decline_on_the_cost_cap(pr, tag):
    ok, why = deep_replace_b3_affordable(
        measured_place_route_s=pr, remaining_s=3400.0,
        finalize_reserve_s=300.0)
    assert ok is False, f"{tag} must decline"
    assert "exceeds small-design cap" in why


def test_cap_boundary_is_the_documented_arithmetic():
    # est(pr) = 2.2*pr + 90 <= 550  <=>  pr <= ~209s: the implicit size bound.
    assert _est(145.0) <= DEEP_REPLACE_B3_MAX_COST_S   # eval mini-ISP, margin
    assert _est(209.0) <= DEEP_REPLACE_B3_MAX_COST_S   # boundary inside
    assert _est(210.0) > DEEP_REPLACE_B3_MAX_COST_S    # boundary outside


@pytest.mark.parametrize("pr", [None, 0.0, -1.0])
def test_unmeasured_anchor_fails_closed(pr):
    ok, why = deep_replace_b3_affordable(
        measured_place_route_s=pr, remaining_s=3000.0,
        finalize_reserve_s=300.0)
    assert ok is False and "fail closed" in why


def test_tight_remaining_declines_on_the_slice():
    pr = 75.0
    need = _est(pr) + 300.0 + DEEP_REPLACE_WRITE_IO_S
    ok, why = deep_replace_b3_affordable(
        measured_place_route_s=pr, remaining_s=need - 1.0,
        finalize_reserve_s=300.0)
    assert ok is False and "usable slice" in why
    ok, _ = deep_replace_b3_affordable(
        measured_place_route_s=pr, remaining_s=need,
        finalize_reserve_s=300.0)
    assert ok is True


# ------------------------------------------------------------------ execution
import re


class _Vivado:
    """Fake Vivado: records commands; heavy steps consume real wall time so
    the B3 gate's measured-anchor arithmetic runs for real. write_checkpoint
    CREATES the target file (with a marker payload naming the writing leg) so
    the B3 temp+rename adopt path exercises the real os.replace."""

    def __init__(self, wns_seq, unrouted=0, whs=0.05, fail_on=None,
                 step_s=0.05):
        self.cmds = []
        self.wns_seq = list(wns_seq)
        self.unrouted, self.whs = unrouted, whs
        self.fail_on = fail_on
        self.step_s = step_s

    async def call_tool(self, name, args):
        cmd = args.get("command", name)
        self.cmds.append(cmd)
        if self.fail_on and self.fail_on in cmd:
            return "TCL ERROR: synthetic failure"
        if cmd.startswith(("place_design", "route_design", "phys_opt_design")):
            await asyncio.sleep(self.step_s)
        if cmd.startswith("write_checkpoint"):
            m = re.search(r"\{(.+)\}", cmd)
            if m:
                import pathlib
                p = pathlib.Path(m.group(1))
                if p.parent.is_dir():
                    payload = ("b3" if ".b3tmp" in p.name
                               else f"leg{sum(1 for c in self.cmds if c.startswith('write_checkpoint'))}")
                    p.write_text(payload)
        return "OK"

    def next_wns(self):
        return self.wns_seq.pop(0) if len(self.wns_seq) > 1 else self.wns_seq[0]


async def _admit_yes(wns_banked):
    return True, "B3 physics-admitted: test stub"


async def _admit_no(wns_banked):
    return False, "B3 declined (physics admission): test stub refusal"


def _run(v, chain_best=-10.399, b3_enabled=True, out_dir=None, **over):
    async def measure(call_tool, tcl, timeout_s=None):
        return v.next_wns(), v.unrouted

    async def measure_hold(call_tool, timeout_s=None):
        return v.whs

    out_dcp = (str(out_dir / "cand.dcp") if out_dir is not None
               else "/out/cand.dcp")
    kw = dict(pristine_dcp="/in/mini.dcp", chain_best_wns=chain_best,
              deadline_ts=time.time() + 5000,
              wns_tcl="get_wns", out_dcp=out_dcp,
              log=lambda m: None, measure=measure, measure_hold=measure_hold,
              tool_ok=lambda r: "ERROR" not in str(r),
              b3_enabled=b3_enabled, b3_admission=_admit_yes)
    kw.update(over)
    return asyncio.run(run_deep_replace_sibling(v.call_tool, **kw))


def _b3_cmds(v):
    return [c for c in v.cmds
            if DEEP_REPLACE_B3_PLACE_DIRECTIVE in c
            or DEEP_REPLACE_B3_PHYSOPT_DIRECTIVE in c
            or DEEP_REPLACE_B3_ROUTE_DIRECTIVE in c]


def test_default_off_issues_no_b3_commands():
    v = _Vivado([-0.904])
    r = _run(v, b3_enabled=False)
    assert r.stage_banked in ("routed", "physopt_retime")
    assert _b3_cmds(v) == []
    assert r.b3_reason == "" and r.b3_wns is None


def test_b3_improvement_is_adopted_and_written(tmp_path):
    # measures: B1 -0.904, B2 -0.904 (declined to write), B3 -0.850 (adopt)
    v = _Vivado([-0.904, -0.904, -0.850])
    r = _run(v, out_dir=tmp_path)
    assert r.stage_banked == "small_floor"
    assert r.post_wns == -0.850 and r.b3_wns == -0.850
    # the od4 chain ran in order, from the pristine checkpoint
    joined = "\n".join(v.cmds)
    i_open2 = joined.rindex("open_checkpoint {/in/mini.dcp}")
    i_pl = joined.index(f"place_design -directive "
                        f"{DEEP_REPLACE_B3_PLACE_DIRECTIVE}")
    i_po = joined.index(f"phys_opt_design -directive "
                        f"{DEEP_REPLACE_B3_PHYSOPT_DIRECTIVE}")
    i_rt = joined.index(f"route_design -directive "
                        f"{DEEP_REPLACE_B3_ROUTE_DIRECTIVE}")
    assert i_open2 < i_pl < i_po < i_rt
    assert sum(1 for c in v.cmds if c.startswith("write_checkpoint")) == 2
    # temp+rename: the B3 write targeted the .b3tmp path, and after the
    # os.replace the registered path holds B3's bytes
    b3_writes = [c for c in v.cmds if c.startswith("write_checkpoint")
                 and ".b3tmp" in c]
    assert len(b3_writes) == 1
    assert (tmp_path / "cand.dcp").read_text() == "b3"
    assert not (tmp_path / "cand.b3tmp.dcp").exists()


def test_b3_failed_adopt_write_keeps_the_banked_bytes(tmp_path):
    """Review-2 MAJOR: a write failure during B3 adopt must leave out_dcp
    holding the banked B1 bytes (temp+rename), and the 'intact' log true."""
    v = _Vivado([-0.904, -0.904, -0.850], fail_on=".b3tmp")
    r = _run(v, out_dir=tmp_path)
    assert r.stage_banked == "routed"
    assert r.post_wns == -0.904
    assert (tmp_path / "cand.dcp").read_text() != "b3"  # B1's payload intact


def test_b3_regression_keeps_the_banked_candidate():
    v = _Vivado([-0.904, -0.904, -0.933])
    r = _run(v)
    assert r.stage_banked == "routed"
    assert r.post_wns == -0.904
    assert sum(1 for c in v.cmds if c.startswith("write_checkpoint")) == 1


def test_b3_step_failure_never_costs_the_banked_result():
    v = _Vivado([-0.904, -0.904],
                fail_on=DEEP_REPLACE_B3_PLACE_DIRECTIVE)
    r = _run(v)
    assert r.stage_banked == "routed"
    assert r.post_wns == -0.904
    assert sum(1 for c in v.cmds if c.startswith("write_checkpoint")) == 1


def test_b3_reached_even_when_b2_declines():
    # A huge finalize reserve makes B2's slice negative -> B2 declines and
    # returns early; the B3 hook must still be reached on that path (its own
    # gate then declines for the same reason, which is the assert).
    v = _Vivado([-0.904])
    r = _run(v, finalize_reserve_s=10_000.0)
    assert "B3 declined" in r.b3_reason


def test_no_admission_attestor_fails_closed():
    """v5.5.3: affordability alone must NEVER arm B3 — a missing physics
    attestor declines (review-1's blocker, now structural)."""
    v = _Vivado([-0.904, -0.904, -0.850])
    r = _run(v, b3_admission=None)
    assert r.stage_banked == "routed"
    assert "no physics admission attestor" in r.b3_reason
    assert _b3_cmds(v) == []


def test_refusing_admission_declines_and_runs_no_b3_commands():
    v = _Vivado([-0.904, -0.904, -0.850])
    r = _run(v, b3_admission=_admit_no)
    assert r.stage_banked == "routed"
    assert "physics admission" in r.b3_reason
    assert _b3_cmds(v) == []


def test_admitting_attestation_receives_the_banked_wns(tmp_path):
    seen = []

    async def _admit_capture(wns_banked):
        seen.append(wns_banked)
        return True, "B3 physics-admitted: capture stub"

    v = _Vivado([-0.904, -0.904, -0.850])
    r = _run(v, b3_admission=_admit_capture, out_dir=tmp_path)
    assert r.stage_banked == "small_floor"
    assert seen == [-0.904]


def test_pristine_is_never_a_write_target():
    v = _Vivado([-0.904, -0.904, -0.850])
    _run(v)
    for c in v.cmds:
        if c.startswith("write_checkpoint"):
            assert "/in/mini.dcp" not in c


def test_b3_budget_exhaustion_aborts_and_keeps_the_banked_result():
    """Reserve-aware budget (review-1 MAJOR): with the run deadline close
    enough that remaining - finalize_reserve - write_io < 30s, every B3 grant
    must RAISE (caught inside _run_b3) rather than hand out a 60s floor past
    the reserve. B1 has already banked by then, so the run result stands."""
    v = _Vivado([-0.904, -0.904])
    # B1+B2 take ~0.2s of fake wall; leave ~340s so B1/B2 fit but B3's
    # reserve-aware remaining (340 - 300 - 30 = 10s) is under the 30s floor.
    r = _run(v, deadline_ts=time.time() + 340.0)
    assert r.stage_banked == "routed"
    assert r.post_wns == -0.904
    assert _b3_cmds(v) == []  # no B3 tcl ever issued
    assert sum(1 for c in v.cmds if c.startswith("write_checkpoint")) == 1


def test_b3_repair_route_is_bounded_by_the_b3_budget(tmp_path):
    """The unrouted-repair inside _measure_and_gate must use the B3 grant,
    not the run-level heavy timeout, when called from B3 (review-1 MAJOR).
    Detect via the repair route_design's timeout arg."""
    class _V(_Vivado):
        def __init__(self):
            super().__init__([-0.904, -0.904, -0.850])
            self.timeouts = []
            self.calls = 0

        async def call_tool(self, name, args):
            self.timeouts.append((args.get("command", name),
                                  args.get("timeout")))
            return await super().call_tool(name, args)

    v = _V()
    # unrouted on the B3 measure only: first two gates see 0, B3's sees 2
    seq = iter([0, 0, 2, 0])

    async def measure(call_tool, tcl, timeout_s=None):
        return v.next_wns(), next(seq)

    async def measure_hold(call_tool, timeout_s=None):
        return v.whs

    r = _run(v, measure=measure, measure_hold=measure_hold, out_dir=tmp_path)
    # B1's main route is ALSO a bare `route_design` (heavy grant ~1800s), so
    # only bare routes issued AFTER B3's place step are the repair. Those must
    # carry the B3 grant (b3_deadline = est*1.5 floor 240s << heavy 1800s).
    i_b3_place = next(i for i, (c, _) in enumerate(v.timeouts)
                      if DEEP_REPLACE_B3_PLACE_DIRECTIVE in c)
    repair = [t for c, t in v.timeouts[i_b3_place:] if c == "route_design"]
    assert repair, "repair route was not exercised"
    assert all(t is not None and t < 700.0 for t in repair), repair
    assert r.stage_banked == "small_floor" and r.post_wns == -0.850
