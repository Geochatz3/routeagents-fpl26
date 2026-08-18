"""Phantom-best guard (jul05, preview #10 v2 score-0 incident).

A WNS measured on an UNROUTED design is an estimate; accepting it as best
lets the eager mirror publish an unrouted checkpoint that the harness cannot
measure (attempt #10: estimated -0.824 post-place beat every real routed
result; shipped DCP had 3488/3488 nets unrouted -> vivado_measurement_failed,
design score 0). _routed_ok_for_best gates every main-loop best update."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dcp_optimizer as do


def _probe(route_status_text):
    async def call_tool(name, args):
        return route_status_text

    class _Stub:
        pass
    stub = _Stub()
    stub.call_tool = call_tool
    return asyncio.run(do.DCPOptimizer._routed_ok_for_best(stub))


def test_rejects_fully_unrouted_design():
    # attempt #10 shape: every routable net unrouted, zero "errors".
    txt = ("# of routable nets...................... : 3488 :\n"
           "# of unrouted nets.................. : 3488 :\n"
           "# of nets with routing errors....... : 0 :\n")
    assert _probe(txt) is False


def test_accepts_fully_routed_2025_1_report():
    # 2025.1 quirk: fully routed reports OMIT the unrouted-nets line.
    txt = ("# of routable nets...................... : 3488 :\n"
           "# of fully routed nets.................. : 3488 :\n"
           "# of nets with routing errors....... : 0 :\n")
    assert _probe(txt) is True


def test_rejects_routing_errors():
    txt = ("# of routable nets...................... : 100 :\n"
           "# of nets with routing errors....... : 7 :\n")
    assert _probe(txt) is False


def test_fails_closed_on_error_envelope():
    # jul20 whole-file review S1: a probe hiccup must REJECT the
    # best-accept — a rejected real improvement recurs at the next
    # measurement, but an accepted phantom is protected forever by
    # never-worse and scores 0 (the tail-of-wall timeout regime).
    assert _probe('{"error": "tool_timed_out_budget"}') is False


def test_fails_closed_on_exception():
    async def boom(name, args):
        raise RuntimeError("dead session")

    class _Stub:
        pass
    stub = _Stub()
    stub.call_tool = boom
    assert asyncio.run(do.DCPOptimizer._routed_ok_for_best(stub)) is False


def test_rejects_unplaced_design_via_tracker():
    # jul23 wave-4 Q1/Q4: an UNPLACED design (place_design -unplace)
    # reports NEITHER regex line — the report text looks "clean" — but
    # the in-process routed-state tracker flipped False.  The tracker
    # check must reject BEFORE the report heuristics (the phantom shipped
    # a near-MET unplaced DCP on logicnets and spam otherwise).
    import asyncio
    import dcp_optimizer as do

    async def call_tool(name, args):
        raise AssertionError("probe must not even run — tracker rejects first")

    class _Stub:
        pass
    stub = _Stub()
    stub.call_tool = call_tool
    stub._design_routed_state = False
    assert asyncio.run(do.DCPOptimizer._routed_ok_for_best(stub)) is False


def test_tracker_true_defers_to_report_probe():
    import asyncio
    import dcp_optimizer as do

    async def call_tool(name, args):
        return ("# of routable nets...................... : 3488 :\n"
                "# of nets with routing errors....... : 0 :\n")

    class _Stub:
        pass
    stub = _Stub()
    stub.call_tool = call_tool
    stub._design_routed_state = True
    assert asyncio.run(do.DCPOptimizer._routed_ok_for_best(stub)) is True
