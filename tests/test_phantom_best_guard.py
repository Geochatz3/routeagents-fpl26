"""Verify that unrouted timing estimates cannot become the published best result.

Timing measured before routing is only an estimate and may appear better than
valid routed results. `_routed_ok_for_best` therefore gates every main-loop
best update so the selected checkpoint remains measurable and fully routed.
"""
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
    # Best-state acceptance fails closed when the verification probe reports
    # an error. This prevents an unverified result from becoming protected
    # by the never-worse guard.
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
    # An unplaced design may omit both report patterns and appear valid.
    # The in-process routed-state tracker must reject it before any report
    # probe runs.
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
