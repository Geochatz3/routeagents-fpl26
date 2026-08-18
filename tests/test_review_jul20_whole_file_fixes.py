"""jul20 whole-file external review (codex+terra, verified) — regressions.

Covers the verified fixes:
  S2 — piggyback mirror must not stamp best_wns onto a checkpoint written
       AFTER a design-mutating op (mutation-epoch guard).
  S4 — _structural_validate_dcp must treat an open_checkpoint error
       envelope as invalid (call_tool returns envelopes, does not raise),
       and a placed-but-unrouted design (0 routing errors) must fail on
       the unrouted-nets line.
S1's fail-closed flip is covered in test_phantom_best_guard.py; S5's
emergency-handler preference lives in main()'s closure (flag mechanics
asserted here, behavior exercised at rehearsal).
"""
import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dcp_optimizer as do


class _Stub:
    pass


# ---------------------------------------------------------------- S2 ----

def _piggyback_stub(tmp_path, epoch, pending_epoch):
    stub = _Stub()
    stub._mutation_epoch = epoch
    stub._pending_best_mirror_epoch = pending_epoch
    stub._pending_best_mirror = True
    stub.run_dir = tmp_path
    stub.best_wns = -0.5
    stub._path_guard_check = lambda *a, **k: None
    stub._bump_lineage = lambda *a, **k: None
    return stub


def test_piggyback_skips_and_disarms_on_epoch_mismatch(tmp_path):
    src = tmp_path / "llm_written.dcp"
    src.write_bytes(b"x" * 64)
    stub = _piggyback_stub(tmp_path, epoch=3, pending_epoch=2)
    do.DCPOptimizer._piggyback_mirror_checkpoint(
        stub, {"dcp_path": str(src)})
    assert stub._pending_best_mirror is False          # disarmed
    assert not hasattr(stub, "_best_valid_dcp")        # no stamp
    assert not (tmp_path / "best_valid.dcp").exists()  # no copy


def test_piggyback_stamps_when_epoch_unchanged(tmp_path):
    src = tmp_path / "llm_written.dcp"
    src.write_bytes(b"x" * 64)
    stub = _piggyback_stub(tmp_path, epoch=2, pending_epoch=2)
    do.DCPOptimizer._piggyback_mirror_checkpoint(
        stub, {"dcp_path": str(src)})
    assert stub._pending_best_mirror is False
    assert stub._best_valid_dcp == (tmp_path / "best_valid.dcp").resolve()
    assert stub._best_valid_dcp_wns == -0.5
    assert (tmp_path / "best_valid.dcp").exists()


# ---------------------------------------------------------------- S4 ----

def _validate(dcp_path, responses):
    """Drive _structural_validate_dcp with canned per-tool responses."""
    async def call_tool(name, args):
        return responses[name]
    stub = _Stub()
    stub.call_tool = call_tool
    return asyncio.run(
        do.DCPOptimizer._structural_validate_dcp(stub, dcp_path))


def test_structural_validate_rejects_open_error_envelope(tmp_path):
    dcp = tmp_path / "out.dcp"
    dcp.write_bytes(b"x" * 64)
    val = _validate(dcp, {
        "vivado_open_checkpoint": '{"error": "tool_timed_out_budget"}',
    })
    assert val["valid"] is False
    assert "error envelope" in val["reason"]


def test_structural_validate_rejects_unrouted_nets(tmp_path):
    # attempt #10 shape at finalize: 0 routing errors, everything
    # unrouted — must NOT validate.
    dcp = tmp_path / "out.dcp"
    dcp.write_bytes(b"x" * 64)
    rs = ("# of routable nets...................... : 3488 :\n"
          "# of unrouted nets.................. : 3488 :\n"
          "# of nets with routing errors....... : 0 :\n")
    val = _validate(dcp, {
        "vivado_open_checkpoint": "open_checkpoint completed",
        "vivado_report_route_status": rs,
    })
    assert val["valid"] is False
    assert "unrouted nets" in val["reason"]


def test_structural_validate_accepts_fully_routed_2025_1(tmp_path):
    # 2025.1 omits the unrouted-nets line when fully routed.
    dcp = tmp_path / "out.dcp"
    dcp.write_bytes(b"x" * 64)
    rs = ("# of routable nets...................... : 3488 :\n"
          "# of fully routed nets.................. : 3488 :\n"
          "# of nets with routing errors....... : 0 :\n")
    ts = "WNS(ns)      TNS(ns)\n  -0.123      -1.0\n"
    val = _validate(dcp, {
        "vivado_open_checkpoint": "open_checkpoint completed",
        "vivado_report_route_status": rs,
        "vivado_report_timing_summary": ts,
    })
    # Reaches the timing stage (WNS parse governs validity from there).
    assert val["reason"] in ("ok", "could not parse WNS from timing summary")


# ---------------------------------------------------------------- S5 ----

def test_finalize_completed_flag_defaults_false():
    # The SIGTERM emergency handler keys need_dcp on this flag; it must
    # exist and default False so a mid-run LLM write never masquerades
    # as a finalized artifact.
    assert do.DCPOptimizerBase.__init__ is not None
    src = open(do.__file__).read()
    assert "self._finalize_completed: bool = False" in src
    assert "or not finalize_done" in src


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
