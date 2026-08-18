"""C1-T5 corrupt_output fault class — checksum-verified shipping (2026-07-21).

Farm matrix finding (2/2 designs, verify_cell G4): the T5 harness injected
1081344B of garbage at the multi-restart ATTEMPT path mid-run, then SIGTERMed
the process group.  The wrapper's `_emergency_publish` treated "attempt file
exists" as "agent wrote it" and published the garbage to the scored location
instantly — winning the race against the agent's own (correct)
EMERGENCY_BEST_VALID_COPY that restored the attempt path ~1s later.

Fix contract pinned here:
  * agent records the shipped artifact's identity (size+md5) at finalize and
    at every emergency copy, and publishes it as <output>.shipped.json;
  * the wrapper only publishes an in-flight attempt file after verifying it
    against that manifest (or, past the wait deadline, after a zip-magic
    plausibility gate) — NEVER implausible bytes;
  * the agent's emergency handler distrusts a post-finalize output whose
    digest mismatches the recorded identity, and distrusts a best_valid
    mirror whose size mismatches its banked size (truncation variant).
"""

import threading
import time
from pathlib import Path

import pytest

import dcp_optimizer as dopt
from dcp_optimizer import (
    _artifact_identity,
    _verify_shipped_identity,
    _write_shipped_manifest,
)
from scripts.multi_restart_optimize import (
    _emergency_publish,
    _matches_manifest,
    _plausible_dcp,
    _read_shipped_manifest,
)

# Exact byte pattern + size the T5 matrix injects (runner.sh corrupt_output).
T5_GARBAGE = b"T5_GARBAGE_DCP__NOT_A_CHECKPOINT\n" * 32768
MIRROR_BYTES = b"PK\x03\x04" + b"mirror-best-valid-bytes" * 1000


# ---------------------------------------------------------------- helpers --

class TestArtifactIdentity:
    def test_identity_records_size_and_md5(self, tmp_path):
        p = tmp_path / "out.dcp"
        p.write_bytes(MIRROR_BYTES)
        ident = _artifact_identity(p)
        assert ident["size"] == len(MIRROR_BYTES)
        assert len(ident["md5"]) == 32
        assert ident["path"] == str(p)

    def test_identity_none_for_missing_or_empty(self, tmp_path):
        assert _artifact_identity(tmp_path / "nope.dcp") is None
        empty = tmp_path / "empty.dcp"
        empty.write_bytes(b"")
        assert _artifact_identity(empty) is None

    def test_manifest_roundtrip_agent_to_wrapper(self, tmp_path):
        # The agent writes <output>.shipped.json; the WRAPPER must be able
        # to read it back — cross-module contract.
        p = tmp_path / "out.dcp"
        p.write_bytes(MIRROR_BYTES)
        ident = _artifact_identity(p)
        _write_shipped_manifest(p, ident)
        manifest = _read_shipped_manifest(p)
        assert manifest is not None
        assert manifest["size"] == ident["size"]
        assert manifest["md5"] == ident["md5"]
        assert _matches_manifest(p, manifest)


class TestVerifyShippedIdentity:
    def test_matching_artifact_verifies(self, tmp_path):
        p = tmp_path / "out.dcp"
        p.write_bytes(MIRROR_BYTES)
        ok, why = _verify_shipped_identity(p, _artifact_identity(p))
        assert ok, why

    def test_garbage_swap_detected(self, tmp_path):
        # The matrix shape: identity recorded for the real artifact, then
        # garbage bytes land at the same path.
        p = tmp_path / "out.dcp"
        p.write_bytes(MIRROR_BYTES)
        ident = _artifact_identity(p)
        p.write_bytes(T5_GARBAGE)
        ok, why = _verify_shipped_identity(p, ident)
        assert not ok
        assert "size_mismatch" in why

    def test_same_size_garbage_detected_by_magic(self, tmp_path):
        # Same-size overwrite: size check passes, zip-magic catches it.
        p = tmp_path / "out.dcp"
        p.write_bytes(MIRROR_BYTES)
        ident = _artifact_identity(p)
        p.write_bytes(b"XX" + MIRROR_BYTES[2:])
        ok, why = _verify_shipped_identity(p, ident)
        assert not ok
        assert why == "not_a_zip_dcp"

    def test_same_size_pk_prefixed_tamper_detected_by_md5(self, tmp_path):
        p = tmp_path / "out.dcp"
        p.write_bytes(MIRROR_BYTES)
        ident = _artifact_identity(p)
        tampered = bytearray(MIRROR_BYTES)
        tampered[100] ^= 0xFF
        p.write_bytes(bytes(tampered))
        ok, why = _verify_shipped_identity(p, ident)
        assert not ok
        assert "md5_mismatch" in why

    def test_missing_output_fails_closed(self, tmp_path):
        p = tmp_path / "out.dcp"
        p.write_bytes(MIRROR_BYTES)
        ident = _artifact_identity(p)
        p.unlink()
        ok, why = _verify_shipped_identity(p, ident)
        assert not ok
        assert why == "output_missing"

    def test_unreadable_output_fails_closed_without_raising(self, tmp_path):
        # Signal-context contract: verification must NEVER raise — an
        # unreadable path returns (False, ...) so the caller restores.
        p = tmp_path / "out.dcp"
        p.mkdir()  # a directory at the output path: stat OK, open raises
        ident = {"path": str(p), "size": p.stat().st_size, "md5": "0" * 32}
        ok, _why = _verify_shipped_identity(p, ident)
        assert not ok

    def test_no_recorded_identity_fails_closed(self, tmp_path):
        p = tmp_path / "out.dcp"
        p.write_bytes(MIRROR_BYTES)
        assert _verify_shipped_identity(p, None)[0] is False
        assert _verify_shipped_identity(p, {})[0] is False


class TestPlausibleDcp:
    def test_rejects_t5_garbage(self, tmp_path):
        p = tmp_path / "g.dcp"
        p.write_bytes(T5_GARBAGE)
        assert not _plausible_dcp(p)

    def test_accepts_zip_magic(self, tmp_path):
        p = tmp_path / "d.dcp"
        p.write_bytes(MIRROR_BYTES)
        assert _plausible_dcp(p)

    def test_rejects_empty_missing_unreadable(self, tmp_path):
        e = tmp_path / "e.dcp"
        e.write_bytes(b"")
        assert not _plausible_dcp(e)
        assert not _plausible_dcp(tmp_path / "missing.dcp")
        d = tmp_path / "dir.dcp"
        d.mkdir()
        assert not _plausible_dcp(d)


# ------------------------------------------------- wrapper emergency path --

class TestEmergencyPublishChecksumTruth:
    """Replays of the farm matrix sequence against the fixed wrapper."""

    def test_matrix_replay_garbage_then_agent_restore(self, tmp_path):
        """THE fired sequence: garbage pre-exists at the attempt path when
        SIGTERM lands; the agent's emergency finalize restores the mirror
        bytes (+ manifest) ~1s later.  The wrapper must wait, verify, and
        ship the MIRROR bytes — never the garbage."""
        cur = tmp_path / "mr_boom_soc_2025.1_v2_1_1.dcp"
        cur.write_bytes(T5_GARBAGE)
        assert cur.stat().st_size == 1081344  # exact farm-observed size
        final = tmp_path / "boom_soc_2025.1_v2_optimized.dcp"

        def _agent_emergency_finalize():
            time.sleep(0.7)
            # agent side: atomic copy then manifest (same order as
            # _emergency_baseline_copy)
            tmp = cur.with_name(cur.name + ".tmpX")
            tmp.write_bytes(MIRROR_BYTES)
            tmp.replace(cur)
            _write_shipped_manifest(cur, _artifact_identity(cur))

        t = threading.Thread(target=_agent_emergency_finalize)
        t.start()
        try:
            r = _emergency_publish(final, cur, wait_s=10.0)
        finally:
            t.join()
        assert r == "published_inflight_verified"
        assert final.read_bytes() == MIRROR_BYTES

    def test_garbage_never_replaced_is_rejected_not_shipped(self, tmp_path):
        # Agent died hard: garbage remains, no manifest ever appears.
        # Shipping nothing is strictly better than shipping garbage.
        cur = tmp_path / "mr_bench_1_1.dcp"
        cur.write_bytes(T5_GARBAGE)
        final = tmp_path / "bench_optimized.dcp"
        r = _emergency_publish(final, cur, wait_s=1.2)
        assert r == "inflight_rejected_not_a_dcp"
        assert not final.exists()

    def test_manifest_mismatch_falls_back_to_magic_gate(self, tmp_path):
        # Stale manifest + garbage bytes: verification can never pass, and
        # the plausibility gate must still refuse the garbage.
        cur = tmp_path / "mr_bench_1_1.dcp"
        cur.write_bytes(MIRROR_BYTES)
        _write_shipped_manifest(cur, _artifact_identity(cur))
        cur.write_bytes(T5_GARBAGE)  # injected after manifest
        final = tmp_path / "bench_optimized.dcp"
        r = _emergency_publish(final, cur, wait_s=1.2)
        assert r == "inflight_rejected_not_a_dcp"
        assert not final.exists()

    def test_plausible_dcp_without_manifest_still_ships(self, tmp_path):
        # Never-worse vs the pre-manifest agent: a real (zip-magic) DCP with
        # no manifest is still published after the wait deadline.
        cur = tmp_path / "mr_bench_1_1.dcp"
        cur.write_bytes(MIRROR_BYTES)
        final = tmp_path / "bench_optimized.dcp"
        r = _emergency_publish(final, cur, wait_s=0.6)
        assert r == "published_inflight_unverified"
        assert final.read_bytes() == MIRROR_BYTES

    def test_verified_publish_is_immediate(self, tmp_path):
        # Manifest + matching bytes present up front: no deadline wait.
        cur = tmp_path / "mr_bench_1_1.dcp"
        cur.write_bytes(MIRROR_BYTES)
        _write_shipped_manifest(cur, _artifact_identity(cur))
        final = tmp_path / "bench_optimized.dcp"
        t0 = time.monotonic()
        r = _emergency_publish(final, cur, wait_s=30.0)
        assert r == "published_inflight_verified"
        assert time.monotonic() - t0 < 5.0
        assert final.read_bytes() == MIRROR_BYTES

    def test_never_worse_rules_unchanged(self, tmp_path):
        # kept_existing / nothing_to_publish / no_final_output contracts.
        final = tmp_path / "bench_optimized.dcp"
        final.write_bytes(b"best-of-completed-attempts")
        assert _emergency_publish(final, tmp_path / "x.dcp",
                                  wait_s=0.1) == "kept_existing"
        final2 = tmp_path / "other_optimized.dcp"
        assert _emergency_publish(final2, None,
                                  wait_s=0.1) == "nothing_to_publish"
        assert _emergency_publish(None, None,
                                  wait_s=0.1) == "no_final_output"
        r = _emergency_publish(final2, tmp_path / "never.dcp", wait_s=0.6)
        assert r == "inflight_never_landed"


# ------------------------------------------- agent-side lifecycle pinning --

def _src() -> str:
    return Path(dopt.__file__).read_text()


class TestAgentLifecycleSourcePins:
    """Source pins in the style of test_review_jul20_whole_file_fixes.py —
    the emergency handler is a closure inside main() and cannot be imported;
    these pin the decision wiring the behavioral tests above rely on."""

    def test_finalize_records_shipped_identity(self):
        src = _src()
        assert "self._shipped_artifact = ident" in src
        assert "_write_shipped_manifest(Path(output_dcp), ident)" in src
        # recording happens only after _finalize_completed is set
        assert src.index("self._finalize_completed = True") < src.index(
            "self._shipped_artifact = ident")

    def test_emergency_verifies_output_against_recorded_identity(self):
        src = _src()
        assert "SHIP INTEGRITY FAIL: output does not" in src
        # mismatch demotes the on-disk output to need_dcp
        assert ("or not finalize_done or ship_integrity_fail" in src)

    def test_emergency_no_op_path_gated_on_integrity(self):
        # Path 2 (leave existing output) must be unreachable when the
        # integrity check failed.
        assert ("out.stat().st_size > 0 and not ship_integrity_fail"
                in _src())

    def test_regression_pin_prefinalize_injection_still_covered(self):
        # Injection BEFORE finalize: _finalize_completed is False, so
        # need_dcp must remain True regardless of the on-disk file — the
        # jul20 S5 gate.  (The T5 pre-finalize cells passed via this.)
        assert "or not finalize_done" in _src()

    def test_mirror_size_recorded_and_checked(self):
        src = _src()
        assert "self._best_valid_mirror_size: Optional[int] = None" in src
        # banked at mirror time (eager/backstop + piggyback + polish/ILS)
        assert src.count("_best_valid_mirror_size = ") >= 5
        # distrusted on size mismatch in the emergency path
        assert "distrusting mirror" in src

    def test_emergency_copies_republish_identity(self):
        # Both emergency copy paths (best_valid + baseline) refresh the
        # manifest so the wrapper can verify what was just shipped.
        assert _src().count("optimizer._shipped_artifact = _ident") == 2

    def test_lifecycle_metadata_carries_shipped_identity(self):
        assert '"shipped_artifact": getattr(' in _src()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
