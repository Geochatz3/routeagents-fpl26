"""INSURED-COMPARE final-candidate MUX tests.

Mechanism under test (dcp_optimizer.py):

  * register_final_candidate(path, wns, source_label, ...) enrolls a
    never-worse FINAL candidate DCP — but ONLY after the SAME gates the
    auto-bank path uses (tracker-first _routed_ok_for_best True; hold via
    self.call_tool, whs < 0 rejects / None fails-open loud; cell-count
    band [0.5x, 3x] of the Phase-1 entry count).

  * _maybe_ship_final_candidate_mux(output_dcp) ships the argmax by
    scored-clock WNS across the verified candidates.  The pipeline's
    best_valid is the implicit candidate #0 (source="pipeline").  When
    only the pipeline candidate exists — or the pipeline wins/ties, or a
    winning candidate fails structural validate — it is a strict no-op:
    output_dcp is left UNTOUCHED (byte-identical finalize) and no
    "[mux] FINAL" line is logged.

Load-bearing invariants:
  - pipeline-only  => zero diff, no [mux] log, output byte-identical;
  - a strictly-better verified candidate is shipped + logged loudly;
  - an UNVERIFIED candidate (unrouted / hold-dirty / cell out of band)
    is NEVER enrolled -> never reachable by the MUX;
  - an exact tie goes to the pipeline (stability).

Harness style mirrors tests/test_fresh_presweep.py: stubbed sessions,
NO Vivado, NO RapidWright, NO network, NO real sleeps.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import (
    FINAL_CANDIDATE_MUX_MIN_GAIN_NS,
    DCPOptimizer,
)
from optimizer.ils_polish import ILSPolishConfig
from optimizer.replace_gamble import ReplaceGambleResult


def _async(coro):
    return asyncio.run(coro)


class _FakeContent:
    def __init__(self, text: str):
        self.text = text


class _FakeResult:
    def __init__(self, text: str):
        self.content = [_FakeContent(text)]


class _FakeSession:
    def __init__(self, response_text: str = "ok"):
        self.response_text = response_text
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        return _FakeResult(self.response_text)


def _make_optimizer(tmp_path: Path, *, initial_wns=-2.0,
                    best_wns=-1.0) -> DCPOptimizer:
    opt = DCPOptimizer(api_key="test", run_dir=tmp_path)
    opt.vivado_session = _FakeSession()
    opt.rapidwright_session = _FakeSession()
    opt.initial_wns = initial_wns
    opt.best_wns = best_wns
    opt.clock_period = 5.0
    opt.input_dcp_path = tmp_path / "pristine.dcp"
    opt.input_dcp_path.write_bytes(b"PRISTINE")
    return opt


def _candidate_dcp(tmp_path: Path, name: str = "presweep.dcp") -> Path:
    d = tmp_path / "final_candidates"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(b"CANDIDATE-DCP-BYTES")
    return p


# The MUX: argmax + never-worse + byte-identical no-op

class MuxShipTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.output = self.tmp_path / "out.dcp"

    def tearDown(self):
        self.tmp.cleanup()

    def _stub_ship(self, opt):
        """Stub the Vivado-touching helpers of the ship path."""
        opt._path_guard_check = mock.MagicMock(return_value=True)
        opt._write_readable_edif = mock.AsyncMock(return_value=True)
        opt.call_tool = mock.AsyncMock(return_value="ok")
        # structural validate echoes the candidate's own WNS as valid.
        async def fake_validate(path):
            return {"valid": True, "wns": self._val_wns,
                    "route_errors": 0, "reason": None}
        opt._structural_validate_dcp = fake_validate

    # --- pipeline-only: strict no-op, byte-identical ---

    def test_pipeline_only_is_byte_identical_no_op(self):
        opt = _make_optimizer(self.tmp_path)
        # An output artifact that must NOT be touched by the MUX.
        self.output.write_bytes(b"PIPELINE-OUTPUT")
        # Pipeline-only: strict no-op — not even a [mux] log line fires.
        with self.assertNoLogs("dcp_optimizer", level="INFO"):
            shipped = _async(opt._maybe_ship_final_candidate_mux(self.output))
        self.assertFalse(shipped)
        self.assertEqual(self.output.read_bytes(), b"PIPELINE-OUTPUT")

    def test_pipeline_only_no_mux_log_at_all(self):
        # No candidates => the method returns before emitting ANY [mux] line.
        opt = _make_optimizer(self.tmp_path)
        shipped = _async(opt._maybe_ship_final_candidate_mux(self.output))
        self.assertFalse(shipped)
        self.assertFalse(self.output.exists())

    # --- a strictly-better candidate wins + logs ---

    def test_better_candidate_is_shipped_and_logged(self):
        opt = _make_optimizer(self.tmp_path, initial_wns=-2.0, best_wns=-1.0)
        cand = _candidate_dcp(self.tmp_path)
        opt._final_candidates.append({
            "path": str(cand), "wns": -0.4, "whs": 0.05,
            "source_label": "presweep", "cell_count": 1000,
            "verified_routed": True,
        })
        self._val_wns = -0.4
        self._stub_ship(opt)
        with self.assertLogs("dcp_optimizer", level="INFO") as logs:
            shipped = _async(opt._maybe_ship_final_candidate_mux(self.output))
        self.assertTrue(shipped)
        # Shipped the candidate bytes.
        self.assertEqual(self.output.read_bytes(), b"CANDIDATE-DCP-BYTES")
        # best_wns now reflects the shipped artifact.
        self.assertEqual(opt.best_wns, -0.4)
        self.assertIn("VALID_OPTIMIZED", opt.final_status)
        text = "\n".join(logs.output)
        self.assertIn("[mux] FINAL = presweep", text)
        self.assertIn("beat pipeline -1.000", text)
        self.assertIn("shipped argmax of 2 candidates", text)

    def test_argmax_picks_best_of_several(self):
        opt = _make_optimizer(self.tmp_path, initial_wns=-2.0, best_wns=-1.5)
        c1 = _candidate_dcp(self.tmp_path, "a.dcp")
        c2 = _candidate_dcp(self.tmp_path, "b.dcp")
        opt._final_candidates.extend([
            {"path": str(c1), "wns": -1.2, "whs": 0.0,
             "source_label": "presweep", "cell_count": 1000,
             "verified_routed": True},
            {"path": str(c2), "wns": -0.3, "whs": 0.0,
             "source_label": "rebite", "cell_count": 1000,
             "verified_routed": True},
        ])
        self._val_wns = -0.3
        self._stub_ship(opt)
        shipped = _async(opt._maybe_ship_final_candidate_mux(self.output))
        self.assertTrue(shipped)
        self.assertEqual(self.output.read_bytes(), c2.read_bytes())
        self.assertEqual(opt.best_wns, -0.3)

    # --- pipeline wins / tie => pipeline (stability, no-op) ---

    def test_pipeline_wins_when_candidate_worse(self):
        opt = _make_optimizer(self.tmp_path, initial_wns=-2.0, best_wns=-0.5)
        cand = _candidate_dcp(self.tmp_path)
        opt._final_candidates.append({
            "path": str(cand), "wns": -1.0, "whs": 0.0,
            "source_label": "presweep", "cell_count": 1000,
            "verified_routed": True,
        })
        self._val_wns = -1.0
        self._stub_ship(opt)
        self.output.write_bytes(b"PIPELINE-OUTPUT")
        shipped = _async(opt._maybe_ship_final_candidate_mux(self.output))
        self.assertFalse(shipped)
        self.assertEqual(self.output.read_bytes(), b"PIPELINE-OUTPUT")

    def test_exact_tie_goes_to_pipeline(self):
        # Candidate WNS == pipeline WNS: NOT greater by the min gain -> pipeline.
        opt = _make_optimizer(self.tmp_path, initial_wns=-2.0, best_wns=-0.9)
        cand = _candidate_dcp(self.tmp_path)
        opt._final_candidates.append({
            "path": str(cand), "wns": -0.9, "whs": 0.0,
            "source_label": "presweep", "cell_count": 1000,
            "verified_routed": True,
        })
        self._val_wns = -0.9
        self._stub_ship(opt)
        self.output.write_bytes(b"PIPELINE-OUTPUT")
        with self.assertLogs("dcp_optimizer", level="INFO") as logs:
            shipped = _async(opt._maybe_ship_final_candidate_mux(self.output))
        self.assertFalse(shipped)
        self.assertEqual(self.output.read_bytes(), b"PIPELINE-OUTPUT")
        self.assertIn("tie -> pipeline", "\n".join(logs.output))

    def test_sub_margin_gain_goes_to_pipeline(self):
        # Better, but by less than the min gain -> pipeline (jitter guard).
        opt = _make_optimizer(self.tmp_path, initial_wns=-2.0, best_wns=-0.9)
        cand = _candidate_dcp(self.tmp_path)
        opt._final_candidates.append({
            "path": str(cand),
            "wns": -0.9 + FINAL_CANDIDATE_MUX_MIN_GAIN_NS / 2.0,
            "whs": 0.0, "source_label": "presweep", "cell_count": 1000,
            "verified_routed": True,
        })
        self._val_wns = -0.9
        self._stub_ship(opt)
        self.output.write_bytes(b"PIPELINE-OUTPUT")
        shipped = _async(opt._maybe_ship_final_candidate_mux(self.output))
        self.assertFalse(shipped)
        self.assertEqual(self.output.read_bytes(), b"PIPELINE-OUTPUT")

    # --- no-improvement pipeline: candidate rescues via baseline compare ---

    def test_candidate_beats_baseline_when_pipeline_no_improvement(self):
        # Pipeline made no improvement (best <= initial) -> it would ship
        # the baseline (initial_wns).  A verified candidate that beats the
        # baseline must win.
        opt = _make_optimizer(self.tmp_path, initial_wns=-2.0, best_wns=-2.0)
        cand = _candidate_dcp(self.tmp_path)
        opt._final_candidates.append({
            "path": str(cand), "wns": -0.5, "whs": 0.0,
            "source_label": "presweep", "cell_count": 1000,
            "verified_routed": True,
        })
        self._val_wns = -0.5
        self._stub_ship(opt)
        shipped = _async(opt._maybe_ship_final_candidate_mux(self.output))
        self.assertTrue(shipped)
        self.assertEqual(opt.best_wns, -0.5)

    def test_initial_wns_none_defers_to_pipeline(self):
        opt = _make_optimizer(self.tmp_path, initial_wns=None, best_wns=-0.5)
        cand = _candidate_dcp(self.tmp_path)
        opt._final_candidates.append({
            "path": str(cand), "wns": -0.1, "whs": 0.0,
            "source_label": "presweep", "cell_count": 1000,
            "verified_routed": True,
        })
        self.output.write_bytes(b"PIPELINE-OUTPUT")
        shipped = _async(opt._maybe_ship_final_candidate_mux(self.output))
        self.assertFalse(shipped)
        self.assertEqual(self.output.read_bytes(), b"PIPELINE-OUTPUT")

    # --- never-worse: a winning-by-WNS candidate that fails structural
    #     validate is REJECTED and the pipeline is kept (output untouched) ---

    def test_candidate_failing_structural_validate_keeps_pipeline(self):
        opt = _make_optimizer(self.tmp_path, initial_wns=-2.0, best_wns=-1.0)
        cand = _candidate_dcp(self.tmp_path)
        opt._final_candidates.append({
            "path": str(cand), "wns": -0.4, "whs": 0.0,
            "source_label": "presweep", "cell_count": 1000,
            "verified_routed": True,
        })
        opt._path_guard_check = mock.MagicMock(return_value=True)
        opt._structural_validate_dcp = mock.AsyncMock(
            return_value={"valid": False, "reason": "unrouted nets"})
        self.output.write_bytes(b"PIPELINE-OUTPUT")
        with self.assertLogs("dcp_optimizer", level="INFO") as logs:
            shipped = _async(opt._maybe_ship_final_candidate_mux(self.output))
        self.assertFalse(shipped)
        # Never touched output_dcp -> byte-identical.
        self.assertEqual(self.output.read_bytes(), b"PIPELINE-OUTPUT")
        self.assertIn("FAILED structural validate", "\n".join(logs.output))

    def test_missing_candidate_file_keeps_pipeline(self):
        opt = _make_optimizer(self.tmp_path, initial_wns=-2.0, best_wns=-1.0)
        opt._final_candidates.append({
            "path": str(self.tmp_path / "final_candidates" / "gone.dcp"),
            "wns": -0.4, "whs": 0.0, "source_label": "presweep",
            "cell_count": 1000, "verified_routed": True,
        })
        self.output.write_bytes(b"PIPELINE-OUTPUT")
        shipped = _async(opt._maybe_ship_final_candidate_mux(self.output))
        self.assertFalse(shipped)
        self.assertEqual(self.output.read_bytes(), b"PIPELINE-OUTPUT")


# register_final_candidate: the SAME gates as the bank path

class RegisterGateTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.cand = _candidate_dcp(self.tmp_path)

    def tearDown(self):
        self.tmp.cleanup()

    def _opt(self, *, routed=True, whs=0.05, cell="1000",
             input_cells=1000):
        opt = _make_optimizer(self.tmp_path)
        opt._input_cell_count = input_cells
        opt._routed_ok_for_best = mock.AsyncMock(return_value=routed)
        # verify=True cell probe goes through self.call_tool.
        opt.call_tool = mock.AsyncMock(return_value=cell)
        self._whs = whs
        return opt

    def _patch_hold(self, whs):
        return mock.patch("optimizer.ils_polish._measure_hold",
                          new=mock.AsyncMock(return_value=whs))

    # --- verify=True path ---

    def test_verified_candidate_enrolled(self):
        opt = self._opt(routed=True, cell="1000", input_cells=1000)
        with self._patch_hold(0.05):
            ok = _async(opt.register_final_candidate(
                self.cand, -0.4, "presweep", verify=True))
        self.assertTrue(ok)
        self.assertEqual(len(opt._final_candidates), 1)
        e = opt._final_candidates[0]
        self.assertEqual(e["source_label"], "presweep")
        self.assertTrue(e["verified_routed"])
        self.assertEqual(e["cell_count"], 1000)

    def test_unrouted_candidate_rejected(self):
        opt = self._opt(routed=False)
        with self._patch_hold(0.05):
            ok = _async(opt.register_final_candidate(
                self.cand, -0.4, "presweep", verify=True))
        self.assertFalse(ok)
        self.assertEqual(opt._final_candidates, [])

    def test_hold_dirty_candidate_rejected(self):
        opt = self._opt(routed=True)
        with self._patch_hold(-0.02):
            ok = _async(opt.register_final_candidate(
                self.cand, -0.4, "presweep", verify=True))
        self.assertFalse(ok)
        self.assertEqual(opt._final_candidates, [])

    def test_hold_unmeasurable_fails_open_loud(self):
        opt = self._opt(routed=True)
        with self._patch_hold(None), \
             self.assertLogs("dcp_optimizer", level="INFO") as logs:
            ok = _async(opt.register_final_candidate(
                self.cand, -0.4, "presweep", verify=True))
        self.assertTrue(ok)
        self.assertIn("HOLD UNMEASURABLE", "\n".join(logs.output))
        self.assertIsNone(opt._final_candidates[0]["whs"])

    def test_cell_count_out_of_band_rejected(self):
        # 999999 vs entry 1000 -> way above 3x -> logic-deletion guard.
        opt = self._opt(routed=True, cell="999999", input_cells=1000)
        with self._patch_hold(0.05):
            ok = _async(opt.register_final_candidate(
                self.cand, -0.4, "presweep", verify=True))
        self.assertFalse(ok)
        self.assertEqual(opt._final_candidates, [])

    # --- verify=False path (caller pre-verified under identical gates) ---

    def test_verify_false_enrolls_with_passed_values(self):
        opt = _make_optimizer(self.tmp_path)
        opt._input_cell_count = 1000
        ok = _async(opt.register_final_candidate(
            self.cand, -0.4, "presweep", whs=0.03, cell_count=1000,
            verify=False))
        self.assertTrue(ok)
        self.assertEqual(opt._final_candidates[0]["whs"], 0.03)

    def test_verify_false_hold_dirty_rejected(self):
        opt = _make_optimizer(self.tmp_path)
        opt._input_cell_count = 1000
        ok = _async(opt.register_final_candidate(
            self.cand, -0.4, "presweep", whs=-0.01, cell_count=1000,
            verify=False))
        self.assertFalse(ok)

    def test_verify_false_cell_out_of_band_rejected(self):
        opt = _make_optimizer(self.tmp_path)
        opt._input_cell_count = 1000
        ok = _async(opt.register_final_candidate(
            self.cand, -0.4, "presweep", whs=0.0, cell_count=100000,
            verify=False))
        self.assertFalse(ok)

    # --- guardrails ---

    def test_reserved_pipeline_label_refused(self):
        opt = _make_optimizer(self.tmp_path)
        ok = _async(opt.register_final_candidate(
            self.cand, -0.4, "pipeline", whs=0.0, verify=False))
        self.assertFalse(ok)

    def test_missing_path_refused(self):
        opt = _make_optimizer(self.tmp_path)
        ok = _async(opt.register_final_candidate(
            self.tmp_path / "nope.dcp", -0.4, "presweep", whs=0.0,
            verify=False))
        self.assertFalse(ok)

    def test_unmeasurable_wns_refused(self):
        opt = _make_optimizer(self.tmp_path)
        ok = _async(opt.register_final_candidate(
            self.cand, None, "presweep", whs=0.0, verify=False))
        self.assertFalse(ok)


# FLAGSHIP (replace_gamble) -> MUX wiring: the best VERIFIED
# gamble draw is enrolled REGARDLESS of the +0.15 adopt gate.

class FlagshipMuxWiringTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.output = self.tmp_path / "out.dcp"

    def tearDown(self):
        self.tmp.cleanup()

    def _opt(self, *, enabled=True, best_wns=-0.8):
        opt = _make_optimizer(self.tmp_path, initial_wns=-2.0,
                              best_wns=best_wns)
        opt.run_dir = self.tmp_path
        opt._ils_polish_cfg = ILSPolishConfig(replace_gamble_enabled=enabled)
        opt._best_valid_dcp = None
        opt._best_valid_dcp_wns = best_wns
        opt._input_cell_count = None
        # register (verify=False path) makes no tool calls, but stub anyway.
        opt.call_tool = mock.AsyncMock(return_value="ok")
        return opt

    def _patch_run(self, res):
        return mock.patch("optimizer.replace_gamble.run_replace_gamble",
                          new=mock.AsyncMock(return_value=res))

    def test_below_margin_verified_draw_is_registered(self):
        # A +0.10 draw (below the +0.15 adopt bar) is surfaced by the runner
        # and enrolled as a "replace_gamble" FINAL candidate — the adopt gate
        # is NOT loosened (not adopted -> mirror not repointed).
        cand = _candidate_dcp(self.tmp_path, "rg_cand.dcp")
        res = ReplaceGambleResult(
            attempted=True, adopted=False, best_wns=-0.8, best_path=None,
            best_draw_path=str(cand), best_draw_wns=-0.7,
            best_draw_whs=0.02)
        opt = self._opt()
        with self._patch_run(res):
            _async(opt._replace_gamble_after_polish(
                "/tmp/banked.dcp", -0.8, time.time() + 5000.0, "SLACK"))
        self.assertEqual(len(opt._final_candidates), 1)
        e = opt._final_candidates[0]
        self.assertEqual(e["source_label"], "replace_gamble")
        self.assertEqual(e["wns"], -0.7)
        self.assertEqual(e["whs"], 0.02)
        # adopt gate UNCHANGED: below +0.15 -> not adopted -> no repoint.
        self.assertIsNone(opt._best_valid_dcp)
        self.assertEqual(opt.best_wns, -0.8)

    def test_registered_below_margin_draw_wins_mux_if_argmax(self):
        # The registered below-margin draw (-0.7) beats the pipeline (-0.8)
        # by > the min gain -> the MUX ships it at finalize.
        cand = _candidate_dcp(self.tmp_path, "rg_cand.dcp")
        res = ReplaceGambleResult(
            attempted=True, adopted=False, best_wns=-0.8, best_path=None,
            best_draw_path=str(cand), best_draw_wns=-0.7, best_draw_whs=0.0)
        opt = self._opt()
        with self._patch_run(res):
            _async(opt._replace_gamble_after_polish(
                "/tmp/banked.dcp", -0.8, time.time() + 5000.0, "SLACK"))
        # Ship path stubs.
        opt._path_guard_check = mock.MagicMock(return_value=True)
        opt._write_readable_edif = mock.AsyncMock(return_value=True)
        opt.call_tool = mock.AsyncMock(return_value="ok")
        async def fake_validate(path):
            return {"valid": True, "wns": -0.7, "route_errors": 0,
                    "reason": None}
        opt._structural_validate_dcp = fake_validate
        shipped = _async(opt._maybe_ship_final_candidate_mux(self.output))
        self.assertTrue(shipped)
        self.assertEqual(self.output.read_bytes(), cand.read_bytes())
        self.assertEqual(opt.best_wns, -0.7)

    def test_flag_off_registers_nothing(self):
        # Default OFF: _replace_gamble_after_polish returns before ever
        # calling the runner — zero candidates, zero diff.
        opt = self._opt(enabled=False)
        ran = {"called": False}
        async def _should_not_run(*a, **k):
            ran["called"] = True
            return ReplaceGambleResult()
        with mock.patch("optimizer.replace_gamble.run_replace_gamble",
                        new=_should_not_run):
            _async(opt._replace_gamble_after_polish(
                "/tmp/banked.dcp", -0.8, time.time() + 5000.0, "SLACK"))
        self.assertFalse(ran["called"])
        self.assertEqual(opt._final_candidates, [])

    def test_verified_draw_worse_than_pipeline_registers_but_never_wins(self):
        # A verified draw WORSE than the pipeline is still enrolled (honest),
        # but the MUX's tie->pipeline keeps the pipeline (byte-identical).
        cand = _candidate_dcp(self.tmp_path, "rg_cand.dcp")
        res = ReplaceGambleResult(
            attempted=True, adopted=False, best_wns=-0.8, best_path=None,
            best_draw_path=str(cand), best_draw_wns=-1.2, best_draw_whs=0.0)
        opt = self._opt(best_wns=-0.8)
        with self._patch_run(res):
            _async(opt._replace_gamble_after_polish(
                "/tmp/banked.dcp", -0.8, time.time() + 5000.0, "SLACK"))
        self.assertEqual(len(opt._final_candidates), 1)     # registered
        self.output.write_bytes(b"PIPELINE-OUTPUT")
        opt._path_guard_check = mock.MagicMock(return_value=True)
        shipped = _async(opt._maybe_ship_final_candidate_mux(self.output))
        self.assertFalse(shipped)                            # never wins
        self.assertEqual(self.output.read_bytes(), b"PIPELINE-OUTPUT")

    def test_no_best_draw_path_registers_nothing(self):
        # Runner surfaced no verified draw (all errored/unrouted) -> nothing
        # to enroll.
        res = ReplaceGambleResult(
            attempted=True, adopted=False, best_wns=-0.8, best_path=None,
            best_draw_path=None)
        opt = self._opt()
        with self._patch_run(res):
            _async(opt._replace_gamble_after_polish(
                "/tmp/banked.dcp", -0.8, time.time() + 5000.0, "SLACK"))
        self.assertEqual(opt._final_candidates, [])


# RESERVE/controller -> MUX wiring: the tail enrolls its
# harvested best_valid; equal-to-pipeline is a byte-identical no-op.

class TailMuxWiringTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.output = self.tmp_path / "out.dcp"
        self.banked = self.tmp_path / "best_valid.dcp"
        self.banked.write_bytes(b"BANKED-BEST-BYTES")

    def tearDown(self):
        self.tmp.cleanup()

    def _opt(self, *, best_wns=-0.9, routed=True):
        opt = _make_optimizer(self.tmp_path, initial_wns=-2.0,
                              best_wns=best_wns)
        opt.run_dir = self.tmp_path
        opt._best_valid_dcp = self.banked
        opt._best_valid_dcp_wns = best_wns
        opt._input_cell_count = None
        opt._routed_ok_for_best = mock.AsyncMock(return_value=routed)
        opt.call_tool = mock.AsyncMock(return_value="ok")
        return opt

    def _patch_hold(self, whs):
        return mock.patch("optimizer.ils_polish._measure_hold",
                          new=mock.AsyncMock(return_value=whs))

    def test_tail_registers_harvested_best_valid(self):
        opt = self._opt(best_wns=-0.9)
        with self._patch_hold(0.02):
            enrolled = _async(
                opt._register_tail_final_candidate("tail_controller"))
        self.assertTrue(enrolled)
        e = opt._final_candidates[0]
        self.assertEqual(e["source_label"], "tail_controller")
        self.assertEqual(e["wns"], -0.9)
        # A copy of the banked best now lives in the private candidate store.
        store = self.tmp_path / "final_candidates" / "tail_controller.dcp"
        self.assertTrue(store.exists())

    def test_equal_to_pipeline_candidate_does_not_override(self):
        # The registered tail candidate == pipeline WNS -> tie -> pipeline;
        # output DCP is byte-identical (the default controller-ON contract).
        opt = self._opt(best_wns=-0.9)
        with self._patch_hold(0.02):
            _async(opt._register_tail_final_candidate("tail_controller"))
        self.output.write_bytes(b"PIPELINE-OUTPUT")
        opt._path_guard_check = mock.MagicMock(return_value=True)
        shipped = _async(opt._maybe_ship_final_candidate_mux(self.output))
        self.assertFalse(shipped)
        self.assertEqual(self.output.read_bytes(), b"PIPELINE-OUTPUT")

    def test_bare_reroute_lineage_registers_too(self):
        opt = self._opt(best_wns=-1.5)
        with self._patch_hold(0.0):
            enrolled = _async(
                opt._register_tail_final_candidate("bare_reroute"))
        self.assertTrue(enrolled)
        self.assertEqual(opt._final_candidates[0]["source_label"],
                         "bare_reroute")

    def test_unrouted_tail_candidate_rejected_by_register_gates(self):
        opt = self._opt(routed=False)          # tracker says NOT routed
        with self._patch_hold(0.02):
            enrolled = _async(
                opt._register_tail_final_candidate("tail_controller"))
        self.assertFalse(enrolled)
        self.assertEqual(opt._final_candidates, [])

    def test_hold_dirty_tail_candidate_rejected_by_register_gates(self):
        opt = self._opt(routed=True)
        with self._patch_hold(-0.05):          # hold dirty
            enrolled = _async(
                opt._register_tail_final_candidate("tail_controller"))
        self.assertFalse(enrolled)
        self.assertEqual(opt._final_candidates, [])

    def test_tail_skips_without_banked_best(self):
        opt = self._opt()
        opt._best_valid_dcp = None
        enrolled = _async(
            opt._register_tail_final_candidate("tail_controller"))
        self.assertFalse(enrolled)

    def test_tail_skips_on_reopen_error(self):
        opt = self._opt()
        opt.call_tool = mock.AsyncMock(return_value='{"error": "boom"}')
        with self._patch_hold(0.02):
            enrolled = _async(
                opt._register_tail_final_candidate("tail_controller"))
        self.assertFalse(enrolled)
        self.assertEqual(opt._final_candidates, [])


if __name__ == "__main__":
    unittest.main()
