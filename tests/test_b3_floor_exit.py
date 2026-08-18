"""Tests for the v5.4 B3-FLOOR-EXIT chain.

Three pieces: (1) the NO_EDIF-aware convergence stop (defect fix — the
exact-match status filter gave zero confirmations on bit-identical NO_EDIF
attempts, burning a third identical attempt on the v5.3 gate's mini-ISP);
(2) the wrapper honoring the b3_floor_saturated.token sentinel — stop the
attempt loop after the attesting attempt and skip winner-polish; (3) the
optimizer-side token writer condition. The B3 unmeasured-hold adopt guard
lives in test_deep_replace_b3_smallfloor.py.

The loop tests drive the REAL run() with only the process/IO seams stubbed
(_run_attempt_process is the sanctioned seam — session lesson aug08), so the
sentinel break, the cost accounting, and the convergence stop all execute
their real arithmetic.
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.multi_restart_optimize as mr


def _a(i, fmax, status="VALID_OPTIMIZED"):
    return {"i": i, "fmax": fmax, "status": status, "exists": True,
            "output": f"/tmp/mr_{i}.dcp"}


class StopEarlyNoEdifTests(unittest.TestCase):
    def test_no_edif_pair_confirms(self):
        self.assertTrue(mr.should_stop_early(
            [_a(1, 413.736, "VALID_OPTIMIZED_NO_EDIF"),
             _a(2, 413.736, "VALID_OPTIMIZED_NO_EDIF")]))

    def test_mixed_tier_pair_confirms(self):
        self.assertTrue(mr.should_stop_early(
            [_a(1, 413.736, "VALID_OPTIMIZED_NO_EDIF"),
             _a(2, 413.736, "VALID_OPTIMIZED")]))

    def test_single_attempt_never_confirms(self):
        self.assertFalse(mr.should_stop_early(
            [_a(1, 413.736, "VALID_OPTIMIZED_NO_EDIF")]))

    def test_fallback_pair_still_does_not_confirm(self):
        self.assertFalse(mr.should_stop_early(
            [_a(1, 307.1, "VALID_FALLBACK_BASELINE"),
             _a(2, 307.1, "VALID_FALLBACK_BASELINE")]))

    def test_divergent_no_edif_pair_does_not_confirm(self):
        self.assertFalse(mr.should_stop_early(
            [_a(1, 404.2, "VALID_OPTIMIZED_NO_EDIF"),
             _a(2, 413.7, "VALID_OPTIMIZED_NO_EDIF")]))


class _LoopHarness:
    """Stubs the process/IO seams of run(); everything else is real.

    Single-mode (token=bool): every attempt reports the same metrics and
    shares one run dir. Sequence-mode (fmax_seq + token_attempts): attempt i
    gets fmax_seq[i-1] and its own run dir; the token exists only in the run
    dirs of the attempts listed in token_attempts.
    """

    def __init__(self, tmp: Path, token: bool = False, fmax_seq=None,
                 token_attempts=()):
        self.tmp = tmp
        self.fmax_seq = fmax_seq
        self.token_attempts = set(token_attempts)
        self.rds = []
        for i in range(1, (len(fmax_seq) if fmax_seq else 1) + 1):
            rd = tmp / f"rd{i}"
            rd.mkdir()
            if (fmax_seq is None and token) or (i in self.token_attempts):
                (rd / "b3_floor_saturated.token").write_text(
                    "b3_floor_wns=-0.85 final_best_wns=-0.847\n")
            self.rds.append(rd)
        self.rd = self.rds[0]
        self.attempts = 0
        self.polish_calls = 0
        self.saved = {}
        self.made = []

    def install(self):
        self.saved = {
            "_run_attempt_process": mr._run_attempt_process,
            "_wrapper_run_dir_base": mr._wrapper_run_dir_base,
            "_snapshot_run_dirs": mr._snapshot_run_dirs,
            "_attribute_run_dir": mr._attribute_run_dir,
            "_read_run_metrics": mr._read_run_metrics,
            "_atomic_publish": mr._atomic_publish,
            "winner_polish": mr.winner_polish,
        }
        mr._run_attempt_process = self._attempt
        mr._wrapper_run_dir_base = lambda repo: self.tmp
        mr._snapshot_run_dirs = lambda base: set()
        mr._attribute_run_dir = lambda base, before: self._cur_rd()
        mr._read_run_metrics = lambda rd: self._metrics()
        mr._atomic_publish = lambda src, dst: True
        mr.winner_polish = self._polish

    def _cur_rd(self):
        idx = min(self.attempts, len(self.rds)) - 1
        return self.rds[max(idx, 0)]

    def _metrics(self):
        fmax = 413.736
        if self.fmax_seq:
            fmax = self.fmax_seq[min(self.attempts, len(self.fmax_seq)) - 1]
        return {"fmax": fmax, "status": "VALID_OPTIMIZED_NO_EDIF",
                "elapsed": 100.0, "cost": 0.08, "initial_fmax": 307.13}

    def restore(self):
        for k, v in self.saved.items():
            setattr(mr, k, v)

    def _attempt(self, cmd, repo, budget, i):
        self.attempts += 1
        # run() hands the agent an output path inside the command; create it
        # so the attempt registers as usable (exists=True) like a real run.
        for tok in (cmd if isinstance(cmd, (list, tuple)) else str(cmd).split()):
            s = str(tok)
            if "/tmp/mr_" in s and s.endswith(".dcp"):
                if "=" in s:  # make-var form, e.g. OUTPUT=/tmp/mr_x_1.dcp
                    s = s.split("=", 1)[1]
                Path(s).write_text("dcp")
                self.made.append(Path(s))

    def _polish(self, final_output, leftover, repo, report=None):
        self.polish_calls += 1
        return False

    def run(self):
        return mr.run(self.tmp / "in.dcp", self.tmp / "out.dcp",
                      total_wall=3500.0, attempt_floor=1200.0,
                      max_attempts=4, repo=self.tmp, polish=True)


def _with_env(key, val):
    class _Ctx:
        def __enter__(self):
            self.prev = os.environ.get(key)
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

        def __exit__(self, *a):
            if self.prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = self.prev
    return _Ctx()


class B3FloorExitLoopTests(unittest.TestCase):
    def _drive(self, token, env):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            h = _LoopHarness(Path(td), token=token)
            h.install()
            try:
                with _with_env("FPL26_B3_FLOOR_EXIT", env):
                    summary = h.run()
            finally:
                h.restore()
                for p in h.made:
                    try:
                        p.unlink()
                    except OSError:
                        pass
            return h, summary

    def test_sentinel_stops_after_one_attempt_and_skips_polish(self):
        h, summary = self._drive(token=True, env="1")
        self.assertEqual(h.attempts, 1)
        self.assertEqual(len(summary["attempts"]), 1)
        self.assertEqual(h.polish_calls, 0)

    def test_env_off_falls_back_to_the_convergence_stop(self):
        # Two bit-identical NO_EDIF attempts confirm (the v5.4 status fix);
        # the token alone must do nothing without the env arm.
        h, summary = self._drive(token=True, env=None)
        self.assertEqual(h.attempts, 2)
        self.assertEqual(h.polish_calls, 1)

    def test_no_token_uses_the_convergence_stop_even_when_armed(self):
        h, summary = self._drive(token=False, env="1")
        self.assertEqual(h.attempts, 2)
        self.assertEqual(h.polish_calls, 1)

    def test_token_on_a_non_best_attempt_does_not_break(self):
        # review-1 (v5.4) finding 1: attempt 1 beats the floor (no token);
        # attempt 2 lands on the floor and carries the token. The break must
        # NOT fire — the design has demonstrated stochastic upside beyond
        # the floor, so restarts and polish keep their value.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            h = _LoopHarness(Path(td), fmax_seq=[500.0, 413.736, 413.736],
                             token_attempts={2, 3})
            h.install()
            try:
                with _with_env("FPL26_B3_FLOOR_EXIT", "1"):
                    summary = h.run()
            finally:
                h.restore()
                for p in h.made:
                    try:
                        p.unlink()
                    except OSError:
                        pass
            # no B3 break on attempts 2+ (their fmax is 86 MHz below the
            # best); the loop runs on and the convergence stop needs two
            # attempts within eps of the BEST (500), which never happens ->
            # the full max_attempts=4 run, then polish.
            self.assertEqual(h.attempts, 4)
            self.assertEqual(h.polish_calls, 1)

    def test_eps_tie_token_attempt_defers_to_the_convergence_stop(self):
        # review-2 (v5.4.1) F1 repro: attempt 2 carries the token and sits
        # 0.76 MHz BELOW attempt 1 — inside should_stop_early's eps, so the
        # loop still stops at 2 via CONVERGENCE, but the polish must NOT be
        # skipped: the winner is attempt 1's artifact, and the floor
        # attestation does not cover it.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            h = _LoopHarness(Path(td), fmax_seq=[414.5, 413.736],
                             token_attempts={2})
            h.install()
            try:
                with _with_env("FPL26_B3_FLOOR_EXIT", "1"):
                    summary = h.run()
            finally:
                h.restore()
                for p in h.made:
                    try:
                        p.unlink()
                    except OSError:
                        pass
            self.assertEqual(h.attempts, 2)       # convergence stop
            self.assertEqual(h.polish_calls, 1)   # polish NOT skipped
            self.assertEqual(summary["chosen"]["i"], 1)


class TokenWriterTests(unittest.TestCase):
    def _stub(self, tmp, floor, best):
        import dcp_optimizer as dco

        class _S:
            pass

        s = _S()
        s._b3_floor_wns = floor
        s.best_wns = best
        s.run_dir = tmp
        s.final_status = "VALID_OPTIMIZED"
        dco.DCPOptimizer._maybe_write_b3_floor_token(s)
        return (Path(tmp) / "b3_floor_saturated.token")

    def test_written_when_final_within_margin_of_floor(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tok = self._stub(td, floor=-0.850, best=-0.847)
            self.assertTrue(tok.exists())
            self.assertIn("b3_floor_wns=-0.85", tok.read_text())

    def test_not_written_when_tail_improved_beyond_margin(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tok = self._stub(td, floor=-0.850, best=-0.820)
            self.assertFalse(tok.exists())

    def test_not_written_without_a_b3_floor(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tok = self._stub(td, floor=None, best=-0.847)
            self.assertFalse(tok.exists())

    def test_not_written_when_shipped_best_is_worse_than_the_floor(self):
        # review-1 (v5.4) finding 4: a regressed ship must never attest —
        # the abs() guard, independent of the promotion invariant.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tok = self._stub(td, floor=-0.850, best=-0.905)
            self.assertFalse(tok.exists())


if __name__ == "__main__":
    unittest.main()
