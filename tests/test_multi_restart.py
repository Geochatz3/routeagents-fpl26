"""Unit tests for scripts/multi_restart_optimize.select_best (pure logic)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.multi_restart_optimize import select_best, ship_tier, should_stop_early


def _a(i, fmax, status="VALID_OPTIMIZED", exists=True):
    return {"i": i, "fmax": fmax, "status": status,
            "output": f"/tmp/mr_{i}.dcp", "exists": exists}


class SelectBestTests(unittest.TestCase):
    def test_none_when_empty(self):
        self.assertIsNone(select_best([]))

    def test_none_when_no_usable_outputs(self):
        # exists=False or fmax=None are not usable
        self.assertIsNone(select_best([
            _a(1, 100.0, exists=False),
            _a(2, None),
        ]))

    def test_picks_highest_fmax(self):
        best = select_best([_a(1, 0.0), _a(2, 113.0), _a(3, 50.0)])
        self.assertEqual(best["i"], 2)
        self.assertEqual(best["fmax"], 113.0)

    def test_best_of_n_recovers_unlucky_draw(self):
        best = select_best([_a(1, 0.0, status="VALID_FALLBACK_BASELINE"),
                            _a(2, 113.0)])
        self.assertEqual(best["fmax"], 113.0)

    def test_tie_prefers_optimized_over_fallback(self):
        best = select_best([
            _a(1, 50.0, status="VALID_FALLBACK_BASELINE"),
            _a(2, 50.0, status="VALID_OPTIMIZED"),
        ])
        self.assertEqual(best["i"], 2)

    def test_ignores_missing_output_even_if_higher_fmax(self):
        best = select_best([
            _a(1, 200.0, exists=False),  # higher fmax but no file
            _a(2, 90.0),
        ])
        self.assertEqual(best["i"], 2)

    def test_all_fallback_still_returns_one(self):
        # every attempt only matched baseline -> still return a valid output
        best = select_best([
            _a(1, 48.2, status="VALID_FALLBACK_BASELINE"),
            _a(2, 48.2, status="VALID_FALLBACK_BASELINE"),
        ])
        self.assertIsNotNone(best)

    # Selection follows the shipped status, not the best fmax tracked by the
    # agent. A fallback ships the unchanged input and therefore represents
    # no optimization regardless of its tracked fmax.

    def test_optimized_beats_higher_fmax_fallback(self):
        best = select_best([
            _a(1, 200.0, status="VALID_FALLBACK_BASELINE"),   # alpha == 0
            _a(2, 100.0, status="VALID_OPTIMIZED"),           # alpha > 0
        ])
        self.assertEqual(best["i"], 2)

    def test_no_edif_optimized_is_not_a_fallback(self):
        # VALID_OPTIMIZED_NO_EDIF is a valid, improved DCP (only the EDIF
        # sidecar write failed) — it must not be binned with the fallbacks.
        best = select_best([
            _a(1, 200.0, status="VALID_FALLBACK_BASELINE"),
            _a(2, 100.0, status="VALID_OPTIMIZED_NO_EDIF"),
        ])
        self.assertEqual(best["i"], 2)

    def test_no_edif_vs_optimized_ordering_is_unchanged_from_before_aug07(self):
        # Missing EDIF affects downstream validation but does not change this
        # selector's ranking between otherwise optimized artifacts.
        best = select_best([
            _a(1, 120.0, status="VALID_OPTIMIZED_NO_EDIF"),
            _a(2, 110.0, status="VALID_OPTIMIZED"),
        ])
        self.assertEqual(best["i"], 1)          # fmax still leads
        tie = select_best([
            _a(1, 120.0, status="VALID_OPTIMIZED_NO_EDIF"),
            _a(2, 120.0, status="VALID_OPTIMIZED"),
        ])
        self.assertEqual(tie["i"], 2)           # exact tie -> full OPTIMIZED

    def test_unknown_status_outranks_known_fallback(self):
        # A truncated status file is NOT evidence of a
        # fallback — the DCP may be a real win.
        best = select_best([
            _a(1, 200.0, status="VALID_FALLBACK_BASELINE"),
            _a(2, 100.0, status=None),
        ])
        self.assertEqual(best["i"], 2)

    def test_unknown_status_is_NOT_demoted_below_optimized(self):
        """Verify that an unknown final status is not ranked below an optimized
        artifact.

        Emergency metadata may contain a null status while token metadata still
        identifies the tracked best artifact. Treating that status as inferior
        could replace a higher-frequency result with a worse fallback.
        """
        best = select_best([
            _a(1, 200.0, status=None),                 # real win, killed early
            _a(2, 100.0, status="VALID_OPTIMIZED"),
        ])
        self.assertEqual(best["i"], 1)
        self.assertEqual(best["fmax"], 200.0)

    def test_fmax_still_decides_within_a_tier(self):
        best = select_best([_a(1, 90.0), _a(2, 113.0), _a(3, 50.0)])
        self.assertEqual(best["i"], 2)

    def test_all_fallback_falls_back_to_fmax_order(self):
        best = select_best([
            _a(1, 40.0, status="VALID_FALLBACK_BASELINE"),
            _a(2, 48.2, status="VALID_FALLBACK_BASELINE_PHASE1_FAILED"),
        ])
        self.assertEqual(best["i"], 2)

    def test_phase1_failed_and_no_improvement_are_tier_zero(self):
        for bad in ("VALID_FALLBACK_BASELINE_NO_EDIF",
                    "VALID_FALLBACK_BASELINE_PHASE1_FAILED",
                    "NO_IMPROVEMENT", "HARD_FAIL_NO_VALID_BASELINE"):
            best = select_best([_a(1, 300.0, status=bad),
                                _a(2, 10.0, status="VALID_OPTIMIZED")])
            self.assertEqual(best["i"], 2, bad)

    def test_earliest_attempt_wins_a_full_tie(self):
        best = select_best([_a(1, 50.0), _a(2, 50.0)])
        self.assertEqual(best["i"], 1)


class ShipTierTests(unittest.TestCase):
    def test_tiers(self):
        # ONLY two tiers: "known worth nothing" and everything else.
        self.assertEqual(ship_tier("VALID_OPTIMIZED"), 1)
        self.assertEqual(ship_tier("VALID_OPTIMIZED_NO_EDIF"), 1)
        self.assertEqual(ship_tier(None), 1)
        self.assertEqual(ship_tier(""), 1)
        self.assertEqual(ship_tier("SOMETHING_NEW"), 1)
        self.assertEqual(ship_tier("VALID_FALLBACK_BASELINE"), 0)
        self.assertEqual(ship_tier("VALID_FALLBACK_BASELINE_NO_EDIF"), 0)
        self.assertEqual(ship_tier("VALID_FALLBACK_BASELINE_PHASE1_FAILED"), 0)
        self.assertEqual(ship_tier("NO_IMPROVEMENT"), 0)
        self.assertEqual(ship_tier("HARD_FAIL_NO_VALID_BASELINE"), 0)


class ShouldStopEarlyTests(unittest.TestCase):
    def test_no_stop_before_min_attempts(self):
        self.assertFalse(should_stop_early([_a(1, 113.0)]))

    def test_stop_when_two_optimized_agree(self):
        # amd-style: 2 identical strong wins -> stop (save cost on winners)
        self.assertTrue(should_stop_early([_a(1, 407.8), _a(2, 407.8)]))

    def test_no_stop_on_two_fallbacks(self):
        self.assertFalse(should_stop_early([
            _a(1, 0.0, status="VALID_FALLBACK_BASELINE"),
            _a(2, 0.0, status="VALID_FALLBACK_BASELINE"),
        ]))

    def test_no_stop_on_single_lucky_then_low(self):
        # one high OPT + one low OPT not within eps of best -> not confirmed
        self.assertFalse(should_stop_early([_a(1, 338.9), _a(2, 297.1)]))

    def test_stop_after_confirmation_following_unlucky(self):
        self.assertTrue(should_stop_early([
            _a(1, 0.0, status="VALID_FALLBACK_BASELINE"),
            _a(2, 113.0), _a(3, 113.0),
        ]))

    def test_eps_tolerance(self):
        self.assertTrue(should_stop_early([_a(1, 113.0), _a(2, 112.5)], eps=1.0))


if __name__ == "__main__":
    unittest.main()


class TestAttemptCmdILS:
    def test_ils_flag_appends_make_var(self):
        from pathlib import Path
        from scripts.multi_restart_optimize import _attempt_cmd
        cmd = _attempt_cmd(Path("in.dcp"), Path("/tmp/o.dcp"), 1800.4, True)
        assert cmd[:2] == ["make", "run_once"]
        assert "ILS=1" in cmd
        assert "MAX_WALL=1800" in cmd

    def test_default_off_no_ils_var(self):
        from pathlib import Path
        from scripts.multi_restart_optimize import _attempt_cmd
        cmd = _attempt_cmd(Path("in.dcp"), Path("/tmp/o.dcp"), 1800.0, False)
        assert "ILS=1" not in cmd


class TestIncrementalRefresh:
    def test_scored_output_refreshed_before_attempt_2(self, tmp_path, monkeypatch):
        """Beta rules score the last best result ON DISK: the scored location
        must already hold attempt 1's DCP when attempt 2 starts, so a
        harness-side wall kill mid-attempt never discards a valid result."""
        from pathlib import Path
        import scripts.multi_restart_optimize as mr
        inp = tmp_path / "bench.dcp"; inp.write_bytes(b"x" * 64)
        final = tmp_path / "bench_optimized.dcp"
        run_dir = tmp_path / "dcp_optimizer_run-1"; run_dir.mkdir()

        seen_at_attempt_start = []
        calls = {"n": 0}
        def fake_attempt(cmd, repo=None, budget_s=None, attempt_i=None,
                         **kw):
            calls["n"] += 1
            seen_at_attempt_start.append(final.exists())
            out = [a.split("=", 1)[1] for a in cmd if a.startswith("OUTPUT=")][0]
            Path(out).write_bytes(b"dcp-attempt-%d" % calls["n"])
            return 0
        # Two attempts with DIFFERENT fmax so should_stop_early (needs 2 within
        # eps of best) never fires before attempt 2.
        fmaxes = iter([100.0, 50.0])
        monkeypatch.setattr(mr, "_run_attempt_process", fake_attempt)
        monkeypatch.setattr(mr, "_attribute_run_dir",
                            lambda base, before: run_dir)
        monkeypatch.setattr(mr, "_read_run_metrics",
                            lambda rd: {"fmax": next(fmaxes),
                                        "status": "VALID_OPTIMIZED", "cost": 0.01})
        s = mr.run(inp, final, total_wall=10_000, attempt_floor=1.0,
                   max_attempts=2, repo=tmp_path, cost_cap=0.85, polish=False)
        assert seen_at_attempt_start == [False, True]  # refreshed mid-loop
        assert final.exists()
        assert s["chosen"]["fmax"] == 100.0            # best kept, not last
        assert final.read_bytes() == b"dcp-attempt-1"


class TestWinnerPolish:
    def _setup(self, tmp_path):
        final = tmp_path / "bench_optimized.dcp"
        final.write_bytes(b"winner")
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "winner_polish.tcl").write_text("# stub")
        return final

    def test_improved_verdict_replaces_scored_file(self, tmp_path, monkeypatch):
        import scripts.multi_restart_optimize as mr
        final = self._setup(tmp_path)
        def fake_run(cmd, **kw):
            from pathlib import Path as _P
            _P(cmd[cmd.index("-tclargs") + 2]).write_bytes(b"polished")
            class R:
                stdout = "POLISH_BASE_WNS=-0.5\nPOLISH_NEW_WNS=-0.4\nPOLISH_VERDICT=IMPROVED delta=0.1\n"
                stderr = ""
            return R()
        monkeypatch.setattr(mr.subprocess, "run", fake_run)
        monkeypatch.setattr(mr, "_resolve_vivado", lambda: "/fake/vivado")
        assert mr.winner_polish(final, 900.0, tmp_path) is True
        assert final.read_bytes() == b"polished"

    def test_no_gain_keeps_winner(self, tmp_path, monkeypatch):
        import scripts.multi_restart_optimize as mr
        final = self._setup(tmp_path)
        def fake_run(cmd, **kw):
            class R:
                stdout = "POLISH_VERDICT=NO_GAIN\n"; stderr = ""
            return R()
        monkeypatch.setattr(mr.subprocess, "run", fake_run)
        monkeypatch.setattr(mr, "_resolve_vivado", lambda: "/fake/vivado")
        assert mr.winner_polish(final, 900.0, tmp_path) is False
        assert final.read_bytes() == b"winner"

    def test_timeout_keeps_winner(self, tmp_path, monkeypatch):
        import scripts.multi_restart_optimize as mr
        final = self._setup(tmp_path)
        def fake_run(cmd, **kw):
            raise mr.subprocess.TimeoutExpired(cmd="vivado", timeout=1)
        monkeypatch.setattr(mr.subprocess, "run", fake_run)
        monkeypatch.setattr(mr, "_resolve_vivado", lambda: "/fake/vivado")
        assert mr.winner_polish(final, 900.0, tmp_path) is False
        assert final.read_bytes() == b"winner"

    def test_skips_below_floor_or_without_vivado(self, tmp_path, monkeypatch):
        import scripts.multi_restart_optimize as mr
        final = self._setup(tmp_path)
        assert mr.winner_polish(final, 300.0, tmp_path) is False  # < floor
        monkeypatch.setattr(mr, "_resolve_vivado", lambda: None)
        assert mr.winner_polish(final, 900.0, tmp_path) is False  # no vivado
        assert final.read_bytes() == b"winner"


class TestSpreadAwareStop:
    def _att(self, fmax):
        return {"status": "VALID_OPTIMIZED", "fmax": fmax}

    def test_high_spread_requires_third_attempt(self):
        from scripts.multi_restart_optimize import should_stop_early
        two_agree = [self._att(100.0), self._att(100.2)]
        assert should_stop_early(two_agree) is True                  # default
        assert should_stop_early(two_agree, high_spread=True) is False
        three = two_agree + [self._att(100.1)]
        assert should_stop_early(three, high_spread=True) is True

    def test_read_run_spread(self, tmp_path):
        from scripts.multi_restart_optimize import _read_run_spread
        rd = tmp_path
        (rd / "decisions.jsonl").write_text(
            '{"action_label": "x"}\n'
            '{"critical_path_spread": 232.4, "action_label": "y"}\n')
        assert _read_run_spread(rd) == 232.4
        (rd / "decisions.jsonl").write_text("")
        assert _read_run_spread(rd) is None


class TestAtomicPublish:
    def test_publishes_content_and_bumps_mtime(self, tmp_path):
        """The eval scores the mtime-NEWEST <stem>_optimized*.dcp: a publish
        must land the bytes AND set mtime to now (copy2 alone preserves the
        source's possibly-minutes-old mtime)."""
        import os, time
        from scripts.multi_restart_optimize import _atomic_publish
        src = tmp_path / "attempt1.dcp"; src.write_bytes(b"best")
        old = time.time() - 3600
        os.utime(src, (old, old))
        dst = tmp_path / "bench_optimized.dcp"
        dst.write_bytes(b"stale")
        assert _atomic_publish(src, dst)
        assert dst.read_bytes() == b"best"
        assert dst.stat().st_mtime > old + 3000          # mtime = now, not src's
        assert not list(tmp_path.glob("*.tmp*"))         # temp cleaned up

    def test_returns_false_when_source_missing(self, tmp_path):
        from scripts.multi_restart_optimize import _atomic_publish
        assert not _atomic_publish(tmp_path / "nope.dcp",
                                   tmp_path / "out.dcp")
        assert not (tmp_path / "out.dcp").exists()


class TestEmergencyPublish:
    """Harness SIGTERM at the wall: the wrapper must never strand the
    in-flight attempt's emergency DCP in /tmp with nothing scored."""

    def test_publishes_inflight_when_nothing_scored(self, tmp_path):
        # An emergency artifact must have plausible DCP zip bytes.
        # Without a shipping manifest it is published as unverified rather
        # than discarded.
        from scripts.multi_restart_optimize import _emergency_publish
        cur = tmp_path / "mr_bench_1_1.dcp"
        cur.write_bytes(b"PK\x03\x04inflight")
        final = tmp_path / "bench_optimized.dcp"
        assert _emergency_publish(
            final, cur, wait_s=0.1) == "published_inflight_unverified"
        assert final.read_bytes() == b"PK\x03\x04inflight"

    def test_never_overwrites_existing_best(self, tmp_path):
        # The scored file is the best of COMPLETED attempts; the in-flight
        # emergency copy may be a mere baseline pass-through. Never-worse.
        from scripts.multi_restart_optimize import _emergency_publish
        cur = tmp_path / "mr_bench_1_2.dcp"; cur.write_bytes(b"baseline")
        final = tmp_path / "bench_optimized.dcp"; final.write_bytes(b"best")
        assert _emergency_publish(final, cur, wait_s=0.1) == "kept_existing"
        assert final.read_bytes() == b"best"

    def test_handles_missing_inflight(self, tmp_path):
        from scripts.multi_restart_optimize import _emergency_publish
        final = tmp_path / "bench_optimized.dcp"
        r = _emergency_publish(final, tmp_path / "never.dcp", wait_s=0.1)
        assert r == "inflight_never_landed"
        assert not final.exists()
        assert _emergency_publish(final, None, wait_s=0.1) == "nothing_to_publish"
        assert _emergency_publish(None, None, wait_s=0.1) == "no_final_output"

    def test_waits_for_agent_finalize_to_land(self, tmp_path):
        # The wrapper waits for the agent's signal handler to finalize the
        # attempt artifact. A shipping manifest then verifies its size and
        # checksum before publication.
        import threading
        from dcp_optimizer import _artifact_identity, _write_shipped_manifest
        from scripts.multi_restart_optimize import _emergency_publish
        cur = tmp_path / "mr_bench_1_1.dcp"
        final = tmp_path / "bench_optimized.dcp"

        def _land():
            cur.write_bytes(b"PK\x03\x04late")
            _write_shipped_manifest(cur, _artifact_identity(cur))

        t = threading.Timer(0.5, _land)
        t.start()
        try:
            assert _emergency_publish(
                final, cur, wait_s=5.0) == "published_inflight_verified"
            assert final.read_bytes() == b"PK\x03\x04late"
        finally:
            t.cancel()

    # A published fallback must not block a manifest-verified optimized
    # artifact finalized by an in-flight attempt.

    def test_verified_inflight_replaces_tier0_incumbent(self, tmp_path):
        from dcp_optimizer import _artifact_identity, _write_shipped_manifest
        from scripts.multi_restart_optimize import _emergency_publish
        cur = tmp_path / "mr_bench_1_2.dcp"
        cur.write_bytes(b"PK\x03\x04realwin")
        _write_shipped_manifest(cur, _artifact_identity(cur))
        final = tmp_path / "bench_optimized.dcp"
        final.write_bytes(b"baseline-passthrough")
        assert _emergency_publish(
            final, cur, wait_s=0.5, incumbent_tier=0,
        ) == "published_inflight_verified_over_tier0"
        assert final.read_bytes() == b"PK\x03\x04realwin"

    def test_tier0_incumbent_never_replaced_by_unverified_bytes(self, tmp_path):
        # A valid baseline beats plausible-but-unproven bytes: the 'PK'
        # deadline fallback must NOT fire against an incumbent (T5 garbage
        # injection discipline is unchanged).
        from scripts.multi_restart_optimize import _emergency_publish
        cur = tmp_path / "mr_bench_1_2.dcp"
        cur.write_bytes(b"PK\x03\x04unproven")
        final = tmp_path / "bench_optimized.dcp"
        final.write_bytes(b"baseline-passthrough")
        assert _emergency_publish(
            final, cur, wait_s=0.1, incumbent_tier=0,
        ) == "kept_existing_tier0"
        assert final.read_bytes() == b"baseline-passthrough"

    def test_tier1_incumbent_wins_even_against_verified_inflight(self, tmp_path):
        # Never-worse rule intact: a real (tier-1) incumbent is the best of
        # all COMPLETED attempts; the in-flight artifact may be worse.
        from dcp_optimizer import _artifact_identity, _write_shipped_manifest
        from scripts.multi_restart_optimize import _emergency_publish
        cur = tmp_path / "mr_bench_1_2.dcp"
        cur.write_bytes(b"PK\x03\x04laterwin")
        _write_shipped_manifest(cur, _artifact_identity(cur))
        final = tmp_path / "bench_optimized.dcp"
        final.write_bytes(b"real-best")
        assert _emergency_publish(
            final, cur, wait_s=0.5, incumbent_tier=1) == "kept_existing"
        assert final.read_bytes() == b"real-best"

    def test_unknown_incumbent_tier_keeps_existing(self, tmp_path):
        # incumbent_tier None (pre-publish state lost / old callsite): treat
        # the incumbent as real — identical to the earlier behavior.
        from scripts.multi_restart_optimize import _emergency_publish
        cur = tmp_path / "mr_bench_1_2.dcp"; cur.write_bytes(b"PK\x03\x04x")
        final = tmp_path / "bench_optimized.dcp"; final.write_bytes(b"best")
        assert _emergency_publish(final, cur, wait_s=0.1) == "kept_existing"
        assert final.read_bytes() == b"best"

    def test_tier0_incumbent_no_inflight_is_kept(self, tmp_path):
        from scripts.multi_restart_optimize import _emergency_publish
        final = tmp_path / "bench_optimized.dcp"
        final.write_bytes(b"baseline-passthrough")
        assert _emergency_publish(
            final, None, wait_s=0.1, incumbent_tier=0) == "kept_existing_tier0"
        assert final.read_bytes() == b"baseline-passthrough"


class TestWedgeGuard:
    """: an attempt that outlives its wall budget must be reaped —
    SIGTERM first (the agent's handler emergency-finalizes on it), SIGKILL
    only as the last resort. Previously subprocess.run had NO timeout and a
    wedged attempt silently converted best-of-N into best-of-1."""

    def test_normal_exit_untouched(self):
        import sys as _sys
        from scripts.multi_restart_optimize import _run_attempt_process
        rc = _run_attempt_process([_sys.executable, "-c", "print('ok')"],
                                  repo=".", budget_s=30, attempt_i=1)
        assert rc == 0

    @staticmethod
    def _make_cmd(tmp_path, py_code: str, token_path: str) -> list:
        """Build a command that reproduces the optimizer's make-shell-Python
        process topology.

        The literal `OUTPUT=<path>` appears only in make's arguments, while the
        shell stores the path in a variable and Python receives it as a
        separate option value. This ensures process guards detect the bare
        output path rather than relying on make-specific argument syntax.
        """
        import sys as _sys
        agent_py = tmp_path / "fake_agent.py"
        agent_py.write_text(py_code)
        mf = tmp_path / "Makefile.wedge"
        mf.write_text(
            "run:\n"
            f"\t_OUTPUT=\"$(OUTPUT)\"; {_sys.executable} "
            f"{agent_py} --output \"$$_OUTPUT\"; true\n"
        )
        return ["make", "-s", "-f", str(mf), "run",
                f"OUTPUT={token_path}"]

    def test_wedged_grandchild_gets_sigterm_and_its_handler_runs(
            self, tmp_path):
        import shutil as _shutil
        import pytest as _pytest
        if not _shutil.which("make"):
            _pytest.skip("needs make")
        from scripts.multi_restart_optimize import _run_attempt_process
        marker = tmp_path / "finalized.txt"
        # Grandchild sleeps far past its budget; on SIGTERM it writes the
        # marker (standing in for the agent's emergency finalize).
        code = (
            "import signal, sys, time\n"
            f"m = {str(marker)!r}\n"
            "def h(s, f):\n"
            "    open(m, 'w').write('finalized')\n"
            "    sys.exit(143)\n"
            "signal.signal(signal.SIGTERM, h)\n"
            "time.sleep(300)\n"
        )
        cmd = self._make_cmd(tmp_path, code, str(tmp_path / "mr_tok_1.dcp"))
        # Generous boot window (budget 3 + grace 2) — make+sh+python boot;
        # the review flagged a 1.0s window as flaky on a loaded box.
        rc = _run_attempt_process(cmd, repo=str(tmp_path), budget_s=3.0,
                                  attempt_i=2, grace_s=2.0)
        assert marker.exists(), (
            "SIGTERM must reach the python GRANDCHILD through make+sh "
            "(bare-path /proc scan) — neither make nor sh forwards it")
        # rc reflects make's own death (proc.terminate() SIGTERMs it too);
        # the wrapper never branches on the attempt rc — the artifacts the
        # finalize left on disk are what matter.
        assert rc is not None

    def test_sigterm_immune_grandchild_is_sigkilled(self, tmp_path,
                                                    monkeypatch):
        import shutil as _shutil
        import pytest as _pytest
        if not _shutil.which("make"):
            _pytest.skip("needs make")
        import subprocess as _sp
        import scripts.multi_restart_optimize as mro
        # Grandchild ignores SIGTERM; sh/make therefore never exit on their
        # own. The guard must SIGKILL the lineage. Patch ONLY the 90s
        # SIGTERM-grace wait to stay fast.
        real_wait = _sp.Popen.wait

        def fast_wait(self, timeout=None):
            if timeout == 90.0:
                timeout = 0.5
            return real_wait(self, timeout=timeout)

        monkeypatch.setattr(_sp.Popen, "wait", fast_wait)
        code = ("import signal, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "time.sleep(300)\n")
        tok = str(tmp_path / "mr_tok_immune_2.dcp")
        cmd = self._make_cmd(tmp_path, code, tok)
        rc = mro._run_attempt_process(cmd, repo=str(tmp_path), budget_s=3.0,
                                      attempt_i=3, grace_s=2.0)
        assert rc is not None and rc != 0
        # Nothing from this attempt's lineage may survive the guard —
        # scanned with the BARE path, i.e. what production scans with.
        import time as _t
        left = []
        deadline = _t.monotonic() + 5.0
        while _t.monotonic() < deadline:
            left = mro._pids_with_cmdline_token(tok)
            if not left:
                break
            _t.sleep(0.2)
        assert not left, f"lineage survived the wedge guard: {left}"


class TestPolishVerdictEchoMasking:
    """live finding: vivado -mode batch echoes every script line with a
    '# ' prefix BEFORE executing it; winner_polish.tcl's usage line contains
    the literal POLISH_VERDICT= string, so a naive first-match parsed the
    ECHO and the replacement NEVER fired (silent no-op since WS1c)."""

    def _setup(self, tmp_path):
        final = tmp_path / "bench_optimized.dcp"
        final.write_bytes(b"winner")
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "winner_polish.tcl").write_text("# stub")
        return final

    BATCH_OUT = (
        '# if {$argc < 2} { puts "POLISH_VERDICT=INVALID reason=usage"; exit 1 }\n'
        '#         puts "POLISH_VERDICT=IMPROVED delta=[expr {$best - $base}]"\n'
        '#         puts "POLISH_VERDICT=NO_GAIN"\n'
        'POLISH_BASE_WNS=-0.5\n'
    )

    def test_real_improved_not_masked_by_echo(self, tmp_path, monkeypatch):
        import scripts.multi_restart_optimize as mr
        final = self._setup(tmp_path)
        out = self.BATCH_OUT + "POLISH_VERDICT=IMPROVED delta=0.1\n"
        def fake_run(cmd, **kw):
            from pathlib import Path as _P
            _P(cmd[cmd.index("-tclargs") + 2]).write_bytes(b"polished")
            class R:
                stdout = out
                stderr = ""
            return R()
        monkeypatch.setattr(mr.subprocess, "run", fake_run)
        monkeypatch.setattr(mr, "_resolve_vivado", lambda: "/fake/vivado")
        assert mr.winner_polish(final, 900.0, tmp_path) is True
        assert final.read_bytes() == b"polished"

    def test_real_no_gain_not_masked_by_echo(self, tmp_path, monkeypatch):
        import scripts.multi_restart_optimize as mr
        final = self._setup(tmp_path)
        out = self.BATCH_OUT + "POLISH_VERDICT=NO_GAIN\n"
        def fake_run(cmd, **kw):
            from pathlib import Path as _P
            _P(cmd[cmd.index("-tclargs") + 2]).write_bytes(b"polished")
            class R:
                stdout = out
                stderr = ""
            return R()
        monkeypatch.setattr(mr.subprocess, "run", fake_run)
        monkeypatch.setattr(mr, "_resolve_vivado", lambda: "/fake/vivado")
        assert mr.winner_polish(final, 900.0, tmp_path) is False
        assert final.read_bytes() == b"winner"


from scripts import multi_restart_optimize as mro


# Feature-aware restart split: classify_design + attempt1_budget +
# --split-aware. Table-driven over the known v1.2.0 benchmark DCP sizes
# (os.stat only, zero Vivado).

# (filename, exact st_size bytes, expected size class)
KNOWN_DCP_SIZES = [
    ("vexriscv_re-place_2025.1.dcp",            1_815_178, "small"),
    ("vexriscv_re-place_v2_2025.1.dcp",         2_063_720, "small"),
    ("amd_mini-isp_2025.1.dcp",                 3_948_685, "small"),
    ("rosetta_3d-rendering_2025.1.dcp",         7_937_701, "small"),
    ("logicnets_jscl_2025.1.dcp",              13_535_583, "small"),
    ("rosetta_spam-filter_2025.1.dcp",         14_281_525, "small"),
    ("fir_systolic_transposed_routed.dcp",     15_989_412, "small"),
    ("rosetta_digit-recognition_2025.1.dcp",   17_946_974, "small"),
    ("rosetta_optical-flow_2025.1.dcp",        26_420_875, "medium"),
    ("vtr_mcml_2025.1_v2.dcp",                 31_026_227, "medium"),
    ("vtr_mcml_2025.1.dcp",                    32_996_381, "medium"),
    ("finn_radioml_2025.1.dcp",                51_824_135, "medium"),
    ("corescore_500_mod_2025.1.dcp",           66_436_556, "medium"),
    ("boom_soc_2025.1_v2.dcp",                141_418_072, "large"),
    ("boom_soc_2025.1.dcp",                   144_822_770, "large"),
    ("ispd16_example2_2025.1.dcp",            152_073_471, "large"),
]


def _sparse(path, size):
    """Create a sparse file with an exact st_size (no real disk cost)."""
    path.touch()
    os.truncate(path, size)
    return path


class TestClassifyDesign:
    def test_known_benchmark_sizes_table(self, tmp_path):
        for name, size, expected in KNOWN_DCP_SIZES:
            p = _sparse(tmp_path / name, size)
            got = mro.classify_design(p)
            assert got == expected, f"{name} ({size}B): {got} != {expected}"

    def test_boundaries(self, tmp_path):
        assert mro.classify_design(
            _sparse(tmp_path / "a.dcp", mro.SMALL_MAX_BYTES - 1)) == "small"
        assert mro.classify_design(
            _sparse(tmp_path / "b.dcp", mro.SMALL_MAX_BYTES)) == "medium"
        assert mro.classify_design(
            _sparse(tmp_path / "c.dcp", mro.MEDIUM_MAX_BYTES)) == "medium"
        assert mro.classify_design(
            _sparse(tmp_path / "d.dcp", mro.MEDIUM_MAX_BYTES + 1)) == "large"

    def test_missing_file_fails_open_to_large(self, tmp_path):
        # Fail-open toward the uncapped single-shot behavior.
        from pathlib import Path
        assert mro.classify_design(tmp_path / "nope.dcp") == "large"

    def test_unreadable_path_fails_open_to_large(self):
        # A path os.stat cannot even parse (embedded NUL) must not raise.
        from pathlib import Path
        assert mro.classify_design(Path("bad\0path.dcp")) == "large"


class TestAttempt1Budget:
    def test_small_cap(self):
        assert mro.attempt1_budget(3500.0, "small") == mro.ATTEMPT1_CAP_SMALL_S

    def test_medium_cap(self):
        assert mro.attempt1_budget(3500.0, "medium") == mro.ATTEMPT1_CAP_MEDIUM_S

    def test_large_uncapped(self):
        assert mro.attempt1_budget(3500.0, "large") == 3500.0

    def test_never_exceeds_total_wall(self):
        # Caps never RAISE the budget above what's actually available.
        assert mro.attempt1_budget(1500.0, "small") == 1500.0
        assert mro.attempt1_budget(2000.0, "medium") == 2000.0
        assert mro.attempt1_budget(900.0, "large") == 900.0

    def test_unknown_class_fails_open_uncapped(self):
        assert mro.attempt1_budget(3500.0, "weird") == 3500.0


class TestSplitAwareSlice:
    """run() attempt-1 MAX_WALL slice: capped only when split_aware AND i==1;
    attempts 2+ always get `remaining` unchanged."""

    def _run_capture(self, tmp_path, monkeypatch, dcp_bytes,
                     total_wall=10_000.0, **run_kwargs):
        from pathlib import Path
        import scripts.multi_restart_optimize as mr
        inp = _sparse(tmp_path / "bench.dcp", dcp_bytes)
        final = tmp_path / "bench_optimized.dcp"
        run_dir = tmp_path / "dcp_optimizer_run-1"
        run_dir.mkdir(exist_ok=True)
        max_walls = []
        calls = {"n": 0}

        def fake_attempt(cmd, repo=None, budget_s=None, attempt_i=None,
                         **kw):
            calls["n"] += 1
            mw = [a.split("=", 1)[1] for a in cmd if a.startswith("MAX_WALL=")][0]
            max_walls.append(int(mw))
            out = [a.split("=", 1)[1] for a in cmd if a.startswith("OUTPUT=")][0]
            Path(out).write_bytes(b"dcp-%d" % calls["n"])
            return 0

        # Different fmax per attempt so should_stop_early never fires early.
        fmaxes = iter([100.0, 50.0])
        monkeypatch.setattr(mr, "_run_attempt_process", fake_attempt)
        monkeypatch.setattr(mr, "_attribute_run_dir",
                            lambda base, before: run_dir)
        monkeypatch.setattr(mr, "_read_run_metrics",
                            lambda rd: {"fmax": next(fmaxes),
                                        "status": "VALID_OPTIMIZED",
                                        "cost": 0.01})
        # Frozen clock: remaining == total_wall on every loop iteration, so
        # the attempt-1 MAX_WALL is exactly comparable across runs.
        monkeypatch.setattr(mr.time, "monotonic", lambda: 1000.0)
        mr.run(inp, final, total_wall=total_wall, attempt_floor=1.0,
               max_attempts=2, repo=tmp_path, cost_cap=0.85, polish=False,
               **run_kwargs)
        return max_walls

    def test_split_aware_small_caps_attempt1_only(self, tmp_path, monkeypatch):
        # For this mid-sized input, only the first attempt is capped at 1,800 s.
        mws = self._run_capture(tmp_path, monkeypatch, 3_948_685,
                                split_aware=True)
        assert mws == [1800, 10_000]

    def test_split_aware_medium_caps_attempt1_only(self, tmp_path, monkeypatch):
        # For this larger input, only the first attempt is capped at 2,400 s.
        mws = self._run_capture(tmp_path, monkeypatch, 26_420_875,
                                split_aware=True)
        assert mws == [2400, 10_000]

    def test_split_aware_large_stays_uncapped(self, tmp_path, monkeypatch):
        # boom-class (>70MB) must keep the full attempt-1 budget (route alone
        # measured 2153.13s on the eval box, AWS leg 2).
        mws = self._run_capture(tmp_path, monkeypatch, 141_418_072,
                                split_aware=True)
        assert mws == [10_000, 10_000]

    def test_split_aware_never_exceeds_remaining(self, tmp_path, monkeypatch):
        # total_wall below the cap: slice is min(remaining, cap) == remaining.
        mws = self._run_capture(tmp_path, monkeypatch, 3_948_685,
                                total_wall=1500.0, split_aware=True)
        assert mws[0] == 1500

    def test_split_aware_off_is_uncapped(self, tmp_path, monkeypatch):
        # Explicit OFF: attempt 1 gets `remaining` exactly as today.
        mws = self._run_capture(tmp_path, monkeypatch, 3_948_685,
                                split_aware=False)
        assert mws == [10_000, 10_000]


class TestSplitAwareDefaultOffParity:
    """Verify that the default split-aware setting preserves disabled-mode
    wall-time allocation.

    The first attempt receives the same wall-time limit when the setting is
    omitted or explicitly disabled, for both small and large checkpoint
    fixtures.
    """

    def _capture(self, tmp_path, monkeypatch, dcp_bytes, **run_kwargs):
        return TestSplitAwareSlice()._run_capture(
            tmp_path, monkeypatch, dcp_bytes, **run_kwargs)

    def test_default_equals_explicit_false_small_dcp(self, tmp_path,
                                                     monkeypatch):
        d1 = tmp_path / "default"
        d2 = tmp_path / "explicit"
        d1.mkdir()
        d2.mkdir()
        default = self._capture(d1, monkeypatch, 3_948_685)  # unset
        explicit = self._capture(d2, monkeypatch, 3_948_685,
                                 split_aware=False)
        assert default == explicit == [10_000, 10_000]

    def test_default_equals_explicit_false_large_dcp(self, tmp_path,
                                                     monkeypatch):
        d1 = tmp_path / "default"
        d2 = tmp_path / "explicit"
        d1.mkdir()
        d2.mkdir()
        default = self._capture(d1, monkeypatch, 141_418_072)  # unset
        explicit = self._capture(d2, monkeypatch, 141_418_072,
                                 split_aware=False)
        assert default == explicit == [10_000, 10_000]


class TestSplitAwareFlag:
    def test_main_threads_split_aware_into_run(self, tmp_path, monkeypatch):
        import scripts.multi_restart_optimize as mr
        inp = tmp_path / "b.dcp"
        inp.write_bytes(b"x")
        seen = {}

        def fake_run(input_dcp, final_output, total_wall, attempt_floor,
                     max_attempts, repo, cost_cap=0.85, **kw):
            seen.clear()
            seen.update(kw)
            return {"chosen": None}

        monkeypatch.setattr(mr, "run", fake_run)
        monkeypatch.setattr(mr, "_install_signal_publisher", lambda: None)
        # FIX 2 (S2): chosen=None now triggers the wrapper-internal budgeted
        # fallback — stub it so no real subprocess launches from this test.
        monkeypatch.setattr(mr, "_run_internal_fallback",
                            lambda *args, **kw: 1)
        mr.main([str(inp), "--split-aware"])
        assert seen.get("split_aware") is True
        mr.main([str(inp)])
        assert seen.get("split_aware") is False


class TestShouldSkipTruncated(unittest.TestCase):
    """Truncation gate."""

    def _opt(self, elapsed, exists=True, status="VALID_OPTIMIZED"):
        return {"i": 1, "fmax": 500.0, "status": status, "exists": exists,
                "elapsed": elapsed, "output": "/tmp/x.dcp"}

    def test_record_run_logicnets_replay(self):
        # A restart using less than 75% of a completed attempt's runtime is
        # treated as truncated and skipped.
        skip, why = mro.should_skip_truncated([self._opt(2100.0)], 1461.0)
        self.assertTrue(skip)
        self.assertIn("1461", why)

    def test_allows_plausible_second_draw(self):
        # fast design: completed in 900s, 2700s remain -> full redraw fits.
        skip, _ = mro.should_skip_truncated([self._opt(900.0)], 2700.0)
        self.assertFalse(skip)

    def test_never_fires_without_optimized_incumbent(self):
        # incumbent is only a FALLBACK baseline: a truncated attempt could
        # still beat it -> always allow.
        skip, _ = mro.should_skip_truncated(
            [self._opt(2100.0, status="VALID_FALLBACK_BASELINE")], 500.0)
        self.assertFalse(skip)

    def test_ignores_missing_output_or_elapsed(self):
        skip, _ = mro.should_skip_truncated(
            [self._opt(2100.0, exists=False), self._opt(None)], 500.0)
        self.assertFalse(skip)

    def test_uses_fastest_completed_attempt(self):
        # attempts of 2100s and 1400s completed; remaining 1200:
        # 1200 >= 0.75*1400 = 1050 -> a redraw is plausible, allow.
        atts = [self._opt(2100.0), self._opt(1400.0)]
        skip, _ = mro.should_skip_truncated(atts, 1200.0)
        self.assertFalse(skip)
        # remaining 1000 < 1050 -> block.
        skip2, _ = mro.should_skip_truncated(atts, 1000.0)
        self.assertTrue(skip2)


class SplitAwareCapLeavesRoomForASecondDraw(unittest.TestCase):
    """Verify that the split-aware cap leaves enough time for a second attempt.

    The first-attempt ceiling is derived from the minimum remaining time
    required to start another attempt. This coupling prevents either threshold
    from silently making split-aware execution ineffective.
    """

    EVAL_WALL = 3500.0
    FLOOR = 1200.0

    def test_medium_cap_alone_would_forbid_attempt_2(self):
        """Pin the defect itself, so this reads as a fixed bug not a style choice."""
        uncorrected = mro.attempt1_budget(self.EVAL_WALL, "medium")
        self.assertLess(self.EVAL_WALL - uncorrected, self.FLOOR)

    def test_capped_classes_clear_BOTH_attempt_2_gates(self):
        """Floor AND truncation.

        The truncation gate (remaining >= 0.75 * elapsed) is the TIGHTER one on
        the eval wall — 2000s against the floor's 2300s — and is what actually
        blocked mediums. A fix that satisfied only the floor would have left
        --split-aware inert while looking correct.
        """
        for size_class in ("small", "medium"):
            with self.subTest(size_class=size_class):
                budget = mro.attempt1_budget(self.EVAL_WALL, size_class, self.FLOOR)
                remaining = self.EVAL_WALL - budget
                self.assertGreaterEqual(
                    remaining, self.FLOOR,
                    f"{size_class}: leaves {remaining:.0f}s < the "
                    f"{self.FLOOR:.0f}s attempt floor")
                # Evaluated exactly as the loop evaluates it: an attempt 1 that
                # spent its whole budget is the reference elapsed.
                skip, why = mro.should_skip_truncated(
                    [{"status": "VALID_OPTIMIZED", "exists": True,
                      "elapsed": budget}], remaining)
                self.assertFalse(
                    skip, f"{size_class}: {why} — the second draw this cap exists "
                          "to fund can never start")

    def test_large_is_still_uncapped(self):
        """Starving a boom-class attempt 1 is how the alpha=0 failure mode returned."""
        self.assertEqual(
            mro.attempt1_budget(self.EVAL_WALL, "large", self.FLOOR), self.EVAL_WALL)

    def test_table_is_unchanged_when_no_floor_is_supplied(self):
        self.assertEqual(mro.attempt1_budget(3500.0, "small"),
                         mro.ATTEMPT1_CAP_SMALL_S)
        self.assertEqual(mro.attempt1_budget(3500.0, "medium"),
                         mro.ATTEMPT1_CAP_MEDIUM_S)

    def test_never_returns_more_than_the_wall(self):
        for cls in ("small", "medium", "large"):
            with self.subTest(cls=cls):
                self.assertLessEqual(mro.attempt1_budget(900.0, cls, self.FLOOR), 900.0)

    def test_a_wall_too_short_to_split_does_not_go_negative(self):
        """total_wall <= floor: no room to reserve, so keep the plain cap."""
        self.assertEqual(mro.attempt1_budget(1000.0, "small", self.FLOOR), 1000.0)
