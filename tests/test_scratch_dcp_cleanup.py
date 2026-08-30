"""Per-attempt and polish scratch DCPs are removed once they can't be published.

DCPs are 100-300MB and every one of them lands in /tmp: one per attempt (up
to --max-attempts) plus one polish staging file per run. Nothing used to
remove them, so a 4-attempt run stranded roughly 0.5-1.5 GB and back-to-back
benchmarks on a single eval box accumulated until /tmp filled -- at which
point _atomic_publish's temp copy is the first thing that fails, which loses
the winner.

The ordering that matters: a scratch file is only ever dropped AFTER the
publish that copies it, and never while it is still the selected best or the
artifact the signal handler might ship.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.multi_restart_optimize as mro  # noqa: E402


class TestDiscardHelper:
    def test_removes_an_existing_file(self, tmp_path):
        f = tmp_path / "a.dcp"
        f.write_bytes(b"x")
        mro._discard_scratch_dcp(f)
        assert not f.exists()

    def test_missing_file_is_not_an_error(self, tmp_path):
        mro._discard_scratch_dcp(tmp_path / "never-existed.dcp")

    def test_never_raises(self, tmp_path):
        # Losing a scratch file must not take the run down, so a directory,
        # a bad type, and None all have to pass through quietly.
        mro._discard_scratch_dcp(tmp_path)
        mro._discard_scratch_dcp(object())


class TestAttemptScratchIsSweptUp:
    def _drive(self, tmp_path, monkeypatch, fmaxes):
        inp = tmp_path / "bench.dcp"
        inp.write_bytes(b"x" * 64)
        final = tmp_path / "bench_optimized.dcp"
        calls = {"n": 0}
        made: list[Path] = []

        def fake_attempt(cmd, repo=None, budget_s=None, attempt_i=None, **kw):
            calls["n"] += 1
            out = [a.split("=", 1)[1] for a in cmd if a.startswith("OUTPUT=")][0]
            Path(out).write_bytes(b"dcp-%d" % calls["n"])
            made.append(Path(out))
            return 0

        def fake_attribute(base, before):
            d = tmp_path / f"dcp_optimizer_run-{calls['n']}"
            d.mkdir(exist_ok=True)
            return d

        fs = iter(fmaxes)
        monkeypatch.setattr(mro, "_run_attempt_process", fake_attempt)
        monkeypatch.setattr(mro, "_attribute_run_dir", fake_attribute)
        monkeypatch.setattr(mro, "_read_run_metrics",
                            lambda rd: {"fmax": next(fs),
                                        "status": "VALID_OPTIMIZED",
                                        "elapsed": None, "cost": 0.1})
        summary = mro.run(inp, final, total_wall=100_000, attempt_floor=1.0,
                          max_attempts=len(fmaxes), repo=tmp_path,
                          cost_cap=99.0, polish=False)
        return summary, final, made

    def test_no_attempt_dcp_survives_the_run(self, tmp_path, monkeypatch):
        # Divergent fmaxes so no early-stop fires and every attempt runs.
        summary, final, made = self._drive(
            tmp_path, monkeypatch, [100.0, 200.0, 150.0, 300.0])
        assert len(made) == 4
        assert final.exists(), "the scored artifact must still be published"
        survivors = [p for p in made if p.exists()]
        assert survivors == [], f"scratch DCPs left in /tmp: {survivors}"

    def test_the_winner_is_published_before_its_scratch_is_dropped(
            self, tmp_path, monkeypatch):
        summary, final, made = self._drive(
            tmp_path, monkeypatch, [100.0, 500.0, 150.0])
        # Attempt 2 won; its bytes have to be what landed at the scored path.
        assert summary["chosen"]["i"] == 2
        assert final.read_bytes() == b"dcp-2"
        assert not Path(summary["chosen"]["output"]).exists()

    def test_summary_still_records_what_each_attempt_produced(
            self, tmp_path, monkeypatch):
        # `exists` is a record of the attempt, not a claim about /tmp after.
        summary, _final, _made = self._drive(
            tmp_path, monkeypatch, [100.0, 200.0])
        assert [a["exists"] for a in summary["attempts"]] == [True, True]

    def test_scratch_is_kept_when_nothing_could_be_published(
            self, tmp_path, monkeypatch):
        # No usable output -> final_output absent -> the sweep must not run,
        # so a human can still find whatever the attempts left behind.
        inp = tmp_path / "bench.dcp"
        inp.write_bytes(b"x" * 64)
        final = tmp_path / "bench_optimized.dcp"
        made: list[Path] = []

        def fake_attempt(cmd, repo=None, budget_s=None, attempt_i=None, **kw):
            out = [a.split("=", 1)[1] for a in cmd if a.startswith("OUTPUT=")][0]
            made.append(Path(out))
            return 1  # never writes the output

        monkeypatch.setattr(mro, "_run_attempt_process", fake_attempt)
        monkeypatch.setattr(mro, "_attribute_run_dir", lambda b, f: None)
        monkeypatch.setattr(mro, "_read_run_metrics",
                            lambda rd: {"fmax": None, "status": None,
                                        "elapsed": None, "cost": 0.0})
        monkeypatch.setattr(mro, "_run_internal_fallback",
                            lambda *a, **k: 1)
        summary = mro.run(inp, final, total_wall=100_000, attempt_floor=1.0,
                          max_attempts=1, repo=tmp_path, cost_cap=99.0,
                          polish=False)
        assert summary["chosen"] is None
        assert not final.exists()
        for p in made:
            mro._discard_scratch_dcp(p)


class TestPolishStagingIsAlwaysDropped:
    def _polish(self, tmp_path, monkeypatch, stdout, boom=None):
        final = tmp_path / "out.dcp"
        final.write_bytes(b"winner")
        staged = Path("/tmp") / f"mr_polish_{os.getpid()}.dcp"
        staged.unlink(missing_ok=True)

        def fake_run(cmd, **kw):
            staged.write_bytes(b"polished")
            if boom is not None:
                raise boom
            return subprocess.CompletedProcess(cmd, 0, stdout, "")

        monkeypatch.setattr(mro, "_resolve_vivado", lambda: "/bin/true")
        monkeypatch.setattr(mro.subprocess, "run", fake_run)
        improved = mro.winner_polish(final, 100_000.0, ROOT)
        return improved, final, staged

    def test_dropped_after_a_no_gain_verdict(self, tmp_path, monkeypatch):
        improved, _final, staged = self._polish(
            tmp_path, monkeypatch, "POLISH_VERDICT=NO_GAIN\n")
        assert improved is False
        assert not staged.exists()

    def test_dropped_after_it_is_published(self, tmp_path, monkeypatch):
        improved, final, staged = self._polish(
            tmp_path, monkeypatch, "POLISH_VERDICT=IMPROVED delta=0.3\n")
        assert improved is True
        assert final.read_bytes() == b"polished", (
            "the staging file must be published before it is dropped")
        assert not staged.exists()

    def test_dropped_after_a_timeout(self, tmp_path, monkeypatch):
        improved, _final, staged = self._polish(
            tmp_path, monkeypatch, "",
            boom=subprocess.TimeoutExpired("vivado", 1))
        assert improved is False
        assert not staged.exists()

    def test_dropped_after_an_unexpected_failure(self, tmp_path, monkeypatch):
        improved, _final, staged = self._polish(
            tmp_path, monkeypatch, "", boom=OSError("vivado died"))
        assert improved is False
        assert not staged.exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
