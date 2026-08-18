"""Tests for scripts/join_run_qor.py — offline join harness.

The script's `main()` is exercised end-to-end against a temporary run
directory + temporary episode_store.  No Vivado is invoked.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import join_run_qor


def _stub_qor_json(long_levels="4 1 1 2"):
    return {
        "Report Information": {
            "Tool Version": "Vivado v.2025.1",
            "Design State": "Physopt postRoute",
        },
        "Design QoR Summary": [
            {
                "Task Name": "route_design",
                "Long Cong Level N-E-S-W": long_levels,
                "Global Cong Level N-E-S-W": "3 2 1 0",
                "Short Cong Level N-E-S-W": "1 1 1 1",
            },
        ],
    }


def _episode(eid, design, dcp_path, run_id="dcp_optimizer_run-FAKE",
             start_features=None):
    return {
        "schema_version": 1, "episode_id": eid, "run_id": run_id,
        "design_name": design,
        "start_features": start_features or {"lut_count": 1000,
                                             "critical_path_spread": 100.0},
        "outcome": {"output_dcp_path": dcp_path},
        "source": {"decisions_jsonl_path": f"/runs/{run_id}/decisions.jsonl",
                   "record_count": 1},
    }


class JoinRunQorScriptTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run_dir = self.root / "run_one"
        self.run_dir.mkdir()
        self.store = self.root / "episode_store.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def _write_store(self, episodes):
        self.store.write_text(
            "\n".join(json.dumps(e) for e in episodes) + "\n"
        )

    def _write_json(self, name, body, subdir=None):
        d = self.run_dir if subdir is None else (self.run_dir / subdir)
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{name}.qor.json"
        p.write_text(json.dumps(body))
        return p

    def _run(self, *cli_args):
        argv = list(cli_args)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            rc = join_run_qor.main(argv)
            return rc, sys.stdout.getvalue(), sys.stderr.getvalue()
        finally:
            sys.stdout, sys.stderr = old_out, old_err

    # ----- happy paths ----------------------------------------------------

    def test_single_runs_dir_joins_one_episode(self):
        self._write_store([
            _episode("E1", "x_design",
                     dcp_path=str(self.run_dir / "x_design_optimized.dcp")),
        ])
        self._write_json("x_design_optimized", _stub_qor_json())
        rc, out, err = self._run(
            "--run-dir", str(self.run_dir),
            "--episode-store", str(self.store),
            "--apply",
        )
        self.assertEqual(rc, 0, err)
        self.assertIn("JOINED", out)
        # episode_store mutated.
        stored = json.loads(self.store.read_text().splitlines()[0])
        self.assertIn("qor_route_bound_score", stored["start_features"])

    def test_repeated_join_is_idempotent(self):
        self._write_store([
            _episode("E1", "x_design",
                     dcp_path=str(self.run_dir / "x_design_optimized.dcp")),
        ])
        self._write_json("x_design_optimized", _stub_qor_json())
        self._run("--run-dir", str(self.run_dir),
                  "--episode-store", str(self.store), "--apply")
        first = self.store.read_text()
        self._run("--run-dir", str(self.run_dir),
                  "--episode-store", str(self.store), "--apply")
        second = self.store.read_text()
        self.assertEqual(first, second)

    def test_dry_run_does_not_mutate(self):
        self._write_store([
            _episode("E1", "x_design",
                     dcp_path=str(self.run_dir / "x_design_optimized.dcp")),
        ])
        self._write_json("x_design_optimized", _stub_qor_json())
        before = self.store.read_text()
        self._run("--run-dir", str(self.run_dir),
                  "--episode-store", str(self.store))
        self.assertEqual(self.store.read_text(), before)

    # ----- safety paths ---------------------------------------------------

    def test_ambiguous_match_refuses_join(self):
        # Two episodes both look like they could match.
        self._write_store([
            _episode("E1", "x_design",
                     dcp_path=str(self.run_dir / "x_design_optimized.dcp")),
            _episode("E2", "x_design",
                     dcp_path=str(self.run_dir / "x_design_optimized.dcp")),
        ])
        self._write_json("x_design_optimized", _stub_qor_json())
        rc, out, _err = self._run(
            "--run-dir", str(self.run_dir),
            "--episode-store", str(self.store), "--apply",
        )
        self.assertEqual(rc, 3)
        self.assertIn("AMBIGUOUS", out)
        # episode_store unchanged.
        stored = json.loads(self.store.read_text().splitlines()[0])
        self.assertNotIn("qor_route_bound_score", stored["start_features"])

    def test_design_name_only_match_refuses_join_by_default(self):
        # No output_dcp_path / run_id overlap — only design name matches.
        self._write_store([
            _episode("E1", "x_design", dcp_path=None,
                     run_id="dcp_optimizer_run-UNRELATED"),
        ])
        self._write_json("x_design", _stub_qor_json())
        rc, out, _err = self._run(
            "--run-dir", str(self.run_dir),
            "--episode-store", str(self.store), "--apply",
        )
        self.assertEqual(rc, 4)
        self.assertIn("REPORT_ONLY", out)
        stored = json.loads(self.store.read_text().splitlines()[0])
        self.assertNotIn("qor_route_bound_score", stored["start_features"])

    def test_design_name_only_can_be_promoted_with_flag(self):
        self._write_store([
            _episode("E1", "x_design", dcp_path=None,
                     run_id="dcp_optimizer_run-UNRELATED"),
        ])
        self._write_json("x_design", _stub_qor_json())
        rc, out, _err = self._run(
            "--run-dir", str(self.run_dir),
            "--episode-store", str(self.store), "--apply",
            "--allow-design-name-hint-join",
        )
        self.assertEqual(rc, 0)
        self.assertIn("JOINED", out)

    def test_unmatched_does_not_write(self):
        self._write_store([
            _episode("E1", "other_design",
                     dcp_path=str(self.run_dir / "other_design_optimized.dcp")),
        ])
        self._write_json("not_in_store", _stub_qor_json())
        rc, out, _err = self._run(
            "--run-dir", str(self.run_dir),
            "--episode-store", str(self.store), "--apply",
        )
        self.assertEqual(rc, 4)
        self.assertIn("UNMATCHED", out)
        stored = json.loads(self.store.read_text().splitlines()[0])
        self.assertNotIn("qor_route_bound_score", stored["start_features"])

    def test_other_start_features_preserved(self):
        self._write_store([
            _episode("E1", "x_design",
                     dcp_path=str(self.run_dir / "x_design_optimized.dcp"),
                     start_features={"lut_count": 9001,
                                     "pathology": "HIGH_FANOUT_DRIVER",
                                     "critical_path_spread": 50.0}),
        ])
        self._write_json("x_design_optimized", _stub_qor_json())
        self._run("--run-dir", str(self.run_dir),
                  "--episode-store", str(self.store), "--apply")
        stored = json.loads(self.store.read_text().splitlines()[0])
        sf = stored["start_features"]
        self.assertEqual(sf["lut_count"], 9001)
        self.assertEqual(sf["pathology"], "HIGH_FANOUT_DRIVER")
        self.assertIn("qor_route_bound_score", sf)

    def test_submission_episode_store_path_refused(self):
        bad = self.root / "submission" / "policy_memory" / "episode_store.jsonl"
        bad.parent.mkdir(parents=True)
        bad.write_text("")
        self._write_json("x", _stub_qor_json())
        rc, _out, err = self._run(
            "--run-dir", str(self.run_dir),
            "--episode-store", str(bad), "--apply",
        )
        # Script must exit with a non-zero code, NOT write into the
        # submission-tree path, and surface the refusal to stderr.
        self.assertEqual(rc, 5)
        self.assertIn("refused", err.lower())
        self.assertEqual(bad.read_text(), "")

    def test_finds_jsons_in_subdirectories(self):
        self._write_store([
            _episode("E1", "x_design",
                     dcp_path=str(self.run_dir / "x_design_optimized.dcp")),
        ])
        self._write_json("x_design_optimized", _stub_qor_json(), subdir="qor")
        rc, out, _err = self._run(
            "--run-dir", str(self.run_dir),
            "--episode-store", str(self.store), "--apply",
        )
        self.assertEqual(rc, 0)
        self.assertIn("JOINED", out)

    def test_no_jsons_returns_nonzero(self):
        self._write_store([_episode("E1", "x", dcp_path=None)])
        rc, _out, _err = self._run(
            "--run-dir", str(self.run_dir),
            "--episode-store", str(self.store),
        )
        self.assertEqual(rc, 1)

    def test_missing_episode_store_returns_nonzero(self):
        rc, _out, _err = self._run(
            "--run-dir", str(self.run_dir),
            "--episode-store", str(self.root / "ghost.jsonl"),
        )
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
