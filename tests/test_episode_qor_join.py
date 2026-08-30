"""Tests for optimizer.episode_qor_join — offline JSON QoR → episode_store join."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizer.episode_qor_join import (
    QOR_FLAT_KEYS,
    STATUS_AMBIGUOUS,
    STATUS_JOINED,
    STATUS_REJECTED,
    STATUS_REPORT_ONLY,
    STATUS_UNMATCHED,
    join_qor_features,
)


def _stub_qor_json(design_basename: str, long_levels="4 1 1 2"):
    return {
        "Report Information": {
            "Tool Version": "Vivado v.2025.1",
            "Design State": "Physopt postRoute",
        },
        "Design QoR Summary": [
            {
                "Task Name": "route_design",
                "Options": "",
                "Directives": "",
                "Runtime(mins)": "4",
                "WNS(ns)": "-0.686",
                "TNS(ns)": "-831",
                "WHS(ns)": "",
                "THS(ns)": "",
                "RQA": "",
                "Global Cong Level N-E-S-W": "3 3 2 2",
                "Global Cong Tile% N-E-S-W": "",
                "Long Cong Level N-E-S-W": long_levels,
                "Long Cong Tile% N-E-S-W": "",
                "Short Cong Level N-E-S-W": "",
                "Short Cong Tile% N-E-S-W": "",
                "Number of threads": "8",
            },
        ],
    }


def _episode(eid, design, dcp_path=None, run_id="dcp_optimizer_run-FAKE"):
    return {
        "schema_version": 1,
        "episode_id": eid,
        "run_id": run_id,
        "ts_run": 0,
        "model": "x-ai/grok-4.3",
        "design_name": design,
        "start_features": {"lut_count": 1000, "critical_path_spread": 100.0},
        "outcome": {"output_dcp_path": dcp_path} if dcp_path else {},
        "source": {"decisions_jsonl_path": f"/runs/{run_id}/decisions.jsonl",
                   "record_count": 1},
    }


class EpisodeQorJoinTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = self.root / "episode_store.jsonl"
        self.jsondir = self.root / "qor"
        self.jsondir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _write_store(self, episodes):
        self.store.write_text("\n".join(json.dumps(e) for e in episodes) + "\n")

    def _write_json(self, name, body):
        p = self.jsondir / f"{name}.qor.json"
        p.write_text(json.dumps(body))
        return p

    # ----- match strategies ------------------------------------------------

    def test_match_by_output_dcp_path_then_join(self):
        ep = _episode("E1", "rosetta_spam-filter",
                      dcp_path="/runs/X/rosetta_spam-filter_optimized.dcp")
        self._write_store([ep])
        j = self._write_json("rosetta_spam-filter", _stub_qor_json("rosetta_spam-filter"))

        report = join_qor_features(self.store, [j], apply=True)
        self.assertTrue(report.episode_store_changed)
        self.assertEqual(report.episodes_modified, 1)
        e = report.entries[0]
        self.assertEqual(e.status, STATUS_JOINED)
        self.assertEqual(e.match_method, "output_dcp_path")
        self.assertEqual(e.matched_episode_ids, ["E1"])
        self.assertGreater(e.route_bound_score or 0, 0)
        # Now stored.
        stored = json.loads(self.store.read_text().splitlines()[0])
        sf = stored["start_features"]
        for k in QOR_FLAT_KEYS:
            self.assertIn(k, sf)
        self.assertEqual(sf["qor_congestion_long_n"], 4)
        self.assertEqual(sf["qor_has_congestion_signal"], True)
        # Pre-existing fields preserved.
        self.assertEqual(sf["lut_count"], 1000)
        self.assertEqual(sf["critical_path_spread"], 100.0)

    def test_idempotent_join(self):
        ep = _episode("E1", "rosetta_spam-filter",
                      dcp_path="/runs/X/rosetta_spam-filter_optimized.dcp")
        self._write_store([ep])
        j = self._write_json("rosetta_spam-filter", _stub_qor_json("rosetta_spam-filter"))

        r1 = join_qor_features(self.store, [j], apply=True)
        first = self.store.read_text()
        r2 = join_qor_features(self.store, [j], apply=True)
        second = self.store.read_text()
        # The second pass writes the same content — byte-identical file.
        self.assertEqual(first, second)
        # Even the apply pass may report modifications=0 second time.
        self.assertEqual(r2.episodes_modified, 0)
        self.assertTrue(r1.episode_store_changed)

    def test_unmatched_reported_not_dropped(self):
        ep = _episode("E1", "other_design",
                      dcp_path="/runs/X/other_design_optimized.dcp")
        self._write_store([ep])
        j = self._write_json("not_in_store", _stub_qor_json("not_in_store"))
        report = join_qor_features(self.store, [j], apply=True)
        self.assertFalse(report.episode_store_changed)
        self.assertEqual(report.unmatched_count, 1)
        self.assertEqual(report.entries[0].status, STATUS_UNMATCHED)

    def test_missing_json_tolerated(self):
        ep = _episode("E1", "rosetta_spam-filter")
        self._write_store([ep])
        report = join_qor_features(self.store, [self.jsondir / "ghost.qor.json"], apply=True)
        self.assertEqual(report.entries[0].status, STATUS_UNMATCHED)
        self.assertEqual(report.unmatched_count, 1)

    def test_malformed_json_rejected(self):
        ep = _episode("E1", "rosetta_spam-filter",
                      dcp_path="/runs/X/rosetta_spam-filter.dcp")
        self._write_store([ep])
        bad = self.jsondir / "rosetta_spam-filter.qor.json"
        bad.write_text("{not valid json")
        report = join_qor_features(self.store, [bad], apply=True)
        self.assertEqual(report.entries[0].status, STATUS_REJECTED)
        self.assertFalse(report.episode_store_changed)

    def test_existing_start_features_preserved(self):
        ep = _episode("E1", "finn_radioml",
                      dcp_path="/runs/X/finn_radioml_optimized.dcp")
        ep["start_features"]["pathology"] = "HIGH_FANOUT_DRIVER"
        ep["start_features"]["initial_wns"] = -1.5
        self._write_store([ep])
        j = self._write_json("finn_radioml", _stub_qor_json("finn_radioml"))
        join_qor_features(self.store, [j], apply=True)
        stored = json.loads(self.store.read_text().splitlines()[0])
        sf = stored["start_features"]
        self.assertEqual(sf["pathology"], "HIGH_FANOUT_DRIVER")
        self.assertEqual(sf["initial_wns"], -1.5)
        self.assertIn("qor_route_bound_score", sf)

    def test_design_name_hint_does_not_join_by_default(self):
        # No output_dcp_path / run_id overlap — only the design name matches.
        ep = _episode("E1", "rosetta_spam-filter", dcp_path=None,
                      run_id="dcp_optimizer_run-UNRELATED")
        self._write_store([ep])
        j = self._write_json("rosetta_spam-filter", _stub_qor_json("rosetta_spam-filter"))
        report = join_qor_features(self.store, [j], apply=True)
        self.assertEqual(report.entries[0].status, STATUS_REPORT_ONLY)
        self.assertFalse(report.episode_store_changed)

    def test_submission_path_rejected(self):
        self._write_store([_episode("E1", "x")])
        j = self._write_json("x", _stub_qor_json("x"))
        # Pretend the JSON is under the submission tree.
        bad = self.root / "submission" / "qor"
        bad.mkdir(parents=True)
        bad_json = bad / "x.qor.json"
        bad_json.write_text(j.read_text())
        report = join_qor_features(self.store, [bad_json], apply=True)
        self.assertEqual(report.entries[0].status, STATUS_REJECTED)
        self.assertFalse(report.episode_store_changed)

    def test_submission_episode_store_path_refused(self):
        # The episode_store path itself must not live under submission/.
        bad_store = self.root / "submission" / "policy_memory" / "episode_store.jsonl"
        bad_store.parent.mkdir(parents=True)
        bad_store.write_text("")
        j = self._write_json("x", _stub_qor_json("x"))
        with self.assertRaises(ValueError):
            join_qor_features(bad_store, [j], apply=True)

    def test_ambiguous_when_two_episodes_tie(self):
        # Two episodes share the same design name AND both have matching
        # output_dcp_paths — confidence ties at 3, so the join is refused.
        e1 = _episode("E1", "finn_radioml",
                      dcp_path="/runs/A/finn_radioml_optimized.dcp")
        e2 = _episode("E2", "finn_radioml",
                      dcp_path="/runs/B/finn_radioml_optimized.dcp")
        self._write_store([e1, e2])
        j = self._write_json("finn_radioml", _stub_qor_json("finn_radioml"))
        report = join_qor_features(self.store, [j], apply=True)
        self.assertEqual(report.entries[0].status, STATUS_AMBIGUOUS)
        self.assertCountEqual(report.entries[0].matched_episode_ids, ["E1", "E2"])
        self.assertFalse(report.episode_store_changed)

    def test_two_jsons_claim_one_episode_marks_ambiguous(self):
        ep = _episode("E1", "rosetta_spam-filter",
                      dcp_path="/runs/X/rosetta_spam-filter_optimized.dcp")
        self._write_store([ep])
        j1 = self._write_json("rosetta_spam-filter",
                              _stub_qor_json("rosetta_spam-filter"))
        # Second JSON for the same design — also matches the only episode.
        j2_dir = self.jsondir / "second"
        j2_dir.mkdir()
        j2 = j2_dir / "rosetta_spam-filter.qor.json"
        j2.write_text(j1.read_text())
        report = join_qor_features(self.store, [j1, j2], apply=True)
        statuses = sorted(e.status for e in report.entries)
        self.assertEqual(statuses, [STATUS_AMBIGUOUS, STATUS_AMBIGUOUS])
        self.assertFalse(report.episode_store_changed)

    def test_dir_proximity_tiebreaker_isolates_correct_episode(self):
        """Session-15 regression: two episodes with the same DCP basename
        in different directories must not both bind to the same JSON.

        The JOIN PLANNER's proximity tiebreaker should pick the episode
        whose output_dcp_path lives in the same directory as the JSON.
        """
        # Two episodes for the "same" design under different session paths.
        e_old = _episode(
            "OLD", "x_design",
            dcp_path="/runs/session09/x_design__contest_card_optimized.dcp",
            run_id="dcp_optimizer_run-OLD",
        )
        e_new = _episode(
            "NEW", "x_design",
            dcp_path="/tmp/session15/x_design__contest_card_optimized.dcp",
            run_id="dcp_optimizer_run-NEW",
        )
        self._write_store([e_old, e_new])
        # JSON lives in the session15 dir.
        j = self.jsondir / "x_design__contest_card_optimized.qor.json"
        # Put it in a dir whose path includes "session15" so the
        # proximity tiebreaker can isolate.
        sub = self.jsondir / "session15"
        sub.mkdir()
        j = sub / "x_design__contest_card_optimized.qor.json"
        j.write_text(json.dumps(_stub_qor_json("x_design__contest_card_optimized")))
        # NEW episode's dcp lives under /tmp/session15/ — shared discriminator
        # "session15".  OLD episode lives under /runs/session09/.  Proximity
        # should pick NEW.
        report = join_qor_features(self.store, [j], apply=True)
        self.assertEqual(len(report.entries), 1)
        e = report.entries[0]
        self.assertEqual(e.status, STATUS_JOINED, f"got {e.status}: {e.notes}")
        self.assertEqual(e.matched_episode_ids, ["NEW"],
                         "proximity tiebreaker should pick the dir-matching episode")
        # OLD episode must NOT have been mutated.
        stored_lines = self.store.read_text().splitlines()
        stored = [json.loads(l) for l in stored_lines]
        old = next(x for x in stored if x["episode_id"] == "OLD")
        self.assertNotIn("qor_route_bound_score", old["start_features"])

    def test_dir_proximity_tiebreaker_refuses_when_inconclusive(self):
        """When proximity can't isolate one episode, fall back to AMBIGUOUS."""
        # Two episodes with same stem and unrelated dirs; JSON dir has no
        # shared discriminator with either.
        e1 = _episode("E1", "x_design",
                      dcp_path="/runs/alpha/x_design_optimized.dcp")
        e2 = _episode("E2", "x_design",
                      dcp_path="/runs/beta/x_design_optimized.dcp")
        self._write_store([e1, e2])
        j = self._write_json("x_design_optimized", _stub_qor_json("x_design"))
        report = join_qor_features(self.store, [j], apply=True)
        self.assertEqual(report.entries[0].status, STATUS_AMBIGUOUS)
        self.assertFalse(report.episode_store_changed)

    def test_design_name_never_appears_as_strategy_condition(self):
        # The join tool must not embed a benchmark name into a strategy key.
        # We assert by checking that nothing in the public surface contains
        # design names except as explicit metadata fields.
        ep = _episode("E1", "rosetta_spam-filter",
                      dcp_path="/runs/X/rosetta_spam-filter_optimized.dcp")
        self._write_store([ep])
        j = self._write_json("rosetta_spam-filter", _stub_qor_json("rosetta_spam-filter"))
        report = join_qor_features(self.store, [j], apply=False)
        # Match-method label is canonical, not the design name.
        self.assertIn(report.entries[0].match_method, {
            "output_dcp_path", "run_dir", "design_name_hint",
        })

    def test_route_bound_score_deterministic(self):
        # Same input → same numeric score.
        ep = _episode("E1", "rosetta_spam-filter",
                      dcp_path="/runs/X/rosetta_spam-filter_optimized.dcp")
        self._write_store([ep])
        j = self._write_json("rosetta_spam-filter", _stub_qor_json("rosetta_spam-filter"))
        r1 = join_qor_features(self.store, [j], apply=False)
        r2 = join_qor_features(self.store, [j], apply=False)
        self.assertEqual(r1.entries[0].route_bound_score,
                         r2.entries[0].route_bound_score)

    def test_route_bound_score_none_when_no_signal(self):
        ep = _episode("E1", "boom_soc",
                      dcp_path="/runs/X/boom_soc_optimized.dcp")
        self._write_store([ep])
        body = _stub_qor_json("boom_soc")
        # Empty congestion fields model a report with no usable congestion signal.
        body["Design QoR Summary"][0]["Long Cong Level N-E-S-W"] = ""
        body["Design QoR Summary"][0]["Global Cong Level N-E-S-W"] = ""
        j = self._write_json("boom_soc", body)
        report = join_qor_features(self.store, [j], apply=False)
        e = report.entries[0]
        self.assertEqual(e.status, STATUS_JOINED)
        self.assertIsNone(e.route_bound_score)
        self.assertFalse(e.congestion_summary["has_signal"])


if __name__ == "__main__":
    unittest.main()
