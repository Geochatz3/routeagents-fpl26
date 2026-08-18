"""Tests for RAG contest-mode — session 4 P5.

Pins the hard-rule hygiene contract:
  - contest_mode=True must NOT inject DESIGN_NOTES
  - contest_mode=True must NOT use exact-name retrieval as primary
  - feature-first retrieval (LUT count + spread) is the primary path
  - negative-memory advisory block appears when applicable
  - retrieval metadata is exposed for decision-tracer consumption

These tests work directly against optimizer.strategy_memory and a
mocked DCPOptimizer (no MCP / no Vivado).
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizer.strategy_memory import (
    DESIGN_NOTES,
    RunRecord,
    design_note_for,
    fingerprint_match,
    negative_memory_block,
    retrieval_metadata_for,
    seed_prompt_for,
)


def _fake_memory():
    """A small synthetic memory covering: exact-match for known
    benchmark + a feature-similar untested record + a regressed
    record."""
    return [
        RunRecord(
            design="corescore_500_mod", candidate="v0_3",
            delta_fmax_mhz=+53.12, lut_count=24_000,
            critical_path_spread=42.0, completed=True,
        ),
        RunRecord(
            design="finn_radioml", candidate="v0_3_seed3",
            delta_fmax_mhz=+54.08, lut_count=88_000,
            critical_path_spread=18.0, completed=True,
        ),
        RunRecord(
            design="some_old_design", candidate="anchor",
            delta_fmax_mhz=-2.31, lut_count=23_500,
            critical_path_spread=41.0, completed=True,
            note="regressed on retry",
        ),
        RunRecord(
            design="dead_branch", candidate="exploratory",
            delta_fmax_mhz=None, lut_count=25_000,
            critical_path_spread=43.0, completed=False,
        ),
    ]


class ContestModeRetrievalTests(unittest.TestCase):
    """Pin contest-mode primary-path behavior."""

    def setUp(self):
        # Patch load_memory globally for all tests in this class.
        self._patch = mock.patch(
            "optimizer.strategy_memory.load_memory",
            return_value=_fake_memory(),
        )
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def test_contest_mode_skips_design_notes(self):
        """In contest mode the curated DESIGN_NOTES block must not
        appear in the seed prompt, even for known benchmarks."""
        # corescore_500_mod IS a key in DESIGN_NOTES — sanity check.
        self.assertIn("corescore_500_mod", DESIGN_NOTES)
        # Normal mode: the note IS injected (design recognized).
        prompt_normal = seed_prompt_for(
            "corescore_500_mod", lut_count=24_000,
            critical_path_spread=42.0,
            contest_mode=False,
        )
        self.assertIn("DESIGN-SPECIFIC NOTE", prompt_normal)
        # Contest mode: the note is NOT injected.
        prompt_contest = seed_prompt_for(
            "corescore_500_mod", lut_count=24_000,
            critical_path_spread=42.0,
            contest_mode=True,
        )
        self.assertNotIn("DESIGN-SPECIFIC NOTE", prompt_contest)

    def test_contest_mode_does_not_use_exact_name_primary(self):
        """In contest mode the prompt must not echo the contest
        design's own name as a retrieval key."""
        prompt_contest = seed_prompt_for(
            "corescore_500_mod", lut_count=24_000,
            critical_path_spread=42.0,
            contest_mode=True,
        )
        # The original design name passed in must NOT appear in
        # the returned prompt under contest mode.
        self.assertNotIn("corescore_500_mod", prompt_contest)

    def test_contest_mode_feature_first_works_with_fingerprint(self):
        """Even when design_name is None (truly hidden), the
        feature-first path must return useful prior content."""
        prompt = seed_prompt_for(
            design_name=None, lut_count=88_000,
            critical_path_spread=18.0,
            contest_mode=True,
        )
        # Should return the closest fingerprint match's snippet
        self.assertIn("PRIOR-CAMPAIGN HISTORY", prompt)

    def test_contest_mode_empty_when_no_memory(self):
        """When memory is empty, contest mode returns empty string
        (no fallback to design_note_for or other leaks)."""
        with mock.patch(
            "optimizer.strategy_memory.load_memory",
            return_value=[],
        ):
            prompt = seed_prompt_for(
                "corescore_500_mod", lut_count=24_000,
                critical_path_spread=42.0,
                contest_mode=True,
            )
            self.assertEqual(prompt, "")

    def test_normal_mode_unchanged(self):
        """Non-contest-mode path must remain bit-equivalent to the
        prior session's behavior (DESIGN_NOTES injected when known)."""
        prompt = seed_prompt_for(
            "corescore_500_mod", lut_count=24_000,
            critical_path_spread=42.0,
            contest_mode=False,
        )
        # Both DESIGN_NOTES and exact-name retrieval should fire.
        self.assertIn("DESIGN-SPECIFIC NOTE", prompt)


class NegativeMemoryBlockTests(unittest.TestCase):

    def setUp(self):
        self._patch = mock.patch(
            "optimizer.strategy_memory.load_memory",
            return_value=_fake_memory(),
        )
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def test_negative_memory_block_includes_regressed_record(self):
        block = negative_memory_block(
            lut_count=23_500, critical_path_spread=41.0,
        )
        self.assertIn("NEGATIVE-MEMORY ADVISORY", block)
        # Should include the regressed candidate label.
        self.assertIn("anchor", block)
        # ADVISORY-only wording — no command-control language.
        self.assertNotIn("BLOCK", block)
        self.assertNotIn("MUST NOT", block)

    def test_negative_memory_block_includes_incomplete_record(self):
        block = negative_memory_block(
            lut_count=25_000, critical_path_spread=43.0,
        )
        self.assertIn("NEGATIVE-MEMORY ADVISORY", block)
        # Either the regressed or the dead_branch should show; both
        # qualify as negative.
        self.assertTrue(
            "exploratory" in block or "anchor" in block,
            f"expected exploratory or anchor in block: {block!r}",
        )

    def test_negative_memory_empty_when_all_positive(self):
        with mock.patch(
            "optimizer.strategy_memory.load_memory",
            return_value=[
                RunRecord(design="x", candidate="y",
                          delta_fmax_mhz=+5.0, completed=True),
            ],
        ):
            self.assertEqual(negative_memory_block(), "")


class RetrievalMetadataTests(unittest.TestCase):

    def setUp(self):
        self._patch = mock.patch(
            "optimizer.strategy_memory.load_memory",
            return_value=_fake_memory(),
        )
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def test_metadata_normal_mode_exact_name(self):
        meta = retrieval_metadata_for(
            "corescore_500_mod", lut_count=24_000,
            critical_path_spread=42.0,
            contest_mode=False,
        )
        self.assertEqual(meta["rag_mode"], "normal")
        self.assertEqual(meta["retrieval_mode"], "exact_name")
        self.assertTrue(meta["exact_name_used"])
        self.assertTrue(meta["design_notes_injected"])
        self.assertGreater(len(meta["retrieved_episode_ids"]), 0)

    def test_metadata_contest_mode_feature_first(self):
        meta = retrieval_metadata_for(
            "corescore_500_mod", lut_count=24_000,
            critical_path_spread=42.0,
            contest_mode=True,
        )
        self.assertEqual(meta["rag_mode"], "contest_mode")
        self.assertEqual(meta["retrieval_mode"], "feature_first")
        self.assertFalse(meta["exact_name_used"])
        self.assertFalse(meta["design_notes_injected"])

    def test_metadata_negative_memory_count(self):
        meta = retrieval_metadata_for(
            "corescore_500_mod", lut_count=24_000,
            critical_path_spread=42.0,
            contest_mode=True,
        )
        # Synthetic memory has 2 negative records (regressed + incomplete).
        self.assertGreaterEqual(meta["negative_memory_count"], 2)

    def test_metadata_empty_memory(self):
        with mock.patch(
            "optimizer.strategy_memory.load_memory",
            return_value=[],
        ):
            meta = retrieval_metadata_for(
                "x", lut_count=1, critical_path_spread=1.0,
                contest_mode=True,
            )
            self.assertEqual(meta["retrieval_mode"], "none")
            self.assertEqual(meta["memory_records_considered"], 0)

    def test_episode_ids_are_stable(self):
        """Same record → same episode id across calls."""
        m1 = retrieval_metadata_for(
            "corescore_500_mod", lut_count=24_000,
            critical_path_spread=42.0,
            contest_mode=False,
        )
        m2 = retrieval_metadata_for(
            "corescore_500_mod", lut_count=24_000,
            critical_path_spread=42.0,
            contest_mode=False,
        )
        self.assertEqual(m1["retrieved_episode_ids"], m2["retrieved_episode_ids"])


class OptimizerIntegrationTests(unittest.TestCase):
    """End-to-end: DCPOptimizer with contest_mode=True emits the right
    trace event and does NOT see DESIGN_NOTES in its iter-1 prompt."""

    def test_optimize_contest_mode_emits_rag_retrieval_trace(self):
        # Late import to keep top-level import paths clean.
        import asyncio
        from dcp_optimizer import DCPOptimizer
        from optimizer.decision_tracer import load_decisions

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            baseline = run_dir / "baseline.dcp"
            baseline.write_bytes(b"DCP_FIXTURE")
            output = run_dir / "out.dcp"

            opt = DCPOptimizer(api_key="test", run_dir=run_dir)
            opt.contest_mode = True
            opt._design_name_for_memory = "corescore_500_mod"
            opt.lut_count = 24_000
            opt.critical_path_spread_info = {"avg_distance": 42.0}

            async def fake_phase1(input_dcp):
                opt.initial_wns = -1.0
                opt.clock_period = 2.0
                opt.best_wns = opt.initial_wns
                opt.qor_assessment = {
                    "score": None, "flow_guidance": None,
                    "methodology_violations": None,
                    "ml_strategy_available": None,
                }
                return "phase 1 done"

            async def fake_call_tool(name, args):
                return "ok"

            async def go():
                with mock.patch.object(
                    opt, "perform_initial_analysis",
                    side_effect=fake_phase1,
                ), mock.patch.object(
                    opt, "call_tool", side_effect=fake_call_tool,
                ), mock.patch.object(
                    opt, "_should_skip_for_budget",
                    return_value=(False, ""),
                ), mock.patch(
                    "optimizer.strategy_memory.load_memory",
                    return_value=_fake_memory(),
                ):
                    async def fake_completion():
                        return ("done", True)
                    with mock.patch.object(
                        opt, "get_completion",
                        side_effect=fake_completion,
                    ):
                        opt.mode = "v0_3"
                        await opt.optimize(baseline, output)
            asyncio.run(go())

            records = load_decisions(run_dir / "decisions.jsonl")
            # Find rag_retrieval record(s).
            rag_records = [
                r for r in records
                if r.get("action_label") == "rag_retrieval"
            ]
            self.assertEqual(len(rag_records), 1)
            self.assertEqual(rag_records[0]["rag_mode"], "contest_mode")
            notes = rag_records[0].get("notes", "")
            self.assertIn("retrieval_mode=feature_first", notes)
            self.assertIn("design_notes_injected=False", notes)
            self.assertIn("exact_name_used=False", notes)


if __name__ == "__main__":
    unittest.main()
