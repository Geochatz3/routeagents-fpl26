"""Tests for RAG hidden-design hygiene.

Pins the hygiene contract:
  - retrieval never injects curated per-design notes (none exist)
  - retrieval never uses the design name as a lookup key
  - feature-first retrieval (LUT count + spread) is the primary path
  - negative-memory advisory block appears when applicable
  - retrieval metadata is exposed for decision-tracer consumption

These tests work directly against optimizer.strategy_memory and a
mocked DCPOptimizer (no MCP / no Vivado).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizer.strategy_memory import (
    RunRecord,
    fingerprint_match,
    negative_memory_block,
    retrieval_metadata_for,
    seed_prompt_for,
)


def _fake_memory():
    """Build synthetic memory with exact, similar, and regressed records.

    The records cover an exact known-design match, a feature-similar unseen
    design, and a regression.
    """
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

    def test_no_design_notes_in_any_mode(self):
        """Curated per-design notes were removed from the tree; the
        seed prompt must never contain one, in either mode."""
        for contest_mode in (False, True):
            prompt = seed_prompt_for(
                "corescore_500_mod", lut_count=24_000,
                critical_path_spread=42.0,
                contest_mode=contest_mode,
            )
            self.assertNotIn("DESIGN-SPECIFIC NOTE", prompt)

    def test_never_uses_exact_name_as_key(self):
        """The prompt must not echo the design's own name as a
        retrieval key — in any mode."""
        for contest_mode in (False, True):
            prompt = seed_prompt_for(
                "corescore_500_mod", lut_count=24_000,
                critical_path_spread=42.0,
                contest_mode=contest_mode,
            )
            self.assertNotIn("corescore_500_mod", prompt)

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
        (no fallback content of any kind)."""
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

    def test_normal_mode_is_also_feature_first(self):
        """Non-contest mode uses the same fingerprint-only retrieval
        as contest mode — the two paths return identical content."""
        kwargs = dict(lut_count=24_000, critical_path_spread=42.0)
        prompt_normal = seed_prompt_for(
            "corescore_500_mod", contest_mode=False, **kwargs)
        prompt_contest = seed_prompt_for(
            "corescore_500_mod", contest_mode=True, **kwargs)
        self.assertEqual(prompt_normal, prompt_contest)
        self.assertIn("PRIOR-CAMPAIGN HISTORY", prompt_normal)


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

    def test_metadata_normal_mode_feature_first(self):
        meta = retrieval_metadata_for(
            "corescore_500_mod", lut_count=24_000,
            critical_path_spread=42.0,
            contest_mode=False,
        )
        self.assertEqual(meta["rag_mode"], "normal")
        self.assertEqual(meta["retrieval_mode"], "feature_first")
        self.assertFalse(meta["exact_name_used"])
        self.assertFalse(meta["design_notes_injected"])
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
    trace event with feature-first retrieval metadata."""

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
            # Finalization writes strategy memory during a real optimize call.
            # Redirect it to the temporary directory to avoid modifying the
            # checkout's persistent priors.
            # addCleanup rather than TestCase.enterContext: the latter is
            # Python 3.11+, and README.md promises 3.10. The CI matrix caught
            # this on the day it started testing the stated floor.
            _memory_env = mock.patch.dict(
                os.environ,
                {"STRATEGY_MEMORY_PATH": str(Path(tmp) / "strategy_memory.jsonl")},
            )
            _memory_env.start()
            self.addCleanup(_memory_env.stop)

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
