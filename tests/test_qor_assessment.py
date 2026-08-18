"""Tests for the Vivado report_qor_assessment Phase-1 feature signal.

The post-2026-05-20 RQA integration runs `report_qor_assessment
-exclude_methodology_checks -max_paths 100 -return_string` after the
spread analysis in `perform_initial_analysis`, parses the score and
flow-guidance signals via `parse_qor_assessment_static`, and surfaces a
compact diagnosis block to the LLM.

These tests cover the parser's tolerance to multiple output shapes
Vivado emits, and confirm the diagnosis dict is fully None when RQA was
skipped/unparsed (so Phase 1 can still complete without it).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcp_optimizer import parse_qor_assessment_static


class QoRAssessmentParserTests(unittest.TestCase):
    """Parser must extract score, flow guidance, methodology count, and
    ML-strategy availability from the variety of shapes Vivado emits."""

    def test_empty_input_returns_all_none(self):
        d = parse_qor_assessment_static("")
        self.assertEqual(d, {
            "score": None,
            "flow_guidance": None,
            "methodology_violations": None,
            "ml_strategy_available": None,
        })

    def test_non_string_input_returns_all_none(self):
        # Defensive: callers should pass str but the parser must not raise
        # when a None or list slips through (e.g., dispatcher refusal).
        d = parse_qor_assessment_static(None)  # type: ignore[arg-type]
        self.assertIsNone(d["score"])

    def test_tabular_rqa_score_3(self):
        rqa = (
            "+-----------------+-------+\n"
            "| Description     | Value |\n"
            "+-----------------+-------+\n"
            "| RQA Score       |   3   |\n"
            "| Strategy used   | None  |\n"
            "+-----------------+-------+\n"
        )
        d = parse_qor_assessment_static(rqa)
        self.assertEqual(d["score"], 3)
        # No explicit Flow Guidance line → score-based default kicks in.
        self.assertIn("report_qor_suggestions", d["flow_guidance"])

    def test_free_text_assessment_score_5(self):
        rqa = (
            "Report QoR Assessment\n"
            "=====================\n"
            "Assessment Score: 5 - Design will easily meet timing\n"
            "No methodology violations.\n"
        )
        d = parse_qor_assessment_static(rqa)
        self.assertEqual(d["score"], 5)
        self.assertIn("Run implementation", d["flow_guidance"])

    def test_explicit_flow_guidance_overrides_default(self):
        rqa = (
            "RQA Score: 3\n"
            "Flow Guidance: Use Incremental Compile for final mile\n"
        )
        d = parse_qor_assessment_static(rqa)
        self.assertEqual(d["score"], 3)
        self.assertEqual(d["flow_guidance"],
                          "Use Incremental Compile for final mile")

    def test_methodology_violations_count(self):
        rqa = (
            "RQA Score: 2\n"
            "Methodology Violations: 7\n"
        )
        d = parse_qor_assessment_static(rqa)
        self.assertEqual(d["score"], 2)
        self.assertEqual(d["methodology_violations"], 7)

    def test_ml_strategy_available_true(self):
        rqa = (
            "RQA Score: 4\n"
            "ML Strategies Available for this design.\n"
        )
        d = parse_qor_assessment_static(rqa)
        self.assertEqual(d["score"], 4)
        self.assertIs(d["ml_strategy_available"], True)

    def test_ml_strategy_not_available(self):
        rqa = (
            "RQA Score: 2\n"
            "ML Strategies Not Available on this architecture.\n"
        )
        d = parse_qor_assessment_static(rqa)
        self.assertIs(d["ml_strategy_available"], False)

    def test_score_out_of_range_not_captured(self):
        # Vivado's documented range is 1-5; a stray "Score: 6" or "Score: 0"
        # must NOT be picked up as a real RQA score.
        rqa = (
            "Some other context here Score: 6 unrelated.\n"
            "And: Score: 0 also unrelated.\n"
        )
        d = parse_qor_assessment_static(rqa)
        self.assertIsNone(d["score"])

    def test_real_vivado_output_vexriscv_v2(self):
        """Captured 2026-05-20 from a live `vivado -mode batch` run of
        report_qor_assessment on the vexriscv_re-place_v2 baseline DCP.

        Validates the parser against the EXACT shape Vivado 2025.1 emits.
        Notable shape choices:
          - Score field is named "QoR Assessment Score" (matches our
            "Assessment Score" alternation).
          - Score line contains "2 - <description>" — only the digit
            must be captured.
          - The ML Strategy section ends with a disclaimer "* ML
            Strategies are available only when ..." that the parser
            must NOT treat as a positive availability signal.
          - All 4 ML directive rows are "Not OK" → the parser should
            classify the design as ml_strategy_available=False.
        """
        rqa = (
            "+----------------------+--------------------------------------+\n"
            "| QoR Assessment Score | 2 - Implementation may complete. ... |\n"
            "+----------------------+--------------------------------------+\n"
            "| Flow Guidance        | To see critical timing paths examine |\n"
            "|                      | the CSV file containing timing paths.|\n"
            "+----------------------+--------------------------------------+\n"
            "\n"
            "3. ML Strategy Availability\n"
            "---------------------------\n"
            "\n"
            "+-----------------------------------------+---------------------+--------+\n"
            "| Conditions for ML Strategy Availability | Value               | Status |\n"
            "+-----------------------------------------+---------------------+--------+\n"
            "| opt_design directive                    |    ExploreWithRemap | Not OK |\n"
            "| place_design directive                  | AltSpreadLogic_high | Not OK |\n"
            "| phys_opt_design directive               |             Default | Not OK |\n"
            "| route_design directive                  |             Default | Not OK |\n"
            "+-----------------------------------------+---------------------+--------+\n"
            "* ML Strategies are available only when the default/explore "
            "directives have been used in the implementation flow and the "
            "design is successfully routed design. Refer to UG906 for more "
            "details.\n"
        )
        d = parse_qor_assessment_static(rqa)
        self.assertEqual(d["score"], 2)
        # Vivado's "To see critical timing paths examine the CSV file..."
        # is the canned generic line — the P7 audit (2026-05-20) showed
        # it's not actionable for the LLM (we don't expose the CSV).
        # Parser must drop it and fall back to the score-2 default.
        self.assertNotIn("critical timing paths", (d["flow_guidance"] or "").lower())
        self.assertEqual(d["flow_guidance"], "Review constraints / review RTL")
        # All 4 directive rows are "Not OK" → unavailable.
        self.assertIs(d["ml_strategy_available"], False)

    def test_canned_csv_flow_guidance_falls_back_to_score_default(self):
        """Generic "To see critical timing paths examine the CSV file..."
        is the Vivado canned line for designs without a real flow
        recommendation.  Must fall back to score-mapped default."""
        rqa = (
            "| QoR Assessment Score | 4 - design should close with directives |\n"
            "| Flow Guidance        | To see critical timing paths examine    |\n"
            "|                      | the CSV file containing timing paths.   |\n"
        )
        d = parse_qor_assessment_static(rqa)
        self.assertEqual(d["score"], 4)
        # Score-4 default per the public KB ladder.
        self.assertIn("report_qor_suggestions", d["flow_guidance"])
        # Canned string must not leak through.
        self.assertNotIn("CSV file", d["flow_guidance"])

    def test_real_actionable_flow_guidance_preserved(self):
        """When Vivado emits a real actionable Flow Guidance line, the
        parser must NOT replace it with the score default."""
        rqa = (
            "| QoR Assessment Score | 3 |\n"
            "| Flow Guidance        | Use ML Strategies and incremental compile |\n"
        )
        d = parse_qor_assessment_static(rqa)
        self.assertEqual(d["score"], 3)
        # Real Vivado guidance survives.
        self.assertEqual(d["flow_guidance"],
                          "Use ML Strategies and incremental compile")

    def test_ml_disclaimer_does_not_false_positive(self):
        """The bare disclaimer line, with no actual availability table,
        must leave ml_strategy_available=None (insufficient info)."""
        rqa = (
            "QoR Assessment Score: 3\n"
            "* ML Strategies are available only when the default/explore "
            "directives have been used. Refer to UG906 for more details.\n"
        )
        d = parse_qor_assessment_static(rqa)
        self.assertEqual(d["score"], 3)
        # No table → no positive/negative inference.
        self.assertIsNone(d["ml_strategy_available"])


if __name__ == "__main__":
    unittest.main()
