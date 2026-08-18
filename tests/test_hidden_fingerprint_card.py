"""Tests for the offline Hidden-Design Fingerprint Card v0 prototype.

Pinning the v0 advisory-only contract:
- design-name keys refused,
- BENEFITED neighbour → advisory with a label (not a command),
- saturated CARD_NEUTRAL band → silence,
- route_bound_likely + negative-memory → WARNING ONLY,
- missing QoR → insufficient_data,
- never emits a Vivado command,
- deterministic top-K (tie-break by episode_id).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizer.hidden_fingerprint_card import (
    ADVISORY_FAMILIES,
    Advisory,
    CARD_BENEFITED,
    CARD_HARMFUL,
    CARD_NEUTRAL,
    EpisodeSummary,
    advise,
    feature_band,
)


def _mk(lut, spread, score, has_signal, verdict=CARD_NEUTRAL,
        family=None, eid=None, status="VALID_OPTIMIZED"):
    return EpisodeSummary(
        episode_id=eid or f"e_{int(lut)}_{int(spread)}_{int((score or 0)*100)}",
        lut_count=float(lut),
        critical_path_spread=float(spread),
        qor_route_bound_score=score,
        qor_has_congestion_signal=has_signal,
        card_verdict=verdict,
        final_status=status,
        winning_action_family=family,
    )


class FeatureBandTests(unittest.TestCase):

    def test_buckets_deterministic(self):
        b1 = feature_band({"lut_count": 2128, "critical_path_spread": 11.46,
                           "qor_route_bound_score": 0.29,
                           "qor_has_congestion_signal": True,
                           "initial_wns": -0.946})
        b2 = feature_band({"lut_count": 2128, "critical_path_spread": 11.46,
                           "qor_route_bound_score": 0.29,
                           "qor_has_congestion_signal": True,
                           "initial_wns": -0.946})
        self.assertEqual(b1, b2)
        self.assertEqual(b1, ("small_lut", "low_spread", "route_bound_mild",
                              "wns_mild"))

    def test_boundary_buckets(self):
        b = feature_band({"lut_count": 60_000, "critical_path_spread": 50.0,
                          "qor_route_bound_score": 0.50,
                          "qor_has_congestion_signal": True,
                          "initial_wns": -5.0})
        self.assertEqual(b, ("large_lut", "mid_spread", "route_bound_likely",
                             "wns_severe"))

    def test_missing_qor_routes_to_low_bucket(self):
        b = feature_band({"lut_count": 5000, "critical_path_spread": 30.0,
                          "qor_route_bound_score": None,
                          "qor_has_congestion_signal": None,
                          "initial_wns": None})
        self.assertEqual(b[2], "route_bound_low")
        self.assertEqual(b[3], "wns_mild")


class AdvisoryContractTests(unittest.TestCase):

    def setUp(self):
        # A realistic history with mixed verdicts.
        self.history = [
            # CARD_BENEFITED with known action family.
            _mk(2128, 11.46, 0.14, True, verdict=CARD_BENEFITED,
                family="eager_mirror", eid="B1"),
            # CARD_NEUTRAL near same band.
            _mk(2128, 11.46, 0.29, True, verdict=CARD_NEUTRAL, eid="N1"),
            _mk(2128, 11.46, 0.29, True, verdict=CARD_NEUTRAL, eid="N2"),
            # CARD_NEUTRAL on mid-lut.
            _mk(55_763, 184.00, 0.29, True, verdict=CARD_NEUTRAL, eid="N3"),
            # CARD_HARMFUL synthetic neighbour.
            _mk(2000, 12.0, 0.30, True, verdict=CARD_HARMFUL, eid="H1"),
            # WNS-bound large.
            _mk(241_577, 302.52, 0.0, True, verdict=CARD_NEUTRAL,
                status="VALID_FALLBACK_BASELINE", eid="W1"),
        ]

    def test_refuses_design_name_key(self):
        cand = {"design_name": "rosetta_spam-filter",
                "lut_count": 2128, "critical_path_spread": 11.46,
                "qor_route_bound_score": 0.30,
                "qor_has_congestion_signal": True,
                "initial_wns": -0.9}
        adv = advise(cand, self.history, min_benefited_floor=1)
        self.assertTrue(adv.silence)
        self.assertEqual(adv.silence_reason, "design_name_in_scope")
        self.assertIsNone(adv.advisory_family)

    def test_refuses_benchmark_key(self):
        cand = {"benchmark": "boom_soc",
                "lut_count": 2128, "critical_path_spread": 11.46,
                "qor_route_bound_score": 0.30,
                "qor_has_congestion_signal": True}
        adv = advise(cand, self.history, min_benefited_floor=1)
        self.assertEqual(adv.silence_reason, "design_name_in_scope")

    def test_missing_qor_returns_insufficient_data(self):
        cand = {"lut_count": 2128, "critical_path_spread": 11.46,
                "initial_wns": -0.9}
        # Neither qor_route_bound_score nor qor_has_congestion_signal set.
        adv = advise(cand, self.history, min_benefited_floor=1)
        self.assertTrue(adv.silence)
        self.assertEqual(adv.silence_reason, "insufficient_data")

    def test_missing_baseline_returns_insufficient_data(self):
        cand = {"qor_route_bound_score": 0.5, "qor_has_congestion_signal": True}
        adv = advise(cand, self.history, min_benefited_floor=1)
        self.assertEqual(adv.silence_reason, "insufficient_data")

    def test_close_to_benefited_emits_advisory(self):
        cand = {"lut_count": 2200, "critical_path_spread": 12.0,
                "qor_route_bound_score": 0.15,
                "qor_has_congestion_signal": True,
                "initial_wns": -0.9}
        # K=3 nearest: B1 (BENEFITED), N1/N2 (NEUTRAL).
        adv = advise(cand, self.history, k=3, min_benefited_floor=1)
        self.assertFalse(adv.silence)
        self.assertEqual(adv.advisory_family, "eager_mirror")
        self.assertIn("B1", adv.evidence_episodes)
        # confidence should be "medium" (only one BENEFITED).
        self.assertEqual(adv.confidence, "medium")

    def test_close_to_neutral_only_band_stays_silent(self):
        # Mid-LUT vtr_mcml-ish; only neutrals in that band.
        cand = {"lut_count": 56_000, "critical_path_spread": 185.0,
                "qor_route_bound_score": 0.29,
                "qor_has_congestion_signal": True,
                "initial_wns": -2.0}
        adv = advise(cand, self.history, min_benefited_floor=1)
        self.assertTrue(adv.silence)
        self.assertEqual(adv.silence_reason, "saturated_neutral_band")

    def test_route_bound_likely_with_negative_memory_warns_only(self):
        cand = {"lut_count": 5000, "critical_path_spread": 65.0,
                "qor_route_bound_score": 0.57,
                "qor_has_congestion_signal": True,
                "initial_wns": -1.0}
        adv = advise(cand, self.history, negative_memory_hits=["legacy-N5N7"], min_benefited_floor=1)
        self.assertFalse(adv.silence)
        self.assertEqual(adv.warning, "route_bound_avoid_cell_surgery")
        # NO advisory_family — warn only.
        self.assertIsNone(adv.advisory_family)
        self.assertEqual(adv.negative_memory_hits, ["legacy-N5N7"])

    def test_wns_bound_global_silences_with_diagnostic(self):
        # boom_soc-like: huge LUT, severe WNS, no congestion.
        cand = {"lut_count": 250_000, "critical_path_spread": 310.0,
                "qor_route_bound_score": 0.0,
                "qor_has_congestion_signal": True,
                "initial_wns": -19.0}
        adv = advise(cand, self.history, min_benefited_floor=1)
        self.assertTrue(adv.silence)
        self.assertEqual(adv.silence_reason, "wns_bound_global_no_route_advice")
        self.assertEqual(adv.diagnostic_label, "wns_bound_global")

    def test_harmful_neighbour_silences(self):
        # Candidate close to the synthetic CARD_HARMFUL neighbour H1.
        cand = {"lut_count": 2000, "critical_path_spread": 12.0,
                "qor_route_bound_score": 0.30,
                "qor_has_congestion_signal": True,
                "initial_wns": -0.9}
        adv = advise(cand, self.history, k=2, min_benefited_floor=1)
        self.assertTrue(adv.silence)
        self.assertEqual(adv.silence_reason, "harmful_neighbour_in_topk")

    def test_no_history_silences_no_history(self):
        cand = {"lut_count": 2000, "critical_path_spread": 12.0,
                "qor_route_bound_score": 0.30,
                "qor_has_congestion_signal": True}
        adv = advise(cand, [], min_benefited_floor=0)
        self.assertTrue(adv.silence)
        self.assertEqual(adv.silence_reason, "no_history")

    def test_advisory_family_is_label_not_command(self):
        # The output must never contain a Vivado / Tcl command string.
        cand = {"lut_count": 2200, "critical_path_spread": 12.0,
                "qor_route_bound_score": 0.15,
                "qor_has_congestion_signal": True,
                "initial_wns": -0.9}
        adv = advise(cand, self.history, k=3, min_benefited_floor=1)
        # Pin the value is in the allow-list.
        self.assertIn(adv.advisory_family, ADVISORY_FAMILIES)
        # Pin no forbidden tokens.
        for tok in (" ", "place_design", "route_design", "phys_opt_design",
                    "report_design_analysis", "[", "{", "$",
                    "vivado_run_tcl"):
            self.assertNotIn(tok, adv.advisory_family)

    def test_advisory_family_outside_allowlist_rejected(self):
        # If a benefited neighbour has an unknown family string, the
        # prototype refuses to surface it.
        history = [
            _mk(2128, 11.46, 0.14, True, verdict=CARD_BENEFITED,
                family="vivado_run_tcl place_design",  # forbidden
                eid="BAD"),
        ]
        cand = {"lut_count": 2200, "critical_path_spread": 12.0,
                "qor_route_bound_score": 0.15,
                "qor_has_congestion_signal": True,
                "initial_wns": -0.9}
        adv = advise(cand, history, min_benefited_floor=1)
        self.assertTrue(adv.silence)
        self.assertEqual(adv.silence_reason, "benefited_but_no_action_family")

    def test_deterministic_topk_tiebreak(self):
        # Two episodes with identical features but different IDs:
        # tie-break must be lexicographic on episode_id.
        history = [
            _mk(2000, 12, 0.30, True, verdict=CARD_BENEFITED,
                family="post_route_phys_opt", eid="BZ"),
            _mk(2000, 12, 0.30, True, verdict=CARD_BENEFITED,
                family="eager_mirror", eid="AA"),
        ]
        cand = {"lut_count": 2000, "critical_path_spread": 12.0,
                "qor_route_bound_score": 0.30,
                "qor_has_congestion_signal": True,
                "initial_wns": -1.0}
        adv1 = advise(cand, history, k=2, min_benefited_floor=1)
        adv2 = advise(cand, history, k=2, min_benefited_floor=1)
        self.assertEqual(adv1.as_dict(), adv2.as_dict())
        # AA sorts first → its family wins.
        self.assertEqual(adv1.advisory_family, "eager_mirror")

    def test_no_design_name_in_output_strings(self):
        cand = {"lut_count": 2200, "critical_path_spread": 12.0,
                "qor_route_bound_score": 0.15,
                "qor_has_congestion_signal": True,
                "initial_wns": -0.9}
        adv = advise(cand, self.history, min_benefited_floor=1)
        # advisory_family / warning / silence_reason / diagnostic_label
        # must not embed a benchmark name (history's design_name field
        # doesn't exist on EpisodeSummary; this test is a regression guard).
        bench_tokens = ("vexriscv", "rosetta", "boom_soc", "vtr_mcml",
                        "finn", "corescore", "logicnets", "amd_mini",
                        "ispd16", "spam-filter")
        for k in ("advisory_family", "warning", "silence_reason",
                  "diagnostic_label"):
            v = getattr(adv, k)
            if isinstance(v, str):
                for tok in bench_tokens:
                    self.assertNotIn(tok, v.lower())

    def test_advisory_silence_implies_no_recipe_command(self):
        # A silent advisory must NEVER carry an advisory_family.
        cand = {"lut_count": 250_000, "critical_path_spread": 310.0,
                "qor_route_bound_score": 0.0,
                "qor_has_congestion_signal": True,
                "initial_wns": -19.0}
        adv = advise(cand, self.history, min_benefited_floor=1)
        self.assertTrue(adv.silence)
        self.assertIsNone(adv.advisory_family)

    def test_advisory_family_emitted_is_in_allowlist_only(self):
        # Stress test: every family from the allow-list, when used as
        # the BENEFITED neighbour's winning_action_family, must pass
        # through unchanged.  No family outside the list ever passes.
        for fam in ADVISORY_FAMILIES:
            history = [_mk(2000, 12, 0.20, True, verdict=CARD_BENEFITED,
                            family=fam, eid="OK")]
            cand = {"lut_count": 2000, "critical_path_spread": 12,
                    "qor_route_bound_score": 0.20,
                    "qor_has_congestion_signal": True}
            adv = advise(cand, history, min_benefited_floor=1)
            self.assertEqual(adv.advisory_family, fam)
        for bad in ("post_route_phys_opt; rm -rf /", "drop table episodes",
                    "vivado_run_tcl", "place_design", "phys_opt_design",
                    "report_design_analysis -qor"):
            history = [_mk(2000, 12, 0.20, True, verdict=CARD_BENEFITED,
                            family=bad, eid="BAD")]
            cand = {"lut_count": 2000, "critical_path_spread": 12,
                    "qor_route_bound_score": 0.20,
                    "qor_has_congestion_signal": True}
            adv = advise(cand, history, min_benefited_floor=1)
            self.assertTrue(adv.silence)


class MinBenefitedFloorTests(unittest.TestCase):
    """Session-19 direction correction: HFCv0 must silence when the
    history pool contains too few CARD_BENEFITED examples.

    Default floor is 3 so the corpus today (N=1 BENEFITED) silences
    every advisory — preventing the Session-18 LOO failure mode of
    over-fitting to the single known BENEFITED lineage.
    """

    def setUp(self):
        # Same fixture as the contract tests but expose the BENEFITED count.
        self.history_1_benefited = [
            _mk(2128, 11.46, 0.14, True, verdict=CARD_BENEFITED,
                family="eager_mirror", eid="B1"),
            _mk(2128, 11.46, 0.29, True, verdict=CARD_NEUTRAL, eid="N1"),
            _mk(2128, 11.46, 0.29, True, verdict=CARD_NEUTRAL, eid="N2"),
            _mk(55_763, 184.00, 0.29, True, verdict=CARD_NEUTRAL, eid="N3"),
        ]
        self.history_3_benefited = self.history_1_benefited + [
            _mk(40_000, 80.0, 0.30, True, verdict=CARD_BENEFITED,
                family="post_route_phys_opt", eid="B2"),
            _mk(120_000, 220.0, 0.10, True, verdict=CARD_BENEFITED,
                family="retiming_polish", eid="B3"),
        ]
        self.cand = {"lut_count": 2200, "critical_path_spread": 12.0,
                     "qor_route_bound_score": 0.15,
                     "qor_has_congestion_signal": True,
                     "initial_wns": -0.9}

    def test_default_floor_is_3(self):
        # No explicit min_benefited_floor: default applies.
        adv = advise(self.cand, self.history_1_benefited)
        self.assertTrue(adv.silence)
        self.assertEqual(adv.silence_reason, "insufficient_benefited_floor")

    def test_floor_silences_below_threshold(self):
        adv = advise(self.cand, self.history_1_benefited,
                     min_benefited_floor=2)
        self.assertEqual(adv.silence_reason, "insufficient_benefited_floor")

    def test_floor_above_threshold_emits_advisory(self):
        # 3 BENEFITED in history; default floor of 3 satisfied.
        adv = advise(self.cand, self.history_3_benefited)
        # Candidate is close to B1 (same band) so advisory_family should
        # be eager_mirror (the BENEFITED in candidate's LUT bucket).
        self.assertFalse(adv.silence)
        self.assertEqual(adv.advisory_family, "eager_mirror")

    def test_floor_zero_disables_gate(self):
        # Explicit floor=0 means "no floor"; downstream rules fire.
        adv = advise(self.cand, self.history_1_benefited,
                     min_benefited_floor=0)
        self.assertFalse(adv.silence)
        self.assertEqual(adv.advisory_family, "eager_mirror")

    def test_floor_does_not_override_design_name_refusal(self):
        cand = dict(self.cand)
        cand["design_name"] = "rosetta_spam-filter"
        # Design-name guard fires BEFORE the floor check.  Refusing a
        # benchmark-name-tainted input is non-negotiable.
        adv = advise(cand, self.history_1_benefited, min_benefited_floor=0)
        self.assertEqual(adv.silence_reason, "design_name_in_scope")

    def test_floor_does_not_override_insufficient_data(self):
        # If QoR is missing, insufficient_data fires BEFORE the floor.
        cand = {"lut_count": 2128, "critical_path_spread": 11.46,
                "initial_wns": -0.9}
        adv = advise(cand, self.history_3_benefited, min_benefited_floor=3)
        self.assertEqual(adv.silence_reason, "insufficient_data")

    def test_floor_preserves_negative_memory_hits(self):
        adv = advise(self.cand, self.history_1_benefited,
                     min_benefited_floor=3,
                     negative_memory_hits=["legacy-N5N7"])
        self.assertEqual(adv.silence_reason, "insufficient_benefited_floor")
        self.assertEqual(adv.negative_memory_hits, ["legacy-N5N7"])

    def test_floor_preserves_feature_band(self):
        adv = advise(self.cand, self.history_1_benefited)
        self.assertIsNotNone(adv.feature_band)
        self.assertEqual(adv.feature_band[0], "small_lut")

    def test_floor_silences_label_not_command(self):
        # Floor-induced silence must NEVER carry an advisory_family.
        adv = advise(self.cand, self.history_1_benefited)
        self.assertIsNone(adv.advisory_family)
        self.assertIsNone(adv.warning)
        self.assertIsNone(adv.confidence)


if __name__ == "__main__":
    unittest.main()
