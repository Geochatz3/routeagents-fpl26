"""Test feature-based recipe routing and its guard invariants."""
from __future__ import annotations

import unittest

from optimizer.recipe_router import (
    PhaseOneFeatures,
    RecipeAction,
    RecipePlan,
    decide_recipe_path,
    R1_DEGRADED_MARKER,
    R1_FAILING_ENDPOINTS_MIN,
    R1_ROUTE_FIRST_WNS_NS,
    TIEBREAK_WNS_BAND_NS,
    TIEBREAK_FAILING_BAND,
    R2_DEGRADED_MARKER,
    R2_FAILING_ENDPOINTS_MAX,
    R2_SPREAD_MIN_TILES,
    R3_REMAINING_WALL_MIN_S,
    R3_EXPLORE_SPREAD_MIN_TILES,
    R3_WNS_ABS_MAX_NS,
    R2_WNS_ABS_MIN_NS,
    R5_LARGE_DESIGN_FAILING_ENDPOINTS,
    R6_SCOPE_MULTIPLIER,
    R6_SCOPED_PHYS_OPT_DEFAULT_N,
    R1_FMAX_RATIO_MAX,
    OOB_MARKER,
    OOB_ROUTE_FIRST_WNS_NS,
    R8_HIGH_UTIL_PCT,
    explore_runtime_estimate_s,
    classg_runtime_estimate_s,
    explore_infeasible_reason,
)


# Fixture helpers — construct PhaseOneFeatures for each known design.
# Numbers from the measured-run post-mortem.

def _boom_soc_features(**overrides) -> PhaseOneFeatures:
    """boom_soc fingerprint: huge failing set, extreme WNS, very low fmax ratio."""
    base = dict(
        wns_ns=-19.16,
        # Clock period that yields fmax_ratio ≈ 8%: CP / (CP + 19.16) = 0.08
        # → CP ≈ 1.666 ns. Target fmax ≈ 600 MHz, achievable ≈ 48 MHz.
        clock_period_ns=1.666,
        failing_endpoint_count=217_988,
        critical_path_avg_spread_tiles=302.0,
        remaining_wall_budget_s=45 * 60,  # 45 min — enough for full retiming
    )
    base.update(overrides)
    return PhaseOneFeatures(**base)


def _ispd16_features(**overrides) -> PhaseOneFeatures:
    """Provide a profile with a very large failing set and deep negative slack.

    The profile must route to R1 rather than the medium-failing-set R2 rule.
    """
    base = dict(
        wns_ns=-7.752,
        clock_period_ns=1.538,  # 1.538/(1.538+7.752) → fmax_ratio ≈ 16.6%
        failing_endpoint_count=242_906,
        critical_path_avg_spread_tiles=660.0,
        remaining_wall_budget_s=45 * 60,
    )
    base.update(overrides)
    return PhaseOneFeatures(**base)


def _r2_profile_features(**overrides) -> PhaseOneFeatures:
    """Provide a synthetic R2 profile with high negative slack and broad placement
    spread.

    R2 requires fewer than 100,000 failing endpoints, at least 5 ns of negative
    slack, an Fmax ratio below 25%, and spread of at least 70 tiles.
    """
    base = dict(
        wns_ns=-6.0,
        clock_period_ns=1.5,  # 1.5/(1.5+6.0) → fmax_ratio = 20%
        failing_endpoint_count=50_000,
        critical_path_avg_spread_tiles=90.0,
        remaining_wall_budget_s=45 * 60,
    )
    base.update(overrides)
    return PhaseOneFeatures(**base)


def _finn_features(**overrides) -> PhaseOneFeatures:
    """Provide a retiming-eligible profile with moderate negative slack and broad
    placement spread.
    """
    base = dict(
        wns_ns=-1.91,
        # CP / (CP + 1.91) = 0.46 → CP ≈ 1.628 ns. target ≈ 614 MHz, achievable
        # ≈ 285 MHz.
        clock_period_ns=1.628,
        failing_endpoint_count=46_438,
        critical_path_avg_spread_tiles=255.1,
        remaining_wall_budget_s=30 * 60,  # 30 min — fits Class G
        class_g_attempted=False,
    )
    base.update(overrides)
    return PhaseOneFeatures(**base)


def _corescore_features(**overrides) -> PhaseOneFeatures:
    """Provide a near-timing profile dominated by placement spread.

    The failing-set size blocks cell replacement and leaves the profile
    eligible for R4 exploration.
    """
    base = dict(
        wns_ns=-1.238,
        # CP / (CP + 1.238) = 0.574 → CP ≈ 1.667 ns. Target ≈ 600 MHz,
        # achievable ≈ 344 MHz.
        clock_period_ns=1.667,
        failing_endpoint_count=39_008,
        critical_path_avg_spread_tiles=232.4,
        remaining_wall_budget_s=30 * 60,
    )
    base.update(overrides)
    return PhaseOneFeatures(**base)


# Regression fixtures — the four known designs MUST route correctly.

class KnownDesignRegressionTests(unittest.TestCase):
    """Each known design must trigger its calibrated rule.

    See the internal campaign notes.md §12.6 for the
    fingerprint → expected-rule mapping.
    """

    def test_boom_soc_triggers_r1_global_retiming(self):
        plan = decide_recipe_path(_boom_soc_features())
        self.assertEqual(plan.rule_id, "R1",
            "boom_soc fingerprint must trigger R1 (full-scope retiming)")
        self.assertEqual(plan.confidence, "high")
        self.assertIn("boom_soc", plan.evidence_designs)
        # Deep negative slack at or beyond 12 ns selects route-first recovery:
        # unroute, apply `AggressiveExplore`, then attempt retiming only if
        # the remaining budget permits it.
        self.assertGreater(len(plan.actions), 0)
        names = [a.name for a in plan.actions]
        self.assertEqual(names, ["vivado_run_tcl",
                                 "vivado_route_design",
                                 "vivado_phys_opt_design"])
        self.assertIn("route_design -unroute", plan.actions[0].note)
        self.assertIn("AggressiveExplore", plan.actions[1].note)
        self.assertIn("AlternateFlowWithRetiming", plan.actions[2].note)

    def test_ispd16_triggers_r1_retiming(self):
        # A very large failing-endpoint set selects the high-confidence
        # retiming path rather than the placement-preserving sweep.
        plan = decide_recipe_path(_ispd16_features())
        self.assertEqual(plan.rule_id, "R1",
            "real ispd16 (242k failing) must trigger R1 (huge-failing retiming)")
        self.assertEqual(plan.confidence, "high")
        self.assertIn("ispd16_example2", plan.evidence_designs)
        self.assertEqual(plan.actions[0].name, "vivado_phys_opt_design")
        self.assertIn("AlternateFlowWithRetiming", plan.actions[0].note)

    def test_r1_includes_aggressive_explore_reroute(self):
        # For moderately deep slack with a huge failing set, action order is
        # retiming, unroute, then an `AggressiveExplore` reroute.
        plan = decide_recipe_path(_ispd16_features())
        self.assertEqual(plan.rule_id, "R1")
        names = [a.name for a in plan.actions]
        self.assertEqual(names, ["vivado_phys_opt_design",
                                 "vivado_run_tcl",
                                 "vivado_route_design"])
        self.assertIn("route_design -unroute", plan.actions[1].note)
        self.assertIn("AggressiveExplore", plan.actions[2].note)

    def test_r2_profile_triggers_r2_post_route_sweep(self):
        # A medium failing set with high |WNS| and spread routes to R2's
        # scoped sweep.
        plan = decide_recipe_path(_r2_profile_features())
        self.assertEqual(plan.rule_id, "R2")
        self.assertEqual(plan.actions[0].name, "recipe_post_route_phys_opt_sweep")
        # cell_replacement must be blocked (detour-analysis stall on this profile)
        self.assertIn("recipe_cell_replacement", plan.blocks)

    def test_finn_triggers_r3_class_g(self):
        plan = decide_recipe_path(_finn_features())
        self.assertEqual(plan.rule_id, "R3",
            "finn fingerprint must trigger R3 (Class G re-placement)")
        self.assertEqual(plan.confidence, "medium")
        self.assertIn("finn_radioml", plan.evidence_designs)
        # Class G is a 4-step sequence: unplace → place Explore → route → phys_opt
        names = [a.name for a in plan.actions]
        self.assertEqual(names, [
            "vivado_run_tcl",
            "vivado_place_design",
            "vivado_route_design",
            "vivado_phys_opt_design",
        ])
        self.assertIn("Explore", plan.actions[1].note)

    def test_corescore_triggers_r4_safety_path(self):
        plan = decide_recipe_path(_corescore_features())
        self.assertEqual(plan.rule_id, "R4",
            "corescore fingerprint must trigger R4 (safety path)")
        self.assertEqual(plan.confidence, "high")
        self.assertIn("corescore_500_mod", plan.evidence_designs)
        # High-ceiling profiles preserve this optimization order:
        # retime → unplace → Explore → retime → pin optimization.
        names = [a.name for a in plan.actions]
        self.assertEqual(names, [
            "recipe_register_retiming",
            "vivado_run_tcl",
            "vivado_place_design",
            "recipe_register_retiming",
            "vivado_phys_opt_design",
        ])
        self.assertIn("Explore", plan.actions[2].note)


class RulePartitioningTests(unittest.TestCase):
    """Verify that each feature profile routes to its expected rule.

    R1 accepts multiple very-large-failing-set profiles, while a synthetic
    medium-set profile provides R2 coverage.
    """

    def test_known_designs_route_to_expected_rules(self):
        results = {
            "boom_soc":   decide_recipe_path(_boom_soc_features()).rule_id,
            "ispd16":     decide_recipe_path(_ispd16_features()).rule_id,
            "r2_profile": decide_recipe_path(_r2_profile_features()).rule_id,
            "finn":       decide_recipe_path(_finn_features()).rule_id,
            "corescore":  decide_recipe_path(_corescore_features()).rule_id,
        }
        self.assertEqual(
            results,
            {"boom_soc": "R1", "ispd16": "R1", "r2_profile": "R2",
             "finn": "R3", "corescore": "R4"},
            f"designs must route to expected rules; got {results}",
        )
        # All four positive rules are still reachable.
        self.assertEqual(sorted(set(results.values())), ["R1", "R2", "R3", "R4"])


# Rule boundary tests — each rule must have a sharp, testable boundary.

class R1BoundaryTests(unittest.TestCase):
    def test_just_below_failing_endpoints_threshold_does_not_fire(self):
        # Deep negative slack remains route-first when the failing count is
        # just below the 100k floor because the count-scale tie-break applies.
        f = _boom_soc_features(failing_endpoint_count=R1_FAILING_ENDPOINTS_MIN - 1)
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "R1")
        self.assertIn("[router-tiebreak]", plan.rationale)
        # BELOW the band the floor is sharp as before: no R1.
        f2 = _boom_soc_features(
            failing_endpoint_count=R1_FAILING_ENDPOINTS_MIN - TIEBREAK_FAILING_BAND - 1)
        plan2 = decide_recipe_path(f2)
        self.assertNotEqual(plan2.rule_id, "R1")

    def test_at_failing_endpoints_threshold_fires(self):
        f = _boom_soc_features(failing_endpoint_count=R1_FAILING_ENDPOINTS_MIN)
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "R1")

    def test_low_wns_blocks_r1(self):
        # Even with a large failing set, |WNS| < 7 ns is below R1's bound.
        f = _boom_soc_features(wns_ns=-5.0)
        plan = decide_recipe_path(f)
        self.assertNotEqual(plan.rule_id, "R1")

    def test_ispd16_just_inside_broadened_r1_bounds(self):
        # A profile at |WNS| 7.75 ns and fmax ratio 16.6% clears R1's bounds.
        self.assertEqual(decide_recipe_path(_ispd16_features()).rule_id, "R1")
        # But a design with the same huge failing set yet |WNS| 6 (< 7) must NOT.
        f = _ispd16_features(wns_ns=-6.0)
        self.assertNotEqual(decide_recipe_path(f).rule_id, "R1")

    def test_boom_v2_band_takes_route_first(self):
        # A very large design at |WNS| 11.392 ns with about 220k failing
        # endpoints lies beyond the 10 ns route-first split.
        f = _boom_soc_features(
            wns_ns=-11.392,
            clock_period_ns=1.569,   # ratio = 1.569/(1.569+11.392) ~ 12%
            failing_endpoint_count=220_131,
            critical_path_avg_spread_tiles=249.7,
        )
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "R1")
        self.assertEqual([a.name for a in plan.actions],
                         ["vivado_run_tcl",
                          "vivado_route_design",
                          "vivado_phys_opt_design"])
        self.assertIn("route_design -unroute", plan.actions[0].note)

    def test_route_first_boundary_edges_at_10(self):
        # Exactly at the 10.0 threshold -> route-first (>= comparison).
        at = decide_recipe_path(_boom_soc_features(wns_ns=-10.0))
        self.assertEqual(at.actions[0].name, "vivado_run_tcl")
        # Values just below the split remain inside the 1.5 ns tie-break band.
        # Ambiguous in-band profiles resolve to route-first.
        below = decide_recipe_path(_boom_soc_features(wns_ns=-9.9))
        self.assertEqual(below.actions[0].name, "vivado_run_tcl")
        self.assertIn("[router-tiebreak]", below.rationale)
        # Just below the split band, the retime-first path remains selected
        # without the router tiebreak.
        below_band = decide_recipe_path(_boom_soc_features(wns_ns=-8.4))
        self.assertEqual(below_band.actions[0].name, "vivado_phys_opt_design")
        self.assertNotIn("[router-tiebreak]", below_band.rationale)


class R2BoundaryTests(unittest.TestCase):
    def test_low_spread_blocks_r2(self):
        # Without high spread, R2's "shared driver" hypothesis doesn't hold.
        f = _r2_profile_features(critical_path_avg_spread_tiles=R2_SPREAD_MIN_TILES - 1)
        plan = decide_recipe_path(f)
        self.assertNotEqual(plan.rule_id, "R2")

    def test_missing_spread_routes_r2_only_via_degraded_path(self):
        # A missing spread metric degrades to the placement-preserving sweep
        # instead of falling back to an unconstrained plan. The rationale
        # must mark this path as degraded for downstream auditing.
        f = _r2_profile_features(critical_path_avg_spread_tiles=None)
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "R2")
        self.assertIn(R2_DEGRADED_MARKER, plan.rationale)


class R3BoundaryTests(unittest.TestCase):
    def test_class_g_already_attempted_blocks_r3(self):
        # Re-roll guard — Vivado place_design Explore is nondeterministic.
        f = _finn_features(class_g_attempted=True)
        plan = decide_recipe_path(f)
        self.assertNotEqual(plan.rule_id, "R3",
            "R3 must not fire when Class G has already been attempted this run")

    def test_insufficient_wall_budget_blocks_r3(self):
        f = _finn_features(remaining_wall_budget_s=R3_REMAINING_WALL_MIN_S - 1)
        plan = decide_recipe_path(f)
        self.assertNotEqual(plan.rule_id, "R3")

    def test_timing_met_design_routes_to_r7_not_destructive(self):
        # Positive WNS has zero failing-depth magnitude and must select the
        # timing-met path, which protects placement and uses granular phys_opt.
        from optimizer.recipe_router import PhaseOneFeatures
        met = PhaseOneFeatures(wns_ns=2.0, clock_period_ns=4.0,
                               failing_endpoint_count=0,
                               critical_path_avg_spread_tiles=50.0,
                               remaining_wall_budget_s=3400.0)
        self.assertEqual(met.wns_magnitude_ns, 0.0)
        self.assertAlmostEqual(met.achievable_fmax_mhz, 500.0)  # 1000/(4-2)
        self.assertEqual(decide_recipe_path(met).rule_id, "R7")

    def test_failing_design_magnitude_and_fmax_unchanged(self):
        # All 13 real benchmarks enter failing (wns < 0): the sign-aware
        # forms must equal the old abs() forms there — zero real-vector
        # behavior change.
        from optimizer.recipe_router import PhaseOneFeatures
        f = PhaseOneFeatures(wns_ns=-2.5, clock_period_ns=3.0)
        self.assertEqual(f.wns_magnitude_ns, 2.5)
        self.assertAlmostEqual(f.achievable_fmax_mhz, 1000.0 / 5.5)

    def test_missing_wall_budget_fires_degraded_r3(self):
        # An unknown wall budget does not disqualify an otherwise valid path.
        # The plan is marked degraded, while the execution-time route gate
        # still protects heavy operations.
        from optimizer.recipe_router import R3_DEGRADED_MARKER
        f = _finn_features(remaining_wall_budget_s=None)
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "R3")
        self.assertIn(R3_DEGRADED_MARKER, plan.rationale)

    def test_r3_high_spread_uses_explore(self):
        # A placement-limited profile with 60-tile spread selects Explore.
        plan = decide_recipe_path(_finn_features(critical_path_avg_spread_tiles=60.0))
        self.assertEqual(plan.rule_id, "R3")
        self.assertIn("directive=Explore", plan.actions[1].note)
        self.assertNotIn("directive=Auto_1", plan.actions[1].note)

    def test_r3_low_spread_uses_auto1(self):
        # An R3 profile below the spread floor is treated as placement-satisfactory
        # and falls back to Auto_1 instead of selecting Explore.
        plan = decide_recipe_path(
            _finn_features(critical_path_avg_spread_tiles=R3_EXPLORE_SPREAD_MIN_TILES - 1))
        self.assertEqual(plan.rule_id, "R3")
        self.assertIn("directive=Auto_1", plan.actions[1].note)
        self.assertNotIn("directive=Explore", plan.actions[1].note)

    def test_r3_unknown_spread_keeps_explore(self):
        # Spread analysis is optional. When spread is unknown, R3 retains
        # its Explore default; R4 instead defaults to Auto_1.
        plan = decide_recipe_path(_finn_features(critical_path_avg_spread_tiles=None))
        self.assertEqual(plan.rule_id, "R3")
        self.assertIn("directive=Explore", plan.actions[1].note)

    def test_r3_floor_sits_in_the_real_explore_boundary_gap(self):
        # The spread threshold must remain strictly between cases where
        # Auto_1 is preferred and cases where Explore is preferred.
        LOWEST_EXPLORE_WINNER = 53.5   # vexriscv_re-place (+113)
        HIGHEST_AUTO1_DESIGN = 15.2    # rosetta_optical-flow (Explore hurt it)
        self.assertGreater(R3_EXPLORE_SPREAD_MIN_TILES, HIGHEST_AUTO1_DESIGN,
            "floor must be above the highest Explore-hurts design (optical 15.2)")
        self.assertLess(R3_EXPLORE_SPREAD_MIN_TILES, LOWEST_EXPLORE_WINNER,
            "floor must be below the lowest Explore-winner (vexriscv 53.5)")


class R3GapFillTests(unittest.TestCase):
    """R3 WNS band widened 3.0 → 5.0 (2026-06-04) to close the FALLBACK gap
    at |WNS| 3–5 ns. A moderate-headroom, placement-limited hidden design
    in that band must now route to R3's Class-G Explore lever instead of
    the generic FALLBACK recipe. Generalization-only (no validated design
    sits in (3,5)); these tests pin the intended routing.
    """

    def test_wns_in_gap_with_r3_profile_routes_to_r3(self):
        # |WNS| 4.0 (was FALLBACK pre-2026-06-04), fmax in [0.35,0.55),
        # placement-limited spread, ample budget → R3 Class-G Explore.
        f = PhaseOneFeatures(
            wns_ns=-4.0,
            clock_period_ns=3.0,           # fmax 3/(3+4) = 43% → in R3 band
            failing_endpoint_count=20_000,  # moderate (< R1's 100k)
            critical_path_avg_spread_tiles=120.0,  # ≥ floor → Explore
            remaining_wall_budget_s=30 * 60,
        )
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "R3",
            "a |WNS| 3-5 moderate-headroom design must route to R3, not FALLBACK")
        self.assertIn("directive=Explore", plan.actions[1].note)

    def test_new_upper_bound_boundary(self):
        # At the new max (5.0) R3 still applies; just past it, R3's band no
        # longer matches (R2's |WNS|≥5 owns that side, tried first).
        base = dict(
            clock_period_ns=4.0,            # fmax ~44% at |WNS|≈5 → R3 band
            failing_endpoint_count=20_000,
            critical_path_avg_spread_tiles=120.0,
            remaining_wall_budget_s=30 * 60,
        )
        at_max = decide_recipe_path(PhaseOneFeatures(
            wns_ns=-R3_WNS_ABS_MAX_NS, **base))
        self.assertEqual(at_max.rule_id, "R3", "|WNS| == 5.0 must still hit R3")
        just_over = decide_recipe_path(PhaseOneFeatures(
            wns_ns=-(R3_WNS_ABS_MAX_NS + 0.5), **base))
        self.assertNotEqual(just_over.rule_id, "R3",
            "|WNS| > 5.0 leaves R3's band")

    def test_gap_design_far_from_target_not_forced_into_r3(self):
        # The fmax gate remains [0.35, 0.55). Designs below this range
        # require a deeper-slack path rather than R3's moderate-headroom
        # recipe.
        f = PhaseOneFeatures(
            wns_ns=-4.0,
            clock_period_ns=1.0,           # fmax 1/(1+4) = 20% < 0.35
            failing_endpoint_count=20_000,
            critical_path_avg_spread_tiles=50.0,  # < R2's 70 → not R2 either
            remaining_wall_budget_s=30 * 60,
        )
        plan = decide_recipe_path(f)
        self.assertNotEqual(plan.rule_id, "R3")
        # no longer an action-less FALLBACK — the OOB net
        # gives it the never-worse sweep floor instead.
        self.assertEqual(plan.rule_id, "OOB")
        self.assertIn("place_design -unplace", plan.blocks)

    def test_r2_still_wins_above_5ns(self):
        # Bands meet at 5.0 with R2 first in priority order: a |WNS|≥5 design
        # meeting R2's gates must route to R2, not the widened R3.
        f = PhaseOneFeatures(
            wns_ns=-6.0,
            clock_period_ns=1.5,           # fmax 20% < R2's 0.25
            failing_endpoint_count=50_000,  # < 100k
            critical_path_avg_spread_tiles=90.0,  # ≥ 70
            remaining_wall_budget_s=30 * 60,
        )
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "R2")
        self.assertGreaterEqual(f.wns_magnitude_ns, R2_WNS_ABS_MIN_NS)


class R4BoundaryTests(unittest.TestCase):
    def test_high_wns_blocks_r4(self):
        # R4 excludes profiles with |WNS| > 2 ns.
        f = _corescore_features(wns_ns=-2.5)
        plan = decide_recipe_path(f)
        self.assertNotEqual(plan.rule_id, "R4")

    def test_low_fmax_ratio_blocks_r4(self):
        # R4 excludes fmax ratios below 55%; |WNS| is raised here to exclude
        # R3 as well.
        f = _corescore_features(wns_ns=-2.5, clock_period_ns=2.0)  # ratio ≈ 44%, |WNS|>2 → not R3/R4
        plan = decide_recipe_path(f)
        self.assertNotEqual(plan.rule_id, "R4")

    def test_r4_high_spread_uses_explore(self):
        # A placement-limited profile with 232-tile spread selects Explore.
        plan = decide_recipe_path(_corescore_features(critical_path_avg_spread_tiles=232.0))
        self.assertEqual(plan.rule_id, "R4")
        self.assertIn("directive=Explore", plan.actions[2].note)
        self.assertNotIn("directive=Auto_1", plan.actions[2].note)

    def test_r4_low_spread_uses_auto1(self):
        # An R4 profile with 15-tile spread remains on Auto_1.
        plan = decide_recipe_path(_corescore_features(critical_path_avg_spread_tiles=15.0))
        self.assertEqual(plan.rule_id, "R4")
        self.assertIn("directive=Auto_1", plan.actions[2].note)
        self.assertNotIn("directive=Explore", plan.actions[2].note)

    def test_r4_unknown_spread_falls_back_to_auto1(self):
        # No spread signal → safe Auto_1 (don't gamble on Explore blind).
        plan = decide_recipe_path(_corescore_features(critical_path_avg_spread_tiles=None))
        self.assertEqual(plan.rule_id, "R4")
        self.assertIn("directive=Auto_1", plan.actions[2].note)


# Block-rule (R5, R6) tests — fire independently of positive rule.

class BlockRuleTests(unittest.TestCase):
    def test_r5_blocks_cell_replacement_when_large_and_budget_tight(self):
        # Mid-range design (no positive rule fires) + large + constrained
        # budget → R5 should still add the block.
        f = PhaseOneFeatures(
            # The out-of-band fixture uses a deep-slack route-first base
            # that does not already block cell replacement. This isolates
            # the block added by R5.
            wns_ns=-12.0, clock_period_ns=1.6,
            failing_endpoint_count=R5_LARGE_DESIGN_FAILING_ENDPOINTS,
            critical_path_avg_spread_tiles=50.0,
            remaining_wall_budget_s=30 * 60,  # 30 min < 60 min threshold
        )
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "OOB")  # premise: no positive rule
        self.assertIn("recipe_cell_replacement", plan.blocks)

    def test_r5_does_not_block_when_budget_is_plenty(self):
        # Same design but plenty of budget → R5 should NOT fire.
        f = PhaseOneFeatures(
            wns_ns=-12.0, clock_period_ns=1.6,  # see above: route-first-shaped OOB base
            failing_endpoint_count=R5_LARGE_DESIGN_FAILING_ENDPOINTS,
            critical_path_avg_spread_tiles=50.0,
            remaining_wall_budget_s=2 * 60 * 60,  # 2 hours
        )
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "OOB")  # premise: no positive rule
        self.assertNotIn("recipe_cell_replacement", plan.blocks)

    def test_r6_blocks_scoped_phys_opt_on_huge_failing_set(self):
        # An extreme-WNS case with a huge failing set blocks scoped phys_opt
        # in both R1 and R6; this also verifies block deduplication.
        plan = decide_recipe_path(_boom_soc_features())
        # Exactly one entry — no duplicate even though R1+R6 both add it.
        blocks = list(plan.blocks)
        self.assertEqual(
            blocks.count("recipe_critical_path_focused_phys_opt"), 1,
            f"scoped phys_opt should appear exactly once, got: {blocks}")

    def test_r6_threshold(self):
        threshold = R6_SCOPE_MULTIPLIER * R6_SCOPED_PHYS_OPT_DEFAULT_N
        # Above threshold → block.
        f = PhaseOneFeatures(
            wns_ns=-3.0, clock_period_ns=5.0,
            failing_endpoint_count=threshold + 1,
        )
        plan = decide_recipe_path(f)
        self.assertIn("recipe_critical_path_focused_phys_opt", plan.blocks)
        # At threshold → no block (strict inequality).
        f2 = PhaseOneFeatures(
            wns_ns=-3.0, clock_period_ns=5.0,
            failing_endpoint_count=threshold,
        )
        plan2 = decide_recipe_path(f2)
        self.assertNotIn("recipe_critical_path_focused_phys_opt", plan2.blocks)


# Graceful-degradation tests — missing features must not crash.

class GracefulDegradationTests(unittest.TestCase):
    def test_empty_features_returns_fallback(self):
        plan = decide_recipe_path(PhaseOneFeatures())
        self.assertEqual(plan.rule_id, "FALLBACK")
        self.assertEqual(plan.confidence, "low")
        self.assertEqual(plan.actions, ())

    def test_missing_clock_period_means_no_positive_rule(self):
        f = PhaseOneFeatures(wns_ns=-19.0, failing_endpoint_count=200_000)
        plan = decide_recipe_path(f)
        # No clock period → no fmax_ratio → no rule can fire on ratio guards.
        self.assertEqual(plan.rule_id, "OOB")
        # ratio not confirmed far-from-target => must be the never-worse
        # sweep, NOT the destructive route-first re-route.
        self.assertNotIn("vivado_route_design",
                         [a.name for a in plan.actions])

    def test_partial_features_route_to_safe_floor_not_crash(self):
        f = PhaseOneFeatures(wns_ns=-2.0)
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "OOB")
        self.assertTrue(plan.actions, "OOB floor must carry actions")

    def test_target_fmax_property_handles_zero_clock_period(self):
        f = PhaseOneFeatures(clock_period_ns=0.0)
        self.assertIsNone(f.target_fmax_mhz)

    def test_achievable_fmax_ratio_handles_missing_inputs(self):
        self.assertIsNone(PhaseOneFeatures().achievable_fmax_ratio)
        self.assertIsNone(PhaseOneFeatures(wns_ns=-1.0).achievable_fmax_ratio)


# Plan formatting — the rendered prompt text should be parseable + auditable.

class FormatForPromptTests(unittest.TestCase):
    def test_known_plan_renders_expected_sections(self):
        plan = decide_recipe_path(_corescore_features())
        text = plan.format_for_prompt()
        self.assertIn("rule R4", text)
        self.assertIn("confidence=high", text)
        self.assertIn("REQUIRED iter-1 sequence", text)
        self.assertIn("recipe_register_retiming", text)
        self.assertIn("Evidence:", text)
        self.assertIn("corescore_500_mod", text)

    def test_imperative_language_pins_invariant(self):
        # Imperative ordering is part of the prompt contract; advisory
        # wording may not prevent direct physical-optimization calls.
        plan = decide_recipe_path(_corescore_features())
        text = plan.format_for_prompt()
        # Step 1 must be visually marked as the start point.
        self.assertIn("★ START HERE → recipe_register_retiming", text)
        # The anti-direct-call warning must be present and name the
        # specific risky tools.
        self.assertIn("Do NOT make a direct vivado_phys_opt_design", text)
        self.assertIn("vivado_place_design", text)
        # Override policy must explicitly forbid skipping step 1.
        self.assertIn("Skipping step 1 entirely", text)

    def test_fallback_plan_renders_without_actions_section(self):
        plan = decide_recipe_path(PhaseOneFeatures())
        text = plan.format_for_prompt()
        self.assertIn("rule FALLBACK", text)
        self.assertNotIn("REQUIRED iter-1 sequence", text)
        # Anti-direct-call warning is sequence-specific, so it must also
        # be absent from FALLBACK plans (no sequence to anchor to).
        self.assertNotIn("Do NOT make a direct vivado_phys_opt_design", text)

    def test_blocks_section_renders_when_present(self):
        plan = decide_recipe_path(_r2_profile_features())
        text = plan.format_for_prompt()
        self.assertIn("BLOCKED as iter-1", text)
        self.assertIn("recipe_cell_replacement", text)


if __name__ == "__main__":
    unittest.main()


class TestRuleR7ClosureClass(unittest.TestCase):
    """R7 — near-met closure class (live gap: demo_corundum 2026-06-11)."""

    @staticmethod
    def _corundum_features(**overrides):
        base = dict(
            wns_ns=-0.099,
            # fmax_ratio = CP/(CP+0.099) ≈ 95% at CP=2.0
            clock_period_ns=2.0,
            failing_endpoint_count=42,
            remaining_wall_budget_s=50 * 60,
        )
        base.update(overrides)
        return PhaseOneFeatures(**base)

    def test_corundum_profile_routes_to_r7_not_fallback(self):
        plan = decide_recipe_path(self._corundum_features())
        self.assertEqual(plan.rule_id, "R7")
        self.assertIn("phys_opt", plan.summary.lower() + " " +
                      " ".join(a.name for a in plan.actions))
        # The near-met placement must be protected.
        self.assertIn("place_design -unplace", plan.blocks)
        names = [a.name for a in plan.actions]
        self.assertNotIn("vivado_place_design", names)

    def test_r7_does_not_fire_at_r4_floor(self):
        # |WNS| 0.5 belongs to R4's band, not R7 (no overlap).
        plan = decide_recipe_path(self._corundum_features(
            wns_ns=-0.5, clock_period_ns=2.0))
        self.assertNotEqual(plan.rule_id, "R7")

    def test_r7_requires_high_fmax_ratio(self):
        # Small WNS but tiny clock period -> low ratio -> not the closure class.
        plan = decide_recipe_path(self._corundum_features(
            wns_ns=-0.4, clock_period_ns=0.1))
        self.assertNotEqual(plan.rule_id, "R7")

    def test_known_benchmarks_routing_unchanged(self):
        # R7 fires strictly below R4's floor, so every calibrated design keeps
        # its rule. Spot-check the four fixture profiles.
        self.assertEqual(decide_recipe_path(_boom_soc_features()).rule_id, "R1")
        self.assertEqual(decide_recipe_path(_ispd16_features()).rule_id, "R1")
        for fx in (_finn_features, _corescore_features):
            self.assertNotEqual(decide_recipe_path(fx()).rule_id, "R7")


# Near risk-asymmetric boundaries, ties resolve to R1 route-first.
# This applies only to the 1.5 ns WNS split and the failing-endpoint
# floor for deep-extreme designs.

class RouterTiebreakTests(unittest.TestCase):

    # --- Boundary 1: R1 route-first split (WNS-scale) -------------------

    def test_in_band_wns_flips_retime_first_to_route_first(self):
        # At |WNS| 9.2 ns, R1 selects route-first because the case is inside
        # the >=8.5 ns band, despite being below the 10.0 ns split.
        plan = decide_recipe_path(_boom_soc_features(wns_ns=-9.2))
        self.assertEqual(plan.rule_id, "R1")
        self.assertEqual([a.name for a in plan.actions],
                         ["vivado_run_tcl",
                          "vivado_route_design",
                          "vivado_phys_opt_design"])
        self.assertIn("route_design -unroute", plan.actions[0].note)
        self.assertIn("[router-tiebreak]", plan.rationale)

    def test_wns_band_lower_edge_inclusive(self):
        # Exactly at split - band (8.5) → still in band → flips.
        edge = R1_ROUTE_FIRST_WNS_NS - TIEBREAK_WNS_BAND_NS
        plan = decide_recipe_path(_boom_soc_features(wns_ns=-edge))
        self.assertEqual(plan.actions[0].name, "vivado_run_tcl")
        self.assertIn("[router-tiebreak]", plan.rationale)

    def test_out_of_band_wns_keeps_retime_first(self):
        # Just below the band (8.4) → proven retime-first path untouched.
        plan = decide_recipe_path(_boom_soc_features(wns_ns=-8.4))
        self.assertEqual(plan.rule_id, "R1")
        self.assertEqual(plan.actions[0].name, "vivado_phys_opt_design")
        self.assertNotIn("[router-tiebreak]", plan.rationale)

    def test_ispd16_stays_outside_the_band(self):
        # A |WNS| of 7.75 ns is below the 8.5 ns band and remains retime-first.
        plan = decide_recipe_path(_ispd16_features())
        self.assertEqual(plan.rule_id, "R1")
        self.assertEqual(plan.actions[0].name, "vivado_phys_opt_design")
        self.assertNotIn("[router-tiebreak]", plan.rationale)

    def test_already_route_first_never_marked_as_tiebreak(self):
        # Above the split the tie-break has nothing to do — plan is
        # route-first via the normal rule, no tiebreak marker.
        plan = decide_recipe_path(_boom_soc_features())  # |WNS| 19.16
        self.assertEqual(plan.actions[0].name, "vivado_run_tcl")
        self.assertNotIn("[router-tiebreak]", plan.rationale)

    # --- Boundary 2: R1 failing-endpoint floor (count-scale) ------------

    def test_in_band_count_flips_fallback_to_route_first(self):
        # 95k failing (in [85k, 100k)), |WNS| 14 (unambiguously deep),
        # ratio ~11%, spread unknown → R2 can't fire → old behavior
        # FALLBACK. Must flip to R1 route-first.
        f = PhaseOneFeatures(
            wns_ns=-14.0,
            clock_period_ns=1.666,   # ratio ≈ 11% < 0.20
            failing_endpoint_count=95_000,
            critical_path_avg_spread_tiles=None,
            remaining_wall_budget_s=45 * 60,
        )
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "R1")
        self.assertEqual(plan.actions[0].name, "vivado_run_tcl")
        self.assertIn("[router-tiebreak]", plan.rationale)

    def test_in_band_count_flips_r2_to_route_first(self):
        # Same shape but spread known/high → old behavior R2 (a sweep that
        # cannot move a 14 ns miss). Must flip to R1 route-first.
        f = PhaseOneFeatures(
            wns_ns=-14.0,
            clock_period_ns=1.666,
            failing_endpoint_count=90_000,
            critical_path_avg_spread_tiles=90.0,
            remaining_wall_budget_s=45 * 60,
        )
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "R1")
        self.assertEqual(plan.actions[0].name, "vivado_run_tcl")
        self.assertIn("[router-tiebreak]", plan.rationale)

    def test_count_band_requires_unambiguous_deep_wns(self):
        # In the count band but |WNS| 8.0 < 10.0 (not strictly deep-extreme)
        # → NO band-on-band compounding; stays on the normal rule (R2 here).
        f = PhaseOneFeatures(
            wns_ns=-8.0,
            clock_period_ns=1.666,   # ratio ≈ 17% (< R2's 0.25)
            failing_endpoint_count=95_000,
            critical_path_avg_spread_tiles=302.0,
            remaining_wall_budget_s=45 * 60,
        )
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "R2")
        self.assertNotIn("[router-tiebreak]", plan.rationale)

    def test_out_of_band_count_untouched(self):
        # Below the band (84,999 < 85,000) even a deep-extreme design keeps
        # its normal routing (R2 here — spread 302, ratio ~10%).
        f = PhaseOneFeatures(
            wns_ns=-14.0,
            clock_period_ns=1.666,
            failing_endpoint_count=(
                R1_FAILING_ENDPOINTS_MIN - TIEBREAK_FAILING_BAND - 1),
            critical_path_avg_spread_tiles=302.0,
            remaining_wall_budget_s=45 * 60,
        )
        plan = decide_recipe_path(f)
        self.assertNotEqual(plan.rule_id, "R1")
        self.assertNotIn("[router-tiebreak]", plan.rationale)

    def test_count_band_lower_edge_inclusive(self):
        f = PhaseOneFeatures(
            wns_ns=-14.0,
            clock_period_ns=1.666,
            failing_endpoint_count=(
                R1_FAILING_ENDPOINTS_MIN - TIEBREAK_FAILING_BAND),
            critical_path_avg_spread_tiles=302.0,
            remaining_wall_budget_s=45 * 60,
        )
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "R1")
        self.assertIn("[router-tiebreak]", plan.rationale)

    # --- None-feature guards ---------------------------------------------

    def test_never_fires_on_empty_features(self):
        plan = decide_recipe_path(PhaseOneFeatures())
        self.assertEqual(plan.rule_id, "FALLBACK")
        self.assertNotIn("[router-tiebreak]", plan.rationale)

    def test_never_fires_without_wns(self):
        f = PhaseOneFeatures(failing_endpoint_count=95_000,
                             clock_period_ns=1.666)
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "FALLBACK")
        self.assertNotIn("[router-tiebreak]", plan.rationale)

    def test_never_fires_without_ratio(self):
        # Deep WNS + in-band count but no clock period → ratio is None →
        # the count-scale tie-break must not fire.
        f = PhaseOneFeatures(wns_ns=-14.0, failing_endpoint_count=95_000)
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "OOB")
        # ratio not confirmed far-from-target => must be the never-worse
        # sweep, NOT the destructive route-first re-route.
        self.assertNotIn("vivado_route_design",
                         [a.name for a in plan.actions])
        self.assertNotIn("[router-tiebreak]", plan.rationale)

    # --- Fixture-freeze regression ---------------------------------------

    def test_tiebreak_does_not_reroute_any_existing_fixture(self):
        """Every calibrated fixture keeps BOTH its rule and its first
        action, and none carries the tiebreak marker. If any entry here
        flips, the band is wrong — report it, do not tune around it."""
        expected = {
            # fixture                          (rule, first action)
            "boom_soc": (_boom_soc_features(),
                         "R1", "vivado_run_tcl"),          # route-first (normal)
            "boom_v2": (_boom_soc_features(
                            wns_ns=-11.392, clock_period_ns=1.569,
                            failing_endpoint_count=220_131,
                            critical_path_avg_spread_tiles=249.7),
                        "R1", "vivado_run_tcl"),           # route-first (normal)
            "ispd16": (_ispd16_features(),
                       "R1", "vivado_phys_opt_design"),    # retime-first kept
            "r2_profile": (_r2_profile_features(),
                           "R2", "recipe_post_route_phys_opt_sweep"),
            "finn": (_finn_features(), "R3", "vivado_run_tcl"),
            "corescore": (_corescore_features(),
                          "R4", "recipe_register_retiming"),
            "demo_corundum": (
                PhaseOneFeatures(wns_ns=-0.099, clock_period_ns=2.0,
                                 failing_endpoint_count=42,
                                 remaining_wall_budget_s=50 * 60),
                "R7", "vivado_run_tcl"),
            "r3_gap_fill": (
                PhaseOneFeatures(wns_ns=-4.0, clock_period_ns=3.0,
                                 failing_endpoint_count=20_000,
                                 critical_path_avg_spread_tiles=120.0,
                                 remaining_wall_budget_s=30 * 60),
                "R3", "vivado_run_tcl"),
        }
        for name, (features, want_rule, want_first) in expected.items():
            plan = decide_recipe_path(features)
            self.assertEqual(plan.rule_id, want_rule,
                f"{name}: rule changed to {plan.rule_id} — tie-break band is wrong")
            self.assertEqual(plan.actions[0].name, want_first,
                f"{name}: first action changed — tie-break band is wrong")
            self.assertNotIn("[router-tiebreak]", plan.rationale,
                f"{name}: tie-break fired on a calibrated fixture — band is wrong")


# If one Phase-1 feature is unavailable, routing degrades to the nearest
# safe positive rule rather than an unconstrained fallback.
# This layer runs only after all positive rules decline, so it cannot
# redirect an already matched design.

class RouterFeatureNoneDegradationTests(unittest.TestCase):

    # --- Degraded R1: failing_endpoint_count=None ------------------------

    def test_degraded_r1_fires_on_deep_extreme_with_failing_none(self):
        # boom-shaped WNS/ratio but the failing-endpoint step was skipped
        # (e.g. by the T4a wall cap). Old behavior: FALLBACK. Must route
        # R1's ROUTE-FIRST plan (bankable floor), never retime-first.
        f = _boom_soc_features(failing_endpoint_count=None,
                               critical_path_avg_spread_tiles=None)
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "R1")
        self.assertEqual([a.name for a in plan.actions],
                         ["vivado_run_tcl",
                          "vivado_route_design",
                          "vivado_phys_opt_design"])
        self.assertIn("route_design -unroute", plan.actions[0].note)
        self.assertIn(R1_DEGRADED_MARKER, plan.rationale)
        # Degraded R1 must not carry the tie-break marker: failing=None
        # cannot enter the count-scale band (None-guard kept), and the
        # deep-extreme gate keeps it out of the WNS band.
        self.assertNotIn("[router-tiebreak]", plan.rationale)

    def test_degraded_r1_requires_deep_extreme_wns(self):
        # A 9.0 ns WNS magnitude passes the R1 gate but does not establish
        # an unambiguous route-first profile without a failing count.
        # Spread is also unknown here so R2 cannot claim the scenario.
        f = _boom_soc_features(wns_ns=-9.0,
                               failing_endpoint_count=None,
                               critical_path_avg_spread_tiles=None)
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "OOB")
        # ratio not confirmed far-from-target => must be the never-worse
        # sweep, NOT the destructive route-first re-route.
        self.assertNotIn("vivado_route_design",
                         [a.name for a in plan.actions])
        self.assertNotIn(R1_DEGRADED_MARKER, plan.rationale)
        self.assertNotIn("[router-tiebreak]", plan.rationale)

    def test_degraded_r1_requires_ratio_gate(self):
        # Deep WNS but fmax_ratio ≥ 0.20 fails R1's own ratio gate →
        # profile is not R1-shaped → FALLBACK.
        f = PhaseOneFeatures(
            wns_ns=-12.0,
            clock_period_ns=4.0,   # 4/(4+12) = 25% ≥ 0.20
            failing_endpoint_count=None,
            remaining_wall_budget_s=45 * 60,
        )
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "OOB")
        # ratio not confirmed far-from-target => must be the never-worse
        # sweep, NOT the destructive route-first re-route.
        self.assertNotIn("vivado_route_design",
                         [a.name for a in plan.actions])
        self.assertNotIn(R1_DEGRADED_MARKER, plan.rationale)

    def test_degraded_r1_requires_ratio_present(self):
        # No clock period → no ratio → nothing is unambiguous → FALLBACK.
        f = PhaseOneFeatures(wns_ns=-19.0, failing_endpoint_count=None)
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "OOB")
        # ratio not confirmed far-from-target => must be the never-worse
        # sweep, NOT the destructive route-first re-route.
        self.assertNotIn("vivado_route_design",
                         [a.name for a in plan.actions])

    def test_degraded_r1_inert_when_failing_known(self):
        # Full boom fixture (failing present) → normal R1 route-first, no
        # degraded marker (degradation fires ONLY on feature-None).
        plan = decide_recipe_path(_boom_soc_features())
        self.assertEqual(plan.rule_id, "R1")
        self.assertNotIn(R1_DEGRADED_MARKER, plan.rationale)

    def test_degraded_r1_claims_double_none_when_deep_extreme(self):
        # failing=None AND spread=None on a deep-extreme design: only the
        # degraded-R1 branch may claim it (degraded R2 requires a KNOWN
        # medium failing count).
        f = PhaseOneFeatures(
            wns_ns=-14.0,
            clock_period_ns=1.666,   # ratio ≈ 11% < 0.20
            failing_endpoint_count=None,
            critical_path_avg_spread_tiles=None,
            remaining_wall_budget_s=45 * 60,
        )
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "R1")
        self.assertIn(R1_DEGRADED_MARKER, plan.rationale)
        self.assertNotIn(R2_DEGRADED_MARKER, plan.rationale)

    # --- Degraded R2: spread=None ----------------------------------------

    def test_degraded_r2_fires_on_spread_none_with_profile_match(self):
        # R2 profile with only the spread metric skipped. Old behavior:
        # FALLBACK (R2 hard-requires spread). Must route R2's
        # placement-preserving sweep with the degraded marker.
        f = _r2_profile_features(critical_path_avg_spread_tiles=None)
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "R2")
        self.assertEqual(plan.actions[0].name, "recipe_post_route_phys_opt_sweep")
        self.assertIn("recipe_cell_replacement", plan.blocks)
        self.assertIn(R2_DEGRADED_MARKER, plan.rationale)

    def test_degraded_r2_requires_known_failing_count(self):
        # spread=None AND failing=None with sub-deep WNS: could be an
        # R1-class monster where a sweep is refuted → ambiguous → FALLBACK.
        f = _r2_profile_features(critical_path_avg_spread_tiles=None,
                                 failing_endpoint_count=None)
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "OOB")
        self.assertNotIn(R2_DEGRADED_MARKER, plan.rationale)

    def test_degraded_r2_respects_failing_max(self):
        # failing ≥ 100k is R1 territory (but |WNS| 6.0 < 7.0 misses R1)
        # → profile is not R2's medium class → FALLBACK, no sweep.
        f = _r2_profile_features(
            critical_path_avg_spread_tiles=None,
            failing_endpoint_count=R2_FAILING_ENDPOINTS_MAX + 50_000)
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "OOB")
        self.assertNotIn(R2_DEGRADED_MARKER, plan.rationale)

    def test_degraded_r2_respects_wns_floor(self):
        # |WNS| 4.0 < R2's 5.0 floor (ratio kept < 0.25 so R3 can't fire)
        # → rest-of-profile does NOT match → FALLBACK.
        f = _r2_profile_features(critical_path_avg_spread_tiles=None,
                                 wns_ns=-4.0, clock_period_ns=1.0)
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "OOB")
        self.assertNotIn(R2_DEGRADED_MARKER, plan.rationale)

    def test_degraded_r2_inert_when_spread_known(self):
        # Full R2 fixture routes normally — no degraded marker.
        plan = decide_recipe_path(_r2_profile_features())
        self.assertEqual(plan.rule_id, "R2")
        self.assertNotIn(R2_DEGRADED_MARKER, plan.rationale)

    # --- Composition with the T2/C5 tie-break ----------------------------

    def test_degraded_r2_composes_with_count_tiebreak(self):
        # With spread unknown and a known failing count in [85k, 100k),
        # degraded R2 fires first. The count-scale tie-break then selects
        # R1 route-first because the design is unambiguously deep-extreme.
        f = PhaseOneFeatures(
            wns_ns=-14.0,
            clock_period_ns=1.666,   # ratio ≈ 11%
            failing_endpoint_count=90_000,
            critical_path_avg_spread_tiles=None,
            remaining_wall_budget_s=45 * 60,
        )
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "R1")
        self.assertEqual(plan.actions[0].name, "vivado_run_tcl")
        self.assertIn("[router-tiebreak]", plan.rationale)


class OutOfBandNetTests(unittest.TestCase):
    """The OOB net: a design matching NO calibrated rule must still
    receive a conservative, bankable plan — never an action-less FALLBACK.

    Why this matters: `_build_recipe_router_block` SUPPRESSES an action-less
    FALLBACK entirely, so the LLM receives no routing guidance and freelances.
    That is the documented mechanism that zeroed the boom class, and a fuzz
    sweep put it at ~32% of the reachable feature space. The final round
    scores HIDDEN designs, which land out-of-band by definition.
    """

    def test_uncovered_midband_gets_actions_not_bare_fallback(self):
        # |WNS| 4.0 with ratio 20%: below R2's 5.0 floor, above R3's 5.0 cap
        # by ratio (<0.35), not R4/R7 — a real hole in the calibrated bands.
        f = PhaseOneFeatures(
            wns_ns=-4.0, clock_period_ns=1.0,
            failing_endpoint_count=20_000,
            critical_path_avg_spread_tiles=50.0,
            remaining_wall_budget_s=30 * 60,
        )
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "OOB")
        self.assertTrue(plan.actions, "OOB must never emit an action-less plan")
        self.assertIn(OOB_MARKER, plan.rationale)
        self.assertEqual(plan.confidence, "low")

    def test_deep_and_far_from_target_gets_route_first_floor(self):
        # Deep-extreme AND confirmed far from target, but failing count below
        # R1's 100k gate → no positive rule → OOB must hand back the PROVEN
        # bankable floor (route-first), not improvise.
        f = PhaseOneFeatures(
            wns_ns=-12.0, clock_period_ns=1.6,     # ratio 11.8% < 0.20
            failing_endpoint_count=30_000,          # < R1's 100k
            critical_path_avg_spread_tiles=50.0,    # < R2's 70
            remaining_wall_budget_s=45 * 60,
        )
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "OOB")
        self.assertIn("vivado_route_design", [a.name for a in plan.actions])

    def test_deep_wns_but_nearly_met_must_not_route_first(self):
        """The near-met trap: absolute |WNS| is NOT evidence of a crisis.

        At a 200 ns period, WNS -12 ns is 94% of target. Unroute +
        AggressiveExplore would destroy a near-win, so the OOB net must
        require a KNOWN ratio below R1_FMAX_RATIO_MAX before route-first.
        """
        f = PhaseOneFeatures(wns_ns=-12.0, clock_period_ns=200.0)
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "OOB")
        self.assertNotIn("vivado_route_design", [a.name for a in plan.actions])
        self.assertGreater(f.achievable_fmax_ratio, R1_FMAX_RATIO_MAX)

    def test_deep_wns_without_ratio_must_not_route_first(self):
        # No clock period → ratio unknown → cannot confirm far-from-target.
        f = PhaseOneFeatures(wns_ns=-19.0, failing_endpoint_count=200_000)
        plan = decide_recipe_path(f)
        self.assertEqual(plan.rule_id, "OOB")
        self.assertNotIn("vivado_route_design", [a.name for a in plan.actions])

    def test_oob_always_blocks_unplace(self):
        """Verify that out-of-budget routing always blocks a full unplace
        operation.

        Without a reliable size estimate, full replacement can exceed the
        remaining runtime budget.
        """
        for f in (
            PhaseOneFeatures(wns_ns=-4.0, clock_period_ns=1.0,
                             failing_endpoint_count=20_000,
                             critical_path_avg_spread_tiles=50.0),
            PhaseOneFeatures(wns_ns=-12.0, clock_period_ns=1.6,
                             failing_endpoint_count=30_000,
                             critical_path_avg_spread_tiles=50.0),
            PhaseOneFeatures(wns_ns=-2.0),
        ):
            plan = decide_recipe_path(f)
            self.assertEqual(plan.rule_id, "OOB")
            self.assertIn("place_design -unplace", plan.blocks)

    def test_no_wns_stays_honest_fallback(self):
        # With no WNS there is no region to reason about — FALLBACK is the
        # honest answer and must NOT be dressed up as a routing decision.
        plan = decide_recipe_path(PhaseOneFeatures(failing_endpoint_count=1000))
        self.assertEqual(plan.rule_id, "FALLBACK")
        self.assertFalse(plan.actions)

    def test_oob_never_preempts_a_calibrated_rule(self):
        """Anchor regression: every known-design fixture must still route to
        its calibrated rule. OOB fires ONLY after all positive rules and the
        degraded rescue have declined."""
        for name, feats in (
            ("boom_soc", _boom_soc_features()),
            ("ispd16", _ispd16_features()),
            ("finn", _finn_features()),
            ("corescore", _corescore_features()),
        ):
            plan = decide_recipe_path(feats)
            self.assertNotEqual(plan.rule_id, "OOB",
                                f"{name} must not fall to the OOB net")

    def test_timing_met_design_is_not_sent_to_route_first(self):
        # Positive WNS represents met timing, so the sign-aware magnitude
        # is zero and R7 protects placement. The route tool name alone is
        # not a discriminator because R7 permits incremental cleanup.
        plan = decide_recipe_path(PhaseOneFeatures(wns_ns=+1.5,
                                                   clock_period_ns=10.0))
        self.assertEqual(plan.rule_id, "R7")
        notes = " ".join(a.note for a in plan.actions)
        self.assertNotIn("AggressiveExplore", notes)
        self.assertNotIn("unroute", notes)


class SizeAwareClassGTests(unittest.TestCase):
    """Verify that Class-G feasibility uses the measured size model.

    Runtime estimates must scale with design size rather than assume a fixed
    exploration duration.
    """

    def test_explore_model_matches_measurements(self):
        # 39 timed calls / 12 designs. Extremes are near-exact; mid-range is
        # over-predicted, i.e. the model errs SAFE.
        self.assertAlmostEqual(explore_runtime_estimate_s(3373), 62.1, delta=4)     # measured 60
        self.assertAlmostEqual(explore_runtime_estimate_s(252741), 585.8, delta=8)  # measured 582
        self.assertGreater(explore_runtime_estimate_s(157166), 315)                 # measured 315

    def test_model_returns_none_without_size(self):
        self.assertIsNone(explore_runtime_estimate_s(None))
        self.assertIsNone(explore_runtime_estimate_s(0))
        self.assertIsNone(classg_runtime_estimate_s(None))

    def test_classg_costs_more_than_explore_alone(self):
        # Class G is Explore + route + bank; a gate on Explore alone
        # under-counts the sequence that actually has to fit.
        self.assertGreater(classg_runtime_estimate_s(532160),
                           explore_runtime_estimate_s(532160))

    def test_huge_design_declined_when_wall_cannot_fit_class_g(self):
        f = _finn_features(cell_count=532_160, remaining_wall_budget_s=2000)
        self.assertIsNotNone(explore_infeasible_reason(f))
        self.assertNotEqual(decide_recipe_path(f).rule_id, "R3")

    def test_huge_design_allowed_when_wall_is_ample(self):
        f = _finn_features(cell_count=532_160, remaining_wall_budget_s=3400)
        self.assertIsNone(explore_infeasible_reason(f))
        self.assertEqual(decide_recipe_path(f).rule_id, "R3")

    def test_small_design_never_blocked_by_the_size_gate(self):
        f = _finn_features(cell_count=3_373, remaining_wall_budget_s=1600)
        self.assertIsNone(explore_infeasible_reason(f))

    def test_gate_is_inert_without_cell_count(self):
        # SAFETY-ONLY contract: unknown size must not change any routing.
        f = _finn_features(cell_count=None)
        self.assertIsNone(explore_infeasible_reason(f))
        self.assertEqual(decide_recipe_path(f).rule_id, "R3")


class ResourceBlockRuleTests(unittest.TestCase):
    """R8: resource-risk BLOCKS, never recipes."""

    def _deep(self, **kw):
        return PhaseOneFeatures(
            wns_ns=-19.0, clock_period_ns=1.6, failing_endpoint_count=50_000,
            critical_path_avg_spread_tiles=50.0,
            remaining_wall_budget_s=2700, **kw)

    def test_high_util_blocks_replacement_moves(self):
        plan = decide_recipe_path(self._deep(lut_util_pct=R8_HIGH_UTIL_PCT + 5))
        self.assertIn("place_design -unplace", plan.blocks)
        self.assertIn("recipe_cell_replacement", plan.blocks)

    def test_memory_dominated_blocks_replacement_moves(self):
        plan = decide_recipe_path(self._deep(memory_dominated=True))
        self.assertIn("place_design -unplace", plan.blocks)
        self.assertIn("recipe_cell_replacement", plan.blocks)

    def test_high_util_withholds_oob_route_first(self):
        """The concrete requirement: forbid unroute->AggressiveExplore on a
        dense design. B1's OOB deep branch recommends exactly that."""
        plan = decide_recipe_path(self._deep(lut_util_pct=80.0))
        self.assertEqual(plan.rule_id, "OOB")
        self.assertNotIn("vivado_route_design", [a.name for a in plan.actions])

    def test_normal_util_keeps_oob_route_first(self):
        plan = decide_recipe_path(self._deep(lut_util_pct=25.0))
        self.assertEqual(plan.rule_id, "OOB")
        self.assertIn("vivado_route_design", [a.name for a in plan.actions])

    def test_unknown_util_keeps_oob_route_first(self):
        # Missing feature => rule cannot fire => no behaviour change.
        plan = decide_recipe_path(self._deep())
        self.assertIn("vivado_route_design", [a.name for a in plan.actions])

    def test_r8_never_downgrades_a_calibrated_rule(self):
        """Verify that R8 adds blocks without replacing a calibrated route.

        Utilization alone is insufficient to downgrade a rule supported by
        validated outcomes.
        """
        plan = decide_recipe_path(_boom_soc_features(lut_util_pct=80.0))
        self.assertEqual(plan.rule_id, "R1")


class PlanCoherenceTests(unittest.TestCase):
    """A plan must never recommend a move it simultaneously blocks.

    Found by the constants audit: R8 blocks `place_design -unplace` at
    high utilization, and that op is Class G's FIRST step — so R3 firing there
    produced a self-contradictory plan ("do Class G" + "do not unplace").
    """

    def _plans(self):
        for wns in (-0.3, -1.5, -3.5, -6.0, -12.0, -19.0):
            for util in (None, 25.0, 80.0):
                for mem in (None, True):
                    for cells in (None, 3_373, 252_741):
                        yield decide_recipe_path(PhaseOneFeatures(
                            wns_ns=wns, clock_period_ns=1.6,
                            failing_endpoint_count=50_000,
                            critical_path_avg_spread_tiles=120.0,
                            remaining_wall_budget_s=2700,
                            cell_count=cells, lut_util_pct=util,
                            memory_dominated=mem))

    def test_never_recommends_a_blocked_move(self):
        for plan in self._plans():
            text = " ".join(a.name + " " + a.note for a in plan.actions)
            for blocked in plan.blocks:
                if blocked == "place_design -unplace":
                    self.assertNotIn(
                        "unplace", text,
                        f"{plan.rule_id} recommends unplace while blocking it")

    def test_high_util_never_recommends_class_g(self):
        for cells in (None, 3_373, 252_741):
            plan = decide_recipe_path(PhaseOneFeatures(
                wns_ns=-3.5, clock_period_ns=2.0, failing_endpoint_count=5_000,
                critical_path_avg_spread_tiles=120.0,
                remaining_wall_budget_s=2700, cell_count=cells,
                lut_util_pct=80.0))
            self.assertNotEqual(plan.rule_id, "R3")

    def test_r1_is_not_downgraded_by_resource_guards(self):
        # R1's sequence is unroute->route, which R8 does NOT block, so R1 stays
        # coherent AND keeps its measured evidence. Contrast with R3 above.
        for kw in ({"lut_util_pct": 80.0}, {"memory_dominated": True}):
            plan = decide_recipe_path(_boom_soc_features(**kw))
            self.assertEqual(plan.rule_id, "R1")

    def test_size_aware_gate_recovers_the_vexriscv_case(self):
        """Verify that the size-aware gate admits a short Class G sequence despite
        the fixed time floor.

        A feasible short sequence must not be sent to fallback solely because a
        size-blind minimum budget exceeds its predicted runtime.
        """
        vex = dict(wns_ns=-1.654, clock_period_ns=1.57,
                   failing_endpoint_count=1_937,
                   critical_path_avg_spread_tiles=54.0,
                   remaining_wall_budget_s=1425)
        self.assertEqual(decide_recipe_path(
            PhaseOneFeatures(**vex, cell_count=3_373)).rule_id, "R3")
        # …but with size UNKNOWN we must NOT guess our way past the floor.
        self.assertNotEqual(decide_recipe_path(PhaseOneFeatures(**vex)).rule_id, "R3")
