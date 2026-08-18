"""Pin the EFFECTIVE eval-box configuration so it cannot drift silently.

Two failures three days apart were the same class — nobody could state the
configuration our own submission actually runs with:

  * jul29: the uniform ILS stack was default-OFF and **not on the eval path**.
    Every A/B that measured it was measuring something we would not have shipped.
  * jul30: `FPL26_DEEP_REPLACE_UNBANDED` ships **ON** while its own code comment
    says "DEFAULT OFF ... Ships OFF until farm-validated, per house rule",
    because the Makefile sets it and the comment describes the python layer.

Both were found by hand-auditing, days after the fact. These tests turn that audit
into a suite failure.

The assertions are deliberately *exact set* comparisons, not "contains". A flag
silently appearing in or vanishing from the ship path is precisely the bug, so a
subset check would not catch it. When you intentionally change what ships, this
file must be edited in the same commit — that edit IS the ship decision, and it
should be visible in review rather than emergent from a Makefile diff.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import ship_config  # noqa: E402


# What a clean eval box gets with NO FPL26_*/make variables set.
# Every entry here is a deliberate ship decision, not an accident of layering.
EXPECTED_SHIPS_ON = {
    # --- the uniform ILS stack (jul29 fix put these on the eval path) -------
    "FPL26_ILS_PLACE_RETRY_LADDER",
    # aug02: SKIP_UNAFFORDABLE promoted as a RISK-ACCEPTED user decision after
    # its corescore never-worse gate FAILED as pre-registered (gaps 0.73/3.09,
    # both <= the 3.5 MHz noise floor; treatment values all inside the
    # historical flag-OFF distribution - v2.0's own sweep drew 80.03 with no
    # flag). digit EV ~ +5.4 MHz. A documented trade, NOT a passed gate.
    "FPL26_ILS_LADDER_SKIP_UNAFFORDABLE",
    "FPL26_ILS_MEASURED_BASIS",
    "FPL26_ILS_INCR_ROUTE",
    "FPL26_ILS_INCR_ROUTE_FIRST",
    # --- deep-replace family ------------------------------------------------
    "FPL26_DEEP_REPLACE",
    "FPL26_DEEP_REPLACE_FIRST",
    "FPL26_DEEP_REPLACE_FIRST_BANDED",
    # aug09 (v5.3): B3 small-floor sibling — mid-band + 300s anchor cap, so it
    # treats only mini-ISP among the knowns; hidden tiny mid-band designs get
    # the same insurance MUX candidate. Gate: PREREG_V53_GATE_aug09.
    "FPL26_DEEP_REPLACE_B3",
    # aug09 (v5.5): logic-floor physics attestation — post-B3-adopt, one
    # report_timing round-trip; fires only at a measured unbreakable floor
    # (bound<=10MHz over 32 paths @45ps/hop, logic>=80%, hard-macro>=50%,
    # two-solve agreement); every guard failure is NO-FIRE.
    "FPL26_LOGIC_FLOOR_EXIT",
    # aug09 (v5.4): the wrapper honors the b3_floor_saturated.token sentinel —
    # stops the attempt loop and skips winner-polish when an attempt attests
    # the deterministic B3 floor (census: v5.3 gate attempts 2/3 + cross-box
    # bit-identical; polish NO_GAIN 5/5 from the floor).
    "FPL26_B3_FLOOR_EXIT",
    # ⚠ UNDER REVIEW (jul30): the code comment for this one says "DEFAULT OFF
    # ... Ships OFF until farm-validated". The Makefile ships it ON. Pinned here
    # as CURRENT REALITY, not as endorsement — if the review turns it off, this
    # entry goes with it.
    "FPL26_DEEP_REPLACE_UNBANDED",
    # aug06 SHIP DECISION. Waives the DEEP-extreme band and the
    # double-subtracted finalize reserve for deep-replace[first], for the
    # ILS-SIZE-GATED class ONLY (cells > ILS max_cells 300k) — the class where
    # ILS never arms, so there is no better LLM call to pre-empt.
    #
    # MEASURED, same build, 8-CPU capped, 3500s wall:
    #   ispd16   ship +24.11  ->  size-gated +115.37  VALID_OPTIMIZED (+91.26)
    #   boom_v1  ship +41.43  ->  treatment  +41.43   IDENTICAL (no-op: it
    #            already clears the band at |WNS| 19.16 and already affords)
    # Cannot fire on the other 13 known designs: the highest non-gated is
    # corescore at 252,741 cells vs boom 379,048 — a 50% gap around the
    # pre-existing 300k constant. corescore is exactly the design where FIRST
    # is measured HARMFUL (~-63 MHz), and it stays excluded.
    #
    # Alpha downside is bounded: the stage only repoints the pipeline on
    # VERDICT_ADOPTED *and* strict improvement, so a losing regeneration is
    # discarded — the cost is wall (gamma) only.
    # Kill switch: FPL26_NO_DEEP_FIRST_SIZEGATED=1; off-knob DEEP_FIRST_SIZEGATED=0.
    "FPL26_DEEP_FIRST_SIZEGATED",
    # --- loop economics -----------------------------------------------------
    "FPL26_ILS_HURDLE_CONTINUE",
    "FPL26_PREEMPT_LOOP_CLOCK",
    # --- hygiene ------------------------------------------------------------
    "FPL26_VIVADO_LEAK_FIX",
    # jul30 late, SHIP DECISION, default-ON with an opt-out kill switch.
    # DEFECT, not a lever: on the single-seed path the ILS seeded from
    # `self._best_valid_dcp` ITSELF, so its accept-path `write_checkpoint -force`
    # rewrote the banked mirror in place, while `_best_valid_mirror_size` is only
    # re-stamped after the whole stage. Any SIGTERM between the first accept and
    # the end of the stage (~20-40 min on a 3500s wall) hit the emergency
    # integrity gate -- "SHIP INTEGRITY FAIL: best_valid mirror size N != banked
    # M" -- which distrusts the mirror and falls back to the BASELINE copy:
    # alpha 0 on that design, with the ILS's gain sitting unused on disk.
    # Shipped ON rather than OFF-until-validated because it changes NO
    # optimization decision (same bytes, different path) and leaving it off
    # leaves an unknown-probability alpha->0 path live on the eval path. Ranked
    # #1 of three known defects, unanimously, by the jul30 5-seat panel.
    "FPL26_ILS_SEED_COPY",
    # jul31, SHIP DECISION, default-ON with a kill switch and a MEASURED size
    # gate. deep-replace B2 ran `phys_opt AlternateFlowWithRetiming` in the same
    # Vivado process that had just unplaced, re-placed and routed the design;
    # boom_soc died there three times in one night, the last alone on an idle box
    # after reproducing B1's wns=-10.256 exactly. B1 is banked to disk before B2
    # starts, so a fresh session costs wall and can never cost a result.
    # SCOPE IS THE POINT: it fires only when MEASURED place+route >= 1200s, and
    # the corpus puts the boom cluster at 1398-1528s against a next-highest
    # (corescore) of 539s — an 859s gap. All 15 designs that completed the jul31
    # sweep are excluded by 2.2x or more.
    "FPL26_DEEP_REPLACE_B2_RESTART",
    # --- cost calibration (jul30) -------------------------------------------
    # ExtraNetDelay_high's prior sat at the corpus p90 (3.0) rather than the
    # median (2.10), and on the eval box that refused optical's key cycle by 58s
    # -- alpha +13.07 shipped where the farm reproduces +32.38. Proven 2/2 in
    # wave22 at the discriminating wall: +19.31 and +13.41 MHz, no overrun,
    # wall +16s (the treatment SUBSTITUTES one cycle for two, it does not add).
    # See final_round/COST_PRIOR_COST_US_17_POINTS_jul30.md.
    "FPL26_ILS_MEASURED_PRIORS",
    # jul31, SHIP DECISION, default-ON with a kill switch. The COMPANION to
    # MEASURED_PRIORS above, and the reason both can ship together at all.
    # ExtraNetDelay_high is the costliest combo in the rotation (~1300s) and
    # accepts only 33% of the time across 112 corpus runs on 10 designs. Blocked
    # iff BOTH: failing/spread < 363 (Phase-1 features -> this design does not
    # benefit) AND est/remaining >= 0.60 (this cycle would bet the window).
    # Either condition ALONE has collateral -- density-only costs 5 accepts (4
    # vexriscv's), share-only kills an optical accept at 0.80. Together they lose
    # 0 of 37 corpus accepts and still avoid 8 wasted ~1300s arms.
    # WHY IT SHIPS: matched live A/B, one build, ship config, both directions
    # exercised the same day -- digit (density 175) BLOCKED -> +72.59 fmax
    # 439.56; optical (density 517) ARMED -> +32.38 fmax 357.27. Each is that
    # design's best-ever, reached SIMULTANEOUSLY on one configuration. Before
    # this gate digit reached 72.59 only with MEASURED_PRIORS OFF, and turning
    # that off costs optical the ~17.7 eval points the entry above earned.
    # Fails OPEN on any missing/non-finite feature or unpriceable cycle.
    # See final_round/DIGIT_LADDER_ORDER_jul31.md.
    "FPL26_ILS_ENDHIGH_DENSITY_GATE",
    # aug01: PROMOTED, and this REVERSES the jul28 "stays OFF permanently" note
    # that used to sit in EXPECTED_NOTABLE_OFF. Defensible only because that rule
    # was aimed at TUNING THE CONSTANT to spam: 0.7 is still
    # cfg.route_reroll_max_wns_mag, reused unchanged, and moving it to fit a
    # result remains refused. The old note also mis-stated its own n — two of its
    # "3 near-met designs" (optical, 3d) are DEEP, i.e. provable no-ops, so the
    # prior evidence was n=1.
    # Promoted on ladder_ab_jul31: a paired A/B over the SIX designs the flag can
    # touch. See final_round/LADDER_ORDER_REOPENED_jul31.md.
    "FPL26_ILS_LADDER_ORDER_BY_WNS",
    # --- aug05 pin-drift correction (CURRENT REALITY, shipped in v3.0-v3.3
    # and validated there; this file had simply never been updated, so both
    # pin tests were failing at tag v3.3 already) ---------------------------
    # aug03/v3.0: frozen-Tcl candidate pass (shallow WLD + deep round-trip),
    # MUX-only banking. logicnets 116.47 / digit 73.56 / vtr_v2 5.15 lineage.
    "FPL26_RECIPE_PASS",
    # aug04/v3.0: size-anchored pre-LLM deep gate (vtr_v2 +0.45 SCORED).
    "FPL26_RECIPE_FIRST_DEEP",
    # aug05/v3.2: fir sub-band in-session floor + carve-out (fir 21.30 x5).
    "FPL26_FIR_SUBBAND_FLOOR",
    # --- v4.0 levers (v40-levers, aug05) — STAGING, pinned as CURRENT
    # REALITY of this branch, NOT as a passed gate: ship decision is the
    # PREREG_V40_LEVERS_aug05 3-design A/B + full-16 parity, not this edit.
    "FPL26_VEX2_RETIME_CANDIDATE",
    "FPL26_MINIISP_RETRY_HOLD",
    "FPL26_CORESCORE_ROUTE_RUNG",
    # --- v4.1 rev2 (v41-eggs, aug05 — PREREG_V41_REV2_aug05.md) — STAGING,
    # pinned as CURRENT REALITY of this branch, NOT as a passed gate (ship
    # decision = the rev2 A/B + full-16 parity).  NOTE: the interim
    # v4.1-eggs FPL26_SHALLOW_WLD_RETIME_CANDIDATE was Makefile-armed
    # WITHOUT this pin (both pin tests were red on the branch from that
    # commit until this one; verified red at 97af5fc); superseded by
    # OWNFRONT below.
    "FPL26_OWNFRONT_RETIME_CANDIDATE",
    "FPL26_MUX_MD5_TRUST",
    # --- v4.1.2 (v41-eggs, aug06 — PREREG_V41_REV2_aug05.md v4.1.2
    # section) — STAGING, pinned as CURRENT REALITY of this branch, NOT
    # as a passed gate (ship decision = spam ON x2 at 29.19-class + the
    # rest of full-16 parity).  Spam determinizer third shallow
    # candidate, band [0.60, 0.90) (the ownfront-vacated window);
    # off-knob SPAM_DET=0.
    "FPL26_SPAM_DETERMINIZER_CANDIDATE",
    # aug07 SHIP DECISION (v5.0). Retires the two "timing met -> stop" returns:
    # the met-timing ENTRY exit (ship the input at alpha = 0) and the mid-loop
    # zero-crossing exit. alpha is a DELTA of Fmax, and Fmax = 1000/(T - wns) is
    # UNCLAMPED in the organizers' own reference implementation
    # (docs/optimization_example.md) — so on T = 1.570, WNS 0.000 -> +0.200 is
    # +93 MHz that we were declining by construction.
    #
    # NOT measured, and cannot be: all 16 corpus designs enter deeply negative
    # (fir 0.313 ns is the shallowest) and none has ever crossed zero, so the
    # flag is provably unreachable on the whole farm — full-16 parity holds BY
    # CONSTRUCTION rather than by measurement. That is the argument for arming
    # it and also the honest limit of the claim.
    #
    # Bounded downside: the class it fires on scores alpha = 0 today, so the
    # worst case is gamma on a run that was already worth nothing. Termination
    # still belongs to the untouched guards (LLM cost breach, >=3-iteration
    # plateau, empty-spin at iteration >= 8, max_iterations, wall deadline).
    "FPL26_POSITIVE_SLACK_CONTINUE",
}

# Flags measured but deliberately NOT shipping. Each carries the reason, because
# "why is this off" is the question a future reader will actually have.
EXPECTED_NOTABLE_OFF = {
    # The 3-flag set: +3.39 on mini-ISP, -9.68 optical, clean -3.39 spam. Net
    # -0.80 mean-rank points. HOLD.
    "FPL26_ILS_LADDER_RESERVE",
    "FPL26_ILS_LADDER_STOP_ON_ACCEPT",
    "FPL26_ILS_RETRY_BASELINE_GATE",
    # Does not fix seed poisoning (spam's poisoning accept was 0.021 ns, above
    # the 0.010 threshold), and the corpus shows 4/5 terminal, not 4/4.
    "FPL26_ILS_REJECT_MICRO_ACCEPT",
    # Half of spam's +29.19 pair; evidence is one design whose record came from
    # a poisoned baseline. Needs generality data before it can ship.
    "FPL26_ILS_INCR_ROUTE_TERMINAL",
    # jul30, both built today and both awaiting their A/B. Listed here so that
    # "it defaults OFF" is ENFORCED rather than merely intended — the jul29
    # drift was a flag whose intended state and shipped state disagreed, and a
    # flag added today is exactly when that gap opens.
    #   PLATEAU_PRELOOP_FIX: arms the plateau exit when the best was banked
    #   pre-loop. Predicted beta+gamma-only; any alpha change is a fail.
    "FPL26_PLATEAU_PRELOOP_FIX",
    #   DEEP_REPLACE_NO_DOUBLE_RESERVE: stops the affordability gate
    #   re-subtracting the finalize reserve already removed from the deadline.
    #   22/30 corpus declines arm once corrected.
    "FPL26_DEEP_REPLACE_NO_DOUBLE_RESERVE",
    # jul31: `phys_opt_design -directive Default` to fixpoint, pre-loop. The
    # MCP server drops `path_groups` whenever a directive is set, so the calls
    # jul29 read as "critical-path targeted" were full-design Default runs —
    # which gain 26/30 (86.7%) against 70/479 (15%) for a true sub-option call.
    # fir's alpha is decided by how many the LLM happens to emit: nDefault>=2
    # selected the 21.30 mode 8/8, nDefault<=1 gave 9.07. Worth 12.23 MHz.
    # OFF because all 30 observations are ONE design (fir) — the endhigh gate's
    # overfitting shape. Promotion requires an all-16 A/B with no alpha or wall
    # regression on the other 15. See final_round/PHYSOPT_ARG_DROP_jul31.md.
    "FPL26_PHYSOPT_DEFAULT_FIXPOINT",
}

# CLI opt-ins gated on make variables: unset ⇒ the flag is NOT passed.
#
# WALL_HANDBACK LEFT THIS SET ON jul30 — it now ships default-ON with an OPT-OUT
# (`$(if $(filter 0 false no off,$(WALL_HANDBACK)),,--wall-handback)`, the same
# convention as --ils-polish in the same recipe). That is a deliberate ship
# decision on this evidence, all four pairs firing-verified and baseline-matched:
#   mini-ISP  alpha identical, calls 25 -> 7            +0.40 mean-rank
#   fir       21.30 -> 21.30 at a matched -0.218        null
#   optical   32.38 -> 32.38, wall +38s                 null (-0.03)
#   vtr_mcml  17.55 -> 18.72, wall -140s, calls 9 -> 5   WIN on all three axes
# 2 wins, 2 nulls, 0 harms, which meets the panel's pre-registered 5/5 rule
# ("ship after wave15 if null-or-better"). See test_wall_handback_now_ships below.
EXPECTED_CLI_OPTINS = {
    "SPLIT_AWARE": "--split-aware",
}

# Layer disagreements are ALLOWED but must be reviewed. The Makefile is the ship
# decision; the risk is only that reading the python default misleads you. A NEW
# disagreement appearing here unreviewed is the jul30 bug recurring.
EXPECTED_DISAGREEMENTS = {
    "FPL26_DEEP_REPLACE",
    "FPL26_DEEP_REPLACE_FIRST",
    "FPL26_DEEP_REPLACE_UNBANDED",
    # aug06, REVIEWED: same deliberate pattern as the v4.0 levers below —
    # python default OFF (never-touch-validated-paths), Makefile arms both
    # launch branches. See the EXPECTED_SHIPS_ON entry for the measurements.
    "FPL26_DEEP_FIRST_SIZEGATED",
    # aug09, REVIEWED (v5.3): B3 small-floor sibling — python default OFF
    # (house rule), Makefile arms both halves. Mid-band + anchor-capped;
    # null on every known except mini-ISP by the 300s cap arithmetic.
    # Evidence: od4/mg5 drills (-0.850 deterministic, 149-160s); gate =
    # PREREG_V53_GATE_aug09.
    "FPL26_DEEP_REPLACE_B3",
    # aug09, REVIEWED (v5.4): same python-OFF / Makefile-ON pattern for the
    # B3-floor-exit wrapper behavior. Gate = PREREG_V54 (same document).
    "FPL26_B3_FLOOR_EXIT",
    # aug09, REVIEWED (v5.5): same pattern for the logic-floor attestation.
    "FPL26_LOGIC_FLOOR_EXIT",
    "FPL26_ILS_HURDLE_CONTINUE",
    # jul30, REVIEWED: python default 0 (house rule: a new lever ships OFF until
    # validated), Makefile default 1 (it IS now validated -- wave22, 2/2 at the
    # discriminating wall). Read the MAKEFILE for this one, not the python.
    "FPL26_ILS_MEASURED_PRIORS",
    # aug05, REVIEWED (pin-drift correction — these three have SHIPPED since
    # v3.0/v3.2 with the same python-OFF/Makefile-ON pattern; this file was
    # never updated, so both pin tests were failing at tag v3.3 already):
    "FPL26_RECIPE_PASS",
    "FPL26_RECIPE_FIRST_DEEP",
    "FPL26_FIR_SUBBAND_FLOOR",
    # aug05, REVIEWED (v40-levers): the deliberate pattern for all v4.0
    # levers — python default OFF (never-touch-validated-paths), Makefile
    # arms both launch branches. STAGING until PREREG_V40_LEVERS_aug05
    # A/B + full-16 parity pass; the Makefile is the arming surface.
    "FPL26_VEX2_RETIME_CANDIDATE",
    "FPL26_MINIISP_RETRY_HOLD",
    "FPL26_CORESCORE_ROUTE_RUNG",
    # aug05, REVIEWED (v41-eggs rev2): same deliberate python-OFF /
    # Makefile-ON staging pattern as the v4.0 levers above.
    "FPL26_OWNFRONT_RETIME_CANDIDATE",
    "FPL26_MUX_MD5_TRUST",
    # aug06, REVIEWED (v41-eggs v4.1.2): same python-OFF / Makefile-ON
    # staging pattern — read the Makefile for this one, not the python.
    "FPL26_SPAM_DETERMINIZER_CANDIDATE",
    # aug07, REVIEWED (v5.0): same python-OFF / Makefile-ON pattern. The python
    # default stays OFF so a bare import can never change behaviour; the
    # Makefile is the arming surface. Rationale in the EXPECTED_SHIPS_ON entry.
    "FPL26_POSITIVE_SLACK_CONTINUE",
}


class ShipConfigTests(unittest.TestCase):
    def test_ships_on_set_is_exactly_as_pinned(self):
        eff = ship_config.effective(REPO)
        actual = {n for n, v in eff.items() if v}
        added = actual - EXPECTED_SHIPS_ON
        removed = EXPECTED_SHIPS_ON - actual
        self.assertFalse(
            added or removed,
            "EFFECTIVE ship config changed.\n"
            f"  newly SHIPPING (not reviewed):     {sorted(added)}\n"
            f"  no longer shipping (silent loss):  {sorted(removed)}\n"
            "If deliberate, update EXPECTED_SHIPS_ON in this file in the same "
            "commit — that edit is the ship decision.",
        )

    def test_flags_held_off_are_still_off(self):
        eff = ship_config.effective(REPO)
        leaked = sorted(n for n in EXPECTED_NOTABLE_OFF if eff.get(n))
        self.assertFalse(
            leaked,
            "Flags we decided NOT to ship are now on the eval path: "
            f"{leaked}. Each was held off for a measured reason (see the "
            "comments in EXPECTED_NOTABLE_OFF).",
        )

    def test_cli_optins_unchanged(self):
        self.assertEqual(
            ship_config.cli_optins(REPO), EXPECTED_CLI_OPTINS,
            "The set of CLI opt-ins in the run_optimizer recipe changed. A flag "
            "moving in or out of this set changes what the eval box runs.",
        )

    def test_no_unreviewed_layer_disagreement(self):
        actual = set(ship_config.layer_disagreements(REPO))
        new = actual - EXPECTED_DISAGREEMENTS
        gone = EXPECTED_DISAGREEMENTS - actual
        self.assertFalse(
            new or gone,
            "python-vs-Makefile default disagreements changed.\n"
            f"  NEW (unreviewed — this is the jul30 bug class): {sorted(new)}\n"
            f"  resolved: {sorted(gone)}\n"
            "A disagreement is allowed, but it must be listed deliberately so "
            "nobody trusts the python default for these names.",
        )

    def test_wall_handback_now_ships_and_can_still_be_disabled(self):
        """It left EXPECTED_CLI_OPTINS by being promoted, not by being dropped.

        Removing an entry from that dict is ambiguous on its own — the flag could
        have been deleted from the recipe entirely. Pin the two facts that
        distinguish promotion from deletion: the flag is still THERE, and it is
        gated on the opt-OUT form rather than the opt-IN form.
        """
        recipe = ship_config._recipe(REPO)
        self.assertIn("--wall-handback", recipe,
                      "--wall-handback vanished from the recipe — that is a silent "
                      "LOSS of a shipped lever, not a promotion")
        self.assertIn("$(if $(filter 0 false no off,$(WALL_HANDBACK)),,--wall-handback)",
                      recipe,
                      "--wall-handback is present but not on the opt-OUT form, so it "
                      "is no longer default-ON")
        self.assertNotIn("$(if $(filter 1 true yes on,$(WALL_HANDBACK)),--wall-handback)",
                         recipe, "the old opt-IN gate is still present")

    def test_extractor_finds_the_real_recipe(self):
        """Guard the guard: if the Makefile target is renamed, every assertion
        above would silently fall back to scanning the whole file and could pass
        vacuously."""
        inj = ship_config.makefile_injections(REPO)
        self.assertIn(
            "FPL26_ILS_HURDLE_CONTINUE", inj,
            "Did not find the expected FPL26_* injections in the run_optimizer "
            "recipe — the extractor is probably parsing the wrong target.",
        )
        self.assertTrue(ship_config.cli_optins(REPO), "no CLI opt-ins parsed")


if __name__ == "__main__":
    unittest.main()
