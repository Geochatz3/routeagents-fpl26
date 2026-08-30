"""Pin the effective runtime configuration against silent drift.

Assertions compare exact flag sets rather than subsets so both additions and
removals fail. Intentional changes require updating these expectations
alongside the configuration, making runtime feature decisions explicit in
review.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import ship_config  # noqa: E402


# What a clean eval box gets with NO FPL26_*/make variables set.
# Every entry here is a deliberate ship decision, not an accident of layering.
EXPECTED_SHIPS_ON = {
    # --- the uniform ILS stack -------
    "FPL26_ILS_PLACE_RETRY_LADDER",
    # Skipping unaffordable ILS steps is enabled as an explicit risk
    # tradeoff; timing variation within measurement noise is accepted.
    "FPL26_ILS_LADDER_SKIP_UNAFFORDABLE",
    "FPL26_ILS_MEASURED_BASIS",
    "FPL26_ILS_INCR_ROUTE",
    "FPL26_ILS_INCR_ROUTE_FIRST",
    # --- deep-replace family ------------------------------------------------
    "FPL26_DEEP_REPLACE",
    "FPL26_DEEP_REPLACE_FIRST",
    "FPL26_DEEP_REPLACE_FIRST_BANDED",
    # For small mid-band designs, the floor caps the anchor at 300 s and offers
    # an insurance MUX candidate only after preregistration.
    "FPL26_DEEP_REPLACE_B3",
    # Logic-floor exit requires a timing-report round trip, a bound at most
    # 10 MHz across 32 paths at 45 ps/hop, at least 80% logic and 50%
    # hard-macro delay, and agreement between two solves. Any failed guard
    # prevents the exit.
    "FPL26_LOGIC_FLOOR_EXIT",
    # The b3_floor_saturated.token sentinel terminates the attempt loop and
    # skips winner polishing after an attempt attests the B3 floor.
    "FPL26_B3_FLOOR_EXIT",
    # The packaged Makefile enables this flag although its implementation
    # documentation describes it as default-off. This test pins the
    # packaged configuration.
    "FPL26_DEEP_REPLACE_UNBANDED",
    # For designs above ILS's 300k-cell limit, deep replacement may run first
    # without the deep-extreme band or double-subtracted finalize reserve.
    # Results are adopted only on strict improvement; failure preserves the
    # checkpoint but consumes wall time. Disable with
    # FPL26_NO_DEEP_FIRST_SIZEGATED=1 or DEEP_FIRST_SIZEGATED=0.
    "FPL26_DEEP_FIRST_SIZEGATED",
    # --- loop economics -----------------------------------------------------
    "FPL26_ILS_HURDLE_CONTINUE",
    "FPL26_PREEMPT_LOOP_CLOCK",
    # --- hygiene ------------------------------------------------------------
    "FPL26_VIVADO_LEAK_FIX",
    # ILS seeds from a copy of the banked best-valid checkpoint, never the
    # mirror itself. Accepted moves may overwrite the seed, while mirror-size
    # metadata updates only after the stage. An interruption before that update
    # otherwise invalidates the mirror and falls back to baseline.
    "FPL26_ILS_SEED_COPY",
    # Runs the second deep-replace pass in a fresh tool process after very long
    # place-and-route runs. The 1200 s threshold limits restart overhead to
    # designs most exposed to process instability. The first pass is
    # checkpointed before restart, so the restart can cost time but not the
    # result.
    "FPL26_DEEP_REPLACE_B2_RESTART",
    # Uses measured cycle costs to select iterations that fit the remaining wall budget.
    # Cost-aware scheduling substitutes planned cycles rather than adding work.
    "FPL26_ILS_MEASURED_PRIORS",
    # Suppresses the expensive high-delay arm only when both conditions hold:
    # failing/spread < 363 and estimated cycle time / remaining wall time >=
    # 0.60. Missing, non-finite, or unpriceable inputs fail open and leave the
    # arm enabled. The thresholds target sparse failing sets when one cycle
    # would consume most of the window.
    "FPL26_ILS_ENDHIGH_DENSITY_GATE",
    # Applies WNS-based ladder ordering only within the existing shallow cutoff, |WNS| <= 0.7 ns.
    # The cutoff reuses route_reroll_max_wns_mag rather than introducing another tuning constant.
    "FPL26_ILS_LADDER_ORDER_BY_WNS",
    "FPL26_RECIPE_PASS",
    # Cell count gates the pre-LLM deep path.
    "FPL26_RECIPE_FIRST_DEEP",
    # A shallow failing sub-band uses an in-session floor with a carve-out.
    "FPL26_SUBBAND_PHYSOPT_FLOOR",
    # --- Retime/rung levers — pinned as the CURRENT REALITY of this
    # branch, NOT as a passed gate. The ship decision was a 3-design A/B
    # plus full-16 parity; this list only records what is armed.
    "FPL26_ETO_RETIME_CANDIDATE",
    "FPL26_MIDBAND_RETRY_HOLD",
    "FPL26_MIDBAND_ROUTE_RUNG",
    "FPL26_OWNFRONT_RETIME_CANDIDATE",
    "FPL26_MUX_MD5_TRUST",
    # The shallow determinizer candidate applies only when |WNS| is in [0.60, 0.90) ns.
    # `SHALLOW_DET=0` disables it.
    "FPL26_SHALLOW_DETERMINIZER_CANDIDATE",
    # Allows optimization to continue after timing is met, both at entry and
    # after a zero crossing. The objective uses unclamped Fmax = 1000/(T - WNS)
    # MHz for T and WNS in ns. Cost, plateau, empty-spin, iteration, and wall-
    # deadline guards still bound execution.
    "FPL26_POSITIVE_SLACK_CONTINUE",
}

# Flags measured but deliberately NOT shipping. Each carries the reason, because
# "why is this off" is the question a future reader will actually have.
EXPECTED_NOTABLE_OFF = {
    "FPL26_ILS_LADDER_RESERVE",
    "FPL26_ILS_LADDER_STOP_ON_ACCEPT",
    "FPL26_ILS_RETRY_BASELINE_GATE",
    "FPL26_ILS_REJECT_MICRO_ACCEPT",
    "FPL26_ILS_INCR_ROUTE_TERMINAL",
    # Both features remain opt-in; this set detects accidental default-state drift.
    # The pre-loop plateau feature handles a best checkpoint banked before the loop
    # and must not change the selected result.
    "FPL26_PLATEAU_PRELOOP_FIX",
    #   DEEP_REPLACE_NO_DOUBLE_RESERVE: stops the affordability gate
    #   re-subtracting the finalize reserve already removed from the deadline.
    #   22/30 corpus declines arm once corrected.
    "FPL26_DEEP_REPLACE_NO_DOUBLE_RESERVE",
    # The MCP server drops `path_groups` whenever a directive is set, so Default
    # directive calls run full-design phys_opt rather than critical-path-targeted phys_opt.
    "FPL26_PHYSOPT_DEFAULT_FIXPOINT",
}

# CLI opt-ins are passed only when the corresponding make variable is set.
# Wall handback is default-on and is therefore excluded from the opt-in set.
# `WALL_HANDBACK=0`, `false`, `no`, or `off` disables it.
EXPECTED_CLI_OPTINS = {
    "SPLIT_AWARE": "--split-aware",
}

# Layer disagreements are ALLOWED but must be reviewed. The Makefile is the ship
# decision; the risk is only that reading the python default misleads you. A NEW
# disagreement appearing here unreviewed is the bug recurring.
EXPECTED_DISAGREEMENTS = {
    "FPL26_DEEP_REPLACE",
    "FPL26_DEEP_REPLACE_FIRST",
    "FPL26_DEEP_REPLACE_UNBANDED",
    # REVIEWED: same deliberate pattern as the v4.0 levers below —
    # python default OFF (never-touch-validated-paths), Makefile arms both
    # launch branches. See the EXPECTED_SHIPS_ON entry for the measurements.
    "FPL26_DEEP_FIRST_SIZEGATED",
    # These features default off in Python but are intentionally armed by the Makefile.
    "FPL26_DEEP_REPLACE_B3",
    # REVIEWED (v5.4): same python-OFF / Makefile-ON pattern for the
    # B3-floor-exit wrapper behavior, under the same pre-registered gate.
    "FPL26_B3_FLOOR_EXIT",
    # REVIEWED (v5.5): same pattern for the logic-floor attestation.
    "FPL26_LOGIC_FLOOR_EXIT",
    "FPL26_ILS_HURDLE_CONTINUE",
    # REVIEWED: python default 0 (house rule: a new lever ships OFF until
    # validated), Makefile default 1 (it IS now validated -- 2/2 at the
    # discriminating wall). Read the MAKEFILE for this one, not the python.
    "FPL26_ILS_MEASURED_PRIORS",
    # REVIEWED (pin-drift correction — these three have SHIPPED
    # v3.0/v3.2 with the same python-OFF/Makefile-ON pattern; this file was
    # never updated, so both pin tests were failing at tag v3.3 already):
    "FPL26_RECIPE_PASS",
    "FPL26_RECIPE_FIRST_DEEP",
    "FPL26_SUBBAND_PHYSOPT_FLOOR",
    # These staged features default off in Python; both Makefile launch
    # branches arm them.
    "FPL26_ETO_RETIME_CANDIDATE",
    "FPL26_MIDBAND_RETRY_HOLD",
    "FPL26_MIDBAND_ROUTE_RUNG",
    # REVIEWED (v41-eggs rev2): same deliberate python-OFF /
    # Makefile-ON staging pattern as the v4.0 levers above.
    "FPL26_OWNFRONT_RETIME_CANDIDATE",
    "FPL26_MUX_MD5_TRUST",
    # REVIEWED (v41-eggs v4.1.2): same python-OFF / Makefile-ON
    # staging pattern — read the Makefile for this one, not the python.
    "FPL26_SHALLOW_DETERMINIZER_CANDIDATE",
    # REVIEWED (v5.0): same python-OFF / Makefile-ON pattern. The python
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
            f"  NEW (unreviewed — this is the bug class): {sorted(new)}\n"
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

    def test_numeric_injections_are_not_reported_as_off(self):
        """A Makefile injection carrying a number is armed, not off.

        `is_on` only recognises the boolean spellings, so a numeric setting
        used to land in the "ships off" list — which reads as "this mechanism
        is inactive" for a value that is very much active.
        """
        valued = ship_config.valued_injections(REPO)
        self.assertIn(
            "FPL26_DEEP_WNS_TAIL_RESERVE", valued,
            "the tail reserve is injected with a numeric value and must be "
            "reported as armed, not as off",
        )
        self.assertTrue(valued["FPL26_DEEP_WNS_TAIL_RESERVE"].strip())
        eff = ship_config.effective(REPO)
        for name in valued:
            self.assertFalse(
                eff.get(name, False),
                f"{name} carries a value, so the boolean view should not "
                f"claim it is on either — it belongs in the valued list",
            )

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


class CliRootTests(unittest.TestCase):
    """`--root` is what makes the provenance claim checkable: it reads the
    configuration off another checkout, so the scored tag and `main` can be
    compared with one command. A silently-ignored argument would make that
    comparison read as a match no matter what the other tree said."""

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, str(REPO / "scripts" / "ship_config.py"), *args],
            capture_output=True, text=True, check=True,
        ).stdout

    def test_root_reads_the_tree_it_is_given(self):
        here = self._run()
        with tempfile.TemporaryDirectory() as tmp:
            other = Path(tmp) / "tree"
            (other / "scripts").mkdir(parents=True)
            (other / "Makefile").write_text(
                "run_optimizer:\n"
                "\tFPL26_ONLY_IN_THE_OTHER_TREE=$(if $(X),$(X),1) true\n"
                "\nother_target:\n\t@true\n"
            )
            elsewhere = self._run("--root", str(other))
        self.assertNotEqual(
            here, elsewhere,
            "--root produced this tree's configuration, so the argument is "
            "being ignored and any two trees would compare equal",
        )
        self.assertIn("FPL26_ONLY_IN_THE_OTHER_TREE", elsewhere)

    def test_no_root_is_this_tree(self):
        self.assertIn("FPL26_ILS_HURDLE_CONTINUE", self._run())


if __name__ == "__main__":
    unittest.main()
