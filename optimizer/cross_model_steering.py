"""Cross-model steering hints — replay grok-4.1-fast winning paths under grok-4.3.

Background: pre-migration grok-4.1-fast portfolio earned +79.92 MHz across 4
designs. Post-migration grok-4.3 rebuild delivers only ~+16 MHz because
grok-4.3 picks different first-iteration recipes and exits earlier. Forensic
log comparison (see `.planning/beta/grok43_recovery_report.md`) traces every
lost MHz to a specific tool the new model did not invoke.

This module supplies compact, auditable steering hints that nudge grok-4.3
toward the historical winning recipe on a known design. Hints are advisory
text injected into the user prompt — they do NOT override safety, budget,
or recipe-applicability checks. They never claim a specific final MHz as
guaranteed; they only describe what the prior model did and why.

Enable: set environment variable ENABLE_CROSS_MODEL_STEERING=1. Disabled by
default (no behavior change), so this module is opt-in.

All injected text is logged via the `cross_model_steering` logger at INFO so
runs can be audited post-hoc.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

ENV_FLAG = "ENABLE_CROSS_MODEL_STEERING"


@dataclass(frozen=True)
class RecipeHint:
    """A compact per-design steering hint.

    Fields are evidence-supported by the grok-4.1-fast log; do NOT add
    aspirational moves the prior model didn't actually take.
    """
    design_name: str
    preferred_first_action: str  # exact tool + args or short recipe phrase
    rationale: str               # what grok-4.1-fast did + measured outcome
    avoid: str                   # the trap grok-4.3 fell into post-migration
    historical_target_fmax_mhz: float  # the grok-4.1-fast ΔFmax — for gap math


# Registry. Each entry was derived from the forensic table in
# .planning/beta/grok43_recovery_report.md section 2. Numbers are the
# pre-migration measured ΔFmax, NOT a guarantee of reproduction.
RECIPE_HINTS: dict[str, RecipeHint] = {
    "ispd16_example2": RecipeHint(
        design_name="ispd16_example2",
        preferred_first_action=(
            "recipe_post_route_phys_opt_sweep (it internally invokes "
            "vivado_phys_opt_design -critical_cell_opt which committed +1.421 ns "
            "in the prior model run)"
        ),
        rationale=(
            "Pre-migration grok-4.1-fast ran recipe_post_route_phys_opt_sweep on "
            "this exact design and committed the critical_cell_opt sub-flag in "
            "~34 minutes for +19.44 MHz (-7.752 ns → -6.331 ns). This was the "
            "single winning move on ispd16 — every other recipe stalled or "
            "exhausted budget."
        ),
        avoid=(
            "recipe_cell_replacement — the rapidwright_analyze_net_detour step "
            "took 29 minutes on ispd16 in the grok-4.3 trial and starved the "
            "subsequent route_design (which timed out at 740s of 687s "
            "allowance). Net result: 0 MHz."
        ),
        historical_target_fmax_mhz=19.44,
    ),
    "boom_soc": RecipeHint(
        design_name="boom_soc",
        preferred_first_action=(
            "vivado_phys_opt_design with directive=AlternateFlowWithRetiming "
            "(full design scope — NOT recipe_critical_path_focused_phys_opt)"
        ),
        rationale=(
            "Pre-migration grok-4.1-fast committed +1.165 ns (+2.87 MHz) on "
            "boom_soc with a single phys_opt_design -directive "
            "AlternateFlowWithRetiming call (~40 min). Boom_soc has 217,988 "
            "failing endpoints — scoping to 20 endpoints only addresses ~0.01% "
            "of the failing set."
        ),
        avoid=(
            "recipe_critical_path_focused_phys_opt (scoped to N=20 paths). It "
            "ran in 5.7 min on grok-4.3 but delivered only +0.082 MHz — a 35× "
            "under-baseline result. The scoped recipe is mechanically correct "
            "but the wrong tool for a uniformly-slow design."
        ),
        historical_target_fmax_mhz=2.87,
    ),
    "finn_radioml": RecipeHint(
        design_name="finn_radioml",
        preferred_first_action=(
            "after phys_opt directives plateau, run vivado_run_tcl "
            "'place_design -unplace' then vivado_place_design "
            "directive=Explore (~9 min), then vivado_route_design "
            "directive=Default (~6 min), then vivado_phys_opt_design "
            "directive=AlternateFlowWithRetiming (~1 min) — this Class G "
            "sequence is the +47 MHz move. NOTE on variance: Vivado's "
            "place_design Explore is non-deterministic in interactive "
            "mode (no -seed knob); observed grok-4.3+steering runs landed "
            "+59.10 and +45.46 MHz (spread 13.6 MHz). Commit any positive "
            "result from one Class G attempt; do NOT keep re-running it "
            "to chase variance — wall budget is more valuable than the "
            "marginal MHz from a re-roll."
        ),
        rationale=(
            "Pre-migration grok-4.1-fast went from WNS -1.910 → -1.373 (+51.46 "
            "MHz total). The dominant single contribution was a full "
            "place_design Explore (~9 min) + route Default (~6 min) + "
            "phys_opt_design AlternateFlowWithRetiming (~1 min). The grok-4.3 "
            "rebuild stopped at -1.724 (+15.94 MHz) without ever invoking "
            "place_design Explore — it stuck with phys_opt directive variants "
            "(which can only refine existing placement, not re-place)."
        ),
        avoid=(
            "(1) declaring optimization complete while wall budget > 15 min "
            "and place_design -directive Explore has not been attempted; "
            "the grok-4.3 trial emitted 5 consecutive finish_reason='stop' "
            "messages in 80 seconds with 23 min of budget remaining. "
            "(2) re-running place_design Explore multiple times to chase "
            "variance — Vivado nondeterminism means a re-roll can land WORSE "
            "than the first attempt; one Class G + retime is the budget-"
            "efficient call."
        ),
        historical_target_fmax_mhz=51.46,
    ),
    "corescore_500_mod": RecipeHint(
        design_name="corescore_500_mod",
        preferred_first_action=(
            "the prior stable safety path is a 4-step sequence: "
            "(1) recipe_register_retiming AlternateFlowWithRetiming (~6 min, "
            "~+2.27 MHz), (2) vivado_run_tcl 'place_design -unplace' then "
            "vivado_place_design directive=Auto_1 (~12 min, deterministic, "
            "~+3 MHz from re-placement), (3) recipe_register_retiming "
            "AlternateFlowWithRetiming again (~3 min, ~+0.86 MHz), "
            "(4) vivado_phys_opt_design critical_pin_opt (~1 min, polish). "
            "If step 1 commits but you are still below +6.15 MHz total and "
            "budget remains, do NOT stop — try step 2 next"
        ),
        rationale=(
            "Pre-migration grok-4.1-fast reached +6.15 MHz on corescore by "
            "running the 4-step safety sequence above. The grok-4.3 cross-"
            "model run completed step 1 (+2.27 MHz) then stopped, missing "
            "the ~+3 MHz from place_design Auto_1. Auto_1 is the lighter "
            "placement directive (deterministic, ~12 min) — distinct from "
            "the heavier Explore directive used on finn."
        ),
        avoid=(
            "chasing the +70.88 MHz path via place_design -directive Explore "
            "(Class G) — that result is UNSTABLE RESEARCH; one trial got "
            "+70.88 MHz, the reproducibility trial collapsed to 0 because "
            "Vivado runtime variance pushed Explore past the 50-min cap. "
            "DO NOT prioritize Explore on corescore — Auto_1 first, always. "
            "Only consider Explore after the safety path (+6.15) is "
            "already shipped."
        ),
        historical_target_fmax_mhz=6.15,
    ),
}


def is_steering_enabled() -> bool:
    """Returns True iff ENABLE_CROSS_MODEL_STEERING=1 in the environment.

    Default is False — steering is opt-in. Any value other than "1" (case-
    insensitive) is treated as disabled so a stray "true"/"yes" doesn't
    accidentally enable it.
    """
    raw = os.environ.get(ENV_FLAG, "")
    return raw.strip() == "1"


def get_iter1_hint(design_name: Optional[str]) -> Optional[str]:
    """Return formatted iter-1 hint text for the user message, or None.

    Returns None when:
      - steering is disabled,
      - design_name is None or not in the registry,
      - the registry entry is empty.

    Caller appends the returned string to the Phase 1 / iter-1 user prompt.
    Result is also logged at INFO so audits can confirm the hint reached
    the LLM.
    """
    if not is_steering_enabled():
        return None
    if not design_name:
        return None
    hint = RECIPE_HINTS.get(design_name)
    if hint is None:
        return None

    text = (
        "\nCROSS-MODEL STEERING HINT (advisory; from prior-model run on the "
        "same design — gated by ENABLE_CROSS_MODEL_STEERING=1):\n"
        f"  Design: {hint.design_name}\n"
        f"  Historical pre-migration ΔFmax: +{hint.historical_target_fmax_mhz} MHz\n"
        f"  Preferred first action: {hint.preferred_first_action}\n"
        f"  Why: {hint.rationale}\n"
        f"  Avoid: {hint.avoid}\n"
        "  Note: the historical number is the prior-model outcome, NOT a "
        "guarantee. Use the hint as a starting bias; still respect safety, "
        "budget, and recipe-applicability rules. If you find a strictly "
        "better path, take it."
    )
    logger.info(
        "cross_model_steering: iter-1 hint injected for design='%s' "
        "(target=+%.2f MHz, preferred='%s')",
        hint.design_name,
        hint.historical_target_fmax_mhz,
        hint.preferred_first_action.split("(")[0].strip(),
    )
    return text


def get_continue_hint(
    design_name: Optional[str],
    current_fmax_gain_mhz: float,
    remaining_seconds: float,
) -> Optional[str]:
    """Return an anti-premature-exit continuation message, or None.

    Fires only when ALL conditions hold:
      - steering is enabled
      - design has a historical-target entry
      - current_fmax_gain < 50% of historical target
      - remaining_seconds > 900 (i.e., > 15 min budget left)

    Caller injects this in the force-continue branch of the iteration
    loop (replacing the generic BETA-CTRL-V0 message when a specific hint
    is available).

    Logs at INFO when fired or skipped — so audits can see whether the
    override actually triggered.
    """
    if not is_steering_enabled():
        return None
    if not design_name:
        return None
    hint = RECIPE_HINTS.get(design_name)
    if hint is None:
        return None

    target = hint.historical_target_fmax_mhz
    if target <= 0:
        return None  # can't compute a sensible "fraction recovered"
    fraction_recovered = current_fmax_gain_mhz / target

    if fraction_recovered >= 0.5:
        logger.info(
            "cross_model_steering: continue-hint SKIPPED for '%s' "
            "(current +%.2f MHz / target +%.2f MHz = %.0f%% recovered; "
            "above 50%% threshold)",
            design_name,
            current_fmax_gain_mhz,
            target,
            100 * fraction_recovered,
        )
        return None

    if remaining_seconds <= 900:
        logger.info(
            "cross_model_steering: continue-hint SKIPPED for '%s' "
            "(remaining %.0fs <= 900s threshold; honoring stop)",
            design_name,
            remaining_seconds,
        )
        return None

    logger.info(
        "cross_model_steering: anti-premature-exit override TRIGGERED for "
        "'%s' — current +%.2f MHz vs historical +%.2f MHz "
        "(%.0f%% recovered), remaining %.0fs",
        design_name,
        current_fmax_gain_mhz,
        target,
        100 * fraction_recovered,
        remaining_seconds,
    )
    return (
        f"ANTI-PREMATURE-EXIT (cross-model steering, gated by "
        f"ENABLE_CROSS_MODEL_STEERING=1): you signalled completion at "
        f"+{current_fmax_gain_mhz:.2f} MHz, but the pre-migration "
        f"grok-4.1-fast result on '{design_name}' was +{target:.2f} MHz "
        f"({100*fraction_recovered:.0f}% recovered), and {remaining_seconds:.0f}s "
        f"of wall budget remain.\n"
        f"Suggested next move (from prior-model log): "
        f"{hint.preferred_first_action}.\n"
        f"Why: {hint.rationale}\n"
        f"Avoid: {hint.avoid}\n"
        f"If you truly judge no further safe improvement possible, you may "
        f"stop again on the next turn — but try the suggested move first."
    )
