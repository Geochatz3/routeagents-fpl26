"""Provides optional prompt hints that steer a model toward registered recipe
paths.

Hints are advisory and never bypass safety, budget, or recipe-applicability
checks. The registry is empty by default and can be populated through
`RECIPE_HINTS`.

Set `ENABLE_CROSS_MODEL_STEERING=1` to enable injection. Injected text is
logged through the `cross_model_steering` logger at INFO level for auditing.
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

    Fields should be evidence-supported by the prior model's logs; do NOT
    add aspirational moves the prior model didn't actually take.
    """
    design_name: str
    preferred_first_action: str  # exact tool + args or short recipe phrase
    rationale: str               # what the prior model did + measured outcome
    avoid: str                   # the trap the new model fell into post-migration
    historical_target_fmax_mhz: float  # prior-model measured ΔFmax — for gap math


# Registry of per-design steering hints.  Shipped EMPTY in the public
# tree — the development campaign's entries were derived from internal
# forensic log comparisons and keyed by benchmark name; per-benchmark
# steering data is excluded from this release.  Populate with your own
# RecipeHint entries (keyed by design name) to use the mechanism.
RECIPE_HINTS: dict[str, RecipeHint] = {}


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
        f"prior-model result on '{design_name}' was +{target:.2f} MHz "
        f"({100*fraction_recovered:.0f}% recovered), and {remaining_seconds:.0f}s "
        f"of wall budget remain.\n"
        f"Suggested next move (from prior-model log): "
        f"{hint.preferred_first_action}.\n"
        f"Why: {hint.rationale}\n"
        f"Avoid: {hint.avoid}\n"
        f"If you truly judge no further safe improvement possible, you may "
        f"stop again on the next turn — but try the suggested move first."
    )
