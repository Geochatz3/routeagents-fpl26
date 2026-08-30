"""Per-design dispatch — pick a candidate ordering suited to each design.

Where campaign data exists, the table reflects the empirical winner; for an
unseen design it falls back to a generic ordering chosen by fingerprint (LUT
count and critical-path spread).

The table is a HINT, not a hard constraint.  The scheduler still runs every
candidate in the suggested order within the wall budget.  The point is to put
the most likely winner first, so that a budget which forces a single-candidate
run gets the right one.

The observed pattern is that the anchor controller wins on small, low-spread
designs, and the full controller wins everywhere else — often by a wide margin
where its force-continue behaviour fires.  Designs on the boundary tend to lock
both controllers onto the same deterministic strategy, so the ordering there
does not matter.

The table is kept deliberately shallow: enough rows to reflect what is known,
few enough not to overfit the campaign it came from.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


# Per-known-design candidate priority (highest-confidence winner first).
# Unknown designs fall through to fingerprint heuristics below.
KNOWN_DESIGNS = {
    "amd_mini-isp":              ["anchor", "v0_3"],
    "rosetta_optical-flow":      ["anchor", "v0_3"],
    "vexriscv_re-place":         ["anchor", "v0_3"],   # tie; anchor faster
    "rosetta_spam-filter":       ["v0_3", "anchor"],
    "logicnets_jscl":            ["v0_3", "anchor"],
    "rosetta_digit-recognition": ["v0_3", "anchor"],
    "vtr_mcml":                  ["v0_3", "anchor"],
    "rosetta_3d-rendering":      ["v0_3", "anchor"],
    "finn_radioml":              ["v0_3", "anchor"],
    "corescore_500_mod":         ["v0_3", "anchor"],
    # Hidden/deferred:
    "vexriscv_re-place_v2":      ["v0_3", "anchor"],   # no good signal yet
    "boom_soc":                  ["v0_3", "anchor"],   # never completed locally
    "ispd16_example2":           ["v0_3", "anchor"],   # never completed locally
}


@dataclass
class DesignFingerprint:
    """Coarse design metrics used for fallback candidate-ordering when the
    design name is unknown (e.g. a new hidden benchmark)."""
    lut_count: Optional[int] = None
    critical_path_spread: Optional[float] = None
    initial_fmax_mhz: Optional[float] = None


def fingerprint_for_unknown(fp: DesignFingerprint) -> List[str]:
    """Order candidates heuristically when no known-input dispatch entry exists.

    Inputs with fewer than 5,000 LUTs and spread below 30 try ``anchor`` before
    ``v0_3``; all others reverse that order. The thresholds define the
    dispatcher's compact, low-spread region. If fingerprint data is
    unavailable, ``v0_3`` runs first.
    """
    if fp.lut_count is not None and fp.critical_path_spread is not None:
        if fp.lut_count < 5000 and fp.critical_path_spread < 30:
            return ["anchor", "v0_3"]
    return ["v0_3", "anchor"]


def candidates_for(
    design_name: Optional[str],
    fingerprint: Optional[DesignFingerprint] = None,
) -> List[str]:
    """Return the candidate ordering for a given design.

    Lookup order:
      1. KNOWN_DESIGNS (exact name match) — empirical, highest confidence
      2. fingerprint_for_unknown — fallback based on coarse metrics
      3. global default ['v0_3', 'anchor'] — matches the best aggregate
         ordering when no other information is available
    """
    if design_name in KNOWN_DESIGNS:
        return KNOWN_DESIGNS[design_name]
    if fingerprint is not None:
        return fingerprint_for_unknown(fingerprint)
    return ["v0_3", "anchor"]


def candidates_with_recipe(
    design_name: Optional[str],
    fingerprint: Optional[DesignFingerprint] = None,
) -> List[str]:
    """Like candidates_for, but prepends 'recipe' on designs the
    recipe-applicability table marks as `applicable`.

    `no_op` and `unknown` designs deliberately DO NOT get a recipe slot —
    the recipe takes 3-5 min on average and would burn budget for no
    likely gain (it short-circuits cleanly but the wall is wasted).
    The user can override via an explicit candidate list when they want
    to probe an unknown design.
    """
    base = candidates_for(design_name, fingerprint)
    verdict = RECIPE_APPLICABILITY.get(design_name)
    if verdict == "applicable":
        return ["recipe", *base]
    return base


# Recipe applicability per a 9-design AWS sweep.
# `applicable`  : recipe ran clean, par_routed=true, ΔFmax > 0 (or known small ΔFmax) — safe to run
# `harmful`     : recipe moved cells but route_design left routing errors — would zero the score
# `no_op`       : detour analysis returned no candidates above threshold — recipe runs no work
# `error`       : recipe hung or crashed (digit-recog: 40-min wall + crash) — skip until rerun
# `unknown`     : not yet swept — treat as conservative no-skip (recipe will short-circuit safely)
RECIPE_APPLICABILITY = {
    "amd_mini-isp":              "applicable",
    "logicnets_jscl":            "applicable",
    "rosetta_3d-rendering":      "applicable",
    "vexriscv_re-place":         "applicable",  # +76.67 MHz live-validated
    "corescore_500_mod":         "harmful",
    "finn_radioml":              "no_op",
    "rosetta_optical-flow":      "no_op",
    "rosetta_digit-recognition": "error",
    "rosetta_spam-filter":       "unknown",
    "vtr_mcml":                  "unknown",
    "vexriscv_re-place_v2":      "unknown",
    "boom_soc":                  "unknown",
    "ispd16_example2":           "unknown",
}


def recipe_safe_for(design_name: Optional[str]) -> bool:
    """Whether the cell-replacement recipe is known-safe to run on this design.

    Returns False for `harmful` and `error` designs (sweep-confirmed damage
    or instability).  Returns True for `applicable`, `no_op`, and `unknown` —
    the recipe's own gates (route-error check, no-candidates short-circuit)
    handle the no_op + unknown cases without writing a regression.

    Used by:
      - the LLM-side strategy hint in dcp_optimizer.py (don't suggest the
        recipe path on harmful designs)
      - any future scheduler candidate slot that runs the recipe directly
    """
    return RECIPE_APPLICABILITY.get(design_name, "unknown") not in {"harmful", "error"}


def design_name_from_dcp(dcp_path) -> Optional[str]:
    """Derive a logical design name from a checkpoint filename when possible.

    Removes a recognized version suffix and the ``.dcp`` extension while
    preserving the remaining filename stem. Returns None when the filename does
    not match the expected form.
    """
    import re
    from pathlib import Path
    stem = Path(dcp_path).stem
    # Strip a Vivado version suffix `_2025.1` or any `_v\d+\.\d+` pattern.
    m = re.match(r"^(.+?)(?:_\d{4}\.\d+)?$", stem)
    if m:
        return m.group(1)
    return None
