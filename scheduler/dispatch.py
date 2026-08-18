"""
Per-design dispatch — pick a candidate ordering tuned to each known
design.  For designs we have campaign data on, the table reflects the
empirical winner; for unknown designs (e.g. the contest's hidden
benchmark), we fall back to a generic ordering tuned by design
fingerprint (LUT count, spread).

The table is a *hint*, not a hard constraint.  The scheduler still runs
all candidates in the suggested order within the wall budget.  The
purpose is to put the *most likely winner* first so that if budget
forces a one-candidate run, we get the right one.

Empirical evidence (all 10 portfolio designs, AWS-validated, per
SELECTION.md in seeds_20260510_102618/merged_portfolio/):

  design                    winner      ΔFmax   loss-vs-other-cand
  ---                       ---         ---     ---
  amd_mini-isp              anchor      +87.35  v0_3 was +68.25 (anchor wins by +19.10)
  rosetta_optical-flow      anchor      +44.80  v0_3 was +21.85 (anchor wins by +22.95)
  vexriscv_re-place         tie         +125.37 anchor == v0_3 (deterministic strategy)
  rosetta_spam-filter       v0_3        +82.58  anchor was +2.70 (v0_3 wins by +79.88)
  logicnets_jscl            v0_3 (fc=1) +69.04  anchor was +9.50 (v0_3 wins by +59.54)
  rosetta_digit-recognition v0_3 (fc=1) +29.07  anchor was +0.00 (v0_3 wins by +29.07)
  vtr_mcml                  v0_3        +13.89  anchor was +13.22 (v0_3 wins by +0.67)
  rosetta_3d-rendering      v0_3_seed3  +7.08   anchor was +3.80, v0_3 +5.93 (seed3 wins narrowly)
  finn_radioml              v0_3_seed2  +39.04  anchor was +34.08, v0_3 +30.96 (seed2 wins, fc=3)
  corescore_500_mod         v0_3 (fc=3) +53.38  anchor was +46.39 (v0_3 wins by +6.99)

Pattern: anchor wins on small + low-spread designs (LUT < 5k AND spread < 30
empirically).  Everything else: v0_3 wins, often by a wide margin when
force-continue fires.  vexriscv_re-place is the boundary case — both
candidates lock onto the same deterministic strategy.

We deliberately keep the table SHALLOW (per CLAUDE.md P2: simplicity first).
Six rows is enough to reflect what we know without overfitting to the campaign.
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
    """Heuristic ordering for designs not in KNOWN_DESIGNS.

    Empirical anchor-favouring rule (from the 10-design table above):
        LUT < 5k AND spread < 30  →  ['anchor', 'v0_3']
    Otherwise:
        ['v0_3', 'anchor']

    The cluster boundary (5k, 30) is approximate; vexriscv_re-place
    (2k LUT, spread 63) ties either way.  When we have no fingerprint
    data at all, default to v0_3-first (matches the global best
    ordering at the aggregate level — see scheduler/aggregate.py).
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


# Recipe applicability per the 9-design AWS sweep on 2026-05-10 (RECIPE_SWEEP_2026-05-10.md).
# `applicable`  : recipe ran clean, par_routed=true, ΔFmax > 0 (or known small ΔFmax) — safe to run
# `harmful`     : recipe moved cells but route_design left routing errors — would zero our score
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
    """Best-effort: derive a design name from a DCP filename like
    `amd_mini-isp_2025.1.dcp` → `amd_mini-isp`.  Returns None if we can't.
    """
    import re
    from pathlib import Path
    stem = Path(dcp_path).stem
    # Strip a Vivado version suffix `_2025.1` or any `_v\d+\.\d+` pattern.
    m = re.match(r"^(.+?)(?:_\d{4}\.\d+)?$", stem)
    if m:
        return m.group(1)
    return None
