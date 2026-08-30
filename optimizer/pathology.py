"""Deterministic pathology classifier for FPGA critical-path data.

Pre-fix optimizer flow: LLM gets raw timing/path data → guesses at a
strategy → LLM-driven discovery of which recipe applies.  This works
when the LLM has enough context tokens to think through pathologies,
but burns LLM rounds re-deriving the same conclusion every iteration.

This module produces a **deterministic** classification of a design's
critical-path situation into labelled pathologies (PLACEMENT_DETOUR,
LUT_DEPTH, HIGH_FANOUT_DRIVER, etc.) with reasons + suggested recipes.
The output is structured JSON the LLM can feed into its recipe-
selection logic, replacing free-form guesswork with rule-based
pattern matching.

This is the "timing-onion" identification step:
  current design state
  → pathology classifier  ← this module
  → recipe selector
  → bounded transform
  → measure
  → expose next bottleneck
  → repeat

No Vivado/RapidWright calls live here — the module takes the data
the optimizer has already collected (Phase 1 analysis output, plus
optional per-path enrichments) and emits labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

# Pathology labels are stable identifiers shared by the LLM prompt, recipe
# selection, and retrieval metadata. Renaming one requires updating all three.

PLACEMENT_DETOUR = "PLACEMENT_DETOUR"
ROUTE_DETOUR = "ROUTE_DETOUR"
LUT_DEPTH = "LUT_DEPTH"
HIGH_FANOUT_DRIVER = "HIGH_FANOUT_DRIVER"
HARDBLOCK_DISTANCE = "HARDBLOCK_DISTANCE"
CONGESTED_REGION = "CONGESTED_REGION"
RETIMING_CANDIDATE = "RETIMING_CANDIDATE"
PIN_SWAP_CANDIDATE = "PIN_SWAP_CANDIDATE"
CELL_SPREAD = "CELL_SPREAD"
ROUTE_DOMINATED_NO_SAFE_MOVE = "ROUTE_DOMINATED_NO_SAFE_MOVE"
NO_CLEAR_LOCAL_RECIPE = "NO_CLEAR_LOCAL_RECIPE"

ALL_LABELS = frozenset({
    PLACEMENT_DETOUR, ROUTE_DETOUR, LUT_DEPTH, HIGH_FANOUT_DRIVER,
    HARDBLOCK_DISTANCE, CONGESTED_REGION, RETIMING_CANDIDATE,
    PIN_SWAP_CANDIDATE, CELL_SPREAD, ROUTE_DOMINATED_NO_SAFE_MOVE,
    NO_CLEAR_LOCAL_RECIPE,
})


# Detection thresholds favor sensitivity because an inapplicable recipe is
# cheaper than missing an actionable pathology.

# Detour ratio = routed_path_length / manhattan_distance.  > 2.0 means
# the router took a path more than 2x longer than the geometric minimum.
DETOUR_RATIO_HIGH = 2.0
DETOUR_RATIO_VERY_HIGH = 3.0

# Logic levels > N suggests LUT-cascade pathology.  Vivado classifies
# 6+ levels as a typical timing closure concern.
LOGIC_LEVELS_HIGH = 6
LOGIC_LEVELS_VERY_HIGH = 9

# Net fanout that qualifies as "high fanout driver" of a critical path.
HIGH_FANOUT_THRESHOLD = 100
VERY_HIGH_FANOUT_THRESHOLD = 500

# Fraction of total path delay attributed to routing vs logic.  > 0.65
# means the path is routing-bound; LUT/cell optimizations won't help.
ROUTE_DELAY_FRACTION_HIGH = 0.65
ROUTE_DELAY_FRACTION_VERY_HIGH = 0.80

# Cell spread is the average Manhattan distance between adjacent critical-path
# cells, in RapidWright tile units. The high threshold matches the design
# summary's placement-region recommendation.
CELL_SPREAD_HIGH = 70.0
CELL_SPREAD_VERY_HIGH = 150.0  # boom_soc: 302 tiles → very-high pathology

# Slack budget vs FF count — high FF count + small slack → retiming
# may move logic across registers to balance.
RETIMING_FF_MIN = 3
RETIMING_SLACK_PER_FF_NS = 0.5


# Data shapes. Callers pass plain dicts so this module imposes no dataclass
# on them; the fields documented below are the recognized ones.

@dataclass
class PathPathology:
    """A single pathology label assigned to one critical path."""
    label: str
    reason: str
    confidence: float  # 0.0 to 1.0
    suggested_recipes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DesignPathology:
    """Roll-up of all pathologies detected across a design's top paths."""
    primary_label: str
    counts: dict[str, int] = field(default_factory=dict)
    per_path: list[list[PathPathology]] = field(default_factory=list)
    suggested_recipes_ordered: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "primary_label": self.primary_label,
            "counts": self.counts,
            "per_path": [[p.to_dict() for p in path] for path in self.per_path],
            "suggested_recipes_ordered": self.suggested_recipes_ordered,
            "notes": self.notes,
        }


# Mapping: pathology → recipes that have a chance of fixing it.
# Ordered by likelihood of success.  Update as recipes are landed.

PATHOLOGY_TO_RECIPES: dict[str, tuple[str, ...]] = {
    PLACEMENT_DETOUR: (
        "recipe_cell_replacement",
        "recipe_critical_path_focused_phys_opt",  # scoped placement_opt
        "recipe_post_route_phys_opt_sweep",
    ),
    ROUTE_DETOUR: (
        "recipe_critical_path_focused_phys_opt",  # scoped first
        "recipe_post_route_phys_opt_sweep",
        "recipe_cell_replacement",
    ),
    LUT_DEPTH: (
        "recipe_lut_optimization",
        "recipe_critical_path_focused_phys_opt",  # scoped restruct_opt
        "recipe_post_route_phys_opt_sweep",
    ),
    HIGH_FANOUT_DRIVER: (
        "recipe_high_fanout_timing_replication",
        "recipe_critical_path_focused_phys_opt",  # scoped critical_cell_opt
        "recipe_post_route_phys_opt_sweep",
    ),
    HARDBLOCK_DISTANCE: (
        "recipe_critical_path_focused_phys_opt",  # scoped dsp_register_opt
        "recipe_post_route_phys_opt_sweep",
        "recipe_cell_replacement",
    ),
    CONGESTED_REGION: (
        "recipe_critical_path_focused_phys_opt",  # scoped placement_opt
        "recipe_post_route_phys_opt_sweep",
    ),
    RETIMING_CANDIDATE: (
        "recipe_register_retiming",
        "recipe_critical_path_focused_phys_opt",
        "recipe_post_route_phys_opt_sweep",
    ),
    PIN_SWAP_CANDIDATE: (
        "recipe_critical_path_focused_phys_opt",  # scoped critical_pin_opt
        "recipe_post_route_phys_opt_sweep",
    ),
    CELL_SPREAD: (
        "recipe_cell_replacement",
        "recipe_critical_path_focused_phys_opt",  # scoped placement_opt
        "recipe_post_route_phys_opt_sweep",
    ),
    ROUTE_DOMINATED_NO_SAFE_MOVE: (
        # By definition no safe local recipe applies.  Note for the LLM.
    ),
    NO_CLEAR_LOCAL_RECIPE: (
        "recipe_critical_path_focused_phys_opt",  # try a focused sweep
        "recipe_post_route_phys_opt_sweep",
    ),
}


# Per-path classifier

def classify_path(path: dict) -> list[PathPathology]:
    """Classify a single critical path into one or more pathologies.

    Expected `path` keys (all optional; missing data → no detection for
    that pathology):
      slack_ns:               float  (negative for failing paths)
      logic_levels:           int
      route_delay_fraction:   float  (route_delay / total_delay, 0..1)
      max_detour_ratio:       float  (worst net's routed/Manhattan)
      avg_detour_ratio:       float
      high_fanout_nets:       list[(name, fanout)]  on this path
      ff_count:               int    (registers on the path)
      lut_count:              int    (LUTs on the path)
      dsp_or_bram_endpoint:   bool   (path source/sink is DSP/BRAM)
      cell_spread:            float  (RapidWright spread metric)
      has_pin_swap_opportunity: bool (heuristic: critical LUT input on
                                      non-fastest pin)

    Returns a list — a path can carry multiple pathologies simultaneously
    (e.g., LUT_DEPTH + PLACEMENT_DETOUR is common on logic-heavy paths
    spread across the fabric).
    """
    out: list[PathPathology] = []

    slack = path.get("slack_ns")
    logic_levels = path.get("logic_levels")
    route_delay_frac = path.get("route_delay_fraction")
    max_detour = path.get("max_detour_ratio")
    avg_detour = path.get("avg_detour_ratio")
    hf_nets = path.get("high_fanout_nets") or []
    ff_count = path.get("ff_count") or 0
    lut_count = path.get("lut_count") or 0
    dsp_bram = bool(path.get("dsp_or_bram_endpoint"))
    cell_spread = path.get("cell_spread")
    pin_swap_op = bool(path.get("has_pin_swap_opportunity"))

    # PLACEMENT_DETOUR — most cells on the path have inflated routing.
    if avg_detour is not None and avg_detour >= DETOUR_RATIO_HIGH:
        conf = 0.85 if avg_detour >= DETOUR_RATIO_VERY_HIGH else 0.65
        out.append(PathPathology(
            label=PLACEMENT_DETOUR,
            reason=f"avg_detour_ratio={avg_detour:.2f} >= {DETOUR_RATIO_HIGH}",
            confidence=conf,
            suggested_recipes=list(PATHOLOGY_TO_RECIPES[PLACEMENT_DETOUR]),
        ))

    # ROUTE_DETOUR — one or two nets are pathological, rest are fine.
    # Detected as: max >> avg.
    if (max_detour is not None and max_detour >= DETOUR_RATIO_VERY_HIGH
            and (avg_detour is None or max_detour >= 2.0 * avg_detour)):
        avg_str = f"{avg_detour:.2f}" if avg_detour is not None else "n/a"
        out.append(PathPathology(
            label=ROUTE_DETOUR,
            reason=f"max_detour_ratio={max_detour:.2f} dominates avg ({avg_str})",
            confidence=0.70,
            suggested_recipes=list(PATHOLOGY_TO_RECIPES[ROUTE_DETOUR]),
        ))

    # LUT_DEPTH — many logic levels on the path.
    if logic_levels is not None and logic_levels >= LOGIC_LEVELS_HIGH:
        conf = 0.85 if logic_levels >= LOGIC_LEVELS_VERY_HIGH else 0.60
        out.append(PathPathology(
            label=LUT_DEPTH,
            reason=f"logic_levels={logic_levels} >= {LOGIC_LEVELS_HIGH}",
            confidence=conf,
            suggested_recipes=list(PATHOLOGY_TO_RECIPES[LUT_DEPTH]),
        ))

    # HIGH_FANOUT_DRIVER — at least one net on the path has high fanout.
    if hf_nets:
        max_fanout = max((f for _n, f in hf_nets), default=0)
        if max_fanout >= HIGH_FANOUT_THRESHOLD:
            conf = (0.85 if max_fanout >= VERY_HIGH_FANOUT_THRESHOLD else 0.65)
            out.append(PathPathology(
                label=HIGH_FANOUT_DRIVER,
                reason=f"high-fanout net on path: max fanout={max_fanout} "
                       f"(threshold {HIGH_FANOUT_THRESHOLD})",
                confidence=conf,
                suggested_recipes=list(PATHOLOGY_TO_RECIPES[HIGH_FANOUT_DRIVER]),
            ))

    # HARDBLOCK_DISTANCE — DSP/BRAM endpoint + significant logic on path.
    if dsp_bram and lut_count >= 2:
        out.append(PathPathology(
            label=HARDBLOCK_DISTANCE,
            reason=f"DSP/BRAM endpoint with {lut_count} LUTs on path "
                   "(soft logic likely too far from hardblock)",
            confidence=0.55,
            suggested_recipes=list(PATHOLOGY_TO_RECIPES[HARDBLOCK_DISTANCE]),
        ))

    # CELL_SPREAD — RapidWright reports wide critical-path spread.
    if cell_spread is not None and cell_spread >= CELL_SPREAD_HIGH:
        conf = 0.85 if cell_spread >= CELL_SPREAD_VERY_HIGH else 0.60
        out.append(PathPathology(
            label=CELL_SPREAD,
            reason=f"cell_spread={cell_spread:.1f} >= {CELL_SPREAD_HIGH}",
            confidence=conf,
            suggested_recipes=list(PATHOLOGY_TO_RECIPES[CELL_SPREAD]),
        ))

    # RETIMING_CANDIDATE — path has many FFs + balanced slack budget.
    if ff_count >= RETIMING_FF_MIN and slack is not None:
        slack_per_ff = abs(slack) / ff_count
        if slack_per_ff <= RETIMING_SLACK_PER_FF_NS:
            out.append(PathPathology(
                label=RETIMING_CANDIDATE,
                reason=f"ff_count={ff_count}, |slack|/ff={slack_per_ff:.2f} ns "
                       f"<= {RETIMING_SLACK_PER_FF_NS}",
                confidence=0.65,
                suggested_recipes=list(PATHOLOGY_TO_RECIPES[RETIMING_CANDIDATE]),
            ))

    # PIN_SWAP_CANDIDATE — explicit signal from upstream analysis.
    if pin_swap_op:
        out.append(PathPathology(
            label=PIN_SWAP_CANDIDATE,
            reason="critical LUT input on non-fastest pin (caller signal)",
            confidence=0.70,
            suggested_recipes=list(PATHOLOGY_TO_RECIPES[PIN_SWAP_CANDIDATE]),
        ))

    # ROUTE_DOMINATED_NO_SAFE_MOVE — route delay dominates and no other
    # local recipe applies.  Detected LAST because it's a residual class.
    if (route_delay_frac is not None
            and route_delay_frac >= ROUTE_DELAY_FRACTION_VERY_HIGH
            and not out):
        out.append(PathPathology(
            label=ROUTE_DOMINATED_NO_SAFE_MOVE,
            reason=f"route_delay_fraction={route_delay_frac:.2f} >= "
                   f"{ROUTE_DELAY_FRACTION_VERY_HIGH} with no other pathology",
            confidence=0.50,
            suggested_recipes=list(PATHOLOGY_TO_RECIPES[ROUTE_DOMINATED_NO_SAFE_MOVE]),
        ))

    # Final fallback: nothing fired.
    if not out:
        out.append(PathPathology(
            label=NO_CLEAR_LOCAL_RECIPE,
            reason="no pathology rule fired with the available data",
            confidence=0.30,
            suggested_recipes=list(PATHOLOGY_TO_RECIPES[NO_CLEAR_LOCAL_RECIPE]),
        ))

    return out


def classify_design(paths: list[dict],
                     design_context: Optional[dict] = None
                     ) -> DesignPathology:
    """Roll up per-path classifications into a design-level diagnosis.

    `paths` is a list of path dicts (see classify_path docstring).
    `design_context` is optional metadata (initial_wns, lut_count,
    high_fanout_nets, critical_path_spread_info) that adds notes.

    Returns a DesignPathology with:
      primary_label: the most frequent pathology across paths
      counts: how often each pathology fired
      per_path: full per-path classifications
      suggested_recipes_ordered: deduplicated recipe order based on the
                                  primary pathology, with secondary
                                  pathologies' recipes appended
      notes: design-level observations the LLM should be aware of
    """
    per_path = [classify_path(p) for p in paths]
    counts: dict[str, int] = {}
    # Retain the maximum confidence for each label so one strong detection can
    # outweigh repeated weaker detections of another pathology.
    confidences: dict[str, float] = {}
    for path_labels in per_path:
        for p in path_labels:
            counts[p.label] = counts.get(p.label, 0) + 1
            if p.confidence > confidences.get(p.label, 0.0):
                confidences[p.label] = p.confidence

    if counts:
        # Don't let NO_CLEAR_LOCAL_RECIPE win the primary slot unless
        # nothing else fires across the whole design.
        nontrivial = {k: v for k, v in counts.items()
                      if k != NO_CLEAR_LOCAL_RECIPE}
        if nontrivial:
            # Score = count * max-confidence.  Ties broken by max
            # confidence then by label name (deterministic).
            def _score(label: str) -> tuple:
                return (nontrivial[label] * confidences.get(label, 0.5),
                        confidences.get(label, 0.5),
                        # Negate so max() picks earliest alphabetically.
                        -ord(label[0]) if label else 0)
            primary = max(nontrivial, key=_score)
        else:
            primary = NO_CLEAR_LOCAL_RECIPE
    else:
        primary = NO_CLEAR_LOCAL_RECIPE

    # Build recipe order: primary recipes first, then unique recipes
    # from other observed pathologies.
    seen: set = set()
    ordered: list[str] = []
    for r in PATHOLOGY_TO_RECIPES.get(primary, ()):
        if r not in seen:
            seen.add(r)
            ordered.append(r)
    for label, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
        if label == primary:
            continue
        for r in PATHOLOGY_TO_RECIPES.get(label, ()):
            if r not in seen:
                seen.add(r)
                ordered.append(r)

    notes: list[str] = []
    if design_context:
        wns = design_context.get("initial_wns")
        if wns is not None and wns >= 0:
            notes.append("design already meets timing — no recipes needed")
        global_fanout = design_context.get("high_fanout_nets") or []
        if global_fanout:
            top = max(global_fanout, key=lambda x: x[1] if isinstance(x, (list, tuple)) and len(x) > 1 else 0)
            if isinstance(top, (list, tuple)) and len(top) > 1 and top[1] >= VERY_HIGH_FANOUT_THRESHOLD:
                notes.append(
                    f"design has very-high-fanout net globally: "
                    f"{top[0]} fanout={top[1]}"
                )
        spread_info = design_context.get("critical_path_spread_info")
        if isinstance(spread_info, dict):
            avg = spread_info.get("avg_distance")
            if avg is not None and avg >= CELL_SPREAD_HIGH:
                notes.append(
                    f"global critical-path avg spread = {avg:.1f} — "
                    "design-wide placement concern"
                )

    return DesignPathology(
        primary_label=primary,
        counts=counts,
        per_path=per_path,
        suggested_recipes_ordered=ordered,
        notes=notes,
    )
