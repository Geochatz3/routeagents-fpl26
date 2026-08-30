"""Routes optimization recipes from measurable design features.

Uses Phase-1 features only: WNS in ns, target Fmax, failing-endpoint count,
critical-path spread, and remaining wall time. Returns a `RecipePlan`
containing a rule identifier, ordered actions, blocked actions, and evidence
references. Routing is pure and performs no tool, file, or mutation operations.
Missing features prevent dependent rules from firing rather than raising
errors. When no rule matches, the fallback plan delegates to the
pathology-label recipe list. Every plan retains its rule and evidence metadata
for auditing.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from typing import Optional

logger = logging.getLogger(__name__)


# Inputs

@dataclass(frozen=True)
class PhaseOneFeatures:
    """Measurables collected during Phase 1 that drive recipe routing.

    All fields are optional; the router treats missing data as "rule
    cannot fire" rather than guessing. Callers should pass whatever
    they have — the more fields, the more rules can engage.

    Fields:
      wns_ns: signed WNS in ns (negative for failing).
      clock_period_ns: clock period in ns (used to compute target Fmax).
      failing_endpoint_count: number of failing timing endpoints (Phase 1).
      critical_path_avg_spread_tiles: RapidWright avg cell spread on the
        worst paths (tiles). Same field used by pathology classifier.
      remaining_wall_budget_s: seconds remaining in the wall budget AT
        the moment routing decisions are made (caller passes; not
        derived). Drives R3 feasibility check.
      class_g_attempted: True iff `place_design -directive Explore` has
        already been tried on this design in the current run. R3 must
        not re-recommend Class G on a re-roll (Vivado nondeterminism).
    """
    wns_ns: Optional[float] = None
    clock_period_ns: Optional[float] = None
    failing_endpoint_count: Optional[int] = None
    critical_path_avg_spread_tiles: Optional[float] = None
    remaining_wall_budget_s: Optional[float] = None
    class_g_attempted: bool = False
    # Optional size and utilization features populated by phase-one probes.
    # A rule requiring a missing feature does not fire.
    cell_count: Optional[int] = None
    lut_util_pct: Optional[float] = None
    bram_util_pct: Optional[float] = None
    uram_util_pct: Optional[float] = None
    dsp_util_pct: Optional[float] = None
    memory_dominated: Optional[bool] = None
    route_pct: Optional[float] = None

    @property
    def wns_magnitude_ns(self) -> Optional[float]:
        """FAILING depth in ns — 0.0 when timing is already met.

        Out-of-distribution stress fix: the old
        abs(wns_ns) was SIGN-BLIND — a timing-MET design (wns > 0) read as
        deep-failing and could route into R1/R2/R3's destructive moves.
        All 13 real benchmarks enter with wns < 0, where max(0, -wns) ==
        abs(wns) — zero behavior change on every real vector; met designs
        (a plausible hidden-set corner) now fall through to the near-met/
        polish families instead.
        """
        if self.wns_ns is None:
            return None
        return max(0.0, -self.wns_ns)

    @property
    def target_fmax_mhz(self) -> Optional[float]:
        if self.clock_period_ns is None or self.clock_period_ns <= 0:
            return None
        return 1000.0 / self.clock_period_ns

    @property
    def achievable_fmax_mhz(self) -> Optional[float]:
        if self.wns_ns is None or self.clock_period_ns is None:
            return None
        # The achievable period is clock period minus WNS for both failing and
        # timing-met designs. A nonpositive result is treated as unavailable.
        denom = self.clock_period_ns - self.wns_ns
        if denom <= 0:
            return None
        return 1000.0 / denom

    @property
    def achievable_fmax_ratio(self) -> Optional[float]:
        a = self.achievable_fmax_mhz
        t = self.target_fmax_mhz
        if a is None or t is None or t <= 0:
            return None
        return a / t


# Outputs

@dataclass(frozen=True)
class RecipeAction:
    """One step in a recommended iter-1 sequence."""
    name: str        # tool or recipe identifier
    note: str = ""   # short one-liner: why this step + expected effect


@dataclass(frozen=True)
class RecipePlan:
    """Structured routing decision for the LLM's iter-1 prompt.

    rule_id: stable identifier ("R1".."R4" for positive recommendations,
      "FALLBACK" when no rule fires confidently).
    confidence: "high" / "medium" / "low".
    summary: one-sentence label ("Full-scope global retiming").
    actions: ordered iter-1 sequence the LLM should prefer.
    blocks: recipes the LLM must NOT pick as iter-1 (from R5/R6).
    rationale: short paragraph explaining why this rule fired.
    evidence_designs: known designs whose forensic data backs this rule.
      Used by tests as regression fixtures.
    """
    rule_id: str
    confidence: str
    summary: str
    actions: tuple[RecipeAction, ...] = ()
    blocks: tuple[str, ...] = ()
    rationale: str = ""
    evidence_designs: tuple[str, ...] = ()

    def format_for_prompt(self) -> str:
        """Render the recipe plan as prompt text for the language model.

        The layout matches other steering hints so registry-based and
        feature-based guidance use a consistent format. For the retiming safety
        path, the text makes `recipe_register_retiming` the required first
        action and disallows direct `phys_opt_design` calls unless supporting
        evidence is available.
        """
        lines: list[str] = []
        lines.append(
            f"\nFEATURE-BASED RECIPE ROUTING (deterministic, rule {self.rule_id}, "
            f"confidence={self.confidence}):"
        )
        lines.append(f"  Summary: {self.summary}")
        if self.actions:
            first = self.actions[0]
            lines.append(
                "  REQUIRED iter-1 sequence — execute step 1 FIRST, "
                "BEFORE any other tool call. Only proceed to step N+1 after "
                "step N completes and you have measured its WNS impact:"
            )
            for i, a in enumerate(self.actions, 1):
                marker = "★ START HERE → " if i == 1 else ""
                if a.note:
                    lines.append(f"    {i}. {marker}{a.name} — {a.note}")
                else:
                    lines.append(f"    {i}. {marker}{a.name}")
            lines.append(
                f"  Do NOT make a direct vivado_phys_opt_design, "
                f"vivado_place_design, vivado_route_design, or "
                f"vivado_run_tcl call BEFORE step 1 ({first.name}). "
                f"Forensic data shows that on this design profile, "
                f"direct phys_opt exploration delivers 0 MHz; the "
                f"sequence above is the calibrated win path."
            )
        if self.blocks:
            lines.append("  BLOCKED as iter-1 (per-feature applicability):")
            for b in self.blocks:
                lines.append(f"    - {b}")
        if self.rationale:
            lines.append(f"  Why: {self.rationale}")
        if self.evidence_designs:
            lines.append(
                f"  Evidence: rule validated on "
                f"{', '.join(self.evidence_designs)} (forensic data, "
                f"prior-model forensic log comparison)."
            )
        lines.append(
            "  Override policy: if step 1 of the sequence completes "
            "and produces a regression (worse WNS), revert and you may "
            "then try an alternative — but only after measuring step 1. "
            "Skipping step 1 entirely to try cheap probes is the "
            "anti-pattern that lost +22.89 MHz on a prior R4 design."
        )
        return "\n".join(lines)


# Each rule returns a plan only when all guards and required features match.
# The router evaluates rules in priority order.

# Thresholds. Kept as module constants so tests can
# import them and verify rule boundaries.

# R1 handles very large failing sets with deep negative slack and low achieved
# frequency. Scoped changes affect too little of the failing set, so this
# profile requires full-scope retiming.
R1_FAILING_ENDPOINTS_MIN = 100_000     # measured 217k and 243k; gates to huge designs only
R1_WNS_ABS_MIN_NS = 7.0                # measured 19.16 and 7.75
R1_FMAX_RATIO_MAX = 0.20               # measured 8% and 17%
# R1 routes first when negative slack is at least 10 ns, banking a usable
# checkpoint before the slower retiming step. Retiming then runs only when
# the remaining wall-clock budget permits.
R1_ROUTE_FIRST_WNS_NS = 10.0
# A very large failing set with at least 7 ns of negative slack remains an R1
# timing crisis even when a slower target clock raises the frequency ratio.
# The 0.35 cap ends at R3's moderate-headroom boundary; the endpoint-count
# guards keep R1 disjoint from R2.
R1_FMAX_RATIO_MAX_HUGE_FAILING = 0.35

# Boundary tie-breaks absorb feature noise without changing rule thresholds.
# They may only select R1's route-first plan and only within these bands.
# The WNS band [8.5, 10.0) ns favors the checkpoint that can be banked first.
# The endpoint band [85k, 100k) applies only when |WNS| is at least 10 ns
# and the frequency ratio is below 0.20, preventing compounded ambiguity.
# Its 15% width matches the WNS band's relative width.
# Do not add bands to derived-ratio, cross-profile, or feasibility boundaries;
# those lack a uniquely safer branch or could launch work without enough time.
TIEBREAK_WNS_BAND_NS = 1.5
# Count-scale equivalent distance: same relative width as the WNS band
# (1.5 / 10.0 = 15%) applied to the failing-endpoint threshold → 15,000.
TIEBREAK_FAILING_BAND = round(
    R1_FAILING_ENDPOINTS_MIN * TIEBREAK_WNS_BAND_NS / R1_ROUTE_FIRST_WNS_NS
)

# R2 handles medium failing sets with large negative slack and high spatial
# spread, using a placement-preserving replication sweep.
R2_WNS_ABS_MIN_NS = 5.0                # calibration design: |WNS|=14.5
R2_FMAX_RATIO_MAX = 0.25               # calibration design: 9.6%
R2_SPREAD_MIN_TILES = 70.0             # calibration design: spread 184 (high)
R2_FAILING_ENDPOINTS_MAX = 100_000     # calibration design: 27k (< R1's 100k floor)

R3_WNS_ABS_MIN_NS = 1.0                # calibration design: |WNS|=1.91
# R3 covers moderate-headroom, placement-limited failures through 5 ns.
# R2 is evaluated first, so exactly 5 ns belongs to R2 when its other guards match.
# The frequency-ratio bounds keep lower-headroom designs out of this path.
# The 25-minute wall gate covers an approximately 16-minute pass plus margin.
R3_WNS_ABS_MAX_NS = 5.0
R3_FMAX_RATIO_MIN = 0.35               # calibration design: 46%
R3_FMAX_RATIO_MAX = 0.55               # below the next design's 57%
R3_REMAINING_WALL_MIN_S = 25 * 60      # 25 min — Class G ≈ 16 min + safety margin
# High critical-path spread indicates placement limitation and selects Explore;
# known low spread selects the placement-preserving Auto_1 strategy.
# Missing spread fails open to Explore because spread analysis is optional.
R3_EXPLORE_SPREAD_MIN_TILES = 30.0

# Optional R3 post-placement chain: route Explore, critical_pin_opt, then route
# AggressiveExplore. It is disabled by default because the extra step changes
# the established four-step plan and can perturb otherwise suitable placement.
def _r3_reroute_chain() -> tuple:
    """Optional 5th R3 step. Empty tuple unless explicitly armed."""
    if os.environ.get("FPL26_R3_REROUTE_CHAIN", "0") != "1":
        return ()
    return (
        RecipeAction(
            name="vivado_route_design",
            note=("ONLY IF >= ~25 min wall remains after the steps above: "
                  "directive=Explore, then phys_opt_design critical_pin_opt, "
                  "then route_design directive=AggressiveExplore. Measure WNS "
                  "after each, STOP on the first non-improving pass, revert on "
                  "regression. Worth 0.158 ns on 3d-rendering — twice the "
                  "re-place above it. Skip entirely if budget is tight: this "
                  "must never displace the placement work. Sizing is MEASURED "
                  "from the evidence run: route Explore 375s + critical_pin_opt "
                  "46s + route AggressiveExplore 882s = 1303s, so a 15-min gate "
                  "would start a 22-min chain and be killed mid-reroute."),
        ),
    )

R4_WNS_ABS_MIN_NS = 0.5                # calibration design: |WNS|=1.24
R4_WNS_ABS_MAX_NS = 2.0
R4_FMAX_RATIO_MIN = 0.55               # calibration design: 57%
# Within R4, high critical-path spread selects Explore for placement-limited
# designs; low spread selects Auto_1 to preserve suitable placement.
# The 100-tile threshold separates these regimes, and missing spread fails
# closed to the safer Auto_1 path.
R4_EXPLORE_SPREAD_MIN_TILES = 100.0

# R7 handles near-met designs with a high achieved-frequency ratio.
# Preserve their placement and apply targeted physical optimization; broad
# replacement would discard an asset while closing only a small timing gap.
R7_WNS_ABS_MAX_NS = 0.5                # strictly below R4's floor — no overlap
R7_FMAX_RATIO_MIN = 0.85               # demo_corundum 95%

# Block rules — independent of positive rule that fires.

R5_LARGE_DESIGN_FAILING_ENDPOINTS = 30_000  # detour analysis becomes expensive
R5_CONSTRAINED_WALL_S = 60 * 60             # < 60 min → can't afford detour stall

R6_SCOPED_PHYS_OPT_DEFAULT_N = 20           # default scope of recipe_critical_path_focused_phys_opt
R6_SCOPE_MULTIPLIER = 50                    # spec: 50× scope → block as primary


def explore_runtime_estimate_s(cell_count: Optional[int]) -> Optional[float]:
    """Estimates Explore placement wall time from cell count.

    Returns seconds from a cell-count scaling model fitted to timed placement
    runs. Returns `None` when cell count is unavailable; callers must then use
    the existing static gate rather than guess.
    """
    if cell_count is None or cell_count <= 0:
        return None
    return EXPLORE_BASE_S + EXPLORE_S_PER_CELL * float(cell_count)


def fmt_est_s(seconds: Optional[float]) -> Optional[str]:
    """Render a modelled duration for the LLM prompt ("~45 s" / "~9 min")."""
    if seconds is None:
        return None
    return f"~{seconds:.0f} s" if seconds < 90 else f"~{seconds/60:.0f} min"


def place_est_note(f: "PhaseOneFeatures", static_text: str) -> str:
    """Formats a size-derived Explore placement estimate when available.

    Preserves the caller-provided static text verbatim when cell count is unknown.
    """
    est = fmt_est_s(explore_runtime_estimate_s(f.cell_count))
    if est is None:
        return static_text
    return f"{est} for {f.cell_count:,} cells (measured size model)"


def route_est_note(f: "PhaseOneFeatures", static_text: str,
                   s_per_cell: Optional[float] = None) -> str:
    """Size-derived route estimate, or the original static text.

    `s_per_cell` defaults to CLASSG_ROUTE_S_PER_CELL, resolved INSIDE the body:
    the constant is defined later in this module, and a default argument would be
    evaluated at def-time and raise NameError.
    """
    if f.cell_count is None or f.cell_count <= 0:
        return static_text
    if s_per_cell is None:
        s_per_cell = CLASSG_ROUTE_S_PER_CELL
    est = fmt_est_s(max(5.0, s_per_cell * float(f.cell_count)))
    return f"{est} for {f.cell_count:,} cells (measured size model)"


def classg_runtime_estimate_s(cell_count: Optional[int]) -> Optional[float]:
    """Estimates wall-clock seconds for the complete Class G sequence.

    The estimate combines fitted timing models for Explore placement and
    default-directive routing, including the banking step.
    """
    e = explore_runtime_estimate_s(cell_count)
    if e is None:
        return None
    return e + CLASSG_ROUTE_S_PER_CELL * float(cell_count) + CLASSG_BANK_RESERVE_S


def explore_infeasible_reason(f: PhaseOneFeatures) -> Optional[str]:
    """Returns a reason when the estimated Class G sequence cannot finish in time.

    This guard may only tighten Explore eligibility; it never admits a run that
    the static budget gate rejects. Model-based estimates are used
    conservatively because underestimating runtime can consume the remaining
    optimization budget.
    """
    est = classg_runtime_estimate_s(f.cell_count)
    if est is None or f.remaining_wall_budget_s is None:
        return None
    needed = est * CLASSG_SAFETY_FACTOR
    if needed > f.remaining_wall_budget_s:
        _ex = explore_runtime_estimate_s(f.cell_count) or 0.0
        return (f"Class G is modelled at {est:.0f}s for {f.cell_count:,} cells "
                f"(Explore {_ex:.0f}s + route + bank); with a "
                f"{CLASSG_SAFETY_FACTOR:.2f}x margin that needs {needed:.0f}s "
                f"but only {f.remaining_wall_budget_s:.0f}s remain — the "
                f"placement would not leave room to route and bank (the "
                f"alpha=0 wall-race shape)")
    return None


def high_congestion_risk(f: PhaseOneFeatures) -> bool:
    """True when utilization alone puts us in the high-utilization risk band."""
    return (f.lut_util_pct is not None
            and f.lut_util_pct >= R8_HIGH_UTIL_PCT)


def _r1_route_first_plan(rationale: str) -> RecipePlan:
    """Build R1's route-first plan (the proven bankable floor).

    Shared by _rule_r1's DEEP-extreme branch and the T2/C5 boundary
    tie-break — the tie-break must flip to EXACTLY this plan, not a
    near-copy that could drift.
    """
    return RecipePlan(
        rule_id="R1",
        confidence="high",
        summary="Route-first AggressiveExplore re-route, then budget-gated "
                "retiming round (huge failing set, DEEP-extreme WNS)",
        actions=(
            RecipeAction(
                name="vivado_run_tcl",
                note=("command='route_design -unroute' — REQUIRED before "
                      "the re-route (routing an already-routed design is "
                      "only incremental cleanup)"),
            ),
            RecipeAction(
                name="vivado_route_design",
                note=("directive=AggressiveExplore (~20-35 min): on this "
                      "DEEP-extreme class the re-route alone carries ~84% "
                      "of the known gain from the pristine netlist "
                      "(evidence +4.49 of +5.33 ns, hold-safe). Verify WNS "
                      "improved, let the mirror publish, THEN consider "
                      "step 3"),
            ),
            RecipeAction(
                name="vivado_phys_opt_design",
                note=("directive=AlternateFlowWithRetiming — BONUS round, "
                      "run ONLY if >=30 min wall remains (it takes ~20-25 "
                      "min here and needs a re-route after it to matter; "
                      "if <30 min remain, STOP — the routed result from "
                      "step 2 is the deliverable)"),
            ),
        ),
        blocks=(
            "recipe_critical_path_focused_phys_opt",
        ),
        rationale=rationale,
        evidence_designs=("boom_soc",),
    )


def _rule_r1(f: PhaseOneFeatures) -> Optional[RecipePlan]:
    """Routes very large failing sets with deep negative slack to global retiming.

    Schedules a single full-scope `phys_opt` pass with global retiming as the
    first action. Scoped optimization is avoided because it reaches too little
    of the failing set, and the full pass must begin before the wall budget is
    depleted.
    """
    if f.failing_endpoint_count is None:
        return None
    if f.wns_magnitude_ns is None:
        return None
    if f.achievable_fmax_ratio is None:
        return None
    if not (f.failing_endpoint_count >= R1_FAILING_ENDPOINTS_MIN
            and f.wns_magnitude_ns >= R1_WNS_ABS_MIN_NS
            and f.achievable_fmax_ratio < R1_FMAX_RATIO_MAX_HUGE_FAILING):
        return None
    if f.achievable_fmax_ratio >= R1_FMAX_RATIO_MAX:
        logger.info(
            "[router-r1-valve] fmax_ratio %.0f%% exceeds the calibrated "
            "%.0f%% bound but failing=%d (>= %d) at |WNS|=%.2f ns is an "
            "R1-class crisis — claiming it here rather than letting a "
            "slower target clock drop it out of the band.",
            f.achievable_fmax_ratio * 100, R1_FMAX_RATIO_MAX * 100,
            f.failing_endpoint_count, R1_FAILING_ENDPOINTS_MIN,
            f.wns_magnitude_ns,
        )
    if f.wns_magnitude_ns >= R1_ROUTE_FIRST_WNS_NS:
        # For deep negative slack, reroute first to bank a usable checkpoint.
        # Retiming follows only when the remaining budget can support it.
        return _r1_route_first_plan(
            rationale=(
                f"failing_endpoints={f.failing_endpoint_count:,} (≥ {R1_FAILING_ENDPOINTS_MIN:,}), "
                f"|WNS|={f.wns_magnitude_ns:.2f} ns (≥ {R1_ROUTE_FIRST_WNS_NS} — DEEP-extreme "
                "sub-class: retiming is the slow step on this class and the "
                "re-route alone carries most of the gain; route-first makes "
                "it the safe floor instead of a wall race)."
            ),
        )
    return RecipePlan(
        rule_id="R1",
        confidence="high",
        summary="Full-scope global retiming + AggressiveExplore re-route "
                "(huge failing set, extreme WNS)",
        actions=(
            RecipeAction(
                name="vivado_phys_opt_design",
                note=("directive=AlternateFlowWithRetiming, full-design scope, "
                      "single ~40 min call (UG835: 'aggressive replication + retiming')"),
            ),
            # AggressiveExplore must follow a full unroute; routing an already-routed
            # placement performs only incremental cleanup.
            # The keep-best mirror preserves the retimed checkpoint if rerouting
            # degrades timing. Zero hold slack passes the scorecard gate.
            RecipeAction(
                name="vivado_run_tcl",
                note=("command='route_design -unroute' — REQUIRED before the "
                      "re-route; run ONLY if >=25 min wall remains after the "
                      "retiming pass (the re-route takes ~20-30 min here)"),
            ),
            RecipeAction(
                name="vivado_route_design",
                note=("directive=AggressiveExplore (~20-30 min; ispd16-class "
                      "evidence +2.53 ns over Default at the same placement). "
                      "Verify WNS improved afterwards"),
            ),
        ),
        blocks=(
            # R6 will also add this — keep here for clarity even when R6
            # is independently checked (idempotent merge).
            "recipe_critical_path_focused_phys_opt",
        ),
        rationale=(
            f"failing_endpoints={f.failing_endpoint_count:,} (≥ {R1_FAILING_ENDPOINTS_MIN:,}), "
            f"|WNS|={f.wns_magnitude_ns:.2f} ns (≥ {R1_WNS_ABS_MIN_NS}), "
            f"fmax_ratio={f.achievable_fmax_ratio:.0%} "
            f"(< {R1_FMAX_RATIO_MAX_HUGE_FAILING:.0%}). "
            "Scoped recipes only touch ~0.01% of the failing set; "
            "only full-scope retiming reaches enough cells."
        ),
        evidence_designs=("boom_soc", "ispd16_example2"),
    )


def _r2_sweep_plan(rationale: str) -> RecipePlan:
    """Build R2's placement-preserving sweep plan.

    Shared by _rule_r2 and the C1-T4b spread=None degraded route — the
    degraded path must emit EXACTLY this plan, not a near-copy that could
    drift (same convention as _r1_route_first_plan).
    """
    return RecipePlan(
        rule_id="R2",
        confidence="medium",  # profile-based; no live evidence design
        summary="Post-route phys_opt sweep, critical_cell_opt first",
        actions=(
            RecipeAction(
                name="recipe_post_route_phys_opt_sweep",
                note=("internally invokes critical_cell_opt (UG835: "
                      "replication-based on critical nets); early-exit on first commit"),
            ),
        ),
        blocks=("recipe_cell_replacement",),  # detour analysis would stall
        rationale=rationale,
        evidence_designs=("vtr_mcml",),
    )


def _rule_r2(f: PhaseOneFeatures) -> Optional[RecipePlan]:
    """Routes medium failing sets with large WNS and high critical-path spread.

    Uses `critical_cell_opt` to replicate shared drivers on the worst paths.
    Blocks `cell_replacement` because detour analysis can stall on this profile.
    """
    if (f.wns_magnitude_ns is None
            or f.achievable_fmax_ratio is None
            or f.critical_path_avg_spread_tiles is None):
        return None
    fec = f.failing_endpoint_count  # may be None — treat None as "not too large"
    if not (f.wns_magnitude_ns >= R2_WNS_ABS_MIN_NS
            and f.achievable_fmax_ratio < R2_FMAX_RATIO_MAX
            and f.critical_path_avg_spread_tiles >= R2_SPREAD_MIN_TILES
            and (fec is None or fec < R2_FAILING_ENDPOINTS_MAX)):
        return None
    return _r2_sweep_plan(
        rationale=(
            f"|WNS|={f.wns_magnitude_ns:.2f} ns (≥ {R2_WNS_ABS_MIN_NS}), "
            f"fmax_ratio={f.achievable_fmax_ratio:.0%} (< {R2_FMAX_RATIO_MAX:.0%}), "
            f"avg_spread={f.critical_path_avg_spread_tiles:.1f} tiles "
            f"(≥ {R2_SPREAD_MIN_TILES}). Worst paths share drivers — "
            "replication is the high-leverage move."
        ),
    )


def _rule_r3(f: PhaseOneFeatures) -> Optional[RecipePlan]:
    """Routes moderate WNS and mid-range Fmax ratios through a one-shot Class G
    sequence.

    Orders unplacement and placement before routing and
    `AlternateFlowWithRetiming`. Selects `Explore` when spread is unknown or at
    least `R3_EXPLORE_SPREAD_MIN_TILES`; otherwise selects `Auto_1`. Unknown
    spread defaults to `Explore` because spread analysis is optional and the
    rule targets placement-limited designs. The placement is not retried
    because `Explore` has no seed control and reruns can regress.
    """
    if (f.wns_magnitude_ns is None
            or f.achievable_fmax_ratio is None):
        return None
    if f.class_g_attempted:
        return None
    # A missing wall budget does not reject R3; the plan is marked as degraded
    # because first-iteration planning normally has most of the wall budget.
    # Execution remains fail-closed: the route gate rejects destructive
    # operations when the available wall time cannot rebuild the state.
    budget_degraded = f.remaining_wall_budget_s is None
    # Wall feasibility uses the size model when cell count is known and the
    # static floor otherwise. Class G begins by unplacing, which resource-risk
    # policy forbids under high congestion risk or memory dominance.
    # Declining here avoids emitting a plan whose first operation is blocked.
    if high_congestion_risk(f) or f.memory_dominated:
        logger.info(
            "[router-r3-resource] declining Class G: lut_util=%s%% "
            "memory_dominated=%s — R8 blocks place_design -unplace, which is "
            "Class G's first step; deferring to the OOB safety floor.",
            f.lut_util_pct, f.memory_dominated)
        return None
    _classg_need = classg_runtime_estimate_s(f.cell_count)
    if _classg_need is not None:
        _wall_min = _classg_need * CLASSG_SAFETY_FACTOR
    else:
        _wall_min = R3_REMAINING_WALL_MIN_S
    if not (R3_WNS_ABS_MIN_NS <= f.wns_magnitude_ns <= R3_WNS_ABS_MAX_NS
            and R3_FMAX_RATIO_MIN <= f.achievable_fmax_ratio < R3_FMAX_RATIO_MAX
            and (budget_degraded
                 or f.remaining_wall_budget_s >= _wall_min)):
        return None
    # Class G must leave enough time after full placement for routing and
    # banking. Decline when the size-based model cannot fit the whole sequence;
    # this feasibility check can only make R3 stricter.
    _infeasible = explore_infeasible_reason(f)
    if _infeasible is not None:
        logger.info("[router-r3-size] declining Class G: %s", _infeasible)
        return None
    if _classg_need is not None:
        logger.info(
            "[router-r3-size] Class G modelled at %.0fs for %s cells "
            "(size-aware wall floor %.0fs, static floor would have been %.0fs)",
            _classg_need, f"{f.cell_count:,}", _wall_min,
            float(R3_REMAINING_WALL_MIN_S))
    if budget_degraded:
        logger.info(
            "[router-degraded] remaining_wall_budget_s=None but the rest of "
            "the R3 profile matches (|WNS|=%.3f ns, ratio %.0f%%) — routing "
            "Class G instead of FALLBACK (%s).",
            f.wns_magnitude_ns, f.achievable_fmax_ratio * 100,
            R3_DEGRADED_MARKER,
        )
    spread = f.critical_path_avg_spread_tiles
    use_safe = spread is not None and spread < R3_EXPLORE_SPREAD_MIN_TILES
    if use_safe:
        place_note = (f"directive=Auto_1 ({place_est_note(f, '~9 min')}, "
                      "ML-selected best directive — UG904; low spread → "
                      "placement already good, Explore underperforms here, "
                      f"spread {spread:.0f}<{R3_EXPLORE_SPREAD_MIN_TILES:.0f})")
        summary = "Class G safety path: re-place with Auto_1 (placement-OK, no re-roll)"
        place_clause = ("low spread → place_design Auto_1 (ML-selected; Explore "
                        "underperforms on placement-OK designs, cf. optical-flow)")
    else:
        place_note = (f"directive=Explore ({place_est_note(f, '~9 min')}, "
                      "heavy placement effort, UG904; placement-limited "
                      "profile; validated +54 MHz on finn)")
        summary = "Class G re-placement (one-shot, no re-roll)"
        place_clause = ("spread ≥ threshold (or unknown) → place_design Explore "
                        "for high-ceiling placement (validated +54 MHz on finn)")
    return RecipePlan(
        rule_id="R3",
        confidence="medium",  # variance in MHz is high, but path is consistent
        summary=summary,
        actions=(
            RecipeAction(
                name="vivado_run_tcl",
                note="`place_design -unplace` (clear existing placement)",
            ),
            RecipeAction(
                name="vivado_place_design",
                note=place_note,
            ),
            RecipeAction(
                name="vivado_route_design",
                note=f"directive=Default ({route_est_note(f, '~6 min')})",
            ),
            RecipeAction(
                name="vivado_phys_opt_design",
                note="directive=AlternateFlowWithRetiming (~1 min polish)",
            ),
        ) + _r3_reroute_chain(),
        rationale=(
            f"|WNS|={f.wns_magnitude_ns:.2f} ns (in [{R3_WNS_ABS_MIN_NS}, {R3_WNS_ABS_MAX_NS}]), "
            f"fmax_ratio={f.achievable_fmax_ratio:.0%} (in [{R3_FMAX_RATIO_MIN:.0%}, {R3_FMAX_RATIO_MAX:.0%})), "
            f"spread={spread if spread is None else round(spread)}, "
            + (f"[{R3_DEGRADED_MARKER}] wall budget unavailable (Phase-1 "
               "gap) — assume-enough; execution-time route_gate still "
               "guards the heavy ops. "
               if budget_degraded else
               f"wall_budget={f.remaining_wall_budget_s/60:.0f} min "
               f"(≥ {R3_REMAINING_WALL_MIN_S/60:.0f} min). ")
            + f"Placement quality gates retiming gains here — {place_clause}, "
            "retime polishes. Do NOT re-roll: variance "
            "can land worse than the first attempt."
        ),
        evidence_designs=("finn_radioml",),
    )


def _rule_r4(f: PhaseOneFeatures) -> Optional[RecipePlan]:
    """Routes near-target, retiming-eligible designs through placement refinement.

    Orders retiming, unplacement and placement, a second retiming pass, then
    `critical_pin_opt`. Selects `Explore` when spread reaches
    `R4_EXPLORE_SPREAD_MIN_TILES`; low or unknown spread selects `Auto_1` to
    limit placement variance. The `best_valid` checkpoint bounds the downside
    of a degraded placement.
    """
    if f.wns_magnitude_ns is None or f.achievable_fmax_ratio is None:
        return None
    if not (R4_WNS_ABS_MIN_NS <= f.wns_magnitude_ns <= R4_WNS_ABS_MAX_NS
            and f.achievable_fmax_ratio >= R4_FMAX_RATIO_MIN):
        return None
    spread = f.critical_path_avg_spread_tiles
    use_explore = spread is not None and spread >= R4_EXPLORE_SPREAD_MIN_TILES
    if use_explore:
        place_note = (f"directive=Explore ({place_est_note(f, '~15 min')}, "
                      "high-ceiling exhaustive placement — UG904; "
                      f"placement-limited profile, spread {spread:.0f}≥"
                      f"{R4_EXPLORE_SPREAD_MIN_TILES:.0f}; validated "
                      "+82-83 MHz 2/2 on corescore)")
        unplace_note = "`place_design -unplace` (clear placement before Explore)"
        summary = "Placement-limited: retime → Explore → retime → polish"
        place_clause = ("high spread → place_design Explore for high-ceiling "
                        "placement (validated +82-83 MHz 2/2 on corescore)")
    else:
        place_note = (f"directive=Auto_1 ({place_est_note(f, '~12 min')}, "
                      "ML-selected best directive — UG904; deterministic, "
                      "~+3 MHz; low spread → placement already good, "
                      "Explore underperforms here)")
        unplace_note = "`place_design -unplace` (clear placement before Auto_1)"
        summary = "Safety path: retime → Auto_1 → retime → polish"
        place_clause = ("low spread → place_design Auto_1 (ML-selected; Explore "
                        "underperforms on placement-OK designs)")
    return RecipePlan(
        rule_id="R4",
        confidence="high",
        summary=summary,
        actions=(
            RecipeAction(
                name="recipe_register_retiming",
                note="AlternateFlowWithRetiming (~6 min, ~+2.27 MHz)",
            ),
            RecipeAction(
                name="vivado_run_tcl",
                note=unplace_note,
            ),
            RecipeAction(
                name="vivado_place_design",
                note=place_note,
            ),
            RecipeAction(
                name="recipe_register_retiming",
                note="AlternateFlowWithRetiming again (~3 min, ~+0.86 MHz)",
            ),
            RecipeAction(
                name="vivado_phys_opt_design",
                note="critical_pin_opt (~1 min polish, UG904)",
            ),
        ),
        rationale=(
            f"|WNS|={f.wns_magnitude_ns:.2f} ns (in [{R4_WNS_ABS_MIN_NS}, {R4_WNS_ABS_MAX_NS}]), "
            f"fmax_ratio={f.achievable_fmax_ratio:.0%} (≥ {R4_FMAX_RATIO_MIN:.0%}), "
            f"spread={spread if spread is None else round(spread)}. "
            f"Near-target Fmax + small WNS → retime first for reliable gain; "
            f"{place_clause}; final critical_pin_opt polish."
        ),
        evidence_designs=("corescore_500_mod",),
    )


def _rule_r7(f: PhaseOneFeatures) -> Optional[RecipePlan]:
    """Routes near-met timing to a placement-preserving physical-optimization
    ladder.

    Applies when absolute WNS is below 0.5 ns and the Fmax ratio is at least
    85%. Groups the worst endpoints and runs granular `phys_opt` passes while
    they continue to improve. Preserves the existing placement because
    unplacement exposes a near-closed design to unnecessary placer variance.
    """
    if f.wns_magnitude_ns is None or f.achievable_fmax_ratio is None:
        return None
    # Zero sign-aware WNS magnitude means timing is met and remains eligible.
    # Such designs keep their placement and use granular physical optimization
    # to improve slack rather than falling back to unconstrained planning.
    if not (0.0 <= f.wns_magnitude_ns < R7_WNS_ABS_MAX_NS
            and f.achievable_fmax_ratio >= R7_FMAX_RATIO_MIN):
        return None
    return RecipePlan(
        rule_id="R7",
        confidence="medium",
        summary="Closure ladder: targeted phys_opt iteration, protect placement",
        actions=(
            RecipeAction(
                name="vivado_run_tcl",
                note=("group_path the worst ~20 endpoints into a named path "
                      "group (focus the optimizer on the actual misses)"),
            ),
            RecipeAction(
                name="vivado_phys_opt_design",
                note=("critical_cell_opt + path_groups=<the group> (live: "
                      "+0.041 ns on demo_corundum in one pass)"),
            ),
            RecipeAction(
                name="vivado_phys_opt_design",
                note="equ_drivers_opt (duplicate high-fanout equivalent drivers)",
            ),
            RecipeAction(
                name="vivado_phys_opt_design",
                note=("repeat granular passes (critical_pin_opt, retiming) "
                      "WHILE WNS improves; stop on first non-improving pass"),
            ),
            RecipeAction(
                name="vivado_route_design",
                note=("no directive (incremental cleanup) ONLY if phys_opt "
                      "left partially-routed nets"),
            ),
        ),
        blocks=("place_design -unplace",),
        rationale=(
            f"|WNS|={f.wns_magnitude_ns:.3f} ns (< {R7_WNS_ABS_MAX_NS}, below "
            f"R4's floor), fmax_ratio={f.achievable_fmax_ratio:.0%} "
            f"(≥ {R7_FMAX_RATIO_MIN:.0%}). Near-met: the placement is an "
            "asset — do NOT unplace/re-place; close the last fraction with a "
            "targeted phys_opt ladder on the worst endpoints."
        ),
        evidence_designs=("demo_corundum_25g",),
    )


def _check_r5_block(f: PhaseOneFeatures, current_blocks: set[str]) -> Optional[str]:
    """R5 — cell_replacement detour analysis stalls on large+budget-tight."""
    if f.failing_endpoint_count is None or f.remaining_wall_budget_s is None:
        return None
    if (f.failing_endpoint_count >= R5_LARGE_DESIGN_FAILING_ENDPOINTS
            and f.remaining_wall_budget_s < R5_CONSTRAINED_WALL_S):
        if "recipe_cell_replacement" not in current_blocks:
            return "recipe_cell_replacement"
    return None


def _check_r6_block(f: PhaseOneFeatures, current_blocks: set[str]) -> Optional[str]:
    """R6 — scoped phys_opt (N=20) is useless when failing set >> scope."""
    if f.failing_endpoint_count is None:
        return None
    threshold = R6_SCOPE_MULTIPLIER * R6_SCOPED_PHYS_OPT_DEFAULT_N
    if f.failing_endpoint_count > threshold:
        if "recipe_critical_path_focused_phys_opt" not in current_blocks:
            return "recipe_critical_path_focused_phys_opt"
    return None


# Optional phase-one extraction may leave a rule-required feature unavailable.
# Degraded routing runs only after all positive rules decline, and only when
# exactly one feature is missing while the remaining profile is unambiguous.
# Each degraded plan carries a marker for post-run auditing.

# Rationale markers (tests import these — keep the literals stable).
OOB_ROUTE_FIRST_WNS_NS = R1_ROUTE_FIRST_WNS_NS

# Resource-risk handling adds blocks rather than recommending new recipes.
# Iteration-one congestion is unavailable without optional post-route data,
# so high utilization acts as a conservative proxy and may over-block dense
# but uncongested designs. This protection intentionally fails closed.
R8_HIGH_UTIL_PCT = 75.0

# Explore runtime coefficients use seconds and seconds per cell.
# The linear estimate is intentionally conservative for mid-sized designs.
EXPLORE_BASE_S = 55.0
EXPLORE_S_PER_CELL = 0.0021
# Class G feasibility budgets Explore, default routing, and bank/write as one
# sequence. The routing coefficient is seconds per cell; the 60-second bank
# reserve and safety factor cover fixed overhead and runtime-model error.
CLASSG_ROUTE_S_PER_CELL = 0.0010
CLASSG_BANK_RESERVE_S = 60.0
CLASSG_SAFETY_FACTOR = 1.25
OOB_MARKER = "out-of-band: no calibrated rule"
R1_DEGRADED_MARKER = "degraded: failing=None"
R2_DEGRADED_MARKER = "degraded: spread=None"
# R3 handles a missing wall budget in-rule because its plan construction is
# spread-gated and must remain single-sited. The plan records the degradation.
R3_DEGRADED_MARKER = "degraded: wall_budget=None"


def _check_r8_blocks(f: PhaseOneFeatures,
                     current_blocks: set) -> "tuple[str, ...]":
    """Identify resource-risk conditions that should block disruptive routing
    actions.

    Dense designs block full re-placement and full unroute-and-reroute because
    either operation may fail to converge. Memory-dominated designs receive the
    same protection because large, location-constrained memory macros make
    re-placement especially disruptive. The resulting blocks protect otherwise
    uncovered feature bands.
    """
    out: list[str] = []

    def add(*names: str) -> None:
        for n in names:
            if n not in current_blocks and n not in out:
                out.append(n)

    if high_congestion_risk(f):
        add("place_design -unplace", "recipe_cell_replacement")
    if f.memory_dominated:
        add("place_design -unplace", "recipe_cell_replacement")
    return tuple(out)


def _degraded_feature_route(f: PhaseOneFeatures) -> Optional[RecipePlan]:
    """Select a safe routing plan when a required feature is unavailable.

        With no failing-endpoint count, the route-first branch requires |WNS| at least R1_ROUTE_FIRST_WNS_NS and a passing ratio gate; it never selects retiming without evidence of a large failing set.
    With no spread value, the sweep branch requires the remaining profile to
    match and a known failing count below 100,000, the medium-set cutoff. If
    both values are unavailable, only the deep-slack route-first branch may
    match; otherwise the result remains `FALLBACK`. Returned plans pass through
    `_apply_boundary_tiebreak` downstream. The route-first branch lies outside
    the WNS ambiguity band and cannot enter the count-scale band, while the
    sweep branch participates normally with its known count.
    """
    wns = f.wns_magnitude_ns
    ratio = f.achievable_fmax_ratio
    if wns is None or ratio is None:
        # WNS/clock come from MANDATORY Phase-1 steps; without them no
        # profile is unambiguous enough to degrade into.
        return None

    # Degraded R1 — the ONLY blocker is failing_endpoint_count=None.
    if (f.failing_endpoint_count is None
            and wns >= R1_ROUTE_FIRST_WNS_NS
            and ratio < R1_FMAX_RATIO_MAX):
        logger.info(
            "[router-degraded] failing_endpoint_count=None but |WNS|=%.3f ns "
            "is DEEP-extreme (>= %.1f) with ratio %.0f%% — routing R1 "
            "route-first (bankable floor) instead of FALLBACK.",
            wns, R1_ROUTE_FIRST_WNS_NS, ratio * 100,
        )
        return _r1_route_first_plan(
            rationale=(
                f"[{R1_DEGRADED_MARKER}] failing_endpoint_count unavailable "
                f"(Phase-1 step skipped), |WNS|={wns:.2f} ns "
                f"(≥ {R1_ROUTE_FIRST_WNS_NS} — DEEP-extreme), "
                f"fmax_ratio={ratio:.0%} (< {R1_FMAX_RATIO_MAX:.0%}). "
                "Profile is unambiguous without the count; FALLBACK here is "
                "the LLM-freelancing pattern that zeroed the boom class. "
                "Route-first only — retime-first needs the (unknown) huge "
                "failing set to justify its wall race."
            ),
        )

    # Degraded R2 — the ONLY blocker is spread=None (failing count KNOWN).
    if (f.critical_path_avg_spread_tiles is None
            and f.failing_endpoint_count is not None
            and wns >= R2_WNS_ABS_MIN_NS
            and ratio < R2_FMAX_RATIO_MAX
            and f.failing_endpoint_count < R2_FAILING_ENDPOINTS_MAX):
        logger.info(
            "[router-degraded] spread=None but the rest of the R2 profile "
            "matches (|WNS|=%.3f ns, ratio %.0f%%, failing=%d) — routing "
            "R2's placement-preserving sweep instead of FALLBACK.",
            wns, ratio * 100, f.failing_endpoint_count,
        )
        return _r2_sweep_plan(
            rationale=(
                f"[{R2_DEGRADED_MARKER}] spread metric unavailable (Phase-1 "
                f"step skipped), |WNS|={wns:.2f} ns (≥ {R2_WNS_ABS_MIN_NS}), "
                f"fmax_ratio={ratio:.0%} (< {R2_FMAX_RATIO_MAX:.0%}), "
                f"failing_endpoints={f.failing_endpoint_count:,} "
                f"(< {R2_FAILING_ENDPOINTS_MAX:,}). The sweep is "
                "placement-preserving — safe without spread evidence; "
                "FALLBACK is the riskier branch on this profile."
            ),
        )

    return None


def _oob_safe_plan(f: PhaseOneFeatures) -> Optional[RecipePlan]:
    """Choose a conservative plan for a feature profile that matches no calibrated
    rule.

    An actionless fallback is suppressed before prompt construction, so this
    function selects an existing shared plan rather than leaving the model
    unguided.
        When |WNS| reaches OOB_ROUTE_FIRST_WNS_NS and the known ratio is below R1_FMAX_RATIO_MAX, it returns `_r1_route_first_plan` without speculative retiming.
    All other known-WNS profiles use `_r2_sweep_plan`, which preserves
    placement. Both branches block `place_design -unplace` because no size
    feature is available to bound the cost of full re-placement. Returns `None`
    when WNS is unknown, since no routing region can then be selected safely.
    """
    wns = f.wns_magnitude_ns
    if wns is None:
        return None
    ratio = f.achievable_fmax_ratio

    # Route-first requires both deep negative slack and a known frequency ratio
    # confirming that the design is far from target; absolute WNS alone can
    # describe near-met timing at a long clock period. High utilization or
    # memory dominance also blocks the destructive full-reroute sequence.
    _aggressive_reroute_unsafe = high_congestion_risk(f) or bool(f.memory_dominated)
    if (wns >= OOB_ROUTE_FIRST_WNS_NS
            and ratio is not None
            and ratio < R1_FMAX_RATIO_MAX
            and not _aggressive_reroute_unsafe):
        base = _r1_route_first_plan(rationale="")
        why = (
            f"|WNS|={wns:.2f} ns (>= {OOB_ROUTE_FIRST_WNS_NS}) — DEEP-extreme "
            f"and confirmed far from target (ratio < {R1_FMAX_RATIO_MAX:.0%}). "
            "Route-first is the measured bankable floor on this class "
            "(~84% of known gain from the pristine netlist, hold-safe). "
            "The retiming BONUS round is budget-gated by its own step note; "
            "out-of-band, treat the routed result as the deliverable."
        )
    else:
        base = _r2_sweep_plan(rationale="")
        _guard_note = ""
        if _aggressive_reroute_unsafe and wns >= OOB_ROUTE_FIRST_WNS_NS:
            _guard_note = (
                " Route-first was WITHHELD despite the DEEP-extreme profile: "
                f"lut_util={f.lut_util_pct}% / memory_dominated="
                f"{f.memory_dominated} puts this design in the "
                "high-risk band, where a full unroute+re-route is forbidden.")
        why = (
            f"|WNS|={wns:.2f} ns, ratio="
            + (f"{ratio:.0%}" if ratio is not None else "UNKNOWN")
            + " — not a confirmed deep-extreme crisis (needs |WNS| >= "
            f"{OOB_ROUTE_FIRST_WNS_NS} AND a known ratio < "
            f"{R1_FMAX_RATIO_MAX:.0%}). The placement-preserving post-route "
            "sweep is never-worse, so it is the correct floor when no "
            "calibrated rule vouches for a placement gamble." + _guard_note
        )

    ratio_txt = f"{ratio:.0%}" if ratio is not None else "unknown"
    fec = f.failing_endpoint_count
    fec_txt = f"{fec:,}" if fec is not None else "unknown"
    spread = f.critical_path_avg_spread_tiles
    spread_txt = f"{spread:.1f}" if spread is not None else "unknown"

    logger.info(
        "[router-oob] no calibrated rule matched (wns=%.3f ratio=%s "
        "failing=%s spread=%s) — emitting conservative %s plan instead of "
        "action-less FALLBACK.",
        wns, ratio_txt, fec_txt, spread_txt, base.rule_id,
    )

    # Full replacement is permitted only when a known-size runtime estimate
    # comfortably fits and resource guards are clear. Unknown size, insufficient
    # time, high utilization, or memory dominance keeps the block fail-closed.
    blocks = tuple(base.blocks)
    _classg = classg_runtime_estimate_s(f.cell_count)
    _replace_affordable = (
        _classg is not None
        and f.remaining_wall_budget_s is not None
        and _classg * CLASSG_SAFETY_FACTOR <= f.remaining_wall_budget_s
        and not _aggressive_reroute_unsafe
    )
    if not _replace_affordable and "place_design -unplace" not in blocks:
        blocks = blocks + ("place_design -unplace",)

    return replace(
        base,
        rule_id="OOB",
        confidence="low",
        summary="Out-of-band safe floor — " + base.summary,
        blocks=blocks,
        rationale=(
            f"[{OOB_MARKER}] Phase-1 features matched no calibrated rule "
            f"(R1..R4, R7): |WNS|={wns:.2f} ns, fmax_ratio={ratio_txt}, "
            f"failing_endpoints={fec_txt}, spread={spread_txt}. " + why + " "
            "This plan is a SAFETY FLOOR, not a calibrated recommendation: "
            "no design in our evidence set has these features, so prefer the "
            "steps below over improvising, and BANK every improvement as "
            "soon as it is measured. Do NOT unplace/re-place — Explore's "
            "runtime is uncalibrated for this design."
        ),
        evidence_designs=(),
    )


def _apply_boundary_tiebreak(f: PhaseOneFeatures, plan: RecipePlan) -> RecipePlan:
    """Resolve narrow boundary ambiguities after normal rule evaluation.

    This function must run after the primary routing rules. It may change a
    decision only toward the safer route-first plan when available features
    fall within a configured ambiguity band. Missing required features disable
    the tie-break, and the operation never changes thresholds or flips away
    from the safe branch.
    """
    wns = f.wns_magnitude_ns
    if wns is None:
        return plan

    # Near the R1 WNS split, resolve the tie toward route-first.
    # Values below the tiebreak band retain the retime-first variant.
    if (plan.rule_id == "R1"
            and R1_ROUTE_FIRST_WNS_NS - TIEBREAK_WNS_BAND_NS <= wns
            < R1_ROUTE_FIRST_WNS_NS):
        logger.info(
            "[router-tiebreak] |WNS|=%.3f ns is within %.1f ns below the "
            "R1 route-first split (%.1f) — flipping retime-first -> "
            "route-first (proven bankable floor; retime-first here is the "
            "wall race that zeroed boom_v2 at eval).",
            wns, TIEBREAK_WNS_BAND_NS, R1_ROUTE_FIRST_WNS_NS,
        )
        return _r1_route_first_plan(
            rationale=(
                f"failing_endpoints={f.failing_endpoint_count:,} "
                f"(≥ {R1_FAILING_ENDPOINTS_MIN:,}), "
                f"|WNS|={wns:.2f} ns. [router-tiebreak] within "
                f"{TIEBREAK_WNS_BAND_NS} ns below the "
                f"{R1_ROUTE_FIRST_WNS_NS} ns route-first split — boundary "
                "ambiguity resolves to the proven bankable floor "
                "(route-first banks a measured route; retime-first is the "
                "50/50 wall race that scored alpha=0 on boom_v2)."
            ),
        )

    # Apply the failing-endpoint tiebreak only to profiles already beyond the
    # strict route-first WNS and ratio gates. This avoids compounding boundary
    # bands while steering clearly deep misses to the stronger recovery path.
    fec = f.failing_endpoint_count
    ratio = f.achievable_fmax_ratio
    if (plan.rule_id != "R1"
            and fec is not None
            and ratio is not None
            and R1_FAILING_ENDPOINTS_MIN - TIEBREAK_FAILING_BAND <= fec
            < R1_FAILING_ENDPOINTS_MIN
            and wns >= R1_ROUTE_FIRST_WNS_NS
            and ratio < R1_FMAX_RATIO_MAX):
        logger.info(
            "[router-tiebreak] failing_endpoints=%d is within %d below the "
            "R1 floor (%d) with unambiguous DEEP-extreme WNS %.3f ns "
            "(>= %.1f) and ratio %.0f%% — flipping %s -> R1 route-first "
            "(proven bankable floor for this shape).",
            fec, TIEBREAK_FAILING_BAND, R1_FAILING_ENDPOINTS_MIN,
            wns, R1_ROUTE_FIRST_WNS_NS, ratio * 100, plan.rule_id,
        )
        return _r1_route_first_plan(
            rationale=(
                f"failing_endpoints={fec:,}, |WNS|={wns:.2f} ns "
                f"(≥ {R1_ROUTE_FIRST_WNS_NS} — DEEP-extreme), "
                f"fmax_ratio={ratio:.0%} (< {R1_FMAX_RATIO_MAX:.0%}). "
                f"[router-tiebreak] failing count is within "
                f"{TIEBREAK_FAILING_BAND:,} below the R1 floor "
                f"({R1_FAILING_ENDPOINTS_MIN:,}) — count-scale boundary "
                "ambiguity on an unambiguously deep-extreme design "
                "resolves to the proven bankable floor (a sweep/FALLBACK "
                "cannot move a 10+ ns miss)."
            ),
        )

    return plan


# Entry point

# Positive rules in priority order. R1 is most specific (huge designs);
# R4 is the most permissive (near-target). Order matters because the
# first matching rule wins.
_POSITIVE_RULES = (_rule_r1, _rule_r2, _rule_r3, _rule_r4, _rule_r7)


def blocked_reason(plan, tool_name: str, arguments) -> Optional[str]:
    """Return a refusal reason iff the ACTIVE plan forbids this operation.

    PURE. Returns None (=allow) for every ambiguous case: no plan, no blocks,
    unrecognised shapes, or any internal error. Blocking is the exception and
    must be explicit; allowing is the default.

    A block entry matches when it is either the tool/recipe name itself
    (e.g. "recipe_cell_replacement") or a Tcl fragment appearing in the
    command argument (e.g. "place_design -unplace").
    """
    try:
        if plan is None:
            return None
        blocks = tuple(getattr(plan, "blocks", ()) or ())
        if not blocks:
            return None
        cmd = ""
        if isinstance(arguments, dict):
            cmd = str(arguments.get("command", "") or "")
        norm_cmd = " ".join(cmd.split())
        for b in blocks:
            if not b:
                continue
            if b == tool_name or b in str(tool_name):
                return (f"recipe plan {getattr(plan, 'rule_id', '?')} blocks "
                        f"'{b}' (matched tool {tool_name})")
            if " " in b and b in norm_cmd:
                return (f"recipe plan {getattr(plan, 'rule_id', '?')} blocks "
                        f"'{b}' (matched command)")
        return None
    except Exception:  # pragma: no cover — fail OPEN, always
        return None


def decide_recipe_path(features: PhaseOneFeatures) -> RecipePlan:
    """Map Phase-1 features → structured RecipePlan.

    Walks the rules in priority order. First positive rule that fires
    wins. Block rules (R5, R6) are then checked independently and any
    additional blocks are merged into the returned plan.

    When no positive rule fires, returns a FALLBACK plan that tells the
    LLM to follow the existing pathology-label recipe ordering (no
    feature-based override).
    """
    plan: Optional[RecipePlan] = None
    for rule_fn in _POSITIVE_RULES:
        plan = rule_fn(features)
        if plan is not None:
            logger.info(
                "recipe_router: positive rule fired rule_id=%s "
                "(evidence=%s, confidence=%s)",
                plan.rule_id, plan.evidence_designs, plan.confidence,
            )
            break

    if plan is None:
        # Before fallback, recover profiles obscured by exactly one unavailable
        # phase-one feature. This runs only after normal routing declines, so it
        # cannot replace a plan selected by a positive rule.
        plan = _degraded_feature_route(features)

    if plan is None:
        # Before actionless fallback, map out-of-distribution profiles to the
        # nearest conservative, bankable plan. This limits unconstrained LLM
        # planning when no calibrated rule applies.
        plan = _oob_safe_plan(features)

    if plan is None:
        plan = RecipePlan(
            rule_id="FALLBACK",
            confidence="low",
            summary="No high-confidence feature-based routing — defer to pathology classifier",
            actions=(),
            rationale=(
                "Phase-1 features did not match any calibrated rule "
                "(R1..R4). Use the recipe ordering from the pathology "
                "classifier as the iter-1 prior."
            ),
            evidence_designs=(),
        )
        logger.info(
            "recipe_router: no positive rule fired — returning FALLBACK "
            "(wns=%s, fmax_ratio=%s, failing=%s)",
            features.wns_magnitude_ns,
            features.achievable_fmax_ratio,
            features.failing_endpoint_count,
        )

    # Boundary tie-break: inside a narrow band at a risk-asymmetric
    # boundary, resolve toward the proven bankable floor. Runs before the
    # block-rule merge so R5/R6 apply to the FINAL plan.
    plan = _apply_boundary_tiebreak(features, plan)

    # Merge block rules. R5/R6 fire independently of which positive rule
    # won, so a plan may end up with blocks not implied by its positive
    # rule alone.
    current_blocks = set(plan.blocks)
    additional: list[str] = []
    for check in (_check_r5_block, _check_r6_block):
        b = check(features, current_blocks)
        if b is not None:
            additional.append(b)
            current_blocks.add(b)
    # R8 returns 0..n blocks rather than one.
    for b in _check_r8_blocks(features, current_blocks):
        additional.append(b)
        current_blocks.add(b)
    if additional:
        plan = RecipePlan(
            rule_id=plan.rule_id,
            confidence=plan.confidence,
            summary=plan.summary,
            actions=plan.actions,
            blocks=tuple(plan.blocks) + tuple(additional),
            rationale=plan.rationale,
            evidence_designs=plan.evidence_designs,
        )
        logger.info(
            "recipe_router: appended block rules %s to plan %s",
            additional, plan.rule_id,
        )

    return plan
