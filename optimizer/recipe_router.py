"""Feature-based recipe routing — generalizes cross-model steering beyond design names.

Background: `optimizer/cross_model_steering.py` keys hints on `design_name`,
so it works perfectly on the 4 calibrated designs (ispd16, boom_soc, finn,
corescore) but contributes nothing to unseen contest benchmarks at
submission time. The pathology classifier in `optimizer/pathology.py`
gives a label (CELL_SPREAD, RETIMING_CANDIDATE, ...) but on the 4 known
designs the label is the same (CELL_SPREAD) for four different winning
recipes — so labels alone don't route.

This module adds a second, finer-grained routing layer on top of the
pathology classifier. Inputs are Phase-1 measurables only (|WNS|,
target Fmax from clock period, failing endpoint count, critical-path
average spread, remaining wall budget). Output is a structured
RecipePlan with rule_id + ordered actions + blocks + evidence pointers.

Design contract:
  - PURE function. No Vivado/RapidWright/file IO. No mutation.
  - Graceful degradation: missing features → rule doesn't fire (not a crash).
  - Conservative: when no rule fires, returns a FALLBACK plan that tells
    the LLM to follow the existing pathology-label recipe list.
  - Auditable: every returned plan includes rule_id + evidence_designs so
    a post-run audit can ask "did rule R1 fire on the right designs?".

Rules are derived from .planning/beta/grok43_recovery_report.md §12.4
and validated against the §12.6 known-design fingerprint table.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

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
    # --- jul25 (B4) size + resource features -------------------------
    # All OPTIONAL, like every field above: missing => a rule keyed on
    # them simply cannot fire. Sourced from the Phase-1 probes
    # (cell_count was already measured; the rest come from the
    # report_utilization step that replaced report_qor_assessment).
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

        jul22 OOD stress fix (ood_router_stress_jul22 finding 3): the old
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
        # Sign-correct achievable period (jul22 OOD stress fix, finding 3):
        # period - wns  ==  period + |wns| when failing (wns < 0, all real
        # vectors — unchanged) and  period - slack  when met (wns > 0; the
        # OLD abs() UNDERestimated achievable fmax on met designs).
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


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

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
    confidence: "high" / "medium" / "low". Mirrors the §12.4 table.
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
        """Render the plan as injectable LLM prompt text.

        Mirrors the cross_model_steering hint style so the LLM sees a
        consistent format whether the steering came from a per-design
        registry hit or from this feature-based router.

        Empirical observation 2026-05-18 (uncovered-campaign):
          - R3 (Class G) followed by LLM in 1/1 runs → +91.92 MHz win.
          - R4 (safety path) NOT followed by LLM in 2/3 runs → 0 MHz.
        The R4 prose was treated as advisory ("preferred"); the LLM did
        direct phys_opt_design calls instead of step 1
        (recipe_register_retiming). Language tightened below to make
        step 1 the explicit FIRST move and direct phys_opt calls a
        violation absent evidence.
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
                f".planning/beta/grok43_recovery_report.md §12)."
            )
        lines.append(
            "  Override policy: if step 1 of the sequence completes "
            "and produces a regression (worse WNS), revert and you may "
            "then try an alternative — but only after measuring step 1. "
            "Skipping step 1 entirely to try cheap probes is the "
            "anti-pattern that lost +22.89 MHz on a prior R4 design."
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
# Each `_rule_RX` returns RecipePlan|None. Returns None when the rule's
# guards aren't satisfied OR required features are missing. Caller
# (decide_recipe_path) tries rules in priority order.

# Thresholds — pulled from §12.4. Kept as module constants so tests can
# import them and verify rule boundaries.

# R1 = the HUGE-failing-set + WNS-bound family whose only viable move is
# full-scope retiming (scoped recipes touch ~0.01% of the failing set).
# Calibrated to cover BOTH huge designs (2026-06-02 live measurement):
#   boom_soc : failing 217,988 / |WNS| 19.16 ns / fmax_ratio  8%
#   ispd16   : failing 242,906 / |WNS|  7.75 ns / fmax_ratio 17% / spread 660
# ispd16 was MISATTRIBUTED to R2 (fixture said 30k failing); the real design
# has MORE failing endpoints than boom and routed to FALLBACK → 0 (it never
# reached the retiming move that won it +16.95 on 2026-05-12, F3). The
# failing≥100k guard restricts R1 to ONLY these two designs (no other
# benchmark has ≥100k failing endpoints), so widening the WNS/fmax bounds to
# include ispd16 cannot pull in any R3/R4-class design.
R1_FAILING_ENDPOINTS_MIN = 100_000     # boom_soc 217k, ispd16 243k; gates to huge designs only
R1_WNS_ABS_MIN_NS = 7.0                # boom 19.16, ispd16 7.75 (was 10.0)
R1_FMAX_RATIO_MAX = 0.20               # boom 8%, ispd16 17% (was 0.15)
# R1 ROUTE-FIRST split (jul05 probe): on the DEEP-extreme sub-class the
# re-route alone carries most of the gain and retiming is the slow step —
# boom (|WNS| 19.16): retiming 46 min local vs ispd16 (7.75): 17 min; the
# jul05 probe recovered +4.49 of the +5.33 ns (84%) from the PRISTINE
# netlist via unroute+AggressiveExplore alone (whs +0.001, ~57 min local,
# fits the eval hour). Retime-first on this sub-class is a 50/50 wall race
# (worst case ships the retime-only mirror ~ +2.9); route-first makes the
# big chunk the SAFE floor and retiming the budget-gated bonus round.
# Threshold history: 12.0 split 19.16 from 7.75 "with wide margin" — but the
# (7.75, 12.0) band was uncalibrated, and the beta eval (2026-07-14) landed
# boom_soc_v2 EXACTLY there (|WNS| 11.392, 220k failing, 379k cells): it
# took retime-first, the 23-min retime pass + >=2888s re-route could not fit
# the 3199s eval window, and the design scored alpha=0 (rank 20/21). The
# jul19 AWS leg confirmed the shape at eval speed (retime 1444s measured;
# predicted re-route 2888s refused by the D1 gate; banked-retime alpha only
# +2.76 vs route-first's probe-proven +13.3-class floor). Lowered 12.0 ->
# 10.0 (jul20): boom_v2 (11.392) now takes the SAFE route-first floor with
# 1.39 ns margin; ispd16 (7.75) keeps its +39.6 retime-first path with 2.25
# ns margin. For unknown hidden designs in the old dead band, route-first is
# the conservative pick (retime-first is the 50/50 wall race).
R1_ROUTE_FIRST_WNS_NS = 10.0
# B2 (jul22 stress-test rec #5): the coverage hole where a design with a
# HUGE failing set and DEEP WNS misses R1 only because its target clock is
# slower, pushing fmax_ratio just over 0.20 -> no rule fires. That is the
# exact shape of the historical ispd16 misattribution bug (FALLBACK on an
# R1-shaped design -> phys_opt budget-skipped -> 0 MHz).
# A >=100k failing set at |WNS| >= 7 ns IS a timing crisis regardless of
# the ratio; the ratio gate exists only to exclude near-met designs.
# 0.35 ~= 2x the worst CALIBRATED ratio (ispd16 17%; boom 8%) and stays
# strictly below R3's 0.35 floor, so R1 cannot steal from R3/R4/R7. It
# cannot steal from R2 either: R2 requires failing < 100k, R1 requires
# >= 100k -- the two are disjoint on that feature.
R1_FMAX_RATIO_MAX_HUGE_FAILING = 0.35

# ---------------------------------------------------------------------------
# T2/C5 ambiguity tie-break (jul20 plan-review consensus, diff-list #5)
# ---------------------------------------------------------------------------
# When features land WITHIN A NARROW BAND of a rule boundary AND the two
# branches straddling that boundary differ in risk profile, resolve the
# ambiguity toward the branch with the PROVEN BANKABLE FLOOR (route-first /
# R1-class: unroute + AggressiveExplore banks a measured route) instead of
# the gamble branch. This is a TIE-BREAK at existing boundaries only — no
# threshold is re-tuned, and the check can only flip a decision TOWARD the
# safer branch inside the band (never away from it, never outside it).
#
# Boundary-by-boundary risk audit (which boundaries get the tie-break):
#
#   TIE-BREAK — R1_ROUTE_FIRST_WNS_NS (10.0, internal R1 split):
#     Below = retime-first, the 50/50 wall race that scored alpha=0 on
#     boom_soc_v2 at eval (|WNS| 11.392 in the old dead band; 23-min retime
#     + >=2888s re-route missed the 3199s window). Above = route-first, the
#     probe-proven bankable floor (+4.49 of +5.33 ns from the pristine
#     netlist; +13.3-class at eval). This is the ONLY boundary where both
#     sides serve the SAME design profile (all R1 gates already passed) and
#     exactly one side is the proven safe floor. Band = 1.5 ns below the
#     split: [8.5, 10.0). ispd16 (7.75, the proven +39.6 retime-first
#     winner) stays outside with 0.75 ns margin.
#
#   TIE-BREAK — R1_FAILING_ENDPOINTS_MIN (100k, count-scale), ONLY when the
#     design is otherwise unambiguously deep-extreme (|WNS| >= 10.0 strict,
#     no band-on-band compounding, ratio < R1_FMAX_RATIO_MAX): a design
#     measured just below 100k failing endpoints currently gets R2's
#     placement-preserving sweep (evidence +6.08 MHz on a 27k/14.5ns
#     design — cannot move a 10+ ns miss) or FALLBACK (LLM freelancing,
#     historically 0 on this class). Route-first is the proven floor for
#     the deep-extreme shape and failing-endpoint counts are the noisiest
#     Phase-1 measurable. Equivalent count distance derived from the WNS
#     band ratio: 1.5/10.0 = 15% of the 100k threshold = 15,000 → band
#     [85k, 100k), empty of measured designs (finn 46k ... boom 217k).
#
#   LEFT ALONE (documented so nobody adds generic fuzz later):
#     - R1_WNS_ABS_MIN (7.0): both sides are gambles (below → FALLBACK,
#       above → retime-first). Route-first below 8.5 ns has no probe
#       evidence, and retime-first at 7.75 is the proven +39.6 winner —
#       there is no bankable-floor side to prefer.
#     - R1_FMAX_RATIO_MAX (0.20): ratio is DERIVED from wns + clock period,
#       so a ratio band would double-count the WNS band's noise model.
#     - R2/R3 meeting at 5.0 ns, R3/R4 at fmax 0.55, R4 band [0.5, 2.0],
#       R4/R7 at 0.5 ns: neither side is route-first/R1-class; "safer" is
#       ambiguous (sweep vs Class-G re-place vs placement-protecting
#       ladder) and downside there is already bounded by the keep-best
#       mirror. Fuzzing these would be threshold re-tuning by stealth.
#     - R3_REMAINING_WALL_MIN_S: a feasibility gate, not a risk-profile
#       choice — flipping near it would either launch Class G without the
#       budget to finish or neuter R3.
TIEBREAK_WNS_BAND_NS = 1.5
# Count-scale equivalent distance: same relative width as the WNS band
# (1.5 / 10.0 = 15%) applied to the failing-endpoint threshold → 15,000.
TIEBREAK_FAILING_BAND = round(
    R1_FAILING_ENDPOINTS_MIN * TIEBREAK_WNS_BAND_NS / R1_ROUTE_FIRST_WNS_NS
)

# R2 real design (measured 2026-06-03): vtr_mcml — |WNS| 14.5, fmax 9.6%,
# spread 184, failing 27,003 (< 100k so it misses R1). Replication sweep
# yielded +6.08 MHz. (ispd16, the former R2 fixture, is actually R1 — 243k
# failing.)
R2_WNS_ABS_MIN_NS = 5.0                # vtr |WNS|=14.5
R2_FMAX_RATIO_MAX = 0.25               # vtr 9.6%
R2_SPREAD_MIN_TILES = 70.0             # vtr spread 184 (high)
R2_FAILING_ENDPOINTS_MAX = 100_000     # vtr 27k (< R1's 100k floor)

R3_WNS_ABS_MIN_NS = 1.0                # finn |WNS|=1.91
# Max raised 3.0 → 5.0 (2026-06-04) to close the realistic FALLBACK gap at
# |WNS| 3–5 ns.  The feature-space sweep found a band BETWEEN R3 (was |WNS|≤3)
# and R2 (|WNS|≥5) where a moderate-headroom, placement-limited design would
# fall through to the generic FALLBACK recipe instead of the proven Class-G
# Explore lever.  Measured full-13 |WNS| set is {0.95,0.98,1.03,1.24,1.65,1.69,
# 1.91,2.15,7.75,14.53,19.16} (+2 timing-met) — NONE sit in (3,5), so widening
# the band cannot re-route any validated design.  R2 (min 5.0) is tried BEFORE
# R3 in _POSITIVE_RULES, so the bands meet at 5.0 with R2 taking precedence; a
# |WNS|≥5 design still hits R2 first.  R3's fmax gate [0.35,0.55) is deliberately
# left unchanged so only the same moderate-headroom profile is admitted — a
# (3,5)-WNS design far from target (fmax<0.35) still falls through, as intended.
# Generalization-only: unvalidated on a real gap-design; shipped test-gated.
R3_WNS_ABS_MAX_NS = 5.0
R3_FMAX_RATIO_MIN = 0.35               # finn 46%
R3_FMAX_RATIO_MAX = 0.55               # < corescore 57%
R3_REMAINING_WALL_MIN_S = 25 * 60      # 25 min — Class G ≈ 16 min + safety margin
# R3's Class-G default is `place_design Explore`. The spread→Explore audit
# that gated R4 shows Explore HURTS placement-OK (low-spread) designs but WINS
# on placement-limited (high-spread) ones.  Full-13 real-spread measurement
# (2026-06-03) brackets the boundary tightly:
#   Explore WINS  : vexriscv 53.5 (+113), amd 56.9 (+100.7), 3d 65.8, finn 255 (+52)
#   Explore HURTS : optical 15.2 (Auto_1 +27 vs Explore 14-23), spam 8.4
# i.e. the crossover lies in the EMPTY gap (15.2, 53.5].  The floor 30 sits in
# that gap — below every measured Explore-winner (lowest 53.5) and above every
# Auto_1 design (highest 15.2).  Set well below R4's 100 because R3 designs have
# more placement headroom (fmax 35-55% vs R4's 57%+) so Explore pays off at
# lower spread.  KNOWN spread < floor → safe Auto_1; UNKNOWN keeps Explore
# (R3's validated default; spread analysis is optional/often skipped, so
# defaulting unknown→safe would neuter R3).
R3_EXPLORE_SPREAD_MIN_TILES = 30.0

# Post-placement REROUTE CHAIN — OFF by default (FPL26_R3_REROUTE_CHAIN=1 to arm).
#
# Evidence: rosetta_3d-rendering's artifact-verified +19.09 (jul04; the DCP was
# opened jul27 — 11197 fully routed nets, 0 routing errors, WNS -1.910 matching
# the headline). Its gain splits:
#     re-place    -2.153 -> -2.068   0.085 ns
#     REROUTE     -2.068 -> -1.910   0.158 ns   <- ~2x the placement work
# route Explore -> critical_pin_opt -> route AggressiveExplore. No recipe
# prescribes it; the jul26 run reached the retime step, stopped, and shipped
# +7.39. This is the jun15 AggressiveExplore lever, never promoted into R3.
#
# DEFAULT OFF because R3 also routes finn_radioml, whose +61.96 parity record
# does NOT use this chain. A 5th step changes what finn is told, and the
# existing 4-step contract is asserted by
# test_recipe_router.test_finn_triggers_r3_class_g. Arming it is an A/B, not a
# default: the flag exists so 3d can be tested without altering finn.
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
                  "from 3d's jul04 run: route Explore 375s + critical_pin_opt "
                  "46s + route AggressiveExplore 882s = 1303s, so a 15-min gate "
                  "would start a 22-min chain and be killed mid-reroute."),
        ),
    )

R4_WNS_ABS_MIN_NS = 0.5                # corescore |WNS|=1.24
R4_WNS_ABS_MAX_NS = 2.0
R4_FMAX_RATIO_MIN = 0.55               # corescore 57%
# Within the R4 band, place_design Explore wins big on PLACEMENT-LIMITED
# designs (high critical-path spread) but HURTS placement-OK designs (low
# spread).  Probe 2026-06-02: corescore (spread 232) Explore +82-83 MHz 2/2
# vs Auto_1 +6; optical-flow (spread 15) Explore 14-23 vs Auto_1 ~26 (worse +
# high variance).  Threshold 100 separates them with margin; unknown spread
# falls back to the safe Auto_1 path.
R4_EXPLORE_SPREAD_MIN_TILES = 100.0

# R7 — closure class (hidden-benchmark gap found live 2026-06-11): NEAR-MET
# designs (|WNS| below R4's 0.5ns floor, high fmax ratio) fell through every
# positive rule to FALLBACK — none of the 13 benchmarks have this profile, but
# the organizer's demo_corundum does (wns=-0.099, ratio 95%, failing=42) and
# the hidden benchmark may too. The winning live moves were targeted phys_opt
# iteration (group_path focus + critical_cell_opt: +0.041ns in one pass), NOT
# re-placement — a near-met placement is an asset; unplace would gamble it.
R7_WNS_ABS_MAX_NS = 0.5                # strictly below R4's floor — no overlap
R7_FMAX_RATIO_MIN = 0.85               # demo_corundum 95%

# Block rules — independent of positive rule that fires.

R5_LARGE_DESIGN_FAILING_ENDPOINTS = 30_000  # detour analysis becomes expensive
R5_CONSTRAINED_WALL_S = 60 * 60             # < 60 min → can't afford detour stall

R6_SCOPED_PHYS_OPT_DEFAULT_N = 20           # default scope of recipe_critical_path_focused_phys_opt
R6_SCOPE_MULTIPLIER = 50                    # spec: 50× scope → block as primary


def explore_runtime_estimate_s(cell_count: Optional[int]) -> Optional[float]:
    """Estimated `place_design -directive Explore` wall for a design this size.

    Replaces the hardcoded "~9-15 min" prose, which was calibrated on corescore
    (252,741 cells, measured 582 s) alone and applied across a 158x cell-count
    range: on vexriscv (3,373 cells) Explore genuinely costs ~60 s, so the old
    constant was ~10x too pessimistic; extrapolated to an ispd16-sized 532k-cell
    design it is ~1,225 s, which is far beyond it in the other direction.

    Returns None when cell_count is unknown — callers must then fall back to the
    old static gate rather than guess.
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
    """Size-derived Explore/place estimate, or the original static text.

    Returns the measured-model estimate when cell_count is known; otherwise the
    caller's hardcoded string is preserved verbatim (no guessing without size).
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
    """Modelled wall for the WHOLE Class-G sequence (Explore + route + bank).

    Both terms are measured, not assumed: Explore from 39 timed calls across 12
    designs, route from 163 timed calls at the Default directive Class G uses.
    """
    e = explore_runtime_estimate_s(cell_count)
    if e is None:
        return None
    return e + CLASSG_ROUTE_S_PER_CELL * float(cell_count) + CLASSG_BANK_RESERVE_S


def explore_infeasible_reason(f: PhaseOneFeatures) -> Optional[str]:
    """Non-None iff a size-aware estimate says Class G cannot complete.

    SAFETY-ONLY, BY DESIGN: this can make Explore-recommending rules STRICTER,
    never looser. The measured model also shows the old 25-minute floor is far
    too pessimistic for small designs (a 3.4k-cell design needs ~60 s, not 25
    min) — but LOOSENING a gate on a model rather than a live A/B is exactly the
    kind of projection this project has been burned by, so the loosening side is
    deliberately NOT implemented here. It is logged as an opportunity in
    EXPLORE_RUNTIME_MODEL_jul25.md pending a live trial.
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
    """True when utilization alone puts us in the panel's band-2 risk region."""
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
    """R1 — huge-failing-set + WNS-bound family (boom_soc, ispd16).

    Single full-scope phys_opt with global retiming. Scoped recipes
    address ~0.01% of the failing set and are demonstrably 35× under
    baseline (boom_soc scoped: +0.082 MHz vs +2.87 baseline). ispd16
    (failing 242,906) joined this family 2026-06-02: its proven win is
    the same AlternateFlowWithRetiming move (F3, +16.95 MHz), and routing
    it here makes that the FIRST action so the ~33-min pass runs before
    the LLM exhausts the budget exploring (the FALLBACK baseline did
    exactly that → phys_opt budget-skipped → 0).
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
            "slower target clock drop it out of the band (B2/jul22 rec #5).",
            f.achievable_fmax_ratio * 100, R1_FMAX_RATIO_MAX * 100,
            f.failing_endpoint_count, R1_FAILING_ENDPOINTS_MIN,
            f.wns_magnitude_ns,
        )
    if f.wns_magnitude_ns >= R1_ROUTE_FIRST_WNS_NS:
        # DEEP-extreme sub-class (see R1_ROUTE_FIRST_WNS_NS): re-route first
        # (the cheap 84% chunk becomes the safe floor), retiming second as a
        # budget-gated bonus round. jul05 probe: baseline netlist unroute +
        # AggressiveExplore = -19.162 -> -14.675 (whs +0.001) on the class
        # evidence design.
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
            # DRILL H (jun15, integrated jul02): this family's input routing
            # is Default-quality; re-routing the SAME placement with
            # AggressiveExplore gained +2.53 ns GT on ispd16's scored clock on
            # top of the retiming result. Routing an already-routed design is
            # only incremental cleanup, so a full unroute must precede it.
            # Hold: the official scorecard gate passes whs=0.0 (jul02 preview
            # evidence), and the agent's keep-best mirror preserves the
            # retiming result if the re-route lands worse.
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
        # B3 (jul22 rec #4): R2 is the LARGEST region of the router's
        # feature space (~50% of Monte-Carlo draws) yet carried an EMPTY
        # evidence tuple, which reads as "no rule ever validated this".
        # vtr_mcml is R2's real, live-confirmed design (2026-05-18,
        # +6.08 MHz) and every R2 threshold in this file is commented
        # against it (|WNS|=14.5, spread 184, failing 27k). Backfilled so
        # the audit trail matches the calibration. ispd16 stays out —
        # reattributed to R1 (2026-06-02).
        evidence_designs=("vtr_mcml",),
    )


def _rule_r2(f: PhaseOneFeatures) -> Optional[RecipePlan]:
    """R2 — medium failing set, large WNS, high spread (no live evidence design).

    NOTE (2026-06-02): R2's original evidence design was ispd16, but a live
    measurement showed real ispd16 has 242,906 failing endpoints (not the
    ~30k once assumed) and is now correctly routed to R1 (huge-failing-set
    retiming family). R2 therefore has NO current benchmark — it remains as a
    profile-based rule for a medium-failing (<100k) + high-WNS + high-spread
    design should one appear in the hidden set. `critical_cell_opt` replicates
    shared drivers on the worst paths; cell_replacement is blocked (detour
    analysis would stall).
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
    """R3 — finn-like: moderate WNS, mid fmax ratio, sufficient wall budget.

    Class G: place_design -unplace + {Explore|Auto_1} + route +
    AlternateFlowWithRetiming. One-shot — Vivado place_design Explore is
    nondeterministic (no -seed knob in 2024+), and re-rolls can land worse
    than the first attempt.

    The placement directive is spread-gated (mirrors R4, 2026-06-02 audit):
      - spread ≥ R3_EXPLORE_SPREAD_MIN_TILES (or UNKNOWN) → `Explore`
        (placement-limited; finn spread 60 → +54 MHz). This is R3's
        validated default; unknown spread keeps it because spread analysis
        is optional and defaulting unknown→safe would neuter R3.
      - KNOWN spread below the floor → `Auto_1` (placement already good;
        Explore would add variance/loss, as it did on optical-flow under R4).
    """
    if (f.wns_magnitude_ns is None
            or f.achievable_fmax_ratio is None):
        return None
    if f.class_g_attempted:
        return None
    # jul22 OOD stress fix (finding 2): wall_budget=None no longer hard-
    # kills R3 — that exact hole FALLBACK'd finn_radioml and rosetta_3d
    # live (2026-05-20/21 archived logs) on otherwise-unambiguous R3
    # profiles, leaving +54 MHz-class gains uncaptured.  Degraded route:
    # assume-enough with an audit marker (mirrors the R1/R2 degraded
    # convention).  Safe because (a) recipes are iter-1 plans where the
    # true remaining wall ≈ the full hour, and (b) the destructive-op
    # route_gate still refuses the heavy ops at EXECUTION time if the
    # real wall cannot rebuild the state.
    budget_degraded = f.remaining_wall_budget_s is None
    # jul25 constants audit: the wall gate is size-AWARE when we know the size.
    # R3_REMAINING_WALL_MIN_S (25 min) was calibrated on finn/corescore and
    # applied across a 158x cell-count range; it refused Class G on vexriscv
    # attempt 3 (3,373 cells, sequence models at ~125 s) by ~95 s — and that
    # attempt produced our single biggest alpha. The static floor is kept ONLY
    # for the unknown-size case, where guessing is not allowed.
    # jul25 audit: R8 blocks `place_design -unplace` at high utilization and on
    # memory-dominated designs. That op is Class G's FIRST STEP, so firing R3
    # there would emit a self-contradictory plan ("do Class G" + "do not unplace").
    # Decline instead and let the OOB floor take it with the never-worse sweep.
    #
    # This IS a downgrade of a calibrated rule on an uncalibrated feature, which
    # we refused to do for R1 — the difference is that R1's route-first sequence
    # (unroute -> route) is untouched by R8's blocks, so R1 stays coherent, while
    # R3's core move is precisely the blocked one. Coherence forces the choice;
    # the panel (4/5) also puts this exact case behind the safety floor.
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
    # jul25 (B4): size-aware Explore feasibility. R3 recommends Class G, whose
    # FIRST step is a full re-place; the static 25-min floor was calibrated on
    # finn/corescore and is size-blind across a 158x cell-count range. If the
    # measured size model says the placement alone cannot leave room to route
    # and bank, decline R3 and let the OOB floor take it. SAFETY-ONLY: this can
    # only make R3 stricter, never looser.
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
    """R4 — corescore-like: near-target Fmax, small WNS, retime-eligible.

    retime → unplace + place → retime → critical_pin_opt. The placement
    directive is chosen by critical-path spread (2026-06-02 probe):
      - HIGH spread (≥ R4_EXPLORE_SPREAD_MIN_TILES) → placement-limited →
        `place_design Explore` (corescore, spread 232: +82-83 MHz 2/2 vs the
        old safe Auto_1 +6.15).
      - LOW spread / unknown → placement already good → `place_design Auto_1`
        (optical-flow, spread 15: Explore gave 14-23 < Auto_1 ~26, worse +
        high variance → keep the safe ML-selected directive).
    The agent's best_valid mirror bounds downside if a placement collapses.
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
    """R7 — closure class: near-met timing (|WNS| < 0.5 ns, fmax ratio ≥ 85%).

    Found live 2026-06-11 on the organizer's demo_corundum (wns −0.099,
    ratio 95%, failing 42): below R4's WNS floor NO positive rule fires and
    the design fell to FALLBACK. The right move on a near-met design is a
    targeted phys_opt LADDER — group the worst endpoints, iterate granular
    phys_opt passes while they keep improving — and to PROTECT the placement
    (it is 95% of the way there; unplace/re-place gambles a near-win on
    placer variance). Live evidence: critical_cell_opt with a path group
    gained +0.041 ns in one pass on demo_corundum.
    """
    if f.wns_magnitude_ns is None or f.achievable_fmax_ratio is None:
        return None
    # jul22 OOD stress fix (finding 3 follow-through): magnitude 0.0 =
    # timing MET (sign-aware wns_magnitude_ns) — include it: a met hidden
    # design belongs exactly here (protect the placement, push slack with
    # granular phys_opt), not in FALLBACK freelancing.  No real benchmark
    # enters met, so real-vector routing is unchanged.
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


# ---------------------------------------------------------------------------
# C1-T4b feature-None degradation (jul20 phase-1 red-team audit)
# ---------------------------------------------------------------------------
# Phase-1 optional steps can be skipped (timeout, RapidWright OOM, and now
# the T4a cumulative wall cap), nulling exactly the features some rules
# hard-require: R1 needs failing_endpoint_count, R2 needs spread.  Before
# this layer, a design whose profile was otherwise unambiguous fell through
# to FALLBACK — LLM freelancing, the known rank-20 boom pattern (beta eval
# 2026-07-14: FALLBACK on an R1-shaped design scored alpha=0).
#
# This layer runs ONLY when no positive rule fired (the plan would be
# FALLBACK), so it can never re-route a design that routes today — every
# existing fixture is unchanged by construction.  Each degraded path fires
# only when its ONE feature is None and the remaining features make the
# profile unambiguous; the plan carries an explicit rationale marker so
# post-run audits can count degraded routings.

# Rationale markers (tests import these — keep the literals stable).
OOB_ROUTE_FIRST_WNS_NS = R1_ROUTE_FIRST_WNS_NS

# --- R8 (jul25 uncovered-bands panel) --------------------------------
# The panel was 4/5 that the OOB safety floor is the right POLICY for the
# four uncovered bands, and unanimous in direction: less aggression, not
# more. Blocking needs far weaker evidence than recommending, so these are
# expressed as BLOCKS, never as new recipes.
# NOTE ON HONESTY: the panel's band is "util > 75% WITH CONGESTION". We
# cannot measure congestion at iter-1 (it is only reliable from a
# post-route QoR JSON; --capture-qor is default-OFF and post-finalize), so
# this fires on UTILIZATION ALONE. That is a deliberate over-trigger: it
# may block a gamble on a dense-but-uncongested design. Accepted because
# the blocked moves are all unproven on this band anyway.
R8_HIGH_UTIL_PCT = 75.0

# --- Explore runtime model (jul25, 39 measured calls / 12 designs) -----
# explore_s ~= 55 + 0.0021*cells. Predicts 62 s at 3,373 cells (measured
# 60) and 581 s at 252,741 (measured 582). Mid-range is over-predicted by
# ~20-30%, i.e. the model errs SAFE. See EXPLORE_RUNTIME_MODEL_jul25.md.
EXPLORE_BASE_S = 55.0
EXPLORE_S_PER_CELL = 0.0021
# Explore is only ONE step of Class G — a route and a bank must still fit
# afterwards, so feasibility is judged on the WHOLE sequence rather than on a
# guessed fraction of the remaining wall. The route term is measured the same
# way Explore was (163 timed route_design calls): at the Default directive
# Class G actually uses, large designs route at ~0.70-1.06 s per 1k cells
# (252,741 cells -> 176 s; 157,166 -> 166 s). AggressiveExplore routing is ~4x
# that (379k -> ~1,680 s) but Class G does not use it. Bank/write measured at
# ~20 s on the largest design (PHASE1_AUDIT_jul25.md); 60 s is a safe reserve.
CLASSG_ROUTE_S_PER_CELL = 0.0010
CLASSG_BANK_RESERVE_S = 60.0
CLASSG_SAFETY_FACTOR = 1.25
OOB_MARKER = "out-of-band: no calibrated rule"
R1_DEGRADED_MARKER = "degraded: failing=None"
R2_DEGRADED_MARKER = "degraded: spread=None"
# jul22 OOD stress fix (finding 2): R3 fires with an audit marker when
# ONLY the wall-budget feature is missing (in-rule, not in
# _degraded_feature_route — R3's plan construction is spread-gated and
# must stay single-sited).
R3_DEGRADED_MARKER = "degraded: wall_budget=None"


def _check_r8_blocks(f: PhaseOneFeatures,
                     current_blocks: set) -> "tuple[str, ...]":
    """R8 — resource-risk blocks (jul25 uncovered-bands panel).

    High utilization: a full re-place or a full unroute+re-route on a design
    that is already dense risks not converging at all, and the ONE datapoint we
    own anywhere near this regime went the wrong way (pblock compaction to raise
    utilization: local 2-CPU +17 MHz -> eval-parity 8-vCPU **-63 MHz**).

    Memory-dominated (BRAM/URAM >> LUT): the panel's band 4, "entirely
    untested" at 5/5. Memory macros are large, often location-constrained, and
    re-placing them is the refuted cell-replacement lever in another costume.
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
    """Map a single-feature-None profile to the nearest safe positive rule.

    Degraded R1 (failing_endpoint_count=None): fires only when |WNS| is
    DEEP-extreme (>= R1_ROUTE_FIRST_WNS_NS) and R1's ratio gate passes.
    Emits the ROUTE-FIRST plan only — the proven bankable floor.  Never
    the retime-first gamble: without the failing count we cannot confirm
    the huge failing set that justifies a 20-40 min retime-first bet, and
    route-first banks a measured route on any deep-extreme design.

    Degraded R2 (spread=None): fires only when the REST of R2's profile
    matches with the failing count KNOWN and medium (< 100k) — the sweep
    is placement-preserving, so running it without spread evidence risks
    little; skipping it (FALLBACK) risks the freelancing pattern.  If the
    failing count is ALSO None the profile is ambiguous (could be an
    R1-class monster where a sweep is refuted) — only the deep-extreme
    degraded-R1 branch may claim that shape; otherwise stay FALLBACK.

    The returned plans compose with T2's _apply_boundary_tiebreak
    downstream: degraded R1 requires |WNS| >= the route-first split so it
    can never sit in the WNS band, and its failing=None cannot enter the
    count-scale band (the tie-break already guards None — kept that way).
    Degraded R2 carries a real failing count and participates in the
    count-scale tie-break exactly like a normally-routed R2.
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
    """Out-of-band net — a design whose features match NO calibrated rule.

    Why this exists (jul22 OOD stress test + jul24 all-16 panel): plain
    FALLBACK is `RecipePlan(actions=(), blocks=())`, and its only consumer
    (`_build_recipe_router_block`) suppresses an action-less FALLBACK
    entirely — so the LLM receives NO routing guidance and freelances.
    That is fail-OPEN to the most expensive branch, and it is the documented
    mechanism that zeroed the boom class. The fuzz sweep put FALLBACK at
    23.7-39.7% of the reachable feature space, and the final round scores
    HIDDEN designs, which land out-of-band by definition.

    Contract — this net introduces NO new technique. It routes an uncovered
    design to the nearest ALREADY-PROVEN shared plan builder:

      |WNS| >= OOB_ROUTE_FIRST_WNS_NS AND ratio known AND
      ratio < R1_FMAX_RATIO_MAX -> _r1_route_first_plan
          The proven bankable floor. Route-first ONLY, never the retime-first
          gamble: the same reasoning _degraded_feature_route documents — a
          20-40 min retime-first wall race is justified only by a CONFIRMED
          huge failing set, and out-of-band means nothing is confirmed.
      otherwise -> _r2_sweep_plan
          Placement-preserving, therefore never-worse. It may gain little on
          a deep design, but "small gain, banked" strictly dominates "wall
          burned on an uncalibrated gamble" under mean-rank scoring.

    Both branches additionally BLOCK `place_design -unplace`: Explore's
    ~9-15 min estimate is calibrated on finn/corescore only, and the router
    has no design-size feature (jul22 risk #6), so a full re-place on an
    unknown-size design is exactly the wall-blowout that scored alpha=0.

    Returns None when |WNS| is unknown — with no WNS there is no region to
    reason about, and true FALLBACK remains the honest answer.
    """
    wns = f.wns_magnitude_ns
    if wns is None:
        return None
    ratio = f.achievable_fmax_ratio

    # Route-first demands BOTH a deep-extreme |WNS| AND a KNOWN ratio that
    # confirms the design is far from target. Absolute |WNS| alone is not
    # evidence of a crisis: at a 200 ns period, WNS -12 ns is 94% of target
    # (nearly met) and unroute + AggressiveExplore would wreck a near-win.
    # Panel guard (jul25): at high utilization, and on memory-dominated
    # designs, a full unroute -> AggressiveExplore is exactly the move the
    # panel asked us to FORBID. Route-first is our proven floor on DEEP
    # designs, but "proven" was measured on boom/ispd16 — neither dense nor
    # memory-dominated. Out-of-band we do not get to assume it transfers.
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
                f"{f.memory_dominated} puts this design in the panel's "
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

    # jul25 constants audit: this block is CONDITIONAL, not unconditional.
    # Its original justification was jul22 risk #6 — "Explore's ~9-15 min
    # estimate is uncalibrated because the router has no design-size feature".
    # B4 added that feature, so where the measured Class-G model says the
    # sequence comfortably fits, forbidding a re-place is no longer justified:
    # on vexriscv attempt 3 the router FALLBACK'd and the LLM's own full
    # re-place produced our biggest alpha (+157.99). Blocking still applies when
    # size is UNKNOWN (no basis to permit), when the sequence does not fit, or
    # when the panel's high-util / memory-dominated guards fire.
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
    """T2/C5 ambiguity tie-break — runs AFTER normal rule evaluation.

    Can only FLIP a decision toward the proven bankable floor (R1's
    route-first plan: unroute + AggressiveExplore banks a measured route)
    when features sit inside a narrow band at a risk-asymmetric boundary
    (see the audit above TIEBREAK_WNS_BAND_NS for which boundaries qualify
    and why the rest were deliberately left alone). Never fires when the
    features it needs are None; never widens, narrows, or re-tunes any
    threshold; never flips AWAY from the safe branch.
    """
    wns = f.wns_magnitude_ns
    if wns is None:
        return plan

    # Boundary 1 — R1 internal route-first split (WNS-scale band).
    # plan.rule_id == "R1" with wns < R1_ROUTE_FIRST_WNS_NS is by
    # construction the retime-first (gamble) variant; inside [split - band,
    # split) resolve to route-first. ispd16 (7.75) is below the band and
    # keeps its proven retime-first path.
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

    # Boundary 2 — R1 failing-endpoint floor (count-scale band), only for
    # designs that are otherwise UNAMBIGUOUSLY deep-extreme (strict
    # route-first WNS + R1's own ratio gate — no band-on-band compounding).
    # Without the flip these land in R2's sweep or FALLBACK, neither of
    # which can move a 10+ ns miss.
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
        # C1-T4b: before conceding FALLBACK, check whether exactly one
        # None'd Phase-1 feature is masking an otherwise-unambiguous
        # profile (see _degraded_feature_route).  Runs only here, so a
        # design that routes normally can never be re-routed by it.
        plan = _degraded_feature_route(features)

    if plan is None:
        # jul25 OOB net: before conceding an action-less FALLBACK (which
        # _build_recipe_router_block suppresses entirely, leaving the LLM
        # to freelance), route out-of-band designs to the nearest proven
        # bankable plan. Hidden designs land here by definition.
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

    # T2/C5 boundary tie-break: inside a narrow band at a risk-asymmetric
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
    # R8 (jul25) returns 0..n blocks rather than one.
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
