# The optimization play library

[METHODS.md](METHODS.md) catalogs the *control* mechanisms, how the run
decides when to stop, what to trust, and how gains are banked. This file
catalogs the **optimization plays**: the concrete moves that physically
change your placed-and-routed design to make it faster. Every play below
is something the tool can actually execute on **your** checkpoint; each
entry says what the move does to the design, when the agent reaches for
it, where it lives in the code, and what must be true for its result to
be kept.

Every MHz figure in this file is a **development-gate measurement**, our
hardware, the 16-design gate, not the 19-DCP tarball and not the official
seven ([ARTIFACT.md](../ARTIFACT.md)), recorded at the play's code home. Read
them as the evidence that put a play in the library, not as results, and not as
comparable to the seven-benchmark official table. [ARTIFACT.md](../ARTIFACT.md)
lists what *is* checkable on your machine.

Two layers:

- **Deterministic plays**, scripted sequences of Vivado/RapidWright
  operations, fired by measured features (|WNS| band, spread,
  failing-endpoint count, remaining wall). No LLM involved.
- **LLM-layer plays**: moves the LLM agent can choose during its tool
  loop: six pre-packaged recipe macros plus the manual strategy classes
  the system prompt teaches.

**Totals: 42 plays**: 7 ILS moves, 7 router recipe classes, 6
pre-proven recipe chains, 3 deep-replace legs, 6 tail/polish passes,
6 LLM recipe macros, 7 LLM manual strategy classes.

The router IDs skip R5 and R6 because those two rules are *negative*: they
block a play the LLM would otherwise reach for (detour analysis on a large
budget-tight design; scoped phys_opt when the failing set dwarfs the scope).
They select nothing, so they are not plays.

## Lead table

| # | Play | One-liner | Band / trigger | Code home |
|---|------|-----------|----------------|-----------|
| D1 | ILS full-ruin re-place | Unplace everything, re-place with a rotated directive, re-route, keep only strict wins | LLM loop stalled ≥ 10 min, or loop exit with budget left | `optimizer/ils_polish.py:ILS_COMBOS` |
| D2 | ILS partial ruin | Unplace only the worst-200-paths' fabric cells, incremental re-place | In rotation; skipped when critical-path spread < 30 tiles | `optimizer/ils_polish.py:partial_ruin_tcl` |
| D3 | ILS last-mile cycle | Netlist-level phys_opt (clock/retime/LUT) + incremental `LastMile` re-place | In rotation; only when best WNS ≥ −1.0 ns | `optimizer/ils_polish.py:LASTMILE_PD` |
| D4 | ILS route re-roll | Unroute + from-scratch `AggressiveExplore` route, nothing else | In rotation; near-met only (\|WNS\| ≤ 0.7 ns) | `optimizer/ils_polish.py:ROUTE_REROLL_PD` |
| D5 | ILS route-only re-route | Keep placement, unroute, re-route `AggressiveExplore` + phys_opt | In rotation; cheapest cycle, any WNS | `optimizer/ils_polish.py:ROUTE_ONLY_PD` |
| D6 | ILS incremental route escalation | Re-route the *incumbent* routing in place with a stronger directive (no unroute) | Appended to rotation; needs an incumbent routed with a weaker directive | `optimizer/ils_polish.py:INCR_ROUTE_PD` |
| D7 | ILS place-retry ladder | On a cycle-1 regression, force a ladder of untried placement families | Cycle regresses vs incumbent; order WNS-keyed | `optimizer/ils_polish.py:PLACE_RETRY_LADDER` |
| R1a | Crisis retiming (router R1) | One full-scope `phys_opt -directive AlternateFlowWithRetiming`, then unroute + `AggressiveExplore` re-route | ≥ 100k failing endpoints, 7 ≤ \|WNS\| < 10 ns | `optimizer/recipe_router.py:_rule_r1` |
| R1b | Route-first floor (router R1) | Unroute + `AggressiveExplore` re-route first; retiming only as budget-gated bonus | ≥ 100k failing, \|WNS\| ≥ 10 ns (deep-extreme) | `optimizer/recipe_router.py:_r1_route_first_plan` |
| R2 | Placement-preserving sweep (router R2) | Granular post-route phys_opt sweep, `critical_cell_opt` first, placement untouched | \|WNS\| ≥ 5 ns, spread ≥ 70 tiles, < 100k failing | `optimizer/recipe_router.py:_rule_r2` |
| R3 | Class-G re-place (router R3) | Unplace + full re-place (`Explore` or `Auto_1` by spread) + route + retime polish, one shot | 1 ≤ \|WNS\| ≤ 5 ns, fmax ratio 35–55%, wall fits | `optimizer/recipe_router.py:_rule_r3` |
| R4 | Retime sandwich (router R4) | Retime → unplace + re-place (spread-gated directive) → retime again → pin polish | 0.5 ≤ \|WNS\| ≤ 2 ns, fmax ratio ≥ 55% | `optimizer/recipe_router.py:_rule_r4` |
| R7 | Closure ladder (router R7) | Group the worst ~20 endpoints, iterate scoped granular phys_opt; never re-place | \|WNS\| < 0.5 ns, fmax ratio ≥ 85% (near-met) | `optimizer/recipe_router.py:_rule_r7` |
| R8 | Out-of-band safe floor | Route-first (deep) or the R2 sweep (else); full re-place blocked | Features match no calibrated rule (hidden designs) | `optimizer/recipe_router.py:_oob_safe_plan` |
| F1 | Shallow recipe chain | Unroute/unplace → place `WLDrivenBlockPlacement` → phys_opt → route `AggressiveExplore` → phys_opt | \|WNS\| ≤ 1.05 ns, pre-LLM slot | `optimizer/recipe_passes.py:RECIPE_PASS_SHALLOW_TCL` |
| F2 | Deep recipe chain | Pin-swap ×2 → fresh-Vivado round-trip → unplace → place `Explore` → phys_opt → route `AggressiveExplore` | \|WNS\| ≥ 8 ns, pre-LLM slot | `optimizer/recipe_passes.py:RECIPE_PASS_DEEP_TCL` |
| F3 | Mid recipe chain | Retime/route/re-place/re-route chain across two Vivado sessions | 1.05 < \|WNS\| < 8 ns; needs ~2800 s, fires only on walls well beyond 1 h | `optimizer/recipe_passes.py:RECIPE_PASS_MID_TCL` |
| F4 | Own-front retime candidate | Place with the band's proven directive (`ExtraTimingOpt` or `WLDrivenBlockPlacement`) → retime → phys_opt AE → route AE | \|WNS\| in [0.90, 1.05]; sub-band picks the chain | `dcp_optimizer.py:ETO_RETIME_TCL` / `WLD_RETIME_TCL` |
| F5 | Shallow determinizer | Place `AltSpreadLogic_medium` → route `Explore` → *incremental* route `AggressiveExplore` → retime polish | \|WNS\| in [0.60, 0.90) | `dcp_optimizer.py:SHALLOW_DET_TCL` |
| F6 | Sub-band phys_opt floor | Path-group the worst 20 endpoints, then scoped phys_opt ×3, banked in-session | \|WNS\| in [0.10, 0.45], ≤ 1000 failing endpoints | `dcp_optimizer.py:_maybe_run_fir_subband_floor` |
| B1 | Deep-replace from pristine | Reopen the *original* input, unplace, full re-place `Explore`, route | Deep-extreme band pre-LLM; any design at the tail if budget remains | `optimizer/deep_replace_sibling.py` |
| B2 | Deep-replace retiming tail | `phys_opt AlternateFlowWithRetiming` on the freshly re-placed design | After B1 banks, whenever a ≥ 0.20× slice of the measured place+route cost remains | `optimizer/deep_replace_sibling.py:DEEP_REPLACE_B2_MIN_SLICE_FRAC` |
| B3 | Small-floor sibling | Place `ExtraNetDelay_low` → phys_opt `AggressiveExplore` → route `AggressiveExplore` from pristine | Mid band, first stage only; cost ≤ 550 s + physics admission | `optimizer/deep_replace_sibling.py:DEEP_REPLACE_B3_*` |
| T1 | Tail-controller portfolio | Rotate 4 proven moves (re-route, phys_opt ladder, fanout opt, phys_opt AE) by measured ns/s | Post-loop, deep-WNS states (best ≤ −1.0 ns) | `optimizer/tail_controller.py:TAIL_MENU` |
| T2 | Bare re-route compounding loop | Plain `route_design` iterated while each pass still pays ≥ 0.020 ns | Routed banked best + wall fits; deep WNS compounds | `optimizer/polish_ladder.py:_run_bare_reroute_polish` |
| T3 | Mid-band break rung | From the banked best: unroute → `phys_opt -retime` → phys_opt AE → route AE | Post-loop, mid band, ~1350 s priced wall | `dcp_optimizer.py:MIDBAND_BREAK_RUNG_TCL` |
| P1 | Fanout polish | One `phys_opt -directive AggressiveFanoutOpt` pass on the final best | Post-ILS; cheap designs only, strict hold floor +0.010 ns | `optimizer/ils_polish.py:fanout_polish_accept` |
| P2 | Last-mile final polish | The D3 cycle applied once to the *final* plateaued best | Post-ILS; best WNS ≥ −1.0 ns, budget fits | `optimizer/ils_polish.py:lastmile_polish_accept` |
| P3 | Winner polish | Up to 4 `phys_opt` passes (default `AggressiveExplore`) on the shipped winner | Wrapper wall too short for another attempt (stranded time) | `scripts/winner_polish.tcl` |
| L1 | Cell re-placement macro | Find cells whose routing detours ≥ 2× Manhattan distance, move them to better sites | LLM call; critical paths with movable, detoured cells | `optimizer/recipe_passes.py:_recipe_cell_replacement` |
| L2 | LUT-cone optimization macro | Merge cascaded small LUTs on the worst paths into fewer, larger LUTs | LLM call; LUT-bound critical path | `optimizer/recipe_passes.py:_recipe_lut_optimization` |
| L3 | Register-retiming macro | `phys_opt` with a retiming directive, re-route, measure | LLM call; plateaued designs, near-target Fmax | `optimizer/recipe_passes.py:_recipe_register_retiming` |
| L4 | Granular phys_opt sweep macro | Try each phys_opt sub-optimization one at a time, keep wins, revert losses | LLM call; budget-tight, or huge designs | `optimizer/recipe_passes.py:_recipe_post_route_phys_opt_sweep` |
| L5 | Scoped phys_opt macro | Path-group the worst N paths, run one phys_opt sub-flag on just that group | LLM call; huge designs where global phys_opt is too slow | `optimizer/recipe_passes.py:_recipe_critical_path_focused_phys_opt` |
| L6 | High-fanout replication macro | Surgically replicate the drivers of named high-fanout nets on the worst paths | LLM call; fanout-dominated critical paths | `optimizer/recipe_passes.py:_recipe_high_fanout_timing_replication` |
| L7 | Manual full re-place (Class G) | Unplace + `place_design` with an escalating directive list + route + retime polish | LLM judgment; plateaued/LOSS designs | `prompts/system_prompt_scored.txt` (class G) |
| L8 | Pblock re-placement (Class C) | Constrain the whole design into a computed compact region, re-place inside it | LLM judgment; spread-out designs (avg spread > 70 tiles) | `VivadoMCP:create_and_apply_pblock` + `RapidWrightMCP:analyze_fabric_for_pblock` |
| L9 | Fanout split (Class D manual) | Split one high-fanout net's driver into replicated drivers by factor | LLM judgment; a single dominating high-fanout net | `RapidWrightMCP:optimize_fanout` |
| L10 | Surgical phys_opt flags | `-critical_pin_opt` (LUT pin swap), `-clock_opt` (skew), `-force_replication_on_nets`, `-tns_cleanup` | LLM judgment; 1–2 stubborn paths after coarser passes | `VivadoMCP:phys_opt_design` |
| L11 | Route-directive escalation (Class I) | Re-route with `Explore` → `AggressiveExplore` → `NoTimingRelaxation` / congestion directives | LLM judgment; route errors or WNS lost in routing | `VivadoMCP:route_design` |
| L12 | BRAM/DSP relocation (Class F) | Move memory/DSP macros whose path crosses the die via the cell-move play | LLM judgment; macro-crossing critical paths | via L1 targeting BRAM/DSP cells |
| L13 | ML strategy recommender (Class H) | Run Vivado's `report_qor_suggestions`, apply the suggested directives | LLM judgment; unknown/stubborn designs | `prompts/system_prompt_scored.txt` (class H) |

## Common acceptance discipline

Every play answers to the same referee. A result is kept only when it is
**measured better** on the scored clock, **fully routed** (0 routing
errors), and **hold-clean**; anything else is discarded and the banked
best stands. Three tiers of accept margin:

- **In-loop accepts** (ILS cycles, tail moves): strictly better than the
  banked best by ≥ 0.002 ns, hold worst-slack ≥ −0.001 ns (the official
  scorecard's own hold gate passes 0.0).
- **Play-off candidates** (pre-proven recipe chains, retime candidates, rungs):
  registered into the finalize MUX and win only by ≥ 0.005 ns
  (`optimizer/finalization.py:FINAL_CANDIDATE_MUX_MIN_GAIN_NS`); ties ship the
  pipeline result.
- **Full re-places from scratch** (deep-replace, B-legs): must beat the
  chain best by ≥ 0.15 ns, a from-scratch result discards accumulated
  polish, so a marginal win is treated as noise
  (`deep_replace_adopt_margin_ns`).

Retiming plays additionally pass a **latency audit**: flip-flop count is
probed before/after the retime step and the candidate is aborted if the
count drifts beyond ±1% (`optimizer/recipe_policy.py:eto_retime_ff_drift_ok`),
retiming is legal, pipeline-stage insertion is not.

---

## Deterministic plays

### ILS ruin-and-rebuild rotation (D1–D7)

Home: `optimizer/ils_polish.py:run_ils_polish`. Triggered when the LLM
loop **stalls** (no best-WNS improvement for 600 s with ≥ 1500 s left,
design ≤ 300k cells) or at **loop exit** with ≥ 600 s left and timing
still unmet (or met, when met-surplus mode is on, positive slack still
raises Fmax). Before the first cycle the Vivado process is restarted
(session state from the recipe phase measurably degrades and slows later
`place_design` calls) and the seed is chosen: the **raw input** when the
recipe phase gained < 0.15 ns (the design is stuck; a from-scratch
global re-place wins), else the **recipe best**. Stuck designs run both
seeds (raw first, recipe-best as a corrective probe on the leftover).

Each cycle applies one combo `(place directive, route directive,
phys_opt directive)` and is accepted only under the in-loop rules above.
Micro-accepts (< 0.010 ns) are kept but do not reset the futility
counter; **two consecutive non-meaningful cycles end the seed** (replay
evidence: this saved 400–1900 s per stuck run with zero forfeited
accepts). A per-cycle affordability gate prices each combo from measured
cycle costs (relative priors per directive, `combo_cost_prior`); an
optional scoring-formula hurdle can buy one extra cycle when a cheap
cycle's cost is below the score it must earn.

**D1, Full-ruin re-place.** Unplaces the entire design and re-places it
with a different placement directive each cycle, then routes and
polishes. This is the global "shake the box" move: it escapes placement
basins the incremental optimizers cannot leave. Rotation order (ordered
by measured accept counts/costs): `Explore`, `ExtraTimingOpt`,
`AltSpreadLogic_high`, `ExtraNetDelay_high`, `SSI_SpreadLogic_high`,
`EarlyBlockPlacement`. `ExtraNetDelay_high`, the costliest combo
(~1.4–3.2× an `Explore` cycle, 33% accept rate corpus-wide), is
additionally gated by **failing-endpoint density** (failing endpoints ÷
critical-path spread ≥ 363 arms it; below that it is skipped only when
the cycle would also bet ≥ 60% of the remaining window). Evidence: the
rotation is the backbone of the polish phase; documented per-directive
gains include a +114 MHz `Explore` cycle on one benchmark and an
`ExtraNetDelay_high` accept worth ~19 MHz on another.

**D2: Partial ruin.** Instead of full unplace, collects the cells on
the worst 200 setup paths (registers, LUTs, carries, muxes) and unplaces
*only those*, then re-places incrementally. Cheaper (~0.7× a full-ruin
cycle) and targeted. Scope 200 is a swept optimum (50/200/500 measured
unimodal). Skipped when the critical path is spatially co-located
(spread < 30 tiles): corpus mining shows multi-cell surgery on a
co-located path is 26/26 negative, while on spread paths it is strongly
positive (134 wins / 5 losses, mean +0.068 ns).

**D3, Last-mile cycle.** A different operator class: netlist-level
phys_opt (clock_opt / retiming / LUT restructuring) followed by an
*incremental* `place_design -directive LastMile` re-place and re-route.
Designed for plateaued, nearly-closed states where ruin combos stop
moving; the `LastMile` directive fails outright far from closure, so the
play only fires when best WNS ≥ −1.0 ns. Probe evidence: +0.152 ns
(~+29 MHz) and +0.059 ns (~+15 MHz) on two plateaued states, both
passing 10000-vector equivalence.

**D4, Route re-roll.** `route_design -unroute` then a from-scratch
`route_design -directive AggressiveExplore`, and nothing else. A fresh
route solve from a good placement re-rolls the "route lottery" without
inheriting the incumbent solution's compromises. Near-met designs only
(|WNS| ≤ 0.7 ns): the measured bite decays with depth (+0.070 ns at
−0.31; flat at −0.59; negative at −0.84). Deep-WNS designs are owned by
the bare-reroute tail loop instead.

**D5: Route-only re-route.** Keep the placement, unroute, re-route with
`AggressiveExplore`, then a phys_opt pass. The cheapest cycle in the set
(no place step) and the broadest single lever measured: improved 8 of 12
optimized checkpoints in a breadth sweep, including one +2.53 ns
ground-truth gain.

**D6, Incremental route escalation.** Re-route the *incumbent* routing
in place with a **more aggressive** directive than the one that produced
it: no unroute, so the solver refines rather than re-solves. The
escalation order is causal: `Explore` then `AggressiveExplore` reached
−0.543 on the evidence design where the reverse order stalled at −0.595;
three structurally different escalation paths converged on exactly the
same result. Skipped when the incumbent was already routed aggressively
(the precondition is perishable, so this play may jump the queue ahead
of the from-scratch route sentinels when eligible).

**D7, Place-retry ladder.** When cycle 1's re-place comes back *worse
than the incumbent*, the placement family is suspected wrong for this
design and a ladder of untried families is forced:
`ExtraNetDelay_high` → `AltSpreadLogic_medium` → `ExtraNetDelay_low`
(evidence-ordered; each rung is one keep-best cycle). Near-met designs
(|WNS| < 0.7) get `AltSpreadLogic_medium` first; spreading logic is the
fine adjustment a nearly-met design wants, while a deep miss needs
heavier net-delay weighting (3/3 measured). An unaffordable rung
advances to the next rung within the same cycle rather than surrendering
it. A mid-band-scoped hold (the retry-baseline gate) requires the
regression to be against the *pristine baseline*, not just the polished
incumbent, before the ladder may hijack the rotation.

### Band-routed first moves (R1a–R8)

Home: `optimizer/recipe_router.py:decide_recipe_path`. Runs once after
Phase-1 measurement; maps measured features: |WNS| magnitude,
failing-endpoint count, achievable-vs-target Fmax ratio, critical-path
spread (tiles), remaining wall, to a structured first-move plan that
the LLM is instructed to execute *before* freelancing. No design-name
keys. Missing features degrade gracefully (a rule that cannot fire
does not guess); boundary ambiguity resolves toward the plan with a
provable bankable floor.

**R1a, Crisis retiming.** For huge timing crises (≥ 100k failing
endpoints, 7 ≤ |WNS| < 10 ns, far from target): one full-scope
`phys_opt_design -directive AlternateFlowWithRetiming` (~40 min;
aggressive replication + retiming across the whole design, scoped
recipes touch ~0.01% of a failing set this size), then unroute and
re-route with `AggressiveExplore`. Evidence: +16.95 MHz from the
retiming move on the largest benchmark, +2.53 ns from the re-route on
the same placement.

**R1b, Route-first floor.** For *deep-extreme* crises (|WNS| ≥ 10 ns):
the order flips. Unroute + `AggressiveExplore` re-route runs FIRST,
measured to carry ~84% of the known gain from the pristine netlist
(+4.49 of +5.33 ns, hold-safe), and retiming becomes a bonus round only
if ≥ 30 min remain. Rationale: on this class retiming is the slow step,
and running it first turns the run into a wall race (a design in this
band once scored zero exactly that way).

**R2, Placement-preserving sweep.** For deep-but-not-huge misses with
high spread: the granular post-route phys_opt sweep (play L4) with
`critical_cell_opt` first: the worst paths share drivers, so
replication is the high-leverage move. Never touches placement, so it is
never-worse. Evidence: +6.08 MHz on the calibration design.

**R3, Class-G re-place.** For moderate misses with placement headroom
(1–5 ns, fmax ratio 35–55%, wall affordable by a measured size model):
`place_design -unplace` → full re-place → route → retiming polish.
Placement directive is spread-gated: `Explore` when the critical path is
spread ≥ 30 tiles or unknown (placement-limited; +54 MHz validated),
`Auto_1` when measurably compact (placement already good, `Explore`
adds variance there). Deliberately one-shot: the placer is
nondeterministic and re-rolls can land worse.

**R4: Retime sandwich.** For near-target designs (0.5–2 ns, ratio
≥ 55%): retime → unplace + re-place (`Explore` if spread ≥ 100 tiles,
else `Auto_1`) → retime again → `critical_pin_opt` polish. Retiming pays
only from a placed, timing-annotated state, which is why it brackets the
re-place. Evidence: +82–83 MHz twice on the calibration design vs +6 for
the safe path.

**R7: Closure ladder.** For near-met designs (< 0.5 ns, ratio ≥ 85%):
the placement is an asset (re-placing gambles a near-win) so the play
is a targeted phys_opt ladder: `group_path` the worst ~20 endpoints,
then iterate granular passes (`critical_cell_opt` on the group,
`equ_drivers_opt`, `critical_pin_opt`, retiming) while WNS improves.
Evidence: +0.041 ns in a single scoped pass on the calibration design.

**R8, Out-of-band safe floor.** Hidden designs matching no calibrated
rule get the nearest proven *safe* plan instead of no guidance:
route-first when deep-extreme and confirmed far from target, otherwise
the placement-preserving sweep, with full re-place blocked unless the
measured size model proves the sequence fits. Resource-risk blocks
(utilization ≥ 75%, memory-dominated designs) are implemented behind
`optimizer/utilization_features.py`, which is **absent from both the scored
tree and `main`**, so these rules take their 'no data' fallback and have
never fired ([PROVENANCE.md](PROVENANCE.md)). Where the features exist they
forbid
unplace/re-place and destructive reroutes everywhere.

### Pre-proven recipe chains (F1–F6)

Each of these is a *fixed, pre-proven step sequence*, the exact tool
chain was validated on a design class ahead of time and is replayed
verbatim as one more insured candidate for the final play-off; nothing
about it is improvised at run time.

Deterministic, drilled Tcl chains that run from the pristine input at
the pre-LLM slot (or in-session for F6), keyed **only by the measured
input |WNS| band**, no name or checksum keys. Except F6 they are
**MUX-additive**: the result is written to a private store and enters
the finalize play-off; the LLM loop then starts from an untouched
session. Retime chains carry the ±1% FF-count latency audit.

**F1, Shallow recipe chain** (|WNS| ≤ 1.05 ns):
`unroute; unplace; place WLDrivenBlockPlacement; phys_opt;
route AggressiveExplore; phys_opt`. Wirelength-driven placement suits
shallow misses. Evidence: byte-reproducible across independent machines
on two benchmarks (n=5 each); ~300 s.

**F2, Deep recipe chain** (|WNS| ≥ 8 ns): two `critical_pin_opt`
passes, then a **session round-trip** (write checkpoint → restart Vivado
→ reopen; the fresh process is a load-bearing ingredient, replays
without it land worse), then unplace → `place Explore` → phys_opt →
`route AggressiveExplore`; ~900 s. Reproduced byte-exact ×3 across
machines.

**F3, Mid recipe chain** (1.05 < |WNS| < 8 ns): a longer two-session
chain (retime → route → re-place `Explore` → route ladder → fresh
session → unroute → `AggressiveExplore` route → phys_opt `Explore`).
Measured ~2800 s, so its fail-closed wall gate never fires inside a 1 h
run; it is a long-wall play. The mid band's *affordable* play is T3.

**F4: Own-front retime candidate** ([0.90, 1.05] ns): one retime
candidate whose placement front is selected by sub-band: below 1.00,
`place ExtraTimingOpt` (then `phys_opt -retime`, phys_opt
`AggressiveExplore`, route `AggressiveExplore`); at/above 1.00, the same
chain from `place WLDrivenBlockPlacement`. Mechanism: retiming pays on
the design's own winning placement front; foreign fronts are no-ops or
harmful. Exactly one retime candidate spends wall per run.

**F5: Shallow determinizer** ([0.60, 0.90) ns):
`place AltSpreadLogic_medium; route Explore; route AggressiveExplore`
(no unroute between the two routes; this is the D6 incremental
escalation frozen into a recipe) then a retiming polish. Exists to
convert a known budget-conditional lottery into a deterministic floor:
the chain reproduced bit-identically ×3 and registers as a MUX
candidate that wins only when the pipeline's draw stalls.

**F6: Sub-band phys_opt floor** ([0.10, 0.45] ns, ≤ 1000 failing
endpoints): `group_path` the worst 20 endpoints, then `phys_opt Default`
×2 and a scoped `critical_pin_opt`, banked **in-session** so the ILS
seeding and stop logic see the floor as recipe gain. Runs after
deep-replace-first (its gains are conditional on that in-memory place
state: 0.095 ns vs 0.010 ns from a pristine session). For very shallow
designs the shallow recipe pass is carved out entirely (< 0.50 ns),
its pristine reset was the one mechanism that could score below
baseline there.

### Full re-place siblings (B1–B3)

Home: `optimizer/deep_replace_sibling.py`. The "throw it away and start
over" family: insured-compare candidates re-placed **from the pristine
input**, deliberately discarding accumulated pipeline state (this is
what distinguishes them from everything else, the archived record they
generalize reopened the original checkpoint mid-run and beat the
operated result by +22 MHz). All legs bank to dedicated files and adopt
only by the +0.15 ns MUX margin; a losing leg simply loses the compare.

**B1, Pristine full re-place.** Open the original input → unplace →
`place Explore` → route. Scheduled FIRST (before the LLM loop) on the
deep-extreme band, where the recipe needs most of the wall and a tail
slot can never fund it; on every other design it runs at the tail from
leftover budget, unbanded; a benefit prediction may schedule work but
never refuse it (measured counter-example: a design three orders of
magnitude outside the band still gained ~+94 MHz from regeneration).
Affordability is priced by a measured s/cell place+route model.
Evidence: +35.47 vs +13.32 α on the biggest calibration design;
+100.23 vs +6.45 on another.

**B2, Retiming tail.** `phys_opt AlternateFlowWithRetiming` on B1's
result, in a fresh session for very large designs (peak-memory
protection). B1's checkpoint is written *before* B2 starts, so B2 can
only cost wall, never the banked gain; it runs whenever a slice ≥ 0.20×
the measured place+route cost remains. Evidence: the tail was worth
~6.5 MHz on a scored-class benchmark when an over-cautious predictive
gate was removed.

**B3: Small-floor sibling.** A third leg for small, floor-bound
mid-band designs: `place ExtraNetDelay_low` → `phys_opt
AggressiveExplore` → `route AggressiveExplore` from pristine. Converts a
rare 2-in-21 ILS lottery draw into a deterministic banked candidate
(~150–160 s on the evidence class). Two admission gates keep it null
elsewhere: cost cap 550 s sized from B1's just-measured place+route, and
a **physics admission** (`optimizer/logic_floor.py`) that asks the
design, not the stopwatch, only a solve whose worst paths are
hard-macro-dominated at a near-floor bound admits. When B3 is adopted
and the logic-floor certificate then fires, the run finalizes
immediately (the B3-floor exit). B3 followed by the certificate is how the
agent produces a banked result early, without spending the rest of the hour
looking for one it has already bounded.

### Tail moves (T1–T3)

**T1, Tail-controller portfolio.** Home:
`optimizer/tail_controller.py`. After the main loop, deep-WNS designs
(best ≤ −1.0 ns) run a portfolio of four proven moves ranked by
measured Δns per second: `m1` bare `route_design` (prior +0.404 ns),
`m4` a granular phys_opt ladder (`critical_pin_opt`, `placement_opt`,
`restruct_opt`, `critical_cell_opt`; +0.431 ns), `m3` `phys_opt
AggressiveFanoutOpt` (+0.369 ns), `m2` `phys_opt AggressiveExplore`
(+0.595 ns). A move that stops paying is retired; an accepted move
**un-retires** the others, chain evidence shows moves re-bite after
another move changes the state (+0.726 ns cumulative over a
three-move chain). Deep states harvest to the wall: the measured
gradient (~0.1 ns / 15 min) out-earns the γ cost. phys_opt moves bank
only through a hold-checked gate; unmeasurable hold rejects.

**T2, Bare re-route compounding loop.** Home:
`optimizer/polish_ladder.py:_run_bare_reroute_polish`. Plain `route_design` on the
routed banked best, iterated while each pass gains ≥ 0.020 ns (max 4
iterations). On deep-WNS states bare re-routing *compounds*, a second
pass on one calibration state gained another +0.404 ns, and decays to
noise near plateaus, which is what the min-gain floor detects.

**T3, Mid-band break rung.** Home:
`dcp_optimizer.py:MIDBAND_BREAK_RUNG_TCL`. From the banked best:
unroute → `phys_opt -retime` → `phys_opt AggressiveExplore` → `route
AggressiveExplore`. Breaks a mid-band modal attractor by retiming the
placed state before an aggressive re-route: deterministic bit-identical
×4 (including one cross-machine repetition), +7.76 MHz over the banked
record on the evidence design; the pure-reroll control shows the retime
steps are load-bearing. Post-loop only (a pre-loop variant measurably
pre-empted better LLM moves), priced at ~1350 s of stranded wall,
MUX-additive, FF latency audit applies.

### Final polish passes (P1–P3)

**P1, Fanout polish.** One `phys_opt_design -directive
AggressiveFanoutOpt` pass on the final best, after the ILS rotation.
Replicates high-fanout drivers; measurably erodes hold, so it carries
the strictest hold floor in the tree (whs ≥ +0.010 ns and never worse
than before) and runs only on cheap designs (measured cycle anchor
< 600 s). Evidence: +0.029 ns (~+6 MHz, ground-truth-confirmed) on a
hold-safe benchmark.

**P2, Last-mile final polish.** The D3 last-mile cycle applied once to
the *final* global best. The rotation only ever runs LASTMILE early in a
seed; the probe evidence (+29/+15 MHz, equivalence-verified) is
specifically on plateaued final states; this stage gives the final
best that exact condition. Entry gate |WNS| ≥ −1.0 ns; accept requires
routed + hold ≥ −0.001 + a cell-count sanity band (its `-lut_opt` can
shrink netlists).

**P3, Winner polish.** Home: `scripts/winner_polish.tcl` (wrapper:
`scripts/multi_restart_optimize.py:winner_polish`). When the remaining
wall is too short for another restart attempt, the wrapper spends it on
up to 4 phys_opt passes (default `AggressiveExplore`) over the shipped
winner in a fresh batch Vivado. The polished file replaces the scored
artifact only on the IMPROVED verdict: strictly better WNS, still fully
routed, hold not made worse. The floor is always the unpolished winner.
Skipped when the physics floor certificate already attested the result.

---

## LLM-layer plays

During the tool loop the LLM sees the full Vivado/RapidWright MCP tool
set plus six **recipe macros**, pre-packaged plays that bundle an
analyze → transform → re-route → measure cycle into one call with
built-in revert. The system prompt (`prompts/system_prompt_scored.txt`)
teaches when to reach for each and enumerates the manual strategy
classes; the deadline-aware dispatcher, route gate, and plan blocks
referee what actually runs.

### The six recipe macros (L1–L6)

**L1, Cell re-placement** (`recipe_cell_replacement`; engine:
`recipes/cell_replacement.py` + `RapidWrightMCP:analyze_net_detour`,
`optimize_cell_placement`). Extracts the worst critical-path pins, runs
RapidWright *detour analysis*, comparing each net's actual routed
length to its straight-line distance, and physically moves the cells
whose nets detour ≥ 2× to free sites nearer their neighbors, then
re-routes and measures. The contest's canonical example recipe.
Acceptance: 0 route errors and positive ΔFmax, else the result is
discarded downstream. Evidence: ~+76 MHz on the organizers' example
design; the recipe refuses to run on design profiles flagged harmful.

**L2, LUT-cone optimization** (`recipe_lut_optimization`; engine:
`RapidWrightMCP:optimize_lut_input_cone`). Finds cascaded small LUTs
feeding the worst paths and merges them into fewer, larger LUTs,
shortening the logic chain by one or more levels: then writes, reopens,
re-routes, measures. For LUT-bound critical paths.

**L3, Register retiming** (`recipe_register_retiming`). One
`phys_opt_design` call with a retiming directive
(`AlternateFlowWithRetiming` aggressive, `AddRetime` conservative):
Vivado moves registers across combinational logic to balance path
delays. Re-routes and measures; the go-to play when placement-level
moves have plateaued near target frequency.

**L4, Granular phys_opt sweep** (`recipe_post_route_phys_opt_sweep`).
Runs one phys_opt sub-optimization at a time, `critical_cell_opt`
(replicate critical cells), `equ_drivers_opt` (duplicate equivalent
drivers), `placement_opt`, `dsp_register_opt`, `restruct_opt`,
`slr_crossing_opt`, measuring after each; improvements ≥ 0.010 ns
commit, regressions revert via the banked mirror. Each sub-flag is
minutes where a full directive is tens of minutes: the budget-tight
substitute.

**L5, Scoped phys_opt** (`recipe_critical_path_focused_phys_opt`).
Creates a Vivado path group containing only the worst N (default 20)
endpoints and runs one phys_opt sub-flag restricted to that group, the
same transform at a fraction of the cost on huge designs, leaving budget
for a second layer.

**L6, High-fanout replication**
(`recipe_high_fanout_timing_replication`). Detects high-fanout nets
(default ≥ 100 loads) on the worst paths and runs `phys_opt_design
-force_replication_on_nets` on the top N: the shared driver is cloned so
each copy drives a subset of loads over shorter wires. Reverts on
regression.

### Manual strategy classes (L7–L13)

The prompt's escalation ladder for moves the macros don't package:

**L7, Full re-placement (Class G).** The LLM's own version of the
Class-G play: `place_design -unplace`, then `place_design` with an
escalating directive list (`Auto_1` → `Explore` → `AggressiveExplore` →
congestion/net-delay variants), route, retime polish. Reserved for
confirmed plateaus; it can regress a strong result. Evidence quoted to
the model: +85 MHz on the design class that motivated it; a router
FALLBACK run where the LLM chose this play produced the campaign's
single biggest per-design gain.

**L8: Pblock re-placement (Class C).** Measures utilization, has
RapidWright compute a compact rectangular fabric region that fits the
design at ~1.5× resources, converts it to a pblock constraint, unplaces,
and re-places the whole design inside the tighter box, shorter wires by
construction. Gated to spread-out designs (avg spread > 70 tiles) and
flagged in-prompt as catastrophic if mis-applied (a wrong pblock can be
unroutable; a full re-place without the pblock empirically beats it on
some classes).

**L9, Fanout split (Class D manual).**
`rapidwright_optimize_fanout(net, split_factor)`, split one named
high-fanout net's driver into replicated drivers (factor ~3–8 by
fanout), then reopen in Vivado and re-route.

**L10, Surgical phys_opt flags.** Single-shot scalpel moves:
`-critical_pin_opt` (remap logical LUT inputs onto physically faster
pins: nearly free, often +0.5–2 MHz), `-clock_opt` (post-route clock
skew balancing), `-force_replication_on_nets` (targeted driver cloning
after a route names its critical nets), `-tns_cleanup
-slr_crossing_opt` (total-negative-slack cleanup).

**L11, Route-directive escalation (Class I).** When `Default` routing
leaves errors or gives back post-place WNS: re-route with `Explore`,
then `AggressiveExplore`, `NoTimingRelaxation`, `MoreGlobalIterations`,
`HigherDelayCost`, or `AlternateCLBRouting` for congestion; `-tns_cleanup`
when a phys_opt pass follows.

**L12, BRAM/DSP relocation (Class F).** The organizers' macro-move
technique, realized through the L1 cell-move play targeted at BRAM/DSP
cells whose critical path crosses a die region.

**L13, ML strategy recommender (Class H)** (`report_qor_suggestions`).
Vivado's own ML recommender, run post-place for clocking/congestion/
strategy suggestions, applied via the `RQS` directives. Cheap (1–2 min)
and occasionally surfaces a directive the sweeps miss.

---

*Where a play's evidence names a specific gain, the number comes from
the measurement record quoted at the play's code home; development-gate
numbers are not comparable to the official hidden-benchmark scores (see
CONFIGURATION.md "Honesty notes"). Flags and defaults for every play are in
[CONFIGURATION.md](CONFIGURATION.md); which plays were active in the
scored run is pinned in [PROVENANCE.md](PROVENANCE.md).*

---

## How three techniques became forty-two plays

The starter kit's three techniques are seeds, not the library. The rest came
out of a loop the code runs on itself:

1. **The model explores.** On a development design it drives the 34 tools
   freely, sequences nobody scripted.
2. **The run attributes its own gains.** `winning_tools_from_call_details()`
   walks the call log, tracks the running best WNS, and attributes every
   improvement to the last transformative call before it. A run therefore
   ends knowing *which move earned each nanosecond*, not just that it
   improved. This is recency credit assignment: a gain produced by two moves
   in sequence is credited entirely to the second, and every boundary below
   inherits that bias. Read them as counted under a stated rule, not as
   ground truth.
3. **The corpus answers when, and when not.** Those attributions accumulate
   across the campaign into `strategy_memory` and, for the moves that lost,
   `negative_memory`. Mining them is what produces a play's *boundary*,
   partial ruin at 26/26 negative on a co-located critical path against 134–5
   positive on a spread one; the route re-roll's bite decaying with depth.
   A boundary is not something you can guess; it has to be counted.
4. **The survivor is frozen as a play**: with its trigger, its exclusion and
   its measured cost, and enters this document.
5. **The model calls it back.** At runtime the same memory seeds iteration 1
 (retrieved by fingerprint distance, never by design name) and the LLM
   picks among the frozen plays instead of rediscovering them at $1 an hour.

That loop is the reason the library is not a list of things that once worked.
Exploration is expensive and happens once, in development; the contest run
spends its budget executing what exploration already paid for.

**What is new here is not the plays; it is their boundaries.** Ruin-and-
rebuild, re-placement and retiming are standard moves; what
this document adds is a measured trigger and a measured
*exclusion* for each of the deterministic ones, mined from a corpus rather
than guessed. Partial ruin
is 26/26 negative when the critical path is spatially co-located and 134 wins
to 5 losses when it is spread, so it is gated on spread < 30 tiles. The route
re-roll's bite decays with depth (+0.070 ns at −0.31 ns, flat at −0.59,
negative at −0.84), so it is fenced to |WNS| ≤ 0.7. Knowing when a play stops
paying is what makes a portfolio affordable inside one hour.
