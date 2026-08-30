# Methods and heuristics catalog

One subsection per mechanism, with file pointers and the load-bearing
constants. Flags and defaults are in [CONFIGURATION.md](CONFIGURATION.md);
behavioral differences from the scored tag are in
[PROVENANCE.md](PROVENANCE.md). This file covers the
*control* mechanisms; the optimization moves themselves are cataloged in
[PLAYBOOK.md](PLAYBOOK.md).

## The pipeline

A checkpoint enters a six-stage deterministic pipeline with the **LLM loop** in
the middle of it, and two layers act on every stage: the **playbook** above,
supplying each stage with the plays its measured features admit, and the
**control layer** below, which every stage answers to before anything is
banked.

The agent (`dcp_optimizer.py`) talks to the vendor tools through two MCP
servers: `VivadoMCP/` exposes Vivado's Tcl flow (place, route, phys_opt,
timing reports) and `RapidWrightMCP/` exposes
[RapidWright](https://www.rapidwright.io/), AMD's open-source framework for
reading and transforming design checkpoints. Vivado does all placement,
routing and timing signoff; RapidWright supplies the fast structural analysis
between Vivado calls (LUT counts, critical-path spread, net topology) that the
fingerprinting and admission gates depend on.

| Path | What |
|---|---|
| `dcp_optimizer.py` | The agent's run state and the loop that sequences the stages: Phase-1 measurement and the LLM completion path |
| `optimizer/` | The stages and mechanisms: `recipe_passes`, `polish_ladder`, `tool_dispatch`, `finalization`, `recipe_router`, `logic_floor`, `wall_economics`, `deep_replace_sibling`, `ils_polish`, `strategy_memory`, … |
| `scripts/multi_restart_optimize.py` | Best-of-N entry point: restart legs + progressive banking |
| `recipes/`, `scheduler/` | Standalone recipe implementations; portfolio scheduling |
| `VivadoMCP/`, `RapidWrightMCP/` | MCP servers giving the LLM Vivado/RapidWright tools |
| `prompts/` | System prompts (which one the scored run used is documented there) |
| `validate_dcps.py` | Equivalence validation harness (from the contest starter kit) |
| `scripts/ship_config.py`, `scripts/check_docs.py` | Configuration audit; Markdown link and anchor check |
| `tests/` | Offline test suite: `make test` runs 2425 tests and 87 subtests |

### The six stages, and where each one lives

These six names are used unchanged in the code, the documentation and the
write-up, so a stage can be followed across all three.

| Stage | What it does | Code |
|---|---|---|
| 1 · SENSE | measure first, no LLM: timing, path anatomy, utilization, a fingerprint, and this box's own tool timings | `optimizer/static_parsers.py`, `optimizer/phase1_sense.py` |
| 2 · DECIDE | route the design to a recipe class from measured features only, never from its name | `optimizer/recipe_router.py`, `optimizer/recipe_policy.py` |
| 3 · RECIPE PASS | deterministic, LLM-free tool sequences; may finish before any LLM call | `optimizer/recipe_passes.py` |
| 4 · LLM LOOP | the model proposes moves through 40 tools (34 served by the two MCP servers, 6 recipe macros the agent registers itself) and a constraint-guarded Tcl console | `optimizer/llm_runtime.py`, `optimizer/tool_dispatch.py`, `optimizer/tool_source.py`, `optimizer/api_resilience.py` |
| 5 · POLISH LADDER | ILS ruin-and-rebuild, one insured re-place retry, one extra routing push | `optimizer/polish_ladder.py`, `optimizer/ils_polish.py`, `optimizer/deep_replace_sibling.py` |
| 6 · FINALIZE | the candidate play-off, re-validation, and publishing the winner | `optimizer/finalization.py`, `optimizer/finalize_mux.py`, `scripts/multi_restart_optimize.py` |

Stage 2 is what makes the other five design-specific. The three mechanisms
that cut across all six are below.


### The governors

They earn no MHz themselves. They are what lets the plays run at full
aggression inside a fixed budget.

| | What it does | Code |
|---|---|---|
| **The bank** | a worse attempt never replaces the best, and every risk forks from a pristine copy, so a failed gamble costs wall time and nothing else | `optimizer/deep_replace_sibling.py`, `scripts/multi_restart_optimize.py` |
| **The priced clock** | the stop rule is algebra on `α·(1 − 0.1·(β + γ))`, not a timer: the next second must out-earn the tax it adds | `optimizer/wall_economics.py` |
| **Valid-state admission** | a result is banked only if its timing was measured on a design that is still fully routed. An unrouted design reports optimistic timing, so a reading taken there is not evidence of anything. | `optimizer/route_gate.py`, `optimizer/logic_floor.py` |

`optimizer/logic_floor.py` also carries the delay-bound halt: it computes a
lower bound on the achievable critical-path delay from per-hop net-delay floors
and stops when the banked result is within measurement noise of it. Two
independent timing measurements must agree before it fires, and it fails open:
anything unmeasurable means "keep optimizing".

## The logic floor

`optimizer/logic_floor.py`

The same attestation is used in two places and has been called by two names.
Both names are kept because they answer different questions:

| Name | Question | Effect |
|---|---|---|
| **floor exit** (the "irreducible-path stop") | is the banked result already at the bound? | ends the run early |
| **B3 physics admission** | is this design's worst path the kind a re-place can move? | admits or blocks the B3 leg ([PLAYBOOK.md](PLAYBOOK.md) B3) |

Attests that a design's WNS is bounded by logic depth (not routing), so
further routing effort cannot pay. A design qualifies only when all of
the following hold on the top-32 setup paths
(`LOGIC_FLOOR_NPATHS = 32`):

- optimistic re-bound with a net floor of 45 ps/hop
  (`LOGIC_FLOOR_NET_FLOOR_NS_PER_HOP = 0.045`) shows at most 10 MHz of
  headroom (`LOGIC_FLOOR_MAX_BOUND_ALPHA_MHZ = 10.0`);
- worst-path logic fraction ≥ 80% (`LOGIC_FLOOR_MIN_LOGIC_FRAC = 0.80`)
  and macro-of-logic fraction ≥ 50% (`LOGIC_FLOOR_MIN_MACRO_FRAC = 0.50`);
- two independent timing solves agree within 0.08 ns
  (`LOGIC_FLOOR_TWO_SOLVE_MAX_DELTA_NS = 0.08`);
- the 32-path window covers the near-critical population
  (`LOGIC_FLOOR_COVERAGE_MARGIN_NS = 0.100`).

Fail-open: any attestation error means "no certificate" and the run
keeps optimizing; the certificate can only grant an early exit, never
block work.

## Insured banking (atomic publish of the best artifact)

`optimizer/deep_replace_sibling.py` + the wrapper's publish path
(`scripts/multi_restart_optimize.py`)

Any candidate that measures better than the banked best is *banked*:
written to a private store, then atomically published (temp file +
rename, checksum verified). The monotone rule: the published artifact
only ever improves; a later worse draw can never replace it. Publish is
also wired into the SIGTERM path so an externally killed run still
surfaces the best artifact it had already banked.

## Wall economics + handback

`optimizer/wall_economics.py`

Derived (not fitted) from the published scoring function
`score = α · (1 − 0.1·(β + γ_hours))`: continuing is worth it only while
the marginal improvement rate exceeds the marginal γ cost,
`α′(t) > α(t) · 0.1 / (3600 · P)` (P = current penalty factor). The stop
decision uses a measured min-heavy-move window: the attempt stops only
when the remaining wall cannot fit the cheapest historically useful
heavy move. Disclosed admission edge: a 1-second tool call participates
in the min() window, so a pathologically short measured move can make
the window optimistic; the guard is deliberately fail-open toward
continuing. Wall *handback* returns stranded end-of-run wall to the
wrapper, which converts it into score by declining further attempts.

## Predictive admission family

- **Cost gate**, `scripts/multi_restart_optimize.py`:
  `COST_CEILING_DEFAULT = 0.80` (USD): no new attempt starts once the
  spend forecast crosses the ceiling (contest cap $1.00/benchmark);
  in-attempt LLM spend exits at `LLM_COST_EXIT_USD = 0.75`
  (`optimizer/llm_runtime.py`, re-exported from `dcp_optimizer`).
- **Route gate**, `optimizer/route_gate.py`: refuses
  routed-state-destroying reroutes when the *predicted* duration
  (calibrated s/cell rates by design class) does not fit the remaining
  wall; preserving vs destructive reroutes are assessed separately.
- **Truncation gate**, `scripts/multi_restart_optimize.py`:
  `TRUNCATION_FACTOR = 0.75`; a new attempt is skipped unless the
  remaining wall is at least 0.75× the expected attempt duration
  (`W − E ≥ 0.75·E`), so attempts that would be killed mid-flight are
  never started.

## The brief (the WNS-band recipe router)

What the model actually receives, from one run on `rosetta_digit`:

```json
{ "wns_ns": -1.02,
  "fmax_ratio": 0.62,
  "critical_path_spread_tiles": 131.4,
  "max_fanout_net": 1172,
  "blocked": ["critical_path_focused_phys_opt"] }
```

and two of the calls it made against that brief:

```
vivado_phys_opt_design  {"directive": "AlternateFlowWithRetiming"}
vivado_place_design     {"directive": "Explore"}
```

The numbers are the design's own measurements. `blocked` is the router ruling
a play out before the model can reach for it.


`optimizer/recipe_router.py`

Routes each design to a recipe class by measured features only:
|WNS| band, failing-endpoint count, achievable-fmax ratio, and
critical-path spread (RapidWright average cell spread in tiles). No
design-name keys. The band edges and per-band recipes are
evidence-commented at the constants. Band-scoped augmentations (the
sub-band phys_opt floor, the ETO retime candidate, the mid-band
route rung / retry hold, the shallow determinizer) hang off the same
measured |WNS| bands (see CONFIGURATION.md).

## Fingerprint + priors / anti-priors

`optimizer/strategy_memory.py`, `optimizer/negative_memory.py`

A design's fingerprint is (LUT count, critical-path spread). Retrieval
of prior-campaign records is fingerprint-only on `main` (distance
`|Δlut|/10k + |Δspread|/50`), with a global aggregate fallback; the
matched record's winning tool chain and hints seed the iter-1 prompt.
Anti-priors: `negative_memory_block()` surfaces feature-similar episodes
that regressed or wasted budget, as an advisory "what to avoid" block,
it never gates a tool call.

## Tail controller (measured-ns/s portfolio)

`optimizer/tail_controller.py`

After the main loop, deep-WNS designs run a portfolio of four proven moves
ranked by measured Δns per second: a bare `route_design` (`m1_route`), a
four-step phys_opt ladder (`m4_ladder`), `AggressiveFanoutOpt` (`m3_fanout`)
and `phys_opt -directive AggressiveExplore` (`m2_physopt_ae`); the keys are
the ones in `TAIL_MENU` and in the run log. A move that stops paying is
retired; an accepted move *un-retires* the others (moves re-bite after
the state changes). Arms only on deep-WNS states; stops on wall floor,
plateau, or runaway caps. Fail-closed to the simple bare-reroute tail.

## ILS ruin-and-rebuild polish

`optimizer/ils_polish.py`

Iterated local search over routed state: controlled "ruin" (unroute /
targeted displacement) followed by rebuild, with a directive rotation
ladder, per-cycle accept/reject against the banked best, and a
retry-baseline gate (mid-band scoped) that holds the ladder until the
rotation has had a fair chance. Monotone by construction: rejects
restore the banked state.

## Spread-forced third restart

`scripts/multi_restart_optimize.py`

The best-of-N restart wrapper forces a third restart when the first two
attempts' results are tightly clustered (low spread ⇒ likely a modal
attractor, and a fresh draw has positive expected value), provided the
truncation and cost gates admit it.

## Constraint fingerprint attestation

`optimizer/constraint_guard.py`

Fingerprints the design's timing constraints at load and re-attests
before finalize: any tool path that would ship an artifact whose
constraints differ from the input's is refused (protects against
accidental constraint relaxation "improving" WNS).

## API resilience / LLM-dead finalize

`optimizer/api_resilience.py` + `dcp_optimizer.py`

Pure error classifier + deadline-aware backoff: key-level auth errors
(401 storms) are never treated as model-unavailable (which would pin a
useless fallback model sharing the same dead key); backoff is
exponential (15/30/60/120/240 s, per-episode cap 600 s) and wall-aware.
When the LLM is conclusively dead, the run still finalizes: the banked
best artifact is published rather than losing the run.

## Finalize MUX (candidate play-off)

`optimizer/finalization.py` (candidate registration + finalize; the
methods are mixed into `DCPOptimizer`, so they are still reached as
`optimizer.register_final_candidate(...)`)

All speculative candidates (recipe passes, retime candidates, rungs)
are *MUX-additive*: they register into a play-off, never touch the
banked best directly. Finalize picks argmax of measured results,
never-worse by construction. md5-trust: byte-identical candidates skip
re-measurement; a re-validation gate re-measures the chosen winner
before shipping.

## Process-group ownership

`VivadoMCP/vivado_mcp_server.py`

The Vivado MCP server spawns its Vivado subprocess with
`start_new_session=True`, so that subtree gets its own process group and
cleanup can `killpg` the group rather than only the direct child, no
orphaned engine holds a license or burns wall after the run. It is gated
on `FPL26_VIVADO_LEAK_FIX`; with the flag off, Vivado inherits the
server's group and only the child is signalled.

Two things this deliberately does **not** do. The RapidWright JVM is
in-process (JPype), so there is no separate group to own. And
`scripts/multi_restart_optimize.py` keeps its attempts in the wrapper's
own group on purpose: the evaluation harness delivers one group SIGTERM
to wrapper and agent together, and the emergency-finalize path depends
on both receiving it. Liveness is tracked by scanning for a command-line
token (`_pids_with_cmdline_token`); no PID file is written.
