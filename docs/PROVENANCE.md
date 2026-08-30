# Provenance

If you are here to use the software, you do not need this file. It exists so
the contest artifact can be checked, and so the rest of the repository can be
about optimizing designs.

## The scored tree

The initial commit, tag `fpl26-final-submission`, is the byte-exact tree
submitted to the FPL'26 FPGA optimization contest. Every later commit on `main`
is cleanup on top. The repository was published with a fresh history, so
nothing from the development machines travelled with it.

| | |
|---|---|
| Submitted archive md5 | `0b03890def0f5ed444217b9d3b241af1` |
| Version, submitted | v5.5.3, 2026-08-10T10:54Z |
| Scored commit | `d8ae7bd8f1f8e4f434d21c751fd853f73be7ed96` |
| Scored tree | `4e1402384c3a8bc8e4e6d19bd56792f171ba4610` |
| Files | 239 |

```bash
git rev-parse fpl26-final-submission^{tree}      # -> 4e1402384c3a8bc...

mkdir -p /tmp/scored && git archive fpl26-final-submission | tar -x -C /tmp/scored
cd /tmp/scored && sha256sum -c /path/to/clone/docs/scored_submission.sha256
# 239 lines, every one OK
```

`docs/scored_submission.sha256` is a plain checksum file, so it verifies the
scored bytes for someone holding only the submitted archive. Re-packing the tag
will not reproduce the archive md5: tar and gzip embed ordering and timestamps,
which is what the per-file checksums are for.

## The official evaluation

Seven hidden benchmarks, one run each inside one hour and one dollar, scored
`alpha * (1 - 0.1 * (beta + gamma))` where alpha is delta-Fmax in MHz, beta is
LLM dollars and gamma is wall hours.

| Benchmark | α (MHz) | β ($) | γ (h) | Score |
|---|---|---|---|---|
| amd_mini-isp_v2 | **125.2** | 0.00 | 0.11 | **123.8** |
| rosetta_digit | 92.4 | 0.11 | 0.79 | 84.0 |
| finn_radioml | 65.2 | 0.07 | 0.88 | 59.0 |
| rosetta_3d_v2 | 48.7 | 0.18 | 0.74 | 44.2 |
| fir_symmetric | 37.6 | 0.03 | 0.74 | 34.7 |
| fir_transposed | 21.3 | 0.06 | 0.80 | 19.5 |
| vtr_mcml_v2 | 5.1 | 0.09 | 0.85 | 4.7 |
| **Total** | | | | **369.9** |

These are the organizers' numbers, measured on their hardware. The seven are
drawn from the 19 public benchmarks, so the same inputs are runnable here;
placement and routing are stochastic and the model samples, so the figures are
not reproducible design by design.

## What came from the starter kit

Every team started from the same
[starter kit](https://github.com/Xilinx/fpl26_optimization_contest).

| | Starter kit | Here |
|---|---|---|
| MCP tools | 34 (17 Vivado + 17 RapidWright) | the same 34, unchanged |
| Optimization techniques taught | 3: high-fanout split, pblock re-placement, phys_opt directives | 42 plays; those 3 survive as L8 to L10 |
| Agent | 2,448 lines, three per-benchmark demo paths | 30,336 lines, no design-name keys |

`recipe_router`, `wall_economics`, `logic_floor`, `ils`, `deep_replace`,
`tail_controller`, `strategy_memory` and `multi_restart` appear nowhere in the
starter kit's agent.

## What differs between the tag and `main`

| | Tag | `main` | Effect on the scored behaviour |
|---|---|---|---|
| Strategy-memory retrieval | name-keyed retrieval present | fingerprint-only | **None**: the ship target always passed `--contest-mode`, which already forced feature-first retrieval |
| Cross-model steering | registry keyed by design name | mechanism kept, registry empty | **None**: opt-in via `ENABLE_CROSS_MODEL_STEERING`, never enabled by any ship target |
| Flag names | benchmark-derived | mechanism-based | **None**: constants, band edges and behaviour unchanged; mapping below |
| `dcp_optimizer.py` | one 19,890-line file | split into `optimizer/`, 10,301 lines | **None**: verbatim moves with delegating methods and re-exports |
| `make validate-submission` | present | removed | **None**: it required a `submission/` directory absent from both trees, so it exited 2 wherever it was run |
| `make run_test` | present | renamed `run_no_llm` | **None**: same recipe; it runs the optimizer LLM-free, and never ran tests |
| `optimizer/data/seed_memory.jsonl` | present | absent | see *Disclosures* |
| `optimizer/hidden_fingerprint_card.py` | present | **deleted** | **None**: default-OFF and imported by nothing outside its own test |
| `optimizer/escalation_policy.py` | present | **deleted** | **None**: same |
| `optimizer/pathology_features.py` | present | **deleted** | **None**: its own docstring called it an offline prototype with "no live wiring" |

Flag renames, applied to the ship targets in `main`:

| Tag | `main` |
|---|---|
| `FPL26_FIR_SUBBAND_FLOOR` | `FPL26_SUBBAND_PHYSOPT_FLOOR` |
| `FPL26_VEX2_RETIME_CANDIDATE` | `FPL26_ETO_RETIME_CANDIDATE` |
| `FPL26_MINIISP_RETRY_HOLD` | `FPL26_MIDBAND_RETRY_HOLD` |
| `FPL26_CORESCORE_ROUTE_RUNG` | `FPL26_MIDBAND_ROUTE_RUNG` |
| `FPL26_SPAM_DETERMINIZER_CANDIDATE` | `FPL26_SHALLOW_DETERMINIZER_CANDIDATE` |
| `FPL26_SPAM_ESCALATION_GUARD` | `FPL26_SHALLOW_ESCALATION_GUARD` |

### Corrections made after the scored run

A code review of the published tree in August 2026 found five defects. They are
fixed on `main` and **none of them changes the scored result**: the seven scored
benchmarks ran from the tag, with the defect present. This table exists so the
difference is on the record rather than inferred from a diff.

| Where | Defect | Fix on `main` | Effect on the scored result |
|---|---|---|---|
| `scripts/winner_polish.tcl` | an empty `get_timing_paths -quiet` returned `0.0`, indistinguishable from a real 0 ns slack, so an unmeasurable pass could score as a large improvement and publish over the banked artifact | the empty case returns a sentinel; unmeasurable stops the ladder instead of winning it | **None**: the scored run predates this fix |
| `scripts/multi_restart_optimize.py` | an attempt with no readable cost record was charged $0 unless the cost-ceiling breaker was armed, so `--cost-ceiling 0` blinded the independent `cost_cap` meter too | the estimate is charged regardless of breaker state | **None**: every ship target passes `--cost-ceiling 0.80`, so the blinded path was never on the scored run |
| `scripts/multi_restart_optimize.py` | the B3-floor-exit's `high_spread` guard could never be true, because path spread was only measured under `--spread-aware` and no ship target passes it | spread is measured unconditionally; `--spread-aware` still gates the early-stop floor itself | **None**: the guard was inert before and the floor is still opt-in after |
| `scripts/multi_restart_optimize.py` | per-attempt and polish scratch DCPs in `/tmp` were never unlinked, so back-to-back benchmarks could fill the disk under the artifact publisher | `_discard_scratch_dcp`, pruning in-loop and sweeping at end of run, always after `_atomic_publish` has copied the winner | **None**: housekeeping only |
| `scripts/ab_wall_economics.py` | a failed OFF baseline returned `{}` and still passed the presence check, so every mechanism graded HELPS | the call site skips grading and exits 1; `grade_pair` raises on an empty OFF | **None**: an offline A/B analysis tool, not on any run path |

`tests/test_cost_breaker.py` previously pinned the `--cost-ceiling 0` -> charge
$0 path as "legacy behavior preserved". The git history shows that phrase was
inherited prose from a comment pass, not a recorded decision, and `cost_gate`'s
own docstring contradicts it: breaker-off hands down no budget, so the agent
keeps its default and spend still happens. The test now pins the corrected
behaviour.

Each also has its `FPL26_NO_...` counterpart. Anything not in these tables is
non-behavioural: paths, comments, documentation, and files nothing referenced.
**Reproduce the scored run from the tag, not from `main`.** For the composed
flag truth on any checkout, run `python3 scripts/ship_config.py`; the reference
is [CONFIGURATION.md](CONFIGURATION.md).

## Disclosures

- **The strategy priors were measured on the public benchmarks.** The scored
  run started from priors mined during our own development campaign on the 19
  public development designs (`optimizer/data/seed_memory.jsonl`). The
  organizers scored seven designs drawn from those same 19, so to be explicit:
  we did not know which seven, the priors were measured before the final round
  on runs of our own, and none is a record of an organizer's evaluation run.
  `main` ships without them; fresh installs build their own.
- **Development numbers are not comparable to the official scores.** Figures in
  code comments come from our own 16-design gate on our own hardware.
- **One admission threshold was corrected before submission**, not after the
  hidden benchmarks ran (B3 physics admission, 12 to 18). See
  [CONFIGURATION.md](CONFIGURATION.md).
- **`optimizer/utilization_features.py` is absent by design.** It was missing
  from the scored submission, so the resource-keyed rules ran on their no-data
  fallbacks. `tests/test_import_completeness.py` pins that state.
- **Twenty-one Python files were deleted from `main`**, disclosed rather than
  quiet. Three are named in the table above (`hidden_fingerprint_card.py`,
  `escalation_policy.py`, `pathology_features.py`) and four more are their own
  tests plus one stale review test. The rest are
  `optimizer/episode_validation_join.py`, `scheduler/dispatch_replay.py`, the
  five-file `gepa/` package, `scripts/probe_qor.py`,
  `scripts/rehprep_fanout_glue.py`, `scripts/rehprep_routeonly_glue.py`,
  `scripts/retrofit_edif.py`, `scripts/submission_check.py`,
  `test_validate_dcps.py` and `VivadoMCP/test_vivado_mcp.py`. Every non-test one
  has no caller anywhere in the tag
  (`git grep` on the tag returns nothing outside the file itself), ran in no
  scored benchmark, and `scripts/ship_config.py` reports the same 31 flags
  before and after. All fifteen remain byte-exact at the tag; to see the list
  for yourself:

  ```bash
  comm -23 <(git ls-tree -r --name-only fpl26-final-submission | sort) \
           <(git ls-tree -r --name-only main | sort) | grep '\.py$'
  ```
- **The tag carries the organizers' contest-site files**, about twenty under
  `docs/`: their Jekyll site, an AMD logo and a promo video, unmodified from the
  starter kit. They are there because the tag is the scored tree byte for byte.
  The starter kit is Apache-2.0 (see `NOTICE`), so redistribution is licensed,
  but Apache-2.0 section 6 grants no trademark rights and nothing here implies
  AMD's or the organizers' endorsement.
- **No credential material in any revision**, and none is needed to read or
  test: `make test` runs without Vivado, network or an API key. API keys are
  read from the environment at runtime; nothing in the tree embeds one.

To report a sensitive finding, use GitHub's private security-advisory reporting
on this repository. Anything else, open an issue.
