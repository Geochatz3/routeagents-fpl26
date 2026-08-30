# optimizer/

The pipeline stages and the mechanisms they use. Everything here is
importable and testable without spawning Vivado or RapidWright, which is why
it lives outside `dcp_optimizer.py`.

## Stage mixins

Four modules are **mixins** on `DCPOptimizer` — cohesive groups of methods
carved out of the orchestrator, composed back onto the class, and reached
through it as before:

| File | Stage |
|---|---|
| `recipe_passes.py` | deterministic \|WNS\|-band tool sequences, before any LLM call |
| `tool_dispatch.py` | `call_tool`, the budget and unroute gates, auto-banking |
| `polish_ladder.py` | ILS ruin-and-rebuild, the polish stages, the exit tail |
| `finalization.py` | the candidate play-off, mirrors, lifecycle status |

## Mechanisms

The rest hold no optimizer state, grouped by what they decide:

| Area | Files |
|---|---|
| Routing a design to a play | `recipe_router.py`, `recipe_policy.py`, `pathology.py` |
| Physics and stopping | `logic_floor.py`, `route_gate.py`, `tail_controller.py` |
| Wall and cost economics | `wall_economics.py`, `config_resolution.py` |
| The LLM loop | `llm_runtime.py`, `tool_source.py`, `api_resilience.py`, `tool_errors.py`, `plan_critic.py` |
| Re-place and polish | `deep_replace_sibling.py`, `ils_polish.py`, `replace_gamble.py` |
| Measuring and parsing | `phase1_sense.py`, `static_parsers.py`, `qor_parsing.py`, `qor_json_features.py` |
| Shipping | `finalize_mux.py`, `path_guard.py`, `constraint_guard.py` |
| Prompt shaping | `policy_card.py`, `cross_model_steering.py` |
| Memory and observability | `strategy_memory.py`, `negative_memory.py`, `policy_memory.py`, `decision_tracer.py`, `gate_log.py`, `episode_qor_join.py` |

**Five of these ship default-OFF** and ran in no scored benchmark, so no
result should be attributed to them. Each is *reachable* — a caller invokes it
behind a flag or a switch — which is what distinguishes them from dead code:

| Module | What it waits on |
|---|---|
| `gate_log.py` | `FPL26_GATE_LOG` |
| `cross_model_steering.py` | `ENABLE_CROSS_MODEL_STEERING` (registry also empty on `main`) |
| `plan_critic.py` | `--plan-critic` |
| `policy_card.py` | `--policy-card` |
| `replace_gamble.py` | `--replace-gamble` |

Two more were default-OFF *and* had no caller outside their own tests. They
were deleted from `main` rather than documented — see
[docs/PROVENANCE.md](../docs/PROVENANCE.md).

Note the trap that `replace_gamble` (re-places from the *banked best*, off) is
a different mechanism from `deep_replace_sibling` (re-places from the
*untouched input*, on) — see
[docs/CONFIGURATION.md](../docs/CONFIGURATION.md).

[docs/PROVENANCE.md](../docs/PROVENANCE.md) records which of these were
extracted from the orchestrator, which were always separate, and which ran in
the scored configuration. `test_strategy_memory.py` sits beside the code it
covers; every other test is under `tests/`.

## Reading `dcp_optimizer.py`

Start from the stage table in the top-level README: four of the six stages
live here as mixins, so the pipeline reads module-by-module rather than
top-to-bottom through one file. What remains in `dcp_optimizer.py` is the run
state every stage shares, the wiring that composes the mixins, Phase-1
orchestration, and the LLM completion path — the parts interleaved with
session calls and the wall ledger, which is why they stayed. Its module
docstring maps what moved where.

## Strategy memory

`strategy_memory.py` is the one mechanism with a life of its own, so it is
worth a paragraph here rather than a lookup in `docs/METHODS.md`.

Every finished run appends a `RunRecord` to `./strategy_memory.jsonl`, so the
memory rebuilds itself on your own designs after a few runs. Records are read
back at iteration 1 and injected into the user message as
`PRIOR-CAMPAIGN HISTORY`. Retrieval is **fingerprint-only** — the closest
design by `|Δlut|/10k + |Δspread|/50`, falling back to a global aggregate so
even a wholly unknown design gets some prior. A design's *name* is never a
lookup key, which is what keeps the mechanism honest on a hidden benchmark.

Sources are consulted in order: `$STRATEGY_MEMORY_PATH`, then
`./strategy_memory.jsonl`, then `data/seed_memory.jsonl` — the bundled dev
history, which is absent from `main` and present at the provenance tag (see
[data/README.md](data/README.md)). Regressions are persisted with
`completed=False`: kept for analysis, filtered out of retrieval.

```bash
python -m optimizer.strategy_memory --query --lut-count 24000 --spread 42
python -m optimizer.strategy_memory --list     # designs by best ΔFmax
python -m optimizer.strategy_memory --count    # records + source files
python -m optimizer.strategy_memory --verify   # schema check, exit 0/2
```

`--no-rag-seed` on the optimizer CLI disables the read while still persisting
at the end, which is how the seed's contribution was A/B tested.
