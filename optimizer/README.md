# optimizer/ — RAG strategy memory

Helper modules for the LLM-driven optimizer.  Kept separate from the
monolithic `dcp_optimizer.py` so each helper can be unit-tested without
spawning Vivado/RapidWright.

## What lives here

| File | Purpose |
|---|---|
| `strategy_memory.py` | Load, query, format, and persist run records.  Loaded by `dcp_optimizer.py` at iter 1 and after each run completes. |
| `data/seed_memory.jsonl` | Bundled 26-record campaign history.  Ships with the submission so the eval env has a useful prior without depending on `/mnt/d/`. |
| `test_strategy_memory.py` | Unit tests — pure offline, no Vivado. |

## Strategy memory pipeline

```
              ┌──────────────────────────────────────┐
              │      RunRecord JSONL                 │
   write ◀──  │  optimizer/data/seed_memory.jsonl    │  ──▶ read
              │  ./strategy_memory.jsonl  (growing)  │
              │  $STRATEGY_MEMORY_PATH    (user)     │
              └──────────────────────────────────────┘
                          │                  ▲
                          │ load_memory()    │ append_run()
                          ▼                  │
                  ┌───────────────┐   ┌───────────────┐
                  │ retrieve      │   │ persist       │
                  │ best_for_     │   │ _persist_to_  │
                  │   design()    │   │   strategy_   │
                  │ top_n_for_    │   │   memory()    │
                  │   design()    │   │ (end of run)  │
                  │ fingerprint_  │   └───────────────┘
                  │   match()     │
                  │ global_       │
                  │   aggregate() │
                  └───────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ format        │
                  │ format_for_   │
                  │   prompt      │
                  │ format_top_n_ │
                  │   for_prompt  │
                  │ design_note_  │
                  │   for         │
                  └───────────────┘
                          │
                          ▼
            Injected into iter-1 user message
            as "PRIOR-CAMPAIGN HISTORY" + "DESIGN-SPECIFIC NOTE"
```

## Retrieval priority order

`seed_prompt_for(design_name, lut_count, spread, top_n=3)`:

1. **Top-N by name** — `top_n_for_design(memory, name, n=3)`.
   Deduped by candidate class (collapses `v0_3`, `v0_3_seed2`, `v0_3_seed3`
   into one row, keeping the highest ΔFmax of the class).
2. **Fingerprint fallback** — `fingerprint_match(memory, lut_count, spread)`.
   Returns the closest design by `|Δlut|/10k + |Δspread|/50` distance.
3. **Global aggregate** — `global_aggregate(memory)` produces a one-line
   summary so even truly unknown designs get *some* prior.

Then `design_note_for(name)` appends a curated note (DESIGN_NOTES table)
for known-LOSS / known-tight designs.

## CLI

```bash
# Inspect what the iter-1 snippet looks like for a design
python -m optimizer.strategy_memory --query amd_mini-isp

# List every design with at least one record, sorted by best ΔFmax
python -m optimizer.strategy_memory --list

# Show record count + source files
python -m optimizer.strategy_memory --count

# Validate every record against schema (CI-friendly: exit 0/2)
python -m optimizer.strategy_memory --verify
```

## How runs grow the memory

`dcp_optimizer.DCPOptimizer._persist_to_strategy_memory()` runs at the end
of every optimization run.  It writes a `RunRecord` with:

| Field | Source |
|---|---|
| `design` | derived from input DCP filename via `scheduler.dispatch.design_name_from_dcp` |
| `candidate` | `self.mode` (`v0_3` or `anchor`) |
| `delta_fmax_mhz` | best_fmax − initial_fmax |
| `iterations`, `tool_calls`, `force_continues` | optimizer state |
| `total_cost_usd`, `wall_time_s` | optimizer state |
| `lut_count`, `critical_path_spread` | from initial analysis (rapidwright_get_design_info + spread analysis) |
| `winning_tools` | `winning_tools_from_call_details()` derives the transformative tool chain that produced WNS improvements |
| `completed` | True iff delta > 0 (filtered out of retrieval but kept in raw data) |

Regressions are persisted but marked `completed=False` so they don't poison
the retrieved snippet but stay in the raw data for future analysis.

## How the optimizer reads at iter 1

`dcp_optimizer.py:optimize()` calls `seed_prompt_for()` after the initial
analysis is complete (so `lut_count` and `spread` are populated).  The
snippet is appended to the iter-1 user message under the
"PRIOR-CAMPAIGN HISTORY" header, with the curated design note (when
present) right after.

`--no-rag-seed` (CLI) disables the read.  Persistence still happens at
the end — useful for A/B testing the RAG seed's contribution.
