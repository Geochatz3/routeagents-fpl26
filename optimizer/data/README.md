# optimizer/data — strategy memory seed

The contest submission bundled `seed_memory.jsonl` here: a JSONL file of
measured run records from our own dev campaign, used by
`optimizer/strategy_memory.py` as a retrieval memory of what worked per design
class (measured priors, not tuned constants).

That file held per-design results from our own development runs on the 19
public contest benchmarks. Seven of those designs were later chosen for the
final evaluation, so the file does contain priors for designs that were
subsequently scored — measured on our runs, before the final round, without
knowing which seven would be picked. It is removed from `main` regardless:
priors measured on someone else's designs are not a useful starting point for
yours. It remains part of the scored submission and is therefore
visible at the tag `fpl26-final-submission`, whose whole point is byte-exact
provenance.

The mechanism is unchanged. Memory sources are consulted in this order (see
`strategy_memory.py`):

1. `STRATEGY_MEMORY_PATH` (env var, explicit path)
2. `./strategy_memory.jsonl` in the working directory — grows from your runs
3. `optimizer/data/seed_memory.jsonl` — optional bundled seed (absent here)

Record schema (one JSON object per line; all fields optional):

```json
{"design": "…", "candidate": "…", "commit": "…", "exit_code": 0,
 "wall_time_s": 0.0, "initial_wns_ns": 0.0, "final_wns_ns": 0.0,
 "initial_fmax_mhz": 0.0, "final_fmax_mhz": 0.0, "delta_fmax_mhz": 0.0,
 "delta_wns_ns": 0.0, "iterations": 0, "tool_calls": 0,
 "total_cost_usd": 0.0, "completed": true}
```

Every completed run appends to `./strategy_memory.jsonl`, so the memory
rebuilds itself on your own designs after a few runs.
