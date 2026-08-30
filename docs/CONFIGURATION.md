# Configuration, the FPL26_* ship flags

The scored contest run is `make run_optimizer`, which composes **two layers**:
env vars injected by that Makefile target, and in-code defaults read with
`os.environ.get(...)`. **The Makefile layer wins**, several flags whose code
default is OFF ship ON via the injection.

`make run_once` runs one `dcp_optimizer.py` attempt in contest mode and
injects no `FPL26_*` at all, so every flag sits at its in-code default. Use it
to debug a single attempt; it does **not** reproduce the scored configuration.

## Environment variables

| Variable | Needed when | Example |
|---|---|---|
| `OPENROUTER_API_KEY` | LLM-guided modes (`run_optimizer`, `run_once`), get one at [openrouter.ai](https://openrouter.ai/) | `sk-or-v1-…` |
| `VIVADO_EXEC` | Vivado is not on `PATH` (WSL2: the Windows `vivado.bat`) | `/tools/Xilinx/Vivado/2025.1/bin/vivado` |
| `OPENROUTER_BASE_URL` | Using another OpenAI-compatible endpoint | `https://openrouter.ai/api/v1` |
| `FPL26_RUN_DIR_BASE` | Run artifacts should land on a bigger disk | `/data/runs` |

Put them in `.env` (copy `.env.example`; every `make` target loads it) or
export them in the shell. The LLM-free mode (`make run_no_llm`), validation,
and the offline test suite need **no key at all**. The `FPL26_*` flags that
shape a run are catalogued below, under
[Flags injected by the Makefile ship target](#flags-injected-by-the-makefile-ship-target)
and [Flags ON via code default](#flags-on-via-code-default-not-in-the-makefile).
To read the composed truth off any checkout, in-code defaults and Makefile
injection resolved together, run `python3 scripts/ship_config.py`.

### Four counts, four sets

The docs quote four flag totals. They are not the same set, and
`scripts/ship_config.py` is authoritative for all four:

| Count | Set |
|---|---|
| **23** | `FPL26_*` assignments in the `run_optimizer` recipe: 22 unconditional, plus `FPL26_POLISH_RESERVE_S` only when `POLISH_RESERVE_S=` is passed |
| **20** | of those, the boolean flags injected as ON |
| **19** | of *those twenty*, the ones whose in-code default disagrees, so a bare `make run_once` from a clean shell leaves 19 off. The twentieth, `FPL26_ILS_LADDER_ORDER_BY_WNS`, is ON either way |
| **31** | `SHIPS ON`, the 20 above plus the 11 that are ON by code default. This is the number that must match between the tag and `main` |

Two audit tools print the composed truth:

```bash
python3 scripts/ship_config.py      # effective config + layer disagreements
python3 scripts/ship_path_audit.py  # per-flag makefile/code/effective table
```

## Flags injected by the Makefile ship target

| Flag | Ship default | Description |
|---|---|---|
| `FPL26_DEEP_WNS_TAIL_RESERVE` | 2400 s | Wall reserved for the deep-WNS tail; CLI `--deep-wns-tail-reserve` wins; 0 = off; a value in (0,1) is a fraction of `--max-wall-seconds`. |
| `FPL26_DEEP_REPLACE` | 1 | Enables the deep-replace sibling stage: a full re-place from the pristine input on the deep-WNS class, entered as a never-worse MUX candidate. |
| `FPL26_DEEP_REPLACE_FIRST` | 1 | Runs deep-replace BEFORE the LLM loop (it needs most of the wall, so it cannot be a tail supplement). |
| `FPL26_DEEP_REPLACE_UNBANDED` | 1 | The deep-WNS band no longer vetoes deep-replace, affordability plus the insured-compare MUX decide instead. |
| `FPL26_DEEP_FIRST_SIZEGATED` | 1 | Relaxes two mis-scoped gates on deep-replace's FIRST decision for the ILS-size-gated class only; kill switch `FPL26_NO_DEEP_FIRST_SIZEGATED`. |
| `FPL26_DEEP_REPLACE_B3` | 1 | Arms the B3 small-floor deep-replace sibling (mid recipe band, FIRST stage only), behind an affordability cap and a physics admission attestation. |
| `FPL26_B3_FLOOR_EXIT` | 1 | Multi-restart wrapper: a `b3_floor_saturated.token` attesting this attempt's best IS the deterministic B3 floor stops the attempt loop and skips winner polish. |
| `FPL26_LOGIC_FLOOR_EXIT` | 1 | After a B3 small-floor bank, one ~25 s Tcl logic-floor attestation on the live session; if it fires, the LLM loop and polish are skipped and the floor is finalized. |
| `FPL26_ILS_HURDLE_CONTINUE` | 1 | The ILS futility counter can be overridden for one more cycle when the scoring function says the extra cycle pays. |
| `FPL26_ILS_MEASURED_PRIORS` | 1 | Replaces two demonstrably overstated ILS combo cost priors with corpus-measured medians. |
| `FPL26_PHYSOPT_DEFAULT_FIXPOINT` | **0** | (Ships OFF.) Would run a phys_opt Default fixpoint stage after the LLM loop. |
| `FPL26_ILS_LADDER_ORDER_BY_WNS` | 1 | Orders the ILS place-retry ladder by baseline WNS (near-met designs get `AltSpreadLogic_medium` first). |
| `FPL26_RECIPE_PASS` | 1 | Master flag for the frozen-Tcl recipe candidate pass, band-selected from the Phase-1 measured \|WNS\|; `FPL26_NO_RECIPE_PASS` kills it and all sub-candidates. |
| `FPL26_RECIPE_FIRST_DEEP` | 1 | Size-anchored pre-LLM deep-band recipe gate variant on top of `RECIPE_PASS`. |
| `FPL26_SUBBAND_PHYSOPT_FLOOR` | 1 | Sub-band in-session floor on \|wns_in\| ∈ [0.10, 0.45] with ≤ 1000 failing endpoints (pre-LLM pipeline change + carve-out from the shallow recipe pass); kill switch `FPL26_NO_SUBBAND_PHYSOPT_FLOOR`. |
| `FPL26_ETO_RETIME_CANDIDATE` | 1 | Second shallow recipe-pass MUX candidate; its own gate is \|wns_in\| ∈ [0.60, 1.05]. **Dead under the ship configuration**, see the note below. |
| `FPL26_MIDBAND_RETRY_HOLD` | 1 | Retry-baseline gate scoped to the mid band only; kill switch `FPL26_NO_MIDBAND_RETRY_HOLD`. |
| `FPL26_MIDBAND_ROUTE_RUNG` | 1 | Mid-band post-loop route-Explore MUX candidate ("route rung"); kill switch `FPL26_NO_MIDBAND_ROUTE_RUNG`. |
| `FPL26_OWNFRONT_RETIME_CANDIDATE` | 1 | Unified own-front retime candidate: one second shallow candidate whose front is picked per run from the measured \|wns_in\|. |
| `FPL26_MUX_MD5_TRUST` | 1 | Finalize MUX trusts the registration-time re-measure when the winner's md5+size still match, skipping a ~120 s re-open; mismatch falls back to full validation. |
| `FPL26_SHALLOW_DETERMINIZER_CANDIDATE` | 1 | Adds the determinizer chain (place ASL_medium → route Explore → route AE incremental → phys_opt AFWR) as a third shallow candidate on \|wns_in\| ∈ [0.60, 0.90). |
| `FPL26_POSITIVE_SLACK_CONTINUE` | 1 | Keep optimizing when WNS is non-negative instead of declaring victory at the zero crossing. |
| `FPL26_POLISH_RESERVE_S` | 500 s (code) | End-of-wall reserve fencing routed-state-destroying dispatch out so the final post-route polish stays affordable; injected only when `POLISH_RESERVE_S=` is passed to make. |

## Flags ON via code default (not in the Makefile)

| Flag | Default | Description |
|---|---|---|
| `FPL26_ILS_PLACE_RETRY_LADDER` | 1 | Arms the ILS place-retry ladder (`ExtraNetDelay_high`, `AltSpreadLogic_medium`, `ExtraNetDelay_low`). |
| `FPL26_ILS_MEASURED_BASIS` | 1 | Prices an ILS cycle from a real measured full-place cycle wall, not the derived cold-start anchor. |
| `FPL26_ILS_INCR_ROUTE` | 1 | Enables the incremental escalated re-route combo in the ILS rotation. |
| `FPL26_ILS_INCR_ROUTE_FIRST` | 1 | Lets that incremental re-route jump the queue as first escalation, bounded by the displacement guard. |
| `FPL26_ILS_LADDER_SKIP_UNAFFORDABLE` | 1 | An unaffordable ladder rung advances within the cycle instead of ending it. |
| `FPL26_ILS_ENDHIGH_DENSITY_GATE` | 1 | Gates the end-high rung on a density threshold (`FPL26_ILS_ENDHIGH_DENSITY_MIN`). |
| `FPL26_ILS_SEED_COPY` | 1 | Copies the recipe-best into `ils_recipe_seed.dcp` so the banked mirror stays immutable. |
| `FPL26_DEEP_REPLACE_B2_RESTART` | 1 | Restarts the Vivado session before deep-replace leg B2 on memory-risk designs (budget-checked). |
| `FPL26_DEEP_REPLACE_FIRST_BANDED` | 1 | Keeps the FIRST deep-replace stage band-gated (the tail stays unbanded). |
| `FPL26_PREEMPT_LOOP_CLOCK` | 1 | Preempt/stall is measured from when the LLM loop began, not before it existed. |
| `FPL26_VIVADO_LEAK_FIX` | 1 | Vivado runs in its own process group so cleanup can `killpg` the subtree without orphans. |

### A flag that ships ON and never fires

`FPL26_ETO_RETIME_CANDIDATE=1` is injected, and under the shipped
configuration it cannot run. Its gate is |wns_in| ∈ [0.60, 1.05], and exactly
one retime candidate is allowed to spend wall per pass, so it defers across
that whole interval:

| Sub-band | Who takes the slot instead | Also ships ON |
|---|---|---|
| [0.90, 1.05] | `FPL26_OWNFRONT_RETIME_CANDIDATE` | yes |
| [0.60, 0.90) | `FPL26_SHALLOW_DETERMINIZER_CANDIDATE` | yes |

It is kept because the scored tree contained it and its kill switch
(`FPL26_NO_ETO_RETIME_CANDIDATE`) is the A/B handle: turn one of the two
superseding flags off and this candidate takes the band back. `dcp_optimizer.py`
logs `ETO-RETIME: skipped reason=ownfront_supersedes` /
`reason=shallow_det_supersedes` when it stands down, so a run says so.

### One flag not listed above

`FPL26_SHALLOW_ESCALATION_GUARD` (`optimizer/ils_polish.py`) **defaults off and
is injected by nothing**, so it did not run in any scored benchmark. It appears
in the rename table in [PROVENANCE.md](PROVENANCE.md) because it was renamed
from a benchmark-derived name along with the six that do ship; the rename is
the only reason it is mentioned. When enabled it changes ILS escalation
ordering.

### Symbols that kept their old names

The `FPL26_*` flags were renamed away from benchmark names on `main`; the
Python symbols behind two of them were not, because renaming them would have
diverged the code from the scored tree for no behavioural gain. So
`dcp_optimizer.py:_maybe_run_fir_subband_floor` implements
`FPL26_SUBBAND_PHYSOPT_FLOOR`, and the ETO chain's constants keep their
`ETO_*` prefix. The band edges, not the names, are what the router keys on.

## Modules that ship off

Five modules under `optimizer/` are shipped but **default-OFF**, so no result
should be attributed to them: `plan_critic`, `cross_model_steering`,
`replace_gamble`, `gate_log`, `policy_card`. Each is reachable, a caller
invokes it behind a flag or a CLI switch; and each carries a header saying it
did not run. `python3 scripts/ship_config.py` prints the composed truth for any
checkout.

Two further modules that were default-OFF **and had no caller at all** were
deleted from `main`; see [PROVENANCE.md](PROVENANCE.md). Unreachable code is
not a feature behind a flag, and a public repository should not have to explain
the difference.

Two of them re-place a design and are easy to confuse:

| | Seeded from | Ships |
|---|---|---|
| `deep_replace_sibling` | the **untouched input**, the "insured re-place" | **on** |
| `replace_gamble` | the **banked best**, a different mechanism | **off** |

## The model

`DEFAULT_MODEL = "x-ai/grok-4.3"` (`dcp_optimizer.py`) is what the scored runs
drove. It is a CLI parameter, not an environment flag:
`dcp_optimizer.py --model <openrouter-id>` accepts any OpenAI-compatible id.

`FALLBACK_MODEL = "google/gemini-3.1-flash-lite"` is a robustness net, not a
second opinion. The agent pins it only when the primary is conclusively
unreachable on the key in use: a 404, a deprecation, a model-scoped key, or a
run of consecutive transient failures; so one model outage cannot zero every
benchmark. It was never the primary in any scored run. The plan critic, which
ships default-OFF, derives its model from the same constant.

## Other environment

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | LLM access for full agent mode. Every `make` target loads `.env` at the repo root (copy `.env.example`); direct `python3 dcp_optimizer.py` calls read only the real environment (or `--api-key`). Never commit `.env`. |
| `VIVADO_EXEC` | Path to the Vivado executable (or Windows `vivado.bat` under WSL2). |
| `FPL26_WSL2_WIN_CWD` | WSL2 only: Windows-visible working directory for the Vivado wrapper (default `/mnt/c/`). |
| `FPL26_PROMPT_V2` | `1` loads `prompts/system_prompt_v2_experimental.txt`. **No ship target sets it, the scored run used `prompts/system_prompt_scored.txt`.** |
| `STRATEGY_MEMORY_PATH`, `FPL26_SEED_PORTFOLIO` | Optional strategy-memory sources (see `optimizer/data/README.md`). |
| `FPL26_RUN_DIR_BASE` | Base directory for run artifacts (defaults under the repo). |

## Honesty notes

- Many flags carry "DEFAULT OFF" docstrings while the ship target injects
  them ON: the code default was the safety posture during development; the
  Makefile injection is the shipped decision. `scripts/ship_config.py`
  exists precisely because these two layers once disagreed silently.
- `FPL26_PHYSOPT_DEFAULT_FIXPOINT` is injected as `0`, present on the ship
  command line but OFF.
- The wall-economics observation window is `min()` over measured heavy-move
  durations, but the admission filter accepts any call ≥ 1 s whose command
  string matches a risky-operation pattern, so a cheap 1 s call can become
  the window and make the stop rule fire earlier than a "cheapest heavy
  move" reading suggests. A stricter admission fix was built and tested
  during the final round and **dropped** (it lost score); the shipped
  behavior is the scored behavior, disclosed here.
- One admission threshold (the B3 physics admission, 12 → 18) was corrected
  after cloud eval-parity testing exposed a cross-platform discrepancy. It is
  gate-corrected, not benchmark-fitted, and it is **in the scored artifact**,
  changed before the submission that was evaluated, not after the hidden
  benchmarks ran. Disclosed here rather than claimed as "no tuned parameters".
