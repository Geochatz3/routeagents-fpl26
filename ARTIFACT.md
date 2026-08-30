# What you can check, and how

Claims in this repository fall into two buckets.

1. **Checkable on this tree**: a command below runs it, and this file says
   what a pass looks like. Timings are from an 8-core x86-64 Linux box.
2. **Measurement records**: the per-play MHz figures in
   [docs/PLAYBOOK.md](docs/PLAYBOOK.md) and in code comments. Those come from
   our own development gate on our own hardware. They are evidence for *why a
   play exists*, they are not re-runnable here, and they are not comparable to
   the official table. None of them has a row below.

This file is bucket 1 only. The contest score is in neither: not because the
inputs are secret (all seven scored designs are in the public benchmark set)
but because the runs were the organizers', on their hardware, and the flow is
stochastic. You can run the same designs; you will not land on the same
numbers.

---

## Without Vivado, without an API key, without network

`pip install -r requirements-dev.txt` is the only setup these need.

| Claim | Command | Pass looks like | Time |
|---|---|---|---|
| The code is tested | `make test` | `2509 passed, 87 subtests passed` | ~90 s |
| Termination is a measured bound, not a timer | `pytest tests/test_logic_floor.py -q` | `29 passed` | 1 s |
| The stop rule is algebra on the score formula | `pytest tests/test_wall_economics.py -q` | `16 passed` | 1 s |
| A regressing candidate never replaces the banked best | `pytest tests/test_deep_replace_sibling.py -q` | `50 passed` | 2 s |
| The best attempt is published inside the restart loop | `pytest tests/test_multi_restart.py -q` | `76 passed, 5 subtests` | ~45 s |
| One module was missing from the scored run | `pytest tests/test_import_completeness.py -q` | `21 passed` | 2 s |
| Configuration composes as documented | `pytest tests/test_ship_config.py -q` | `9 passed` | 1 s |
| The agent exposes 40 tools (17 VivadoMCP + 17 RapidWrightMCP + 6 recipe macros the agent registers itself) and both server READMEs list theirs | `pytest tests/test_tool_surface.py -q` | `4 passed, 4 subtests` | 1 s |
| Every link in these docs resolves, `#anchors` included | `python3 scripts/check_docs.py` | `0 broken links` | 1 s |
| `optimizer/README.md` maps every module in `optimizer/` | `pytest tests/test_module_map.py -q` | `4 passed, 5 subtests` | 1 s |

## The archive, and that `main` still ships what was scored

| Claim | Command |
|---|---|
| The tag is the byte-exact scored tree | `git rev-parse fpl26-final-submission^{tree}` → `4e1402384c3a8bc8e4e6d19bd56792f171ba4610` |
| …file by file | see the block below → 239 lines, every one `OK` |
| The composed configuration | `python3 scripts/ship_config.py` → 31 flags under `SHIPS ON` |

```bash
REPO=$PWD
mkdir -p /tmp/scored && git archive fpl26-final-submission | tar -x -C /tmp/scored
(cd /tmp/scored && sha256sum -c "$REPO/docs/scored_submission.sha256") | grep -c ': OK$'
# -> 239
```

The checksum file lives on `main`, not at the tag: it is a post-hoc record of
the scored bytes, so it verifies them even for someone holding only the
submitted archive.

`main` renamed six flags away from benchmark names (the mapping is in
[docs/PROVENANCE.md](docs/PROVENANCE.md)). Folding those renames, the set of
flags that ship on is **identical** in the two trees:

```bash
git worktree add --detach /tmp/scored_tag fpl26-final-submission

shipson() { python3 scripts/ship_config.py "$@" \
  | sed -n '/^=== SHIPS ON/,/^$/p' | grep '^  FPL26' | tr -d ' ' | sort; }

shipson > /tmp/on_main.txt
shipson --root /tmp/scored_tag \
  | sed -e s/FPL26_FIR_SUBBAND_FLOOR/FPL26_SUBBAND_PHYSOPT_FLOOR/ \
        -e s/FPL26_VEX2_RETIME_CANDIDATE/FPL26_ETO_RETIME_CANDIDATE/ \
        -e s/FPL26_MINIISP_RETRY_HOLD/FPL26_MIDBAND_RETRY_HOLD/ \
        -e s/FPL26_CORESCORE_ROUTE_RUNG/FPL26_MIDBAND_ROUTE_RUNG/ \
        -e s/FPL26_SPAM_DETERMINIZER_CANDIDATE/FPL26_SHALLOW_DETERMINIZER_CANDIDATE/ \
        -e s/FPL26_SPAM_ESCALATION_GUARD/FPL26_SHALLOW_ESCALATION_GUARD/ \
  | sort > /tmp/on_tag.txt

diff /tmp/on_tag.txt /tmp/on_main.txt && echo "identical -- 31 flags"
```

This checks the shipped configuration, not the binaries: it says `main` would
run a design under the same flags the scored tree did.

## With Vivado

`make setup` fetches the 19 public contest benchmarks,
[release v1.2.0](https://github.com/Xilinx/fpl26_optimization_contest/releases/tag/v1.2.0),
pinned in the Makefile because it is the set the scored runs were developed
against. (The organizers have since published
[v1.3.0](https://github.com/Xilinx/fpl26_optimization_contest/releases/tag/v1.3.0).)
Seven of these 19 are the designs the final round scored, so the inputs behind
the README's table are downloadable; what is not repeatable is the run.

Four different counts of "our designs" appear across these docs, and they are
four different things:

| | |
|---|---|
| **19** | the public benchmarks the organizers released. `make setup` fetches all of them |
| **16** | the gate we ran end to end before each submission, the source of every development figure quoted anywhere in this repository |
| **13** | the subset with per-design decision logs complete enough to attribute a shipped result to one play, which is the denominator for the per-play evidence in docs/PLAYBOOK.md |
| **12** | the subset for which the organizers also published baseline-agent figures. We make no comparison against them here: it would not be a head-to-head on one machine |

The seven scored designs are drawn from the 19, so they are inside the first
row and partly inside the others. What we did not have was the knowledge of
*which* seven would be scored.

| Claim | Command | Time |
|---|---|---|
| It runs end to end without an LLM | `make run_no_llm DCP=fpl26_contest_benchmarks/logicnets_jscl_2025.1.dcp` | ≤ 1 h |
| It runs end to end as scored | `make run_optimizer DCP=fpl26_contest_benchmarks/logicnets_jscl_2025.1.dcp` | ≤ 1 h, ≤ $1 |
| The output is logically equivalent to the input | `make validate GOLDEN=<in>.dcp REVISED=<out>.dcp` | minutes |

**A run is not expected to reproduce a number.** Placement and routing are
stochastic, the model's proposals vary between calls, and the stop rule is
priced against wall time that differs on your hardware. What is reproducible
is the *shape*: gains are banked as they are measured, nothing is adopted
without a re-measurement, and the run ends inside its budget. The decision log
in `dcp_optimizer_run-<timestamp>/` records which mechanism made each call.
`recipes/cell_replacement.py`'s module docstring walks one deterministic
candidate through all thirteen steps, from baseline WNS to the re-routed
re-measurement.

## Not checkable here

| Claim | Why not | Where it comes from |
|---|---|---|
| The 369.9 score, and the seven per-benchmark rows | The inputs are public and you can run them, but the scored runs were the organizers' own, on their hardware; placement, routing and the model all sample | The organizers' official evaluation |
| Development figures quoted in code comments | Our own 16-design gate, on our hardware | Not comparable to the official numbers, and labelled as such |
