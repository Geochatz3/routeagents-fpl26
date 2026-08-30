<img src="docs/assets/routeagents.png" alt="RouteAgents" width="380" align="right">

# RouteAgents

An LLM-guided autonomous agent that raises the Fmax of placed-and-routed FPGA
designs, driving Vivado and RapidWright through MCP (Model Context Protocol)
servers. Built for the [Agentic FPGA Backend Optimization Competition @
FPL'26](https://xilinx.github.io/fpl26_optimization_contest/).

A `.dcp` goes in and a faster one comes out inside a fixed wall-clock and
LLM-cost budget, re-validated: routing, setup and hold timing, and a
flip-flop-count drift check. A worse result never replaces the best one found,
and nothing is adopted on a model's say-so.

## Results

Built for the [FPL'26 Agentic FPGA Backend Optimization
Contest](https://xilinx.github.io/fpl26_optimization_contest/). The official
evaluation scored it **369.9**: 7 of 7 hidden benchmarks improved, none failed
validation, inside a budget of one hour and one dollar per design. The score
is the megahertz each design gained, less a penalty for the wall time and the
model spend it took.

All seven are in the public benchmark set,
[`fpl26_contest_benchmarks_v1.2.0`](https://github.com/Xilinx/fpl26_optimization_contest/releases/tag/v1.2.0),
which `make setup` downloads, so you can run the same inputs. Or point it at
your own design and see what it finds.

The 42 plays, the feature router and the governors are this project's. The
per-benchmark figures, what is inherited from the contest starter kit, and what
is and is not reproducible are all in
[docs/PROVENANCE.md](docs/PROVENANCE.md).

## How it works

Three layers. Each one only works because of the one under it.

| | |
|---|---|
| **The playbook** | 42 plays: concrete moves on a routed design, such as ruin a region and rebuild it, re-place from the pristine input, retime across register boundaries, path-group the worst endpoints. The 29 that fire without the model carry a measured trigger **and a measured exclusion**, mined offline and frozen before the run starts; the other 13 are macros the model can call, each with the conditions it applies under. [docs/PLAYBOOK.md](docs/PLAYBOOK.md) |
| **The brief** | a router picks the play class from measured features alone, never from the design's name, and writes the model an ordered, costed plan with the inapplicable plays marked `BLOCKED`. Inside that brief the model does what a global directive cannot: it reads the timing report it just produced and names the operands, which cells to move, which nets to replicate, which endpoints to group. [docs/METHODS.md](docs/METHODS.md#the-brief-the-wns-band-recipe-router) |
| **The governors** | keep only what the tools verified, and stop at the right moment: the bank, where a worse attempt never replaces the best; the priced clock, where the next second must out-earn the tax it adds; and valid-state admission, where a number counts only if the design that produced it is still fully routed. [docs/METHODS.md](docs/METHODS.md#the-governors) |

The model is a parameter, not a dependency: `--model <id>` takes any
OpenAI-compatible id, and `make run_no_llm` runs the deterministic path alone.

## Requirements

| | |
|---|---|
| OS | Linux, or Windows via WSL2 |
| Python | >= 3.10 |
| AMD Vivado | 2025.1 (licensed; not redistributable) |
| Java | 11+, Vivado's bundled JRE works |
| LLM access | An OpenRouter key, for the LLM-guided modes only |

The offline test suite needs none of the above beyond Python.

## Quick start

```bash
git clone https://github.com/Geochatz3/routeagents-fpl26.git
cd routeagents-fpl26
make setup                      # deps, RapidWright, and the 19 public dev DCPs
cp .env.example .env            # your OpenRouter key

make run_optimizer DCP=path/to/your_design.dcp     # optimize
make run_no_llm    DCP=path/to/your_design.dcp     # same, without an LLM
make validate GOLDEN=your_design.dcp \
     REVISED=your_design_optimized-<timestamp>.dcp # prove equivalence

pip install -r requirements-dev.txt && make test   # offline suite, no Vivado
```

You get `your_design_optimized-<timestamp>.dcp` next to the input, plus a run
directory holding the decision log, per-tool timing and cost ledger. Run modes,
every `FPL26_*` flag and the environment variables are in
[docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Security

The agent executes tool commands an LLM chose, including Vivado Tcl, and
nothing sandboxes the Tcl interpreter. `optimizer/constraint_guard.py` refuses
timing-constraint edits at that boundary and fingerprints the design's
constraints after load, re-checking them before the result is written; the
deny-list is string matching and defeatable, the fingerprint is what catches
the effect regardless of route.

In the LLM-guided modes your design's measurements leave the machine: timing
summaries, utilization, path anatomy and cell and net names go to the model
provider. `make run_no_llm` sends nothing anywhere.

Run it as a user that can only reach the design you are optimizing, keep a copy
of your input outside that user's reach, and always `make validate` the result.
The agent's own gates cover timing, routing, hold and flip-flop drift;
`make validate` is what establishes logical equivalence.

## Documentation

| | |
|---|---|
| [docs/PLAYBOOK.md](docs/PLAYBOOK.md) | every play the agent can choose, and when |
| [docs/METHODS.md](docs/METHODS.md) | the pipeline, the governors, and what each mechanism is measured to be worth |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | run modes, environment variables, every flag and its default |
| [docs/PROVENANCE.md](docs/PROVENANCE.md) | the contest artifact: results, what differs from the tag, how to verify it |
| [ARTIFACT.md](ARTIFACT.md) | every claim in this repository, and the command that checks it |
| [docs/optimization_example.md](docs/optimization_example.md) | the three definitions this repo cites from the organizers' worked example, excerpted; the full page is upstream |

`main` is the tool. The first commit, tag `fpl26-final-submission`, is the
byte-exact tree submitted to the FPL'26 Agentic FPGA Backend Optimization
Contest, kept as an archive and not maintained. Its evaluation results and how
to verify it are in [docs/PROVENANCE.md](docs/PROVENANCE.md).

## License and citation

Apache-2.0 (see `LICENSE`, `NOTICE`). Started from the organizers' Apache-2.0
contest starter kit; the agent, optimizer modules, recipes, scheduler and
prompts are copyright Georgios Chatzitsompanis. Citation metadata is in
[`CITATION.cff`](CITATION.cff).
