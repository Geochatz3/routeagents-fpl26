# Optimization example (excerpt)

> **This is an excerpt, not the full page.** The FPL'26 contest organizers
> publish a worked optimization example in their starter kit — the four-step
> recipe pattern, a cell re-placement walkthrough, and a runnable script.
> The full text, with its figures, lives upstream at
> [Xilinx/fpl26_optimization_contest](https://github.com/Xilinx/fpl26_optimization_contest)
> and on the [contest site](https://xilinx.github.io/fpl26_optimization_contest/).
>
> Reproduced in full in the scored tree (tag `fpl26-final-submission`), where
> it is byte-identical to what was submitted. On `main` it is reduced to the
> three definitions this repository actually cites, so that each citation can
> still be checked from a clone. Apache-2.0, © the contest organizers; see
> `NOTICE`.

## The four-step recipe pattern

Both system prompts (`prompts/system_prompt_scored.txt`,
`prompts/system_prompt_v2_experimental.txt`) name this cycle as the origin of
the optimisation loop they instruct the model to follow.

| Step | Goal | Typical tools |
|------|------|---------------|
| **1. Identify** | Pick a specific physical optimization target | Domain knowledge, literature review |
| **2. Analyze** | Build a tool to find where this optimization applies | Vivado timing reports, RapidWright analysis scripts |
| **3. Optimize** | Implement the transformation | RapidWright APIs, Vivado ECO commands |
| **4. Measure** | Quantify the impact on Fmax | Vivado `report_timing_summary` |

The cycle is meant to be iterated: measure, refine the analysis heuristics,
tune the parameters, repeat.

## The detour ratio

The heuristic behind the organizers' cell re-placement example, and the origin
of the ratio that `RapidWrightMCP`'s `analyze_net_detour` computes and that
`optimizer/pathology.py` thresholds (`DETOUR_RATIO_HIGH`):

```
detour_ratio = routed_path_length / manhattan_distance
```

A ratio of 1.0 means the route is perfectly direct; ratios above ~2.0 suggest
the cell may benefit from re-placement. A cell should only be moved if the
surrounding path segments have slack to absorb the perturbation.

## The Fmax definition

The organizers' reference calculation, transcribed verbatim from their script.
`tests/test_positive_slack_continue.py` pins this repository's `calculate_fmax`
against it — including the property that positive slack keeps paying, with no
sign test and no clamp at met timing:

```python
def get_fmax():
    """Return (wns, fmax_mhz) for the contest clock."""
    # wns  = SLACK of the worst path in the contest clock group
    # period = PERIOD of the contest clock
    fmax = 1000.0 / (period - wns)
    return wns, period, fmax
```

## The recipe, as this repository runs it

`recipes/cell_replacement.py` adapts the organizers' script into a reusable
deterministic candidate; its module docstring carries the full thirteen-step
pipeline, from baseline WNS through `analyze_net_detour` and
`optimize_cell_placement` to the re-routed re-measurement. That docstring, not
this page, is the current description of what the repository actually runs.
