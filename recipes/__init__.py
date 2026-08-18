"""
Optimization recipes — deterministic, no-LLM optimization strategies that
follow the contest's "Identify → Analyze → Optimize → Measure" pattern
documented in docs/optimization_example.md.

Each recipe is a Python module that:
  1. Imports the MCP-server tool functions directly (bypassing MCP/stdio
     protocol — fastest path; the LLM-side optimizer continues to use the
     stdio MCP servers).
  2. Exposes an `apply(input_dcp, output_dcp, **kwargs) -> dict` entrypoint
     that runs the recipe end-to-end and returns metrics (Δ Fmax, cells
     touched, route-error count, wall time, …).
  3. Provides a CLI for ad-hoc / scheduler-integrated runs:
        python3 -m recipes.<recipe_name> INPUT.dcp -o OUTPUT.dcp [...]

Recipes are deterministic — same input + same Vivado version → same output
DCP.  This makes them attractive scheduler candidates: cheaper than the
LLM-driven candidates, often faster, and predictable for designs where
the recipe applies.

Index:
  cell_replacement   — re-place cells with high routed-path detour ratio
                       to the centroid of their connections.  Contest
                       example, +40 MHz on vexriscv_re-place_2025.1.dcp.

See `.planning/beta/FINAL_DEV_ROADMAP.md` and the contest's
`docs/optimization_example.md` for the recipe pattern.
"""
