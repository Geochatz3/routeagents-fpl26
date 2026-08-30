# RapidWrightMCP — the structural analysis server

An [MCP](https://modelcontextprotocol.io/) server that exposes
[RapidWright](https://github.com/Xilinx/RapidWright) — AMD's open-source
framework for reading and transforming design checkpoints — to the agent as
17 named tools.

The companion server is [VivadoMCP](../VivadoMCP/README.md). Vivado is what
changes and measures a design; RapidWright is what makes the checkpoint
legible in between — cell placement coordinates, net topology, fabric
geometry — without paying for a Vivado invocation.

## The 17 tools

| group | tools |
|---|---|
| session | `initialize_rapidwright`, `read_checkpoint`, `write_checkpoint` |
| device | `get_supported_devices`, `get_device_info`, `get_tile_info`, `search_sites` |
| design | `get_design_info`, `search_cells`, `compare_design_structure` |
| geometry | `analyze_critical_path_spread`, `analyze_net_detour` |
| floorplan | `analyze_fabric_for_pblock`, `convert_fabric_region_to_pblock` |
| transform | `optimize_lut_input_cone`, `optimize_fanout`, `optimize_cell_placement` |

`initialize_rapidwright` must be called before any other tool; the JVM starts
once per session.

**All 17 come from the contest starter kit** and this server's tool surface is
unchanged from it — verify with
`diff <(curl -s https://raw.githubusercontent.com/Xilinx/fpl26_optimization_contest/main/RapidWrightMCP/server.py) server.py`.
What this project added is above the tool layer, not in it.

Four of them carry the rest of the system: `analyze_critical_path_spread`
produces the critical-path spread that, with the LUT count, forms the design
fingerprint the recipe router and the strategy memory key on;
`analyze_net_detour` tells a placement play which cells are worth moving;
`analyze_fabric_for_pblock` and `convert_fabric_region_to_pblock` are what make
an automatic pblock play possible. The plays built on them are catalogued in
[docs/PLAYBOOK.md](../docs/PLAYBOOK.md). The organizers' own worked example
chains `analyze_net_detour` into `optimize_cell_placement`; the detour-ratio
definition it turns on is excerpted in
[docs/optimization_example.md](../docs/optimization_example.md), and
`recipes/cell_replacement.py` is this repository's version of that chain.

## Running it

The optimizer starts this server itself; you do not normally launch it by
hand. `make setup` from the repository root installs the Python dependencies,
builds RapidWright from the `RapidWright/` submodule (`./gradlew
compileJava`), and sets `RAPIDWRIGHT_PATH` and `CLASSPATH`.

```bash
python3 server.py            # stdio MCP server
```

The `rapidwright` pip package supplies the JPype bridge to Java. The two
environment variables redirect it away from the package's own bundled jars
and onto the classes compiled from the local submodule — which is what makes
a modified RapidWright take effect:

| Variable | Value |
|---|---|
| `RAPIDWRIGHT_PATH` | absolute path to the `RapidWright/` submodule |
| `CLASSPATH` | `…/RapidWright/bin:…/RapidWright/jars/*` |

After changing RapidWright's Java source, rebuild with `make
build-rapidwright` from the repository root. Java 11 or newer is required;
Vivado's bundled JRE works.

## Adding a tool

Three edits, in this order: implement it in `rapidwright_tools.py` (return a
dict, and guard on `_initialized`), register a `Tool(...)` in `server.py`'s
`list_tools()`, and add its branch in `call_tool()`. Then add its name to the
table above — `tests/test_tool_surface.py` fails if the code and this README
disagree, which is how the starter kit's version of this file came to document
11 of its own 17 tools.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `RapidWright not initialized` | call `initialize_rapidwright` first, or Java is missing (`java -version`) |
| Local Java changes have no effect | not rebuilt (`make build-rapidwright`), or `CLASSPATH` still points at the pip package's jars |
| Out of memory | raise `jvm_max_memory` on `initialize_rapidwright` (e.g. `"8G"`) |
| JPype import errors | `pip3 install --force-reinstall rapidwright` |

`rapidwright_mcp.log` is written next to the server at runtime.
`test_server.py` is a manual smoke check against real RapidWright — it is
**not** part of `make test`, which needs no Java and no Vivado.

## Reference

[RapidWright docs](https://www.rapidwright.io/docs/) ·
[Javadoc](https://www.rapidwright.io/javadoc/) ·
[Model Context Protocol](https://modelcontextprotocol.io/)
