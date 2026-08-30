# VivadoMCP — the Vivado tool server

An [MCP](https://modelcontextprotocol.io/) server that exposes AMD Vivado to
the agent as 17 named tools. It keeps **one long-lived Vivado process** in
Tcl-console mode and sends commands to it, so a design is opened once and then
placed, routed, timed and written without paying startup again.

The companion server is [RapidWrightMCP](../RapidWrightMCP/README.md), which
reads netlist structure. Vivado is what actually changes and measures a design;
RapidWright is what reasons about it.

## The 17 tools

| group | tools |
|---|---|
| session | `open_checkpoint`, `write_checkpoint`, `restart_vivado`, `run_tcl` |
| measure | `report_route_status`, `report_timing_summary`, `get_wns` |
| transform | `place_design`, `route_design`, `phys_opt_design` |
| analyse | `get_critical_high_fanout_nets`, `extract_critical_path_cells`, `extract_critical_path_pins` |
| floorplan | `report_utilization_for_pblock`, `create_and_apply_pblock` |
| export | `write_edif`, `write_verilog_simulation` |

`run_tcl` is the open console: the agent can issue any Tcl the constraint guard
allows, which is how a play that has no dedicated tool still runs.

## Running it

The optimizer starts this server itself; you do not normally launch it by hand.

```bash
pip install -r requirements.txt
python vivado_mcp_server.py          # stdio MCP server
```

`vivado` must be on `PATH`, or set `VIVADO_EXEC` to its full path.

## WSL

Vivado is frequently a Windows install driven from WSL. The server detects
this (`_detect_wsl`) and rewrites paths inside Tcl commands to Windows form
before sending them, so `/mnt/c/...` reaches Vivado as `C:\...`. Nothing on the
agent side needs to know which side of the boundary it is on.

## Checking it by hand

`manual_vivado_check.py` opens a checkpoint and exercises the measure and
transform tools against real Vivado. It is a manual smoke check, **not** part
of the offline suite — `make test` needs no Vivado at all.
