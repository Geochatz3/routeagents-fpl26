#!/usr/bin/env python3
"""Retroactively write readable EDIF files alongside DCPs.

Vivado's default write_checkpoint embeds an encrypted EDIF that
RapidWright cannot read.  validate_dcps.py fails with "Unable to find a
readable EDIF file" when the optimizer didn't explicitly write_edif.

This script:
  1. Spawns the Vivado MCP server (same env-compat path as the optimizer).
  2. For each input DCP: open_checkpoint, write_edif (force=true), close.
  3. Exits.

Usage:
  .venv/bin/python scripts/retrofit_edif.py <dcp> [<dcp> ...]
"""
from __future__ import annotations

import asyncio
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run(dcps: list[Path]) -> int:
    script_dir = Path(__file__).resolve().parent.parent
    vivado_args = [str(script_dir / "VivadoMCP" / "vivado_mcp_server.py")]
    params = StdioServerParameters(command=sys.executable, args=vivado_args, env=None)

    failed = 0
    async with AsyncExitStack() as stack:
        transport = await stack.enter_async_context(stdio_client(params))
        v_read, v_write = transport
        session = await stack.enter_async_context(ClientSession(v_read, v_write))
        await session.initialize()
        print(f"[retrofit_edif] MCP Vivado session up; {len(dcps)} DCP(s) to process.")

        for dcp in dcps:
            dcp = dcp.resolve()
            edif = dcp.with_suffix(".edf")
            print(f"\n=== {dcp.name} ===")
            try:
                print(f"  open_checkpoint {dcp}")
                r = await session.call_tool(
                    "open_checkpoint",
                    {"dcp_path": str(dcp), "timeout": 1800.0},
                )
                txt = "\n".join(c.text for c in r.content if hasattr(c, "text"))
                if "error" in txt.lower() and "opened" not in txt.lower():
                    print(f"  ✗ open_checkpoint failed: {txt[:200]}")
                    failed += 1
                    continue

                print(f"  write_edif {edif}")
                r = await session.call_tool(
                    "write_edif",
                    {"edif_path": str(edif), "force": True, "timeout": 600.0},
                )
                if edif.exists() and edif.stat().st_size > 0:
                    print(f"  ✓ EDIF written ({edif.stat().st_size:,} bytes)")
                else:
                    print(f"  ✗ EDIF missing/empty after write_edif")
                    failed += 1
            except Exception as e:
                print(f"  ✗ exception: {type(e).__name__}: {e}")
                failed += 1

    return failed


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    dcps = [Path(a) for a in sys.argv[1:]]
    missing = [p for p in dcps if not p.exists()]
    if missing:
        print(f"Error: missing DCPs: {missing}", file=sys.stderr)
        return 2
    failed = asyncio.run(run(dcps))
    if failed:
        print(f"\nDONE — {failed} failure(s).")
        return 1
    print(f"\nDONE — all {len(dcps)} DCPs retrofitted with readable EDIF.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
