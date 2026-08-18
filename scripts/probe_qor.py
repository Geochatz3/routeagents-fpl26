#!/usr/bin/env python3
"""Session-14 standalone JSON QoR probe.

Spawns a subprocess-isolated Vivado batch to emit
``<run_dir>/<dcp_stem>.qor.json`` for an existing finalised DCP.  Uses
the SAME Tcl + subprocess invocation pattern as
``DCPOptimizer._run_qor_capture_subprocess`` so backfill runs validate
the same mechanism the live ``--capture-qor`` hook will use.

This script:
- never opens the DCP via the MCP session (subprocess-isolated),
- never writes into ``submission/`` (refuses such targets),
- never invokes the LLM, contest controller, or MCP servers,
- enforces a 60 s per-DCP timeout by default,
- never raises — failure / timeout / missing-DCP all reported in the
  returned table.

Usage::

    python3 scripts/probe_qor.py \
        --dcp <path> --run-dir <dir> \
        [--dcp <path> --run-dir <dir> ...] \
        [--timeout 60]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dcp_optimizer import _path_is_in_submission


TCL_PROBE = (
    "set dcp [lindex $argv 0]\n"
    "set out [lindex $argv 1]\n"
    "if {![file exists $dcp]} { puts \"ERR_NO_DCP\"; exit 3 }\n"
    "open_checkpoint $dcp\n"
    "if {[catch {report_design_analysis -qor_summary -json $out} err]} {\n"
    "    puts \"ERR_REPORT: $err\"\n"
    "    exit 4\n"
    "}\n"
    "puts \"QOR_JSON: $out\"\n"
    "exit 0\n"
)


def _to_windows(p: str) -> str:
    """Translate WSL path → Windows path for cmd.exe invocation."""
    return (
        p.replace("/mnt/c/", "C:\\")
         .replace("/mnt/d/", "D:\\")
         .replace("/", "\\")
    )


async def probe_one(dcp_path: Path, run_dir: Path, timeout_s: float = 60.0):
    """Probe a single DCP.  Returns ``(status, json_path, runtime_s, error)``.

    ``status`` is one of ``"success"``, ``"timeout"``, ``"error"``,
    ``"skipped"``, ``"refused_submission_path"``.
    """
    dcp_path = Path(dcp_path).resolve()
    run_dir = Path(run_dir).resolve()
    json_path = run_dir / f"{dcp_path.stem}.qor.json"

    if _path_is_in_submission(json_path):
        return ("refused_submission_path", str(json_path), 0.0,
                "json target under submission/")
    if not dcp_path.exists() or dcp_path.stat().st_size == 0:
        return ("skipped", str(json_path), 0.0, "dcp missing or empty")

    run_dir.mkdir(parents=True, exist_ok=True)
    tcl_path = run_dir / "_qor_capture_probe.tcl"
    tcl_path.write_text(TCL_PROBE)

    if sys.platform.startswith("linux") and Path("/mnt/c/WINDOWS/system32/cmd.exe").exists():
        win_dcp = _to_windows(str(dcp_path))
        win_out = _to_windows(str(json_path))
        win_tcl = _to_windows(str(tcl_path))
        cmd = [
            "/mnt/c/WINDOWS/system32/cmd.exe", "/c",
            "D:\\Xilinx\\2025.1\\Vivado\\bin\\vivado.bat",
            "-mode", "batch", "-nojournal", "-nolog",
            "-source", win_tcl, "-tclargs", win_dcp, win_out,
        ]
    else:
        cmd = [
            "vivado", "-mode", "batch", "-nojournal", "-nolog",
            "-source", str(tcl_path), "-tclargs",
            str(dcp_path), str(json_path),
        ]
    t0 = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(run_dir),
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            return ("timeout", str(json_path),
                    round(time.monotonic() - t0, 2),
                    f"timed out after {timeout_s} s")
        rc = proc.returncode
        runtime = round(time.monotonic() - t0, 2)
        tail = (stdout or b"")[-400:].decode("utf-8", errors="replace")
        if rc == 0 and json_path.exists() and json_path.stat().st_size > 0:
            return ("success", str(json_path), runtime, None)
        return ("error", str(json_path), runtime,
                f"rc={rc} tail={tail!r}")
    except FileNotFoundError as exc:
        return ("error", str(json_path),
                round(time.monotonic() - t0, 2),
                f"vivado not found: {exc}")
    except Exception as exc:
        return ("error", str(json_path),
                round(time.monotonic() - t0, 2),
                f"subprocess error: {exc!r}")


async def _main_async(args):
    pairs = list(zip(args.dcp, args.run_dir))
    rows = ["dcp\trun_dir\tstatus\tjson_path\truntime_s\terror"]
    for dcp, rd in pairs:
        status, json_path, runtime, err = await probe_one(
            Path(dcp), Path(rd), timeout_s=args.timeout,
        )
        rows.append("\t".join(str(x) for x in (
            dcp, rd, status, json_path, runtime, err or "-",
        )))
        # Stream as we go.
        print(rows[-1], flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Standalone Vivado JSON QoR probe.")
    parser.add_argument("--dcp", action="append", default=[], required=True,
                        help="Path to a DCP file (repeatable).")
    parser.add_argument("--run-dir", action="append", default=[], required=True,
                        help="Run directory to write JSON into (repeatable; "
                             "must align positionally with --dcp).")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Per-DCP Vivado timeout in seconds (default 60).")
    args = parser.parse_args(argv)
    if len(args.dcp) != len(args.run_dir):
        parser.error("--dcp and --run-dir must be given the same number of times.")
    print("dcp\trun_dir\tstatus\tjson_path\truntime_s\terror")
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
