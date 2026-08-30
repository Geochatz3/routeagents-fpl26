"""
Cell Re-Placement Recipe — re-place cells whose outgoing/incoming nets
have high routed-path detour ratios, then re-route and measure.

Adapts the working script from docs/optimization_example.md (the contest
organizers' canonical recipe example).  Wraps it as a reusable function
plus CLI so the scheduler can call it as a deterministic candidate.

Pipeline (per the contest's spec):
  1. open_checkpoint  → baseline WNS / Fmax
  2. extract_critical_path_pins(num_paths)
  3. close Vivado
  4. RapidWright: read_checkpoint
  5. RapidWright: analyze_net_detour(critical_paths, threshold)
  6. Filter to unique cells on the worst N paths (default: paths 1-2)
  7. RapidWright: optimize_cell_placement(cell_names)
  8. RapidWright: write_checkpoint(output_dcp)
  9. open_checkpoint(output_dcp)
 10. route_design
 11. report_route_status (capture error count)
 12. report_timing_summary → new WNS / Fmax
 13. cleanup Vivado, return metrics dict

Returns:
  {
    "status": "success" | "error" | "no_candidates",
    "input_dcp": "...",
    "output_dcp": "..." | None,
    "baseline_wns_ns": -0.946,  baseline_fmax_mhz: 310.17,
    "final_wns_ns":    -0.683,  final_fmax_mhz:    350.21,
    "delta_fmax_mhz":  +40.04,
    "candidates_total": 3,
    "candidates_targeted": ["base/lut2"],
    "cells_moved": ["base/lut2"],
    "cells_unmoved": [],
    "route_errors": 0,
    "wall_time_s": 132.8,
    "error": None,
  }

A `delta_fmax_mhz < 0` is a regression — the recipe didn't help on this
design.  The scheduler should treat the result as INVALID in that case
(downstream selection picks the higher-Fmax candidate, including the
input DCP if no candidate improved over baseline).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Importing the MCP server modules directly — same approach as the
# contest's example.  These imports require the script's working dir
# to be the repo root (RapidWrightMCP/ and VivadoMCP/ must be importable).
def _setup_imports():
    repo_root = Path(__file__).parent.parent
    for sub in ("VivadoMCP", "RapidWrightMCP"):
        sys.path.insert(0, str(repo_root / sub))


CONTEST_CLOCK = "clk_fpl26contest"


def _get_fmax(vivado_module) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Read timing metrics for the target clock.

    Returns WNS and period in nanoseconds and Fmax in MHz. A tool-reported 0.0
    sentinel indicating that no path exists on the clock is normalized to None.
    """
    result = vivado_module.run_tcl_command(
        f"set p [get_timing_paths -max_paths 1 -group {CONTEST_CLOCK}]; "
        "if {[llength $p] > 0} {get_property SLACK $p} else {puts 0.0}",
        timeout=60,
    )
    m = re.search(r"[-]?\d+\.\d+", result)
    wns = float(m.group()) if m else None

    result = vivado_module.run_tcl_command(
        f"get_property PERIOD [get_clocks {CONTEST_CLOCK}]", timeout=60
    )
    m = re.search(r"\d+\.\d+", result)
    period = float(m.group()) if m else None

    if wns is not None and period is not None:
        fmax = 1000.0 / (period - wns)
    else:
        fmax = None
    return wns, period, fmax


def _select_target_cells(candidates: list[dict], max_path: int = 2) -> list[str]:
    """Filter detour-analysis candidates to unique cells on the worst N
    critical paths (default: paths 1 and 2).  Preserves the input
    ordering (which is detour-ratio descending out of analyze_net_detour),
    deduping while keeping the first-seen position so the highest-detour
    cell is tried first.

    Pure-function — easily unit-testable without Vivado/RapidWright.
    """
    seen = set()
    result = []
    for c in candidates:
        if c.get("path") is None or c["path"] > max_path:
            continue
        name = str(c["cell"])
        if name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


def apply(
    input_dcp: str | Path,
    output_dcp: str | Path,
    num_paths: int = 10,
    detour_threshold: float = 2.0,
    target_max_path: int = 5,
    max_cells_to_try: int = 8,
    open_checkpoint_timeout: int = 300,
    route_design_timeout: int = 600,
) -> dict:
    """Run the cell-replacement recipe end-to-end.  See module docstring."""
    _setup_imports()
    import vivado_mcp_server as vivado  # noqa: WPS433
    import rapidwright_tools as rw       # noqa: WPS433

    input_dcp = str(Path(input_dcp).resolve())
    output_dcp = str(Path(output_dcp).resolve())
    Path(output_dcp).parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    result: dict = {
        "status": "error",
        "input_dcp": input_dcp,
        "output_dcp": None,
        "baseline_wns_ns": None,
        "baseline_fmax_mhz": None,
        "final_wns_ns": None,
        "final_fmax_mhz": None,
        "delta_fmax_mhz": None,
        "candidates_total": 0,
        "candidates_targeted": [],
        "cells_moved": [],
        "cells_unmoved": [],
        "route_errors": None,
        "wall_time_s": 0.0,
        "error": None,
    }

    try:
        # ── Step 1: Baseline ───────────────────────────────────────────
        logger.info("[1/4] Vivado baseline")
        vivado.start_vivado()
        vivado.run_tcl_command(f"open_checkpoint {{{input_dcp}}}", timeout=open_checkpoint_timeout)
        baseline_wns, clk_period, baseline_fmax = _get_fmax(vivado)
        result["baseline_wns_ns"] = baseline_wns
        result["baseline_fmax_mhz"] = baseline_fmax
        logger.info(f"  baseline WNS: {baseline_wns}  Fmax: {baseline_fmax}")

        pins_json = vivado.extract_critical_path_pins(num_paths=num_paths)
        critical_paths = json.loads(pins_json) if isinstance(pins_json, str) else pins_json
        logger.info(f"  extracted {len(critical_paths)} critical paths")
        vivado.cleanup_vivado()

        # ── Step 2: Analyze ────────────────────────────────────────────
        logger.info("[2/4] RapidWright detour analysis")
        rw.initialize_rapidwright()
        rw.read_checkpoint(input_dcp)

        analysis = rw.analyze_net_detour(
            critical_paths_data=critical_paths,
            detour_threshold=detour_threshold,
        )
        candidates = analysis.get("candidates", [])
        result["candidates_total"] = len(candidates)
        logger.info(
            f"  cells_analyzed={analysis.get('cells_analyzed')}  "
            f"candidates(detour>{detour_threshold})={len(candidates)}"
        )

        target_cells = _select_target_cells(candidates, max_path=target_max_path)
        result["candidates_targeted"] = target_cells

        if not target_cells:
            logger.info("  no candidates above threshold on worst paths — recipe does not apply")
            result["status"] = "no_candidates"
            return result

        # Try cells one at a time so that a single bad candidate — some cell
        # types make the placer library throw — does not kill the whole recipe.
        # Stop at the first cell that successfully moves.  Handled as a batch,
        # one bad register cell throws inside the unplace call, the whole batch
        # returns empty, and the recipe never recovers.
        cells_to_try = target_cells[:max_cells_to_try]
        logger.info(
            f"  {len(target_cells)} unique candidate cells in worst-{target_max_path} paths; "
            f"trying up to {len(cells_to_try)} in detour-ratio descending order"
        )

        # ── Step 3: Optimize ───────────────────────────────────────────
        logger.info("[3/4] RapidWright optimize_cell_placement (sequential)")
        for attempt_idx, cell_name in enumerate(cells_to_try):
            logger.info(f"  [{attempt_idx+1}/{len(cells_to_try)}] trying {cell_name}")
            try:
                opt_result = rw.optimize_cell_placement(cell_names=[cell_name])
            except Exception as e:
                logger.warning(f"    optimize_cell_placement raised: {e!r}")
                result["cells_unmoved"].append({
                    "cell": cell_name, "status": "exception", "message": repr(e),
                })
                continue

            opt_results = opt_result.get("results", []) or []
            cell_moved_this_attempt = False
            for r in opt_results:
                cell = r.get("cell")
                status = r.get("status")
                msg = r.get("message")
                if status == "success":
                    result["cells_moved"].append(cell)
                    cell_moved_this_attempt = True
                    logger.info(f"    ✓ {cell}: {msg}")
                else:
                    result["cells_unmoved"].append({"cell": cell, "status": status, "message": msg})
                    logger.info(f"    ✗ {cell}: {status} — {msg}")
            if not opt_results:
                # Empty results — RapidWright caught an exception internally.
                # Record as a special "internal_error" status so the diagnosis
                # is preserved even when the recipe overall succeeds.
                result["cells_unmoved"].append({
                    "cell": cell_name, "status": "internal_error",
                    "message": "optimize_cell_placement returned empty results "
                               "(RapidWright likely caught an exception silently)",
                })
                logger.info(f"    ✗ {cell_name}: empty result list (silent failure)")

            if cell_moved_this_attempt:
                # A successful move.  Stop trying more cells —
                # additional moves compound risk of breaking the design.
                logger.info(f"  stopping after first successful move (attempt {attempt_idx+1})")
                break

        # Gate: if no cells actually moved, the design state may be partially
        # mutated — some cells unplaced but not re-placed by the failed call.
        # Bail out without writing: a checkpoint written here would persist a
        # corrupt state, and the following route would emit routing errors.
        if not result["cells_moved"]:
            logger.info(
                f"  optimize_cell_placement moved 0 cells "
                f"after {len(cells_to_try)} attempts. "
                "Skipping write_checkpoint to avoid persisting partial state."
            )
            result["status"] = "no_cells_moved"
            return result

        rw.write_checkpoint(output_dcp)
        logger.info(f"  wrote {output_dcp} (cells_moved={len(result['cells_moved'])})")

        # ── Step 4: Measure ────────────────────────────────────────────
        logger.info("[4/4] Vivado re-route + remeasure")
        vivado.start_vivado()
        vivado.run_tcl_command(f"open_checkpoint {{{output_dcp}}}", timeout=open_checkpoint_timeout)
        vivado.run_tcl_command("route_design", timeout=route_design_timeout)

        route_status = vivado.run_tcl_command(
            "report_route_status -return_string", timeout=60,
        )
        m = re.search(r"# of nets with routing errors.*?:\s+(\d+)", route_status)
        result["route_errors"] = int(m.group(1)) if m else -1

        new_wns, _, new_fmax = _get_fmax(vivado)
        result["final_wns_ns"] = new_wns
        result["final_fmax_mhz"] = new_fmax
        if new_fmax is not None and baseline_fmax is not None:
            result["delta_fmax_mhz"] = round(new_fmax - baseline_fmax, 4)
        result["output_dcp"] = output_dcp

        logger.info(
            f"  route_errors={result['route_errors']}  "
            f"final_wns={new_wns}  final_fmax={new_fmax}  "
            f"delta={result['delta_fmax_mhz']}"
        )
        vivado.cleanup_vivado()

        # GATE: a successful recipe run must produce a fully-routed DCP.
        # The contest scoring rules require par_routed=true (0 routing
        # errors) for the design to receive any score; partial routing
        # would set the run's score to 0 even with a high reported Fmax.
        if result["route_errors"] is None or result["route_errors"] > 0:
            logger.warning(
                f"  routing errors detected ({result['route_errors']}); "
                "marking recipe as 'route_errors' (would score 0 in contest)."
            )
            result["status"] = "route_errors"
        else:
            result["status"] = "success"

    except Exception as e:
        logger.exception("recipe failed")
        result["error"] = repr(e)
        result["status"] = "error"
        try:
            vivado.cleanup_vivado()
        except Exception:
            pass

    result["wall_time_s"] = round(time.time() - start, 1)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("input_dcp", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True, dest="output_dcp")
    parser.add_argument("--num-paths", type=int, default=10,
                        help="How many critical paths to analyze (default 10).")
    parser.add_argument("--detour-threshold", type=float, default=2.0,
                        help="Routed-path / Manhattan ratio above which a cell "
                             "is a re-placement candidate (default 2.0).")
    parser.add_argument("--target-max-path", type=int, default=5,
                        help="Only target cells on the worst N paths (default 5).")
    parser.add_argument("--max-cells-to-try", type=int, default=8,
                        help="Try up to this many candidate cells sequentially "
                             "(highest detour first), stopping at the first that "
                             "moves successfully (default 8).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    result = apply(
        input_dcp=args.input_dcp,
        output_dcp=args.output_dcp,
        num_paths=args.num_paths,
        detour_threshold=args.detour_threshold,
        target_max_path=args.target_max_path,
        max_cells_to_try=args.max_cells_to_try,
    )

    print()
    print(json.dumps(result, indent=2, default=str))
    print()

    # Process-style exit code:
    #   0  success — output DCP fully routed and (if delta_fmax > 0) better
    #   1  hard error (Vivado crash, NPE, etc.)
    #   2  no_candidates — recipe didn't apply (no cells with detour > thresh)
    #   3  no_cells_moved — analysis found candidates but optimize_cell_placement
    #                       didn't move any (e.g. RapidWright NPE on registers)
    #   4  route_errors  — recipe produced an output DCP but route_design left
    #                      routing errors (would score 0 in contest)
    if result["status"] == "success":
        return 0
    if result["status"] == "no_candidates":
        return 2
    if result["status"] == "no_cells_moved":
        return 3
    if result["status"] == "route_errors":
        return 4
    return 1


if __name__ == "__main__":
    sys.exit(main())
