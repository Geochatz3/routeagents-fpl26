#!/usr/bin/env python3
"""Glue test (jul02): drive run_ils_polish with the NEW __ROUTE_ONLY__ combo on
logicnets's optimized DCP through the real Vivado MCP connection. One cycle,
rotation pinned at the ROUTE_ONLY index. DRILL H expectation: -0.601 -> ~-0.588
ACCEPT with whs ~0.100 (hold IMPROVES). Validates the unroute -> route
AggressiveExplore -> phys_opt -> measure -> hold -> accept glue end-to-end.
"""
import asyncio, os, shutil, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dcp_optimizer import DCPOptimizer
from optimizer.ils_polish import (ILSPolishConfig, ILS_COMBOS, ROUTE_ONLY_PD,
                                  run_ils_polish)

OPT_DCP = ("/mnt/c/Users/Giorgos/fpl26_optimization_contest/"
           "fpl26_contest_benchmarks/logicnets_jscl_2025.1_optimized.dcp")
TARGET_CLOCK = "clk_fpl26contest"
WNS_TCL = (
    f"set clk_obj [get_clocks -quiet {{{TARGET_CLOCK}}}]; "
    f"if {{$clk_obj ne {{}}}} {{ set tp [get_timing_paths -max_paths 1 -setup "
    f"-to $clk_obj]; if {{[llength $tp] > 0}} {{get_property SLACK $tp}} "
    f"else {{puts 0.0}} }} else {{ set tp [get_timing_paths -max_paths 1 "
    f"-slack_lesser_than 999]; if {{[llength $tp] > 0}} {{get_property SLACK $tp}} "
    f"else {{puts 0.0}} }}")


async def main():
    rd = Path("/tmp/routeonly_glue"); rd.mkdir(exist_ok=True)
    # the work DCP must live on the Windows-visible filesystem (WSL Vivado
    # wrapper rewrites /mnt/<drive>/ paths; /tmp is invisible to Windows)
    work = Path("/mnt/c/Users/Giorgos/fpl26_optimization_contest/"
                "drill_d_out/routeonly_glue_work.dcp")
    shutil.copy(OPT_DCP, work)          # keep-best mutates the work copy
    opt = DCPOptimizer(api_key=os.environ.get("OPENROUTER_API_KEY", "unused"),
                       debug=False, run_dir=rd)
    await opt.start_servers()
    try:
        offset = next(i for i, c in enumerate(ILS_COMBOS)
                      if c[0] == ROUTE_ONLY_PD)
        cfg = ILSPolishConfig(enabled=True, max_cycles=1,
                              final_seed_no_improve_stop=0)
        res = await run_ils_polish(
            opt.call_tool, best_dcp_path=str(work), baseline_wns=-0.601,
            deadline_ts=time.time() + 1800.0, wns_tcl=WNS_TCL, cfg=cfg,
            log=lambda m: print(f"[ils] {m}", flush=True),
            no_improve_stop=0, combo_offset=offset)
        print(f"GLUE RESULT {res.summary()}", flush=True)
        print("GLUE_VERDICT=" + ("ACCEPTED" if res.improved else
                                  "NO_ACCEPT (see cycle log above)"), flush=True)
    finally:
        await opt.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
