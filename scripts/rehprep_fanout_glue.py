#!/usr/bin/env python3
"""Rehearsal-prep glue test (jun15): drive _fanout_polish_after_ils directly on
logicnets's optimized DCP through the optimizer's REAL Vivado MCP connection,
bypassing the slow LLM recipe (which saturates the local budget so ILS never
fires on WSL). Validates the integration glue: open best -> phys_opt
AggressiveFanoutOpt -> reroute-if-needed -> measure -> hold -> accept -> ship.
Expect ACCEPT on logicnets (fanout = +0.029ns, hold-safe whs 0.068).
"""
import asyncio, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dcp_optimizer import DCPOptimizer
from optimizer.ils_polish import ILSPolishConfig, _measure

OPT_DCP = ("/mnt/c/Users/Giorgos/fpl26_optimization_contest/"
           "fpl26_contest_benchmarks/logicnets_jscl_2025.1_optimized.dcp")
TARGET_CLOCK = "clk_fpl26contest"


def _wns_tcl(clk):
    return (
        f"set clk_obj [get_clocks -quiet {{{clk}}}]; "
        f"if {{$clk_obj ne {{}}}} {{ set tp [get_timing_paths -max_paths 1 -setup "
        f"-to $clk_obj]; if {{[llength $tp] > 0}} {{get_property SLACK $tp}} "
        f"else {{puts 0.0}} }} else {{ set tp [get_timing_paths -max_paths 1 "
        f"-slack_lesser_than 999]; if {{[llength $tp] > 0}} {{get_property SLACK $tp}} "
        f"else {{puts 0.0}} }}")


async def main():
    rd = Path("/tmp/rehprep_glue"); rd.mkdir(exist_ok=True)
    # api_key only required by the constructor; this test makes NO LLM call.
    opt = DCPOptimizer(api_key=os.environ.get("OPENROUTER_API_KEY", "unused-glue-test"),
                       debug=False, run_dir=rd)
    await opt.start_servers()
    try:
        opt.target_clock = TARGET_CLOCK
        # cheap-gate must pass: anchor known + < fanout_max_cycle_s (600)
        opt._ils_polish_cfg.expected_heavy_cycle_s = 200.0
        wns_tcl = _wns_tcl(TARGET_CLOCK)

        await opt.call_tool("vivado_run_tcl",
                            {"command": f"open_checkpoint {{{OPT_DCP}}}", "timeout": 900})
        w0, ur0 = await _measure(opt.call_tool, wns_tcl, timeout_s=300)
        print(f"GLUE BASELINE best_wns={w0} unrouted={ur0}", flush=True)
        opt.best_wns = w0
        opt._best_valid_dcp = Path(OPT_DCP)
        opt._best_valid_dcp_wns = w0

        deadline = time.time() + 1800.0
        await opt._fanout_polish_after_ils(OPT_DCP, w0, deadline, wns_tcl)

        improved = (opt.best_wns is not None and w0 is not None
                    and opt.best_wns > w0 + 1e-9)
        print(f"GLUE RESULT best_wns={opt.best_wns} dcp={opt._best_valid_dcp} "
              f"improved={improved}", flush=True)
        print("GLUE_VERDICT=" + ("ACCEPTED_AND_SHIPPED" if improved
                                  else "NO_SHIP (rejected/skipped — see fanout-polish log line)"),
              flush=True)
    finally:
        await opt.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
