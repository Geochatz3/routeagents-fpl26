"""STAGE 3 — RECIPE PASS (deterministic, LLM-free)

The deterministic recipe passes.

Pre-proven tool sequences selected by the design's |WNS| band, run before any
LLM call and enrolled as candidates for the finalize play-off. Nothing here
proposes a move; the band decides, and each pass either measures better or is
discarded.

Named for the stage it implements: the slide, the video and the paper all
call it by this name, so a reader can move between them without a glossary.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

# Log through the orchestrator's logger: this code still belongs to the same
# run, and a module-named logger would change every line it emits.
logger = logging.getLogger("dcp_optimizer")


from typing import Optional

try:
    from scheduler.dispatch import design_name_from_dcp as _design_name_from_dcp
    from scheduler.dispatch import recipe_safe_for as _recipe_safe_for
except Exception:  # pragma: no cover — defensive
    _design_name_from_dcp = None
    _recipe_safe_for = None

try:
    from optimizer.recipe_router import (
        PhaseOneFeatures as _RouterFeatures,
        decide_recipe_path as _decide_recipe_path,
        blocked_reason as _router_blocked_reason,
    )
except Exception:  # pragma: no cover — defensive
    _RouterFeatures = None
    _decide_recipe_path = None
    _router_blocked_reason = None

from optimizer.static_parsers import (
    parse_timing_summary_static as parse_timing_summary_static,
)

from optimizer.finalize_mux import (
    _artifact_identity,
    _atomic_copy,
    _path_is_in_submission,
    _stream_md5,
    _verify_shipped_identity,
    _write_shipped_manifest,
    dcp_zip_sane,
    mux_md5_digest,
    mux_md5_trust_enabled,
)

from optimizer.config_resolution import (  # noqa: F401
    BARE_REROUTE_MAX_ITERS_DEFAULT,
    BARE_REROUTE_MIN_GAIN_NS_DEFAULT,
    DEEP_WNS_TAIL_RESERVE_S_DEFAULT,
    POLISH_RESERVE_S_DEFAULT,
    TAIL_CTRL_DEEP_WNS_NS_DEFAULT,
    TAIL_CTRL_MAX_MOVES_DEFAULT,
    TAIL_RESERVE_STAGNANT_S_DEFAULT,
    _arg_float,
    _arg_int,
    _run_dir_base,
    resolve_bare_reroute_max_iters,
    resolve_bare_reroute_min_gain_ns,
    resolve_bare_reroute_polish_enabled,
    resolve_deep_wns_tail_reserve_s,
    resolve_polish_reserve_s,
    resolve_tail_controller_enabled,
    resolve_tail_ctrl_deep_wns_ns,
    resolve_tail_ctrl_m1_echo,
    resolve_tail_ctrl_max_moves,
    resolve_tail_reserve_stagnant_s,
)

from optimizer.tool_source import (  # noqa: F401
    LINEAGE_SOURCES,
    _PLACE_STMT_RE,
    _ROUTE_STMT_RE,
    _looks_like_tool_error,
    _make_lineage_entry,
    _routed_state_transition,
    _tcl_statements,
    _tool_cmd_text,
    compose_iter1_user_message,
    convert_mcp_tool_to_openai,
    is_routed_state_destroying,
    load_system_prompt,
)

from optimizer.recipe_policy import (  # noqa: F401
    ETO_RETIME_FF_DRIFT_MAX_FRAC,
    ETO_RETIME_WNS_MAG_MAX_NS,
    ETO_RETIME_WNS_MAG_MIN_NS,
    FRESH_PRESWEEP_COST_SAFETY,
    FRESH_PRESWEEP_DRAWS_MAX,
    OWNFRONT_RETIME_SPLIT_NS,
    OWNFRONT_RETIME_WNS_MAG_MAX_NS,
    OWNFRONT_RETIME_WNS_MAG_MIN_NS,
    RECIPE_PASS_DEEP_WNS_MAG_MIN_NS,
    RECIPE_PASS_SHALLOW_WNS_MAG_MAX_NS,
    SHALLOW_DET_WNS_MAG_MAX_NS,
    SHALLOW_DET_WNS_MAG_MIN_NS,
    SUBBAND_CARVEOUT_WNS_MAG_MAX_NS,
    SUBBAND_FLOOR_MAX_FAILING_ENDPOINTS,
    SUBBAND_FLOOR_WNS_MAG_MAX_NS,
    SUBBAND_FLOOR_WNS_MAG_MIN_NS,
    _fanout_polish_cost_basis,
    _presweep_draw_allowance,
    deep_first_sizegated_enabled,
    eto_retime_candidate_enabled,
    eto_retime_ff_drift_ok,
    eto_retime_parse_ffcount,
    eto_retime_subband_match,
    fir_like_failing_ok,
    fir_subband_floor_enabled,
    fir_subband_match,
    midband_retry_hold_enabled,
    midband_route_rung_enabled,
    ownfront_retime_enabled,
    ownfront_retime_front,
    plateau_armed_from_loop_start,
    polish_gate_reserve_s,
    positive_slack_continue_enabled,
    positive_slack_entry_decision,
    recipe_first_deep_enabled,
    recipe_pass_band,
    recipe_pass_enabled,
    resolve_fresh_presweep_draws,
    shallow_det_subband_match,
    shallow_determinizer_enabled,
    subband_carveout_match,
    v40_flag_env_present,
    FRESH_PRESWEEP_DRAWS_DEFAULT,
)

FRESH_PRESWEEP_BUDGET_FRAC = 0.2        # total spend <= 0.2 x max_wall

FRESH_PRESWEEP_FIRST_DRAW_FRAC = 0.12   # draw-1 timeout (first-draw-measures)

FRESH_PRESWEEP_FALLBACK_BUDGET_S = 600.0

FRESH_PRESWEEP_FALLBACK_FIRST_TIMEOUT_S = 360.0

FRESH_PRESWEEP_PIPELINE_REFERENCE_S = 3600.0

FRESH_PRESWEEP_MIN_WALL_S = (
    FRESH_PRESWEEP_PIPELINE_REFERENCE_S / (1.0 - FRESH_PRESWEEP_BUDGET_FRAC)
)  # = 4500.0s

RECIPE_PASS_SHALLOW_EXPECTED_S = 300.0

RECIPE_PASS_DEEP_EXPECTED_S = 900.0

RECIPE_PASS_TIMEOUT_FACTOR = 1.5

RECIPE_PASS_FINALIZE_RESERVE_S = 300.0

RECIPE_PASS_MEASURE_RESERVE_S = 120.0   # get_wns          (timeout cap 600s)

RECIPE_PASS_STORE_RESERVE_S = 180.0     # write_checkpoint (timeout cap 900s)

RECIPE_PASS_REGISTER_RESERVE_S = 240.0  # routed+hold+cell (timeout cap ~540s)

RECIPE_PASS_RESET_RESERVE_S = 360.0     # restart + reopen (cap 600s x scale)

RECIPE_PASS_OVERHEAD_RESERVE_S = (
    RECIPE_PASS_MEASURE_RESERVE_S + RECIPE_PASS_STORE_RESERVE_S
    + RECIPE_PASS_REGISTER_RESERVE_S + RECIPE_PASS_RESET_RESERVE_S
)  # = 900.0

RECIPE_PASS_SESSION_ROUNDTRIP = "__RECIPE_PASS_SESSION_ROUNDTRIP__"

RECIPE_PASS_SHALLOW_TCL = (
    "route_design -unroute",
    "place_design -unplace",
    "place_design -directive WLDrivenBlockPlacement",
    "phys_opt_design",
    "route_design -directive AggressiveExplore",
    "phys_opt_design",
)

RECIPE_PASS_DEEP_TCL = (
    "phys_opt_design -critical_pin_opt",
    "phys_opt_design -critical_pin_opt",
    RECIPE_PASS_SESSION_ROUNDTRIP,
    "route_design -unroute",
    "place_design -unplace",
    "place_design -directive Explore",
    "phys_opt_design",
    "route_design -directive AggressiveExplore",
)

RECIPE_PASS_MID_TCL = (
    # session A
    "phys_opt_design -directive AlternateFlowWithRetiming",
    "route_design -directive Default",
    "place_design -unplace",
    "place_design -directive Explore",
    "route_design -directive Default",
    "phys_opt_design -directive AlternateFlowWithRetiming",
    "route_design -directive Default",
    "route_design -directive Explore",
    RECIPE_PASS_SESSION_ROUNDTRIP,
    # session B (FRESH Vivado — the record restarted here; keep the restart)
    "route_design -unroute",
    "route_design -directive AggressiveExplore",
    "phys_opt_design -directive Explore",
)

RECIPE_PASS_MID_EXPECTED_S = 2800.0

RECIPE_PASS_POSTLOOP_OVERHEAD_RESERVE_S = (
    RECIPE_PASS_MEASURE_RESERVE_S + RECIPE_PASS_STORE_RESERVE_S
    + RECIPE_PASS_REGISTER_RESERVE_S
)  # = 540.0

RECIPE_FIRST_DEEP_COST_RATIO = 2.5

RECIPE_FIRST_DEEP_MARGIN = 1.2

RECIPE_FIRST_DEEP_DYNAMIC_ABORT_FRAC = 0.75

RECIPE_FIRST_DEEP_LLM_FLOOR_FRAC = 0.35

RECIPE_FIRST_DEEP_ANCHOR_FLOOR_S = 100.0


class RecipePassesMixin:
    """Methods mixed into DCPOptimizer; they run against its instance state."""

    async def _recipe_register_retiming(self, args: dict) -> str:
        """Apply register retiming through physical optimization.

        The recipe uses either `AlternateFlowWithRetiming` or `AddRetime`,
        optionally reroutes the result, and measures its timing contribution.
        It returns a JSON object containing status, the selected directive,
        route errors, and the frequency change in MHz.
        """
        directive = args.get("directive", "AlternateFlowWithRetiming")
        if directive not in ("AlternateFlowWithRetiming", "AddRetime"):
            return json.dumps({
                "status": "error",
                "error": f"directive must be AlternateFlowWithRetiming or AddRetime, got {directive!r}",
            })
        rewrite_dcp = bool(args.get("rewrite_dcp", True))

        result = {
            "status": "error",
            "directive_used": directive,
            "route_errors": None,
            "delta_fmax_mhz": None,
            "initial_fmax_mhz": None,
            "final_fmax_mhz": None,
            "error": None,
        }

        pre_recipe_wns = self.best_wns
        try:
            # Step 1 — run phys_opt_design with the retiming directive.
            phys_str = await self.call_tool("vivado_phys_opt_design", {
                "directive": directive,
            })
            if _looks_like_tool_error(phys_str):
                result["error"] = f"phys_opt_design failed: {phys_str[:200]}"
                return json.dumps(result)

            # Step 2 — re-route.  Retiming inserts/moves FFs, which can
            # invalidate the route on at least the affected paths.
            route_str = await self.call_tool("vivado_route_design", {"directive": "Default"})
            if _looks_like_tool_error(route_str):
                result["error"] = f"route_design after retiming failed: {route_str[:200]}"
                return json.dumps(result)

            # Step 3 — check routing health.
            route_status = await self.call_tool("vivado_report_route_status", {})
            m = re.search(r"# of nets with routing errors.*?:\s+(-?\d+)", route_status)
            route_errors = int(m.group(1)) if m else -1
            result["route_errors"] = route_errors

            if route_errors != 0:
                result["status"] = "route_errors"
                result["error"] = (
                    f"route_design left {route_errors} routing errors after "
                    f"{directive} retiming — revert."
                )
                return json.dumps(result)

            # Step 4 — measure.
            await self.call_tool("vivado_report_timing_summary", {})
            initial_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
            pre_fmax = (
                self.calculate_fmax(pre_recipe_wns, self.clock_period)
                if pre_recipe_wns is not None and pre_recipe_wns > float("-inf") else None
            )
            final_fmax = (
                self.calculate_fmax(self.best_wns, self.clock_period)
                if self.best_wns > float("-inf") else None
            )
            if initial_fmax is not None and final_fmax is not None:
                result["initial_fmax_mhz"] = round(initial_fmax, 2)
                result["final_fmax_mhz"] = round(final_fmax, 2)
                if pre_fmax is not None:
                    result["delta_fmax_mhz"] = round(final_fmax - pre_fmax, 2)
                else:
                    result["delta_fmax_mhz"] = round(final_fmax - initial_fmax, 2)
            result["status"] = "success"
            return json.dumps(result)

        except Exception as e:  # pragma: no cover — defensive
            logger.exception("recipe_register_retiming raised")
            result["error"] = repr(e)
            return json.dumps(result)

    async def _recipe_lut_optimization(self, args: dict) -> str:
        """Optimize LUT input cones on critical paths.

        Extracts I0 through I5 pins from the worst paths, optimizes their input
        cones, then writes, opens, routes, and measures the resulting
        checkpoint. A `no_optimization` result is a normal no-op, so the pass
        can run on any design.
        """
        num_paths = _arg_int(args, "num_paths", 10)
        target_max_path = _arg_int(args, "target_max_path", 2)
        max_pins_to_try = _arg_int(args, "max_pins_to_try", 12)

        result = {
            "status": "error",
            "pins_extracted": 0,
            "pins_targeted": [],
            "pins_optimized": [],
            "pins_skipped": [],
            "route_errors": None,
            "delta_fmax_mhz": None,
            "initial_fmax_mhz": None,
            "final_fmax_mhz": None,
            "error": None,
        }

        # Snapshot pre-recipe state for accurate per-recipe delta.
        pre_recipe_wns = self.best_wns

        try:
            # Step 1 — extract critical-path pins.  A fresh file each time, so
            # callers running this twice cannot read stale data.
            pins_file = Path(self.temp_dir) / "recipe_lut_pins.json"
            await self.call_tool("vivado_extract_critical_path_pins", {
                "num_paths": num_paths,
                "output_file": str(pins_file),
            })

            # Step 2 — filter to LUT-input pins on the worst N paths.
            try:
                critical_paths = json.loads(pins_file.read_text())
            except (OSError, json.JSONDecodeError) as e:
                result["error"] = f"could not read critical-path pins JSON: {e}"
                return json.dumps(result)

            candidate_pins = []
            seen = set()
            for path_idx, path in enumerate(critical_paths):
                if path_idx >= target_max_path:
                    break
                if not isinstance(path, list):
                    continue
                for pin in path:
                    if not isinstance(pin, str) or "/" not in pin:
                        continue
                    # LUT input pins end in /I0..I5.  optimize_lut_input_cone
                    # gracefully reports no_optimization for non-LUT cells, so the
                    # cell type needs no check here.
                    if not re.search(r"/I[0-5]$", pin):
                        continue
                    if pin in seen:
                        continue
                    seen.add(pin)
                    candidate_pins.append(pin)
            result["pins_extracted"] = len(candidate_pins)
            result["pins_targeted"] = candidate_pins[:max_pins_to_try]

            if not result["pins_targeted"]:
                result["status"] = "no_candidates"
                return json.dumps(result)

            # Step 3 — call optimize_lut_input_cone (single batched call).
            opt_str = await self.call_tool("rapidwright_optimize_lut_input_cone", {
                "hierarchical_input_pins": result["pins_targeted"],
            })
            try:
                opt = json.loads(opt_str) if isinstance(opt_str, str) else opt_str
            except (json.JSONDecodeError, TypeError) as e:
                result["error"] = f"optimize_lut_input_cone returned non-JSON: {e}"
                return json.dumps(result)
            if isinstance(opt, dict) and opt.get("error"):
                result["error"] = f"optimize_lut_input_cone: {opt['error']}"
                return json.dumps(result)

            for r in opt.get("results", []) if isinstance(opt, dict) else []:
                if r.get("status") == "optimized":
                    result["pins_optimized"].append(r.get("pin"))
                else:
                    result["pins_skipped"].append({
                        "pin": r.get("pin"),
                        "status": r.get("status"),
                        "message": r.get("message"),
                    })

            if not result["pins_optimized"]:
                result["status"] = "no_optimization_possible"
                return json.dumps(result)

            # Step 4 — write / open / route / measure (with error gates).
            rw_out = Path(self.temp_dir) / "recipe_lut_optimized.dcp"
            write_str = await self.call_tool("rapidwright_write_checkpoint", {
                "dcp_path": str(rw_out),
                "overwrite": True,
            })
            if _looks_like_tool_error(write_str):
                result["error"] = f"rapidwright_write_checkpoint failed: {write_str[:200]}"
                return json.dumps(result)
            if not rw_out.exists():
                result["error"] = f"write_checkpoint succeeded but {rw_out} is missing"
                return json.dumps(result)

            open_str = await self.call_tool("vivado_open_checkpoint", {
                "dcp_path": str(rw_out),
            })
            if _looks_like_tool_error(open_str):
                result["error"] = f"vivado_open_checkpoint failed: {open_str[:200]}"
                return json.dumps(result)

            route_str = await self.call_tool("vivado_route_design", {"directive": "Default"})
            if _looks_like_tool_error(route_str):
                result["error"] = f"vivado_route_design failed: {route_str[:200]}"
                return json.dumps(result)

            route_status = await self.call_tool("vivado_report_route_status", {})
            m = re.search(r"# of nets with routing errors.*?:\s+(-?\d+)", route_status)
            route_errors = int(m.group(1)) if m else -1
            result["route_errors"] = route_errors

            if route_errors != 0:
                result["status"] = "route_errors"
                result["error"] = (
                    f"route_design left {route_errors} routing errors — "
                    "do NOT accept this DCP."
                )
                return json.dumps(result)

            await self.call_tool("vivado_report_timing_summary", {})
            initial_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
            pre_fmax = (
                self.calculate_fmax(pre_recipe_wns, self.clock_period)
                if pre_recipe_wns is not None and pre_recipe_wns > float("-inf") else None
            )
            final_fmax = (
                self.calculate_fmax(self.best_wns, self.clock_period)
                if self.best_wns > float("-inf") else None
            )
            if initial_fmax is not None and final_fmax is not None:
                result["initial_fmax_mhz"] = round(initial_fmax, 2)
                result["final_fmax_mhz"] = round(final_fmax, 2)
                # delta_fmax_mhz reflects the recipe's own contribution.
                if pre_fmax is not None:
                    result["delta_fmax_mhz"] = round(final_fmax - pre_fmax, 2)
                else:
                    result["delta_fmax_mhz"] = round(final_fmax - initial_fmax, 2)
            result["status"] = "success"
            return json.dumps(result)

        except Exception as e:  # pragma: no cover — defensive
            logger.exception("recipe_lut_optimization raised")
            result["error"] = repr(e)
            return json.dumps(result)

    async def _recipe_cell_replacement(self, args: dict) -> str:
        """Synthetic high-level tool — runs the contest's published cell-
        replacement recipe via the optimizer's existing MCP sessions.

        Returns a JSON string with the recipe outcome.  Sub-steps go
        through self.call_tool(), so WNS tracking, force-continue logic,
        and tool_call_details all see each underlying call.

        Refuses on designs the recipe-applicability table marks as
        harmful or error.  No-cells-moved and no-candidates branches
        return cleanly without writing a DCP — the LLM can pick a
        different strategy on the next iter.
        """
        num_paths = _arg_int(args, "num_paths", 10)
        detour_threshold = _arg_float(args, "detour_threshold", 2.0)
        target_max_path = _arg_int(args, "target_max_path", 2)
        max_cells_to_try = _arg_int(args, "max_cells_to_try", 8)

        result = {
            "status": "error",
            "candidates_total": 0,
            "candidates_targeted": [],
            "cells_moved": [],
            "cells_unmoved": [],
            "route_errors": None,
            "delta_fmax_mhz": None,
            "initial_fmax_mhz": None,
            "final_fmax_mhz": None,
            "error": None,
        }

        # Apply the optional design-specific safety gate only outside contest mode.
        # Unknown or hidden designs proceed through normal execution, with tool
        # errors handled by the standard failure path.
        design = getattr(self, "_design_name_for_memory", None)
        contest_mode = bool(getattr(self, "contest_mode", False))
        if (_recipe_safe_for is not None and design
                and not contest_mode
                and not _recipe_safe_for(design)):
            result["error"] = (
                f"recipe_cell_replacement is BLOCKED for '{design}' — prior sweeps "
                "marked it harmful or error.  Choose a different strategy."
            )
            return json.dumps(result)

        # Snapshot pre-recipe state so delta_fmax_mhz reflects the recipe's
        # own contribution, not the cumulative best since run start.
        pre_recipe_wns = self.best_wns

        try:
            # Step 1 — extract critical-path pins (Vivado already has design open).
            pins_file = Path(self.temp_dir) / "recipe_critical_path_pins.json"
            await self.call_tool("vivado_extract_critical_path_pins", {
                "num_paths": num_paths,
                "output_file": str(pins_file),
            })

            # Step 2 — RapidWright detour analysis.
            analysis_str = await self.call_tool("rapidwright_analyze_net_detour", {
                "input_file": str(pins_file),
                "detour_threshold": detour_threshold,
            })
            try:
                analysis = json.loads(analysis_str) if isinstance(analysis_str, str) else analysis_str
            except (json.JSONDecodeError, TypeError) as e:
                result["error"] = f"analyze_net_detour returned non-JSON: {e}"
                return json.dumps(result)
            if isinstance(analysis, dict) and analysis.get("error"):
                result["error"] = f"analyze_net_detour: {analysis['error']}"
                return json.dumps(result)
            candidates = analysis.get("candidates", []) if isinstance(analysis, dict) else []
            result["candidates_total"] = len(candidates)

            # Filter to unique cells on the worst N paths, preserving detour-
            # ratio descending order.
            seen = set()
            target_cells = []
            for c in candidates:
                if not isinstance(c, dict):
                    continue
                p = c.get("path")
                if p is None or p > target_max_path:
                    continue
                name = str(c.get("cell"))
                if name in seen or name == "None":
                    continue
                seen.add(name)
                target_cells.append(name)
            result["candidates_targeted"] = target_cells[:max_cells_to_try]

            if not result["candidates_targeted"]:
                result["status"] = "no_candidates"
                return json.dumps(result)

            # Step 3 — try cells one at a time so a single bad candidate
            # doesn't kill the whole recipe (RapidWright NPE on certain
            # cell types).
            moved_any = False
            for cell_name in result["candidates_targeted"]:
                try:
                    opt_str = await self.call_tool("rapidwright_optimize_cell_placement", {
                        "cell_names": [cell_name],
                    })
                    opt = json.loads(opt_str) if isinstance(opt_str, str) else opt_str
                except (json.JSONDecodeError, TypeError, RuntimeError) as e:
                    result["cells_unmoved"].append(
                        {"cell": cell_name, "status": "exception", "message": str(e)}
                    )
                    continue
                opt_results = opt.get("results", []) if isinstance(opt, dict) else []
                if not opt_results:
                    result["cells_unmoved"].append(
                        {"cell": cell_name, "status": "internal_error",
                         "message": "RapidWright returned no results (internal exception)"}
                    )
                    continue
                for r in opt_results:
                    if r.get("status") == "success":
                        result["cells_moved"].append(r.get("cell"))
                        moved_any = True
                    else:
                        result["cells_unmoved"].append({
                            "cell": r.get("cell"),
                            "status": r.get("status"),
                            "message": r.get("message"),
                        })

            if not moved_any:
                result["status"] = "no_cells_moved"
                return json.dumps(result)

            # Step 4 — write RapidWright DCP, open in Vivado, route, measure.
            # call_tool returns JSON-error strings on failure (does not raise),
            # so each step's output is checked explicitly before proceeding.
            rw_out = Path(self.temp_dir) / "recipe_rw_optimized.dcp"
            write_str = await self.call_tool("rapidwright_write_checkpoint", {
                "dcp_path": str(rw_out),
                "overwrite": True,
            })
            if _looks_like_tool_error(write_str):
                result["error"] = f"rapidwright_write_checkpoint failed: {write_str[:200]}"
                return json.dumps(result)
            if not rw_out.exists():
                result["error"] = f"write_checkpoint succeeded but {rw_out} is missing"
                return json.dumps(result)

            open_str = await self.call_tool("vivado_open_checkpoint", {
                "dcp_path": str(rw_out),
            })
            if _looks_like_tool_error(open_str):
                result["error"] = f"vivado_open_checkpoint failed: {open_str[:200]}"
                return json.dumps(result)

            route_str = await self.call_tool("vivado_route_design", {"directive": "Default"})
            if _looks_like_tool_error(route_str):
                result["error"] = f"vivado_route_design failed: {route_str[:200]}"
                return json.dumps(result)

            route_status = await self.call_tool("vivado_report_route_status", {})
            m = re.search(r"# of nets with routing errors.*?:\s+(-?\d+)", route_status)
            route_errors = int(m.group(1)) if m else -1
            result["route_errors"] = route_errors

            # Hard gate: any routing errors → recipe failed for this design.
            if route_errors != 0:
                result["status"] = "route_errors"
                result["error"] = (
                    f"route_design left {route_errors} routing errors — "
                    "do NOT accept this DCP.  Try a different strategy."
                )
                return json.dumps(result)

            # Measure ΔFmax via the optimizer's existing tracking machinery.
            await self.call_tool("vivado_report_timing_summary", {})
            # Report both the recipe-local frequency change and cumulative change,
            # in MHz. The local delta determines whether the recipe helped; the
            # cumulative value describes the checkpoint or revert state.
            initial_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
            pre_fmax = (
                self.calculate_fmax(pre_recipe_wns, self.clock_period)
                if pre_recipe_wns is not None and pre_recipe_wns > float("-inf") else None
            )
            final_fmax = (
                self.calculate_fmax(self.best_wns, self.clock_period)
                if self.best_wns > float("-inf") else None
            )
            if initial_fmax is not None and final_fmax is not None:
                result["initial_fmax_mhz"] = round(initial_fmax, 2)
                result["final_fmax_mhz"] = round(final_fmax, 2)
                # delta_fmax_mhz: recipe's contribution (post − pre)
                if pre_fmax is not None:
                    result["delta_fmax_mhz"] = round(final_fmax - pre_fmax, 2)
                else:
                    # No pre-recipe baseline → fall back to run-level delta
                    result["delta_fmax_mhz"] = round(final_fmax - initial_fmax, 2)
            result["status"] = "success"
            return json.dumps(result)

        except Exception as e:  # pragma: no cover — defensive
            logger.exception("recipe_cell_replacement raised")
            result["error"] = repr(e)
            return json.dumps(result)

    async def _recipe_post_route_phys_opt_sweep(self, args: dict) -> str:
        """Run a bounded sweep of granular physical optimizations on a routed
        design.

        Runs one sub-optimization at a time to bound cost and isolate its
        effect. Improvements are retained; regressions reopen `best_valid.dcp`
        so the tool session matches the tracked best WNS. Returns JSON
        containing status, WNS delta in ns, Fmax delta in MHz, and elapsed time
        in seconds for each flag, plus cumulative deltas.
        """
        flags_arg = args.get("flags")
        flags = list(flags_arg) if flags_arg else list(self._DEFAULT_PHYS_OPT_SWEEP)
        epsilon_ns = _arg_float(args, "epsilon_ns", 0.010)
        skip_route_recheck = bool(args.get("skip_route_recheck", True))
        # Onion-layering options.
        early_exit_on_first_gain = bool(args.get("early_exit_on_first_gain", True))
        max_successful_subpasses = _arg_int(args, "max_successful_subpasses", 1)
        min_remaining_time_for_next_layer_s = float(
            args.get("min_remaining_time_for_next_layer_s", 600.0)
        )

        # Cap to known sub-flags so a stray entry can't smuggle a
        # full-directive call past the dispatcher gate.
        valid_flags = {
            "critical_cell_opt", "equ_drivers_opt", "placement_opt",
            "dsp_register_opt", "restruct_opt", "slr_crossing_opt",
        }
        flags = [f for f in flags if f in valid_flags]
        if not flags:
            return json.dumps({
                "status": "error",
                "error": "no valid flags after filtering; nothing to sweep",
            })

        result = {
            "status": "running",
            "flags_attempted": [],
            "flags_committed": [],
            "flags_reverted": [],
            "per_flag": [],
            "initial_wns_ns": None,
            "final_wns_ns": None,
            "initial_fmax_mhz": None,
            "final_fmax_mhz": None,
            "delta_wns_ns": 0.0,
            "delta_fmax_mhz": 0.0,
            "elapsed_total_s": 0.0,
            "error": None,
        }

        pre_recipe_wns = self.best_wns
        if pre_recipe_wns is None or pre_recipe_wns == float("-inf"):
            return json.dumps({
                "status": "error",
                "error": "best_wns not yet established; run report_timing_summary first",
            })
        result["initial_wns_ns"] = float(pre_recipe_wns)
        result["initial_fmax_mhz"] = (
            self.calculate_fmax(pre_recipe_wns, self.clock_period)
        )

        sweep_start = time.time()

        for flag in flags:
            # Budget pre-flight at the sweep level — abort gracefully
            # if remaining < a conservative single-flag estimate.
            single_flag_estimate_s = self._estimate_tool_runtime(
                "vivado_phys_opt_design", is_risky=True)
            # Use a smaller estimate for sub-flag calls since they're
            # bounded — but err high: half the full-directive estimate.
            single_flag_estimate_s = max(120.0, single_flag_estimate_s * 0.5)
            if self._budget_deadline is not None:
                if self._budget_remaining() < single_flag_estimate_s:
                    logger.info(
                        f"phys_opt_sweep: skipping {flag} — remaining "
                        f"{self._budget_remaining():.0f}s < est "
                        f"{single_flag_estimate_s:.0f}s"
                    )
                    result["per_flag"].append({
                        "flag": flag,
                        "status": "skipped_budget",
                        "remaining_s": self._budget_remaining(),
                    })
                    break

            flag_start = time.time()
            pre_wns = self.best_wns
            result["flags_attempted"].append(flag)
            try:
                # Call phys_opt_design with ONLY this sub-flag.  directive is
                # deliberately NOT passed — the MCP server rejects mixing
                # directive with specific options.
                phys_str = await self.call_tool(
                    "vivado_phys_opt_design", {flag: True}
                )
                if _looks_like_tool_error(phys_str):
                    elapsed_s = time.time() - flag_start
                    result["per_flag"].append({
                        "flag": flag,
                        "status": "tool_error",
                        "elapsed_s": round(elapsed_s, 1),
                        "error": str(phys_str)[:200],
                    })
                    # Continue to next flag; one error doesn't poison the sweep.
                    continue

                # Optionally verify routing didn't break.  phys_opt is
                # conservative about legality, but the paranoid mode is
                # available.  Pre-flight check the route call too.
                route_errors = None
                if not skip_route_recheck:
                    rs = await self.call_tool("vivado_report_route_status", {})
                    if not _looks_like_tool_error(rs):
                        m = re.search(r"# of nets with routing errors.*?:\s+(-?\d+)", rs)
                        if m:
                            route_errors = int(m.group(1))

                # Measure WNS using a cheap call (get_wns is faster than a full
                # timing_summary).  call_tool's improvement-detection branch
                # updates self.best_wns as a side effect.
                wns_str = await self.call_tool("vivado_get_wns", {})
                if _looks_like_tool_error(wns_str):
                    elapsed_s = time.time() - flag_start
                    result["per_flag"].append({
                        "flag": flag,
                        "status": "measure_failed",
                        "elapsed_s": round(elapsed_s, 1),
                        "error": str(wns_str)[:200],
                    })
                    continue

                try:
                    post_wns = float(wns_str.strip())
                except (ValueError, AttributeError):
                    elapsed_s = time.time() - flag_start
                    result["per_flag"].append({
                        "flag": flag,
                        "status": "measure_parse_failed",
                        "elapsed_s": round(elapsed_s, 1),
                        "wns_str": wns_str[:80],
                    })
                    continue

                elapsed_s = time.time() - flag_start
                delta_wns_ns = post_wns - (
                    pre_wns if pre_wns is not None and pre_wns != float("-inf")
                    else post_wns
                )
                delta_fmax_mhz = 0.0
                if self.clock_period:
                    pre_fmax = self.calculate_fmax(pre_wns, self.clock_period) or 0.0
                    post_fmax = self.calculate_fmax(post_wns, self.clock_period) or 0.0
                    delta_fmax_mhz = post_fmax - pre_fmax

                entry = {
                    "flag": flag,
                    "elapsed_s": round(elapsed_s, 1),
                    "pre_wns_ns": round(pre_wns, 4) if pre_wns is not None else None,
                    "post_wns_ns": round(post_wns, 4),
                    "delta_wns_ns": round(delta_wns_ns, 4),
                    "delta_fmax_mhz": round(delta_fmax_mhz, 3),
                    "route_errors": route_errors,
                }

                # Keep gains at or above the WNS epsilon; the tool-call handler has
                # already mirrored that best state. Otherwise restore best_valid.dcp
                # so a no-gain or regressing subpass cannot degrade the session.
                if delta_wns_ns >= epsilon_ns:
                    entry["status"] = "committed"
                    result["flags_committed"].append(flag)
                else:
                    entry["status"] = "reverted_no_gain" if delta_wns_ns >= -epsilon_ns else "reverted_regression"
                    result["flags_reverted"].append(flag)
                    # Revert Vivado state to best_valid.dcp so the next
                    # sub-flag starts from the right baseline.
                    if (self._best_valid_dcp is not None
                            and Path(self._best_valid_dcp).exists()):
                        try:
                            revert_str = await self.call_tool(
                                "vivado_open_checkpoint",
                                {"dcp_path": str(self._best_valid_dcp.resolve())},
                            )
                            if _looks_like_tool_error(revert_str):
                                entry["revert_status"] = "open_checkpoint_failed"
                            else:
                                entry["revert_status"] = "reopened_best_valid"
                                # best_wns is a soft tracker; reset it
                                # to pre_wns since Vivado's state matches
                                # the prior best now.
                                self.best_wns = pre_wns if pre_wns is not None else self.best_wns
                        except Exception as e:
                            entry["revert_status"] = f"revert_raised:{type(e).__name__}"
                    else:
                        entry["revert_status"] = "no_best_valid_to_revert_to"

                result["per_flag"].append(entry)
                logger.info(
                    f"phys_opt_sweep[{flag}]: pre_wns={pre_wns:.3f} → "
                    f"post_wns={post_wns:.3f}, Δ={delta_wns_ns:+.3f} ns, "
                    f"status={entry['status']}, elapsed={elapsed_s:.1f}s"
                )

                # The optional counter tracks consecutive no-gain or regression
                # reverts. Any commit resets it, regardless of gain size; other
                # statuses leave it unchanged. In mid-sweep mode, check after
                # each update and stop on a triggered or unaffordable rescue.
                # End-sweep mode defers the check until completion.
                if self.phys_opt_preempt_after is not None:
                    _entry_status = entry.get("status")
                    if _entry_status == "committed":
                        self._physopt_no_gain_streak = 0
                    elif _entry_status in ("reverted_no_gain",
                                            "reverted_regression"):
                        self._physopt_no_gain_streak += 1
                    if getattr(self, "phys_opt_preempt_mode",
                               "end_sweep") == "mid_sweep":
                        # Index of current flag in the original flags
                        # list; flags_remaining excludes current.
                        try:
                            _idx = flags.index(flag)
                            _remaining = max(0, len(flags) - _idx - 1)
                        except (ValueError, AttributeError):
                            _remaining = None
                        _outcome = self._maybe_emit_phys_opt_preempt(
                            result,
                            current_flag=flag,
                            flags_remaining=_remaining,
                        )
                        if _outcome == "triggered":
                            result["early_exit_reason"] = "phys_opt_preempt"
                            logger.info(
                                "phys_opt_sweep: early-exit — preempt "
                                f"triggered mid-sweep after flag "
                                f"{flag!r}; breaking {_remaining} "
                                "remaining flag(s)."
                            )
                            break
                        if _outcome == "skipped_budget":
                            result["early_exit_reason"] = (
                                "phys_opt_preempt_insufficient_budget"
                            )
                            logger.info(
                                "phys_opt_sweep: early-exit — preempt "
                                f"streak met at flag {flag!r} but "
                                "budget gate failed; breaking remaining "
                                "flags."
                            )
                            break

                # After a commit, optionally return control to the outer loop so it
                # can select a different transform for the newly exposed bottleneck
                # instead of spending the remaining budget within this recipe.
                if entry["status"] == "committed":
                    committed_count = len(result["flags_committed"])
                    if (early_exit_on_first_gain
                            and committed_count >= max_successful_subpasses):
                        result["early_exit_reason"] = (
                            f"early_exit_on_first_gain (committed {committed_count}"
                            f"/{max_successful_subpasses})"
                        )
                        logger.info(
                            f"phys_opt_sweep: early-exit after {committed_count} "
                            "commit(s); leaving wall time for next-layer recipe."
                        )
                        break
                    if (self._budget_deadline is not None
                            and self._budget_remaining() < min_remaining_time_for_next_layer_s):
                        result["early_exit_reason"] = (
                            f"remaining {self._budget_remaining():.0f}s < "
                            f"min_remaining_for_next_layer "
                            f"{min_remaining_time_for_next_layer_s:.0f}s"
                        )
                        logger.info(
                            f"phys_opt_sweep: early-exit — "
                            f"remaining {self._budget_remaining():.0f}s "
                            "below next-layer floor; finalize wins so far."
                        )
                        break

            except Exception as e:
                elapsed_s = time.time() - flag_start
                result["per_flag"].append({
                    "flag": flag,
                    "status": "raised",
                    "elapsed_s": round(elapsed_s, 1),
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                })
                logger.exception(f"phys_opt_sweep[{flag}] raised")
                # Don't break the sweep — try the next flag.

        result["elapsed_total_s"] = round(time.time() - sweep_start, 1)
        # Cumulative deltas relative to the recipe's starting point.
        result["final_wns_ns"] = float(self.best_wns) if self.best_wns is not None else None
        if (self.best_wns is not None and self.best_wns != float("-inf")
                and self.clock_period):
            final_fmax = self.calculate_fmax(self.best_wns, self.clock_period)
            initial_fmax = result["initial_fmax_mhz"]
            if final_fmax is not None and initial_fmax is not None:
                result["final_fmax_mhz"] = round(final_fmax, 2)
                result["delta_fmax_mhz"] = round(final_fmax - initial_fmax, 3)
        if self.best_wns is not None and pre_recipe_wns is not None:
            result["delta_wns_ns"] = round(self.best_wns - pre_recipe_wns, 4)
        result["status"] = (
            "success" if result["flags_committed"] else "no_gain"
        )

        # After the configured no-gain streak, optionally direct the LLM to run
        # the heavy place-and-route rescue next. Emit the request only when the
        # predicted remaining wall time can cover both operations.
        # The request fires at most once per optimization run.
        self._maybe_emit_phys_opt_preempt(result)

        return json.dumps(result)

    async def _recipe_high_fanout_timing_replication(self, args: dict) -> str:
        """Replicate high-fanout drivers associated with critical timing paths.

        Finds parent nets exhibiting critical fanout pathology and applies
        forced driver replication without a full optimization directive or
        replacement pass. Commits an improvement and reverts a regression. This
        focused operation is appropriate when a small number of nets dominate
        the critical paths.
        """
        num_paths = _arg_int(args, "num_paths", 20)
        min_fanout = _arg_int(args, "min_fanout", 100)
        max_nets = _arg_int(args, "max_nets_to_replicate", 10)
        epsilon_ns = _arg_float(args, "epsilon_ns", 0.010)

        result = {
            "status": "error",
            "candidates_total": 0,
            "nets_replicated": [],
            "pre_wns_ns": None,
            "post_wns_ns": None,
            "delta_wns_ns": 0.0,
            "delta_fmax_mhz": 0.0,
            "elapsed_total_s": 0.0,
            "revert_status": None,
            "error": None,
        }

        pre_wns = self.best_wns
        if pre_wns is None or pre_wns == float("-inf"):
            result["error"] = "best_wns not yet established; run report_timing_summary first"
            return json.dumps(result)
        result["pre_wns_ns"] = float(pre_wns)

        sweep_start = time.time()
        try:
            # Step 1 — IDENTIFY: get high-fanout nets on critical paths.
            hf_str = await self.call_tool("vivado_get_critical_high_fanout_nets", {
                "num_paths": num_paths,
                "min_fanout": min_fanout,
                "exclude_clocks": True,
            })
            if _looks_like_tool_error(hf_str):
                result["error"] = f"get_critical_high_fanout_nets failed: {hf_str[:200]}"
                return json.dumps(result)

            # Parse the response.  The Vivado MCP returns one net name per
            # line, optionally with a fanout suffix.  Both formats are
            # tolerated deliberately, since the server has shipped both.
            nets = []
            for raw in hf_str.splitlines():
                raw = raw.strip()
                if not raw or raw.startswith("#") or raw.startswith("(") or raw.startswith("✓"):
                    continue
                # Drop any trailing "  fanout=NNN" annotation if present.
                name = raw.split()[0]
                if name and "/" in name or "." in name or "[" in name:
                    nets.append(name)
            result["candidates_total"] = len(nets)
            if not nets:
                result["status"] = "no_candidates"
                result["error"] = "no high-fanout nets found on critical paths"
                return json.dumps(result)

            # Filter to the top N — too many nets makes the
            # force_replication_on_nets argument huge and slow.
            nets = nets[:max_nets]
            result["nets_replicated"] = nets

            # Step 2 — OPTIMIZE: phys_opt with force_replication_on_nets.
            # The MCP expects a single string, so the net list is wrapped as a
            # Tcl get_nets call for Vivado to resolve any wildcards safely.
            tcl_nets = " ".join(nets)
            force_arg = f"[get_nets {{{tcl_nets}}}]"
            phys_str = await self.call_tool("vivado_phys_opt_design", {
                "force_replication_on_nets": force_arg,
            })
            if _looks_like_tool_error(phys_str):
                result["error"] = f"phys_opt_design (replication) failed: {phys_str[:200]}"
                return json.dumps(result)

            # Step 3 — verify route still legal (cheap).
            rs = await self.call_tool("vivado_report_route_status", {})
            if not _looks_like_tool_error(rs):
                m = re.search(r"# of nets with routing errors.*?:\s+(-?\d+)", rs)
                if m:
                    route_errors = int(m.group(1))
                    result["route_errors"] = route_errors
                    if route_errors != 0:
                        # Replication broke routing — revert.
                        result["status"] = "route_errors_revert"
                        result["error"] = f"replication left {route_errors} routing errors"
                        await self._revert_to_best_valid(result)
                        return json.dumps(result)

            # Step 4 — MEASURE.
            wns_str = await self.call_tool("vivado_get_wns", {})
            if _looks_like_tool_error(wns_str):
                result["status"] = "measure_failed_tool_error"
                result["error"] = f"get_wns after replication failed: {wns_str[:200]}"
                # Vivado state may be inconsistent (phys_opt + replication
                # can leave partial routing).  Revert to best_valid so the
                # LLM's next move starts from a known-good state.
                await self._revert_to_best_valid(result)
                return json.dumps(result)
            try:
                post_wns = float(wns_str.strip())
            except (ValueError, AttributeError):
                result["status"] = "measure_failed_parse"
                result["error"] = f"could not parse WNS: {wns_str[:120]!r}"
                await self._revert_to_best_valid(result)
                return json.dumps(result)
            result["post_wns_ns"] = post_wns

            delta_wns_ns = post_wns - pre_wns
            result["delta_wns_ns"] = round(delta_wns_ns, 4)
            if self.clock_period:
                pre_fmax = self.calculate_fmax(pre_wns, self.clock_period) or 0.0
                post_fmax = self.calculate_fmax(post_wns, self.clock_period) or 0.0
                result["delta_fmax_mhz"] = round(post_fmax - pre_fmax, 3)

            # Step 5 — COMMIT or REVERT.
            if delta_wns_ns >= epsilon_ns:
                result["status"] = "committed"
            else:
                result["status"] = "reverted_no_gain" if delta_wns_ns >= -epsilon_ns else "reverted_regression"
                await self._revert_to_best_valid(result)

        except Exception as e:
            logger.exception("recipe_high_fanout_timing_replication raised")
            result["status"] = "raised"
            result["error"] = repr(e)
        finally:
            result["elapsed_total_s"] = round(time.time() - sweep_start, 1)
            # Never leak status="error" with a populated nets_replicated —
            # if step 2 succeeded and a later step failed silently, mark it,
            # so the LLM can tell "didn't run" from "ran, no signal".
            if result["status"] == "error" and result["nets_replicated"]:
                result["status"] = "no_status_recorded_check_logs"
                if not result["error"]:
                    result["error"] = (
                        "phys_opt ran but no terminal status fired — "
                        "likely fell through after route-status check"
                    )

        return json.dumps(result)

    async def _recipe_critical_path_focused_phys_opt(self, args: dict) -> str:
        """Apply bounded physical optimization to the worst critical paths.

        Collects the worst paths, groups their endpoint pins, and applies one
        selected sub-optimization only to that path group. Checks route status
        and WNS, committing improvements of at least epsilon and otherwise
        reopening `best_valid.dcp`. Restricting the scope bounds runtime while
        preserving access to later transformations.
        """
        num_paths = _arg_int(args, "num_paths", 20)
        sub_flag = str(args.get("sub_flag", "critical_cell_opt"))
        epsilon_ns = _arg_float(args, "epsilon_ns", 0.010)

        # Guard: only allow known sub-flags so a stray name can't
        # smuggle a full-directive call past the dispatcher.
        valid_sub_flags = {
            "critical_cell_opt", "equ_drivers_opt", "placement_opt",
            "restruct_opt", "dsp_register_opt", "critical_pin_opt",
        }
        if sub_flag not in valid_sub_flags:
            return json.dumps({
                "status": "error",
                "error": f"invalid sub_flag {sub_flag!r}; must be in {sorted(valid_sub_flags)}",
            })

        pg_name = "pg_critical_focus"
        result = {
            "status": "error",
            "num_paths": num_paths,
            "sub_flag": sub_flag,
            "path_group_name": pg_name,
            "endpoints_captured": 0,
            "pre_wns_ns": None,
            "post_wns_ns": None,
            "delta_wns_ns": 0.0,
            "delta_fmax_mhz": 0.0,
            "elapsed_total_s": 0.0,
            "scoped_phys_opt_elapsed_s": 0.0,
            "revert_status": None,
            "error": None,
        }

        pre_wns = self.best_wns
        if pre_wns is None or pre_wns == float("-inf"):
            result["error"] = "best_wns not yet established; run report_timing_summary first"
            return json.dumps(result)
        result["pre_wns_ns"] = float(pre_wns)

        sweep_start = time.time()
        try:
            # Fetch critical endpoints in one Tcl round trip to preserve a clean list.
            # Vivado rejects -delay_type combined with -setup or -hold;
            # -setup already selects maximum-delay analysis.
            tcl_extract = (
                f"set tps [get_timing_paths -max_paths {num_paths} "
                f"-setup -sort_by slack]; "
                f"set ends [list]; "
                f"foreach tp $tps {{ "
                f"  set ep [get_property ENDPOINT_PIN $tp]; "
                f"  if {{$ep ne {{}}}} {{lappend ends $ep}} "
                f"}}; "
                f"puts \"ENDPOINTS:[llength $ends]\"; "
                f"foreach e $ends {{ puts \"EP:$e\" }}"
            )
            extract_str = await self.call_tool(
                "vivado_run_tcl", {"command": tcl_extract}
            )
            if _looks_like_tool_error(extract_str):
                result["error"] = f"endpoint extraction failed: {str(extract_str)[:200]}"
                return json.dumps(result)
            endpoints: list[str] = []
            for line in str(extract_str).splitlines():
                line = line.strip()
                if line.startswith("EP:"):
                    ep = line[3:].strip()
                    if ep:
                        endpoints.append(ep)
            result["endpoints_captured"] = len(endpoints)
            if not endpoints:
                result["status"] = "no_endpoints"
                result["error"] = "no critical path endpoints captured (get_timing_paths returned empty)"
                return json.dumps(result)

            # Clear an existing group with the same name before recreating it.
            # Recipes may run repeatedly within one tool session.
            endpoints_tcl = " ".join("{" + e + "}" for e in endpoints)
            tcl_pg = (
                f"catch {{group_path -name {pg_name} -default}}; "
                f"group_path -name {pg_name} -to [get_pins {{{' '.join(endpoints)}}}]; "
                f"puts \"PATH_GROUP:{pg_name}\""
            )
            pg_str = await self.call_tool(
                "vivado_run_tcl", {"command": tcl_pg}
            )
            if _looks_like_tool_error(pg_str):
                result["error"] = f"group_path setup failed: {str(pg_str)[:200]}"
                return json.dumps(result)

            # Step 3 — OPTIMIZE: scoped phys_opt with the chosen
            # sub-flag and a custom path group.
            phys_start = time.time()
            phys_str = await self.call_tool("vivado_phys_opt_design", {
                sub_flag: True,
                "path_groups": pg_name,
            })
            result["scoped_phys_opt_elapsed_s"] = round(time.time() - phys_start, 1)
            if _looks_like_tool_error(phys_str):
                result["error"] = f"scoped phys_opt failed: {str(phys_str)[:200]}"
                await self._revert_to_best_valid(result)
                return json.dumps(result)

            # Step 4 — MEASURE.  Use the cheap get_wns call.
            wns_str = await self.call_tool("vivado_get_wns", {})
            if _looks_like_tool_error(wns_str):
                result["status"] = "measure_failed_tool_error"
                result["error"] = f"get_wns after scoped phys_opt failed: {str(wns_str)[:200]}"
                await self._revert_to_best_valid(result)
                return json.dumps(result)
            try:
                post_wns = float(str(wns_str).strip())
            except (ValueError, AttributeError):
                result["status"] = "measure_failed_parse"
                result["error"] = f"could not parse WNS: {str(wns_str)[:120]!r}"
                await self._revert_to_best_valid(result)
                return json.dumps(result)
            result["post_wns_ns"] = post_wns

            delta_wns_ns = post_wns - pre_wns
            result["delta_wns_ns"] = round(delta_wns_ns, 4)
            if self.clock_period:
                pre_fmax = self.calculate_fmax(pre_wns, self.clock_period) or 0.0
                post_fmax = self.calculate_fmax(post_wns, self.clock_period) or 0.0
                result["delta_fmax_mhz"] = round(post_fmax - pre_fmax, 3)

            # Step 5 — COMMIT or REVERT.
            if delta_wns_ns >= epsilon_ns:
                result["status"] = "committed"
            else:
                result["status"] = "reverted_no_gain" if delta_wns_ns >= -epsilon_ns else "reverted_regression"
                await self._revert_to_best_valid(result)

        except Exception as e:
            logger.exception("recipe_critical_path_focused_phys_opt raised")
            result["status"] = "raised"
            result["error"] = repr(e)
        finally:
            result["elapsed_total_s"] = round(time.time() - sweep_start, 1)

        return json.dumps(result)

    def _presweep_now(self) -> float:
        """Clock seam for the pre-sweep budget accounting (tests script
        it; production is time.time)."""
        return time.time()

    def _presweep_budget_shape(self) -> tuple[float, float]:
        """(budget_total_s, first_draw_timeout_s) for this run.

        With a wall cap: 0.2 x wall total and 0.12 x wall
        for the first-draw-measures timeout.  Without one (local
        probes): conservative fixed fallbacks."""
        if self.max_wall_seconds:
            wall = float(self.max_wall_seconds)
            return (FRESH_PRESWEEP_BUDGET_FRAC * wall,
                    FRESH_PRESWEEP_FIRST_DRAW_FRAC * wall)
        return (FRESH_PRESWEEP_FALLBACK_BUDGET_S,
                FRESH_PRESWEEP_FALLBACK_FIRST_TIMEOUT_S)

    async def _presweep_reopen(self, dcp_path: Path) -> bool:
        """Re-open a checkpoint from disk (the pristine input DCP after
        a worse/failed draw — the never-worse insurance — or the banked
        best_valid mirror at hand-off).  Returns True on success;
        open_checkpoint flips the routed-state tracker back to True via
        _routed_state_transition."""
        try:
            scale = float(getattr(self, "phase1_timeout_scale", 1.0) or 1.0)
            res = await asyncio.wait_for(
                self.call_tool("vivado_open_checkpoint", {
                    "dcp_path": str(Path(dcp_path).resolve())}),
                timeout=600.0 * max(1.0, scale))
            if _looks_like_tool_error(res):
                logger.warning(
                    f"[pre-sweep] pristine re-open FAILED: {str(res)[:160]}")
                return False
            return True
        except Exception as e:
            logger.warning(
                f"[pre-sweep] pristine re-open raised {type(e).__name__}: {e}")
            return False

    async def _presweep_execute_draw(
            self, op: str, timeout_s: float
    ) -> tuple[Optional[float], Optional[float], bool, Optional[str]]:
        """Execute and measure one pre-sweep routing draw.

        `unroute_ae` unrouts the design and reroutes with `AggressiveExplore`;
        because it is destructive, it is legal only for the first draw, when
        the pristine checkpoint can be restored. `bare` invokes `route_design`
        directly and is used for later draws because routing an identical
        netlist and placement is deterministic. Returns `(wns, whs, routed_ok,
        error)`. A missing WNS marks a failed draw; missing WHS is unmeasurable
        and is treated permissively by the caller. The routed verdict uses the
        tracker-first validity check, and `error` contains a short operation
        failure message. The caller must suppress automatic banking because
        this pre-sweep performs its own gated acceptance and measurement.
        """
        if op == "unroute_ae":
            cmd = ("route_design -unroute; "
                   "route_design -directive AggressiveExplore")
        else:
            cmd = "route_design"
        try:
            res = await asyncio.wait_for(
                self.call_tool("vivado_run_tcl",
                               {"command": cmd, "timeout": float(timeout_s)}),
                timeout=float(timeout_s) + 60.0)
        except Exception as e:
            return (None, None, False, f"op_raised:{type(e).__name__}")
        if _looks_like_tool_error(res):
            return (None, None, False, f"op_failed:{str(res)[:120]}")
        # Measure WNS the same way the auto-bank hook does (outer cap —
        # never rely on the server's 300 s default).
        try:
            wns = await asyncio.wait_for(
                self.get_wns_for_target_clock(self._call_vivado_tool),
                timeout=600.0)
        except Exception:
            wns = None
        if wns is None:
            return (None, None, False, "wns_unmeasurable")
        # Tracker-first routed gate (phantom-best guard; fails CLOSED).
        routed = await self._routed_ok_for_best()
        # Pass self.call_tool directly because _measure_hold supplies complete
        # tool names. The prefixed wrapper would construct an invalid name and
        # fail open.
        try:
            from optimizer.ils_polish import _measure_hold
            whs = await asyncio.wait_for(
                _measure_hold(self.call_tool, timeout_s=120.0),
                timeout=180.0)
        except Exception:
            whs = None
        return (wns, whs, routed, None)

    async def _run_fresh_presweep(self, input_dcp: Path) -> None:
        """Select the best fresh-state routing draw as the pipeline entry state.

        Must run after baseline WNS is measured on the pristine state and
        before all remaining phase-one feature capture. Routing preserves
        placement, so adopted timing features are refreshed from the selected
        draw; TNS and failing endpoints are reparsed after adoption. Only draws
        no worse than the baseline are adopted and banked. A failed or rejected
        draw restores the pristine checkpoint, while rejection after an earlier
        adoption reopens the banked best mirror. The handoff therefore uses the
        best accepted draw, falling back to the pristine state.
        """
        K = int(self._fresh_presweep_draws)
        if K <= 0:
            return  # OFF — zero tool calls, zero diff.
        if self.initial_wns is None:
            logger.info("[pre-sweep] skip: entry WNS unmeasured — cannot "
                        "gate never-worse.")
            return
        if self.initial_wns >= 0.0:
            logger.info("[pre-sweep] skip: timing already met at entry "
                        f"(wns={self.initial_wns:.3f}).")
            return
        # Candidate mode enables the 0.2× pre-sweep only for walls of at least
        # 4,500 s, leaving a full 3,600 s evaluation budget after its overhead.
        # Adopt-entry mode bypasses this cost gate.
        if not self._presweep_adopt_entry:
            if (not self.max_wall_seconds
                    or float(self.max_wall_seconds)
                    < FRESH_PRESWEEP_MIN_WALL_S):
                logger.info(
                    "[pre-sweep] skip (candidate mode): max_wall="
                    f"{self.max_wall_seconds} < "
                    f"{FRESH_PRESWEEP_MIN_WALL_S:.0f}s cost gate — the 0.2x "
                    "candidate-producer sweep would steal wall from the "
                    "full-wall pristine pipeline (needs a DEBUG-WALL / farm "
                    "run with headroom).")
                return
        budget_total, first_timeout = self._presweep_budget_shape()
        if not self.max_wall_seconds:
            logger.info(
                "[pre-sweep] no wall cap configured — fallback budget "
                f"{budget_total:.0f}s / first-draw timeout {first_timeout:.0f}s.")
        entry_wns = float(self.initial_wns)
        sweep_t0 = self._presweep_now()
        logger.info(
            f"[pre-sweep] START: K={K} draws, entry wns={entry_wns:.3f}, "
            f"budget={budget_total:.0f}s "
            f"({FRESH_PRESWEEP_BUDGET_FRAC:.0%} of wall), first-draw "
            f"timeout={first_timeout:.0f}s (first-draw-measures).")
        spent = 0.0
        max_observed: Optional[float] = None
        state_tag = "pristine"
        tried: set[tuple[str, str]] = set()
        adopted_any = False
        in_memory_is_best = False
        for k in range(1, K + 1):
            # Op selection + deterministic-duplicate stop: route_design
            # is deterministic given identical netlist+placement, so a
            # (state, op) pair already drawn cannot yield a new result.
            if state_tag == "pristine" and ("pristine", "unroute_ae") not in tried:
                op = "unroute_ae"
            else:
                op = "bare"
            key = (state_tag, op)
            if key in tried:
                logger.info(
                    f"[pre-sweep] draw={k}/{K} skipped: deterministic "
                    f"duplicate (op={op} already drawn from state="
                    f"{state_tag}); stopping sweep.")
                break
            allowed, timeout_k, reason = _presweep_draw_allowance(
                k, spent, max_observed, budget_total, first_timeout)
            if not allowed:
                logger.info(
                    f"[pre-sweep] draw={k}/{K} skipped: {reason}; "
                    f"stopping sweep (spent={spent:.0f}s of "
                    f"{budget_total:.0f}s).")
                break
            # Wall-floor sanity on top of the pre-sweep budget: never
            # start a draw the global deadline cannot absorb.
            if (self._budget_deadline is not None
                    and self._budget_remaining() < timeout_k + 120.0):
                logger.info(
                    f"[pre-sweep] draw={k}/{K} skipped: global wall floor "
                    f"(remaining {self._budget_remaining():.0f}s < "
                    f"draw timeout {timeout_k:.0f}s + 120s margin).")
                break
            pre_wns = (float(self.best_wns)
                       if (self.best_wns is not None
                           and self.best_wns != float("-inf"))
                       else entry_wns)
            t0 = self._presweep_now()
            prev_suppress = self._tail_ctrl_suppress_autobank
            self._tail_ctrl_suppress_autobank = True
            try:
                wns, whs, routed, err = await self._presweep_execute_draw(
                    op, timeout_k)
            finally:
                self._tail_ctrl_suppress_autobank = prev_suppress
            cost = max(0.0, self._presweep_now() - t0)
            spent += cost
            max_observed = max(max_observed or 0.0, cost)
            tried.add(key)
            # Verdict (banking discipline).
            reject_why = ""
            if err is not None or wns is None:
                verdict = "ERROR"
                reject_why = err or "wns_unmeasurable"
            elif not routed:
                verdict = "REJECTED"
                reject_why = "not_routed (tracker-first guard)"
            elif whs is not None and whs < 0.0:
                verdict = "REJECTED"
                reject_why = f"hold_dirty whs={whs:.3f}"
            elif wns > pre_wns:
                verdict = "ADOPTED"
                if whs is None:
                    # Fail-open like the auto-bank hook (a measurement
                    # gap must never strand a real gain) but LOUD.
                    self._presweep_hold_failopen_count += 1
                    logger.warning(
                        "[pre-sweep] HOLD UNMEASURABLE on adopted draw "
                        f"#{self._presweep_hold_failopen_count} — banking "
                        "FAIL-OPEN (whs unknown; validator gates "
                        "hold_passed).")
            else:
                verdict = "REJECTED"
                reject_why = "no strict improvement"
            if verdict == "ADOPTED":
                old_best = self.best_wns
                self.best_wns = wns
                self.last_improvement_iter = self.iteration
                self.last_improvement_time = time.time()
                self.regression_warning_sent = False
                self._pending_best_mirror = True
                self._pending_best_mirror_epoch = self._mutation_epoch
                await self._mirror_best_valid_now(eager=True)
                adopted_any = True
                in_memory_is_best = True
                self._presweep_adopted = True
                self._presweep_post_wns = float(wns)
                state_tag = f"best@{wns:.6f}"
                logger.info(
                    f"[pre-sweep] draw {k} BANKED via eager mirror "
                    f"(best {old_best if old_best is not None else 'None'} "
                    f"-> {wns:.3f}).")
            else:
                # Worse/failed draw: re-open the pristine input DCP
                # before the next draw / hand-off (never hand the
                # pipeline a worse-than-entry state).
                in_memory_is_best = False
                reopened = await self._presweep_reopen(input_dcp)
                if not reopened:
                    logger.warning(
                        f"[pre-sweep] draw={k}/{K}: pristine re-open "
                        "FAILED after a non-adopted draw — aborting sweep "
                        "(hand-off will retry the re-open).")
                    state_tag = "unknown"
                    self._log_presweep_draw(k, K, pre_wns, wns, whs,
                                            verdict, reject_why, cost,
                                            budget_total - spent, op)
                    break
                state_tag = "pristine"
            self._log_presweep_draw(k, K, pre_wns, wns, whs, verdict,
                                    reject_why, cost,
                                    budget_total - spent, op)
        # Candidate mode banks the best pre-sweep result separately, then runs the
        # pipeline from pristine input. Finalization selects the better result,
        # so the pre-sweep cannot force the pipeline into a different basin.
        if not self._presweep_adopt_entry:
            registered = False
            if (adopted_any and self._presweep_post_wns is not None
                    and self._best_valid_dcp is not None
                    and Path(self._best_valid_dcp).exists()):
                try:
                    # In-memory := the banked best draw, so register's
                    # verify=True re-runs the exact routed/hold/cell gates
                    # against the candidate state (the mirror on disk).
                    reopened = await self._presweep_reopen(
                        Path(self._best_valid_dcp))
                    if reopened:
                        store_dir = self.run_dir / "final_candidates"
                        store_dir.mkdir(parents=True, exist_ok=True)
                        store = store_dir / "presweep.dcp"
                        _atomic_copy(str(self._best_valid_dcp), str(store))
                        src_edif = Path(self._best_valid_dcp).with_suffix(".edf")
                        if src_edif.exists() and src_edif.stat().st_size > 0:
                            try:
                                _atomic_copy(str(src_edif),
                                             str(store.with_suffix(".edf")))
                            except Exception:
                                pass
                        registered = await self.register_final_candidate(
                            store, self._presweep_post_wns, "presweep",
                            verify=True)
                    else:
                        logger.warning(
                            "[pre-sweep] candidate mode: could not re-open "
                            "the banked best draw to verify — SKIPPING "
                            "registration (pipeline still runs pristine).")
                except Exception as e:
                    logger.warning(
                        "[pre-sweep] candidate registration raised "
                        f"{type(e).__name__} (non-fatal): {e}")
            # Reopen the pristine input and reset pipeline lineage accounting.
            # The banked pre-sweep result remains only in the candidate store.
            await self._presweep_reopen(input_dcp)
            self.best_wns = entry_wns
            self._best_valid_dcp = None
            self._best_valid_dcp_wns = None
            self._best_valid_edif = None
            self._best_valid_edif_wns = None
            self._best_valid_lineage = None
            self._pending_best_mirror = False
            self.last_improvement_iter = self.iteration
            self.last_improvement_time = time.time()
            self.regression_warning_sent = False
            self._presweep_adopted = False
            self._presweep_post_wns = None
            sweep_elapsed = max(0.0, self._presweep_now() - sweep_t0)
            if self._phase1_start_ts is not None:
                self._phase1_start_ts += sweep_elapsed
            logger.info(
                f"[pre-sweep] COMPLETE (candidate mode): adopted={adopted_any} "
                f"registered={registered} entry {entry_wns:.3f}; pipeline "
                f"enters PRISTINE, presweep draw insured as a FINAL candidate "
                f"({len(self._final_candidates)} enrolled). spent={spent:.0f}s "
                f"of {budget_total:.0f}s (phase-1 allowance shifted by "
                f"{sweep_elapsed:.0f}s).")
            return
        # ---- hand-off (never-worse) ----  [adopt-entry kill-switch only]
        if adopted_any and not in_memory_is_best:
            handed_best = False
            if (self._best_valid_dcp is not None
                    and Path(self._best_valid_dcp).exists()):
                handed_best = await self._presweep_reopen(
                    Path(self._best_valid_dcp))
            if handed_best:
                logger.info(
                    "[pre-sweep] hand-off: re-opened banked best draw "
                    f"({self._presweep_post_wns:.3f} ns) as the pipeline "
                    "entry state.")
                in_memory_is_best = True
            else:
                logger.warning(
                    "[pre-sweep] hand-off: could not re-open the banked "
                    "best mirror — pipeline enters at PRISTINE (the "
                    "on-disk mirror still protects the banked gain "
                    "through finalize).")
                # Feature extraction must describe the reopened pristine state, even
                # while best_wns mirrors the separately banked candidate.
                self._presweep_adopted = False
                self._presweep_post_wns = None
                if state_tag == "unknown":
                    await self._presweep_reopen(input_dcp)
        elif not adopted_any and state_tag == "unknown":
            # Last resort: one more attempt to leave a sane state.
            await self._presweep_reopen(input_dcp)
        # ---- feature refresh (adopted state only) ----
        if adopted_any and in_memory_is_best:
            try:
                scale = float(getattr(self, "phase1_timeout_scale", 1.0) or 1.0)
                rpt = await asyncio.wait_for(
                    self.call_tool("vivado_report_timing_summary", {}),
                    timeout=300.0 * max(1.0, scale) + 60.0)
                if not _looks_like_tool_error(rpt):
                    ti = parse_timing_summary_static(rpt)
                    self._presweep_post_tns = ti.get("tns")
                    self._presweep_post_failing_endpoints = ti.get(
                        "failing_endpoints")
                    logger.info(
                        "[pre-sweep] post-sweep feature refresh: tns="
                        f"{self._presweep_post_tns} failing_endpoints="
                        f"{self._presweep_post_failing_endpoints}.")
            except Exception as e:
                logger.warning(
                    f"[pre-sweep] post-sweep timing re-parse failed "
                    f"(router falls back to entry TNS/endpoints; WNS "
                    f"feature is already post-sweep): {e}")
        # Exclude pre-sweep time from Phase 1 because it has a separate 0.2× wall
        # allowance; otherwise later optional analyses may be skipped prematurely.
        sweep_elapsed = max(0.0, self._presweep_now() - sweep_t0)
        if self._phase1_start_ts is not None:
            self._phase1_start_ts += sweep_elapsed
        logger.info(
            f"[pre-sweep] COMPLETE: adopted={adopted_any} entry "
            f"{entry_wns:.3f} -> best "
            f"{self.best_wns if self.best_wns != float('-inf') else 'None'} "
            f"spent={spent:.0f}s of {budget_total:.0f}s "
            f"(phase-1 allowance shifted by {sweep_elapsed:.0f}s).")

    def _log_presweep_draw(self, k: int, K: int, pre_wns: float,
                           post_wns: Optional[float], whs: Optional[float],
                           verdict: str, reject_why: str, cost: float,
                           remaining: float, op: str) -> None:
        """Structured per-draw log line + decision-trace
        record + in-memory record for post-mortems."""
        post_str = f"{post_wns:.3f}" if post_wns is not None else "None"
        whs_str = f"{whs:.3f}" if whs is not None else "None"
        line = (f"[pre-sweep] draw={k}/{K} wns {pre_wns:.3f} -> {post_str} "
                f"whs={whs_str} verdict={verdict} cost={cost:.0f}s "
                f"remaining={remaining:.0f}s")
        if reject_why:
            line += f" ({reject_why})"
        logger.info(line)
        rec = {
            "draw": k, "draws_total": K, "op": op,
            "wns_before": pre_wns, "wns_after": post_wns, "whs": whs,
            "verdict": verdict, "reject_why": reject_why or None,
            "cost_s": round(cost, 1), "budget_remaining_s": round(remaining, 1),
        }
        self._presweep_draw_records.append(rec)
        try:
            self._emit_decision({
                "decision_source": "presweep",
                "phase": "phase_1",
                "action_label": f"presweep_draw_{verdict.lower()}",
                "tool_name": "vivado_run_tcl",
                "wns_before": pre_wns,
                "wns_after": post_wns,
                "runtime_s": cost,
                "notes": (f"op={op} draw={k}/{K} whs={whs_str} "
                          f"remaining={remaining:.0f}s"
                          + (f" reject_why={reject_why}" if reject_why else "")),
            })
        except Exception:  # pragma: no cover — trace must never break a run
            pass

    def _build_recipe_router_block(self) -> list[str]:
        """Return prompt-block lines from the feature-based recipe router.

        Constructs a PhaseOneFeatures snapshot from already-collected
        Phase-1 state (self.initial_wns, self.clock_period,
        self.initial_failing_endpoints, self.critical_path_spread_info,
        and the live wall-budget deadline) and asks the router for a
        recipe plan. Returns formatted lines, or [] when the router is
        unavailable or only emits the low-confidence FALLBACK plan
        (which adds no signal beyond the pathology classifier).
        """
        if _decide_recipe_path is None or _RouterFeatures is None:
            return []

        spread: Optional[float] = None
        if (isinstance(self.critical_path_spread_info, dict)
                and self.critical_path_spread_info.get("avg_distance") is not None):
            spread = float(self.critical_path_spread_info["avg_distance"])

        remaining_budget: Optional[float] = None
        if self._budget_deadline is not None:
            remaining_budget = max(0.0, self._budget_deadline - time.time())

        try:
            # Pre-sweep feature view: when a step-0 route re-roll
            # was adopted, route on the POST-SWEEP timing state — the
            # pipeline starts there, not at the pristine entry.
            _u = self.utilization if isinstance(self.utilization, dict) else {}
            features = _RouterFeatures(
                # Expose measured size and resource utilization to the router;
                # static estimates are unreliable across widely varying scales.
                cell_count=self._input_cell_count,
                lut_util_pct=_u.get("lut_pct"),
                bram_util_pct=_u.get("bram_pct"),
                uram_util_pct=_u.get("uram_pct"),
                dsp_util_pct=_u.get("dsp_pct"),
                memory_dominated=self.memory_dominated,
                route_pct=self.route_pct,
                wns_ns=self._phase1_wns_for_features(),
                clock_period_ns=self.clock_period,
                failing_endpoint_count=self._phase1_failing_endpoints_for_features(),
                critical_path_avg_spread_tiles=spread,
                remaining_wall_budget_s=remaining_budget,
                class_g_attempted=False,  # iter-1 — no Class G yet
            )
            plan = _decide_recipe_path(features)
        except Exception as e:
            logger.warning(f"recipe_router failed: {e}")
            return []

        # Suppress FALLBACK with no blocks — it adds no signal beyond
        # the pathology classifier. Emit on a positive rule match, or
        # when block rules (R5/R6) produced applicability info.
        if plan.rule_id == "FALLBACK" and not plan.blocks:
            return []

        # Store the plan on self so downstream consumers (RAG, audit logs,
        # later-iter prompts) can reference which rule fired.
        self.recipe_router_plan = plan

        text = plan.format_for_prompt()
        # format_for_prompt returns a multi-line string; split into the
        # list[str] shape expected by the summary builder. Drop a leading
        # blank line so it doesn't double-space with the previous block.
        lines = text.split("\n")
        if lines and lines[0] == "":
            lines = lines[1:]
        return lines

    async def _maybe_run_recipe_first(self) -> None:
        """Run the forced deep-extreme recipe before the LLM loop when explicitly
        enabled.

        The recipe must precede the loop because it can consume most of the
        available budget and cannot reliably serve as a tail operation. At
        startup, the size model derives its cost basis from the primitive cell
        count because no heavy-step timing anchor exists yet. Baseline
        finalization protects against regressions; the tradeoff is less
        remaining time for the LLM loop. This path is a strict no-op unless
        both `deep_replace_enabled` and `deep_replace_first_enabled` are set.
        """
        cfg = self._ils_polish_cfg
        _en = bool(getattr(cfg, "deep_replace_enabled", False))
        _first = bool(getattr(cfg, "deep_replace_first_enabled", False))
        # REACHED before the enable check, for the same reason as the tail
        # stage: "never called" and "called but disabled" must be
        # distinguishable in the log.
        logger.info(f"deep-replace[first]: hook REACHED "
                    f"(enabled={_en} first={_first})")
        if not (_en and _first):
            return
        try:
            from optimizer.deep_replace_sibling import predict_place_route_s
            basis, why = predict_place_route_s(
                primitive_cells=getattr(self, "_input_cell_count", None))
            if basis <= 0:
                logger.info(f"deep-replace[first]: skipped ({why})")
                return
            await self._deep_replace_sibling_after_polish(
                time.time() + self._budget_remaining(),
                self._wns_tcl_for_stages(),
                stage_label="first",
                cost_basis_override=basis,
                basis_note=why,
            )
        except Exception as e:
            # Must never sink the run: the LLM loop still has the whole wall.
            logger.warning(f"deep-replace[first] raised "
                           f"{type(e).__name__} (non-fatal): {e!r}")

    async def _maybe_run_recipe_pass(self, input_dcp: Path) -> None:
        """FPL26_RECIPE_PASS (DEFAULT OFF): frozen-Tcl candidate pass.

        Runs at the post-Phase-1/pre-LLM slot and clones the presweep
        INSURED-COMPARE candidate-mode contract (see _run_fresh_presweep's
        candidate branch): run the frozen recipe FROM THE PRISTINE INPUT,
        copy the result to the private final_candidates store, enroll via
        register_final_candidate (routed/hold/cell gates), restart Vivado
        (session-pollution mitigation — open_checkpoint does NOT clear it),
        reopen the pristine input and hand the LLM loop an untouched run.
        The result competes ONLY at the finalize MUX (never-worse enforced
        there); it never touches best_valid.  A failed/timed-out recipe
        costs wall only — nothing else.

        Stable log keys (firing audit): "RECIPE-PASS: ARMED band=...",
        "RECIPE-PASS: fired recipe=... wns_in=...",
        "RECIPE-PASS: skipped reason=...".
        """
        if not recipe_pass_enabled():
            # Reached-but-disabled must be distinguishable from never-called
            # (feedback_firing_check_must_key_on_treatment).
            logger.info("RECIPE-PASS: skipped reason=disabled "
                        "(FPL26_RECIPE_PASS off or FPL26_NO_RECIPE_PASS set)")
            # The first-deep variant requires the master recipe flag.
            # Log an explicit skip when the variant is enabled without its master.
            if (os.environ.get("FPL26_RECIPE_FIRST_DEEP", "")
                    .strip().lower() in ("1", "true", "on", "yes")):
                logger.info(
                    "RECIPE-FIRST-DEEP: skipped reason=flag_off "
                    "(FPL26_RECIPE_FIRST_DEEP is set but the master "
                    "FPL26_RECIPE_PASS is off or FPL26_NO_RECIPE_PASS is "
                    "set — BOTH flags must be truthy; the kill switch "
                    "kills both)")
            return
        fired = False
        t_pass0 = time.time()
        # Tag recipe tool calls so prompt consumers exclude them from the model-visible
        # tried history. Behavioral consumers still retain the entries for cost
        # accounting because the recorded tool costs remain valid.
        _tc0 = len(self.tool_call_details)
        try:
            try:
                fired = await self._run_recipe_pass_inner(input_dcp)
            except Exception as e:
                logger.warning(f"RECIPE-PASS: skipped reason=exception "
                               f"{type(e).__name__}: {e!r} (non-fatal)")
            if fired:
                # Once the recipe touches the session, always start a fresh
                # tool session, reopen pristine input, and update accounting on
                # success or failure.
                await self._recipe_pass_reset_session(input_dcp)
                # Phase-1 allowance shift (presweep contract 5993-5994
                # parity): the pass's elapsed wall must not squeeze
                # allowance-keyed gates.
                if self._phase1_start_ts is not None:
                    self._phase1_start_ts += max(0.0,
                                                 time.time() - t_pass0)
                # Apply the reduced tail only after the size-based deep gate fires.
                # This emits logs but no tool calls and remains inside prompt tagging.
                if self._recipe_first_deep_fired:
                    self._recipe_first_deep_apply_reduced_tail()
        finally:
            for _d in self.tool_call_details[_tc0:]:
                try:
                    _d["recipe_pass_internal"] = True
                except Exception:
                    pass

    async def _run_recipe_pass_inner(self, input_dcp: Path) -> bool:
        """Key/wall gates + recipe execution + candidate registration.

        Returns True iff the recipe MUTATED the Vivado session (caller must
        then reset to pristine).  Never banks into best_valid: the
        auto-bank hook is suppressed for the whole pass (same bracket as the
        presweep draws) and banking happens only via register_final_candidate.
        """
        wns_in = self.initial_wns  # measured once at open; NOT re-measured
        band = recipe_pass_band(wns_in)
        if band is None:
            logger.info("RECIPE-PASS: skipped reason=wns_unmeasured "
                        "(initial_wns is None; no honest band key)")
            return False
        # Drift audit (log-only): live banked wns beside the band key,
        # so a deep-replace[first] adopt that moved the design across a band
        # edge is visible in every log without changing any decision.
        logger.info(f"RECIPE-PASS: ARMED band={band} wns_in={wns_in:.3f} "
                    f"live_best_wns={self.best_wns}")
        # Skip the shallow pass for near-met inputs (|WNS| < 0.50 ns) with at most
        # 1,000 failing endpoints when the floor succeeds; a pristine reset can
        # degrade this sparse-failure regime. A floor failure runs the pass as fallback.
        # Returning False declares the session unmodified, so the caller skips reset.
        if (band == "shallow" and fir_subband_floor_enabled()
                and subband_carveout_match(
                    wns_in, self._phase1_failing_endpoints_for_features())
                and not getattr(self, "_subband_floor_failed", False)):
            logger.info("RECIPE-PASS: skipped reason=fir_subband_carveout "
                        "(fir-like |wns_in| < "
                        f"{SUBBAND_CARVEOUT_WNS_MAG_MAX_NS}, failing ≤ "
                        f"{SUBBAND_FLOOR_MAX_FAILING_ENDPOINTS}, floor ok)")
            return False
        if not (input_dcp and Path(input_dcp).exists()):
            logger.info("RECIPE-PASS: skipped reason=no_pristine_input "
                        "(input DCP missing on disk — cannot honor the "
                        "pristine-reopen contract)")
            return False
        steps = {"shallow": RECIPE_PASS_SHALLOW_TCL,
                 "deep": RECIPE_PASS_DEEP_TCL,
                 "mid": RECIPE_PASS_MID_TCL}[band]
        expected_s = {"shallow": RECIPE_PASS_SHALLOW_EXPECTED_S,
                      "deep": RECIPE_PASS_DEEP_EXPECTED_S,
                      "mid": RECIPE_PASS_MID_EXPECTED_S}[band]
        remaining = self._budget_remaining()
        # Reserve the full pass timeout cap (1.5× expected cost) plus fixed time for
        # measurement, storage, registration, and reset outside the pass deadline.
        pass_budget_cap = RECIPE_PASS_TIMEOUT_FACTOR * expected_s
        overhead = RECIPE_PASS_OVERHEAD_RESERVE_S
        _overhead_terms = (
            f"[measure {RECIPE_PASS_MEASURE_RESERVE_S:.0f} + store "
            f"{RECIPE_PASS_STORE_RESERVE_S:.0f} + register "
            f"{RECIPE_PASS_REGISTER_RESERVE_S:.0f} + reset "
            f"{RECIPE_PASS_RESET_RESERVE_S:.0f}]")
        # FPL26_RECIPE_FIRST_DEEP fire context: (anchor, predicted, need,
        # remaining) when the size-anchored deep gate armed this pass;
        # None on every other path (including ALL flag-off runs).
        _rfd_ctx = None

        # Render infinite wall values as "unbounded" and finite values as whole
        # seconds. Use the same formatter for gate and execution logs to keep
        # their schema stable.
        def _rfd_s(v) -> str:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return "unbounded"
            return "unbounded" if fv == float("inf") else f"{fv:.0f}s"

        if band == "deep" and recipe_first_deep_enabled():
            # The opt-in gate requires 1.2 × (2.5 × size anchor) + 360 s reset
            # + the LLM floor, using predict_place_route_s as the calibrated
            # anchor. Measurement, storage, and registration reserves are
            # already included in that calibration; the LLM floor replaces
            # rather than adds to the tail reserve.
            try:
                llm_floor = float(getattr(
                    self, "phys_opt_preempt_budget_floor_s", 1100.0))
            except (TypeError, ValueError):
                llm_floor = 1100.0
            # For finite walls, scale the LLM floor with remaining time to preserve a
            # planning tail on large budgets. Unwalled runs retain the fixed floor.
            if remaining != float("inf"):
                llm_floor = max(
                    llm_floor,
                    RECIPE_FIRST_DEEP_LLM_FLOOR_FRAC * remaining)
            try:
                from optimizer.deep_replace_sibling import predict_place_route_s
                anchor, anchor_why = predict_place_route_s(
                    primitive_cells=getattr(
                        self, "_input_cell_count", None))
            except Exception as e:
                anchor, anchor_why = 0.0, (
                    f"size model raised {type(e).__name__}")
            anchor = float(anchor)
            # Clamp positive anchors because designs below ~23k cells can route much
            # slower than the cell-count model predicts.
            # Nonpositive anchors remain unmeasurable, so the gate fails closed.
            if 0.0 < anchor < RECIPE_FIRST_DEEP_ANCHOR_FLOOR_S:
                anchor_why += (
                    f" [clamped {anchor:.0f}s->"
                    f"{RECIPE_FIRST_DEEP_ANCHOR_FLOOR_S:.0f}s anchor "
                    f"floor — small-cell honesty guard]")
                anchor = RECIPE_FIRST_DEEP_ANCHOR_FLOOR_S
            predicted = RECIPE_FIRST_DEEP_COST_RATIO * anchor
            need = (RECIPE_FIRST_DEEP_MARGIN * predicted
                    + RECIPE_PASS_RESET_RESERVE_S + llm_floor)
            # Require the margined runtime prediction to fit the pass's hard
            # deadline; available wall time alone does not guarantee completion.
            # The 1,350 s limit is the timeout factor times the expected duration.
            # Predictions beyond that limit fail closed before consuming wall time.
            _deadline_funds_s = (RECIPE_PASS_TIMEOUT_FACTOR
                                 * RECIPE_PASS_DEEP_EXPECTED_S)  # 1350
            _deadline_unfundable = (
                RECIPE_FIRST_DEEP_MARGIN * predicted > _deadline_funds_s)
            # Emit one additive telemetry line per gate evaluation without changing
            # the existing armed/skipped counters.
            # Reported slack is remaining time minus required time, in seconds.
            _gate_decision = ("FIRE" if (anchor > 0.0
                                         and not _deadline_unfundable
                                         and remaining >= need)
                              else "NOFIRE")
            logger.info(
                f"RECIPE-FIRST-DEEP: GATE decision={_gate_decision} "
                f"anchor={anchor:.0f}s predicted={predicted:.0f}s "
                f"need={need:.0f}s remaining={_rfd_s(remaining)} "
                f"slack={_rfd_s(remaining - need)}")
            if anchor <= 0.0:
                # Unmeasured size must not arm a full place+route recipe
                # (same fail-closed rule predict_place_route_s documents).
                logger.info(
                    f"RECIPE-FIRST-DEEP: skipped reason=cost_model "
                    f"(anchor=0s predicted=0s need={need:.0f}s "
                    f"remaining={_rfd_s(remaining)} — size anchor "
                    f"unmeasurable ({anchor_why}); fail closed)")
                return False
            if _deadline_unfundable:
                logger.info(
                    f"RECIPE-FIRST-DEEP: skipped reason=cost_model "
                    f"(anchor={anchor:.0f}s predicted={predicted:.0f}s "
                    f"need={need:.0f}s remaining={_rfd_s(remaining)}; "
                    f"margined predicted "
                    f"{RECIPE_FIRST_DEEP_MARGIN * predicted:.0f}s > pass "
                    f"deadline {_deadline_funds_s:.0f}s (1.5x900 band "
                    f"constant) — predicted exceeds the pass deadline the "
                    f"band constant funds — deadline-unfundable, fail "
                    f"closed)")
                return False
            if remaining < need:
                logger.info(
                    f"RECIPE-FIRST-DEEP: skipped reason=cost_model "
                    f"(anchor={anchor:.0f}s predicted={predicted:.0f}s "
                    f"need={need:.0f}s remaining={_rfd_s(remaining)}; "
                    f"need = "
                    f"{RECIPE_FIRST_DEEP_MARGIN}x predicted + reset "
                    f"{RECIPE_PASS_RESET_RESERVE_S:.0f}s + LLM floor "
                    f"{llm_floor:.0f}s — fail closed; the post-loop slot "
                    f"may still arm)")
                return False
            _rfd_ctx = (anchor, predicted, need, remaining)
            # Gate-time LLM floor for the reduced-tail step (hardening
            # (2)): the loop must keep what THIS gate promised, not the
            # smaller base floor.
            self._recipe_first_deep_gate_llm_floor_s = llm_floor
            logger.info(
                f"RECIPE-FIRST-DEEP: ARMED band=deep wns_in={wns_in:.3f} "
                f"(anchor={anchor:.0f}s predicted={predicted:.0f}s "
                f"need={need:.0f}s remaining={_rfd_s(remaining)}; "
                f"{anchor_why})")
        elif band == "deep":
            # The wall gate reserves the pass budget, overhead, and downstream
            # deep-timing tail. It fails closed when the full pipeline cannot fit,
            # preventing this pass from starving later recovery work.
            tail_reserve = self._deep_wns_tail_reserve_effective_s()
            need = pass_budget_cap + overhead + tail_reserve
            if remaining < need:
                logger.info(
                    f"RECIPE-PASS: skipped reason=wall band=deep "
                    f"(remaining {remaining:.0f}s < pass budget "
                    f"{pass_budget_cap:.0f}s (1.5x{expected_s:.0f}) + "
                    f"overheads {overhead:.0f}s {_overhead_terms} + deep-WNS "
                    f"tail reserve {tail_reserve:.0f}s = {need:.0f}s. Deep "
                    f"recipe requires remaining >= {need:.0f}s; the ship "
                    f"wall (MAX_WALL 3500, deadline-300) provides at most "
                    f"3200s — UNARMABLE AS BUILT, fail-closed by design, "
                    f"slot redesign pending (REVIEW_V30 MAJOR-1). "
                    f"(postloop slot may still arm)")
                return False
        elif band == "mid":
            # Mid-band admission includes the estimated pass cost, overhead, and
            # minimum useful LLM-loop budget.
            # The gate fails closed when the complete requirement exceeds the wall.
            try:
                llm_floor = float(getattr(
                    self, "phys_opt_preempt_budget_floor_s", 1100.0))
            except (TypeError, ValueError):
                llm_floor = 1100.0
            need = expected_s + overhead + llm_floor
            if remaining < need:
                logger.info(
                    f"RECIPE-PASS: skipped reason=wall band=mid "
                    f"(remaining {remaining:.0f}s < measured recipe cost "
                    f"{expected_s:.0f}s (cross-box probe, boxes 4/5) + "
                    f"overheads {overhead:.0f}s {_overhead_terms} + "
                    f"LLM-loop floor {llm_floor:.0f}s = {need:.0f}s. Mid "
                    f"recipe requires remaining >= {need:.0f}s; the ship "
                    f"wall (MAX_WALL 3500, deadline-300) provides at most "
                    f"3200s — UNARMABLE AS BUILT, fail-closed by design, "
                    f"slot redesign pending (post-loop stranded-wall slot "
                    f"is the intended home). (postloop slot may still arm)")
                return False
        else:
            # Shallow admission reserves enough time for both this pass and a useful
            # downstream LLM step, whose minimum budget is set by
            # phys_opt_preempt_budget_floor_s.
            try:
                llm_floor = float(getattr(
                    self, "phys_opt_preempt_budget_floor_s", 1100.0))
            except (TypeError, ValueError):
                llm_floor = 1100.0
            need = pass_budget_cap + overhead + llm_floor
            if remaining < need:
                logger.info(
                    f"RECIPE-PASS: skipped reason=wall band=shallow "
                    f"(remaining {remaining:.0f}s < pass budget "
                    f"{pass_budget_cap:.0f}s (1.5x{expected_s:.0f}) + "
                    f"overheads {overhead:.0f}s {_overhead_terms} + "
                    f"LLM-loop floor {llm_floor:.0f}s = {need:.0f}s — "
                    f"fail closed)")
                return False
        # Hard per-pass deadline: 1.5x the verified cost, additionally capped
        # so the finalize reserve is never spendable by a runaway step.
        pass_budget = RECIPE_PASS_TIMEOUT_FACTOR * expected_s
        if remaining != float("inf"):
            pass_budget = min(pass_budget,
                              remaining - RECIPE_PASS_FINALIZE_RESERVE_S)
        if pass_budget < 60.0:
            logger.info(f"RECIPE-PASS: skipped reason=wall "
                        f"(pass budget {pass_budget:.0f}s < 60s floor)")
            return False
        # Double-fire guard for the post-loop slot: mark BEFORE the
        # first session mutation so a pre-LLM pass that fired but raised
        # mid-flight still blocks a second run at the exit tail.
        self._recipe_pass_preloop_fired = True
        if _rfd_ctx is not None:
            # Set this marker after the preloop fired marker and before the first
            # session mutation. This ordering keeps tail selection and the
            # postloop already-fired guard consistent.
            self._recipe_first_deep_fired = True
            _a, _p, _n, _r = _rfd_ctx
            logger.info(
                f"RECIPE-FIRST-DEEP: fired recipe=deep "
                f"wns_in={wns_in:.3f} (anchor={_a:.0f}s "
                f"predicted={_p:.0f}s need={_n:.0f}s "
                f"remaining={_rfd_s(_r)}; "
                f"tail reserve reduced after the pass iff a candidate "
                f"banks)")
        logger.info(f"RECIPE-PASS: fired recipe={band} wns_in={wns_in:.3f} "
                    f"(pass_budget={pass_budget:.0f}s, "
                    f"steps={len(steps)})")
        t0 = time.time()
        pass_deadline = t0 + pass_budget
        # Recipes are verified relative to the input checkpoint, so reopen it before
        # applying any recipe to avoid inheriting an in-memory candidate state.
        # From this point, the caller treats the session as mutated and resets it.
        prev_suppress = self._tail_ctrl_suppress_autobank
        self._tail_ctrl_suppress_autobank = True
        try:
            if not await self._presweep_reopen(input_dcp):
                logger.warning("RECIPE-PASS: failed step=pristine_preopen "
                               "(recipe not run; wall lost, nothing banked)")
                return True
            for i, step in enumerate(steps, 1):
                # Check for runtime overrun only between steps, using the
                # existing deadline fences rather than asynchronous
                # cancellation. The cap is 0.75 of the fire-time need; with the
                # 1.2 admission margin, this provides 1.6x overrun protection.
                # Banking occurs only after all steps, so an abort preserves
                # the full tail.
                if _rfd_ctx is not None:
                    _elapsed = time.time() - t0
                    # Bound runtime by both 0.75 of the fire-time need and the wall
                    # remaining after reserving the LLM floor and reset budget.
                    # This preserves the budget promised to the rest of the run.
                    # Without a wall deadline, the fractional cap is the active bound.
                    _gate_floor_s = float(
                        self._recipe_first_deep_gate_llm_floor_s or 0.0)
                    _abort_cap = min(
                        RECIPE_FIRST_DEEP_DYNAMIC_ABORT_FRAC
                        * _rfd_ctx[2],
                        _rfd_ctx[3] - _gate_floor_s
                        - RECIPE_PASS_RESET_RESERVE_S)
                    if _elapsed > _abort_cap:
                        logger.warning(
                            f"RECIPE-FIRST-DEEP: ABORTED "
                            f"reason=dynamic_overrun "
                            f"elapsed={_elapsed:.0f}s "
                            f"cap={_abort_cap:.0f}s (reverted pristine; "
                            f"tail reserve UNTOUCHED — a miss keeps the "
                            f"full tail)")
                        logger.info(
                            f"RECIPE-FIRST-DEEP: RESULT "
                            f"recipe_wall={_elapsed:.0f}s wns_out=NA "
                            f"registered=F abort=dynamic_overrun")
                        # Session is mutated -> True: the caller runs the
                        # existing failure path (_recipe_pass_reset_
                        # session: restart + pristine reopen).
                        return True
                step_left = pass_deadline - time.time()
                if step_left < 30.0:
                    logger.warning(
                        f"RECIPE-PASS: failed step={i}/{len(steps)} "
                        f"reason=pass_timeout ({pass_budget:.0f}s budget "
                        f"exhausted before '{step}'; aborting — wall lost, "
                        f"nothing banked)")
                    if _rfd_ctx is not None:
                        # Record per-step fence timeouts in result telemetry so they
                        # remain distinguishable from dynamic-overrun aborts.
                        logger.info(
                            f"RECIPE-FIRST-DEEP: RESULT "
                            f"recipe_wall={time.time() - t0:.0f}s "
                            f"wns_out=NA registered=F "
                            f"abort=pass_timeout")
                    return True
                if step == RECIPE_PASS_SESSION_ROUNDTRIP:
                    ok = await self._recipe_pass_roundtrip(step_left)
                    if not ok:
                        logger.warning(
                            f"RECIPE-PASS: failed step={i}/{len(steps)} "
                            f"reason=session_roundtrip (aborting)")
                        return True
                    continue
                try:
                    res = await asyncio.wait_for(
                        self.call_tool("vivado_run_tcl",
                                       {"command": step,
                                        "timeout": float(step_left)}),
                        timeout=float(step_left) + 60.0)
                except Exception as e:
                    logger.warning(
                        f"RECIPE-PASS: failed step={i}/{len(steps)} "
                        f"reason=step_raised:{type(e).__name__} "
                        f"cmd='{step}' (aborting)")
                    return True
                if _looks_like_tool_error(res):
                    logger.warning(
                        f"RECIPE-PASS: failed step={i}/{len(steps)} "
                        f"reason=step_error cmd='{step}' "
                        f"detail={str(res)[:120]} (aborting)")
                    return True
            # ---- measure + private store + register (contract 1-3) ----
            try:
                wns_out = await asyncio.wait_for(
                    self.get_wns_for_target_clock(self._call_vivado_tool),
                    timeout=600.0)
            except Exception:
                wns_out = None
            if wns_out is None:
                logger.warning("RECIPE-PASS: failed step=measure "
                               "reason=wns_unmeasurable (nothing banked)")
                return True
            store_dir = self.run_dir / "final_candidates"
            store_dir.mkdir(parents=True, exist_ok=True)
            store = store_dir / f"recipe_pass_{band}.dcp"
            try:
                res = await asyncio.wait_for(
                    self.call_tool("vivado_write_checkpoint", {
                        "dcp_path": str(store.resolve()),
                        "force": True}),
                    timeout=900.0)
            except Exception as e:
                res = f'{{"error": "write_checkpoint raised {type(e).__name__}"}}'
            if _looks_like_tool_error(res) or not (
                    store.exists() and store.stat().st_size > 0):
                logger.warning("RECIPE-PASS: failed step=store "
                               f"reason=write_checkpoint detail="
                               f"{str(res)[:120]} (nothing banked)")
                return True
            # Registration rechecks the routed, hold, and cell gates against the
            # candidate currently represented both in memory and on disk.
            # A registration exception still returns a mutated-session result so the
            # caller resets before continuing the LLM loop.
            # The exception path precedes the banked marker, so failed registration
            # cannot enable tail reduction.
            try:
                registered = await self.register_final_candidate(
                    store, float(wns_out), f"recipe_pass_{band}", verify=True)
            except Exception as e:
                logger.warning(
                    f"RECIPE-PASS: register FAILED "
                    f"({type(e).__name__}: {e!r}); candidate dropped; "
                    f"session reset proceeds (pristine contract held)")
                return True
            if _rfd_ctx is not None and registered:
                # BANKED, not just fired: the reduced-tail step (caller)
                # keys on this — a fire that died pre-registration keeps
                # the full deterministic tail (fence-failure-mode guard).
                self._recipe_first_deep_banked = True
            if _rfd_ctx is not None:
                # GREPPABLE RESULT TELEMETRY (hardening (4)), ADDITIVE —
                # the RECIPE-PASS result line below is unchanged.
                logger.info(
                    f"RECIPE-FIRST-DEEP: RESULT "
                    f"recipe_wall={time.time() - t0:.0f}s "
                    f"wns_out={float(wns_out):.3f} "
                    f"registered={'T' if registered else 'F'} abort=NONE")
            logger.info(
                f"RECIPE-PASS: result recipe={band} wns_in={wns_in:.3f} "
                f"wns_out={float(wns_out):.3f} registered={registered} "
                f"dt={time.time() - t0:.0f}s (candidate competes ONLY at "
                f"the finalize MUX; pipeline enters PRISTINE)")
            # Run the optional second shallow candidate after the primary candidate
            # while auto-banking remains suppressed.
            # It is reached only when the primary path completes; its exceptions do
            # not invalidate an already registered primary candidate.
            # The caller's single reset covers both candidates.
            if band == "shallow":
                try:
                    await self._maybe_run_eto_retime_candidate(
                        input_dcp, wns_in)
                except Exception as e:
                    logger.warning(
                        f"ETO-RETIME: skipped reason=exception "
                        f"{type(e).__name__}: {e!r} (non-fatal; primary "
                        f"candidate already registered)")
                # At most one optional retime candidate runs in this pass.
                # Enabling this candidate makes the earlier retime candidate
                # defer; otherwise this call is a no-op and the earlier
                # candidate proceeds. Exceptions preserve earlier
                # registrations, and one reset covers all candidates.
                try:
                    await self._maybe_run_ownfront_retime_candidate(
                        input_dcp, wns_in)
                except Exception as e:
                    logger.warning(
                        f"OWNFRONT-RETIME: skipped reason=exception "
                        f"{type(e).__name__}: {e!r} (non-fatal; earlier "
                        f"candidates already registered)")
                # In the [0.60, 0.90) band, this shallow determinizer replaces
                # the earlier retime candidate rather than stacking with it.
                # Outside that band it only adds a deterministic floor after
                # retiming. Exceptions preserve earlier registrations, and the
                # pass uses one reset.
                try:
                    await self._maybe_run_shallow_determinizer_candidate(
                        input_dcp, wns_in)
                except Exception as e:
                    logger.warning(
                        f"SHALLOW-DET: skipped reason=exception "
                        f"{type(e).__name__}: {e!r} (non-fatal; earlier "
                        f"candidates already registered)")
            return True
        finally:
            self._tail_ctrl_suppress_autobank = prev_suppress

    async def _recipe_pass_roundtrip(self, step_left: float,
                                     tag: str = "RECIPE-PASS") -> bool:
        """Deep-recipe session round-trip: write checkpoint -> restart
        Vivado -> reopen the checkpoint.  The restart is REQUIRED (the
        verified recipe's key ingredient is fresh session state; reusing the
        exact vivado_restart_vivado call + error-envelope check the
        candidate contract documents — the tool name trap and the error-
        STRING return are both real).  ``tag`` only prefixes the log lines
        (default keeps the pre-LLM slot's lines byte-identical; the
        post-loop slot passes "RECIPE-PASS[postloop]")."""
        mid = self.run_dir / "recipe_pass_midpoint.dcp"
        try:
            res = await asyncio.wait_for(
                self.call_tool("vivado_write_checkpoint", {
                    "dcp_path": str(mid.resolve()), "force": True}),
                timeout=max(60.0, step_left))
        except Exception as e:
            logger.warning(f"{tag}: roundtrip write raised "
                           f"{type(e).__name__}: {e!r}")
            return False
        if _looks_like_tool_error(res) or not (
                mid.exists() and mid.stat().st_size > 0):
            logger.warning(f"{tag}: roundtrip write failed "
                           f"({str(res)[:120]})")
            return False
        try:
            _r = await self.call_tool("vivado_restart_vivado", {})
            if isinstance(_r, str) and '"error"' in _r:
                logger.warning(f"{tag}: roundtrip restart returned "
                               f"error ({_r[:140]}) — aborting (the recipe "
                               f"is conditional on the fresh session)")
                return False
        except Exception as e:
            logger.warning(f"{tag}: roundtrip restart raised {e!r}")
            return False
        if not await self._presweep_reopen(mid):
            logger.warning(f"{tag}: roundtrip reopen failed")
            return False
        logger.info(f"{tag}: session round-trip complete "
                    f"(checkpoint -> fresh Vivado -> reopen)")
        return True

    async def _recipe_pass_reset_session(self, input_dcp: Path) -> None:
        """Contract steps 4-5 (presweep candidate mode, copied): restart
        Vivado BEFORE the LLM loop (M1: recipe place/route leaves process
        state that degrades every later place_design and open_checkpoint
        does NOT clear it), reopen the PRISTINE input, and do the lineage/
        allowance handling so downstream accounting starts clean."""
        try:
            _r = await self.call_tool("vivado_restart_vivado", {})
            if isinstance(_r, str) and '"error"' in _r:
                logger.warning(f"RECIPE-PASS: post-pass restart returned "
                               f"error ({_r[:140]}); staying in current "
                               f"session (pristine reopen still follows).")
            else:
                logger.info("RECIPE-PASS: restarted Vivado fresh before the "
                            "LLM loop (clears recipe session pollution).")
        except Exception as e:
            logger.warning(f"RECIPE-PASS: post-pass restart failed ({e!r}); "
                           "current session.")
        reopened = await self._presweep_reopen(input_dcp)
        if not reopened:
            logger.warning(
                "RECIPE-PASS: pristine re-open FAILED after the pass — the "
                "LLM loop will retry tool calls against whatever session "
                "state exists (finalize baseline-copy safety still holds).")
        # Auto-banking is suppressed, and candidates enter through registration, so
        # this pass must not modify the pending best mirror.
        # Preserve and warn on any existing mirror because it may represent valid
        # pre-pass lineage.
        if self._pending_best_mirror:
            logger.warning("RECIPE-PASS: unexpected pending best-mirror "
                           "after the pass (invariant breach — leaving "
                           "lineage untouched).")
        # Improvement clock (contract parity with the presweep, so the
        # pass's elapsed wall is not read as pipeline stagnation; the
        # Phase-1 allowance shift happens in the caller over the FULL pass).
        self.last_improvement_iter = self.iteration
        self.last_improvement_time = time.time()
        logger.info(
            f"RECIPE-PASS: COMPLETE — pipeline enters "
            f"{'PRISTINE' if reopened else 'UNVERIFIED (reopen failed)'}; "
            f"{len(self._final_candidates)} non-pipeline candidate(s) "
            f"enrolled; remaining wall {self._budget_remaining():.0f}s.")

    def _recipe_first_deep_apply_reduced_tail(self) -> None:
        """Reduce the deep-tail reserve after banking a recipe-first candidate.

        Runs after the post-pass reset, when the remaining budget reflects the
        completed recipe. The caller must gate this step on both a fired recipe
        and a banked candidate; a recipe miss preserves the full reserve. The
        cap is `min(current_effective, max(0, remaining - llm_floor))` and is
        stored in `_recipe_first_deep_tail_reserve_cap_s`.
        `_deep_wns_tail_reserve_effective_s` consumes the cap, which is
        reduce-only by construction. This preserves the LLM floor while leaving
        the tail the reduced reserve plus any unused loop budget. All branches
        emit a log entry.
        """
        cur = self._deep_wns_tail_reserve_effective_s()
        if not self._recipe_first_deep_banked:
            logger.info(
                f"RECIPE-FIRST-DEEP: tail reserve UNCHANGED at "
                f"{cur:.0f}s (recipe fired but banked no candidate — the "
                f"full deterministic tail is kept; reducing it on a miss "
                f"is the fence failure mode)")
            return
        if cur <= 0.0:
            logger.info(
                "RECIPE-FIRST-DEEP: tail reserve UNCHANGED (reserve not "
                "armed on this run — nothing to reduce; the loop already "
                "owns the remaining wall)")
            return
        try:
            llm_floor = float(getattr(
                self, "phys_opt_preempt_budget_floor_s", 1100.0))
        except (TypeError, ValueError):
            llm_floor = 1100.0
        # SCALED LLM FLOOR coherence: the fire gate reserved
        # max(base, 0.35 x remaining_at_gate) — the loop must keep
        # THAT figure, not the smaller base.
        _gate_floor = self._recipe_first_deep_gate_llm_floor_s
        if _gate_floor is not None:
            llm_floor = max(llm_floor, float(_gate_floor))
        remaining = self._budget_remaining()
        if remaining == float("inf"):
            logger.info(
                f"RECIPE-FIRST-DEEP: tail reserve UNCHANGED at "
                f"{cur:.0f}s (no wall deadline — no boundary to reduce "
                f"against)")
            return
        new = max(0.0, min(cur, remaining - llm_floor))
        if new >= cur:
            logger.info(
                f"RECIPE-FIRST-DEEP: tail reserve UNCHANGED at "
                f"{cur:.0f}s (remaining {remaining:.0f}s already leaves "
                f"the loop >= the {llm_floor:.0f}s LLM floor above it)")
            return
        self._recipe_first_deep_tail_reserve_cap_s = new
        if new <= 0.0:
            # Review fix: when remaining <= the floor the cap is ZERO —
            # the deterministic tail window is gone, not shortened, and
            # the log must say so.
            logger.info(
                f"RECIPE-FIRST-DEEP: recipe-first: tail reserve reduced "
                f"{cur:.0f}s->0s because the recipe candidate is "
                f"banked (remaining {remaining:.0f}s; loop keeps the "
                f"{llm_floor:.0f}s LLM floor; tail ELIMINATED "
                f"(remaining <= floor) — the measured reduced-tail bet, "
                f"see RECIPE_FIRST_DEEP_* constants)")
        else:
            logger.info(
                f"RECIPE-FIRST-DEEP: recipe-first: tail reserve reduced "
                f"{cur:.0f}s->{new:.0f}s because the recipe candidate is "
                f"banked (remaining {remaining:.0f}s; loop keeps the "
                f"{llm_floor:.0f}s LLM floor; tail runs SHORTENED — the "
                f"measured reduced-tail bet, see RECIPE_FIRST_DEEP_* "
                f"constants)")

    async def _maybe_run_recipe_pass_postloop(self) -> None:
        """Run the frozen Tcl candidate pass in the post-loop tail.

        Runs after the LLM loop and all polish stages, immediately before final
        output, so completed downstream stages require no additional reserve.
        The budget calculation must not subtract the finalization reserve again
        because the global deadline already includes it. No session-reset
        reserve is needed because no later LLM loop consumes the tool state.
        The result is registered through the standard routed, hold, and
        cell-validity gates, then competes by WNS under structural validation
        to preserve the never-worse guarantee. Logs use the stable prefixes
        `RECIPE-PASS[postloop]: ARMED`, `RECIPE-PASS[postloop]: fired`, and
        `RECIPE-PASS[postloop]: skipped reason=`. Skip reasons are `disabled`,
        `already_fired`, `band`, `no_pristine_input`, `wall`, or `exception`.
        """
        if not recipe_pass_enabled():
            # Log disabled invocations so they remain distinguishable from hooks
            # that were never reached.
            logger.info("RECIPE-PASS[postloop]: skipped reason=disabled "
                        "(FPL26_RECIPE_PASS off or FPL26_NO_RECIPE_PASS set)")
            return
        _tc0 = len(self.tool_call_details)
        try:
            await self._run_recipe_pass_postloop_inner()
        except Exception as e:
            # Recipe failures are non-fatal; finalization must still run.
            # The live session may be partially mutated, so finalization relies
            # on disk state.
            logger.warning(f"RECIPE-PASS[postloop]: skipped reason=exception "
                           f"{type(e).__name__}: {e!r} (non-fatal; finalize "
                           f"proceeds on disk truth)")
        finally:
            for _d in self.tool_call_details[_tc0:]:
                try:
                    _d["recipe_pass_internal"] = True
                except Exception:
                    pass

    async def _run_recipe_pass_postloop_inner(self) -> None:
        """Post-loop gates + recipe execution + candidate registration.

        Execution is a faithful copy of _run_recipe_pass_inner's execution
        contract (pristine preopen -> frozen steps under a hard pass
        deadline -> measure -> private store -> register_final_candidate
        verify=True); ONLY the gate arithmetic and the post-pass session
        handling differ, and both differences are argued inline.

        SESSION STATE LEFT BEHIND (requirement: preserve the downstream
        contract EXACTLY): this slot deliberately does NOT restart/reopen
        after the recipe.  Audit of everything between this slot and
        process exit — the exit tail is in optimizer/polish_ladder.py and
        finalize in optimizer/finalization.py:
          * the wall-handback log block and _finalize_output_dcp are
            the only code after the slot in _exit_with_ils_polish;
          * MUX candidate-wins path: _structural_validate_dcp re-opens the
            candidate itself before trusting it, and the EDIF regen
            re-opens the shipped output first — both replace whatever
            session state exists;
          * pipeline fast path: pure disk copy of the best_valid mirror
            (_atomic_copy); fresh-EDIF-mirror branch is also a disk copy;
          * no-improvement path: baseline disk copy; its slow EDIF path
            re-opens output_dcp before write_edif;
          * finalize Step 3: re-opens output_dcp before writing EDIF (and
            refuses on an error envelope);
          * the ONE live-session reader is _finalize_refresh_edif (fast
            path with a STALE EDIF mirror), which writes EDIF from memory
            WITHOUT reopening — a documented pre-existing F1 risk
            ("trace, do not patch"), and the deep-replace tail already
            leaves ITS candidate/attempt state in-session at this exact
            point (registers verify=False, "no session re-open, shipping
            state undisturbed").  Exposure class is PARITY with the
            validated surface, not new — and HARDENED beyond it:
            once this slot mutates the
            session it sets _finalize_session_untrusted, and that branch
            now REFUSES the live-session regen (ships
            VALID_OPTIMIZED_NO_EDIF) instead of stamping a
            different-flow netlist EDIF onto the shipped DCP.
        Restoring the session would cost the ship-scale restart+reopen
        reserve (up to 1800s) that made the pre-LLM deep gate dead code —
        the arithmetic this redesign exists to fix.

        Never touches best_valid: auto-bank is suppressed for the whole
        pass (_tail_ctrl_suppress_autobank bracket, presweep pattern) AND
        the _in_ils_stage bracket bypasses the auto-bank hook as well;
        banking happens only via register_final_candidate -> finalize MUX.
        """
        wns_in = self.initial_wns  # measured once at open; NOT re-measured
        band = recipe_pass_band(wns_in)
        if band is None:
            logger.info("RECIPE-PASS[postloop]: skipped reason=band "
                        "(initial_wns unmeasured/non-negative — no honest "
                        "band key)")
            return
        logger.info(f"RECIPE-PASS[postloop]: ARMED band={band} "
                    f"wns_in={wns_in:.3f}")
        if self._recipe_pass_preloop_fired:
            # The marker is set before the pre-LLM pass first mutates the session.
            # It proves only that the pass started, not that it completed or
            # registered a candidate. Skipping here prevents applying the same
            # recipe twice.
            logger.info("RECIPE-PASS[postloop]: skipped reason=already_fired "
                        "(pre-LLM pass already mutated a session this "
                        "process; registration not implied)")
            return
        if band == "shallow":
            # Shallow recipes run in the pre-LLM slot, whose wall gate can arm
            # within a one-hour budget. The post-loop slot handles bands whose
            # pre-LLM gates cannot arm.
            logger.info("RECIPE-PASS[postloop]: skipped reason=band "
                        "(shallow is the pre-LLM slot's band; post-loop "
                        "targets pre-LLM-unarmable bands only)")
            return
        input_dcp = self.input_dcp_path
        if not (input_dcp and Path(input_dcp).exists()):
            logger.info("RECIPE-PASS[postloop]: skipped reason="
                        "no_pristine_input (input DCP missing on disk — "
                        "cannot honor the pristine-reopen contract, "
                        "EXPLORATION_PLAN FIND #2)")
            return
        if self._wall_handback_break_due():
            # Wall handback is enabled by default and propagated through each
            # optimization attempt. When saturation arms it, the remaining wall
            # time is reserved for handback, so this pass defers.
            # Evaluate this guard before any session mutation.
            logger.info(
                f"RECIPE-PASS[postloop]: skipped reason=handback_armed "
                f"band={band} remaining={self._budget_remaining():.0f}s "
                f"(wall-handback is the proven lever on this run class; "
                f"recipe defers — A/B the residual firing set only)")
            return
        steps = {"deep": RECIPE_PASS_DEEP_TCL,
                 "mid": RECIPE_PASS_MID_TCL}[band]
        expected_s = {"deep": RECIPE_PASS_DEEP_EXPECTED_S,
                      "mid": RECIPE_PASS_MID_EXPECTED_S}[band]
        remaining = self._budget_remaining()
        overhead = RECIPE_PASS_POSTLOOP_OVERHEAD_RESERVE_S
        _overhead_terms = (
            f"[measure {RECIPE_PASS_MEASURE_RESERVE_S:.0f} + store "
            f"{RECIPE_PASS_STORE_RESERVE_S:.0f} + register "
            f"{RECIPE_PASS_REGISTER_RESERVE_S:.0f}; NO reset term — "
            f"session left as-is, NO tail reserve — tail stages already "
            f"ran, NO LLM floor — loop is done]")
        if band == "deep":
            # Require a 1.5x runtime margin plus 540 s for post-pass work.
            # This admits the deep recipe only when sufficient wall time remains.
            need = RECIPE_PASS_TIMEOUT_FACTOR * expected_s + overhead
            if remaining < need:
                logger.info(
                    f"RECIPE-PASS[postloop]: skipped reason=wall band=deep "
                    f"(remaining {remaining:.0f}s < pass budget "
                    f"{RECIPE_PASS_TIMEOUT_FACTOR * expected_s:.0f}s "
                    f"(1.5x{expected_s:.0f}) + overheads {overhead:.0f}s "
                    f"{_overhead_terms} = {need:.0f}s — fail closed; "
                    f"stranded wall too small for the deep recipe)")
                return
        else:  # band == "mid"
            # With a one-hour wall budget, the 2,800 s estimate plus 540 s
            # overhead exceeds the 3,200 s maximum remaining time.
            # The gate fails closed; longer wall budgets can arm the mid recipe.
            need = expected_s + overhead
            if remaining < need:
                logger.info(
                    f"RECIPE-PASS[postloop]: skipped reason=wall band=mid "
                    f"(remaining {remaining:.0f}s < measured recipe cost "
                    f"{expected_s:.0f}s (cross-box probe, boxes 4/5) + "
                    f"overheads {overhead:.0f}s {_overhead_terms} = "
                    f"{need:.0f}s.  Stranded wall (750-1300s measured) "
                    f"cannot fund 2800s — DORMANT-BY-ARITHMETIC on the "
                    f"ship wall, same honest-skip contract as the "
                    f"pre-LLM mid gate)")
                return
        # Limit the recipe body to 1.5x expected cost while reserving estimated
        # post-pass overhead. Do not subtract the finalization reserve here;
        # _budget_deadline already excludes it. Later operations enforce their
        # own caps, and tool calls remain fenced by the global budget deadline.
        pass_budget = RECIPE_PASS_TIMEOUT_FACTOR * expected_s
        if remaining != float("inf"):
            pass_budget = min(pass_budget, remaining - overhead)
        if pass_budget < 60.0:
            logger.info(f"RECIPE-PASS[postloop]: skipped reason=wall "
                        f"(pass budget {pass_budget:.0f}s < 60s floor)")
            return
        logger.info(f"RECIPE-PASS[postloop]: fired recipe={band} "
                    f"wns_in={wns_in:.3f} (pass_budget={pass_budget:.0f}s, "
                    f"steps={len(steps)}, remaining={remaining:.0f}s)")
        t0 = time.time()
        pass_deadline = t0 + pass_budget
        # Recipe validity is defined relative to the input checkpoint, not the
        # loop's final session. Reopen the pristine input before mutation.
        prev_suppress = self._tail_ctrl_suppress_autobank
        self._tail_ctrl_suppress_autobank = True
        # Temporarily enter the self-capped stage so generic unroute and polish
        # guards do not reject the recipe's own unroute. Preserve the prior flag
        # for nested callers. Tool calls remain bounded by the pass and global
        # budget deadlines.
        prev_ils_flag = self._in_ils_stage
        self._in_ils_stage = True
        try:
            # Mark the live session untrusted before reopening; even a failed open
            # may disturb it. Finalization must reopen disk state rather than
            # combine an unrelated EDIF with the selected checkpoint.
            # This path never clears the flag.
            self._finalize_session_untrusted = True
            if not await self._presweep_reopen(input_dcp):
                logger.warning("RECIPE-PASS[postloop]: failed "
                               "step=pristine_preopen (recipe not run; "
                               "wall lost, nothing banked)")
                return
            for i, step in enumerate(steps, 1):
                step_left = pass_deadline - time.time()
                if step_left < 30.0:
                    logger.warning(
                        f"RECIPE-PASS[postloop]: failed step={i}/{len(steps)} "
                        f"reason=pass_timeout ({pass_budget:.0f}s budget "
                        f"exhausted before '{step}'; aborting — wall lost, "
                        f"nothing banked)")
                    return
                if step == RECIPE_PASS_SESSION_ROUNDTRIP:
                    ok = await self._recipe_pass_roundtrip(
                        step_left, tag="RECIPE-PASS[postloop]")
                    if not ok:
                        logger.warning(
                            f"RECIPE-PASS[postloop]: failed "
                            f"step={i}/{len(steps)} "
                            f"reason=session_roundtrip (aborting)")
                        return
                    continue
                try:
                    res = await asyncio.wait_for(
                        self.call_tool("vivado_run_tcl",
                                       {"command": step,
                                        "timeout": float(step_left)}),
                        timeout=float(step_left) + 60.0)
                except Exception as e:
                    logger.warning(
                        f"RECIPE-PASS[postloop]: failed step={i}/{len(steps)} "
                        f"reason=step_raised:{type(e).__name__} "
                        f"cmd='{step}' (aborting)")
                    return
                if _looks_like_tool_error(res):
                    logger.warning(
                        f"RECIPE-PASS[postloop]: failed step={i}/{len(steps)} "
                        f"reason=step_error cmd='{step}' "
                        f"detail={str(res)[:120]} (aborting)")
                    return
            # ---- measure + private store + register (contract 1-3, same
            # as the pre-LLM inner) ----
            try:
                wns_out = await asyncio.wait_for(
                    self.get_wns_for_target_clock(self._call_vivado_tool),
                    timeout=600.0)
            except Exception:
                wns_out = None
            if wns_out is None:
                logger.warning("RECIPE-PASS[postloop]: failed step=measure "
                               "reason=wns_unmeasurable (nothing banked)")
                return
            store_dir = self.run_dir / "final_candidates"
            store_dir.mkdir(parents=True, exist_ok=True)
            store = store_dir / f"recipe_pass_postloop_{band}.dcp"
            try:
                res = await asyncio.wait_for(
                    self.call_tool("vivado_write_checkpoint", {
                        "dcp_path": str(store.resolve()),
                        "force": True}),
                    timeout=900.0)
            except Exception as e:
                res = (f'{{"error": "write_checkpoint raised '
                       f'{type(e).__name__}"}}')
            if _looks_like_tool_error(res) or not (
                    store.exists() and store.stat().st_size > 0):
                logger.warning("RECIPE-PASS[postloop]: failed step=store "
                               f"reason=write_checkpoint detail="
                               f"{str(res)[:120]} (nothing banked)")
                return
            # In-memory state == the candidate on disk, so verify=True
            # re-runs the exact routed/hold/cell gates against it (the
            # same contract the presweep + pre-LLM registrations rely on).
            registered = await self.register_final_candidate(
                store, float(wns_out), f"recipe_pass_postloop_{band}",
                verify=True)
            logger.info(
                f"RECIPE-PASS[postloop]: result recipe={band} "
                f"wns_in={wns_in:.3f} wns_out={float(wns_out):.3f} "
                f"registered={registered} dt={time.time() - t0:.0f}s "
                f"(candidate competes ONLY at the finalize MUX; session "
                f"left on the recipe state — finalize is disk-truth-"
                f"driven, see the session audit in this method's "
                f"docstring)")
        finally:
            self._in_ils_stage = prev_ils_flag
            self._tail_ctrl_suppress_autobank = prev_suppress
            # Post-loop execution does not reset the session, shift phase
            # allowances, or update the stagnation clock; their consumers have
            # already finished. The pass never banks into best_valid, so a
            # pending mirror is a lineage breach and must remain unapplied.
            if self._pending_best_mirror:
                logger.warning("RECIPE-PASS[postloop]: unexpected pending "
                               "best-mirror after the pass (invariant "
                               "breach — leaving lineage untouched).")
