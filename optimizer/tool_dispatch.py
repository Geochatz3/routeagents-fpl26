"""STAGE 4 — LLM LOOP (the tool layer it acts through)

The tool dispatch loop: assembling the tool list the model may call, and
executing one call with its timing, cost and error accounting.

Named for the stage it implements: the slide, the video and the paper all
call it by this name, so a reader can move between them without a glossary.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time

# Log through the orchestrator's logger: this code still belongs to the same
# run, and a module-named logger would change every line it emits.
logger = logging.getLogger("dcp_optimizer")


FINALIZE_PER_CALL_TIMEOUT_S = 120.0
MIN_USEFUL_TOOL_SECONDS = 30.0


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


class ToolDispatchMixin:
    """Methods mixed into DCPOptimizer; they run against its instance state."""

    async def _collect_tools(self):
        """Collect and convert tools from both MCP servers."""
        self.tools = []

        rw_response = await self.rapidwright_session.list_tools()
        for tool in rw_response.tools:
            self.tools.append(convert_mcp_tool_to_openai(tool, "rapidwright"))

        v_response = await self.vivado_session.list_tools()
        for tool in v_response.tools:
            self.tools.append(convert_mcp_tool_to_openai(tool, "vivado"))

        # This high-level recipe tool runs the complete cell-replacement cycle
        # as one LLM-visible call. Substeps use the existing MCP sessions so WNS
        # tracking and force-continue handling still observe every tool call.
        # The implementation is in _recipe_cell_replacement().
        self.tools.append({
            "type": "function",
            "function": {
                "name": "recipe_register_retiming",
                "description": (
                    "Run a register-retiming pass via phys_opt_design with one "
                    "of the two retiming-enabled directives "
                    "(AlternateFlowWithRetiming or AddRetime). Routes the "
                    "result and measures.  This is the contest-promoted "
                    "technique #5 — works on designs where pblock + standard "
                    "phys_opt have plateaued (e.g. corescore_500_mod's "
                    "+53 MHz ceiling vs +79 published BL).  Returns JSON "
                    "with directive_used, delta_fmax_mhz, route_errors."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directive": {
                            "type": "string",
                            "description": (
                                "Which retiming directive to use.  "
                                "'AlternateFlowWithRetiming' (default) does aggressive "
                                "replication + DSP/BRAM optimization with retiming.  "
                                "'AddRetime' is more conservative — default flow + retiming."
                            ),
                            "enum": ["AlternateFlowWithRetiming", "AddRetime"],
                            "default": "AlternateFlowWithRetiming",
                        },
                        "rewrite_dcp": {
                            "type": "boolean",
                            "description": (
                                "When true (default), write the post-retime DCP to a "
                                "temp file and open it back to ensure routing is "
                                "fresh.  When false, runs phys_opt in-place and "
                                "trusts Vivado's internal state."
                            ),
                            "default": True,
                        },
                    },
                    "required": [],
                },
            },
        })
        self.tools.append({
            "type": "function",
            "function": {
                "name": "recipe_lut_optimization",
                "description": (
                    "Run the LUT-input-cone optimisation cycle: extract LUT input "
                    "pins from the worst critical paths, call "
                    "optimize_lut_input_cone to merge cascaded small LUTs into "
                    "single larger LUTs, then write/open/route/measure. Returns "
                    "JSON with status, pins_extracted, pins_optimized, "
                    "route_errors, delta_fmax_mhz. Use this when the critical "
                    "path is LUT-bound (multiple LUT levels in cascade)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "num_paths": {
                            "type": "integer",
                            "description": "Critical paths to extract pins from (default 10).",
                            "default": 10,
                        },
                        "target_max_path": {
                            "type": "integer",
                            "description": "Limit to LUT inputs on the worst N paths (default 2).",
                            "default": 2,
                        },
                        "max_pins_to_try": {
                            "type": "integer",
                            "description": "Cap how many pins to feed to optimize_lut_input_cone (default 12).",
                            "default": 12,
                        },
                    },
                    "required": [],
                },
            },
        })
        self.tools.append({
            "type": "function",
            "function": {
                "name": "recipe_cell_replacement",
                "description": (
                    "Run the contest's published cell-replacement recipe end-to-end: "
                    "extract critical-path pins, run RapidWright detour analysis, "
                    "move the worst-detour cells with optimize_cell_placement, "
                    "write a new DCP, route it in Vivado, and report the result. "
                    "Returns a JSON object with status, candidates_total, cells_moved, "
                    "route_errors, delta_fmax_mhz.  Refuses to run on designs flagged "
                    "harmful or error in the per-design recipe applicability table.  "
                    "Use this when the initial analysis shows critical paths with "
                    "movable cells (typical when avg spread is moderate but specific "
                    "cells are mis-placed).  After this returns success, the "
                    "Vivado session has the optimized DCP open."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "num_paths": {
                            "type": "integer",
                            "description": "How many critical paths to extract pins from (default 10).",
                            "default": 10,
                        },
                        "detour_threshold": {
                            "type": "number",
                            "description": "Minimum routed/Manhattan ratio to flag a cell as a candidate (default 2.0).",
                            "default": 2.0,
                        },
                        "target_max_path": {
                            "type": "integer",
                            "description": "Limit candidates to the worst N critical paths (default 2).",
                            "default": 2,
                        },
                        "max_cells_to_try": {
                            "type": "integer",
                            "description": "Cap how many cells to try moving (default 8). Each is tried sequentially.",
                            "default": 8,
                        },
                    },
                    "required": [],
                },
            },
        })
        self.tools.append({
            "type": "function",
            "function": {
                "name": "recipe_post_route_phys_opt_sweep",
                "description": (
                    "Run a bounded, granular phys_opt_design sweep on the "
                    "currently-routed design.  Each step in the sweep enables "
                    "ONE sub-optimization (critical_cell_opt, equ_drivers_opt, "
                    "placement_opt, dsp_register_opt, restruct_opt, or "
                    "slr_crossing_opt) and measures WNS.  Improvements are "
                    "kept; regressions revert to the prior best via the "
                    "best_valid.dcp mirror.  Designed for the budget-tight "
                    "case where a full phys_opt -directive Explore would be "
                    "refused by the deadline-aware dispatcher (~5-8 min per "
                    "sub-flag on huge designs, vs 30-50 min for full "
                    "directive).  Returns JSON with per-flag {status, "
                    "delta_wns_ns, delta_fmax_mhz, elapsed_s} plus cumulative "
                    "deltas.  Use this when a Class G / Class E move has "
                    "been refused, or as the first move on huge designs."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "flags": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["critical_cell_opt", "equ_drivers_opt",
                                         "placement_opt", "dsp_register_opt",
                                         "restruct_opt", "slr_crossing_opt"],
                            },
                            "description": (
                                "Subset of phys_opt sub-flags to try.  "
                                "Default sweeps all 6 in order of expected "
                                "value.  Pass a custom list to focus on "
                                "specific optimizations (e.g., "
                                "['critical_cell_opt', 'equ_drivers_opt'] "
                                "for fanout-heavy designs)."
                            ),
                        },
                        "epsilon_ns": {
                            "type": "number",
                            "description": (
                                "Minimum WNS improvement (ns) to commit a "
                                "sub-flag's result.  Below this the change "
                                "is treated as noise and reverted.  Default "
                                "0.010 ns absorbs Vivado's report-to-report "
                                "jitter."
                            ),
                            "default": 0.010,
                        },
                        "skip_route_recheck": {
                            "type": "boolean",
                            "description": (
                                "When true (default), skip the explicit "
                                "route-status check between sub-flags — "
                                "phys_opt_design is conservative about "
                                "routing legality.  Set false to be paranoid "
                                "(adds ~10s per sub-flag)."
                            ),
                            "default": True,
                        },
                        "early_exit_on_first_gain": {
                            "type": "boolean",
                            "description": (
                                "When true (DEFAULT for the timing-onion "
                                "loop), the recipe returns as soon as ONE "
                                "sub-flag commits.  Leaves wall time for "
                                "the LLM to layer a different transform "
                                "on top.  Set false to sweep all flags "
                                "back-to-back regardless of intermediate "
                                "gains (the original behaviour)."
                            ),
                            "default": True,
                        },
                        "max_successful_subpasses": {
                            "type": "integer",
                            "description": (
                                "Hard cap on how many sub-flags can commit "
                                "in one recipe invocation.  Default 1 to "
                                "match early_exit_on_first_gain.  Set "
                                "higher (e.g. 3) to allow chained sub-flag "
                                "commits within a single recipe call."
                            ),
                            "default": 1,
                        },
                        "min_remaining_time_for_next_layer_s": {
                            "type": "number",
                            "description": (
                                "After a commit, the recipe returns if "
                                "remaining budget falls below this value, "
                                "preserving room for a follow-up recipe.  "
                                "Default 600s (10 min) — typical floor "
                                "for a useful second-layer transform."
                            ),
                            "default": 600.0,
                        },
                    },
                    "required": [],
                },
            },
        })
        self.tools.append({
            "type": "function",
            "function": {
                "name": "recipe_critical_path_focused_phys_opt",
                "description": (
                    "Bounded scoped phys_opt_design.  Extracts the top-N "
                    "critical-path endpoints, creates a Vivado path group "
                    "constrained to those endpoints "
                    "(group_path -name pg_critical_focus -to <endpoints>), "
                    "then runs vivado_phys_opt_design with the chosen "
                    "sub-flag AND path_groups=pg_critical_focus so the "
                    "optimization only touches the worst N paths.  "
                    "Designed to break the boom_soc 35-min critical_cell_opt "
                    "bottleneck — by restricting scope to the worst paths, "
                    "the same transform completes in a fraction of the "
                    "time, leaving budget for layer-2 optimization.  "
                    "Reverts via best_valid.dcp on regression."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "num_paths": {
                            "type": "integer",
                            "description": "How many worst critical paths to include in the focused group (default 20).",
                            "default": 20,
                        },
                        "sub_flag": {
                            "type": "string",
                            "enum": ["critical_cell_opt", "equ_drivers_opt",
                                     "placement_opt", "restruct_opt",
                                     "dsp_register_opt", "critical_pin_opt"],
                            "description": "Which phys_opt sub-optimization to run on the focused path group (default 'critical_cell_opt').",
                            "default": "critical_cell_opt",
                        },
                        "epsilon_ns": {
                            "type": "number",
                            "description": "Min WNS improvement (ns) to commit. Default 0.010 ns.",
                            "default": 0.010,
                        },
                    },
                    "required": [],
                },
            },
        })
        self.tools.append({
            "type": "function",
            "function": {
                "name": "recipe_high_fanout_timing_replication",
                "description": (
                    "Bounded high-fanout-driver replication recipe.  "
                    "Detects high-fanout nets on the worst critical "
                    "paths via vivado_get_critical_high_fanout_nets, "
                    "then runs vivado_phys_opt_design with "
                    "force_replication_on_nets=<filtered net list> "
                    "to replicate the drivers surgically.  Re-measures "
                    "WNS; if regression, reverts via best_valid.dcp. "
                    "Use when the design's critical paths are dominated "
                    "by a few high-fanout drivers (boom_soc, "
                    "corescore_500_mod patterns).  Returns JSON with "
                    "nets_replicated, delta_wns_ns, delta_fmax_mhz."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "num_paths": {
                            "type": "integer",
                            "description": "How many critical paths to scan for high-fanout nets (default 20).",
                            "default": 20,
                        },
                        "min_fanout": {
                            "type": "integer",
                            "description": "Minimum fanout threshold for replication candidates (default 100).",
                            "default": 100,
                        },
                        "max_nets_to_replicate": {
                            "type": "integer",
                            "description": "Cap how many high-fanout nets to feed force_replication_on_nets (default 10).",
                            "default": 10,
                        },
                        "epsilon_ns": {
                            "type": "number",
                            "description": "Min WNS improvement (ns) to commit the replication. Default 0.010 absorbs jitter.",
                            "default": 0.010,
                        },
                    },
                    "required": [],
                },
            },
        })

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call on the appropriate MCP server."""
        # Synthetic high-level recipe tools — see _collect_tools.
        if tool_name == "recipe_cell_replacement":
            return await self._recipe_cell_replacement(arguments)
        if tool_name == "recipe_lut_optimization":
            return await self._recipe_lut_optimization(arguments)
        if tool_name == "recipe_register_retiming":
            return await self._recipe_register_retiming(arguments)
        if tool_name == "recipe_post_route_phys_opt_sweep":
            return await self._recipe_post_route_phys_opt_sweep(arguments)
        if tool_name == "recipe_high_fanout_timing_replication":
            return await self._recipe_high_fanout_timing_replication(arguments)
        if tool_name == "recipe_critical_path_focused_phys_opt":
            return await self._recipe_critical_path_focused_phys_opt(arguments)

        # Finalization bypasses the budget gate so the current in-memory gain can
        # be written after the iteration budget expires. ILS calls use their own
        # deadline discipline. Per-call timeouts still protect against tool hangs.
        if self._in_finalize or self._in_ils_stage:
            skip, skip_reason = (False, "")
        else:
            # The preflight policy refuses calls when the budget is exhausted, too
            # little useful time remains after finalization, or a risky call cannot
            # fit its estimated window. Refusals return an error envelope and mark
            # the budget killed so the iteration loop proceeds to finalization.
            skip, skip_reason = self._should_skip_for_budget(tool_name, arguments)
        # Timing-constraint mutations are prohibited by the optimization rules.
        # Reject them at the tool boundary so the planner can choose a legal
        # physical change. Final fingerprint validation remains authoritative
        # because command-string filtering is not a complete security boundary.
        if tool_name == "vivado_run_tcl" and arguments:
            try:
                from optimizer.constraint_guard import (deny_reason,
                                                        is_constraint_mutating)
                _cmd = (arguments.get("command")
                        or arguments.get("tcl_command") or "")
                _hit, _which = is_constraint_mutating(_cmd)
                if _hit:
                    logger.warning(
                        f"[constraint-guard] REFUSED {_which}: timing "
                        f"constraint edits are disqualifying. Payload: "
                        f"{str(_cmd)[:200]}")
                    self.tool_call_details.append({
                        "tool_name": tool_name,
                        "status": "refused_constraint_edit",
                        "detail": _which,
                        # Every consumer of tool_call_details sums this key;
                        # a guard entry without it crashed a completed run.
                        "elapsed_time": 0.0,
                    })
                    return deny_reason(_which)
            except Exception as e:  # never let the guard break a run
                logger.warning(f"[constraint-guard] check raised (ignored): {e!r}")
        # Optionally request a low-cost critique before heavy tool work. The critique
        # is advisory and cannot skip, cancel, or rewrite the call. It runs after
        # budget admission and is a no-op when disabled.
        if not skip:
            try:
                await self._maybe_plan_critique(tool_name, arguments)
            except Exception as e:
                logger.warning(f"plan-critic raised (ignored): {e!r}")
        if skip:
            self._strategies_skipped_budget.append(f"{tool_name} ({skip_reason})")
            self._budget_killed = True
            # "Nothing schedulable within the remaining
            # wall" — the sanctioned budget-kill saturation signal.
            self._maybe_arm_wall_handback(f"budget_skip:{skip_reason}")
            mw = self.max_wall_seconds
            mw_str = f"{mw:.0f}s" if mw is not None else "n/a"
            logger.warning(
                f"Budget skip: {tool_name} not started ({skip_reason}); "
                f"remaining={self._budget_remaining():.0f}s of {mw_str}."
            )
            self.tool_call_details.append({
                "tool_name": tool_name,
                "iteration": self.iteration,
                "elapsed_time": 0.0,
                "wns": None,
                "error": True,
                "error_message": f"budget_skip:{skip_reason}",
            })
            skip_payload = json.dumps({
                "error": "tool_skipped_budget",
                "reason": skip_reason,
                "remaining_seconds": self._budget_remaining(),
            })
            # Classify skip + emit trace event.
            # Skip path is the BUDGET_SKIP code in the taxonomy.
            tool_err = self._classify_tool_payload(
                skip_payload, context="call_tool_skip"
            )
            self._emit_decision({
                "decision_source": "executor",
                "phase": "call_tool",
                "tool_name": tool_name,
                "action_label": "budget_skip",
                "runtime_s": 0.0,
                "tool_error_code": tool_err.code if tool_err is not None else "BUDGET_SKIP",
                "notes": skip_reason,
            })
            return skip_payload

        # Routed-state-destroying operations require a later full route before the
        # result is bankable, so admission includes the predicted reroute cost.
        # Router-plan blocks are enforced here rather than left as prompt advice.
        # Refusals steer planning toward bankable state-preserving operations.
        # Finalization and ILS bypass both gates because they have separate bounds.
        # Assessment is pure and fails open if routing policy evaluation fails.
        if (not self._in_finalize and not self._in_ils_stage
                and _router_blocked_reason is not None):
            _blk = _router_blocked_reason(
                getattr(self, "recipe_router_plan", None), tool_name, arguments)
            if _blk:
                logger.warning(f"[router-block] REFUSED {tool_name}: {_blk}")
                self.tool_call_details.append({
                    "tool_name": tool_name,
                    "iteration": self.iteration,
                    "elapsed_time": 0.0,
                    "wns": None,
                    "error": True,
                    "error_message": f"router_block_refused:{_blk[:160]}",
                })
                return (f"REFUSE: {_blk}. This move is forbidden for this "
                        f"design's feature profile — choose a step from the "
                        f"recipe plan's action list instead.")

        if (self._unroute_gate_enabled
                and not self._in_finalize
                and not self._in_ils_stage
                and self.is_routed_state_destroying(tool_name, arguments)):
            from optimizer.route_gate import assess_destructive_reroute
            assessment = assess_destructive_reroute(
                self._budget_remaining(),
                self.tool_call_details,
                self._input_cell_count,
            )
            if not assessment.feasible:
                # Observable in harness logs: long-wall rehearsals grep for
                # this line, the same discipline as the "Budget skip:" log
                # above.
                logger.warning(
                    f"[unroute-gate] REFUSED {tool_name}: "
                    f"{assessment.reason}"
                )
                self.tool_call_details.append({
                    "tool_name": tool_name,
                    "iteration": self.iteration,
                    "elapsed_time": 0.0,
                    "wns": None,
                    "error": True,
                    "error_message":
                        f"unroute_gate_refused:{assessment.reason[:160]}",
                })
                # Use the standard error envelope so the refusal is classified
                # as a tool error. Do not mark the budget killed; incremental
                # work may continue.
                refusal_payload = json.dumps({
                    "error": "unroute_gate_refused",
                    "reason": assessment.reason,
                    "predicted_reroute_s": assessment.predicted_reroute_s,
                    "remaining_seconds": assessment.remaining_wall_s,
                    "steering": (
                        "Do NOT destroy the current routed state — the "
                        "mandatory full re-route cannot finish in the "
                        "remaining wall budget. Run a bankable "
                        "incremental step instead: phys_opt_design on "
                        "the routed design (e.g. -directive "
                        "AggressiveExplore or AlternateFlowWithRetiming) "
                        "or an incremental route_design that preserves "
                        "routed state, then measure WNS so the "
                        "improvement is banked."
                    ),
                })
                tool_err = self._classify_tool_payload(
                    refusal_payload, context="call_tool_unroute_gate"
                )
                self._emit_decision({
                    "decision_source": "executor",
                    "phase": "call_tool",
                    "tool_name": tool_name,
                    "action_label": "unroute_gate_refused",
                    "runtime_s": 0.0,
                    "tool_error_code": (tool_err.code
                                        if tool_err is not None
                                        else "UNROUTE_GATE_REFUSED"),
                    "notes": assessment.reason,
                })
                return refusal_payload

        # While post-route polish is reserved, speculative routed-state-destroying
        # operations may use only the time remaining outside that reserve.
        # State-preserving polish operations retain the full remaining window.
        # A reserve refusal does not end the run; bankable incremental work continues.
        if (not self._in_finalize
                and not self._in_ils_stage
                and self._is_risky(tool_name, arguments)
                and self.is_routed_state_destroying(tool_name, arguments)):
            _reserve = self._polish_reserve_armed_s()
            if _reserve > 0.0:
                _remaining = self._budget_remaining()
                _est = self._estimate_tool_runtime(tool_name, is_risky=True)
                _window = _remaining - _reserve
                if _est > _window:
                    _reason = (
                        f"estimated_{_est:.0f}s_exceeds_speculative_window_"
                        f"{max(0.0, _window):.0f}s_(polish_reserve_"
                        f"{_reserve:.0f}s_of_{_remaining:.0f}s_remaining)")
                    logger.warning(
                        f"[polish-reserve] REFUSED speculative {tool_name}: "
                        f"{_reason}")
                    self.tool_call_details.append({
                        "tool_name": tool_name,
                        "iteration": self.iteration,
                        "elapsed_time": 0.0,
                        "wns": None,
                        "error": True,
                        "error_message": f"polish_reserve_refused:{_reason}",
                    })
                    reserve_payload = json.dumps({
                        "error": "polish_reserve_refused",
                        "reason": _reason,
                        "polish_reserve_s": _reserve,
                        "remaining_seconds": _remaining,
                        "steering": (
                            "The last "
                            f"{_reserve:.0f}s of the wall are reserved for "
                            "the final post-route polish of the banked "
                            "best. Do not start another full place/route "
                            "gamble now. Run a state-preserving step "
                            "instead: phys_opt_design on the routed design "
                            "(e.g. -directive AggressiveExplore) or an "
                            "incremental route_design, then measure WNS so "
                            "the improvement is banked."
                        ),
                    })
                    tool_err = self._classify_tool_payload(
                        reserve_payload, context="call_tool_polish_reserve")
                    self._emit_decision({
                        "decision_source": "executor",
                        "phase": "call_tool",
                        "tool_name": tool_name,
                        "action_label": "polish_reserve_refused",
                        "runtime_s": 0.0,
                        "tool_error_code": (tool_err.code
                                            if tool_err is not None
                                            else "POLISH_RESERVE_REFUSED"),
                        "notes": _reason,
                    })
                    return reserve_payload

        # Parse server prefix from tool name
        if tool_name.startswith("rapidwright_"):
            session = self.rapidwright_session
            actual_name = tool_name[len("rapidwright_"):]
        elif tool_name.startswith("vivado_"):
            session = self.vivado_session
            actual_name = tool_name[len("vivado_"):]
        else:
            return json.dumps({"error": f"Unknown tool prefix in: {tool_name}"})

        # Track timing for this tool call
        start_time = time.time()
        wns_measured = None
        error_occurred = False
        # Snapshot best_wns before the call so the
        # trace record can report wns_before → wns_after deltas without
        # extra Vivado round-trips.  None when no measurement exists.
        wns_before_call = (
            float(self.best_wns)
            if (self.best_wns is not None and self.best_wns != float("-inf"))
            else None
        )

        try:
            logger.info(f"Calling {tool_name} with args: {json.dumps(arguments)[:200]}...")
            # Finalization uses a fixed timeout because its deadline-derived timeout
            # may already be zero and would cancel writes immediately. The 120 s
            # limit accommodates large-design output writes while bounding tool hangs.
            if self._in_finalize:
                timeout = FINALIZE_PER_CALL_TIMEOUT_S
            else:
                # Tool identity allows speculative destroying operations to be capped
                # at the time remaining outside the polish reserve. ILS calls omit
                # that identity so polish stages retain the full window; internal
                # destructive cycles are bounded by the ILS-specific deadline.
                timeout = self._deadline_aware_timeout(
                    None if self._in_ils_stage else tool_name, arguments)
            if timeout is None:
                # No budget set (legacy dev mode) — call without a timeout.
                result = await session.call_tool(actual_name, arguments)
            else:
                # Bound each tool call so an uninterruptible operation cannot consume
                # the remaining wall-clock budget. A timeout records the skipped
                # strategy, marks the budget killed, and returns an error envelope.
                try:
                    result = await asyncio.wait_for(
                        session.call_tool(actual_name, arguments),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    elapsed_time = time.time() - start_time
                    self._budget_killed = True
                    # Distinguish a polish-reserve fence timeout from true deadline
                    # exhaustion for diagnostics. Preserve the first recorded cause.
                    if (self._last_timeout_fence_capped
                            and self._budget_kill_cause is None):
                        self._budget_kill_cause = "polish_reserve_fence"
                    self._strategies_skipped_budget.append(
                        f"{tool_name} (timed_out after {elapsed_time:.0f}s of "
                        f"{timeout:.0f}s allowance)"
                    )
                    logger.warning(
                        f"Budget timeout: {tool_name} cancelled after "
                        f"{elapsed_time:.0f}s (allowance={timeout:.0f}s); "
                        f"remaining={self._budget_remaining():.0f}s. Session "
                        "may be compromised — finalizing soon."
                    )
                    self.tool_call_details.append({
                        "tool_name": tool_name,
                        "iteration": self.iteration,
                        "elapsed_time": elapsed_time,
                        "wns": None,
                        "error": True,
                        "error_message": f"budget_timeout_after_{elapsed_time:.0f}s",
                    })
                    # Classify + trace timeout.
                    self._emit_decision({
                        "decision_source": "executor",
                        "phase": "call_tool",
                        "tool_name": tool_name,
                        "action_label": "budget_timeout",
                        "runtime_s": elapsed_time,
                        "tool_error_code": "TIMEOUT_BUDGET",
                        "notes": (
                            f"allowance_s={timeout:.0f} elapsed_s={elapsed_time:.0f}"
                        ),
                    })
                    return json.dumps({
                        "error": "tool_timed_out_budget",
                        "elapsed_seconds": elapsed_time,
                        "allowance_seconds": timeout,
                    })
            
            # Extract text content from result
            if result.content:
                text_parts = [c.text for c in result.content if hasattr(c, 'text')]
                result_text = "\n".join(text_parts)
            else:
                result_text = "(no output)"
            
            # Track WNS from timing reports and get_wns calls
            if tool_name == "vivado_report_timing_summary":
                # If target clock is set, get clock-specific WNS instead of overall
                if self.target_clock:
                    clock_wns = None
                    try:
                        # Resolve this helper through self; using super() from the mixin
                        # would follow the mixin's MRO rather than the optimizer base.
                        clock_wns = await self.get_wns_for_target_clock(self._call_vivado_tool)
                    except Exception as e:
                        # Only failure of the clock-specific probe may disable
                        # target-clock tracking. Keep this exception scope
                        # narrow: unrelated mirror or guard failures must not
                        # make later scored-clock improvements invisible behind
                        # another clock's worse slack.
                        logger.warning(f"Failed to get clock-specific WNS, falling back to overall: {e}")
                        self.target_clock = None  # Fall through to overall WNS parsing
                    if clock_wns is not None:
                        try:
                            current_wns = clock_wns
                            wns_measured = current_wns
                            current_fmax = self.calculate_fmax(current_wns, self.clock_period)
                            fmax_str = f", fmax: {current_fmax:.2f} MHz" if current_fmax is not None else ""
                            if current_wns > self.best_wns and not await self._routed_ok_for_best():
                                logger.warning(f"WNS {current_wns:.3f} beats best but design is NOT fully routed (estimated timing) — REJECTED as best (phantom-best guard).")
                            elif current_wns > self.best_wns:
                                old_best_wns = self.best_wns
                                logger.info(f"New best WNS (clock: {self.target_clock}): {current_wns:.3f} ns{fmax_str} (improved from {old_best_wns:.3f} ns)")
                                self.best_wns = current_wns
                                self.last_improvement_iter = self.iteration  # [BETA-CTRL-V0]
                                self.last_improvement_time = time.time()  # ILS stagnation clock
                                self.regression_warning_sent = False  # [BETA-CTRL-V0.1] re-arm
                                self._pending_best_mirror = True  # mirror best_valid.{dcp,edf} next iter top
                                self._pending_best_mirror_epoch = self._mutation_epoch
                                # Capture the best-valid disk mirror
                                # immediately, before another tool call can
                                # degrade the improved in-memory state.
                                await self._mirror_best_valid_now(eager=True)
                            else:
                                logger.info(f"Current WNS (clock: {self.target_clock}): {current_wns:.3f} ns{fmax_str} (best is still {self.best_wns:.3f} ns)")
                                # Mid-iteration regressions are observational
                                # only; they do not stop exploration.
                                pass
                        except Exception as e:
                            # Best-tracking machinery failed AFTER a good
                            # measurement — swallow (as before) but KEEP the
                            # clock; the next report re-measures normally.
                            logger.warning(
                                f"clock-WNS best-tracking raised (ignored; "
                                f"target_clock kept): {e!r}")

                if not self.target_clock or wns_measured is None:
                    timing_info = parse_timing_summary_static(result_text)
                    if timing_info["wns"] is not None:
                        current_wns = timing_info["wns"]
                        wns_measured = current_wns
                        current_fmax = self.calculate_fmax(current_wns, self.clock_period)
                        fmax_str = f", fmax: {current_fmax:.2f} MHz" if current_fmax is not None else ""
                        if current_wns > self.best_wns and not await self._routed_ok_for_best():
                            logger.warning(f"WNS {current_wns:.3f} beats best but design is NOT fully routed (estimated timing) — REJECTED as best (phantom-best guard).")
                        elif current_wns > self.best_wns:
                            old_best_wns = self.best_wns
                            logger.info(f"New best WNS: {current_wns:.3f} ns{fmax_str} (improved from {old_best_wns:.3f} ns)")
                            self.best_wns = current_wns
                            self.last_improvement_iter = self.iteration  # [BETA-CTRL-V0]
                            self.last_improvement_time = time.time()  # ILS stagnation clock
                            self.regression_warning_sent = False  # [BETA-CTRL-V0.1] re-arm
                            self._pending_best_mirror = True  # mirror best_valid.{dcp,edf} next iter top
                            self._pending_best_mirror_epoch = self._mutation_epoch
                            # [mirror-fix] Eager-capture (see
                            # docstring of _mirror_best_valid_now).
                            await self._mirror_best_valid_now(eager=True)
                        else:
                            logger.info(f"Current WNS: {current_wns:.3f} ns{fmax_str} (best is still {self.best_wns:.3f} ns)")
                            # Regression detection intentionally not run here;
                            # tracking only.
            
            # Also track WNS from get_wns tool (returns just the numeric WNS value)
            elif tool_name == "vivado_get_wns":
                try:
                    current_wns = float(result_text.strip())
                    wns_measured = current_wns
                    current_fmax = self.calculate_fmax(current_wns, self.clock_period)
                    fmax_str = f", fmax: {current_fmax:.2f} MHz" if current_fmax is not None else ""
                    if current_wns > self.best_wns and not await self._routed_ok_for_best():
                        logger.warning(f"WNS {current_wns:.3f} beats best but design is NOT fully routed (estimated timing) — REJECTED as best (phantom-best guard).")
                    elif current_wns > self.best_wns:
                        old_best_wns = self.best_wns
                        logger.info(f"New best WNS (from get_wns): {current_wns:.3f} ns{fmax_str} (improved from {old_best_wns:.3f} ns)")
                        self.best_wns = current_wns
                        self.last_improvement_iter = self.iteration  # [BETA-CTRL-V0]
                        self.last_improvement_time = time.time()  # ILS stagnation clock
                        self.regression_warning_sent = False  # [BETA-CTRL-V0.1] re-arm
                        self._pending_best_mirror = True  # mirror best_valid.{dcp,edf} next iter top
                        self._pending_best_mirror_epoch = self._mutation_epoch
                        # [mirror-fix] Eager-capture (see
                        # docstring of _mirror_best_valid_now).
                        await self._mirror_best_valid_now(eager=True)
                    else:
                        logger.info(f"Current WNS (from get_wns): {current_wns:.3f} ns{fmax_str} (best is still {self.best_wns:.3f} ns)")
                        # Regression detection intentionally not run here;
                        # tracking only.
                except (ValueError, AttributeError):
                    logger.warning(f"Could not parse WNS from get_wns output: {result_text[:100]}")
            
            elapsed_time = time.time() - start_time
            # Feed runtime history → _estimate_tool_runtime for future
            # deadline-aware gating decisions.
            self._record_tool_runtime(tool_name, elapsed_time)
            # Wall economics: size the observation window from COMPLETED
            # heavy moves only (see _heavy_move_seconds).
            if elapsed_time >= 1.0 and self._is_risky(tool_name, arguments):
                self._heavy_move_seconds.append(elapsed_time)
                if len(self._heavy_move_seconds) > 20:
                    del self._heavy_move_seconds[:len(self._heavy_move_seconds) - 20]

            # Reuse successful checkpoint and EDIF writes to update best-valid mirrors.
            # This captures the exact saved state before later in-memory mutations and
            # avoids an additional tool round trip.
            if tool_name == "vivado_write_checkpoint" and self._pending_best_mirror:
                self._piggyback_mirror_checkpoint(arguments)
            # Same piggyback for write_edif so the EDIF mirror tracks the
            # DCP mirror.  Cheap: just a shutil.copy when the EDIF target
            # path matches conventional naming.
            if tool_name == "vivado_write_edif" and self._best_valid_dcp is not None:
                self._piggyback_mirror_edif(arguments)

            # Keep routed-state tracking synchronized after every successful mutating
            # operation, including finalization and ILS. Tracking remains active when
            # the unroute gate is disabled; only refusal behavior is controlled by it.
            if not _looks_like_tool_error(result_text):
                _new_routed = _routed_state_transition(tool_name, arguments)
                # Any successful design mutation invalidates the assumption that
                # the written checkpoint still represents the measured best state.
                _cmd_low_mut = _tool_cmd_text(tool_name, arguments)
                if (_new_routed is not None
                        or tool_name in ("vivado_phys_opt_design",
                                         "vivado_opt_design")
                        or re.search(r"\b(phys_opt_design|opt_design)\b",
                                     _cmd_low_mut)):
                    self._mutation_epoch += 1
                if _new_routed is not None:
                    if _new_routed != self._design_routed_state:
                        logger.info(
                            f"[unroute-gate] routed-state "
                            f"{self._design_routed_state} -> {_new_routed} "
                            f"after {tool_name}")
                    self._design_routed_state = _new_routed

            # Record command details so downstream logic can classify operations
            # embedded in Tcl, including placement and routing work not exposed
            # as granular tool calls.
            self.tool_call_details.append({
                "tool_name": tool_name,
                "iteration": self.iteration,
                "elapsed_time": elapsed_time,
                "wns": wns_measured,
                "error": False,
                "cmd_head": str(arguments.get("command", ""))[:200]
                            if isinstance(arguments, dict) else "",
            })

            # Track each router step and directive so stop handling can detect
            # plan-recommended heavy operations that have not been attempted.
            try:
                directive = ""
                if isinstance(arguments, dict):
                    d = arguments.get("directive")
                    if d is not None:
                        directive = str(d)
                self._tool_calls_seen.append((tool_name, directive))
            except Exception:
                pass

            # Measure successful heavy mutations immediately so routed improvements
            # can be banked without waiting for an explicit measurement call.
            # Measurement tools are excluded to avoid duplicate work; internal
            # timing, status, and write operations are non-risky and cannot recurse.
            if (self._auto_bank_enabled
                    and not self._in_finalize
                    and not self._in_ils_stage
                    # Tail-controller phys_opt bracket: the hook has no
                    # hold gate; those moves bank via the controller's
                    # own hold-gated accept (see __init__ comment).
                    and not self._tail_ctrl_suppress_autobank
                    and tool_name not in ("vivado_report_timing_summary",
                                          "vivado_get_wns")
                    and self._is_risky(tool_name, arguments)
                    and not _looks_like_tool_error(result_text)):
                try:
                    if (self._budget_deadline is not None
                            and self._budget_remaining()
                            < MIN_USEFUL_TOOL_SECONDS):
                        logger.info(
                            "[auto-bank] skip: %.0fs remaining below %.0fs "
                            "measurement floor after %s — not measuring.",
                            self._budget_remaining(),
                            MIN_USEFUL_TOOL_SECONDS, tool_name)
                    else:
                        # Apply an explicit 600-second outer measurement limit
                        # rather than relying on the server default; this
                        # matches the safety cap for the largest designs and
                        # retains the inner deadline cap.
                        auto_wns = await asyncio.wait_for(
                            self.get_wns_for_target_clock(
                                self._call_vivado_tool),
                            timeout=600.0)
                        if auto_wns is not None and auto_wns > self.best_wns:
                            if not await self._routed_ok_for_best():
                                logger.warning(
                                    f"[auto-bank] WNS {auto_wns:.3f} after "
                                    f"{tool_name} beats best but design is "
                                    "NOT fully routed — REJECTED as best "
                                    "(phantom-best guard).")
                                _bank_ok = False
                            else:
                                _bank_ok = True
                                # On a WNS improvement, measure worst hold
                                # slack (WHS, ns) before banking to avoid
                                # preserving a hold-violating state. Reject
                                # measured WHS below zero; if hold cannot be
                                # measured, fail open and allow the bank.
                                try:
                                    from optimizer.ils_polish import (
                                        _measure_hold as _ab_measure_hold)
                                    # The hold probe supplies complete tool
                                    # names, so it must dispatch through
                                    # call_tool without adding a name prefix.
                                    # Measurement commands are non-risky and
                                    # cannot recurse.
                                    _bank_whs = await asyncio.wait_for(
                                        _ab_measure_hold(
                                            self.call_tool,
                                            timeout_s=120.0),
                                        timeout=180.0)
                                except Exception:
                                    _bank_whs = None
                                if _bank_whs is None:
                                    # Hold-measurement failures remain fail-
                                    # open, but are counted so repeated
                                    # unverified banks remain visible in
                                    # diagnostics.
                                    self._autobank_hold_failopen_count = (
                                        getattr(self,
                                                "_autobank_hold_failopen_count",
                                                0) + 1)
                                    logger.warning(
                                        "[auto-bank] HOLD UNMEASURABLE on "
                                        "improvement event #%d — banking "
                                        "FAIL-OPEN (whs unknown; validator "
                                        "gates hold_passed).",
                                        self._autobank_hold_failopen_count)
                                if (_bank_whs is not None
                                        and _bank_whs < 0.0):
                                    _bank_ok = False
                                    logger.warning(
                                        f"[auto-bank] WNS {auto_wns:.3f} "
                                        f"after {tool_name} beats best but "
                                        f"HOLD is dirty (whs={_bank_whs:.3f}"
                                        " < 0 — validator gates hold_passed)"
                                        " — REJECTED as best (hold guard, "
                                        "out-of-distribution stress finding).")
                            if _bank_ok and self._input_cell_count:
                                # Routed-state tracking misses destructive
                                # logic edits that can improve WNS while
                                # breaking logical equivalence. Reject only
                                # measured cell counts outside 0.5x-3x the
                                # phase-entry count; unavailable counts fail
                                # open. The band exceeds valid changes.
                                try:
                                    _cc_r = await asyncio.wait_for(
                                        self.call_tool(
                                            "vivado_run_tcl",
                                            {"command":
                                             "llength [get_cells -quiet "
                                             "-hierarchical -filter "
                                             "{IS_PRIMITIVE}]",
                                             "timeout": 120.0}),
                                        timeout=180.0)
                                    # Detect tool error envelopes before
                                    # parsing integers because budget and
                                    # timing fields can resemble cell counts.
                                    # Such responses are treated as unavailable
                                    # data and fail open.
                                    if _looks_like_tool_error(_cc_r):
                                        raise RuntimeError(
                                            f"cell-count query returned error "
                                            f"envelope: {str(_cc_r)[:120]}")
                                    _cc_m = re.search(r"(\d+)",
                                                      str(_cc_r) or "")
                                    _cc = (int(_cc_m.group(1))
                                           if _cc_m else None)
                                except Exception:
                                    _cc = None
                                if (_cc is not None
                                        and not (0.5 * self._input_cell_count
                                                 <= _cc
                                                 <= 3.0 * self._input_cell_count)):
                                    _bank_ok = False
                                    logger.warning(
                                        f"[auto-bank] WNS {auto_wns:.3f} "
                                        f"after {tool_name} beats best but "
                                        f"cell count {_cc:,} is outside "
                                        f"[0.5x, 3x] of entry "
                                        f"{self._input_cell_count:,} — "
                                        "REJECTED as best (logic-deletion "
                                        "guard).")
                            if _bank_ok:
                                old_best_wns = self.best_wns
                                logger.info(
                                    f"[auto-bank] New best WNS after "
                                    f"{tool_name}: {auto_wns:.3f} ns "
                                    f"(improved from {old_best_wns:.3f} ns) "
                                    "— banking best_valid now.")
                                self.best_wns = auto_wns
                                self.last_improvement_iter = self.iteration
                                self.last_improvement_time = time.time()
                                self.regression_warning_sent = False
                                self._pending_best_mirror = True
                                self._pending_best_mirror_epoch = self._mutation_epoch
                                await self._mirror_best_valid_now(eager=True)
                        elif auto_wns is not None:
                            logger.info(
                                f"[auto-bank] WNS after {tool_name}: "
                                f"{auto_wns:.3f} ns (best is still "
                                f"{self.best_wns:.3f} ns) — no bank.")
                except Exception as e:
                    # Banking must never crash the run; fail toward
                    # not-banking.
                    logger.warning(
                        f"[auto-bank] hook failed after {tool_name} "
                        f"(non-fatal, not banking): {e}")

            # Dispatcher failures may be returned as JSON strings rather than
            # raised. Classify every payload and combine that result with
            # actual error status: tool_call_success/tool_call_warning indicate
            # no actual error without/with a match;
            # tool_call_error/tool_call_classified_error indicate an error
            # without/with a match.
            tool_err = self._classify_tool_payload(
                result_text, context="call_tool"
            )
            is_real_error = _looks_like_tool_error(result_text)
            if is_real_error and tool_err is not None:
                action_label = "tool_call_classified_error"
            elif is_real_error:
                action_label = "tool_call_error"
            elif tool_err is not None:
                action_label = "tool_call_warning"
            else:
                action_label = "tool_call_success"
            wns_after_call = (
                float(self.best_wns)
                if (self.best_wns is not None and self.best_wns != float("-inf"))
                else None
            )
            delta_wns = (
                (wns_after_call - wns_before_call)
                if (wns_after_call is not None and wns_before_call is not None)
                else None
            )
            self._emit_decision({
                "decision_source": "executor",
                "phase": "call_tool",
                "tool_name": tool_name,
                "action_label": action_label,
                "wns_before": wns_before_call,
                "wns_after": wns_after_call,
                "delta_wns": delta_wns,
                "runtime_s": elapsed_time,
                "tool_error_code": tool_err.code if tool_err is not None else None,
                "notes": (
                    f"args_keys={list(arguments.keys()) if isinstance(arguments, dict) else []} "
                    f"directive={directive if 'directive' in locals() else ''}"
                ),
            })

            return result_text

        except Exception as e:
            error_occurred = True
            elapsed_time = time.time() - start_time

            # Record failed tool call
            self.tool_call_details.append({
                "tool_name": tool_name,
                "iteration": self.iteration,
                "elapsed_time": elapsed_time,
                "wns": None,
                "error": True,
                "error_message": str(e)
            })

            # Classify the exception message +
            # emit trace event for raised-exception path.
            err_payload = json.dumps({"error": str(e)})
            tool_err = self._classify_tool_payload(
                err_payload, context="call_tool_exception"
            )
            self._emit_decision({
                "decision_source": "executor",
                "phase": "call_tool",
                "tool_name": tool_name,
                "action_label": "tool_call_exception",
                "wns_before": wns_before_call,
                "runtime_s": elapsed_time,
                "tool_error_code": tool_err.code if tool_err is not None else "UNKNOWN_TOOL_ERROR",
                "notes": f"exception={str(e)[:160]}",
            })

            logger.error(f"Tool call failed: {e}")
            return err_payload
