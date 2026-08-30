#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (C) 2026, Georgios Chatzitsompanis.
# Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

"""
FPGA Design Optimization Agent — the orchestrator.

An autonomous AI agent that analyzes FPGA designs and applies optimizations
using RapidWright and Vivado via MCP servers.

This file holds the run state every stage shares and the loop that
sequences them.  DCPOptimizer composes four stage mixins, each in its own
module under optimizer/:

  RecipePassesMixin   optimizer/recipe_passes.py   deterministic |WNS|-band
                                                   tool sequences
  PolishLadderMixin   optimizer/polish_ladder.py   ILS ruin-and-rebuild, the
                                                   polish stages, the exit tail
  FinalizationMixin   optimizer/finalization.py    the candidate play-off,
                                                   mirrors, lifecycle status
  ToolDispatchMixin   optimizer/tool_dispatch.py   call_tool, the budget and
                                                   unroute gates, auto-bank

What is still HERE, deliberately: Phase-1 orchestration and the LLM
completion path.  Both are interleaved with session calls and the wall-cap
ledger on self, so moving them would have been a behavior risk rather than
a re-arrangement.  Pure mechanisms live in their own modules too —
finalize_mux (ship integrity, md5 trust), llm_runtime (API-error
classification, the prompt-size guard, the cost ledger), phase1_sense
(measurement math and report parsing), config_resolution, qor_parsing,
tool_source — alongside recipe_router, ils_polish, wall_economics,
deep_replace_sibling and the rest.

Every extraction was verbatim, with thin delegating methods and
re-exports left under the historical names, so call sites, subclass
overrides, monkeypatch targets and test imports are unchanged.
docs/PROVENANCE.md records what differs from the scored tree.

READING ORDER, which is the pipeline order:
  1. Phase-1 sense       — measurement + feature extraction, no LLM (here)
  2. Recipe passes       — optimizer/recipe_passes.py
  3. The LLM tool loop   — here, dispatching through optimizer/tool_dispatch.py
  4. Polish ladder       — optimizer/polish_ladder.py
  5. Finalize MUX        — optimizer/finalization.py
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

# Optional RAG seed (strategy memory).  Soft-imported so a missing
# `optimizer/` package never breaks the optimizer.
try:
    from optimizer.strategy_memory import (
        RunRecord as _RunRecord,
        append_run as _append_run,
        seed_prompt_for as _seed_prompt_for,
        winning_tools_from_call_details as _winning_tools_from_call_details,
        negative_memory_block as _negative_memory_block,
        retrieval_metadata_for as _retrieval_metadata_for,
    )
except Exception:  # pragma: no cover — defensive
    _RunRecord = None
    _append_run = None
    _seed_prompt_for = None
    _winning_tools_from_call_details = None
    _negative_memory_block = None
    _retrieval_metadata_for = None

# API resilience: pure error classifier + deadline-aware
# backoff scheduler.  Same soft-import pattern; when unavailable the call
# path falls back to the legacy behavior (see
# _create_completion_with_fallback).
try:
    from optimizer.api_resilience import (
        classify_api_error as _classify_api_error,
        compute_backoff_sleep as _compute_backoff_sleep,
    )
except Exception:  # pragma: no cover — defensive
    _classify_api_error = None
    _compute_backoff_sleep = None

# Optional dispatch / recipe gates.  Same soft-import pattern.
try:
    from scheduler.dispatch import design_name_from_dcp as _design_name_from_dcp
    from scheduler.dispatch import recipe_safe_for as _recipe_safe_for
except Exception:  # pragma: no cover — defensive
    _design_name_from_dcp = None
    _recipe_safe_for = None

# Soft-import pathology classifier.  If unavailable (e.g., stripped
# package), runtime falls back to the pre-classifier prompt structure.
try:
    from optimizer.pathology import (
        classify_design as _classify_design,
        HIGH_FANOUT_DRIVER as _PATH_HIGH_FANOUT,
        CELL_SPREAD as _PATH_CELL_SPREAD,
        NO_CLEAR_LOCAL_RECIPE as _PATH_NO_CLEAR,
    )
except Exception:  # pragma: no cover — defensive
    _classify_design = None
    _PATH_HIGH_FANOUT = None
    _PATH_CELL_SPREAD = None
    _PATH_NO_CLEAR = None

# Soft-import cross-model steering (post-grok-4.3-migration recovery).
# Hints are gated by env var ENABLE_CROSS_MODEL_STEERING=1 inside the
# module, so this is safe to import unconditionally.
try:
    from optimizer.cross_model_steering import (
        get_iter1_hint as _steering_iter1_hint,
        get_continue_hint as _steering_continue_hint,
        is_steering_enabled as _steering_enabled,
    )
except Exception:  # pragma: no cover — defensive
    _steering_iter1_hint = lambda _: None
    _steering_continue_hint = lambda *_a, **_kw: None
    _steering_enabled = lambda: False

# Soft-import Phase-1 utilization parsing.  Pure functions; the fallbacks
# yield 'no data', so a missing module simply means the resource-keyed rules
# cannot fire.  optimizer/utilization_features.py was not part of the scored
# submission, so those rules ran on their fallbacks in the scored run; the
# soft-import is kept as-is to preserve that behaviour, and
# tests/test_import_completeness.py pins which soft-imports resolve.
try:
    from optimizer.utilization_features import (
        parse_utilization,
        memory_dominated,
        parse_route_status,
        split_reports,
    )
except Exception:  # pragma: no cover — defensive
    parse_utilization = lambda _t: {}
    memory_dominated = lambda _u, **_kw: None
    parse_route_status = lambda _t: None
    split_reports = lambda _t: (_t, None)

# Soft-import feature-based recipe router. This is a deterministic
# layer on top of the pathology classifier — it takes Phase-1
# measurables and routes to a calibrated recipe sequence (rules R1..R4
# validated on the 4 known designs). Always on (no env gate); falls
# back silently when features are missing or no rule fires confidently.
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

# Decision tracer for per-event/per-iteration JSONL
# telemetry.  Stays optional — if the module is missing the optimizer
# still works.  Wired lazily inside DCPOptimizer.__init__.
try:
    from optimizer.decision_tracer import (
        DecisionTracer as _DecisionTracer,
    )
except Exception:  # pragma: no cover — defensive
    _DecisionTracer = None

# Structured ToolError taxonomy — log-only first phase.
# classify_tool_error converts call_tool / finalize payloads into a
# typed envelope.  Existing _looks_like_tool_error stays the gate for
# behavior; ToolError is used for observability + decision-trace
# records only in this phase.
try:
    from optimizer.tool_errors import (
        ToolError as _ToolError,
        classify_tool_error as _classify_tool_error,
    )
except Exception:  # pragma: no cover — defensive
    _ToolError = None
    _classify_tool_error = None

# DCP write-path PathGuard — audit-mode by default.
# Wired lazily in optimize() once run_dir + output_dcp are known.  Calls
# in DCP write/copy paths invoke .check() but never block (audit mode).
try:
    from optimizer.path_guard import PathGuard as _PathGuard
except Exception:  # pragma: no cover — defensive
    _PathGuard = None

# Static report parsers.  A hard import, unlike the soft helpers above: five
# WNS-read call sites depend on these, so a None fallback would raise
# mid-run rather than here.  The re-export is part of the contract, not
# style — the names must stay module globals of dcp_optimizer and call sites
# must use the bare name, so that patching dcp_optimizer.<name> in tests
# still intercepts every call.
from optimizer.static_parsers import (
    parse_timing_summary_static as parse_timing_summary_static,
)

# Typed gate ledger. Default OFF via
# FPL26_GATE_LOG; `emit()` swallows every exception by design, so no call site needs a
# guard and telemetry can never alter a decision. See optimizer/gate_log.py.
from optimizer import gate_log as _gl


# Ship-integrity and finalize-MUX mechanisms (atomic artifact writes,
# checksum-truth identity, md5-trust digests).  Re-exported under their
# original names for call sites and the test suite.
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

# LLM runtime mechanisms: API-error classification, prompt-size guard,
# per-request timeout, in-attempt cost exit, crash-safe cost ledger.
# DCPOptimizer keeps thin delegating methods under the original names.
from optimizer import llm_runtime as _llm_rt
from optimizer.llm_runtime import (
    LLM_COST_EXIT_USD,
    _llm_timeout_s,
    resolve_llm_cost_exit,
)

# Phase-1 sensing mechanisms (fmax math, high-fanout-nets report parsing,
# Phase-1 wall-cap resolution) were extracted verbatim to
# optimizer/phase1_sense.py.  DCPOptimizerBase keeps thin delegating
# methods (calculate_fmax, parse_high_fanout_nets); the module-level
# names below are re-exported unchanged.  Plain import (stdlib-only).
from optimizer import phase1_sense as _p1_sense
from optimizer.phase1_sense import (
    PHASE1_WALL_FRAC_DEFAULT,
    resolve_phase1_wall_frac,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)

# Default model
# Migrated after x-ai/grok-4.1-fast was deprecated by xAI/OpenRouter (HTTP 404).
# The historical +79.92 MHz portfolio (four of the large designs) was
# earned under grok-4.1-fast and is treated as
# pre-migration evidence; grok-4.3 results were re-validated separately.
DEFAULT_MODEL = "x-ai/grok-4.3"

# Robustness fallback: if the primary model is unavailable on the contest's
# OpenRouter key (404 / deprecated / no-endpoints / model-scoped key), fall back
# to the organizer-blessed gemini model so a model outage can't zero ALL 13
# benchmarks (alpha 0.0).  grok-4.3 stays primary (current flagship; ~+9 MHz
# better than gemini per an A/B comparison); gemini is the safety net only.
FALLBACK_MODEL = "google/gemini-3.1-flash-lite"

# _llm_timeout_s (per-request SDK timeout, FPL26_LLM_TIMEOUT_S override)
# lives in optimizer/llm_runtime.py, imported above.

# The plan critic's default model is derived from FALLBACK_MODEL rather than
# written out again: one literal, one place to change, and a literal the API
# is already known to accept because the fallback path uses it.
PLAN_CRITIC_DEFAULT_MODEL = FALLBACK_MODEL
# Transient-API resilience (eval-day insurance): one short-backoff retry per
# call; after this many CONSECUTIVE transient failures, pin FALLBACK_MODEL.
TRANSIENT_RETRY_BACKOFF_S = 2.0
TRANSIENT_FALLBACK_THRESHOLD = 3
# Bound on consecutive API-error episodes that reach the optimize() loop.
# Each propagated episode has already exhausted its in-call backoff cap
# (~600 s of absorbed wall), so this many in a row means the LLM path is
# gone for good.  Failed calls no longer burn iterations, and tests may run
# with an infinite wall, so this bound is what guarantees the loop
# terminates into the finalize tail and ships whatever Phase 1 banked.
PERMANENT_API_FAILURE_LIMIT = 3

# --- Prompt-size guard ----------------------------------------------------
# A provisioned API key can enforce a per-request prompt-token limit.  Once
# a conversation crosses it every later call fails, and an error handler
# that appends the failure text to the conversation grows the prompt on each
# retry — a doom loop the run never escapes.  Guard both ways: prune the
# conversation before a call when the estimated prompt exceeds the soft
# limit, and prune-and-retry on a prompt-limit rejection.  Estimates use
# chars/3, which is conservative for JSON-heavy tool output, and the tool
# schema rides on top of every request — hence the wide margin.
PROMPT_TOKEN_SOFT_LIMIT = 45_000     # estimated tokens that trigger a prune
PROMPT_PRUNE_TARGET = 30_000         # estimated tokens to prune down to
PROMPT_PRUNE_HEAD_KEEP = 2           # always keep: system + initial analysis


# Extracted for readability; re-exported so every name below still resolves as
# a module global of dcp_optimizer -- call sites, monkeypatching in tests and
# `from dcp_optimizer import ...` all behave exactly as before.
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
    resolve_preempt_loop_clock_enabled,
    resolve_tail_controller_enabled,
    resolve_tail_ctrl_deep_wns_ns,
    resolve_tail_ctrl_m1_echo,
    resolve_tail_ctrl_max_moves,
    resolve_tail_reserve_stagnant_s,
)
from optimizer.qor_parsing import (  # noqa: F401
    parse_qor_assessment_static,
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


# ---------------------------------------------------------------------------
# Stage mechanisms live in the optimizer/ package (see optimizer/README.md for
# the map).  They are re-exported here so that `dcp_optimizer.<name>` remains
# the public name for every one of them: call sites, `from dcp_optimizer import
# ...` and the test suite's monkeypatch targets are unchanged by the split.
# The per-import suppression markers below mark those deliberate
# re-exports.  (Spelled out rather than quoted, because a linter reads
# this comment as a directive and warns that the prose is malformed --
# the same trap as a docstring that mints a flag name.)
# ---------------------------------------------------------------------------

# Seconds of wall handed to the deterministic exit tail instead of the LLM
# loop on deep-WNS designs.  2400 s is about two tail moves at evaluation
# speed; the ship path sets it explicitly and the module default stays off.
DEEP_WNS_TAIL_RESERVE_S_RECOMMENDED = 2400.0

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
from optimizer.polish_ladder import PolishLadderMixin, MIDBAND_RETRY_HOLD_WNS_MAG_MIN_NS, TAIL_RESERVE_WALL_CLAMP_FRAC  # noqa: F401
from optimizer.finalization import FinalizationMixin, FINAL_CANDIDATE_MUX_MIN_GAIN_NS  # noqa: F401
from optimizer.recipe_passes import (  # noqa: F401
    RecipePassesMixin,
    FRESH_PRESWEEP_BUDGET_FRAC,
    FRESH_PRESWEEP_FALLBACK_BUDGET_S,
    FRESH_PRESWEEP_FALLBACK_FIRST_TIMEOUT_S,
    FRESH_PRESWEEP_FIRST_DRAW_FRAC,
    FRESH_PRESWEEP_MIN_WALL_S,
    FRESH_PRESWEEP_PIPELINE_REFERENCE_S,
    RECIPE_FIRST_DEEP_ANCHOR_FLOOR_S,
    RECIPE_FIRST_DEEP_COST_RATIO,
    RECIPE_FIRST_DEEP_DYNAMIC_ABORT_FRAC,
    RECIPE_FIRST_DEEP_LLM_FLOOR_FRAC,
    RECIPE_FIRST_DEEP_MARGIN,
    RECIPE_PASS_DEEP_EXPECTED_S,
    RECIPE_PASS_DEEP_TCL,
    RECIPE_PASS_FINALIZE_RESERVE_S,
    RECIPE_PASS_MEASURE_RESERVE_S,
    RECIPE_PASS_MID_EXPECTED_S,
    RECIPE_PASS_MID_TCL,
    RECIPE_PASS_OVERHEAD_RESERVE_S,
    RECIPE_PASS_POSTLOOP_OVERHEAD_RESERVE_S,
    RECIPE_PASS_REGISTER_RESERVE_S,
    RECIPE_PASS_RESET_RESERVE_S,
    RECIPE_PASS_SESSION_ROUNDTRIP,
    RECIPE_PASS_SHALLOW_EXPECTED_S,
    RECIPE_PASS_SHALLOW_TCL,
    RECIPE_PASS_STORE_RESERVE_S,
    RECIPE_PASS_TIMEOUT_FACTOR,
    logger,
)
from optimizer.tool_dispatch import ToolDispatchMixin, FINALIZE_PER_CALL_TIMEOUT_S, MIN_USEFUL_TOOL_SECONDS  # noqa: F401


# --- Frozen recipe chains --------------------------------------------------
#
# Each chain below is a fixed list of Tcl steps, frozen verbatim from the run
# that measured it, and registered as a finalize-MUX candidate: the MUX takes
# the argmax over the pipeline result and every candidate, so a candidate can
# only ever add.  Chains are selected from measured features of the design
# (|WNS| at entry, cell count, failing-endpoint count) and never from a design
# name or a checkpoint hash, so an unseen design is treated like any other of
# its shape.  Each EXPECTED_S is the measured chain wall; the fire gate
# reserves it plus the fixed measure/store/register overheads and refuses,
# fail-closed and logged, when the remaining wall is short.
#
# Chains that start from a placed front carry an `unroute` + `unplace` prefix:
# `place_design -directive` cannot run on the pristine routed input without it.

# Sub-band phys_opt floor: group the twenty worst endpoints into a path group,
# then phys_opt against that objective.  Banked in-session rather than through
# the MUX, because the banked floor is what makes the ILS ladder reject its
# first cycle and cascade instead of settling.
SUBBAND_FLOOR_GROUP_TCL = (
    "set worst_paths [get_timing_paths -nworst 20]; "
    "set endpoints [get_property ENDPOINT_PIN $worst_paths]; "
    "group_path -name critical_endpoints -to $endpoints")
SUBBAND_FLOOR_EXPECTED_S = 150.0   # measured: group + 3x phys_opt + measures ~90 s
SUBBAND_FLOOR_RESERVE_S = 300.0    # finalize-reserve convention


# Retime-from-placed chain.  Retiming pays only from a placed,
# timing-annotated state, so the chain places before it retimes; run from a
# bare unrouted basin the same retime step is actively harmful.
ETO_RETIME_TCL = (
    "route_design -unroute",
    "place_design -unplace",
    "place_design -directive ExtraTimingOpt",
    "phys_opt_design -retime",
    "phys_opt_design -directive AggressiveExplore",
    "route_design -directive AggressiveExplore",
)
ETO_RETIME_EXPECTED_S = 460.0      # measured chain wall ~420-460 s (place dominates)

# Latency audit.  Retiming is legal for this contest; adding pipeline stages is
# not.  The flip-flop count is probed around the retime step and registration is
# aborted if it drifts by more than 1%.  An unmeasurable count aborts too:
# this is a protection, and protections fail on.
ETO_RETIME_FF_COUNT_TCL = (
    "puts FFCOUNT=[llength [get_cells -quiet -hier "
    "-filter {PRIMITIVE_TYPE =~ REGISTER.*}]]")

# Out-of-deadline overheads every candidate pays: measure + store + register.
# No reset term — one post-pass reset covers every candidate in the pass.
ETO_RETIME_OVERHEAD_RESERVE_S = (
    RECIPE_PASS_MEASURE_RESERVE_S + RECIPE_PASS_STORE_RESERVE_S
    + RECIPE_PASS_REGISTER_RESERVE_S
)  # = 540.0


# Mid-band break rung: run from the loop's banked best state, after the LLM
# loop and every polish stage, out of wall that would otherwise be handed
# back.  Route-only variants of this rung reach a worse basin, so the retime
# and phys_opt steps are load-bearing rather than route variance.
MIDBAND_ROUTE_RUNG_TCL = "route_design -directive Explore"
MIDBAND_ROUTE_RUNG_EXPECTED_S = 500.0
# ^ The single-step form is superseded by the chain below and no longer read;
# kept because tests/test_v40_levers.py pins both names.
MIDBAND_BREAK_RUNG_TCL = (
    "route_design -unroute",
    "phys_opt_design -retime",
    "phys_opt_design -directive AggressiveExplore",
    "route_design -directive AggressiveExplore",
)
MIDBAND_BREAK_RUNG_EXPECTED_S = 810.0


# Wire-length-driven placement front.  Which of the two placed fronts a run
# selects is decided by its own measured |WNS|: retiming pays on a design's
# own winning front and transfers poorly to a foreign one, so exactly one
# retime candidate spends wall per run.
WLD_RETIME_TCL = (
    "route_design -unroute",
    "place_design -unplace",
    "place_design -directive WLDrivenBlockPlacement",
    "phys_opt_design -retime",
    "phys_opt_design -directive AggressiveExplore",
    "route_design -directive AggressiveExplore",
)
OWNFRONT_RETIME_EXPECTED_S = 520.0  # measured chain wall ~435-520 s
OWNFRONT_RETIME_OVERHEAD_RESERVE_S = (
    RECIPE_PASS_MEASURE_RESERVE_S + RECIPE_PASS_STORE_RESERVE_S
    + RECIPE_PASS_REGISTER_RESERVE_S
)  # = 540.0


# Determinizer chain for the shallowest retime sub-band.  The pipeline's own
# result on this class depends on whether an incremental-route escalation
# window is still open when the ILS ladder reaches it, which is downstream of
# LLM sampling; registering the measured state as a MUX candidate floors the
# run at that state when the window is missed, and ties (so the pipeline
# ships) when it is hit.
#
# Load-bearing invariant: no unroute between the two route steps — the second
# route is an incremental escalation of the first, and Explore must precede
# AggressiveExplore.
SHALLOW_DET_TCL = (
    "route_design -unroute",
    "place_design -unplace",
    "place_design -directive AltSpreadLogic_medium",
    "route_design -directive Explore",
    "route_design -directive AggressiveExplore",
    "phys_opt_design -directive AlternateFlowWithRetiming",
)
SHALLOW_DET_EXPECTED_S = 390.0     # measured chain wall ~390 s
SHALLOW_DET_OVERHEAD_RESERVE_S = (
    RECIPE_PASS_MEASURE_RESERVE_S + RECIPE_PASS_STORE_RESERVE_S
    + RECIPE_PASS_REGISTER_RESERVE_S
)  # = 540.0


# --- Wall-time budget enforcement constants -------------------------------
# Tools known to run for many minutes on the contest's larger designs.  The
# deadline-aware dispatcher uses this set to decide whether starting a call
# is safe given remaining wall-clock budget.  Anything not in this set is
# treated as cheap analysis/report and only the global deadline applies.
#
# Note: vivado_run_tcl is NOT in this set unconditionally.  Most uses are
# cheap property queries (get_clocks, get_property) — gating them blocks
# Phase 1 analysis on small designs.  When the tcl payload itself invokes
# a heavy implementation step, detected via RISKY_TCL_SUBSTRINGS below.
RISKY_VIVADO_TOOLS = frozenset({
    "vivado_place_design",
    "vivado_route_design",
    "vivado_phys_opt_design",
    # NOTE: rapidwright_optimize_cell_placement is intentionally NOT here.
    # Per-call cost is typically <10s (it moves one cell at a time); the
    # heavy orchestration lives in recipe_cell_replacement, which IS gated.
    "recipe_cell_replacement",
    "recipe_lut_optimization",
    "recipe_register_retiming",
    "recipe_post_route_phys_opt_sweep",
    "recipe_high_fanout_timing_replication",
    "recipe_critical_path_focused_phys_opt",
})

# When a tool is `vivado_run_tcl` the command string is inspected and treated
# as risky if it contains any of these substrings (case-insensitive).
# Anything that wraps an implementation step belongs here.
RISKY_TCL_SUBSTRINGS = (
    "place_design",
    "route_design",
    "phys_opt_design",
    "power_opt_design",
    "opt_design",
    "synth_design",
)

# Conservative initial-runtime guess (s) for a risky tool when no measured
# history exists yet.  Observed data on the two largest designs shows a single
# phys_opt/place_design taking 30–50 min — better to skip a call than
# start one that won't finish.
DEFAULT_RISKY_RUNTIME_S = 600.0

# Initial guess (s) for non-risky analysis/report tools when no history.
DEFAULT_CHEAP_RUNTIME_S = 60.0

V05_MIN_REMAINING_S = 300.0

# Soft global state used by signal handlers (SIGTERM / SIGHUP).  Set by
# main() once an Optimizer is built so the handler can copy the best known
# good DCP to the output path before the process dies.
_ACTIVE_OPTIMIZER = None
_TERMINATION_STATE = {"finalized": False, "reason": None}


class DCPOptimizerBase:
    """Base class with shared functionality for FPGA optimization."""
    
    def __init__(self, debug: bool = False, run_dir: Optional[Path] = None):
        self.debug = debug
        
        # Create run directory if not provided
        if run_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.run_dir = _run_dir_base() / f"dcp_optimizer_run-{timestamp}"
            self.run_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created run directory: {self.run_dir}")
        else:
            self.run_dir = run_dir
            self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.exit_stack = AsyncExitStack()
        self.rapidwright_session: Optional[ClientSession] = None
        self.vivado_session: Optional[ClientSession] = None

        # Optional post-finalize JSON QoR capture (default OFF).
        # Set to True by main() when --capture-qor is passed.  Lifecycle
        # behaviour does NOT depend on this flag.
        self.capture_qor: bool = False
        # Policy-card prompt variant.  Default value preserves the
        # baseline behaviour byte-for-byte.  Variants {post_route_polish_v1,
        # route_bound_v1} append a cluster-derived advisory only when the
        # candidate's features match the variant's target cluster.  No
        # design-name conditional; no command forcing.
        self.policy_card_variant: str = "default"
        # Timeout (seconds) for the post-finalize Vivado batch.
        # Default 60s preserves the original behaviour; 180s is recommended
        # for DCPs > 80 MB.  Affects only the QoR-capture subprocess; does
        # NOT touch the optimizer's wall budget or recipe selection.
        self.capture_qor_timeout: float = 60.0
        # Opt-in controller-side phys_opt preempt.  When set to an integer
        # N >= 1, the controller counts consecutive no-gain or regression
        # outcomes from the post-route phys_opt sweep and, after N of them
        # with no committed gain in between, instructs the model to run the
        # deterministic heavy rescue family instead.  None disables it.  A
        # once-per-run latch prevents repeated triggering.
        self.phys_opt_preempt_after: Optional[int] = None
        self._physopt_no_gain_streak: int = 0
        self._phys_opt_preempt_fired: bool = False
        # Preempt check mode and budget floor.  "end_sweep" checks once at
        # the end of the sweep; "mid_sweep" checks after each per-flag entry
        # and breaks the loop on a trigger, so the heavy rescue can start
        # while wall remains.  The floor is the wall the rescue needs to be
        # worth starting; lowering it accepts a tighter route margin.
        self.phys_opt_preempt_mode: str = "end_sweep"
        self.phys_opt_preempt_budget_floor_s: float = 1100.0

        # Use run directory for all temporary files
        self.temp_dir = self.run_dir
        logger.info(f"Working directory: {self.temp_dir}")
        
        # Timing tracking
        self.initial_wns = None
        self.initial_tns = None
        self.initial_failing_endpoints = None
        self.high_fanout_nets = []
        self.clock_period = None
        self.target_clock = None  # Set to clock name (e.g. "clk_fpl26contest") for clock-specific Fmax
        # Pathology diagnosis from optimizer.pathology.classify_design,
        # populated in perform_initial_analysis after Phase 1 data is in.
        # None until that runs.  Consumed by the iter-1 user message and
        # by RAG records (winning_tools + primary_label).
        self.design_pathology = None
        # Feature-based recipe router plan (rule_id, actions, blocks),
        # populated in _build_recipe_router_block after Phase 1. Available
        # to later iters + RAG records, so an audit can see which rule fired.
        self.recipe_router_plan = None
        # RapidWright critical-path spread metric, populated by Phase 1.
        # Used by the pathology classifier's CELL_SPREAD detector.
        self.critical_path_spread_info = None
        # Vivado report_qor_assessment feature signals.
        # Compact diagnosis input — NOT raw RQA text — that surfaces in
        # the Phase-1 summary the LLM sees on iter 1.  All keys None when
        # RQA was skipped or unparsed; the LLM still proceeds.
        self.qor_assessment: dict = {
            "score": None,
            "flow_guidance": None,
            "methodology_violations": None,
            "ml_strategy_available": None,
        }

        # Phase-1 robustness controls.  phase1_timeout_scale multiplies every
        # Phase-1 step's base timeout (checkpoint open, timing report,
        # high-fanout and spread analysis); values above 1.0 are for very
        # large designs whose analysis steps exceed the server-side budget.
        # phase1_skipped records which optional steps were skipped, so the
        # degradation can be surfaced to the model and to consumers.
        self.phase1_timeout_scale: float = 1.0
        self.phase1_skipped: list[str] = []
        # Cumulative Phase-1 wall cap.  Once Phase 1 has consumed this
        # fraction of the wall, remaining OPTIONAL steps are skipped with a
        # recorded reason and an optional step's scaled timeout is clamped to
        # the allowance left.  Mandatory steps are never capped.
        self.phase1_wall_frac: float = PHASE1_WALL_FRAC_DEFAULT
        self.phase1_skip_reasons: dict[str, str] = {}
        self._phase1_start_ts: Optional[float] = None

        # Wall-time budget.  The evaluation harness enforces a hard per-design
        # cap; None means no cap (development mode).  The deadline (start +
        # budget - reserve) is checked between iterations, and the controller
        # stops cleanly when exhausted and falls through to finalize, so the
        # best valid result is always emitted.
        self.max_wall_seconds: Optional[float] = None
        self._finalize_reserve_seconds: float = 300.0   # 5 min reserved
        # POST-ROUTE POLISH RESERVE.  Polish opportunities arrive at the end
        # of the wall, by which time speculative heavy operations have burned
        # the window — so the budget gate refuses the polish every time and
        # the design ships unpolished.  The window has to be reserved in
        # advance rather than hoped for.
        #
        # Reserve layering, outermost to innermost:
        #   real wall .................. the hard cap the harness enforces
        #   - _finalize_reserve_seconds  the finalize tail; _budget_deadline
        #     is start + wall - reserve, so _budget_remaining() is already
        #     finalize-protected and finalize runs inside that tail.
        #   - _polish_reserve_s ........ this reserve, inside the
        #     finalize-protected window.  While armed (a routed banked best
        #     exists, polish stages are pending, polish is not disabled)
        #     dispatch of speculative routed-state-destroying operations sees
        #     (remaining - reserve), while the polish stages themselves,
        #     finalize, and cheap banking writes all see the full window.
        #     It releases once the polish stages have run or been skipped,
        #     and unconditionally at finalize, so it never strands wall.
        #
        # Orthogonal to the cost circuit-breaker: that gates spend, this
        # gates seconds, and neither reads the other's accounting.
        self._polish_reserve_s: float = POLISH_RESERVE_S_DEFAULT
        # None = not yet released; a string records WHY it released
        # (forensics).  First release wins; releasing is one-way.
        self._polish_reserve_release_reason: Optional[str] = None
        self._budget_deadline: Optional[float] = None
        self._budget_warned_low: bool = False
        # Empty-spin guard: tracks consecutive iters where the LLM made no
        # tool calls.  Used by the v0_3 stop condition to terminate ceiling
        # designs early instead of burning the entire iter cap on done-loops.
        self._consecutive_empty_iters: int = 0
        self._strategies_skipped_budget: list[str] = []
        # Set True when the deadline-aware dispatcher hard-cancels a tool
        # call (asyncio.wait_for timeout) or skips it because the remaining
        # budget would not allow useful work.  optimize() checks this at the
        # top of each iteration and finalizes cleanly when set.
        self._budget_killed: bool = False
        # Forensic cause label for _budget_killed: set when the timeout that
        # killed a call was capped by the polish-reserve fence rather than by
        # the genuine remaining window, so a fence-capped kill can be told
        # apart from true budget exhaustion afterwards.  First cause wins.
        # No behaviour change — the exit path is identical either way.
        self._budget_kill_cause: Optional[str] = None
        # True iff the LAST _deadline_aware_timeout computation returned a
        # fence-capped window (set per computation; read only by the
        # TimeoutError handler of the same call).
        self._last_timeout_fence_capped: bool = False
        # WALL HANDBACK (default off).  Saturated runs keep burning tail wall
        # that is charged against the score, sometimes for a worse result
        # than a shorter run would have given.  _exit_early_reason composes
        # the three existing saturation signals only — the ILS no-improve
        # stop, the last-mile reject verdict, and the budget-kill skip — and
        # is armed by _maybe_arm_wall_handback, which enforces the
        # never-trim-before-a-banked-accept guard: a trimmed zero is worse
        # than a slow zero.  The polish stages still run once and must never
        # clear the reason.
        self._exit_early_reason: Optional[str] = None
        self._wall_handback_enabled: bool = False
        self._consecutive_transient_failures: int = 0
        # API RESILIENCE (default on).  A key-level authentication failure
        # reads like a model-unavailable error if it is matched by string, so
        # the client switches to the fallback model — which shares the same
        # dead key, also fails, and stays pinned for the rest of the run,
        # burning iterations in seconds against a flat retry.  When on,
        # _create_completion_with_fallback classifies the error and
        # backoff-retries authentication and transient failures on the SAME
        # model with a deadline-aware exponential schedule; a genuine
        # model-level 404 keeps the one-shot fallback.  Off restores the
        # legacy path.
        self._api_resilience_enabled: bool = True
        # Run-level observability counters (forensics must be
        # able to reconstruct episodes from the final stats block).
        self._api_error_episodes: int = 0
        self._total_backoff_s: float = 0.0
        # Per-episode backoff accumulator compared against
        # API_BACKOFF_EPISODE_CAP_S; reset whenever a call succeeds
        # (episode resolved) and at the start of a new episode.
        self._api_backoff_used_s: float = 0.0
        # Guard subtracted from _budget_remaining() before any backoff
        # sleep.  _budget_remaining() is ALREADY finalize-reserve-
        # protected (the deadline subtracts _finalize_reserve_seconds);
        # this extra 60s (2 × MIN_USEFUL_TOOL_SECONDS) leaves room for
        # one final retry + the finalize handoff after the last sleep.
        self._api_backoff_finalize_guard_s: float = 2 * MIN_USEFUL_TOOL_SECONDS
        # Injectable sleep indirection so tests never sleep for real.
        self._backoff_sleep = time.sleep
        # Conversation hygiene: the highest episode number already summarized
        # into the conversation.  At most one compact "[api-status]" summary
        # is appended per newly resolved episode, so identical error text is
        # never repeated and per-failure junk appends do not accumulate.
        self._last_summarized_episode: int = 0
        # Consecutive permanent episodes — those whose in-call backoff cap was
        # exhausted and whose exception reached the optimize() handler.  Reset
        # on any success.  At PERMANENT_API_FAILURE_LIMIT the loop breaks so
        # that a run with an infinite wall still terminates into finalize.
        self._consecutive_permanent_api_failures: int = 0
        # Failure count of the most recent episode (resolved or given
        # up) — feeds the single-line per-episode forensic log.
        self._api_last_episode_failures: int = 0
        # True for the duration of _finalize_output_dcp and its inner writes.
        # While set, call_tool bypasses the budget skip and uses a generous
        # fixed timeout instead of the deadline-derived one: the finalize path
        # must be able to run, or the run ships the baseline and loses gain
        # the iteration loop already paid for.
        self._in_finalize: bool = False
        # True only after _finalize_output_dcp ran to completion — the
        # SIGTERM emergency handler uses it to tell a finalized output
        # from a mid-run LLM write.
        self._finalize_completed: bool = False
        # corrupt_output guard: identity (path/size/md5) of the
        # artifact _finalize_output_dcp actually shipped, recorded the moment
        # finalize completes.  The emergency handler verifies the on-disk
        # output against this before trusting it — an existing file with the
        # wrong digest is externally-injected garbage and must be restored.
        self._shipped_artifact: Optional[dict] = None
        # Size of best_valid.dcp recorded at bank time (mirror/piggyback).
        # Cheap integrity gate for the bank_mirror-truncation fault variant:
        # a mirror whose on-disk size no longer matches its banked size is
        # corrupt and must not be shipped.  (Full mirror md5 deferred — banks
        # fire per-improvement and hashing 100-300MB each time is real wall.)
        self._best_valid_mirror_size: Optional[int] = None
        # AUTO-BANK AFTER HEAVY OP (default on).  A heavy operation can
        # succeed and still be lost: if the model skips the measurement call
        # the recipe asked for and writes no checkpoint, a later destructive
        # step can throw the gain away and finalize ships the baseline.  When
        # on, call_tool measures the scored-clock WNS after every heavy
        # mutating operation that returns OK — a cheap slack query, not a full
        # timing report — and banks best_valid through the existing eager
        # mirror path when the result is both improved and routed.
        self._auto_bank_enabled: bool = True
        # UNBANKABLE-SEQUENCE GATE (default on).  Unrouting is cheap and sails
        # through a pre-flight budget check, but it commits the run to a full
        # re-route that may not fit the remaining wall — and a design left
        # unrouted scores nothing.  When on, call_tool refuses
        # routed-state-destroying operations whose predicted follow-up
        # re-route plus banking cannot fit (see optimizer.route_gate, which
        # scales its prediction with design size) and returns an error
        # envelope steering the model to the bankable incremental phys_opt
        # ladder instead.
        self._unroute_gate_enabled: bool = True
        # BARE RE-ROUTE POLISH (default on).  An undirected rip-up and repair
        # beats a directed one on already-good routing.  The ILS route-only
        # combo already harvests that where ILS runs, but designs above the
        # ILS cell-count gate never reach it.  When on, the exit tail runs one
        # bare route_design — no directive, no unroute — on the routed banked
        # best, for exactly the case where ILS never ran, a routed banked best
        # exists, and the wall fits.  Never-worse via the normal auto-bank and
        # phantom-guard path.  Kill switches: --no-bare-reroute-polish or
        # FPL26_NO_BARE_REROUTE_POLISH=1.
        self._bare_reroute_polish_enabled: bool = True
        # Iteration loop knobs.  The rip-up re-roll compounds while slack is
        # deeply negative and decays near a plateau, so the loop continues
        # while each pass gains at least min-gain, the next pass still fits
        # the wall, and the iteration count is under max — the last being a
        # runaway guard that the decay normally reaches first.
        self._bare_reroute_min_gain_ns: float = BARE_REROUTE_MIN_GAIN_NS_DEFAULT
        self._bare_reroute_max_iters: int = BARE_REROUTE_MAX_ITERS_DEFAULT
        # Adaptive banked tail controller: on deeply-negative-slack states the
        # exit tail replaces the plain repeat loop with a measured ns/s
        # portfolio policy over the proven move menu (optimizer/
        # tail_controller.py).  Fail-closed — any controller error falls back
        # to the plain loop.  _tail_ctrl_suppress_autobank brackets the
        # controller's phys_opt moves only: the auto-bank hook does not check
        # hold, and banking a hold-dirty phys_opt result would fail the
        # validator outright, so the controller banks those through its own
        # hold-gated accept.  The bare-route move keeps riding the hook, being
        # hold-neutral.
        self._tail_controller_enabled: bool = True
        self._tail_ctrl_deep_wns_ns: float = TAIL_CTRL_DEEP_WNS_NS_DEFAULT
        self._tail_ctrl_max_moves: int = TAIL_CTRL_MAX_MOVES_DEFAULT
        self._tail_ctrl_suppress_autobank: bool = False
        # Double-fire guard for the post-loop recipe slot: set the moment the
        # pre-LLM slot logs that it fired — before its first session mutation —
        # so the post-loop slot can never run the same recipe twice in one
        # wall, even if the pre-LLM pass raised after firing.  Marking at the
        # fired log is race-free: nothing mutates the session between that line
        # and the pristine preopen.
        self._recipe_pass_preloop_fired: bool = False
        # Deep-band recipe-first state.  All three stay inert unless the deep
        # gate fires, so flag-off behaviour is untouched:
        #   _recipe_first_deep_fired: the pre-LLM deep fire passed the
        #     size-anchored gate.  Set at the fired log, so the post-loop
        #     already-fired guard covers it unchanged.
        #   _recipe_first_deep_banked: that fire actually registered a
        #     candidate.  The reduced-tail step keys on this and never on
        #     fired alone, so a miss keeps the full deterministic tail.
        #   _recipe_first_deep_tail_reserve_cap_s: post-fire cap on the tail
        #     reserve; None means no cap.
        #   _recipe_first_deep_gate_llm_floor_s: the scaled floor the fire
        #     gate actually reserved.  The reduced-tail step honours this so
        #     the loop keeps what the gate promised, not the smaller base.
        self._recipe_first_deep_fired: bool = False
        self._recipe_first_deep_banked: bool = False
        self._recipe_first_deep_tail_reserve_cap_s: Optional[float] = None
        self._recipe_first_deep_gate_llm_floor_s: Optional[float] = None
        # Set when the post-loop recipe pass mutates the Vivado session, and
        # never cleared: from then on the live session holds a different
        # flow's state than best_valid.  The one finalize branch that writes
        # from the live session without reopening checks this and refuses,
        # shipping without a refreshed EDIF rather than stamping a
        # mismatched netlist onto the shipped checkpoint.  A recipe that fired
        # and then lost the MUX makes that divergence certain.
        self._finalize_session_untrusted: bool = False
        # Post-accept echo — default off.  When on, every adopted non-bare-route
        # controller move is followed by one bare route_design echo with
        # identical mechanics to a picked bare route: it rides the auto-bank
        # hook, uses the same cost prediction and observed history, and records
        # into the same move state.  The echo counts toward the move budget and
        # needs its own affordability check.
        self._tail_ctrl_m1_echo: bool = False
        # DEEP-SLACK TAIL RESERVE — default 0.0, i.e. no behaviour change.
        # When positive and the design is still deeply negative at the moment
        # remaining wall falls to the reserve, the main loop exits early
        # through the shared exit tail, so the deterministic tail gets a
        # deliberate window instead of whatever the loop happens to leave it —
        # which, on the largest designs, is nothing at all.  A value in (0, 1)
        # is read as a fraction of the wall.
        self._deep_wns_tail_reserve_s: float = DEEP_WNS_TAIL_RESERVE_S_DEFAULT
        # TAIL RESERVE V2: stagnation-guard window (env
        # FPL26_TAIL_RESERVE_STAGNANT_S, default 240 s; 0 = guard off)
        # + once-per-run flag so the wall-clamp warning logs loudly but
        # not on every predicate poll.
        self._tail_reserve_stagnant_s: float = resolve_tail_reserve_stagnant_s()
        self._tail_reserve_clamp_logged: bool = False
        # FRESH-STATE ROUTE PRE-SWEEP — default 0 draws, i.e. no behaviour
        # change.  Takes K banked route re-rolls on the pristine input at step
        # 0 and makes the best draw the pipeline's entry state.  Never-worse:
        # a bad draw re-opens the input checkpoint, which is still on disk.
        self._fresh_presweep_draws: int = FRESH_PRESWEEP_DRAWS_DEFAULT
        # The adopt-entry pre-sweep is retired: it banked entry gains but cost
        # more than it earned by the end of the run — a better point in a worse
        # basin.  The replacement runs the pipeline from the pristine state and
        # registers the best draw as a separate never-worse final candidate.
        # This flag keeps the retired path reachable as an explicit override;
        # nothing ships it.
        self._presweep_adopt_entry: bool = False
        # Pre-sweep outcome, consumed by the Phase-1 feature view: when a draw
        # was adopted the recipe router and pathology classifier must see the
        # post-sweep timing state, while initial_wns stays pristine — the
        # improvement accounting and finalize's no-improvement guard both hang
        # off it, so mutating it would silently discard the banked gain.
        self._presweep_adopted: bool = False
        self._presweep_post_wns: Optional[float] = None
        self._presweep_post_tns: Optional[float] = None
        self._presweep_post_failing_endpoints: Optional[int] = None
        self._presweep_draw_records: list[dict] = []
        self._presweep_hold_failopen_count: int = 0
        # Final-candidate MUX registry.  Each entry is an independently
        # verified, never-worse candidate checkpoint that competes with the
        # pipeline's best_valid at finalize on scored-clock WNS.  best_valid is
        # the implicit candidate zero; anything appended here is an additional
        # one.  Empty by default, in which case the MUX is a strict no-op.
        self._final_candidates: list[dict] = []
        # Harness-tracked routedness of the in-memory design.  Contest inputs
        # arrive placed and routed, so this starts True; call_tool updates it
        # after every successful route, place or checkpoint-open.  The tracking
        # always runs — only the gate's refusal is behind the enable flag.
        self._design_routed_state: bool = True
        # Primitive cell count of the input DCP — measured later in
        # perform_initial_analysis.  Initialized here so the gate's
        # pre-analysis assessments (call_tool runs during Phase 1) fall
        # back cleanly to the heavy-op-history branch instead of
        # raising AttributeError.
        self._input_cell_count: Optional[int] = None
        # Phase-1 resource features. Empty dict /
        # None until report_utilization runs; every consumer treats a
        # missing value as 'rule cannot fire', never as zero.
        self.utilization: dict = {}
        self.memory_dominated: Optional[bool] = None
        self.route_pct: Optional[float] = None
        # Router-plan step coverage, used by the force-continue branch to
        # detect that the model stopped without attempting a recommended heavy
        # step — most often placement.  Recorded on every successful tool call
        # and checked in the stop handler.  Without it a run can end with a
        # third of the wall unspent, having repeated one cheap operation.
        self._tool_calls_seen: list[tuple[str, str]] = []  # (tool_name, directive_or_empty)
        # Provenance of the most recent risky-tool estimate (HISTORY / MODEL /
        # CONSTANT / UNKNOWN). Read by the typed gate ledger so a refusal can be
        # classified as legitimately bounded vs prediction-driven.
        self._last_estimate_provenance = None
        # Per-tool runtime history (rolling).  Feeds _estimate_tool_runtime,
        # which the dispatcher uses to decide whether to start a risky call.
        self._tool_runtime_history: dict[str, list[float]] = {}
        # Wall economics: elapsed seconds of COMPLETED heavy moves this run
        # (risky calls only, so a placement wrapped in raw Tcl counts and a
        # cheap property query does not).  The cheapest entry sizes the stop
        # rule's observation window — a measurement rather than a tuned
        # constant, which is what keeps the rule free of fitted parameters.
        self._heavy_move_seconds: list[float] = []
        # Stable best-valid mirror — the last good checkpoint is copied to
        # <run_dir>/best_valid.{dcp,edf} after each improvement, so the signal
        # handler and the emergency finalize have a fast path that needs no
        # live Vivado session.
        #
        # _best_valid_dcp_wns / _best_valid_edif_wns record which best_wns the
        # on-disk mirror was written for; None means unknown.  Finalize uses
        # them to detect a stale mirror: if the recorded value does not match
        # the current best_wns, the file on disk is not what the optimizer
        # thinks it is shipping, and the fast-path copy would label a baseline
        # checkpoint as the optimized result.
        self._best_valid_dcp: Optional[Path] = None
        self._best_valid_edif: Optional[Path] = None
        self._best_valid_dcp_wns: Optional[float] = None
        self._best_valid_edif_wns: Optional[float] = None
        self._pending_best_mirror: bool = False
        # Monotone counter of successful
        # design-mutating tool calls.  Recorded when _pending_best_mirror
        # arms; the piggyback mirror only stamps best_wns onto an LLM
        # checkpoint written in the SAME epoch (no mutation between the
        # best-WNS measurement and the write).
        self._mutation_epoch: int = 0
        self._pending_best_mirror_epoch: int = 0
        self._termination_reason: Optional[str] = None

        # Best-valid / ship-DCP lineage.
        # _best_valid_token is monotonic per run; each successful mirror
        # capture (eager / piggyback / backstop) bumps it.  Finalize uses
        # the lineage to record which artifact was actually shipped so
        # later audits can prove the final DCP traces back to disk truth.
        self._best_valid_token: int = 0
        self._best_valid_lineage: Optional[dict] = None
        self._ship_lineage: Optional[dict] = None

        # Decision tracer — initialized lazily in
        # optimize() because run_dir may not exist at __init__ time.
        # Held as Optional so the optimizer keeps working when the
        # module is unavailable.
        self._decision_tracer = None  # type: ignore[var-annotated]

        # DCP write-path PathGuard — audit
        # mode default.  Lazy-initialised in optimize() once
        # run_dir and output_dcp.parent so the roots are concrete.
        self._path_guard = None  # type: ignore[var-annotated]
        # Enforcing by default; PATH_GUARD_MODE=audit downgrades violations to
        # log lines for canary or rollback purposes, and a caller may also set
        # the mode directly before optimize().
        self._path_guard_mode: str = os.environ.get(
            "PATH_GUARD_MODE", "enforce"
        )
        # Aggregated audit-mode violations across the run (kept even
        # after a check() returns False so the summary printer can show
        # them).  Each entry: {"path", "context", "ts"}.
        self._path_guard_violations: list[dict] = []

        # DCP lifecycle tracking.  input_dcp_path is the baseline (golden); it
        # must always remain valid and is the fallback of last resort.
        # final_status is one of:
        #   VALID_OPTIMIZED           — output_dcp passed structural validation
        #   VALID_OPTIMIZED_NO_EDIF   — output_dcp valid but EDIF write failed
        #   VALID_FALLBACK_BASELINE   — output_dcp invalid/missing; copied baseline
        #   NO_IMPROVEMENT            — best_wns <= initial_wns; no output written
        #   HARD_FAIL_NO_VALID_BASELINE — even baseline is missing/unreadable
        self.input_dcp_path: Optional[Path] = None
        self.final_status: Optional[str] = None
        self.lifecycle_log: list[dict] = []

        # Log file handles
        self._rw_log_file = None
        self._v_log_file = None
    
    async def start_servers(self, log_prefix: str = ""):
        """Start and connect to both MCP servers."""
        script_dir = Path(__file__).parent.resolve()
        
        # Create log files in run directory
        rapidwright_log = self.run_dir / "rapidwright.log"
        rapidwright_mcp_log = self.run_dir / "rapidwright-mcp.log"
        vivado_log = self.run_dir / "vivado.log"
        vivado_journal = self.run_dir / "vivado.jou"
        vivado_mcp_log = self.run_dir / "vivado-mcp.log"
        
        # Open log files (if not in debug mode, redirect stderr to log)
        if self.debug:
            self._rw_log_file = None
            self._v_log_file = None
            logger.info("Debug mode: MCP server output will be shown in console")
            if log_prefix:
                print(f"{log_prefix} Debug mode: MCP server output will be shown in console")
        else:
            self._rw_log_file = open(rapidwright_mcp_log, 'w')
            self._v_log_file = open(vivado_mcp_log, 'w')
            logger.info(f"RapidWright Java output: {rapidwright_log}")
            logger.info(f"RapidWright MCP output: {rapidwright_mcp_log}")
            logger.info(f"Vivado output: {vivado_log}")
            logger.info(f"Vivado journal: {vivado_journal}")
            logger.info(f"Vivado MCP output: {vivado_mcp_log}")
            print(f"Log files in {self.run_dir.name}/: {rapidwright_log.name}, {rapidwright_mcp_log.name}, {vivado_log.name}, {vivado_journal.name}, {vivado_mcp_log.name}")
        
        # RapidWright MCP server config
        rapidwright_args = [str(script_dir / "RapidWrightMCP" / "server.py")]
        if not self.debug:
            rapidwright_args.extend([
                "--java-log", str(rapidwright_log),
                "--mcp-log", str(rapidwright_mcp_log)
            ])
        
        env = {**os.environ}
        rapidwright_submodule = script_dir / "RapidWright"
        # Only point pip's rapidwright at the local submodule once it has been
        # built.  Unpacked from a release archive the submodule directory
        # exists but is empty, and setting RAPIDWRIGHT_PATH would make pip
        # rapidwright skip its bundled-jar fallback, starting a JVM with no
        # RapidWright classes on it.  See the Makefile's environment setup.
        rapidwright_built_jar = rapidwright_submodule / "build" / "libs" / "rapidwright.jar"
        if rapidwright_built_jar.is_file() and "RAPIDWRIGHT_PATH" not in env:
            env["RAPIDWRIGHT_PATH"] = str(rapidwright_submodule)
            env["CLASSPATH"] = f"{rapidwright_submodule}/bin:{rapidwright_submodule}/jars/*:{rapidwright_submodule}/build/libs/*"
        
        rapidwright_config = {
            "command": sys.executable,
            "args": rapidwright_args,
            "cwd": str(self.run_dir),
            "env": env
        }
        
        # Vivado MCP server config
        vivado_args = [str(script_dir / "VivadoMCP" / "vivado_mcp_server.py")]
        if not self.debug:
            vivado_args.extend([
                "--vivado-log", str(vivado_log),
                "--vivado-journal", str(vivado_journal)
            ])
        
        vivado_config = {
            "command": sys.executable,
            "args": vivado_args,
            "cwd": str(self.run_dir),
            "env": {**os.environ}
        }
        
        # Start RapidWright MCP
        logger.info("Starting RapidWright MCP server...")
        if log_prefix:
            print(f"{log_prefix} Starting RapidWright MCP server...")
        start_time = time.time()
        
        rw_params = StdioServerParameters(**rapidwright_config)
        rw_transport = await self.exit_stack.enter_async_context(
            stdio_client(rw_params, errlog=self._rw_log_file)
        )
        rw_read, rw_write = rw_transport
        self.rapidwright_session = await self.exit_stack.enter_async_context(
            ClientSession(rw_read, rw_write)
        )
        await self.rapidwright_session.initialize()
        
        elapsed = time.time() - start_time
        logger.info(f"RapidWright MCP server started in {elapsed:.2f}s")
        if log_prefix:
            print(f"{log_prefix} RapidWright MCP server started in {elapsed:.2f}s")
        
        # Start Vivado MCP
        logger.info("Starting Vivado MCP server...")
        if log_prefix:
            print(f"{log_prefix} Starting Vivado MCP server...")
        start_time = time.time()
        
        vivado_params = StdioServerParameters(**vivado_config)
        vivado_transport = await self.exit_stack.enter_async_context(
            stdio_client(vivado_params, errlog=self._v_log_file)
        )
        v_read, v_write = vivado_transport
        self.vivado_session = await self.exit_stack.enter_async_context(
            ClientSession(v_read, v_write)
        )
        await self.vivado_session.initialize()
        
        elapsed = time.time() - start_time
        logger.info(f"Vivado MCP server started in {elapsed:.2f}s")
        if log_prefix:
            print(f"{log_prefix} Vivado MCP server started in {elapsed:.2f}s")
        
        logger.info("Both MCP servers connected")
        if log_prefix:
            print(f"{log_prefix} Both MCP servers connected successfully")
    
    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()
        
        if self._rw_log_file:
            self._rw_log_file.close()
        if self._v_log_file:
            self._v_log_file.close()
        
        logger.info(f"Run directory preserved at: {self.run_dir}")
    
    def calculate_fmax(self, wns: Optional[float], clock_period: Optional[float]) -> Optional[float]:
        """Calculate achievable clock frequency in MHz from WNS and clock period.

        fmax = 1000 / (clock_period - WNS), for EVERY sign of WNS.

        Positive slack therefore produces a frequency above the target
        rather than one clamped to it.  Returns None when the inputs
        cannot produce a positive achievable period, including when WNS
        is at or beyond the clock period.
        """
        return _p1_sense.calculate_fmax(wns, clock_period)
    
    async def get_clock_period(self, call_tool_fn) -> Optional[float]:
        """
        Query the clock period of the target clock from Vivado in nanoseconds.

        First checks for the contest clock 'clk_fpl26contest'. If found, uses
        its period and sets self.target_clock. Otherwise falls back to the
        endpoint clock of the worst setup timing path.

        Args: call_tool_fn: Function to call Vivado tools, should accept
        (tool_name, arguments)

        Returns the period of the target clock, or None if no clocks found.
        """
        tcl_cmd = (
            "set contest_clk [get_clocks -quiet clk_fpl26contest]; "
            "if {$contest_clk ne {}} { "
            "  puts \"CLOCK:clk_fpl26contest\"; "
            "  puts [get_property PERIOD $contest_clk]; "
            "} else { "
            "  set tp [get_timing_paths -max_paths 1 -setup]; "
            "  if {$tp ne {}} { "
            "    set clk [get_property ENDPOINT_CLOCK $tp]; "
            "    if {$clk ne {}} { "
            "      puts \"CLOCK:$clk\"; "
            "      puts [get_property PERIOD [get_clocks $clk]]; "
            "    } "
            "  } "
            "}"
        )
        try:
            result = await call_tool_fn("run_tcl", {"command": tcl_cmd})
            
            clock_name = None
            for token in result.strip().split():
                if token.startswith('CLOCK:'):
                    clock_name = token[len('CLOCK:'):]
                    continue
                if token.startswith('ERROR') or token.startswith('WARNING'):
                    continue
                try:
                    period = float(token)
                    if period > 0:
                        if clock_name:
                            self.target_clock = clock_name
                            logger.info(f"Target clock: {clock_name}, period: {period:.3f} ns")
                        else:
                            logger.info(f"Critical clock period: {period:.3f} ns")
                        return period
                except ValueError:
                    continue
        except Exception as e:
            logger.warning(f"Failed to get clock period: {e}")
        
        logger.warning("Could not determine clock period from Vivado")
        return None
    
    async def get_wns_for_target_clock(self, call_tool_fn) -> Optional[float]:
        """
        Get WNS specifically for the target clock domain.

        When target_clock is set (e.g. 'clk_fpl26contest'), queries WNS
        filtered to that clock's timing paths. Falls back to overall WNS if no
        target clock.

        Args: call_tool_fn: Function to call Vivado tools, should accept
        (tool_name, arguments)

        Returns WNS in nanoseconds, or None if query fails.
        """
        if self.target_clock:
            tcl_cmd = (
                f"set clk_obj [get_clocks -quiet {{{self.target_clock}}}]; "
                f"if {{$clk_obj ne {{}}}} {{ "
                f"  set tp [get_timing_paths -max_paths 1 -setup -to $clk_obj]; "
                f"  if {{[llength $tp] > 0}} {{get_property SLACK $tp}} else {{puts 0.0}} "
                f"}} else {{ "
                f"  set tp [get_timing_paths -max_paths 1 -slack_lesser_than 999]; "
                f"  if {{[llength $tp] > 0}} {{get_property SLACK $tp}} else {{puts 0.0}} "
                f"}}"
            )
        else:
            tcl_cmd = (
                "set tp [get_timing_paths -max_paths 1 -slack_lesser_than 999]; "
                "if {[llength $tp] > 0} {get_property SLACK $tp} else {puts 0.0}"
            )
        
        try:
            result = await call_tool_fn("run_tcl", {"command": tcl_cmd})
            for token in result.strip().split('\n'):
                token = token.strip()
                if not token or token.startswith('ERROR') or token.startswith('WARNING'):
                    continue
                try:
                    wns = float(token)
                    clock_info = f" (clock: {self.target_clock})" if self.target_clock else ""
                    logger.info(f"WNS{clock_info}: {wns:.3f} ns")
                    return wns
                except ValueError:
                    continue
        except Exception as e:
            logger.warning(f"Failed to get WNS for target clock: {e}")
        
        return None
    
    def parse_high_fanout_nets(self, report: str) -> list[tuple[str, int, int]]:
        """
        Parse high fanout nets report and return list of (net_name, fanout, path_count).
        Mechanism extracted to optimizer/phase1_sense.py (pure function
        of the report text).
        """
        return _p1_sense.parse_high_fanout_nets(report)

    def _format_fmax_results(
        self,
        clock_period: Optional[float],
        initial_wns: Optional[float],
        result_wns: Optional[float],
        result_label: str = "Final",
    ) -> list[str]:
        """Format Fmax/WNS results block as a list of lines.
        
        """
        initial_fmax = self.calculate_fmax(initial_wns, clock_period)
        result_fmax = self.calculate_fmax(result_wns, clock_period)
        result_fmax_label = f"{result_label} Fmax:"
        result_wns_label = f"{result_label} WNS:"
        
        lines: list[str] = []
        if initial_fmax is not None and result_fmax is not None:
            target_fmax = 1000.0 / clock_period
            fmax_change = result_fmax - initial_fmax
            lines.append(f"  {'Target Fmax:':<21s}{target_fmax:8.2f} MHz  (clock period: {clock_period:.3f} ns)")
            lines.append(f"  {'Initial Fmax:':<21s}{initial_fmax:8.2f} MHz  (WNS: {initial_wns:.3f} ns)")
            lines.append(f"  {result_fmax_label:<21s}{result_fmax:8.2f} MHz  (WNS: {result_wns:.3f} ns)")
            lines.append(f"  {'Fmax Improvement:':<21s}{fmax_change:+8.2f} MHz  (WNS: {result_wns - initial_wns:+.3f} ns)")
        else:
            if clock_period is not None:
                target_fmax = 1000.0 / clock_period
                lines.append(f"  {'Clock period:':<21s}{clock_period:8.3f} ns (target: {target_fmax:.2f} MHz)")
            if initial_wns is not None:
                fmax_str = f"  (fmax: {initial_fmax:.2f} MHz)" if initial_fmax else ""
                lines.append(f"  {'Initial WNS:':<21s}{initial_wns:8.3f} ns{fmax_str}")
            if result_wns is not None:
                fmax_str = f"  (fmax: {result_fmax:.2f} MHz)" if result_fmax else ""
                lines.append(f"  {result_wns_label:<21s}{result_wns:8.3f} ns{fmax_str}")
            if initial_wns is not None and result_wns is not None:
                lines.append(f"  {'WNS Improvement:':<21s}{result_wns - initial_wns:+8.3f} ns")
        
        return lines
    
    
    def print_wns_change(
        self,
        initial_wns: Optional[float],
        final_wns: Optional[float],
        clock_period: Optional[float]
    ):
        """Print Fmax/WNS change comparison with improvement/regression status."""
        if final_wns is None or initial_wns is None:
            return
        
        initial_fmax = self.calculate_fmax(initial_wns, clock_period)
        final_fmax = self.calculate_fmax(final_wns, clock_period)
        
        if initial_fmax is not None and final_fmax is not None:
            fmax_improvement = final_fmax - initial_fmax
            pct = (fmax_improvement / initial_fmax) * 100 if initial_fmax else 0
            print(f"\n*** Fmax: {initial_fmax:.2f} -> {final_fmax:.2f} MHz ({fmax_improvement:+.2f} MHz, {pct:+.1f}%) ***")
            print(f"*** WNS:  {initial_wns:.3f} -> {final_wns:.3f} ns ***")
            if fmax_improvement > 0:
                print(f"IMPROVEMENT: Fmax improved by {fmax_improvement:.2f} MHz")
            elif fmax_improvement < 0:
                print(f"REGRESSION: Fmax got worse by {-fmax_improvement:.2f} MHz")
            else:
                print("NO CHANGE: Fmax is the same")
        else:
            wns_improvement = final_wns - initial_wns
            print(f"\n*** WNS: {initial_wns:.3f} -> {final_wns:.3f} ns ({wns_improvement:+.3f} ns) ***")
            if wns_improvement > 0:
                print(f"IMPROVEMENT: WNS improved by {wns_improvement:.3f} ns")
            elif wns_improvement < 0:
                print(f"REGRESSION: WNS got worse by {-wns_improvement:.3f} ns")
            else:
                print("NO CHANGE")
    
    def print_fmax_status(self, label: str, wns: Optional[float]):
        """Print Fmax (primary) and WNS (secondary) for a given measurement point."""
        if wns is None:
            print(f"*** {label}: WNS unknown ***")
            return
        fmax = self.calculate_fmax(wns, self.clock_period)
        clock_info = f" (clock: {self.target_clock})" if self.target_clock else ""
        if fmax is not None:
            print(f"*** {label} Fmax{clock_info}: {fmax:.2f} MHz (WNS: {wns:.3f} ns) ***")
        else:
            print(f"*** {label} WNS{clock_info}: {wns:.3f} ns ***")
    
    def print_test_summary(
        self,
        title: str,
        elapsed_seconds: float,
        initial_wns: Optional[float],
        final_wns: Optional[float],
        clock_period: Optional[float],
        extra_info: str = ""
    ):
        """Print formatted test summary."""
        print("\n" + "="*70)
        print(title)
        print("="*70)
        print(f"Total runtime: {elapsed_seconds:.2f} seconds ({elapsed_seconds/60:.2f} minutes)")
        
        result_lines = self._format_fmax_results(clock_period, initial_wns, final_wns)
        if result_lines:
            print(f"\nFmax Results:")
            print("\n".join(result_lines))
        
        if extra_info:
            print(f"\n{extra_info}")
        print("="*70)


# Extraction philosophy: pure mechanisms — error classifiers, parsers,
# integrity and digest helpers, budget and backoff math — live in optimizer/*
# modules, while this class holds the per-run state (sessions, budgets,
# best-so-far tracking, the candidate registry) and the orchestration that
# sequences them.  Methods whose bodies moved remain as thin delegates with
# unchanged names and signatures.  The mixins carry cohesive method groups and
# hold no state of their own; none overrides a DCPOptimizerBase method, so the
# base stays first and __bases__[0] keeps meaning what it did before.
class DCPOptimizer(DCPOptimizerBase, RecipePassesMixin, PolishLadderMixin,
                   FinalizationMixin, ToolDispatchMixin):
    """FPGA Design Optimization Agent using RapidWright and Vivado MCPs."""
    
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        debug: bool = False,
        run_dir: Optional[Path] = None,
        mode: str = "v0_3",
    ):
        super().__init__(debug=debug, run_dir=run_dir)

        # Controller mode.  "v0_3" applies the slope-aware ceiling lift and the
        # first-improvement-gated cap.  "anchor" stops as soon as the model
        # signals done — no force-continue, no cap — so a true anchor candidate
        # can be run without forking the codebase.
        if mode not in ("v0_3", "anchor"):
            raise ValueError(f"mode must be 'v0_3' or 'anchor', got {mode!r}")
        self.mode = mode

        # RAG seed toggle.  Default: inject the prior-run strategy
        # snippet at iter 1.  Set
        # rag_seed=False from the CLI (--no-rag-seed) for A/B testing or
        # to measure the seed's actual ΔFmax contribution.
        self.rag_seed = True
        # Contest mode: hidden-design hygiene.  Strategy-memory
        # retrieval is fingerprint-only in every mode; contest mode
        # additionally appends a negative-memory advisory block to the
        # iter-1 prompt.  Default off; opt in via --contest-mode.
        self.contest_mode: bool = False
        # Opt-in advisory policy card.
        self.policy_card: bool = False

        self.api_key = api_key
        self.model = model
        self.tools: list[dict] = []
        self.messages: list[dict] = []
        
        self.openai = OpenAI(
            api_key=api_key,
            # Bound the SDK rather than take its defaults: 600 s per request
            # with two invisible internal retries means one stalled request can
            # eat half the wall without the resilience episode accounting ever
            # seeing it.  The call is synchronous, and across full-suite run
            # logs the largest gap before any response line is a few seconds,
            # so this bound is orders of magnitude above the observed worst
            # case.  Retries belong to optimizer.api_resilience alone: a
            # timeout classifies as transient and enters its backoff schedule.
            timeout=_llm_timeout_s(),
            max_retries=0,
            # T5 harness: env override lets the failure-injection matrix
            # interpose its API-storm proxy (fabricated 401/500 cells);
            # eval/default behavior unchanged when the env is unset.
            base_url=os.environ.get("OPENROUTER_BASE_URL",
                                    "https://openrouter.ai/api/v1")
        )
        
        # Track optimization progress
        self.iteration = 0
        self.best_wns = float('-inf')
        self.no_improvement_count = 0
        # ILS-as-polish (ruin-and-recreate) — stagnation-triggered, never-worse.
        # Default OFF (protects the anchor); enabled via --ils-polish. See
        # optimizer/ils_polish.py.
        from optimizer.ils_polish import ILSPolishConfig as _ILSCfg
        self._ils_polish_cfg = _ILSCfg(enabled=False)
        self._ils_preempt_requested = False
        self._design_cells: Optional[int] = None
        self.last_improvement_time: Optional[float] = None  # ILS stagnation clock
        # When True, call_tool bypasses the budget-skip gate (like _in_finalize):
        # the ILS stage runs after the main loop and self-bounds via its own
        # deadline + per-cycle time gating, so the per-call kill must not fire.
        self._in_ils_stage = False
        # Iteration of last best-WNS improvement.  Initialized
        # to 0 (treats initial WNS as the baseline against which improvements
        # are measured).  Updated whenever best_wns increases.  Used in the
        # main loop to enforce "stop after 3 no-improve iters" on the agent
        # side rather than relying on the LLM's self-report.
        self.last_improvement_iter = 0
        # Count of LLM-signalled stops that were overridden
        # because best_wns improved within the last 2 iters.  Logged in the
        # final summary, so the slope-aware lift can be quantified.
        self.force_continue_count = 0
        # FPL26_POSITIVE_SLACK_CONTINUE: one-shot latch so the zero-crossing
        # is announced once, not once per iteration, after it happens.
        self._positive_slack_logged = False
        # Set by the armed met-timing entry branch; arms the plateau exit from
        # loop start (otherwise structurally dead on that class).
        self._positive_slack_entry_met = False
        # Warn once when current WNS falls more than REGRESSION_NS below the best
        # checkpoint, prompting restoration before another strategy is attempted.
        # A new best WNS re-arms the warning.
        self.regression_warning_sent = False
        self.regression_warning_count = 0
        self.llm_call_count = 0
        
        # Track token usage and costs
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        # β circuit-breaker: effective in-attempt LLM cost exit.
        # Default = the $0.75 constant; the CLI/env budget from the
        # multi-restart wrapper can only LOWER it (resolve_llm_cost_exit).
        self.llm_cost_exit_usd: float = LLM_COST_EXIT_USD
        # ---- cheap pre-flight plan critic ----
        # A second, cheaper model reviews a heavy Vivado move before it burns
        # 10-20 minutes of the eval hour. Advisory only (cannot skip/cancel/
        # rewrite). Default OFF until validated on more runs.
        # See optimizer/plan_critic.py for the measured motivation.
        self.plan_critic_enabled: bool = False
        self.plan_critic_model: str = PLAN_CRITIC_DEFAULT_MODEL
        self.plan_critic_max_calls: int = 4
        # Fire only while spend is below this fraction of the budget — the
        # "unspent beta is the bug" observation made operational.
        self.plan_critic_beta_headroom_frac: float = 0.7
        self._plan_critic_calls: int = 0
        # ---- constraint-guard detection baseline ----
        # Set once from the Phase-1 timing analysis; compared at finalize.
        # None means the baseline was never captured -> reported UNVERIFIED,
        # never silently treated as "unchanged".
        self._constraint_fp_baseline = None
        self.api_call_details = []
        
        # Track all tool calls with timing and WNS
        self.tool_call_details = []
        
        # Track total runtime
        self.start_time = None
        self.end_time = None

        # Seed the incremental cost ledger at $0 as soon as
        # the run dir exists.  A benign crash BEFORE any LLM call then
        # leaves an explicit $0 ledger (wrapper charges $0) instead of NO
        # record (wrapper charges the predictive estimate) — retries after
        # no-LLM crashes stay cheap while real-spend crashes stay charged.
        self._write_cost_ledger()

    async def start_servers(self):
        """Start and connect to both MCP servers."""
        await super().start_servers()
        await self._collect_tools()
        logger.info(f"Connected to servers with {len(self.tools)} tools available")
    


    # Default phys_opt sweep ordering — picked from observed
    # contest-design behaviour: critical_cell_opt and equ_drivers_opt
    # tend to win first on fanout-bound paths; placement_opt and
    # restruct_opt help LUT-dense paths; dsp_register_opt is design-
    # specific; slr_crossing_opt only matters on multi-SLR parts.
    _DEFAULT_PHYS_OPT_SWEEP = (
        "critical_cell_opt",
        "equ_drivers_opt",
        "placement_opt",
        "restruct_opt",
        "dsp_register_opt",
        "slr_crossing_opt",
    )

    # Evidence-backed constants for _physopt_default_fixpoint.  Each is a
    # measurement, not a tuned knob.
    _PODF_MAX_CALLS = 4       # gains by streak position: 10/11, 8/8, 6/6, 1/4
    _PODF_EPSILON_NS = 0.002  # below this a "gain" is measurement noise
    _PODF_MAX_SHARE = 0.15    # one call may never bet >15% of the window

    async def _physopt_default_fixpoint(self) -> None:
        """Run ``phys_opt_design -directive Default`` until it no longer improves
        the result.

        The directive produces a full-design call because the command builder
        discards path groups when a directive is present. Each call that fails
        to exceed the improvement threshold is reverted by reopening the banked
        best checkpoint, and the stage stops after that first non-gain.

        The first wall-time gate uses the design-aware place-and-route
        estimate; later gates use the preceding call's observed duration. Run
        this stage after the model loop and before the polish ladder captures
        its baseline, because earlier execution can pre-empt other model
        actions. The stage is disabled by default.
        """
        if os.environ.get(
                "FPL26_PHYSOPT_DEFAULT_FIXPOINT", "0").strip().lower() not in (
                "1", "true", "on", "yes"):
            return
        if self.best_wns is None or self.best_wns == float("-inf"):
            logger.info("[podf] skipped: no baseline WNS established yet.")
            return
        if self.best_wns >= 0:
            logger.info("[podf] skipped: timing already met "
                        f"(best_wns={self.best_wns:.3f} ns).")
            return

        # Call 1 is costed from the design-aware size model; every later call
        # from the measured duration of the one before it.
        est_s = self._estimate_tool_runtime(
            "vivado_phys_opt_design", is_risky=True)
        committed = 0
        for n in range(1, self._PODF_MAX_CALLS + 1):
            remaining = self._budget_remaining()
            budget_cap = self._PODF_MAX_SHARE * remaining
            if est_s * 1.3 > budget_cap:
                logger.info(
                    f"[podf] stop before call {n}: need {est_s * 1.3:.0f}s "
                    f"(basis {'measured' if n > 1 else 'size-model'} "
                    f"{est_s:.0f}s x1.3) > {self._PODF_MAX_SHARE:.0%} of "
                    f"{remaining:.0f}s remaining = {budget_cap:.0f}s.")
                break

            pre_wns = self.best_wns
            t0 = time.time()
            try:
                out = await self.call_tool(
                    "vivado_phys_opt_design",
                    {"directive": "Default", "timeout": 3600.0})
            except Exception as exc:                      # noqa: BLE001
                logger.warning(f"[podf] call {n} raised {type(exc).__name__}: "
                               f"{exc} — stopping (fail open).")
                break
            observed_s = time.time() - t0
            if _looks_like_tool_error(out):
                logger.info(f"[podf] call {n} returned a tool error after "
                            f"{observed_s:.0f}s — stopping (fail open).")
                break

            # call_tool's auto-bank hook has already measured and, if the
            # design improved AND is routed, moved best_wns + mirrored.
            post_wns = self.best_wns
            delta = (post_wns - pre_wns) if (
                post_wns is not None and pre_wns is not None) else 0.0
            if delta >= self._PODF_EPSILON_NS:
                committed += 1
                logger.info(
                    f"[podf] call {n}: {pre_wns:.3f} -> {post_wns:.3f} ns "
                    f"(D={delta:+.3f}) in {observed_s:.0f}s — KEPT.")
                est_s = observed_s          # measured basis for the next call
                continue

            # No gain: revert Vivado's in-memory state to the banked best so
            # the LLM loop does not start from a degraded placement.
            logger.info(
                f"[podf] call {n}: {pre_wns:.3f} -> "
                f"{post_wns if post_wns is not None else float('nan'):.3f} ns "
                f"(D={delta:+.3f}) in {observed_s:.0f}s — no gain, stopping.")
            if (self._best_valid_dcp is not None
                    and Path(self._best_valid_dcp).exists()):
                try:
                    rv = await self.call_tool(
                        "vivado_open_checkpoint",
                        {"dcp_path": str(self._best_valid_dcp.resolve())})
                    if _looks_like_tool_error(rv):
                        logger.warning("[podf] revert: open_checkpoint failed; "
                                       "leaving Vivado state as-is.")
                    else:
                        self.best_wns = pre_wns
                        logger.info("[podf] revert: re-opened best_valid.dcp "
                                    f"(best_wns restored to {pre_wns:.3f} ns).")
                except Exception as exc:                  # noqa: BLE001
                    logger.warning(f"[podf] revert raised "
                                   f"{type(exc).__name__}: {exc}")
            else:
                logger.info("[podf] revert: no best_valid.dcp to re-open.")
            break

        logger.info(f"[podf] done: {committed} call(s) kept, best_wns="
                    f"{self.best_wns:.3f} ns.")


    def _maybe_emit_phys_opt_preempt(self, sweep_result: dict,
                                     *,
                                     current_flag: Optional[str] = None,
                                     flags_remaining: Optional[int] = None) -> str:
        """Inject a heavy-rescue prompt when the configured no-gain streak is
        reached.

        The trigger fires only once, requires a valid positive streak
        threshold, sufficient remaining budget, and no WNS improvement above
        the initial value. On success it latches the trigger, records the
        decision, and appends a user-role rescue message. A satisfied streak
        with a failed budget or lift gate records the corresponding skip
        reason.

        Returns ``"triggered"`` after injection, ``"skipped_budget"`` or
        ``"skipped_lift"`` for those gate failures, and ``"off"`` when
        disabled, latched, invalid, or below threshold. Callers use this status
        to decide whether to stop the current sweep.
        """
        if self.phys_opt_preempt_after is None:
            return "off"
        if self._phys_opt_preempt_fired:
            return "off"
        try:
            n = int(self.phys_opt_preempt_after)
        except (TypeError, ValueError):
            return "off"
        if n < 1:
            return "off"
        if self._physopt_no_gain_streak < n:
            return "off"

        # Streak has reached the threshold — record what was seen and
        # decide whether to fire or skip.
        statuses_seen = [
            e.get("status") for e in sweep_result.get("per_flag", [])
        ]
        remaining_s = self._budget_remaining()
        if remaining_s is None:
            remaining_s_f = -1.0
        else:
            remaining_s_f = float(remaining_s)
        try:
            budget_floor = float(getattr(
                self, "phys_opt_preempt_budget_floor_s", 1100.0))
        except (TypeError, ValueError):
            budget_floor = 1100.0
        mode = str(getattr(self, "phys_opt_preempt_mode", "end_sweep") or "end_sweep")
        best = self.best_wns if self.best_wns is not None else None
        init = self.initial_wns if self.initial_wns is not None else None
        if best is None or init is None or best == float("-inf"):
            wns_lift_ns = None
        else:
            wns_lift_ns = round(float(best) - float(init), 4)

        gate_budget_ok = remaining_s_f >= budget_floor
        gate_no_lift = (wns_lift_ns is None) or (wns_lift_ns <= 0.0)
        should_fire = gate_budget_ok and gate_no_lift

        base_record = {
            "action_label": "phys_opt_preempt",
            "decision_source": "controller",
            "phase": "phys_opt_preempt",
            "schema_version": 1,
            "extra": {
                "enabled": True,
                "mode": mode,
                "N": n,
                "trigger_count": int(self._physopt_no_gain_streak),
                "statuses_seen": statuses_seen,
                "current_flag": current_flag,
                "flags_remaining": flags_remaining,
                "best_wns_before": (float(best) if best is not None
                                     and best != float("-inf") else None),
                "wns_lift_ns": wns_lift_ns,
                "remaining_wall_seconds": (round(remaining_s_f, 1)
                                            if remaining_s is not None
                                            else None),
                "budget_floor_seconds": budget_floor,
                "preempt_action_family": (
                    "place_design_unplace_then_Auto1_then_route_design_Default"
                ),
                "used_for_decision": bool(should_fire),
            },
        }

        if not should_fire:
            base_record["notes"] = "preempt streak reached threshold but gate failed"
            base_record["extra"]["triggered"] = False
            if not gate_budget_ok:
                base_record["extra"]["skipped_reason"] = "insufficient_budget"
                outcome = "skipped_budget"
            elif not gate_no_lift:
                base_record["extra"]["skipped_reason"] = "wns_already_lifted"
                outcome = "skipped_lift"
            else:
                base_record["extra"]["skipped_reason"] = "unknown"
                outcome = "skipped_budget"
            self._emit_decision(base_record)
            return outcome

        # Trigger.
        self._phys_opt_preempt_fired = True
        base_record["notes"] = "preempt triggered; heavy rescue directive injected"
        base_record["extra"]["triggered"] = True
        self._emit_decision(base_record)

        preempt_message = (
            f"CONTROLLER PREEMPT: phys_opt sweep produced "
            f"{self._physopt_no_gain_streak} consecutive no-gain per-flag "
            f"outcomes (threshold N={n}) with best WNS still at "
            f"{best:.3f} ns and "
            f"{remaining_s_f/60.0:.0f} min of wall budget remaining "
            f"(floor {budget_floor/60.0:.0f} min). "
            "Stop the phys_opt sweep iteration.  Run the deterministic "
            "heavy rescue sequence next, in this order:\n"
            "  1. `vivado_run_tcl` with command `place_design -unplace`\n"
            "  2. `vivado_place_design` with `directive=\"Auto_1\"`\n"
            "  3. `vivado_route_design` with `directive=\"Default\"`\n"
            "Optionally insert one `vivado_phys_opt_design` pass between "
            "steps 2 and 3 if wall budget allows.  Measure WNS after step 3; "
            "if it regresses, revert; otherwise continue.  Do NOT issue "
            "further `phys_opt_design` calls before running the heavy step."
        )
        try:
            self.messages.append({
                "role": "user",
                "content": preempt_message,
            })
        except Exception:  # pragma: no cover — defensive
            logger.exception("phys_opt_preempt: failed to append message")
        logger.info(
            f"[CONTROLLER-PREEMPT] phys_opt preempt TRIGGERED: "
            f"mode={mode} streak={self._physopt_no_gain_streak} N={n} "
            f"remaining={remaining_s_f:.0f}s floor={budget_floor:.0f}s "
            f"best_wns={best} wns_lift_ns={wns_lift_ns} "
            f"current_flag={current_flag} flags_remaining={flags_remaining}"
        )
        return "triggered"


    async def _revert_to_best_valid(self, result: dict) -> None:
        """Restore Vivado state from best_valid.dcp.  Recipe helper.

        Updates result['revert_status'] with the outcome of the revert.
        Idempotent — safe to call when best_valid mirror doesn't exist
        (records the absence instead of erroring out).
        """
        if (self._best_valid_dcp is None
                or not Path(self._best_valid_dcp).exists()):
            result["revert_status"] = "no_best_valid_to_revert_to"
            return
        try:
            revert_str = await self.call_tool(
                "vivado_open_checkpoint",
                {"dcp_path": str(self._best_valid_dcp.resolve())},
            )
            if _looks_like_tool_error(revert_str):
                result["revert_status"] = "open_checkpoint_failed"
            else:
                result["revert_status"] = "reopened_best_valid"
        except Exception as e:
            result["revert_status"] = f"revert_raised:{type(e).__name__}"

    def _phase1_wall_allowance_remaining_s(self) -> Optional[float]:
        """Return the remaining cumulative Phase 1 wall-time allowance in seconds.

        The allowance is ``phase1_wall_frac * max_wall_seconds`` measured from
        the start of initial analysis. Returns ``None`` when no cap applies
        because the wall budget is unset, Phase 1 has not started, or the
        fraction is non-positive; callers must treat this sentinel as uncapped.
        """
        if (self.max_wall_seconds is None
                or self._phase1_start_ts is None
                or self.phase1_wall_frac <= 0.0):
            return None
        allowance = float(self.phase1_wall_frac) * float(self.max_wall_seconds)
        return allowance - (time.time() - self._phase1_start_ts)

    def _phase1_skipped_display(self) -> list[str]:
        """Render phase1_skipped labels with their skip reason when known
        (wall-capped skips carry reason "phase1_wall_cap")."""
        return [
            f"{label} ({self.phase1_skip_reasons[label]})"
            if label in self.phase1_skip_reasons else label
            for label in self.phase1_skipped
        ]

    async def _phase1_call(
        self,
        tool_name: str,
        arguments: dict,
        timeout_seconds: float,
        *,
        mandatory: bool = False,
        label: Optional[str] = None,
    ) -> Optional[str]:
        """Run a Phase 1 tool call with scaled timeouts and graceful failure
        handling.

        Scales the tool timeout by ``phase1_timeout_scale``, passes it to the
        tool, and applies a client-side ``asyncio.wait_for`` limit with 30
        seconds of headroom. This prevents optional analysis calls from
        blocking entry into the optimization loop.

        On failure, mandatory calls raise ``RuntimeError``. Optional calls log
        a warning, append their label to ``phase1_skipped``, and return
        ``None``. Successful calls return the raw tool response string.
        """
        label = label or tool_name
        effective = float(timeout_seconds) * float(self.phase1_timeout_scale)
        # Cumulative Phase-1 wall cap.  Optional steps stop launching once
        # Phase 1 has consumed its share of the wall, and their scaled timeout
        # never exceeds the allowance still remaining — uncapped optional steps
        # at a high timeout scale were the last mechanism that could still
        # burn a whole wall on analysis.  Mandatory steps are never capped.
        if not mandatory:
            remaining = self._phase1_wall_allowance_remaining_s()
            if remaining is not None:
                if remaining <= 0.0:
                    msg = (
                        f"Phase 1 step '{label}' skipped: cumulative Phase-1 "
                        f"wall cap reached ({self.phase1_wall_frac:.0%} of "
                        f"{self.max_wall_seconds:.0f}s wall budget)."
                    )
                    logger.warning(msg)
                    self.phase1_skipped.append(label)
                    self.phase1_skip_reasons[label] = "phase1_wall_cap"
                    return None
                if effective > remaining:
                    logger.info(
                        f"Phase 1 step '{label}': scaled timeout "
                        f"{effective:.0f}s clamped to remaining Phase-1 "
                        f"allowance {remaining:.0f}s (Phase-1 wall cap)."
                    )
                    effective = remaining
        args = dict(arguments)
        args.setdefault("timeout", effective)
        try:
            result = await asyncio.wait_for(
                self.call_tool(tool_name, args),
                timeout=effective + 30.0,
            )
        except asyncio.TimeoutError:
            msg = (
                f"Phase 1 step '{label}' timed out client-side after "
                f"{effective + 30.0:.0f}s."
            )
            logger.warning(msg)
            self.phase1_skipped.append(label)
            if mandatory:
                raise RuntimeError(msg)
            return None
        except Exception as e:
            msg = f"Phase 1 step '{label}' raised {type(e).__name__}: {e}"
            logger.warning(msg)
            self.phase1_skipped.append(label)
            if mandatory:
                raise RuntimeError(msg)
            return None

        if _looks_like_tool_error(result):
            msg = f"Phase 1 step '{label}' tool-error envelope: {str(result)[:200]}"
            logger.warning(msg)
            self.phase1_skipped.append(label)
            if mandatory:
                raise RuntimeError(msg)
            return None
        return result

    
    async def _call_vivado_tool(self, tool_name: str, arguments: dict) -> str:
        """Helper to call Vivado tools (for use with base class methods)."""
        return await self.call_tool(f"vivado_{tool_name}", arguments)

    # Auto-revert injection.  A WNS regression larger than the threshold
    # triggers a revert warning: left alone, the model will accept a worse
    # state and build on it for many iterations without recovering.  The
    # warning is injected as a user message the next time the model is
    # invoked, telling it to revert before trying anything new.  The
    # sent-flag prevents repeats and is re-armed whenever best_wns improves.
    REGRESSION_THRESHOLD_NS = 0.05

    def _maybe_inject_revert_warning(self, current_wns: float) -> None:
        """Inject a 'revert before new strategy' user message when WNS
        regresses materially below best_wns.  No-op once per regression-
        episode (re-armed on next improvement)."""
        if self.regression_warning_sent:
            return
        if current_wns >= self.best_wns - self.REGRESSION_THRESHOLD_NS:
            return
        if self.best_wns == float('-inf'):
            return  # no baseline yet; first reading
        self.regression_warning_sent = True
        self.regression_warning_count += 1
        delta = self.best_wns - current_wns
        logger.info(
            f"[BETA-CTRL-V0.1] Regression detected: best={self.best_wns:.3f} ns -> "
            f"current={current_wns:.3f} ns (delta={delta:.3f} ns). "
            f"Injecting revert instruction (#{self.regression_warning_count})."
        )
        self.messages.append({
            "role": "user",
            "content": (
                f"WNS regressed from best={self.best_wns:.3f} ns to current={current_wns:.3f} ns "
                f"(degradation of {delta:.3f} ns).  "
                f"DO NOT propose a new strategy on this degraded state. "
                f"FIRST revert: open the DCP checkpoint that was saved when WNS was {self.best_wns:.3f} ns or better. "
                f"Look back at your earlier `vivado_write_checkpoint` / `rapidwright_write_checkpoint` calls — "
                f"identify the most recent checkpoint at or above the best WNS and `vivado_open_checkpoint` it. "
                f"After you've reverted, then choose a DIFFERENT strategy class than the one that just regressed."
            )
        })

    async def process_response(self, response) -> tuple[str, bool]:
        """Process LLM response, execute tool calls, return final text and done flag."""
        # Validate response structure with detailed logging
        try:
            if not response:
                raise ValueError("Response is None")
            if not hasattr(response, 'choices'):
                raise ValueError(f"Response has no 'choices' attribute. Response type: {type(response)}, Response: {response}")
            if response.choices is None:
                raise ValueError("Response.choices is None")
            if len(response.choices) == 0:
                raise ValueError("Response choices list is empty")
            
            message = response.choices[0].message
            if not message:
                raise ValueError("Message is None")
        except Exception as e:
            logger.error(f"Failed to parse response structure: {e}")
            logger.error(f"Response object: {response}")
            raise
        
        # Convert message to dict, excluding None values which can cause issues
        message_dict = message.model_dump(exclude_none=True)
        self.messages.append(message_dict)
        
        if self.debug:
            logger.debug(f"Added message to conversation: {json.dumps(message_dict, indent=2)[:500]}...")
        
        # Check for tool calls
        if message.tool_calls:
            tool_results = []
            
            for tool_call in message.tool_calls:
                # Validate tool_call structure
                if not tool_call or not hasattr(tool_call, 'function') or not tool_call.function:
                    logger.warning(f"Invalid tool_call structure: {tool_call}")
                    continue
                
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                except json.JSONDecodeError:
                    tool_args = {}
                
                result = await self.call_tool(tool_name, tool_args)
                
                # Truncate very long results to avoid API issues
                MAX_RESULT_LENGTH = 50000  # characters
                if len(result) > MAX_RESULT_LENGTH:
                    logger.warning(f"Tool result from {tool_name} is {len(result)} chars, truncating to {MAX_RESULT_LENGTH}")
                    result = result[:MAX_RESULT_LENGTH] + f"\n...[truncated {len(result) - MAX_RESULT_LENGTH} characters]"
                
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": result
                })
                
                # Debug logging
                if self.debug:
                    logger.debug(f"Tool {tool_name} result: {result[:500]}...")
            
            # Add tool results to messages
            self.messages.extend(tool_results)
            
            # Continue conversation
            return await self.get_completion()
        
        # No tool calls - check whether the run is done
        content = message.content or ""
        
        # Check for completion indicators
        is_done = any(phrase in content.lower() for phrase in [
            "optimization complete",
            "timing is met",
            "wns >= 0",
            "no more optimizations",
            "design meets timing",
            "successfully saved",
            "final design saved"
        ])
        
        return content, is_done
    
    # ------------------------------------------------------------------
    # FRESH-STATE ROUTE-LOTTERY PRE-SWEEP
    # ------------------------------------------------------------------


    def _phase1_wns_for_features(self) -> Optional[float]:
        """WNS the Phase-1 feature consumers (recipe router, pathology
        classifier) should route on: the post-sweep value when a
        pre-sweep draw was adopted, else the pristine entry WNS.
        initial_wns itself stays pristine (accounting — see __init__)."""
        if (getattr(self, "_presweep_adopted", False)
                and self._presweep_post_wns is not None):
            return self._presweep_post_wns
        return self.initial_wns

    def _phase1_failing_endpoints_for_features(self) -> Optional[int]:
        """Failing-endpoint count for the Phase-1 feature consumers
        (post-sweep re-parse when adopted, else entry value)."""
        if (getattr(self, "_presweep_adopted", False)
                and self._presweep_post_failing_endpoints is not None):
            return self._presweep_post_failing_endpoints
        return self.initial_failing_endpoints

    async def perform_initial_analysis(self, input_dcp: Path) -> str:
        """
        Perform initial analysis without LLM:
        1. Initialize RapidWright
        2. Open checkpoint in Vivado
        3. Report timing summary
        4. Get critical high fanout nets
        
        Returns a formatted summary of the analysis.
        """
        logger.info("Performing initial design analysis...")
        print("\n=== Initial Design Analysis ===\n")

        # Anchor the cumulative Phase-1 wall clock.  Every call in
        # this method (wrapped or not) counts against the Phase-1 allowance
        # checked by _phase1_wall_allowance_remaining_s.
        self._phase1_start_ts = time.time()

        # Step 1: Initialize RapidWright
        logger.info("Initializing RapidWright...")
        print("Initializing RapidWright...")
        result = await self.call_tool("rapidwright_initialize_rapidwright", {})
        if "error" in result.lower() and "success" not in result.lower():
            raise RuntimeError(f"Failed to initialize RapidWright: {result}")
        print("✓ RapidWright initialized\n")
        
        # Step 2: Open checkpoint in Vivado (MANDATORY).  Base 600 s — large
        # designs (the two largest) need longer than the server-
        # side 300 s default.  Scale via --phase1-timeout-scale on the CLI.
        logger.info(f"Opening checkpoint: {input_dcp}")
        print(f"Opening checkpoint: {input_dcp.name}")
        result = await self._phase1_call(
            "vivado_open_checkpoint",
            {"dcp_path": str(input_dcp.resolve())},
            timeout_seconds=600.0,
            mandatory=True,
            label="open_checkpoint",
        )
        if "error" in result.lower() and "opened successfully" not in result.lower():
            raise RuntimeError(f"Failed to open checkpoint: {result}")
        print("✓ Checkpoint opened in Vivado\n")

        # Golden cell count.  The validator's cell-count gate is informational
        # rather than blocking, but measuring the input once here lets
        # netlist-transforming accepts be checked against a wide sanity band
        # that catches wholesale netlist mangling.  Non-mandatory: the check
        # fails open.
        try:
            _cc = await self.call_tool("vivado_run_tcl", {
                "command": "llength [get_cells -quiet -hierarchical "
                           "-filter {IS_PRIMITIVE}]",
                # A flat cap is not scaled by phase1_timeout_scale, so a single
                # timeout on a slow cold machine flips three gates at once on a
                # very large design.  This is plain call_tool and is not
                # clamped by the Phase-1 allowance, so a much larger value
                # would instead exhaust that allowance and skip the spread
                # analysis — itself a behavioural change on a degraded run.
                "timeout": 600})
            # Tool-error envelopes can contain digits that resemble a cell count.
            # Reject them before parsing so they cannot corrupt size-based decisions.
            # Query failures return None, preserving the parser's fail-open behavior.
            if _looks_like_tool_error(_cc):
                raise RuntimeError(
                    f"cell-count query returned error envelope: "
                    f"{str(_cc)[:120]}")
            _ccm = re.search(r"(\d+)", _cc or "")
            self._input_cell_count = int(_ccm.group(1)) if _ccm else None
            if self._input_cell_count:
                self._ils_polish_cfg.golden_cell_count = self._input_cell_count
                _fl = self._ils_polish_cfg.cell_floor_ratio
                _cl = self._ils_polish_cfg.cell_ceil_ratio
                logger.info(f"Input primitive cell count: {self._input_cell_count:,} "
                            f"(sanity band: {int(self._input_cell_count*_fl):,}"
                            f"..{int(self._input_cell_count*_cl):,})")
        except Exception as e:
            self._input_cell_count = None
            logger.warning(f"input cell-count measurement failed ({e!r}); "
                           "cell guard disabled (fail-open)")

        # Step 3: Report timing summary (MANDATORY).  Base 300 s — large
        # designs may need longer; scaled by phase1_timeout_scale.
        logger.info("Analyzing timing...")
        print("Analyzing timing...")
        timing_report = await self._phase1_call(
            "vivado_report_timing_summary",
            {},
            timeout_seconds=300.0,
            mandatory=True,
            label="report_timing_summary",
        )
        
        # Parse timing
        timing_info = parse_timing_summary_static(timing_report)
        self.initial_tns = timing_info["tns"]
        self.initial_failing_endpoints = timing_info["failing_endpoints"]
        
        # Get clock period for fmax calculation (also detects target clock)
        self.clock_period = await super().get_clock_period(self._call_vivado_tool)
        
        # Get WNS for the target clock domain
        target_wns = await super().get_wns_for_target_clock(self._call_vivado_tool)
        if target_wns is not None:
            self.initial_wns = target_wns
        else:
            self.initial_wns = timing_info["wns"]
        self.best_wns = self.initial_wns if self.initial_wns is not None else float('-inf')
        # Constraint-guard detection: fingerprint the timing constraints at the
        # baseline so finalize can prove they were not altered.  Detection is
        # the load-bearing half of the guard — the deny-list at the raw-Tcl
        # boundary is string matching, and string matching is defeatable by
        # alias, source or eval; a fingerprint diff is not.  Never blocks
        # Phase 1.
        try:
            self._constraint_fp_baseline = \
                await self._capture_constraint_fingerprint("baseline")
        except Exception as e:
            logger.warning(f"[constraint-guard] baseline capture failed "
                           f"(ignored): {e!r}")

        clock_info = f" (clock: {self.target_clock})" if self.target_clock else ""
        print(f"✓ Timing analyzed:")
        if self.clock_period is not None:
            target_fmax = 1000.0 / self.clock_period
            print(f"  - Clock period: {self.clock_period:.3f} ns (target fmax: {target_fmax:.2f} MHz)")
        if self.target_clock:
            print(f"  - Target clock: {self.target_clock}")
        if self.initial_wns is not None:
            print(f"  - WNS{clock_info}: {self.initial_wns:.3f} ns")
            initial_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
            if initial_fmax is not None:
                print(f"  - Achievable fmax: {initial_fmax:.2f} MHz")
        if self.initial_tns is not None:
            print(f"  - TNS: {self.initial_tns:.3f} ns")
        if self.initial_failing_endpoints is not None:
            print(f"  - Failing endpoints: {self.initial_failing_endpoints}")
        print()

        # Fresh-state route pre-sweep (default off).  Insertion contract: after
        # the entry WNS is measured on the pristine state — initial_wns must
        # stay pristine so a banked pre-sweep gain counts as improvement and
        # survives finalize's no-improvement guard — and before every remaining
        # Phase-1 feature capture, so those see the post-sweep state.
        if self._fresh_presweep_draws > 0:
            try:
                await self._run_fresh_presweep(input_dcp)
            except Exception as e:
                logger.warning(
                    f"[pre-sweep] raised {type(e).__name__} (non-fatal): {e}")
                if not self._presweep_adopted:
                    # Leave the pipeline a sane pristine state; an adopted
                    # draw is already the in-memory state + banked mirror.
                    try:
                        await self._presweep_reopen(input_dcp)
                    except Exception:
                        pass

        # Step 4: Get critical high fanout nets (OPTIONAL).  Base 600 s.
        # Used by the LLM to gauge Class D applicability; on failure this sets
        # an empty list and the LLM picks a different strategy class.
        logger.info("Identifying critical high fanout nets...")
        print("Identifying critical high fanout nets...")
        nets_report = await self._phase1_call(
            "vivado_get_critical_high_fanout_nets",
            {"num_paths": 50, "min_fanout": 100},
            timeout_seconds=600.0,
            mandatory=False,
            label="get_critical_high_fanout_nets",
        )
        if nets_report is None:
            self.high_fanout_nets = []
            print("⚠ Skipped high-fanout-net analysis; LLM will not see Class D candidates.\n")
        else:
            self.high_fanout_nets = self.parse_high_fanout_nets(nets_report)
            print(f"✓ Found {len(self.high_fanout_nets)} high fanout nets (>100 fanout)\n")

        # Step 5: Load design in RapidWright for spread analysis (OPTIONAL).
        # If this fails (RapidWright OOM, parse error, JNI crash on huge
        # netlists), skip steps 6+7 too — they depend on it.
        critical_path_spread_info = None
        self.lut_count = None

        logger.info("Loading design in RapidWright...")
        print("Loading design in RapidWright for spread analysis...")
        result = await self._phase1_call(
            "rapidwright_read_checkpoint",
            {"dcp_path": str(input_dcp.resolve())},
            timeout_seconds=900.0,
            mandatory=False,
            label="rapidwright_read_checkpoint",
        )
        if result is None or ("error" in result.lower() and "success" not in result.lower()):
            if result is not None:
                logger.warning(f"RapidWright load returned non-success: {result[:200]}")
                if "rapidwright_read_checkpoint" not in self.phase1_skipped:
                    self.phase1_skipped.append("rapidwright_read_checkpoint")
            print("⚠ Skipped spread analysis (RapidWright load unavailable).\n")
        else:
            print("✓ Design loaded in RapidWright\n")

            # Capture cell-count fingerprint (used by strategy_memory's
            # fingerprint_match for unknown designs).  Best-effort.
            info_str = await self._phase1_call(
                "rapidwright_get_design_info",
                {},
                timeout_seconds=120.0,
                mandatory=False,
                label="rapidwright_get_design_info",
            )
            if info_str is not None:
                try:
                    info = json.loads(info_str) if isinstance(info_str, str) else info_str
                    top_types = info.get("top_cell_types", {}) if isinstance(info, dict) else {}
                    self.lut_count = sum(
                        cnt for typ, cnt in top_types.items()
                        if isinstance(typ, str) and typ.upper().startswith("LUT")
                    )
                    if self.lut_count:
                        print(f"  - LUT count (fingerprint): {self.lut_count:,}\n")
                except Exception as e:
                    logger.debug(f"lut_count fingerprint capture failed: {e}")

            # Step 6: Extract critical path cells from Vivado (OPTIONAL).
            logger.info("Extracting critical path cells for spread analysis...")
            temp_path = Path(self.temp_dir) / "initial_critical_paths.json"
            cells_json = await self._phase1_call(
                "vivado_extract_critical_path_cells",
                {"num_paths": 50, "output_file": str(temp_path)},
                timeout_seconds=600.0,
                mandatory=False,
                label="extract_critical_path_cells",
            )

            # Step 7: Analyze spread in RapidWright (OPTIONAL, depends on 6).
            spread_result = None
            if cells_json is not None and temp_path.exists():
                spread_result = await self._phase1_call(
                    "rapidwright_analyze_critical_path_spread",
                    {"input_file": str(temp_path)},
                    timeout_seconds=300.0,
                    mandatory=False,
                    label="analyze_critical_path_spread",
                )
            else:
                if "extract_critical_path_cells" not in self.phase1_skipped:
                    self.phase1_skipped.append("extract_critical_path_cells")

            if spread_result is not None:
                try:
                    spread_data = json.loads(spread_result)
                    critical_path_spread_info = {
                        "max_distance": spread_data.get("max_distance_found", 0),
                        "avg_distance": spread_data.get("avg_max_distance", 0),
                        "paths_analyzed": spread_data.get("paths_analyzed", 0)
                    }
                    print(f"✓ Critical path spread analyzed:")
                    print(f"  - Max distance: {critical_path_spread_info['max_distance']} tiles")
                    print(f"  - Avg distance: {critical_path_spread_info['avg_distance']:.1f} tiles")
                    print(f"  - Paths analyzed: {critical_path_spread_info['paths_analyzed']}")
                    print()
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning(f"Could not parse spread results: {e}")
                    if "analyze_critical_path_spread" not in self.phase1_skipped:
                        self.phase1_skipped.append("analyze_critical_path_spread")

        # Stash for downstream consumers (RAG strategy memory uses spread
        # for fingerprint-matching when the design name is unknown).
        self.critical_path_spread_info = critical_path_spread_info
        # Spread-gate plumbing: hand the measured avg
        # critical-path spread to the ILS config so the PARTIAL_RUIN
        # skip-gate can fire (optimizer/ils_polish.py, held-out rule 2:
        # spread~0 cell surgery is 26/26 negative). A missing/failed
        # measurement leaves it None -> gate stays open (behavior unchanged).
        try:
            if (isinstance(critical_path_spread_info, dict)
                    and critical_path_spread_info.get("avg_distance") is not None):
                self._ils_polish_cfg.critical_path_avg_spread_tiles = float(
                    critical_path_spread_info["avg_distance"])
                logger.info(
                    f"ILS spread-gate feature: critical_path_avg_spread_tiles="
                    f"{self._ils_polish_cfg.critical_path_avg_spread_tiles:.1f}")
        except (TypeError, ValueError) as e:
            logger.warning(f"ILS spread-gate plumbing skipped ({e!r}); "
                           "PARTIAL_RUIN gate stays open (fail-open).")
        # Pair the spread above with the Phase-1 failing-endpoint count so
        # the ILS can compute failing-endpoint density, the ExtraNetDelay_high
        # gate feature (optimizer/ils_polish.py). Uses the same accessor the
        # recipe router consumes, so the two can never disagree about what
        # Phase 1 measured. Missing -> stays None -> that gate fails OPEN.
        try:
            _fe = self._phase1_failing_endpoints_for_features()
            # Assign UNCONDITIONALLY, including the None case. Setting the field
            # only on success lets a stale value from an earlier Phase 1 on the
            # same optimizer instance masquerade as a fresh measurement, so the
            # gate would decide from the previous design instead of failing open.
            self._ils_polish_cfg.phase1_failing_endpoints = (
                int(_fe) if _fe is not None else None)
            if _fe is not None:
                _sp = self._ils_polish_cfg.critical_path_avg_spread_tiles
                if _sp:
                    logger.info(
                        f"ILS endhigh-density feature: failing={int(_fe)} / "
                        f"spread={float(_sp):.1f} = "
                        f"{int(_fe) / float(_sp):.0f}")
        except (TypeError, ValueError, ZeroDivisionError) as e:
            logger.warning(f"ILS endhigh-density plumbing skipped ({e!r}); "
                           "ExtraNetDelay_high gate stays open (fail-open).")

        # Step 8: resource features (optional).  report_utilization yields
        # utilisation and the block-RAM/URAM/DSP mix in one call, which is what
        # makes memory-dominated and highly-utilised design classes visible to
        # the router at all.  Output goes through the tolerant parser; on any
        # failure the whole feature set stays None and Phase 1 still completes.
        #
        # This replaced a QoR-assessment call that cost up to a quarter of
        # Phase 1 on the largest designs while its downstream signal measured
        # neutral.  The step was already optional and observed skipping
        # cleanly, so every consumer already handles absence.
        logger.info("Running Vivado report_utilization for Phase-1 resource features...")
        print("Running utilization report...")
        util_text = await self._phase1_call(
            "vivado_run_tcl",
            {"command": ("set _u [report_utilization -return_string]; "
                         "set _r \"\"; "
                         "catch {set _r [report_route_status -return_string]}; "
                         "puts $_u; puts \"===FPL26_ROUTE_STATUS===\"; puts $_r")},
            timeout_seconds=180.0,
            mandatory=False,
            label="report_utilization",
        )
        if isinstance(util_text, str) and util_text and not _looks_like_tool_error(util_text):
            try:
                _util_txt, _route_txt = split_reports(util_text)
                self.utilization = parse_utilization(_util_txt)
                self.route_pct = parse_route_status(_route_txt)
                _lu = self.utilization.get("lut_pct")
                _br = self.utilization.get("bram_pct")
                _ur = self.utilization.get("uram_pct")
                _ds = self.utilization.get("dsp_pct")
                self.memory_dominated = memory_dominated(self.utilization)
                _fmt = lambda v: "?" if v is None else f"{v:.1f}%"
                logger.info(
                    "Utilization: LUT %s  BRAM %s  URAM %s  DSP %s  "
                    "memory_dominated=%s",
                    _fmt(_lu), _fmt(_br), _fmt(_ur), _fmt(_ds),
                    self.memory_dominated,
                )
                print(f"✓ Utilization: LUT {_fmt(_lu)} | BRAM {_fmt(_br)} | "
                      f"URAM {_fmt(_ur)} | DSP {_fmt(_ds)} | "
                      f"routed {_fmt(self.route_pct)}")
            except Exception as e:
                logger.warning(f"utilization parse failed: {e}; "
                               "continuing without resource features.")
        else:
            print("⚠ Skipped report_utilization (Vivado returned no usable output).\n")

        if self.phase1_skipped:
            print(f"⚠ Phase 1 degraded: skipped {len(self.phase1_skipped)} optional step(s): "
                  f"{', '.join(self._phase1_skipped_display())}.")
            print("  Optimization continues — LLM will operate without these analyses.\n")

        # Create concise summary for LLM
        summary = []
        summary.append("=== Initial Design Analysis ===\n")

        # Phase-1 degradation banner.  Tell the LLM which optional analyses
        # weren't available so it can pick a strategy without the missing
        # signals (e.g., skip Class D if high-fanout list is empty, skip
        # PBLOCK heuristic if spread analysis was skipped).
        if self.phase1_skipped:
            summary.append("PHASE-1 NOTICE: the following analyses were unavailable")
            summary.append(f"(skipped on this design): {', '.join(self._phase1_skipped_display())}.")
            summary.append("Choose strategy without these signals — proceed with timing data alone.")
            summary.append("")

        # Timing status
        summary.append("TIMING STATUS:")
        if self.clock_period is not None:
            target_fmax = 1000.0 / self.clock_period
            summary.append(f"  Clock period: {self.clock_period:.3f} ns (target fmax: {target_fmax:.2f} MHz)")
        if self.initial_wns is not None:
            if self.initial_wns >= 0:
                summary.append(f"  WNS: {self.initial_wns:.3f} ns - TIMING MET ✓")
            else:
                summary.append(f"  WNS: {self.initial_wns:.3f} ns - TIMING VIOLATED")
            # Add fmax information
            initial_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
            if initial_fmax is not None:
                summary.append(f"  Achievable fmax: {initial_fmax:.2f} MHz")
        if (getattr(self, "_presweep_adopted", False)
                and self._presweep_post_wns is not None
                and self.initial_wns is not None):
            summary.append(
                f"  PRE-SWEEP: a fresh-state route re-roll was BANKED before "
                f"this analysis — WNS {self.initial_wns:.3f} -> "
                f"{self._presweep_post_wns:.3f} ns. You are optimizing from "
                f"this improved (already banked) state; do not repeat a bare "
                f"re-route expecting the same gain.")
        if self.initial_tns is not None:
            summary.append(f"  TNS: {self.initial_tns:.3f} ns")
        if self.initial_failing_endpoints is not None:
            summary.append(f"  Failing endpoints: {self.initial_failing_endpoints}")
        summary.append("")
        
        # Critical path spread analysis
        if critical_path_spread_info:
            summary.append("CRITICAL PATH SPREAD ANALYSIS:")
            summary.append(f"  Max cell distance: {critical_path_spread_info['max_distance']} tiles")
            summary.append(f"  Avg cell distance: {critical_path_spread_info['avg_distance']:.1f} tiles")
            summary.append(f"  Paths analyzed: {critical_path_spread_info['paths_analyzed']}")
            
            # Recommendation based on spread
            if critical_path_spread_info['avg_distance'] > 70 and critical_path_spread_info['paths_analyzed'] >= 5:
                summary.append(f"  ⚠ RECOMMENDATION: Use PBLOCK strategy (high spread detected)")
            summary.append("")
        
        # High fanout nets (show top 10)
        if self.high_fanout_nets:
            summary.append("CRITICAL HIGH FANOUT NETS (top 10):")
            for i, (net_name, fanout, path_count) in enumerate(self.high_fanout_nets[:10]):
                summary.append(f"  {i+1}. {net_name}")
                summary.append(f"     Fanout: {fanout}, Critical paths: {path_count}")
            if len(self.high_fanout_nets) > 10:
                summary.append(f"  ... and {len(self.high_fanout_nets) - 10} more nets")
        else:
            summary.append("CRITICAL HIGH FANOUT NETS: None found")
        
        summary.append("")
        summary.append(f"Total nets available for optimization: {len(self.high_fanout_nets)}")

        # Vivado QoR Assessment feature block.  Compact:
        # score + flow guidance only; no raw RQA text bloating the prompt.
        # The LLM treats this as advisory diagnosis, not a command source.
        rqa_score = self.qor_assessment.get("score") if isinstance(self.qor_assessment, dict) else None
        if rqa_score is not None:
            summary.append("")
            summary.append("VIVADO QoR ASSESSMENT (advisory):")
            score_label = {
                1: "design unlikely to complete implementation",
                2: "implementation completes but timing will not close",
                3: "small chance of success — try directives or report_qor_suggestions",
                4: "should close with a few directives or ML strategies",
                5: "design will easily meet timing",
            }.get(rqa_score, "")
            summary.append(f"  RQA Score: {rqa_score}/5"
                            + (f" — {score_label}" if score_label else ""))
            if self.qor_assessment.get("flow_guidance"):
                summary.append(f"  Recommended next action: "
                                f"{self.qor_assessment['flow_guidance']}")
            if self.qor_assessment.get("ml_strategy_available") is True:
                summary.append("  ML Strategies available for this design.")
            if isinstance(self.qor_assessment.get("methodology_violations"), int):
                summary.append(
                    f"  Methodology violations flagged: "
                    f"{self.qor_assessment['methodology_violations']}"
                )

        # Deterministic pathology classifier — feeds the model a structured
        # recipe-priority list derived from the Phase-1 data, before it makes
        # its first strategy decision.  It also drops recipes an applicability
        # table marks harmful for the design at hand.
        design_name_for_classifier = None
        if _design_name_from_dcp is not None:
            try:
                design_name_for_classifier = _design_name_from_dcp(input_dcp)
            except Exception:
                pass
        pathology_block = self._build_pathology_block(design_name_for_classifier)
        if pathology_block:
            summary.append("")
            summary.extend(pathology_block)

        # Feature-based recipe router: deterministic second-stage routing on
        # top of the pathology classifier.  The pathology label is often the
        # same across designs whose winning recipes differ, so the router uses
        # |WNS|, the fmax ratio, failing-endpoint count, spread and remaining
        # budget to choose among them.  Always on; silent when features are
        # missing.
        router_block = self._build_recipe_router_block()
        if router_block:
            summary.append("")
            summary.extend(router_block)

        # Cross-model steering (added after a primary-model migration): when enabled AND
        # the design matches a known historical winner, append a compact
        # hint describing what the prior model did. No-op when the env flag
        # is unset, so default behavior is unchanged.
        steering_hint = _steering_iter1_hint(design_name_for_classifier)
        if steering_hint:
            summary.append("")
            summary.append(steering_hint)

        summary_text = "\n".join(summary)
        print(summary_text)
        print()

        return summary_text

    def _build_pathology_block(self, design_name: Optional[str] = None) -> list[str]:
        """Build summary lines describing the deterministic Phase 1 pathology
        diagnosis.

        Creates one design-level synthetic path carrying cell spread and slack,
        plus entries for leading high-fanout nets so fanout-related diagnoses
        can activate. If the design applicability table marks cell replacement
        as harmful or invalid, that recipe is omitted and reported as blocked.

        Returns an empty list when the classifier is unavailable or no signal
        data exists.
        """
        if _classify_design is None:
            return []
        paths: list[dict] = []
        # Global design-level "path" so cell_spread + slack can fire
        # at design-level even when no per-path enrichment is available.
        design_level_path: dict = {}
        # Pre-sweep feature view: classify on the post-sweep
        # timing state when a step-0 route re-roll was adopted.
        _slack_for_features = self._phase1_wns_for_features()
        if _slack_for_features is not None:
            design_level_path["slack_ns"] = _slack_for_features
        if (isinstance(self.critical_path_spread_info, dict)
                and self.critical_path_spread_info.get("avg_distance") is not None):
            design_level_path["cell_spread"] = float(
                self.critical_path_spread_info["avg_distance"]
            )
        # Roll up top fanouts as "nets on this path" so HIGH_FANOUT_DRIVER
        # can fire from a single design-level path.
        if self.high_fanout_nets:
            top_nets = [(name, fanout) for name, fanout, _ in self.high_fanout_nets[:5]]
            design_level_path["high_fanout_nets"] = top_nets
        if design_level_path:
            paths.append(design_level_path)

        # One synthetic path per top-3 high-fanout net, so that count-based
        # tie-breaking can surface a high-fanout driver as the primary label
        # when fanout is the dominant signal.  Each path also inherits the
        # design-level cell spread, which is a global property — every critical
        # path sits in the same fabric — so a design with very large spread is
        # not out-voted by moderate fanout signals.
        cell_spread_global = None
        if (isinstance(self.critical_path_spread_info, dict)
                and self.critical_path_spread_info.get("avg_distance") is not None):
            cell_spread_global = float(
                self.critical_path_spread_info["avg_distance"]
            )
        for name, fanout, _path_count in (self.high_fanout_nets or [])[:3]:
            p = {
                "high_fanout_nets": [(name, fanout)],
                "slack_ns": _slack_for_features,
            }
            if cell_spread_global is not None:
                p["cell_spread"] = cell_spread_global
            paths.append(p)

        if not paths:
            return []

        try:
            diagnosis = _classify_design(
                paths,
                design_context={
                    "initial_wns": self.initial_wns,
                    "high_fanout_nets": [
                        (n, f) for n, f, _ in (self.high_fanout_nets or [])
                    ],
                    "critical_path_spread_info": self.critical_path_spread_info,
                },
            )
        except Exception as e:
            logger.warning(f"pathology classifier failed: {e}")
            return []

        out: list[str] = []
        out.append("PATHOLOGY DIAGNOSIS (deterministic, from Phase 1 data):")
        out.append(f"  Primary: {diagnosis.primary_label}")
        if diagnosis.counts:
            non_primary = [
                f"{k}={v}" for k, v in sorted(
                    diagnosis.counts.items(), key=lambda kv: -kv[1])
                if k != diagnosis.primary_label
            ]
            if non_primary:
                out.append(f"  Secondary: {', '.join(non_primary)}")
        # Filter blocked recipes out of the recommendation list using the
        # per-design applicability table.  Gated off in contest mode: it is a
        # design-name conditional, and the hidden-benchmark rule forbids those
        # on the scored path.
        blocked_recipes: set = set()
        contest_mode = bool(getattr(self, "contest_mode", False))
        if (design_name and _recipe_safe_for is not None
                and not contest_mode
                and not _recipe_safe_for(design_name)):
            blocked_recipes.add("recipe_cell_replacement")
        recommended = [
            r for r in diagnosis.suggested_recipes_ordered
            if r not in blocked_recipes
        ]
        if recommended:
            out.append("  RECOMMENDED RECIPE ORDER (try in this order; "
                       "each is bounded and reverts on regression):")
            for i, r in enumerate(recommended, 1):
                out.append(f"    {i}. {r}")
        if blocked_recipes:
            out.append("  BLOCKED RECIPES (per-design applicability table; "
                       "DO NOT call):")
            for r in sorted(blocked_recipes):
                out.append(f"    - {r}")
        if diagnosis.notes:
            for n in diagnosis.notes:
                out.append(f"  Note: {n}")
        out.append(
            "  This diagnosis is DETERMINISTIC — derived from measured "
            "Phase 1 data, not LLM guesswork.  Strongly prefer recipes "
            "in the order shown unless live evidence contradicts."
        )
        # Store the diagnosis for downstream consumers (RAG records,
        # post-run analysis, optional prompt-injection in later iters).
        self.design_pathology = diagnosis
        return out

    def _router_unattempted_heavy_step(self) -> Optional[str]:
        """Return a one-liner describing the next unattempted heavy step
        from the active router plan, or None if all heavy steps have been
        attempted (or no plan is active).

        "Heavy step" = a vivado_place_design call (any directive) — this
        is an empirically-observed compliance gap seen in
        reruns: one design stopped at 16 min with 34 min budget
        remaining without ever invoking place_design Auto_1, the R4
        plan's step 3. Feature-based, not name-based: any router plan
        that has `vivado_place_design` in its `actions` triggers this
        check whenever the LLM signals stop.

        Returns the action's name + note string so the force-continue
        prompt can name the exact tool + directive to try next.
        """
        plan = getattr(self, "recipe_router_plan", None)
        if plan is None or not getattr(plan, "actions", None):
            return None
        # Have any place_design calls happened?
        seen_place_design = any(t == "vivado_place_design"
                                for t, _ in self._tool_calls_seen)
        if seen_place_design:
            return None
        # Find the first action.name == "vivado_place_design" in the plan.
        for action in plan.actions:
            if action.name == "vivado_place_design":
                return f"{action.name} ({action.note})"
        return None

    def should_inject_v05_router_nudge(self,
                                       current_gain_mhz: float,
                                       remaining_s: float
                                       ) -> tuple[bool, Optional[str]]:
        """Decide whether to issue a router-step-aware optimization nudge.

        Returns a Boolean decision and the description of the next unattempted
        heavy router step. The decision is true only when such a step exists,
        current gain is non-negative, the remaining budget meets
        ``V05_MIN_REMAINING_S``, and the session has not been budget-killed.
        The default five-minute threshold is pinned by that constant.
        """
        unattempted = self._router_unattempted_heavy_step()
        if unattempted is None:
            return (False, None)
        if current_gain_mhz < 0.0:                # clear regression
            return (False, unattempted)
        if remaining_s < V05_MIN_REMAINING_S:     # budget too low
            return (False, unattempted)
        if self._budget_killed:                   # session compromised
            return (False, unattempted)
        return (True, unattempted)


    @staticmethod
    def _is_model_unavailable_error(e: Exception) -> bool:
        """True iff the error indicates the MODEL is unavailable (not a transient
        rate-limit/timeout/5xx).  Only these warrant a model fallback.

        Reclassification note: the auth signatures
        ("unauthorized", "401", "403", "permission") were removed from this
        classifier — they are key-level outages (see
        _is_key_level_auth_error), not model outages.  The eval-day defect:
        a key-level 401 ("User not found") matched here and spuriously
        pinned the fallback model, which shares the same dead key.
        Delegates to optimizer.api_resilience.classify_api_error, whose
        key_auth check precedes the model check so "user not found" can
        never match the bare "not found" model signature.
        Mechanism extracted to optimizer/llm_runtime.py."""
        return _llm_rt.is_model_unavailable_error(e)

    @staticmethod
    def _is_key_level_auth_error(e: Exception) -> bool:
        """True iff the error is a KEY-level auth outage (the eval-day 401
        'User not found' storm class, observed twice in the same
        outage).  These must backoff-retry the SAME primary model —
        switching models cannot fix a dead key.
        Mechanism extracted to optimizer/llm_runtime.py."""
        return _llm_rt.is_key_level_auth_error(e)

    @staticmethod
    def _is_transient_api_error(e: Exception) -> bool:
        """True for retryable provider/API conditions (rate-limit, 5xx, network
        timeouts).  Disjoint from _is_model_unavailable_error: that one means
        the MODEL is gone (fall back immediately); this one means the provider
        is having a moment (retry, then fall back if it persists).
        Mechanism extracted to optimizer/llm_runtime.py."""
        return _llm_rt.is_transient_api_error(e)

    @staticmethod
    def _is_prompt_limit_error(e: Exception) -> bool:
        """True for the provider's per-request prompt-size rejection (the
        contest key's 402 'Prompt tokens limit exceeded').
        Mechanism extracted to optimizer/llm_runtime.py."""
        return _llm_rt.is_prompt_limit_error(e)

    def _log_prompt_limit_exhaustion(self, e: Exception) -> None:
        """Log prompt-limit exhaustion that persists after minimum-size pruning.

        A 402 response after pruning to the configured 8,000-token floor
        indicates that no viable LLM prompt fits the remaining allowance. The
        message is prominent and searchable so the resulting LLM-free execution
        is not mistaken for a normal optimization pass. Optimization continues
        using the deterministic scaffold rather than aborting.
        """
        self._key_exhaustion_suspect_count = getattr(
            self, "_key_exhaustion_suspect_count", 0) + 1
        logger.error(
            f"KEY-EXHAUSTION-SUSPECT #{self._key_exhaustion_suspect_count}: "
            f"402 prompt-limit persists after pruning to the 8k floor "
            f"({str(e)[:160]}) — provider key allowance is below any viable "
            f"prompt (credit likely exhausted). LLM is DEAD for this run; "
            f"any result from here is the LLM-free scaffold floor, NOT a "
            f"measured optimization outcome.")
        print("🚨 KEY-EXHAUSTION-SUSPECT: LLM calls failing 402 below the "
              "prune floor — treat this run's result as LLM-free.")

    @staticmethod
    def _estimate_message_tokens(m) -> int:
        """Conservative token estimate (chars/3 — JSON-heavy tool output runs
        denser than prose's chars/4; overestimating is the safe direction).
        Mechanism extracted to optimizer/llm_runtime.py."""
        return _llm_rt.estimate_message_tokens(m)

    def _estimate_messages_tokens(self) -> int:
        return sum(self._estimate_message_tokens(m) for m in self.messages)

    def _prune_conversation(self, target_tokens: int) -> int:
        """Drop middle messages so the conversation fits ~target_tokens.

        Keeps the head (system prompt + initial analysis/recipe message,
        PROMPT_PRUNE_HEAD_KEEP) and a contiguous SUFFIX of recent messages,
        replacing the dropped middle with one marker message carrying the
        essential state (best WNS). A contiguous suffix can never orphan a
        tool response from a LATER parent, but it may START with tool
        messages whose assistant parent was dropped — those are popped so
        the API never sees an orphaned tool_call_id. Returns #dropped.
        Mechanism extracted to optimizer/llm_runtime.py (pure; this
        delegate assigns the pruned list back to self.messages)."""
        new_messages, dropped = _llm_rt.prune_conversation(
            self.messages, target_tokens,
            head_keep=PROMPT_PRUNE_HEAD_KEEP,
            best_wns=getattr(self, "best_wns", None),
            estimate_fn=self._estimate_message_tokens)
        if dropped > 0:
            self.messages = new_messages
        return dropped

    def _chat_create(self, model: str):
        return self.openai.chat.completions.create(
            model=model,
            messages=self.messages,
            tools=self.tools,
            tool_choice="auto",
            max_tokens=4096,
            extra_body={"usage": {"include": True}},
        )

    def _create_completion_with_fallback(self):
        """Call the primary model with eval-day resilience (kill switch
        _api_resilience_enabled, default ON):
          - KEY-level auth outage (401 'User not found' class) -> backoff and
            retry the SAME primary model on a deadline-aware exponential
            schedule.  Never switch models: the fallback shares the same dead
            key (both observed reproductions of the outage confirm this);
          - model-UNAVAILABLE error (404/model_not_found/...) -> fall back
            once to FALLBACK_MODEL and pin it for the rest of the run
            (existing behavior preserved);
          - TRANSIENT error (429/5xx/timeout) -> backoff-retry the SAME model
            on the schedule; if it persists across
            TRANSIENT_FALLBACK_THRESHOLD consecutive failures, pin
            FALLBACK_MODEL (existing escape preserved — transient only,
            never for key_auth);
          - 402 prompt-limit -> existing prune-and-retry path (prompt
            guard), untouched, never enters backoff;
          - anything else propagates (the iteration loop logs and continues).

        Backoff sleeps go through self._backoff_sleep (injectable) and are
        computed by optimizer.api_resilience.compute_backoff_sleep, which is
        deadline-aware (never sleeps past remaining-wall-minus-guard; the
        finalize path always survives) and capped per episode
        (API_BACKOFF_EPISODE_CAP_S) — after the cap the error propagates to
        the existing 'LLM dead' path.  The loop is synchronous so a storm is
        absorbed HERE without returning to optimize() (no iteration burn, no
        conversation pollution — the loop side keeps its own hygiene)."""
        if not self._api_resilience_enabled or _classify_api_error is None:
            return self._create_completion_legacy()
        attempt = 0
        in_episode = False
        while True:
            try:
                resp = self._chat_create(self.model)
                self._consecutive_transient_failures = 0
                if in_episode:
                    # Single-line episode-resolution forensic log:
                    # one greppable line per episode so eval
                    # forensics can reconstruct a storm from the log alone.
                    self._api_last_episode_failures = attempt
                    logger.warning(
                        f"[api-resilience] episode "
                        f"#{self._api_error_episodes} resolved: recovered "
                        f"after {attempt} failure(s), "
                        f"{self._api_backoff_used_s:.0f}s backoff."
                    )
                self._api_backoff_used_s = 0.0   # episode resolved
                return resp
            except Exception as e:
                kind = _classify_api_error(f"{type(e).__name__}: {e}")
                # Prompt too large for the provider key (402 doom
                # loop): prune and retry — appending the error to the
                # conversation (the old path) only grows the prompt and
                # makes every later call fail.  Unchanged from legacy.
                if kind == "prompt_limit":
                    for _target in (PROMPT_PRUNE_TARGET // 2, 8_000):
                        _dropped = self._prune_conversation(_target)
                        logger.warning(
                            f"[prompt-guard] provider prompt-limit hit "
                            f"({str(e)[:120]}); pruned {_dropped} messages to "
                            f"~{_target} est. tokens and retrying.")
                        try:
                            resp = self._chat_create(self.model)
                            self._consecutive_transient_failures = 0
                            self._api_backoff_used_s = 0.0
                            return resp
                        except Exception as e2:
                            if not self._is_prompt_limit_error(e2):
                                raise
                            e = e2
                    self._log_prompt_limit_exhaustion(e)
                    raise
                if (kind == "model_unavailable"
                        and self.model != FALLBACK_MODEL):
                    logger.warning(
                        f"Primary model {self.model!r} unavailable ({e!r}); "
                        f"falling back to {FALLBACK_MODEL!r} for the rest of "
                        f"the run."
                    )
                    self.model = FALLBACK_MODEL
                    return self._chat_create(self.model)
                if kind in ("key_auth", "transient"):
                    if not in_episode:
                        in_episode = True
                        self._api_error_episodes += 1
                        self._api_backoff_used_s = 0.0
                    if kind == "transient":
                        self._consecutive_transient_failures += 1
                        if (self.model != FALLBACK_MODEL
                                and self._consecutive_transient_failures
                                >= TRANSIENT_FALLBACK_THRESHOLD):
                            logger.warning(
                                f"{self._consecutive_transient_failures} "
                                f"consecutive transient failures on "
                                f"{self.model!r} ({e!r}); pinning "
                                f"{FALLBACK_MODEL!r} for the rest of the run."
                            )
                            self.model = FALLBACK_MODEL
                            continue   # immediate retry on the fallback
                    dur = _compute_backoff_sleep(
                        attempt=attempt,
                        remaining_budget_s=self._budget_remaining(),
                        finalize_guard_s=self._api_backoff_finalize_guard_s,
                        backoff_used_s=self._api_backoff_used_s,
                    )
                    if dur is None:
                        # Episode cap reached or no room before the
                        # finalize reserve -> give up, propagate to the
                        # existing LLM-dead path.
                        # Record the episode's failure count
                        # for the loop-side forensic line.
                        self._api_last_episode_failures = attempt + 1
                        logger.warning(
                            f"[api-resilience] giving up after "
                            f"{self._api_backoff_used_s:.0f}s backoff "
                            f"(episode #{self._api_error_episodes}, "
                            f"attempt {attempt + 1}, {kind}) on "
                            f"{self.model!r}: {e!r}"
                        )
                        raise
                    logger.warning(
                        f"[api-resilience] {kind} API error on "
                        f"{self.model!r} (attempt {attempt + 1}, episode "
                        f"#{self._api_error_episodes}): {e!r}; backing off "
                        f"{dur:.1f}s and retrying the SAME model."
                    )
                    self._backoff_sleep(dur)
                    self._api_backoff_used_s += dur
                    self._total_backoff_s += dur
                    attempt += 1
                    continue
                raise

    def _create_completion_legacy(self):
        """Legacy call path, preserved byte-for-behavior under the
        _api_resilience_enabled kill switch (OFF):
          - model-unavailable OR auth error -> one-shot fallback-model pin
            (legacy lumped key-level 401/403 into model-unavailable — the
            exact eval-day misclassification the resilient path fixes);
          - transient -> single flat TRANSIENT_RETRY_BACKOFF_S retry, pin
            fallback after TRANSIENT_FALLBACK_THRESHOLD consecutive
            failures;
          - anything else propagates.
        Only change vs the pre-change code: the sleep goes through the
        injectable self._backoff_sleep (default time.sleep — identical
        behavior, testable without real sleeps)."""
        try:
            resp = self._chat_create(self.model)
            self._consecutive_transient_failures = 0
            return resp
        except Exception as e:
            # Prompt too large for the provider key (402 doom loop):
            # prune and retry — appending the error to the conversation (the
            # old path) only grows the prompt and makes every later call fail.
            if self._is_prompt_limit_error(e):
                for _target in (PROMPT_PRUNE_TARGET // 2, 8_000):
                    _dropped = self._prune_conversation(_target)
                    logger.warning(
                        f"[prompt-guard] provider prompt-limit hit "
                        f"({str(e)[:120]}); pruned {_dropped} messages to "
                        f"~{_target} est. tokens and retrying.")
                    try:
                        resp = self._chat_create(self.model)
                        self._consecutive_transient_failures = 0
                        return resp
                    except Exception as e2:
                        if not self._is_prompt_limit_error(e2):
                            raise
                        e = e2
                self._log_prompt_limit_exhaustion(e)
                raise
            if self.model != FALLBACK_MODEL and (
                    self._is_model_unavailable_error(e)
                    or self._is_key_level_auth_error(e)):
                logger.warning(
                    f"Primary model {self.model!r} unavailable ({e!r}); "
                    f"falling back to {FALLBACK_MODEL!r} for the rest of the run."
                )
                self.model = FALLBACK_MODEL
                return self._chat_create(self.model)
            if self._is_transient_api_error(e):
                self._consecutive_transient_failures += 1
                logger.warning(
                    f"Transient API error #{self._consecutive_transient_failures} "
                    f"on {self.model!r} ({e!r}); retrying once after backoff."
                )
                self._backoff_sleep(TRANSIENT_RETRY_BACKOFF_S)
                try:
                    resp = self._chat_create(self.model)
                    self._consecutive_transient_failures = 0
                    return resp
                except Exception as e2:
                    if (self.model != FALLBACK_MODEL
                            and self._consecutive_transient_failures
                            >= TRANSIENT_FALLBACK_THRESHOLD):
                        logger.warning(
                            f"{self._consecutive_transient_failures} consecutive "
                            f"transient failures on {self.model!r} ({e2!r}); "
                            f"pinning {FALLBACK_MODEL!r} for the rest of the run."
                        )
                        self.model = FALLBACK_MODEL
                        return self._chat_create(self.model)
                    raise
            raise

    def _llm_cost_breached(self) -> bool:
        """β circuit-breaker: has this attempt's LLM spend crossed the
        effective in-attempt exit?  Threshold <= 0 disables (never breached)."""
        return (self.llm_cost_exit_usd > 0
                and self.total_cost >= self.llm_cost_exit_usd)

    def _write_cost_ledger(self) -> None:
        """Persist the incremental LLM cost ledger.

        The ledger is seeded with zero cost and updated after every API cost
        accrual so crashes cannot erase recorded spend. Writes use atomic
        replacement to prevent truncated files. Errors are logged and swallowed
        because auditing must not terminate optimization.
        """
        _llm_rt.write_cost_ledger(
            getattr(self, "run_dir", None),
            getattr(self, "total_cost", 0.0),
            getattr(self, "llm_call_count", 0))

    async def get_completion(self) -> tuple[str, bool]:
        """Get LLM completion and process it."""
        try:
            # Cost circuit-breaker, pre-call gate.  The iteration loop's cost
            # exit only runs between outer iterations, but process_message
            # recurses back here after every tool round — so a single
            # iteration can chain many model calls with no cost check between
            # them, which is how a run crosses its budget cap and scores zero.
            # Refuse to issue another call once spend crosses the effective
            # exit; the loop's own breach checks then finalize with the banked
            # best, rather than abandoning an operation mid-flight.
            if self._llm_cost_breached():
                logger.info(
                    f"[cost-exit] refusing LLM call: spend "
                    f"${self.total_cost:.4f} >= exit "
                    f"${self.llm_cost_exit_usd:.2f}")
                return (f"[cost-exit] LLM spend ${self.total_cost:.2f} reached "
                        f"the in-attempt exit ${self.llm_cost_exit_usd:.2f}; "
                        f"no further LLM calls this attempt.", False)

            self.llm_call_count += 1
            logger.info(f"LLM API call #{self.llm_call_count}")

            # Prompt-size guard: keep every request under the eval
            # key's per-request prompt-token limit so the LLM stays available
            # for the whole run instead of 402-ing out mid-budget.
            _est = self._estimate_messages_tokens()
            if _est > PROMPT_TOKEN_SOFT_LIMIT:
                _dropped = self._prune_conversation(PROMPT_PRUNE_TARGET)
                logger.info(
                    f"[prompt-guard] conversation ~{_est} est. tokens > soft "
                    f"limit {PROMPT_TOKEN_SOFT_LIMIT}; pruned {_dropped} middle "
                    f"messages -> ~{self._estimate_messages_tokens()} est. tokens")

            # Request usage accounting from OpenRouter (with model fallback)
            response = self._create_completion_with_fallback()
            
            # Validate response immediately
            if response is None:
                raise ValueError("API returned None response")
            
            # Extract token usage information from OpenRouter
            if hasattr(response, 'usage') and response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                
                # Update cumulative totals
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_tokens += total_tokens
                
                # Get actual cost from OpenRouter (in credits/dollars)
                call_cost = 0.0
                if hasattr(response.usage, 'cost') and response.usage.cost is not None:
                    call_cost = float(response.usage.cost)
                    self.total_cost += call_cost
                else:
                    logger.warning("OpenRouter did not provide cost information")
                
                # Extract additional usage details if available
                cached_tokens = 0
                reasoning_tokens = 0
                if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                    if hasattr(response.usage.prompt_tokens_details, 'cached_tokens'):
                        cached_tokens = response.usage.prompt_tokens_details.cached_tokens or 0
                if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
                    if hasattr(response.usage.completion_tokens_details, 'reasoning_tokens'):
                        reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens or 0
                
                # Store details for this call
                call_detail = {
                    "call_number": self.llm_call_count,
                    "iteration": self.iteration,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost": call_cost,
                    "cached_tokens": cached_tokens,
                    "reasoning_tokens": reasoning_tokens
                }
                self.api_call_details.append(call_detail)
                
                # Log token usage
                cache_info = f", Cached: {cached_tokens:,}" if cached_tokens > 0 else ""
                reasoning_info = f", Reasoning: {reasoning_tokens:,}" if reasoning_tokens > 0 else ""
                cost_info = f" | Cost: ${call_cost:.4f}" if call_cost > 0 else ""
                
                logger.info(f"API call #{self.llm_call_count} - Tokens: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total{cost_info}{cache_info}{reasoning_info}")
                print(f"[API Call #{self.llm_call_count}] Tokens: {total_tokens:,} (Prompt: {prompt_tokens:,}, Completion: {completion_tokens:,}{cache_info}{reasoning_info}){cost_info}")
            else:
                logger.warning("No usage information in API response")

            # Persist the incremental spend ledger after
            # every API call so a later crash cannot lose the accrued cost
            # (never raises — see _write_cost_ledger).
            self._write_cost_ledger()

            # Debug logging
            if self.debug:
                logger.debug(f"Response type: {type(response)}")
                logger.debug(f"Response: {response}")
            
            # Check if response has error
            if hasattr(response, 'error') and response.error:
                raise ValueError(f"API returned error: {response.error}")
            
            return await self.process_response(response)
            
        except Exception as e:
            logger.error(f"Error in get_completion: {e}")
            logger.error(f"Number of messages in conversation: {len(self.messages)}")
            if self.messages:
                logger.error(f"Last message: {self.messages[-1]}")
            raise
    
    # ------- wall-time budget helpers -------
    def _budget_remaining(self) -> float:
        """Seconds left before the wall-time deadline. inf when no cap set."""
        if self._budget_deadline is None:
            return float("inf")
        return max(0.0, self._budget_deadline - time.time())

    def _budget_exhausted(self) -> bool:
        """True iff we're past the wall-time deadline (no margin)."""
        return self._budget_remaining() <= 0.0

    def _strategy_fits_in_budget(self, estimated_seconds: float,
                                  label: str = "<unknown>") -> bool:
        """Check whether estimated work is likely to finish before the budget
        deadline.

        Returns true when no budget is configured. A rejected estimate is
        logged and recorded for the optimization summary.
        """
        if self._budget_deadline is None:
            return True
        remaining = self._budget_remaining()
        if estimated_seconds > remaining:
            self._strategies_skipped_budget.append(
                f"{label} (need ~{estimated_seconds:.0f}s, have {remaining:.0f}s)"
            )
            logger.info(
                f"Budget guard: skipping {label} (need ~{estimated_seconds:.0f}s, "
                f"only {remaining:.0f}s remaining of {self.max_wall_seconds:.0f}s)"
            )
            return False
        return True

    def _estimate_tool_runtime(self, tool_name: str,
                                is_risky: Optional[bool] = None) -> float:
        """Estimate a tool call's runtime in seconds.

        Uses the maximum of the five most recent recorded runtimes, or a
        conservative fallback when no history exists. Estimates intentionally
        favor skipping work that is unlikely to finish. The risk hint overrides
        automatic classification by tool name when provided.
        """
        history = self._tool_runtime_history.get(tool_name, [])
        if history:
            self._last_estimate_provenance = _gl.PROV_HISTORY
            return max(history[-5:])
        self._last_estimate_provenance = _gl.PROV_UNKNOWN
        if is_risky is None:
            is_risky = tool_name in RISKY_VIVADO_TOOLS
        if is_risky:
            return self._no_history_risky_estimate_s()
        return DEFAULT_CHEAP_RUNTIME_S

    def _no_history_risky_estimate_s(self) -> float:
        """Estimate a risky tool's runtime from design size when no history
        exists.

        The estimate is the smaller of the fixed risky-tool fallback and 1.3
        times the predicted place-and-route runtime, with a useful-work floor;
        the 1.3 factor provides conservative model margin. Place-and-route time
        also serves as a conservative ceiling for physical optimization.

        Unknown size, a nonpositive prediction, an exception, or a disabled
        size-aware estimate returns the fixed fallback.
        """
        _flag = os.environ.get("FPL26_SIZE_AWARE_TOOL_EST", "").strip().lower()
        if _flag in ("0", "false", "no", "off"):
            self._last_estimate_provenance = _gl.PROV_CONSTANT
            return DEFAULT_RISKY_RUNTIME_S
        cells = getattr(self, "_input_cell_count", None)
        if not cells or cells <= 0:
            self._last_estimate_provenance = _gl.PROV_CONSTANT
            return DEFAULT_RISKY_RUNTIME_S
        try:
            from optimizer.deep_replace_sibling import predict_place_route_s
            est, why = predict_place_route_s(primitive_cells=int(cells))
            if not est or est <= 0:
                self._last_estimate_provenance = _gl.PROV_CONSTANT
                return DEFAULT_RISKY_RUNTIME_S
            sized = min(DEFAULT_RISKY_RUNTIME_S,
                        max(est * 1.3, MIN_USEFUL_TOOL_SECONDS))
            if sized < DEFAULT_RISKY_RUNTIME_S:
                logger.info(
                    f"[tool-est] no history for a risky tool; sizing from the "
                    f"design instead of the blind {DEFAULT_RISKY_RUNTIME_S:.0f}s "
                    f"constant: {sized:.0f}s ({why}). A prediction may schedule "
                    f"work, never refuse it.")
            self._last_estimate_provenance = (
                _gl.PROV_MODEL if sized < DEFAULT_RISKY_RUNTIME_S else _gl.PROV_CONSTANT)
            return sized
        except Exception as e:  # pragma: no cover — defensive
            logger.warning(f"[tool-est] size-aware estimate unavailable "
                           f"({type(e).__name__}); using the constant.")
            self._last_estimate_provenance = _gl.PROV_CONSTANT
            return DEFAULT_RISKY_RUNTIME_S

    def _record_tool_runtime(self, tool_name: str, elapsed: float) -> None:
        """Record a completed tool call's elapsed time for future estimates.

        Durations below one second are ignored because they usually indicate a
        short-circuited call and would bias estimates downward.
        """
        if elapsed < 1.0:
            return
        # Observed leg of the predicted-vs-observed calibration pair. Joining these
        # against the budget_skip rows is what lets a blind constant be replaced by a
        # 2-4 param cost model under leave-one-design-out (fitting a COST model, which
        # the methodology invariant permits).
        _gl.emit("tool_runtime", _gl.VERDICT_OBSERVE,
                 design=getattr(self, "_design_label", None),
                 run_id=getattr(self, "run_id", None),
                 iteration=getattr(self, "iteration", None),
                 tool=tool_name, observed_s=elapsed,
                 site="dcp_optimizer._record_tool_runtime")
        bucket = self._tool_runtime_history.setdefault(tool_name, [])
        bucket.append(elapsed)
        if len(bucket) > 20:
            del bucket[: len(bucket) - 20]

    def _deadline_aware_timeout(self, tool_name: Optional[str] = None,
                                arguments: Optional[dict] = None
                                ) -> Optional[float]:
        """Compute the runtime timeout for the next asynchronous tool call.

        Returns `None` when no budget is configured, `0.0` when too little
        useful time remains, and otherwise the allowed runtime in seconds. The
        caller must skip a call when the result is zero.

        The budget deadline already excludes the finalization reserve, so this
        function must not subtract that reserve again. When the polish reserve
        is armed, speculative operations that destroy routed state are capped
        at the reserve boundary. Polish, finalization, and non-destroying
        banking operations remain unaffected.
        """
        # Remember whether THIS computation fence-capped the
        # window, so the TimeoutError handler can label the kill cause.
        self._last_timeout_fence_capped = False
        if self._budget_deadline is None:
            return None
        remaining = self._budget_remaining()
        if remaining < MIN_USEFUL_TOOL_SECONDS:
            return 0.0
        if tool_name is not None:
            reserve = self._polish_reserve_armed_s()
            if (reserve > 0.0
                    and self._is_risky(tool_name, arguments)
                    and self.is_routed_state_destroying(tool_name, arguments)):
                self._last_timeout_fence_capped = True
                return max(0.0, remaining - reserve)
        return remaining

    def _is_risky(self, tool_name: str, arguments: Optional[dict] = None) -> bool:
        """Classify a tool call as risky (subject to estimate gating).

        - All tools in RISKY_VIVADO_TOOLS are always risky.
        - vivado_run_tcl is risky only when its tcl payload mentions one of
          RISKY_TCL_SUBSTRINGS (place_design, route_design, etc).
        """
        if tool_name in RISKY_VIVADO_TOOLS:
            return True
        if tool_name == "vivado_run_tcl" and arguments:
            cmd = arguments.get("command") or arguments.get("tcl_command") or ""
            cmd_low = str(cmd).lower()
            return any(s in cmd_low for s in RISKY_TCL_SUBSTRINGS)
        return False

    def is_routed_state_destroying(self, tool_name: str,
                                   arguments: Optional[dict] = None) -> bool:
        """Instance view of the module-level pure classifier,
        bound to the harness-tracked _design_routed_state.  See the
        module-level is_routed_state_destroying for the three-case scope
        and the incremental-route exclusion."""
        return is_routed_state_destroying(tool_name, arguments,
                                          self._design_routed_state)

    async def _capture_constraint_fingerprint(self, label: str):
        """Snapshot the design's timing constraints (constraint guard).

        Runs two read-only reports and digests them. Returns a
        ConstraintFingerprint; ``captured=False`` when either report could not
        be obtained, which the comparison surfaces as UNVERIFIED rather than
        silently claiming 'unchanged'.

        Read-only and best-effort by design: this must never be able to break
        or slow a run. Both reports are cheap (no timing update forced)."""
        from optimizer.constraint_guard import build_fingerprint
        clocks = excs = None
        try:
            clocks = await self.call_tool(
                "vivado_run_tcl",
                {"command": "report_clocks -return_string", "timeout": 120.0})
            if _looks_like_tool_error(clocks):
                clocks = None
        except Exception:
            clocks = None
        try:
            excs = await self.call_tool(
                "vivado_run_tcl",
                {"command": "report_exceptions -return_string", "timeout": 120.0})
            if _looks_like_tool_error(excs):
                excs = None
        except Exception:
            excs = None
        fp = build_fingerprint(clocks, excs)
        logger.info(f"[constraint-guard] {label} fingerprint: "
                    f"captured={fp.captured} clocks={fp.clock_count} "
                    f"exceptions={fp.exception_count}")
        return fp

    async def _verify_constraints_unchanged(self) -> bool:
        """Compare ship-time constraints with the baseline constraints.

        Returns true when the constraints match or cannot be verified. This
        check deliberately fails open: mismatches and verification failures are
        logged and recorded, but never block shipping, because an uncertain
        monitoring result must not invalidate an otherwise valid output.
        """
        base = getattr(self, "_constraint_fp_baseline", None)
        if base is None:
            logger.info("[constraint-guard] no baseline fingerprint; "
                        "constraint integrity UNVERIFIED for this run.")
            return True
        try:
            now = await self._capture_constraint_fingerprint("ship")
        except Exception as e:
            logger.warning(f"[constraint-guard] ship capture failed: {e!r}")
            return True
        changed, why = base.differs_from(now)
        if changed:
            logger.error(
                f"[constraint-guard] *** TIMING CONSTRAINTS CHANGED *** {why}. "
                f"XDC/timing-constraint edits are DISQUALIFYING under the "
                f"contest rules — this DCP must be reviewed before submission.")
            try:
                self.tool_call_details.append({
                    "tool_name": "constraint_guard",
                    "status": "CONSTRAINTS_CHANGED",
                    "detail": why,
                    # See the note at the refused_constraint_edit append: the
                    # summary printer sums this key unconditionally.
                    "elapsed_time": 0.0,
                })
            except Exception:
                pass
            return False
        logger.info(f"[constraint-guard] ship check: {why}")
        return True

    async def _maybe_plan_critique(self, tool_name: str,
                                   arguments: Optional[dict]) -> None:
        """Request an advisory critique before an expensive FPGA tool operation.

        A valid reply is appended to the planner messages but never skips,
        cancels, or rewrites the pending operation. Malformed or empty replies
        are discarded. The critique runs only with sufficient LLM-cost headroom
        and is a strict no-op when the feature flag is disabled.
        """
        from optimizer.plan_critic import (CRITIC_MAX_TOKENS, CRITIC_SYSTEM,
                                           build_critic_prompt,
                                           parse_critic_verdict,
                                           should_critique)
        if not getattr(self, "plan_critic_enabled", False):
            return
        remaining = (self._budget_remaining()
                     if self._budget_deadline is not None else float("inf"))
        run, why = should_critique(
            enabled=True,
            tool_name=tool_name,
            arguments=arguments,
            calls_made=getattr(self, "_plan_critic_calls", 0),
            max_calls=getattr(self, "plan_critic_max_calls", 4),
            spent_usd=float(self.total_cost or 0.0),
            # The live per-attempt LLM budget is llm_cost_exit_usd (the spend
            # at which the attempt bails). 0/None means "no budget known" and
            # should_critique fails CLOSED on it — deliberate: firing without
            # a known budget could push beta past its 10%-of-alpha cap.
            budget_usd=(self.llm_cost_exit_usd
                        if getattr(self, "llm_cost_exit_usd", 0) else None),
            beta_headroom_frac=getattr(self, "plan_critic_beta_headroom_frac",
                                       0.7),
            remaining_s=remaining,
            min_remaining_s=600.0,
        )
        if not run:
            logger.debug(f"plan-critic: skipped ({why})")
            return
        prompt = build_critic_prompt(
            tool_name=tool_name,
            arguments=arguments,
            initial_wns=getattr(self, "initial_wns", None),
            current_wns=self.best_wns,
            failing_endpoints=getattr(self, "initial_failing_endpoints", None),
            router_rule=getattr(self, "_router_rule_label", None),
            router_why=getattr(self, "_router_rule_why", None),
            # LLM-invisibility contract: exclude
            # recipe-pass machinery from the critic's model-visible
            # window — the pass is log-only by contract, and its
            # vivado_run_tcl/open/restart entries would otherwise leak
            # what the pass did into a model prompt on fired runs.
            tried=[d.get("tool_name", "")
                   for d in self.tool_call_details
                   if not d.get("recipe_pass_internal")][-8:],
            remaining_s=(None if remaining == float("inf") else remaining),
        )
        model = getattr(self, "plan_critic_model", PLAN_CRITIC_DEFAULT_MODEL)
        try:
            resp = await asyncio.to_thread(
                self.openai.chat.completions.create,
                model=model,
                messages=[{"role": "system", "content": CRITIC_SYSTEM},
                          {"role": "user", "content": prompt}],
                max_tokens=CRITIC_MAX_TOKENS,
                extra_body={"usage": {"include": True}},
            )
        except Exception as e:
            logger.info(f"plan-critic: call failed ({e!r}); continuing.")
            return
        self._plan_critic_calls = getattr(self, "_plan_critic_calls", 0) + 1
        try:
            text = resp.choices[0].message.content
        except Exception:
            text = None
        # Best-effort beta accounting: the critic's spend is real and must be
        # charged, or the headroom gate above would be measuring a lie.
        try:
            usage = getattr(resp, "usage", None)
            cost = float(getattr(usage, "cost", 0.0) or 0.0)
            if cost:
                self.total_cost += cost
        except Exception:
            pass
        v = parse_critic_verdict(text)
        note = v.advisory_note()
        if not note:
            logger.info("plan-critic: reply unparseable; note dropped.")
            return
        logger.info(f"plan-critic [{model}] {v.verdict}: {v.why[:160]}")
        self.messages.append({"role": "user", "content": note})

    def _should_skip_for_budget(self, tool_name: str,
                                 arguments: Optional[dict] = None
                                 ) -> tuple[bool, str]:
        """Decide whether budget pressure requires skipping a tool call.

        A call is skipped when the reserve-protected deadline has passed, when
        less than the minimum useful runtime remains, or when a risky call's
        estimated runtime exceeds the remaining window. Risk classification
        includes known risky tools and generic script execution containing a
        risky payload.
        """
        if self._budget_deadline is None:
            return (False, "")
        remaining = self._budget_remaining()
        if remaining <= 0.0:
            return (True, "deadline_passed")
        if remaining < MIN_USEFUL_TOOL_SECONDS:
            return (True,
                    f"only_{remaining:.0f}s_remain_below_min_useful_"
                    f"{MIN_USEFUL_TOOL_SECONDS:.0f}s")
        if self._is_risky(tool_name, arguments):
            est = self._estimate_tool_runtime(tool_name, is_risky=True)
            # Typed gate ledger. Records the operands and, crucially, the
            # provenance of the estimate: a refusal bounded by history (measured on this
            # design, this run) is legitimate; one bounded by a constant or a model is
            # the defect family that cost this project ~+55 points. Default OFF and
            # cannot raise — see optimizer/gate_log.py.
            _gl.emit(
                "budget_skip",
                _gl.VERDICT_REFUSE if est > remaining else _gl.VERDICT_ALLOW,
                design=getattr(self, "_design_label", None),
                run_id=getattr(self, "run_id", None),
                iteration=getattr(self, "iteration", None),
                tool=tool_name,
                predicted_s=est,
                threshold_s=remaining,
                remaining_wall_s=remaining,
                provenance=getattr(self, "_last_estimate_provenance", None),
                best_wns_ns=(self.best_wns if self.best_wns != float("-inf") else None),
                reason_code=("estimated_exceeds_remaining" if est > remaining
                             else "fits"),
                site="dcp_optimizer._should_skip_for_budget",
            )
            if est > remaining:
                return (True,
                        f"estimated_{est:.0f}s_exceeds_remaining_{remaining:.0f}s")
        return (False, "")


    def _maybe_arm_wall_handback(self, reason: str) -> None:
        """Arm the saturation early-exit reason — guarded.

        Called only from the three sanctioned existing signal sites:
          (a) ILS no-improve stop (run_ils_polish's futility break, detected
              post-hoc via the result notes in _ils_polish_body),
          (b) LASTMILE reject verdict (_lastmile_polish_after_ils),
          (c) _should_skip_for_budget budget-kill (call_tool skip path).
        GUARD (locked): never arm before a banked accept exists
        (_best_valid_dcp) — a trimmed zero is worse than a slow zero.
        First reason wins (the earliest saturation verdict is the one that
        matters for gamma attribution).  Arming is observational when the
        --wall-handback kill switch is OFF: only _wall_handback_break_due()
        acts on it, and that predicate requires _wall_handback_enabled.
        """
        if self._exit_early_reason is not None:
            return
        if self._best_valid_dcp is None:
            logger.debug(f"wall-handback: signal '{reason}' NOT armed "
                         f"(no banked accept yet — never trim before a "
                         f"banked accept).")
            return
        self._exit_early_reason = reason
        logger.info(
            f"wall-handback: saturation signal armed ({reason}); "
            + ("acting at the next loop check (kill switch ON)."
               if self._wall_handback_enabled
               else "observe-only (kill switch OFF).")
        )

    def _wall_economics_window_s(self) -> Optional[float]:
        """Observation window = the CHEAPEST completed heavy move this run.

        A measurement, not a constant: if a full heavy move's worth of wall
        has passed with no gain, the maximum-likelihood forward rate is 0.
        Returns None when no heavy move has completed yet (fail open — the
        rule then cannot fire).
        """
        return min(self._heavy_move_seconds) if self._heavy_move_seconds else None

    def _maybe_arm_wall_economics_stop(self) -> None:
        """Arm wall-clock handback when marginal improvement no longer justifies
        its time penalty.

        The decision follows directly from the scoring function and uses no
        fitted threshold. It can arm only after an accepted result is banked,
        preserves the first handback reason, and remains observational when
        wall handback is disabled.

        An active handback exits the LLM loop through the shared finalization
        path, so deterministic polish stages still run once.
        """
        if self._exit_early_reason is not None:      # first reason wins
            return
        if self.best_wns == float("-inf") or self.initial_wns is None:
            return
        best_fmax = self.calculate_fmax(self.best_wns, self.clock_period)
        init_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
        if best_fmax is None or init_fmax is None:
            return
        try:
            from optimizer.wall_economics import should_stop_for_wall
        except Exception:
            return
        # Time since the last banked improvement; by construction the
        # realized gain across that window is zero (banking paths are what
        # update last_improvement_time), so gain_in_window_mhz stays 0.0.
        since = time.time() - (self.last_improvement_time or self.start_time)
        stop, reason = should_stop_for_wall(
            alpha_mhz=best_fmax - init_fmax,
            remaining_s=self._budget_remaining(),
            since_last_gain_s=since,
            observation_window_s=self._wall_economics_window_s(),
        )
        if stop:
            logger.info(f"[wall-economics] {reason}")
            self._maybe_arm_wall_handback(f"wall_economics:{reason[:160]}")

    def _wall_handback_break_due(self) -> bool:
        """True iff the LLM iteration loop should exit early to
        the finalize tail and hand the remaining wall back to the wrapper.
        Requires BOTH the --wall-handback kill switch AND an armed
        saturation reason; with the switch OFF (default) this is always
        False — behavior byte-identical to today."""
        return bool(self._wall_handback_enabled and self._exit_early_reason)

    def _deep_wns_tail_reserve_effective_s(self) -> float:
        """Computes the effective reserve for the deep-WNS deterministic tail.

        A value of 0 disables the reserve, a value between 0 and 1 is a
        fraction of the wall limit, and a value of at least 1 is seconds.
        Fractional reserves require a wall limit; absolute reserves remain
        unchanged but are inert without a deadline. When the wall limit is
        known, the reserve is capped by TAIL_RESERVE_WALL_CLAMP_FRAC to
        preserve time for the main optimization loop.
        """
        v = self._deep_wns_tail_reserve_s
        if v <= 0.0:
            return 0.0
        if v < 1.0:
            if self.max_wall_seconds is None:
                return 0.0
            base = v * float(self.max_wall_seconds)
        else:
            base = v
        if self.max_wall_seconds is not None:
            cap = TAIL_RESERVE_WALL_CLAMP_FRAC * float(self.max_wall_seconds)
            if base > cap:
                if not self._tail_reserve_clamp_logged:
                    logger.warning(
                        f"[tail-reserve] WALL CLAMP: requested reserve "
                        f"{base:.0f}s exceeds "
                        f"{TAIL_RESERVE_WALL_CLAMP_FRAC:.3f} x wall "
                        f"({float(self.max_wall_seconds):.0f}s) = "
                        f"{cap:.0f}s — clamping to {cap:.0f}s "
                        f"(wave-3 refutation: reserve >= wall truncates "
                        f"the recipe before its big move; proven arm = "
                        f"2400s on the 3500s eval wall).")
                    self._tail_reserve_clamp_logged = True
                base = cap
        # Reduced-tail cap: set only after a gated fire that actually banked a
        # candidate.  None — the default, and the only value a flag-off run can
        # see — leaves every path above unchanged.  It can only ever reduce the
        # reserve, never raise it.
        rfd_cap = self._recipe_first_deep_tail_reserve_cap_s
        if rfd_cap is not None and rfd_cap < base:
            return rfd_cap
        return base

    def _tail_reserve_break_due(self) -> bool:
        """Returns whether the LLM loop should yield the remaining wall time to
        the deterministic exit tail.

        The reserve triggers only with a positive effective reserve, a wall
        deadline, and a finite banked WNS. The measured cell count must exceed
        the live ILS size limit; an unknown count keeps the reserve disarmed.
        The current best WNS must remain at or below the live deep-WNS
        threshold so recovered near-met states stay in the normal flow.
        Remaining wall time must be within the effective reserve window. The
        best WNS must also be stagnant for the configured interval; zero
        disables this guard, and a missing reference clock fails safe by
        preventing the break.
        """
        reserve = self._deep_wns_tail_reserve_effective_s()
        if reserve <= 0.0:
            return False
        if self._budget_deadline is None:
            return False
        # V2 SIZE GATE (ILS-size-gated class only; unknown -> not armed).
        max_cells = getattr(self._ils_polish_cfg, "max_cells", None)
        if not self._input_cell_count or not max_cells:
            return False
        if self._input_cell_count <= max_cells:
            return False
        if self.best_wns is None or self.best_wns == float("-inf"):
            return False
        if self.best_wns > self._tail_ctrl_deep_wns_ns:
            return False
        if self._budget_remaining() > reserve:
            return False
        # V2 STAGNATION GUARD (only fire on a stagnant loop).
        stagnant_s = self._tail_reserve_stagnant_s
        if stagnant_s > 0.0:
            ref = self.last_improvement_time or getattr(
                self, "start_time", None)
            if ref is None:
                return False  # no clock yet — cannot prove stagnation
            if (time.time() - ref) < stagnant_s:
                return False  # loop improved recently — mid-recipe
        return True

    def _ensure_decision_tracer(self) -> None:
        """Lazy-init the decision tracer once run_dir exists.

        Safe to call any number of times.  Failures are swallowed —
        telemetry must never block the optimizer.
        """
        if self._decision_tracer is not None:
            return
        if _DecisionTracer is None:
            return
        try:
            run_dir = getattr(self, "run_dir", None)
            if not run_dir:
                return
            self._decision_tracer = _DecisionTracer(run_dir)
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug(f"decision_tracer init failed: {exc}")
            self._decision_tracer = None

    def _emit_decision(self, record: dict) -> None:
        """Emits a best-effort decision record without propagating errors.

        Missing design, iteration, and model fields are filled from instance
        state when available.
        """
        self._ensure_decision_tracer()
        if self._decision_tracer is None:
            return
        try:
            record.setdefault("design",
                              getattr(self, "_design_name_for_memory", None))
            record.setdefault("iteration", getattr(self, "iteration", None))
            record.setdefault("model", getattr(self, "model", None))
            record.setdefault(
                "best_valid_token",
                getattr(self, "_best_valid_token", None),
            )
            self._decision_tracer.emit(record)
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug(f"decision_tracer emit failed: {exc}")

    def _classify_tool_payload(self, payload, *, context: str = "call_tool"):
        """Wrap optimizer.tool_errors.classify_tool_error so callers
        can use it without importing the module.  Returns None when
        the module is unavailable or the payload is not an error."""
        if _classify_tool_error is None:
            return None
        try:
            return _classify_tool_error(payload, context=context)
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug(f"classify_tool_error raised: {exc}")
            return None

    def _ensure_path_guard(self, output_dcp: Optional[Path] = None) -> None:
        """Lazy-init the DCP write-path PathGuard.

        Safe to call any number of times; only the first call wires the
        guard.  Roots:
          - self.run_dir (best_valid mirror + intermediates)
          - output_dcp.parent (ship target)
          - submission/dcps (final packaging tree), best-effort
        Mode comes from self._path_guard_mode (default "audit").  All
        failures are swallowed — the guard is defense-in-depth, never
        a critical-path dependency.
        """
        if self._path_guard is not None:
            return
        if _PathGuard is None:
            return
        try:
            roots: list[Path] = []
            run_dir = getattr(self, "run_dir", None)
            if run_dir is not None:
                roots.append(Path(run_dir))
            if output_dcp is not None:
                roots.append(Path(output_dcp).parent)
            # Submission/dcps tree — resolve relative to repo root if
            # discoverable; otherwise skip.  This is the packaged-DCP
            # destination tree; non-existence is fine, allow_root works
            # on un-created paths.
            try:
                script_dir = Path(__file__).resolve().parent
                subm = script_dir / "submission" / "dcps"
                roots.append(subm)
            except Exception:
                pass
            mode = self._path_guard_mode if self._path_guard_mode in (
                "audit", "enforce"
            ) else "audit"
            self._path_guard = _PathGuard(roots, mode=mode)
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug(f"path_guard init failed: {exc}")
            self._path_guard = None

    def _path_guard_check(self, path, *, context: str = "") -> bool:
        """Checks a path with PathGuard and emits the audit outcome as a decision
        event.

        Returns true for an allowed path or when the guard is uninitialized,
        preserving permissive production behavior. In audit mode, an
        out-of-root path returns false and is added to the recorded violations.
        Audit failures never raise, and every checked checkpoint target
        produces a trace event regardless of mode.
        """
        # If the guard isn't ready, treat as permissive but trace.
        guard = self._path_guard
        allowed = True
        try:
            if guard is not None:
                allowed = bool(guard.check(path, context=context))
                if not allowed:
                    # Record violation locally so summary printers can
                    # report aggregate audit findings without grepping
                    # decisions.jsonl.
                    self._path_guard_violations.append({
                        "path": str(path),
                        "context": context,
                        "ts": time.time(),
                    })
        except Exception as exc:  # pragma: no cover — defensive
            # PathGuardError in enforce mode bubbles up; everything else
            # is swallowed.  Re-raised only when the configured mode is
            # "enforce", so blocked writes are never silently swallowed.
            if guard is not None and getattr(guard, "mode", "audit") == "enforce":
                raise
            logger.debug(f"path_guard.check raised: {exc}")
        # Emit a trace event regardless of mode.  Decision tracer is
        # best-effort, so this never blocks the caller.
        try:
            self._emit_decision({
                "decision_source": "executor",
                "phase": "path_guard",
                "action_label": "path_guard_check",
                "tool_name": None,
                "notes": context,
                "output_dcp_path": str(path),
                "path_guard_allowed": allowed,
                "path_guard_mode": (
                    getattr(guard, "mode", None) if guard is not None
                    else None
                ),
            })
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug(f"path_guard trace emit failed: {exc}")
        return allowed

    def _record_ship_lineage(
        self,
        *,
        dcp_path: Path,
        edif_ok: bool,
        inherit_best_valid: bool = False,
        fallback_source: Optional[str] = None,
        baseline: bool = False,
        no_improvement: bool = False,
        hard_fail: bool = False,
        branch: str = "",
    ) -> dict:
        """Tag self._ship_lineage with the lineage of the final shipped DCP.

        Exactly one of the following must be passed truthy:
          - inherit_best_valid (fast path / stale-mirror) — copies current
            best-valid lineage and bumps role to "ship"
          - baseline — explicit baseline fallback after a failed optimized
            ship
          - no_improvement — baseline ship when best_wns ≤ initial_wns
          - hard_fail — neither baseline nor mirror is usable
          - fallback_source given explicitly (used for stale_mirror_file or
            external_write_validated)

        Returns the recorded entry.
        """
        size = None
        try:
            p = Path(dcp_path)
            if p.exists():
                size = p.stat().st_size
        except Exception:
            pass
        if hard_fail:
            entry = _make_lineage_entry(
                "hard_fail_no_ship",
                token=None,
                dcp_path=str(dcp_path),
                dcp_size=size,
                extra={"branch": branch, "edif_ok": edif_ok},
            )
        elif no_improvement:
            entry = _make_lineage_entry(
                "no_improvement_baseline",
                token=None,
                wns=self.initial_wns,
                fmax=self.calculate_fmax(self.initial_wns, self.clock_period)
                if self.initial_wns is not None and self.clock_period else None,
                dcp_path=str(dcp_path),
                dcp_size=size,
                extra={"branch": branch, "edif_ok": edif_ok},
            )
        elif baseline:
            entry = _make_lineage_entry(
                "baseline",
                token=None,
                wns=self.initial_wns,
                fmax=self.calculate_fmax(self.initial_wns, self.clock_period)
                if self.initial_wns is not None and self.clock_period else None,
                dcp_path=str(dcp_path),
                dcp_size=size,
                extra={"branch": branch, "edif_ok": edif_ok},
            )
        elif inherit_best_valid and self._best_valid_lineage is not None:
            # Copy the current best-valid lineage, override path/size to
            # reflect the SHIPPED file (not the mirror file), preserve
            # the original token so downstream audits can trace.
            bv = dict(self._best_valid_lineage)
            entry = _make_lineage_entry(
                bv.get("source", "eager_mirror"),
                token=bv.get("token"),
                wns=bv.get("wns"),
                fmax=bv.get("fmax"),
                iteration=bv.get("iteration"),
                tool_name=bv.get("tool_name"),
                dcp_path=str(dcp_path),
                dcp_size=size,
                extra={
                    "branch": branch,
                    "edif_ok": edif_ok,
                    "best_valid_token": bv.get("token"),
                    "best_valid_source": bv.get("source"),
                },
            )
        else:
            # Either inherit_best_valid was requested but no lineage exists,
            # OR caller passed fallback_source explicitly.
            src = fallback_source or "external_write_validated"
            entry = _make_lineage_entry(
                src,
                token=None,
                wns=self.best_wns if self.best_wns != float("-inf") else None,
                fmax=self.calculate_fmax(self.best_wns, self.clock_period)
                if self.best_wns is not None and self.best_wns != float("-inf")
                and self.clock_period else None,
                dcp_path=str(dcp_path),
                dcp_size=size,
                extra={
                    "branch": branch,
                    "edif_ok": edif_ok,
                    "reason": "no best_valid lineage available at finalize",
                },
            )
        self._ship_lineage = entry
        try:
            self.lifecycle_log.append({
                "event": "ship_lineage_recorded",
                **{k: v for k, v in entry.items() if k != "extra"},
            })
        except Exception:
            pass
        return entry

    def _bump_lineage(
        self,
        source: str,
        *,
        wns: Optional[float] = None,
        fmax: Optional[float] = None,
        tool_name: Optional[str] = None,
        dcp_path: Optional[Path] = None,
        extra: Optional[dict] = None,
    ) -> dict:
        """Advances the monotonic best-valid lineage token and returns its new
        entry.

        The new entry replaces the current lineage state and is recorded as a
        best-valid lineage update in the lifecycle log.
        """
        self._best_valid_token += 1
        size = None
        path_str = None
        if dcp_path is not None:
            try:
                p = Path(dcp_path)
                path_str = str(p)
                if p.exists():
                    size = p.stat().st_size
            except Exception:
                pass
        if fmax is None and wns is not None and self.clock_period:
            fmax = self.calculate_fmax(wns, self.clock_period)
        entry = _make_lineage_entry(
            source,
            token=self._best_valid_token,
            wns=wns,
            fmax=fmax,
            iteration=getattr(self, "iteration", None),
            tool_name=tool_name,
            dcp_path=path_str,
            dcp_size=size,
            extra=extra,
        )
        self._best_valid_lineage = entry
        try:
            self.lifecycle_log.append({
                "event": "best_valid_lineage_update",
                **{k: v for k, v in entry.items() if k != "extra"},
            })
        except Exception:
            pass
        # Emit a decision-trace event whenever the
        # best-valid lineage bumps.  Covers eager_mirror / piggyback /
        # backstop / explicit caller-supplied sources in one place
        # (instead of three separate emits at the call sites).
        try:
            self._emit_decision({
                "decision_source": "executor",
                "phase": "best_valid_update",
                "tool_name": tool_name,
                "action_label": f"best_valid_{source}",
                "wns_after": wns,
                "fmax_after": fmax,
                "best_valid_checkpoint_path": path_str,
                "ship_lineage_source": source,
                "notes": f"token={self._best_valid_token} extra={extra}",
            })
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug(f"best_valid trace emit failed: {exc}")
        return entry

    def _piggyback_mirror_checkpoint(self, arguments: dict) -> None:
        """Mirrors a newly written checkpoint into the best-valid checkpoint
        without another tool invocation.

        This runs after a successful checkpoint write when a WNS improvement
        has left a best mirror pending. Success records the destination and
        clears the pending flag; failure preserves both so the full mirror
        fallback can retry. When source and destination are the same path, the
        existing file is recorded without copying and the pending flag is
        cleared.
        """
        try:
            # If any design-mutating operation ran between the best-WNS
            # measurement and this write, the checkpoint does not represent
            # best_wns, and stamping it would let a regressed state ship via
            # the finalize fast path.  Skip and disarm: every later write in
            # this window is post-mutation too, and a genuine new best re-arms
            # the flag at its own measurement.
            if self._mutation_epoch != self._pending_best_mirror_epoch:
                logger.warning(
                    "[piggyback] design mutated since best-WNS measurement "
                    f"(epoch {self._pending_best_mirror_epoch} -> "
                    f"{self._mutation_epoch}) — skipping best_valid stamp")
                self._pending_best_mirror = False
                return
            target = arguments.get("dcp_path")
            if not target:
                return
            src = Path(target).resolve()
            if not src.exists() or src.stat().st_size == 0:
                return
            dst = (self.run_dir / "best_valid.dcp").resolve()
            # Audit dst path before writing.
            self._path_guard_check(dst, context="piggyback_mirror_checkpoint")
            if src == dst:
                self._best_valid_dcp = dst
                # Piggyback fires only when _pending_best_mirror was True,
                # which means self.best_wns was just bumped.  The on-disk
                # file represents that exact best_wns.
                self._best_valid_dcp_wns = self.best_wns
                # Bank-time size = mirror integrity reference.
                try:
                    self._best_valid_mirror_size = dst.stat().st_size
                except OSError:
                    pass
                self._pending_best_mirror = False
                self._bump_lineage(
                    "piggyback",
                    wns=self.best_wns,
                    tool_name="vivado_write_checkpoint",
                    dcp_path=dst,
                    extra={"src_eq_dst": True},
                )
                return
            _atomic_copy(str(src), str(dst))
            self._best_valid_dcp = dst
            # See above — piggyback success implies the LLM-written
            # checkpoint that was copied represents the current best_wns.
            self._best_valid_dcp_wns = self.best_wns
            # Bank-time size = mirror integrity reference.
            try:
                self._best_valid_mirror_size = dst.stat().st_size
            except OSError:
                pass
            self._pending_best_mirror = False
            self._bump_lineage(
                "piggyback",
                wns=self.best_wns,
                tool_name="vivado_write_checkpoint",
                dcp_path=dst,
                extra={"src": str(src)},
            )
            logger.info(
                f"Piggyback best_valid mirror (wns={self.best_wns:.3f} ns, "
                f"token={self._best_valid_token}): "
                f"{src.name} → {dst.name} ({dst.stat().st_size//1024} KiB)"
            )
        except Exception as e:
            logger.warning(f"Piggyback DCP mirror failed: {e}")

    def _piggyback_mirror_edif(self, arguments: dict) -> None:
        """Mirrors a newly written EDIF beside the current best-valid checkpoint.

        The mirror runs after a successful EDIF write when the best-valid
        checkpoint already exists, keeping validator inputs synchronized. When
        source and destination are the same path, the existing file is accepted
        without copying.
        """
        try:
            target = arguments.get("edif_path")
            if not target:
                return
            src = Path(target).resolve()
            if not src.exists() or src.stat().st_size == 0:
                return
            dst = (self.run_dir / "best_valid.edf").resolve()
            # Audit dst path before writing.
            self._path_guard_check(dst, context="piggyback_mirror_edif")
            if src == dst:
                self._best_valid_edif = dst
                self._best_valid_edif_wns = self.best_wns
                return
            _atomic_copy(str(src), str(dst))
            self._best_valid_edif = dst
            # EDIF freshness marker mirrors the DCP one — same rationale.
            self._best_valid_edif_wns = self.best_wns
            logger.info(f"Piggyback best_valid EDIF mirror: {src.name} → {dst.name}")
        except Exception as e:
            logger.warning(f"Piggyback EDIF mirror failed: {e}")

    async def _routed_ok_for_best(self) -> bool:
        """Accepts a best-WNS update only when the in-memory design is fully
        routed.

        Unrouted timing is an estimate and must not replace a valid routed
        result. Returns true only when no unrouted nets or routing errors are
        reported; a fully routed report may omit the unrouted-net field. Probe
        failures return false so an unverifiable checkpoint cannot become
        protected by never-worse selection.
        """
        # An unplaced design reports neither of the lines the regexes below
        # look for, so a naive probe returns True and the auto-bank hook will
        # happily bank and ship an unplaced checkpoint with a phantom near-met
        # slack — which scores nothing.  The harness's own routed-state tracker
        # is authoritative and free, and it runs unconditionally, so consult it
        # first.
        if not getattr(self, "_design_routed_state", True):
            logger.warning(
                "phantom-best guard: in-process routed-state tracker says "
                "NOT ROUTED — REJECTING best-accept (unplaced/unrouted "
                "phantom).")
            return False
        try:
            res = await self.call_tool(
                "vivado_run_tcl",
                {"command": "report_route_status -return_string",
                 "timeout": 600.0})
            txt = str(res or "")
            if _looks_like_tool_error(txt):
                logger.warning("phantom-best guard: route-status probe "
                               "unavailable — REJECTING best-accept "
                               f"(fail closed): {txt[:120]}")
                return False
            m_err = re.search(r"# of nets with routing errors.*?:\s+(-?\d+)", txt)
            if m_err and int(m_err.group(1)) != 0:
                return False
            m_unr = re.search(r"# of unrouted nets.*?:\s+(-?\d+)", txt)
            if m_unr and int(m_unr.group(1)) != 0:
                return False
            return True
        except Exception as e:
            logger.warning("phantom-best guard: route-status probe raised "
                           f"{type(e).__name__} — REJECTING best-accept "
                           "(fail closed)")
            return False


        # Full re-place from the pristine input for deeply-negative designs.
        # Reopening the original checkpoint and re-placing reaches a basin that
        # a retime-first sequence does not.  Runs last: like the gamble it
        # discards chain polish, and it seeds from pristine, so it depends on
        # nothing above it.  Never-worse — it only ever adds a MUX candidate.


    async def _maybe_run_fir_subband_floor(self) -> None:
        """Runs an optional in-session physical-optimization floor for a
        configured shallow timing band.

        It must run after the first recipe stage because it depends on that
        stage's in-memory placement, and before any recipe pass that resets the
        session. Tool calls use the normal banking path so accepted WNS
        improvements update best-valid state and recipe accounting
        consistently. A losing refinement leaves the banked best unchanged, and
        exceptions return control to the LLM loop.
        """
        if not fir_subband_floor_enabled():
            logger.info("SUBBAND-FLOOR: skipped reason=disabled "
                        "(FPL26_SUBBAND_PHYSOPT_FLOOR off or "
                        "FPL26_NO_SUBBAND_PHYSOPT_FLOOR set)")
            return
        wns_in = self.initial_wns  # same key the recipe-pass gate uses
        if not fir_subband_match(wns_in):
            logger.info(f"SUBBAND-FLOOR: skipped reason=out_of_band "
                        f"wns_in={wns_in}")
            return
        # Secondary key.  The floor is fitted to one congestion regime, so a
        # measurably congested in-band design gets normal shallow treatment
        # instead.  Direction matters: an UNMEASURED count does not skip.  The
        # validated path keys on slack alone, and making it depend on a second
        # measurement would let a parse failure downgrade a deterministic
        # result to the sampling lottery.  The key may exclude on positive
        # evidence only; it may never gate the validated path on measurement
        # success.
        _failing = self._phase1_failing_endpoints_for_features()
        if _failing is not None and not fir_like_failing_ok(_failing):
            logger.info(f"SUBBAND-FLOOR: skipped reason=not_subband_like "
                        f"(measured failing_endpoints={_failing} > "
                        f"{SUBBAND_FLOOR_MAX_FAILING_ENDPOINTS})")
            return
        cap_s = SUBBAND_FLOOR_EXPECTED_S * RECIPE_PASS_TIMEOUT_FACTOR
        need = cap_s + SUBBAND_FLOOR_RESERVE_S
        remaining = self._budget_remaining()
        if remaining < need:
            logger.info(f"SUBBAND-FLOOR: skipped reason=insufficient_wall "
                        f"(remaining {remaining:.0f}s < need {need:.0f}s = "
                        f"cap {cap_s:.0f}s + reserve "
                        f"{SUBBAND_FLOOR_RESERVE_S:.0f}s)")
            return
        logger.info(f"SUBBAND-FLOOR: ARMED wns_in={wns_in:.3f} "
                    f"(scripted floor, in-session bank, cap {cap_s:.0f}s)")
        deadline = time.time() + cap_s
        steps = (
            ("vivado_run_tcl", {"command": SUBBAND_FLOOR_GROUP_TCL}),
            # Arg-drop EFFECTIVE forms of the traced floor (:100/:121/:137):
            ("vivado_phys_opt_design", {"directive": "Default"}),
            ("vivado_get_wns", {}),
            ("vivado_phys_opt_design", {"directive": "Default"}),
            ("vivado_get_wns", {}),
            ("vivado_phys_opt_design", {"critical_pin_opt": True,
                                        "path_groups": "critical_endpoints"}),
            ("vivado_get_wns", {}),
        )
        try:
            for i, (tool, args) in enumerate(steps, 1):
                left = deadline - time.time()
                if left <= 0:
                    self._subband_floor_failed = True
                    logger.warning(f"SUBBAND-FLOOR: aborted at step {i}/7 "
                                   f"(wall cap reached) — partial floor "
                                   f"stays banked; the shallow pass will "
                                   f"run as fallback (floor marked FAILED)")
                    return
                call_args = dict(args)
                if tool != "vivado_get_wns":
                    call_args.setdefault("timeout", float(max(60.0, left)))
                _res = await asyncio.wait_for(
                    self.call_tool(tool, call_args),
                    timeout=float(left) + 60.0)
                # call_tool reports failures as strings — a JSON error envelope,
                # a budget skip, or raw Tcl error text — never as exceptions.
                # Without this check a failed step falls through and the
                # completion line lies.  Abort on the first bad payload;
                # whatever earlier steps banked stays banked.
                if isinstance(_res, str) and _looks_like_tool_error(_res):
                    self._subband_floor_failed = True
                    logger.warning(
                        f"SUBBAND-FLOOR: aborted at step {i}/7 — {tool} "
                        f"returned a tool error ({_res[:160]}); partial "
                        f"floor stays banked; the shallow pass will run "
                        f"as fallback (floor marked FAILED)")
                    return
            logger.info(f"SUBBAND-FLOOR: done best_wns={self.best_wns} "
                        f"(banked in-session; ILS seeding/hurdle see it as "
                        f"recipe gain)")
        except Exception as e:
            self._subband_floor_failed = True
            logger.warning(f"SUBBAND-FLOOR: aborted ({type(e).__name__}: {e!r}) "
                           f"— non-fatal; the shallow pass will run as "
                           f"fallback (floor marked FAILED)")


    async def _eto_retime_ff_count(self, timeout_s: float,
                                    tag: str = "ETO-RETIME"):
        """FF-count probe for the latency audit (ETO_RETIME_FF_COUNT_TCL).
        Returns int or None (unmeasurable — the caller fails CLOSED).
        Parses ONLY the FFCOUNT= sentinel token: a bare
        first-parseable-token scan could bind to an unrelated number in
        interleaved Vivado chatter and silently pass a wrong audit.
        SHARED helper: the WLD-retime candidate reuses this probe
        verbatim, passing tag="WLD-RETIME" so its log lines keep the
        stable per-candidate key; the default tag keeps the ETO armed
        path's log bytes unchanged."""
        try:
            res = await asyncio.wait_for(
                self.call_tool("vivado_run_tcl",
                               {"command": ETO_RETIME_FF_COUNT_TCL,
                                "timeout": float(max(60.0, timeout_s))}),
                timeout=float(max(60.0, timeout_s)) + 60.0)
        except Exception as e:
            logger.warning(f"{tag}: ff-count probe raised "
                           f"{type(e).__name__}: {e!r}")
            return None
        if not isinstance(res, str) or _looks_like_tool_error(res):
            logger.warning(f"{tag}: ff-count probe errored "
                           f"({str(res)[:120]})")
            return None
        return eto_retime_parse_ffcount(res)

    async def _maybe_run_eto_retime_candidate(
            self, input_dcp: Path, wns_in: float) -> None:
        """FPL26_ETO_RETIME_CANDIDATE (DEFAULT OFF): the q07 retime
        chain as a SECOND shallow-pass candidate.  See the ETO_RETIME_*
        constants block for evidence, sub-band rationale, the latency-audit
        contract, and the recomputed wall arithmetic (need2 = 2330 s).

        Candidate contract (identical to the primary): pristine reopen ->
        frozen steps under a hard deadline -> FF latency audit -> measure ->
        private store -> register_final_candidate verify=True.  Never banks
        into best_valid (caller's auto-bank suppression bracket is still
        active); competes ONLY at the finalize MUX = never-worse by
        construction.  Every skip is fail-closed and logged.

        Stable log keys: "ETO-RETIME: skipped reason=...",
        "ETO-RETIME: fired", "ETO-RETIME: ff-audit", "ETO-RETIME: result".
        """
        if not eto_retime_candidate_enabled():
            if v40_flag_env_present("FPL26_ETO_RETIME_CANDIDATE",
                                    "FPL26_NO_ETO_RETIME_CANDIDATE"):
                logger.info("ETO-RETIME: skipped reason=disabled "
                            "(FPL26_ETO_RETIME_CANDIDATE off or "
                            "FPL26_NO_ETO_RETIME_CANDIDATE set)")
            return
        # One-candidate rule: when the unified own-front candidate is armed it
        # owns the retime slot, and this candidate defers before spending any
        # wall, so exactly one retime candidate runs per pass.  Stacking them
        # is self-defeating — the first spends the wall the second then cannot
        # afford.  With the own-front flag off this branch is dead.
        if ownfront_retime_enabled() and ownfront_retime_front(wns_in) is not None:
            # Parity-measured (+17.51 against 29.19): the defer
            # applies ONLY when the ownfront candidate actually HANDLES this
            # design's band.  Out-of-band designs (below the 0.90
            # floor) fall through to THIS candidate = the exact
            # validated wall spend (that class's cycle-5 fork needs the tightness).
            logger.info("ETO-RETIME: skipped reason=ownfront_supersedes "
                        "(FPL26_OWNFRONT_RETIME_CANDIDATE armed — the "
                        "unified own-front candidate owns the retime "
                        "slot; exactly ONE retime candidate spends wall "
                        "per run)")
            return
        if shallow_determinizer_enabled() and shallow_det_subband_match(wns_in):
            # In this sub-band the determinizer replaces this candidate rather
            # than stacking after it: this candidate's result is discarded by
            # the MUX on that class every time, and the stacked shape does not
            # fit the wall anyway.  The swap makes the determinizer's
            # deterministic floor affordable exactly where it matters.
            logger.info("ETO-RETIME: skipped reason=shallow_det_supersedes "
                        "(FPL26_SHALLOW_DETERMINIZER_CANDIDATE armed and "
                        "|wns_in| in [0.60, 0.90) — the deterministic "
                        "floor candidate owns this sub-band's wall; "
                        "this candidate is measured MUX-discarded x3 here)")
            return
        if not eto_retime_subband_match(wns_in):
            logger.info(f"ETO-RETIME: skipped reason=out_of_band "
                        f"wns_in={wns_in} (sub-band "
                        f"[{ETO_RETIME_WNS_MAG_MIN_NS}, "
                        f"{ETO_RETIME_WNS_MAG_MAX_NS}])")
            return
        # Recomputed second-candidate gate (constants block): the primary
        # shallow gate (2450 s) is UNTOUCHED; this one prices the second
        # candidate against what is left NOW, after the primary spent its
        # wall.  Fail-closed skip keeps the run byte-identical to flag-off.
        try:
            llm_floor = float(getattr(
                self, "phys_opt_preempt_budget_floor_s", 1100.0))
        except (TypeError, ValueError):
            llm_floor = 1100.0
        cap_s = RECIPE_PASS_TIMEOUT_FACTOR * ETO_RETIME_EXPECTED_S  # 690
        need = cap_s + ETO_RETIME_OVERHEAD_RESERVE_S + llm_floor    # 2330
        remaining = self._budget_remaining()
        if remaining < need:
            logger.info(
                f"ETO-RETIME: skipped reason=wall "
                f"(remaining {remaining:.0f}s < cap {cap_s:.0f}s "
                f"(1.5x{ETO_RETIME_EXPECTED_S:.0f}) + overheads "
                f"{ETO_RETIME_OVERHEAD_RESERVE_S:.0f}s [measure "
                f"{RECIPE_PASS_MEASURE_RESERVE_S:.0f} + store "
                f"{RECIPE_PASS_STORE_RESERVE_S:.0f} + register "
                f"{RECIPE_PASS_REGISTER_RESERVE_S:.0f}; reset covered by "
                f"the caller's single post-pass reset] + LLM-loop floor "
                f"{llm_floor:.0f}s = {need:.0f}s — fail closed, primary "
                f"candidate unaffected)")
            return
        pass_budget = cap_s
        if remaining != float("inf"):
            pass_budget = min(pass_budget,
                              remaining - RECIPE_PASS_FINALIZE_RESERVE_S)
        if pass_budget < 60.0:
            logger.info(f"ETO-RETIME: skipped reason=wall "
                        f"(pass budget {pass_budget:.0f}s < 60s floor)")
            return
        logger.info(f"ETO-RETIME: fired wns_in={wns_in:.3f} "
                    f"(pass_budget={pass_budget:.0f}s, "
                    f"steps={len(ETO_RETIME_TCL)}, MUX-additive second "
                    f"shallow candidate)")
        t0 = time.time()
        pass_deadline = t0 + pass_budget
        # This chain is defined relative to the pristine input checkpoint.
        # The primary recipe mutates the active session, so reopen pristine state first.
        if not await self._presweep_reopen(input_dcp):
            logger.warning("ETO-RETIME: failed step=pristine_preopen "
                           "(chain not run; wall lost, nothing banked)")
            return
        ff_before = None
        ff_after = None
        for i, step in enumerate(ETO_RETIME_TCL, 1):
            step_left = pass_deadline - time.time()
            if step_left < 30.0:
                logger.warning(
                    f"ETO-RETIME: failed step={i}/{len(ETO_RETIME_TCL)} "
                    f"reason=pass_timeout ({pass_budget:.0f}s budget "
                    f"exhausted before '{step}'; aborting — wall lost, "
                    f"nothing banked)")
                return
            try:
                res = await asyncio.wait_for(
                    self.call_tool("vivado_run_tcl",
                                   {"command": step,
                                    "timeout": float(step_left)}),
                    timeout=float(step_left) + 60.0)
            except Exception as e:
                logger.warning(
                    f"ETO-RETIME: failed step={i}/{len(ETO_RETIME_TCL)} "
                    f"reason=step_raised:{type(e).__name__} "
                    f"cmd='{step}' (aborting)")
                return
            if _looks_like_tool_error(res):
                logger.warning(
                    f"ETO-RETIME: failed step={i}/{len(ETO_RETIME_TCL)} "
                    f"reason=step_error cmd='{step}' "
                    f"detail={str(res)[:120]} (aborting)")
                return
            # LATENCY AUDIT probes: FF count after the place step (=before
            # retime) and after the retime step.  Cheap queries; bounded by
            # the step budget.
            if step == "place_design -directive ExtraTimingOpt":
                ff_before = await self._eto_retime_ff_count(
                    min(300.0, max(60.0, pass_deadline - time.time())))
            elif step == "phys_opt_design -retime":
                ff_after = await self._eto_retime_ff_count(
                    min(300.0, max(60.0, pass_deadline - time.time())))
        # ---- latency audit (DQ protection; fails CLOSED) ----
        if not eto_retime_ff_drift_ok(ff_before, ff_after):
            logger.warning(
                f"ETO-RETIME: ABORTED reason=latency_audit "
                f"ff_before={ff_before} ff_after={ff_after} "
                f"(drift beyond +-{ETO_RETIME_FF_DRIFT_MAX_FRAC:.0%} or "
                f"unmeasured — protections fail ON; candidate NOT "
                f"registered, primary candidate unaffected)")
            return
        logger.info(f"ETO-RETIME: ff-audit PASS ff_before={ff_before} "
                    f"ff_after={ff_after} "
                    f"(drift {abs(ff_after - ff_before) / float(ff_before):.4%}"
                    f" <= {ETO_RETIME_FF_DRIFT_MAX_FRAC:.0%})")
        # ---- measure + private store + register (contract 1-3) ----
        try:
            wns_out = await asyncio.wait_for(
                self.get_wns_for_target_clock(self._call_vivado_tool),
                timeout=600.0)
        except Exception:
            wns_out = None
        if wns_out is None:
            logger.warning("ETO-RETIME: failed step=measure "
                           "reason=wns_unmeasurable (nothing banked)")
            return
        store_dir = self.run_dir / "final_candidates"
        store_dir.mkdir(parents=True, exist_ok=True)
        store = store_dir / "recipe_pass_shallow_etoretime.dcp"
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
            logger.warning("ETO-RETIME: failed step=store "
                           f"reason=write_checkpoint detail="
                           f"{str(res)[:120]} (nothing banked)")
            return
        try:
            registered = await self.register_final_candidate(
                store, float(wns_out), "recipe_pass_shallow_etoretime",
                verify=True)
        except Exception as e:
            logger.warning(
                f"ETO-RETIME: register FAILED ({type(e).__name__}: "
                f"{e!r}); candidate dropped; caller's reset proceeds")
            return
        logger.info(
            f"ETO-RETIME: result wns_in={wns_in:.3f} "
            f"wns_out={float(wns_out):.3f} registered={registered} "
            f"ff_before={ff_before} ff_after={ff_after} "
            f"dt={time.time() - t0:.0f}s (candidate competes ONLY at the "
            f"finalize MUX; caller resets to PRISTINE)")

    async def _maybe_run_ownfront_retime_candidate(
            self, input_dcp: Path, wns_in: float) -> None:
        """Runs one optional retiming candidate using a placement front selected
        from the input WNS magnitude.

        Magnitudes in [0.60, 1.00) ns and [1.00, 1.05) ns select their
        respective configured fronts; at most one retiming candidate may
        consume wall time. The candidate reopens a pristine design, executes
        fixed steps under a hard deadline, and audits FF-count drift within
        ±1%. An unavailable audit aborts the candidate, so latency protection
        fails closed. Results are stored privately and enter only the verified
        final never-worse selection; they never update best-valid state
        directly. Every skipped or aborted path is recorded.
        """
        if not ownfront_retime_enabled():
            if v40_flag_env_present("FPL26_OWNFRONT_RETIME_CANDIDATE",
                                    "FPL26_NO_OWNFRONT_RETIME_CANDIDATE"):
                logger.info("OWNFRONT-RETIME: skipped reason=disabled "
                            "(FPL26_OWNFRONT_RETIME_CANDIDATE off or "
                            "FPL26_NO_OWNFRONT_RETIME_CANDIDATE set)")
            return
        front = ownfront_retime_front(wns_in)
        if front is None:
            logger.info(f"OWNFRONT-RETIME: skipped reason=out_of_band "
                        f"wns_in={wns_in} (band "
                        f"[{OWNFRONT_RETIME_WNS_MAG_MIN_NS}, "
                        f"{OWNFRONT_RETIME_WNS_MAG_MAX_NS}]; split "
                        f"{OWNFRONT_RETIME_SPLIT_NS} — eto below, wld at "
                        f"or above; logicnets-class <0.60 uncovered this "
                        f"round by design)")
            return
        chain = ETO_RETIME_TCL if front == "eto" else WLD_RETIME_TCL
        place_step = chain[2]  # the selected front's place directive
        # A single recomputed gate for the one-candidate shape.  The primary
        # shallow gate is untouched; this prices the one retime candidate
        # against what is left now, after the primary has spent its wall.
        # There is no earlier retime candidate to stack against, since the
        # other one defers.  A fail-closed skip leaves the run unchanged from
        # this point.
        try:
            llm_floor = float(getattr(
                self, "phys_opt_preempt_budget_floor_s", 1100.0))
        except (TypeError, ValueError):
            llm_floor = 1100.0
        cap_s = RECIPE_PASS_TIMEOUT_FACTOR * OWNFRONT_RETIME_EXPECTED_S  # 780
        need = cap_s + OWNFRONT_RETIME_OVERHEAD_RESERVE_S + llm_floor   # 2420
        remaining = self._budget_remaining()
        if remaining < need:
            logger.info(
                f"OWNFRONT-RETIME: skipped reason=wall "
                f"(remaining {remaining:.0f}s < cap {cap_s:.0f}s "
                f"(1.5x{OWNFRONT_RETIME_EXPECTED_S:.0f}) + overheads "
                f"{OWNFRONT_RETIME_OVERHEAD_RESERVE_S:.0f}s [measure "
                f"{RECIPE_PASS_MEASURE_RESERVE_S:.0f} + store "
                f"{RECIPE_PASS_STORE_RESERVE_S:.0f} + register "
                f"{RECIPE_PASS_REGISTER_RESERVE_S:.0f}; reset covered by "
                f"the caller's single post-pass reset] + LLM-loop floor "
                f"{llm_floor:.0f}s = {need:.0f}s — fail closed, primary "
                f"candidate unaffected)")
            return
        pass_budget = cap_s
        if remaining != float("inf"):
            pass_budget = min(pass_budget,
                              remaining - RECIPE_PASS_FINALIZE_RESERVE_S)
        if pass_budget < 60.0:
            logger.info(f"OWNFRONT-RETIME: skipped reason=wall "
                        f"(pass budget {pass_budget:.0f}s < 60s floor)")
            return
        logger.info(f"OWNFRONT-RETIME: fired wns_in={wns_in:.3f} "
                    f"front={front} (pass_budget={pass_budget:.0f}s, "
                    f"steps={len(chain)}, MUX-additive unified second "
                    f"shallow candidate — the ONE retime candidate on "
                    f"this run)")
        t0 = time.time()
        pass_deadline = t0 + pass_budget
        # Both subchains are defined relative to the pristine input checkpoint.
        # The primary recipe mutates the active session, so reopen pristine state first.
        if not await self._presweep_reopen(input_dcp):
            logger.warning("OWNFRONT-RETIME: failed step=pristine_preopen "
                           "(chain not run; wall lost, nothing banked)")
            return
        ff_before = None
        ff_after = None
        for i, step in enumerate(chain, 1):
            step_left = pass_deadline - time.time()
            if step_left < 30.0:
                logger.warning(
                    f"OWNFRONT-RETIME: failed step={i}/{len(chain)} "
                    f"reason=pass_timeout ({pass_budget:.0f}s budget "
                    f"exhausted before '{step}'; aborting — wall lost, "
                    f"nothing banked)")
                return
            try:
                res = await asyncio.wait_for(
                    self.call_tool("vivado_run_tcl",
                                   {"command": step,
                                    "timeout": float(step_left)}),
                    timeout=float(step_left) + 60.0)
            except Exception as e:
                logger.warning(
                    f"OWNFRONT-RETIME: failed step={i}/{len(chain)} "
                    f"reason=step_raised:{type(e).__name__} "
                    f"cmd='{step}' (aborting)")
                return
            if _looks_like_tool_error(res):
                logger.warning(
                    f"OWNFRONT-RETIME: failed step={i}/{len(chain)} "
                    f"reason=step_error cmd='{step}' "
                    f"detail={str(res)[:120]} (aborting)")
                return
            # LATENCY AUDIT probes (shared FFCOUNT= sentinel helper): FF
            # count after the SELECTED place step (=before retime) and
            # after the retime step.  Cheap queries; bounded by the step
            # budget.
            if step == place_step:
                ff_before = await self._eto_retime_ff_count(
                    min(300.0, max(60.0, pass_deadline - time.time())),
                    tag="OWNFRONT-RETIME")
            elif step == "phys_opt_design -retime":
                ff_after = await self._eto_retime_ff_count(
                    min(300.0, max(60.0, pass_deadline - time.time())),
                    tag="OWNFRONT-RETIME")
        # ---- latency audit (DQ protection; fails CLOSED) ----
        if not eto_retime_ff_drift_ok(ff_before, ff_after):
            logger.warning(
                f"OWNFRONT-RETIME: ABORTED reason=latency_audit "
                f"front={front} ff_before={ff_before} ff_after={ff_after} "
                f"(drift beyond +-{ETO_RETIME_FF_DRIFT_MAX_FRAC:.0%} or "
                f"unmeasured — protections fail ON; candidate NOT "
                f"registered, primary candidate unaffected)")
            return
        logger.info(f"OWNFRONT-RETIME: ff-audit PASS ff_before={ff_before} "
                    f"ff_after={ff_after} "
                    f"(drift {abs(ff_after - ff_before) / float(ff_before):.4%}"
                    f" <= {ETO_RETIME_FF_DRIFT_MAX_FRAC:.0%})")
        # ---- measure + private store + register (contract 1-3) ----
        try:
            wns_out = await asyncio.wait_for(
                self.get_wns_for_target_clock(self._call_vivado_tool),
                timeout=600.0)
        except Exception:
            wns_out = None
        if wns_out is None:
            logger.warning("OWNFRONT-RETIME: failed step=measure "
                           "reason=wns_unmeasurable (nothing banked)")
            return
        store_dir = self.run_dir / "final_candidates"
        store_dir.mkdir(parents=True, exist_ok=True)
        store = store_dir / f"recipe_pass_shallow_ownfront_{front}.dcp"
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
            logger.warning("OWNFRONT-RETIME: failed step=store "
                           f"reason=write_checkpoint detail="
                           f"{str(res)[:120]} (nothing banked)")
            return
        try:
            registered = await self.register_final_candidate(
                store, float(wns_out),
                f"recipe_pass_shallow_ownfront_{front}", verify=True)
        except Exception as e:
            logger.warning(
                f"OWNFRONT-RETIME: register FAILED ({type(e).__name__}: "
                f"{e!r}); candidate dropped; caller's reset proceeds")
            return
        logger.info(
            f"OWNFRONT-RETIME: result wns_in={wns_in:.3f} front={front} "
            f"wns_out={float(wns_out):.3f} registered={registered} "
            f"ff_before={ff_before} ff_after={ff_after} "
            f"dt={time.time() - t0:.0f}s (candidate competes ONLY at the "
            f"finalize MUX; caller resets to PRISTINE)")

    async def _maybe_run_shallow_determinizer_candidate(
            self, input_dcp: Path, wns_in: float) -> None:
        """Run the optional deterministic candidate for the shallow eligibility
        band.

        When enabled, it applies only in the [0.60, 0.90) band and runs after
        the earlier shallow candidates. Execution reopens the pristine
        checkpoint, performs fixed steps under a hard deadline, audits FF
        latency, measures the result, and registers a verified final candidate.
        Deadline, audit, and execution failures skip the candidate without
        changing the current best. Auto-banking remains suppressed, the
        best-valid state is never overwritten, and final selection can only
        adopt a verified improvement.
        """
        if not shallow_determinizer_enabled():
            if v40_flag_env_present("FPL26_SHALLOW_DETERMINIZER_CANDIDATE",
                                    "FPL26_NO_SHALLOW_DETERMINIZER_CANDIDATE"):
                logger.info("SHALLOW-DET: skipped reason=disabled "
                            "(FPL26_SHALLOW_DETERMINIZER_CANDIDATE off or "
                            "FPL26_NO_SHALLOW_DETERMINIZER_CANDIDATE set)")
            return
        if not shallow_det_subband_match(wns_in):
            logger.info(f"SHALLOW-DET: skipped reason=out_of_band "
                        f"wns_in={wns_in} (sub-band "
                        f"[{SHALLOW_DET_WNS_MAG_MIN_NS}, "
                        f"{SHALLOW_DET_WNS_MAG_MAX_NS}) — high bound "
                        f"exclusive; [0.90, 1.05] is the ownfront "
                        f"candidate's)")
            return
        # Recomputed third-candidate gate.  The primary and second-candidate
        # gates are untouched; this prices the determinizer against what is
        # left after the primary spent its wall.  On a typical ship wall too
        # little remains and this skip fires — which is the intended answer to
        # "which candidate defers": the determinizer does, and the swap above
        # frees the wall the other candidate would have wasted in this band.
        try:
            llm_floor = float(getattr(
                self, "phys_opt_preempt_budget_floor_s", 1100.0))
        except (TypeError, ValueError):
            llm_floor = 1100.0
        cap_s = RECIPE_PASS_TIMEOUT_FACTOR * SHALLOW_DET_EXPECTED_S  # 585
        need = cap_s + SHALLOW_DET_OVERHEAD_RESERVE_S + llm_floor    # 2225
        remaining = self._budget_remaining()
        if remaining < need:
            logger.info(
                f"SHALLOW-DET: skipped reason=wall "
                f"(remaining {remaining:.0f}s < cap {cap_s:.0f}s "
                f"(1.5x{SHALLOW_DET_EXPECTED_S:.0f}) + overheads "
                f"{SHALLOW_DET_OVERHEAD_RESERVE_S:.0f}s [measure "
                f"{RECIPE_PASS_MEASURE_RESERVE_S:.0f} + store "
                f"{RECIPE_PASS_STORE_RESERVE_S:.0f} + register "
                f"{RECIPE_PASS_REGISTER_RESERVE_S:.0f}; reset covered by "
                f"the caller's single post-pass reset] + LLM-loop floor "
                f"{llm_floor:.0f}s = {need:.0f}s — fail closed, earlier "
                f"candidates unaffected; NOTE post-swap this leaves the "
                f"band with no retime candidate and a ~700s-larger tail "
                f"budget — a known-uncovered cell, disclosed in "
                f"docs/CONFIGURATION.md)")
            return
        pass_budget = cap_s
        if remaining != float("inf"):
            pass_budget = min(pass_budget,
                              remaining - RECIPE_PASS_FINALIZE_RESERVE_S)
        if pass_budget < 60.0:
            logger.info(f"SHALLOW-DET: skipped reason=wall "
                        f"(pass budget {pass_budget:.0f}s < 60s floor)")
            return
        logger.info(f"SHALLOW-DET: fired wns_in={wns_in:.3f} "
                    f"(pass_budget={pass_budget:.0f}s, "
                    f"steps={len(SHALLOW_DET_TCL)}, MUX-additive third "
                    f"shallow candidate — determinizer floor for the "
                    f"spam-class timing lottery)")
        t0 = time.time()
        pass_deadline = t0 + pass_budget
        # The chain is verified RELATIVE TO THE PRISTINE INPUT (the
        # probes ran from the pristine DCP), and the earlier candidates
        # just mutated the session — explicit pristine reopen first.
        if not await self._presweep_reopen(input_dcp):
            logger.warning("SHALLOW-DET: failed step=pristine_preopen "
                           "(chain not run; wall lost, nothing banked)")
            return
        ff_before = None
        ff_after = None
        for i, step in enumerate(SHALLOW_DET_TCL, 1):
            step_left = pass_deadline - time.time()
            if step_left < 30.0:
                logger.warning(
                    f"SHALLOW-DET: failed step={i}/{len(SHALLOW_DET_TCL)} "
                    f"reason=pass_timeout ({pass_budget:.0f}s budget "
                    f"exhausted before '{step}'; aborting — wall lost, "
                    f"nothing banked)")
                return
            try:
                res = await asyncio.wait_for(
                    self.call_tool("vivado_run_tcl",
                                   {"command": step,
                                    "timeout": float(step_left)}),
                    timeout=float(step_left) + 60.0)
            except Exception as e:
                logger.warning(
                    f"SHALLOW-DET: failed step={i}/{len(SHALLOW_DET_TCL)} "
                    f"reason=step_raised:{type(e).__name__} "
                    f"cmd='{step}' (aborting)")
                return
            if _looks_like_tool_error(res):
                logger.warning(
                    f"SHALLOW-DET: failed step={i}/{len(SHALLOW_DET_TCL)} "
                    f"reason=step_error cmd='{step}' "
                    f"detail={str(res)[:120]} (aborting)")
                return
            # Latency-audit probes bracketing the retiming-enabled phys_opt
            # step: flip-flop count after the route step (i.e. before it) and
            # again after.  Cheap queries, bounded by the step budget.
            if step == "route_design -directive AggressiveExplore":
                ff_before = await self._eto_retime_ff_count(
                    min(300.0, max(60.0, pass_deadline - time.time())),
                    tag="SHALLOW-DET")
            elif step == "phys_opt_design -directive AlternateFlowWithRetiming":
                ff_after = await self._eto_retime_ff_count(
                    min(300.0, max(60.0, pass_deadline - time.time())),
                    tag="SHALLOW-DET")
        # ---- latency audit (DQ protection; fails CLOSED) ----
        if not eto_retime_ff_drift_ok(ff_before, ff_after):
            logger.warning(
                f"SHALLOW-DET: ABORTED reason=latency_audit "
                f"ff_before={ff_before} ff_after={ff_after} "
                f"(drift beyond +-{ETO_RETIME_FF_DRIFT_MAX_FRAC:.0%} or "
                f"unmeasured — protections fail ON; candidate NOT "
                f"registered, earlier candidates unaffected)")
            return
        logger.info(f"SHALLOW-DET: ff-audit PASS ff_before={ff_before} "
                    f"ff_after={ff_after} "
                    f"(drift {abs(ff_after - ff_before) / float(ff_before):.4%}"
                    f" <= {ETO_RETIME_FF_DRIFT_MAX_FRAC:.0%})")
        # ---- measure + private store + register (contract 1-3) ----
        try:
            wns_out = await asyncio.wait_for(
                self.get_wns_for_target_clock(self._call_vivado_tool),
                timeout=600.0)
        except Exception:
            wns_out = None
        if wns_out is None:
            logger.warning("SHALLOW-DET: failed step=measure "
                           "reason=wns_unmeasurable (nothing banked)")
            return
        store_dir = self.run_dir / "final_candidates"
        store_dir.mkdir(parents=True, exist_ok=True)
        store = store_dir / "recipe_pass_shallow_spamdet.dcp"
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
            logger.warning("SHALLOW-DET: failed step=store "
                           f"reason=write_checkpoint detail="
                           f"{str(res)[:120]} (nothing banked)")
            return
        try:
            registered = await self.register_final_candidate(
                store, float(wns_out), "recipe_pass_shallow_spamdet",
                verify=True)
        except Exception as e:
            logger.warning(
                f"SHALLOW-DET: register FAILED ({type(e).__name__}: "
                f"{e!r}); candidate dropped; caller's reset proceeds")
            return
        logger.info(
            f"SHALLOW-DET: result wns_in={wns_in:.3f} "
            f"wns_out={float(wns_out):.3f} registered={registered} "
            f"ff_before={ff_before} ff_after={ff_after} "
            f"dt={time.time() - t0:.0f}s (candidate competes ONLY at the "
            f"finalize MUX; caller resets to PRISTINE)")


    async def _maybe_run_midband_route_rung_postloop(self) -> None:
        """Run the optional post-loop route rung for eligible mid-band designs.

        This stage must run immediately before wall handback and claims its
        budget before the handback decision. It starts from the banked-best
        state, then unroutes, retimes, applies physical optimization, and
        reroutes; it must not run from the pristine state or before the main
        loop. Auto-banking remains suppressed, the banked-best mirror is
        read-only, and the session becomes untrusted on the first mutation.
        Because the chain retimes registers, the shared FF latency audit
        applies and fails closed. The result is registered as a verified final
        candidate, so final selection cannot replace the banked best with a
        worse result.
        """
        if not midband_route_rung_enabled():
            if v40_flag_env_present("FPL26_MIDBAND_ROUTE_RUNG",
                                    "FPL26_NO_MIDBAND_ROUTE_RUNG"):
                logger.info("ROUTE-RUNG[postloop]: skipped reason=disabled "
                            "(FPL26_MIDBAND_ROUTE_RUNG off or "
                            "FPL26_NO_MIDBAND_ROUTE_RUNG set)")
            return
        wns_in = self.initial_wns
        band = recipe_pass_band(wns_in)
        if band != "mid":
            logger.info(f"ROUTE-RUNG[postloop]: skipped reason=band "
                        f"band={band} wns_in={wns_in} (mid-band-only "
                        f"scope, 1.05 < |wns_in| < 8.0)")
            return
        best = self._best_valid_dcp
        if best is None or not Path(best).exists():
            # The break's mechanism needs the polished attractor state; a
            # run that banked nothing has no evidence-bearing input.
            logger.info("ROUTE-RUNG[postloop]: skipped reason="
                        "no_banked_best (break is proven FROM the polished "
                        "state; pristine input is out of evidence)")
            return
        # NO handback defer here (the earlier rung variant's inert
        # cause).  The rung claims its priced wall ahead of the handback
        # exit decision; handback returns whatever remains after it.
        # Scope of the reallocation = mid-band + banked-best only
        # (the two gates above).
        if self._wall_handback_break_due():
            logger.info(
                f"ROUTE-RUNG[postloop]: claiming priced wall AHEAD of "
                f"wall-handback (remaining={self._budget_remaining():.0f}s; "
                f"handback returns the remainder after the rung — "
                f"corescore-mid-only scope bounds the reallocation)")
        remaining = self._budget_remaining()
        overhead = RECIPE_PASS_POSTLOOP_OVERHEAD_RESERVE_S  # 540
        need = MIDBAND_BREAK_RUNG_EXPECTED_S + overhead   # 1350
        if remaining < need:
            logger.info(
                f"ROUTE-RUNG[postloop]: skipped reason=wall "
                f"(remaining {remaining:.0f}s < measured break cost "
                f"{MIDBAND_BREAK_RUNG_EXPECTED_S:.0f}s (un-margined, "
                f"measured-cost convention; cs3 in-session 806s) + "
                f"overheads {overhead:.0f}s "
                f"[measure {RECIPE_PASS_MEASURE_RESERVE_S:.0f} + store "
                f"{RECIPE_PASS_STORE_RESERVE_S:.0f} + register "
                f"{RECIPE_PASS_REGISTER_RESERVE_S:.0f}; no reset/tail/LLM "
                f"terms post-loop] = {need:.0f}s — fail closed; whatever "
                f"wall exists stays with handback)")
            return
        pass_budget = (RECIPE_PASS_TIMEOUT_FACTOR
                       * MIDBAND_BREAK_RUNG_EXPECTED_S)  # 1215
        if remaining != float("inf"):
            pass_budget = min(pass_budget, remaining - overhead)
        if pass_budget < 60.0:
            logger.info(f"ROUTE-RUNG[postloop]: skipped reason=wall "
                        f"(pass budget {pass_budget:.0f}s < 60s floor)")
            return
        logger.info(f"ROUTE-RUNG[postloop]: fired band=mid "
                    f"wns_in={wns_in:.3f} best_wns={self.best_wns} "
                    f"(pass_budget={pass_budget:.0f}s, break chain "
                    f"steps={len(MIDBAND_BREAK_RUNG_TCL)} from the "
                    f"banked best state)")
        t0 = time.time()
        pass_deadline = t0 + pass_budget
        _tc0 = len(self.tool_call_details)
        prev_suppress = self._tail_ctrl_suppress_autobank
        self._tail_ctrl_suppress_autobank = True
        prev_ils_flag = self._in_ils_stage
        self._in_ils_stage = True
        try:
            # Session diverges from the pipeline state from here on —
            # same session-untrusted contract as the postloop RECIPE_PASS slot.
            self._finalize_session_untrusted = True
            if not await self._presweep_reopen(Path(best)):
                logger.warning("ROUTE-RUNG[postloop]: failed "
                               "step=best_reopen (break not run; wall "
                               "lost, nothing banked)")
                return
            ff_before = None
            ff_after = None
            for i, step in enumerate(MIDBAND_BREAK_RUNG_TCL, 1):
                step_left = pass_deadline - time.time()
                if step_left < 30.0:
                    logger.warning(
                        f"ROUTE-RUNG[postloop]: failed step="
                        f"{i}/{len(MIDBAND_BREAK_RUNG_TCL)} "
                        f"reason=pass_timeout ({pass_budget:.0f}s budget "
                        f"exhausted before '{step}'; aborting — wall "
                        f"lost, nothing banked)")
                    return
                try:
                    res = await asyncio.wait_for(
                        self.call_tool("vivado_run_tcl",
                                       {"command": step,
                                        "timeout": float(step_left)}),
                        timeout=float(step_left) + 60.0)
                except Exception as e:
                    logger.warning(
                        f"ROUTE-RUNG[postloop]: failed step="
                        f"{i}/{len(MIDBAND_BREAK_RUNG_TCL)} "
                        f"reason=step_raised:{type(e).__name__} "
                        f"cmd='{step}' (aborting)")
                    return
                if _looks_like_tool_error(res):
                    logger.warning(
                        f"ROUTE-RUNG[postloop]: failed step="
                        f"{i}/{len(MIDBAND_BREAK_RUNG_TCL)} "
                        f"reason=step_error cmd='{step}' "
                        f"detail={str(res)[:120]} (aborting)")
                    return
                # LATENCY AUDIT probes (shared FFCOUNT= sentinel
                # helper): FF count after the unroute step (= before
                # retime) and after the retime step.
                if step == "route_design -unroute":
                    ff_before = await self._eto_retime_ff_count(
                        min(300.0, max(60.0, pass_deadline - time.time())),
                        tag="ROUTE-RUNG[postloop]")
                elif step == "phys_opt_design -retime":
                    ff_after = await self._eto_retime_ff_count(
                        min(300.0, max(60.0, pass_deadline - time.time())),
                        tag="ROUTE-RUNG[postloop]")
            # ---- latency audit (DQ protection; fails CLOSED) ----
            if not eto_retime_ff_drift_ok(ff_before, ff_after):
                logger.warning(
                    f"ROUTE-RUNG[postloop]: ABORTED reason=latency_audit "
                    f"ff_before={ff_before} ff_after={ff_after} "
                    f"(drift beyond +-{ETO_RETIME_FF_DRIFT_MAX_FRAC:.0%} "
                    f"or unmeasured — protections fail ON; candidate NOT "
                    f"registered)")
                return
            logger.info(
                f"ROUTE-RUNG[postloop]: ff-audit PASS "
                f"ff_before={ff_before} ff_after={ff_after} "
                f"(drift "
                f"{abs(ff_after - ff_before) / float(ff_before):.4%}"
                f" <= {ETO_RETIME_FF_DRIFT_MAX_FRAC:.0%})")
            # ---- measure + private store + register ----
            try:
                wns_out = await asyncio.wait_for(
                    self.get_wns_for_target_clock(self._call_vivado_tool),
                    timeout=600.0)
            except Exception:
                wns_out = None
            if wns_out is None:
                logger.warning("ROUTE-RUNG[postloop]: failed step=measure "
                               "reason=wns_unmeasurable (nothing banked)")
                return
            store_dir = self.run_dir / "final_candidates"
            store_dir.mkdir(parents=True, exist_ok=True)
            store = store_dir / "route_rung_mid_break.dcp"
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
                logger.warning("ROUTE-RUNG[postloop]: failed step=store "
                               f"reason=write_checkpoint detail="
                               f"{str(res)[:120]} (nothing banked)")
                return
            try:
                registered = await self.register_final_candidate(
                    store, float(wns_out), "route_rung_mid_break",
                    verify=True)
            except Exception as e:
                logger.warning(f"ROUTE-RUNG[postloop]: register FAILED "
                               f"({type(e).__name__}: {e!r}); candidate "
                               f"dropped (finalize proceeds on disk truth)")
                return
            logger.info(
                f"ROUTE-RUNG[postloop]: result wns_in={wns_in:.3f} "
                f"wns_out={float(wns_out):.3f} registered={registered} "
                f"ff_before={ff_before} ff_after={ff_after} "
                f"dt={time.time() - t0:.0f}s (candidate competes ONLY at "
                f"the finalize MUX; session left on the break state — "
                f"finalize is disk-truth-driven, postloop-slot contract)")
        finally:
            self._in_ils_stage = prev_ils_flag
            self._tail_ctrl_suppress_autobank = prev_suppress
            for _d in self.tool_call_details[_tc0:]:
                try:
                    _d["recipe_pass_internal"] = True
                except Exception:
                    pass
            if self._pending_best_mirror:
                logger.warning("ROUTE-RUNG[postloop]: unexpected pending "
                               "best-mirror after the rung (invariant "
                               "breach — leaving lineage untouched).")

    def _wns_tcl_for_stages(self) -> str:
        """Build a target-clock-aware Tcl query for terminal-stage WNS.

        The shared query is usable from the common exit path even when
        iterative polishing is disabled, including for very large designs.
        """
        if self.target_clock:
            return (
                f"set clk_obj [get_clocks -quiet {{{self.target_clock}}}]; "
                f"if {{$clk_obj ne {{}}}} {{ set tp [get_timing_paths -max_paths 1 "
                f"-setup -to $clk_obj]; if {{[llength $tp] > 0}} {{get_property SLACK $tp}} "
                f"else {{puts 0.0}} }} else {{ set tp [get_timing_paths -max_paths 1 "
                f"-slack_lesser_than 999]; if {{[llength $tp] > 0}} {{get_property SLACK $tp}} "
                f"else {{puts 0.0}} }}")
        return ("set tp [get_timing_paths -max_paths 1 -slack_lesser_than 999]; "
                "if {[llength $tp] > 0} {get_property SLACK $tp} else {puts 0.0}")


    def _observed_open_cost_s(self) -> float:
        """Max successful open_checkpoint cost observed this run (0.0 when
        none — the affordability margins cover small designs)."""
        cost = 0.0
        for _tc in self.tool_call_details:
            if not isinstance(_tc, dict) or _tc.get("error"):
                continue
            if (_tc.get("tool_name") == "vivado_open_checkpoint"
                    or "open_checkpoint" in str(_tc.get("cmd_head", ""))):
                try:
                    cost = max(cost, float(_tc.get("elapsed_time") or 0.0))
                except (TypeError, ValueError):
                    pass
        return cost


    # Flags whose state changes what the optimizer DOES, as opposed to tuning
    # knobs. Recorded explicitly (resolved, not just "present in environ") so a
    # banked row can be read back years later without guessing.
    _MANIFEST_FLAGS = (
        "FPL26_ILS_PLACE_RETRY_LADDER", "FPL26_ILS_MEASURED_BASIS",
        "FPL26_ILS_INCR_ROUTE", "FPL26_ILS_INCR_ROUTE_FIRST",
        "FPL26_ILS_INCR_ROUTE_TERMINAL", "FPL26_ILS_LADDER_ORDER_BY_WNS",
        "FPL26_ILS_LADDER_RESERVE", "FPL26_ILS_LADDER_STOP_ON_ACCEPT",
        "FPL26_ILS_RETRY_BASELINE_GATE", "FPL26_ILS_REJECT_MICRO_ACCEPT",
        # Any flag that can change a decision belongs in this list.  The
        # effective-config line is the runtime firing check, so a flag missing
        # from it reads as "the flag was off" when it is in fact no evidence
        # either way — a probe that cannot report its own treatment is broken
        # rather than null.
        "FPL26_ILS_MEASURED_PRIORS", "FPL26_ILS_SEED_COPY",
        "FPL26_ILS_HURDLE_CONTINUE", "FPL26_DEEP_REPLACE",
        "FPL26_DEEP_REPLACE_FIRST", "FPL26_DEEP_REPLACE_FIRST_BANDED",
        "FPL26_DEEP_REPLACE_UNBANDED",
        # The size-gated FIRST waiver and its kill switch. Both listed
        # so the "[effective-config] ON:" firing check can report the
        # treatment either way.
        "FPL26_DEEP_FIRST_SIZEGATED", "FPL26_NO_DEEP_FIRST_SIZEGATED",
        "FPL26_DEEP_REPLACE_NO_DOUBLE_RESERVE",
        "FPL26_PREEMPT_LOOP_CLOCK",
        "FPL26_PLATEAU_PRELOOP_FIX", "FPL26_VIVADO_LEAK_FIX",
        "FPL26_PHYSOPT_DEFAULT_FIXPOINT",
        # Same reason as above: without these, a banked row cannot prove which
        # gates were active during the run that produced it, which is exactly
        # the property this list exists to guarantee.
        "FPL26_ILS_REDRAW_RESERVE", "FPL26_ILS_LADDER_SKIP_UNAFFORDABLE",
        "FPL26_ILS_ENDHIGH_DENSITY_GATE", "FPL26_DEEP_REPLACE_B2_RESTART",
        "FPL26_PROMPT_V2",
        "FPL26_NO_DEEP_REPLACE", "FPL26_NO_DEEP_REPLACE_FIRST",
        "FPL26_NO_TAIL_CONTROLLER", "FPL26_NO_PLAN_CRITIC",
        "FPL26_NO_ROUTE_REROLL", "FPL26_NO_REPLACE_GAMBLE",
        "FPL26_NO_BARE_REROUTE_POLISH",
        # Recipe pass (frozen-Tcl candidate pass, DEFAULT OFF) — the
        # flag changes what the optimizer DOES, so it belongs here (the
        # omission class bit twice; see the comments above).
        "FPL26_RECIPE_PASS", "FPL26_NO_RECIPE_PASS",
        # Displacement-free incr-route escalation (DEFAULT OFF) —
        # changes an ILS combo-pick DECISION.
        "FPL26_SHALLOW_ESCALATION_GUARD",
        # shallowest sub-band in-session floor (DEFAULT OFF) —
        # changes the pre-LLM pipeline AND
        # carves the sub-band out of the shallow recipe pass.
        "FPL26_SUBBAND_PHYSOPT_FLOOR", "FPL26_NO_SUBBAND_PHYSOPT_FLOOR",
        # Candidate levers (all DEFAULT OFF):
        # each changes a DECISION (second shallow candidate / ILS retry
        # trigger scope / post-loop rung), so all three + kill switches
        # belong here per this block's own rule.
        "FPL26_ETO_RETIME_CANDIDATE", "FPL26_NO_ETO_RETIME_CANDIDATE",
        "FPL26_MIDBAND_RETRY_HOLD", "FPL26_NO_MIDBAND_RETRY_HOLD",
        "FPL26_MIDBAND_ROUTE_RUNG", "FPL26_NO_MIDBAND_ROUTE_RUNG",
        # Unified own-front retime candidate (DEFAULT OFF)
        # — supersedes the earlier
        # FPL26_SHALLOW_WLD_RETIME_CANDIDATE stacking design; changes a
        # DECISION (which retime candidate runs, and defers the ETO
        # one), so flag + kill switch belong here per this block's rule.
        "FPL26_OWNFRONT_RETIME_CANDIDATE",
        "FPL26_NO_OWNFRONT_RETIME_CANDIDATE",
        # Finalize-MUX md5-trust (DEFAULT OFF)
        # — changes the finalize verify DECISION
        # for a winning registered candidate, so flag + kill switch
        # belong here per this block's own rule.
        "FPL26_MUX_MD5_TRUST", "FPL26_NO_MUX_MD5_TRUST",
        # The optional third candidate for shallow negative slack is disabled
        # by default. Its enable and kill-switch flags control whether it
        # enters final selection.
        "FPL26_SHALLOW_DETERMINIZER_CANDIDATE",
        "FPL26_NO_SHALLOW_DETERMINIZER_CANDIDATE",
    )

    def _write_effective_config_manifest(self) -> None:
        """Record the effective runtime configuration beside the result artifacts.

        The manifest captures configuration resolved from build inputs and the
        process environment rather than relying on static defaults. `code_md5`
        fingerprints the orchestrator and every module under `optimizer/` so
        extracted or independently changed modules remain covered. Manifest
        generation is best-effort and must never terminate an optimization run.
        """
        try:
            def _md5(p: Path):
                try:
                    return hashlib.md5(p.read_bytes()).hexdigest()
                except OSError:
                    return None

            here = Path(__file__).resolve().parent
            flags = {}
            for name in self._MANIFEST_FLAGS:
                raw = os.environ.get(name)
                flags[name] = {
                    "set": raw is not None,
                    "raw": raw,
                    "on": (raw or "").strip().lower() in ("1", "true", "on", "yes"),
                }
            manifest = {
                "schema": "fpl26.effective_config.v1",
                "argv": sys.argv,
                "mode": getattr(self, "mode", None),
                "contest_mode": bool(getattr(self, "contest_mode", False)),
                "max_wall_seconds": self.max_wall_seconds,
                "code_md5": {
                    # The orchestrator plus every module carved out of it.
                    # Enumerated rather than listed, so a module added by a
                    # later extraction is fingerprinted without an edit here.
                    # The two original keys keep their exact spelling, so rows
                    # banked before this stay comparable.
                    "dcp_optimizer.py": _md5(here / "dcp_optimizer.py"),
                    **{
                        f"optimizer/{_p.name}": _md5(_p)
                        for _p in sorted((here / "optimizer").glob("*.py"))
                    },
                },
                "flags": flags,
                # Full FPL26_* environment, so a variable not yet listed
                # explicitly is still captured.
                "fpl26_env": {k: v for k, v in sorted(os.environ.items())
                              if k.startswith("FPL26_")},
            }
            on_now = sorted(n for n, s in flags.items() if s["on"])
            logger.info("[effective-config] ON: %s", ", ".join(on_now) or "(none)")
            run_dir = getattr(self, "run_dir", None)
            if run_dir:
                out = Path(run_dir) / "effective_config.json"
                out.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                               encoding="utf-8")
                logger.info("[effective-config] wrote %s", out)
        except Exception as exc:  # noqa: BLE001 - provenance is never fatal
            logger.warning("[effective-config] could not record manifest: %s", exc)

    async def optimize(self, input_dcp: Path, output_dcp: Path) -> bool:
        """Run the optimization workflow."""
        # Start timing the optimization process
        self.start_time = time.time()
        # ILS stagnation clock starts now (no improvement yet).
        self.last_improvement_time = self.start_time

        # Set up wall-time budget if a cap is configured.  The deadline is set
        # to (now + max_wall_seconds − finalize_reserve) so the controller has
        # time to write EDIF + structurally validate before the real cap.
        if self.max_wall_seconds is not None:
            self._budget_deadline = (
                self.start_time + self.max_wall_seconds - self._finalize_reserve_seconds
            )
            logger.info(
                f"Wall-time budget: {self.max_wall_seconds:.0f}s "
                f"(deadline at {self._finalize_reserve_seconds:.0f}s before kill; "
                f"effective optimization window: "
                f"{self.max_wall_seconds - self._finalize_reserve_seconds:.0f}s)"
            )

        # Stash baseline path so _finalize_output_dcp can use it for fallback.
        self.input_dcp_path = Path(input_dcp).resolve()

        # Wire the DCP write-path guard now that
        # run_dir + output_dcp are known.  Audit mode by default.
        self._ensure_path_guard(output_dcp=Path(output_dcp))

        # Emit run-start trace event so the
        # decisions.jsonl has a clear opening frame even before Phase 1
        # tool calls fire.  Cheap, runs once per optimize() invocation.
        self._emit_decision({
            "decision_source": "executor",
            "phase": "run_start",
            "action_label": "optimize_begin",
            "notes": (
                f"input={Path(input_dcp).name} output={Path(output_dcp).name} "
                f"mode={getattr(self, 'mode', None)} "
                f"max_wall_s={self.max_wall_seconds}"
            ),
            "output_dcp_path": str(Path(output_dcp).resolve()),
            "best_valid_checkpoint_path": str(Path(input_dcp).resolve()),
        })

        # Provenance BEFORE any work: record the configuration this run actually
        # resolved, so the row it banks can be read without guessing which arm it
        # was. See _write_effective_config_manifest for why this exists.
        self._write_effective_config_manifest()

        # Perform initial analysis without LLM
        try:
            initial_analysis = await self.perform_initial_analysis(input_dcp)
            # Emit the Phase 1 summary record once there is a measured
            # initial WNS/Fmax plus whatever RQA/router context
            # classify_design picked up.  This is the opening frame a
            # post-mortem reads: what the run saw before it acted.
            try:
                initial_fmax = self.calculate_fmax(
                    self.initial_wns, self.clock_period
                ) if (self.initial_wns is not None and self.clock_period) else None
                pathology = getattr(self, "design_pathology", None)
                pathology_label = None
                if isinstance(pathology, dict):
                    pathology_label = pathology.get("primary_label")
                else:
                    pathology_label = getattr(pathology, "primary_label", None)
                rqa = getattr(self, "qor_assessment", {}) or {}
                router_plan = getattr(self, "recipe_router_plan", None)
                router_rule = None
                if isinstance(router_plan, dict):
                    router_rule = router_plan.get("rule_id")
                else:
                    router_rule = getattr(router_plan, "rule_id", None)
                # Feature-first retrieval fields — emit
                # the same values the strategy_memory fingerprinter
                # consumes so backfill can key episodes by feature.
                _spread_info = getattr(self, "critical_path_spread_info", None)
                _spread_avg = (_spread_info.get("avg_distance")
                               if isinstance(_spread_info, dict) else None)
                self._emit_decision({
                    "decision_source": "executor",
                    "phase": "phase_1",
                    "action_label": "initial_analysis",
                    "tool_name": "perform_initial_analysis",
                    "wns_before": None,
                    "wns_after": self.initial_wns,
                    "fmax_after": initial_fmax,
                    "rqa_mode": "advisory" if rqa.get("score") is not None else "off",
                    "lut_count": getattr(self, "lut_count", None),
                    "critical_path_spread": _spread_avg,
                    "notes": (
                        f"pathology={pathology_label} "
                        f"router_rule={router_rule} "
                        f"target_clock={self.target_clock} "
                        f"phase1_skipped={self.phase1_skipped}"
                    ),
                })
                # If RQA produced a score, emit a separate compact RQA
                # context event so consumers can find it without
                # parsing notes.
                if rqa.get("score") is not None:
                    self._emit_decision({
                        "decision_source": "rqa",
                        "phase": "phase_1",
                        "action_label": "qor_assessment",
                        "rqa_mode": "advisory",
                        "notes": (
                            f"score={rqa.get('score')} "
                            f"flow_guidance={rqa.get('flow_guidance')} "
                            f"ml_strategy_available={rqa.get('ml_strategy_available')}"
                        ),
                    })
            except Exception as exc:  # pragma: no cover — defensive
                logger.debug(f"phase_1 trace emit failed: {exc}")
        except Exception as e:
            # Submission-safety contract: even when a mandatory Phase-1 step
            # fails — checkpoint open times out, Vivado crashes, the JNI layer
            # dies — a valid baseline copy must still be left at the output
            # path, so packaging does not fail on a missing file.  The EDIF
            # write is best-effort and can be retrofitted later.
            logger.exception(f"Initial analysis failed: {e}")
            print(f"\n✗ Initial analysis failed: {e}")
            print("Emitting baseline-copy DCP for submission safety...")
            try:
                self._path_guard_check(
                    output_dcp,
                    context="phase1_failed_baseline_copy",
                )
                _atomic_copy(str(self.input_dcp_path), str(output_dcp))
                try:
                    # Vivado may or may not still be reachable here; the
                    # call_tool layer either succeeds or raises, and the
                    # exception is simply swallowed.  No EDIF is acceptable — the DCP
                    # itself is still a valid submission artifact.
                    await self.call_tool("vivado_open_checkpoint", {
                        "dcp_path": str(output_dcp.resolve())
                    })
                    await self._write_readable_edif(output_dcp)
                except Exception as edif_e:
                    logger.warning(
                        f"Could not write EDIF after Phase 1 failure: {edif_e}"
                    )
                self.final_status = "VALID_FALLBACK_BASELINE_PHASE1_FAILED"
                self.lifecycle_log.append({
                    "event": "phase1_failed_baseline_copied",
                    "reason": str(e)[:200],
                })
                self._record_ship_lineage(
                    dcp_path=output_dcp,
                    edif_ok=False,
                    baseline=True,
                    branch="phase1_failed_baseline_copy",
                )
                print(f"[LIFECYCLE] {self.final_status}: Phase 1 failed; baseline "
                      f"copied to output for submission safety (ΔFmax = 0).")
            except Exception as copy_e:
                logger.exception(f"Baseline copy after Phase 1 failure also failed: {copy_e}")
                self.final_status = "HARD_FAIL_NO_VALID_BASELINE"
                self._record_ship_lineage(
                    dcp_path=output_dcp,
                    edif_ok=False,
                    hard_fail=True,
                    branch="phase1_failed_baseline_copy_failed",
                )
            self.end_time = time.time()
            return False
        
        # Timing already met is not the same as nothing left to win: the score
        # is a delta of an unclamped 1000/(T - WNS), so a met-timing input
        # still has every megahertz above 1000/T available.  With the flag
        # armed the run falls through into the normal path instead of shipping
        # the input unchanged.  Every development-corpus design enters with
        # negative slack, so this is unreachable there by construction.
        _met_decision = positive_slack_entry_decision(
            self.initial_wns, positive_slack_continue_enabled())
        if _met_decision == "optimize":
            # Arm the >=3-iteration plateau exit from loop start: nothing can
            # ever improve off a positive baseline, so last_improvement_iter
            # stays 0 and the plateau guard would otherwise be dead.
            self._positive_slack_entry_met = True
            _met_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
            print(f"✓ Design already meets timing (WNS {self.initial_wns:+.3f} ns"
                  + (f", Fmax {_met_fmax:.2f} MHz" if _met_fmax else "")
                  + ") — POSITIVE_SLACK_CONTINUE armed: optimizing anyway, "
                    "alpha is a delta of an UNCLAMPED 1000/(T-wns).\n")
            logger.info(
                "[positive-slack] met-timing input, continuing: "
                f"wns_in={self.initial_wns:+.3f} "
                f"fmax_in={_met_fmax if _met_fmax is not None else 'None'}")
        elif _met_decision == "early_exit":
            print("✓ Design already meets timing! No optimization needed.\n")
            logger.info("Design already meets timing")
            # Save the design as-is, then finalize (EDIF + validate).  Wrap in
            # try/except so a Vivado checkpoint failure here still falls through
            # to _finalize_output_dcp which will baseline-copy if needed.
            try:
                await self.call_tool("vivado_write_checkpoint", {
                    "dcp_path": str(output_dcp.resolve()),
                    "force": True
                })
                print(f"Saved design to: {output_dcp}\n")
            except Exception as e:
                logger.warning(f"write_checkpoint failed on timing-met early exit: {e}")

            # Finalize: write EDIF, validate, fall back to baseline if needed.
            # _finalize_output_dcp's no-improvement guard handles the missing
            # output case by copying baseline + EDIF.
            try:
                await self._finalize_output_dcp(output_dcp)
            except Exception as e:
                logger.exception(f"_finalize_output_dcp raised on early-exit path: {e}")
                # Last-resort baseline copy.
                try:
                    if not output_dcp.exists():
                        self._path_guard_check(
                            output_dcp, context="timing_met_baseline_last_resort",
                        )
                        _atomic_copy(str(self.input_dcp_path), str(output_dcp))
                    self.final_status = self.final_status or "VALID_FALLBACK_BASELINE"
                except Exception:
                    self.final_status = "HARD_FAIL_NO_VALID_BASELINE"

            # End timing
            self.end_time = time.time()
            total_runtime = self.end_time - self.start_time
            
            # Print summary even for early exit
            print("\n=== No Optimization Required ===")
            initial_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
            if initial_fmax is not None:
                print(f"Design already meets timing - Fmax: {initial_fmax:.2f} MHz (WNS: {self.initial_wns:.3f} ns)")
            else:
                print(f"Design already meets timing (WNS: {self.initial_wns:.3f} ns)")
            print(f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
            print(f"LLM API calls: 0 (analysis performed without LLM)")
            print(f"Estimated cost: $0.00")
            print("="*70 + "\n")
            return True
        
        # RECIPE-FIRST: for the published DEEP-extreme band
        # the forced recipe runs HERE — after Phase 1 has measured the physics
        # and the cell count, but before the LLM loop can spend the wall. It
        # cannot go later: it needs ~79% of the budget on the largest class, and the
        # loop leaves ~2%. Strict no-op unless both flags are on (default OFF).
        await self._maybe_run_recipe_first()

        # SUBBAND-FLOOR (FPL26_SUBBAND_PHYSOPT_FLOOR, DEFAULT OFF): scripted
        # in-session floor for the shallowest sub-band.  MUST sit between
        # recipe_first (whose in-memory state it builds on) and recipe_pass
        # (whose reset would wipe that state) — see the method's ordering
        # contract.  Strict no-op with the flag off.
        await self._maybe_run_fir_subband_floor()

        # Recipe pass (default off): a frozen-Tcl candidate pass at the
        # post-Phase-1, pre-LLM slot — the features exist, the pristine input
        # is still on disk, and the loop has spent nothing yet.  Banks only
        # through register_final_candidate and the finalize MUX, where
        # never-worse is enforced.  A strict no-op with the flag off.
        await self._maybe_run_recipe_pass(input_dcp)

        # Load and fill in system prompt with temp directory and input DCP path
        system_prompt_template = load_system_prompt()
        system_prompt = system_prompt_template.format(
            temp_dir=self.temp_dir,
            input_dcp=input_dcp.resolve()
        )

        # RAG seed: prior-run winner for this design (or fingerprint match).
        # Empty string when memory is unavailable or no usable record exists —
        # the prompt section is omitted in that case.
        rag_section = ""
        recipe_hint = ""
        design_name = None
        if _design_name_from_dcp is not None:
            design_name = _design_name_from_dcp(input_dcp)
        contest_mode = bool(getattr(self, "contest_mode", False))
        if _seed_prompt_for is not None and getattr(self, "rag_seed", True):
            spread = None
            if isinstance(getattr(self, "critical_path_spread_info", None), dict):
                spread = self.critical_path_spread_info.get("avg_distance")
            try:
                rag_section = _seed_prompt_for(
                    design_name=design_name,
                    lut_count=getattr(self, "lut_count", None),
                    critical_path_spread=spread,
                    contest_mode=contest_mode,
                )
            except Exception as e:
                logger.warning(f"strategy_memory lookup failed: {e}")
                rag_section = ""
            # Negative-memory advisory block in
            # contest mode.  Appended to the RAG section so the LLM
            # sees "what to avoid" alongside "what worked".
            if contest_mode and _negative_memory_block is not None:
                try:
                    neg = _negative_memory_block(
                        lut_count=getattr(self, "lut_count", None),
                        critical_path_spread=spread,
                    )
                    if neg:
                        rag_section = (rag_section + "\n\n" + neg
                                       if rag_section else neg)
                except Exception as e:
                    logger.warning(f"negative_memory_block raised: {e}")
            # Emit retrieval metadata into the decision trace.
            if _retrieval_metadata_for is not None:
                try:
                    meta = _retrieval_metadata_for(
                        design_name=design_name,
                        lut_count=getattr(self, "lut_count", None),
                        critical_path_spread=spread,
                        contest_mode=contest_mode,
                    )
                    self._emit_decision({
                        "decision_source": "rag",
                        "phase": "phase_1",
                        "action_label": "rag_retrieval",
                        "rag_mode": meta.get("rag_mode"),
                        "notes": (
                            f"retrieval_mode={meta.get('retrieval_mode')} "
                            f"design_notes_injected={meta.get('design_notes_injected')} "
                            f"exact_name_used={meta.get('exact_name_used')} "
                            f"retrieved_episode_ids={meta.get('retrieved_episode_ids')} "
                            f"negative_memory_count={meta.get('negative_memory_count')} "
                            f"memory_records_considered={meta.get('memory_records_considered')}"
                        ),
                    })
                except Exception as e:
                    logger.warning(f"retrieval_metadata_for raised: {e}")
            # Advisory policy card (opt-in).  Default off.
            # Appends a compact POLICY_MEMORY_CARD block to the iter-1
            # prompt when the user passed --policy-card.  The card is
            # feature-first only and never echoes design names.
            policy_card_enabled = bool(getattr(self, "policy_card", False))
            if policy_card_enabled:
                try:
                    from optimizer.policy_card import render_policy_card
                    card = render_policy_card(
                        lut_count=getattr(self, "lut_count", None),
                        critical_path_spread=spread,
                        contest_mode=contest_mode,
                        variant=getattr(self, "policy_card_variant", "default"),
                    )
                    if card.get("text"):
                        rag_section = (rag_section + "\n\n" + card["text"]
                                       if rag_section else card["text"])
                    self._emit_decision({
                        "decision_source": "rag",
                        "phase": "phase_1",
                        "action_label": "policy_card",
                        "rag_mode": ("contest_mode" if contest_mode
                                     else "normal"),
                        "notes": (
                            f"policy_card_enabled=True "
                            f"matched_episode_ids={card.get('matched_episode_ids')} "
                            f"negative_memory_ids={card.get('negative_memory_ids')} "
                            f"matched_count={card.get('similarity_basis',{}).get('matched_count')} "
                            f"positive_count={card.get('similarity_basis',{}).get('positive_count')} "
                            f"beat_ship_count={card.get('similarity_basis',{}).get('beat_ship_count')} "
                            f"confidence={card.get('confidence')} "
                            f"card_token_estimate={card.get('card_token_estimate')}"
                        ),
                    })
                except Exception as e:
                    logger.warning(f"policy_card raised: {e}")
        elif _seed_prompt_for is not None:
            logger.info(
                "RAG seed disabled (--no-rag-seed); skipping iter-1 memory lookup "
                f"(contest_mode={contest_mode})"
            )
        # Recipe applicability: tell the model up front when the
        # cell-replacement recipe is known-harmful on this design, so it does
        # not spend tokens on detour analysis there.  Gated off in contest
        # mode — it is a design-name-keyed rule, and the hidden-benchmark rule
        # forbids those.  Without it the model may waste a few iterations; the
        # trade is deliberate, generality over a benchmark-specific shortcut.
        if (_recipe_safe_for is not None and design_name is not None
                and not contest_mode):
            if not _recipe_safe_for(design_name):
                recipe_hint = (
                    "\nRECIPE GATE: cell-replacement recipe (analyze_net_detour + "
                    f"optimize_cell_placement) is BLOCKED for '{design_name}' — "
                    "prior sweeps showed it produces route errors or hangs here.  "
                    "Choose a different strategy class."
                )

        # Compose the iter-1 user message (pure function of its inputs —
        # the LLM-invisibility acceptance check depends on that; see
        # compose_iter1_user_message).
        iter1_user_message = compose_iter1_user_message(
            input_dcp=input_dcp,
            output_dcp=output_dcp,
            temp_dir=self.temp_dir,
            initial_analysis=initial_analysis,
            rag_section=rag_section,
            recipe_hint=recipe_hint,
        )

        # Initialize conversation with analysis results
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": iter1_user_message},
        ]
        # Stash for end-of-run RAG persistence
        self._design_name_for_memory = design_name
        self._initial_fmax_for_memory = self.calculate_fmax(self.initial_wns, self.clock_period)
        
        max_iterations = 50  # Safety limit
        
        print("=== Starting LLM-Driven Optimization ===\n")
        
        # Loop-exit reason tracking.  The success return value must follow the
        # finalize outcome, not the exit path: a clean budget-exhausted break
        # after a successful improvement is still a success, and returning
        # False there makes downstream tooling read a shipped result as a
        # failure.
        loop_exit_reason: str = "max_iterations"

        # ILS stagnation clock: re-anchor to the start of the LLM loop.
        #
        # optimize() sets last_improvement_time to the process start, so the
        # clock counts every pre-loop stage — Phase-1 analysis and, when the
        # deep-replace-first probe is armed, a full place-and-route.  On a
        # large design that probe can outlast the stagnation threshold on its
        # own, so the preempt fires within milliseconds of the first iteration,
        # with zero model calls made, and the recipe never executes a step.
        #
        # A loop with zero iterations is neither productive nor stalled: the
        # gate cannot tell, and was being handed a verdict formed before the
        # loop existed.  Measuring stall from loop start restores the intent.
        #
        # Deliberately narrow — it changes what the preempt concludes, not what
        # work runs, so designs that benefit from the pre-loop probe are
        # untouched, and designs that genuinely need ruin-and-recreate still
        # get it from the at-exit path whenever the loop ends with timing
        # unmet.
        if resolve_preempt_loop_clock_enabled():
            self.last_improvement_time = time.time()

        # Pre-loop plateau exit (default off): the iteration-domain twin of the
        # clock fix above.
        #
        # The plateau exit is armed by last_improvement_iter > 0, a sentinel
        # that conflates two states calling for opposite decisions:
        #   (a) nothing achieved yet and the model is still ramping up — must
        #       NOT exit; cutting a run off here ships no gain at all;
        #   (b) the best slack was banked BEFORE iteration 1, so the loop has
        #       run and produced nothing — the textbook plateau this guard
        #       exists to catch.
        # In state (b) last_improvement_iter stays 0 forever, the guard is
        # structurally unreachable, and only the empty-spin backstop can stop
        # the loop.
        #
        # State (b) is now the default path, because the pre-loop re-place
        # banks its result before the loop starts.  The cost is iterations and
        # model calls that bank nothing — pure spend and wall against identical
        # gain.
        #
        # Deliberately narrow: it arms an EXISTING exit rather than changing
        # what work runs, and the exit path still runs the polish stages and
        # finalize at no model cost.  When it is wrong, the cost is forgone
        # iterations on top of an already-banked best, never a lost result.
        self._plateau_preloop_fix = (
            os.environ.get("FPL26_PLATEAU_PRELOOP_FIX", "0").strip().lower()
            in ("1", "true", "on", "yes"))
        self._pre_loop_best_banked = plateau_armed_from_loop_start(
            self._plateau_preloop_fix, self.initial_wns, self.best_wns,
            getattr(self, "_positive_slack_entry_met", False))
        if self._pre_loop_best_banked:
            logger.info(
                f"[PLATEAU_PRELOOP] best WNS {self.best_wns:.3f} ns was banked "
                f"before iteration 1 (baseline {self.initial_wns:.3f} ns) — the "
                f"plateau exit is armed from loop start, so 3 iterations with no "
                f"further improvement will hand over to the zero-cost tail.")

        while self.iteration < max_iterations:
            # LOGIC-FLOOR TERMINATION: attested pre-loop — no LLM
            # iteration ever starts; the exit tail finalizes the floor.
            if getattr(self, "_logic_floor_terminate", None):
                logger.info(
                    f"[logic-floor] terminating before LLM iteration "
                    f"{self.iteration + 1}: {self._logic_floor_terminate}")
                loop_exit_reason = "logic_floor"
                break
            # Compensation anchor — when the LLM call
            # fails with an API-error episode, the except handler restores
            # self.iteration to this value so a storm never burns iteration
            # budget (both repro logs burned 8+ iterations in <1s).
            _iter_loop_entry = self.iteration
            self.iteration += 1
            logger.info(f"=== Iteration {self.iteration} ===")

            # Wall-time budget check — stop cleanly when time is out.
            # _finalize_output_dcp runs after the loop, so the best valid
            # output (whatever was achieved so far) is preserved.
            if self._budget_exhausted():
                logger.warning(
                    f"Wall-time budget exhausted after "
                    f"{time.time()-self.start_time:.0f}s of "
                    f"{self.max_wall_seconds:.0f}s; finalizing with best="
                    f"{self.best_wns:.3f} ns."
                )
                loop_exit_reason = "budget_exhausted"
                break
            # Deadline-aware dispatcher flipped _budget_killed because a
            # tool call was skipped or timed out — also stop cleanly, rather
            # than burn LLM tokens on impossible work.
            if self._budget_killed:
                logger.warning(
                    "Budget-killed flag set by tool dispatcher; finalizing "
                    f"with best={self.best_wns:.3f} ns."
                )
                # Distinct label when the kill was caused by the
                # polish-reserve runtime fence (behavior identical — the
                # tail still runs from fresh Vivado; forensics only).
                loop_exit_reason = (self._budget_kill_cause
                                    or "budget_killed")
                break
            # Deep-slack tail reserve (default off).  Left alone, this loop can
            # run the entire wall and leave the deterministic exit tail zero
            # seconds, so the tail's wall-fit gate refuses — even though on
            # deeply-negative designs that tail has a measurable gradient while
            # late model iterations there mostly produce nothing.  When armed
            # and the design is still deeply negative — by CURRENT best slack,
            # not entry slack — stop iterating once the remaining wall falls to
            # the reserve and hand the rest to the shared exit tail below.
            if self._tail_reserve_break_due():
                logger.info(
                    f"[tail-reserve] deep-WNS reserve reached: exiting "
                    f"LLM loop with {self._budget_remaining():.0f}s "
                    f"remaining for the deterministic tail (best_wns="
                    f"{self.best_wns:.3f} ns, reserve="
                    f"{self._deep_wns_tail_reserve_effective_s():.0f}s).")
                loop_exit_reason = "tail_reserve"
                break
            # Wall economics: the derived stop rule arms the same handback
            # signal from arithmetic on the scoring function rather than from a
            # sub-mechanism's saturation verdict.  Evaluated after the
            # tail-reserve check so deeply-negative designs keep their
            # precedence: that path re-allocates the wall to a tail with a
            # measured gradient, this one hands it back.  Observe-only unless
            # --wall-handback is passed.
            self._maybe_arm_wall_economics_stop()
            # Wall handback (default off): a sanctioned saturation signal armed
            # the early-exit reason, and its guard guarantees a banked accept
            # exists.  Exit to the finalize tail now instead of re-entering the
            # loop — the polish stages still run once, then finalize and
            # process exit return the remaining wall to the wrapper.
            if self._wall_handback_break_due():
                logger.info(
                    f"Wall-handback: exiting LLM loop early "
                    f"({self._exit_early_reason}); finalizing with best="
                    f"{self.best_wns:.3f} ns and returning remaining "
                    f"{self._budget_remaining():.0f}s to the wrapper."
                )
                loop_exit_reason = "wall_handback"
                break

            # ILS-polish preempt: if the loop has stalled and the design is
            # size-viable with budget for at least two ruin-and-recreate
            # cycles, stop the stuck loop and switch the remaining budget to
            # ILS.  It never fires on a productive loop, so it cannot steal
            # time from a design the recipe is still improving, and keep-best
            # makes it never-worse.
            if (self._ils_polish_cfg.enabled
                    and not self._ils_preempt_requested
                    and self.best_wns > float("-inf")):
                from optimizer.ils_polish import should_trigger as _ils_should
                _secs_since = time.time() - (self.last_improvement_time
                                             or self.start_time)
                # Lazily measure design size once stagnation is in range (design
                # is open mid-loop). Used only for the size gate.
                if (self._design_cells is None
                        and _secs_since >= self._ils_polish_cfg.stagnation_seconds):
                    try:
                        _cres = await self.call_tool("vivado_run_tcl", {
                            "command": "llength [get_cells -hierarchical "
                                       "-filter {IS_PRIMITIVE==1}]"})
                        _cm = re.search(r"(\d+)", _cres or "")
                        self._design_cells = int(_cm.group(1)) if _cm else -1
                    except Exception:
                        self._design_cells = -1
                _trig, _why = _ils_should(
                    cells=self._design_cells,
                    remaining_s=self._budget_remaining(),
                    seconds_since_improve=_secs_since,
                    best_wns=self.best_wns,
                    baseline_wns=self.initial_wns,
                    cfg=self._ils_polish_cfg,
                )
                # Ledger the verdict either way.  A gate that emits nothing on
                # the path that matters cannot be used to find the defect it
                # exists to surface.  The iteration number is the field that
                # matters here — a refusal at iteration 1 means the loop was
                # cut off before it ran at all.
                _gl.emit("ils_preempt",
                         _gl.VERDICT_REFUSE if _trig else _gl.VERDICT_ALLOW,
                         design=getattr(self, "_design_name_for_memory", None),
                         iteration=self.iteration,
                         reason_code=_why,
                         observed_s=_secs_since,
                         threshold_s=float(self._ils_polish_cfg.stagnation_seconds),
                         remaining_wall_s=self._budget_remaining(),
                         best_wns_ns=self.best_wns,
                         site="dcp_optimizer.ils_polish_preempt")
                if _trig:
                    self._ils_preempt_requested = True
                    loop_exit_reason = "ils_preempt"
                    logger.info(
                        f"ILS-polish preempt ({_why}); switching remaining "
                        f"{self._budget_remaining():.0f}s to ruin-and-recreate."
                    )
                    break

            # Mirror the latest improved state to a stable best_valid path, so
            # the emergency finalize path has a Vivado-independent fast path to
            # ship.  Done here, between iterations, rather than inside
            # call_tool's detection branch, so the dispatcher is never
            # re-entered with half-built state.  Skipped when nothing has
            # improved since the last mirror.
            if self._pending_best_mirror:
                await self._mirror_best_valid_now()
            # Budget-low warning — give the LLM one nudge to wrap up cheaply
            # when <= 10 min remain (only injected once per run).
            if (not self._budget_warned_low
                    and self._budget_deadline is not None
                    and self._budget_remaining() <= 600.0):
                self._budget_warned_low = True
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"WALL-TIME BUDGET WARNING: only "
                        f"{self._budget_remaining():.0f}s of optimization budget "
                        f"remain.  Prioritize cheap, high-confidence moves and a "
                        f"final vivado_write_checkpoint to the output path. "
                        f"AVOID starting any expensive directive (place_design "
                        f"Explore/Auto_X, full route_design Explore) — they "
                        f"won't finish."
                    ),
                })

            try:
                # Snapshot tool-call count to detect empty-iter behaviour.
                _tool_calls_before = len(self.tool_call_details)
                response_text, is_done = await self.get_completion()
                # A successful completion ends any permanent-death streak and,
                # at most once per resolved episode, summarizes the absorbed
                # storm into the conversation.  The episode number makes each
                # summary unique and the watermark guarantees no text is ever
                # appended twice.  Unresolved episodes append nothing — the
                # model would never get to read it.
                self._consecutive_permanent_api_failures = 0
                if (self._api_resilience_enabled
                        and self._api_error_episodes
                        > self._last_summarized_episode):
                    _n_new_episodes = (self._api_error_episodes
                                       - self._last_summarized_episode)
                    self._last_summarized_episode = self._api_error_episodes
                    self.messages.append({
                        "role": "user",
                        "content": (
                            f"[api-status] {_n_new_episodes} LLM API error "
                            f"episode(s) were absorbed by backoff-retry "
                            f"(episode #{self._api_error_episodes} resolved; "
                            f"{self._total_backoff_s:.0f}s total backoff this "
                            f"run). The API recovered — continue your "
                            f"optimization plan."
                        ),
                    })
                _tool_calls_this_iter = len(self.tool_call_details) - _tool_calls_before
                if _tool_calls_this_iter == 0:
                    self._consecutive_empty_iters += 1
                else:
                    self._consecutive_empty_iters = 0
                print(f"\n{response_text}\n")

                # MODE: anchor — upstream's stop logic (stop iff LLM signals done).
                if self.mode == "anchor":
                    # Cost circuit-breaker for anchor mode, which otherwise has
                    # no cost exit at all and can spend a run straight to the
                    # per-benchmark cap.  Same deterministic
                    # finalize-with-banked-best as anchor's done path; anchor
                    # never uses the ILS tail.
                    if self._llm_cost_breached():
                        logger.info(
                            f"Anchor-mode cost exit: ${self.total_cost:.2f} >= "
                            f"${self.llm_cost_exit_usd:.2f}; finalizing with "
                            f"banked best.")
                        await self._finalize_output_dcp(output_dcp)
                        self.end_time = time.time()
                        self._print_optimization_summary()
                        return True
                    if is_done:
                        logger.info("Optimization workflow completed (anchor mode: LLM signalled done)")
                        await self._finalize_output_dcp(output_dcp)
                        self.end_time = time.time()
                        self._print_optimization_summary()
                        return True
                    # otherwise fall through to next iteration
                else:
                    # Stop conditions, applied after every iteration and not
                    # gated on the model signalling done.  Two cases:
                    #   1. timing met — always stop;
                    #   2. a plateau after at least one improvement — accept it.
                    # The cap is gated on having seen an improvement because an
                    # ungated cap also fires while the model is still in
                    # analysis mode, cutting a run off after a couple of
                    # iterations with nothing banked.  Gated, those designs get
                    # the full iteration budget instead.
                    iters_since_improve = self.iteration - self.last_improvement_iter
                    # Crossing zero slack is not the finish line: Fmax =
                    # 1000/(T - WNS) is unclamped in the reference
                    # implementation, so every megahertz past the crossing is
                    # still worth score.  With the flag armed this stop is
                    # retired and the loop terminates on the same guards every
                    # other design uses — cost breach, the plateau exit, the
                    # empty-spin guard, max iterations, the wall deadline —
                    # none of which is removed here.
                    if self.best_wns >= 0 and not positive_slack_continue_enabled():
                        logger.info(f"Optimization workflow completed (timing met: best_wns={self.best_wns:.3f} ns)")
                        await self._exit_with_ils_polish(output_dcp)
                        return True
                    if self.best_wns >= 0 and not self._positive_slack_logged:
                        self._positive_slack_logged = True
                        logger.info(
                            "[positive-slack] crossed zero at iteration "
                            f"{self.iteration} (best_wns={self.best_wns:+.3f} ns"
                            f", initial={self.initial_wns}) — CONTINUING; the "
                            "plateau / empty-spin / cost / wall guards still "
                            "own termination.")
                    # In-attempt cost exit.  The prompt guard keeps the model
                    # alive for the whole run, so spend has to be capped
                    # explicitly; the polish stages and finalize cost nothing.
                    # The threshold is the instance value — the configured
                    # ceiling less what earlier attempts already billed — so
                    # attempt N cannot re-spend a full fresh allowance on top
                    # of them.
                    if self._llm_cost_breached():
                        logger.info(
                            f"Optimization loop exiting on LLM cost: "
                            f"${self.total_cost:.2f} >= "
                            f"${self.llm_cost_exit_usd:.2f} "
                            f"(best_wns={self.best_wns:.3f} ns); handing over to "
                            f"the zero-cost tail (ILS polish + finalize).")
                        await self._exit_with_ils_polish(output_dcp)
                        return True
                    # The pre-loop-banked disjunct (default off) arms this exit
                    # for the case where the best was banked before the loop,
                    # which pins last_improvement_iter at 0 forever and makes
                    # the guard structurally unreachable.  With it at 0 the
                    # iterations-since-improvement count equals the iteration
                    # count, so this fires after three iterations that added
                    # nothing on top of an already-banked best.
                    if ((self.last_improvement_iter > 0
                         or getattr(self, "_pre_loop_best_banked", False))
                            and iters_since_improve >= 3):
                        _since = (f"since iter {self.last_improvement_iter}"
                                  if self.last_improvement_iter > 0
                                  else "since loop start (best was banked pre-loop)")
                        logger.info(f"Optimization workflow completed (no improvement for {iters_since_improve} iters {_since}; best_wns={self.best_wns:.3f} ns)")
                        await self._exit_with_ils_polish(output_dcp)
                        return True
                    # Empty-spin guard: when nothing has improved and the model
                    # has been signalling done with no tool calls for several
                    # consecutive iterations, terminate.  This saves wall on
                    # designs already at their ceiling.  The threshold is high
                    # enough to let force-continue prod the model once or twice
                    # first.
                    if (self.last_improvement_iter == 0
                            and self.iteration >= 8
                            and self._consecutive_empty_iters >= 5):
                        logger.info(
                            f"Optimization stopping early: "
                            f"{self._consecutive_empty_iters} consecutive empty iters "
                            f"with no improvement ever (iter {self.iteration}/"
                            f"{max_iterations}, best_wns={self.best_wns:.3f} ns). "
                            f"Saves wall time on ceiling designs."
                        )
                        await self._exit_with_ils_polish(output_dcp)
                        return True

                    # [BETA-CTRL-V0] Slope-aware iteration-ceiling lift.
                    # When the LLM signals stop but WNS is still improving
                    # recently, inject a user message and continue.  The stop
                    # conditions above already handle "timing met" and "true
                    # plateau"; this only fires when the LLM gives up too early.
                    if is_done:
                        self.force_continue_count += 1
                        logger.info(f"[BETA-CTRL-V0] LLM signalled stop but WNS improved {iters_since_improve} iter(s) ago (best={self.best_wns:.3f} ns) — force-continue #{self.force_continue_count}")
                        # Cross-model steering: when enabled, and the design has
                        # a historical target, and less than half of the prior
                        # model's gain has been recovered, and enough budget
                        # remains, use the specific anti-premature-exit hint
                        # instead of the generic message.  Falls back to the
                        # generic prompt otherwise, so behaviour is unchanged
                        # when steering is off.
                        current_gain_mhz = 0.0
                        if (self.initial_wns is not None
                                and self.clock_period
                                and self.best_wns is not None):
                            try:
                                pre_fmax = self.calculate_fmax(self.initial_wns, self.clock_period) or 0.0
                                post_fmax = self.calculate_fmax(self.best_wns, self.clock_period) or 0.0
                                current_gain_mhz = float(post_fmax - pre_fmax)
                            except Exception:
                                current_gain_mhz = 0.0
                        remaining_s = float(self._budget_remaining() or 0.0)
                        steering_msg = _steering_continue_hint(
                            getattr(self, "_design_name_for_memory", None),
                            current_gain_mhz,
                            remaining_s,
                        )
                        if steering_msg is not None:
                            self.messages.append({"role": "user", "content": steering_msg})
                        else:
                            # Router-aware continuation: when the active router
                            # plan recommends a heavy step, the model has not
                            # attempted it, at least five minutes of budget
                            # remain, there is no clear regression, and a valid
                            # session still exists, inject a continuation
                            # prompt naming the missed step.
                            #
                            # The no-regression gate admits exactly zero gain,
                            # not just positive gain: the case this exists to
                            # catch is a run that has improved nothing yet and
                            # has left the recommended heavy step unattempted —
                            # typically many cheap phys_opt calls, no placement,
                            # and an exit with most of the budget unspent.
                            should_fire, unattempted = self.should_inject_v05_router_nudge(
                                current_gain_mhz, remaining_s)
                            if should_fire:
                                logger.info(
                                    f"[BETA-CTRL-V0.5] router-step-aware "
                                    f"continuation TRIGGERED: best={self.best_wns:.3f} ns "
                                    f"(gain={current_gain_mhz:.2f} MHz), "
                                    f"remaining={remaining_s:.0f}s, "
                                    f"unattempted heavy step: {unattempted}"
                                )
                                # Slope phrasing — distinguish positive vs
                                # flat for the LLM-facing message.
                                if current_gain_mhz > 0.0:
                                    slope_msg = "current slope is positive"
                                else:
                                    slope_msg = ("current slope is flat (no "
                                                 "improvement yet, no "
                                                 "regression either)")
                                self.messages.append({
                                    "role": "user",
                                    "content": (
                                        f"You signalled completion at +{current_gain_mhz:.2f} MHz, "
                                        f"but the deterministic feature-based router "
                                        f"(rule {getattr(self.recipe_router_plan, 'rule_id', '?')}) "
                                        f"recommends a heavy step you have NOT attempted yet "
                                        f"in this run: {unattempted}. "
                                        f"You have {remaining_s/60:.0f} min wall-time remaining "
                                        f"(threshold: 5 min) and {slope_msg}. "
                                        f"Run that exact tool with that directive now, then "
                                        f"measure WNS. If it regresses, revert; otherwise "
                                        f"continue to subsequent router steps. Do NOT stop "
                                        f"without attempting this step at least once."
                                    )
                                })
                            else:
                                self.messages.append({
                                    "role": "user",
                                    "content": (
                                        f"You signalled completion, but timing has been improving recently — "
                                        f"best WNS is {self.best_wns:.3f} ns and it improved at iteration {self.last_improvement_iter} "
                                        f"(now iteration {self.iteration}).  "
                                        # On the POSITIVE_SLACK_CONTINUE class
                                        # this message used to assert "Timing
                                        # is NOT met yet" while WNS was
                                        # positive — a falsehood the model can
                                        # check against its own tool output, in
                                        # the one class where it most needs to
                                        # keep working. State the real reason.
                                        + (f"Timing is MET, but Fmax = 1000/(T - WNS) is NOT capped at the "
                                           f"target frequency: every extra picosecond of POSITIVE slack is "
                                           f"more Fmax and more score.  Keep going.  "
                                           if self.best_wns >= 0 else
                                           f"Timing is NOT met yet.  Don't stop.  ") +
                                        f"Try a different strategy: a different phys_opt directive (Explore, AggressiveExplore, "
                                        f"AlternateFlowWithRetiming, AggressiveFanoutOpt), a placement directive variant, "
                                        f"a route_design directive (Explore, ExploreWithAggressiveHoldFix), targeted re-routing "
                                        f"of remaining critical-path nets, fanout splits, or surgical unplace/re-place of "
                                        f"the worst critical-path cells.  Choose one and apply it.  What's your next move?"
                                    )
                                })
                        # Fall through to next iteration of the while loop.

            except Exception as e:
                # API-error episodes cost wall-clock only — never iterations,
                # never conversation pollution.  Storms are absorbed inside the
                # call path; what reaches this handler is the permanent case,
                # one propagation per episode whose backoff cap was exhausted.
                # Classification mirrors the call path; with the kill switch
                # off, or the soft import unavailable, the legacy path below
                # applies.
                _api_kind = None
                if (self._api_resilience_enabled
                        and _classify_api_error is not None):
                    _api_kind = _classify_api_error(f"{type(e).__name__}: {e}")
                if _api_kind in ("key_auth", "transient"):
                    # Iteration-burn compensation: restore the
                    # loop-entry value, bounded so it can never go below it
                    # (nor below zero — _iter_loop_entry is always >= 0).
                    if self.iteration > _iter_loop_entry:
                        self.iteration -= 1
                    self._consecutive_permanent_api_failures += 1
                    # Single-line per-episode forensic log: one
                    # greppable line naming the episode's failure count and
                    # the accumulated backoff.  NO conversation append —
                    # the per-failure "An error occurred" junk was the
                    # prompt-growth amplifier in both repro logs.
                    logger.warning(
                        f"[api-resilience] episode "
                        f"#{self._api_error_episodes} permanent ({_api_kind}): "
                        f"{self._api_last_episode_failures} failure(s), "
                        f"{self._total_backoff_s:.0f}s total backoff this run; "
                        f"iteration not consumed (compensated to "
                        f"{self.iteration}); consecutive permanent episodes: "
                        f"{self._consecutive_permanent_api_failures}."
                    )
                    if (self._consecutive_permanent_api_failures
                            >= PERMANENT_API_FAILURE_LIMIT):
                        # Defensive termination bound: with compensation the
                        # while-guard can't expire, and with no wall cap the
                        # budget guards never fire — break to the finalize
                        # tail, which ships Phase-1 banked state when any
                        # exists (_exit_with_ils_polish -> _finalize_output_dcp).
                        logger.warning(
                            f"[api-resilience] "
                            f"{self._consecutive_permanent_api_failures} "
                            f"consecutive permanent API-error episodes — LLM "
                            f"path is dead; finalizing with banked state."
                        )
                        loop_exit_reason = "api_dead"
                        break
                else:
                    # Legacy path (non-API exceptions, e.g. a genuine
                    # tool/logic error the LLM must see) — unchanged.
                    logger.exception(f"Error during optimization: {e}")
                    # Add error context to conversation
                    self.messages.append({
                        "role": "user",
                        # Truncated: appending FULL provider errors grew the
                        # prompt on every failed call — the 402 doom-loop amplifier.
                        "content": f"An error occurred: {str(e)[:300]}. Please verify your approach and continue or report if unrecoverable."
                    })
        
        # Post-loop finalize.  Reached when the loop broke out on budget, or
        # when the iteration limit expired.  The success return value follows
        # the finalize outcome rather than the exit path taken — otherwise a
        # budget-killed exit that had already shipped a valid optimized result
        # reports failure.
        if loop_exit_reason == "max_iterations":
            logger.warning("Reached maximum iterations")
        else:
            logger.info(
                f"Optimization loop exited via {loop_exit_reason}; "
                f"running finalize with best_wns={self.best_wns:.3f} ns."
            )
        # Single shared exit tail: arms the ILS-polish loop-exit trigger (if the
        # loop ended at wns<0 with budget to spare), runs the ILS stage if armed
        # (or if a mid-loop stagnation preempt already armed it), then finalizes.
        await self._exit_with_ils_polish(
            output_dcp,
            max_iterations_reached=(loop_exit_reason == "max_iterations"),
        )
        # Success contract: finalize wrote a usable ship artifact.
        # final_status values that count as success:
        #   VALID_OPTIMIZED, VALID_OPTIMIZED_NO_EDIF,
        #   VALID_FALLBACK_BASELINE, VALID_FALLBACK_BASELINE_NO_EDIF,
        #   VALID_FALLBACK_BASELINE_PHASE1_FAILED
        # NO_IMPROVEMENT and HARD_FAIL_* are failures.
        status = getattr(self, "final_status", None) or ""
        return status.startswith(("VALID_OPTIMIZED", "VALID_FALLBACK_BASELINE"))
    
    def save_token_usage_report(self, output_path: Path):
        """Save detailed token usage report to JSON file."""
        # Calculate total cached and reasoning tokens
        total_cached = sum(detail.get('cached_tokens', 0) for detail in self.api_call_details)
        total_reasoning = sum(detail.get('reasoning_tokens', 0) for detail in self.api_call_details)
        
        # Calculate tool call statistics. Bookkeeping-only entries (the
        # constraint guard appends one with no timing) must never raise here.
        total_tool_time = sum(detail.get('elapsed_time', 0.0)
                              for detail in self.tool_call_details)
        tool_counts = {}
        for detail in self.tool_call_details:
            tool_name = detail.get('tool_name', 'unknown')
            if tool_name not in tool_counts:
                tool_counts[tool_name] = 0
            tool_counts[tool_name] += 1
        
        # Calculate total runtime
        total_runtime = None
        if self.start_time is not None:
            total_runtime = (self.end_time or time.time()) - self.start_time
        
        # Calculate fmax values
        initial_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
        # Report the shipped artifact's WNS when a finalize branch shipped
        # something older than the tracked best — a stale-mirror copy, or the
        # emergency best_valid copy.  Attempt selection ranks by this number
        # and treats it as a property of the artifact, so an in-memory claim
        # the shipped checkpoint does not hold could outrank a genuinely
        # better attempt.  min() keeps it conservative: never better than
        # tracked, never better than shipped.
        _report_wns = self.best_wns
        _shipped_wns = getattr(self, "_shipped_wns_ns", None)
        if _shipped_wns is not None and _shipped_wns < _report_wns:
            _report_wns = _shipped_wns
        best_fmax = self.calculate_fmax(_report_wns, self.clock_period) if _report_wns > float('-inf') else None
        fmax_improvement = (best_fmax - initial_fmax) if (initial_fmax is not None and best_fmax is not None) else None
        
        report = {
            "model": self.model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_runtime_seconds": total_runtime,
                "total_llm_calls": self.llm_call_count,
                "total_iterations": self.iteration,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
                "total_cached_tokens": total_cached,
                "total_reasoning_tokens": total_reasoning,
                "total_cost": self.total_cost,
                "clock_period_ns": self.clock_period,
                "initial_wns": self.initial_wns,
                "best_wns": self.best_wns,
                "wns_improvement": self.best_wns - self.initial_wns if self.initial_wns is not None else None,
                "initial_fmax_mhz": initial_fmax,
                "best_fmax_mhz": best_fmax,
                "fmax_improvement_mhz": fmax_improvement,
                "total_tool_calls": len(self.tool_call_details),
                "total_tool_time_seconds": total_tool_time,
                "tool_call_counts": tool_counts
            },
            "per_llm_call_details": self.api_call_details,
            "per_tool_call_details": self.tool_call_details
        }
        
        # Atomic write: select_best sources fmax solely from this
        # file; a wall-kill mid-write used to leave truncated JSON, which
        # reads as fmax=None and silently drops a finished valid artifact
        # from ranking. Same temp+replace discipline as every DCP publish.
        _tmp_path = f"{output_path}.tmp{os.getpid()}"
        with open(_tmp_path, 'w') as f:
            json.dump(report, f, indent=2)
        os.replace(_tmp_path, output_path)

        logger.info(f"Token usage report saved to {output_path}")
    
    async def _ensure_output_dcp_written(self, output_dcp: Path) -> None:
        """Ensure an improved result has an output checkpoint without trusting the
        current tool state.

        An existing output checkpoint is left untouched, and a run without WNS
        improvement performs no write. For an improvement, only a fresh
        best-valid disk mirror may be copied to the expected output path. A
        stale or missing mirror causes a no-op so downstream validation can
        select the baseline fallback. The current in-memory checkpoint must
        never be written because it may have regressed after the best state was
        saved.
        """
        try:
            # Single source of truth — initial_wns=None counts as
            # no improvement (unmeasured baseline), matching finalize's
            # Step-2 verdict, so no mirror is copied that Step 2 would
            # immediately overwrite with the baseline.
            if self._finalize_no_improvement():
                return  # no improvement → no DCP per contract
            output_dcp = Path(output_dcp)
            if output_dcp.exists():
                logger.info(
                    f"_ensure_output_dcp_written: output {output_dcp.name} already "
                    "exists; assuming LLM wrote it."
                )
                return

            # SAFETY: only ship the best-valid disk mirror if it's fresh.
            # Never write Vivado in-memory state from here.
            mirror_fresh = (
                self._best_valid_dcp is not None
                and Path(self._best_valid_dcp).exists()
                and Path(self._best_valid_dcp).stat().st_size > 0
                and self._best_valid_dcp_wns is not None
                and abs(self._best_valid_dcp_wns - self.best_wns) < 0.005
            )
            if mirror_fresh:
                try:
                    self._path_guard_check(
                        output_dcp,
                        context="ensure_output_dcp_written_mirror_copy",
                    )
                    _atomic_copy(str(self._best_valid_dcp), str(output_dcp))
                    logger.info(
                        f"_ensure_output_dcp_written: shutil-copied fresh "
                        f"best_valid mirror to {output_dcp.name} "
                        f"(mirror_wns={self._best_valid_dcp_wns:.3f})."
                    )
                except Exception as e:
                    logger.warning(
                        f"_ensure_output_dcp_written: mirror copy failed: {e}"
                    )
                return

            # No fresh mirror — do not write in-memory state. The downstream
            # baseline-fallback step will ship the baseline DCP.
            logger.warning(
                f"_ensure_output_dcp_written: best_wns={self.best_wns:.3f} ns "
                f"improved from initial_wns={self.initial_wns} but no fresh "
                f"best_valid mirror exists "
                f"(_best_valid_dcp={self._best_valid_dcp}, "
                f"_best_valid_dcp_wns={self._best_valid_dcp_wns}). "
                f"REFUSING to dump Vivado in-memory state (would risk shipping "
                f"degraded DCP — root cause of an observed regression class). "
                f"Downstream will ship baseline."
            )
            self.lifecycle_log.append({
                "event": "ensure_output_skipped_no_fresh_mirror",
                "best_wns": self.best_wns,
                "best_valid_dcp_wns": self._best_valid_dcp_wns,
                "best_valid_dcp": str(self._best_valid_dcp) if self._best_valid_dcp else None,
            })
        except Exception as e:
            logger.warning(f"_ensure_output_dcp_written failed: {e}")

    async def _write_readable_edif(self, dcp_path: Path) -> bool:
        """Write a RapidWright-readable EDIF alongside dcp_path.

        Vivado's default write_checkpoint embeds an encrypted EDIF that
        RapidWright cannot read (this is what validate_dcps.py needs).
        Call write_edif on the path with `.edf` extension so RapidWright
        finds it automatically.

        Caller's responsibility: ensure Vivado's in-memory state matches
        dcp_path's contents before invoking (open_checkpoint if unsure).

        Returns True on apparent success, False on any failure.
        """
        edif_path = dcp_path.with_suffix(".edf")
        try:
            await self.call_tool("vivado_write_edif", {
                "edif_path": str(edif_path.resolve()),
                "force": True,
            })
            if edif_path.exists() and edif_path.stat().st_size > 0:
                logger.info(f"Wrote readable EDIF: {edif_path}")
                return True
            logger.warning(f"vivado_write_edif returned but file missing/empty: {edif_path}")
            return False
        except Exception as e:
            logger.warning(f"vivado_write_edif failed for {dcp_path.name}: {e}")
            return False

    async def _structural_validate_dcp(self, dcp_path: Path) -> dict:
        """Lightweight structural validity check on a written DCP.

        Opens the DCP in Vivado, runs report_route_status, and checks that
        report_timing_summary returns a finite WNS.  Returns a dict with:
          {
            "valid": bool,
            "wns": float | None,
            "route_errors": int | None,
            "reason": str,
          }

        This is NOT a replacement for validate_dcps.py functional sim, but it
        catches the failure modes that would make a DCP unsubmittable
        (missing/zero-byte file, unrouted nets, broken netlist).
        """
        if not dcp_path.exists() or dcp_path.stat().st_size == 0:
            return {"valid": False, "wns": None, "route_errors": None,
                    "reason": "DCP missing or zero-byte"}
        # Re-open the DCP to make Vivado's in-memory state match the file.
        try:
            open_res = await self.call_tool("vivado_open_checkpoint", {
                "dcp_path": str(dcp_path.resolve())
            })
            # call_tool returns error
            # envelopes as strings (it does not raise) — an unchecked failed
            # open leaves the prior design in memory and the reports
            # below would happily "validate" it for a corrupt output.
            if _looks_like_tool_error(open_res):
                return {"valid": False, "wns": None, "route_errors": None,
                        "reason": "open_checkpoint error envelope: "
                                  f"{str(open_res)[:120]}"}
        except Exception as e:
            return {"valid": False, "wns": None, "route_errors": None,
                    "reason": f"open_checkpoint raised: {e}"}
        # report_route_status — non-zero error count → invalid.
        try:
            rs = await self.call_tool("vivado_report_route_status", {})
            # Vivado reports "Routing Errors :   0" or similar; parse loosely.
            route_errors = None
            for line in rs.splitlines():
                low = line.lower().strip()
                if "routing errors" in low or "route errors" in low:
                    # find the first integer on the line
                    import re as _re
                    m = _re.search(r"(\d+)", line)
                    if m:
                        route_errors = int(m.group(1))
                        break
            if route_errors is not None and route_errors > 0:
                return {"valid": False, "wns": None, "route_errors": route_errors,
                        "reason": f"{route_errors} routing errors"}
            # A placed-but-unrouted design
            # reports 0 routing errors — check unrouted nets too (2025.1
            # omits the line entirely when fully routed, so absent = OK).
            m_unr = re.search(r"# of unrouted nets.*?:\s+(-?\d+)", rs)
            if m_unr and int(m_unr.group(1)) != 0:
                return {"valid": False, "wns": None, "route_errors": route_errors,
                        "reason": f"{m_unr.group(1)} unrouted nets"}
        except Exception as e:
            logger.warning(f"_structural_validate_dcp: report_route_status failed: {e}")
            # Continue — route_status failure isn't always fatal.
            route_errors = None
        # report_timing_summary — must yield a parseable WNS.
        try:
            ts = await self.call_tool("vivado_report_timing_summary", {})
            ti = parse_timing_summary_static(ts)
            wns = ti.get("wns") if isinstance(ti, dict) else None
            if wns is None:
                return {"valid": False, "wns": None, "route_errors": route_errors,
                        "reason": "could not parse WNS from timing summary"}
            return {"valid": True, "wns": float(wns), "route_errors": route_errors,
                    "reason": "ok"}
        except Exception as e:
            return {"valid": False, "wns": None, "route_errors": route_errors,
                    "reason": f"report_timing_summary raised: {e}"}


    async def _run_qor_capture_subprocess(
        self, dcp_path: Path, json_path: Path, timeout_s: float = 60.0,
    ) -> tuple:
        """Spawn a single Vivado batch to emit the QoR JSON.

        Returns ``(status, error_summary_or_None)`` where ``status`` is
        one of ``"success"``, ``"timeout"``, ``"error"``.  Tests
        monkey-patch this method to avoid invoking Vivado.
        """
        import asyncio
        import subprocess
        # Build Tcl inline so the feature is self-contained.
        tcl = (
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
        run_dir = Path(self.run_dir).resolve()
        # Defence-in-depth: resolve dcp_path / json_path here too in
        # case a caller bypasses _maybe_capture_qor_post_finalize.
        try:
            dcp_path = Path(dcp_path).resolve()
            json_path = Path(json_path).resolve()
        except Exception:
            dcp_path = Path(dcp_path)
            json_path = Path(json_path)
        tcl_path = run_dir / "_qor_capture_probe.tcl"
        tcl_path.write_text(tcl)
        # Platform-specific Vivado invocation.  WSL: cmd.exe → vivado.bat.
        # Linux: vivado on PATH.
        import sys as _sys
        if _sys.platform.startswith("linux") and Path("/mnt/c/WINDOWS/system32/cmd.exe").exists():
            # Translate WSL paths to Windows paths.
            win_dcp = str(dcp_path).replace("/mnt/c/", "C:\\").replace("/mnt/d/", "D:\\").replace("/", "\\")
            win_out = str(json_path).replace("/mnt/c/", "C:\\").replace("/mnt/d/", "D:\\").replace("/", "\\")
            win_tcl = str(tcl_path).replace("/mnt/c/", "C:\\").replace("/mnt/d/", "D:\\").replace("/", "\\")
            # VIVADO_EXEC should name the Windows vivado.bat when running
            # under WSL2 (e.g. D:\Xilinx\2025.1\Vivado\bin\vivado.bat).
            win_vivado = os.environ.get(
                "VIVADO_EXEC", "D:\\Xilinx\\2025.1\\Vivado\\bin\\vivado.bat")
            cmd = [
                "/mnt/c/WINDOWS/system32/cmd.exe", "/c",
                win_vivado,
                "-mode", "batch", "-nojournal", "-nolog",
                "-source", win_tcl, "-tclargs", win_dcp, win_out,
            ]
        else:
            cmd = [
                os.environ.get("VIVADO_EXEC", "vivado"),
                "-mode", "batch", "-nojournal", "-nolog",
                "-source", str(tcl_path), "-tclargs",
                str(dcp_path), str(json_path),
            ]
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
                return ("timeout", f"timed out after {timeout_s} s")
            rc = proc.returncode
            tail = (stdout or b"")[-400:].decode("utf-8", errors="replace")
            if rc == 0 and json_path.exists() and json_path.stat().st_size > 0:
                return ("success", None)
            return ("error", f"rc={rc} tail={tail!r}")
        except FileNotFoundError as exc:
            return ("error", f"vivado not found: {exc}")
        except Exception as exc:
            return ("error", f"subprocess error: {exc!r}")


    def _maybe_write_b3_floor_token(self) -> None:
        """Write the floor-exit sentinel when the adopted deterministic floor
        matches the shipped best.

        The token is written only after adopting the deep-replacement
        small-floor candidate and finishing within 0.004 ns of its floor; this
        tolerance stays below the 0.005 ns boundary for a meaningful
        stochastic-tail gain. The restart wrapper uses the token to stop
        further attempts and skip winner polishing. Writing is best-effort, and
        any failure merely disables the early exit.
        """
        try:
            floor = getattr(self, "_b3_floor_wns", None)
            # abs(): a shipped best worse than the floor must never attest.
            # The promotion invariant makes that state unreachable today, but
            # the writer must not depend on it.  The margin is set just above
            # the largest tail difference observed in archived draws, which
            # keeps behaviour identical while leaving the smallest possible
            # false-attestation window.
            if (floor is None or self.best_wns is None
                    or abs(self.best_wns - floor) > 0.004):
                return
            if not self.run_dir:
                return
            tok = Path(self.run_dir) / "b3_floor_saturated.token"
            tok.write_text(
                f"b3_floor_wns={floor} final_best_wns={self.best_wns} "
                f"margin_ns=0.004 status={self.final_status}\n")
            logger.info(
                f"[b3-floor-exit] token written: shipped best "
                f"{self.best_wns:.3f} is within 0.004 of the B3 floor "
                f"{floor:.3f} — attempt attested deterministic.")
        except Exception as e:
            logger.warning(
                f"[b3-floor-exit] token write failed (non-fatal): {e!r}")


    def _persist_to_strategy_memory(self) -> None:
        """Append a successful run outcome to the strategy-memory JSONL file.

        Persistence is best-effort and never raises into the caller. It is
        skipped when the strategy-memory module is unavailable, the design
        identity cannot be derived, or Fmax did not improve.
        """
        if _RunRecord is None or _append_run is None:
            return
        design = getattr(self, "_design_name_for_memory", None)
        if not design:
            return
        try:
            initial_fmax = getattr(self, "_initial_fmax_for_memory", None)
            if initial_fmax is None:
                initial_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
            best_fmax = (
                self.calculate_fmax(self.best_wns, self.clock_period)
                if self.best_wns > float("-inf") else None
            )
            delta = (
                (best_fmax - initial_fmax)
                if (initial_fmax is not None and best_fmax is not None) else None
            )
            wall = (
                (self.end_time or time.time()) - self.start_time
                if self.start_time is not None else None
            )
            spread = None
            if isinstance(getattr(self, "critical_path_spread_info", None), dict):
                spread = self.critical_path_spread_info.get("avg_distance")
            winning_tools = None
            if _winning_tools_from_call_details is not None and hasattr(self, "tool_call_details"):
                try:
                    winning_tools = _winning_tools_from_call_details(
                        self.tool_call_details, self.initial_wns
                    )
                    if not winning_tools:
                        winning_tools = None
                except Exception:
                    winning_tools = None
            record = _RunRecord(
                design=design,
                candidate=getattr(self, "mode", None),
                delta_fmax_mhz=round(delta, 2) if delta is not None else None,
                initial_fmax_mhz=round(initial_fmax, 2) if initial_fmax is not None else None,
                final_fmax_mhz=round(best_fmax, 2) if best_fmax is not None else None,
                iterations=self.iteration,
                tool_calls=len(self.tool_call_details) if hasattr(self, "tool_call_details") else None,
                force_continues=getattr(self, "force_continue_count", None),
                total_cost_usd=round(self.total_cost, 4) if self.total_cost else None,
                wall_time_s=round(wall, 1) if wall is not None else None,
                completed=delta is not None and delta > 0,
                lut_count=getattr(self, "lut_count", None),
                critical_path_spread=round(spread, 1) if spread is not None else None,
                winning_tools=winning_tools,
            )
            written = _append_run(record)
            if written:
                logger.info(f"strategy_memory: appended run for '{design}' to {written}")
        except Exception as e:
            logger.warning(f"strategy_memory persistence failed: {e}")


    


class FPGAOptimizerTest(DCPOptimizerBase):
    """
    Test mode for FPGA Design Optimization - hardcodes all tool calls to
    diagnose issues.

    This class runs a deterministic optimization flow without using any LLM,
    making it easier to identify where MCP servers or Vivado might hang.
    """
    
    def __init__(self, debug: bool = False, run_dir: Optional[Path] = None):
        super().__init__(debug=debug, run_dir=run_dir)
        self.final_wns = None
    
    async def start_servers(self):
        """Start and connect to both MCP servers."""
        await super().start_servers(log_prefix="[TEST]")
    
    async def call_vivado_tool(self, tool_name: str, arguments: dict, timeout: float = 300.0) -> str:
        """Execute a Vivado tool call with timing and logging."""
        logger.info(f"[VIVADO] Calling {tool_name} with args: {json.dumps(arguments)[:200]}...")
        print(f"[TEST] Calling vivado_{tool_name}...")
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                self.vivado_session.call_tool(tool_name, arguments),
                timeout=timeout
            )
            
            elapsed = time.time() - start_time
            logger.info(f"[VIVADO] {tool_name} completed in {elapsed:.2f}s")
            print(f"[TEST] vivado_{tool_name} completed in {elapsed:.2f}s")
            
            # Extract text content from result
            if result.content:
                text_parts = [c.text for c in result.content if hasattr(c, 'text')]
                return "\n".join(text_parts)
            return "(no output)"
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"[VIVADO] {tool_name} TIMED OUT after {elapsed:.2f}s")
            print(f"[TEST] ERROR: vivado_{tool_name} TIMED OUT after {elapsed:.2f}s")
            raise
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VIVADO] {tool_name} FAILED after {elapsed:.2f}s: {e}")
            print(f"[TEST] ERROR: vivado_{tool_name} failed after {elapsed:.2f}s: {e}")
            raise
    
    async def call_rapidwright_tool(self, tool_name: str, arguments: dict, timeout: float = 300.0) -> str:
        """Execute a RapidWright tool call with timing and logging."""
        logger.info(f"[RAPIDWRIGHT] Calling {tool_name} with args: {json.dumps(arguments)[:200]}...")
        print(f"[TEST] Calling rapidwright_{tool_name}...")
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                self.rapidwright_session.call_tool(tool_name, arguments),
                timeout=timeout
            )
            
            elapsed = time.time() - start_time
            logger.info(f"[RAPIDWRIGHT] {tool_name} completed in {elapsed:.2f}s")
            print(f"[TEST] rapidwright_{tool_name} completed in {elapsed:.2f}s")
            
            # Extract text content from result
            if result.content:
                text_parts = [c.text for c in result.content if hasattr(c, 'text')]
                return "\n".join(text_parts)
            return "(no output)"
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"[RAPIDWRIGHT] {tool_name} TIMED OUT after {elapsed:.2f}s")
            print(f"[TEST] ERROR: rapidwright_{tool_name} TIMED OUT after {elapsed:.2f}s")
            raise
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[RAPIDWRIGHT] {tool_name} FAILED after {elapsed:.2f}s: {e}")
            print(f"[TEST] ERROR: rapidwright_{tool_name} failed after {elapsed:.2f}s: {e}")
            raise
    
    def parse_wns_from_timing_report(self, timing_report: str) -> Optional[float]:
        """Extract WNS from timing report using shared parsing logic."""
        return parse_timing_summary_static(timing_report)["wns"]
    
    async def _call_vivado_for_clock(self, tool_name: str, arguments: dict) -> str:
        """Helper to call Vivado tools for clock period query."""
        return await self.call_vivado_tool(tool_name, arguments, timeout=60.0)
    
    async def fetch_clock_period(self) -> Optional[float]:
        """Query clock period with test-mode logging."""
        period = await super().get_clock_period(self._call_vivado_for_clock)
        if period is not None:
            clock_info = f" (target clock: {self.target_clock})" if self.target_clock else ""
            print(f"[TEST] Clock period: {period:.3f} ns{clock_info}")
        else:
            print("[TEST] WARNING: Could not parse clock period from Vivado")
        return period
    
    async def run_test(self, input_dcp: Path, output_dcp: Path, max_nets_to_optimize: int = 5) -> bool:
        """
        Run the deterministic test optimization flow.
        
        Steps:
        1. Open the input DCP in Vivado
        2. Report timing in Vivado
        3. Get the critical high fan out nets from Vivado
        4. Open the DCP in RapidWright
        5. Apply the fanout optimization for each high fanout net
        6. Write a DCP out from RapidWright
        7. Read the RapidWright generated DCP into Vivado
        8. Route the design in Vivado
        9. Report timing and compare WNS
        """
        print("\n" + "="*70)
        print("FPGA OPTIMIZER TEST MODE")
        print("="*70)
        print(f"Input DCP:  {input_dcp}")
        print(f"Output DCP: {output_dcp}")
        print(f"Temp dir:   {self.temp_dir}")
        print(f"Max nets to optimize: {max_nets_to_optimize}")
        print("="*70 + "\n")
        
        overall_start = time.time()
        
        try:
            # Step 0: Initialize RapidWright (Vivado starts automatically)
            print("\n" + "-"*60)
            print("STEP 0: Initialize RapidWright")
            print("-"*60)
            
            # Initialize RapidWright (Vivado will auto-start when first used)
            result = await self.call_rapidwright_tool("initialize_rapidwright", {
                "jvm_max_memory": "8G"
            }, timeout=120.0)
            print(f"RapidWright init result:\n{result[:500]}...")
            logger.info(f"RapidWright init result: {result}")
            
            # Step 1: Open the input DCP in Vivado
            print("\n" + "-"*60)
            print("STEP 1: Open input DCP in Vivado")
            print("-"*60)
            
            result = await self.call_vivado_tool("open_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            print(f"Open checkpoint result:\n{result}")
            logger.info(f"Open checkpoint result: {result}")
            
            # Step 2: Report timing in Vivado
            print("\n" + "-"*60)
            print("STEP 2: Report timing in Vivado")
            print("-"*60)
            
            result = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
            print(f"Timing summary (first 2000 chars):\n{result[:2000]}...")
            logger.info(f"Initial timing summary: {result}")
            
            # Get clock period for fmax calculation (also detects target clock)
            self.clock_period = await self.fetch_clock_period()
            
            # Get WNS for the target clock domain
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.initial_wns = target_wns
            else:
                self.initial_wns = self.parse_wns_from_timing_report(result)
            
            self.print_fmax_status("Initial", self.initial_wns)
            logger.info(f"Initial WNS: {self.initial_wns} ns")
            print()
            
            # Step 3: Get critical high fanout nets
            print("\n" + "-"*60)
            print("STEP 3: Get critical high fanout nets")
            print("-"*60)
            
            result = await self.call_vivado_tool("get_critical_high_fanout_nets", {
                "num_paths": 50,
                "min_fanout": 100,
                "exclude_clocks": True
            }, timeout=600.0)
            print(f"High fanout nets report:\n{result}")
            logger.info(f"High fanout nets: {result}")
            
            # Parse the nets
            self.high_fanout_nets = self.parse_high_fanout_nets(result)
            print(f"\nParsed {len(self.high_fanout_nets)} high fanout nets")
            
            if not self.high_fanout_nets:
                print("WARNING: No high fanout nets found to optimize!")
                logger.warning("No high fanout nets found to optimize")
            
            # Select top nets to optimize
            nets_to_optimize = self.high_fanout_nets[:max_nets_to_optimize]
            print(f"Will optimize {len(nets_to_optimize)} nets:")
            for net_name, fanout, path_count in nets_to_optimize:
                print(f"  - {net_name} (fanout={fanout}, paths={path_count})")
            
            # Step 4: Open the DCP in RapidWright
            print("\n" + "-"*60)
            print("STEP 4: Open DCP in RapidWright")
            print("-"*60)
            
            result = await self.call_rapidwright_tool("read_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            print(f"RapidWright read checkpoint result:\n{result}")
            logger.info(f"RapidWright read checkpoint: {result}")
            
            # Step 5: Apply fanout optimization for each high fanout net
            print("\n" + "-"*60)
            print("STEP 5: Apply fanout optimizations in RapidWright")
            print("-"*60)
            
            successful_optimizations = 0
            for i, (net_name, fanout, path_count) in enumerate(nets_to_optimize):
                print(f"\n[{i+1}/{len(nets_to_optimize)}] Optimizing net: {net_name}")
                print(f"    Fanout: {fanout}, Critical paths: {path_count}")
                
                # Calculate split factor: fanout/100, min 2, max 8
                split_factor = max(2, min(8, fanout // 100))
                print(f"    Split factor: {split_factor}")
                
                try:
                    result = await self.call_rapidwright_tool("optimize_fanout", {
                        "net_name": net_name,
                        "split_factor": split_factor
                    }, timeout=300.0)
                    print(f"    Result: {result[:500]}...")
                    logger.info(f"Optimize fanout {net_name}: {result}")
                    
                    # Check if successful
                    if "error" not in result.lower() or "success" in result.lower():
                        successful_optimizations += 1
                except Exception as e:
                    print(f"    FAILED: {e}")
                    logger.error(f"Failed to optimize {net_name}: {e}")
            
            print(f"\nSuccessfully optimized {successful_optimizations}/{len(nets_to_optimize)} nets")
            
            # Step 6: Write DCP from RapidWright
            print("\n" + "-"*60)
            print("STEP 6: Write DCP from RapidWright")
            print("-"*60)
            
            rapidwright_dcp = Path(self.temp_dir) / "rapidwright_optimized.dcp"
            result = await self.call_rapidwright_tool("write_checkpoint", {
                "dcp_path": str(rapidwright_dcp),
                "overwrite": True
            }, timeout=600.0)
            print(f"Write checkpoint result:\n{result}")
            logger.info(f"RapidWright write checkpoint: {result}")
            
            # Check if the file was created
            if rapidwright_dcp.exists():
                print(f"DCP file created: {rapidwright_dcp} ({rapidwright_dcp.stat().st_size} bytes)")
            else:
                print("WARNING: DCP file was not created!")
                logger.warning("RapidWright DCP file not created")
            
            # Step 7: Read RapidWright DCP into Vivado
            print("\n" + "-"*60)
            print("STEP 7: Read RapidWright DCP into Vivado")
            print("-"*60)
            
            # Note: Opening a RapidWright-generated DCP takes MUCH longer than
            # opening the original DCP because:
            # 1. Vivado must reload encrypted IP blocks from disk
            # 2. Vivado must reconstruct internal data structures
            # For large designs, this can take 10-30 minutes
            RAPIDWRIGHT_DCP_TIMEOUT = 300.0  # 5 minutes
            
            # A Tcl script may need sourcing first (for encrypted IP)
            tcl_script = rapidwright_dcp.with_suffix('.tcl')
            if tcl_script.exists():
                print(f"Found Tcl script for encrypted IP: {tcl_script}")
                print(f"Note: This may take 10-30 minutes for large designs...")
                # Source the Tcl script instead of directly opening the DCP
                result = await self.call_vivado_tool("run_tcl", {
                    "command": f"source {{{tcl_script}}}"
                }, timeout=RAPIDWRIGHT_DCP_TIMEOUT)
                print(f"Source Tcl script result:\n{result}")
            else:
                # Opening a RapidWright-generated DCP can take longer than original
                # because Vivado needs to reconstruct some internal data structures
                result = await self.call_vivado_tool("open_checkpoint", {
                    "dcp_path": str(rapidwright_dcp)
                }, timeout=RAPIDWRIGHT_DCP_TIMEOUT)
                print(f"Open RapidWright DCP result:\n{result}")
            logger.info(f"Open RapidWright DCP: {result}")
            
            # Step 8: Route the design in Vivado
            print("\n" + "-"*60)
            print("STEP 8: Route design in Vivado")
            print("-"*60)
            
            # First check route status
            result = await self.call_vivado_tool("report_route_status", {
                "show_unrouted": True,
                "show_errors": True,
                "max_nets": 20
            }, timeout=300.0)
            print(f"Route status before routing:\n{result[:1500]}...")
            logger.info(f"Route status before routing: {result}")
            
            # Route the design
            result = await self.call_vivado_tool("route_design", {
                "directive": "Default",
            }, timeout=600.0)  # 2 hour timeout for routing
            print(f"Route design result:\n{result}")
            logger.info(f"Route design: {result}")
            
            # Check route status again
            result = await self.call_vivado_tool("report_route_status", {
                "show_unrouted": True,
                "show_errors": True,
                "max_nets": 20
            }, timeout=300.0)
            print(f"Route status after routing:\n{result[:1500]}...")
            logger.info(f"Route status after routing: {result}")
            
            # Step 9: Report final timing
            print("\n" + "-"*60)
            print("STEP 9: Report final timing")
            print("-"*60)
            
            result = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
            print(f"Final timing summary (first 2000 chars):\n{result[:2000]}...")
            logger.info(f"Final timing summary: {result}")
            
            # Get final WNS for the target clock domain
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.final_wns = target_wns
            else:
                self.final_wns = self.parse_wns_from_timing_report(result)
            
            self.print_fmax_status("Final", self.final_wns)
            logger.info(f"Final WNS: {self.final_wns} ns")
            print()
            
            # Write final DCP and report results
            self.print_wns_change(self.initial_wns, self.final_wns, self.clock_period)
            
            # Always write the final checkpoint (regardless of improvement)
            print(f"\nWriting final DCP to: {output_dcp}")
            result = await self.call_vivado_tool("write_checkpoint", {
                "dcp_path": str(output_dcp.resolve()),
                "force": True
            }, timeout=600.0)
            print(f"Write final DCP result:\n{result}")
            
            # ================================================================
            # Summary
            # ================================================================
            elapsed = time.time() - overall_start
            self.print_test_summary(
                title="TEST SUMMARY",
                elapsed_seconds=elapsed,
                initial_wns=self.initial_wns,
                final_wns=self.final_wns,
                clock_period=self.clock_period,
                extra_info=f"Nets optimized: {successful_optimizations}/{len(nets_to_optimize)}"
            )
            
            return True
            
        except Exception as e:
            logger.exception(f"Test failed with exception: {e}")
            print(f"\n*** TEST FAILED ***")
            print(f"Exception: {type(e).__name__}: {e}")
            return False
    
    async def run_test_logicnets(self, input_dcp: Path, output_dcp: Path) -> bool:
        """Run a fixed-pblock placement-and-routing optimization flow.

        The flow opens the input DCP, measures baseline WNS, extracts
        critical-path cells, and analyzes their placement spread. It then
        unplaces the design, constrains all cells to
        `SLICE_X55Y60:SLICE_X111Y254`, places and routes the result, and
        reports the resulting WNS. The fixed region is pinned to the target
        device floorplan.
        """
        pblock_ranges = "SLICE_X55Y60:SLICE_X111Y254"
        
        print("\n" + "="*70)
        print("FPGA OPTIMIZER TEST MODE - LOGICNETS PBLOCK FLOW")
        print("="*70)
        print(f"Input DCP:  {input_dcp}")
        print(f"Output DCP: {output_dcp}")
        print(f"Temp dir:   {self.temp_dir}")
        print("="*70 + "\n")
        
        overall_start = time.time()
        
        try:
            # Step 0: Initialize RapidWright (Vivado starts automatically)
            print("\n" + "-"*60)
            print("STEP 0: Initialize RapidWright")
            print("-"*60)
            
            result = await self.call_rapidwright_tool("initialize_rapidwright", {
                "jvm_max_memory": "8G"
            }, timeout=120.0)
            print(f"RapidWright init result:\n{result[:500]}...")
            logger.info(f"RapidWright init result: {result}")
            
            # Step 1: Open the input DCP in Vivado
            print("\n" + "-"*60)
            print("STEP 1: Open input DCP in Vivado")
            print("-"*60)
            
            result = await self.call_vivado_tool("open_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            print(f"Open checkpoint result:\n{result}")
            logger.info(f"Open checkpoint result: {result}")
            
            # Step 2: Report timing in Vivado (Initialize WNS)
            print("\n" + "-"*60)
            print("STEP 2: Report initial timing in Vivado")
            print("-"*60)
            
            result = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
            print(f"Timing summary (first 2000 chars):\n{result[:2000]}...")
            logger.info(f"Initial timing summary: {result}")
            
            # Get clock period for fmax calculation (also detects target clock)
            self.clock_period = await self.fetch_clock_period()
            
            # Get WNS for the target clock domain
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.initial_wns = target_wns
            else:
                self.initial_wns = self.parse_wns_from_timing_report(result)
            
            self.print_fmax_status("Initial", self.initial_wns)
            logger.info(f"Initial WNS: {self.initial_wns} ns")
            print()
            
            # Step 3: Extract critical path cells from Vivado
            print("\n" + "-"*60)
            print("STEP 3: Extract critical path cells")
            print("-"*60)
            
            # Write to a file for efficient data transfer
            critical_paths_file = Path(self.temp_dir) / "critical_paths.json"
            result = await self.call_vivado_tool("extract_critical_path_cells", {
                "num_paths": 50,
                "output_file": str(critical_paths_file)
            }, timeout=600.0)
            print(f"Extract critical paths result:\n{result[:2000]}...")
            logger.info(f"Extract critical paths: {result}")
            
            # Step 4: Open DCP in RapidWright and analyze critical path spread
            print("\n" + "-"*60)
            print("STEP 4: Analyze critical path spread in RapidWright")
            print("-"*60)
            
            # First, open the DCP in RapidWright
            result = await self.call_rapidwright_tool("read_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            print(f"RapidWright read checkpoint result:\n{result}")
            logger.info(f"RapidWright read checkpoint: {result}")
            
            # Analyze critical path spread
            result = await self.call_rapidwright_tool("analyze_critical_path_spread", {
                "input_file": str(critical_paths_file)
            }, timeout=300.0)
            print(f"Critical path spread analysis:\n{result[:3000] if isinstance(result, str) else str(result)[:3000]}...")
            logger.info(f"Critical path spread: {result}")
            
            # Parse the spread analysis result to check if pblock is recommended
            spread_result = result if isinstance(result, str) else str(result)
            pblock_recommended = "spread-out" in spread_result.lower() or "pblock" in spread_result.lower()
            print(f"\n*** Pblock optimization {'RECOMMENDED' if pblock_recommended else 'may not be needed'} ***")
            
            print("\n" + "-"*60)
            print("STEP 5: Apply pblock for LogicNets")
            print("-"*60)
            
            print(f"Using pblock range: {pblock_ranges}")
            
            # Step 6: Unplace the design in Vivado
            print("\n" + "-"*60)
            print("STEP 6: Unplace the design in Vivado")
            print("-"*60)
            
            # Use place_design -unplace to remove all placement
            result = await self.call_vivado_tool("run_tcl", {
                "command": "place_design -unplace"
            }, timeout=300.0)
            print(f"Unplace result:\n{result}")
            logger.info(f"Unplace result: {result}")
            
            # Step 7: Create and apply pblock to entire design
            print("\n" + "-"*60)
            print("STEP 7: Create and apply pblock to entire design")
            print("-"*60)
            
            result = await self.call_vivado_tool("create_and_apply_pblock", {
                "pblock_name": "pblock_opt",
                "ranges": pblock_ranges,
                "apply_to": "current_design",  # Apply to entire design
                "is_soft": False  # Hard constraint
            }, timeout=300.0)
            print(f"Create and apply pblock result:\n{result}")
            logger.info(f"Create pblock result: {result}")
            
            # Step 8: Place the design in Vivado
            print("\n" + "-"*60)
            print("STEP 8: Place the design in Vivado")
            print("-"*60)
            
            result = await self.call_vivado_tool("place_design", {
                "directive": "Default"
            }, timeout=3600.0)  # 1 hour timeout for placement
            print(f"Place design result:\n{result}")
            logger.info(f"Place design: {result}")
            
            # Step 9: Route the design in Vivado
            print("\n" + "-"*60)
            print("STEP 9: Route the design in Vivado")
            print("-"*60)
            
            result = await self.call_vivado_tool("route_design", {
                "directive": "Default"
            }, timeout=3600.0)  # 1 hour timeout for routing
            print(f"Route design result:\n{result}")
            logger.info(f"Route design: {result}")
            
            # Check route status
            result = await self.call_vivado_tool("report_route_status", {}, timeout=300.0)
            print(f"Route status after routing:\n{result[:1500]}...")
            logger.info(f"Route status after routing: {result}")
            
            # Step 10: Report timing and compare WNS
            print("\n" + "-"*60)
            print("STEP 10: Report final timing")
            print("-"*60)
            
            result = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
            print(f"Final timing summary (first 2000 chars):\n{result[:2000]}...")
            logger.info(f"Final timing summary: {result}")
            
            # Get final WNS for the target clock domain
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.final_wns = target_wns
            else:
                self.final_wns = self.parse_wns_from_timing_report(result)
            
            self.print_fmax_status("Final", self.final_wns)
            logger.info(f"Final WNS: {self.final_wns} ns")
            print()
            
            # Write final DCP and report results
            self.print_wns_change(self.initial_wns, self.final_wns, self.clock_period)
            
            # Always write the final checkpoint
            print(f"\nWriting final DCP to: {output_dcp}")
            result = await self.call_vivado_tool("write_checkpoint", {
                "dcp_path": str(output_dcp.resolve()),
                "force": True
            }, timeout=600.0)
            print(f"Write final DCP result:\n{result}")
            
            # ================================================================
            # Summary
            # ================================================================
            elapsed = time.time() - overall_start
            self.print_test_summary(
                title="TEST SUMMARY - LOGICNETS PBLOCK OPTIMIZATION",
                elapsed_seconds=elapsed,
                initial_wns=self.initial_wns,
                final_wns=self.final_wns,
                clock_period=self.clock_period,
                extra_info=f"Pblock applied: {pblock_ranges}"
            )
            
            return True
            
        except Exception as e:
            logger.exception(f"LogicNets test failed with exception: {e}")
            print(f"\n*** TEST FAILED ***")
            print(f"Exception: {type(e).__name__}: {e}")
            return False

    async def run_test_vexriscv(self, input_dcp: Path, output_dcp: Path) -> bool:
        """Run a critical-path-guided cell re-placement optimization flow.

        The flow measures the baseline and extracts critical-path pins,
        analyzes net detours, filters placement candidates, and applies
        cell-placement changes. It writes the modified DCP, reopens and routes
        it, then measures the resulting Fmax for verification.
        """
        overall_start = time.time()
        
        try:
            # Step 1: Vivado baseline
            print("=" * 60)
            print("Step 1  Vivado baseline")
            print("=" * 60)
            
            result = await self.call_vivado_tool("open_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            logger.info(f"Open checkpoint result: {result}")
            
            self.clock_period = await self.fetch_clock_period()
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.initial_wns = target_wns
            else:
                ts = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
                self.initial_wns = self.parse_wns_from_timing_report(ts)
            
            baseline_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
            print(f"  Clock period:   {self.clock_period} ns")
            print(f"  Baseline WNS:   {self.initial_wns} ns")
            if baseline_fmax is not None:
                print(f"  Baseline Fmax:  {baseline_fmax:.2f} MHz")
            
            pins_file = Path(self.temp_dir) / "critical_path_pins.json"
            result = await self.call_vivado_tool("extract_critical_path_pins", {
                "num_paths": 10,
                "output_file": str(pins_file)
            }, timeout=600.0)
            
            critical_paths = json.loads(Path(pins_file).read_text()) if pins_file.exists() else json.loads(result)
            print(f"  Extracted {len(critical_paths)} critical path pin lists")
            
            # Step 2: RapidWright analysis
            print("\n" + "=" * 60)
            print("Step 2  RapidWright analysis")
            print("=" * 60)
            
            result = await self.call_rapidwright_tool("initialize_rapidwright", {
                "jvm_max_memory": "8G"
            }, timeout=120.0)
            logger.info(f"RapidWright init: {result}")
            
            result = await self.call_rapidwright_tool("read_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            logger.info(f"RapidWright read checkpoint: {result}")
            
            result = await self.call_rapidwright_tool("analyze_net_detour", {
                "input_file": str(pins_file),
                "detour_threshold": 2.0
            }, timeout=300.0)
            logger.info(f"analyze_net_detour: {result}")
            
            analysis = json.loads(result) if isinstance(result, str) else result
            if "error" in analysis:
                raise RuntimeError(f"analyze_net_detour failed: {analysis['error']}")
            candidates = analysis.get("candidates", [])
            print(f"  Cells analyzed: {analysis.get('cells_analyzed', '?')}")
            print(f"  Candidates (detour > 2.0): {len(candidates)}")
            for c in candidates[:5]:
                print(f"    {str(c['cell']):55s}  ratio={c['max_detour_ratio']}")
            
            if not candidates:
                print("\n  No candidates found — nothing to optimize")
                self.final_wns = self.initial_wns
                return True
            
            worst_path_cells = list(set(
                str(c["cell"]) for c in candidates if c.get("path", 0) <= 2
            ))
            if not worst_path_cells:
                worst_path_cells = [str(candidates[0]["cell"])]
            
            print(f"\n  Targeting {len(worst_path_cells)} cells on paths 1-2:")
            for name in worst_path_cells:
                print(f"    {name}")
            
            # Step 3: RapidWright optimization
            print("\n" + "=" * 60)
            print("Step 3  RapidWright optimization")
            print("=" * 60)
            
            result = await self.call_rapidwright_tool("optimize_cell_placement", {
                "cell_names": worst_path_cells
            }, timeout=300.0)
            logger.info(f"optimize_cell_placement: {result}")
            
            opt_result = json.loads(result) if isinstance(result, str) else result
            for r in opt_result.get("results", []):
                print(f"  {r['cell']}: {r['status']} — {r['message']}")
            
            rw_output = Path(self.temp_dir) / "vexriscv_rw_optimized.dcp"
            result = await self.call_rapidwright_tool("write_checkpoint", {
                "dcp_path": str(rw_output)
            }, timeout=600.0)
            print(f"  Wrote {rw_output.name}")
            
            # Step 4: Vivado verification
            print("\n" + "=" * 60)
            print("Step 4  Vivado verification")
            print("=" * 60)
            
            result = await self.call_vivado_tool("open_checkpoint", {
                "dcp_path": str(rw_output)
            }, timeout=600.0)
            logger.info(f"Open optimized checkpoint: {result}")
            
            result = await self.call_vivado_tool("route_design", {
                "directive": "Default"
            }, timeout=3600.0)
            logger.info(f"Route design: {result}")
            
            route_result = await self.call_vivado_tool("report_route_status", {}, timeout=300.0)
            error_match = re.search(r"# of nets with routing errors.*?:\s+(\d+)", route_result)
            error_count = int(error_match.group(1)) if error_match else -1
            
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.final_wns = target_wns
            else:
                ts = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
                self.final_wns = self.parse_wns_from_timing_report(ts)
            
            new_fmax = self.calculate_fmax(self.final_wns, self.clock_period)
            
            print(f"  Routing errors:  {error_count}")
            if baseline_fmax is not None and new_fmax is not None:
                print(f"  Baseline WNS:    {self.initial_wns} ns  →  Fmax {baseline_fmax:.2f} MHz")
                print(f"  Optimized WNS:   {self.final_wns} ns  →  Fmax {new_fmax:.2f} MHz")
                delta = new_fmax - baseline_fmax
                print(f"  Fmax improvement: {delta:+.2f} MHz")
            else:
                print(f"  Baseline WNS:  {self.initial_wns} ns")
                print(f"  Optimized WNS: {self.final_wns} ns")
            
            # Write final DCP
            print(f"\nWriting final DCP to: {output_dcp}")
            result = await self.call_vivado_tool("write_checkpoint", {
                "dcp_path": str(output_dcp.resolve()),
                "force": True
            }, timeout=600.0)
            
            # Summary
            elapsed = time.time() - overall_start
            cells_info = ", ".join(worst_path_cells)
            self.print_test_summary(
                title="TEST SUMMARY - VEXRISCV CELL RE-PLACEMENT",
                elapsed_seconds=elapsed,
                initial_wns=self.initial_wns,
                final_wns=self.final_wns,
                clock_period=self.clock_period,
                extra_info=f"Cells re-placed: {cells_info}"
            )
            
            return True
            
        except Exception as e:
            logger.exception(f"VexRiscv test failed with exception: {e}")
            print(f"\n*** TEST FAILED ***")
            print(f"Exception: {type(e).__name__}: {e}")
            return False

    async def cleanup(self):
        """Clean up resources."""
        print("\n[TEST] Cleaning up...")
        await super().cleanup()
        print(f"[TEST] Run directory preserved at: {self.run_dir}")


async def run_test_mode(input_dcp: Path, output_dcp: Path, debug: bool = False, max_nets: int = 5, run_dir: Optional[Path] = None):
    """Run the optimization flow associated with the selected example checkpoint.

    Detects the checkpoint type and applies either constrained-region placement
    optimization or cell re-placement.
    """
    # Detect which DCP is being used based on filename
    dcp_name = input_dcp.name.lower()
    
    if "logicnets" in dcp_name:
        design_type = "logicnets"
        print(f"[TEST] Detected LogicNets design - using pblock optimization flow")
    elif "vexriscv" in dcp_name:
        design_type = "vexriscv"
        print(f"[TEST] Detected VexRiscv design - using cell re-placement flow")
    else:
        print(f"\n[TEST] ERROR: Unsupported DCP file: {input_dcp.name}")
        print(f"[TEST] Test mode supports these benchmark DCPs:")
        print(f"[TEST]   - fpl26_contest_benchmarks/logicnets_jscl_2025.1.dcp")
        print(f"[TEST]   - fpl26_contest_benchmarks/vexriscv_re-place_2025.1.dcp")
        print(f"[TEST]")
        print(f"[TEST] For custom DCPs, run without --test to use the LLM-guided optimizer.")
        return 1
    
    tester = FPGAOptimizerTest(debug=debug, run_dir=run_dir)
    
    try:
        await tester.start_servers()
        
        if design_type == "logicnets":
            success = await tester.run_test_logicnets(input_dcp, output_dcp)
        else:
            success = await tester.run_test_vexriscv(input_dcp, output_dcp)
        
        if success:
            print("\n[TEST] Test completed successfully")
            print(f"\n[TEST] Output files:")
            print(f"[TEST]   Optimized DCP: {output_dcp}")
            print(f"[TEST]   Run directory: {tester.run_dir}")
            return 0
        else:
            print("\n[TEST] Test failed")
            print(f"[TEST] Run directory: {tester.run_dir}")
            return 1
            
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")
        print(f"[TEST] Run directory: {tester.run_dir}")
        return 130
    except Exception as e:
        logger.exception(f"Test mode fatal error: {e}")
        print(f"\n[TEST] Fatal error: {e}")
        print(f"[TEST] Run directory: {tester.run_dir}")
        return 1
    finally:
        await tester.cleanup()


async def main():
    parser = argparse.ArgumentParser(
        description="FPGA Design Optimization Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dcp_optimizer.py input.dcp
  python dcp_optimizer.py input.dcp --output output.dcp
  python dcp_optimizer.py input.dcp --model x-ai/grok-4.3
  python dcp_optimizer.py input.dcp --debug
  python dcp_optimizer.py fpl26_contest_benchmarks/logicnets_jscl_2025.1.dcp --test
  python dcp_optimizer.py fpl26_contest_benchmarks/vexriscv_re-place_2025.1.dcp --test
        """
    )
    parser.add_argument("input_dcp", type=Path, help="Input design checkpoint (.dcp)")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        dest="output_dcp",
        help="Output optimized checkpoint (.dcp). Default: <input_name>_optimized-<timestamp>.dcp in same directory as input"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="OpenRouter API key (default: OPENROUTER_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (verbose logging, save intermediate checkpoints)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: run without LLM. Pblock for LogicNets, cell re-placement for VexRiscv (see docs/optimization_example.md)."
    )
    parser.add_argument(
        "--max-nets",
        type=int,
        default=5,
        help="Maximum number of high fanout nets to optimize in test mode (default: 5)"
    )
    parser.add_argument(
        "--mode",
        choices=("v0_3", "anchor"),
        default="v0_3",
        help="Controller mode.  v0_3 (default): slope-aware ceiling lift + "
             "first-improvement-gated unconditional cap (this branch's behaviour). "
             "anchor: emulate upstream — stop iff LLM signals done; no force-continue, "
             "no cap.  Used by the scheduler to run a true anchor candidate.",
    )
    parser.add_argument(
        "--no-rag-seed",
        action="store_true",
        help="Disable the iter-1 strategy-memory injection.  Used for A/B "
             "testing the RAG seed's contribution to ΔFmax.  When set, the "
             "optimizer still PERSISTS its run outcome at the end (so the "
             "memory keeps growing) but does not READ from memory at iter 1.",
    )
    parser.add_argument(
        "--contest-mode",
        action="store_true",
        help="Hidden-design hygiene.  Strategy-memory retrieval is always "
             "fingerprint-only (LUT count + critical-path spread, global "
             "aggregate fallback); this flag additionally appends the "
             "negative-memory advisory block to the iter-1 prompt.  Use it "
             "when running on benchmarks the LLM has never seen.",
    )
    parser.add_argument(
        "--ils-polish",
        action="store_true",
        help="Enable ILS-as-polish (ruin-and-recreate): when the LLM loop "
             "STALLS on a size-viable design with budget remaining, switch the "
             "rest of the budget to full unplace -> re-place (Explore/"
             "ExtraTimingOpt) -> route -> phys_opt, keeping a result only if "
             "strictly better (never-worse). Default OFF.",
    )
    parser.add_argument("--ils-stagnation-seconds", type=float, default=None,
                        help="ILS: wall-seconds w/o WNS improvement before preempt "
                             "(default 600). Lower = fire sooner.")
    parser.add_argument("--ils-min-remaining-s", type=float, default=None,
                        help="ILS: min budget (s) remaining required to preempt "
                             "(default 1100; ~2 cycles).")
    parser.add_argument("--ils-max-cells", type=int, default=None,
                        help="ILS: max design cells to attempt (default 300000).")
    parser.add_argument("--ils-futility-stop", type=int, default=None,
                        help="ILS: final-seed futility stop — exit ILS after this "
                             "many CONSECUTIVE non-accepting cycles instead of "
                             "burning to the wall (default 2; replay evidence: saves "
                             "400-1900s of -0.1*alpha*gamma penalty with zero "
                             "forfeited accepts). 0 = legacy run-to-budget.")
    parser.add_argument("--no-partial-ruin-spread-gate", action="store_true",
                        help="ILS: DISABLE the spread gate that skips "
                             "PARTIAL_RUIN combos when Phase 1 measured "
                             "critical-path avg spread below 30 tiles "
                             "(whole-history mining, held-out: "
                             "spread~0 cell surgery 26/26 negative, mean "
                             "-1.10ns). Env fallback: "
                             "FPL26_PARTIAL_RUIN_SPREAD_GATE=0 also "
                             "disables. Default: gate ON; spread "
                             "unmeasured -> gate inert.")
    parser.add_argument("--no-route-reroll", action="store_true",
                        help="ILS: DISABLE the __ROUTE_REROLL__ combo "
                             "(near-met route lottery re-roll: full "
                             "route_design -unroute + AggressiveExplore "
                             "from-scratch solve; plateau-probe evidence: "
                             "+0.070ns/+9.5MHz hold-improving). Env "
                             "fallback: FPL26_NO_ROUTE_REROLL=1 also "
                             "disables. Default: combo ON, picker-gated to "
                             "near-met (wns >= -1.5) + full-route budget "
                             "fit.")
    parser.add_argument("--deep-replace-first", action="store_true",
                        help="Run the DEEP-extreme forced recipe BEFORE the "
                             "LLM loop instead of at the exit tail. Requires "
                             "--deep-replace. The recipe needs ~79%% of the "
                             "wall (boom_soc: 2750s of 3500s), so as a tail "
                             "stage it can never fire -- chain8 measured 72s "
                             "left at exit. Measured on 8-vCPU eval "
                             "parity: boom_soc agent +16.25 vs forced +41.43; "
                             "boom_soc_v2 best-ever +6.41 vs forced +14.06. "
                             "Never-worse on fmax (a losing recipe ships the "
                             "baseline via the finalize MUX); the cost is the "
                             "LLM loop's opportunity. Env fallback: "
                             "FPL26_DEEP_REPLACE_FIRST=1; belt kill: "
                             "FPL26_NO_DEEP_REPLACE_FIRST=1. Default OFF.")
    parser.add_argument("--replace-gamble", action="store_true",
                        help="ILS: ENABLE the insured terminal re-place "
                             "gamble stage: "
                             "late-wall full re-place draws with "
                             "place_design -net_delay_weight {medium,high} "
                             "variants from the banked best; adopt only at "
                             ">= +0.15ns over chain-best, routed + hold "
                             "clean; banked best on disk never touched; "
                             "fail-closed without a measured cost anchor. "
                             "Default OFF (RC ships it off; the all-in "
                             "window flips it on). Env fallback: "
                             "FPL26_REPLACE_GAMBLE=1 enables; "
                             "FPL26_NO_REPLACE_GAMBLE=1 force-disables "
                             "(wins over both).")
    parser.add_argument("--deep-replace", action="store_true",
                        help="ILS: ENABLE the DEEP-WNS full-replace sibling "
                             "(review-vetted). On the R1 "
                             "DEEP-extreme class (failing_endpoints >= 100k "
                             "AND |WNS| >= 10ns) re-place from the PRISTINE "
                             "input -- unplace, place_design -directive "
                             "Explore, route_design, phys_opt -directive "
                             "AlternateFlowWithRetiming -- the exact sequence "
                             "that produced boom_soc's +35.47 record against "
                             "our +13.32. Registers an insured-compare MUX "
                             "candidate, so it is never-worse; it changes NO "
                             "router decision. Fail-closed without measured "
                             "physics or a cost anchor. Default OFF. Env "
                             "fallback: FPL26_DEEP_REPLACE=1 enables; "
                             "FPL26_NO_DEEP_REPLACE=1 force-disables (wins "
                             "over both).")
    parser.add_argument("--ils-hurdle-continue", action="store_true",
                        help="ILS: when the futility counter (K) trips, consult "
                             "the SCORING FUNCTION before stopping. One more "
                             "cycle costs hurdle = alpha*0.1*(dt/3600)/P; a "
                             "~120s cycle at alpha 97 is only ~0.34 MHz, while "
                             "a record run's winning __LASTMILE__ "
                             "cycle yielded 5.4 MHz. Measured on chain29: ILS "
                             "held 2984s, spent 233s, stopped on K=2. Bounded — "
                             "authorises ONE cycle at a time; deadline, budget "
                             "and cycle caps still apply. No fitted constant. "
                             "Default OFF. Env: FPL26_ILS_HURDLE_CONTINUE=1.")
    parser.add_argument("--deep-replace-unbanded", action="store_true",
                        help="ILS: stop using the DEEP-extreme band "
                             "(failing>=100k AND |WNS|>=10ns) to REFUSE the "
                             "deep-replace stage. That band is a benefit "
                             "PREDICTION used as a VETO and is measurably "
                             "mis-scoped: logicnets (failing 1,529, |WNS| "
                             "0.978) sits three orders of magnitude outside it "
                             "and regen-from-pristine still beats operate by "
                             "+93.77 MHz in a measured run. Affordability, "
                             "pristine-input and cost-anchor gates all remain, "
                             "and the stage stays an insured-compare MUX "
                             "candidate, so a losing regeneration is discarded "
                             "BY MEASUREMENT. Deliberately does NOT substitute "
                             "a new fitted boundary. Default OFF. Env "
                             "fallback: FPL26_DEEP_REPLACE_UNBANDED=1.")
    parser.add_argument("--plan-critic", action="store_true",
                        help="LLM: ENABLE the cheap pre-flight PLAN CRITIC "
                             "(review-vetted). Before a "
                             "heavy Vivado move (place/route/phys_opt) a "
                             "second, cheaper model reviews the plan and its "
                             "verdict is appended as an ADVISORY note the "
                             "planner may ignore — it can never skip, cancel "
                             "or rewrite the call. Fires only while LLM spend "
                             "is under 70%% of budget (our worst design spent "
                             "36%% of ~$1 while losing 22 MHz). Capped at 4 "
                             "calls. Default OFF. Env fallback: "
                             "FPL26_PLAN_CRITIC=1 enables; "
                             "FPL26_NO_PLAN_CRITIC=1 force-disables (wins).")
    parser.add_argument("--plan-critic-model", default=PLAN_CRITIC_DEFAULT_MODEL,
                        help="Model used by --plan-critic. Deliberately a "
                             "cheap one: gemini nets ~401.8 vs grok's ~410.7 "
                             "at roughly 20%% of the cost, so it is a sound "
                             "reviewer even though grok stays the planner.")
    parser.add_argument("--ils-corrective-local-climb", action="store_true",
                        help="ILS dual-seed: let the corrective (recipe-best) "
                             "seed accept against its OWN seed floor and start "
                             "its combo rotation at the raw seed's pristine "
                             "position (forensic evidence: "
                             "re-enables the ExtraTimingOpt->PARTIAL_RUIN chain "
                             "that produced v2's best draw). Global keep-best "
                             "still decides what ships. Default OFF.")
    parser.add_argument(
        "--policy-card",
        action="store_true",
        help="Advisory only.  When passed (default OFF), the optimizer "
             "appends a compact POLICY_MEMORY_CARD block to the iter-1 "
             "prompt.  The card is feature-first (lut_count + "
             "critical_path_spread), validator-backed, and never echoes "
             "design names.  Default flow is unchanged when the flag is "
             "absent.",
    )
    parser.add_argument(
        "--capture-qor",
        action="store_true",
        help="Session 14 data accumulation: after finalize, spawn ONE "
             "subprocess-isolated Vivado batch that emits "
             "<run_dir>/<dcp_stem>.qor.json via report_design_analysis "
             "-qor_summary -json.  Default OFF.  Timeout governed by "
             "--capture-qor-timeout (default 60 s); "
             "failure/timeout/missing-DCP are logged as decision-trace "
             "records and NEVER block finalize, change lifecycle status, "
             "or write into submission/.  REPORT-ONLY; not consumed by "
             "live recipe selection, policy-card prompt, or pathology "
             "classifier in this version.",
    )
    parser.add_argument(
        "--policy-card-variant",
        choices=("default", "post_route_polish_v1", "route_bound_v1"),
        default="default",
        help="Session 20 mechanism-experiment variant for the policy-card "
             "prompt.  Default preserves Session-9 behaviour exactly.  "
             "Variants append a CLUSTER-DERIVED advisory ONLY when the "
             "candidate's lut_count + critical_path_spread match the "
             "variant's target cluster.  Never references design name.  "
             "Advisory_family is a LABEL from the HFCv0 allow-list, "
             "never a Vivado command.  Has no effect unless --policy-card "
             "is also passed.",
    )
    parser.add_argument(
        "--capture-qor-timeout",
        type=float,
        default=60.0,
        help="Session 17 tunable: timeout (seconds) for the post-finalize "
             "Vivado QoR-capture subprocess.  Default 60 s preserves "
             "Session-14 behaviour.  Recommended 180 s for DCPs > 80 MB "
             "(e.g. boom_soc, corescore_500_mod) where 60 s is too tight "
             "and the Tcl probe is killed before the JSON is fully "
             "flushed.  Affects ONLY the QoR-capture subprocess; does "
             "NOT touch the optimizer wall budget, recipe selection, "
             "policy-card prompt, or lifecycle status.  Values "
             "<= 0 are clamped to 60.",
    )
    parser.add_argument(
        "--phys-opt-preempt-after",
        type=int,
        default=None,
        help="Session 23 opt-in controller mechanism.  When set to an "
             "integer N >= 1, the controller tracks consecutive "
             "`reverted_no_gain` / `reverted_regression` per-flag outcomes "
             "from `recipe_post_route_phys_opt_sweep` and, after N of them "
             "with no intervening committed gain, injects a user message "
             "directing the LLM to run the deterministic heavy rescue "
             "sequence (`place_design -unplace` + Auto_1 + "
             "`route_design Default`) — provided wall budget remaining "
             ">= the budget floor and no meaningful WNS lift has been "
             "achieved yet.  Default None = OFF = Session-22 byte-identical "
             "behaviour.  Fired at most once per run.  Feature-blind and "
             "design-name-blind.  Records a `phys_opt_preempt` "
             "decision-trace event whether triggered or skipped.",
    )
    parser.add_argument(
        "--phys-opt-preempt-mode",
        choices=("end_sweep", "mid_sweep"),
        default="end_sweep",
        help="Session 24 opt-in: controls when the preempt check runs "
             "within `recipe_post_route_phys_opt_sweep`.  `end_sweep` "
             "(default) preserves Session-23 behaviour — check once "
             "after the full sweep returns.  `mid_sweep` checks after "
             "each per-flag entry and breaks the sweep loop on trigger "
             "(or on `skipped_reason=insufficient_budget`), saving wall "
             "time when the stall is detectable early.  Has no effect "
             "unless --phys-opt-preempt-after is also set.",
    )
    parser.add_argument(
        "--phys-opt-preempt-budget-floor",
        type=float,
        default=1100.0,
        help="Session 24 opt-in: wall-budget floor (seconds) below "
             "which the preempt is skipped with "
             "`skipped_reason=insufficient_budget`.  Default 1100 s "
             "matches Session-23 conservative floor (heavy rescue "
             "nominal ~1043 s + safety pad).  Lowering this trades "
             "route_design completion margin for an earlier trigger "
             "opportunity — operator-explicit choice only.  Clamped to "
             "[300, 1800].  Has no effect unless "
             "--phys-opt-preempt-after is also set.",
    )
    parser.add_argument(
        "--wall-handback",
        action="store_true",
        help="EXPERIMENTAL (04-01 D3, default off, needs A/B): gamma trim / "
             "wall handback.  When a sanctioned EXISTING saturation signal "
             "fires (ILS no-improve stop, LASTMILE reject, budget-kill) AND "
             "a banked accept exists, the LLM iteration loop exits to the "
             "finalize tail early — the ILS/LASTMILE/fanout polish stages "
             "still run once (locked OQ1) — then finalize + exit return the "
             "remaining wall to the multi-restart wrapper (gamma saved or a "
             "second draw funded).  OFF (default) = byte-identical behavior.",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=None,
        help="Wall-time budget per design (seconds).  None=unlimited (default, dev). "
             "Recommended Beta/eval value: 3300 (55 min, 5-min margin under the "
             "60-min contest cap).  When exhausted, optimizer stops cleanly and "
             "emits the best valid output found so far.",
    )
    parser.add_argument(
        "--llm-cost-budget",
        type=float,
        default=None,
        help="β circuit-breaker: per-attempt LLM spend budget ($). "
             "Effective in-attempt cost exit = min(0.75, this).  The "
             "multi-restart wrapper passes (cumulative ceiling − prior "
             "attempts' spend) so attempt N never re-spends a full fresh "
             "allowance — the eval ZEROES a benchmark at $1.00 cumulative "
             "LLM spend (preview #15).  Unset/<=0 keeps the default $0.75 "
             "exit.  Env fallback: FPL26_LLM_COST_BUDGET (CLI wins).",
    )
    parser.add_argument(
        "--polish-reserve-s",
        type=float,
        default=None,
        help="Post-route polish reserve (seconds). Once a routed "
             "banked best exists and polish stages are pending, "
             "speculative routed-state-destroying ops (new ILS ruin "
             "cycles, additional full place/route attempts) must fit "
             "(remaining − reserve); the polish stages, finalize, and "
             "banking keep the full remaining window. boom_soc_v2 "
             "official beta: final post-route phys_opt polish refused at "
             "est 600s > 443s remaining because the tail was already "
             "burned. Default 500. 0 disables. Sits ON TOP of the 300s "
             "finalize reserve (orthogonal to the cost ceiling). "
             "Env fallback: FPL26_POLISH_RESERVE_S (CLI wins).",
    )
    parser.add_argument(
        "--no-bare-reroute-polish",
        action="store_true",
        help="Disable the bare re-route polish (default ON; measured "
             "+0.094 ns deterministically, with the undirected re-route "
             "beating every directed variant).  When ILS never armed (e.g. boom-class "
             "designs size-gated at >300k cells) and a routed banked "
             "best exists and the re-route fits the remaining wall per "
             "the route_gate predictor, the exit tail runs ONE bare "
             "`route_design` (no directive, no -unroute) on the banked "
             "best via the normal auto-bank/phantom-guard path "
             "(never-worse; counts as a polish stage and releases "
             "the reserve on completion).  Env kill switch: "
             "FPL26_NO_BARE_REROUTE_POLISH=1 (CLI or env disables).",
    )
    parser.add_argument(
        "--bare-reroute-min-gain",
        type=float,
        default=None,
        help="Bare re-route polish loop: minimum measured WNS gain (ns) "
             "per iteration to keep re-rolling (plateau-probe evidence: the "
             "rip-up re-roll COMPOUNDS while WNS is deep — boom 2nd pass "
             "+0.404, cumulative +0.62 over two passes — and decays near "
             "plateaus, fir +0.004).  Default 0.020 (above measurement "
             "noise, below the observed +0.094/+0.215/+0.404 gain band). "
             "Negative/invalid keeps the default.  Env fallback: "
             "FPL26_BARE_REROUTE_MIN_GAIN (CLI wins).",
    )
    parser.add_argument(
        "--bare-reroute-max-iters",
        type=int,
        default=None,
        help="Bare re-route polish loop: runaway guard on the number of "
             "route_design iterations (default 4; the observed plateau "
             "decay stops the loop first).  1 restores the pre-loop "
             "one-shot behavior.  <1/invalid keeps the default.  Env "
             "fallback: FPL26_BARE_REROUTE_MAX_ITERS (CLI wins).",
    )
    parser.add_argument(
        "--no-tail-controller",
        action="store_true",
        help="Disable the adaptive banked tail CONTROLLER (default ON; "
             "measured-portfolio policy).  On deep-WNS states (best_wns <= "
             "--tail-ctrl-deep-wns, default -1.0) the exit tail replaces "
             "the plain M1 bare-reroute loop with a measured-dns/s "
             "portfolio policy over the proven move menu (bare route / "
             "granular ladder / AggressiveFanoutOpt / phys_opt "
             "AggressiveExplore; plateau-probe chain evidence +0.726 on "
             "boom) and harvests to the wall floor.  phys_opt moves bank "
             "only through a hold-gated accept; fail-closed to the plain "
             "M1 loop on any controller error.  Env kill switch: "
             "FPL26_NO_TAIL_CONTROLLER=1 (CLI or env disables).",
    )
    parser.add_argument(
        "--tail-ctrl-deep-wns",
        type=float,
        default=None,
        help="Tail controller arming threshold in ns (default -1.0, "
             "round-2 loosened gate; must be <= 0).  The controller arms "
             "when the banked best WNS <= threshold; shallower states "
             "keep the shipped plain M1 loop (plateau probe: the move "
             "menu decays to no-op near-met).  Env fallback: "
             "FPL26_TAIL_CTRL_DEEP_WNS (CLI wins).",
    )
    parser.add_argument(
        "--tail-ctrl-max-moves",
        type=int,
        default=None,
        help="Tail controller global runaway guard on move executions "
             "(default 12; per-move retire/decay and the wall floor "
             "normally stop the loop first).  Env fallback: "
             "FPL26_TAIL_CTRL_MAX_MOVES (CLI wins).",
    )
    parser.add_argument(
        "--tail-ctrl-m1-echo",
        action="store_true",
        help="POST-ACCEPT M1 ECHO (default OFF; "
             "terra — A/B mechanism, zero diff when off).  When set, "
             "every ADOPTED non-M1 tail-controller move is immediately "
             "followed by ONE bare route_design echo from the fresh "
             "banked state (M1 re-bite evidence: +0.048/681s on "
             "already-harvested boom states).  The echo rides the "
             "auto-bank hook, counts toward --tail-ctrl-max-moves, "
             "needs its own wall affordability, and records into the "
             "m1_route move ledger (a below-min echo retires M1 like a "
             "picked M1 would).  Env fallback: FPL26_TAIL_CTRL_M1_ECHO=1 "
             "(either enables).",
    )
    parser.add_argument(
        "--deep-wns-tail-reserve",
        type=float,
        default=None,
        help="DEEP-WNS TAIL RESERVE in seconds (DEFAULT 0 = OFF = zero "
             "behavior change; LLM-bypass A/B infrastructure). "
             "When > 0 and the design is STILL deep-WNS (current "
             "best_wns <= --tail-ctrl-deep-wns, default -1.0) at the "
             "moment remaining wall <= reserve, the main LLM loop exits "
             "early through the shared exit tail so the deterministic "
             "tail (bare-reroute loop / tail controller) gets a "
             "deliberate window (leg_boom evidence: the loop otherwise "
             "runs the entire wall and the tail gets ZERO seconds). "
             "Recommended 2400 (~2 tail moves at eval speed: bare route "
             "~850-1600 s + phys_opt ~400-770 s + banking margins). "
             "A value in (0, 1) is a fraction of --max-wall-seconds; a "
             "reserve >= the wall is the full LLM bypass on deep-WNS "
             "designs.  Env fallback: FPL26_DEEP_WNS_TAIL_RESERVE "
             "(CLI wins).",
    )
    parser.add_argument(
        "--phase1-timeout-scale",
        type=float,
        default=1.0,
        help="Scale factor for every Phase-1 timeout (open_checkpoint, "
             "report_timing, high-fanout, spread analysis).  Default 1.0; "
             "use 2.0-4.0 for very large designs (boom_soc, ispd16_example2) "
             "where the default 300-600 s server-side budget is too tight. "
             "Mandatory steps (open + report_timing) failing under this "
             "scale will abort; optional steps will be skipped gracefully.",
    )
    parser.add_argument(
        "--phase1-wall-frac",
        type=float,
        default=None,
        help="Cumulative Phase-1 wall cap: fraction of "
             "--max-wall-seconds Phase 1 may consume before remaining "
             "OPTIONAL analysis steps are skipped (reason phase1_wall_cap) "
             "and optional timeouts are clamped to the remaining allowance. "
             "Mandatory steps (open_checkpoint, report_timing_summary) are "
             "never capped. Default 0.15 (largest known design uses ~4.8% "
             "at eval speed). Must be in (0, 1]. Env fallback: "
             "FPL26_PHASE1_WALL_FRAC (CLI wins). Inert when "
             "--max-wall-seconds is unset.",
    )
    parser.add_argument(
        "--fresh-presweep-draws",
        type=int,
        default=None,
        help="FRESH-STATE ROUTE-LOTTERY PRE-SWEEP: "
             "number K of banked route re-rolls (clamped 1..3) taken on "
             "the PRISTINE input state at step 0, before the recipe/LLM "
             "loop. Draw 1 = route_design -unroute + route_design "
             "-directive AggressiveExplore (the wave-1 measured form: fir "
             "+0.070 / vtr +0.306 / optical +0.075 from fresh states); "
             "chained draws use plain route_design. Best draw (never-worse "
             "vs entry; tracker-first routed gate + hold gate) is banked "
             "via the eager mirror and becomes the pipeline entry state; "
             "worse/failed draws re-open the input DCP (zero alpha risk). "
             "Total spend capped at 0.2 x --max-wall-seconds; draw 1 runs "
             "under a 0.12 x wall timeout and its OBSERVED cost gates "
             "draws 2..K (spent + observed*1.3 <= 0.2 x wall). Default "
             "0/unset = OFF (zero behavior change). Env fallback: "
             "FPL26_FRESH_PRESWEEP_DRAWS (CLI wins).",
    )

    args = parser.parse_args()
    
    # Validate inputs
    if not args.input_dcp.exists():
        print(f"Error: Input file not found: {args.input_dcp}", file=sys.stderr)
        sys.exit(1)
    
    # Generate default output DCP name if not provided
    if args.output_dcp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        input_stem = args.input_dcp.stem  # Filename without extension
        input_dir = args.input_dcp.parent  # Directory of input file
        args.output_dcp = input_dir / f"{input_stem}_optimized-{timestamp}.dcp"
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory if needed
    args.output_dcp.parent.mkdir(parents=True, exist_ok=True)
    
    # Test mode - run without LLM
    if args.test:
        # Create run directory with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = _run_dir_base() / f"dcp_optimizer_run-{timestamp}"
        
        print(f"FPGA Design Optimization - TEST MODE")
        print(f"=====================================")
        print(f"Input:       {args.input_dcp.resolve()}")
        print(f"Output:      {args.output_dcp.resolve()}")
        print(f"Run dir:     {run_dir}")
        print(f"Max nets to optimize: {args.max_nets}")
        print()
        
        exit_code = await run_test_mode(
            args.input_dcp, 
            args.output_dcp, 
            debug=args.debug,
            max_nets=args.max_nets,
            run_dir=run_dir
        )
        sys.exit(exit_code)
    
    # Normal mode - requires API key and LLM
    if not args.api_key:
        print("Error: OpenRouter API key required. Set OPENROUTER_API_KEY or use --api-key", file=sys.stderr)
        print("       Use --test flag to run in test mode without LLM", file=sys.stderr)
        sys.exit(1)
    
    if OpenAI is None:
        print("Error: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)
    
    # Create run directory with timestamp (before the optimizer, so it can be shown)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = _run_dir_base() / f"dcp_optimizer_run-{timestamp}"
    
    print(f"FPGA Design Optimization Agent")
    print(f"================================")
    print(f"Input:       {args.input_dcp.resolve()}")
    print(f"Output:      {args.output_dcp.resolve()}")
    print(f"Run dir:     {run_dir}")
    print(f"Model:       {args.model}")
    print()
    
    optimizer = DCPOptimizer(
        api_key=args.api_key,
        model=args.model,
        debug=args.debug,
        run_dir=run_dir,
        mode=args.mode,
    )
    optimizer.rag_seed = not args.no_rag_seed
    optimizer.contest_mode = bool(args.contest_mode)
    if getattr(args, "ils_polish", False):
        optimizer._ils_polish_cfg.enabled = True
        if getattr(args, "ils_stagnation_seconds", None) is not None:
            optimizer._ils_polish_cfg.stagnation_seconds = float(args.ils_stagnation_seconds)
        if getattr(args, "ils_min_remaining_s", None) is not None:
            optimizer._ils_polish_cfg.min_remaining_s = float(args.ils_min_remaining_s)
        if getattr(args, "ils_max_cells", None) is not None:
            optimizer._ils_polish_cfg.max_cells = int(args.ils_max_cells)
        if getattr(args, "ils_futility_stop", None) is not None:
            optimizer._ils_polish_cfg.final_seed_no_improve_stop = int(args.ils_futility_stop)
        if getattr(args, "ils_corrective_local_climb", False):
            optimizer._ils_polish_cfg.corrective_local_climb = True
    # K3 PARTIAL_RUIN spread-gate kill switch (default ON in the dataclass):
    # CLI --no-partial-ruin-spread-gate OR env FPL26_PARTIAL_RUIN_SPREAD_GATE
    # in {0,false,off,no} disables. Wired outside the --ils-polish block so
    # the switch also covers any future ILS enable path.
    _sg_env = os.environ.get("FPL26_PARTIAL_RUIN_SPREAD_GATE", "").strip().lower()
    if (getattr(args, "no_partial_ruin_spread_gate", False)
            or _sg_env in ("0", "false", "off", "no")):
        optimizer._ils_polish_cfg.partial_ruin_spread_gate = False
        print("ILS: PARTIAL_RUIN spread gate DISABLED (kill switch)")
    # ROUTE_REROLL kill switch (default ON in the dataclass): CLI
    # --no-route-reroll OR env FPL26_NO_ROUTE_REROLL in {1,true,on,yes}
    # disables. Wired outside the --ils-polish block per the spread-gate
    # convention so the switch covers any ILS enable path.
    _rr_env = os.environ.get("FPL26_NO_ROUTE_REROLL", "").strip().lower()
    if (getattr(args, "no_route_reroll", False)
            or _rr_env in ("1", "true", "on", "yes")):
        optimizer._ils_polish_cfg.route_reroll_enabled = False
        print("ILS: ROUTE_REROLL combo DISABLED (kill switch)")
    # ROUTE_REROLL near-met band override.  Measured scatter: the re-roll
    # bites on fresh states out to at least -1.1, but post-recipe-state
    # evidence still says 0.7 — the env knob lets a wider band be A/B
    # tested without code edits.  Invalid/<=0
    # keeps the dataclass default 0.7.
    _rr_band = os.environ.get("FPL26_ROUTE_REROLL_BAND", "").strip()
    if _rr_band:
        try:
            _rr_band_v = float(_rr_band)
            if _rr_band_v > 0:
                optimizer._ils_polish_cfg.route_reroll_max_wns_mag = _rr_band_v
                print(f"ILS: ROUTE_REROLL band override -> {_rr_band_v} ns "
                      f"(env FPL26_ROUTE_REROLL_BAND; default 0.7)")
        except ValueError:
            logger.warning(
                f"FPL26_ROUTE_REROLL_BAND={_rr_band!r} not a float; "
                f"keeping default band.")
    # Replace-gamble enable switch, default off: --replace-gamble or
    # FPL26_REPLACE_GAMBLE enables it, and FPL26_NO_REPLACE_GAMBLE
    # force-disables and wins over both.  Wired outside the --ils-polish
    # block, following the spread-gate convention.
    _rg_env = os.environ.get("FPL26_REPLACE_GAMBLE", "").strip().lower()
    _rg_off = os.environ.get("FPL26_NO_REPLACE_GAMBLE", "").strip().lower()
    if (getattr(args, "replace_gamble", False)
            or _rg_env in ("1", "true", "on", "yes")):
        optimizer._ils_polish_cfg.replace_gamble_enabled = True
        print("ILS: terminal REPLACE_GAMBLE stage ENABLED "
              "(insured re-place gamble; adopt >= +0.15ns over chain-best)")
    if _rg_off in ("1", "true", "on", "yes"):
        if optimizer._ils_polish_cfg.replace_gamble_enabled:
            print("ILS: terminal REPLACE_GAMBLE stage DISABLED (kill switch "
                  "FPL26_NO_REPLACE_GAMBLE wins)")
        optimizer._ils_polish_cfg.replace_gamble_enabled = False
    # DEEP_REPLACE enable switch. Same
    # convention/precedence as REPLACE_GAMBLE above: CLI --deep-replace OR
    # env FPL26_DEEP_REPLACE enables; FPL26_NO_DEEP_REPLACE force-disables
    # and wins over both. Default OFF until validated at scale.
    _dr_env = os.environ.get("FPL26_DEEP_REPLACE", "").strip().lower()
    _dr_off = os.environ.get("FPL26_NO_DEEP_REPLACE", "").strip().lower()
    if (getattr(args, "deep_replace", False)
            or _dr_env in ("1", "true", "on", "yes")):
        optimizer._ils_polish_cfg.deep_replace_enabled = True
        print("ILS: DEEP_REPLACE sibling ENABLED (full re-place from the "
              "PRISTINE input on the DEEP-extreme class; MUX candidate, "
              "never-worse)")
    if _dr_off in ("1", "true", "on", "yes"):
        if optimizer._ils_polish_cfg.deep_replace_enabled:
            print("ILS: DEEP_REPLACE sibling DISABLED (kill switch "
                  "FPL26_NO_DEEP_REPLACE wins)")
        optimizer._ils_polish_cfg.deep_replace_enabled = False
    # UNBANDED (default OFF): stop using the DEEP-extreme band to refuse
    # the stage. Same resolver convention as above; affordability and never-worse
    # are untouched. Ships OFF until validated at scale.
    # ILS hurdle continuation resolver (default OFF).
    _ihc_env = os.environ.get("FPL26_ILS_HURDLE_CONTINUE", "").strip().lower()
    if (getattr(args, "ils_hurdle_continue", False)
            or _ihc_env in ("1", "true", "on", "yes")):
        optimizer._ils_hurdle_continue = True
        print("ILS: HURDLE CONTINUATION enabled — futility K may be overridden "
              "for one more cycle when the scoring function says it pays")
    _dru_env = os.environ.get("FPL26_DEEP_REPLACE_UNBANDED", "").strip().lower()
    if (getattr(args, "deep_replace_unbanded", False)
            or _dru_env in ("1", "true", "on", "yes")):
        optimizer._deep_replace_unbanded = True
        print("ILS: DEEP_REPLACE UNBANDED — the DEEP-extreme band no longer "
              "vetoes the stage; affordability + insured-compare MUX decide "
              "(a prediction may schedule work, never refuse it)")
    # DEEP_REPLACE_FIRST: promote the recipe from the exit tail to BEFORE the
    # LLM loop. Same precedence convention; additionally requires
    # deep_replace_enabled, since "first" only reorders an enabled stage.
    _drf_env = os.environ.get("FPL26_DEEP_REPLACE_FIRST", "").strip().lower()
    _drf_off = os.environ.get("FPL26_NO_DEEP_REPLACE_FIRST", "").strip().lower()
    if (getattr(args, "deep_replace_first", False)
            or _drf_env in ("1", "true", "on", "yes")):
        if optimizer._ils_polish_cfg.deep_replace_enabled:
            optimizer._ils_polish_cfg.deep_replace_first_enabled = True
            print("ILS: DEEP_REPLACE runs FIRST (before the LLM loop; the "
                  "recipe needs ~79% of the wall, so it cannot be a tail "
                  "supplement). Loop gets the remainder; baseline stays "
                  "insured via the finalize MUX.")
        else:
            print("ILS: --deep-replace-first IGNORED (requires "
                  "--deep-replace / FPL26_DEEP_REPLACE)")
    if _drf_off in ("1", "true", "on", "yes"):
        if optimizer._ils_polish_cfg.deep_replace_first_enabled:
            print("ILS: DEEP_REPLACE_FIRST DISABLED (kill switch "
                  "FPL26_NO_DEEP_REPLACE_FIRST wins); stage stays at the tail")
        optimizer._ils_polish_cfg.deep_replace_first_enabled = False
    # PLAN_CRITIC enable switch. Same convention/precedence as
    # the stages above: CLI --plan-critic OR env FPL26_PLAN_CRITIC enables;
    # FPL26_NO_PLAN_CRITIC force-disables and wins. Default OFF.
    _pc_env = os.environ.get("FPL26_PLAN_CRITIC", "").strip().lower()
    _pc_off = os.environ.get("FPL26_NO_PLAN_CRITIC", "").strip().lower()
    if (getattr(args, "plan_critic", False)
            or _pc_env in ("1", "true", "on", "yes")):
        optimizer.plan_critic_enabled = True
        optimizer.plan_critic_model = str(
            getattr(args, "plan_critic_model", PLAN_CRITIC_DEFAULT_MODEL))
        print(f"LLM: PLAN_CRITIC ENABLED (advisory pre-flight review by "
              f"{optimizer.plan_critic_model}; cannot veto)")
    if _pc_off in ("1", "true", "on", "yes"):
        if optimizer.plan_critic_enabled:
            print("LLM: PLAN_CRITIC DISABLED (kill switch "
                  "FPL26_NO_PLAN_CRITIC wins)")
        optimizer.plan_critic_enabled = False
    optimizer.policy_card = bool(args.policy_card)
    optimizer.policy_card_variant = str(args.policy_card_variant)
    optimizer.capture_qor = bool(args.capture_qor)
    optimizer.capture_qor_timeout = float(args.capture_qor_timeout)
    # Wire the opt-in preempt flag.  None = OFF.
    if args.phys_opt_preempt_after is not None:
        try:
            _n = int(args.phys_opt_preempt_after)
            optimizer.phys_opt_preempt_after = _n if _n >= 1 else None
        except (TypeError, ValueError):
            optimizer.phys_opt_preempt_after = None
    else:
        optimizer.phys_opt_preempt_after = None
    # Preempt mode and budget floor.  Both inert when preempt-after
    # is None.  Mode is choice-validated by argparse; budget floor is
    # clamped to [300, 1800] with a warning on out-of-range.
    optimizer.phys_opt_preempt_mode = str(
        getattr(args, "phys_opt_preempt_mode", "end_sweep") or "end_sweep"
    )
    try:
        _floor = float(getattr(args, "phys_opt_preempt_budget_floor", 1100.0))
    except (TypeError, ValueError):
        _floor = 1100.0
    if _floor < 300.0 or _floor > 1800.0:
        logger.warning(
            f"--phys-opt-preempt-budget-floor={_floor} outside [300, 1800]; "
            f"falling back to 1100."
        )
        _floor = 1100.0
    optimizer.phys_opt_preempt_budget_floor_s = _floor
    optimizer.phase1_timeout_scale = float(args.phase1_timeout_scale)
    # Cumulative Phase-1 wall cap fraction (CLI wins over env;
    # invalid/unset keeps the 0.15 default — see resolve_phase1_wall_frac).
    optimizer.phase1_wall_frac = resolve_phase1_wall_frac(
        getattr(args, "phase1_wall_frac", None))
    # Post-route polish reserve (CLI wins over env; 0 disables;
    # unset/invalid keeps the 500s default — see resolve_polish_reserve_s).
    optimizer._polish_reserve_s = resolve_polish_reserve_s(
        getattr(args, "polish_reserve_s", None))
    if optimizer._polish_reserve_s != POLISH_RESERVE_S_DEFAULT:
        logger.info(
            f"[polish-reserve] window set to "
            f"{optimizer._polish_reserve_s:.0f}s (default "
            f"{POLISH_RESERVE_S_DEFAULT:.0f}s"
            + ("; reserve disabled" if optimizer._polish_reserve_s == 0.0
               else "") + ").")
    # Bare re-route polish (measured +0.094 ns deterministic in a probe on
    # one benchmark; undirected beat directed there).  Default ON;
    # --no-bare-reroute-polish or FPL26_NO_BARE_REROUTE_POLISH disables.
    optimizer._bare_reroute_polish_enabled = resolve_bare_reroute_polish_enabled(
        getattr(args, "no_bare_reroute_polish", False))
    if not optimizer._bare_reroute_polish_enabled:
        print("bare-reroute polish DISABLED (kill switch)")
    # Iteration-loop knobs (measured: re-roll compounds while
    # WNS is deep; decays near plateaus).  CLI wins over env; invalid/
    # unset keeps the defaults (0.020 ns / 4 iters).
    optimizer._bare_reroute_min_gain_ns = resolve_bare_reroute_min_gain_ns(
        getattr(args, "bare_reroute_min_gain", None))
    optimizer._bare_reroute_max_iters = resolve_bare_reroute_max_iters(
        getattr(args, "bare_reroute_max_iters", None))
    if (optimizer._bare_reroute_min_gain_ns != BARE_REROUTE_MIN_GAIN_NS_DEFAULT
            or optimizer._bare_reroute_max_iters
            != BARE_REROUTE_MAX_ITERS_DEFAULT):
        logger.info(
            f"[bare-reroute] loop knobs: min-gain "
            f"{optimizer._bare_reroute_min_gain_ns:.3f} ns (default "
            f"{BARE_REROUTE_MIN_GAIN_NS_DEFAULT:.3f}), max-iters "
            f"{optimizer._bare_reroute_max_iters} (default "
            f"{BARE_REROUTE_MAX_ITERS_DEFAULT}).")
    # Adaptive banked tail controller: default ON,
    # deep-WNS-gated, fail-closed to the plain M1 loop.
    optimizer._tail_controller_enabled = resolve_tail_controller_enabled(
        getattr(args, "no_tail_controller", False))
    if not optimizer._tail_controller_enabled:
        logger.info("[tail-ctrl] disabled (CLI/env kill switch); deep-WNS "
                    "states keep the plain bare-reroute loop.")
    optimizer._tail_ctrl_deep_wns_ns = resolve_tail_ctrl_deep_wns_ns(
        getattr(args, "tail_ctrl_deep_wns", None))
    optimizer._tail_ctrl_max_moves = resolve_tail_ctrl_max_moves(
        getattr(args, "tail_ctrl_max_moves", None))
    if (optimizer._tail_ctrl_deep_wns_ns != TAIL_CTRL_DEEP_WNS_NS_DEFAULT
            or optimizer._tail_ctrl_max_moves != TAIL_CTRL_MAX_MOVES_DEFAULT):
        logger.info(
            f"[tail-ctrl] knobs: deep-wns "
            f"{optimizer._tail_ctrl_deep_wns_ns:.3f} ns (default "
            f"{TAIL_CTRL_DEEP_WNS_NS_DEFAULT:.3f}), max-moves "
            f"{optimizer._tail_ctrl_max_moves} (default "
            f"{TAIL_CTRL_MAX_MOVES_DEFAULT}).")
    # POST-ACCEPT M1 ECHO: default OFF,
    # opt-in via CLI --tail-ctrl-m1-echo or env FPL26_TAIL_CTRL_M1_ECHO
    # (resolve_tail_ctrl_m1_echo; zero behavior diff when off).
    optimizer._tail_ctrl_m1_echo = resolve_tail_ctrl_m1_echo(
        getattr(args, "tail_ctrl_m1_echo", False))
    if optimizer._tail_ctrl_m1_echo:
        logger.info(
            "[tail-ctrl] M1 ECHO ARMED (default OFF): one bare "
            "route_design after every ADOPTED non-M1 move, counted "
            "toward max-moves, affordability-gated.")
    # Deep-WNS tail reserve: default OFF (0.0 =
    # zero behavior change); CLI wins over env — see
    # resolve_deep_wns_tail_reserve_s.
    optimizer._deep_wns_tail_reserve_s = resolve_deep_wns_tail_reserve_s(
        getattr(args, "deep_wns_tail_reserve", None))
    if optimizer._deep_wns_tail_reserve_s > 0.0:
        _rsv = optimizer._deep_wns_tail_reserve_s
        logger.info(
            "[tail-reserve] ARMED (v2): "
            + (f"{_rsv:.2f} of the wall" if _rsv < 1.0 else f"{_rsv:.0f}s")
            + " reserved for the deterministic tail on deep-WNS designs "
            f"(threshold {optimizer._tail_ctrl_deep_wns_ns:.3f} ns; "
            f"default OFF; v2 gates: wall clamp "
            f"{TAIL_RESERVE_WALL_CLAMP_FRAC:.3f}x, stagnation "
            f"{optimizer._tail_reserve_stagnant_s:.0f}s, "
            f"ILS-size-gated designs only).")
    # FRESH-STATE ROUTE-LOTTERY PRE-SWEEP: default
    # OFF (0 draws = zero behavior change); CLI wins over env — see
    # resolve_fresh_presweep_draws.
    optimizer._fresh_presweep_draws = resolve_fresh_presweep_draws(
        getattr(args, "fresh_presweep_draws", None))
    if optimizer._fresh_presweep_draws > 0:
        logger.info(
            f"[pre-sweep] ARMED: K={optimizer._fresh_presweep_draws} "
            "fresh-state route re-roll draw(s) at step 0 (budget "
            f"{FRESH_PRESWEEP_BUDGET_FRAC:.0%} of wall, first-draw "
            f"timeout {FRESH_PRESWEEP_FIRST_DRAW_FRAC:.0%} of wall; "
            "default OFF).")
    # Wall handback: kill switch, default OFF.
    optimizer._wall_handback_enabled = bool(getattr(args, "wall_handback", False))
    optimizer.max_wall_seconds = float(args.max_wall_seconds) if args.max_wall_seconds else None
    # β circuit-breaker: tightening-only per-attempt spend budget.
    # CLI wins over env; the wrapper plumbs it as make var LLM_COST_BUDGET
    # -> --llm-cost-budget.  resolve_llm_cost_exit() clamps to <= $0.75.
    _cost_budget = getattr(args, "llm_cost_budget", None)
    if _cost_budget is None:
        _env_budget = os.environ.get("FPL26_LLM_COST_BUDGET", "").strip()
        if _env_budget:
            try:
                _cost_budget = float(_env_budget)
            except ValueError:
                logger.warning(
                    f"FPL26_LLM_COST_BUDGET={_env_budget!r} not a float; "
                    f"keeping default ${LLM_COST_EXIT_USD:.2f} cost exit.")
    optimizer.llm_cost_exit_usd = resolve_llm_cost_exit(_cost_budget)
    if optimizer.llm_cost_exit_usd != LLM_COST_EXIT_USD:
        logger.info(
            f"[cost-exit] in-attempt LLM cost exit tightened to "
            f"${optimizer.llm_cost_exit_usd:.2f} (budget from wrapper/env; "
            f"default ${LLM_COST_EXIT_USD:.2f}).")

    # Make the optimizer visible to signal handlers — they need to read
    # _best_valid_dcp / _best_valid_edif / run_dir to ship a fast emergency
    # finalize without re-entering Vivado.
    global _ACTIVE_OPTIMIZER
    _ACTIVE_OPTIMIZER = optimizer

    def _emergency_baseline_copy(reason: str) -> None:
        """Idempotent emergency finalize.

        Preference order (best → worst):
          1. <run_dir>/best_valid.dcp + .edf — produced after each
             improvement by _mirror_best_valid_now.  Vivado-independent
             fast path (pure shutil.copy).  Ships the actual optimization
             gains even if the optimizer was killed mid-iteration.
          2. The optimizer's own _finalize_output_dcp output, if it ran
             to completion.
          3. The baseline DCP copied to the output path — guarantees a
             submission-safe DCP exists, ΔFmax = 0.

        Always writes <run_dir>/lifecycle_metadata.json with the
        termination reason, elapsed wall time, best WNS known, and the
        chosen finalize source so post-hoc analysis can tell at a glance
        what actually shipped.

        This function is callable from signal handlers, async exception
        handlers, and the normal post-optimize() path.  Idempotent: the
        first invocation wins and later calls are no-ops.
        """
        if _TERMINATION_STATE["finalized"]:
            return
        _TERMINATION_STATE["finalized"] = True
        _TERMINATION_STATE["reason"] = reason

        chosen = None  # "best_valid" | "optimizer_output" | "baseline" | "none"
        try:
            out = Path(args.output_dcp)
            best_dcp = getattr(optimizer, "_best_valid_dcp", None)
            best_edf = getattr(optimizer, "_best_valid_edif", None)
            out_edf = out.with_suffix(".edf")

            # An existing output_dcp only
            # proves finalize ran if _finalize_completed is set — the LLM
            # is explicitly instructed to write to the output path
            # mid-run, and a regressed mid-run write must not outrank the
            # best_valid mirror (an observed incident class).
            finalize_done = bool(getattr(optimizer, "_finalize_completed",
                                         False))
            # A finalize that reported done, plus a file at the output path, is
            # still not proof the file is ours: the fault-injection tests write
            # garbage there after finalize.  Verify size, magic bytes and md5
            # against the identity finalize recorded; any mismatch, or an
            # exception during verification, demotes the on-disk file so the
            # mirror or baseline restore path wins.  No recorded identity
            # keeps the legacy behaviour.
            ship_integrity_fail = False
            _shipped = getattr(optimizer, "_shipped_artifact", None)
            if (finalize_done and isinstance(_shipped, dict)
                    and out.exists()):
                _ok, _why = _verify_shipped_identity(out, _shipped)
                if not _ok:
                    ship_integrity_fail = True
                    print(
                        "[LIFECYCLE] SHIP INTEGRITY FAIL: output does not "
                        f"match finalized artifact ({_why}) — restoring."
                    )
            need_dcp = ((not out.exists()) or out.stat().st_size == 0
                        or not finalize_done or ship_integrity_fail)

            # T5 bank_mirror-truncation variant: distrust a mirror whose
            # on-disk size no longer matches the size recorded when it was
            # banked (an externally truncated/garbaged mirror must never be
            # shipped as "best").  Size-only by design — cheap and catches
            # both truncation and the injected-garbage byte pattern.
            mirror_usable = bool(
                best_dcp and Path(best_dcp).exists()
                and Path(best_dcp).stat().st_size > 0)
            _mirror_size_ref = getattr(
                optimizer, "_best_valid_mirror_size", None)
            if (mirror_usable and _mirror_size_ref is not None
                    and Path(best_dcp).stat().st_size != _mirror_size_ref):
                print(
                    "[LIFECYCLE] SHIP INTEGRITY FAIL: best_valid mirror "
                    f"size {Path(best_dcp).stat().st_size} != banked size "
                    f"{_mirror_size_ref} — distrusting mirror."
                )
                mirror_usable = False

            # Path 1: stable best_valid mirror is on disk.  Prefer this over
            # anything else — it represents the LLM's best validated state
            # and survives Vivado death.
            if need_dcp and mirror_usable:
                # Each rung of this ladder needs its own try.  In one flat try,
                # a raise from this copy — a full destination disk is the live
                # case, since the atomic copy re-raises once its own fallback
                # fails — jumps straight to the outer handler, so the rung that
                # guarantees a submission-safe checkpoint never runs, and the
                # finalized latch blocks every later retry.  A failed copy must
                # fall through the ladder, not out of it.
                try:
                    # Audit emergency-path copies too.
                    # Optimizer may have died but its guard reference still
                    # works in audit mode (never raises).
                    try:
                        optimizer._path_guard_check(
                            out, context="emergency_best_valid_copy",
                        )
                        if best_edf and Path(best_edf).exists() and Path(best_edf).stat().st_size > 0:
                            optimizer._path_guard_check(
                                out_edf, context="emergency_best_valid_edif_copy",
                            )
                    except Exception:
                        pass
                    _atomic_copy(str(best_dcp), str(out))
                    # The EDIF copy gets its own guard: a raise here would void
                    # the choice with the checkpoint already restored, and the
                    # ladder would then fall through and overwrite a good mirror
                    # with the baseline.  Losing the EDIF is not fatal — the
                    # validator regenerates the sidecar itself.
                    try:
                        if best_edf and Path(best_edf).exists() and Path(best_edf).stat().st_size > 0:
                            _atomic_copy(str(best_edf), str(out_edf))
                    except Exception as e_edf:
                        print(f"[LIFECYCLE] EMERGENCY EDIF copy failed "
                              f"(non-fatal, DCP already restored): {e_edf!r}")
                    chosen = "best_valid"
                    # The mirror may have banked less than the
                    # tracked best; the last-resort token_usage write below
                    # must describe the artifact (see _shipped_wns_ns in
                    # save_token_usage_report).
                    _mw = getattr(optimizer, "_best_valid_dcp_wns", None)
                    if _mw is not None:
                        optimizer._shipped_wns_ns = _mw
                    print(
                        f"[LIFECYCLE] EMERGENCY_BEST_VALID_COPY ({reason}): "
                        f"copied {Path(best_dcp).name} → {out.name} "
                        "(plus EDIF if present)."
                    )
                    # Publish the identity of what was just shipped, so the
                    # multi-restart wrapper's verified emergency publish can
                    # distinguish this artifact from injected garbage.
                    try:
                        _ident = _artifact_identity(out)
                        if _ident is not None:
                            optimizer._shipped_artifact = _ident
                            _write_shipped_manifest(out, _ident)
                    except Exception:
                        pass
                except Exception as e1:
                    chosen = None
                    print(
                        f"[LIFECYCLE] EMERGENCY_BEST_VALID_COPY FAILED "
                        f"({reason}): {e1!r} — falling through to the "
                        f"baseline guarantee."
                    )
            if chosen is None and out.exists() and out.stat().st_size > 0 and not ship_integrity_fail:
                # Path 2: optimizer's own output_dcp already exists (normal
                # finalize ran).  Leave it; signal/exception arrived after
                # the finalize completed.
                chosen = "optimizer_output"
                print(
                    f"[LIFECYCLE] EMERGENCY_NO_OP ({reason}): "
                    f"output_dcp already exists ({out.stat().st_size} B); leaving in place."
                )
            elif chosen is None:
                # Path 3: nothing optimized → copy baseline.  Ensures a
                # submission-safe DCP regardless of upstream state.
                try:
                    optimizer._path_guard_check(
                        out, context="emergency_baseline_copy",
                    )
                except Exception:
                    pass
                _atomic_copy(str(args.input_dcp), str(out))
                chosen = "baseline"
                print(
                    f"[LIFECYCLE] EMERGENCY_BASELINE_COPY ({reason}): "
                    f"copied {args.input_dcp.name} → {out.name} for submission safety."
                )
                # Publish shipped identity (see Path 1).
                try:
                    _ident = _artifact_identity(out)
                    if _ident is not None:
                        optimizer._shipped_artifact = _ident
                        _write_shipped_manifest(out, _ident)
                except Exception:
                    pass
        except Exception as e:
            chosen = "none"
            print(f"[LIFECYCLE] HARD_FAIL — emergency baseline copy failed: {e}")

        # Persist termination metadata so post-hoc analysis (and the
        # wrapper scripts) can tell at a glance what shipped and why.
        # Best-effort — never raise from this path.
        try:
            elapsed = None
            best_wns = None
            start_time = getattr(optimizer, "start_time", None)
            if start_time is not None:
                elapsed = time.time() - start_time
            if hasattr(optimizer, "best_wns") and optimizer.best_wns != float("-inf"):
                best_wns = optimizer.best_wns
            meta = {
                "reason": reason,
                "finalize_source": chosen,
                "elapsed_seconds": elapsed,
                "best_wns_ns": best_wns,
                "initial_wns_ns": getattr(optimizer, "initial_wns", None),
                "max_wall_seconds": getattr(optimizer, "max_wall_seconds", None),
                "final_status": getattr(optimizer, "final_status", None),
                "output_dcp": str(args.output_dcp),
                "output_dcp_size_bytes": (
                    Path(args.output_dcp).stat().st_size
                    if Path(args.output_dcp).exists() else 0
                ),
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                # Identity of what this run believes it shipped (size+md5),
                # so anyone can md5 the scored file afterwards and tell this
                # artifact apart from an external write.
                "shipped_artifact": getattr(
                    optimizer, "_shipped_artifact", None),
                "budget_killed": getattr(optimizer, "_budget_killed", False),
                "strategies_skipped_budget": list(getattr(
                    optimizer, "_strategies_skipped_budget", [])),
            }
            (run_dir / "lifecycle_metadata.json").write_text(
                json.dumps(meta, indent=2)
            )
        except Exception as e:
            print(f"[LIFECYCLE] metadata write failed: {e}")

        # ---- Last-resort token_usage.json ---------------------------------
        # The multi-restart wrapper counts an attempt as usable only if it can
        # read a non-None Fmax, and Fmax has exactly one source: this file.
        # Cost and status both have fallbacks; Fmax does not.  So a missing
        # token_usage.json makes the wrapper declare that there is no usable
        # attempt output and fall back, discarding a completed, valid artifact.
        #
        # The file is written as the last statement of the summary printer, so
        # anything that stops the printer early leaves it absent — the printer
        # raising, or an exit path that never calls it at all.  This runs on
        # every exit path, writes only when nothing else did, and cannot change
        # which checkpoint ships.
        #
        # Not written on the baseline path.  The reported Fmax derives from the
        # tracked best, not from whatever is actually at the output path, and
        # on the baseline path the shipped artifact is the unmodified input
        # while the tracked best can be high — so writing here would hand
        # attempt selection a systematically optimistic number.  That matters
        # because selection ranks by Fmax first and uses status only as a
        # tie-breaker, so an optimistic baseline attempt could outrank an
        # honest optimized one.  Restricting this to the paths where the
        # artifact corresponds to the tracked best keeps the recovery without
        # widening that gap.
        try:
            _tu = run_dir / "token_usage.json"
            if not _tu.exists() and chosen in ("best_valid",
                                               "optimizer_output"):
                optimizer.save_token_usage_report(_tu)
                print(f"[LIFECYCLE] token_usage.json was missing — wrote it "
                      f"from the emergency path so the wrapper can rank this "
                      f"attempt ({reason}, source={chosen}).")
            elif not _tu.exists():
                print(f"[LIFECYCLE] token_usage.json missing and source="
                      f"{chosen} — NOT writing it: best_wns describes the "
                      f"tracked best, not the baseline artifact that shipped.")
        except Exception as e:
            print(f"[LIFECYCLE] last-resort token_usage write failed: {e}")

    def _signal_finalize(signum, frame):  # noqa: ARG001
        """Handle SIGTERM or SIGHUP by finalizing artifacts and exiting
        immediately.

        The handler may run between Python bytecodes or after a blocked tool
        socket returns with interruption. It uses only direct file copying and
        `os._exit`; it must not enter asyncio because the event loop may be
        wedged.
        """
        try:
            # `import signal as _signal` below; the bare name `signal` never
            # exists here, so this line always died on a NameError and every
            # signal printed as "signal_<num>" (the T5 smoke symptom).
            name = _signal.Signals(signum).name
        except Exception:
            name = f"signal_{signum}"
        print(f"\n[LIFECYCLE] Received {name}; running emergency finalize.")
        try:
            # Fallback name is already "signal_<num>" — don't double-prefix
            # (T5 smoke observed "signal_signal_15"); named signals keep the
            # historical "signal_SIGTERM" reason shape.
            _emergency_baseline_copy(
                name if name.startswith("signal_") else f"signal_{name}")
        finally:
            # 128 + signal number is the conventional exit code for signal
            # termination; lets wrappers detect why the process died.
            os._exit(128 + signum)

    # Install SIGTERM (contest validator hard-kill) and SIGHUP (shell
    # disconnect) handlers BEFORE optimizing starts.  SIGINT is left to
    # the existing KeyboardInterrupt path, for compatibility with the
    # KeyboardInterrupt test fixtures.
    import signal as _signal
    try:
        _signal.signal(_signal.SIGTERM, _signal_finalize)
    except Exception as e:
        logger.warning(f"Could not install SIGTERM handler: {e}")
    if hasattr(_signal, "SIGHUP"):
        try:
            _signal.signal(_signal.SIGHUP, _signal_finalize)
        except Exception as e:
            logger.warning(f"Could not install SIGHUP handler: {e}")

    try:
        await optimizer.start_servers()
        success = await optimizer.optimize(args.input_dcp, args.output_dcp)

        if success:
            print("\n✓ Optimization completed successfully")
            print(f"\nOutput files:")
            print(f"  Optimized DCP: {args.output_dcp}")
            print(f"  Run directory: {run_dir}")
            _emergency_baseline_copy("normal_exit_success")
            sys.exit(0)
        else:
            print("\n✗ Optimization did not complete successfully")
            # Optimizer may have already produced an output_dcp (the
            # Phase-1 safety net does this); the new emergency_baseline_copy
            # prefers best_valid.dcp when present, falls back to baseline.
            _emergency_baseline_copy("optimize_returned_false")
            print(f"\nRun directory: {run_dir}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        _emergency_baseline_copy("keyboard_interrupt")
        print(f"Run directory: {run_dir}")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        _emergency_baseline_copy(f"fatal_{type(e).__name__}")
        print(f"Run directory: {run_dir}")
        sys.exit(1)
    finally:
        await optimizer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
