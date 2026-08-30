"""STAGE 4 — LLM LOOP (prompt assembly and the Tcl safety check)

The tool-call boundary.

Composing the model's first user message, loading the system prompt,
classifying a tool result as an error, and deciding whether a Tcl payload
would destroy routed state -- the last of which gates every raw run_tcl call.

Named for the stage it implements: the slide, the video and the paper all
call it by this name, so a reader can move between them without a glossary.
"""
from __future__ import annotations

import json
import os
import re
import time
import logging
from pathlib import Path
from typing import Optional

# Log through the orchestrator's logger, not this module's own: the extracted
# code still belongs to the same run, and a module-named logger would change
# every line it emits in the run log.
logger = logging.getLogger("dcp_optimizer")


LINEAGE_SOURCES = {
    "eager_mirror",
    "piggyback",
    "backstop",
    "stale_mirror_file",
    "baseline",
    "no_improvement_baseline",
    # The output passed structural validation but has no recorded lineage from
    # a recognized publication event. Treat it as valid but provenance-unconfirmed.
    "external_write_validated",
    # hard_fail_no_ship: finalize did not produce a usable output DCP
    # (no baseline available, no mirror available).  Lineage is recorded
    # so downstream tooling can detect the failure mode.
    "hard_fail_no_ship",
}


_PLACE_STMT_RE = re.compile(r"\bplace_design\b")


_ROUTE_STMT_RE = re.compile(r"\broute_design\b")


def _make_lineage_entry(
    source: str,
    *,
    token: Optional[int] = None,
    wns: Optional[float] = None,
    fmax: Optional[float] = None,
    iteration: Optional[int] = None,
    tool_name: Optional[str] = None,
    dcp_path: Optional[str] = None,
    dcp_size: Optional[int] = None,
    ts: Optional[float] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Compact lineage entry for best-valid / ship-DCP provenance.

    Source must be one of LINEAGE_SOURCES.  ts is epoch seconds (float)
    — easy to JSON-serialize and stable across timezones.  extra is for
    branch-specific debugging fields without polluting the canonical shape.
    """
    if source not in LINEAGE_SOURCES:
        raise ValueError(f"unknown lineage source: {source!r}")
    entry = {
        "source": source,
        "token": token,
        "wns": wns,
        "fmax": fmax,
        "iteration": iteration,
        "tool_name": tool_name,
        "dcp_path": dcp_path,
        "dcp_size": dcp_size,
        "ts": ts if ts is not None else time.time(),
    }
    if extra:
        entry["extra"] = dict(extra)
    return entry


def compose_iter1_user_message(input_dcp: Path, output_dcp: Path,
                               temp_dir, initial_analysis: str,
                               rag_section: str, recipe_hint: str) -> str:
    """Compose the initial user message while omitting empty sections.

    Recipe-pass state must never appear in the message. Initial analysis is
    captured before that pass and remains immutable, while strategy-memory
    content depends only on design features and the recipe hint belongs to a
    separate gate.

    The composition must remain pure and environment-independent; reading
    runtime pipeline state would create an unintended exposure channel.
    """
    user_sections = [
        "Optimize this FPGA design for timing.",
        "",
        "PATHS:",
        f"- Input DCP: {input_dcp.resolve()}",
        f"- Output DCP (save final result here): {output_dcp.resolve()}",
        f"- Run directory (for intermediate files): {temp_dir}",
        "",
        "CURRENT STATE:",
        "- Vivado has the input design ALREADY OPEN and analyzed",
        "- RapidWright has the input design ALREADY LOADED (from initial analysis)",
        "",
        "INITIAL ANALYSIS RESULTS:",
        initial_analysis,
    ]
    if rag_section:
        user_sections.extend(["", rag_section])
    if recipe_hint:
        user_sections.append(recipe_hint)
    user_sections.extend([
        "",
        "Proceed with optimization strategy based on the analysis above. "
        "Do NOT reload the design in either Vivado or RapidWright - both "
        "already have it loaded.",
    ])
    return "\n".join(user_sections)


def load_system_prompt() -> str:
    """Load the system prompt.  prompts/system_prompt_scored.txt unless
    FPL26_PROMPT_V2=1 selects prompts/system_prompt_v2_experimental.txt.

    Why a second file instead of editing the first: the system prompt is
    a ship-surface file, so prompt changes ride the same default-OFF flag
    discipline as every other behavioral change — editing it in place
    would change the build for every design at once, with no way to
    measure or revert the change independently.

    Default off: with the env unset this returns the scored prompt byte
    for byte.  The V2 variant was never active in the scored run (no
    ship target sets FPL26_PROMPT_V2) — see prompts/README.md.

    Falls back to the scored prompt with a loud log line if V2 is
    requested but absent, because a missing prompt must not be a silent
    behavior change.
    """
    # The prompts live at the repository root, next to the orchestrator.
    # Anchored on the package's parent rather than on __file__ so the lookup
    # keeps pointing at that directory wherever this module is moved to.
    script_dir = Path(__file__).resolve().parent.parent
    use_v2 = os.environ.get(
        "FPL26_PROMPT_V2", "0").strip().lower() in ("1", "true", "on", "yes")
    prompt_file = script_dir / "prompts" / "system_prompt_scored.txt"
    if use_v2:
        v2 = script_dir / "prompts" / "system_prompt_v2_experimental.txt"
        if v2.exists():
            prompt_file = v2
            logger.info("[prompt] FPL26_PROMPT_V2=1 -> "
                        "prompts/system_prompt_v2_experimental.txt")
        else:
            logger.warning(
                "[prompt] FPL26_PROMPT_V2=1 but "
                "prompts/system_prompt_v2_experimental.txt is MISSING; falling "
                "back to the scored prompt — this arm is a broken probe, "
                "do not score it as a null.")

    try:
        with open(prompt_file, 'r') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"System prompt file not found: {prompt_file}")
        raise
    except Exception as e:
        logger.error(f"Failed to load system prompt: {e}")
        raise


def _looks_like_tool_error(result_text: str) -> bool:
    """Heuristic: did the underlying MCP tool report a failure?

    Two failure shapes need to be detected:
      1. JSON envelope: call_tool catches exceptions and returns
         json.dumps({"error": ...}) on failure.
      2. Vivado TCL error text: Vivado returns free-form output that
         includes "TCL ERROR:" or "User Exception:" or "ERROR: [Common 17-69]"
         when a Tcl command fails.  These come back as plain text, not JSON.

    The Vivado patterns matter because path-translation issues (WSL2 /tmp
    vs Windows C:\tmp) cause Vivado to "succeed" the tool call at the MCP
    layer but emit a TCL ERROR inside the response.  Without this check,
    the recipe-as-tool implementations would silently accept a failed
    open_checkpoint / route_design and continue.

    Returns True for either error shape; False for clean success.
    """
    if not isinstance(result_text, str) or not result_text:
        return False
    s = result_text.strip()

    # Shape 1: JSON error envelope
    if s.startswith("{") and s.endswith("}"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict) and "error" in parsed:
                return True
        except json.JSONDecodeError:
            pass  # not JSON; fall through to text checks

    # Shape 2: Vivado/Tcl error strings.  These markers appear in the
    # free-form response body when a Tcl command fails.
    error_markers = (
        "TCL ERROR:",
        "User Exception:",
        "ERROR: [Common 17-",  # Vivado Tcl errors (17-53, 17-69, etc.)
        "ERROR: [Vivado 12-",  # design open failures
        "ERROR: [Vivado 4-",   # general Vivado errors
    )
    for marker in error_markers:
        if marker in s:
            return True

    # The MCP server reports dispatcher failures as plain text beginning with
    # "Error: "; without this check, timeouts and process failures appear successful.
    # Match only the prefix because valid report text may contain the same token.
    if s.startswith("Error: "):
        return True
    return False


def _tool_cmd_text(tool_name: str, arguments: Optional[dict]) -> str:
    """Lower-cased Tcl payload of a vivado_run_tcl call ('' otherwise)."""
    if tool_name != "vivado_run_tcl" or not isinstance(arguments, dict):
        return ""
    cmd = arguments.get("command") or arguments.get("tcl_command") or ""
    return str(cmd).lower()


def _tcl_statements(cmd_low: str) -> list:
    """Split a lower-cased Tcl payload into executable statements.

    Tcl separates top-level commands with ';' OR newlines, so both are
    split points; comment statements (leading '#') are dropped so a
    commented-out route_design is never mistaken for an executed one.
    (An earlier ';'-only split misread newline-separated scripts and
    comments.)
    """
    stmts = []
    for stmt in re.split(r"[;\n]", cmd_low):
        stmt = stmt.strip()
        if not stmt or stmt.startswith("#"):
            continue
        stmts.append(stmt)
    return stmts


def is_routed_state_destroying(tool_name: str, arguments: Optional[dict],
                               currently_routed: bool) -> bool:
    """Classify operations that destroy or rebuild the routed state.

    The classified cases are an unroute command, placement while currently
    routed, and a full nonincremental route dispatch while currently unrouted.
    Incremental or state-preserving reroutes on an already routed design are
    not classified as destructive and must not be gated.

    Unplacing an already unrouted design is outside this classifier's scope.
    Routed status is supplied explicitly so the policy remains pure and
    testable.
    """
    cmd_low = _tool_cmd_text(tool_name, arguments)
    stmts = _tcl_statements(cmd_low)
    # Dedicated (non-run_tcl) dispatches carry no Tcl payload — model
    # them as a single synthetic statement so the walk below applies.
    if tool_name == "vivado_route_design" and not any(
            _ROUTE_STMT_RE.search(s) for s in stmts):
        stmts.append("route_design"
                     + (" -unroute" if "-unroute" in cmd_low else ""))
    if tool_name == "vivado_place_design" and not any(
            _PLACE_STMT_RE.search(s) for s in stmts):
        stmts.append("place_design")
    # Walk statements in execution order, simulating routedness — a
    # compound script is destroying if ANY step hits a destroy case.
    routed = currently_routed
    for stmt in stmts:
        if _PLACE_STMT_RE.search(stmt):
            # Placement on a routed design destroys routed state.
            if routed:
                return True
            routed = False
        if _ROUTE_STMT_RE.search(stmt):
            # Case 1: -unroute commits the design to the destroy path
            # even when the same script re-routes afterwards — the full
            # re-route must fit.
            if "-unroute" in stmt:
                return True
            # Case 3: full re-route while unrouted (the mandatory
            # rebuild).  An explicitly incremental/preserving route is
            # given through.
            if (not routed and "-preserve" not in stmt
                    and "-incremental" not in stmt):
                return True
            routed = True
    return False


def _routed_state_transition(tool_name: str,
                             arguments: Optional[dict]) -> Optional[bool]:
    """Post-SUCCESS routed-state transition for a tool call.

    Returns the new routed state (True/False) or None when the op does
    not change routedness.  Only called on ops that returned OK — a
    failed/skipped op leaves the harness state untouched.

      route_design (non-unroute)  -> True   (design ends routed; also
                                             covers place+route scripts)
      route_design -unroute       -> False
      place_design                -> False  (place unroutes)
      open_checkpoint             -> True   (contest input DCPs and
                                             best_valid mirrors are
                                             placed+routed / routed-gated)
    """
    cmd_low = _tool_cmd_text(tool_name, arguments)
    # Evaluate compound Tcl statements in execution order and retain the final
    # routing transition. Ignore command names found only in comments or filenames.
    state: Optional[bool] = None
    for stmt in _tcl_statements(cmd_low):
        if _PLACE_STMT_RE.search(stmt):
            state = False
        if _ROUTE_STMT_RE.search(stmt):
            state = False if "-unroute" in stmt else True
    if state is not None:
        return state
    # Dedicated (non-run_tcl) dispatches carry no Tcl payload.
    if tool_name == "vivado_route_design":
        return False if "-unroute" in cmd_low else True
    if tool_name == "vivado_place_design":
        return False
    if tool_name == "vivado_open_checkpoint":
        return True
    return None


def convert_mcp_tool_to_openai(tool, server_prefix: str) -> dict:
    """Convert MCP tool definition to OpenAI-compatible format with server prefix."""
    schema = tool.inputSchema or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": f"{server_prefix}_{tool.name}",
            "description": tool.description or "",
            "parameters": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", [])
            }
        }
    }
