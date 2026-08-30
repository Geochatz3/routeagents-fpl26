"""Structured ToolError taxonomy — log-only first phase.

Converts the ad-hoc text/JSON error envelopes that call_tool and
finalize emit into a closed-set typed ToolError shape.  Used by the
decision tracer's tool_error_code field and lifecycle_log events.

Phase 1 (this commit): CLASSIFICATION + LOGGING only.  Recovery
behavior is unchanged.  Future phases can use the policy table to
drive retry/rollback/escalation decisions.

Codes:
  VIVADO_TIMEOUT, BUDGET_SKIP, TIMEOUT_BUDGET,
  INVALID_DCP, WRONG_CLOCK, ROUTE_FAILED, PLACE_FAILED,
  REGRESSION_DETECTED, STALE_MIRROR,
  TCL_SYNTAX_ERROR, TCL_RUNTIME_ERROR, MCP_UNAVAILABLE,
  PARSER_FALSE_POSITIVE, VALIDATOR_MISMATCH, MISSING_ARTIFACT,
  RQA_PARSE_FAILED, RQS_GENERATOR_REFUSED, UNKNOWN_TOOL_ERROR.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Closed set — used for assertion at write time + recipient validation.
TOOL_ERROR_CODES = frozenset({
    "VIVADO_TIMEOUT",
    "BUDGET_SKIP",
    "TIMEOUT_BUDGET",
    "INVALID_DCP",
    "WRONG_CLOCK",
    "ROUTE_FAILED",
    "PLACE_FAILED",
    "REGRESSION_DETECTED",
    "STALE_MIRROR",
    "TCL_SYNTAX_ERROR",
    "TCL_RUNTIME_ERROR",
    "MCP_UNAVAILABLE",
    "PARSER_FALSE_POSITIVE",
    "VALIDATOR_MISMATCH",
    "MISSING_ARTIFACT",
    "RQA_PARSE_FAILED",
    "RQS_GENERATOR_REFUSED",
    "UNKNOWN_TOOL_ERROR",
})

# Severity levels (lowest → highest).
SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"
SEVERITY_CRITICAL = "critical"


@dataclass(frozen=True)
class ToolError:
    """Structured tool-error envelope.  Frozen for hashability +
    safety: nothing should mutate a classified error post-hoc."""
    code: str
    severity: str
    retryable: bool
    rollback: str           # "none" | "mirror" | "baseline"
    reason: str
    raw_payload: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "code": self.code,
            "severity": self.severity,
            "retryable": self.retryable,
            "rollback": self.rollback,
            "reason": self.reason,
        }
        if self.raw_payload is not None:
            d["raw_payload"] = self.raw_payload[:500]  # cap payload size
        if self.extra:
            d["extra"] = dict(self.extra)
        return d


# Policy metadata table — phase 1 is advisory only.  Future code can
# pivot recovery on these flags without redefining the codes.
@dataclass(frozen=True)
class ErrorPolicy:
    severity: str
    retryable: bool
    rollback: str
    reviewer_escalation: bool
    baseline_fallback: bool
    negative_memory: bool


_POLICY: Dict[str, ErrorPolicy] = {
    "VIVADO_TIMEOUT": ErrorPolicy(
        severity=SEVERITY_MEDIUM, retryable=True, rollback="mirror",
        reviewer_escalation=False, baseline_fallback=False,
        negative_memory=True,
    ),
    "BUDGET_SKIP": ErrorPolicy(
        severity=SEVERITY_LOW, retryable=False, rollback="none",
        reviewer_escalation=False, baseline_fallback=False,
        negative_memory=False,
    ),
    "TIMEOUT_BUDGET": ErrorPolicy(
        severity=SEVERITY_MEDIUM, retryable=False, rollback="none",
        reviewer_escalation=False, baseline_fallback=False,
        negative_memory=False,
    ),
    "INVALID_DCP": ErrorPolicy(
        severity=SEVERITY_HIGH, retryable=False, rollback="baseline",
        reviewer_escalation=True, baseline_fallback=True,
        negative_memory=True,
    ),
    "WRONG_CLOCK": ErrorPolicy(
        severity=SEVERITY_HIGH, retryable=False, rollback="baseline",
        reviewer_escalation=True, baseline_fallback=True,
        negative_memory=False,
    ),
    "ROUTE_FAILED": ErrorPolicy(
        severity=SEVERITY_HIGH, retryable=False, rollback="mirror",
        reviewer_escalation=True, baseline_fallback=False,
        negative_memory=True,
    ),
    "PLACE_FAILED": ErrorPolicy(
        severity=SEVERITY_MEDIUM, retryable=True, rollback="mirror",
        reviewer_escalation=False, baseline_fallback=False,
        negative_memory=True,
    ),
    "REGRESSION_DETECTED": ErrorPolicy(
        severity=SEVERITY_LOW, retryable=False, rollback="mirror",
        reviewer_escalation=False, baseline_fallback=False,
        negative_memory=True,
    ),
    "STALE_MIRROR": ErrorPolicy(
        severity=SEVERITY_MEDIUM, retryable=True, rollback="none",
        reviewer_escalation=False, baseline_fallback=False,
        negative_memory=False,
    ),
    "TCL_SYNTAX_ERROR": ErrorPolicy(
        severity=SEVERITY_LOW, retryable=True, rollback="none",
        reviewer_escalation=False, baseline_fallback=False,
        negative_memory=True,
    ),
    "TCL_RUNTIME_ERROR": ErrorPolicy(
        severity=SEVERITY_MEDIUM, retryable=False, rollback="none",
        reviewer_escalation=False, baseline_fallback=False,
        negative_memory=True,
    ),
    "MCP_UNAVAILABLE": ErrorPolicy(
        severity=SEVERITY_HIGH, retryable=True, rollback="mirror",
        reviewer_escalation=False, baseline_fallback=True,
        negative_memory=False,
    ),
    "PARSER_FALSE_POSITIVE": ErrorPolicy(
        severity=SEVERITY_LOW, retryable=False, rollback="none",
        reviewer_escalation=False, baseline_fallback=False,
        negative_memory=False,
    ),
    "VALIDATOR_MISMATCH": ErrorPolicy(
        severity=SEVERITY_CRITICAL, retryable=False, rollback="baseline",
        reviewer_escalation=True, baseline_fallback=True,
        negative_memory=True,
    ),
    "MISSING_ARTIFACT": ErrorPolicy(
        severity=SEVERITY_HIGH, retryable=True, rollback="mirror",
        reviewer_escalation=False, baseline_fallback=True,
        negative_memory=True,
    ),
    "RQA_PARSE_FAILED": ErrorPolicy(
        severity=SEVERITY_LOW, retryable=False, rollback="none",
        reviewer_escalation=False, baseline_fallback=False,
        negative_memory=False,
    ),
    "RQS_GENERATOR_REFUSED": ErrorPolicy(
        severity=SEVERITY_LOW, retryable=False, rollback="none",
        reviewer_escalation=False, baseline_fallback=False,
        negative_memory=False,
    ),
    "UNKNOWN_TOOL_ERROR": ErrorPolicy(
        severity=SEVERITY_HIGH, retryable=False, rollback="none",
        reviewer_escalation=True, baseline_fallback=False,
        negative_memory=True,
    ),
}


def get_policy(code: str) -> ErrorPolicy:
    """Look up policy metadata for a code.  Falls back to UNKNOWN."""
    return _POLICY.get(code, _POLICY["UNKNOWN_TOOL_ERROR"])


def _new_error(code: str, reason: str, *,
               raw: Optional[str] = None,
               extra: Optional[Dict[str, Any]] = None) -> ToolError:
    pol = get_policy(code)
    return ToolError(
        code=code,
        severity=pol.severity,
        retryable=pol.retryable,
        rollback=pol.rollback,
        reason=reason,
        raw_payload=raw,
        extra=extra or {},
    )


# Pre-compiled patterns for the most common shapes the optimizer sees.
# Ordered roughly by specificity; first match wins.
_BUDGET_SKIP_PAT = re.compile(r"tool_skipped_budget", re.IGNORECASE)
_DEADLINE_PASSED_PAT = re.compile(r"deadline_passed", re.IGNORECASE)
_TIMED_OUT_BUDGET_PAT = re.compile(r"tool_timed_out_budget", re.IGNORECASE)
_VIVADO_TIMEOUT_PAT = re.compile(
    r"timed out|timeout|TimeoutError|asyncio.*TimeoutError", re.IGNORECASE,
)
_TCL_SYNTAX_PAT = re.compile(
    r"(?:invalid command name|wrong # args|extra characters after close-quote|"
    r"unmatched open (?:brace|bracket|quote)|Tcl[- ]?parse|syntax error)",
    re.IGNORECASE,
)
_TCL_RUNTIME_PAT = re.compile(
    r"ERROR:\s*\[(?:Common|Vivado|Synth|Place|Route|Phys[_ ]?Opt)\s+\d+-\d+\]",
)
_ROUTE_FAILED_PAT = re.compile(
    r"(?:route_design failed|unrouted nets|routing failed|cannot route|"
    r"\d+ unrouted)", re.IGNORECASE,
)
_PLACE_FAILED_PAT = re.compile(
    r"(?:place_design failed|placer failed|cannot place|"
    r"placement error)", re.IGNORECASE,
)
_INVALID_DCP_PAT = re.compile(
    r"(?:DCP missing|zero-byte|invalid checkpoint|cannot open checkpoint|"
    r"corrupt(?:ed)? DCP|missing.*\.dcp)", re.IGNORECASE,
)
_WRONG_CLOCK_PAT = re.compile(
    r"(?:CLOCK_NAME=NOT_FOUND|clk_fpl26contest.*not found|"
    r"contest clock missing|no paths on contest clock)", re.IGNORECASE,
)
_STALE_MIRROR_PAT = re.compile(
    r"stale[- ]mirror|mirror.*STALE|stale_mirror_detected", re.IGNORECASE,
)
_REGRESSION_PAT = re.compile(
    r"regression_detected|WNS regressed|regress(?:ed|ion)", re.IGNORECASE,
)
_MCP_UNAVAIL_PAT = re.compile(
    r"(?:MCP\s+unavailable|session.*(?:closed|dropped)|"
    r"ClientSession.*closed|JSONRPC.*ValidationError)", re.IGNORECASE,
)
_VALIDATOR_MISMATCH_PAT = re.compile(
    r"FAIL_DELTA_|claim-vs-validated drift|validator mismatch",
    re.IGNORECASE,
)
# MISSING_ARTIFACT — must be error/warning-tagged or a specific
# named code.  The prior `file.*missing` substring matched benign
# Vivado info-level chatter (an observed smoke-test
# false-positive).  Tightened to require either:
#   - an explicit error/warning tag preceding the phrase, OR
#   - a specific dispatcher / output-DCP envelope phrase
_MISSING_ARTIFACT_PAT = re.compile(
    r"(?:"
    r"MISSING_DCP|MISSING_ARTIFACT|"
    r"(?:ERROR|WARNING):.*file.*missing|"
    r"output[_ ]dcp\s+missing|"
    r"returned but.*missing/empty|"
    r"expected\s+file.*(?:missing|not found|does not exist)|"
    r"file\s+(?:not found|does not exist)\s+on disk"
    r")",
    re.IGNORECASE,
)
_RQA_PARSE_PAT = re.compile(r"RQA parse failed|RQA_PARSE_FAILED",
                              re.IGNORECASE)
_RQS_REFUSED_PAT = re.compile(
    r"ML Strateg.*Not Available|RQS.*refused|strategy.*unavailable",
    re.IGNORECASE,
)


def classify_tool_error(
    payload: Union[str, dict, None],
    *,
    context: Optional[str] = None,
) -> Optional[ToolError]:
    """Classify an error-like tool, finalization, or validation payload.

    Returns a typed `ToolError` for recognized failures and `None` for clear
    success or a null payload. Dictionaries are inspected for structured error
    fields, while strings are inspected as tool or dispatcher output. The
    optional context tag is included in generated reasons.
    """
    if payload is None:
        return None
    if isinstance(payload, dict):
        return _classify_dict(payload, context=context)
    if isinstance(payload, str):
        return _classify_text(payload, context=context)
    # Anything else: try string conversion.
    try:
        return _classify_text(str(payload), context=context)
    except Exception:
        return None


def _classify_dict(d: dict, *, context: Optional[str]) -> Optional[ToolError]:
    # Explicit error code in the envelope
    err = d.get("error")
    if not err and d.get("status") not in (None, "OK", "ok", "success"):
        err = d.get("status")
    if err is None:
        return None
    err_str = str(err)
    # Common dispatcher envelopes have a "reason" or "error" field that
    # is itself one of this module's codes.
    err_upper = err_str.upper()
    if err_upper in TOOL_ERROR_CODES:
        return _new_error(err_upper,
                          reason=str(d.get("reason") or err_str),
                          raw=json.dumps(d)[:500])
    # Reuse text classifier on the stringified payload
    blob = json.dumps(d)
    return _classify_text(blob, context=context)


def _classify_text(text: str, *, context: Optional[str]) -> Optional[ToolError]:
    if not text:
        return None
    if _BUDGET_SKIP_PAT.search(text):
        return _new_error("BUDGET_SKIP", "dispatcher refused tool call "
                          "(estimated runtime exceeds remaining budget)",
                          raw=text)
    if _TIMED_OUT_BUDGET_PAT.search(text) or (
        _DEADLINE_PASSED_PAT.search(text) and "tool_skipped" not in text.lower()
    ):
        return _new_error("TIMEOUT_BUDGET",
                          "tool call cancelled because budget deadline passed",
                          raw=text)
    # Validator drift first — it's the most specific shape and
    # otherwise might match VIVADO_TIMEOUT / TCL_RUNTIME wildcards.
    if _VALIDATOR_MISMATCH_PAT.search(text):
        return _new_error("VALIDATOR_MISMATCH",
                          "validated_dfmax differs from claimed_dfmax by > 1 MHz",
                          raw=text)
    if _STALE_MIRROR_PAT.search(text):
        return _new_error("STALE_MIRROR",
                          "best_valid mirror is stale relative to current best_wns",
                          raw=text)
    if _REGRESSION_PAT.search(text):
        return _new_error("REGRESSION_DETECTED",
                          "WNS regressed below initial baseline",
                          raw=text)
    if _ROUTE_FAILED_PAT.search(text):
        return _new_error("ROUTE_FAILED", "route_design failed or left unrouted nets",
                          raw=text)
    if _PLACE_FAILED_PAT.search(text):
        return _new_error("PLACE_FAILED", "place_design failed",
                          raw=text)
    if _INVALID_DCP_PAT.search(text):
        return _new_error("INVALID_DCP", "DCP is missing, corrupt, or zero-byte",
                          raw=text)
    if _WRONG_CLOCK_PAT.search(text):
        return _new_error("WRONG_CLOCK", "contest clock not present or unparseable",
                          raw=text)
    if _MISSING_ARTIFACT_PAT.search(text):
        return _new_error("MISSING_ARTIFACT", "expected file not on disk after operation",
                          raw=text)
    if _MCP_UNAVAIL_PAT.search(text):
        return _new_error("MCP_UNAVAILABLE", "MCP session dropped or unparseable",
                          raw=text)
    if _TCL_SYNTAX_PAT.search(text):
        return _new_error("TCL_SYNTAX_ERROR", "Tcl parse / syntax error",
                          raw=text)
    if _TCL_RUNTIME_PAT.search(text):
        return _new_error("TCL_RUNTIME_ERROR",
                          "Vivado emitted ERROR [Subsys NN-NN]",
                          raw=text)
    if _RQA_PARSE_PAT.search(text):
        return _new_error("RQA_PARSE_FAILED", "RQA output unparseable", raw=text)
    if _RQS_REFUSED_PAT.search(text):
        return _new_error("RQS_GENERATOR_REFUSED",
                          "RQS / ML strategy generator refused", raw=text)
    if _VIVADO_TIMEOUT_PAT.search(text):
        return _new_error("VIVADO_TIMEOUT", "Vivado tool call timed out",
                          raw=text)
    # Generic "ERROR" mention with no recognized specific shape.
    if re.search(r"\bERROR\b", text):
        return _new_error("UNKNOWN_TOOL_ERROR",
                          "unrecognized error pattern in tool payload",
                          raw=text)
    return None


def is_error_payload(payload: Union[str, dict, None]) -> bool:
    """Convenience boolean wrapper for callers that don't need the
    typed object."""
    return classify_tool_error(payload) is not None
