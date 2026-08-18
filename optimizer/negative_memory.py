"""Negative-memory persistence v0 — derive only HIGH-confidence
"don't do this again" records from `decisions.jsonl`.

Design contract (per Session 5 brief):

- Only emit a NegativeMemory record for events that have either
  (a) a `tool_error_code` whose policy declares
  `negative_memory == True` AND a tool-call action_label that
  represents a real failure (not a false-positive classifier match
  on a successful call), OR
  (b) a PathGuard violation (`path_guard_allowed=False`).
- Explicitly EXCLUDE: `BUDGET_SKIP`, `PARSER_FALSE_POSITIVE`,
  `RQA_PARSE_FAILED`, `RQS_GENERATOR_REFUSED`, `tool_call_warning`
  (the optimiser's signal that the classifier matched but the call
  actually succeeded), parser-warning records, and any code whose
  policy says `negative_memory == False`.
- Dedupe by `memory_id` (stable hash of run_id + record_index +
  error_code + tool_name).
- design_name is metadata only.  Future code must NOT use it to
  filter or retrieve memories.

This module is READ-ONLY over `decisions.jsonl`.  It produces an
append-only `negative_memory.jsonl` next to `episode_store.jsonl`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from optimizer.policy_memory import (
    _extract_start_features,
    _load_records,
    _parse_notes_kv,
    build_episode,
)
from optimizer.tool_errors import TOOL_ERROR_CODES, get_policy

logger = logging.getLogger(__name__)

NEG_MEM_SCHEMA_VERSION = 1


# Action labels that REPRESENT a real failure, not a classifier
# warning.  Anything outside this set is filtered out even when
# `tool_error_code` is set.
_FAILURE_LABELS = frozenset({
    "tool_call_classified_error",
    "finalize_end",       # validity_state may indicate failure mode
    "validator_failure",  # future label; tolerated if it appears
})

# Observed-effect names per code (compact taxonomy for downstream
# consumers).  Codes not listed default to "tool_error_generic".
_OBSERVED_EFFECT = {
    "VALIDATOR_MISMATCH": "validator_mismatch",
    "INVALID_DCP": "invalid_dcp",
    "REGRESSION_DETECTED": "regression",
    "ROUTE_FAILED": "route_failed",
    "PLACE_FAILED": "place_failed",
    "TCL_SYNTAX_ERROR": "tcl_syntax",
    "TCL_RUNTIME_ERROR": "tcl_runtime",
    "VIVADO_TIMEOUT": "timeout",
    "MISSING_ARTIFACT": "missing_artifact",
    "WRONG_CLOCK": "wrong_clock",
    "STALE_MIRROR": "stale_mirror",
    "MCP_UNAVAILABLE": "mcp_unavailable",
    "UNKNOWN_TOOL_ERROR": "tool_error_generic",
}

# Confidence per code.  Path-guard violations get "high" too (they
# are hard signals).
_CONFIDENCE = {
    "VALIDATOR_MISMATCH": "high",
    "INVALID_DCP": "high",
    "REGRESSION_DETECTED": "high",
    "ROUTE_FAILED": "high",
    "PLACE_FAILED": "medium",
    "TCL_SYNTAX_ERROR": "medium",
    "TCL_RUNTIME_ERROR": "medium",
    "VIVADO_TIMEOUT": "medium",
    "MISSING_ARTIFACT": "medium",
    "WRONG_CLOCK": "high",
    "STALE_MIRROR": "low",
    "MCP_UNAVAILABLE": "low",
    "UNKNOWN_TOOL_ERROR": "low",
}


def _memory_id(run_id: str, idx: int, code: str, tool: Optional[str]) -> str:
    blob = f"{run_id}|{idx}|{code}|{tool or ''}".encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def _is_real_failure(rec: Dict[str, Any]) -> bool:
    """Filter out false-positive classifier matches on successful calls.

    The optimiser tags `tool_call_warning` when the regex matched but
    the call actually returned a usable result.  Those records have a
    `tool_error_code` set but must NOT enter negative memory.
    """
    label = rec.get("action_label")
    if label is None:
        return False
    if label in _FAILURE_LABELS:
        return True
    # finalize_end is a failure only when validity_state implies one.
    if label == "finalize_end":
        return rec.get("validity_state") in (
            "INVALID", "INVALID_FALLBACK", "VALIDATOR_MISMATCH",
        )
    return False


# Final-status values that mean the run actually finished cleanly.
# A MISSING_ARTIFACT on a vivado/rapidwright `open_checkpoint` call
# preceded by one of these states is suspicious — if the file were
# really missing the run couldn't have reached finalize_end at all.
_HEALTHY_FINAL_STATES = frozenset({
    "VALID_OPTIMIZED",
    "VALID_OPTIMIZED_NO_EDIF",
    "VALID_FALLBACK_BASELINE",
})


def _eligible_tool_error(rec: Dict[str, Any],
                        *, final_status: Optional[str] = None) -> bool:
    """Is this record a tool-error event that policy says should be a
    negative memory?

    Extra rule (per session-5 brief): a MISSING_ARTIFACT on an
    `open_checkpoint` call is only a real failure when the run's
    `final_status` confirms it (i.e., the run didn't reach a healthy
    finalize).  This filters out a known pre-label-gate-fix
    contamination where the classifier matched a benign info-level
    Vivado line that mentioned a missing tmp file.
    """
    code = rec.get("tool_error_code")
    if not code or code not in TOOL_ERROR_CODES:
        return False
    if not get_policy(code).negative_memory:
        return False
    if not _is_real_failure(rec):
        return False
    if code == "MISSING_ARTIFACT":
        tool = rec.get("tool_name") or ""
        if tool.endswith("_open_checkpoint"):
            # Trust the run-level outcome over a classifier match if
            # the run finished healthy.
            if final_status in _HEALTHY_FINAL_STATES:
                return False
    return True


def _eligible_pathguard_violation(rec: Dict[str, Any]) -> bool:
    return (rec.get("phase") == "path_guard"
            and rec.get("path_guard_allowed") is False)


def _profile_snapshot(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Capture the per-run start features so future advisory code can
    associate negative memories with feature buckets without re-keying
    on design name."""
    sf = _extract_start_features(records)
    # Drop fields likely null in v0 to keep records compact.
    return {k: v for k, v in sf.items() if v is not None}


def build_negative_memories(decisions_jsonl_path: Path | str
                            ) -> List[Dict[str, Any]]:
    """Return all negative-memory records derivable from one
    decisions.jsonl.  Empty list when none match (legitimate clean
    run, or trace too short)."""
    jsonl = Path(decisions_jsonl_path)
    records = _load_records(jsonl)
    if not records:
        return []
    run_id = records[0].get("run_id") or jsonl.parent.name

    # Mode flags for metadata.
    rag_recs = [r for r in records if r.get("action_label") == "rag_retrieval"]
    rag_mode = rag_recs[-1].get("rag_mode") if rag_recs else None
    contest_mode = (rag_mode == "contest_mode")

    profile = _profile_snapshot(records)

    # Run-level outcome (used to gate suspect MISSING_ARTIFACT matches).
    final_recs = [r for r in records
                  if r.get("action_label") == "finalize_end"]
    final_status: Optional[str] = (final_recs[-1].get("validity_state")
                                   if final_recs else None)

    out: List[Dict[str, Any]] = []
    src_path = str(jsonl.resolve(strict=False))

    for idx, rec in enumerate(records):
        if _eligible_tool_error(rec, final_status=final_status):
            code = rec["tool_error_code"]
            pol = get_policy(code)
            mid = _memory_id(run_id, idx, code, rec.get("tool_name"))
            out.append({
                "schema_version": NEG_MEM_SCHEMA_VERSION,
                "memory_id": mid,
                "run_id": run_id,
                "timestamp": rec.get("timestamp"),
                "profile_features": profile,
                "action_label": rec.get("action_label"),
                "tool_name": rec.get("tool_name"),
                "tool_error_code": code,
                "severity": pol.severity,
                "observed_effect": _OBSERVED_EFFECT.get(code, "tool_error_generic"),
                "confidence": _CONFIDENCE.get(code, "low"),
                "evidence": {
                    "decision_record_index": idx,
                    "decisions_jsonl_path": src_path,
                },
                "design_name": rec.get("design"),
                "contest_mode": contest_mode,
            })
            continue
        if _eligible_pathguard_violation(rec):
            mid = _memory_id(run_id, idx, "PATHGUARD_VIOLATION",
                             rec.get("tool_name"))
            out.append({
                "schema_version": NEG_MEM_SCHEMA_VERSION,
                "memory_id": mid,
                "run_id": run_id,
                "timestamp": rec.get("timestamp"),
                "profile_features": profile,
                "action_label": rec.get("action_label"),
                "tool_name": rec.get("tool_name"),
                "tool_error_code": None,
                "severity": "critical",
                "observed_effect": "path_violation",
                "confidence": "high",
                "evidence": {
                    "decision_record_index": idx,
                    "decisions_jsonl_path": src_path,
                    "context_notes": rec.get("notes"),
                    "output_dcp_path": rec.get("output_dcp_path"),
                },
                "design_name": rec.get("design"),
                "contest_mode": contest_mode,
            })
    return out


# ---------------------------------------------------------------------------
# Store I/O — append-only with memory_id dedupe.

def load_negative_memories(store_path: Path | str) -> List[Dict[str, Any]]:
    p = Path(store_path)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(
                    f"negative_memory.load: skip {p}:{line_no} ({exc})"
                )
    return out


def _existing_ids(store_path: Path) -> set:
    return {m.get("memory_id") for m in load_negative_memories(store_path)
            if m.get("memory_id")}


def append_negative_memories(store_path: Path | str,
                             memories: List[Dict[str, Any]]) -> int:
    """Append distinct memories to the store; return number actually
    appended."""
    p = Path(store_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = _existing_ids(p)
    appended = 0
    try:
        with p.open("a", encoding="utf-8") as fh:
            for m in memories:
                mid = m.get("memory_id")
                if not mid or mid in existing:
                    continue
                fh.write(json.dumps(m, ensure_ascii=False))
                fh.write("\n")
                existing.add(mid)
                appended += 1
    except OSError as exc:
        logger.warning(f"negative_memory.append: write {p} failed: {exc}")
    return appended


def ingest_decisions(decisions_jsonl_path: Path | str,
                     store_path: Path | str) -> int:
    """End-to-end: build negative memories from one decisions.jsonl
    and append the new ones to `store_path`.  Returns the count
    actually appended (deduped)."""
    mems = build_negative_memories(decisions_jsonl_path)
    if not mems:
        return 0
    return append_negative_memories(store_path, mems)


DEFAULT_STORE_RELATIVE = Path("policy_memory") / "negative_memory.jsonl"


def default_store_path(repo_root: Optional[Path | str] = None) -> Path:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent
    env = os.environ.get("NEGATIVE_MEMORY_STORE")
    if env:
        return Path(env)
    return Path(repo_root) / DEFAULT_STORE_RELATIVE
