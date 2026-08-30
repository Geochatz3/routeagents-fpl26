"""Policy-memory episode store v0 — derive trustworthy episodes from
`runs/<run_id>/decisions.jsonl`.

Design contract:

- READ-ONLY over decisions.jsonl.  This module never touches the
  optimizer's live trace; it only summarises a finished run into one
  Episode dict and appends it to `policy_memory/episode_store.jsonl`.
- v0 schema, intentionally minimal.  Every field is optional; the
  builder tolerates partial / mid-run decision streams without
  raising.
- design_name is METADATA, not a retrieval key.  Callers must NOT
  use it to look up episodes; that re-introduces the hidden-design
  hard-rule violation that contest-mode just closed.
- Dedupe by `episode_id` (derived from run_id).  Re-ingesting the
  same run is a noop.
- No I/O cost on the hot path: the optimizer doesn't import this
  module; it's invoked offline by a backfill / nightly job.

Public API:
  - build_episode(decisions_jsonl_path) -> dict
  - ingest_decisions(decisions_jsonl_path, store_path) -> dict
        # appends episode to store_path if not already there;
        # returns the episode dict either way.
  - load_episodes(store_path) -> list[dict]
  - episode_id_for_run(run_id) -> str

Schema version bumps:
  v1: initial schema below.

Field documentation lives in `EPISODE_V0_FIELDS`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

EPISODE_SCHEMA_VERSION = 1

EPISODE_V0_FIELDS = (
    # identity
    "schema_version", "episode_id", "run_id", "ts_run",
    "git_commit", "model",
    # mode
    "rag_mode", "contest_mode", "rqa_mode",
    # metadata-only (NOT a retrieval key)
    "design_name", "design_name_masked",
    # start state
    "start_features",
    # retrieval metadata captured during the run
    "retrieval_metadata",
    # aggregated trace
    "action_trace_summary",
    # outcome
    "outcome",
    # trust + provenance
    "trust",
    "source",
)


def episode_id_for_run(run_id: str) -> str:
    """Stable 12-char hex id derived from run_id.  Same run_id always
    maps to the same episode_id."""
    if not run_id:
        run_id = "unknown_run"
    return hashlib.sha1(run_id.encode("utf-8")).hexdigest()[:12]


# Trace parsing helpers — all tolerant of missing fields.

# Best-effort regex pulls for the few values the optimizer dropped into
# free-form `notes` strings instead of dedicated columns.
_NOTES_KV_RE = re.compile(r"(\w+)=([^\s]+)")


def _parse_notes_kv(notes: Optional[str]) -> Dict[str, str]:
    """Parse `key=value` tokens out of a free-form notes string.

    Optimizer notes follow `key=value key=value` shape (see
    decision_tracer wiring sites in dcp_optimizer.py).  Returns {} on
    missing input.  Multi-word values are NOT supported; this is
    intentionally simple.
    """
    if not isinstance(notes, str) or not notes:
        return {}
    out: Dict[str, str] = {}
    for m in _NOTES_KV_RE.finditer(notes):
        k, v = m.group(1), m.group(2)
        # `negative_memory_count=40` etc.
        out[k] = v
    return out


def _maybe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _maybe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _maybe_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def _first_non_null(records: Iterable[Dict[str, Any]], key: str) -> Any:
    for r in records:
        v = r.get(key)
        if v is not None:
            return v
    return None


def _design_from_run_start(rec: Dict[str, Any]) -> Optional[str]:
    """Extracts the design slug from a run-start note's input checkpoint.

    The `_2025.1.dcp` suffix is removed when present.
    """
    notes = rec.get("notes")
    if not isinstance(notes, str):
        return None
    m = re.search(r"input=([^\s]+)", notes)
    if not m:
        return None
    fname = m.group(1)
    # Strip trailing `_2025.1.dcp` or `.dcp`
    fname = re.sub(r"\.dcp$", "", fname)
    fname = re.sub(r"_\d{4}\.\d+$", "", fname)
    return fname or None


# Episode builder

def _extract_retrieval_metadata(rag_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extracts retrieval metadata from a `rag_retrieval` decision record.

    The function parses retrieval mode, design-note and exact-name flags,
    retrieved episode IDs, and memory counts from the key-value notes string.
    It also reads the top-level `rag_mode` field.
    """
    out: Dict[str, Any] = {
        "retrieved_episode_ids": [],
        "retrieval_mode": None,
        "design_notes_injected": None,
        "exact_name_used": None,
        "negative_memory_count": None,
        "memory_records_considered": None,
    }
    if not rag_records:
        return out
    # Use the most recent rag_retrieval record (last-write-wins).
    rec = rag_records[-1]
    kv = _parse_notes_kv(rec.get("notes"))
    out["retrieval_mode"] = kv.get("retrieval_mode")
    out["design_notes_injected"] = _maybe_bool(kv.get("design_notes_injected"))
    out["exact_name_used"] = _maybe_bool(kv.get("exact_name_used"))
    out["negative_memory_count"] = _maybe_int(kv.get("negative_memory_count"))
    out["memory_records_considered"] = _maybe_int(kv.get("memory_records_considered"))
    # retrieved_episode_ids is bracketed list-ish — pull ids out crudely.
    ids_raw = kv.get("retrieved_episode_ids", "")
    if ids_raw:
        # e.g. "['a114d039']" or "['a','b']"
        ids = re.findall(r"[A-Fa-f0-9]{6,}", ids_raw)
        out["retrieved_episode_ids"] = ids
    return out


def _aggregate_action_trace(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    tool_calls_by_name: Counter = Counter()
    action_labels: Counter = Counter()
    tool_error_codes: Counter = Counter()
    pathguard_violations = 0
    pathguard_checks = 0
    best_valid_token_max = 0
    ship_lineage_source: Optional[str] = None
    for r in records:
        label = r.get("action_label")
        if label:
            action_labels[label] += 1
        tn = r.get("tool_name")
        if tn and label in ("tool_call_success", "tool_call_warning",
                            "tool_call_classified_error"):
            tool_calls_by_name[tn] += 1
        err = r.get("tool_error_code")
        if err:
            tool_error_codes[err] += 1
        if r.get("phase") == "path_guard":
            pathguard_checks += 1
            allowed = r.get("path_guard_allowed")
            if allowed is False:
                pathguard_violations += 1
        bvt = r.get("best_valid_token")
        if isinstance(bvt, (int, float)) and bvt > best_valid_token_max:
            best_valid_token_max = int(bvt)
        sl = r.get("ship_lineage_source")
        if sl:
            # Last wins: the lineage wanted is the final finalize record's.
            ship_lineage_source = sl
    return {
        "tool_calls_by_name": dict(tool_calls_by_name),
        "action_labels": dict(action_labels),
        "tool_error_codes": dict(tool_error_codes),
        "pathguard_checks": pathguard_checks,
        "pathguard_violations": pathguard_violations,
        "best_valid_token_max": best_valid_token_max,
        "ship_lineage_source": ship_lineage_source,
    }


def _extract_outcome(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Outcome is taken from the LAST finalize_end record if present."""
    final_recs = [r for r in records if r.get("action_label") == "finalize_end"]
    out: Dict[str, Any] = {
        "final_status": None,
        "final_wns": None,
        "final_fmax": None,
        "optimizer_delta_fmax": None,
        "validated_delta_fmax": None,  # filled offline by validator
        "output_dcp_path": None,
        "contributed_to_ship": "unknown",
    }
    if not final_recs:
        return out
    rec = final_recs[-1]
    out["final_status"] = rec.get("validity_state")
    out["final_wns"] = _maybe_float(rec.get("wns_after"))
    out["final_fmax"] = _maybe_float(rec.get("fmax_after"))
    out["output_dcp_path"] = rec.get("output_dcp_path")
    return out


def _extract_start_features(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pull initial WNS / fmax / RQA / target_clock from the
    `initial_analysis` + `qor_assessment` records.

    `critical_path_spread` and `lut_count` are not yet emitted into the
    decision trace; left as null until that wiring lands.
    """
    init_recs = [r for r in records if r.get("action_label") == "initial_analysis"]
    qor_recs = [r for r in records if r.get("action_label") == "qor_assessment"]
    out: Dict[str, Any] = {
        "initial_wns": None,
        "initial_fmax": None,
        "target_clock": None,
        "rqa_score": None,
        "rqa_flow_guidance": None,
        "pathology": None,
        "router_rule": None,
        "critical_path_spread": None,
        "lut_count": None,
    }
    if init_recs:
        rec = init_recs[0]
        out["initial_wns"] = _maybe_float(rec.get("wns_after"))
        out["initial_fmax"] = _maybe_float(rec.get("fmax_after"))
        kv = _parse_notes_kv(rec.get("notes"))
        out["target_clock"] = kv.get("target_clock")
        out["pathology"] = kv.get("pathology")
        out["router_rule"] = kv.get("router_rule")
        # Feature-first retrieval fields.  Prefer
        # top-level dedicated fields; tolerate older traces that
        # don't carry them.
        out["lut_count"] = _maybe_int(rec.get("lut_count"))
        out["critical_path_spread"] = _maybe_float(
            rec.get("critical_path_spread")
        )
    if qor_recs:
        kv = _parse_notes_kv(qor_recs[0].get("notes"))
        out["rqa_score"] = _maybe_int(kv.get("score"))
        out["rqa_flow_guidance"] = kv.get("flow_guidance")
    return out


def _trust_for(records: List[Dict[str, Any]],
               outcome: Dict[str, Any]) -> Dict[str, Any]:
    """Heuristic initial trust.  Future code can recompute this offline
    using the validator's claim-vs-validated drift.

    v0 rules:
          - high       — finalize_end says VALID_OPTIMIZED and zero pathguard violations.
          - medium     — VALID_FALLBACK_BASELINE (no improvement attempted/landed) — episode is structurally fine but uninformative.
          - low        — finalize never fired or unrecognised final_status.
    """
    final = outcome.get("final_status")
    pg_v = sum(1 for r in records
               if r.get("phase") == "path_guard"
               and r.get("path_guard_allowed") is False)
    if final == "VALID_OPTIMIZED" and pg_v == 0:
        return {"trust_initial": "high",
                "trust_reason": "VALID_OPTIMIZED + zero pathguard violations",
                "bug_epoch_labels": []}
    if final == "VALID_FALLBACK_BASELINE":
        return {"trust_initial": "medium",
                "trust_reason": "validator-safe fallback; no improvement signal",
                "bug_epoch_labels": []}
    if final is None:
        return {"trust_initial": "low",
                "trust_reason": "finalize_end never recorded — run did not finish",
                "bug_epoch_labels": []}
    return {"trust_initial": "low",
            "trust_reason": f"unrecognised final_status={final!r} or pg_v={pg_v}",
            "bug_epoch_labels": []}


def _load_records(jsonl_path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not jsonl_path.exists():
        return out
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(
                    f"policy_memory.load: skip {jsonl_path}:{line_no} ({exc})"
                )
    return out


def build_episode(decisions_jsonl_path: Path | str,
                  *,
                  git_commit: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Read one decisions.jsonl and return a v0 Episode dict.

    Returns None when the file is missing or unreadable.
    """
    jsonl = Path(decisions_jsonl_path)
    records = _load_records(jsonl)
    if not records:
        logger.info(f"policy_memory.build_episode: empty / missing {jsonl}")
        return None

    # Identity ------------------------------------------------------------
    run_id = (records[0].get("run_id")
              or jsonl.parent.name
              or "unknown_run")
    episode_id = episode_id_for_run(run_id)

    # ts_run: first record timestamp if available
    ts_run = _maybe_float(records[0].get("timestamp"))

    # Design name: metadata only.  Prefer the first non-null `design`
    # field; fall back to parsing run_start notes.
    design_name = _first_non_null(records, "design")
    if design_name is None:
        run_start = next((r for r in records if r.get("phase") == "run_start"),
                         None)
        if run_start:
            design_name = _design_from_run_start(run_start)

    # Mode flags
    rag_records = [r for r in records if r.get("action_label") == "rag_retrieval"]
    rag_mode = None
    if rag_records:
        rag_mode = rag_records[-1].get("rag_mode")
    contest_mode = (rag_mode == "contest_mode")

    rqa_mode = _first_non_null(records, "rqa_mode")
    model = _first_non_null(records, "model")

    start_features = _extract_start_features(records)
    retrieval_metadata = _extract_retrieval_metadata(rag_records)
    action_trace_summary = _aggregate_action_trace(records)
    outcome = _extract_outcome(records)

    # Compute optimizer_delta_fmax if both endpoints present.
    init_fmax = start_features.get("initial_fmax")
    final_fmax = outcome.get("final_fmax")
    if init_fmax is not None and final_fmax is not None:
        outcome["optimizer_delta_fmax"] = round(final_fmax - init_fmax, 4)

    trust = _trust_for(records, outcome)

    episode: Dict[str, Any] = {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "episode_id": episode_id,
        "run_id": run_id,
        "ts_run": ts_run,
        "git_commit": git_commit,
        "model": model,
        "rag_mode": rag_mode,
        "contest_mode": contest_mode,
        "rqa_mode": rqa_mode,
        "design_name": design_name,
        "design_name_masked": None,  # caller-populated for hidden-design sim
        "start_features": start_features,
        "retrieval_metadata": retrieval_metadata,
        "action_trace_summary": action_trace_summary,
        "outcome": outcome,
        "trust": trust,
        "source": {
            "decisions_jsonl_path": str(jsonl.resolve(strict=False)),
            "record_count": len(records),
        },
    }
    return episode


# Store I/O — append-only JSONL with episode_id-based dedupe.

def load_episodes(store_path: Path | str) -> List[Dict[str, Any]]:
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
                    f"policy_memory.load_episodes: skip {p}:{line_no} ({exc})"
                )
    return out


def _existing_episode_ids(store_path: Path) -> set:
    return {ep.get("episode_id") for ep in load_episodes(store_path)
            if ep.get("episode_id")}


def append_episode(store_path: Path | str, episode: Dict[str, Any]) -> bool:
    """Append `episode` to `store_path` unless an episode with the same
    episode_id is already there.  Returns True if appended, False if
    skipped (duplicate) or on write failure.
    """
    p = Path(store_path)
    eid = episode.get("episode_id")
    if not eid:
        logger.warning("policy_memory.append_episode: missing episode_id; skipped")
        return False
    if eid in _existing_episode_ids(p):
        logger.info(f"policy_memory.append_episode: dedupe skip {eid}")
        return False
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(episode, ensure_ascii=False))
            fh.write("\n")
    except OSError as exc:
        logger.warning(f"policy_memory.append_episode: write {p} failed: {exc}")
        return False
    return True


def ingest_decisions(decisions_jsonl_path: Path | str,
                     store_path: Path | str,
                     *,
                     git_commit: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """End-to-end: build one Episode from a decisions.jsonl and append
    it to the store with dedupe.  Returns the episode dict (whether or
    not it was appended) or None on build failure.
    """
    ep = build_episode(decisions_jsonl_path, git_commit=git_commit)
    if ep is None:
        return None
    append_episode(store_path, ep)
    return ep


# Default store path: next to the optimizer repo root by default.  Caller
# can override via STORE env or explicit arg.

DEFAULT_STORE_RELATIVE = Path("policy_memory") / "episode_store.jsonl"


def default_store_path(repo_root: Optional[Path | str] = None) -> Path:
    if repo_root is None:
        # __file__ = optimizer/policy_memory.py → repo root is parent.parent
        repo_root = Path(__file__).resolve().parent.parent
    env = os.environ.get("POLICY_MEMORY_STORE")
    if env:
        return Path(env)
    return Path(repo_root) / DEFAULT_STORE_RELATIVE
