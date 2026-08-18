"""Decision tracer — minimal per-iteration / per-event JSONL telemetry.

Goal: emit a structured record for every optimizer decision (tool call,
finalize event, validator hit) into `runs/<run_id>/decisions.jsonl`.
Future infra (policy memory, ablation harness, structured error
recovery) consumes this stream.

This first implementation is INTENTIONALLY MINIMAL.  No policy card,
no retrieval, no ML.  Just:
  - one append-atomic JSONL writer
  - a stable, documented schema (SCHEMA_VERSION)
  - tolerance for missing fields (everything optional)
  - silent failure on write errors (never blocks optimization)

Design contract:
  DecisionTracer.emit(record_dict) → bool
  DecisionTracer.path_for_run(run_id) → Path
  load_decisions(path) → list[dict]

Schema documented in DECISION_RECORD_FIELDS below.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

# Schema version bumps:
# v1 (2026-05-20 P1): initial fields below.
SCHEMA_VERSION = 1

# Documented set of fields the schema RECOGNIZES.  Unknown fields are
# preserved on write but flagged in the parser.  Missing fields are
# allowed — every field is optional (caller fills what it knows).
DECISION_RECORD_FIELDS = {
    # identity
    "schema_version", "run_id", "timestamp", "design", "iteration", "phase",
    # action
    "tool_name", "action_label", "decision_source",
    # configuration
    "rag_mode", "rqa_mode", "model",
    # measurements
    "wns_before", "wns_after", "fmax_before", "fmax_after",
    "delta_wns", "delta_fmax",
    # lineage / state
    "validity_state", "best_valid_checkpoint_path",
    "best_valid_token", "lineage_token", "ship_lineage_source",
    "output_dcp_path",
    # feature-first retrieval (initial_analysis only, Session 6 P1)
    "lut_count", "critical_path_spread",
    # error + cost
    "tool_error_code", "runtime_s",
    "token_count", "cost_usd",
    # free-form
    "notes",
}

# Recognized decision_source values (closed set; off-spec values are
# allowed but logged as a warning).
DECISION_SOURCES = {
    "router", "planner", "executor", "reviewer",
    "rag", "rqa", "fallback", "finalize", "validator",
    # MCP doc-search is read-only evidence, not yet a decision source;
    # add when it becomes one.
}


class DecisionTracer:
    """Append-only JSONL writer for decision records.

    Public API:
      - emit(record): best-effort append of one record.  Returns True
        on success, False if write failed (and logs a warning).
      - path_for_run(run_id): the canonical path under a run dir.
      - close(): noop today; reserved for future buffered modes.

    Threading: not thread-safe.  Single-writer per run (the optimizer
    runs one iteration at a time).
    """

    def __init__(self, run_dir: Path | str, run_id: Optional[str] = None):
        self.run_dir = Path(run_dir)
        # Run id defaults to the run_dir name (e.g.
        # dcp_optimizer_run-20260520_083834).  Callers can override.
        self.run_id = run_id or self.run_dir.name
        self._path = self.run_dir / "decisions.jsonl"
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(
                f"decision_tracer: cannot ensure run_dir {self.run_dir}: {exc}"
            )

    @classmethod
    def path_for_run(cls, run_dir: Path | str) -> Path:
        """Return the JSONL path that would be written for this run."""
        return Path(run_dir) / "decisions.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    def emit(self, record: Dict[str, Any]) -> bool:
        """Best-effort append of one record.  Stamps schema_version,
        run_id, and timestamp if missing.  Never raises.

        Returns True on successful write, False otherwise.  Failures
        are logged at WARNING level once per emit so the optimizer
        keeps running.
        """
        if not isinstance(record, dict):
            logger.warning("decision_tracer.emit: record is not a dict; skipped")
            return False

        # Stamp standard headers; do not overwrite caller-provided values.
        record.setdefault("schema_version", SCHEMA_VERSION)
        record.setdefault("run_id", self.run_id)
        record.setdefault("timestamp", time.time())

        # Sanity check: decision_source, if present, should be in the
        # closed set.  Off-spec values are warned but not rejected.
        ds = record.get("decision_source")
        if ds is not None and ds not in DECISION_SOURCES:
            logger.debug(
                f"decision_tracer.emit: unrecognized decision_source={ds!r}; "
                f"recording anyway"
            )

        try:
            line = json.dumps(record, default=_safe_json_default,
                              ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            logger.warning(
                f"decision_tracer.emit: cannot json-encode record "
                f"({exc}); fields={list(record.keys())}"
            )
            return False

        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.write("\n")
        except OSError as exc:
            logger.warning(
                f"decision_tracer.emit: append to {self._path} failed: {exc}"
            )
            return False

        return True

    def close(self) -> None:
        """Reserved for future buffered modes; noop today."""
        return None


def _safe_json_default(obj: Any) -> Any:
    """Conservative fallback so unusual values (Path, set, custom dataclass)
    don't crash the writer.  Tries common shapes; otherwise stringifies."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(list(obj))
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return {k: v for k, v in obj.__dict__.items()
                    if not k.startswith("_")}
        except Exception:
            pass
    return str(obj)


def load_decisions(path: Path | str) -> List[Dict[str, Any]]:
    """Read a decisions.jsonl back into a list of dicts.  Skips
    malformed lines (logs at WARNING) so a partially-written file
    after a crash is still readable.

    Returns [] when the file is missing.
    """
    path = Path(path)
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.warning(
                        f"load_decisions: skipping {path}:{line_no} ({exc})"
                    )
    except OSError as exc:
        logger.warning(f"load_decisions: cannot read {path}: {exc}")
    return out


def filter_decisions(
    records: Iterable[Dict[str, Any]],
    *,
    design: Optional[str] = None,
    decision_source: Optional[str] = None,
    iteration: Optional[int] = None,
    has_tool_error: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Tiny query helper for ad-hoc inspection.  Pass None to ignore
    a filter."""
    out = []
    for r in records:
        if design is not None and r.get("design") != design:
            continue
        if decision_source is not None and r.get("decision_source") != decision_source:
            continue
        if iteration is not None and r.get("iteration") != iteration:
            continue
        if has_tool_error is not None:
            has_err = r.get("tool_error_code") is not None
            if has_err != has_tool_error:
                continue
        out.append(r)
    return out
