"""Tolerant reader for Vivado JSON QoR Summary reports.

Parses the JSON produced by:
    report_design_analysis -qor_summary -json <filename>

Returns a flat dict whose keys map onto the destinations documented in
`.planning/session12_qor_feature_mapping.md`:

    qor_steps                       : list[str]
    runtime_per_step                : dict[step_name → int_minutes]
    directives_observed             : list[str]
    per_step_wns                    : list[float | None]
    per_step_tns                    : list[float | None]
    global_cong_level_NESW          : list[int | None] length 4
    global_cong_tile_NESW           : list[float | None] length 4
    long_cong_level_NESW            : list[int | None] length 4
    long_cong_tile_NESW             : list[float | None] length 4
    short_cong_level_NESW           : list[int | None] length 4
    short_cong_tile_NESW            : list[float | None] length 4
    qor_tool_version                : str | None
    qor_design_state                : str | None

This module is INTENTIONALLY ADVISORY-ONLY:
  - never invokes Vivado,
  - never reads/writes DCPs,
  - never references benchmark names,
  - never raises on malformed input — returns nulls per field instead,
  - NOT wired into the optimizer's control flow.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

QOR_SCHEMA_VERSION = 1


def _parse_4tuple(s: Optional[str], cast) -> List[Optional[Any]]:
    """Parse '0 1 1 0' or '- 0 - -' (4 tokens) into [cast|None × 4]."""
    if not isinstance(s, str) or not s.strip():
        return [None, None, None, None]
    tokens = s.strip().split()
    out: List[Optional[Any]] = []
    for i in range(4):
        tok = tokens[i] if i < len(tokens) else None
        if tok is None or tok == "-" or tok == "":
            out.append(None)
            continue
        try:
            out.append(cast(tok))
        except (ValueError, TypeError):
            out.append(None)
    return out


def _maybe_float(s: Any) -> Optional[float]:
    if s is None:
        return None
    if isinstance(s, str):
        s = s.strip()
        if not s or s == "-":
            return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _maybe_int(s: Any) -> Optional[int]:
    if s is None:
        return None
    if isinstance(s, str):
        s = s.strip()
        if not s or s == "-":
            return None
    try:
        return int(float(s))  # tolerant of "4.0"
    except (ValueError, TypeError):
        return None


def parse_qor_json(path: Union[Path, str]) -> Dict[str, Any]:
    """Read a JSON QoR file and return the flat feature dict.

    On ANY error (missing file, malformed JSON, unexpected shape),
    returns a dict with all-null fields and a `schema_ok=False`
    marker.  Never raises.
    """
    empty4_int = [None, None, None, None]
    empty4_flt = [None, None, None, None]
    out: Dict[str, Any] = {
        "schema_version": QOR_SCHEMA_VERSION,
        "schema_ok": False,
        "qor_steps": [],
        "runtime_per_step": {},
        "directives_observed": [],
        "per_step_wns": [],
        "per_step_tns": [],
        "global_cong_level_NESW": list(empty4_int),
        "global_cong_tile_NESW": list(empty4_flt),
        "long_cong_level_NESW": list(empty4_int),
        "long_cong_tile_NESW": list(empty4_flt),
        "short_cong_level_NESW": list(empty4_int),
        "short_cong_tile_NESW": list(empty4_flt),
        "qor_tool_version": None,
        "qor_design_state": None,
        "source_path": str(path),
    }

    p = Path(path)
    if not p.exists():
        return out
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        logger.warning(f"qor_json_features: cannot read {p}: {exc}")
        return out

    if not isinstance(data, dict):
        return out

    # Report Information ------------------------------------------------
    ri = data.get("Report Information")
    if isinstance(ri, dict):
        tv = ri.get("Tool Version")
        ds = ri.get("Design State")
        out["qor_tool_version"] = tv if isinstance(tv, str) else None
        out["qor_design_state"] = ds if isinstance(ds, str) else None

    rows = data.get("Design QoR Summary")
    if not isinstance(rows, list):
        return out

    out["schema_ok"] = True
    out["qor_steps"] = []
    out["runtime_per_step"] = {}
    out["directives_observed"] = []
    out["per_step_wns"] = []
    out["per_step_tns"] = []

    # Track the LAST route-step row for congestion fields — that's where
    # Vivado emits the design-final-routed congestion numbers.
    route_row: Optional[Dict[str, Any]] = None

    for r in rows:
        if not isinstance(r, dict):
            continue
        tn = r.get("Task Name")
        if isinstance(tn, str) and tn:
            out["qor_steps"].append(tn)
            rt = _maybe_int(r.get("Runtime(mins)"))
            if rt is not None:
                # If a step name repeats (e.g., phys_opt_design pre+post route)
                # keep the LAST value — closer to final compile cost.
                out["runtime_per_step"][tn] = rt
        dirs = r.get("Directives")
        if isinstance(dirs, str) and dirs.strip():
            out["directives_observed"].append(dirs.strip())
        out["per_step_wns"].append(_maybe_float(r.get("WNS(ns)")))
        out["per_step_tns"].append(_maybe_float(r.get("TNS(ns)")))
        if isinstance(tn, str) and "route" in tn.lower():
            route_row = r

    if route_row is not None:
        out["global_cong_level_NESW"] = _parse_4tuple(
            route_row.get("Global Cong Level N-E-S-W"), int)
        out["global_cong_tile_NESW"] = _parse_4tuple(
            route_row.get("Global Cong Tile% N-E-S-W"), float)
        out["long_cong_level_NESW"] = _parse_4tuple(
            route_row.get("Long Cong Level N-E-S-W"), int)
        out["long_cong_tile_NESW"] = _parse_4tuple(
            route_row.get("Long Cong Tile% N-E-S-W"), float)
        out["short_cong_level_NESW"] = _parse_4tuple(
            route_row.get("Short Cong Level N-E-S-W"), int)
        out["short_cong_tile_NESW"] = _parse_4tuple(
            route_row.get("Short Cong Tile% N-E-S-W"), float)

    return out
