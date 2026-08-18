"""Cross-join standalone-validation JSON results into the policy-memory
episode store.

Inputs (per design):
  - /tmp/session6_campaign/standalone_validation/<design>_validation.json
    {
      "design": str,
      "output_dcp": str,
      "contest_clk_wns_ns": float | None,
      "contest_clk_fmax_mhz": float | None,
      "clock_name": str | None,
    }
  - submission/baselines.tsv   (design → baseline_fmax_mhz)

Output:
  - in-place rewrite of policy_memory/episode_store.jsonl with
    matching episodes' `outcome` block extended with:
      - validated_fmax           (float | None)
      - validated_wns            (float | None)
      - validated_delta_fmax     (validated_fmax − baseline_fmax)
      - validation_source_path   (str)
      - validation_status        ("ok" | "missing_baseline" | "missing_fmax" | "unmatched")
      - candidate_beats_ship     (bool, when ship reference is known)
      - ship_fmax_reference      (float | None)
      - ship_delta_reference     (float | None)
      - contributed_to_ship      ("no" — campaigns never auto-ship)

Matching contract:
  - PRIMARY:  episode.outcome.output_dcp_path == validation_json.output_dcp
  - design_name is metadata only — never the match key.

Idempotency:
  - If episode.outcome.validated_delta_fmax is already populated from
    the same validation_source_path AND values match, skip (noop).
  - Otherwise overwrite the validation fields (deterministic update).

Unmatched validation records (no corresponding episode by output_dcp_path)
are reported to the caller; they do NOT silently get dropped.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from optimizer.policy_memory import default_store_path, load_episodes

logger = logging.getLogger(__name__)


# Ship references — the +513.89 anchor portfolio.  Read from
# submission/results.tsv when available; falls back to a hard-coded
# snapshot so the join still works without the file present (e.g.
# in unit tests).
_HARDCODED_SHIP_FMAX = {
    "amd_mini-isp": 407.83,
    "boom_soc": 48.24,
    "corescore_500_mod": 406.01,
    "finn_radioml": 338.98,
    "ispd16_example2": 131.51,
    "logicnets_jscl": 470.59,
    "rosetta_3d-rendering": 277.55,
    "rosetta_digit-recognition": 408.00,
    "rosetta_optical-flow": 348.55,
    "rosetta_spam-filter": 442.28,
    "vexriscv_re-place": 435.54,
    "vexriscv_re-place_v2": 397.46,
    "vtr_mcml": 67.15,
}


@dataclass
class JoinResult:
    matched: int = 0
    updated: int = 0
    skipped_idempotent: int = 0
    unmatched: List[str] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.unmatched is None:
            self.unmatched = []
        if self.errors is None:
            self.errors = []


def _read_baselines(baselines_tsv: Path | str) -> Dict[str, float]:
    """Parse a TSV with `design` + `baseline_fmax_mhz` columns.
    Tolerant of CRLF + stray `\\r` characters mid-line that some
    Windows-side tooling emits.
    """
    out: Dict[str, float] = {}
    p = Path(baselines_tsv)
    if not p.exists():
        return out
    # Read as bytes so Python's universal-newlines mode doesn't
    # convert mid-cell `\r` characters to `\n`s.  Some Windows-side
    # tooling emits CRs INSIDE cells, which text-mode reads then
    # alias as line breaks.
    try:
        raw = p.read_bytes()
    except OSError:
        return out
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return out
    text = text.replace("\r", "")
    rows = [line.split("\t") for line in text.split("\n") if line]
    if not rows:
        return out
    header = rows[0]
    try:
        i_design = header.index("design")
        i_fmax = header.index("baseline_fmax_mhz")
    except ValueError:
        return out
    for cells in rows[1:]:
        if len(cells) <= max(i_design, i_fmax):
            continue
        try:
            out[cells[i_design].strip()] = float(cells[i_fmax].strip())
        except (ValueError, TypeError):
            continue
    return out


def _read_ship_fmax(results_tsv: Path | str) -> Dict[str, float]:
    """Read ship Fmax per design from submission/results.tsv.  Falls
    back to the hard-coded anchor portfolio if the file is missing /
    unparseable.

    Tolerant of CRLF + stray `\\r` characters mid-line.
    """
    out: Dict[str, float] = {}
    p = Path(results_tsv)
    if not p.exists():
        return dict(_HARDCODED_SHIP_FMAX)
    try:
        raw = p.read_bytes()
    except OSError:
        return dict(_HARDCODED_SHIP_FMAX)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return dict(_HARDCODED_SHIP_FMAX)
    text = text.replace("\r", "")
    rows = [line.split("\t") for line in text.split("\n") if line]
    if not rows:
        return dict(_HARDCODED_SHIP_FMAX)
    header = rows[0]
    try:
        i_design = header.index("design")
        i_fmax = header.index("ship_fmax_mhz")
    except ValueError:
        return dict(_HARDCODED_SHIP_FMAX)
    for cells in rows[1:]:
        if len(cells) <= max(i_design, i_fmax):
            continue
        try:
            out[cells[i_design].strip()] = float(cells[i_fmax].strip())
        except (ValueError, TypeError):
            continue
    return out if out else dict(_HARDCODED_SHIP_FMAX)


def _load_validation_records(dirpath: Path | str) -> List[Dict[str, Any]]:
    p = Path(dirpath)
    out: List[Dict[str, Any]] = []
    if not p.is_dir():
        return out
    for jf in sorted(p.glob("*_validation.json")):
        try:
            with jf.open() as fh:
                rec = json.load(fh)
            rec["_source_path"] = str(jf.resolve())
            out.append(rec)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"skip {jf}: {exc}")
    return out


def _candidate_payload(
    val_rec: Dict[str, Any],
    baselines: Dict[str, float],
    ship_fmaxs: Dict[str, float],
) -> Dict[str, Any]:
    """Compute the payload to merge into episode.outcome from one
    validation JSON record.  Always returns a dict; `validation_status`
    encodes any issues."""
    design = val_rec.get("design")
    fmax = val_rec.get("contest_clk_fmax_mhz")
    wns = val_rec.get("contest_clk_wns_ns")
    baseline = baselines.get(design)
    ship = ship_fmaxs.get(design)

    if fmax is None:
        status = "missing_fmax"
        delta = None
    elif baseline is None:
        status = "missing_baseline"
        delta = None
    else:
        status = "ok"
        delta = round(float(fmax) - float(baseline), 4)

    if (fmax is not None) and (ship is not None):
        beats_ship = fmax > ship + 0.5  # noise band
        ship_delta = round(float(fmax) - float(ship), 4)
    else:
        beats_ship = None
        ship_delta = None

    return {
        "validated_fmax": fmax,
        "validated_wns": wns,
        "validated_delta_fmax": delta,
        "validation_source_path": val_rec.get("_source_path"),
        "validation_status": status,
        "ship_fmax_reference": ship,
        "ship_delta_reference": ship_delta,
        "candidate_beats_ship": beats_ship,
        "contributed_to_ship": "no",
    }


def _atomic_rewrite(store_path: Path, episodes: List[Dict[str, Any]]) -> None:
    """Write the full episode list back to disk atomically."""
    store_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False,
        dir=str(store_path.parent), prefix=".episode_store.",
        suffix=".jsonl.tmp",
    )
    try:
        for ep in episodes:
            tmp.write(json.dumps(ep, ensure_ascii=False))
            tmp.write("\n")
        tmp.flush()
        os.fsync(tmp.fileno())
    finally:
        tmp.close()
    os.replace(tmp.name, str(store_path))


def join_validation_into_store(
    *,
    validation_dir: Path | str,
    baselines_tsv: Path | str,
    store_path: Optional[Path | str] = None,
    results_tsv: Optional[Path | str] = None,
) -> JoinResult:
    """Idempotent in-place join.  Returns a JoinResult describing
    what was updated / skipped / unmatched."""
    store_path = Path(store_path) if store_path else default_store_path()
    episodes = load_episodes(store_path)
    val_recs = _load_validation_records(validation_dir)
    baselines = _read_baselines(baselines_tsv)
    if results_tsv is not None:
        ship = _read_ship_fmax(results_tsv)
    else:
        # Default to repo submission/results.tsv if it exists.
        default_results = Path(store_path).parent.parent / "submission" / "results.tsv"
        ship = _read_ship_fmax(default_results)

    # Index episodes by output_dcp_path for O(1) match.
    by_output: Dict[str, int] = {}
    for i, ep in enumerate(episodes):
        out_path = (ep.get("outcome") or {}).get("output_dcp_path")
        if out_path:
            by_output[str(out_path)] = i

    res = JoinResult()
    for vrec in val_recs:
        out_dcp = vrec.get("output_dcp")
        if not out_dcp:
            res.errors.append(f"{vrec.get('_source_path')}: missing output_dcp")
            continue
        idx = by_output.get(str(out_dcp))
        if idx is None:
            res.unmatched.append(out_dcp)
            continue
        res.matched += 1
        payload = _candidate_payload(vrec, baselines, ship)
        outcome = episodes[idx].setdefault("outcome", {})
        # Idempotency check: if every key already matches, skip.
        unchanged = all(
            outcome.get(k) == v
            for k, v in payload.items()
        )
        if unchanged:
            res.skipped_idempotent += 1
            continue
        for k, v in payload.items():
            outcome[k] = v
        res.updated += 1

    if res.updated:
        _atomic_rewrite(store_path, episodes)
    return res
