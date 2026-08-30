"""Offline attachment of parsed JSON QoR features to episode_store records.

This module is INTENTIONALLY OFFLINE-ONLY:

- Never calls Vivado.
- Never reads or writes DCPs.
- Never mutates the submission tree (no path under ``submission/``).
- Never uses ``design_name`` as a *retrieval key* or *strategy condition* —
  it may be used only as a low-priority *match hint* when no stronger
  provenance is available.
- Never marks an episode as ``contributed_to_ship``.
- Never overwrites ``outcome.validated_delta_fmax`` or related ship-level
  fields.
- Idempotent: re-running with the same inputs produces a byte-identical
  output file.

Inputs:
- ``policy_memory/episode_store.jsonl`` (read).
- One or more ``*.qor.json`` files produced by the tolerant QoR probe.

Outputs:
- Updated ``policy_memory/episode_store.jsonl`` (optional; only if matches are
  unambiguous and ``apply=True``).
- A JoinReport dict describing matched / report-only / unmatched / ambiguous.

Strategy hints (lowest cost first):

1. ``output_dcp_path`` substring match (strongest provenance).
2. ``run_dir`` / ``run_id`` substring match.
3. ``design_name`` as last-resort hint (still flagged as weak match).

If multiple episodes match the same JSON under stronger provenance, the join
refuses to write and labels it ``AMBIGUOUS``.

Storage keys written under ``episode.start_features`` (when JOINED):

    qor_congestion_global_n / _e / _s / _w
    qor_congestion_long_n   / _e / _s / _w
    qor_congestion_short_n  / _e / _s / _w
    qor_route_bound_score
    qor_has_congestion_signal
    qor_schema_version
    qor_tool_version
    qor_design_state
    qor_source_path

The flat scalar shape keeps episode_store diffable and round-trip stable.
"""
from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - import shape varies by entry point
    from optimizer.qor_json_features import parse_qor_json
except ImportError:  # pragma: no cover
    from qor_json_features import parse_qor_json  # type: ignore

logger = logging.getLogger(__name__)


# Flat keys written under episode.start_features when joining.
QOR_FLAT_KEYS: Tuple[str, ...] = (
    "qor_congestion_global_n",
    "qor_congestion_global_e",
    "qor_congestion_global_s",
    "qor_congestion_global_w",
    "qor_congestion_long_n",
    "qor_congestion_long_e",
    "qor_congestion_long_s",
    "qor_congestion_long_w",
    "qor_congestion_short_n",
    "qor_congestion_short_e",
    "qor_congestion_short_s",
    "qor_congestion_short_w",
    "qor_route_bound_score",
    "qor_has_congestion_signal",
    "qor_schema_version",
    "qor_tool_version",
    "qor_design_state",
    "qor_source_path",
)

STATUS_JOINED = "JOINED"
STATUS_REPORT_ONLY = "REPORT_ONLY"
STATUS_UNMATCHED = "UNMATCHED"
STATUS_AMBIGUOUS = "AMBIGUOUS"
STATUS_REJECTED = "REJECTED"

MATCH_OUTPUT_DCP = "output_dcp_path"
MATCH_RUN_DIR = "run_dir"
MATCH_DESIGN_NAME_HINT = "design_name_hint"
MATCH_NONE = "none"


@dataclass
class JoinEntry:
    json_path: str
    status: str
    matched_episode_ids: List[str] = field(default_factory=list)
    match_method: str = MATCH_NONE
    fields_attached: List[str] = field(default_factory=list)
    route_bound_score: Optional[float] = None
    congestion_summary: Optional[Dict[str, Any]] = None
    notes: str = ""


@dataclass
class JoinReport:
    entries: List[JoinEntry] = field(default_factory=list)
    episode_store_changed: bool = False
    episodes_modified: int = 0
    unmatched_count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "episode_store_changed": self.episode_store_changed,
            "episodes_modified": self.episodes_modified,
            "unmatched_count": self.unmatched_count,
            "entries": [
                {
                    "json_path": e.json_path,
                    "status": e.status,
                    "matched_episode_ids": list(e.matched_episode_ids),
                    "match_method": e.match_method,
                    "fields_attached": list(e.fields_attached),
                    "route_bound_score": e.route_bound_score,
                    "congestion_summary": e.congestion_summary,
                    "notes": e.notes,
                }
                for e in self.entries
            ],
        }


def _compute_route_bound_score(parsed: Dict[str, Any]) -> Tuple[Optional[float], bool]:
    """Deterministic in [0,1] from concatenated long+global NESW levels.

    Returns ``(score, has_signal)``.  ``score`` is None when every entry is
    null.  Uses max of clipped 0..7 levels divided by 7 — keeps the metric
    monotone in the worst direction.
    """
    longs = parsed.get("long_cong_level_NESW") or [None] * 4
    globs = parsed.get("global_cong_level_NESW") or [None] * 4
    levels = [v for v in (list(longs) + list(globs)) if v is not None]
    if not levels:
        return None, False
    try:
        clipped = [min(max(int(v), 0), 7) for v in levels]
    except (TypeError, ValueError):
        return None, False
    return round(max(clipped) / 7.0, 4), True


def _flatten_qor_for_storage(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Map the parser dict to the flat scalar keys we persist under start_features."""
    def _gv(key: str, idx: int) -> Optional[int]:
        v = (parsed.get(key) or [None] * 4)
        return v[idx] if idx < len(v) else None

    score, has_signal = _compute_route_bound_score(parsed)

    return {
        "qor_congestion_global_n": _gv("global_cong_level_NESW", 0),
        "qor_congestion_global_e": _gv("global_cong_level_NESW", 1),
        "qor_congestion_global_s": _gv("global_cong_level_NESW", 2),
        "qor_congestion_global_w": _gv("global_cong_level_NESW", 3),
        "qor_congestion_long_n":   _gv("long_cong_level_NESW", 0),
        "qor_congestion_long_e":   _gv("long_cong_level_NESW", 1),
        "qor_congestion_long_s":   _gv("long_cong_level_NESW", 2),
        "qor_congestion_long_w":   _gv("long_cong_level_NESW", 3),
        "qor_congestion_short_n":  _gv("short_cong_level_NESW", 0),
        "qor_congestion_short_e":  _gv("short_cong_level_NESW", 1),
        "qor_congestion_short_s":  _gv("short_cong_level_NESW", 2),
        "qor_congestion_short_w":  _gv("short_cong_level_NESW", 3),
        "qor_route_bound_score":   score,
        "qor_has_congestion_signal": has_signal,
        "qor_schema_version":      parsed.get("schema_version"),
        "qor_tool_version":        parsed.get("qor_tool_version"),
        "qor_design_state":        parsed.get("qor_design_state"),
        "qor_source_path":         parsed.get("source_path"),
    }


def _is_submission_path(path: str) -> bool:
    """Return True if ``path`` looks like a submission tree write target."""
    if not isinstance(path, str):
        return False
    norm = path.replace("\\", "/").lower()
    return "/submission/" in norm or norm.endswith("/submission") or norm.startswith("submission/")


def _load_episodes(episode_store_path: Path) -> List[Dict[str, Any]]:
    eps: List[Dict[str, Any]] = []
    if not episode_store_path.exists():
        return eps
    for line in episode_store_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            eps.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning(f"episode_qor_join: skipping malformed episode line in {episode_store_path}")
    return eps


def _episode_matches_json(episode: Dict[str, Any], json_path: Path) -> Tuple[str, int]:
    """Return ``(match_method, confidence)`` for one (episode, JSON) pair.

    Confidence: 0 = no match, 1 = design_name hint only, 2 = run_dir / run_id,
    3 = output_dcp_path substring.  Larger wins.

    A directory-proximity tiebreaker lives in the JOIN
    PLANNER (not here) — when multiple episodes tie at confidence 3,
    the planner prefers the one whose output_dcp_path shares a
    discriminator directory with the JSON's parent.
    """
    name = json_path.stem
    if name.endswith(".qor"):
        name = name[:-4]
    name_l = name.lower()

    outcome = episode.get("outcome") or {}
    out_dcp = outcome.get("output_dcp_path")
    if isinstance(out_dcp, str) and out_dcp:
        out_l = out_dcp.lower().replace("\\", "/")
        if name_l in Path(out_l).stem.lower():
            return MATCH_OUTPUT_DCP, 3

    run_id = episode.get("run_id")
    if isinstance(run_id, str) and run_id:
        if name_l in run_id.lower():
            return MATCH_RUN_DIR, 2
    src = episode.get("source") or {}
    decisions_path = src.get("decisions_jsonl_path")
    if isinstance(decisions_path, str) and decisions_path:
        if name_l in decisions_path.lower():
            return MATCH_RUN_DIR, 2

    design_name = episode.get("design_name")
    if isinstance(design_name, str) and design_name:
        if design_name.lower() == name_l:
            return MATCH_DESIGN_NAME_HINT, 1

    return MATCH_NONE, 0


def _dir_proximity_score(episode: Dict[str, Any], json_path: Path) -> int:
    """Return a directory-proximity score for a (json, episode) pair.

    Higher is better.  Used to break ties when multiple episodes match
    a JSON at the same confidence tier.

      3  json sits in the same directory as the episode's output_dcp.
      2  json's parent path shares a non-trivial directory segment with
         the episode's output_dcp parent (excluding `live_tests` /
         `submission` / the design name itself — those are too generic).
      1  json's parent contains the episode's run_id.
      0  no proximity.
    """
    out_dcp = (episode.get("outcome") or {}).get("output_dcp_path")
    if not isinstance(out_dcp, str) or not out_dcp:
        return 0
    try:
        json_parent = str(Path(json_path).resolve().parent).lower().replace("\\", "/")
    except Exception:
        json_parent = str(Path(json_path).parent).lower().replace("\\", "/")
    dcp_parent = str(Path(out_dcp).parent).lower().replace("\\", "/")
    if json_parent == dcp_parent:
        return 3
    design_name_l = (episode.get("design_name") or "").lower()
    generic = {"live_tests", "submission", "submission_tree", "dcps",
                "policy_memory", "tests", "scripts", design_name_l, ""}
    j_parts = {p for p in json_parent.split("/") if p and p not in generic}
    d_parts = {p for p in dcp_parent.split("/") if p and p not in generic}
    if j_parts & d_parts:
        return 2
    run_id = episode.get("run_id")
    if isinstance(run_id, str) and run_id and run_id.lower() in json_parent:
        return 1
    return 0


def _attach_qor_fields(episode: Dict[str, Any], flat: Dict[str, Any]) -> List[str]:
    """Idempotently merge ``flat`` into ``episode['start_features']``.

    Returns the list of keys actually written (empty when nothing changed).
    Existing non-qor fields are preserved.  Same input → no diff.
    """
    sf = episode.setdefault("start_features", {})
    if not isinstance(sf, dict):
        return []
    changed: List[str] = []
    for k in QOR_FLAT_KEYS:
        new_val = flat.get(k)
        if k not in sf or sf[k] != new_val:
            sf[k] = new_val
            changed.append(k)
    return changed


def _atomic_write_lines(target: Path, lines: Iterable[str]) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    text = "\n".join(lines) + ("\n" if lines else "")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)


def join_qor_features(
    episode_store_path: Path,
    json_paths: Iterable[Path],
    apply: bool = False,
    allow_design_name_hint_join: bool = False,
) -> JoinReport:
    """Plan (and optionally apply) an offline QoR → episode_store join.

    Parameters
    ----------
    episode_store_path : Path
        Path to ``policy_memory/episode_store.jsonl``.
    json_paths : Iterable[Path]
        JSON QoR files to consider.  Missing files become UNMATCHED.
    apply : bool
        When True, write the updated episode_store back to disk atomically.
        When False (default), perform a dry-run and return the report only.
    allow_design_name_hint_join : bool
        When False (default), confidence==1 (design-name-only) is recorded as
        ``REPORT_ONLY``, not JOINED.  Hidden-design safety: do NOT enable
        this in production paths.

    Returns
    -------
    JoinReport
    """
    if _is_submission_path(str(episode_store_path)):
        raise ValueError(
            f"refusing to write into submission tree: {episode_store_path}"
        )

    episodes = _load_episodes(episode_store_path)
    report = JoinReport()
    # Track which episodes are claimed by which json — for ambiguity detection.
    claims: Dict[str, List[str]] = {}  # episode_id -> [json_path]
    plan: Dict[str, Tuple[Dict[str, Any], Dict[str, Any], int, str]] = {}
    # Value tuple: (episode, flat QoR dict, confidence, match method).

    for jp in json_paths:
        jp = Path(jp)
        entry = JoinEntry(json_path=str(jp), status=STATUS_UNMATCHED)
        if _is_submission_path(str(jp)):
            entry.status = STATUS_REJECTED
            entry.notes = "json path inside submission tree"
            report.entries.append(entry)
            continue
        if not jp.exists():
            entry.status = STATUS_UNMATCHED
            entry.notes = "file does not exist"
            report.entries.append(entry)
            report.unmatched_count += 1
            continue

        parsed = parse_qor_json(jp)
        if not parsed.get("schema_ok"):
            entry.status = STATUS_REJECTED
            entry.notes = "parser returned schema_ok=False"
            report.entries.append(entry)
            continue

        flat = _flatten_qor_for_storage(parsed)
        entry.route_bound_score = flat.get("qor_route_bound_score")
        entry.congestion_summary = {
            "global": [flat["qor_congestion_global_n"], flat["qor_congestion_global_e"],
                       flat["qor_congestion_global_s"], flat["qor_congestion_global_w"]],
            "long":   [flat["qor_congestion_long_n"], flat["qor_congestion_long_e"],
                       flat["qor_congestion_long_s"], flat["qor_congestion_long_w"]],
            "short":  [flat["qor_congestion_short_n"], flat["qor_congestion_short_e"],
                       flat["qor_congestion_short_s"], flat["qor_congestion_short_w"]],
            "has_signal": flat["qor_has_congestion_signal"],
        }

        # Score every episode against this JSON.
        scored: List[Tuple[int, str, Dict[str, Any]]] = []
        for ep in episodes:
            method, conf = _episode_matches_json(ep, jp)
            if conf == 0:
                continue
            scored.append((conf, method, ep))

        if not scored:
            entry.status = STATUS_UNMATCHED
            entry.notes = "no episode matched on output_dcp_path / run_id / design_name"
            report.entries.append(entry)
            report.unmatched_count += 1
            continue

        # Pick the highest-confidence tier.
        best_conf = max(c for c, _, _ in scored)
        best_tier = [(m, ep) for c, m, ep in scored if c == best_conf]

        if best_conf == 1 and not allow_design_name_hint_join:
            entry.status = STATUS_REPORT_ONLY
            entry.match_method = MATCH_DESIGN_NAME_HINT
            entry.matched_episode_ids = [ep.get("episode_id", "?") for _, ep in best_tier]
            entry.notes = (
                f"design-name hint only ({len(best_tier)} candidate episode(s)); "
                "join disabled by allow_design_name_hint_join=False"
            )
            report.entries.append(entry)
            continue

        # Proximity tiebreaker: when multiple episodes match at the same
        # confidence tier (typically because two runs produced DCPs with
        # the same basename in different directories), prefer the episode
        # whose output_dcp_path is in the same directory as the JSON, or
        # shares a non-trivial path segment (e.g., the session/run name).
        if len(best_tier) > 1:
            proximity = [(_dir_proximity_score(ep, jp), m, ep) for m, ep in best_tier]
            top_prox = max(p for p, _, _ in proximity)
            survivors = [(m, ep) for p, m, ep in proximity if p == top_prox]
            if len(survivors) == 1 and top_prox >= 2:
                best_tier = survivors  # tiebreaker resolved it
            else:
                entry.status = STATUS_AMBIGUOUS
                entry.match_method = best_tier[0][0]
                entry.matched_episode_ids = [ep.get("episode_id", "?") for _, ep in best_tier]
                entry.notes = (
                    f"{len(best_tier)} episodes tied at confidence {best_conf}; "
                    f"proximity tiebreaker did not isolate one (max_proximity={top_prox})"
                )
                report.entries.append(entry)
                continue

        method, ep = best_tier[0]
        ep_id = ep.get("episode_id", "?")
        # If another JSON already claimed this episode, mark BOTH ambiguous.
        if ep_id in claims:
            entry.status = STATUS_AMBIGUOUS
            entry.match_method = method
            entry.matched_episode_ids = [ep_id]
            entry.notes = (
                f"episode already claimed by {claims[ep_id]!r}; refusing to join"
            )
            # Previous entry promoted to AMBIGUOUS too.
            for prior in report.entries:
                if prior.json_path in claims[ep_id] and prior.status == STATUS_JOINED:
                    prior.status = STATUS_AMBIGUOUS
                    prior.notes = (
                        f"second JSON ({jp}) also claimed this episode; rolled back"
                    )
                    plan.pop(ep_id, None)
            claims[ep_id].append(str(jp))
            report.entries.append(entry)
            continue

        entry.status = STATUS_JOINED
        entry.match_method = method
        entry.matched_episode_ids = [ep_id]
        entry.fields_attached = list(QOR_FLAT_KEYS)
        claims[ep_id] = [str(jp)]
        plan[ep_id] = (ep, flat, best_conf, method)
        report.entries.append(entry)

    # Drop any plan entries whose entry was downgraded to AMBIGUOUS above.
    joined_paths = {e.json_path for e in report.entries if e.status == STATUS_JOINED}
    plan = {
        ep_id: tpl for ep_id, tpl in plan.items()
        if any(p in joined_paths for p in claims.get(ep_id, []))
    }

    if apply and plan:
        # Build the rewritten file: every original line preserved; matched
        # episodes have qor_* keys merged into start_features.  Idempotent.
        ep_index = {ep.get("episode_id"): i for i, ep in enumerate(episodes)}
        modified = 0
        for ep_id, (_, flat, _, _) in plan.items():
            if ep_id not in ep_index:
                continue
            ep_copy = copy.deepcopy(episodes[ep_index[ep_id]])
            changed = _attach_qor_fields(ep_copy, flat)
            if changed:
                episodes[ep_index[ep_id]] = ep_copy
                modified += 1
        if modified:
            lines = [json.dumps(ep, separators=(", ", ": ")) for ep in episodes]
            _atomic_write_lines(episode_store_path, lines)
            report.episode_store_changed = True
            report.episodes_modified = modified

    return report
