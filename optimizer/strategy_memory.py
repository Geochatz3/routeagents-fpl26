"""Strategy memory — RAG seed for the optimizer's iter-1 user prompt.

At iter 1 the LLM otherwise has zero history.  This module loads past-run
records (from prior portfolio campaigns and ongoing runs) and surfaces
the highest-ΔFmax record for the current design — or, when the design is
unknown, the closest fingerprint-matched record.

Sources, in priority order:
  1. STRATEGY_MEMORY_PATH env var → JSONL file written by past runs
  2. ./strategy_memory.jsonl (repo-local, if present)
  3. /mnt/d/fpl26_optimization_contest/analysis/all_10_designs.jsonl
     (the canonical 10-design portfolio campaign data — see
     reference_run_artifacts_path memory)

Record schema (subset of portfolio_results.jsonl, all optional):
  {
    "design": "amd_mini-isp",
    "candidate": "anchor"|"v0_3"|"v0_3_seedN",
    "delta_fmax_mhz": 87.35,
    "initial_fmax_mhz": 295.40,
    "final_fmax_mhz": 382.75,
    "iterations": 5,
    "tool_calls": 41,
    "force_continues": 1,
    "total_cost_usd": 0.11,
    "wall_time_s": 615.0,
    "completed": true
  }

Why JSONL: append-only, line-atomic — concurrent runs from the
scheduler can append without locking.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_BASENAME = "strategy_memory.jsonl"
# In-repo bundled seed memory (ships with the submission).  Loaded at
# eval-time so the LLM has prior-campaign context even on a clean machine
# with no access to dev-side /mnt/d/ artefacts.
SEED_PORTFOLIO_BUNDLED = Path(__file__).parent / "data" / "seed_memory.jsonl"
# Dev-side full path (contains the same data plus possibly newer campaigns).
# Used only when bundled file is missing or shorter — never overrides eval.
SEED_PORTFOLIO_DEV = Path("/mnt/d/fpl26_optimization_contest/analysis/all_10_designs.jsonl")
# Back-compat alias for tests / CLI; first existing path among candidates.
SEED_PORTFOLIO_PATH = SEED_PORTFOLIO_BUNDLED if SEED_PORTFOLIO_BUNDLED.exists() else SEED_PORTFOLIO_DEV


# Per-design curated notes — explicit guidance for designs where the
# campaign-derived hints aren't enough.  Loss / tight-margin designs need
# strategy classes our portfolio didn't explore.  Source: BASELINE_COMPARISON.md.
DESIGN_NOTES = {
    "corescore_500_mod": (
        "LOSS vs published BL by ~25 MHz.  Strategy ceiling at +53 MHz across "
        "anchor + v0_3 + 2 seeds — pblock + LLM-driven phys_opt cannot close the "
        "gap.  Try a NEW class: vivado_phys_opt_design with directive="
        "'AlternateFlowWithRetiming' or 'AddRetime' (register retiming), "
        "or split into multi-PBLOCK floor-plan aligned to critical-path clusters."
    ),
    "finn_radioml": (
        "TIE vs published BL.  Repeated v0_3 seeds gained +8 MHz over single "
        "v0_3 — this design rewards seed variance.  Don't refine one chain; "
        "try multiple distinct strategy classes within the budget."
    ),
    "rosetta_3d-rendering": (
        "TIE vs published BL with narrow gap (publ.+v0_3_seed3 ≈ +7 MHz).  "
        "Don't regress — small wins matter here."
    ),
    "vexriscv_re-place_v2": (
        "Initial Fmax 397.5 MHz — very tight slack, near device ceiling.  "
        "Standard pblock + place + route can't extract more.  Try "
        "vivado_phys_opt_design directive='AggressiveExplore' chains, "
        "register retiming (AlternateFlowWithRetiming), or BRAM/DSP "
        "relocation if critical path crosses a die boundary."
    ),
}


@dataclass
class RunRecord:
    """One past run.  Keep fields simple — JSONL serializable."""
    design: Optional[str] = None
    candidate: Optional[str] = None
    delta_fmax_mhz: Optional[float] = None
    initial_fmax_mhz: Optional[float] = None
    final_fmax_mhz: Optional[float] = None
    iterations: Optional[int] = None
    tool_calls: Optional[int] = None
    force_continues: Optional[int] = None
    total_cost_usd: Optional[float] = None
    wall_time_s: Optional[float] = None
    completed: Optional[bool] = None
    # Coarse fingerprint for fallback matching when name is unknown
    lut_count: Optional[int] = None
    critical_path_spread: Optional[float] = None
    # Tool sequence that produced WNS improvements (in order).  Lets RAG
    # show "the strategy that worked" not just "the candidate that won".
    winning_tools: Optional[List[str]] = None
    # Optional human note — e.g., "force-continue lifted ceiling"
    note: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        # Only keep fields we know about; ignore extras (forward-compat)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _candidate_paths() -> List[Path]:
    """Return memory-file paths in priority order.  All optional — missing
    files are skipped silently.

    Order:
      1. STRATEGY_MEMORY_PATH env var (user override)
      2. ./strategy_memory.jsonl in cwd (per-project memory growing from runs)
      3. optimizer/data/seed_memory.jsonl (bundled with submission)
      4. /mnt/d/.../all_10_designs.jsonl (dev-side; only if bundled missing,
         to avoid loading duplicate records of the same campaign)
    """
    paths: List[Path] = []
    env = os.environ.get("STRATEGY_MEMORY_PATH")
    if env:
        paths.append(Path(env))
    paths.append(Path.cwd() / DEFAULT_MEMORY_BASENAME)
    if SEED_PORTFOLIO_BUNDLED.exists():
        paths.append(SEED_PORTFOLIO_BUNDLED)
    else:
        paths.append(SEED_PORTFOLIO_DEV)
    # De-dupe while preserving order
    seen = set()
    out = []
    for p in paths:
        rp = p.resolve() if p.exists() else p
        if rp not in seen:
            out.append(p)
            seen.add(rp)
    return out


def _load_jsonl(path: Path) -> List[RunRecord]:
    if not path.exists():
        return []
    records = []
    try:
        with path.open() as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(RunRecord.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"strategy_memory: skipping {path}:{line_no} ({e})")
    except OSError as e:
        logger.warning(f"strategy_memory: cannot read {path}: {e}")
    return records


def load_memory(extra_paths: Optional[Iterable[Path]] = None) -> List[RunRecord]:
    """Load all past runs from default + optional extra sources.

    Records are returned in the order encountered (first source first),
    deduplication is *not* performed — callers can choose to use the
    most-recent or highest-ΔFmax record per design.
    """
    paths = _candidate_paths()
    if extra_paths:
        paths = list(extra_paths) + paths
    out: List[RunRecord] = []
    for p in paths:
        out.extend(_load_jsonl(p))
    return out


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def all_for_design(
    memory: List[RunRecord],
    design_name: Optional[str],
) -> List[RunRecord]:
    """All completed records for this design, sorted by delta_fmax descending."""
    if not design_name:
        return []
    matches = [
        r for r in memory
        if r.design == design_name
        and r.delta_fmax_mhz is not None
        and r.completed is not False
    ]
    matches.sort(key=lambda r: -(r.delta_fmax_mhz or 0))
    return matches


def best_for_design(
    memory: List[RunRecord],
    design_name: Optional[str],
) -> Optional[RunRecord]:
    """Highest delta_fmax_mhz record matching this design name.  Returns
    None if no completed record exists for the design."""
    matches = all_for_design(memory, design_name)
    return matches[0] if matches else None


def top_n_for_design(
    memory: List[RunRecord],
    design_name: Optional[str],
    n: int = 3,
) -> List[RunRecord]:
    """Top N records for this design, dedup by candidate (so e.g. v0_3
    seed1/seed2/seed3 don't all crowd the list with similar shape)."""
    matches = all_for_design(memory, design_name)
    seen_candidates = set()
    out: List[RunRecord] = []
    for r in matches:
        # Strip _seedN suffix so different seeds collapse to one candidate class
        cand = (r.candidate or "?").split("_seed")[0]
        if cand in seen_candidates:
            continue
        seen_candidates.add(cand)
        out.append(r)
        if len(out) >= n:
            break
    return out


def global_aggregate(memory: List[RunRecord]) -> str:
    """When no per-design or fingerprint match exists, summarize what
    candidates won across all designs in memory.  Gives the LLM a baseline
    prior even on a totally unknown benchmark (e.g. a new hidden benchmark).

    Returns a one-line summary like:
        "Across 10 prior campaigns: v0_3 won 6×, anchor won 4× (global mean ΔFmax +52.5 MHz)"
    """
    if not memory:
        return ""
    # Per-design winners
    per_design: dict = {}
    for r in memory:
        if not r.design or r.delta_fmax_mhz is None or r.completed is False:
            continue
        cand = (r.candidate or "?").split("_seed")[0]
        cur = per_design.get(r.design)
        if cur is None or (r.delta_fmax_mhz or 0) > (cur.delta_fmax_mhz or 0):
            per_design[r.design] = RunRecord(
                design=r.design, candidate=cand,
                delta_fmax_mhz=r.delta_fmax_mhz,
            )
    if not per_design:
        return ""
    counts: dict = {}
    deltas = []
    for r in per_design.values():
        c = r.candidate or "?"
        counts[c] = counts.get(c, 0) + 1
        if r.delta_fmax_mhz is not None:
            deltas.append(r.delta_fmax_mhz)
    parts = [
        f"Across {len(per_design)} prior campaigns: " +
        ", ".join(f"{c} won {n}×" for c, n in sorted(counts.items(), key=lambda kv: -kv[1])),
    ]
    if deltas:
        parts.append(f"global mean ΔFmax {sum(deltas) / len(deltas):+.1f} MHz")
    return " (".join(parts) + ")"


def fingerprint_match(
    memory: List[RunRecord],
    lut_count: Optional[int],
    spread: Optional[float],
) -> Optional[RunRecord]:
    """Fallback: pick the closest fingerprint match when design name is
    unknown.  Distance = relative-LUT + abs-spread (cheap heuristic).

    Returns the highest-ΔFmax record from the closest design's runs.
    """
    if lut_count is None or spread is None:
        return None
    candidates = [
        r for r in memory
        if r.lut_count is not None
        and r.critical_path_spread is not None
        and r.delta_fmax_mhz is not None
        and r.completed is not False
    ]
    if not candidates:
        return None

    def dist(r: RunRecord) -> float:
        # Normalize LUT distance by 10k so it's roughly comparable to spread tiles
        lut_d = abs((r.lut_count or 0) - lut_count) / 10000.0
        sp_d = abs((r.critical_path_spread or 0) - spread) / 50.0
        return lut_d + sp_d

    return min(candidates, key=lambda r: (dist(r), -(r.delta_fmax_mhz or 0)))


# ---------------------------------------------------------------------------
# Format for prompt
# ---------------------------------------------------------------------------

def format_top_n_for_prompt(
    records: List[RunRecord],
    design_name: Optional[str],
) -> str:
    """Multi-record variant: emit a small comparison table so the LLM
    sees the spread between candidates.  Falls back to single-record
    format when only one match exists.

    Empty string when no records.
    """
    if not records:
        return ""
    if len(records) == 1:
        return format_for_prompt(records[0], design_name)

    header = "PRIOR-CAMPAIGN HISTORY (top candidates for this design):"
    if records[0].design and design_name and records[0].design != design_name:
        header = (
            f"PRIOR-CAMPAIGN HISTORY (closest fingerprint match: "
            f"'{records[0].design}'):"
        )
    lines = [header]
    for r in records:
        bits = []
        if r.candidate:
            bits.append(r.candidate)
        if r.delta_fmax_mhz is not None:
            bits.append(f"ΔFmax={r.delta_fmax_mhz:+.2f} MHz")
        if r.iterations is not None:
            bits.append(f"iters={r.iterations}")
        if r.force_continues:
            bits.append(f"force_continues={r.force_continues}")
        if r.total_cost_usd is not None:
            bits.append(f"cost=${r.total_cost_usd:.3f}")
        if r.wall_time_s is not None:
            bits.append(f"wall={r.wall_time_s:.0f}s")
        lines.append("  - " + ", ".join(bits))
    # Hints from the best record
    hints = _hints_from_record(records[0])
    if hints:
        lines.append("  - HINTS (from best run):")
        for h in hints:
            lines.append(f"    * {h}")
    return "\n".join(lines)


def format_for_prompt(record: Optional[RunRecord], design_name: Optional[str]) -> str:
    """Produce a short, prompt-ready snippet from a memory record.

    Empty string when no record (caller can omit the section entirely).
    Kept terse — every prompt token costs against β.
    """
    if record is None:
        return ""

    parts = ["PRIOR-CAMPAIGN HISTORY (this design's best past run):"]
    if record.design and design_name and record.design != design_name:
        parts[0] = (
            f"PRIOR-CAMPAIGN HISTORY (closest fingerprint match: '{record.design}'):"
        )
    fields = []
    if record.candidate:
        fields.append(f"candidate={record.candidate}")
    if record.delta_fmax_mhz is not None:
        fields.append(f"ΔFmax={record.delta_fmax_mhz:+.2f} MHz")
    if record.iterations is not None:
        fields.append(f"iters={record.iterations}")
    if record.tool_calls is not None:
        fields.append(f"tool_calls={record.tool_calls}")
    if record.force_continues:
        fields.append(f"force_continues={record.force_continues}")
    if record.total_cost_usd is not None:
        fields.append(f"cost=${record.total_cost_usd:.3f}")
    if record.wall_time_s is not None:
        fields.append(f"wall={record.wall_time_s:.0f}s")
    if fields:
        parts.append("  - " + ", ".join(fields))

    # Heuristic guidance derived from the record's shape
    hints = _hints_from_record(record)
    if hints:
        parts.append("  - HINTS:")
        for h in hints:
            parts.append(f"    * {h}")
    return "\n".join(parts)


def _hints_from_record(r: RunRecord) -> List[str]:
    """Translate raw counters into actionable LLM guidance."""
    hints: List[str] = []
    cand = r.candidate or ""
    if cand == "anchor":
        hints.append("anchor strategy worked here — keep PBLOCK-first or fanout if no spread.")
    elif cand.startswith("v0_3"):
        hints.append("v0_3 strategy won here — multi-iteration exploration is profitable.")
    # Seed-specific hint: if a *non-default* seed won, the design is highly
    # sensitive to LLM stochasticity — encourage aggressive variation.
    if cand.startswith("v0_3_seed"):
        hints.append(
            "the winning run was from a non-default seed — this design has "
            "high LLM-stochasticity variance; try multiple distinct strategy "
            "approaches rather than refining a single chain."
        )
    if r.force_continues and r.force_continues > 0:
        hints.append(
            f"force-continue fired {r.force_continues}× and lifted the ceiling — don't stop "
            "after the first plateau if you've not exhausted strategy classes."
        )
    if r.iterations is not None and r.iterations <= 2 and (r.delta_fmax_mhz or 0) > 30:
        hints.append("a single strong strategy converged in ≤2 iters — don't over-iterate.")
    if (r.delta_fmax_mhz or 0) <= 5:
        hints.append(
            "this design's published-BL gap is narrow — small wins are still wins; do "
            "not regress to a worse DCP."
        )
    if r.note:
        hints.append(r.note)
    if r.winning_tools:
        # Show first 6 — that's typically the full winning chain.
        chain = " → ".join(r.winning_tools[:6])
        hints.append(f"prior winning tool sequence: {chain}")
    return hints


_MEASUREMENT_TOOLS = {
    "vivado_report_timing_summary",
    "vivado_get_wns",
    "vivado_report_route_status",
    "vivado_report_utilization_for_pblock",
}


def winning_tools_from_call_details(
    tool_call_details: List[dict],
    initial_wns: Optional[float],
) -> List[str]:
    """Derive the sequence of TRANSFORMATIVE tools that produced WNS
    improvements.

    Algorithm: walk in order, track running best WNS.  When a measurement
    tool detects an improvement, attribute it to the LAST non-measurement
    tool seen before this measurement (the one most likely to have caused
    the change — place, route, phys_opt, recipe_*, optimize_*, etc.).
    Dedup adjacent duplicates (re-running route_design after an unsuccessful
    pass shouldn't bloat the chain) but keep separated repeats.

    Returns up to 12 tool names; pathological logs get truncated.
    """
    if not tool_call_details:
        return []
    best = initial_wns if initial_wns is not None else float("-inf")
    last_transformative: Optional[str] = None
    out: List[str] = []
    for d in tool_call_details:
        if d.get("error"):
            continue
        tn = d.get("tool_name")
        if not tn:
            continue
        if tn not in _MEASUREMENT_TOOLS:
            last_transformative = tn
            continue
        wns = d.get("wns")
        if wns is None:
            continue
        if wns > best:
            best = wns
            if last_transformative and (not out or out[-1] != last_transformative):
                out.append(last_transformative)
                if len(out) >= 12:
                    break
    return out


# ---------------------------------------------------------------------------
# Append (producer side — called at end of every run)
# ---------------------------------------------------------------------------

def append_run(record: RunRecord, path: Optional[Path] = None) -> Optional[Path]:
    """Append one record to the memory JSONL.  Caller passes a fully-
    populated RunRecord; this function only handles the IO.

    Path resolution:
      1. explicit `path` arg
      2. STRATEGY_MEMORY_PATH env var
      3. ./strategy_memory.jsonl in cwd

    Returns the path written to, or None if writing failed.
    """
    if path is None:
        env = os.environ.get("STRATEGY_MEMORY_PATH")
        path = Path(env) if env else (Path.cwd() / DEFAULT_MEMORY_BASENAME)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(record.to_dict()) + "\n")
        return path
    except OSError as e:
        logger.warning(f"strategy_memory: cannot append to {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Top-level convenience: one-shot lookup + format
# ---------------------------------------------------------------------------

def design_note_for(design_name: Optional[str]) -> str:
    """Curated per-design note for known-LOSS / known-tight designs.

    Returns a formatted snippet suitable for appending to the iter-1 user
    message.  Empty string when no curated note exists.
    """
    if not design_name or design_name not in DESIGN_NOTES:
        return ""
    return f"DESIGN-SPECIFIC NOTE: {DESIGN_NOTES[design_name]}"


def seed_prompt_for(
    design_name: Optional[str],
    lut_count: Optional[int] = None,
    critical_path_spread: Optional[float] = None,
    top_n: int = 3,
    *,
    contest_mode: bool = False,
) -> str:
    """Top-level: load memory, pick best matches, return formatted snippet.

    By default returns the top-3 candidates' summary so the LLM sees the
    spread between strategies (e.g. anchor +46 vs v0_3 +53 vs another
    seed +46).  Set top_n=1 for the single-record format.

    Empty string when there's nothing useful to inject — caller can drop
    the entire section without conditional logic in the prompt template.

    `contest_mode=True` enforces hidden-design hygiene per the two
    open hard-rule violations:
      - DESIGN_NOTES (benchmark-specific narratives) is NOT injected
      - exact-name retrieval (`all_for_design` / `best_for_design` /
        `top_n_for_design`) is NOT used as the primary lookup
      - feature-first retrieval (LUT count + critical-path spread)
        is preferred, with global aggregate as the final fallback
    Use this on hidden contest designs the LLM has never seen.
    """
    memory = load_memory()
    if not memory:
        return ""
    # In contest mode the curated note is forbidden — it leaks design
    # name to the LLM and contains benchmark-specific policy hints.
    note = "" if contest_mode else design_note_for(design_name)

    # ---- Contest mode: feature-first only ----
    if contest_mode:
        fp = fingerprint_match(memory, lut_count, critical_path_spread)
        if fp:
            # Re-format without leaking design name as a key.  The
            # `format_for_prompt` helper does NOT echo the design field
            # by default — but the caller can pass design_name=None to
            # suppress the optional header line.
            return format_for_prompt(fp, design_name=None)
        agg = global_aggregate(memory)
        if agg:
            return f"PRIOR-CAMPAIGN HISTORY: {agg}"
        return ""

    # ---- Normal (non-contest) mode: exact-name first ----
    if top_n <= 1:
        record = best_for_design(memory, design_name)
        if record is None:
            record = fingerprint_match(memory, lut_count, critical_path_spread)
        single = format_for_prompt(record, design_name)
        if single:
            return _join_with_note(single, note)
        agg = global_aggregate(memory)
        if agg:
            return _join_with_note(f"PRIOR-CAMPAIGN HISTORY: {agg}", note)
        return note

    records = top_n_for_design(memory, design_name, n=top_n)
    if records:
        return _join_with_note(format_top_n_for_prompt(records, design_name), note)

    fp = fingerprint_match(memory, lut_count, critical_path_spread)
    if fp:
        return _join_with_note(format_for_prompt(fp, design_name), note)

    # Truly unknown — fall back to global aggregate so the LLM still sees
    # *some* historical prior rather than starting from scratch.
    agg = global_aggregate(memory)
    if agg:
        return _join_with_note(f"PRIOR-CAMPAIGN HISTORY: {agg}", note)
    return note


def retrieval_metadata_for(
    design_name: Optional[str],
    lut_count: Optional[int] = None,
    critical_path_spread: Optional[float] = None,
    *,
    contest_mode: bool = False,
) -> dict:
    """Return a compact dict describing what retrieval would inject,
    without rendering the actual prompt text.  Used by the decision
    tracer to record retrieval episodes per-run.

    Schema:
      {
        "rag_mode": "off" | "normal" | "contest_mode",
        "retrieval_mode": "exact_name" | "feature_first"
                          | "global_aggregate" | "none",
        "design_notes_injected": bool,
        "exact_name_used": bool,
        "retrieved_episode_ids": list[str],
        "negative_memory_count": int,
        "memory_records_considered": int,
      }

    `retrieved_episode_ids` are deterministic per-record hashes
    (8-char hex of design+candidate+delta_fmax+wall_time_s) so the
    same physical run record always gets the same id across calls.
    """
    out = {
        "rag_mode": "contest_mode" if contest_mode else "normal",
        "retrieval_mode": "none",
        "design_notes_injected": False,
        "exact_name_used": False,
        "retrieved_episode_ids": [],
        "negative_memory_count": 0,
        "memory_records_considered": 0,
    }
    memory = load_memory()
    out["memory_records_considered"] = len(memory)
    if not memory:
        return out

    if contest_mode:
        # No design_note, no exact-name retrieval.
        fp = fingerprint_match(memory, lut_count, critical_path_spread)
        if fp is not None:
            out["retrieval_mode"] = "feature_first"
            out["retrieved_episode_ids"].append(_episode_id(fp))
        elif global_aggregate(memory):
            out["retrieval_mode"] = "global_aggregate"
    else:
        out["design_notes_injected"] = bool(design_note_for(design_name))
        records = top_n_for_design(memory, design_name, n=3)
        if records:
            out["retrieval_mode"] = "exact_name"
            out["exact_name_used"] = True
            out["retrieved_episode_ids"].extend(_episode_id(r) for r in records)
        else:
            fp = fingerprint_match(memory, lut_count, critical_path_spread)
            if fp is not None:
                out["retrieval_mode"] = "feature_first"
                out["retrieved_episode_ids"].append(_episode_id(fp))
            elif global_aggregate(memory):
                out["retrieval_mode"] = "global_aggregate"

    # Negative memory count (same definition as negative_memory_block)
    def is_negative(r: RunRecord) -> bool:
        if r.completed is False:
            return True
        if r.delta_fmax_mhz is not None and r.delta_fmax_mhz <= 0:
            return True
        if r.note and "regress" in str(r.note).lower():
            return True
        return False
    out["negative_memory_count"] = sum(1 for r in memory if is_negative(r))
    return out


def _episode_id(record: RunRecord) -> str:
    """Stable 8-char hex id for a memory record (design+candidate+
    delta_fmax+wall_time)."""
    import hashlib
    key = "|".join(str(x) for x in (
        record.design,
        record.candidate,
        record.delta_fmax_mhz,
        record.wall_time_s,
    ))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


def negative_memory_block(
    memory: Optional[List[RunRecord]] = None,
    *,
    lut_count: Optional[int] = None,
    critical_path_spread: Optional[float] = None,
    max_items: int = 3,
) -> str:
    """Build an advisory block listing prior episodes that REGRESSED
    or wasted budget on feature-similar designs.

    Contest-mode adjunct: gives the LLM a "what to avoid" hint
    without name-keying.  ADVISORY ONLY — does NOT directly gate
    any tool call.  Empty string when no usable negative records
    exist.

    A record qualifies as negative-memory when any of:
      - completed=False (run did not produce a measurable delta)
      - delta_fmax_mhz is not None AND ≤ 0
      - note explicitly mentions regression/regressed
    """
    if memory is None:
        memory = load_memory()
    if not memory:
        return ""

    def is_negative(r: RunRecord) -> bool:
        if r.completed is False:
            return True
        if r.delta_fmax_mhz is not None and r.delta_fmax_mhz <= 0:
            return True
        if r.note and "regress" in str(r.note).lower():
            return True
        return False

    negative = [r for r in memory if is_negative(r)]
    if not negative:
        return ""

    # Prefer fingerprint-similar negatives if we know our features.
    def similarity(r: RunRecord) -> float:
        if (lut_count is None or critical_path_spread is None
                or r.lut_count is None or r.critical_path_spread is None):
            return 1e9  # unknown distance — sorted last
        lc = max(lut_count, r.lut_count) or 1
        rel_lut = abs(lut_count - r.lut_count) / lc
        ds = abs(critical_path_spread - r.critical_path_spread)
        return rel_lut + ds / 100.0

    negative.sort(key=similarity)
    picked = negative[:max_items]
    if not picked:
        return ""
    bullets = []
    for r in picked:
        cand = r.candidate or "unknown"
        delta = (
            f"Δ{r.delta_fmax_mhz:+.2f} MHz"
            if r.delta_fmax_mhz is not None else "no measurable Δ"
        )
        why = "regressed" if (r.delta_fmax_mhz is not None
                                and r.delta_fmax_mhz < 0) else (
            "no improvement" if r.completed is False else "neutral / inconclusive"
        )
        feat = ""
        if r.lut_count is not None and r.critical_path_spread is not None:
            feat = (
                f" (lut≈{r.lut_count:,}, "
                f"spread≈{r.critical_path_spread:.1f} tiles)"
            )
        bullets.append(f"- candidate={cand} {delta} — {why}{feat}")
    header = (
        "NEGATIVE-MEMORY ADVISORY: feature-similar episodes that "
        "previously regressed or wasted budget.  Do not repeat the "
        "same approach.  This is advisory; you may still choose a "
        "different parameterisation."
    )
    return header + "\n" + "\n".join(bullets)


def _join_with_note(snippet: str, note: str) -> str:
    """Glue a prior-campaign snippet and a curated design note with a blank
    line separator.  Either side may be empty."""
    parts = [s for s in (snippet, note) if s]
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# CLI: inspect what the optimizer will see for a given design
# ---------------------------------------------------------------------------

def _cli(argv: Optional[List[str]] = None) -> int:
    """`python -m optimizer.strategy_memory --query <design>` → print the
    snippet the optimizer would inject for that design.  Useful for
    debugging RAG output without running an MCP-bearing optimizer."""
    import argparse
    parser = argparse.ArgumentParser(description=_cli.__doc__)
    parser.add_argument("--query", help="Design name to inspect (e.g. amd_mini-isp).")
    parser.add_argument("--lut-count", type=int, default=None,
                        help="Fingerprint LUT count for unknown-design fallback.")
    parser.add_argument("--spread", type=float, default=None,
                        help="Fingerprint average critical-path spread (tiles).")
    parser.add_argument("--list", action="store_true",
                        help="List every design with at least one record.")
    parser.add_argument("--count", action="store_true",
                        help="Print the total record count and source paths.")
    parser.add_argument("--verify", action="store_true",
                        help="Validate every record; report anomalies (missing "
                             "design, non-numeric ΔFmax, invalid candidate names).")
    args = parser.parse_args(argv)

    if args.verify:
        records = load_memory()
        problems = []
        valid_count = 0
        for i, r in enumerate(records):
            issues = []
            if not r.design:
                issues.append("missing design")
            if r.delta_fmax_mhz is not None and not isinstance(r.delta_fmax_mhz, (int, float)):
                issues.append(f"delta_fmax_mhz not numeric: {r.delta_fmax_mhz!r}")
            if r.candidate is not None and not isinstance(r.candidate, str):
                issues.append(f"candidate not str: {r.candidate!r}")
            if r.iterations is not None and (
                not isinstance(r.iterations, int) or r.iterations < 0
            ):
                issues.append(f"iterations invalid: {r.iterations!r}")
            if r.total_cost_usd is not None and r.total_cost_usd > 1.0:
                issues.append(
                    f"cost ${r.total_cost_usd:.2f} > $1.00 (contest per-benchmark cap)"
                )
            if issues:
                problems.append((i, r.design or "?", issues))
            else:
                valid_count += 1
        print(f"Verified {len(records)} records: {valid_count} valid, "
              f"{len(problems)} with issues.")
        for i, design, issues in problems[:20]:
            print(f"  record #{i} ({design}):")
            for iss in issues:
                print(f"    - {iss}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
        return 0 if not problems else 2

    if args.count or (not args.query and not args.list):
        records = load_memory()
        sources = [str(p) for p in _candidate_paths() if p.exists()]
        print(f"Loaded {len(records)} records.")
        print("Sources (in priority order):")
        for s in sources:
            print(f"  - {s}")
        if not args.query and not args.list:
            return 0

    if args.list:
        records = load_memory()
        per_design: dict = {}
        for r in records:
            if r.design and r.delta_fmax_mhz is not None:
                cur = per_design.get(r.design)
                if cur is None or (r.delta_fmax_mhz or 0) > (cur.delta_fmax_mhz or 0):
                    per_design[r.design] = r
        for design, r in sorted(per_design.items(),
                                key=lambda kv: -(kv[1].delta_fmax_mhz or 0)):
            print(f"  {design:<32s}  best={r.candidate or '?':<12s}  "
                  f"ΔFmax={r.delta_fmax_mhz:+.2f} MHz  iters={r.iterations}")
        return 0

    snippet = seed_prompt_for(
        design_name=args.query,
        lut_count=args.lut_count,
        critical_path_spread=args.spread,
    )
    if not snippet:
        print(f"(no record found for '{args.query}'; "
              f"lut_count={args.lut_count}, spread={args.spread})")
        return 1
    print(snippet)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
