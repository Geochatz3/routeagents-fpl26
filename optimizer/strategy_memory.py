"""Provide fingerprint-based strategy memory for the optimizer's initial prompt.

Records are matched by structural features such as LUT count and critical-path
spread, never by design name. When no close fingerprint exists, retrieval falls
back to an aggregate of available records. Sources are checked in priority
order: the path configured by `STRATEGY_MEMORY_PATH`, a repository-local JSONL
file, and an optional external seed file. Record fields are optional to support
partial histories. JSONL provides append-only, line-atomic storage so
concurrent workers can append without locking.
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
# In-repo bundled seed memory (optional; absent in the public release).
# Loaded at eval-time so the LLM has prior-campaign context even on a
# clean machine.
SEED_PORTFOLIO_BUNDLED = Path(__file__).parent / "data" / "seed_memory.jsonl"
# Optional external seed portfolio (e.g. an archive of past campaigns).
# Used only when the bundled file is missing — never overrides eval.
SEED_PORTFOLIO_DEV = Path(
    os.environ.get("FPL26_SEED_PORTFOLIO", "") or "seed_portfolio.jsonl")
# Back-compat alias for tests / CLI; first existing path among candidates.
SEED_PORTFOLIO_PATH = SEED_PORTFOLIO_BUNDLED if SEED_PORTFOLIO_BUNDLED.exists() else SEED_PORTFOLIO_DEV


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
        # Keep only known fields; ignore extras (forward-compat)
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
      4. FPL26_SEED_PORTFOLIO (external seed; only if bundled missing,
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
    """Load and return strategy-memory records from default and optional sources.

    Records retain source and file order, with earlier sources appearing first.
    Duplicates are preserved so callers can choose their own recency or quality policy.
    """
    paths = _candidate_paths()
    if extra_paths:
        paths = list(extra_paths) + paths
    out: List[RunRecord] = []
    for p in paths:
        out.extend(_load_jsonl(p))
    return out


# Query

def global_aggregate(memory: List[RunRecord]) -> str:
    """Summarize candidate performance across all stored designs.

    Used as a baseline prior when neither a per-design nor fingerprint match
    exists. Returns a one-line summary of win counts and the global mean
    frequency change in MHz.
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
    """Find the strongest record associated with the closest fingerprint.

    Distance is the sum of relative LUT difference and absolute spread
    difference. Returns the record with the highest frequency improvement among
    runs for the closest matching design.
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


# Format for prompt

def format_for_prompt(record: Optional[RunRecord], design_name: Optional[str]) -> str:
    """Format a memory record as a compact prompt snippet.

    Returns an empty string when no record is available, allowing the caller to
    omit the section. The representation stays terse to limit prompt cost.
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


# Append (producer side — called at end of every run)

def append_run(record: RunRecord, path: Optional[Path] = None) -> Optional[Path]:
    """Append a populated run record to the strategy-memory JSONL file.

    The output path is resolved from the explicit path, then
    `STRATEGY_MEMORY_PATH`, then `strategy_memory.jsonl` in the current
    directory. Returns the written path, or `None` if writing fails.
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


# Top-level convenience: one-shot lookup + format

def seed_prompt_for(
    design_name: Optional[str] = None,
    lut_count: Optional[int] = None,
    critical_path_spread: Optional[float] = None,
    top_n: int = 3,
    *,
    contest_mode: bool = False,
) -> str:
    """Top-level: load memory, pick the closest fingerprint match, return
    a formatted snippet.

    Retrieval is fingerprint-only (LUT count + critical-path spread);
    when no fingerprint match exists the global aggregate of past
    campaigns is used as the fallback.  `design_name`, `top_n` and
    `contest_mode` are accepted for call-site compatibility but no
    longer affect retrieval — every mode gets the same feature-first
    behavior the contest ship path used.

    Empty string when there's nothing useful to inject — caller can drop
    the entire section without conditional logic in the prompt template.
    """
    del design_name, top_n, contest_mode  # retrieval never keys on these
    memory = load_memory()
    if not memory:
        return ""
    fp = fingerprint_match(memory, lut_count, critical_path_spread)
    if fp:
        # Format without echoing the matched record's design name as a
        # header key (design_name=None suppresses the optional header).
        return format_for_prompt(fp, design_name=None)
    agg = global_aggregate(memory)
    if agg:
        return f"PRIOR-CAMPAIGN HISTORY: {agg}"
    return ""


def _is_negative(r: RunRecord) -> bool:
    """A run counts as negative evidence: it did not complete, it did not
    gain, or its note says it regressed.

    One definition, deliberately: this was inlined twice with identical
    bodies and a comment asserting they matched, which is the shape a
    silent divergence hides in."""
    if r.completed is False:
        return True
    if r.delta_fmax_mhz is not None and r.delta_fmax_mhz <= 0:
        return True
    if r.note and "regress" in str(r.note).lower():
        return True
    return False


def retrieval_metadata_for(
    design_name: Optional[str] = None,
    lut_count: Optional[int] = None,
    critical_path_spread: Optional[float] = None,
    *,
    contest_mode: bool = False,
) -> dict:
    """Describe a retrieval episode without rendering prompt text.

    The returned mapping contains the RAG and retrieval modes, compatibility
    flags, retrieved episode IDs, negative-memory count, and number of records
    considered. Retrieval is fingerprint-only, so `design_notes_injected` and
    `exact_name_used` are always false.

    Episode IDs are deterministic eight-character hexadecimal hashes of the
    record's design, candidate, frequency delta, and wall time.
    """
    del design_name  # retrieval never keys on the design name
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

    fp = fingerprint_match(memory, lut_count, critical_path_spread)
    if fp is not None:
        out["retrieval_mode"] = "feature_first"
        out["retrieved_episode_ids"].append(_episode_id(fp))
    elif global_aggregate(memory):
        out["retrieval_mode"] = "global_aggregate"

    # Negative memory count.
    out["negative_memory_count"] = sum(1 for r in memory if _is_negative(r))
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
    """Build an advisory prompt block from unsuccessful feature-similar episodes.

    The block does not gate tool calls and is empty when no usable records
    qualify. A record qualifies if it is incomplete, has a nonpositive
    frequency delta, or explicitly notes a regression.
    """
    if memory is None:
        memory = load_memory()
    if not memory:
        return ""


    negative = [r for r in memory if _is_negative(r)]
    if not negative:
        return ""

    # Prefer fingerprint-similar negatives when the features are known.
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


# CLI: inspect what the optimizer will see for a given design

def _cli(argv: Optional[List[str]] = None) -> int:
    """`python -m optimizer.strategy_memory --query --lut-count N --spread S`
    → print the snippet the optimizer would inject for that fingerprint.
    Useful for debugging RAG output without running an MCP-bearing
    optimizer.  Retrieval is fingerprint-only; the query is the
    (lut_count, spread) pair, never a design name."""
    import argparse
    parser = argparse.ArgumentParser(description=_cli.__doc__)
    parser.add_argument("--query", action="store_true",
                        help="Print the snippet retrieval would inject for "
                             "the fingerprint given by --lut-count/--spread.")
    parser.add_argument("--lut-count", type=int, default=None,
                        help="Fingerprint LUT count.")
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
        lut_count=args.lut_count,
        critical_path_spread=args.spread,
    )
    if not snippet:
        print(f"(no record found for fingerprint "
              f"lut_count={args.lut_count}, spread={args.spread})")
        return 1
    print(snippet)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
