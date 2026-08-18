#!/usr/bin/env python3
"""04-02 (R-AB): per-attempt wall attribution + DEBUG-WALL A/B evidence
grading for the D3 (--wall-handback) / D4 (--split-aware) flip decision.

Pure decision/parsing logic (parse_attempt_wall, draw_count, grade_pair)
lives at module top, unit-tested without Vivado (tests/test_ab_wall_
economics.py, fixtures = REAL beta/AWS harness-log excerpts). The thin
__main__ below shells out to `make run_optimizer` per variant at DEBUG-WALL
and appends an evidence row — orchestration only, no decisions.

PITFALL 4 (04-RESEARCH.md, verified against scorecard.json + harness logs):
the scorecard's `wall_time_seconds` is the WHOLE wrapper process (setup +
all attempts + winner_polish) — e.g. amd_mini-isp beta: scorecard 2058.17s
vs attempt 1 alone ~1215s + attempt 2 ~797s. Per-attempt attribution MUST
come from mr_summary_<stem>.json's attempts[].elapsed and/or the wrapper's
own `[multi-restart]` log lines; this module NEVER reads the scorecard
number for per-attempt work.

PITFALL 5: gamma = wall_time_seconds/3600 of the whole wrapper — A/B gamma
deltas must reconcile with the per-attempt rows this module emits; the
whole-wrapper number is reported alongside (from the log-derived remaining
markers) but never substituted for a per-attempt value.

EVIDENCE SCOPE (T-04-03): everything produced locally under the DEBUG-WALL
convention (MAX_WALL 8000-9000, ~2.5x eval speed) is BEHAVIOR evidence
only — never a score claim. Score confirmation rides AWS rehearsal #1.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants (evidence-commented, route_gate.py style)
# ---------------------------------------------------------------------------

# Meaningfulness threshold for an fmax delta between paired A/B runs.
# Prior-sessions convention, applied unchanged: S21 (2026-05-23) graded
# vtr_mcml A 65.57 vs B 65.32 (|d|=0.25 < 0.5) CARD_NEUTRAL; S22 graded
# spam-filter B 437.45 vs A 442.28 (-4.83) HARMFUL. jul06's
# meaningful_accept_ns gate follows the same "don't act on sub-noise
# deltas" philosophy on the WNS side.
MEANINGFUL_FMAX_DELTA_MHZ = 0.5

# Never-worse tolerance = the same threshold: a regression at or beyond
# -0.5 MHz fails never-worse AND grades HARMFUL (boundary ties break
# toward HARMFUL — the codebase's locked prefer-refusing bias, see
# optimizer/route_gate.py).
NEVER_WORSE_EPS_MHZ = MEANINGFUL_FMAX_DELTA_MHZ

# DEBUG-WALL convention (user-endorsed jul04, feedback_debug_wall_convention):
# local mechanism validation at MAX_WALL 8000-9000 (~2.5x the eval box's
# speed ratio). Behavior-only evidence.
DEBUG_WALL_DEFAULT_S = 8500

# ---------------------------------------------------------------------------
# Wrapper log-line grammar (print sites in scripts/multi_restart_optimize.py;
# formats verified verbatim against the official beta harness logs at
# final_round/beta_final_results/logs_x/ and the jul19 AWS boom leg).
# ---------------------------------------------------------------------------
_BUDGET_RE = re.compile(
    r"^\[multi-restart\] attempt (\d+): budget (\d+(?:\.\d+)?)s -> ", re.M)
_RESULT_RE = re.compile(
    r"^\[multi-restart\] attempt (\d+) -> fmax=(\S+) status=(\S+)", re.M)
_STOP_REMAINING_RE = re.compile(
    r"^\[multi-restart\] stop: remaining (\d+(?:\.\d+)?)s < floor", re.M)
_STRANDED_RE = re.compile(
    r"^\[multi-restart\] winner-polish: (\d+(?:\.\d+)?)s stranded", re.M)
_SPLIT_CAP_RE = re.compile(
    r"^\[multi-restart\] split-aware: attempt (\d+) capped to", re.M)


def _to_float(tok: str) -> Optional[float]:
    if tok is None or tok == "None":
        return None
    try:
        return float(tok)
    except ValueError:
        return None


def parse_attempt_wall(mr_summary: dict, harness_log_text: str) -> list[dict]:
    """Attribute wall per attempt (Pitfall 4).

    Sources, in preference order per field:
    - elapsed_s: mr_summary attempts[].elapsed (agent-side runtime) when
      present; else log-only inference from the wrapper's remaining-wall
      markers: attempt i's remaining-at-launch is its `budget Xs` line
      (UNLESS a split-aware cap line marks that budget as a cap, not
      `remaining`), and remaining after the last attempt comes from the
      `stop: remaining Xs < floor` or `winner-polish: Xs stranded` line.
      elapsed_i = remaining_i - remaining_{i+1}. When neither source can
      attribute an attempt, elapsed_s is None — NEVER fabricated, and
      NEVER taken from the scorecard's whole-wrapper wall_time_seconds.
    - budget_s: the `attempt N: budget Xs` log line (None without a log).
    - fmax/status: mr_summary record first, else the `attempt N -> fmax=
      ... status=...` log line.

    Returns [{attempt, budget_s, elapsed_s, fmax, status, elapsed_source}]
    sorted by attempt; elapsed_source is "summary", "log_delta" or None.
    """
    text = harness_log_text or ""
    budgets = {int(m.group(1)): float(m.group(2))
               for m in _BUDGET_RE.finditer(text)}
    results = {int(m.group(1)): (_to_float(m.group(2)),
                                 None if m.group(3) == "None" else m.group(3))
               for m in _RESULT_RE.finditer(text)}
    capped = {int(m.group(1)) for m in _SPLIT_CAP_RE.finditer(text)}
    # Remaining wall after the FINAL attempt, if the log recorded it.
    m_stop = _STOP_REMAINING_RE.search(text)
    m_strand = _STRANDED_RE.search(text)
    remaining_after_last = (float(m_stop.group(1)) if m_stop
                            else float(m_strand.group(1)) if m_strand
                            else None)

    summary_by_i = {a.get("i"): a
                    for a in (mr_summary.get("attempts") or [])
                    if isinstance(a, dict) and a.get("i") is not None}

    order = sorted(set(summary_by_i) | set(budgets) | set(results))
    rows: list[dict] = []
    for pos, i in enumerate(order):
        rec = summary_by_i.get(i, {})
        elapsed = rec.get("elapsed")
        source: Optional[str] = "summary" if elapsed is not None else None
        if elapsed is None:
            # Log-delta inference. A budget line equals remaining-at-launch
            # ONLY when that attempt was not split-aware capped.
            r_i = budgets.get(i) if i not in capped else None
            nxt = order[pos + 1] if pos + 1 < len(order) else None
            if nxt is not None:
                r_next = budgets.get(nxt) if nxt not in capped else None
            else:
                r_next = remaining_after_last
            if r_i is not None and r_next is not None:
                elapsed = r_i - r_next
                source = "log_delta"
        fmax = rec.get("fmax")
        status = rec.get("status")
        log_fmax, log_status = results.get(i, (None, None))
        if fmax is None:
            fmax = log_fmax
        if status is None:
            status = log_status
        rows.append({"attempt": i, "budget_s": budgets.get(i),
                     "elapsed_s": elapsed, "fmax": fmax, "status": status,
                     "elapsed_source": source})
    return rows


def draw_count(mr_summary: dict) -> int:
    """Number of attempts actually LAUNCHED (len(attempts)) — the D4
    metric: did the split let a second draw fire?"""
    return len(mr_summary.get("attempts") or [])


def _attempt_wall_s(mr_summary: dict) -> Optional[float]:
    """Sum of per-attempt elapsed from the summary (attribution basis,
    NOT the scorecard whole-wrapper number)."""
    vals = [a.get("elapsed") for a in (mr_summary.get("attempts") or [])
            if isinstance(a, dict) and a.get("elapsed") is not None]
    return float(sum(vals)) if vals else None


def _chosen_fmax(mr_summary: dict) -> Optional[float]:
    chosen = mr_summary.get("chosen")
    return chosen.get("fmax") if isinstance(chosen, dict) else None


def grade_pair(off_summary: dict, on_summary: dict, *,
               eps_mhz: float = NEVER_WORSE_EPS_MHZ) -> dict:
    """Grade an OFF-baseline vs mechanism-ON pair (BEHAVIOR evidence only).

    - wall_reclaimed_s (D3): OFF total attempt wall minus ON total attempt
      wall (per-attempt attribution, Pitfall 4) — positive when ON hands
      wall back.
    - attempts_delta (D4): ON draws minus OFF draws — positive when the
      split let extra attempts fire.
    - never_worse: ON's chosen fmax must stay above OFF's minus eps;
      the -eps boundary itself FAILS (ties break toward HARMFUL, the
      locked prefer-refusing bias).
    - label: NEUTRAL when |delta| < MEANINGFUL_FMAX_DELTA_MHZ (S21
      convention), HELPS at >= +0.5, HARMFUL at <= -0.5 or when ON lost
      its output entirely.
    """
    off_fmax = _chosen_fmax(off_summary)
    on_fmax = _chosen_fmax(on_summary)
    off_wall = _attempt_wall_s(off_summary)
    on_wall = _attempt_wall_s(on_summary)

    delta: Optional[float] = None
    if off_fmax is None and on_fmax is None:
        label, never_worse = "NEUTRAL", True
    elif on_fmax is None:
        # ON produced no usable output where OFF did — worst outcome.
        label, never_worse = "HARMFUL", False
    elif off_fmax is None:
        label, never_worse = "HELPS", True
    else:
        delta = on_fmax - off_fmax
        never_worse = delta > -eps_mhz
        if delta >= MEANINGFUL_FMAX_DELTA_MHZ:
            label = "HELPS"
        elif delta <= -MEANINGFUL_FMAX_DELTA_MHZ:
            label = "HARMFUL"
        else:
            label = "NEUTRAL"

    return {
        "off_fmax": off_fmax,
        "on_fmax": on_fmax,
        "fmax_delta_mhz": delta,
        "off_attempts": draw_count(off_summary),
        "on_attempts": draw_count(on_summary),
        "attempts_delta": draw_count(on_summary) - draw_count(off_summary),
        "off_attempt_wall_s": off_wall,
        "on_attempt_wall_s": on_wall,
        "wall_reclaimed_s": (off_wall - on_wall
                             if off_wall is not None and on_wall is not None
                             else None),
        "never_worse": never_worse,
        "label": label,
        # T-04-03 mitigation: local DEBUG-WALL numbers are behavior
        # evidence only; score claims come only from eval/AWS.
        "evidence_scope": "BEHAVIOR_ONLY_LOCAL_DEBUG_WALL",
    }


# ---------------------------------------------------------------------------
# Variant matrix (kill switches from 04-01; make-var names pinned by tests
# against Makefile:336-337's $(if $(SPLIT_AWARE),...) / $(if $(WALL_HANDBACK),
# ...) threading).
# ---------------------------------------------------------------------------
VARIANTS = {
    "off": (),                                   # baseline, both flags off
    "handback": ("WALL_HANDBACK=1",),            # D3 only
    "split": ("SPLIT_AWARE=1",),                 # D4 only
    "both": ("WALL_HANDBACK=1", "SPLIT_AWARE=1"),
}


def variant_make_args(variant: str) -> list[str]:
    """make-variable assignments for a named variant (pure)."""
    return list(VARIANTS[variant])


# ---------------------------------------------------------------------------
# Thin orchestration (__main__ only — no logic worth unit-testing with
# Vivado mocked; parsing/grading above stays pure).
# ---------------------------------------------------------------------------

def run_variant(dcp: Path, variant: str, max_wall: int, repo: Path,
                log_path: Path) -> tuple[dict, str]:
    """Run one `make run_optimizer` variant; return (mr_summary, log text).

    The wrapper writes mr_summary_<stem>.json into repo/.planning_baseline/
    inside a bare try/except — the dir must EXIST or the summary is
    silently dropped (multi_restart_optimize.py:500-504), so create it
    first. Stdout+stderr are captured to log_path for the log-side
    attribution channel.
    """
    (repo / ".planning_baseline").mkdir(exist_ok=True)
    stem = dcp.stem
    summary_path = repo / ".planning_baseline" / f"mr_summary_{stem}.json"
    # Remove a stale summary so a failed run can't masquerade as fresh.
    try:
        summary_path.unlink(missing_ok=True)
    except OSError:
        pass
    cmd = ["make", "run_optimizer", f"DCP={dcp}", f"MAX_WALL={max_wall}",
           *variant_make_args(variant)]
    print(f"[ab-wall] {variant}: {' '.join(cmd)}", flush=True)
    t0 = time.monotonic()
    p = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
    wrapper_wall = time.monotonic() - t0
    log_text = (p.stdout or "") + (p.stderr or "")
    log_path.write_text(log_text)
    print(f"[ab-wall] {variant}: rc={p.returncode} "
          f"wrapper_wall={wrapper_wall:.0f}s log -> {log_path}", flush=True)
    summary: dict = {}
    try:
        summary = json.loads(summary_path.read_text())
    except Exception as e:
        print(f"[ab-wall] WARNING: no mr_summary for {variant} ({e}); "
              f"falling back to log-only attribution", flush=True)
    return summary, log_text


def evidence_row(design: str, variant: str, summary: dict,
                 log_text: str) -> str:
    """One markdown evidence row: per-attempt attributed wall (Pitfall 4),
    draw count, chosen fmax. BEHAVIOR evidence only."""
    rows = parse_attempt_wall(summary, log_text)
    per_attempt = "; ".join(
        f"a{r['attempt']}={r['elapsed_s']:.0f}s({r['elapsed_source']})"
        if r["elapsed_s"] is not None else f"a{r['attempt']}=?"
        for r in rows) or "-"
    fmax = _chosen_fmax(summary)
    return (f"| {design} | {variant} | {len(rows)} | {per_attempt} | "
            f"{fmax if fmax is not None else '-'} | BEHAVIOR-ONLY |")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dcp", type=Path, help="input benchmark DCP")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS),
                    choices=list(VARIANTS),
                    help="variant matrix to run (default: all four)")
    ap.add_argument("--max-wall", type=int, default=DEBUG_WALL_DEFAULT_S,
                    help="DEBUG-WALL budget (convention: 8000-9000s, "
                         "~2.5x eval speed; behavior evidence only)")
    ap.add_argument("--repo", type=Path,
                    default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--out", type=Path, default=None,
                    help="markdown file to append evidence rows to")
    a = ap.parse_args(argv)

    design = a.dcp.stem
    header = (f"\n<!-- ab_wall_economics {design} MAX_WALL={a.max_wall} "
              f"(DEBUG-WALL, BEHAVIOR evidence only) -->\n"
              f"| design | variant | draws | per-attempt wall (attributed) "
              f"| chosen fmax | scope |\n|---|---|---|---|---|---|\n")
    lines = []
    summaries: dict[str, dict] = {}
    for variant in a.variants:
        log_path = Path("/tmp") / f"ab_wall_{design}_{variant}_{os.getpid()}.log"
        summary, log_text = run_variant(a.dcp, variant, a.max_wall,
                                        a.repo, log_path)
        summaries[variant] = summary
        row = evidence_row(design, variant, summary, log_text)
        print(f"[ab-wall] {row}", flush=True)
        lines.append(row)
    if "off" in summaries:
        for variant in a.variants:
            if variant == "off" or not summaries[variant]:
                continue
            verdict = grade_pair(summaries["off"], summaries[variant])
            print(f"[ab-wall] grade off-vs-{variant}: "
                  f"{json.dumps(verdict)}", flush=True)
            lines.append(f"<!-- grade off-vs-{variant}: "
                         f"{json.dumps(verdict)} -->")
    if a.out is not None:
        if a.out.is_dir():
            a.out = a.out / f"ab_evidence_{design}.md"
        with a.out.open("a") as fh:
            fh.write(header + "\n".join(lines) + "\n")
        print(f"[ab-wall] evidence appended -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
