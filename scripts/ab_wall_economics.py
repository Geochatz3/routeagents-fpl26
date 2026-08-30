#!/usr/bin/env python3
"""Parse attempt-level wall usage and grade paired wall-allocation runs.

Decision and parsing helpers operate without invoking FPGA tools. The
command-line entry point runs each variant and appends an evidence row.

Per-attempt elapsed time comes from restart summaries or wrapper log markers,
never from the scorecard's whole-process wall time. Gamma uses the
whole-process wall time in hours and remains separate from attempt attribution.
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

# Constants (evidence-commented, route_gate.py style)

# Treat fmax deltas below 0.5 MHz as measurement noise and use the same
# threshold for the never-worse check. WNS acceptance follows the same principle.
MEANINGFUL_FMAX_DELTA_MHZ = 0.5

# Never-worse tolerance = the same threshold: a regression at or beyond
# -0.5 MHz fails never-worse AND grades HARMFUL (boundary ties break
# toward HARMFUL — the codebase's locked prefer-refusing bias, see
# optimizer/route_gate.py).
NEVER_WORSE_EPS_MHZ = MEANINGFUL_FMAX_DELTA_MHZ

# DEBUG-WALL convention: local mechanism validation at MAX_WALL 8000-9000
# (~2.5x the eval machine's speed ratio). Behavior-only evidence.
DEBUG_WALL_DEFAULT_S = 8500

# Wrapper log-line grammar (print sites in scripts/multi_restart_optimize.py;
# formats verified verbatim against official harness logs).
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
    """Attribute elapsed wall time and result metadata to individual attempts.

    For each field, prefer the restart summary over wrapper-log inference.
    Infer elapsed time from consecutive remaining-wall markers only when the
    launch budget represents remaining time rather than a split-aware cap.

    Leave elapsed time as `None` when neither source supports attribution;
    never substitute whole-wrapper wall time. Parse budgets only from attempt
    launch lines, and prefer summary values for frequency and status before log
    values.

    Return records sorted by attempt number with `attempt`, `budget_s`,
    `elapsed_s`, `fmax`, `status`, and `elapsed_source`. The source is
    `summary`, `log_delta`, or `None`.
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
    """Number of attempts actually LAUNCHED (len(attempts)) — the
    --split-aware metric: did the split let a second draw fire?"""
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
    """Grade an OFF-baseline vs mechanism-ON pair (behavior evidence only).

    - wall_reclaimed_s (wall handback): OFF total attempt wall minus ON
      total attempt wall (per-attempt attribution) — positive when ON
      hands wall back.
    - attempts_delta (restart split): ON draws minus OFF draws — positive
      when the split let extra attempts fire.
    - never_worse: ON's chosen fmax must stay above OFF's minus eps;
      the -eps boundary itself FAILS (ties break toward HARMFUL, the
      locked prefer-refusing bias).
    - label: NEUTRAL when |delta| < MEANINGFUL_FMAX_DELTA_MHZ, HELPS at
      >= +0.5, HARMFUL at <= -0.5 or when ON lost its output entirely.
    """
    if not off_summary:
        # An absent OFF summary is a failed baseline, not a baseline of
        # None: the `off_fmax is None` branch below would grade every
        # mechanism HELPS/never_worse off a run that never produced a
        # number. Callers must skip grading instead.
        raise ValueError("grade_pair: empty OFF summary — the baseline run "
                         "produced no mr_summary, so there is nothing to "
                         "grade against")
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
        # Local DEBUG-WALL numbers are behavior evidence only; score
        # claims come only from eval-parity runs.
        "evidence_scope": "BEHAVIOR_ONLY_LOCAL_DEBUG_WALL",
    }


# Variant matrix (make-var names pinned by tests against the Makefile's
# $(if $(SPLIT_AWARE),...) / $(if $(WALL_HANDBACK),...) threading).
VARIANTS = {
    "off": (),                                   # baseline, both flags off
    "handback": ("WALL_HANDBACK=1",),            # wall handback only
    "split": ("SPLIT_AWARE=1",),                 # restart split only
    "both": ("WALL_HANDBACK=1", "SPLIT_AWARE=1"),
}


def variant_make_args(variant: str) -> list[str]:
    """make-variable assignments for a named variant (pure)."""
    return list(VARIANTS[variant])


# Thin orchestration (__main__ only — no logic worth unit-testing with
# Vivado mocked; parsing/grading above stays pure).

def run_variant(dcp: Path, variant: str, max_wall: int, repo: Path,
                log_path: Path) -> tuple[dict, str]:
    """Run one `make run_optimizer` variant; return (mr_summary, log text).

    The wrapper writes mr_summary_<stem>.json into repo/.planning_baseline/
    inside a bare try/except — the dir must EXIST or the summary is
    silently dropped (the summary write in multi_restart_optimize.py), so create it
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
    """One markdown evidence row: per-attempt attributed wall, draw count,
    chosen fmax. Behavior evidence only."""
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
    graded_without_baseline = False
    if "off" in summaries:
        if not summaries["off"]:
            # run_variant returns {} for an unreadable summary. Grading
            # against it is worse than not grading: every mechanism would
            # come out HELPS/never_worse on evidence that does not exist.
            graded_without_baseline = True
            msg = ("OFF baseline produced no mr_summary; grading skipped "
                   "(no baseline to compare against)")
            print(f"[ab-wall] ERROR: {msg}", flush=True)
            lines.append(f"<!-- grade skipped: {msg} -->")
        else:
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
    return 1 if graded_without_baseline else 0


if __name__ == "__main__":
    sys.exit(main())
