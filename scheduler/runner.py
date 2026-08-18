"""
Best-of-N candidate scheduler — runs two or more dcp_optimizer.py invocations
sequentially within a wall-budget, returns the best valid output DCP.

Design (per FINAL_DEV_ROADMAP.md P0):
  Phase A: anchor candidate    — time budget = min(20 min, total_budget/3)
  Phase B: v0_3 candidate      — time budget = remaining - 60 s safety
  Phase C (optional): repeat   — additional v0_3 seeds if budget allows
                                 (per FINAL_DEV_ROADMAP P2)

Each candidate runs as a `dcp_optimizer.py` subprocess with its own output
DCP path.  The scheduler parses each run's `run_summary.txt` to get final
WNS / Fmax, picks the highest final_Fmax among the runs that produced a
DCP file, copies that DCP to the requested output path, and exits 0.

If no candidate produced a valid DCP within budget, exit 1 and leave no
output (matches dcp_optimizer.py's existing "no DCP unless improved"
contract).

Anchor vs v0_3 mode toggle:
  This branch's dcp_optimizer.py is the v0_3-controller version.  To
  emulate anchor mode without forking the codebase, the scheduler passes
  --mode-flag combinations that disable v0_3's force-continue and
  unconditional cap.  This requires a small (<30 LOC) change to
  dcp_optimizer.py to read those flags.  See FINAL_DEV_ROADMAP P0
  subtask 1.

Until that toggle lands, this module runs two v0_3 seeds (matches the
"repeated-seed" path of D22).  The replay test at
`scheduler/test_replay.py` proves the scheduler's selection logic is
correct against the existing campaign data even before the toggle lands.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JAVA_HOME bootstrap (mirrors the Makefile inference)
# ---------------------------------------------------------------------------
# When the scheduler is invoked directly (not via `make run_optimizer`),
# JAVA_HOME isn't auto-set.  RapidWright needs it to find libjvm.so.
# Replicate the Makefile's fallback: PATH java first, then Vivado-bundled
# JRE (jre11* preferred, fall back to any jre*).

def _ensure_java_home() -> None:
    if os.environ.get("JAVA_HOME"):
        return  # caller already set it

    # 1) PATH java
    java_path = shutil.which("java")
    if java_path:
        try:
            real = os.path.realpath(java_path)
            jh = os.path.dirname(os.path.dirname(real))
            if Path(jh).exists():
                os.environ["JAVA_HOME"] = jh
                logger.debug(f"JAVA_HOME from PATH java: {jh}")
                return
        except Exception:
            pass

    # 2) Vivado-bundled JRE
    vivado_exec = os.environ.get("VIVADO_EXEC") or shutil.which("vivado")
    if vivado_exec:
        vivado_root = Path(vivado_exec).resolve().parent.parent
        for pattern in ("tps/lnx64/jre11*", "tps/lnx64/jre*"):
            matches = sorted(glob.glob(str(vivado_root / pattern / "bin/java")))
            if matches:
                jh = str(Path(matches[0]).parent.parent)
                os.environ["JAVA_HOME"] = jh
                os.environ["PATH"] = f"{jh}/bin:{os.environ.get('PATH', '')}"
                logger.info(f"JAVA_HOME derived from Vivado: {jh}")
                return

    logger.warning(
        "JAVA_HOME could not be derived; RapidWright will fail to load. "
        "Set JAVA_HOME or VIVADO_EXEC explicitly before running the scheduler."
    )


# ---------------------------------------------------------------------------
# Per-run data captured from the optimizer
# ---------------------------------------------------------------------------

@dataclass
class CandidateResult:
    """Outcome of one dcp_optimizer.py invocation."""
    candidate_name: str
    output_dcp: Path
    exit_code: int
    wall_time_s: float
    initial_fmax_mhz: Optional[float] = None
    final_wns_ns: Optional[float] = None
    final_fmax_mhz: Optional[float] = None
    delta_fmax_mhz: Optional[float] = None
    iterations: Optional[int] = None
    cost_usd: Optional[float] = None
    log_path: Optional[Path] = None

    @property
    def is_valid(self) -> bool:
        """A run is valid if it exited 0 AND produced a DCP file AND the run
        log reported a final Fmax."""
        return (
            self.exit_code == 0
            and self.output_dcp.exists()
            and self.final_fmax_mhz is not None
        )


# ---------------------------------------------------------------------------
# Output-summary parsers (same patterns as portfolio_runner.sh emit_row)
# ---------------------------------------------------------------------------

_PATTERNS = {
    "initial_fmax_mhz": re.compile(r"Initial Fmax:\s+([\d\.]+)\s*MHz"),
    "final_fmax_mhz":   re.compile(r"Best Fmax:\s+([\d\.]+)\s*MHz"),
    "delta_fmax_mhz":   re.compile(r"Fmax Improvement:\s+([+\-\d\.]+)\s*MHz"),
    "iterations":       re.compile(r"Total iterations:\s+(\d+)"),
    "cost_usd":         re.compile(r"Total cost:\s+\$([\d\.]+)"),
}


def parse_run_log(log_path: Path) -> dict:
    """Pull headline metrics from a dcp_optimizer.py stdout log.  Robust to
    early-aborts (keys missing → values stay None)."""
    out = {k: None for k in _PATTERNS}
    if not log_path.exists():
        return out
    text = log_path.read_text(errors="replace")
    for k, pat in _PATTERNS.items():
        m = pat.search(text)
        if not m:
            continue
        try:
            v = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
        except ValueError:
            v = m.group(1)
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# Candidate launcher
# ---------------------------------------------------------------------------

def parse_recipe_log(log_path: Path) -> dict:
    """Parse the JSON blob recipes/cell_replacement.py prints to stdout
    on completion.  Recipe stdout ends with a pretty-printed JSON object.
    We grab the last balanced { ... } block in the log and read that.

    Returns the same shape as parse_run_log so the two parsers are
    interchangeable for CandidateResult population.  Recipe doesn't
    track LLM cost (always $0); iterations is always 1.
    """
    out = {k: None for k in _PATTERNS}
    if not log_path.exists():
        return out
    text = log_path.read_text(errors="replace")
    # Find last "{" through matching "}" — recipe prints exactly one JSON.
    start = text.rfind("{\n")
    if start < 0:
        start = text.rfind("{")
    if start < 0:
        return out
    blob = text[start:].strip()
    # Trim trailing whitespace / newlines
    end = blob.rfind("}")
    if end < 0:
        return out
    blob = blob[: end + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return out
    out["initial_fmax_mhz"] = data.get("baseline_fmax_mhz")
    out["final_fmax_mhz"]   = data.get("final_fmax_mhz")
    out["delta_fmax_mhz"]   = data.get("delta_fmax_mhz")
    out["iterations"]       = 1
    out["cost_usd"]         = 0.0
    return out


def run_candidate(
    candidate_name: str,
    input_dcp: Path,
    output_dcp: Path,
    log_path: Path,
    time_budget_s: int,
    extra_args: Optional[List[str]] = None,
    optimizer_path: Path = Path(__file__).parent.parent / "dcp_optimizer.py",
) -> CandidateResult:
    """Spawn dcp_optimizer.py (or recipes/cell_replacement.py for the
    'recipe' candidate) as a subprocess with a wall-time cap.  Captures
    stdout/stderr to log_path.  Returns a CandidateResult."""
    if candidate_name == "recipe":
        recipe_path = Path(__file__).parent.parent / "recipes" / "cell_replacement.py"
        args = [
            sys.executable, "-u", str(recipe_path),
            str(input_dcp), "-o", str(output_dcp),
        ]
        if extra_args:
            args.extend(extra_args)
        parser = parse_recipe_log
    else:
        args = [
            sys.executable, "-u", str(optimizer_path),
            str(input_dcp), "-o", str(output_dcp),
        ]
        if extra_args:
            args.extend(extra_args)
        parser = parse_run_log

    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{candidate_name}] launching with budget {time_budget_s}s")
    start = time.time()

    with log_path.open("w") as logf:
        try:
            proc = subprocess.run(
                args,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=time_budget_s,
            )
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            exit_code = 124  # standard timeout exit code
            logf.write(f"\n[scheduler] candidate killed after {time_budget_s}s\n")

    wall = time.time() - start
    parsed = parser(log_path)

    return CandidateResult(
        candidate_name=candidate_name,
        output_dcp=output_dcp,
        exit_code=exit_code,
        wall_time_s=wall,
        log_path=log_path,
        **parsed,
    )


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

@dataclass
class SchedulerConfig:
    """Knobs for the scheduler.  All defaults derive from D22 evidence.

    `candidates` is the candidate ordering.  When None (the default),
    the runner consults `scheduler.dispatch.candidates_for(design_name)`
    (or `candidates_with_recipe` when `include_recipe=True`) to pick a
    per-design ordering — see dispatch.py for the table.  Pass an
    explicit list to override (e.g. for ablation experiments).

    `include_recipe` opts in to the recipe slot.  Off by default until
    we have empirical ΔFmax data per design — the recipe's value vs
    LLM candidates is currently only known on vexriscv (recipe loses
    to anchor +125 vs +76).  Use `--include-recipe` to A/B.
    """
    total_budget_s: int = 3600  # 60 min hard cap (contest's gamma_capped at 1h)
    min_first_candidate_s: int = 600   # at least 10 min for first run
    max_first_candidate_s: int = 1200  # cap first run at 20 min
    min_repeated_seed_s: int = 480     # only fire a repeated-seed slot if ≥ 8 min remain
    safety_margin_s: int = 60          # leave 60 s slack
    candidates: Optional[List[str]] = None
    include_recipe: bool = False
    recipe_max_wall_s: int = 600       # 10-min cap on the recipe slot
    # Repeated-seed feature (P2 in FINAL_DEV_ROADMAP).  After the configured
    # candidates finish, fill remaining minutes with additional v0_3
    # invocations.  Each relies on LLM stochasticity for variance.
    repeated_seeds: int = 0            # 0 = off; N = up to N extra seeds


def select_best(results: List[CandidateResult]) -> Optional[CandidateResult]:
    """Highest final_fmax_mhz among valid candidates.  Tie-break on lower wall
    time, then lower cost.  Returns None if no candidate is valid."""
    valid = [r for r in results if r.is_valid]
    if not valid:
        return None
    return max(
        valid,
        key=lambda r: (
            r.final_fmax_mhz,
            -r.wall_time_s,
            -(r.cost_usd or 0),
        ),
    )


def run_scheduled(
    input_dcp: Path,
    output_dcp: Path,
    config: Optional[SchedulerConfig] = None,
    work_dir: Optional[Path] = None,
) -> int:
    """Top-level entrypoint.  Run candidates sequentially within budget,
    select best, copy to output_dcp.  Return process-style exit code."""
    from .dispatch import (
        candidates_for,
        candidates_with_recipe,
        design_name_from_dcp,
    )

    config = config or SchedulerConfig()
    work_dir = work_dir or Path(f"./scheduler_run-{int(time.time())}")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Make sure JAVA_HOME is set before we spawn child optimizer processes —
    # otherwise RapidWright will fail to load libjvm.so in each child.
    _ensure_java_home()

    # Resolve candidate ordering: explicit override on config, else
    # per-design dispatch table (with optional recipe prepend).
    candidates = config.candidates
    if candidates is None:
        design = design_name_from_dcp(input_dcp)
        if config.include_recipe:
            candidates = candidates_with_recipe(design)
            logger.info(
                f"scheduler: dispatch (with recipe) picked {candidates} "
                f"for design '{design}'"
            )
        else:
            candidates = candidates_for(design)
            logger.info(
                f"scheduler: dispatch picked {candidates} for design '{design}'"
            )

    logger.info(
        f"scheduler: input={input_dcp}, output={output_dcp}, "
        f"budget={config.total_budget_s}s, candidates={candidates}, work_dir={work_dir}"
    )

    results: List[CandidateResult] = []
    elapsed = 0.0
    start = time.time()

    for i, cand in enumerate(candidates):
        remaining = config.total_budget_s - (time.time() - start)
        if remaining < config.min_first_candidate_s and cand != "recipe":
            logger.info(f"scheduler: insufficient budget ({remaining:.0f}s) for further candidates; stopping")
            break

        if cand == "recipe":
            # Recipe is fast and deterministic — give it a tight cap so a
            # hang (we've seen 40-min runs on digit-recog) can't eat the
            # LLM-candidate budget.  10 min is 2× the typical 5-min wall.
            cand_budget = min(config.recipe_max_wall_s, int(remaining))
            if cand_budget < 60:
                logger.info(f"scheduler: skipping recipe slot (budget {cand_budget}s too tight)")
                continue
        elif i == 0 or candidates[0] == "recipe" and i == 1:
            # First LLM candidate gets the same budget as if recipe wasn't
            # present — recipe runs in addition to, not instead of, an
            # LLM candidate.
            cand_budget = min(config.max_first_candidate_s, int(remaining // 3))
            cand_budget = max(cand_budget, config.min_first_candidate_s)
        else:
            cand_budget = int(remaining - config.safety_margin_s)

        cand_out = work_dir / f"{cand}_optimized.dcp"
        cand_log = work_dir / f"{cand}.log"

        result = run_candidate(
            candidate_name=cand,
            input_dcp=input_dcp,
            output_dcp=cand_out,
            log_path=cand_log,
            time_budget_s=cand_budget,
            extra_args=_extra_args_for(cand),
        )
        results.append(result)

        logger.info(
            f"[{cand}] exit={result.exit_code} "
            f"wall={result.wall_time_s:.0f}s "
            f"final_fmax={result.final_fmax_mhz} valid={result.is_valid}"
        )

    # Repeated-seed loop: fill remaining budget with extra v0_3 invocations
    # (P2 from FINAL_DEV_ROADMAP).  Each relies on LLM stochasticity.
    if config.repeated_seeds > 0:
        for seed_idx in range(config.repeated_seeds):
            remaining = config.total_budget_s - (time.time() - start)
            if remaining < config.min_repeated_seed_s:
                logger.info(
                    f"scheduler: repeated-seed budget low ({remaining:.0f}s "
                    f"< {config.min_repeated_seed_s}s); stopping"
                )
                break
            seed_name = f"v0_3_seed{seed_idx + 2}"  # seed1 was the original v0_3
            seed_out = work_dir / f"{seed_name}_optimized.dcp"
            seed_log = work_dir / f"{seed_name}.log"
            seed_budget = int(remaining - config.safety_margin_s)
            logger.info(
                f"scheduler: launching repeated seed {seed_name} with budget {seed_budget}s"
            )
            seed_result = run_candidate(
                candidate_name=seed_name,
                input_dcp=input_dcp,
                output_dcp=seed_out,
                log_path=seed_log,
                time_budget_s=seed_budget,
                extra_args=_extra_args_for("v0_3"),
            )
            results.append(seed_result)
            logger.info(
                f"[{seed_name}] exit={seed_result.exit_code} "
                f"wall={seed_result.wall_time_s:.0f}s "
                f"final_fmax={seed_result.final_fmax_mhz} valid={seed_result.is_valid}"
            )

    best = select_best(results)
    if best is None:
        logger.error("scheduler: no valid candidate produced a DCP")
        # Write a one-line summary so the wrapper script can report
        (work_dir / "scheduler_summary.json").write_text(json.dumps(
            {"selected": None, "candidates": [_serialise(r) for r in results]},
            indent=2,
        ))
        return 1

    logger.info(
        f"scheduler: selected {best.candidate_name} "
        f"(final_fmax={best.final_fmax_mhz} MHz, wall={best.wall_time_s:.0f}s)"
    )
    shutil.copy2(best.output_dcp, output_dcp)
    (work_dir / "scheduler_summary.json").write_text(json.dumps(
        {
            "selected_candidate": best.candidate_name,
            "selected_final_fmax_mhz": best.final_fmax_mhz,
            "selected_wall_s": best.wall_time_s,
            "candidates": [_serialise(r) for r in results],
        },
        indent=2,
    ))
    return 0


def _extra_args_for(candidate_name: str) -> List[str]:
    """Translate a candidate name into the subprocess CLI flags that
    select that candidate's behaviour.

    `anchor`         — `--mode anchor`  (upstream's stop logic)
    `v0_3`           — `--mode v0_3`    (slope-aware ceiling lift, default)
    `v0_3_seedN`     — `--mode v0_3`    (relies on LLM stochasticity for
                                          variance; same code path as v0_3)
    `recipe`         — no args.  Spawns recipes/cell_replacement.py with
                                  its default knobs (num_paths=10,
                                  detour_threshold=2.0).
    """
    if candidate_name == "anchor":
        return ["--mode", "anchor"]
    if candidate_name == "recipe":
        return []  # recipe binary uses its own argparse defaults
    return ["--mode", "v0_3"]


def _serialise(r: CandidateResult) -> dict:
    return {
        "candidate_name": r.candidate_name,
        "exit_code": r.exit_code,
        "wall_time_s": round(r.wall_time_s, 1),
        "initial_fmax_mhz": r.initial_fmax_mhz,
        "final_fmax_mhz": r.final_fmax_mhz,
        "delta_fmax_mhz": r.delta_fmax_mhz,
        "iterations": r.iterations,
        "cost_usd": r.cost_usd,
        "is_valid": r.is_valid,
        "output_dcp": str(r.output_dcp),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("input_dcp", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True, dest="output_dcp")
    parser.add_argument("--budget", type=int, default=3600,
                        help="Total wall-clock budget in seconds (default 3600 = 60 min).")
    parser.add_argument("--candidates", nargs="+", default=None,
                        help="Ordered list of candidate names. Defaults to per-design "
                             "dispatch via scheduler.dispatch.candidates_for().")
    parser.add_argument("--include-recipe", action="store_true",
                        help="Prepend the cell-replacement recipe slot on designs "
                             "the applicability table marks 'applicable'. Recipe "
                             "is $0 LLM and capped at 10 min wall. Off by default.")
    parser.add_argument("--repeated-seeds", type=int, default=0,
                        help="After the configured candidates finish, fill remaining "
                             "budget with up to N extra v0_3 seeds (each relies on LLM "
                             "stochasticity for variance). 0 = off. P2 from "
                             "FINAL_DEV_ROADMAP — recovers finn/3d-rendering wins.")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Working directory for per-candidate runs (default scheduler_run-<ts>).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = SchedulerConfig(
        total_budget_s=args.budget,
        candidates=args.candidates,
        include_recipe=args.include_recipe,
        repeated_seeds=args.repeated_seeds,
    )

    return run_scheduled(
        input_dcp=args.input_dcp,
        output_dcp=args.output_dcp,
        config=cfg,
        work_dir=args.work_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
