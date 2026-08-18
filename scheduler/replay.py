"""
Offline scheduler replay simulator.

Reads our existing portfolio JSONL data (from `portfolio_runner.sh` runs)
and simulates the scheduler.run_scheduled() flow without launching any
new dcp_optimizer.py subprocesses.  Verifies that the scheduler's
selection matches our human-curated portfolio winners on the campaign
data — which is the goal-backward acceptance criterion in
`.planning/beta/FINAL_DEV_ROADMAP.md` P0 subtask 4.

Usage:
    python3 -m scheduler.replay /mnt/d/.../portfolio_results.jsonl

Each line of the JSONL is a row produced by portfolio_runner.sh's emit_row
(design, candidate, exit_code, wall_time_s, final_fmax_mhz, …).  We group
rows by design, simulate the scheduler making selections in candidate
order honouring a 60-min total budget cap, and report:

  - For each design: which candidate the scheduler picks, and whether it
    matches the highest-final_Fmax-among-valid winner from the raw data.
  - A roll-up of agreement vs disagreement.

The replay does NOT introduce randomness; given the same JSONL it
produces the same selection.  Differences from the human-portfolio
selection are exactly the cases where the scheduler's budget heuristic
disagrees with "best wins regardless of cost/time".
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .runner import CandidateResult, SchedulerConfig, select_best


def _row_to_result(row: dict) -> CandidateResult:
    """Map a portfolio_runner.sh JSONL row to a CandidateResult."""
    dcp = row.get("optimized_dcp")
    return CandidateResult(
        candidate_name=row["candidate"],
        output_dcp=Path(dcp) if dcp else Path("/nonexistent"),
        exit_code=row.get("exit_code", -1),
        wall_time_s=row.get("wall_time_s") or 0,
        final_wns_ns=row.get("final_wns_ns"),
        final_fmax_mhz=row.get("final_fmax_mhz"),
        delta_fmax_mhz=row.get("delta_fmax_mhz"),
        iterations=row.get("iterations"),
        cost_usd=row.get("total_cost_usd"),
    )


def simulate_scheduler(
    candidate_results_in_run_order: List[CandidateResult],
    config: Optional[SchedulerConfig] = None,
) -> dict:
    """Simulate run_scheduled() against pre-recorded candidate runs.

    Realistic budget enforcement (matches scheduler.runner.run_candidate's
    subprocess.run(timeout=...) behaviour):
      * Refuse to start a new candidate when remaining budget is below
        min_first_candidate_s.
      * If a started candidate's recorded wall would exceed the remaining
        budget, mark it as a timeout: exit_code=124, output_dcp invalid.
        (In practice the subprocess gets killed mid-run; no usable DCP.)

    Returns a dict with the selected CandidateResult, the candidates
    actually scheduled (a prefix of the input list, possibly with the
    last entry replaced by a timed-out clone), and the oracle pick (best
    over the *un-budgeted* original list, for comparison).
    """
    config = config or SchedulerConfig()
    elapsed = 0.0
    scheduled: List[CandidateResult] = []

    for i, result in enumerate(candidate_results_in_run_order):
        remaining = config.total_budget_s - elapsed
        # The scheduler refuses to start a new candidate if remaining
        # budget < min_first_candidate.  Mirror that here.
        if i > 0 and remaining < config.min_first_candidate_s:
            break

        # Budget-clip: if recorded wall > remaining, the subprocess would
        # be killed by timeout before producing a DCP.  Replace with a
        # timed-out clone (invalid).
        if result.wall_time_s > remaining + 1:  # +1 s slack for measurement noise
            timed_out = CandidateResult(
                candidate_name=result.candidate_name,
                output_dcp=Path("/nonexistent_due_to_timeout"),
                exit_code=124,
                wall_time_s=remaining,
                final_wns_ns=None,
                final_fmax_mhz=None,
                delta_fmax_mhz=None,
                iterations=None,
                cost_usd=None,
                log_path=result.log_path,
            )
            scheduled.append(timed_out)
            elapsed += remaining
            break  # no budget for any further candidate after a timeout

        scheduled.append(result)
        elapsed += result.wall_time_s

    best = select_best(scheduled)
    return {
        "scheduled_candidates": [r.candidate_name for r in scheduled],
        "scheduled_count": len(scheduled),
        "elapsed_s": elapsed,
        "selected": best,
        "would_have_selected_oracle": select_best(candidate_results_in_run_order),
    }


def replay_from_jsonl(
    jsonl_path: Path,
    candidate_order: List[str],
    config: SchedulerConfig,
    per_design_dispatch: bool = False,
) -> dict:
    """Replay the scheduler against every design in the JSONL.  Returns
    a per-design dict and a roll-up summary.

    When `per_design_dispatch=True`, the candidate ordering for each
    design is taken from `scheduler.dispatch.candidates_for(design)`
    (or `candidates_with_recipe` when config.include_recipe is True)
    instead of the uniform `candidate_order` argument.  Use this to
    evaluate the dispatch table's effect.
    """
    from .dispatch import candidates_for, candidates_with_recipe

    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    by_design: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_design[r["design"]].append(r)

    per_design = {}
    matches_oracle = 0
    matches_present = 0  # cases where designs have at least one of the requested candidates

    for design, design_rows in by_design.items():
        # Resolve per-design ordering when requested.
        if per_design_dispatch:
            order = (
                candidates_with_recipe(design)
                if config.include_recipe
                else candidates_for(design)
            )
        else:
            order = candidate_order

        # Pick the rows matching the requested candidate order, preserving order
        ordered_results: List[CandidateResult] = []
        for cand in order:
            matching = [r for r in design_rows if r.get("candidate") == cand]
            if matching:
                # Use the first matching row (most JSONLs have only one per cand)
                ordered_results.append(_row_to_result(matching[0]))

        # Repeated-seed simulation: append v0_3_seedN rows up to the
        # configured count.  Mirrors the runner's repeated_seeds loop.
        if config.repeated_seeds > 0:
            seed_rows = [r for r in design_rows
                         if r.get("candidate", "").startswith("v0_3_seed")]
            # Sort by candidate name for determinism (seed2 before seed3 etc.)
            seed_rows.sort(key=lambda r: r.get("candidate", ""))
            for seed_row in seed_rows[: config.repeated_seeds]:
                ordered_results.append(_row_to_result(seed_row))

        if not ordered_results:
            per_design[design] = {"skipped": "no candidates in order match"}
            continue

        sim = simulate_scheduler(ordered_results, config)
        oracle = sim["would_have_selected_oracle"]
        selected = sim["selected"]

        match = (
            oracle is not None
            and selected is not None
            and oracle.candidate_name == selected.candidate_name
        )
        # Pull initial_fmax from the first row that has it (all rows for
        # one design should agree).
        initial_fmax = next(
            (r["initial_fmax_mhz"] for r in design_rows if r.get("initial_fmax_mhz")),
            None,
        )
        per_design[design] = {
            "scheduled": sim["scheduled_candidates"],
            "elapsed_s": round(sim["elapsed_s"], 1),
            "initial_fmax_mhz": initial_fmax,
            "selected": selected.candidate_name if selected else None,
            "selected_final_fmax_mhz": selected.final_fmax_mhz if selected else None,
            "selected_delta_fmax_mhz": (
                round(selected.final_fmax_mhz - initial_fmax, 2)
                if (selected and selected.final_fmax_mhz and initial_fmax) else None
            ),
            "oracle": oracle.candidate_name if oracle else None,
            "oracle_final_fmax_mhz": oracle.final_fmax_mhz if oracle else None,
            "oracle_delta_fmax_mhz": (
                round(oracle.final_fmax_mhz - initial_fmax, 2)
                if (oracle and oracle.final_fmax_mhz and initial_fmax) else None
            ),
            "match": match,
        }
        matches_present += 1
        if match:
            matches_oracle += 1

    # Sum ΔFmax across designs — actual contest-relevant fitness.
    selected_total_delta = sum(
        d.get("selected_final_fmax_mhz", 0) - d.get("initial_fmax_mhz", 0)
        for d in per_design.values()
        if isinstance(d, dict)
        and d.get("selected_final_fmax_mhz") is not None
        and d.get("initial_fmax_mhz") is not None
    )
    oracle_total_delta = sum(
        d.get("oracle_final_fmax_mhz", 0) - d.get("initial_fmax_mhz", 0)
        for d in per_design.values()
        if isinstance(d, dict)
        and d.get("oracle_final_fmax_mhz") is not None
        and d.get("initial_fmax_mhz") is not None
    )

    return {
        "designs": per_design,
        "matches_oracle": matches_oracle,
        "designs_with_results": matches_present,
        "total_designs": len(by_design),
        "selected_total_delta_mhz": round(selected_total_delta, 2),
        "oracle_total_delta_mhz": round(oracle_total_delta, 2),
        "selected_vs_oracle_gap_mhz": round(oracle_total_delta - selected_total_delta, 2),
    }


def format_table(replay: dict) -> str:
    lines = []
    lines.append(
        f"{'design':<28} {'scheduled':<22} {'elapsed':>9} {'selected':<14} {'oracle':<14} {'match'}"
    )
    lines.append("-" * 100)
    for design in sorted(replay["designs"].keys()):
        d = replay["designs"][design]
        if "skipped" in d:
            lines.append(f"{design:<28} (skipped: {d['skipped']})")
            continue
        lines.append(
            f"{design:<28} "
            f"{','.join(d['scheduled']):<22} "
            f"{d['elapsed_s']:>9.0f} "
            f"{(d.get('selected') or '—'):<14} "
            f"{(d.get('oracle') or '—'):<14} "
            f"{'✓' if d['match'] else '✗'}"
        )
    lines.append("")
    lines.append(
        f"Match-with-oracle rate: {replay['matches_oracle']}/{replay['designs_with_results']} "
        f"(designs with available results out of {replay['total_designs']} total)"
    )
    lines.append(
        f"ΣΔFmax (selected): {replay.get('selected_total_delta_mhz', 0):+.2f} MHz   "
        f"(oracle: {replay.get('oracle_total_delta_mhz', 0):+.2f} MHz, "
        f"gap: {replay.get('selected_vs_oracle_gap_mhz', 0):+.2f} MHz)"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Offline scheduler replay simulator")
    parser.add_argument("jsonl", type=Path, help="Path to portfolio_results.jsonl")
    parser.add_argument("--candidates", nargs="+", default=["anchor", "v0_3"],
                        help="Uniform candidate order across all designs (ignored when "
                             "--per-design-dispatch is set).")
    parser.add_argument("--per-design-dispatch", action="store_true",
                        help="Use scheduler.dispatch.candidates_for(design) per design "
                             "instead of the uniform --candidates list.")
    parser.add_argument("--include-recipe", action="store_true",
                        help="Include the recipe candidate slot on applicable designs "
                             "(only meaningful with --per-design-dispatch).")
    parser.add_argument("--repeated-seeds", type=int, default=0,
                        help="After configured candidates, append up to N v0_3_seedN "
                             "rows from the JSONL to simulate the repeated-seed "
                             "feature.  P2 from FINAL_DEV_ROADMAP.")
    parser.add_argument("--budget", type=int, default=3600)
    parser.add_argument("--json", action="store_true", help="Output JSON instead of a table")
    args = parser.parse_args(argv)

    config = SchedulerConfig(
        total_budget_s=args.budget,
        candidates=args.candidates,
        include_recipe=args.include_recipe,
        repeated_seeds=args.repeated_seeds,
    )
    replay = replay_from_jsonl(
        args.jsonl, args.candidates, config,
        per_design_dispatch=args.per_design_dispatch,
    )

    if args.json:
        print(json.dumps(replay, indent=2, default=str))
    else:
        print(format_table(replay))

    return 0 if replay["matches_oracle"] == replay["designs_with_results"] else 2


if __name__ == "__main__":
    sys.exit(main())
