"""
Replay using the per-design dispatch table — verify the table picks the
right candidate ordering across the 10 portfolio designs.

For each design in the JSONL, ask `dispatch.candidates_for(design)` for
the ordering, then simulate the scheduler with that ordering.  Roll up
ΣΔFmax + W/T/L vs published baseline.  Compare to the global v0_3,anchor
ordering to confirm the table either matches or improves on it.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .aggregate import PUBLISHED_BL, true_oracle_sum
from .dispatch import candidates_for
from .replay import simulate_scheduler, _row_to_result
from .runner import SchedulerConfig, select_best


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--budget", type=int, default=3600)
    args = parser.parse_args(argv)

    rows = [json.loads(l) for l in args.jsonl.read_text().splitlines() if l.strip()]
    by_design = defaultdict(list)
    for r in rows:
        by_design[r["design"]].append(r)

    config = SchedulerConfig(total_budget_s=args.budget, candidates=None)
    oracle = true_oracle_sum(args.jsonl)

    print(f"Per-design dispatch replay (budget {args.budget}s)")
    print(f"Oracle (best-of-all-candidates per design, no budget): {oracle:+.2f} MHz")
    print()
    print(f"{'design':<28} {'ordering':<22} {'ΔFmax':>9} {'BL':>7} {'verdict':<8}")
    print("-" * 78)

    sum_dispatch = 0.0
    wins = ties = losses = 0
    for design in sorted(by_design.keys()):
        ordering = candidates_for(design)
        # Build CandidateResult list in ordering, only including rows
        # whose candidate name matches.  Unknown candidates (e.g. seed2/3)
        # don't appear in dispatch's known orderings, so this naturally
        # restricts to anchor + v0_3.
        results = []
        for cand in ordering:
            matching = [r for r in by_design[design] if r["candidate"] == cand]
            if matching:
                results.append(_row_to_result(matching[0]))

        if not results:
            print(f"{design:<28} {'(no data)':<22} {'—':>9} {'—':>7} {'—':<8}")
            continue

        sim = simulate_scheduler(results, config)
        sel = sim["selected"]
        if sel is None:
            print(f"{design:<28} {'(timeout)':<22} {'—':>9} {'—':>7} {'—':<8}")
            continue

        # Pull ΔFmax from the underlying row
        sel_row = next(r for r in by_design[design] if r["candidate"] == sel.candidate_name)
        df = sel_row.get("delta_fmax_mhz")
        bl = PUBLISHED_BL.get(design)
        verdict = "—"
        if df is not None and bl is not None:
            gap = df - bl
            if gap > 2:    verdict = "WIN"; wins += 1
            elif gap < -2: verdict = "LOSS"; losses += 1
            else:          verdict = "TIE"; ties += 1
            sum_dispatch += df

        print(f"{design:<28} {','.join(ordering):<22} {df:>+9.2f} {bl:>+7.1f} {verdict:<8}")

    n = wins + ties + losses
    print()
    print(f"ΣΔFmax (dispatch replay): {sum_dispatch:>+10.2f} MHz")
    print(f"vs oracle:                {sum_dispatch - oracle:>+10.2f} MHz")
    print(f"W/T/L vs published BL:    {wins} / {ties} / {losses}  (over {n})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
