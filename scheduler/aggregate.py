"""
Aggregate replay results — sum ΔFmax over a design set under different
scheduler configurations.  Lets us compare candidate orderings on the
metric that actually matters for the contest (sum of selected ΔFmax),
not just per-design match counts.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from .replay import replay_from_jsonl, _row_to_result
from .runner import SchedulerConfig, select_best


# Reference baselines (same as portfolio_select_best.py)
PUBLISHED_BL = {
    "vexriscv_re-place": 105.1, "amd_mini-isp": 68.3,
    "rosetta_spam-filter": 56.6, "rosetta_3d-rendering": 8.3,
    "rosetta_digit-recognition": 23.8, "logicnets_jscl": 31.4,
    "rosetta_optical-flow": 5.3, "vtr_mcml": 10.8,
    "finn_radioml": 39.6, "corescore_500_mod": 79.1,
}


def aggregate(jsonl_path: Path, candidate_order: List[str], budget_s: int) -> Tuple[float, int, int, int, dict]:
    """Run the replay; return (sum_selected_dfmax, wins, ties, losses, per_design)."""
    config = SchedulerConfig(total_budget_s=budget_s, candidates=candidate_order)
    rep = replay_from_jsonl(jsonl_path, candidate_order, config)

    # Need to re-resolve selected ΔFmax for each design — replay only stores
    # final_fmax_mhz on the result, not delta.  Pull from the JSONL.
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    by_design_cand = defaultdict(dict)
    for r in rows:
        by_design_cand[r["design"]][r["candidate"]] = r

    sum_selected = 0.0
    sum_oracle = 0.0
    wins = ties = losses = 0
    per_design: Dict[str, dict] = {}

    for design, info in rep["designs"].items():
        if "skipped" in info or info.get("selected") is None:
            per_design[design] = {"selected": None, "delta_fmax_mhz": None}
            continue

        sel_cand = info["selected"]
        ora_cand = info["oracle"]
        sel_row = by_design_cand[design].get(sel_cand)
        ora_row = by_design_cand[design].get(ora_cand)
        sel_df = sel_row.get("delta_fmax_mhz") if sel_row else None
        ora_df = ora_row.get("delta_fmax_mhz") if ora_row else None

        per_design[design] = {
            "selected_cand": sel_cand,
            "selected_dfmax": sel_df,
            "oracle_cand": ora_cand,
            "oracle_dfmax": ora_df,
            "loss_vs_oracle": (ora_df - sel_df) if (sel_df is not None and ora_df is not None) else None,
            "match": info["match"],
        }

        if sel_df is not None:
            sum_selected += sel_df
            pub = PUBLISHED_BL.get(design)
            if pub is not None:
                if sel_df - pub > 2:
                    wins += 1
                elif sel_df - pub < -2:
                    losses += 1
                else:
                    ties += 1
        if ora_df is not None:
            sum_oracle += ora_df

    return sum_selected, wins, ties, losses, {
        "per_design": per_design,
        "sum_oracle_dfmax": sum_oracle,
    }


def true_oracle_sum(jsonl_path: Path) -> float:
    """Best ΔFmax per design across ALL candidates in the JSONL, summed."""
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    by_design = defaultdict(list)
    for r in rows:
        by_design[r["design"]].append(r)
    s = 0.0
    for design, design_rows in by_design.items():
        valid = [r for r in design_rows if r.get("completed")
                 and r.get("optimized_dcp")
                 and r.get("final_fmax_mhz") is not None]
        if not valid:
            continue
        best = max(valid, key=lambda r: r["final_fmax_mhz"])
        if best.get("delta_fmax_mhz") is not None:
            s += best["delta_fmax_mhz"]
    return s


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--budget", type=int, default=3600)
    parser.add_argument("--orderings", nargs="+", default=[
        "anchor,v0_3",
        "v0_3,anchor",
        "v0_3,v0_3_seed2,v0_3_seed3",
        "anchor,v0_3,v0_3_seed2",
        "anchor,v0_3,v0_3_seed2,v0_3_seed3",
    ])
    args = parser.parse_args(argv)

    oracle = true_oracle_sum(args.jsonl)
    print(f"Comparing {len(args.orderings)} orderings under {args.budget}s budget")
    print(f"True oracle (best-of-all-candidates per design, no budget): {oracle:+.2f} MHz")
    print()
    print(f"{'ordering':<48} {'ΣΔFmax':>10} {'W/T/L':<8} {'Δvs-oracle':>11} {'#designs':>9}")
    print("-" * 95)

    for o_csv in args.orderings:
        ordering = o_csv.split(",")
        s_sum, w, t, l, extra = aggregate(args.jsonl, ordering, args.budget)
        n = w + t + l
        loss = s_sum - oracle
        print(f"{o_csv:<48} {s_sum:>+10.2f} {w}/{t}/{l:<5} {loss:>+11.2f} {n:>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
