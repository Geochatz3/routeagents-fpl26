"""Contest-aligned ceiling analysis (2026-06-10): per-design run-level fmax spread.

From strategy_memory.jsonl (326 run summaries): for each design, the distribution
of delta_fmax across runs. Quantifies
  - oracle-vs-median regret: what a PERFECT per-design selector (or infinite
    restarts) could add over a typical single run -> the ceiling for ANY learned
    selector / GEPA-optimized policy, and
  - how much of that multi-restart (best-of-N) already captures.
Also: cluster-CI for the noise-margin AUC cuts from reverify_jun10.
"""
from __future__ import annotations
import json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
runs = [json.loads(l) for l in (HERE.parent / "strategy_memory.jsonl").read_text().splitlines() if l.strip()]
by = defaultdict(list)
for r in runs:
    d = r.get("design") or "?"
    df = r.get("delta_fmax_mhz")
    if isinstance(df, (int, float)):
        by[d].append(float(df))

print(f"{'design':<28}{'n':>4}{'min':>9}{'med':>9}{'max':>9}{'oracle-med':>11}{'bestof3-med':>12}")
tot_or, tot_b3 = 0.0, 0.0
rng = np.random.default_rng(0)
for d, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
    v = np.array(v)
    if len(v) < 3:
        continue
    med = np.median(v); mx = v.max()
    # expected best-of-3 by resampling
    b3 = np.mean([rng.choice(v, 3, replace=True).max() for _ in range(2000)])
    print(f"{d:<28}{len(v):>4}{v.min():>9.1f}{med:>9.1f}{mx:>9.1f}{mx-med:>11.1f}{b3-med:>12.1f}")
    tot_or += mx - med; tot_b3 += b3 - med
print(f"\nsum over designs: oracle-over-median {tot_or:.0f} MHz; "
      f"best-of-3-over-median {tot_b3:.0f} MHz "
      f"-> selector headroom beyond restarts ~{tot_or - tot_b3:.0f} MHz (upper bound, "
      f"mixes recipe/code-era/variance)")

# ---- cluster CI for the margin cuts (uses cached LLM scores + OOF) ----
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
recs = [json.loads(l) for l in (HERE / "dataset.jsonl").read_text().splitlines() if l.strip()]
cache = {json.loads(l)["episode_id"]: json.loads(l)["prob"]
         for l in (HERE / "signal_gate_results.jsonl").read_text().splitlines()
         if l.strip() and json.loads(l).get("prob") is not None}
X = np.array([r["features"] for r in recs]); y = np.array([r["label"] for r in recs])
groups = np.array([r["group"] for r in recs])
oof = np.full(len(y), np.nan)
for tr, te in GroupKFold(5).split(X, y, groups):
    m = HistGradientBoostingClassifier(random_state=0); m.fit(X[tr], y[tr])
    oof[te] = m.predict_proba(X[te])[:, 1]
idx = np.array([i for i, r in enumerate(recs) if r["episode_id"] in cache])
sub = [recs[i] for i in idx]
llm = np.array([cache[r["episode_id"]] for r in sub]); tab = oof[idx]
ys = y[idx]; gs = groups[idx]; wd = np.array([r["wns_delta"] for r in sub])

print("\nmargin-cut cluster CIs (design-level bootstrap):")
for margin in (0.02, 0.05):
    m = np.abs(wd) >= margin
    ugs = sorted(set(gs[m]))
    diffs, aucs_l = [], []
    for _ in range(3000):
        pick = rng.choice(len(ugs), len(ugs), replace=True)
        mask = np.concatenate([np.where((gs == ugs[k]) & m)[0] for k in pick])
        if len(np.unique(ys[mask])) < 2:
            continue
        try:
            al = roc_auc_score(ys[mask], llm[mask]); at = roc_auc_score(ys[mask], tab[mask])
            diffs.append(al - at); aucs_l.append(al)
        except ValueError:
            pass
    d = np.array(diffs); a = np.array(aucs_l)
    print(f"  margin {margin:.2f} (n={m.sum()}, designs={len(ugs)}): "
          f"LLM AUC CI [{np.percentile(a,2.5):.3f},{np.percentile(a,97.5):.3f}]  "
          f"diff CI [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]  "
          f"P(diff<=0)={np.mean(d<=0):.2f}")
