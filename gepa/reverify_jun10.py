"""GEPA verdict re-verification + NEW cuts (2026-06-10).

Checks whether the corrected jun08 analysis holds, and probes cuts it did not try:
  1. Re-verify paired LLM-vs-tabular AUC on the same scored subset + cluster CI.
  2. NOISE-MARGIN labels: drop |wns_delta| < margin (0.02/0.05/0.10 ns) — if the
     "label is noise" claim is right, AUC should RISE for both models as margin
     grows (the separable tail) or stay flat (no signal at all).
  3. Per-action-class AUC (template parsed from text) — is there a class where
     the LLM does read real signal?
  4. Spearman(prob, wns_delta) + decile calibration — uses the continuous delta,
     immune to the binary-threshold-noise critique.
Offline; uses only cached LLM scores. Deterministic.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
recs = [json.loads(l) for l in (HERE / "dataset.jsonl").read_text().splitlines() if l.strip()]
cache = {}
for l in (HERE / "signal_gate_results.jsonl").read_text().splitlines():
    if l.strip():
        d = json.loads(l)
        if d.get("prob") is not None:
            cache[d["episode_id"]] = d["prob"]

# ---- tabular OOF probs over the FULL dataset (same recipe as baseline_tabular) ----
X = np.array([r["features"] for r in recs], dtype=float)
y = np.array([r["label"] for r in recs], dtype=int)
groups = np.array([r["group"] for r in recs])
oof = np.full(len(y), np.nan)
for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
    m = HistGradientBoostingClassifier(random_state=0)
    m.fit(X[tr], y[tr])
    oof[te] = m.predict_proba(X[te])[:, 1]

scored = [(i, r) for i, r in enumerate(recs) if r["episode_id"] in cache]
idx = np.array([i for i, _ in scored])
sub = [r for _, r in scored]
llm = np.array([cache[r["episode_id"]] for r in sub])
tab = oof[idx]
ys = y[idx]
gs = groups[idx]
wd = np.array([r["wns_delta"] for r in sub])
print(f"scored subset n={len(sub)} designs={len(set(gs))} pos={ys.mean():.3f}")

def cluster_ci(stat_fn, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    ugs = sorted(set(gs))
    out = []
    for _ in range(n_boot):
        pick = rng.choice(len(ugs), size=len(ugs), replace=True)
        mask = np.concatenate([np.where(gs == ugs[k])[0] for k in pick])
        try:
            out.append(stat_fn(mask))
        except ValueError:
            pass
    a = np.array(out)
    return np.percentile(a, [2.5, 97.5]), a

auc_llm = roc_auc_score(ys, llm); auc_tab = roc_auc_score(ys, tab)
print(f"\n[1] paired re-verify: LLM {auc_llm:.4f}  tab {auc_tab:.4f}  diff {auc_llm-auc_tab:+.4f}")
(ci, arr) = cluster_ci(lambda m: roc_auc_score(ys[m], llm[m]) - roc_auc_score(ys[m], tab[m]))
print(f"    cluster diff CI [{ci[0]:+.3f},{ci[1]:+.3f}]  P(diff<=0)={np.mean(arr<=0):.2f}")

print("\n[2] noise-margin labels (drop |wns_delta| < margin):")
for margin in (0.0, 0.02, 0.05, 0.10):
    m = np.abs(wd) >= margin
    if len(np.unique(ys[m])) < 2 or m.sum() < 40:
        print(f"    margin {margin:.2f}: n={m.sum()} (too few)"); continue
    print(f"    margin {margin:.2f}: n={m.sum():4d} pos={ys[m].mean():.2f}  "
          f"LLM {roc_auc_score(ys[m], llm[m]):.3f}  tab {roc_auc_score(ys[m], tab[m]):.3f}")

print("\n[3] per-action-class AUC (template from text):")
def template_of(r):
    mm = re.search(r"template=([\w-]+)", r["text"])
    return mm.group(1) if mm else "none"
from collections import defaultdict
bycls = defaultdict(list)
for j, r in enumerate(sub):
    bycls[template_of(r)].append(j)
for cls, js in sorted(bycls.items(), key=lambda kv: -len(kv[1])):
    js = np.array(js)
    if len(js) < 30 or len(np.unique(ys[js])) < 2:
        continue
    print(f"    {cls:<28} n={len(js):4d}  LLM {roc_auc_score(ys[js], llm[js]):.3f}  "
          f"tab {roc_auc_score(ys[js], tab[js]):.3f}")

print("\n[4] continuous-target checks:")
rho, p = spearmanr(llm, wd)
print(f"    Spearman(LLM prob, wns_delta) rho={rho:+.3f} p={p:.3f}")
rho2, p2 = spearmanr(tab, wd)
print(f"    Spearman(tab prob, wns_delta) rho={rho2:+.3f} p={p2:.3f}")
order = np.argsort(llm)
for q in range(5):
    s = order[q*len(order)//5:(q+1)*len(order)//5]
    print(f"    LLM-prob quintile {q+1}: mean prob {llm[s].mean():.2f}  "
          f"mean wns_delta {wd[s].mean():+.4f} ns  median {np.median(wd[s]):+.4f}")
