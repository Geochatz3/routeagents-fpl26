"""GEPA Phase 1A — the TABULAR floor the LLM signal-gate must beat.

Loads gepa/dataset.jsonl (from etl.py) and predicts label (wns_delta>0) from the 5
design_fingerprint numerics ALONE, under the SAME leakage-safe group split (no design
in train AND test). Reports out-of-fold (OOF) ROC-AUC — the number the LLM-over-text
model has to exceed to justify GEPA Phase 2.

Uses the BEST of {logistic regression, gradient-boosted trees} so the floor is fair
(not a weak strawman). Deterministic (fixed random_state). Offline, no LLM.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent / "dataset.jsonl"


def _oof_auc(model_fn, X, y, groups, n_splits):
    """Pooled out-of-fold AUC: train on train-groups, predict held-out groups,
    concat all held-out predictions, score once. Robust to lopsided group sizes."""
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    per_fold = []
    for tr, te in gkf.split(X, y, groups):
        m = model_fn()
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        oof[te] = p
        # per-fold AUC only meaningful if the fold has both classes
        if len(np.unique(y[te])) == 2:
            per_fold.append(roc_auc_score(y[te], p))
    pooled = roc_auc_score(y, oof)
    return pooled, per_fold


def main() -> int:
    recs = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    X = np.array([r["features"] for r in recs], dtype=float)
    y = np.array([r["label"] for r in recs], dtype=int)
    groups = np.array([r["group"] for r in recs])
    n_splits = len(set(groups.tolist()))
    n_splits = min(5, n_splits)
    print(f"records={len(y)} positive={y.mean():.3f} groups={len(set(groups.tolist()))} "
          f"splits={n_splits}")

    models = {
        "logreg": lambda: make_pipeline(StandardScaler(),
                                        LogisticRegression(max_iter=1000)),
        "hist_gbt": lambda: HistGradientBoostingClassifier(random_state=0),
    }
    best = ("-", 0.5)
    for name, fn in models.items():
        pooled, per_fold = _oof_auc(fn, X, y, groups, n_splits)
        pf = ", ".join(f"{a:.3f}" for a in per_fold)
        print(f"  {name:<9} OOF-AUC={pooled:.4f}   per-fold=[{pf}]")
        if pooled > best[1]:
            best = (name, pooled)
    print(f"\nTABULAR FLOOR = {best[1]:.4f}  (best model: {best[0]})")
    print("Phase 1B (LLM over `text`) must beat this OOF-AUC to justify GEPA Phase 2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
