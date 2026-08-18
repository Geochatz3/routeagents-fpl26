"""GEPA Phase 0 — ETL episode_memory.jsonl -> a clean, group-split dataset.

Source: /mnt/d/fpl26_optimization_contest/runs/episode_memory.jsonl (1671 episodes,
rich per-episode hypothesis / root_cause / lesson / wns_delta / design_fingerprint).

Produces gepa/dataset.jsonl: one record per usable episode with
  - text     : the critical-path narrative the LLM signal-gate reads (hypothesis,
               plus compact context: root_cause / template / candidate_type / profile)
  - features : the 5 design_fingerprint numerics = the TABULAR baseline (the 0.525
               AUC "no-signal" floor the LLM must beat)
  - label    : int(wns_delta > 0)  ("did this action improve WNS?")
  - group    : design_fingerprint key  (LEAKAGE-SAFE split unit — same design never
               in train AND test; run_id would leak a design across its runs)
  - fold     : 0..K-1 group-fold assignment (GroupKFold-equivalent, deterministic)
  - profile  : design_profile (for stratified reporting only)

Phase 1 (separate) loads this and asks: does an LLM over `text` beat the tabular
AUC on `features`? Gate GEPA Phase 2 on the answer.

Offline, deterministic (no Date/random), no AWS, no LLM.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

SRC = Path("/mnt/d/fpl26_optimization_contest/runs/episode_memory.jsonl")
OUT = Path(__file__).resolve().parent / "dataset.jsonl"
K_FOLDS = 5
# Fixed feature order = the tabular baseline vector.
FEATURE_KEYS = ["net_delay_frac", "logic_levels", "spread_avg", "util_lut_pct", "congestion"]


def _build_text(r: dict) -> str:
    """The critical-path text the LLM signal-gate reads. hypothesis is the path-
    level narrative; the rest is compact structural context (no leakage of the
    outcome/wns_delta/lesson — lesson encodes the answer, so it is EXCLUDED)."""
    parts = []
    hyp = (r.get("hypothesis") or "").strip()
    if hyp:
        parts.append(f"Hypothesis: {hyp}")
    ctx = []
    for k in ("root_cause", "template", "candidate_type", "design_profile"):
        v = r.get(k)
        if v:
            ctx.append(f"{k}={v}")
    tgts = r.get("targets")
    if isinstance(tgts, list) and tgts:
        ctx.append(f"targets={tgts[:3]}")
    if ctx:
        parts.append("Context: " + ", ".join(ctx))
    return "\n".join(parts)


def _features(r: dict):
    f = r.get("design_fingerprint") or {}
    if not isinstance(f, dict):
        return None
    vec = []
    for k in FEATURE_KEYS:
        v = f.get(k)
        if not isinstance(v, (int, float)):
            return None
        vec.append(float(v))
    return vec


def _group_kfold(groups: list[str], k: int) -> dict[str, int]:
    """Deterministic GroupKFold: assign whole groups to folds, greedily balancing
    fold sizes (largest group -> currently-smallest fold). Same group -> same fold,
    so no design leaks across the train/test boundary."""
    sizes: dict[str, int] = {}
    for g in groups:
        sizes[g] = sizes.get(g, 0) + 1
    fold_load = [0] * k
    assign: dict[str, int] = {}
    # Sort by descending size, then group key for stable ties.
    for g in sorted(sizes, key=lambda x: (-sizes[x], x)):
        f = min(range(k), key=lambda i: (fold_load[i], i))
        assign[g] = f
        fold_load[f] += sizes[g]
    return assign


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: source not found: {SRC}", file=sys.stderr)
        return 1
    raw = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    records, dropped = [], {"no_label": 0, "no_features": 0, "no_text": 0}
    for r in raw:
        wd = r.get("wns_delta")
        if not isinstance(wd, (int, float)):
            dropped["no_label"] += 1
            continue
        feats = _features(r)
        if feats is None:
            dropped["no_features"] += 1
            continue
        text = _build_text(r)
        if not text:
            dropped["no_text"] += 1
            continue
        f = r.get("design_fingerprint") or {}
        group = json.dumps({k: f.get(k) for k in FEATURE_KEYS}, sort_keys=True)
        records.append({
            "episode_id": r.get("episode_id"),
            "text": text,
            "features": feats,
            "label": int(wd > 0),
            "wns_delta": float(wd),
            "group": group,
            "profile": r.get("design_profile"),
        })

    fold_of = _group_kfold([rec["group"] for rec in records], K_FOLDS)
    for rec in records:
        rec["fold"] = fold_of[rec["group"]]

    with OUT.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    # ---- report ----
    n = len(records)
    pos = sum(rec["label"] for rec in records)
    print(f"source episodes : {len(raw)}")
    print(f"usable records  : {n}  (dropped {dict(dropped)})")
    print(f"label balance   : {pos} improved / {n - pos} not  ({pos / n:.3f} positive)")
    print(f"groups (designs): {len(fold_of)}  -> {K_FOLDS}-fold group split")
    print(f"{'fold':>4} {'n':>5} {'pos%':>6}  {'groups':>6}")
    for fi in range(K_FOLDS):
        sub = [rc for rc in records if rc["fold"] == fi]
        gp = len({rc["group"] for rc in sub})
        pr = (sum(rc["label"] for rc in sub) / len(sub)) if sub else 0.0
        print(f"{fi:>4} {len(sub):>5} {pr:>6.3f}  {gp:>6}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
