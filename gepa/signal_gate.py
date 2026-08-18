"""GEPA Phase 1B — does an LLM over the critical-path TEXT beat the tabular floor?

For a stratified ~400 subset of gepa/dataset.jsonl, a cheap-but-strong LLM reads each
episode's `text` (hypothesis + structural context; NO outcome/lesson leak) and emits
P(this action improves WNS) in [0,100]. We then compute ROC-AUC vs the true label and
compare to the 0.537 tabular floor (baseline_tabular.py).

  AUC >> 0.537  -> the text carries signal the fingerprint numerics don't -> GEPA
                   Phase 2 (prompt-optimize a real predictor/policy) is justified.
  AUC ~ 0.537  -> no extra signal -> do NOT spend Phase 2.

Deterministic subset (no random); temperature=0; results cached to a jsonl so reruns
are free. Parallel calls via a thread pool.
"""
from __future__ import annotations
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from openai import OpenAI
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset.jsonl"
CACHE = HERE / "signal_gate_results.jsonl"
MODEL = "google/gemini-3.1-flash-lite"
N_TARGET = 400
WORKERS = 8
TABULAR_FLOOR = 0.537

SYSTEM = (
    "You are an FPGA physical-design timing-closure expert. You are shown a PROPOSED "
    "optimization action and its rationale for an already placed-and-routed design. "
    "Judge ONLY from the description whether the action will improve worst negative "
    "slack (WNS). Many plausible-sounding moves regress or do nothing. Respond with "
    "ONLY an integer 0-100 = the probability the action IMPROVES WNS. No other text."
)


def _stratified_subset(recs, n):
    """Deterministic: ~n/2 per class, round-robin across groups (designs) so no one
    design dominates. Stable ordering -> reproducible without random."""
    by_label_group = {0: defaultdict(list), 1: defaultdict(list)}
    for r in sorted(recs, key=lambda r: (r["group"], str(r["episode_id"]))):
        by_label_group[r["label"]][r["group"]].append(r)
    out = []
    for lab in (1, 0):
        want = n // 2
        groups = sorted(by_label_group[lab])
        idx = {g: 0 for g in groups}
        picked = 0
        progressing = True
        while picked < want and progressing:
            progressing = False
            for g in groups:
                if idx[g] < len(by_label_group[lab][g]):
                    out.append(by_label_group[lab][g][idx[g]])
                    idx[g] += 1
                    picked += 1
                    progressing = True
                    if picked >= want:
                        break
    return out


def _score_one(client, rec):
    try:
        resp = client.chat.completions.create(
            model=MODEL, temperature=0, max_tokens=8,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": rec["text"]}],
        )
        txt = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\d+", txt)
        prob = max(0, min(100, int(m.group()))) / 100.0 if m else None
        return {"episode_id": rec["episode_id"], "label": rec["label"],
                "prob": prob, "raw": txt}
    except Exception as e:
        return {"episode_id": rec["episode_id"], "label": rec["label"],
                "prob": None, "raw": f"ERROR: {e!r}"}


def main() -> int:
    recs = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    subset = _stratified_subset(recs, N_TARGET)
    pos = sum(r["label"] for r in subset)
    print(f"subset n={len(subset)}  positive={pos}  negative={len(subset) - pos}  "
          f"groups={len({r['group'] for r in subset})}  model={MODEL}")

    cached = {}
    if CACHE.exists():
        for l in CACHE.read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                if d.get("prob") is not None:
                    cached[d["episode_id"]] = d
    todo = [r for r in subset if r["episode_id"] not in cached]
    print(f"cached={len(cached)}  to_score={len(todo)}")

    if todo:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
            return 1
        client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
        with ThreadPoolExecutor(max_workers=WORKERS) as ex, CACHE.open("a") as fh:
            for i, res in enumerate(ex.map(lambda r: _score_one(client, r), todo), 1):
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                cached[res["episode_id"]] = res
                if i % 25 == 0:
                    print(f"  scored {i}/{len(todo)}")

    scored = [cached[r["episode_id"]] for r in subset if cached.get(r["episode_id"], {}).get("prob") is not None]
    y = np.array([s["label"] for s in scored])
    p = np.array([s["prob"] for s in scored])
    n_err = len(subset) - len(scored)
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    acc = float(((p >= 0.5).astype(int) == y).mean())
    print(f"\nscored_ok={len(scored)}  failed={n_err}")
    print(f"LLM-over-text AUC = {auc:.4f}   acc@0.5 = {acc:.3f}")
    print(f"TABULAR FLOOR     = {TABULAR_FLOOR:.4f}")
    delta = auc - TABULAR_FLOOR
    verdict = ("BEATS floor -> Phase 2 JUSTIFIED" if delta > 0.05 else
               "marginal -> inconclusive" if delta > 0.02 else
               "NO extra signal -> do NOT spend Phase 2")
    print(f"delta = {delta:+.4f}  =>  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
