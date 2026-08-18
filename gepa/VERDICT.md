# GEPA Phase 0/1 — signal-gate verdict (2026-06-08, CORRECTED; re-audited 2026-06-10)

**Decision: DO NOT RUN GEPA on this target/data.** The feasibility probe is
uninformative, the prediction target measures noise on a dead-end action class, and
no contest-aligned target has enough independent designs to optimize without
overfitting. GEPA-the-optimizer is *not* discredited — it was never run; this is a
verdict about a specific probe, a specific target, and the data we have.

---

## ⚠️ The original framing in this file was INVALID
Earlier this doc claimed: LLM-over-text AUC **0.569** beats tabular floor **0.537**.
Those two AUCs were measured on **different datasets** (LLM on a 400 subset, tabular
OOF on all 1441). That comparison is meaningless. Disregard it.

## Corrected, paired comparison (same 401 episodes)
| | AUC |
|---|---|
| LLM-over-text (gemini-3.1-flash-lite, zero-shot) | 0.569 |
| Tabular (HistGBT, GroupKFold OOF, same episodes) | **0.580** |
| Paired diff (LLM − tabular) | **−0.011** |

- Episode-level bootstrap diff CI [−0.084, +0.058]; **design-level (cluster)**
  bootstrap diff CI **[−0.134, +0.102]**, P(diff≤0)=0.55.
- Each model's own cluster-bootstrap AUC CI touches chance: LLM [0.469, 0.655],
  tabular [0.485, 0.686]. Effective N ≈ number of designs (~25, **2 designs = 60%
  of data**), not 401 episodes.
- **The LLM shows no measurable advantage over 5 tabular features.** Both are
  statistically indistinguishable from chance and from each other.

## The target is broken (the deeper problem)
`label = per_episode_wns_delta > 0`:
1. **Thresholds noise.** Median Δ among "improved" = 0.012 ns (45% < 0.010 ns);
   overall median |Δ| = 0.003 ns — within run-to-run variance. The label is mostly a
   sign-flip on noise, which is *why* every model sits at ~0.5–0.58.
2. **Dead-end action class.** 66% of episodes are cell-level; top templates
   `multi_cell_path_surgery`+`placement_nudge` = 65% — the cell-surgery / placement-
   nudge direction proven a DEAD END (see cell-replacement probe jun04). The
   production agent (router + ILS + N-restart) does **not** make these moves.
3. **Disconnected from the objective.** The contest scores final routed fmax per
   benchmark after the FULL recipe+ILS, then mean-rank. A per-episode micro-move
   delta is the wrong causal unit (a locally-negative move can help downstream and
   vice versa), and the agent makes no per-episode binary decision an AUC maps to.

## What would be required before reconsidering GEPA
A **contest-aligned** target with a real reward and enough independent designs:
- target tied to **final fmax** (e.g. top-k recipe selection, marginal-value-of-
  another-restart, ILS-seed choice), labeled from full-run outcomes;
- ≥ ~30–50 **independent designs** (we have ~13) so a leave-design-out model can
  generalize — otherwise any optimizer overfits the handful of designs;
- evaluation as **MHz regret vs an oracle selector**, not AUC on a proxy.
None of these hold today. The 13-design contest can't supply the task diversity an
optimizer needs, and the eval (~1h/$1 per fmax on AWS) makes GEPA rollouts
infeasible on the remaining budget. Re-open only if a multi-design fmax dataset is
collected; otherwise the EV-positive moves are more AWS restarts, ILS tuning, and
recipe aggressiveness — all evidence-backed.

## 2026-06-10 re-audit (gepa/reverify_jun10.py + regret_ceiling_jun10.py)
The jun08 paired numbers reproduce exactly (LLM 0.569 / tab 0.580 / diff −0.011,
cluster diff CI [−0.133,+0.102]). One claim needed CORRECTION, one stayed:

1. **CORRECTED — "indistinguishable from chance" was too strong.** Dropping the
   noise band (|wns_delta| ≥ margin) raises AUC for BOTH models:
   margin 0.02 → LLM 0.654 / tab 0.640 (n=140); 0.05 → 0.669/0.664 (n=94);
   0.10 → 0.686/0.642 (n=69). At margin 0.05 the LLM's own cluster CI is
   **[0.570, 0.762] — entirely above chance**. Spearman(LLM prob, wns_delta)
   = +0.157 (p=.002, unclustered); top-vs-bottom LLM quintile mean Δ +0.026 vs
   −0.10 ns. So the episodes DO carry modest real signal once noise-band labels
   are removed — the ~0.5 full-set AUC was noise dilution, as hypothesized.
2. **UNCHANGED — the GEPA decision.** On every margin cut the LLM ≈ tabular
   (diff cluster CI e.g. [−0.197,+0.243] at 0.02; P(diff≤0)≈0.5). Prompt
   optimization can only widen an LLM-vs-floor gap; there is no gap to widen.
   And the target remains a dead-end action class disconnected from final fmax.

Also measured (contest-aligned ceiling, strategy_memory.jsonl, 326 runs):
oracle-over-median ≈ 240 MHz summed across designs; best-of-3 restarts already
captures ≈ 111 MHz; the residual concentrates in heavy-tailed designs
(corescore, boom_soc, optical-flow, v2) and is partly code-era mixing. The
implied lever is restart allocation / early-stop tuning in the multi-restart
wrapper — not a learned selector, and not GEPA.

**DECISION UNCHANGED: DO NOT RUN GEPA.**

## Reproduce
`dataset.jsonl` gitignored (regenerable). Source:
`/mnt/d/fpl26_optimization_contest/runs/episode_memory.jsonl`.
```
python gepa/etl.py; python gepa/baseline_tabular.py; python gepa/signal_gate.py
```
Deps: dspy 3.2.1, gepa 0.0.27, scikit-learn 1.9.0.
