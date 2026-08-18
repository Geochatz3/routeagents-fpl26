"""
FPL26 candidate scheduler package.

Wraps `dcp_optimizer.py` so the contest evaluator's single
`make run_optimizer DCP=...` invocation can run multiple candidates
within the per-benchmark wall budget and select the best valid result.

See `.planning/beta/FINAL_DEV_ROADMAP.md` P0 for the design rationale and
the campaign evidence that justifies this approach.

This package is on `final-dev-portfolio-scheduler` branch only — NOT in
the frozen beta submission tag (`beta-v0_3-freeze-2026-05-10`).
"""
