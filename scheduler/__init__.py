"""
FPL26 candidate scheduler package.

Wraps `dcp_optimizer.py` so the contest evaluator's single
`make run_optimizer DCP=...` invocation can run multiple candidates
within the per-benchmark wall budget and select the best valid result.

The design rationale: the agent is stochastic, so running several
candidate configurations and keeping the best valid output converts an
unlucky single draw into a best-of-N result.
"""
