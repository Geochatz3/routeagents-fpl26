#!/usr/bin/env python3
"""Pre-submission readiness check.

Validates the optimizer + scheduler + RAG pipeline can be imported and
the static configuration looks healthy.  Doesn't spawn Vivado — purely
offline so it works in CI / pre-tag environments.

Exit codes:
  0  all checks passed
  1  at least one warning (something is suboptimal but submission would work)
  2  at least one critical issue (submission would likely fail)

Usage: python -m scripts.submission_check
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def check(name, fn):
    """Run one check, return (severity, message)."""
    try:
        result = fn()
        if isinstance(result, tuple):
            severity, msg = result
        else:
            severity, msg = "ok", result
    except Exception as e:
        severity, msg = "critical", f"raised {type(e).__name__}: {e}"
    print(f"  [{severity:8s}] {name}: {msg}")
    return severity


def main() -> int:
    print("=== Pre-submission readiness check ===\n")
    print("Module imports:")
    severities = []
    severities.append(check("dcp_optimizer", lambda: __import__("dcp_optimizer") and "OK"))
    severities.append(check("scheduler.runner", lambda: __import__("scheduler.runner", fromlist=["x"]) and "OK"))
    severities.append(check("scheduler.dispatch", lambda: __import__("scheduler.dispatch", fromlist=["x"]) and "OK"))
    severities.append(check("optimizer.strategy_memory", lambda: __import__("optimizer.strategy_memory", fromlist=["x"]) and "OK"))
    severities.append(check("recipes.cell_replacement", lambda: __import__("recipes.cell_replacement", fromlist=["x"]) and "OK"))

    print("\nFile presence:")

    def must_exist(path):
        p = REPO_ROOT / path
        if p.exists():
            return ("ok", f"{p.stat().st_size} bytes")
        return ("critical", "missing")
    severities.append(check("SYSTEM_PROMPT.TXT", lambda: must_exist("SYSTEM_PROMPT.TXT")))
    severities.append(check("Makefile", lambda: must_exist("Makefile")))
    severities.append(check("optimizer/data/seed_memory.jsonl",
                            lambda: must_exist("optimizer/data/seed_memory.jsonl")))

    print("\nConfig validation:")

    def check_dispatch_table():
        from scheduler.dispatch import KNOWN_DESIGNS, RECIPE_APPLICABILITY
        missing = set(KNOWN_DESIGNS) - set(RECIPE_APPLICABILITY)
        if missing:
            return ("warning", f"recipe applicability missing for: {missing}")
        return ("ok", f"{len(KNOWN_DESIGNS)} known designs, {len(RECIPE_APPLICABILITY)} applicability entries")
    severities.append(check("dispatch tables consistent", check_dispatch_table))

    def check_seed_memory():
        from optimizer.strategy_memory import load_memory
        records = load_memory()
        if len(records) < 10:
            return ("warning", f"only {len(records)} records — RAG impact will be small")
        return ("ok", f"{len(records)} records loaded")
    severities.append(check("strategy memory populated", check_seed_memory))

    def check_prompt_format():
        from dcp_optimizer import load_system_prompt
        p = load_system_prompt()
        try:
            filled = p.format(temp_dir="/tmp", input_dcp="/dcp")
        except KeyError as e:
            return ("critical", f"prompt format failed: missing key {e}")
        if len(filled) > 20000:
            return ("warning", f"prompt is {len(filled)} chars — high β cost")
        return ("ok", f"prompt fills cleanly, {len(filled)} chars")
    severities.append(check("system prompt fillable", check_prompt_format))

    print("\nTool registry:")

    def check_recipe_tools():
        # We can't actually start MCP servers, but we can check the
        # synthetic tool descriptors are constructed by _collect_tools.
        # Just verify the methods exist on DCPOptimizer.
        from dcp_optimizer import DCPOptimizer
        required = (
            "_recipe_cell_replacement",
            "_recipe_lut_optimization",
            "_recipe_register_retiming",
        )
        for m in required:
            if not callable(getattr(DCPOptimizer, m, None)):
                return ("critical", f"DCPOptimizer.{m} missing")
        return ("ok", f"all {len(required)} recipe synthetic methods present")
    severities.append(check("recipe tool methods", check_recipe_tools))

    print("\nMemory verification:")

    def memory_verify():
        from optimizer.strategy_memory import load_memory
        records = load_memory()
        bad = 0
        for r in records:
            if not r.design or r.delta_fmax_mhz is None:
                bad += 1
        if bad:
            return ("warning", f"{bad}/{len(records)} records have missing design or ΔFmax")
        return ("ok", f"all {len(records)} records well-formed")
    severities.append(check("memory record schema", memory_verify))

    print()
    if "critical" in severities:
        print("❌ CRITICAL issues found — submission likely to fail.")
        return 2
    if "warning" in severities:
        print("⚠ Warnings present — submission would work but suboptimally.")
        return 1
    print("✓ All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
