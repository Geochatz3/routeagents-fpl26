"""Import-completeness census for dcp_optimizer's soft-imports.

dcp_optimizer.py guards several `from optimizer.X import ...` blocks with
try/except and silent fallbacks, so a missing module degrades a feature
instead of crashing.  That is deliberate — but it means a packaging mistake
can silently turn a subsystem off.  This test pins the CURRENT, known state:

- every module listed in EXPECTED_PRESENT must import cleanly;
- utilization_features is EXPECTED-ABSENT: it was not part of the scored
  contest submission, so the resource-keyed rules ran with their 'no data'
  fallbacks in the run that produced the official final score.  If someone
  later adds the module, this test fails on purpose so the change is noticed
  and this census (and the honesty note in dcp_optimizer.py) gets updated.
"""

import importlib
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

EXPECTED_PRESENT = [
    "optimizer.strategy_memory",
    "optimizer.api_resilience",
    "optimizer.pathology",
    "optimizer.cross_model_steering",
    "optimizer.recipe_router",
    "optimizer.decision_tracer",
    "optimizer.tool_errors",
    "optimizer.path_guard",
    "optimizer.static_parsers",
    "optimizer.ils_polish",
    "optimizer.constraint_guard",
    "optimizer.route_gate",
    "optimizer.deep_replace_sibling",
    "optimizer.plan_critic",
    "optimizer.wall_economics",
    "optimizer.logic_floor",
    "optimizer.replace_gamble",
]

EXPECTED_ABSENT = [
    "optimizer.utilization_features",
]


@pytest.mark.parametrize("mod", EXPECTED_PRESENT)
def test_soft_import_target_present(mod):
    importlib.import_module(mod)


@pytest.mark.parametrize("mod", EXPECTED_ABSENT)
def test_known_absent_module_stays_absent(mod):
    with pytest.raises(ImportError):
        importlib.import_module(mod)


def test_dcp_optimizer_imports():
    importlib.import_module("dcp_optimizer")


# requirements-dev.txt duplicates two production pins to keep the offline
# test environment free of runtime-only dependencies. This test prevents
# those duplicated pins from drifting.

def _pins(path):
    out = {}
    for line in (REPO / path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~]", line, maxsplit=1)[0].strip()
        out[name] = line
    return out


def test_shared_requirement_pins_agree():
    runtime = _pins("requirements.txt")
    dev = _pins("requirements-dev.txt")
    shared = sorted(set(runtime) & set(dev))
    assert shared, "expected requirements-dev to restate some runtime pins"
    for name in shared:
        assert dev[name] == runtime[name], (
            f"{name} pin drifted: requirements.txt has {runtime[name]!r}, "
            f"requirements-dev.txt has {dev[name]!r}")


def test_dev_requirements_exclude_the_runtime_only_packages():
    dev = _pins("requirements-dev.txt")
    for heavy in ("rapidwright", "pexpect"):
        assert heavy not in dev, (
            f"{heavy} is runtime-only; the offline suite passes without it")
