"""The ship-path flag audit reports a real flag set and a meaningful exit code.

Two failures this pins:

- `all_flags()` used to regex flag names out of raw file text, so
  `scripts/ship_config.py`'s own docstring (`{FPL26_NAME: ...}`) minted a
  `FPL26_NAME` row for a flag that does not exist.
- `main()` returned 1 on a clean tree every single run, because four
  deliberate ship decisions print as OVERRIDDEN. docs/CONFIGURATION.md tells
  readers to run this check, so its exit code has to mean "something
  changed", not "four things are as they have always been".
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ship_path_audit import (  # noqa: E402
    ACKNOWLEDGED_OVERRIDES,
    all_flags,
    classify_overrides,
    code_defaults,
)

# Names that only ever appear as prose or as part of a longer string.
PHANTOMS = ("FPL26_NAME", "FPL26_X", "FPL26_NO_X", "FPL26_ROUTE_STATUS",
            "FPL26_SHALLOW_WLD_RETIME_CANDIDATE")


class TestFlagExtraction:
    def test_docstring_prose_does_not_mint_flags(self):
        flags = all_flags()
        assert not (set(PHANTOMS) & flags), (
            "prose names leaked back into the flag table")

    def test_real_flags_are_still_found(self):
        flags = all_flags()
        for real in ("FPL26_B3_FLOOR_EXIT", "FPL26_ILS_INCR_ROUTE",
                     "FPL26_VIVADO_LEAK_FIX"):
            assert real in flags, f"{real} dropped from the flag table"

    def test_every_flag_is_read_from_the_environment_somewhere(self):
        # The audit is about environment flags; a name that no call site ever
        # reads is the phantom class returning by another door.
        unread = sorted(all_flags() - set(code_defaults()))
        assert unread == [], f"names in the table that no call site reads: {unread}"


class TestOverrideClassification:
    OBSERVED = [("FPL26_B3_FLOOR_EXIT", "1", "0")]

    def test_declared_override_is_not_a_conflict(self):
        declared, undeclared, stale = classify_overrides(
            self.OBSERVED, {"FPL26_B3_FLOOR_EXIT": ("1", "0")})
        assert declared == self.OBSERVED
        assert undeclared == []
        assert stale == []

    def test_undeclared_override_is_a_conflict(self):
        _declared, undeclared, _stale = classify_overrides(self.OBSERVED, {})
        assert undeclared == self.OBSERVED

    def test_a_changed_makefile_value_stops_being_declared(self):
        # The PAIR is acknowledged, not the name: re-injecting a different
        # value is a new decision and must fire.
        _d, undeclared, _s = classify_overrides(
            [("FPL26_B3_FLOOR_EXIT", "2", "0")],
            {"FPL26_B3_FLOOR_EXIT": ("1", "0")})
        assert undeclared == [("FPL26_B3_FLOOR_EXIT", "2", "0")]

    def test_acknowledgement_that_matches_nothing_is_stale(self):
        _d, _u, stale = classify_overrides(
            [], {"FPL26_GONE": ("1", "0")})
        assert stale == ["FPL26_GONE"]


class TestAcknowledgementsAreHonest:
    def test_every_acknowledged_flag_exists(self):
        missing = sorted(set(ACKNOWLEDGED_OVERRIDES) - all_flags())
        assert missing == [], f"acknowledged flags that do not exist: {missing}"

    def test_configuration_md_documents_the_class(self):
        # The allowlist points at a specific section; keep the pointer true.
        text = (ROOT / "docs" / "CONFIGURATION.md").read_text()
        assert "## Honesty notes" in text


class TestExitCode:
    def test_clean_tree_audits_green(self):
        # Run out-of-process: main() strips FPL26_* from os.environ and
        # reloads modules, which must not leak into the rest of the suite.
        p = subprocess.run([sys.executable, "scripts/ship_path_audit.py"],
                           cwd=str(ROOT), capture_output=True, text=True,
                           timeout=300, env={**os.environ, "PYTHONPATH": str(ROOT)})
        assert p.returncode == 0, (
            "the check docs/CONFIGURATION.md recommends is red on a clean "
            f"tree:\n{p.stdout[-2000:]}")
        assert "DECLARED ship-path overrides" in p.stdout, (
            "declared overrides must stay visible, not just silent")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
