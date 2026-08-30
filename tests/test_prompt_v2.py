"""Validate feature-gated system prompt selection and fallback behavior.

With the flag unset, the default prompt remains byte-identical. Enabling the
flag selects the alternate prompt; if that file is missing, selection falls
back to the default and emits a visible warning. The alternate prompt may
report measured rates but must not mandate `-directive Default`.
"""
import importlib
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dcp_optimizer  # noqa: E402


class PromptV2Switch(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("FPL26_PROMPT_V2", None)

    def tearDown(self):
        os.environ.pop("FPL26_PROMPT_V2", None)
        if self._saved is not None:
            os.environ["FPL26_PROMPT_V2"] = self._saved

    def test_default_off_is_byte_identical(self):
        got = dcp_optimizer.load_system_prompt()
        expected = (ROOT / "prompts" / "system_prompt_scored.txt").read_text()
        self.assertEqual(got, expected,
                         "unset FPL26_PROMPT_V2 must return the scored prompt unchanged")

    def test_flag_selects_v2(self):
        v2 = ROOT / "prompts" / "system_prompt_v2_experimental.txt"
        if not v2.exists():
            self.skipTest("system_prompt_v2_experimental.txt not present")
        os.environ["FPL26_PROMPT_V2"] = "1"
        self.assertEqual(dcp_optimizer.load_system_prompt(), v2.read_text())

    def test_zero_is_a_real_kill_switch(self):
        os.environ["FPL26_PROMPT_V2"] = "0"
        self.assertEqual(dcp_optimizer.load_system_prompt(),
                         (ROOT / "prompts" / "system_prompt_scored.txt").read_text())

    def test_missing_v2_falls_back_loudly(self):
        """A missing V2 must not be a silent behaviour change."""
        import logging
        os.environ["FPL26_PROMPT_V2"] = "1"
        v2 = ROOT / "prompts" / "system_prompt_v2_experimental.txt"
        backup = None
        if v2.exists():
            backup = v2.read_text()
            v2.unlink()
        try:
            with self.assertLogs(dcp_optimizer.logger, level=logging.WARNING) as cm:
                got = dcp_optimizer.load_system_prompt()
            self.assertEqual(got, (ROOT / "prompts" / "system_prompt_scored.txt").read_text())
            joined = "\n".join(cm.output)
            self.assertIn("MISSING", joined)
            self.assertIn("broken probe", joined,
                          "the warning must name it a broken probe, not a null")
        finally:
            if backup is not None:
                v2.write_text(backup)


class PromptV2Content(unittest.TestCase):
    """V2 may report measured rates; it may not mandate a directive."""

    def setUp(self):
        p = ROOT / "prompts" / "system_prompt_v2_experimental.txt"
        if not p.exists():
            self.skipTest("system_prompt_v2_experimental.txt not present")
        self.text = p.read_text()

    def test_states_the_arg_drop_semantics(self):
        low = self.text.lower()
        self.assertIn("every other argument is discarded", low)
        self.assertIn("path_groups", low)

    def test_does_not_mandate_default(self):
        """No imperative telling the agent to always use -directive Default."""
        low = self.text.lower()
        for banned in ("always use default", "always use -directive default",
                       "prefer default", "use default first",
                       "default is the highest-yield"):
            self.assertNotIn(banned, low,
                             f"V2 must not mandate Default ({banned!r}); its 86.7% is n=1 design")

    def test_keeps_the_scoped_recipe_available(self):
        """Demoted, not banned — wall pressure can still make it correct."""
        self.assertIn("recipe_critical_path_focused_phys_opt", self.text)

    def test_differs_from_v1_only_in_section_e(self):
        v1 = (ROOT / "prompts" / "system_prompt_scored.txt").read_text()
        # every V1 line outside the edited block must survive verbatim
        edited_markers = ("SCOPED FIRST CHOICE", "MANUAL: vivado_phys_opt_design supports these directives:")
        missing = [ln for ln in v1.splitlines()
                   if ln.strip() and ln not in self.text
                   and not any(m in ln for m in edited_markers)]
        # the 5 continuation lines of the reworded block are expected to differ
        self.assertLessEqual(len(missing), 6,
                             f"V2 changed more than section E: {missing[:8]}")


if __name__ == "__main__":
    unittest.main()
