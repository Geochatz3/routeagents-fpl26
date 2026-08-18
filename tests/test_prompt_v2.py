"""FPL26_PROMPT_V2 — the prompt A/B switch. DEFAULT OFF must be byte-identical.

SYSTEM_PROMPT.TXT is a ship-surface file that had no kill switch and no A/B path,
so it was the one behavioural surface that could only be changed globally and
irreversibly. These tests pin the three properties that make it safe to A/B:

  1. unset  -> SYSTEM_PROMPT.TXT, byte for byte (v2.0 unaffected)
  2. =1     -> SYSTEM_PROMPT_V2.TXT
  3. =1 but the file is missing -> falls back AND says so loudly, because a
     silently-missing prompt would make an arm look like a null instead of the
     broken probe it is (feedback_firing_check_must_key_on_treatment).

Plus a content test: V2 must not MANDATE `-directive Default`. Its 86.7% figure
is n=1 design (fir) and every panel seat on aug01 named that as the overfitting
risk. V2 is allowed to state measured rates; it is not allowed to instruct.
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
        expected = (ROOT / "SYSTEM_PROMPT.TXT").read_text()
        self.assertEqual(got, expected,
                         "unset FPL26_PROMPT_V2 must return SYSTEM_PROMPT.TXT unchanged")

    def test_flag_selects_v2(self):
        v2 = ROOT / "SYSTEM_PROMPT_V2.TXT"
        if not v2.exists():
            self.skipTest("SYSTEM_PROMPT_V2.TXT not present")
        os.environ["FPL26_PROMPT_V2"] = "1"
        self.assertEqual(dcp_optimizer.load_system_prompt(), v2.read_text())

    def test_zero_is_a_real_kill_switch(self):
        os.environ["FPL26_PROMPT_V2"] = "0"
        self.assertEqual(dcp_optimizer.load_system_prompt(),
                         (ROOT / "SYSTEM_PROMPT.TXT").read_text())

    def test_missing_v2_falls_back_loudly(self):
        """A missing V2 must not be a silent behaviour change."""
        import logging
        os.environ["FPL26_PROMPT_V2"] = "1"
        v2 = ROOT / "SYSTEM_PROMPT_V2.TXT"
        backup = None
        if v2.exists():
            backup = v2.read_text()
            v2.unlink()
        try:
            with self.assertLogs(dcp_optimizer.logger, level=logging.WARNING) as cm:
                got = dcp_optimizer.load_system_prompt()
            self.assertEqual(got, (ROOT / "SYSTEM_PROMPT.TXT").read_text())
            joined = "\n".join(cm.output)
            self.assertIn("MISSING", joined)
            self.assertIn("BROKEN PROBE", joined,
                          "the warning must name it a broken probe, not a null")
        finally:
            if backup is not None:
                v2.write_text(backup)


class PromptV2Content(unittest.TestCase):
    """V2 may report measured rates; it may not mandate a directive."""

    def setUp(self):
        p = ROOT / "SYSTEM_PROMPT_V2.TXT"
        if not p.exists():
            self.skipTest("SYSTEM_PROMPT_V2.TXT not present")
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
        v1 = (ROOT / "SYSTEM_PROMPT.TXT").read_text()
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
