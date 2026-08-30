"""Test that the plan critic derives its default model identifier from the
exercised fallback model.

Model identifiers cannot be validated offline, and critic failures are
intentionally non-fatal. Sharing the fallback identifier prevents an invalid
default from silently disabling the critic.
"""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dcp_optimizer  # noqa: E402
from tests.source_corpus import dcp_source_lines, dcp_source_text

KNOWN_BAD = "google/gemini-3.1-flash"   # 400s on OpenRouter, verified


class PlanCriticModelId(unittest.TestCase):
    def test_default_is_derived_from_fallback(self):
        self.assertEqual(dcp_optimizer.PLAN_CRITIC_DEFAULT_MODEL,
                         dcp_optimizer.FALLBACK_MODEL,
                         "one literal, one place to change")

    def test_default_is_not_the_known_bad_id(self):
        self.assertNotEqual(dcp_optimizer.PLAN_CRITIC_DEFAULT_MODEL, KNOWN_BAD)
        self.assertNotEqual(dcp_optimizer.FALLBACK_MODEL, KNOWN_BAD)

    def test_no_bare_bad_literal_in_code(self):
        """The string may appear in a comment explaining the bug, never in code."""
        src = dcp_source_lines()
        offenders = [
            (i + 1, ln.strip())
            for i, ln in enumerate(src)
            if f'"{KNOWN_BAD}"' in ln and not ln.lstrip().startswith("#")
        ]
        self.assertEqual(offenders, [],
                         f"bare invalid model id still referenced: {offenders}")

    def test_critic_model_attribute_uses_the_default(self):
        import inspect
        src = inspect.getsource(dcp_optimizer)
        self.assertIn("self.plan_critic_model: str = PLAN_CRITIC_DEFAULT_MODEL", src)

    def test_fallback_model_is_a_plausible_openrouter_id(self):
        """provider/model shape — catches a bare 'gemini-flash' style typo."""
        self.assertRegex(dcp_optimizer.FALLBACK_MODEL,
                         r"^[a-z0-9-]+/[A-Za-z0-9.\-_:]+$")


if __name__ == "__main__":
    unittest.main()
