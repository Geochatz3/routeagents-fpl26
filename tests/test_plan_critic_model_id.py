"""The PLAN CRITIC's model ID must be one the API actually accepts.

aug02: `FPL26_PLAN_CRITIC` had been a silent no-op for its entire life. The
default model was hardcoded to "google/gemini-3.1-flash", which is NOT a valid
OpenRouter model ID. Every call returned HTTP 400; the module fails open by
design, so the run continued and the log said only

    plan-critic: call failed (BadRequestError ... 'is not a valid model ID')

That is why the flag shows 0 usable rows across 209 corpus runs AND a dedicated
6-row A/B: the A/B armed it, it tried 5+ times per run, and every call died.

A model ID cannot be validated offline, so these tests pin the STRUCTURE that
makes it wrong-by-construction to drift again: the critic's default is DERIVED
from FALLBACK_MODEL, which the resilience path already exercises against the live
API, so a bad ID would break the fallback long before it silently broke a critic
that fails open.
"""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dcp_optimizer  # noqa: E402

KNOWN_BAD = "google/gemini-3.1-flash"   # 400s on OpenRouter, verified aug02


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
        src = (ROOT / "dcp_optimizer.py").read_text().splitlines()
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
