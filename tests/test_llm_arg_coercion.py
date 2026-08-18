"""An LLM-emitted "20.0" cost corescore 63 MHz. Pin the coercion. (jul30)

THE INCIDENT, from the corpus. corescore_500_mod, jul27, run2 arm C1, env='' —
the SHIPPED configuration:

    recipe_high_fanout_timing_replication({"num_paths": "20.0"})
    ERROR - Error in get_completion: invalid literal for int() with base 10: '20.0'

`int("20.0")` raises ValueError. The model re-emitted the identical call and the
error repeated SIX times at the same timestamp, each round burning LLM-loop
budget. When the loop finally gave up, ~3 minutes of wall remained: the ILS ran
one cycle and the run shipped

    alpha +17.56   against +75.05 .. +86.80 over the eight other corescore runs
                   (identical initial_fmax 344.23 in all nine)

a 63 MHz loss to a string format. It is not a hard abort — the optimizer catches
the error and asks the model to retry — which is precisely why it hid: the row
banks as VALID_OPTIMIZED with a plausible alpha, and only shows up against the
design's own distribution.

The schema declares these fields as integers. A model is free to violate its own
schema, and on a HIDDEN suite we get one run per design with no second chance, so
the parser must absorb it rather than argue about it.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dcp_optimizer import _arg_float, _arg_int  # noqa: E402


class ArgIntTests(unittest.TestCase):
    def test_the_exact_corescore_payload(self):
        """The literal value that cost 63 MHz."""
        self.assertEqual(_arg_int({"num_paths": "20.0"}, "num_paths", 20), 20)

    def test_raw_int_still_raises_without_the_fix(self):
        """Negative control: the OLD expression really does blow up.

        Without this, the test above would pass on any implementation and prove
        nothing about the bug it claims to fix.
        """
        with self.assertRaises(ValueError):
            int("20.0")

    def test_accepts_the_forms_a_model_actually_emits(self):
        for raw, want in (("20", 20), (20, 20), (20.0, 20), (20.7, 20),
                          ("  20 ", 20), (True, 1)):
            self.assertEqual(_arg_int({"k": raw}, "k", 99), want, f"raw={raw!r}")

    def test_missing_key_uses_the_default(self):
        self.assertEqual(_arg_int({}, "k", 7), 7)

    def test_uninterpretable_falls_back_and_does_not_raise(self):
        for raw in ("twenty", None, "", [], {}, "1e", float("nan")):
            self.assertEqual(_arg_int({"k": raw}, "k", 5), 5, f"raw={raw!r}")

    def test_never_raises_on_anything(self):
        """The contract is total: no input may propagate an exception."""
        for raw in (object(), b"20", "0x14", "-", "20.0.0", float("inf")):
            try:
                _arg_int({"k": raw}, "k", 3)
            except Exception as e:                       # pragma: no cover
                self.fail(f"_arg_int raised {e!r} on {raw!r}")


class ArgFloatTests(unittest.TestCase):
    def test_accepts_strings_and_numbers(self):
        for raw, want in (("2.5", 2.5), (2.5, 2.5), ("2", 2.0), (3, 3.0)):
            self.assertEqual(_arg_float({"k": raw}, "k", 9.0), want)

    def test_uninterpretable_falls_back(self):
        for raw in ("two point five", None, [], object()):
            self.assertEqual(_arg_float({"k": raw}, "k", 1.5), 1.5)


class CallSiteTests(unittest.TestCase):
    def test_no_raw_coercion_of_llm_args_remains(self):
        """Every LLM-supplied numeric arg must go through the safe helpers.

        A helper that exists but is not used at the call sites is the jul30
        half-deploy lesson applied to a function.
        """
        import re
        src = (Path(__file__).resolve().parents[1] / "dcp_optimizer.py").read_text()
        bad = re.findall(r"(?:int|float)\(args\.get\([^)]*\)\)", src)
        self.assertEqual(bad, [], f"raw coercions still present: {bad}")


if __name__ == "__main__":
    unittest.main()
