"""Runtime provenance: every run records the config it actually used.

Static pinning (`tests/test_ship_config.py`) says what SHOULD ship. This is the
other half — what a given run DID use — because twice in three days our belief
about the effective config was wrong in opposite directions, and the review's
verdict was that *"'proven' attaches to a configuration that may never have
shipped"*. A banked alpha with no config provenance cannot be audited later.

The manifest must satisfy three properties, each tested here:
  1. it records RESOLVED state, not merely which names were present;
  2. it captures FPL26_* vars we have not thought to enumerate;
  3. it can never break a run.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _real_flags():
    import dcp_optimizer
    return dcp_optimizer.DCPOptimizer._MANIFEST_FLAGS


class _Stub:
    """Minimal stand-in carrying only what the manifest writer touches, so the
    test does not depend on constructing a full optimizer.

    `_MANIFEST_FLAGS` is pulled from the REAL class rather than duplicated — a
    copy here would let the enumeration drift without any test noticing.
    """

    mode = "v0_3"
    contest_mode = True
    max_wall_seconds = 3500.0

    def __init__(self, run_dir):
        self.run_dir = run_dir
        self._MANIFEST_FLAGS = _real_flags()


def _writer():
    import dcp_optimizer
    cls = dcp_optimizer.DCPOptimizer
    return cls._write_effective_config_manifest, cls._MANIFEST_FLAGS


class ManifestTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="effcfg_")
        self.stub = _Stub(self.tmp)
        self.write, self.flags = _writer()

    def _run(self, env):
        clean = {k: v for k, v in os.environ.items() if not k.startswith("FPL26_")}
        clean.update(env)
        with mock.patch.dict(os.environ, clean, clear=True):
            self.write(self.stub)
        p = Path(self.tmp) / "effective_config.json"
        self.assertTrue(p.exists(), "manifest was not written")
        return json.loads(p.read_text())

    def test_records_resolved_on_off_not_just_presence(self):
        """The jul30 bug in miniature: a name being present says nothing about
        whether the mechanism is active."""
        m = self._run({"FPL26_DEEP_REPLACE_UNBANDED": "1",
                       "FPL26_ILS_LADDER_ORDER_BY_WNS": "0"})
        self.assertTrue(m["flags"]["FPL26_DEEP_REPLACE_UNBANDED"]["on"])
        self.assertTrue(m["flags"]["FPL26_DEEP_REPLACE_UNBANDED"]["set"])
        self.assertFalse(m["flags"]["FPL26_ILS_LADDER_ORDER_BY_WNS"]["on"])
        self.assertTrue(m["flags"]["FPL26_ILS_LADDER_ORDER_BY_WNS"]["set"],
                        "explicitly-set-to-0 must be distinguishable from unset")

    def test_unset_flag_is_recorded_as_unset_and_off(self):
        m = self._run({})
        e = m["flags"]["FPL26_DEEP_REPLACE_UNBANDED"]
        self.assertFalse(e["set"])
        self.assertFalse(e["on"])
        self.assertIsNone(e["raw"])

    def test_truthy_spellings_resolve(self):
        for raw in ("1", "true", "on", "yes", "ON", " True "):
            m = self._run({"FPL26_PLATEAU_PRELOOP_FIX": raw})
            self.assertTrue(m["flags"]["FPL26_PLATEAU_PRELOOP_FIX"]["on"], raw)

    def test_captures_unenumerated_fpl26_vars(self):
        """A flag added later, before anyone updates _MANIFEST_FLAGS, must still
        appear — otherwise the manifest silently under-reports."""
        m = self._run({"FPL26_SOME_BRAND_NEW_KNOB": "7"})
        self.assertEqual(m["fpl26_env"].get("FPL26_SOME_BRAND_NEW_KNOB"), "7")

    def test_excludes_non_fpl26_environment(self):
        """Must not vacuum up the environment — it would leak secrets into a file
        that gets copied around with results."""
        m = self._run({"OPENROUTER_API_KEY": "sk-should-not-appear",
                       "FPL26_DEEP_REPLACE": "1"})
        self.assertNotIn("OPENROUTER_API_KEY", m["fpl26_env"])
        blob = json.dumps(m)
        self.assertNotIn("sk-should-not-appear", blob,
                         "manifest leaked a non-FPL26 environment value")

    def test_records_both_code_md5s(self):
        """code_md5 in the banked rows covers only dcp_optimizer.py, and
        ils_polish.py drifted independently during the jul29 build confusion."""
        m = self._run({})
        self.assertIsNotNone(m["code_md5"]["dcp_optimizer.py"])
        self.assertIsNotNone(m["code_md5"]["optimizer/ils_polish.py"])
        self.assertNotEqual(m["code_md5"]["dcp_optimizer.py"],
                            m["code_md5"]["optimizer/ils_polish.py"])

    def test_load_bearing_flags_are_all_enumerated(self):
        """Guard against the enumeration rotting: the flags this repo actually
        decides behaviour on should be listed, so they get resolved state and not
        just the raw env dump."""
        for name in ("FPL26_DEEP_REPLACE_UNBANDED", "FPL26_ILS_HURDLE_CONTINUE",
                     "FPL26_ILS_LADDER_ORDER_BY_WNS", "FPL26_PLATEAU_PRELOOP_FIX",
                     "FPL26_ILS_INCR_ROUTE_FIRST"):
            self.assertIn(name, self.flags)

    def test_never_fatal_when_run_dir_is_unwritable(self):
        """Provenance must not be able to break a run."""
        stub = _Stub("/nonexistent/path/that/cannot/be/created")
        with mock.patch.dict(os.environ, {}, clear=False):
            self.write(stub)  # must not raise

    def test_never_fatal_when_run_dir_missing_entirely(self):
        class NoRunDir(_Stub):
            def __init__(self):
                self._MANIFEST_FLAGS = _real_flags()
        self.write(NoRunDir())  # must not raise

    def test_never_fatal_when_the_object_is_missing_everything(self):
        """Belt and braces: the writer is called early in optimize(), so it must
        survive a partially-initialised object too."""
        class Bare:
            pass
        self.write(Bare())  # must not raise


if __name__ == "__main__":
    unittest.main()
