"""FPL26_DEEP_FIRST_SIZEGATED — deep-replace[first] for the ILS-size-gated class.

MEASURED (same build 60109c7b, 8-CPU capped, 3500 s wall, gate_ab_jul27.sh):

    ispd16   ship defaults                   alpha  +24.11  (band-gated at FIRST)
             FIRST_BANDED=0                  alpha  +22.85  (band cleared, then
                                                             refused on cost)
             FIRST_BANDED=0 + NO_DOUBLE_RESERVE=1
                                             alpha +115.37  VALID_OPTIMIZED
    boom_v1  ship defaults                   alpha  +41.43
             same two flags                  alpha  +41.43  IDENTICAL (no-op)

The two gates that blocked ispd16, both read from live logs:
  1. DEEP-extreme band = `failing >= 100k AND |WNS| >= 10 ns`, fitted on boom
     (|WNS| 19.16). ispd16 has MORE failing endpoints than boom (242,906 vs
     217,988) but |WNS| 7.75, so it misses on the |WNS| half alone.
  2. Affordability refused by a MARGIN: need 2315 x 1.3 + 300 = 3309 s vs
     3049 s remaining, while the chain ACTUALLY took 1976 s.

Neither raw flag is safe globally — FIRST_BANDED=0 re-arms corescore where the
stage costs ~63 MHz, and NO_DOUBLE_RESERVE=1 changes affordability for every
design. The size gate confines both to the class where ILS never arms.

These tests drive the real helper and the real decision method (checklist item
15: a test that does not enter the changed code proves nothing).
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dcp_optimizer as d
from dcp_optimizer import DCPOptimizer, deep_first_sizegated_enabled
from optimizer.ils_polish import DEFAULT_MAX_CELLS, ILSPolishConfig
# The band's OWN failing-endpoint threshold — the size gate reuses it rather
# than fitting a new cut. Its definition comment already reads
# "boom_soc 217k, ispd16 243k; gates to huge designs only".
from optimizer.recipe_router import R1_FAILING_ENDPOINTS_MIN

# Measured primitive cell counts (tests/test_mid_tail_floor.py, same corpus).
CELLS = {"vexriscv_v1": 3_373, "mini_isp": 8_414, "3d": 30_874,
         "optical": 84_422, "finn": 157_166, "corescore": 252_741,
         "boom_v1": 379_048, "ispd16": 532_160}
# Measured failing-endpoint counts, from live agent logs on the shipped build.
FAILING = {"ispd16": 242_906, "boom_v1": 217_988, "boom_v2": 220_131,
           "corescore": 39_008}
# Imported, NOT re-declared: if the production constant moves, the lever's
# scope moves with it and these tests must move too, loudly.
MAX_CELLS = DEFAULT_MAX_CELLS

_FLAGS = ("FPL26_DEEP_FIRST_SIZEGATED", "FPL26_NO_DEEP_FIRST_SIZEGATED",
          "FPL26_DEEP_REPLACE_FIRST_BANDED",
          "FPL26_DEEP_REPLACE_NO_DOUBLE_RESERVE")


class _EnvClean(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _FLAGS}
        for k in _FLAGS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


class FlagDisciplineTests(_EnvClean):

    def test_default_is_off(self):
        self.assertFalse(deep_first_sizegated_enabled())

    def test_truthy_spellings(self):
        for v in ("1", "true", "on", "yes", "TRUE", " On "):
            with self.subTest(v=v):
                os.environ["FPL26_DEEP_FIRST_SIZEGATED"] = v
                self.assertTrue(deep_first_sizegated_enabled())

    def test_non_truthy_stays_off(self):
        for v in ("0", "false", "off", "no", ""):
            with self.subTest(v=v):
                os.environ["FPL26_DEEP_FIRST_SIZEGATED"] = v
                self.assertFalse(deep_first_sizegated_enabled())

    def test_kill_switch_wins(self):
        os.environ["FPL26_DEEP_FIRST_SIZEGATED"] = "1"
        os.environ["FPL26_NO_DEEP_FIRST_SIZEGATED"] = "1"
        self.assertFalse(deep_first_sizegated_enabled())

    def test_both_flags_are_manifested(self):
        """Otherwise the '[effective-config] ON:' firing check cannot report
        the treatment either way — a probe that cannot see the treatment is
        broken, not a null."""
        self.assertIn("FPL26_DEEP_FIRST_SIZEGATED", DCPOptimizer._MANIFEST_FLAGS)
        self.assertIn("FPL26_NO_DEEP_FIRST_SIZEGATED",
                      DCPOptimizer._MANIFEST_FLAGS)

    def test_makefile_arms_both_launch_branches(self):
        mk = (Path(__file__).resolve().parent.parent / "Makefile").read_text(
            errors="ignore")
        self.assertEqual(
            mk.count("FPL26_DEEP_FIRST_SIZEGATED=$(if $(DEEP_FIRST_SIZEGATED)"),
            2, "both the wrapper and the || fallback branch must arm it, or "
               "an A/B is unattributable (jul30 'Makefile not in ship surface')")


class SizeGateSeparationTests(_EnvClean):
    """The class boundary must be wide and must exclude corescore."""

    def test_only_boom_and_ispd16_are_size_gated(self):
        gated = {k for k, c in CELLS.items() if c > MAX_CELLS}
        self.assertEqual(gated, {"boom_v1", "ispd16"})

    def test_corescore_is_excluded_with_a_wide_margin(self):
        """corescore is THE design where deep-replace at FIRST is measured
        harmful (~-63 MHz). It must never be admitted."""
        self.assertLess(CELLS["corescore"], MAX_CELLS)
        gap = CELLS["boom_v1"] - CELLS["corescore"]
        self.assertGreater(gap, 100_000,
                           "boundary must be a wide gap, not a fitted edge")
        self.assertGreater(MAX_CELLS - CELLS["corescore"], 40_000)


def _make_opt(tmp: Path, cells, max_cells="REAL") -> DCPOptimizer:
    """Real ILSPolishConfig, not a mock.

    A MagicMock makes every attribute truthy, which silently satisfies the
    stage's own enable gates AND hides the real max_cells — so the lever's
    entire scope constant went untested (review aug06). Here the config is
    real; only the two enable flags are set, exactly as the ship path does
    (Makefile FPL26_DEEP_REPLACE / _FIRST both arm), and max_cells keeps its
    production default unless a test is deliberately probing an unknown one.
    """
    opt = DCPOptimizer(api_key="test", run_dir=tmp)
    opt._input_cell_count = cells
    cfg = ILSPolishConfig()
    cfg.deep_replace_enabled = True
    cfg.deep_replace_first_enabled = True
    if max_cells != "REAL":
        cfg.max_cells = max_cells
    opt._ils_polish_cfg = cfg
    return opt


class DecisionPathTests(_EnvClean):
    """Drive the REAL _deep_replace_sibling_after_polish gate.

    The method is stopped at deep_replace_should_run, whose kwargs are the
    subject: require_physics_band and reserve_already_in_deadline.
    """

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()
        super().tearDown()

    def _call(self, cells, *, flag, stage="first", max_cells="REAL",
              failing=None, wns=-7.75):
        """Returns the kwargs deep_replace_should_run was called with."""
        if flag:
            os.environ["FPL26_DEEP_FIRST_SIZEGATED"] = "1"
        opt = _make_opt(self.tmp_path, cells, max_cells)
        pristine = self.tmp_path / "in.dcp"
        pristine.write_bytes(b"X")
        opt.input_dcp_path = pristine
        _f = FAILING["ispd16"] if failing is None else failing
        opt._phase1_failing_endpoints_for_features = lambda: _f
        opt._phase1_wns_for_features = lambda: wns
        opt.max_wall_seconds = 3500.0
        captured = {}

        def fake_should_run(**kw):
            captured.update(kw)
            return (False, "stopped-by-test")

        # The method does a local `from optimizer.deep_replace_sibling import
        # ... deep_replace_should_run`, so patch it at its source module.
        import optimizer.deep_replace_sibling as drs
        with mock.patch.object(drs, "deep_replace_should_run",
                               fake_should_run):
            asyncio.run(opt._deep_replace_sibling_after_polish(
                deadline=__import__("time").time() + 3049.0,
                wns_tcl="SLACK", stage_label=stage,
                cost_basis_override=2315.0))
        return captured

    def test_ispd16_class_waives_both_gates(self):
        kw = self._call(CELLS["ispd16"], flag=True)
        self.assertFalse(kw.get("require_physics_band"),
                         "the DEEP-extreme band must be waived for the "
                         "size-gated class (ispd16 misses on |WNS| alone)")
        self.assertTrue(kw.get("reserve_already_in_deadline"),
                        "the double-subtracted reserve must be waived")

    def test_corescore_class_is_untouched_even_with_the_flag_on(self):
        kw = self._call(CELLS["corescore"], flag=True)
        self.assertTrue(kw.get("require_physics_band"),
                        "corescore must KEEP the band — FIRST costs it ~63 MHz")
        self.assertFalse(kw.get("reserve_already_in_deadline"))

    def test_flag_off_is_identical_to_shipped(self):
        on = self._call(CELLS["ispd16"], flag=False)
        self.assertTrue(on.get("require_physics_band"))
        self.assertFalse(on.get("reserve_already_in_deadline"))

    def test_tail_stage_is_never_affected(self):
        """The waiver is FIRST-only; the tail keeps shipped behavior."""
        kw = self._call(CELLS["ispd16"], flag=True, stage="tail")
        self.assertFalse(kw.get("reserve_already_in_deadline"))

    def test_unknown_cell_count_fails_closed(self):
        for cells in (None, 0):
            with self.subTest(cells=cells):
                kw = self._call(cells, flag=True)
                self.assertTrue(kw.get("require_physics_band"))
                self.assertFalse(kw.get("reserve_already_in_deadline"))

    def test_unknown_max_cells_fails_closed(self):
        kw = self._call(CELLS["ispd16"], flag=True, max_cells=None)
        self.assertTrue(kw.get("require_physics_band"))
        self.assertFalse(kw.get("reserve_already_in_deadline"))

    def test_kill_switch_restores_shipped_behavior(self):
        os.environ["FPL26_NO_DEEP_FIRST_SIZEGATED"] = "1"
        kw = self._call(CELLS["ispd16"], flag=True)
        self.assertTrue(kw.get("require_physics_band"))
        self.assertFalse(kw.get("reserve_already_in_deadline"))

    # --- the aug06 review BLOCKER: size alone is not enough ---------------

    def test_large_but_SHALLOW_hidden_design_is_refused(self):
        """THE blocker. corescore's physics at a size-gated cell count.

        Waiving the band on cell count ALONE also waives its |WNS| floor, so a
        hidden 400k-cell near-met benchmark would arm a full re-place at FIRST
        — corescore's exact profile, where that stage is measured to cost
        ~63 MHz (+80.94 without vs +17.56 with). At FIRST the loss is the whole
        downstream pipeline's alpha, not just gamma. The band's own
        failing-endpoint half (R1_FAILING_ENDPOINTS_MIN, no new constant)
        excludes it while preserving ispd16.
        """
        kw = self._call(400_000, flag=True,
                        failing=FAILING["corescore"], wns=-1.24)
        self.assertTrue(kw.get("require_physics_band"),
                        "a large SHALLOW design must keep the band — this is "
                        "corescore's measured-harmful profile")
        self.assertFalse(kw.get("reserve_already_in_deadline"))

    def test_large_and_near_met_hidden_design_is_refused(self):
        kw = self._call(350_000, flag=True, failing=200, wns=-0.30)
        self.assertTrue(kw.get("require_physics_band"))

    def test_ispd16_still_arms_after_the_narrowing(self):
        """The narrowing must not cost the +91 alpha it was built for."""
        kw = self._call(CELLS["ispd16"], flag=True,
                        failing=FAILING["ispd16"], wns=-7.75)
        self.assertFalse(kw.get("require_physics_band"))
        self.assertTrue(kw.get("reserve_already_in_deadline"))

    def test_both_boom_variants_are_unaffected_by_the_narrowing(self):
        for name, cells in (("boom_v1", CELLS["boom_v1"]),
                            ("boom_v2", CELLS["boom_v1"])):
            with self.subTest(design=name):
                kw = self._call(cells, flag=True,
                                failing=FAILING[name], wns=-19.16)
                # They clear the real band anyway; the waiver is harmless here.
                self.assertFalse(kw.get("require_physics_band"))

    def test_failing_endpoint_floor_is_the_bands_own_constant(self):
        """No new fitted constant: reuse the band's existing threshold."""
        self.assertEqual(R1_FAILING_ENDPOINTS_MIN, 100_000)
        self.assertGreater(FAILING["ispd16"], R1_FAILING_ENDPOINTS_MIN)
        self.assertLess(FAILING["corescore"], R1_FAILING_ENDPOINTS_MIN)

    def test_real_ils_config_supplies_the_scope_constant(self):
        """Pin the REAL max_cells, not a mock (review aug06)."""
        opt = DCPOptimizer(api_key="test", run_dir=self.tmp_path)
        self.assertEqual(opt._ils_polish_cfg.max_cells, 300_000)
        self.assertEqual(DEFAULT_MAX_CELLS, 300_000)


class ArmArithmeticTests(unittest.TestCase):
    """Execute the REAL affordability comparison, not a stub.

    Every other test here stubs deep_replace_should_run and asserts its kwargs,
    so the actual `remaining >= need` decision was never exercised. The margin
    is thin and worth pinning: MEASURED across six ispd16 runs on four boxes
    over two days, the remaining wall at the deep-replace[first] hook was
        3049 / 3051 / 3052 / 3058 / 3059 / 3059 s   (a 10 s spread)
    against need = 532,160 cells x 0.00435 x 1.3 = 3009.4 s.
    """

    def _need(self, cells):
        from optimizer.deep_replace_sibling import (
            DEEP_REPLACE_PR_S_PER_CELL, DEEP_REPLACE_COST_MARGIN)
        return cells * DEEP_REPLACE_PR_S_PER_CELL * DEEP_REPLACE_COST_MARGIN

    def test_ispd16_arms_at_the_observed_remaining_wall(self):
        from optimizer.deep_replace_sibling import deep_replace_should_run
        need = self._need(CELLS["ispd16"])
        self.assertLess(need, 3049.0,
                        f"need {need:.0f}s must fit the WORST observed "
                        f"remaining (3049s) or ispd16 is a coin flip")
        run, why = deep_replace_should_run(
            enabled=True, pristine_dcp="/tmp/x.dcp",
            failing_endpoint_count=FAILING["ispd16"], wns_magnitude_ns=7.75,
            remaining_s=3049.0, cost_basis_s=need / 1.3,
            finalize_reserve_s=300.0,
            failing_endpoints_min=R1_FAILING_ENDPOINTS_MIN,
            wns_min_ns=10.0,
            require_physics_band=False, reserve_already_in_deadline=True)
        self.assertTrue(run, f"should arm at the observed wall, got: {why}")

    def test_it_declines_cleanly_when_the_wall_is_short(self):
        """A slower box must DECLINE (free) rather than arm and overrun."""
        from optimizer.deep_replace_sibling import deep_replace_should_run
        need = self._need(CELLS["ispd16"])
        run, why = deep_replace_should_run(
            enabled=True, pristine_dcp="/tmp/x.dcp",
            failing_endpoint_count=FAILING["ispd16"], wns_magnitude_ns=7.75,
            remaining_s=need - 100.0, cost_basis_s=need / 1.3,
            finalize_reserve_s=300.0,
            failing_endpoints_min=R1_FAILING_ENDPOINTS_MIN,
            wns_min_ns=10.0,
            require_physics_band=False, reserve_already_in_deadline=True)
        self.assertFalse(run)

    def test_the_margin_is_documented_not_accidental(self):
        """If either constant moves, the margin must be re-derived."""
        from optimizer.deep_replace_sibling import (
            DEEP_REPLACE_PR_S_PER_CELL, DEEP_REPLACE_COST_MARGIN)
        self.assertEqual(DEEP_REPLACE_COST_MARGIN, 1.3)
        need = self._need(CELLS["ispd16"])
        margin = 3049.0 - need
        self.assertGreater(margin, 0.0)
        # The model over-predicts: the chain actually measured 1976s
        # (place 1339 + route 637), so the true buffer is far larger than the
        # nominal margin. Pinned so a coefficient change surfaces here.
        self.assertGreater(need, 1976.0,
                           "the sizing model should remain CONSERVATIVE "
                           "relative to the measured chain cost")


if __name__ == "__main__":
    unittest.main()
