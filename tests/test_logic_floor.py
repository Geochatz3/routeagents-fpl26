"""Tests the pure logic-floor admission policy.

Admission requires a routed, parseable, sparse, net-light, macro-dominated
critical-path signature with consistent solves. Net-heavy paths, retimeable
logic chains, disagreeing solves, broad near-critical populations, unrouted
states, and parse failures must refuse.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimizer.logic_floor import (  # noqa: E402
    LFPath,
    LOGIC_FLOOR_NPATHS,
    evaluate_logic_floor,
    parse_lf_output,
)


def _miniisp_paths():
    """Return synthetic paths representing a sparse, net-light floor state.

    The worst path is macro-dominated with 2.271 ns total delay, 1.911 ns logic
    delay, 0.360 ns route delay, and seven hops; the other 31 paths are at
    least 0.25 ns away.
    """
    paths = [LFPath(slack=-0.850, dp=2.271, lg=1.911, rt=0.360, hops=7,
                    lut=0.111, macro=1.800)]
    for i in range(LOGIC_FLOOR_NPATHS - 1):
        paths.append(LFPath(slack=-0.580 + i * 0.002, dp=1.9, lg=1.5,
                            rt=0.4, hops=8, lut=0.3, macro=1.2))
    return paths


def _eval(paths, **over):
    kw = dict(routed=True, paths=paths, period_ns=1.570,
              wns_b1=-0.904, wns_b3=-0.850)
    kw.update(over)
    return evaluate_logic_floor(**kw)


class FireCaseTests(unittest.TestCase):
    def test_miniisp_floor_fires_with_k3_bound(self):
        v = _eval(_miniisp_paths())
        self.assertTrue(v.fire, v.reason)
        # The arithmetic: bound_slack = -0.850 + (0.360 - 0.045*7) = -0.805
        # bound_alpha = 1000/(1.570+0.805) - 1000/(1.570+0.850) = 7.83 MHz
        self.assertAlmostEqual(v.bound_alpha_mhz, 7.83, delta=0.05)
        self.assertGreaterEqual(v.logic_frac, 0.80)
        self.assertGreaterEqual(v.macro_frac, 0.50)


class RefuseCaseTests(unittest.TestCase):
    def test_net_heavy_corescore_class_refuses_on_bound(self):
        # mid-run net-dominated state: route 1.1ns over 9 hops => harvestable
        paths = _miniisp_paths()
        paths[0] = LFPath(slack=-0.850, dp=2.271, lg=1.171, rt=1.100,
                          hops=9, lut=0.2, macro=0.9)
        v = _eval(paths)
        self.assertFalse(v.fire)
        self.assertIn("bound_alpha", v.reason)

    def test_retimeable_lut_chain_refuses_on_macro_share(self):
        # 84% logic but it's a LUT chain — ILS/retime could earn; must refuse
        paths = _miniisp_paths()
        paths[0] = LFPath(slack=-0.850, dp=2.271, lg=1.911, rt=0.360,
                          hops=7, lut=1.700, macro=0.150)
        v = _eval(paths)
        self.assertFalse(v.fire)
        self.assertIn("macro_frac", v.reason)

    def test_two_solve_disagreement_refuses(self):
        v = _eval(_miniisp_paths(), wns_b1=-0.940)  # 0.090 > 0.080
        self.assertFalse(v.fire)
        self.assertIn("two-solve", v.reason)

    def test_dense_window_of_floor_paths_still_fires(self):
        # The top 32 paths span only 15.5 ps and are uniformly logic-floor-bound,
        # so the detector must fire even if unseen paths reduce the harvest.
        paths = [LFPath(slack=-0.850 + i * 0.0005, dp=2.271, lg=1.911,
                        rt=0.360, hops=7, lut=0.037, macro=1.874)
                 for i in range(LOGIC_FLOOR_NPATHS)]
        v = _eval(paths)
        self.assertTrue(v.fire, v.reason)

    def test_live_box6_measurement_fires(self):
        # Pinned to the lf_drill.log numbers (the real Tcl against the
        # real -0.847 artifact): 4x -0.847, 8x -0.844, 4x -0.842, 8x -0.841,
        # 8x -0.831; lg 1.909-1.911, rt 0.342-0.357, hops 7, macro 1.874.
        raw = ([(-0.847, 2.268, 1.911, 0.357)] * 4
               + [(-0.844, 2.265, 1.911, 0.354)] * 8
               + [(-0.842, 2.262, 1.909, 0.353)] * 4
               + [(-0.841, 2.261, 1.909, 0.352)] * 8
               + [(-0.831, 2.253, 1.911, 0.342)] * 8)
        paths = [LFPath(slack=sl, dp=dp, lg=lg, rt=rt, hops=7,
                        lut=0.037, macro=1.874) for sl, dp, lg, rt in raw]
        v = _eval(paths, wns_b3=-0.847, wns_b1=-0.904)
        self.assertTrue(v.fire, v.reason)
        # window min bound: -0.847 + (0.357 - 0.315) = -0.805
        self.assertAlmostEqual(v.bound_alpha_mhz, 7.3, delta=0.8)
        self.assertGreaterEqual(v.macro_frac, 0.95)

    def test_unrouted_refuses(self):
        v = _eval(_miniisp_paths(), routed=False)
        self.assertFalse(v.fire)

    def test_missing_solve_values_refuse(self):
        self.assertFalse(_eval(_miniisp_paths(), wns_b1=None).fire)
        self.assertFalse(_eval(_miniisp_paths(), wns_b3=None).fire)

    def test_short_path_list_refuses(self):
        v = _eval(_miniisp_paths()[:10])
        self.assertFalse(v.fire)
        self.assertIn("coverage", v.reason)

    def test_low_logic_fraction_refuses(self):
        paths = _miniisp_paths()
        # logic 55%, net small enough to pass the bound gate: route 0.34
        # over 7 hops -> bound ~= +0.025ns ~= 4.3 MHz <= 10
        paths[0] = LFPath(slack=-0.850, dp=0.756, lg=0.416, rt=0.340,
                          hops=7, lut=0.05, macro=0.35)
        v = _eval(paths)
        self.assertFalse(v.fire)
        self.assertIn("logic_frac", v.reason)


class ParseTests(unittest.TestCase):
    def test_parse_roundtrip_and_malformed_lines_dropped(self):
        text = (
            "LFMETA routed=1\n"
            "LFMETA npaths=3\n"
            "LFPATH i=0 slack=-0.850 dp=2.271 lg=1.911 rt=0.360 hops=7 "
            "lut=0.111 macro=1.800\n"
            "LFPATH i=1 slack=-0.580 dp=BAD lg=1.5 rt=0.4 hops=8 lut=0.3 "
            "macro=1.2\n"
            "LFPATH i=2 slack=-0.578 dp=1.9 lg=1.5 rt=0.4 hops=0 lut=0.3 "
            "macro=1.2\n"
            "LFDONE\n")
        routed, paths = parse_lf_output(text)
        self.assertTrue(routed)
        # line 1 malformed (dp=BAD), line 2 dropped (hops=0) -> only 1 parsed
        self.assertEqual(len(paths), 1)
        self.assertAlmostEqual(paths[0].macro, 1.8)

    def test_unrouted_meta(self):
        routed, _ = parse_lf_output("LFMETA routed=0\nLFDONE\n")
        self.assertFalse(routed)


class TransportContractTests(unittest.TestCase):
    def test_tcl_is_single_line(self):
        # v5.5 review-1 BLOCKER 1: the vivado_run_tcl transport sendlines the
        # payload and expects ONE prompt — an embedded newline returns early
        # and desyncs the session for every subsequent call.
        from optimizer.logic_floor import LOGIC_FLOOR_TCL
        self.assertNotIn("\n", LOGIC_FLOOR_TCL)
        self.assertNotIn("\r", LOGIC_FLOOR_TCL)

    def test_tcl_classifies_by_resource_type_token(self):
        # Timing arcs place the resource type before the arc name and may
        # contain bracketed bus indices. The parser must accept both forms.
        from optimizer.logic_floor import LOGIC_FLOOR_TCL
        self.assertIn(r"{([A-Za-z0-9_]+)\s+\(Prop_[^)]*\)\s+([0-9.]+)}",
                      LOGIC_FLOOR_TCL)
        self.assertIn("string match LUT*", LOGIC_FLOOR_TCL)
        self.assertIn("string match DSP*", LOGIC_FLOOR_TCL)


class ConsistencyGuardTests(unittest.TestCase):
    def test_worst_slack_must_match_b3(self):
        # review-1 finding 7: wrong-group/stale-timing reads must not attest.
        v = _eval(_miniisp_paths(), wns_b3=-0.870, wns_b1=-0.904)
        self.assertFalse(v.fire)
        self.assertIn("disagrees with B3", v.reason)


class B1AdmissionTests(unittest.TestCase):
    """Tests physics-based admission for logic-floor decisions.

    A passing decision admits the floor; failure of any admission predicate
    must refuse it.
    """

    def _adm(self, paths, wns_banked=-0.904, **over):
        from optimizer.logic_floor import evaluate_b1_admission
        kw = dict(paths=paths, period_ns=1.570, wns_banked=wns_banked)
        kw.update(over)
        return evaluate_b1_admission(**kw)

    def _b1_paths(self):
        # The paths model a DSP-internal family whose slacks are shifted 54 ps
        # from a recoverable logic-floor state.
        paths = [LFPath(slack=-0.904, dp=2.271, lg=1.911, rt=0.360, hops=7,
                        lut=0.111, macro=1.800)]
        for i in range(31):
            paths.append(LFPath(slack=-0.890 + i * 0.002, dp=2.2, lg=1.85,
                                rt=0.35, hops=7, lut=0.1, macro=1.72))
        return paths

    def test_miniisp_b1_admits_on_any_hardware(self):
        v = self._adm(self._b1_paths())
        self.assertTrue(v.fire, v.reason)
        self.assertLessEqual(v.bound_alpha_mhz, 18.0)
        self.assertGreaterEqual(v.macro_frac, 0.90)

    def test_aws_measured_b1_state_admits_at_18_not_12(self):
        # A macro-dominated path near the admission boundary is classified as
        # floor-limited and must be admitted.
        paths = [LFPath(slack=-0.904, dp=2.271, lg=1.880, rt=0.400, hops=7,
                        lut=0.049, macro=1.831)]
        for i in range(31):
            paths.append(LFPath(slack=-0.889 + i * 0.001, dp=2.25,
                                lg=1.86, rt=0.385, hops=7, lut=0.05,
                                macro=1.80))
        v = self._adm(paths)
        self.assertTrue(v.fire, v.reason)
        # pin the MEASURED bound (14.38): a future threshold cut below it
        # must turn this test red (delta-review item 3).
        self.assertGreaterEqual(v.bound_alpha_mhz, 14.0)
        self.assertLessEqual(v.bound_alpha_mhz, 15.0)

    def test_vexriscv_replace_fabric_refuses_on_macro(self):
        # A LUT-dominated, retimeable fabric path must be refused.
        # Net delay stays near the per-hop floor to isolate the macro check.
        paths = [LFPath(slack=-0.785, dp=2.1, lg=1.55, rt=0.42, hops=9,
                        lut=1.40, macro=0.05)]
        for i in range(31):
            paths.append(LFPath(slack=-0.770 + i * 0.002, dp=2.0, lg=1.45,
                                rt=0.41, hops=9, lut=1.3, macro=0.0))
        v = self._adm(paths, wns_banked=-0.785)
        self.assertFalse(v.fire)
        self.assertIn("macro_frac", v.reason)

    def test_net_heavy_refuses_on_bound(self):
        # the WHOLE window is net-heavy (bound is a min over paths, so a
        # single macro path among floor-bound neighbors would not dominate).
        paths = [LFPath(slack=-0.904, dp=2.271, lg=1.0, rt=1.271, hops=9,
                        lut=0.2, macro=0.8)]
        for i in range(31):
            paths.append(LFPath(slack=-0.890 + i * 0.002, dp=2.2, lg=1.0,
                                rt=1.2, hops=9, lut=0.2, macro=0.8))
        v = self._adm(paths)
        self.assertFalse(v.fire)
        self.assertIn("bound_alpha", v.reason)

    def test_wrong_session_state_refuses(self):
        # banked says -0.785 but the session's worst is -0.904 => not the
        # banked solve; must fail closed rather than attest a stranger.
        v = self._adm(self._b1_paths(), wns_banked=-0.785)
        self.assertFalse(v.fire)
        self.assertIn("wrong state", v.reason)

    def test_b2_equivalent_solve_within_tolerance_admits(self):
        # B2 may leave an equivalent-but-not-identical solve open (AWS run:
        # B2 wns == B1 wns exactly; allow up to 0.10 drift).
        v = self._adm(self._b1_paths(), wns_banked=-0.850)
        self.assertTrue(v.fire, v.reason)

    def test_missing_banked_wns_refuses(self):
        v = self._adm(self._b1_paths(), wns_banked=None)
        self.assertFalse(v.fire)

    def test_short_path_list_refuses(self):
        v = self._adm(self._b1_paths()[:10])
        self.assertFalse(v.fire)
        self.assertIn("coverage", v.reason)


def _lf_text(slack0=-0.904, n=32, lg=1.911, rt=0.360, macro=1.800):
    """Canned Tcl output in the wrapper's real grammar (r1 finding 6)."""
    lines = [f"LFMETA npaths={n}"]
    for i in range(n):
        s = slack0 if i == 0 else slack0 + 0.014 + i * 0.002
        lines.append(f"LFPATH i={i} slack={s:.3f} dp=2.271 lg={lg} rt={rt} "
                     f"hops=7 lut=0.111 macro={macro}")
    lines.append("LFDONE")
    return "\n".join(lines)


class B1AdmissionWrapperTests(unittest.TestCase):
    """run_b1_admission_attestation glue: %CLK% substitution, LFDONE check,
    wrong-state -> reopen-banked -> re-attest, exception -> refuse+restart."""

    def _run(self, responder):
        import asyncio
        from optimizer.logic_floor import run_b1_admission_attestation
        calls = []

        async def call_tool(name, args):
            calls.append((name, args))
            return responder(name, args, len(calls))

        v = asyncio.run(run_b1_admission_attestation(
            call_tool, clock_name="*fpl26contest*", period_ns=1.570,
            wns_banked=-0.904, log=lambda m: None,
            recover_dcp="/cand/banked.dcp"))
        return v, calls

    def test_admit_path_single_tcl_call(self):
        v, calls = self._run(lambda n, a, i: _lf_text())
        self.assertTrue(v.fire, v.reason)
        self.assertEqual(len(calls), 1)
        self.assertIn("*fpl26contest*", calls[0][1]["command"])
        self.assertNotIn("%CLK%", calls[0][1]["command"])
        self.assertNotIn("%N%", calls[0][1]["command"])
        self.assertIn("-max_paths 32", calls[0][1]["command"])

    def test_wrong_state_reopens_banked_and_reattests(self):
        def responder(name, args, i):
            if i == 1:
                return _lf_text(slack0=-0.700)   # drifted session
            if i == 2:
                self.assertIn("open_checkpoint", args["command"])
                self.assertIn("/cand/banked.dcp", args["command"])
                return "open ok"
            return _lf_text()                     # banked artifact attests
        v, calls = self._run(responder)
        self.assertTrue(v.fire, v.reason)
        self.assertEqual(len(calls), 3)

    def test_wrong_state_persisting_after_reopen_refuses(self):
        def responder(name, args, i):
            if i == 2:
                return "open ok"
            return _lf_text(slack0=-0.700)
        v, calls = self._run(responder)
        self.assertFalse(v.fire)
        self.assertEqual(len(calls), 3)

    def test_reopen_error_refuses(self):
        def responder(name, args, i):
            if i == 2:
                return "ERROR: [Common 17-69] Command failed: no such file"
            return _lf_text(slack0=-0.700)
        v, calls = self._run(responder)
        self.assertFalse(v.fire)
        self.assertIn("reopen failed", v.reason)

    def test_missing_lfdone_refuses_and_restarts(self):
        def responder(name, args, i):
            if i == 1:
                return "garbled transport"
            return "OK"
        v, calls = self._run(responder)
        self.assertFalse(v.fire)
        self.assertIn("did not complete", v.reason)
        self.assertIn("vivado_restart_vivado", [c[0] for c in calls])

    def test_tool_exception_refuses_and_restarts(self):
        def responder(name, args, i):
            if i == 1:
                raise RuntimeError("transport dead")
            return "OK"
        v, calls = self._run(responder)
        self.assertFalse(v.fire)
        self.assertIn("fail closed", v.reason)
        self.assertIn("vivado_restart_vivado", [c[0] for c in calls])


if __name__ == "__main__":
    unittest.main()
