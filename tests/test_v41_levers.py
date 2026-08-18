"""v4.1 rev2 bundle (aug05, branch v41-eggs) — unit tests for the three
v4.1 planks (PREREG_V41_REV2_aug05.md):

  1. FPL26_MUX_MD5_TRUST — md5-trust finalize verify: a winning
     candidate whose registration-time md5+size still match skips the
     120s-budget structural re-open (which twice ate a verified
     +92.38-class digit winner by <0.1 s); any mismatch falls back to
     the unchanged _structural_validate_dcp path.
  2. FPL26_OWNFRONT_RETIME_CANDIDATE — the UNIFIED own-front retime
     candidate: ONE second shallow candidate, front selected per run
     (|wns_in| [0.60, 1.00) -> ETO/q07 chain, [1.00, 1.05] -> WLD
     chain, both frozen verbatim); when armed the vex2 candidate defers
     (reason=ownfront_supersedes) so exactly ONE retime candidate
     spends wall per run.  Off/killed -> exact shipped v4.0.1 behavior.
  3. FPL26_CORESCORE_ROUTE_RUNG redesign (break rung) — covered in the
     break-rung classes below (added in the plank-3 commit).

House contract under test (mirroring v4.0's): flags DEFAULT OFF in
python, armed ONLY via the Makefile on BOTH launch branches; kill
switches win; feature keys are measured-characteristic only (NO
md5/name keys — the md5 in plank 1 keys candidate FILE IDENTITY at the
finalize MUX, not design identity); disabled-audit lines gated behind
v40_flag_env_present; flag-off runs byte-identical to 1ba25ec.
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dcp_optimizer as d  # noqa: E402

ALL_V41_ENVS = (
    "FPL26_OWNFRONT_RETIME_CANDIDATE",
    "FPL26_NO_OWNFRONT_RETIME_CANDIDATE",
    "FPL26_MUX_MD5_TRUST",
    "FPL26_NO_MUX_MD5_TRUST",
    # v4.1.2 (aug06): spam determinizer third shallow candidate.
    "FPL26_SPAM_DETERMINIZER_CANDIDATE",
    "FPL26_NO_SPAM_DETERMINIZER_CANDIDATE",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Eval-box-like env: none of the v4.1 flags set."""
    for k in ALL_V41_ENVS:
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------------------
# PLANK 2 — FPL26_OWNFRONT_RETIME_CANDIDATE (unified own-front retime)
# ---------------------------------------------------------------------------


class TestOwnfrontFlagDiscipline:
    def test_default_off(self):
        """No env set -> OFF.  The python default must never arm a lever;
        arming is Makefile-only (never-touch-validated-paths rule)."""
        assert d.ownfront_retime_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "on", "yes", " 1 ", "TRUE"])
    def test_armed_by_truthy(self, val, monkeypatch):
        monkeypatch.setenv("FPL26_OWNFRONT_RETIME_CANDIDATE", val)
        assert d.ownfront_retime_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "off", "", "no", "2"])
    def test_non_truthy_stays_off(self, val, monkeypatch):
        monkeypatch.setenv("FPL26_OWNFRONT_RETIME_CANDIDATE", val)
        assert d.ownfront_retime_enabled() is False

    def test_kill_switch_wins(self, monkeypatch):
        monkeypatch.setenv("FPL26_OWNFRONT_RETIME_CANDIDATE", "1")
        monkeypatch.setenv("FPL26_NO_OWNFRONT_RETIME_CANDIDATE", "1")
        assert d.ownfront_retime_enabled() is False

    def test_kill_switch_alone_is_off(self, monkeypatch):
        monkeypatch.setenv("FPL26_NO_OWNFRONT_RETIME_CANDIDATE", "1")
        assert d.ownfront_retime_enabled() is False

    def test_old_wld_flag_is_gone(self):
        """The v4.1-eggs stacking flag is SUPERSEDED, not aliased — a
        stale WLD_RETIME env var must arm nothing."""
        assert not hasattr(d, "shallow_wld_retime_enabled")
        assert not hasattr(d, "wld_retime_subband_match")
        assert not hasattr(d.DCPOptimizer,
                           "_maybe_run_shallow_wld_retime_candidate")


class TestOwnfrontFrontSelect:
    """Front select: [0.60, 1.00) -> eto (vex2-class), [1.00, 1.05] ->
    wld (digit-class), None outside.  None / positive / garbage fail
    OFF.  Boundary law: vex2 (0.946) MUST select eto; digit (1.025)
    MUST select wld; logicnets/spam-class (<0.60) stay UNCOVERED this
    round by design (no low-band extension)."""

    @pytest.mark.parametrize("wns,front", [
        (-0.946, "eto"),   # vex2 — the validated ETO evidence design
        (-0.90, "eto"),    # v4.1.1 low bound inclusive (spam-fallback floor)
        (-0.999, "eto"),   # just below the split
        (-1.00, "wld"),    # split -> WLD side (digit-class begins)
        (-1.025, "wld"),   # digit — the drilled WLD evidence design
        (-1.05, "wld"),    # top bound inclusive (== shallow band max)
    ])
    def test_front_select(self, wns, front):
        assert d.ownfront_retime_front(wns) == front

    @pytest.mark.parametrize("wns", [
        -0.686,    # spam — v4.1.1: OUT (falls back to the vex2 candidate)
        -0.899,    # just below the v4.1.1 floor
        -0.395,    # logicnets-class — deliberately UNCOVERED this round
        -0.313,    # fir-class — below band AND carved out upstream
        -1.051,    # just past the shallow bound
        -1.078,    # optical (mid)
        -1.686,    # mini-isp (mid; drills: attractor flatline)
        -8.0,      # deep
        None,      # unmeasured -> treatments fail OFF
        0.0,       # met
        0.5,       # positive slack
        "garbage",
    ])
    def test_out_of_band(self, wns):
        assert d.ownfront_retime_front(wns) is None

    def test_band_constants(self):
        """[0.60, 1.05] total — same low edge as the vex2 band (no low
        extension this round), same top as the shallow band."""
        assert d.OWNFRONT_RETIME_WNS_MAG_MIN_NS == 0.90  # v4.1.1 floor
        assert d.VEX2_RETIME_WNS_MAG_MIN_NS == 0.60      # fallback band unchanged
        assert (d.OWNFRONT_RETIME_WNS_MAG_MAX_NS
                == d.RECIPE_PASS_SHALLOW_WNS_MAG_MAX_NS == 1.05)
        assert d.OWNFRONT_RETIME_SPLIT_NS == 1.00

    def test_split_separates_the_evidence_designs(self):
        """The split exists to route vex2 (0.946) to its validated ETO
        front and digit (1.025) to its drilled WLD front."""
        assert 0.946 < d.OWNFRONT_RETIME_SPLIT_NS < 1.025

    def test_every_in_band_value_gets_exactly_one_front(self):
        for mag in (0.90, 0.95, 0.9999, 1.00, 1.02, 1.05):  # v4.1.1 floor 0.90
            assert d.ownfront_retime_front(-mag) in ("eto", "wld")


class TestOwnfrontFrozenChains:
    """Both sub-chains FROZEN VERBATIM from their evidence
    (feedback_same_build_not_just_same_config)."""

    def test_eto_chain_is_vex2_chain_verbatim(self):
        assert d.VEX2_RETIME_TCL == (
            "route_design -unroute",
            "place_design -unplace",
            "place_design -directive ExtraTimingOpt",
            "phys_opt_design -retime",
            "phys_opt_design -directive AggressiveExplore",
            "route_design -directive AggressiveExplore",
        )

    def test_wld_chain_frozen_verbatim(self):
        assert d.WLD_RETIME_TCL == (
            "route_design -unroute",
            "place_design -unplace",
            "place_design -directive WLDrivenBlockPlacement",
            "phys_opt_design -retime",
            "phys_opt_design -directive AggressiveExplore",
            "route_design -directive AggressiveExplore",
        )

    def test_chains_differ_in_front_only(self):
        """retime -> poAE -> routeAE tail shared; only the place front
        differs (the drill law: retime pays on the OWN winning front)."""
        assert d.WLD_RETIME_TCL[3:] == d.VEX2_RETIME_TCL[3:]
        assert d.WLD_RETIME_TCL[:2] == d.VEX2_RETIME_TCL[:2]
        assert d.WLD_RETIME_TCL[2] != d.VEX2_RETIME_TCL[2]

    def test_retime_after_place_in_both(self):
        for chain in (d.VEX2_RETIME_TCL, d.WLD_RETIME_TCL):
            steps = list(chain)
            assert steps.index("phys_opt_design -retime") == 3
            assert steps[2].startswith("place_design -directive ")

    def test_candidate_selects_between_the_two_frozen_chains(self):
        src = inspect.getsource(
            d.DCPOptimizer._maybe_run_ownfront_retime_candidate)
        assert ('chain = VEX2_RETIME_TCL if front == "eto" '
                'else WLD_RETIME_TCL') in src


class TestOwnfrontGateArithmetic:
    """SINGLE need constant, recomputed honestly for the one-candidate
    shape — the numbers the prereg cites."""

    def test_primary_shallow_gate_untouched(self):
        """need = 1.5x300 + 900 + 1100 = 2450 s, exactly as validated."""
        assert (d.RECIPE_PASS_TIMEOUT_FACTOR * d.RECIPE_PASS_SHALLOW_EXPECTED_S
                + d.RECIPE_PASS_OVERHEAD_RESERVE_S + 1100.0) == 2450.0

    def test_vex2_gate_untouched(self):
        """need2 = 1.5x460 + 540 + 1100 = 2330 s, exactly as v4.0 built
        (still live whenever the ownfront flag is off)."""
        assert (d.RECIPE_PASS_TIMEOUT_FACTOR * d.VEX2_RETIME_EXPECTED_S
                + d.VEX2_RETIME_OVERHEAD_RESERVE_S + 1100.0) == 2330.0

    def test_ownfront_single_need(self):
        """need = 1.5x520 + 540 + 1100 = 2420 s <= 3200 s max remaining
        — fundable on the ship wall AFTER the primary pass, because no
        earlier retime candidate stacks against it (vex2 defers)."""
        cap = d.RECIPE_PASS_TIMEOUT_FACTOR * d.OWNFRONT_RETIME_EXPECTED_S
        assert cap == 780.0
        need = cap + d.OWNFRONT_RETIME_OVERHEAD_RESERVE_S + 1100.0
        assert need == 2420.0
        assert need <= 3200.0
        assert d.OWNFRONT_RETIME_OVERHEAD_RESERVE_S == 540.0
        assert d.OWNFRONT_RETIME_OVERHEAD_RESERVE_S == (
            d.RECIPE_PASS_MEASURE_RESERVE_S + d.RECIPE_PASS_STORE_RESERVE_S
            + d.RECIPE_PASS_REGISTER_RESERVE_S)

    def test_expected_covers_both_measured_chain_ranges(self):
        """expected = 520 s covers ETO/q07 ~420-460 and WLD ~435-517 —
        the 'cover the measured range' convention across BOTH chains."""
        assert d.OWNFRONT_RETIME_EXPECTED_S == 520.0
        assert d.OWNFRONT_RETIME_EXPECTED_S >= 517.0
        assert d.OWNFRONT_RETIME_EXPECTED_S >= d.VEX2_RETIME_EXPECTED_S

    def test_one_candidate_shape_fits_where_stacking_did_not(self):
        """The v4.1-eggs A/B measured the stacked shape self-defeating
        (second candidate skipped at 2195 < 2420 after a vex2 fire).
        With ONE candidate, a typical primary spend leaves >= 2420."""
        max_remaining = 3200.0
        need = (d.RECIPE_PASS_TIMEOUT_FACTOR * d.OWNFRONT_RETIME_EXPECTED_S
                + d.OWNFRONT_RETIME_OVERHEAD_RESERVE_S + 1100.0)
        # stacked shape (historical): primary + vex2 fire -> starved
        assert 2195.0 < need
        # unified shape: primary spend 450-600 s only
        assert max_remaining - 600.0 >= need


class TestOwnfrontSharedFfAuditReuse:
    """The latency audit must REUSE the v4.0 helpers, not fork them —
    one sentinel probe, one parse, one drift gate (fail-closed each)."""

    def test_candidate_calls_shared_helpers(self):
        src = inspect.getsource(
            d.DCPOptimizer._maybe_run_ownfront_retime_candidate)
        assert "_vex2_retime_ff_count" in src
        assert "vex2_retime_ff_drift_ok" in src
        assert 'tag="OWNFRONT-RETIME"' in src
        assert not hasattr(d, "ownfront_retime_parse_ffcount")
        assert not hasattr(d.DCPOptimizer, "_ownfront_retime_ff_count")

    def test_probe_brackets_the_selected_place_step(self):
        """ff_before keys on the SELECTED front's place step (chain[2]),
        not a hardcoded directive — both chains stay audited."""
        src = inspect.getsource(
            d.DCPOptimizer._maybe_run_ownfront_retime_candidate)
        assert "place_step = chain[2]" in src
        assert "if step == place_step:" in src
        assert 'elif step == "phys_opt_design -retime":' in src

    def test_probe_tag_default_preserves_vex2_log_bytes(self):
        sig = inspect.signature(d.DCPOptimizer._vex2_retime_ff_count)
        assert sig.parameters["tag"].default == "VEX2-RETIME"

    def test_drift_gate_still_fails_closed(self):
        assert d.vex2_retime_ff_drift_ok(None, 1682) is False
        assert d.vex2_retime_ff_drift_ok(1678, None) is False
        assert d.vex2_retime_ff_drift_ok(1000, 1010) is True
        assert d.vex2_retime_ff_drift_ok(1000, 1011) is False

    def test_sentinel_parse_still_fails_closed(self):
        assert d.vex2_retime_parse_ffcount("FFCOUNT=1678") == 1678
        assert d.vex2_retime_parse_ffcount("1678") is None


class TestOwnfrontCandidateSiteDiscipline:
    """Source-level invariants of the candidate body: MUX-additive,
    verify=True, fail-closed logging, ONE retime candidate per run."""

    def _src(self):
        return inspect.getsource(
            d.DCPOptimizer._maybe_run_ownfront_retime_candidate)

    def test_registers_with_verify_true(self):
        src = self._src()
        assert "register_final_candidate" in src
        assert "verify=True" in src
        assert 'f"recipe_pass_shallow_ownfront_{front}"' in src

    def test_never_touches_best_valid(self):
        src = self._src()
        assert "self.best_valid" not in src
        assert ".best_valid =" not in src

    def test_disabled_audit_line_gated_behind_env_present(self):
        src = self._src()
        i_gate = src.index(
            'v40_flag_env_present("FPL26_OWNFRONT_RETIME_CANDIDATE"')
        i_line = src.index("OWNFRONT-RETIME: skipped reason=disabled")
        assert i_gate < i_line

    def test_stable_log_keys(self):
        src = self._src()
        for key in ("OWNFRONT-RETIME: skipped reason=disabled ",
                    "OWNFRONT-RETIME: skipped reason=out_of_band ",
                    "OWNFRONT-RETIME: skipped reason=wall ",
                    "OWNFRONT-RETIME: fired ",
                    "OWNFRONT-RETIME: ff-audit ",
                    "OWNFRONT-RETIME: result "):
            assert key in src, key

    def test_vex2_defers_when_ownfront_armed(self):
        """ONE-CANDIDATE RULE: the vex2 candidate defers (logged) BEFORE
        any band/wall work, so exactly one retime candidate spends wall
        per run.  The defer keys on ownfront_retime_enabled() — kill
        switch included — so ownfront off/killed = vex2 exactly as
        shipped v4.0.1."""
        src = inspect.getsource(
            d.DCPOptimizer._maybe_run_vex2_retime_candidate)
        i_defer = src.index("reason=ownfront_supersedes")
        i_gate = src.index("if ownfront_retime_enabled() and ownfront_retime_front(wns_in) is not None:")
        i_band = src.index("vex2_retime_subband_match")
        i_wall = src.index("_budget_remaining")
        # NB: anchor past the docstring's "Stable log keys" listing
        i_fired = src.index("VEX2-RETIME: fired wns_in=")
        assert i_gate < i_defer < i_band < i_wall < i_fired

    def test_vex2_own_gates_untouched_after_the_defer(self):
        """Below the defer branch the vex2 body is the shipped one:
        band gate, recomputed 2330 gate, frozen chain, ff audit."""
        src = inspect.getsource(
            d.DCPOptimizer._maybe_run_vex2_retime_candidate)
        for key in ("VEX2-RETIME: skipped reason=out_of_band ",
                    "VEX2-RETIME: skipped reason=wall ",
                    "VEX2-RETIME: fired ",
                    "VEX2-RETIME: ff-audit ",
                    "VEX2-RETIME: result "):
            assert key in src, key

    def test_ordered_after_vex2_candidate(self):
        """Call order in the shallow bracket: vex2 first (it defers
        itself when ownfront is armed), ownfront second — so flag-off
        keeps the shipped call shape exactly."""
        src = inspect.getsource(d.DCPOptimizer._run_recipe_pass_inner)
        i_vex2 = src.index("_maybe_run_vex2_retime_candidate")
        i_own = src.index("_maybe_run_ownfront_retime_candidate")
        assert i_vex2 < i_own

    def test_fir_carveout_precedes_the_pass_body(self):
        """fir protection ORDERING: the fir_subband_carveout skip returns
        False BEFORE the pass fires; the candidate call site lives inside
        the fired pass body — so fir-like designs never reach it while
        the floor flag is armed (and fir 0.313 is below the 0.60 band
        floor anyway — double protection this round)."""
        src = inspect.getsource(d.DCPOptimizer._run_recipe_pass_inner)
        i_carveout = src.index("reason=fir_subband_carveout")
        i_fired = src.index("RECIPE-PASS: fired")
        i_own = src.index("_maybe_run_ownfront_retime_candidate")
        assert i_carveout < i_fired < i_own

    def test_env_gate_for_disabled_line(self, monkeypatch):
        assert d.v40_flag_env_present(
            "FPL26_OWNFRONT_RETIME_CANDIDATE",
            "FPL26_NO_OWNFRONT_RETIME_CANDIDATE") is False
        monkeypatch.setenv("FPL26_OWNFRONT_RETIME_CANDIDATE", "0")
        assert d.v40_flag_env_present(
            "FPL26_OWNFRONT_RETIME_CANDIDATE",
            "FPL26_NO_OWNFRONT_RETIME_CANDIDATE") is True
        monkeypatch.delenv("FPL26_OWNFRONT_RETIME_CANDIDATE")
        monkeypatch.setenv("FPL26_NO_OWNFRONT_RETIME_CANDIDATE", "1")
        assert d.v40_flag_env_present(
            "FPL26_OWNFRONT_RETIME_CANDIDATE",
            "FPL26_NO_OWNFRONT_RETIME_CANDIDATE") is True


class TestManifestRecords:
    """Every decision-changing flag must be readable back from a banked
    row (the _MANIFEST_FLAGS omission class bit twice)."""

    @pytest.mark.parametrize("flag", ALL_V41_ENVS)
    def test_flag_in_manifest(self, flag):
        assert flag in d.DCPOptimizer._MANIFEST_FLAGS

    def test_old_wld_flag_not_in_manifest(self):
        assert ("FPL26_SHALLOW_WLD_RETIME_CANDIDATE"
                not in d.DCPOptimizer._MANIFEST_FLAGS)


class TestMakefileArming:
    """Both launch branches (wrapper + `||` safety net) must arm every
    v4.1 flag — one branch armed alone measures nothing (jul30)."""

    @pytest.fixture(scope="class")
    def branches(self):
        text = (ROOT / "Makefile").read_text()
        wrapper = [ln for ln in text.splitlines()
                   if "multi_restart_optimize.py" in ln
                   and "FPL26_FIR_SUBBAND_FLOOR" in ln]
        fallback = [ln for ln in text.splitlines()
                    if "dcp_optimizer.py" in ln and ln.strip().startswith("||")
                    and "FPL26_FIR_SUBBAND_FLOOR" in ln]
        assert len(wrapper) == 1 and len(fallback) == 1
        return wrapper[0], fallback[0]

    @pytest.mark.parametrize("var,knob", [
        ("FPL26_OWNFRONT_RETIME_CANDIDATE", "OWNFRONT_RETIME"),
        ("FPL26_MUX_MD5_TRUST", "MUX_MD5_TRUST"),
    ])
    def test_both_branches_armed(self, branches, var, knob):
        expected = f"{var}=$(if $({knob}),$({knob}),1)"
        for branch in branches:
            assert expected in branch

    def test_old_wld_knob_gone(self, branches):
        for branch in branches:
            assert "FPL26_SHALLOW_WLD_RETIME_CANDIDATE" not in branch

    def test_v40_arming_untouched(self, branches):
        """The v4.0 knobs must still be armed on both branches."""
        for var, knob in (
                ("FPL26_VEX2_RETIME_CANDIDATE", "VEX2_RETIME"),
                ("FPL26_MINIISP_RETRY_HOLD", "MINIISP_RETRY_HOLD"),
                ("FPL26_CORESCORE_ROUTE_RUNG", "CORESCORE_ROUTE_RUNG")):
            expected = f"{var}=$(if $({knob}),$({knob}),1)"
            for branch in branches:
                assert expected in branch


# ---------------------------------------------------------------------------
# PLANK 1 — FPL26_MUX_MD5_TRUST (md5-trust finalize verify)
# ---------------------------------------------------------------------------


class TestMd5TrustFlagDiscipline:
    def test_default_off(self):
        assert d.mux_md5_trust_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "on", "yes", " 1 ", "TRUE"])
    def test_armed_by_truthy(self, val, monkeypatch):
        monkeypatch.setenv("FPL26_MUX_MD5_TRUST", val)
        assert d.mux_md5_trust_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "off", "", "no", "2"])
    def test_non_truthy_stays_off(self, val, monkeypatch):
        monkeypatch.setenv("FPL26_MUX_MD5_TRUST", val)
        assert d.mux_md5_trust_enabled() is False

    def test_kill_switch_wins(self, monkeypatch):
        monkeypatch.setenv("FPL26_MUX_MD5_TRUST", "1")
        monkeypatch.setenv("FPL26_NO_MUX_MD5_TRUST", "1")
        assert d.mux_md5_trust_enabled() is False

    def test_kill_switch_alone_is_off(self, monkeypatch):
        monkeypatch.setenv("FPL26_NO_MUX_MD5_TRUST", "1")
        assert d.mux_md5_trust_enabled() is False

    def test_flags_in_manifest(self):
        assert "FPL26_MUX_MD5_TRUST" in d.DCPOptimizer._MANIFEST_FLAGS
        assert "FPL26_NO_MUX_MD5_TRUST" in d.DCPOptimizer._MANIFEST_FLAGS


class TestMd5Digest:
    """mux_md5_digest: (md5, size) or None — anything un-hashable falls
    back to the full structural-validate path (fail closed)."""

    def test_digest_matches_hashlib(self, tmp_path):
        import hashlib
        f = tmp_path / "cand.dcp"
        payload = b"fpl26-candidate-bytes" * 1000
        f.write_bytes(payload)
        dig = d.mux_md5_digest(f)
        assert dig == (hashlib.md5(payload).hexdigest(), len(payload))

    def test_missing_file_none(self, tmp_path):
        assert d.mux_md5_digest(tmp_path / "nope.dcp") is None

    def test_empty_file_none(self, tmp_path):
        """A zero-byte candidate can never be trusted (registration
        would have refused it; a truncated file must not match)."""
        f = tmp_path / "empty.dcp"
        f.write_bytes(b"")
        assert d.mux_md5_digest(f) is None

    def test_mutation_changes_digest(self, tmp_path):
        """The trust key is md5 AND size — either moves on any rewrite."""
        f = tmp_path / "cand.dcp"
        f.write_bytes(b"AAAA")
        before = d.mux_md5_digest(f)
        f.write_bytes(b"AAAB")
        after = d.mux_md5_digest(f)
        assert before != after and before[1] == after[1]

    def test_streams_via_shared_helper(self):
        src = inspect.getsource(d.mux_md5_digest)
        assert "_stream_md5" in src  # one hasher in the codebase


class TestMd5TrustRegistrationSite:
    """Source-level contract of the digest RECORDING site."""

    def test_records_only_verify_true_and_flag_on(self):
        src = inspect.getsource(d.DCPOptimizer.register_final_candidate)
        assert "if verify and mux_md5_trust_enabled():" in src
        assert "mux_md5_digest" in src

    def test_registration_gates_unchanged_order(self):
        """Digest recording happens AFTER every gate (routed/hold/cell) —
        a rejected candidate is never digested/enrolled."""
        src = inspect.getsource(d.DCPOptimizer.register_final_candidate)
        i_cell = src.index("logic-deletion guard")
        i_dig = src.index("mux_md5_digest")
        i_enroll = src.index("candidate ENROLLED")
        assert i_cell < i_dig < i_enroll


class TestMd5TrustFinalizeSite:
    """Source-level contract of the finalize-MUX TRUST site."""

    def _src(self):
        return inspect.getsource(
            d.DCPOptimizer._maybe_ship_final_candidate_mux)

    def test_trusted_log_line(self):
        src = self._src()
        assert "verify=trusted (md5 match + zip-sane, registered " in src

    def test_fallback_path_retained_verbatim(self):
        """Mismatch / missing digest / flag off all reach the UNCHANGED
        _structural_validate_dcp path (never-worse preserved)."""
        src = self._src()
        assert "_structural_validate_dcp" in src
        assert "if not _md5_trusted:" in src
        assert "FAILED " in src and "structural validate" in src
        assert "WNS regressed" in src

    def test_trust_requires_stored_digest_and_match(self):
        src = self._src()
        i_flag = src.index("if mux_md5_trust_enabled():")
        i_match = src.index('_dig_now == (_stored_md5,')
        assert i_flag < i_match
        # trust is keyed on the file NOW, not the registration snapshot
        assert "mux_md5_digest(src)" in src

    def test_disabled_audit_gated_behind_env_present(self):
        src = self._src()
        i_gate = src.index('v40_flag_env_present("FPL26_MUX_MD5_TRUST"')
        i_line = src.index("md5-trust skipped reason=disabled")
        assert i_gate < i_line

    def test_tool_budget_not_raised(self):
        """The fix is to SKIP the re-open, not to fatten it: the
        structural validator's open_checkpoint call must carry no new
        timeout override (eval 1h cap protection)."""
        vsrc = inspect.getsource(d.DCPOptimizer._structural_validate_dcp)
        assert "vivado_open_checkpoint" in vsrc
        assert "timeout" not in vsrc.split("vivado_open_checkpoint")[1] \
            .split(")")[0]

    def test_trusted_branch_cannot_ship_missing_file(self):
        """The existing missing/empty-file MUX guard sits BEFORE the
        trust check — a deleted candidate never ships via trust."""
        src = self._src()
        i_missing = src.index("missing/empty at")
        i_trust = src.index("verify=trusted")
        assert i_missing < i_trust


# ---------------------------------------------------------------------------
# PLANK 3 — FPL26_CORESCORE_ROUTE_RUNG redesign (corescore BREAK rung)
# ---------------------------------------------------------------------------


class TestBreakRungChain:
    """The break chain is FROZEN verbatim from the box1 cs-series
    evidence (-0.612/438.79 deterministic x4 incl. in-session cs3 and
    cross-box box5): unroute -> retime -> poAE -> route AE from the
    BANKED BEST state."""

    def test_frozen_steps_verbatim(self):
        assert d.CORESCORE_BREAK_RUNG_TCL == (
            "route_design -unroute",
            "phys_opt_design -retime",
            "phys_opt_design -directive AggressiveExplore",
            "route_design -directive AggressiveExplore",
        )

    def test_no_unplace_no_replace(self):
        """The break keeps the polished PLACEMENT (unroute only) — the
        mechanism is a routed-state break, not a re-place; cs2's pure
        reroll control (-0.669) proves the retime step is load-bearing."""
        for step in d.CORESCORE_BREAK_RUNG_TCL:
            assert "unplace" not in step and "place_design" not in step

    def test_tail_shared_with_retime_candidates(self):
        """retime -> poAE -> routeAE tail == the shallow retime chains'
        tail (same drilled shape, different entry state)."""
        assert d.CORESCORE_BREAK_RUNG_TCL[1:] == d.VEX2_RETIME_TCL[3:]

    def test_v40_explore_constants_retained_as_history(self):
        """The v4.0 route-Explore constants stay pinned verbatim (the
        test_v40_levers record) but the rung method no longer uses them."""
        assert d.CORESCORE_ROUTE_RUNG_TCL == "route_design -directive Explore"
        assert d.CORESCORE_ROUTE_RUNG_EXPECTED_S == 500.0
        src = inspect.getsource(
            d.DCPOptimizer._maybe_run_corescore_route_rung_postloop)
        assert "CORESCORE_ROUTE_RUNG_TCL" not in src
        assert "CORESCORE_ROUTE_RUNG_EXPECTED_S" not in src
        assert "CORESCORE_BREAK_RUNG_TCL" in src


class TestBreakRungPricing:
    """Measured-cost convention (un-margined need), derived from the
    same reserve constants — the numbers the prereg cites."""

    def test_expected_covers_measured_cost(self):
        """cs3 in-session break measured 806 s (~417 + ~483)."""
        assert d.CORESCORE_BREAK_RUNG_EXPECTED_S == 810.0
        assert d.CORESCORE_BREAK_RUNG_EXPECTED_S >= 806.0

    def test_need_arithmetic(self):
        """need = 810 + 540 = 1350 s; hard deadline 1.5x810 = 1215 s."""
        overhead = d.RECIPE_PASS_POSTLOOP_OVERHEAD_RESERVE_S
        assert overhead == 540.0
        assert d.CORESCORE_BREAK_RUNG_EXPECTED_S + overhead == 1350.0
        assert (d.RECIPE_PASS_TIMEOUT_FACTOR
                * d.CORESCORE_BREAK_RUNG_EXPECTED_S) == 1215.0

    def test_overhead_is_the_shared_reserve_sum(self):
        assert d.RECIPE_PASS_POSTLOOP_OVERHEAD_RESERVE_S == (
            d.RECIPE_PASS_MEASURE_RESERVE_S + d.RECIPE_PASS_STORE_RESERVE_S
            + d.RECIPE_PASS_REGISTER_RESERVE_S)


class TestBreakRungSiteDiscipline:
    """Source-level invariants of the redesigned rung method."""

    def _src(self):
        return inspect.getsource(
            d.DCPOptimizer._maybe_run_corescore_route_rung_postloop)

    def test_handback_defer_removed(self):
        """The v4.0 INERT cause: the rung deferred to handback
        (reason=handback_armed, 0 fires x2).  rev2 removes the defer —
        the rung claims its priced wall AHEAD of the handback exit
        decision (log-only observation of break_due remains)."""
        src = self._src()
        assert "skipped reason=handback_armed" not in src
        assert "claiming priced wall AHEAD of" in src

    def test_gate_order_band_then_best_then_wall(self):
        """Scope gates FIRST (corescore-mid class + banked best exists),
        so the wall reallocation from handback is bounded to exactly
        that class; priced-wall check after them, fail-closed."""
        src = self._src()
        # anchor past the docstring (its "Stable log keys" listing
        # contains the same substrings)
        body = src[src.index('"""', src.index('"""') + 3):]
        i_dis = body.index("reason=disabled")
        i_band = body.index("reason=band")
        i_best = body.index("no_banked_best")
        i_claim = body.index("claiming priced wall")
        i_wall = body.index("reason=wall")
        assert i_dis < i_band < i_best < i_claim < i_wall

    def test_wall_skip_is_fail_closed_and_leaves_handback_whole(self):
        src = self._src()
        assert "fail closed" in src
        assert "stays with handback" in src

    def test_ff_audit_shared_helpers_bracket_retime(self):
        src = self._src()
        assert "_vex2_retime_ff_count" in src
        assert "vex2_retime_ff_drift_ok" in src
        assert 'tag="ROUTE-RUNG[postloop]"' in src
        i_unroute = src.index('if step == "route_design -unroute":')
        i_retime = src.index('elif step == "phys_opt_design -retime":')
        assert i_unroute < i_retime

    def test_registers_break_label_verify_true(self):
        src = self._src()
        assert '"route_rung_mid_break"' in src
        assert "verify=True" in src
        assert "route_rung_mid_break.dcp" in src

    def test_bracket_contract_unchanged(self):
        """Auto-bank suppression, ILS-stage bracket, session-untrusted
        mark, pending-mirror invariant check — all retained."""
        src = self._src()
        assert "_tail_ctrl_suppress_autobank = True" in src
        assert "_in_ils_stage = True" in src
        assert "_finalize_session_untrusted = True" in src
        assert "_pending_best_mirror" in src

    def test_never_writes_best_valid(self):
        src = self._src()
        assert ".best_valid =" not in src
        assert "_best_valid_dcp =" not in src
        assert "self._best_valid_dcp" in src  # READ-only input

    def test_disabled_audit_gated_behind_env_present(self):
        src = self._src()
        i_gate = src.index('v40_flag_env_present("FPL26_CORESCORE_ROUTE_RUNG"')
        i_line = src.index("skipped reason=disabled")
        assert i_gate < i_line


class TestBreakRungSlotOrdering:
    """The rung slot must sit BEFORE the wall-handback exit decision in
    the single exit tail, and the neighbouring slots stay untouched."""

    def test_rung_before_handback_exit_decision(self):
        src = inspect.getsource(d.DCPOptimizer._exit_with_ils_polish)
        i_recipe = src.index("_maybe_run_recipe_pass_postloop")
        i_rung = src.index("_maybe_run_corescore_route_rung_postloop")
        i_handback = src.index("survived the polish stages")
        i_finalize = src.index("_finalize_output_dcp(output_dcp)")
        assert i_recipe < i_rung < i_handback < i_finalize

    def test_recipe_postloop_slot_keeps_its_handback_defer(self):
        """ONLY the corescore rung claims wall ahead of handback; the
        deep-band recipe postloop slot still defers (untouched)."""
        src = inspect.getsource(
            d.DCPOptimizer._run_recipe_pass_postloop_inner)
        assert "skipped reason=handback_armed" in src


class TestZipSanityF2:
    """Review F2 (executed-path): the trust skip requires a sane zip
    container — a digest-matching but corrupt/truncated store must fall
    back to the full validator."""

    def _mk_zip(self, tmp_path):
        import zipfile
        p = tmp_path / "cand.dcp"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("design.xdef", "x" * 4096)
        return p

    def test_real_zip_is_sane(self, tmp_path):
        assert d.dcp_zip_sane(self._mk_zip(tmp_path)) is True

    def test_truncated_zip_fails(self, tmp_path):
        p = self._mk_zip(tmp_path)
        data = p.read_bytes()
        p.write_bytes(data[: len(data) // 2])
        assert d.dcp_zip_sane(p) is False

    def test_garbage_and_missing_fail(self, tmp_path):
        g = tmp_path / "garbage.dcp"
        g.write_bytes(b"\x00" * 200000)
        assert d.dcp_zip_sane(g) is False
        assert d.dcp_zip_sane(tmp_path / "absent.dcp") is False
        tiny = tmp_path / "tiny.dcp"
        tiny.write_bytes(b"PK\x03\x04")
        assert d.dcp_zip_sane(tiny) is False

    def test_trust_branch_requires_zip_sanity_in_source(self):
        import inspect
        src = open(d.__file__).read()
        i_trust = src.index("_md5_trusted = True")
        i_zip = src.index("dcp_zip_sane(src)")
        assert i_zip < i_trust


class TestV411SpamFallback:
    """v4.1.1 (parity-measured): spam-class out-of-band designs must get the
    exact v4.0.1 vex2-candidate path, not a no-candidate run."""

    @pytest.mark.parametrize("wns,front", [
        (-0.686, None),      # spam — OUT after the 0.90 floor
        (-0.899, None),
        (-0.90, "eto"),      # inclusive floor
        (-0.93, "eto"),      # logicnets
        (-0.946, "eto"),     # vex2
        (-0.999, "eto"),
        (-1.00, "wld"),
        (-1.025, "wld"),     # digit
        (-1.05, "wld"),
        (-1.051, None),
    ])
    def test_floor_and_split(self, wns, front):
        assert d.ownfront_retime_front(wns) == front

    def test_defer_is_band_conditional_in_source(self):
        src = open(d.__file__).read()
        i = src.rindex('reason=ownfront_supersedes "')
        window = src[i-900:i]
        assert "ownfront_retime_front(wns_in) is not None" in window


# ---------------------------------------------------------------------------
# v4.1.2 — FPL26_SPAM_DETERMINIZER_CANDIDATE (spam determinizer, aug06)
# ---------------------------------------------------------------------------


class TestSpamDetFlagDiscipline:
    def test_default_off(self):
        """No env set -> OFF.  The python default must never arm a lever;
        arming is Makefile-only (never-touch-validated-paths rule)."""
        assert d.spam_determinizer_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "on", "yes", " 1 ", "TRUE"])
    def test_armed_by_truthy(self, val, monkeypatch):
        monkeypatch.setenv("FPL26_SPAM_DETERMINIZER_CANDIDATE", val)
        assert d.spam_determinizer_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "off", "", "no", "2"])
    def test_non_truthy_stays_off(self, val, monkeypatch):
        monkeypatch.setenv("FPL26_SPAM_DETERMINIZER_CANDIDATE", val)
        assert d.spam_determinizer_enabled() is False

    def test_kill_switch_wins(self, monkeypatch):
        monkeypatch.setenv("FPL26_SPAM_DETERMINIZER_CANDIDATE", "1")
        monkeypatch.setenv("FPL26_NO_SPAM_DETERMINIZER_CANDIDATE", "1")
        assert d.spam_determinizer_enabled() is False

    def test_kill_switch_alone_is_off(self, monkeypatch):
        monkeypatch.setenv("FPL26_NO_SPAM_DETERMINIZER_CANDIDATE", "1")
        assert d.spam_determinizer_enabled() is False


class TestSpamDetBand:
    """Band [0.60, 0.90) — low bound INCLUSIVE, high bound EXCLUSIVE:
    the exact complement of the ownfront v4.1.1 floor.  spam 0.686 IN;
    logicnets 0.93 OUT; vex2 0.946 OUT."""

    @pytest.mark.parametrize("wns", [
        -0.686,    # spam — THE evidence design (drilled x3 bit-identical)
        -0.60,     # low bound inclusive (== vex2 band low edge)
        -0.899,    # just under the ownfront floor
        -0.75,
    ])
    def test_in_band(self, wns):
        assert d.spam_det_subband_match(wns) is True

    @pytest.mark.parametrize("wns", [
        -0.90,     # EXCLUSIVE high bound — this one is ownfront's (eto)
        -0.93,     # logicnets — OUT (ownfront's band)
        -0.946,    # vex2 — OUT (ownfront's band)
        -1.025,    # digit — OUT (ownfront's band, wld side)
        -0.599,    # just below the low bound
        -0.313,    # fir-class — below band AND carved out upstream
        -1.078,    # optical (mid)
        -8.0,      # deep
        None,      # unmeasured -> treatments fail OFF
        0.0,       # met
        0.5,       # positive slack
        "garbage",
    ])
    def test_out_of_band(self, wns):
        assert d.spam_det_subband_match(wns) is False

    def test_band_constants(self):
        """Complement law: the determinizer band's high bound IS the
        ownfront v4.1.1 floor, and its low bound IS the vex2 band's —
        [0.60, 0.90) + [0.90, 1.05] partition the retime range with no
        gap and no overlap."""
        assert d.SPAM_DET_WNS_MAG_MIN_NS == 0.60
        assert d.SPAM_DET_WNS_MAG_MIN_NS == d.VEX2_RETIME_WNS_MAG_MIN_NS
        assert d.SPAM_DET_WNS_MAG_MAX_NS == 0.90
        assert (d.SPAM_DET_WNS_MAG_MAX_NS
                == d.OWNFRONT_RETIME_WNS_MAG_MIN_NS)

    def test_boundary_belongs_to_ownfront(self):
        """At exactly |wns|=0.90 the determinizer must NOT fire and the
        ownfront candidate MUST — no double coverage, no hole."""
        assert d.spam_det_subband_match(-0.90) is False
        assert d.ownfront_retime_front(-0.90) == "eto"

    def test_every_boundary_value_covered_by_exactly_one(self):
        for wns in (-0.60, -0.686, -0.899, -0.90, -0.93, -0.946, -1.05):
            n = int(d.spam_det_subband_match(wns)) + int(
                d.ownfront_retime_front(wns) is not None)
            assert n == 1, wns


class TestSpamDetFrozenChain:
    """The four drilled steps VERBATIM (box2, x3 bit-identical
    -0.543/466.64), behind the same two-step unroute/unplace
    normalization prefix every placed-front chain carries."""

    def test_drilled_steps_verbatim(self):
        assert d.SPAM_DET_TCL[-4:] == (
            "place_design -directive AltSpreadLogic_medium",
            "route_design -directive Explore",
            "route_design -directive AggressiveExplore",
            "phys_opt_design -directive AlternateFlowWithRetiming",
        )

    def test_normalization_prefix_matches_vex2_chain(self):
        assert d.SPAM_DET_TCL[:2] == d.VEX2_RETIME_TCL[:2] == (
            "route_design -unroute", "place_design -unplace")

    def test_no_unroute_between_the_route_steps(self):
        """THE load-bearing invariant: the AE route is an INCREMENTAL
        escalation of the Explore route (ils_polish __INCR_ROUTE__
        semantics) — an unroute between them would be a different,
        undrilled chain."""
        i_explore = d.SPAM_DET_TCL.index("route_design -directive Explore")
        i_ae = d.SPAM_DET_TCL.index(
            "route_design -directive AggressiveExplore")
        assert i_ae == i_explore + 1
        assert "route_design -unroute" not in d.SPAM_DET_TCL[i_explore:]

    def test_s6_ordering_explore_before_aggressive(self):
        """S6 ordering law: Explore first, then AggressiveExplore."""
        assert (d.SPAM_DET_TCL.index("route_design -directive Explore")
                < d.SPAM_DET_TCL.index(
                    "route_design -directive AggressiveExplore"))

    def test_afwr_is_the_terminal_step(self):
        assert d.SPAM_DET_TCL[-1] == (
            "phys_opt_design -directive AlternateFlowWithRetiming")

    def test_no_bare_retime_step(self):
        """The drilled chain has no bare `phys_opt_design -retime` —
        the retiming exposure is the AFWR flow (audited; see
        TestSpamDetFfAudit)."""
        assert "phys_opt_design -retime" not in d.SPAM_DET_TCL


class TestSpamDetGateArithmetic:
    """Recomputed third-candidate need — the numbers the prereg cites —
    including the HONEST 'which defers' arithmetic."""

    def test_primary_shallow_gate_untouched(self):
        assert (d.RECIPE_PASS_TIMEOUT_FACTOR * d.RECIPE_PASS_SHALLOW_EXPECTED_S
                + d.RECIPE_PASS_OVERHEAD_RESERVE_S + 1100.0) == 2450.0

    def test_vex2_gate_untouched(self):
        """Spam's validated 29.19 path includes vex2's exact spend — the
        2330 gate must survive this lever verbatim."""
        assert (d.RECIPE_PASS_TIMEOUT_FACTOR * d.VEX2_RETIME_EXPECTED_S
                + d.VEX2_RETIME_OVERHEAD_RESERVE_S + 1100.0) == 2330.0

    def test_ownfront_gate_untouched(self):
        assert (d.RECIPE_PASS_TIMEOUT_FACTOR * d.OWNFRONT_RETIME_EXPECTED_S
                + d.OWNFRONT_RETIME_OVERHEAD_RESERVE_S + 1100.0) == 2420.0

    def test_spam_det_need(self):
        """need3 = 1.5x390 (cap 585) + 540 overheads + 1100 LLM floor
        = 2225 s."""
        cap = d.RECIPE_PASS_TIMEOUT_FACTOR * d.SPAM_DET_EXPECTED_S
        assert cap == 585.0
        need = cap + d.SPAM_DET_OVERHEAD_RESERVE_S + 1100.0
        assert need == 2225.0
        assert d.SPAM_DET_OVERHEAD_RESERVE_S == 540.0
        assert d.SPAM_DET_OVERHEAD_RESERVE_S == (
            d.RECIPE_PASS_MEASURE_RESERVE_S + d.RECIPE_PASS_STORE_RESERVE_S
            + d.RECIPE_PASS_REGISTER_RESERVE_S)

    def test_expected_is_the_measured_chain_cost(self):
        """~390 s x3 bit-identical on box2 — deterministic chain, so
        expected pins to the measured cost."""
        assert d.SPAM_DET_EXPECTED_S == 390.0

    def test_swap_makes_the_fire_affordable(self):
        """Post-swap arithmetic pin (review F2 rewrite — the old version
        asserted the STACK scenario with a vacuous inequality).  With
        the vex2 candidate DEFERRED in [0.60, 0.90), only the primary
        spends before this gate: even the slow primary leaves >= need,
        so the determinizer FIRES on typical ship-wall draws.  The
        pre-swap stacked shape could never fire — pinned as the
        impossibility that justified a5dcddb."""
        max_remaining = 3200.0
        need = (d.RECIPE_PASS_TIMEOUT_FACTOR * d.SPAM_DET_EXPECTED_S
                + d.SPAM_DET_OVERHEAD_RESERVE_S + 1100.0)
        assert need == 2225.0
        # POST-SWAP: primary 450-600 only -> fires
        assert max_remaining - 600.0 >= need
        assert max_remaining - 450.0 >= need
        # PRE-SWAP stack (history): even the FASTEST stacked draw could
        # not fund the gate — the shape was null-by-construction
        vex2_cap = d.RECIPE_PASS_TIMEOUT_FACTOR * d.VEX2_RETIME_EXPECTED_S
        assert max_remaining - 450.0 - 400.0 - 200.0 < need
        assert max_remaining - 600.0 - vex2_cap - 200.0 < need


class TestSpamDetFfAudit:
    """The audit is ADDED despite the no-bare-retime-step spec (the AFWR
    directive IS a retiming-enabled flow) — SHARED helpers, fail closed,
    probes bracketing the AFWR step."""

    def _src(self):
        return inspect.getsource(
            d.DCPOptimizer._maybe_run_spam_determinizer_candidate)

    def test_shared_helpers_no_fork(self):
        src = self._src()
        assert "_vex2_retime_ff_count" in src
        assert "vex2_retime_ff_drift_ok" in src
        assert 'tag="SPAM-DET"' in src
        assert not hasattr(d, "spam_det_parse_ffcount")
        assert not hasattr(d.DCPOptimizer, "_spam_det_ff_count")

    def test_probes_bracket_the_afwr_step(self):
        """ff_before after the route-AE step (= immediately before
        AFWR), ff_after after AFWR — both step strings are unique in
        the chain."""
        src = self._src()
        assert 'if step == "route_design -directive AggressiveExplore":' in src
        assert ('elif step == "phys_opt_design -directive '
                'AlternateFlowWithRetiming":') in src
        assert d.SPAM_DET_TCL.count(
            "route_design -directive AggressiveExplore") == 1
        assert d.SPAM_DET_TCL.count(
            "phys_opt_design -directive AlternateFlowWithRetiming") == 1

    def test_audit_fails_closed(self):
        src = self._src()
        assert "vex2_retime_ff_drift_ok(ff_before, ff_after)" in src
        assert "ABORTED reason=latency_audit" in src


class TestSpamDetSiteDiscipline:
    """Source-level invariants: MUX-additive, verify=True, fail-closed
    logging, ordered LAST in the shallow bracket, vex2 deferred in-band post-swap."""

    def _src(self):
        return inspect.getsource(
            d.DCPOptimizer._maybe_run_spam_determinizer_candidate)

    def test_registers_with_verify_true(self):
        src = self._src()
        assert "register_final_candidate" in src
        assert "verify=True" in src
        assert '"recipe_pass_shallow_spamdet"' in src

    def test_never_touches_best_valid(self):
        src = self._src()
        assert "self.best_valid" not in src
        assert ".best_valid =" not in src

    def test_disabled_audit_line_gated_behind_env_present(self):
        src = self._src()
        i_gate = src.index(
            'v40_flag_env_present("FPL26_SPAM_DETERMINIZER_CANDIDATE"')
        i_line = src.index("SPAM-DET: skipped reason=disabled")
        assert i_gate < i_line

    def test_stable_log_keys(self):
        src = self._src()
        for key in ("SPAM-DET: skipped reason=disabled ",
                    "SPAM-DET: skipped reason=out_of_band ",
                    "SPAM-DET: skipped reason=wall ",
                    "SPAM-DET: fired ",
                    "SPAM-DET: ff-audit ",
                    "SPAM-DET: result "):
            assert key in src, key

    def test_gate_order_flag_band_wall_fire(self):
        src = self._src()
        i_flag = src.index("if not spam_determinizer_enabled():")
        i_band = src.index("spam_det_subband_match")
        i_wall = src.index("_budget_remaining")
        i_fired = src.index("SPAM-DET: fired wns_in=")
        assert i_flag < i_band < i_wall < i_fired

    def test_ordered_after_vex2_and_ownfront(self):
        """Call order in the shallow bracket: vex2 (validated spend
        first), ownfront (skips out_of_band on spam-class), determinizer
        LAST — so flag-off keeps the c37f2c0 call shape exactly."""
        src = inspect.getsource(d.DCPOptimizer._run_recipe_pass_inner)
        i_vex2 = src.index("_maybe_run_vex2_retime_candidate")
        i_own = src.index("_maybe_run_ownfront_retime_candidate")
        i_det = src.index("_maybe_run_spam_determinizer_candidate")
        assert i_vex2 < i_own < i_det

    def test_vex2_defers_to_determinizer_in_its_band(self):
        """v4.1.2 SWAP (aug06): in [0.60, 0.90) the determinizer REPLACES
        the vex2 candidate (measured MUX-discarded x3 there; tonight's r1
        refuted the c37f2c0 spend-timing theory).  The defer must be BOTH
        flag- and band-conditional, and must precede the band check."""
        src = inspect.getsource(
            d.DCPOptimizer._maybe_run_vex2_retime_candidate)
        i_defer = src.index("reason=spam_det_supersedes")
        i_cond = src.index("spam_determinizer_enabled() and spam_det_subband_match(wns_in)")
        i_band = src.index("vex2_retime_subband_match")
        assert i_cond < i_defer < i_band

    def test_swap_band_alignment(self):
        """The swap window == the determinizer band == the ownfront floor
        complement; vex2 (0.946) keeps its own candidate."""
        assert d.spam_det_subband_match(-0.686) is True
        assert d.spam_det_subband_match(-0.899) is True
        assert d.spam_det_subband_match(-0.90) is False
        assert d.spam_det_subband_match(-0.946) is False

    def test_pristine_reopen_before_the_chain(self):
        src = self._src()
        i_reopen = src.index("_presweep_reopen")
        i_loop = src.index("for i, step in enumerate(SPAM_DET_TCL, 1):")
        assert i_reopen < i_loop

    def test_wall_skip_is_fail_closed(self):
        src = self._src()
        assert "fail closed" in src
        i_wall = src.index("SPAM-DET: skipped reason=wall ")
        i_fired = src.index("SPAM-DET: fired wns_in=")
        assert i_wall < i_fired


class TestSpamDetManifestAndMakefile:
    """Provenance + arming surface (the jul30 'Makefile not in the ship
    surface' lesson: one branch armed alone measures nothing)."""

    @pytest.mark.parametrize("flag", [
        "FPL26_SPAM_DETERMINIZER_CANDIDATE",
        "FPL26_NO_SPAM_DETERMINIZER_CANDIDATE",
    ])
    def test_flag_in_manifest(self, flag):
        assert flag in d.DCPOptimizer._MANIFEST_FLAGS

    @pytest.fixture(scope="class")
    def branches(self):
        text = (ROOT / "Makefile").read_text()
        wrapper = [ln for ln in text.splitlines()
                   if "multi_restart_optimize.py" in ln
                   and "FPL26_FIR_SUBBAND_FLOOR" in ln]
        fallback = [ln for ln in text.splitlines()
                    if "dcp_optimizer.py" in ln and ln.strip().startswith("||")
                    and "FPL26_FIR_SUBBAND_FLOOR" in ln]
        assert len(wrapper) == 1 and len(fallback) == 1
        return wrapper[0], fallback[0]

    def test_both_branches_armed(self, branches):
        expected = ("FPL26_SPAM_DETERMINIZER_CANDIDATE="
                    "$(if $(SPAM_DET),$(SPAM_DET),1)")
        for branch in branches:
            assert expected in branch

    def test_v41_rev2_arming_untouched(self, branches):
        for var, knob in (
                ("FPL26_OWNFRONT_RETIME_CANDIDATE", "OWNFRONT_RETIME"),
                ("FPL26_MUX_MD5_TRUST", "MUX_MD5_TRUST"),
                ("FPL26_VEX2_RETIME_CANDIDATE", "VEX2_RETIME")):
            expected = f"{var}=$(if $({knob}),$({knob}),1)"
            for branch in branches:
                assert expected in branch
