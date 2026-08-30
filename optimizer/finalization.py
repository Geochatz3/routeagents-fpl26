"""Revalidate enrolled candidates and publish the highest-scoring valid result.

Optimization stages register candidates without modifying the banked best
result. Finalization measures the candidates again, selects the valid argmax,
and publishes its artifacts.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

# Log through the orchestrator's logger: this code still belongs to the same
# run, and a module-named logger would change every line it emits.
logger = logging.getLogger("dcp_optimizer")

from optimizer.recipe_policy import (  # noqa: F401
    v40_flag_env_present,
)

FINAL_CANDIDATE_MUX_MIN_GAIN_NS = 0.005


from typing import Optional

from optimizer.finalize_mux import (
    _artifact_identity,
    _atomic_copy,
    _path_is_in_submission,
    _stream_md5,
    _verify_shipped_identity,
    _write_shipped_manifest,
    dcp_zip_sane,
    mux_md5_digest,
    mux_md5_trust_enabled,
)

from optimizer.tool_source import (  # noqa: F401
    LINEAGE_SOURCES,
    _PLACE_STMT_RE,
    _ROUTE_STMT_RE,
    _looks_like_tool_error,
    _make_lineage_entry,
    _routed_state_transition,
    _tcl_statements,
    _tool_cmd_text,
    compose_iter1_user_message,
    convert_mcp_tool_to_openai,
    is_routed_state_destroying,
    load_system_prompt,
)


class FinalizationMixin:
    """Methods mixed into DCPOptimizer; they run against its instance state."""

    async def _mirror_best_valid_now(self, eager: bool = False) -> None:
        """Write a stable `best_valid.dcp` and `best_valid.edf` pair to the run
        directory.

        Eager mode runs when an improvement is detected and captures the
        verified in-memory best state before it can change; it bypasses the
        budget-floor check because preserving an improvement takes priority.
        Backstop mode runs at the top of an iteration and observes the budget
        floor. The operation is a no-op when no improvement is pending or when
        checkpoint or EDIF generation fails.
        """
        if not self._pending_best_mirror:
            return
        # In backstop (non-eager) mode, refuse the mirror if remaining
        # budget is below reserve + mirror cost. In eager mode, always try
        # to capture an improvement — it is always worth the time.
        mirror_budget_floor = self._finalize_reserve_seconds + 120.0
        budget_remaining = (
            self._budget_remaining() if self._budget_deadline is not None else None
        )
        prior_mirror_wns = getattr(self, "_best_valid_dcp_wns", None)
        current_best_wns = float(self.best_wns) if self.best_wns is not None else None
        improvement_delta = (
            (current_best_wns - prior_mirror_wns)
            if (current_best_wns is not None and prior_mirror_wns is not None)
            else None
        )
        iteration_for_log = getattr(self, "iteration", None)
        budget_bypassed = False
        # Entry log — proves the eager path fired, and snapshots the state
        # so post-mortem debugging can show exactly when on-disk capture
        # happened relative to the WNS-improvement measurement.
        logger.info(
            "[mirror] enter: eager=%s iter=%s best_wns=%s prior_mirror_wns=%s "
            "delta=%s budget_remaining=%s floor=%.0f",
            eager,
            iteration_for_log,
            f"{current_best_wns:.3f}" if current_best_wns is not None else "None",
            f"{prior_mirror_wns:.3f}" if prior_mirror_wns is not None else "None",
            f"{improvement_delta:+.3f}" if improvement_delta is not None else "None",
            f"{budget_remaining:.0f}" if budget_remaining is not None else "None",
            mirror_budget_floor,
        )
        if (not eager
                and self._budget_deadline is not None
                and self._budget_remaining() < mirror_budget_floor):
            logger.info(
                "[mirror] skip: backstop budget-tight — remaining "
                f"{self._budget_remaining():.0f}s < {mirror_budget_floor:.0f}s "
                "floor; keeping pending_best_mirror=True. Eager mirror should "
                "have captured this already; if not, finalize will fall back "
                "to baseline (safe)."
            )
            return
        if eager and budget_remaining is not None and budget_remaining < mirror_budget_floor:
            budget_bypassed = True
            logger.info(
                "[mirror] eager-bypass: budget_remaining=%.0fs < floor=%.0fs, "
                "writing anyway to capture measured improvement.",
                budget_remaining, mirror_budget_floor,
            )
        dcp_target = self.run_dir / "best_valid.dcp"
        edf_target = self.run_dir / "best_valid.edf"
        # Audit both write targets before dispatch.
        self._path_guard_check(
            dcp_target.resolve(),
            context=f"mirror_best_valid_now_dcp_{'eager' if eager else 'backstop'}",
        )
        self._path_guard_check(
            edf_target.resolve(),
            context=f"mirror_best_valid_now_edif_{'eager' if eager else 'backstop'}",
        )
        # Capture file metadata before writing so freshness requires an observed
        # change; a budget-skipped dispatch can leave an existing stale file untouched.
        pre_dcp_mtime = dcp_target.stat().st_mtime if dcp_target.exists() else 0.0
        pre_dcp_size = dcp_target.stat().st_size if dcp_target.exists() else 0
        pre_edf_mtime = edf_target.stat().st_mtime if edf_target.exists() else 0.0
        pre_edf_size = edf_target.stat().st_size if edf_target.exists() else 0

        snapshot_wns = float(self.best_wns) if self.best_wns is not None else None

        # ---- DCP write attempt ----
        dcp_fresh = False
        try:
            dcp_result = await self.call_tool("vivado_write_checkpoint", {
                "dcp_path": str(dcp_target.resolve()),
                "force": True,
            })
            if _looks_like_tool_error(dcp_result):
                logger.warning(
                    f"best_valid DCP mirror skipped by dispatcher: "
                    f"{str(dcp_result)[:200]}"
                )
            elif dcp_target.exists() and dcp_target.stat().st_size > 0:
                # mtime advancement is the canonical "the write landed" signal.
                # On WSL2/NTFS some writes back-date to the original mtime if
                # nothing changed; tolerate that by also accepting a size change.
                post_dcp_mtime = dcp_target.stat().st_mtime
                post_dcp_size = dcp_target.stat().st_size
                if (post_dcp_mtime > pre_dcp_mtime
                        or post_dcp_size != pre_dcp_size):
                    dcp_fresh = True
                else:
                    logger.warning(
                        f"best_valid DCP mirror: file unchanged after "
                        f"write_checkpoint (mtime/size identical to pre-call); "
                        f"treating as stale to be safe."
                    )
            else:
                logger.warning(
                    f"best_valid DCP mirror returned but file missing/empty: {dcp_target}"
                )
        except Exception as e:
            logger.warning(f"best_valid DCP mirror raised: {e}")

        if dcp_fresh:
            self._best_valid_dcp = dcp_target
            self._best_valid_dcp_wns = snapshot_wns
            # Store the checkpoint size at bank time as an integrity reference.
            # Emergency finalization rejects a mirror whose size no longer matches.
            try:
                self._best_valid_mirror_size = dcp_target.stat().st_size
            except OSError:
                pass
            self._bump_lineage(
                "eager_mirror" if eager else "backstop",
                wns=snapshot_wns,
                tool_name="vivado_write_checkpoint",
                dcp_path=dcp_target,
                extra={"budget_bypassed": budget_bypassed},
            )
            logger.info(
                f"Mirrored best_valid DCP (wns={snapshot_wns:.3f} ns, "
                f"token={self._best_valid_token}) → "
                f"{dcp_target.name} ({dcp_target.stat().st_size//1024} KiB)"
            )
        else:
            # The on-disk best_valid.dcp does NOT reflect the current best_wns
            # — clear the tracked-WNS so finalize knows the fast path is unsafe.
            self._best_valid_dcp_wns = None

        # ---- EDIF write attempt (only worth trying if DCP is fresh) ----
        edif_fresh = False
        if dcp_fresh:
            try:
                edif_result = await self.call_tool("vivado_write_edif", {
                    "edif_path": str(edf_target.resolve()),
                    "force": True,
                })
                if _looks_like_tool_error(edif_result):
                    logger.warning(
                        f"best_valid EDIF mirror skipped by dispatcher: "
                        f"{str(edif_result)[:200]}"
                    )
                elif edf_target.exists() and edf_target.stat().st_size > 0:
                    post_edf_mtime = edf_target.stat().st_mtime
                    post_edf_size = edf_target.stat().st_size
                    if (post_edf_mtime > pre_edf_mtime
                            or post_edf_size != pre_edf_size):
                        edif_fresh = True
                    else:
                        logger.warning(
                            "best_valid EDIF mirror: file unchanged after "
                            "write_edif; treating as stale."
                        )
                else:
                    logger.warning(f"best_valid EDIF write returned but file missing: {edf_target}")
            except Exception as e:
                logger.warning(f"best_valid EDIF write raised: {e}")

        if edif_fresh:
            self._best_valid_edif = edf_target
            self._best_valid_edif_wns = snapshot_wns
            logger.info(f"Mirrored best_valid EDIF → {edf_target.name}")
        else:
            # Mark EDIF provenance stale so finalization regenerates it from the
            # optimized checkpoint instead of shipping an older netlist.
            self._best_valid_edif_wns = None

        # Clear the pending mirror only after the checkpoint refresh succeeds.
        # A skipped or failed write leaves it armed for a later retry.
        if dcp_fresh:
            self._pending_best_mirror = False

        # Exit log — pairs with the entry log so post-mortems can see the
        # full eager-mirror lifecycle: snapshot vs. outcome.
        dcp_exists_post = dcp_target.exists()
        dcp_size_post = dcp_target.stat().st_size if dcp_exists_post else 0
        logger.info(
            "[mirror] exit:  eager=%s iter=%s dcp_fresh=%s edif_fresh=%s "
            "dcp_exists=%s dcp_size=%d wns_snapshot=%s budget_bypassed=%s "
            "pending_after=%s",
            eager,
            iteration_for_log,
            dcp_fresh,
            edif_fresh,
            dcp_exists_post,
            dcp_size_post,
            f"{snapshot_wns:.3f}" if snapshot_wns is not None else "None",
            budget_bypassed,
            self._pending_best_mirror,
        )

    async def _register_tail_final_candidate(self, source_label: str) -> bool:
        """Enroll a tail lineage's harvested best_valid into the finalize MUX.

        BELT-AND-SUSPENDERS TODAY: the tail advances best_valid via
        _mirror_best_valid_now(eager=True), so the DCP registered here is
        USUALLY byte-identical to the implicit pipeline candidate #0
        (source="pipeline"). A candidate equal to pipeline can NEVER win the
        MUX (FINAL_CANDIDATE_MUX_MIN_GAIN_NS + tie->pipeline), so the
        default (controller-ON) SHIPPED artifact stays byte-identical. This
        is the hook a RESERVE fork-compare can use later to
        enroll a DIVERGENT tail DCP (which the MUX would then legitimately
        pick if it beats the pipeline).

        verify=True re-runs the full never-worse gate stack (tracker-routed +
        hold + cell-band) against the CURRENT in-memory state. The tail loop
        may have left a rejected draw loaded (pending_reopen), so re-open the
        banked best FIRST — exactly the pre-sweep's verify-mode discipline —
        so in-memory == the registered candidate. Never raises into the
        caller (a registration hiccup must never break a run)."""
        try:
            bv = self._best_valid_dcp
            if (bv is None or not Path(bv).exists()
                    or self.best_wns is None
                    or self.best_wns == float("-inf")):
                return False
            _ro = await self.call_tool(
                "vivado_open_checkpoint",
                {"dcp_path": str(Path(bv).resolve())})
            if _looks_like_tool_error(_ro):
                logger.info(
                    f"[mux] {source_label}: banked-best re-open failed; "
                    "SKIPPING registration (never-worse).")
                return False
            store_dir = (self.run_dir or Path(".")) / "final_candidates"
            store_dir.mkdir(parents=True, exist_ok=True)
            store = store_dir / f"{source_label}.dcp"
            _atomic_copy(str(bv), str(store))
            src_edif = Path(bv).with_suffix(".edf")
            if src_edif.exists() and src_edif.stat().st_size > 0:
                try:
                    _atomic_copy(str(src_edif),
                                 str(store.with_suffix(".edf")))
                except Exception:
                    pass
            return await self.register_final_candidate(
                store, self.best_wns, source_label, verify=True)
        except Exception as e:
            logger.warning(
                f"[mux] {source_label} registration raised "
                f"{type(e).__name__} (non-fatal): {e}")
            return False

    async def _finalize_logic_floor(self, output_dcp: Path,
                                    max_iterations_reached: bool) -> None:
        """Short exit: finalize + summary for a logic-floor-attested run.
        Lives OUTSIDE _exit_with_ils_polish so the stage-ordering contract of
        that function's source (recipe -> rung -> handback -> finalize, pinned
        by test_v41_levers) stays literally true."""
        await self._finalize_output_dcp(output_dcp)
        self.end_time = time.time()
        self._print_optimization_summary(
            max_iterations_reached=max_iterations_reached)

    async def _finalize_refresh_edif(self, out_edif: Path) -> bool:
        """Write a fresh EDIF from Vivado's in-memory state.

        Used when the fast-path DCP is correct but the EDIF mirror is
        stale (e.g. retiming added pipeline registers but the previous
        EDIF mirror wrote pre-retiming state).  Best-effort.

        Known limitation: this function still writes EDIF
        from Vivado's in-memory state without reopening the shipped
        DCP.  Risks:
          (a) in-memory state may have diverged from shipped DCP after
              a directive applied to the live session
          (b) write_edif silently writes encrypted EDIF if RW JNI is
              not active — unrecoverable in finalize budget
          (c) no consistency check between out_edif and the paired DCP
        These risks are traced (decision records below) rather than
        patched; a reopen+verify behind a flag would be the fix if the
        trace ever shows real divergence.

        One narrow exception, added on evidence rather than
        speculation: when the post-loop recipe pass has mutated the
        session (_finalize_session_untrusted), the in-memory netlist is
        a different flow's state than the shipped best_valid DCP —
        recipe-fired-but-MUX-lost makes the (a) divergence certain, not
        merely possible.  Refuse the regen; the caller's
        edif_ok=False path ships VALID_OPTIMIZED_NO_EDIF (existing
        accepted status).  All other callers/branches keep the original
        trace-only behaviour.
        """
        out_edif_path = Path(out_edif)
        if getattr(self, "_finalize_session_untrusted", False):
            logger.warning(
                "_finalize_refresh_edif: REFUSED — session marked "
                "untrusted (post-loop recipe pass mutated it; in-memory "
                "netlist is a different flow's state than the shipped "
                "DCP).  Shipping without EDIF (VALID_OPTIMIZED_NO_EDIF) "
                "instead of a mismatched-netlist EDIF.")
            try:
                self._emit_decision({
                    "decision_source": "finalize",
                    "phase": "finalize",
                    "action_label": "finalize_refresh_edif_refused",
                    "tool_name": "vivado_write_edif",
                    "tool_error_code": "UNTRUSTED_SESSION",
                    "notes": ("post-loop recipe pass mutated the session; "
                              "live-state EDIF regen refused (MAJOR-2)"),
                })
            except Exception as exc:  # pragma: no cover — defensive
                logger.debug(
                    f"finalize_refresh_edif refusal trace emit failed: {exc}")
            return False
        # Audit the EDIF write target before dispatch.
        self._path_guard_check(
            out_edif_path, context="finalize_refresh_edif",
        )
        # Trace finalize_refresh_edif entry with the full context
        # post-mortems need to detect in-memory/shipped divergence.
        try:
            out_dcp_path = out_edif_path.with_suffix(".dcp")
            self._emit_decision({
                "decision_source": "finalize",
                "phase": "finalize",
                "action_label": "finalize_refresh_edif_start",
                "tool_name": "vivado_write_edif",
                "output_dcp_path": str(out_dcp_path),
                "ship_lineage_source": (
                    self._ship_lineage.get("source")
                    if isinstance(getattr(self, "_ship_lineage", None), dict)
                    else None
                ),
                "best_valid_token": self._best_valid_token,
                "notes": (
                    f"out_edif={out_edif_path} "
                    f"output_dcp_exists={out_dcp_path.exists()} "
                    f"reopened=False"
                ),
            })
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug(f"finalize_refresh_edif trace emit failed: {exc}")
        try:
            edif_result = await self.call_tool("vivado_write_edif", {
                "edif_path": str(out_edif_path.resolve()),
                "force": True,
            })
            if _looks_like_tool_error(edif_result):
                logger.warning(
                    f"_finalize_refresh_edif: write_edif refused by "
                    f"dispatcher: {str(edif_result)[:200]}"
                )
                # Classify + trace the refusal.
                tool_err = self._classify_tool_payload(
                    edif_result, context="finalize_refresh_edif",
                )
                self._emit_decision({
                    "decision_source": "finalize",
                    "phase": "finalize",
                    "action_label": "finalize_refresh_edif_refused",
                    "tool_name": "vivado_write_edif",
                    "tool_error_code": (
                        tool_err.code if tool_err is not None else "UNKNOWN_TOOL_ERROR"
                    ),
                    "notes": f"edif_result_excerpt={str(edif_result)[:160]}",
                })
                return False
            ok = out_edif_path.exists() and out_edif_path.stat().st_size > 0
            self._emit_decision({
                "decision_source": "finalize",
                "phase": "finalize",
                "action_label": (
                    "finalize_refresh_edif_ok" if ok
                    else "finalize_refresh_edif_missing"
                ),
                "tool_name": "vivado_write_edif",
                "notes": (
                    f"edif_exists={out_edif_path.exists()} "
                    f"size={out_edif_path.stat().st_size if out_edif_path.exists() else 0}"
                ),
            })
            return ok
        except Exception as e:
            logger.warning(f"_finalize_refresh_edif raised: {e}")
            self._emit_decision({
                "decision_source": "finalize",
                "phase": "finalize",
                "action_label": "finalize_refresh_edif_exception",
                "tool_name": "vivado_write_edif",
                "tool_error_code": "UNKNOWN_TOOL_ERROR",
                "notes": f"exception={str(e)[:160]}",
            })
            return False

    async def register_final_candidate(
        self,
        path,
        wns: Optional[float],
        source_label: str,
        *,
        whs: Optional[float] = None,
        cell_count: Optional[int] = None,
        verify: bool = True,
    ) -> bool:
        """Register a persisted final checkpoint only after insured-comparison
        validation.

        Validation requires the routed tracker to report success; missing
        routed state fails closed. Hold slack is measured through
        `self.call_tool` using the full tool name. Negative slack rejects the
        candidate, while unavailable hold data fails open with a warning. Cell
        count must be between 0.5 and 3 times the initial count when
        measurable; this sanity band is pinned to detecting substantial logic
        deletion, and unavailable counts fail open. The checkpoint must remain
        stable until finalization; callers must copy candidates stored in
        mutable mirror paths. The source label records provenance, while
        `pipeline` remains reserved for the implicit live candidate. With
        verification enabled, the caller guarantees that the in-memory tool
        state matches the checkpoint. With verification disabled, previously
        measured hold slack and cell count pass through the same gates. Returns
        whether enrollment succeeded and never propagates registration failures
        to the caller.
        """
        try:
            if source_label == "pipeline":
                logger.warning(
                    "[mux] register refused: 'pipeline' is reserved for the "
                    "implicit best_valid candidate #0.")
                return False
            if wns is None or wns == float("-inf"):
                logger.warning(
                    f"[mux] register refused ({source_label}): wns unmeasurable.")
                return False
            src = Path(path)
            if not (src.exists() and src.stat().st_size > 0):
                logger.warning(
                    f"[mux] register refused ({source_label}): candidate DCP "
                    f"missing/empty at {src}.")
                return False
            # ---- Gate 1: tracker-first routed ----
            if verify:
                routed = await self._routed_ok_for_best()
            else:
                # Caller pre-verified routed via the identical gate.
                routed = True
            if not routed:
                logger.warning(
                    f"[mux] register REJECTED ({source_label}): NOT routed "
                    "(tracker-first phantom-best guard).")
                return False
            # ---- Gate 2: hold ----
            if verify:
                try:
                    from optimizer.ils_polish import _measure_hold
                    whs = await asyncio.wait_for(
                        _measure_hold(self.call_tool, timeout_s=120.0),
                        timeout=180.0)
                except Exception:
                    whs = None
            if whs is not None and whs < 0.0:
                logger.warning(
                    f"[mux] register REJECTED ({source_label}): HOLD dirty "
                    f"(whs={whs:.3f} < 0 — validator gates hold_passed).")
                return False
            if whs is None:
                logger.warning(
                    f"[mux] register ({source_label}): HOLD UNMEASURABLE — "
                    "enrolling FAIL-OPEN (whs unknown; validator gates "
                    "hold_passed).")
            # ---- Gate 3: cell-count band ----
            if verify and self._input_cell_count:
                try:
                    _cc_r = await asyncio.wait_for(
                        self.call_tool(
                            "vivado_run_tcl",
                            {"command": "llength [get_cells -quiet "
                             "-hierarchical -filter {IS_PRIMITIVE}]",
                             "timeout": 120.0}),
                        timeout=180.0)
                    # Detect tool error envelopes before parsing integers
                    # because timeout and budget fields can resemble cell
                    # counts. This tail check fails open when the count is
                    # unavailable.
                    if _looks_like_tool_error(_cc_r):
                        raise RuntimeError(
                            f"cell-count query returned error envelope: "
                            f"{str(_cc_r)[:120]}")
                    _cc_m = re.search(r"(\d+)", str(_cc_r) or "")
                    cell_count = int(_cc_m.group(1)) if _cc_m else None
                except Exception:
                    cell_count = None
            if (cell_count is not None and self._input_cell_count
                    and not (0.5 * self._input_cell_count
                             <= cell_count
                             <= 3.0 * self._input_cell_count)):
                logger.warning(
                    f"[mux] register REJECTED ({source_label}): cell count "
                    f"{cell_count:,} outside [0.5x, 3x] of entry "
                    f"{self._input_cell_count:,} (logic-deletion guard).")
                return False
            entry = {
                "path": str(src),
                "wns": float(wns),
                "whs": (float(whs) if whs is not None else None),
                "source_label": str(source_label),
                "cell_count": (int(cell_count)
                               if cell_count is not None else None),
                "verified_routed": True,
            }
            # Record the identity of fully verified candidates so finalization can
            # avoid reopening unchanged artifacts. Digest failures leave the
            # candidate on the full-validation path.
            if verify and mux_md5_trust_enabled():
                _dig = mux_md5_digest(src)
                if _dig is not None:
                    entry["md5"], entry["size"] = _dig
                    logger.info(
                        f"[mux] candidate digest recorded ({source_label}): "
                        f"md5={_dig[0]} size={_dig[1]} (finalize "
                        f"verify=trusted eligible — registration already "
                        f"re-measured routed/hold/cells)")
                else:
                    logger.warning(
                        f"[mux] candidate digest unavailable "
                        f"({source_label}) — finalize keeps the full "
                        f"structural-validate path for this candidate")
            self._final_candidates.append(entry)
            logger.info(
                f"[mux] candidate ENROLLED: source={source_label} "
                f"wns={float(wns):.3f} whs="
                f"{('%.3f' % whs) if whs is not None else 'None'} "
                f"cells={cell_count if cell_count is not None else 'None'} "
                f"path={src.name} (now {len(self._final_candidates)} "
                "non-pipeline candidate(s)).")
            return True
        except Exception as e:  # pragma: no cover — must never break a run
            logger.warning(
                f"[mux] register_final_candidate raised "
                f"{type(e).__name__} ({source_label}): {e}")
            return False

    async def _maybe_ship_final_candidate_mux(self, output_dcp: Path) -> bool:
        """INSURED-COMPARE final-candidate MUX.

        Ship argmax-by-scored-clock-WNS across the verified candidates.
        The pipeline's best_valid is the implicit candidate #0
        (source="pipeline"); the registry holds additional verified
        candidates (pre-sweep lottery winners, etc.).

        Returns True iff a NON-pipeline candidate won and was shipped to
        output_dcp (the caller then returns early).  Returns False in
        every other case — and when it returns False it leaves output_dcp
        untouched, so the normal finalize path runs byte-identically:

          * no registered candidates            (the default) -> no-op;
          * initial_wns is None                 (ultra-conservative
            baseline path) -> defer to normal finalize;
          * pipeline WNS >= best candidate WNS + margin (tie -> pipeline);
          * the winning candidate FAILS structural validate (never-worse:
            keep the pipeline).

        A pure additive never-worse harness: it changes nothing when only
        the pipeline candidate exists.
        """
        cands = [c for c in self._final_candidates if c.get("verified_routed")]
        if not cands:
            return False  # zero diff — pipeline is the only candidate.
        # An unmeasured baseline makes any WNS comparison unverifiable
        # (same reasoning as _finalize_no_improvement) — never override.
        if self.initial_wns is None:
            return False
        # The pipeline's shipped WNS: best_wns when it improved, else the
        # baseline it will ship on the no-improvement path.
        no_improvement = self._finalize_no_improvement()
        pipeline_ship_wns = (
            float(self.initial_wns) if no_improvement
            else float(self.best_wns))
        best = max(cands, key=lambda c: c["wns"])
        n_total = len(cands) + 1  # + implicit pipeline candidate #0
        if not (best["wns"] > pipeline_ship_wns + FINAL_CANDIDATE_MUX_MIN_GAIN_NS):
            logger.info(
                f"[mux] FINAL = pipeline wns={pipeline_ship_wns:.3f} "
                f"(best non-pipeline candidate {best['source_label']} "
                f"wns={best['wns']:.3f} did not beat it by "
                f"{FINAL_CANDIDATE_MUX_MIN_GAIN_NS} ns; tie -> pipeline); "
                f"argmax of {n_total} candidates unchanged — byte-identical.")
            return False
        src = Path(best["path"])
        if not (src.exists() and src.stat().st_size > 0):
            logger.warning(
                f"[mux] winning candidate {best['source_label']} DCP "
                f"missing/empty at {src}; KEEPING pipeline (never-worse).")
            return False
        # An unchanged, previously verified candidate can skip the structural
        # reopen, avoiding dependence on the tool's validation budget.
        # Missing or mismatched identity data fails closed to full validation.
        _md5_trusted = False
        if mux_md5_trust_enabled():
            _stored_md5 = best.get("md5")
            _stored_size = best.get("size")
            if _stored_md5 and _stored_size:
                _dig_now = mux_md5_digest(src)
                if (_dig_now is not None
                        and _dig_now == (_stored_md5, _stored_size)
                        and not dcp_zip_sane(src)):
                    # A matching digest proves identity, not container validity.
                    # An invalid ZIP therefore fails closed to the full validator.
                    logger.warning(
                        f"[mux] winning candidate {best['source_label']} "
                        f"digest matches but container FAILS the zip sanity "
                        f"check — NOT trusting; full structural-validate "
                        f"path (fail closed)")
                elif _dig_now is not None and _dig_now == (_stored_md5,
                                                           _stored_size):
                    _md5_trusted = True
                    logger.info(
                        f"[mux] winning candidate {best['source_label']} "
                        f"verify=trusted (md5 match + zip-sane, registered "
                        f"wns={best['wns']:.3f}; file unchanged since the "
                        f"verify=True registration re-measure — finalize "
                        f"re-open SKIPPED, its 120s tool budget not spent)")
                else:
                    logger.warning(
                        f"[mux] winning candidate {best['source_label']} "
                        f"digest MISMATCH vs registration "
                        f"(stored md5={_stored_md5} size={_stored_size}, "
                        f"now={_dig_now}) — falling back to the full "
                        f"structural-validate path (fail closed)")
            else:
                logger.info(
                    f"[mux] winning candidate {best['source_label']} has "
                    f"no registration digest (verify=False registration "
                    f"or digest failure) — full structural-validate path")
        elif v40_flag_env_present("FPL26_MUX_MD5_TRUST",
                                  "FPL26_NO_MUX_MD5_TRUST"):
            logger.info("[mux] md5-trust skipped reason=disabled "
                        "(FPL26_MUX_MD5_TRUST off or FPL26_NO_MUX_MD5_TRUST "
                        "set) — full structural-validate path")
        # Never-worse: structurally validate the candidate source before
        # touching output_dcp.  If it fails, output_dcp stays untouched
        # and normal finalize ships the pipeline byte-identically.
        if not _md5_trusted:
            try:
                val = await self._structural_validate_dcp(src)
            except Exception as e:
                logger.warning(
                    f"[mux] candidate structural-validate raised "
                    f"{type(e).__name__}; KEEPING pipeline: {e}")
                return False
            if not val.get("valid"):
                logger.warning(
                    f"[mux] winning candidate {best['source_label']} FAILED "
                    f"structural validate ({val.get('reason')}); KEEPING "
                    "pipeline (never-worse).")
                return False
            val_wns = val.get("wns")
            if (val_wns is not None
                    and val_wns < float(self.initial_wns) - 0.005):
                logger.warning(
                    f"[mux] winning candidate {best['source_label']} routes "
                    f"cleanly but WNS regressed ({val_wns:.3f} < initial "
                    f"{self.initial_wns:.3f}); KEEPING pipeline.")
                return False
        # Ship the candidate.
        try:
            self._path_guard_check(output_dcp, context="finalize_candidate_mux")
            _atomic_copy(str(src), str(output_dcp))
        except Exception as e:
            logger.warning(
                f"[mux] candidate copy to output failed ({type(e).__name__}: "
                f"{e}); KEEPING pipeline.")
            return False
        # EDIF: re-open the shipped output and regenerate (best-effort;
        # DCP-only ship is still VALID_OPTIMIZED_NO_EDIF).
        edif_ok = False
        out_edif = Path(output_dcp).with_suffix(".edf")
        try:
            _reopen = await self.call_tool("vivado_open_checkpoint", {
                "dcp_path": str(Path(output_dcp).resolve())})
            if not _looks_like_tool_error(_reopen):
                edif_ok = await self._write_readable_edif(Path(output_dcp))
        except Exception as e:
            logger.warning(f"[mux] EDIF regen failed: {e}")
        self.final_status = (
            "VALID_OPTIMIZED" if edif_ok else "VALID_OPTIMIZED_NO_EDIF")
        # Reflect the shipped artifact in best_wns so the finalize_end
        # trace / reporting describe what actually shipped.
        beaten = pipeline_ship_wns
        self.best_wns = float(best["wns"])
        self.lifecycle_log.append({
            "event": "final_candidate_mux_ship",
            "source": best["source_label"],
            "candidate_wns": best["wns"],
            "pipeline_wns": beaten,
            "n_candidates": n_total,
            "edif_written": edif_ok,
        })
        self._record_ship_lineage(
            dcp_path=Path(output_dcp),
            edif_ok=edif_ok,
            fallback_source="external_write_validated",
            branch="final_candidate_mux",
        )
        logger.info(
            f"[mux] FINAL = {best['source_label']} wns={best['wns']:.3f} "
            f"(beat pipeline {beaten:.3f}); shipped argmax of {n_total} "
            f"candidates.")
        print(
            f"\n[LIFECYCLE] {self.final_status}: shipped INSURED-COMPARE "
            f"candidate '{best['source_label']}' "
            f"(wns={best['wns']:.3f} ns, beat pipeline {beaten:.3f} ns) "
            f"via final-candidate MUX (argmax of {n_total}).")
        return True

    async def _finalize_output_dcp(self, output_dcp: Path) -> None:
        """End-of-optimization DCP finalization with EDIF + validation + fallback.

        This is the single entry point all optimize() exit paths must use to
                produce a submission-safe artifact.  It guarantees one of these
        outcomes (recorded in self.final_status):

                  VALID_OPTIMIZED             — output_dcp passes structural checks + EDIF written.
                  VALID_OPTIMIZED_NO_EDIF     — output_dcp passes structural checks but EDIF write
        failed (DCP usable for self-checking but NOT for RapidWright-based
        validators).
                  VALID_FALLBACK_BASELINE     — output_dcp invalid/missing; baseline copied in its
                                                place.  Score 0 but submission-safe.
                  NO_IMPROVEMENT              — best_wns <= initial_wns; per contest contract, no
        DCP is written (score 0).
                  HARD_FAIL_NO_VALID_BASELINE — neither output nor baseline is usable.  Loud failure.

        Sets self._in_finalize=True for the duration so that call_tool
                bypasses the budget gate.  Inner EDIF refresh
                (_finalize_refresh_edif) inherits the bypass.  The legacy
        _finalize_fresh_write_from_vivado helper that dumped Vivado's
        unverified in-memory state has been removed.
        """
        self._in_finalize = True
        # Backstop: finalize unconditionally releases the polish
        # reserve — whatever polish did or did not happen, nothing may
        # hold reserved wall past this point (never strand wall time).
        self._release_polish_reserve("finalize")
        # Emit finalize-start so the trace has a
        # clear lifecycle anchor.  Final status is emitted after
        # _impl returns (so the record reflects what actually shipped).
        try:
            self._emit_decision({
                "decision_source": "finalize",
                "phase": "finalize",
                "action_label": "finalize_start",
                "output_dcp_path": str(Path(output_dcp).resolve()),
                "wns_after": (
                    float(self.best_wns)
                    if (self.best_wns is not None
                        and self.best_wns != float("-inf"))
                    else None
                ),
                "best_valid_token": self._best_valid_token,
                "ship_lineage_source": (
                    self._best_valid_lineage.get("source")
                    if isinstance(self._best_valid_lineage, dict) else None
                ),
            })
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug(f"finalize_start trace emit failed: {exc}")
        try:
            await self._finalize_output_dcp_impl(output_dcp)
            # The emergency handler trusts output_dcp only after this flag is set.
            # Before then, the best_valid mirror takes precedence over any
            # partially written or regressed output artifact.
            self._finalize_completed = True
            # Persist the finalized artifact's size and streamed MD5 in a sidecar.
            # Later existence checks trust the output only when its identity
            # matches, protecting against post-finalization replacement.
            # A missing artifact identity preserves the fallback behavior.
            try:
                ident = _artifact_identity(Path(output_dcp))
                self._shipped_artifact = ident
                if ident is not None:
                    _write_shipped_manifest(Path(output_dcp), ident)
                    logger.info(
                        "[LIFECYCLE] shipped-artifact identity recorded: "
                        f"size={ident['size']} md5={ident['md5']}")
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    f"shipped-artifact identity record failed: {exc}")
        finally:
            self._in_finalize = False
            # Always emit the canonical finalization record, including lineage,
            # status, and artifact paths, even if finalization raised.
            try:
                fmax_after = None
                if (self.best_wns is not None
                        and self.best_wns != float("-inf")
                        and self.clock_period):
                    fmax_after = self.calculate_fmax(
                        self.best_wns, self.clock_period
                    )
                self._emit_decision({
                    "decision_source": "finalize",
                    "phase": "finalize",
                    "action_label": "finalize_end",
                    "output_dcp_path": str(Path(output_dcp).resolve()),
                    "validity_state": self.final_status,
                    "wns_after": (
                        float(self.best_wns)
                        if (self.best_wns is not None
                            and self.best_wns != float("-inf"))
                        else None
                    ),
                    "fmax_after": fmax_after,
                    "ship_lineage_source": (
                        self._ship_lineage.get("source")
                        if isinstance(self._ship_lineage, dict) else None
                    ),
                    "best_valid_checkpoint_path": (
                        str(self._best_valid_dcp)
                        if self._best_valid_dcp is not None else None
                    ),
                    "best_valid_token": self._best_valid_token,
                })
            except Exception as exc:  # pragma: no cover — defensive
                logger.debug(f"finalize_end trace emit failed: {exc}")
            # Optional post-finalization QoR capture runs in an isolated subprocess
            # with a 60 s timeout. Failures are non-fatal and cannot alter the
            # shipped artifacts or lifecycle status.
            try:
                if getattr(self, "capture_qor", False):
                    await self._maybe_capture_qor_post_finalize(output_dcp)
            except Exception as exc:  # pragma: no cover — defensive
                logger.debug(f"_maybe_capture_qor_post_finalize raised: {exc}")

    async def _maybe_capture_qor_post_finalize(self, output_dcp: Path) -> None:
        """Run report_design_analysis -qor_summary -json on the finalised DCP.

        - Outputs JSON into ``<run_dir>/<dcp_stem>.qor.json``.
        - Spawns a single Vivado batch process (subprocess-isolated; does
          NOT reuse the optimizer's possibly-wedged MCP session).
        - 60 s timeout.  Timeout, failure, missing DCP all logged as
          decision-trace records but NEVER raise.
        - Refuses to write into a path under ``submission/``.
        - Never consults design_name as a strategy condition.

        This method is report-only data collection.  The optimizer's
        lifecycle status (VALID_OPTIMIZED / NO_IMPROVEMENT / etc.) is
        already set before this runs.
        """
        # Resolve to absolute BEFORE handing to subprocess — the Vivado
        # batch runs with cwd=run_dir, so a relative path would resolve
        # under run_dir/ and the Tcl probe would emit ERR_NO_DCP.
        try:
            out = Path(output_dcp).resolve()
        except Exception:
            out = Path(output_dcp)
        run_dir = getattr(self, "run_dir", None)
        if run_dir is None:
            self._emit_decision({
                "decision_source": "qor_capture",
                "phase": "finalize",
                "action_label": "qor_capture",
                "enabled": True, "status": "skipped",
                "json_path": None, "source_dcp_path": str(out),
                "runtime_seconds": 0.0,
                "error_summary": "run_dir not set on optimizer",
                "report_only": True, "used_for_decision": False,
            })
            return
        run_dir = Path(run_dir)
        # PathGuard: refuse to write JSON anywhere under submission/.
        json_path = run_dir / f"{out.stem}.qor.json"
        if _path_is_in_submission(json_path):
            self._emit_decision({
                "decision_source": "qor_capture",
                "phase": "finalize",
                "action_label": "qor_capture",
                "enabled": True, "status": "refused_submission_path",
                "json_path": str(json_path),
                "source_dcp_path": str(out),
                "runtime_seconds": 0.0,
                "error_summary": "json target under submission/",
                "report_only": True, "used_for_decision": False,
            })
            return
        # Skip if final DCP is missing / empty.
        if not out.exists() or out.stat().st_size == 0:
            self._emit_decision({
                "decision_source": "qor_capture",
                "phase": "finalize",
                "action_label": "qor_capture",
                "enabled": True, "status": "skipped",
                "json_path": str(json_path),
                "source_dcp_path": str(out),
                "runtime_seconds": 0.0,
                "error_summary": "output_dcp missing or empty",
                "report_only": True, "used_for_decision": False,
            })
            return
        # Tunable timeout; defaults to 60s on legacy callers.
        try:
            timeout_s = float(getattr(self, "capture_qor_timeout", 60.0))
            if not (timeout_s > 0):
                timeout_s = 60.0
        except (TypeError, ValueError):
            timeout_s = 60.0
        t0 = time.monotonic()
        try:
            status, err = await self._run_qor_capture_subprocess(
                dcp_path=out, json_path=json_path, timeout_s=timeout_s,
            )
        except Exception as exc:  # truly defensive
            status, err = "error", f"runner_exception: {exc!r}"
        runtime = round(time.monotonic() - t0, 3)
        self._emit_decision({
            "decision_source": "qor_capture",
            "phase": "finalize",
            "action_label": "qor_capture",
            "enabled": True, "status": status,
            "json_path": str(json_path),
            "source_dcp_path": str(out),
            "runtime_seconds": runtime,
            "timeout_seconds": timeout_s,
            "error_summary": err,
            "report_only": True, "used_for_decision": False,
        })

    def _finalize_no_improvement(self) -> bool:
        """Determine whether finalization must report no improvement and retain
        the baseline.

        An unavailable initial WNS always means no improvement because no
        candidate can be proven better than an unmeasured baseline. When both
        values are measurable, the result is no improvement when the best WNS
        does not exceed the initial WNS. Candidate lineage does not establish a
        comparable baseline measurement, so uncertainty resolves toward
        shipping the baseline rather than risking a regression.
        """
        if self.best_wns is None or self.best_wns == float("-inf"):
            return True
        if self.initial_wns is None:
            return True
        return self.best_wns <= self.initial_wns

    async def _finalize_output_dcp_impl(self, output_dcp: Path) -> None:
        """Inner implementation of _finalize_output_dcp.

        Split out so the outer wrapper can guarantee _in_finalize is
        cleared even if the body raises.
        """
        output_dcp = Path(output_dcp)

        # Check whether timing constraints changed during optimization.
        # The guard warns but fails open so detection errors cannot block output.
        try:
            await self._verify_constraints_unchanged()
        except Exception as e:
            logger.warning(f"[constraint-guard] ship check raised "
                           f"(ignored): {e!r}")

        # Run the final-candidate mux before pipeline finalization.
        # A structurally valid, verified non-pipeline candidate replaces the
        # pipeline result only when its scored-clock WNS is strictly better.
        # Validation failure or a tie leaves the pipeline output unchanged.
        if await self._maybe_ship_final_candidate_mux(output_dcp):
            return

        # Prefer the on-disk best_valid mirror when it represents a verified
        # improvement; copying it remains safe if the tool session is unavailable.
        # The regression guard still validates the mirror structurally.
        # Unmeasured or unimproved runs must use the baseline fallback instead.
        no_improvement = self._finalize_no_improvement()

        # The mirror is current only when its recorded WNS matches best_wns.
        # A missing or mismatched marker may identify a stale baseline artifact,
        # which must not be labeled as the optimized result.
        mirror_dcp_fresh = (
            not no_improvement
            and self._best_valid_dcp is not None
            and Path(self._best_valid_dcp).exists()
            and Path(self._best_valid_dcp).stat().st_size > 0
            and self._best_valid_dcp_wns is not None
            and self.best_wns is not None
            and abs(self._best_valid_dcp_wns - self.best_wns) < 0.005
        )

        if (not no_improvement
                and self._best_valid_dcp is not None
                and not mirror_dcp_fresh):
            # Finalization trusts known-good disk artifacts, not the tool's current
            # in-memory state, which may have regressed after the best result.
            # A stale mirror may ship only if its recorded WNS beats the baseline;
            # otherwise finalization falls back without writing the current state.
            stale_mirror_path = Path(self._best_valid_dcp)
            mirror_wns = self._best_valid_dcp_wns
            stale_mirror_is_improvement = (
                mirror_wns is not None
                and self.initial_wns is not None
                and mirror_wns > self.initial_wns
                and stale_mirror_path.exists()
                and stale_mirror_path.stat().st_size > 0
            )
            logger.warning(
                f"_finalize_output_dcp: best_valid mirror is STALE "
                f"(mirror_wns={mirror_wns}, current best_wns="
                f"{self.best_wns:.3f}); REFUSING to dump Vivado in-memory "
                f"state (root cause of an observed regression class). "
                f"stale_mirror_is_improvement={stale_mirror_is_improvement}, "
                f"will {'ship mirror file' if stale_mirror_is_improvement else 'fall back to baseline'}."
            )
            self.lifecycle_log.append({
                "event": "stale_mirror_detected",
                "mirror_wns": mirror_wns,
                "current_best_wns": self.best_wns,
                "initial_wns": self.initial_wns,
                "decision": ("ship_stale_mirror_file"
                             if stale_mirror_is_improvement
                             else "fall_through_to_baseline"),
            })
            if stale_mirror_is_improvement:
                try:
                    self._path_guard_check(
                        output_dcp,
                        context="finalize_stale_mirror_disk_copy",
                    )
                    _atomic_copy(str(stale_mirror_path), str(output_dcp))
                    # Also classify the stale-mirror branch so the
                    # decision trace records it as a STALE_MIRROR event.
                    tool_err = self._classify_tool_payload(
                        f"stale_mirror_detected mirror_wns={mirror_wns} best_wns={self.best_wns}",
                        context="finalize_stale_mirror",
                    )
                    self._emit_decision({
                        "decision_source": "finalize",
                        "phase": "finalize",
                        "action_label": "stale_mirror_disk_copy",
                        "wns_after": self.best_wns,
                        "tool_error_code": tool_err.code if tool_err is not None else "STALE_MIRROR",
                        "output_dcp_path": str(output_dcp),
                        "notes": (
                            f"mirror_wns={mirror_wns} best_wns={self.best_wns} "
                            f"initial_wns={self.initial_wns}"
                        ),
                    })
                    # Try to also bring the matching EDIF if present.
                    src_edif = stale_mirror_path.with_suffix(".edf")
                    out_edif = output_dcp.with_suffix(".edf")
                    edif_ok = False
                    if (src_edif.exists() and src_edif.stat().st_size > 0):
                        try:
                            self._path_guard_check(
                                out_edif,
                                context="finalize_stale_mirror_disk_copy_edif",
                            )
                            _atomic_copy(str(src_edif), str(out_edif))
                            edif_ok = (out_edif.exists()
                                       and out_edif.stat().st_size > 0)
                        except Exception as e:
                            logger.warning(
                                f"stale-mirror EDIF copy failed: {e}")
                    self.final_status = (
                        "VALID_OPTIMIZED" if edif_ok
                        else "VALID_OPTIMIZED_NO_EDIF"
                    )
                    # What shipped holds mirror_wns, not best_wns —
                    # make the token_usage report (select_best's fmax
                    # source) describe the artifact, not the claim.
                    self._shipped_wns_ns = mirror_wns
                    self.lifecycle_log.append({
                        "event": "stale_mirror_recovered_via_disk_copy",
                        "src": str(stale_mirror_path),
                        "mirror_wns": mirror_wns,
                        "edif_written": edif_ok,
                    })
                    self._record_ship_lineage(
                        dcp_path=output_dcp,
                        edif_ok=edif_ok,
                        fallback_source="stale_mirror_file",
                        branch="stale_mirror_disk_copy",
                    )
                    print(
                        f"\n[LIFECYCLE] {self.final_status}: shipped stale "
                        f"best_valid mirror via disk copy "
                        f"(mirror_wns={mirror_wns:.3f} ns, "
                        f"edif={'yes' if edif_ok else 'no'}). "
                        f"NOTE: optimizer tracked best_wns={self.best_wns:.3f} "
                        f"but mirror only captured up to {mirror_wns:.3f}; "
                        f"shipping disk truth, not in-memory claim."
                    )
                    return
                except Exception as e:
                    logger.warning(
                        f"stale-mirror disk copy raised: {e}; "
                        "falling through to baseline."
                    )
            # Else: stale mirror is at/below baseline — useless. Fall
            # through to baseline fallback (Step 2 below).

        if mirror_dcp_fresh:
            try:
                self._path_guard_check(
                    output_dcp, context="finalize_fast_path_best_valid_copy",
                )
                _atomic_copy(str(self._best_valid_dcp), str(output_dcp))
                # EDIF freshness must track the DCP's best WNS because netlist-changing
                # transforms can invalidate an older EDIF mirror.
                edif_mirror_fresh = (
                    self._best_valid_edif is not None
                    and Path(self._best_valid_edif).exists()
                    and self._best_valid_edif_wns is not None
                    and self.best_wns is not None
                    and abs(self._best_valid_edif_wns - self.best_wns) < 0.005
                )
                edif_ok = False
                out_edif = output_dcp.with_suffix(".edf")
                if edif_mirror_fresh:
                    try:
                        self._path_guard_check(
                            out_edif,
                            context="finalize_fast_path_best_valid_copy_edif",
                        )
                        _atomic_copy(str(self._best_valid_edif), str(out_edif))
                        edif_ok = out_edif.exists() and out_edif.stat().st_size > 0
                    except Exception as e:
                        logger.warning(f"best_valid EDIF copy to output failed: {e}")
                else:
                    # Regenerate a stale EDIF from the optimized in-memory netlist.
                    # This is best-effort; failure does not block a DCP-only result.
                    logger.info(
                        f"best_valid EDIF mirror is stale "
                        f"(edif_wns={self._best_valid_edif_wns}, "
                        f"current best_wns={self.best_wns:.3f}); regenerating "
                        f"fresh EDIF from Vivado in-memory state."
                    )
                    edif_ok = await self._finalize_refresh_edif(out_edif)
                self.final_status = (
                    "VALID_OPTIMIZED" if edif_ok else "VALID_OPTIMIZED_NO_EDIF"
                )
                self.lifecycle_log.append({
                    "event": "fast_path_best_valid_copy",
                    "src": str(self._best_valid_dcp),
                    "best_wns": self.best_wns,
                    "mirror_wns": self._best_valid_dcp_wns,
                    "edif_mirror_fresh": edif_mirror_fresh,
                    "edif_written": edif_ok,
                })
                # Ship lineage: fast path == current best-valid lineage,
                # since the disk mirror it points at was just shutil-copied.
                self._record_ship_lineage(
                    dcp_path=output_dcp,
                    edif_ok=edif_ok,
                    inherit_best_valid=True,
                    fallback_source="external_write_validated",
                    branch="fast_path_best_valid_copy",
                )
                print(
                    f"\n[LIFECYCLE] {self.final_status}: shipped best_valid mirror "
                    f"({Path(self._best_valid_dcp).name}, "
                    f"best_wns={self.best_wns:.3f} ns) via shutil.copy. "
                    f"ship_lineage_token={self._ship_lineage.get('token')}"
                )
                self._maybe_write_b3_floor_token()
                return
            except Exception as e:
                logger.warning(
                    f"_finalize_output_dcp fast path (best_valid copy) failed: "
                    f"{e}; falling through to Vivado-based finalize."
                )

        # Step 1: Defensive auto-write (preserves prior behaviour).
        await self._ensure_output_dcp_written(output_dcp)
        # Only the fast path writes the floor sentinel. Its absence on the slower
        # finalize path preserves the wrapper's fail-safe default behavior.

        # A run without a verified improvement must emit the baseline artifact,
        # overwriting any output written during optimization.
        # An unknown initial WNS also counts as no improvement because a gain
        # cannot be established.
        no_improvement = self._finalize_no_improvement()
        if no_improvement:
            logger.info("No improvement detected; copying baseline as submission-safe output "
                        "(overwrites any LLM-written DCP).")
            self.lifecycle_log.append({"event": "no_improvement_baseline_copy",
                                        "best_wns": self.best_wns,
                                        "initial_wns": self.initial_wns,
                                        "output_existed": output_dcp.exists()})
            if self.input_dcp_path and self.input_dcp_path.exists():
                try:
                    self._path_guard_check(
                        output_dcp,
                        context="finalize_no_improvement_baseline_copy",
                    )
                    _atomic_copy(str(self.input_dcp_path), str(output_dcp))
                    out_edif = output_dcp.with_suffix(".edf")
                    edif_ok = False
                    # Reuse the mirrored best-valid EDIF when available, avoiding a
                    # live tool call during finalization. This preserves an EDIF even
                    # when the dispatcher exhausts its budget.
                    if (self._best_valid_edif is not None
                            and Path(self._best_valid_edif).exists()
                            and Path(self._best_valid_edif).stat().st_size > 0):
                        try:
                            self._path_guard_check(
                                out_edif,
                                context="finalize_no_improvement_baseline_edif_copy",
                            )
                            _atomic_copy(str(self._best_valid_edif), str(out_edif))
                            edif_ok = out_edif.exists() and out_edif.stat().st_size > 0
                            if edif_ok:
                                logger.info(
                                    f"No-improvement EDIF via best_valid mirror: "
                                    f"{Path(self._best_valid_edif).name} → {out_edif.name}"
                                )
                        except Exception as e:
                            logger.warning(f"best_valid EDIF copy failed: {e}")
                    # Slow path: regenerate EDIF from Vivado.  This only
                    # succeeds when the deadline-aware dispatcher hasn't
                    # already flipped _budget_killed.
                    if not edif_ok:
                        try:
                            await self.call_tool("vivado_open_checkpoint", {
                                "dcp_path": str(output_dcp.resolve())
                            })
                            edif_ok = await self._write_readable_edif(output_dcp)
                        except Exception as e:
                            logger.warning(f"EDIF for no-improvement baseline copy failed: {e}")
                    self.final_status = (
                        "VALID_FALLBACK_BASELINE" if edif_ok
                        else "VALID_FALLBACK_BASELINE_NO_EDIF"
                    )
                    self._record_ship_lineage(
                        dcp_path=output_dcp,
                        edif_ok=edif_ok,
                        no_improvement=True,
                        branch="no_improvement_baseline_copy",
                    )
                    print(f"\n[LIFECYCLE] {self.final_status}: no improvement found; "
                          f"copied baseline DCP to output (ΔFmax = 0).")
                    return
                except Exception as e:
                    logger.exception(f"Baseline copy failed: {e}")
            self.final_status = "HARD_FAIL_NO_VALID_BASELINE"
            self._record_ship_lineage(
                dcp_path=output_dcp,
                edif_ok=False,
                hard_fail=True,
                branch="no_improvement_baseline_missing",
            )
            print(f"\n[LIFECYCLE] {self.final_status}: no improvement and baseline missing.")
            return

        # Step 3: Write a RapidWright-readable EDIF.  Re-open output_dcp first
        # so Vivado's state matches what's on disk — write_edif writes the
        # current in-memory netlist, not the file's contents.
        edif_ok = False
        if output_dcp.exists():
            try:
                _reopen = await self.call_tool("vivado_open_checkpoint", {
                    "dcp_path": str(output_dcp.resolve())
                })
                # An error envelope means the open failed and the
                # prior design is still in memory — writing an EDIF from
                # it would pair a mismatched netlist with the output DCP.
                if _looks_like_tool_error(_reopen):
                    logger.warning("_finalize_output_dcp: re-open returned "
                                   f"error envelope: {str(_reopen)[:120]}")
                else:
                    edif_ok = await self._write_readable_edif(output_dcp)
            except Exception as e:
                logger.warning(f"_finalize_output_dcp: re-open of output_dcp failed: {e}")

        # Accept the output only if it is structurally valid and its WNS
        # does not regress from the baseline; either failure is fatal.
        val = {"valid": False, "reason": "output_dcp missing"}
        if output_dcp.exists():
            val = await self._structural_validate_dcp(output_dcp)

        if val.get("valid"):
            # WNS-regression guard: tolerate a tiny epsilon to absorb
            # Vivado's report-to-report jitter; anything worse is a fail.
            val_wns = val.get("wns")
            if (self.initial_wns is not None and val_wns is not None
                    and val_wns < self.initial_wns - 0.005):
                logger.warning(
                    f"_finalize_output_dcp: output DCP routes cleanly but WNS regressed "
                    f"({val_wns:.3f} ns < initial {self.initial_wns:.3f} ns). Falling back."
                )
                self.lifecycle_log.append({"event": "regression_detected",
                                            "validated_wns": val_wns,
                                            "initial_wns": self.initial_wns})
                # Fall through to the baseline-fallback branch.
            else:
                self.final_status = "VALID_OPTIMIZED" if edif_ok else "VALID_OPTIMIZED_NO_EDIF"
                self.lifecycle_log.append({"event": "valid_optimized",
                                            "wns": val_wns,
                                            "edif_written": edif_ok})
                # Attribute the validated output to the best-valid lineage when
                # possible. Otherwise record external-write validation to mark
                # that the artifact is valid but lacks a traced mirror event.
                self._record_ship_lineage(
                    dcp_path=output_dcp,
                    edif_ok=edif_ok,
                    inherit_best_valid=True,
                    fallback_source="external_write_validated",
                    branch="step4_structurally_validated",
                )
                print(f"\n[LIFECYCLE] {self.final_status}: output DCP validated "
                      f"(WNS={val_wns}, route_errors={val.get('route_errors')}). "
                      f"ship_lineage_source={self._ship_lineage.get('source')}")
                return

        # Step 5: Fallback to baseline.  Copy input → output and try to EDIF it.
        logger.warning(f"_finalize_output_dcp: output invalid ({val.get('reason')}); "
                       f"falling back to baseline.")
        self.lifecycle_log.append({"event": "fallback_to_baseline",
                                    "reason": val.get("reason")})
        # Classify the validator reason so the
        # decision trace records the specific failure code.
        try:
            tool_err = self._classify_tool_payload(
                str(val.get("reason") or "validator_mismatch"),
                context="finalize_validator",
            )
            self._emit_decision({
                "decision_source": "validator",
                "phase": "finalize",
                "action_label": "step5_fallback_to_baseline",
                "wns_after": self.best_wns,
                "tool_error_code": tool_err.code if tool_err is not None else "VALIDATOR_MISMATCH",
                "notes": f"reason={val.get('reason')}",
            })
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug(f"finalize validator trace emit failed: {exc}")
        if self.input_dcp_path and self.input_dcp_path.exists():
            try:
                self._path_guard_check(
                    output_dcp, context="finalize_step5_baseline_fallback",
                )
                _atomic_copy(str(self.input_dcp_path), str(output_dcp))
                # Try to write EDIF for the fallback too — open baseline
                # so Vivado state matches.
                try:
                    await self.call_tool("vivado_open_checkpoint", {
                        "dcp_path": str(output_dcp.resolve())
                    })
                    await self._write_readable_edif(output_dcp)
                except Exception as e:
                    logger.warning(f"EDIF for baseline fallback failed: {e}")
                self.final_status = "VALID_FALLBACK_BASELINE"
                self._record_ship_lineage(
                    dcp_path=output_dcp,
                    edif_ok=False,
                    baseline=True,
                    branch="step5_invalid_output_baseline_fallback",
                )
                print(f"\n[LIFECYCLE] {self.final_status}: optimization output was invalid "
                      f"({val.get('reason')}); copied baseline DCP to output path. "
                      f"ship_lineage_source={self._ship_lineage.get('source')}")
                return
            except Exception as e:
                logger.exception(f"Failed to copy baseline to output: {e}")
        self.final_status = "HARD_FAIL_NO_VALID_BASELINE"
        self._record_ship_lineage(
            dcp_path=output_dcp,
            edif_ok=False,
            hard_fail=True,
            branch="step5_no_valid_baseline",
        )
        print(f"\n[LIFECYCLE] {self.final_status}: no valid baseline available — "
              f"reason: {val.get('reason')}.")

    def _print_optimization_summary(self, max_iterations_reached: bool = False):
        """Print the optimization summary without affecting finalization or
        raising exceptions.

        Reporting runs after the artifact is finalized and tolerates incomplete
        bookkeeping entries, including missing elapsed-time values. All
        formatting and output failures remain isolated so they cannot change
        the published artifact or the run status.
        """
        try:
            self._print_optimization_summary_inner(
                max_iterations_reached=max_iterations_reached)
        except Exception as e:
            logger.warning(
                f"optimization summary failed (ignored, artifact already "
                f"finalized): {type(e).__name__}: {e!r}")

    def _print_optimization_summary_inner(self,
                                          max_iterations_reached: bool = False):
        """Body of the summary printer — may raise; the wrapper absorbs it."""
        # Persist run outcome to strategy memory before printing — so even
        # if console rendering fails the record lands.
        try:
            self._persist_to_strategy_memory()
        except Exception:
            pass
        title = "Optimization Summary (Max Iterations Reached)" if max_iterations_reached else "Optimization Summary"
        print(f"\n{'='*70}")
        print(f"{title}")
        print(f"{'='*70}")

        # DCP lifecycle final status — front and centre so it's the first
        # thing in the summary.  Required by the submission-safety contract.
        if self.final_status is not None:
            print(f"\nLIFECYCLE STATUS: {self.final_status}")

        # Wall-time budget summary.  Surfaced prominently so a reader can tell
        # at a glance whether the run was budget-constrained.
        if self.max_wall_seconds is not None:
            elapsed = (self.end_time or time.time()) - (self.start_time or time.time())
            print(f"\nWALL BUDGET: {elapsed:.0f}s elapsed of "
                  f"{self.max_wall_seconds:.0f}s "
                  f"(deadline-reserve={self._finalize_reserve_seconds:.0f}s)")
            if self._strategies_skipped_budget:
                print(f"  Strategies skipped due to budget: "
                      f"{len(self._strategies_skipped_budget)}")
                for s in self._strategies_skipped_budget[:5]:
                    print(f"    - {s}")
        
        # Calculate total runtime
        if self.start_time is not None:
            total_runtime = (self.end_time or time.time()) - self.start_time
            print(f"\nTOTAL RUNTIME: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
        
        best_wns = self.best_wns if self.best_wns > float('-inf') else None
        result_lines = self._format_fmax_results(
            self.clock_period, self.initial_wns, best_wns, result_label="Best"
        )
        if result_lines:
            print(f"\nFMAX RESULTS:")
            print("\n".join(result_lines))
        
        # Iteration stats
        print(f"\nITERATION STATS:")
        print(f"  Total iterations:    {self.iteration}")
        print(f"  LLM API calls:       {self.llm_call_count}")
        print(f"  Force-continues:     {self.force_continue_count}  [BETA-CTRL-V0: slope-aware ceiling lifts]")
        print(f"  Revert warnings:     {self.regression_warning_count}  [BETA-CTRL-V0.1: regression-revert nudges]")
        print(f"  Last improvement:    iter {self.last_improvement_iter}")
        # Eval-forensics counters — an API storm is
        # reconstructable from the run summary alone (api_error_episodes /
        # total_backoff_s pair with the [api-resilience] log lines).
        print(f"  API error episodes:  {self._api_error_episodes}  [key/transient storms absorbed by in-call backoff]")
        print(f"  Total backoff:       {self._total_backoff_s:.0f}s")
        
        # Token usage
        print(f"\nTOKEN USAGE:")
        print(f"  Prompt tokens:       {self.total_prompt_tokens:,}")
        print(f"  Completion tokens:   {self.total_completion_tokens:,}")
        print(f"  Total tokens:        {self.total_tokens:,}")
        
        # Calculate total cached and reasoning tokens
        total_cached = sum(detail.get('cached_tokens', 0) for detail in self.api_call_details)
        total_reasoning = sum(detail.get('reasoning_tokens', 0) for detail in self.api_call_details)
        
        if total_cached > 0:
            print(f"  Cached tokens:       {total_cached:,} (saved cost)")
        if total_reasoning > 0:
            print(f"  Reasoning tokens:    {total_reasoning:,}")
        
        # Cost
        print(f"\nCOST:")
        print(f"  Model:               {self.model}")
        if self.total_cost > 0:
            print(f"  Total cost:          ${self.total_cost:.4f}")
        else:
            print(f"  Total cost:          Not available")
        
        # Tool call summary
        if self.tool_call_details:
            print(f"\nTOOL CALLS SUMMARY:")
            print(f"  Total tool calls:    {len(self.tool_call_details)}")
            
            # Calculate total time spent in tool calls
            total_tool_time = sum(detail.get('elapsed_time', 0.0)
                                  for detail in self.tool_call_details)
            print(f"  Total tool time:     {total_tool_time:.2f}s")

            # Count by tool type
            tool_counts = {}
            for detail in self.tool_call_details:
                tool_name = detail.get('tool_name', 'unknown')
                if tool_name not in tool_counts:
                    tool_counts[tool_name] = 0
                tool_counts[tool_name] += 1
            
            print(f"\n  Tool call breakdown:")
            for tool_name, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
                print(f"    {tool_name}: {count}")
            
            # Detailed tool call list
            print(f"\n  Detailed tool call log:")
            print(f"  {'#':<5} {'Iter':<6} {'Tool':<40} {'Time (s)':<12} {'WNS (ns)':<12} {'Status':<10}")
            print(f"  {'-'*5} {'-'*6} {'-'*40} {'-'*12} {'-'*12} {'-'*10}")
            
            for i, detail in enumerate(self.tool_call_details, 1):
                tool_name = detail.get('tool_name', 'unknown')
                iteration = detail.get('iteration', 0)
                elapsed = detail.get('elapsed_time', 0.0)
                wns = detail.get('wns')
                error = detail.get('error', False)
                
                # Format WNS column
                wns_str = f"{wns:.3f}" if wns is not None else "-"
                
                # Format status
                status_str = "ERROR" if error else "OK"
                
                print(f"  {i:<5} {iteration:<6} {tool_name:<40} {elapsed:<12.2f} {wns_str:<12} {status_str:<10}")
                
                # If error, show error message on next line
                if error and 'error_message' in detail:
                    print(f"        Error: {detail['error_message'][:80]}")
        
        # Per-call breakdown if debug mode
        if self.debug and self.api_call_details:
            print(f"\nPER-CALL BREAKDOWN:")
            
            # Cached and reasoning tokens are only shown when present.
            has_cached = any(detail.get('cached_tokens', 0) > 0 for detail in self.api_call_details)
            has_reasoning = any(detail.get('reasoning_tokens', 0) > 0 for detail in self.api_call_details)
            has_cost = any(detail.get('cost', 0) > 0 for detail in self.api_call_details)
            
            # Build header
            header = f"  {'Call':<6} {'Iter':<6} {'Prompt':<10} {'Completion':<12}"
            if has_cached:
                header += f" {'Cached':<10}"
            if has_reasoning:
                header += f" {'Reasoning':<10}"
            header += f" {'Total':<10}"
            if has_cost:
                header += f" {'Cost':<12}"
            print(header)
            
            # Build separator
            separator = f"  {'-'*6} {'-'*6} {'-'*10} {'-'*12}"
            if has_cached:
                separator += f" {'-'*10}"
            if has_reasoning:
                separator += f" {'-'*10}"
            separator += f" {'-'*10}"
            if has_cost:
                separator += f" {'-'*12}"
            print(separator)
            
            # Print details
            for detail in self.api_call_details:
                line = (f"  {detail['call_number']:<6} {detail['iteration']:<6} "
                       f"{detail['prompt_tokens']:<10,} {detail['completion_tokens']:<12,}")
                if has_cached:
                    line += f" {detail.get('cached_tokens', 0):<10,}"
                if has_reasoning:
                    line += f" {detail.get('reasoning_tokens', 0):<10,}"
                line += f" {detail['total_tokens']:<10,}"
                if has_cost:
                    cost = detail.get('cost', 0)
                    line += f" ${cost:<11.4f}" if cost > 0 else f" {'N/A':<12}"
                print(line)
        
        print(f"\n{'='*70}\n")
        
        # Save detailed report to JSON in run directory
        try:
            report_path = self.run_dir / "token_usage.json"
            self.save_token_usage_report(report_path)
            print(f"Detailed token usage report saved to: {report_path}\n")
        except Exception as e:
            logger.warning(f"Failed to save token usage report: {e}")
