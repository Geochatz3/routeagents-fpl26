"""STAGE 5 — POLISH LADDER (runs after the LLM loop exits)

The polish ladder: ILS ruin-and-rebuild, the tail controller, and the
bare re-route pass that run after the LLM loop exits.

Each rung banks its win before the next risk is taken, so the ladder can be
entered at any depth and abandoned at any point without losing a gain.

Named for the stage it implements: the slide, the video and the paper all
call it by this name, so a reader can move between them without a glossary.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import time
from pathlib import Path

# Log through the orchestrator's logger: this code still belongs to the same
# run, and a module-named logger would change every line it emits.
logger = logging.getLogger("dcp_optimizer")

from optimizer.recipe_policy import (  # noqa: F401
    _fanout_polish_cost_basis,
    deep_first_sizegated_enabled,
    midband_retry_hold_enabled,
    polish_gate_reserve_s,
    recipe_pass_band,
    v40_flag_env_present,
)

MIDBAND_RETRY_HOLD_WNS_MAG_MIN_NS = 1.20  # inclusive floor, |wns_in|
TAIL_RESERVE_WALL_CLAMP_FRAC = 2400.0 / 3500.0  # ≈ 0.686 — the proven arm


from optimizer.config_resolution import (  # noqa: F401
    BARE_REROUTE_MAX_ITERS_DEFAULT,
    BARE_REROUTE_MIN_GAIN_NS_DEFAULT,
    DEEP_WNS_TAIL_RESERVE_S_DEFAULT,
    POLISH_RESERVE_S_DEFAULT,
    TAIL_CTRL_DEEP_WNS_NS_DEFAULT,
    TAIL_CTRL_MAX_MOVES_DEFAULT,
    TAIL_RESERVE_STAGNANT_S_DEFAULT,
    _arg_float,
    _arg_int,
    _run_dir_base,
    resolve_bare_reroute_max_iters,
    resolve_bare_reroute_min_gain_ns,
    resolve_bare_reroute_polish_enabled,
    resolve_deep_wns_tail_reserve_s,
    resolve_polish_reserve_s,
    resolve_tail_controller_enabled,
    resolve_tail_ctrl_deep_wns_ns,
    resolve_tail_ctrl_m1_echo,
    resolve_tail_ctrl_max_moves,
    resolve_tail_reserve_stagnant_s,
)

from typing import Optional

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


class PolishLadderMixin:
    """Methods mixed into DCPOptimizer; they run against its instance state."""

    def _polish_reserve_armed_s(self) -> float:
        """Active post-route polish reserve in seconds (0.0 = not
        armed).  The reserve arms only when reserving can pay off, so an
        unarmed run behaves byte-identically to today (never-worse):

          1. reserve enabled (_polish_reserve_s > 0; 0 = kill switch),
          2. a wall budget is set (contest mode — dev runs unaffected),
          3. not yet released (release is one-way, see
             _release_polish_reserve),
          4. a routed banked best EXISTS (_best_valid_dcp; the bank only
             ever holds validated routed states) — before the first bank
             there is nothing to polish and speculative work must not be
             blocked ("a trimmed zero is worse than a slow zero"),
          5. the polish machinery is not disabled: the ILS/polish stage
             is enabled AND at least one of its polish sub-stages
             (LASTMILE / fanout) is flag-enabled — the existing
             eligibility flags, reused, not re-derived.  Dynamic
             per-design gates (LASTMILE wns entry gate, fanout cheap-
             design anchor gate) are evaluated at ILS-stage time and
             release the reserve when NEITHER stage can fire
             (_ils_polish_body precheck) so a polish-ineligible design
             class never has speculative work fenced for a polish that
             cannot happen.
        """
        if self._polish_reserve_s <= 0.0:
            return 0.0
        if self._budget_deadline is None:
            return 0.0
        if self._polish_reserve_release_reason is not None:
            return 0.0
        if self._best_valid_dcp is None:
            return 0.0
        cfg = self._ils_polish_cfg
        if not getattr(cfg, "enabled", False):
            return 0.0
        if not (getattr(cfg, "lastmile_polish_enabled", True)
                or getattr(cfg, "fanout_polish_enabled", True)):
            return 0.0
        return self._polish_reserve_s

    def _release_polish_reserve(self, reason: str) -> None:
        """One-way release of the polish reserve.  Called when the
        polish stages have run or been correctly skipped (ILS-stage tail,
        dynamic-ineligibility precheck) and unconditionally at finalize —
        the reserve must never strand wall time.  First reason wins."""
        if self._polish_reserve_release_reason is not None:
            return
        self._polish_reserve_release_reason = reason
        if self._polish_reserve_s > 0.0:
            logger.info(f"polish-reserve: released ({reason}); speculative "
                        f"dispatch sees the full remaining window again.")

    async def _run_ils_polish_stage(self) -> None:
        """Run keep-best ILS ruin-and-recreate polishing in a fresh Vivado
        process.

        The process restarts from the raw input DCP because prior recipe
        activity can leave persistent state that degrades later placement, and
        reopening a checkpoint does not clear it. The best existing result is
        retained whenever the new placement underperforms.
        """
        from optimizer.ils_polish import run_ils_polish
        # Set the ILS-stage guard before entering the body so restart and polish
        # calls bypass the near-deadline budget gate. A skipped restart is a
        # silent no-op and would leave ILS running in a polluted tool session.
        self._in_ils_stage = True
        try:
            await self._ils_polish_body(run_ils_polish)
        finally:
            self._in_ils_stage = False

    async def _ils_polish_body(self, run_ils_polish) -> None:
        # Recipe-best artifact: the bar to beat AND the never-worse fallback.
        # Snapshot it to disk BEFORE restarting Vivado (the restart wipes the live
        # design); the DCP carries the design, not the polluted session state.
        recipe_best = self._best_valid_dcp
        if recipe_best is None or not Path(recipe_best).exists():
            recipe_best = (self.run_dir or Path(".")) / "ils_recipe_best.dcp"
            await self.call_tool("vivado_run_tcl",
                                 {"command": f"write_checkpoint -force {{{recipe_best}}}"})
        recipe_best = str(recipe_best)
        # Start ILS in a fresh Vivado session to clear recipe-side state.
        # Invoke restart as vivado_restart_vivado; dispatch requires the
        # vivado_ prefix. Check the response envelope because call_tool returns
        # failures as strings. Use pristine initial WNS for design-relative
        # gates; recipe-best may be improved.
        try:
            if self.initial_wns is not None:
                self._ils_polish_cfg.design_baseline_wns = float(self.initial_wns)
        except Exception as e:                                    # pragma: no cover
            logger.warning(f"ILS: could not publish design baseline wns "
                           f"({type(e).__name__}: {e}); retry baseline gate inert")
        # FPL26_MIDBAND_RETRY_HOLD applies only when 1.05 < |initial WNS| < 8.0 ns.
        # Band classification uses the pristine initial WNS and leaves the global
        # retry setting unchanged. Missing WNS fails off; logging distinguishes
        # a disabled treatment from an unreached code path.
        try:
            if midband_retry_hold_enabled():
                _mrh_band = recipe_pass_band(self.initial_wns)
                if (_mrh_band == "mid"
                        and abs(self.initial_wns)
                        < MIDBAND_RETRY_HOLD_WNS_MAG_MIN_NS):
                    # Re-scope fallback: the low-spread near-met band
                    # [1.05, 1.20) reverts to the prior behavior.
                    logger.info(
                        f"MIDBAND-RETRY-HOLD: skipped reason=subband_floor "
                        f"wns_in={self.initial_wns} (|wns| < "
                        f"{MIDBAND_RETRY_HOLD_WNS_MAG_MIN_NS}; parity-measured "
                        f"optical harm; treatments fail OFF)")
                elif (_mrh_band == "mid"
                        and self._ils_polish_cfg.design_baseline_wns
                        is not None):
                    self._ils_polish_cfg.retry_baseline_gate_scoped = True
                    logger.info(
                        f"MIDBAND-RETRY-HOLD: ARMED band=mid "
                        f"wns_in={self.initial_wns} (retry baseline gate "
                        f"scoped in; global FPL26_ILS_RETRY_BASELINE_GATE "
                        f"untouched)")
                else:
                    logger.info(
                        f"MIDBAND-RETRY-HOLD: skipped reason=band "
                        f"band={_mrh_band} wns_in={self.initial_wns} "
                        f"(mid-band-only scope; treatments fail OFF)")
            else:
                if v40_flag_env_present("FPL26_MIDBAND_RETRY_HOLD",
                                        "FPL26_NO_MIDBAND_RETRY_HOLD"):
                    logger.info("MIDBAND-RETRY-HOLD: skipped reason=disabled "
                                "(FPL26_MIDBAND_RETRY_HOLD off or "
                                "FPL26_NO_MIDBAND_RETRY_HOLD set)")
        except Exception as e:                                    # pragma: no cover
            logger.warning(f"MIDBAND-RETRY-HOLD: skipped reason=exception "
                           f"{type(e).__name__}: {e!r} (non-fatal; gate "
                           f"stays unscoped)")
        try:
            from optimizer.ils_polish import derive_cost_anchors
            _est, _fan, _maxes = derive_cost_anchors(self.tool_call_details)
            if _est > 0:
                self._ils_polish_cfg.expected_heavy_cycle_s = _est + 60.0
                logger.info(
                    f"ILS cold-start anchor: expected_heavy_cycle_s="
                    f"{self._ils_polish_cfg.expected_heavy_cycle_s:.0f} "
                    f"(singles={_maxes})")
            if _fan > 0:
                # Provide fanout polish with a cost anchor when the recipe has no
                # placement sample. The observed fanout duration plus 60 s supplies
                # scheduling margin, and route presence records the anchor's scope.
                self._ils_polish_cfg.fanout_cost_anchor_s = _fan + 60.0
                self._ils_polish_cfg.fanout_anchor_has_route = (
                    "route_design" in _maxes)
                logger.info(
                    f"ILS cold-start anchor: fanout_cost_anchor_s="
                    f"{self._ils_polish_cfg.fanout_cost_anchor_s:.0f} "
                    f"has_route={self._ils_polish_cfg.fanout_anchor_has_route} "
                    f"(singles={_maxes})")
            _route_single = _maxes.get("route_design", 0.0)
            if _route_single > 0:
                # Base route-reroll affordability on this design's observed single-stage
                # route duration. The estimator applies its 1.3 multiplier; this anchor
                # adds the shared 60 s scheduling margin.
                self._ils_polish_cfg.route_cost_anchor_s = _route_single + 60.0
                logger.info(
                    f"ILS cold-start anchor: route_cost_anchor_s="
                    f"{self._ils_polish_cfg.route_cost_anchor_s:.0f} "
                    f"(route-reroll basis; singles={_maxes})")
        except Exception as e:
            logger.warning(f"ILS cold-start anchor derivation failed ({e!r}); "
                           "legacy optimistic cold-start.")
        if self._ils_polish_cfg.restart_vivado_before_ils:
            try:
                _r = await self.call_tool("vivado_restart_vivado", {})
                if isinstance(_r, str) and '"error"' in _r:
                    logger.warning(f"ILS: restart_vivado returned error "
                                   f"({_r[:140]}); staying in current session.")
                else:
                    logger.info("ILS: restarted Vivado fresh (clears recipe session "
                                "pollution that degrades + slows place_design).")
            except Exception as e:
                logger.warning(f"ILS: restart_vivado failed ({e!r}); current session.")
        # For a stuck verdict, probe the raw seed first and recipe-best second.
        # The raw probe receives priority budget and stops early without improvement,
        # leaving time for the corrective recipe-best probe. Global keep-best selects
        # the winner; non-stuck designs use only recipe-best.
        from optimizer.ils_polish import choose_ils_seed, stuck_gain_threshold
        _seed_kind, _gain = choose_ils_seed(
            initial_wns=self.initial_wns, recipe_wns=self.best_wns,
            cfg=self._ils_polish_cfg)
        _g = None if _gain is None else round(_gain, 3)
        _cfg = self._ils_polish_cfg
        # Raw seed copy (only needed for the STUCK path). recipe_best stays pristine.
        raw_seed = None
        if (_seed_kind == "raw" and self.input_dcp_path
                and self.input_dcp_path.exists()):
            try:
                _rs = (self.run_dir or Path(".")) / "ils_ruin_seed.dcp"
                shutil.copy(str(self.input_dcp_path), str(_rs))
                raw_seed = str(_rs)
            except Exception as e:
                logger.warning(f"ILS raw-seed copy failed ({e!r}); recipe-best only.")
                raw_seed = None
        # Keep the banked recipe checkpoint immutable throughout the ILS stage.
        # ILS writes accepted checkpoints into its seed path, while emergency shipping
        # validates the banked mirror against size metadata recorded before the stage.
        # FPL26_ILS_SEED_COPY=0 permits in-place mutation and weakens that invariant.
        recipe_seed = recipe_best
        if os.environ.get("FPL26_ILS_SEED_COPY", "1").strip().lower() in (
                "1", "true", "on", "yes"):
            try:
                _cs = (self.run_dir or Path(".")) / "ils_recipe_seed.dcp"
                shutil.copy(recipe_best, str(_cs))
                recipe_seed = str(_cs)
            except Exception as e:
                # Never fatal: the in-place path is the prior behaviour.
                logger.warning(f"ILS recipe-seed copy failed ({e!r}); recipe-best in place.")
                recipe_seed = recipe_best
        # Each final or sole seed uses the final-seed futility limit so repeated
        # non-improving cycles return unused wall time. A value of 0 runs to budget.
        _K = _cfg.no_improve_stop_cycles
        _KF = _cfg.final_seed_no_improve_stop
        if _seed_kind == "raw" and raw_seed and _cfg.dual_seed:
            seeds = [("raw", raw_seed, _K), ("recipe_best", recipe_seed, _KF)]
            # Log the effective threshold because the environment may override the
            # configured default.
            logger.info(f"ILS STUCK design (recipe_gain={_g}ns < "
                        f"{stuck_gain_threshold(_cfg)}); "
                        f"DUAL-SEED raw->recipe-best (keep-best, never-worse).")
        elif _seed_kind == "raw" and raw_seed:
            seeds = [("raw", raw_seed, _KF)]    # dual-seed disabled -> legacy single raw
            logger.info(f"ILS STUCK design (recipe_gain={_g}ns); single RAW seed "
                        f"(dual_seed off); keep-best.")
        else:
            seeds = [("recipe_best", recipe_seed, _KF)]
            logger.info(f"ILS seeding from recipe-best (recipe_gain={_g}ns; recipe "
                        f"helped or unknown -> preserve it); keep-best.")
        # Target-clock-aware WNS query (mirrors get_wns_for_target_clock).
        if self.target_clock:
            wns_tcl = (
                f"set clk_obj [get_clocks -quiet {{{self.target_clock}}}]; "
                f"if {{$clk_obj ne {{}}}} {{ set tp [get_timing_paths -max_paths 1 "
                f"-setup -to $clk_obj]; if {{[llength $tp] > 0}} {{get_property SLACK $tp}} "
                f"else {{puts 0.0}} }} else {{ set tp [get_timing_paths -max_paths 1 "
                f"-slack_lesser_than 999]; if {{[llength $tp] > 0}} {{get_property SLACK $tp}} "
                f"else {{puts 0.0}} }}"
            )
        else:
            wns_tcl = ("set tp [get_timing_paths -max_paths 1 -slack_lesser_than 999]; "
                       "if {[llength $tp] > 0} {get_property SLACK $tp} else {puts 0.0}")
        deadline = self._budget_deadline or (time.time() + 1200.0)
        # Reserve part of the ILS window for eligible last-mile and fanout polish.
        # Speculative ruin cycles use the fenced deadline, while polish retains the
        # full deadline. If neither polish stage can pass its own eligibility gate,
        # release the reserve immediately rather than strand wall time.
        _pr = self._polish_reserve_armed_s()
        if _pr > 0.0:
            _w_known = (self.best_wns if self.best_wns > float("-inf")
                        else self._best_valid_dcp_wns)
            _lm_ok = (getattr(_cfg, "lastmile_polish_enabled", True)
                      and _w_known is not None
                      and _w_known >= _cfg.lastmile_min_wns_ns)
            _fan_est, _ = _fanout_polish_cost_basis(_cfg)
            _fan_ok = (getattr(_cfg, "fanout_polish_enabled", True)
                       and 0.0 < _fan_est <= _cfg.fanout_max_cycle_s)
            if not (_lm_ok or _fan_ok):
                self._release_polish_reserve(
                    f"ils_polish_ineligible(lastmile_ok={_lm_ok},"
                    f"fanout_ok={_fan_ok},wns={_w_known})")
                _pr = 0.0
            else:
                logger.info(
                    f"polish-reserve: ILS ruin cycles fenced to "
                    f"deadline−{_pr:.0f}s (lastmile_ok={_lm_ok}, "
                    f"fanout_ok={_fan_ok}); polish stages keep the full "
                    f"window.")
        _ils_spec_deadline = deadline - _pr
        # Preserve the unfenced wrapper deadline for redraw-reserve checks.
        # Using the ILS deadline would subtract the polish reserve twice.
        try:
            _cfg.wrapper_deadline_ts = float(deadline)
        except (TypeError, ValueError):
            _cfg.wrapper_deadline_ts = None
        recipe_baseline = self.best_wns if self.best_wns > float("-inf") else None
        # The caller's ILS-stage guard keeps restart, placement, and shipping calls
        # outside the budget-skip gate. Run seeds in order and retain the global best;
        # seed-local baselines may allow stepping stones, but only a checkpoint that
        # beats the global incumbent becomes the winner.
        global_best_wns = recipe_baseline
        global_best_path = recipe_best       # pristine floor if nothing improves
        global_improved = False
        _cycles_done = 0   # corrective probe CONTINUES the combo rotation
        _seed0_res = None
        for _i, (_kind, _path, _nis) in enumerate(seeds):
            # Don't start a later seed without budget for >=1 real cycle.
            # Measured against the reserve-fenced deadline — a
            # corrective seed is speculative work too.
            if _i > 0 and (_ils_spec_deadline - time.time()) < _cfg.exit_min_remaining_s:
                logger.info("ILS: insufficient budget for the corrective seed; stopping.")
                break
            _base = global_best_wns
            _off = _cycles_done
            if _i > 0 and getattr(_cfg, "corrective_local_climb", False):
                # Let the corrective recipe seed accept stepping stones from
                # its own floor. Continue combo rotation from the raw seed's
                # pristine-state offset so completed probes are not repeated.
                # Global keep-best controls output.
                _base = recipe_baseline
                if _seed0_res is not None:
                    _off = _seed0_res.pristine_rot
            logger.info(f"ILS seed[{_i}]={_kind} ({Path(_path).name}); "
                        f"baseline={_base} no_improve_stop={_nis} "
                        f"combo_offset={_off}.")
            # Hurdle continuation requires the current improvement metric to be
            # populated before each seed. Enable it with --ils-hurdle-continue
            # or FPL26_ILS_HURDLE_CONTINUE=1.
            try:
                if getattr(self, "_ils_hurdle_continue", False):
                    _a_now = self.calculate_fmax(self.best_wns, self.clock_period)
                    _a_0 = self.calculate_fmax(self.initial_wns, self.clock_period)
                    _alpha = (_a_now - _a_0) if (_a_now and _a_0) else 0.0
                    self._ils_polish_cfg.hurdle_continue_alpha_mhz = max(0.0, _alpha)
                    logger.info(
                        f"ILS: hurdle-continuation ARMED with banked alpha "
                        f"{_alpha:.2f} MHz — the scoring function, not the K "
                        f"counter, decides whether one more cycle pays.")
                else:
                    self._ils_polish_cfg.hurdle_continue_alpha_mhz = 0.0
            except Exception as _e:                          # pragma: no cover
                self._ils_polish_cfg.hurdle_continue_alpha_mhz = 0.0
                logger.warning(f"ILS: could not arm hurdle continuation "
                               f"({type(_e).__name__}); K-counter only.")
            res = await run_ils_polish(
                self.call_tool, best_dcp_path=_path, baseline_wns=_base,
                deadline_ts=_ils_spec_deadline, wns_tcl=wns_tcl,
                cfg=self._ils_polish_cfg,
                log=lambda m: logger.info(m), no_improve_stop=_nis,
                combo_offset=_off,
                combo_cost_seed=(_seed0_res.combo_cost if _seed0_res else None),
            )
            if _i == 0:
                _seed0_res = res
            self._ils_observed_costs = dict(res.combo_cost or {})
            _cycles_done += res.cycles
            logger.info(f"ILS seed[{_i}]={_kind}: {res.summary()}")
            # Treat the ILS futility-stop note as a wall-handback signal.
            # ILSPolishResult has no structured early-stop field, so this detection
            # depends on the stable note text emitted by run_ils_polish.
            try:
                if any("cycles; stopped" in str(n) for n in (res.notes or [])):
                    self._maybe_arm_wall_handback(
                        f"ils_no_improve_stop(seed={_kind})")
            except Exception:
                pass
            if (res.improved and res.best_wns is not None
                    and (global_best_wns is None
                         or res.best_wns > global_best_wns)):
                global_best_wns = res.best_wns
                global_best_path = _path
                global_improved = True
        # Ship the global best: re-open it so finalize never ships a worse seed
        # (never-worse — recipe_best is the floor if nothing improved).
        try:
            await self.call_tool("vivado_run_tcl",
                                 {"command": f"open_checkpoint {{{global_best_path}}}"})
        except Exception:
            pass
        if global_improved:
            self.best_wns = global_best_wns
            self._best_valid_dcp = Path(global_best_path)
            self._best_valid_dcp_wns = global_best_wns
            # Keep the banked-size reference in sync with the
            # re-pointed banked best (emergency integrity gate).
            try:
                self._best_valid_mirror_size = (
                    Path(global_best_path).stat().st_size)
            except OSError:
                self._best_valid_mirror_size = None
            logger.info(f"ILS shipped {Path(global_best_path).name} "
                        f"wns={global_best_wns} (improved over recipe-best).")
        else:
            logger.info("ILS: no seed beat recipe-best; shipping recipe-best (never-worse).")
        # Release the polish reserve before entering the protected stages.
        # The release is one-way even if both stages subsequently skip themselves.
        self._release_polish_reserve("ils_polish_ladder_reached")
        # Run final aggressive fanout polish outside the futility-gated rotation
        # so it remains reachable on plateaus. It is budget-limited to cheap
        # designs, and acceptance requires clean hold timing with no regression.
        try:
            await self._lastmile_polish_after_ils(
                global_best_path, self.best_wns, deadline, wns_tcl)
        except Exception as e:
            logger.warning(f"lastmile-polish stage failed (ignored, best intact): {e!r}")
        # fanout polish runs on whatever best now stands (LASTMILE accept
        # updates _best_valid_dcp); it never places, so LastMile session
        # poisoning cannot affect it.
        _fp_path = str(self._best_valid_dcp) if self._best_valid_dcp else global_best_path
        try:
            await self._fanout_polish_after_ils(
                _fp_path, self.best_wns, deadline, wns_tcl)
        except Exception as e:
            logger.warning(f"fanout-polish stage failed (ignored, best intact): {e!r}")
        # Optional terminal re-placement runs after all cheaper polish because it
        # discards the current polish chain. It starts from the final banked best.
        _rg_path = str(self._best_valid_dcp) if self._best_valid_dcp else global_best_path
        try:
            await self._replace_gamble_after_polish(
                _rg_path, self.best_wns, deadline, wns_tcl)
        except Exception as e:
            logger.warning(f"replace-gamble stage failed (ignored, best intact): {e!r}")

    def _polish_gate_reserve_s(self, cfg) -> "tuple[float, str]":
        """Bound shim over the module-level pure function (refactor policy:
        keep patch targets module-global).  A caller without
        `_budget_deadline` — every stage stub in the suite — resolves to None
        and therefore keeps the reserve, which is the fail-safe direction."""
        return polish_gate_reserve_s(
            cfg, getattr(self, "_budget_deadline", None))

    async def _lastmile_polish_after_ils(self, best_path: str,
                                         best_wns: "Optional[float]",
                                         deadline: float, wns_tcl: str) -> None:
        """Apply last-mile optimization to the plateaued global best.

        A rejected result leaves the prior best unchanged. This stage must run
        before fanout polishing so its session state cannot affect a later
        placement operation.
        """
        from optimizer.ils_polish import (_measure, _measure_hold,
                                          lastmile_polish_accept)
        cfg = self._ils_polish_cfg
        if not getattr(cfg, "lastmile_polish_enabled", True):
            return
        if best_wns is None or not best_path:
            return
        if best_wns < cfg.lastmile_min_wns_ns:
            logger.info(f"lastmile-polish: skipped (wns {best_wns} below UG906 "
                        f"entry gate {cfg.lastmile_min_wns_ns}).")
            return
        # Cost: observed LASTMILE cycle from this run's ILS seeds if any,
        # else a conservative default (probe 607s; eval observed 144-636s).
        est = 700.0
        try:
            from optimizer.ils_polish import ILS_COMBOS, LASTMILE_PD
            _lm = [c[0] for c in ILS_COMBOS].index(LASTMILE_PD)
            _obs = (getattr(self, "_ils_observed_costs", None) or {}).get(_lm)
            if _obs:
                est = float(_obs)
        except Exception:
            pass
        remaining = deadline - time.time()
        _rsv, _rsv_tag = polish_gate_reserve_s(
            cfg, getattr(self, "_budget_deadline", None))
        need = est * 1.3 + _rsv
        if remaining < need:
            logger.info(f"lastmile-polish: insufficient budget (remaining "
                        f"{remaining:.0f}s < need {need:.0f}s = est "
                        f"{est:.0f}s x1.3 + {_rsv_tag}); skipped.")
            return

        def _to() -> float:
            return max(300.0, min(cfg.heavy_cmd_timeout_s, deadline - time.time()))
        out_dcp = str((self.run_dir or Path(".")) / "ils_lastmile_polish.dcp")
        for _cmd in (f"open_checkpoint {{{best_path}}}",
                     "phys_opt_design -clock_opt -retime -lut_opt",
                     "place_design -directive LastMile",
                     "phys_opt_design -directive Explore",
                     "route_design -directive Explore",
                     "phys_opt_design -directive Explore"):
            _r = await self.call_tool("vivado_run_tcl",
                                      {"command": _cmd, "timeout": _to()})
            if isinstance(_r, str) and ('"error"' in _r or "TCL ERROR:" in _r):
                logger.info(f"lastmile-polish: step failed ({_cmd.split()[0]}); "
                            f"keeping best (never-worse).")
                return
        w, ur = await _measure(self.call_tool, wns_tcl,
                               timeout_s=cfg.measure_cmd_timeout_s)
        if ur and ur > 0:
            # Last-mile optimization may leave a small number of nets unrouted.
            # Attempt one incremental completion pass; reject the candidate if
            # routing still fails or timing regresses.
            await self.call_tool("vivado_run_tcl",
                                 {"command": "route_design",
                                  "timeout": _to()})
            w, ur = await _measure(self.call_tool, wns_tcl,
                                   timeout_s=cfg.measure_cmd_timeout_s)
        whs = await _measure_hold(self.call_tool,
                                  timeout_s=cfg.measure_cmd_timeout_s)
        cells = None
        try:
            _cr = await self.call_tool("vivado_run_tcl", {
                "command": "llength [get_cells -quiet -hierarchical "
                           "-filter {IS_PRIMITIVE}]",
                "timeout": cfg.measure_cmd_timeout_s})
            # Reject tool error envelopes before parsing integers as cell counts.
            # When a golden count exists, an unknown count makes last-mile
            # acceptance fail closed, so raising preserves acceptance semantics.
            if _looks_like_tool_error(_cr):
                raise RuntimeError(
                    f"cell-count query returned error envelope: "
                    f"{str(_cr)[:120]}")
            _cm = re.search(r"(\d+)", str(_cr) or "")
            cells = int(_cm.group(1)) if _cm else None
        except Exception:
            pass
        ok, why = lastmile_polish_accept(new_wns=w, best_wns=best_wns,
                                         unrouted=ur, whs=whs,
                                         cell_count=cells, cfg=cfg)
        if ok:
            _wr = await self.call_tool(
                "vivado_run_tcl",
                {"command": f"write_checkpoint -force {{{out_dcp}}}",
                 "timeout": _to()})
            if isinstance(_wr, str) and ('"error"' in _wr or "TCL ERROR:" in _wr):
                logger.info("lastmile-polish: write_checkpoint failed; "
                            "keeping best (never-worse).")
                return
            self.best_wns = w
            self._best_valid_dcp = Path(out_dcp)
            self._best_valid_dcp_wns = w
            # Keep the banked-size reference in sync.
            try:
                self._best_valid_mirror_size = Path(out_dcp).stat().st_size
            except OSError:
                self._best_valid_mirror_size = None
            logger.info(f"lastmile-polish ACCEPT: {why}; shipped LASTMILE "
                        f"final polish.")
        else:
            logger.info(f"lastmile-polish: rejected ({why}); keeping best "
                        f"(never-worse).")
            # Wall-handback signal (b): the LASTMILE reject verdict — the plateaued
            # best could not be improved by the last dedicated polish.
            self._maybe_arm_wall_handback(f"lastmile_reject:{why[:120]}")

    async def _fanout_polish_after_ils(self, best_path: str,
                                       best_wns: Optional[float],
                                       deadline: float, wns_tcl: str) -> None:
        """Apply one aggressive fanout optimization pass to the final routed best.

        The pass runs only when a measured cold-start cycle is below the
        configured cost limit and sufficient budget remains. Adoption requires
        strict hold acceptance and no WNS regression; rejection preserves the
        prior best.
        """
        from optimizer.ils_polish import (_measure, _measure_hold,
                                          fanout_polish_accept)
        cfg = self._ils_polish_cfg
        if not getattr(cfg, "fanout_polish_enabled", True):
            return
        if best_wns is None or not best_path:
            return
        est, anchor_src = _fanout_polish_cost_basis(cfg)
        if est <= 0 or est > cfg.fanout_max_cycle_s:
            logger.info(f"fanout-polish: skipped ({anchor_src} cost anchor "
                        f"{est:.0f}s vs gate {cfg.fanout_max_cycle_s:.0f}s — "
                        f"fast designs only).")
            return
        remaining = deadline - time.time()
        _rsv, _rsv_tag = polish_gate_reserve_s(
            cfg, getattr(self, "_budget_deadline", None))
        need = est * 1.3 + _rsv
        if remaining < need:
            logger.info(f"fanout-polish: insufficient budget (remaining "
                        f"{remaining:.0f}s < need {need:.0f}s = est "
                        f"{est:.0f}s x1.3 + {_rsv_tag}); skipped.")
            return

        def _to() -> float:
            return max(300.0, min(cfg.heavy_cmd_timeout_s, deadline - time.time()))
        fanout_dcp = str((self.run_dir or Path(".")) / "ils_fanout_polish.dcp")
        await self.call_tool("vivado_run_tcl",
                             {"command": f"open_checkpoint {{{best_path}}}",
                              "timeout": cfg.measure_cmd_timeout_s})
        base_whs = await _measure_hold(self.call_tool,
                                       timeout_s=cfg.measure_cmd_timeout_s)
        _r = await self.call_tool(
            "vivado_run_tcl",
            {"command": "phys_opt_design -directive AggressiveFanoutOpt",
             "timeout": _to()})
        if isinstance(_r, str) and '"error"' in _r:
            logger.info("fanout-polish: phys_opt errored; keeping best (never-worse).")
            return
        w, ur = await _measure(self.call_tool, wns_tcl,
                               timeout_s=cfg.measure_cmd_timeout_s)
        if ur != 0:
            # replication can unroute affected nets — reroute once before measuring
            await self.call_tool("vivado_run_tcl",
                                 {"command": "route_design -directive Explore",
                                  "timeout": _to()})
            w, ur = await _measure(self.call_tool, wns_tcl,
                                   timeout_s=cfg.measure_cmd_timeout_s)
        whs = await _measure_hold(self.call_tool,
                                  timeout_s=cfg.measure_cmd_timeout_s)
        ok, why = fanout_polish_accept(new_wns=w, best_wns=best_wns, unrouted=ur,
                                       whs=whs, base_whs=base_whs, cfg=cfg)
        if ok:
            await self.call_tool(
                "vivado_run_tcl",
                {"command": f"write_checkpoint -force {{{fanout_dcp}}}",
                 "timeout": cfg.measure_cmd_timeout_s})
            self.best_wns = w
            self._best_valid_dcp = Path(fanout_dcp)
            self._best_valid_dcp_wns = w
            # Keep the banked-size reference in sync.
            try:
                self._best_valid_mirror_size = Path(fanout_dcp).stat().st_size
            except OSError:
                self._best_valid_mirror_size = None
            logger.info(f"fanout-polish ACCEPT: {why} (hold {base_whs}->{whs}); "
                        f"shipped AggressiveFanoutOpt polish.")
        else:
            logger.info(f"fanout-polish: rejected ({why}); keeping best (never-worse).")

    async def _replace_gamble_after_polish(self, best_path: str,
                                           best_wns: Optional[float],
                                           deadline: float,
                                           wns_tcl: str) -> None:
        """Try insured full re-placement draws from the banked best.

        Each draw varies the untried net-delay weight, then routes and measures
        the result. The banked DCP is never overwritten; adoption requires a
        routed, hold-clean result at least 0.15 ns better than the chain best
        and a verified write to a draw-specific file before the mirror is
        repointed.

        The stage fails closed when no full-cycle cost measurement or reserve
        headroom is available. It is disabled by default through
        `cfg.replace_gamble_enabled`.
        """
        cfg = self._ils_polish_cfg
        if not getattr(cfg, "replace_gamble_enabled", False):
            return
        from optimizer.replace_gamble import run_replace_gamble
        out_base = str((self.run_dir or Path(".")) / "ils_replace_gamble")
        res = await run_replace_gamble(
            self.call_tool, best_dcp_path=best_path, best_wns=best_wns,
            deadline_ts=deadline, wns_tcl=wns_tcl, cfg=cfg,
            log=lambda m: logger.info(m), out_dcp_base=out_base,
            observed_costs=getattr(self, "_ils_observed_costs", None))
        if res.adopted and res.best_path and res.best_wns is not None:
            self.best_wns = res.best_wns
            self._best_valid_dcp = Path(res.best_path)
            self._best_valid_dcp_wns = res.best_wns
            # Keep the banked-size reference in sync.
            try:
                self._best_valid_mirror_size = (
                    Path(res.best_path).stat().st_size)
            except OSError:
                self._best_valid_mirror_size = None
            logger.info(f"replace-gamble ADOPTED: shipped "
                        f"{Path(res.best_path).name} wns={res.best_wns} "
                        f"(insured re-place beat chain-best by >= "
                        f"{cfg.replace_gamble_adopt_margin_ns}ns).")
        # A fully routed, hold-clean gamble draw also enters final selection even
        # when it misses the in-stage adoption threshold. It was measured under
        # the same gates, so skipping re-verification avoids disturbing the live
        # session. Missing cell count fails open because replacement preserves cells.
        if (res.best_draw_path and res.best_draw_wns is not None
                and Path(res.best_draw_path).exists()
                and Path(res.best_draw_path).stat().st_size > 0):
            try:
                await self.register_final_candidate(
                    res.best_draw_path, res.best_draw_wns, "replace_gamble",
                    whs=res.best_draw_whs, verify=False)
            except Exception as e:
                logger.warning(
                    f"[mux] replace_gamble registration raised "
                    f"{type(e).__name__} (non-fatal): {e}")

    async def _deep_replace_sibling_after_polish(
        self, deadline: float, wns_tcl: str, *,
        stage_label: str = "tail",
        cost_basis_override: Optional[float] = None,
        basis_note: str = "",
    ) -> None:
        """Generate a full re-placement candidate from the pristine input for deep
        negative slack.

        The candidate is added to the insured selector without modifying router
        state, and the selector retains the result with the best measured WNS.
        The gate uses the router's existing `R1_FAILING_ENDPOINTS_MIN` and
        `R1_ROUTE_FIRST_WNS_NS` thresholds.

        At loop exit, the cost basis comes from an observed ILS cycle; before
        the main loop, it comes from the size model. Stage labels and basis
        notes affect logging only and must not change gate behavior. The stage
        is a strict no-op when its default-off flag is disabled.
        """
        # Emit reachability before checking enablement so diagnostics distinguish
        # a disabled stage from a stage that was never called.
        cfg = self._ils_polish_cfg
        _en = bool(getattr(cfg, "deep_replace_enabled", False))
        logger.info(f"deep-replace[{stage_label}]: stage REACHED "
                    f"(enabled={_en})")
        if not _en:
            return
        from optimizer.deep_replace_sibling import (
            DEEP_REPLACE_COST_MARGIN, VERDICT_ADOPTED, VERDICT_ERROR,
            deep_replace_should_run, run_deep_replace_sibling)
        from optimizer.ils_polish import (_measure, _measure_hold, _tool_ok,
                                          derive_cost_anchors)
        from optimizer.replace_gamble import replace_gamble_cost_basis
        try:
            from optimizer.recipe_router import (R1_FAILING_ENDPOINTS_MIN,
                                                 R1_ROUTE_FIRST_WNS_NS)
        except Exception:
            logger.warning("deep-replace: router constants unavailable; "
                           "skipping (fail closed).")
            return

        pristine = (str(self.input_dcp_path)
                    if self.input_dcp_path and self.input_dcp_path.exists()
                    else None)
        if cost_basis_override and cost_basis_override > 0:
            basis = float(cost_basis_override)
            logger.info(f"deep-replace[{stage_label}]: cost basis from "
                        f"{basis_note or 'caller override'} = {basis:.0f}s")
        else:
            basis = replace_gamble_cost_basis(
                cfg, getattr(self, "_ils_observed_costs", None))
            # If no heavy step has published a measured cost, estimate placement
            # and routing cost from the size model. Affordability still requires
            # 1.3 times the basis plus reserve, avoiding a skip caused only by
            # the absence of an earlier measurement.
            if basis <= 0:
                try:
                    from optimizer.deep_replace_sibling import predict_place_route_s
                    _est, _why = predict_place_route_s(
                        primitive_cells=getattr(self, "_input_cell_count", None))
                    if _est > 0:
                        basis = float(_est)
                        logger.info(
                            f"deep-replace[{stage_label}]: no measured anchor "
                            f"(first stage never ran) — falling back to the SAME size "
                            f"model the first stage trusts at t=0: {basis:.0f}s ({_why}). "
                            f"Affordability still gated below; without this the stage "
                            f"would be skipped for a missing measurement rather than "
                            f"for cost.")
                except Exception as e:
                    logger.warning(f"deep-replace[{stage_label}]: size-model fallback "
                                   f"raised {type(e).__name__} (non-fatal): {e!r}")
        _wns = self._phase1_wns_for_features()
        # Optional unbanded mode removes the benefit-prediction veto for deep
        # replacement. Affordability, pristine-input, and cost-anchor gates still
        # apply, and final insured comparison discards regressions.
        _unbanded = bool(getattr(self, "_deep_replace_unbanded", False))
        # The budget deadline already excludes the finalize reserve. Optional
        # no-double-reserve mode prevents the affordability gate from subtracting
        # it again; genuine infeasibility still fails closed.
        _no_double_reserve = (
            os.environ.get("FPL26_DEEP_REPLACE_NO_DOUBLE_RESERVE", "0")
            .strip().lower() in ("1", "true", "on", "yes"))
        # The optional B3 sibling is limited to the first stage in the mid band.
        # Both the affordability cap and live physics admission must pass;
        # admission failures reject the candidate. The deterministic recipe is
        # not repeated at the tail after its result has already been banked.
        _b3_on = (
            os.environ.get("FPL26_DEEP_REPLACE_B3", "0")
            .strip().lower() in ("1", "true", "on", "yes")
            and recipe_pass_band(self.initial_wns) == "mid"
            and stage_label == "first")
        # The band gate applies only to the first stage, where replacement consumes
        # loop budget. The tail spends leftover budget and remains unbanded.
        # An out-of-band first stage may still run through the affordability
        # override below, so inexpensive regeneration is not merely deferred.
        _band_first = (stage_label == "first"
                       and os.environ.get(
                           "FPL26_DEEP_REPLACE_FIRST_BANDED", "1"
                       ).strip().lower() in ("1", "true", "on", "yes"))
        # Optional first-stage size gating applies only when the input exceeds the
        # configured ILS cell limit. It relaxes the band and duplicate-reserve
        # gates for that class; an unknown count or limit grants no relaxation.
        _sg_first = False
        if stage_label == "first" and deep_first_sizegated_enabled():
            _sg_cells = getattr(self, "_input_cell_count", None)
            _sg_max = getattr(self._ils_polish_cfg, "max_cells", None)
            # Size-gated relaxation requires both an oversized input and at least
            # R1_FAILING_ENDPOINTS_MIN failing endpoints. The endpoint floor keeps
            # large but shallow, near-met designs from bypassing the timing band.
            # Unknown or zero values do not qualify.
            _sg_fail = self._phase1_failing_endpoints_for_features()
            if (_sg_cells and _sg_max and _sg_cells > _sg_max
                    and (_sg_fail or 0) >= R1_FAILING_ENDPOINTS_MIN):
                _sg_first = True
                logger.info(
                    f"deep-replace[first]: SIZE-GATED CLASS "
                    f"({_sg_cells:,} cells > ILS max_cells {_sg_max:,}, "
                    f"failing={_sg_fail:,} >= {R1_FAILING_ENDPOINTS_MIN:,}) — "
                    f"ILS never arms here and the design is broadly failing, so "
                    f"the DEEP-extreme |WNS| floor and the double-subtracted "
                    f"finalize reserve are waived for this stage (ispd16 "
                    f"measured +24.11 -> +115.37; boom_v1 unchanged +41.43).")
            else:
                logger.info(
                    f"deep-replace[first]: size-gate NOT applied "
                    f"(cells={_sg_cells} max_cells={_sg_max} "
                    f"failing={_sg_fail}) — needs BOTH cells > max_cells AND "
                    f"failing >= {R1_FAILING_ENDPOINTS_MIN:,}; behavior "
                    f"unchanged.")
        if _sg_first:
            _band_first = False
        # A band miss defers first-stage replacement only when its estimated cost
        # is genuinely unaffordable. The available budget is wall time outside
        # the reserved tail fraction; 1.3 times the cost basis provides shared
        # estimate headroom. Unknown wall time leaves the band decision unchanged.
        if _band_first:
            _wall = getattr(self, "max_wall_seconds", None)
            _first_budget = (float(_wall) * (1.0 - TAIL_RESERVE_WALL_CLAMP_FRAC)
                             if _wall else 0.0)
            _need = (basis or 0.0) * DEEP_REPLACE_COST_MARGIN
            if _first_budget > 0 and _need > 0 and _need <= _first_budget:
                _band_first = False
                logger.info(
                    f"deep-replace[first]: band gate OVERRIDDEN by affordability — "
                    f"stage needs {_need:.0f}s (basis {basis:.0f}s x"
                    f"{DEEP_REPLACE_COST_MARGIN}) which fits the "
                    f"{_first_budget:.0f}s left over the tail reserve, so FIRST costs "
                    f"the loop nothing it was budgeted. Deferring a cheap stage cost "
                    f"a measured 10.64 MHz even though the tail did run it.")
        if _band_first:
            logger.info("deep-replace[first]: BAND-GATED (opt-in) — FIRST "
                        "requires the DEEP-extreme band; a miss DEFERS the stage to "
                        "the tail rather than refusing it.")
        run, why = deep_replace_should_run(
            enabled=True,
            pristine_dcp=pristine,
            failing_endpoint_count=self._phase1_failing_endpoints_for_features(),
            wns_magnitude_ns=(abs(_wns) if _wns is not None else None),
            remaining_s=deadline - time.time(),
            cost_basis_s=basis,
            finalize_reserve_s=cfg.deep_replace_finalize_reserve_s,
            failing_endpoints_min=R1_FAILING_ENDPOINTS_MIN,
            wns_min_ns=R1_ROUTE_FIRST_WNS_NS,
            require_physics_band=((_band_first or (not _unbanded))
                                  and not _sg_first),
            # Callers pass a deadline that already excludes the finalize reserve.
            # Prevent a second subtraction when the optional correction is enabled;
            # size-gated first-stage candidates always receive the same treatment.
            reserve_already_in_deadline=(_no_double_reserve or _sg_first),
        )
        if not run:
            logger.info(f"deep-replace[{stage_label}]: skipped ({why})")
            return
        logger.info(f"deep-replace[{stage_label}]: {why}")
        out_dcp = str((self.run_dir or Path(".")) / "deep_replace_candidate.dcp")
        # Deep replacement temporarily uses the ILS-stage bypass for route and
        # unroute budget gates. The recipe has already unplaced the design, and
        # every step is capped by the deadline, so no routed state needs protection.
        # Save and restore the prior stage flag because tail execution may be nested.
        # B3 admission runs only after affordability passes and checks the live
        # session for a hard-macro-dominated near-floor case. Tool or parse errors
        # reject admission.
        _b3_admit = None
        if _b3_on:
            async def _b3_admit(wns_banked, _out=out_dcp):
                # Any setup error (e.g. clock_period None ->
                # TypeError) must refuse HERE with an admission-labeled
                # reason, not surface as a mislabeled B3-leg failure.
                try:
                    from optimizer.logic_floor import (
                        run_b1_admission_attestation)
                    v = await run_b1_admission_attestation(
                        self.call_tool,
                        clock_name="*fpl26contest*",
                        period_ns=float(self.clock_period),
                        wns_banked=wns_banked,
                        log=lambda m: logger.info(m),
                        recover_dcp=_out or None)
                except Exception as e:
                    return False, (f"B3 declined (physics admission): "
                                   f"attestor setup raised {e!r} "
                                   f"(fail closed)")
                return v.fire, (
                    ("B3 physics-admitted: " if v.fire
                     else "B3 declined (physics admission): ") + v.reason)
        _prev_ils_flag = self._in_ils_stage
        self._in_ils_stage = True
        try:
            res = await run_deep_replace_sibling(
                self.call_tool,
                pristine_dcp=pristine,
                chain_best_wns=self.best_wns,
                deadline_ts=deadline,
                wns_tcl=wns_tcl,
                out_dcp=out_dcp,
                log=lambda m: logger.info(m),
                measure=_measure,
                measure_hold=_measure_hold,
                tool_ok=_tool_ok,
                hold_slack_floor_ns=cfg.hold_slack_floor_ns,
                heavy_timeout_s=cfg.heavy_cmd_timeout_s,
                light_timeout_s=cfg.measure_cmd_timeout_s,
                adopt_margin_ns=cfg.deep_replace_adopt_margin_ns,
                # Funds the B2 (retiming-tail) gate, decided from the MEASURED
                # B1 elapsed time rather than the arm-time anchor.
                finalize_reserve_s=cfg.deep_replace_finalize_reserve_s,
                b3_enabled=_b3_on,
                b3_admission=_b3_admit,
            )
        finally:
            self._in_ils_stage = _prev_ils_flag
        # A server-side timeout can leave a command pending while the FPGA tool
        # continues running, causing subsequent calls to block during synchronization.
        # Restart only after an error with no candidate; recovery is best-effort.
        # The prefixed restart name is required for call_tool routing.
        try:
            if (getattr(res, "verdict", None) == VERDICT_ERROR
                    and not getattr(res, "candidate_path", None)):
                logger.warning(
                    f"deep-replace[{stage_label}]: stage ERRORED with no "
                    f"candidate ({str(getattr(res, 'reason', ''))[:160]}); "
                    f"restarting Vivado so a wedged session cannot cost the "
                    f"remaining wall.")
                _rs = await self.call_tool("vivado_restart_vivado", {})
                if _looks_like_tool_error(_rs):
                    logger.warning(
                        f"deep-replace[{stage_label}]: restart_vivado returned "
                        f"an error envelope ({str(_rs)[:120]}) — continuing; "
                        f"the banked best is unaffected either way.")
                else:
                    logger.info(
                        f"deep-replace[{stage_label}]: Vivado restarted; the "
                        f"loop continues on a clean session.")
        except Exception as e:
            logger.warning(f"deep-replace[{stage_label}]: post-error restart "
                           f"raised (ignored): {e!r}")
        # Publish the measured place-and-route cycle for downstream cost gates.
        # Measurements from the current design and machine take precedence over
        # derived estimates. Never lower an existing anchor, which would make
        # affordability checks less conservative.
        try:
            _measured_cycle_s = float(getattr(res, "place_s", 0.0) or 0.0) + \
                                float(getattr(res, "route_s", 0.0) or 0.0)
            if _measured_cycle_s > 0:
                _prev_anchor = float(getattr(cfg, "expected_heavy_cycle_s", 0.0) or 0.0)
                if _measured_cycle_s > _prev_anchor:
                    cfg.expected_heavy_cycle_s = _measured_cycle_s
                    logger.info(
                        f"deep-replace[{stage_label}]: published measured "
                        f"place+route anchor {_measured_cycle_s:.0f}s "
                        f"(place={float(getattr(res, 'place_s', 0.0)):.0f}s + "
                        f"route={float(getattr(res, 'route_s', 0.0)):.0f}s); "
                        f"previous anchor {_prev_anchor:.0f}s. Downstream "
                        f"cost gates now size from a measurement of THIS "
                        f"design on THIS box.")
                else:
                    logger.info(
                        f"deep-replace[{stage_label}]: measured cycle "
                        f"{_measured_cycle_s:.0f}s <= existing anchor "
                        f"{_prev_anchor:.0f}s; anchor left unchanged "
                        f"(raise-only).")
        except Exception as e:                                # pragma: no cover
            logger.warning(f"deep-replace[{stage_label}]: could not publish "
                           f"cost anchor ({type(e).__name__}: {e}) — "
                           f"non-fatal, downstream gates keep their basis.")
        # Preserve the adopted deterministic floor so finalization can attest the
        # attempt outcome. Later promotion is safe because the final mux cannot
        # select a result below this floor.
        if (getattr(res, "stage_banked", "") == "small_floor"
                and getattr(res, "post_wns", None) is not None):
            self._b3_floor_wns = float(res.post_wns)
            logger.info(f"deep-replace[{stage_label}]: B3 floor recorded "
                        f"({res.post_wns:.3f}) for the floor-exit sentinel.")
            # Logic-floor attestation queries the live routed state before later
            # stages can modify it. A successful match finalizes the floor and skips
            # downstream optimization; any failed guard proceeds normally.
            _lf_on = (os.environ.get("FPL26_LOGIC_FLOOR_EXIT", "0").strip()
                      .lower() in ("1", "true", "on", "yes")
                      and stage_label == "first")
            if _lf_on:
                try:
                    from optimizer.logic_floor import (
                        run_logic_floor_attestation)
                    _lfv = await run_logic_floor_attestation(
                        self.call_tool,
                        clock_name="*fpl26contest*",
                        period_ns=float(self.clock_period),
                        wns_b1=getattr(res, "b1_wns", None),
                        wns_b3=float(res.post_wns),
                        log=lambda m: logger.info(m),
                        recover_dcp=str(getattr(res, "candidate_path", "")
                                        or "") or None)
                    if _lfv.fire:
                        self._logic_floor_terminate = _lfv.reason
                except Exception as e:
                    logger.warning(f"[logic-floor] attestation call raised "
                                   f"{e!r} — NO-FIRE (fail-safe).")
        # The sibling stage already validates routing and hold timing under the same
        # final-candidate gates, so verification is skipped to avoid reopening the
        # session and disturbing shipping state. A missing cell count fails open.
        if (res.candidate_path and res.post_wns is not None
                and Path(res.candidate_path).exists()
                and Path(res.candidate_path).stat().st_size > 0):
            try:
                res.registered = await self.register_final_candidate(
                    res.candidate_path, res.post_wns, "deep_replace",
                    whs=res.post_whs, verify=False)
            except Exception as e:
                logger.warning(
                    f"[mux] deep_replace registration raised "
                    f"{type(e).__name__} (non-fatal): {e}")

            # Propagate an adopted checkpoint into the pipeline best so downstream
            # stages continue from it rather than reopening the baseline.
            # Recheck strict improvement because another stage may have banked a
            # better result since this stage captured its comparison point.
            if (res.verdict == VERDICT_ADOPTED and res.candidate_path
                    and res.post_wns is not None
                    and (self.best_wns is None
                         or self.best_wns == float("-inf")
                         or res.post_wns > self.best_wns)):
                _prev_best = self.best_wns
                self.best_wns = res.post_wns
                self._best_valid_dcp = Path(res.candidate_path)
                self._best_valid_dcp_wns = res.post_wns
                # Convention: keep the banked-size reference in sync.
                try:
                    self._best_valid_mirror_size = (
                        Path(res.candidate_path).stat().st_size)
                except OSError:
                    self._best_valid_mirror_size = None
                logger.info(
                    f"deep-replace[{stage_label}]: promoted to pipeline best "
                    f"{_prev_best} -> {res.post_wns} ({res.stage_banked}); "
                    f"downstream stages now continue FROM this state.")

    async def _run_bare_reroute_polish(self) -> None:
        """Iteratively reroute the banked best when the main polish ladder never
        arms.

        The stage requires auto-banking, a validated routed checkpoint, and
        enough remaining wall time after the measured checkpoint-open cost.
        Affordability uses the preserving cost basis because the on-disk bank
        remains intact if an in-memory reroute overruns.

        Each iteration uses an undirected `route_design` pass. From the second
        iteration onward, the previous pass supplies the route-cost estimate.
        Iteration stops on insufficient WNS gain, insufficient wall time, the
        configured cap, or a route error.

        Normal dispatch preserves routed-state tracking, phantom-best
        protection, and eager auto-banking. Improved routed results are banked;
        worse results remain in memory only, and finalization ships the on-disk
        best.

        All iterations form one polish stage and may consume the reserved
        window. Completion releases the reserve once; a first-pass route error
        leaves release to finalization.

        Disable with `--no-bare-reroute-polish` or
        `FPL26_NO_BARE_REROUTE_POLISH=1`. Minimum gain and iteration count are
        controlled by the matching command-line and environment settings.
        """
        if not self._bare_reroute_polish_enabled:
            logger.info("bare-reroute polish: disabled (kill switch); skipped.")
            return
        if self._ils_preempt_requested:
            # ILS ran — its __ROUTE_ONLY__ combo already covers this lever.
            return
        banked = self._best_valid_dcp
        if (banked is None or not Path(banked).exists()
                or Path(banked).stat().st_size <= 0):
            logger.info("bare-reroute polish: no routed banked best; skipped.")
            return
        if not self._auto_bank_enabled:
            # Without auto-bank the re-route would mutate state with no
            # measurement/banking ride-along — pure wall waste.
            logger.info("bare-reroute polish: auto-bank disabled; skipped.")
            return
        # ---- Wall-fit gate: PRESERVING cost basis (docstring (c)) ----
        from optimizer.route_gate import assess_preserving_reroute
        # Reopen the banked best before measuring further gains because the live
        # session may have drifted. Charge the observed checkpoint-open cost from
        # this run; when unavailable, use zero and rely on the assessment margin.
        open_cost_s = 0.0
        for _tc in self.tool_call_details:
            if not isinstance(_tc, dict) or _tc.get("error"):
                continue
            if (_tc.get("tool_name") == "vivado_open_checkpoint"
                    or "open_checkpoint" in str(_tc.get("cmd_head", ""))):
                try:
                    open_cost_s = max(open_cost_s,
                                      float(_tc.get("elapsed_time") or 0.0))
                except (TypeError, ValueError):
                    pass
        remaining = self._budget_remaining()
        assessment = assess_preserving_reroute(
            remaining - open_cost_s,
            self.tool_call_details,
        )
        if not assessment.feasible:
            # Honest skip — includes the no-route-sample case (predicted
            # inf: the run's history carries no route_design sample to
            # size the op with — no data, no gamble).
            logger.info(
                f"bare-reroute polish: skipped (wall unfit after "
                f"{open_cost_s:.0f}s re-open charge): {assessment.reason}")
            return
        min_gain_ns = float(getattr(self, "_bare_reroute_min_gain_ns",
                                    BARE_REROUTE_MIN_GAIN_NS_DEFAULT))
        max_iters = int(getattr(self, "_bare_reroute_max_iters",
                                BARE_REROUTE_MAX_ITERS_DEFAULT))
        logger.info(
            f"bare-reroute polish: firing (measured +0.094 ns "
            f"deterministically; undirected beats directed; "
            f"re-roll compounds while WNS is deep) — "
            f"predicted route {assessment.predicted_reroute_s:.0f}s + "
            f"re-open {open_cost_s:.0f}s of {remaining:.0f}s remaining; "
            f"iterating while gain >= {min_gain_ns:.3f} ns, wall fits, "
            f"iters < {max_iters}.")
        # Use the dedicated checkpoint opener so a successful transition restores
        # the routed-state tracker and the following route is classified as preserving.
        _r = await self.call_tool(
            "vivado_open_checkpoint",
            {"dcp_path": str(Path(banked).resolve())})
        if _looks_like_tool_error(_r):
            logger.info("bare-reroute polish: re-open of banked best failed; "
                        "keeping best (never-worse).")
            return
        # Deep negative-slack states use the adaptive measured-gain controller;
        # shallow states use the plain reroute loop. Controller errors reopen the
        # banked best and fall back to the plain loop.
        if (self._tail_controller_enabled
                and self.best_wns is not None
                and self.best_wns != float("-inf")
                and self.best_wns <= self._tail_ctrl_deep_wns_ns):
            try:
                await self._run_tail_controller(min_gain_ns=min_gain_ns)
                return
            except Exception as e:
                logger.warning(
                    f"[tail-ctrl] controller failed ({e!r}); FAIL-CLOSED "
                    f"to plain bare-reroute loop (banked best intact).")
                try:
                    _r = await self.call_tool(
                        "vivado_open_checkpoint",
                        {"dcp_path": str(Path(banked).resolve())})
                    if _looks_like_tool_error(_r):
                        logger.info(
                            "bare-reroute polish: fail-closed re-open "
                            "failed; keeping best (never-worse).")
                        return
                except Exception:
                    return
        # Bare reroute iterations allow unrestricted rip-up and share one polish-reserve
        # window. The reserve is released once after the entire loop completes.
        iters_done = 0
        stop_reason = "max_iters"
        observed_cost_s = 0.0  # measured cost of the previous iteration
        while True:
            if iters_done > 0:
                # Recheck whether the next pass fits using the previous bare-
                # route cost. Use preserve_factor=1.0 because the observation
                # is already a preserving reroute; discounting it again would
                # underpredict runtime. No checkpoint-open cost applies while
                # the design remains open.
                remaining = self._budget_remaining()
                assessment = assess_preserving_reroute(
                    remaining,
                    [{"tool_name": "vivado_run_tcl",
                      "cmd_head": "route_design",
                      "elapsed_time": observed_cost_s}],
                    preserve_factor=1.0,
                )
                if not assessment.feasible:
                    stop_reason = "wall_unfit"
                    logger.info(
                        f"bare-reroute polish: stop (wall_unfit) before "
                        f"iteration {iters_done + 1}: {assessment.reason}")
                    break
            # Allow roughly twice the predicted reroute time for the server-
            # side timeout. The outer deadline still bounds wall time, and the
            # banked checkpoint protects the best result if this pass overruns.
            _route_to = max(600.0, assessment.predicted_reroute_s * 2.0)
            if remaining != float("inf"):
                _route_to = min(_route_to, max(600.0, remaining))
            _pre_wns = self.best_wns
            _hist_mark = len(self.tool_call_details)
            _r = await self.call_tool(
                "vivado_run_tcl",
                {"command": "route_design", "timeout": _route_to})
            if isinstance(_r, str) and ('"error"' in _r or "TCL ERROR:" in _r):
                if iters_done == 0:
                    # Single-pass semantics: a first-pass route failure
                    # returns WITHOUT releasing the reserve (finalize
                    # backstops).
                    logger.info(
                        "bare-reroute polish: route_design failed/skipped; "
                        "banked best intact (never-worse).")
                    return
                stop_reason = "route_error"
                logger.info(
                    f"bare-reroute polish: stop (route_error) on iteration "
                    f"{iters_done + 1}; banked best intact (never-worse).")
                break
            iters_done += 1
            # Derive pass gain from the banked best: call_tool updates it only for an
            # improved, fully routed result. Neutral, worse, or rejected passes
            # therefore report zero gain and stop honestly.
            _gain_ns = (
                self.best_wns - _pre_wns
                if (_pre_wns is not None and _pre_wns != float("-inf")
                    and self.best_wns is not None
                    and self.best_wns != float("-inf"))
                else 0.0)
            # Observed cost of THIS pass -> next iteration's cost basis
            # (call_tool appended the route entry at _hist_mark; the
            # auto-bank measurement entries that follow don't match).
            _obs = 0.0
            for _tc in self.tool_call_details[_hist_mark:]:
                if (isinstance(_tc, dict) and not _tc.get("error")
                        and "route_design" in str(_tc.get("cmd_head", ""))):
                    try:
                        _obs = max(_obs, float(_tc.get("elapsed_time") or 0.0))
                    except (TypeError, ValueError):
                        pass
            # If the measured runtime is missing or zero, retain the prediction used
            # for this pass so the next wall-fit check cannot treat it as free.
            observed_cost_s = _obs if _obs > 0.0 else assessment.predicted_reroute_s
            if _gain_ns < min_gain_ns:
                stop_reason = "gain_below_min"
                logger.info(
                    f"bare-reroute polish: stop (gain_below_min) after "
                    f"iteration {iters_done}: gain {_gain_ns:+.3f} ns < "
                    f"min {min_gain_ns:.3f} ns (plateau-decay shape).")
                break
            if iters_done >= max_iters:
                stop_reason = "max_iters"
                logger.info(
                    f"bare-reroute polish: stop (max_iters) after iteration "
                    f"{iters_done}: gain {_gain_ns:+.3f} ns still >= min "
                    f"{min_gain_ns:.3f} ns but runaway guard reached.")
                break
            logger.info(
                f"bare-reroute polish: iteration {iters_done} gained "
                f"{_gain_ns:+.3f} ns >= min {min_gain_ns:.3f} ns — "
                f"re-rolling (observed pass cost {observed_cost_s:.0f}s "
                f"feeds the next wall-fit).")
        # Each pass is measured and conditionally banked by call_tool. Release the
        # shared polish reserve exactly once, regardless of the loop's stop reason.
        self._release_polish_reserve("bare_reroute_polish_done")
        logger.info(
            f"bare-reroute polish: complete after {iters_done} iteration(s), "
            f"stop={stop_reason} (best_wns={self.best_wns}); "
            f"auto-bank decided keep/no-keep (never-worse).")
        # Enroll the plain reroute lineage in the final-candidate mux so both adaptive
        # and non-adaptive tail paths participate in final selection.
        await self._register_tail_final_candidate("bare_reroute")

    async def _run_tail_controller(self, min_gain_ns: float) -> None:
        """Select affordable tail moves by measured WNS gain per second.

        This controller runs only after the bare-reroute entry gates pass and
        the banked best is reopened in the current session. It chooses the
        highest-rate eligible move, retires moves whose latest gain is below
        the minimum, and re-enables other moves after any accepted state
        change.

        The loop ends at the wall-time floor, when no move remains eligible, or
        at the global or four-executions-per-move caps. Route moves use normal
        auto-banking. Physical-optimization moves suppress that hook and bank
        only when WNS improves, routing is complete, and `hold_accept()` passes
        its initial-hold-relative floor.

        Rejected, failed, or non-improving moves leave the mirror untouched and
        force the banked best to reopen before the next move. An overrun
        therefore falls back to the on-disk mirror.

        The default-off `_tail_ctrl_m1_echo` option follows each adopted
        non-route move with one affordable bare-route move. An echo counts
        toward move limits and updates the route move's cost, gain, and
        retirement state exactly like a normally selected route move.
        """
        from optimizer.route_gate import assess_preserving_reroute
        from optimizer.tail_controller import (
            TAIL_CTRL_MAX_EXECS_PER_MOVE, TAIL_MENU, hold_accept,
            move_by_key, new_states, pick_next, predict_move_cost_s,
            record_result)
        from optimizer.ils_polish import _measure_hold

        _cfg = getattr(self, "_ils_polish_cfg", None)
        _measure_to = float(getattr(_cfg, "measure_cmd_timeout_s", 120.0)
                            if _cfg is not None else 120.0)
        states = new_states()
        max_moves = int(self._tail_ctrl_max_moves)
        moves_done = 0
        stop_reason = "max_moves"
        pending_reopen = False
        entry_wns = self.best_wns
        logger.info(
            f"[tail-ctrl] ARMED: entry wns={entry_wns:.3f} <= gate "
            f"{self._tail_ctrl_deep_wns_ns:.3f}; menu="
            f"{[m.key for m in TAIL_MENU]}; max_moves={max_moves}; "
            f"min_gain={min_gain_ns:.3f} ns; harvest-to-wall.")
        while moves_done < max_moves:
            remaining = self._budget_remaining()
            base_assess = assess_preserving_reroute(
                remaining, self.tool_call_details)
            route_pred = base_assess.predicted_reroute_s
            reopen_extra = (self._observed_open_cost_s()
                            if pending_reopen else 0.0)
            afford: dict = {}
            preds: dict = {}
            for m in TAIL_MENU:
                pred = predict_move_cost_s(m, states[m.key], route_pred)
                preds[m.key] = pred
                # Same margin stack as the preserving assessment: x1.3
                # cost slack + 30s banking + 120s safety, plus the
                # banked-best re-open when the session drifted.
                need = pred * 1.3 + 30.0 + 120.0 + reopen_extra
                afford[m.key] = (pred != float("inf")
                                 and remaining != float("inf")
                                 and remaining >= need) or (
                    pred != float("inf") and remaining == float("inf"))
            pick = pick_next(states, afford)
            if pick is None:
                _all_dead = all(
                    st.retired or st.executed >= TAIL_CTRL_MAX_EXECS_PER_MOVE
                    for st in states.values())
                stop_reason = ("plateau_all_retired" if _all_dead
                               else "wall_unfit")
                break
            move = move_by_key(pick)
            if pending_reopen:
                _banked = self._best_valid_dcp
                if _banked is None or not Path(_banked).exists():
                    stop_reason = "banked_best_missing"
                    break
                _r = await self.call_tool(
                    "vivado_open_checkpoint",
                    {"dcp_path": str(Path(_banked).resolve())})
                if _looks_like_tool_error(_r):
                    stop_reason = "reopen_failed"
                    break
                pending_reopen = False
            pre_wns = self.best_wns
            _hist_mark = len(self.tool_call_details)
            _move_to = max(600.0, preds[pick] * 2.0)
            if remaining != float("inf"):
                _move_to = min(_move_to, max(600.0, remaining))
            errored = False
            banked = False
            gain = 0.0
            whs = None
            verdict = "REJECTED_NO_GAIN"
            if move.rides_autobank:
                _r = await self.call_tool(
                    "vivado_run_tcl",
                    {"command": move.cmds[0], "timeout": _move_to})
                if isinstance(_r, str) and ('"error"' in _r
                                            or "TCL ERROR:" in _r):
                    errored = True
                    verdict = "ERROR"
                else:
                    gain = (
                        self.best_wns - pre_wns
                        if (pre_wns is not None
                            and pre_wns != float("-inf")
                            and self.best_wns is not None
                            and self.best_wns != float("-inf"))
                        else 0.0)
                    banked = gain > 0.0
                    verdict = "ADOPTED" if banked else "REJECTED_NO_GAIN"
            else:
                base_whs = await _measure_hold(self.call_tool,
                                               timeout_s=_measure_to)
                self._tail_ctrl_suppress_autobank = True
                try:
                    for _cmd in move.cmds:
                        _r = await self.call_tool(
                            "vivado_run_tcl",
                            {"command": _cmd, "timeout": _move_to})
                        if isinstance(_r, str) and ('"error"' in _r
                                                    or "TCL ERROR:" in _r):
                            errored = True
                            verdict = "ERROR"
                            break
                finally:
                    self._tail_ctrl_suppress_autobank = False
                if not errored:
                    try:
                        w = await asyncio.wait_for(
                            self.get_wns_for_target_clock(
                                self._call_vivado_tool),
                            timeout=600.0)
                    except Exception:
                        w = None
                    if (w is not None and pre_wns is not None
                            and pre_wns != float("-inf") and w > pre_wns):
                        if not await self._routed_ok_for_best():
                            verdict = "REJECTED_UNROUTED"
                        else:
                            whs = await _measure_hold(self.call_tool,
                                                      timeout_s=_measure_to)
                            if hold_accept(whs, base_whs):
                                old_best = self.best_wns
                                self.best_wns = w
                                self.last_improvement_iter = self.iteration
                                self.last_improvement_time = time.time()
                                self.regression_warning_sent = False
                                self._pending_best_mirror = True
                                self._pending_best_mirror_epoch = (
                                    self._mutation_epoch)
                                await self._mirror_best_valid_now(eager=True)
                                banked = True
                                gain = w - (old_best if old_best is not None
                                            else w)
                                verdict = "ADOPTED"
                            else:
                                verdict = "REJECTED_HOLD"
                    else:
                        verdict = "REJECTED_NO_GAIN"
            # Observed cost of THIS move from the history slice (sum of
            # the move's own heavy ops; measurement entries don't match).
            _obs = 0.0
            for _tc in self.tool_call_details[_hist_mark:]:
                if (isinstance(_tc, dict) and not _tc.get("error")
                        and any(str(_tc.get("cmd_head", "")).startswith(
                            _c.split()[0]) for _c in move.cmds)):
                    try:
                        _obs += float(_tc.get("elapsed_time") or 0.0)
                    except (TypeError, ValueError):
                        pass
            cost_s = _obs if _obs > 0.0 else preds[pick]
            moves_done += 1
            record_result(states, pick, gain if not errored else 0.0,
                          cost_s, min_gain_ns, errored=errored)
            if not banked:
                pending_reopen = True
            logger.info(
                f"[tail-ctrl] move={pick} exec#{states[pick].executed} "
                f"verdict={verdict} gain={gain:+.3f}ns cost={cost_s:.0f}s "
                f"wns={self.best_wns} whs={whs} "
                f"remaining={self._budget_remaining():.0f}s")
            # Optionally probes one bare reroute after an accepted non-route move,
            # starting from the newly banked state. Treat the probe as a normal
            # route move: it uses route cost prediction and history accounting,
            # consumes a move slot, and requires its own affordability check.
            if self._tail_ctrl_m1_echo and banked and pick != "m1_route":
                _m1 = move_by_key("m1_route")
                _m1_st = states["m1_route"]
                _rem = self._budget_remaining()
                _echo_pred = predict_move_cost_s(
                    _m1, _m1_st,
                    assess_preserving_reroute(
                        _rem, self.tool_call_details).predicted_reroute_s)
                _echo_need = _echo_pred * 1.3 + 30.0 + 120.0
                _echo_ok = (_echo_pred != float("inf")
                            and (_rem == float("inf") or _rem >= _echo_need))
                if moves_done >= max_moves:
                    logger.info(
                        f"[tail-ctrl] echo after {pick} SKIPPED "
                        f"(max_moves cap {max_moves} reached).")
                elif _m1_st.executed >= TAIL_CTRL_MAX_EXECS_PER_MOVE:
                    logger.info(
                        f"[tail-ctrl] echo after {pick} SKIPPED (m1_route "
                        f"exec cap {TAIL_CTRL_MAX_EXECS_PER_MOVE} reached).")
                elif not _echo_ok:
                    logger.info(
                        f"[tail-ctrl] echo after {pick} SKIPPED "
                        f"(unaffordable: need {_echo_need:.0f}s vs "
                        f"remaining {_rem:.0f}s).")
                else:
                    _e_pre = self.best_wns
                    _e_mark = len(self.tool_call_details)
                    _e_to = max(600.0, _echo_pred * 2.0)
                    if _rem != float("inf"):
                        _e_to = min(_e_to, max(600.0, _rem))
                    _e_errored = False
                    _e_banked = False
                    _e_gain = 0.0
                    _r = await self.call_tool(
                        "vivado_run_tcl",
                        {"command": _m1.cmds[0], "timeout": _e_to})
                    if isinstance(_r, str) and ('"error"' in _r
                                                or "TCL ERROR:" in _r):
                        _e_errored = True
                        _e_verdict = "ERROR"
                    else:
                        _e_gain = (
                            self.best_wns - _e_pre
                            if (_e_pre is not None
                                and _e_pre != float("-inf")
                                and self.best_wns is not None
                                and self.best_wns != float("-inf"))
                            else 0.0)
                        _e_banked = _e_gain > 0.0
                        _e_verdict = ("ADOPTED" if _e_banked
                                      else "REJECTED_NO_GAIN")
                    _e_obs = 0.0
                    for _tc in self.tool_call_details[_e_mark:]:
                        if (isinstance(_tc, dict) and not _tc.get("error")
                                and any(str(_tc.get("cmd_head", "")
                                            ).startswith(_c.split()[0])
                                        for _c in _m1.cmds)):
                            try:
                                _e_obs += float(
                                    _tc.get("elapsed_time") or 0.0)
                            except (TypeError, ValueError):
                                pass
                    _e_cost = _e_obs if _e_obs > 0.0 else _echo_pred
                    moves_done += 1
                    # Attribute the echo to the route move so rate control observes it.
                    # Below-threshold gain retires the move as normal plateau evidence.
                    record_result(states, "m1_route",
                                  _e_gain if not _e_errored else 0.0,
                                  _e_cost, min_gain_ns,
                                  errored=_e_errored)
                    if not _e_banked:
                        pending_reopen = True
                    logger.info(
                        f"[tail-ctrl] echo after {pick} move=m1_route "
                        f"exec#{_m1_st.executed} verdict={_e_verdict} "
                        f"gain={_e_gain:+.3f}ns cost={_e_cost:.0f}s "
                        f"wns={self.best_wns} whs=None "
                        f"remaining={self._budget_remaining():.0f}s")
        self._release_polish_reserve("tail_controller_done")
        logger.info(
            f"[tail-ctrl] COMPLETE moves={moves_done} stop={stop_reason} "
            f"wns {entry_wns} -> {self.best_wns} (banked mirror is disk "
            f"truth; never-worse).")
        # INSURED-COMPARE wiring (RESERVE/controller -> final-candidate
        # MUX): enroll the harvested best_valid as a FINAL candidate.
        await self._register_tail_final_candidate("tail_controller")

    async def _exit_with_ils_polish(self, output_dcp: Path, *,
                                    max_iterations_reached: bool = False) -> None:
        """Run the shared loop-exit polish tail and finalize the output.

        This must remain the single tail for every loop-exit path. It arms ILS
        when timing remains unmet, budget remains, and design size is viable;
        an ILS stage already armed by stagnation also runs here. Finalization
        and summary reporting occur only after the keep-best polish step.
        """
        # A logic-floor attestation terminates optimization when later physical
        # stages cannot improve the limiting paths. It requires a bound within
        # 10 MHz across 32 paths, a 45 ps/hop net floor, dominant logic and
        # hard-macro delay, and agreement between two solves.
        if getattr(self, "_logic_floor_terminate", None):
            logger.info(
                f"[logic-floor] exit tail: skipping fixpoint/ILS/polish "
                f"stages ({self._logic_floor_terminate}); finalizing the "
                f"attested floor now.")
            await self._finalize_logic_floor(output_dcp, max_iterations_reached)
            return
        # Run the optional default-directive fixpoint only after the LLM loop.
        # This ordering prevents it from pre-empting a transform the loop would
        # otherwise select while still improving the baseline for exit stages.
        await self._physopt_default_fixpoint()

        # ILS-polish LOOP-EXIT trigger (generalizable): catches early LLM/loop
        # exits the mid-loop stagnation preempt misses.
        if (not self._ils_preempt_requested and self._ils_polish_cfg.enabled
                and self.best_wns > float("-inf")):
            from optimizer.ils_polish import should_trigger_at_exit as _ils_exit
            if self._design_cells is None:
                try:
                    _cr = await self.call_tool("vivado_run_tcl", {
                        "command": "llength [get_cells -hierarchical "
                                   "-filter {IS_PRIMITIVE==1}]"})
                    _cm = re.search(r"(\d+)", _cr or "")
                    self._design_cells = int(_cm.group(1)) if _cm else -1
                except Exception:
                    self._design_cells = -1
            _et, _ew = _ils_exit(cells=self._design_cells,
                                 remaining_s=self._budget_remaining(),
                                 best_wns=self.best_wns, cfg=self._ils_polish_cfg)
            if _et:
                self._ils_preempt_requested = True
                logger.info(f"ILS-polish loop-exit trigger ({_ew}); using "
                            f"remaining {self._budget_remaining():.0f}s.")
            else:
                logger.info(f"ILS-polish loop-exit trigger NOT armed ({_ew}).")
        # ILS-polish stage: runs if the loop preempted (stagnation) OR the
        # loop-exit trigger fired. Exception-safe — failure leaves best intact.
        if self._ils_preempt_requested:
            try:
                await self._run_ils_polish_stage()
            except Exception as e:
                logger.warning(f"ILS-polish stage failed (ignored): {e!r}")
        else:
            # Give size-gated designs one bare reroute when the ILS route-only
            # path is unavailable. The method requires a routed banked best,
            # automatic banking, and sufficient predicted wall time.
            # Failure is ignored and leaves the banked best intact.
            try:
                await self._run_bare_reroute_polish()
            except Exception as e:
                logger.warning(f"bare-reroute polish failed (ignored, "
                               f"banked best intact): {e!r}")
        # Evaluate deep replacement from the common exit tail so it remains
        # reachable when size gating disables the ILS polish stage.
        # The method is a strict no-op when its feature flag is disabled.
        try:
            await self._deep_replace_sibling_after_polish(
                self._budget_deadline
                if self._budget_deadline is not None else time.time(),
                self._wns_tcl_for_stages())
        except Exception as e:
            logger.warning(f"deep-replace stage failed (ignored, banked best "
                           f"intact): {e!r}")
        # Run the optional recipe pass after all polish stages, when remaining
        # wall time is otherwise unallocated, but before final candidate selection.
        # It targets deep negative slack (|WNS| >= 8 ns). Failures are ignored
        # so finalization can still select the best banked candidate.
        try:
            await self._maybe_run_recipe_pass_postloop()
        except Exception as e:
            logger.warning(f"RECIPE-PASS[postloop]: skipped reason=exception "
                           f"{type(e).__name__}: {e!r} (non-fatal)")
        # Run the optional mid-band route rung after the deep-band recipe slot
        # and before wall-time handback and final candidate selection.
        # Its 1,350 s requirement reflects the predicted tool cost and may claim
        # time otherwise returned; failures are ignored so finalization proceeds.
        try:
            await self._maybe_run_midband_route_rung_postloop()
        except Exception as e:
            logger.warning(f"ROUTE-RUNG[postloop]: skipped reason=exception "
                           f"{type(e).__name__}: {e!r} (non-fatal)")
        # Checked AFTER the polish stages, BEFORE finalize — the
        # saturation signal must SURVIVE one polish pass (locked contract:
        # polish stages run once and never clear/reset _exit_early_reason).
        if self._exit_early_reason is not None:
            logger.info(
                f"wall-handback: exit-early reason "
                f"'{self._exit_early_reason}' survived the polish stages; "
                + ("finalizing now returns the remaining wall to the "
                   "wrapper." if self._wall_handback_enabled
                   else "kill switch OFF — observational only.")
            )
        await self._finalize_output_dcp(output_dcp)
        self.end_time = time.time()
        # Reporting only — _print_optimization_summary never raises (a
        # KeyError here once discarded a completed artifact; see its
        # docstring).
        self._print_optimization_summary(
            max_iterations_reached=max_iterations_reached)
