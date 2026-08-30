"""SUPPORT — the shipped FPL26_* configuration

Environment-driven configuration resolution.

Each function reads one FPL26_* variable and returns the effective value,
falling back to the shipped default. They are pure -- no optimizer state and
no I/O beyond os.environ -- so the ship configuration can be read, reasoned
about and tested without instantiating the agent.

Named for the stage it implements: the slide, the video and the paper all
call it by this name, so a reader can move between them without a glossary.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path

# Log through the orchestrator's logger, not this module's own: the extracted
# code still belongs to the same run, and a module-named logger would change
# every line it emits in the shipped log.
logger = logging.getLogger("dcp_optimizer")


POLISH_RESERVE_S_DEFAULT = 500.0


BARE_REROUTE_MIN_GAIN_NS_DEFAULT = 0.020


BARE_REROUTE_MAX_ITERS_DEFAULT = 4


TAIL_CTRL_DEEP_WNS_NS_DEFAULT = -1.0


TAIL_CTRL_MAX_MOVES_DEFAULT = 12


TAIL_RESERVE_STAGNANT_S_DEFAULT = 240.0


DEEP_WNS_TAIL_RESERVE_S_DEFAULT = 0.0        # OFF (mechanisms ship OFF; enabled at ship path)


def _arg_int(args: dict, key: str, default: int) -> int:
    """Convert an LLM-supplied numeric tool argument to an integer without
    raising.

    Integer-valued strings and floats, including "20.0", are accepted through
    float conversion. Uninterpretable values return the schema default so
    malformed tool calls do not consume retry budget.
    """
    v = args.get(key, default)
    try:
        return int(v)
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        f = float(v)
        # nan and inf survive float() and then blow up inside int() -- nan with
        # ValueError, inf with OverflowError. The "never raises on anything"
        # test caught inf slipping through the first version of this helper.
        if f != f or f in (float("inf"), float("-inf")):
            raise ValueError("non-finite")
        return int(f)
    except (TypeError, ValueError, OverflowError):
        logger.warning(
            f"tool arg {key}={v!r} is not a number; using default {default}")
        return int(default)


def _arg_float(args: dict, key: str, default: float) -> float:
    """float twin of _arg_int. NEVER raises; same rationale."""
    v = args.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        logger.warning(
            f"tool arg {key}={v!r} is not a number; using default {default}")
        return float(default)


def _run_dir_base() -> Path:
    """Base directory for dcp_optimizer_run-* artifact dirs. Defaults to CWD
    (contest/eval behavior, unchanged — the env var is never set there). Local
    ops set FPL26_RUN_DIR_BASE (e.g. a large scratch volume) to
    keep run artifacts off a nearly-full system drive (555 run dirs had
    accumulated ~50 GB)."""
    base = os.environ.get("FPL26_RUN_DIR_BASE", "").strip()
    if base:
        p = Path(base)
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            pass
    return Path.cwd()


def resolve_polish_reserve_s(cli_value,
                             default: float = POLISH_RESERVE_S_DEFAULT) -> float:
    """Effective post-route polish reserve (seconds).

    CLI (--polish-reserve-s) wins over env (FPL26_POLISH_RESERVE_S).
    0 disables the reserve (kill switch); unset/unparseable/negative
    keeps the default.
    """
    v = cli_value
    if v is None:
        env = os.environ.get("FPL26_POLISH_RESERVE_S", "").strip()
        if env:
            try:
                v = float(env)
            except ValueError:
                logger.warning(
                    f"FPL26_POLISH_RESERVE_S={env!r} not a float; keeping "
                    f"default {default:.0f}s polish reserve.")
                return default
    if v is None:
        return default
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    if v < 0.0:
        logger.warning(
            f"polish_reserve_s={v} negative; keeping default {default:.0f}s.")
        return default
    return v


def resolve_bare_reroute_polish_enabled(cli_disable) -> bool:
    """Resolve whether bare reroute polishing is enabled.

    Polishing is enabled by default. It is disabled by the
    `--no-bare-reroute-polish` flag or when `FPL26_NO_BARE_REROUTE_POLISH` is
    set to `1`, `true`, `yes`, or `on`.
    """
    if cli_disable:
        return False
    env = os.environ.get("FPL26_NO_BARE_REROUTE_POLISH", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return False
    return True


def resolve_bare_reroute_min_gain_ns(
        cli_value,
        default: float = BARE_REROUTE_MIN_GAIN_NS_DEFAULT) -> float:
    """Minimum per-iteration WNS gain (ns) to keep the bare re-route
    loop rolling.  CLI (--bare-reroute-min-gain) wins over env
    (FPL26_BARE_REROUTE_MIN_GAIN); unset/unparseable/negative keeps the
    default (same convention as resolve_polish_reserve_s)."""
    v = cli_value
    if v is None:
        env = os.environ.get("FPL26_BARE_REROUTE_MIN_GAIN", "").strip()
        if env:
            try:
                v = float(env)
            except ValueError:
                logger.warning(
                    f"FPL26_BARE_REROUTE_MIN_GAIN={env!r} not a float; "
                    f"keeping default {default:.3f} ns.")
                return default
    if v is None:
        return default
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    if v < 0.0:
        logger.warning(
            f"bare_reroute_min_gain={v} negative; keeping default "
            f"{default:.3f} ns.")
        return default
    return v


def resolve_bare_reroute_max_iters(
        cli_value,
        default: int = BARE_REROUTE_MAX_ITERS_DEFAULT) -> int:
    """Bare re-route loop runaway guard (max iterations).  CLI
    (--bare-reroute-max-iters) wins over env
    (FPL26_BARE_REROUTE_MAX_ITERS); unset/unparseable/<1 keeps the
    default."""
    v = cli_value
    if v is None:
        env = os.environ.get("FPL26_BARE_REROUTE_MAX_ITERS", "").strip()
        if env:
            try:
                v = int(float(env))
            except ValueError:
                logger.warning(
                    f"FPL26_BARE_REROUTE_MAX_ITERS={env!r} not an int; "
                    f"keeping default {default}.")
                return default
    if v is None:
        return default
    try:
        v = int(v)
    except (TypeError, ValueError):
        return default
    if v < 1:
        logger.warning(
            f"bare_reroute_max_iters={v} < 1; keeping default {default}.")
        return default
    return v


def resolve_tail_controller_enabled(cli_disable) -> bool:
    """Tail-controller kill-switch resolution (default ON; fail-closed —
    any controller error falls back to the plain M1 bare-reroute loop).
    Disabled by CLI --no-tail-controller OR env FPL26_NO_TAIL_CONTROLLER
    truthy (same convention as the bare-reroute switch)."""
    if cli_disable:
        return False
    env = os.environ.get("FPL26_NO_TAIL_CONTROLLER", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return False
    return True


def resolve_tail_ctrl_deep_wns_ns(
        cli_value,
        default: float = TAIL_CTRL_DEEP_WNS_NS_DEFAULT) -> float:
    """Deep-WNS arming threshold (ns, must be <= 0).  The controller arms
    when best_wns <= threshold; shallower states keep the shipped plain
    M1 loop (plateau probe: every menu move decays to no-op near-met).
    CLI --tail-ctrl-deep-wns wins over env FPL26_TAIL_CTRL_DEEP_WNS;
    unset/unparseable/positive keeps the default."""
    v = cli_value
    if v is None:
        env = os.environ.get("FPL26_TAIL_CTRL_DEEP_WNS", "").strip()
        if env:
            try:
                v = float(env)
            except ValueError:
                logger.warning(
                    f"FPL26_TAIL_CTRL_DEEP_WNS={env!r} not a float; "
                    f"keeping default {default:.3f} ns.")
                return default
    if v is None:
        return default
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    if v > 0.0:
        logger.warning(
            f"tail_ctrl_deep_wns={v} positive; keeping default "
            f"{default:.3f} ns.")
        return default
    return v


def resolve_tail_ctrl_max_moves(
        cli_value,
        default: int = TAIL_CTRL_MAX_MOVES_DEFAULT) -> int:
    """Controller global runaway guard (max move executions).  CLI
    --tail-ctrl-max-moves wins over env FPL26_TAIL_CTRL_MAX_MOVES;
    unset/unparseable/<1 keeps the default."""
    v = cli_value
    if v is None:
        env = os.environ.get("FPL26_TAIL_CTRL_MAX_MOVES", "").strip()
        if env:
            try:
                v = int(float(env))
            except ValueError:
                logger.warning(
                    f"FPL26_TAIL_CTRL_MAX_MOVES={env!r} not an int; "
                    f"keeping default {default}.")
                return default
    if v is None:
        return default
    try:
        v = int(v)
    except (TypeError, ValueError):
        return default
    if v < 1:
        logger.warning(
            f"tail_ctrl_max_moves={v} < 1; keeping default {default}.")
        return default
    return v


def resolve_tail_ctrl_m1_echo(cli_flag) -> bool:
    """Resolve whether to reroute immediately after an accepted non-M1 controller
    move.

    The echo is disabled by default and enabled by `--tail-ctrl-m1-echo` or a
    truthy `FPL26_TAIL_CTRL_M1_ECHO` value. When enabled, it performs one
    inexpensive bare route while the accepted state is fresh.
    """
    if cli_flag:
        return True
    env = os.environ.get("FPL26_TAIL_CTRL_M1_ECHO", "").strip().lower()
    return env in ("1", "true", "yes", "on")


def resolve_preempt_loop_clock_enabled() -> bool:
    """Re-anchor the improvement clock when the loop preempts (default ON).

    Kill switch: FPL26_PREEMPT_LOOP_CLOCK set to any of the house falsy
    values.  It used to be compared against the single string "0", which
    silently ignored `false`/`no`/`off` — the spellings the Makefile's own
    switches accept — so the documented way to turn this off did not always
    turn it off."""
    # The default is spelled "1" rather than "" so scripts/ship_config.py's
    # static reader sees the shipped value: it resolves a flag by the literal
    # default in os.environ.get(), and an empty default would read as OFF.
    env = os.environ.get("FPL26_PREEMPT_LOOP_CLOCK", "1").strip().lower()
    return env not in ("0", "false", "no", "off")


def resolve_tail_reserve_stagnant_s(
        default: float = TAIL_RESERVE_STAGNANT_S_DEFAULT) -> float:
    """Stagnation-guard window for the tail reserve (seconds).

    Env-only knob (FPL26_TAIL_RESERVE_STAGNANT_S); unset/unparseable/
    negative keeps the default (house convention).  0 disables the guard
    (loop always treated as stagnant at the boundary)."""
    env = os.environ.get("FPL26_TAIL_RESERVE_STAGNANT_S", "").strip()
    if not env:
        return default
    try:
        v = float(env)
    except ValueError:
        logger.warning(
            f"FPL26_TAIL_RESERVE_STAGNANT_S={env!r} not a float; "
            f"keeping default {default:.0f}s.")
        return default
    if v < 0.0:
        logger.warning(
            f"FPL26_TAIL_RESERVE_STAGNANT_S={v} negative; keeping "
            f"default {default:.0f}s.")
        return default
    return v


def resolve_deep_wns_tail_reserve_s(
        cli_value,
        default: float = DEEP_WNS_TAIL_RESERVE_S_DEFAULT) -> float:
    """Deep-WNS tail reserve resolution (seconds; DEFAULT 0 = OFF).

    CLI (--deep-wns-tail-reserve) wins over env
    (FPL26_DEEP_WNS_TAIL_RESERVE); 0/unset = OFF (zero behavior
    change); a value in (0, 1) is a fraction of --max-wall-seconds;
    unparseable/negative keeps the default (same convention as
    resolve_polish_reserve_s)."""
    v = cli_value
    if v is None:
        env = os.environ.get("FPL26_DEEP_WNS_TAIL_RESERVE", "").strip()
        if env:
            try:
                v = float(env)
            except ValueError:
                logger.warning(
                    f"FPL26_DEEP_WNS_TAIL_RESERVE={env!r} not a float; "
                    f"keeping default {default:.0f}s (reserve OFF).")
                return default
    if v is None:
        return default
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    if v < 0.0:
        logger.warning(
            f"deep_wns_tail_reserve={v} negative; keeping default "
            f"{default:.0f}s (reserve OFF).")
        return default
    return v
