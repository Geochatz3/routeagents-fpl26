#!/usr/bin/env python3
"""Run a stochastic optimizer repeatedly within one wall budget and retain the
best valid artifact.

Each attempt invokes the existing optimizer target without changing its search
behavior. Pure selection logic is isolated in `select_best`, while `run`
manages subprocesses, budgets, and publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def _atomic_publish(src, dst) -> bool:
    """Publish ``src`` to the scored location ``dst`` atomically, mtime=now.

    The eval scores the mtime-NEWEST ``<stem>_optimized*.dcp`` and on timeout
    takes "the last solution generated" — so the published best must (a) never
    be observable half-written (a wall kill mid-copy would ship a truncated
    DCP that fails validation and scores zero) and (b) always be the newest
    candidate on disk (plain copy2 preserves the SOURCE's mtime, which can be
    minutes old by the time a later refresh re-publishes attempt 1's file).
    Copy to a temp file in dst's directory, os.replace (atomic on the same
    filesystem), then bump mtime to now.
    """
    src_path, dst_path = Path(src), Path(dst)
    tmp = dst_path.with_name(dst_path.name + f".tmp{os.getpid()}")
    try:
        shutil.copy2(src_path, tmp)
        os.replace(tmp, dst_path)
        os.utime(dst_path, None)
        return True
    except Exception as e:
        print(f"[multi-restart] WARNING: atomic publish failed ({e}); "
              f"trying direct copy", flush=True)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            shutil.copy2(src_path, dst_path)
            os.utime(dst_path, None)
            return True
        except Exception as e2:
            print(f"[multi-restart] WARNING: publish failed entirely: {e2}",
                  flush=True)
            return False


def _discard_scratch_dcp(path) -> None:
    """Drop a /tmp staging DCP once it can no longer be published.

    Attempt outputs and the polish staging file are 100-300MB each and live
    in /tmp.  Nothing else removes them, so a 4-attempt run stranded ~1GB and
    back-to-back benchmarks on one eval box accumulated until /tmp filled --
    at which point _atomic_publish's temp copy is the first thing to fail.
    Best-effort by construction: losing a scratch file must never take the
    run down, so every failure is swallowed."""
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass


# Shared with the signal handler: the scored location and the in-flight
# attempt's output path. Plain dict mutation is async-signal-safe enough for
# a single-threaded wrapper.
_SIG_STATE: dict = {"final_output": None, "current_attempt_out": None,
                    # ship_tier() of what this wrapper last published to the
                    # scored location; None until the first publish. Lets the
                    # signal handler distinguish a real incumbent from a
                    # tier-0 (known-worthless, alpha==0 by construction) one.
                    "published_tier": None}


def _stream_md5(path, chunk_size: int = 1024 * 1024) -> str:
    """md5 of a file, streamed (DCPs are 100-300MB; ~1-2s in the handler)."""
    h = hashlib.md5()
    with open(str(path), "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _read_shipped_manifest(cur) -> Optional[dict]:
    """Read ``<cur>.shipped.json`` — the identity (size+md5) the AGENT
    recorded for the artifact it actually wrote (finalize or emergency
    finalize).  None when absent/unparseable.  Best-effort, never raises."""
    try:
        mf = Path(str(cur) + ".shipped.json")
        if not mf.exists():
            return None
        data = json.loads(mf.read_text())
        if isinstance(data, dict) and data.get("md5") and data.get("size"):
            return data
    except Exception:
        pass
    return None


def _matches_manifest(cur, manifest) -> bool:
    """True iff the on-disk file matches the agent-recorded identity."""
    try:
        p = Path(cur)
        if p.stat().st_size != manifest.get("size"):
            return False
        return _stream_md5(p) == manifest.get("md5")
    except Exception:
        return False


def _plausible_dcp(cur) -> bool:
    """Cheap sanity gate: a Vivado DCP is a zip archive → 'PK' magic +
    nonzero size.  Rejects the random-garbage injection class even when
    no manifest is available.  Never raises."""
    try:
        p = Path(cur)
        if p.stat().st_size <= 0:
            return False
        with open(str(p), "rb") as f:
            return f.read(2) == b"PK"
    except Exception:
        return False


def _emergency_publish(final_output, current_out,
                       wait_s: float = 20.0,
                       incumbent_tier=None) -> str:
    """Publish a valid in-flight artifact when the harness terminates the wrapper.

    Preserve an existing scored artifact unless it is a wrapper-published
    tier-zero fallback and the in-flight artifact is manifest-verified.

    Wait up to `wait_s` for the agent's shipment manifest and an attempt file
    whose size and MD5 match it. This pair is the authoritative proof that
    finalization produced the file.

    If no verified pair appears before the deadline, publish only bytes with
    plausible DCP ZIP magic. Never publish an implausible file merely because
    the attempt path exists.
    """
    if final_output is None:
        return "no_final_output"
    fo = Path(final_output)
    incumbent_is_tier0 = fo.exists() and incumbent_tier == 0
    if fo.exists() and not incumbent_is_tier0:
        return "kept_existing"
    # A tier-0 incumbent scores nothing by construction — the artifact is the
    # unmodified input — so a manifest-verified in-flight win must be allowed
    # to replace it, or a harness kill at the wrong moment forfeits the whole
    # benchmark to the never-worse rule.  Replacement demands the full
    # authorship proof; the unverified deadline fallback still never
    # overwrites an incumbent, since a valid baseline beats plausible but
    # unproven bytes.
    if current_out is None:
        return "kept_existing_tier0" if incumbent_is_tier0 \
            else "nothing_to_publish"
    cur = Path(current_out)
    deadline = time.monotonic() + wait_s
    while True:
        manifest = _read_shipped_manifest(cur)
        if (manifest is not None and cur.exists()
                and _matches_manifest(cur, manifest)):
            ok = _atomic_publish(cur, fo)
            if ok:
                return ("published_inflight_verified_over_tier0"
                        if incumbent_is_tier0
                        else "published_inflight_verified")
            return "publish_failed"
        if time.monotonic() >= deadline:
            break
        time.sleep(0.5)
    if incumbent_is_tier0:
        return "kept_existing_tier0"
    if cur.exists():
        if _plausible_dcp(cur):
            ok = _atomic_publish(cur, fo)
            return ("published_inflight_unverified" if ok
                    else "publish_failed")
        return "inflight_rejected_not_a_dcp"
    return "inflight_never_landed"


def _install_signal_publisher() -> None:
    def _handler(signum, frame):  # noqa: ARG001
        print(f"\n[multi-restart] received signal {signum}; emergency publish",
              flush=True)
        try:
            action = _emergency_publish(
                _SIG_STATE.get("final_output"),
                _SIG_STATE.get("current_attempt_out"),
                incumbent_tier=_SIG_STATE.get("published_tier"))
            print(f"[multi-restart] emergency publish: {action}", flush=True)
        finally:
            os._exit(128 + signum)

    for s in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if s is None:
            continue
        try:
            signal.signal(s, _handler)
        except Exception as e:
            print(f"[multi-restart] WARNING: could not install handler for "
                  f"{s}: {e}", flush=True)


def ship_tier(status) -> int:
    """Classify whether an attempt artifact is known to have no optimization
    value.

    Return tier 0 for baseline fallbacks, no-improvement outcomes, and hard
    failures. Return tier 1 for optimized outcomes and unrecognized or missing
    statuses.

    Unknown status remains tier 1 because emergency publication can preserve a
    genuinely improved artifact before final lifecycle metadata is available.
    The tier distinguishes only known-zero outcomes from all others.
    """
    s = str(status or "")
    if (s.startswith("VALID_FALLBACK") or s.startswith("NO_IMPROVEMENT")
            or s.startswith("HARD_FAIL")):
        return 0
    return 1


def select_best(attempts: list[dict]) -> Optional[dict]:
    """Select the best usable attempt while demoting artifacts known to be
    fallbacks.

    Consider only attempts whose output exists. Rank by `ship_tier` first, then
    highest tracked frequency, then recognized optimized status, and finally
    earliest attempt. Return `None` when no usable output exists.

    Frequency and shipment status come from different records: a high tracked
    frequency may accompany a shipped baseline after validation fails. Tiering
    therefore precedes frequency. Unknown statuses remain eligible because they
    do not prove fallback publication.
    """
    usable = [a for a in attempts if a.get("exists") and a.get("fmax") is not None]
    if not usable:
        return None

    def key(a):
        # Tail is the previous key, unchanged: (fmax, opt, -i).
        opt = 1 if a.get("status") == "VALID_OPTIMIZED" else 0
        return (ship_tier(a.get("status")), a["fmax"], opt, -a["i"])

    return max(usable, key=key)


# Placement-limited designs above the 100-tile spread threshold can have
# bimodal outcomes, so the exploration band requires at least three attempts.
HIGH_SPREAD_TILES = 100.0
HIGH_SPREAD_MIN_ATTEMPTS = 3
# Truncation gate: a new attempt whose budget is well below what a completed
# full-stack attempt needed on this design can only reach recipe-stage
# quality, which keep-best then discards — while its wall is still charged
# against the score.  This factor still admits attempts with a plausible
# shot, since attempt durations vary with the draw and with where futility
# stops them.
TRUNCATION_FACTOR = 0.75

# Cumulative LLM-spend ceiling across attempts.  Cost is billed per
# benchmark and a design that crosses the contest cap scores zero, so a
# retrospective per-attempt check is not enough: at just under the cap it
# will still launch another attempt carrying a fresh full allowance.
# cost_gate() closes both holes with a predictive pre-launch estimate and a
# shrinking per-attempt budget — the ceiling less what is already spent —
# handed to the agent.  The default leaves margin under the cap for a single
# call's overshoot, since the agent's exit is checked after each call lands.
# Kill switch: set the ceiling to zero.
COST_CEILING_DEFAULT = 0.80

# Cumulative OpenRouter spend at which no further attempt is launched.
# One definition: the function default and the CLI default were the same
# literal in two places, which is how they drift.
COST_CAP_DEFAULT = 0.85


def cost_gate(attempts: list[dict], cost_so_far: float,
              ceiling: Optional[float]) -> "tuple[bool, Optional[float], str]":
    """Pure pre-attempt spend gate. Returns (launch, allowance, reason).

    - ceiling None/<=0: breaker OFF -> (True, None, "breaker_off"); no
      budget is passed down (agent keeps its default $0.75 exit).
    - refuse when cost_so_far already >= ceiling;
    - refuse when cost_so_far + estimate would CROSS the ceiling, where
      estimate = max of prior attempts' known costs (conservative: the
      worst draw seen on THIS design predicts the next). No priors (first
      attempt / no cost records) -> no estimate -> launch.
    - allowance = ceiling − cost_so_far, truncated DOWN to the cent
      (conservative); < $0.01 refuses (an LLM-less attempt only burns
      wall the keep-best would discard).
    """
    if ceiling is None or ceiling <= 0:
        return True, None, "breaker_off"
    if cost_so_far >= ceiling:
        return False, 0.0, (f"spent ${cost_so_far:.2f} >= ceiling "
                            f"${ceiling:.2f}")
    est = max((float(a["cost"]) for a in attempts if a.get("cost")),
              default=0.0)
    if est > 0 and cost_so_far + est > ceiling:
        return False, 0.0, (f"predicted spend ${cost_so_far:.2f} + "
                            f"${est:.2f} (max prior attempt) > ceiling "
                            f"${ceiling:.2f} — the #15 zeroing shape")
    allowance = int((ceiling - cost_so_far) * 100) / 100.0
    if allowance < 0.01:
        if attempts:
            return False, 0.0, (f"remaining allowance ${allowance:.2f} < "
                                f"$0.01")
        # Never refuse the FIRST attempt outright: a sub-cent ceiling clamps
        # the allowance up to one minimal attempt, because one lean draw beats
        # shipping nothing.  Attempts 2 and later keep the strict floor — with
        # an incumbent banked, a model-less redraw only burns wall that
        # keep-best discards.
        allowance = 0.01
    return True, allowance, ""


# FEATURE-AWARE RESTART SPLIT (--split-aware, default off).  Attempt 1 can
# consume nearly the whole budget, leaving less than the attempt floor, so
# attempt 2 never fires and best-of-N variance protection silently degrades
# to a single shot.  Attempt-1 consumption scales monotonically with
# checkpoint file size, so the split key is os.stat().st_size: instant and
# free, where a cell count would need a checkpoint open inside the agent —
# far too late for a pre-launch decision.
#
# Size-class boundaries, in bytes.  The thresholds sit in the gaps between
# the observed size clusters, so they separate the classes with margin
# rather than cutting through one.
SMALL_MAX_BYTES = 20_000_000
MEDIUM_MAX_BYTES = 70_000_000
# Attempt-1 caps in seconds at evaluation scale.  Conservative: about the
# observed productive attempt-1 usage, so a capped attempt 1 can still
# complete a full stack while leaving the floor reachable for attempt 2.
# Small designs saturate quickly; mediums use most of the wall.  The largest
# class is deliberately uncapped — a single route can take most of an hour
# there, and any cap risks leaving such a design unrouted, which scores
# nothing.
ATTEMPT1_CAP_SMALL_S = 1800.0
ATTEMPT1_CAP_MEDIUM_S = 2400.0


def classify_design(dcp_path: Path) -> str:
    """Pure pre-launch size classification: "small" | "medium" | "large".

    Keyed on DCP file size only (os.stat — no Vivado, no subprocess).
    Fail-open: a missing/unreadable/malformed path returns
    "large", i.e. the uncapped single-shot behavior — mirrors the
    _input_cell_count fail-open in dcp_optimizer.py.
    """
    try:
        size = os.stat(dcp_path).st_size
    except Exception:
        return "large"
    if size < SMALL_MAX_BYTES:
        return "small"
    if size <= MEDIUM_MAX_BYTES:
        return "medium"
    return "large"


def attempt1_budget(total_wall: float, size_class: str,
                    attempt_floor: Optional[float] = None) -> float:
    """Compute the wall-time budget available to the first optimization attempt.

    Small and medium designs use their configured caps; large or unknown
    classes use the full wall budget. The result never exceeds the total wall
    time.

    When `attempt_floor` is provided, also cap the first attempt so the
    remainder can satisfy both launch gates: `total_wall - elapsed >=
    attempt_floor` and `total_wall - elapsed >= TRUNCATION_FACTOR * elapsed`.
    This derives ceilings of `total_wall - attempt_floor` and `total_wall / (1
    + TRUNCATION_FACTOR)`.

    The derived cap is necessary but not sufficient because finalization may
    make actual elapsed time exceed the budget. Without `attempt_floor`, only
    the class table applies.
    """
    if size_class == "small":
        budget = min(total_wall, ATTEMPT1_CAP_SMALL_S)
    elif size_class == "medium":
        budget = min(total_wall, ATTEMPT1_CAP_MEDIUM_S)
    else:
        # Large (boom-class) stays UNCAPPED and is NOT clamped by the floor:
        # route_design alone measured 2153.13s there, and starving attempt 1 to
        # fund a second draw is how the alpha=0 boom failure mode came back.
        return total_wall
    # Only clamp a cap that is actually BINDING. When total_wall is already below
    # the class cap there is no split being imposed — attempt 1 legitimately gets
    # the whole (short) wall, and shaving it to reserve for a draw the wall cannot
    # fund anyway would just lose optimization time.
    if attempt_floor and attempt_floor > 0 and budget < total_wall:
        # Both gates, not just the floor: the truncation gate is the tighter one
        # on the eval wall (2000 s vs 2300 s) and is what actually blocked mediums.
        ceiling = min(total_wall - attempt_floor,
                      total_wall / (1.0 + TRUNCATION_FACTOR))
        if ceiling > 0:
            budget = min(budget, ceiling)
    return budget


def should_skip_truncated(attempts: list[dict], remaining: float,
                          factor: float = TRUNCATION_FACTOR) -> "tuple[bool, str]":
    """Pure gate: skip launching another attempt when a VALID_OPTIMIZED
    incumbent exists and `remaining` cannot fit what the FASTEST completed
    OPTIMIZED attempt needed (x factor). Never fires without an OPTIMIZED
    incumbent (a truncated attempt may still beat a FALLBACK baseline)."""
    ref = 0.0
    for a in attempts:
        if (a.get("status") == "VALID_OPTIMIZED" and a.get("exists")
                and (a.get("elapsed") or 0) > 0):
            e = float(a["elapsed"])
            if ref == 0.0 or e < ref:
                ref = e
    if ref > 0 and remaining < factor * ref:
        return True, (f"remaining {remaining:.0f}s < {factor:.2f}x fastest "
                      f"completed OPTIMIZED attempt ({ref:.0f}s)")
    return False, ""


def should_stop_early(attempts: list[dict], eps: float = 1.0,
                      min_attempts: int = 2, min_confirmations: int = 2,
                      high_spread: bool = False) -> bool:
    """Stop launching attempts after a strong result receives sufficient
    confirmation.

    Return `True` only after at least `min_attempts` have run and at least
    `min_confirmations` valid optimized attempts fall within `eps` MHz of the
    best observed frequency. Baseline or fallback outcomes never establish
    confirmation.
    """
    if high_spread:
        min_attempts = max(min_attempts, HIGH_SPREAD_MIN_ATTEMPTS)
    if len(attempts) < min_attempts:
        return False
    # A result that shipped without a refreshed EDIF is the same scoreable
    # artifact class: the validator regenerates the EDIF from the checkpoint
    # and distrusts sidecars.  An exact-status match therefore breaks this
    # stop exactly where it is needed most — on a design whose attempts are
    # bit-identical but carry that status — and spends another attempt to
    # re-derive the same number.
    opt = [a for a in attempts
           if a.get("status") in ("VALID_OPTIMIZED", "VALID_OPTIMIZED_NO_EDIF")
           and a.get("fmax") is not None]
    if not opt:
        return False
    best = max(a["fmax"] for a in opt)
    confirmations = sum(1 for a in opt if abs(a["fmax"] - best) <= eps)
    return confirmations >= min_confirmations


def _read_run_spread(run_dir: Path) -> Optional[float]:
    """First numeric critical_path_spread from the run's decision trace (the
    only per-run artifact that carries it)."""
    try:
        with (run_dir / "decisions.jsonl").open() as fh:
            for line in fh:
                try:
                    v = json.loads(line).get("critical_path_spread")
                except Exception:
                    continue
                if isinstance(v, (int, float)):
                    return float(v)
    except Exception:
        pass
    return None


def _read_run_metrics(run_dir: Path) -> dict:
    """Read frequency, final status, elapsed time, and LLM cost from a run
    directory.

    Prefer `token_usage.json` for cost because it contains the complete
    normal-exit summary. If it is missing or unreadable, use the crash-safe
    incremental `cost_ledger.json`.

    Leave cost as `None` only when both sources are unreadable; the caller must
    then charge its predictive estimate rather than treating the attempt as
    free.
    """
    out = {"fmax": None, "status": None, "elapsed": None, "cost": None,
           "initial_fmax": None}
    tu = run_dir / "token_usage.json"
    lc = run_dir / "lifecycle_metadata.json"
    try:
        s = json.loads(tu.read_text())["summary"]
        out["fmax"] = s.get("best_fmax_mhz")
        # Carried only so a post-polish correction can restate the IMPROVEMENT,
        # not just the Fmax — the agent's own summary is already stale by then.
        out["initial_fmax"] = s.get("initial_fmax_mhz")
        out["elapsed"] = s.get("total_runtime_seconds")
        out["cost"] = s.get("total_cost")
    except Exception:
        pass
    if out["cost"] is None:
        try:
            led = json.loads((run_dir / "cost_ledger.json").read_text())
            c = led.get("total_cost")
            if isinstance(c, (int, float)):
                out["cost"] = float(c)
        except Exception:
            pass
    try:
        m = json.loads(lc.read_text())
        out["status"] = m.get("final_status")
        if out["elapsed"] is None:
            out["elapsed"] = m.get("elapsed_seconds")
    except Exception:
        pass
    return out


# Predictive per-attempt cost estimate, charged when an attempt launched with
# a model budget leaves no readable cost record at all.  Uses the maximum of
# prior attempts' known costs, or this conservative prior — the top of the
# documented per-attempt band.  Because the agent seeds its ledger at zero,
# a benign crash with no model calls leaves an explicit zero record and is
# charged nothing; the estimate only fires when spend is genuinely
# unknowable, which preserves the never-cross-the-ceiling guarantee without
# bricking retries.
COST_ESTIMATE_FALLBACK_USD = 0.35


def _estimate_attempt_cost(attempts: list[dict]) -> float:
    """Predictive charge for an attempt with no readable cost record."""
    est = max((float(a["cost"]) for a in attempts if a.get("cost")),
              default=0.0)
    return est if est > 0 else COST_ESTIMATE_FALLBACK_USD


def _wrapper_run_dir_base(repo: Path) -> Path:
    """The SAME base dir the agent's _run_dir_base() will use.

    Replicates dcp_optimizer._run_dir_base(): FPL26_RUN_DIR_BASE wins when
    set (pre-existing bug: the agent honored it while the wrapper globbed
    the repo only, so attribution silently missed every run dir on local
    ops machines); otherwise the attempt's cwd — the make subprocess runs
    with cwd=repo, so that is `repo`."""
    base = os.environ.get("FPL26_RUN_DIR_BASE", "").strip()
    if base:
        p = Path(base)
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            pass
    return repo


def _snapshot_run_dirs(base: Path) -> "set[str]":
    """Names of the run dirs that exist BEFORE an attempt launches."""
    try:
        return {p.name for p in base.glob("dcp_optimizer_run-*")}
    except Exception:
        return set()


def _attribute_run_dir(base: Path, before: "set[str]") -> Optional[Path]:
    """Snapshot-diff attribution — the newest run dir NOT in
    the pre-attempt snapshot.  Returns None when the attempt created no run
    dir (crashed pre-init, or only foreign/concurrent dirs exist): the
    caller must NOT fall back to the newest pre-existing dir — that is
    exactly the double-count/mis-attribution bug the repo-global mtime
    glob had — and charges the predictive estimate instead."""
    try:
        new = [p for p in base.glob("dcp_optimizer_run-*")
               if p.name not in before]
        if not new:
            return None
        return max(new, key=lambda p: p.stat().st_mtime)
    except Exception:
        return None


# Wedge guard: how long past its own wall budget an attempt may stay alive
# before the wrapper intervenes.  The agent finalizes at its internal
# deadline and normally exits well inside its budget, but elapsed time can
# legitimately overrun by a bounded teardown, so the grace is generous.
# What this closes: an attempt wedged in teardown after a successful
# finalize used to block an unbounded wait forever, silently turning
# best-of-N into best-of-one and leaving the scored location unwritten.
WEDGE_GRACE_S = 420.0


def _pids_with_cmdline_token(token: str) -> list:
    """Return process IDs whose command lines contain an attempt token, excluding
    the current process.

    The token must be the bare `/tmp/mr_<stem>_<pid>_<i>.dcp` path, not an
    environment assignment. This path remains contiguous across the process
    lineage, uniquely identifies an attempt, and is absent from the wrapper
    command line. Direct `/proc` scanning avoids pattern-matching tools that
    can match their own invocation.
    """
    me = os.getpid()
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            with open(f"/proc/{d}/cmdline", "rb") as f:
                cl = f.read().decode("utf-8", "replace")
        except Exception:
            continue
        if token in cl:
            out.append(int(d))
    return out


def _run_attempt_process(cmd, repo, budget_s: float, attempt_i: int,
                         grace_s: float = WEDGE_GRACE_S) -> int:
    """Run one agent attempt, bounded at budget + grace.

    Termination order matters, and the TARGET matters more (verified
    empirically): `cmd` is a make invocation, and SIGTERM to
    make kills make and its sh while the PYTHON GRANDCHILD SURVIVES —
    orphaned, still holding a 10-26 GB Vivado while the next attempt
    launches on a 32 GB machine. So the guard signals every process that
    carries this attempt's unique OUTPUT= token directly: the agent's
    SIGTERM handler runs its emergency finalize (artifacts + token_usage
    land on disk) and a wedged attempt degrades to "the design's normal
    result", not a loss. SIGKILL only if the SIGTERM path is itself wedged.
    Deliberately NO process-group games (no start_new_session): the eval
    harness delivers its group SIGTERM to wrapper AND agent together, and
    that design must keep working.
    """
    _tok_arg = next((str(a) for a in cmd if str(a).startswith("OUTPUT=")),
                    None)
    # Bare path, not the OUTPUT= form — see _pids_with_cmdline_token.
    token = _tok_arg.split("=", 1)[1] if _tok_arg else None
    proc = subprocess.Popen(cmd, cwd=str(repo))
    try:
        return proc.wait(timeout=budget_s + grace_s)
    except subprocess.TimeoutExpired:
        victims = _pids_with_cmdline_token(token) if token else []
        print(f"[multi-restart] attempt {attempt_i} WEDGED: alive "
              f"{grace_s:.0f}s past its {budget_s:.0f}s budget — SIGTERM to "
              f"{len(victims)} lineage pid(s) + make (agent "
              f"emergency-finalizes), then SIGKILL", flush=True)
        for pid in victims:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        proc.terminate()
        rc = None
        try:
            rc = proc.wait(timeout=90.0)
        except subprocess.TimeoutExpired:
            pass
        # The build tool may be gone while the agent is still finalizing — give
        # the lineage a bounded window to finish writing before metrics are
        # read, then hard-kill whatever remains.  If the build tool itself
        # wedged past its own termination grace, skip the window: nothing in
        # this lineage is finalizing sanely.
        deadline = time.monotonic() + (30.0 if rc is not None else 0.0)
        remaining = [p for p in (_pids_with_cmdline_token(token)
                                 if token else [])]
        while remaining and time.monotonic() < deadline:
            time.sleep(0.5)
            remaining = [p for p in remaining if _pid_alive(p)]
        if rc is None or remaining:
            print(f"[multi-restart] attempt {attempt_i}: "
                  f"{len(remaining)} pid(s) survived SIGTERM — SIGKILL",
                  flush=True)
            for pid in remaining:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            if rc is None:
                proc.kill()
                try:
                    rc = proc.wait(timeout=30.0)
                except subprocess.TimeoutExpired:
                    print(f"[multi-restart] attempt {attempt_i} unreapable "
                          f"even after SIGKILL; abandoning", flush=True)
                    return -1
        return rc if rc is not None else -1


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _attempt_cmd(input_dcp: Path, out_i: Path, remaining: float,
                 ils_polish: bool, wall_handback: bool = False,
                 llm_cost_budget: Optional[float] = None) -> list[str]:
    """Build the per-attempt make invocation. ILS=1 turns on the in-agent
    ILS-polish stage (AWS-validated: converts leftover wall on
    stuck/ceiling designs into gains; trigger gates keep it off when the
    budget is tight or the recipe met timing). WALL_HANDBACK=1
    (default off) turns on the agent's saturation early-exit so a saturated
    attempt hands its unused wall back to this wrapper's next-loop
    `remaining` computation."""
    cmd = ["make", "run_once",
           f"DCP={input_dcp}", f"OUTPUT={out_i}",
           f"MAX_WALL={int(remaining)}"]
    if ils_polish:
        cmd.append("ILS=1")
    if wall_handback:
        cmd.append("WALL_HANDBACK=1")
    # C1-T1: shrinking per-attempt LLM budget (cumulative ceiling − spent);
    # the Makefile forwards it as --llm-cost-budget, where the agent clamps
    # its in-attempt cost exit to min($0.75, budget). None = breaker off.
    if llm_cost_budget is not None:
        cmd.append(f"LLM_COST_BUDGET={llm_cost_budget:.2f}")
    return cmd


POLISH_MIN_LEFTOVER_S = 600.0   # don't bother below this (vivado boot ~60-120s)
POLISH_RESERVE_S = 120.0        # leave margin between polish kill and the wall


def _resolve_vivado() -> Optional[str]:
    exe = os.environ.get("VIVADO_EXEC") or shutil.which("vivado")
    return exe if exe and (os.path.sep not in exe or Path(exe).exists()) else None


def polish_corrected_fmax(fmax_mhz: Optional[float],
                          delta_ns: Optional[float]) -> Optional[float]:
    """Fmax the polished artifact holds, from the pre-polish Fmax and the WNS gain.

    fmax = 1000 / (T + |WNS|), and the polish reduces |WNS| by ``delta_ns`` while
    moving nothing else, so the corrected value follows without needing T:

        total_ns  = 1000 / fmax_logged        # == T + |WNS|
        fmax_true = 1000 / (total_ns - delta)

    Returns None when the inputs cannot support the correction, so a caller never
    prints a number it did not derive.
    """
    if not fmax_mhz or fmax_mhz <= 0:
        return None
    if delta_ns is None or delta_ns <= 0:
        return None
    total_ns = 1000.0 / fmax_mhz
    remaining_ns = total_ns - delta_ns
    if remaining_ns <= 0:
        return None
    return 1000.0 / remaining_ns


def winner_polish(final_output: Path, remaining: float, repo: Path,
                  report: Optional[dict] = None) -> bool:
    """Wall-audit evidence: 723-940s stranded below the attempt floor per
    run. Spend stranded wall on one never-worse phys_opt pass over the shipped
    winner. Replaces the scored file ONLY on the Tcl script's verified IMPROVED
    verdict (wns strictly better AND still fully routed); any failure/timeout
    leaves the existing winner untouched. Returns True iff polished."""
    if remaining < POLISH_MIN_LEFTOVER_S or not final_output.exists():
        return False
    vivado = _resolve_vivado()
    if vivado is None:
        print("[multi-restart] polish: no vivado on PATH/VIVADO_EXEC; skip",
              flush=True)
        return False
    tcl = repo / "scripts" / "winner_polish.tcl"
    # The polish candidate must not live next to the scored file: a sibling
    # name matches the evaluation glob, and polish runs at the very end of the
    # wall, so a kill mid-checkpoint-write — the write is not atomic — would
    # leave a truncated, newest-mtime file that gets scored and fails
    # validation.  Stage elsewhere; only the gated atomic publish ever touches
    # the scored location.
    polished = Path("/tmp") / f"mr_polish_{os.getpid()}.dcp"
    try:
        print(f"[multi-restart] winner-polish: {remaining:.0f}s stranded -> "
              f"phys_opt pass on {final_output.name}", flush=True)
        try:
            budget = max(60.0, remaining - POLISH_RESERVE_S)
            p = subprocess.run(
                [vivado, "-mode", "batch", "-nolog", "-nojournal",
                 "-source", str(tcl), "-tclargs", str(final_output), str(polished),
                 "AggressiveExplore", "4", str(int(budget))],
                cwd=str(repo), capture_output=True, text=True, timeout=budget)
            out = (p.stdout or "") + (p.stderr or "")
        except subprocess.TimeoutExpired:
            print("[multi-restart] polish: timed out; keeping unpolished winner",
                  flush=True)
            return False
        except Exception as e:
            print(f"[multi-restart] polish: failed ({e}); keeping winner", flush=True)
            return False
        # Vivado in batch mode echoes every script line, prefixed, before
        # executing it — and the polish script's usage text contains the verdict
        # token itself, so a naive first-match grabs the echo and masks the real
        # verdict.  Skip echo lines.
        verdict = next((l for l in out.splitlines()
                        if "POLISH_VERDICT=" in l
                        and not l.lstrip().startswith("#")), "")
        print(f"[multi-restart] polish: {verdict or 'no verdict emitted'}", flush=True)
        if "POLISH_VERDICT=IMPROVED" in verdict and polished.exists():
            _atomic_publish(polished, final_output)
            print(f"[multi-restart] polish: scored output REPLACED with improved "
                  f"DCP -> {final_output}", flush=True)
            if report is not None:
                m = re.search(r"delta=([0-9.eE+-]+)", verdict)
                if m:
                    try:
                        report["delta_ns"] = float(m.group(1))
                    except ValueError:
                        pass
            return True
        return False
    finally:
        # The staging DCP is a 100-300MB copy source for the gated
        # publish above and nothing else.  Left behind, one per run,
        # back-to-back benchmarks on a single eval box fill /tmp until
        # a later _atomic_publish cannot write its temp copy.
        _discard_scratch_dcp(polished)


def run(input_dcp: Path, final_output: Path, total_wall: float,
        attempt_floor: float, max_attempts: int, repo: Path,
        cost_cap: float = COST_CAP_DEFAULT, ils_polish: bool = False,
        polish: bool = True, spread_aware: bool = False,
        split_aware: bool = False, wall_handback: bool = False,
        cost_ceiling: Optional[float] = COST_CEILING_DEFAULT) -> dict:
    stem = input_dcp.stem
    attempts: list[dict] = []
    cost_so_far = 0.0
    # Run dirs whose cost has been accumulated — the same dir
    # must never be charged twice (crashed attempts used to re-attribute
    # and double-count the previous attempt's dir).
    charged_run_dirs: "set[str]" = set()
    high_spread: Optional[bool] = None   # learned from attempt 1's trace
    skip_polish_b3 = False               # set by the B3-FLOOR-EXIT break
    _SIG_STATE["final_output"] = final_output
    start = time.monotonic()
    i = 0
    while i < max_attempts:
        elapsed_total = time.monotonic() - start
        remaining = total_wall - elapsed_total
        if remaining < attempt_floor:
            print(f"[multi-restart] stop: remaining {remaining:.0f}s < floor "
                  f"{attempt_floor:.0f}s", flush=True)
            break
        _trunc, _why = should_skip_truncated(attempts, remaining)
        if _trunc:
            print(f"[multi-restart] stop: {_why} — a truncated attempt cannot "
                  f"beat the full-stack incumbent; exiting saves gamma "
                  f"(measured record-run evidence)", flush=True)
            break
        # Cost guard: respect the eval's $1/benchmark hard cap. Stop launching
        # once spend would risk exceeding cost_cap (each attempt ~$0.1-0.35).
        if cost_so_far >= cost_cap:
            print(f"[multi-restart] stop: cost ${cost_so_far:.2f} >= cap "
                  f"${cost_cap:.2f}", flush=True)
            break
        # C1-T1 β circuit-breaker: predictive cumulative-ceiling gate. The
        # retrospective cap above launched attempt N at $0.84 spent with a
        # fresh $0.75 in-attempt allowance — the observed zeroing shape
        # ($1.00 cumulative zeroes the benchmark; a rehearsal hit cum $0.76).
        _launch, _allowance, _cost_why = cost_gate(attempts, cost_so_far,
                                                   cost_ceiling)
        if not _launch:
            print(f"[multi-restart] stop (cost breaker): {_cost_why}; "
                  f"shipping banked best", flush=True)
            break
        i += 1
        # Restart split (default off): cap only attempt 1's wall by size class,
        # so a second draw can fire on small and medium designs.  Attempts 2
        # and later get the remaining budget unchanged, and the wrapper's own
        # accounting is untouched — only the slice handed to the agent changes.
        budget_i = remaining
        if split_aware and i == 1:
            _cls = classify_design(input_dcp)
            budget_i = min(remaining,
                           attempt1_budget(total_wall, _cls, attempt_floor))
            if budget_i < remaining:
                print(f"[multi-restart] split-aware: attempt 1 capped to "
                      f"{budget_i:.0f}s (size class {_cls}; D4 restart split)",
                      flush=True)
        # PID-suffixed so two wrappers on the same machine (parallel benchmarks
        # sharing a stem, or a retried harness) can't clobber each other's
        # attempt outputs in /tmp.
        out_i = Path("/tmp") / f"mr_{stem}_{os.getpid()}_{i}.dcp"
        print(f"[multi-restart] attempt {i}: budget {budget_i:.0f}s -> {out_i}",
              flush=True)
        cmd = _attempt_cmd(input_dcp, out_i, budget_i, ils_polish,
                           wall_handback=wall_handback,
                           llm_cost_budget=_allowance)
        _SIG_STATE["current_attempt_out"] = out_i
        # Snapshot-diff run-dir attribution, in the same base directory the
        # agent will write to, so an overridden base is honoured.  Only a
        # directory created by this attempt is attributed; if none appeared,
        # the result is None rather than the previous attempt's directory.
        _rd_base = _wrapper_run_dir_base(repo)
        _before = _snapshot_run_dirs(_rd_base)
        _run_attempt_process(cmd, repo, budget_i, i)
        rd = _attribute_run_dir(_rd_base, _before)
        m = (_read_run_metrics(rd) if rd
             else {"fmax": None, "status": None, "elapsed": None,
                   "cost": None})
        attempt_cost = m.get("cost")
        cost_estimated = False
        if attempt_cost is None:
            # An attempt left no readable cost record — charge the predictive
            # estimate rather than zero.  A silent zero here is a breaker
            # bypass: a crash after real spend would reset the meter.  This
            # must NOT be gated on the breaker being armed: `cost_so_far` is
            # also the meter the retrospective cost_cap guard above reads, so
            # gating it on `_allowance is not None` made `--cost-ceiling 0`
            # (the documented ceiling kill switch) blind the independent
            # $1.00/benchmark cap as well.  A genuinely free attempt still
            # writes cost=0.0, which is not None and is charged nothing.
            attempt_cost = _estimate_attempt_cost(attempts)
            cost_estimated = True
            print(f"[multi-restart] attempt {i}: no readable cost record "
                  f"(run_dir={'missing' if rd is None else rd.name}); "
                  f"charging predictive estimate ${attempt_cost:.2f}",
                  flush=True)
        _rd_key = str(rd) if rd is not None else None
        if attempt_cost:
            if _rd_key is not None and _rd_key in charged_run_dirs:
                print(f"[multi-restart] run dir already charged; skipping "
                      f"duplicate charge for {rd.name}", flush=True)
            else:
                cost_so_far += attempt_cost
                if _rd_key is not None:
                    charged_run_dirs.add(_rd_key)
        # Measured unconditionally: the B3-floor-exit's high-spread guard
        # below is a safety check, not a lever, and gating the measurement on
        # --spread-aware (which no ship target passes) left that guard reading
        # None on every scored run.  --spread-aware still gates the only
        # BEHAVIOUR this number drives, the early-stop floor.
        if high_spread is None and rd is not None:
            _sp = _read_run_spread(rd)
            if _sp is not None:
                high_spread = _sp >= HIGH_SPREAD_TILES
                _floor = (HIGH_SPREAD_MIN_ATTEMPTS
                          if (spread_aware and high_spread) else 2)
                print(f"[multi-restart] spread={_sp:.1f} tiles -> "
                      f"high_spread={high_spread} (early-stop floor "
                      f"{_floor}"
                      f"{'' if spread_aware else '; --spread-aware off'})",
                      flush=True)
        # rec carries the CHARGED cost (measured or predictive estimate) so the
        # cost_gate's max-prior predictive estimate stays conservative;
        # cost_estimated flags the difference for forensics.
        rec = {"i": i, "fmax": m["fmax"], "status": m["status"],
               "cost": attempt_cost, "cost_estimated": cost_estimated,
               "elapsed": m.get("elapsed"),
               "initial_fmax": m.get("initial_fmax"),
               "output": str(out_i),
               "exists": out_i.exists(), "run_dir": str(rd) if rd else None}
        attempts.append(rec)
        print(f"[multi-restart] attempt {i} -> fmax={m['fmax']} "
              f"status={m['status']} cost=${attempt_cost}"
              f"{' (estimated)' if cost_estimated else ''} "
              f"cum_cost=${cost_so_far:.2f} exists={out_i.exists()}", flush=True)
        # Refresh the SCORED location after every attempt (beta rules: "the
        # last best result on disk is what will be scored"). Without this, a
        # harness-side wall kill mid-attempt-N would discard attempt-(N-1)'s
        # valid result because the copy only happened after the loop.
        best_so_far = select_best(attempts)
        if best_so_far is not None:
            if _atomic_publish(best_so_far["output"], final_output):
                _SIG_STATE["published_tier"] = ship_tier(
                    best_so_far.get("status"))
                print(f"[multi-restart] refreshed scored output "
                      f"(attempt {best_so_far['i']}, fmax={best_so_far['fmax']})"
                      f" -> {final_output}", flush=True)
        # Prune superseded attempt DCPs as soon as a better one is banked, so
        # peak /tmp use stays flat in the number of attempts.  Two files are
        # never touched: the selected best (still the publish source on a
        # later refresh) and _SIG_STATE's current_attempt_out, which the
        # signal handler may verify and publish.
        _keep = {best_so_far["output"] if best_so_far else None,
                 str(_SIG_STATE.get("current_attempt_out") or "")}
        for _a in attempts:
            if _a["output"] not in _keep:
                _discard_scratch_dcp(_a["output"])
        # Floor-exit (default off in code, armed by the Makefile like the other
        # levers): the agent attests, via a run-directory token, that this
        # attempt's shipped best is the deterministic recipe floor, with only a
        # sub-noise stochastic tail beyond it.  A further attempt would re-roll
        # only that tail, reproducing the same Fmax while billing another
        # attempt's wall and spend for no gain.  So stop the attempt loop here
        # and skip the winner polish, which measures no-gain from this floor.
        _b3_exit_on = os.environ.get("FPL26_B3_FLOOR_EXIT", "0").strip(
            ).lower() in ("1", "true", "on", "yes")
        # The attesting attempt must also BE the run's selected best.  The
        # token attests attempt-level determinism only, so without this check
        # an on-floor later attempt could stop the loop and skip polish while
        # an earlier attempt stands as proof that the design has stochastic
        # upside beyond the floor.  Requiring identity with the selected best —
        # not merely equality within an epsilon band — also prevents stopping
        # the loop and cancelling polish on a different attempt's artifact,
        # which the attestation does not cover.  High-spread designs never
        # break here: they need their draws.
        if (_b3_exit_on and rd is not None
                and (rd / "b3_floor_saturated.token").exists()
                and str(m.get("status") or "").startswith("VALID_OPTIMIZED")
                and best_so_far is not None
                and best_so_far.get("i") == i
                and not bool(high_spread)):
            skip_polish_b3 = True
            print(f"[multi-restart] B3-FLOOR-EXIT: attempt {i} shipped the "
                  f"deterministic recipe floor (b3_floor_saturated.token) "
                  f"and IS the run's best; stopping the attempt loop and "
                  f"skipping winner-polish — a further attempt re-rolls only "
                  f"a stochastic tail that measured <=0.004ns, and the "
                  f"wall/cost it bills is pure gamma/beta.", flush=True)
            break
        if should_stop_early(attempts,
                             high_spread=bool(spread_aware and high_spread)):
            print(f"[multi-restart] stop: strong result confirmed by >=2 "
                  f"attempts after {i} runs", flush=True)
            break

    best = select_best(attempts)
    # The loop is over, so nothing is in flight any more.  Left set, a signal
    # landing after the winner polish has replaced the scored file could
    # verify the last attempt's artifact instead and overwrite the polished
    # winner through the tier-0 override.  Clearing it makes the handler keep
    # whatever is on disk from here on.
    _SIG_STATE["current_attempt_out"] = None
    if best is not None:
        if _atomic_publish(best["output"], final_output):
            _SIG_STATE["published_tier"] = ship_tier(best.get("status"))
        print(f"[multi-restart] BEST = attempt {best['i']} fmax={best['fmax']} "
              f"-> {final_output}", flush=True)
        # FIRING RECORD for the tier fix: name the attempt the OLD
        # fmax-first key would have shipped whenever it differs. Additive line
        # (no existing label changes) so an A/B can count firings by grep.
        _usable = [a for a in attempts
                   if a.get("exists") and a.get("fmax") is not None]
        if _usable:
            _old = max(_usable, key=lambda a: (
                a["fmax"], 1 if a.get("status") == "VALID_OPTIMIZED" else 0,
                -a["i"]))
            if _old["i"] != best["i"]:
                print(f"[multi-restart] TIER-FIX FIRED: shipped attempt "
                      f"{best['i']} (tier={ship_tier(best.get('status'))} "
                      f"status={best.get('status')} fmax={best['fmax']}) "
                      f"INSTEAD OF attempt {_old['i']} "
                      f"(tier={ship_tier(_old.get('status'))} "
                      f"status={_old.get('status')} fmax={_old['fmax']})",
                      flush=True)
        if polish and skip_polish_b3:
            print("[multi-restart] winner-polish skipped (B3-FLOOR-EXIT): "
                  "phys_opt from the attested floor measured NO_GAIN 5/5; "
                  "its budget would be pure gamma.", flush=True)
        if polish and not skip_polish_b3:
            leftover = total_wall - (time.monotonic() - start)
            # Restate the summary when the polish moved the artifact.
            #
            # The winner polish replaces the scored checkpoint after the agent
            # has already printed its summary, so on an improvement that
            # summary understates what shipped.  No score is lost — the better
            # checkpoint is what gets scored — but every downstream comparison
            # parses that block, so the rows read low by more than the
            # measurement noise floor, which is enough to invert a comparison.
            #
            # Restate it here with the same labels the harness greps, so a
            # tail of the log picks the corrected values up automatically.
            # Print-only: nothing about the shipped artifact changes.
            _polish: dict = {}
            if winner_polish(final_output, leftover, repo, report=_polish):
                _delta = _polish.get("delta_ns")
                _corrected = polish_corrected_fmax(best.get("fmax"), _delta)
                if _corrected is None:
                    print("[multi-restart] polish: artifact improved but the "
                          "corrected Fmax could not be derived; the summary "
                          "above understates what shipped.", flush=True)
                elif not best.get("initial_fmax"):
                    # Restating Fmax without the improvement would leave a parser
                    # reading a CORRECTED fmax next to a STALE alpha and treating
                    # the pair as consistent. Emit neither; say so instead.
                    print(f"[multi-restart] polish: artifact improved by "
                          f"{_delta:.3f} ns but the initial Fmax is unknown, so "
                          f"the summary cannot be restated consistently; it "
                          f"understates what shipped.", flush=True)
                else:
                    _init = best["initial_fmax"]
                    print(f"[multi-restart] polish: the summary above predates "
                          f"this polish and understates the shipped DCP by "
                          f"{_delta:.3f} ns of WNS. Corrected:", flush=True)
                    print(f"  Best Fmax: {_corrected:.2f}", flush=True)
                    print(f"  Initial Fmax: {_init:.2f}", flush=True)
                    print(f"  Fmax Improvement: "
                          f"{_corrected - _init:+.2f}", flush=True)
                    summary_correction = {
                        "polish_delta_ns": _delta,
                        "fmax_before_polish": best.get("fmax"),
                        "fmax_after_polish": round(_corrected, 4),
                    }
                    # Copy, so `attempts` keeps what each ATTEMPT produced while
                    # `chosen` reports what actually SHIPPED. Conflating the two
                    # is how the stale number spread in the first place.
                    best = dict(best)
                    best["fmax"] = round(_corrected, 4)
                    best["polish_correction"] = summary_correction
    else:
        print("[multi-restart] WARNING: no usable attempt output", flush=True)

    # The scored artifact is on disk and the polish has run, so no attempt's
    # /tmp copy can be published again.  `attempts[].exists` keeps recording
    # what each attempt PRODUCED; it was never a claim about /tmp afterwards.
    if final_output.exists():
        for _a in attempts:
            _discard_scratch_dcp(_a["output"])

    summary = {"input": str(input_dcp), "final_output": str(final_output),
               "attempts": attempts, "chosen": best,
               "total_wall": total_wall,
               # Cost-breaker forensics: cumulative KNOWN spend + active ceiling.
               "llm_cost_total": round(cost_so_far, 4),
               "cost_ceiling": cost_ceiling,
               # Lets main()'s internal budgeted fallback size its
               # wall slice from what the attempt loop actually consumed.
               "wall_elapsed_s": round(time.monotonic() - start, 1)}
    try:
        out_dir = repo / ".planning_baseline"
        out_dir.mkdir(exist_ok=True)
        (out_dir / f"mr_summary_{stem}.json").write_text(
            json.dumps(summary, indent=2))
    except Exception:
        pass
    return summary


def _run_internal_fallback(input_dcp: Path, final_output: Path,
                           total_wall: float, ils_polish: bool,
                           cost_ceiling: Optional[float],
                           cost_so_far: float, elapsed_s: float,
                           repo: Path) -> int:
    """Run one budget-aware fallback attempt inside the wrapper.

    The remaining LLM budget is the ceiling minus accumulated cost, with a
    minimum of $0.01 required by the agent. A missing or nonpositive ceiling
    omits the budget and uses the agent default. The return code is zero only
    when a valid output exists after the attempt.
    """
    budget: Optional[float] = None
    if cost_ceiling is not None and cost_ceiling > 0:
        budget = max(0.01, int((cost_ceiling - cost_so_far) * 100) / 100.0)
    remaining = max(60.0, total_wall - elapsed_s)
    cmd = _attempt_cmd(input_dcp, final_output, remaining, ils_polish,
                       llm_cost_budget=budget)
    print(f"[multi-restart] no usable attempt output — internal budgeted "
          f"fallback: one contest-mode run, wall {remaining:.0f}s, LLM "
          f"budget "
          f"{'agent-default' if budget is None else '$%.2f' % budget}",
          flush=True)
    try:
        subprocess.run(cmd, cwd=str(repo))
    except Exception as e:
        print(f"[multi-restart] internal fallback failed to launch: {e}",
              flush=True)
    return 0 if final_output.exists() else 1


def _default_cost_ceiling() -> float:
    """C1-T1: --cost-ceiling default, overridable via FPL26_COST_CEILING
    (unparseable env falls back to the shipped default, never crashes the
    contest entrypoint)."""
    raw = os.environ.get("FPL26_COST_CEILING", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            print(f"[multi-restart] WARNING: FPL26_COST_CEILING={raw!r} not "
                  f"a float; using default {COST_CEILING_DEFAULT}", flush=True)
    return COST_CEILING_DEFAULT


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_dcp", type=Path)
    ap.add_argument("--final-output", type=Path, default=None,
                    help="Where to write the best DCP. Default: "
                         "<input_dir>/<stem>_optimized.dcp (the location the "
                         "eval harness scans for).")
    ap.add_argument("--total-wall", type=float, default=3500.0)
    ap.add_argument("--attempt-floor", type=float, default=1200.0,
                    help="Don't start a new attempt with less than this many s "
                         "left. Default 1200s: an attempt needs ~setup(340s) + "
                         "phys_opt(600s) + finalize_reserve(300s) of budget or "
                         "it does place+route then budget-skips phys_opt and "
                         "falls back to baseline (observed in the 1400s "
                         "validation). At the real 3500s budget this still "
                         "allows ~2-3 full attempts on early-stopping designs.")
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--cost-cap", type=float, default=COST_CAP_DEFAULT,
                    help="Stop launching attempts once cumulative OpenRouter "
                         "cost reaches this ($). Respects the eval's $1/"
                         "benchmark hard cap.")
    ap.add_argument("--cost-ceiling", type=float,
                    default=_default_cost_ceiling(),
                    help="Beta circuit-breaker: hard CUMULATIVE LLM-spend "
                         "ceiling ($) across attempts, enforced BEFORE launch "
                         "with a predictive estimate (max prior attempt cost) "
                         "and passed down as a shrinking per-attempt budget "
                         "(ceiling − spent). Default 0.80 (the eval ZEROES a "
                         "benchmark at $1.00 cumulative). "
                         "Env default: FPL26_COST_CEILING. Kill switch: 0.")
    ap.add_argument("--repo", type=Path,
                    default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--ils-polish", action="store_true",
                    help="Pass ILS=1 to each attempt: enables the in-agent "
                         "ILS ruin-and-recreate polish stage (AWS-validated: "
                         "v2 +24.7, vtr +17.4, 3d +14.4 MHz; "
                         "never-worse).")
    ap.add_argument("--no-winner-polish", action="store_true",
                    help="Disable the never-worse phys_opt polish of the "
                         "winner on wall stranded below the attempt floor.")
    ap.add_argument("--spread-aware", action="store_true",
                    help="EXPERIMENTAL (default off, needs A/B): on high-spread "
                         "designs (>=100 tiles, the Explore band) require a "
                         "3rd attempt before early-stop — bimodal draws "
                         "(corescore-class 0/0/+85) defeat 2-confirmation "
                         "stopping.")
    ap.add_argument("--split-aware", action="store_true",
                    help="EXPERIMENTAL (default off, needs A/B): "
                         "cap attempt 1's MAX_WALL by DCP-size class (small "
                         "1800s / medium 2400s / boom-class uncapped) so a "
                         "second draw can actually fire — beta evidence: "
                         "attempt 2 never launched on 4/5 designs because "
                         "attempt 1 consumed ~the whole 3500s budget.")
    ap.add_argument("--wall-handback", action="store_true",
                    help="EXPERIMENTAL (default off, needs A/B): "
                         "pass WALL_HANDBACK=1 to each attempt so a "
                         "saturated agent (ILS no-improve stop / LASTMILE "
                         "reject / budget-kill, with a banked accept) "
                         "finalizes early and returns its unused wall to "
                         "this wrapper (gamma saved or a second draw "
                         "funded — composes with --split-aware).")
    a = ap.parse_args(argv)
    final_output = a.final_output
    if final_output is None:
        final_output = a.input_dcp.parent / f"{a.input_dcp.stem}_optimized.dcp"
    # Harness wall kill (SIGTERM at 3600s) must never strand the in-flight
    # attempt's emergency DCP in /tmp with nothing at the scored location.
    _install_signal_publisher()
    s = run(a.input_dcp, final_output, a.total_wall, a.attempt_floor,
            a.max_attempts, a.repo, a.cost_cap, ils_polish=a.ils_polish,
            polish=not a.no_winner_polish, spread_aware=a.spread_aware,
            split_aware=a.split_aware, wall_handback=a.wall_handback,
            cost_ceiling=a.cost_ceiling)
    if s["chosen"] is not None:
        return 0
    # No-usable-output path: the wrapper, which knows the cumulative spend,
    # runs the last-resort contest-mode attempt itself with the remaining
    # budget.  Exit non-zero only if even that produced no output, at which
    # point the Makefile's fallback fires as a pure pre-interpreter-crash
    # safety net on a small fixed budget.
    return _run_internal_fallback(
        a.input_dcp, final_output, a.total_wall, a.ils_polish,
        a.cost_ceiling, float(s.get("llm_cost_total") or 0.0),
        float(s.get("wall_elapsed_s") or 0.0), a.repo)


if __name__ == "__main__":
    sys.exit(main())
