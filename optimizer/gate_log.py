"""Provides a typed, append-only ledger for optimizer gate decisions.

The ledger distinguishes inapplicable, refused, executed-invalid, and
valid-unpromoted work so decisions can be aggregated mechanically. It is
disabled by default and becomes a no-op when unconfigured. Emission never
raises, flushes each row, and performs no additional tool calls, predicate
evaluation, or hashing. Verdicts are ``ALLOW``, ``REFUSE``, or ``OBSERVE``.
Provenance is ``HISTORY``, ``MODEL``, ``CONSTANT``, ``MEASURED_ANCHOR``, or
``UNKNOWN``. Refusals from modeled or constant estimates remain distinguishable
from refusals bounded by direct measurements.
"""
import json
import os
import threading
import time

_LOCK = threading.Lock()
_PATH_CACHE = []          # one-element cache: the resolved path, or None
_SCHEMA_VERSION = 1

VERDICT_ALLOW = "ALLOW"
VERDICT_REFUSE = "REFUSE"
VERDICT_OBSERVE = "OBSERVE"

PROV_HISTORY = "HISTORY"
PROV_MODEL = "MODEL"
PROV_CONSTANT = "CONSTANT"
PROV_MEASURED_ANCHOR = "MEASURED_ANCHOR"
PROV_UNKNOWN = "UNKNOWN"


def enabled() -> bool:
    """True only when FPL26_GATE_LOG names a writable path."""
    return bool(_resolve())


def _resolve():
    if _PATH_CACHE:
        return _PATH_CACHE[0]
    raw = (os.environ.get("FPL26_GATE_LOG") or "").strip()
    path = raw or None
    if path:
        try:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
        except Exception:
            path = None
    _PATH_CACHE.append(path)
    return path


def reset_for_test():
    """Tests only — clears the memoized path so env changes take effect."""
    _PATH_CACHE.clear()


def emit(gate, verdict, *, design=None, run_id=None, iteration=None, phase=None,
         reason_code=None, predicted_s=None, observed_s=None, threshold_s=None,
         remaining_wall_s=None, provenance=None, tool=None, wns_ns=None,
         best_wns_ns=None, site=None, extra=None):
    """Append one typed row. Never raises. No-op unless FPL26_GATE_LOG is set.

    `gate` and `verdict` are required and should come from the constants above so the
    resulting file is aggregatable without string cleanup.
    """
    try:
        path = _resolve()
        if not path:
            return
        row = {
            "schema_version": _SCHEMA_VERSION,
            "ts": time.time(),
            "gate": gate,
            "verdict": verdict,
            "reason_code": reason_code,
            "design": design,
            "run_id": run_id,
            "iteration": iteration,
            "phase": phase,
            "tool": tool,
            "predicted_s": predicted_s,
            "observed_s": observed_s,
            "threshold_s": threshold_s,
            "remaining_wall_s": remaining_wall_s,
            "provenance": provenance,
            "wns_ns": wns_ns,
            "best_wns_ns": best_wns_ns,
            "site": site,
        }
        # margin is what the mining step actually sorts on: how badly the prediction
        # missed the bound it was compared against.
        try:
            if predicted_s is not None and threshold_s is not None:
                row["margin_s"] = float(threshold_s) - float(predicted_s)
        except Exception:
            pass
        try:
            if predicted_s and observed_s:
                row["over_estimate_ratio"] = float(predicted_s) / float(observed_s)
        except Exception:
            pass
        if isinstance(extra, dict):
            for k, v in extra.items():
                row.setdefault(k, v)
        line = json.dumps(row, default=str, sort_keys=True)
        with _LOCK:
            with open(path, "a") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
    except Exception:
        # Deliberately silent and total: a telemetry failure must never alter a
        # decision, and must never add log noise that changes LLM context.
        return
