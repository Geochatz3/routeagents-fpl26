"""Typed gate/decision ledger — the jul26 panel's unanimous #1 missing signal.

WHY THIS EXISTS
  jul26 recovered ~+55 score points from ONE defect family: *a prediction was allowed to
  REFUSE work*. All five instances were found BY ACCIDENT, hours each. A 3-seat panel
  (gpt-5.6-sol, kimi-k3, deepseek-v4-pro) independently converged on the same fix: the
  run already logs every decision, but as free text, so decisions cannot be AGGREGATED.
  sol's framing — "it supplies denominators": with typed rows you can separate
  "mechanism never applicable" from "applicable but vetoed" from "executed but invalid"
  from "valid improvement not promoted". The existing 255-row decisions.jsonl cannot.

  It also produces the predicted-vs-observed pairs needed to replace blind constants
  (like the 600 s one that refused a 21 s route_design) with a 2-4 parameter cost model
  under leave-one-design-out. That is FITTING A COST MODEL, which the project's
  methodology invariant permits — as opposed to fitting a decision boundary, which it
  forbids.

DESIGN CONSTRAINTS, because the module being instrumented IS the contest submission
  1. **Default OFF.** With `FPL26_GATE_LOG` unset this module does nothing at all, so
     behaviour is byte-identical to before. The sweep sets it; the eval does not.
  2. **Cannot raise, ever.** `emit()` swallows every exception internally. A logger that
     can break a gate is worse than no logger — so call sites need no try/except and
     there is no code path where a logging failure changes an optimization decision.
  3. **Append-only, flushed per row.** The operator's session can die mid-sweep (this
     box kills jobs when the browser tab closes), so a row must survive the instant it
     is written. Never buffered for assembly at the end.
  4. **No new work.** Every value is copied from state the optimizer already computed.
     No extra Vivado calls, no extra LLM calls, no re-evaluated predicates, no hashing.

TYPED VOCABULARY — the point is that these are enums, not prose.
  verdict:      ALLOW | REFUSE | OBSERVE
  provenance:   HISTORY (measured on this design, this run) | MODEL (size model) |
                CONSTANT (blind) | MEASURED_ANCHOR (published measurement) | UNKNOWN
  A refusal whose provenance is CONSTANT or MODEL is the defect family. A refusal whose
  provenance is HISTORY or MEASURED_ANCHOR is legitimate — it is bounded by something
  real. That single field is what makes the ledger mechanically minable.
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
