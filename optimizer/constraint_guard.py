"""Timing-constraint integrity guard.

Contest rules make edits to the timing constraints disqualifying, and a
disqualification is a zero, which outranks any amount of gain.  The stack
otherwise has no defence: the model has unrestricted raw Tcl, the risk
classifier reads Tcl only to estimate runtime rather than legality, the
forbidden-token list validates advisory labels rather than commands, and the
submission validator checks slack and routing state but never whether the
constraints themselves were altered.  A false-path exception on the critical
path would inflate the reported slack, pass every gate, and be a hard
disqualification.

Honest scope: this is a latent hole, not an observed leak.  A search across
every historical run log found no constraint-modifying command ever issued
inside a raw-Tcl payload.  The guard exists because the downside is
catastrophic and the guard is cheap.

Two layers:

  1. PREVENTION — a deny-list checked at the raw-Tcl boundary.  It refuses the
     command and tells the model why, so it can pick a legal move.
  2. DETECTION — a constraint fingerprint captured after the checkpoint is
     opened and re-checked before ship.

Detection is the load-bearing layer.  Prevention can be bypassed through an
alias, a sourced file or an eval; deny-lists are string matching and string
matching is defeatable, while a fingerprint diff is not.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# Commands that create, delete or relax timing constraints. Ordered roughly by
# how attractive each is as a "cheat" — the exception commands first.
CONSTRAINT_MUTATING_COMMANDS = (
    "set_false_path",
    "set_multicycle_path",
    "set_max_delay",
    "set_min_delay",
    "set_clock_uncertainty",
    "set_clock_latency",
    "set_clock_groups",
    "set_disable_timing",
    "set_case_analysis",
    "create_clock",
    "create_generated_clock",
    "remove_clock",
    "set_input_delay",
    "set_output_delay",
    "set_data_check",
    "set_system_jitter",
    "set_external_delay",
    "read_xdc",
    "source",          # an XDC can be sourced from a file
)

# `report_*` and `get_*` forms are read-only and must stay allowed, otherwise
# the guard would break ordinary timing analysis.
_READONLY_PREFIXES = ("report_", "get_", "all_", "check_", "write_")


@dataclass
class ConstraintFingerprint:
    """A comparable summary of the design's timing constraints."""
    clock_count: int = 0
    clock_digest: str = ""
    exception_count: int = 0
    exception_digest: str = ""
    captured: bool = False
    note: str = ""

    def differs_from(self, other: "ConstraintFingerprint") -> Tuple[bool, str]:
        """Returns (changed, human-readable reason)."""
        if not (self.captured and other.captured):
            # Fail LOUD but OPEN: an uncapturable fingerprint must not brick a
            # run, but it must be visible. (Ship-blocking on an unmeasurable
            # property would turn a monitoring gap into a lost submission.)
            return False, ("fingerprint unavailable "
                           f"(before={self.captured} after={other.captured}) "
                           "— constraint integrity UNVERIFIED")
        diffs = []
        if self.clock_count != other.clock_count:
            diffs.append(f"clock count {self.clock_count} -> {other.clock_count}")
        elif self.clock_digest != other.clock_digest:
            diffs.append("clock definitions changed")
        if self.exception_count != other.exception_count:
            diffs.append(f"timing exceptions {self.exception_count} -> "
                         f"{other.exception_count}")
        elif self.exception_digest != other.exception_digest:
            diffs.append("timing exception set changed")
        if diffs:
            return True, "; ".join(diffs)
        return False, "constraints unchanged"


def is_constraint_mutating(command: Optional[str]) -> Tuple[bool, str]:
    """True iff the Tcl payload would create/alter/remove timing constraints.

    Read-only report/get forms are explicitly allowed. Matching is on word
    boundaries so ``report_clocks`` never trips ``create_clock``.
    """
    if not command:
        return False, ""
    text = str(command)
    # Strip Tcl comments — a command name inside a comment is not a command.
    text = re.sub(r"#[^\n]*", " ", text)
    low = text.lower()
    for cmd in CONSTRAINT_MUTATING_COMMANDS:
        if not re.search(rf"(?<![\w-]){re.escape(cmd)}(?![\w-])", low):
            continue
        # Allow read-only relatives (e.g. `report_clocks`, `get_clocks`).
        idx = low.find(cmd)
        prefix = low[max(0, idx - 12):idx]
        if any(p in prefix for p in _READONLY_PREFIXES):
            continue
        return True, cmd
    return False, ""


def deny_reason(cmd: str) -> str:
    """Message returned to the model in place of executing the command."""
    return (
        f"REFUSED: '{cmd}' modifies timing constraints. Editing the XDC / "
        f"timing constraints is DISQUALIFYING under the contest rules — the "
        f"design must be sped up physically (placement, routing, phys_opt, "
        f"retiming), not by relaxing what is being measured. Pick a physical "
        f"optimisation instead."
    )


def _digest(lines: Sequence[str]) -> str:
    """Order-independent digest so a re-ordered report is not a false alarm."""
    norm = sorted(" ".join(str(l).split()) for l in lines if str(l).strip())
    return hashlib.sha256("\n".join(norm).encode("utf-8", "replace")).hexdigest()[:16]


def parse_clock_report(text: Optional[str]) -> List[str]:
    """Extract clock name/period pairs from `report_clocks` output."""
    if not text:
        return []
    out = []
    for line in str(text).splitlines():
        s = line.strip()
        if not s or s.startswith(("#", "-", "=")):
            continue
        # typical: "clk_fpl26contest   2.000   0.000 1.000   ..."
        m = re.match(r"^([\w/\[\]\.\$-]+)\s+(-?\d+\.\d+)", s)
        if m:
            out.append(f"{m.group(1)} {m.group(2)}")
    return out


def parse_exception_report(text: Optional[str]) -> List[str]:
    """Extract timing exceptions from `report_exceptions` output."""
    if not text:
        return []
    out = []
    for line in str(text).splitlines():
        s = line.strip()
        if not s or s.startswith(("#", "-", "=")):
            continue
        if any(k in s.lower() for k in ("false_path", "multicycle", "max_delay",
                                        "min_delay", "case_analysis",
                                        "disable_timing")):
            out.append(s)
    return out


def build_fingerprint(clock_report: Optional[str],
                      exception_report: Optional[str]) -> ConstraintFingerprint:
    """Build a constraint fingerprint from clock and exception reports.

    If either report is `None`, the result has `captured=False`; partial
    fingerprints are not comparable and must produce an `UNVERIFIED` result
    rather than a change warning. This guard fails open because a missing
    report cannot distinguish absent measurements from deleted constraints.
    """
    fp = ConstraintFingerprint()
    if clock_report is None or exception_report is None:
        missing = []
        if clock_report is None:
            missing.append("clocks")
        if exception_report is None:
            missing.append("exceptions")
        fp.note = f"report(s) unavailable: {', '.join(missing)}"
        return fp
    clocks = parse_clock_report(clock_report)
    excs = parse_exception_report(exception_report)
    fp.clock_count = len(clocks)
    fp.clock_digest = _digest(clocks)
    fp.exception_count = len(excs)
    fp.exception_digest = _digest(excs)
    fp.captured = True
    return fp
