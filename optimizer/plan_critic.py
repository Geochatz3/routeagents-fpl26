"""Cheap pre-flight PLAN CRITIC — a second opinion before a heavy Vivado move.

WHY THIS EXISTS (jul25 panel, 4 of 5 seats converged on this independently):

Today we run ONE model (``x-ai/grok-4.3``) in a loop with no second opinion of
any kind — no reviewer, no verifier, no critic, no self-check.  The panel's
measured indictment:

    design      our gap    API calls   LLM spend   % of ~$1 budget
    boom_soc    -22.15     16          $0.357      36 %
    ispd16       -6.78     26          $0.226      23 %
    rend3d      -10.06     58          $0.689      69 %

The design we lose the most on made the FEWEST calls and spent the LEAST
budget.  Since score = ``alpha - 0.1*alpha*beta - 0.1*alpha*gamma`` and beta
caps at 10 % of alpha, unspent LLM budget is not saved money — it is unused
deliberation.  A heavy Vivado move costs 600-1200 s of the eval hour; a critic
call costs ~$0.01-0.05 and a few seconds.  Spending a cent to avoid wasting ten
minutes of wall is the trade this module makes.

NEVER-WORSE POSTURE — read before changing:
The critic is **ADVISORY ONLY**.  It cannot veto, cannot cancel, and cannot
rewrite an action.  Its output is injected as a note the planner may ignore.
A vetoing critic could talk the planner out of a winning move and would not be
never-worse; the panel's own rule was to prefer proposals that cannot regress.
The stricter "run both and let the MUX decide" variant was rejected here because
it doubles heavy Vivado wall, which the 1-hour eval does not have.

FREE PARAMETERS: three (``enabled``, ``max_calls``, ``beta_headroom_frac``).
No decision boundary is fitted to any design outcome.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

# Actions worth a second opinion: the ones that cost real wall. Mirrors the
# established "risky Tcl" notion (a heavy structural move), NOT a new cut.
HEAVY_ACTION_TOKENS = (
    "place_design",
    "route_design",
    "phys_opt_design",
)

VERDICT_APPROVE = "APPROVE"
VERDICT_CONCERN = "CONCERN"
VERDICT_UNPARSED = "UNPARSED"

# Deliberately small: the critic must be cheap enough that its beta cost is
# noise against a 10 %-of-alpha cap.
CRITIC_MAX_TOKENS = 700

CRITIC_SYSTEM = (
    "You are a senior FPGA timing-closure engineer reviewing ONE proposed "
    "Vivado action before it consumes 10-20 minutes of a 1-hour budget. "
    "You are ADVISORY: you cannot veto, and the planner may ignore you. "
    "Be blunt and specific; a vague 'looks fine' is worthless.\n\n"
    "Device is xcvu3p (UltraScale+, NOT Versal). Retiming is legal. "
    "Pipeline insertion and XDC timing-constraint edits are DISQUALIFYING — "
    "if the proposed action edits constraints, say so first.\n\n"
    "Answer in at most 5 lines:\n"
    "VERDICT: APPROVE or CONCERN\n"
    "WHY: one sentence.\n"
    "BETTER: one concrete alternative Vivado action, or 'none'."
)


@dataclass
class CriticVerdict:
    verdict: str = VERDICT_UNPARSED
    why: str = ""
    better: str = ""
    raw: str = ""

    @property
    def is_concern(self) -> bool:
        return self.verdict == VERDICT_CONCERN

    def advisory_note(self) -> str:
        """The text injected into the planner's context. Advisory framing is
        deliberate — the planner must feel free to proceed."""
        if self.verdict == VERDICT_UNPARSED:
            return ""
        head = ("PRE-FLIGHT REVIEW (advisory, from a second model; you may "
                "proceed anyway — you own the decision):")
        body = f"  verdict: {self.verdict}"
        if self.why:
            body += f"\n  why: {self.why}"
        if self.better and self.better.strip().lower() not in ("none", "n/a"):
            body += f"\n  alternative worth considering: {self.better}"
        return f"{head}\n{body}"


def is_heavy_action(tool_name: str, arguments: Optional[dict]) -> bool:
    """True iff this call is expensive enough to be worth reviewing."""
    if tool_name in ("vivado_place_design", "vivado_route_design",
                     "vivado_phys_opt_design"):
        return True
    if tool_name == "vivado_run_tcl" and arguments:
        cmd = str(arguments.get("command")
                  or arguments.get("tcl_command") or "").lower()
        return any(tok in cmd for tok in HEAVY_ACTION_TOKENS)
    return False


def should_critique(
    *,
    enabled: bool,
    tool_name: str,
    arguments: Optional[dict],
    calls_made: int,
    max_calls: int,
    spent_usd: float,
    budget_usd: Optional[float],
    beta_headroom_frac: float,
    remaining_s: float,
    min_remaining_s: float,
) -> Tuple[bool, str]:
    """Pure firing gate — no I/O, fully unit-testable.

    Fails CLOSED (no critique) on anything unmeasurable: a critic that fires
    when the budget is unknown could push beta past its cap, and beta is
    subtracted from score.
    """
    if not enabled:
        return False, "disabled (default OFF)"
    if not is_heavy_action(tool_name, arguments):
        return False, f"{tool_name} is not a heavy action"
    if calls_made >= max_calls:
        return False, f"critic call cap reached ({calls_made}/{max_calls})"
    if remaining_s < min_remaining_s:
        return False, (f"only {remaining_s:.0f}s wall left "
                       f"(< {min_remaining_s:.0f}s) — spend it on Vivado")
    if budget_usd is None or budget_usd <= 0:
        return False, "LLM budget unknown — fail closed"
    # Only review while there is genuine beta headroom. This is the panel's
    # "unspent budget is the bug" point made operational: we critique BECAUSE
    # the budget is underused, and we stop as soon as it is not.
    if spent_usd >= budget_usd * beta_headroom_frac:
        return False, (f"beta headroom exhausted (${spent_usd:.3f} >= "
                       f"{beta_headroom_frac:.0%} of ${budget_usd:.2f})")
    return True, (f"armed (${spent_usd:.3f} of ${budget_usd:.2f} spent; "
                  f"call {calls_made + 1}/{max_calls})")


def build_critic_prompt(
    *,
    tool_name: str,
    arguments: Optional[dict],
    initial_wns: Optional[float],
    current_wns: Optional[float],
    failing_endpoints: Optional[int],
    router_rule: Optional[str],
    router_why: Optional[str],
    tried: Optional[Sequence[str]] = None,
    remaining_s: Optional[float] = None,
) -> str:
    """Compact user prompt. Kept small on purpose — this must stay cheap."""
    args_txt = ""
    if arguments:
        cmd = arguments.get("command") or arguments.get("tcl_command")
        args_txt = str(cmd) if cmd else str(arguments)
    lines = [
        "PROPOSED ACTION:",
        f"  {tool_name}: {args_txt[:400]}",
        "",
        "DESIGN STATE:",
        f"  initial WNS: {initial_wns}",
        f"  current best WNS: {current_wns}",
        f"  failing endpoints: {failing_endpoints}",
    ]
    if remaining_s is not None:
        lines.append(f"  wall remaining: {remaining_s:.0f}s")
    if router_rule:
        lines += ["", f"ROUTER CHOSE: {router_rule}"]
        if router_why:
            lines.append(f"  rationale: {str(router_why)[:300]}")
    if tried:
        lines += ["", "ALREADY TRIED THIS RUN:"]
        lines += [f"  - {t}" for t in list(tried)[-8:]]
    lines += ["", "Is this the best use of the remaining wall? Answer in the "
                  "5-line format."]
    return "\n".join(lines)


_VERDICT_RE = re.compile(r"VERDICT:\s*(APPROVE|CONCERN)", re.I)
_WHY_RE = re.compile(r"WHY:\s*(.+)", re.I)
_BETTER_RE = re.compile(r"BETTER:\s*(.+)", re.I)


def parse_critic_verdict(text: Optional[str]) -> CriticVerdict:
    """Parse the critic reply. An unparseable reply is UNPARSED and yields an
    EMPTY advisory note — a malformed second opinion must never perturb the
    planner's context."""
    v = CriticVerdict(raw=(text or "")[:2000])
    if not text:
        return v
    m = _VERDICT_RE.search(text)
    if m:
        v.verdict = m.group(1).upper()
    mw = _WHY_RE.search(text)
    if mw:
        v.why = mw.group(1).strip()[:300]
    mb = _BETTER_RE.search(text)
    if mb:
        v.better = mb.group(1).strip()[:300]
    return v
