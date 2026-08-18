"""Hidden-Design Fingerprint Card v0 — offline advisory prototype.

This module is INTENTIONALLY:

- pure-Python, no Vivado, no MCP, no network, no filesystem,
- design-name-blind — refuses to use any benchmark name as a retrieval
  key, returns ``insufficient_data`` if a name-like key appears,
- offline-only — NEVER mutates ``policy_memory/episode_store.jsonl``,
  the policy-card prompt, or any optimizer control flow,
- advisory-only — outputs an ``Advisory`` dataclass whose
  ``advisory_family`` field is drawn from a frozen allow-list of
  *labels*, never a Vivado / Tcl command string,
- deterministic — same inputs → same advisory; tests pin this.

The intended consumer is a future SHADOW-MODE decision-trace emitter
that logs HFCv0's advisory alongside live runs without showing it to
the LLM.  See ``.planning/session18_hidden_fingerprint_card_design.md``
and ``.planning/session18_shadow_mode_plan.md`` for the full contract.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

# Frozen allow-list of advisory family LABELS.  v0 must never emit
# values outside this set (other than None).  Pinned by tests.
ADVISORY_FAMILIES = frozenset({
    "post_route_phys_opt",
    "eager_mirror",
    "cell_replacement_centroid",
    "scope_re_place",
    "route_directive_sweep",
    "retiming_polish",
    "ship_baseline",
    "high_fanout_replication",
})

# Forbidden tokens — if any of these appear in advisory_family the
# prototype is mis-emitting a recipe command instead of a label.
_FORBIDDEN_COMMAND_TOKENS = (
    " ", "_design", "place_design", "route_design", "phys_opt_design",
    "report_", "[", "]", "{", "}", "$", "vivado_",
)

CARD_BENEFITED = "CARD_BENEFITED"
CARD_NEUTRAL = "CARD_NEUTRAL"
CARD_HARMFUL = "CARD_HARMFUL"
CARD_INCONCLUSIVE = "CARD_INCONCLUSIVE"

# Names of design-name-like keys we REFUSE to consume.
_DESIGN_NAME_KEYS = ("design_name", "design", "benchmark", "name",
                     "design_name_masked")


@dataclass
class EpisodeSummary:
    """Compact record used as a neighbour in HFCv0's nearest-neighbour search."""
    episode_id: str
    lut_count: float
    critical_path_spread: float
    qor_route_bound_score: Optional[float]
    qor_has_congestion_signal: Optional[bool]
    card_verdict: Optional[str]          # CARD_BENEFITED / CARD_NEUTRAL / ...
    final_status: Optional[str] = None    # VALID_OPTIMIZED / ...
    winning_action_family: Optional[str] = None
    # design_name is intentionally NOT a field.


@dataclass
class Advisory:
    silence: bool
    confidence: Optional[str] = None         # "high" | "medium" | None
    advisory_family: Optional[str] = None
    warning: Optional[str] = None
    diagnostic_label: Optional[str] = None
    silence_reason: Optional[str] = None
    negative_memory_hits: List[str] = field(default_factory=list)
    evidence_episodes: List[str] = field(default_factory=list)
    feature_band: Optional[Tuple[str, str, str, str]] = None

    def as_dict(self) -> dict:
        return {
            "silence": self.silence,
            "confidence": self.confidence,
            "advisory_family": self.advisory_family,
            "warning": self.warning,
            "diagnostic_label": self.diagnostic_label,
            "silence_reason": self.silence_reason,
            "negative_memory_hits": list(self.negative_memory_hits),
            "evidence_episodes": list(self.evidence_episodes),
            "feature_band": list(self.feature_band) if self.feature_band else None,
        }


def _lut_bucket(lut: float) -> str:
    if lut < 10_000:
        return "small_lut"
    if lut < 60_000:
        return "mid_lut"
    if lut < 150_000:
        return "large_lut"
    return "huge_lut"


def _spread_bucket(spread: float) -> str:
    if spread < 50:
        return "low_spread"
    if spread < 200:
        return "mid_spread"
    return "high_spread"


def _route_bucket(score: Optional[float]) -> str:
    if score is None or score < 0.20:
        return "route_bound_low"
    if score < 0.50:
        return "route_bound_mild"
    return "route_bound_likely"


def _wns_bucket(initial_wns: Optional[float]) -> str:
    if initial_wns is None or initial_wns > -5.0:
        return "wns_mild"
    return "wns_severe"


def feature_band(features: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    """Return the deterministic (lut, spread, route, wns) bucket tuple."""
    return (
        _lut_bucket(float(features.get("lut_count") or 0)),
        _spread_bucket(float(features.get("critical_path_spread") or 0)),
        _route_bucket(features.get("qor_route_bound_score")),
        _wns_bucket(features.get("initial_wns")),
    )


def _augmented_vector(
    lut: float, spread: float,
    score: Optional[float], has_signal: Optional[bool],
) -> Tuple[float, float, float, float]:
    """Augmented 4-tuple used by HFCv0's K-NN.

    Missing route_bound_score is mapped to 0 (treated as "no signal");
    missing has_signal is treated as False.  This keeps numeric
    distance well-defined while never *fabricating* signal.
    """
    return (
        float(lut),
        float(spread),
        float(score) if score is not None else 0.0,
        1.0 if has_signal else 0.0,
    )


def _z_normalize(values: Sequence[float]) -> Tuple[float, float]:
    """Return (mean, stdev) for ``values``; stdev is clamped >= 1.0
    to avoid division-by-zero on degenerate axes."""
    if not values:
        return (0.0, 1.0)
    m = statistics.mean(values)
    s = statistics.pstdev(values) or 1.0
    if s < 1.0:
        # If the axis is degenerate (constant), use 1.0 to keep the
        # distance scale comparable across axes.
        s = 1.0
    return (m, s)


def _topk_nearest(
    candidate: Tuple[float, float, float, float],
    history: Sequence[EpisodeSummary],
    k: int,
) -> List[Tuple[float, EpisodeSummary]]:
    """Return up to k nearest neighbours under z-normalised distance.

    Deterministic: ties are broken by ``episode_id`` lexicographic order.
    """
    if not history:
        return []
    pool_vecs = [_augmented_vector(h.lut_count, h.critical_path_spread,
                                    h.qor_route_bound_score,
                                    h.qor_has_congestion_signal)
                 for h in history]
    means_stds = [_z_normalize([v[i] for v in pool_vecs]) for i in range(4)]
    def _norm(v):
        return tuple((v[i] - means_stds[i][0]) / means_stds[i][1]
                     for i in range(4))
    cand_n = _norm(candidate)
    scored = []
    for h, v in zip(history, pool_vecs):
        vn = _norm(v)
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(cand_n, vn)))
        scored.append((d, h))
    scored.sort(key=lambda dh: (dh[0], dh[1].episode_id))
    return scored[: max(0, int(k))]


def _has_design_name_key(features: Mapping[str, Any]) -> bool:
    for k in features.keys():
        if k in _DESIGN_NAME_KEYS or k.lower() in _DESIGN_NAME_KEYS:
            return True
    return False


def _validate_advisory_family(family: Optional[str]) -> Optional[str]:
    """Return family if it's a valid label; None otherwise."""
    if family is None:
        return None
    if not isinstance(family, str):
        return None
    if family not in ADVISORY_FAMILIES:
        return None
    for tok in _FORBIDDEN_COMMAND_TOKENS:
        if tok in family:
            return None
    return family


def advise(
    candidate_features: Mapping[str, Any],
    history: Iterable[EpisodeSummary],
    negative_memory_hits: Optional[Sequence[str]] = None,
    k: int = 3,
    min_benefited_floor: int = 3,
) -> Advisory:
    """Compute the HFCv0 advisory for one candidate design.

    Returns an ``Advisory`` dataclass.  Pure function: same input,
    same output.  No side effects.

    ``min_benefited_floor`` (Session-19 direction-correction): the
    history pool must contain at least this many CARD_BENEFITED
    episodes before HFCv0 will surface any advisory.  At the current
    corpus N=1 BENEFITED, the default value (3) silences every
    advisory with reason ``insufficient_benefited_floor`` — preventing
    overfit to the single known BENEFITED lineage.  Pass a smaller
    value only for unit tests that intentionally exercise downstream
    rules with synthetic histories.
    """
    history = list(history)
    if negative_memory_hits is None:
        negative_memory_hits = []
    neg_hits = list(negative_memory_hits)

    if not isinstance(candidate_features, Mapping):
        return Advisory(silence=True, silence_reason="insufficient_data")

    if _has_design_name_key(candidate_features):
        return Advisory(silence=True, silence_reason="design_name_in_scope")

    lut = candidate_features.get("lut_count")
    spread = candidate_features.get("critical_path_spread")
    if not isinstance(lut, (int, float)) or not isinstance(spread, (int, float)):
        return Advisory(silence=True, silence_reason="insufficient_data")

    score = candidate_features.get("qor_route_bound_score")
    has_signal = candidate_features.get("qor_has_congestion_signal")
    if score is None and has_signal is None:
        return Advisory(silence=True, silence_reason="insufficient_data",
                        feature_band=feature_band(candidate_features))

    band = feature_band(candidate_features)

    # Session-19 floor: refuse to surface ANY advisory until the
    # history contains enough BENEFITED examples that the advisory
    # can be backed by cross-family evidence.  This prevents the
    # Session-18-LOO failure mode of "mirror the single BENEFITED
    # lineage" — at N=1 BENEFITED, every advice path silences.
    benefited_in_history = sum(1 for h in history
                                if h.card_verdict == CARD_BENEFITED)
    if benefited_in_history < int(min_benefited_floor):
        return Advisory(silence=True,
                        silence_reason="insufficient_benefited_floor",
                        feature_band=band,
                        negative_memory_hits=neg_hits)

    cand_vec = _augmented_vector(lut, spread, score, has_signal)
    neighbours = _topk_nearest(cand_vec, history, k)

    if not neighbours:
        return Advisory(silence=True, silence_reason="no_history",
                        feature_band=band, negative_memory_hits=neg_hits)

    top = [n for _, n in neighbours]
    benefited = [n for n in top if n.card_verdict == CARD_BENEFITED]
    neutral = [n for n in top if n.card_verdict == CARD_NEUTRAL]
    harmful = [n for n in top if n.card_verdict == CARD_HARMFUL]

    route_likely = band[2] == "route_bound_likely"
    wns_severe = band[3] == "wns_severe"
    route_low = band[2] == "route_bound_low"
    cand_lut_bucket = band[0]

    # Rule priority (top wins).  Order is chosen so the SAFEST output
    # is preferred when multiple rules could fire:
    #
    #   1. route_bound_likely + negative-memory → WARN ONLY (no family)
    #      — defensive; safe even if a harmful neighbour exists.
    #   2. CARD_HARMFUL in top-K → SILENCE
    #      — refuse to advise when similar designs got hurt.
    #   3. wns_bound_global (huge negative WNS + no route signal) →
    #      SILENCE with diagnostic — never recommend route-side action.
    #   4. BENEFITED neighbour in the SAME LUT bucket → ADVISORY.
    #      — same-bucket constraint avoids cross-class recipe leakage.
    #   5. neutral neighbours → SILENCE (saturated band).
    #   6. default → SILENCE.

    # Rule 1
    if route_likely and neg_hits:
        return Advisory(
            silence=False, confidence="medium",
            advisory_family=None,
            warning="route_bound_avoid_cell_surgery",
            negative_memory_hits=neg_hits,
            feature_band=band,
        )

    # Rule 2 — harmful neighbours only count when in the SAME LUT
    # bucket as the candidate (cross-bucket harmful is noise).
    same_band_harmful = [n for n in harmful
                         if _lut_bucket(n.lut_count) == cand_lut_bucket]
    if same_band_harmful:
        return Advisory(silence=True, silence_reason="harmful_neighbour_in_topk",
                        feature_band=band, negative_memory_hits=neg_hits)

    # Rule 3
    if route_low and wns_severe:
        return Advisory(silence=True,
                        silence_reason="wns_bound_global_no_route_advice",
                        diagnostic_label="wns_bound_global",
                        feature_band=band, negative_memory_hits=neg_hits)

    # Rule 4 — BENEFITED neighbour MUST be in the same LUT bucket as the
    # candidate.  This prevents a small_lut BENEFITED from advising a
    # mid_lut design (recipes don't generalise across LUT bands cleanly).
    same_band_benefited = []
    for n in benefited:
        n_band = (_lut_bucket(n.lut_count),)
        if n_band[0] == cand_lut_bucket:
            same_band_benefited.append(n)

    if same_band_benefited:
        chosen = None
        for n in same_band_benefited:
            if _validate_advisory_family(n.winning_action_family) is not None:
                chosen = n
                break
        if chosen is None:
            return Advisory(silence=True,
                            silence_reason="benefited_but_no_action_family",
                            feature_band=band,
                            negative_memory_hits=neg_hits,
                            evidence_episodes=[n.episode_id for n in same_band_benefited])
        conf = "high" if len(same_band_benefited) >= 2 else "medium"
        return Advisory(
            silence=False, confidence=conf,
            advisory_family=_validate_advisory_family(chosen.winning_action_family),
            warning=None,
            feature_band=band,
            negative_memory_hits=neg_hits,
            evidence_episodes=[n.episode_id for n in same_band_benefited],
        )

    # Rule 5
    if neutral:
        return Advisory(silence=True, silence_reason="saturated_neutral_band",
                        feature_band=band, negative_memory_hits=neg_hits)

    return Advisory(silence=True, silence_reason="no_match",
                    feature_band=band, negative_memory_hits=neg_hits)
