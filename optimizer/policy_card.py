"""Builds an advisory policy card from feature-similar prior episodes.

The result is an approximately 200–500-token text block describing relevant
actions and failures; it never selects commands, strategies, or tools.
Retrieval uses features only and must not match or branch on design names.
Positive patterns require `validation_status == "ok"` and a real
`validated_delta_fmax`; excluded episodes may still contribute eligible
negative-memory warnings. Contest-mode consumers use only contest-mode
episodes. One matching positive episode is low confidence; at least three
similar positive validated outcomes are medium confidence; at least three
validator-confirmed target-beating outcomes are high confidence. The policy
card is enabled only when `--policy-card` is passed.
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from optimizer.negative_memory import load_negative_memories
from optimizer.policy_memory import load_episodes

logger = logging.getLogger(__name__)


# Feature distance — Euclidean over normalised (lut_count, spread).
# Constants tuned to the v0 store: LUT range 1.9k–295k spans ≈ 150×;
# spread range 8–660 spans ≈ 80×.  Z-normalised crudely by dividing
# by typical scales so neither feature dominates.
_LUT_SCALE = 50_000.0
_SPREAD_SCALE = 100.0


@dataclass
class PolicyCardConfig:
    """Tunable knobs.  Defaults are conservative — increase k or relax
    `min_validated_delta_for_positive` only with evidence."""
    top_k: int = 5
    # Minimum validated_delta_fmax to count an episode as a positive
    # example.  +1.0 MHz noise band: anything below is "no clear gain".
    min_validated_delta_for_positive: float = 1.0
    # Min episodes to upgrade a confidence band.
    min_episodes_for_medium: int = 3
    min_episodes_for_high: int = 3
    # Token budget for the card body (rough — characters / 4 heuristic).
    max_chars: int = 1800


def _feature_dist(a_lut: Optional[int], a_spread: Optional[float],
                  b_lut: Optional[int], b_spread: Optional[float],
                  ) -> Optional[float]:
    if (a_lut is None or a_spread is None
            or b_lut is None or b_spread is None):
        return None
    dl = (a_lut - b_lut) / _LUT_SCALE
    ds = (a_spread - b_spread) / _SPREAD_SCALE
    return math.sqrt(dl * dl + ds * ds)


def _episode_features(ep: Dict[str, Any]) -> Tuple[Optional[int], Optional[float]]:
    sf = ep.get("start_features") or {}
    return sf.get("lut_count"), sf.get("critical_path_spread")


def _is_validated_positive(ep: Dict[str, Any], threshold: float) -> bool:
    o = ep.get("outcome") or {}
    if o.get("validation_status") != "ok":
        return False
    v = o.get("validated_delta_fmax")
    return v is not None and v >= threshold


def _tool_classes_from_episode(ep: Dict[str, Any]) -> List[str]:
    """Group tool calls by coarse class for the card.  We never
    suggest a specific directive — only the broad tool family."""
    by_name = (ep.get("action_trace_summary") or {}).get("tool_calls_by_name") or {}
    classes: List[str] = []
    for name, count in by_name.items():
        if not name:
            continue
        if "phys_opt" in name:
            classes.extend(["phys_opt_design"] * count)
        elif "route" in name:
            classes.extend(["route_design"] * count)
        elif "place" in name:
            classes.extend(["place_design"] * count)
        elif "pblock" in name:
            classes.extend(["pblock"] * count)
        elif "retim" in name.lower():
            classes.extend(["retiming"] * count)
        elif "open_checkpoint" in name or "read_checkpoint" in name:
            # Infrastructure — not a strategy.  Skip.
            continue
        elif "write_checkpoint" in name:
            continue
        else:
            classes.append(name)
    return classes


def select_similar_episodes(
    *,
    lut_count: Optional[int],
    critical_path_spread: Optional[float],
    contest_mode: bool,
    store_path: Optional[Any] = None,
    config: Optional[PolicyCardConfig] = None,
) -> List[Tuple[float, Dict[str, Any]]]:
    """Return up to top_k (distance, episode) pairs sorted by feature
    distance ascending.  Filters to validated-ok contest episodes
    (when contest_mode) and to episodes with non-null features."""
    cfg = config or PolicyCardConfig()
    eps = load_episodes(store_path) if store_path else load_episodes(
        # Lazy import to avoid circular reference at module level.
        __import__("optimizer.policy_memory",
                   fromlist=["default_store_path"]).default_store_path()
    )
    ranked: List[Tuple[float, Dict[str, Any]]] = []
    for ep in eps:
        # Mode filter
        if contest_mode and not ep.get("contest_mode"):
            continue
        # Feature filter — must have both features to be comparable.
        ep_lut, ep_spread = _episode_features(ep)
        if ep_lut is None or ep_spread is None:
            continue
        dist = _feature_dist(lut_count, critical_path_spread,
                              ep_lut, ep_spread)
        if dist is None:
            continue
        ranked.append((dist, ep))
    ranked.sort(key=lambda t: t[0])
    return ranked[: cfg.top_k]


def select_relevant_negative_memories(
    *,
    lut_count: Optional[int],
    critical_path_spread: Optional[float],
    contest_mode: bool,
    neg_store_path: Optional[Any] = None,
    config: Optional[PolicyCardConfig] = None,
) -> List[Tuple[float, Dict[str, Any]]]:
    """Selects relevant negative memories using stored profile features.

    Negative-memory records use `profile_features`, a snapshot of starting
    features, for distance calculations. Records without profile features
    remain eligible at the configured fallback distance.
    """
    cfg = config or PolicyCardConfig()
    mems = load_negative_memories(neg_store_path) if neg_store_path else load_negative_memories(
        __import__("optimizer.negative_memory",
                   fromlist=["default_store_path"]).default_store_path()
    )
    ranked: List[Tuple[float, Dict[str, Any]]] = []
    for m in mems:
        if contest_mode and not m.get("contest_mode"):
            continue
        pf = m.get("profile_features") or {}
        m_lut = pf.get("lut_count")
        m_spread = pf.get("critical_path_spread")
        if m_lut is None or m_spread is None:
            # Admit at maximum distance (still useful, just lower priority).
            dist = float("inf")
        else:
            d = _feature_dist(lut_count, critical_path_spread,
                              m_lut, m_spread)
            dist = d if d is not None else float("inf")
        ranked.append((dist, m))
    ranked.sort(key=lambda t: t[0])
    return ranked[: cfg.top_k * 2]


def _confidence_band(positive_count: int, beat_ship_count: int,
                     cfg: PolicyCardConfig) -> str:
    if beat_ship_count >= cfg.min_episodes_for_high:
        return "high"
    if positive_count >= cfg.min_episodes_for_medium:
        return "medium"
    return "low"


VALID_VARIANTS = ("default", "post_route_polish_v1", "route_bound_v1")


def _variant_advisory_text(
    variant: str,
    lut_count: Optional[int],
    critical_path_spread: Optional[float],
) -> Optional[str]:
    """Return cluster-derived variant advisory text, or None if no advisory
    applies.  Never references benchmark / design names; emits feature
    cluster reasoning only.  Returns a label-style advisory family from
    the same allow-list used by HFCv0 — never a Vivado / Tcl command.
    """
    if variant == "default" or not variant:
        return None
    lut = float(lut_count) if isinstance(lut_count, (int, float)) else None
    spread = float(critical_path_spread) if isinstance(critical_path_spread, (int, float)) else None

    if variant == "post_route_polish_v1":
        # Target: C2 cluster (mid_lut 10k..60k, mid_spread 50..200).
        if (lut is not None and 10_000 <= lut < 60_000
                and spread is not None and 50.0 <= spread < 200.0):
            return (
                "VARIANT_ADVISORY post_route_polish_v1 (advisory only):\n"
                "  cluster: mid_lut + mid_spread (no design-name lookup)\n"
                "  suggested action family: post_route_phys_opt\n"
                "  rationale: in this feature cluster, prior validated\n"
                "    lifts have come from post-route phys_opt polish\n"
                "    (e.g., -directive AggressiveFanoutOpt or\n"
                "    -critical_pin_opt) rather than pre-route surgery.\n"
                "    Consider running post-route phys_opt before\n"
                "    declaring done; not a command — feature-cluster\n"
                "    hint only."
            )
        return None

    if variant == "route_bound_v1":
        # Target: route_bound_likely candidates (high congestion signal).
        # Without QoR features in the card-rendering scope there is no way to gate
        # on route_bound_score directly; gate by small-lut + low_spread
        # proxy for now and emit a WARNING ONLY (no advisory_family).
        if (lut is not None and lut < 10_000
                and spread is not None and spread < 50.0):
            return (
                "VARIANT_ADVISORY route_bound_v1 (advisory only):\n"
                "  cluster: small_lut + low_spread (route-bound prone)\n"
                "  warning: avoid `phase2_multi_cell_path_surgery` and\n"
                "    other cell-surgery families on this fingerprint.\n"
                "  rationale: feature-similar past failures matched the\n"
                "    legacy-N5N7 mid-spread / low-LL anti-harm pattern.\n"
                "    Prefer routing-directive variation over\n"
                "    cell-side restructuring.\n"
                "  no advisory_family (warn only) — feature-cluster\n"
                "    hint, not a command."
            )
        return None

    return None


def render_policy_card(
    *,
    lut_count: Optional[int],
    critical_path_spread: Optional[float],
    contest_mode: bool,
    store_path: Optional[Any] = None,
    neg_store_path: Optional[Any] = None,
    config: Optional[PolicyCardConfig] = None,
    variant: str = "default",
) -> Dict[str, Any]:
    """Return a dict containing:
      - text:           the rendered prompt block (str, may be empty)
      - matched_episode_ids:     list[str]
      - negative_memory_ids:     list[str]
      - similarity_basis:        dict (features used + counts)
      - card_token_estimate:     int (chars/4 heuristic)
      - confidence:              "high" | "medium" | "low" | "none"

    The text block follows the POLICY_MEMORY_CARD format.
    When no similar validated episode exists, the
    card returns empty text but still records what was attempted.
    """
    cfg = config or PolicyCardConfig()
    similar = select_similar_episodes(
        lut_count=lut_count,
        critical_path_spread=critical_path_spread,
        contest_mode=contest_mode,
        store_path=store_path,
        config=cfg,
    )
    neg = select_relevant_negative_memories(
        lut_count=lut_count,
        critical_path_spread=critical_path_spread,
        contest_mode=contest_mode,
        neg_store_path=neg_store_path,
        config=cfg,
    )

    matched_ids = [ep.get("episode_id") for _, ep in similar]
    neg_ids = [m.get("memory_id") for _, m in neg]

    # Positive-pattern aggregation: only episodes with validated gain.
    positives = [(d, ep) for d, ep in similar
                 if _is_validated_positive(ep, cfg.min_validated_delta_for_positive)]
    beats = [(d, ep) for d, ep in positives
             if (ep.get("outcome") or {}).get("candidate_beats_ship")]
    confidence = _confidence_band(len(positives), len(beats), cfg)

    # Tool-class histogram across positives (broad family, not directive).
    tool_class_count: Counter = Counter()
    deltas: List[float] = []
    for _, ep in positives:
        for cls in _tool_classes_from_episode(ep):
            tool_class_count[cls] += 1
        v = (ep.get("outcome") or {}).get("validated_delta_fmax")
        if v is not None:
            deltas.append(float(v))

    deltas_sorted = sorted(deltas)
    median_delta = (deltas_sorted[len(deltas) // 2] if deltas
                    else None)
    best_delta = max(deltas) if deltas else None

    # Negative-memory histogram by code.
    neg_code_count: Counter = Counter()
    for _, m in neg:
        c = m.get("tool_error_code") or m.get("observed_effect") or "unknown"
        neg_code_count[c] += 1

    text_lines: List[str] = []
    if positives or neg:
        text_lines.append("POLICY_MEMORY_CARD:")
        text_lines.append("(advisory — consider, not mandatory; "
                          "feature-first retrieval, no design-name lookup)")
        text_lines.append("")
        text_lines.append("similarity basis:")
        text_lines.append(f"  features: lut_count, critical_path_spread")
        text_lines.append(f"  matched episodes (positive validated): {len(positives)}"
                          f"   total candidates considered: {len(similar)}")
        if similar:
            min_d = similar[0][0]
            max_d = similar[-1][0]
            text_lines.append(f"  feature distance range: {min_d:.3f} .. {max_d:.3f}")
        text_lines.append("")

    if positives:
        text_lines.append("positive historical patterns (validated):")
        for cls, count in tool_class_count.most_common(5):
            text_lines.append(f"  - {cls}: appeared in {count} action(s) "
                              f"across {len(positives)} similar episode(s)")
        if median_delta is not None:
            text_lines.append(f"  - median validated_delta_fmax: "
                              f"{median_delta:+.2f} MHz")
        if best_delta is not None:
            text_lines.append(f"  - best validated_delta_fmax: "
                              f"{best_delta:+.2f} MHz")
        if beats:
            text_lines.append(f"  - beat-ship validations: {len(beats)}")
        text_lines.append(f"  - confidence: {confidence}")
        text_lines.append("")

    if neg:
        text_lines.append("negative memory warnings (do not auto-ban; "
                          "feature-similar past failures):")
        for code, count in neg_code_count.most_common(5):
            text_lines.append(f"  - {code}: {count} record(s) — proceed with"
                              " caution if a similar action is on the table")
        text_lines.append("")

    if positives or neg:
        text_lines.append("evidence:")
        text_lines.append(f"  episode_ids: {matched_ids[:5]}")
        if neg_ids:
            text_lines.append(f"  negative_memory_ids: {neg_ids[:5]}")
        text_lines.append("guidance: \"consider\" / \"avoid\" only; "
                          "no mandatory command.  Design name is not "
                          "a key here.")

    # Variant advisory.  Default = unchanged behaviour.
    # When variant != "default", append a cluster-derived advisory
    # block at the END of the card so any historical card content
    # is preserved verbatim above it.  Never references design names.
    variant_text: Optional[str] = None
    if variant and variant != "default":
        if variant not in VALID_VARIANTS:
            logger.warning(
                f"policy_card: ignoring unknown variant {variant!r}; "
                f"valid values are {VALID_VARIANTS}"
            )
        else:
            variant_text = _variant_advisory_text(
                variant, lut_count, critical_path_spread,
            )
            if variant_text is not None:
                if text_lines:
                    text_lines.append("")
                text_lines.append(variant_text)

    text = "\n".join(text_lines)
    if len(text) > cfg.max_chars:
        text = text[: cfg.max_chars] + "\n... (card truncated)"

    return {
        "text": text,
        "matched_episode_ids": matched_ids,
        "negative_memory_ids": neg_ids,
        "similarity_basis": {
            "features_used": ["lut_count", "critical_path_spread"],
            "matched_count": len(similar),
            "positive_count": len(positives),
            "beat_ship_count": len(beats),
        },
        "card_token_estimate": (len(text) + 3) // 4,  # chars/4 heuristic
        "confidence": confidence if (positives or neg) else "none",
        "variant": variant if variant in VALID_VARIANTS else "default",
        "variant_advisory_emitted": variant_text is not None,
    }
