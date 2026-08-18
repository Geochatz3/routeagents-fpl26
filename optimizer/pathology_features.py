"""Offline report-only pathology classifier prototype (no live wiring).

Consumes a feature dict — typically the flat dict produced by
``optimizer.episode_qor_join._flatten_qor_for_storage`` plus optionally a
small subset of existing ``start_features`` (e.g., ``initial_wns``,
``lut_count``).  Emits a small set of labels describing the most likely
*pathology* of the design.

This module is INTENTIONALLY:

- pure-Python, no Vivado, no DCP reads, no network,
- design-name-agnostic — refuses to look at any ``design_name`` key,
- offline / report-only — it is NOT wired into the optimizer's control flow,
- side-effect-free,
- deterministic — same input dict → same labels.

The labels are stable strings that callers may store in
``episode.action_trace_summary.pathology_prototype_labels``.  No live
recipe selection should branch on them yet.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

# Labels emitted (kept small and orthogonal where possible).
LABEL_INSUFFICIENT_DATA = "insufficient_data"
LABEL_ROUTE_BOUND_LIKELY = "route_bound_likely"
LABEL_ROUTE_BOUND_MILD = "route_bound_mild"
LABEL_CONGESTION_LOW = "congestion_low"
LABEL_GLOBAL_WNS_BOUND_LIKELY = "global_wns_bound_likely"
LABEL_NO_CONGESTION_SIGNAL = "no_congestion_signal"

# Thresholds — pinned here so they're inspectable and testable.
ROUTE_BOUND_STRONG = 0.50    # >= → route_bound_likely
ROUTE_BOUND_MILD = 0.20      # in [0.2, 0.5) → route_bound_mild
ROUTE_BOUND_LOW = 0.20       # < → congestion_low
WNS_LARGE_NEGATIVE = -5.0    # initial_wns < this (ns) → severe timing failure


def _is_design_name_key(k: str) -> bool:
    k_low = k.lower()
    return k_low in {"design_name", "design", "benchmark", "name"}


def classify_pathology(features: Mapping[str, Any]) -> List[str]:
    """Return a sorted, deterministic list of pathology labels.

    ``features`` is a flat dict.  Accepted keys (others are ignored):

    From QoR-flat (preferred):
        qor_route_bound_score      : float | None
        qor_has_congestion_signal  : bool

    Optionally supplemental (from existing start_features):
        initial_wns                 : float | None      (Vivado pre-opt WNS)
        lut_count                   : int | None
        critical_path_spread        : float | None      (RapidWright spread)

    Any key whose name suggests a design / benchmark identifier is
    refused — the function returns LABEL_INSUFFICIENT_DATA and ignores
    the rest (this guards against accidental design-name leakage into
    a future call site).
    """
    if not isinstance(features, Mapping):
        return [LABEL_INSUFFICIENT_DATA]

    for k in features.keys():
        if _is_design_name_key(k):
            # Refuse to classify when a design name is in scope at all.
            return [LABEL_INSUFFICIENT_DATA]

    labels: List[str] = []

    has_signal = features.get("qor_has_congestion_signal")
    score = features.get("qor_route_bound_score")
    initial_wns = features.get("initial_wns")

    if has_signal is None and score is None:
        labels.append(LABEL_INSUFFICIENT_DATA)
        # Still allow a WNS-bound hint when initial_wns is informative.
        if isinstance(initial_wns, (int, float)) and initial_wns <= WNS_LARGE_NEGATIVE:
            labels.append(LABEL_GLOBAL_WNS_BOUND_LIKELY)
        return sorted(set(labels))

    # No congestion signal at all (e.g., empty NESW vectors) — bridge to
    # WNS-bound hint when initial_wns supports it.
    if has_signal is False or score is None:
        labels.append(LABEL_NO_CONGESTION_SIGNAL)
        if isinstance(initial_wns, (int, float)) and initial_wns <= WNS_LARGE_NEGATIVE:
            labels.append(LABEL_GLOBAL_WNS_BOUND_LIKELY)
        else:
            # Treat zero/no congestion as "not route-bound by this signal".
            labels.append(LABEL_CONGESTION_LOW)
        return sorted(set(labels))

    if not isinstance(score, (int, float)):
        return [LABEL_INSUFFICIENT_DATA]

    # Real numeric route_bound_score in [0,1].
    if score >= ROUTE_BOUND_STRONG:
        labels.append(LABEL_ROUTE_BOUND_LIKELY)
    elif score >= ROUTE_BOUND_MILD:
        labels.append(LABEL_ROUTE_BOUND_MILD)
    else:
        labels.append(LABEL_CONGESTION_LOW)

    # WNS-bound axis is independent of route-bound axis.  Both can hold.
    if isinstance(initial_wns, (int, float)) and initial_wns <= WNS_LARGE_NEGATIVE:
        labels.append(LABEL_GLOBAL_WNS_BOUND_LIKELY)

    return sorted(set(labels))
