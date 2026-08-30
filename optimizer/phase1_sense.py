# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (C) 2026, Georgios Chatzitsompanis.
# Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

"""Phase-1 sensing mechanisms: measurement math, report parsing, wall cap.

Extracted verbatim from dcp_optimizer.py (no behavior change): the pure,
self-state-free helpers under Phase-1 measurement / feature extraction —

- fmax math (`calculate_fmax`): the organizers' reference formula
  fmax = 1000 / (period − WNS), for every sign of WNS;
- report parsing (`parse_high_fanout_nets`): the high-fanout-nets table
  → (net_name, fanout, path_count) rows;
- the cumulative Phase-1 wall cap (`PHASE1_WALL_FRAC_DEFAULT`,
  `resolve_phase1_wall_frac`): CLI-over-env resolution of the fraction
  of max_wall Phase 1's OPTIONAL analysis steps may burn before the
  iteration loop starts.

DCPOptimizerBase keeps thin delegating methods with their original names
and signatures (`calculate_fmax`, `parse_high_fanout_nets`), so call
sites, subclasses, and the test suite are unchanged.  The Phase-1
ORCHESTRATION (perform_initial_analysis, `_phase1_call`, the per-step
timeout/skip ledger, feature-cache readers like
`_phase1_wns_for_features`) stays in dcp_optimizer.py — it is
run-state-coupled; only the mechanisms live here.  Deeper static report
parsers already live in optimizer/static_parsers.py and
optimizer/qor_json_features.py.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def calculate_fmax(wns: Optional[float], clock_period: Optional[float]) -> Optional[float]:
    """Calculates achievable clock frequency in MHz from worst negative slack and
    clock period.

    The calculation is `1000 / (clock_period - WNS)` for either sign of WNS, so
    positive slack may produce a frequency above the target. Returns `None`
    when the value cannot be calculated, including when `WNS >= clock_period`
    and the implied period is nonpositive.
    """
    if clock_period is None or clock_period <= 0:
        return None
    if wns is None:
        return None

    achievable_period_ns = clock_period - wns
    if achievable_period_ns <= 0:
        return None

    return 1000.0 / achievable_period_ns


def parse_high_fanout_nets(report: str) -> list[tuple[str, int, int]]:
    """
    Parse high fanout nets report and return list of (net_name, fanout, path_count).
    """
    nets = []
    lines = report.split('\n')
    in_net_section = False

    for line in lines:
        if 'Paths' in line and 'Fanout' in line and 'Parent Net Name' in line:
            in_net_section = True
            continue

        if in_net_section:
            if line.startswith('---') or not line.strip():
                continue
            if line.startswith('==='):
                break

            parts = line.split()
            if len(parts) >= 3:
                try:
                    path_count = int(parts[0])
                    fanout = int(parts[1])
                    net_name = parts[2]

                    if (net_name and
                        '/' in net_name and
                        not net_name.startswith('get_') and
                        not net_name.startswith('ERROR') and
                        not net_name.startswith('WARNING')):
                        nets.append((net_name, fanout, path_count))
                except ValueError:
                    continue

    return nets


# Cap cumulative optional Phase-1 analysis time to prevent scaled per-step
# timeouts from consuming the wall budget before optimization begins.
# Skip optional steps after the cap and limit each timeout to the remaining
# allowance. Checkpoint opening and timing-summary reporting remain uncapped.
# The 15% default stays above expected sensing cost for very large designs.
PHASE1_WALL_FRAC_DEFAULT = 0.15


def resolve_phase1_wall_frac(cli_value, default: float = PHASE1_WALL_FRAC_DEFAULT) -> float:
    """Effective Phase-1 wall-cap fraction.

    CLI (--phase1-wall-frac) wins over env (FPL26_PHASE1_WALL_FRAC);
    unset/unparseable/out-of-range (must be in (0, 1]) keeps the default.
    """
    v = cli_value
    if v is None:
        env = os.environ.get("FPL26_PHASE1_WALL_FRAC", "").strip()
        if env:
            try:
                v = float(env)
            except ValueError:
                logger.warning(
                    f"FPL26_PHASE1_WALL_FRAC={env!r} not a float; keeping "
                    f"default {default}.")
                return default
    if v is None:
        return default
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    if not (0.0 < v <= 1.0):
        logger.warning(
            f"phase1_wall_frac={v} outside (0, 1]; keeping default {default}.")
        return default
    return v
