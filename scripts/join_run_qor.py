#!/usr/bin/env python3
"""Session-14 offline harness: join JSON QoR captures into episode_store.

Walks one or more run directories, finds ``*.qor.json`` files inside, and
calls ``optimizer.episode_qor_join.join_qor_features`` to attach the
parsed QoR features to matching episodes in
``policy_memory/episode_store.jsonl``.

This script is OFFLINE: it never opens a DCP, never invokes Vivado,
never touches the ``submission/`` tree, never imports the optimizer's
LLM machinery.

Match rules (inherited from ``episode_qor_join``):
- Confidence 3: ``output_dcp_path`` substring matches the JSON stem.
- Confidence 2: ``run_dir`` / ``run_id`` substring matches the JSON stem.
- Confidence 1: ``design_name`` matches the JSON stem
  (REPORT_ONLY by default; refuses to JOIN).

Usage::

    python3 scripts/join_run_qor.py --run-dir <dir> [<dir> ...] \
        [--episode-store policy_memory/episode_store.jsonl] \
        [--apply]

By default the script is a dry run.  Pass ``--apply`` to write the
updated episode_store atomically.  Submission writes are still refused.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from optimizer.episode_qor_join import (
    STATUS_AMBIGUOUS,
    STATUS_JOINED,
    STATUS_REJECTED,
    STATUS_REPORT_ONLY,
    STATUS_UNMATCHED,
    join_qor_features,
)


STATUS_ORDER = (
    STATUS_JOINED, STATUS_REPORT_ONLY, STATUS_AMBIGUOUS,
    STATUS_UNMATCHED, STATUS_REJECTED,
)


def find_qor_jsons(run_dirs):
    """Discover *.qor.json files inside the given run directories.

    Returns a sorted list of absolute Paths.  Recurses one level so the
    common ``<run_dir>/<design>.qor.json`` and
    ``<run_dir>/qor/<design>.qor.json`` layouts both work.
    """
    seen = set()
    out = []
    for rd in run_dirs:
        rd = Path(rd).resolve()
        if not rd.exists() or not rd.is_dir():
            continue
        for p in rd.rglob("*.qor.json"):
            try:
                rp = p.resolve()
            except OSError:
                continue
            if rp in seen:
                continue
            seen.add(rp)
            out.append(rp)
    return sorted(out)


def format_table(report) -> str:
    rows = ["status\tjson_path\tmatched_episode_ids\tmatch_method\troute_bound_score\tnotes"]
    by_status = {s: [] for s in STATUS_ORDER}
    for e in report.entries:
        by_status.setdefault(e.status, []).append(e)
    for s in STATUS_ORDER:
        for e in by_status.get(s, []):
            rows.append(
                "\t".join(str(x) for x in (
                    e.status,
                    e.json_path,
                    ",".join(e.matched_episode_ids) or "-",
                    e.match_method,
                    e.route_bound_score if e.route_bound_score is not None else "-",
                    e.notes,
                ))
            )
    return "\n".join(rows)


def summarise(report) -> dict:
    counts = {s: 0 for s in STATUS_ORDER}
    for e in report.entries:
        counts[e.status] = counts.get(e.status, 0) + 1
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline JSON QoR → episode_store join helper.",
    )
    parser.add_argument(
        "--run-dir", "-r", action="append", default=[], required=True,
        help="Run directory to search for *.qor.json files.  Repeatable.",
    )
    parser.add_argument(
        "--episode-store", "-e",
        default=str(ROOT / "policy_memory" / "episode_store.jsonl"),
        help="Path to episode_store.jsonl (default: policy_memory/episode_store.jsonl).",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Persist the join when matches are unambiguous.  Default: dry-run.",
    )
    parser.add_argument(
        "--allow-design-name-hint-join", action="store_true",
        help=("Promote design-name-only matches from REPORT_ONLY to JOINED.  "
              "OFF by default to avoid faking provenance.  Hidden-design safety: "
              "never enable for contest-day runs."),
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit the full JoinReport as JSON to stdout (in addition to the table).",
    )
    args = parser.parse_args(argv)

    store_path = Path(args.episode_store)
    if not store_path.exists():
        print(f"ERROR: episode_store not found: {store_path}", file=sys.stderr)
        return 2

    jsons = find_qor_jsons(args.run_dir)
    if not jsons:
        print(f"no *.qor.json found under: {args.run_dir}", file=sys.stderr)
        return 1

    print(f"found {len(jsons)} JSON QoR file(s):", file=sys.stderr)
    for j in jsons:
        print(f"  {j}", file=sys.stderr)
    print(f"episode_store: {store_path}", file=sys.stderr)
    print(f"apply       : {args.apply}", file=sys.stderr)

    try:
        report = join_qor_features(
            episode_store_path=store_path,
            json_paths=jsons,
            apply=args.apply,
            allow_design_name_hint_join=args.allow_design_name_hint_join,
        )
    except ValueError as exc:
        # Submission-path refusal or similar.  Loud, never silent.
        print(f"ERROR: join refused: {exc}", file=sys.stderr)
        return 5

    print(format_table(report))
    counts = summarise(report)
    print(
        "\nsummary: "
        + " ".join(f"{k}={v}" for k, v in counts.items())
        + f"  episode_store_changed={report.episode_store_changed}"
        + f"  episodes_modified={report.episodes_modified}",
        file=sys.stderr,
    )
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, default=str))

    # Exit code: 0 if all JOINED or any JOINED + nothing AMBIGUOUS;
    # 3 when ambiguous matches present (operator must intervene);
    # 4 when only unmatched / REPORT_ONLY (informational).
    if counts.get(STATUS_JOINED, 0) > 0 and counts.get(STATUS_AMBIGUOUS, 0) == 0:
        return 0
    if counts.get(STATUS_AMBIGUOUS, 0) > 0:
        return 3
    return 4


if __name__ == "__main__":
    sys.exit(main())
