"""SUPPORT — parsing a QoR assessment payload

Static parsing of a QoR assessment payload into the fields the loop reads.

Named for the stage it implements: the slide, the video and the paper all
call it by this name, so a reader can move between them without a glossary.
"""
from __future__ import annotations

import re


def parse_qor_assessment_static(rqa_text: str) -> dict:
    """Extract compact feature signals from Vivado QoR assessment text.

    Returns a mapping containing an assessment score from 1 to 5, recommended
    flow guidance, a methodology-violation count, and whether an ML strategy is
    available. Missing or unparseable fields are returned as `None`.

    The parser accepts both tabular and free-text report variants. Callers must
    treat every feature as optional so routing can continue when the report is
    unavailable or malformed.
    """
    result = {
        "score": None,
        "flow_guidance": None,
        "methodology_violations": None,
        "ml_strategy_available": None,
    }
    if not isinstance(rqa_text, str) or not rqa_text:
        return result

    # 1) score
    # Tabular form: lines like `| RQA Score |  3  |` or `| Score | 3 |`
    # Free form:    `Assessment Score: 3` / `RQA Score: 3 - "..."`
    score_pat = re.compile(
        r"(?:RQA\s*Score|Assessment\s*Score|Overall\s*Score)\s*[:|]+\s*([1-5])\b",
        re.IGNORECASE,
    )
    m = score_pat.search(rqa_text)
    if m:
        try:
            result["score"] = int(m.group(1))
        except ValueError:
            result["score"] = None

    # Vivado reports ML availability through both real status text and a
    # canned disclaimer that appears regardless of availability. Ignore the
    # disclaimer and accept either the availability sentence or Yes/No table.
    rqa_lower = rqa_text.lower()
    if "ml strategies are available only when" in rqa_lower \
            or "ml strategy are available only when" in rqa_lower:
        # When the disclaimer is present, infer availability from the table.
        # At least one "OK" row means available; all "Not OK" rows mean
        # unavailable. Limit the match to the availability section.
        ml_section = re.search(
            r"ML\s+Strategy\s+Availability(.+?)"
            r"(?:Refer\s+to\s+UG906|\Z)",
            rqa_text, re.IGNORECASE | re.DOTALL,
        )
        if ml_section:
            section_text = ml_section.group(1)
            # Count status verdicts inside the section.
            ok_count = len(re.findall(r"\|\s*OK\s*\|", section_text))
            not_ok = len(re.findall(r"\|\s*Not\s*OK\s*\|", section_text,
                                       re.IGNORECASE))
            if ok_count > 0 and not_ok == 0:
                result["ml_strategy_available"] = True
            elif not_ok > 0 and ok_count == 0:
                result["ml_strategy_available"] = False
            # else: mixed / unparseable — leave None
    elif re.search(r"ML\s*Strateg(?:y|ies)\s+Available\s+(?:for|on)\b",
                    rqa_text, re.IGNORECASE):
        # Explicit "Available for this design" / "Available on ..." form.
        result["ml_strategy_available"] = True
    elif re.search(r"ML\s*Strateg(?:y|ies)\s+Not\s+Available",
                    rqa_text, re.IGNORECASE):
        result["ml_strategy_available"] = False

    # 3) methodology violations (count)
    m2 = re.search(
        r"(?:Methodology\s*Violations|Critical\s*Methodology(?:\s*Issues)?)\s*[:|]+\s*(\d+)",
        rqa_text, re.IGNORECASE,
    )
    if m2:
        try:
            result["methodology_violations"] = int(m2.group(1))
        except ValueError:
            pass

    # 4) flow guidance
    # Prefer an explicit "Flow Guidance" section if present.
    fg_match = re.search(
        r"Flow\s*Guidance\s*[:|]+\s*(.+?)(?:\r?\n)",
        rqa_text, re.IGNORECASE,
    )
    flow_from_vivado = None
    if fg_match:
        flow_from_vivado = fg_match.group(1).strip().strip("|").strip()

    # Suppress generic Flow Guidance that only directs users to a CSV file;
    # it provides no actionable prompt context. Its absence triggers the
    # score-based fallback.
    CANNED_FLOW_PATTERNS = (
        "to see critical timing paths examine the csv",
        "examine the csv file containing timing paths",
        # Multi-line table cells truncate Vivado's full line; the
        # captured first line is "To see critical timing paths
        # examine |", so this stem matches too.
        "to see critical timing paths examine",
        "examine the csv",
    )
    if flow_from_vivado:
        low = flow_from_vivado.lower()
        if any(pat in low for pat in CANNED_FLOW_PATTERNS):
            flow_from_vivado = None

    if flow_from_vivado:
        result["flow_guidance"] = flow_from_vivado
    else:
        # When the tool provides no actionable guidance, use the typical next
        # action from public knowledge-base article ka04U000001110fQAA.
        score_to_action = {
            1: "Redesign HLS modules / review target part",
            2: "Review constraints / review RTL",
            3: "Run report_qor_suggestions or ML Strategies",
            4: "Run report_qor_suggestions or ML Strategies",
            5: "Run implementation",
        }
        if result["score"] in score_to_action:
            result["flow_guidance"] = score_to_action[result["score"]]

    return result
