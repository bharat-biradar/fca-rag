"""Split merged rules that accidentally swallowed subsequent rules during parsing.

Detection: a rule's text contains **bold rule IDs** (e.g., **COBS 4.12A.9B**)
which are actually separate rules, not cross-references.

This runs between parser and chunker as a post-processing safety net.
"""

from __future__ import annotations

import re
from copy import deepcopy

from src.config import SOURCEBOOK_PATTERN, RULE_TYPES
from src.internal.ingestion.parser import ParsedRule, _extract_defined_terms, _extract_cross_references

# 3-segment rule number pattern: chapter.section.rule (e.g., 4.12A.9B, 2.1.1, 1A.1.1)
RULE_NUM = r"\d+[A-Z]?\.\d+[A-Z]?\.\d+[A-Z]*"

# Matches bold rule IDs embedded in text in various formats:
#   **COBS 4.12A.9B**
#   **COBS 4.12A.9B** **R**
#   **COBS 4.12A.21 R**  (space + type inside bold)
#   **COBS 4.12A.9B** <u>R</u>
BOLD_RULE_RE = re.compile(
    rf"\*\*({SOURCEBOOK_PATTERN})\s+({RULE_NUM})"
    r"(?:\s+([A-Z]{1,2}))?"  # optional type INSIDE bold: **COBS 4.12A.21 R**
    r"\*\*"
    r"(?:\s*\*\*([A-Z]{1,2})\*\*)?"  # optional bold type OUTSIDE: **R**
    r"(?:\s*(?:<u>|<mark>|<span[^>]*>)?([A-Z]{1,2})(?:</u>|</mark>|</span>))?"  # or tagged type
)

# Matches non-bold rule IDs at line starts (fallback for rules that lost their bold formatting)
LINESTART_RULE_RE = re.compile(
    rf"(?:^|\n)\s*({SOURCEBOOK_PATTERN})\s+({RULE_NUM})\s",
)


def split_merged_rules(rules: list[ParsedRule]) -> list[ParsedRule]:
    """Scan all rules for embedded bold rule IDs and split them out."""
    result = []
    split_count = 0

    for rule in rules:
        splits = _try_split(rule)
        if len(splits) > 1:
            split_count += 1
        result.extend(splits)

    if split_count > 0:
        print(f"    Split {split_count} merged rules → {len(result) - len(rules) + split_count} new rules freed")

    return result


def _try_split(rule: ParsedRule) -> list[ParsedRule]:
    """If rule text contains embedded rule IDs, split at those boundaries."""
    own_id = rule.rule_id
    split_points = []

    # Pass 1: Bold rule IDs
    for m in BOLD_RULE_RE.finditer(rule.text):
        found_id = f"{m.group(1)} {m.group(2)}"
        if found_id != own_id:
            # Type can be in group 3 (inside bold), 4 (outside bold **R**), or 5 (tagged)
            rule_type = m.group(3) or m.group(4) or m.group(5) or ""
            split_points.append((m.start(), found_id, rule_type))

    # Pass 2: Non-bold rule IDs at line starts
    # Run if: we already found bold IDs (known merge bug), OR rule is suspiciously long (>3000 chars)
    if split_points or len(rule.text) > 3000:
        seen_ids = {sp[1] for sp in split_points}
        seen_ids.add(own_id)
        for m in LINESTART_RULE_RE.finditer(rule.text):
            raw_num = m.group(2)
            # Strip trailing type chars from the number if present
            clean_num = re.sub(r'[RGDEUK]+$', '', raw_num)
            type_suffix = raw_num[len(clean_num):]
            found_id = f"{m.group(1)} {clean_num}"
            if found_id not in seen_ids:
                split_points.append((m.start(), found_id, type_suffix))
                seen_ids.add(found_id)

    if not split_points:
        return [rule]

    # Sort by position
    split_points.sort(key=lambda x: x[0])

    if not split_points:
        return [rule]

    # Build split rules
    result = []
    text = rule.text

    # First segment: the original rule's text up to the first split point
    first_text = text[:split_points[0][0]].strip()
    if first_text:
        original = deepcopy(rule)
        original.text = first_text
        original.is_deleted = not first_text or "[deleted]" in first_text.lower()
        original.defined_terms = _extract_defined_terms(first_text)
        original.cross_references = _extract_cross_references(first_text, f"{rule.rule_id}{rule.rule_type}")
        result.append(original)

    # Subsequent segments: each split-out rule
    for i, (start, found_id, rtype) in enumerate(split_points):
        end = split_points[i + 1][0] if i + 1 < len(split_points) else len(text)
        segment_text = text[start:end].strip()

        # Remove the bold ID prefix from the text
        segment_text = BOLD_RULE_RE.sub("", segment_text, count=1).strip()
        # Also strip any leading type tag that was part of the bold pattern
        segment_text = re.sub(r"^(?:<u>|<mark>|<span[^>]*>)?[A-Z]{1,2}(?:</u>|</mark>|</span>)?\s*", "", segment_text, count=1).strip()

        if not segment_text or "[deleted]" in segment_text.lower():
            # Still create the rule so it's tracked, just mark deleted
            pass

        new_rule = ParsedRule(
            rule_id=found_id,
            rule_type=rtype if rtype in RULE_TYPES else "",
            sourcebook=rule.sourcebook,
            text=segment_text,
            page=rule.page,
            section_id=rule.section_id,
            section_title=rule.section_title,
            chapter_id=rule.chapter_id,
            chapter_title=rule.chapter_title,
            is_annex=rule.is_annex,
            is_table=False,
            is_deleted=not segment_text or "[deleted]" in segment_text.lower(),
            defined_terms=_extract_defined_terms(segment_text),
            cross_references=_extract_cross_references(segment_text, found_id),
        )
        result.append(new_rule)

    return result


def split_all_sourcebooks(rules: dict[str, list[ParsedRule]]) -> dict[str, list[ParsedRule]]:
    """Apply splitting across all sourcebooks."""
    result = {}
    for sb, sb_rules in rules.items():
        result[sb] = split_merged_rules(sb_rules)
    return result


# --- Runnable standalone ---

if __name__ == "__main__":
    import json
    from pathlib import Path
    from src.internal.ingestion.parser import load_parsed_rules, save_parsed_rules

    print("Loading parsed rules...")
    rules = load_parsed_rules()

    total_before = sum(len(r) for r in rules.values())
    print(f"\nBefore splitting: {total_before} rules")

    rules = split_all_sourcebooks(rules)

    total_after = sum(len(r) for r in rules.values())
    print(f"After splitting: {total_after} rules (+{total_after - total_before})")

    # Save the split results
    print("\nSaving split rules...")
    save_parsed_rules(rules)

    # Check the worst offenders are fixed
    for sb, sb_rules in rules.items():
        long_bugs = [r for r in sb_rules if len(r["text"] if isinstance(r, dict) else r.text) > 8000
                     and BOLD_RULE_RE.search(r["text"] if isinstance(r, dict) else r.text)]
        if long_bugs:
            for r in long_bugs:
                text = r["text"] if isinstance(r, dict) else r.text
                rid = r["rule_id"] if isinstance(r, dict) else r.rule_id
                print(f"  ⚠ Still merged: {rid} ({len(text)} chars)")
