"""Analyze parsed rules for inconsistencies and quality issues.

Usage:
    python scripts/analyze_parsed.py                  # full report
    python scripts/analyze_parsed.py COBS             # single sourcebook
    python scripts/analyze_parsed.py --long 5000      # rules over N chars
    python scripts/analyze_parsed.py --empty-type     # rules with missing type
    python scripts/analyze_parsed.py --rule COBS 2.1.1  # inspect specific rule
"""

import json
import re
import sys
from pathlib import Path

DATA_DIR = Path("data/parsed")


def load_rules(sourcebook: str | None = None) -> list[dict]:
    rules = []
    for f in sorted(DATA_DIR.glob("*_rules.json")):
        sb = f.stem.replace("_rules", "")
        if sourcebook and sb != sourcebook:
            continue
        with open(f) as fh:
            rules.extend(json.load(fh))
    return rules


def full_report(rules: list[dict]):
    print(f"Total: {len(rules)} rules\n")

    # Per sourcebook
    from collections import Counter
    sb_counts = Counter(r["sourcebook"] for r in rules)
    for sb, count in sorted(sb_counts.items()):
        print(f"  {sb:8s}: {count:5d}")

    # Type distribution
    type_counts = Counter(r["rule_type"] for r in rules)
    print(f"\nRule types: {dict(type_counts)}")

    # Empty types
    empty_type = [r for r in rules if not r["rule_type"]]
    print(f"Empty rule_type: {len(empty_type)}")

    # Length stats
    lengths = [len(r["text"]) for r in rules]
    print(f"\nText length (chars): min={min(lengths)}, max={max(lengths)}, median={sorted(lengths)[len(lengths)//2]}")
    print(f"  <200:    {sum(1 for l in lengths if l < 200)}")
    print(f"  200-1K:  {sum(1 for l in lengths if 200 <= l < 1000)}")
    print(f"  1K-5K:   {sum(1 for l in lengths if 1000 <= l < 5000)}")
    print(f"  >5K:     {sum(1 for l in lengths if l >= 5000)}")

    # Xref stats
    total_xrefs = sum(len(r["cross_references"]) for r in rules)
    rule_ids = {r["rule_id"] for r in rules}
    all_xrefs = [x for r in rules for x in r["cross_references"]]
    matched = sum(1 for x in all_xrefs if x in rule_ids)
    print(f"\nCross-references: {total_xrefs} total, {matched} matched ({100*matched//max(total_xrefs,1)}%)")

    # Empty text
    empty = [r for r in rules if not r["text"].strip()]
    if empty:
        print(f"\nEmpty text: {len(empty)}")
        for r in empty:
            print(f"  {r['rule_id']} ({r['sourcebook']})")

    # Missing sections
    no_section = [r for r in rules if not r["section_id"]]
    if no_section:
        print(f"\nMissing section_id: {len(no_section)}")
        for r in no_section[:5]:
            print(f"  {r['rule_id']} page={r['page']}")

    # Missing chapters
    no_chapter = [r for r in rules if not r["chapter_id"]]
    if no_chapter:
        print(f"\nMissing chapter_id: {len(no_chapter)}")
        for r in no_chapter[:5]:
            print(f"  {r['rule_id']} page={r['page']}")


def show_long(rules: list[dict], threshold: int):
    long = sorted([r for r in rules if len(r["text"]) > threshold], key=lambda r: len(r["text"]), reverse=True)
    print(f"Rules > {threshold} chars: {len(long)}\n")
    for r in long:
        print(f"  {r['sourcebook']:8s} {r['rule_id']}{r['rule_type']:20s} {len(r['text']):6d} chars  section={r['section_id']}")


def show_empty_type(rules: list[dict]):
    empty = [r for r in rules if not r["rule_type"]]
    print(f"Rules with empty rule_type: {len(empty)}\n")
    for r in empty:
        print(f"  {r['sourcebook']:8s} {r['rule_id']:25s} page={r['page']:4d}  text={r['text'][:80]}...")


def show_rule(rules: list[dict], rule_id: str):
    matches = [r for r in rules if rule_id in r["rule_id"]]
    if not matches:
        print(f"No rules matching '{rule_id}'")
        return
    for r in matches:
        print(f"Rule: {r['rule_id']}{r['rule_type']}")
        print(f"  sourcebook: {r['sourcebook']}")
        print(f"  chapter: {r['chapter_id']} — {r['chapter_title']}")
        print(f"  section: {r['section_id']} — {r['section_title']}")
        print(f"  page: {r['page']}")
        print(f"  is_table: {r['is_table']}, is_annex: {r['is_annex']}")
        print(f"  defined_terms: {r['defined_terms'][:10]}")
        print(f"  cross_references: {r['cross_references']}")
        print(f"  text ({len(r['text'])} chars):")
        print(f"    {r['text'][:500]}")
        if len(r["text"]) > 500:
            print(f"    ... ({len(r['text'])} chars total)")
        print()


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--rule" in args:
        idx = args.index("--rule")
        rule_id = " ".join(args[idx + 1:])
        show_rule(load_rules(), rule_id)
    elif "--long" in args:
        idx = args.index("--long")
        threshold = int(args[idx + 1]) if idx + 1 < len(args) else 5000
        show_long(load_rules(), threshold)
    elif "--empty-type" in args:
        show_empty_type(load_rules())
    elif args and args[0].isupper():
        rules = load_rules(args[0])
        if not rules:
            print(f"No rules for {args[0]}")
        else:
            full_report(rules)
    else:
        full_report(load_rules())
