"""Parse LlamaParse JSON output into structured ParsedRule objects."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.config import (
    RULE_ID_RE,
    RULE_TYPES,
    SOURCEBOOK_NAMES,
    SOURCEBOOK_PATTERN,
    VALID_SOURCEBOOKS,
    XREF_RE,
)

# --- Data structures ---


@dataclass
class ParsedRule:
    rule_id: str  # "CMCOB 2.1.1" (without type suffix)
    rule_type: str  # "R", "G", "E", "D", "EU", "UK"
    sourcebook: str
    text: str  # full markdown text (preserves *defined terms*)
    page: int
    section_id: str = ""
    section_title: str = ""
    chapter_id: str = ""
    chapter_title: str = ""
    is_annex: bool = False
    is_table: bool = False
    is_deleted: bool = False
    defined_terms: list[str] = field(default_factory=list)
    cross_references: list[str] = field(default_factory=list)


# --- Regex patterns ---

# Table separator line (to skip): | --- | --- | --- |
TABLE_SEP_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")

# Heading rule: ### CMCOB 2.1.13 R
HEADING_RULE_RE = re.compile(
    rf"^#+\s+({SOURCEBOOK_PATTERN})\s+([\d.]+[A-Z]*)\s+([A-Z]{{1,2}})\s*$"
)

# Inline text rule: **CMCOB 1.2.1** <u>R</u> text  or  **CMCOB 1.2.1** <mark>R</mark> text
INLINE_RULE_RE = re.compile(
    rf"\*\*({SOURCEBOOK_PATTERN})\s+([\d.]+[A-Z]*)\*\*\s*"
    r"(?:<u>|<mark>|<span[^>]*>)([A-Z]{1,2})(?:</u>|</mark>|</span>)\s*(.*)",
    re.DOTALL,
)

# Section header: Section : CMCOB 2.1 General principles
SECTION_RE = re.compile(
    rf"Section\s*:\s*({SOURCEBOOK_PATTERN})\s+([\d.]+[A-Z]*)\s+(.*)"
)

# Chapter heading: CMCOB 2 Conduct of business
CHAPTER_RE = re.compile(
    rf"^({SOURCEBOOK_PATTERN})\s+(\d+[A-Z]?)\s+(.+)$"
)

# Annex/Schedule/TP detection
ANNEX_RE = re.compile(
    rf"({SOURCEBOOK_PATTERN})\s+(Sch|TP|Annex)\s+(\d+)"
)

# Defined terms: *term* (but not ** or empty)
DEFINED_TERM_RE = re.compile(r"(?<!\*)\*([^*\n]{2,50})\*(?!\*)")

# Noise patterns to skip
NOISE_PATTERNS = [
    re.compile(r"^.{0,10}\s*logo\s*$", re.IGNORECASE),
    re.compile(r"^square\s+(icon|logo)\s*$", re.IGNORECASE),
    re.compile(r"^(CHAPTER|SOURCEBOOK)\s*$"),
    re.compile(r"^#+\s*(CHAPTER|SOURCEBOOK)\s*$"),
    re.compile(r"^#+\s*Table of Contents\s*$", re.IGNORECASE),
    re.compile(r"^FCA\s+logo\s*$", re.IGNORECASE),
]


def _is_noise(md: str) -> bool:
    md = md.strip()
    if not md or md == ".":
        return True
    # Bare sourcebook abbreviation
    if md.strip("# *") in VALID_SOURCEBOOKS:
        return True
    return any(p.match(md) for p in NOISE_PATTERNS)


def _extract_defined_terms(text: str) -> list[str]:
    return list(set(DEFINED_TERM_RE.findall(text)))


def _extract_cross_references(text: str, own_rule_id: str = "") -> list[str]:
    refs = []
    for m in XREF_RE.finditer(text):
        ref_id = f"{m.group(1)} {m.group(2)}{m.group(3)}"
        if ref_id != own_rule_id and ref_id not in refs:
            refs.append(ref_id)
    return refs


def _clean_rule_text(text: str) -> str:
    """Light cleanup of rule text while preserving *defined terms*."""
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"</?(?:u|mark|span|strong|em)(?:\s[^>]*)?>", "", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Remove pipe table formatting artifacts
    text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-|:\s]+$", "", text, flags=re.MULTILINE)
    # Collapse excessive whitespace but keep paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- Parsing logic ---


@dataclass
class _Context:
    """Mutable state while iterating through pages."""

    sourcebook: str
    chapter_id: str = ""
    chapter_title: str = ""
    section_id: str = ""
    section_title: str = ""
    is_annex: bool = False
    annex_prefix: str = ""
    last_rule: ParsedRule | None = None


def _parse_table_rules(item: dict, ctx: _Context, page_num: int) -> list[ParsedRule]:
    """Extract rules from a table item using its md field (preserves formatting)."""
    rules = []
    md = item.get("md", "")

    # Parse pipe-delimited rows line by line
    for line in md.split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        if TABLE_SEP_RE.match(line):
            continue

        # Split on | and take the columns
        cols = [c.strip() for c in line.split("|")]
        # First and last elements are empty (before first | and after last |)
        cols = [c for c in cols if c]
        if len(cols) < 3:
            continue

        raw_id, rule_type, text = cols[0], cols[1], cols[2]

        # Validate it looks like a rule ID
        if not RULE_ID_RE.match(raw_id):
            continue
        if rule_type not in RULE_TYPES:
            continue

        text = _clean_rule_text(text)
        full_id = raw_id.strip()

        rule = ParsedRule(
            rule_id=full_id,
            rule_type=rule_type,
            sourcebook=ctx.sourcebook,
            text=text,
            page=page_num,
            section_id=ctx.section_id,
            section_title=ctx.section_title,
            chapter_id=ctx.chapter_id,
            chapter_title=ctx.chapter_title,
            is_annex=ctx.is_annex,
            is_table=True,
            is_deleted="[deleted]" in text.lower(),
            defined_terms=_extract_defined_terms(text),
            cross_references=_extract_cross_references(text, full_id),
        )
        rules.append(rule)

    return rules


def _parse_heading_rule(md: str, ctx: _Context, page_num: int) -> ParsedRule | None:
    """Try to parse a heading as a rule ID. Returns rule with empty text (body added later)."""
    m = HEADING_RULE_RE.match(md.strip())
    if not m:
        return None

    sb, num, rtype = m.group(1), m.group(2), m.group(3)
    return ParsedRule(
        rule_id=f"{sb} {num}",
        rule_type=rtype,
        sourcebook=ctx.sourcebook,
        text="",
        page=page_num,
        section_id=ctx.section_id,
        section_title=ctx.section_title,
        chapter_id=ctx.chapter_id,
        chapter_title=ctx.chapter_title,
        is_annex=ctx.is_annex,
    )


def _parse_inline_rule(md: str, ctx: _Context, page_num: int) -> ParsedRule | None:
    """Try to parse an inline text rule: **CMCOB 1.2.1** <u>R</u> text."""
    m = INLINE_RULE_RE.match(md.strip())
    if not m:
        return None

    sb, num, rtype, text = m.group(1), m.group(2), m.group(3), m.group(4)
    text = _clean_rule_text(text)
    full_id = f"{sb} {num}"

    return ParsedRule(
        rule_id=full_id,
        rule_type=rtype,
        sourcebook=ctx.sourcebook,
        text=text,
        page=page_num,
        section_id=ctx.section_id,
        section_title=ctx.section_title,
        chapter_id=ctx.chapter_id,
        chapter_title=ctx.chapter_title,
        is_annex=ctx.is_annex,
        is_deleted="[deleted]" in text.lower(),
        defined_terms=_extract_defined_terms(text),
        cross_references=_extract_cross_references(text, full_id),
    )


def parse_sourcebook(json_path: str) -> list[ParsedRule]:
    """Parse a single LlamaParse JSON file into a list of ParsedRules."""
    with open(json_path) as f:
        data = json.load(f)

    sourcebook = Path(json_path).stem  # e.g., "CMCOB"
    ctx = _Context(sourcebook=sourcebook)
    rules: list[ParsedRule] = []

    for page in data["pages"]:
        page_num = page["page_number"]

        for item in page["items"]:
            itype = item["type"]
            md = item.get("md", "").strip()

            # Skip noise
            if itype in ("header", "footer"):
                continue
            if _is_noise(md):
                continue

            # --- Update context from headings ---
            if itype == "heading":
                # Section header?
                sec_m = SECTION_RE.search(md)
                if sec_m:
                    ctx.section_id = sec_m.group(2)
                    ctx.section_title = sec_m.group(3).strip()
                    # Detect annex/schedule
                    annex_m = ANNEX_RE.search(md)
                    if annex_m:
                        ctx.is_annex = True
                        ctx.annex_prefix = f"{annex_m.group(1)} {annex_m.group(2)} {annex_m.group(3)}"
                    continue

                # Chapter heading?
                ch_m = CHAPTER_RE.match(md.strip("# ").strip())
                if ch_m and ch_m.group(1) == sourcebook:
                    ctx.chapter_id = ch_m.group(2)
                    ctx.chapter_title = ch_m.group(3).strip()
                    ctx.is_annex = False
                    continue

                # Heading-format rule?
                rule = _parse_heading_rule(md, ctx, page_num)
                if rule:
                    if ctx.last_rule and not ctx.last_rule.is_deleted:
                        _finalize_rule(ctx.last_rule)
                    rules.append(rule)
                    ctx.last_rule = rule
                    continue

            # --- Table with rule rows ---
            if itype == "table" and "rows" in item:
                # Finalize any pending heading rule
                if ctx.last_rule and not ctx.last_rule.text and not ctx.last_rule.is_deleted:
                    _finalize_rule(ctx.last_rule)

                table_rules = _parse_table_rules(item, ctx, page_num)
                if table_rules:
                    rules.extend(table_rules)
                    ctx.last_rule = table_rules[-1]
                    continue

            # --- Inline text rule ---
            if itype == "text":
                rule = _parse_inline_rule(md, ctx, page_num)
                if rule:
                    if ctx.last_rule and not ctx.last_rule.is_deleted:
                        _finalize_rule(ctx.last_rule)
                    rules.append(rule)
                    ctx.last_rule = rule
                    continue

            # --- Orphaned content: append to last rule (cross-page merge) ---
            if ctx.last_rule and itype in ("text", "list"):
                cleaned = _clean_rule_text(md)
                if cleaned and not _is_noise(cleaned):
                    ctx.last_rule.text += "\n" + cleaned

    # Finalize last rule
    if ctx.last_rule:
        _finalize_rule(ctx.last_rule)

    return [r for r in rules if not r.is_deleted]


def _finalize_rule(rule: ParsedRule):
    """Fill in derived fields after all text has been collected."""
    rule.text = rule.text.strip()
    if not rule.defined_terms:
        rule.defined_terms = _extract_defined_terms(rule.text)
    if not rule.cross_references:
        rule.cross_references = _extract_cross_references(
            rule.text, f"{rule.rule_id}{rule.rule_type}"
        )
    rule.is_deleted = not rule.text or "[deleted]" in rule.text.lower()


def _detect_sourcebook(filename: str) -> str | None:
    """Map a filename like 'COBS.json', 'COBS_part1.json', 'COBS_1.json' to its sourcebook."""
    stem = Path(filename).stem
    # Exact match first
    if stem in SOURCEBOOK_NAMES:
        return stem
    # Check if stem starts with a known sourcebook + separator
    for sb in sorted(SOURCEBOOK_NAMES.keys(), key=len, reverse=True):
        if stem.startswith(sb) and (len(stem) == len(sb) or stem[len(sb)] in ("_", "-", ".")):
            return sb
    return None


def parse_all_sourcebooks(json_dir: str) -> dict[str, list[ParsedRule]]:
    """Parse all JSON files in the directory. Returns {sourcebook: [rules]}.

    Handles multi-part files (e.g., COBS_part1.json ... COBS_part9.json)
    by grouping them, parsing in sorted order, and merging results.
    """
    json_dir = Path(json_dir)

    # Group files by sourcebook
    sb_files: dict[str, list[Path]] = {}
    for json_path in sorted(json_dir.glob("*.json")):
        sb = _detect_sourcebook(json_path.name)
        if sb:
            sb_files.setdefault(sb, []).append(json_path)

    result = {}
    for sb, files in sorted(sb_files.items()):
        all_rules: list[ParsedRule] = []
        label = f"{sb} ({len(files)} parts)" if len(files) > 1 else sb
        print(f"Parsing {label}...")
        for f in sorted(files):
            rules = parse_sourcebook(str(f))
            all_rules.extend(rules)
        result[sb] = all_rules
        print(f"  → {len(all_rules)} rules extracted")

    return result


# --- Save / Load parsed rules (avoid re-parsing if later stages fail) ---

def save_parsed_rules(rules: dict[str, list[ParsedRule]], out_dir: str = "data/parsed"):
    """Serialize parsed rules to JSON for reuse."""
    from dataclasses import asdict

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for sb, sb_rules in rules.items():
        filepath = out_path / f"{sb}_rules.json"
        with open(filepath, "w") as f:
            json.dump([asdict(r) for r in sb_rules], f, indent=2)
        print(f"  Saved {len(sb_rules)} rules → {filepath}")


def load_parsed_rules(parsed_dir: str = "data/parsed") -> dict[str, list[ParsedRule]]:
    """Load previously saved parsed rules."""
    result = {}
    parsed_dir = Path(parsed_dir)
    for filepath in sorted(parsed_dir.glob("*_rules.json")):
        sb = filepath.stem.replace("_rules", "")
        with open(filepath) as f:
            raw = json.load(f)
        result[sb] = [ParsedRule(**r) for r in raw]
        print(f"  Loaded {len(result[sb])} rules ← {filepath}")
    return result


# --- Runnable standalone ---

if __name__ == "__main__":
    import sys

    json_dir = sys.argv[1] if len(sys.argv) > 1 else "llama_parse_output"
    all_rules = parse_all_sourcebooks(json_dir)
    save_parsed_rules(all_rules)

    total = sum(len(r) for r in all_rules.values())
    print(f"\nTotal: {total} rules across {len(all_rules)} sourcebooks")
