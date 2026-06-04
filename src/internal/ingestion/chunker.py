"""Context-Enriched Flat Chunking for FCA Handbook rules.

Every chunk is self-contained: context header + preamble + content.
No parent lookups needed at retrieval time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.config import SOURCEBOOK_NAMES
from src.internal.ingestion.parser import ParsedRule, _extract_defined_terms, _extract_cross_references

# Split at level-1 sub-paragraphs: (1), (2), (3), ...
SUB_PARA_L1 = re.compile(r"(?:^|\n)\s*\((\d+)\)\s", re.MULTILINE)

# Rough token estimate: 1 token ≈ 4 chars
MIN_CHUNK_TOKENS = 50
MIN_CHUNK_CHARS = MIN_CHUNK_TOKENS * 4
MAX_CHUNK_TOKENS = 1000
MAX_CHUNK_CHARS = MAX_CHUNK_TOKENS * 4


@dataclass
class Chunk:
    chunk_id: str
    text: str  # self-contained: header + preamble + content

    # Metadata for Weaviate filtering + citation
    sourcebook: str
    sourcebook_full: str
    chapter: str
    chapter_title: str
    section: str
    section_title: str
    rule_id: str
    rule_type: str
    sub_paragraph: str
    page: int
    is_annex: bool
    is_table: bool
    defined_terms: list[str] = field(default_factory=list)
    cross_references: list[str] = field(default_factory=list)


def _build_header(rule: ParsedRule) -> str:
    """Build context header: [SOURCEBOOK > Chapter X: title > Section X.X: title > RULE_ID]"""
    parts = [rule.sourcebook]
    if rule.chapter_id:
        parts.append(f"Chapter {rule.chapter_id}: {rule.chapter_title}")
    if rule.section_id:
        parts.append(f"Section {rule.section_id}: {rule.section_title}")
    parts.append(f"{rule.rule_id}{rule.rule_type}")
    return f"[{' > '.join(parts)}]"


def _split_sub_paragraphs(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split rule text into preamble + list of (sub_id, sub_text).

    Returns (preamble, [(\"(1)\", text), (\"(2)\", text), ...])
    """
    matches = list(SUB_PARA_L1.finditer(text))
    if not matches:
        return text, []

    preamble = text[: matches[0].start()].strip()
    subs = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sub_text = text[start:end].strip()
        sub_id = f"({m.group(1)})"
        subs.append((sub_id, sub_text))

    return preamble, subs


def _make_chunk_id(rule: ParsedRule, sub_para: str = "") -> str:
    display = f"{rule.rule_id}{rule.rule_type}".replace(" ", "_")
    if sub_para:
        display += f"_{sub_para}"
    return display


def _make_chunk(rule: ParsedRule, text: str, sub_para: str, header: str) -> Chunk:
    return Chunk(
        chunk_id=_make_chunk_id(rule, sub_para),
        text=text,
        sourcebook=rule.sourcebook,
        sourcebook_full=SOURCEBOOK_NAMES.get(rule.sourcebook, rule.sourcebook),
        chapter=rule.chapter_id,
        chapter_title=rule.chapter_title,
        section=rule.section_id,
        section_title=rule.section_title,
        rule_id=rule.rule_id,
        rule_type=rule.rule_type,
        sub_paragraph=sub_para,
        page=rule.page,
        is_annex=rule.is_annex,
        is_table=rule.is_table,
        defined_terms=_extract_defined_terms(text),
        cross_references=_extract_cross_references(text, rule.rule_id),
    )


def _chunk_rule(rule: ParsedRule) -> list[Chunk]:
    """Turn one ParsedRule into one or more self-contained Chunks."""
    if rule.is_deleted or not rule.text.strip():
        return []

    header = _build_header(rule)

    # Tables/annexes: don't split at sub-paragraphs (would break structure)
    if rule.is_table:
        text = f"{header}\n{rule.text}"
        if len(text) > MAX_CHUNK_CHARS:
            return _split_table(rule, text, header)
        return [_make_chunk(rule, text, "", header)]

    preamble, subs = _split_sub_paragraphs(rule.text)

    if not subs:
        # Standalone rule — one chunk
        text = f"{header}\n{rule.text}"
        chunk = _make_chunk(rule, text, "", header)

        # Fallback: recursive split if too long
        if len(text) > MAX_CHUNK_CHARS:
            return _recursive_split(rule, text, header)

        return [chunk]

    # Has sub-paragraphs — one chunk per (N), each with header + preamble
    preamble_line = f"{preamble}\n" if preamble else ""
    chunks = []

    for sub_id, sub_text in subs:
        chunk_text = f"{header}\n{preamble_line}{sub_text}"
        if len(chunk_text) > MAX_CHUNK_CHARS:
            chunks.extend(_recursive_split(rule, chunk_text, header, sub_id))
        else:
            chunks.append(_make_chunk(rule, chunk_text, sub_id, header))

    # Merge short children
    chunks = _merge_short(chunks)

    return chunks


def _merge_short(chunks: list[Chunk]) -> list[Chunk]:
    """Merge children under MIN_CHUNK_CHARS with the next sibling."""
    if len(chunks) <= 1:
        return chunks

    merged = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        # Keep merging with next if too short (but don't exceed max)
        while len(current.text) < MIN_CHUNK_CHARS and i + 1 < len(chunks):
            if len(current.text) + len(chunks[i + 1].text) > MAX_CHUNK_CHARS:
                break
            i += 1
            nxt = chunks[i]
            # Append next chunk's content (skip its header — already in current)
            # Find content after header line
            nxt_content = nxt.text.split("\n", 1)[1] if "\n" in nxt.text else nxt.text
            current.text += "\n" + nxt_content
            current.sub_paragraph += f"+{nxt.sub_paragraph}"
            current.chunk_id = _make_chunk_id_from_parts(
                current.chunk_id.rsplit("_", 1)[0] if "_(" in current.chunk_id else current.chunk_id,
                current.sub_paragraph,
            )
            current.defined_terms = _extract_defined_terms(current.text)
            current.cross_references = _extract_cross_references(current.text, current.rule_id)
        merged.append(current)
        i += 1

    return merged


def _make_chunk_id_from_parts(base: str, sub_para: str) -> str:
    return f"{base}_{sub_para}" if sub_para else base


def _split_table(rule: ParsedRule, text: str, header: str) -> list[Chunk]:
    """Split large tables at line boundaries to keep rows intact."""
    lines = text.split("\n")
    parts = []
    current = header
    header_len = len(header)

    for line in lines:
        if line.startswith("[") and line.endswith("]"):
            continue  # skip header line — we add it ourselves
        # If adding this line exceeds limit and we have content beyond header
        if len(current) + len(line) + 1 > MAX_CHUNK_CHARS and len(current) > header_len:
            parts.append(current.strip())
            current = header
        current += "\n" + line
    if current.strip() != header:
        parts.append(current.strip())

    # Fallback: if a single line exceeds the limit, split at sentence level
    final = []
    for i, part in enumerate(parts):
        if len(part) > MAX_CHUNK_CHARS:
            final.extend(_recursive_split(rule, part, header, f"part_{i + 1}"))
        else:
            sub = f"part_{i + 1}"
            final.append(_make_chunk(rule, part, sub, header))
    return final


def _recursive_split(rule: ParsedRule, text: str, header: str, base_sub: str = "") -> list[Chunk]:
    """Split very long text at sentence boundaries. Rare fallback."""
    # Strip header from text so we split only content
    content = text
    if text.startswith("[") and "]\n" in text:
        content = text.split("]\n", 1)[1]

    # Budget for content = max - header - newline
    max_content = MAX_CHUNK_CHARS - len(header) - 1

    sentences = re.split(r"(?<=[.;])\s+", content)
    parts = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) > max_content and current:
            parts.append(current.strip())
            current = ""
        current += sent + " "
    if current.strip():
        parts.append(current.strip())

    chunks = []
    for i, part in enumerate(parts):
        sub = f"{base_sub}_part_{i + 1}" if base_sub else f"part_{i + 1}"
        part = f"{header}\n{part}"
        chunks.append(_make_chunk(rule, part, sub, header))

    return chunks


def build_chunks(rules: dict[str, list[ParsedRule]]) -> list[Chunk]:
    """Build all chunks from parsed rules. Main entry point."""
    all_chunks = []

    for sb in sorted(rules.keys()):
        sb_chunks = []
        for rule in rules[sb]:
            sb_chunks.extend(_chunk_rule(rule))
        all_chunks.extend(sb_chunks)
        print(f"  {sb:8s}: {len(rules[sb]):5d} rules → {len(sb_chunks):5d} chunks")

    return all_chunks


# --- Runnable standalone ---

if __name__ == "__main__":
    import json
    import statistics
    from pathlib import Path

    from src.internal.ingestion.parser import load_parsed_rules

    print("Loading parsed rules...")
    rules = load_parsed_rules()
    total_rules = sum(len(r) for r in rules.values())
    print(f"Loaded {total_rules} rules\n")

    print("Building chunks...")
    chunks = build_chunks(rules)

    # --- Stats ---
    print(f"\n{'=' * 60}")
    print(f"CHUNK STATISTICS")
    print(f"{'=' * 60}")

    sizes = [len(c.text) // 4 for c in chunks]  # approx tokens
    sizes.sort()

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"\nSize (approx tokens):")
    print(f"  Min:    {min(sizes)}")
    print(f"  Max:    {max(sizes)}")
    print(f"  Mean:   {statistics.mean(sizes):.0f}")
    print(f"  Median: {statistics.median(sizes):.0f}")
    print(f"  P25:    {sizes[len(sizes) // 4]}")
    print(f"  P75:    {sizes[3 * len(sizes) // 4]}")
    print(f"  P95:    {sizes[int(len(sizes) * 0.95)]}")

    print(f"\nDistribution:")
    brackets = [(0, 50), (50, 150), (150, 300), (300, 500), (500, 1000), (1000, float("inf"))]
    for lo, hi in brackets:
        count = sum(1 for s in sizes if lo <= s < hi)
        label = f"{lo}-{hi}" if hi != float("inf") else f">{lo}"
        print(f"  {label:>10s}: {count:5d} ({100 * count / len(sizes):.0f}%)")

    # Type breakdown
    standalone = sum(1 for c in chunks if not c.sub_paragraph)
    with_sub = sum(1 for c in chunks if c.sub_paragraph and "+" not in c.sub_paragraph)
    merged = sum(1 for c in chunks if "+" in c.sub_paragraph)
    parts = sum(1 for c in chunks if "part_" in c.sub_paragraph)
    print(f"\nChunk types:")
    print(f"  Standalone (whole rule): {standalone}")
    print(f"  Sub-paragraph:          {with_sub}")
    print(f"  Merged short:           {merged}")
    print(f"  Recursive split:        {parts}")

    # Per sourcebook
    from collections import Counter
    sb_counts = Counter(c.sourcebook for c in chunks)
    print(f"\nPer sourcebook:")
    for sb, count in sorted(sb_counts.items()):
        print(f"  {sb:8s}: {count:5d} chunks")

    # Sample output
    print(f"\n{'=' * 60}")
    print("SAMPLE CHUNKS")
    print(f"{'=' * 60}")
    for c in chunks[:3]:
        print(f"\n--- {c.chunk_id} ({len(c.text) // 4} tokens) ---")
        print(c.text[:400])
        if len(c.text) > 400:
            print("...")
