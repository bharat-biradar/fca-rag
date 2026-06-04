"""Context-Enriched Grouped Chunking for FCA Handbook rules (v2).

Same self-contained principle as v1, but groups consecutive sub-paragraphs
to reduce chunk count and increase average chunk size.
"""

from __future__ import annotations

import re

from src.internal.ingestion.chunker import (
    Chunk,
    _build_header,
    _make_chunk,
    _make_chunk_id,
    _split_sub_paragraphs,
)
from src.internal.ingestion.parser import ParsedRule, load_parsed_rules

# --- v2 constants (higher minimum for fewer, meatier chunks) ---
MIN_CHUNK_TOKENS = 125
MIN_CHUNK_CHARS = MIN_CHUNK_TOKENS * 4   # 500
MAX_CHUNK_TOKENS = 1000
MAX_CHUNK_CHARS = MAX_CHUNK_TOKENS * 4   # 4000

# Minimum meaningful content after stripping markdown/whitespace
MIN_CONTENT_CHARS = 30

# Strip markdown formatting to get raw text length
_STRIP_MD_RE = re.compile(r"\*{1,2}|<[^>]+>|\\[\[\]]|#{1,6}\s*")


def _is_junk_rule(rule: ParsedRule) -> bool:
    """Detect rules that should be skipped — deleted, parser artifacts, table headers."""
    text = rule.text.strip()
    if not text:
        return True
    # Deleted rules: handle escaped brackets \[deleted.] and variants
    if re.search(r"\\?\[deleted", text, re.IGNORECASE):
        return True
    # Strip markdown to measure real content length
    clean = _STRIP_MD_RE.sub("", text).strip()
    if len(clean) < MIN_CONTENT_CHARS:
        return True
    return False


def _is_preamble_stub(rule: ParsedRule) -> bool:
    """Detect orphaned preamble rules — text ends with ':' and is very short.

    These are preambles whose sub-paragraphs were split into separate rules
    by the rule_splitter. The preamble content is already duplicated inside
    each child chunk, so this standalone chunk is redundant noise.
    """
    text = rule.text.strip()
    return text.endswith(":") and len(text) < 150


def _group_sub_paragraphs(
    subs: list[tuple[str, str]],
    header: str,
    preamble: str,
) -> list[tuple[str, str]]:
    """Group consecutive sub-paragraphs into chunks respecting size bounds.

    Returns list of (sub_para_label, combined_body_text) where body_text
    is the raw sub-paragraph text (caller prepends header + preamble).
    """
    if not subs:
        return []

    preamble_line = f"{preamble}\n" if preamble else ""
    overhead = len(header) + 1 + len(preamble_line)  # header + \n + preamble

    groups: list[tuple[str, str]] = []
    group_start_id = subs[0][0]
    group_end_id = subs[0][0]
    group_body = subs[0][1]

    for sub_id, sub_text in subs[1:]:
        candidate_len = overhead + len(group_body) + 1 + len(sub_text)

        if candidate_len <= MAX_CHUNK_CHARS:
            group_body += "\n" + sub_text
            group_end_id = sub_id
        else:
            label = group_start_id if group_start_id == group_end_id else f"{group_start_id}-{group_end_id}"
            groups.append((label, group_body))
            group_start_id = sub_id
            group_end_id = sub_id
            group_body = sub_text

    # Flush final group
    label = group_start_id if group_start_id == group_end_id else f"{group_start_id}-{group_end_id}"
    groups.append((label, group_body))

    # Post-pass: merge trailing undersized group into predecessor
    if len(groups) > 1:
        last_label, last_body = groups[-1]
        last_full_len = overhead + len(last_body)
        if last_full_len < MIN_CHUNK_CHARS:
            prev_label, prev_body = groups[-2]
            merged_len = overhead + len(prev_body) + 1 + len(last_body)
            if merged_len <= MAX_CHUNK_CHARS:
                new_start = prev_label.split("-")[0]
                new_end = last_label.split("-")[-1]
                new_label = new_start if new_start == new_end else f"{new_start}-{new_end}"
                groups[-2] = (new_label, prev_body + "\n" + last_body)
                groups.pop()

    return groups


def _split_table_v2(rule: ParsedRule, text: str, header: str) -> list[Chunk]:
    """Split large tables at line boundaries to keep rows intact."""
    lines = text.split("\n")
    parts: list[str] = []
    current = header
    header_len = len(header)

    for line in lines:
        if line.startswith("[") and line.endswith("]"):
            continue  # skip header line — we add it ourselves
        if len(current) + len(line) + 1 > MAX_CHUNK_CHARS and len(current) > header_len:
            parts.append(current.strip())
            current = header
        current += "\n" + line
    if current.strip() != header:
        parts.append(current.strip())

    final: list[Chunk] = []
    for i, part in enumerate(parts):
        if len(part) > MAX_CHUNK_CHARS:
            final.extend(_recursive_split_v2(rule, part, header, f"part_{i + 1}"))
        else:
            final.append(_make_chunk(rule, part, f"part_{i + 1}", header))
    return final


def _recursive_split_v2(
    rule: ParsedRule, text: str, header: str, base_sub: str = "",
) -> list[Chunk]:
    """Split very long text at sentence boundaries. Rare fallback."""
    content = text
    if text.startswith("[") and "]\n" in text:
        content = text.split("]\n", 1)[1]

    max_content = MAX_CHUNK_CHARS - len(header) - 1

    sentences = re.split(r"(?<=[.;])\s+", content)
    parts: list[str] = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) > max_content and current:
            parts.append(current.strip())
            current = ""
        current += sent + " "
    if current.strip():
        parts.append(current.strip())

    chunks: list[Chunk] = []
    for i, part in enumerate(parts):
        sub = f"{base_sub}_part_{i + 1}" if base_sub else f"part_{i + 1}"
        part = f"{header}\n{part}"
        chunks.append(_make_chunk(rule, part, sub, header))
    return chunks


def _chunk_rule_v2(rule: ParsedRule) -> list[Chunk]:
    """Turn one ParsedRule into one or more self-contained Chunks.

    v2: groups consecutive sub-paragraphs to reduce chunk count.
    """
    if rule.is_deleted or not rule.text.strip():
        return []
    if _is_junk_rule(rule) or _is_preamble_stub(rule):
        return []

    header = _build_header(rule)

    # Tables/annexes: don't split at sub-paragraphs (would break structure)
    if rule.is_table:
        text = f"{header}\n{rule.text}"
        if len(text) > MAX_CHUNK_CHARS:
            return _split_table_v2(rule, text, header)
        return [_make_chunk(rule, text, "", header)]

    preamble, subs = _split_sub_paragraphs(rule.text)

    if not subs:
        # Standalone rule — one chunk
        text = f"{header}\n{rule.text}"
        if len(text) > MAX_CHUNK_CHARS:
            return _recursive_split_v2(rule, text, header)
        return [_make_chunk(rule, text, "", header)]

    # Has sub-paragraphs — group consecutive ones
    groups = _group_sub_paragraphs(subs, header, preamble)
    preamble_line = f"{preamble}\n" if preamble else ""
    chunks: list[Chunk] = []

    for label, body in groups:
        chunk_text = f"{header}\n{preamble_line}{body}"
        if len(chunk_text) > MAX_CHUNK_CHARS:
            chunks.extend(_recursive_split_v2(rule, chunk_text, header, label))
        else:
            chunks.append(_make_chunk(rule, chunk_text, label, header))

    return chunks


def build_chunks(rules: dict[str, list[ParsedRule]]) -> list[Chunk]:
    """Build all chunks from parsed rules. Main entry point."""
    all_chunks: list[Chunk] = []

    for sb in sorted(rules.keys()):
        sb_chunks: list[Chunk] = []
        for rule in rules[sb]:
            sb_chunks.extend(_chunk_rule_v2(rule))
        all_chunks.extend(sb_chunks)
        print(f"  {sb:8s}: {len(rules[sb]):5d} rules → {len(sb_chunks):5d} chunks")

    return all_chunks


# --- Runnable standalone ---

if __name__ == "__main__":
    import statistics
    from collections import Counter

    print("Loading parsed rules...")
    rules = load_parsed_rules()
    total_rules = sum(len(r) for r in rules.values())
    print(f"Loaded {total_rules} rules\n")

    print("Building v2 chunks...")
    chunks = build_chunks(rules)

    # --- Stats ---
    print(f"\n{'=' * 60}")
    print(f"CHUNK STATISTICS (v2: MIN={MIN_CHUNK_CHARS}, MAX={MAX_CHUNK_CHARS})")
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
    with_sub = sum(1 for c in chunks if c.sub_paragraph and "-" not in c.sub_paragraph and "part_" not in c.sub_paragraph)
    grouped = sum(1 for c in chunks if "-" in c.sub_paragraph and "part_" not in c.sub_paragraph)
    parts = sum(1 for c in chunks if "part_" in c.sub_paragraph)
    print(f"\nChunk types:")
    print(f"  Standalone (whole rule): {standalone}")
    print(f"  Single sub-paragraph:   {with_sub}")
    print(f"  Grouped sub-paragraphs: {grouped}")
    print(f"  Recursive split:        {parts}")

    # Per sourcebook
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
