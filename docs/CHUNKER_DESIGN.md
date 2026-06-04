# Chunker Design: Implementation Details

## Input

`dict[str, list[ParsedRule]]` — loaded from `data/parsed/*_rules.json` or directly from parser.

Parser gave us **2,867 rules** across 10 sourcebooks. Each rule has: `rule_id`, `rule_type`, `text`, `page`, `section_id`, `section_title`, `chapter_id`, `chapter_title`, `sourcebook`, `cross_references`, `defined_terms`, `is_table`, `is_annex`.

## Output

`list[Chunk]` — flat list of chunks ready for embedding. Each chunk has text + all metadata needed for Weaviate storage and citation.

---

## Data Analysis (from 1,492 rules in CMCOB+COBS+CASS)

| Metric | Value |
|--------|-------|
| Rules with sub-paragraphs (1),(2)... | 603 (40%) |
| Rules without sub-paragraphs (standalone) | 889 (60%) |
| Rules with preamble before (1) | 329 of 603 (55%) |
| Text length — median | ~89 tokens |
| Text length — mean | ~282 tokens |
| Text length — <50 tokens | 329 rules (22%) |
| Text length — 50-200 tokens | 853 rules (57%) |
| Text length — 200-500 tokens | 150 rules (10%) |
| Text length — 500-1000 tokens | 67 rules (4.5%) |
| Text length — >1000 tokens | 93 rules (6%) |
| Longest rule | COBS 4.12A.43G — 42,440 chars (~10,610 tokens) |
| Max sub-paragraphs in one rule | 27 (CMCOB 2.1.21R) |
| Empty text rules | 0 |
| Very short rules (<50 chars) | 12 |

---

## The Decision: Do We Split Into Parent/Child?

### Option A: Parent-child splitting (split rules at sub-paragraph level)

- Rules with `(1)`, `(2)`, ... get split: each sub-paragraph is a child chunk, full rule is parent
- Embed children (small, precise) → retrieve parent (full context for LLM)

**Pros:**
- Precise retrieval — small chunks match specific queries better
- Sub-paragraph `(2)(a)` about "screen sharing risks" won't be diluted by sub-paragraphs `(1)`, `(3)`, `(4)` about unrelated topics

**Cons:**
- Complexity: need parent_text field, parent-child ID linking, merge/split logic
- 40% of rules have sub-paragraphs → creates ~1,500+ additional chunks
- Some sub-paragraphs are too short alone (e.g., "(1) [deleted]" or "(a) integrity;")
- Preamble handling: 55% of sub-paragraph rules have text before `(1)` — needs special treatment

### Option B: Whole-rule chunks (each rule = one chunk)

- Every rule is one chunk, no splitting
- Embed the full rule text

**Pros:**
- Simple — no parent/child complexity
- Each chunk is self-contained with its full rule context
- 57% of rules are already 50-200 tokens (good chunk size)
- Metadata already provides section/chapter context

**Cons:**
- Large rules (93 rules >1000 tokens, max 10K tokens) produce poor embeddings — too much content dilutes the vector
- A query about a specific sub-paragraph may not rank well against a 10K-token chunk

### Decision: **Option A (parent-child) but with pragmatic limits**

The 93 rules over 1000 tokens (including one at 10K tokens) need splitting. But we keep it simple:

1. **Rules with sub-paragraphs**: Split at `(1)`, `(2)` level. Each child = one sub-paragraph (including any nested `(a)`, `(b)` beneath it). Preamble text prepended to child `(1)`.
2. **Standalone rules (no sub-paragraphs)**: Keep as-is. The rule is both parent and child.
3. **Very long standalone rules (>1500 tokens, no sub-paragraphs)**: Recursive text split as fallback. Rare.
4. **Very short children (<50 tokens)**: Merge with next sibling.

---

## Chunk Dataclass

```python
@dataclass
class Chunk:
    # Content
    chunk_id: str           # deterministic: "CMCOB_2.1.1R" or "CMCOB_2.1.1R_(1)"
    text: str               # the chunk text to embed
    parent_text: str        # full rule text (for LLM context after retrieval)

    # Citation metadata
    sourcebook: str         # "CMCOB"
    sourcebook_full: str    # "Claims Management: Conduct of Business Sourcebook"
    chapter: str            # "2"
    chapter_title: str      # "Conduct of business"
    section: str            # "2.1"
    section_title: str      # "General principles"
    rule_id: str            # "CMCOB 2.1.1R" (with type suffix)
    rule_type: str          # "R"
    sub_paragraph: str      # "(1)" or "" for standalone
    page: int

    # Features
    is_annex: bool
    is_table: bool
    defined_terms: list[str]
    cross_references: list[str]
```

---

## Algorithm

```
for each rule:
    skip if is_deleted or empty text

    parent_text = build_parent_text(rule)
    display_id = f"{rule.rule_id}{rule.rule_type}"

    if rule has sub-paragraphs at (1) level:
        preamble, subs = split_at_sub_paragraphs(rule.text)

        for each (sub_id, sub_text) in subs:
            if sub_id == "(1)" and preamble:
                sub_text = preamble + "\n" + sub_text

            child = Chunk(
                chunk_id = f"{display_id}_{sub_id}",
                text = sub_text,
                parent_text = parent_text,
                sub_paragraph = sub_id,
                ...all metadata from rule...
            )
            children.append(child)

        merge_short_children(children)

    else:
        # Standalone rule — it IS the chunk
        chunk = Chunk(
            chunk_id = display_id,
            text = rule.text,
            parent_text = parent_text,
            sub_paragraph = "",
            ...
        )
        chunks.append(chunk)

        # Fallback: if >1500 tokens and no sub-paragraphs, recursive split
        if too_long(chunk):
            chunks.extend(recursive_split(chunk))
```

---

## Key Design Decisions & Assumptions

### 1. Split only at level-1 sub-paragraphs `(1)`, `(2)`, `(3)`

We do NOT split at `(a)`, `(b)` level. Reason: `(a)` items are typically short qualifiers under a `(1)` paragraph. Splitting them would create too many tiny chunks. A child chunk `(1)` includes all its nested `(a)`, `(b)`, `(i)`, `(ii)` beneath it.

**Trade-off**: Some level-1 sub-paragraphs with many nested items can be long. But this is rare, and the reranker will handle precision at retrieval time.

### 2. Preamble text goes into child (1)

55% of rules with sub-paragraphs have introductory text before `(1)`. Example:

```
"A firm must not carry on regulated activity unless:     ← preamble
(1) condition A is met; and                              ← sub (1)
(2) condition B is met."                                 ← sub (2)
```

We prepend preamble to child `(1)` rather than making it a separate chunk. Reason: the preamble alone is often not a complete thought — "A firm must not carry on regulated activity unless:" is meaningless without the conditions.

**Trade-off**: Child `(1)` is slightly longer than the others. Acceptable.

### 3. parent_text includes section context header

```
[CMCOB > Chapter 2: Conduct of business > Section 2.1: General principles]

CMCOB 2.1.1R
A firm must act honestly, fairly and professionally...
```

This header is prepended to parent_text (NOT to the child text that gets embedded). It provides the LLM with hierarchical context when generating an answer. The embedded child text stays clean for precise vector matching.

**Trade-off**: parent_text is slightly larger. But it's metadata, not embedded — no impact on search.

### 4. Very short children (<50 tokens) merged with next sibling

Some sub-paragraphs are extremely short:
- `(1) [deleted]`
- `(a) integrity;`
- `(3) the FCA.`

These produce poor embeddings. We merge them with the next sibling until the combined text reaches 50 tokens.

**Trade-off**: Merged chunks span multiple sub-paragraph IDs. The `sub_paragraph` field becomes `"(1)+(2)"`. Citation is slightly less precise but the chunk is searchable.

### 5. Recursive split for very long standalone rules (>1500 tokens, no sub-paragraphs)

Only ~2-3% of rules are this long AND have no sub-paragraphs. We split at sentence boundaries with 50-token overlap. Each sub-chunk inherits the rule's metadata and gets `sub_paragraph = "part_1"`, `"part_2"`, etc.

**Assumption**: This is a rare fallback. Most long rules have sub-paragraphs and get split naturally.

### 6. chunk_id is deterministic

Format: `"{SOURCEBOOK}_{num}{type}"` or `"{SOURCEBOOK}_{num}{type}_(N)"`.
- `CMCOB_2.1.1R` — standalone rule
- `CMCOB_2.1.1R_(1)` — sub-paragraph 1
- `CMCOB_2.1.1R_(1)+(2)` — merged short children

This enables idempotent Weaviate upserts (deterministic UUID from chunk_id).

### 7. Defined terms and cross-references are per-child

When splitting a rule into children, each child gets its OWN defined_terms and cross_references (extracted from its text), not the parent's full set. This makes filtering and graph-building more precise.

### 8. We don't de-duplicate across sourcebooks

If two sourcebooks reference the same concept with similar text, both chunks are kept. De-duplication would lose regulatory specificity (BCOBS vs ICOBS may have subtly different rules on the same topic).

---

## Expected Output Scale

| Metric | Estimate |
|--------|----------|
| Input rules | 2,867 |
| Standalone chunks (60% of rules) | ~1,720 |
| Sub-paragraph children (40% of rules, avg ~3 subs each) | ~3,400 |
| Total chunks (rough) | ~5,000-6,000 |
| Weaviate free tier | 150K-300K objects |
| Usage | ~2-4% of free tier |

---

## External Cross-References (known limitation)

Out of 6,902 sourcebook-style references found across all rule text:

| Category | Count | % |
|----------|-------|---|
| Internal (our 10 sourcebooks) | 6,727 | 97.5% |
| External (not in our set) | 175 | 2.5% |

**17 external sourcebooks referenced:**

| Sourcebook | References | Description |
|---|---|---|
| SYSC | 55 | Senior Management Arrangements, Systems and Controls |
| GEN | 33 | General Provisions |
| SUP | 29 | Supervision |
| DISP | 10 | Dispute Resolution: Complaints |
| PRIN | 10 | Principles for Businesses |
| COLL | 7 | Collective Investment Schemes |
| PROD | 5 | Product Intervention and Product Governance |
| INSPRU | 5 | Prudential sourcebook for Insurers |
| PERG | 4 | Perimeter Guidance |
| COMP | 3 | Compensation |
| FEES | 3 | Fees Manual |
| Others (MIPRU, FUND, CONC, TP, UKLR, TC, REC, DEPP) | 9 each ≤2 | Various |

**How we handle them:**
- **Parser**: Our regex only captures references to our 10 sourcebooks. External references remain in the rule text (readable by LLM) but are not extracted as structured `cross_references` metadata.
- **Neo4j graph**: No nodes/edges for external rules. The graph covers internal references only.
- **Retrieval**: If a user asks about SYSC/PRIN/SUP, the system will correctly report no matching documents. The LLM can still cite external references from the text (e.g., "see SYSC 3.1.1R") as informational.
- **README note**: Document as a known limitation — "covers 10 FCA sourcebooks; references to external sourcebooks are preserved in text but cannot be resolved."

**Future improvement**: Expand the regex to capture all FCA sourcebook prefixes and create stub nodes in Neo4j for external references. This would show dependency boundaries — "this rule depends on something outside our document set."

---

## What the chunker does NOT do

- **No contextual enrichment** — we dropped Contextual Retrieval. Chunks are stored as-is.
- **No semantic chunking** — we split at structural boundaries (sub-paragraphs), not embedding distance.
- **No overlap between rules** — each rule is independent. Cross-references handle inter-rule connections via the Neo4j graph.
- **No embedding** — that's the embedder's job (next stage).
