# Chunker Design: Context-Enriched Flat Chunking

## Strategy

Every chunk is **self-contained** — no parent lookups needed at retrieval time. Each sub-paragraph gets a context header + preamble baked into its text. Embed it, retrieve it, send it directly to the LLM.

```
[COBS > Chapter 4: Communicating with clients > Section 4.2: Fair, clear and not misleading > COBS 4.2.1R]
A firm must ensure that when communicating a financial promotion, it:
(2) is fair;
```

## Input / Output

**Input**: `dict[str, list[ParsedRule]]` — 5,753 rules from `data/parsed/*_rules.json`
**Output**: `list[Chunk]` — flat list, each chunk self-contained with text + metadata

---

## Data Analysis (5,753 rules after parser + splitter)

| Metric | Value |
|--------|-------|
| Total rules | 5,753 |
| With sub-paragraphs | 2,091 (39%) |
| Without sub-paragraphs (standalone) | 3,308 (61%) |
| With preamble before (1) | 1,114 (53% of sub-para rules) |
| Longest rule | COBS 19.5.4G — 36,131 chars |
| Rules > 8K chars (genuinely long) | 12 |

### Simulated chunk output with Strategy 1

| Metric | Value |
|--------|-------|
| **Total chunks** | **8,601** |
| Median chunk size | 108 tokens |
| Mean chunk size | 148 tokens |
| P25 / P75 | 79 / 159 tokens |
| P95 | 345 tokens |
| <50 tokens | 318 (4%) |
| 50-150 tokens (sweet spot) | 5,879 (68%) |
| 150-300 tokens | 1,806 (21%) |
| 300-500 tokens | 404 (5%) |
| >1000 tokens | 37 (0.4%) |
| Header overhead | ~28 tokens avg |
| Preamble overhead (repeated) | ~60 tokens avg |

---

## How It Works

### For rules WITH sub-paragraphs (39%):

Each sub-paragraph becomes its own chunk with context header + preamble prepended:

```
Original rule COBS 4.2.1R:
  "A firm must ensure that a financial promotion:
   (1) is clear;
   (2) is fair;
   (3) is not misleading."

Produces 3 chunks:

Chunk 1:
  [COBS > Chapter 4 > Section 4.2: Fair, clear and not misleading > COBS 4.2.1R]
  A firm must ensure that a financial promotion:
  (1) is clear;

Chunk 2:
  [COBS > Chapter 4 > Section 4.2: Fair, clear and not misleading > COBS 4.2.1R]
  A firm must ensure that a financial promotion:
  (2) is fair;

Chunk 3:
  [COBS > Chapter 4 > Section 4.2: Fair, clear and not misleading > COBS 4.2.1R]
  A firm must ensure that a financial promotion:
  (3) is not misleading.
```

Key: **preamble is repeated in every child**, not just (1). Each chunk is independently meaningful.

### For rules WITHOUT sub-paragraphs (61%):

One chunk = header + full rule text.

### For table/annex rules:

**No sub-paragraph splitting.** Tables and annexes use (1), (2) as row/list identifiers within structured content. Splitting would break the table. Keep as one chunk with header.

### For very long chunks (>1000 tokens, no sub-paragraphs):

Recursive split at sentence boundaries. Each part gets the header + `[part N of M]`. Rare — only 37 chunks.

---

## Chunk Dataclass

```python
@dataclass
class Chunk:
    chunk_id: str           # "COBS_4.2.1R" or "COBS_4.2.1R_(2)"
    text: str               # self-contained: header + preamble + content
    
    # Metadata (for Weaviate filtering + citation)
    sourcebook: str         # "COBS"
    sourcebook_full: str    # "Conduct of Business Sourcebook"
    chapter: str            # "4"
    chapter_title: str      # "Communicating with clients"
    section: str            # "4.2"
    section_title: str      # "Fair, clear and not misleading"
    rule_id: str            # "COBS 4.2.1"
    rule_type: str          # "R"
    sub_paragraph: str      # "(2)" or "" for standalone
    page: int
    is_annex: bool
    is_table: bool
    defined_terms: list[str]
    cross_references: list[str]
```

No `parent_text` field — eliminated. Each chunk's `text` IS self-contained.

---

## Design Decisions

### 1. Preamble repeated in ALL children (not just child 1)

Without the preamble, `"(2) is fair;"` embeds to a meaningless vector. With preamble: `"A firm must ensure that a financial promotion: (2) is fair;"` — now it's a complete legal statement.

Cost: ~60 extra tokens per child. Worth it for retrieval quality.

### 2. Context header baked into chunk text

```
[COBS > Chapter 4: Communicating with clients > Section 4.2: Fair, clear > COBS 4.2.1R]
```

This is part of the embedded text, not just metadata. It means the embedding captures "this is about COBS Chapter 4, financial promotions, communicating with clients" — improving semantic search.

Cost: ~28 extra tokens per chunk. The header is also what the LLM sees — no need to construct context at retrieval time.

### 3. No sub-paragraph splitting for tables/annexes

Tables use `(1)`, `(2)` as row identifiers. Splitting would tear the table apart. Rules with `is_table=True` are kept whole.

### 4. Split only at level-1: `(1)`, `(2)`, `(3)`

Not at `(a)`, `(b)` level. Child `(1)` includes all nested `(a)`, `(b)`, `(i)`, `(ii)` beneath it. Deeper splitting creates too many tiny fragments.

### 5. Short children (<50 tokens) merged with next sibling

`"(1) [deleted]"` or `"(3) the FCA."` are too short. Merge with next sibling. Chunk ID becomes `"(1)+(2)"`.

### 6. Deterministic chunk IDs

`COBS_4.2.1R_(2)` — enables idempotent Weaviate upserts.

### 7. Defined terms and cross-references are per-chunk

Extracted from each chunk's text (after splitting), not from the full rule. More precise for filtering.

---

## What's NOT in this design

- **No parent-child lookups** — eliminated by baking context into every chunk
- **No sliding window** — considered but adds chunk duplication and complexity. The preamble repetition achieves the same goal (context) without overlapping content.
- **No multi-vector indexing** — Weaviate named vectors are powerful but complex to explain and debug. Flat chunks are simpler.
- **No semantic chunking** — structural boundaries (sub-paragraphs) are better for legal text than embedding-distance splits.

---

## External Cross-References (known limitation)

- 97.5% of cross-references are internal (our 10 sourcebooks)
- 2.5% reference 17 external sourcebooks (SYSC, GEN, SUP, etc.)
- External refs preserved in text but not extracted as structured metadata
- See PARSING_CAVEATS.md for full details
