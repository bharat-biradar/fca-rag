# Chunking Strategy: FCA Handbook Documents

## Document Analysis

### What we're working with

10 PDF sourcebooks from the UK FCA Handbook (February 2026 edition), totalling ~3,022 pages of financial regulatory content.

| Document | Pages | Domain |
|----------|-------|--------|
| COBS | 970 | General conduct of business (MiFID, inducements, suitability) |
| MCOB | 557 | Mortgages & home finance |
| CASS | 424 | Client assets, custody, client money |
| MAR | 294 | Market conduct, market abuse, trading venues |
| ICOBS | 192 | Insurance conduct of business |
| FPCOB | 154 | Funeral plan conduct of business |
| BCOBS | 146 | Banking conduct of business |
| PDCOB | 127 | Pensions dashboards |
| CMCOB | 93 | Claims management conduct of business |
| ESG | 65 | Environmental, social & governance / TCFD |

### Observed text structure from PDF extraction

Every page in every sourcebook follows a consistent layout. Here is what PyMuPDF `page.get_text()` produces:

#### 1. Page header/footer (repeated on every page — must strip)

```
COBS
COBS Conduct of Business Sourcebook
www.handbook.fca.org.uk
February 2026
```

This appears at the top of every page. Sometimes followed by dots (`.`) as visual separators. Must be removed during parsing to avoid polluting chunks.

#### 2. Chapter markers

```
CHAPTER
COBS 1 Application
COBS 1 Application
```

Chapters are signalled by the word `CHAPTER` on its own line, followed by the chapter title (sometimes duplicated).

#### 3. Section headers

```
Section : COBS 1.1 General application
```

Consistent format: `Section : {SOURCEBOOK} {chapter}.{section} {title}`

#### 4. Rule identifiers and text

Rules follow this pattern:

```
COBS 1.1.1
This sourcebook applies to a firm with respect to the following activities carried on from an
establishment maintained by it, or its appointed representative, in the United Kingdom:
(1) [deleted]
(2) designated investment business;
(3) long-term insurance business in relation to life policies;
and activities connected with them.
```

The rule ID format is: `{SOURCEBOOK} {chapter}.{section}.{rule}`

Examples observed:
- `COBS 1.1.1` / `COBS 1.1.1AA` / `COBS 1.1.1A` (suffixed variants)
- `BCOBS 1.1.1` / `BCOBS 1.1.2` / `BCOBS 1.1.4`
- `CASS 1.2.12` / `CASS 1.3.2`
- `ESG 1A.1.1`

Some rules have alpha suffixes: `COBS 1.1.1AR`, `COBS 1.1.1AER`, `BCOBS 1.1.5A`

#### 5. Rule type indicators (R/G/E/D)

**Critical finding**: The rule type (R = Rule, G = Guidance, E = Evidential, D = Direction) appears as a **separate line** after the rule text, not inline with the rule ID. This is a PDF layout artifact — the original document has the type indicator in a margin column.

Example from BCOBS page 6-7:
```
BCOBS 1.1.6
A firm or a provider must not seek to exclude or restrict, or rely on any exclusion or
restriction of, any duty or liability it may have to a banking customer...
R
```

The `R` sits alone on a line after the rule text. Sometimes multiple rule type indicators appear grouped:
```
R
R
G
G
```

This means we need to associate rule types with their rules via proximity, not by parsing them inline.

#### 6. Sub-paragraph structure

Rules contain nested sub-paragraphs:
```
(1) references to customer are to the OPS or welfare trust...
    (a) the application of this sourcebook is in COBS 1 Annex 1, Part 3; and
    (b) the interpretation of certain words...
        (i) receive information about...
        (ii) attend a meeting with...
```

Nesting levels: `(1)` → `(a)` → `(i)` — up to 3 levels deep.

#### 7. Tables

Tables appear in the source PDFs (fee schedules, application tables, threshold tables). PyMuPDF's `get_text()` **destroys table structure** — columns get interleaved into nonsensical text.

Example from COBS page 12-13 (an application table):
```
Section / chapter
Application in relation
to deposits
The rules specified in
A MiFID investment
(1)
COBS 1.1.1A
G
R
R
```

This was originally a clean table with columns for section, application scope, and firm type. After text extraction, columns are mixed together and illegible.

Example from ESG page 9 (a firm type table):
```
Column 1: type of firm
Column 2: TCFD in-scope business
Part A: Asset managers
Any firm
Portfolio management
UK UCITS management
company
Managing a UK UCITS
```

Tables need special handling — either PyMuPDF's `page.find_tables()` API or conversion to markdown format.

#### 8. Cross-references

Extensive cross-referencing between and within sourcebooks:
- `"see COBS 1.1.2R"`
- `"as defined in CASS 5.5.14R"`
- `"BCOBS 1 Annex 1 paragraph 1.1R applies"`
- `"This Annex belongs to CASS 5.5.14 R"`

These references use the full rule ID including the type suffix, even when the source text separates the type indicator.

#### 9. Annexes

Sourcebooks contain Annexes with slightly different naming:
- `BCOBS 1 Annex 1 Structured deposit business`
- `COBS 1 Annex 1 Application (see COBS 1.1.2R)`
- `CASS 5 Annex 1 Segregation of designated investments`
- `FPCOB 3 Annex 1 Form of a beneficial trust`

Annex content follows the same rule structure but with different numbering (e.g., paragraph `1.1`, `1.2`, `1.3` within the annex).

#### 10. Transitional Provisions and Schedules

Some sourcebooks end with:
- `CMCOB TP 1 Transitional Provisions`
- `CMCOB Sch 1 Record-keeping requirements`
- `CMCOB Sch 2 Notification and reporting requirements`
- `ESG TP 1 Transitional provisions`

These follow similar structure patterns.

---

## Chunking Strategy: Hierarchical Chunking with Parent-Child Retrieval

### The hierarchy

```
Module (Sourcebook)           ← COBS, BCOBS, CASS, etc.
  └── Chapter                 ← COBS 1: Application
       └── Section            ← COBS 1.1: General application
            └── Rule (PARENT) ← COBS 1.1.1R
                 ├── Sub-para (CHILD) ← (1) ...
                 ├── Sub-para (CHILD) ← (2) ...
                 │    ├── (a) ...
                 │    └── (b) ...
                 └── Sub-para (CHILD) ← (3) ...
```

### Embedding strategy

**Embed the CHILD chunks** (smallest logical units — individual rules or sub-paragraphs) for precise semantic matching. Small chunks produce specific embeddings that won't get diluted by surrounding content.

**Retrieve the PARENT** on match. When a child chunk matches a query, return the full parent rule (with all its sub-paragraphs) plus the hierarchical metadata to the LLM. This gives the LLM enough context to answer properly.

### Chunk levels defined

**Child chunk** (what gets embedded and searched):
- A single rule with no sub-paragraphs: the full rule text
- A sub-paragraph `(1)`, `(2)`, etc. of a rule that has sub-paragraphs
- Target size: 50-500 tokens (natural rule/sub-para length)
- Very short children (<50 tokens) get merged with adjacent siblings

**Parent chunk** (what gets sent to the LLM on retrieval):
- The full rule with all its sub-paragraphs
- If a rule is very short, the parent can be the full section (multiple rules)
- Target size: 200-1500 tokens
- Includes section title for context

### Metadata per chunk

Every chunk (child and parent) carries metadata that doubles as the citation:

```json
{
  "sourcebook": "COBS",
  "sourcebook_full": "Conduct of Business Sourcebook",
  "chapter": "2",
  "chapter_title": "Conduct of business obligations",
  "section": "2.1",
  "section_title": "Acting honestly, fairly and professionally",
  "rule_id": "COBS 2.1.1",
  "rule_type": "R",
  "sub_paragraph": "(1)(a)",
  "page": 45,
  "is_annex": false,
  "is_table": false,
  "parent_rule_id": "COBS 2.1.1R",
  "parent_text": "full text of the parent rule...",
  "chunk_type": "child"
}
```

**Citation output** is just the metadata: `Source: COBS 2.1.1R(1)(a), page 45`

No post-processing extraction needed — the citation is structural, not inferred.

### Scale estimate

| Metric | Estimate |
|--------|----------|
| Total pages | ~3,022 |
| Estimated rules across all sourcebooks | ~3,000 - 5,000 |
| Estimated child chunks (after sub-para splitting) | ~8,000 - 15,000 |
| Average child chunk size | ~100-300 tokens |
| Average parent chunk size | ~300-800 tokens |
| Weaviate free tier capacity | 150,000 - 300,000 objects |
| Usage of free tier | ~5-10% |

Well within all limits.

---

## PDF Parsing Pipeline

### Step 1: Text extraction + header stripping

Extract text with PyMuPDF (`fitz`), preserving page numbers. Strip the repeated header/footer from every page:

```
Strip pattern (per page):
  {SOURCEBOOK}
  {SOURCEBOOK} {full name}
  www.handbook.fca.org.uk
  February 2026
  [optional dots/whitespace]
```

### Step 2: Table detection and preservation

Before text-based chunking, detect tables using PyMuPDF's `page.find_tables()` API.

- If a table is detected, extract it as structured data (list of rows/columns)
- Convert to Markdown table format for embedding
- Tag the chunk as `"is_table": true` in metadata
- Keep tables intact — never split a table across chunks

Fallback: if `find_tables()` misses a table, heuristic detection based on text patterns (multiple short lines with column-like alignment, `Column 1:` / `Column 2:` markers).

### Step 3: Rule boundary detection

Parse the extracted text to identify rule boundaries using regex patterns:

```
Rule ID pattern (covers all observed variants):
^(BCOBS|CASS|CMCOB|COBS|ESG|FPCOB|ICOBS|MAR|MCOB|PDCOB)\s+
(\d+[A-Z]?)\.                     # chapter (e.g., 1, 1A, 5A)
(\d+[A-Z]?)\.                     # section (e.g., 1, 1A)  
(\d+[A-Z]*)                       # rule (e.g., 1, 1A, 1AA, 1AER)
```

Also detect:
- Section headers: `^Section : {SOURCEBOOK} ...`
- Chapter headers: `^CHAPTER` or `^{SOURCEBOOK} \d+ [A-Z]`
- Annex markers: `{SOURCEBOOK} \d+ Annex \d+`
- Schedule markers: `{SOURCEBOOK} Sch \d+`
- Transitional provisions: `{SOURCEBOOK} TP \d+`

### Step 4: Rule type association

Since rule types (R/G/E/D) appear on separate lines after the rule text:

1. After identifying a rule's text span (from one rule ID to the next)
2. Look for isolated single-character lines matching `^[RGDE]$` within or immediately after that span
3. Associate the first such character with the rule
4. Trim the type indicator from the chunk text

Edge case: Some rules reference types inline in cross-references (e.g., `"COBS 1.1.2R"`). These are NOT type indicators for the current rule — only isolated single-char lines are.

### Step 5: Sub-paragraph splitting (child chunk creation)

Within each rule's text, detect sub-paragraphs:

```
Level 1: ^\s*\(\d+\)          →  (1), (2), (3)
Level 2: ^\s*\([a-z]\)        →  (a), (b), (c)
Level 3: ^\s*\([ivxl]+\)      →  (i), (ii), (iii), (iv)
```

Splitting rules:
- If a rule has sub-paragraphs at level 1, each `(1)`, `(2)`, etc. becomes a child chunk (including any nested (a), (b) beneath it)
- If a rule has NO sub-paragraphs, the entire rule is both parent and child
- Introductory text before `(1)` is included in child (1) or kept as a separate preamble child

### Step 6: Merge short chunks, split long chunks

**Merge**: Children under 50 tokens get merged with the next sibling. Very short rules (<50 tokens) get merged with the next rule in the same section.

**Split**: Parents over 1500 tokens get recursive-split as a fallback, with each sub-chunk inheriting the parent's metadata and carrying a `part` indicator (e.g., `part 1 of 3`).

### Step 7: Build hierarchy and embed

1. Build the full hierarchy tree: Sourcebook → Chapter → Section → Rule → Sub-paragraph
2. For each child chunk, create the embedding text (the child's text content)
3. Store in Weaviate with:
   - The child text (for BM25 + vector search)
   - The dense embedding vector (from BGE-M3)
   - All metadata fields (for filtering, citation, and parent retrieval)
   - The parent text (stored as a metadata field, not embedded separately)

---

## How each RAG approach uses the chunks

### Approach 1: Hybrid Search + Reranking
1. Query hits Weaviate hybrid search (BM25 on child text + dense vector similarity)
2. Top-50 child chunks retrieved
3. Cross-encoder reranks to top-5
4. For each top-5 child, pull the parent text from metadata
5. Send parent texts + query to LLM
6. Citation: return rule IDs from chunk metadata

### Approach 2: Contextual Retrieval
1. **At index time**: For each child chunk, use an LLM to generate a 1-2 sentence context based on the parent and section title. Prepend this context to the child text before embedding.
   - Example: `"This chunk is from COBS Chapter 2, Section 2.1 'Acting honestly, fairly and professionally'. It describes the general obligation for firms conducting designated investment business."` + original child text
2. Store the contextually-enriched text + embedding in a **separate Weaviate collection** (so Approach 1 has the un-enriched chunks for fair comparison)
3. At query time: same hybrid search + reranking as Approach 1, but on the enriched collection
4. For each top-5 child, pull parent text from metadata
5. Send parent texts + query to LLM
6. Citation: return rule IDs from chunk metadata

### Approach 3: Agentic RAG
1. Agent receives query, reasons about it
2. For simple queries: single hybrid search + rerank (same as Approach 1)
3. For cross-sourcebook queries: decomposes into sub-queries per sourcebook, retrieves for each, merges results
4. For ambiguous queries: reformulates query, retrieves, evaluates quality, may re-retrieve
5. Uses the same underlying chunk collection as Approach 1
6. Parent text retrieval and citation same as above

**Key point**: All three approaches use the same base chunks and metadata. The chunking is shared. Only the retrieval mechanism differs. This ensures a fair, controlled comparison.

---

## Challenges and mitigations

| Challenge | Mitigation |
|-----------|-----------|
| Rule type (R/G/E) on separate line | Regex-based proximity matching: isolated single-char lines `[RGDE]` after rule text |
| Tables destroyed by text extraction | Use PyMuPDF `find_tables()` first; convert to Markdown; tag as `is_table` in metadata; keep intact |
| Cross-references between sourcebooks | Store cross-ref targets as metadata where detectable; Agentic RAG can follow them at query time |
| Annex numbering differs from main text | Detect annex boundaries separately; use `{SOURCEBOOK} {chapter} Annex {annex_num}` as prefix for rule IDs within annexes |
| Very short rules (<50 tokens) | Merge with adjacent sibling or include section context |
| Very long rules (>1500 tokens) | Recursive split as fallback, preserving metadata on each sub-chunk |
| Some rules have no explicit sub-paragraphs | Entire rule becomes both parent and child — embed the full rule text |
| Header/footer noise on every page | Strip using sourcebook-specific regex pattern before any chunking |
| Dot separators between sections | Strip lines consisting only of dots/whitespace |

---

## Tools and dependencies

| Need | Tool | Why |
|------|------|-----|
| PDF text extraction | PyMuPDF (`fitz`) | Already installed. Gives text + page numbers. Good enough for structured text extraction. |
| Table extraction | PyMuPDF `page.find_tables()` | Built into PyMuPDF, no extra dependency. Returns structured table data. |
| Rule boundary detection | Python `re` (regex) | The document structure is regular enough for regex. No NLP/ML needed. |
| Markdown table conversion | Custom Python function | Simple: take table rows/cols → format as Markdown `|` table |

No additional dependencies beyond PyMuPDF (already installed) and Python stdlib.
