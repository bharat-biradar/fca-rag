# PDF Processing Strategy: FCA Handbook Documents

## Document Set Overview

10 PDF sourcebooks from the UK Financial Conduct Authority (FCA) Handbook, February 2026 edition. ~3,022 total pages of financial regulatory content.

---

## Raw PDF Structure Analysis

### What PyMuPDF extraction revealed

We sampled pages from MAR, COBS, BCOBS, CASS, and ESG to understand the text layout. Two extraction methods were tested:
- `page.get_text()` — raw text in reading order
- `page.get_text('dict')` — structured output with font metadata per span

### Consistent elements across all sourcebooks

#### 1. Page header/footer (every page)

```
{SOURCEBOOK_ABBREV}
{SOURCEBOOK_ABBREV} {Full Name}
www.handbook.fca.org.uk
February 2026
```

Sometimes followed by dots (`.`) as visual separators. Must be stripped during parsing.

#### 2. Chapter markers

```
CHAPTER
COBS 1 Application
COBS 1 Application
```

Signalled by the word `CHAPTER` on its own line, followed by the chapter title (sometimes duplicated).

#### 3. Section headers

```
Section : COBS 1.1 General application
```

Consistent format: `Section : {SOURCEBOOK} {chapter}.{section} {title}`

#### 4. Rule identifiers

Main text format: `{SOURCEBOOK} {chapter}.{section}.{rule}`

Examples observed:
- `COBS 1.1.1` / `COBS 1.1.1AA` / `COBS 1.1.1A` / `COBS 1.1.1AER` (alpha suffixes)
- `BCOBS 1.1.1` / `BCOBS 1.1.2` / `BCOBS 1.1.4`
- `CASS 1.2.12` / `CASS 1.3.2`
- `ESG 1A.1.1` (alpha chapter numbers like `1A`)
- `MAR 1.3.1A` / `MAR 1.3.2` / `MAR 1.3.7`

#### 5. Rule type indicators (R/G/E/D/EU/UK)

**Critical finding:** The type indicator appears as a **separate line** after the rule text, NOT inline with the rule ID. This is a PDF layout artifact — the original document renders the type in a left margin column alongside the rule ID.

Full set of observed type indicators:
- `R` — Rule (legally binding)
- `G` — Guidance
- `E` — Evidential provision
- `D` — Direction
- `EU` — EU-origin rule (retained in UK law)
- `UK` — UK-specific provision

Example from BCOBS page 6-7:
```
BCOBS 1.1.6
A firm or a provider must not seek to exclude or restrict...
R
```

The `R` sits alone on a line after the rule text. Sometimes multiple type indicators cluster together at the bottom of a page:
```
R
R
G
G
```

#### 6. Sub-paragraph structure

Nested numbering up to 3 levels:
```
(1) references to customer are to the OPS or welfare trust...
    (a) the application of this sourcebook is in COBS 1 Annex 1, Part 3; and
    (b) the interpretation of certain words...
        (i) receive information about...
        (ii) attend a meeting with...
```

Levels: `(1)` → `(a)` → `(i)`

#### 7. Cross-references

Extensive, both within and between sourcebooks:
- `"see COBS 1.1.2R"`
- `"as defined in CASS 5.5.14R"`
- `"BCOBS 1 Annex 1 paragraph 1.1R applies"`

Cross-references use the full rule ID including type suffix, even when the source text separates the type indicator.

#### 8. Annex content

Annexes use **different numbering** — no sourcebook prefix:

```
Section : MAR 1 Annex 1 Provisions of the Buy-back and Stabilisation Regulation
relating to buy-back programmes

1.1.1
G
[deleted]

1.1.8
G
The FCA accepts as "adequate public disclosure":
(1) disclosure through a regulatory information service...
(2) the equivalent disclosure mechanism...
```

Rule IDs inside annexes are just `1.1.1`, `1.1.2`, etc. — the annex context (`MAR 1 Annex 1`) must be tracked and prepended to construct the full citation.

#### 9. Deleted rules

Common pattern:
```
1.1.1
G
[deleted]
```

These should be skipped during chunking — no value in embedding deleted content.

#### 10. Transitional provisions and schedules

End-of-sourcebook content:
- `CMCOB TP 1 Transitional Provisions`
- `CMCOB Sch 1 Record-keeping requirements`
- `ESG TP 1 Transitional provisions`

Follow similar structure patterns to main content.

---

## Font-Based Structure Detection

Using `get_text('dict')`, PyMuPDF reveals consistent font patterns that make structure detection reliable for **main text** pages:

| Element | Font | Size | Bold? | Italic? | Detection method |
|---------|------|------|-------|---------|-----------------|
| Page header (sourcebook name) | Helvetica-Bold | 19.2pt | Yes | No | Largest font on page — always strip |
| Page sub-header | Helvetica-Bold | 9.6pt | Yes | No | Helvetica bold — always strip |
| Page URL/date | Helvetica | 9.9pt | No | No | Helvetica regular — always strip |
| Section header | Arial-BoldMT | 12.0pt | Yes | No | Bold, 12pt — section boundary |
| Sub-section heading | Arial-BoldMT | 12.0pt | Yes | No | Bold, 12pt (same as section) |
| Rule ID (main text) | Arial-BoldMT | **7.2pt** | Yes | No | Bold, smallest font — rule boundary |
| Rule type (R/G/E/D/EU/UK) | ArialMT | **7.2pt** | No | No | Non-bold, smallest font, short token |
| Body text | ArialMT | 9.9pt | No | No | Regular weight, 9.9pt |
| FCA defined terms | Arial-ItalicMT | 9.9pt | No | Yes | Italic — bonus metadata for search |
| Annex footer label | Arial-BoldMT | 7.2pt | Yes | No | e.g., "MAR 1 Annex 1" |

**Important limitation:** In **annex content**, rule IDs and types are all 9.9pt (same as body text), not 7.2pt. Font-based detection alone does NOT work for annexes — regex fallback is needed.

---

## Table Handling

### Tables in the FCA documents

Tables appear throughout the sourcebooks: application tables, fee schedules, threshold tables, firm classification tables, compliance requirement tables.

### PyMuPDF table detection results

**`page.find_tables()` tested on known table pages:**

| Page | Tables found | Quality | Notes |
|------|-------------|---------|-------|
| COBS p12 | 1 table (2x3) | Partial — only captured header row | Table spans pages 12-13 |
| COBS p13 | 2 tables | Table 0: messy blob merging table + non-table text. Table 1: **properly parsed** (7 rows x 3 cols with clean data) | Duplicate detection, one good one bad |
| ESG p9 | 1 table (16x3) | **Good** — clean rows/columns, meaningful data | Firm type classification table |
| MAR p27 | 2 tables | **False positives** — annex layout misdetected as table (40 rows x 10 cols, lots of None). Raw get_text() was actually cleaner. | Annex content, not a real table |
| MAR p28 | 2 tables | **False positives** — same as p27, annex layout misread as table | Continuation of annex |

### Key findings on tables

1. **Raw `get_text()` on real table pages produces garbage** — columns interleave into nonsensical text (COBS p13)
2. **`find_tables()` works for real tables** (ESG p9, COBS p13 Table 1) — returns structured rows/columns
3. **`find_tables()` gives false positives on annex pages** — the annex margin layout triggers table detection even though it's not a table
4. **Tables can span multiple pages** — only the portion on each page is captured
5. **Duplicate detection occurs** — same content detected as both a blob (Table 0) and properly parsed (Table 1)

### Table handling strategy

1. Run `find_tables()` on every page
2. For each detected "table", validate it's real:
   - Check: does it have consistent columns? (not mostly None)
   - Check: is the col count reasonable? (>10 cols likely a false positive)
   - Check: is meaningful content present in cells?
3. Real tables → extract as structured data → convert to Markdown format
4. False positives → fall back to `get_text()` / `get_text('dict')` for that region
5. Tag table chunks with `is_table: true` in metadata

---

## PDF Parsing Tools Evaluated

### Tool comparison

| Tool | Tables | Multi-column/margin | Speed (3K pages, CPU) | Free? | License | Output |
|------|--------|--------------------|-----------------------|-------|---------|--------|
| **Docling** (IBM) | Best open-source (TableFormer model) | Known issue with multi-column layouts (GitHub #2067) | ~30-60 min | Yes | MIT | MD/JSON/HTML |
| **PyMuPDF4LLM** | Poor ("not anywhere near original") | Decent with layout extension | ~2-5 min (10x fastest) | Yes | AGPL | Markdown |
| **PyMuPDF raw** | Destroys structure | get_text('dict') gives font metadata | ~1-3 min | Yes | AGPL | Text/Dict |
| **LlamaParse** | Good (LLM-powered) | Struggles with nested structures | Cloud, fast | 10K pages/mo free | Commercial | Markdown |
| **Marker** | Weaker than Docling | Good (Surya OCR) | **~45 hours on CPU (impractical)** | GPL + restricted model weights | GPL | MD/JSON/HTML |
| **Unstructured** | Good (Hi-Res mode) | Element labelling (Title, Table, etc.) | Varies | OSS limited; API ~$3 | Apache 2.0 | HTML/JSON |
| **MinerU** | Good (SLANet+, cross-page merge) | Decent | GPU recommended | Apache 2.0 (with conditions) | Apache 2.0 | MD/JSON |
| **GROBID** | Not its strength | Scientific papers only — wrong domain | N/A | Apache 2.0 | Apache 2.0 | XML/TEI |
| **Azure Doc Intelligence** | Excellent | Excellent | Fast (cloud) | 500 pages/mo free (2 pages/doc limit) | Commercial | JSON |
| **Google Document AI** | Excellent | Excellent | Fast (cloud) | No free tier (billing required) | Commercial | JSON |

### Tool deep-dives

#### Docling (IBM) — Primary recommendation

- **What:** IBM's open-source document conversion toolkit. 58.6K GitHub stars.
- **Table handling:** Uses TableFormer vision-transformer model specifically for table structure recovery. Captures rows, columns, multi-level headers, cell boundaries. Can export tables as Pandas DataFrames.
- **Structure:** Produces unified `DoclingDocument` representation with layout, reading order, table boundaries, document hierarchy.
- **Risk for FCA docs:** Open GitHub issue #2067 reports multi-column layout extraction failures in financial documents. FCA's 3-column layout (rule ID | type indicator | rule text) is exactly this problematic pattern.
- **Install:** `pip install docling` (Python 3.10+)
- **New (2026):** IBM released Granite-Docling-258M, ultra-compact vision-language model using DocTags markup for structure preservation.
- **Integrations:** Native support for LangChain, LlamaIndex, CrewAI, Haystack.

#### PyMuPDF4LLM — Fast fallback

- **What:** Lightweight extension for PyMuPDF adding layout-aware Markdown extraction. Bundles `pymupdf_layout`.
- **Strength:** 10x faster than alternatives. No GPU, no cloud. Pure CPU.
- **Weakness:** Benchmarks show poor table extraction — "not anywhere near the original."
- **Install:** `pip install pymupdf4llm`
- **Best for:** Fast baseline; regular text pages where tables aren't present.

#### LlamaParse — Cloud alternative

- **What:** Cloud-based LLM-powered document parser by LlamaIndex.
- **Pricing:**
  - Free: 10,000 credits/month
  - Parse without AI: 1 credit/page (~10,000 pages free)
  - Cost-effective: 3 credits/page (~3,333 pages free)
  - Agentic (Sonnet 4.0): 90 credits/page (~111 pages free)
- **For 3,022 pages:** Basic tier fits in free allowance. Better tiers are tight.
- **Drawback:** Cloud-only, data leaves machine. Basic tier quality comparable to PyMuPDF.
- **Install:** `pip install llama-parse`

#### Marker — GPU-only practical

- **What:** Open-source PDF-to-Markdown converter using Surya OCR.
- **Critical limitation:** On CPU, ~54 seconds/page = **~45 hours for 3,022 pages**. Impractical without GPU (5GB VRAM per worker). Supports Apple MPS (Metal) which helps on Mac.
- **Tables:** Preserves tables but weaker complex table support than Docling.
- **License:** GPL code + modified AI Pubs license for model weights (free for research/personal/startups under $2M).
- **Install:** `pip install marker-pdf`

#### Unstructured.io — Element detection

- **What:** Document converter that returns semantically labeled elements (Title, NarrativeText, Table, ListItem, Header, etc.).
- **Tables:** When `pdf_infer_table_structure=True`, outputs tables as HTML strings preserving structure.
- **Gap:** Significant quality gap between open-source and API versions. Hi-Res models only available via API.
- **Pricing:** Open-source free but limited. API: $1/1,000 pages (~$3 for corpus).
- **Install:** `pip install unstructured`

#### MinerU — Strong alternative

- **What:** OpenDataLab's document parser. Uses PaddlePaddle for layout detection + SLANet+ for tables.
- **Strength:** Cross-page table merging, image recognition in tables, truncated paragraph merging.
- **2026 update:** MinerU2.5-Pro model claims state-of-the-art parsing accuracy.
- **Drawback:** PaddlePaddle installation can be tricky. GPU recommended.
- **Install:** `pip install mineru`

### Tools eliminated

| Tool | Why eliminated |
|------|---------------|
| GROBID | Purpose-built for scientific publications. 68 labels for academic paper elements (abstract, bibliography). Wrong domain for regulatory docs. |
| pdf2md (opengovsg) | Lightweight Node.js CLI. Limited structure awareness, won't handle FCA complexity. |
| MarkItDown (Microsoft) | General-purpose. Good breadth (PDF, Word, PPT) but not specialized for complex PDFs. |
| Pandoc | Good for well-structured docs. No layout detection for complex PDFs. |
| Azure Doc Intelligence | Free tier: 500 pages/month with 2-page-per-document limit. Would take ~6 months. Pay: ~$4.50-$30. |
| Google Document AI | No free tier. Requires billing-enabled GCP project. ~$30 for corpus. |

---

## Parser Comparison: Docling vs LlamaParse (tested on PDCOB.pdf)

Both tools were tested on PDCOB.pdf (127 pages). The results were decisive.

### Head-to-head on identical content

**Section 2.3 — simple rules (2 rules, no sub-paragraphs):**

Docling (broken — IDs, texts, and types separated into three groups):
```
PDCOB 2.3.1

PDCOB 2.3.2

In this sourcebook, references to an active election...

A failure by a customer to change a default option...

R

G
```

LlamaParse (perfect — ID + type + text all inline):
```
**PDCOB 2.3.1** <u>R</u> In this sourcebook, references to an active election by a *customer*...

**PDCOB 2.3.2** <u>G</u> A failure by a *customer* to change a default option...
```

**Section 5.4 — complex rules (3 rules, nested sub-paragraphs):**

Docling: All 3 rule IDs listed first, then types, then text. Sub-paragraphs get wrong numbering (`3.`, `4.` instead of `(a)`, `(b)`). Rule IDs misdetected as headings (`## PDCOB 5.4.1`). Cannot reliably map IDs to their texts.

LlamaParse: Each rule with ID + type + text inline. Sub-paragraphs `(1)`, `(a)`, `(i)` correctly nested under their parent rule. Clean, parseable.

### Quantitative comparison

| Metric | Docling | LlamaParse (3cr) |
|--------|---------|-----------------|
| Processing time | 31 sec (local CPU) | Cloud, ~1-2 min |
| Output size | 225,656 chars | 138,213 chars (39% smaller) |
| `<!-- image -->` noise | **253** | 0 |
| `<!-- layout -->` noise | 0 | 50 |
| Header repeats | 1 | 128 (≈ page breaks, useful!) |
| Logo/icon noise | 0 | 57+10 (`PDCOB logo`, `Square icon`) |
| **Rule ID + type + text inline** | **No** — separated, broken mapping | **Yes** — trivial to parse |
| **Defined terms preserved** | No | Yes — `*customer*`, `*firm*` (italic) |
| **Sub-paragraph structure** | Broken numbering | Correct `(1)`, `(a)`, `(i)` |
| **Rule type inline with ID** | No — on separate lines | Yes — `<u>R</u>` or `<span>G</span>` |

### Decision: LlamaParse (cost-effective tier, 3 credits/page)

**Why LlamaParse wins:**
1. Rule ID + type + text inline — the single most important property for our chunking pipeline
2. Defined terms preserved as italic (`*term*`) — useful metadata for search
3. Sub-paragraphs correctly nested — no re-association logic needed
4. 39% smaller output — less noise, more signal
5. Post-processing is ~50 lines of regex vs hundreds of lines of re-association code for Docling

**Credit math:** 3,022 pages × 3 credits = 9,066 of 10,000 free monthly credits. Fits.

**Trade-off acknowledged:** Data leaves the machine (cloud processing). Acceptable for public FCA Handbook documents and a take-home assignment.

---

## Finalized Parsing Pipeline

### Step 1: LlamaParse (PDF → Markdown)

Run all 10 PDFs through LlamaParse cost-effective tier (3 credits/page):
- Input: 10 PDF files, ~3,022 pages total
- Output: 10 Markdown files
- Cost: 9,066 of 10,000 free credits
- Time: ~5-10 min (cloud processing)

### Step 2: Noise stripping

Remove known noise patterns from LlamaParse markdown output:

```python
# Header repeats (page breaks)
text = re.sub(r'{SOURCEBOOK}.*?www\.handbook\.fca\.org\.uk.*?February 2026', '', text)
# Logo/icon text
text = re.sub(r'{SOURCEBOOK} logo|Square icon|Square logo', '', text)
# Layout comments
text = re.sub(r'<!--\s*layout:.*?-->', '', text)
# Empty headings
text = re.sub(r'##\s*\n', '', text)
```

### Step 3: Metadata extraction from clean markdown

All metadata is extractable via regex from the LlamaParse output:

| Metadata field | Pattern | Example match | Count (PDCOB) |
|---|---|---|---|
| **Chapter** | `CHAPTER\s*\n\s*\n\s*({SB} (\d+)\s+(.+))` | `PDCOB 2 General principles` | 11 |
| **Section** | `Section\s*:\s*({SB}\s+([\d.]+)\s+(.+))` | `PDCOB 2.1 The customer's best interests rule` | 86 |
| **Rule ID** | `\*\*({SB}\s+[\d.]+\w*)\*\*` | `PDCOB 2.3.1` | 169 |
| **Rule type** | `(?:<u>\|<span[^>]*>)(\w+)(?:</u>\|</span>)` | `R`, `G`, `E`, `D`, `EU`, `UK` | inline with ID |
| **Rule text** | Everything after type until next `**{SB}` | Full rule body | inline |
| **Sub-paragraphs** | `^\s*\((\d+\|[a-z]\|[ivxl]+)\)` in text | `(1)`, `(a)`, `(i)` | 234 |
| **Defined terms** | `\*([^*]{3,40})\*` within rule text | `*customer*`, `*firm*` | 645 unique |
| **Cross-references** | `({SB}\s+[\d.]+\w*(?:\([^)]+\))?)` | `PDCOB 1.3.1R(1)` | 413 unique |
| **Annexes/Schedules** | `(?:Annex\|Sch\|TP)\s+\d+` in section headers | `Sch 1 Right of action for damages` | detectable |
| **Page boundaries** | Header repeat count | ~1 per page | 128 ≈ 127 pages |
| **Deleted rules** | `\[deleted\]` in rule text | skip during chunking | detectable |

### Step 4: Build hierarchy

Using extracted metadata, construct the document tree:

```
Sourcebook (from filename)
  └── Chapter (from CHAPTER markers)
       └── Section (from Section : markers)
            └── Rule/Parent (from **RULE_ID** <u>TYPE</u> text)
                 └── Sub-paragraph/Child (from (1), (a), (i) within text)
```

### Step 5: Page number mapping (optional)

If exact page numbers are needed (nice-to-have, not essential for FCA citations):
- Count header repeats as approximate page boundaries (128 repeats ≈ 127 pages)
- Or: one-time PyMuPDF pass to map rule IDs → page numbers via text search

FCA rules are always cited by rule ID (`COBS 2.1.1R`), never by page number. Page numbers are informational only.

### Post-processing summary

LlamaParse's inline format dramatically simplifies post-processing vs Docling:

| Task | Docling (would need) | LlamaParse (actual) |
|------|---------------------|-------------------|
| Associate rule IDs with types | Complex proximity matching across separated blocks | Already inline: `**ID** <u>TYPE</u>` |
| Associate rule IDs with text | Complex block-order reconstruction | Already inline: `**ID** <u>TYPE</u> text...` |
| Detect rule boundaries | Font-based + regex, inconsistent across main/annex | Single regex: `\*\*{SB} [\d.]+\*\*` |
| Preserve sub-paragraph nesting | Re-number broken lists | Already correct in output |
| Extract defined terms | Not preserved | Already italic: `*term*` |
| Strip noise | 253 `<!-- image -->` markers | ~50 `<!-- layout -->` + header repeats |

---

## Key References and Sources

### PDF parsing tool comparisons
- [Best PDF Parsers for AI and RAG Workflows in 2026](https://www.firecrawl.dev/blog/best-pdf-parsers) — Comprehensive comparison of Docling, LlamaParse, Marker, Unstructured, MinerU
- [Best PDF Parser for RAG Pipelines (April 2026)](https://www.unsiloed.ai/blog/best-pdf-parser-rag-pipelines) — Benchmarks across parser tools
- [5 Best Document Parsers in 2026 (Tested on Financial PDFs)](https://www.f22labs.com/blogs/5-best-document-parsers-in-2025-tested/) — Financial document-specific testing
- [A Comparative Study of PDF Parsing Tools (arXiv)](https://arxiv.org/html/2410.09871v1) — Academic benchmark, 55-point accuracy gap between easy (legal) and hard (academic) domains
- [The State of PDF Parsing: What 800+ Documents Taught Us](https://www.applied-ai.com/briefings/pdf-parsing-benchmark/) — Large-scale benchmark
- [From PDFs to Markdown: Evaluating Document Parsers for Air-Gapped RAG Systems](https://dev.to/ashokan/from-pdfs-to-markdown-evaluating-document-parsers-for-air-gapped-rag-systems-58eh)
- [Document Parsing for Production RAG](https://medium.com/@manikandan_t/document-parsing-for-production-rag-architecture-tradeoffs-and-when-to-use-what-7a89ab0af7b7)

### Individual tool references
- [Docling GitHub](https://github.com/docling-project/docling) — 58.6K stars, MIT license
- [Docling Technical Report (arXiv)](https://arxiv.org/pdf/2408.09869)
- [Docling Multi-Column Layout Issue #2067](https://github.com/docling-project/docling/issues/2067) — The specific issue affecting FCA-style multi-column layouts
- [Docling Is Quietly Changing How We Build AI Document Pipelines](https://medium.com/@sanikachavan1806/docling-is-quietly-changing-how-we-build-ai-document-pipelines-4ee9e6f57164)
- [IBM Granite-Docling-258M Announcement](https://www.ibm.com/new/announcements/granite-docling-end-to-end-document-conversion) — New compact VLM for document conversion
- [PyMuPDF4LLM Documentation](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)
- [PyMuPDF-Layout: 10x Faster PDF Parsing Without GPUs](https://pymupdf.io/blog/pymupdf-layout-10-faster-pdf-parsing-without-gpus)
- [LlamaParse Pricing](https://www.llamaindex.ai/pricing)
- [LlamaParse V2 Announcement](https://www.llamaindex.ai/blog/introducing-llamaparse-v2-simpler-better-cheaper)
- [Marker GitHub](https://github.com/datalab-to/marker) — GPL license, Surya OCR
- [Unstructured: Table Extraction from PDF](https://docs.unstructured.io/examplecode/codesamples/apioss/table-extraction-from-pdf)
- [Unstructured Pricing](https://unstructured.io/pricing)
- [MinerU GitHub](https://github.com/opendatalab/mineru) — Apache 2.0
- [GROBID Documentation](https://grobid.readthedocs.io/en/latest/Introduction/)
- [Azure Document Intelligence Pricing](https://azure.microsoft.com/en-us/pricing/details/document-intelligence/)
- [Google Document AI Pricing](https://cloud.google.com/document-ai/pricing)

### FCA Handbook structure
- [FCA Handbook Reader's Guide](https://www.fca.org.uk/publication/handbook/readers-guide_0.pdf) — Official guide to reading the FCA Handbook structure

### RAG chunking for legal/regulatory documents
- [Best Chunking Strategies for RAG in 2026](https://www.firecrawl.dev/blog/best-chunking-strategies-rag)
- [Best Open-Source PDF-to-Markdown Tools in 2026](https://themenonlab.blog/blog/best-open-source-pdf-to-markdown-tools-2026)

---

## Benchmark insights relevant to our use case

- Legal/financial documents consistently score **high F1** (~95%) across most parsing tools — these are "easy" documents compared to academic papers (~40%)
- PyMuPDF and pypdfium consistently perform well on legal/financial document types for **basic text extraction**
- LlamaParse achieves ~78% edit similarity overall; higher on structured legal docs
- Tables without borders, nested tables, and multi-column layouts remain the **hardest challenges** across all parsers
- The 3-column FCA layout (rule ID | type | text) is specifically the pattern that breaks most parsers
- **Our own testing confirmed:** Docling's multi-column issue (#2067) breaks FCA margin layout. LlamaParse handles it correctly, producing inline rule ID + type + text.

---

## Next steps

- [x] ~~Install Docling and test on FCA pages~~ — tested on PDCOB.pdf, 31 sec, but broken rule mapping
- [x] ~~Compare with LlamaParse~~ — LlamaParse wins decisively (inline rule format, correct sub-paragraphs)
- [x] ~~Finalize parser choice~~ — **LlamaParse cost-effective tier (3 credits/page)**
- [x] ~~Output format decision~~ — **JSON primary** (page numbers, pre-parsed table rows, element types). Markdown kept as human-readable reference.
- [ ] Run LlamaParse on all 10 PDFs — JSON output (9,066 of 10,000 free credits)
- [ ] Build post-processing pipeline (iterate JSON items, extract metadata, merge cross-page content)
- [ ] Integrate with chunking strategy (see CHUNKING_STRATEGY.md) — hierarchy building + child/parent splitting

---

## Future Improvement: Hybrid Parsing Pipeline

The current approach (LlamaParse JSON → post-process → chunk) is pragmatic for the time constraint. A more robust production pipeline would be:

```
[Raw PDF] → [Layout Parser (Docling/Marker)] → [Structured JSON with layout elements]
                                                         │
                                                         ▼
                                              [Metadata Extraction & Filtering]
                                              - Classify elements (rule, guidance, table, annex)
                                              - Extract rule IDs, types, cross-references
                                              - Filter noise (headers, footers, decorative elements)
                                                         │
                                                         ▼
                                              [Convert to Clean Markdown per chunk]
                                              - Tables → Markdown tables
                                              - Rules → structured text with metadata
                                              - Preserve defined terms (italic)
                                                         │
                                                         ▼
                                              [Embed & Store]
```

**Why this is better:**
- Layout parser gives bounding boxes and element classification at the PDF level (more accurate than LLM-based parsing)
- Metadata extraction happens on structured elements, not raw text
- Clean markdown output is LLM-friendly for generation
- Each step is testable independently

**Why we didn't do this now:** Time constraint. Docling's multi-column issue (#2067) would require workarounds for FCA margin layout. LlamaParse JSON gives us 80% of the benefit with 20% of the effort.

**To discuss in walkthrough:** "In production, I'd replace LlamaParse with a self-hosted layout parser (Docling or MinerU) to eliminate the cloud dependency and get bounding-box-level precision. The hybrid pipeline would also handle edge cases like cross-page table merging more reliably."
