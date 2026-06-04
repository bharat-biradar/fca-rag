# RAG QA Service — Planning & Trade-offs

## Assignment Summary

Build a retrieval-augmented QA service over a provided document set that:
- Answers questions grounded in the documents with citations
- Compares **at least 3 different retrieval approaches** with reasoned decision on which to ship
- Includes architecture diagram, evaluation harness, unit/integration tests
- 24-hour timeline

---

## Document Set: UK FCA Handbook

10 PDF sourcebooks from the UK Financial Conduct Authority (FCA) Handbook, February 2026 edition. **~3,022 total pages** of financial regulatory content.

| Document | Full Name | Pages | Domain |
|----------|-----------|-------|--------|
| BCOBS | Banking: Conduct of Business Sourcebook | 146 | Banking conduct rules |
| CASS | Client Assets Sourcebook | 424 | Client asset protection, custody, money rules |
| CMCOB | Claims Management: Conduct of Business | 93 | Claims management regulation |
| COBS | Conduct of Business Sourcebook | 970 | General conduct (MiFID, inducements, suitability) |
| ESG | Environmental, Social & Governance | 65 | Climate/sustainability disclosures (TCFD) |
| FPCOB | Funeral Plan: Conduct of Business | 154 | Funeral plan regulation |
| ICOBS | Insurance: Conduct of Business | 192 | Insurance conduct rules |
| MAR | Market Conduct | 294 | Market abuse, trading venues, MTFs/OTFs |
| MCOB | Mortgages & Home Finance: Conduct of Business | 557 | Mortgage regulation |
| PDCOB | Pensions Dashboards: Conduct of Business | 127 | Pensions dashboards regulation |

### Document Characteristics

**Structure**: Highly hierarchical — each rule has a unique identifier (e.g., `COBS 2.1.1R`, `BCOBS 4.1.2G`). Suffixes indicate type: `R` = Rule (binding), `G` = Guidance, `E` = Evidential provision, `D` = Direction.

**Language**: Formal regulatory/legal text. Precise defined terms where small differences carry legal weight ("retail client" vs "professional client" vs "eligible counterparty").

**Cross-references**: Documents reference each other extensively ("see COBS 1.1.2R", "as defined in CASS 5.5.14R"). Questions may require pulling from multiple sourcebooks.

**Abbreviations/terms**: MiFID, TCFD, SRD, AIFMD, UCITS, PRIIPs, KID, ESIS — domain-specific acronyms that BM25 keyword search must handle correctly.

### Implications for RAG Design

1. **Citations are built-in** — Every paragraph has a unique rule ID (e.g., `COBS 2.1.1R`). We preserve these as metadata and return them as citations. No need to invent citation logic.

2. **Hybrid search is essential** — Regulatory terms are keyword-specific. Pure semantic search would blur legally distinct categories. BM25 must catch exact terms like "MiFID", "eligible counterparty", specific rule references.

3. **Structure-aware chunking** — Natural chunk boundaries are sections/rules, not arbitrary 512-token windows. We should parse the document structure and chunk along rule boundaries. (See Chunking Strategy section.)

4. **Contextual Retrieval is high-value** — Regulatory text is meaningless without context. "The firm must comply with this requirement" only makes sense when you know which sourcebook, chapter, and rule it belongs to.

5. **Cross-sourcebook queries need Agentic RAG** — "What are the rules about inducements across all sourcebooks?" requires searching COBS 2.3, MCOB 2.3, ICOBS 2.3, FPCOB 2.2 across four documents. An agent decomposes this naturally.

6. **Scale considerations** — ~3,022 pages will produce roughly 5,000-15,000 chunks (depending on chunking strategy). Well within Weaviate free tier limits (150K-300K objects).

---

## Finalized Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Language** | Python 3.11+ | ML ecosystem (sentence-transformers, RAGAS, spaCy, flashrank) is Python-native. TypeScript would require API-only models or Python sidecars for embeddings/reranking |
| **API Framework** | FastAPI | Async, auto-generated Swagger docs, streaming support, industry standard for ML services |
| **Vector Database** | Weaviate Cloud (free tier) | Native hybrid search (built-in BM25 + vector fusion via `alpha` param). No need to compute sparse vectors manually. 150K-300K objects, 20GB disk — far exceeds our needs |
| **Graph Database** | Neo4j Aura (free tier) | Graph RAG — rule cross-references form a natural graph. Nodes = rules (already parsed), edges = cross-references (regex-extractable). 200K nodes, 400K rels — we'll use ~5-10% |
| **Embedding Model** | BGE-M3 via sentence-transformers (local) | Dense embeddings, 568M params, runs on CPU, multilingual, strong MTEB scores. We use dense only since Weaviate handles BM25 natively |
| **Reranker** | FlashRank or BGE-Reranker-v2-m3 (local) | Free, local, fast (15-30ms on CPU). No API costs. Cross-encoder for two-stage retrieval |
| **LLM (Generation + Agent)** | OpenRouter free tier — `gpt-oss-120b` (primary), `gemma-4-31b` (fallback) | gpt-oss-120b: OpenAI 120B, 131K context, reliable function/tool calling. Single model for both generation and agent. Fallback: Gemma 4 31B (262K context, native tool calling, faster) |
| **Evaluation** | RAGAS | Industry standard. Computes faithfulness, context precision, context recall. RAG Triad framework |
| **Observability** | Langfuse (free cloud) | Open-source, MIT license, hierarchical traces, `@observe()` decorator. Tracks latency, token counts, costs per step |
| **Caching** | In-memory Python (dict/lru_cache) | Sufficient for assignment scale. L1 exact match, L2 semantic similarity |
| **Testing** | pytest | Unit + integration + eval harness |

---

## Infrastructure

**No Docker required.** All services use cloud free tiers:

| Service | Tier | Limits | Sufficient? |
|---------|------|--------|-------------|
| Weaviate Cloud | Free sandbox | 150K-300K objects, 20GB, 14-day inactivity expiry | Yes — we'll use <2K objects |
| Neo4j Aura | Free | 200K nodes, 400K rels | Yes — ~5K-15K rules as nodes, ~5K-20K cross-refs as edges |
| OpenRouter | Free | ~20 req/min, ~200 req/day per model | Yes — eval + demo usage |
| Langfuse | Free cloud | Generous free tier | Yes |

**Setup for evaluator**: Clone repo → `pip install -r requirements.txt` → set env vars → run

---

## Retrieval Approaches

Each approach operates at a different level, giving a genuinely orthogonal comparison. Detailed in [RETRIEVAL_APPROACHES.md](RETRIEVAL_APPROACHES.md).

```
Search Time              Structure Time           Query Time
    │                        │                        │
    ▼                        ▼                        ▼
Hybrid +                 Graph                    Agentic
Reranking                RAG                      RAG
(BM25 + vector           (traverse cross-ref      (decompose query,
+ cross-encoder)         graph between rules)     orchestrate search+graph)
```

### Approach 1: Hybrid Search + Cross-Encoder Reranking
- **Paradigm**: "Retrieve better" — search-time optimization
- **How**: Weaviate hybrid search (BM25 + dense vector) → top-50 → cross-encoder reranks to top-5 → LLM
- **Why for FCA docs**: BM25 catches exact regulatory terms (MiFID, TCFD, eligible counterparty)
- **When to ship**: 80% of queries. The production default.

### Approach 2: Graph RAG
- **Paradigm**: "Traverse relationships" — structure-time graph exploitation
- **How**: Build graph from explicit cross-references (nodes = rules, edges = references). On query, find relevant rules via hybrid search, then expand via graph traversal to find connected rules. Combine vector + graph results.
- **Why for FCA docs**: Thousands of explicit cross-references between rules across sourcebooks. "What rules reference COBS 2.1.1R?" is a graph query, not a text search. Cheap to build — no NER needed, all rule IDs and references already parsed from LlamaParse output.
- **When to ship**: Relationship-heavy queries, impact analysis, regulatory dependency mapping.

### Approach 3: Agentic RAG
- **Paradigm**: "Reason about retrieval" — query-time adaptive orchestration
- **How**: Agent with tools (vector search, graph traversal, query reformulation) reasons about the query, decomposes complex questions, retrieves iteratively, self-evaluates. Uses Approaches 1 and 2 as building blocks.
- **Why for FCA docs**: Cross-sourcebook questions ("compare inducement rules across COBS, ICOBS, MCOB") need multi-step retrieval with strategy selection.
- **When to ship**: Complex regulatory research, cross-sourcebook analysis.

### Why these 3 (and not others)

| Rejected | Why |
|----------|-----|
| Contextual Retrieval | Our hierarchical chunking with rich metadata (sourcebook, section, rule ID, rule type) already solves the context problem. LLM-enriching each chunk at index time adds cost for marginal gain. |
| CRAG / Self-RAG | Require fine-tuned models in their true forms. Simplified versions are just "LLM-as-judge retry loops" — not different enough from Agentic RAG. |
| FLARE | Depends on token-level logprobs with known calibration issues. Brittle. |
| RAG Fusion | Too simple standalone (one RRF formula). Better as a component within Hybrid Search. |
| ColBERT | Different embedding model, not a different paradigm. |
| CAG | 3,022 pages won't fit in any context window. Not applicable. |
| PlanRAG | Tested on 2 video game scenarios only. No practical ecosystem. |

---

## Chunking Strategy

**Strategy: Hierarchical Chunking with Parent-Child Retrieval** — detailed in [CHUNKING_STRATEGY.md](CHUNKING_STRATEGY.md)

**PDF Parsing: LlamaParse (cost-effective tier)** — detailed in [PDF_PROCESSING.md](PDF_PROCESSING.md)

Summary:
- **PDF parsing**: LlamaParse cost-effective tier (3 credits/page). Tested against Docling on PDCOB.pdf — LlamaParse wins decisively with inline rule ID + type + text format, correct sub-paragraph nesting, and preserved defined terms (italic). 9,066 of 10,000 free monthly credits for full corpus.
- **Build hierarchy**: Sourcebook → Chapter → Section → Rule (parent) → Sub-paragraph (child)
- **Embed children** (small, precise chunks ~100-300 tokens) for retrieval
- **Retrieve parents** (full rule with all sub-paragraphs ~300-800 tokens) for LLM context
- **Metadata = citations**: each chunk carries `{sourcebook, chapter, section, rule_id, rule_type, page}` — citation is structural, not inferred. All extractable via simple regex from LlamaParse output.
- **Post-processing is lightweight**: strip noise (header repeats, logo text, layout comments) + regex extraction. ~50 lines of code vs hundreds for Docling.
- **Same chunks shared** across all 3 retrieval approaches for fair comparison. Graph RAG adds a Neo4j graph layer on top of the same chunks.
- Estimated ~8,000-15,000 child chunks total, well within Weaviate free tier

---

## Evaluation Methodology

### Golden Dataset
- 20-25 Q&A pairs crafted from the FCA Handbook documents
- Each pair includes: question, expected answer, source rule IDs (e.g., COBS 2.1.1R)
- Question types designed to test different retrieval strengths:
  - **Simple factual** (single sourcebook): "What is the cancellation period for banking products under BCOBS?"
  - **Keyword-specific**: "What are the MiFID provisions for inducements?" (tests BM25)
  - **Cross-sourcebook**: "Compare inducement rules across COBS, ICOBS, and MCOB" (tests Agentic RAG)
  - **Context-dependent**: "What must a firm comply with under section 2.1?" (tests Contextual Retrieval)
  - **Multi-hop**: "If a firm is classified as CASS large, what reconciliation frequency is required?"

### RAG Triad Metrics (via RAGAS)
1. **Context Precision** — Are retrieved chunks relevant? (retrieval quality)
2. **Context Recall** — Did we find all relevant chunks? (retrieval completeness)
3. **Faithfulness** — Is the answer grounded in context? (hallucination detection)
4. **Answer Relevancy** — Does the answer address the question? (generation quality)

### Comparison Methodology
- Run all 3 approaches on the same golden dataset
- Compute RAGAS metrics for each
- Measure latency per approach (retrieval time, generation time, total)
- Generate comparison table with metrics + latency
- Qualitative analysis: where does each approach succeed/fail?

### Production Thresholds
- Faithfulness > 0.8 (hallucination guard)
- Context precision > 0.7
- Empty retrieval rate < 10%
- Total latency < 5s

---

## Observability

### Langfuse Tracing
- Trace every step: query → embedding → retrieval → reranking → generation
- Log: retrieved chunks, reranker scores, LLM response, latency per step, token counts

### Alerting Thresholds (design, not necessarily implemented)
- Hallucination score > 0.2
- Empty retrieval rate > 10% (5-min window)
- Generation latency > 5s

---

## Testing Strategy

| Layer | What | Tools |
|-------|------|-------|
| **Unit** | Chunking logic, config parsing, prompt formatting, citation extraction | pytest |
| **Integration** | Query X returns document Y in top-k, end-to-end pipeline smoke test | pytest + Weaviate/Neo4j |
| **Eval harness** | RAGAS metrics on golden dataset, approach comparison | RAGAS + custom runner |
| **Regression** | Changing chunking/embedding doesn't degrade metrics | pytest + RAGAS baselines |

---

## Project Structure

```
./
├── README.md                          # Architecture diagram, trade-offs, setup
├── pyproject.toml                     # Dependencies
├── .env.example                       # Required API keys template
│
├── src/
│   ├── config.py                      # All configurable params (models, top-k, alpha, etc.)
│   ├── dependencies.py                # Shared clients (Weaviate, Neo4j, OpenRouter, Langfuse)
│   │
│   ├── internal/                      # ── Business logic layer ──
│   │   ├── ingestion/
│   │   │   ├── parser.py              # LlamaParse markdown → clean text + metadata extraction
│   │   │   ├── chunker.py             # Hierarchical chunking (parent/child splitting)
│   │   │   ├── embedder.py            # BGE-M3 dense embeddings + Weaviate storage
│   │   │   └── graph_builder.py       # Cross-reference extraction → Neo4j graph
│   │   │
│   │   ├── retrieval/
│   │   │   ├── base.py                # Abstract BaseRetriever interface
│   │   │   ├── hybrid_rerank.py       # Approach 1: Weaviate hybrid + FlashRank reranking
│   │   │   ├── graph_rag.py           # Approach 2: Vector seed → Neo4j expansion → rerank
│   │   │   └── agentic.py             # Approach 3: Tool-calling agent (uses 1 + 2)
│   │   │
│   │   ├── generation/
│   │   │   ├── llm.py                 # OpenRouter client (gpt-oss-120b / gemma-4-31b)
│   │   │   └── prompts.py             # System prompts, citation instructions
│   │   │
│   │   ├── evaluation/
│   │   │   ├── golden_dataset.py      # Load/manage golden Q&A pairs
│   │   │   ├── eval_harness.py        # RAGAS evaluation runner
│   │   │   └── compare.py             # Side-by-side approach comparison
│   │   │
│   │   └── observability/
│   │       └── tracing.py             # Langfuse integration
│   │
│   └── delivery/                      # ── External interface layer ──
│       └── api/
│           ├── app.py                 # FastAPI application setup
│           ├── routes.py              # /query, /ingest, /compare endpoints
│           └── schemas.py             # Pydantic request/response models
│
├── data/
│   ├── raw/                           # Original PDFs (gitignored, symlink to Documents/)
│   ├── parsed/                        # LlamaParse markdown output
│   └── golden/                        # Golden dataset JSON files
│
├── tests/
│   ├── unit/                          # Chunking, parsing, citation extraction
│   ├── integration/                   # Query → retrieval → expected chunks
│   └── eval/                          # RAGAS evaluation suite
│
├── results/                           # Comparison tables, metrics, traces
│
└── docs/                              # Planning docs (PLAN.md, etc.)
```

**Layer separation:**
- `src/internal/` — all business logic. Retrieval approaches, ingestion, generation, evaluation. No HTTP concerns.
- `src/delivery/` — external interfaces only. FastAPI routes, request/response schemas. Thin layer that calls into internal.
- `src/config.py` — all tunable parameters (model names, top-k, alpha, chunk sizes).
- `src/dependencies.py` — shared client instances (Weaviate, Neo4j, LLM, Langfuse). Initialized once, injected where needed.

---

## Key Trade-off Decisions

| Decision | Chose | Over | Why |
|----------|-------|------|-----|
| Python | Python | TypeScript | ML library ecosystem (sentence-transformers, RAGAS, spaCy, flashrank) is Python-native |
| Weaviate | Weaviate Cloud | Qdrant, Pinecone | Native hybrid search (built-in BM25). Qdrant requires manual sparse vector computation. Pinecone also needs external sparse vectors |
| BGE-M3 | BGE-M3 | nomic-embed, all-MiniLM | Strong MTEB scores, 568M params on CPU, multilingual. Dense-only since Weaviate handles BM25 |
| Custom pipeline | No framework | LlamaIndex, LangChain | Shows understanding of each component. Evaluators can read the actual logic, not framework abstractions. Agent loop built with raw LLM tool calling (~50-80 lines). |
| LlamaParse | LlamaParse (3cr/page) | Docling, PyMuPDF | Tested both on PDCOB.pdf. LlamaParse produces inline rule ID + type + text; Docling separates them (broken mapping). LlamaParse also preserves italic defined terms and correct sub-paragraph nesting. 9,066 of 10,000 free credits for full corpus. |
| Cloud free tiers | Weaviate/OpenRouter/LlamaParse cloud | Docker self-hosted | Zero infrastructure setup for evaluator. No Docker = lower friction |
| FlashRank | FlashRank | Cohere Rerank API | Free, local, fast. No API dependency for reranking |
| RAGAS | RAGAS | DeepEval, TruLens | Industry standard, well-documented, computes the exact metrics the RAG Triad needs |
| Langfuse | Langfuse | Galileo, LangSmith | Free, open-source, MIT license. Appropriate for a take-home vs paid platforms |
| Graph RAG | Graph RAG | Contextual Retrieval | FCA documents have thousands of explicit cross-references. Graph construction is cheap (no NER — rule IDs and references already parsed). Contextual Retrieval dropped because hierarchical chunking with rich metadata already solves the context problem. |
| Neo4j | Neo4j Aura (free) | NetworkX (in-memory) | Persistent, Cypher queries expressive for traversal, visual explorer for evaluator, production signal |
| No frameworks | Raw LLM tool calling | LangGraph, LangChain | ~50-80 lines for agent loop. More transparent for evaluators. Fallback to LangGraph if tool calling flaky on free LLM. |
| 3 approach paradigms | Search/Structure/Query time | Incremental variations | Each approach operates at a different level — genuinely different paradigms, not just tweaks. Approach 3 composes 1+2. |

---

## Completed Planning

- [x] Document set analysis — UK FCA Handbook, 10 PDFs, ~3,022 pages
- [x] Tech stack finalized — Python, FastAPI, Weaviate, Neo4j, BGE-M3, FlashRank, OpenRouter
- [x] 3 retrieval approaches — Hybrid+Rerank, Graph RAG, Agentic RAG (RETRIEVAL_APPROACHES.md)
- [x] Chunking strategy — Hierarchical parent-child (CHUNKING_STRATEGY.md)
- [x] PDF parsing — LlamaParse cost-effective tier, tested vs Docling (PDF_PROCESSING.md)
- [x] LLM choice — gpt-oss-120b (primary), gemma-4-31b (fallback)

---

## Implementation TODO

### Phase 1: Ingestion Pipeline
- [ ] Run LlamaParse on all 10 PDFs (9,066 of 10,000 free credits)
- [ ] Build markdown post-processor (noise stripping, metadata extraction regex)
- [ ] Build hierarchical chunker (rule → parent/child splitting, metadata attachment)
- [ ] Set up Weaviate Cloud collection + schema
- [ ] Embed chunks with BGE-M3 and store in Weaviate
- [ ] Set up Neo4j Aura + build cross-reference graph from parsed rules
- [ ] Verify: sample queries return expected chunks

### Phase 2: Retrieval Approaches
- [ ] Implement BaseRetriever interface
- [ ] Implement Approach 1: Hybrid Search + Reranking
- [ ] Implement Approach 2: Graph RAG (vector seed → graph expansion → rerank)
- [ ] Implement Approach 3: Agentic RAG (tool-calling loop with search + graph tools)
- [ ] Verify: each approach returns results for sample queries

### Phase 3: Generation + Citations
- [ ] Build LLM interface (OpenRouter, gpt-oss-120b)
- [ ] Design generation prompts (cite rule IDs, ground in context, say "I don't know")
- [ ] Citation extraction + validation (does cited rule ID exist in our chunks?)
- [ ] Streaming response support

### Phase 4: API Layer
- [ ] FastAPI app with `/ingest`, `/query`, `/compare` endpoints
- [ ] Swagger docs auto-generated
- [ ] `.env.example` with all required API keys

### Phase 5: Evaluation
- [ ] Create golden dataset (20-25 questions across question types from RETRIEVAL_APPROACHES.md)
- [ ] Run RAGAS eval: faithfulness, context precision, context recall per approach
- [ ] Measure latency per approach
- [ ] Generate comparison table + analysis
- [ ] Custom citation accuracy metric

### Phase 6: Testing
- [ ] Unit tests: chunking logic, metadata extraction, citation parsing
- [ ] Integration tests: query → retrieval → expected chunks
- [ ] Eval harness: RAGAS on golden dataset, comparison runner

### Phase 7: Polish
- [ ] Architecture diagram (Mermaid in README)
- [ ] README: setup instructions, design decisions, trade-off analysis
- [ ] Demo script with 5-6 showcase queries
- [ ] Langfuse observability wiring
- [ ] One-command ingestion script (evaluator can rebuild from scratch)
- [ ] Walkthrough prep: where each approach breaks, scale considerations, production concerns
