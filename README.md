# Retrieval-Augmented QA over FCA Handbook

A RAG service that answers questions grounded in 10 UK FCA Handbook sourcebooks (~3,022 pages), with citations to specific rules. Compares three retrieval approaches to determine which to ship for production.

## Architecture

### Ingestion Pipeline

```
+----------------+     +----------------+     +------------------+     +------------------+
| 10 FCA PDFs    | --> | LlamaParse     | --> | Parser           | --> | Rule Splitter    |
| (~3,022 pages) |     | (JSON output)  |     | 3 extraction     |     | Fix merge bugs   |
|                |     |                |     | formats: table,  |     | Bold + line-start|
|                |     |                |     | heading, inline  |     | ID splitting     |
+----------------+     +----------------+     +--------+---------+     +--------+---------+
                                                       |                        |
                                                       v                        v
                                              5,753 ParsedRules -------> 5,753 clean rules
                                                                                |
                                              +--------------------------------+
                                              |                                |
                                     +--------v---------+            +---------v--------+
                                     | Chunker          |            | Graph Builder    |
                                     | Context-enriched |            | Neo4j: Rule nodes|
                                     | flat chunks      |            | + REFERENCES     |
                                     | [header+preamble |            |   edges from     |
                                     |  baked in]       |            |   cross-refs     |
                                     +--------+---------+            +---------+--------+
                                              |                                |
                                     8,459 Chunks                     Neo4j Aura Graph
                                              |
                                     +--------v---------+
                                     | Embedder         |
                                     | BGE-M3 (1024-dim)|
                                     | + Weaviate store |
                                     | (BM25 + vector)  |
                                     +------------------+
                                              |
                                     8,398 objects in
                                     Weaviate Cloud
```

### Query Pipeline

```
                          +------------------+
                          |   User Question  |
                          +--------+---------+
                                   |
                          +--------v---------+
                          |  BGE-M3 Embed    |
                          |  (1024-dim)      |
                          +--------+---------+
                                   |
            +----------------------+----------------------+
            |                      |                      |
   +--------v---------+  +--------v---------+  +---------v--------+
   | Approach 1       |  | Approach 2       |  | Approach 3       |
   | Hybrid + Rerank  |  | Graph RAG        |  | Agentic RAG      |
   |                  |  |                  |  |                  |
   | Weaviate hybrid  |  | Weaviate seeds   |  | LLM agent with   |
   | (BM25 + vector)  |  | + Neo4j graph    |  | search + graph   |
   | -> FlashRank     |  |   expansion      |  | tools + query    |
   |    rerank        |  | -> rerank        |  | reformulation    |
   +--------+---------+  +--------+---------+  +---------+--------+
            |                      |                      |
            +----------------------+----------------------+
                                   |
                          +--------v---------+
                          |  Top-K Chunks    |
                          |  (with context   |
                          |   headers)       |
                          +--------+---------+
                                   |
                          +--------v---------+
                          |  LLM Generation  |
                          |  (OpenRouter)    |
                          |  + Rule Citations|
                          +--------+---------+
                                   |
                          +--------v---------+
                          |  Cited Answer    |
                          +------------------+
```

## Document Set

10 FCA Handbook sourcebooks:

| Sourcebook | Full Name | Rules |
|---|---|---|
| BCOBS | Banking: Conduct of Business | 213 |
| CASS | Client Assets | 1,027 |
| CMCOB | Claims Management: Conduct of Business | 154 |
| COBS | Conduct of Business | 1,783 |
| ESG | Environmental, Social and Governance | 162 |
| FPCOB | Funeral Plan: Conduct of Business | 288 |
| ICOBS | Insurance: Conduct of Business | 328 |
| MAR | Market Conduct | 364 |
| MCOB | Mortgages and Home Finance | 1,218 |
| PDCOB | Pensions Dashboards: Conduct of Business | 216 |

Total: **5,753 rules** -> **8,459 chunks** after splitting

## Retrieval Approaches

### Approach 1: Hybrid Search + Cross-Encoder Rerank
- Weaviate hybrid search (BM25 + dense vector, alpha=0.5) retrieves 50 candidates
- FlashRank cross-encoder reranks to top 5
- Fast (~500ms retrieval), deterministic, easy to debug
- Struggles with vague/ambiguous queries

### Approach 2: Graph RAG
- Seeds from hybrid search, expands via Neo4j cross-reference graph (1-2 hops)
- Discovers related rules that text search misses
- Strong for relationship queries ("what rules reference COBS 2.1.1R?")

### Approach 3: Agentic RAG
- LLM agent uses Approaches 1+2 as tools
- Query decomposition, reformulation, iterative refinement
- Best for complex/ambiguous queries, highest latency

## Trade-off Decision

<!-- TODO: Fill in after running evals on all 3 approaches -->

## Project Structure

```
src/
  config.py                     # Central configuration
  dependencies.py               # Lazy singletons (Weaviate, Neo4j, BGE-M3)
  internal/
    ingestion/
      parser.py                 # LlamaParse JSON -> ParsedRule objects
      rule_splitter.py          # Post-processing: split merged rules
      chunker.py                # Context-enriched flat chunking
      embedder.py               # BGE-M3 embed + Weaviate storage
      graph_builder.py          # Neo4j cross-reference graph
    retrieval/
      base.py                   # RetrievedChunk, BaseRetriever
      hybrid_rerank.py          # Approach 1: hybrid + FlashRank
      graph_rag.py              # Approach 2: graph expansion
      agentic.py                # Approach 3: LLM agent
    generation/
      llm.py                    # OpenRouter LLM client
      prompts.py                # System/user prompts, citation extraction
    evaluation/
      golden_dataset.py         # 60-question golden QA set
      eval_harness.py           # RAGAS metrics + custom metrics
      compare.py                # Side-by-side comparison
tests/
  integration/
    test_weaviate.py            # 37 checks on Weaviate data
    test_hybrid_rerank.py       # Retriever integration tests
data/
  golden/golden_qa.json         # Golden QA dataset (6 tiers x 10)
results/
  eval_hybrid_rerank.json       # Full eval results per approach
```

## Stack

| Component | Choice | Why |
|---|---|---|
| PDF parsing | LlamaParse | Best table extraction for regulatory docs |
| Embeddings | BGE-M3 (1024-dim) | Strong multilingual, long-context support |
| Vector + keyword | Weaviate Cloud | Native hybrid BM25 + vector in one query |
| Graph | Neo4j Aura | Free tier, Cypher for cross-reference traversal |
| Reranker | FlashRank | Lightweight cross-encoder, runs locally, no API |
| LLM | OpenRouter (gpt-oss-120b) | Free, OpenAI-compatible API |
| Evaluation | RAGAS + custom | Context recall/precision + citation accuracy |

No frameworks (LangChain, LlamaIndex). All retrieval logic is direct library calls — every line is readable and debuggable.

## Setup

```bash
# Install dependencies
pip install sentence-transformers weaviate-client flashrank openai ragas

# Environment variables (.env)
WEAVIATE_URL=...
WEAVIATE_API_KEY=...
NEO4J_URI=...
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
OPENROUTER_API_KEY=...
GEMINI_API_KEY=...          # optional, for RAGAS evaluation

# Run ingestion (parser -> chunker -> embedder)
python3 -m src.internal.ingestion.embedder

# Run retrieval test
python3 -m src.internal.retrieval.hybrid_rerank

# Run evaluation
python3 -m src.internal.evaluation.eval_harness --mini    # 18 questions
python3 -m src.internal.evaluation.eval_harness           # full 60 questions

# Validate Weaviate data
python3 -m tests.integration.test_weaviate
```

## Evaluation Methodology

**Golden dataset**: 60 questions across 6 tiers (simple factual, keyword-specific, cross-reference, cross-sourcebook, ambiguous, unanswerable) — generated backwards from known rules to guarantee correct expected answers.

**Metrics**:
- **Context Recall** (RAGAS): does the retrieved context contain enough information to answer?
- **Context Precision** (RAGAS): are the retrieved chunks relevant?
- **Citation Accuracy** (custom): do the LLM's cited rule IDs match expected rules?
- **Token Usage**: prompt + completion tokens per question

Same golden dataset, same generation pipeline, same metrics — only the retrieval step changes between approaches.

## Key Design Decisions

1. **Context-Enriched Flat Chunking**: Every chunk is self-contained with `[Sourcebook > Chapter > Section > Rule ID]` header baked in. No parent lookups needed at retrieval time.

2. **Vectorizer=none in Weaviate**: We provide our own BGE-M3 vectors. This means hybrid queries must pass both text (for BM25) and vector (for dense search).

3. **Rule Splitter as post-processing**: Parser merge bugs (rules absorbing subsequent rules) are fixed by a separate rule_splitter rather than complicating the parser. Found and split 1,340 merged rules.

4. **Deterministic chunk IDs**: `generate_uuid5(chunk_id)` enables idempotent re-ingestion. Pipeline can be re-run safely.

## Assumptions

- The 10 sourcebooks are the complete document set (no external references resolved beyond stub nodes)
- Rule IDs follow the pattern `SOURCEBOOK X.X.X[TYPE]` consistently
- Cross-references in rule text are explicit (regex-extractable)
- Free-tier infrastructure (Weaviate Cloud, Neo4j Aura, OpenRouter) is sufficient for evaluation
