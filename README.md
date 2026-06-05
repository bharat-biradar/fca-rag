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
                                     5,720 Chunks                     Neo4j Aura Graph
                                              |
                                     +--------v---------+
                                     | Embedder         |
                                     | BGE-M3 (1024-dim)|
                                     | + Weaviate store |
                                     | (BM25 + vector)  |
                                     +------------------+
                                              |
                                     5,720 objects in
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

Total: **5,753 rules** -> **5,720 chunks** (v2 chunker with grouped sub-paragraphs)

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

## Evaluation Results

Evaluated on 40 questions across 8 tiers (simple factual, keyword-specific, cross-sourcebook, ambiguous, scenario, exception/negation, relationship, unanswerable). RAGAS context recall and precision measured by Bedrock Claude Haiku 3.

### Overall

| Metric | Hybrid+Rerank | Graph RAG | Agentic v2 |
|---|---|---|---|
| Context Recall | 0.834 | 0.795 | **0.911** |
| Context Precision | 0.855 | **0.875** | 0.854 |
| Citation Accuracy | 0.378 | 0.406 | **0.431** |

### Per Question Type (Context Recall)

| Type | Hybrid | Graph | Agentic | Winner |
|---|---|---|---|---|
| Simple factual | 0.700 | 0.700 | **0.850** | Agentic |
| Keyword-specific | 0.860 | **0.920** | 0.860 | Graph |
| Cross-sourcebook | **1.000** | **1.000** | **1.000** | Tie |
| Ambiguous | 0.833 | 0.667 | **0.960** | Agentic |
| Scenario | 0.467 | 0.600 | **0.933** | Agentic |
| Exception/negation | **1.000** | 0.850 | 0.967 | Hybrid |
| Relationship | **0.883** | 0.700 | 0.753 | Hybrid |
| Unanswerable | 0.927 | 0.927 | **0.967** | Agentic |

### Where Each Approach Wins

**Hybrid+Rerank** dominates exception/negation (1.000) and relationship queries (0.883). A single unfiltered search captures the right rules when the question mentions specific rule IDs or regulatory concepts. No LLM calls during retrieval — fast, deterministic, cheapest to operate.

**Graph RAG** leads on keyword-specific queries (0.920). The Neo4j graph expansion discovers related rules that share terminology but aren't direct text matches. Adds ~3-5s for graph traversal. Same deterministic behavior as Hybrid.

**Agentic v2** dominates scenario (0.933 vs 0.467), ambiguous (0.960 vs 0.833), and simple factual (0.850 vs 0.700). Query decomposition by Sonnet 4.6 breaks complex multi-product questions into targeted sub-queries. Concurrent execution keeps latency to ~7-10s for retrieval.

## Trade-off Decision

**Ship Hybrid+Rerank for production.** It scores 0.834 recall at sub-second latency with zero retrieval LLM cost. It's deterministic, easy to monitor, and handles exception/negation and relationship queries best.

**Where it breaks down:** Scenario questions (0.467 recall). Multi-product, multi-sourcebook situations require query decomposition that a single search can't provide. A banking customer with an ISA and insurance policy triggers three regulatory frameworks — Hybrid finds one, Agentic finds all three.

**Recommendation for deployment:**
- Hybrid+Rerank as the default path (~500ms, handles 70% of queries well)
- Agentic v2 as a "deep search" mode for complex questions (~10s, user-initiated)
- Graph RAG integrated as a component within both (graph expansion improves keyword recall)

**Why not Agentic for everything?** Three reasons:
1. Non-deterministic — same query can produce different results
2. Depends on an external LLM for planning — adds a failure mode and cost per query
3. On this collection size (5,720 chunks), a single hybrid search already covers significant ground

**What would change this decision:** A larger document set (100K+ chunks) where single-pass search can't cover enough ground, or a use case dominated by scenario/ambiguous questions where the 2x recall improvement justifies the latency and cost.

### Iteration History

The final results reflect several rounds of systematic improvement:

1. **V1 chunker** (over-granular sub-paragraph splitting) → **V2 chunker** (grouped sub-paragraphs): Hybrid recall +14%, cross-reference recall +68%
2. **V1 agentic** (LLM-in-the-loop, 5-7 LLM calls) → **V2 agentic** (plan-once-execute, 1 LLM call): latency 40s → 10s, more reliable
3. **Sourcebook filtering removed** from agent's default search: fixed persistent failures on BCOBS and cross-sourcebook queries
4. **Reranker comparison**: MiniLM-L-12 (stricter) helped Hybrid but hurt Agentic — kept TinyBERT for consistency across approaches

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

**Document set:**
- The 10 sourcebooks are the complete document set. Cross-references to sourcebooks outside this set (SYSC, SUP, PRIN, etc.) create stub nodes in Neo4j but aren't resolved.
- Rule IDs follow the pattern `SOURCEBOOK X.X.X[TYPE]` consistently. The parser handles three extraction formats (table rows, heading rules, inline text) covering ~98% of rules.
- Cross-references in rule text are explicit and regex-extractable. Implicit references ("the relevant conduct rules") are not captured.

**Infrastructure:**
- Free-tier Weaviate Cloud (150K objects at 1024-dim) is sufficient for 5,720 chunks. Production would need a dedicated cluster.
- Embedding on CPU is acceptable for batch ingestion (~27 min for 5,720 chunks). Production would use GPU or a managed embedding service.
- The evaluation uses multiple LLM providers (Gemini Flash for generation, Bedrock Haiku for RAGAS evaluation, Sonnet 4.6 for agent planning). A production system would standardize on one provider.

**Evaluation:**
- Golden dataset expected_rule_ids are sometimes narrower than the set of valid answers. Adjacent rules in the same section (e.g., BCOBS 5.1.2G vs expected 5.1.1R) address the same topic. RAGAS content-based metrics are fairer than strict ID matching for these cases.
- Answer relevancy scores are penalized by honest partial answers. The system prompt instructs the LLM to refuse when context is insufficient, which produces lower relevancy scores compared to systems that hallucinate confidently.
- 40 questions across 8 tiers is sufficient for directional comparison but not statistically significant. A production evaluation would use 200+ questions.
- The evaluation focuses strictly on retrieval quality and answer generation. It does not account for conversational memory, multi-turn follow-ups, or user session context — each query is evaluated independently.

### Production Considerations

- **Observability**: Add per-request tracing (e.g., LangFuse) to track token usage, retrieval latency, and reranker scores. Monitor for retrieval failures via citation verification.
- **Scaling**: Move to a dedicated Weaviate cluster, GPU-based embedding, and read replicas for Neo4j as the document set grows.
- **Testing**: Run the eval harness on every deployment as a regression suite. Shadow-test new retrieval approaches alongside the production path before switching.
