# Retrieval-Augmented QA over FCA Handbook

Answers questions grounded in 10 UK FCA Handbook sourcebooks (~3,022 pages), with citations to specific rules. Compares four retrieval approaches and recommends one for production.

## Assumptions

- The 10 sourcebooks are the complete document set. References to sourcebooks outside this set (SYSC, SUP, etc.) create stub nodes in Neo4j but aren't resolved.
- Cross-references are explicit and regex-extractable. Implicit references aren't captured.
- The evaluation focuses on retrieval quality. The generation prompt is functional but not optimized — it produces honest, cited answers but hedges on partial context, lowering answer relevancy scores.
- 40 questions across 8 tiers gives directional comparison, not statistical significance. Production evaluation would need a much larger dataset, human annotation, domain expert review, and real user queries — not just synthetic questions generated from known rules.
- No conversational memory or multi-turn support — each query is independent.

## Architecture

### Ingestion Pipeline

```
+----------------+     +----------------+     +------------------+     +------------------+
| 10 FCA PDFs    | --> | LlamaParse     | --> | Parser           | --> | Rule Splitter    |
| (~3,022 pages) |     | (JSON output)  |     | 3 extraction     |     | Fix merge bugs   |
|                |     |                |     | formats: table,  |     | (1,340 rules     |
|                |     |                |     | heading, inline  |     |  freed)          |
+----------------+     +----------------+     +--------+---------+     +--------+---------+
                                                       |                        |
                                                       v                        v
                                              5,753 ParsedRules -------> 5,753 clean rules
                                                                                |
                                              +--------------------------------+
                                              |                                |
                                     +--------v---------+            +---------v--------+
                                     | Chunker (v2)     |            | Graph Builder    |
                                     | Grouped sub-     |            | Neo4j: 6,006     |
                                     | paragraphs       |            | rule nodes +     |
                                     | [header+preamble |            | 4,175 cross-ref  |
                                     |  baked in]       |            | edges            |
                                     +--------+---------+            +------------------+
                                              |
                                     5,720 Chunks
                                              |
                                     +--------v---------+
                                     | Embedder         |
                                     | BGE-M3 (1024-dim)|
                                     | + Weaviate store |
                                     | (BM25 + vector)  |
                                     +------------------+
```

### Query Pipeline

```
                          +------------------+
                          |   User Question  |
                          +--------+---------+
                                   |
                          +--------v---------+
                          |    Adaptive      |
                          |    Router        |
                          +--------+---------+
                                   |
                    +--------------+--------------+
                    |                             |
           +--------v---------+          +--------v---------+
           | Hybrid + Rerank  |          | Agentic RAG      |
           | (fast path)      |          | (deep path)      |
           |                  |          |                  |
           | Weaviate hybrid  |          | LLM plans query  |
           | BM25 + vector    |          | + rule ID lookup |
           | -> FlashRank     |          | + graph expand   |
           | -> top 5         |          | -> FlashRank     |
           +--------+---------+          +--------+---------+
                    |                             |
                    | self-eval: 3/5              |
                    | chunks relevant?            |
                    | YES: done                   |
                    | NO: escalate ------>--------+
                    |                             |
                    +-----------------------------+
                                   |
                          +--------v---------+
                          |  LLM Generation  |
                          |  + Rule Citations|
                          +--------+---------+
                                   |
                          +--------v---------+
                          |  Cited Answer    |
                          +------------------+
```

5,753 rules extracted across three formats (table rows, headings, inline text). The rule splitter catches parser merge bugs where rules absorb subsequent rules — found and fixed 1,340 cases. The v2 chunker groups sub-paragraphs into 500-4000 char chunks instead of splitting at every (1), (2), (3) — this change alone lifted retrieval recall by 14%.

Each chunk is self-contained: `[Sourcebook > Chapter > Section > Rule ID]` header baked in, no parent lookups at retrieval time. 5,720 chunks stored in Weaviate (BM25 + vector) and 6,006 rule nodes with 4,175 cross-reference edges in Neo4j.

### Retrieval

Four approaches, all sharing the same ingestion layer:

**1. Hybrid+Rerank** — Weaviate hybrid search (BM25 + vector, k=50) → FlashRank cross-encoder rerank → top 5. Single pass, ~500ms, deterministic.

**2. Graph RAG** — Same hybrid search for seeds → Neo4j expands 1-2 hops via cross-reference edges → rerank combined pool. Deterministic, ~3-5s.

**3. Agentic RAG** — LLM plans the search strategy (query decomposition, reformulation), executes searches concurrently, expands graph, directly looks up any rule IDs mentioned in the question. Single planning call + deterministic execution, ~7-10s. This is "plan-then-execute" rather than a true agentic loop with self-evaluation — the LLM plans once but doesn't evaluate its own results or iterate. Given the scope of this exercise, a full self-evaluation loop (plan → retrieve → evaluate → re-retrieve) was explored but not fully productionized. It would be the natural next step.

**4. Adaptive** — Runs Hybrid first, then asks an LLM to grade each retrieved chunk as relevant/irrelevant. If fewer than 3 of 5 chunks are relevant, escalates to Agentic. Simple questions get Hybrid speed; complex ones get Agentic quality.

### Generation

All approaches feed their top-5 chunks into the same generation pipeline: system prompt (cite rules, refuse if insufficient context) → LLM → answer with `[COBS 2.1.1R]` citations.

## Evaluation

40 questions across 8 tiers, generated backwards from known rules. Two datasets tested (v1: one-line questions, v2: detailed multi-part questions). RAGAS context recall/precision evaluated by Bedrock Claude Haiku.

### Results

| Metric | Hybrid | Graph | Agentic v2 | Agentic v3 | Adaptive |
|---|---|---|---|---|---|
| Context Recall | 0.834 | 0.795 | **0.911** | 0.856 | 0.863 |
| Context Precision | 0.855 | 0.875 | 0.854 | **0.859** | 0.833 |
| Answer Relevancy | 0.630 | — | 0.555 | **0.754** | 0.666 |
| Tokens/Question | **1,826** | 1,863 | ~2,400 | 2,457 | 3,055 |
| Retrieval Latency | **~500ms** | ~3-5s | ~7-10s | ~7-10s | ~2s or ~10s |

### What each approach is best at

| Question Type | Best Approach | Why |
|---|---|---|
| Simple factual | Agentic (0.85) | Rule ID lookup finds the exact rule |
| Keyword-specific | Graph (0.92) | Graph expansion finds related terminology rules |
| Exception/negation | Hybrid (1.00) | Single search nails "when does X not apply" questions |
| Relationship | Hybrid (0.88) | Cross-reference questions surprisingly well-handled by BM25 |
| Scenario | Agentic (0.93) | Multi-product questions need query decomposition |
| Ambiguous | Agentic (0.96) | Broad questions benefit from reformulation |
| Cross-sourcebook | All (1.00) | Detailed questions with sourcebook names are easy for everyone |

### What I learned during evaluation

These are covered in detail in the Trade-offs section below, but the short version: chunking quality had more impact than retrieval architecture, hybrid search can't find rules by ID (leading to the lookup tool in v3), and the agentic approach needed careful prompt engineering to avoid being worse than the simpler Hybrid baseline.

## Decision

No single approach wins on every metric:

- **Best recall**: Agentic v2 (0.911) — finds the most relevant rules
- **Best answer quality**: Agentic v3 (0.754 relevancy) — rule ID lookup means the LLM gets the exact rules and answers confidently
- **Cheapest**: Hybrid (1,826 tokens/q) — no LLM calls during retrieval
- **Best latency profile**: Adaptive — 57% of queries at ~2s, rest at ~10s

**For production, I'd ship Agentic v3.** For a regulatory compliance tool, accuracy matters more than latency — giving a wrong or incomplete answer about FCA rules has real consequences. Agentic v3 has the best answer relevancy (0.754) and strong recall (0.856), with rule ID lookup ensuring that when a user asks about a specific rule, they get that exact rule in context. The ~7-10s retrieval latency is acceptable for a tool where users expect thorough, cited answers.

The Adaptive approach is a strong alternative if latency becomes a concern — it routes simple questions through the fast Hybrid path while escalating complex ones to Agentic. But for regulatory use, the consistency of always running the thorough path is worth the extra seconds. You'd rather wait 10s for a complete answer than get a fast incomplete one.

Production would also benefit from adaptive top-k (returning 3 chunks for simple queries, 10-15 for complex ones) rather than the fixed k=5 used here. At this collection size (~5,700 chunks), k=5 is reasonable, but a larger document set would need wider retrieval windows.

At scale, the gap between approaches would widen — Hybrid's single-pass search covers a lot of ground on 5,700 chunks but wouldn't on 100K+.

## Trade-offs

**Why not Graph RAG?** Marginal recall gain over Hybrid (0.795 vs 0.834) while adding Neo4j as a dependency. It helps on keyword queries (0.92) but doesn't justify the infrastructure cost as a standalone approach. Better used as a component inside Agentic.

**Why not Hybrid for production?** Fast and cheap, but 0.467 recall on scenario questions. For regulatory compliance, missing relevant rules on multi-product questions is unacceptable. Strong baseline, not sufficient alone.

**Why not Adaptive?** Best latency profile (57% fast path) but highest token cost (3,055/q) and routing isn't perfect — some questions that need Agentic stay on Hybrid. For a compliance tool, consistent thoroughness beats variable speed.

**Why Agentic v3 over v2?** v2 has higher recall (0.911 vs 0.856) but v3 has much better answer relevancy (0.754 vs 0.555). The rule ID lookup in v3 means the LLM gets the exact rules asked about and answers confidently instead of hedging. For regulatory QA, confident correct answers matter more than raw retrieval recall.

**Where the chosen approach breaks down:** Unanswerable questions — the broader search sometimes finds tangentially related rules and the LLM presents them as answers. And at ~7-10s per query, it's too slow for a conversational interface.

**What I'd do differently with more time:**
- Full self-evaluation loop in the agentic retriever (plan → retrieve → evaluate → re-retrieve)
- Faithfulness evaluation (dropped — RAGAS faithfulness makes 8-10 LLM calls per question, too slow with rate limits)
- Generation prompt tuning — the LLM hedges on partial context, lowering answer relevancy
- Merge remaining small chunks (<250 chars, ~5% of total) with neighbouring chunks for richer embeddings
- Test on a larger document set to see where Hybrid genuinely can't compete

**Practical constraints that shaped the evaluation**: Free-tier LLM APIs (OpenRouter, Gemini) imposed rate limits of 16 requests/min, making full 60-question eval runs take hours. Switched to Bedrock Haiku for RAGAS evaluation (20x faster), but token-per-minute limits still caused intermittent failures — some RAGAS scores defaulted to 0.0 on rate-limited questions. To work within these constraints, evaluation was done on 40-question mini datasets (5 per tier). Results are directionally valid but would benefit from a larger, more realistic sample size with dedicated API quotas. Additionally, citation accuracy scores are conservative — the golden dataset expects specific rule IDs (e.g., BCOBS 5.1.1) but the retriever often finds adjacent rules in the same section (e.g., BCOBS 5.1.2G) that address the same topic. RAGAS content-based metrics are fairer for these cases.

## How the iterations went

1. **Chunking**: v1 split every sub-paragraph → 8,459 tiny chunks. Eval showed fragmented context. v2 grouped sub-paragraphs → 5,720 denser chunks. Hybrid recall jumped 14%.

2. **Agentic v1**: LLM-in-the-loop with 5-7 calls per query. Slow (40s), non-deterministic, sometimes worse than Hybrid because the agent filtered searches prematurely. Switched to plan-once-execute (v2): 1 LLM call for planning, deterministic execution. Latency dropped to ~10s.

3. **Agentic v3**: Added rule ID lookup — when the question mentions specific rules, fetch them directly from Weaviate instead of searching. Relationship recall went from 0.75 to 0.90.

4. **Adaptive**: Built after seeing that Hybrid is excellent for simple queries but poor for scenarios. Per-chunk binary relevance grading (following Self-RAG / CRAG patterns) routes ~50% of queries through the fast path.

## Testing

- **Unit tests** (26 tests): Parser regex, cross-reference extraction, chunker splitting/merging/headers
- **Integration tests** (37 checks): Weaviate data validation — count, hybrid search, BM25, filters, metadata, context headers
- **Eval harness**: RAGAS context recall/precision + custom citation accuracy. Supports all 4 approaches, multiple datasets, resumable runs
- **Answer relevancy**: Post-hoc scoring of generated answers via RAGAS AnswerRelevancy

```bash
# Setup
pip install -r requirements.txt
cp .env.example .env  # add your API keys

# Ask a question (pass as argument, or omit for sample queries)
python3 -m src.internal.retrieval.hybrid_rerank "What must a firm do under COBS 2.1?"
python3 -m src.internal.retrieval.graph_rag "What rules reference COBS 2.1.1R?"
python3 -m src.internal.retrieval.agentic_v3 "What protections exist for consumers?"
python3 -m src.internal.retrieval.adaptive "What are the cancellation rights for banking customers?"

# Run evaluation
python3 -m src.internal.evaluation.eval_harness --mini --dataset-v2 --chunks-v2 --adaptive --name=test
python3 -m src.internal.evaluation.eval_harness --mini --dataset-v2 --chunks-v2 --agentic-v3 --name=test

# Answer relevancy (post-hoc on existing results)
python3 -m scripts.eval_answer_relevancy results/<result_file>.json

# Tests
python3 -m pytest tests/ -v

# Re-run ingestion (if needed)
python3 -m src.internal.ingestion.embedder            # Weaviate
python3 -m src.internal.ingestion.graph_builder       # Neo4j
```

## Stack

| Component | Choice |
|---|---|
| PDF parsing | LlamaParse |
| Embeddings | BGE-M3 (1024-dim, local CPU) |
| Vector + keyword search | Weaviate Cloud (hybrid BM25 + vector) |
| Knowledge graph | Neo4j Aura (cross-reference edges) |
| Reranker | FlashRank (local, 3MB) |
| Agent planner | Bedrock Claude Sonnet 4.6 |
| Answer generation | Bedrock Claude Haiku 4.5 |
| Self-eval router | Bedrock Claude Haiku 3 |
| RAGAS evaluation | Bedrock Claude Haiku 3 |

No frameworks (LangChain, LlamaIndex) — limited experience with LangChain and none with the others. Adopting them would have added boilerplate and learning overhead that would have slowed down the iteration cycle on the actual retrieval problem.

### Production considerations

- **Observability**: Currently only tracking token counts naively per request. Production would need proper tracing (e.g., LangFuse) to track retrieval latency, reranker scores, and routing decisions end-to-end. Citation verification to catch retrieval failures.
- **Scaling**: Dedicated Weaviate cluster, GPU embedding, Neo4j read replicas as the document set grows.
- **Testing**: Eval harness as a regression suite on every deployment. Shadow-test new approaches alongside the production path.
