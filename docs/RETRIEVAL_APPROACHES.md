# Retrieval Approaches: Detailed Design

## Overview

Three approaches, each operating at a different level:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Shared Infrastructure                       │
│  LlamaParse → Hierarchical Chunking → BGE-M3 → Weaviate        │
│                                          ↓                      │
│                              Neo4j Graph (cross-references)     │
└─────────────────────────────────────────────────────────────────┘
        │                        │                      │
        ▼                        ▼                      ▼
   Approach 1               Approach 2              Approach 3
   Hybrid + Rerank          Graph RAG               Agentic RAG
   (search-time)            (structure-time)        (query-time)
                                                    uses 1 + 2
                                                    as tools
```

All three share the same base: Weaviate vector store with hierarchical chunks + Neo4j graph.
The only difference is **how they retrieve** — the variable under test.

---

## Shared Components

### Weaviate Collection

Single collection storing all child chunks with:

```python
{
    # Searchable text (for BM25 + vector)
    "text": "A firm must act honestly, fairly and professionally...",
    
    # Dense vector embedding (BGE-M3)
    "vector": [0.023, -0.118, ...],  # 1024 dims
    
    # Metadata (for filtering + citation)
    "sourcebook": "COBS",
    "chapter": "2",
    "section": "2.1",
    "section_title": "Acting honestly, fairly and professionally",
    "rule_id": "COBS 2.1.1R",
    "rule_type": "R",
    "parent_text": "full text of parent rule with all sub-paragraphs...",
}
```

### Neo4j Graph

Nodes = rules, edges = cross-references. Built from LlamaParse output:

```
Nodes (~5,000-15,000):
  (:Rule {id: "COBS 2.1.1R", sourcebook: "COBS", type: "R", section: "2.1"})

Edges (~5,000-20,000):
  (:Rule)-[:REFERENCES]->(:Rule)
  Built from regex: every rule ID mentioned in another rule's text = an edge
  
Additional edges:
  (:Rule)-[:BELONGS_TO]->(:Section)
  (:Section)-[:BELONGS_TO]->(:Chapter)
  (:Chapter)-[:BELONGS_TO]->(:Sourcebook)
```

Graph construction is ~30 lines — no NER, no ML, just regex on already-parsed text:

```python
import re
from neo4j import GraphDatabase

RULE_REF_PATTERN = r'(BCOBS|CASS|CMCOB|COBS|ESG|FPCOB|ICOBS|MAR|MCOB|PDCOB)\s+[\d.]+\w*'

for rule in all_rules:
    driver.execute_query("CREATE (:Rule {id: $id, ...})", id=rule.id, ...)
    
    references = re.findall(RULE_REF_PATTERN, rule.text)
    for ref in references:
        driver.execute_query(
            "MATCH (a:Rule {id: $from}), (b:Rule {id: $to}) "
            "MERGE (a)-[:REFERENCES]->(b)",
            from_=rule.id, to=ref
        )
```

### Generation (shared across all approaches)

All three approaches produce a set of retrieved chunks → same generation pipeline:

```python
def generate_answer(query: str, chunks: list[Chunk]) -> Answer:
    # 1. Pull parent text for each chunk (richer context)
    contexts = [chunk.parent_text for chunk in chunks]
    
    # 2. Build prompt with citation instructions
    prompt = build_prompt(query, contexts)  # includes "cite rule IDs"
    
    # 3. Call LLM (OpenRouter)
    response = llm.chat(prompt)
    
    # 4. Extract citations from response
    citations = extract_rule_ids(response)
    
    return Answer(text=response, citations=citations, chunks=chunks)
```

---

## Approach 1: Hybrid Search + Cross-Encoder Reranking

### Paradigm
"Retrieve better" — optimize the search step itself.

### Pipeline

```
Query
  │
  ▼
Weaviate Hybrid Search (BM25 + dense vector, alpha=0.5)
  │  retrieve top-50 candidates
  ▼
Cross-Encoder Reranking (FlashRank / BGE-Reranker)
  │  rerank to top-5
  ▼
Pull parent text for each top-5 chunk
  │
  ▼
LLM Generation with citations
```

### Implementation

```python
class HybridRerankRetriever(BaseRetriever):
    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        # Step 1: Weaviate hybrid search (BM25 + vector)
        candidates = self.weaviate.query.hybrid(
            query=query,
            alpha=0.5,       # balance BM25 and vector
            limit=50,        # broad initial retrieval
            return_metadata=["score"]
        )
        
        # Step 2: Cross-encoder reranking
        reranked = self.reranker.rerank(
            query=query,
            documents=[c.text for c in candidates],
            top_k=top_k
        )
        
        # Step 3: Pull parent text for context enrichment
        for chunk in reranked:
            chunk.context = chunk.parent_text
        
        return reranked
```

### Why it works for FCA docs

- **BM25 catches exact regulatory terms**: "MiFID", "eligible counterparty", "COBS 2.1.1R" — keyword matching is essential for legal precision
- **Dense vectors catch semantic similarity**: "What are a firm's obligations to act in client interests?" matches rules about "best interests" even without exact keyword overlap
- **Cross-encoder reranking**: eliminates noise from the broad top-50 retrieval, keeping only the most relevant rules
- **Alpha tuning**: adjustable balance between keyword and semantic — for regulatory text, higher alpha (more BM25) likely works better

### Strengths
- Fast: single retrieval pass + reranking = ~200-500ms total
- Predictable: deterministic results for the same query
- Easy to evaluate: straightforward RAGAS metrics
- Easy to debug: can inspect retrieved chunks and reranker scores

### Weaknesses
- Single-pass: no query reformulation, no iterative refinement
- Cannot traverse relationships: "what rules reference COBS 2.1.1R?" requires text matching the reference string, which may fail if the reference is in a different format
- Cannot decompose complex queries: "compare inducement rules across all sourcebooks" returns a jumble of results from different sourcebooks without structure

### Configuration

```python
HYBRID_ALPHA = 0.5          # BM25 vs vector balance (tune via eval)
INITIAL_RETRIEVAL_K = 50    # candidates for reranking
FINAL_TOP_K = 5             # chunks sent to LLM
RERANKER_MODEL = "flashrank" # or "BAAI/bge-reranker-v2-m3"
```

---

## Approach 2: Graph RAG

### Paradigm
"Traverse relationships" — exploit the explicit cross-reference structure of FCA documents.

### Pipeline

```
Query
  │
  ├──► Weaviate Hybrid Search (same as Approach 1)
  │       retrieve top-10 seed rules
  │
  ▼
For each seed rule:
  Neo4j graph traversal (1-2 hops via REFERENCES edges)
  │  find connected rules
  │
  ▼
Combine: seed rules + graph-expanded rules
  │  deduplicate, score by (vector_score + graph_proximity)
  │
  ▼
Cross-Encoder Reranking on combined set
  │  rerank to top-5
  │
  ▼
Pull parent text + graph context (relationship chain)
  │
  ▼
LLM Generation with citations + relationship context
```

### Implementation

```python
class GraphRetriever(BaseRetriever):
    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        # Step 1: Vector search for seed rules
        seeds = self.weaviate.query.hybrid(
            query=query, alpha=0.5, limit=10
        )
        seed_ids = [s.rule_id for s in seeds]
        
        # Step 2: Graph expansion (1-2 hops)
        expanded_ids = self.neo4j.execute_query("""
            MATCH (seed:Rule)-[:REFERENCES*1..2]-(connected:Rule)
            WHERE seed.id IN $seed_ids
            RETURN DISTINCT connected.id AS id,
                   min(length(path)) AS hops
            ORDER BY hops
            LIMIT 30
        """, seed_ids=seed_ids)
        
        # Step 3: Fetch full chunks for expanded rules from Weaviate
        all_candidates = seeds + self.fetch_chunks_by_ids(expanded_ids)
        
        # Step 4: Deduplicate and score
        # Combined score = vector_similarity + (1 / (1 + hops)) * graph_weight
        scored = self.score_candidates(all_candidates, seeds)
        
        # Step 5: Rerank combined set
        reranked = self.reranker.rerank(
            query=query,
            documents=[c.text for c in scored],
            top_k=top_k
        )
        
        # Step 6: Add graph context (relationship chain)
        for chunk in reranked:
            chunk.graph_context = self.get_relationship_chain(chunk.rule_id, seed_ids)
        
        return reranked
```

### Graph queries for different use cases

```cypher
-- "What rules reference COBS 2.1.1R?"
MATCH (r:Rule)-[:REFERENCES]->(target:Rule {id: 'COBS 2.1.1R'})
RETURN r.id, r.sourcebook, r.text

-- "What's the dependency chain for this rule?"
MATCH path = (start:Rule {id: 'PDCOB 5.3.1R'})-[:REFERENCES*1..3]->(end:Rule)
RETURN path

-- "Find all rules about client money across sourcebooks"
MATCH (r:Rule)-[:REFERENCES*1..2]-(connected:Rule)
WHERE r.text CONTAINS 'client money'
RETURN DISTINCT connected.id, connected.sourcebook

-- "Impact analysis: what breaks if we change COBS 2.1.1R?"
MATCH (r:Rule)-[:REFERENCES]->(target:Rule {id: 'COBS 2.1.1R'})
RETURN r.id, r.sourcebook, r.type
ORDER BY r.sourcebook
```

### Why it works for FCA docs

- **Thousands of explicit cross-references**: PDCOB alone had 413. Across 10 sourcebooks, the graph has thousands of edges — this is a natural graph, not a forced one.
- **No NER or ML needed for graph construction**: Rule IDs are explicit (`COBS 2.1.1R`), references are explicit in text. Regex builds the entire graph.
- **Cross-sourcebook discovery**: A query about client money can start from CASS rules and traverse to connected rules in COBS, MCOB, etc. — discovering relationships that text search alone would miss.
- **Impact analysis**: "What depends on this rule?" is a native graph query, impossible with vector search.

### Strengths
- Discovers related rules that share no text similarity (connected via cross-references)
- Cross-sourcebook traversal finds rules text search misses
- Answers relationship/dependency questions natively
- Graph construction is cheap (~30 lines, no ML)
- Neo4j Cypher is expressive and readable

### Weaknesses
- Only useful when cross-references exist — rules with no references get no graph benefit
- Doesn't help with semantic similarity queries ("what rules are about fairness?")
- Graph quality depends on reference extraction quality (regex may miss non-standard references)
- Adds Neo4j as a dependency (mitigated by free Aura tier)
- 1-2 hop expansion can pull in too many irrelevant connected rules — needs scoring/filtering

### Configuration

```python
SEED_RETRIEVAL_K = 10       # initial vector search seeds
GRAPH_HOPS = 2              # max traversal depth
GRAPH_EXPANSION_LIMIT = 30  # max connected rules per query
GRAPH_WEIGHT = 0.3          # weight of graph proximity in combined score
RERANKER_TOP_K = 5          # final selection after reranking
```

---

## Approach 3: Agentic RAG

### Paradigm
"Reason about retrieval" — an agent that uses Approaches 1 and 2 as tools, choosing the right strategy per query.

### Pipeline

```
Query
  │
  ▼
Agent (LLM with tools) ◄──── self-evaluation loop
  │
  ├── Tool: search_vectors(query, sourcebook_filter?)
  │     → Hybrid search + reranking (Approach 1)
  │
  ├── Tool: search_graph(rule_id, hops?)
  │     → Graph traversal (Approach 2)
  │
  ├── Tool: reformulate_query(original, feedback)
  │     → LLM rewrites query for better retrieval
  │
  └── Tool: answer(context, query)
        → Generate final answer with citations
```

### Implementation

```python
class AgenticRetriever(BaseRetriever):
    def __init__(self, hybrid_retriever, graph_retriever, llm):
        self.tools = [
            {
                "name": "search_vectors",
                "description": "Search for FCA rules by text similarity. Use for factual questions about specific topics.",
                "parameters": {
                    "query": "search query string",
                    "sourcebook": "optional: filter to specific sourcebook (COBS, BCOBS, etc.)",
                    "top_k": "number of results (default 5)"
                }
            },
            {
                "name": "search_graph",
                "description": "Find rules connected to a specific rule via cross-references. Use when you have a specific rule ID and need to find related rules.",
                "parameters": {
                    "rule_id": "e.g. COBS 2.1.1R",
                    "hops": "traversal depth (1 or 2)"
                }
            },
            {
                "name": "reformulate_query",
                "description": "Reformulate the query if initial search results are poor.",
                "parameters": {
                    "original_query": "the original user query",
                    "feedback": "what was wrong with previous results"
                }
            }
        ]
    
    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
        
        all_chunks = []
        max_steps = 5
        
        for step in range(max_steps):
            response = self.llm.chat(messages, tools=self.tools)
            
            if not response.tool_calls:
                break  # agent is done, ready to answer
            
            for tool_call in response.tool_calls:
                result = self.execute_tool(tool_call)
                all_chunks.extend(result.chunks)
                messages.append({"role": "tool", "content": result.summary})
        
        # Deduplicate and return top-k by relevance
        return self.deduplicate_and_rank(all_chunks, top_k)
```

### Agent system prompt

```
You are a regulatory research assistant with access to the UK FCA Handbook.
You have tools to search rules by text and traverse cross-references.

For each user query:
1. Analyze what type of question it is:
   - Simple factual → use search_vectors once
   - Cross-sourcebook comparison → search_vectors for each relevant sourcebook
   - Relationship/dependency → search_vectors to find a seed rule, then search_graph to expand
   - Ambiguous → reformulate_query, then search
   
2. Execute your chosen strategy using the available tools.

3. After each search, evaluate: do I have enough relevant rules to answer?
   If not, search again with different terms or expand via graph.

4. When you have sufficient context, stop calling tools. 
   The system will generate the final answer from your collected results.

Important: 
- Be efficient. Most queries need 1-2 tool calls, not 5.
- Use sourcebook filters when the query mentions a specific domain.
- Use graph search when you find a highly relevant rule and need connected rules.
```

### Example agent traces

**Simple query**: "What is the cancellation period under BCOBS?"
```
Step 1: search_vectors("cancellation period", sourcebook="BCOBS") → 5 chunks
Agent: sufficient results, done.
Total tool calls: 1
```

**Cross-sourcebook query**: "Compare inducement rules across banking, insurance, and mortgages"
```
Step 1: search_vectors("inducement rules", sourcebook="BCOBS") → 3 chunks
Step 2: search_vectors("inducement rules", sourcebook="ICOBS") → 3 chunks  
Step 3: search_vectors("inducement rules", sourcebook="MCOB") → 3 chunks
Agent: have rules from all 3 sourcebooks, done.
Total tool calls: 3
```

**Relationship query**: "What rules are affected if COBS 2.1.1R changes?"
```
Step 1: search_graph(rule_id="COBS 2.1.1R", hops=2) → 12 connected rules
Agent: sufficient for impact analysis, done.
Total tool calls: 1
```

**Ambiguous query**: "Tell me about client obligations"
```
Step 1: search_vectors("client obligations") → 5 chunks (too vague, mixed results)
Step 2: reformulate_query("client obligations", "results too broad, mixing different types of obligations")
  → reformulated: "firm obligations to clients under conduct of business rules"
Step 3: search_vectors("firm obligations to clients conduct of business") → 5 better chunks
Agent: improved results, done.
Total tool calls: 3
```

### Why it works for FCA docs

- **Query complexity varies**: Simple factual questions need 1 vector search. Cross-sourcebook comparisons need multiple targeted searches. Relationship queries need graph traversal. An agent adapts its strategy.
- **Composes Approaches 1 + 2**: The agent has both vector search and graph traversal as tools, choosing the right one (or both) per query.
- **Self-correcting**: If initial results are poor, the agent can reformulate and retry.
- **Sourcebook filtering**: The agent can strategically filter searches to specific sourcebooks when the query implies a domain.

### Strengths
- Handles any query complexity — from simple to multi-hop
- Composes vector search + graph traversal intelligently
- Self-correcting via reformulation
- Shows production-readiness (adaptive retrieval is SOTA in 2026)

### Weaknesses
- Higher latency: 2-5x slower than Approach 1 (multiple LLM calls for tool selection)
- Non-deterministic: same query may produce different tool call sequences
- Harder to evaluate: tool traces vary, making RAGAS comparison less clean
- Depends on LLM's tool-calling quality (free tier models may be less reliable)
- Agent can over-retrieve or make unnecessary tool calls

### Configuration

```python
MAX_AGENT_STEPS = 5         # prevent infinite loops
AGENT_LLM = "qwen3"        # needs good tool calling support
SEARCH_TOP_K_PER_CALL = 5  # results per vector search
GRAPH_HOPS_DEFAULT = 2     # default graph expansion depth
```

---

## Comparison Framework

### How the three approaches will be evaluated

Same golden dataset (20-25 questions), same generation pipeline, same metrics. Only the retrieval step differs.

| Metric | What it measures | Tool |
|--------|-----------------|------|
| Context Precision | Are retrieved chunks relevant? | RAGAS |
| Context Recall | Did we find all relevant chunks? | RAGAS |
| Faithfulness | Is the answer grounded in context? | RAGAS |
| Answer Relevancy | Does the answer address the question? | RAGAS |
| Retrieval Latency | How fast is the retrieval step? | Custom timing |
| Total Latency | End-to-end response time | Custom timing |
| Citation Accuracy | Do citations point to correct rules? | Custom metric |

### Question types designed to expose strengths/weaknesses

| Question type | Expected winner | Example |
|---|---|---|
| Simple factual (single sourcebook) | Hybrid+Rerank (fastest, sufficient) | "What is the cancellation period under BCOBS?" |
| Keyword-specific | Hybrid+Rerank (BM25 catches exact terms) | "What are the MiFID provisions for inducements?" |
| Relationship/dependency | Graph RAG (native graph traversal) | "What rules reference COBS 2.1.1R?" |
| Cross-sourcebook comparison | Agentic RAG (decomposes per sourcebook) | "Compare inducement rules across COBS, ICOBS, MCOB" |
| Ambiguous/vague | Agentic RAG (reformulates query) | "Tell me about client obligations" |
| Multi-hop reasoning | Graph RAG or Agentic RAG | "If a firm is CASS large, what reconciliation rules apply?" |

### The trade-off story (for the walkthrough)

> "Hybrid+Rerank is what I'd ship for production — it handles 80% of queries well, it's fast (~300ms), deterministic, and easy to monitor. Graph RAG adds value specifically for relationship queries and cross-sourcebook discovery — the FCA Handbook's thousands of cross-references form a natural graph that text search can't exploit. I'd deploy Graph RAG as an enhancement layer for power users doing impact analysis. Agentic RAG gives the best results on complex questions but at 2-5x the latency and with non-determinism that makes it harder to test and monitor. I'd reserve it for a 'deep research' mode with appropriate UX expectations."

---

## Framework Decision

**No frameworks.** All three approaches are built with direct library calls:

| Need | Library | Not |
|------|---------|-----|
| Vector search | `weaviate-client` | LangChain Weaviate wrapper |
| Graph queries | `neo4j` Python driver | LangChain Neo4j wrapper |
| Reranking | `flashrank` | LangChain reranker |
| Embeddings | `sentence-transformers` | LangChain embeddings |
| LLM + tool calling | `openai` SDK (OpenRouter-compatible) | LangChain/LangGraph |
| Agent loop | Raw while loop with tool dispatch (~50-80 lines) | LangGraph state machine |

**Why no framework**: Evaluators can read every line of retrieval logic. No hidden abstractions. The agent loop is simple enough that a framework adds complexity without value.

**Fallback**: If the free-tier LLM's tool calling is unreliable, add LangGraph for just the agent loop. This is a targeted dependency, not a full-framework adoption.
