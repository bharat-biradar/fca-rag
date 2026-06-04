# Evaluation Findings & Observations

## Reranker Model Impact

**FlashRank TinyBERT (3MB, default):** Scores cluster at 0.99 — poor discrimination. A generic rule about client agreements scores the same as the exact rule about "fair, clear and not misleading." The model treats everything as relevant.

**FlashRank MiniLM-L-12 (120MB, upgraded):** Scores range 0.01–0.14 — much more discriminating. But still misses some expected rules because they're not in the candidate pool to begin with.

**Takeaway:** Reranker quality matters, but can only reorder what's retrieved. The ceiling is set by the retrieval step (Weaviate hybrid search).

## Reformulated Query Reranking Paradox

We tried reranking the final candidate pool against the reformulated query (expanded with regulatory terms) instead of the original query. Result: **worse performance.** The reformulated query is so broad ("client best interests product disclosure requirements pre-contractual information obligations fair clear not misleading communications standards...") that generic rules outscore specific ones.

**Fix:** Rerank against the original query — it's more focused and discriminating.

## Citation Accuracy vs RAGAS Scores

These metrics tell different stories:

| Scenario | RAGAS | Citation Accuracy |
|---|---|---|
| Retrieved BCOBS 2.2.2G, expected BCOBS 2.2.1R | 1.00 (content matches) | 0.00 (wrong rule ID) |
| Found the right section but wrong sub-rule | High | Low |

**Why:** RAGAS judges content relevance. Citation accuracy does strict rule ID matching. The retriever consistently finds rules *adjacent* to the expected ones — same section, same topic, but different rule numbers.

**Implication:** The golden dataset's expected_rule_ids may be too narrow. Multiple rules in the same section often address the same topic. RAGAS is the fairer measure for comparing approaches.

## Sub-Query Length Impact

| Approach | Keyword Recall | Ambiguous Recall |
|---|---|---|
| Short sub-queries | 0.611 | 0.333 |
| Long sub-queries (15-30 words) | 0.694 | 0.167 |

Long sub-queries help keyword queries (+0.08) but hurt ambiguous ones (-0.17). Short sub-queries are more stable overall.

## Sourcebook Filtering in Agent

**Problem found:** The agent's system prompt included a sourcebook mapping that caused it to filter searches by default. For "What are the FCA rules on inducements?", the agent added `sourcebook=COBS` to its first search, narrowing the candidate pool and missing cross-sourcebook results.

**Fix:** Removed sourcebook mapping from prompt. Added guideline: "ALWAYS start with an unfiltered search." Agent now starts from the same baseline as Hybrid and can only improve.

**Impact:** Q2 (BCOBS 5.1.1) went from 0/1 to 1/1. Q5 (inducements) went from 0/2 to 1/2.

## Double Reranking Problem

The v1 agentic retriever reranked twice:
1. Each `search_rules` call: FlashRank reranked 50→5 against the sub-query
2. Final: FlashRank reranked all collected chunks against the original query

A chunk ranked #1 for a targeted sub-query might drop to #8 when reranked against the broad original query. This is why v1 agentic sometimes performed worse than Hybrid.

**Fix in v2:** Skip internal FlashRank for agent searches. Collect raw Weaviate candidates, do one final rerank at the end.

## Collection Size Impact

At 8,398 chunks, a single Weaviate hybrid search with k=50 covers a significant fraction of the collection. Multiple targeted searches (the agent's strategy) overlap heavily and the extra LLM reasoning adds noise.

**Hypothesis:** Agentic RAG would show more value on a much larger collection (100K+ chunks) where a single search can't cover enough ground.

## Model Quality for Agent Planning

| Model | Tool calling quality | Query decomposition |
|---|---|---|
| OpenRouter gpt-oss-120b (free) | Poor — malformed JSON, premature stopping | Basic |
| Gemini 2.5 Flash | Good — reliable JSON, follows instructions | Good |
| Bedrock Sonnet 4.6 | Best — clean structured output | Best decomposition |

Agent quality is directly proportional to the planning LLM quality.

## V2 Chunker Impact

Grouped sub-paragraphs (800-1500 chars) instead of splitting at every (1), (2):

| Metric | V1 Chunks | V2 Chunks | Delta |
|---|---|---|---|
| Hybrid Recall | 0.452 | 0.516 | **+14%** |
| Cross-reference | 0.389 | 0.653 | **+68%** |
| Simple factual | 0.833 | 1.000 | +20% |

Better chunks lifted all approaches. The chunking strategy matters more than the retrieval architecture.

## RAGAS Evaluator Speed

| Evaluator | Time per question | Notes |
|---|---|---|
| OpenRouter (free tier) | 40-50s | Rate limited at 16 req/min |
| Gemini 2.5 Flash | 25-35s | Decent but still slow |
| Bedrock Haiku 3 | 1.5-5s | 20x faster, best option |
| Bedrock Haiku 4.5 | Failed | Scores 0 — likely structured output format mismatch |
| Ollama cogito:8b | Failed | Empty JSON output |
| Ollama qwen3:14b | Failed | Thinking mode interferes |
