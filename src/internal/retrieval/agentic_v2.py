"""Approach 3 (v2): Agentic RAG — LLM plans once, then deterministic execution.

Phase 1: LLM decomposes query into sub-queries + sourcebooks (1 call)
Phase 2: Broad unfiltered search with original + reformulated query
Phase 3: Targeted searches per sourcebook and sub-query
Phase 4: Graph expansion on top seeds via Neo4j
Phase 5: Single FlashRank rerank against original query
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import weaviate.classes.query as wvq
from flashrank import RerankRequest

from src.config import Settings, settings
from src.dependencies import get_embedding_model, get_neo4j_driver, get_weaviate_client
from src.internal.retrieval.base import (
    BaseRetriever,
    RetrievedChunk,
    RetrievalResult,
    weaviate_obj_to_chunk,
)
from src.internal.retrieval.hybrid_rerank import _get_ranker

# ---------------------------------------------------------------------------
# Planner prompt
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """\
You are a query analyzer for the UK FCA Handbook. Given a user question, output a JSON search plan.

The FCA Handbook has these sourcebooks:
BCOBS (Banking), CASS (Client Assets), CMCOB (Claims Management), COBS (Conduct of Business),
ESG (Environmental/Social/Governance), FPCOB (Funeral Plans), ICOBS (Insurance),
MAR (Market Conduct), MCOB (Mortgages), PDCOB (Pensions Dashboards).

Output this exact JSON structure:
{
  "sub_queries": ["list of 1-3 specific search queries derived from the question"],
  "reformulated_query": "the question rewritten using specific FCA regulatory terminology"
}

Rules:
- sub_queries: Break the question into specific, DETAILED search queries (15-30 words each). Each sub-query should be a complete sentence or phrase, NOT short keyword fragments.
  BAD: ["CASS primary pooling event", "client money CASS"]
  GOOD: ["What happens to client money entitlements when a primary pooling event is triggered under CASS", "How are client money claims calculated and distributed after a firm failure under CASS 7A"]
- reformulated_query: Expand the question with synonyms and regulatory terms that the actual rule text might use. Do NOT guess specific rule IDs or section numbers.
  BAD: "Consequences of a primary pooling event on client money entitlements under CASS rules" (just rephrased, no new terms)
  BAD: "CASS 7A primary pooling event" (guessing a section number)
  GOOD: "primary pooling event client money entitlements distribution claims calculation firm failure insolvency segregated funds" (adds terms the rule text likely contains)
- Output ONLY the JSON, no other text."""


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class AgenticV2Retriever(BaseRetriever):
    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg
        self.client_wv = get_weaviate_client(cfg)
        self.collection = self.client_wv.collections.get(cfg.weaviate_collection)
        self.embed_model = get_embedding_model(cfg)
        self.ranker = _get_ranker()
        self.neo4j = get_neo4j_driver(cfg)

        # LLM for planning (Gemini if available, else OpenRouter)
        from openai import OpenAI
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if gemini_key:
            self.llm_client = OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=gemini_key,
            )
            self.llm_model = "gemini-2.5-flash"
        else:
            self.llm_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=cfg.openrouter_api_key,
            )
            self.llm_model = cfg.generation_model

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        sourcebook_filter: str | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self.cfg.final_top_k
        start = time.time()

        # Phase 1: LLM plans the search strategy
        plan = self._plan_query(query)
        print(f"    plan: {json.dumps(plan, ensure_ascii=False)[:120]}")

        # Phase 2 & 3: Execute all searches concurrently
        candidates = self._execute_searches(query, plan, sourcebook_filter)
        print(f"    searches: {len(candidates)} raw candidates")

        # Phase 4: Graph expansion on top seeds
        graph_chunks = self._graph_expand(candidates)
        candidates.extend(graph_chunks)
        print(f"    graph: +{len(graph_chunks)} expanded, {len(candidates)} total")

        # Phase 5: Deduplicate + single FlashRank rerank
        chunks = self._final_rerank(query, candidates, top_k)

        elapsed_ms = (time.time() - start) * 1000
        return RetrievalResult(
            query=query,
            chunks=chunks,
            retrieval_time_ms=elapsed_ms,
            approach=f"agentic_v2({len(candidates)}candidates)",
        )

    # ------------------------------------------------------------------
    # Phase 1: Query planning
    # ------------------------------------------------------------------

    def _plan_query(self, query: str) -> dict:
        """Single LLM call to decompose the query."""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            raw = response.choices[0].message.content or "{}"

            # Strip markdown fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw.strip())

            plan = json.loads(raw)

            # Validate structure
            plan.setdefault("sub_queries", [])
            plan.setdefault("reformulated_query", query)

            # Cap to avoid runaway plans
            plan["sub_queries"] = plan["sub_queries"][:3]

            return plan
        except (json.JSONDecodeError, Exception) as e:
            print(f"    [warn] planning failed: {e}")
            return {"sub_queries": [], "reformulated_query": query}

    # ------------------------------------------------------------------
    # Phase 2 & 3: Search execution (concurrent)
    # ------------------------------------------------------------------

    def _execute_searches(
        self, query: str, plan: dict, sourcebook_filter: str | None
    ) -> list[RetrievedChunk]:
        """Run all searches concurrently and collect raw candidates."""
        search_tasks = []

        # Phase 2: Broad unfiltered searches
        search_tasks.append((query, None, 20))
        if plan["reformulated_query"] != query:
            search_tasks.append((plan["reformulated_query"], None, 20))

        # Phase 3: Sub-query searches (all unfiltered — let reranker decide relevance)
        for sub_q in plan["sub_queries"]:
            search_tasks.append((sub_q, None, 10))

        # Only apply sourcebook filter if explicitly passed by caller
        if sourcebook_filter:
            search_tasks.append((query, sourcebook_filter, 10))

        # Execute all concurrently
        all_chunks: list[RetrievedChunk] = []

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(self._raw_hybrid_search, q, sb, k)
                for q, sb, k in search_tasks
            ]
            for future in futures:
                all_chunks.extend(future.result())

        return all_chunks

    def _raw_hybrid_search(
        self, query: str, sourcebook: str | None, k: int
    ) -> list[RetrievedChunk]:
        """Weaviate hybrid search WITHOUT FlashRank reranking. Returns raw candidates."""
        query_vec = self.embed_model.encode(query, normalize_embeddings=True).tolist()

        filters = None
        if sourcebook:
            filters = wvq.Filter.by_property("sourcebook").equal(sourcebook)

        results = self.collection.query.hybrid(
            query=query,
            vector=query_vec,
            alpha=self.cfg.hybrid_alpha,
            limit=k,
            filters=filters,
            return_metadata=wvq.MetadataQuery(score=True),
        )

        chunks = []
        for obj in results.objects:
            score = obj.metadata.score if obj.metadata.score else 0.0
            chunks.append(weaviate_obj_to_chunk(obj, score=score))
        return chunks

    # ------------------------------------------------------------------
    # Phase 4: Graph expansion
    # ------------------------------------------------------------------

    def _graph_expand(self, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Expand top seed rules via Neo4j REFERENCES edges."""
        # Get top 5 unique rule_ids by score
        seen_ids: dict[str, float] = {}
        for c in candidates:
            if c.rule_id not in seen_ids or c.score > seen_ids[c.rule_id]:
                seen_ids[c.rule_id] = c.score
        seed_ids = sorted(seen_ids, key=seen_ids.get, reverse=True)[:5]

        if not seed_ids:
            return []

        # Neo4j traversal with per-seed limit
        hops = self.cfg.graph_hops
        cypher = f"""
            MATCH (seed:Rule)-[:REFERENCES*1..{hops}]-(connected:Rule)
            WHERE seed.id IN $seed_ids
              AND NOT connected.id IN $seed_ids
              AND connected.stub = false
            RETURN DISTINCT connected.id AS id
            LIMIT $limit
        """

        try:
            with self.neo4j.session(database=self.cfg.neo4j_database) as session:
                result = session.run(
                    cypher,
                    seed_ids=seed_ids,
                    limit=self.cfg.graph_expansion_limit,
                )
                expanded_ids = [record["id"] for record in result]
        except Exception as e:
            print(f"    [warn] graph expansion failed: {e}")
            return []

        # Fetch chunks from Weaviate for expanded rule IDs
        chunks = []
        for rule_id in expanded_ids:
            results = self.collection.query.fetch_objects(
                filters=wvq.Filter.by_property("rule_id").equal(rule_id),
                limit=1,
            )
            if results.objects:
                chunks.append(weaviate_obj_to_chunk(results.objects[0], score=0.0))

        return chunks

    # ------------------------------------------------------------------
    # Phase 5: Final rerank
    # ------------------------------------------------------------------

    def _final_rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Deduplicate by chunk_id, then single FlashRank rerank against original query."""
        # Deduplicate
        seen: dict[str, RetrievedChunk] = {}
        for c in candidates:
            if c.chunk_id not in seen:
                seen[c.chunk_id] = c
        unique = list(seen.values())

        if not unique:
            return []

        # Single FlashRank pass
        passages = [
            {"id": i, "text": c.text, "meta": c}
            for i, c in enumerate(unique)
        ]
        rerank_req = RerankRequest(query=query, passages=passages)
        reranked = self.ranker.rerank(rerank_req)

        for item in reranked:
            item["meta"].score = float(item["score"])

        return [item["meta"] for item in reranked[:top_k]]


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    retriever = AgenticV2Retriever()

    test_queries = [
        "What is the client's best interests rule under COBS?",
        "What are the obligations for fair, clear and not misleading communications across sourcebooks?",
        "What protections exist for consumers buying financial products?",
        "How should firms handle conflicts of interest?",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        result = retriever.retrieve(q)
        print(f"Approach: {result.approach}")
        print(f"Time: {result.retrieval_time_ms:.0f}ms")
        for c in result.chunks:
            print(f"  [{c.score:.4f}] {c.display_id} ({c.sourcebook}) — {c.text[:80]}...")
