"""Approach 3 (v3): Agentic RAG — Enhanced plan-then-execute with rule ID lookup.

Phase 1: LLM decomposes query + extracts mentioned rule IDs (1 call)
Phase 1b: Direct lookup for any rule IDs found in query/plan
Phase 2: Broad unfiltered search with original + reformulated query
Phase 3: Targeted searches per sub-query
Phase 4: Graph expansion on top seeds via Neo4j
Phase 5: Single FlashRank rerank against original query

Tracks planning token usage.
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
  "reformulated_query": "the question rewritten using specific FCA regulatory terminology",
  "rule_ids": ["list of any specific rule IDs mentioned in the question, e.g. COBS 2.1.1, CASS 7.11.34"]
}

Rules:
- sub_queries: Break the question into specific searchable parts. "What are obligations for fair communications across banking and insurance?" becomes ["fair clear not misleading communications", "banking customer disclosure requirements", "insurance product information disclosure"].
- reformulated_query: Rewrite using FCA terminology. "consumer protections" becomes "client best interests fair treatment product disclosure requirements".
- rule_ids: Extract any rule IDs explicitly mentioned in the question. Include the base ID without type suffix. If none mentioned, return empty list [].
- Output ONLY the JSON, no other text."""


# ---------------------------------------------------------------------------
# Helpers (extracted for testability)
# ---------------------------------------------------------------------------


def parse_plan(raw: str, fallback_query: str) -> dict:
    """Parse LLM plan response, strip markdown, apply defaults and caps."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    plan = json.loads(raw)
    plan.setdefault("sub_queries", [])
    plan.setdefault("reformulated_query", fallback_query)
    plan.setdefault("rule_ids", [])
    plan["sub_queries"] = plan["sub_queries"][:3]
    plan["rule_ids"] = plan["rule_ids"][:10]
    return plan


def clean_rule_id(rule_id: str) -> str:
    """Strip type suffix: COBS 2.1.1R -> COBS 2.1.1"""
    return re.sub(r"[RGDEUK]{1,2}$", "", rule_id.strip()).strip()


def deduplicate_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Deduplicate by chunk_id, keeping first seen."""
    seen: dict[str, RetrievedChunk] = {}
    for c in chunks:
        if c.chunk_id not in seen:
            seen[c.chunk_id] = c
    return list(seen.values())


def inject_lookups(
    reranked: list[RetrievedChunk],
    lookups: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """Guarantee looked-up rules appear in final results."""
    already_in = {c.chunk_id for c in reranked}
    missing = [c for c in lookups if c.chunk_id not in already_in]
    if missing:
        slots = min(len(missing), top_k)
        while len(reranked) + slots > top_k and reranked:
            reranked.pop()
        reranked.extend(missing[:slots])
    return reranked


def parse_self_eval(text: str) -> int:
    """Parse YES/NO lines from self-eval response. Returns count of YES."""
    lines = [l.strip() for l in text.upper().split("\n") if l.strip()]
    return sum(1 for l in lines if l.startswith("YES"))


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class AgenticV3Retriever(BaseRetriever):
    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg
        self.client_wv = get_weaviate_client(cfg)
        self.collection = self.client_wv.collections.get(cfg.weaviate_collection)
        self.embed_model = get_embedding_model(cfg)
        self.ranker = _get_ranker()
        self.neo4j = get_neo4j_driver(cfg)

        # LLM for planning: Bedrock Sonnet 4.6 > Gemini Flash > OpenRouter
        from openai import OpenAI
        aws_key = os.getenv("AWS_ACCESS_KEY_ID", "")
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if aws_key:
            import litellm
            self._use_litellm = True
            self.llm_model = "bedrock/global.anthropic.claude-sonnet-4-6"
        elif gemini_key:
            self._use_litellm = False
            self.llm_client = OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=gemini_key,
            )
            self.llm_model = "gemini-2.5-flash"
        else:
            self._use_litellm = False
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
        plan, planning_tokens = self._plan_query(query)
        print(f"    plan: {json.dumps(plan, ensure_ascii=False)[:120]}")
        print(f"    planning tokens: {planning_tokens}")

        # Phase 1b: Direct lookup for any rule IDs mentioned in query/plan
        lookup_chunks = self._lookup_rule_ids(plan.get("rule_ids", []))
        if lookup_chunks:
            print(f"    lookups: {len(lookup_chunks)} rules fetched directly")

        # Phase 2 & 3: Execute all searches concurrently
        candidates = self._execute_searches(query, plan, sourcebook_filter)
        print(f"    searches: {len(candidates)} raw candidates")

        # Phase 4: Graph expansion on top seeds (include lookup chunks as seeds)
        all_for_graph = candidates + lookup_chunks
        graph_chunks = self._graph_expand(all_for_graph)
        candidates.extend(graph_chunks)
        candidates.extend(lookup_chunks)
        print(f"    graph: +{len(graph_chunks)} expanded, {len(candidates)} total")

        # Phase 5: Deduplicate + single FlashRank rerank
        chunks = self._final_rerank(query, candidates, top_k)

        # Phase 6: Guarantee looked-up rules are in final results
        if lookup_chunks:
            pre_inject = len(chunks)
            chunks = inject_lookups(chunks, lookup_chunks, top_k)
            injected = len(chunks) - pre_inject + (pre_inject - len([c for c in chunks if c.chunk_id not in {l.chunk_id for l in lookup_chunks}]))
            if any(c.chunk_id in {l.chunk_id for l in lookup_chunks} for c in chunks):
                print(f"    lookups injected into results")

        elapsed_ms = (time.time() - start) * 1000
        return RetrievalResult(
            query=query,
            chunks=chunks,
            retrieval_time_ms=elapsed_ms,
            approach=f"agentic_v3({len(candidates)}cand)",
            planning_tokens=planning_tokens,
        )

    # ------------------------------------------------------------------
    # Phase 1: Query planning
    # ------------------------------------------------------------------

    def _llm_call(self, messages, **kwargs):
        """Route LLM call through litellm (Bedrock) or OpenAI client."""
        if getattr(self, "_use_litellm", False):
            import litellm
            return litellm.completion(model=self.llm_model, messages=messages, **kwargs)
        return self.llm_client.chat.completions.create(model=self.llm_model, messages=messages, **kwargs)

    def _plan_query(self, query: str) -> tuple[dict, int]:
        """Single LLM call to decompose the query. Returns (plan, planning_tokens)."""
        try:
            response = self._llm_call(
                messages=[
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            raw = response.choices[0].message.content or "{}"

            # Track planning tokens
            usage = response.usage
            planning_tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0

            plan = parse_plan(raw, query)
            return plan, planning_tokens
        except (json.JSONDecodeError, Exception) as e:
            print(f"    [warn] planning failed: {e}")
            return {"sub_queries": [], "reformulated_query": query, "rule_ids": []}, 0

    # ------------------------------------------------------------------
    # Phase 1b: Direct rule ID lookup
    # ------------------------------------------------------------------

    def _lookup_rule_ids(self, rule_ids: list[str]) -> list[RetrievedChunk]:
        """Fetch rules directly by ID — guaranteed retrieval, no search needed."""
        if not rule_ids:
            return []

        chunks = []
        for rule_id in rule_ids:
            clean_id = clean_rule_id(rule_id)
            results = self.collection.query.fetch_objects(
                filters=wvq.Filter.by_property("rule_id").equal(clean_id),
                limit=1,
            )
            if results.objects:
                # Score 1.0 so lookups are prioritized as graph expansion seeds
                chunks.append(weaviate_obj_to_chunk(results.objects[0], score=1.0))
        return chunks

    # ------------------------------------------------------------------
    # Phase 2 & 3: Search execution (concurrent)
    # ------------------------------------------------------------------

    def _execute_searches(
        self, query: str, plan: dict, sourcebook_filter: str | None
    ) -> list[RetrievedChunk]:
        """Run all searches concurrently and collect raw candidates."""
        search_tasks = []

        # Phase 2: Broad unfiltered searches
        search_tasks.append((query, None, 10))
        if plan["reformulated_query"] != query:
            search_tasks.append((plan["reformulated_query"], None, 10))

        # Phase 3: Sub-query searches (all unfiltered — let reranker decide relevance)
        for sub_q in plan["sub_queries"]:
            search_tasks.append((sub_q, None, 5))

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
        seed_ids = sorted(seen_ids, key=seen_ids.get, reverse=True)[:3]

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
    import sys
    from src.internal.generation.llm import LLMClient
    from src.internal.generation.prompts import SYSTEM_PROMPT, build_user_prompt, extract_citations

    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]
    else:
        queries = [
            "What is the client's best interests rule under COBS?",
            "What are the obligations for fair, clear and not misleading communications across sourcebooks?",
            "What protections exist for consumers buying financial products?",
        ]

    retriever = AgenticV3Retriever()
    llm = LLMClient()

    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        result = retriever.retrieve(q)
        print(f"Approach: {result.approach}")
        print(f"Time: {result.retrieval_time_ms:.0f}ms")
        for c in result.chunks:
            print(f"  [{c.score:.4f}] {c.display_id} ({c.sourcebook})")

        resp = llm.generate(SYSTEM_PROMPT, build_user_prompt(q, result.chunks))
        print(f"\n--- Answer ---\n{resp.text}")
        print(f"\nCitations: {extract_citations(resp.text)}")
