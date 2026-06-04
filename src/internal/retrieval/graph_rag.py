"""Approach 2: Graph RAG — Hybrid seed search + Neo4j cross-reference expansion.

Seeds from Weaviate hybrid search, expands via Neo4j REFERENCES edges (1-2 hops),
fetches chunks for expanded rules, late-stage FlashRank rerank against original query.
"""

from __future__ import annotations

import time

import weaviate.classes.query as wvq

from src.config import Settings, settings
from src.dependencies import get_neo4j_driver, get_weaviate_client, get_embedding_model
from src.internal.retrieval.base import (
    BaseRetriever,
    RetrievedChunk,
    RetrievalResult,
    weaviate_obj_to_chunk,
)
from src.internal.retrieval.hybrid_rerank import HybridRerankRetriever


class GraphRAGRetriever(BaseRetriever):
    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg
        self.search = HybridRerankRetriever(cfg)
        self.driver = get_neo4j_driver(cfg)
        self.client = get_weaviate_client(cfg)
        self.collection = self.client.collections.get(cfg.weaviate_collection)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        sourcebook_filter: str | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self.cfg.final_top_k
        start = time.time()

        # Step 1: Hybrid search for seed rules (wider net than approach 1)
        seed_result = self.search.retrieve(query, top_k=10, sourcebook_filter=sourcebook_filter)
        seed_chunks = seed_result.chunks
        seed_rule_ids = list({c.rule_id for c in seed_chunks})

        print(f"    seeds: {seed_rule_ids[:5]}")

        # Step 2: Graph expansion — find connected rules via REFERENCES edges
        expanded_ids = self._graph_expand(seed_rule_ids)
        print(f"    graph expanded: {len(expanded_ids)} connected rules")

        # Step 3: Fetch chunks from Weaviate for expanded rule IDs
        expanded_chunks = self._fetch_chunks_by_ids(expanded_ids)

        # Step 4: Combine seeds + expanded, deduplicate by chunk_id
        all_chunks = seed_chunks + expanded_chunks
        seen: dict[str, RetrievedChunk] = {}
        for c in all_chunks:
            if c.chunk_id not in seen:
                seen[c.chunk_id] = c
        candidates = list(seen.values())

        print(f"    candidates after dedup: {len(candidates)}")

        # Step 5: Late-stage FlashRank rerank against original query
        if candidates:
            reranked = self.search.rerank_chunks(query, candidates)
            chunks = reranked[:top_k]
        else:
            chunks = []

        elapsed_ms = (time.time() - start) * 1000
        return RetrievalResult(
            query=query,
            chunks=chunks,
            retrieval_time_ms=elapsed_ms,
            approach=f"graph_rag({len(expanded_ids)}expanded)",
        )

    def _graph_expand(self, seed_ids: list[str]) -> list[str]:
        """Traverse Neo4j REFERENCES edges 1-2 hops from seed rules."""
        if not seed_ids:
            return []

        hops = self.cfg.graph_hops
        cypher = f"""
            MATCH (seed:Rule)-[:REFERENCES*1..{hops}]-(connected:Rule)
            WHERE seed.id IN $seed_ids
              AND NOT connected.id IN $seed_ids
              AND connected.stub = false
            RETURN DISTINCT connected.id AS id
            LIMIT $limit
        """

        with self.driver.session(database=self.cfg.neo4j_database) as session:
            result = session.run(
                cypher,
                seed_ids=seed_ids,
                limit=self.cfg.graph_expansion_limit,
            )
            return [record["id"] for record in result]

    def _fetch_chunks_by_ids(self, rule_ids: list[str]) -> list[RetrievedChunk]:
        """Fetch chunks from Weaviate for a list of rule IDs."""
        if not rule_ids:
            return []

        chunks = []
        for rule_id in rule_ids:
            results = self.collection.query.fetch_objects(
                filters=wvq.Filter.by_property("rule_id").equal(rule_id),
                limit=1,
            )
            if results.objects:
                chunk = weaviate_obj_to_chunk(results.objects[0], score=0.0)
                chunks.append(chunk)

        return chunks


# --- Runnable standalone ---

if __name__ == "__main__":
    retriever = GraphRAGRetriever()

    test_queries = [
        "What must a firm do when providing investment services?",
        "What rules reference COBS 2.1.1R?",
        "What are the client money segregation requirements?",
    ]

    for q in test_queries:
        result = retriever.retrieve(q)
        print(f"\nQuery: {q}")
        print(f"Approach: {result.approach}")
        print(f"Time: {result.retrieval_time_ms:.0f}ms")
        for c in result.chunks:
            print(f"  [{c.score:.4f}] {c.display_id} ({c.sourcebook}) — {c.text[:80]}...")
