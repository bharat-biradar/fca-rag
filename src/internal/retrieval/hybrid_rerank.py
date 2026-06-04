"""Approach 1: Hybrid Search + Cross-Encoder Reranking.

Weaviate hybrid (BM25 + dense vector) → FlashRank rerank → top-k.
"""

from __future__ import annotations

import time

import weaviate.classes.query as wvq
from flashrank import Ranker, RerankRequest

from src.config import Settings, settings
from src.dependencies import get_weaviate_client, get_embedding_model
from src.internal.retrieval.base import (
    BaseRetriever,
    RetrievedChunk,
    RetrievalResult,
)

_ranker: Ranker | None = None


def _get_ranker() -> Ranker:
    global _ranker
    if _ranker is None:
        _ranker = Ranker()
    return _ranker


class HybridRerankRetriever(BaseRetriever):
    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg
        self.client = get_weaviate_client(cfg)
        self.model = get_embedding_model(cfg)
        self.collection = self.client.collections.get(cfg.weaviate_collection)
        self.ranker = _get_ranker()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        sourcebook_filter: str | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self.cfg.final_top_k
        start = time.time()

        # 1. Embed query
        query_vec = self.model.encode(query, normalize_embeddings=True).tolist()

        # 2. Weaviate hybrid search
        filters = None
        if sourcebook_filter:
            filters = wvq.Filter.by_property("sourcebook").equal(sourcebook_filter)

        results = self.collection.query.hybrid(
            query=query,
            vector=query_vec,
            alpha=self.cfg.hybrid_alpha,
            limit=self.cfg.initial_retrieval_k,
            filters=filters,
            return_metadata=wvq.MetadataQuery(score=True),
        )

        if not results.objects:
            return RetrievalResult(
                query=query, chunks=[], retrieval_time_ms=0, approach="hybrid_rerank"
            )

        # 3. Prepare passages for FlashRank
        passages = []
        for i, obj in enumerate(results.objects):
            passages.append({
                "id": i,
                "text": obj.properties["text"],
                "meta": obj.properties,
            })

        # 4. Rerank
        rerank_req = RerankRequest(query=query, passages=passages)
        reranked = self.ranker.rerank(rerank_req)

        # 5. Take top-k, convert to RetrievedChunk
        chunks = []
        for item in reranked[:top_k]:
            meta = item["meta"]
            chunks.append(RetrievedChunk(
                text=item["text"],
                rule_id=meta.get("rule_id", ""),
                rule_type=meta.get("rule_type", ""),
                score=float(item["score"]),
                sourcebook=meta.get("sourcebook", ""),
                chapter=meta.get("chapter", ""),
                chapter_title=meta.get("chapter_title", ""),
                section=meta.get("section", ""),
                section_title=meta.get("section_title", ""),
                chunk_id=meta.get("chunk_id", ""),
                sub_paragraph=meta.get("sub_paragraph", ""),
                page=meta.get("page", 0),
                is_annex=meta.get("is_annex", False),
                is_table=meta.get("is_table", False),
                defined_terms=meta.get("defined_terms", []),
                cross_references=meta.get("cross_references", []),
            ))

        elapsed_ms = (time.time() - start) * 1000
        return RetrievalResult(
            query=query,
            chunks=chunks,
            retrieval_time_ms=elapsed_ms,
            approach="hybrid_rerank",
        )

    def rerank_chunks(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Rerank an arbitrary list of RetrievedChunks against a query using FlashRank."""
        if not chunks:
            return []

        passages = [
            {"id": i, "text": c.text, "meta": c}
            for i, c in enumerate(chunks)
        ]

        rerank_req = RerankRequest(query=query, passages=passages)
        reranked = self.ranker.rerank(rerank_req)

        for item in reranked:
            item["meta"].score = float(item["score"])

        return [item["meta"] for item in reranked]


# --- Runnable standalone ---

if __name__ == "__main__":
    retriever = HybridRerankRetriever()

    test_queries = [
        "firm must act honestly fairly professionally",
        "client money segregation requirements",
        "mortgage affordability assessment",
    ]

    for q in test_queries:
        result = retriever.retrieve(q)
        print(f"\nQuery: {q}")
        print(f"Time: {result.retrieval_time_ms:.0f}ms")
        for c in result.chunks:
            print(f"  [{c.score:.4f}] {c.display_id} ({c.sourcebook}) — {c.text[:80]}...")
