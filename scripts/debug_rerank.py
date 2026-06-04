"""Debug reranker: show before/after rankings to compare.

Usage: python3 -m scripts.debug_rerank
"""

from __future__ import annotations

import weaviate.classes.query as wvq
from flashrank import RerankRequest

from src.internal.retrieval.hybrid_rerank import HybridRerankRetriever

QUERY = "What are the general obligations for firms under COBS 2.1?"


def main():
    r = HybridRerankRetriever()
    query_vec = r.model.encode(QUERY, normalize_embeddings=True).tolist()

    # Raw Weaviate hybrid results (before rerank)
    results = r.collection.query.hybrid(
        query=QUERY,
        vector=query_vec,
        alpha=r.cfg.hybrid_alpha,
        limit=r.cfg.initial_retrieval_k,
        return_metadata=wvq.MetadataQuery(score=True),
    )

    print(f"=== BEFORE RERANK (Weaviate hybrid, top {len(results.objects)}) ===")
    passages = []
    for i, obj in enumerate(results.objects):
        p = obj.properties
        rid = f"{p['rule_id']}{p['rule_type']}"
        score = obj.metadata.score
        marker = " <<<" if "COBS 2.1.1" in p["rule_id"] else ""
        print(f"  {i+1:2d}. [{score:.4f}] {rid:25s} ({p['sourcebook']}) — {p['text'][:60]}...{marker}")
        passages.append({"id": i, "text": p["text"], "meta": p})

    # After rerank
    reranked = r.ranker.rerank(RerankRequest(query=QUERY, passages=passages))

    print(f"\n=== AFTER RERANK (FlashRank, all {len(reranked)}) ===")
    for i, item in enumerate(reranked):
        rid = f"{item['meta']['rule_id']}{item['meta']['rule_type']}"
        marker = " <<<" if "COBS 2.1.1" in item["meta"]["rule_id"] else ""
        print(f"  {i+1:2d}. [{item['score']:.4f}] {rid:25s} — {item['text'][:60]}...{marker}")

    # Without rerank — what would top 5 be?
    print("\n=== TOP 5 WITHOUT RERANK ===")
    for i, obj in enumerate(results.objects[:5]):
        p = obj.properties
        rid = f"{p['rule_id']}{p['rule_type']}"
        print(f"  {i+1}. {rid:25s} — {p['text'][:80]}...")

    # Top 5 with rerank
    print("\n=== TOP 5 WITH RERANK ===")
    for i, item in enumerate(reranked[:5]):
        rid = f"{item['meta']['rule_id']}{item['meta']['rule_type']}"
        print(f"  {i+1}. {rid:25s} — {item['text'][:80]}...")


if __name__ == "__main__":
    main()
