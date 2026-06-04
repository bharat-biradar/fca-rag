"""Validate Weaviate ingestion: count, search quality, filters, edge cases.

Usage: python3 -m tests.integration.test_weaviate
"""

from __future__ import annotations

import sys

import weaviate.classes.query as wvq

from src.config import settings
from src.dependencies import get_weaviate_client, get_embedding_model

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}{f' — {detail}' if detail else ''}")


def get_rule_ids(results) -> list[str]:
    return [obj.properties["rule_id"] + obj.properties["rule_type"] for obj in results.objects]


def hybrid(col, model, query: str, limit: int = 5, **kwargs):
    vec = model.encode(query, normalize_embeddings=True).tolist()
    return col.query.hybrid(query=query, vector=vec, alpha=0.5, limit=limit, **kwargs)


def main():
    print("Connecting...")
    client = get_weaviate_client(settings)
    model = get_embedding_model(settings)
    col = client.collections.get(settings.weaviate_collection)

    # --- 1. Count ---
    print("\n[Count]")
    count = col.aggregate.over_all(total_count=True).total_count
    print(f"  Collection has {count} objects")
    check("count > 8000", count > 8000, f"got {count}")
    check("count < 9000", count < 9000, f"got {count}, possible duplicates")

    # --- 2. Hybrid search: known rules ---
    print("\n[Hybrid search — known rules]")

    results = hybrid(col, model, "firm must act honestly fairly professionally")
    ids = get_rule_ids(results)
    check("COBS 2.1.1R in top 5 for 'act honestly fairly'", "COBS 2.1.1R" in ids, f"got {ids}")

    results = hybrid(col, model, "client money segregation requirements")
    ids = get_rule_ids(results)
    cass_hits = [r for r in ids if r.startswith("CASS")]
    check("CASS rules in top 5 for 'client money segregation'", len(cass_hits) >= 2, f"got {ids}")

    results = hybrid(col, model, "responsible lending mortgage affordability assessment")
    ids = get_rule_ids(results)
    mcob_hits = [r for r in ids if r.startswith("MCOB")]
    check("MCOB rules in top 5 for 'mortgage affordability'", len(mcob_hits) >= 1, f"got {ids}")

    results = hybrid(col, model, "insurance product information disclosure")
    ids = get_rule_ids(results)
    icobs_hits = [r for r in ids if r.startswith("ICOBS")]
    check("ICOBS rules in top 5 for 'insurance disclosure'", len(icobs_hits) >= 1, f"got {ids}")

    results = hybrid(col, model, "ESG climate-related financial disclosures")
    ids = get_rule_ids(results)
    esg_hits = [r for r in ids if r.startswith("ESG")]
    check("ESG rules in top 5 for 'climate-related disclosures'", len(esg_hits) >= 1, f"got {ids}")

    # --- 3. BM25 keyword search ---
    print("\n[BM25 keyword search]")

    results = col.query.bm25(query="best execution", limit=5)
    ids = get_rule_ids(results)
    check("BM25 'best execution' returns results", len(ids) > 0, "empty results")
    cobs_hits = [r for r in ids if r.startswith("COBS")]
    check("COBS rules for 'best execution'", len(cobs_hits) >= 1, f"got {ids}")

    results = col.query.bm25(query="complaints handling", limit=5)
    check("BM25 'complaints handling' returns results", len(results.objects) > 0)

    # --- 4. Sourcebook filter ---
    print("\n[Sourcebook filter]")

    results = hybrid(
        col, model, "disclosure requirements",
        filters=wvq.Filter.by_property("sourcebook").equal("COBS"),
    )
    sourcebooks = {obj.properties["sourcebook"] for obj in results.objects}
    check("Filter by COBS returns only COBS", sourcebooks == {"COBS"}, f"got {sourcebooks}")

    results = hybrid(
        col, model, "client assets",
        filters=wvq.Filter.by_property("sourcebook").equal("CASS"),
    )
    sourcebooks = {obj.properties["sourcebook"] for obj in results.objects}
    check("Filter by CASS returns only CASS", sourcebooks == {"CASS"}, f"got {sourcebooks}")

    # --- 5. All sourcebooks present ---
    print("\n[Sourcebook coverage]")
    expected_sourcebooks = {"BCOBS", "CASS", "CMCOB", "COBS", "ESG", "FPCOB", "ICOBS", "MAR", "MCOB", "PDCOB"}

    for sb in sorted(expected_sourcebooks):
        results = col.query.fetch_objects(
            filters=wvq.Filter.by_property("sourcebook").equal(sb),
            limit=1,
        )
        check(f"{sb} has objects", len(results.objects) > 0)

    # --- 6. Metadata fields populated ---
    print("\n[Metadata fields]")

    results = hybrid(col, model, "firm must act honestly", limit=1)
    if results.objects:
        props = results.objects[0].properties
        check("text populated", bool(props.get("text")))
        check("sourcebook populated", bool(props.get("sourcebook")))
        check("chapter populated", bool(props.get("chapter")))
        check("section populated", bool(props.get("section")))
        check("rule_id populated", bool(props.get("rule_id")))
        check("rule_type populated", bool(props.get("rule_type")))
        check("chunk_id populated", bool(props.get("chunk_id")))
        check("chapter_title populated", bool(props.get("chapter_title")))
        check("section_title populated", bool(props.get("section_title")))
        check("text starts with context header", props["text"].startswith("["), f"starts with: {props['text'][:30]}")
    else:
        check("got result for metadata check", False, "empty results")

    # --- 7. Context header format ---
    print("\n[Context headers]")

    results = hybrid(col, model, "conduct of business obligations", limit=3)
    for obj in results.objects:
        text = obj.properties["text"]
        has_header = text.startswith("[") and ">" in text.split("\n")[0]
        check(f"Header format on {obj.properties['chunk_id'][:30]}", has_header, f"text starts: {text[:60]}")

    # --- 8. Data quality ---
    print("\n[Data quality]")

    results = col.query.fetch_objects(limit=100)
    empty = [obj for obj in results.objects if not obj.properties["text"].strip()]
    check("No empty text fields (in sample of 100)", len(empty) == 0, f"{len(empty)} empty")

    short = [obj for obj in results.objects if len(obj.properties["text"]) < 50]
    check("No very short chunks <50 chars (in sample)", len(short) == 0, f"{len(short)} short")

    # --- Summary ---
    print(f"\n{'=' * 50}")
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("All checks passed!")
    print(f"{'=' * 50}")

    client.close()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
