"""Embed chunks with BGE-M3 and store in Weaviate.

Weaviate handles both dense vector search and BM25 keyword search natively.
We provide our own vectors (vectorizer=none) and let Weaviate build the BM25 index.
"""

from __future__ import annotations

import time
from dataclasses import asdict

import weaviate
import weaviate.classes.config as wvc
from weaviate.util import generate_uuid5

from src.config import Settings, settings
from src.dependencies import get_weaviate_client, get_embedding_model
from src.internal.ingestion.chunker import Chunk


def create_collection(client: weaviate.WeaviateClient, cfg: Settings = settings):
    """Create the Weaviate collection with schema. Idempotent — deletes if exists."""
    name = cfg.weaviate_collection

    if client.collections.exists(name):
        client.collections.delete(name)
        print(f"  Deleted existing collection '{name}'")

    client.collections.create(
        name=name,
        vectorizer_config=wvc.Configure.Vectorizer.none(),
        properties=[
            # Searchable text — BM25 indexed automatically
            wvc.Property(name="text", data_type=wvc.DataType.TEXT),
            # Filter fields — exact match, no tokenization
            wvc.Property(name="sourcebook", data_type=wvc.DataType.TEXT,
                         tokenization=wvc.Tokenization.FIELD),
            wvc.Property(name="chapter", data_type=wvc.DataType.TEXT,
                         tokenization=wvc.Tokenization.FIELD),
            wvc.Property(name="section", data_type=wvc.DataType.TEXT,
                         tokenization=wvc.Tokenization.FIELD),
            wvc.Property(name="rule_id", data_type=wvc.DataType.TEXT,
                         tokenization=wvc.Tokenization.FIELD),
            wvc.Property(name="rule_type", data_type=wvc.DataType.TEXT,
                         tokenization=wvc.Tokenization.FIELD),
            wvc.Property(name="chunk_id", data_type=wvc.DataType.TEXT,
                         tokenization=wvc.Tokenization.FIELD),
            # Metadata — stored but not searched
            wvc.Property(name="sourcebook_full", data_type=wvc.DataType.TEXT,
                         skip_vectorization=True),
            wvc.Property(name="chapter_title", data_type=wvc.DataType.TEXT,
                         skip_vectorization=True),
            wvc.Property(name="section_title", data_type=wvc.DataType.TEXT,
                         skip_vectorization=True),
            wvc.Property(name="sub_paragraph", data_type=wvc.DataType.TEXT,
                         skip_vectorization=True),
            wvc.Property(name="page", data_type=wvc.DataType.INT),
            wvc.Property(name="is_annex", data_type=wvc.DataType.BOOL),
            wvc.Property(name="is_table", data_type=wvc.DataType.BOOL),
            wvc.Property(name="defined_terms", data_type=wvc.DataType.TEXT_ARRAY,
                         skip_vectorization=True),
            wvc.Property(name="cross_references", data_type=wvc.DataType.TEXT_ARRAY,
                         skip_vectorization=True),
        ],
    )
    print(f"  Created collection '{name}'")


def embed_and_store(chunks: list[Chunk], cfg: Settings = settings):
    """Embed all chunks with BGE-M3 and batch-insert into Weaviate."""
    client = get_weaviate_client(cfg)
    model = get_embedding_model(cfg)
    collection = client.collections.get(cfg.weaviate_collection)

    total = len(chunks)
    batch_size = cfg.embedding_batch_size
    inserted = 0
    start = time.time()

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]

        # Embed
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        # Batch insert
        with collection.batch.fixed_size(batch_size=batch_size) as batch_writer:
            for chunk, vector in zip(batch, vectors):
                props = {
                    "text": chunk.text,
                    "sourcebook": chunk.sourcebook,
                    "sourcebook_full": chunk.sourcebook_full,
                    "chapter": chunk.chapter,
                    "chapter_title": chunk.chapter_title,
                    "section": chunk.section,
                    "section_title": chunk.section_title,
                    "rule_id": chunk.rule_id,
                    "rule_type": chunk.rule_type,
                    "chunk_id": chunk.chunk_id,
                    "sub_paragraph": chunk.sub_paragraph,
                    "page": chunk.page,
                    "is_annex": chunk.is_annex,
                    "is_table": chunk.is_table,
                    "defined_terms": chunk.defined_terms,
                    "cross_references": chunk.cross_references,
                }
                batch_writer.add_object(
                    properties=props,
                    vector=vector.tolist(),
                    uuid=generate_uuid5(chunk.chunk_id),
                )

        inserted += len(batch)
        elapsed = time.time() - start
        rate = inserted / elapsed if elapsed > 0 else 0
        eta = (total - inserted) / rate if rate > 0 else 0
        print(f"\r  Embedded {inserted}/{total} ({100*inserted//total}%) — {rate:.0f} chunks/s — ETA {eta:.0f}s", end="", flush=True)

    print(f"\n  Done: {inserted} chunks in {time.time()-start:.1f}s")


def validate(cfg: Settings = settings):
    """Quick validation: count + spot-check hybrid search."""
    client = get_weaviate_client(cfg)
    model = get_embedding_model(cfg)
    collection = client.collections.get(cfg.weaviate_collection)

    count = collection.aggregate.over_all(total_count=True).total_count
    print(f"  Collection count: {count}")

    # Spot-check: hybrid search (must provide our own query vector since vectorizer=none)
    query_text = "firm must act honestly fairly professionally"
    query_vec = model.encode(query_text, normalize_embeddings=True).tolist()

    results = collection.query.hybrid(
        query=query_text,
        vector=query_vec,
        alpha=0.5,
        limit=3,
    )
    print(f"  Hybrid search test (top 3):")
    for obj in results.objects:
        props = obj.properties
        print(f"    {props['rule_id']}{props['rule_type']} — {props['text'][:80]}...")


def run_ingestion(chunks: list[Chunk], cfg: Settings = settings):
    """Full embedding + storage pipeline."""
    client = get_weaviate_client(cfg)

    print("Creating Weaviate collection...")
    create_collection(client, cfg)

    print("Embedding and storing chunks...")
    embed_and_store(chunks, cfg)

    print("Validating...")
    validate(cfg)


# --- Runnable standalone ---

if __name__ == "__main__":
    from src.internal.ingestion.parser import load_parsed_rules
    from src.internal.ingestion.chunker import build_chunks

    print("Loading parsed rules...")
    rules = load_parsed_rules()

    print("Building chunks...")
    chunks = build_chunks(rules)
    print(f"Total: {len(chunks)} chunks\n")

    print("Running Weaviate ingestion...")
    run_ingestion(chunks)
