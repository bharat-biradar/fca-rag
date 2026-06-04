# Ingestion Pipeline — Rough Plan

## Coding Guidelines

- **SRP**: One function = one job. But don't create abstractions for one-time operations.
- **Compact**: No boilerplate, no over-engineering. This is a take-home, not a framework.
- **Independently runnable**: Each module has `if __name__ == "__main__"` so you can run `python -m src.internal.ingestion.parser` directly to test just that stage.
- **Flat data**: Use simple dataclasses, not deep inheritance. Avoid classes where a function will do.
- **No unnecessary comments**: Code should be self-explanatory. Comment only non-obvious logic.
- **Config from env**: All secrets from env vars, all tunable params in config.py with sensible defaults.

## Stages

### Stage 0: config.py + dependencies.py
- Config dataclass with all params (paths, model names, DB URLs, chunking thresholds)
- Lazy singleton getters for Weaviate, Neo4j, embedding model

### Stage 1: parser.py
- Input: LlamaParse JSON files
- Output: `dict[str, list[ParsedRule]]`
- Handles 3 rule formats: table rows, heading rules, inline text rules
- Cross-page merging: orphaned text appended to last rule
- Extracts: rule_id, type, text, page, section, chapter, cross_references, defined_terms
- Skips: deleted rules, headers/footers, noise

### Stage 2: chunker.py
- Input: parsed rules from Stage 1
- Output: `list[Chunk]` with all metadata
- Parent-child split at level-1 sub-paragraphs (1), (2), (3)
- Standalone rules = both parent and child
- Merge short children (<50 tokens), split long parents (>1500 tokens)
- Chunk IDs deterministic: `CMCOB_2.1.1R` or `CMCOB_2.1.1R_(1)`

### Stage 3: embedder.py
- Input: chunks from Stage 2
- Output: chunks stored in Weaviate with BGE-M3 vectors
- Collection schema: text (BM25+vector), parent_text, all metadata fields
- Filter properties with FIELD tokenization (exact match)
- Batch embed (32 at a time), deterministic UUIDs for idempotent upserts

### Stage 4: graph_builder.py
- Input: parsed rules from Stage 1 (not chunks)
- Output: Neo4j graph with nodes (Sourcebook→Chapter→Section→Rule) and REFERENCES edges
- Cross-ref regex extracts rule IDs from text
- MERGE for idempotency, UNWIND for batch performance
- Indexes on all node ID fields

### Stage 5: Orchestrator (__init__.py)
- `run_ingestion()` calls stages 1→2→3→4 sequentially
- Also runnable as: `python -m src.internal.ingestion`

## Data flow

```
JSON files → parser.py → {sourcebook: [ParsedRule]} → chunker.py → [Chunk] → embedder.py → Weaviate
                                                    ↘ graph_builder.py → Neo4j
```

## What gets stored where

### Weaviate (per child chunk)
| Field | Type | Searchable? | Purpose |
|---|---|---|---|
| text | TEXT | BM25 + vector | The chunk text, embedded by BGE-M3 |
| parent_text | TEXT | No | Full parent rule for LLM context |
| sourcebook | TEXT (FIELD) | Filter only | Exact match filter: "COBS" |
| chapter | TEXT (FIELD) | Filter only | "2" |
| chapter_title | TEXT | No | "Conduct of business obligations" |
| section | TEXT (FIELD) | Filter only | "2.1" |
| section_title | TEXT | No | "Acting honestly, fairly..." |
| rule_id | TEXT (FIELD) | Filter only | "COBS 2.1.1R" |
| rule_type | TEXT (FIELD) | Filter only | "R" |
| page | INT | No | 45 |
| chunk_id | TEXT (FIELD) | Filter only | "COBS_2.1.1R_(1)" |
| sub_paragraph | TEXT | No | "(1)" |
| is_annex | BOOL | No | false |
| is_table | BOOL | No | false |
| defined_terms | TEXT_ARRAY | No | ["firm", "customer"] |
| cross_references | TEXT_ARRAY | No | ["COBS 2.2", "CASS 5.5.14R"] |

### Neo4j (graph)
**Nodes:**
- `(:Sourcebook {id: "COBS", full_name: "Conduct of Business Sourcebook"})`
- `(:Chapter {id: "COBS_2", sourcebook: "COBS", title: "..."})`
- `(:Section {id: "COBS_2.1", sourcebook: "COBS", title: "..."})`
- `(:Rule {id: "COBS 2.1.1R", sourcebook: "COBS", type: "R", section: "2.1", text_preview: "first 200 chars"})`

**Edges:**
- `(:Rule)-[:REFERENCES]->(:Rule)` — from cross-references in rule text
- `(:Rule)-[:BELONGS_TO]->(:Section)`
- `(:Section)-[:BELONGS_TO]->(:Chapter)`
- `(:Chapter)-[:BELONGS_TO]->(:Sourcebook)`

## Verify with
1. Parse CMCOB.json → check rule count (~96), spot-check a rule's text
2. Chunk → verify parent/child for a multi-sub-paragraph rule
3. Weaviate → hybrid search "firm must act honestly" → expect CMCOB 2.1.1R
4. Neo4j → `MATCH (r:Rule {id:'CMCOB 2.1.8G'})-[:REFERENCES]->(t) RETURN t.id` → expect CMCOB 2.1.7R
