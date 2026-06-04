"""Build Neo4j knowledge graph from parsed FCA regulatory rules.

Creates a hierarchy (Sourcebook -> Chapter -> Section -> Rule) with
cross-reference edges between rules. Stub nodes are created for
dangling references that point to rules not in our dataset.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.config import SOURCEBOOK_NAMES, Settings, settings
from src.dependencies import get_neo4j_driver
from src.internal.ingestion.parser import ParsedRule, load_parsed_rules

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 500

# Cypher: node creation
SOURCEBOOK_CYPHER = """
UNWIND $batch AS row
MERGE (s:Sourcebook {id: row.id})
SET s.full_name = row.full_name
"""

CHAPTER_CYPHER = """
UNWIND $batch AS row
MERGE (c:Chapter {id: row.id})
SET c.sourcebook = row.sourcebook,
    c.number = row.number,
    c.title = row.title
"""

SECTION_CYPHER = """
UNWIND $batch AS row
MERGE (s:Section {id: row.id})
SET s.sourcebook = row.sourcebook,
    s.number = row.number,
    s.title = row.title
"""

RULE_CYPHER = """
UNWIND $batch AS row
MERGE (r:Rule {id: row.id})
SET r.sourcebook = row.sourcebook,
    r.type = row.type,
    r.section = row.section,
    r.chapter = row.chapter,
    r.text_preview = row.text_preview,
    r.is_annex = row.is_annex,
    r.is_table = row.is_table,
    r.stub = row.stub
"""

# Cypher: edge creation
RULE_SECTION_CYPHER = """
UNWIND $batch AS row
MATCH (r:Rule {id: row.rule_id})
MATCH (s:Section {id: row.section_id})
MERGE (r)-[:BELONGS_TO]->(s)
"""

SECTION_CHAPTER_CYPHER = """
UNWIND $batch AS row
MATCH (s:Section {id: row.section_id})
MATCH (c:Chapter {id: row.chapter_id})
MERGE (s)-[:BELONGS_TO]->(c)
"""

CHAPTER_SOURCEBOOK_CYPHER = """
UNWIND $batch AS row
MATCH (c:Chapter {id: row.chapter_id})
MATCH (s:Sourcebook {id: row.sourcebook_id})
MERGE (c)-[:BELONGS_TO]->(s)
"""

ORPHAN_RULE_SOURCEBOOK_CYPHER = """
UNWIND $batch AS row
MATCH (r:Rule {id: row.rule_id})
MATCH (s:Sourcebook {id: row.sourcebook_id})
MERGE (r)-[:BELONGS_TO]->(s)
"""

REFERENCES_CYPHER = """
UNWIND $batch AS row
MATCH (a:Rule {id: row.from_id})
MATCH (b:Rule {id: row.to_id})
MERGE (a)-[:REFERENCES]->(b)
"""


# ---------------------------------------------------------------------------
# Collected entities (returned by _collect_entities)
# ---------------------------------------------------------------------------

@dataclass
class GraphEntities:
    sourcebooks: list[dict] = field(default_factory=list)
    chapters: list[dict] = field(default_factory=list)
    sections: list[dict] = field(default_factory=list)
    rules: list[dict] = field(default_factory=list)
    stubs: list[dict] = field(default_factory=list)
    rule_section_edges: list[dict] = field(default_factory=list)
    section_chapter_edges: list[dict] = field(default_factory=list)
    chapter_sb_edges: list[dict] = field(default_factory=list)
    orphan_rule_sb_edges: list[dict] = field(default_factory=list)
    ref_edges: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRAILING_TYPE_RE = re.compile(r"[RGDE]+$")


def _normalize_ref(ref: str, known_ids: set[str]) -> str:
    """Try to resolve a cross-reference to a known rule_id.

    Handles cases like 'BCOBS 2.3.7BR' where trailing 'R' is a type
    suffix, not part of the base ID.  Returns the original ref if no
    known ID matches (will become a stub node).
    """
    if ref in known_ids:
        return ref
    parts = ref.split(maxsplit=1)
    if len(parts) == 2:
        stripped = _TRAILING_TYPE_RE.sub("", parts[1])
        if stripped != parts[1]:
            candidate = f"{parts[0]} {stripped}"
            if candidate in known_ids:
                return candidate
    return ref


def _collect_entities(rules: dict[str, list[ParsedRule]]) -> GraphEntities:
    """Single pass over all rules — build every param list needed."""
    ent = GraphEntities()

    # --- Sourcebooks (from config, not from rules) ---
    for sb, full_name in sorted(SOURCEBOOK_NAMES.items()):
        ent.sourcebooks.append({"id": sb, "full_name": full_name})

    # --- Collect chapters, sections, rules, edges ---
    chapters: dict[str, dict] = {}
    sections: dict[str, dict] = {}
    all_rule_ids: set[str] = set()

    for sb, sb_rules in rules.items():
        for r in sb_rules:
            all_rule_ids.add(r.rule_id)

            # Chapter
            ch_num = r.chapter_id
            if not ch_num and r.section_id:
                ch_num = r.section_id.split(".")[0]
            if ch_num:
                ch_key = f"{sb}_{ch_num}"
                if ch_key not in chapters:
                    chapters[ch_key] = {
                        "id": ch_key,
                        "sourcebook": sb,
                        "number": ch_num,
                        "title": r.chapter_title or "",
                    }
                elif r.chapter_title and not chapters[ch_key]["title"]:
                    chapters[ch_key]["title"] = r.chapter_title

            # Section
            if r.section_id:
                sec_key = f"{sb}_{r.section_id}"
                if sec_key not in sections:
                    sections[sec_key] = {
                        "id": sec_key,
                        "sourcebook": sb,
                        "number": r.section_id,
                        "title": r.section_title or "",
                    }
                elif r.section_title and not sections[sec_key]["title"]:
                    sections[sec_key]["title"] = r.section_title

            # Rule node params
            ent.rules.append({
                "id": r.rule_id,
                "sourcebook": sb,
                "type": r.rule_type,
                "section": r.section_id,
                "chapter": r.chapter_id,
                "text_preview": r.text[:200],
                "is_annex": r.is_annex,
                "is_table": r.is_table,
                "stub": False,
            })

            # Rule -> Section edge
            if r.section_id:
                ent.rule_section_edges.append({
                    "rule_id": r.rule_id,
                    "section_id": f"{sb}_{r.section_id}",
                })
            elif ch_num:
                # Has chapter but no section — skip (will attach via orphan if needed)
                pass
            else:
                # No section, no chapter — orphan -> Sourcebook
                ent.orphan_rule_sb_edges.append({
                    "rule_id": r.rule_id,
                    "sourcebook_id": sb,
                })

    # Section -> Chapter edges
    for sec_key, sec in sections.items():
        ch_num = sec["number"].split(".")[0]
        ch_key = f"{sec['sourcebook']}_{ch_num}"
        if ch_key in chapters:
            ent.section_chapter_edges.append({
                "section_id": sec_key,
                "chapter_id": ch_key,
            })

    # Chapter -> Sourcebook edges
    for ch_key, ch in chapters.items():
        ent.chapter_sb_edges.append({
            "chapter_id": ch_key,
            "sourcebook_id": ch["sourcebook"],
        })

    ent.chapters = list(chapters.values())
    ent.sections = list(sections.values())

    # --- Cross-reference edges + stub detection ---
    all_refs: set[str] = set()
    raw_ref_edges: list[tuple[str, str]] = []

    for sb_rules in rules.values():
        for r in sb_rules:
            for ref in r.cross_references:
                target = _normalize_ref(ref, all_rule_ids)
                raw_ref_edges.append((r.rule_id, target))
                all_refs.add(target)

    # Deduplicate reference edges
    seen_edges: set[tuple[str, str]] = set()
    for from_id, to_id in raw_ref_edges:
        if from_id != to_id and (from_id, to_id) not in seen_edges:
            seen_edges.add((from_id, to_id))
            ent.ref_edges.append({"from_id": from_id, "to_id": to_id})

    # Stub nodes for dangling references
    dangling = all_refs - all_rule_ids
    for ref in sorted(dangling):
        parts = ref.split(maxsplit=1)
        sb = parts[0] if len(parts) >= 2 else ""
        ent.stubs.append({
            "id": ref,
            "sourcebook": sb,
            "type": "",
            "section": "",
            "chapter": "",
            "text_preview": "",
            "is_annex": False,
            "is_table": False,
            "stub": True,
        })

    return ent


def _batched_write(driver, cypher: str, params: list[dict], label: str):
    """Execute a Cypher UNWIND statement in batches."""
    total = len(params)
    if total == 0:
        print(f"  {label}: 0 items (skipped)")
        return
    for i in range(0, total, BATCH_SIZE):
        batch = params[i: i + BATCH_SIZE]
        driver.execute_query(cypher, batch=batch)
    print(f"  {label}: {total}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_indexes(driver):
    """Create uniqueness constraints (implicitly indexes) on node ID fields."""
    constraints = [
        "CREATE CONSTRAINT sourcebook_id IF NOT EXISTS FOR (s:Sourcebook) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT chapter_id IF NOT EXISTS FOR (c:Chapter) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT rule_id IF NOT EXISTS FOR (r:Rule) REQUIRE r.id IS UNIQUE",
    ]
    for stmt in constraints:
        driver.execute_query(stmt)
    print(f"  {len(constraints)} constraints ensured")


def clear_graph(driver):
    """Remove all nodes and relationships."""
    driver.execute_query("MATCH (n) DETACH DELETE n")
    print("  Graph cleared")


def validate(driver):
    """Print graph stats and run spot-checks."""
    # Node counts
    records, _, _ = driver.execute_query(
        "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY label"
    )
    print("  Node counts:")
    for rec in records:
        print(f"    {rec['label']}: {rec['cnt']}")

    # Edge counts
    records, _, _ = driver.execute_query(
        "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt ORDER BY rel"
    )
    print("  Edge counts:")
    for rec in records:
        print(f"    {rec['rel']}: {rec['cnt']}")

    # Stub count
    records, _, _ = driver.execute_query(
        "MATCH (r:Rule {stub: true}) RETURN count(r) AS stubs"
    )
    print(f"  Stub rules: {records[0]['stubs']}")

    # Orphan check (non-stub rules without BELONGS_TO)
    records, _, _ = driver.execute_query(
        "MATCH (r:Rule) WHERE NOT (r)-[:BELONGS_TO]->() AND r.stub = false "
        "RETURN count(r) AS orphans"
    )
    print(f"  Orphan rules (non-stub, no BELONGS_TO): {records[0]['orphans']}")


def run_graph_ingestion(
    rules: dict[str, list[ParsedRule]],
    cfg: Settings = settings,
    clear: bool = True,
):
    """Full graph construction pipeline."""
    driver = get_neo4j_driver(cfg)

    print("Creating indexes/constraints...")
    create_indexes(driver)

    if clear:
        print("Clearing existing graph...")
        clear_graph(driver)

    print("Collecting entities...")
    ent = _collect_entities(rules)
    total_rules = len(ent.rules)
    print(
        f"  {len(ent.sourcebooks)} sourcebooks, {len(ent.chapters)} chapters, "
        f"{len(ent.sections)} sections, {total_rules} rules, "
        f"{len(ent.stubs)} stubs, {len(ent.ref_edges)} references"
    )

    print("Ingesting nodes...")
    _batched_write(driver, SOURCEBOOK_CYPHER, ent.sourcebooks, "Sourcebooks")
    _batched_write(driver, CHAPTER_CYPHER, ent.chapters, "Chapters")
    _batched_write(driver, SECTION_CYPHER, ent.sections, "Sections")
    _batched_write(driver, RULE_CYPHER, ent.rules, "Rules")
    _batched_write(driver, RULE_CYPHER, ent.stubs, "Stubs")

    print("Ingesting edges...")
    _batched_write(driver, RULE_SECTION_CYPHER, ent.rule_section_edges, "Rule->Section")
    _batched_write(driver, SECTION_CHAPTER_CYPHER, ent.section_chapter_edges, "Section->Chapter")
    _batched_write(driver, CHAPTER_SOURCEBOOK_CYPHER, ent.chapter_sb_edges, "Chapter->Sourcebook")
    _batched_write(driver, ORPHAN_RULE_SOURCEBOOK_CYPHER, ent.orphan_rule_sb_edges, "OrphanRule->Sourcebook")
    _batched_write(driver, REFERENCES_CYPHER, ent.ref_edges, "References")

    print("Validating...")
    validate(driver)

    print("Done.")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading parsed rules...")
    rules = load_parsed_rules()
    total = sum(len(r) for r in rules.values())
    print(f"Loaded {total} rules from {len(rules)} sourcebooks\n")

    print("Running Neo4j graph ingestion...")
    run_graph_ingestion(rules)
