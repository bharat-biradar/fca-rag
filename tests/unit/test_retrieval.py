"""Unit tests for retrieval logic — tests actual extracted functions, not copies."""

import json

from src.internal.retrieval.agentic_v3 import (
    clean_rule_id,
    deduplicate_chunks,
    inject_lookups,
    parse_plan,
    parse_self_eval,
)
from src.internal.retrieval.base import RetrievedChunk


def _make_chunk(chunk_id="c1", rule_id="COBS 2.1.1", rule_type="R", score=0.5, **kw):
    defaults = dict(
        text="test text", sourcebook="COBS", chapter="2",
        chapter_title="", section="2.1", section_title="",
        sub_paragraph="", page=1, is_annex=False, is_table=False,
    )
    defaults.update(kw)
    return RetrievedChunk(chunk_id=chunk_id, rule_id=rule_id, rule_type=rule_type, score=score, **defaults)


# --- clean_rule_id ---

def test_clean_strips_R():
    assert clean_rule_id("COBS 2.1.1R") == "COBS 2.1.1"


def test_clean_strips_EU():
    assert clean_rule_id("MAR 1.3.2EU") == "MAR 1.3.2"


def test_clean_preserves_alpha_suffix():
    assert clean_rule_id("COBS 4.12A.9B") == "COBS 4.12A.9B"


def test_clean_handles_whitespace():
    assert clean_rule_id("  COBS 2.1.1R  ") == "COBS 2.1.1"


# --- parse_plan ---

def test_plan_caps_sub_queries():
    raw = json.dumps({"sub_queries": ["a", "b", "c", "d", "e"]})
    plan = parse_plan(raw, "original")
    assert len(plan["sub_queries"]) == 3


def test_plan_caps_rule_ids():
    raw = json.dumps({"rule_ids": [f"COBS {i}.1.1" for i in range(15)]})
    plan = parse_plan(raw, "original")
    assert len(plan["rule_ids"]) == 10


def test_plan_defaults_missing_fields():
    raw = json.dumps({"sub_queries": ["x"]})
    plan = parse_plan(raw, "my query")
    assert plan["reformulated_query"] == "my query"
    assert plan["rule_ids"] == []
    assert plan["chunk_budget"] == 5


def test_plan_chunk_budget_respected():
    raw = json.dumps({"sub_queries": ["x"], "chunk_budget": 8})
    plan = parse_plan(raw, "q")
    assert plan["chunk_budget"] == 8


def test_plan_chunk_budget_clamped():
    raw = json.dumps({"sub_queries": ["x"], "chunk_budget": 20})
    plan = parse_plan(raw, "q")
    assert plan["chunk_budget"] == 10

    raw = json.dumps({"sub_queries": ["x"], "chunk_budget": 2})
    plan = parse_plan(raw, "q")
    assert plan["chunk_budget"] == 5


def test_plan_strips_markdown_fences():
    raw = '```json\n{"sub_queries": ["a"]}\n```'
    plan = parse_plan(raw, "original")
    assert plan["sub_queries"] == ["a"]


def test_plan_strips_json_fence():
    raw = '```\n{"sub_queries": ["b"]}\n```'
    plan = parse_plan(raw, "original")
    assert plan["sub_queries"] == ["b"]


# --- deduplicate_chunks ---

def test_dedup_removes_duplicates():
    c1 = _make_chunk(chunk_id="abc", score=0.9)
    c2 = _make_chunk(chunk_id="abc", score=0.5)
    c3 = _make_chunk(chunk_id="def", score=0.8)
    result = deduplicate_chunks([c1, c2, c3])
    assert len(result) == 2


def test_dedup_keeps_first_seen():
    c1 = _make_chunk(chunk_id="abc", score=0.9)
    c2 = _make_chunk(chunk_id="abc", score=0.5)
    result = deduplicate_chunks([c1, c2])
    assert result[0].score == 0.9


def test_dedup_empty_list():
    assert deduplicate_chunks([]) == []


# --- inject_lookups ---

def test_inject_adds_missing_lookup():
    reranked = [_make_chunk(f"c{i}", score=0.9 - i * 0.1) for i in range(5)]
    lookup = _make_chunk("lookup1", rule_id="CASS 7.11.34", score=1.0)
    result = inject_lookups(reranked, [lookup], top_k=5)
    assert len(result) == 5
    assert any(c.chunk_id == "lookup1" for c in result)


def test_inject_doesnt_exceed_top_k():
    reranked = [_make_chunk(f"c{i}", score=0.5) for i in range(5)]
    lookups = [_make_chunk(f"l{i}", score=1.0) for i in range(3)]
    result = inject_lookups(reranked, lookups, top_k=5)
    assert len(result) == 5


def test_inject_skips_already_present():
    reranked = [_make_chunk("c1", score=0.9)]
    lookup = _make_chunk("c1", score=1.0)  # same chunk_id
    result = inject_lookups(reranked, [lookup], top_k=5)
    assert len(result) == 1  # no duplicate added


def test_inject_multiple_lookups():
    reranked = [_make_chunk(f"c{i}", score=0.5) for i in range(5)]
    lookups = [_make_chunk(f"l{i}", score=1.0) for i in range(4)]
    result = inject_lookups(reranked, lookups, top_k=5)
    assert len(result) == 5
    lookup_ids = {c.chunk_id for c in result if c.chunk_id.startswith("l")}
    assert len(lookup_ids) == 4  # all 4 lookups present


# --- parse_self_eval ---

def test_self_eval_basic():
    assert parse_self_eval("YES\nNO\nYES\nYES\nNO") == 3


def test_self_eval_noisy_output():
    assert parse_self_eval("YES - relevant chunk\nNO - off topic\nYES\nYES\nNO") == 3


def test_self_eval_empty():
    assert parse_self_eval("") == 0


def test_self_eval_all_yes():
    assert parse_self_eval("YES\nYES\nYES\nYES\nYES") == 5


def test_self_eval_all_no():
    assert parse_self_eval("NO\nNO\nNO\nNO\nNO") == 0


def test_self_eval_case_insensitive():
    assert parse_self_eval("yes\nNo\nYES\nno\nyes") == 3


# --- display_id ---

def test_display_id():
    c = _make_chunk(rule_id="COBS 2.1.1", rule_type="R")
    assert c.display_id == "COBS 2.1.1R"


def test_display_id_no_type():
    c = _make_chunk(rule_id="COBS 2.1.1", rule_type="")
    assert c.display_id == "COBS 2.1.1"
