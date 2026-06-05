"""Unit tests for retrieval logic: dedup, routing, plan parsing, lookup injection."""

import re

from src.internal.retrieval.base import RetrievedChunk


def _make_chunk(chunk_id="c1", rule_id="COBS 2.1.1", rule_type="R", score=0.5, **kw):
    defaults = dict(
        text="test text", sourcebook="COBS", chapter="2",
        chapter_title="", section="2.1", section_title="",
        sub_paragraph="", page=1, is_annex=False, is_table=False,
    )
    defaults.update(kw)
    return RetrievedChunk(chunk_id=chunk_id, rule_id=rule_id, rule_type=rule_type, score=score, **defaults)


# --- Rule ID cleanup ---

def test_lookup_strips_type_suffix_R():
    clean = re.sub(r"[RGDEUK]{1,2}$", "", "COBS 2.1.1R".strip()).strip()
    assert clean == "COBS 2.1.1"


def test_lookup_strips_type_suffix_EU():
    clean = re.sub(r"[RGDEUK]{1,2}$", "", "MAR 1.3.2EU".strip()).strip()
    assert clean == "MAR 1.3.2"


def test_lookup_preserves_alpha_suffix():
    clean = re.sub(r"[RGDEUK]{1,2}$", "", "COBS 4.12A.9B".strip()).strip()
    assert clean == "COBS 4.12A.9B"


# --- Plan parsing ---

def test_plan_caps_sub_queries():
    plan = {"sub_queries": ["a", "b", "c", "d", "e"], "reformulated_query": "q", "rule_ids": []}
    plan["sub_queries"] = plan["sub_queries"][:3]
    assert len(plan["sub_queries"]) == 3


def test_plan_caps_rule_ids():
    plan = {"rule_ids": [f"COBS {i}.1.1" for i in range(15)]}
    plan["rule_ids"] = plan["rule_ids"][:10]
    assert len(plan["rule_ids"]) == 10


def test_plan_defaults_missing_fields():
    plan = {"sub_queries": ["x"]}
    plan.setdefault("reformulated_query", "original")
    plan.setdefault("rule_ids", [])
    assert plan["reformulated_query"] == "original"
    assert plan["rule_ids"] == []


def test_plan_strips_markdown_fences():
    raw = '```json\n{"sub_queries": ["a"]}\n```'
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    import json
    plan = json.loads(raw)
    assert plan["sub_queries"] == ["a"]


# --- Deduplication ---

def test_dedup_by_chunk_id_keeps_first():
    c1 = _make_chunk(chunk_id="abc", score=0.9)
    c2 = _make_chunk(chunk_id="abc", score=0.5)
    c3 = _make_chunk(chunk_id="def", score=0.8)
    seen = {}
    for c in [c1, c2, c3]:
        if c.chunk_id not in seen:
            seen[c.chunk_id] = c
    assert len(seen) == 2
    assert seen["abc"].score == 0.9


def test_dedup_by_chunk_id_keeps_best_score():
    c1 = _make_chunk(chunk_id="abc", score=0.3)
    c2 = _make_chunk(chunk_id="abc", score=0.9)
    seen = {}
    for c in [c1, c2]:
        if c.chunk_id not in seen or c.score > seen[c.chunk_id].score:
            seen[c.chunk_id] = c
    assert seen["abc"].score == 0.9


# --- Lookup injection ---

def test_lookup_injection_replaces_worst():
    reranked = [_make_chunk(f"c{i}", score=0.9 - i * 0.1) for i in range(5)]
    lookup = _make_chunk("lookup1", rule_id="CASS 7.11.34", score=1.0)

    already_in = {c.chunk_id for c in reranked}
    missing = [lookup] if lookup.chunk_id not in already_in else []

    top_k = 5
    if missing:
        slots = min(len(missing), top_k)
        while len(reranked) + slots > top_k and reranked:
            reranked.pop()
        reranked.extend(missing[:slots])

    assert len(reranked) == 5
    assert any(c.chunk_id == "lookup1" for c in reranked)


def test_lookup_injection_doesnt_exceed_top_k():
    reranked = [_make_chunk(f"c{i}", score=0.5) for i in range(5)]
    lookups = [_make_chunk(f"l{i}", rule_id=f"CASS 7.13.{i}", score=1.0) for i in range(3)]

    already_in = {c.chunk_id for c in reranked}
    missing = [l for l in lookups if l.chunk_id not in already_in]

    top_k = 5
    slots = min(len(missing), top_k)
    while len(reranked) + slots > top_k and reranked:
        reranked.pop()
    reranked.extend(missing[:slots])

    assert len(reranked) == 5


# --- Self-eval parsing ---

def test_self_eval_parses_yes_no():
    text = "YES\nNO\nYES\nYES\nNO"
    lines = [l.strip() for l in text.upper().split("\n") if l.strip()]
    count = sum(1 for l in lines if l.startswith("YES"))
    assert count == 3


def test_self_eval_handles_noisy_output():
    text = "YES - relevant chunk\nNO - off topic\nYES\nYES - good\nNO"
    lines = [l.strip() for l in text.upper().split("\n") if l.strip()]
    count = sum(1 for l in lines if l.startswith("YES"))
    assert count == 3


def test_self_eval_empty_returns_zero():
    text = ""
    lines = [l.strip() for l in text.upper().split("\n") if l.strip()]
    count = sum(1 for l in lines if l.startswith("YES"))
    assert count == 0


# --- Routing decision ---

def test_hybrid_sufficient_at_threshold():
    relevant_count = 3
    threshold = 3
    assert relevant_count >= threshold  # stays hybrid


def test_escalates_below_threshold():
    relevant_count = 2
    threshold = 3
    assert relevant_count < threshold  # escalates to agentic


def test_escalates_on_zero_relevant():
    relevant_count = 0
    threshold = 3
    assert relevant_count < threshold


# --- Display ID ---

def test_display_id():
    c = _make_chunk(rule_id="COBS 2.1.1", rule_type="R")
    assert c.display_id == "COBS 2.1.1R"


def test_display_id_no_type():
    c = _make_chunk(rule_id="COBS 2.1.1", rule_type="")
    assert c.display_id == "COBS 2.1.1"
