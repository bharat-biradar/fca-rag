"""Unit tests for chunker: splitting, merging, headers, size limits."""

from src.internal.ingestion.parser import ParsedRule
from src.internal.ingestion.chunker import (
    _build_header,
    _split_sub_paragraphs,
    _chunk_rule,
    build_chunks,
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
)


def _make_rule(rule_id="TEST 1.1.1", rule_type="R", text="", **kwargs):
    defaults = dict(
        sourcebook=rule_id.split()[0] if " " in rule_id else "TEST",
        page=1, section_id="1.1", section_title="Test Section",
        chapter_id="1", chapter_title="Test Chapter",
        is_annex=False, is_table=False, is_deleted=False,
        defined_terms=[], cross_references=[],
    )
    defaults.update(kwargs)
    return ParsedRule(rule_id=rule_id, rule_type=rule_type, text=text, **defaults)


# --- Header ---

def test_header_full_context():
    rule = _make_rule("COBS 2.1.1", "R", chapter_id="2", chapter_title="Conduct",
                      section_id="2.1", section_title="Acting honestly")
    header = _build_header(rule)
    assert header == "[COBS > Chapter 2: Conduct > Section 2.1: Acting honestly > COBS 2.1.1R]"


def test_header_no_chapter():
    rule = _make_rule("COBS 2.1.1", "R", chapter_id="", section_id="", section_title="")
    header = _build_header(rule)
    assert header == "[COBS > COBS 2.1.1R]"


# --- Sub-paragraph splitting ---

def test_split_no_sub_paragraphs():
    preamble, subs = _split_sub_paragraphs("A firm must act honestly.")
    assert preamble == "A firm must act honestly."
    assert subs == []


def test_split_basic_sub_paragraphs():
    text = "Preamble text.\n(1) First point.\n(2) Second point."
    preamble, subs = _split_sub_paragraphs(text)
    assert preamble == "Preamble text."
    assert len(subs) == 2
    assert subs[0][0] == "(1)"
    assert "First point" in subs[0][1]


def test_split_nested_sub_paragraphs_stay_together():
    text = "Preamble.\n(1) First:\n  (a) sub-a\n  (b) sub-b\n(2) Second."
    preamble, subs = _split_sub_paragraphs(text)
    # (a) and (b) should stay inside (1), not become separate subs
    assert len(subs) == 2
    assert "(a) sub-a" in subs[0][1]


# --- Chunk rule ---

def test_standalone_rule_one_chunk():
    rule = _make_rule(text="A firm must act honestly, fairly and professionally.")
    chunks = _chunk_rule(rule)
    assert len(chunks) == 1
    assert "honestly" in chunks[0].text
    assert chunks[0].sub_paragraph == ""


def test_deleted_rule_no_chunks():
    rule = _make_rule(text="[deleted]", is_deleted=True)
    chunks = _chunk_rule(rule)
    assert len(chunks) == 0


def test_empty_text_no_chunks():
    rule = _make_rule(text="   ")
    chunks = _chunk_rule(rule)
    assert len(chunks) == 0


def test_table_kept_whole():
    rule = _make_rule(text="| Col1 | Col2 |\n| a | b |\n| c | d |", is_table=True)
    chunks = _chunk_rule(rule)
    assert len(chunks) == 1
    assert chunks[0].is_table is True


def test_no_chunk_exceeds_max():
    # Long standalone rule with sentence boundaries should be split
    rule = _make_rule(text="A firm must comply with this rule. " * 200)
    chunks = _chunk_rule(rule)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= MAX_CHUNK_CHARS


def test_context_header_on_all_chunks():
    rule = _make_rule("COBS 2.1.1", "R", text="Preamble.\n(1) First.\n(2) Second.\n(3) Third.")
    chunks = _chunk_rule(rule)
    for c in chunks:
        assert c.text.startswith("[")
        assert "COBS 2.1.1R" in c.text


def test_chunk_metadata_populated():
    rule = _make_rule("COBS 2.1.1", "R", text="A firm must act honestly.",
                      chapter_id="2", section_id="2.1")
    chunks = _chunk_rule(rule)
    c = chunks[0]
    assert c.sourcebook == "COBS"
    assert c.rule_id == "COBS 2.1.1"
    assert c.rule_type == "R"
    assert c.chapter == "2"
    assert c.section == "2.1"


def test_chunk_id_deterministic():
    rule = _make_rule("COBS 2.1.1", "R", text="A firm must act honestly.")
    c1 = _chunk_rule(rule)
    c2 = _chunk_rule(rule)
    assert c1[0].chunk_id == c2[0].chunk_id


# --- Build chunks ---

def test_build_chunks_skips_deleted():
    rules = {
        "TEST": [
            _make_rule(text="Active rule."),
            _make_rule(text="[deleted]", is_deleted=True),
        ]
    }
    chunks = build_chunks(rules)
    assert len(chunks) == 1
