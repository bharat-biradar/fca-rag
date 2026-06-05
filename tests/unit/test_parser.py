"""Unit tests for parser: regex matching, cross-references, defined terms."""

from src.config import RULE_ID_RE, XREF_RE
from src.internal.ingestion.parser import (
    _extract_cross_references,
    _extract_defined_terms,
)


# --- Rule ID regex ---

def test_rule_id_basic():
    m = RULE_ID_RE.search("COBS 2.1.1")
    assert m and m.group(1) == "COBS" and m.group(2) == "2.1.1"


def test_rule_id_alpha_suffix():
    m = RULE_ID_RE.search("COBS 4.12A.9B")
    assert m and m.group(2) == "4.12A.9B"


def test_rule_id_all_sourcebooks():
    for sb in ["BCOBS", "CASS", "CMCOB", "COBS", "ESG", "FPCOB", "ICOBS", "MAR", "MCOB", "PDCOB"]:
        m = RULE_ID_RE.search(f"{sb} 1.1.1")
        assert m and m.group(1) == sb


def test_rule_id_no_match_invalid():
    assert RULE_ID_RE.search("SYSC 10.1.7") is None  # not in our sourcebooks
    assert RULE_ID_RE.search("COBS 2.1") is None  # only 2 segments


# --- Cross-reference extraction ---

def test_xref_basic():
    text = "See COBS 2.1.1R for details"
    refs = _extract_cross_references(text)
    assert "COBS 2.1.1" in refs


def test_xref_strips_type_suffix():
    text = "as required by CASS 7.13.3R and BCOBS 4.1.2G"
    refs = _extract_cross_references(text)
    assert "CASS 7.13.3" in refs
    assert "BCOBS 4.1.2" in refs


def test_xref_preserves_alpha_suffix():
    text = "under COBS 4.12A.9B"
    refs = _extract_cross_references(text)
    assert "COBS 4.12A.9B" in refs


def test_xref_excludes_self():
    text = "COBS 2.1.1R requires firms to act honestly. See also COBS 2.1.2G."
    refs = _extract_cross_references(text, own_rule_id="COBS 2.1.1")
    assert "COBS 2.1.1" not in refs
    assert "COBS 2.1.2" in refs


def test_xref_no_duplicates():
    text = "COBS 2.1.1R is referenced in COBS 2.1.1R again"
    refs = _extract_cross_references(text)
    assert refs.count("COBS 2.1.1") == 1


def test_xref_multiple_sourcebooks():
    text = "See COBS 9.2.1R, ICOBS 6.1.5R and BCOBS 5.1.1G"
    refs = _extract_cross_references(text)
    assert len(refs) == 3
    assert "COBS 9.2.1" in refs
    assert "ICOBS 6.1.5" in refs
    assert "BCOBS 5.1.1" in refs


# --- Defined terms ---

def test_defined_terms_italic():
    text = "A *firm* must act in the *client's* best interests"
    terms = _extract_defined_terms(text)
    assert "firm" in terms
    assert "client's" in terms


def test_defined_terms_no_match():
    text = "No italic terms here"
    terms = _extract_defined_terms(text)
    assert terms == []
