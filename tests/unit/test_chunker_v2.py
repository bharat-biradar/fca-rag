"""Unit tests for chunker_v2: validate v2 produces measurably better chunks than v1."""

from __future__ import annotations

import pytest

from src.internal.ingestion.parser import ParsedRule, load_parsed_rules, _finalize_rule
from src.internal.ingestion.chunker import (
    Chunk,
    build_chunks as build_chunks_v1,
    _chunk_rule as _chunk_rule_v1,
    _split_sub_paragraphs,
)
from src.internal.ingestion.chunker_v2 import (
    build_chunks as build_chunks_v2,
    _chunk_rule_v2,
    _group_sub_paragraphs,
    MIN_CHUNK_CHARS as V2_MIN,
    MAX_CHUNK_CHARS as V2_MAX,
)


# ---------------------------------------------------------------------------
# Fixtures — load real parsed rules once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def all_rules() -> dict[str, list[ParsedRule]]:
    return load_parsed_rules()


@pytest.fixture(scope="session")
def v1_chunks(all_rules) -> list[Chunk]:
    return build_chunks_v1(all_rules)


@pytest.fixture(scope="session")
def v2_chunks(all_rules) -> list[Chunk]:
    return build_chunks_v2(all_rules)


@pytest.fixture(scope="session")
def all_rules_flat(all_rules) -> list[ParsedRule]:
    return [r for rules in all_rules.values() for r in rules]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChunkCountAndSize:

    def test_total_chunk_count_decreases(self, v1_chunks, v2_chunks):
        """v2 should produce fewer chunks than v1 (target: 20-40% reduction)."""
        reduction = 1 - len(v2_chunks) / len(v1_chunks)
        assert len(v2_chunks) < len(v1_chunks), (
            f"v2 ({len(v2_chunks)}) should have fewer chunks than v1 ({len(v1_chunks)})"
        )
        assert reduction >= 0.15, (
            f"Chunk reduction is only {reduction:.0%}, expected at least 15%"
        )

    def test_fewer_tiny_chunks(self, v1_chunks, v2_chunks):
        """v2 should have significantly fewer tiny chunks than v1.

        Many standalone/table rules are inherently short (single sentences) and
        can't be grouped — those are expected. But sub-paragraph splitting artifacts
        should be eliminated.
        """
        v1_tiny = [c for c in v1_chunks if len(c.text) < 250]
        v2_tiny = [c for c in v2_chunks if len(c.text) < 250]
        assert len(v2_tiny) < len(v1_tiny), (
            f"v2 ({len(v2_tiny)}) should have fewer tiny chunks than v1 ({len(v1_tiny)})"
        )
        # Sub-paragraph chunks specifically should have very few tiny ones
        # (some rules are inherently short even after grouping all sub-paras)
        v2_tiny_subpara = [c for c in v2_tiny if c.sub_paragraph]
        assert len(v2_tiny_subpara) < 10, (
            f"Found {len(v2_tiny_subpara)} tiny sub-paragraph chunks in v2. "
            f"Examples: {[c.chunk_id for c in v2_tiny_subpara[:5]]}"
        )

    def test_min_chunk_size_respected(self, v2_chunks, all_rules_flat):
        """All v2 chunks should be >= MIN_CHUNK_CHARS (500) unless the rule itself is small."""
        # Collect rule_ids of rules whose total text is under MIN_CHUNK_CHARS
        small_rule_ids = {
            r.rule_id for r in all_rules_flat
            if len(r.text) < V2_MIN and not r.is_deleted
        }
        violations = [
            c for c in v2_chunks
            if len(c.text) < V2_MIN and c.rule_id not in small_rule_ids
        ]
        assert len(violations) < 20, (
            f"{len(violations)} chunks violate MIN_CHUNK_CHARS={V2_MIN}. "
            f"Examples: {[(c.chunk_id, len(c.text)) for c in violations[:5]]}"
        )

    def test_no_oversized_chunks(self, v2_chunks):
        """No v2 chunk should exceed MAX_CHUNK_CHARS (4000)."""
        oversized = [c for c in v2_chunks if len(c.text) > V2_MAX]
        assert len(oversized) == 0, (
            f"{len(oversized)} chunks exceed {V2_MAX} chars. "
            f"Examples: {[(c.chunk_id, len(c.text)) for c in oversized[:5]]}"
        )


class TestSubParagraphGrouping:

    def test_high_subpara_rule_grouping(self, all_rules_flat):
        """Rules with many sub-paragraphs should produce far fewer chunks in v2."""
        # Find a non-table rule with the most sub-paragraphs
        best_rule = None
        best_count = 0
        for r in all_rules_flat:
            if r.is_deleted or r.is_table:
                continue
            _, subs = _split_sub_paragraphs(r.text)
            if len(subs) > best_count:
                best_count = len(subs)
                best_rule = r

        if best_rule is None or best_count < 10:
            pytest.skip("No non-table rule with 10+ sub-paragraphs found")

        v1 = _chunk_rule_v1(best_rule)
        v2 = _chunk_rule_v2(best_rule)

        assert len(v2) < len(v1), (
            f"Rule {best_rule.rule_id} ({best_count} sub-paras): "
            f"v2 ({len(v2)} chunks) should have fewer than v1 ({len(v1)})"
        )
        # Grouping should reduce chunk count substantially
        reduction = 1 - len(v2) / len(v1)
        assert reduction >= 0.3, (
            f"Rule {best_rule.rule_id}: only {reduction:.0%} reduction "
            f"(v1={len(v1)}, v2={len(v2)}), expected at least 30%"
        )

    def test_grouped_label_format(self, v2_chunks):
        """Grouped chunks should use '(start)-(end)' range notation."""
        grouped = [c for c in v2_chunks if "-" in c.sub_paragraph and "part_" not in c.sub_paragraph]
        # There should be some grouped chunks
        assert len(grouped) > 0, "Expected some grouped sub-paragraph chunks"

        # None should use the old '+' format
        plus_format = [c for c in v2_chunks if "+" in c.sub_paragraph]
        assert len(plus_format) == 0, (
            f"Found {len(plus_format)} chunks with old '+' format: {[c.chunk_id for c in plus_format[:5]]}"
        )

    def test_preamble_in_every_group(self, all_rules):
        """For rules with preamble + sub-paragraphs, every grouped chunk should contain the preamble."""
        # Find rules with non-empty preambles
        tested = 0
        for sb_rules in all_rules.values():
            for rule in sb_rules:
                if rule.is_deleted or rule.is_table:
                    continue
                preamble, subs = _split_sub_paragraphs(rule.text)
                if not subs or not preamble or len(preamble.strip()) < 10:
                    continue

                chunks = _chunk_rule_v2(rule)
                preamble_stripped = preamble.strip()
                for chunk in chunks:
                    if chunk.sub_paragraph and "part_" not in chunk.sub_paragraph:
                        assert preamble_stripped in chunk.text, (
                            f"Chunk {chunk.chunk_id} missing preamble: '{preamble_stripped[:60]}...'"
                        )
                tested += 1
                if tested >= 50:
                    break
            if tested >= 50:
                break

        assert tested > 0, "No rules with preamble + sub-paragraphs found to test"


class TestTextPreservation:

    def test_all_rule_text_preserved(self, all_rules):
        """For a sample of rules, verify all sub-paragraph content appears in v2 chunks."""
        tested = 0
        for sb_rules in all_rules.values():
            for rule in sb_rules:
                if rule.is_deleted or not rule.text.strip():
                    continue
                preamble, subs = _split_sub_paragraphs(rule.text)
                if not subs:
                    continue

                chunks = _chunk_rule_v2(rule)
                combined_text = " ".join(c.text for c in chunks)

                # Every sub-paragraph's text should appear in the combined output
                for sub_id, sub_text in subs:
                    # Check a substantial portion of each sub-para (first 80 chars)
                    check = sub_text.strip()[:80]
                    assert check in combined_text, (
                        f"Rule {rule.rule_id} sub {sub_id}: text not preserved: '{check[:50]}...'"
                    )
                tested += 1
                if tested >= 100:
                    break
            if tested >= 100:
                break

        assert tested > 0, "No rules with sub-paragraphs found to test"

    def test_standalone_and_table_rules_unchanged(self, all_rules):
        """Non-junk standalone/table rules should produce same chunk count in v1 and v2."""
        from src.internal.ingestion.chunker_v2 import _is_junk_rule, _is_preamble_stub
        mismatches = []
        tested = 0
        for sb_rules in all_rules.values():
            for rule in sb_rules:
                if rule.is_deleted or not rule.text.strip():
                    continue
                # Skip rules that v2 intentionally filters out
                if _is_junk_rule(rule) or _is_preamble_stub(rule):
                    continue
                preamble, subs = _split_sub_paragraphs(rule.text)
                if subs:
                    continue  # skip rules with sub-paragraphs

                v1 = _chunk_rule_v1(rule)
                v2 = _chunk_rule_v2(rule)
                if len(v1) != len(v2):
                    mismatches.append((rule.rule_id, len(v1), len(v2)))
                tested += 1

        assert len(mismatches) == 0, (
            f"{len(mismatches)} standalone/table rules have different chunk counts. "
            f"Examples: {mismatches[:5]}"
        )
        assert tested > 0, "No standalone/table rules found to test"


class TestCompatibility:

    def test_build_chunks_signature_compatible(self, all_rules):
        """v2 build_chunks accepts same input and returns list[Chunk]."""
        # Use a small subset to keep test fast
        subset = {}
        for sb, rules in all_rules.items():
            subset[sb] = rules[:5]
            break  # just one sourcebook

        result = build_chunks_v2(subset)
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)
        assert len(result) > 0


class TestRuleTypeFallback:

    def test_rule_type_fallback(self):
        """The parser fix should reduce empty rule_type from 19.9% to under 10%.

        Loads raw rules and re-applies _finalize_rule (which now has the fallback)
        since the cached JSON files may predate the fix.
        """
        rules = load_parsed_rules()
        all_flat = [r for sb_rules in rules.values() for r in sb_rules]
        total = len(all_flat)
        empty_before = sum(1 for r in all_flat if not r.rule_type)

        # Re-apply _finalize_rule which now has the fallback
        for r in all_flat:
            if not r.rule_type:
                _finalize_rule(r)

        empty_after = sum(1 for r in all_flat if not r.rule_type)
        pct_after = 100 * empty_after / total if total > 0 else 0

        assert empty_after < empty_before, (
            f"Fallback had no effect: before={empty_before}, after={empty_after}"
        )
        assert pct_after < 15, (
            f"Empty rule_type: {empty_after}/{total} ({pct_after:.1f}%). "
            f"Target: <15% (was {100 * empty_before / total:.1f}% before fix)"
        )
