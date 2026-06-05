"""Shared data structures and base class for all retrieval approaches."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    text: str
    rule_id: str
    rule_type: str
    score: float
    sourcebook: str
    chapter: str
    chapter_title: str
    section: str
    section_title: str
    chunk_id: str
    sub_paragraph: str
    page: int
    is_annex: bool
    is_table: bool
    defined_terms: list[str] = field(default_factory=list)
    cross_references: list[str] = field(default_factory=list)

    @property
    def display_id(self) -> str:
        return f"{self.rule_id}{self.rule_type}"


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    retrieval_time_ms: float
    approach: str
    planning_tokens: int = 0


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        sourcebook_filter: str | None = None,
    ) -> RetrievalResult:
        ...


def weaviate_obj_to_chunk(obj, score: float = 0.0) -> RetrievedChunk:
    """Convert a Weaviate response object to a RetrievedChunk."""
    p = obj.properties
    return RetrievedChunk(
        text=p.get("text", ""),
        rule_id=p.get("rule_id", ""),
        rule_type=p.get("rule_type", ""),
        score=score,
        sourcebook=p.get("sourcebook", ""),
        chapter=p.get("chapter", ""),
        chapter_title=p.get("chapter_title", ""),
        section=p.get("section", ""),
        section_title=p.get("section_title", ""),
        chunk_id=p.get("chunk_id", ""),
        sub_paragraph=p.get("sub_paragraph", ""),
        page=p.get("page", 0),
        is_annex=p.get("is_annex", False),
        is_table=p.get("is_table", False),
        defined_terms=p.get("defined_terms", []),
        cross_references=p.get("cross_references", []),
    )
