"""Prompt templates and citation extraction for FCA Handbook QA."""

from __future__ import annotations

import re

from src.internal.retrieval.base import RetrievedChunk

SYSTEM_PROMPT = """\
You are a regulatory expert on the UK FCA Handbook. Answer questions using ONLY the provided context passages from FCA sourcebooks.

Rules:
1. Ground every claim in a specific rule. Cite rule IDs in the format [COBS 2.1.1R], [BCOBS 4.1.2G], etc.
2. Some context passages may not be relevant to the question. Focus on the passages that directly address the question and ignore the rest. Only say you cannot answer if NONE of the passages are relevant.
3. Do not make up rules or invent rule IDs.
4. When multiple rules are relevant, cite all of them.
5. Distinguish between Rules (R — legally binding), Guidance (G — expected practice), and Evidential provisions (E — presumption of compliance).
6. Keep your answer concise and precise. Regulatory accuracy matters more than length.
7. If the passages partially address the question, answer what you can and note what is not covered. Do not refuse entirely when partial information is available."""


def build_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """Build the user prompt with retrieved context chunks."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        label = chunk.display_id
        context_parts.append(f"--- Context {i} [{label}] ---\n{chunk.text}")

    context_block = "\n\n".join(context_parts)

    return f"""Context from FCA Handbook:

{context_block}

Question: {query}

Answer with specific rule citations:"""


_CITATION_RE = re.compile(
    r"(?:BCOBS|CASS|CMCOB|COBS|ESG|FPCOB|ICOBS|MAR|MCOB|PDCOB)"
    r"\s+\d+[A-Z]?\.\d+[A-Z]?\.\d+[A-Z]*[RGDEUK]{0,2}"
)


def extract_citations(text: str | None) -> list[str]:
    """Extract rule IDs cited in LLM output."""
    if not text:
        return []
    return list(set(_CITATION_RE.findall(text)))
