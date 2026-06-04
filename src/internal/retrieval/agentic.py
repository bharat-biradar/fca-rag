"""Approach 3: Agentic RAG — LLM agent with search + reference expansion tools.

The agent decides when to search, what to filter, when to expand references,
and when it has enough context to stop.
"""

from __future__ import annotations

import json
import os
import time

import weaviate.classes.query as wvq
from openai import OpenAI

from src.config import Settings, settings
from src.internal.retrieval.base import (
    BaseRetriever,
    RetrievedChunk,
    RetrievalResult,
    weaviate_obj_to_chunk,
)
from src.internal.retrieval.hybrid_rerank import HybridRerankRetriever

AGENT_SYSTEM_PROMPT = """\
You are a regulatory research assistant searching the UK FCA Handbook.

SOURCEBOOKS: BCOBS (Banking), CASS (Client Assets), CMCOB (Claims Management), \
COBS (Conduct of Business), ESG, FPCOB (Funeral Plans), ICOBS (Insurance), \
MAR (Market Conduct), MCOB (Mortgages), PDCOB (Pensions Dashboards).

You have two tools:
- search_rules: search by text with optional sourcebook filter. You can also ask it to reformulate your query into regulatory terms before searching.
- expand_references: given a rule ID, fetch the rules it cross-references.

GUIDELINES:
1. ALWAYS start with an unfiltered search (no sourcebook filter) using the original query. This gives the broadest coverage.
2. After seeing the first results, decide if you need targeted follow-up searches in specific sourcebooks.
3. Only use the sourcebook filter for follow-up searches when you want results from a specific sourcebook that didn't appear in the initial broad search.
4. Read the returned snippets carefully. If the top results don't address the actual question, search again with different terms or use reformulate=true.
5. When you find a highly relevant rule, call expand_references to discover linked rules.
6. You may stop as soon as you have results that clearly answer the question — there is no minimum number of calls."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_rules",
            "description": "Search FCA Handbook rules. Returns top 5 most relevant chunks with relevance scores. Optionally reformulates the query into specific regulatory terms before searching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query — use specific regulatory terms for best results",
                    },
                    "sourcebook": {
                        "type": "string",
                        "description": "Optional: filter to a specific sourcebook",
                        "enum": ["BCOBS", "CASS", "CMCOB", "COBS", "ESG", "FPCOB", "ICOBS", "MAR", "MCOB", "PDCOB"],
                    },
                    "reformulate": {
                        "type": "boolean",
                        "description": "If true, the query will be reformulated into specific FCA regulatory terms before searching. Use when the query is vague or produced poor results.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand_references",
            "description": "Given a rule ID from a previous search, fetch the rules it cross-references. Use when you found a highly relevant rule and want to discover related rules it links to.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_id": {
                        "type": "string",
                        "description": "The rule ID to expand, e.g. 'COBS 2.1.1' or 'CASS 7.13.1'",
                    },
                },
                "required": ["rule_id"],
            },
        },
    },
]


class AgenticRetriever(BaseRetriever):
    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg
        self.search = HybridRerankRetriever(cfg)

        # Use Gemini for agent reasoning if available (better tool calling)
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if gemini_key:
            self.client = OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=gemini_key,
            )
            self.model = "gemini-2.5-flash"
        else:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=cfg.openrouter_api_key,
            )
            self.model = cfg.generation_model

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        sourcebook_filter: str | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self.cfg.final_top_k
        start = time.time()

        # If caller passed a sourcebook filter, include it in the user message
        user_msg = query
        if sourcebook_filter:
            user_msg = f"{query} (focus on {sourcebook_filter} sourcebook)"

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        all_chunks: list[RetrievedChunk] = []
        agent_reasoning = ""
        steps = 0

        for step in range(self.cfg.max_agent_steps):
            steps = step + 1

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                temperature=0.1,
                max_tokens=1024,
            )

            choice = response.choices[0]

            # No tool calls — agent is done, capture its reasoning
            if not choice.message.tool_calls:
                agent_reasoning = choice.message.content or ""
                break

            # Append assistant message with tool calls
            messages.append(choice.message)

            # Execute each tool call
            for tool_call in choice.message.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                    args_short = {k: (v[:60] + '...' if isinstance(v, str) and len(v) > 60 else v) for k, v in args.items()}
                    print(f"    step {steps}: {name}({args_short})")
                except json.JSONDecodeError:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "Error: could not parse arguments. Try again with valid JSON.",
                    })
                    continue

                result_text = self._execute_tool(name, args, all_chunks)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })

        # Deduplicate, then late-stage rerank all candidates against original query
        chunks = self._deduplicate_and_rerank(all_chunks, query, top_k)

        elapsed_ms = (time.time() - start) * 1000
        return RetrievalResult(
            query=query,
            chunks=chunks,
            retrieval_time_ms=elapsed_ms,
            approach=f"agentic({steps}steps)",
        )

    def _execute_tool(
        self,
        name: str,
        args: dict,
        all_chunks: list[RetrievedChunk],
    ) -> str:
        """Execute a tool call and return a text summary for the agent."""
        if name == "search_rules":
            query = args["query"]
            sourcebook = args.get("sourcebook")
            reformulate = args.get("reformulate", False)

            # Reformulate + search in one round-trip
            if reformulate:
                ref_resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Rewrite the following query using specific UK FCA regulatory terminology. "
                            "Output ONLY the rewritten query, nothing else.",
                        },
                        {"role": "user", "content": query},
                    ],
                    temperature=0.1,
                    max_tokens=200,
                )
                rewritten = (ref_resp.choices[0].message.content or query).strip()
                query = rewritten

            result = self.search.retrieve(query, sourcebook_filter=sourcebook)
            all_chunks.extend(result.chunks)

            if not result.chunks:
                return "No results found."

            # Show 300+ chars so agent can judge relevance
            lines = []
            if reformulate:
                lines.append(f"Reformulated query: {query}")
            lines.append(f"Found {len(result.chunks)} rules:")
            for c in result.chunks:
                snippet = c.text[:300].replace('\n', ' ')
                xref_note = f" [has {len(c.cross_references)} cross-refs]" if c.cross_references else ""
                lines.append(
                    f"  - [{c.score:.2f}] {c.display_id} ({c.sourcebook}){xref_note}: {snippet}..."
                )
            return "\n".join(lines)

        elif name == "expand_references":
            rule_id = args["rule_id"].strip()
            collection = self.search.collection

            # Fetch the rule to get its cross_references field
            results = collection.query.fetch_objects(
                filters=wvq.Filter.by_property("rule_id").equal(rule_id),
                limit=1,
            )

            if not results.objects:
                return f"Rule {rule_id} not found."

            parent_props = results.objects[0].properties
            xrefs = parent_props.get("cross_references", [])
            if not xrefs:
                return f"Rule {rule_id} has no cross-references."

            # Fetch referenced rules with placeholder score (FlashRank will score them at the end)
            lines = [f"Rule {rule_id} references {len(xrefs)} rules:"]
            for ref_id in xrefs[:8]:  # cap to avoid excessive lookups
                ref_results = collection.query.fetch_objects(
                    filters=wvq.Filter.by_property("rule_id").equal(ref_id),
                    limit=1,
                )
                if ref_results.objects:
                    chunk = weaviate_obj_to_chunk(ref_results.objects[0], score=0.0)
                    all_chunks.append(chunk)
                    snippet = chunk.text[:200].replace('\n', ' ')
                    lines.append(
                        f"  - {chunk.display_id} ({chunk.sourcebook}): {snippet}..."
                    )
                else:
                    lines.append(f"  - {ref_id}: not found in collection")

            return "\n".join(lines)

        return f"Unknown tool: {name}"

    def _deduplicate_and_rerank(
        self, chunks: list[RetrievedChunk], original_query: str, top_k: int
    ) -> list[RetrievedChunk]:
        """Deduplicate by chunk_id, then late-stage FlashRank rerank against original query."""
        # Unique by chunk_id
        seen: dict[str, RetrievedChunk] = {}
        for c in chunks:
            if c.chunk_id not in seen:
                seen[c.chunk_id] = c
        candidates = list(seen.values())

        if not candidates:
            return []

        # Single FlashRank pass against the original user query
        reranked = self.search.rerank_chunks(original_query, candidates)
        return reranked[:top_k]


# --- Runnable standalone ---

if __name__ == "__main__":
    retriever = AgenticRetriever()

    test_queries = [
        "What must a firm do when providing investment services?",
        "Compare disclosure rules across banking and insurance",
        "What are a firm's obligations to retail clients?",
    ]

    for q in test_queries:
        result = retriever.retrieve(q)
        print(f"\nQuery: {q}")
        print(f"Approach: {result.approach}")
        print(f"Time: {result.retrieval_time_ms:.0f}ms")
        for c in result.chunks:
            print(f"  [{c.score:.4f}] {c.display_id} ({c.sourcebook}) — {c.text[:80]}...")
