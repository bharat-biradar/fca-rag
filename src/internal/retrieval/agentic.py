"""Approach 3: Agentic RAG — LLM agent with search + reformulation tools.

The agent decides when to search, what to filter, when to reformulate,
and when it has enough context to stop.
"""

from __future__ import annotations

import json
import os
import time

from openai import OpenAI

from src.config import Settings, settings
from src.internal.retrieval.base import (
    BaseRetriever,
    RetrievedChunk,
    RetrievalResult,
)
from src.internal.retrieval.hybrid_rerank import HybridRerankRetriever

AGENT_SYSTEM_PROMPT = """\
You are a regulatory research assistant with access to the UK FCA Handbook.
You have tools to search rules by text and reformulate queries for better results.

The FCA Handbook covers these sourcebooks: BCOBS (Banking), CASS (Client Assets), \
CMCOB (Claims Management), COBS (Conduct of Business), ESG, FPCOB (Funeral Plans), \
ICOBS (Insurance), MAR (Market Conduct), MCOB (Mortgages), PDCOB (Pensions Dashboards).

For each user query:
1. Analyze what type of question it is.
2. If the query explicitly names a sourcebook (e.g. "under COBS"), filter to that sourcebook.
3. If the query is broad or could span multiple sourcebooks, do NOT filter — search across all sourcebooks first. Then do targeted follow-up searches in specific sourcebooks if needed.
4. If the query is vague, reformulate it into specific regulatory terms before searching.
5. For topics that span multiple areas (e.g. "consumer protections"), search relevant sourcebooks separately: COBS for general conduct, ICOBS for insurance, BCOBS for banking, MCOB for mortgages, etc.
6. After each search, look at the scores. If scores are below 0.90, consider reformulating or searching a different sourcebook.
7. If you find a highly relevant rule, use expand_references to discover linked rules.
8. When you have relevant rules covering the query from the appropriate sourcebooks, stop calling tools.

Most queries need 2-4 tool calls."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_rules",
            "description": "Search for FCA Handbook rules by text similarity. Returns top 5 most relevant rule chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query — use specific regulatory terms for best results",
                    },
                    "sourcebook": {
                        "type": "string",
                        "description": "Optional: filter to a specific sourcebook (BCOBS, CASS, CMCOB, COBS, ESG, FPCOB, ICOBS, MAR, MCOB, PDCOB)",
                        "enum": ["BCOBS", "CASS", "CMCOB", "COBS", "ESG", "FPCOB", "ICOBS", "MAR", "MCOB", "PDCOB"],
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reformulate_query",
            "description": "Reformulate a query into more specific regulatory terms for better search results. Use when initial results are poor or the query is vague.",
            "parameters": {
                "type": "object",
                "properties": {
                    "original_query": {
                        "type": "string",
                        "description": "The original user query",
                    },
                    "feedback": {
                        "type": "string",
                        "description": "What was wrong with previous results — too broad, wrong sourcebook, missing specific topic, etc.",
                    },
                },
                "required": ["original_query", "feedback"],
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

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        all_chunks: list[RetrievedChunk] = []
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

            # No tool calls — agent is done
            if not choice.message.tool_calls:
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
                    # LLM returned malformed JSON — skip this tool call
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

        # Deduplicate by chunk_id, keep highest score
        chunks = self._deduplicate(all_chunks, top_k)

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
            result = self.search.retrieve(query, sourcebook_filter=sourcebook)
            all_chunks.extend(result.chunks)

            # Build summary for agent
            if not result.chunks:
                return "No results found."

            lines = [f"Found {len(result.chunks)} rules (score 0-1, higher=more relevant):"]
            for c in result.chunks:
                lines.append(
                    f"  - [{c.score:.2f}] {c.display_id} ({c.sourcebook}): {c.text[:120]}..."
                )
            return "\n".join(lines)

        elif name == "reformulate_query":
            # Use LLM to reformulate
            reformulate_resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Rewrite the following query using specific UK FCA regulatory terminology. "
                        "Output ONLY the rewritten query, nothing else.",
                    },
                    {
                        "role": "user",
                        "content": f"Original: {args['original_query']}\nFeedback: {args['feedback']}",
                    },
                ],
                temperature=0.1,
                max_tokens=200,
            )
            rewritten = reformulate_resp.choices[0].message.content.strip()
            return f"Reformulated query: {rewritten}"

        elif name == "expand_references":
            # Find the chunk with this rule_id and look up its cross-references
            rule_id = args["rule_id"].strip()
            import weaviate.classes.query as wvq

            collection = self.search.collection

            # Fetch the rule to get its cross_references field
            results = collection.query.fetch_objects(
                filters=wvq.Filter.by_property("rule_id").equal(rule_id),
                limit=1,
            )

            if not results.objects:
                return f"Rule {rule_id} not found."

            xrefs = results.objects[0].properties.get("cross_references", [])
            if not xrefs:
                return f"Rule {rule_id} has no cross-references."

            # Search for each referenced rule
            lines = [f"Rule {rule_id} references {len(xrefs)} rules:"]
            for ref_id in xrefs[:10]:  # cap at 10 to avoid excessive searches
                ref_results = collection.query.fetch_objects(
                    filters=wvq.Filter.by_property("rule_id").equal(ref_id),
                    limit=1,
                )
                if ref_results.objects:
                    from src.internal.retrieval.base import weaviate_obj_to_chunk
                    chunk = weaviate_obj_to_chunk(ref_results.objects[0], score=0.8)
                    all_chunks.append(chunk)
                    lines.append(
                        f"  - {chunk.display_id} ({chunk.sourcebook}): {chunk.text[:120]}..."
                    )
                else:
                    lines.append(f"  - {ref_id}: not found in collection")

            return "\n".join(lines)

        return f"Unknown tool: {name}"

    def _deduplicate(
        self, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Deduplicate by rule_id, keeping highest-scoring chunk per rule."""
        seen: dict[str, RetrievedChunk] = {}
        for c in chunks:
            key = c.rule_id  # one chunk per rule, not per chunk_id
            if key not in seen or c.score > seen[key].score:
                seen[key] = c
        ranked = sorted(seen.values(), key=lambda c: c.score, reverse=True)
        return ranked[:top_k]


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
