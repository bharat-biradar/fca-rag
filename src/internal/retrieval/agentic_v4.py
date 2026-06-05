"""Approach 5: Agentic RAG v4 — Sonnet decides, Haiku grades, diversity-aware rerank.

Loop (max 5 iterations):
  1. Sonnet calls tools (search, lookup, expand_refs, graph_expand)
  2. New chunks filtered — skip already-graded chunk_ids
  3. Haiku grades only NEW chunks (YES/NO per chunk, ~200 chars each)
  4. Approved chunk_ids stored in approved_set (persists across iterations)
  5. Haiku builds compact summary: approved rules by sourcebook + gaps
  6. Sonnet sees ONLY Haiku's summary (~80 tokens) — decides stop or next tool

Final: FlashRank rerank full pool with sourcebook diversity guarantee.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter, defaultdict

import weaviate.classes.query as wvq
from flashrank import RerankRequest

from src.config import Settings, settings
from src.dependencies import get_embedding_model, get_neo4j_driver, get_weaviate_client
from src.internal.retrieval.base import (
    BaseRetriever,
    RetrievedChunk,
    RetrievalResult,
    weaviate_obj_to_chunk,
)
from src.internal.retrieval.hybrid_rerank import _get_ranker
from src.internal.retrieval.agentic_v3 import clean_rule_id, deduplicate_chunks

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_ITERATIONS = 5

AGENT_SYSTEM_PROMPT = """\
You are a regulatory research assistant searching the UK FCA Handbook.

SOURCEBOOKS: BCOBS (Banking), CASS (Client Assets), CMCOB (Claims Management), \
COBS (Conduct of Business), ESG (Environmental/Social/Governance), \
FPCOB (Funeral Plans), ICOBS (Insurance), MAR (Market Conduct), \
MCOB (Mortgages), PDCOB (Pensions Dashboards).

You have four tools:
- search_rules: hybrid search by text query, returns top 10 chunks.
- lookup_rule_ids: directly fetch specific rules by ID (e.g. COBS 2.1.1).
- expand_refs: given a rule ID, fetches rules it cross-references. Forward 1-hop.
- graph_expand: Neo4j 1-2 hops bidirectional. Use for reverse references or multi-hop chains.

GUIDELINES:
1. Start with broad searches covering the main topics in the question.
2. After each tool call, you will see a grading summary showing which rules were judged relevant, which sourcebooks are covered, and what gaps remain.
3. If the summary shows missing sourcebooks or topics, search specifically for those.
4. Use lookup_rule_ids when specific rules are mentioned in the question.
5. Use expand_refs or graph_expand when you need connected rules.
6. Stop when the grading summary shows sufficient relevant rules for the question.
7. Output a plain text message (no tool call) when done.

OUTPUT: Either call a tool OR output a plain text message to stop."""

HAIKU_GRADE_PROMPT = """\
For each chunk below, determine if it is relevant to answering the question.

Question: {question}

{chunks_text}

For each chunk, output one line in this exact format:
- If relevant: YES | <one-sentence summary of what this rule requires or states>
- If not relevant: NO

Output exactly {num_chunks} lines, nothing else."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_rules",
            "description": "Search FCA Handbook rules via hybrid search. Returns top 10 chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query — use specific FCA regulatory terminology",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_rule_ids",
            "description": "Directly fetch specific rules by their ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of rule IDs, e.g. ['COBS 2.1.1', 'CASS 7.11.34']",
                    },
                },
                "required": ["rule_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand_refs",
            "description": "Read a rule's cross_references and fetch those rules. Forward 1-hop.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_id": {
                        "type": "string",
                        "description": "The rule ID to expand",
                    },
                },
                "required": ["rule_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_expand",
            "description": "Neo4j 1-2 hops bidirectional from a rule. For reverse lookups and multi-hop.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_id": {
                        "type": "string",
                        "description": "The seed rule ID",
                    },
                    "hops": {
                        "type": "integer",
                        "description": "Hops (1 or 2). Default 2.",
                        "enum": [1, 2],
                    },
                },
                "required": ["rule_id"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class AgenticV4Retriever(BaseRetriever):
    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg
        self.client_wv = get_weaviate_client(cfg)
        self.collection = self.client_wv.collections.get(cfg.weaviate_collection)
        self.embed_model = get_embedding_model(cfg)
        self.ranker = _get_ranker()
        self.neo4j = get_neo4j_driver(cfg)

        # Agent driver + grader models
        aws_key = os.getenv("AWS_ACCESS_KEY_ID", "")
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if aws_key:
            self._use_litellm = True
            self.agent_model = "bedrock/global.anthropic.claude-sonnet-4-6"
            self.haiku_model = "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0"
        elif gemini_key:
            from openai import OpenAI
            self._use_litellm = False
            self.llm_client = OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=gemini_key,
            )
            self.agent_model = "gemini-2.5-flash"
            self.haiku_model = "gemini-2.5-flash"
        else:
            from openai import OpenAI
            self._use_litellm = False
            self.llm_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=cfg.openrouter_api_key,
            )
            self.agent_model = cfg.generation_model
            self.haiku_model = cfg.generation_model

    def _llm_call(self, model: str, messages: list, **kwargs):
        """Route LLM call through litellm (Bedrock) or OpenAI client."""
        if getattr(self, "_use_litellm", False):
            import litellm
            return litellm.completion(model=model, messages=messages, **kwargs)
        return self.llm_client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )

    # ------------------------------------------------------------------
    # Haiku chunk grading
    # ------------------------------------------------------------------

    def _haiku_grade(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[tuple[RetrievedChunk, bool, str]]:
        """Haiku grades each chunk YES/NO with summary. Returns (chunk, is_relevant, summary)."""
        if not chunks:
            return []

        chunks_text = "\n".join(
            f"Chunk {i+1} [{c.display_id}] ({c.sourcebook}): {c.text[:200]}"
            for i, c in enumerate(chunks)
        )
        prompt = HAIKU_GRADE_PROMPT.format(
            question=query,
            chunks_text=chunks_text,
            num_chunks=len(chunks),
        )

        try:
            resp = self._llm_call(
                model=self.haiku_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
            )
            text = (resp.choices[0].message.content or "").strip()
            lines = [l.strip() for l in text.split("\n") if l.strip()]

            usage = resp.usage
            tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0
            self._last_haiku_tokens = tokens

            results = []
            for i, c in enumerate(chunks):
                if i < len(lines):
                    line = lines[i]
                    if line.upper().startswith("YES"):
                        # Extract summary after "YES | "
                        summary = line.split("|", 1)[1].strip() if "|" in line else ""
                        results.append((c, True, summary))
                    else:
                        results.append((c, False, ""))
                else:
                    results.append((c, False, ""))
            return results
        except Exception as e:
            print(f"    [warn] haiku grading failed: {e}")
            self._last_haiku_tokens = 0
            return [(c, True, "") for c in chunks]

    def _build_grading_summary(
        self,
        approved_chunks: dict[str, RetrievedChunk],
        approved_summaries: dict[str, str],
        rejected_ids: set[str],
        graded_ids: set[str],
        iteration: int,
        new_approvals_this_round: int,
    ) -> str:
        """Build compact summary for Sonnet. Includes rule summaries, fetched IDs, and stop signal."""
        if not approved_chunks:
            return f"No relevant rules found yet. Iteration {iteration}/{MAX_ITERATIONS}."

        # Approved rules with summaries (no sourcebook grouping)
        rule_lines = []
        for chunk_id, c in approved_chunks.items():
            summary = approved_summaries.get(chunk_id, "")
            if summary:
                rule_lines.append(f"  - {c.display_id}: {summary}")
            else:
                rule_lines.append(f"  - {c.display_id}")

        # Already-fetched rule IDs (so Sonnet doesn't re-lookup)
        fetched_rule_ids = sorted(set(c.rule_id for c in approved_chunks.values()))

        # Stop signal
        if new_approvals_this_round == 0:
            stop_signal = "WARNING: No new relevant rules found this round. Consider stopping."
        else:
            stop_signal = f"New approvals this round: {new_approvals_this_round}"

        lines = [
            f"Relevant rules ({len(approved_chunks)} approved, {len(rejected_ids)} rejected):",
            *rule_lines[:15],
        ]
        if len(rule_lines) > 15:
            lines.append(f"  +{len(rule_lines)-15} more")
        lines.extend([
            f"",
            f"Already fetched (do NOT re-lookup): {', '.join(fetched_rule_ids)}",
            f"{stop_signal}",
            f"Iteration: {iteration}/{MAX_ITERATIONS}",
        ])
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Main retrieve loop
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        sourcebook_filter: str | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self.cfg.final_top_k
        start = time.time()
        total_sonnet_tokens = 0
        total_haiku_tokens = 0

        all_chunks: list[RetrievedChunk] = []
        approved_chunks: dict[str, RetrievedChunk] = {}  # chunk_id -> chunk
        approved_summaries: dict[str, str] = {}  # chunk_id -> haiku summary
        rejected_ids: set[str] = set()
        graded_ids: set[str] = set()  # all graded (approved + rejected)

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        for iteration in range(1, MAX_ITERATIONS + 1):
            print(f"    --- iteration {iteration}/{MAX_ITERATIONS} ---")

            # Sonnet decides next action
            response = self._llm_call(
                model=self.agent_model,
                messages=messages,
                tools=TOOLS,
                temperature=0.1,
                max_tokens=1024,
            )

            usage = response.usage
            total_sonnet_tokens += (usage.prompt_tokens + usage.completion_tokens) if usage else 0

            choice = response.choices[0]

            # No tool calls — Sonnet is done
            if not choice.message.tool_calls:
                reasoning = choice.message.content or ""
                print(f"    agent stopped: {reasoning[:100]}")
                break

            # Execute tool calls
            messages.append(choice.message)
            round_approvals = 0

            for tool_call in choice.message.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "Error: could not parse arguments.",
                    })
                    continue

                args_short = {
                    k: (v[:60] + "..." if isinstance(v, str) and len(v) > 60 else v)
                    for k, v in args.items()
                }
                print(f"    tool: {name}({args_short})")

                # Execute tool
                new_chunks = self._execute_tool(name, args)
                all_chunks.extend(new_chunks)

                # Filter out already-graded chunks
                ungraded = [c for c in new_chunks if c.chunk_id not in graded_ids]
                # Dedup within batch
                seen_batch: set[str] = set()
                ungraded_deduped = []
                for c in ungraded:
                    if c.chunk_id not in seen_batch:
                        seen_batch.add(c.chunk_id)
                        ungraded_deduped.append(c)
                ungraded = ungraded_deduped

                print(f"    new chunks: {len(new_chunks)}, ungraded: {len(ungraded)}")

                # Haiku grades only if 2+ ungraded chunks (skip overhead for 0-1)
                if len(ungraded) >= 2:
                    self._last_haiku_tokens = 0
                    graded_results = self._haiku_grade(query, ungraded)
                    total_haiku_tokens += self._last_haiku_tokens

                    newly_approved = 0
                    newly_rejected = 0
                    for c, is_relevant, summary_text in graded_results:
                        graded_ids.add(c.chunk_id)
                        if is_relevant:
                            approved_chunks[c.chunk_id] = c
                            approved_summaries[c.chunk_id] = summary_text
                            newly_approved += 1
                        else:
                            rejected_ids.add(c.chunk_id)
                            newly_rejected += 1
                    round_approvals += newly_approved
                    print(f"    haiku: +{newly_approved} approved, +{newly_rejected} rejected (total: {len(approved_chunks)} approved)")
                elif len(ungraded) == 1:
                    # Single chunk — assume relevant, skip Haiku call
                    c = ungraded[0]
                    graded_ids.add(c.chunk_id)
                    approved_chunks[c.chunk_id] = c
                    approved_summaries[c.chunk_id] = ""
                    round_approvals += 1
                    print(f"    skipped haiku (1 chunk), auto-approved")

                # Build summary for Sonnet (last tool call gets the full summary)
                summary = self._build_grading_summary(
                    approved_chunks, approved_summaries, rejected_ids,
                    graded_ids, iteration, round_approvals,
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": summary,
                })

            # Trim: keep system + user + last assistant/tool exchange only
            messages = self._trim_messages(messages)

        # Final: dedup full pool + FlashRank with diversity guarantee
        unique = deduplicate_chunks(all_chunks)
        sb_count = len(set(c.sourcebook for c in unique))
        approved_sb_count = len(set(c.sourcebook for c in approved_chunks.values()))
        dynamic_top_k = max(top_k, min(10, approved_sb_count * 2))

        print(f"    final: {len(unique)} unique, {len(approved_chunks)} approved, {sb_count} sourcebooks -> top_k={dynamic_top_k}")
        print(f"    tokens: sonnet={total_sonnet_tokens}, haiku={total_haiku_tokens}")

        chunks = self._final_rerank(query, unique, dynamic_top_k, approved_chunks)

        elapsed_ms = (time.time() - start) * 1000
        return RetrievalResult(
            query=query,
            chunks=chunks,
            retrieval_time_ms=elapsed_ms,
            approach=f"agentic_v4({iteration}iter,{len(approved_chunks)}approved)",
            planning_tokens=total_sonnet_tokens + total_haiku_tokens,
        )

    # ------------------------------------------------------------------
    # Message trimming
    # ------------------------------------------------------------------

    def _trim_messages(self, messages: list[dict]) -> list[dict]:
        """Keep system + user + only the last assistant/tool exchange."""
        system_msg = messages[0]
        user_msg = messages[1]

        last_assistant_idx = None
        for i in range(len(messages) - 1, 1, -1):
            msg = messages[i]
            if hasattr(msg, "tool_calls") or (isinstance(msg, dict) and msg.get("role") == "assistant"):
                last_assistant_idx = i
                break

        if last_assistant_idx is None:
            return messages

        return [system_msg, user_msg] + messages[last_assistant_idx:]

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(self, name: str, args: dict) -> list[RetrievedChunk]:
        """Execute a tool and return the chunks found."""
        if name == "search_rules":
            return self._tool_search(args["query"])
        elif name == "lookup_rule_ids":
            return self._tool_lookup(args["rule_ids"])
        elif name == "expand_refs":
            return self._tool_expand_refs(args["rule_id"])
        elif name == "graph_expand":
            return self._tool_graph_expand(args["rule_id"], args.get("hops", 2))
        return []

    def _tool_search(self, query: str) -> list[RetrievedChunk]:
        """Hybrid search, no sourcebook filter forced."""
        query_vec = self.embed_model.encode(query, normalize_embeddings=True).tolist()
        results = self.collection.query.hybrid(
            query=query,
            vector=query_vec,
            alpha=self.cfg.hybrid_alpha,
            limit=10,
            return_metadata=wvq.MetadataQuery(score=True),
        )
        chunks = []
        for obj in results.objects:
            score = obj.metadata.score if obj.metadata.score else 0.0
            chunks.append(weaviate_obj_to_chunk(obj, score=score))
        return chunks

    def _tool_lookup(self, rule_ids: list[str]) -> list[RetrievedChunk]:
        """Direct fetch by rule ID."""
        chunks = []
        for rule_id in rule_ids[:10]:
            clean_id = clean_rule_id(rule_id)
            results = self.collection.query.fetch_objects(
                filters=wvq.Filter.by_property("rule_id").equal(clean_id),
                limit=1,
            )
            if results.objects:
                chunks.append(weaviate_obj_to_chunk(results.objects[0], score=1.0))
        return chunks

    def _tool_expand_refs(self, rule_id: str) -> list[RetrievedChunk]:
        """Forward 1-hop via cross_references metadata."""
        clean_id = clean_rule_id(rule_id)
        results = self.collection.query.fetch_objects(
            filters=wvq.Filter.by_property("rule_id").equal(clean_id),
            limit=1,
        )
        if not results.objects:
            return []

        xrefs = results.objects[0].properties.get("cross_references", [])
        if not xrefs:
            return []

        chunks = []
        for ref_id in xrefs[:8]:
            ref_results = self.collection.query.fetch_objects(
                filters=wvq.Filter.by_property("rule_id").equal(ref_id),
                limit=1,
            )
            if ref_results.objects:
                chunks.append(weaviate_obj_to_chunk(ref_results.objects[0], score=0.5))
        return chunks

    def _tool_graph_expand(self, rule_id: str, hops: int = 2) -> list[RetrievedChunk]:
        """Bidirectional Neo4j traversal 1-2 hops."""
        clean_id = clean_rule_id(rule_id)
        hops = min(hops, 2)

        cypher = f"""
            MATCH (seed:Rule)-[:REFERENCES*1..{hops}]-(connected:Rule)
            WHERE seed.id = $seed_id
              AND connected.id <> $seed_id
              AND connected.stub = false
            RETURN DISTINCT connected.id AS id
            LIMIT $limit
        """
        try:
            with self.neo4j.session(database=self.cfg.neo4j_database) as session:
                result = session.run(
                    cypher, seed_id=clean_id, limit=self.cfg.graph_expansion_limit,
                )
                expanded_ids = [record["id"] for record in result]
        except Exception as e:
            print(f"    [warn] graph_expand failed: {e}")
            return []

        chunks = []
        for eid in expanded_ids:
            results = self.collection.query.fetch_objects(
                filters=wvq.Filter.by_property("rule_id").equal(eid),
                limit=1,
            )
            if results.objects:
                chunks.append(weaviate_obj_to_chunk(results.objects[0], score=0.3))
        return chunks

    # ------------------------------------------------------------------
    # Final rerank with diversity guarantee + approved prioritization
    # ------------------------------------------------------------------

    def _final_rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
        approved_chunks: dict[str, RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """FlashRank rerank with:
        1. Sourcebook diversity floor (best from each sourcebook first)
        2. Haiku-approved chunks prioritized over non-approved
        """
        if not candidates:
            return []

        passages = [
            {"id": i, "text": c.text, "meta": c}
            for i, c in enumerate(candidates)
        ]
        rerank_req = RerankRequest(query=query, passages=passages)
        reranked = self.ranker.rerank(rerank_req)

        for item in reranked:
            item["meta"].score = float(item["score"])

        scored = [item["meta"] for item in reranked]
        approved_ids = set(approved_chunks.keys())

        # Step 1: Best APPROVED chunk from each sourcebook (diversity floor)
        selected: list[RetrievedChunk] = []
        selected_ids: set[str] = set()
        seen_sbs: set[str] = set()

        for c in scored:
            if c.sourcebook not in seen_sbs and c.chunk_id in approved_ids:
                selected.append(c)
                selected_ids.add(c.chunk_id)
                seen_sbs.add(c.sourcebook)
            if len(selected) >= top_k:
                break

        # Step 2: Fill with remaining approved chunks by score
        for c in scored:
            if len(selected) >= top_k:
                break
            if c.chunk_id not in selected_ids and c.chunk_id in approved_ids:
                selected.append(c)
                selected_ids.add(c.chunk_id)

        # Step 3: If still not full, fill with non-approved by score
        for c in scored:
            if len(selected) >= top_k:
                break
            if c.chunk_id not in selected_ids:
                selected.append(c)
                selected_ids.add(c.chunk_id)

        return selected


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()

    from src.config import Settings
    from src.internal.generation.llm import LLMClient
    from src.internal.generation.prompts import SYSTEM_PROMPT, build_user_prompt, extract_citations

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What due diligence must firms perform on third parties they work with?"

    cfg = Settings(weaviate_collection="FCARule_v2")
    retriever = AgenticV4Retriever(cfg)

    print(f"Query: {query}\n")
    result = retriever.retrieve(query)

    print(f"\n--- Retrieved {len(result.chunks)} chunks in {result.retrieval_time_ms:.0f}ms ({result.approach}) ---")
    for c in result.chunks:
        approved_tag = " [approved]" if c.chunk_id in {ch.chunk_id for ch in result.chunks} else ""
        print(f"  [{c.score:.4f}] {c.display_id} ({c.sourcebook})")

    # Generate answer
    llm = LLMClient(cfg)
    user_prompt = build_user_prompt(query, result.chunks)
    response = llm.generate(SYSTEM_PROMPT, user_prompt)

    print(f"\n--- Answer ---")
    print(response.text)
    print(f"\nCitations: {extract_citations(response.text)}")
    print(f"Tokens: {response.prompt_tokens} in / {response.completion_tokens} out / {result.planning_tokens} planning")
