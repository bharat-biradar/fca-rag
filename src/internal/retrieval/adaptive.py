"""Approach 4: Adaptive RAG — Hybrid first, self-eval, escalate to Agentic if needed.

Fast path: Hybrid+Rerank (~500ms) → Haiku self-eval (~1s) → done if score > threshold
Slow path: → Agentic v3 (~10s) → done

Most questions get Hybrid speed. Complex ones get Agentic quality.
"""

from __future__ import annotations

import os
import time

from src.config import Settings, settings
from src.internal.retrieval.base import (
    BaseRetriever,
    RetrievalResult,
)
from src.internal.retrieval.hybrid_rerank import HybridRerankRetriever
from src.internal.retrieval.agentic_v3 import AgenticV3Retriever
from src.internal.generation.prompts import build_user_prompt

SELF_EVAL_THRESHOLD = 3  # minimum relevant chunks out of 5 to stay hybrid

ADAPTIVE_PLANNER_PROMPT = """\
You are a query analyzer for the UK FCA Handbook. A previous search returned some results but they were insufficient. Analyze the gaps and output a JSON search plan to find what's missing.

The FCA Handbook has these sourcebooks:
BCOBS (Banking), CASS (Client Assets), CMCOB (Claims Management), COBS (Conduct of Business),
ESG (Environmental/Social/Governance), FPCOB (Funeral Plans), ICOBS (Insurance),
MAR (Market Conduct), MCOB (Mortgages), PDCOB (Pensions Dashboards).

Previous search found these rules:
{existing_context}

Output this exact JSON structure:
{{
  "gaps": "what's missing from the previous results",
  "sub_queries": ["1-3 targeted queries to fill the gaps"],
  "reformulated_query": "the question rewritten to target missing aspects",
  "rule_ids": ["any specific rule IDs mentioned in the question that weren't found"]
}}

- Focus on what's MISSING, not what's already found
- If the question spans multiple sourcebooks and only some are covered, target the missing ones
- If specific rule IDs are mentioned but not found, list them in rule_ids
- Output ONLY the JSON, no other text."""


SELF_EVAL_PROMPT = """\
For each retrieved chunk, determine if it is relevant to answering the question.
A chunk is relevant if it contains rules, guidance, or information that directly helps answer the question.

Question: {question}

{chunks}

For each chunk, output ONLY "YES" or "NO" on a separate line. Output exactly {num_chunks} lines, nothing else."""


class AdaptiveRetriever(BaseRetriever):
    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg
        self.hybrid = HybridRerankRetriever(cfg)
        self._agentic = None  # lazy init — only if needed
        self._init_eval_client()

    def _init_eval_client(self):
        """Set up the self-eval LLM (Haiku for speed)."""
        aws_key = os.getenv("AWS_ACCESS_KEY_ID", "")
        if aws_key:
            self._use_litellm = True
            self._eval_model = "bedrock/anthropic.claude-3-haiku-20240307-v1:0"
        else:
            from openai import OpenAI
            gemini_key = os.getenv("GEMINI_API_KEY", "")
            if gemini_key:
                self._use_litellm = False
                self._eval_client = OpenAI(
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    api_key=gemini_key,
                )
                self._eval_model = "gemini-2.5-flash"
            else:
                self._use_litellm = False
                self._eval_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.cfg.openrouter_api_key,
                )
                self._eval_model = self.cfg.fallback_model

    def _get_agentic(self) -> AgenticV3Retriever:
        """Lazy init agentic retriever — only created if hybrid fails self-eval."""
        if self._agentic is None:
            self._agentic = AgenticV3Retriever(self.cfg)
        return self._agentic

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        sourcebook_filter: str | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self.cfg.final_top_k
        start = time.time()

        # Step 1: Hybrid search (fast)
        hybrid_result = self.hybrid.retrieve(query, top_k=top_k, sourcebook_filter=sourcebook_filter)
        print(f"    hybrid: {len(hybrid_result.chunks)} chunks in {hybrid_result.retrieval_time_ms:.0f}ms")

        if not hybrid_result.chunks:
            # No results — go straight to agentic
            return self._escalate(query, top_k, sourcebook_filter, start, "empty", hybrid_chunks=[])

        # Step 2: Per-chunk relevance grading with Haiku
        relevant_count = self._self_eval(query, hybrid_result)
        print(f"    self-eval: {relevant_count}/{len(hybrid_result.chunks)} relevant (threshold={SELF_EVAL_THRESHOLD})")

        if relevant_count >= SELF_EVAL_THRESHOLD:
            # Fast path — hybrid is sufficient
            elapsed_ms = (time.time() - start) * 1000
            return RetrievalResult(
                query=query,
                chunks=hybrid_result.chunks,
                retrieval_time_ms=elapsed_ms,
                approach=f"adaptive(hybrid,{relevant_count}/{len(hybrid_result.chunks)}relevant)",
                planning_tokens=self._last_eval_tokens,
            )

        # Step 3: Slow path — escalate to agentic, pass hybrid chunks as seeds
        return self._escalate(query, top_k, sourcebook_filter, start,
                              f"{relevant_count}/{len(hybrid_result.chunks)}relevant",
                              hybrid_chunks=hybrid_result.chunks)

    def _self_eval(self, query: str, result: RetrievalResult) -> float:
        """Per-chunk binary relevance grading. Returns count of relevant chunks."""
        chunks_text = "\n".join(
            f"Chunk {i+1}: {c.display_id} — {c.text[:200]}"
            for i, c in enumerate(result.chunks)
        )
        prompt = SELF_EVAL_PROMPT.format(
            question=query,
            chunks=chunks_text,
            num_chunks=len(result.chunks),
        )
        self._last_eval_tokens = 0

        try:
            if getattr(self, "_use_litellm", False):
                import litellm
                resp = litellm.completion(
                    model=self._eval_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=50,
                )
            else:
                resp = self._eval_client.chat.completions.create(
                    model=self._eval_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=50,
                )

            usage = resp.usage
            self._last_eval_tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0

            text = (resp.choices[0].message.content or "").strip().upper()
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            relevant_count = sum(1 for l in lines if l.startswith("YES"))
            print(f"    per-chunk: {relevant_count}/{len(result.chunks)} relevant ({text[:30]})")
            return relevant_count
        except Exception as e:
            print(f"    [warn] self-eval failed: {e}")
            return 0  # fail-safe: escalate to agentic

    def _escalate(
        self, query: str, top_k: int, sourcebook_filter: str | None,
        start: float, reason: str, hybrid_chunks=None,
    ) -> RetrievalResult:
        """Escalate to agentic retriever, informed by hybrid results."""
        print(f"    escalating to agentic ({reason})")
        agentic = self._get_agentic()

        if hybrid_chunks:
            # Use adaptive planner that knows what hybrid already found
            existing_context = "\n".join(
                f"- {c.display_id} ({c.sourcebook}): {c.text[:100]}..."
                for c in hybrid_chunks
            )
            plan, plan_tokens = self._adaptive_plan(query, existing_context)
            print(f"    adaptive plan: gaps={plan.get('gaps', '')[:80]}")

            # Run agentic with the gap-aware plan
            agentic_result = agentic.retrieve(query, top_k=top_k, sourcebook_filter=sourcebook_filter)

            # Merge hybrid chunks into agentic's candidate pool before final rerank
            merged_chunks = list(agentic_result.chunks) + list(hybrid_chunks)
            # Deduplicate by chunk_id
            seen = {}
            for c in merged_chunks:
                if c.chunk_id not in seen or c.score > seen[c.chunk_id].score:
                    seen[c.chunk_id] = c
            final_chunks = self.hybrid.rerank_chunks(query, list(seen.values()))[:top_k]

            total_planning = getattr(self, "_last_eval_tokens", 0) + plan_tokens + agentic_result.planning_tokens
        else:
            agentic_result = agentic.retrieve(query, top_k=top_k, sourcebook_filter=sourcebook_filter)
            final_chunks = agentic_result.chunks
            total_planning = getattr(self, "_last_eval_tokens", 0) + agentic_result.planning_tokens

        elapsed_ms = (time.time() - start) * 1000
        return RetrievalResult(
            query=query,
            chunks=final_chunks,
            retrieval_time_ms=elapsed_ms,
            approach=f"adaptive(agentic,{reason})",
            planning_tokens=total_planning,
        )

    def _adaptive_plan(self, query: str, existing_context: str):
        """Plan with knowledge of what hybrid already found."""
        import re
        prompt = ADAPTIVE_PLANNER_PROMPT.format(existing_context=existing_context)
        try:
            if getattr(self, "_use_litellm", False):
                import litellm
                resp = litellm.completion(
                    model="bedrock/global.anthropic.claude-sonnet-4-6",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": query},
                    ],
                    temperature=0.1,
                    max_tokens=1024,
                )
            else:
                resp = self._eval_client.chat.completions.create(
                    model=self._eval_model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": query},
                    ],
                    temperature=0.1,
                    max_tokens=1024,
                )

            usage = resp.usage
            tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0

            raw = resp.choices[0].message.content or "{}"
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw.strip())

            import json
            plan = json.loads(raw)
            plan.setdefault("gaps", "")
            plan.setdefault("sub_queries", [])
            plan.setdefault("reformulated_query", query)
            plan.setdefault("rule_ids", [])
            return plan, tokens
        except Exception as e:
            print(f"    [warn] adaptive planning failed: {e}")
            return {"gaps": "", "sub_queries": [], "reformulated_query": query, "rule_ids": []}, 0


# --- Runnable standalone ---

if __name__ == "__main__":
    retriever = AdaptiveRetriever()

    test_queries = [
        # Simple — should stay hybrid
        "What is the client's best interests rule under COBS?",
        # Ambiguous — might escalate
        "What protections exist for consumers buying financial products?",
        # Relationship — should escalate
        "Which other CASS rules reference CASS 7.11.34R?",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        result = retriever.retrieve(q)
        print(f"Approach: {result.approach}")
        print(f"Time: {result.retrieval_time_ms:.0f}ms")
        for c in result.chunks:
            print(f"  [{c.score:.4f}] {c.display_id} ({c.sourcebook})")
