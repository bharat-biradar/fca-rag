"""Evaluation harness: run retrieval + generation over golden dataset, compute metrics.

Uses RAGAS (v0.4.3 collections API) for faithfulness, context_recall, context_precision.
Adds custom citation_accuracy and token tracking on top.

Usage: python3 -m src.internal.evaluation.eval_harness
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)

from src.config import Settings, settings
from src.internal.evaluation.golden_dataset import GoldenQA, load_golden_dataset
from src.internal.generation.llm import LLMClient
from src.internal.generation.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    extract_citations,
)
from src.internal.retrieval.base import BaseRetriever


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SingleEvalResult:
    question: str
    question_type: str
    difficulty: str
    # Retrieval
    retrieved_rule_ids: list[str]
    expected_rule_ids: list[str]
    # Generation
    generated_answer: str
    cited_rule_ids: list[str]
    # RAGAS metrics
    faithfulness: float
    context_recall: float
    context_precision: float
    # Custom metrics
    citation_accuracy: float
    # Token usage
    prompt_tokens: int
    completion_tokens: int


@dataclass
class EvalSummary:
    approach: str
    num_questions: int
    # RAGAS averages
    avg_faithfulness: float
    avg_context_recall: float
    avg_context_precision: float
    # Custom averages
    avg_citation_accuracy: float
    # Token totals
    total_prompt_tokens: int
    total_completion_tokens: int
    # Breakdown
    per_type_scores: dict[str, dict[str, float]]
    results: list[SingleEvalResult]


# ---------------------------------------------------------------------------
# RAGAS evaluator
# ---------------------------------------------------------------------------

def _build_ragas_metrics(cfg: Settings = settings, use_ollama: bool = False):
    """Create RAGAS metric instances with a fast evaluator LLM."""
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        async_client = AsyncOpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=gemini_key,
        )
        evaluator_llm = llm_factory("gemini-2.5-flash", client=async_client, max_tokens=8192)
    elif use_ollama:
        async_client = AsyncOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        evaluator_llm = llm_factory("cogito:8b", client=async_client)
    else:
        async_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=cfg.openrouter_api_key,
        )
        evaluator_llm = llm_factory(cfg.fallback_model, client=async_client)

    return {
        # Faithfulness disabled for speed (8-10 internal LLM calls per question).
        # Enable when a faster evaluator is available (e.g., local Ollama or Anthropic API).
        # "faithfulness": Faithfulness(llm=evaluator_llm),
        "context_recall": ContextRecall(llm=evaluator_llm),
        "context_precision": ContextPrecisionWithReference(llm=evaluator_llm),
    }


async def _score_ragas(
    metrics: dict,
    user_input: str,
    response: str,
    retrieved_contexts: list[str],
    reference: str,
) -> dict[str, float]:
    """Score a single sample with all RAGAS metrics (concurrently)."""

    async def _safe_score(name, coro):
        try:
            result = await coro
            return name, float(result.value) if hasattr(result, "value") else float(result)
        except Exception as e:
            print(f"    [warn] {name} failed: {e}")
            return name, 0.0

    # Each metric takes different arguments
    metric_kwargs = {
        "faithfulness": {
            "user_input": user_input, "response": response, "retrieved_contexts": retrieved_contexts,
        },
        "context_recall": {
            "user_input": user_input, "retrieved_contexts": retrieved_contexts, "reference": reference,
        },
        "context_precision": {
            "user_input": user_input, "retrieved_contexts": retrieved_contexts, "reference": reference,
        },
    }

    # Run sequentially to avoid rate limits on free tier
    results = []
    for name, metric in metrics.items():
        kwargs = metric_kwargs[name]
        results.append(await _safe_score(name, metric.ascore(**kwargs)))

    # Fill defaults for disabled metrics
    scores = dict(results)
    for key in ("faithfulness", "context_recall", "context_precision"):
        scores.setdefault(key, 0.0)
    return scores


# ---------------------------------------------------------------------------
# Custom metrics
# ---------------------------------------------------------------------------

def _normalize_rule_id(rid: str) -> str:
    """Strip type suffix and normalize whitespace for comparison."""
    rid = rid.strip()
    rid = re.sub(r"[RGDEUK]{1,2}$", "", rid).strip()
    rid = re.sub(r"\s+", " ", rid)
    return rid


def _citation_accuracy(cited_ids: list[str], expected_ids: list[str]) -> float:
    """Fraction of cited rule IDs that match expected."""
    if not cited_ids:
        return 0.0
    norm_cited = {_normalize_rule_id(r) for r in cited_ids}
    norm_expected = {_normalize_rule_id(r) for r in expected_ids}
    if not norm_cited:
        return 0.0
    return len(norm_cited & norm_expected) / len(norm_cited)


# ---------------------------------------------------------------------------
# Single question evaluation
# ---------------------------------------------------------------------------

async def evaluate_single(
    qa: GoldenQA,
    retriever: BaseRetriever,
    llm: LLMClient,
    ragas_metrics: dict,
) -> SingleEvalResult:
    """Run retrieval + generation + scoring for one question."""
    # 1. Retrieve
    result = retriever.retrieve(qa.question)
    retrieved_ids = [c.display_id for c in result.chunks]
    retrieved_texts = [c.text for c in result.chunks]

    # 2. Generate
    user_prompt = build_user_prompt(qa.question, result.chunks)
    response = llm.generate(SYSTEM_PROMPT, user_prompt)
    cited_ids = extract_citations(response.text)

    # 3. Build reference from expected keywords (for RAGAS context_recall/precision)
    reference = f"The answer should reference rules: {', '.join(qa.expected_rule_ids)}. "
    if qa.expected_answer_keywords:
        reference += f"Key concepts: {', '.join(qa.expected_answer_keywords)}."

    # Debug: show what we're comparing
    print(f"    retrieved: {retrieved_ids[:3]}")
    print(f"    expected:  {qa.expected_rule_ids}")
    print(f"    cited:     {cited_ids}")

    # 4. RAGAS scoring
    ragas_scores = await _score_ragas(
        ragas_metrics,
        user_input=qa.question,
        response=response.text,
        retrieved_contexts=retrieved_texts,
        reference=reference,
    )

    # 5. Custom metrics
    cite_acc = _citation_accuracy(cited_ids, qa.expected_rule_ids)

    return SingleEvalResult(
        question=qa.question,
        question_type=qa.question_type,
        difficulty=qa.difficulty,
        retrieved_rule_ids=retrieved_ids,
        expected_rule_ids=qa.expected_rule_ids,
        generated_answer=response.text,
        cited_rule_ids=cited_ids,
        faithfulness=ragas_scores["faithfulness"],
        context_recall=ragas_scores["context_recall"],
        context_precision=ragas_scores["context_precision"],
        citation_accuracy=cite_acc,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
    )


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------

def run_eval(
    retriever: BaseRetriever,
    llm: LLMClient,
    golden: list[GoldenQA],
    approach_name: str,
    cfg: Settings = settings,
    use_ollama: bool = False,
    start_from: int = 0,
) -> EvalSummary:
    """Run full evaluation over the golden dataset."""
    ragas_metrics = _build_ragas_metrics(cfg, use_ollama=use_ollama)
    results: list[SingleEvalResult] = []

    # Resume from partial results if starting mid-run
    Path("results").mkdir(exist_ok=True)
    incremental_path = f"results/eval_{approach_name}_partial.json"
    if start_from > 0 and Path(incremental_path).exists():
        partial_data = json.load(open(incremental_path))
        results = [SingleEvalResult(**r) for r in partial_data["results"][:start_from]]
        print(f"  Resuming from question {start_from + 1} ({len(results)} results loaded)")

    for i, qa in enumerate(golden):
        if i < start_from:
            continue
        print(f"  [{i+1}/{len(golden)}] {qa.question[:60]}...")
        result = asyncio.run(evaluate_single(qa, retriever, llm, ragas_metrics))
        results.append(result)
        print(
            f"    faith={result.faithfulness:.2f}  "
            f"ctx_recall={result.context_recall:.2f}  "
            f"ctx_prec={result.context_precision:.2f}  "
            f"cite_acc={result.citation_accuracy:.2f}  "
            f"tokens={result.prompt_tokens}+{result.completion_tokens}"
        )

        # Save incrementally after each question
        partial = _aggregate(results, approach_name)
        with open(incremental_path, "w") as f:
            json.dump(asdict(partial), f, indent=2)

        # Throttle to avoid OpenRouter rate limits on free tier
        if i < len(golden) - 1:
            time.sleep(3)

    return _aggregate(results, approach_name)


def _aggregate(results: list[SingleEvalResult], approach: str) -> EvalSummary:
    """Aggregate individual results into a summary."""
    n = len(results)
    avg = lambda vals: sum(vals) / len(vals) if vals else 0.0

    # Per-type breakdown
    type_groups: dict[str, list[SingleEvalResult]] = {}
    for r in results:
        type_groups.setdefault(r.question_type, []).append(r)

    per_type_scores = {}
    for qtype, group in type_groups.items():
        per_type_scores[qtype] = {
            "faithfulness": avg([r.faithfulness for r in group]),
            "context_recall": avg([r.context_recall for r in group]),
            "context_precision": avg([r.context_precision for r in group]),
            "citation_accuracy": avg([r.citation_accuracy for r in group]),
            "count": len(group),
        }

    return EvalSummary(
        approach=approach,
        num_questions=n,
        avg_faithfulness=avg([r.faithfulness for r in results]),
        avg_context_recall=avg([r.context_recall for r in results]),
        avg_context_precision=avg([r.context_precision for r in results]),
        avg_citation_accuracy=avg([r.citation_accuracy for r in results]),
        total_prompt_tokens=sum(r.prompt_tokens for r in results),
        total_completion_tokens=sum(r.completion_tokens for r in results),
        per_type_scores=per_type_scores,
        results=results,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(summary: EvalSummary, path: str | None = None):
    """Save eval results to JSON."""
    if path is None:
        Path("results").mkdir(exist_ok=True)
        path = f"results/eval_{summary.approach}.json"
    with open(path, "w") as f:
        json.dump(asdict(summary), f, indent=2)
    print(f"  Results saved to {path}")


def print_summary(summary: EvalSummary):
    """Print a formatted summary."""
    print(f"\n{'=' * 65}")
    print(f"  EVALUATION: {summary.approach}")
    print(f"{'=' * 65}")
    print(f"  Questions:           {summary.num_questions}")
    print(f"  Avg Faithfulness:    {summary.avg_faithfulness:.3f}")
    print(f"  Avg Context Recall:  {summary.avg_context_recall:.3f}")
    print(f"  Avg Context Prec:    {summary.avg_context_precision:.3f}")
    print(f"  Avg Citation Acc:    {summary.avg_citation_accuracy:.3f}")
    print(f"  Total Tokens:        {summary.total_prompt_tokens} in / {summary.total_completion_tokens} out")

    print(f"\n  Per question type:")
    print(f"  {'Type':<25s} {'Faith':>6s} {'Recall':>7s} {'Prec':>6s} {'Cite':>6s} {'N':>4s}")
    print(f"  {'-'*54}")
    for qtype in sorted(summary.per_type_scores):
        s = summary.per_type_scores[qtype]
        print(
            f"  {qtype:<25s} "
            f"{s['faithfulness']:>6.3f} "
            f"{s['context_recall']:>7.3f} "
            f"{s['context_precision']:>6.3f} "
            f"{s['citation_accuracy']:>6.3f} "
            f"{int(s['count']):>4d}"
        )
    print(f"{'=' * 65}")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    from src.internal.retrieval.hybrid_rerank import HybridRerankRetriever

    # Flags: --mini (18 questions), --ollama (local evaluator), --agentic (use agentic retriever)
    mini = "--mini" in sys.argv
    use_ollama = "--ollama" in sys.argv
    use_agentic = "--agentic" in sys.argv
    start_from = 0
    for arg in sys.argv:
        if arg.startswith("--start="):
            start_from = int(arg.split("=")[1]) - 1  # 1-indexed input
    dataset_path = "data/golden/golden_qa_mini.json" if mini else "data/golden/golden_qa.json"

    print(f"Loading golden dataset ({dataset_path})...")
    golden = load_golden_dataset(dataset_path)
    print(f"  {len(golden)} questions")
    gemini = bool(os.getenv("GEMINI_API_KEY", ""))
    evaluator = "Gemini Flash" if gemini else ("Ollama" if use_ollama else "OpenRouter")
    print(f"  RAGAS evaluator: {evaluator}")

    if use_agentic:
        from src.internal.retrieval.agentic import AgenticRetriever
        agent_llm = "gemini" if gemini else "openrouter"
        approach = f"agentic_{agent_llm}"
        print(f"  Retriever: Agentic RAG ({agent_llm}, max {settings.max_agent_steps} steps)\n")
        retriever = AgenticRetriever()
    else:
        approach = "hybrid_rerank"
        print(f"  Retriever: Hybrid + Rerank\n")
        retriever = HybridRerankRetriever()

    llm = LLMClient()

    print(f"Running eval: {approach}\n")
    summary = run_eval(retriever, llm, golden, approach, use_ollama=use_ollama, start_from=start_from)

    print_summary(summary)
    save_results(summary)
