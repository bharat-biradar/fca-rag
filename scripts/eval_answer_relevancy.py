"""Post-processing: score answer relevancy on existing eval results.

Reads generated_answer + question from result files, scores with RAGAS AnswerRelevancy.
No retrieval or generation needed — just scoring saved answers.

Usage: python3 -m scripts.eval_answer_relevancy results/eval_agentic_v2_chunks_v2_mini_full_bedrock.json
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy
from ragas.embeddings import HuggingFaceEmbeddings


def _build_evaluator():
    """Build RAGAS AnswerRelevancy metric with Bedrock LLM + local embeddings."""
    import litellm

    litellm.suppress_debug_info = True
    import logging

    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    class _BedrockAsyncOpenAI(AsyncOpenAI):
        def __init__(self, bedrock_model):
            self.bedrock_model = bedrock_model
            self._is_async = True

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        async def create(self, **kwargs):
            kwargs.pop("model", None)
            kwargs.pop("top_p", None)
            return await litellm.acompletion(model=self.bedrock_model, **kwargs)

    bedrock_model = "bedrock/anthropic.claude-3-haiku-20240307-v1:0"
    client = _BedrockAsyncOpenAI(bedrock_model)
    evaluator_llm = llm_factory(bedrock_model, client=client)

    # Use local HuggingFace embeddings (BGE-M3 already cached)
    embeddings = HuggingFaceEmbeddings(model="BAAI/bge-m3")

    metric = AnswerRelevancy(llm=evaluator_llm, embeddings=embeddings)
    return metric


async def _score_one(metric, question: str, answer: str, retries: int = 3) -> float:
    """Score a single question-answer pair with retry on rate limits."""
    for attempt in range(retries):
        try:
            result = await metric.ascore(
                user_input=question,
                response=answer,
            )
            return float(result.value) if hasattr(result, "value") else float(result)
        except Exception as e:
            if "RateLimit" in str(type(e).__name__) or "Too many tokens" in str(e):
                wait = 10 * (attempt + 1)
                print(f" [rate limited, waiting {wait}s]", end="", flush=True)
                await asyncio.sleep(wait)
                continue
            print(f"    [warn] answer_relevancy failed: {e}")
            return 0.0
    print(f"    [warn] exhausted retries")
    return 0.0


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m scripts.eval_answer_relevancy <file1.json> [file2.json] [file3.json]")
        sys.exit(1)

    result_files = sys.argv[1:]

    for results_path in result_files:
        _run_one(results_path)
        print()


def _run_one(results_path: str):
    with open(results_path) as f:
        data = json.load(f)

    print(f"Scoring answer relevancy for: {results_path}")
    print(f"  Questions: {data['num_questions']}")
    print(f"  Approach: {data['approach']}")

    metric = _build_evaluator()

    async def _run_all():
        scores = []
        for i, r in enumerate(data["results"]):
            print(f"  [{i+1}/{len(data['results'])}] {r['question'][:55]}...", end="", flush=True)

            score = await _score_one(metric, r["question"], r["generated_answer"])
            scores.append(score)
            print(f" score={score:.2f}")

            # Throttle for Bedrock — sequential, one at a time
            if i < len(data["results"]) - 1:
                await asyncio.sleep(5)
        return scores

    scores = asyncio.run(_run_all())

    # Aggregate — separate answerable vs unanswerable
    from collections import defaultdict

    type_scores = defaultdict(list)
    answerable_scores = []
    unanswerable_scores = []
    for r, s in zip(data["results"], scores):
        type_scores[r["question_type"]].append(s)
        if r["question_type"] == "unanswerable":
            unanswerable_scores.append(s)
        else:
            answerable_scores.append(s)

    avg_answerable = sum(answerable_scores) / len(answerable_scores) if answerable_scores else 0.0
    avg_unanswerable = sum(unanswerable_scores) / len(unanswerable_scores) if unanswerable_scores else 0.0
    # Penalize if unanswerable questions get high relevancy (= hallucinated answers)
    hallucination_penalty = avg_unanswerable * 0.5
    adjusted_score = avg_answerable - hallucination_penalty
    raw_avg = sum(scores) / len(scores) if scores else 0.0

    print(f"\n{'=' * 50}")
    print(f"ANSWER RELEVANCY: {data['approach']}")
    print(f"{'=' * 50}")
    print(f"  Raw overall:       {raw_avg:.3f}")
    print(f"  Answerable avg:    {avg_answerable:.3f} ({len(answerable_scores)} questions)")
    print(f"  Unanswerable avg:  {avg_unanswerable:.3f} ({len(unanswerable_scores)} questions)")
    print(f"  Hallucination pen: -{hallucination_penalty:.3f}")
    print(f"  Adjusted score:    {adjusted_score:.3f}")
    print(f"\n  Per type:")
    for t in sorted(type_scores):
        avg = sum(type_scores[t]) / len(type_scores[t])
        print(f"    {t:<22s} {avg:.3f} (n={len(type_scores[t])})")

    # Save alongside original results
    output_path = results_path.replace(".json", "_answer_relevancy.json")
    output = {
        "source": results_path,
        "approach": data["approach"],
        "raw_avg_answer_relevancy": raw_avg,
        "avg_answerable": avg_answerable,
        "avg_unanswerable": avg_unanswerable,
        "hallucination_penalty": hallucination_penalty,
        "adjusted_score": adjusted_score,
        "per_type": {t: sum(s) / len(s) for t, s in type_scores.items()},
        "per_question": [
            {"question": r["question"], "question_type": r["question_type"], "answer_relevancy": s}
            for r, s in zip(data["results"], scores)
        ],
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
