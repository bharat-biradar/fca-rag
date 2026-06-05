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
from ragas.embeddings import HuggingfaceEmbeddings


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
    embeddings = HuggingfaceEmbeddings(model_name="BAAI/bge-m3")

    metric = AnswerRelevancy(llm=evaluator_llm, embeddings=embeddings)
    return metric


async def _score_one(metric, question: str, answer: str, contexts: list[str]) -> float:
    """Score a single question-answer pair."""
    try:
        result = await metric.ascore(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )
        return float(result.value) if hasattr(result, "value") else float(result)
    except Exception as e:
        print(f"    [warn] answer_relevancy failed: {e}")
        return 0.0


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m scripts.eval_answer_relevancy <results_file.json>")
        sys.exit(1)

    results_path = sys.argv[1]
    data = json.load(open(results_path))

    print(f"Scoring answer relevancy for: {results_path}")
    print(f"  Questions: {data['num_questions']}")
    print(f"  Approach: {data['approach']}")

    metric = _build_evaluator()

    scores = []
    for i, r in enumerate(data["results"]):
        print(f"  [{i+1}/{len(data['results'])}] {r['question'][:55]}...", end="", flush=True)

        # Get contexts from retrieved chunks (stored as retrieved_rule_ids, but we need text)
        # Use the generated_answer as the response
        # For contexts, build from the answer itself since we don't store chunk texts in results
        contexts = [r["generated_answer"]]  # self-referential but RAGAS uses it for embedding comparison

        score = asyncio.run(_score_one(metric, r["question"], r["generated_answer"], contexts))
        scores.append(score)
        print(f" score={score:.2f}")

        # Throttle for Bedrock
        if i < len(data["results"]) - 1:
            time.sleep(10)

    # Aggregate
    avg_score = sum(scores) / len(scores) if scores else 0.0

    # Per-type breakdown
    from collections import defaultdict

    type_scores = defaultdict(list)
    for r, s in zip(data["results"], scores):
        type_scores[r["question_type"]].append(s)

    print(f"\n{'=' * 50}")
    print(f"ANSWER RELEVANCY: {data['approach']}")
    print(f"{'=' * 50}")
    print(f"  Overall: {avg_score:.3f}")
    print(f"\n  Per type:")
    for t in sorted(type_scores):
        avg = sum(type_scores[t]) / len(type_scores[t])
        print(f"    {t:<22s} {avg:.3f} (n={len(type_scores[t])})")

    # Save alongside original results
    output_path = results_path.replace(".json", "_answer_relevancy.json")
    output = {
        "source": results_path,
        "approach": data["approach"],
        "avg_answer_relevancy": avg_score,
        "per_type": {t: sum(s) / len(s) for t, s in type_scores.items()},
        "per_question": [
            {"question": r["question"], "question_type": r["question_type"], "answer_relevancy": s}
            for r, s in zip(data["results"], scores)
        ],
    }
    json.dump(output, open(output_path, "w"), indent=2)
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
