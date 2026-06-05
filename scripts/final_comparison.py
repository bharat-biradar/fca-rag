"""Final comparison across all approaches. Run after all evals complete.

Usage: python3 -m scripts.final_comparison
"""

import json
import os
from pathlib import Path


def load_if_exists(path):
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    results = {
        "Hybrid": load_if_exists("results/eval_hybrid_rerank_chunks_v2_mini_minilm_bedrock.json"),
        "Graph": load_if_exists("results/eval_graph_rag_chunks_v2_mini_minilm_bedrock.json"),
        "Agentic v2": load_if_exists("results/eval_agentic_v2_chunks_v2_mini_full_bedrock.json"),
        "Adaptive": None,  # will check multiple names
    }

    # Find adaptive result
    for name in ["eval_adaptive_chunks_v2_mini_binary_final.json",
                 "eval_adaptive_chunks_v2_mini_binary_bedrock.json",
                 "eval_adaptive_chunks_v2_mini_bedrock.json"]:
        r = load_if_exists(f"results/{name}")
        if r:
            results["Adaptive"] = r
            break

    # Filter out missing
    results = {k: v for k, v in results.items() if v}

    if not results:
        print("No result files found.")
        return

    # --- Overall ---
    print("=" * 75)
    print("  FINAL COMPARISON")
    print("=" * 75)

    header = f"{'Metric':<22s}" + "".join(f"{k:>12s}" for k in results)
    print(f"\n{header}")
    print("-" * len(header))

    for metric, label in [
        ("avg_context_recall", "Context Recall"),
        ("avg_context_precision", "Context Precision"),
        ("avg_citation_accuracy", "Citation Accuracy"),
    ]:
        vals = {k: v[metric] for k, v in results.items()}
        best = max(vals.values())
        line = f"{label:<22s}"
        for k in results:
            star = "*" if vals[k] == best else " "
            line += f"{vals[k]:>11.3f}{star}"
        print(line)

    # Token efficiency
    print()
    for label, key in [("Prompt Tokens", "total_prompt_tokens"),
                       ("Completion Tokens", "total_completion_tokens"),
                       ("Planning Tokens", "total_planning_tokens")]:
        line = f"{label:<22s}"
        for k in results:
            val = results[k].get(key, 0)
            line += f"{val:>12d}"
        print(line)

    # Per-question token efficiency
    print()
    line = f"{'Tokens/Question':<22s}"
    for k in results:
        r = results[k]
        total = r.get("total_prompt_tokens", 0) + r.get("total_completion_tokens", 0) + r.get("total_planning_tokens", 0)
        per_q = total / r["num_questions"]
        line += f"{per_q:>12.0f}"
    print(line)

    line = f"{'Tokens/Recall':<22s}"
    for k in results:
        r = results[k]
        total = r.get("total_prompt_tokens", 0) + r.get("total_completion_tokens", 0) + r.get("total_planning_tokens", 0)
        recall = r["avg_context_recall"]
        efficiency = total / recall if recall > 0 else float("inf")
        line += f"{efficiency:>12.0f}"
    print(line)

    # --- Per Type ---
    print(f"\n{'=' * 75}")
    print("  PER TYPE RECALL")
    print(f"{'=' * 75}")

    types = ["simple_factual", "keyword_specific", "cross_sourcebook", "ambiguous",
             "scenario", "exception_negation", "relationship", "unanswerable"]

    header = f"{'Type':<22s}" + "".join(f"{k:>12s}" for k in results)
    print(f"\n{header}")
    print("-" * len(header))

    for t in types:
        vals = {}
        for k in results:
            vals[k] = results[k]["per_type_scores"].get(t, {}).get("context_recall", 0)
        best = max(vals.values())
        line = f"{t:<22s}"
        for k in results:
            star = "*" if vals[k] == best and best > 0 else " "
            line += f"{vals[k]:>11.3f}{star}"
        print(line)

    # --- Adaptive Routing ---
    if "Adaptive" in results:
        ad = results["Adaptive"]
        hybrid_count = sum(1 for r in ad["results"] if "hybrid" in r.get("retrieval_approach", ""))
        agentic_count = sum(1 for r in ad["results"] if "agentic" in r.get("retrieval_approach", ""))
        unknown = ad["num_questions"] - hybrid_count - agentic_count

        print(f"\n{'=' * 75}")
        print("  ADAPTIVE ROUTING")
        print(f"{'=' * 75}")
        print(f"  Hybrid path:  {hybrid_count}/{ad['num_questions']}")
        print(f"  Agentic path: {agentic_count}/{ad['num_questions']}")
        if unknown:
            print(f"  Unknown:      {unknown}/{ad['num_questions']}")

    # --- Answer Relevancy ---
    ar_files = {
        "Hybrid": "results/eval_hybrid_rerank_chunks_v2_mini_minilm_bedrock_answer_relevancy.json",
        "Graph": "results/eval_graph_rag_chunks_v2_mini_minilm_bedrock_answer_relevancy.json",
        "Agentic v2": "results/eval_agentic_v2_chunks_v2_mini_full_bedrock_answer_relevancy.json",
    }

    ar_results = {}
    for k, path in ar_files.items():
        r = load_if_exists(path)
        if r:
            ar_results[k] = r

    if ar_results:
        print(f"\n{'=' * 75}")
        print("  ANSWER RELEVANCY")
        print(f"{'=' * 75}")

        header = f"{'Metric':<22s}" + "".join(f"{k:>12s}" for k in ar_results)
        print(f"\n{header}")
        print("-" * len(header))

        for metric, label in [
            ("avg_answer_relevancy", "Raw Overall"),
            ("avg_answerable", "Answerable Avg"),
            ("avg_unanswerable", "Unanswerable Avg"),
            ("adjusted_score", "Adjusted Score"),
        ]:
            line = f"{label:<22s}"
            for k in ar_results:
                val = ar_results[k].get(metric, ar_results[k].get("avg_answer_relevancy", 0))
                line += f"{val:>12.3f}"
            print(line)

    print()


if __name__ == "__main__":
    main()
