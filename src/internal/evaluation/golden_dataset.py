"""Golden QA dataset utilities: load, save, verify, and inspect.

Usage: python3 -m src.internal.evaluation.golden_dataset
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_GOLDEN_PATH = "data/golden/golden_qa.json"


@dataclass
class ReferenceChunk:
    rule_id: str
    text: str


@dataclass
class GoldenQA:
    question: str
    expected_rule_ids: list[str]
    reference_chunks: list[ReferenceChunk] = field(default_factory=list)
    expected_answer_keywords: list[str] = field(default_factory=list)
    question_type: str = "simple_factual"
    sourcebook_hint: str | None = None
    difficulty: str = "easy"
    notes: str = ""


def load_golden_dataset(path: str = DEFAULT_GOLDEN_PATH) -> list[GoldenQA]:
    """Load golden QA dataset from JSON."""
    with open(path) as f:
        data = json.load(f)
    result = []
    for item in data:
        chunks = [ReferenceChunk(**c) for c in item.pop("reference_chunks", [])]
        result.append(GoldenQA(**item, reference_chunks=chunks))
    return result


def save_golden_dataset(
    dataset: list[GoldenQA], path: str = DEFAULT_GOLDEN_PATH
) -> None:
    """Save golden QA dataset to JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump([asdict(q) for q in dataset], f, indent=2)
    print(f"Saved {len(dataset)} questions → {out}")


def verify_golden_rules(
    dataset: list[GoldenQA] | None = None,
    parsed_dir: str = "data/parsed",
) -> tuple[bool, list[str]]:
    """Check all expected_rule_ids exist in the parsed rules.

    Returns (all_valid, list_of_missing_ids).
    """
    from src.internal.ingestion.parser import load_parsed_rules

    if dataset is None:
        dataset = load_golden_dataset()

    rules = load_parsed_rules(parsed_dir)
    all_ids = {r.rule_id for sb_rules in rules.values() for r in sb_rules}

    missing = []
    for q in dataset:
        for rid in q.expected_rule_ids:
            if rid not in all_ids:
                missing.append(f"{rid} (from: {q.question[:50]}...)")

    return len(missing) == 0, missing


def print_stats(dataset: list[GoldenQA]) -> None:
    """Print summary statistics for the golden dataset."""
    print(f"Total questions: {len(dataset)}")
    print()

    # By type
    type_counts = Counter(q.question_type for q in dataset)
    print("By question type:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")
    print()

    # By difficulty
    diff_counts = Counter(q.difficulty for q in dataset)
    print("By difficulty:")
    for d, c in sorted(diff_counts.items()):
        print(f"  {d}: {c}")
    print()

    # Sourcebook coverage
    sourcebooks = set()
    for q in dataset:
        for rid in q.expected_rule_ids:
            parts = rid.split(" ")
            if parts:
                sourcebooks.add(parts[0])
    print(f"Sourcebooks covered: {sorted(sourcebooks)}")
    print()

    # Answerable vs unanswerable
    answerable = sum(1 for q in dataset if q.expected_rule_ids)
    unanswerable = sum(1 for q in dataset if not q.expected_rule_ids)
    print(f"Answerable: {answerable}, Unanswerable: {unanswerable}")


if __name__ == "__main__":
    print("Loading golden dataset...")
    dataset = load_golden_dataset()
    print()

    print_stats(dataset)
    print()

    print("Verifying rule IDs against parsed rules...")
    valid, missing = verify_golden_rules(dataset)
    if valid:
        print("All expected_rule_ids verified!")
    else:
        print(f"MISSING rule IDs ({len(missing)}):")
        for m in missing:
            print(f"  - {m}")
