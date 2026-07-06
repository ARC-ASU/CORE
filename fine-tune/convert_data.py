#!/usr/bin/env python3
"""
Convert concept quizzes to LLaMA-Factory SFT format.

The quiz data includes embedded concept text (in the `related_concept` field),
so standalone concept definition files are not needed.

Output format: JSON list of {"instruction": ..., "input": ..., "output": ...}
"""

import json
import argparse
from pathlib import Path


def load_quizzes(quiz_file: str):
    """Load quizzes from JSONL file."""
    quizzes = []
    with open(quiz_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                quizzes.append(json.loads(line))
    print(f"Loaded {len(quizzes)} quizzes from {quiz_file}")
    return quizzes


def convert_to_sft_format(quizzes):
    """Convert to LLaMA-Factory instruction format."""
    converted = []

    for item in quizzes:
        question = item.get("question", item.get("original_question", ""))
        options = item.get("options", [])
        answer = item.get("answer", "")
        explanation = item.get("explanation", "")
        concept = item.get("related_concept", "")

        options_text = "\n".join(options) if options else ""
        input_text = f"{question}\n\n{options_text}" if options_text else question

        # Build output with explanation + boxed answer
        answer_letter = answer.split(".")[0].strip() if "." in answer else answer.strip()
        if explanation:
            output_text = f"{explanation}\n\nThe final answer is \\boxed{{{answer_letter}}}."
        else:
            output_text = f"The final answer is \\boxed{{{answer_letter}}}."

        converted.append({
            "instruction": "You are a mathematics expert. Solve the following problem step by step.",
            "input": input_text,
            "output": output_text,
        })

    return converted


def main():
    parser = argparse.ArgumentParser(
        description="Convert concept quiz data to LLaMA-Factory SFT format"
    )
    parser.add_argument(
        "--quizzes", default="../data/quizzes/concept_quizzes.jsonl",
        help="Path to concept_quizzes.jsonl"
    )
    parser.add_argument(
        "--output", "-o", default="LLaMA-Factory/data/conceptandquiz_sft.json",
        help="Output JSON file for LLaMA-Factory"
    )
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    quizzes = load_quizzes(args.quizzes)
    converted = convert_to_sft_format(quizzes)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"\nConversion complete:")
    print(f"  Quizzes:  {len(quizzes)}")
    print(f"  Total:    {len(converted)}")
    print(f"  Output:   {args.output}")


if __name__ == "__main__":
    main()
