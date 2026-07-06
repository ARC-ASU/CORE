# Data Pipeline

This directory documents the data curation pipeline used to generate concept-aligned quizzes for CORE training.

> **Note**: Quiz generation and validation involve LLM calls (Qwen2.5-72B-Instruct for generation, GPT-4o for validation), which are inherently non-deterministic. We provide the **final curated data** in `data/` and document the prompts and pipeline below for transparency.

## Pipeline Overview

```
Textbook (Advanced Algebra, 3rd Ed.)
    ↓  OCR + manual correction + GPT-4o translation (Chinese → English)
236 concept definitions + 703 examples + 140 exercises
    ↓  Quiz generation (Qwen2.5-72B-Instruct, 5-10 per concept)
~1,200 candidate quizzes
    ↓  GPT-4o validation (6 dimensions)
1,110 high-quality concept-aligned quizzes
    ↓  convert_to_verl_format.py
train.parquet / val.parquet (verl format)
```

## Step 1: Quiz Generation

We prompted **Qwen2.5-72B-Instruct** to generate 5–10 multiple-choice quizzes for each of the 236 concept texts. The model was served via vLLM and queried with the following prompt template:

```
You are an expert in mathematics education. Your task is to create a high-quality
quiz based on the provided "Learning Material".

--- Learning Material Starts ---
{concept_title}

{concept_text}
--- Learning Material Ends ---

Please strictly follow these requirements and generate the quiz in the specified
JSON format.

Requirements:
- Analyze the Entire Material: Base your questions on the entire "Learning Material"
  provided above. The material's first line is its title.
- Difficulty: intermediate
- Question Type: multiple_choice
- Number of Questions: Based on the provided material, generate between 5 to 10
  high-quality questions. Aim for more questions for complex, core concepts and
  fewer for simple definitions.
- Question Design Philosophy: Do not just test factual recall. Create questions
  that test for deeper understanding. This includes:
    - Application: Questions that require applying the concept to a simple,
      concrete problem.
    - Common Misconceptions: Design incorrect options based on common mistakes
      or misunderstandings of the concept.
    - The "Why": Questions that probe the reasoning behind the concept or its
      connection to other ideas.
- Math Notation: Use single dollar signs (e.g., $ ... $) for all inline
  mathematical formulas.
- Questions must be strictly based on the "Learning Material" provided above.
- Use English for all content.

JSON Format:
{
    "title": "Quiz on {concept_title}",
    "concept": "{concept_title}",
    "difficulty": "intermediate",
    "questions": [
        {
            "id": 1,
            "question": "Question content here",
            "type": "multiple_choice",
            "options": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"],
            "correct_answer": "Correct answer here",
            "explanation": "Detailed explanation here",
            "tags": ["relevant", "tags"]
        }
    ]
}
```

This produced a candidate pool of approximately **1,200 quizzes**.

## Step 2: Quiz Validation (GPT-4o)

Each candidate quiz was validated by **GPT-4o** using the following prompt, which evaluates six quality dimensions:

```
You are a mathematics education expert and quiz quality reviewer. Please carefully
analyze the following mathematical quiz question and evaluate it from these aspects:

1. Question Clarity: Is the question statement clear, accurate, and unambiguous?
2. Option Quality: Are the options well-designed? Are there any obvious errors or
   duplicates?
3. Answer Correctness: Is the marked correct answer actually correct?
4. Uniqueness: Is there only one correct answer? Are all other options definitely
   wrong?
5. Explanation Accuracy: Is the provided explanation correct, complete, and easy
   to understand?
6. Mathematical Accuracy: Are all mathematical expressions, calculations, and
   concepts correct?

Please respond strictly in the following JSON format only (no extra commentary):
{
    "overall_quality": "excellent/good/fair/poor",
    "issues": [
        {
            "type": "question_clarity/option_quality/answer_correctness/...",
            "severity": "critical/major/minor",
            "description": "Specific description of the issue"
        }
    ],
    "correct_answer": "If the original answer is wrong, provide the correct
                        answer, otherwise null",
    "suggestions": "Improvement suggestions",
    "confidence": "Confidence level in your assessment (1-10)"
}

Quiz Information:
Concept: {concept_title}
Question: {question}
Options: {options}
Marked Answer: {answer}
Explanation: {explanation}
```

We discarded 90 quizzes rated "Fair" or "Poor" with high confidence, yielding a final set of **1,110 high-quality quizzes**.

## Step 3: Convert to Training Format

```bash
python scripts/data/convert_to_verl_format.py
```

This converts the JSONL quiz data into verl-compatible parquet files (`train.parquet`, `val.parquet`) with chat-formatted prompts.

## Output Data

All final data is provided in `data/`:
- `data/train.parquet` — 1,110 training quizzes (verl format)
- `data/val.parquet` — 111 validation quizzes (verl format)
- `data/quizzes/concept_quizzes.jsonl` — raw quiz data with concept labels
- `data/textbook/` — source textbook corpus (concepts, examples, exercises) — **not included in this release** due to textbook copyright; see the note in [`data/README.md`](../../data/README.md)
