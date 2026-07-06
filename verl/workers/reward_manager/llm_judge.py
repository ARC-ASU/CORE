# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import re
import os
import requests
import json

# LLM as Judge prompt template - two-step evaluation
CONCEPT_JUDGE_PROMPT = """You are an expert mathematics teacher evaluating a student's answer.

**Question:**
{question}

**Student's Answer:**
{answer}

**Task:**
Evaluate whether the student used the correct mathematical concept in their REASONING PROCESS (not just whether they got the right answer).

**Step 1: Identify Required Concept**
Analyze the question carefully and determine: What mathematical concept is this question actually asking about?

**Step 2: Analyze Student's Reasoning Process**
Look at the student's SOLUTION STEPS and identify:
- What concept did the student apply in their reasoning?
- What method or approach did they use?
- Does their reasoning process align with what the question is asking?

**IMPORTANT:**
- Focus on the REASONING PROCESS, not the final answer
- Even if the final answer is correct, if the student used the wrong concept/approach, judge as [[NO]]
- Even if the final answer is wrong, if the student used the correct concept/approach, judge as [[YES]]

**Step 3: Make Judgment**
Compare the required concept with the concept actually used in the student's reasoning.

**CRITICAL OUTPUT FORMAT (MUST FOLLOW EXACTLY):**
You MUST START your response with the judgment on the FIRST LINE:
Judgment: [[YES]] or [[NO]]

Then provide your reasoning:
Required Concept: [What concept the question is asking about]
Used Concept: [What concept/method the student actually applied in their reasoning]
Match: [Do they match? Did student use the right approach?]

IMPORTANT: Put the Judgment with [[YES]] or [[NO]] as the VERY FIRST LINE!

**Examples:**

Example 1 - Correct Concept Usage:
Question: What is the order of a 2×2 determinant?
Answer: "The order is determined by rows/columns. This is 2×2, so order is 2. Answer: B"
Judgment: [[YES]]
Required Concept: Determinant order (number of rows/columns)
Used Concept: Student correctly identified that order = number of rows/columns
Match: Yes, correct concept applied

Example 2 - Wrong Concept (despite correct answer):
Question: What is the order of a 2×2 determinant?
Answer: "I'll calculate: 2×5 - 3×4 = -2. The answer is B"
Judgment: [[NO]]
Required Concept: Determinant order (number of rows/columns)
Used Concept: Student calculated determinant VALUE, not order
Match: No, wrong concept - question asks for order, not value

Example 3 - Correct Concept (despite wrong answer):
Question: Solve x² + 2x + 1 = 0
Answer: "Using quadratic formula: x = (-2 ± √0)/2 = -1. But I wrote x = -2"
Judgment: [[YES]]
Required Concept: Quadratic formula
Used Concept: Student correctly applied quadratic formula
Match: Yes, right concept (calculation correct, just final transcription error)
"""


class LLMJudgeRewardManager:
    """
    Reward Manager with LLM-as-judge for concept usage evaluation.

    This manager:
    1. Computes base reward using standard grading (correct/incorrect answer)
    2. Uses the same LLM to judge if the student identified and used the correct concept
    3. Adds bonus reward (default 0.4) if concept identification and usage is correct

    Reward structure:
    - Answer correct + Concept correct: 1.0 + 0.4 = 1.4
    - Answer correct + Concept wrong:   1.0 + 0.0 = 1.0
    - Answer wrong   + Concept correct: 0.0 + 0.4 = 0.4  <- Encourages learning concepts
    - Answer wrong   + Concept wrong:   0.0 + 0.0 = 0.0
    """

    def __init__(self, tokenizer, num_examine, compute_score=None,
                 judge_model_path="Qwen/Qwen2-Math-7B",
                 concept_bonus=0.4,
                 judge_api_url="http://localhost:8001/v1/completions") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.concept_bonus = concept_bonus
        self.judge_api_url = judge_api_url

        # Use vLLM API instead of loading model locally
        print(f"[LLMJudge] Using vLLM API for judge model")
        print(f"[LLMJudge] API URL: {self.judge_api_url}")
        print(f"[LLMJudge] Model: {judge_model_path}")

        # Test API connection
        try:
            response = requests.get(judge_api_url.replace("/v1/completions", "/health"), timeout=2)
            print(f"[LLMJudge] ✅ Successfully connected to judge API")
        except Exception as e:
            print(f"[LLMJudge] ⚠️  Warning: Could not connect to judge API: {e}")
            print(f"[LLMJudge] ⚠️  Make sure judge server is running on {judge_api_url}")

    def _judge_concept_usage(self, question, answer):
        """
        Use LLM to judge if the student identified and used the correct concept.

        Args:
            question: The original question
            answer: Student's answer

        Returns:
            tuple: (judgment: bool, explanation: str)
        """
        # Format the prompt
        prompt = CONCEPT_JUDGE_PROMPT.format(
            question=question,
            answer=answer
        )

        # Build full prompt text
        full_prompt = f"You are an expert mathematics teacher.\n\n{prompt}"

        # Call vLLM API
        try:
            payload = {
                "model": "Qwen/Qwen2-Math-7B",
                "prompt": full_prompt,
                "max_tokens": 600,
                "temperature": 0.7,
                "top_p": 0.95,
                "stop": None
            }

            response = requests.post(
                self.judge_api_url,
                json=payload,
                timeout=30  # 30 second timeout
            )

            if response.status_code != 200:
                print(f"[LLMJudge Error] API returned status {response.status_code}: {response.text[:200]}")
                return False, ""

            result = response.json()
            generated_text = result['choices'][0]['text']

            # Parse the judgment
            # Look for [[YES]] or [[NO]] in the response
            if "[[YES]]" in generated_text.upper():
                return True, generated_text
            elif "[[NO]]" in generated_text.upper():
                return False, generated_text
            else:
                # If we can't parse, default to False (no bonus)
                print(f"[LLMJudge Warning] Could not parse judgment from response: {generated_text[:100]}...")
                return False, generated_text

        except requests.exceptions.Timeout:
            print(f"[LLMJudge Error] API request timed out")
            return False, ""
        except Exception as e:
            print(f"[LLMJudge Error] API request failed: {e}")
            return False, ""

    def _extract_question_from_prompt(self, prompt_str):
        """
        Extract the actual question text from the prompt string.

        The prompt typically contains system message + instructions + question.
        We need to extract just the question part.

        Args:
            prompt_str: Full prompt string

        Returns:
            str: Extracted question text, or the full prompt if extraction fails
        """
        # Try to find the question after "Question:" marker
        question_match = re.search(r"Question:\s*(.+?)(?:\n[A-E]\.|$)", prompt_str, re.DOTALL)
        if question_match:
            return question_match.group(1).strip()

        # Fallback: try to find content after "---" separator
        if "---" in prompt_str:
            parts = prompt_str.split("---")
            if len(parts) > 1:
                # Get everything after the last ---
                after_separator = parts[-1].strip()
                # Try to extract just the question part (before options)
                question_match = re.search(r"Question:\s*(.+?)(?:\n[A-E]\.|$)", after_separator, re.DOTALL)
                if question_match:
                    return question_match.group(1).strip()
                return after_separator

        # Last resort: return full prompt
        return prompt_str

    def __call__(self, data: DataProto):
        """
        Compute rewards with LLM-as-judge concept bonus.
        """
        # If there is rm score, we directly return rm score
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        score_record = []

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            # Compute base score (correctness)
            base_score = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            # Initialize final score with base score
            final_score = base_score
            concept_bonus_applied = 0.0
            judge_explanation = ""

            # Extract question from prompt
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            question_text = self._extract_question_from_prompt(prompt_str)

            # Always try to judge concept usage (even for incorrect answers, for logging purposes)
            # But only apply bonus for correct answers
            try:
                # Decode only the response part for concept judgment
                response_str = self.tokenizer.decode(valid_response_ids)

                # Judge if concept was correctly identified and used
                concept_used_correctly, explanation = self._judge_concept_usage(
                    question=question_text,
                    answer=response_str
                )

                judge_explanation = explanation

                # Apply bonus if concept usage is correct, regardless of answer correctness
                # This encourages learning correct concepts even if calculation is wrong
                if concept_used_correctly:
                    concept_bonus_applied = self.concept_bonus
                    final_score = base_score + concept_bonus_applied

            except Exception as e:
                print(f"[LLMJudge Error] Failed to judge concept usage: {e}")
                import traceback
                traceback.print_exc()
                # On error, just use base score
                pass

            # Assign reward
            reward_tensor[i, valid_response_length - 1] = final_score

            # Logging
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("\n" + "="*80)
                print(f"[Sample {i}] Data source: {data_source}")
                print(f"Base score: {base_score:.2f}, Concept bonus: {concept_bonus_applied:.2f}, Final score: {final_score:.2f}")
                if judge_explanation:
                    print(f"\nJudge Evaluation:")
                    print(judge_explanation)
                print("-"*80)
                print("Full Response:")
                print(sequences_str)
                print("="*80 + "\n")

            record = {
                "sequences_str": sequences_str,
                "ground_truth": ground_truth,
                "index": extra_info["index"] if extra_info and "index" in extra_info else None,
                "base_score": base_score,
                "concept_bonus": concept_bonus_applied,
                "final_score": final_score,
                "judge_explanation": judge_explanation
            }
            score_record.append(record)

        return reward_tensor, score_record
