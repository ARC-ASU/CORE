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

"""
GPT-4o Baseline Reward Manager

When encountering an all-wrong group:
- Don't use concept-enhanced prompts
- Directly call GPT-4o API to generate 2 new responses
- The prompt only contains the original question (no concept information)
- Give these 2 new responses a 0.4 bonus reward
- Replace 2 of the wrong answers in the all-wrong group
"""

from collections import defaultdict
import torch
import random
import re
import os
import time
from verl import DataProto
from verl.utils.reward_score import _default_compute_score


# GPT-4o prompt template - without concept information, only solution steps
GPT4O_BASELINE_PROMPT = """You are a helpful assistant that solves multiple-choice math questions with step-by-step reasoning.

**Question:**
{question}

**Options:**
{options}

**Instructions:**
- Provide a clear, step-by-step solution to the problem
- Show your reasoning and calculations
- At the end, clearly state your answer choice (A, B, C, D, or E)
- Do NOT include any discussion about mathematical concepts or definitions
- Focus only on solving this specific problem

**Important:** Your response should contain ONLY the solution steps and the final answer. Do not mention concepts, theorems, or general mathematical principles.

Answer: """


class GPT4oBaselineRewardManager:
    """
    GPT-4o Baseline Reward Manager for all-wrong group intervention.

    When a group of responses (usually n=4) are all incorrect:
    - Use GPT-4o API to generate 2 new responses (without concept enhancement)
    - Randomly replace 2 of the 4 wrong responses
    - Give these 2 new responses +0.4 bonus reward

    Reward structure:
    - Original wrong answer: 0.0
    - GPT-4o generated answer (correct): 1.0 + 0.4 = 1.4
    - GPT-4o generated answer (wrong): 0.0 + 0.4 = 0.4
    """

    def __init__(self, tokenizer, num_examine, compute_score=None,
                 openai_api_key: str = None,
                 model_name: str = "gpt-4o",
                 bonus_reward: float = 0.4,
                 max_retries: int = 3) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.bonus_reward = bonus_reward
        self.model_name = model_name
        self.max_retries = max_retries

        # Get OpenAI API key from environment or parameter
        self.api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass openai_api_key parameter.")

        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            print(f"🤖 GPT4O_BASELINE: Initialized with model {self.model_name}")
            print(f"💰 GPT4O_BASELINE: Bonus reward = {self.bonus_reward}")
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        self.step_counter = 0

    def _normalize_text(self, s: str) -> str:
        """Normalize text for comparison"""
        normalized = s.strip()
        normalized = normalized.replace('\\\\\\\\', '\\\\').replace('\\\\begin', '\\begin').replace('\\\\end', '\\end')
        normalized = ' '.join(normalized.split())
        return normalized

    def _extract_question_key(self, prompt_str: str) -> str:
        """Extract the key of the question for grouping"""
        try:
            # Find the actual question after "---" separator
            if "---" in prompt_str and "Question:" in prompt_str:
                after_separator = prompt_str.split("---", 1)[-1]
                if "Question:" in after_separator and "A." in after_separator:
                    start = after_separator.find("Question:") + len("Question:")
                    end = after_separator.find("A.", start)
                    if end > start:
                        question_text = after_separator[start:end].strip()
                        return self._normalize_text(question_text)

            # Fallback: use the last "Question:"
            elif "Question:" in prompt_str and "A." in prompt_str:
                question_positions = [i for i in range(len(prompt_str)) if prompt_str[i:].startswith("Question:")]
                if question_positions:
                    last_question_pos = question_positions[-1] + len("Question:")
                    end = prompt_str.find("A.", last_question_pos)
                    if end > last_question_pos:
                        question_text = prompt_str[last_question_pos:end].strip()
                        return self._normalize_text(question_text)
        except Exception as e:
            print(f"⚠️  Warning: question key extraction failed: {e}")
        return str(hash(prompt_str))

    def _extract_question_and_options(self, prompt_str: str) -> tuple:
        """
        Extract the question text and options from the prompt.

        Returns:
            tuple: (question_text, options_text)
        """
        try:
            # Find content after "---" separator
            if "---" in prompt_str:
                after_separator = prompt_str.split("---", 1)[-1]
            else:
                after_separator = prompt_str

            # Extract question
            question_text = ""
            if "Question:" in after_separator:
                start = after_separator.find("Question:") + len("Question:")
                end = after_separator.find("A.", start)
                if end > start:
                    question_text = after_separator[start:end].strip()

            # Extract options (A. ... B. ... C. ... D. ... E. ...)
            options_text = ""
            if "A." in after_separator:
                options_start = after_separator.find("A.")
                # Find where options end (usually before "Answer:" or end of string)
                options_end = len(after_separator)
                if "Answer:" in after_separator[options_start:]:
                    options_end = after_separator.find("Answer:", options_start)

                options_text = after_separator[options_start:options_end].strip()

            return question_text, options_text

        except Exception as e:
            print(f"⚠️  Warning: question/options extraction failed: {e}")
            return prompt_str, ""

    def _generate_gpt4o_responses(self, question: str, options: str, n: int = 2) -> list:
        """
        Generate responses using GPT-4o API.

        Args:
            question: The question text
            options: The options text (A. ... B. ... etc.)
            n: Number of responses to generate

        Returns:
            list: Generated responses (may be fewer than n if some fail)
        """
        # Format the prompt
        prompt = GPT4O_BASELINE_PROMPT.format(
            question=question,
            options=options
        )

        responses = []
        for i in range(n):
            for attempt in range(self.max_retries):
                try:
                    print(f"🔄 GPT4O_BASELINE: Generating response {i+1}/{n} (attempt {attempt+1}/{self.max_retries})")

                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that solves math problems step-by-step."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1024,
                        top_p=0.9
                    )

                    response_text = completion.choices[0].message.content.strip()
                    responses.append(response_text)
                    print(f"✅ GPT4O_BASELINE: Successfully generated response {i+1}/{n}")
                    break  # Success, move to next response

                except Exception as e:
                    print(f"❌ GPT4O_BASELINE: API call failed (attempt {attempt+1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        print(f"⚠️  GPT4O_BASELINE: Failed to generate response {i+1} after {self.max_retries} attempts")

        return responses

    def __call__(self, data: DataProto):
        """
        Compute rewards with GPT-4o baseline intervention for all-wrong groups.
        """
        # If rm_scores provided, return directly
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        score_record = []
        already_print_data_sources = {}

        # Statistics for wandb
        wandb_stats = {
            'gpt4o_baseline': {
                'total_groups': 0,
                'all_wrong_groups': 0,
                'successfully_augmented_groups': 0,
                'total_responses_generated': 0,
                'total_reward_boost': 0.0,
                'examples': []
            }
        }

        self.step_counter += 1

        # Step 1: Calculate base scores and group by question
        groups = defaultdict(list)
        base_items = []

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(data_item.batch['attention_mask'][:prompt_length].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = int(data_item.batch['attention_mask'][prompt_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            prompt_str = self.tokenizer.decode(valid_prompt_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            # Get response part only
            response_str = sequences_str[len(prompt_str):].strip()

            # Calculate base score
            base_score, extracted_answer = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                return_extracted_answer=True
            )

            item = {
                'index': i,
                'base_score': base_score,
                'valid_response_length': valid_response_length,
                'sequences_str': sequences_str,
                'prompt_str': prompt_str,
                'response_str': response_str,
                'extracted_answer': extracted_answer,
                'ground_truth': ground_truth,
                'data_source': data_source,
                'extra_info': extra_info
            }
            base_items.append(item)

            # Group by question
            key = self._extract_question_key(prompt_str)
            groups[key].append(item)

        # Step 2: Initialize all rewards with base scores
        for item in base_items:
            reward_tensor[item['index'], item['valid_response_length'] - 1] = float(item['base_score'])

        # Step 3: GPT-4o baseline processing (only for all-wrong groups)
        wandb_stats['gpt4o_baseline']['total_groups'] = len(groups)

        augmented_groups = 0
        for key, group_items in groups.items():
            # Check if all answers are wrong and at least 2 answers
            group_scores = [item['base_score'] for item in group_items]
            all_wrong = all(s <= 0.0 for s in group_scores) and len(group_items) >= 2

            if all_wrong:
                wandb_stats['gpt4o_baseline']['all_wrong_groups'] += 1

            if not all_wrong:
                continue

            # Found all-wrong group, attempting GPT-4o generation
            example_data = {
                'group_size': len(group_items),
                'original_scores': group_scores,
                'original_responses': [item['sequences_str'] for item in group_items],
                'augmented': False
            }

            # Extract question and options from prompt
            prompt_str = group_items[0]['prompt_str']
            question_text, options_text = self._extract_question_and_options(prompt_str)

            if not question_text:
                print(f"⚠️  GPT4O_BASELINE: Could not extract question from prompt")
                wandb_stats['gpt4o_baseline']['examples'].append(example_data)
                continue

            example_data['question'] = question_text
            example_data['options'] = options_text

            try:
                print(f"🔄 GPT4O_BASELINE: Generating 2 responses using GPT-4o API")

                # Generate 2 new responses using GPT-4o
                gpt4o_responses = self._generate_gpt4o_responses(question_text, options_text, n=2)

                if not gpt4o_responses or len(gpt4o_responses) < 2:
                    print(f"❌ GPT4O_BASELINE: Failed to generate enough responses (got {len(gpt4o_responses)}/2)")
                    wandb_stats['gpt4o_baseline']['examples'].append(example_data)
                    continue

                print(f"✅ GPT4O_BASELINE: Successfully generated {len(gpt4o_responses)} responses")
                example_data['generated_responses'] = gpt4o_responses

            except Exception as e:
                print(f"❌ GPT4O_BASELINE: Generation failed: {e}")
                import traceback
                traceback.print_exc()
                wandb_stats['gpt4o_baseline']['examples'].append(example_data)
                continue

            # Calculate scores for GPT-4o responses
            gpt4o_scores = []
            for r in gpt4o_responses:
                gt = group_items[0]['ground_truth']
                base_gpt4o_score = self.compute_score(
                    data_source='math',
                    solution_str=r,
                    ground_truth=gt,
                    extra_info=None
                )
                # Add bonus reward
                final_gpt4o_score = float(base_gpt4o_score) + self.bonus_reward
                gpt4o_scores.append(final_gpt4o_score)
                print(f"💰 GPT4O_BASELINE: Response score: {base_gpt4o_score:.3f} + {self.bonus_reward} = {final_gpt4o_score:.3f}")

            # Randomly select 2 positions to replace
            replace_indices = random.sample(range(len(group_items)), k=min(2, len(group_items)))

            # Replace the rewards and responses
            for j, replace_idx in enumerate(replace_indices):
                if j < len(gpt4o_scores):
                    item = group_items[replace_idx]
                    # Store original data
                    item['original_prompt_str'] = item['prompt_str']
                    item['original_response_str'] = item['response_str']
                    item['original_extracted_answer'] = item['extracted_answer']
                    item['original_base_score'] = item['base_score']

                    old_reward = float(item['base_score'])
                    new_reward = float(gpt4o_scores[j])

                    # Replace response tokens
                    new_response_text = gpt4o_responses[j].strip()
                    new_response_tokens = self.tokenizer.encode(new_response_text, add_special_tokens=False)
                    original_idx = item['index']

                    # Update tokens in data.batch
                    max_response_length = data.batch['responses'].shape[-1]
                    new_response_length = min(len(new_response_tokens), max_response_length)

                    with torch.no_grad():
                        # Clear and write new tokens
                        data.batch['responses'][original_idx].fill_(self.tokenizer.pad_token_id or 0)
                        if new_response_length > 0:
                            new_tensor = torch.tensor(new_response_tokens[:new_response_length],
                                                     dtype=data.batch['responses'].dtype,
                                                     device=data.batch['responses'].device)
                            data.batch['responses'][original_idx][:new_response_length] = new_tensor

                        # Update attention mask
                        prompt_length = data.batch['prompts'].shape[-1]
                        data.batch['attention_mask'][original_idx, prompt_length:] = 0
                        if new_response_length > 0:
                            response_end = min(prompt_length + new_response_length, data.batch['attention_mask'].shape[-1])
                            data.batch['attention_mask'][original_idx, prompt_length:response_end] = 1

                        # Update reward
                        reward_tensor[original_idx, item['valid_response_length'] - 1] = new_reward

                    # Update item for logging
                    item['response_str'] = new_response_text
                    item['base_score'] = new_reward

                    wandb_stats['gpt4o_baseline']['total_responses_generated'] += 1
                    wandb_stats['gpt4o_baseline']['total_reward_boost'] += (new_reward - old_reward)

                    print(f"🔄 GPT4O_BASELINE: Replaced response at index {original_idx}: {old_reward:.3f} → {new_reward:.3f}")

            example_data['augmented'] = True
            example_data['replaced_indices'] = replace_indices
            example_data['gpt4o_scores'] = gpt4o_scores
            wandb_stats['gpt4o_baseline']['examples'].append(example_data)
            augmented_groups += 1

        wandb_stats['gpt4o_baseline']['successfully_augmented_groups'] = augmented_groups

        # Step 4: Generate score records
        for item in base_items:
            data_source = item['data_source']

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"[GPT4O_BASELINE] {item['sequences_str'][:100]}...")
                final_score = reward_tensor[item['index'], item['valid_response_length'] - 1].item()
                is_gpt4o = 'original_base_score' in item
                print(f"[REWARD] Final: {final_score:.3f} (GPT-4o generated: {is_gpt4o})")

            final_score = reward_tensor[item['index'], item['valid_response_length'] - 1].item()
            record = {
                "sequences_str": item['sequences_str'],
                "ground_truth": item['ground_truth'],
                "index": item['extra_info']["index"] if item['extra_info'] else None,
                "score": final_score,
                "original_score": item.get('original_base_score', item['base_score']),
                "is_gpt4o_generated": 'original_base_score' in item
            }
            score_record.append(record)

        print(f"📊 GPT4O_BASELINE Step {self.step_counter}: Total groups={len(groups)}, All-wrong={wandb_stats['gpt4o_baseline']['all_wrong_groups']}, Augmented={augmented_groups}")

        return reward_tensor, score_record
