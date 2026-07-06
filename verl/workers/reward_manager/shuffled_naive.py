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
import random
import numpy as np
from collections import defaultdict

class ShuffledNaiveRewardManager:
    """
    RewardManager that shuffles rewards within each question group.

    Strategy:
    1. Compute true rewards for all responses in the batch
    2. Group responses by question (based on prompt content)
    3. Shuffle rewards within each group while preserving statistical properties (mean, std)
    4. Return shuffled rewards
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, shuffle_seed=42) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.shuffle_seed = shuffle_seed
        random.seed(shuffle_seed)
        
    def __call__(self, data: DataProto):
        """Compute rewards with intra-group shuffling."""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}
        score_record = []
        
        # Step 1: Compute true rewards for all responses
        original_rewards = []
        question_groups = defaultdict(list)  # group by question
        
        for i in range(len(data)):
            data_item = data[i]
            
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            
            # Compute true reward
            score = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            # Extract question content as grouping key
            # Assumes question is at a fixed position in prompt format; uses a simple extraction method
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            question_key = self._extract_question_key(prompt_str)
            
            question_groups[question_key].append({
                'index': i,
                'score': score,
                'valid_response_length': valid_response_length,
                'sequences_str': sequences_str,
                'ground_truth': ground_truth,
                'extra_info': extra_info,
                'data_source': data_source
            })
            
        # Step 2: Shuffle rewards within each question group
        for question_key, group_items in question_groups.items():
            if len(group_items) <= 1:
                # Single-response group, keep as is
                for item in group_items:
                    reward_tensor[item['index'], item['valid_response_length'] - 1] = item['score']
            else:
                # Multi-response group, shuffle rewards
                original_scores = [item['score'] for item in group_items]
                shuffled_scores = original_scores.copy()
                random.shuffle(shuffled_scores)
                
                # Assign shuffled rewards
                for item, shuffled_score in zip(group_items, shuffled_scores):
                    reward_tensor[item['index'], item['valid_response_length'] - 1] = shuffled_score
                    item['shuffled_score'] = shuffled_score
        
        # Step 3: Generate score_record and print info
        for question_key, group_items in question_groups.items():
            for item in group_items:
                data_source = item['data_source']
                
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0
                
                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print(f"[SHUFFLED] {item['sequences_str']}")
                    if 'shuffled_score' in item:
                        print(f"[REWARD] Original: {item['score']:.3f} → Shuffled: {item['shuffled_score']:.3f}")
                
                final_score = item.get('shuffled_score', item['score'])
                record = {
                    "sequences_str": item['sequences_str'],
                    "ground_truth": item['ground_truth'],
                    "index": item['extra_info']["index"] if item['extra_info'] else None,
                    "score": final_score,
                    "original_score": item['score'],
                    "is_shuffled": 'shuffled_score' in item
                }
                score_record.append(record)

        return reward_tensor, score_record
    
    def _extract_question_key(self, prompt_str):
        """
        Extract the key part of the question from the prompt for grouping.
        Uses a simple method; adjust based on actual prompt format.
        """
        # Use the text between "Question:" and "A." as the key
        try:
            if "Question:" in prompt_str and "A." in prompt_str:
                start = prompt_str.find("Question:") + len("Question:")
                end = prompt_str.find("A.", start)
                if end > start:
                    return prompt_str[start:end].strip()
        except:
            pass
        
        # If extraction fails, use hash of the prompt
        return str(hash(prompt_str)) 