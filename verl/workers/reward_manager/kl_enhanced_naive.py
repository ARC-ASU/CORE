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

class KLEnhancedNaiveRewardManager:
    """
    KL divergence enhanced reward manager
    
    key features:
    1. recognize difficult questions（all responses are wrong）
    2. mark the difficult questions that need KL divergence constraint
    3. keep the original reward calculation logic
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        
    def __call__(self, data: DataProto):
        """KL divergence enhanced reward calculation"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}
        score_record = []
        
        # Step 1: calculate all the real rewards and group by question
        question_groups = defaultdict(list)
        
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
            
            # calculate the real reward
            score = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            # extract the question content as the group key
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            question_key = self._extract_question_key(prompt_str)
            
            # store the concept related information (if exists)
            concept_info = data_item.non_tensor_batch.get('concept_info', {})
            
            question_groups[question_key].append({
                'index': i,
                'score': score,
                'valid_response_length': valid_response_length,
                'sequences_str': sequences_str,
                'ground_truth': ground_truth,
                'extra_info': extra_info,
                'data_source': data_source,
                'concept_info': concept_info,
                'prompt_str': prompt_str,
                'response_str': self.tokenizer.decode(valid_response_ids)
            })
            
        # Step 2: recognize difficult groups and assign rewards
        for question_key, group_items in question_groups.items():
            if len(group_items) <= 1:
                # single response group, keep the original
                for item in group_items:
                    reward_tensor[item['index'], item['valid_response_length'] - 1] = item['score']
                    item['is_difficult'] = False
            else:
                # check if it is a difficult group (all responses are wrong)
                is_difficult_group = all(item['score'] == 0 for item in group_items)
                
                # assign reward and mark the difficult state
                for item in group_items:
                    reward_tensor[item['index'], item['valid_response_length'] - 1] = item['score']
                    item['is_difficult'] = is_difficult_group
                
                if is_difficult_group:
                    print(f"[DIFFICULT GROUP DETECTED] Question: {question_key[:100]}...")
                    print(f"[DIFFICULT GROUP] {len(group_items)} responses all scored 0")
        
        # Step 3: generate score_record and print information
        for question_key, group_items in question_groups.items():
            for item in group_items:
                data_source = item['data_source']
                
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0
                
                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print(f"[KL_ENHANCED] {item['sequences_str']}")
                    if item['is_difficult']:
                        print(f"[REWARD] Score: {item['score']:.3f} (DIFFICULT - will add KL loss)")
                    else:
                        print(f"[REWARD] Score: {item['score']:.3f}")
                
                record = {
                    "sequences_str": item['sequences_str'],
                    "ground_truth": item['ground_truth'],
                    "index": item['extra_info']["index"] if item['extra_info'] else None,
                    "score": item['score'],
                    "is_difficult": item['is_difficult'],
                    "concept_info": item['concept_info'],
                    "prompt_str": item['prompt_str'],
                    "response_str": item['response_str']
                }
                score_record.append(record)

        return reward_tensor, score_record
    
    def _extract_question_key(self, prompt_str):
        """
        extract the key part of the question as the group key
        use the same logic as the original version
        """
        # simply use the part after Question: and before A.
        try:
            if "Question:" in prompt_str and "A." in prompt_str:
                start = prompt_str.find("Question:") + len("Question:")
                end = prompt_str.find("A.", start)
                if end > start:
                    return prompt_str[start:end].strip()
        except:
            pass
        
        # if extraction fails, use the hash value of the prompt
        return str(hash(prompt_str))