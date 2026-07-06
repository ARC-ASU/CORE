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

print("DEBUG: Loading RLVR main_ppo_kl.py with KL divergence enhancement!")

"""
KL divergence enhanced version of GRPO training script
"""
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
import ray
import hydra
from verl.utils.reward_score import deepscaler

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from verl.utils.kl_regularizer import ConceptKLRegularizer
import torch
import random
from collections import defaultdict
import wandb
from tensordict import TensorDict
from typing import List, Dict, Any

# Import ConceptAugmentedRewardManager for inheritance
from verl.trainer.main_ppo import ConceptAugmentedRewardManager

class ConceptKLRewardManager(ConceptAugmentedRewardManager):
    """KL divergence enhanced RewardManager - inherit all logic from concept enhance"""
    
    def __init__(self, tokenizer, num_examine, compute_score=None, 
                 concept_file: str = "data/quizzes/concept_quizzes.jsonl",
                 model_path: str = None, trainer=None) -> None:
        # call the parent class constructor, get all concept enhancement features (using the same parameters as concept enhance)
        super().__init__(tokenizer, num_examine, compute_score, concept_file, model_path, trainer)
        print(f"🔧 KL: ConceptKLRewardManager initialized with model_path={model_path}")
        
        # initialize step_counter (inherited from parent class concept)
        self.step_counter = 0
        self._kl_concept_cache = {}
        
        # KL specific attributes
        self.kl_statistics = {
            'total_groups': 0,
            'all_wrong_groups': 0, 
            'kl_computed_groups': 0,
            'concept_effective_responses': 0,
            'total_kl_loss': 0.0
        }
        
    def __call__(self, data: DataProto):
        """KL divergence enhanced version of reward calculation - identify all wrong groups but do not replace responses"""
        print("🔧 KL: ConceptKLRewardManager.__call__ started")
        
        # increase step counter
        self.step_counter += 1
        
        # 🔥 Step 1: reuse all analysis logic from parent class, but skip actual response replacement
        # first call parent class basic logic (excluding concept enhancement)
        import torch
        from collections import defaultdict
        
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']
        
        print(f"Step {self.step_counter}: starting concept-enhanced reward computation")
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        base_items = []
        groups = defaultdict(list)
        already_print = 0
        
        # 🔥 Step 1: calculate base reward and group (fully reuse parent class logic)
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
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            
            # get pure response part
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = sequences_str[len(prompt_str):].strip()
            
            # use parent class compute_score function, get base_score and extracted_answer
            base_score, extracted_answer = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                return_extracted_answer=True
            )
            
            # 🐞 BUGFIX: Correctly get the question key and enhanced prompt.
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            question_key_from_extraction = self._extract_question_key(prompt_str)

            # find the enhanced prompt and check correctness for later use
            enhanced_prompt, is_correct, matched_question_key = self._find_enhanced_prompt(prompt_str)
            
            original_prompt_id_list = valid_prompt_ids.detach().cpu().tolist()
            original_response_id_list = valid_response_ids.detach().cpu().tolist()

            item = {
                'index': i,
                'base_score': base_score,
                'extracted_answer': extracted_answer,
                'valid_response_length': valid_response_length,
                'sequences_str': sequences_str,
                'prompt_str': prompt_str,
                'response_str': response_str,
                'ground_truth': ground_truth,
                'extra_info': extra_info,
                'data_source': data_source,
                'enhanced_prompt': enhanced_prompt,
                'is_concept_correct': is_correct,
                'question_key': question_key_from_extraction,
                'kl_info': {
                    'has_concept': False,
                    'question_key': question_key_from_extraction,
                    'original_prompt_ids': original_prompt_id_list,
                    'original_response_ids': original_response_id_list
                }
            }

            # Use the matched key from concept data if available, otherwise use extracted key.
            final_question_key = matched_question_key if matched_question_key else question_key_from_extraction
            groups[final_question_key].append(item)
            base_items.append(item)

            # set base reward
            reward_tensor[i, valid_response_length - 1] = base_score
            
            # debug output
            if already_print < self.num_examine:
                print(f"🔍 KL[{i}]: {sequences_str[:100]}...")
                print(f"[REWARD] Score: {base_score:.3f}")
                already_print += 1
        
        # 🔥 Step 2: identify all wrong groups (reuse parent class logic)
        self.kl_statistics['total_groups'] = len(groups)
        all_wrong_groups = []
        
        for key, group_items in groups.items():
            group_scores = [item['base_score'] for item in group_items]
            all_wrong = all(s <= 0.0 for s in group_scores) and len(group_items) >= 2
            
            if all_wrong:
                self.kl_statistics['all_wrong_groups'] += 1
                all_wrong_groups.append((key, group_items))
                print(f"🎯 KL: Found all-wrong group with {len(group_items)} items: {key[:50]}...")
        
        # 🔥 Step 3: prepare concept-enhanced information for all wrong groups, but do not replace
        self._prepare_concept_enhanced_data_for_kl(all_wrong_groups, data)
        
        # build score_record
        score_record = []
        all_wrong_item_ids = {id(it) for _, grp in all_wrong_groups for it in grp}
        for item in base_items:
            record = {
                "sequences_str": item['sequences_str'],
                "ground_truth": item['ground_truth'],
                "index": item['extra_info']['index'] if item['extra_info'] else None,
                "score": item['base_score'],
                "extracted_answer": item['extracted_answer'],
                "prompt_str": item['prompt_str'],
                "response_str": item['response_str'],
                "is_all_wrong_group": id(item) in all_wrong_item_ids,
                "kl_info": getattr(item, 'kl_info', {})  # KL specific information
            }
            score_record.append(record)

        print(f"🔧 KL: Processed {len(score_record)} items, found {self.kl_statistics['all_wrong_groups']} all-wrong groups")
        print(f"📊 Step {self.step_counter}: KL data preparation completed")
        self._attach_kl_payload(data, base_items)
        self._apply_core_kl_penalty(data, base_items, reward_tensor)
        self.step_counter += 1

        return reward_tensor, score_record
    
    def _prepare_concept_enhanced_data_for_kl(self, all_wrong_groups, data):
        """prepare concept-enhanced data for KL calculation, but do not replace original responses"""
        print(f"🔧 KL: Preparing concept-enhanced data for {len(all_wrong_groups)} all-wrong groups")
        
        if not self._can_use_actor_rollout():
            print("❌ KL: Actor rollout not available for concept generation")
            return
        
        for key, group_items in all_wrong_groups:
            try:
                # 🔥 reuse concept enhance logic to generate concept-enhanced prompts
                aug_prompt, is_correct, question_key = self._find_enhanced_prompt(group_items[0]['prompt_str'])
                
                if not aug_prompt:
                    print(f"⚠️ KL: No concept found for group {key[:30]}...")
                    continue
                
                print(f"✅ KL: Found concept for group {key[:30]}... -> {aug_prompt[:50]}...")
                
                # 🔥 use training rollout to generate concept-enhanced responses (4)
                cached_responses = self._kl_concept_cache.get(question_key)
                if cached_responses and len(cached_responses) >= len(group_items):
                    concept_responses = cached_responses
                    print(f"🔁 KL: Reusing cached concept responses for group {key[:30]}...")
                else:
                    concept_responses = self._generate_enhanced_responses_with_rollout(aug_prompt, n=4)
                    if concept_responses:
                        self._kl_concept_cache[question_key] = concept_responses
                
                if concept_responses and len(concept_responses) >= 4:
                    print(f"✅ KL: Generated {len(concept_responses)} concept-enhanced responses")
                    
                    # add concept information to each item for subsequent KL calculation
                    for i, item in enumerate(group_items):
                        if i < len(concept_responses):
                            concept_response = concept_responses[i]
                            concept_score, _ = self.compute_score(
                                data_source=item['data_source'],
                                solution_str=concept_response,
                                ground_truth=item['ground_truth'],
                                extra_info=item['extra_info'],
                                return_extracted_answer=True
                            )
                            concept_prompt_ids = self.tokenizer.encode(aug_prompt, add_special_tokens=False)
                            concept_response_ids = self.tokenizer.encode(concept_response, add_special_tokens=False)
                            item['kl_info'] = {
                                'has_concept': True,
                                'concept_prompt': aug_prompt,
                                'concept_response': concept_response,
                                'concept_prompt_ids': concept_prompt_ids,
                                'concept_response_ids': concept_response_ids,
                                'original_prompt_ids': item['kl_info'].get('original_prompt_ids', []),
                                'original_response_ids': item['kl_info'].get('original_response_ids', []),
                                'original_response': item['response_str'],
                                'question_key': question_key,
                                'concept_score': concept_score
                            }
                            self.kl_statistics['concept_effective_responses'] += int(concept_score > item['base_score'])
                        else:
                            item['kl_info'] = {'has_concept': False}

                    self.kl_statistics['kl_computed_groups'] += 1
                else:
                    print(f"❌ KL: Failed to generate concept responses for group {key[:30]}...")
                    # mark as no concept information
                    for item in group_items:
                        item['kl_info'] = {'has_concept': False}
                        
            except Exception as e:
                print(f"❌ KL: Error preparing concept data for group {key[:30]}...: {e}")
                for item in group_items:
                    item['kl_info'] = {'has_concept': False}

    def _attach_kl_payload(self, data: DataProto, base_items):
        """Attach KL candidates to DataProto meta_info for actor consumption."""
        if data.meta_info is None:
            data.meta_info = {}

        kl_ready = []
        for item in base_items:
            kl_info = item.get('kl_info', {})
            if not kl_info.get('has_concept'):
                continue
            sanitized = {
                'batch_index': item['index'],
                'response_length': item['valid_response_length'],
                'prompt_str': item['prompt_str'],
                'response_str': item['response_str'],
                'ground_truth': item['ground_truth'],
                'base_score': item['base_score'],
                'kl_info': {
                    'has_concept': True,
                    'concept_prompt': kl_info.get('concept_prompt'),
                    'concept_response': kl_info.get('concept_response'),
                    'concept_prompt_ids': list(kl_info.get('concept_prompt_ids', [])),
                    'concept_response_ids': list(kl_info.get('concept_response_ids', [])),
                    'original_prompt_ids': list(kl_info.get('original_prompt_ids', [])),
                    'original_response_ids': list(kl_info.get('original_response_ids', [])),
                    'question_key': kl_info.get('question_key'),
                    'concept_score': kl_info.get('concept_score', 0.0)
                }
            }
            kl_ready.append(sanitized)

        if kl_ready:
            data.meta_info['kl_items'] = kl_ready
            data.meta_info['kl_cfg'] = self._resolve_kl_cfg()
            print(f"🔍 KL DEBUG: Set current_kl_items to {len(kl_ready)} items")
        else:
            data.meta_info.pop('kl_items', None)
            data.meta_info.pop('kl_cfg', None)

    def _resolve_kl_cfg(self):
        default_cfg = {
            'base_lambda': 0.01,
            'effective_multiplier': 3.0,
            'ineffective_multiplier': 0.5
        }
        trainer_cfg = getattr(self.trainer, 'config', None)
        if trainer_cfg is None:
            return default_cfg
        try:
            kl_cfg = trainer_cfg.get('kl_enhancement', {})
        except AttributeError:
            kl_cfg = trainer_cfg.kl_enhancement if hasattr(trainer_cfg, 'kl_enhancement') else {}
        return {
            'base_lambda': float(getattr(kl_cfg, 'base_lambda', kl_cfg.get('base_lambda', 0.01))),
            'effective_multiplier': float(getattr(kl_cfg, 'effective_multiplier', kl_cfg.get('effective_multiplier', 3.0))),
            'ineffective_multiplier': float(getattr(kl_cfg, 'ineffective_multiplier', kl_cfg.get('ineffective_multiplier', 0.5)))
        }

    def _ensure_token_ids_list(self, cached_ids, fallback_text: str) -> List[int]:
        """Ensure we have token ids available, encoding text when necessary."""
        if cached_ids:
            try:
                return list(cached_ids)
            except TypeError:
                pass
        if not fallback_text:
            return []
        return self.tokenizer.encode(fallback_text, add_special_tokens=False)

    def _apply_core_kl_penalty(self, data: DataProto, base_items: List[Dict[str, Any]], reward_tensor: torch.Tensor) -> None:
        """Approximate forward KL penalty using guided vs unguided log-probs."""
        if not isinstance(data.meta_info, dict):
            return
        if 'kl_items' not in data.meta_info:
            return
        if 'old_log_probs' not in data.batch.keys():
            return
        if not self._can_use_actor_rollout():
            print("❌ KL: Unable to compute CORE-KL penalty (actor rollout unavailable)")
            return

        kl_cfg = data.meta_info.get('kl_cfg', self._resolve_kl_cfg())
        base_lambda = float(kl_cfg.get('base_lambda', 0.01))
        effective_mult = float(kl_cfg.get('effective_multiplier', 3.0))
        ineffective_mult = float(kl_cfg.get('ineffective_multiplier', 0.5))

        kl_candidates = [item for item in base_items if item.get('kl_info', {}).get('has_concept')]
        if not kl_candidates:
            return

        try:
            indices = [item['index'] for item in kl_candidates]
            prompts = data.batch['prompts'][indices].clone()
            responses = data.batch['responses'][indices].clone()
            attention_mask = data.batch['attention_mask'][indices].clone()
            position_ids = data.batch.get('position_ids', None)
            if position_ids is not None:
                position_ids = position_ids[indices].clone()
            prompt_seq_len = prompts.shape[1]
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

            for row_idx, item in enumerate(kl_candidates):
                kl_info = item.get('kl_info', {})
                concept_prompt_tokens = self._ensure_token_ids_list(
                    kl_info.get('concept_prompt_ids'),
                    kl_info.get('concept_prompt', '')
                )
                if not concept_prompt_tokens:
                    continue

                prompt_tensor = prompts[row_idx]
                prompt_tensor.fill_(pad_id)

                token_tensor = torch.tensor(
                    concept_prompt_tokens[-prompt_seq_len:],
                    dtype=prompt_tensor.dtype
                )
                token_count = token_tensor.shape[0]
                prompt_tensor[-token_count:] = token_tensor

                attention_mask[row_idx, :prompt_seq_len] = 0
                attention_mask[row_idx, prompt_seq_len - token_count:prompt_seq_len] = 1

            seq_len = attention_mask.shape[1]
            if position_ids is None:
                position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).repeat(len(indices), 1)

            input_ids = torch.cat([prompts, responses], dim=1)

            concept_batch = DataProto(
                batch=TensorDict({
                    'input_ids': input_ids,
                    'prompts': prompts,
                    'responses': responses,
                    'attention_mask': attention_mask,
                    'position_ids': position_ids
                }, batch_size=len(indices)),
                meta_info={
                    'micro_batch_size': getattr(self.trainer.config.actor_rollout_ref.rollout,
                                                'log_prob_micro_batch_size_per_gpu', None),
                    'max_token_len': getattr(self.trainer.config.actor_rollout_ref.rollout,
                                             'log_prob_max_token_len_per_gpu', None),
                    'use_dynamic_bsz': getattr(self.trainer.config.actor_rollout_ref.rollout,
                                               'log_prob_use_dynamic_bsz', False)
                }
            )

            concept_output = self.trainer.actor_rollout_wg.compute_log_prob(concept_batch)
            guided_log_probs = concept_output.batch['old_log_probs']
            unguided_log_probs = data.batch['old_log_probs'][indices]

            total_penalty = 0.0
            applied = 0

            for row_idx, item in enumerate(kl_candidates):
                valid_resp_len = max(0, int(item.get('valid_response_length', responses.shape[1])))
                if valid_resp_len == 0:
                    continue

                guided_slice = guided_log_probs[row_idx, :valid_resp_len]
                unguided_slice = unguided_log_probs[row_idx, :valid_resp_len]

                kl_value = (guided_slice - unguided_slice).sum().item()
                is_effective = item['kl_info'].get('concept_score', 0.0) > item.get('base_score', 0.0)
                weight = base_lambda * (effective_mult if is_effective else ineffective_mult)
                penalty = weight * kl_value

                reward_pos = max(0, valid_resp_len - 1)
                reward_tensor[item['index'], reward_pos] = reward_tensor[item['index'], reward_pos] - penalty

                item['kl_info']['core_kl'] = float(kl_value)
                item['kl_info']['core_kl_penalty'] = float(penalty)

                total_penalty += penalty
                applied += 1

            if applied > 0:
                print(f"✅ CORE-KL: Applied penalties to {applied} samples (total penalty {total_penalty:.4f})")
                if not hasattr(self, '_pending_kl_metrics'):
                    self._pending_kl_metrics = {}
                self._pending_kl_metrics.update({
                    "kl_enhancement/core_kl_penalty": float(total_penalty),
                    "kl_enhancement/core_kl_samples": applied
                })

        except Exception as e:
            print(f"❌ CORE-KL: Failed to apply forward KL penalty: {e}")
            import traceback
            traceback.print_exc()
    
    def _find_enhanced_prompt(self, prompt_str: str):
        """
        Finds the enhanced prompt from concept data.
        
        Returns:
            A tuple (enhanced_prompt, is_correct, question_key) or (None, False, None) if not found.
        """
        q_text = self._extract_question_key(prompt_str)
        
        if q_text:
            # First, try exact match
            if q_text in self.concept_data:
                entry = self.concept_data[q_text]
                return entry.get('enhanced_prompt'), entry.get('is_correct', False), q_text
            
            # Then, try fuzzy match
            for key, data in self.concept_data.items():
                q_simple = q_text.replace('\\\\', '\\').replace('\\', '').replace('$', '').replace('{', '').replace('}', '')
                key_simple = key.replace('\\\\', '\\').replace('\\', '').replace('$', '').replace('{', '').replace('}', '')
                
                if q_simple == key_simple or q_simple in key_simple or key_simple in q_simple:
                    return data.get('enhanced_prompt'), data.get('is_correct', False), key
        
        # 🐞 BUGFIX: Always return a tuple to avoid TypeError
        return None, False, None

    def _extract_question_key(self, prompt_str: str) -> str:
        """extract the key part of the question as the grouping basis from the prompt"""
        # simply use the part after Question: to the part before A.
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


class KLEnhancedRayPPOTrainer(RayPPOTrainer):
    """KL divergence enhanced PPO trainer"""
    
    def __init__(self, config, tokenizer, role_worker_mapping, resource_pool_manager, 
                 ray_worker_group_cls, reward_fn, val_reward_fn):
        super().__init__(config, tokenizer, role_worker_mapping, resource_pool_manager, 
                        ray_worker_group_cls, reward_fn, val_reward_fn)
        
        # initialize KL regularizer
        print("DEBUG: Initializing ConceptKLRegularizer...")
        
        self.kl_regularizer = ConceptKLRegularizer(
            tokenizer=tokenizer,
            base_lambda=config.get('kl_enhancement', {}).get('base_lambda', 0.01),
            effective_multiplier=config.get('kl_enhancement', {}).get('effective_multiplier', 3.0),
            ineffective_multiplier=config.get('kl_enhancement', {}).get('ineffective_multiplier', 0.5)
        )
        
        print(f"DEBUG: KL regularizer initialized with base_lambda={self.kl_regularizer.base_lambda}")
        
        # store the current rollout score_record, for KL loss calculation
        self.current_score_record = []
    
    def _extract_question_key(self, prompt_str: str) -> str:
        """extract the key part of the question as the grouping basis from the prompt"""
        try:
            if "Question:" in prompt_str and "A." in prompt_str:
                start = prompt_str.find("Question:") + len("Question:")
                end = prompt_str.find("A.", start)
                if end > start:
                    return prompt_str[start:end].strip()
        except:
            pass
        return str(hash(prompt_str))

    def _post_reward_kl_enhancement(self, score_record):
        """
        Post-reward processing for KL enhancement.
        This analyzes the score record to identify difficult groups and prepare KL data.
        """
        try:
            print(f"🔧 KL: Analyzing {len(score_record)} items for KL enhancement")

            # Group items by question for all-wrong detection
            question_groups = defaultdict(list)
            for item in score_record:
                question_key = self._extract_question_key(item.get('prompt_str', ''))
                question_groups[question_key].append(item)
            
            # Analyze each group and add KL enhancement data
            kl_enhanced_count = 0
            for question_key, group_items in question_groups.items():
                if len(group_items) >= 4:  # Ensure we have enough items for group analysis
                    # Check if this is an all-wrong group
                    all_scores = [item.get('score', 0.0) for item in group_items]
                    is_all_wrong = all(score <= 0.0 for score in all_scores)
                    
                    if is_all_wrong:
                        print(f"🎯 KL: Found all-wrong group with {len(group_items)} items")
                        
                        # Mark items for KL enhancement
                        for item in group_items:
                            item['kl_enhancement_candidate'] = True
                            kl_enhanced_count += 1
            
            if kl_enhanced_count > 0:
                print(f"🔧 KL: Marked {kl_enhanced_count} items for KL enhancement")
                
                # Store KL metrics
                self._pending_kl_metrics = {
                    "kl_enhancement/candidate_items": kl_enhanced_count,
                    "kl_enhancement/total_items": len(score_record),
                    "kl_enhancement/enhancement_ratio": kl_enhanced_count / len(score_record)
                }
            
            return score_record
            
        except Exception as e:
            print(f"❌ KL: Error in post-reward KL enhancement: {e}")
            return score_record

    def fit(self):
        """
        Override fit to add KL enhancement analysis.
        This preserves all the standard training metrics while adding KL analysis.
        """
        # Initialize workers before anything else - this fixes the actor_rollout_wg error
        print("🔧 Initializing worker groups...")
        self.init_workers()
        print("✅ Worker groups initialized successfully")
        
        # Store original reward function
        original_reward_fn = self.reward_fn
        
        def enhanced_reward_fn(data_batch):
            """Wrapper that adds KL enhancement analysis"""
            reward_tensor, score_record = original_reward_fn(data_batch)
            
            # Add KL enhancement analysis
            enhanced_score_record = self._post_reward_kl_enhancement(score_record)
            
            return reward_tensor, enhanced_score_record
        
        # Replace reward function temporarily
        self.reward_fn = enhanced_reward_fn
        
        try:
            # Call parent fit which preserves all standard metrics
            super().fit()
        finally:
            # Restore original functions
            self.reward_fn = original_reward_fn
    
    def _get_kl_metrics_for_logging(self):
        """
        Get pending KL metrics for logging
        """
        if hasattr(self, '_pending_kl_metrics'):
            metrics = self._pending_kl_metrics.copy()
            # Clear the metrics after retrieving them
            self._pending_kl_metrics = {}
            return metrics
        return {}


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config, compute_score=None):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config, compute_score))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config, compute_score=None):
    from verl.utils.fs import copy_local_path_from_hdfs
    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    print("🚀 Starting KL-enhanced GRPO training...")
    print(f"🔧 KL enhancement config: {config.get('kl_enhancement', {})}")

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }
    
    # debug Role object
    print(f"DEBUG: Role.ActorRollout = {Role.ActorRollout}")
    print(f"DEBUG: Role.ActorRollout.__class__ = {Role.ActorRollout.__class__}")
    print(f"DEBUG: Role.ActorRollout.__class__.__module__ = {Role.ActorRollout.__class__.__module__}")
    print(f"DEBUG: mapping keys = {list(mapping.keys())}")
    print(f"DEBUG: role_worker_mapping keys = {list(role_worker_mapping.keys())}")

    # reward model setup
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    print(f"DEBUG: reward_manager_name = {reward_manager_name}")
    
    # use our KL enhanced RewardManager
    if reward_manager_name == 'kl_enhanced_naive':
        reward_manager_cls = ConceptKLRewardManager
        print("DEBUG: Using ConceptKLRewardManager (KL enhanced with concept integration)")
    elif reward_manager_name == 'naive':
        # use old KL version as fallback
        reward_manager_cls = KLEnhancedNaiveRewardManager  
        print("DEBUG: Using KLEnhancedNaiveRewardManager (basic KL version)")
    elif reward_manager_name == 'shuffled_naive':
        from verl.workers.reward_manager import ShuffledNaiveRewardManager
        reward_manager_cls = ShuffledNaiveRewardManager
        print("DEBUG: Using original ShuffledNaiveRewardManager")
    elif reward_manager_name == 'prime':
        from verl.workers.reward_manager import PrimeRewardManager
        reward_manager_cls = PrimeRewardManager
        print("DEBUG: Using PrimeRewardManager")
    else:
        from verl.workers.reward_manager import NaiveRewardManager
        reward_manager_cls = NaiveRewardManager
        print(f"DEBUG: Using original {reward_manager_name}")

    # score function setup 
    if config.actor_rollout_ref.model.path.strip().startswith("Qwen") or 'llama' in config.actor_rollout_ref.model.path.lower() or config.actor_rollout_ref.model.use_think == False:
        print("\nQwen or LLAMA---------------------------------\n")
        compute_score = deepscaler.compute_score
    
    # create reward functions
    if reward_manager_name == 'kl_enhanced_naive':
        # KL enhanced version uses the same initialization as concept enhance, and passes concept_file
        concept_file_path = config.reward_model.get('concept_file',
            "data/quizzes/concept_quizzes.jsonl")
        print(f"DEBUG: Using concept_file: {concept_file_path}")
        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score,
                                      concept_file=concept_file_path, model_path=local_path, trainer=None)
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score,
                                          concept_file=concept_file_path, model_path=local_path, trainer=None)
        print(f"DEBUG: Initialized ConceptKLRewardManager with concept_file={concept_file_path}")
    else:
        # other versions use standard initialization
        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)
        print("DEBUG: Initialized standard reward manager")

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    # create KL enhanced trainer
    trainer = KLEnhancedRayPPOTrainer(config=config,
                                     tokenizer=tokenizer,
                                     role_worker_mapping=role_worker_mapping,
                                     resource_pool_manager=resource_pool_manager,
                                     ray_worker_group_cls=ray_worker_group_cls,
                                     reward_fn=reward_fn,
                                     val_reward_fn=val_reward_fn)
    
    # set the trainer reference to the KL enhanced reward managers (learn from concept enhance)
    if reward_manager_name == 'kl_enhanced_naive':
        reward_fn.trainer = trainer
        val_reward_fn.trainer = trainer
        print("🔧 KL: Set trainer reference for dynamic checkpoint loading and rollout generation")
    
    trainer.fit()


if __name__ == '__main__':
    main()
