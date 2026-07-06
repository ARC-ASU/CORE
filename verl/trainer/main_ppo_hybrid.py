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

print("DEBUG: Loading RLVR main_ppo_hybrid.py with Concept Replacement + KL hybrid!")

"""
hybrid version: Concept Replacement + Concept KL
- for all wrong groups, half use concept replacement (directly replace tokens)
- half use concept KL (KL penalty but do not replace)
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
from typing import List, Dict, Any, Tuple

# Import parent classes
import sys
import os
# Use relative imports, no need to modify sys.path
from verl.trainer.main_ppo import ConceptAugmentedRewardManager
from verl.trainer.main_ppo_kl import ConceptKLRewardManager


class ConceptHybridRewardManager(ConceptKLRewardManager):
    """
    hybrid version of RewardManager:
    - for all wrong groups, randomly split into two halves
    - half use Concept Replacement (directly replace response tokens + 0.4 bonus)
    - half use Concept KL (KL penalty guide, do not replace)
    """

    def __init__(self, tokenizer, num_examine, compute_score=None,
                 concept_file: str = "data/quizzes/concept_quizzes.jsonl",
                 model_path: str = None, trainer=None,
                 replacement_ratio: float = 0.5) -> None:
        """
        Args:
            replacement_ratio: the ratio of using replacement in all wrong groups, default 0.5 (half)
        """
        super().__init__(tokenizer, num_examine, compute_score, concept_file, model_path, trainer)
        self.replacement_ratio = replacement_ratio
        print(f"🔧 HYBRID: ConceptHybridRewardManager initialized")
        print(f"   - Replacement ratio: {replacement_ratio:.1%}")
        print(f"   - KL ratio: {1 - replacement_ratio:.1%}")

        # statistics information
        self.hybrid_statistics = {
            'total_all_wrong_groups': 0,
            'replacement_groups': 0,
            'kl_groups': 0,
            'replacement_responses': 0,
            'kl_responses': 0,
        }

    def __call__(self, data: DataProto):
        """hybrid version of reward calculation"""
        print("🔧 HYBRID: ConceptHybridRewardManager.__call__ started")

        self.step_counter += 1

        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        print(f"📊 Step {self.step_counter}: start Hybrid (Replacement + KL) reward calculation")

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        base_items = []
        groups = defaultdict(list)
        already_print = 0

        # ========== Step 1: calculate base reward and group ==========
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

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = sequences_str[len(prompt_str):].strip()

            base_score, extracted_answer = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                return_extracted_answer=True
            )

            question_key = self._extract_question_key(prompt_str)
            enhanced_prompt, is_correct, matched_key = self._find_enhanced_prompt(prompt_str)

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
                'question_key': question_key,
                'kl_info': {
                    'has_concept': False,
                    'question_key': question_key,
                    'original_prompt_ids': original_prompt_id_list,
                    'original_response_ids': original_response_id_list
                },
                'hybrid_mode': None  # 'replacement' or 'kl'
            }

            final_key = matched_key if matched_key else question_key
            groups[final_key].append(item)
            base_items.append(item)

            # set base reward
            reward_tensor[i, valid_response_length - 1] = base_score

            if already_print < self.num_examine:
                print(f"🔍 HYBRID[{i}]: {sequences_str[:100]}...")
                print(f"[REWARD] Score: {base_score:.3f}")
                already_print += 1

        # ========== Step 2: identify all wrong groups and assign processing strategy ==========
        all_wrong_groups = []

        for key, group_items in groups.items():
            group_scores = [item['base_score'] for item in group_items]
            all_wrong = all(s <= 0.0 for s in group_scores) and len(group_items) >= 2

            if all_wrong:
                all_wrong_groups.append((key, group_items))
                self.hybrid_statistics['total_all_wrong_groups'] += 1
                print(f"🎯 HYBRID: Found all-wrong group with {len(group_items)} items: {key[:50]}...")

        # shuffle all wrong groups randomly, then allocate according to the ratio
        random.shuffle(all_wrong_groups)
        split_idx = int(len(all_wrong_groups) * self.replacement_ratio)

        replacement_groups = all_wrong_groups[:split_idx]
        kl_groups = all_wrong_groups[split_idx:]

        print(f"📊 HYBRID: Total all-wrong groups: {len(all_wrong_groups)}")
        print(f"   - Replacement groups: {len(replacement_groups)}")
        print(f"   - KL groups: {len(kl_groups)}")

        # ========== Step 3: process Replacement groups ==========
        self._process_replacement_groups(replacement_groups, data, reward_tensor)

        # ========== Step 4: process KL groups ==========
        self._process_kl_groups(kl_groups, data, reward_tensor)

        # ========== Step 5: build score_record ==========
        score_record = []
        all_wrong_item_ids = {id(it) for _, grp in all_wrong_groups for it in grp}

        for item in base_items:
            final_reward = reward_tensor[item['index'], item['valid_response_length'] - 1].item()
            record = {
                "sequences_str": item['sequences_str'],
                "ground_truth": item['ground_truth'],
                "index": item['extra_info']['index'] if item['extra_info'] else None,
                "score": final_reward,
                "original_score": item.get('original_base_score', item['base_score']),
                "extracted_answer": item['extracted_answer'],
                "prompt_str": item['prompt_str'],
                "response_str": item['response_str'],
                "is_all_wrong_group": id(item) in all_wrong_item_ids,
                "hybrid_mode": item.get('hybrid_mode'),
                "concept_enhanced": item.get('hybrid_mode') == 'replacement',
                "kl_info": item.get('kl_info', {})
            }
            score_record.append(record)

        # attach KL payload for actor training
        self._attach_kl_payload(data, base_items)

        # print statistics information
        print(f"📊 HYBRID Step {self.step_counter} Summary:")
        print(f"   - Replacement responses: {self.hybrid_statistics['replacement_responses']}")
        print(f"   - KL responses: {self.hybrid_statistics['kl_responses']}")

        return reward_tensor, score_record

    def _process_replacement_groups(self, replacement_groups: List[Tuple], data: DataProto, reward_tensor: torch.Tensor):
        """process groups that need replacement (reuse concept_aug logic)"""
        if not replacement_groups:
            return

        if not self._can_use_actor_rollout():
            print("❌ HYBRID: Actor rollout not available for replacement")
            return

        print(f"🔄 HYBRID: Processing {len(replacement_groups)} replacement groups")

        for key, group_items in replacement_groups:
            try:
                # mark as replacement mode
                for item in group_items:
                    item['hybrid_mode'] = 'replacement'

                # get enhanced prompt
                aug_prompt, is_correct, matched_key = self._find_enhanced_prompt(group_items[0]['prompt_str'])

                if not aug_prompt:
                    print(f"⚠️ HYBRID: No concept found for replacement group {key[:30]}...")
                    continue

                # generate concept-enhanced responses
                aug_responses = self._generate_enhanced_responses_with_rollout(aug_prompt, n=len(group_items))

                if not aug_responses or len(aug_responses) < 2:
                    print(f"❌ HYBRID: Failed to generate replacement responses")
                    continue

                print(f"✅ HYBRID: Generated {len(aug_responses)} replacement responses")

                # calculate scores of new responses
                aug_scores = []
                for r in aug_responses:
                    gt = group_items[0]['ground_truth']
                    base_aug_score = self.compute_score(
                        data_source='math',
                        solution_str=r,
                        ground_truth=gt,
                        extra_info=None
                    )
                    # +0.4 extra bonus
                    final_aug_score = float(base_aug_score) + 0.4
                    aug_scores.append(final_aug_score)

                # replace responses (reuse concept_aug logic)
                replace_count = min(len(aug_responses), len(group_items))
                replace_indices = random.sample(range(len(group_items)), k=replace_count)

                for j, replace_idx in enumerate(replace_indices):
                    if j >= len(aug_scores):
                        break

                    item = group_items[replace_idx]

                    # save original information
                    item['original_base_score'] = item['base_score']
                    item['original_response_str'] = item['response_str']

                    old_reward = float(item['base_score'])
                    new_reward = float(aug_scores[j])
                    new_response_text = aug_responses[j].strip()
                    new_response_tokens = self.tokenizer.encode(new_response_text, add_special_tokens=False)
                    original_idx = item['index']

                    # replace tokens
                    max_response_length = data.batch['responses'].shape[-1]
                    new_response_length = min(len(new_response_tokens), max_response_length)

                    with torch.no_grad():
                        pad_id = self.tokenizer.pad_token_id or 0
                        data.batch['responses'][original_idx].fill_(pad_id)
                        if new_response_length > 0:
                            new_tensor = torch.tensor(
                                new_response_tokens[:new_response_length],
                                dtype=data.batch['responses'].dtype,
                                device=data.batch['responses'].device
                            )
                            data.batch['responses'][original_idx][:new_response_length] = new_tensor

                        # update attention_mask
                        prompt_length = data.batch['prompts'].shape[-1]
                        data.batch['attention_mask'][original_idx, prompt_length:] = 0
                        if new_response_length > 0:
                            response_end = min(prompt_length + new_response_length, data.batch['attention_mask'].shape[-1])
                            data.batch['attention_mask'][original_idx, prompt_length:response_end] = 1

                        # update reward
                        reward_tensor[original_idx, :] = 0.0
                        if new_response_length > 0:
                            reward_pos = min(new_response_length - 1, reward_tensor.shape[-1] - 1)
                            reward_tensor[original_idx, reward_pos] = new_reward

                    # update item information
                    item['valid_response_length'] = new_response_length
                    item['response_str'] = new_response_text

                    self.hybrid_statistics['replacement_responses'] += 1
                    print(f"✅ HYBRID REPLACEMENT: idx {original_idx}: {old_reward:.3f} → {new_reward:.3f}")

                self.hybrid_statistics['replacement_groups'] += 1

            except Exception as e:
                print(f"❌ HYBRID: Error in replacement for group {key[:30]}...: {e}")
                import traceback
                traceback.print_exc()

    def _process_kl_groups(self, kl_groups: List[Tuple], data: DataProto, reward_tensor: torch.Tensor):
        """process groups that need KL (reuse concept_kl logic)"""
        if not kl_groups:
            return

        if not self._can_use_actor_rollout():
            print("❌ HYBRID: Actor rollout not available for KL")
            return

        print(f"🔄 HYBRID: Processing {len(kl_groups)} KL groups")

        for key, group_items in kl_groups:
            try:
                # mark as KL mode
                for item in group_items:
                    item['hybrid_mode'] = 'kl'

                # get enhanced prompt
                aug_prompt, is_correct, matched_key = self._find_enhanced_prompt(group_items[0]['prompt_str'])

                if not aug_prompt:
                    print(f"⚠️ HYBRID: No concept found for KL group {key[:30]}...")
                    continue

                # generate concept responses (do not replace)
                concept_responses = self._generate_enhanced_responses_with_rollout(aug_prompt, n=len(group_items))

                if not concept_responses:
                    print(f"❌ HYBRID: Failed to generate KL concept responses")
                    continue

                print(f"✅ HYBRID: Generated {len(concept_responses)} KL concept responses")

                # prepare KL information for each item
                for i, item in enumerate(group_items):
                    if i >= len(concept_responses):
                        break

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
                        'question_key': matched_key or item['question_key'],
                        'concept_score': concept_score
                    }

                    self.hybrid_statistics['kl_responses'] += 1

                self.hybrid_statistics['kl_groups'] += 1

            except Exception as e:
                print(f"❌ HYBRID: Error in KL for group {key[:30]}...: {e}")
                import traceback
                traceback.print_exc()

        # apply KL penalty (reuse parent class logic)
        kl_items = [item for _, grp in kl_groups for item in grp if item.get('kl_info', {}).get('has_concept')]
        if kl_items:
            self._apply_core_kl_penalty(data, kl_items, reward_tensor)


class HybridRayPPOTrainer(RayPPOTrainer):
    """hybrid version of PPO trainer"""

    def __init__(self, config, tokenizer, role_worker_mapping, resource_pool_manager,
                 ray_worker_group_cls, reward_fn, val_reward_fn):
        super().__init__(config, tokenizer, role_worker_mapping, resource_pool_manager,
                        ray_worker_group_cls, reward_fn, val_reward_fn)

        # initialize KL regularizer
        print("DEBUG: Initializing ConceptKLRegularizer for Hybrid trainer...")

        self.kl_regularizer = ConceptKLRegularizer(
            tokenizer=tokenizer,
            base_lambda=config.get('kl_enhancement', {}).get('base_lambda', 0.01),
            effective_multiplier=config.get('kl_enhancement', {}).get('effective_multiplier', 3.0),
            ineffective_multiplier=config.get('kl_enhancement', {}).get('ineffective_multiplier', 0.5)
        )

        print(f"DEBUG: Hybrid trainer initialized with KL regularizer")

    def fit(self):
        """
        Override fit to initialize workers first.
        This fixes the actor_rollout_wg AttributeError.
        """
        # Initialize workers before anything else
        print("🔧 HYBRID: Initializing worker groups...")
        self.init_workers()
        print("✅ HYBRID: Worker groups initialized successfully")

        # Call parent's fit method
        super().fit()


# ============ Hydra Entry Point ============

@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_hybrid_ppo(config)


def run_hybrid_ppo(config):
    """run hybrid version of PPO training"""
    print("🚀 Starting Hybrid (Replacement + KL) GRPO training...")

    ray.init(
        runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'VLLM_ATTENTION_BACKEND': os.environ.get('VLLM_ATTENTION_BACKEND', 'XFORMERS')
            }
        }
    )

    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils.tokenizer import hf_tokenizer

    # copy model to local (if remote path)
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # load tokenizer
    tokenizer = hf_tokenizer(local_path)

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, RayWorkerGroup
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    ray_worker_group_cls = RayWorkerGroup

    # resource pool configuration
    resource_pool_spec = {
        'actor_rollout_ref': [config.trainer.n_gpus_per_node] * config.trainer.nnodes
    }

    global_pool_id = 'actor_rollout_ref'
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(ActorRolloutRefWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id
    }

    # Reward model setup
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_manager_name = config.reward_model.get("reward_manager", "hybrid")
    print(f"DEBUG: reward_manager_name = {reward_manager_name}")

    # use hybrid version of RewardManager
    if reward_manager_name == 'hybrid':
        reward_manager_cls = ConceptHybridRewardManager
        print("DEBUG: Using ConceptHybridRewardManager (Replacement + KL hybrid)")
    else:
        reward_manager_cls = ConceptHybridRewardManager
        print("DEBUG: Defaulting to ConceptHybridRewardManager")

    # Score function setup
    if config.actor_rollout_ref.model.path.strip().startswith("Qwen") or 'llama' in config.actor_rollout_ref.model.path.lower():
        print("\nQwen or LLAMA model detected\n")
        compute_score = deepscaler.compute_score

    # get replacement_ratio configuration
    replacement_ratio = config.get('hybrid', {}).get('replacement_ratio', 0.5)

    # create reward functions
    concept_file_path = config.reward_model.get('concept_file',
        "data/quizzes/concept_quizzes.jsonl")
    print(f"DEBUG: Using concept_file: {concept_file_path}")
    print(f"DEBUG: replacement_ratio: {replacement_ratio}")

    reward_fn = reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=0,
        compute_score=compute_score,
        concept_file=concept_file_path,
        model_path=local_path,
        trainer=None,
        replacement_ratio=replacement_ratio
    )
    val_reward_fn = reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=1,
        compute_score=compute_score,
        concept_file=concept_file_path,
        model_path=local_path,
        trainer=None,
        replacement_ratio=replacement_ratio
    )

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    # create hybrid version of trainer
    trainer = HybridRayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn
    )

    # set trainer reference
    reward_fn.trainer = trainer
    val_reward_fn.trainer = trainer
    print("🔧 HYBRID: Set trainer reference for reward managers")

    trainer.fit()


if __name__ == '__main__':
    main()
