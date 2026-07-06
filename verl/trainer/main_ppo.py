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

# RLVR main_ppo.py loaded

"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import ray
import hydra
from verl.utils.reward_score import deepscaler

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import random
from collections import defaultdict

class ShuffledNaiveRewardManager:
    """RewardManager with intra-group reward shuffling."""
    
    def __init__(self, tokenizer, num_examine, compute_score=None, shuffle_seed=42) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.shuffle_seed = shuffle_seed
        random.seed(shuffle_seed)
        
    def __call__(self, data: DataProto):
        """Compute rewards with intra-group shuffling."""
        # ShuffledNaiveRewardManager processing
        
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}
        score_record = []
        
        # Step 1: calculate all the true rewards and group by question
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
            
            # calculate the true reward
            score = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            # extract the question key (simplified version, using the hash of the prompt)
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            question_key = str(hash(prompt_str))
            
            question_groups[question_key].append({
                'index': i,
                'score': score,
                'valid_response_length': valid_response_length,
                'sequences_str': sequences_str,
                'ground_truth': ground_truth,
                'extra_info': extra_info,
                'data_source': data_source
            })
            
        # Step 2: shuffle the reward in each question group
        for question_key, group_items in question_groups.items():
            if len(group_items) <= 1:
                # group with single answer, keep the original
                for item in group_items:
                    reward_tensor[item['index'], item['valid_response_length'] - 1] = item['score']
            else:
                # group with multiple answers, shuffle the reward
                original_scores = [item['score'] for item in group_items]
                shuffled_scores = original_scores.copy()
                random.shuffle(shuffled_scores)
                
                print(f"DEBUG: Shuffled group with {len(group_items)} items. Original: {original_scores}, Shuffled: {shuffled_scores}")
                
                # assign the shuffled reward
                for item, shuffled_score in zip(group_items, shuffled_scores):
                    reward_tensor[item['index'], item['valid_response_length'] - 1] = shuffled_score
                    item['shuffled_score'] = shuffled_score
        
        # Step 3: generate the score_record
        for question_key, group_items in question_groups.items():
            for item in group_items:
                data_source = item['data_source']
                
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0
                
                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print(f"[SHUFFLED] {item['sequences_str'][:100]}...")
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


class RandomBinaryRewardManager:
    """Random binary reward manager: give 1 with probability p, otherwise give 0."""
    def __init__(self, tokenizer, num_examine, compute_score=None, p: float = 0.5, shuffle_seed: int = 42) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.p = p
        random.seed(shuffle_seed)

    def __call__(self, data: DataProto):
        import torch
        print("DEBUG: RandomBinaryRewardManager is being called! (p=0.5)")
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        score_record = []
        already_print = 0

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

            # 50% probability give 1, otherwise give 0
            r = 1.0 if random.random() < self.p else 0.0
            reward_tensor[i, valid_response_length - 1] = r

            if already_print < self.num_examine:
                print(f"[RANDOM] reward={r:.0f} | text preview: {sequences_str[:100]}...")
                already_print += 1

            score_record.append({
                "sequences_str": sequences_str,
                "ground_truth": None,
                "index": data_item.non_tensor_batch.get('extra_info', {}).get('index') if data_item.non_tensor_batch.get('extra_info') else None,
                "score": r,
                "original_score": None,
                "is_shuffled": False,
                "is_random": True
            })

        return reward_tensor, score_record


# ---------------- Concept-augmented RewardManager ----------------
from collections import defaultdict
import os
try:
    from vllm import LLM, SamplingParams
    _VLLM_OK = True
except Exception:
    _VLLM_OK = False


class ConceptAugmentedRewardManager:
    """
    Concept augmented reward manager:
    - when the group of the same question (usually n=4) all answer wrong
    - use concept+question to replace the original question, let the model generate two responses
    - then randomly replace two of the four errors with the two new responses
    - for these two new responses, add 0.4 reward (if they answer correctly, still have the correct reward)
    """

    def __init__(self, tokenizer, num_examine, compute_score=None,
                 concept_file: str = "data/quizzes/concept_quizzes.jsonl",
                 model_path: str = None,
                 trainer=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or deepscaler.compute_score
        self.model_path = model_path  # model path
        self.trainer = trainer  # trainer reference (currently not used)
        self.concept_data = self._load_concept_data(concept_file)
        self._llm = None
        
        print(f"🔄 CONCEPT: ConceptAugmentedRewardManager initialized")
        print(f"📝 CONCEPT: Important - vLLM will load the SAME model weights as the training actor")
        print(f"🎯 CONCEPT: At step N, both training actor and concept vLLM use identical model state")

    def _load_concept_data(self, concept_file):
        """load the concept data, build the mapping from original question to enhanced prompt"""
        mapping = {}
        try:
            import json
            with open(concept_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    
                    # get the original question and enhanced question
                    original_q = item.get('original_question', '')
                    enhanced_q = item.get('enhanced_question') or item.get('question', '')
                    options = item.get('options', [])
                    
                    if original_q and enhanced_q:
                        # build the complete enhanced prompt (include the options)
                        if options:
                            enhanced_prompt = enhanced_q + "\n" + "\n".join(options) + "\n\nAnswer: "
                        else:
                            enhanced_prompt = enhanced_q + "\n\nAnswer: "
                        
                        # use the normalized original question as the key
                        key = self._normalize_text(original_q)
                        mapping[key] = {
                            'enhanced_prompt': enhanced_prompt,
                            'original_question': original_q,
                            'options': options
                        }
        except Exception as e:
            print(f"WARN: failed to load concept file {concept_file}: {e}")
        print(f"DEBUG: ConceptAugmentedRewardManager loaded {len(mapping)} question->enhanced_prompt entries")
        # print examples of the first few mapping keys, for debugging matching problems
        if mapping:
            sample_keys = list(mapping.keys())[:3]
            print(f"DEBUG: Sample concept keys:")
            for i, key in enumerate(sample_keys):
                print(f"  {i+1}: {repr(key[:100])}...")
        return mapping


    def _normalize_text(self, s: str) -> str:
        # normalize text, handle LaTeX escape differences
        normalized = s.strip()
        # handle LaTeX escape: \\\\begin -> \\begin, \\\\\\\\ -> \\\\
        normalized = normalized.replace('\\\\\\\\', '\\\\').replace('\\\\begin', '\\begin').replace('\\\\end', '\\end')
        # remove extra spaces
        normalized = ' '.join(normalized.split())
        return normalized
    

    def _can_use_actor_rollout(self):
        """check if the current training actor model can be used for generation"""
        return (self.trainer is not None and 
                hasattr(self.trainer, 'actor_rollout_wg') and 
                self.trainer.actor_rollout_wg is not None)

    def _generate_enhanced_responses_with_rollout(self, enhanced_prompt, n=2):
        """Generate enhanced responses using the training rollout system"""
        if not self._can_use_actor_rollout():
            print(f"❌ CONCEPT: Actor rollout not available")
            return None
        
        try:
            # convert enhanced_prompt to the format expected by rollout
            from verl import DataProto
            
            # create chat messages like the training data
            messages = [
                {"role": "system", "content": "You are a helpful assistant that solves multiple-choice math questions with step-by-step reasoning."},
                {"role": "user", "content": enhanced_prompt}
            ]
            
            # tokenize the enhanced prompt
            enhanced_prompt_tokens = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            
            # create batch for generation - Ray requires batch_size divisible by num_workers
            # Get the number of workers from trainer config
            num_workers = getattr(self.trainer.config.trainer, 'n_gpus_per_node', 2)

            # For concept augmentation: batch_size = n (total responses needed)
            # With concept_augmentation flag, each worker generates n=1 per prompt
            # So batch_size=n ensures we get n total responses across all workers
            batch_size = n  # e.g., n=4 with 2 workers → 4 prompts, 2 per worker, 1 response each = 4 total
            
            prompts_batch = torch.tensor([enhanced_prompt_tokens] * batch_size, dtype=torch.long)
            attention_mask = torch.ones_like(prompts_batch)
            
            seq_len = prompts_batch.shape[1]
            position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            
            # use exact same format as training in ray_trainer.py line 1394-1404
            from tensordict import TensorDict
            
            gen_batch = DataProto(batch=TensorDict({
                'input_ids': prompts_batch,
                'attention_mask': attention_mask,
                'position_ids': position_ids
            }, batch_size=batch_size))  # batch_size divisible by num_workers
            
            print(f"🔧 CONCEPT: Created batch_size={batch_size} for {num_workers} workers")
            
            # dynamically read generation parameters from training config
            gen_params = {}
            if hasattr(self, 'trainer') and self.trainer and hasattr(self.trainer, 'config'):
                rollout_config = getattr(self.trainer.config, 'actor_rollout_ref', {}).get('rollout', {})
                data_config = getattr(self.trainer.config, 'data', {})
                gen_params = {
                    'temperature': getattr(rollout_config, 'temperature', 0.7),
                    'top_p': getattr(rollout_config, 'top_p', 0.9),
                    'max_new_tokens': getattr(data_config, 'max_response_length', 1024),
                    'do_sample': True,  # keep sampling for diversity
                }
                print(f"🔧 CONCEPT: Using training config generation params: {gen_params}")
            else:
                # fallback to default values
                gen_params = {
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'max_new_tokens': 1024,
                    'do_sample': True,
                }
                print(f"🔧 CONCEPT: Using default generation params: {gen_params}")
            
            gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                'recompute_log_prob': False,
                'concept_augmentation': True,  # special flag to disable batch expansion
                **gen_params  # use dynamic parameters from training config
            }
            
            print(f"🔄 CONCEPT: Calling actor_rollout_wg.generate_sequences with {n} samples")
            
            # generate using the training rollout system
            # multi-GPU fix (CORE-CR): the concept batch size (= #all-wrong groups this step)
            # is variable and may not divide n_gpus, which crashes DataProto.chunk during
            # dispatch (e.g. size 2 with 4 GPUs). Pad to a multiple of world_size, then unpad.
            from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
            _cc_ws = getattr(self.trainer.actor_rollout_wg, 'world_size', 1)
            gen_batch_padded, _cc_pad = pad_dataproto_to_divisor(gen_batch, _cc_ws)
            gen_output_padded = self.trainer.actor_rollout_wg.generate_sequences(gen_batch_padded)
            gen_output = unpad_dataproto(gen_output_padded, pad_size=_cc_pad)
            
            # extract responses from the output
            response_tokens = gen_output.batch['responses']  # shape: [actual_n, seq_len]
            
            aug_responses = []
            for i in range(min(response_tokens.shape[0], n)):  # only take the first n responses
                # decode the response tokens
                response_text = self.tokenizer.decode(response_tokens[i], skip_special_tokens=True)
                aug_responses.append(response_text.strip())
            
            print(f"✅ CONCEPT: Successfully generated {len(aug_responses)} responses (requested {n})")
            return aug_responses
            
        except Exception as e:
            print(f"❌ CONCEPT: Rollout generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_with_actor_rollout(self, prompts, sampling_params):
        """use the current training actor model to generate the answer"""
        if not self._can_use_actor_rollout():
            print(f"🔄 CONCEPT: Actor rollout not available, falling back to vLLM")
            return None
        
        try:
            print(f"🚀 CONCEPT: Using LATEST training model (current step) for concept generation")
            
            # construct the generation request - use the same interface as the training rollout
            from verl import DataProto
            import torch
            
            # simplified generation request
            batch_prompts = prompts  # directly use the prompt string
            
            # call the generate method of the actor (same as the training)
            outputs = self.trainer.actor_rollout_wg.generate_sequences(
                prompts=batch_prompts,
                max_new_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                num_return_sequences=sampling_params.n,
                stop_sequences=sampling_params.stop or []
            )
            
            print(f"✅ CONCEPT: Generated {len(outputs)} responses using LATEST model weights")
            return outputs
            
        except Exception as e:
            print(f"WARN: actor rollout generation failed: {e}, falling back to vLLM")
            return None

    def _get_current_model_path(self):
        """get the model path of the current training step"""
        if (self.trainer is not None and 
            hasattr(self.trainer, 'checkpoints_dir') and 
            hasattr(self.trainer, 'global_step')):
            
            # find the latest checkpoint
            import os
            import glob
            
            ckpt_dir = self.trainer.checkpoints_dir
            if os.path.exists(ckpt_dir):
                # find the latest step checkpoint
                pattern = os.path.join(ckpt_dir, "checkpoint_*")
                checkpoints = glob.glob(pattern)
                if checkpoints:
                    # sort by step number, get the latest
                    latest_ckpt = max(checkpoints, key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
                    print(f"🔄 CONCEPT: Found latest checkpoint: {latest_ckpt}")
                    return latest_ckpt
        
        # if no checkpoint found, use the original model path
        print(f"🔄 CONCEPT: No checkpoints found, using initial model: {self.model_path}")
        return self.model_path

    def _ensure_llm(self):
        """ensure the vLLM is initialized - use the latest model"""
        # note: we still need to initialize vLLM as fallback even if actor rollout is available
        can_use_actor = self._can_use_actor_rollout()
        if can_use_actor:
            print(f"🚀 CONCEPT: Will use LATEST training actor model directly (with vLLM fallback)")
            
        # alternative: get the latest checkpoint path and initialize the new vLLM
        current_model_path = self._get_current_model_path()
        
        # if the model path changed, reinitialize the vLLM
        if (self._llm is not None and 
            hasattr(self, '_current_model_path') and 
            self._current_model_path != current_model_path):
            print(f"🔄 CONCEPT: Model path changed, reinitializing vLLM")
            self._llm = None
        
        if self._llm is not None:
            return True
            
        if not _VLLM_OK:
            print("WARN: vLLM not available; concept augmentation disabled")
            print(f"DEBUG: _VLLM_OK = {_VLLM_OK}")
            return False
        if not current_model_path:
            print("WARN: model_path not provided; concept augmentation disabled")
            return False
        
        print(f"🔧 CONCEPT DEBUG: About to initialize vLLM")
        print(f"🔧 CONCEPT DEBUG: _VLLM_OK = {_VLLM_OK}")
        print(f"🔧 CONCEPT DEBUG: current_model_path = {current_model_path}")
        
        try:
            print(f"🔄 CONCEPT: Initializing vLLM with LATEST model: {current_model_path}")
            # read vLLM parameters from training config, keep consistent with training script
            vllm_config = {}
            if hasattr(self, 'trainer') and self.trainer and hasattr(self.trainer, 'config'):
                rollout_config = getattr(self.trainer.config, 'actor_rollout_ref', {}).get('rollout', {})
                vllm_config = {
                    'tensor_parallel_size': getattr(rollout_config, 'tensor_model_parallel_size', 1),
                    'gpu_memory_utilization': getattr(rollout_config, 'gpu_memory_utilization', 0.3),
                    'trust_remote_code': True,
                }
                print(f"📝 CONCEPT: Using training config vLLM params: {vllm_config}")
            else:
                # fallback to default configuration
                vllm_config = {
                    'tensor_parallel_size': 1,
                    'gpu_memory_utilization': 0.3,
                    'trust_remote_code': True,
                }
                print(f"📝 CONCEPT: Using default vLLM params: {vllm_config}")
            
            print(f"🔧 CONCEPT DEBUG: About to create LLM instance with config: {vllm_config}")
            self._llm = LLM(model=current_model_path, **vllm_config)
            self._current_model_path = current_model_path
            print(f"✅ CONCEPT: vLLM initialized with LATEST model weights")
            return True
            
        except Exception as e:
            print(f"❌ CONCEPT ERROR: vLLM initialization failed: {e}")
            print(f"DEBUG: Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            self._llm = None
            
            # if vLLM initialization failed but actor rollout is available, still return True
            if can_use_actor:
                print(f"⚠️ CONCEPT: vLLM fallback failed but actor rollout available")
                return True
            else:
                return False

    def _extract_question_key(self, prompt_str: str) -> str:
        """extract the key of the question for grouping"""
        try:
            # 🔧 FIX: extract the actual question, not the example question
            # find the actual question after the "---" separator
            if "---" in prompt_str and "Question:" in prompt_str:
                # find the content after the separator
                after_separator = prompt_str.split("---", 1)[-1]
                if "Question:" in after_separator and "A." in after_separator:
                    start = after_separator.find("Question:") + len("Question:")
                    end = after_separator.find("A.", start)
                    if end > start:
                        question_text = after_separator[start:end].strip()
                        return self._normalize_text(question_text)
            
            # alternative: if no separator, use the last "Question:"
            elif "Question:" in prompt_str and "A." in prompt_str:
                # find the last "Question:"
                question_positions = [i for i in range(len(prompt_str)) if prompt_str[i:].startswith("Question:")]
                if question_positions:
                    last_question_pos = question_positions[-1] + len("Question:")
                    end = prompt_str.find("A.", last_question_pos)
                    if end > last_question_pos:
                        question_text = prompt_str[last_question_pos:end].strip()
                        return self._normalize_text(question_text)
        except Exception as e:
            print(f"⚠️  Warning: question key extraction failed: {e}")
            pass
        return str(hash(prompt_str))

    def _find_enhanced_prompt(self, prompt_str: str) -> str:
        """find the enhanced prompt corresponding to the original prompt"""
        # extract the question part using the same logic as _extract_question_key
        q_text = None
        try:
            # 🔧 FIX: use the same logic as _extract_question_key to extract the actual question
            if "---" in prompt_str and "Question:" in prompt_str:
                # find the content after the separator
                after_separator = prompt_str.split("---", 1)[-1]
                if "Question:" in after_separator and "A." in after_separator:
                    start = after_separator.find("Question:") + len("Question:")
                    end = after_separator.find("A.", start)
                    if end > start:
                        q_text = self._normalize_text(after_separator[start:end].strip())
            
            # alternative: if no separator, use the last "Question:"
            elif "Question:" in prompt_str and "A." in prompt_str:
                # find the last "Question:"
                question_positions = [i for i in range(len(prompt_str)) if prompt_str[i:].startswith("Question:")]
                if question_positions:
                    last_question_pos = question_positions[-1] + len("Question:")
                    end = prompt_str.find("A.", last_question_pos)
                    if end > last_question_pos:
                        q_text = self._normalize_text(prompt_str[last_question_pos:end].strip())
        except Exception:
            pass
        
        if q_text:            
            # find the enhanced prompt corresponding to the original prompt in concept_data
            # try to match exactly
            if q_text in self.concept_data:
                return self.concept_data[q_text]['enhanced_prompt']
            
            # fuzzy match: handle the difference of LaTeX format (\\\\ and \ problems)
            for key, data in self.concept_data.items():
                # simplify the comparison: remove the difference of LaTeX format
                q_simple = q_text.replace('\\\\', '\\').replace('\\', '').replace('$', '').replace('{', '').replace('}', '')
                key_simple = key.replace('\\\\', '\\').replace('\\', '').replace('$', '').replace('{', '').replace('}', '')
                
                if q_simple == key_simple or q_simple in key_simple or key_simple in q_simple:
                    return data['enhanced_prompt']
        return None

    def __call__(self, data: DataProto):
        # if rm_scores is provided, return directly
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        import torch
        import random
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        score_record = []
        already_print_data_sources = {}
        
        # for wandb record the statistics
        wandb_stats = {
            'concept_augmentation': {
                'total_groups': 0,
                'all_wrong_groups': 0,
                'successfully_augmented_groups': 0,
                'total_responses_enhanced': 0,
                'total_reward_boost': 0.0,
                'examples': []
            }
        }
        
        # detailed record of the training process for each sample
        detailed_samples = []
        step_counter = getattr(self, 'step_counter', 0)
        self.step_counter = step_counter + 1

        # Step 1: calculate the base score and group by question
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
            
            # Debug: print the decoded result
            if i == 0:  # only print the first example
                print(f"DEBUG: Decoded prompt: {prompt_str[:200]}...")

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            # get pure response part
            response_str = sequences_str[len(prompt_str):].strip()
            
            # use the compute_score function during training, directly get extracted_answer
            # 🚨 FIX: use response_str instead of sequences_str, avoid extracting wrong answer from prompt
            base_score, extracted_answer = self.compute_score(
                data_source=data_source,
                solution_str=response_str,  # only pass the response part, not the prompt
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

            # group by question
            key = self._extract_question_key(prompt_str)
            groups[key].append(item)

        # Step 2: initialize all the reward to the base score
        for item in base_items:
            reward_tensor[item['index'], item['valid_response_length'] - 1] = float(item['base_score'])

        # Step 3: concept enhanced processing (only all wrong groups)
        wandb_stats['concept_augmentation']['total_groups'] = len(groups)
        
        # check if we can use actor rollout for concept augmentation
        if hasattr(self, 'trainer') and self.trainer and hasattr(self.trainer, 'actor_rollout_wg'):
            print("🚀 CONCEPT: Will use training actor_rollout_wg for concept augmentation")
            augmented_groups = 0
            for key, group_items in groups.items():
                # check if all answer wrong and at least 2 answers
                group_scores = [item['base_score'] for item in group_items]
                all_wrong = all(s <= 0.0 for s in group_scores) and len(group_items) >= 2
                
                if all_wrong:
                    wandb_stats['concept_augmentation']['all_wrong_groups'] += 1
                
                if not all_wrong:
                    continue

                # Found all-wrong group, attempting concept augmentation
                
                # record the detailed information of the all wrong group (full content)
                example_data = {
                    'group_size': len(group_items),
                    'original_scores': group_scores,
                    'original_responses': [item['sequences_str'] for item in group_items],  # full response
                    'augmented': False
                }
                
                # find the corresponding enhanced prompt
                prompt_str = group_items[0]['prompt_str']
                aug_prompt = self._find_enhanced_prompt(prompt_str)

                if not aug_prompt:
                    wandb_stats['concept_augmentation']['examples'].append(example_data)
                    continue
                
                # Using enhanced prompt for concept augmentation
                
                # record the used full prompt (not truncated)
                example_data['enhanced_prompt'] = aug_prompt
                example_data['original_prompt'] = prompt_str
                
                try:
                    # use training rollout to generate enhanced responses directly
                    print(f"🔄 CONCEPT: Generating enhanced responses using training rollout")
                    
                    # construct enhanced prompts for the group (use rollout n=2 to get 2 new responses)
                    aug_responses = self._generate_enhanced_responses_with_rollout(aug_prompt, n=2)
                    
                    if aug_responses is None or len(aug_responses) < 2:
                        print(f"❌ CONCEPT: Failed to generate enhanced responses")
                        wandb_stats['concept_augmentation']['examples'].append(example_data)
                        continue
                    
                    print(f"✅ CONCEPT: Generated {len(aug_responses)} enhanced responses using training rollout")
                    
                    # record the generated enhanced responses
                    example_data['generated_responses'] = aug_responses
                    print(f"✅ CONCEPT: Generated {len(aug_responses)} enhanced responses with SAME model weights as training")
                    
                except Exception as e:
                    print(f"WARN: concept augmentation generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    wandb_stats['concept_augmentation']['examples'].append(example_data)
                    continue

                # calculate the score of the enhanced responses
                aug_scores = []
                for r in aug_responses:
                    seq = aug_prompt + r
                    gt = group_items[0]['ground_truth']
                    base_aug_score = self.compute_score(data_source='math', solution_str=seq, ground_truth=gt, extra_info=None)
                    # add 0.4 extra reward
                    final_aug_score = float(base_aug_score) + 0.4
                    aug_scores.append(final_aug_score)
                    print(f"DEBUG: Augmented response score: {base_aug_score:.3f} + 0.4 = {final_aug_score:.3f}")

                # randomly select 2 positions to replace
                replace_indices = random.sample(range(len(group_items)), k=min(2, len(group_items)))
                
                # replace the reward and the full response text at the selected positions
                replaced_details = []
                for j, replace_idx in enumerate(replace_indices):
                    if j < len(aug_scores):
                        item = group_items[replace_idx]
                        # Store original data for detailed logging before modification
                        item['original_prompt_str'] = item['prompt_str']  # original prompt
                        item['original_response_str'] = item['response_str']  # original response
                        item['original_extracted_answer'] = item['extracted_answer']  # original extracted answer
                        item['original_base_score'] = item['base_score']  # original score

                        old_reward = float(item['base_score'])
                        new_reward = float(aug_scores[j])
                        # 🔥 really replace the response tokens instead of only the string
                        new_response_text = aug_responses[j].strip()
                        new_response_tokens = self.tokenizer.encode(new_response_text, add_special_tokens=False)
                        original_idx = item['index']
                        
                        # replace the actual tokens in data.batch
                        max_response_length = data.batch['responses'].shape[-1]
                        new_response_length = min(len(new_response_tokens), max_response_length)
                        
                        with torch.no_grad():
                            # clear and write the new tokens
                            data.batch['responses'][original_idx].fill_(self.tokenizer.pad_token_id or 0)
                            if new_response_length > 0:
                                new_tensor = torch.tensor(new_response_tokens[:new_response_length], 
                                                         dtype=data.batch['responses'].dtype,
                                                         device=data.batch['responses'].device)
                                data.batch['responses'][original_idx][:new_response_length] = new_tensor
                            
                            # update the attention mask
                            prompt_length = data.batch['prompts'].shape[-1]
                            data.batch['attention_mask'][original_idx, prompt_length:] = 0
                            if new_response_length > 0:
                                response_end = min(prompt_length + new_response_length, data.batch['attention_mask'].shape[-1])
                                data.batch['attention_mask'][original_idx, prompt_length:response_end] = 1
                            
                            # update the reward tensor
                            reward_tensor[original_idx, :] = 0.0
                            if new_response_length > 0:
                                reward_pos = min(new_response_length - 1, reward_tensor.shape[-1] - 1)
                                reward_tensor[original_idx, reward_pos] = new_reward
                        
                        # construct the full enhanced prompt with proper chat template format
                        original_full_prompt = item['prompt_str']
                        
                        # replace the question content while keeping the system+user structure
                        if "---" in original_full_prompt:
                            # format: system + user + "---" + question
                            before_separator = original_full_prompt.split("---", 1)[0] + "---"
                            enhanced_full_prompt = before_separator + "\n" + aug_prompt
                        else:
                            # fallback: replace the entire content (this shouldn't happen normally)
                            enhanced_full_prompt = aug_prompt
                        
                        # update the item information
                        item['valid_response_length'] = new_response_length
                        item['prompt_str'] = enhanced_full_prompt  # full concept-enhanced prompt
                        item['response_str'] = new_response_text  # new response
                        item['sequences_str'] = enhanced_full_prompt + new_response_text  # full sequence
                        
                        # 🐞 Bugfix: Re-compute extracted_answer for the new response
                        # 🚨 FIX: use response part instead of full sequence, avoid extracting wrong answer from prompt
                        new_response_str = item['sequences_str'][len(enhanced_full_prompt):].strip()
                        _new_score, new_extracted_answer = self.compute_score(
                            data_source=item['data_source'],
                            solution_str=new_response_str,  # only use the response part
                            ground_truth=item['ground_truth'],
                            extra_info=item['extra_info'],
                            return_extracted_answer=True
                        )
                        item['extracted_answer'] = new_extracted_answer
                        print(f"DEBUG: Replaced response for index {item['index']}. New extracted answer: {new_extracted_answer}")

                        replaced_details.append({
                            'position': replace_idx,
                            'index': item['index'],
                            'old_reward': old_reward,
                            'new_reward': new_reward,
                            'boost': new_reward - old_reward
                        })
                        
                        wandb_stats['concept_augmentation']['total_responses_enhanced'] += 1
                        wandb_stats['concept_augmentation']['total_reward_boost'] += (new_reward - old_reward)
                        
                        print(f"DEBUG: Replaced reward at index {item['index']}: {old_reward:.3f} -> {new_reward:.3f}")
                
                # complete the record of this group
                example_data.update({
                    'augmented': True,
                    'aug_scores': aug_scores,
                    'replaced_positions': replace_indices,
                    'replacement_details': replaced_details
                })
                wandb_stats['concept_augmentation']['examples'].append(example_data)
                
                augmented_groups += 1
                
            wandb_stats['concept_augmentation']['successfully_augmented_groups'] = augmented_groups
            print(f"✅ CONCEPT: Successfully augmented {augmented_groups} groups using training rollout")
        else:
            print("❌ CONCEPT: Training actor_rollout_wg not available, concept augmentation disabled")

        # Step 4: generate the score_record
        for item in base_items:
            data_source = item['data_source']
            
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                final_reward = reward_tensor[item['index'], item['valid_response_length'] - 1].item()
                print(f"[CONCEPT] {item['sequences_str'][:100]}...")
                if final_reward != item['base_score']:
                    print(f"[REWARD] Base: {item['base_score']:.3f} → Enhanced: {final_reward:.3f}")
                else:
                    print(f"[REWARD] {final_reward:.3f}")
            
            final_score = reward_tensor[item['index'], item['valid_response_length'] - 1].item()
            # 🔥 FIX: Correctly detect concept-enhanced samples by checking 'original_base_score' field
            # This field is set when responses are replaced (line 771), even if scores happen to match
            was_replaced = 'original_base_score' in item
            record = {
                "sequences_str": item['sequences_str'],
                "extracted_answer": item['extracted_answer'],  # add extracted answer
                "ground_truth": item['ground_truth'],
                "index": item['extra_info']["index"] if item['extra_info'] else None,
                # sample_idx is the in-batch index after balancing; used by PPO to locate replacements
                "sample_idx": item['index'],
                "score": final_score,
                "original_score": item.get('original_base_score', item['base_score']),
                "concept_enhanced": was_replaced  # Use consistent key name for ray_trainer.py
            }
            score_record.append(record)

        # directly log the rollout data to wandb during training
        self._log_rollout_to_wandb_directly(base_items, reward_tensor, groups)
        
        print(f"📊 Step {self.step_counter}: data is logged to wandb directly during training")

        return reward_tensor, score_record

    def _log_rollout_to_wandb_directly(self, base_items, reward_tensor, groups):
        """directly log the rollout data to wandb during training"""
        try:
            import wandb
            
            # check if wandb can be used
            if not hasattr(wandb, 'log') or wandb.run is None:
                return
            
            # prepare wandb table data
            rollout_rows = []
            all_wrong_uids = set()
            
            # 🔧 FIX 1: create stable question ID mapping
            if not hasattr(self, '_question_id_counter'):
                self._question_id_counter = 0
            if not hasattr(self, '_question_key_to_id'):
                self._question_key_to_id = {}
            
            # 🔧 FIX 2: create cumulative rollout data list, create new table each time
            if not hasattr(self, '_cumulative_rollout_data'):
                self._cumulative_rollout_columns = ["step", "uid", "idx", "all_wrong_group", "concept_enhanced", 
                                                  "current_prompt", "current_response", 
                                                  "original_prompt", "original_response", 
                                                  "current_extracted_answer", "original_extracted_answer", 
                                                  "ground_truth", 
                                                  "current_score", "original_score", 
                                                  "is_correct"]
                # 🔧 FIX: use data list instead of fixed table object
                self._cumulative_rollout_data = []
            
            # assign ID to new question keys
            for key in groups.keys():
                if key not in self._question_key_to_id:
                    self._question_id_counter += 1
                    self._question_key_to_id[key] = self._question_id_counter
            
            # 🔧 FIX 3: correctly find all wrong groups (based on original base_score, not concept enhanced score)
            for key, group_items in groups.items():
                # get all original base_score (scores before concept enhancement)
                base_scores = []
                for item in group_items:
                    base_scores.append(float(item['base_score']))
                
                # Only show debug info for unexpected group sizes in first few steps
                if self.step_counter <= 2 and len(group_items) not in [1, 4]:
                    print(f"🔍 DEBUG: Unexpected group size - Group {key[:30]}... has {len(group_items)} items")
                
                # if all original scores are 0 and there are multiple samples in the group, mark as all wrong group
                if all(s <= 0.0 for s in base_scores) and len(group_items) >= 2:
                    all_wrong_uids.add(key)
            
            # record data by group, ensure different attempts for the same question are placed together
            import random
            sampled_groups = random.sample(list(groups.items()), min(12, len(groups)))  # sample 12 groups
            
            for question_key, group_items in sampled_groups:
                is_all_wrong_group = question_key in all_wrong_uids
                
                # 🔧 FIX 2: sort items by index within each group, ensure consistent order, and index starts from 0
                sorted_items = sorted(group_items, key=lambda x: x['index'])
                
                for local_idx, item in enumerate(sorted_items):
                    # 🔧 FIXED: Use tensor_score as current_score, and original_base_score as original_score
                    current_score = reward_tensor[item['index'], item['valid_response_length'] - 1].item()
                    original_score = item.get('original_base_score', item['base_score'])  # use saved original score
                    
                    is_concept_enhanced = 'original_base_score' in item  # if this field exists, it means it has been enhanced
                    final_score = float(current_score)  # used for subsequent logic
                    
                    # If not enhanced, original is same as final.
                    # If enhanced, the 'original_*' keys should exist from the __call__ method.
                    original_prompt = item.get('original_prompt_str', item['prompt_str'])
                    original_response = item.get('original_response_str', item['response_str'])
                    original_extracted_answer = item.get('original_extracted_answer', item['extracted_answer'])
                    original_score = item.get('original_base_score', item['base_score'])
                    
                    # 🔧 FIX 1: use stable question ID, ensure each question has a unique ID and is in order
                    question_id = self._question_key_to_id[question_key]
                    uid_str = f"q_{question_id:05d}"
                    
                    rollout_rows.append([
                        int(self.step_counter),  # step
                        uid_str,  # uid - use simplified question id
                        int(local_idx),  # idx - local index within the same question (0,1,2,3...)
                        bool(is_all_wrong_group),  # all_wrong_group
                        bool(is_concept_enhanced),  # concept_enhanced/replaced
                        # current prompt and response (possibly concept-enhanced) - 🔧 FIX: display full content, not truncated
                        str(item['prompt_str']),  # current_prompt - full display
                        str(item['response_str']),  # current_response - full display
                        # original prompt and response
                        str(original_prompt),  # original_prompt - full display
                        str(original_response),   # original_response
                        # process extracted_answer
                        str(item['extracted_answer']) if item['extracted_answer'] and item['extracted_answer'] != '' else 'N/A',  # current_extracted_answer
                        str(original_extracted_answer) if original_extracted_answer and original_extracted_answer != '' else 'N/A',  # original_extracted_answer
                        # process ground_truth - possibly in list format
                        (str(item['ground_truth'][0]) if isinstance(item['ground_truth'], list) and item['ground_truth'] 
                         else str(item['ground_truth'])) if item['ground_truth'] is not None else 'N/A',  # ground_truth
                        float(current_score),  # current_score - use current score from tensor
                        float(original_score), # original_score - use saved original score
                        # 🔧 FIXED: Use original score to determine is_correct (score > 0 means correct)
                        bool(float(original_score) > 0.0)  # is_correct - based on original score (score > 0 means correct)
                    ])
            
            if rollout_rows:
                # 🔧 FIX: add new data to cumulative data list
                self._cumulative_rollout_data.extend(rollout_rows)
                
                # 🔍 DEBUG: check cumulative data rows
                total_rows = len(self._cumulative_rollout_data)
                # Added rollout data to wandb table
                
                # 🔧 FIX: create new cumulative table object each time, so wandb can update correctly
                cumulative_table = wandb.Table(columns=self._cumulative_rollout_columns, data=self._cumulative_rollout_data)
                
                # record cumulative table and statistics
                metrics = {
                    "training_rollout_samples_all_steps": cumulative_table,  # cumulative data for all steps
                    "training_rollout_samples_current_step": wandb.Table(columns=self._cumulative_rollout_columns, 
                                                                        data=rollout_rows),  # data for current step only
                    "training/num_samples": len(rollout_rows),
                    "training/num_all_wrong_groups": len(all_wrong_uids),
                    "training/num_concept_enhanced": sum(1 for r in rollout_rows if r[4])
                }
                
                # use trainer's global step, if available
                current_step = self.step_counter
                if self.trainer and hasattr(self.trainer, 'global_steps'):
                    current_step = self.trainer.global_steps
                
                wandb.log(metrics, step=current_step)
                print(f"[wandb] Logged {len(rollout_rows)} training rollout samples at step {self.step_counter}")
                
                # debug information - check data quality and repair effect
                # 🔧 temporary force display debug information
                if True:  # self.step_counter <= 5:
                    sample_row = rollout_rows[0] if rollout_rows else None
                    if sample_row:
                        print(f"[DEBUG] Sample row: uid={sample_row[1]}, idx={sample_row[2]}, all_wrong={sample_row[3]}, concept_enhanced={sample_row[4]}")
                        print(f"[DEBUG]   current_extracted={sample_row[9]}, original_extracted={sample_row[10]}, ground_truth={sample_row[11]}")
                        print(f"[DEBUG]   current_score={sample_row[12]}, original_score={sample_row[13]}")
                    
                    # display some group statistics
                    print(f"[DEBUG] Total groups: {len(groups)}, All-wrong groups: {len(all_wrong_uids)}")
                    if rollout_rows:
                        # display full information for each group
                        group_stats = {}
                        for row in rollout_rows[:8]:  # display first 8 rows
                            uid = row[1]
                            if uid not in group_stats:
                                group_stats[uid] = []
                            group_stats[uid].append({
                                'idx': row[2], 
                                'all_wrong': row[3],
                                'concept_enhanced': row[4],
                                'current_score': row[12],
                                'current_extracted': row[9]
                            })
                        for uid, items in group_stats.items():
                            print(f"[DEBUG] Group {uid}: {items}")
                
        except Exception as e:
            print(f"[wandb] WARN: Training rollout logging failed: {e}")
            import traceback
            print(f"[DEBUG] Full error: {traceback.format_exc()}")


# ---------------- Self-Consistent Concept RewardManager ----------------
class SelfConsistentConceptRewardManager:
    """
    Self-Consistent Concept enhanced reward manager (refined strategy):
    
    Judgment standard:
    - 4 answers completely consistent → considered all correct, not processed
    - 4 answers not completely consistent → considered all wrong, concept+example enhanced
    
    Smart replacement strategy:
    - use concept+example to generate 2 new responses, add 0.4 reward (regardless of right or wrong)
    - the replacement logic of the original 4 answers:
      * 2 answers are the same and ≠unknown → replace the other 2 different ones
      * 3 answers are the same → replace 1 different one + randomly replace 1 of the 3
      * all answers are different → randomly replace 2 of them
      * priority: always replace unknown first regardless of the situation
    """

    def __init__(self, tokenizer, num_examine, compute_score=None,
                 concept_file: str = "data/quizzes/concept_quizzes.jsonl",
                 model_path: str = None,
                 trainer=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or deepscaler.compute_score
        self.model_path = model_path
        self.trainer = trainer
        self.concept_data = self._load_concept_data(concept_file)
        self._llm = None
        
        print(f"SELF-CONSISTENT: SelfConsistentConceptRewardManager initialized (fine-grained strategy)")
        print(f"📝 SELF-CONSISTENT: 4 answers completely consistent → all correct, not consistent → all wrong and concept+example enhanced")
        print(f"🎯 SELF-CONSISTENT: Smart replacement strategy: always replace unknown first, smartly select positions based on answer frequency")

    def _load_concept_data(self, concept_file):
        """load the concept data, build the mapping from original question to enhanced prompt"""
        mapping = {}
        try:
            import json
            with open(concept_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    
                    original_q = item.get('original_question', '')
                    enhanced_q = item.get('enhanced_question') or item.get('question', '')
                    options = item.get('options', [])
                    
                    if original_q and enhanced_q:
                        if options:
                            enhanced_prompt = enhanced_q + "\\n" + "\\n".join(options) + "\\n\\nAnswer: "
                        else:
                            enhanced_prompt = enhanced_q + "\\n\\nAnswer: "
                        
                        key = self._normalize_text(original_q)
                        mapping[key] = {
                            'enhanced_prompt': enhanced_prompt,
                            'original_question': original_q,
                            'options': options
                        }
        except Exception as e:
            print(f"WARN: failed to load concept file {concept_file}: {e}")
        print(f"DEBUG: SelfConsistentConceptRewardManager loaded {len(mapping)} question->enhanced_prompt entries")
        return mapping

    def _normalize_text(self, s: str) -> str:
        # normalize text, handle LaTeX escape differences
        normalized = s.strip()
        # handle LaTeX escape: \\\\begin -> \\begin, \\\\\\\\ -> \\\\
        normalized = normalized.replace('\\\\\\\\', '\\\\').replace('\\\\begin', '\\begin').replace('\\\\end', '\\end')
        # remove extra spaces
        normalized = ' '.join(normalized.split())
        return normalized

    def _extract_answer(self, response_str: str) -> str:
        """extract the answer options (A/B/C/D) from the response - use verl's extract_answer function"""
        from verl.utils.reward_score.utils.utils import extract_answer
        
        # use verl's standard answer extraction function (designed for MCQ)
        extracted = extract_answer(response_str, is_mcq=True)
        
        if extracted is not None and extracted.upper() in ['A', 'B', 'C', 'D']:
            return extracted.upper()
        
        return "UNKNOWN"

    def _check_consistency(self, responses):
        """
        check the consistency of 4 answers (refined judgment standard)
        
        Judgment standard:
        - 4 answers completely consistent → considered all correct (is_consistent=True)
        - 4 answers not completely consistent → considered all wrong (is_consistent=False)
        """
        answers = [self._extract_answer(resp) for resp in responses]
        
        # check if all 4 answers are completely consistent (including UNKNOWN)
        unique_answers = set(answers)
        is_consistent = len(unique_answers) == 1
        
        print(f"DEBUG: Consistency check - answers: {answers}, unique: {unique_answers}, consistent: {is_consistent}")
        
        return is_consistent, answers
    
    def _get_smart_replacement_indices(self, answers):
        """
        determine the replacement positions based on the refined strategy
        
        Strategy:
        - 2 answers are the same and ≠unknown → replace the other 2 different ones
        - 3 answers are the same → replace 1 different one + randomly replace 1 of the 3
        - all answers are different → randomly replace 2 of them
        - priority: always replace unknown first regardless of the situation
        """
        import random
        from collections import Counter
        
        # find the unknown positions (highest priority)
        unknown_indices = [i for i, ans in enumerate(answers) if ans == "UNKNOWN"]
        
        # if there are >=2 unknowns, replace them first
        if len(unknown_indices) >= 2:
            replace_indices = random.sample(unknown_indices, 2)
            print(f"DEBUG: Strategy - Found {len(unknown_indices)} unknowns, replacing 2 of them: {replace_indices}")
            return replace_indices
        
        # if there is 1 unknown, replace it
        if len(unknown_indices) == 1:
            remaining_indices = [i for i in range(4) if i not in unknown_indices]
            second_replace = random.choice(remaining_indices)
            replace_indices = [unknown_indices[0], second_replace]
            print(f"DEBUG: Strategy - Found 1 unknown at {unknown_indices[0]}, also replacing {second_replace}")
            return replace_indices
        
        # if there is no unknown, analyze the frequency of the answers
        non_unknown_answers = [ans for ans in answers if ans != "UNKNOWN"]
        non_unknown_counts = Counter(non_unknown_answers)
        
        # find the most frequent answer
        if non_unknown_counts:
            most_common_answer, max_count = non_unknown_counts.most_common(1)[0]
            
            if max_count == 3:
                # 3 answers are the same
                same_indices = [i for i, ans in enumerate(answers) if ans == most_common_answer]
                different_indices = [i for i, ans in enumerate(answers) if ans != most_common_answer]
                
                # replace 1 different one + randomly replace 1 of the 3
                replace_different = different_indices[0]  # only 1 different
                replace_same = random.choice(same_indices)
                replace_indices = [replace_different, replace_same]
                print(f"DEBUG: Strategy - 3 same answers '{most_common_answer}', replacing different ({replace_different}) + one same ({replace_same})")
                return replace_indices
                
            elif max_count == 2:
                # 2 answers are the same
                same_indices = [i for i, ans in enumerate(answers) if ans == most_common_answer]
                different_indices = [i for i, ans in enumerate(answers) if ans != most_common_answer]
                
                if len(different_indices) >= 2:
                    # replace the other 2 different ones
                    replace_indices = random.sample(different_indices, 2)
                    print(f"DEBUG: Strategy - 2 same answers '{most_common_answer}', replacing 2 different ones: {replace_indices}")
                    return replace_indices
                else:
                    # this case should not happen (2 same, but <2 different)
                    replace_indices = random.sample(range(4), 2)
                    print(f"DEBUG: Strategy - Unexpected case, random replacement: {replace_indices}")
                    return replace_indices
        
        # all answers are different
        replace_indices = random.sample(range(4), 2)
        print(f"DEBUG: Strategy - All different answers, random replacement: {replace_indices}")
        return replace_indices

    def _can_use_actor_rollout(self):
        """check if the actor model in the current training can be used for generation"""
        return (self.trainer is not None and 
                hasattr(self.trainer, 'actor_rollout_wg') and 
                self.trainer.actor_rollout_wg is not None)

    def _generate_with_actor_rollout(self, prompts, sampling_params):
        """use the actor model in the current training to generate responses"""
        if not self._can_use_actor_rollout():
            print(f"🔄 SELF-CONSISTENT: Actor rollout not available, falling back to vLLM")
            return None
        
        try:
            print(f"🚀 SELF-CONSISTENT: Using LATEST training model for concept generation")
            
            from verl import DataProto
            import torch
            
            batch_prompts = prompts
            
            outputs = self.trainer.actor_rollout_wg.generate_sequences(
                prompts=batch_prompts,
                max_new_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                num_return_sequences=sampling_params.n,
                stop_sequences=sampling_params.stop or []
            )
            
            print(f"✅ SELF-CONSISTENT: Generated {len(outputs)} responses using LATEST model weights")
            return outputs
            
        except Exception as e:
            print(f"WARN: actor rollout generation failed: {e}, falling back to vLLM")
            return None

    def _get_current_model_path(self):
        """get the model path of the current training step"""
        if (self.trainer is not None and 
            hasattr(self.trainer, 'checkpoints_dir') and 
            hasattr(self.trainer, 'global_step')):
            
            import os
            import glob
            
            ckpt_dir = self.trainer.checkpoints_dir
            if os.path.exists(ckpt_dir):
                pattern = os.path.join(ckpt_dir, "checkpoint_*")
                checkpoints = glob.glob(pattern)
                if checkpoints:
                    latest_ckpt = max(checkpoints, key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
                    print(f"🔄 SELF-CONSISTENT: Found latest checkpoint: {latest_ckpt}")
                    return latest_ckpt
        
        print(f"🔄 SELF-CONSISTENT: No checkpoints found, using initial model: {self.model_path}")
        return self.model_path

    def _ensure_llm(self):
        """ensure vLLM is initialized - use the latest model"""
        if self._can_use_actor_rollout():
            print(f"🚀 SELF-CONSISTENT: Will use LATEST training actor model directly")
            return True
            
        current_model_path = self._get_current_model_path()
        
        if (self._llm is not None and 
            hasattr(self, '_current_model_path') and 
            self._current_model_path != current_model_path):
            print(f"🔄 SELF-CONSISTENT: Model path changed, reinitializing vLLM")
            self._llm = None
        
        if self._llm is not None:
            return True
            
        if not _VLLM_OK:
            print("WARN: vLLM not available; self-consistent concept augmentation disabled")
            return False
        if not current_model_path:
            print("WARN: model_path not provided; self-consistent concept augmentation disabled")
            return False
        try:
            print(f"🔄 SELF-CONSISTENT: Initializing vLLM with LATEST model: {current_model_path}")
            # read vLLM parameters from training config, keep consistent with training script
            vllm_config = {}
            if hasattr(self, 'trainer') and self.trainer and hasattr(self.trainer, 'config'):
                rollout_config = getattr(self.trainer.config, 'actor_rollout_ref', {}).get('rollout', {})
                vllm_config = {
                    'tensor_parallel_size': getattr(rollout_config, 'tensor_model_parallel_size', 1),
                    'gpu_memory_utilization': getattr(rollout_config, 'gpu_memory_utilization', 0.3),
                    'trust_remote_code': True,
                }
                print(f"📝 SELF-CONSISTENT: Using training config vLLM params: {vllm_config}")
            else:
                # fallback to default configuration
                vllm_config = {
                    'tensor_parallel_size': 1,
                    'gpu_memory_utilization': 0.3,
                    'trust_remote_code': True,
                }
                print(f"📝 SELF-CONSISTENT: Using default vLLM params: {vllm_config}")
            
            self._llm = LLM(model=current_model_path, **vllm_config)
            self._current_model_path = current_model_path
            print(f"✅ SELF-CONSISTENT: vLLM initialized with LATEST model weights")
            return True
        except Exception as e:
            print(f"WARN: fail to init vLLM for self-consistent concept augmentation: {e}")
            self._llm = None
            return False

    def _extract_question_key(self, prompt_str: str) -> str:
        """extract the key of the question for grouping"""
        try:
            # 🔧 FIX: extract the actual question, not the example question
            # find the actual question after the "---" separator
            if "---" in prompt_str and "Question:" in prompt_str:
                # find the content after the separator
                after_separator = prompt_str.split("---", 1)[-1]
                if "Question:" in after_separator and "A." in after_separator:
                    start = after_separator.find("Question:") + len("Question:")
                    end = after_separator.find("A.", start)
                    if end > start:
                        question_text = after_separator[start:end].strip()
                        return self._normalize_text(question_text)
            
            # alternative: if no separator, use the last "Question:"
            elif "Question:" in prompt_str and "A." in prompt_str:
                # find the last "Question:"
                question_positions = [i for i in range(len(prompt_str)) if prompt_str[i:].startswith("Question:")]
                if question_positions:
                    last_question_pos = question_positions[-1] + len("Question:")
                    end = prompt_str.find("A.", last_question_pos)
                    if end > last_question_pos:
                        question_text = prompt_str[last_question_pos:end].strip()
                        return self._normalize_text(question_text)
        except Exception as e:
            print(f"⚠️  Warning: question key extraction failed: {e}")
            pass
        return str(hash(prompt_str))

    def _find_enhanced_prompt(self, prompt_str: str) -> str:
        """find the corresponding enhanced prompt based on the original prompt"""
        q_text = None
        try:
            # 🔧 FIX: use the same logic as _extract_question_key to extract the actual question
            if "---" in prompt_str and "Question:" in prompt_str:
                # find the content after the separator
                after_separator = prompt_str.split("---", 1)[-1]
                if "Question:" in after_separator and "A." in after_separator:
                    start = after_separator.find("Question:") + len("Question:")
                    end = after_separator.find("A.", start)
                    if end > start:
                        q_text = self._normalize_text(after_separator[start:end].strip())
            elif "Question:" in prompt_str and "A." in prompt_str:
                # alternative: find the last "Question:"
                question_positions = [i for i in range(len(prompt_str)) if prompt_str[i:].startswith("Question:")]
                if question_positions:
                    last_question_pos = question_positions[-1] + len("Question:")
                    end = prompt_str.find("A.", last_question_pos)
                    if end > last_question_pos:
                        q_text = self._normalize_text(prompt_str[last_question_pos:end].strip())
        except Exception:
            pass
        
        if q_text:
            print(f"DEBUG: Extracted question: {repr(q_text[:100])}...")
            
            if q_text in self.concept_data:
                print(f"DEBUG: Exact match found")
                return self.concept_data[q_text]['enhanced_prompt']
            else:
                print(f"DEBUG: No exact match found. Checking similar keys...")
                # display several most similar keys for debugging
                all_keys = list(self.concept_data.keys())
                similar_keys = [k for k in all_keys if q_text[:50] in k or k[:50] in q_text][:3]
                if similar_keys:
                    print(f"DEBUG: Similar keys found:")
                    for i, key in enumerate(similar_keys):
                        print(f"  Similar {i+1}: {repr(key[:100])}...")
                else:
                    print(f"DEBUG: No similar keys found. Sample keys:")
                    for i, key in enumerate(all_keys[:3]):
                        print(f"  Sample {i+1}: {repr(key[:100])}...")
            
            for key, data in self.concept_data.items():
                q_simple = q_text.replace('\\\\\\\\', '\\\\').replace('\\\\', '').replace('$', '').replace('{', '').replace('}', '')
                key_simple = key.replace('\\\\\\\\', '\\\\').replace('\\\\', '').replace('$', '').replace('{', '').replace('}', '')
                
                if q_simple == key_simple or q_simple in key_simple or key_simple in q_simple:
                    print(f"DEBUG: Fuzzy match found with key: {key[:50]}...")
                    return data['enhanced_prompt']
        
        print(f"DEBUG: No match found for question")
        return None

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        import torch
        import random
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        score_record = []
        already_print_data_sources = {}
        
        wandb_stats = {
            'self_consistent_concept': {
                'total_groups': 0,
                'inconsistent_groups': 0,
                'successfully_augmented_groups': 0,
                'total_responses_enhanced': 0,
                'total_reward_boost': 0.0,
                'examples': []
            }
        }

        # Step 1: calculate the base score and group by question
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
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            # here we use consistency instead of ground truth to give reward
            # temporarily give 0 score, adjust later based on consistency
            base_score = 0.0

            item = {
                'index': i,
                'base_score': base_score,
                'valid_response_length': valid_response_length,
                'sequences_str': sequences_str,
                'prompt_str': prompt_str,
                'response_str': response_str,
                'ground_truth': ground_truth,
                'data_source': data_source,
                'extra_info': extra_info
            }
            base_items.append(item)

            key = self._extract_question_key(prompt_str)
            groups[key].append(item)

        # Step 2: give reward based on self-consistency
        for key, group_items in groups.items():
            if len(group_items) != 4:
                # not a group of 4 answers, give base score
                for item in group_items:
                    basic_score = self.compute_score(
                        data_source=item['data_source'],
                        solution_str=item['sequences_str'],
                        ground_truth=item['ground_truth'],
                        extra_info=item['extra_info'],
                    )
                    item['base_score'] = basic_score
                    reward_tensor[item['index'], item['valid_response_length'] - 1] = float(basic_score)
                continue
            
            responses = [item['response_str'] for item in group_items]
            is_consistent, answers = self._check_consistency(responses)
            
            print(f"DEBUG: Group consistency check - answers: {answers}, consistent: {is_consistent}")
            
            if is_consistent:
                # consistent group, give high reward (1.0)
                for item in group_items:
                    item['base_score'] = 1.0
                    reward_tensor[item['index'], item['valid_response_length'] - 1] = 1.0
            else:
                # inconsistent group, give low reward (0.0), then do concept augmentation
                for item in group_items:
                    item['base_score'] = 0.0
                    reward_tensor[item['index'], item['valid_response_length'] - 1] = 0.0

        # Step 3: concept enhancement (only for inconsistent groups)
        can_aug = self._ensure_llm()
        wandb_stats['self_consistent_concept']['total_groups'] = len([g for g in groups.values() if len(g) == 4])
        
        if not can_aug:
            print("DEBUG: Self-consistent concept augmentation disabled")
        else:
            augmented_groups = 0
            for key, group_items in groups.items():
                if len(group_items) != 4:
                    continue
                
                responses = [item['response_str'] for item in group_items]
                is_consistent, answers = self._check_consistency(responses)
                
                if is_consistent:
                    continue  # consistent groups do not need enhancement
                
                wandb_stats['self_consistent_concept']['inconsistent_groups'] += 1
                
                print(f"DEBUG: Found inconsistent group with answers: {answers}")
                
                example_data = {
                    'group_size': len(group_items),
                    'original_answers': answers,
                    'original_responses': [item['response_str'] for item in group_items],
                    'augmented': False
                }
                
                prompt_str = group_items[0]['prompt_str']
                aug_prompt = self._find_enhanced_prompt(prompt_str)

                if not aug_prompt:
                    print(f"DEBUG: No enhanced prompt found for question")
                    wandb_stats['self_consistent_concept']['examples'].append(example_data)
                    continue
                
                print(f"DEBUG: Using enhanced prompt: {aug_prompt[:200]}...")
                
                example_data['enhanced_prompt'] = aug_prompt
                example_data['original_prompt'] = prompt_str
                
                try:
                    print(f"🔄 SELF-CONSISTENT: Generating enhanced responses (temp=0.7, top_p=0.9, max_tokens=1024)")
                    
                    from vllm import SamplingParams
                    sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1024, n=2, stop=["<|im_end|>", "</s>"])
                    
                    actor_outputs = self._generate_with_actor_rollout([aug_prompt], sampling)
                    
                    if actor_outputs is not None:
                        aug_responses = actor_outputs[:2]
                        print(f"✅ SELF-CONSISTENT: Used LATEST training actor for generation")
                    else:
                        print(f"🔄 SELF-CONSISTENT: Using vLLM for generation")
                        outputs = self._llm.generate([aug_prompt], sampling)
                        aug_responses = []
                        for out in outputs:
                            for comp in out.outputs:
                                aug_responses.append(comp.text.strip())
                        aug_responses = aug_responses[:2]
                    
                    if not aug_responses or len(aug_responses) < 2:
                        # Insufficient augmented responses generated
                        wandb_stats['self_consistent_concept']['examples'].append(example_data)
                        continue
                    
                    example_data['generated_responses'] = aug_responses
                    print(f"✅ SELF-CONSISTENT: Generated {len(aug_responses)} enhanced responses")
                    
                except Exception as e:
                    print(f"WARN: self-consistent concept augmentation generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    wandb_stats['self_consistent_concept']['examples'].append(example_data)
                    continue

                # analyze the newly generated answers
                aug_answers = [self._extract_answer(resp) for resp in aug_responses]
                aug_consistent, _ = self._check_consistency(aug_responses)
                
                print(f"DEBUG: Generated answers: {aug_answers}, consistent: {aug_consistent}")
                
                # smart replacement strategy (refined version)
                replace_indices = self._get_smart_replacement_indices(answers)
                print(f"DEBUG: Original answers: {answers}, replacement strategy result: {replace_indices}")
                
                # 🔥 key fix: really replace response tokens instead of just reward
                replaced_details = []
                for j, replace_idx in enumerate(replace_indices):
                    if j < len(aug_responses):
                        item = group_items[replace_idx]
                        old_reward = float(item['base_score'])
                        
                        # 🚀 convert the newly generated answers to tokens
                        new_response_text = aug_responses[j].strip()
                        new_response_tokens = self.tokenizer.encode(new_response_text, add_special_tokens=False)
                        
                        # get the original data related information
                        original_idx = item['index']
                        original_response_length = item['valid_response_length']
                        
                        # 🔄 really replace the response tokens in data.batch (fix tensor operations)
                        max_response_length = data.batch['responses'].shape[-1]
                        new_response_length = min(len(new_response_tokens), max_response_length)
                        
                        # get the device and dtype information of the original tensor
                        original_device = data.batch['responses'].device
                        original_dtype = data.batch['responses'].dtype
                        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                        
                        print(f"🔧 DEBUG: Replacing tokens at index {original_idx}, device: {original_device}, dtype: {original_dtype}")
                        
                        try:
                            # safely clear the original response position (use .fill_() instead of direct assignment)
                            with torch.no_grad():
                                data.batch['responses'][original_idx].fill_(pad_token_id)
                            
                            # write the new response tokens
                            if new_response_length > 0:
                                # ensure new_response_tokens is a list of integers
                                new_tokens_list = [int(token) for token in new_response_tokens[:new_response_length]]
                                new_response_tensor = torch.tensor(new_tokens_list, 
                                                                  dtype=original_dtype, 
                                                                  device=original_device)
                                # safely write the new tokens
                                with torch.no_grad():
                                    data.batch['responses'][original_idx][:new_response_length] = new_response_tensor
                            
                            print(f"✅ Successfully replaced {new_response_length} tokens")
                            
                        except Exception as token_error:
                            print(f"❌ Token replacement failed: {token_error}")
                            # if token replacement fails, at least do not crash, continue using the original tokens but update the reward
                            new_response_length = original_response_length
                        
                        # 🎯 update the attention_mask to match the new response length
                        try:
                            prompt_length = int(data.batch['prompts'].shape[-1])
                            total_length = int(data.batch['attention_mask'].shape[-1])
                            
                            with torch.no_grad():
                                # clear the attention mask of the response part
                                data.batch['attention_mask'][original_idx, prompt_length:] = 0
                                
                                # set the attention mask for the new response
                                if new_response_length > 0:
                                    response_end = min(prompt_length + new_response_length, total_length)
                                    data.batch['attention_mask'][original_idx, prompt_length:response_end] = 1
                            
                            print(f"✅ Successfully updated attention mask")
                            
                        except Exception as mask_error:
                            print(f"❌ Attention mask update failed: {mask_error}")
                            # if the attention mask update fails, keep the original
                            pass
                        
                        # 💰 update the reward tensor position to match the new response length 
                        try:
                            with torch.no_grad():
                                # clear the original reward
                                reward_tensor[original_idx, :] = 0.0
                        
                                # set the reward at the last valid position of the new response
                                if new_response_length > 0:
                                    reward_pos = min(new_response_length - 1, reward_tensor.shape[-1] - 1)
                                    new_reward = 0.4  # fixed reward for concept enhancement
                                    reward_tensor[original_idx, reward_pos] = new_reward
                                else:
                                    new_reward = 0.0  # if there are no valid tokens, reward is 0
                            
                            print(f"✅ Successfully updated reward tensor")
                            
                        except Exception as reward_error:
                            print(f"❌ Reward tensor update failed: {reward_error}")
                            # if the reward update fails, at least give a base score
                            new_reward = 0.4
                            
                        # update the item information
                        item['valid_response_length'] = new_response_length
                        
                        replaced_details.append({
                            'position': replace_idx,
                            'index': original_idx,
                            'old_reward': old_reward,
                            'new_reward': new_reward,
                            'boost': new_reward - old_reward,
                            'old_answer': answers[replace_idx],
                            'new_answer': aug_answers[j] if j < len(aug_answers) else 'UNKNOWN',
                            'old_response_length': original_response_length,
                            'new_response_length': new_response_length,
                            'new_response_text': new_response_text[:100] + '...' if len(new_response_text) > 100 else new_response_text
                        })
                        
                        wandb_stats['self_consistent_concept']['total_responses_enhanced'] += 1
                        wandb_stats['self_consistent_concept']['total_reward_boost'] += (new_reward - old_reward)
                        
                        print(f"🔥 FIXED: Replaced BOTH response tokens AND reward at index {original_idx}:")
                        print(f"   - Reward: {old_reward:.3f} -> {new_reward:.3f}")
                        print(f"   - Answer: {answers[replace_idx]} -> {aug_answers[j] if j < len(aug_answers) else 'UNKNOWN'}")
                        print(f"   - Response length: {original_response_length} -> {new_response_length} tokens")
                        print(f"   - New text: {new_response_text[:150]}...")
                
                # 🔄 key addition: recheck consistency after replacement
                
                # collect all answers after replacement (including the original answers that were not replaced)
                final_responses = []
                for idx, item in enumerate(group_items):
                    if idx in replace_indices:
                        # use the newly generated answers at the replaced positions
                        replace_pos = replace_indices.index(idx)
                        if replace_pos < len(aug_responses):
                            final_responses.append(aug_responses[replace_pos])
                        else:
                            final_responses.append(item['response_str'])  # fallback
                    else:
                        # use the original answers at the positions that were not replaced
                        final_responses.append(item['response_str'])
                
                # recheck the consistency after replacement
                final_is_consistent, final_answers = self._check_consistency(final_responses)
                print(f"🔄 RECHECK: After replacement - answers: {final_answers}, consistent: {final_is_consistent}")
                
                # 🤔 fix logic: replacing to make it consistent should not make everyone upgrade!
                if final_is_consistent:
                    print(f"✨ Replacement made the group CONSISTENT!")
                    print(f"💡 But we should NOT upgrade all positions - only concept-enhanced ones deserve 0.4")
                    print(f"🎯 Keeping differential rewards: original=0.0, concept-enhanced=0.4")
                    
                    # keep the original reward distribution:
                    # - original wrong answers: 0.0
                    # - concept enhanced answers: 0.4
                    # so that the GRPO advantage function can work normally (group with differences!)
                    
                    # update the statistics
                    for detail in replaced_details:
                        detail['final_reward'] = detail['new_reward']  # keep the original 0.4
                        detail['consistency_achieved'] = True
                        # no consistency_boost, because everyone did not upgrade
                    
                    print(f"📊 Final strategy: Keep 0.0 vs 0.4 differential to enable GRPO learning!")
                else:
                    print(f"🔄 Still inconsistent after replacement, differential rewards: 0.0 vs 0.4")
                
                example_data.update({
                    'augmented': True,
                    'aug_answers': aug_answers,
                    'aug_consistent': aug_consistent,
                    'final_is_consistent': final_is_consistent,  # new
                    'final_answers': final_answers,  # new
                    'consistency_achieved': final_is_consistent,  # new
                    'replacement_strategy': 'smart' if aug_consistent else 'random',
                    'replaced_positions': replace_indices,
                    'replacement_details': replaced_details
                })
                wandb_stats['self_consistent_concept']['examples'].append(example_data)
                
                augmented_groups += 1
                
            wandb_stats['self_consistent_concept']['successfully_augmented_groups'] = augmented_groups
            print(f"DEBUG: Self-consistent concept augmentation completed for {augmented_groups} groups")

        # Step 4: generate score_record
        for item in base_items:
            data_source = item['data_source']
            
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                final_reward = reward_tensor[item['index'], item['valid_response_length'] - 1].item()
                print(f"[SELF-CONSISTENT] {item['sequences_str'][:100]}...")
                if final_reward != item['base_score']:
                    print(f"[REWARD] Base: {item['base_score']:.3f} → Enhanced: {final_reward:.3f}")
                else:
                    print(f"[REWARD] {final_reward:.3f}")
            
            final_score = reward_tensor[item['index'], item['valid_response_length'] - 1].item()
            record = {
                "sequences_str": item['sequences_str'],
                "ground_truth": item['ground_truth'],
                "index": item['extra_info']["index"] if item['extra_info'] else None,
                "sample_idx": item['index'],
                "score": final_score,
                "original_score": item['base_score'],
                "is_self_consistent_enhanced": final_score != item['base_score']
            }
            score_record.append(record)

        # remove the wandb record code, avoid interfering with the original parameters

        return reward_tensor, score_record


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config, compute_score=None):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {
            'TOKENIZERS_PARALLELISM': 'true',
            'NCCL_DEBUG': 'WARN',
            'TRANSFORMERS_ALLOW_UNSAFE_LOAD': '1'
        }})

    ray.get(main_task.remote(config, compute_score))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config, compute_score=None):
    from verl.utils.fs import copy_local_path_from_hdfs
    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.utils.reward_score import deepscaler

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

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

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

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
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
    
    if reward_manager_name == 'naive':
        from verl.workers.reward_manager import NaiveRewardManager
        reward_manager_cls = NaiveRewardManager
        print("DEBUG: Using NaiveRewardManager")
    elif reward_manager_name == 'shuffled_naive':
        # directly use the ShuffledNaiveRewardManager defined in this file
        reward_manager_cls = ShuffledNaiveRewardManager
        print("DEBUG: Using ShuffledNaiveRewardManager (defined locally)")
    elif reward_manager_name == 'random_binary':
        # use random binary reward (50% 1, 50% 0)
        reward_manager_cls = RandomBinaryRewardManager
        print("DEBUG: Using RandomBinaryRewardManager (p=0.5)")
    elif reward_manager_name == 'prime':
        from verl.workers.reward_manager import PrimeRewardManager
        reward_manager_cls = PrimeRewardManager
        print("DEBUG: Using PrimeRewardManager")
    elif reward_manager_name == 'concept_aug':
        # concept enhanced custom RewardManager
        reward_manager_cls = ConceptAugmentedRewardManager
        print("DEBUG: Using ConceptAugmentedRewardManager")
    elif reward_manager_name == 'self_consistent_concept':
        # Self-Consistent concept enhanced custom RewardManager
        reward_manager_cls = SelfConsistentConceptRewardManager
        print("DEBUG: Using SelfConsistentConceptRewardManager")
    elif reward_manager_name == 'llm_judge':
        # LLM as Judge for concept usage evaluation
        from verl.workers.reward_manager import LLMJudgeRewardManager
        reward_manager_cls = LLMJudgeRewardManager
        print("DEBUG: Using LLMJudgeRewardManager")
    elif reward_manager_name == 'gpt4o_baseline':
        # GPT-4o Baseline for all-wrong group intervention
        from verl.workers.reward_manager.gpt4o_baseline import GPT4oBaselineRewardManager
        reward_manager_cls = GPT4oBaselineRewardManager
        print("DEBUG: Using GPT4oBaselineRewardManager")
    else:
        raise NotImplementedError(f"Unknown reward_manager: {reward_manager_name}")

    # if config.actor_rollout_ref.model.path.strip().startswith("Qwen") or config.actor_rollout_ref.model.path.strip().startswith("meta-llama"):
    if config.actor_rollout_ref.model.path.strip().startswith("Qwen") or 'llama' in config.actor_rollout_ref.model.path.lower() or config.actor_rollout_ref.model.use_think == False:
        print("\nQwen or LLAMA---------------------------------\n")
        compute_score = deepscaler.compute_score
        
    if reward_manager_name == 'concept_aug':
        # Get concept_file from config, use default MCQ file if not specified
        concept_file = config.reward_model.get("concept_file", "data/quizzes/concept_quizzes.jsonl")
        print(f"🔄 CONCEPT: Using concept file: {concept_file}")
        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score, model_path=local_path, trainer=None, concept_file=concept_file)  # trainer is set below
        # Note: validation uses same manager for consistency
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score, model_path=local_path, trainer=None, concept_file=concept_file)
    elif reward_manager_name == 'self_consistent_concept':
        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score, model_path=local_path, trainer=None)  # trainer is set below
        # Note: validation uses same manager for consistency
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score, model_path=local_path, trainer=None)
    elif reward_manager_name == 'llm_judge':
        # LLM Judge uses vLLM API
        concept_bonus = config.reward_model.get("concept_bonus", 0.4)
        judge_model_path = config.reward_model.get("judge_model_path", config.actor_rollout_ref.model.path)
        judge_api_url = config.reward_model.get("judge_api_url", "http://localhost:8001/v1/completions")
        print(f"🔍 LLM Judge: Using judge model via API: {judge_model_path}")
        print(f"🔍 LLM Judge: API URL: {judge_api_url}, concept_bonus: {concept_bonus}")
        reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=compute_score,
            judge_model_path=judge_model_path,
            concept_bonus=concept_bonus,
            judge_api_url=judge_api_url
        )
        val_reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=1,
            compute_score=compute_score,
            judge_model_path=judge_model_path,
            concept_bonus=concept_bonus,
            judge_api_url=judge_api_url
        )
    elif reward_manager_name == 'gpt4o_baseline':
        # GPT-4o Baseline uses OpenAI API
        bonus_reward = config.reward_model.get("bonus_reward", 0.4)
        openai_api_key = config.reward_model.get("openai_api_key", None)  # Can also use env var OPENAI_API_KEY
        model_name = config.reward_model.get("model_name", "gpt-4o")
        print(f"🤖 GPT-4o Baseline: Using model: {model_name}")
        print(f"💰 GPT-4o Baseline: Bonus reward: {bonus_reward}")
        reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=compute_score,
            openai_api_key=openai_api_key,
            model_name=model_name,
            bonus_reward=bonus_reward
        )
        val_reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=1,
            compute_score=compute_score,
            openai_api_key=openai_api_key,
            model_name=model_name,
            bonus_reward=bonus_reward
        )
    else:
        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    
    # set the trainer reference to the concept augmented reward managers
    if reward_manager_name == 'concept_aug':
        reward_fn.trainer = trainer
        val_reward_fn.trainer = trainer
        print("🔄 CONCEPT: Set trainer reference for dynamic checkpoint loading")
    elif reward_manager_name == 'self_consistent_concept':
        reward_fn.trainer = trainer
        val_reward_fn.trainer = trainer
        print("🔄 SELF-CONSISTENT: Set trainer reference for dynamic checkpoint loading")
    
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
