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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn

# [Concept Debug] Import wandb and answer extraction utility
import wandb
import re
from verl.utils.reward_score.utils.utils import extract_answer
# [Concept Debug End]

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        # [Concept Debug] Store wandb run path passed from the worker
        self.wandb_run_path = kwargs.get('wandb_run_path', None)
        # [Concept Debug End]
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        # Print memory configuration for debugging
        print(f"🔧 [vLLM Config] max_num_batched_tokens: {max_num_batched_tokens}")
        print(f"🔧 [vLLM Config] gpu_memory_utilization: {config.gpu_memory_utilization}")

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)
                                                  
        # Calculate target sequence length
        target_seq_len = config.prompt_length + config.response_length

        # Prepare LLM kwargs
        llm_kwargs = {
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": config.dtype,
            "enforce_eager": config.enforce_eager,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "skip_tokenizer_init": False,
            "max_model_len": target_seq_len,
            "load_format": config.load_format,
            "disable_log_stats": config.disable_log_stats,
            "max_num_batched_tokens": max_num_batched_tokens,
            "enable_chunked_prefill": config.enable_chunked_prefill,
        }

        # Add swap_space if configured (for CPU offloading when GPU memory is full)
        if hasattr(config, 'swap_space') and config.swap_space is not None:
            llm_kwargs["swap_space"] = config.swap_space
            print(f"🔧 [Memory] CPU swap space: {config.swap_space}GB")

        # Check if we need RoPE scaling to extend context length
        if model_hf_config.max_position_embeddings < target_seq_len:
            # Calculate scaling factor needed
            scaling_factor = target_seq_len / model_hf_config.max_position_embeddings
            # Use linear RoPE scaling (recommended for moderate extensions)
            rope_scaling_config = {"type": "linear", "factor": scaling_factor}
            llm_kwargs["rope_scaling"] = rope_scaling_config
            print(f"🔧 [RoPE Scaling] Extending context: {model_hf_config.max_position_embeddings} -> {target_seq_len} tokens (factor: {scaling_factor:.2f})")
        else:
            # No scaling needed, original behavior
            assert model_hf_config.max_position_embeddings >= target_seq_len, \
                "model context length should be greater than total sequence length"

        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            **llm_kwargs
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # we may detokenize the result all together later
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

        # [Concept Debug] Initialize wandb table and step counter for logging
        self._debug_table = None
        self._debug_step_counter = 0
        # [Concept Debug End]

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validation = prompts.meta_info.get('validate', False)
        
        # DEBUG: add debug info
        print(f"🔍 ROLLOUT DEBUG: do_sample={do_sample}, validate={is_validation}, config.n={self.config.n}")
        print(f"🔍 BATCH DEBUG: batch_size={batch_size}")
        
        if not do_sample:
            # FIX: Only use greedy mode n=1 during validation
            # Training should follow config parameters (temperature=0.7, n=4)
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # greedy mode requires n=1, mainly used for validation
            }
        else:
            # FIX: Use sampling mode during training, follow config parameters
            # 🔧 CONCEPT: For concept augmentation, use n=1 per worker (total n=2 across 2 workers)
            is_concept_aug = prompts.meta_info.get('concept_augmentation', False)
            n_samples = 1 if is_concept_aug else self.config.n
            
            kwargs = {
                'n': n_samples,
                'best_of': n_samples,
                'temperature': getattr(self.config, 'temperature', 0.7),
                'top_p': getattr(self.config, 'top_p', 0.9)
            }
            
            if is_concept_aug:
                print(f"🔧 CONCEPT: Using n={n_samples} per worker for concept augmentation (instead of config.n={self.config.n})")

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            output = self.inference_engine.generate(
                prompts=None,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False)

        # [Concept Debug] FINAL FIX: Handle dynamic return type from vLLM.
        # It returns a list of RequestOutput during validation and a tuple of tensors during training rollout.
        
        # [Concept Debug] Ensure wandb is initialized in the worker process
        if self.wandb_run_path and wandb.run is None:
            wandb.init(resume=self.wandb_run_path)
        # [Concept Debug End]

        # Initialize tensors to be populated based on output type
        response = None
        log_probs = None

        if isinstance(output, list):
            # Case 1: Output is a list of RequestOutput (typically during validation)
            if wandb.run:
                try:
                    if self._debug_table is None:
                        self._debug_table = wandb.Table(columns=["step", "prompt", "full_response", "extracted_answer", "correct_answer"])
                    
                    prompt_texts = self.inference_engine.tokenizer.batch_decode(prompts.batch['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    # FIX: correctly get ground_truth from reward_model field
                    ground_truths = []
                    for i in range(len(prompt_texts)):
                        try:
                            # Get ground_truth from non_tensor_batch's reward_model field
                            gt = prompts.non_tensor_batch['reward_model']['ground_truth'][i] if prompts.non_tensor_batch and 'reward_model' in prompts.non_tensor_batch else None
                            # If it's a list, take the first element
                            if isinstance(gt, list) and len(gt) > 0:
                                gt = gt[0]
                            ground_truths.append(gt)
                        except (KeyError, IndexError, TypeError):
                            ground_truths.append(None)

                    for i, request_output in enumerate(output):
                        prompt_text = prompt_texts[i]
                        correct_answer = ground_truths[i]
                        correct_answer_str = str(correct_answer) if correct_answer is not None else "N/A"
                        
                        for completion_output in request_output.outputs:
                            full_response = completion_output.text
                            extracted_answer = extract_answer(full_response, is_mcq=True)
                            # DEBUG: print detailed answer extraction info
                            if "Therefore, the answer is" in full_response:
                                expected_pattern = re.search(r"Therefore,?\s+the\s+answer\s+is\s+([A-D])", full_response, re.IGNORECASE)
                                expected = expected_pattern.group(1) if expected_pattern else "UNKNOWN"
                                print(f"🔍 ANSWER EXTRACTION DEBUG (list mode):")
                                print(f"   Expected from 'Therefore, the answer is X': {expected}")
                                print(f"   Extracted by extract_answer(): {extracted_answer}")
                                if expected != extracted_answer:
                                    print(f"   ⚠️  MISMATCH! Full response: {repr(full_response[:1000])}")
                                    print(f"   ⚠️  Debug extraction: {extract_answer(full_response, is_mcq=True, debug=True)}")
                                else:
                                    print(f"   ✅ Match!")
                            self._debug_table.add_data(self._debug_step_counter, prompt_text, full_response, extracted_answer, correct_answer_str)
                    
                    wandb.log({"rollout_debug_samples": self._debug_table})
                    self._debug_table = None
                    self._debug_step_counter += 1
                except Exception as e:
                    print(f"❌ ERROR in wandb rollout logging (list mode): {e}")

            # Extract response and log_probs for downstream processing
            response_token_ids_list = [comp.token_ids for req_out in output for comp in req_out.outputs]
            log_probs_list = [
                [comp.logprobs[i][token_id] for i, token_id in enumerate(comp.token_ids)]
                for req_out in output for comp in req_out.outputs
            ]
            response_tensors = [torch.tensor(ids, dtype=torch.long) for ids in response_token_ids_list]
            response = torch.nn.utils.rnn.pad_sequence(response_tensors, batch_first=True, padding_value=self.pad_token_id).to(idx.device)
            log_probs_tensors = [torch.tensor(lps, dtype=torch.float) for lps in log_probs_list]
            log_probs = torch.nn.utils.rnn.pad_sequence(log_probs_tensors, batch_first=True, padding_value=0.0).to(idx.device)

        elif isinstance(output, tuple):
            # Case 2: Output is a tuple of (response_token_ids, log_probs) (typically during training)
            response_token_ids, log_probs_tensor = output
            
            if wandb.run:
                try:
                    if self._debug_table is None:
                        self._debug_table = wandb.Table(columns=["step", "prompt", "full_response", "extracted_answer", "correct_answer"])

                    prompt_texts = self.inference_engine.tokenizer.batch_decode(prompts.batch['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    full_responses = self.inference_engine.tokenizer.batch_decode(response_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    # FIX: correctly get ground_truth from reward_model field
                    ground_truths = []
                    for i in range(len(prompt_texts)):
                        try:
                            # Get ground_truth from non_tensor_batch's reward_model field
                            gt = prompts.non_tensor_batch['reward_model']['ground_truth'][i] if prompts.non_tensor_batch and 'reward_model' in prompts.non_tensor_batch else None
                            # If it's a list, take the first element
                            if isinstance(gt, list) and len(gt) > 0:
                                gt = gt[0]
                            ground_truths.append(gt)
                        except (KeyError, IndexError, TypeError):
                            ground_truths.append(None)

                    for i in range(len(prompt_texts)):
                        correct_answer = ground_truths[i]
                        correct_answer_str = str(correct_answer) if correct_answer is not None else "N/A"
                        full_response = full_responses[i]
                        extracted_answer = extract_answer(full_response, is_mcq=True)
                        # DEBUG: print detailed answer extraction info
                        if "Therefore, the answer is" in full_response:
                            expected_pattern = re.search(r"Therefore,?\s+the\s+answer\s+is\s+([A-D])", full_response, re.IGNORECASE)
                            expected = expected_pattern.group(1) if expected_pattern else "UNKNOWN"
                            print(f"🔍 ANSWER EXTRACTION DEBUG (tuple mode):")
                            print(f"   Expected from 'Therefore, the answer is X': {expected}")
                            print(f"   Extracted by extract_answer(): {extracted_answer}")
                            if expected != extracted_answer:
                                print(f"   ⚠️  MISMATCH! Full response: {repr(full_response[:1000])}")
                                print(f"   ⚠️  Debug extraction: {extract_answer(full_response, is_mcq=True, debug=True)}")
                            else:
                                print(f"   ✅ Match!")
                        self._debug_table.add_data(
                            self._debug_step_counter,
                            prompt_texts[i],
                            full_response,
                            extracted_answer,
                            correct_answer_str
                        )
                    
                    wandb.log({"rollout_debug_samples": self._debug_table})
                    self._debug_table = None
                    self._debug_step_counter += 1
                except Exception as e:
                    print(f"❌ ERROR in wandb rollout logging (tuple mode): {e}")

            response = response_token_ids.to(idx.device)
            log_probs = log_probs_tensor.to(idx.device)
        
        else:
            raise TypeError(f"Unexpected output type from vLLM generate: {type(output)}")
        # [Concept Debug FIX End]

        # check if this is concept augmentation (should not expand)
        is_concept_aug = prompts.meta_info.get('concept_augmentation', False)
        expected_n = 1 if is_concept_aug else self.config.n
        
        # DEBUG: check response tensor shape
        print(f"🔍 RESPONSE DEBUG: response.shape={response.shape}, expected n={expected_n}")
        print(f"🔍 CONDITION DEBUG: config.n > 1: {self.config.n > 1}, do_sample: {do_sample}, concept_aug: {is_concept_aug}")
        
        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            log_probs = pad_sequence_to_length(log_probs, self.config.response_length, 0.0)
        
        if self.config.n > 1 and do_sample and not is_concept_aug:
            print(f"🔍 EXPANSION DEBUG: Expanding batch from {batch_size} to {batch_size * self.config.n}")
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        elif is_concept_aug and do_sample:
            # 🔧 CONCEPT: For concept augmentation, no expansion needed since n=1 per worker
            print(f"🔍 CONCEPT NO EXPANSION: concept augmentation mode - keeping batch size {batch_size} (n=1 per worker)")
        else:
            if is_concept_aug:
                print(f"🔍 NO EXPANSION: concept augmentation but do_sample={do_sample}")
            else:
                print(f"🔍 NO EXPANSION: condition failed - config.n={self.config.n}, do_sample={do_sample}")
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)
