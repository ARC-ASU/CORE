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
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F
from verl.utils.kl_regularizer import ConceptKLRegularizer

try:
    from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("Warning: flash_attn not available, using fallback implementation")

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False) and HAS_FLASH_ATTN
        if self.config.get('use_remove_padding', False) and not HAS_FLASH_ATTN:
            print('Warning: use_remove_padding requested but flash_attn not available, disabling padding removal')
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        # Avoid using torch.compile here to prevent potential graph cache growth under dynamic shapes
        # that can gradually increase GPU memory usage across steps. Use the eager function directly
        # to keep numerics identical without impacting training effectiveness.
        # self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad = logits_rmpad / temperature  # out-of-place operation

                # compute entropy
                # Compute entropy without torch.compile to avoid dynamic graph cache accumulation
                entropy_rmpad = verl_F.entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits = logits / temperature  # out-of-place operation
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            # Return clones to break the view relationship and prevent inplace modification errors during backward pass.
            return entropy.clone(), log_probs.clone()

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        # KL distillation prepared by trainer via meta_info
        kl_items = data.meta_info.get('kl_items', None)
        kl_cfg = data.meta_info.get('kl_cfg', {})
        kl_loss_total = None
        kl_metrics = None
        kl_items_cache = None
        if kl_items is not None and hasattr(self, 'tokenizer'):
            try:
                # keep a shallow copy so we can recompute with autograd per mini-batch
                kl_items_cache = list(kl_items)
                kl_cfg_cache = dict(kl_cfg)

                preview_regularizer = ConceptKLRegularizer(
                    tokenizer=self.tokenizer,
                    base_lambda=float(kl_cfg_cache.get('base_lambda', 0.01)),
                    effective_multiplier=float(kl_cfg_cache.get('effective_multiplier', 3.0)),
                    ineffective_multiplier=float(kl_cfg_cache.get('ineffective_multiplier', 0.5)),
                )
                try:
                    preview_regularizer._debug_token_kl = True
                    print("[Actor] KL debug enabled (_debug_token_kl=True)")
                except Exception:
                    pass
                device = next(self.actor_module.parameters()).device
                preview_results = preview_regularizer.compute_token_level_kl_loss(
                    model=self.actor_module,
                    kl_data_items=kl_items_cache,
                    device=device,
                    require_grad=False,
                )
                kl_loss_total = preview_results.get('kl_loss', None)
                if kl_loss_total is not None:
                    kl_metrics = {
                        'actor/kl_distill_loss': float(kl_loss_total.item()),
                        'actor/kl_distill_items': int(preview_results.get('kl_items_count', 0)),
                        'actor/kl_distill_tokens': int(preview_results.get('total_token_count', 0)),
                        'actor/kl_distill_effective': int(preview_results.get('concept_effective_count', 0)),
                    }
            except Exception as e:
                print(f"[Actor] KL inside update_policy failed: {e}")
                kl_loss_total = None
                kl_items_cache = None

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                # Only affects memory-safe chunking strategy, does not change training semantics
                max_token_len_cfg = self.config.ppo_max_token_len_per_gpu
                try:
                    override = int(os.getenv('ACTOR_MAX_TOKENS_PER_GPU', '0'))
                    if override > 0:
                        max_token_len_cfg = min(max_token_len_cfg, override)
                except Exception:
                    pass
                max_token_len = max_token_len_cfg * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

            # Free gradient storage aggressively to reduce peak memory
            self.actor_optimizer.zero_grad(set_to_none=True)

            # --- KL LOGIC: recompute KL loss per mini-batch with gradients ---
            if kl_items_cache is not None:
                try:
                    grad_regularizer = ConceptKLRegularizer(
                        tokenizer=self.tokenizer,
                        base_lambda=float(kl_cfg_cache.get('base_lambda', 0.01)),
                        effective_multiplier=float(kl_cfg_cache.get('effective_multiplier', 3.0)),
                        ineffective_multiplier=float(kl_cfg_cache.get('ineffective_multiplier', 0.5)),
                    )
                    device = next(self.actor_module.parameters()).device
                    grad_results = grad_regularizer.compute_token_level_kl_loss(
                        model=self.actor_module,
                        kl_data_items=kl_items_cache,
                        device=device,
                        require_grad=True,
                    )
                    kl_grad = grad_results.get('kl_loss', None)
                    if kl_grad is not None:
                        num_minibatches = len(dataloader)
                        if num_minibatches > 0:
                            (kl_grad / num_minibatches).backward(retain_graph=True)
                except Exception as kl_e:
                    print(f"[Actor] KL grad computation failed: {kl_e}")

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1)
                attention_mask = data['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                old_log_prob = data['old_log_probs']
                advantages = data['advantages']

                clip_ratio = self.config.clip_ratio
                entropy_coeff = self.config.entropy_coeff
                # pg_loss_coeff is 1.0 if config has no pg_loss_coeff
                pg_loss_coeff = self.config.get('pg_loss_coeff', 1.0)
                print(f'### pg_loss_coeff: {pg_loss_coeff}')

                # all return: (bsz, response_length)
                entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                             log_prob=log_prob,
                                                                             advantages=advantages,
                                                                             eos_mask=response_mask,
                                                                             cliprange=clip_ratio)
                # compute entropy loss from entropy
                entropy_loss = verl_F.masked_mean(entropy, response_mask)

                # compute policy loss
                policy_loss = pg_loss * pg_loss_coeff - entropy_loss * entropy_coeff

                if self.config.use_kl_loss:
                    ref_log_prob = data['ref_log_prob']
                    # compute kl loss
                    kld = core_algos.kl_penalty(logprob=log_prob,
                                               ref_logprob=ref_log_prob,
                                               kl_penalty=self.config.kl_loss_type)
                    kl_loss = masked_mean(kld, response_mask)

                    policy_loss = torch.add(policy_loss, kl_loss * self.config.kl_loss_coef)
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef
                    metrics['actor/pg_loss_coeff'] = pg_loss_coeff

                # Extract metrics BEFORE computing final loss to avoid referencing deleted tensors
                metrics_data = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                }

                if self.config.use_dynamic_bsz:
                    # relative to the dynamic bsz
                    loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                else:
                    loss = policy_loss / self.gradient_accumulation
                
                # --- REMOVED OLD KL LOSS LOGIC ---

                # Add loss to metrics
                loss_value = loss.detach().item()
                metrics_data['actor/loss'] = loss_value
                if kl_metrics is not None:
                    metrics_data.update(kl_metrics)

                # 🔧 Memory safety: Clear intermediate tensors before backward
                # This aggressive cleanup reduces memory pressure during gradient computation
                del entropy, log_prob
                if self.config.use_kl_loss:
                    del kld, ref_log_prob
                del policy_loss, entropy_loss, ppo_kl, pg_loss, pg_clipfrac
                if self.config.use_kl_loss:
                    del kl_loss

                # Force garbage collection before backward to free maximum memory
                torch.cuda.empty_cache()
                import gc
                gc.collect()

                # No retain_graph is needed anymore as the KL graph is handled separately.
                loss.backward()

                # Use pre-extracted metrics
                data = metrics_data

                # Clear final loss tensor
                del loss
                torch.cuda.empty_cache()
                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
            del grad_norm  # Clear gradient norm tensor
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad(set_to_none=True)
        return metrics
