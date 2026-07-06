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

# rm -rf verl/trainer/ppo/ray_trainer.py; vim verl/trainer/ppo/ray_trainer.py
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

import tempfile
from filelock import FileLock
import json
from collections import Counter
import wandb
import re
import matplotlib.pyplot as plt
import random
from numbers import Integral


WorkerType = Type[Worker]



class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'reinforce_plus_plus':
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'reinforce_group':
        # REINFORCE with group-wise baseline for concept enhancement
        from verl.trainer.ppo import core_algos_reinforce_group
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        # Check if we have concept enhancement flags
        concept_flags = {}
        if 'concept_enhanced' in data.non_tensor_batch:
            concept_flags = data.non_tensor_batch['concept_enhanced']

        advantages, returns = core_algos_reinforce_group.compute_reinforce_group_baseline_advantage(
            token_level_rewards=token_level_rewards,
            eos_mask=response_mask,
            index=index,
            gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'group_gae':
        # Group-Aware GAE: use Critic to reduce variance + group-level normalize to preserve signal
        from verl.trainer.ppo import core_algos_group_gae
        token_level_rewards = data.batch['token_level_rewards']
        values = data.batch['values']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        advantages, returns = core_algos_group_gae.compute_group_aware_gae_advantage_v2(
            token_level_rewards=token_level_rewards,
            values=values,
            eos_mask=response_mask,
            index=index,
            gamma=gamma,
            lam=lam,
            normalize_style='partial')  # options: 'none', 'partial', 'full'
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'remax':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == 'gae':
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'group_gae':
            self.use_critic = True  # Group-Aware GAE requires Critic
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'reinforce_plus_plus':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'reinforce_group':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'remax':
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

        self.history_accuracy = {} 
        self.history_accuracy_with_step = {}
        self.validation_accuracy_by_source = {}

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')

        print("Train_dataset size", len(self.train_dataset))

        # Create regular dataloader
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)
        
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                                batch_size=self.config.data.train_batch_size,
                                                drop_last=True,
                                                collate_fn=collate_fn,
                                                sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                    tokenizer=self.tokenizer,
                                    prompt_key=self.config.data.prompt_key,
                                    max_prompt_length=self.config.data.max_prompt_length,
                                    filter_prompts=True,
                                    return_raw_chat=self.config.data.get('return_raw_chat', False),
                                    truncation='error')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                        batch_size=len(self.val_dataset),
                                        shuffle=True,
                                        drop_last=True,
                                        collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _log_scores_to_wandb(self, score_records, epoch):

        new_metrics = {}

        if 'wandb' not in self.config.trainer.logger:
            return new_metrics

        aggregate_score = {}
        index_c = {}

        for record in score_records:
            index = record.get("index")
            if index not in index_c:
                index_c[index] = 1
            else:
                index_c[index] = index_c[index]+1
            score = record.get("score", 0)
            sequences_str = record.get("sequences_str", "")

            # Extract input (text between user and assistant)
            input_match = re.search(r'<\|im_start\|>user\n([\s\S]*?)<\|im_end\|>', sequences_str)
            input_text = input_match.group(1).strip() if input_match else ""

            if index in aggregate_score:
                aggregate_score[index]["total_score"] += score
            else:
                aggregate_score[index] = {
                    "total_score": score,
                    "index": index,
                    "input": input_text
                }

        for index, data in aggregate_score.items():
            score = int(data["total_score"]) // (index_c[index] / self.config.actor_rollout_ref.rollout.n)
            accuracy = score / self.config.actor_rollout_ref.rollout.n

            if index not in self.history_accuracy:
                self.history_accuracy[index] = []
            self.history_accuracy[index].append(accuracy)

            # Update new history_accuracy_with_step dictionary
            if index not in self.history_accuracy_with_step:
                self.history_accuracy_with_step[index] = {}
            self.history_accuracy_with_step[index][self.global_steps] = accuracy

        accuracy_columns = ["index", "accuracy", "history"]
        self.accuracy_table = wandb.Table(columns=accuracy_columns)
        for index in self.history_accuracy:
            self.accuracy_table.add_data(
                index,
                self.history_accuracy[index][-1], 
                self.history_accuracy.get(index, []),
            )

        # Get all data from the table
        all_data = self.accuracy_table.data
        sample_size = min(2, len(all_data))
        
        if sample_size > 0 and sample_size < len(all_data):
            # Create a new table with the same columns
            sampled_accuracy_table = wandb.Table(columns=self.accuracy_table.columns)
            
            # Randomly sample rows and add them to the new table
            sampled_indices = random.sample(range(len(all_data)), sample_size)
            for idx in sampled_indices:
                sampled_accuracy_table.add_data(*all_data[idx])
        else:
            sampled_accuracy_table = self.accuracy_table

        # Calculate sampling frequency statistics
        sampling_counts = [len(self.history_accuracy[idx]) for idx in self.history_accuracy]
        # here the average is for the whole dataset, which may contain the data that haven't been sampled
        avg_sampling_count = sum(sampling_counts) / len(self.train_dataset)
        max_sampling_count = max(sampling_counts) if sampling_counts else 0
        min_sampling_count = min(sampling_counts) if sampling_counts else 0
        
        new_metrics.update({
            # "score_detail_table": self.score_detail_table,
            "sampled_accuracy_table": sampled_accuracy_table,
            "sampling_frequency/avg": avg_sampling_count,
            "sampling_frequency/max": max_sampling_count,
            "sampling_frequency/min": min_sampling_count
        })

        return new_metrics
    
    def _maybe_log_val_generations_to_wandb(self, inputs, outputs, scores):
        """Log a table of validation samples to wandb"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and 'wandb' not in self.config.trainer.logger:
            print(
                'WARNING: `val_generations_to_log_to_wandb` is set to a positive value, but no wandb logger is found. ')
            return

        import wandb
        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Create column names for all samples
        columns = ["step"] + sum([[f"input_{i+1}", f"output_{i+1}", f"score_{i+1}"] for i in range(len(samples))], [])

        if not hasattr(self, 'validation_table'):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(self.global_steps)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log (avoid overwriting other metrics from the same step)
        # Commented out this standalone wandb.log call to avoid overwriting training metrics
        # if self.global_steps > 0:
        #     wandb.log({"generations": new_table}, step=self.global_steps)
        self.validation_table = new_table

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            
            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            n_val_samples = self.config.actor_rollout_ref.rollout.n_val
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            
            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, score_record= self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        self._maybe_log_val_generations_to_wandb(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            mean_reward = np.mean(rewards)
            metric_dict[f'val/test_score/{data_source}'] = mean_reward
            
            # Store validation accuracy by data source and step
            if data_source not in self.validation_accuracy_by_source:
                self.validation_accuracy_by_source[data_source] = {}
            self.validation_accuracy_by_source[data_source][self.global_steps] = mean_reward

        return metric_dict

    def _log_rollout_samples_to_wandb(self, batch, replaced_indices, all_wrong_uids, score_records=None, max_groups=10, extra_all_wrong_groups=3):
        if 'wandb' not in self.config.trainer.logger:
            return
        
        # record every step for the first 5 steps, then record every 5 steps
        if self.global_steps > 5 and self.global_steps % 5 != 0:
            return
        try:
            import wandb
            import os
            import json
            import torch
            responses = batch.batch['responses']
            response_length = responses.size(-1)
            attn = batch.batch['attention_mask']
            uids = batch.non_tensor_batch['uid']
            # calculate prompt length
            total_length = attn.size(-1)
            prompt_length = total_length - response_length
            group_map = {}
            for i in range(len(uids)):
                uid = uids[i]
                group_map.setdefault(uid, []).append(i)

            def decode_prompt(idx):
                try:
                    # try to get prompt from prompts field
                    if 'prompts' in batch.batch:
                        prompt_attn = attn[idx, :prompt_length]
                        valid_prompt_len = int(prompt_attn.sum().item())
                        if valid_prompt_len > 0:
                            prompt_ids = batch.batch['prompts'][idx, -valid_prompt_len:]
                            return self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
                    
                    # alternative: get prompt from input_ids
                    if 'input_ids' in batch.batch:
                        prompt_attn = attn[idx, :prompt_length]
                        valid_positions = torch.where(prompt_attn == 1)[0]
                        if len(valid_positions) > 0:
                            start_pos = valid_positions[0].item()
                            end_pos = valid_positions[-1].item() + 1
                            input_ids = batch.batch['input_ids'][idx, start_pos:end_pos]
                            return self.tokenizer.decode(input_ids, skip_special_tokens=True)
                    
                    return "Prompt unavailable"
                except Exception as e:
                    return f"Prompt decode error: {str(e)[:50]}"

            def decode_response(idx):
                try:
                    valid_resp_len = int(attn[idx, -response_length:].sum().item())
                    if valid_resp_len > 0:
                        resp_ids = batch.batch['responses'][idx, :valid_resp_len]
                        return self.tokenizer.decode(resp_ids, skip_special_tokens=False)
                    return "Empty response"
                except Exception as e:
                    return f"Response decode error: {str(e)[:50]}"

            def sample_groups(order):
                rows = []
                count_groups = 0
                seen = set()
                for uid in order:
                    if uid in seen:
                        continue
                    seen.add(uid)
                    idx_list = group_map.get(uid, [])
                    if not idx_list:
                        continue
                    for local_idx, i_idx in enumerate(idx_list):
                        prompt_text = decode_prompt(i_idx)
                        resp_text = decode_response(i_idx)
                        is_replaced = bool(i_idx in replaced_indices) if replaced_indices is not None else False
                        is_all_wrong = bool(uid in all_wrong_uids) if all_wrong_uids is not None else False
                        
                        # get real information from score_records
                        extracted_answer = "N/A"
                        ground_truth = "N/A"
                        score = 0.0
                        is_correct = False
                        
                        if score_records:
                            # find the corresponding score_record by index
                            for record in score_records:
                                if isinstance(record, dict) and record.get('index') == i_idx:
                                    extracted_answer = record.get('extracted_answer', 'N/A')
                                    ground_truth = str(record.get('ground_truth', 'N/A'))
                                    score = float(record.get('score', 0.0))
                                    is_correct = score > 0.0
                                    break
                        
                        
                        rows.append([
                            int(self.global_steps), str(uid), int(local_idx), is_all_wrong, is_replaced,
                            str(prompt_text),       # ensure it is a string
                            str(resp_text),         # ensure it is a string  
                            str(extracted_answer),  # extracted answer
                            str(ground_truth),      # ground truth
                            float(score),           # score
                            bool(is_correct)        # is correct
                        ])
                    count_groups += 1
                    if count_groups >= max_groups:
                        break
                return rows

            uid_list = [u for u in uids]
            rows = sample_groups(uid_list)
            if extra_all_wrong_groups > 0 and not any(r[3] for r in rows):
                extra_uids = [u for u in group_map.keys() if u in all_wrong_uids and u not in set([r[1] for r in rows])]
                extra_rows = sample_groups(extra_uids[:extra_all_wrong_groups])
                rows.extend(extra_rows)

            if rows:
                columns = ["step", "uid", "idx", "all_wrong_group", "replaced", "prompt", "response", "extracted_answer", "ground_truth", "score", "is_correct"]
                table = wandb.Table(columns=columns, data=rows)
                
                # 2) scalar statistics, like other actor/critic metrics
                num_rows = len(rows)
                num_replaced = sum(1 for r in rows if r[4])
                num_all_wrong = sum(1 for r in rows if r[3])
                metrics_payload = {
                    "rollout_samples": table,  # 1) table format for batch browsing/export
                    "rollout/num_logged_samples": num_rows,
                    "rollout/num_replaced": num_replaced,
                    "rollout/num_all_wrong_flags": num_all_wrong,
                }

                # 3) text preview: use different names to avoid wandb UI conflict
                preview_k = min(5, num_rows)
                sample_texts = []
                for i in range(preview_k):
                    r = rows[i]
                    sample_info = f"""Sample {i+1}:
Prompt: {str(r[5])}
Response: {str(r[6])}
Answer: {str(r[7])} | Ground Truth: {str(r[8])}
Score: {float(r[9]):.3f} | Correct: {bool(r[10])}
All Wrong: {bool(r[3])} | Replaced: {bool(r[4])}
---"""
                    sample_texts.append(sample_info)
                
                if sample_texts:
                    metrics_payload["rollout_preview"] = "\n".join(sample_texts)

                # log to wandb (allow step 0)
                wandb.log(metrics_payload, step=self.global_steps)
                print(f"[wandb] Logged {len(rows)} rollout samples at step {self.global_steps}")

                # 4) local file to disk (JSONL) for offline debugging
                try:
                    log_to_file = os.environ.get("VERL_LOG_ROLLOUT_TO_FILE", "1") == "1"
                    if log_to_file:
                        file_path = os.environ.get("VERL_ROLLOUT_FILE", "rollout_samples.jsonl")
                        with open(file_path, 'a', encoding='utf-8') as f:
                            for r in rows:
                                rec = {
                                    'step': int(self.global_steps),
                                    'uid': r[1],
                                    'idx': int(r[2]),
                                    'all_wrong_group': bool(r[3]),
                                    'replaced': bool(r[4]),
                                    'prompt': r[5],
                                    'response': r[6],
                                }
                                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"[rollout_file] WARN: write JSONL failed: {e}")
        except Exception as e:
            print(f"[wandb] rollout_samples log failed: {e}")


    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_history_accuracy(self, accuracy):
        # score_dir = os.path.join(os.getcwd(), 'accuracy')
        score_dir = os.path.join(self.config.trainer.default_local_dir, 'accuracy')
        os.makedirs(score_dir, exist_ok=True)
        

        ### 1. save accuracy
        score_filename = f"acc_step_{self.global_steps}_{uuid.uuid4().hex}.json"
        score_path = os.path.join(score_dir, score_filename)

        lock_path = os.path.join(tempfile.gettempdir(), "accuracy_records.lock")
        with FileLock(lock_path):  
            tmp_file = score_path + ".tmp"
            with open(tmp_file, 'w') as f:
                json.dump(accuracy, f, indent=4)
            os.replace(tmp_file, score_path)

        ### 2. save accuracy with step
        score_with_step_filename = f"acc_step_with_step_{self.global_steps}_{uuid.uuid4().hex}.json"
        score_with_step_path = os.path.join(score_dir, score_with_step_filename)

        lock_score_with_step_path = os.path.join(tempfile.gettempdir(), "accuracy_with_step_records.lock")
        with FileLock(lock_score_with_step_path):  
            tmp_file = score_with_step_path + ".tmp"
            with open(tmp_file, 'w') as f:
                json.dump(self.history_accuracy_with_step, f, indent=4)
            os.replace(tmp_file, score_with_step_path)

        ### 3. save validation accuracy by source
        score_source_filename = f"val_acc_source_with_step_{self.global_steps}_{uuid.uuid4().hex}.json"
        score_source_path = os.path.join(score_dir, score_source_filename)

        lock_score_source_path = os.path.join(tempfile.gettempdir(), "accuracy_source_with_step_records.lock")
        with FileLock(lock_score_source_path):  
            tmp_file = score_source_path + ".tmp"
            with open(tmp_file, 'w') as f:
                json.dump(self.validation_accuracy_by_source, f, indent=4)
            os.replace(tmp_file, score_source_path)

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        # Check if we should only save inference weights (skip optimizer states, dataloader, etc.)
        inference_only = getattr(self.config.trainer, 'save_inference_only', False)

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save,
                                              inference_only=inference_only)

        # Skip critic and training state if inference_only
        if not inference_only:
            if self.use_critic:
                critic_local_path = os.path.join(local_global_step_folder, 'critic')
                critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                    self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
                self.critic_wg.save_checkpoint(critic_local_path,
                                               critic_remote_path,
                                               self.global_steps,
                                               remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

            # save dataloader
            dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
            import dill
            torch.save(self.train_dataloader, dataloader_local_path, pickle_module=dill)

            # Save additional training state information
            training_state = {
                'history_accuracy': self.history_accuracy,
                'history_accuracy_with_step': self.history_accuracy_with_step,
                'validation_accuracy_by_source': self.validation_accuracy_by_source
            }

            training_state_path = os.path.join(local_global_step_folder, 'training_state.json')
            with open(training_state_path, 'w') as f:
                json.dump(training_state, f)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            # On fresh start, set the initial dataloader based on configuration
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
            
        training_state_path = os.path.join(global_step_folder, 'training_state.json')
        if os.path.exists(training_state_path):
            with open(training_state_path, 'r') as f:
                training_state = json.load(f)
            # json.dump serializes a None dict key as the string "null"; skip any
            # non-integer index keys so resuming never crashes on such entries
            self.history_accuracy = {}
            for k, v in training_state['history_accuracy'].items():
                try:
                    self.history_accuracy[int(k)] = v
                except (TypeError, ValueError):
                    continue
            self.history_accuracy_with_step = {}
            for k, v in training_state.get('history_accuracy_with_step', {}).items():
                try:
                    self.history_accuracy_with_step[int(k)] = {int(step): acc for step, acc in v.items()}
                except (TypeError, ValueError):
                    continue
            
            if 'validation_accuracy_by_source' in training_state:
                self.validation_accuracy_by_source = {
                    k: {int(step): acc for step, acc in v.items()} 
                    for k, v in training_state['validation_accuracy_by_source'].items()
                }
            else:
                self.validation_accuracy_by_source = {}

        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        self.train_dataloader = torch.load(dataloader_local_path)
        if isinstance(self.train_dataloader.dataset, RLHFDataset):
            self.train_dataloader.dataset.resume_dataset_state()
        
    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # do not save checkpoint at step 0

        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                score_records = []
                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == 'remax':
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor, score_record = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # Initialize variables for tracking replacements
                    replaced_indices = []
                    all_wrong_uids = set()

                    # Concept augmentation: replace responses before log_prob/reward if a group is all-wrong
                    try:
                        import json
                        from collections import defaultdict
                        from verl.utils.reward_score import deepscaler as _concept_score
                        from tensordict import TensorDict

                        # lazy-load concept map
                        if not hasattr(self, '_concept_map_loaded'):
                            self._concept_map = {}
                            concept_file = "data/quizzes/concept_quizzes.jsonl"
                            try:
                                with open(concept_file, 'r', encoding='utf-8') as f_concept:
                                    for line in f_concept:
                                        line = line.strip()
                                        if not line:
                                            continue
                                        item = json.loads(line)
                                        q = item.get('question') or item.get('original_question')
                                        c = item.get('related_concept')
                                        if q and c:
                                            key = ' '.join(str(q).strip().split())
                                            self._concept_map[key] = c
                                print(f"[concept_aug] loaded {len(self._concept_map)} concept mappings")
                            except Exception as e:
                                print(f"[concept_aug] WARN: cannot load concept file: {e}")
                            self._concept_map_loaded = True

                        def _extract_q_and_opts(prompt_str: str):
                            q_text = None
                            opts = []
                            try:
                                if "Question:" in prompt_str and "A." in prompt_str:
                                    start = prompt_str.find("Question:") + len("Question:")
                                    tail = prompt_str[start:]
                                    for label in ["A.", "B.", "C.", "D."]:
                                        idx_l = tail.find(label)
                                        if idx_l != -1:
                                            end_pos = len(tail)
                                            for nxt in ["A.", "B.", "C.", "D."]:
                                                if nxt == label:
                                                    continue
                                                j = tail.find(nxt, idx_l + 2)
                                                if j != -1:
                                                    end_pos = min(end_pos, j)
                                            piece = tail[idx_l:end_pos].strip()
                                            opts.append(piece)
                                    a_pos = tail.find("A.")
                                    if a_pos != -1:
                                        q_text = tail[:a_pos].strip()
                                    else:
                                        q_text = tail.strip()
                            except Exception:
                                pass
                            return q_text, opts

                        def _build_aug_prompt(concept_text: str, prompt_str: str) -> str:
                            if "Related Concept:" in prompt_str:
                                return prompt_str
                            if "Question:" in prompt_str:
                                pos = prompt_str.find("Question:")
                                return prompt_str[:pos] + f"Related Concept: {concept_text}\n\n" + prompt_str[pos:]
                            return f"Related Concept: {concept_text}\n\n" + prompt_str

                        def _compute_score_text(solution_text: str, ground_truth):
                            try:
                                score = float(_concept_score.compute_score(
                                    data_source='math_mcq', solution_str=solution_text, ground_truth=ground_truth, extra_info=None, use_think=False
                                ))
                                # DEBUG: record calculation process
                                print(f"[DEBUG] _compute_score_text: gt={ground_truth}, gt_type={type(ground_truth)}, score={score}")
                                if hasattr(ground_truth, 'shape'):
                                    print(f"[DEBUG] gt.shape={ground_truth.shape}, gt.dtype={ground_truth.dtype}")
                                return score
                            except Exception as e:
                                print(f"[DEBUG] _compute_score_text EXCEPTION: {e}, gt={ground_truth}, gt_type={type(ground_truth)}")
                                import traceback
                                traceback.print_exc()
                                return 0.0

                        prompts_ids = batch.batch['prompts']
                        responses_ids = batch.batch['responses']
                        attention_mask = batch.batch['attention_mask']
                        position_ids = batch.batch['position_ids']
                        batch_size = prompts_ids.shape[0]
                        prompt_len = prompts_ids.shape[1]
                        resp_len = responses_ids.shape[1]

                        uids = batch.non_tensor_batch['uid']
                        group_map = defaultdict(list)
                        for i in range(batch_size):
                            group_map[uids[i]].append(i)

                        for uid, idx_list in group_map.items():
                            # FIX: Convert all indices to int to avoid tensor indexing errors
                            idx_list = [int(idx) if hasattr(idx, 'item') else int(idx) for idx in idx_list]

                            if len(idx_list) < 2:
                                continue
                            group_scores = []
                            prompt_text_cache = None
                            first_idx = idx_list[0]  # Already int now

                            # FIX: reward_model is a numpy array of dicts, need to index it first
                            reward_model_data = batch.non_tensor_batch['reward_model']
                            # Get the first item's reward_model dict and extract ground_truth
                            ground_truth = reward_model_data[first_idx]['ground_truth']
                            for i_idx in idx_list:
                                v_prompt_len = int(attention_mask[i_idx, :prompt_len].sum().item())
                                v_resp_len = int(attention_mask[i_idx, prompt_len:].sum().item())
                                v_prompt_ids = prompts_ids[i_idx, -v_prompt_len:]
                                v_resp_ids = responses_ids[i_idx, :v_resp_len]
                                seq_ids = torch.cat((v_prompt_ids, v_resp_ids), dim=0)
                                seq_text = self.tokenizer.decode(seq_ids, skip_special_tokens=False)
                                print(f"[DEBUG] Computing score for idx={i_idx}")
                                print(f"[DEBUG] seq_text[:200]: {seq_text[:200]}...")
                                print(f"[DEBUG] seq_text[-200:]: ...{seq_text[-200:]}")
                                s = _compute_score_text(seq_text, ground_truth)
                                group_scores.append(s)
                                if prompt_text_cache is None:
                                    prompt_text_cache = self.tokenizer.decode(v_prompt_ids, skip_special_tokens=False)

                            # DEBUG: detailed check for all-wrong group logic
                            all_scores_le_zero = all(s <= 0.0 for s in group_scores)
                            print(f"[DEBUG] UID={uid[:8]}, group_scores={group_scores}, all_le_zero={all_scores_le_zero}")
                            
                            # first unconditionally add all-wrong groups to all_wrong_uids (for statistics and debugging)
                            if all_scores_le_zero:
                                all_wrong_uids.add(uid)
                                print(f"[DEBUG] ADDED to all_wrong_uids: {uid[:8]} with scores {group_scores}")
                            
                            # then check if concept enhancement can be performed (requires additional conditions)
                            if not all_scores_le_zero:
                                continue

                            q_text, opts = _extract_q_and_opts(prompt_text_cache or "")
                            if not q_text:
                                print(f"[DEBUG] Skipping concept enhancement for {uid[:8]} - no question text extracted")
                                continue
                            concept_text = self._concept_map.get(' '.join(q_text.split()))
                            if not concept_text:
                                print(f"[DEBUG] Skipping concept enhancement for {uid[:8]} - no concept mapping found")
                                continue

                            aug_prompt_text = _build_aug_prompt(concept_text, prompt_text_cache)
                            aug_ids = self.tokenizer.encode(aug_prompt_text, add_special_tokens=False)
                            if len(aug_ids) > prompt_len:
                                aug_ids = aug_ids[-prompt_len:]
                            pad_len = prompt_len - len(aug_ids)
                            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                            aug_prompt_ids = torch.tensor([([pad_id] * pad_len) + aug_ids], device=prompts_ids.device, dtype=prompts_ids.dtype)
                            aug_attn = torch.tensor([[0] * pad_len + [1] * len(aug_ids)], device=attention_mask.device, dtype=attention_mask.dtype)
                            sample_pos = position_ids[first_idx, :prompt_len].unsqueeze(0).clone()

                            # copy prompt to generate 2 responses
                            aug_prompt_ids = aug_prompt_ids.repeat(2, 1)  # [2, prompt_len]
                            aug_attn = aug_attn.repeat(2, 1)  # [2, prompt_len]
                            sample_pos = sample_pos.repeat(2, 1)  # [2, prompt_len]
                            
                            aug_dp = DataProto(batch=TensorDict({
                                'input_ids': aug_prompt_ids,
                                'attention_mask': aug_attn,
                                'position_ids': sample_pos
                            }, batch_size=2))  # set batch_size=2
                            aug_dp.meta_info = {
                                'eos_token_id': self.tokenizer.eos_token_id,
                                'pad_token_id': self.tokenizer.pad_token_id,
                                'recompute_log_prob': False,
                                'do_sample': True,
                            }
                            aug_out = self.actor_rollout_wg.generate_sequences(aug_dp)
                            aug_responses = aug_out.batch['responses']
                            k_take = min(2, aug_responses.shape[0])
                            if k_take == 0:
                                continue
                            import random as _r
                            replace_targets = _r.sample(idx_list, k=min(k_take, len(idx_list)))
                            import os as _os
                            debug_rollout = _os.environ.get("VERL_DEBUG_ROLLOUT", "") == "1"
                            for j, i_idx in enumerate(replace_targets):
                                # idx_list elements are already converted to int, just validate range
                                if i_idx < 0 or i_idx >= batch_size:
                                    print(f"[concept_aug] WARN: invalid index {i_idx}, skipping")
                                    continue
                                
                                new_resp = aug_responses[j]
                                if new_resp.shape[0] < resp_len:
                                    pad = torch.full((resp_len - new_resp.shape[0],), pad_id, device=new_resp.device, dtype=new_resp.dtype)
                                    new_resp = torch.cat([new_resp, pad], dim=0)
                                elif new_resp.shape[0] > resp_len:
                                    new_resp = new_resp[:resp_len]
                                # optional debug: preview before/after
                                if debug_rollout:
                                    try:
                                        old_resp_ids = batch.batch['responses'][i_idx].detach().clone()
                                        old_text = self.tokenizer.decode(old_resp_ids, skip_special_tokens=False)
                                        new_text = self.tokenizer.decode(new_resp, skip_special_tokens=False)
                                        print(f"[rollout_debug] uid={uid} replace idx={i_idx} | group_scores={group_scores}")
                                        print(f"[rollout_debug] aug_prompt: {aug_prompt_text[:200]}...")
                                        print(f"[rollout_debug] old_resp: {old_text[:200]}...")
                                        print(f"[rollout_debug] new_resp: {new_text[:200]}...")
                                    except Exception as _e_dbg:
                                        print(f"[rollout_debug] WARN: {_e_dbg}")
                                # replace responses
                                batch.batch['responses'][i_idx] = new_resp
                                # update attention mask (response part) based on new EOS
                                from verl.utils.torch_functional import get_eos_mask
                                resp_mask = get_eos_mask(response_id=new_resp.unsqueeze(0), eos_token=self.tokenizer.eos_token_id, dtype=attention_mask.dtype)[0]
                                attention_mask[i_idx, -resp_len:] = resp_mask
                                # recompute position_ids for this sample to match new mask
                                pos_ids = torch.cumsum(attention_mask[i_idx], dim=-1) - 1
                                pos_ids = pos_ids.masked_fill(attention_mask[i_idx] == 0, 0)
                                batch.batch['position_ids'][i_idx] = pos_ids
                                # update input_ids as prompt + new_resp
                                batch.batch['input_ids'][i_idx] = torch.cat([prompts_ids[i_idx], new_resp], dim=-1)
                                replaced_indices.append(i_idx)
                    except Exception as e:
                        import traceback
                        print(f"[concept_aug] WARN: augmentation failed with error: {e}")
                        print(f"[concept_aug] Full traceback:")
                        traceback.print_exc()

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                            
                        # we combine with rule-based rm
                        reward_tensor, score_record = self.reward_fn(batch)

                        # add +0.4 bonus for replaced indices at last valid response token
                        # NOTE: matches the original (paper) implementation. The concept-replaced
                        # trajectories were generated with the concept-augmented prompt; recomputing
                        # their old_log_prob against the un-augmented batch poisons the GRPO importance
                        # ratio and collapses training, so we deliberately do NOT recompute log_probs.
                        try:
                            if 'replaced_indices' not in locals():
                                replaced_indices = []
                            if replaced_indices:
                                responses = batch.batch['responses']
                                response_length = responses.size(-1)
                                attn = batch.batch['attention_mask']
                                for i_idx in replaced_indices:
                                    prompt_mask_len = int(attn[i_idx, :-response_length].sum().item())
                                    valid_resp_len = int(attn[i_idx, -response_length:].sum().item())
                                    if valid_resp_len > 0:
                                        last_pos = valid_resp_len - 1
                                        reward_tensor[i_idx, last_pos] = reward_tensor[i_idx, last_pos] + 0.4
                        except Exception as e:
                            print(f"[concept_aug] WARN: add bonus failed: {e}")

                        score_records.extend(score_record)
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # Rollout data is now logged to wandb directly during training, no need to process here again
                    print(f"📊 Step {self.global_steps}: Rollout data is now logged to wandb directly during training")

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    with _timer('log_scores', timing_raw):
                        score_metrics = self._log_scores_to_wandb(score_records, epoch)
                        # Update metrics with returned values instead of direct logging
                        # logger.log(data=score_metrics, step=self.global_steps)
                        metrics.update(score_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        self._save_history_accuracy(self.history_accuracy)

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    return
