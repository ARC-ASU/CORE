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

import torch
import torch.nn.functional as F
from verl.utils.reward_score import _default_compute_score
from verl.utils.reward_score.math import compute_score as math_compute_score, last_boxed_only_string, remove_boxed, is_equiv
from typing import List, Dict, Any
import wandb

class ConceptKLRegularizer:
    """
    Concept knowledge injection KL divergence regularizer
    
    key features:
    1. calculate the KL divergence loss for difficult questions
    2. adaptively adjust the weights based on the correctness of the concept answer
    3. use the original answer extraction and evaluation logic
    """
    
    def __init__(
        self, 
        tokenizer,
        base_lambda: float = 0.01,
        effective_multiplier: float = 3.0,
        ineffective_multiplier: float = 0.5
    ):
        self.tokenizer = tokenizer
        self.base_lambda = base_lambda
        self.effective_multiplier = effective_multiplier
        self.ineffective_multiplier = ineffective_multiplier
        self.kl_call_count = 0  # controls logging frequency
        
        # prompt template for generating concept enhanced answers
        self.concept_prompt_template = """Please solve the following question carefully using the provided concept. Explain your reasoning step by step, and conclude with the final answer using the format: \\boxed{{X}}, where X is A, B, C, or D.

{concept_enhanced_question}

Answer:"""
    
    def compute_token_level_kl_loss(
        self, 
        model,
        kl_data_items: List[Dict[str, Any]],
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate token-level KL divergence loss: D_KL[P(Y | q, c) || P(Y | q)]
        
        Args:
            model: the current actor model
            kl_data_items: items with both original and concept-enhanced response data
            device: computing device
            
        Returns:
            dictionary containing total KL loss and statistics
        """
        if not kl_data_items:
            return {
                'kl_loss': torch.tensor(0.0, device=device),
                'kl_items_count': 0,
                'concept_effective_count': 0,
                'total_token_count': 0
            }
        
        print(f"🔧 KL: Computing token-level KL loss for {len(kl_data_items)} items")
        
        total_kl_loss = torch.tensor(0.0, device=device)
        concept_effective_count = 0
        total_token_count = 0
        
        model.eval()  # set to evaluation mode
        self.kl_call_count += 1
        
        with torch.no_grad():
            for idx, item in enumerate(kl_data_items):
                kl_info = item.get('kl_info', {})
                
                if not kl_info.get('has_concept', False):
                    continue
                
                try:
                    # 🔥 Core KL computation: D_KL[P(Y | q, c) || P(Y | q)]
                    item_kl_loss, token_count = self._compute_token_level_kl(
                        model=model,
                        original_prompt=item['prompt_str'],
                        original_response=kl_info['original_response'],
                        concept_response=kl_info['concept_response'],
                        device=device
                    )
                    
                    # Check if concept is effective (concept response is better)
                    concept_score = self._evaluate_response_quality(
                        kl_info['concept_response'], item.get('ground_truth')
                    )
                    original_score = item.get('score', 0.0)
                    
                    is_concept_effective = concept_score > original_score
                    if is_concept_effective:
                        concept_effective_count += 1
                    
                    # Apply adaptive weighting
                    if is_concept_effective:
                        adaptive_weight = self.base_lambda * self.effective_multiplier
                    else:
                        adaptive_weight = self.base_lambda * self.ineffective_multiplier
                    
                    # Accumulate weighted KL loss
                    weighted_kl_loss = adaptive_weight * item_kl_loss
                    total_kl_loss += weighted_kl_loss
                    total_token_count += token_count
                    
                    if idx < 3:  # Debug first few items
                        print(f"🔍 KL[{idx}]: token_count={token_count}, kl_loss={item_kl_loss:.6f}, "
                              f"weight={adaptive_weight:.3f}, effective={is_concept_effective}")
                    
                except Exception as e:
                    print(f"❌ KL: Error computing KL for item {idx}: {e}")
                    continue
        
        model.train()  # restore training mode
        
        print(f"🔧 KL: Total KL loss={total_kl_loss:.6f}, effective_count={concept_effective_count}/{len(kl_data_items)}")
        
        return {
            'kl_loss': total_kl_loss,
            'kl_items_count': len(kl_data_items),
            'concept_effective_count': concept_effective_count,
            'total_token_count': total_token_count,
            'average_kl_per_token': total_kl_loss / max(total_token_count, 1)
        }
    
    def _compute_token_level_kl(
        self,
        model,
        original_prompt: str,
        original_response: str, 
        concept_response: str,
        device: torch.device
    ) -> tuple[torch.Tensor, int]:
        """
        Compute token-level KL divergence: D_KL[P(Y | q, c) || P(Y | q)]
        
        Returns:
            (kl_loss, token_count): KL loss and number of tokens compared
        """
        try:
            # Tokenize both responses to the same length for fair comparison
            concept_tokens = self.tokenizer.encode(concept_response, add_special_tokens=False)
            original_tokens = self.tokenizer.encode(original_response, add_special_tokens=False)
            
            # Use the shorter length to avoid padding issues
            max_len = min(len(concept_tokens), len(original_tokens))
            if max_len == 0:
                return torch.tensor(0.0, device=device), 0
            
            concept_tokens = concept_tokens[:max_len]
            original_tokens = original_tokens[:max_len]
            
            # Create input for model: prompt + partial response for each position
            prompt_tokens = self.tokenizer.encode(original_prompt, add_special_tokens=False)
            
            total_kl = torch.tensor(0.0, device=device)
            valid_tokens = 0
            
            # For each token position, compute KL between P(token | q, c, y_<i) and P(token | q, y_<i)
            for i in range(max_len):
                # Context for position i: prompt + response tokens up to position i
                if i == 0:
                    context_tokens = prompt_tokens
                else:
                    context_tokens = prompt_tokens + original_tokens[:i]
                
                # Get model predictions at position i
                context_input = torch.tensor([context_tokens], device=device)
                
                with torch.no_grad():
                    outputs = model(context_input)
                    logits = outputs.logits[0, -1, :]  # last position logits
                
                # P(Y | q): probability distribution from original context
                p_original = torch.softmax(logits, dim=-1)
                
                # P(Y | q, c): we approximate this by looking at what concept response actually chose
                # This is a simplification - ideally we'd run model with concept-enhanced context
                concept_token_id = concept_tokens[i]
                
                # Create target distribution (one-hot for concept token)
                p_concept = torch.zeros_like(p_original)
                p_concept[concept_token_id] = 1.0
                
                # Compute KL divergence: D_KL[P(concept) || P(original)]
                kl_div = torch.sum(p_concept * torch.log(p_concept / (p_original + 1e-8) + 1e-8))
                
                if not torch.isnan(kl_div) and torch.isfinite(kl_div):
                    total_kl += kl_div
                    valid_tokens += 1
            
            # Average KL per token
            if valid_tokens > 0:
                avg_kl = total_kl / valid_tokens
            else:
                avg_kl = torch.tensor(0.0, device=device)
            
            return avg_kl, valid_tokens
            
        except Exception as e:
            print(f"❌ KL: Error in token-level KL computation: {e}")
            return torch.tensor(0.0, device=device), 0
    
    def _evaluate_response_quality(self, response: str, ground_truth) -> float:
        """Evaluate response quality using the same logic as training"""
        try:
            # Use the math scoring function
            if ground_truth is None:
                return 0.0
            
            correct_answer = ground_truth[0] if isinstance(ground_truth, list) else ground_truth
            score = math_compute_score(response, correct_answer)
            return float(score)
            
        except Exception as e:
            print(f"❌ KL: Error evaluating response quality: {e}")
            return 0.0
    
    def _generate_concept_answer(
        self, 
        model, 
        item: Dict[str, Any], 
        device: torch.device
    ) -> tuple[str, bool]:
        """generate the concept enhanced answer in real time"""
        try:
            concept_info = item.get('concept_info', {})
            enhanced_question = concept_info.get('enhanced_question', '')
            
            if not enhanced_question:
                return "", False
            
            # build the generation prompt
            generation_prompt = self.concept_prompt_template.format(
                concept_enhanced_question=enhanced_question
            )
            
            # tokenize
            inputs = self.tokenizer.encode(generation_prompt, return_tensors='pt').to(device)
            
            # generate the answer
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # decode the generated answer
            generated_text = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            # check the correctness of the answer using original verl logic
            correct_answer = item.get('ground_truth', [''])[0]
            is_correct = math_compute_score(generated_text, correct_answer) > 0
            
            return generated_text.strip(), is_correct
            
        except Exception as e:
            print(f"Error generating concept answer: {e}")
            return "", False
    
    def _compute_item_kl_loss(
        self, 
        model, 
        item: Dict[str, Any], 
        golden_answer: str, 
        device: torch.device
    ) -> torch.Tensor:
        """calculate the KL divergence loss for a single item"""
        try:
            # get the original response
            response_str = item.get('response_str', '')
            if not response_str or not golden_answer:
                return torch.tensor(0.0, device=device)
            
            # tokenize the golden answer
            golden_tokens = self.tokenizer.encode(golden_answer, add_special_tokens=False)
            
            # build the input with the same length as the original response
            prompt_str = item.get('prompt_str', '')
            full_input = prompt_str + response_str
            
            # tokenize the full input
            input_tokens = self.tokenizer.encode(full_input, return_tensors='pt').to(device)
            prompt_length = len(self.tokenizer.encode(prompt_str, add_special_tokens=False))
            
            # get the logits of the model in the current state
            with torch.no_grad():
                outputs = model(input_tokens)
                logits = outputs.logits[0]  # [seq_len, vocab_size]
            
            # calculate the KL divergence (only in the response part)
            kl_loss = torch.tensor(0.0, device=device)
            effective_tokens = 0
            
            # iterate over each position of the response
            response_start = prompt_length
            max_compare_length = min(len(golden_tokens), logits.shape[0] - response_start)
            
            for i in range(max_compare_length):
                if response_start + i >= logits.shape[0]:
                    break
                
                # the model prediction distribution at the current position
                model_probs = F.softmax(logits[response_start + i], dim=-1)
                
                # the one-hot distribution of the golden answer at the current position
                golden_token = golden_tokens[i]
                target_dist = torch.zeros_like(model_probs)
                target_dist[golden_token] = 1.0
                
                # calculate the KL divergence: KL(target || model)
                kl_div = F.kl_div(
                    torch.log(model_probs + 1e-8),
                    target_dist,
                    reduction='sum'
                )
                
                kl_loss += kl_div
                effective_tokens += 1
            
            # average the KL loss
            if effective_tokens > 0:
                kl_loss = kl_loss / effective_tokens
            
            return kl_loss
            
        except Exception as e:
            print(f"Error computing KL loss: {e}")
            return torch.tensor(0.0, device=device)
    
