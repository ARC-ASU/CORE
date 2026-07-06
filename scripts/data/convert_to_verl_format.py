#!/usr/bin/env python3
"""
Convert math MCQ data to verl-format parquet files.
Supports both original and concept-enhanced modes.
"""

import json
import pandas as pd
import argparse
from pathlib import Path
from transformers import AutoTokenizer

def convert_to_verl_format(input_file: str, output_file: str, use_concept: bool = True):
    """Convert to verl format.

    Args:
        input_file: input file path
        output_file: output file path
        use_concept: True to use concept-enhanced version, False to use original
    """
    data = []
    
    # Load tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-Math-7B')
    
    print(f"Mode: {'Concept-enhanced' if use_concept else 'Original'}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                
                # Check if prompt field already exists (concept-enhanced data)
                if 'prompt' in item:
                    # If prompt already exists, convert it to chat format
                    # Assuming prompt is a string, wrap it in chat format
                    prompt_content = item['prompt']
                    # Build chat format
                    prompt = [
                        {"role": "system", "content": "You are a helpful assistant that solves multiple-choice math questions with step-by-step reasoning."},
                        {"role": "user", "content": prompt_content}
                    ]
                else:
                    # Select question content field based on mode
                    if use_concept and 'enhanced_question' in item:
                        question = item['enhanced_question']
                    else:
                        question = item.get('original_question', '')
                    
                    options = item.get('options', item.get('original_options', []))
                    options_text = "\n".join(options) if options else ""
                    
                    # Build chat format
                    prompt = [
                        {"role": "system", "content": "You are a helpful assistant that solves multiple-choice math questions with step-by-step reasoning."},
                        {"role": "user", "content": f"Please solve the following question carefully. Explain your reasoning, and conclude with the final answer using the format: \\boxed{{X}}, where X is A, B, C, or D.\n\nExample:\nQuestion: What is 2 + 3?\nA. 4\nB. 5\nC. 6\nD. 7\n\nAnswer: 2 + 3 = 5, which is option B.\nThe final answer is \\boxed{{B}}.\n\n---\nQuestion: {question}\n{options_text}"}
                    ]
                
                # verl-format data item using chat-format prompt
                verl_item = {
                    "data_source": "math_mcq",
                    "prompt": prompt,  # chat array format
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": [item.get('correct_answer') or (item.get('answer', '').split('.')[0] if item.get('answer') else '')]  # correct answer, e.g. ["A"]
                    }
                }
                
                data.append(verl_item)
    
    # Convert to DataFrame and save as parquet
    df = pd.DataFrame(data)
    df.to_parquet(output_file, index=False)
    
    print(f"Conversion complete: {len(data)} samples")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    # Display sample data
    if len(data) > 0:
        print("\nSample data:")
        sample = data[0]
        print(f"Data source: {sample['data_source']}")
        print(f"Ability: {sample['ability']}")
        print(f"Prompt user content: {sample['prompt'][1]['content'][:200]}...")
        print(f"Ground truth: {sample['reward_model']['ground_truth']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert math MCQ data to verl format")
    parser.add_argument("--use_concept", action="store_true", default=False,
                       help="Use concept-enhanced version (default: original)")
    
    args = parser.parse_args()
    use_concept = args.use_concept
    
    print(f"Training mode: {'Concept-enhanced' if use_concept else 'Original'}")

    # Prefer enhanced_questions_with_concept.jsonl as it has properly separated fields
    if Path("enhanced_questions_with_concept.jsonl").exists():
        print("Using enhanced_questions_with_concept.jsonl as data source")
        convert_to_verl_format("enhanced_questions_with_concept.jsonl", "train_math_mcq.parquet", use_concept)
        
        # Create validation data (last 10% of the same file)
        print("Creating validation set from the same data source")
        # Create a temporary validation dataset
        import random
        all_data = []
        with open("enhanced_questions_with_concept.jsonl", 'r') as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line))
        
        # Shuffle and split
        random.seed(42)  # fixed seed for reproducibility
        random.shuffle(all_data)
        split_point = int(len(all_data) * 0.9)
        val_data = all_data[split_point:]
        
        # Save validation data to temporary file
        with open("temp_validation.jsonl", 'w') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        convert_to_verl_format("temp_validation.jsonl", "val_math_mcq.parquet", use_concept)
        
        # Clean up temporary file
        Path("temp_validation.jsonl").unlink()
        
    else:
        # Fallback: keep original train/validation split logic
        if Path("train_grpo_concept.jsonl").exists():
            print("Using concept-enhanced training data")
            convert_to_verl_format("train_grpo_concept.jsonl", "train_math_mcq.parquet", use_concept)
        elif Path("train_grpo.jsonl").exists():
            print("Using original training data")
            convert_to_verl_format("train_grpo.jsonl", "train_math_mcq.parquet", use_concept)
        else:
            print("Warning: training data file not found")
        
        # Convert validation data
        if Path("validation_grpo_concept.jsonl").exists():
            print("Using concept-enhanced validation data (variant data)")
            convert_to_verl_format("validation_grpo_concept.jsonl", "val_math_mcq.parquet", use_concept)
        elif Path("validation_grpo.jsonl").exists():
            print("Using original validation data")
            convert_to_verl_format("validation_grpo.jsonl", "val_math_mcq.parquet", use_concept)
        else:
            print("Warning: validation data file not found, skipping validation data conversion")